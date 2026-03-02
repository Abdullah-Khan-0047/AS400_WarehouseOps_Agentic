import unittest
import os
from datetime import datetime, timedelta

from config.settings import DC_CONFIG, AGENT_CONFIG
from src.agents.base_agent import GeminiClient
from src.simulation.history_generator import HistoryGenerator
from src.agents.dock_assignment_agent import DockAssignmentAgent
from src.agents.shrinkage_detection_agent import ShrinkageDetectionAgent
from src.agents.audit_agent import AuditAgent
from src.agents.orchestrator import Orchestrator
from src.analytics.pattern_detector import PatternDetector
from src.legacy.as400_simulator import AS400Simulator


def setUpModule():
    GeminiClient.configure()  # will gracefully fail if no key, agents run stats-only


class TestHistoryGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = HistoryGenerator()
        cls.data = cls.generator.generate_all()

    def test_stores_generated(self):
        self.assertEqual(len(self.data["stores"]), DC_CONFIG.num_stores)

    def test_dock_doors_generated(self):
        self.assertEqual(len(self.data["dock_doors"]), DC_CONFIG.total_dock_doors)

    def test_shipments_generated(self):
        self.assertGreater(len(self.data["shipments"]), 0)

    def test_shrinkage_events_generated(self):
        self.assertGreater(len(self.data["shrinkage_events"]), 0)

    def test_preferred_doors_assigned(self):
        for store in self.data["stores"]:
            self.assertGreater(len(store.preferred_dock_doors), 0)

    def test_dock_assignments_recorded(self):
        self.assertGreater(len(self.data["dock_assignments"]), 0)

    def test_preferred_vs_non_preferred_rates(self):
        assignments = self.data["dock_assignments"]
        non_preferred = [a for a in assignments if not a.was_preferred_door]
        rate = len(non_preferred) / len(assignments)
        self.assertGreater(rate, 0.10)
        self.assertLess(rate, 0.35)


class TestDockAssignmentAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generator = HistoryGenerator()
        cls.data = generator.generate_all()
        cls.agent = DockAssignmentAgent()
        cls.agent.initialize(cls.data)

    def test_agent_initialized(self):
        self.assertTrue(self.agent.is_initialized)

    def test_get_best_door(self):
        store = self.data["stores"][0]
        door, confidence = self.agent.get_best_door(store.store_id)
        self.assertGreater(door, 0)
        self.assertGreaterEqual(confidence, 0.0)

    def test_get_best_door_with_occupied(self):
        store = self.data["stores"][0]
        all_doors = list(DC_CONFIG.outbound_doors)
        occupied = all_doors[:-1]
        door, confidence = self.agent.get_best_door(store.store_id, occupied)
        self.assertEqual(door, all_doors[-1])

    def test_analyze_returns_findings(self):
        findings = self.agent.analyze()
        self.assertIsInstance(findings, list)

    def test_recommend_returns_recommendations(self):
        recs = self.agent.recommend()
        self.assertIsInstance(recs, list)

    def test_ai_explain_returns_string_or_none(self):
        result = self.agent.ai_explain_assignment("STR-0001", 25, 0.5)
        self.assertTrue(result is None or isinstance(result, str))  # None if Gemini offline


class TestShrinkageDetectionAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generator = HistoryGenerator()
        cls.data = generator.generate_all()
        cls.agent = ShrinkageDetectionAgent()
        cls.agent.initialize(cls.data)

    def test_agent_initialized(self):
        self.assertTrue(self.agent.is_initialized)

    def test_baselines_computed(self):
        self.assertGreater(len(self.agent.store_baselines), 0)
        self.assertGreater(len(self.agent.door_baselines), 0)
        self.assertGreater(len(self.agent.shift_baselines), 0)
        self.assertGreater(len(self.agent.category_baselines), 0)

    def test_analyze_finds_preferred_impact(self):
        findings = self.agent.analyze()
        pref_findings = [
            f for f in findings if f.get("dimension") == "assignment_type"
        ]
        self.assertGreater(len(pref_findings), 0)
        for f in pref_findings:
            self.assertGreater(f["non_preferred_avg"], f["preferred_avg"])

    def test_evaluate_new_shipment(self):
        result = self.agent.evaluate_new_shipment(
            store_id="STR-0001",
            dock_door_id=25,
            shift="3rd",
            category="jewelry",
        )
        self.assertIn("risk_score", result)
        self.assertIn("risk_level", result)
        self.assertIn("factors", result)
        self.assertIn("ai_assessment", result)  # present even if None

    def test_recommend_returns_list(self):
        recs = self.agent.recommend()
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)


class TestAuditAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generator = HistoryGenerator()
        cls.data = generator.generate_all()
        cls.agent = AuditAgent()
        cls.agent.initialize(cls.data)

    def test_agent_initialized(self):
        self.assertTrue(self.agent.is_initialized)

    def test_risk_scores_computed(self):
        self.assertGreater(len(self.agent.risk_scores), 0)

    def test_should_audit_returns_tuple(self):
        result = self.agent.should_audit_shipment(25, "3rd")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_recommend_returns_audits(self):
        recs = self.agent.recommend()
        audit_recs = [r for r in recs if r.get("type") == "scheduled_audit"]
        self.assertGreater(len(audit_recs), 0)


class TestOrchestrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generator = HistoryGenerator()
        cls.data = generator.generate_all()
        cls.orchestrator = Orchestrator()
        cls.orchestrator.initialize_all(cls.data)

    def test_orchestrator_initialized(self):
        self.assertTrue(self.orchestrator.is_initialized)

    def test_run_analysis_cycle(self):
        results = self.orchestrator.run_analysis_cycle()
        self.assertIn("findings", results)
        self.assertIn("recommendations", results)
        self.assertIn("ai_insights", results)
        self.assertGreater(results["findings"]["total_count"], 0)

    def test_simulate_incoming_shipment(self):
        result = self.orchestrator.simulate_incoming_shipment(
            store_id="STR-0001",
            shift="2nd",
            category="apparel",
        )
        self.assertIn("assigned_door", result)
        self.assertIn("risk_assessment", result)
        self.assertIn("audit_recommended", result)
        self.assertIn("ai_explanation", result)

    def test_executive_summary(self):
        results = self.orchestrator.run_analysis_cycle()
        summary = self.orchestrator.generate_executive_summary(results)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 100)


class TestPatternDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generator = HistoryGenerator()
        cls.data = generator.generate_all()
        cls.detector = PatternDetector()

    def test_temporal_patterns(self):
        self.detector.load_data(self.data)
        patterns = self.detector.detect_temporal_patterns()
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

    def test_store_door_affinity(self):
        self.detector.load_data(self.data)
        affinities = self.detector.detect_store_door_affinity()
        self.assertIsInstance(affinities, list)

    def test_category_vulnerabilities(self):
        self.detector.load_data(self.data)
        vulns = self.detector.detect_category_vulnerabilities()
        self.assertIsInstance(vulns, list)
        self.assertGreater(len(vulns), 0)

    def test_potential_savings(self):
        self.detector.load_data(self.data)
        savings = self.detector.compute_potential_savings(self.data)
        self.assertIn("estimated_savings_dollars", savings)
        self.assertGreaterEqual(savings["estimated_savings_dollars"], 0)


class TestAS400Simulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generator = HistoryGenerator()
        cls.data = generator.generate_all()
        cls.simulator = AS400Simulator()
        cls.simulator.initialize(cls.data)

    def test_simulator_initialized(self):
        self.assertTrue(self.simulator.is_initialized)

    def test_assign_dock_door(self):
        door, method = self.simulator.assign_dock_door("STR-0001")
        self.assertGreater(door, 0)
        self.assertEqual(method, "preferred_table_lookup")

    def test_assign_with_all_preferred_occupied(self):
        store = self.data["stores"][0]
        preferred = store.preferred_dock_doors
        door, method = self.simulator.assign_dock_door(store.store_id, preferred)
        self.assertNotIn(door, preferred)
        self.assertEqual(method, "random_fallback")

    def test_weekly_report(self):
        report = self.simulator.generate_weekly_shrinkage_report(
            self.data["dock_assignments"][:50]
        )
        self.assertIn("SHRINKAGE", report)

    def test_comparison(self):
        dock_agent = DockAssignmentAgent()
        dock_agent.initialize(self.data)
        agentic_door, confidence = dock_agent.get_best_door("STR-0001")

        comparison = self.simulator.compare_with_agentic(
            "STR-0001", [], agentic_door, confidence
        )
        self.assertIn("as400", comparison)
        self.assertIn("agentic_ai", comparison)
        self.assertFalse(comparison["as400"]["considers_shrinkage"])
        self.assertTrue(comparison["agentic_ai"]["considers_shrinkage"])


class TestGeminiClient(unittest.TestCase):
    """Tests for Gemini integration layer."""

    def test_configure_without_key(self):
        # Reset state for isolated test
        GeminiClient._configured = False
        GeminiClient._available = False
        GeminiClient._model = None
        os.environ.pop("GEMINI_API_KEY", None)  # ensure no key

        result = GeminiClient.configure()
        self.assertFalse(result)  # should fail gracefully

    def test_query_when_unavailable(self):
        GeminiClient._available = False
        result = GeminiClient.query("test prompt")
        self.assertIsNone(result)

    def test_is_available_reflects_state(self):
        GeminiClient._available = False
        self.assertFalse(GeminiClient.is_available())
        GeminiClient._available = True
        self.assertTrue(GeminiClient.is_available())
        GeminiClient._available = False  # reset


if __name__ == "__main__":
    unittest.main()