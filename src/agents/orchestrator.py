from datetime import datetime
from typing import Dict, List, Optional
import textwrap

from config.settings import DC_CONFIG, AGENT_CONFIG
from src.agents.base_agent import BaseAgent, AgentMessage, GeminiClient, truncate_for_prompt
from src.agents.dock_assignment_agent import DockAssignmentAgent
from src.agents.shrinkage_detection_agent import ShrinkageDetectionAgent
from src.agents.audit_agent import AuditAgent


class Orchestrator:
    """Coordinates all agents, routes inter-agent messages, uses Gemini for cross-agent reasoning."""

    def __init__(self):
        self.dock_agent = DockAssignmentAgent()
        self.shrinkage_agent = ShrinkageDetectionAgent()
        self.audit_agent = AuditAgent()

        self.agents: Dict[str, BaseAgent] = {
            self.dock_agent.name: self.dock_agent,
            self.shrinkage_agent.name: self.shrinkage_agent,
            self.audit_agent.name: self.audit_agent,
        }

        self.message_bus: List[AgentMessage] = []
        self.global_findings: List[Dict] = []
        self.global_recommendations: List[Dict] = []
        self.ai_insights: List[Dict] = []  # orchestrator-level Gemini insights
        self.is_initialized = False

        self.system_prompt = (
            "You are the orchestrating AI for a retail distribution center. "
            "You coordinate dock assignment, shrinkage detection, and audit agents. "
            "Synthesize their findings into a unified strategic view. Be concise and actionable."
        )

    def query_gemini(self, prompt: str) -> Optional[str]:
        return GeminiClient.query(prompt, context=self.system_prompt)

    def initialize_all(self, historical_data: Dict) -> None:
        print("\n" + "=" * 60)
        print("  ORCHESTRATOR — Initializing All Agents")
        print("=" * 60)

        for name, agent in self.agents.items():
            print(f"\n  Initializing {name}...")
            agent.initialize(historical_data)
            print(f"  ✓ {name} ready.")

        self.is_initialized = True
        gemini_status = "CONNECTED" if GeminiClient.is_available() else "OFFLINE (stats-only mode)"
        print(f"\n  All agents initialized. Gemini: {gemini_status}")
        print("=" * 60)

    def run_analysis_cycle(self) -> Dict:
        if not self.is_initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize_all() first.")

        print("\n" + "=" * 60)
        print("  ORCHESTRATOR — Running Full Analysis Cycle")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        print("\n  Phase 1: Independent Analysis")
        print("  " + "-" * 40)

        dock_findings = self.dock_agent.analyze()
        print(f"  DockAssignmentAgent: {len(dock_findings)} findings")

        shrinkage_findings = self.shrinkage_agent.analyze()
        print(f"  ShrinkageDetectionAgent: {len(shrinkage_findings)} findings")

        audit_findings = self.audit_agent.analyze()
        print(f"  AuditAgent: {len(audit_findings)} findings")

        print("\n  Phase 2: Cross-Agent Communication")
        print("  " + "-" * 40)
        self._coordinate_agents(dock_findings, shrinkage_findings, audit_findings)

        print("\n  Phase 3: Generating Recommendations")
        print("  " + "-" * 40)

        dock_recs = self.dock_agent.recommend()
        print(f"  DockAssignmentAgent: {len(dock_recs)} recommendations")

        shrinkage_recs = self.shrinkage_agent.recommend()
        print(f"  ShrinkageDetectionAgent: {len(shrinkage_recs)} recommendations")

        audit_recs = self.audit_agent.recommend()
        print(f"  AuditAgent: {len(audit_recs)} recommendations")

        self.global_findings = dock_findings + shrinkage_findings + audit_findings
        self.global_recommendations = dock_recs + shrinkage_recs + audit_recs

        # Phase 4: Gemini cross-agent synthesis
        print("\n  Phase 4: AI Cross-Agent Synthesis")
        print("  " + "-" * 40)
        self._run_cross_agent_synthesis()

        results = {
            "timestamp": datetime.now().isoformat(),
            "findings": {
                "dock_assignment": dock_findings,
                "shrinkage_detection": shrinkage_findings,
                "audit": audit_findings,
                "total_count": len(self.global_findings),
            },
            "recommendations": {
                "dock_assignment": dock_recs,
                "shrinkage_detection": shrinkage_recs,
                "audit": audit_recs,
                "total_count": len(self.global_recommendations),
            },
            "messages_exchanged": len(self.message_bus),
            "ai_insights": self._collect_all_insights(),
        }

        print(f"\n  Cycle complete. {len(self.global_findings)} findings, "
              f"{len(self.global_recommendations)} recommendations.")
        print("=" * 60)

        return results

    def _run_cross_agent_synthesis(self):
        """Ask Gemini to synthesize insights across all three agents."""
        all_agent_insights = self._collect_all_insights()

        if not GeminiClient.is_available():
            print("  Gemini offline — skipping cross-agent synthesis.")
            return

        summary_data = {
            "dock_agent_findings_count": len([f for f in self.global_findings if f.get("type") in ("high_non_preferred_rate", "high_shrinkage_door")]),
            "shrinkage_anomalies_count": len([f for f in self.global_findings if f.get("dimension") in ("store", "dock_door", "category")]),
            "critical_audit_areas": len([f for f in self.global_findings if f.get("risk_level") == "critical"]),
            "agent_insights": [i["content"][:500] for i in all_agent_insights[:6]],  # cap for token limit
        }

        ai_response = self.query_gemini(
            f"Three agents analyzed a distribution center independently. "
            f"Synthesize their findings into a unified strategic assessment. "
            f"Identify connections between dock assignment problems, shrinkage patterns, "
            f"and audit priorities that individual agents may have missed:\n"
            f"{truncate_for_prompt(summary_data)}"
        )
        if ai_response:
            self.ai_insights.append({"phase": "cross_agent_synthesis", "content": ai_response})
            print(f"  Cross-agent synthesis complete.")
        else:
            print("  No synthesis generated.")

    def _collect_all_insights(self) -> List[Dict]:
        """Gather AI insights from all agents and orchestrator."""
        all_insights = []
        for name, agent in self.agents.items():
            for insight in agent.ai_insights:
                all_insights.append({"agent": name, **insight})
        for insight in self.ai_insights:
            all_insights.append({"agent": "Orchestrator", **insight})
        return all_insights

    def _coordinate_agents(
        self,
        dock_findings: List[Dict],
        shrinkage_findings: List[Dict],
        audit_findings: List[Dict],
    ):
        messages_sent = 0

        high_shrink_doors = [
            f for f in shrinkage_findings
            if f.get("dimension") == "dock_door" and f.get("severity") in ("high", "critical")
        ]
        for finding in high_shrink_doors:
            msg = self.shrinkage_agent.send_message(
                recipient=self.dock_agent.name,
                msg_type="alert",
                payload={
                    "alert_type": "high_shrinkage_door",
                    "door_id": finding["key"],
                    "avg_loss_rate": finding["value"],
                },
                priority=2,
            )
            self.dock_agent.receive_message(msg)
            self.message_bus.append(msg)
            messages_sent += 1

        pref_findings = [
            f for f in shrinkage_findings
            if f.get("dimension") == "assignment_type"
        ]
        for finding in pref_findings:
            msg = self.shrinkage_agent.send_message(
                recipient=self.audit_agent.name,
                msg_type="alert",
                payload={
                    "alert_type": "non_preferred_impact",
                    "increase_factor": finding.get("increase_factor", 0),
                },
                priority=2,
            )
            self.audit_agent.receive_message(msg)
            self.message_bus.append(msg)
            messages_sent += 1

        misassigned = [
            f for f in dock_findings
            if f.get("type") == "high_non_preferred_rate" and f.get("severity") == "high"
        ]
        for finding in misassigned:
            msg = self.dock_agent.send_message(
                recipient=self.audit_agent.name,
                msg_type="recommendation",
                payload={
                    "rec_type": "audit_misassigned_store",
                    "store_id": finding["store_id"],
                    "non_preferred_rate": finding["non_preferred_rate"],
                },
                priority=1,
            )
            self.audit_agent.receive_message(msg)
            self.message_bus.append(msg)
            messages_sent += 1

        print(f"  {messages_sent} inter-agent messages exchanged.")

    def simulate_incoming_shipment(
        self,
        store_id: str,
        shift: str,
        category: str,
        occupied_doors: Optional[List[int]] = None,
    ) -> Dict:
        if not self.is_initialized:
            raise RuntimeError("Orchestrator not initialized.")

        print(f"\n  ▶ Incoming shipment: Store={store_id}, Shift={shift}, Category={category}")

        best_door, confidence = self.dock_agent.get_best_door(store_id, occupied_doors or [])
        print(f"    [Dock Agent] Assigned door {best_door} (confidence: {confidence:.3f})")

        risk = self.shrinkage_agent.evaluate_new_shipment(store_id, best_door, shift, category)
        print(f"    [Shrinkage Agent] Risk: {risk['risk_level']} ({risk['risk_score']:.3f})")
        for factor in risk["factors"]:
            print(f"      ⚠ {factor}")

        should_audit, audit_risk = self.audit_agent.should_audit_shipment(best_door, shift)
        print(f"    [Audit Agent] Audit recommended: {'YES' if should_audit else 'No'} "
              f"(risk: {audit_risk:.3f})")

        # Gemini explains the dock assignment decision
        ai_explanation = self.dock_agent.ai_explain_assignment(store_id, best_door, confidence)
        if ai_explanation:
            print(f"    [Gemini] {ai_explanation}")

        # Gemini risk assessment from shrinkage agent
        if risk.get("ai_assessment"):
            print(f"    [Gemini Risk] {risk['ai_assessment']}")

        return {
            "store_id": store_id,
            "assigned_door": best_door,
            "door_confidence": confidence,
            "risk_assessment": risk,
            "audit_recommended": should_audit,
            "audit_risk_score": audit_risk,
            "ai_explanation": ai_explanation,
        }

    def generate_executive_summary(self, results: Dict) -> str:
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 68 + "╗")
        lines.append("║" + "  AGENTIC AI DC OPTIMIZER — EXECUTIVE SUMMARY".center(68) + "║")
        lines.append("║" + f"  Generated: {results['timestamp'][:19]}".center(68) + "║")
        gemini_tag = "Gemini: ACTIVE" if GeminiClient.is_available() else "Gemini: OFFLINE"
        lines.append("║" + f"  {gemini_tag}".center(68) + "║")
        lines.append("╠" + "═" * 68 + "╣")

        findings = results["findings"]
        lines.append("║" + "  FINDINGS".ljust(68) + "║")
        lines.append("║" + f"    Dock Assignment Issues:    {len(findings['dock_assignment'])}".ljust(68) + "║")
        lines.append("║" + f"    Shrinkage Anomalies:       {len(findings['shrinkage_detection'])}".ljust(68) + "║")
        lines.append("║" + f"    Audit Risk Assessments:    {len(findings['audit'])}".ljust(68) + "║")
        lines.append("║" + f"    Total Findings:            {findings['total_count']}".ljust(68) + "║")

        lines.append("╠" + "═" * 68 + "╣")

        recs = results["recommendations"]
        lines.append("║" + "  RECOMMENDATIONS".ljust(68) + "║")
        lines.append("║" + f"    Dock Reassignments:        {len(recs['dock_assignment'])}".ljust(68) + "║")
        lines.append("║" + f"    Shrinkage Actions:         {len(recs['shrinkage_detection'])}".ljust(68) + "║")
        lines.append("║" + f"    Audit Actions:             {len(recs['audit'])}".ljust(68) + "║")
        lines.append("║" + f"    Total Recommendations:     {recs['total_count']}".ljust(68) + "║")

        lines.append("╠" + "═" * 68 + "╣")

        lines.append("║" + "  CRITICAL ITEMS REQUIRING IMMEDIATE ACTION".ljust(68) + "║")
        lines.append("║" + "  " + "-" * 64 + "  ║")

        critical_count = 0
        all_recs = recs["dock_assignment"] + recs["shrinkage_detection"] + recs["audit"]
        for rec in all_recs:
            priority = rec.get("priority", rec.get("risk_level", "low"))
            if priority in ("critical", "high"):
                action = rec.get("action", str(rec.get("type", "Unknown")))
                wrapped = action[:62]
                lines.append("║" + f"  ▸ {wrapped}".ljust(68) + "║")
                critical_count += 1
                if critical_count >= 10:
                    remaining = len([r for r in all_recs if r.get('priority', '') in ('critical', 'high')]) - 10
                    lines.append("║" + f"    ... and {remaining} more".ljust(68) + "║")
                    break

        if critical_count == 0:
            lines.append("║" + "    No critical items.".ljust(68) + "║")

        # AI insights section
        ai_insights = results.get("ai_insights", [])
        if ai_insights:
            lines.append("╠" + "═" * 68 + "╣")
            lines.append("║" + "  GEMINI AI INSIGHTS".ljust(68) + "║")
            lines.append("║" + "  " + "-" * 64 + "  ║")

            # for insight in ai_insights:
            #     agent_name = insight.get("agent", "Unknown")
            #     phase = insight.get("phase", "")
            #     content = insight.get("content", "")
            #     header = f"[{agent_name} / {phase}]"
            #     lines.append("║" + f"  {header}".ljust(68) + "║")
            #     for content_line in content.split("\n")[:4]:  # cap at 4 lines per insight
            #         trimmed = content_line.strip()[:62]
            #         if trimmed:
            #             lines.append("║" + f"    {trimmed}".ljust(68) + "║")
            #     lines.append("║" + "".ljust(68) + "║")

            for insight in ai_insights:
                agent_name = insight.get("agent", "Unknown")
                phase = insight.get("phase", "")
                content = insight.get("content", "")
                header = f"[{agent_name} / {phase}]"
                lines.append("║" + f"  {header}".ljust(68) + "║")
                
                # Split into paragraphs and wrap each one
                for paragraph in content.split("\n"):
                    if not paragraph.strip():
                        continue
                        
                    # Wrap text to 62 chars so it fits perfectly inside your 68-char box
                    wrapped_lines = textwrap.wrap(paragraph.strip(), width=62)
                    for wrapped_line in wrapped_lines:
                        lines.append("║" + f"    {wrapped_line}".ljust(68) + "║")
                        
                lines.append("║" + "".ljust(68) + "║")

        else:
            lines.append("╠" + "═" * 68 + "╣")
            lines.append("║" + "  GEMINI AI INSIGHTS: None (Gemini offline)".ljust(68) + "║")

        lines.append("╚" + "═" * 68 + "╝")

        return "\n".join(lines)