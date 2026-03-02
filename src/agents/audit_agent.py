from typing import Dict, List, Tuple

import numpy as np

from config.settings import DC_CONFIG, AGENT_CONFIG
from src.agents.base_agent import BaseAgent, GeminiClient, truncate_for_prompt
from src.models.domain import ShrinkageEvent, RiskLevel


class AuditAgent(BaseAgent):
    """Risk-scores door-shift combos, uses Gemini to reason about audit priorities."""

    def __init__(self):
        super().__init__("AuditAgent")
        self.system_prompt = (
            "You are an audit planning AI for a retail distribution center. "
            "You decide where to deploy limited auditor resources based on risk scores, "
            "shrinkage history, and high-value product exposure. Be specific about door "
            "numbers, shifts, and risk scores. Focus on maximizing audit ROI."
        )
        self.risk_scores: Dict[str, float] = {}
        self.audit_queue: List[Dict] = []
        self.events: List[ShrinkageEvent] = []
        self.assignments = []

    def initialize(self, historical_data: Dict) -> None:
        self.log_action("initialize", {"source": "historical_data"})

        self.events = historical_data["shrinkage_events"]
        self.assignments = historical_data["dock_assignments"]

        self._compute_risk_scores()
        self.is_initialized = True

        self.log_action("initialization_complete", {
            "risk_profiles_built": len(self.risk_scores),
        })

    def _compute_risk_scores(self):
        from collections import defaultdict

        door_shift_events = defaultdict(list)
        for event in self.events:
            key = f"door_{event.dock_door_id}_shift_{event.shift}"
            door_shift_events[key].append(event)

        door_shift_assignments = defaultdict(list)
        for asgn in self.assignments:
            key = f"door_{asgn.dock_door_id}_shift_{asgn.shift}"
            door_shift_assignments[key].append(asgn)

        for key, events in door_shift_events.items():
            if not events:
                continue

            avg_loss = np.mean([e.loss_rate for e in events])
            frequency = len(events)
            high_val_count = sum(1 for e in events if e.category in AGENT_CONFIG.high_value_categories)
            high_val_ratio = high_val_count / len(events)

            assignments = door_shift_assignments.get(key, [])
            non_pref_rate = 0
            if assignments:
                non_pref_count = sum(1 for a in assignments if not a.was_preferred_door)
                non_pref_rate = non_pref_count / len(assignments)

            risk = (
                0.35 * min(avg_loss / 0.10, 1.0) +
                0.20 * min(frequency / 100.0, 1.0) +
                0.25 * high_val_ratio +
                0.20 * non_pref_rate
            )

            self.risk_scores[key] = min(risk, 1.0)

    def analyze(self) -> List[Dict]:
        findings = []

        sorted_risks = sorted(
            self.risk_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for key, score in sorted_risks[:15]:
            parts = key.split("_")
            door_id = parts[1]
            shift = parts[3]

            if score >= 0.7:
                level = "critical"
            elif score >= 0.5:
                level = "high"
            elif score >= 0.3:
                level = "medium"
            else:
                level = "low"

            findings.append({
                "type": "audit_risk_assessment",
                "door_id": int(door_id),
                "shift": shift,
                "risk_score": score,
                "risk_level": level,
                "detail": f"Door {door_id} / {shift} shift: risk score {score:.3f}",
            })

        if findings:
            ai_response = self.query_gemini(
                f"Review these audit risk assessments for a distribution center. "
                f"Explain which areas deserve immediate attention and why:\n"
                f"{truncate_for_prompt(findings)}"
            )
            if ai_response:
                self.ai_insights.append({"phase": "analysis", "content": ai_response})

        self.log_action("analyze_complete", {"findings_count": len(findings)})
        return findings

    def recommend(self) -> List[Dict]:
        recommendations = []
        analysis = self.analyze()

        critical_high = [f for f in analysis if f["risk_level"] in ("critical", "high")]

        for i, finding in enumerate(critical_high[:AGENT_CONFIG.max_audits_per_shift * 3]):
            recommendations.append({
                "type": "scheduled_audit",
                "priority_rank": i + 1,
                "door_id": finding["door_id"],
                "shift": finding["shift"],
                "risk_score": finding["risk_score"],
                "risk_level": finding["risk_level"],
                "action": (
                    f"Audit dock door {finding['door_id']} during {finding['shift']} shift. "
                    f"Risk score: {finding['risk_score']:.3f}"
                ),
                "audit_type": "full_count" if finding["risk_level"] == "critical" else "spot_check",
            })

        if any(f["risk_level"] == "critical" for f in analysis):
            recommendations.append({
                "type": "process_change",
                "action": "Install cameras/additional oversight at critical-risk dock doors",
                "priority": "critical",
            })

        if recommendations:
            ai_response = self.query_gemini(
                f"Given these audit recommendations and limited auditor staff "
                f"({AGENT_CONFIG.max_audits_per_shift} audits per shift max), "
                f"create a prioritized deployment plan:\n"
                f"{truncate_for_prompt(recommendations)}"
            )
            if ai_response:
                self.ai_insights.append({"phase": "recommendations", "content": ai_response})

        self.log_action("recommendations_generated", {"count": len(recommendations)})
        return recommendations

    def should_audit_shipment(self, door_id: int, shift: str) -> Tuple[bool, float]:
        key = f"door_{door_id}_shift_{shift}"
        risk = self.risk_scores.get(key, 0.0)
        return (risk >= DC_CONFIG.audit_risk_threshold, risk)
    
