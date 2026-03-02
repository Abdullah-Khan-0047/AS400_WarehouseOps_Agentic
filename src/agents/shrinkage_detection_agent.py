from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from config.settings import DC_CONFIG, AGENT_CONFIG
from src.agents.base_agent import BaseAgent, GeminiClient, truncate_for_prompt
from src.models.domain import ShrinkageEvent, RiskLevel


class ShrinkageDetectionAgent(BaseAgent):
    """Builds per-dimension shrinkage baselines, flags anomalies, uses Gemini to reason about patterns."""

    def __init__(self):
        super().__init__("ShrinkageDetectionAgent")
        self.system_prompt = (
            "You are a shrinkage detection AI for a retail distribution center. "
            "You analyze inventory loss patterns across stores, dock doors, shifts, "
            "and product categories. Be specific about percentages and IDs. "
            "Focus on identifying root causes and correlations humans would miss."
        )
        self.store_baselines: Dict[str, Dict] = {}
        self.door_baselines: Dict[int, Dict] = {}
        self.shift_baselines: Dict[str, Dict] = {}
        self.category_baselines: Dict[str, Dict] = {}
        self.door_shift_patterns: Dict[str, Dict] = {}
        self.events: List[ShrinkageEvent] = []
        self.anomalies: List[Dict] = []

    def initialize(self, historical_data: Dict) -> None:
        self.log_action("initialize", {"source": "shrinkage_events"})

        self.events = historical_data["shrinkage_events"]
        assignments = historical_data["dock_assignments"]

        self._build_baselines_by_key(
            [(e.store_id, e.loss_rate) for e in self.events],
            self.store_baselines
        )
        self._build_baselines_by_key(
            [(e.dock_door_id, e.loss_rate) for e in self.events],
            self.door_baselines
        )
        self._build_baselines_by_key(
            [(e.shift, e.loss_rate) for e in self.events],
            self.shift_baselines
        )
        self._build_baselines_by_key(
            [(e.category, e.loss_rate) for e in self.events],
            self.category_baselines
        )
        self._build_baselines_by_key(
            [(f"{e.dock_door_id}-{e.shift}", e.loss_rate) for e in self.events],
            self.door_shift_patterns
        )

        self.preferred_shrinkage_rates = [a.shrinkage_rate for a in assignments if a.was_preferred_door]
        self.non_preferred_shrinkage_rates = [a.shrinkage_rate for a in assignments if not a.was_preferred_door]

        self.is_initialized = True
        self.log_action("initialization_complete", {
            "events_analyzed": len(self.events),
            "store_baselines": len(self.store_baselines),
            "door_baselines": len(self.door_baselines),
        })

    @staticmethod
    def _build_baselines_by_key(pairs, target_dict):
        grouped = defaultdict(list)
        for key, value in pairs:
            grouped[key].append(value)
        for key, values in grouped.items():
            target_dict[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
                "max": float(np.max(values)),
                "p95": float(np.percentile(values, 95)),
            }

    def analyze(self) -> List[Dict]:
        findings = []
        mult = AGENT_CONFIG.anomaly_std_multiplier

        for store_id, baseline in self.store_baselines.items():
            if baseline["mean"] > DC_CONFIG.shrinkage_alert_threshold:
                findings.append({
                    "dimension": "store",
                    "key": store_id,
                    "metric": "avg_loss_rate",
                    "value": baseline["mean"],
                    "threshold": DC_CONFIG.shrinkage_alert_threshold,
                    "severity": "critical" if baseline["mean"] > DC_CONFIG.shrinkage_critical_threshold else "high",
                    "detail": f"Store {store_id} avg shrinkage {baseline['mean']*100:.2f}% exceeds threshold",
                })

        overall_door_mean = np.mean([b["mean"] for b in self.door_baselines.values()])
        overall_door_std = np.std([b["mean"] for b in self.door_baselines.values()])

        for door_id, baseline in self.door_baselines.items():
            if baseline["mean"] > overall_door_mean + mult * overall_door_std:
                findings.append({
                    "dimension": "dock_door",
                    "key": door_id,
                    "metric": "avg_loss_rate",
                    "value": baseline["mean"],
                    "expected": overall_door_mean,
                    "std_devs_above": (baseline["mean"] - overall_door_mean) / overall_door_std if overall_door_std > 0 else 0,
                    "severity": "high",
                    "detail": f"Door {door_id} shrinkage significantly above average",
                })

        for shift, baseline in self.shift_baselines.items():
            findings.append({
                "dimension": "shift",
                "key": shift,
                "metric": "avg_loss_rate",
                "value": baseline["mean"],
                "event_count": baseline["count"],
                "severity": "info",
                "detail": f"Shift {shift}: avg loss rate {baseline['mean']*100:.2f}%, {baseline['count']} events",
            })

        for category, baseline in self.category_baselines.items():
            if baseline["mean"] > DC_CONFIG.shrinkage_alert_threshold:
                findings.append({
                    "dimension": "category",
                    "key": category,
                    "metric": "avg_loss_rate",
                    "value": baseline["mean"],
                    "severity": "high" if category in AGENT_CONFIG.high_value_categories else "medium",
                    "detail": f"Category '{category}' avg loss {baseline['mean']*100:.2f}%",
                })

        if self.preferred_shrinkage_rates and self.non_preferred_shrinkage_rates:
            pref_mean = np.mean(self.preferred_shrinkage_rates)
            non_pref_mean = np.mean(self.non_preferred_shrinkage_rates)
            findings.append({
                "dimension": "assignment_type",
                "key": "preferred_vs_non_preferred",
                "preferred_avg": pref_mean,
                "non_preferred_avg": non_pref_mean,
                "increase_factor": non_pref_mean / pref_mean if pref_mean > 0 else 0,
                "severity": "critical",
                "detail": (
                    f"Non-preferred dock assignments show {non_pref_mean/pref_mean:.1f}x "
                    f"higher shrinkage ({non_pref_mean*100:.3f}% vs {pref_mean*100:.3f}%)"
                ),
            })

        if findings:
            ai_response = self.query_gemini(
                f"Analyze these shrinkage anomalies from a distribution center. "
                f"Identify hidden correlations, root causes, and which issues are connected:\n"
                f"{truncate_for_prompt(findings)}"
            )
            if ai_response:
                self.ai_insights.append({"phase": "analysis", "content": ai_response})

        self.anomalies = findings
        self.log_action("analyze_complete", {"findings_count": len(findings)})
        return findings

    def recommend(self) -> List[Dict]:
        recommendations = []

        store_issues = sorted(
            [(sid, b["mean"]) for sid, b in self.store_baselines.items()],
            key=lambda x: x[1], reverse=True
        )[:5]

        for store_id, avg_loss in store_issues:
            recommendations.append({
                "type": "store_shrinkage_review",
                "store_id": store_id,
                "avg_loss_rate": avg_loss,
                "action": f"Review processes for {store_id} — avg loss {avg_loss*100:.2f}%",
                "priority": "high",
            })

        door_issues = sorted(
            [(did, b["mean"]) for did, b in self.door_baselines.items()],
            key=lambda x: x[1], reverse=True
        )[:5]

        for door_id, avg_loss in door_issues:
            recommendations.append({
                "type": "door_investigation",
                "door_id": door_id,
                "avg_loss_rate": avg_loss,
                "action": f"Investigate door {door_id} — potential security or process issue",
                "priority": "high",
            })

        shift_sorted = sorted(self.shift_baselines.items(), key=lambda x: x[1]["mean"], reverse=True)
        if shift_sorted:
            worst_shift = shift_sorted[0]
            recommendations.append({
                "type": "shift_process_improvement",
                "shift": worst_shift[0],
                "avg_loss_rate": worst_shift[1]["mean"],
                "action": f"Add supervision/training for {worst_shift[0]} shift — highest shrinkage",
                "priority": "medium",
            })

        if self.preferred_shrinkage_rates and self.non_preferred_shrinkage_rates:
            pref_mean = np.mean(self.preferred_shrinkage_rates)
            non_pref_mean = np.mean(self.non_preferred_shrinkage_rates)
            if non_pref_mean > pref_mean * 1.5:
                recommendations.append({
                    "type": "enforce_preferred_assignments",
                    "action": "CRITICAL: Enforce preferred dock door assignments. "
                              f"Non-preferred doors show {non_pref_mean/pref_mean:.1f}x higher shrinkage.",
                    "estimated_savings_pct": (non_pref_mean - pref_mean) * 100,
                    "priority": "critical",
                })

        if recommendations:
            ai_response = self.query_gemini(
                f"Given these shrinkage reduction recommendations for a DC, "
                f"rank them by expected ROI and suggest implementation order:\n"
                f"{truncate_for_prompt(recommendations)}"
            )
            if ai_response:
                self.ai_insights.append({"phase": "recommendations", "content": ai_response})

        self.log_action("recommendations_generated", {"count": len(recommendations)})
        return recommendations

    def evaluate_new_shipment(self, store_id: str, dock_door_id: int,
                               shift: str, category: str) -> Dict:
        risk_score = 0.0
        risk_factors = []

        store_bl = self.store_baselines.get(store_id, {})
        if store_bl.get("mean", 0) > DC_CONFIG.shrinkage_alert_threshold:
            risk_score += 0.3
            risk_factors.append(f"Store {store_id} has elevated shrinkage history")

        door_bl = self.door_baselines.get(dock_door_id, {})
        if door_bl.get("mean", 0) > DC_CONFIG.shrinkage_alert_threshold:
            risk_score += 0.3
            risk_factors.append(f"Door {dock_door_id} has elevated shrinkage history")

        shift_bl = self.shift_baselines.get(shift, {})
        if shift_bl.get("mean", 0) > DC_CONFIG.shrinkage_alert_threshold:
            risk_score += 0.2
            risk_factors.append(f"Shift {shift} has higher shrinkage rates")

        cat_bl = self.category_baselines.get(category, {})
        if cat_bl.get("mean", 0) > DC_CONFIG.shrinkage_alert_threshold:
            risk_score += 0.2
            risk_factors.append(f"Category '{category}' is high-shrinkage")

        risk_score = min(risk_score, 1.0)

        if risk_score >= 0.7:
            level = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        result = {
            "risk_score": risk_score,
            "risk_level": level.value,
            "factors": risk_factors,
            "recommend_audit": risk_score >= DC_CONFIG.audit_risk_threshold,
            "ai_assessment": None,
        }

        if risk_score >= 0.3 and GeminiClient.is_available():  # only call Gemini for non-trivial risk 
            ai_response = self.query_gemini(
                f"Shipment arriving: store={store_id}, door={dock_door_id}, "
                f"shift={shift}, category={category}.\n"
                f"Risk score: {risk_score:.2f}, level: {level.value}.\n"
                f"Risk factors: {risk_factors}\n"
                f"In 2-3 sentences, explain what could go wrong and one specific preventive action."
            )
            if ai_response:
                result["ai_assessment"] = ai_response

        return result