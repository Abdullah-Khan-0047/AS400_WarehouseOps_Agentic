from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import DC_CONFIG, AGENT_CONFIG
from src.agents.base_agent import BaseAgent, GeminiClient, truncate_for_prompt
from src.models.domain import DockAssignmentRecord, Store


class DockAssignmentAgent(BaseAgent):
    """Learns optimal store-to-door mappings and uses Gemini to reason about assignments."""

    def __init__(self):
        super().__init__("DockAssignmentAgent")
        self.system_prompt = (
            "You are a dock assignment optimization AI for a retail distribution center. "
            "You analyze historical store-to-door assignment data and shrinkage outcomes "
            "to recommend optimal dock door assignments. Be concise, specific about door "
            "numbers, store IDs, and percentages. Focus on actionable insights."
        )
        self.store_door_scores: Dict[str, Dict[int, float]] = {}
        self.store_door_frequency: Dict[str, Counter] = {}
        self.door_shrinkage_history: Dict[int, List[float]] = defaultdict(list)
        self.door_avg_shrinkage: Dict[int, float] = {}
        self.assignments: List[DockAssignmentRecord] = []
        self.stores: List[Store] = []

    def initialize(self, historical_data: Dict) -> None:
        self.log_action("initialize", {"source": "historical_data"})

        self.assignments = historical_data["dock_assignments"]
        self.stores = historical_data["stores"]

        for store in self.stores:
            self.store_door_frequency[store.store_id] = Counter()

        for record in self.assignments:
            self.store_door_frequency[record.store_id][record.dock_door_id] += 1
            self.door_shrinkage_history[record.dock_door_id].append(record.shrinkage_rate)

        for door_id, rates in self.door_shrinkage_history.items():
            self.door_avg_shrinkage[door_id] = float(np.mean(rates)) if rates else 0.0

        self._compute_assignment_scores()

        self.is_initialized = True
        self.log_action("initialization_complete", {
            "stores_mapped": len(self.store_door_scores),
            "doors_profiled": len(self.door_avg_shrinkage),
        })

    def _compute_assignment_scores(self):
        w1 = AGENT_CONFIG.dock_history_weight
        w2 = AGENT_CONFIG.dock_proximity_weight
        w3 = AGENT_CONFIG.dock_shrinkage_weight

        for store in self.stores:
            sid = store.store_id
            freq = self.store_door_frequency[sid]
            total_assignments = sum(freq.values()) if freq else 1

            self.store_door_scores[sid] = {}

            for door_id in DC_CONFIG.outbound_doors:
                freq_score = freq.get(door_id, 0) / total_assignments
                proximity_score = 1.0 if freq_score > 0.1 else 0.5
                avg_shrink = self.door_avg_shrinkage.get(door_id, 0.0)
                shrinkage_score = max(0, 1.0 - (avg_shrink / 0.05))

                composite = (w1 * freq_score) + (w2 * proximity_score) + (w3 * shrinkage_score)
                self.store_door_scores[sid][door_id] = composite

    def get_best_door(self, store_id: str, occupied_doors: List[int] = None) -> Tuple[int, float]:
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        occupied = set(occupied_doors or [])
        scores = self.store_door_scores.get(store_id, {})

        if not scores:
            available = [d for d in DC_CONFIG.outbound_doors if d not in occupied]
            return (available[0] if available else -1, 0.0)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for door_id, score in ranked:
            if door_id not in occupied:
                return (door_id, score)

        all_doors = [d for d in DC_CONFIG.outbound_doors if d not in occupied]
        if all_doors:
            return (all_doors[0], 0.0)
        return (-1, 0.0)

    def ai_explain_assignment(self, store_id: str, door_id: int, confidence: float) -> Optional[str]:
        """Ask Gemini why this door is optimal for this store."""
        freq = self.store_door_frequency.get(store_id, {})
        top_doors = freq.most_common(3)
        door_shrink = self.door_avg_shrinkage.get(door_id, 0)

        prompt = (
            f"Store {store_id} has been assigned to dock door {door_id} "
            f"(confidence score: {confidence:.3f}).\n"
            f"This store's most frequently used doors: {top_doors}.\n"
            f"Door {door_id} average shrinkage rate: {door_shrink*100:.3f}%.\n"
            f"Explain in 2-3 sentences why this assignment is optimal and any risks."
        )
        return self.query_gemini(prompt)

    def analyze(self) -> List[Dict]:
        findings = []

        for store in self.stores:
            sid = store.store_id
            store_assignments = [a for a in self.assignments if a.store_id == sid]
            if not store_assignments:
                continue

            non_preferred = [a for a in store_assignments if not a.was_preferred_door]
            non_pref_rate = len(non_preferred) / len(store_assignments)

            if non_pref_rate > 0.25:
                pref_shrink = np.mean([a.shrinkage_rate for a in store_assignments if a.was_preferred_door]) \
                    if any(a.was_preferred_door for a in store_assignments) else 0
                non_pref_shrink = np.mean([a.shrinkage_rate for a in non_preferred]) if non_preferred else 0

                findings.append({
                    "type": "high_non_preferred_rate",
                    "store_id": sid,
                    "non_preferred_rate": non_pref_rate,
                    "preferred_shrinkage": pref_shrink,
                    "non_preferred_shrinkage": non_pref_shrink,
                    "shrinkage_increase": non_pref_shrink - pref_shrink,
                    "severity": "high" if non_pref_shrink > 0.03 else "medium",
                })

        for door_id, avg_shrink in self.door_avg_shrinkage.items():
            if avg_shrink > DC_CONFIG.shrinkage_alert_threshold:
                findings.append({
                    "type": "high_shrinkage_door",
                    "door_id": door_id,
                    "avg_shrinkage_rate": avg_shrink,
                    "severity": "critical" if avg_shrink > DC_CONFIG.shrinkage_critical_threshold else "high",
                })

        if findings:  # ask Gemini to reason about the findings
            ai_response = self.query_gemini(
                f"Analyze these dock assignment issues at a distribution center. "
                f"Identify root causes and patterns:\n{truncate_for_prompt(findings)}"
            )
            if ai_response:
                self.ai_insights.append({"phase": "analysis", "content": ai_response})

        self.log_action("analyze_complete", {"findings_count": len(findings)})
        return findings

    def recommend(self) -> List[Dict]:
        recommendations = []

        for store in self.stores:
            sid = store.store_id
            best_door, confidence = self.get_best_door(sid)

            current_preferred = store.preferred_dock_doors
            if best_door not in current_preferred:
                recommendations.append({
                    "type": "reassign_preferred_door",
                    "store_id": sid,
                    "current_preferred": current_preferred,
                    "recommended_door": best_door,
                    "confidence": confidence,
                    "reason": "Historical data suggests better performance at this door.",
                })

        if recommendations:  # ask Gemini to prioritize and explain
            ai_response = self.query_gemini(
                f"Given these dock reassignment recommendations, prioritize them "
                f"and explain which changes would have the biggest impact on reducing shrinkage:\n"
                f"{truncate_for_prompt(recommendations)}"
            )
            if ai_response:
                self.ai_insights.append({"phase": "recommendations", "content": ai_response})

        self.log_action("recommendations_generated", {"count": len(recommendations)})
        return recommendations