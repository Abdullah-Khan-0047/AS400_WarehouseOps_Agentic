from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from datetime import datetime, timedelta

from src.models.domain import (
    DockAssignmentRecord, ShrinkageEvent, Shipment
)


class PatternDetector:
    """Finds temporal, affinity, and category patterns in DC operational data."""

    def __init__(self):
        self.assignments: List[DockAssignmentRecord] = []
        self.shrinkage_events: List[ShrinkageEvent] = []
        self.shipments: List[Shipment] = []

    def load_data(self, historical_data: Dict):
        self.assignments = historical_data.get("dock_assignments", [])
        self.shrinkage_events = historical_data.get("shrinkage_events", [])
        self.shipments = historical_data.get("shipments", [])

    def detect_temporal_patterns(self) -> List[Dict]:
        patterns = []

        dow_shrinkage = defaultdict(list)  # day-of-week -> [loss_rates]
        for event in self.shrinkage_events:
            dow = event.date.strftime("%A")
            dow_shrinkage[dow].append(event.loss_rate)

        dow_means = {dow: np.mean(rates) for dow, rates in dow_shrinkage.items() if rates}
        if dow_means:
            worst_day = max(dow_means, key=dow_means.get)
            best_day = min(dow_means, key=dow_means.get)
            overall_mean = np.mean(list(dow_means.values()))

            patterns.append({
                "pattern_type": "day_of_week",
                "detail": f"Worst day: {worst_day} ({dow_means[worst_day]*100:.2f}%), "
                          f"Best day: {best_day} ({dow_means[best_day]*100:.2f}%)",
                "data": dow_means,
                "worst": worst_day,
                "best": best_day,
                "overall_mean": overall_mean,
            })

        monthly_shrinkage = defaultdict(list)  # "YYYY-MM" -> [loss_rates]
        for event in self.shrinkage_events:
            month_key = event.date.strftime("%Y-%m")
            monthly_shrinkage[month_key].append(event.loss_rate)

        if monthly_shrinkage:
            monthly_means = {m: np.mean(rates) for m, rates in sorted(monthly_shrinkage.items())}
            values = list(monthly_means.values())

            midpoint = len(values) // 2
            if midpoint > 0:
                first_half = np.mean(values[:midpoint])
                second_half = np.mean(values[midpoint:])
                trend = "improving" if second_half < first_half else "worsening"

                patterns.append({
                    "pattern_type": "monthly_trend",
                    "detail": f"Shrinkage trend is {trend}: "
                              f"H1 avg {first_half*100:.3f}% → H2 avg {second_half*100:.3f}%",
                    "trend": trend,
                    "first_half_avg": first_half,
                    "second_half_avg": second_half,
                    "monthly_data": monthly_means,
                })

        shift_data = defaultdict(lambda: {"rates": [], "counts": defaultdict(int)})
        for event in self.shrinkage_events:
            shift_data[event.shift]["rates"].append(event.loss_rate)
            shift_data[event.shift]["counts"][event.category] += 1

        for shift, data in shift_data.items():
            top_category = max(data["counts"], key=data["counts"].get) if data["counts"] else "N/A"
            patterns.append({
                "pattern_type": "shift_detail",
                "shift": shift,
                "avg_loss_rate": np.mean(data["rates"]),
                "event_count": len(data["rates"]),
                "most_affected_category": top_category,
                "detail": f"{shift} shift: {len(data['rates'])} events, "
                          f"avg loss {np.mean(data['rates'])*100:.2f}%, "
                          f"most affected: {top_category}",
            })

        return patterns

    def detect_store_door_affinity(self) -> List[Dict]:
        """Finds which store-door combos yield best/worst shrinkage outcomes."""
        affinities = []

        store_assignments = defaultdict(list)
        for asgn in self.assignments:
            store_assignments[asgn.store_id].append(asgn)

        for store_id, assignments in store_assignments.items():
            door_perf = defaultdict(list)  # door_id -> [shrinkage_rates]
            for a in assignments:
                door_perf[a.dock_door_id].append(a.shrinkage_rate)

            if not door_perf:
                continue

            door_avgs = {door: np.mean(rates) for door, rates in door_perf.items() if rates}
            if len(door_avgs) < 2:
                continue

            best_door = min(door_avgs, key=door_avgs.get)
            worst_door = max(door_avgs, key=door_avgs.get)

            if door_avgs[worst_door] > door_avgs[best_door] * 1.5:  # meaningful difference only
                affinities.append({
                    "store_id": store_id,
                    "best_door": best_door,
                    "best_door_shrinkage": door_avgs[best_door],
                    "best_door_count": len(door_perf[best_door]),
                    "worst_door": worst_door,
                    "worst_door_shrinkage": door_avgs[worst_door],
                    "worst_door_count": len(door_perf[worst_door]),
                    "improvement_potential": door_avgs[worst_door] - door_avgs[best_door],
                    "all_doors": door_avgs,
                })

        affinities.sort(key=lambda x: x["improvement_potential"], reverse=True)
        return affinities

    def detect_category_vulnerabilities(self) -> List[Dict]:
        vulnerabilities = []

        cat_shift = defaultdict(lambda: defaultdict(list))  # category -> shift -> [rates]
        for event in self.shrinkage_events:
            cat_shift[event.category][event.shift].append(event.loss_rate)

        for category, shifts in cat_shift.items():
            for shift, rates in shifts.items():
                avg_rate = np.mean(rates)
                if avg_rate > 0.02:  # flag if >2%
                    vulnerabilities.append({
                        "category": category,
                        "shift": shift,
                        "avg_loss_rate": avg_rate,
                        "event_count": len(rates),
                        "max_loss_rate": np.max(rates),
                        "detail": f"{category} on {shift} shift: "
                                  f"avg loss {avg_rate*100:.2f}% over {len(rates)} events",
                    })

        cat_type = defaultdict(lambda: defaultdict(int))  # category -> shrinkage_type -> count
        for event in self.shrinkage_events:
            cat_type[event.category][event.shrinkage_type.value] += 1

        for category, types in cat_type.items():
            dominant_type = max(types, key=types.get)
            total = sum(types.values())
            vulnerabilities.append({
                "category": category,
                "dominant_shrinkage_type": dominant_type,
                "type_ratio": types[dominant_type] / total,
                "all_types": dict(types),
                "detail": f"{category}: {types[dominant_type]/total*100:.0f}% "
                          f"of losses are '{dominant_type}'",
            })

        vulnerabilities.sort(key=lambda x: x.get("avg_loss_rate", 0), reverse=True)
        return vulnerabilities

    def compute_potential_savings(self, historical_data: Dict) -> Dict:
        """Estimates shrinkage reduction if all assignments used preferred doors."""
        assignments = historical_data.get("dock_assignments", [])
        shipments = historical_data.get("shipments", [])

        if not assignments:
            return {"error": "No assignment data available"}

        preferred = [a for a in assignments if a.was_preferred_door]
        non_preferred = [a for a in assignments if not a.was_preferred_door]

        pref_avg = np.mean([a.shrinkage_rate for a in preferred]) if preferred else 0
        non_pref_avg = np.mean([a.shrinkage_rate for a in non_preferred]) if non_preferred else 0

        total_units = sum(s.total_expected for s in shipments)
        non_pref_units = sum(
            s.total_expected for s in shipments
            for a in non_preferred
            if a.shipment_id == s.shipment_id
        )

        current_non_pref_loss = non_pref_avg * non_pref_units if non_pref_units else 0
        potential_non_pref_loss = pref_avg * non_pref_units if non_pref_units else 0
        potential_savings_units = current_non_pref_loss - potential_non_pref_loss

        avg_unit_value = 15.0  # rough avg for off-price retail
        potential_savings_dollars = potential_savings_units * avg_unit_value

        return {
            "preferred_avg_shrinkage": pref_avg,
            "non_preferred_avg_shrinkage": non_pref_avg,
            "shrinkage_increase_factor": non_pref_avg / pref_avg if pref_avg > 0 else 0,
            "total_units_processed": total_units,
            "non_preferred_units": non_pref_units,
            "current_loss_estimate_units": current_non_pref_loss,
            "potential_loss_with_fix_units": potential_non_pref_loss,
            "potential_savings_units": potential_savings_units,
            "estimated_savings_dollars": potential_savings_dollars,
            "avg_unit_value_assumed": avg_unit_value,
        }

    def generate_full_report(self, historical_data: Dict) -> Dict:
        self.load_data(historical_data)

        return {
            "temporal_patterns": self.detect_temporal_patterns(),
            "store_door_affinities": self.detect_store_door_affinity(),
            "category_vulnerabilities": self.detect_category_vulnerabilities(),
            "potential_savings": self.compute_potential_savings(historical_data),
        }
    