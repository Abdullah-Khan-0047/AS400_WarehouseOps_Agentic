import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

from config.settings import DC_CONFIG
from src.models.domain import Store, DockAssignmentRecord


class AS400Simulator:
    """Simulates AS/400 static-table dock assignment and batch reporting as a baseline."""

    def __init__(self):
        self.store_door_table: Dict[str, List[int]] = {}  # static lookup, no learning
        self.is_initialized = False

    def initialize(self, historical_data: Dict) -> None:
        self.store_door_table = dict(historical_data["store_preferred_doors"])
        self.stores = historical_data["stores"]
        self.is_initialized = True

    def assign_dock_door(
        self, store_id: str, occupied_doors: Optional[List[int]] = None
    ) -> Tuple[int, str]:
        """Static table lookup then random fallback — no intelligence."""
        occupied = set(occupied_doors or [])
        preferred = self.store_door_table.get(store_id, [])

        for door in preferred:  # try preferred first
            if door not in occupied:
                return (door, "preferred_table_lookup")

        available = [d for d in DC_CONFIG.outbound_doors if d not in occupied]  # random fallback
        if available:
            chosen = random.choice(available)
            return (chosen, "random_fallback")

        return (-1, "no_doors_available")

    def generate_weekly_shrinkage_report(self, assignments: List[DockAssignmentRecord]) -> str:
        """Simulates the weekly batch spooled file report printed from AS/400."""
        lines = []
        lines.append("")
        lines.append("  " + "=" * 72)
        lines.append("  WEEKLY SHRINKAGE SUMMARY REPORT              SYSTEM: AS400-DC001")
        lines.append("  PROGRAM: SHRNKRPT    LIBRARY: DCPRODLIB     RUN DATE: WEEKLY BATCH")
        lines.append("  " + "=" * 72)
        lines.append("")
        lines.append(f"  {'STORE':<12} {'DOOR':<8} {'SHIFT':<8} {'EXPECTED':>10} {'ACTUAL':>10} {'SHRINK%':>10} {'STATUS':<10}")
        lines.append("  " + "-" * 72)

        store_data = defaultdict(lambda: {"shrinkage": 0, "count": 0})

        for asgn in assignments:
            store_data[asgn.store_id]["shrinkage"] += asgn.shrinkage_rate
            store_data[asgn.store_id]["count"] += 1

        for store_id in sorted(store_data.keys()):
            data = store_data[store_id]
            avg_shrink = data["shrinkage"] / data["count"] if data["count"] > 0 else 0

            status = "OK"
            if avg_shrink > DC_CONFIG.shrinkage_critical_threshold:
                status = "**CRIT**"
            elif avg_shrink > DC_CONFIG.shrinkage_alert_threshold:
                status = "*ALERT*"

            lines.append(
                f"  {store_id:<12} {'MULTI':<8} {'ALL':<8} "
                f"{'N/A':>10} {'N/A':>10} "
                f"{avg_shrink*100:>9.3f}% {status:<10}"
            )

        lines.append("  " + "-" * 72)
        lines.append("  END OF REPORT")
        lines.append("  " + "=" * 72)
        lines.append("")
        lines.append("  *** NOTE: This report shows PAST data only. For real-time    ***")
        lines.append("  *** monitoring, see the Agentic AI DC Optimizer dashboard.   ***")
        lines.append("")

        return "\n".join(lines)

    def compare_with_agentic(
        self,
        store_id: str,
        occupied_doors: List[int],
        agentic_door: int,
        agentic_confidence: float,
    ) -> Dict:
        as400_door, as400_method = self.assign_dock_door(store_id, occupied_doors)

        return {
            "store_id": store_id,
            "as400": {
                "assigned_door": as400_door,
                "method": as400_method,
                "uses_history": False,
                "considers_shrinkage": False,
                "real_time_risk": False,
            },
            "agentic_ai": {
                "assigned_door": agentic_door,
                "confidence": agentic_confidence,
                "uses_history": True,
                "considers_shrinkage": True,
                "real_time_risk": True,
            },
            "same_assignment": as400_door == agentic_door,
        }
    