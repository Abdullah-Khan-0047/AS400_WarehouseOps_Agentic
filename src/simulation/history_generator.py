import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from config.settings import DC_CONFIG, AGENT_CONFIG
from src.models.domain import (
    Store, DockDoor, Trailer, Shipment, ShipmentLine, ShipmentStatus,
    ShrinkageEvent, ShrinkageType, RiskLevel, DockAssignmentRecord,
)


class HistoryGenerator:
    """Generates 1 year of simulated DC history with embedded shrinkage patterns."""

    REGIONS = ["Northeast", "Southeast", "Mid-Atlantic", "New England"]
    CARRIERS = ["Schneider", "JB Hunt", "Werner", "Swift", "Heartland"]
    CATEGORIES = ["apparel", "footwear", "accessories", "home_goods", "jewelry"]

    SKU_POOL = {
        "apparel": [
            ("SKU-APP-001", "Women's Blouse"),
            ("SKU-APP-002", "Men's Polo"),
            ("SKU-APP-003", "Kids Jacket"),
            ("SKU-APP-004", "Women's Dress"),
            ("SKU-APP-005", "Men's Jeans"),
        ],
        "footwear": [
            ("SKU-FTW-001", "Running Shoes"),
            ("SKU-FTW-002", "Sandals"),
            ("SKU-FTW-003", "Boots"),
        ],
        "accessories": [
            ("SKU-ACC-001", "Handbag"),
            ("SKU-ACC-002", "Sunglasses"),
            ("SKU-ACC-003", "Belt"),
        ],
        "home_goods": [
            ("SKU-HMG-001", "Throw Pillow"),
            ("SKU-HMG-002", "Candle Set"),
            ("SKU-HMG-003", "Picture Frame"),
        ],
        "jewelry": [
            ("SKU-JWL-001", "Necklace"),
            ("SKU-JWL-002", "Bracelet"),
        ],
    }

    def __init__(self, config=DC_CONFIG):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        random.seed(config.random_seed)

        self.stores: List[Store] = []
        self.dock_doors: List[DockDoor] = []
        self.trailers: List[Trailer] = []
        self.shipments: List[Shipment] = []
        self.shrinkage_events: List[ShrinkageEvent] = []
        self.dock_assignments: List[DockAssignmentRecord] = []
        self.store_preferred_doors: Dict[str, List[int]] = {}  # store_id -> [door_ids]

    def generate_all(self) -> Dict:
        print("=" * 60)
        print("  HISTORY GENERATOR — Simulating AS/400 Legacy Data")
        print("=" * 60)

        self._generate_stores()
        self._generate_dock_doors()
        self._assign_preferred_doors()
        self._generate_trailers()
        self._generate_shipment_history()

        summary = {
            "stores": self.stores,
            "dock_doors": self.dock_doors,
            "trailers": self.trailers,
            "shipments": self.shipments,
            "shrinkage_events": self.shrinkage_events,
            "dock_assignments": self.dock_assignments,
            "store_preferred_doors": self.store_preferred_doors,
        }

        self._print_summary()
        return summary

    def _generate_stores(self):
        print("\n[1/5] Generating store master data...")
        for i in range(1, self.config.num_stores + 1):
            store = Store(
                store_id=f"STR-{i:04d}",
                store_name=f"Store #{i}",
                region=random.choice(self.REGIONS),
                avg_weekly_shipments=random.randint(2, 5),
            )
            self.stores.append(store)
        print(f"       Created {len(self.stores)} stores.")

    def _generate_dock_doors(self):
        print("[2/5] Generating dock door layout...")
        zones = ["A", "B", "C", "D", "E"]
        for door_id in range(1, self.config.total_dock_doors + 1):
            door_type = "inbound" if door_id in self.config.inbound_doors else "outbound"
            zone = zones[(door_id - 1) // 10]
            door = DockDoor(door_id=door_id, door_type=door_type, zone=zone)
            self.dock_doors.append(door)
        print(f"       Created {len(self.dock_doors)} dock doors "
              f"({len(self.config.inbound_doors)} inbound, "
              f"{len(self.config.outbound_doors)} outbound).")

    def _assign_preferred_doors(self):
        print("[3/5] Assigning preferred dock doors (institutional knowledge)...")
        outbound = list(self.config.outbound_doors)
        random.shuffle(outbound)

        idx = 0
        for store in self.stores:
            num_preferred = random.choice([1, 2])
            preferred = []
            for _ in range(num_preferred):
                preferred.append(outbound[idx % len(outbound)])
                idx += 1
            store.preferred_dock_doors = preferred
            self.store_preferred_doors[store.store_id] = preferred

        print(f"       Mapped {len(self.stores)} stores to preferred doors.")

    def _generate_trailers(self):
        print("[4/5] Generating trailer pool...")
        for i in range(1, self.config.num_trailers + 1):
            trailer = Trailer(
                trailer_id=f"TRL-{i:04d}",
                carrier=random.choice(self.CARRIERS),
                store_id="",
            )
            self.trailers.append(trailer)
        print(f"       Created {len(self.trailers)} trailers.")

    def _generate_shipment_history(self):
        print("[5/5] Generating shipment history (this simulates the AS/400 SHPHIST)...")

        start_date = datetime.now() - timedelta(days=self.config.history_days)
        shipment_count = 0
        shrinkage_count = 0

        for day_offset in range(self.config.history_days):
            current_date = start_date + timedelta(days=day_offset)

            if current_date.weekday() == 6:  # skip Sundays
                continue

            volume_mult = self._seasonal_multiplier(current_date.month)

            for store in self.stores:
                daily_prob = (store.avg_weekly_shipments / 6.0) * volume_mult
                if self.rng.random() > daily_prob:
                    continue

                shift = random.choice(self.config.shifts)
                trailer = random.choice(self.trailers)

                preferred_doors = self.store_preferred_doors[store.store_id]
                if self.rng.random() < 0.80:  # 80% assigned to preferred door
                    dock_door_id = random.choice(preferred_doors)
                    is_preferred = True
                else:  # 20% assigned to non-preferred door (congestion, manual override)
                    non_preferred = [d for d in self.config.outbound_doors
                                     if d not in preferred_doors]
                    dock_door_id = random.choice(non_preferred)
                    is_preferred = False

                num_lines = random.randint(3, 8)
                lines = self._generate_shipment_lines(
                    num_lines, store.store_id, is_preferred, shift
                )

                shipment_id = f"SHP-{current_date.strftime('%Y%m%d')}-{shipment_count:05d}"

                shipment = Shipment(
                    shipment_id=shipment_id,
                    store_id=store.store_id,
                    trailer_id=trailer.trailer_id,
                    dock_door_id=dock_door_id,
                    shift=shift,
                    date=current_date,
                    status=ShipmentStatus.COMPLETE,
                    lines=lines,
                )
                self.shipments.append(shipment)

                assignment = DockAssignmentRecord(
                    date=current_date,
                    shift=shift,
                    store_id=store.store_id,
                    dock_door_id=dock_door_id,
                    trailer_id=trailer.trailer_id,
                    shipment_id=shipment_id,
                    shrinkage_rate=shipment.shrinkage_rate,
                    misdirected_units=sum(
                        l.shrinkage_units for l in lines
                        if l.actual_qty < l.expected_qty
                    ),
                    was_preferred_door=is_preferred,
                )
                self.dock_assignments.append(assignment)

                for line in lines:
                    if line.shrinkage_units > 0:
                        event = self._create_shrinkage_event(
                            shipment, line, dock_door_id, shift, current_date
                        )
                        self.shrinkage_events.append(event)
                        shrinkage_count += 1

                shipment_count += 1

        print(f"       Generated {shipment_count} shipments with "
              f"{shrinkage_count} shrinkage events.")

    def _generate_shipment_lines(
        self, num_lines: int, store_id: str, is_preferred_door: bool, shift: str
    ) -> List[ShipmentLine]:
        lines = []
        for i in range(num_lines):
            category = random.choice(self.CATEGORIES)
            sku, desc = random.choice(self.SKU_POOL[category])
            expected_qty = random.randint(10, 200)

            base_rate = self.config.baseline_shrinkage_rates[category]

            if not is_preferred_door:
                base_rate *= self.rng.uniform(2.0, 3.5)  # non-preferred doors = higher shrinkage

            if shift == "3rd":
                base_rate *= 1.5  # fatigue, less supervision
            elif shift == "2nd":
                base_rate *= 1.2

            if self.rng.random() < 0.15:  # 15% of lines have some shrinkage
                shrinkage_units = max(1, int(expected_qty * base_rate * self.rng.uniform(0.5, 3.0)))
                actual_qty = max(0, expected_qty - shrinkage_units)
            else:
                actual_qty = expected_qty

            line = ShipmentLine(
                line_id=i + 1,
                sku=sku,
                category=category,
                description=desc,
                expected_qty=expected_qty,
                actual_qty=actual_qty,
                destination_store_id=store_id,
            )
            lines.append(line)

        return lines

    def _create_shrinkage_event(
        self, shipment: Shipment, line: ShipmentLine,
        dock_door_id: int, shift: str, date: datetime
    ) -> ShrinkageEvent:
        if line.category in ["jewelry", "accessories"]:  # high-value = more likely missing
            shrink_type = random.choice([
                ShrinkageType.MISSING, ShrinkageType.MISSING,
                ShrinkageType.COUNT_DISCREPANCY,
            ])
        else:
            shrink_type = random.choice([
                ShrinkageType.COUNT_DISCREPANCY,
                ShrinkageType.MISDIRECTED,
                ShrinkageType.DAMAGED,
                ShrinkageType.MISSING,
            ])

        loss_rate = line.shrinkage_units / line.expected_qty if line.expected_qty > 0 else 0
        if loss_rate > 0.05:
            risk = RiskLevel.CRITICAL
        elif loss_rate > 0.025:
            risk = RiskLevel.HIGH
        elif loss_rate > 0.01:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        return ShrinkageEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:8].upper()}",
            shipment_id=shipment.shipment_id,
            store_id=shipment.store_id,
            dock_door_id=dock_door_id,
            shift=shift,
            date=date,
            shrinkage_type=shrink_type,
            category=line.category,
            expected_qty=line.expected_qty,
            actual_qty=line.actual_qty,
            units_lost=line.shrinkage_units,
            risk_level=risk,
        )

    @staticmethod
    def _seasonal_multiplier(month: int) -> float:
        seasonal = {
            1: 0.8, 2: 0.8, 3: 0.9, 4: 0.9,
            5: 1.0, 6: 1.0, 7: 0.9, 8: 1.0,
            9: 1.1, 10: 1.3, 11: 1.5, 12: 1.4,  # Oct-Dec holiday ramp
        }
        return seasonal.get(month, 1.0)

    def _print_summary(self):
        total_expected = sum(s.total_expected for s in self.shipments)
        total_actual = sum(s.total_actual for s in self.shipments)
        total_shrinkage = total_expected - total_actual

        preferred_assignments = sum(1 for a in self.dock_assignments if a.was_preferred_door)
        non_preferred = len(self.dock_assignments) - preferred_assignments

        pref_shrink = [a.shrinkage_rate for a in self.dock_assignments if a.was_preferred_door]
        non_pref_shrink = [a.shrinkage_rate for a in self.dock_assignments if not a.was_preferred_door]

        print("\n" + "=" * 60)
        print("  GENERATED DATA SUMMARY")
        print("=" * 60)
        print(f"  Stores:                {len(self.stores)}")
        print(f"  Dock Doors:            {len(self.dock_doors)}")
        print(f"  Trailers:              {len(self.trailers)}")
        print(f"  Shipments:             {len(self.shipments)}")
        print(f"  Shrinkage Events:      {len(self.shrinkage_events)}")
        print(f"  Total Units Expected:  {total_expected:,}")
        print(f"  Total Units Actual:    {total_actual:,}")
        print(f"  Total Shrinkage Units: {total_shrinkage:,}")
        print(f"  Overall Shrinkage %:   {(total_shrinkage/total_expected)*100:.3f}%")
        print(f"  Preferred Door Asgn:   {preferred_assignments} ({preferred_assignments/len(self.dock_assignments)*100:.1f}%)")
        print(f"  Non-Preferred Asgn:    {non_preferred} ({non_preferred/len(self.dock_assignments)*100:.1f}%)")
        if pref_shrink:
            print(f"  Avg Shrink (Preferred):     {np.mean(pref_shrink)*100:.3f}%")
        if non_pref_shrink:
            print(f"  Avg Shrink (Non-Preferred): {np.mean(non_pref_shrink)*100:.3f}%")
        print("=" * 60)