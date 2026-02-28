from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class ShipmentStatus(Enum):
    SCHEDULED = "scheduled"
    IN_TRANSIT = "in_transit"
    DOCKED = "docked"
    UNLOADING = "unloading"
    COMPLETE = "complete"
    FLAGGED = "flagged"


class ShrinkageType(Enum):
    COUNT_DISCREPANCY = "count_discrepancy"
    MISDIRECTED = "misdirected"
    DAMAGED = "damaged"
    MISSING = "missing"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Store:
    store_id: str
    store_name: str
    region: str
    preferred_dock_doors: List[int] = field(default_factory=list)
    avg_weekly_shipments: int = 3

    def __hash__(self):
        return hash(self.store_id)


@dataclass
class DockDoor:
    door_id: int
    door_type: str  # "inbound" or "outbound"
    zone: str = ""
    is_occupied: bool = False
    current_trailer_id: Optional[str] = None
    historical_shrinkage_rate: float = 0.0
    historical_misdirect_count: int = 0
    most_frequent_store: Optional[str] = None
    assignment_count: int = 0

    def __hash__(self):
        return hash(self.door_id)


@dataclass
class Trailer:
    trailer_id: str
    carrier: str
    store_id: str
    trailer_type: str = "53ft"
    seal_number: Optional[str] = None

    def __hash__(self):
        return hash(self.trailer_id)


@dataclass
class ShipmentLine:
    line_id: int
    sku: str
    category: str
    description: str
    expected_qty: int
    actual_qty: int = 0
    destination_store_id: str = ""

    @property
    def variance(self) -> int:
        return self.actual_qty - self.expected_qty

    @property
    def shrinkage_units(self) -> int:
        return max(0, self.expected_qty - self.actual_qty)


@dataclass
class Shipment:
    shipment_id: str
    store_id: str
    trailer_id: str
    dock_door_id: int
    shift: str
    date: datetime
    status: ShipmentStatus
    lines: List[ShipmentLine] = field(default_factory=list)

    @property
    def total_expected(self) -> int:
        return sum(l.expected_qty for l in self.lines)

    @property
    def total_actual(self) -> int:
        return sum(l.actual_qty for l in self.lines)

    @property
    def total_shrinkage(self) -> int:
        return sum(l.shrinkage_units for l in self.lines)

    @property
    def shrinkage_rate(self) -> float:
        if self.total_expected == 0:
            return 0.0
        return self.total_shrinkage / self.total_expected


@dataclass
class ShrinkageEvent:
    event_id: str
    shipment_id: str
    store_id: str
    dock_door_id: int
    shift: str
    date: datetime
    shrinkage_type: ShrinkageType
    category: str
    expected_qty: int
    actual_qty: int
    units_lost: int
    risk_level: RiskLevel = RiskLevel.LOW
    investigated: bool = False
    root_cause: Optional[str] = None

    @property
    def loss_rate(self) -> float:
        if self.expected_qty == 0:
            return 0.0
        return self.units_lost / self.expected_qty


@dataclass
class DockAssignmentRecord:
    date: datetime
    shift: str
    store_id: str
    dock_door_id: int
    trailer_id: str
    shipment_id: str
    shrinkage_rate: float
    misdirected_units: int
    was_preferred_door: bool