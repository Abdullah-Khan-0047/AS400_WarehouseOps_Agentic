from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DCConfig:
    dc_id: str = "DC-001"
    dc_name: str = "Northeast Distribution Center"
    total_dock_doors: int = 50
    inbound_doors: List[int] = field(default_factory=lambda: list(range(1, 21)))
    outbound_doors: List[int] = field(default_factory=lambda: list(range(21, 51)))
    shifts: List[str] = field(default_factory=lambda: ["1st", "2nd", "3rd"])

    baseline_shrinkage_rates: Dict[str, float] = field(default_factory=lambda: {
        "apparel": 0.018,
        "footwear": 0.022,
        "accessories": 0.030,
        "home_goods": 0.012,
        "jewelry": 0.045,
    })

    num_stores: int = 30
    history_days: int = 365
    num_trailers: int = 80

    shrinkage_alert_threshold: float = 0.025
    shrinkage_critical_threshold: float = 0.05
    misdirect_alert_threshold: int = 3
    audit_risk_threshold: float = 0.7

    random_seed: int = 42


@dataclass
class AgentConfig:
    dock_history_weight: float = 0.6
    dock_proximity_weight: float = 0.2
    dock_shrinkage_weight: float = 0.2

    anomaly_lookback_days: int = 30
    anomaly_std_multiplier: float = 2.0

    max_audits_per_shift: int = 5
    high_value_categories: List[str] = field(
        default_factory=lambda: ["jewelry", "accessories", "footwear"]
    )


@dataclass
class GeminiConfig:
    model_name: str = "gemini-2.5-flash" #"gemini-2.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 2048
    max_prompt_items: int = 10  # cap items sent in prompts to control token usage


DC_CONFIG = DCConfig()
AGENT_CONFIG = AgentConfig()
GEMINI_CONFIG = GeminiConfig()

