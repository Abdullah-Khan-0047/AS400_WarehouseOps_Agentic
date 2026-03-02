import os
from datetime import datetime
import json
from typing import Dict

from config.settings import DC_CONFIG, AGENT_CONFIG
from src.agents.base_agent import GeminiClient
from src.simulation.history_generator import HistoryGenerator
from src.agents.orchestrator import Orchestrator
from src.analytics.pattern_detector import PatternDetector
from src.legacy.as400_simulator import AS400Simulator


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def print_section_header(title: str):
    print("\n")
    print("█" * 70)
    print(f"█  {title}")
    print("█" * 70)


def demo_legacy_comparison(historical_data: Dict, orchestrator: Orchestrator):
    print_section_header("LEGACY AS/400 vs AGENTIC AI COMPARISON")

    as400 = AS400Simulator()
    as400.initialize(historical_data)

    print("\n  AS/400 Weekly Shrinkage Report (legacy batch output):")
    recent_assignments = historical_data["dock_assignments"][-50:]
    report = as400.generate_weekly_shrinkage_report(recent_assignments)
    print(report)

    print("\n  Side-by-Side Dock Assignment Comparison:")
    print("  " + "-" * 66)
    print(f"  {'STORE':<12} {'AS/400 DOOR':<14} {'AS/400 METHOD':<22} {'AI DOOR':<10} {'CONFIDENCE':<12}")
    print("  " + "-" * 66)

    test_stores = historical_data["stores"][:10]
    occupied = [22, 23, 24]

    for store in test_stores:
        as400_door, method = as400.assign_dock_door(store.store_id, occupied)
        ai_door, confidence = orchestrator.dock_agent.get_best_door(store.store_id, occupied)

        match = "✓" if as400_door == ai_door else "≠"
        print(
            f"  {store.store_id:<12} {as400_door:<14} {method:<22} "
            f"{ai_door:<10} {confidence:<10.3f}  {match}"
        )

    print("  " + "-" * 66)
    print("  ✓ = Same assignment  ≠ = Different (AI optimized)")


def demo_realtime_decisions(orchestrator: Orchestrator, historical_data: Dict):
    print_section_header("REAL-TIME SHIPMENT PROCESSING DEMO")

    scenarios = [
        {"store_id": "STR-0001", "shift": "1st", "category": "apparel"},
        {"store_id": "STR-0005", "shift": "3rd", "category": "jewelry"},
        {"store_id": "STR-0010", "shift": "2nd", "category": "accessories"},
        {"store_id": "STR-0015", "shift": "3rd", "category": "footwear"},
        {"store_id": "STR-0020", "shift": "1st", "category": "home_goods"},
    ]

    occupied_doors = [21, 22, 23, 30, 31]

    for scenario in scenarios:
        result = orchestrator.simulate_incoming_shipment(
            occupied_doors=occupied_doors,
            **scenario,
        )
        if result["assigned_door"] > 0:
            occupied_doors.append(result["assigned_door"])

        print()


def demo_pattern_analysis(historical_data: Dict):
    print_section_header("DEEP PATTERN ANALYSIS")

    detector = PatternDetector()
    report = detector.generate_full_report(historical_data)

    print("\n  Temporal Patterns:")
    print("  " + "-" * 50)
    for pattern in report["temporal_patterns"]:
        if pattern["pattern_type"] in ("day_of_week", "monthly_trend"):
            print(f"    {pattern['detail']}")

    print("\n  Top Store-Door Affinities (strongest relationships):")
    print("  " + "-" * 50)
    for aff in report["store_door_affinities"][:5]:
        print(
            f"    {aff['store_id']}: Best=Door {aff['best_door']} "
            f"({aff['best_door_shrinkage']*100:.2f}%), "
            f"Worst=Door {aff['worst_door']} "
            f"({aff['worst_door_shrinkage']*100:.2f}%)"
        )

    print("\n  Category Vulnerability Insights:")
    print("  " + "-" * 50)
    shown = set()
    for vuln in report["category_vulnerabilities"]:
        if "dominant_shrinkage_type" in vuln and vuln["category"] not in shown:
            print(f"    {vuln['detail']}")
            shown.add(vuln["category"])

    savings = report["potential_savings"]
    if "error" not in savings:
        print("\n  Potential Savings Estimate:")
        print("  " + "-" * 50)
        print(f"    Shrinkage increase from non-preferred doors: "
              f"{savings['shrinkage_increase_factor']:.1f}x")
        print(f"    Estimated unit savings:   {savings['potential_savings_units']:,.0f}")
        print(f"    Estimated $ savings:      ${savings['estimated_savings_dollars']:,.2f}")
        print(f"    (Assuming ${savings['avg_unit_value_assumed']:.2f} avg unit value)")

def export_for_powerbi(results: Dict, folder_name: str = "power_bi_data", filename: str = "powerbi_export.json"):
    try:
        os.makedirs(folder_name, exist_ok=True)
        
        filepath = os.path.join(folder_name, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        print(f"\n Successfully exported data to {filepath} for Power BI")
    except Exception as e:
        print(f"\n Failed to export data: {e}")

def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  AGENTIC AI DISTRIBUTION CENTER OPTIMIZER".center(68) + "║")
    print("║" + "  Replacing Legacy AS/400 with Intelligent Agents".center(68) + "║")
    print("║" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    load_env()

    print_section_header("STEP 0: CONNECTING TO GEMINI")
    GeminiClient.configure()

    print_section_header("STEP 1: GENERATING SIMULATED HISTORY (AS/400 Legacy Data)")
    generator = HistoryGenerator()
    historical_data = generator.generate_all()

    print_section_header("STEP 2: INITIALIZING AGENTIC AI SYSTEM")
    orchestrator = Orchestrator()
    orchestrator.initialize_all(historical_data)

    print_section_header("STEP 3: FULL ANALYSIS CYCLE")
    results = orchestrator.run_analysis_cycle()

    print_section_header("STEP 4: EXECUTIVE SUMMARY")
    summary = orchestrator.generate_executive_summary(results)
    print(summary)

    export_for_powerbi(results) # saving for power bi

    demo_realtime_decisions(orchestrator, historical_data)

    demo_legacy_comparison(historical_data, orchestrator)

    demo_pattern_analysis(historical_data)

    print_section_header("COMPLETE")

    gemini_status = "ACTIVE" if GeminiClient.is_available() else "OFFLINE"

    print(f"""
    Gemini Status: {gemini_status}

    ┌─────────────────────────┬──────────────────────┬───────────────────────────┐
    │ Function                │ AS/400 (Legacy)      │ Agentic AI (New)          │
    ├─────────────────────────┼──────────────────────┼───────────────────────────┤
    │ Dock Assignment         │ Static lookup table  │ ML-scored + Gemini reason │
    │ Shrinkage Detection     │ Weekly batch report  │ Real-time + AI anomaly    │
    │ Audit Scheduling        │ Fixed rotation       │ Risk-based + AI priority  │
    │ Cross-correlation       │ Manual (manager)     │ Inter-agent + AI synthesis│
    │ Institutional Knowledge │ In people's heads    │ Learned + AI explained    │
    │ Adaptation              │ None (code change)   │ Continuous + AI reasoning │
    │ New Shipment Risk       │ Not possible         │ Instant AI risk scoring   │
    └─────────────────────────┴──────────────────────┴───────────────────────────┘

    To enable Gemini AI reasoning:
      1. Get an API key from https://aistudio.google.com/app/apikey
      2. Create a .env file with: GEMINI_API_KEY=your-key-here
      3. Run again: python main.py
    """)


if __name__ == "__main__":
    main()