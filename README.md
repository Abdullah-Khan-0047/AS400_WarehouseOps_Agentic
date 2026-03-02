# Agentic AI Distribution Center Optimizer

Uses AI agents to replace legacy mainframe functions at a distribution center. The focus is on cutting shrinkage and eliminating human error in dock-to-store trailer assignments by learning from historical data.

---

## What Problem Does This Solve

Distribution centers running on older mainframe systems handle dock assignments, shipment tracking, and shrinkage reporting through a mix of:

- Batch programs running on fixed schedules
- Static lookup tables maintained through terminal data entry
- Weekly printed reports reviewed manually by supervisors
- Knowledge that only exists in experienced workers' heads

This causes problems:

| Problem | Why It Happens |
|---|---|
| Trailers docked at wrong doors | Static assignment table plus manual overrides when doors are busy |
| Shrinkage found days or weeks late | Batch reporting, no real-time visibility |
| Audits miss high-risk shipments | Fixed rotation schedules, no risk weighting |
| Knowledge lost when people leave | Best door assignments were never captured in the system |
| Nobody connects the dots between systems | Dock assignment, shrinkage, and audit programs run separately |

The big insight behind this project: when a store's shipment goes to the dock door it has historically been assigned to, shrinkage stays low. When it gets routed to a different door because of congestion or a new supervisor or a manual override, shrinkage goes up 2 to 3 times. That relationship was never in the old system. It was in people's heads.

---

## What This Does

Simulates a year of distribution center history with realistic patterns baked in, then runs three AI agents that learn from that history and make decisions the old system never could. Each agent also uses Gemini to reason about its findings in plain English.

### Agents

**Dock Assignment Agent** — Replaces the batch program that assigned trailers to doors using a flat lookup table. This one scores every store-to-door combination using historical frequency, shrinkage outcomes, and proximity. When a trailer shows up it picks the best available door, then asks Gemini to explain why that door was chosen.

**Shrinkage Detection Agent** — Replaces the weekly batch shrinkage report. Builds statistical baselines for loss rates across every dimension (store, door, shift, product category) and flags anomalies in real time. Asks Gemini to find hidden correlations between the anomalies that a human reviewing numbers in a spreadsheet would miss.

**Audit Agent** — Replaces the fixed-rotation audit schedule. Computes a risk score for every door-shift combination using loss history, event frequency, high-value product exposure, and misassignment rates. Asks Gemini to build a deployment plan so limited auditor staff goes where it matters most.

**Orchestrator** — Coordinates the three agents. Routes messages between them so they can act on each other's findings. For example, the shrinkage agent warns the dock agent about a problematic door, and the dock agent tells the audit agent about a store that keeps getting misassigned. Then asks Gemini to synthesize all three agents' findings into one strategic view. The old system had nothing like this. A manager would read three separate printed reports and try to connect them mentally.

### Gemini Integration

Every agent sends its statistical findings to Gemini and gets back:
- Root cause analysis for detected anomalies
- Hidden correlations across stores, doors, shifts, and categories
- Prioritized action plans ranked by expected impact
- Plain English explanations of each dock assignment decision and risk score

The system works without Gemini (stats-only mode) but the AI reasoning layer is what makes it actually agentic. Without it, the agents just run math and return numbers. With it, they explain their reasoning, find connections between each other's findings, and generate action plans a shift supervisor can actually follow.

---

## Setup

Install dependencies:

```bash
pip install numpy google-genai python-dotenv