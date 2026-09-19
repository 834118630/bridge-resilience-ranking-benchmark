"""Two-dimensional scan of the capacity-cost and speed-cost coefficients.

The primary specification uses a single scenario coefficient for both terms,
alpha_c = alpha_s. This script separates them, because the linear cost surface
implies an exact condition for whether L3 is dominated by L4.

With theta_i = 1 for L0 to L4, the linear model gives

    C_i / C_L4 = 1 - alpha_c (1 - c_i) + alpha_s (1.75 - m_i).

For L3 (c = 0.75, m = 1.50) and L4 (c = 1.00, m = 1.75),

    C_L3 - C_L4 = 0.25 (alpha_s - alpha_c).

L3 is therefore never more expensive than L4 when alpha_c >= alpha_s, and it is
cheaper when alpha_s > alpha_c. Because L4 also has the lower LossDays, L4
dominates L3 exactly when alpha_c >= alpha_s.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .provenance import write_run_metadata

ROOT = Path(__file__).resolve().parents[2]


def _results_root() -> Path:
    """Support both layouts: the released package uses results/, the working
    project keeps generated output under tmp/."""

    for name in ("results", "tmp"):
        candidate = ROOT / name
        if (candidate / "e1_candidate_cost_frontier_final").is_dir():
            return candidate
    raise FileNotFoundError("no results directory found under the project root")


RESULTS_ROOT = _results_root()
OUTPUT_DIR = RESULTS_ROOT / "e1_alpha_sensitivity"
LOSS_PATH = RESULTS_ROOT / "e1_candidate_cost_frontier_final" / "candidate_loss_days.csv"

L4_MULTIPLIER = 1.75
L4_CAPACITY = 1.0
LABELS = ["L0", "L1", "L2", "L3", "L4", "F_fast"]
CAPACITIES = {"L0": 0.0, "L1": 0.25, "L2": 0.5, "L3": 0.75, "L4": 1.0, "F_fast": 0.5}
MULTIPLIERS = {
    "shape_only": {"L0": 1.0, "L1": 1.0, "L2": 1.0, "L3": 1.0, "L4": 1.0, "F_fast": 1.0},
    "evidence_tradeoff": {"L0": 1.0, "L1": 1.15, "L2": 1.30, "L3": 1.50, "L4": 1.75, "F_fast": 0.10},
    "boundary_tradeoff": {"L0": 1.0, "L1": 1.15, "L2": 1.30, "L3": 1.50, "L4": 1.75, "F_fast": 0.50},
}
TOLERANCE = 1e-9


def linear_cost(alpha_c: float, alpha_s: float, capacity: float, multiplier: float, theta: float = 1.0) -> float:
    factor = 1.0 - alpha_c * (1.0 - capacity) + alpha_s * (L4_MULTIPLIER - multiplier)
    if capacity == L4_CAPACITY and multiplier == L4_MULTIPLIER:
        factor = 1.0
    if factor < 0.0:
        raise ValueError("negative cost factor")
    # factor == 0 occurs only at the boundary alpha_c = 1, alpha_s = 0, where the
    # linear surface assigns the L0 baseline a cost index of zero. That point is
    # kept in the scan and reported as the edge of the model's validity.
    return theta * factor


def pareto_flags(costs: np.ndarray, losses: np.ndarray) -> np.ndarray:
    flags = np.ones(costs.shape, dtype=bool)
    for i in range(costs.size):
        for j in range(costs.size):
            if i == j:
                continue
            no_worse = costs[j] <= costs[i] + TOLERANCE and losses[j] <= losses[i] + TOLERANCE
            strictly_better = costs[j] < costs[i] - TOLERANCE or losses[j] < losses[i] - TOLERANCE
            if no_worse and strictly_better:
                flags[i] = False
                break
    return flags


def load_losses(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["scenario"], {})[row["candidate"]] = float(row["mean_loss_days"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-f", type=float, default=1.0)
    parser.add_argument("--grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    losses = load_losses(LOSS_PATH)
    grid = list(args.grid)
    rows = []
    for scenario in ["evidence_tradeoff", "boundary_tradeoff"]:
        loss_vec = np.array([losses[scenario][label] for label in LABELS], dtype=float)
        for alpha_c in grid:
            for alpha_s in grid:
                costs = np.array([
                    linear_cost(alpha_c, alpha_s, CAPACITIES[label], MULTIPLIERS[scenario][label],
                                args.theta_f if label == "F_fast" else 1.0)
                    for label in LABELS
                ])
                flags = pareto_flags(costs, loss_vec)
                l3 = LABELS.index("L3")
                l4 = LABELS.index("L4")
                l4_dominates_l3 = bool(
                    costs[l4] <= costs[l3] + TOLERANCE
                    and loss_vec[l4] <= loss_vec[l3] + TOLERANCE
                    and (
                        costs[l4] < costs[l3] - TOLERANCE
                        or loss_vec[l4] < loss_vec[l3] - TOLERANCE
                    )
                )
                rows.append({
                    "scenario": scenario,
                    "alpha_c": alpha_c,
                    "alpha_s": alpha_s,
                    "theta_f": args.theta_f,
                    "cost_L3_minus_L4": float(costs[l3] - costs[l4]),
                    "analytic_0p25_alpha_s_minus_alpha_c": float(0.25 * (alpha_s - alpha_c)),
                    "L4_dominates_L3": l4_dominates_l3,
                    "L3_pareto_efficient": bool(flags[l3]),
                    "pareto_members": ",".join(label for label, flag in zip(LABELS, flags) if flag),
                })

    out_csv = args.output_dir / "alpha_sensitivity.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    identity_error = max(abs(r["cost_L3_minus_L4"] - r["analytic_0p25_alpha_s_minus_alpha_c"]) for r in rows)
    zero_cost_rows = [r for r in rows if abs(r["cost_L3_minus_L4"]) >= 0 and r["alpha_c"] == 1.0 and r["alpha_s"] == 0.0]
    l4_domination_violations = [
        r for r in rows if r["L4_dominates_L3"] != (r["alpha_s"] >= r["alpha_c"])
    ]
    summary = {
        "grid": grid,
        "theta_f": args.theta_f,
        "rows": len(rows),
        "max_abs_error_analytic_identity": identity_error,
        "L4_dominates_L3_iff_alpha_s_ge_alpha_c": len(l4_domination_violations) == 0,
        "L4_domination_violations": l4_domination_violations[:5],
        "L3_pareto_efficient_count": {
            scenario: sum(
                1
                for r in rows
                if r["scenario"] == scenario and r["L3_pareto_efficient"]
            )
            for scenario in sorted({r["scenario"] for r in rows})
        },
        "boundary_note": "alpha_c = 1.0 with alpha_s = 0.0 puts the L0 cost index at zero; the point is retained as the edge of the linear model's validity.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_metadata(
        args.output_dir,
        command="python -m e1.run_alpha_sensitivity "
        f"--theta-f {args.theta_f} --grid "
        + " ".join(str(value) for value in grid)
        + f" --output-dir {args.output_dir}",
        parameters={"theta_f": args.theta_f, "grid": grid},
        input_paths=(LOSS_PATH,),
    )
    print(json.dumps(summary, indent=2))
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
