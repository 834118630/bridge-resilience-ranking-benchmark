"""Plot the final E1 economic-winner decision map."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "results" / "e1_candidate_cost_frontier_final" / "economic_utility.csv"
OUTPUT = PROJECT_ROOT / "figures" / "legacy_economic_winner_decision_map.png"
CANDIDATES = ["L0", "L1", "L2", "L3", "L4", "F_fast"]
SCENARIOS = ["evidence_tradeoff", "boundary_tradeoff"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def winner_for_cell(rows: list[dict[str, str]], scenario: str, theta: float, vday: float) -> int:
    cell = [
        row for row in rows
        if row["scenario"] == scenario
        and row["cost_form"] == "linear"
        and row["alpha_scenario"] == "moderate"
        and np.isclose(float(row["theta_f"]), theta)
        and np.isclose(float(row["normalized_v_day"]), vday)
    ]
    winners = [row for row in cell if row["is_winner"].lower() == "true"]
    if not winners:
        raise RuntimeError(f"no winner for {scenario}, theta={theta}, vday={vday}")
    winner = max(winners, key=lambda row: float(row["winner_frequency_bootstrap"]))
    return CANDIDATES.index(winner["candidate"])


def main() -> None:
    rows = load_rows(INPUT)
    vdays = sorted({float(row["normalized_v_day"]) for row in rows})
    thetas = sorted({float(row["theta_f"]) for row in rows if row["cost_form"] == "linear" and row["alpha_scenario"] == "moderate"})

    colors = ["#B8B8B8", "#9CC3E5", "#67A9CF", "#D9D9D9", "#2CA02C", "#D62728"]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(CANDIDATES) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for axis, scenario in zip(axes, SCENARIOS):
        matrix = np.empty((len(thetas), len(vdays)), dtype=float)
        for row_index, theta in enumerate(thetas):
            for column_index, vday in enumerate(vdays):
                matrix[row_index, column_index] = winner_for_cell(rows, scenario, theta, vday)
        axis.pcolormesh(np.arange(len(vdays) + 1), np.arange(len(thetas) + 1), matrix, cmap=cmap, norm=norm)
        axis.set_xticks(np.arange(len(vdays)) + 0.5, [f"{value:g}" for value in vdays], rotation=45, ha="right")
        axis.set_yticks(np.arange(len(thetas)) + 0.5, [f"{value:g}" for value in thetas])
        axis.set_xlabel(r"$u_{\mathrm{day}}=v_{\mathrm{day}}/C_{L4}$")
        axis.set_title(scenario.replace("_", " "))
        axis.grid(False)
    axes[0].set_ylabel("F_fast cost multiplier theta_F")
    legend = [Patch(facecolor=color, edgecolor="black", label=candidate) for color, candidate in zip(colors, CANDIDATES)]
    figure.legend(handles=legend, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Economic winner map: linear cost, alpha=0.5", fontsize=13)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
