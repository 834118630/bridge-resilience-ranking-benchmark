"""Generate diagnostic figures for the E1 Monte Carlo outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "e1_monte_carlo"
FIGURE_DIR = RESULT_DIR / "figures"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def heatmap(rows: list[dict[str, str]], statistic: str, scenario: str, output: Path) -> None:
    selected = [
        row
        for row in rows
        if row["statistic"] == statistic and row["recovery_scenario"] == scenario
    ]
    cases = sorted({row["case"] for row in selected})
    metrics = sorted({row["metric"] for row in selected})
    sa_values = sorted({float(row["sa"]) for row in selected})
    figure, axes = plt.subplots(
        len(cases),
        len(metrics),
        figsize=(4.2 * len(metrics), 3.0 * len(cases)),
        squeeze=False,
    )
    for row_index, case_id in enumerate(cases):
        for col_index, metric in enumerate(metrics):
            matrix = np.zeros((len(sa_values), 1), dtype=float)
            for sa_index, sa in enumerate(sa_values):
                match = next(
                    row
                    for row in selected
                    if row["case"] == case_id
                    and row["metric"] == metric
                    and float(row["sa"]) == sa
                )
                matrix[sa_index, 0] = float(match["mean"])
            axis = axes[row_index][col_index]
            image = axis.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
            axis.set_xticks([0], [case_id])
            axis.set_yticks(range(len(sa_values)), [f"{value:g}" for value in sa_values])
            axis.set_title(metric.replace("_", " "), fontsize=8)
            axis.set_ylabel("Sa (g)" if col_index == 0 else "")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{scenario}: mean {statistic.replace('_', ' ')}")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def regret_bar(path: Path, scenario: str, output: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))[scenario]
    labels = list(data)
    means = np.asarray([data[label]["mean_sample_minimax_regret"] for label in labels])
    lows = np.asarray([data[label]["bootstrap_ci95_low"] for label in labels])
    highs = np.asarray([data[label]["bootstrap_ci95_high"] for label in labels])
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.vlines(x, lows, highs, color="#4878a8", linewidth=4, alpha=0.55)
    axis.plot(x, means, "o", color="#183b56", markersize=7, label="mean")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Sample minimax regret")
    axis.legend(frameon=False)
    axis.set_title(f"{scenario}: candidate regret under recovery-time uncertainty")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(RESULT_DIR / "uncertainty_summary.csv")
    scenarios = sorted({row["recovery_scenario"] for row in rows})
    for scenario in scenarios:
        for statistic in ("pairwise_reversal_rate", "top1_change_rate"):
            output = FIGURE_DIR / f"{scenario}_{statistic}.png"
            heatmap(rows, statistic, scenario, output)
            if scenario == "shape_only":
                heatmap(rows, statistic, scenario, FIGURE_DIR / f"{statistic}.png")
        regret_output = FIGURE_DIR / f"{scenario}_minimax_regret.png"
        regret_bar(RESULT_DIR / "minimax_regret_uncertainty.json", scenario, regret_output)
        if scenario == "shape_only":
            regret_bar(
                RESULT_DIR / "minimax_regret_uncertainty.json",
                scenario,
                FIGURE_DIR / "minimax_regret.png",
            )
    print("figures written to", FIGURE_DIR)


if __name__ == "__main__":
    main()
