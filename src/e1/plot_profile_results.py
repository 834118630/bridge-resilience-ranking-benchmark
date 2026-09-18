"""Diagnostic figures for the profile-based Monte Carlo run."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "e1_profile_monte_carlo"
FIGURE_DIR = RESULT_DIR / "figures"
SCENARIO_ORDER = ["shape_only", "evidence_tradeoff", "boundary_tradeoff"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def regret_panel(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, len(SCENARIO_ORDER), figsize=(15, 4.5), sharey=True)
    for axis, scenario in zip(axes, SCENARIO_ORDER):
        selected = [row for row in rows if row["scenario"] == scenario]
        selected.sort(key=lambda row: float(row["mean_regret"]))
        labels = [row["candidate"] for row in selected]
        mean = np.asarray([float(row["mean_regret"]) for row in selected])
        low = np.asarray([float(row["bootstrap_ci95_low"]) for row in selected])
        high = np.asarray([float(row["bootstrap_ci95_high"]) for row in selected])
        x = np.arange(len(labels))
        axis.vlines(x, low, high, color="#4C78A8", linewidth=4, alpha=0.55)
        axis.plot(x, mean, "o", color="#1B3A57", markersize=7)
        axis.set_xticks(x, labels)
        axis.set_ylim(-0.02, 1.05)
        axis.set_title(scenario.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean normalized regret")
    figure.suptitle("Candidate mean regret under three recovery scenarios")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "mean_regret_by_scenario.png", dpi=240)
    plt.close(figure)


def top1_panel(rows: list[dict[str, str]]) -> None:
    candidates = sorted({row["candidate"] for row in rows})
    x = np.arange(len(candidates))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for offset, (scenario, color) in enumerate(zip(SCENARIO_ORDER, colors)):
        selected = {
            row["candidate"]: float(row["top1_frequency"])
            for row in rows
            if row["scenario"] == scenario
        }
        axis.bar(
            x + (offset - 1) * width,
            [selected.get(candidate, 0.0) for candidate in candidates],
            width,
            label=scenario.replace("_", " "),
            color=color,
        )
    axis.set_xticks(x, candidates)
    axis.set_ylabel("Top-1 frequency")
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "top1_frequency_by_scenario.png", dpi=240)
    plt.close(figure)


def ranking_heatmap(rows: list[dict[str, str]], statistic: str, scenario: str) -> None:
    selected = [
        row
        for row in rows
        if row["statistic"] == statistic and row["scenario"] == scenario
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
            matrix = np.zeros((len(sa_values), 1))
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
    figure.savefig(FIGURE_DIR / f"{scenario}_{statistic}.png", dpi=220)
    plt.close(figure)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    candidate_rows = load_rows(RESULT_DIR / "candidate_summary.csv")
    ranking_rows = load_rows(RESULT_DIR / "ranking_summary.csv")
    regret_panel(candidate_rows)
    top1_panel(candidate_rows)
    for scenario in ("evidence_tradeoff", "boundary_tradeoff"):
        ranking_heatmap(ranking_rows, "top1_change_rate", scenario)
        ranking_heatmap(ranking_rows, "pairwise_reversal_rate", scenario)
    print("figures written to", FIGURE_DIR)


if __name__ == "__main__":
    main()
