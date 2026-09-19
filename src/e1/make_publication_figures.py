"""Generate publication-quality figures for the E1 manuscript."""

from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
STYLE = ROOT / "styles" / "publication.mplstyle"
if STYLE.exists():
    plt.style.use(STYLE)
else:
    plt.style.use("seaborn-v0_8-whitegrid")
# Embed TrueType fonts in vector output (Type 3 fonts are often rejected by publishers).
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
CANDIDATES = ["F_fast", "L4", "L3", "L2", "L1", "L0"]
COLORS = {"L0": "#999999", "L1": "#56B4E9", "L2": "#0072B2", "L3": "#666666", "L4": "#009E73", "F_fast": "#D55E00"}
MARKERS = {"L0": "s", "L1": "^", "L2": "o", "L3": "v", "L4": "D", "F_fast": "P"}
SCENARIOS = ["shape_only", "evidence_tradeoff", "boundary_tradeoff"]
# Display labels. The scenario keys are legacy internal identifiers and are kept so that the
# archived result files remain readable without modification.
SCENARIO_LABELS = {"shape_only": "Shape-only", "evidence_tradeoff": "Strong-speed-advantage", "boundary_tradeoff": "Near-boundary"}


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIG_DIR / f"{name}.eps", format="eps", bbox_inches="tight")
    figure.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def figure_mean_regret() -> None:
    rows = read_rows(ROOT / "results" / "e1_profile_monte_carlo" / "candidate_summary.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), sharex=True)
    for axis, scenario in zip(axes, SCENARIOS):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        for candidate in CANDIDATES:
            row = next(item for item in scenario_rows if item["candidate"] == candidate)
            mean = float(row["mean_regret"])
            low = float(row["bootstrap_ci95_low"])
            high = float(row["bootstrap_ci95_high"])
            axis.errorbar(mean, candidate, xerr=np.array([[mean - low], [high - mean]]), fmt=MARKERS[candidate], color=COLORS[candidate], ecolor=COLORS[candidate], elinewidth=0.8, capsize=2, markersize=4.5)
            axis.annotate(f"{mean:.3f}", xy=(high, candidate), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5)
        axis.grid(axis="x", color="#E3E3E3", linewidth=0.5)
        axis.set_title(SCENARIO_LABELS[scenario], fontsize=8)
        axis.set_xlim(-0.03, 1.18)
        axis.set_xlabel("Mean normalized regret")
    axes[0].set_ylabel("Candidate")
    figure.suptitle("Mean regret with 95% bootstrap intervals", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, "fig2_mean_regret_dotplot")


def figure_top1_matrix() -> None:
    candidate_rows = read_rows(ROOT / "results" / "e1_profile_monte_carlo" / "candidate_summary.csv")
    ranking_rows = read_rows(ROOT / "results" / "e1_profile_monte_carlo" / "ranking_summary.csv")
    matrix = np.zeros((len(CANDIDATES), len(SCENARIOS)))
    for row in candidate_rows:
        if row["candidate"] in CANDIDATES and row["scenario"] in SCENARIOS:
            matrix[CANDIDATES.index(row["candidate"]), SCENARIOS.index(row["scenario"])] = float(
                row["top1_frequency"]
            )

    figure, (axis_a, axis_b) = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.9),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    image = axis_a.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis_a.set_xticks(
        range(len(SCENARIOS)),
        [SCENARIO_LABELS[s].replace("-", "-\n") for s in SCENARIOS],
        fontsize=7,
    )
    axis_a.set_yticks(range(len(CANDIDATES)), CANDIDATES)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis_a.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.55 else "black",
            )
            if value >= 0.5:
                axis_a.add_patch(
                    plt.Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#D55E00",
                        linewidth=1.2,
                    )
                )
    axis_a.set_title("(a) Metric-aggregated Top-1 frequency", fontsize=8.5)
    figure.colorbar(image, ax=axis_a, fraction=0.046, pad=0.04, label="Frequency")

    metric_order = [
        "resilience_index",
        "functionality_day_30",
        "functionality_day_90",
        "recovery_rapidity_90",
    ]
    metric_labels = {
        "resilience_index": "Resilience\nindex",
        "functionality_day_30": "Functionality\n30 d",
        "functionality_day_90": "Functionality\n90 d",
        "recovery_rapidity_90": "Recovery\nrapidity",
    }
    scenario_order = ["shape_only", "evidence_tradeoff", "boundary_tradeoff"]
    mean_matrix = np.zeros((len(metric_order), len(scenario_order)))
    max_matrix = np.zeros_like(mean_matrix)
    for metric_index, metric_name in enumerate(metric_order):
        for scenario_index, scenario_name in enumerate(scenario_order):
            cells = [
                float(row["mean"])
                for row in ranking_rows
                if row["scenario"] == scenario_name
                and row["metric"] == metric_name
                and row["statistic"] == "top1_change_rate"
            ]
            if not cells:
                raise RuntimeError(
                    f"missing top1_change_rate for {scenario_name}/{metric_name}"
                )
            mean_matrix[metric_index, scenario_index] = float(np.mean(cells))
            max_matrix[metric_index, scenario_index] = float(np.max(cells))

    image_b = axis_b.imshow(
        mean_matrix,
        cmap="OrRd",
        vmin=0.0,
        vmax=0.75,
        aspect="auto",
    )
    axis_b.set_xticks(
        range(len(scenario_order)),
        [SCENARIO_LABELS[s].replace("-", "-\n") for s in scenario_order],
        fontsize=7,
    )
    axis_b.set_yticks(
        range(len(metric_order)),
        [metric_labels[m] for m in metric_order],
        fontsize=7,
    )
    for row in range(mean_matrix.shape[0]):
        for column in range(mean_matrix.shape[1]):
            mean_value = mean_matrix[row, column]
            max_value = max_matrix[row, column]
            axis_b.text(
                column,
                row,
                f"mean {mean_value:.3f}\nmax {max_value:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if mean_value > 0.42 else "black",
            )
    axis_b.set_title("(b) Shape-induced Top-1 change rate", fontsize=8.5)
    figure.colorbar(
        image_b,
        ax=axis_b,
        fraction=0.046,
        pad=0.04,
        label="Mean change rate",
    )
    figure.tight_layout()
    save_figure(figure, "fig1_top1_matrix")


def figure_phase_boundary() -> None:
    rows = read_rows(ROOT / "results" / "e1_profile_phase_diagram" / "phase_diagram.csv")
    capacities = sorted({float(row["capacity"]) for row in rows})
    speeds = sorted({float(row["speed_multiplier"]) for row in rows})
    z = np.zeros((len(capacities), len(speeds)))
    for row in rows:
        z[capacities.index(float(row["capacity"])), speeds.index(float(row["speed_multiplier"]))] = float(row["regret_difference_F_minus_L4"])
    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    levels = np.linspace(-0.8, 0.8, 17)
    image = axis.contourf(speeds, capacities, z, levels=levels, cmap="RdBu_r", extend="both")
    zero = axis.contour(speeds, capacities, z, levels=[0.0], colors="black", linewidths=1.4)
    axis.clabel(zero, fmt={0.0: "decision boundary"}, fontsize=7)
    axis.text(0.08, 0.08, "F_fast better", color="#0072B2", fontsize=8, fontweight="bold")
    axis.text(0.75, 0.9, "L4 high better", color="#B2182B", fontsize=8, fontweight="bold")
    axis.set_xlabel("Fast-profile recovery multiplier")
    axis.set_ylabel("Fast-profile capacity level")
    axis.set_title("Performance boundary: mean regret difference", fontsize=9)
    figure.colorbar(image, ax=axis, label="Mean regret: F_fast - L4 high", fraction=0.046, pad=0.04)
    figure.tight_layout()
    save_figure(figure, "fig3_phase_boundary")


def figure_pareto_frontier() -> None:
    rows = read_rows(ROOT / "results" / "e1_candidate_cost_frontier_final" / "candidate_frontier.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    for axis, scenario in zip(axes, ["evidence_tradeoff", "boundary_tradeoff"]):
        selected = [row for row in rows if row["scenario"] == scenario and row["cost_form"] == "linear" and row["alpha_scenario"] == "moderate" and abs(float(row["theta_f"]) - 1.0) < 1e-9]
        selected = sorted(selected, key=lambda row: CANDIDATES.index(row["candidate"]))
        for row in selected:
            candidate = row["candidate"]
            x = float(row["cost_index"])
            y = float(row["mean_loss_days"])
            if candidate == "L0":
                axis.scatter(x, y, marker="s", s=48, facecolors="white", edgecolors="#666666", linewidths=1.2, label="L0 baseline")
            elif row["pareto_actionable"].lower() == "true":
                axis.scatter(x, y, marker="D", s=48, color=COLORS[candidate], label=candidate)
            else:
                axis.scatter(x, y, marker="x", s=42, color="#999999", label=candidate if candidate == "L3" else None)
            axis.annotate(candidate, (x, y), xytext=(8, 5), textcoords="offset points", fontsize=7, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        actionable = [row for row in selected if row["pareto_actionable"].lower() == "true"]
        actionable.sort(key=lambda row: float(row["cost_index"]))
        axis.plot([float(row["cost_index"]) for row in actionable], [float(row["mean_loss_days"]) for row in actionable], linestyle="--", color="#33B18F", linewidth=1.1)
        axis.set_xlabel("Normalized cost index (L4 = 1)")
        axis.set_title(SCENARIO_LABELS[scenario], fontsize=9)
        axis.grid(color="#F1F1F1", linewidth=0.6)
    axes[0].set_ylabel("Discounted loss days")
    handles = [Line2D([], [], marker="x", linestyle="none", color="#999999", markersize=5, label="Dominated"), Patch(edgecolor="#666666", facecolor="white", label="L0 baseline"), Line2D([], [], color="#009E73", linestyle="--", linewidth=1.1, label="Engineering frontier")]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    figure.suptitle("Six-candidate performance–cost frontier", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(figure, "fig4_pareto_frontier")


def figure_decision_map() -> None:
    rows = read_rows(ROOT / "results" / "e1_candidate_cost_frontier_final" / "economic_utility.csv")
    vdays = sorted({float(row["normalized_v_day"]) for row in rows if float(row["normalized_v_day"]) > 0.0})
    thetas = sorted({float(row["theta_f"]) for row in rows if row["cost_form"] == "linear" and row["alpha_scenario"] == "moderate"})
    index = {candidate: position for position, candidate in enumerate(CANDIDATES)}
    cmap = ListedColormap([COLORS[candidate] for candidate in CANDIDATES])
    boundaries = np.arange(-0.5, len(CANDIDATES) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for axis, scenario in zip(axes, ["evidence_tradeoff", "boundary_tradeoff"]):
        matrix = np.zeros((len(thetas), len(vdays)))
        for row_index, theta in enumerate(thetas):
            for column_index, vday in enumerate(vdays):
                cell = [row for row in rows if row["scenario"] == scenario and row["cost_form"] == "linear" and row["alpha_scenario"] == "moderate" and abs(float(row["theta_f"]) - theta) < 1e-9 and abs(float(row["normalized_v_day"]) - vday) < 1e-12 and row["is_winner"].lower() == "true"]
                winner = max(cell, key=lambda row: float(row["winner_frequency_bootstrap"]))
                matrix[row_index, column_index] = index[winner["candidate"]]
        axis.pcolormesh(np.arange(len(vdays) + 1), np.arange(len(thetas) + 1), matrix, cmap=cmap, norm=norm)
        axis.set_xticks(np.arange(len(vdays)) + 0.5, [f"{value:g}" for value in vdays], rotation=45, ha="right")
        axis.set_yticks(np.arange(len(thetas)) + 0.5, [f"{value:g}" for value in thetas])
        axis.set_xlabel(r"$u_{\mathrm{day}}=v_{\mathrm{day}}/C_{L4}$")
        axis.set_title(SCENARIO_LABELS[scenario], fontsize=9)
        reference_row = thetas.index(1.0) if 1.0 in thetas else len(thetas) // 2
        sequence = [CANDIDATES[int(round(matrix[reference_row, column]))] for column in range(len(vdays))]
        run_start = 0
        for column in range(1, len(sequence) + 1):
            if column == len(sequence) or sequence[column] != sequence[run_start]:
                run_label = sequence[run_start]
                if sequence.count(run_label) == column - run_start:
                    annotation = axis.text(run_start + (column - run_start) / 2.0, reference_row + 0.5, run_label, ha="center", va="center", fontsize=7, fontweight="bold", color="black")
                    annotation.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="white")])
                run_start = column
    axes[0].set_ylabel(r"$\theta_F$")
    handles = [Patch(facecolor=COLORS[candidate], edgecolor="white", label=candidate) for candidate in ["L0", "L4", "F_fast"]]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    figure.suptitle("Economic winner map: linear cost, alpha = 0.5", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(figure, "fig5_decision_map")


def figure_sampling_mechanism() -> None:
    rows = read_rows(ROOT / "results" / "e1_independent_sampling_final" / "sampling_mechanism_diagnostics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for axis, scenario in zip(axes, ["evidence_tradeoff", "boundary_tradeoff"]):
        selected = [row for row in rows if row["scenario"] == scenario]
        positions = [0, 1]
        means = np.array([float(row["mean_loss_difference"]) for row in selected])
        lower = np.array([float(row["loss_difference_q025"]) for row in selected])
        upper = np.array([float(row["loss_difference_q975"]) for row in selected])
        sds = np.array([float(row["sd_loss_difference"]) for row in selected])
        axis.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
        axis.errorbar(positions, means, yerr=np.vstack([means - lower, upper - means]), fmt="o", color="#0072B2", ecolor="#0072B2", capsize=4, markersize=5, label="95% interval")
        axis.scatter(positions, means - sds, marker="_", color="#D55E00", s=55)
        axis.scatter(positions, means + sds, marker="_", color="#D55E00", s=55, label="±1 SD")
        axis.set_xticks(positions, ["Paired", "Independent"])
        axis.set_title(SCENARIO_LABELS[scenario], fontsize=9)
        axis.grid(axis="y", color="#EFEFEF")
        axis.set_ylim(float(min(lower)) - 5.0, float(max(upper)) + 6.0)
        axis.set_xlim(-0.45, 1.45)
        correlation_text = "\n".join(f"{name} r = {float(row['loss_correlation']):.3f}" for name, row in zip(["Paired", "Independent"], selected))
        axis.text(0.035, 0.955, correlation_text, transform=axis.transAxes, ha="left", va="top", fontsize=6.5)
    axes[0].set_ylabel("F_fast - L4 LossDays")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    figure.suptitle("Sampling correlation changes candidate-difference variance", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, "fig6_sampling_mechanism")


def main() -> None:
    figure_mean_regret()
    figure_top1_matrix()
    figure_phase_boundary()
    figure_pareto_frontier()
    figure_decision_map()
    figure_sampling_mechanism()
    print(FIG_DIR)


if __name__ == "__main__":
    main()
