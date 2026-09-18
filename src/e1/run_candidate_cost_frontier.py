"""Six-candidate cost-performance frontier for the E1 benchmark.

The script keeps the ranking metric (mean regret) separate from the engineering
metric (discounted loss days).  It evaluates six candidate profiles on a common
cost surface, identifies Pareto-efficient candidates, and maps the economic
winner over a normalized daily-loss-value grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

from .monte_carlo import (
    candidate_recovery_times,
    damage_state_probabilities_batch,
    recovery_parameter_arrays,
    sample_recovery_days_matrix,
)
from .run_cost_break_even import (
    DISCOUNT_RATE,
    L4_MULTIPLIER,
    WINDOW_DAYS,
    discounted_loss_days,
    expected_curves,
)
from .run_monte_carlo import load_beta
from .run_pilot import (
    DAMAGE_STATES,
    DATA_DIR,
    PROJECT_ROOT,
    RECOVERY_MODELS,
    SA_LEVELS,
    interpolate_medians,
    load_fragility,
    load_recovery,
)
from .run_profile_monte_carlo import (
    PROFILE_CAPACITIES,
    PROFILE_LABELS,
    PROFILE_SCENARIOS,
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_candidate_cost_frontier"
COST_FORMS = ("linear", "exponential")
ALPHA_SCENARIOS = {
    "independent": 0.0,
    "moderate": 0.5,
    "strong": 1.0,
}
THETA_F_GRID = (0.5, 0.8, 1.0, 1.25, 1.5, 2.0)
VDAY_GRID = (
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    2e-2,
    3e-2,
    4e-2,
    5e-2,
    6e-2,
    8e-2,
    1e-1,
    1.5e-1,
    2e-1,
)
PARETO_TOLERANCE = 1e-10


def cost_factor(
    cost_form: str,
    alpha: float,
    capacity: float,
    multiplier: float,
) -> float:
    """Return the common cost-surface factor before candidate-specific theta."""

    if cost_form not in COST_FORMS:
        raise KeyError(f"unknown cost form: {cost_form}")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    capacity_term = 1.0 - float(capacity)
    speed_term = L4_MULTIPLIER - float(multiplier)
    if cost_form == "linear":
        factor = 1.0 - alpha * capacity_term + alpha * speed_term
    else:
        factor = float(np.exp(-alpha * capacity_term + alpha * speed_term))
    if factor <= 0.0:
        raise ValueError(f"non-positive cost factor for {cost_form}")
    return 1.0 if capacity == 1.0 and multiplier == L4_MULTIPLIER else factor


def candidate_cost_index(
    cost_form: str,
    alpha: float,
    theta: float,
    capacity: float,
    multiplier: float,
) -> float:
    """Return candidate cost with an explicit candidate-specific multiplier."""

    if theta <= 0.0:
        raise ValueError("theta must be positive")
    return theta * cost_factor(cost_form, alpha, capacity, multiplier)


def candidate_cost_vector(
    cost_form: str,
    alpha: float,
    theta_f: float,
    capacities: np.ndarray,
    multipliers: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Return costs for all candidates, applying theta_f only to F_fast."""

    costs = []
    for label, capacity, multiplier in zip(labels, capacities, multipliers):
        theta = theta_f if label == "F_fast" else 1.0
        costs.append(
            candidate_cost_index(
                cost_form,
                alpha,
                theta=theta,
                capacity=float(capacity),
                multiplier=float(multiplier),
            )
        )
    return np.asarray(costs, dtype=float)


def equivalent_groups(
    costs: np.ndarray,
    losses: np.ndarray,
    tolerance: float = PARETO_TOLERANCE,
) -> list[str]:
    """Assign identical (loss, cost) points to the same equivalent group."""

    groups: list[str] = []
    representatives: list[tuple[float, float]] = []
    for loss, cost in zip(losses, costs):
        for index, (representative_loss, representative_cost) in enumerate(representatives):
            if (
                abs(loss - representative_loss) <= tolerance
                and abs(cost - representative_cost) <= tolerance
            ):
                groups.append(f"G{index + 1}")
                break
        else:
            representatives.append((float(loss), float(cost)))
            groups.append(f"G{len(representatives)}")
    return groups


def pareto_flags(
    costs: np.ndarray,
    losses: np.ndarray,
    tolerance: float = PARETO_TOLERANCE,
) -> np.ndarray:
    """Return boolean Pareto-efficiency flags for minimization of both objectives."""

    costs = np.asarray(costs, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if costs.shape != losses.shape:
        raise ValueError("costs and losses must have the same shape")
    flags = np.ones(costs.shape, dtype=bool)
    for index in range(costs.size):
        dominated = False
        for other in range(costs.size):
            if other == index:
                continue
            no_worse = costs[other] <= costs[index] + tolerance and losses[other] <= losses[index] + tolerance
            strictly_better = (
                costs[other] < costs[index] - tolerance
                or losses[other] < losses[index] - tolerance
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags[index] = not dominated
    return flags


def evaluate_candidate_sample_losses(
    capacities: np.ndarray,
    multipliers: np.ndarray,
    base_days: np.ndarray,
    conventional: list[float],
    seismic: list[float],
    beta: float,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    time_days: np.ndarray,
) -> np.ndarray:
    """Return sample x candidate loss days for one case and recovery model."""

    medians = np.asarray(
        [interpolate_medians(conventional, seismic, float(capacity)) for capacity in capacities],
        dtype=float,
    )
    probabilities = damage_state_probabilities_batch(medians, beta, SA_LEVELS)
    recovery_times = candidate_recovery_times(base_days, multipliers)
    curves = expected_curves(
        time_days,
        recovery_times,
        probabilities,
        q0,
        qend,
        model_name,
    )
    losses = discounted_loss_days(time_days, curves, DISCOUNT_RATE, WINDOW_DAYS)
    return losses.mean(axis=2)


def bootstrap_loss_means(
    sample_losses: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap candidate mean loss days over paired recovery samples."""

    if sample_losses.ndim != 2:
        raise ValueError("sample_losses must have shape (samples, candidates)")
    indices = rng.integers(
        0,
        sample_losses.shape[0],
        size=(n_resamples, sample_losses.shape[0]),
    )
    return sample_losses[indices].mean(axis=1)


def load_mean_regret(path: Path) -> dict[str, dict[str, float]]:
    """Load the existing profile-MC mean-regret summary when available."""

    result: dict[str, dict[str, float]] = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["scenario"], {})[row["candidate"]] = float(row["mean_regret"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--time-step-days", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.samples <= 0 or args.bootstrap_resamples <= 0:
        raise ValueError("sample and bootstrap counts must be positive")

    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fragility_path = DATA_DIR / "07_hazus_fragility.csv"
    recovery_path = DATA_DIR / "08_hazus_recovery.csv"
    fragility = load_fragility(fragility_path)
    recovery = load_recovery(recovery_path)
    beta = load_beta(fragility_path)
    q0, qend = recovery_parameter_arrays(recovery)
    rng = np.random.default_rng(args.seed)
    base_days = sample_recovery_days_matrix(recovery, args.samples, rng)
    time_days = np.arange(0.0, 730.0 + args.time_step_days, args.time_step_days)
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}
    capacities = np.asarray(PROFILE_CAPACITIES, dtype=float)

    scenario_loss_arrays: dict[str, np.ndarray] = {}
    loss_rows: list[dict[str, object]] = []
    for scenario_name, multipliers_list in PROFILE_SCENARIOS.items():
        multipliers = np.asarray(multipliers_list, dtype=float)
        condition_losses: list[np.ndarray] = []
        for case_id, (conventional_class, seismic_class) in case_pairs.items():
            conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
            seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
            for model_name in RECOVERY_MODELS:
                condition_losses.append(
                    evaluate_candidate_sample_losses(
                        capacities,
                        multipliers,
                        base_days,
                        conventional,
                        seismic,
                        beta,
                        q0,
                        qend,
                        model_name,
                        time_days,
                    )
                )
        sample_losses = np.mean(np.stack(condition_losses, axis=0), axis=0)
        scenario_loss_arrays[scenario_name] = sample_losses
        mean_losses = sample_losses.mean(axis=0)
        bootstrap_losses = bootstrap_loss_means(sample_losses, args.bootstrap_resamples, rng)
        for index, candidate in enumerate(PROFILE_LABELS):
            loss_rows.append(
                {
                    "scenario": scenario_name,
                    "candidate": candidate,
                    "mean_loss_days": float(mean_losses[index]),
                    "bootstrap_ci95_low": float(np.quantile(bootstrap_losses[:, index], 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(bootstrap_losses[:, index], 0.975)),
                }
            )

    with (output_dir / "candidate_loss_days.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(loss_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loss_rows)

    frontier_rows: list[dict[str, object]] = []
    utility_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "seed": args.seed,
        "samples": args.samples,
        "bootstrap_resamples": args.bootstrap_resamples,
        "cost_forms": list(COST_FORMS),
        "alpha_scenarios": ALPHA_SCENARIOS,
        "theta_f_grid": list(THETA_F_GRID),
        "vday_grid": list(VDAY_GRID),
        "scenarios": {},
    }

    for scenario_name, multipliers_list in PROFILE_SCENARIOS.items():
        multipliers = np.asarray(multipliers_list, dtype=float)
        mean_losses = scenario_loss_arrays[scenario_name].mean(axis=0)
        bootstrap_losses = bootstrap_loss_means(
            scenario_loss_arrays[scenario_name],
            args.bootstrap_resamples,
            rng,
        )
        scenario_summary: dict[str, object] = {
            "mean_loss_days": {
                candidate: float(mean_losses[index])
                for index, candidate in enumerate(PROFILE_LABELS)
            },
            "cost_forms": {},
        }
        for cost_form in COST_FORMS:
            form_summary: dict[str, object] = {}
            for alpha_name, alpha in ALPHA_SCENARIOS.items():
                alpha_summary: dict[str, object] = {}
                for theta_f in THETA_F_GRID:
                    costs = candidate_cost_vector(
                        cost_form,
                        alpha,
                        theta_f,
                        capacities,
                        multipliers,
                        PROFILE_LABELS,
                    )
                    frontier = pareto_flags(costs, mean_losses)
                    groups = equivalent_groups(costs, mean_losses)
                    actionable_indices = [
                        index
                        for index, candidate in enumerate(PROFILE_LABELS)
                        if candidate != "L0"
                    ]
                    actionable_frontier = np.zeros(len(PROFILE_LABELS), dtype=bool)
                    actionable_frontier[actionable_indices] = pareto_flags(
                        costs[actionable_indices],
                        mean_losses[actionable_indices],
                    )
                    for index, candidate in enumerate(PROFILE_LABELS):
                        frontier_rows.append(
                            {
                                "scenario": scenario_name,
                                "cost_form": cost_form,
                                "alpha_scenario": alpha_name,
                                "alpha": alpha,
                                "theta_f": theta_f,
                                "candidate": candidate,
                                "capacity": float(capacities[index]),
                                "speed_multiplier": float(multipliers[index]),
                                "mean_loss_days": float(mean_losses[index]),
                                "cost_index": float(costs[index]),
                                "pareto_frontier": bool(frontier[index]),
                                "engineering_action_candidate": candidate != "L0",
                                "pareto_actionable": bool(actionable_frontier[index]),
                                "equivalent_group": groups[index],
                            }
                        )

                    winner_by_vday: dict[str, object] = {}
                    for vday in VDAY_GRID:
                        utility = costs[None, :] + vday * bootstrap_losses
                        winners = np.argmin(utility, axis=1)
                        winner_frequencies = np.bincount(
                            winners,
                            minlength=len(PROFILE_LABELS),
                        ).astype(float) / len(winners)
                        winner_index = int(np.argmin(costs + vday * mean_losses))
                        winner_by_vday[f"{vday:.4g}"] = {
                            "winner": PROFILE_LABELS[winner_index],
                            "winner_frequency": {
                                candidate: float(winner_frequencies[index])
                                for index, candidate in enumerate(PROFILE_LABELS)
                            },
                        }
                        for index, candidate in enumerate(PROFILE_LABELS):
                            total_utility = float(costs[index] + vday * mean_losses[index])
                            utility_rows.append(
                                {
                                    "scenario": scenario_name,
                                    "cost_form": cost_form,
                                    "alpha_scenario": alpha_name,
                                    "alpha": alpha,
                                    "theta_f": theta_f,
                                    "normalized_v_day": vday,
                                    "candidate": candidate,
                                    "cost_index": float(costs[index]),
                                    "mean_loss_days": float(mean_losses[index]),
                                    "total_utility": total_utility,
                                    "net_benefit_vs_l4": float(
                                        (costs[PROFILE_LABELS.index("L4")] + vday * mean_losses[PROFILE_LABELS.index("L4")])
                                        - total_utility
                                    ),
                                    "is_winner": index == winner_index,
                                    "winner_frequency_bootstrap": float(winner_frequencies[index]),
                                }
                            )

                    alpha_summary[f"{theta_f:.2f}"] = {
                        "costs": {
                            candidate: float(costs[index])
                            for index, candidate in enumerate(PROFILE_LABELS)
                        },
                        "pareto_members": [
                            PROFILE_LABELS[index]
                            for index, flag in enumerate(frontier)
                            if flag
                        ],
                        "pareto_members_actionable": [
                            PROFILE_LABELS[index]
                            for index, flag in enumerate(actionable_frontier)
                            if flag
                        ],
                        "equivalent_groups": groups,
                        "winner_by_vday": winner_by_vday,
                    }
                form_summary[alpha_name] = alpha_summary
            scenario_summary["cost_forms"][cost_form] = form_summary
        summary["scenarios"][scenario_name] = scenario_summary

    with (output_dir / "candidate_frontier.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frontier_rows)

    with (output_dir / "economic_utility.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(utility_rows[0].keys()))
        writer.writeheader()
        writer.writerows(utility_rows)

    with (output_dir / "frontier_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    regret_path = PROJECT_ROOT / "results" / "e1_profile_monte_carlo" / "candidate_summary.csv"
    mean_regret = load_mean_regret(regret_path)
    concordance_rows: list[dict[str, object]] = []
    for scenario_name in PROFILE_SCENARIOS:
        if scenario_name not in mean_regret:
            continue
        regret = np.asarray([mean_regret[scenario_name][candidate] for candidate in PROFILE_LABELS])
        losses = scenario_loss_arrays[scenario_name].mean(axis=0)
        top1_regret = PROFILE_LABELS[int(np.argmin(regret))]
        top1_loss = PROFILE_LABELS[int(np.argmin(losses))]
        concordance_rows.append(
            {
                "scenario": scenario_name,
                "kendall_tau": float(kendalltau(regret, losses).statistic),
                "spearman_rho": float(spearmanr(regret, losses).statistic),
                "top1_mean_regret": top1_regret,
                "top1_loss_days": top1_loss,
                "top1_same": top1_regret == top1_loss,
            }
        )
    if concordance_rows:
        with (output_dir / "rank_concordance.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(concordance_rows[0].keys()))
            writer.writeheader()
            writer.writerows(concordance_rows)

    # Compact visualization for the primary linear/moderate scenario.
    for scenario_name, multipliers_list in PROFILE_SCENARIOS.items():
        multipliers = np.asarray(multipliers_list, dtype=float)
        mean_losses = scenario_loss_arrays[scenario_name].mean(axis=0)
        costs = candidate_cost_vector(
            "linear",
            0.5,
            1.0,
            capacities,
            multipliers,
            PROFILE_LABELS,
        )
        frontier = pareto_flags(costs, mean_losses)
        actionable_indices = [
            index for index, candidate in enumerate(PROFILE_LABELS) if candidate != "L0"
        ]
        actionable_frontier = np.zeros(len(PROFILE_LABELS), dtype=bool)
        actionable_frontier[actionable_indices] = pareto_flags(
            costs[actionable_indices],
            mean_losses[actionable_indices],
        )
        dominated = np.logical_not(frontier)
        math_only = np.logical_and(frontier, np.asarray([candidate == "L0" for candidate in PROFILE_LABELS]))
        figure, axis = plt.subplots(figsize=(7.2, 4.9))
        if np.any(dominated):
            axis.scatter(
                costs[dominated],
                mean_losses[dominated],
                s=52,
                color="tab:blue",
                marker="o",
                label="Dominated candidates",
            )
        if np.any(math_only):
            axis.scatter(
                costs[math_only],
                mean_losses[math_only],
                s=64,
                facecolors="white",
                edgecolors="tab:gray",
                marker="s",
                linewidths=1.5,
                label="Mathematical frontier: non-action baseline (L0)",
            )
        if np.any(actionable_frontier):
            axis.scatter(
                costs[actionable_frontier],
                mean_losses[actionable_frontier],
                s=58,
                color="tab:green",
                marker="D",
                label="Engineering action frontier",
            )
        for index, candidate in enumerate(PROFILE_LABELS):
            axis.annotate(candidate, (costs[index], mean_losses[index]), xytext=(4, 4), textcoords="offset points")
        axis.set_xlabel("Normalized cost index (L4 = 1)")
        axis.set_ylabel("Discounted loss days")
        axis.set_title(f"{scenario_name}: linear cost, alpha=0.5, theta_F=1.0")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="best")
        figure.tight_layout()
        figure.savefig(figure_dir / f"{scenario_name}_linear_moderate_frontier.png", dpi=220)
        plt.close(figure)

    print(
        json.dumps(
            {
                "output": str(output_dir),
                "scenarios": list(PROFILE_SCENARIOS),
                "cost_forms": list(COST_FORMS),
                "theta_f_grid": list(THETA_F_GRID),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
