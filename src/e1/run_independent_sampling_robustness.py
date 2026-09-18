"""Paired-versus-independent recovery-duration robustness for E1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .monte_carlo import (
    METRIC_NAMES,
    candidate_recovery_times,
    damage_state_probabilities_batch,
    metrics_batch,
    normalized_regret_for_scene,
    recovery_parameter_arrays,
    sample_recovery_days_matrix,
)
from .run_candidate_cost_frontier import (
    candidate_cost_vector,
    pareto_flags,
)
from .run_cost_break_even import (
    DISCOUNT_RATE,
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_independent_sampling"


def independent_recovery_base(
    recovery: dict[str, dict[str, float]],
    n_samples: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample independent damage-state durations for every candidate."""

    return np.stack(
        [sample_recovery_days_matrix(recovery, n_samples, rng) for _ in range(n_candidates)],
        axis=1,
    )


def recovery_times_for_mode(
    base_days: np.ndarray,
    multipliers: np.ndarray,
    sampling_mode: str,
) -> np.ndarray:
    """Build candidate recovery times for paired or independent sampling."""

    if sampling_mode == "paired":
        return candidate_recovery_times(base_days, multipliers)
    if sampling_mode == "independent":
        if base_days.ndim != 3:
            raise ValueError("independent base days must have shape (samples, candidates, states)")
        return base_days * multipliers[None, :, None]
    raise KeyError(f"unknown sampling mode: {sampling_mode}")


def condition_losses(
    recovery_times: np.ndarray,
    probabilities: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    time_days: np.ndarray,
) -> np.ndarray:
    """Return sample x candidate discounted loss days for one condition."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--time-step-days", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("samples must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fragility_path = DATA_DIR / "07_hazus_fragility.csv"
    recovery_path = DATA_DIR / "08_hazus_recovery.csv"
    fragility = load_fragility(fragility_path)
    recovery = load_recovery(recovery_path)
    beta = load_beta(fragility_path)
    q0, qend = recovery_parameter_arrays(recovery)
    time_days = np.arange(0.0, 730.0 + args.time_step_days, args.time_step_days)
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}
    capacities = np.asarray(PROFILE_CAPACITIES, dtype=float)

    probabilities_by_case: dict[str, np.ndarray] = {}
    for case_id, (conventional_class, seismic_class) in case_pairs.items():
        conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
        seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
        medians = np.asarray(
            [interpolate_medians(conventional, seismic, capacity) for capacity in capacities],
            dtype=float,
        )
        probabilities_by_case[case_id] = damage_state_probabilities_batch(
            medians,
            beta,
            SA_LEVELS,
        )

    paired_rng = np.random.default_rng(args.seed)
    independent_rng = np.random.default_rng(args.seed + 1)
    paired_base = sample_recovery_days_matrix(recovery, args.samples, paired_rng)
    independent_base = independent_recovery_base(
        recovery,
        args.samples,
        len(PROFILE_LABELS),
        independent_rng,
    )

    metrics_rows: list[dict[str, object]] = []
    sample_losses_by_mode: dict[tuple[str, str], np.ndarray] = {}
    sample_regret_by_mode: dict[tuple[str, str], np.ndarray] = {}
    mean_regret_by_mode: dict[tuple[str, str], np.ndarray] = {}
    top1_by_mode: dict[tuple[str, str], np.ndarray] = {}

    for sampling_mode in ("paired", "independent"):
        base_days = paired_base if sampling_mode == "paired" else independent_base
        for scenario_name, multipliers_list in PROFILE_SCENARIOS.items():
            multipliers = np.asarray(multipliers_list, dtype=float)
            recovery_times = recovery_times_for_mode(base_days, multipliers, sampling_mode)
            regret_conditions: list[np.ndarray] = []
            loss_conditions: list[np.ndarray] = []
            for case_id, probabilities in probabilities_by_case.items():
                for model_name in RECOVERY_MODELS:
                    metrics = metrics_batch(
                        recovery_times,
                        probabilities,
                        q0,
                        qend,
                        model_name,
                    )
                    for metric_name in METRIC_NAMES:
                        for sa_index in range(len(SA_LEVELS)):
                            regret_conditions.append(
                                normalized_regret_for_scene(metrics[metric_name][:, :, sa_index])
                            )
                    loss_conditions.append(
                        condition_losses(
                            recovery_times,
                            probabilities,
                            q0,
                            qend,
                            model_name,
                            time_days,
                        )
                    )

            sample_regret = np.mean(np.stack(regret_conditions, axis=0), axis=0)
            mean_regret = sample_regret.mean(axis=0)
            winners = np.argmin(sample_regret, axis=1)
            top1 = np.bincount(winners, minlength=len(PROFILE_LABELS)).astype(float) / len(winners)
            sample_losses = np.mean(np.stack(loss_conditions, axis=0), axis=0)
            mean_losses = sample_losses.mean(axis=0)
            sample_losses_by_mode[(sampling_mode, scenario_name)] = sample_losses
            sample_regret_by_mode[(sampling_mode, scenario_name)] = sample_regret
            mean_regret_by_mode[(sampling_mode, scenario_name)] = mean_regret
            top1_by_mode[(sampling_mode, scenario_name)] = top1

            for index, candidate in enumerate(PROFILE_LABELS):
                metrics_rows.append(
                    {
                        "sampling_mode": sampling_mode,
                        "scenario": scenario_name,
                        "candidate": candidate,
                        "mean_regret": float(mean_regret[index]),
                        "top1_frequency_mean_regret": float(top1[index]),
                        "mean_loss_days": float(mean_losses[index]),
                    }
                )

    with (output_dir / "sampling_candidate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    comparison_rows: list[dict[str, object]] = []
    for scenario_name, multipliers_list in PROFILE_SCENARIOS.items():
        multipliers = np.asarray(multipliers_list, dtype=float)
        paired_loss = sample_losses_by_mode[("paired", scenario_name)].mean(axis=0)
        independent_loss = sample_losses_by_mode[("independent", scenario_name)].mean(axis=0)
        costs = candidate_cost_vector(
            "linear",
            0.5,
            1.0,
            capacities,
            multipliers,
            PROFILE_LABELS,
        )
        paired_frontier = pareto_flags(costs, paired_loss)
        independent_frontier = pareto_flags(costs, independent_loss)
        paired_top1 = top1_by_mode[("paired", scenario_name)]
        independent_top1 = top1_by_mode[("independent", scenario_name)]
        paired_winner = PROFILE_LABELS[int(np.argmin(mean_regret_by_mode[("paired", scenario_name)]))]
        independent_winner = PROFILE_LABELS[int(np.argmin(mean_regret_by_mode[("independent", scenario_name)]))]
        f_index = PROFILE_LABELS.index("F_fast")
        delta_top1 = float(paired_top1[f_index] - independent_top1[f_index])
        if scenario_name == "shape_only":
            gate = "not_applicable_shape_only"
        elif paired_winner != independent_winner:
            gate = "re-evaluate"
        elif independent_top1[f_index] < 0.60 or delta_top1 > 0.15:
            gate = "downgrade"
        else:
            gate = "keep"
        comparison_rows.append(
            {
                "scenario": scenario_name,
                "paired_top1_winner": paired_winner,
                "independent_top1_winner": independent_winner,
                "paired_fast_top1": float(paired_top1[f_index]),
                "independent_fast_top1": float(independent_top1[f_index]),
                "fast_top1_delta": delta_top1,
                "paired_pareto_members": ",".join(
                    PROFILE_LABELS[index] for index, flag in enumerate(paired_frontier) if flag
                ),
                "independent_pareto_members": ",".join(
                    PROFILE_LABELS[index] for index, flag in enumerate(independent_frontier) if flag
                ),
                "gate_status": gate,
            }
        )

    with (output_dir / "sampling_gate_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    diagnostics_rows: list[dict[str, object]] = []
    f_index = PROFILE_LABELS.index("F_fast")
    l4_index = PROFILE_LABELS.index("L4")
    for scenario_name in PROFILE_SCENARIOS:
        for sampling_mode in ("paired", "independent"):
            loss_diff = (
                sample_losses_by_mode[(sampling_mode, scenario_name)][:, f_index]
                - sample_losses_by_mode[(sampling_mode, scenario_name)][:, l4_index]
            )
            regret_diff = (
                sample_regret_by_mode[(sampling_mode, scenario_name)][:, f_index]
                - sample_regret_by_mode[(sampling_mode, scenario_name)][:, l4_index]
            )
            diagnostics_rows.append(
                {
                    "scenario": scenario_name,
                    "sampling_mode": sampling_mode,
                    "pair": "F_fast-L4",
                    "mean_loss_difference": float(loss_diff.mean()),
                    "sd_loss_difference": float(loss_diff.std(ddof=1)),
                    "loss_difference_q025": float(np.quantile(loss_diff, 0.025)),
                    "loss_difference_q975": float(np.quantile(loss_diff, 0.975)),
                    "loss_correlation": float(
                        np.corrcoef(
                            sample_losses_by_mode[(sampling_mode, scenario_name)][:, f_index],
                            sample_losses_by_mode[(sampling_mode, scenario_name)][:, l4_index],
                        )[0, 1]
                    ),
                    "mean_regret_difference": float(regret_diff.mean()),
                    "sd_regret_difference": float(regret_diff.std(ddof=1)),
                    "regret_difference_q025": float(np.quantile(regret_diff, 0.025)),
                    "regret_difference_q975": float(np.quantile(regret_diff, 0.975)),
                    "regret_correlation": float(
                        np.corrcoef(
                            sample_regret_by_mode[(sampling_mode, scenario_name)][:, f_index],
                            sample_regret_by_mode[(sampling_mode, scenario_name)][:, l4_index],
                        )[0, 1]
                    ),
                }
            )

    with (output_dir / "sampling_mechanism_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics_rows)

    summary = {
        "seed": args.seed,
        "samples": args.samples,
        "sampling_modes": ["paired", "independent"],
        "comparison": comparison_rows,
        "mechanism_diagnostics": diagnostics_rows,
    }
    with (output_dir / "sampling_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
