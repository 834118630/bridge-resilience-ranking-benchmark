"""Compare model-conditional, model-robust, and global-best regret definitions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .monte_carlo import (
    METRIC_NAMES,
    candidate_recovery_times,
    damage_state_probabilities_batch,
    metrics_batch,
    normalized_regret_for_scene,
    recovery_parameter_arrays,
    sample_recovery_days_matrix,
)
from .run_monte_carlo import load_beta
from .run_profile_monte_carlo import (
    PROFILE_CAPACITIES,
    PROFILE_LABELS,
    PROFILE_SCENARIOS,
)
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
from .uncertainty import stratified_paired_bootstrap

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_regret_variants"


def bca_interval(
    condition_values: np.ndarray,
    strata: list[str],
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, str]:
    """Return percentile and BCa intervals for the mean of condition values."""

    values = np.asarray(condition_values, dtype=float)
    theta_hat = float(values.mean())
    indices = stratified_paired_bootstrap(
        np.arange(len(values)),
        strata,
        n_resamples=n_resamples,
        rng=rng,
    )
    bootstrap = np.asarray([values[index].mean() for index in indices], dtype=float)
    percentile_low, percentile_high = np.quantile(bootstrap, [0.025, 0.975])

    proportion = float(np.mean(bootstrap < theta_hat))
    proportion = min(max(proportion, 1.0 / (n_resamples + 1)), 1.0 - 1.0 / (n_resamples + 1))
    z0 = float(norm.ppf(proportion))

    jackknife = np.asarray(
        [
            np.delete(values, index).mean()
            for index in range(len(values))
        ],
        dtype=float,
    )
    jack_mean = jackknife.mean()
    deviations = jack_mean - jackknife
    denominator = 6.0 * (np.sum(deviations**2) ** 1.5)
    acceleration = float(np.sum(deviations**3) / denominator) if denominator > 0 else 0.0

    z_low = norm.ppf(0.025)
    z_high = norm.ppf(0.975)
    low_adjust = norm.cdf(
        z0 + (z0 + z_low) / (1.0 - acceleration * (z0 + z_low))
    )
    high_adjust = norm.cdf(
        z0 + (z0 + z_high) / (1.0 - acceleration * (z0 + z_high))
    )
    if not np.isfinite(low_adjust) or not np.isfinite(high_adjust) or low_adjust >= high_adjust:
        return float(percentile_low), float(percentile_high), float(percentile_low), float(percentile_high), "percentile_fallback"
    bca_low, bca_high = np.quantile(bootstrap, [low_adjust, high_adjust])
    return float(percentile_low), float(percentile_high), float(bca_low), float(bca_high), "bca"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fragility_path = DATA_DIR / "07_hazus_fragility.csv"
    recovery_path = DATA_DIR / "08_hazus_recovery.csv"
    fragility = load_fragility(fragility_path)
    recovery = load_recovery(recovery_path)
    beta = load_beta(fragility_path)
    q0, qend = recovery_parameter_arrays(recovery)
    rng = np.random.default_rng(args.seed)
    base_days = sample_recovery_days_matrix(recovery, args.samples, rng)
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}

    probabilities_by_case: dict[str, np.ndarray] = {}
    for case_id, (conventional_class, seismic_class) in case_pairs.items():
        conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
        seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
        medians = np.asarray(
            [interpolate_medians(conventional, seismic, capacity) for capacity in PROFILE_CAPACITIES],
            dtype=float,
        )
        probabilities_by_case[case_id] = damage_state_probabilities_batch(
            medians,
            beta=beta,
            sa_levels=SA_LEVELS,
        )

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    condition_store: dict[str, dict[str, object]] = {}
    for scenario_name, multipliers in PROFILE_SCENARIOS.items():
        mc_condition_means: list[np.ndarray] = []
        condition_strata: list[str] = []
        mc_condition_max = np.zeros(len(PROFILE_LABELS), dtype=float)
        mr_condition_means: list[np.ndarray] = []
        mr_condition_max = np.zeros(len(PROFILE_LABELS), dtype=float)
        gb_condition_means: list[np.ndarray] = []
        gb_condition_max = np.zeros(len(PROFILE_LABELS), dtype=float)

        for case_id, probabilities in probabilities_by_case.items():
            recovery_times = candidate_recovery_times(base_days, multipliers)
            model_metrics = {
                model_name: metrics_batch(
                    recovery_times,
                    probabilities,
                    q0,
                    qend,
                    model_name,
                )
                for model_name in RECOVERY_MODELS
            }

            for metric_name in METRIC_NAMES:
                model_values = {
                    model_name: model_metrics[model_name][metric_name]
                    for model_name in RECOVERY_MODELS
                }
                for sa_index in range(len(SA_LEVELS)):
                    model_regrets = [
                        normalized_regret_for_scene(values[:, :, sa_index])
                        for values in model_values.values()
                    ]
                    model_regret_stack = np.stack(model_regrets, axis=0)
                    mc_condition_means.append(model_regret_stack.mean(axis=(0, 1)))
                    condition_strata.append(case_id)
                    mc_condition_max = np.maximum(
                        mc_condition_max,
                        model_regret_stack.max(axis=(0, 1)),
                    )

                    mr_per_sample = model_regret_stack.max(axis=0)
                    mr_condition_means.append(mr_per_sample.mean(axis=0))
                    mr_condition_max = np.maximum(
                        mr_condition_max,
                        mr_per_sample.max(axis=0),
                    )

                    values_stack = np.stack(
                        [values[:, :, sa_index] for values in model_values.values()],
                        axis=0,
                    )
                    candidate_best = values_stack.max(axis=0)
                    global_best = candidate_best.max(axis=1, keepdims=True)
                    spread = global_best - candidate_best.min(axis=1, keepdims=True)
                    gb_regret = np.divide(
                        global_best - candidate_best,
                        spread,
                        out=np.zeros_like(candidate_best),
                        where=spread > 0,
                    )
                    gb_condition_means.append(gb_regret.mean(axis=0))
                    gb_condition_max = np.maximum(gb_condition_max, gb_regret.max(axis=0))

        mc_mean = np.mean(np.stack(mc_condition_means, axis=0), axis=0)
        mr_mean = np.mean(np.stack(mr_condition_means, axis=0), axis=0)
        gb_mean = np.mean(np.stack(gb_condition_means, axis=0), axis=0)
        for index, candidate in enumerate(PROFILE_LABELS):
            rows.append(
                {
                    "scenario": scenario_name,
                    "candidate": candidate,
                    "model_conditional_mean_regret": float(mc_mean[index]),
                    "model_conditional_minimax_regret": float(mc_condition_max[index]),
                    "model_robust_mean_regret": float(mr_mean[index]),
                    "model_robust_minimax_regret": float(mr_condition_max[index]),
                    "global_best_mean_regret": float(gb_mean[index]),
                    "global_best_minimax_regret": float(gb_condition_max[index]),
                }
            )
        summary[scenario_name] = {
            "best_model_conditional": PROFILE_LABELS[int(np.argmin(mc_mean))],
            "best_model_robust": PROFILE_LABELS[int(np.argmin(mr_mean))],
            "best_global_best": PROFILE_LABELS[int(np.argmin(gb_mean))],
        }
        condition_store[scenario_name] = {
            "model_conditional": np.stack(mc_condition_means, axis=0),
            "model_robust": np.stack(mr_condition_means, axis=0),
            "global_best": np.stack(gb_condition_means, axis=0),
            "strata": condition_strata,
        }

    with (output_dir / "regret_variants.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary.update(
        {
            "samples": args.samples,
            "seed": args.seed,
            "output": str(output_dir),
            "definition_note": "model-conditional is primary; model-robust and global-best are secondary stress checks",
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    comparison_rows = []
    comparison_rng = np.random.default_rng(args.seed + 1)
    for scenario_name, stored in condition_store.items():
        for definition in ("model_conditional", "model_robust", "global_best"):
            matrix = np.asarray(stored[definition], dtype=float)
            strata = list(stored["strata"])
            for candidate_index, candidate in enumerate(PROFILE_LABELS):
                percentile_low, percentile_high, bca_low, bca_high, status = bca_interval(
                    matrix[:, candidate_index],
                    strata,
                    args.bootstrap_resamples,
                    comparison_rng,
                )
                comparison_rows.append(
                    {
                        "scenario": scenario_name,
                        "definition": definition,
                        "candidate": candidate,
                        "percentile_ci95_low": percentile_low,
                        "percentile_ci95_high": percentile_high,
                        "bca_ci95_low": bca_low,
                        "bca_ci95_high": bca_high,
                        "method": status,
                    }
                )
    comparison_path = output_dir / "bootstrap_comparison.csv"
    if comparison_rows:
        with comparison_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
