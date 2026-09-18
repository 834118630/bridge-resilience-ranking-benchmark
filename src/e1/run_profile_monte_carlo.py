"""Monte Carlo evaluation with an evidence-anchored fast-recovery profile."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy

from .monte_carlo import (
    METRIC_NAMES,
    candidate_recovery_times,
    damage_state_probabilities_batch,
    metrics_batch,
    normalized_regret_for_scene,
    ranking_stability_statistics,
    recovery_parameter_arrays,
    sample_recovery_days_matrix,
)
from .run_monte_carlo import load_beta, sha256_file
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
from .uncertainty import percentile_ci, stratified_paired_bootstrap

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_profile_monte_carlo"
PROFILE_LABELS = ["L0", "L1", "L2", "L3", "L4", "F_fast"]
PROFILE_CAPACITIES = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5]
PROFILE_SCENARIOS = {
    "shape_only": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "evidence_tradeoff": [1.0, 1.15, 1.30, 1.50, 1.75, 0.10],
    "boundary_tradeoff": [1.0, 1.15, 1.30, 1.50, 1.75, 0.50],
}
STATISTIC_NAMES = (
    "top1_change_rate",
    "pairwise_reversal_rate",
    "mean_top3_jaccard",
    "mean_kendall_tau",
    "mean_spearman_rho",
)


def aggregate_condition_means(
    condition_values: list[tuple[str, np.ndarray]],
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average condition-level vectors and bootstrap over conditions by case."""

    cases = [case_id for case_id, _ in condition_values]
    values = np.stack([value for _, value in condition_values], axis=0)
    mean = values.mean(axis=0)
    strata = np.asarray(cases)
    resamples = stratified_paired_bootstrap(
        np.arange(len(values)),
        strata,
        n_resamples=n_resamples,
        rng=rng,
    )
    bootstrap = np.stack([values[index].mean(axis=0) for index in resamples], axis=0)
    low = np.quantile(bootstrap, 0.025, axis=0)
    high = np.quantile(bootstrap, 0.975, axis=0)
    return mean, low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.samples <= 0 or args.bootstrap_resamples <= 0:
        raise ValueError("sample and bootstrap counts must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fragility_path = DATA_DIR / "07_hazus_fragility.csv"
    recovery_path = DATA_DIR / "08_hazus_recovery.csv"
    fragility = load_fragility(fragility_path)
    recovery = load_recovery(recovery_path)
    beta = load_beta(fragility_path)
    q0, qend = recovery_parameter_arrays(recovery)
    rng = np.random.default_rng(args.seed)
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

    base_recovery_days = sample_recovery_days_matrix(recovery, args.samples, rng)
    ranking_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    minimax_by_scenario: dict[str, dict[str, dict[str, float]]] = {}

    for scenario_name, multipliers in PROFILE_SCENARIOS.items():
        recovery_times = candidate_recovery_times(base_recovery_days, multipliers)
        condition_means: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        top1_counts = {
            metric_name: {
                candidate: 0
                for candidate in PROFILE_LABELS
            }
            for metric_name in METRIC_NAMES
        }
        total_top1_draws = {metric_name: 0 for metric_name in METRIC_NAMES}

        for case_id, probabilities in probabilities_by_case.items():
            metrics_by_model: dict[str, dict[str, np.ndarray]] = {}
            for model_name in RECOVERY_MODELS:
                metrics_by_model[model_name] = metrics_batch(
                    recovery_times,
                    probabilities,
                    q0,
                    qend,
                    model_name,
                )

            for metric_name in METRIC_NAMES:
                model_arrays = {
                    model_name: metrics_by_model[model_name][metric_name]
                    for model_name in RECOVERY_MODELS
                }
                statistics = ranking_stability_statistics(model_arrays, k=3)
                for statistic_name in STATISTIC_NAMES:
                    statistic_array = statistics[statistic_name]
                    for sa_index, sa_value in enumerate(SA_LEVELS):
                        series = statistic_array[:, sa_index]
                        low, high = percentile_ci(series)
                        ranking_rows.append(
                            {
                                "scenario": scenario_name,
                                "case": case_id,
                                "sa": float(sa_value),
                                "metric": metric_name,
                                "statistic": statistic_name,
                                "mean": float(np.mean(series)),
                                "mc_ci95_low": low,
                                "mc_ci95_high": high,
                                "n_samples": args.samples,
                            }
                        )

                for model_name in RECOVERY_MODELS:
                    values = model_arrays[model_name]
                    for sa_index in range(len(SA_LEVELS)):
                        scene_values = values[:, :, sa_index]
                        winners = np.argmax(scene_values, axis=1)
                        for candidate_index, candidate in enumerate(PROFILE_LABELS):
                            top1_counts[metric_name][candidate] += int(
                                np.sum(winners == candidate_index)
                            )
                        total_top1_draws[metric_name] += len(winners)

                for model_name in RECOVERY_MODELS:
                    values = model_arrays[model_name]
                    for sa_index in range(len(SA_LEVELS)):
                        condition_means[case_id].append(
                            (
                                case_id,
                                normalized_regret_for_scene(
                                    values[:, :, sa_index]
                                )[0],
                            )
                        )

        condition_entries = [
            entry for entries in condition_means.values() for entry in entries
        ]
        all_condition_regrets = [value for _, value in condition_entries]
        condition_matrix = np.stack(all_condition_regrets, axis=0)
        condition_bootstrap_indices = stratified_paired_bootstrap(
            np.arange(condition_matrix.shape[0]),
            [case_id for case_id, _ in condition_entries],
            n_resamples=args.bootstrap_resamples,
            rng=rng,
        )
        bootstrap_minimax = np.stack(
            [condition_matrix[index].max(axis=0) for index in condition_bootstrap_indices],
            axis=0,
        )
        condition_minimax = condition_matrix.max(axis=0)
        scenario_minimax = {
            candidate: {
                "condition_minimax_regret": float(condition_minimax[index]),
                "bootstrap_ci95_low": float(
                    np.quantile(bootstrap_minimax[:, index], 0.025)
                ),
                "bootstrap_ci95_high": float(
                    np.quantile(bootstrap_minimax[:, index], 0.975)
                ),
            }
            for index, candidate in enumerate(PROFILE_LABELS)
        }

        mean_regret_by_candidate, low_by_candidate, high_by_candidate = aggregate_condition_means(
            condition_entries,
            args.bootstrap_resamples,
            rng,
        )
        for index, candidate in enumerate(PROFILE_LABELS):
            candidate_rows.append(
                {
                    "scenario": scenario_name,
                    "candidate": candidate,
                    "mean_regret": float(mean_regret_by_candidate[index]),
                    "bootstrap_ci95_low": float(low_by_candidate[index]),
                    "bootstrap_ci95_high": float(high_by_candidate[index]),
                    "top1_frequency": float(
                        sum(
                            top1_counts[metric_name][candidate]
                            for metric_name in METRIC_NAMES
                        )
                        / sum(total_top1_draws.values())
                    ),
                }
            )
        minimax_by_scenario[scenario_name] = scenario_minimax

    with (output_dir / "ranking_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranking_rows)

    with (output_dir / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)

    with (output_dir / "minimax_regret.json").open("w", encoding="utf-8") as handle:
        json.dump(minimax_by_scenario, handle, ensure_ascii=False, indent=2)

    metadata = {
        "seed": args.seed,
        "n_samples": args.samples,
        "bootstrap_resamples": args.bootstrap_resamples,
        "profile_labels": PROFILE_LABELS,
        "profile_capacities": PROFILE_CAPACITIES,
        "profile_scenarios": PROFILE_SCENARIOS,
        "recovery_models": list(RECOVERY_MODELS),
        "metrics": list(METRIC_NAMES),
        "input_hashes": {
            str(fragility_path): sha256_file(fragility_path),
            str(recovery_path): sha256_file(recovery_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "samples": args.samples,
                "bootstrap_resamples": args.bootstrap_resamples,
                "scenarios": list(PROFILE_SCENARIOS),
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
