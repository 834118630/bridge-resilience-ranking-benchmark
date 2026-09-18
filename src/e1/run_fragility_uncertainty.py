"""Stress-test ranking conclusions under candidate-specific fragility uncertainty.

The perturbation ranges are deliberately labelled as stress-test assumptions,
not as HAZUS-estimated uncertainty distributions. They test whether the main
ranking conclusion is sensitive to candidate-specific changes in fragility
medians and beta.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

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
from .uncertainty import stratified_paired_bootstrap

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_fragility_uncertainty"
LABELS = ["L0", "L1", "L2", "L3", "L4", "F_fast"]
CAPACITIES = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5]
SCENARIOS = {
    "shape_only": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "evidence_tradeoff": [1.0, 1.15, 1.30, 1.50, 1.75, 0.10],
}


def bootstrap_condition_mean(
    values: list[tuple[str, np.ndarray]],
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cases = [case_id for case_id, _ in values]
    matrix = np.stack([value for _, value in values], axis=0)
    mean = matrix.mean(axis=0)
    samples = stratified_paired_bootstrap(
        np.arange(len(values)),
        cases,
        n_resamples=n_resamples,
        rng=rng,
    )
    boot = np.stack([matrix[index].mean(axis=0) for index in samples], axis=0)
    return mean, np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragility-scenarios", type=int, default=500)
    parser.add_argument("--recovery-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--beta-low", type=float, default=0.5)
    parser.add_argument("--beta-high", type=float, default=0.7)
    parser.add_argument("--median-sigma", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fragility_path = DATA_DIR / "07_hazus_fragility.csv"
    recovery_path = DATA_DIR / "08_hazus_recovery.csv"
    fragility = load_fragility(fragility_path)
    recovery = load_recovery(recovery_path)
    q0, qend = recovery_parameter_arrays(recovery)
    rng = np.random.default_rng(args.seed)
    base_days = sample_recovery_days_matrix(recovery, args.recovery_samples, rng)
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}

    condition_regrets: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    top1_counts = {
        scenario: {label: 0 for label in LABELS}
        for scenario in SCENARIOS
    }
    pairwise_reversals: dict[str, list[float]] = defaultdict(list)
    f_better_than_l4: dict[str, list[bool]] = defaultdict(list)

    for scenario_name, multipliers in SCENARIOS.items():
        for scenario_index in range(args.fragility_scenarios):
            beta_draw = rng.uniform(args.beta_low, args.beta_high)
            median_factors = rng.lognormal(
                mean=0.0,
                sigma=args.median_sigma,
                size=(len(case_pairs), len(LABELS), len(DAMAGE_STATES)),
            )
            recovery_times = candidate_recovery_times(
                base_days[scenario_index % args.recovery_samples : scenario_index % args.recovery_samples + 1],
                multipliers,
            )

            for case_index, (case_id, (conventional_class, seismic_class)) in enumerate(case_pairs.items()):
                conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
                seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
                base_medians = np.asarray(
                    [interpolate_medians(conventional, seismic, capacity) for capacity in CAPACITIES],
                    dtype=float,
                )
                medians = base_medians * median_factors[case_index]
                probabilities = damage_state_probabilities_batch(
                    medians,
                    beta=beta_draw,
                    sa_levels=SA_LEVELS,
                )

                model_metrics: dict[str, dict[str, np.ndarray]] = {}
                for model_name in RECOVERY_MODELS:
                    model_metrics[model_name] = metrics_batch(
                        recovery_times,
                        probabilities,
                        q0,
                        qend,
                        model_name,
                    )

                for metric_name in METRIC_NAMES:
                    values_by_model = {
                        model: model_metrics[model][metric_name]
                        for model in RECOVERY_MODELS
                    }
                    stats = ranking_stability_statistics(values_by_model, k=3)
                    pairwise_reversals[scenario_name].append(
                        float(stats["pairwise_reversal_rate"].mean())
                    )

                    for model_name, values in values_by_model.items():
                        for sa_index in range(len(SA_LEVELS)):
                            regret = normalized_regret_for_scene(values[:, :, sa_index])[0]
                            condition_regrets[scenario_name].append((case_id, regret))
                            winner = int(np.argmax(values[0, :, sa_index]))
                            top1_counts[scenario_name][LABELS[winner]] += 1
                            f_better_than_l4[scenario_name].append(
                                bool(regret[LABELS.index("F_fast")] < regret[LABELS.index("L4")])
                            )

    candidate_rows: list[dict[str, object]] = []
    for scenario_name, entries in condition_regrets.items():
        mean, low, high = bootstrap_condition_mean(entries, args.bootstrap_resamples, rng)
        total = sum(top1_counts[scenario_name].values())
        for index, label in enumerate(LABELS):
            candidate_rows.append(
                {
                    "scenario": scenario_name,
                    "candidate": label,
                    "mean_regret": float(mean[index]),
                    "bootstrap_ci95_low": float(low[index]),
                    "bootstrap_ci95_high": float(high[index]),
                    "top1_frequency": top1_counts[scenario_name][label] / total,
                }
            )

    ranking_rows = []
    for scenario_name, values in pairwise_reversals.items():
        series = np.asarray(values, dtype=float)
        ranking_rows.append(
            {
                "scenario": scenario_name,
                "statistic": "pairwise_reversal_rate",
                "mean": float(series.mean()),
                "ci95_low": float(np.quantile(series, 0.025)),
                "ci95_high": float(np.quantile(series, 0.975)),
            }
        )

    with (output_dir / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (output_dir / "ranking_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranking_rows)

    summary = {
        "status": "stress_test_not_hazus_estimated_uncertainty",
        "assumptions": {
            "beta_uniform": [args.beta_low, args.beta_high],
            "candidate_state_median_lognormal_sigma": args.median_sigma,
            "fragility_scenarios": args.fragility_scenarios,
            "recovery_samples": args.recovery_samples,
        },
        "f_fast_better_than_l4_fraction": {
            scenario: float(np.mean(values))
            for scenario, values in f_better_than_l4.items()
        },
        "output": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
