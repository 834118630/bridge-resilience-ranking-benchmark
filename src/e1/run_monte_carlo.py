"""Monte Carlo evaluation using HAZUS recovery-time uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy

from .monte_carlo import (
    METRIC_NAMES,
    RECOVERY_SCENARIOS,
    candidate_recovery_times,
    damage_state_probabilities_batch,
    metrics_batch,
    normalized_regret_for_scene,
    ranking_stability_statistics,
    recovery_parameter_arrays,
    sample_recovery_days_matrix,
)
from .run_pilot import (
    CAPACITY_LEVELS,
    CANDIDATES,
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_monte_carlo"
STATISTIC_NAMES = (
    "top1_change_rate",
    "pairwise_reversal_rate",
    "mean_top3_jaccard",
    "mean_kendall_tau",
    "mean_spearman_rho",
)
BOOTSTRAP_SCOPE = "both_cases_stratified_by_case_all_sa_and_metrics"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_beta(path: Path) -> float:
    values = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["beta"]))
    if not values or not np.allclose(values, values[0]):
        raise ValueError("all fragility rows must have one common beta")
    if values[0] <= 0.0:
        raise ValueError("beta must be positive")
    return values[0]


def bootstrap_mean_ci(
    values: list[float],
    strata: list[str] | None,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        raise ValueError("bootstrap values must not be empty")
    if strata is None:
        draws = np.empty(n_resamples, dtype=float)
        for index in range(n_resamples):
            draws[index] = np.mean(rng.choice(array, size=len(array), replace=True))
    else:
        indices = stratified_paired_bootstrap(
            np.arange(len(array)),
            strata,
            n_resamples=n_resamples,
            rng=rng,
        )
        draws = np.asarray([np.mean(array[index]) for index in indices], dtype=float)
    low, high = percentile_ci(draws)
    return float(np.mean(array)), low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--time-window-days", type=float, default=500.0)
    parser.add_argument("--recovery-target", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")

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
    case_probabilities: dict[str, np.ndarray] = {}
    for case_id, (conventional_class, seismic_class) in case_pairs.items():
        conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
        seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
        medians = np.asarray(
            [
                interpolate_medians(conventional, seismic, capacity)
                for capacity in CAPACITY_LEVELS
            ],
            dtype=float,
        )
        case_probabilities[case_id] = damage_state_probabilities_batch(
            medians,
            beta=beta,
            sa_levels=SA_LEVELS,
        )

    base_recovery_days = sample_recovery_days_matrix(recovery, args.samples, rng)
    uncertainty_rows: list[dict[str, object]] = []
    bootstrap_records: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    regret_by_sample_by_scenario = {
        scenario_name: np.zeros((args.samples, len(CANDIDATES)), dtype=float)
        for scenario_name in RECOVERY_SCENARIOS
    }
    scene_regret_means_by_scenario: dict[str, list[np.ndarray]] = {
        scenario_name: [] for scenario_name in RECOVERY_SCENARIOS
    }
    scene_strata_by_scenario: dict[str, list[str]] = {
        scenario_name: [] for scenario_name in RECOVERY_SCENARIOS
    }
    scene_keys_by_scenario: dict[str, list[str]] = {
        scenario_name: [] for scenario_name in RECOVERY_SCENARIOS
    }

    for case_id in case_pairs:
        probabilities = case_probabilities[case_id]
        for scenario_name, multipliers in RECOVERY_SCENARIOS.items():
            recovery_times = candidate_recovery_times(base_recovery_days, multipliers)
            metrics_by_model: dict[str, dict[str, np.ndarray]] = {}
            for model_name in RECOVERY_MODELS:
                metrics_by_model[model_name] = metrics_batch(
                    recovery_times,
                    probabilities,
                    q0,
                    qend,
                    model_name,
                    time_window_days=args.time_window_days,
                    recovery_target=args.recovery_target,
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
                        mc_low, mc_high = percentile_ci(series)
                        mean_value = float(np.mean(series))
                        uncertainty_rows.append(
                            {
                                "case": case_id,
                                "recovery_scenario": scenario_name,
                                "sa": float(sa_value),
                                "metric": metric_name,
                                "statistic": statistic_name,
                                "mean": mean_value,
                                "mc_ci95_low": mc_low,
                                "mc_ci95_high": mc_high,
                                "n_mc_samples": args.samples,
                            }
                        )
                        bootstrap_records[(scenario_name, statistic_name)].append(
                            (case_id, mean_value)
                        )

                for model_name in RECOVERY_MODELS:
                    values = model_arrays[model_name]
                    for sa_index, sa_value in enumerate(SA_LEVELS):
                        scene_values = values[:, :, sa_index]
                        scene_regret = normalized_regret_for_scene(scene_values)
                        regret_by_sample_by_scenario[scenario_name] = np.maximum(
                            regret_by_sample_by_scenario[scenario_name],
                            scene_regret,
                        )
                        scene_regret_means_by_scenario[scenario_name].append(
                            scene_regret.mean(axis=0)
                        )
                        scene_strata_by_scenario[scenario_name].append(case_id)
                        scene_keys_by_scenario[scenario_name].append(
                            f"{case_id}|{scenario_name}|{sa_value:g}|{metric_name}|{model_name}"
                        )

    with (output_dir / "uncertainty_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(uncertainty_rows[0].keys()))
        writer.writeheader()
        writer.writerows(uncertainty_rows)

    bootstrap_rows: list[dict[str, object]] = []
    for (scenario_name, statistic_name), records in sorted(bootstrap_records.items()):
        values = [value for _, value in records]
        strata = [case_id for case_id, _ in records]
        mean_value, low, high = bootstrap_mean_ci(
            values,
            strata,
            args.bootstrap_resamples,
            rng,
        )
        bootstrap_rows.append(
            {
                "scope": BOOTSTRAP_SCOPE,
                "case": "A+B",
                "recovery_scenario": scenario_name,
                "statistic": statistic_name,
                "mean": mean_value,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "n_conditions": len(values),
                "n_resamples": args.bootstrap_resamples,
            }
        )
        for case_id in case_pairs:
            case_values = [value for current_case, value in records if current_case == case_id]
            case_mean, case_low, case_high = bootstrap_mean_ci(
                case_values,
                None,
                args.bootstrap_resamples,
                rng,
            )
            bootstrap_rows.append(
                {
                    "scope": "single_case_all_sa_and_metrics",
                    "case": case_id,
                    "recovery_scenario": scenario_name,
                    "statistic": statistic_name,
                    "mean": case_mean,
                    "bootstrap_ci95_low": case_low,
                    "bootstrap_ci95_high": case_high,
                    "n_conditions": len(case_values),
                    "n_resamples": args.bootstrap_resamples,
                }
            )

    with (output_dir / "bootstrap_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bootstrap_rows)

    regret_summary: dict[str, dict[str, dict[str, object]]] = {}
    minimax_candidate_by_scenario: dict[str, str] = {}
    for scenario_name in RECOVERY_SCENARIOS:
        scene_regret_matrix = np.stack(scene_regret_means_by_scenario[scenario_name], axis=0)
        scene_index = np.arange(scene_regret_matrix.shape[0])
        scene_bootstrap_indices = stratified_paired_bootstrap(
            scene_index,
            scene_strata_by_scenario[scenario_name],
            n_resamples=args.bootstrap_resamples,
            rng=rng,
        )
        bootstrap_regret = np.empty((args.bootstrap_resamples, len(CANDIDATES)), dtype=float)
        bootstrap_mean_regret = np.empty((args.bootstrap_resamples, len(CANDIDATES)), dtype=float)
        for index, selected in enumerate(scene_bootstrap_indices):
            bootstrap_regret[index] = np.max(scene_regret_matrix[selected], axis=0)
            bootstrap_mean_regret[index] = np.mean(scene_regret_matrix[selected], axis=0)

        scenario_summary: dict[str, dict[str, object]] = {}
        sample_regret = regret_by_sample_by_scenario[scenario_name]
        for candidate_index, candidate in enumerate(CANDIDATES):
            sample_values = sample_regret[:, candidate_index]
            mc_low, mc_high = percentile_ci(sample_values)
            worst_scene_index = int(np.argmax(scene_regret_matrix[:, candidate_index]))
            scenario_summary[candidate] = {
                "mean_sample_minimax_regret": float(np.mean(sample_values)),
                "mc_ci95_low": mc_low,
                "mc_ci95_high": mc_high,
                "p95_sample_minimax_regret": float(np.quantile(sample_values, 0.95)),
                "mean_scene_regret": float(np.mean(scene_regret_matrix[:, candidate_index])),
                "mean_regret_bootstrap_ci95_low": float(
                    np.quantile(bootstrap_mean_regret[:, candidate_index], 0.025)
                ),
                "mean_regret_bootstrap_ci95_high": float(
                    np.quantile(bootstrap_mean_regret[:, candidate_index], 0.975)
                ),
                "worst_scene_mean_regret": float(
                    scene_regret_matrix[worst_scene_index, candidate_index]
                ),
                "worst_scene": scene_keys_by_scenario[scenario_name][worst_scene_index],
                "bootstrap_ci95_low": float(
                    np.quantile(bootstrap_regret[:, candidate_index], 0.025)
                ),
                "bootstrap_ci95_high": float(
                    np.quantile(bootstrap_regret[:, candidate_index], 0.975)
                ),
            }
        regret_summary[scenario_name] = scenario_summary
        minimax_candidate_by_scenario[scenario_name] = min(
            scenario_summary,
            key=lambda candidate: float(
                scenario_summary[candidate]["mean_sample_minimax_regret"]
            ),
        )

    with (output_dir / "minimax_regret_uncertainty.json").open("w", encoding="utf-8") as handle:
        json.dump(regret_summary, handle, ensure_ascii=False, indent=2)

    metadata = {
        "seed": args.seed,
        "n_mc_samples": args.samples,
        "bootstrap_resamples": args.bootstrap_resamples,
        "time_window_days": args.time_window_days,
        "recovery_target": args.recovery_target,
        "sa_levels_g": list(SA_LEVELS),
        "damage_states": list(DAMAGE_STATES),
        "candidate_levels": {
            candidate: capacity for candidate, capacity in zip(CANDIDATES, CAPACITY_LEVELS)
        },
        "recovery_models": list(RECOVERY_MODELS),
        "recovery_scenarios": {
            name: list(multipliers) for name, multipliers in RECOVERY_SCENARIOS.items()
        },
        "metrics": list(METRIC_NAMES),
        "ranking_k": 3,
        "minimax_candidate_by_scenario": minimax_candidate_by_scenario,
        "uncertainty_sources": {
            "monte_carlo": "truncated-normal damage-state restoration durations",
            "bootstrap": BOOTSTRAP_SCOPE,
        },
        "input_files": {
            str(fragility_path): sha256_file(fragility_path),
            str(recovery_path): sha256_file(recovery_path),
        },
        "code_files": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str((PROJECT_ROOT / "src" / "e1" / "monte_carlo.py").resolve()): sha256_file(
                (PROJECT_ROOT / "src" / "e1" / "monte_carlo.py").resolve()
            ),
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

    metric_means = np.asarray([row["mean"] for row in uncertainty_rows], dtype=float)
    print(
        json.dumps(
            {
                "samples": args.samples,
                "bootstrap_resamples": args.bootstrap_resamples,
                "conditions": len(uncertainty_rows),
                "output": str(output_dir),
                "max_top1_change_rate_mean": float(
                    max(
                        row["mean"]
                        for row in uncertainty_rows
                        if row["statistic"] == "top1_change_rate"
                    )
                ),
                "max_pairwise_reversal_rate_mean": float(
                    max(
                        row["mean"]
                        for row in uncertainty_rows
                        if row["statistic"] == "pairwise_reversal_rate"
                    )
                ),
                "mean_statistic_overall": float(metric_means.mean()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
