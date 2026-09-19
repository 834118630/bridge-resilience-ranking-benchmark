"""Recovery-duration distribution sensitivity for the E1 ranking analysis.

The primary analysis uses a lower-truncated normal distribution with a 0.05-day
bound. This supporting analysis checks the ranking conclusion against nearby
bounds and a truncated lognormal alternative. It is a robustness analysis, not
a new primary model.
"""

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
    fractional_top1_credit,
    metrics_batch,
    normalized_regret_for_scene,
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_recovery_distribution_sensitivity"
PROFILE_LABELS = ["L0", "L1", "L2", "L3", "L4", "F_fast"]
PROFILE_CAPACITIES = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5]
PROFILE_SCENARIOS = {
    "evidence_tradeoff": [1.0, 1.15, 1.30, 1.50, 1.75, 0.10],
    "boundary_tradeoff": [1.0, 1.15, 1.30, 1.50, 1.75, 0.50],
}
VARIANTS = (
    {"name": "truncnorm_bound_0.01", "distribution": "truncnorm", "lower_bound_days": 0.01},
    {"name": "truncnorm_bound_0.05_baseline", "distribution": "truncnorm", "lower_bound_days": 0.05},
    {"name": "truncnorm_bound_0.10", "distribution": "truncnorm", "lower_bound_days": 0.10},
    {"name": "truncated_lognormal_bound_0.05", "distribution": "lognormal", "lower_bound_days": 0.05},
)


def summarize(
    recovery_times: np.ndarray,
    probabilities_by_case: dict[str, np.ndarray],
    q0: np.ndarray,
    qend: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    condition_values: list[np.ndarray] = []
    top1_credit = np.zeros(len(PROFILE_LABELS), dtype=float)
    draws = 0
    for probabilities in probabilities_by_case.values():
        for model_name in RECOVERY_MODELS:
            batch = metrics_batch(recovery_times, probabilities, q0, qend, model_name)
            for metric_name in METRIC_NAMES:
                values = batch[metric_name]
                for sa_index in range(len(SA_LEVELS)):
                    scene_values = values[:, :, sa_index]
                    condition_values.append(
                        normalized_regret_for_scene(scene_values).mean(axis=0)
                    )
                    top1_credit += fractional_top1_credit(scene_values).sum(axis=0)
                    draws += scene_values.shape[0]
    mean_regret = np.stack(condition_values, axis=0).mean(axis=0)
    top1_frequency = {
        candidate: float(top1_credit[index] / draws)
        for index, candidate in enumerate(PROFILE_LABELS)
    }
    return mean_regret, top1_frequency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260914)
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
    summaries: dict[str, object] = {}
    for variant_index, variant in enumerate(VARIANTS):
        rng = np.random.default_rng(args.seed + 1000 * variant_index)
        base_days = sample_recovery_days_matrix(
            recovery,
            args.samples,
            rng,
            lower_bound_days=float(variant["lower_bound_days"]),
            distribution=str(variant["distribution"]),
        )
        variant_key = str(variant["name"])
        summaries[variant_key] = {
            "distribution": variant["distribution"],
            "lower_bound_days": variant["lower_bound_days"],
            "scenarios": {},
        }
        for scenario_name, multipliers in PROFILE_SCENARIOS.items():
            recovery_times = candidate_recovery_times(base_days, multipliers)
            mean_regret, top1 = summarize(
                recovery_times,
                probabilities_by_case,
                q0,
                qend,
            )
            winner = PROFILE_LABELS[int(np.argmin(mean_regret))]
            summaries[variant_key]["scenarios"][scenario_name] = {
                "top1_mean_regret_candidate": winner,
                "mean_regret": {
                    label: float(mean_regret[index])
                    for index, label in enumerate(PROFILE_LABELS)
                },
                "metric_aggregated_top1_frequency": top1,
            }
            for index, candidate in enumerate(PROFILE_LABELS):
                rows.append(
                    {
                        "variant": variant_key,
                        "distribution": variant["distribution"],
                        "lower_bound_days": variant["lower_bound_days"],
                        "scenario": scenario_name,
                        "candidate": candidate,
                        "mean_regret": float(mean_regret[index]),
                        "metric_aggregated_top1_frequency": top1[candidate],
                    }
                )

    with (output_dir / "recovery_distribution_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "seed": args.seed,
        "samples": args.samples,
        "variants": list(VARIANTS),
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
        "command": "python -m e1.run_recovery_distribution_sensitivity "
        f"--samples {args.samples} --seed {args.seed} "
        f"--output-dir {output_dir}",
        "status": "complete",
        "failure_status": "none",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_dir), "variants": len(VARIANTS)}, indent=2))


if __name__ == "__main__":
    main()
