"""Bootstrap uncertainty for the cost break-even boundary.

This script keeps the P0 cost model fixed and propagates recovery-duration
sampling uncertainty into the effective social value per lost functionality
day, v_day* / C_L4. The reported interval is a bootstrap interval over Monte
Carlo replicates, distinct from the p10-p90 parameter-grid interval.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .monte_carlo import recovery_parameter_arrays, sample_recovery_days_matrix
from .run_cost_break_even import (
    ALPHA_SCENARIOS,
    CAPACITY_GRID,
    GAMMA_GRID,
    L4_MULTIPLIER,
    SPEED_GRID,
    WINDOW_DAYS,
    DISCOUNT_RATE,
    OUTPUT_DIR,
    evaluate_point_sample_losses,
)
from .run_monte_carlo import load_beta
from .run_pilot import DAMAGE_STATES, DATA_DIR, RECOVERY_MODELS, load_fragility, load_recovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fragility = load_fragility(DATA_DIR / "07_hazus_fragility.csv")
    recovery = load_recovery(DATA_DIR / "08_hazus_recovery.csv")
    beta = load_beta(DATA_DIR / "07_hazus_fragility.csv")
    q0, qend = recovery_parameter_arrays(recovery)
    rng = np.random.default_rng(args.seed)
    base_days = sample_recovery_days_matrix(recovery, args.samples, rng)
    time_days = np.arange(0.0, 730.0 + 2.0, 2.0)
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}

    sample_losses: dict[tuple[float, float], np.ndarray] = {}
    for capacity in CAPACITY_GRID:
        for speed_multiplier in SPEED_GRID:
            values = []
            for case_id, (conventional_class, seismic_class) in case_pairs.items():
                conventional = [
                    fragility[(case_id, conventional_class, state)]
                    for state in DAMAGE_STATES
                ]
                seismic = [
                    fragility[(case_id, seismic_class, state)]
                    for state in DAMAGE_STATES
                ]
                for model_name in RECOVERY_MODELS:
                    values.append(
                        evaluate_point_sample_losses(
                            capacity,
                            speed_multiplier,
                            base_days,
                            conventional,
                            seismic,
                            beta,
                            q0,
                            qend,
                            model_name,
                            time_days,
                            DISCOUNT_RATE,
                            WINDOW_DAYS,
                        )
                    )
            sample_losses[(capacity, speed_multiplier)] = np.mean(values, axis=0)

    bootstrap_indices = rng.integers(
        0,
        args.samples,
        size=(args.bootstrap_resamples, args.samples),
    )
    summary: dict[str, object] = {
        "samples": args.samples,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "v_day_definition": "v_day_star / C_L4; finite only when F is costlier and has lower discounted loss days",
        "scenarios": {},
    }
    pointwise_rows = []

    for scenario_name, (alpha_capacity, alpha_speed) in ALPHA_SCENARIOS.items():
        boot_matrix = np.full((len(CAPACITY_GRID) * len(SPEED_GRID) * len(GAMMA_GRID), args.bootstrap_resamples), np.nan)
        point_means = np.full(boot_matrix.shape[0], np.nan)
        row_index = 0
        for capacity in CAPACITY_GRID:
            for speed_multiplier in SPEED_GRID:
                loss_samples = sample_losses[(capacity, speed_multiplier)]
                delta_loss_samples = loss_samples[:, 0] - loss_samples[:, 1]
                for gamma in GAMMA_GRID:
                    cost_l4 = 1.0
                    cost_fast = gamma * np.exp(
                        -alpha_capacity * (1.0 - capacity)
                        + alpha_speed * (L4_MULTIPLIER - speed_multiplier)
                    )
                    delta_cost = float(cost_fast - cost_l4)
                    delta_loss_boot = delta_loss_samples[bootstrap_indices].mean(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        v_day_boot = np.where(
                            (delta_loss_boot > 0.0) & (delta_cost > 0.0),
                            delta_cost / delta_loss_boot,
                            np.nan,
                        )
                    boot_matrix[row_index] = v_day_boot
                    point_means[row_index] = delta_cost / delta_loss_samples.mean()
                    finite = v_day_boot[np.isfinite(v_day_boot)]
                    pointwise_rows.append(
                        {
                            "scenario": scenario_name,
                            "alpha_capacity": alpha_capacity,
                            "alpha_speed": alpha_speed,
                            "capacity": capacity,
                            "speed_multiplier": speed_multiplier,
                            "gamma": gamma,
                            "v_day_star_mean": float(delta_cost / delta_loss_samples.mean()),
                            "bootstrap_ci95_low": float(np.quantile(finite, 0.025)) if len(finite) else None,
                            "bootstrap_ci95_high": float(np.quantile(finite, 0.975)) if len(finite) else None,
                            "finite_bootstrap_share": float(len(finite) / args.bootstrap_resamples),
                        }
                    )
                    row_index += 1

        with np.errstate(all="ignore"):
            median_by_resample = np.nanmedian(boot_matrix, axis=0)
        finite_medians = median_by_resample[np.isfinite(median_by_resample)]
        summary["scenarios"][scenario_name] = {
            "median_v_day_star_mean": float(np.nanmedian(point_means)),
            "median_bootstrap_ci95": [
                float(np.quantile(finite_medians, 0.025)),
                float(np.quantile(finite_medians, 0.975)),
            ],
            "median_bootstrap_iqr": [
                float(np.quantile(finite_medians, 0.25)),
                float(np.quantile(finite_medians, 0.75)),
            ],
            "finite_median_share": float(len(finite_medians) / args.bootstrap_resamples),
        }

    with (output_dir / "cost_break_even_bootstrap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "cost_break_even_bootstrap_pointwise.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pointwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pointwise_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
