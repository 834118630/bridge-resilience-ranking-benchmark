"""Two-dimensional sensitivity of a fast-recovery profile.

Sweep fast-profile capacity and recovery-speed multiplier against a fixed
high-capacity L4 reference. The output identifies regions where the fast
profile has lower mean regret than L4 and where the decision is ambiguous.
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

from .monte_carlo import (
    METRIC_NAMES,
    candidate_recovery_times,
    damage_state_probabilities_batch,
    metrics_batch,
    normalized_regret_for_scene,
    recovery_parameter_arrays,
    sample_recovery_days_matrix,
)
from .provenance import write_run_metadata
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_profile_phase_diagram"
CAPACITY_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
SPEED_GRID = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
LABELS = ["L0", "L1", "L2", "L3", "L4_high", "F_fast"]


def evaluate_grid_point(
    capacity: float,
    speed_multiplier: float,
    base_days: np.ndarray,
    probabilities_by_case: dict[str, np.ndarray],
    q0: np.ndarray,
    qend: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    capacities = [0.0, 0.25, 0.50, 0.75, 1.0, capacity]
    multipliers = [1.0, 1.15, 1.30, 1.50, 1.75, speed_multiplier]
    recovery_times = candidate_recovery_times(base_days, multipliers)
    condition_regrets = []
    top1_counts = {label: 0 for label in LABELS}
    total_draws = 0

    for probabilities in probabilities_by_case.values():
        for model_name in RECOVERY_MODELS:
            model_metrics = metrics_batch(
                recovery_times,
                probabilities,
                q0,
                qend,
                model_name,
            )
            for metric_name in METRIC_NAMES:
                values = model_metrics[metric_name]
                for sa_index in range(len(SA_LEVELS)):
                    scene_values = values[:, :, sa_index]
                    condition_regrets.append(
                        normalized_regret_for_scene(scene_values)
                    )
                    winners = np.argmax(scene_values, axis=1)
                    for index, label in enumerate(LABELS):
                        top1_counts[label] += int(np.sum(winners == index))
                    total_draws += len(winners)

    mean_regret = np.mean(np.stack(condition_regrets, axis=0), axis=(0, 1))
    top1_frequency = {
        label: top1_counts[label] / total_draws
        for label in LABELS
    }
    return mean_regret, top1_frequency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fragility_path = DATA_DIR / "07_hazus_fragility.csv"
    recovery_path = DATA_DIR / "08_hazus_recovery.csv"
    fragility = load_fragility(fragility_path)
    recovery = load_recovery(recovery_path)
    q0, qend = recovery_parameter_arrays(recovery)
    rng = np.random.default_rng(args.seed)
    base_days = sample_recovery_days_matrix(recovery, args.recovery_samples, rng)
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}

    probabilities_by_case = {}
    for case_id, (conventional_class, seismic_class) in case_pairs.items():
        conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
        seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
        medians = np.asarray(
            [interpolate_medians(conventional, seismic, capacity) for capacity in CAPACITY_GRID],
            dtype=float,
        )
        probabilities_by_case[case_id] = damage_state_probabilities_batch(
            medians,
            beta=load_beta(fragility_path),
            sa_levels=SA_LEVELS,
        )

    rows = []
    difference_matrix = np.zeros((len(CAPACITY_GRID), len(SPEED_GRID)), dtype=float)
    winner_matrix = np.zeros_like(difference_matrix, dtype=int)
    for capacity_index, capacity in enumerate(CAPACITY_GRID):
        for speed_index, speed_multiplier in enumerate(SPEED_GRID):
            # Rebuild probabilities only for the fast profile; the fixed reference
            # curves use the first five capacities in CAPACITY_GRID.
            reference_probabilities = {
                case_id: probabilities_by_case[case_id][:5]
                for case_id in probabilities_by_case
            }
            # Append the fast profile by evaluating its own medians.
            for case_id, (conventional_class, seismic_class) in case_pairs.items():
                conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
                seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
                fast_medians = np.asarray(
                    [interpolate_medians(conventional, seismic, capacity)],
                    dtype=float,
                )
                fast_probability = damage_state_probabilities_batch(
                    fast_medians,
                    beta=load_beta(fragility_path),
                    sa_levels=SA_LEVELS,
                )
                reference_probabilities[case_id] = np.concatenate(
                    [reference_probabilities[case_id], fast_probability],
                    axis=0,
                )

            mean_regret, top1_frequency = evaluate_grid_point(
                capacity,
                speed_multiplier,
                base_days,
                reference_probabilities,
                q0,
                qend,
            )
            difference = float(mean_regret[5] - mean_regret[4])
            difference_matrix[capacity_index, speed_index] = difference
            winner_matrix[capacity_index, speed_index] = 0 if difference < 0 else 1
            rows.append(
                {
                    "capacity": capacity,
                    "speed_multiplier": speed_multiplier,
                    "mean_regret_F_fast": float(mean_regret[5]),
                    "mean_regret_L4_high": float(mean_regret[4]),
                    "regret_difference_F_minus_L4": difference,
                    "winner": "F_fast" if difference < 0 else "L4_high",
                    "F_top1_frequency": top1_frequency["F_fast"],
                    "L4_top1_frequency": top1_frequency["L4_high"],
                }
            )

    with (output_dir / "phase_diagram.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    image = axis.imshow(
        difference_matrix,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-max(abs(difference_matrix.min()), abs(difference_matrix.max())),
        vmax=max(abs(difference_matrix.min()), abs(difference_matrix.max())),
    )
    axis.set_xticks(range(len(SPEED_GRID)), [f"{value:g}" for value in SPEED_GRID])
    axis.set_yticks(range(len(CAPACITY_GRID)), [f"{value:g}" for value in CAPACITY_GRID])
    axis.set_xlabel("Fast-profile recovery multiplier")
    axis.set_ylabel("Fast-profile capacity level")
    axis.set_title("Mean regret difference: F_fast minus L4_high")
    figure.colorbar(image, ax=axis, label="Negative values favor F_fast")
    figure.tight_layout()
    figure.savefig(figure_dir / "phase_diagram_mean_regret_difference.png", dpi=240)
    plt.close(figure)

    summary = {
        "status": "stress_test_not_technology_ranking",
        "capacity_grid": CAPACITY_GRID,
        "speed_multiplier_grid": SPEED_GRID,
        "F_better_count": int(np.sum(winner_matrix == 0)),
        "L4_better_count": int(np.sum(winner_matrix == 1)),
        "output": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_run_metadata(
        output_dir,
        command="python -m e1.run_profile_phase_diagram "
        f"--recovery-samples {args.recovery_samples} --seed {args.seed} "
        f"--output-dir {output_dir}",
        parameters={
            "recovery_samples": args.recovery_samples,
            "seed": args.seed,
            "capacity_grid": CAPACITY_GRID,
            "speed_grid": SPEED_GRID,
        },
        input_paths=(fragility_path, recovery_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
