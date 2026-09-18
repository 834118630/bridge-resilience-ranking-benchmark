"""Sweep a generic fast-recovery, moderate-capacity profile.

This diagnostic asks when a candidate with a recovery-time advantage can
overtake higher-capacity levels. The profile is not a technology claim; it is
a sensitivity scenario bounded by published repair-duration evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .monte_carlo import (
    METRIC_NAMES,
    damage_state_probabilities_batch,
    metrics_batch,
    normalized_regret_for_scene,
    recovery_parameter_arrays,
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_fast_profile_sweep"
LABELS = ["L0", "L1", "L2", "L3", "L4", "F_fast"]
CAPACITIES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 0.5], dtype=float)
BASE_MULTIPLIERS = np.asarray([1.0, 1.15, 1.30, 1.50, 1.75], dtype=float)
FAST_MULTIPLIERS = np.asarray([0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fragility_path = DATA_DIR / "07_hazus_fragility.csv"
    recovery_path = DATA_DIR / "08_hazus_recovery.csv"
    fragility = load_fragility(fragility_path)
    recovery = load_recovery(recovery_path)
    beta = load_beta(fragility_path)
    q0, qend = recovery_parameter_arrays(recovery)
    base_days = np.asarray(
        [[recovery[state]["mean_days"] for state in DAMAGE_STATES]],
        dtype=float,
    )
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}

    probabilities_by_case: dict[str, np.ndarray] = {}
    for case_id, (conventional_class, seismic_class) in case_pairs.items():
        conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
        seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
        medians = np.asarray(
            [interpolate_medians(conventional, seismic, capacity) for capacity in CAPACITIES],
            dtype=float,
        )
        probabilities_by_case[case_id] = damage_state_probabilities_batch(
            medians,
            beta=beta,
            sa_levels=SA_LEVELS,
        )

    rows: list[dict[str, float | str]] = []
    for fast_multiplier in FAST_MULTIPLIERS:
        multipliers = np.concatenate((BASE_MULTIPLIERS, [fast_multiplier]))
        recovery_times = base_days[:, None, :] * multipliers[None, :, None]
        scene_regrets = []
        top1_counts = {label: 0 for label in LABELS}
        scene_count = 0

        for probabilities in probabilities_by_case.values():
            for metric_name in METRIC_NAMES:
                for model_name in RECOVERY_MODELS:
                    values = metrics_batch(
                        recovery_times,
                        probabilities,
                        q0,
                        qend,
                        model_name,
                    )[metric_name]
                    for sa_index in range(len(SA_LEVELS)):
                        scene_values = values[:, :, sa_index]
                        scene_regrets.append(
                            normalized_regret_for_scene(scene_values)[0]
                        )
                        winner = LABELS[int(np.argmax(scene_values[0]))]
                        top1_counts[winner] += 1
                        scene_count += 1

        mean_regret = np.mean(np.stack(scene_regrets, axis=0), axis=0)
        best = LABELS[int(np.argmin(mean_regret))]
        row: dict[str, float | str] = {
            "fast_multiplier": float(fast_multiplier),
            "best_mean_regret_candidate": best,
        }
        row.update(
            {
                f"mean_regret_{label}": float(mean_regret[index])
                for index, label in enumerate(LABELS)
            }
        )
        row.update(
            {
                f"top1_fraction_{label}": top1_counts[label] / scene_count
                for label in LABELS
            }
        )
        rows.append(row)

    output_path = OUTPUT_DIR / "fast_profile_sweep.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    first_fast_win = next(
        (row for row in rows if row["best_mean_regret_candidate"] == "F_fast"),
        None,
    )
    last_fast_win = next(
        (
            row
            for row in reversed(rows)
            if row["best_mean_regret_candidate"] == "F_fast"
        ),
        None,
    )
    summary = {
        "candidate_definition": {
            "L0_L4_capacities": CAPACITIES[:5].tolist(),
            "F_fast_capacity": float(CAPACITIES[5]),
            "L0_L4_recovery_multipliers": BASE_MULTIPLIERS.tolist(),
        },
        "first_fast_win": first_fast_win,
        "last_fast_win": last_fast_win,
        "output": str(output_path),
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
