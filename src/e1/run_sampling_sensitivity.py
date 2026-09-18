"""Paired-versus-independent recovery-duration sensitivity for E1.

The main E1 analysis is paired: each Monte Carlo replication reuses one sampled
damage-state duration vector across all candidates. This script adds the
alternative independent-sampling design, where every candidate receives its own
sampled duration vector. It is a stress test, not a replacement for the main
paired analysis.
"""

from __future__ import annotations

import argparse
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_sampling_sensitivity"
PROFILE_LABELS = ["L0", "L1", "L2", "L3", "L4", "F_fast"]
PROFILE_CAPACITIES = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5]
PROFILE_SCENARIOS = {
    "evidence_tradeoff": [1.0, 1.15, 1.30, 1.50, 1.75, 0.10],
    "boundary_tradeoff": [1.0, 1.15, 1.30, 1.50, 1.75, 0.50],
}


def sample_candidate_recovery_times(
    recovery: dict[str, dict[str, float]],
    samples: int,
    multipliers: list[float],
    sampling_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample candidate recovery-time matrices for one sampling design."""

    if sampling_mode == "paired":
        base_days = sample_recovery_days_matrix(recovery, samples, rng)
        return candidate_recovery_times(base_days, multipliers)
    if sampling_mode == "independent":
        independent = np.stack(
            [
                sample_recovery_days_matrix(recovery, samples, rng)
                for _ in multipliers
            ],
            axis=1,
        )
        return independent * np.asarray(multipliers, dtype=float)[None, :, None]
    raise ValueError("sampling_mode must be 'paired' or 'independent'")


def summarize_mode(
    probabilities_by_case: dict[str, np.ndarray],
    recovery_times: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
) -> dict[str, object]:
    """Summarize mean regret and Top-1 frequency across all condition layers."""

    condition_means: dict[str, list[np.ndarray]] = defaultdict(list)
    top1_counts = {candidate: 0 for candidate in PROFILE_LABELS}
    total_top1_draws = 0
    fast_minus_l4_condition_wins = 0
    condition_count = 0

    for case_id, probabilities in probabilities_by_case.items():
        for model_name in RECOVERY_MODELS:
            metrics = metrics_batch(recovery_times, probabilities, q0, qend, model_name)
            for metric_name in METRIC_NAMES:
                values = metrics[metric_name]
                winners = np.argmax(values, axis=1)
                flat_winners = winners.reshape(-1)
                for index, candidate in enumerate(PROFILE_LABELS):
                    top1_counts[candidate] += int(np.sum(flat_winners == index))
                total_top1_draws += int(flat_winners.size)
            for metric_name in METRIC_NAMES:
                for sa_index in range(len(SA_LEVELS)):
                    regret = normalized_regret_for_scene(
                        metrics[metric_name][:, :, sa_index]
                    )[0]
                    condition_means[case_id].append(regret)
                    fast_minus_l4_condition_wins += int(regret[5] < regret[4])
                    condition_count += 1

    all_conditions = [
        value for values in condition_means.values() for value in values
    ]
    mean_regret = np.mean(np.stack(all_conditions, axis=0), axis=0)
    candidate_summary = {}
    for index, candidate in enumerate(PROFILE_LABELS):
        candidate_summary[candidate] = {
            "mean_regret": float(mean_regret[index]),
            "top1_frequency": float(top1_counts[candidate] / total_top1_draws),
        }
    return {
        "candidates": candidate_summary,
        "fast_minus_l4_mean_regret": float(mean_regret[5] - mean_regret[4]),
        "fast_minus_l4_top1_frequency": float(
            candidate_summary["F_fast"]["top1_frequency"]
            - candidate_summary["L4"]["top1_frequency"]
        ),
        "fast_lower_regret_condition_share": float(
            fast_minus_l4_condition_wins / condition_count
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fragility = load_fragility(DATA_DIR / "07_hazus_fragility.csv")
    recovery = load_recovery(DATA_DIR / "08_hazus_recovery.csv")
    beta = load_beta(DATA_DIR / "07_hazus_fragility.csv")
    q0, qend = recovery_parameter_arrays(recovery)
    rng = np.random.default_rng(args.seed)
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}

    probabilities_by_case: dict[str, np.ndarray] = {}
    for case_id, (conventional_class, seismic_class) in case_pairs.items():
        conventional = [
            fragility[(case_id, conventional_class, state)]
            for state in DAMAGE_STATES
        ]
        seismic = [
            fragility[(case_id, seismic_class, state)]
            for state in DAMAGE_STATES
        ]
        medians = np.asarray(
            [
                interpolate_medians(conventional, seismic, capacity)
                for capacity in PROFILE_CAPACITIES
            ],
            dtype=float,
        )
        probabilities_by_case[case_id] = damage_state_probabilities_batch(
            medians,
            beta,
            SA_LEVELS,
        )

    output: dict[str, object] = {
        "samples": args.samples,
        "seed": args.seed,
        "scenarios": {},
    }
    for scenario_name, multipliers in PROFILE_SCENARIOS.items():
        scenario_output: dict[str, object] = {}
        for sampling_mode in ("paired", "independent"):
            scenario_rng = np.random.default_rng(args.seed + (0 if sampling_mode == "paired" else 1000))
            recovery_times = sample_candidate_recovery_times(
                recovery,
                args.samples,
                multipliers,
                sampling_mode,
                scenario_rng,
            )
            scenario_output[sampling_mode] = summarize_mode(
                probabilities_by_case,
                recovery_times,
                q0,
                qend,
            )
        paired = scenario_output["paired"]
        independent = scenario_output["independent"]
        scenario_output["change_from_paired_to_independent"] = {
            "F_fast_mean_regret_delta": float(
                independent["candidates"]["F_fast"]["mean_regret"]
                - paired["candidates"]["F_fast"]["mean_regret"]
            ),
            "F_fast_top1_frequency_delta": float(
                independent["candidates"]["F_fast"]["top1_frequency"]
                - paired["candidates"]["F_fast"]["top1_frequency"]
            ),
            "L4_top1_frequency_delta": float(
                independent["candidates"]["L4"]["top1_frequency"]
                - paired["candidates"]["L4"]["top1_frequency"]
            ),
        }
        output["scenarios"][scenario_name] = scenario_output

    with (output_dir / "sampling_sensitivity.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
