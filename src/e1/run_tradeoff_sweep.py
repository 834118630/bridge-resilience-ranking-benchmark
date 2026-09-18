"""Deterministic sweep of capacity-recovery tradeoff strength.

The sweep identifies when the mean-regret-best performance level changes as
higher capacity is penalized by slower recovery. It is a diagnostic for the
E1 value gate, not evidence that any particular technology is optimal.
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
    ranking_stability_statistics,
    recovery_parameter_arrays,
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_tradeoff_sweep"
TRADEOFF_VECTOR = np.asarray([0.0, 0.15, 0.30, 0.50, 0.75])


def load_beta(path: Path) -> float:
    values = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["beta"]))
    if not values or not np.allclose(values, values[0]):
        raise ValueError("all fragility rows must share one beta")
    return values[0]


def mean_utility_regret(
    values_by_model: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    """Return candidate mean regret and mean top-1 change rate."""

    regrets = []
    for values in values_by_model.values():
        n_samples, n_candidates, n_sa = values.shape
        if n_samples != 1 or n_candidates != len(CANDIDATES):
            raise ValueError("sweep values must have shape (1, candidates, sa)")
        for sa_index in range(n_sa):
            regrets.append(normalized_regret_for_scene(values[:, :, sa_index])[0])
    mean_regret = np.mean(np.stack(regrets, axis=0), axis=0)
    stats = ranking_stability_statistics(values_by_model, k=3)
    return mean_regret, float(np.mean(stats["top1_change_rate"]))


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
            [interpolate_medians(conventional, seismic, capacity) for capacity in CAPACITY_LEVELS],
            dtype=float,
        )
        probabilities_by_case[case_id] = damage_state_probabilities_batch(
            medians,
            beta=beta,
            sa_levels=SA_LEVELS,
        )

    rows = []
    strengths = np.linspace(0.0, 2.0, 41)
    for strength in strengths:
        multipliers = 1.0 + strength * TRADEOFF_VECTOR
        recovery_times = base_days[:, None, :] * multipliers[None, :, None]
        case_metric_regrets = []
        case_metric_top1 = []

        for case_id, probabilities in probabilities_by_case.items():
            for metric_name in METRIC_NAMES:
                values_by_model = {}
                for model_name in RECOVERY_MODELS:
                    values_by_model[model_name] = metrics_batch(
                        recovery_times,
                        probabilities,
                        q0,
                        qend,
                        model_name,
                    )[metric_name]
                mean_regret, mean_top1 = mean_utility_regret(values_by_model)
                case_metric_regrets.append(mean_regret)
                case_metric_top1.append(mean_top1)

        candidate_regret = np.mean(np.stack(case_metric_regrets, axis=0), axis=0)
        best_candidate = CANDIDATES[int(np.argmin(candidate_regret))]
        row = {
            "tradeoff_strength": float(strength),
            "best_mean_regret_candidate": best_candidate,
            "mean_top1_change_rate": float(np.mean(case_metric_top1)),
        }
        row.update(
            {
                f"mean_regret_{candidate}": float(candidate_regret[index])
                for index, candidate in enumerate(CANDIDATES)
            }
        )
        rows.append(row)

    output_path = OUTPUT_DIR / "tradeoff_sweep.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    first_non_l4 = next(
        (row for row in rows if row["best_mean_regret_candidate"] != "L4"),
        None,
    )
    summary = {
        "tradeoff_vector": TRADEOFF_VECTOR.tolist(),
        "strength_range": [float(strengths[0]), float(strengths[-1])],
        "best_candidate_at_zero": rows[0]["best_mean_regret_candidate"],
        "first_non_l4": first_non_l4,
        "output": str(output_path),
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
