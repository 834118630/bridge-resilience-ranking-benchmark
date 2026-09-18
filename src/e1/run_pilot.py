"""Minimal E1 pilot.

This script validates the evaluation workflow using HAZUS fragility and
recovery parameters. Candidate recovery-time multipliers are synthetic
prototype scenarios and must not be reported as empirical findings.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import norm

from .metrics import functionality_curve, resilience_index, time_to_target
from .ranking import (
    kendall_tau,
    normalized_minimax_regret,
    pairwise_reversal_rate,
    rank_candidates,
    spearman_rho,
    top1_stable,
    topk_jaccard,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_pilot"

DAMAGE_STATES = ["Slight", "Moderate", "Extensive", "Complete"]
STATE_PROBABILITY_NAMES = ["None"] + DAMAGE_STATES
RECOVERY_MODELS = ["linear", "exponential", "trigonometric", "delayed_recovery"]
CAPACITY_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
CANDIDATES = [f"L{index}" for index in range(len(CAPACITY_LEVELS))]
RECOVERY_SCENARIOS = {
    "equal_recovery": [1.0, 1.0, 1.0, 1.0, 1.0],
    "capacity_recovery_tradeoff": [1.0, 1.15, 1.30, 1.50, 1.75],
    "higher_capacity_faster_recovery": [1.35, 1.20, 1.05, 0.95, 0.85],
}
SA_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]


def load_fragility(path: Path) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[(row["case_id"], row["bridge_class"], row["damage_state"])] = float(row["median_sa_1s_g"])
    return result


def load_recovery(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["damage_state"]] = {
                "mean_days": float(row["mean_restoration_days"]),
                "sd_days": float(row["sd_days"]),
                "q0": float(row["day1_functionality_proxy"]),
                "qend": float(row["final_target"]),
            }
    return result


def exceeded_probabilities(medians: Iterable[float], beta: float, sa: float) -> np.ndarray:
    return np.array([norm.cdf(np.log(sa / median) / beta) for median in medians], dtype=float)


def damage_state_probabilities(medians: Iterable[float], beta: float, sa: float) -> np.ndarray:
    exceed = np.clip(exceeded_probabilities(medians, beta, sa), 0.0, 1.0)
    probabilities = np.empty(len(DAMAGE_STATES) + 1, dtype=float)
    probabilities[0] = 1.0 - exceed[0]
    for index in range(len(DAMAGE_STATES) - 1):
        probabilities[index + 1] = max(exceed[index] - exceed[index + 1], 0.0)
    probabilities[-1] = exceed[-1]
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("invalid damage-state probabilities")
    return probabilities / total


def interpolate_medians(conventional: Iterable[float], seismic: Iterable[float], capacity: float) -> list[float]:
    conv = np.asarray(list(conventional), dtype=float)
    seis = np.asarray(list(seismic), dtype=float)
    return np.exp((1.0 - capacity) * np.log(conv) + capacity * np.log(seis)).tolist()


def expected_functionality(
    time_days: np.ndarray,
    state_probabilities: np.ndarray,
    recovery_parameters: dict[str, dict[str, float]],
    recovery_multiplier: float,
    model_name: str,
) -> np.ndarray:
    curves = []
    for state, probability in zip(STATE_PROBABILITY_NAMES, state_probabilities):
        if state == "None":
            curves.append(probability * np.ones_like(time_days))
            continue
        params = recovery_parameters[state]
        curves.append(
            probability
            * functionality_curve(
                time_days=time_days,
                recovery_days=params["mean_days"] * recovery_multiplier,
                initial_functionality=params["q0"],
                final_functionality=params["qend"],
                model_name=model_name,
            )
        )
    return np.sum(curves, axis=0)


def utility_metrics(time_days: np.ndarray, functionality: np.ndarray) -> dict[str, float]:
    recovery_target = 0.9
    t90 = time_to_target(time_days, functionality, recovery_target)
    return {
        "resilience_index": resilience_index(time_days, functionality),
        "functionality_day_30": float(np.interp(30.0, time_days, functionality)),
        "functionality_day_90": float(np.interp(90.0, time_days, functionality)),
        "recovery_rapidity_90": 1.0 - t90 / float(time_days[-1]),
        "integral_first_90_days": float(np.trapezoid(functionality[time_days <= 90], time_days[time_days <= 90]) / 90.0),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fragility = load_fragility(DATA_DIR / "07_hazus_fragility.csv")
    recovery = load_recovery(DATA_DIR / "08_hazus_recovery.csv")
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}
    time_days = np.arange(0.0, 501.0, 1.0)

    rankings_by_condition: dict[tuple[str, str, float, str], list[list[str]]] = {}
    utility_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for case_id, (conventional_class, seismic_class) in case_pairs.items():
        conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
        seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
        for recovery_scenario, multipliers in RECOVERY_SCENARIOS.items():
            medians_by_candidate = {
                candidate: interpolate_medians(conventional, seismic, capacity)
                for candidate, capacity in zip(CANDIDATES, CAPACITY_LEVELS)
            }
            for sa in SA_LEVELS:
                for model_name in RECOVERY_MODELS:
                    metrics_by_candidate: dict[str, dict[str, float]] = {candidate: {} for candidate in CANDIDATES}
                    for candidate, medians in medians_by_candidate.items():
                        probabilities = damage_state_probabilities(medians, beta=0.6, sa=sa)
                        q = expected_functionality(
                            time_days,
                            probabilities,
                            recovery,
                            multipliers[CANDIDATES.index(candidate)],
                            model_name,
                        )
                        metrics = utility_metrics(time_days, q)
                        metrics_by_candidate[candidate] = metrics
                        for metric_name, value in metrics.items():
                            utility_rows.append(
                                {
                                    "case": case_id,
                                    "recovery_scenario": recovery_scenario,
                                    "sa": sa,
                                    "recovery_model": model_name,
                                    "candidate": candidate,
                                    "metric": metric_name,
                                    "utility": value,
                                }
                            )
                    for metric_name in utility_metrics(time_days, np.ones_like(time_days)):
                        values = {candidate: metrics_by_candidate[candidate][metric_name] for candidate in CANDIDATES}
                        ranking = rank_candidates(values)
                        key = (case_id, recovery_scenario, sa, metric_name)
                        rankings_by_condition.setdefault(key, []).append(ranking)

    for (case_id, recovery_scenario, sa, metric_name), rankings in rankings_by_condition.items():
        if len(rankings) < 2:
            continue
        base = rankings[0]
        summary_rows.append(
            {
                "case": case_id,
                "recovery_scenario": recovery_scenario,
                "sa": sa,
                "metric": metric_name,
                "top1_change_rate": float(np.mean([not top1_stable(base, ranking) for ranking in rankings[1:]])),
                "pairwise_reversal_rate": pairwise_reversal_rate(rankings),
                "mean_top3_jaccard": float(np.mean([topk_jaccard(base, ranking, 3) for ranking in rankings[1:]])),
                "mean_kendall_tau": float(np.mean([kendall_tau(base, ranking) for ranking in rankings[1:]])),
                "mean_spearman_rho": float(np.mean([spearman_rho(base, ranking) for ranking in rankings[1:]])),
            }
        )

    minimax_input: dict[str, dict[str, float]] = {}
    candidate_metrics: dict[tuple[str, str, float, str], dict[str, float]] = {}
    for row in utility_rows:
        key = (str(row["case"]), str(row["recovery_scenario"]), float(row["sa"]), str(row["metric"]))
        candidate_metrics.setdefault(key, {})[str(row["candidate"])] = float(row["utility"])
    for key, values in candidate_metrics.items():
        minimax_input["|".join(map(str, key))] = values
    minimax = normalized_minimax_regret(minimax_input)

    with (OUTPUT_DIR / "rankings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "recovery_scenario", "sa", "recovery_model", "metric", "ranking"])
        for (case_id, recovery_scenario, sa, metric_name), rankings in rankings_by_condition.items():
            for model_name, ranking in zip(RECOVERY_MODELS, rankings):
                writer.writerow([case_id, recovery_scenario, sa, model_name, metric_name, " > ".join(ranking)])
    with (OUTPUT_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUTPUT_DIR / "minimax_regret.json").open("w", encoding="utf-8") as handle:
        json.dump(minimax, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        "cases": sorted(case_pairs),
        "conditions": len(rankings_by_condition),
        "models": RECOVERY_MODELS,
        "output": str(OUTPUT_DIR),
        "max_pairwise_reversal_rate": max(row["pairwise_reversal_rate"] for row in summary_rows),
        "max_top1_change_rate": max(row["top1_change_rate"] for row in summary_rows),
        "minimax_regret": minimax,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


