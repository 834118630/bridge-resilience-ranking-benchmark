"""Ranking-stability and regret metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def rank_candidates(values: dict[str, float], descending: bool = True) -> list[str]:
    return sorted(values, key=values.get, reverse=descending)


def top1_stable(rank_a: list[str], rank_b: list[str]) -> bool:
    return bool(rank_a and rank_b and rank_a[0] == rank_b[0])


def pairwise_reversal_rate(rankings: list[list[str]]) -> float:
    if len(rankings) < 2:
        return 0.0
    candidates = rankings[0]
    pairs = [(candidates[i], candidates[j]) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
    reversals = 0
    comparisons = 0
    for a, b in pairs:
        signs = []
        for ranking in rankings:
            positions = {candidate: idx for idx, candidate in enumerate(ranking)}
            signs.append(np.sign(positions[a] - positions[b]))
        for i in range(len(signs)):
            for j in range(i + 1, len(signs)):
                comparisons += 1
                if signs[i] != signs[j]:
                    reversals += 1
    return float(reversals / comparisons) if comparisons else 0.0


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    positions_a = {candidate: idx for idx, candidate in enumerate(rank_a)}
    positions_b = {candidate: idx for idx, candidate in enumerate(rank_b)}
    x = [positions_a[c] for c in rank_a]
    y = [positions_b[c] for c in rank_a]
    return float(kendalltau(x, y).statistic)


def spearman_rho(rank_a: list[str], rank_b: list[str]) -> float:
    positions_a = {candidate: idx for idx, candidate in enumerate(rank_a)}
    positions_b = {candidate: idx for idx, candidate in enumerate(rank_b)}
    x = [positions_a[c] for c in rank_a]
    y = [positions_b[c] for c in rank_a]
    return float(spearmanr(x, y).statistic)


def topk_jaccard(rank_a: list[str], rank_b: list[str], k: int) -> float:
    set_a = set(rank_a[:k])
    set_b = set(rank_b[:k])
    union = set_a | set_b
    return float(len(set_a & set_b) / len(union)) if union else 1.0


def normalized_minimax_regret(utilities: dict[str, dict[str, float]]) -> dict[str, float]:
    """Return candidate regret with scenarios on the first key.

    utilities maps scenario name -> candidate name -> utility, where larger is better.
    """
    scenario_regrets: dict[str, dict[str, float]] = {}
    for scenario, values in utilities.items():
        spread = max(values.values()) - min(values.values())
        if spread <= 0:
            scenario_regrets[scenario] = {candidate: 0.0 for candidate in values}
            continue
        best = max(values.values())
        scenario_regrets[scenario] = {candidate: (best - value) / spread for candidate, value in values.items()}
    candidates = list(next(iter(utilities.values())).keys())
    return {candidate: max(sr[candidate] for sr in scenario_regrets.values()) for candidate in candidates}
