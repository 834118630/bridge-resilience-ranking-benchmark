"""Uncertainty and multiple-comparison tools for E1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

import numpy as np
from scipy.stats import truncnorm


def sample_truncated_normal(
    mean: float,
    sd: float,
    size: int,
    rng: np.random.Generator,
    lower: float = 0.05,
) -> np.ndarray:
    if mean <= 0 or sd <= 0:
        raise ValueError("mean and sd must be positive")
    a = (lower - mean) / sd
    b = np.inf
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size, random_state=rng)


def stratified_paired_bootstrap(
    keys: Sequence[Hashable],
    strata: Sequence[Hashable],
    n_resamples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Resample paired scenarios with replacement within each stratum."""
    if len(keys) != len(strata):
        raise ValueError("keys and strata must have equal length")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[stratum].append(index)
    samples: list[np.ndarray] = []
    for _ in range(n_resamples):
        selected: list[int] = []
        for indices in groups.values():
            chosen = rng.choice(indices, size=len(indices), replace=True)
            selected.extend(chosen.tolist())
        samples.append(np.asarray(selected, dtype=int))
    return samples


def percentile_ci(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted
