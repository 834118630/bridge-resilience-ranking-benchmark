"""Vectorized Monte Carlo evaluation for the E1 resilience-ranking study.

The comparison unit is paired: each Monte Carlo replication samples one vector
of damage-state recovery durations, then applies the same sampled durations to
all candidate performance levels within a recovery scenario. Candidate-specific
multipliers are used only to represent the secondary capacity-recovery
tradeoff scenarios; the shape-only scenario keeps them equal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np
from scipy.stats import norm, truncnorm

from .recovery_models import RECOVERY_MODELS

DAMAGE_STATES = ("Slight", "Moderate", "Extensive", "Complete")
RECOVERY_SCENARIOS: dict[str, tuple[float, ...]] = {
    "shape_only": (1.0, 1.0, 1.0, 1.0, 1.0),
    "capacity_recovery_tradeoff": (1.0, 1.15, 1.30, 1.50, 1.75),
    "aligned_capacity_recovery": (1.35, 1.20, 1.05, 0.95, 0.85),
}
METRIC_NAMES = (
    "resilience_index",
    "functionality_day_30",
    "functionality_day_90",
    "recovery_rapidity_90",
)


def sample_recovery_days_matrix(
    recovery: Mapping[str, Mapping[str, float]],
    n_samples: int,
    rng: np.random.Generator,
    states: Sequence[str] = DAMAGE_STATES,
    lower_bound_days: float = 0.05,
) -> np.ndarray:
    """Sample paired damage-state recovery durations.

    Returns an array with shape (n_samples, n_states). The same sampled vector
    is reused across every candidate in a replication.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    sampled = []
    for state in states:
        mean = float(recovery[state]["mean_days"])
        sd = float(recovery[state]["sd_days"])
        if mean <= 0.0 or sd <= 0.0:
            raise ValueError("recovery mean and sd must be positive")
        a = (lower_bound_days - mean) / sd
        sampled.append(
            truncnorm.rvs(
                a,
                np.inf,
                loc=mean,
                scale=sd,
                size=n_samples,
                random_state=rng,
            )
        )
    return np.column_stack(sampled)


def recovery_parameter_arrays(
    recovery: Mapping[str, Mapping[str, float]],
    states: Sequence[str] = DAMAGE_STATES,
) -> tuple[np.ndarray, np.ndarray]:
    q0 = np.asarray([float(recovery[state]["q0"]) for state in states], dtype=float)
    qend = np.asarray([float(recovery[state]["qend"]) for state in states], dtype=float)
    if np.any(q0 < 0.0) or np.any(qend > 1.0) or np.any(qend < q0):
        raise ValueError("invalid initial or final functionality values")
    return q0, qend


def damage_state_probabilities_batch(
    medians: np.ndarray,
    beta: float,
    sa_levels: Sequence[float],
) -> np.ndarray:
    """Evaluate HAZUS exceedance curves for candidates and Sa levels.

    Parameters
    ----------
    medians:
        Array with shape (n_candidates, n_damage_states), ordered from Slight
        to Complete.
    beta:
        Lognormal dispersion.
    sa_levels:
        Spectral acceleration levels in g.

    Returns
    -------
    probabilities:
        Array with shape (n_candidates, n_sa, n_damage_states + 1). The first
        column is no damage, followed by the four damage states.
    """

    medians_array = np.asarray(medians, dtype=float)
    sa_array = np.asarray(sa_levels, dtype=float)
    if medians_array.ndim != 2:
        raise ValueError("medians must have shape (candidates, damage_states)")
    if sa_array.ndim != 1 or len(sa_array) == 0:
        raise ValueError("sa_levels must be a non-empty one-dimensional sequence")
    if np.any(medians_array <= 0.0) or np.any(sa_array <= 0.0):
        raise ValueError("medians and sa levels must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    exceedance = norm.cdf(
        (np.log(sa_array)[None, :, None] - np.log(medians_array)[:, None, :]) / beta
    )
    probabilities = np.empty(
        (medians_array.shape[0], len(sa_array), medians_array.shape[1] + 1),
        dtype=float,
    )
    probabilities[:, :, 0] = 1.0 - exceedance[:, :, 0]
    probabilities[:, :, 1:-1] = exceedance[:, :, :-1] - exceedance[:, :, 1:]
    probabilities[:, :, -1] = exceedance[:, :, -1]
    probabilities = np.clip(probabilities, 0.0, 1.0)
    totals = probabilities.sum(axis=2, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("invalid damage-state probabilities")
    return probabilities / totals


@lru_cache(maxsize=None)
def cumulative_shape_integral(
    model_name: str,
    n_grid: int = 20001,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a dense cumulative integral of a normalized shape function."""

    if model_name not in RECOVERY_MODELS:
        raise KeyError(f"unknown recovery model: {model_name}")
    if n_grid < 1001:
        raise ValueError("n_grid must be at least 1001")
    u = np.linspace(0.0, 1.0, n_grid)
    progress = np.asarray(RECOVERY_MODELS[model_name](u), dtype=float)
    increments = 0.5 * (progress[1:] + progress[:-1]) * np.diff(u)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    return u, cumulative


def candidate_recovery_times(
    base_days: np.ndarray,
    multipliers: Sequence[float],
) -> np.ndarray:
    """Combine sampled base durations with candidate-specific multipliers."""

    base = np.asarray(base_days, dtype=float)
    multiplier_array = np.asarray(multipliers, dtype=float)
    if base.ndim != 2:
        raise ValueError("base_days must have shape (samples, damage_states)")
    if multiplier_array.ndim != 1 or len(multiplier_array) == 0:
        raise ValueError("multipliers must be a non-empty one-dimensional sequence")
    if np.any(multiplier_array <= 0.0):
        raise ValueError("multipliers must be positive")
    return base[:, None, :] * multiplier_array[None, :, None]


def expected_functionality_at(
    time: np.ndarray | float,
    recovery_times: np.ndarray,
    probabilities: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Expected functionality at one or more times.

    recovery_times has shape (samples, candidates, damage_states). probabilities
    has shape (candidates, sa_levels, damage_states + 1). For an array-valued
    time with shape (samples, candidates, sa_levels), the result has the same
    shape.
    """

    if model_name not in RECOVERY_MODELS:
        raise KeyError(f"unknown recovery model: {model_name}")
    times = np.asarray(recovery_times, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if times.ndim != 3:
        raise ValueError("recovery_times must have shape (samples, candidates, states)")
    if probs.ndim != 3 or probs.shape[0] != times.shape[1] or probs.shape[2] != times.shape[2] + 1:
        raise ValueError("probabilities must have shape (candidates, sa_levels, states + 1)")
    if np.any(times <= 0.0):
        raise ValueError("recovery times must be positive")

    q0_array = np.asarray(q0, dtype=float)
    qend_array = np.asarray(qend, dtype=float)
    delta = qend_array - q0_array
    progress_fn = RECOVERY_MODELS[model_name]

    if np.ndim(time) == 0:
        progress = float(time) / times[:, :, None, :]
        progress = progress_fn(progress)
        damage = np.sum(
            probs[None, :, :, 1:]
            * (q0_array[None, None, None, :] + delta[None, None, None, :] * progress),
            axis=-1,
        )
        q = probs[None, :, :, 0] + damage
    else:
        time_array = np.asarray(time, dtype=float)
        expected_shape = times.shape[:2] + (probs.shape[1],)
        if time_array.shape != expected_shape:
            raise ValueError(f"array-valued time must have shape {expected_shape}")
        progress = time_array[:, :, :, None] / times[:, :, None, :]
        progress = progress_fn(progress)
        damage = np.sum(
            probs[None, :, :, 1:]
            * (q0_array[None, None, None, :] + delta[None, None, None, :] * progress),
            axis=-1,
        )
        q = probs[None, :, :, 0] + damage
    return np.clip(q, 0.0, 1.0)


def resilience_index_batch(
    recovery_times: np.ndarray,
    probabilities: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    time_window_days: float = 500.0,
) -> np.ndarray:
    """Finite-window mean functionality without constructing full time curves."""

    if time_window_days <= 0.0:
        raise ValueError("time_window_days must be positive")
    times = np.asarray(recovery_times, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    q0_array = np.asarray(q0, dtype=float)
    delta = np.asarray(qend, dtype=float) - q0_array
    u_grid, cumulative = cumulative_shape_integral(model_name)
    end_fraction = np.minimum(1.0, time_window_days / times)
    cumulative_at_end = np.interp(end_fraction, u_grid, cumulative)
    shape_integrals = times * cumulative_at_end + np.maximum(time_window_days - times, 0.0)
    damage = np.sum(
        probs[None, :, :, 1:]
        * (
            q0_array[None, None, None, :]
            + delta[None, None, None, :] * (shape_integrals[:, :, None, :] / time_window_days)
        ),
        axis=-1,
    )
    return np.clip(probs[None, :, :, 0] + damage, 0.0, 1.0)


def time_to_target_batch(
    recovery_times: np.ndarray,
    probabilities: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    target: float = 0.9,
    time_window_days: float = 500.0,
    n_iterations: int = 28,
) -> np.ndarray:
    """Vectorized bisection for the first time expected functionality reaches target."""

    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be in [0, 1]")
    times = np.asarray(recovery_times, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    shape = times.shape[:2] + (probs.shape[1],)
    low = np.zeros(shape, dtype=float)
    high = np.full(shape, float(time_window_days), dtype=float)
    q_zero = expected_functionality_at(0.0, times, probs, q0, qend, model_name)
    q_high = expected_functionality_at(time_window_days, times, probs, q0, qend, model_name)
    reachable = (q_high >= target) | (q_zero >= target)
    for _ in range(n_iterations):
        mid = 0.5 * (low + high)
        q_mid = expected_functionality_at(mid, times, probs, q0, qend, model_name)
        below = q_mid < target
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
    result = 0.5 * (low + high)
    result = np.where(q_zero >= target, 0.0, result)
    return np.where(reachable, result, float(time_window_days))


def metrics_batch(
    recovery_times: np.ndarray,
    probabilities: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    time_window_days: float = 500.0,
    recovery_target: float = 0.9,
) -> dict[str, np.ndarray]:
    """Evaluate all frozen E1 utility metrics for one model and scenario."""

    t_target = time_to_target_batch(
        recovery_times,
        probabilities,
        q0,
        qend,
        model_name,
        target=recovery_target,
        time_window_days=time_window_days,
    )
    return {
        "resilience_index": resilience_index_batch(
            recovery_times,
            probabilities,
            q0,
            qend,
            model_name,
            time_window_days=time_window_days,
        ),
        "functionality_day_30": expected_functionality_at(
            30.0, recovery_times, probabilities, q0, qend, model_name
        ),
        "functionality_day_90": expected_functionality_at(
            90.0, recovery_times, probabilities, q0, qend, model_name
        ),
        "recovery_rapidity_90": 1.0 - t_target / float(time_window_days),
    }


def mean_shape_over_window(
    model_name: str,
    recovery_times: np.ndarray,
    time_window_days: float,
) -> np.ndarray:
    """Mean recovery progress over a finite window.

    This is the linear weight needed for the finite-window resilience index:
    (1 / W) * integral_0^W g(t / T) dt.
    """

    if time_window_days <= 0.0:
        raise ValueError("time_window_days must be positive")
    times = np.asarray(recovery_times, dtype=float)
    if np.any(times <= 0.0):
        raise ValueError("recovery times must be positive")
    u_grid, cumulative = cumulative_shape_integral(model_name)
    end_fraction = np.minimum(1.0, time_window_days / times)
    cumulative_at_end = np.interp(end_fraction, u_grid, cumulative)
    area = times * cumulative_at_end + np.maximum(time_window_days - times, 0.0)
    return area / float(time_window_days)


def linear_metric_weights(
    recovery_times: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    metric_name: str,
    time_window_days: float = 500.0,
    fixed_day: float | None = None,
) -> np.ndarray:
    """Return per-damage-state weights for linear resilience metrics.

    For a damage-state probability vector p and weights alpha_s, the metric is

        M = p[no damage] * 1 + sum_s p_s * alpha_s.

    `resilience_index`, `functionality_day_30`, and `functionality_day_90`
    are linear in the curve and therefore admit this decomposition.
    """

    if metric_name == "resilience_index":
        progress = mean_shape_over_window(
            model_name,
            recovery_times,
            time_window_days,
        )
    elif metric_name in ("functionality_day_30", "functionality_day_90"):
        if fixed_day is None:
            fixed_day = 30.0 if metric_name == "functionality_day_30" else 90.0
        progress = RECOVERY_MODELS[model_name](float(fixed_day) / np.asarray(recovery_times, dtype=float))
    else:
        raise ValueError(f"metric is not linear in damage-state probabilities: {metric_name}")

    q0_array = np.asarray(q0, dtype=float)
    qend_array = np.asarray(qend, dtype=float)
    return q0_array + (qend_array - q0_array) * progress


def linear_metric_values(
    probabilities: np.ndarray,
    recovery_times: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    metric_name: str,
    time_window_days: float = 500.0,
    fixed_day: float | None = None,
) -> np.ndarray:
    """Evaluate a linear metric from damage-state probabilities and weights."""

    probs = np.asarray(probabilities, dtype=float)
    weights = linear_metric_weights(
        recovery_times,
        q0,
        qend,
        model_name,
        metric_name,
        time_window_days=time_window_days,
        fixed_day=fixed_day,
    )
    if probs.ndim != 3 or probs.shape[2] != weights.shape[2] + 1:
        raise ValueError("probabilities must have shape (candidates, sa_levels, states + 1)")
    values = probs[None, :, :, 0] + np.sum(
        probs[None, :, :, 1:] * weights[:, :, None, :],
        axis=-1,
    )
    return np.clip(values, 0.0, 1.0)


def pairwise_metric_differences(
    values: np.ndarray,
    candidate_pairs: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Return paired candidate metric differences with shape (samples, pairs, sa)."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError("values must have shape (samples, candidates, sa_levels)")
    n_candidates = array.shape[1]
    if candidate_pairs is None:
        candidate_pairs = [
            (first, second)
            for first in range(n_candidates)
            for second in range(first + 1, n_candidates)
        ]
    return np.stack(
        [array[:, first, :] - array[:, second, :] for first, second in candidate_pairs],
        axis=1,
    )


def _ranking_positions(orders: np.ndarray) -> np.ndarray:
    n_samples, n_candidates, n_sa = orders.shape
    positions = np.empty_like(orders)
    sample_index = np.arange(n_samples)[:, None, None]
    sa_index = np.arange(n_sa)[None, None, :]
    positions[sample_index, orders, sa_index] = np.arange(n_candidates)[None, :, None]
    return positions


def ranking_stability_statistics(
    metric_arrays_by_model: Mapping[str, np.ndarray],
    k: int = 3,
) -> dict[str, np.ndarray]:
    """Compute ranking-stability statistics for paired model comparisons.

    Every array in metric_arrays_by_model must have shape
    (samples, candidates, sa_levels). The first mapping key is the base model.
    All returned arrays have shape (samples, sa_levels).
    """

    model_names = list(metric_arrays_by_model)
    if len(model_names) < 2:
        raise ValueError("at least two recovery models are required")
    arrays = [np.asarray(metric_arrays_by_model[name], dtype=float) for name in model_names]
    shape = arrays[0].shape
    if len(shape) != 3 or any(array.shape != shape for array in arrays):
        raise ValueError("all metric arrays must have shape (samples, candidates, sa_levels)")
    n_samples, n_candidates, n_sa = shape
    if not 1 <= k <= n_candidates:
        raise ValueError("k must be between 1 and the number of candidates")

    orders = np.stack([np.argsort(-array, axis=1, kind="stable") for array in arrays], axis=0)
    positions = np.stack([_ranking_positions(order) for order in orders], axis=0)
    top1 = orders[:, 0, :, :]

    top1_change = np.mean(top1[1:] != top1[0][None, :, :], axis=0)

    topk_masks = positions < k
    jaccard_values = []
    tau_values = []
    spearman_values = []
    for model_index in range(1, len(model_names)):
        intersection = np.sum(topk_masks[0] & topk_masks[model_index], axis=1)
        union = np.sum(topk_masks[0] | topk_masks[model_index], axis=1)
        jaccard_values.append(intersection / np.where(union == 0, 1, union))

        concordant = np.zeros((n_samples, n_sa), dtype=float)
        n_pairs = n_candidates * (n_candidates - 1) / 2.0
        for first in range(n_candidates):
            for second in range(first + 1, n_candidates):
                sign_a = np.sign(positions[0, :, first, :] - positions[0, :, second, :])
                sign_b = np.sign(
                    positions[model_index, :, first, :] - positions[model_index, :, second, :]
                )
                concordant += sign_a == sign_b
        tau_values.append(2.0 * concordant / n_pairs - 1.0)

        differences = positions[0] - positions[model_index]
        spearman_values.append(
            1.0 - 6.0 * np.sum(differences * differences, axis=1)
            / (n_candidates * (n_candidates * n_candidates - 1))
        )

    pairwise_reversals = np.zeros((n_samples, n_sa), dtype=float)
    pair_signs = np.empty(
        (len(model_names), n_samples, n_candidates * (n_candidates - 1) // 2, n_sa),
        dtype=float,
    )
    pair_index = 0
    for first in range(n_candidates):
        for second in range(first + 1, n_candidates):
            pair_signs[:, :, pair_index, :] = np.sign(
                positions[:, :, first, :] - positions[:, :, second, :]
            )
            pair_index += 1
    model_pair_count = 0
    for first_model in range(len(model_names)):
        for second_model in range(first_model + 1, len(model_names)):
            pairwise_reversals += np.sum(
                pair_signs[first_model] != pair_signs[second_model], axis=1
            )
            model_pair_count += 1
    pairwise_reversal_rate = pairwise_reversals / (model_pair_count * pair_index)

    return {
        "top1_change_rate": top1_change,
        "pairwise_reversal_rate": pairwise_reversal_rate,
        "mean_top3_jaccard": np.mean(np.stack(jaccard_values, axis=0), axis=0),
        "mean_kendall_tau": np.mean(np.stack(tau_values, axis=0), axis=0),
        "mean_spearman_rho": np.mean(np.stack(spearman_values, axis=0), axis=0),
    }


def normalized_regret_for_scene(values: np.ndarray) -> np.ndarray:
    """Normalized regret array for values with shape (samples, candidates)."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must have shape (samples, candidates)")
    best = np.max(array, axis=1, keepdims=True)
    spread = np.max(array, axis=1, keepdims=True) - np.min(array, axis=1, keepdims=True)
    regret = np.divide(best - array, spread, out=np.zeros_like(array), where=spread > 0.0)
    return regret
