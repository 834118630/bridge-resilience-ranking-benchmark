"""Resilience metrics for normalized bridge recovery curves."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np

from .recovery_models import RECOVERY_MODELS


def functionality_curve(
    time_days: np.ndarray,
    recovery_days: float,
    initial_functionality: float,
    final_functionality: float,
    model_name: str,
) -> np.ndarray:
    if recovery_days <= 0:
        raise ValueError("recovery_days must be positive")
    if not 0.0 <= initial_functionality <= 1.0:
        raise ValueError("initial_functionality must be in [0, 1]")
    if not initial_functionality <= final_functionality <= 1.0:
        raise ValueError("final_functionality must be in [initial, 1]")
    shape_fn: Callable = RECOVERY_MODELS[model_name]
    u = np.asarray(time_days, dtype=float) / recovery_days
    progress = shape_fn(u)
    return initial_functionality + (final_functionality - initial_functionality) * progress


def resilience_index(time_days: np.ndarray, functionality: np.ndarray) -> float:
    if len(time_days) != len(functionality) or len(time_days) < 2:
        raise ValueError("time and functionality arrays must have equal length >= 2")
    duration = float(time_days[-1] - time_days[0])
    if duration <= 0:
        raise ValueError("time range must be positive")
    return float(np.trapezoid(functionality, time_days) / duration)


def time_to_target(
    time_days: np.ndarray,
    functionality: np.ndarray,
    target: float,
) -> float:
    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be in [0, 1]")
    indices = np.flatnonzero(functionality >= target)
    if len(indices) == 0:
        return float(time_days[-1])
    return float(time_days[indices[0]])
