"""Normalized recovery-function models.

All functions map normalized recovery time u in [0, 1] to progress in [0, 1].
The main comparison fixes initial functionality, final functionality, and
recovery duration before comparing these shapes.
"""

from __future__ import annotations

import numpy as np


def _u(value: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), 0.0, 1.0)


def linear(u: np.ndarray | float) -> np.ndarray:
    return _u(u)


def exponential(u: np.ndarray | float, rate: float = 5.0) -> np.ndarray:
    if rate <= 0:
        raise ValueError("rate must be positive")
    x = _u(u)
    return (1.0 - np.exp(-rate * x)) / (1.0 - np.exp(-rate))


def trigonometric(u: np.ndarray | float) -> np.ndarray:
    x = _u(u)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def delayed_recovery(
    u: np.ndarray | float,
    delay_fraction: float = 0.2,
) -> np.ndarray:
    if not 0.0 <= delay_fraction < 1.0:
        raise ValueError("delay_fraction must be in [0, 1)")
    x = _u(u)
    if delay_fraction == 0.0:
        s = x
    else:
        s = np.clip((x - delay_fraction) / (1.0 - delay_fraction), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


RECOVERY_MODELS = {
    "linear": linear,
    "exponential": exponential,
    "trigonometric": trigonometric,
    "delayed_recovery": delayed_recovery,
}

