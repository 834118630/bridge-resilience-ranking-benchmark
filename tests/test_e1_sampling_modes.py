from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e1.monte_carlo import sample_recovery_days_matrix
from e1.run_independent_sampling_robustness import (
    independent_recovery_base,
    recovery_times_for_mode,
)


def _recovery():
    return {
        "Slight": {"mean_days": 0.6, "sd_days": 0.2},
        "Moderate": {"mean_days": 2.5, "sd_days": 0.5},
        "Extensive": {"mean_days": 75.0, "sd_days": 10.0},
        "Complete": {"mean_days": 230.0, "sd_days": 20.0},
    }


def test_independent_recovery_base_shape():
    rng = np.random.default_rng(123)
    base = independent_recovery_base(_recovery(), n_samples=8, n_candidates=6, rng=rng)
    assert base.shape == (8, 6, 4)
    assert np.all(base > 0.0)


def test_recovery_times_for_paired_and_independent_modes():
    rng = np.random.default_rng(123)
    paired_base = sample_recovery_days_matrix(_recovery(), 8, rng)
    independent_base = independent_recovery_base(_recovery(), 8, 3, rng)
    multipliers = np.array([1.0, 0.5, 1.75])
    paired = recovery_times_for_mode(paired_base, multipliers, "paired")
    independent = recovery_times_for_mode(independent_base, multipliers, "independent")
    assert paired.shape == (8, 3, 4)
    assert independent.shape == (8, 3, 4)
    assert np.allclose(independent[:, 1, :], independent_base[:, 1, :] * 0.5)
