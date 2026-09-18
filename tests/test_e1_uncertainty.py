from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e1.uncertainty import holm_adjust, percentile_ci, stratified_paired_bootstrap


def test_holm_adjust_is_monotone():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.all(adjusted >= np.array([0.01, 0.04, 0.03]))
    assert np.all(adjusted <= 1.0)


def test_bootstrap_preserves_group_sizes():
    keys = np.arange(10)
    strata = np.repeat(["A", "B"], 5)
    samples = stratified_paired_bootstrap(keys, strata, n_resamples=20, rng=np.random.default_rng(1))
    assert len(samples) == 20
    assert all(len(sample) == 10 for sample in samples)


def test_percentile_interval_contains_median():
    values = np.arange(101)
    low, high = percentile_ci(values)
    assert low < 50 < high
