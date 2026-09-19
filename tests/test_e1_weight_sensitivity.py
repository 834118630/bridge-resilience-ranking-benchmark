from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import e1.run_metric_weight_sensitivity as weight_module
from e1.run_metric_weight_sensitivity import METRICS


def test_weighted_regret_changes_winner_with_weights():
    values = {
        "resilience_index": {"A": 0.0, "B": 1.0},
        "functionality_day_30": {"A": 1.0, "B": 0.0},
        "functionality_day_90": {"A": 0.0, "B": 1.0},
        "recovery_rapidity_90": {"A": 0.0, "B": 1.0},
    }
    original = weight_module.CANDIDATES
    try:
        weight_module.CANDIDATES = ("A", "B")
        equal = weight_module.weighted_regret(
            values, {metric: 0.25 for metric in METRICS}
        )
        early = weight_module.weighted_regret(
            values,
            {
                "resilience_index": 0.1,
                "functionality_day_30": 0.7,
                "functionality_day_90": 0.1,
                "recovery_rapidity_90": 0.1,
            },
        )
    finally:
        weight_module.CANDIDATES = original
    assert np.argmin(equal) == 0
    assert np.argmin(early) == 1
