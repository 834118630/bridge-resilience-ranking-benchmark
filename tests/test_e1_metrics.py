from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e1.metrics import functionality_curve, resilience_index
from e1.ranking import pairwise_reversal_rate, top1_stable
from e1.recovery_models import RECOVERY_MODELS


def test_recovery_endpoints():
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for model in RECOVERY_MODELS.values():
        result = model(values)
        assert result[0] == 0.0
        assert np.isclose(result[-1], 1.0)


def test_linear_resilience_index_is_half():
    time = np.linspace(0.0, 10.0, 1001)
    q = functionality_curve(time, recovery_days=10.0, initial_functionality=0.0, final_functionality=1.0, model_name="linear")
    assert np.isclose(resilience_index(time, q), 0.5, atol=1e-6)


def test_identical_rankings_have_no_reversal():
    ranking = ["A", "B", "C"]
    assert top1_stable(ranking, ranking)
    assert pairwise_reversal_rate([ranking, ranking]) == 0.0


def test_expected_functionality_never_exceeds_one():
    from e1.run_pilot import expected_functionality, load_recovery, DATA_DIR
    time = np.arange(0.0, 501.0, 1.0)
    recovery = load_recovery(DATA_DIR / '08_hazus_recovery.csv')
    probabilities = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    curve = expected_functionality(time, probabilities, recovery, 1.0, 'linear')
    assert np.all(curve >= 0.0)
    assert np.all(curve <= 1.0)

def test_damage_state_probabilities_sum_to_one():
    from e1.run_pilot import damage_state_probabilities
    probabilities = damage_state_probabilities([0.25, 0.35, 0.45, 0.70], 0.6, 0.6)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0.0)


def test_expected_functionality_is_monotone_with_capacity():
    from e1.run_pilot import CAPACITY_LEVELS, DAMAGE_STATES, DATA_DIR
    from e1.run_pilot import damage_state_probabilities, expected_functionality
    from e1.run_pilot import interpolate_medians, load_fragility, load_recovery
    fragility = load_fragility(DATA_DIR / '07_hazus_fragility.csv')
    recovery = load_recovery(DATA_DIR / '08_hazus_recovery.csv')
    conventional = [fragility[('A', 'HWB5', state)] for state in DAMAGE_STATES]
    seismic = [fragility[('A', 'HWB7', state)] for state in DAMAGE_STATES]
    time = np.arange(0.0, 501.0, 1.0)
    curves = []
    for capacity in CAPACITY_LEVELS:
        medians = interpolate_medians(conventional, seismic, capacity)
        probabilities = damage_state_probabilities(medians, 0.6, 0.8)
        curves.append(expected_functionality(time, probabilities, recovery, 1.0, 'linear'))
    for lower, higher in zip(curves, curves[1:]):
        assert np.all(higher >= lower - 1e-12)
