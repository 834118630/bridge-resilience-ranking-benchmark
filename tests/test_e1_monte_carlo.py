from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e1.metrics import functionality_curve, resilience_index, time_to_target
from e1.monte_carlo import (
    cumulative_shape_integral,
    damage_state_probabilities_batch,
    linear_metric_values,
    metrics_batch,
    pairwise_metric_differences,
    ranking_stability_statistics,
    recovery_parameter_arrays,
)
from e1.recovery_models import RECOVERY_MODELS
from e1.run_pilot import DAMAGE_STATES, damage_state_probabilities


def test_batch_probabilities_match_reference():
    medians = np.array(
        [
            [0.25, 0.35, 0.45, 0.70],
            [0.50, 0.80, 1.10, 1.70],
        ]
    )
    sa_levels = [0.2, 0.6]
    batch = damage_state_probabilities_batch(medians, 0.6, sa_levels)
    for candidate in range(medians.shape[0]):
        for sa_index, sa in enumerate(sa_levels):
            reference = damage_state_probabilities(medians[candidate], 0.6, sa)
            assert np.allclose(batch[candidate, sa_index], reference)


def _reference_metrics(recovery_times, probabilities, q0, qend, model_name):
    time = np.arange(0.0, 500.0 + 0.05, 0.05)
    n_samples, n_candidates, _ = recovery_times.shape
    n_sa = probabilities.shape[1]
    values = np.empty((n_samples, n_candidates, n_sa), dtype=float)
    for sample in range(n_samples):
        for candidate in range(n_candidates):
            for sa_index in range(n_sa):
                curve = probabilities[candidate, sa_index, 0] * np.ones_like(time)
                for state_index, _ in enumerate(DAMAGE_STATES):
                    curve += probabilities[candidate, sa_index, state_index + 1] * functionality_curve(
                        time,
                        recovery_times[sample, candidate, state_index],
                        q0[state_index],
                        qend[state_index],
                        model_name,
                    )
                values[sample, candidate, sa_index] = resilience_index(time, curve)
    return values


def test_batch_resilience_matches_curve_integration():
    recovery = {
        "Slight": {"mean_days": 0.6, "sd_days": 0.6, "q0": 0.70, "qend": 1.0},
        "Moderate": {"mean_days": 2.5, "sd_days": 2.7, "q0": 0.30, "qend": 1.0},
        "Extensive": {"mean_days": 75.0, "sd_days": 42.0, "q0": 0.02, "qend": 1.0},
        "Complete": {"mean_days": 230.0, "sd_days": 110.0, "q0": 0.0, "qend": 1.0},
    }
    q0, qend = recovery_parameter_arrays(recovery)
    base_days = np.array(
        [
            [0.5, 2.0, 70.0, 210.0],
            [0.8, 3.0, 90.0, 260.0],
        ]
    )
    multipliers = [1.0, 1.2]
    recovery_times = base_days[:, None, :] * np.asarray(multipliers)[None, :, None]
    probabilities = np.tile(np.array([[[0.10, 0.20, 0.30, 0.25, 0.15]]]), (2, 2, 1))
    for model_name in RECOVERY_MODELS:
        actual = metrics_batch(
            recovery_times,
            probabilities,
            q0,
            qend,
            model_name,
            time_window_days=500.0,
        )["resilience_index"]
        expected = _reference_metrics(
            recovery_times,
            probabilities,
            q0,
            qend,
            model_name,
        )
        assert np.allclose(actual, expected, atol=1e-5, rtol=1e-6)


def test_batch_time_to_target_matches_curve_search():
    q0 = np.array([0.2])
    qend = np.array([1.0])
    base_days = np.array([[10.0], [50.0]])
    recovery_times = base_days[:, :, None]
    probabilities = np.array([[[0.0, 1.0]]])
    time = np.arange(0.0, 500.0 + 0.01, 0.01)
    for sample in range(2):
        for model_name in RECOVERY_MODELS:
            curve = functionality_curve(
                time,
                recovery_times[sample, 0, 0],
                q0[0],
                qend[0],
                model_name,
            )
            expected = 1.0 - time_to_target(time, curve, 0.9) / 500.0
            actual = metrics_batch(
                recovery_times[sample : sample + 1],
                probabilities,
                q0,
                qend,
                model_name,
            )["recovery_rapidity_90"][0, 0, 0]
            assert np.isclose(actual, expected, atol=0.01)


def test_ranking_stability_detects_complete_reversal():
    metric_a = np.array([[[2.0], [1.0]]])
    metric_b = np.array([[[1.0], [2.0]]])
    stats = ranking_stability_statistics({"linear": metric_a, "exponential": metric_b}, k=2)
    assert np.isclose(stats["top1_change_rate"][0, 0], 1.0)
    assert np.isclose(stats["pairwise_reversal_rate"][0, 0], 1.0)


def test_linear_cumulative_integral_at_one():
    u, cumulative = cumulative_shape_integral("linear")
    assert np.isclose(u[-1], 1.0)
    assert np.isclose(cumulative[-1], 0.5, atol=1e-8)


def test_linear_metric_values_match_metrics_batch():
    recovery = {
        "Slight": {"mean_days": 0.6, "sd_days": 0.6, "q0": 0.70, "qend": 1.0},
        "Moderate": {"mean_days": 2.5, "sd_days": 2.7, "q0": 0.30, "qend": 1.0},
        "Extensive": {"mean_days": 75.0, "sd_days": 42.0, "q0": 0.02, "qend": 1.0},
        "Complete": {"mean_days": 230.0, "sd_days": 110.0, "q0": 0.0, "qend": 1.0},
    }
    q0, qend = recovery_parameter_arrays(recovery)
    base_days = np.array(
        [
            [0.5, 2.0, 70.0, 210.0],
            [0.8, 3.0, 90.0, 260.0],
        ]
    )
    recovery_times = base_days[:, None, :] * np.asarray([1.0, 1.2])[None, :, None]
    probabilities = np.tile(
        np.array([[[0.10, 0.20, 0.30, 0.25, 0.15]]]),
        (2, 2, 1),
    )
    for model_name in RECOVERY_MODELS:
        batch = metrics_batch(
            recovery_times,
            probabilities,
            q0,
            qend,
            model_name,
            time_window_days=500.0,
        )
        for metric_name in (
            "resilience_index",
            "functionality_day_30",
            "functionality_day_90",
        ):
            decomposed = linear_metric_values(
                probabilities,
                recovery_times,
                q0,
                qend,
                model_name,
                metric_name,
                time_window_days=500.0,
            )
            assert np.allclose(decomposed, batch[metric_name], atol=1e-10, rtol=1e-10)


def test_pairwise_metric_differences_match_manual_difference():
    values = np.array(
        [
            [[1.0, 0.8], [0.5, 0.4], [0.2, 0.1]],
            [[0.9, 0.7], [0.6, 0.5], [0.1, 0.0]],
        ]
    )
    actual = pairwise_metric_differences(values)
    assert actual.shape == (2, 3, 2)
    assert np.allclose(actual[0, 0], values[0, 0] - values[0, 1])
    assert np.allclose(actual[1, 2], values[1, 1] - values[1, 2])
