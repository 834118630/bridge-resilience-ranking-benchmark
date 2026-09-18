from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e1.run_candidate_cost_frontier import (
    candidate_cost_index,
    candidate_cost_vector,
    equivalent_groups,
    pareto_flags,
)


def test_l4_cost_is_one_for_both_cost_forms():
    for cost_form in ("linear", "exponential"):
        assert np.isclose(
            candidate_cost_index(cost_form, alpha=0.5, theta=1.0, capacity=1.0, multiplier=1.75),
            1.0,
        )


def test_linear_cost_direction_matches_capacity_and_speed():
    lower_capacity = candidate_cost_index(
        "linear", alpha=0.5, theta=1.0, capacity=0.25, multiplier=1.75
    )
    higher_capacity = candidate_cost_index(
        "linear", alpha=0.5, theta=1.0, capacity=1.0, multiplier=1.75
    )
    faster_recovery = candidate_cost_index(
        "linear", alpha=0.5, theta=1.0, capacity=0.5, multiplier=0.10
    )
    slower_recovery = candidate_cost_index(
        "linear", alpha=0.5, theta=1.0, capacity=0.5, multiplier=1.75
    )
    assert lower_capacity < higher_capacity
    assert faster_recovery > slower_recovery


def test_candidate_cost_vector_applies_theta_only_to_fast_profile():
    capacities = np.array([0.5, 1.0])
    multipliers = np.array([0.1, 1.75])
    costs = candidate_cost_vector(
        "linear",
        alpha=0.5,
        theta_f=1.5,
        capacities=capacities,
        multipliers=multipliers,
        labels=["F_fast", "L4"],
    )
    assert costs[1] == 1.0
    assert np.isclose(costs[0], 1.5 * (1.0 - 0.5 * 0.5 + 0.5 * 1.65))


def test_pareto_flags_remove_dominated_candidate():
    costs = np.array([1.0, 1.2, 0.8, 1.5])
    losses = np.array([10.0, 12.0, 12.0, 15.0])
    flags = pareto_flags(costs, losses)
    assert flags.tolist() == [True, False, True, False]


def test_equivalent_groups_detect_identical_points():
    costs = np.array([1.0, 1.0, 1.2])
    losses = np.array([10.0, 10.0, 12.0])
    assert equivalent_groups(costs, losses) == ["G1", "G1", "G2"]
