"""Capacity-speed-cost break-even analysis for the E1 benchmark.

The analysis uses discounted equivalent functionality-loss days and reports
break-even social value per lost functionality day (v_day*), normalized by the
L4 reference cost. Cost-capacity-speed correlation is represented by explicit
elasticity scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .monte_carlo import (
    candidate_recovery_times,
    damage_state_probabilities_batch,
    expected_functionality_at,
    recovery_parameter_arrays,
    sample_recovery_days_matrix,
)
from .run_monte_carlo import load_beta
from .run_pilot import (
    DAMAGE_STATES,
    DATA_DIR,
    PROJECT_ROOT,
    RECOVERY_MODELS,
    SA_LEVELS,
    interpolate_medians,
    load_fragility,
    load_recovery,
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "e1_cost_break_even"
CAPACITY_GRID = [0.25, 0.50, 0.75]
SPEED_GRID = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
GAMMA_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
ALPHA_SCENARIOS = {
    "independent": (0.0, 0.0),
    "moderate": (0.5, 0.5),
    "strong": (1.0, 1.0),
}
WINDOW_DAYS = 500.0
DISCOUNT_RATE = 0.03
REFERENCE_VDay = 1e-3
ECONOMIC_VDay_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
L4_MULTIPLIER = 1.75


def expected_curves(
    time_days: np.ndarray,
    recovery_times: np.ndarray,
    probabilities: np.ndarray,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Return time x sample x candidate x Sa expected functionality."""

    curves = []
    for time_value in time_days:
        curves.append(
            expected_functionality_at(
                float(time_value),
                recovery_times,
                probabilities,
                q0,
                qend,
                model_name,
            )
        )
    return np.stack(curves, axis=0)


def discounted_loss_days(
    time_days: np.ndarray,
    curves: np.ndarray,
    discount_rate: float,
    window_days: float,
) -> np.ndarray:
    """Integrate discounted lost functionality to a finite window."""

    mask = time_days <= window_days
    selected_time = time_days[mask]
    selected_curves = curves[mask]
    discount = np.exp(-discount_rate * selected_time / 365.0)
    deficit = (1.0 - selected_curves) * discount[:, None, None, None]
    return np.trapezoid(deficit, selected_time, axis=0)


def evaluate_point(
    capacity: float,
    speed_multiplier: float,
    base_days: np.ndarray,
    conventional: list[float],
    seismic: list[float],
    beta: float,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    time_days: np.ndarray,
    discount_rate: float,
    window_days: float,
) -> tuple[float, float]:
    sample_loss = evaluate_point_sample_losses(
        capacity,
        speed_multiplier,
        base_days,
        conventional,
        seismic,
        beta,
        q0,
        qend,
        model_name,
        time_days,
        discount_rate,
        window_days,
    )
    mean_loss = sample_loss.mean(axis=0)
    return float(mean_loss[0]), float(mean_loss[1])


def evaluate_point_sample_losses(
    capacity: float,
    speed_multiplier: float,
    base_days: np.ndarray,
    conventional: list[float],
    seismic: list[float],
    beta: float,
    q0: np.ndarray,
    qend: np.ndarray,
    model_name: str,
    time_days: np.ndarray,
    discount_rate: float,
    window_days: float,
) -> np.ndarray:
    """Return sample-level mean loss days for L4 and the fast profile."""

    medians = np.asarray(
        [
            interpolate_medians(conventional, seismic, 1.0),
            interpolate_medians(conventional, seismic, capacity),
        ],
        dtype=float,
    )
    probabilities = damage_state_probabilities_batch(medians, beta, SA_LEVELS)
    recovery_times = candidate_recovery_times(
        base_days,
        [L4_MULTIPLIER, speed_multiplier],
    )
    curves = expected_curves(
        time_days,
        recovery_times,
        probabilities,
        q0,
        qend,
        model_name,
    )
    loss = discounted_loss_days(time_days, curves, discount_rate, window_days)
    return loss.mean(axis=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--time-step-days", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fragility_path = DATA_DIR / "07_hazus_fragility.csv"
    recovery_path = DATA_DIR / "08_hazus_recovery.csv"
    fragility = load_fragility(fragility_path)
    recovery = load_recovery(recovery_path)
    beta = load_beta(fragility_path)
    q0, qend = recovery_parameter_arrays(recovery)
    rng = np.random.default_rng(args.seed)
    base_days = sample_recovery_days_matrix(recovery, args.samples, rng)
    time_days = np.arange(0.0, 730.0 + args.time_step_days, args.time_step_days)
    case_pairs = {"A": ("HWB5", "HWB7"), "B": ("HWB10", "HWB11")}
    rows = []

    loss_table: dict[tuple[float, float], tuple[float, float]] = {}
    for capacity in CAPACITY_GRID:
        for speed_multiplier in SPEED_GRID:
            values = []
            for case_id, (conventional_class, seismic_class) in case_pairs.items():
                conventional = [fragility[(case_id, conventional_class, state)] for state in DAMAGE_STATES]
                seismic = [fragility[(case_id, seismic_class, state)] for state in DAMAGE_STATES]
                for model_name in RECOVERY_MODELS:
                    values.append(
                        evaluate_point(
                            capacity,
                            speed_multiplier,
                            base_days,
                            conventional,
                            seismic,
                            beta,
                            q0,
                            qend,
                            model_name,
                            time_days,
                            DISCOUNT_RATE,
                            WINDOW_DAYS,
                        )
                    )
            loss_table[(capacity, speed_multiplier)] = (
                float(np.mean([value[0] for value in values])),
                float(np.mean([value[1] for value in values])),
            )

    for scenario_name, (alpha_capacity, alpha_speed) in ALPHA_SCENARIOS.items():
        for capacity in CAPACITY_GRID:
            for speed_multiplier in SPEED_GRID:
                loss_l4, loss_fast = loss_table[(capacity, speed_multiplier)]
                delta_loss = loss_l4 - loss_fast
                for gamma in GAMMA_GRID:
                    cost_l4 = 1.0
                    cost_fast = gamma * np.exp(
                        -alpha_capacity * (1.0 - capacity)
                        + alpha_speed * (L4_MULTIPLIER - speed_multiplier)
                    )
                    delta_cost = float(cost_fast - cost_l4)
                    if delta_loss > 0.0 and delta_cost <= 0.0:
                        classification = "F_dominates"
                        v_day_star = 0.0
                    elif delta_loss > 0.0 and delta_cost > 0.0:
                        classification = "positive_breakeven"
                        v_day_star = float(delta_cost / delta_loss)
                    elif delta_loss <= 0.0 and delta_cost >= 0.0:
                        classification = "L4_dominates"
                        v_day_star = float("inf")
                    else:
                        classification = "ambiguous_tradeoff"
                        v_day_star = float("inf")
                    net_benefit = float(REFERENCE_VDay * delta_loss - delta_cost)
                    rows.append(
                        {
                            "cost_scenario": scenario_name,
                            "alpha_capacity": alpha_capacity,
                            "alpha_speed": alpha_speed,
                            "capacity": capacity,
                            "speed_multiplier": speed_multiplier,
                            "gamma": gamma,
                            "loss_days_l4": loss_l4,
                            "loss_days_fast": loss_fast,
                            "delta_loss_days": delta_loss,
                            "cost_l4": cost_l4,
                            "cost_fast": cost_fast,
                            "delta_cost": delta_cost,
                            "v_day_star": v_day_star,
                            "net_benefit_at_reference_v_day": net_benefit,
                            "classification": classification,
                        }
                    )

    with (output_dir / "cost_break_even_grid.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "reference_v_day": REFERENCE_VDay,
        "economic_v_day_grid": list(ECONOMIC_VDay_GRID),
        "discount_rate": DISCOUNT_RATE,
        "window_days": WINDOW_DAYS,
    }
    for scenario_name in ALPHA_SCENARIOS:
        scenario_rows = [row for row in rows if row["cost_scenario"] == scenario_name]
        economic = sum(row["net_benefit_at_reference_v_day"] > 0 for row in scenario_rows)
        performance = sum(row["delta_loss_days"] > 0 for row in scenario_rows)
        finite_breakeven = [
            row["v_day_star"]
            for row in scenario_rows
            if np.isfinite(row["v_day_star"]) and row["v_day_star"] > 0.0
        ]
        summary[scenario_name] = {
            "performance_dominance_fraction": performance / len(scenario_rows),
            "economic_dominance_fraction_at_reference_v_day": economic / len(scenario_rows),
            "economic_dominance_fraction_by_reference_v_day": {
                f"{value:.0e}": sum(
                    value * row["delta_loss_days"] - row["delta_cost"] > 0.0
                    for row in scenario_rows
                )
                / len(scenario_rows)
                for value in ECONOMIC_VDay_GRID
            },
            "positive_breakeven_v_day_quantiles": {
                f"p{int(quantile * 100)}": float(np.quantile(finite_breakeven, quantile))
                for quantile in (0.10, 0.25, 0.50, 0.75, 0.90)
            }
            if finite_breakeven
            else {},
            "classification_counts": {
                label: sum(row["classification"] == label for row in scenario_rows)
                for label in ("F_dominates", "positive_breakeven", "L4_dominates", "ambiguous_tradeoff")
            },
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    # Slice plots at each capacity for the moderate correlation scenario.
    for scenario_name in ("independent", "moderate"):
        for capacity in CAPACITY_GRID:
            matrix = np.full((len(GAMMA_GRID), len(SPEED_GRID)), np.nan)
            for row in rows:
                if (
                    row["cost_scenario"] == scenario_name
                    and row["capacity"] == capacity
                ):
                    gi = GAMMA_GRID.index(row["gamma"])
                    si = SPEED_GRID.index(row["speed_multiplier"])
                    matrix[gi, si] = row["net_benefit_at_reference_v_day"]
            figure, axis = plt.subplots(figsize=(8.5, 4.2))
            image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="RdBu_r")
            axis.set_xticks(range(len(SPEED_GRID)), [f"{value:g}" for value in SPEED_GRID])
            axis.set_yticks(range(len(GAMMA_GRID)), [f"{value:g}" for value in GAMMA_GRID])
            axis.set_xlabel("Fast-recovery multiplier")
            axis.set_ylabel("Cost premium gamma")
            axis.set_title(f"{scenario_name}: capacity={capacity:.2f}, net benefit at v_day={REFERENCE_VDay:g}")
            figure.colorbar(image, ax=axis, label="F minus L4 net benefit")
            figure.tight_layout()
            figure.savefig(figure_dir / f"{scenario_name}_capacity_{capacity:.2f}_cost_slice.png", dpi=220)
            plt.close(figure)

    # Interaction overview at gamma=1.25 for moderate scenario.
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for capacity in CAPACITY_GRID:
        selected = [
            row
            for row in rows
            if row["cost_scenario"] == "moderate"
            and row["capacity"] == capacity
            and row["gamma"] == 1.25
        ]
        selected.sort(key=lambda row: row["speed_multiplier"])
        y = [
            row["v_day_star"] if np.isfinite(row["v_day_star"]) else np.nan
            for row in selected
        ]
        axis.plot(
            [row["speed_multiplier"] for row in selected],
            y,
            marker="o",
            label=f"capacity={capacity:.2f}",
        )
    axis.set_xlabel("Fast-recovery multiplier")
    axis.set_ylabel("Break-even v_day*")
    axis.set_title("Moderate cost correlation, gamma=1.25")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "moderate_gamma1.25_v_day_interaction.png", dpi=240)
    plt.close(figure)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
