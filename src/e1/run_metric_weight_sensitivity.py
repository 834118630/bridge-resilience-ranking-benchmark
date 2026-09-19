"""Metric-weight sensitivity for the archived mean-regret estimand."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

from .provenance import sha256_file, write_run_metadata

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tmp" / "e1_profile_monte_carlo" / "metric_regret_summary.csv"
OUTPUT_DIR = ROOT / "tmp" / "e1_metric_weight_sensitivity"
METRICS = (
    "resilience_index",
    "functionality_day_30",
    "functionality_day_90",
    "recovery_rapidity_90",
)
CANDIDATES = ("L0", "L1", "L2", "L3", "L4", "F_fast")
WEIGHT_SCHEMES = {
    "equal": {metric: 0.25 for metric in METRICS},
    "rapidity_emphasis": {
        "resilience_index": 0.20,
        "functionality_day_30": 0.20,
        "functionality_day_90": 0.20,
        "recovery_rapidity_90": 0.40,
    },
    "resilience_emphasis": {
        "resilience_index": 0.40,
        "functionality_day_30": 0.20,
        "functionality_day_90": 0.20,
        "recovery_rapidity_90": 0.20,
    },
    "early_functionality_emphasis": {
        "resilience_index": 0.20,
        "functionality_day_30": 0.40,
        "functionality_day_90": 0.20,
        "recovery_rapidity_90": 0.20,
    },
}


def load_rows(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    values: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["scenario"], {}).setdefault(row["metric"], {})[row["candidate"]] = float(
                row["mean_regret"]
            )
    return values


def weighted_regret(
    scenario_values: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            sum(
                weights[metric] * scenario_values[metric][candidate]
                for metric in METRICS
            )
            for candidate in CANDIDATES
        ],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.draws <= 0:
        raise ValueError("draws must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_values = load_rows(args.input)
    deterministic_rows: list[dict[str, object]] = []
    alert_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)

    for scenario, metric_values in scenario_values.items():
        for scheme, weights in WEIGHT_SCHEMES.items():
            regrets = weighted_regret(metric_values, weights)
            winner = CANDIDATES[int(np.argmin(regrets))]
            for index, candidate in enumerate(CANDIDATES):
                deterministic_rows.append(
                    {
                        "scenario": scenario,
                        "weight_scheme": scheme,
                        "candidate": candidate,
                        "weighted_mean_regret": float(regrets[index]),
                        "top1_candidate": winner,
                    }
                )

        weights = rng.dirichlet(np.ones(len(METRICS)), size=args.draws)
        top1_counts = {candidate: 0 for candidate in CANDIDATES}
        for draw in weights:
            weight_map = {metric: float(draw[index]) for index, metric in enumerate(METRICS)}
            regrets = weighted_regret(metric_values, weight_map)
            top1_counts[CANDIDATES[int(np.argmin(regrets))]] += 1
        for candidate in CANDIDATES:
            alert_rows.append(
                {
                    "scenario": scenario,
                    "candidate": candidate,
                    "top1_frequency": top1_counts[candidate] / args.draws,
                }
            )

    with (output_dir / "metric_weight_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deterministic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(deterministic_rows)

    with (output_dir / "metric_weight_winner_frequency.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alert_rows[0].keys()))
        writer.writeheader()
        writer.writerows(alert_rows)

    summary = {
        "input": str(args.input),
        "draws": args.draws,
        "seed": args.seed,
        "weight_schemes": WEIGHT_SCHEMES,
        "dirichlet_prior": [1.0] * len(METRICS),
        "interpretation": "weight sensitivity is post-hoc robustness, not a calibrated prior",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_run_metadata(
        output_dir,
        command="python -m e1.run_metric_weight_sensitivity "
        f"--draws {args.draws} --seed {args.seed} "
        f"--input {args.input} --output-dir {output_dir}",
        parameters={
            "draws": args.draws,
            "seed": args.seed,
            "weight_schemes": WEIGHT_SCHEMES,
        },
        input_paths=(args.input,),
    )
    print(json.dumps({"output": str(output_dir), "scenarios": list(scenario_values)}, indent=2))


if __name__ == "__main__":
    main()
