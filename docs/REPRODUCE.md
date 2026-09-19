# Reproduction notes

## Environment

Python 3.12. The archived results were produced with NumPy, pandas,
Matplotlib 3.11.2, SciPy, and pytest.

## Fast check

The figure script reads the archived aggregate results and does not rerun the
simulation.

```
python src/e1/make_publication_figures.py
```

Expected outcome on the archived inputs: the six PNG files in `figures/` are
reproduced byte for byte. The PDF and EPS files are regenerated with the same
content; embedded creation timestamps may differ.

Run the integration test for this exact step with:

```
set RUN_INTEGRATION=1
python -m pytest -q -m integration
```

On Linux or macOS, use `export RUN_INTEGRATION=1`.

## Unit tests

```
python -m pytest -q -m "not integration"
```

Expected outcome: 26 passed and 1 deselected.

## Re-running the analyses from scratch

The runners use relative imports, so they must be invoked as modules with the
`src` directory on the import path, not as file paths. From the repository root,
set `PYTHONPATH=src` on Windows or `export PYTHONPATH=src` on Linux and macOS.

Run the commands in this order. The first command writes the primary candidate
summary used by the cost-frontier analysis.

```
python -m e1.run_profile_monte_carlo --samples 10000 --seed 20260914 --bootstrap-resamples 10000 --recovery-lower-bound 0.05 --recovery-distribution truncnorm --output-dir results/e1_profile_monte_carlo
python -m e1.run_profile_phase_diagram --recovery-samples 1000 --seed 20260916 --output-dir results/e1_profile_phase_diagram
python -m e1.run_candidate_cost_frontier --samples 10000 --seed 20260919 --time-step-days 2.0 --bootstrap-resamples 10000 --output-dir results/e1_candidate_cost_frontier_final
python -m e1.run_independent_sampling_robustness --samples 10000 --seed 20260919 --time-step-days 2.0 --output-dir results/e1_independent_sampling_final
python -m e1.run_alpha_sensitivity --theta-f 1.0 --grid 0.0 0.25 0.5 0.75 1.0 --output-dir results/e1_alpha_sensitivity
python -m e1.run_recovery_distribution_sensitivity --samples 10000 --seed 20260914 --output-dir results/e1_recovery_distribution_sensitivity
python -m e1.run_monte_carlo
```

The six commands with explicit `--output-dir` reproduce the archived aggregate
results used by the manuscript. `run_monte_carlo` is retained as the original
uncertainty-summary workflow.

## Run metadata

Each final output directory contains `run_metadata.json` with:

- the command and parameters;
- input and output SHA256 hashes;
- Python, NumPy, and SciPy versions;
- platform information;
- completion and failure status.

## Known caveat

`run_profile_monte_carlo.py` and `run_independent_sampling_robustness.py`
compute two different Top-1 frequency definitions. The first uses
metric-aggregated draws and fractional credit for exact ties. The second uses
mean regret and one winner per replication. See the top-level README before
comparing their outputs.
