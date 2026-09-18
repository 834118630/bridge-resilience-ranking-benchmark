# Changing the design

This repository runs the design described in the companion paper. The design
elements are Python constants rather than configuration entries. This page lists
exactly what to edit if you want to reuse the code with a different candidate set,
different inputs, or a different metric.

After any change, re-run the affected runner and then
`python src/e1/make_publication_figures.py`. The tests in `tests/` cover the
metric and Monte Carlo primitives and should still pass.

## Where each design element lives

| Element | File | Constant or function |
|---|---|---|
| Candidate labels | `src/e1/run_profile_monte_carlo.py` | `PROFILE_LABELS` |
| Candidate capacity indices | `src/e1/run_profile_monte_carlo.py` | `PROFILE_CAPACITIES` |
| Scenario recovery multipliers | `src/e1/run_profile_monte_carlo.py` | `PROFILE_SCENARIOS` |
| Resilience metrics | `src/e1/monte_carlo.py` | `METRIC_NAMES`, and the matching entries returned by `metrics_batch` |
| Intensity levels | `src/e1/monte_carlo.py` | `SA_LEVELS` |
| Damage states | `src/e1/monte_carlo.py` | `DAMAGE_STATES` |
| Recovery shapes | `src/e1/recovery_models.py` | the four functions and `RECOVERY_MODELS` |
| Exponential rate | `src/e1/recovery_models.py` | `exponential(..., rate=5.0)` |
| Delayed-recovery delay fraction | `src/e1/recovery_models.py` | `delayed_recovery(..., delay_fraction=0.2)` |
| Fragility medians and dispersions | `data/07_hazus_fragility.csv` | replace the file, keep the column names |
| Restoration durations and day-1 proxies | `data/08_hazus_recovery.csv` | replace the file, keep the column names |
| Cost surface coefficients | `src/e1/run_candidate_cost_frontier.py` | the scenario table inside `main()` |
| F_fast cost multiplier scan | `src/e1/run_candidate_cost_frontier.py` | the `theta_f` values used to build the grid |

`PROFILE_LABELS`, `PROFILE_CAPACITIES`, and every row of `PROFILE_SCENARIOS`
must have the same length. The metrics named in `METRIC_NAMES` must all be
oriented so that larger values are better, because regret uses `argmax` to
identify the best candidate within a scene.

## Adding a metric

1. Add the name to `METRIC_NAMES` in `src/e1/monte_carlo.py`.
2. Return an array for it from `metrics_batch`, with shape
   `(samples, candidates, intensity levels)`.
3. Make sure larger values are better. If a natural metric is smaller-is-better,
   negate it or invert it before returning.
4. Re-run the runners. Note that adding a metric changes the denominator of the
   metric-aggregated Top-1 frequency described in the README, because that
   frequency is aggregated over all metrics.

## Adding a candidate

1. Append a label to `PROFILE_LABELS`.
2. Append its capacity index to `PROFILE_CAPACITIES`.
3. Append its recovery multiplier to every row of `PROFILE_SCENARIOS`.
4. Check `CANDIDATES` and any fixed palette in `src/e1/make_publication_figures.py`
   so that the new candidate has a colour and marker.

## Replacing the input data

Both CSVs under `data/` are read by `src/e1/run_pilot.py` through `DATA_DIR`.
Keep the existing column names and the `case_id` / `bridge_class` convention.
`07_hazus_fragility.csv` needs one row per case, bridge class, and damage state.
`08_hazus_recovery.csv` needs one row per damage state.

## What is not parameterised

The runner entry points accept `--samples`, `--seed`, `--time-step-days`,
`--bootstrap-resamples`, and `--output-dir`. Nothing else is exposed on the
command line, and there is no configuration file for the design elements above.
