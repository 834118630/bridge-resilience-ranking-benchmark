# Bridge resilience ranking benchmark

Code and aggregate results for a controlled ranking and cost sensitivity benchmark for
seismic resilience assessment of highway bridges.

The study fixes recovery endpoints and completion time, compares four normalized
recovery-function shapes, evaluates a six-candidate capacity and recovery-speed set,
extends the comparison to a normalized performance-cost surface, and tests ranking
stability under paired and independent recovery-duration sampling.

> **Scope of this benchmark.** This repository is the case-study benchmark for the controlled comparison reported in the companion paper. It archives the inputs, the analysis code, the aggregate results, and the publication figures so that the reported numbers can be re-run and audited. It is not a plug-in evaluation harness. The candidate set, scenario multipliers, metrics, and intensity levels are defined as constants in the source rather than registered through a configuration file. See `docs/EXTENDING.md` for the exact constants to edit if you want to reuse the code with a different design.

## Repository layout

| Path | Contents |
|---|---|
| `src/e1/` | Analysis code: Monte Carlo runners, cost frontier, sampling robustness, figure generation |
| `tests/` | Unit tests for metrics, Monte Carlo, cost frontier, sampling modes, uncertainty |
| `data/` | HAZUS 6.1 input parameter tables (fragility medians, restoration durations) |
| `configs/` | Example storage configuration |
| `results/` | Aggregate outputs that back the figures and tables of the paper |
| `figures/` | Publication figures (PNG at 600 dpi and vector PDF) |
| `styles/` | Matplotlib style sheet used for the figures |
| `docs/` | Data sources and step by step reproduction notes |

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

## Run the tests

```
python -m pytest -q
```

Expected outcome: 23 passed. Two tests read the HAZUS parameter tables in `data/`.

## Reproduce the figures

The figure script reads only from `results/` and writes to `figures/`.

```
python src/e1/make_publication_figures.py
```

Verified behaviour on the archived inputs: the six PNG figures are regenerated
byte for byte. The PDF figures differ only in the embedded PDF creation timestamp.

## Mapping from code to the manuscript

| Script | Output directory | Used for |
|---|---|---|
| `run_profile_monte_carlo.py` | `results/e1_profile_monte_carlo/` | Figure 1, Figure 2, Sections 4.1 and 4.2 |
| `run_profile_phase_diagram.py` | `results/e1_profile_phase_diagram/` | Figure 3 |
| `run_candidate_cost_frontier.py` | `results/e1_candidate_cost_frontier_final/` | Figure 4, Figure 5, Sections 4.4 and 4.5 |
| `run_independent_sampling_robustness.py` | `results/e1_independent_sampling_final/` | Figure 6, Table 2, Section 4.6 |
| `run_monte_carlo.py` | `results/e1_monte_carlo/` | Bootstrap and uncertainty summaries |
| `make_publication_figures.py` | `figures/` | All six publication figures |
| `run_regret_variants.py` | `results/e1_regret_variants/` | Regret variant comparison |

Other runners in `src/e1/` are supporting or historical analyses. Their outputs are not
archived here.

## IMPORTANT: two different quantities are both called Top-1 frequency

Anyone reading or reusing these results must know this. The repository contains two
different Top-1 frequency definitions and the archived outputs do not agree numerically
for the same scenario labels.

| Source | Definition | Scope |
|---|---|---|
| `run_profile_monte_carlo.py` | Per draw, the candidate with the maximum metric value (`numpy.argmax`). Counts are summed over all metrics in `METRIC_NAMES` and all `RECOVERY_MODELS`, then divided by the total number of draws. | Aggregated across metrics and recovery models |
| `run_independent_sampling_robustness.py` | Per draw, the candidate with the minimum regret (`numpy.argmin`), using the mean-regret metric only. | Single metric, per draw |

For the evidence tradeoff scenario the first definition gives an F_fast Top-1 frequency of
0.871 while the second gives 1.0000. Both are correct for their own definition. Any
comparison between `results/e1_profile_monte_carlo/candidate_summary.csv` and
`results/e1_independent_sampling_final/sampling_gate_comparison.csv` must name which
definition is meant.

## Data sources

All fragility and restoration parameters come from the publicly available Hazus
Earthquake Model Technical Manual 6.1 published by the US Federal Emergency Management
Agency in 2024. See `docs/DATA_SOURCES.md`. No proprietary or restricted data are used.

## Scope

The benchmark is synthetic and controlled. The fast-recovery profile is an abstract
stress test profile and not a specific retrofit technology. Costs are normalized indices
and not monetary amounts. The cost correlation coefficients are scenario parameters and
not calibrated elasticities. Results apply only to the tested bridge classes, damage
states, intensity levels, recovery parameters, candidate profiles, and cost scenarios.

## License

MIT. See `LICENSE`.
