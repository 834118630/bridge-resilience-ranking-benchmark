# Reproduction notes

## Environment

Python 3.12. The archived results were produced with:
numpy, pandas, matplotlib 3.11.2, scipy, pytest.

## Fast check

The figure script reads the archived aggregate results and needs no simulation run.

```
python src/e1/make_publication_figures.py
```

Expected outcome on the archived inputs: the six PNG files in `figures/` are
reproduced byte for byte. The PDF files are reproduced with the same content and size;
only the embedded PDF creation timestamp changes.

## Unit tests

```
python -m pytest -q
```

## Re-running the analyses from scratch

Each runner writes its full output tree under `results/`. The archived directories in
this repository are the aggregate outputs only. Full per-run trees, logs, and raw
Monte Carlo draws are not archived. To regenerate everything:

```
python src/e1/run_profile_monte_carlo.py
python src/e1/run_profile_phase_diagram.py
python src/e1/run_candidate_cost_frontier.py
python src/e1/run_independent_sampling_robustness.py
python src/e1/run_monte_carlo.py
```

Run them in that order. `run_candidate_cost_frontier.py` reads the candidate summary
produced by `run_profile_monte_carlo.py`.

## Random seeds

Runners record the seed in their `run_metadata.json` next to the outputs. Check the
metadata file in each archived result directory for the seed used for that run.

## Known caveat

`run_profile_monte_carlo.py` and `run_independent_sampling_robustness.py` compute two
different Top-1 frequency definitions. See the dedicated section of the top level
README before comparing their outputs.
