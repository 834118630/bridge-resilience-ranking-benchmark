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

The runners use relative imports, so they must be invoked as modules with the
`src` directory on the path, not as file paths.

```
python -m e1.run_profile_monte_carlo --samples 10000 --seed 20260914 --bootstrap-resamples 10000
python -m e1.run_profile_phase_diagram
python -m e1.run_candidate_cost_frontier
python -m e1.run_independent_sampling_robustness
python -m e1.run_monte_carlo
```

Run `python -m e1.run_profile_monte_carlo` from a directory that contains `e1` on
the import path, for example after `set PYTHONPATH=src` on Windows or
`export PYTHONPATH=src` on Linux and macOS.

Run them in that order. `run_candidate_cost_frontier` reads the candidate summary
produced by `run_profile_monte_carlo`.

## Random seeds

Runners record the seed in their `run_metadata.json` next to the outputs. Check the
metadata file in each archived result directory for the seed used for that run.

## Known caveat

`run_profile_monte_carlo.py` and `run_independent_sampling_robustness.py` compute two
different Top-1 frequency definitions. See the dedicated section of the top level
README before comparing their outputs.
