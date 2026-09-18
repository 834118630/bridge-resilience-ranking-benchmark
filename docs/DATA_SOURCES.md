# Data sources

## Primary input

Federal Emergency Management Agency. Hazus Earthquake Model Technical Manual 6.1.
FEMA, Washington, DC (2024).

This public technical manual supplies the bridge fragility medians and dispersions for
the four damage states (Slight, Moderate, Extensive, Complete) and the restoration
parameters used to build the recovery duration distributions. The manual is the single
fragility source for the study. It was chosen to control fragility definitions and
comparison conditions, not because independent data are unavailable.

## Context sources, not used as numerical validation

Two further published sources are used only as qualitative context in the manuscript.
Neither is used to generate any number in this repository.

- A study of curved reinforced concrete bridge piers, used as cross-geometry context.
- A study of a long-span rigid-frame bridge on a soft clay site, used as supplementary
  cross-system context.

## Cost inputs

All cost values are normalized indices relative to the L4 candidate. They are scenario
parameters, not monetary construction costs. No cost database is used.

## No restricted data

No proprietary, licensed, or personal data are included. Everything in `results/` is
derived from the public manual above plus the scenario parameters defined in the
analysis code.
