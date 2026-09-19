# Supplementary Methods

This supplement contains the derivations, extended sensitivity analyses, and pre-registered sampling protocol that support the main manuscript. The main text retains all variable definitions, estimands, sampling gates, and decision rules needed to interpret the results without consulting this supplement.

## S1. Recovery-duration distribution sensitivity

The primary analysis uses a lower-truncated normal distribution with a 0.05-day lower bound. The table reports the mean-regret winner and the metric-aggregated Top-1 frequency for F_fast and L4 under nearby lower bounds and a truncated lognormal alternative. The lognormal option matches the nominal mean and standard deviation before conditioning on the lower bound.

| Variant | Distribution | Lower bound (days) | Scenario | Mean-regret winner | F_fast metric-aggregated Top-1 | L4 metric-aggregated Top-1 | F_fast mean regret | L4 mean regret |
|---|---|---:|---|---|---:|---:|---:|---:|
| truncnorm_bound_0.01 | truncnorm | 0.01 | evidence_tradeoff | F_fast | 0.886741 | 0.076929 | 0.001506 | 0.304981 |
| truncnorm_bound_0.01 | truncnorm | 0.01 | boundary_tradeoff | F_fast | 0.525803 | 0.437831 | 0.048814 | 0.191343 |
| truncnorm_bound_0.05_baseline | truncnorm | 0.05 | evidence_tradeoff | F_fast | 0.887324 | 0.076864 | 0.001498 | 0.303464 |
| truncnorm_bound_0.05_baseline | truncnorm | 0.05 | boundary_tradeoff | F_fast | 0.523998 | 0.440152 | 0.049106 | 0.189193 |
| truncnorm_bound_0.10 | truncnorm | 0.10 | evidence_tradeoff | F_fast | 0.887359 | 0.076523 | 0.001479 | 0.302994 |
| truncnorm_bound_0.10 | truncnorm | 0.10 | boundary_tradeoff | F_fast | 0.521886 | 0.441959 | 0.049280 | 0.188245 |
| truncated_lognormal_bound_0.05 | lognormal | 0.05 | evidence_tradeoff | F_fast | 0.886687 | 0.082305 | 0.001820 | 0.299662 |
| truncated_lognormal_bound_0.05 | lognormal | 0.05 | boundary_tradeoff | F_fast | 0.536312 | 0.432681 | 0.045517 | 0.186767 |

The winner is unchanged across all four variants in both tradeoff scenarios. The largest change in F_fast metric-aggregated Top-1 frequency is 0.014. These analyses are post-hoc robustness checks, not additional pre-registered sampling gates.

## S2. Metric-weight sensitivity

The primary mean-regret estimand gives equal weight to the four metrics. The deterministic schemes below are stress tests, not calibrated preferences. The Dirichlet analysis uses 10,000 draws from Dirichlet(1,1,1,1) and reports the frequency with which each candidate has the lowest weighted regret.

| Scenario | Weight scheme | Top-1 candidate | Weighted mean regret F_fast | Weighted mean regret L4 |
|---|---|---|---:|---:|
| boundary_tradeoff | equal | F_fast | 0.048798 | 0.191351 |
| boundary_tradeoff | rapidity_emphasis | F_fast | 0.050515 | 0.208613 |
| boundary_tradeoff | resilience_emphasis | F_fast | 0.042422 | 0.183452 |
| boundary_tradeoff | early_functionality_emphasis | F_fast | 0.060131 | 0.164201 |
| evidence_tradeoff | equal | F_fast | 0.001493 | 0.304997 |
| evidence_tradeoff | rapidity_emphasis | F_fast | 0.002373 | 0.311523 |
| evidence_tradeoff | resilience_emphasis | F_fast | 0.001195 | 0.301951 |
| evidence_tradeoff | early_functionality_emphasis | F_fast | 0.001211 | 0.292309 |
| shape_only | equal | L4 | 0.377537 | 0.001441 |
| shape_only | rapidity_emphasis | L4 | 0.390590 | 0.001584 |
| shape_only | resilience_emphasis | L4 | 0.375220 | 0.001188 |
| shape_only | early_functionality_emphasis | L4 | 0.377013 | 0.001375 |

| Scenario | Candidate | Dirichlet winner frequency |
|---|---|---:|
| shape_only | L4 | 1.000000 |
| evidence_tradeoff | F_fast | 1.000000 |
| boundary_tradeoff | L4 | 0.006400 |
| boundary_tradeoff | F_fast | 0.993600 |

The equal, rapidity-emphasis, resilience-emphasis, and early-functionality-emphasis schemes all select L4 in the shape-only scenario and F_fast in both tradeoff scenarios. Under Dirichlet weight perturbation, F_fast wins 100% of strong-speed-advantage draws and 99.36% of near-boundary draws; L4 wins the remaining 0.64% of near-boundary draws. The weight analysis is a robustness check and is not used to claim a calibrated metric preference.

## S3. Pre-registered paired-versus-independent sampling gate

The following rules were fixed before the final 10,000-replication run. They are copied here so that the decision rule is auditable independently of the main text.

Let `T_p` and `T_i` be the paired and independent mean-regret Top-1 frequencies for F_fast, `ΔT = T_p - T_i`, and let `R_p` and `R_i` be the corresponding pairwise reversal rates.

### Keep

- the best candidate is unchanged;
- `T_i >= 0.60`;
- `ΔT <= 0.15`;
- `R_i - R_p <= 0.10`.

### Downgrade

- the best candidate is unchanged; and
- `T_i < 0.60`, or `ΔT > 0.15`, or `R_i - R_p > 0.10`.

### Re-evaluate

- the strong-speed-advantage best candidate changes; or
- the near-boundary and strong-speed-advantage scenarios both change their best candidate; or
- the reversal increase exceeds 0.10 in multiple capacity or intensity conditions.

The shape-only scenario does not use the F_fast gate. It only checks whether L2 and F_fast belong to the same equivalence class. The thresholds are decision gates, not statistical significance thresholds, and were not changed after the final run.

## S4. Cost-coefficient scan and L3/L4 derivation

For the linear cost model with `theta_i = 1` for L0-L4, the cost difference between L3 and L4 follows directly from the capacity and recovery multipliers:

$$\frac{C_{L3}}{C_{L4}} - 1 = 0.25(\alpha_s - \alpha_c).$$

L4 therefore has a cost no greater than L3 when `alpha_s >= alpha_c`. Because L4 also has the lower discounted loss, L4 dominates L3 on and above the diagonal of the coefficient grid. Below the diagonal, L3 is cheaper than L4 and can enter the frontier; whether it does depends on whether another candidate is both cheaper and lower-loss.

The coefficient scan contains 50 scenario-coefficient rows and reproduces the analytic identity to machine precision (maximum absolute error 0.0). L3 is Pareto-efficient in 6 of 25 strong-speed-advantage cells and 5 of 25 near-boundary cells, all below the diagonal. The primary specification `alpha_c = alpha_s = 0.5` lies on the diagonal.

At `alpha_c = 1.0` and `alpha_s = 0.0`, the linear model assigns the L0 cost index a value of zero. This edge point is retained and reported as the boundary of the model's validity rather than removed.

## S5. Reproducibility record

Each final output directory contains a `run_metadata.json` file with the exact command, random seed, input and output SHA256 hashes, software environment, and completion or failure status. The archived aggregate results reproduce the six publication figures through the documented command:

```text
python src/e1/make_publication_figures.py
```

The repository-level unit tests cover metrics, Monte Carlo calculations, cost-frontier logic, sampling modes, uncertainty, provenance metadata, and the figure-reproduction integration path. The paired and independent sampling gates, the Monte Carlo aggregation rule, and the recovery-shape definitions are all represented in the archived code and metadata.

## S6. Additional model and time-basis notes

The main analysis is organized into five modules: recovery-shape comparison, ranking and regret evaluation, engineering-loss quantification, cost-Pareto decision analysis, and sampling-robustness analysis. The modular structure keeps the effect of each assumption visible and avoids conflating ranking results with economic results.

The incremental cost representation is used because subtracting the same constant from every candidate cannot change the utility-minimizing candidate; it therefore produces the same economic winner and switch points as the L4-anchored form while giving the no-action option a directly interpretable zero incremental cost.

Both terms of the economic utility use the decision-time basis. The cost is a one-time outlay at that time, and the loss stream is discounted to the same instant. Future repair, maintenance, and lifecycle expenditure are outside the model, and no claim is made about costs incurred after the recovery window.
