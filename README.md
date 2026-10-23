# SIMTS Bayesian Optimization

A Python project for Bayesian optimization experiments using scientific-computing and convex-optimization tools.

## Requirements

- Python 3.10 or newer

## Setup

Create and activate a virtual environment, then install the package and its development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the tests

```bash
pytest
```

## Milestone C1 mock experiment

The C1 simulator maps a low-dimensional physical setting to repeated scans
with correlated channel noise, scan-to-scan drift, nuisance perturbations, and
a matched pump-blocked control. It provides the three handoff regimes:
`redundant`, `useful`, and `fake` IM.

```python
from simts_optimization import IMMode, evaluate_c1_regime, simulate_mock_experiment

experiment = simulate_mock_experiment(mode=IMMode.USEFUL)
outcome = evaluate_c1_regime(experiment)

print(outcome.basis_audit)
print(outcome.atomicity)
print(outcome.frontier_expanded)
```

The IM pair is admitted to the frontier optimizer only when it is both a
repeatable basis expansion and valid under the external atomicity control.

## Milestone C2 numerical DDC and scan pipeline

The C2 reference pipeline preserves raw ADC blocks and acquisition metadata,
reconstructs coherent references for integer-combination channels, and uses the
declared complex convention:

```text
Z = 2 * LPF{x(t) * exp[-i(2*pi*f_ref*t + phi_ref)]}
I = Re(Z), Q = Im(Z)
```

The factor of two makes `Z` equal to the peak phasor of a real input cosine.
For a channel `(p, q)`, the reference frequency and phase are
`p*f1 + q*f2` and `p*phi1 + q*phi2`.

```python
from simts_optimization import (
    ChannelReference,
    DerivativeConfig,
    process_raw_scan,
)

references = (
    ChannelReference("fundamental", 1, 0),
    ChannelReference("im", 2, -1),
)

processed = process_raw_scan(
    raw_record,
    references,
    derivative_config=DerivativeConfig(window_length=11, polynomial_order=3),
)
```

Detuning calibration retains the original actuator coordinate. Registration is
restricted to monotonic interpolation on a declared target frequency grid.
Channel derivatives use a fixed Savitzky-Golay local polynomial, and repeated
scan covariance estimation records diagonal loading, eigenvalues, condition
number, and method metadata.

## Milestone C3 basis audit

The C3 audit keeps development and validation scans separate. It fits the
candidate-from-baseline projection on development scans and scores residual
energy and repeatability only on held-out scans. It also reports principal
angles, a noise-whitened residual spectrum, stable rank, and deterministic
scan-bootstrap confidence intervals.

```python
from simts_optimization import (
    BasisAuditConfig,
    IMMode,
    audit_mock_experiment,
    simulate_mock_experiment,
)

experiment = simulate_mock_experiment(mode=IMMode.USEFUL)
audit = audit_mock_experiment(
    experiment,
    config=BasisAuditConfig(bootstrap_resamples=128),
)

print(audit.basis_expanding)
print(audit.projection.normalized_residual_energy)
print(audit.noise_whitened_svd.singular_value_ratios)
print(audit.bootstrap.leading_singular_value_ratio)
```

The acceptance decision requires a nontrivial and repeatable held-out
residual, a stable residual subspace, and at least one singular direction above
the declared noise threshold. Stable non-atomic artifacts can still expand the
mathematical basis; the separate C1 atomicity control is what excludes those
artifacts from scientific optimization.

## Milestone C4 reusable inner Pareto solver

`solve_inner_pareto(development, held_out, settings)` runs the C0 QP for each
capture requirement, checks its numerical constraints, validates every usable
weight vector, and builds a frontier from candidates that pass **all** held-out
datasets. The existing C0 weight and frontier APIs remain available.

### Input contract

Each `ChannelDataset` supplies:

| Field | Shape / meaning |
| --- | --- |
| `dataset_id` | Unique development or held-out data identifier |
| `channel_names` | Unique names in the same order across all datasets |
| `detuning_axis` | `(samples,)`, finite, strictly increasing, covering lock point zero |
| `channels` | `(samples, channels)`, processed real channel responses |
| `basis_derivative` | Same shape, channel derivatives with respect to detuning |
| `frequency_slope` | `(channels,)`, local channel slopes at zero |
| `covariance` | `(channels, channels)`, symmetric positive definite noise covariance |

Use the same physical units, channel calibration, and lock origin throughout a
run. Split complex data into real I/Q columns. Covariance describes observation
noise; perform any covariance regularization explicitly before creating the
dataset. Arrays are copied and made read-only. Held-out grids may differ from
the development grid. Provide independently acquired held-out data and keep
preprocessing fits separate; distinct IDs alone cannot establish independence.
C2 `ProcessedScan.basis`, `.detuning_hz`, and `.derivative_estimate` provide the
response and derivative arrays; `estimate_noise_covariance` provides covariance
from repeats. Make sure the local slope estimate is evaluated at lock point zero.

Here is a runnable synthetic example with a small held-out gain change:

```python
import numpy as np
from simts_optimization import ChannelDataset, ParetoSettings, solve_inner_pareto

def make_dataset(dataset_id, gain):
    x = np.linspace(-1.5, 1.5, 121)
    channels = gain * np.column_stack((x - 2 * x**3 / 3, x))
    derivative = gain * np.column_stack((1 - 2 * x**2, np.ones_like(x)))
    return ChannelDataset(
        dataset_id=dataset_id,
        channel_names=("narrow", "broad"),
        detuning_axis=x,
        channels=channels,
        basis_derivative=derivative,
        frequency_slope=gain * np.ones(2),
        covariance=np.diag([1.0, 9.0]),
    )

settings = ParetoSettings(
    capture_ranges=(0.0, 0.5, 1.0, 1.5),
    minimum_slope=0.2,
    capture_reference=1.5,
    noise_reference=4.0,
)
run = solve_inner_pareto(
    make_dataset("development", 1.0),
    (make_dataset("held-out", 1.01),),
    settings,
)
settings.save("outputs/c4_settings.json")
run.save("outputs/c4_run.json")
print(run.status, run.summary)
for candidate in run.frontier:
    print(candidate.capture_range, candidate.solve.weights,
          candidate.worst_validation_noise)
```

### Constraints, validation, and scores

The QP minimizes `w.T @ covariance @ w` subject to unit development gain and
the minimum slope throughout `[-capture_range, +capture_range]`. Constraints
include grid samples, zero, and interpolated interval endpoints. This guarantees
the slope bound for the piecewise-linear derivative model; behavior between
samples depends on that interpolation assumption.

Held-out checks use the fixed weights. They report local gain, minimum slope,
equivalent frequency noise `sqrt(w.T @ held_out_covariance @ w) / abs(gain)`,
lock-offset drift, and response RMSE within the required range. RMSE compares
the weighted response with detuning using the development lock offset and is
sample-weighted, including interpolated endpoints. Gain and slope tolerances
determine acceptance; offset and RMSE are diagnostics. No held-out gain or
offset is fitted away. A zero gain produces a failed validation and null noise.

`run.candidates` retains every solve, including failures; `training_frontier`
uses development noise, while `frontier` minimizes the worst noise across
held-out datasets and maximizes required capture range. Validation occurs
before pruning, so a training-dominated candidate can still qualify when a
larger-range candidate fails validation.

`run.summary` includes feasible/validated fractions (over all requested
conditions), best frontier noise, maximum frontier capture, and normalized
hypervolume. Hypervolume is the union of rectangles
`[0, capture] x [noise, noise_reference]`, clipped to the reference box and
divided by `capture_reference * noise_reference`; larger is better. Keep both
references, sweep, units, and validation policy fixed across BO evaluations.
An empty validated frontier scores zero and has null best noise/capture.

### Reproducibility and failures

`ParetoSettings.save/load` persists the `c4-v1` settings schema and rejects
unknown fields or versions. Reports include settings, dataset IDs, channel
order, weights, metrics, raw solver statuses, iterations, failure messages,
and package/backend versions. Retain the input arrays under those IDs to replay
a run. `QPSettings` selects OSQP (default) or Clarabel and controls tolerances,
iteration limits, and whether inaccurate optima can be considered. Every
accepted solution must independently pass numerical constraint checks.

Malformed inputs raise `ValueError`. Individual conditions record `infeasible`,
`out_of_domain`, `solver_error`, iteration-limit/backend statuses, or
`invalid_solution` and the sweep continues. Run status distinguishes `success`,
`partial_success`, `no_validated_candidates`, `no_feasible_solution` (all
conditions proven infeasible), and `no_usable_solution` (no accepted solve,
without proof that every condition is infeasible).

Run C4 acceptance tests with `pytest tests/test_c4_inner_pareto.py`. They cover
multiple synthetic channel counts, correlated/ill-conditioned covariance,
infeasibility and injected solver failures, held-out isolation, serialization,
repeatability, and OSQP/Clarabel agreement on synthetic and C0 datasets.

## Milestone C5 outer Bayesian optimization

C5 uses Bayesian optimization (BO) to choose the experiment settings θ. The C4
QP still chooses the channel weights `w` inside every evaluation. No real data
is involved yet: the "apparatus" is the C1 simulator driven by a hidden physics
map.

### Settings θ

| Parameter | Range | Unit |
| --- | --- | --- |
| `modulation_frequency` | 1.0 – 10.0 | MHz |
| `modulation_depth` | 0.1 – 3.0 | rad |
| `pump_power` | 0.1 – 4.0 | P_sat |

`C5_DESIGN_SPACE` defines the box, and optimizers work in the unit cube.
`C5_REFERENCE_SETTINGS` (4 MHz, 1 rad, 1 P_sat) is the drift reference.

### Pipeline for one measurement

```text
θ → HiddenLandscape (SNR, linewidth, IM strength; failures; drift)
  → C1 simulate_mock_experiment (12 scans)
  → C2 Savitzky-Golay derivatives + repeat covariance
  → C1 atomicity check
  → C4 solve_inner_pareto, run twice with dev/held-out halves swapped
  → score = mean normalized hypervolume, uncertainty = its standard error
```

`SimulatedExperiment.evaluate` returns an `EvaluationRecord` containing:

- the settings, seed, and measurement clock;
- `status` (`ok`, `constraint_violation`, or `failed`), plus `passed_constraints` and reasons;
- score and uncertainty;
- the atomicity ratio and per-fold C4 summaries.

Every run uses the fixed `C5_PARETO_SETTINGS` so that scores are comparable
across evaluations.

### Hidden good region

`simts_optimization.sim.landscape.HiddenLandscape` holds the ground truth:

- IM conversion is resonantly enhanced near a hidden modulation frequency.
- Signal follows `J1(depth)`.
- Pump power trades saturation against intensity noise.

The good region is where the drift-free quality is at least 75% of the best.
It covers 1.7% of the box and does not contain the reference setting. The
optimizer (`bo/loop.py`, `bo/space.py`) never imports the landscape; a test
enforces this. Only the benchmark referee calls `in_good_region`, and only
after each campaign ends.

### Loop, noise, drift, and failures

`run_campaign("bo", seed)` runs the following sequence:

1. Take a reference measurement.
2. Measure 8 scrambled-Sobol settings.
3. Fit a BoTorch `SingleTaskGP` and choose the next setting with
   `qLogNoisyExpectedImprovement`.
4. Evaluate it, save the record, refit, and repeat until the budget is spent.

Within that loop:

- **Noise:** every 12 measurements, the current incumbent is re-measured. The
  pooled within-setting variance from these repeats becomes the GP's fixed
  noise once it has at least 2 degrees of freedom.
- **Drift:** the reference is re-measured every 12 measurements. A shrinkage GP
  smooths the reference scores into an additive offset. The offset's variance
  is set to the reference variance in excess of the repeat noise, so the
  correction is zero when drift is indistinguishable from noise.
- **Failures:** laser unlock at high depth and pump, random acquisition
  dropout, and pipeline exceptions are recorded as `failed` with a reason and
  no score. The GP imputes these as the worst observed score. If a GP fit or
  the acquisition step itself errors, the loop records a `fallback` Sobol
  point instead of stopping.
- **Constraint violations:** runs with no validated C4 frontier or a failed
  atomicity check keep their measured score and are modelled normally.

### Comparison against simple searches

```bash
python -m simts_optimization.bo --seeds 20 --budget 60 --output-dir outputs/c5
```

This runs BO, random search, and Sobol search on the same seeds. For each
seed, all three strategies share the same noise seeds and Sobol scramble. Every
campaign gets a budget of 60 measurements. BO's 60 include its references
(5) and incumbent repeats (4). The baselines spend all 60 on new settings,
which is the stricter comparison. Results from `outputs/c5/benchmark.json`
(20 seeds; runs that never found the region are counted as 61):

| Strategy | Found region | Median measurements to find it | Mean |
| --- | --- | --- | --- |
| Bayesian optimization | 20/20 | 20 | 23.7 |
| Random search | 15/20 | 41 | 37.8 |
| Sobol search | 14/20 | 36 | 40.3 |

BO found the region sooner than random search on 75% of seeds and sooner than
Sobol on 80% (ties count as half). `outputs/c5/benchmark.png` plots discovery
against measurements, and `outputs/c5/campaigns/` holds every record. The
completion check (`BenchmarkResult.complete`) requires BO to beat both
baselines on median and mean discovery, on discovery rate, and on head-to-head
win rate.

The final recommendation is the incumbent under the GP posterior mean. It lies
inside the region for 45% of BO seeds. Near the optimum the hypervolume
saturates, so noise can make an in-region point and a point just outside the
threshold score about the same.

Fast C5 tests run with plain `pytest`. The 10-seed acceptance benchmark takes
about 4 minutes; run it with `pytest -m slow`.

## Project layout

```text
.
├── pyproject.toml
├── src/
│   └── simts_optimization/
└── tests/
```
