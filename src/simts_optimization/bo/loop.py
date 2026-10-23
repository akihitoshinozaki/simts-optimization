"""Outer optimization campaigns: Bayesian optimization and simple baselines.

Every campaign spends one shared budget of measurements. New settings, repeats
of the incumbent (noise estimation), and fixed reference settings (drift
tracking) all count against that budget.

Handling rules, applied identically wherever they appear:

* Drift: reference scores define an additive offset over the measurement
  clock, smoothed by a shrinkage GP (see :func:`estimate_drift`). Model
  training and incumbent choice use drift-corrected scores.
* Noise: repeated measurements of the same setting (incumbent repeats and the
  references) give a pooled within-setting variance, used as the GP's fixed
  observation noise once it has at least two degrees of freedom. Before that,
  the GP infers a homoscedastic noise level.
* Failures: failed runs carry no score. For modelling they are imputed with the
  worst corrected score seen so far, which steers BO away without inventing a
  plausible value. Constraint violations keep their measured (often zero)
  hypervolume and are modelled normally.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import warnings

import numpy as np

from .evaluation import EvaluationRecord, SimulatedExperiment, _json_value
from .space import C5_REFERENCE_SETTINGS, DesignSpace


STRATEGIES = ("bo", "random", "sobol")


@dataclass(frozen=True)
class CampaignConfig:
    """Budget and bookkeeping policy for one campaign.

    ``reference_interval``/``repeat_interval`` of zero disable those
    measurements. The first reference is taken at the start, and the first
    incumbent repeat right after the initial design.
    """

    budget: int = 60
    initial_points: int = 8
    reference_interval: int = 12
    repeat_interval: int = 12
    reference_settings: dict = field(default_factory=lambda: dict(C5_REFERENCE_SETTINGS))
    num_restarts: int = 8
    raw_samples: int = 256

    def __post_init__(self) -> None:
        if self.budget < 1 or self.initial_points < 1:
            raise ValueError("budget and initial_points must be positive")
        if self.reference_interval < 0 or self.repeat_interval < 0:
            raise ValueError("intervals must be nonnegative")


BASELINE_CONFIG = CampaignConfig(reference_interval=0, repeat_interval=0)
"""Baselines spend their whole budget on new settings (the stricter comparison)."""


@dataclass(frozen=True)
class NoiseEstimate:
    pooled_std: float | None
    degrees_of_freedom: int
    groups: int


@dataclass(frozen=True)
class CampaignResult:
    strategy: str
    seed: int
    config: CampaignConfig
    records: tuple[EvaluationRecord, ...]
    corrected_scores: tuple[float | None, ...]
    drift_offsets: tuple[float, ...]
    noise: NoiseEstimate
    recommendation: dict[str, float] | None
    recommendation_index: int | None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["records"] = [r.to_dict() for r in self.records]
        return _json_value(payload)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


def estimate_drift(
    records,
    noise_std: float | None,
    lengthscale: float = 20.0,
) -> np.ndarray:
    """Additive score offset at each record's clock, from reference measurements.

    Reference scores are smoothed with a zero-mean-offset GP whose signal
    variance is the reference variance in excess of measurement noise
    (empirical Bayes). With no excess variance, or no noise estimate yet, the
    offset is zero: noisy references alone must not inject drift.
    """
    refs = [r for r in records if r.kind == "reference" and r.score is not None]
    if len(refs) < 2 or noise_std is None:
        return np.zeros(len(records))
    t = np.array([r.index for r in refs], dtype=float)
    y = np.array([r.score for r in refs])
    signal_var = float(np.var(y, ddof=1)) - noise_std**2
    if signal_var <= 0.0:
        return np.zeros(len(records))

    def kernel(a, b):
        return signal_var * np.exp(-0.5 * np.subtract.outer(a, b) ** 2 / lengthscale**2)

    weights = np.linalg.solve(kernel(t, t) + noise_std**2 * np.eye(t.size), y - y.mean())
    clocks = np.array([r.index for r in records], dtype=float)
    return kernel(clocks, t) @ weights


def _setting_key(settings: dict[str, float]) -> tuple:
    return tuple(sorted((k, round(v, 12)) for k, v in settings.items()))


def estimate_noise(records, corrected: np.ndarray) -> NoiseEstimate:
    """Pooled within-setting standard deviation of corrected scores."""
    groups: dict[tuple, list[float]] = {}
    for record, value in zip(records, corrected):
        if np.isfinite(value) and not record.failed:
            groups.setdefault(_setting_key(record.settings), []).append(float(value))
    repeated = [np.asarray(v) for v in groups.values() if len(v) > 1]
    dof = sum(v.size - 1 for v in repeated)
    if dof == 0:
        return NoiseEstimate(None, 0, 0)
    pooled = sum(float(np.sum((v - v.mean()) ** 2)) for v in repeated) / dof
    return NoiseEstimate(float(np.sqrt(pooled)), int(dof), len(repeated))


def _corrected(records) -> tuple[np.ndarray, np.ndarray]:
    """Drift-corrected scores. Noise for the drift smoother comes from
    non-reference repeats, which are not themselves the drift signal."""
    raw = np.array([np.nan if r.score is None else r.score for r in records])
    repeats_only = [r.kind != "reference" for r in records]
    noise = estimate_noise([r for r, keep in zip(records, repeats_only) if keep], raw[repeats_only])
    drift = estimate_drift(records, noise.pooled_std)
    return raw - drift, drift


def _training_data(records, space: DesignSpace, corrected: np.ndarray):
    """Unit-cube inputs and targets, with failed runs imputed pessimistically."""
    valid = np.isfinite(corrected)
    if not np.any(valid):
        return None, None
    worst = float(np.min(corrected[valid]))
    x = np.array([space.to_unit(r.settings) for r in records])
    y = np.where(valid, corrected, worst)
    return x, y


def _fit_gp(x: np.ndarray, y: np.ndarray, noise: NoiseEstimate):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_x = torch.tensor(x, dtype=torch.float64)
    train_y = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)
    kwargs = {}
    if noise.pooled_std is not None and noise.degrees_of_freedom >= 2:
        variance = max(noise.pooled_std**2, 1e-8)
        kwargs["train_Yvar"] = torch.full_like(train_y, variance)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1), **kwargs)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def _posterior_mean(model, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model.posterior(torch.tensor(x, dtype=torch.float64)).mean.squeeze(-1).numpy()


def _propose_bo(model, x: np.ndarray, dimension: int, config: CampaignConfig, seed: int) -> np.ndarray:
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.optim import optimize_acqf

    torch.manual_seed(seed)
    acquisition = qLogNoisyExpectedImprovement(
        model, X_baseline=torch.tensor(x, dtype=torch.float64), prune_baseline=True)
    bounds = torch.stack([torch.zeros(dimension), torch.ones(dimension)]).to(torch.float64)
    candidate, _ = optimize_acqf(acquisition, bounds=bounds, q=1,
                                 num_restarts=config.num_restarts, raw_samples=config.raw_samples)
    return candidate.detach().numpy()[0]


def _incumbent(records, space, corrected, model) -> int | None:
    """Index of the best observed (non-failed, constraint-passing if any) setting."""
    usable = [i for i, r in enumerate(records) if not r.failed and np.isfinite(corrected[i])]
    passing = [i for i in usable if records[i].passed_constraints]
    pool = passing or usable
    if not pool:
        return None
    if model is not None:
        values = _posterior_mean(model, np.array([space.to_unit(records[i].settings) for i in pool]))
    else:
        values = corrected[pool]
    return pool[int(np.argmax(values))]


def run_campaign(
    strategy: str,
    seed: int,
    config: CampaignConfig | None = None,
    experiment_factory: Callable[[], SimulatedExperiment] = SimulatedExperiment,
    callback: Callable[[EvaluationRecord], None] | None = None,
) -> CampaignResult:
    """Run one campaign with ``strategy`` in {"bo", "random", "sobol"}."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if config is None:
        config = CampaignConfig() if strategy == "bo" else BASELINE_CONFIG
    experiment = experiment_factory()
    space = experiment.space
    reference = space.validate(config.reference_settings)
    rng = np.random.default_rng([seed, STRATEGIES.index(strategy)])
    measurement_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=config.budget)
    # Sobol points are shared by BO's initial design and the Sobol baseline.
    sobol = space.sobol(seed, max(config.budget, 2))
    new_points = 0
    last_reference = last_repeat = None
    records: list[EvaluationRecord] = []
    notes: list[str] = []
    model = None

    def measure(settings, kind, extra=None):
        record = experiment.evaluate(settings, kind=kind, seed=int(measurement_seeds[len(records)]))
        if extra:
            record = replace(record, extra=extra)
        records.append(record)
        if callback:
            callback(record)
        return record

    while len(records) < config.budget:
        n = len(records)
        corrected, _ = _corrected(records) if records else (np.array([]), None)
        if config.reference_interval and (last_reference is None
                                          or n - last_reference >= config.reference_interval):
            measure(reference, "reference")
            last_reference = n
            continue
        if (config.repeat_interval and new_points >= config.initial_points
                and (last_repeat is None or n - last_repeat >= config.repeat_interval)):
            best = _incumbent(records, space, corrected, model)
            if best is not None:
                measure(records[best].settings, "repeat", {"repeat_of": records[best].index})
                last_repeat = n
                continue

        if strategy == "random":
            unit, kind, extra = space.sample_uniform(rng, 1)[0], "random", None
        elif strategy == "sobol" or new_points < config.initial_points:
            unit, kind, extra = sobol[new_points], ("sobol" if strategy == "sobol" else "initial"), None
        else:
            try:
                x, y = _training_data(records, space, corrected)
                if x is None:
                    raise ValueError("no successful measurements to model")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = _fit_gp(x, y, estimate_noise(records, corrected))
                    unit = _propose_bo(model, x, space.dimension, config, seed * 10007 + n)
                kind, extra = "bo", None
            except Exception as error:  # fall back rather than stall the experiment
                unit, kind = sobol[new_points], "fallback"
                extra = {"reason": f"{type(error).__name__}: {error}"}
                notes.append(f"measurement {n}: BO fallback ({extra['reason']})")
                model = None
        measure(space.from_unit(unit), kind, extra)
        new_points += 1

    corrected, drift = _corrected(records)
    noise = estimate_noise(records, corrected)
    if strategy == "bo":
        try:
            x, y = _training_data(records, space, corrected)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _fit_gp(x, y, noise) if x is not None else None
        except Exception as error:
            notes.append(f"final model fit failed: {error}")
            model = None
    else:
        model = None
    best = _incumbent(records, space, corrected, model)
    return CampaignResult(
        strategy=strategy,
        seed=seed,
        config=config,
        records=tuple(records),
        corrected_scores=tuple(None if not np.isfinite(v) else float(v) for v in corrected),
        drift_offsets=tuple(float(d) for d in drift),
        noise=noise,
        recommendation=None if best is None else records[best].settings,
        recommendation_index=best,
        notes=tuple(notes),
    )
