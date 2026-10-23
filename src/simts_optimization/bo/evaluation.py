"""Settings -> C1 scans -> C2 estimators -> C4 Pareto solver -> scores.

The evaluator is the only object that touches the (simulated) apparatus. It
returns an :class:`EvaluationRecord` with the score, its uncertainty, whether
all constraints passed, and an explicit status when the run failed.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import time
import warnings

import numpy as np

from ..atomicity import check_atomicity
from ..estimation import (
    DerivativeConfig,
    estimate_noise_covariance,
    estimate_smoothed_derivatives,
)
from ..inner_pareto import ChannelDataset, ParetoRun, ParetoSettings, solve_inner_pareto
from ..sim.landscape import AcquisitionFailure, HiddenLandscape
from ..sim.mock_experiment import MockExperiment, simulate_mock_experiment
from ..sim.synthetic_basis import IMMode
from .space import C5_DESIGN_SPACE, DesignSpace


C5_PARETO_SETTINGS = ParetoSettings(
    capture_ranges=tuple(float(c) for c in np.arange(0.25, 3.01, 0.25)),
    minimum_slope=0.2,
    validation_gain_tolerance=0.10,
    # Held-out slopes come from noisy Savitzky-Golay derivatives; a 1e-7 margin
    # turns estimator noise into random validation failures.
    validation_slope_tolerance=0.05,
    capture_reference=3.0,
    noise_reference=0.05,
)
"""Fixed C4 policy for every C5 evaluation so scores stay comparable."""

C5_DERIVATIVE_CONFIG = DerivativeConfig(window_length=21, polynomial_order=3)

STATUS_OK = "ok"
STATUS_CONSTRAINT = "constraint_violation"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class FoldScore:
    """C4 outcome with one development/held-out assignment of the scans."""

    development_scans: tuple[int, ...]
    status: str
    hypervolume: float
    best_validated_noise: float | None
    maximum_validated_capture: float | None
    validated_count: int


@dataclass(frozen=True)
class EvaluationRecord:
    """Everything saved about one measurement of one setting."""

    index: int
    kind: str
    settings: dict[str, float]
    seed: int
    status: str
    score: float | None
    score_uncertainty: float | None
    passed_constraints: bool
    constraint_reasons: tuple[str, ...] = ()
    failure_reason: str | None = None
    message: str = ""
    atomicity_ratio: float | None = None
    folds: tuple[FoldScore, ...] = ()
    elapsed_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAILED

    def to_dict(self) -> dict:
        return _json_value(asdict(self))


def _json_value(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _dataset(name: str, experiment: MockExperiment, scan_indices: tuple[int, ...]) -> ChannelDataset:
    scans = np.array([experiment.scans[i].channels for i in scan_indices])
    derivative = estimate_smoothed_derivatives(
        experiment.detuning, scans.mean(axis=0), C5_DERIVATIVE_CONFIG)
    covariance = estimate_noise_covariance(scans).covariance
    return ChannelDataset(
        dataset_id=name,
        channel_names=experiment.channel_names,
        detuning_axis=experiment.detuning,
        channels=derivative.smoothed_basis,
        basis_derivative=derivative.basis_derivative,
        frequency_slope=derivative.frequency_slope,
        covariance=covariance,
    )


def _fold(experiment: MockExperiment, development: tuple[int, ...],
          held_out: tuple[tuple[int, ...], ...], settings: ParetoSettings) -> tuple[FoldScore, ParetoRun]:
    run = solve_inner_pareto(
        _dataset("development", experiment, development),
        tuple(_dataset(f"held-out-{i}", experiment, idx) for i, idx in enumerate(held_out)),
        settings,
    )
    s = run.summary
    return FoldScore(development, run.status, s.normalized_hypervolume, s.best_validated_noise,
                     s.maximum_validated_capture, s.validated_count), run


def score_experiment(
    experiment: MockExperiment,
    settings: ParetoSettings = C5_PARETO_SETTINGS,
) -> tuple[float, float, tuple[FoldScore, ...]]:
    """Two-fold C4 score: mean hypervolume and its standard error.

    Scans are split in half; each half serves once as development data while the
    other half supplies two held-out datasets. Swapping roles gives two
    independent-noise estimates of the same quantity.
    """
    count = len(experiment.scans)
    half = count // 2
    first, second = tuple(range(half)), tuple(range(half, 2 * half))

    def held(indices):
        mid = len(indices) // 2
        return (indices[:mid], indices[mid:])

    folds = tuple(_fold(experiment, dev, held(other), settings)[0]
                  for dev, other in ((first, second), (second, first)))
    values = np.array([f.hypervolume for f in folds])
    return float(values.mean()), float(np.std(values, ddof=1) / np.sqrt(values.size)), folds


class SimulatedExperiment:
    """Black-box C5 objective: settings in, scored EvaluationRecord out.

    ``clock`` counts measurements so that drift progresses with every run,
    including repeats and reference measurements.
    """

    def __init__(
        self,
        landscape: HiddenLandscape | None = None,
        space: DesignSpace = C5_DESIGN_SPACE,
        pareto_settings: ParetoSettings = C5_PARETO_SETTINGS,
        simulator: Callable = simulate_mock_experiment,
    ) -> None:
        self._landscape = landscape or HiddenLandscape()
        self.space = space
        self.pareto_settings = pareto_settings
        self._simulator = simulator
        self.clock = 0

    def evaluate(self, settings: Mapping[str, float], *, kind: str, seed: int) -> EvaluationRecord:
        values = self.space.validate(settings)
        index = self.clock
        self.clock += 1
        start = time.perf_counter()
        common = dict(index=index, kind=kind, settings=values, seed=int(seed))
        try:
            state = self._landscape.physical_state(values, clock=index, seed=seed)
            experiment = self._simulator(state.theta, IMMode.USEFUL, state.config)
            atomicity = check_atomicity(experiment)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # cvxpy "may be inaccurate"
                score, uncertainty, folds = score_experiment(experiment, self.pareto_settings)
        except AcquisitionFailure as error:
            return EvaluationRecord(**common, status=STATUS_FAILED, score=None,
                                    score_uncertainty=None, passed_constraints=False,
                                    failure_reason=error.reason, message=str(error),
                                    elapsed_seconds=time.perf_counter() - start)
        except (ValueError, np.linalg.LinAlgError, ArithmeticError) as error:
            return EvaluationRecord(**common, status=STATUS_FAILED, score=None,
                                    score_uncertainty=None, passed_constraints=False,
                                    failure_reason="pipeline_error",
                                    message=f"{type(error).__name__}: {error}",
                                    elapsed_seconds=time.perf_counter() - start)
        if not (np.isfinite(score) and np.isfinite(uncertainty)):
            return EvaluationRecord(**common, status=STATUS_FAILED, score=None,
                                    score_uncertainty=None, passed_constraints=False,
                                    failure_reason="nonfinite_score", folds=folds,
                                    elapsed_seconds=time.perf_counter() - start)

        reasons = list(atomicity.reason_codes)
        if any(f.validated_count == 0 for f in folds):
            reasons.append("no_validated_frontier")
        return EvaluationRecord(
            **common,
            status=STATUS_OK if not reasons else STATUS_CONSTRAINT,
            score=score,
            score_uncertainty=uncertainty,
            passed_constraints=not reasons,
            constraint_reasons=tuple(reasons),
            atomicity_ratio=atomicity.pump_blocked_ratio,
            folds=folds,
            elapsed_seconds=time.perf_counter() - start,
        )
