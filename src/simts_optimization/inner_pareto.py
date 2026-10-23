"""C4 reusable inner Pareto solver with independent held-out evaluation."""

from dataclasses import asdict, dataclass, fields
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np

from .frontier import CaptureRangeResult, build_pareto_frontier
from .optimization import (
    QPResult, QPSettings, _capture_samples, _validate_derivatives,
    _validate_inputs, solve_capture_range_qp,
)


SETTINGS_VERSION = "c4-v1"


@dataclass(frozen=True)
class ChannelDataset:
    """One processed development or held-out dataset, in a shared channel order.

    Channels and derivatives have shape (detuning samples, real channels).
    Covariance is per-observation channel noise, not uncertainty of the mean.
    All detunings are offsets from the fixed lock point zero in the same units.
    """

    dataset_id: str
    channel_names: tuple[str, ...]
    detuning_axis: np.ndarray
    channels: np.ndarray
    basis_derivative: np.ndarray
    frequency_slope: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a nonempty string")
        names = tuple(self.channel_names)
        if (not names or any(not isinstance(n, str) or not n.strip() for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("channel_names must be nonempty and unique")
        for name in ("detuning_axis", "channels", "basis_derivative", "frequency_slope", "covariance"):
            if np.iscomplexobj(getattr(self, name)):
                raise ValueError(f"{name} must be real; split complex channels into I/Q")
        covariance, slope = _validate_inputs(self.covariance, self.frequency_slope)
        derivative, axis = _validate_derivatives(self.basis_derivative, self.detuning_axis, slope.size)
        channels = np.asarray(self.channels, dtype=float)
        if channels.shape != derivative.shape or not np.all(np.isfinite(channels)):
            raise ValueError("channels must be finite and match basis_derivative shape")
        if len(names) != slope.size:
            raise ValueError("channel_names must match the number of channels")
        object.__setattr__(self, "channel_names", names)
        for name, value in (("covariance", covariance), ("frequency_slope", slope),
                            ("basis_derivative", derivative), ("detuning_axis", axis),
                            ("channels", channels)):
            copied = np.array(value, dtype=float, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


@dataclass(frozen=True)
class ParetoSettings:
    """Versioned run policy; fixed references make BO scores comparable.

    Hypervolume rewards larger capture range and smaller worst-held-out noise,
    inside [0, capture_reference] x [0, noise_reference], normalized to [0, 1].
    References must stay fixed across BO evaluations (in the dataset's units).
    """

    capture_ranges: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    minimum_slope: float = 0.4
    validation_gain_tolerance: float = 0.05
    validation_slope_tolerance: float = 1e-7
    noise_tolerance: float = 1e-9
    capture_reference: float = 2.0
    noise_reference: float = 10.0
    qp: QPSettings = QPSettings()
    version: str = SETTINGS_VERSION

    def __post_init__(self) -> None:
        if self.version != SETTINGS_VERSION:
            raise ValueError(f"unsupported settings version: {self.version}")
        ranges = np.asarray(self.capture_ranges, dtype=float)
        if (ranges.ndim != 1 or ranges.size == 0 or not np.all(np.isfinite(ranges))
                or np.any(ranges < 0) or np.any(np.diff(ranges) <= 0)):
            raise ValueError("capture_ranges must be finite, nonnegative, and strictly increasing")
        object.__setattr__(self, "capture_ranges", tuple(float(c) for c in ranges))
        for name in ("minimum_slope", "validation_gain_tolerance", "validation_slope_tolerance", "noise_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.validation_gain_tolerance >= 1:
            raise ValueError("validation_gain_tolerance must be less than one")
        for name in ("capture_reference", "noise_reference"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.qp, QPSettings):
            raise ValueError("qp must be QPSettings")

    def to_dict(self) -> dict:
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict) -> "ParetoSettings":
        if payload.get("version") != SETTINGS_VERSION:
            raise ValueError("missing or unsupported settings version")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        values = dict(payload)
        if "qp" in values:
            try:
                values["qp"] = QPSettings(**values["qp"])
            except TypeError as error:
                raise ValueError("invalid QP settings") from error
        return cls(**values)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ParetoSettings":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ValidationMetrics:
    dataset_id: str
    status: str
    gain: float | None = None
    minimum_slope: float | None = None
    equivalent_frequency_noise: float | None = None
    response_rmse: float | None = None
    lock_offset: float | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True)
class ParetoCandidate:
    capture_range: float
    solve: QPResult
    training_noise: float | None
    validation: tuple[ValidationMetrics, ...]

    @property
    def validated(self) -> bool:
        return self.solve.feasible and bool(self.validation) and all(v.passed for v in self.validation)

    @property
    def worst_validation_noise(self) -> float | None:
        if not self.validated:
            return None
        return max(v.equivalent_frequency_noise for v in self.validation)


@dataclass(frozen=True)
class ParetoSummary:
    condition_count: int
    feasible_count: int
    validated_count: int
    frontier_count: int
    feasible_fraction: float
    validated_fraction: float
    best_validated_noise: float | None
    maximum_validated_capture: float | None
    normalized_hypervolume: float


@dataclass(frozen=True)
class ParetoRun:
    settings: ParetoSettings
    development_id: str
    held_out_ids: tuple[str, ...]
    channel_names: tuple[str, ...]
    candidates: tuple[ParetoCandidate, ...]
    training_frontier: tuple[ParetoCandidate, ...]
    frontier: tuple[ParetoCandidate, ...]
    summary: ParetoSummary
    status: str
    software_versions: dict[str, str]

    def to_dict(self) -> dict:
        """JSON-safe report, including every failed condition and fixed weights."""
        return _json_value(asdict(self))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _software_versions(solver: str) -> dict[str, str]:
    versions = {}
    for name in ("simts-optimization", "numpy", "cvxpy", solver.lower()):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def validate_candidate(
    weights: np.ndarray, development: ChannelDataset, held_out: ChannelDataset,
    *, capture_range: float, settings: ParetoSettings,
) -> ValidationMetrics:
    """Evaluate fixed weights without refitting gain, offset, or covariance.

    Noise is sqrt(w.T Sigma_holdout w) / abs(w.T d_holdout). Response RMSE
    compares weighted channels to detuning using the development lock offset;
    it therefore includes held-out offset drift and nonlinearity. RMSE/offset
    are diagnostics; gain and minimum slope determine validation acceptance.
    """
    if development.channel_names != held_out.channel_names:
        raise ValueError("held-out channel_names and order must match development")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != development.frequency_slope.shape or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and match the channel count")
    if not np.isfinite(capture_range) or capture_range < 0:
        raise ValueError("capture_range must be finite and nonnegative")
    axis = held_out.detuning_axis
    if axis[0] > -capture_range or axis[-1] < capture_range:
        return ValidationMetrics(held_out.dataset_id, "out_of_domain")
    grid, derivative = _capture_samples(held_out.basis_derivative, axis, capture_range)
    _, channels = _capture_samples(held_out.channels, axis, capture_range)
    _, development_lock = _capture_samples(development.channels, development.detuning_axis, 0.0)
    _, held_out_lock = _capture_samples(held_out.channels, axis, 0.0)
    gain = float(weights @ held_out.frequency_slope)
    minimum_slope = float(np.min(derivative @ weights))
    variance = float(weights @ held_out.covariance @ weights)
    noise = float(np.sqrt(variance) / abs(gain)) if abs(gain) > np.finfo(float).eps else None
    response = channels @ weights - float(development_lock[0] @ weights)
    rmse = float(np.sqrt(np.mean((response - grid) ** 2)))
    offset = float((held_out_lock[0] - development_lock[0]) @ weights)
    if not all(np.isfinite(v) for v in (gain, minimum_slope, variance, rmse, offset)) or (noise is not None and not np.isfinite(noise)):
        return ValidationMetrics(held_out.dataset_id, "nonfinite_metrics")
    passed = (noise is not None and abs(gain - 1.0) <= settings.validation_gain_tolerance
              and minimum_slope >= settings.minimum_slope - settings.validation_slope_tolerance)
    return ValidationMetrics(held_out.dataset_id, "passed" if passed else "constraint_violation",
                             gain, minimum_slope, noise, rmse, offset)


def _frontier(candidates: tuple[ParetoCandidate, ...], settings: ParetoSettings,
              *, validated: bool) -> tuple[ParetoCandidate, ...]:
    eligible = [c for c in candidates if c.validated] if validated else [c for c in candidates if c.solve.feasible]
    points = [CaptureRangeResult(c.capture_range, settings.minimum_slope, c.solve.weights,
                                 c.worst_validation_noise if validated else c.training_noise,
                                 c.solve.status) for c in eligible]
    selected = {p.capture_range for p in build_pareto_frontier(points, noise_tolerance=settings.noise_tolerance)}
    return tuple(c for c in eligible if c.capture_range in selected)


def _hypervolume(frontier: tuple[ParetoCandidate, ...], settings: ParetoSettings) -> float:
    """Area of the union of rectangles [0, C] x [noise, reference_noise]."""
    area = 0.0
    previous_capture = 0.0
    # Compute the union explicitly so tolerance-based frontier ordering cannot
    # overcount overlapping rectangles or depend on sweep density.
    points = [(min(c.capture_range, settings.capture_reference), c.worst_validation_noise)
              for c in frontier if c.worst_validation_noise < settings.noise_reference]
    for capture in sorted({c for c, _ in points}):
        noise = min(n for c, n in points if c >= capture)
        area += (capture - previous_capture) / settings.capture_reference * (1 - noise / settings.noise_reference)
        previous_capture = capture
    return float(np.clip(area, 0.0, 1.0))


def solve_inner_pareto(
    development: ChannelDataset,
    held_out: tuple[ChannelDataset, ...],
    settings: ParetoSettings = ParetoSettings(),
) -> ParetoRun:
    """Solve each development QP, validate all solutions, and score the frontier.

    Malformed inputs raise ValueError before solving. Unsupported ranges and
    numerical solver failures are recorded per condition, and the sweep proceeds.
    Held-out IDs must be disjoint from development; the caller is responsible for
    providing independently acquired data and development-only preprocessing.
    """
    held_out = tuple(held_out)
    if not held_out:
        raise ValueError("at least one held-out dataset is required")
    ids = [development.dataset_id, *(data.dataset_id for data in held_out)]
    if len(set(ids)) != len(ids):
        raise ValueError("development and held-out dataset IDs must be distinct")
    if any(data.channel_names != development.channel_names for data in held_out):
        raise ValueError("held-out channel_names and order must match development")
    candidates = []
    for capture in settings.capture_ranges:
        if development.detuning_axis[0] > -capture or development.detuning_axis[-1] < capture:
            result = QPResult("out_of_domain", None, message="Capture range exceeds development grid")
        else:
            result = solve_capture_range_qp(
                development.covariance, development.frequency_slope,
                development.basis_derivative, development.detuning_axis,
                capture_range=capture, minimum_slope=settings.minimum_slope, settings=settings.qp,
            )
        validation = tuple(validate_candidate(result.weights, development, data,
                           capture_range=capture, settings=settings) for data in held_out) if result.feasible else ()
        candidates.append(ParetoCandidate(capture, result,
                          float(np.sqrt(result.noise_variance)) if result.feasible else None, validation))
    candidates = tuple(candidates)
    training_frontier = _frontier(candidates, settings, validated=False)
    frontier = _frontier(candidates, settings, validated=True)
    feasible = sum(c.solve.feasible for c in candidates)
    validated = sum(c.validated for c in candidates)
    summary = ParetoSummary(
        len(candidates), feasible, validated, len(frontier), feasible / len(candidates),
        validated / len(candidates), min((c.worst_validation_noise for c in frontier), default=None),
        max((c.capture_range for c in frontier), default=None), _hypervolume(frontier, settings),
    )
    if feasible == 0:
        status = ("no_feasible_solution" if all(c.solve.status == "infeasible" for c in candidates)
                  else "no_usable_solution")
    elif validated == 0:
        status = "no_validated_candidates"
    elif validated < len(candidates):
        status = "partial_success"
    else:
        status = "success"
    return ParetoRun(settings, development.dataset_id, tuple(ids[1:]), development.channel_names,
                     candidates, training_frontier, frontier, summary, status,
                     _software_versions(settings.qp.solver))
