"""Capture-range sweeps and Pareto-front construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .basis import d_nu, phi_prime, sigma
from .optimization import DEFAULT_MINIMUM_SLOPE, solve_capture_range_qp
from .synthetic import detuning


DEFAULT_CAPTURE_RANGES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)


@dataclass(frozen=True)
class CaptureRangeResult:
    """One point from the capture-range epsilon-constraint sweep."""

    capture_range: float
    minimum_slope: float
    weights: np.ndarray | None
    equivalent_frequency_noise: float | None
    status: str
    message: str | None = None

    @property
    def feasible(self) -> bool:
        """Whether the optimization returned a usable solution."""
        return self.weights is not None


def sweep_capture_ranges(
    capture_ranges: Iterable[float],
    *,
    minimum_slope: float,
    covariance: np.ndarray = sigma,
    frequency_slope: np.ndarray = d_nu,
    basis_derivative: np.ndarray = phi_prime,
    detuning_axis: np.ndarray = detuning,
) -> tuple[CaptureRangeResult, ...]:
    """Solve the minimum-noise QP for increasing capture requirements."""
    ranges = np.asarray(tuple(capture_ranges), dtype=float)
    if ranges.ndim != 1 or ranges.size == 0:
        raise ValueError("capture_ranges must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(ranges)) or np.any(ranges < 0):
        raise ValueError("capture_ranges must be finite and nonnegative")
    if np.any(np.diff(ranges) <= 0.0):
        raise ValueError("capture_ranges must be strictly increasing")

    results: list[CaptureRangeResult] = []
    for capture_range in ranges:
        solve = solve_capture_range_qp(
            covariance,
            frequency_slope,
            basis_derivative,
            detuning_axis,
            capture_range=float(capture_range),
            minimum_slope=minimum_slope,
        )
        if not solve.feasible:
            results.append(
                CaptureRangeResult(
                    capture_range=float(capture_range),
                    minimum_slope=minimum_slope,
                    weights=None,
                    equivalent_frequency_noise=None,
                    status=solve.status,
                    message=solve.message,
                )
            )
            continue

        weights = solve.weights
        noise_variance = solve.noise_variance
        results.append(
            CaptureRangeResult(
                capture_range=float(capture_range),
                minimum_slope=minimum_slope,
                weights=weights,
                equivalent_frequency_noise=float(np.sqrt(noise_variance)),
                status=solve.status,
            )
        )

    return tuple(results)


def build_pareto_frontier(
    results: Sequence[CaptureRangeResult],
    *,
    noise_tolerance: float = 1e-9,
) -> tuple[CaptureRangeResult, ...]:
    """Remove points dominated in capture range and equivalent noise."""
    if not np.isfinite(noise_tolerance) or noise_tolerance < 0:
        raise ValueError("noise_tolerance must be finite and nonnegative")
    feasible_results = [result for result in results if result.feasible]
    frontier: list[CaptureRangeResult] = []

    for candidate in feasible_results:
        candidate_noise = candidate.equivalent_frequency_noise
        assert candidate_noise is not None

        dominated = False
        for other in feasible_results:
            if other is candidate:
                continue

            other_noise = other.equivalent_frequency_noise
            assert other_noise is not None
            range_is_no_worse = other.capture_range >= candidate.capture_range
            noise_is_no_worse = other_noise <= candidate_noise + noise_tolerance
            strictly_better = (
                other.capture_range > candidate.capture_range
                or other_noise < candidate_noise - noise_tolerance
            )
            if range_is_no_worse and noise_is_no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return tuple(sorted(frontier, key=lambda result: result.capture_range))


def plot_pareto_frontier(
    frontier: Sequence[CaptureRangeResult],
) -> tuple[Figure, Axes]:
    """Plot capture range against equivalent noise in normalized units."""
    import matplotlib.pyplot as plt

    if not frontier:
        raise ValueError("frontier must contain at least one feasible point")

    capture_ranges = [point.capture_range for point in frontier]
    equivalent_noise = [point.equivalent_frequency_noise for point in frontier]

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    axes.plot(capture_ranges, equivalent_noise, marker="o", linewidth=2.0)
    axes.set_title("Capture-range / noise Pareto frontier")
    axes.set_xlabel("Required capture range, C (normalized detuning)")
    axes.set_ylabel("Equivalent frequency noise (normalized detuning)")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure, axes


def __getattr__(name: str):
    """Lazily compute legacy C0 examples so importing C4 never invokes a solver."""
    if name not in {"capture_range_sweep", "pareto_front"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if "capture_range_sweep" not in globals():
        globals()["capture_range_sweep"] = sweep_capture_ranges(
            DEFAULT_CAPTURE_RANGES, minimum_slope=DEFAULT_MINIMUM_SLOPE,
        )
    if name == "pareto_front":
        globals()[name] = build_pareto_frontier(globals()["capture_range_sweep"])
    return globals()[name]
