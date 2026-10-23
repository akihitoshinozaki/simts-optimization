"""Minimum-noise channel-weight calculations."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .basis import d_nu, phi_prime, sigma
from .synthetic import detuning


DEFAULT_CAPTURE_RANGE = 1.0
DEFAULT_MINIMUM_SLOPE = 0.4


def _validate_inputs(
    covariance: np.ndarray,
    frequency_slope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(covariance) or np.iscomplexobj(frequency_slope):
        raise ValueError("covariance and frequency_slope must be real")
    covariance = np.asarray(covariance, dtype=float)
    frequency_slope = np.asarray(frequency_slope, dtype=float)

    if frequency_slope.ndim != 1 or frequency_slope.size == 0:
        raise ValueError("frequency_slope must be one-dimensional")
    if covariance.shape != (frequency_slope.size, frequency_slope.size):
        raise ValueError("covariance shape must match the frequency-slope vector")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(frequency_slope)):
        raise ValueError("covariance and frequency_slope must be finite")
    if not np.any(frequency_slope):
        raise ValueError("frequency_slope must contain a nonzero direction")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")

    covariance = covariance / 2.0 + covariance.T / 2.0
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error

    return covariance, frequency_slope


def calculate_blu_weights(
    covariance: np.ndarray,
    frequency_slope: np.ndarray,
) -> np.ndarray:
    """Calculate the analytical best linear unbiased (BLU) weights."""
    covariance, frequency_slope = _validate_inputs(covariance, frequency_slope)

    # Solve Sigma x = d_nu instead of explicitly forming Sigma^{-1}.
    inverse_covariance_slope = np.linalg.solve(covariance, frequency_slope)
    normalization = frequency_slope @ inverse_covariance_slope
    if normalization <= 0.0:
        raise ValueError("frequency_slope must contain a nonzero direction")

    return inverse_covariance_slope / normalization


def solve_minimum_noise_weights(
    covariance: np.ndarray,
    frequency_slope: np.ndarray,
) -> np.ndarray:
    """Solve the unit-gain minimum-noise quadratic program with OSQP."""
    covariance, frequency_slope = _validate_inputs(covariance, frequency_slope)

    weights = cp.Variable(frequency_slope.size)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(weights, covariance)),
        [weights @ frequency_slope == 1.0],
    )
    problem.solve(
        solver=cp.OSQP,
        eps_abs=1e-10,
        eps_rel=1e-10,
        polishing=True,
        verbose=False,
    )

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"OSQP failed with status: {problem.status}")
    if weights.value is None:
        raise RuntimeError("OSQP did not return a weight vector")

    return np.asarray(weights.value, dtype=float)


@dataclass(frozen=True)
class QPSettings:
    """Explicit backend and numerical acceptance policy for the C0 QP."""

    solver: str = "OSQP"
    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-7
    max_iterations: int = 100_000
    accept_inaccurate: bool = False

    def __post_init__(self) -> None:
        if self.solver not in {"OSQP", "CLARABEL"}:
            raise ValueError("solver must be OSQP or CLARABEL")
        for name in ("absolute_tolerance", "relative_tolerance", "feasibility_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if type(self.accept_inaccurate) is not bool:
            raise ValueError("accept_inaccurate must be boolean")


@dataclass(frozen=True)
class QPResult:
    """Numerically checked solution, or a failure with no usable weights."""

    status: str
    solver_status: str | None
    weights: np.ndarray | None = None
    noise_variance: float | None = None
    gain_error: float | None = None
    slope_violation: float | None = None
    iterations: int | None = None
    message: str | None = None

    @property
    def feasible(self) -> bool:
        return self.weights is not None


def _validate_derivatives(
    basis_derivative: np.ndarray, detuning_axis: np.ndarray, channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(basis_derivative) or np.iscomplexobj(detuning_axis):
        raise ValueError("basis_derivative and detuning_axis must be real")
    derivative = np.asarray(basis_derivative, dtype=float)
    axis = np.asarray(detuning_axis, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError("detuning_axis must be a finite vector with at least two samples")
    if not np.all(np.diff(axis) > 0):
        raise ValueError("detuning_axis must be strictly increasing")
    if axis[0] > 0 or axis[-1] < 0:
        raise ValueError("detuning_axis must cover zero detuning")
    if derivative.shape != (axis.size, channels):
        raise ValueError("basis_derivative shape must match detuning and channels")
    if not np.all(np.isfinite(derivative)):
        raise ValueError("basis_derivative must be finite")
    return derivative, axis


def _capture_samples(
    values: np.ndarray, axis: np.ndarray, capture_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Include exact interval endpoints under piecewise-linear interpolation."""
    if not np.isfinite(capture_range) or capture_range < 0:
        raise ValueError("capture_range must be finite and nonnegative")
    if axis[0] > -capture_range or axis[-1] < capture_range:
        raise ValueError("capture_range extends beyond the detuning axis")
    grid = np.unique(np.concatenate((
        [-capture_range, 0.0, capture_range],
        axis[np.abs(axis) <= capture_range],
    )))
    return grid, np.column_stack([
        np.interp(grid, axis, column) for column in values.T
    ])


def solve_capture_range_qp(
    covariance: np.ndarray,
    frequency_slope: np.ndarray,
    basis_derivative: np.ndarray,
    detuning_axis: np.ndarray,
    *,
    capture_range: float,
    minimum_slope: float,
    settings: QPSettings = QPSettings(),
) -> QPResult:
    """Solve C0 with explicit settings; invalid input raises, solve failures return.

    The derivative model is piecewise linear. Constraints include zero and both
    exact capture endpoints, even when these are absent from the input grid.
    """
    covariance, frequency_slope = _validate_inputs(covariance, frequency_slope)
    derivative, axis = _validate_derivatives(
        basis_derivative, detuning_axis, frequency_slope.size,
    )
    if not np.isfinite(minimum_slope) or minimum_slope < 0:
        raise ValueError("minimum_slope must be finite and nonnegative")
    _, capture_derivative = _capture_samples(derivative, axis, capture_range)
    weights = cp.Variable(frequency_slope.size)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(weights, cp.psd_wrap(covariance))),
        [weights @ frequency_slope == 1.0,
         capture_derivative @ weights >= minimum_slope],
    )
    if settings.solver == "OSQP":
        options = dict(eps_abs=settings.absolute_tolerance,
                       eps_rel=settings.relative_tolerance,
                       max_iter=settings.max_iterations, polishing=True)
    else:
        options = dict(tol_gap_abs=settings.absolute_tolerance,
                       tol_gap_rel=settings.relative_tolerance,
                       tol_feas=settings.absolute_tolerance,
                       max_iter=settings.max_iterations)
    try:
        problem.solve(solver=settings.solver, warm_start=False, verbose=False, **options)
    except cp.error.SolverError as error:
        return QPResult("solver_error", None, message=str(error))

    raw_status = str(problem.status)
    iterations = problem.solver_stats.num_iters
    accepted = {cp.OPTIMAL}
    if settings.accept_inaccurate:
        accepted.add(cp.OPTIMAL_INACCURATE)
    if raw_status not in accepted:
        return QPResult(raw_status, raw_status, iterations=iterations,
                        message="Solver did not return an accepted optimum")
    if weights.value is None:
        return QPResult("invalid_solution", raw_status, iterations=iterations,
                        message="Solver did not return weights")
    value = np.array(weights.value, dtype=float, copy=True)
    if value.shape != frequency_slope.shape or not np.all(np.isfinite(value)):
        return QPResult("invalid_solution", raw_status, iterations=iterations,
                        message="Solver returned malformed or nonfinite weights")
    gain_error = float(abs(value @ frequency_slope - 1.0))
    slope_violation = float(max(0.0, minimum_slope - np.min(capture_derivative @ value)))
    variance = float(value @ covariance @ value)
    if (not np.isfinite(gain_error) or not np.isfinite(slope_violation)
            or max(gain_error, slope_violation) > settings.feasibility_tolerance
            or not np.isfinite(variance) or variance <= 0):
        return QPResult("invalid_solution", raw_status, gain_error=gain_error,
                        slope_violation=slope_violation, iterations=iterations,
                        message="Returned weights failed independent numerical checks")
    value.setflags(write=False)
    return QPResult(raw_status, raw_status, value, variance, gain_error,
                    slope_violation, iterations)


def solve_capture_range_weights(
    covariance: np.ndarray,
    frequency_slope: np.ndarray,
    basis_derivative: np.ndarray,
    detuning_axis: np.ndarray,
    *,
    capture_range: float,
    minimum_slope: float,
) -> np.ndarray:
    """Compatibility wrapper returning weights or raising on a failed C0 QP."""
    result = solve_capture_range_qp(
        covariance, frequency_slope, basis_derivative, detuning_axis,
        capture_range=capture_range, minimum_slope=minimum_slope,
        settings=QPSettings(accept_inaccurate=True),
    )
    if not result.feasible:
        raise RuntimeError(f"capture-range optimization failed with status: {result.status}")
    return result.weights


def __getattr__(name: str):
    """Keep the C0 demonstration weights available without solving on import."""
    if name == "w_blu":
        value = calculate_blu_weights(sigma, d_nu)
    elif name == "w_qp":
        value = solve_minimum_noise_weights(sigma, d_nu)
    elif name == "w_capture":
        value = solve_capture_range_weights(
            sigma, d_nu, phi_prime, detuning,
            capture_range=DEFAULT_CAPTURE_RANGE, minimum_slope=DEFAULT_MINIMUM_SLOPE,
        )
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
