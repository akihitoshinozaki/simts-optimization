"""Declared derivative and covariance estimators for processed scans."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


def _read_only(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DerivativeConfig:
    """Fixed local-polynomial smoothing rule used before differentiation."""

    window_length: int = 11
    polynomial_order: int = 3


@dataclass(frozen=True)
class DerivativeEstimate:
    """Smoothed basis and its derivative on a registered detuning grid."""

    smoothed_basis: np.ndarray
    basis_derivative: np.ndarray
    frequency_slope: np.ndarray
    lock_index: int
    method: str


@dataclass(frozen=True)
class CovarianceEstimate:
    """Repeated-scan covariance with conditioning and regularization metadata."""

    covariance: np.ndarray
    unregularized_covariance: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    diagonal_loading: float
    residual_sample_count: int
    method: str


def estimate_smoothed_derivatives(
    detuning_hz: np.ndarray,
    basis: np.ndarray,
    config: DerivativeConfig = DerivativeConfig(),
) -> DerivativeEstimate:
    """Smooth each channel with a fixed Savitzky-Golay local polynomial."""
    detuning = np.asarray(detuning_hz, dtype=float)
    values = np.asarray(basis, dtype=float)
    if detuning.ndim != 1 or detuning.size < 5:
        raise ValueError("detuning_hz must contain at least five samples")
    if values.ndim != 2 or values.shape[0] != detuning.size:
        raise ValueError("basis must have one row per detuning sample")
    if not np.all(np.diff(detuning) > 0.0):
        raise ValueError("detuning_hz must be strictly increasing")
    if config.window_length < 5 or config.window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer of at least 5")
    if config.window_length > detuning.size:
        raise ValueError("window_length cannot exceed the detuning grid")
    if not 1 <= config.polynomial_order < config.window_length:
        raise ValueError("polynomial_order must be below window_length")

    steps = np.diff(detuning)
    step = float(np.median(steps))
    if not np.allclose(steps, step, rtol=1e-6, atol=0.0):
        raise ValueError("Savitzky-Golay estimation requires uniform detuning")

    smoothed = savgol_filter(
        values,
        window_length=config.window_length,
        polyorder=config.polynomial_order,
        axis=0,
        mode="interp",
    )
    derivative = savgol_filter(
        values,
        window_length=config.window_length,
        polyorder=config.polynomial_order,
        deriv=1,
        delta=step,
        axis=0,
        mode="interp",
    )
    lock_index = int(np.argmin(np.abs(detuning)))
    return DerivativeEstimate(
        smoothed_basis=_read_only(smoothed),
        basis_derivative=_read_only(derivative),
        frequency_slope=_read_only(derivative[lock_index]),
        lock_index=lock_index,
        method=(
            "Savitzky-Golay "
            f"(window={config.window_length}, order={config.polynomial_order})"
        ),
    )


def estimate_noise_covariance(
    repeated_basis_scans: np.ndarray,
    *,
    diagonal_loading: float = 1e-12,
) -> CovarianceEstimate:
    """Estimate full channel covariance after removing the repeat mean shape."""
    scans = np.asarray(repeated_basis_scans, dtype=float)
    if scans.ndim != 3 or scans.shape[0] < 2 or scans.shape[2] < 1:
        raise ValueError(
            "repeated_basis_scans must have shape (repeats, detuning, channels)"
        )
    if diagonal_loading < 0.0:
        raise ValueError("diagonal_loading must be nonnegative")
    if not np.all(np.isfinite(scans)):
        raise ValueError("repeated basis scans must be finite")

    residuals = scans - np.mean(scans, axis=0, keepdims=True)
    residual_samples = residuals.reshape(-1, scans.shape[2])
    degrees_of_freedom = (scans.shape[0] - 1) * scans.shape[1]
    unregularized = np.atleast_2d(
        residual_samples.T @ residual_samples / degrees_of_freedom
    )
    covariance = unregularized + diagonal_loading * np.eye(scans.shape[2])
    eigenvalues = np.linalg.eigvalsh(covariance)
    condition_number = float(np.linalg.cond(covariance))

    return CovarianceEstimate(
        covariance=_read_only(covariance),
        unregularized_covariance=_read_only(unregularized),
        eigenvalues=_read_only(eigenvalues),
        condition_number=condition_number,
        diagonal_loading=float(diagonal_loading),
        residual_sample_count=residual_samples.shape[0],
        method="repeat-mean residual covariance with diagonal loading",
    )
