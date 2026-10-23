"""Noise-whitened singular-value diagnostics."""

from dataclasses import dataclass

import numpy as np


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def estimate_residual_noise_covariance(
    development_residuals: np.ndarray,
    *,
    diagonal_loading: float,
) -> np.ndarray:
    """Estimate channel covariance from scan-to-scan residual fluctuations."""

    if diagonal_loading < 0.0:
        raise ValueError("diagonal_loading must be non-negative")
    residuals = np.asarray(development_residuals, dtype=float)
    fluctuations = residuals - residuals.mean(axis=0, keepdims=True)
    samples = fluctuations.reshape(-1, residuals.shape[-1])
    covariance = samples.T @ samples / max(samples.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance) / covariance.shape[0]), 1.0)
    covariance += diagonal_loading * scale * np.eye(covariance.shape[0])
    return covariance


def _inverse_square_root(covariance: np.ndarray) -> tuple[np.ndarray, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = np.finfo(float).eps * max(float(eigenvalues[-1]), 1.0)
    clipped = np.maximum(eigenvalues, floor)
    inverse_root = (eigenvectors / np.sqrt(clipped)) @ eigenvectors.T
    return inverse_root, float(clipped[-1] / clipped[0])


@dataclass(frozen=True)
class NoiseWhitenedSVD:
    """Whitened spectrum of the held-out mean projection residual."""

    noise_covariance: np.ndarray
    covariance_condition_number: float
    singular_values: np.ndarray
    random_noise_edge: float
    singular_value_ratios: np.ndarray
    stable_rank: int


def noise_whitened_residual_svd(
    development_residuals: np.ndarray,
    validation_residuals: np.ndarray,
    *,
    diagonal_loading: float = 1e-8,
    stable_ratio_threshold: float = 1.5,
) -> NoiseWhitenedSVD:
    """Whiten the held-out mean and compare it with a random-noise edge.

    The covariance is learned only from development-scan fluctuations.  The
    Marchenko-Pastur-style edge ``sqrt(n_detuning) + sqrt(n_channels)`` keeps a
    mean of pure noise from being counted as a stable basis direction.
    """

    development = np.asarray(development_residuals, dtype=float)
    validation = np.asarray(validation_residuals, dtype=float)
    if development.ndim != 3 or validation.ndim != 3:
        raise ValueError("residual arrays must have shape (scans, detuning, channels)")
    if development.shape[1:] != validation.shape[1:]:
        raise ValueError("development and validation residual shapes must align")
    if stable_ratio_threshold <= 0.0:
        raise ValueError("stable_ratio_threshold must be positive")

    # The audit is about line-shape directions, not arbitrary per-scan DC
    # offsets.  Column centering also prevents noise in a single lock sample
    # from being broadcast into an apparently stable constant direction.
    development = development - development.mean(axis=1, keepdims=True)
    validation = validation - validation.mean(axis=1, keepdims=True)

    covariance = estimate_residual_noise_covariance(
        development, diagonal_loading=diagonal_loading
    )
    inverse_root, condition_number = _inverse_square_root(covariance)
    whitened_mean = (
        np.sqrt(validation.shape[0]) * validation.mean(axis=0) @ inverse_root
    )
    singular_values = np.linalg.svd(whitened_mean, compute_uv=False)
    noise_edge = float(
        np.sqrt(validation.shape[1]) + np.sqrt(validation.shape[2])
    )
    ratios = singular_values / noise_edge
    return NoiseWhitenedSVD(
        noise_covariance=_readonly(covariance),
        covariance_condition_number=condition_number,
        singular_values=_readonly(singular_values),
        random_noise_edge=noise_edge,
        singular_value_ratios=_readonly(ratios),
        stable_rank=int(np.count_nonzero(ratios >= stable_ratio_threshold)),
    )
