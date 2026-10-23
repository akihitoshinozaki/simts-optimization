"""Scan-resampling uncertainty estimates for C3 basis audits."""

from dataclasses import dataclass

import numpy as np

from .projection import evaluate_held_out_projection
from .svd import noise_whitened_residual_svd


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile confidence interval with its bootstrap median."""

    lower: float
    median: float
    upper: float


@dataclass(frozen=True)
class BootstrapSummary:
    """Bootstrap intervals for key basis-expansion statistics."""

    normalized_residual_energy: BootstrapInterval
    leading_singular_value_ratio: BootstrapInterval
    resamples: int
    confidence_level: float


def _percentile_interval(
    samples: np.ndarray,
    confidence_level: float,
) -> BootstrapInterval:
    tail = (1.0 - confidence_level) / 2.0
    lower, median, upper = np.quantile(samples, (tail, 0.5, 1.0 - tail))
    return BootstrapInterval(float(lower), float(median), float(upper))


def bootstrap_basis_metrics(
    development_baseline: np.ndarray,
    development_candidate: np.ndarray,
    validation_baseline: np.ndarray,
    validation_candidate: np.ndarray,
    *,
    lock_index: int,
    resamples: int,
    confidence_level: float,
    seed: int,
    diagonal_loading: float,
    stable_ratio_threshold: float,
) -> BootstrapSummary:
    """Resample whole scans and recompute held-out residual/SVD statistics."""

    if resamples < 1:
        raise ValueError("resamples must be at least one")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one")

    rng = np.random.default_rng(seed)
    energies = np.empty(resamples, dtype=float)
    leading_ratios = np.empty(resamples, dtype=float)
    for sample_index in range(resamples):
        development_indices = rng.integers(
            development_baseline.shape[0], size=development_baseline.shape[0]
        )
        validation_indices = rng.integers(
            validation_baseline.shape[0], size=validation_baseline.shape[0]
        )
        projection = evaluate_held_out_projection(
            development_baseline[development_indices],
            development_candidate[development_indices],
            validation_baseline[validation_indices],
            validation_candidate[validation_indices],
            lock_index=lock_index,
        )
        spectrum = noise_whitened_residual_svd(
            projection.development_residuals,
            projection.validation_residuals,
            diagonal_loading=diagonal_loading,
            stable_ratio_threshold=stable_ratio_threshold,
        )
        energies[sample_index] = projection.normalized_residual_energy
        leading_ratios[sample_index] = spectrum.singular_value_ratios[0]

    return BootstrapSummary(
        normalized_residual_energy=_percentile_interval(
            energies, confidence_level
        ),
        leading_singular_value_ratio=_percentile_interval(
            leading_ratios, confidence_level
        ),
        resamples=resamples,
        confidence_level=confidence_level,
    )
