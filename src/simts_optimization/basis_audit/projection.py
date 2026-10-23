"""Cross-validated projection residuals for basis-expansion tests."""

from dataclasses import dataclass

import numpy as np


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def validate_scan_pair(
    development_baseline: np.ndarray,
    development_candidate: np.ndarray,
    validation_baseline: np.ndarray,
    validation_candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return four ``(scan, detuning, channel)`` arrays."""

    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (
            development_baseline,
            development_candidate,
            validation_baseline,
            validation_candidate,
        )
    )
    if any(values.ndim != 3 for values in arrays):
        raise ValueError("basis data must have shape (scans, detuning, channels)")

    dev_base, dev_candidate, val_base, val_candidate = arrays
    if dev_base.shape[:2] != dev_candidate.shape[:2]:
        raise ValueError("development baseline and candidate shapes do not align")
    if val_base.shape[:2] != val_candidate.shape[:2]:
        raise ValueError("validation baseline and candidate shapes do not align")
    if dev_base.shape[1] != val_base.shape[1]:
        raise ValueError("development and validation detuning grids must align")
    if dev_base.shape[2] != val_base.shape[2]:
        raise ValueError("baseline channel counts must match")
    if dev_candidate.shape[2] != val_candidate.shape[2]:
        raise ValueError("candidate channel counts must match")
    if min(dev_base.shape[0], val_base.shape[0]) < 2:
        raise ValueError("development and validation sets each need at least two scans")
    if min(dev_base.shape[1], dev_base.shape[2], dev_candidate.shape[2]) < 1:
        raise ValueError("basis data cannot contain an empty dimension")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("basis data must contain only finite values")
    return arrays


def center_scans_at_lock(scans: np.ndarray, lock_index: int) -> np.ndarray:
    """Subtract each scan's value at the declared lock point."""

    values = np.asarray(scans, dtype=float)
    if not 0 <= lock_index < values.shape[1]:
        raise ValueError("lock_index is outside the detuning axis")
    return values - values[:, lock_index : lock_index + 1, :]


def fit_projection_coefficients(
    baseline_scans: np.ndarray,
    candidate_scans: np.ndarray,
) -> np.ndarray:
    """Fit candidate-from-baseline coefficients using development scans only."""

    baseline = np.asarray(baseline_scans, dtype=float)
    candidate = np.asarray(candidate_scans, dtype=float)
    design = baseline.reshape(-1, baseline.shape[-1])
    target = candidate.reshape(-1, candidate.shape[-1])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def projection_residuals(
    baseline_scans: np.ndarray,
    candidate_scans: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Return candidate content that is not reconstructed by the baseline."""

    return np.asarray(candidate_scans) - np.asarray(baseline_scans) @ coefficients


def _normalized_residual_energy(
    candidate_scans: np.ndarray,
    residuals: np.ndarray,
) -> float:
    denominator = float(np.sum(np.square(candidate_scans)))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(np.square(residuals)) / denominator)


def _leave_one_out_repeatability(residuals: np.ndarray) -> float:
    """Average cosine agreement with a leave-one-scan-out residual mean."""

    flattened = np.asarray(residuals, dtype=float).reshape(residuals.shape[0], -1)
    total = flattened.sum(axis=0)
    similarities: list[float] = []
    for scan_index, scan in enumerate(flattened):
        reference = (total - scan) / (flattened.shape[0] - 1)
        denominator = float(np.linalg.norm(scan) * np.linalg.norm(reference))
        similarities.append(
            0.0 if denominator <= np.finfo(float).eps else float(scan @ reference / denominator)
        )
    return float(np.mean(similarities))


@dataclass(frozen=True)
class ProjectionMetrics:
    """Development fit and its held-out validation scores."""

    coefficients: np.ndarray
    development_residuals: np.ndarray
    validation_residuals: np.ndarray
    normalized_residual_energy: float
    residual_repeatability: float


def evaluate_held_out_projection(
    development_baseline: np.ndarray,
    development_candidate: np.ndarray,
    validation_baseline: np.ndarray,
    validation_candidate: np.ndarray,
    *,
    lock_index: int,
) -> ProjectionMetrics:
    """Fit on development scans and score projection residuals on held-out scans."""

    dev_base, dev_candidate, val_base, val_candidate = validate_scan_pair(
        development_baseline,
        development_candidate,
        validation_baseline,
        validation_candidate,
    )
    dev_base = center_scans_at_lock(dev_base, lock_index)
    dev_candidate = center_scans_at_lock(dev_candidate, lock_index)
    val_base = center_scans_at_lock(val_base, lock_index)
    val_candidate = center_scans_at_lock(val_candidate, lock_index)

    coefficients = fit_projection_coefficients(dev_base, dev_candidate)
    development_residuals = projection_residuals(dev_base, dev_candidate, coefficients)
    validation_residuals = projection_residuals(val_base, val_candidate, coefficients)

    return ProjectionMetrics(
        coefficients=_readonly(coefficients),
        development_residuals=_readonly(development_residuals),
        validation_residuals=_readonly(validation_residuals),
        normalized_residual_energy=_normalized_residual_energy(
            val_candidate, validation_residuals
        ),
        residual_repeatability=_leave_one_out_repeatability(validation_residuals),
    )
