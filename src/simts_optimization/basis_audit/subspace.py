"""Principal-angle diagnostics for development/validation stability."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import subspace_angles


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def _numerical_column_space(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    return left[:, :rank]


def principal_angles_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return principal angles between two sampled function subspaces."""

    left_space = _numerical_column_space(left)
    right_space = _numerical_column_space(right)
    if left_space.shape[0] != right_space.shape[0]:
        raise ValueError("subspaces must be sampled on the same detuning grid")
    if left_space.shape[1] == 0 or right_space.shape[1] == 0:
        return _readonly(np.array([90.0]))
    return _readonly(np.degrees(subspace_angles(left_space, right_space)))


@dataclass(frozen=True)
class SubspaceMetrics:
    """Repeatability of baseline, augmented, and residual subspaces."""

    baseline_angles_degrees: np.ndarray
    augmented_angles_degrees: np.ndarray
    residual_angles_degrees: np.ndarray
    maximum_residual_angle_degrees: float


def evaluate_subspace_stability(
    development_baseline: np.ndarray,
    development_candidate: np.ndarray,
    validation_baseline: np.ndarray,
    validation_candidate: np.ndarray,
    development_residuals: np.ndarray,
    validation_residuals: np.ndarray,
) -> SubspaceMetrics:
    """Compare mean function subspaces across the held-out split."""

    development_baseline_mean = np.mean(development_baseline, axis=0)
    validation_baseline_mean = np.mean(validation_baseline, axis=0)
    development_augmented_mean = np.mean(
        np.concatenate((development_baseline, development_candidate), axis=2), axis=0
    )
    validation_augmented_mean = np.mean(
        np.concatenate((validation_baseline, validation_candidate), axis=2), axis=0
    )
    development_residual_mean = np.mean(development_residuals, axis=0)
    validation_residual_mean = np.mean(validation_residuals, axis=0)

    baseline_angles = principal_angles_degrees(
        development_baseline_mean, validation_baseline_mean
    )
    augmented_angles = principal_angles_degrees(
        development_augmented_mean, validation_augmented_mean
    )
    residual_angles = principal_angles_degrees(
        development_residual_mean, validation_residual_mean
    )
    return SubspaceMetrics(
        baseline_angles_degrees=baseline_angles,
        augmented_angles_degrees=augmented_angles,
        residual_angles_degrees=residual_angles,
        maximum_residual_angle_degrees=float(np.max(residual_angles)),
    )
