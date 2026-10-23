"""Held-out basis-expansion diagnostics for synthetic IM channels."""

from dataclasses import dataclass

import numpy as np

from .sim.mock_experiment import MockExperiment


@dataclass(frozen=True)
class BasisAudit:
    """Cross-validated projection result for an IM channel pair."""

    normalized_residual_energy: float
    residual_repeatability: float
    basis_expanding: bool
    projection_coefficients: np.ndarray


def _center_at_lock(channels: np.ndarray, lock_index: int) -> np.ndarray:
    return channels - channels[lock_index]


def audit_im_basis(
    experiment: MockExperiment,
    *,
    residual_threshold: float = 0.01,
    repeatability_threshold: float = 0.80,
) -> BasisAudit:
    """Fit IM channels from the baseline and score residuals on held-out scans."""
    if not 0.0 <= residual_threshold < 1.0:
        raise ValueError("residual_threshold must lie in [0, 1)")
    if not 0.0 <= repeatability_threshold <= 1.0:
        raise ValueError("repeatability_threshold must lie in [0, 1]")

    lock_index = int(np.argmin(np.abs(experiment.detuning)))
    split_index = len(experiment.scans) // 2
    development = experiment.scans[:split_index]
    validation = experiment.scans[split_index:]

    development_channels = [
        _center_at_lock(scan.channels, lock_index) for scan in development
    ]
    baseline_development = np.concatenate(
        [channels[:, :2] for channels in development_channels],
        axis=0,
    )
    im_development = np.concatenate(
        [channels[:, 2:] for channels in development_channels],
        axis=0,
    )
    coefficients = np.linalg.lstsq(
        baseline_development,
        im_development,
        rcond=None,
    )[0]

    validation_residuals: list[np.ndarray] = []
    normalized_energies: list[float] = []
    for scan in validation:
        channels = _center_at_lock(scan.channels, lock_index)
        residual = channels[:, 2:] - channels[:, :2] @ coefficients
        im_energy = float(np.sum(channels[:, 2:] ** 2))
        if im_energy <= 0.0:
            raise ValueError("validation IM channels contain no signal energy")
        validation_residuals.append(residual.ravel())
        normalized_energies.append(float(np.sum(residual**2) / im_energy))

    mean_residual = np.mean(validation_residuals, axis=0)
    repeatability_values: list[float] = []
    for residual in validation_residuals:
        if np.std(residual) == 0.0 or np.std(mean_residual) == 0.0:
            repeatability_values.append(0.0)
        else:
            repeatability_values.append(
                float(np.corrcoef(residual, mean_residual)[0, 1])
            )

    normalized_residual_energy = float(np.mean(normalized_energies))
    residual_repeatability = float(np.mean(repeatability_values))
    basis_expanding = (
        normalized_residual_energy >= residual_threshold
        and residual_repeatability >= repeatability_threshold
    )

    coefficients = np.array(coefficients, copy=True)
    coefficients.setflags(write=False)
    return BasisAudit(
        normalized_residual_energy=normalized_residual_energy,
        residual_repeatability=residual_repeatability,
        basis_expanding=basis_expanding,
        projection_coefficients=coefficients,
    )

