"""End-to-end C3 basis audit orchestration."""

from dataclasses import dataclass

import numpy as np

from ..sim.mock_experiment import MockExperiment
from .bootstrap import BootstrapSummary, bootstrap_basis_metrics
from .projection import (
    ProjectionMetrics,
    center_scans_at_lock,
    evaluate_held_out_projection,
)
from .subspace import SubspaceMetrics, evaluate_subspace_stability
from .svd import NoiseWhitenedSVD, noise_whitened_residual_svd


@dataclass(frozen=True)
class BasisAuditConfig:
    """Declared thresholds and resampling controls for a C3 audit."""

    residual_energy_threshold: float = 0.01
    residual_repeatability_threshold: float = 0.70
    stable_singular_value_ratio: float = 1.5
    maximum_residual_angle_degrees: float = 35.0
    diagonal_loading: float = 1e-8
    bootstrap_resamples: int = 128
    confidence_level: float = 0.95
    bootstrap_seed: int = 20260831

    def __post_init__(self) -> None:
        if self.residual_energy_threshold < 0.0:
            raise ValueError("residual_energy_threshold must be non-negative")
        if not -1.0 <= self.residual_repeatability_threshold <= 1.0:
            raise ValueError("residual_repeatability_threshold must lie in [-1, 1]")
        if self.stable_singular_value_ratio <= 0.0:
            raise ValueError("stable_singular_value_ratio must be positive")
        if not 0.0 <= self.maximum_residual_angle_degrees <= 90.0:
            raise ValueError("maximum_residual_angle_degrees must lie in [0, 90]")
        if self.diagonal_loading < 0.0:
            raise ValueError("diagonal_loading must be non-negative")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be at least one")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie between zero and one")


@dataclass(frozen=True)
class C3BasisAudit:
    """Complete evidence and decision for one proposed basis expansion."""

    projection: ProjectionMetrics
    subspace: SubspaceMetrics
    noise_whitened_svd: NoiseWhitenedSVD
    bootstrap: BootstrapSummary
    basis_expanding: bool
    decision_checks: dict[str, bool]


def _column_indices(
    columns: tuple[int, ...],
    channel_count: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(columns, dtype=int)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must contain at least one column")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} cannot contain duplicates")
    if np.any(result < 0) or np.any(result >= channel_count):
        raise ValueError(f"{name} contains an out-of-range column")
    return result


def audit_basis_expansion(
    detuning: np.ndarray,
    development_scans: np.ndarray,
    validation_scans: np.ndarray,
    *,
    baseline_columns: tuple[int, ...],
    candidate_columns: tuple[int, ...],
    config: BasisAuditConfig | None = None,
) -> C3BasisAudit:
    """Audit a candidate basis using disjoint development/validation scans."""

    settings = BasisAuditConfig() if config is None else config
    axis = np.asarray(detuning, dtype=float)
    development = np.asarray(development_scans, dtype=float)
    validation = np.asarray(validation_scans, dtype=float)
    if axis.ndim != 1 or axis.size < 3 or not np.all(np.isfinite(axis)):
        raise ValueError("detuning must be a finite one-dimensional array")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("detuning must be strictly increasing")
    if development.ndim != 3 or validation.ndim != 3:
        raise ValueError("scans must have shape (scans, detuning, channels)")
    if development.shape[1:] != validation.shape[1:]:
        raise ValueError("development and validation scan shapes must align")
    if development.shape[1] != axis.size:
        raise ValueError("scan samples must match the detuning axis")
    if not np.all(np.isfinite(development)) or not np.all(np.isfinite(validation)):
        raise ValueError("scans must contain only finite values")

    baseline = _column_indices(
        baseline_columns, development.shape[2], "baseline_columns"
    )
    candidate = _column_indices(
        candidate_columns, development.shape[2], "candidate_columns"
    )
    if np.intersect1d(baseline, candidate).size:
        raise ValueError("baseline and candidate columns must be disjoint")

    lock_index = int(np.argmin(np.abs(axis)))
    development_baseline = development[:, :, baseline]
    development_candidate = development[:, :, candidate]
    validation_baseline = validation[:, :, baseline]
    validation_candidate = validation[:, :, candidate]

    projection = evaluate_held_out_projection(
        development_baseline,
        development_candidate,
        validation_baseline,
        validation_candidate,
        lock_index=lock_index,
    )
    spectrum = noise_whitened_residual_svd(
        projection.development_residuals,
        projection.validation_residuals,
        diagonal_loading=settings.diagonal_loading,
        stable_ratio_threshold=settings.stable_singular_value_ratio,
    )
    subspace = evaluate_subspace_stability(
        center_scans_at_lock(development_baseline, lock_index),
        center_scans_at_lock(development_candidate, lock_index),
        center_scans_at_lock(validation_baseline, lock_index),
        center_scans_at_lock(validation_candidate, lock_index),
        projection.development_residuals,
        projection.validation_residuals,
    )
    bootstrap = bootstrap_basis_metrics(
        development_baseline,
        development_candidate,
        validation_baseline,
        validation_candidate,
        lock_index=lock_index,
        resamples=settings.bootstrap_resamples,
        confidence_level=settings.confidence_level,
        seed=settings.bootstrap_seed,
        diagonal_loading=settings.diagonal_loading,
        stable_ratio_threshold=settings.stable_singular_value_ratio,
    )

    checks = {
        "nontrivial_held_out_residual": (
            projection.normalized_residual_energy
            >= settings.residual_energy_threshold
        ),
        "repeatable_held_out_residual": (
            projection.residual_repeatability
            >= settings.residual_repeatability_threshold
        ),
        "stable_whitened_direction": spectrum.stable_rank > 0,
        "stable_residual_subspace": (
            subspace.maximum_residual_angle_degrees
            <= settings.maximum_residual_angle_degrees
        ),
        "bootstrap_stable_direction": (
            bootstrap.leading_singular_value_ratio.lower
            >= settings.stable_singular_value_ratio
        ),
    }
    return C3BasisAudit(
        projection=projection,
        subspace=subspace,
        noise_whitened_svd=spectrum,
        bootstrap=bootstrap,
        basis_expanding=all(checks.values()),
        decision_checks=checks,
    )


def audit_mock_experiment(
    experiment: MockExperiment,
    *,
    config: BasisAuditConfig | None = None,
) -> C3BasisAudit:
    """Run the C3 audit on the baseline and IM pairs from a mock experiment."""

    scans = np.stack([scan.channels for scan in experiment.scans])
    split = scans.shape[0] // 2
    if split < 2 or scans.shape[0] - split < 2:
        raise ValueError("mock experiment needs at least four scans for held-out auditing")
    return audit_basis_expansion(
        experiment.detuning,
        scans[:split],
        scans[split:],
        baseline_columns=(0, 1),
        candidate_columns=(2, 3),
        config=config,
    )
