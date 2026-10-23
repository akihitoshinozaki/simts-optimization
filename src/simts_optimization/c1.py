"""Milestone C1 regime evaluation and acceptance outcomes."""

from dataclasses import dataclass

import numpy as np

from .atomicity import AtomicityResult, check_atomicity
from .basis_validation import BasisAudit, audit_im_basis
from .frontier import CaptureRangeResult, build_pareto_frontier, sweep_capture_ranges
from .sim.mock_experiment import MockExperiment


C1_CAPTURE_RANGES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
C1_MINIMUM_SLOPE = 0.20


@dataclass(frozen=True)
class C1RegimeOutcome:
    """Basis, atomicity, and frontier decisions for one synthetic regime."""

    basis_audit: BasisAudit
    atomicity: AtomicityResult
    im_accepted: bool
    baseline_front: tuple[CaptureRangeResult, ...]
    selected_front: tuple[CaptureRangeResult, ...]
    frontier_expanded: bool


def _five_point_slope(basis: np.ndarray, detuning: np.ndarray) -> np.ndarray:
    lock_index = int(np.argmin(np.abs(detuning)))
    step = detuning[lock_index + 1] - detuning[lock_index]
    return (
        basis[lock_index - 2]
        - 8.0 * basis[lock_index - 1]
        + 8.0 * basis[lock_index + 1]
        - basis[lock_index + 2]
    ) / (12.0 * step)


def _build_front(
    basis: np.ndarray,
    covariance: np.ndarray,
    detuning: np.ndarray,
) -> tuple[CaptureRangeResult, ...]:
    lock_index = int(np.argmin(np.abs(detuning)))
    centered_basis = basis - basis[lock_index]
    frequency_slope = _five_point_slope(centered_basis, detuning)
    basis_derivative = np.gradient(
        centered_basis,
        detuning,
        axis=0,
        edge_order=2,
    )
    basis_derivative[lock_index] = frequency_slope

    sweep = sweep_capture_ranges(
        C1_CAPTURE_RANGES,
        minimum_slope=C1_MINIMUM_SLOPE,
        covariance=covariance,
        frequency_slope=frequency_slope,
        basis_derivative=basis_derivative,
        detuning_axis=detuning,
    )
    return build_pareto_frontier(sweep)


def _maximum_feasible_capture(
    frontier: tuple[CaptureRangeResult, ...],
) -> float:
    return max((point.capture_range for point in frontier), default=0.0)


def evaluate_c1_regime(experiment: MockExperiment) -> C1RegimeOutcome:
    """Apply basis and atomicity gates, then compare accepted Pareto fronts."""
    basis_audit = audit_im_basis(experiment)
    atomicity = check_atomicity(experiment)
    im_accepted = basis_audit.basis_expanding and atomicity.valid

    # C1 uses the simulator's known clean response and covariance to verify the
    # mock physics. Estimation from raw scans is deliberately deferred to C2.
    mean_clean_basis = np.mean(
        [scan.clean_channels for scan in experiment.scans],
        axis=0,
    )
    baseline_front = _build_front(
        mean_clean_basis[:, :2],
        experiment.noise_covariance[:2, :2],
        experiment.detuning,
    )

    if im_accepted:
        selected_front = _build_front(
            mean_clean_basis,
            experiment.noise_covariance,
            experiment.detuning,
        )
    else:
        selected_front = baseline_front

    frontier_expanded = _maximum_feasible_capture(
        selected_front
    ) > _maximum_feasible_capture(baseline_front)
    return C1RegimeOutcome(
        basis_audit=basis_audit,
        atomicity=atomicity,
        im_accepted=im_accepted,
        baseline_front=baseline_front,
        selected_front=selected_front,
        frontier_expanded=frontier_expanded,
    )

