"""SIMTS Bayesian optimization package."""

from .atomicity import AtomicityResult, check_atomicity
from .basis import d_nu, phi, phi_prime, sigma
from .basis_validation import BasisAudit, audit_im_basis
from .basis_audit import (
    BasisAuditConfig,
    BootstrapInterval,
    BootstrapSummary,
    C3BasisAudit,
    NoiseWhitenedSVD,
    ProjectionMetrics,
    SubspaceMetrics,
    audit_basis_expansion,
    audit_mock_experiment,
    evaluate_held_out_projection,
    noise_whitened_residual_svd,
    principal_angles_degrees,
    rotate_iq_pairs,
)
from .c1 import (
    C1_CAPTURE_RANGES,
    C1_MINIMUM_SLOPE,
    C1RegimeOutcome,
    evaluate_c1_regime,
)
from .dsp import (
    DDCConfig,
    DDCResult,
    DemodulatedScan,
    NEGATIVE_EXPONENTIAL_CONVENTION,
    ChannelReference,
    coherent_ddc,
    demodulate_scan_blocks,
    generate_synthetic_rf_record,
    rectangular_window_leakage,
    reference_frequency_hz,
    reference_phase_rad,
)
from .estimation import (
    CovarianceEstimate,
    DerivativeConfig,
    DerivativeEstimate,
    estimate_noise_covariance,
    estimate_smoothed_derivatives,
)
from .frontier import (
    DEFAULT_CAPTURE_RANGES,
    CaptureRangeResult,
    build_pareto_frontier,
    plot_pareto_frontier,
    sweep_capture_ranges,
)
from .optimization import (
    QPResult,
    QPSettings,
    DEFAULT_CAPTURE_RANGE,
    DEFAULT_MINIMUM_SLOPE,
    calculate_blu_weights,
    solve_capture_range_weights,
    solve_capture_range_qp,
    solve_minimum_noise_weights,
)
from .inner_pareto import (
    SETTINGS_VERSION,
    ChannelDataset,
    ParetoCandidate,
    ParetoRun,
    ParetoSettings,
    ParetoSummary,
    ValidationMetrics,
    solve_inner_pareto,
    validate_candidate,
)
from .pipeline import ProcessedScan, complex_channels_to_real_basis, process_raw_scan
from .io import (
    DetuningCalibration,
    RawWaveformRecord,
    calibrate_detuning,
    ingest_raw_waveform,
    register_complex_channels,
)
from .synthetic import detuning, dispersive, i1, i2, lorentzian, q1, q2
from .sim import (
    DEFAULT_MOCK_CONFIG,
    DEFAULT_THETA,
    IMMode,
    MockExperiment,
    MockExperimentConfig,
    MockScan,
    MockTheta,
    simulate_mock_experiment,
)

__version__ = "0.1.0"


def __getattr__(name: str):
    # Legacy demonstration outputs remain importable, but ordinary package
    # imports must not depend on a successful synthetic OSQP solve.
    from importlib import import_module

    if name in {"w_blu", "w_capture", "w_qp"}:
        return getattr(import_module(".optimization", __name__), name)
    if name in {"capture_range_sweep", "pareto_front"}:
        return getattr(import_module(".frontier", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SETTINGS_VERSION",
    "ChannelDataset",
    "ParetoCandidate",
    "ParetoRun",
    "ParetoSettings",
    "ParetoSummary",
    "QPResult",
    "QPSettings",
    "ValidationMetrics",
    "solve_inner_pareto",
    "solve_capture_range_qp",
    "validate_candidate",
    "AtomicityResult",
    "BasisAudit",
    "BasisAuditConfig",
    "BootstrapInterval",
    "BootstrapSummary",
    "C1_CAPTURE_RANGES",
    "C1_MINIMUM_SLOPE",
    "C1RegimeOutcome",
    "C3BasisAudit",
    "CaptureRangeResult",
    "ChannelReference",
    "CovarianceEstimate",
    "DDCConfig",
    "DDCResult",
    "DEFAULT_CAPTURE_RANGES",
    "DEFAULT_CAPTURE_RANGE",
    "DEFAULT_MINIMUM_SLOPE",
    "DEFAULT_MOCK_CONFIG",
    "DEFAULT_THETA",
    "DemodulatedScan",
    "DerivativeConfig",
    "DerivativeEstimate",
    "DetuningCalibration",
    "IMMode",
    "MockExperiment",
    "MockExperimentConfig",
    "MockScan",
    "MockTheta",
    "NEGATIVE_EXPONENTIAL_CONVENTION",
    "NoiseWhitenedSVD",
    "ProcessedScan",
    "ProjectionMetrics",
    "RawWaveformRecord",
    "SubspaceMetrics",
    "audit_basis_expansion",
    "audit_im_basis",
    "audit_mock_experiment",
    "calculate_blu_weights",
    "build_pareto_frontier",
    "capture_range_sweep",
    "calibrate_detuning",
    "check_atomicity",
    "coherent_ddc",
    "complex_channels_to_real_basis",
    "d_nu",
    "detuning",
    "dispersive",
    "demodulate_scan_blocks",
    "estimate_noise_covariance",
    "estimate_smoothed_derivatives",
    "evaluate_c1_regime",
    "evaluate_held_out_projection",
    "i1",
    "i2",
    "ingest_raw_waveform",
    "lorentzian",
    "noise_whitened_residual_svd",
    "pareto_front",
    "phi",
    "phi_prime",
    "q1",
    "q2",
    "rectangular_window_leakage",
    "reference_frequency_hz",
    "reference_phase_rad",
    "register_complex_channels",
    "sigma",
    "plot_pareto_frontier",
    "solve_capture_range_weights",
    "solve_minimum_noise_weights",
    "simulate_mock_experiment",
    "generate_synthetic_rf_record",
    "sweep_capture_ranges",
    "w_blu",
    "w_capture",
    "w_qp",
    "process_raw_scan",
    "principal_angles_degrees",
    "rotate_iq_pairs",
]
