"""Numerical coherent-demodulation reference implementation."""

from .ddc import (
    DDCConfig,
    DDCResult,
    DemodulatedScan,
    coherent_ddc,
    demodulate_scan_blocks,
    rectangular_window_leakage,
)
from .phase import (
    NEGATIVE_EXPONENTIAL_CONVENTION,
    ChannelReference,
    reference_frequency_hz,
    reference_phase_rad,
)
from .synthetic_rf import generate_synthetic_rf_record

__all__ = [
    "DDCConfig",
    "DDCResult",
    "DemodulatedScan",
    "NEGATIVE_EXPONENTIAL_CONVENTION",
    "ChannelReference",
    "coherent_ddc",
    "demodulate_scan_blocks",
    "generate_synthetic_rf_record",
    "rectangular_window_leakage",
    "reference_frequency_hz",
    "reference_phase_rad",
]

