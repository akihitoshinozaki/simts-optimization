"""End-to-end raw waveform to registered channel-basis processing."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .dsp.ddc import DemodulatedScan, demodulate_scan_blocks
from .dsp.phase import NEGATIVE_EXPONENTIAL_CONVENTION, ChannelReference
from .estimation import (
    DerivativeConfig,
    DerivativeEstimate,
    estimate_smoothed_derivatives,
)
from .io.calibration import register_complex_channels
from .io.records import RawWaveformRecord


def _read_only(array: np.ndarray, *, dtype: type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ProcessedScan:
    """Registered complex channels and real I/Q basis derived from raw ADC data."""

    run_id: str
    detuning_hz: np.ndarray
    complex_channels: np.ndarray
    basis: np.ndarray
    channel_references: tuple[ChannelReference, ...]
    derivative_estimate: DerivativeEstimate
    phase_convention: str = NEGATIVE_EXPONENTIAL_CONVENTION


def complex_channels_to_real_basis(complex_channels: np.ndarray) -> np.ndarray:
    """Interleave each complex channel as [I1, Q1, I2, Q2, ...]."""
    channels = np.asarray(complex_channels, dtype=complex)
    if channels.ndim != 2:
        raise ValueError("complex_channels must be a two-dimensional matrix")

    basis = np.empty((channels.shape[0], 2 * channels.shape[1]), dtype=float)
    basis[:, 0::2] = channels.real
    basis[:, 1::2] = channels.imag
    return basis


def process_raw_scan(
    record: RawWaveformRecord,
    channel_references: Sequence[ChannelReference],
    *,
    derivative_config: DerivativeConfig = DerivativeConfig(),
    target_detuning_hz: np.ndarray | None = None,
) -> ProcessedScan:
    """Demodulate, register, assemble I/Q, and estimate channel derivatives."""
    demodulated: DemodulatedScan = demodulate_scan_blocks(
        record,
        channel_references,
    )
    source_detuning = demodulated.calibrated_detuning_hz
    complex_channels = demodulated.complex_channels

    if target_detuning_hz is not None:
        target = np.asarray(target_detuning_hz, dtype=float)
        complex_channels = register_complex_channels(
            source_detuning,
            complex_channels,
            target,
        )
        detuning = target
    elif np.all(np.diff(source_detuning) < 0.0):
        detuning = source_detuning[::-1]
        complex_channels = complex_channels[::-1]
    elif np.all(np.diff(source_detuning) > 0.0):
        detuning = source_detuning
    else:
        raise ValueError("calibrated detuning must be strictly monotonic")

    basis = complex_channels_to_real_basis(complex_channels)
    derivative_estimate = estimate_smoothed_derivatives(
        detuning,
        basis,
        derivative_config,
    )
    return ProcessedScan(
        run_id=record.run_id,
        detuning_hz=_read_only(detuning, dtype=float),
        complex_channels=_read_only(complex_channels, dtype=complex),
        basis=_read_only(basis, dtype=float),
        channel_references=tuple(channel_references),
        derivative_estimate=derivative_estimate,
    )

