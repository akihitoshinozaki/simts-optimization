"""Software lock-in and block-decimation routines."""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy import signal

from ..io.records import RawWaveformRecord
from .phase import (
    NEGATIVE_EXPONENTIAL_CONVENTION,
    ChannelReference,
    reference_frequency_hz,
    reference_phase_rad,
)


def _read_only(array: np.ndarray, *, dtype: type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DDCConfig:
    """FIR low-pass and decimation settings for continuous DDC."""

    lowpass_cutoff_hz: float
    decimation: int
    fir_taps: int = 129

    def validate(self, sample_rate_hz: float) -> None:
        if self.lowpass_cutoff_hz <= 0.0:
            raise ValueError("lowpass_cutoff_hz must be positive")
        if self.decimation < 1:
            raise ValueError("decimation must be at least 1")
        if self.fir_taps < 3 or self.fir_taps % 2 == 0:
            raise ValueError("fir_taps must be an odd integer of at least 3")
        output_nyquist_hz = sample_rate_hz / (2.0 * self.decimation)
        if self.lowpass_cutoff_hz >= output_nyquist_hz:
            raise ValueError(
                "lowpass cutoff must be below the decimated Nyquist frequency"
            )


@dataclass(frozen=True)
class DDCResult:
    """Complex baseband samples produced by the software reference DDC."""

    values: np.ndarray
    sample_indices: np.ndarray
    output_sample_rate_hz: float
    reference_frequency_hz: float
    reference_phase_rad: float
    convention: str = NEGATIVE_EXPONENTIAL_CONVENTION


@dataclass(frozen=True)
class DemodulatedScan:
    """One complex value per ADC block and requested channel reference."""

    complex_channels: np.ndarray
    channel_references: tuple[ChannelReference, ...]
    calibrated_detuning_hz: np.ndarray
    convention: str = NEGATIVE_EXPONENTIAL_CONVENTION


def coherent_ddc(
    raw_adc: np.ndarray,
    *,
    sample_rate_hz: float,
    reference_frequency_hz: float,
    reference_phase_rad: float = 0.0,
    config: DDCConfig,
    start_sample_index: int = 0,
) -> DDCResult:
    """Mix, low-pass filter, and decimate a continuous real ADC waveform."""
    samples = np.asarray(raw_adc, dtype=float)
    if samples.ndim != 1 or samples.size <= config.fir_taps:
        raise ValueError("raw_adc must be one-dimensional and longer than the FIR")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not 0.0 < abs(reference_frequency_hz) < sample_rate_hz / 2.0:
        raise ValueError("reference frequency must lie inside the ADC Nyquist band")
    config.validate(sample_rate_hz)

    local_indices = np.arange(samples.size, dtype=np.int64)
    absolute_indices = start_sample_index + local_indices
    phase = (
        2.0 * np.pi * reference_frequency_hz * absolute_indices / sample_rate_hz
        + reference_phase_rad
    )
    mixed = 2.0 * samples * np.exp(-1j * phase)
    taps = signal.firwin(
        config.fir_taps,
        config.lowpass_cutoff_hz,
        fs=sample_rate_hz,
    )
    filtered = signal.fftconvolve(mixed, taps, mode="same")

    edge = (config.fir_taps - 1) // 2
    retained_local_indices = np.arange(
        edge,
        samples.size - edge,
        config.decimation,
        dtype=np.int64,
    )
    retained_absolute_indices = start_sample_index + retained_local_indices
    return DDCResult(
        values=_read_only(filtered[retained_local_indices], dtype=complex),
        sample_indices=_read_only(retained_absolute_indices, dtype=np.int64),
        output_sample_rate_hz=sample_rate_hz / config.decimation,
        reference_frequency_hz=float(reference_frequency_hz),
        reference_phase_rad=float(reference_phase_rad),
    )


def demodulate_scan_blocks(
    record: RawWaveformRecord,
    channel_references: Sequence[ChannelReference],
) -> DemodulatedScan:
    """Apply a rectangular software lock-in and decimate each block to one value."""
    references = tuple(channel_references)
    if not references:
        raise ValueError("at least one channel reference is required")

    block_count, samples_per_block = record.raw_adc.shape
    local_indices = np.arange(samples_per_block, dtype=np.int64)
    demodulated = np.empty((block_count, len(references)), dtype=complex)

    for channel_index, channel in enumerate(references):
        frequency_hz = reference_frequency_hz(
            channel,
            record.measured_tone_frequencies_hz,
        )
        phase_rad = reference_phase_rad(channel, record.tone_phases_rad)
        if not 0.0 < abs(frequency_hz) < record.sample_rate_hz / 2.0:
            raise ValueError(
                f"channel {channel.name!r} lies outside the ADC Nyquist band"
            )

        for block_index in range(block_count):
            sample_indices = (
                record.block_start_sample_indices[block_index] + local_indices
            )
            phase = (
                2.0
                * np.pi
                * frequency_hz
                * sample_indices
                / record.sample_rate_hz
                + phase_rad
            )
            demodulated[block_index, channel_index] = 2.0 * np.mean(
                record.raw_adc[block_index] * np.exp(-1j * phase)
            )

    return DemodulatedScan(
        complex_channels=_read_only(demodulated, dtype=complex),
        channel_references=references,
        calibrated_detuning_hz=record.calibrated_detuning_hz,
    )


def rectangular_window_leakage(
    frequency_offset_hz: float,
    integration_time_seconds: float,
) -> float:
    """Return the ideal amplitude leakage of a rectangular block average."""
    if integration_time_seconds <= 0.0:
        raise ValueError("integration_time_seconds must be positive")
    return float(abs(np.sinc(frequency_offset_hz * integration_time_seconds)))

