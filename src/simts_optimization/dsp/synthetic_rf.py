"""Known synthetic RF waveforms for validating the C2 pipeline."""

from collections.abc import Sequence

import numpy as np

from ..io.calibration import DetuningCalibration
from ..io.records import RawWaveformRecord, ingest_raw_waveform
from .phase import ChannelReference, reference_frequency_hz, reference_phase_rad


def generate_synthetic_rf_record(
    *,
    calibrated_detuning_hz: np.ndarray,
    channel_references: Sequence[ChannelReference],
    channel_phasors: np.ndarray,
    sample_rate_hz: float,
    samples_per_block: int,
    tone_frequencies_hz: tuple[float, float],
    tone_phases_rad: tuple[float, float] = (0.0, 0.0),
    calibration: DetuningCalibration,
    adc_noise_standard_deviation: float = 0.0,
    seed: int = 0,
    run_id: str = "synthetic-rf",
    timestamp_utc: str = "2026-01-01T00:00:00Z",
) -> RawWaveformRecord:
    """Generate ADC blocks whose relative complex phasors are known exactly."""
    detuning = np.asarray(calibrated_detuning_hz, dtype=float)
    references = tuple(channel_references)
    phasors = np.asarray(channel_phasors, dtype=complex)

    if detuning.ndim != 1 or detuning.size == 0:
        raise ValueError("calibrated_detuning_hz must be a nonempty vector")
    if not references:
        raise ValueError("at least one channel reference is required")
    if phasors.shape != (detuning.size, len(references)):
        raise ValueError("channel_phasors must have shape (blocks, channels)")
    if sample_rate_hz <= 0.0 or samples_per_block < 2:
        raise ValueError("sample rate and samples_per_block must be positive")
    if adc_noise_standard_deviation < 0.0:
        raise ValueError("ADC noise standard deviation must be nonnegative")

    block_starts = np.arange(detuning.size, dtype=np.int64) * samples_per_block
    local_indices = np.arange(samples_per_block, dtype=np.int64)
    raw_adc = np.zeros((detuning.size, samples_per_block), dtype=float)

    for channel_index, channel in enumerate(references):
        frequency_hz = reference_frequency_hz(channel, tone_frequencies_hz)
        phase_rad = reference_phase_rad(channel, tone_phases_rad)
        if not 0.0 < abs(frequency_hz) < sample_rate_hz / 2.0:
            raise ValueError(
                f"channel {channel.name!r} lies outside the ADC Nyquist band"
            )

        sample_indices = block_starts[:, None] + local_indices[None, :]
        carrier = np.exp(
            1j
            * (
                2.0 * np.pi * frequency_hz * sample_indices / sample_rate_hz
                + phase_rad
            )
        )
        raw_adc += np.real(phasors[:, channel_index, None] * carrier)

    if adc_noise_standard_deviation > 0.0:
        rng = np.random.default_rng(seed)
        raw_adc += rng.normal(
            scale=adc_noise_standard_deviation,
            size=raw_adc.shape,
        )

    raw_actuator_coordinate = (
        detuning / calibration.hertz_per_unit + calibration.resonance_coordinate
    )
    differences = np.diff(detuning)
    if np.all(differences > 0.0):
        scan_direction = "increasing"
    elif np.all(differences < 0.0):
        scan_direction = "decreasing"
    else:
        raise ValueError("calibrated detuning must be strictly monotonic")

    return ingest_raw_waveform(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        raw_adc=raw_adc,
        sample_rate_hz=sample_rate_hz,
        commanded_tone_frequencies_hz=tone_frequencies_hz,
        measured_tone_frequencies_hz=tone_frequencies_hz,
        tone_phases_rad=tone_phases_rad,
        raw_actuator_coordinate=raw_actuator_coordinate,
        calibrated_detuning_hz=detuning,
        calibration_version=calibration.version,
        scan_direction=scan_direction,
        block_start_sample_indices=block_starts,
    )

