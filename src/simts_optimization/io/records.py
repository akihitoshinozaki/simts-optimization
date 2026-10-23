"""Immutable raw waveform records for the numerical reference pipeline."""

from dataclasses import dataclass

import numpy as np


def _read_only(array: np.ndarray, *, dtype: type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RawWaveformRecord:
    """Raw ADC blocks and the metadata needed to reproduce demodulation."""

    run_id: str
    timestamp_utc: str
    raw_adc: np.ndarray
    sample_rate_hz: float
    block_start_sample_indices: np.ndarray
    commanded_tone_frequencies_hz: tuple[float, float]
    measured_tone_frequencies_hz: tuple[float, float]
    tone_phases_rad: tuple[float, float]
    raw_actuator_coordinate: np.ndarray
    calibrated_detuning_hz: np.ndarray
    calibration_version: str
    scan_direction: str


def ingest_raw_waveform(
    *,
    run_id: str,
    timestamp_utc: str,
    raw_adc: np.ndarray,
    sample_rate_hz: float,
    commanded_tone_frequencies_hz: tuple[float, float],
    measured_tone_frequencies_hz: tuple[float, float],
    tone_phases_rad: tuple[float, float],
    raw_actuator_coordinate: np.ndarray,
    calibrated_detuning_hz: np.ndarray,
    calibration_version: str,
    scan_direction: str,
    block_start_sample_indices: np.ndarray | None = None,
) -> RawWaveformRecord:
    """Validate and copy raw acquisition inputs into an immutable record."""
    adc = np.asarray(raw_adc, dtype=float)
    raw_coordinate = np.asarray(raw_actuator_coordinate, dtype=float)
    detuning = np.asarray(calibrated_detuning_hz, dtype=float)

    if not run_id or not timestamp_utc:
        raise ValueError("run_id and timestamp_utc must not be empty")
    if adc.ndim != 2 or adc.shape[0] == 0 or adc.shape[1] < 2:
        raise ValueError("raw_adc must have shape (blocks, samples_per_block)")
    if raw_coordinate.shape != (adc.shape[0],):
        raise ValueError("raw actuator coordinate must have one value per block")
    if detuning.shape != (adc.shape[0],):
        raise ValueError("calibrated detuning must have one value per block")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not np.all(np.isfinite(adc)):
        raise ValueError("raw_adc must contain only finite samples")
    if any(frequency <= 0.0 for frequency in measured_tone_frequencies_hz):
        raise ValueError("measured tone frequencies must be positive")
    if scan_direction not in {"increasing", "decreasing"}:
        raise ValueError("scan_direction must be 'increasing' or 'decreasing'")
    if not calibration_version:
        raise ValueError("calibration_version must not be empty")

    if block_start_sample_indices is None:
        starts = np.arange(adc.shape[0], dtype=np.int64) * adc.shape[1]
    else:
        starts = np.asarray(block_start_sample_indices, dtype=np.int64)
        if starts.shape != (adc.shape[0],):
            raise ValueError("block starts must have one value per ADC block")
        if np.any(np.diff(starts) <= 0):
            raise ValueError("block starts must be strictly increasing")

    return RawWaveformRecord(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        raw_adc=_read_only(adc, dtype=float),
        sample_rate_hz=float(sample_rate_hz),
        block_start_sample_indices=_read_only(starts, dtype=np.int64),
        commanded_tone_frequencies_hz=tuple(
            float(value) for value in commanded_tone_frequencies_hz
        ),
        measured_tone_frequencies_hz=tuple(
            float(value) for value in measured_tone_frequencies_hz
        ),
        tone_phases_rad=tuple(float(value) for value in tone_phases_rad),
        raw_actuator_coordinate=_read_only(raw_coordinate, dtype=float),
        calibrated_detuning_hz=_read_only(detuning, dtype=float),
        calibration_version=calibration_version,
        scan_direction=scan_direction,
    )

