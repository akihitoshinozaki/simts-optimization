"""Declared affine detuning calibration and scan registration."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetuningCalibration:
    """Convert a raw actuator coordinate to frequency detuning in hertz."""

    resonance_coordinate: float
    hertz_per_unit: float
    version: str = "affine-v1"

    def __post_init__(self) -> None:
        if self.hertz_per_unit == 0.0:
            raise ValueError("hertz_per_unit must be nonzero")
        if not self.version:
            raise ValueError("calibration version must not be empty")


def calibrate_detuning(
    raw_actuator_coordinate: np.ndarray,
    calibration: DetuningCalibration,
) -> np.ndarray:
    """Return calibrated detuning without modifying the raw coordinate."""
    raw = np.asarray(raw_actuator_coordinate, dtype=float)
    if raw.ndim != 1:
        raise ValueError("raw_actuator_coordinate must be one-dimensional")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_actuator_coordinate must be finite")

    return (raw - calibration.resonance_coordinate) * calibration.hertz_per_unit


def register_complex_channels(
    source_detuning_hz: np.ndarray,
    complex_channels: np.ndarray,
    target_detuning_hz: np.ndarray,
) -> np.ndarray:
    """Interpolate complex channels onto a declared common detuning grid."""
    source = np.asarray(source_detuning_hz, dtype=float)
    target = np.asarray(target_detuning_hz, dtype=float)
    channels = np.asarray(complex_channels, dtype=complex)

    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("source and target detuning must be one-dimensional")
    if channels.ndim != 2 or channels.shape[0] != source.size:
        raise ValueError("complex_channels must have one row per source sample")
    if source.size < 2 or target.size == 0:
        raise ValueError("registration requires nonempty sampled axes")

    differences = np.diff(source)
    if np.all(differences < 0.0):
        source = source[::-1]
        channels = channels[::-1]
    elif not np.all(differences > 0.0):
        raise ValueError("source detuning must be strictly monotonic")
    if not np.all(np.diff(target) > 0.0):
        raise ValueError("target detuning must be strictly increasing")
    if target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError("target detuning must lie within the source range")

    registered = np.empty((target.size, channels.shape[1]), dtype=complex)
    for channel_index in range(channels.shape[1]):
        registered[:, channel_index] = np.interp(
            target,
            source,
            channels[:, channel_index].real,
        ) + 1j * np.interp(
            target,
            source,
            channels[:, channel_index].imag,
        )
    return registered

