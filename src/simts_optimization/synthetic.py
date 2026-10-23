"""Synthetic spectroscopy inputs."""

import numpy as np


# Normalized laser-frequency offset from resonance. An odd number of samples
# makes resonance (zero detuning) the exact center point of the array.
detuning = np.linspace(-5.0, 5.0, num=1001, dtype=float)


def lorentzian(
    delta: np.ndarray,
    *,
    gamma: float = 1.0,
    center: float = 0.0,
) -> np.ndarray:
    """Return a unit-height Lorentzian-like curve."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")

    normalized_detuning = (np.asarray(delta, dtype=float) - center) / gamma
    return 1.0 / (1.0 + normalized_detuning**2)


def dispersive(
    delta: np.ndarray,
    *,
    gamma: float = 1.0,
    center: float = 0.0,
) -> np.ndarray:
    """Return a dispersive-like curve with a zero crossing at ``center``."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")

    normalized_detuning = (np.asarray(delta, dtype=float) - center) / gamma
    return normalized_detuning / (1.0 + normalized_detuning**2)


# Two dimensionless synthetic I/Q-like channel pairs. Different resonance
# centers, linewidths, and amplitudes keep the four channels nonidentical.
i1 = lorentzian(detuning, gamma=0.80, center=-0.75)
q1 = dispersive(detuning, gamma=0.80, center=-0.75)
i2 = 0.85 * lorentzian(detuning, gamma=1.15, center=1.00)
q2 = 0.85 * dispersive(detuning, gamma=1.15, center=1.00)
