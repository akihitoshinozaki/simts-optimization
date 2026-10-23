"""Nuisance and electronic-artifact models for synthetic scans."""

import numpy as np

from ..synthetic import dispersive, lorentzian
from .synthetic_basis import MockTheta


def nuisance_perturbation(
    detuning: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Return a RAM-like perturbation for all four mock channels."""
    broad_even = lorentzian(detuning, gamma=3.0, center=0.0)
    broad_odd = dispersive(detuning, gamma=3.0, center=0.0)
    return amplitude * np.column_stack(
        (
            0.60 * broad_even,
            0.25 * broad_odd,
            0.45 * broad_even,
            -0.30 * broad_odd,
        )
    )


def electronic_im_spur(
    detuning: np.ndarray,
    theta: MockTheta,
) -> np.ndarray:
    """Return a large IM spur unrelated to the atomic resonance parameters."""
    phase = theta.im_phase
    amplitude = 1.4 * theta.im_strength
    return amplitude * np.column_stack(
        (
            np.cos(1.35 * detuning + phase),
            np.sin(1.35 * detuning + phase),
        )
    )

