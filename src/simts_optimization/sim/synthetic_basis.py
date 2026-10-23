"""Theta-dependent clean basis functions for the C1 mock experiment."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..synthetic import dispersive, lorentzian


class IMMode(str, Enum):
    """Synthetic intermodulation regimes required by Milestone C1."""

    REDUNDANT = "redundant"
    USEFUL = "useful"
    FAKE = "fake"


@dataclass(frozen=True)
class MockTheta:
    """Low-dimensional physical setting for the mock experiment."""

    resonance_shift: float = 0.0
    linewidth_scale: float = 1.0
    im_strength: float = 0.75
    im_phase: float = 0.20

    def __post_init__(self) -> None:
        if self.linewidth_scale <= 0.0:
            raise ValueError("linewidth_scale must be positive")
        if self.im_strength < 0.0:
            raise ValueError("im_strength must be nonnegative")

    def as_array(self) -> np.ndarray:
        """Return theta in a stable numerical order."""
        return np.array(
            [
                self.resonance_shift,
                self.linewidth_scale,
                self.im_strength,
                self.im_phase,
            ],
            dtype=float,
        )


DEFAULT_THETA = MockTheta()


def build_clean_channels(
    detuning: np.ndarray,
    theta: MockTheta,
    mode: IMMode,
    *,
    line_center_offset: float,
    gain_scale: float,
) -> np.ndarray:
    """Create baseline I/Q and one IM I/Q pair for an active scan."""
    center = theta.resonance_shift + line_center_offset
    linewidth = theta.linewidth_scale

    baseline_i = gain_scale * lorentzian(
        detuning,
        gamma=linewidth,
        center=center,
    )
    baseline_q = gain_scale * dispersive(
        detuning,
        gamma=linewidth,
        center=center,
    )

    redundant_i = 0.65 * baseline_i - 0.30 * baseline_q
    redundant_q = 0.20 * baseline_i + 0.70 * baseline_q

    if mode is IMMode.REDUNDANT:
        im_i = redundant_i
        im_q = redundant_q
    elif mode is IMMode.USEFUL:
        phase_cosine = np.cos(theta.im_phase)
        phase_sine = np.sin(theta.im_phase)
        independent_i = dispersive(
            detuning,
            gamma=0.48 * linewidth,
            center=center + 0.32,
        )
        independent_q = lorentzian(
            detuning,
            gamma=0.62 * linewidth,
            center=center - 0.28,
        )
        im_i = redundant_i + theta.im_strength * (
            phase_cosine * independent_i - phase_sine * independent_q
        )
        im_q = redundant_q + theta.im_strength * (
            phase_sine * independent_i + phase_cosine * independent_q
        )
    elif mode is IMMode.FAKE:
        from .artifacts import electronic_im_spur

        im_i, im_q = electronic_im_spur(detuning, theta).T
    else:
        raise ValueError(f"unsupported IM mode: {mode}")

    return np.column_stack((baseline_i, baseline_q, im_i, im_q))


def build_pump_blocked_channels(
    detuning: np.ndarray,
    theta: MockTheta,
    mode: IMMode,
) -> np.ndarray:
    """Create the expected signal during the pump-blocked atomicity control."""
    blocked = np.zeros((detuning.size, 4), dtype=float)
    if mode is IMMode.FAKE:
        from .artifacts import electronic_im_spur

        blocked[:, 2:] = electronic_im_spur(detuning, theta)
    return blocked

