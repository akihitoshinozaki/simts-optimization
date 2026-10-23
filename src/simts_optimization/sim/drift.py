"""Deterministic drift states for the C1 mock experiment."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriftState:
    """Known scan-to-scan drift applied by the simulator."""

    line_center_offset: float
    gain_scale: float
    nuisance_amplitude: float


def drift_state_for_scan(
    scan_index: int,
    scan_count: int,
    *,
    line_center_step: float,
    gain_step: float,
    nuisance_scale: float,
) -> DriftState:
    """Return a smooth, reproducible drift state for one scan."""
    if scan_count < 2:
        raise ValueError("scan_count must be at least 2")
    if not 0 <= scan_index < scan_count:
        raise ValueError("scan_index must lie inside the scan sequence")

    centered_index = scan_index - (scan_count - 1) / 2.0
    phase = 2.0 * np.pi * scan_index / scan_count
    return DriftState(
        line_center_offset=float(centered_index * line_center_step),
        gain_scale=float(1.0 + centered_index * gain_step),
        nuisance_amplitude=float(nuisance_scale * np.sin(phase)),
    )

