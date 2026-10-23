"""External atomicity gate for synthetic IM channels."""

from dataclasses import dataclass

import numpy as np

from .sim.mock_experiment import MockExperiment


@dataclass(frozen=True)
class AtomicityResult:
    """Result of the pump-blocked IM persistence control."""

    valid: bool
    pump_blocked_ratio: float
    reason_codes: tuple[str, ...]


def check_atomicity(
    experiment: MockExperiment,
    *,
    maximum_pump_blocked_ratio: float = 0.20,
) -> AtomicityResult:
    """Reject IM responses that persist when the atomic pump is blocked."""
    if not 0.0 <= maximum_pump_blocked_ratio < 1.0:
        raise ValueError("maximum_pump_blocked_ratio must lie in [0, 1)")

    lock_index = int(np.argmin(np.abs(experiment.detuning)))
    active_im = np.mean(
        [
            scan.channels[:, 2:] - scan.channels[lock_index, 2:]
            for scan in experiment.scans
        ],
        axis=0,
    )
    blocked_im = np.mean(
        [
            scan.pump_blocked_channels[:, 2:]
            - scan.pump_blocked_channels[lock_index, 2:]
            for scan in experiment.scans
        ],
        axis=0,
    )

    active_norm = float(np.linalg.norm(active_im))
    if active_norm == 0.0:
        return AtomicityResult(
            valid=False,
            pump_blocked_ratio=float("inf"),
            reason_codes=("no_active_im_response",),
        )

    pump_blocked_ratio = float(np.linalg.norm(blocked_im) / active_norm)
    if pump_blocked_ratio > maximum_pump_blocked_ratio:
        return AtomicityResult(
            valid=False,
            pump_blocked_ratio=pump_blocked_ratio,
            reason_codes=("pump_blocked_im_persists",),
        )

    return AtomicityResult(
        valid=True,
        pump_blocked_ratio=pump_blocked_ratio,
        reason_codes=(),
    )

