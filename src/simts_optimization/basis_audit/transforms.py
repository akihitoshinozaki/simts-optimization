"""Basis transformations used by C3 invariance tests."""

from collections.abc import Sequence

import numpy as np


def rotate_iq_pairs(scans: np.ndarray, angles: float | Sequence[float]) -> np.ndarray:
    """Apply coherent orthogonal rotations to adjacent ``(I, Q)`` pairs."""

    values = np.asarray(scans, dtype=float)
    if values.ndim < 2 or values.shape[-1] % 2:
        raise ValueError("the final axis must contain complete adjacent I/Q pairs")
    pair_count = values.shape[-1] // 2
    if np.isscalar(angles):
        pair_angles = np.full(pair_count, float(angles))
    else:
        pair_angles = np.asarray(tuple(angles), dtype=float)
        if pair_angles.shape != (pair_count,):
            raise ValueError("provide one rotation angle per I/Q pair")

    rotated = values.copy()
    for pair_index, angle in enumerate(pair_angles):
        first = 2 * pair_index
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.array([[cosine, sine], [-sine, cosine]])
        rotated[..., first : first + 2] = values[..., first : first + 2] @ rotation
    return rotated
