"""Basis assembly and synthetic channel statistics."""

import numpy as np

from .synthetic import detuning, i1, i2, q1, q2


# Each row is one detuning sample and each column is one channel, ordered as
# [I1, Q1, I2, Q2].
phi = np.column_stack((i1, q1, i2, q2))


# Estimate each channel's frequency slope at the target resonance (Delta = 0)
# with a five-point centered finite difference using nearby detuning samples.
_resonance_index = int(np.argmin(np.abs(detuning)))
_detuning_step = detuning[_resonance_index + 1] - detuning[_resonance_index]
d_nu = (
    phi[_resonance_index - 2]
    - 8.0 * phi[_resonance_index - 1]
    + 8.0 * phi[_resonance_index + 1]
    - phi[_resonance_index + 2]
) / (12.0 * _detuning_step)


# Derivative of every basis channel across the full detuning axis. At the
# target resonance, use the higher-accuracy five-point estimate defined above.
phi_prime = np.gradient(phi, detuning, axis=0, edge_order=2)
phi_prime[_resonance_index] = d_nu


# Manually chosen covariance in dimensionless squared-noise units. Diagonal
# entries are channel variances; off-diagonal entries represent correlations.
sigma = np.array(
    [
        [1.00, 0.20, 0.10, 0.00],
        [0.20, 0.80, 0.05, 0.10],
        [0.10, 0.05, 1.20, 0.25],
        [0.00, 0.10, 0.25, 0.90],
    ],
    dtype=float,
)
