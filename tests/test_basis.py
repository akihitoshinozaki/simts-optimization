import numpy as np

from simts_optimization.basis import d_nu, phi, phi_prime, sigma
from simts_optimization.synthetic import detuning, i1, i2, q1, q2


def test_basis_has_one_channel_per_column() -> None:
    assert phi.shape == (detuning.size, 4)
    np.testing.assert_array_equal(phi[:, 0], i1)
    np.testing.assert_array_equal(phi[:, 1], q1)
    np.testing.assert_array_equal(phi[:, 2], i2)
    np.testing.assert_array_equal(phi[:, 3], q2)


def test_frequency_slope_matches_analytic_toy_curve_slopes() -> None:
    x1 = (0.0 - (-0.75)) / 0.80
    x2 = (0.0 - 1.00) / 1.15
    expected = np.array(
        [
            -2.0 * x1 / (0.80 * (1.0 + x1**2) ** 2),
            (1.0 - x1**2) / (0.80 * (1.0 + x1**2) ** 2),
            0.85 * -2.0 * x2 / (1.15 * (1.0 + x2**2) ** 2),
            0.85 * (1.0 - x2**2) / (1.15 * (1.0 + x2**2) ** 2),
        ]
    )

    assert d_nu.shape == (4,)
    np.testing.assert_allclose(d_nu, expected, rtol=5e-4, atol=1e-5)


def test_basis_derivative_matches_basis_shape_and_resonance_slope() -> None:
    assert phi_prime.shape == phi.shape
    resonance_index = int(np.argmin(np.abs(detuning)))
    np.testing.assert_array_equal(phi_prime[resonance_index], d_nu)


def test_noise_covariance_is_symmetric_positive_definite() -> None:
    assert sigma.shape == (4, 4)
    np.testing.assert_array_equal(sigma, sigma.T)
    assert np.all(np.diag(sigma) > 0.0)
    assert np.all(np.linalg.eigvalsh(sigma) > 0.0)
