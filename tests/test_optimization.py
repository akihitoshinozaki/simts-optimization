import numpy as np
import pytest

from simts_optimization.basis import d_nu, phi_prime, sigma
from simts_optimization.optimization import (
    DEFAULT_CAPTURE_RANGE,
    DEFAULT_MINIMUM_SLOPE,
    calculate_blu_weights,
    solve_capture_range_weights,
    solve_minimum_noise_weights,
    w_blu,
    w_capture,
    w_qp,
)
from simts_optimization.synthetic import detuning


def test_blu_weights_have_unit_frequency_gain() -> None:
    assert w_blu.shape == d_nu.shape
    assert np.all(np.isfinite(w_blu))
    np.testing.assert_allclose(w_blu @ d_nu, 1.0, atol=1e-12)


def test_local_only_qp_matches_blu_c0_acceptance() -> None:
    assert w_qp.shape == d_nu.shape
    assert np.all(np.isfinite(w_qp))
    np.testing.assert_allclose(w_qp @ d_nu, 1.0, atol=1e-9)
    np.testing.assert_allclose(w_qp, w_blu, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        w_qp @ sigma @ w_qp,
        w_blu @ sigma @ w_blu,
        rtol=1e-10,
    )


def test_weight_functions_on_identity_covariance() -> None:
    covariance = np.eye(2)
    frequency_slope = np.array([3.0, 4.0])
    expected = frequency_slope / (frequency_slope @ frequency_slope)

    np.testing.assert_allclose(
        calculate_blu_weights(covariance, frequency_slope),
        expected,
    )
    np.testing.assert_allclose(
        solve_minimum_noise_weights(covariance, frequency_slope),
        expected,
        atol=1e-10,
    )


def test_capture_solution_maintains_required_slope() -> None:
    capture_mask = np.abs(detuning) <= DEFAULT_CAPTURE_RANGE
    capture_slopes = phi_prime[capture_mask] @ w_capture

    np.testing.assert_allclose(w_capture @ d_nu, 1.0, atol=1e-9)
    assert np.min(capture_slopes) >= DEFAULT_MINIMUM_SLOPE - 1e-8
    assert w_capture @ sigma @ w_capture >= w_blu @ sigma @ w_blu
    assert not np.allclose(w_capture, w_blu)


def test_capture_solver_rejects_range_beyond_sampled_axis() -> None:
    with pytest.raises(ValueError, match="beyond the detuning axis"):
        solve_capture_range_weights(
            sigma,
            d_nu,
            phi_prime,
            detuning,
            capture_range=6.0,
            minimum_slope=0.1,
        )
