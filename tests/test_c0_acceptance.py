"""Milestone C0 inner-optimizer acceptance tests."""

import numpy as np

from simts_optimization.basis import d_nu, phi_prime
from simts_optimization.frontier import (
    capture_range_sweep,
    pareto_front,
    sweep_capture_ranges,
)
from simts_optimization.optimization import w_blu, w_qp
from simts_optimization.synthetic import detuning


def test_qp_weights_match_closed_form_blu_weights() -> None:
    np.testing.assert_allclose(w_qp, w_blu, rtol=1e-8, atol=1e-10)


def test_every_feasible_sweep_solution_has_unit_frequency_gain() -> None:
    feasible_results = [result for result in capture_range_sweep if result.feasible]
    assert feasible_results

    for result in feasible_results:
        assert result.weights is not None
        np.testing.assert_allclose(result.weights @ d_nu, 1.0, atol=1e-8)


def test_every_feasible_sweep_solution_satisfies_capture_constraint() -> None:
    feasible_results = [result for result in capture_range_sweep if result.feasible]
    assert feasible_results

    for result in feasible_results:
        assert result.weights is not None
        capture_mask = np.abs(detuning) <= result.capture_range
        achieved_slopes = phi_prime[capture_mask] @ result.weights
        assert np.min(achieved_slopes) >= result.minimum_slope - 1e-7


def test_impossible_capture_constraint_returns_infeasible_without_weights() -> None:
    # At Delta = 0, phi_prime @ w equals d_nu @ w, which is fixed to 1.
    # Requiring it to be at least 1.01 is therefore mathematically impossible.
    result = sweep_capture_ranges([0.0], minimum_slope=1.01)[0]

    assert not result.feasible
    assert result.status == "infeasible"
    assert result.weights is None
    assert result.equivalent_frequency_noise is None


def test_final_pareto_points_are_nondominated() -> None:
    feasible_results = [result for result in capture_range_sweep if result.feasible]
    assert pareto_front

    for candidate in pareto_front:
        assert candidate.equivalent_frequency_noise is not None

        for other in feasible_results:
            if other is candidate:
                continue
            assert other.equivalent_frequency_noise is not None

            range_is_no_worse = other.capture_range >= candidate.capture_range
            noise_is_no_worse = (
                other.equivalent_frequency_noise
                <= candidate.equivalent_frequency_noise + 1e-9
            )
            strictly_better = (
                other.capture_range > candidate.capture_range
                or other.equivalent_frequency_noise
                < candidate.equivalent_frequency_noise - 1e-9
            )
            assert not (range_is_no_worse and noise_is_no_worse and strictly_better)

