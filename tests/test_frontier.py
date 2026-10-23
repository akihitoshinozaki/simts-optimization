import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from simts_optimization.basis import d_nu, phi_prime, sigma
from simts_optimization.frontier import (
    DEFAULT_CAPTURE_RANGES,
    CaptureRangeResult,
    build_pareto_frontier,
    capture_range_sweep,
    pareto_front,
    plot_pareto_frontier,
)
from simts_optimization.optimization import DEFAULT_MINIMUM_SLOPE
from simts_optimization.synthetic import detuning


def test_capture_range_sweep_records_feasible_and_infeasible_results() -> None:
    assert len(capture_range_sweep) == len(DEFAULT_CAPTURE_RANGES)
    assert capture_range_sweep[0].feasible
    assert not capture_range_sweep[-1].feasible
    assert capture_range_sweep[-1].status == "infeasible"

    feasible_noise = []
    for result in capture_range_sweep:
        if not result.feasible:
            continue

        assert result.weights is not None
        assert result.equivalent_frequency_noise is not None
        mask = np.abs(detuning) <= result.capture_range
        np.testing.assert_allclose(result.weights @ d_nu, 1.0, atol=1e-8)
        assert np.min(phi_prime[mask] @ result.weights) >= (
            DEFAULT_MINIMUM_SLOPE - 1e-7
        )
        np.testing.assert_allclose(
            result.equivalent_frequency_noise,
            np.sqrt(result.weights @ sigma @ result.weights),
        )
        feasible_noise.append(result.equivalent_frequency_noise)

    assert np.all(np.diff(feasible_noise) >= -1e-8)


def test_pareto_front_contains_only_nondominated_feasible_points() -> None:
    assert pareto_front
    assert all(point.feasible for point in pareto_front)
    assert all(
        left.capture_range < right.capture_range
        for left, right in zip(pareto_front, pareto_front[1:])
    )

    for candidate in pareto_front:
        assert candidate.equivalent_frequency_noise is not None
        assert not any(
            other.capture_range >= candidate.capture_range
            and other.equivalent_frequency_noise is not None
            and other.equivalent_frequency_noise
            < candidate.equivalent_frequency_noise - 1e-9
            for other in pareto_front
            if other is not candidate
        )


def test_pareto_builder_removes_equal_noise_smaller_range() -> None:
    points = (
        CaptureRangeResult(0.5, 0.4, np.ones(1), 2.0, "optimal"),
        CaptureRangeResult(1.0, 0.4, np.ones(1), 2.0, "optimal"),
        CaptureRangeResult(1.5, 0.4, np.ones(1), 3.0, "optimal"),
        CaptureRangeResult(2.0, 0.4, None, None, "infeasible"),
    )

    frontier = build_pareto_frontier(points)
    assert [point.capture_range for point in frontier] == [1.0, 1.5]


def test_pareto_plot_declares_normalized_units() -> None:
    figure, axes = plot_pareto_frontier(pareto_front)
    try:
        assert "normalized detuning" in axes.get_xlabel()
        assert "normalized detuning" in axes.get_ylabel()
        assert len(axes.lines[0].get_xdata()) == len(pareto_front)
    finally:
        plt.close(figure)
