"""C4 acceptance: datasets, held-out isolation, reproducibility and solver parity."""

from dataclasses import replace
import json
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from simts_optimization import (
    ChannelDataset, ParetoSettings, QPSettings, solve_capture_range_qp,
    solve_inner_pareto, validate_candidate,
)
from simts_optimization.basis import d_nu, phi, phi_prime, sigma
from simts_optimization.synthetic import detuning


def dataset(dataset_id="development", *, seed=0, channels=2, curved=True):
    """Known polynomial response mixed into arbitrary channel coordinates."""
    axis = np.linspace(-1.5, 1.5, 61)
    basis = np.tile(axis[:, None], (1, channels))
    derivative = np.ones_like(basis)
    if curved:
        basis[:, 0] -= 2 * axis**3 / 3
        derivative[:, 0] -= 2 * axis**2
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(channels, channels)))
    covariance = rotation.T @ np.diag(np.linspace(1, 9, channels)) @ rotation
    return ChannelDataset(dataset_id, tuple(f"ch{i}" for i in range(channels)), axis,
                          basis @ rotation, derivative @ rotation,
                          np.ones(channels) @ rotation, covariance)


def policy(**kwargs):
    return ParetoSettings(capture_ranges=(0.0, 0.5, 1.0), minimum_slope=0.2, **kwargs)


@pytest.mark.parametrize("seed,channels", [(0, 1), (1, 2), (2, 4), (3, 8)])
def test_multiple_synthetic_datasets_are_reproducible_and_match_other_solver(seed, channels):
    development = dataset(seed=seed, channels=channels)
    held_out = replace(development, dataset_id="held-out")
    first = solve_inner_pareto(development, (held_out,), policy())
    second = solve_inner_pareto(development, (held_out,), policy())
    reference = solve_inner_pareto(development, (held_out,), policy(qp=QPSettings(solver="CLARABEL")))
    assert first.to_dict() == second.to_dict()
    assert first.summary.normalized_hypervolume > 0
    for actual, other in zip(first.candidates, reference.candidates):
        assert actual.solve.feasible == other.solve.feasible
        if actual.solve.feasible:
            np.testing.assert_allclose(actual.solve.weights, other.solve.weights, atol=3e-6, rtol=3e-5)
            np.testing.assert_allclose(actual.solve.noise_variance, other.solve.noise_variance, rtol=1e-7)
            assert actual.solve.gain_error < 1e-7
            assert actual.solve.slope_violation < 1e-7
            assert actual.validated


def test_c0_dataset_matches_clarabel_and_records_infeasibility():
    development = ChannelDataset("c0", ("I1", "Q1", "I2", "Q2"), detuning, phi, phi_prime, d_nu, sigma)
    held_out = replace(development, dataset_id="c0-held-out")
    settings = ParetoSettings()
    run = solve_inner_pareto(development, (held_out,), settings)
    other = solve_inner_pareto(development, (held_out,), replace(settings, qp=QPSettings(solver="CLARABEL")))
    assert run.candidates[-1].solve.status == "infeasible"
    assert run.status == "partial_success"
    assert run.frontier
    for actual, reference in zip(run.candidates, other.candidates):
        assert actual.solve.feasible == reference.solve.feasible
        if actual.solve.feasible:
            np.testing.assert_allclose(actual.solve.weights, reference.solve.weights, atol=2e-5, rtol=2e-5)


def test_held_out_changes_do_not_change_qp_weights_and_metrics_use_held_out_covariance():
    development = dataset()
    held_out = replace(development, dataset_id="held-out", covariance=4 * development.covariance,
                       channels=development.channels + 0.1)
    run = solve_inner_pareto(development, (held_out,), policy())
    clean = solve_inner_pareto(development, (replace(development, dataset_id="clean"),), policy())
    for point, original in zip(run.candidates, clean.candidates):
        np.testing.assert_array_equal(point.solve.weights, original.solve.weights)
        metric = point.validation[0]
        np.testing.assert_allclose(metric.equivalent_frequency_noise, 2 * point.training_noise)
        np.testing.assert_allclose(metric.lock_offset, 0.1 * np.sum(point.solve.weights))
        assert metric.response_rmse > 0


def test_gain_drift_fails_validation_without_refitting():
    development = dataset()
    held_out = replace(development, dataset_id="drift", frequency_slope=0.5 * development.frequency_slope)
    run = solve_inner_pareto(development, (held_out,), policy())
    assert run.status == "no_validated_candidates"
    assert not run.frontier
    assert run.training_frontier
    assert run.summary.normalized_hypervolume == 0
    assert run.summary.best_validated_noise is None
    assert run.summary.maximum_validated_capture is None
    for candidate in run.candidates:
        assert candidate.solve.feasible
        assert candidate.validation[0].status == "constraint_violation"
        np.testing.assert_allclose(candidate.validation[0].gain, 0.5)
        np.testing.assert_allclose(candidate.validation[0].equivalent_frequency_noise, 2 * candidate.training_noise)


def test_validate_before_pruning_training_dominated_candidates():
    development = dataset(channels=1, curved=False)
    held_out = dataset("curved-held-out", channels=1, curved=True)
    run = solve_inner_pareto(development, (held_out,), policy())
    assert [c.capture_range for c in run.training_frontier] == [1.0]
    assert [c.capture_range for c in run.frontier] == [0.5]
    assert run.candidates[-1].validation[0].status == "constraint_violation"
    # Single valid point: rectangle of width .5, height 10 - 1.
    assert run.summary.normalized_hypervolume == pytest.approx(0.5 * 9 / 20)


def test_all_held_out_datasets_must_pass_and_worst_noise_is_used():
    development = dataset()
    first = replace(development, dataset_id="first")
    second = replace(development, dataset_id="second", covariance=9 * development.covariance)
    run = solve_inner_pareto(development, (first, second), policy())
    for candidate in run.candidates:
        assert len(candidate.validation) == 2
        assert candidate.worst_validation_noise == pytest.approx(3 * candidate.training_noise)
    rejected = solve_inner_pareto(development, (first, replace(second, frequency_slope=2 * second.frequency_slope)), policy())
    assert not rejected.frontier


def test_unsupported_range_is_recorded_and_held_out_can_have_a_different_grid():
    development = dataset()
    held_out = replace(development, dataset_id="short-grid", detuning_axis=development.detuning_axis[15:-15],
                       channels=development.channels[15:-15], basis_derivative=development.basis_derivative[15:-15])
    run = solve_inner_pareto(development, (held_out,), ParetoSettings(capture_ranges=(0, 0.25, 1, 2), minimum_slope=0.2))
    assert run.candidates[0].validated
    assert run.candidates[1].validated
    assert run.candidates[2].validation[0].status == "out_of_domain"
    assert run.candidates[3].solve.status == "out_of_domain"


def test_no_feasible_solution_has_finite_empty_scores_and_no_weights():
    development = dataset()
    run = solve_inner_pareto(development, (replace(development, dataset_id="test"),),
                             ParetoSettings(capture_ranges=(0, 0.5), minimum_slope=1.01))
    assert run.status == "no_feasible_solution"
    assert not run.frontier
    assert run.summary.feasible_count == 0
    assert run.summary.normalized_hypervolume == 0
    assert all(c.solve.weights is None and c.solve.status == "infeasible" for c in run.candidates)
    json.dumps(run.to_dict(), allow_nan=False)


def test_solver_exception_is_recorded_and_sweep_continues(monkeypatch):
    original = cp.Problem.solve
    calls = 0

    def fail_once(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise cp.error.SolverError("injected backend failure")
        return original(self, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_once)
    development = dataset()
    run = solve_inner_pareto(development, (replace(development, dataset_id="test"),), policy())
    assert run.candidates[0].solve.status == "solver_error"
    assert "injected" in run.candidates[0].solve.message
    assert run.candidates[1].validated
    assert run.status == "partial_success"


@pytest.mark.parametrize("status,weights,accept,expected", [
    (cp.OPTIMAL, [0.0, 0.0], False, "invalid_solution"),
    (cp.OPTIMAL_INACCURATE, [0.5, 0.5], False, "optimal_inaccurate"),
    (cp.OPTIMAL_INACCURATE, [0.5, 0.5], True, "optimal_inaccurate"),
    (cp.OPTIMAL_INACCURATE, [0.0, 0.0], True, "invalid_solution"),
    (cp.USER_LIMIT, [0.5, 0.5], False, "user_limit"),
])
def test_solver_status_and_numerical_acceptance(monkeypatch, status, weights, accept, expected):
    def fake_solve(self, **kwargs):
        self._status = status
        self._solver_stats = SimpleNamespace(num_iters=2)
        self.variables()[0].value = weights

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    result = solve_capture_range_qp(np.eye(2), np.ones(2), np.ones((3, 2)), np.array([-1, 0, 1]),
                                    capture_range=0.5, minimum_slope=0.2,
                                    settings=QPSettings(accept_inaccurate=accept))
    assert result.status == expected
    assert result.solver_status == status
    assert result.feasible == (accept and weights == [0.5, 0.5])


def test_constraints_include_interpolated_endpoints_and_zero_without_grid_sample():
    # At +/- .5 the interpolated derivative is .5, though interior samples are 1.
    axis = np.array([-1, -0.2, 0.2, 1])
    derivative = np.array([[-1 / 3], [1], [1], [-1 / 3]])
    zero = solve_capture_range_qp(np.eye(1), np.ones(1), derivative, axis, capture_range=0, minimum_slope=0.9)
    endpoint = solve_capture_range_qp(np.eye(1), np.ones(1), derivative, axis, capture_range=0.5, minimum_slope=0.6)
    assert zero.feasible
    assert endpoint.status == "infeasible"


def test_settings_and_complete_report_round_trip(tmp_path):
    settings = policy(qp=QPSettings(solver="CLARABEL", max_iterations=500))
    path = tmp_path / "settings.json"
    settings.save(path)
    assert ParetoSettings.load(path) == settings
    development = dataset()
    run = solve_inner_pareto(development, (replace(development, dataset_id="test"),), settings)
    report = tmp_path / "run.json"
    run.save(report)
    payload = json.loads(report.read_text())
    assert payload["settings"] == settings.to_dict()
    assert payload["development_id"] == "development"
    assert payload["held_out_ids"] == ["test"]
    assert payload["software_versions"]["clarabel"]
    assert payload["candidates"][0]["solve"]["weights"]
    with pytest.raises(ValueError, match="version"):
        ParetoSettings.from_dict({**settings.to_dict(), "version": "future"})
    with pytest.raises(ValueError, match="unknown"):
        ParetoSettings.from_dict({**settings.to_dict(), "typo": True})


@pytest.mark.parametrize("changes", [
    {"capture_ranges": ()}, {"capture_ranges": (0, np.nan)}, {"capture_ranges": (1, 0)},
    {"capture_ranges": (-1, 0)}, {"capture_ranges": (0, 0)}, {"minimum_slope": np.inf},
    {"validation_gain_tolerance": 1}, {"validation_slope_tolerance": -1},
    {"noise_reference": 0}, {"capture_reference": np.nan}, {"noise_tolerance": -1},
])
def test_settings_reject_invalid_conditions(changes):
    with pytest.raises(ValueError):
        ParetoSettings(**changes)


@pytest.mark.parametrize("changes", [
    {"solver": "missing"}, {"absolute_tolerance": np.nan}, {"relative_tolerance": 0},
    {"feasibility_tolerance": -1}, {"max_iterations": 2.5}, {"accept_inaccurate": "yes"},
])
def test_qp_settings_reject_invalid_conditions(changes):
    with pytest.raises(ValueError):
        QPSettings(**changes)


@pytest.mark.parametrize("field,value", [
    ("channels", np.zeros((2, 2))), ("frequency_slope", [np.nan, 1]),
    ("frequency_slope", [0, 0]), ("covariance", [[1, 2], [0, 1]]),
    ("covariance", [[1, 1], [1, 1]]), ("covariance", [[np.inf, 0], [0, 1]]),
    ("basis_derivative", np.ones((61, 2)) * np.nan),
    ("detuning_axis", np.zeros(61)), ("detuning_axis", np.arange(61) + 1),
    ("channel_names", ("same", "same")), ("channel_names", ("one",)),
    ("dataset_id", ""), ("channels", np.ones((61, 2), dtype=complex)),
])
def test_dataset_checks(field, value):
    with pytest.raises(ValueError):
        replace(dataset(), **{field: value})


def test_dataset_copies_inputs_and_rejects_split_or_channel_mismatches():
    development = dataset()
    source = development.channels.copy()
    copied = replace(development, channels=source)
    source[:] = 100
    np.testing.assert_array_equal(copied.channels, development.channels)
    assert not copied.channels.flags.writeable
    with pytest.raises(ValueError, match="held-out"):
        solve_inner_pareto(development, (), policy())
    with pytest.raises(ValueError, match="distinct"):
        solve_inner_pareto(development, (development,), policy())
    with pytest.raises(ValueError, match="order"):
        solve_inner_pareto(development, (replace(development, dataset_id="test", channel_names=("ch1", "ch0")),), policy())


def test_zero_held_out_gain_is_a_validation_failure_not_infinite_noise():
    development = dataset()
    weights = np.array([1, 0])
    held_out = replace(development, dataset_id="zero-gain", frequency_slope=np.array([0, 1]))
    metric = validate_candidate(weights, development, held_out, capture_range=0, settings=policy())
    assert not metric.passed
    assert metric.equivalent_frequency_noise is None


def test_import_and_failure_reporting_do_not_require_a_working_solver():
    import subprocess
    import sys

    code = """
import cvxpy as cp

def broken(*args, **kwargs):
    raise cp.error.SolverError('backend unavailable')

cp.Problem.solve = broken
from simts_optimization import ChannelDataset, ParetoSettings, solve_inner_pareto
from dataclasses import replace
import numpy as np
x = np.array([-1., 0., 1.])
data = ChannelDataset('dev', ('ch',), x, x[:, None], np.ones((3, 1)), np.ones(1), np.eye(1))
run = solve_inner_pareto(data, (replace(data, dataset_id='test'),), ParetoSettings(capture_ranges=(0, .5)))
assert run.status == 'no_usable_solution'
assert all(c.solve.status == 'solver_error' for c in run.candidates)
assert run.summary.normalized_hypervolume == 0
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_hypervolume_rectangle_union_and_reference_clipping():
    development = dataset(channels=2)
    settings = policy(capture_reference=0.75, noise_reference=2.0)
    run = solve_inner_pareto(development, (replace(development, dataset_id="test"),), settings)
    # Verify the rectangle union independently by vertical slices of all valid
    # candidates, including those excluded by Pareto pruning.
    points = [(min(c.capture_range, settings.capture_reference), c.worst_validation_noise)
              for c in run.candidates if c.validated]
    bounds = sorted({0.0, *(x for x, _ in points)})
    expected = 0
    for left, right in zip(bounds, bounds[1:]):
        best_noise = min(n for x, n in points if x >= right)
        expected += (right - left) * max(0, settings.noise_reference - best_noise)
    assert run.summary.normalized_hypervolume == pytest.approx(expected / (0.75 * 2.0))


def test_known_active_constraint_solution_and_correlated_ill_conditioned_covariance():
    axis = np.linspace(-1, 1, 21)
    derivative = np.column_stack((1 - 2 * axis**2, np.ones_like(axis)))
    active = solve_capture_range_qp(np.diag([1., 9.]), np.ones(2), derivative, axis,
                                   capture_range=1, minimum_slope=0.2)
    np.testing.assert_allclose(active.weights, [0.4, 0.6], atol=1e-8)
    covariance = np.array([[1., 0.9999], [0.9999, 1.]])
    from simts_optimization import calculate_blu_weights
    expected = calculate_blu_weights(covariance, np.array([1., 2.]))
    local = solve_capture_range_qp(covariance, np.array([1., 2.]), np.tile([1., 2.], (21, 1)), axis,
                                  capture_range=0.5, minimum_slope=0.2)
    np.testing.assert_allclose(local.weights, expected, atol=1e-7)
