"""C5 acceptance: outer BO over theta, noise/drift/failure handling, baselines."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from simts_optimization.bo import (
    C5_DESIGN_SPACE,
    C5_REFERENCE_SETTINGS,
    CampaignConfig,
    EvaluationRecord,
    SimulatedExperiment,
    estimate_drift,
    estimate_noise,
    run_benchmark,
    run_campaign,
)
from simts_optimization.bo.benchmark import good_region_volume_fraction, summarize_campaign
from simts_optimization.bo.loop import _training_data
from simts_optimization.sim.landscape import HiddenLandscape

GOOD = {"modulation_frequency": 6.2, "modulation_depth": 1.84, "pump_power": 1.3}
POOR = {"modulation_frequency": 2.0, "modulation_depth": 0.4, "pump_power": 0.3}
QUIET = HiddenLandscape(dropout_probability=0.0, drift_amplitude=0.0)


def record(index, kind, score, settings=None, status="ok"):
    return EvaluationRecord(index, kind, dict(settings or C5_REFERENCE_SETTINGS), index, status,
                            score, None if score is None else 0.01, status == "ok")


def test_design_space_bounds_round_trip_and_sobol_are_explicit():
    unit = C5_DESIGN_SPACE.to_unit(GOOD)
    assert np.all((0 <= unit) & (unit <= 1))
    assert C5_DESIGN_SPACE.from_unit(unit) == pytest.approx(GOOD)
    np.testing.assert_array_equal(C5_DESIGN_SPACE.sobol(3, 8), C5_DESIGN_SPACE.sobol(3, 8))
    with pytest.raises(ValueError):
        C5_DESIGN_SPACE.validate({**GOOD, "pump_power": 9.0})
    with pytest.raises(ValueError):
        C5_DESIGN_SPACE.validate({"pump_power": 1.0})


def test_hidden_good_region_is_small_and_excludes_the_reference():
    landscape = HiddenLandscape()
    assert landscape.quality(GOOD) == pytest.approx(1.0, abs=1e-3)
    assert landscape.in_good_region(GOOD)
    assert not landscape.in_good_region(POOR)
    assert not landscape.in_good_region(C5_REFERENCE_SETTINGS)
    fraction = good_region_volume_fraction(landscape, SimulatedExperiment(landscape), 50_000)
    assert 0.005 < fraction < 0.03


def test_optimizer_code_never_reads_the_hidden_region():
    package = Path(__file__).parents[1] / "src/simts_optimization/bo"
    for name in ("loop.py", "space.py"):
        source = (package / name).read_text()
        for secret in ("landscape", "in_good_region", "quality(", "resonance_frequency"):
            assert secret not in source, (name, secret)


def test_pipeline_scores_good_region_above_poor_settings_with_uncertainty():
    experiment = SimulatedExperiment(QUIET)
    good = [experiment.evaluate(GOOD, kind="test", seed=s) for s in range(3)]
    poor = [experiment.evaluate(POOR, kind="test", seed=s) for s in range(3)]
    for r in good:
        assert r.status == "ok" and r.passed_constraints
        assert r.score_uncertainty is not None and np.isfinite(r.score_uncertainty)
        assert len(r.folds) == 2 and r.atomicity_ratio is not None
    assert min(r.score for r in good) > max(r.score for r in poor) + 0.1
    assert [r.index for r in good + poor] == list(range(6))  # every measurement ticks the clock


def test_evaluation_is_reproducible_for_a_fixed_seed_and_clock():
    first = SimulatedExperiment(QUIET).evaluate(GOOD, kind="test", seed=11)
    second = SimulatedExperiment(QUIET).evaluate(GOOD, kind="test", seed=11)
    assert first.score == second.score and first.folds == second.folds


def test_failures_are_recorded_explicitly_not_raised():
    unlocked = SimulatedExperiment(QUIET).evaluate(
        {"modulation_frequency": 5.0, "modulation_depth": 2.9, "pump_power": 3.8}, kind="test", seed=0)
    assert unlocked.failed and unlocked.failure_reason == "laser_unlock" and unlocked.score is None

    dropout = SimulatedExperiment(replace(QUIET, dropout_probability=1.0)).evaluate(
        GOOD, kind="test", seed=0)
    assert dropout.failed and dropout.failure_reason == "acquisition_dropout"

    def broken_simulator(*args):
        raise ValueError("covariance must be positive definite")

    broken = SimulatedExperiment(QUIET, simulator=broken_simulator).evaluate(GOOD, kind="test", seed=0)
    assert broken.failed and broken.failure_reason == "pipeline_error"
    assert "positive definite" in broken.message

    with pytest.raises(ValueError):
        SimulatedExperiment(QUIET).evaluate({**GOOD, "modulation_depth": -1.0}, kind="test", seed=0)


def test_low_snr_settings_report_constraint_violations():
    dim = {"modulation_frequency": 1.0, "modulation_depth": 0.1, "pump_power": 0.1}
    result = SimulatedExperiment(QUIET).evaluate(dim, kind="test", seed=0)
    assert result.status == "constraint_violation" and not result.passed_constraints
    assert result.constraint_reasons and result.score is not None


def test_noise_is_pooled_across_repeated_settings():
    other = {**C5_REFERENCE_SETTINGS, "pump_power": 2.0}
    records = [record(0, "x", 1.0), record(1, "x", 3.0), record(2, "x", 5.0, other),
               record(3, "x", 7.0, other), record(4, "x", None, other, status="failed")]
    estimate = estimate_noise(records, np.array([1.0, 3.0, 5.0, 7.0, np.nan]))
    assert estimate.degrees_of_freedom == 2 and estimate.groups == 2
    assert estimate.pooled_std == pytest.approx(np.sqrt(2.0))


def test_drift_is_tracked_from_references_and_shrunk_when_indistinguishable_from_noise():
    clocks = np.arange(0, 61, 6)
    true_offset = 0.1 * np.sin(2 * np.pi * clocks / 60)
    rng = np.random.default_rng(0)
    records = [record(int(t), "reference", 0.3 + d + 0.005 * rng.normal())
               for t, d in zip(clocks, true_offset)]
    estimate = estimate_drift(records, noise_std=0.005)
    np.testing.assert_allclose(estimate - estimate.mean(), true_offset - true_offset.mean(), atol=0.02)

    flat = [record(int(t), "reference", 0.3 + 0.01 * rng.normal()) for t in clocks]
    assert not np.any(estimate_drift(flat, noise_std=0.02))
    assert not np.any(estimate_drift(records, noise_std=None))


def test_failed_runs_are_imputed_with_the_worst_observed_score():
    records = [record(0, "x", 0.4), record(1, "x", 0.1), record(2, "x", None, status="failed")]
    x, y = _training_data(records, C5_DESIGN_SPACE, np.array([0.4, 0.1, np.nan]))
    assert x.shape == (3, 3)
    np.testing.assert_array_equal(y, [0.4, 0.1, 0.1])


def test_bo_campaign_counts_repeats_and_references_in_the_budget(tmp_path):
    config = CampaignConfig(budget=16, initial_points=5, reference_interval=6, repeat_interval=6,
                            num_restarts=2, raw_samples=64)
    result = run_campaign("bo", 1, config)
    kinds = [r.kind for r in result.records]
    assert len(result.records) == config.budget
    assert kinds[:6] == ["reference"] + ["initial"] * 5
    assert kinds.index("repeat") > kinds.index("initial") + 4  # after the initial design
    assert kinds.count("reference") >= 2 and kinds.count("repeat") >= 2
    assert {"bo", "fallback"} & set(kinds)
    assert [r.index for r in result.records] == list(range(config.budget))
    assert result.recommendation is not None
    result.save(tmp_path / "campaign.json")
    saved = json.loads((tmp_path / "campaign.json").read_text())
    assert len(saved["records"]) == config.budget and "passed_constraints" in saved["records"][0]
    assert summarize_campaign(result, HiddenLandscape()).repeat_count == kinds.count("repeat")


@pytest.mark.parametrize("strategy", ["random", "sobol"])
def test_baselines_spend_the_same_budget_on_new_settings(strategy):
    result = run_campaign(strategy, 2, CampaignConfig(budget=10, reference_interval=0,
                                                      repeat_interval=0))
    assert [r.kind for r in result.records] == [strategy] * 10


def test_benchmark_rejects_unequal_budgets():
    with pytest.raises(ValueError):
        run_benchmark(seeds=(0,), budget=10, bo_config=CampaignConfig(budget=12))


@pytest.mark.slow
def test_c5_bo_finds_hidden_region_sooner_than_baselines_across_seeds():
    result = run_benchmark(seeds=tuple(range(10)), budget=60)
    bo, random, sobol = (result.summaries[s] for s in ("bo", "random", "sobol"))
    assert result.complete, result.summaries
    assert bo.median_evaluations_to_discovery < min(random.median_evaluations_to_discovery,
                                                    sobol.median_evaluations_to_discovery)
    assert bo.discovery_rate >= max(random.discovery_rate, sobol.discovery_rate)
