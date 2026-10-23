"""Milestone C1 synthetic-experiment acceptance tests."""

import pytest

from simts_optimization.c1 import evaluate_c1_regime
from simts_optimization.sim import IMMode, simulate_mock_experiment


@pytest.mark.parametrize(
    (
        "mode",
        "expected_basis_expansion",
        "expected_atomicity",
        "expected_acceptance",
        "expected_frontier_expansion",
    ),
    [
        (IMMode.REDUNDANT, False, True, False, False),
        (IMMode.USEFUL, True, True, True, True),
        (IMMode.FAKE, True, False, False, False),
    ],
)
def test_required_c1_regimes_trigger_expected_outcomes(
    mode: IMMode,
    expected_basis_expansion: bool,
    expected_atomicity: bool,
    expected_acceptance: bool,
    expected_frontier_expansion: bool,
) -> None:
    experiment = simulate_mock_experiment(mode=mode)
    outcome = evaluate_c1_regime(experiment)

    assert outcome.basis_audit.basis_expanding is expected_basis_expansion
    assert outcome.atomicity.valid is expected_atomicity
    assert outcome.im_accepted is expected_acceptance
    assert outcome.frontier_expanded is expected_frontier_expansion


def test_fake_im_is_rejected_with_an_external_reason_code() -> None:
    experiment = simulate_mock_experiment(mode=IMMode.FAKE)
    outcome = evaluate_c1_regime(experiment)

    assert outcome.basis_audit.basis_expanding
    assert outcome.atomicity.reason_codes == ("pump_blocked_im_persists",)
    assert not outcome.im_accepted
    assert outcome.selected_front == outcome.baseline_front


def test_useful_im_expands_the_accepted_capture_frontier() -> None:
    experiment = simulate_mock_experiment(mode=IMMode.USEFUL)
    outcome = evaluate_c1_regime(experiment)

    baseline_maximum = max(
        point.capture_range for point in outcome.baseline_front
    )
    selected_maximum = max(
        point.capture_range for point in outcome.selected_front
    )
    assert baseline_maximum == 0.50
    assert selected_maximum == 1.25
    assert outcome.frontier_expanded

