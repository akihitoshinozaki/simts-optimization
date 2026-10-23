"""Milestone C3 basis-audit acceptance and stress tests."""

import numpy as np
import pytest

from simts_optimization.basis_audit import (
    BasisAuditConfig,
    audit_basis_expansion,
    audit_mock_experiment,
    rotate_iq_pairs,
)
from simts_optimization.sim import IMMode, simulate_mock_experiment


TEST_CONFIG = BasisAuditConfig(bootstrap_resamples=40, bootstrap_seed=303)


def _baseline(detuning: np.ndarray) -> np.ndarray:
    lorentzian = 1.0 / (1.0 + detuning**2)
    dispersive = detuning / (1.0 + detuning**2)
    return np.column_stack((dispersive, lorentzian))


def _independent_candidate(detuning: np.ndarray) -> np.ndarray:
    narrow_dispersive = (detuning / 0.70) / (1.0 + (detuning / 0.70) ** 2)
    shifted_lorentzian = 1.0 / (1.0 + ((detuning - 1.20) / 0.55) ** 2)
    return np.column_stack((narrow_dispersive, shifted_lorentzian))


def _noisy_scans(
    *,
    seed: int,
    candidate_kind: str,
    scan_count: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    detuning = np.linspace(-5.0, 5.0, 501)
    baseline = _baseline(detuning)

    def make_split() -> np.ndarray:
        baseline_scans = np.repeat(baseline[None, :, :], scan_count, axis=0)
        baseline_scans += rng.normal(
            scale=0.02, size=(scan_count, detuning.size, 2)
        )
        if candidate_kind == "noise":
            candidate_scans = rng.normal(
                scale=0.02, size=(scan_count, detuning.size, 2)
            )
        elif candidate_kind == "independent":
            candidate = _independent_candidate(detuning)
            candidate_scans = np.repeat(candidate[None, :, :], scan_count, axis=0)
            candidate_scans += rng.normal(
                scale=0.02, size=(scan_count, detuning.size, 2)
            )
        elif candidate_kind == "redundant":
            coefficients = np.array([[0.60, -0.20], [0.25, 0.75]])
            candidate_scans = baseline_scans @ coefficients
        else:
            raise ValueError(candidate_kind)
        return np.concatenate((baseline_scans, candidate_scans), axis=2)

    return detuning, make_split(), make_split()


def _audit(
    detuning: np.ndarray,
    development: np.ndarray,
    validation: np.ndarray,
):
    return audit_basis_expansion(
        detuning,
        development,
        validation,
        baseline_columns=(0, 1),
        candidate_columns=(2, 3),
        config=TEST_CONFIG,
    )


def test_exactly_redundant_candidate_is_rank_neutral() -> None:
    detuning, development, validation = _noisy_scans(
        seed=10, candidate_kind="redundant"
    )
    result = _audit(detuning, development, validation)

    assert not result.basis_expanding
    assert result.projection.normalized_residual_energy < 1e-20
    assert result.noise_whitened_svd.stable_rank == 0


def test_known_independent_direction_is_detected() -> None:
    detuning, development, validation = _noisy_scans(
        seed=11, candidate_kind="independent"
    )
    result = _audit(detuning, development, validation)

    assert result.basis_expanding
    assert result.projection.residual_repeatability > 0.90
    assert result.noise_whitened_svd.stable_rank >= 1
    assert all(result.decision_checks.values())


def test_pure_noise_does_not_inflate_stable_rank() -> None:
    detuning, development, validation = _noisy_scans(
        seed=12, candidate_kind="noise"
    )
    result = _audit(detuning, development, validation)

    assert not result.basis_expanding
    assert result.noise_whitened_svd.stable_rank == 0
    assert not result.decision_checks["repeatable_held_out_residual"]
    assert not result.decision_checks["stable_residual_subspace"]


def test_basis_audit_is_invariant_to_coherent_iq_rotations() -> None:
    detuning, development, validation = _noisy_scans(
        seed=13, candidate_kind="independent"
    )
    original = _audit(detuning, development, validation)
    rotated = _audit(
        detuning,
        rotate_iq_pairs(development, (0.37, -0.82)),
        rotate_iq_pairs(validation, (0.37, -0.82)),
    )

    assert original.basis_expanding == rotated.basis_expanding
    assert original.decision_checks == rotated.decision_checks
    assert rotated.projection.normalized_residual_energy == pytest.approx(
        original.projection.normalized_residual_energy, rel=1e-12
    )
    np.testing.assert_allclose(
        rotated.noise_whitened_svd.singular_value_ratios,
        original.noise_whitened_svd.singular_value_ratios,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rotated.subspace.residual_angles_degrees,
        original.subspace.residual_angles_degrees,
        rtol=1e-10,
        atol=1e-10,
    )


def test_bootstrap_intervals_are_ordered_and_reproducible() -> None:
    detuning, development, validation = _noisy_scans(
        seed=14, candidate_kind="independent"
    )
    first = _audit(detuning, development, validation)
    second = _audit(detuning, development, validation)

    assert first.bootstrap == second.bootstrap
    for interval in (
        first.bootstrap.normalized_residual_energy,
        first.bootstrap.leading_singular_value_ratio,
    ):
        assert interval.lower <= interval.median <= interval.upper


@pytest.mark.parametrize(
    ("mode", "expected_expansion"),
    [(IMMode.REDUNDANT, False), (IMMode.USEFUL, True)],
)
def test_c1_mock_data_runs_through_c3_audit(
    mode: IMMode,
    expected_expansion: bool,
) -> None:
    result = audit_mock_experiment(
        simulate_mock_experiment(mode=mode), config=TEST_CONFIG
    )
    assert result.basis_expanding is expected_expansion


def test_development_and_validation_must_share_a_registered_grid() -> None:
    detuning, development, validation = _noisy_scans(
        seed=15, candidate_kind="independent"
    )
    with pytest.raises(ValueError, match="shapes must align"):
        _audit(detuning, development, validation[:, :-1, :])
