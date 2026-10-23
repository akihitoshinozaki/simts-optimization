import numpy as np

from simts_optimization.sim import (
    DEFAULT_THETA,
    IMMode,
    MockTheta,
    simulate_mock_experiment,
)


def test_theta_maps_reproducibly_to_repeated_scans() -> None:
    first = simulate_mock_experiment(DEFAULT_THETA, IMMode.USEFUL)
    second = simulate_mock_experiment(DEFAULT_THETA, IMMode.USEFUL)

    assert first.detuning.shape == (1001,)
    assert len(first.scans) == first.config.scan_count == 8
    assert first.channel_names == (
        "fundamental_i",
        "fundamental_q",
        "im_i",
        "im_q",
    )
    for first_scan, second_scan in zip(first.scans, second.scans):
        assert first_scan.channels.shape == (1001, 4)
        np.testing.assert_array_equal(first_scan.channels, second_scan.channels)
        assert not first_scan.channels.flags.writeable


def test_clean_basis_changes_with_theta() -> None:
    shifted_theta = MockTheta(
        resonance_shift=0.12,
        linewidth_scale=0.90,
        im_strength=0.60,
        im_phase=0.35,
    )
    default_experiment = simulate_mock_experiment(DEFAULT_THETA, IMMode.USEFUL)
    shifted_experiment = simulate_mock_experiment(shifted_theta, IMMode.USEFUL)

    assert not np.allclose(
        default_experiment.scans[0].clean_channels,
        shifted_experiment.scans[0].clean_channels,
    )
    assert not np.array_equal(DEFAULT_THETA.as_array(), shifted_theta.as_array())


def test_noise_is_correlated_across_channels() -> None:
    experiment = simulate_mock_experiment(DEFAULT_THETA, IMMode.USEFUL)
    noise = np.concatenate(
        [scan.correlated_noise for scan in experiment.scans],
        axis=0,
    )
    empirical_correlation = np.corrcoef(noise, rowvar=False)

    assert np.all(np.linalg.eigvalsh(experiment.noise_covariance) > 0.0)
    np.testing.assert_allclose(
        empirical_correlation[0, 1],
        experiment.config.noise_correlation,
        atol=0.03,
    )
    assert abs(empirical_correlation[2, 3]) > 0.20


def test_scans_include_drift_and_nuisance_perturbations() -> None:
    experiment = simulate_mock_experiment(DEFAULT_THETA, IMMode.USEFUL)
    center_offsets = [scan.drift.line_center_offset for scan in experiment.scans]
    gain_scales = [scan.drift.gain_scale for scan in experiment.scans]
    nuisance_norms = [np.linalg.norm(scan.nuisance) for scan in experiment.scans]

    assert center_offsets[0] < center_offsets[-1]
    assert gain_scales[0] < gain_scales[-1]
    assert max(nuisance_norms) > 0.0
    assert not np.allclose(
        experiment.scans[0].clean_channels,
        experiment.scans[-1].clean_channels,
    )
    assert any(
        not np.allclose(scan.channels, scan.clean_channels)
        for scan in experiment.scans
    )


def test_redundant_clean_im_pair_lies_in_baseline_span() -> None:
    experiment = simulate_mock_experiment(DEFAULT_THETA, IMMode.REDUNDANT)

    for scan in experiment.scans:
        baseline_i = scan.clean_channels[:, 0]
        baseline_q = scan.clean_channels[:, 1]
        np.testing.assert_allclose(
            scan.clean_channels[:, 2],
            0.65 * baseline_i - 0.30 * baseline_q,
        )
        np.testing.assert_allclose(
            scan.clean_channels[:, 3],
            0.20 * baseline_i + 0.70 * baseline_q,
        )
