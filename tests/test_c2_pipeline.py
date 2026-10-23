"""Milestone C2 scan registration and statistics acceptance tests."""

import numpy as np

from simts_optimization.dsp import ChannelReference, generate_synthetic_rf_record
from simts_optimization.estimation import (
    DerivativeConfig,
    estimate_noise_covariance,
    estimate_smoothed_derivatives,
)
from simts_optimization.io import (
    DetuningCalibration,
    calibrate_detuning,
    register_complex_channels,
)
from simts_optimization.pipeline import process_raw_scan


def test_raw_and_calibrated_detuning_are_preserved_separately() -> None:
    calibration = DetuningCalibration(2.0, 5.0e6, "cal-v1")
    raw_coordinate = np.linspace(1.8, 2.2, 21)
    original_raw = raw_coordinate.copy()
    detuning = calibrate_detuning(raw_coordinate, calibration)

    np.testing.assert_array_equal(raw_coordinate, original_raw)
    np.testing.assert_allclose(detuning, np.linspace(-1.0e6, 1.0e6, 21))


def test_complex_scan_registration_uses_declared_detuning_axis() -> None:
    source = np.linspace(-1.1e6, 1.1e6, 221)
    target = np.linspace(-1.0e6, 1.0e6, 101)
    normalized = source / 1.0e6
    channels = np.column_stack(
        (normalized**2 + 1j * normalized, normalized + 0.5j * normalized**2)
    )

    registered = register_complex_channels(source, channels, target)
    target_normalized = target / 1.0e6
    expected = np.column_stack(
        (
            target_normalized**2 + 1j * target_normalized,
            target_normalized + 0.5j * target_normalized**2,
        )
    )
    np.testing.assert_allclose(registered, expected, atol=3e-5)


def test_raw_scan_pipeline_recovers_known_complex_channel_curves() -> None:
    sample_rate_hz = 100_000.0
    tone_frequencies_hz = (10_000.0, 13_000.0)
    references = (
        ChannelReference("fundamental", 1, 0),
        ChannelReference("im", 2, -1),
    )
    detuning = np.linspace(-1.0e6, 1.0e6, 101)
    normalized = detuning / 500_000.0
    lorentzian = 1.0 / (1.0 + normalized**2)
    dispersive = normalized / (1.0 + normalized**2)
    expected_channels = np.column_stack(
        (lorentzian + 1j * dispersive, 0.50 * dispersive + 0.30j * lorentzian)
    )
    calibration = DetuningCalibration(1.5, 2.0e6, "pipeline-cal-v1")
    record = generate_synthetic_rf_record(
        calibrated_detuning_hz=detuning,
        channel_references=references,
        channel_phasors=expected_channels,
        sample_rate_hz=sample_rate_hz,
        samples_per_block=1000,
        tone_frequencies_hz=tone_frequencies_hz,
        tone_phases_rad=(0.20, -0.15),
        calibration=calibration,
    )

    processed = process_raw_scan(
        record,
        references,
        derivative_config=DerivativeConfig(window_length=11, polynomial_order=3),
    )
    np.testing.assert_allclose(processed.complex_channels, expected_channels, atol=2e-12)
    np.testing.assert_allclose(processed.basis[:, 0], lorentzian, atol=2e-12)
    np.testing.assert_allclose(processed.basis[:, 1], dispersive, atol=2e-12)
    np.testing.assert_allclose(
        processed.derivative_estimate.frequency_slope[:2],
        [0.0, 1.0 / 500_000.0],
        rtol=0.02,
        atol=2e-9,
    )
    assert not record.raw_adc.flags.writeable
    assert not record.raw_actuator_coordinate.flags.writeable


def test_smoothed_derivative_recovers_known_noisy_function_slopes() -> None:
    rng = np.random.default_rng(123)
    detuning = np.linspace(-2.0, 2.0, 401)
    lorentzian = 1.0 / (1.0 + detuning**2)
    dispersive = detuning / (1.0 + detuning**2)
    noisy_basis = np.column_stack((lorentzian, dispersive))
    noisy_basis += rng.normal(scale=0.001, size=noisy_basis.shape)

    estimate = estimate_smoothed_derivatives(
        detuning,
        noisy_basis,
        DerivativeConfig(window_length=31, polynomial_order=3),
    )
    np.testing.assert_allclose(
        estimate.frequency_slope,
        [0.0, 1.0],
        atol=0.02,
    )
    assert estimate.method == "Savitzky-Golay (window=31, order=3)"


def test_repeated_scan_covariance_recovers_cross_channel_noise() -> None:
    rng = np.random.default_rng(456)
    target_covariance = np.array(
        [
            [1.0, 0.35, 0.10],
            [0.35, 0.80, -0.15],
            [0.10, -0.15, 0.60],
        ]
    )
    deterministic_shape = np.column_stack(
        (
            np.linspace(-1.0, 1.0, 101),
            np.linspace(-1.0, 1.0, 101) ** 2,
            np.sin(np.linspace(-1.0, 1.0, 101)),
        )
    )
    noise = rng.multivariate_normal(
        np.zeros(3),
        target_covariance,
        size=(50, 101),
    )
    repeated_scans = deterministic_shape[None, :, :] + noise

    estimate = estimate_noise_covariance(
        repeated_scans,
        diagonal_loading=1e-6,
    )
    np.testing.assert_allclose(
        estimate.unregularized_covariance,
        target_covariance,
        atol=0.04,
    )
    assert np.all(estimate.eigenvalues > 0.0)
    assert estimate.condition_number > 1.0
    assert estimate.diagonal_loading == 1e-6

