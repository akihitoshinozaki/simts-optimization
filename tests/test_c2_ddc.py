"""Milestone C2 coherent-demodulation acceptance tests."""

import numpy as np
import pytest

from simts_optimization.dsp import (
    DDCConfig,
    NEGATIVE_EXPONENTIAL_CONVENTION,
    ChannelReference,
    coherent_ddc,
    demodulate_scan_blocks,
    generate_synthetic_rf_record,
    rectangular_window_leakage,
)
from simts_optimization.io import DetuningCalibration


SAMPLE_RATE_HZ = 100_000.0
TONE_FREQUENCIES_HZ = (10_000.0, 13_000.0)
TONE_PHASES_RAD = (0.30, -0.20)
CALIBRATION = DetuningCalibration(
    resonance_coordinate=2.5,
    hertz_per_unit=2.0e6,
    version="synthetic-affine-v1",
)


def test_single_tone_amplitude_and_phase_recovery() -> None:
    frequency_hz = 10_000.0
    amplitude = 1.70
    relative_phase = 0.63
    reference_phase = 0.25
    sample_count = 20_000
    time = np.arange(sample_count) / SAMPLE_RATE_HZ
    waveform = amplitude * np.cos(
        2.0 * np.pi * frequency_hz * time + reference_phase + relative_phase
    )

    result = coherent_ddc(
        waveform,
        sample_rate_hz=SAMPLE_RATE_HZ,
        reference_frequency_hz=frequency_hz,
        reference_phase_rad=reference_phase,
        config=DDCConfig(
            lowpass_cutoff_hz=750.0,
            decimation=20,
            fir_taps=257,
        ),
    )

    recovered = np.mean(result.values)
    np.testing.assert_allclose(
        recovered,
        amplitude * np.exp(1j * relative_phase),
        rtol=2e-3,
        atol=2e-3,
    )
    assert result.convention == NEGATIVE_EXPONENTIAL_CONVENTION


def test_two_simultaneous_tones_recover_without_excess_crosstalk() -> None:
    references = (
        ChannelReference("tone_1", 1, 0),
        ChannelReference("tone_2", 0, 1),
    )
    detuning = np.linspace(-2.0e6, 2.0e6, 5)
    expected = np.column_stack(
        (
            np.full(detuning.size, 1.20 * np.exp(0.40j)),
            np.full(detuning.size, 0.75 * np.exp(-0.55j)),
        )
    )
    record = generate_synthetic_rf_record(
        calibrated_detuning_hz=detuning,
        channel_references=references,
        channel_phasors=expected,
        sample_rate_hz=SAMPLE_RATE_HZ,
        samples_per_block=1000,
        tone_frequencies_hz=TONE_FREQUENCIES_HZ,
        tone_phases_rad=TONE_PHASES_RAD,
        calibration=CALIBRATION,
    )

    recovered = demodulate_scan_blocks(record, references).complex_channels
    integration_time = record.raw_adc.shape[1] / record.sample_rate_hz
    declared_leakage = rectangular_window_leakage(
        TONE_FREQUENCIES_HZ[1] - TONE_FREQUENCIES_HZ[0],
        integration_time,
    )
    assert declared_leakage < 1e-14
    np.testing.assert_allclose(recovered, expected, atol=1e-12)


def test_intermodulation_frequency_and_combination_phase_recovery() -> None:
    reference = ChannelReference("2f1_minus_f2", 2, -1)
    detuning = np.linspace(-1.0e6, 1.0e6, 3)
    expected = np.array([0.40 + 0.20j, -0.10 + 0.50j, 0.30 - 0.25j])[:, None]
    record = generate_synthetic_rf_record(
        calibrated_detuning_hz=detuning,
        channel_references=(reference,),
        channel_phasors=expected,
        sample_rate_hz=SAMPLE_RATE_HZ,
        samples_per_block=1000,
        tone_frequencies_hz=TONE_FREQUENCIES_HZ,
        tone_phases_rad=TONE_PHASES_RAD,
        calibration=CALIBRATION,
    )

    recovered = demodulate_scan_blocks(record, (reference,)).complex_channels
    np.testing.assert_allclose(recovered, expected, atol=1e-12)


def test_positive_reference_phase_shift_rotates_result_negatively() -> None:
    frequency_hz = 10_000.0
    sample_count = 20_000
    time = np.arange(sample_count) / SAMPLE_RATE_HZ
    waveform = np.cos(2.0 * np.pi * frequency_hz * time + 0.30)
    config = DDCConfig(750.0, decimation=20, fir_taps=257)

    nominal = coherent_ddc(
        waveform,
        sample_rate_hz=SAMPLE_RATE_HZ,
        reference_frequency_hz=frequency_hz,
        reference_phase_rad=0.0,
        config=config,
    )
    shifted = coherent_ddc(
        waveform,
        sample_rate_hz=SAMPLE_RATE_HZ,
        reference_frequency_hz=frequency_hz,
        reference_phase_rad=np.pi / 2.0,
        config=config,
    )

    np.testing.assert_allclose(
        np.mean(shifted.values),
        -1j * np.mean(nominal.values),
        atol=1e-12,
    )


def test_decimation_requires_an_antialias_safe_lowpass() -> None:
    with pytest.raises(ValueError, match="decimated Nyquist"):
        coherent_ddc(
            np.ones(1000),
            sample_rate_hz=SAMPLE_RATE_HZ,
            reference_frequency_hz=10_000.0,
            config=DDCConfig(
                lowpass_cutoff_hz=3_000.0,
                decimation=20,
                fir_taps=129,
            ),
        )

    time = np.arange(30_000) / SAMPLE_RATE_HZ
    waveform = np.cos(2.0 * np.pi * 10_000.0 * time)
    waveform += 0.80 * np.cos(2.0 * np.pi * 16_000.0 * time)
    result = coherent_ddc(
        waveform,
        sample_rate_hz=SAMPLE_RATE_HZ,
        reference_frequency_hz=10_000.0,
        config=DDCConfig(
            lowpass_cutoff_hz=1_000.0,
            decimation=20,
            fir_taps=513,
        ),
    )
    np.testing.assert_allclose(np.mean(result.values), 1.0, atol=3e-3)
    assert np.std(result.values) < 0.01

