"""Reproducible theta-to-scans mock experiment for Milestone C1."""

from dataclasses import dataclass

import numpy as np

from .artifacts import nuisance_perturbation
from .drift import DriftState, drift_state_for_scan
from .synthetic_basis import (
    DEFAULT_THETA,
    IMMode,
    MockTheta,
    build_clean_channels,
    build_pump_blocked_channels,
)


CHANNEL_NAMES = ("fundamental_i", "fundamental_q", "im_i", "im_q")


@dataclass(frozen=True)
class MockExperimentConfig:
    """Versioned controls for reproducible C1 synthetic evaluations."""

    detuning_min: float = -5.0
    detuning_max: float = 5.0
    detuning_samples: int = 1001
    scan_count: int = 8
    noise_standard_deviation: float = 0.012
    noise_correlation: float = 0.40
    line_center_drift_per_scan: float = 0.004
    gain_drift_per_scan: float = 0.003
    nuisance_scale: float = 0.025
    seed: int = 1729

    def __post_init__(self) -> None:
        if self.detuning_samples < 5 or self.detuning_samples % 2 == 0:
            raise ValueError("detuning_samples must be an odd integer of at least 5")
        if self.detuning_min >= 0.0 or self.detuning_max <= 0.0:
            raise ValueError("detuning bounds must straddle zero")
        if self.scan_count < 4:
            raise ValueError("scan_count must be at least 4")
        if self.noise_standard_deviation <= 0.0:
            raise ValueError("noise_standard_deviation must be positive")
        if not 0.0 <= self.noise_correlation < 0.8:
            raise ValueError("noise_correlation must lie in [0, 0.8)")


DEFAULT_MOCK_CONFIG = MockExperimentConfig()


@dataclass(frozen=True)
class MockScan:
    """One active scan and its matched pump-blocked control."""

    channels: np.ndarray
    pump_blocked_channels: np.ndarray
    clean_channels: np.ndarray
    correlated_noise: np.ndarray
    nuisance: np.ndarray
    drift: DriftState


@dataclass(frozen=True)
class MockExperiment:
    """Self-contained synthetic evaluation at one theta and IM regime."""

    theta: MockTheta
    mode: IMMode
    config: MockExperimentConfig
    detuning: np.ndarray
    noise_covariance: np.ndarray
    scans: tuple[MockScan, ...]
    channel_names: tuple[str, ...] = CHANNEL_NAMES


def _noise_correlation_matrix(correlation: float) -> np.ndarray:
    return np.array(
        [
            [1.0, correlation, 0.25 * correlation, 0.0],
            [correlation, 1.0, 0.0, 0.50 * correlation],
            [0.25 * correlation, 0.0, 1.0, 0.75 * correlation],
            [0.0, 0.50 * correlation, 0.75 * correlation, 1.0],
        ],
        dtype=float,
    )


def _read_only(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def simulate_mock_experiment(
    theta: MockTheta = DEFAULT_THETA,
    mode: IMMode = IMMode.USEFUL,
    config: MockExperimentConfig = DEFAULT_MOCK_CONFIG,
) -> MockExperiment:
    """Map one physical setting theta to repeated noisy synthetic scans."""
    mode = IMMode(mode)
    detuning = np.linspace(
        config.detuning_min,
        config.detuning_max,
        config.detuning_samples,
        dtype=float,
    )

    correlation = _noise_correlation_matrix(config.noise_correlation)
    theta_noise_scale = 1.0 + 0.08 * theta.resonance_shift**2 + 0.05 * (
        theta.linewidth_scale - 1.0
    ) ** 2
    noise_variance = (config.noise_standard_deviation * theta_noise_scale) ** 2
    noise_covariance = noise_variance * correlation
    np.linalg.cholesky(noise_covariance)

    rng = np.random.default_rng(config.seed)
    scans: list[MockScan] = []
    for scan_index in range(config.scan_count):
        drift = drift_state_for_scan(
            scan_index,
            config.scan_count,
            line_center_step=config.line_center_drift_per_scan,
            gain_step=config.gain_drift_per_scan,
            nuisance_scale=config.nuisance_scale,
        )
        clean_channels = build_clean_channels(
            detuning,
            theta,
            mode,
            line_center_offset=drift.line_center_offset,
            gain_scale=drift.gain_scale,
        )
        nuisance = nuisance_perturbation(detuning, drift.nuisance_amplitude)
        blocked_clean = build_pump_blocked_channels(detuning, theta, mode)

        active_noise = rng.multivariate_normal(
            np.zeros(4),
            noise_covariance,
            size=detuning.size,
        )
        blocked_noise = rng.multivariate_normal(
            np.zeros(4),
            noise_covariance,
            size=detuning.size,
        )
        channels = clean_channels + nuisance + active_noise
        pump_blocked_channels = blocked_clean + 0.20 * nuisance + blocked_noise

        scans.append(
            MockScan(
                channels=_read_only(channels),
                pump_blocked_channels=_read_only(pump_blocked_channels),
                clean_channels=_read_only(clean_channels),
                correlated_noise=_read_only(active_noise),
                nuisance=_read_only(nuisance),
                drift=drift,
            )
        )

    return MockExperiment(
        theta=theta,
        mode=mode,
        config=config,
        detuning=_read_only(detuning),
        noise_covariance=_read_only(noise_covariance),
        scans=tuple(scans),
    )
