"""Synthetic S-IMTS experiment used for offline development and validation."""

from .mock_experiment import (
    DEFAULT_MOCK_CONFIG,
    MockExperiment,
    MockExperimentConfig,
    MockScan,
    simulate_mock_experiment,
)
from .synthetic_basis import DEFAULT_THETA, IMMode, MockTheta

__all__ = [
    "DEFAULT_MOCK_CONFIG",
    "DEFAULT_THETA",
    "IMMode",
    "MockExperiment",
    "MockExperimentConfig",
    "MockScan",
    "MockTheta",
    "simulate_mock_experiment",
]

