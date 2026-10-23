"""Hidden instrument physics that maps outer settings theta to C1 mock inputs.

This module plays the role of the real apparatus for Milestone C5. It contains
the location of the good operating region, which the optimizer must never read:
``simts_optimization.bo`` only sees scores returned by the evaluation pipeline.
Benchmarks use :meth:`HiddenLandscape.in_good_region` purely as a referee.

The mapping is deliberately physical in shape:

* IM conversion is resonantly enhanced only near a hidden modulation frequency.
* Signal follows the first-order Bessel sideband ``J1(depth)``; depth also
  broadens the line.
* Pump power saturates the signal while adding intensity noise, so the best
  SNR sits at an interior power.
* A slow multiplicative SNR drift acts over the evaluation clock.
* Very deep modulation at high pump power unlocks the laser (failed run), and a
  small fraction of acquisitions drop out at random.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
from scipy.special import jv

from .mock_experiment import DEFAULT_MOCK_CONFIG, MockExperimentConfig
from .synthetic_basis import MockTheta


class AcquisitionFailure(RuntimeError):
    """Raised when the simulated apparatus fails to produce a measurement."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PhysicalState:
    """Inputs handed to the C1 simulator for one acquisition."""

    theta: MockTheta
    config: MockExperimentConfig
    snr_factor: float
    drift_factor: float


_BESSEL_PEAK = 1.8411837813406593  # argmax of J1


@dataclass(frozen=True)
class HiddenLandscape:
    """Ground-truth apparatus response. Optimizers must treat it as a black box."""

    resonance_frequency: float = 6.2
    resonance_width: float = 0.4
    resonance_enhancement: float = 2.2
    pump_noise_scale: float = 1.5
    depth_broadening: float = 0.06
    pump_broadening: float = 0.15
    im_coupling: float = 0.9
    im_phase: float = 0.2
    base_noise: float = 0.0125
    best_snr_gain: float = 3.0
    drift_amplitude: float = 0.25
    drift_period: float = 50.0
    drift_phase: float = 0.7
    unlock_depth: float = 2.4
    unlock_pump: float = 3.0
    dropout_probability: float = 0.02
    good_region_fraction_of_best: float = 0.75
    detuning_samples: int = 401
    scan_count: int = 12
    base_config: MockExperimentConfig = DEFAULT_MOCK_CONFIG

    def _raw_quality(self, frequency, depth, pump) -> np.ndarray:
        eta = np.exp(-0.5 * ((frequency - self.resonance_frequency) / self.resonance_width) ** 2)
        sideband = jv(1, depth) / jv(1, _BESSEL_PEAK)
        pump_snr = (pump / (1.0 + pump)) / np.sqrt(1.0 + (pump / self.pump_noise_scale) ** 2)
        return sideband * pump_snr * (1.0 + self.resonance_enhancement * eta)

    @cached_property
    def _best_raw_quality(self) -> float:
        # Frequency and depth factors peak at known points; search pump densely.
        pumps = np.linspace(1e-3, 10.0, 20001)
        return float(np.max(self._raw_quality(self.resonance_frequency, _BESSEL_PEAK, pumps)))

    def quality(self, settings: Mapping[str, float]) -> float:
        """Drift-free SNR quality in [0, 1]; 1 is the global optimum."""
        frequency, depth, pump = _unpack(settings)
        return float(self._raw_quality(frequency, depth, pump) / self._best_raw_quality)

    def in_good_region(self, settings: Mapping[str, float]) -> bool:
        """Referee-only membership test for the hidden good region."""
        return self.quality(settings) >= self.good_region_fraction_of_best

    def drift_factor(self, clock: float) -> float:
        """Slow multiplicative SNR drift at evaluation clock ``clock``."""
        return float(1.0 + self.drift_amplitude * np.sin(
            2.0 * np.pi * clock / self.drift_period + self.drift_phase))

    def physical_state(
        self,
        settings: Mapping[str, float],
        *,
        clock: float,
        seed: int,
    ) -> PhysicalState:
        """Map settings to simulator inputs, or raise :class:`AcquisitionFailure`."""
        frequency, depth, pump = _unpack(settings)
        if depth > self.unlock_depth and pump > self.unlock_pump:
            raise AcquisitionFailure(
                "laser_unlock", "laser lost lock at high modulation depth and pump power")
        rng = np.random.default_rng([seed, 0xC5])
        if rng.random() < self.dropout_probability:
            raise AcquisitionFailure("acquisition_dropout", "digitizer returned no data")

        quality = max(self.quality(settings), 1e-6)
        drift = self.drift_factor(clock)
        snr = self.best_snr_gain * quality * drift
        eta = np.exp(-0.5 * ((frequency - self.resonance_frequency) / self.resonance_width) ** 2)
        sideband = max(jv(1, depth) / jv(1, _BESSEL_PEAK), 0.0)
        theta = MockTheta(
            resonance_shift=0.0,
            linewidth_scale=float((1.0 + self.depth_broadening * depth**2)
                                  * np.sqrt(1.0 + self.pump_broadening * pump)),
            im_strength=float(self.im_coupling * eta * sideband),
            im_phase=self.im_phase,
        )
        config = replace(
            self.base_config,
            detuning_samples=self.detuning_samples,
            scan_count=self.scan_count,
            noise_standard_deviation=float(self.base_noise / snr),
            seed=int(seed),
        )
        return PhysicalState(theta, config, float(snr), drift)


DEFAULT_LANDSCAPE = HiddenLandscape()


def _unpack(settings: Mapping[str, float]) -> tuple[float, float, float]:
    try:
        return (float(settings["modulation_frequency"]),
                float(settings["modulation_depth"]),
                float(settings["pump_power"]))
    except KeyError as error:
        raise ValueError(f"missing setting: {error.args[0]}") from error
