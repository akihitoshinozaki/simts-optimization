"""Outer experiment settings theta that the optimizer is allowed to change."""

from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Parameter:
    """One bounded, continuous experiment setting."""

    name: str
    lower: float
    upper: float
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must be nonempty")
        if not (np.isfinite(self.lower) and np.isfinite(self.upper) and self.lower < self.upper):
            raise ValueError(f"{self.name}: bounds must be finite with lower < upper")


@dataclass(frozen=True)
class DesignSpace:
    """Box of allowed settings; optimizers work in the unit cube."""

    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        if not names or len(set(names)) != len(names):
            raise ValueError("parameters must be nonempty with unique names")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def dimension(self) -> int:
        return len(self.parameters)

    @property
    def lower(self) -> np.ndarray:
        return np.array([p.lower for p in self.parameters])

    @property
    def upper(self) -> np.ndarray:
        return np.array([p.upper for p in self.parameters])

    def validate(self, settings: Mapping[str, float]) -> dict[str, float]:
        """Return settings as floats, or raise ValueError if outside the box."""
        if set(settings) != set(self.names):
            raise ValueError(f"settings must define exactly {self.names}")
        values = {name: float(settings[name]) for name in self.names}
        for p in self.parameters:
            if not np.isfinite(values[p.name]) or not p.lower <= values[p.name] <= p.upper:
                raise ValueError(f"{p.name}={values[p.name]} outside [{p.lower}, {p.upper}]")
        return values

    def to_unit(self, settings: Mapping[str, float]) -> np.ndarray:
        values = self.validate(settings)
        x = np.array([values[name] for name in self.names])
        return (x - self.lower) / (self.upper - self.lower)

    def from_unit(self, unit: np.ndarray) -> dict[str, float]:
        unit = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0)
        if unit.shape != (self.dimension,):
            raise ValueError(f"unit point must have shape ({self.dimension},)")
        x = self.lower + unit * (self.upper - self.lower)
        return {name: float(v) for name, v in zip(self.names, x)}

    def sample_uniform(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return rng.random((count, self.dimension))

    def sobol(self, seed: int, count: int) -> np.ndarray:
        """First ``count`` points of a scrambled Sobol sequence in the unit cube."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # non-power-of-two balance
            return qmc.Sobol(self.dimension, scramble=True, seed=seed).random(count)


C5_DESIGN_SPACE = DesignSpace((
    Parameter("modulation_frequency", 1.0, 10.0, "MHz"),
    Parameter("modulation_depth", 0.1, 3.0, "rad"),
    Parameter("pump_power", 0.1, 4.0, "P_sat"),
))
"""Default C5 search box. The good region's location is not encoded here."""

C5_REFERENCE_SETTINGS = {
    "modulation_frequency": 4.0,
    "modulation_depth": 1.0,
    "pump_power": 1.0,
}
"""Nominal operating point measured periodically to track drift."""
