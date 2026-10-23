"""Explicit coherent phase convention for all numerical DDC channels."""

from dataclasses import dataclass


NEGATIVE_EXPONENTIAL_CONVENTION = (
    "Z = 2 * LPF{x(t) * exp[-i(2*pi*f_ref*t + phi_ref)]}; "
    "I = Re(Z), Q = Im(Z)"
)


@dataclass(frozen=True)
class ChannelReference:
    """Integer combination p*f1 + q*f2 and its coherent phase reference."""

    name: str
    p: int
    q: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel name must not be empty")
        if self.p == 0 and self.q == 0:
            raise ValueError("a channel reference cannot be DC")


def reference_frequency_hz(
    channel: ChannelReference,
    tone_frequencies_hz: tuple[float, float],
) -> float:
    """Calculate the signed integer-combination reference frequency."""
    return float(channel.p * tone_frequencies_hz[0] + channel.q * tone_frequencies_hz[1])


def reference_phase_rad(
    channel: ChannelReference,
    tone_phases_rad: tuple[float, float],
) -> float:
    """Calculate the phase p*phi1 + q*phi2 for a combination channel."""
    return float(channel.p * tone_phases_rad[0] + channel.q * tone_phases_rad[1])

