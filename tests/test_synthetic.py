import numpy as np
import pytest

from simts_optimization.synthetic import (
    detuning,
    dispersive,
    i1,
    i2,
    lorentzian,
    q1,
    q2,
)


def test_detuning_axis_spans_resonance_symmetrically() -> None:
    assert isinstance(detuning, np.ndarray)
    assert detuning.shape == (1001,)
    assert detuning[0] == -5.0
    assert detuning[len(detuning) // 2] == 0.0
    assert detuning[-1] == 5.0
    assert np.all(np.diff(detuning) > 0.0)
    np.testing.assert_allclose(detuning, -detuning[::-1], atol=1e-15)


def test_toy_curve_definitions_at_characteristic_points() -> None:
    sample_points = np.array([1.0, 3.0])

    np.testing.assert_allclose(
        lorentzian(sample_points, gamma=2.0, center=1.0),
        [1.0, 0.5],
    )
    np.testing.assert_allclose(
        dispersive(sample_points, gamma=2.0, center=1.0),
        [0.0, 0.5],
    )


def test_toy_channels_match_detuning_axis() -> None:
    for channel in (i1, q1, i2, q2):
        assert isinstance(channel, np.ndarray)
        assert channel.shape == detuning.shape
        assert np.all(np.isfinite(channel))

    assert not np.array_equal(i1, i2)
    assert not np.array_equal(q1, q2)


@pytest.mark.parametrize("curve", [lorentzian, dispersive])
def test_toy_curves_reject_nonpositive_linewidth(curve) -> None:
    with pytest.raises(ValueError, match="gamma must be positive"):
        curve(detuning, gamma=0.0)
