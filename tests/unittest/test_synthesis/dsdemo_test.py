"""Check that ``delsig.scaleABCD`` balances modulator state swings."""

import numpy as np

import cbadc

ORDER = 5
OSR = 42
N_SIM = 1 << 11


def _state_maxima(ABCD: np.ndarray, amplitude: float) -> np.ndarray:
    """Simulate one ABCD at a sine amplitude and return peak ``|x_n|``."""
    t = np.arange(N_SIM)
    u = amplitude * np.sin(2.0 * np.pi * t / (4 * OSR))
    _, _, xmax, _ = cbadc.delsig.simulateDSM(u, ABCD)
    return np.asarray(xmax).flatten()


def _modulator():
    """Build one fifth-order DELSIG loop for before/after scaling."""
    ntf = cbadc.delsig.synthesizeNTF(ORDER, OSR, 1)
    a, g, b, c = cbadc.delsig.realizeNTF(ntf)
    b[1:] = 0.0
    return cbadc.delsig.stuffABCD(a, g, b, c)


def test_scaleABCD_equalises_the_state_maxima():
    """Scaling ABCD gives every state a similar share of the swing budget.

    Simulate the original and scaled fifth-order loop at comparable input
    amplitudes. Compare the spread of peak ``|x_n|`` and ensure each scaled
    state remains below full scale.
    """
    ABCD = _modulator()
    ABCDs, umax, _ = cbadc.delsig.scaleABCD(ABCD)

    # `umax` is the scaled modulator's usable input range, a sensible fraction
    # of full scale rather than a degenerate 0 or a value above full scale.
    assert 0.0 < umax <= 1.0

    unscaled = _state_maxima(ABCD, 0.3)
    scaled = _state_maxima(ABCDs, 0.3 * umax)

    assert np.all(np.isfinite(unscaled)) and np.all(np.isfinite(scaled))

    # Ignore any state that never moves -- the ratio is meaningless there.
    moving = scaled > 1e-3
    assert moving.sum() == ORDER, f"a scaled state never moved: {scaled}"

    def spread(m):
        return float(np.max(m) / max(float(np.min(m)), 1e-12))

    assert spread(unscaled) > 10.0, (
        f"the unscaled chain was expected to be badly balanced, got {unscaled}"
    )
    assert spread(scaled) < 2.0, (
        f"scaleABCD did not equalise the state maxima: {scaled}"
    )

    # And no scaled state may exceed the swing budget it was scaled into.
    assert np.max(scaled) < 1.0, scaled


def test_scaleABCD_preserves_the_modulator_order():
    """State-coordinate scaling retains ABCD shape and finite entries."""
    ABCD = _modulator()
    ABCDs, _, _ = cbadc.delsig.scaleABCD(ABCD)
    assert ABCDs.shape == ABCD.shape
    assert np.all(np.isfinite(ABCDs))
