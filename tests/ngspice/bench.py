"""Compare a frontend's clocked decisions and states with an ngspice circuit.

The two simulators may eventually make different decisions when a comparator
input lies extremely close to a threshold. Compare their states until the
first differing decision; after that, their feedback drives different paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cbadc.circuit import GmC

from . import spice


@dataclass
class Comparison:
    """Clock-aligned records from one frontend in both simulators.

    ``r`` holds cbadc's records, including comparator input ``y``;
    ``v_spice`` holds ngspice controls ``(size, M)``; ``x`` holds its state
    voltages ``(size, N)``. ``first_mismatch`` is the first different decision,
    or ``size`` if all agree.
    """

    r: dict
    v_spice: np.ndarray
    x: np.ndarray
    first_mismatch: int


def run_both(
    af: GmC,
    size: int,
    x0: np.ndarray,
) -> Comparison:
    """Compare a gmC frontend with its physical circuit netlist.

    Read ngspice decisions halfway through each period and states at clock
    edges, matching cbadc's records. Unsupported frontend features raise in
    ``GmC.netlist``; this suite does not synthesize substitute circuits.

    Parameters
    ----------
    af : GmC
        Single realized frontend (J=1) to compare in both simulators.
    size : int
        Number of clock steps.
    x0 : ndarray of float, shape (N,)
        Initial frontend state for both runs.
    Returns
    -------
    Comparison
        Clock-aligned records and the first differing decision step.
    """
    if not isinstance(af, GmC):
        raise TypeError("ngspice comparisons require a GmC frontend")
    tr = spice.run(af.netlist(size, x0))
    v_spice = spice.decisions(tr, af.N, af.M, af.dt, size)
    x = spice.states(tr, af.N, af.dt, size)
    r = af.simulate(size, x0=x0, record="all", verbose=False)
    agree = np.all(np.isclose(v_spice, r["v"][:, :, 0], atol=1e-9), axis=1)
    # False sorts before True, so argmin finds the first differing step.
    fm = size if agree.all() else int(np.argmin(agree))
    return Comparison(r, v_spice, x, fm)


def assert_agree(
    cmp: Comparison,
    *,
    min_agree: int,
    margin: float,
    state_tol: float,
    n_states: int | None = None,
) -> None:
    """Check decisions and states until the first differing decision.

    Require at least ``min_agree`` identical decisions. A later mismatch is
    accepted only when cbadc's comparator input is within ``margin`` of a
    binary threshold at zero. States must agree within ``state_tol`` through
    that decision; later states have different feedback histories.

    Parameters
    ----------
    cmp : Comparison
        Clock-aligned cbadc and ngspice records from one frontend.
    min_agree : int
        Minimum initial steps with identical decisions on every channel.
    margin : float
        Maximum absolute comparator input at an allowed mismatch.
    state_tol : float
        Maximum absolute state difference before the first mismatch.
    n_states : int, optional
        Number of state columns to check; defaults to all ngspice states.

    Returns
    -------
    None
        Raises an assertion error when a required comparison fails.
    """
    fm = cmp.first_mismatch
    size = cmp.v_spice.shape[0]
    if fm < size:
        flipped = ~np.isclose(cmp.v_spice[fm], cmp.r["v"][fm, :, 0], atol=1e-9)
        y = cmp.r["y"][fm, :, 0][flipped]
        assert fm >= min_agree, f"decisions part at period {fm}, |y| = {y}"
        assert np.abs(y).max() < margin, (
            f"decision {fm} differs at y={y}, far from threshold"
        )
    N = cmp.x.shape[1] if n_states is None else n_states
    checked = min(fm + 1, size)
    delta = cmp.x[:checked, :N] - cmp.r["x"][:checked, :N, 0]
    err = np.max(np.abs(delta))
    assert err < state_tol, f"state error {err:.2e} over {checked} steps"
