"""Read ngspice results from circuits made by ``GmC.netlist``.

``tran.dat`` holds time, state voltages ``x_0`` through ``x_(N-1)``, and
control voltages ``v_0`` through ``v_(M-1)``. This module aligns the
transient with cbadc's clocked records.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np


def run(netlist: str) -> np.ndarray:
    """Run an ngspice netlist and return ``[time, x, v]`` at solver times.

    ``GmC.netlist`` supplies the analysis command and ``wrdata``
    instruction. Ngspice writes a header, then one row per transient time;
    the first column is time in seconds.
    """
    # Ngspice otherwise retains every internal node at every solver step.
    # Save only the vectors that GmC.netlist writes to tran.dat.
    vectors = netlist.split("wrdata tran.dat ", 1)[1].splitlines()[0]
    netlist = netlist.replace(".control", f".save {vectors}\n.control", 1)
    with tempfile.TemporaryDirectory(prefix="cbadc-ngspice-") as directory:
        workdir = pathlib.Path(directory)
        (workdir / "bench.cir").write_text(netlist)
        subprocess.run(
            ["ngspice", "-b", "bench.cir"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return np.loadtxt(workdir / "tran.dat", skiprows=1, ndmin=2)


def decisions(tran: np.ndarray, N: int, M: int, dt: float, size: int) -> np.ndarray:
    """Return binary controls ``v`` of shape ``(size, M)``.

    The XSPICE DAC bridge settles after the edge. Read its latest value at
    or before ``(i + 1/2) dt``, then round to the binary values -1 and +1.
    """
    t = tran[:, 0]
    t_mid = (np.arange(size) + 0.5) * dt
    if t[0] > t_mid[0] or t[-1] < t_mid[-1]:
        raise ValueError("ngspice transient does not span the decision times")
    idx = np.searchsorted(t, t_mid, side="right") - 1
    return np.rint(tran[idx, 1 + N : 1 + N + M])


def states(tran: np.ndarray, N: int, dt: float, size: int) -> np.ndarray:
    """Return states ``x`` of shape ``(size, N)`` at ``t_i = i dt``.

    Ngspice puts a row at each clock breakpoint. Use that row because the
    state derivative changes at the edge; interpolate only if a row is absent.
    """
    t = tran[:, 0]
    t_i = np.arange(size) * dt
    idx = np.searchsorted(t, t_i * (1 + 1e-9), side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    if np.max(np.abs(t[idx] - t_i)) > 1e-6 * dt:
        return np.column_stack([np.interp(t_i, t, tran[:, 1 + n]) for n in range(N)])
    return tran[idx, 1 : 1 + N]
