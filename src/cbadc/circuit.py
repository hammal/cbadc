"""Physical circuit realizations of continuous-time frontends.

``GmC`` retains the frontend's states as capacitor voltages. Its
transconductances are sized by ``C_design``; the realized ``Cint + Cp`` and
finite output resistance determine the actual dynamics.
``ActiveRC`` adds one finite-bandwidth op-amp state per integrator.
"""

from __future__ import annotations

import math
import operator
import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .frontend import Frontend, Samples
from .noise import k_B, per_state_intensity, positive_semidefinite


def _component(value: ArrayLike, name: str, N: int) -> np.ndarray:
    """Normalize a per-state component to (J, K, N)."""
    result = np.asarray(value, dtype=float)
    if result.ndim <= 1:
        try:
            result = np.broadcast_to(result, (N,))[None, None, :]
        except ValueError as exc:
            raise ValueError(f"{name} must end in ({N},)") from exc
    elif result.ndim == 2 and result.shape[-1] == N:
        result = result[:, None, :]
    elif result.ndim != 3 or result.shape[-1] != N:
        raise ValueError(f"{name} must have shape (N,), (J, N), or (J, K, N)")
    if 0 in result.shape:
        raise ValueError(f"{name} must have nonempty J and K axes")
    return result.copy()


def _component_axes(af: Frontend, *values: np.ndarray) -> tuple[int, int]:
    """Combine component batches and cyclic phases with the ideal frontend."""
    try:
        J = np.broadcast_shapes((af.J_frontend,), *(v.shape[:1] for v in values))[0]
    except ValueError as exc:
        raise ValueError(
            "incompatible J batch dimensions for circuit components"
        ) from exc
    return J, math.lcm(af.K, *(v.shape[1] for v in values))


def _phases(value: np.ndarray, J: int, K: int, N: int) -> np.ndarray:
    """Broadcast a component over batches and repeat its K phases."""
    return np.broadcast_to(value[:, np.arange(K) % value.shape[1]], (J, K, N))


def _add_state_intensity(af: Frontend, intensity: np.ndarray) -> None:
    """Add independent circuit noise to the frontend's (J, K, N, N) intensity."""
    previous = af.state_covariance
    if previous is None:
        af.state_covariance = intensity
        return
    J = np.broadcast_shapes(previous.shape[:1], intensity.shape[:1])[0]
    K = math.lcm(previous.shape[1], intensity.shape[1])
    af.state_covariance = np.broadcast_to(
        previous[:, np.arange(K) % previous.shape[1]], (J, K) + previous.shape[-2:]
    ) + np.broadcast_to(
        intensity[:, np.arange(K) % intensity.shape[1]], (J, K) + intensity.shape[-2:]
    )


def _add_comparator_noise(af: Frontend, rms: Optional[ArrayLike]) -> None:
    """Add independent, input-referred comparator variance per decision."""
    af.comparator_input_noise_rms = None
    if rms is None:
        return
    try:
        value = np.broadcast_to(np.asarray(rms, dtype=float), (af.M,))
    except ValueError as exc:
        raise ValueError(
            f"comparator_input_noise_rms must broadcast to ({af.M},)"
        ) from exc
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError("comparator_input_noise_rms must be finite and nonnegative")
    af.comparator_input_noise_rms = value.copy()
    covariance = np.diag(value**2)
    if af.output_covariance is not None:
        covariance += af.output_covariance
    af.output_covariance = covariance


class GmC(Frontend):
    """Realize ``dx/dt = A x + B [u; v]`` with transconductors and capacitors.

    The fixed transconductances are ``gm_A = C_design[..., :, None] * A``
    and ``gm_B = C_design[..., :, None] * B``. The realized capacitor is
    ``Ctot = Cint + Cp``. Row-wise division gives
    ``A_real = gm_A/Ctot - diag(1/(Ro*Ctot))`` and ``B_real = gm_B/Ctot``.
    Component arrays use ``(J, K, N)``; a ``(J, N)`` array varies by batch
    but is shared across phases. ``C_design`` defaults to ``Cint``. To
    compensate a nominal parasitic, set ``C_design = Cint_nominal + Cp_nominal``
    and pass the actual ``Cint`` and ``Cp`` separately. These actual values
    may deviate above or below nominal, but ``Cint > 0`` and ``Cp >= 0``.
    ``C`` and ``D`` still define the comparator inputs. A realized frontend
    cannot be realized again; start each topology from the ideal frontend.

    Parameters
    ----------
    af : Frontend
        Ideal continuous-time frontend, left unchanged.
    Cint : float or array_like, shape (N,), (J, N), or (J, K, N)
        Actual integration capacitance in farads; defaults to ``1e-12``.
    Ro : float or array_like, shape (N,), (J, N), or (J, K, N), optional
        Output resistance in ohms; defaults to ``inf`` (no leak).
    Cp : float or array_like, shape (N,), (J, N), or (J, K, N), optional
        Actual parasitic capacitance in farads; defaults to zero.
    v_n : float or array_like, shape (N,), optional
        One-sided, state-referred noise density in V/sqrt(Hz). Overrides the
        frontend's state covariance when supplied; defaults to ``None``.
        Set ``thermal_noise=False`` when using this legacy override.
    C_design : float or array_like, shape (N,), (J, N), or (J, K, N), optional
        Capacitance used to size the transconductors, in farads. Defaults to
        ``Cint``, preserving the original realization.
    thermal_noise : bool, optional
        Add independent white current noise from every modeled transconductor
        and finite ``Ro`` to the inherited state covariance. Defaults to
        ``True``; pass ``False`` for a noiseless circuit model. Each
        transconductor contributes one-sided current PSD
        ``4*k_B*temperature*gamma*abs(gm)``; ``Ro`` contributes
        ``4*k_B*temperature/Ro``. These are divided by ``Ctot**2`` and by
        two to obtain the two-sided state-noise intensity. This assumes one
        independent noisy OTA for each nonzero ``gm_A`` and ``gm_B`` entry.
    temperature : float, optional
        Absolute temperature in kelvin; defaults to ``300``.
    gamma : float, optional
        OTA excess thermal-noise factor; defaults to ``1.5``. It is a model
        parameter, not determined by ``gm`` or the capacitors.
    comparator_input_noise_rms : float or array_like, shape (M,), optional
        Independent input-referred comparator noise in RMS volts per decision.
        Adds ``diag(comparator_input_noise_rms**2)`` to the inherited
        ``output_covariance``; defaults to ``None`` (no added noise). To use
        a white voltage density, first multiply it by the square root of the
        comparator's equivalent noise bandwidth.

    Returns
    -------
    GmC
        Independent circuit realization, with the frontend's inputs, DACs,
        comparator settings and other non-idealities copied.
    """

    def __init__(
        self,
        af: Frontend,
        Cint: ArrayLike = 1e-12,
        Ro: Optional[ArrayLike] = None,
        Cp: Optional[ArrayLike] = None,
        v_n: Optional[ArrayLike] = None,
        *,
        C_design: Optional[ArrayLike] = None,
        thermal_noise: bool = True,
        temperature: float = 300.0,
        gamma: float = 1.5,
        comparator_input_noise_rms: Optional[ArrayLike] = None,
    ) -> None:
        if not isinstance(af, Frontend) or af.discrete:
            raise ValueError("GmC needs a continuous-time Frontend")
        if getattr(af, "_realization", None) is not None:
            raise ValueError(
                "realize an ideal Frontend, not another circuit realization"
            )
        if thermal_noise and v_n is not None:
            raise ValueError("choose thermal_noise or v_n, not both")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")

        def per_state(value: ArrayLike, name: str) -> np.ndarray:
            try:
                result = np.broadcast_to(np.asarray(value, dtype=float), (af.N,)).copy()
            except ValueError as exc:
                raise ValueError(f"{name} must broadcast to ({af.N},)") from exc
            return result

        self.Cint = _component(Cint, "Cint", af.N)
        self.Ro = _component(np.inf if Ro is None else Ro, "Ro", af.N)
        self.Cp = _component(0.0 if Cp is None else Cp, "Cp", af.N)
        self.C_design = (
            self.Cint.copy()
            if C_design is None
            else _component(C_design, "C_design", af.N)
        )
        if not np.all(np.isfinite(self.Cint)) or np.any(self.Cint <= 0):
            raise ValueError("Cint must be finite and positive")
        if np.any(np.isnan(self.Ro)) or np.any(self.Ro <= 0):
            raise ValueError("Ro must be positive or infinite")
        if not np.all(np.isfinite(self.Cp)) or np.any(self.Cp < 0):
            raise ValueError("Cp must be finite and nonnegative")
        if not np.all(np.isfinite(self.C_design)) or np.any(self.C_design <= 0):
            raise ValueError("C_design must be finite and positive")

        J, K = _component_axes(af, self.Cint, self.Cp, self.Ro, self.C_design)
        self.Ctot = _phases(self.Cint, J, K, af.N) + _phases(self.Cp, J, K, af.N)
        if not np.all(np.isfinite(self.Ctot)) or np.any(self.Ctot <= 0):
            raise ValueError("Cint + Cp must be finite")
        A_ideal = np.broadcast_to(
            af.A[:, np.arange(K) % af.A.shape[1]], (J, K, af.N, af.N)
        )
        B_ideal = np.broadcast_to(
            af.B[:, np.arange(K) % af.B.shape[1]], (J, K, af.N, af.L + af.M)
        )
        self.gm_A = _phases(self.C_design, J, K, af.N)[..., :, None] * A_ideal
        self.gm_B = _phases(self.C_design, J, K, af.N)[..., :, None] * B_ideal
        self.thermal_noise = bool(thermal_noise)
        self.temperature, self.gamma = float(temperature), float(gamma)
        leak = (
            np.eye(af.N)
            * (1 / (_phases(self.Ro, J, K, af.N) * self.Ctot))[..., :, None]
        )
        A = self.gm_A / self.Ctot[..., :, None] - leak
        B = self.gm_B / self.Ctot[..., :, None]
        super().__init__(
            A,
            B,
            af.C.copy(),
            af.D.copy(),
            dt=af.dt,
            levels=af.levels,
            input=deepcopy(af.input),
            dac=deepcopy(af.dac),
            C_hat=None if af.C_hat is None else af.C_hat.copy(),
            D_hat=None if af.D_hat is None else af.D_hat.copy(),
        )
        self._realization = "GmC"
        self.x0 = deepcopy(af.x0)
        self.state_covariance = deepcopy(af.state_covariance)
        self.output_covariance = deepcopy(af.output_covariance)
        _add_comparator_noise(self, comparator_input_noise_rms)
        self.comparator_input_offset = af.comparator_input_offset
        self._flicker = deepcopy(af._flicker)
        for name in self._per_state:
            setattr(self, name, getattr(af, name))
        if v_n is not None:
            density = per_state(v_n, "v_n")
            if not np.all(np.isfinite(density)) or np.any(density < 0):
                raise ValueError("v_n must be finite and nonnegative")
            self.state_covariance = per_state_intensity(density)
        if thermal_noise:
            conductance = gamma * (
                np.abs(self.gm_A).sum(axis=-1) + np.abs(self.gm_B).sum(axis=-1)
            ) + 1 / _phases(self.Ro, J, K, af.N)
            variance = 2 * k_B * temperature * conductance / self.Ctot**2
            intensity = np.eye(af.N) * variance[..., :, None]
            _add_state_intensity(self, intensity)

    def __getitem__(self, j: int) -> "GmC":
        """Select batch member ``j`` with its matching circuit components.

        Parameters
        ----------
        j : int
            Batch index in ``[-J, J)``.

        Returns
        -------
        GmC
            Independent circuit realization with ``J = 1``.
        """
        j = range(self.J)[operator.index(j)]
        new = super().__getitem__(j)
        for name in ("Cint", "Cp", "C_design", "Ro", "Ctot", "gm_A", "gm_B"):
            value = getattr(new, name)
            setattr(new, name, value[[j % value.shape[0]]])
        return new

    def _state_netlist_lines(self, F: Optional[np.ndarray]) -> list[str]:
        """Emit capacitor, leak, transconductor, and state-noise rows.

        Parameters
        ----------
        F : ndarray, shape (N, N), or None
            Factor of the state-noise intensity.

        Returns
        -------
        list[str]
            Ngspice component lines for the state equations.
        """
        lines = []
        # Gname from to sensed_plus sensed_minus gm injects gm*V into "to".
        for n in range(self.N):
            lines.append(f"Cx{n} x{n} 0 {float(self.Ctot[0, 0, n])!r}")
            if np.isfinite(self.Ro[0, 0, n]):
                lines.append(f"Rx{n} x{n} 0 {float(self.Ro[0, 0, n])!r}")
            for j, gm in enumerate(self.gm_A[0, 0, n]):
                if gm:
                    lines.append(f"Gax{n}_{j} 0 x{n} x{j} 0 {float(gm)!r}")
            for ell, gm in enumerate(self.gm_B[0, 0, n, : self.L]):
                if gm:
                    lines.append(f"Gu{n}_{ell} 0 x{n} u{ell} 0 {float(gm)!r}")
            for m, gm in enumerate(self.gm_B[0, 0, n, self.L :]):
                if gm:
                    lines.append(f"Gd{n}_{m} 0 x{n} d{m} 0 {float(gm)!r}")
            if F is not None:
                # F w is a state derivative; Ctot converts it to current.
                rhs = [f"{float(a)!r}*v(w{p})" for p, a in enumerate(F[n]) if a]
                if rhs:
                    lines.append(
                        f"Bwn{n} 0 x{n} I = {float(self.Ctot[0, 0, n])!r}*({' + '.join(rhs)})"
                    )
        return lines

    def netlist(
        self, size: int, x0: Optional[ArrayLike] = None, seed: Optional[int] = None
    ) -> str:
        """Generate an ngspice circuit to validate :meth:`simulate`.

        A netlist describes a circuit, one component or command per line.
        ``Cx`` stores each state voltage; ``R`` models finite output resistance;
        ``G`` transconductors inject the ``gm_A x + gm_B [u; d]`` current.
        ``V`` and ``B`` provide inputs, noise, comparators and DAC voltages;
        ``A`` denotes XSPICE clocked devices. ``0`` is ground. The footer
        initializes states, runs transient analysis and records voltages.

        Parameters
        ----------
        size : int
            Number of clock decisions, including the initial decision.
        x0 : array_like, shape (N,), optional
            Initial frontend state; defaults to zero.
        seed : int, optional
            Ngspice noise seed; defaults to ngspice's seed.

        Returns
        -------
        str
            Complete netlist for ``J = K = 1`` gmC frontends
            with binary comparators and windowed constant DACs.
            The initial decision is deterministic; compare noisy decisions
            from ``t = dt`` onward.

        >>> af = Frontend([[0.0]], [[0.0, 0.0]], [[1.0]], dt=1e-5)
        >>> circuit = GmC(af).netlist(2)
        >>> "Cx0 x0 0" in circuit and "wrdata tran.dat" in circuit
        True
        """
        # Reject frontend features this SPICE validation circuit cannot represent.
        if self.discrete or self.J != 1 or self.K != 1 or size < 2:
            raise NotImplementedError("need continuous time, J=K=1, size>=2")
        if np.any(self.levels != 2) or np.any(self.D[..., self.L :]):
            raise NotImplementedError(
                "need binary comparators without decision feedthrough"
            )
        if np.any(np.isfinite(self.state_min)) or np.any(np.isfinite(self.state_max)):
            raise NotImplementedError("state rails are not represented")
        if np.any(np.isfinite(self.slew_rate)):
            raise NotImplementedError("slew limits are not represented")
        if self._flicker:
            raise NotImplementedError("flicker noise is not represented")
        if any(
            isinstance(w, Samples) or not w.autonomous or w.J != 1 or w.K != 1
            for w in self.input
        ):
            raise NotImplementedError("need one autonomous source per input")
        # Read each constant DAC pulse's amplitude and (delay, width) window.
        pulse_amplitudes, windows = [], []
        for dac in self.dac:
            if dac.N != 1 or np.any(dac.A):
                raise NotImplementedError("need constant DAC waveforms")
            pulse_amplitudes.append(float(dac.C[0, 0, 0] @ dac.x0[0]))
            windows.append(dac.window or (0.0, 1.0))

        x0 = np.zeros(self.N) if x0 is None else np.asarray(x0, dtype=float)
        if x0.shape != (self.N,):
            raise ValueError(f"x0 must have shape ({self.N},)")
        # XSPICE's 1 ps bridge delay can skew short-period simulations:
        # warn when four such delays exceed 0.01% of dt (dt < 40 ns).
        bridge_delay = 1e-12
        if 4 * bridge_delay > 1e-4 * self.dt:
            warnings.warn(
                "XSPICE bridge delays exceed 0.01% of dt and may skew validation; use a longer dt",
                RuntimeWarning,
                stacklevel=2,
            )
        # One matrix set: C (M,N), D (M,L+M); gm_A and gm_B drive the capacitors.
        C, D = self.C[0, 0], self.D[0, 0]
        # F (N,N), G (M,M) are noise factors, not covariance matrices:
        # F @ F.T = state_covariance[0,0]; G @ G.T = output_covariance.
        # Factor with Cholesky or, for semidefinite matrices, eigenvectors.
        F = (
            None
            if self.state_covariance is None
            else positive_semidefinite(self.state_covariance[0, 0])
        )
        G = (
            None
            if self.output_covariance is None
            else positive_semidefinite(self.output_covariance)
        )
        # Tight validation tolerances, all stricter than ngspice defaults:
        # reltol: relative error (1e-6 vs 1e-3 default).
        # abstol: absolute current error, A (1e-15 vs 1e-12).
        # vntol: absolute voltage error, V (1e-9 vs 1e-6).
        # chgtol: charge tolerance, C (1e-20 vs 1e-14).
        # Build the final netlist line by line: title, then simulation settings.
        lines = [
            f"* cbadc {getattr(self, '_netlist_title', 'gm-C')} frontend: capacitor voltages are states",
            ".options reltol=1e-6 abstol=1e-15 vntol=1e-9 chgtol=1e-20",
        ]

        # Netlist alternatives for u0 (node u0 relative to ground 0):
        #   Vu0 u0 0 dc 1.0                 independent voltage source
        #   Bu0 u0 0 V = 1.0*exp(-1.0*time) formula-driven voltage
        u0 = np.empty(self.L)
        for ell, source in enumerate(self.input):
            a, c, z0 = source.A[0, 0], source.C[0, 0, 0], source.x0[0]
            u_init = float(c @ z0)
            u0[ell] = u_init
            # One state gives a constant when its rate is zero, an exponential otherwise.
            if source.N == 1:
                rate = float(a[0, 0])
                value = (
                    f"dc {u_init!r}"
                    if rate == 0
                    else f"V = {u_init!r}*exp({rate!r}*time)"
                )
            elif source.N == 2 and a[0, 0] == a[1, 1] == 0 and a[0, 1] == -a[1, 0]:
                # A two-state rotation produces a sinusoid.
                omega = float(a[1, 0])
                sine = float(c[1] * z0[0] - c[0] * z0[1])
                value = (
                    f"V = {u_init!r}*cos({omega!r}*time) + {sine!r}*sin({omega!r}*time)"
                )
            else:
                raise NotImplementedError(
                    "input source must be constant, exponential, or sinusoidal"
                )
            element = "V" if value.startswith("dc ") else "B"
            lines.append(f"{element}u{ell} u{ell} 0 {value}")

        # Example netlist line: Vw0 w0 0 dc 0 trnoise(1000.0 1e-06 0 0)
        # trnoise(rms_V, step_s, 0, 0) draws Gaussian transient noise at
        # each step; the zeros disable 1/f noise. This approximates white noise.
        # Independent Vw and Veta noise voltages are mixed by F and G below.
        if F is not None:
            h = self.dt / 32
            noise_rms = float(1 / np.sqrt(h))
            for p in range(self.N):
                if np.any(F[:, p]):
                    lines.append(f"Vw{p} w{p} 0 dc 0 trnoise({noise_rms!r} {h!r} 0 0)")
        if G is not None:
            for p in range(self.M):
                if np.any(G[:, p]):
                    lines.append(f"Veta{p} eta{p} 0 dc 0 trnoise(1 {self.dt!r} 0 0)")

        # Emit C, R and G rows for the realized state equations. For gm-C,
        # Gax0_1 0 x0 x1 0 2e-6 injects 2e-6*v(x1) amperes into x0.
        lines.extend(self._state_netlist_lines(F))

        # Vfreq gives the analog solver a breakpoint before each decision.
        # Both control levels select fs; Aclk clocks the flip-flops at i*dt.
        rise = 4 * bridge_delay
        lines += [
            f"Vfreq fclk 0 PULSE(0 1 {self.dt - rise!r} {rise!r} {rise!r} {self.dt / 2 - rise!r} {self.dt!r})",
            f".model osc d_osc(cntl_array=[0 1] freq_array=[{self.fs!r} {self.fs!r}] duty_cycle=0.5 init_phase=180 rise_delay={bridge_delay!r} fall_delay={bridge_delay!r})",
            f".model adc adc_bridge(in_low=0.5 in_high=0.5 rise_delay={bridge_delay!r} fall_delay={bridge_delay!r})",
            f".model dac dac_bridge(out_low=0 out_high=1 t_rise={bridge_delay!r} t_fall={bridge_delay!r})",
            "Aclk fclk dclk osc",
        ]
        # Example signal path for m=0, y0=x0 (model lines omitted):
        #   By0 y0 0 V = 1.0*v(x0) + 0.0
        #   Bcmp0 cmp0 0 V = v(y0) >= 0 ? 1 : 0
        #   Acmp0 [cmp0] [dcmp0] adc
        #   Aff0 dcmp0 dclk NULL NULL q0 NULL ff0
        #   Aout0 [q0] [qb0] dac
        #   Bv0 v0 0 V = -1 + 2*v(qb0)
        for m in range(self.M):
            # Comparator input: y_m = C_m x + D_m,u u + G_m eta + offset.
            y = [f"{float(c)!r}*v(x{n})" for n, c in enumerate(C[m]) if c]
            y += [
                f"{float(d)!r}*v(u{ell})" for ell, d in enumerate(D[m, : self.L]) if d
            ]
            if G is not None:
                y += [f"{float(a)!r}*v(eta{p})" for p, a in enumerate(G[m]) if a]
            y.append(repr(float(self.comparator_input_offset[m])))
            # Initialize the flip-flop from the noise-free comparator input.
            ic = int(
                float(C[m] @ x0 + D[m, : self.L] @ u0 + self.comparator_input_offset[m])
                >= 0
            )
            # Acmp digitizes the threshold result; Aff stores it on dclk.
            # Aout returns the bit as a voltage; Bv maps 0/1 to -1/+1.
            lines += [
                f"By{m} y{m} 0 V = {' + '.join(y)}",
                f"Bcmp{m} cmp{m} 0 V = v(y{m}) >= 0 ? 1 : 0",
                f"Acmp{m} [cmp{m}] [dcmp{m}] adc",
                f".model ff{m} d_dff(clk_delay={bridge_delay!r} set_delay={bridge_delay!r} reset_delay={bridge_delay!r} ic={ic} rise_delay={bridge_delay!r} fall_delay={bridge_delay!r})",
                f"Aff{m} dcmp{m} dclk NULL NULL q{m} NULL ff{m}",
                f"Aout{m} [q{m}] [qb{m}] dac",
                f"Bv{m} v{m} 0 V = -1 + 2*v(qb{m})",
            ]

            # A DAC window (delay, width) may cross period boundaries.
            # In period n, (lag, lo, hi) uses decision v[n-lag] on the
            # local interval [lo, hi), with all times measured in periods.
            # Example: (0.5, 1.0) -> (0, 0.5, 1), (1, 0, 0.5).
            delay, width = windows[m]
            active = [
                (lag, max(0.0, delay - lag), min(1.0, delay + width - lag))
                for lag in range(math.ceil(delay + width))
                if min(1.0, delay + width - lag) > max(0.0, delay - lag)
            ]
            # Shift-register flip-flops retain the older decisions; for lag=1:
            #   Aff0_1 q0 dclk NULL NULL q0_1 NULL ff0
            #   Aout0_1 [q0_1] [qb0_1] dac
            for lag in range(1, active[-1][0] + 1):
                # Each new tap samples the preceding tap on dclk.
                prior = f"q{m}" if lag == 1 else f"q{m}_{lag - 1}"
                lines += [
                    f"Aff{m}_{lag} {prior} dclk NULL NULL q{m}_{lag} NULL ff{m}",
                    f"Aout{m}_{lag} [q{m}_{lag}] [qb{m}_{lag}] dac",
                ]
            # Each overlap contributes a gated, bipolar decision to Bd.
            terms = []
            for lag, lo, hi in active:
                # qb is the current 0/1 bit; qb_lag is a retained older bit.
                decision = f"v(qb{m})" if lag == 0 else f"v(qb{m}_{lag})"
                # A full-period overlap needs no gate; a partial one gets Vgate.
                if lo == 0 and hi == 1:
                    gate = "1"
                else:
                    # PULSE starts rising at lo and falling at hi each period;
                    # its high-time parameter excludes the bridge rise time.
                    duration = (hi - lo) * self.dt - bridge_delay
                    if duration <= 0:
                        raise ValueError(
                            "DAC window is shorter than ngspice bridge delay"
                        )
                    lines.append(
                        f"Vgate{m}_{lag} gate{m}_{lag} 0 PULSE(0 1 {lo * self.dt!r} {bridge_delay!r} {bridge_delay!r} {duration!r} {self.dt!r})"
                    )
                    gate = f"v(gate{m}_{lag})"
                if lag:
                    # Vvalid blocks nonexistent prehistory until lag clock edges
                    # have filled the register and its bridge has settled.
                    ready = lag * self.dt
                    lines.append(
                        f"Vvalid{m}_{lag} valid{m}_{lag} 0 PWL(0 0 {ready + 4 * bridge_delay!r} 0 {ready + 8 * bridge_delay!r} 1)"
                    )
                    gate += f"*v(valid{m}_{lag})"
                # Map the 0/1 bit to -1/+1 only while its gate is active.
                terms.append(f"{gate}*(-1 + 2*{decision})")
            # Example full-period DAC: Bd0 d0 0 V = 1.0*(1*(-1 + 2*v(qb0)))
            # Bd sums the pulse pieces; B[:, L:] sets their frontend coupling.
            lines.append(
                f"Bd{m} d{m} 0 V = {pulse_amplitudes[m]!r}*({' + '.join(terms)})"
            )

        # Record state voltages x_n and bipolar controls v_m after time.
        vectors = [
            *(f"v(x{n})" for n in range(self.N)),
            *(f"v(v{m})" for m in range(self.M)),
        ]
        # Example footer for dt=1e-5, size=2, N=M=1:
        #   .ic v(x0)=0.0
        #   tran 1.25e-06 2e-05 0 7.8125e-08 uic
        #   wrdata tran.dat v(x0) v(v0)
        # .control starts the script; set requests ASCII, column names and one
        # time axis. Optional setseed fixes the noise sequence.
        # .ic sets x(0); uic skips the DC solve. tran suggests dt/8 for output
        # and caps solver steps at dt/128; wrdata saves the selected voltages.
        lines += [
            ".ic " + " ".join(f"v(x{n})={float(v)!r}" for n, v in enumerate(x0)),
            ".control",
            "set filetype=ascii",
            "set wr_vecnames",
            "set wr_singlescale",
            *([f"setseed {seed}"] if seed is not None else []),
            f"tran {self.dt / 8!r} {size * self.dt!r} 0 {self.dt / 128!r} uic",
            "wrdata tran.dat " + " ".join(vectors),
            "quit",
            ".endc",
            ".end",
        ]
        return "\n".join(lines) + "\n"


class ActiveRC(Frontend):
    """Realize each ideal integrator with a fixed resistor and actual capacitor.

    The first ``N`` states are the original output voltages ``x``; the next
    ``N`` are internal op-amp voltages ``p``. With
    ``Ctot = Cint + Cp``, ``R_int = 1/(omega_u*C_design)``,
    ``omega = 1/(R_int*Ctot)``, ``omega_t = gm/Co`` and ``A0 = gm*Ro``::

        dx/dt = omega*p - (omega/A0)*x
        dp/dt = omega_t*((A*x + B*[u; v])/omega_u - p)

    Each division is row-wise. ``C_design`` defaults to ``Cint`` and ``Cp``
    to zero, giving the original model. Fix ``C_design`` at nominal
    ``Cint + Cp`` while varying the actual components to model mismatch.
    The comparator and twin read the first ``N`` states. An already realized
    frontend cannot be wrapped again. ``Ro = inf`` removes the integrator
    leak while retaining the finite op-amp pole at ``-omega_t``.

    With thermal noise enabled, the physical integration resistor contributes
    an independent current with one-sided PSD ``4*k_B*T/R_int`` at each
    integrator capacitor. Optional input-referred op-amp voltage noise ``e_n``
    perturbs the internal pole's target; input current noise ``i_n`` injects
    current into the integration capacitor. Thus the extra state derivatives are
    ``dx/dt += (i_R + i_n)/Ctot`` and ``dp/dt += omega_t*e_n``. The densities
    are one-sided, while ``state_covariance`` stores two-sided intensity:
    ``Q_xx = (4*k_B*T/R_int + i_n**2)/(2*Ctot**2)`` and
    ``Q_pp = (omega_t*e_n)**2/2``. All sources are assumed independent.
    ``Ro`` sets finite gain in this macromodel; its equivalent ``Rleak`` is
    not an additional physical resistor noise source. An op-amp's ``e_n`` and
    ``i_n`` depend on its internal design and must come from characterization
    or a datasheet; ``gm``, ``Ro``, and ``Co`` cannot determine them [1]_.

    Parameters
    ----------
    af : Frontend
        Ideal continuous-time frontend, left unchanged.
    Cint, Co : float or array_like, shape (N,), (J, N), or (J, K, N)
        Actual integration and op-amp pole capacitances in farads.
    gm : float or array_like, shape (N,), (J, N), or (J, K, N)
        Op-amp transconductance in siemens.
    Ro : float or array_like, shape (N,), (J, N), or (J, K, N)
        Op-amp output resistance in ohms; ``inf`` gives infinite DC gain.
    omega_u : float or array_like, shape (N,), optional
        Integrator reference angular frequency in rad/s. Defaults to the
        largest absolute entry of each ideal ``[A | B]`` row over ``J, K``.
    v_n : float or array_like, shape (N,), optional
        One-sided state-referred noise density in V/sqrt(Hz); overrides the
        ideal frontend's state covariance when supplied. Defaults to ``None``.
        Set ``thermal_noise=False`` when using this legacy override.
    Cp : float or array_like, shape (N,), (J, N), or (J, K, N), optional
        Actual parasitic capacitance at each integrator node, in farads;
        defaults to zero.
    C_design : float or array_like, shape (N,), (J, N), or (J, K, N), optional
        Nominal total capacitance used to size ``R_int``, in farads. Defaults
        to ``Cint``.
    thermal_noise : bool, optional
        Add Johnson noise of the physical ``R_int`` to inherited state noise.
        Defaults to ``True``; pass ``False`` for a noiseless circuit model.
    temperature : float, optional
        Absolute temperature in kelvin; defaults to ``300``.
    opamp_voltage_noise : float or array_like, shape (N,), (J, N), or (J, K, N), optional
        White input-referred voltage density ``e_n`` in V/sqrt(Hz), applied
        before the op-amp pole. Defaults to zero. Supply a measured or
        datasheet value; this parameter also works without ``thermal_noise``.
    opamp_current_noise : float or array_like, shape (N,), (J, N), or (J, K, N), optional
        White current density ``i_n`` in A/sqrt(Hz), referred to the
        integrating input node. Defaults to zero. The macromodel has one
        equivalent input current source per stage, not separate currents at
        both op-amp terminals.
    comparator_input_noise_rms : float or array_like, shape (M,), optional
        Independent input-referred comparator noise in RMS volts per decision.
        Adds ``diag(comparator_input_noise_rms**2)`` to the inherited
        ``output_covariance``; defaults to ``None`` (no added noise). A white
        voltage density requires an equivalent noise bandwidth to convert
        it to RMS volts.

    Returns
    -------
    ActiveRC
        Independent ``2N``-state circuit realization.

    References
    ----------
    .. [1] Analog Devices, `MT-047: Op Amp Noise
       <https://www.analog.com/media/en/training-seminars/tutorials/MT-047.pdf>`_.
       Describes independent input voltage/current noise, their density units,
       and the separate Johnson noise of external resistors.
    """

    _netlist_title = "active-RC"

    def __init__(
        self,
        af: Frontend,
        Cint: ArrayLike,
        gm: ArrayLike,
        Ro: ArrayLike,
        Co: ArrayLike,
        omega_u: Optional[ArrayLike] = None,
        v_n: Optional[ArrayLike] = None,
        *,
        Cp: Optional[ArrayLike] = None,
        C_design: Optional[ArrayLike] = None,
        thermal_noise: bool = True,
        temperature: float = 300.0,
        opamp_voltage_noise: Optional[ArrayLike] = None,
        opamp_current_noise: Optional[ArrayLike] = None,
        comparator_input_noise_rms: Optional[ArrayLike] = None,
    ) -> None:
        if not isinstance(af, Frontend) or af.discrete:
            raise ValueError("ActiveRC needs a continuous-time Frontend")
        if getattr(af, "_realization", None) is not None:
            raise ValueError(
                "realize an ideal Frontend, not another circuit realization"
            )
        if v_n is not None and (
            thermal_noise
            or opamp_voltage_noise is not None
            or opamp_current_noise is not None
        ):
            raise ValueError("choose circuit noise parameters or v_n, not both")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")

        def positive(value: ArrayLike, name: str) -> np.ndarray:
            try:
                result = np.broadcast_to(np.asarray(value, dtype=float), (af.N,)).copy()
            except ValueError as exc:
                raise ValueError(f"{name} must broadcast to ({af.N},)") from exc
            if not np.all(np.isfinite(result)) or np.any(result <= 0):
                raise ValueError(f"{name} must be finite and positive")
            return result

        self.N_ideal = N = af.N
        self.Cint = _component(Cint, "Cint", N)
        self.Cp = _component(0.0 if Cp is None else Cp, "Cp", N)
        self.C_design = (
            self.Cint.copy()
            if C_design is None
            else _component(C_design, "C_design", N)
        )
        self.gm = _component(gm, "gm", N)
        self.Ro = _component(Ro, "Ro", N)
        self.Co = _component(Co, "Co", N)
        for name in ("Cint", "C_design", "gm", "Co"):
            value = getattr(self, name)
            if not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if np.any(np.isnan(self.Ro)) or np.any(self.Ro <= 0):
            raise ValueError("Ro must be positive or infinite")
        if not np.all(np.isfinite(self.Cp)) or np.any(self.Cp < 0):
            raise ValueError("Cp must be finite and nonnegative")
        self.opamp_voltage_noise = _component(
            0.0 if opamp_voltage_noise is None else opamp_voltage_noise,
            "opamp_voltage_noise",
            N,
        )
        self.opamp_current_noise = _component(
            0.0 if opamp_current_noise is None else opamp_current_noise,
            "opamp_current_noise",
            N,
        )
        for name in ("opamp_voltage_noise", "opamp_current_noise"):
            density = getattr(self, name)
            if not np.all(np.isfinite(density)) or np.any(density < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        rows = np.concatenate(
            (
                np.broadcast_to(
                    af.A[:, np.arange(af.K) % af.A.shape[1]],
                    (af.J_frontend, af.K, N, N),
                ),
                np.broadcast_to(
                    af.B[:, np.arange(af.K) % af.B.shape[1]],
                    (af.J_frontend, af.K, N, af.L + af.M),
                ),
            ),
            axis=-1,
        )
        default_omega = np.max(np.abs(rows), axis=(0, 1, 3))
        self.omega_u = positive(
            default_omega if omega_u is None else omega_u, "omega_u"
        )
        J, K = _component_axes(
            af,
            self.Cint,
            self.Cp,
            self.C_design,
            self.gm,
            self.Ro,
            self.Co,
            self.opamp_voltage_noise,
            self.opamp_current_noise,
        )
        self.Ctot = _phases(self.Cint, J, K, N) + _phases(self.Cp, J, K, N)
        if not np.all(np.isfinite(self.Ctot)) or np.any(self.Ctot <= 0):
            raise ValueError("Cint + Cp must be finite and positive")
        gm_actual = _phases(self.gm, J, K, N)
        Co_actual = _phases(self.Co, J, K, N)
        self.GBWP = gm_actual / (2 * np.pi * Co_actual)
        self.DC_gain = gm_actual * _phases(self.Ro, J, K, N)
        self.R_int = 1 / (self.omega_u * _phases(self.C_design, J, K, N))
        omega = 1 / (self.R_int * self.Ctot)
        omega_t = gm_actual / Co_actual
        self.thermal_noise = bool(thermal_noise)
        self.temperature = float(temperature)
        A = np.zeros((J, K, 2 * N, 2 * N))
        B = np.zeros((J, K, 2 * N, af.L + af.M))
        A[..., :N, :N] = -np.eye(N) * (omega / self.DC_gain)[..., :, None]
        A[..., :N, N:] = np.eye(N) * omega[..., :, None]
        gain = (omega_t / self.omega_u)[..., :, None]
        A[..., N:, :N] = gain * af.A[:, np.arange(K) % af.A.shape[1]]
        A[..., N:, N:] = -np.eye(N) * omega_t[..., :, None]
        B[..., N:, :] = gain * af.B[:, np.arange(K) % af.B.shape[1]]
        C = np.pad(af.C, ((0, 0), (0, 0), (0, 0), (0, N)))
        C_hat = (
            None
            if af.C_hat is None
            else np.pad(af.C_hat, ((0, 0), (0, 0), (0, 0), (0, N)))
        )
        super().__init__(
            A,
            B,
            C,
            af.D.copy(),
            dt=af.dt,
            levels=af.levels,
            input=deepcopy(af.input),
            dac=deepcopy(af.dac),
            C_hat=C_hat,
            D_hat=None if af.D_hat is None else af.D_hat.copy(),
        )
        self._realization = "ActiveRC"
        self.x0 = np.pad(af.x0, ((0, 0), (0, N)))
        self.state_covariance = (
            None
            if af.state_covariance is None
            else np.pad(af.state_covariance, ((0, 0), (0, 0), (0, N), (0, N)))
        )
        self.output_covariance = deepcopy(af.output_covariance)
        _add_comparator_noise(self, comparator_input_noise_rms)
        self.comparator_input_offset = af.comparator_input_offset.copy()
        self._flicker = deepcopy(af._flicker)
        for name, fill in self._per_state.items():
            setattr(self, name, np.r_[getattr(af, name), np.full(N, fill)])
        if v_n is not None:
            density = np.broadcast_to(np.asarray(v_n, dtype=float), (N,))
            if not np.all(np.isfinite(density)) or np.any(density < 0):
                raise ValueError("v_n must be finite and nonnegative")
            self.state_covariance = np.pad(
                per_state_intensity(density), ((0, N), (0, N))
            )
        if (
            thermal_noise
            or opamp_voltage_noise is not None
            or opamp_current_noise is not None
        ):
            en = _phases(self.opamp_voltage_noise, J, K, N)
            ina = _phases(self.opamp_current_noise, J, K, N)
            intensity = np.zeros((J, K, 2 * N, 2 * N))
            current_psd = ina**2
            if thermal_noise:
                current_psd = current_psd + 4 * k_B * temperature / self.R_int
            idx = np.arange(N)
            intensity[..., idx, idx] = current_psd / (2 * self.Ctot**2)
            intensity[..., N + idx, N + idx] = (omega_t * en) ** 2 / 2
            _add_state_intensity(self, intensity)

    def __getitem__(self, j: int) -> "ActiveRC":
        """Select batch member ``j`` and its matching circuit components.

        Parameters
        ----------
        j : int
            Batch index in ``[-J, J)``.

        Returns
        -------
        ActiveRC
            Independent circuit realization with ``J = 1``.
        """
        j = range(self.J)[operator.index(j)]
        new = super().__getitem__(j)
        for name in (
            "Cint",
            "Cp",
            "C_design",
            "Ctot",
            "gm",
            "Ro",
            "Co",
            "GBWP",
            "DC_gain",
            "R_int",
            "opamp_voltage_noise",
            "opamp_current_noise",
        ):
            value = getattr(new, name)
            setattr(new, name, value[[j % value.shape[0]]])
        return new

    def _state_netlist_lines(self, F: Optional[np.ndarray]) -> list[str]:
        """Emit an RC/VCCS macromodel of the augmented op-amp equations.

        Parameters
        ----------
        F : ndarray, shape (2N, 2N), or None
            State-noise factor; only its integrator rows are usually nonzero.

        Returns
        -------
        list[str]
            Ngspice component lines for ``x`` and ``p`` state voltages.
        """
        lines = []
        for n in range(self.N_ideal):
            p = self.N_ideal + n
            lines += [
                f"Cx{n} x{n} 0 {float(self.Ctot[0, 0, n])!r}",
                f"Gxp{n} 0 x{n} x{p} 0 {float(1 / self.R_int[0, 0, n])!r}",
                f"Cop{n} x{p} 0 {float(self.Co[0, 0, n])!r}",
                f"Rop{n} x{p} 0 {float(1 / self.gm[0, 0, n])!r}",
            ]
            if np.isfinite(self.Ro[0, 0, n]):
                lines.append(
                    f"Rleak{n} x{n} 0 {float(self.DC_gain[0, 0, n] * self.R_int[0, 0, n])!r}"
                )
            for j, a in enumerate(self.A[0, 0, p, : self.N_ideal]):
                g = self.Co[0, 0, n] * a
                if g:
                    lines.append(f"Gax{n}_{j} 0 x{p} x{j} 0 {float(g)!r}")
            for ell, b in enumerate(self.B[0, 0, p, : self.L]):
                g = self.Co[0, 0, n] * b
                if g:
                    lines.append(f"Gu{n}_{ell} 0 x{p} u{ell} 0 {float(g)!r}")
            for m, b in enumerate(self.B[0, 0, p, self.L :]):
                g = self.Co[0, 0, n] * b
                if g:
                    lines.append(f"Gd{n}_{m} 0 x{p} d{m} 0 {float(g)!r}")
        if F is not None:
            caps = np.r_[self.Ctot[0, 0], self.Co[0, 0]]
            for n, C in enumerate(caps):
                rhs = [f"{float(a)!r}*v(w{p})" for p, a in enumerate(F[n]) if a]
                if rhs:
                    lines.append(f"Bwn{n} 0 x{n} I = {float(C)!r}*({' + '.join(rhs)})")
        return lines

    def netlist(
        self, size: int, x0: Optional[ArrayLike] = None, seed: Optional[int] = None
    ) -> str:
        """Generate an ngspice validation macromodel with RC op-amp states.

        The common source, comparator and DAC rows are shared with
        :meth:`GmC.netlist`; capacitor, resistor and VCCS rows implement this
        class's ``2N`` state equations. This validates the macromodel, not a
        transistor-level op-amp.

        Parameters
        ----------
        size : int
            Number of decisions, including the initial decision.
        x0 : array_like, shape (2N,), optional
            Initial integrator and op-amp states; defaults to zero.
        seed : int, optional
            Ngspice noise seed; defaults to ngspice's seed.

        Returns
        -------
        str
            Complete ngspice netlist for the supported ``J = K = 1`` model.
        """
        return GmC.netlist(self, size, x0, seed)
