"""Delta-sigma modulator synthesis and simulation.

This module wraps ``cbadc-deltasigma`` (imported as ``cbadc_deltasigma``) and
bridges it into cbadc's state-space form. The functions reproduce the algorithms
and results of the delta-sigma toolbox described in R. Schreier and G. C. Temes,
*Understanding Delta-Sigma Data Converters* (IEEE Press / Wiley, 2005), and the
function names follow that book so the two can be read side by side. None of
the toolbox's own code is included here.
"""

from typing import Optional, Sequence, Union

try:
    import cbadc_deltasigma as ds
except ModuleNotFoundError as exc:
    if exc.name != "cbadc_deltasigma":
        raise
    raise ImportError(
        "cbadc.delsig requires cbadc-deltasigma; install the optional extra with "
        "`pip install 'cbadc[delsig]'` or `uv sync --extra delsig`."
    ) from exc
import numpy as np
from scipy.signal import StateSpace, ZerosPolesGain, dlti


def synthesizeNTF(
    order: int, OSR: int, opt: int, H_inf: float = 1.5, f0: float = 0.0, matlab=False
) -> ZerosPolesGain:
    """A wrapper for cbadc_deltasigma.synthesizeNTF

    Parameters
    ----------
    order : int
        The order of the modulator
    OSR : int
        The oversampling ratio
    opt : int
        Flag for optimized zeros

        * 0 -> not optimized,
        * 1 -> optimized,
        * 2 -> optimized with at least one zero at band-center,
        * [z] -> zero locations in complex form
    H_inf : float, optional
        Maximum out-of-band NTF magnitude.
        Defaults to ``1.5``.
    f0 : float, optional
        Normalized center frequency in cycles per sample.
        Defaults to ``0.0``.
    matlab : bool, optional
        Whether to use the MATLAB version of the function, by default False

    Returns
    -------
    scipy.signal.ZerosPolesGain
        The noise transfer function in zero-pole-gain form.

    """
    if matlab:
        raise NotImplementedError
    else:
        zpk = ds.synthesizeNTF(order, OSR, opt, H_inf=H_inf, f0=f0)
    return ZerosPolesGain(*zpk)


def realizeNTF(
    ntf: ZerosPolesGain, form: str = "CRFB", stf: ZerosPolesGain = None, matlab=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wrapper for cbadc_deltasigma.realizeNTF

    Parameters
    ----------
    ntf : scipy.signal.dlti
        The noise transfer function
    form : str, optional
        A structure identifier.

        Supported values:

        - CRFB: Cascade of resonators, feedback form
        - CRFF: Cascade of resonators, feedforward form.
        - CIFB: Cascade of integrators, feedback form.
        - CIFF: Cascade of integrators, feedforward form.
        - CRFBD: CRFB with delaying quantizer.
        - CRFFD: CRFF with delaying quantizer.
        - PFF: Parallel feed-forward.
        - Stratos: A CIFF-like structure with non-delaying resonator feedbacks,
            contributed to the MATLAB Delta Sigma toolbox in 2007 by Jeff Gealow.

        Defaults to ``"CRFB"``.
    stf : scipy.signal.ZerosPolesGain, optional
        Signal transfer function used when realizing the NTF.
        Defaults to ``None``.
    matlab : bool
        Request the unavailable MATLAB implementation; defaults to false.

    Returns
    -------
    a, g, b, c : tuple of ndarrays
        the coefficients for the desired structure

    """
    if isinstance(ntf, ZerosPolesGain):
        ntf = ntf.zeros, ntf.poles, ntf.gain
    if matlab:
        raise NotImplementedError
    else:
        a, g, b, c = ds.realizeNTF(ntf, form, stf)
    return a, g, b, c


def stuffABCD(
    a: np.ndarray, g: np.ndarray, b: np.ndarray, c: np.ndarray, form: str = "CRFB"
) -> np.ndarray:
    """A wrapper for cbadc_deltasigma.stuffABCD

    Parameters
    ----------
    a : ndarray, shape (order,)
        Feedback/feedforward coefficients from/to the quantizer, size=order.
    g : ndarray, shape (floor(order / 2),)
        Resonator coefficients, size=floor(order/2).
    b : ndarray, shape (order + 1,)
        Feed-in coefficients from the modulator input to each integrator, size=order + 1.
    c : ndarray, shape (order,)
        Integrator inter-stage coefficients, size=order.
    form : str, optional
        The structure identifier, by default "CRFB"

        Supported values:

        - CRFB: Cascade of resonators, feedback form
        - CRFF: Cascade of resonators, feedforward form.
        - CIFB: Cascade of integrators, feedback form.
        - CIFF: Cascade of integrators, feedforward form.
        - CRFBD: CRFB with delaying quantizer.
        - CRFFD: CRFF with delaying quantizer.
        - PFF: Parallel feed-forward.
        - Stratos: A CIFF-like structure with non-delaying resonator feedbacks,
            contributed to the MATLAB Delta Sigma toolbox in 2007 by Jeff Gealow.

    Returns
    -------
    ABCD: np.ndarray
        The [[A, B];[C, D]] matrix representation of the modulator

    """
    return ds.stuffABCD(a, g, b, c, form=form)


def mapABCD(
    ABCD: np.ndarray, form: str = "CRFB"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wrapper for cbadc_deltasigma.mapABCD

    Parameters
    ----------
    ABCD : ndarray, shape (N + M, N + L + M)
        State-space block matrix of the modulator loop filter.
    form : str, optional
        The structure identifier, by default "CRFB"

        Supported values:

        - CRFB: Cascade of resonators, feedback form
        - CRFF: Cascade of resonators, feedforward form.
        - CIFB: Cascade of integrators, feedback form.
        - CIFF: Cascade of integrators, feedforward form.
        - CRFBD: CRFB with delaying quantizer.
        - CRFFD: CRFF with delaying quantizer.
        - PFF: Parallel feed-forward.
        - Stratos: A CIFF-like structure with non-delaying resonator feedbacks,
            contributed to the MATLAB Delta Sigma toolbox in 2007 by Jeff Gealow.

    Returns
    -------
    a, g, b, c : tuple of ndarrays
        the coefficients for the desired structure

    """
    return ds.mapABCD(ABCD, form)


def partitionABCD(
    ABCD: np.ndarray, m: int = None, r: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wrapper for cbadc_deltasigma.partitionABCD

    Parameters
    ----------
    ABCD : ndarray, shape (N + r, N + m)
        State-space block matrix to partition.
    m : int, optional
        The number of inputs in the system. It will be calculated from the ABCD matrix,
        if not provided.
        Defaults to ``None``.
    r : int, optional
        The number of outputs in the system. It will be calculated from ``m`` and the
        ABCD matrix, if not provided.
        Defaults to ``None``.

    Returns
    -------
    A, B, C, D : tuple of ndarrays
        The partitioned matrices.

    """
    return ds.partitionABCD(ABCD, m, r)


def scaleABCD(
    ABCD: np.ndarray,
    nlev: int = 2,
    f: float = 0.0,
    xlim: Union[float, np.ndarray] = 1.0,
    ymax: float = None,
    umax: float = None,
    N_sim: int = 10000,
    N0: int = 10,
) -> tuple[np.ndarray, float, np.ndarray]:
    """A wrapper for cbadc_deltasigma.scaleABCD

    Parameters
    ----------
    ABCD : ndarray, shape (N + M, N + L + M)
        State-space block matrix to scale.
    nlev : int, optional
        The number of levels in the quantizer, by default 2.
    f : float, optional
        The normalized frequency of the test sinusoid, by default 0.0.
    xlim : float or ndarray, shape (N,), optional
        The limit for each or all state variable, by default 1.
    ymax : float, optional
        The stability threshold, by default None which corresponds to nlev + 5.
    umax : float, optional
        The maximum allowable input amplitude, by default None in which case it is calculated.
    N_sim : int, optional
        The number of simulation steps, by default 10000.
    N0 : int, optional
        The number of initial transient steps to be ignored, by default 10.

    Returns
    -------
    ABCDs, umax, S : tuple of ndarrays
        The scaled ABCD matrix, the maximum stable input amplitude, and the scaling matrix S.

    """
    return ds.scaleABCD(ABCD, nlev, f, xlim, ymax, umax, N_sim, N0)


def simulateDSM(
    u: np.ndarray,
    ABCD: np.ndarray,
    nlev: Union[int, np.ndarray] = 2,
    x0: Union[float, np.ndarray] = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wrapper for cbadc_deltasigma.simulateDSM

    Parameters
    ----------
    u : ndarray, shape (size,) or (L, size)
        The input vector to be used in the simulation. Multiple inputs are implied by the number of rows in ``u``.
    ABCD : ndarray, shape (N + M, N + L + M), or ZerosPolesGain
        Block matrix describing the modulator, or its NTF.
    nlev : int or ndarray, shape (M,), optional
        Number of levels in the quantizers. Set ``nlev`` to a scalar for a single quantizer modulator. Multiple quantizers
        are implied by making nlev an array, by default 2.
    x0 : float or ndarray, shape (N,), optional
        The initial status of the modulator. If ``x0`` is set to float, its value will be used for all the states. If it is
        set to a sequence of floats, each of its values will be assigned to a state variable, by default 0.0.

    Returns
    -------
    v, xn, xmax, y : tuple of ndarrays
        The quantizer output, the modulator states, the maximum value that each state reached during simulation, and the output.
    """
    if isinstance(ABCD, ZerosPolesGain):
        a, g, b, c = realizeNTF(ABCD)
        ABCD = stuffABCD(a, g, b, c)
    return ds.simulateDSM(u, ABCD, nlev, x0)


def realizeNTF_ct(
    ntf: ZerosPolesGain,
    form: str = "FB",
    tdac: Union[tuple[float, float], np.ndarray] = (0, 1),
    ordering=None,
    bp=None,
    ABCDc: Optional[np.ndarray] = None,
    method: str = "LOOP",
) -> tuple[np.ndarray, np.ndarray]:
    """A wrapper for cbadc_deltasigma.realizeNTF_ct

    Parameters
    ----------
    ntf : :py:class:`scipy.signal.dlti`
        A noise transfer function (NTF).

    form : str, optional
        A string specifying the topology of the loop filter.

         - 'FB': Feedback form,
         - 'FF': Feedforward form

        For the FB structure, the elements of `Bc` are calculated
        so that the sampled pulse response matches the L1 impulse
        response.  For the FF structure, `Cc` is calculated.
        Defaults to ``"FB"``.

    tdac : sequence of pairs of float, optional
        The timing for the feedback DAC(s). If ``tdac[0] >= 1``,
        direct feedback terms are added to the quantizer.

        Multiple timings (one or more per integrator) for the FB
        topology can be specified by making tdac a list of lists,
        e.g. ``tdac = [[1, 2], [1, 2], [[0.5, 1], [1, 1.5]], []]``

        In this example, the first two integrators have
        DACs with ``[1, 2]`` timing, the third has a pair of
        DACs, one with ``[0.5, 1]`` timing and the other with
        ``[1, 1.5]`` timing, and there is no direct feedback
        DAC to the quantizer.
        Defaults to ``(0, 1)``.

    ordering : sequence of int, shape (resonators,), optional
        A vector specifying which NTF zero-pair to use in each resonator
        Default is for the zero-pairs to be used in the order specified
        in the NTF.

    bp : sequence of bool, shape (resonators,), optional
        A vector specifying which resonator sections are bandpass.
        The default (``zeros(...)``) is for all sections to be lowpass.

    ABCDc : ndarray, shape (N + M, N + L + M), optional
        The loop filter structure, in state-space form.
        If this argument is omitted, ABCDc is constructed according
        to "form."
        Defaults to ``None``.

    method : str, optional
        The default fitting method is ``'LOOP'``, which means that
        the DT and CT loop responses will be matched.
        Alternatively, it is possible to set the method to ``'NTF'``,
        which will result in the NTF responses to be matched.

    Returns
    -------
    ABCDc : :py:class:`numpy.ndarray`
        A state-space description of the CT loop filter

    tdac2 : :py:class:`numpy.ndarray`
        A matrix with the DAC timings, including ones
        that were automatically added.

    """
    if isinstance(ntf, ZerosPolesGain):
        ntf = ntf.zeros, ntf.poles, ntf.gain
    ABCDc, tdac2 = ds.realizeNTF_ct(ntf, form, tdac, ordering, bp, ABCDc, method)
    return ABCDc, tdac2


def clans(order: int = 4, OSR: int = 64, Q: int = 5, rmax: float = 0.95, opt: int = 0):
    """Unavailable CLANS design wrapper.

    Parameters
    ----------
    order : int, optional
        Modulator order.
        Defaults to ``4``.
    OSR : int, optional
        Oversampling ratio.
        Defaults to ``64``.
    Q : int, optional
        Quantizer resolution parameter.
        Defaults to ``5``.
    rmax : float, optional
        Maximum pole radius.
        Defaults to ``0.95``.
    opt : int, optional
        Zero-placement option.
        Defaults to ``0``.

    Returns
    -------
    NoReturn
        This unavailable wrapper always raises ``NotImplementedError``.
    """
    raise NotImplementedError
    # return ds.clans(order, OSR, Q, rmax, opt)


def calculateTF(
    ABCD: np.ndarray, k: Optional[Union[float, np.ndarray]] = 1.0
) -> tuple[ZerosPolesGain, ZerosPolesGain]:
    """A wrapper for :py:func:`cbadc_deltasigma.calculateTF`

    Parameters
    ----------
    ABCD : ndarray, shape (N + M, N + L + M)
        The ABCD matrix of the modulator
    k : float or ndarray, shape (M,), optional
        The quantizer gains. If only one quantizer is present, it may be set
        to a float, corresponding to the quantizer gain. If multiple quantizers
        are present, a list should be used, with quantizer gains ordered
        according to the order in which the quantizer inputs appear in the C
        and D submatrices. If not specified, a default of one quantizer with
        gain 1. is assumed.

    Returns
    -------
    ntf, stf :py:class:`scipy.signal.ZerosPolesGain`
        The noise transfer function and the signal transfer function
        If the system has multiple quantizers, multiple STFs and NTFs will be returned.

        In that case:

        * STF[i] is the STF from u to output number i.

        * NTF[i, j] is the NTF from the quantization noise of the quantizer number j to output number i.
    """
    ntf, stf = ds.calculateTF(ABCD, k)
    return ZerosPolesGain(*ntf), ZerosPolesGain(*stf)


def mapCtoD(
    analog_filter: StateSpace, tdac: Optional[np.ndarray] = None, f0: float = 0.0
) -> tuple[StateSpace, dlti]:
    """A wrapper for :py:func:`cbadc_deltasigma.mapCtoD`

    Parameters
    ----------
    analog_filter : :py:class:`scipy.signal.StateSpace`
        The continuous-time filter to be mapped to discrete-time
    tdac : ndarray, shape (L, 2), optional
        The timing for the feedback DAC(s). If ``tdac[0] >= 1``,
        direct feedback terms are added to the quantizer.

        Multiple timings (one or more per integrator) for the FB
        topology can be specified by making tdac a list of lists,
        e.g. ``tdac = [[1, 2], [1, 2], [[0.5, 1], [1, 1.5]], []]``

        In this example, the first two integrators have
        DACs with ``[1, 2]`` timing, the third has a pair of
        DACs, one with ``[0.5, 1]`` timing and the other with
        ``[1, 1.5]`` timing, and there is no direct feedback
        DAC to the quantizer.
        Defaults to ``None``.
    f0 : float, optional
        The (normalized) frequency at which the Gp filters' gains are
         to be set to unity. Default 0 (DC).

    Returns
    -------
    system : scipy.signal.StateSpace
        Sampled equivalent state-space system.
    pre_filters : scipy.signal.dlti
        Discrete prefilters for the analog inputs.
    """

    if tdac is None:
        tdac = np.zeros((analog_filter.L, 2), dtype=float)
        # convention for continuous-time system.
        tdac[0, 0], tdac[1, 1] = -1.0, -1.0
        tdac[1:, 1] = 1.0
        tdac = np.array([0, 1])
    sys, pre_filters = ds.mapCtoD(analog_filter, tdac, f0)
    return StateSpace(*sys), dlti(*pre_filters)


def simulateSNR(
    ABCD: np.ndarray,
    OSR: int,
    amp: Optional[np.ndarray],
    f0: float = 0,
    nlev: int = 2,
    f: Optional[float] = None,
    k: int = 13,
    quadrature: bool = False,
):
    """Simulate the toolbox SNR curve over input amplitudes.

    Parameters
    ----------
    ABCD : ndarray, shape (N + M, N + L + M)
        Modulator state-space block matrix.
    OSR : int
        Oversampling ratio.
    amp : array_like of float, shape (amplitudes,), optional
        Input amplitudes in dBFS.
    f0 : float, optional
        Normalized center frequency in cycles per sample.
        Defaults to ``0``.
    nlev : int, optional
        Number of quantizer levels.
        Defaults to ``2``.
    f : float, optional
        Normalized test-tone frequency in cycles per sample.
        Defaults to ``None``.
    k : int, optional
        Log2 of the simulation record length.
        Defaults to ``13``.
    quadrature : bool, optional
        Use quadrature modulation.
        Defaults to ``False``.

    Returns
    -------
    snr : ndarray of float, shape (amplitudes,)
        Simulated SNR values in dB.
    amp : ndarray of float, shape (amplitudes,)
        Tested amplitudes in dBFS.
    """
    return ds.simulateSNR(ABCD, OSR, amp, f0, nlev, f, k, quadrature)


def plotPZ(
    ntf: ZerosPolesGain, color: str = "b", markersize: int = 5, showlist: bool = False
):
    """A wrapper for :py:func:`cbadc_deltasigma.plotPZ`

    Parameters
    ----------
    ntf : :py:class:`scipy.signal.ZerosPolesGain`
        The noise transfer function to plot
    color : str, optional
        The color of the plot, by default 'b'
    markersize : int, optional
        The size of the markers, by default 5
    showlist : bool, optional
        Whether to show the list of poles and zeros, by default False

    Returns
    -------
    None
        Draws the pole-zero plot using the delta-sigma toolbox.
    """
    ds.plotPZ((ntf.zeros, ntf.poles, ntf.gain), color, markersize, showlist)


def DocumentNTF(ntf: ZerosPolesGain, OSR: int, f0: float, quadrature: bool):
    """A wrapper for :py:func:`cbadc_deltasigma.DocumentNTF`

    Parameters
    ----------
    ntf : :py:class:`scipy.signal.ZerosPolesGain`
        The noise transfer function to plot
    OSR : int
        The oversampling ratio
    f0 : float
        The baseband frequency
    quadrature : bool
        Whether the modulator is quadrature or not

    Returns
    -------
    None
        Displays the toolbox NTF report.
    """
    ds.DocumentNTF((ntf.zeros, ntf.poles, ntf.gain), OSR, f0, quadrature)


def changeFig(
    fontsize: int = 12,
    linewidth: int = 1,
    markersize: int = 6,
    xfticks: Optional[str] = None,
    yfticks: Optional[str] = None,
    bw: bool = False,
    fig=None,
):
    """A wrapper for :py:func:`cbadc_deltasigma.changeFig`

    Parameters
    ----------
    fontsize : int, optional
        The font size of the labels, by default 12
    linewidth : int, optional
        The width of the lines, by default 1
    markersize : int, optional
        The size of the markers, by default 6
    xfticks : Optional[str], optional
        The format of the x-axis ticks, by default None
    yfticks : Optional[str], optional
        The format of the y-axis ticks, by default None
    bw : bool, optional
        Whether to use black and white colors, by default False
    fig : matplotlib.figure.Figure, optional
        The figure to change, by default None

    Returns
    -------
    None
        Modifies the selected Matplotlib figure in place.
    """
    ds.changeFig(fontsize, linewidth, markersize, xfticks, yfticks, bw, fig)


def PlotExampleSpectrum(
    ntf: ZerosPolesGain,
    M: int = 1,
    OSR: int = 64,
    f0: float = 0.0,
    quadrature: bool = False,
):
    """A wrapper for :py:func:`cbadc_deltasigma.PlotExampleSpectrum`

    Parameters
    ----------
    ntf : :py:class:`scipy.signal.ZerosPolesGain`
        The noise transfer function to plot
    M : int, optional
        The number of tones to simulate
        Defaults to ``1``.
    OSR : int, optional
        The oversampling ratio
        Defaults to ``64``.
    f0 : float, optional
        The baseband frequency
        Defaults to ``0.0``.
    quadrature : bool, optional
        Whether the modulator is quadrature or not
        Defaults to ``False``.

    Returns
    -------
    None
        Draws the toolbox example spectrum.
    """
    ds.PlotExampleSpectrum((ntf.zeros, ntf.poles, ntf.gain), M, OSR, f0, quadrature)


def is_stable_dsm(
    ntf: ZerosPolesGain,
    nlev: int = 2,
    amplitudes: Sequence[float] = (0.4, 0.6, 0.8),
    n_sim: int = 1 << 12,
    f_frac: float = 0.1,
    overload: Optional[float] = None,
) -> bool:
    """Large-signal stability check for a single-loop :math:`\\Sigma\\Delta`.

    A single-bit (and, less often, multi-bit) modulator whose NTF has too high an
    out-of-band gain *overloads*: the loop states run away rather than settling
    into a bounded limit cycle. ``synthesizeNTF`` is unconstrained in this sense
    -- a high-order, high-``H_inf`` NTF is perfectly valid as a discrete-time
    filter yet drives an unstable modulator -- so a design must be screened before
    it is trusted.

    The screen drives the modulator with a near-full-scale sine at a few
    amplitudes (a modulator can be small-signal stable but large-signal unstable)
    and flags it unstable if any state's peak magnitude (``xmax`` from
    :func:`simulateDSM`) exceeds an overload threshold. The default threshold
    follows the Schreier toolbox convention ``nlev + 5`` -- a clear blow-up,
    not a marginal excursion.

    Parameters
    ----------
    ntf : :py:class:`scipy.signal.ZerosPolesGain`
        the noise transfer function, e.g. from :func:`synthesizeNTF`.
    nlev : int, optional
        the number of quantizer levels, defaults to 2.
    amplitudes : sequence of float, optional
        input amplitudes as a fraction of full scale ``A_FS = nlev - 1`` to test;
        defaults to ``(0.4, 0.6, 0.8)``.
    n_sim : int, optional
        the number of samples per test simulation, defaults to ``1 << 12``.
    f_frac : float, optional
        the test-tone frequency as a fraction of the simulation Nyquist (kept
        well inside any plausible signal band), defaults to 0.1.
    overload : float, optional
        the per-state overload threshold; defaults to ``nlev + 5``.

    Returns
    -------
    bool
        ``True`` if every tested amplitude stays bounded, ``False`` if any
        overloads (or the simulation returns a non-finite state).
    """
    overload = float(nlev + 5) if overload is None else float(overload)
    a_fs = float(max(nlev - 1, 1))
    n = np.arange(n_sim)
    tone = np.sin(2.0 * np.pi * f_frac * 0.5 * n)
    for amp in amplitudes:
        _, _, xmax, _ = simulateDSM(amp * a_fs * tone, ntf, nlev=nlev)
        xmax = np.asarray(xmax, dtype=float)
        if not np.all(np.isfinite(xmax)) or float(np.max(xmax)) > overload:
            return False
    return True


def max_stable_amplitude(
    frontend,
    DSR: int,
    A_grid: Sequence[float] = tuple(round(0.1 * i, 1) for i in range(1, 41)),
    sim_size: int = 1 << 12,
    K: int = 1 << 6,
    overload_factor: float = 5.0,
    seed: int = 90128310230123,
    refine: bool = True,
) -> float:
    """Maximum stable wideband input amplitude of a frontend, as a fraction of full scale.

    Drives a copy of ``frontend`` with a uniform reference held over ``DSR``
    periods at each amplitude of ``A_grid`` and declares overload when a state
    exceeds ``overload_factor`` (full scale is 1) or is not finite. A single-loop
    modulator overloads below ``1``; a control-bounded converter has headroom
    above it. With ``refine`` the last stable grid point is bisected once.

    Parameters
    ----------
    frontend : cbadc.frontend.Frontend
        Converter to test using a repeating wideband input.
    DSR : int
        Number of clock periods for each held reference sample.
    A_grid : sequence of float, shape (amplitudes,), optional
        Full-scale amplitude candidates, examined in ascending order.
        Defaults to 0.1 through 4.0 in steps of 0.1.
    sim_size : int, optional
        Number of simulated clock periods per candidate.
        Defaults to ``1 << 12``.
    K : int, optional
        Extra reference samples at each record end.
        Defaults to ``1 << 6``.
    overload_factor : float, optional
        State magnitude threshold for overload.
        Defaults to ``5.0``.
    seed : int, optional
        Random seed shared across amplitude candidates.
        Defaults to ``90128310230123``.
    refine : bool, optional
        Bisect once between the last stable and first unstable grid points.
        Defaults to ``True``.

    Returns
    -------
    float
        Largest tested stable input amplitude as a fraction of full scale.
    """
    from copy import copy

    from .frontend import Samples

    ref_seed = int(np.random.default_rng(seed).integers(0, 1 << 62))

    def _is_stable(A: float) -> bool:
        """Check one wideband input amplitude for bounded states.

        Parameters
        ----------
        A : float
            Nonnegative amplitude as a fraction of full scale.

        Returns
        -------
        bool
            Whether the simulated states remain finite and below the threshold.
        """
        rng = np.random.default_rng(ref_seed)
        values = rng.uniform(-A, A, size=(sim_size // DSR + 2 * K, 1))
        af = copy(frontend)
        af.input = Samples(np.repeat(values, DSR, axis=0))
        peak = float(np.max(np.abs(af.simulate(sim_size, record="x")["x"])))
        return np.isfinite(peak) and peak <= overload_factor

    grid = sorted(float(a) for a in A_grid)
    msa, first_unstable = grid[0], None
    for A in grid:
        if _is_stable(A):
            msa = A
        else:
            first_unstable = A
            break
    if refine and first_unstable is not None and first_unstable > msa:
        mid = 0.5 * (msa + first_unstable)
        if _is_stable(mid):
            msa = mid
    return float(msa)
