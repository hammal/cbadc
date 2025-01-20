"""The Schreier delta sigma toolbox wrapper

This module provides a wrapper for the Delta Sigma toolbox, a Python package for simulating and synthesizing delta-sigma modulators
which inturn is a port of the MATLAB Delta Sigma toolbox by Richard Schreier.
"""

import numpy as np
from scipy.signal import dlti, ZerosPolesGain, StateSpace
import deltasigma as ds
from typing import Optional, Union


def synthesizeNTF(
    order: int, OSR: int, opt: int, H_inf: float = 1.5, f0: float = 0.0, matlab=False
) -> ZerosPolesGain:
    """A wrapper for deltasigma.synthesizeNTF

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
    matlab : bool, optional
        Whether to use the MATLAB version of the function, by default False

    Returns
    -------
    scipy.signal.StateSpace
        The noise transfer function in state space form

    """
    if matlab:
        raise NotImplementedError
    else:
        zpk = ds.synthesizeNTF(order, OSR, opt, H_inf=H_inf, f0=f0)
    return ZerosPolesGain(*zpk)


def realizeNTF(
    ntf: ZerosPolesGain, form: str = "CRFB", stf: ZerosPolesGain = None, matlab=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wrapper for deltasigma.realizeNTF

    Parameters
    ----------
    ntf : scipy.signal.dlti
        The noise transfer function
    form : str
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
    """A wrapper for deltasigma.stuffABCD

    Parameters
    ----------
    a : np.ndarray
        Feedback/feedforward coefficients from/to the quantizer, size=order.
    g : np.ndarray
        Resonator coefficients, size=floor(order/2).
    b : np.ndarray
        Feed-in coefficients from the modulator input to each integrator, size=order + 1.
    c : np.ndarray
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
    return ds.stuffABCD(a, g, b, c)


def mapABCD(
    ABCD: np.ndarray, form: str = "CRFB"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wrapper for deltasigma.mapABCD

    Parameters
    ----------
    ABCD : np.ndarray
        A state-space description of the modulator loop filter.
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
    """A wrapper for deltasigma.partitionABCD

    Parameters
    ----------
    ABCD : np.ndarray
        The ABCD matrix to be partitioned
    m : int, optional
        The number of inputs in the system. It will be calculated from the ABCD matrix,
        if not provided.
    r : int, optional
        The number of outputs in the system. It will be calculated from ``m`` and the
        ABCD matrix, if not provided.

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
    """A wrapper for deltasigma.scaleABCD

    Parameters
    ----------
    ABCD : np.ndarray
        The ABCD matrix to be scaled
    nlev : int, optional
        The number of levels in the quantizer, by default 2.
    f : float, optional
        The normalized frequency of the test sinusoid, by default 0.0.
    xlim : Union[float, np.ndarray], optional
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
    """A wrapper for deltasigma.simulateDSM

    Parameters
    ----------
    u : np.ndarray
        The input vector to be used in the simulation. Multiple inputs are implied by the number of rows in ``u``.
    ABCD : the ABCD matrix describing the modulator or its NTF.`
    nlev : Union[int,np.ndarray], optional
        Number of levels in the quantizers. Set ``nlev`` to a scalar for a single quantizer modulator. Multiple quantizers
        are implied by making nlev an array, by default 2.
    x0 : Union[float,np.ndarray], optional
        The initial status of the modulator. If ``x0`` is set to float, its value will be used for all the states. If it is
        set to a sequence of floats, each of its values will be assigned to a state variable, by default 0.0.

    Returns
    -------
    v, xn, xmax, y : tuple of ndarrays
        The quantizer output, the modulator states, the maximum value that each state reached during simulation, and the output.
    """
    return ds.simulateDSM(u, ABCD, nlev, x0)


def realizeNTF_ct(
    ntf: ZerosPolesGain,
    form: str = "FB",
    tdac: Union[tuple[float, float], np.ndarray] = (0, 1),
    ordering=None,
    bp=None,
    ABCDc: Optional[np.ndarray] = None,
    method: str = "LOOP",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wrapper for deltasigma.realizeNTF

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

    tdac : sequence, optional
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

    ordering : sequence, optional
        A vector specifying which NTF zero-pair to use in each resonator
        Default is for the zero-pairs to be used in the order specified
        in the NTF.

    bp : sequence, optional
        A vector specifying which resonator sections are bandpass.
        The default (``zeros(...)``) is for all sections to be lowpass.

    ABCDc : :py:class"`numpy.ndarray`, optional
        The loop filter structure, in state-space form.
        If this argument is omitted, ABCDc is constructed according
        to "form."

    method : str, optional
        The default fitting method is ``'LOOP'``, which means that
        the DT and CT loop responses will be matched.
        Alternatively, it is possible to set the method to ``'NTF'``,
        which will result in the NTF responses to be matched.
        See :ref:`discrete-time-to-continuous-time-mapping` for a
        more in-depth discussion.

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
    raise NotImplementedError
    # return ds.clans(order, OSR, Q, rmax, opt)


def calculateTF(
    ABCD: np.ndarray, k: Optional[Union[float, np.ndarray]] = 1.0
) -> tuple[ZerosPolesGain, ZerosPolesGain]:
    """A wrapper for :py:func:`deltasigma.calculateTF`

    Parameters
    ----------
    ABCD : np.ndarray
        The ABCD matrix of the modulator
    k : Union[float,np.ndarray], optional
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
    """A wrapper for :py:func:`deltasigma.mapCtoD`

    Parameters
    ----------
    analog_filter : :py:class:`scipy.signal.StateSpace`
        The continuous-time filter to be mapped to discrete-time
    tdac : Optional[np.ndarray], optional
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
    f0 : float, optional
        The (normalized) frequency at which the Gp filters' gains are
         to be set to unity. Default 0 (DC).

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
    return ds.simulateSNR(ABCD, OSR, amp, f0, nlev, f, k, quadrature)


def plotPZ(
    ntf: ZerosPolesGain, color: str = "b", markersize: int = 5, showlist: bool = False
):
    """A wrapper for :py:func:`deltasigma.plotPZ`

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

    """
    ds.plotPZ((ntf.zeros, ntf.poles, ntf.gain), color, markersize, showlist)


def DocumentNTF(ntf: ZerosPolesGain, OSR: int, f0: float, quadrature: bool):
    """A wrapper for :py:func:`deltasigma.DocumentNTF`

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

    """
    ds.DocumentNTF((ntf.zeros, ntf.poles, ntf.gain), OSR, f0, quadrature)
