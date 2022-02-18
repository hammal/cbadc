"""A selection of standard filters expressed as analog systems."""
import scipy.signal
import logging
from .analog_system import AnalogSystem
from .topology import zpk2abcd

logger = logging.getLogger(__name__)


class ButterWorth(AnalogSystem):
    """A Butterworth filter analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and creates a convenient
    way of creating Butterworth filter analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirfilter`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    N: `int`
        filter order
    Wn: `float`
        array containing the critical frequency of the filter.
        The frequency is specified in rad/s.


    Attributes
    ----------
    N : `int`
        state space order :math:`N`.
    N_tilde : `int`
        number of signal observations :math:`\\tilde{N}`.
    M : `int`
        number of digital control signals :math:`M`.
    M_tilde : `int`
        number of control signal observations :math:`\\tilde{M}`.
    L : `int`
        number of input signals :math:`L`.
    A : `array_like`, shape=(N, N)
        system matrix :math:`\mathbf{A}`.
    B : `array_like`, shape=(N, L)
        input matrix :math:`\mathbf{B}`.
    CT : `array_like`, shape=(N_tilde, N)
        signal observation matrix :math:`\mathbf{C}^\mathsf{T}`.
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    Wn : float
        critical frequency of filter.

    See also
    --------
    :py:class:`cbadc.analog_system.ChebyshevI`
    :py:class:`cbadc.analog_system.ChebyshevII`
    :py:class:`cbadc.analog_system.Cauer`



    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, N: int, Wn: float):
        """Create a Butterworth filter"""
        # State space order
        self.Wn = Wn

        # Create filter as chain of biquadratic filters
        z, p, k = scipy.signal.iirfilter(
            N, Wn, analog=True, btype="lowpass", ftype="butter", output="zpk"
        )

        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


class ChebyshevI(AnalogSystem):
    """A Chebyshev type I filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Chebyshev type I filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirfilter`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    N: `int`
        filter order
    Wn: (float, float)
        array containing the critical frequencies (low, high) of the filter.
        The frequencies are specified in rad/s.
    rp: float
        maximum ripple in passband. Specified in dB.


    Attributes
    ----------
    N : `int`
        state space order :math:`N`.
    N_tilde : `int`
        number of signal observations :math:`\\tilde{N}`.
    M : `int`
        number of digital control signals :math:`M`.
    M_tilde : `int`
        number of control signal observations :math:`\\tilde{M}`.
    L : `int`
        number of input signals :math:`L`.
    A : `array_like`, shape=(N, N)
        system matrix :math:`\mathbf{A}`.
    B : `array_like`, shape=(N, L)
        input matrix :math:`\mathbf{B}`.
    CT : `array_like`, shape=(N_tilde, N)
        signal observation matrix :math:`\mathbf{C}^\mathsf{T}`.
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    Wn : float
        critical frequency of filter.

    See also
    --------
    :py:class:`cbadc.analog_system.ButterWorth`
    :py:class:`cbadc.analog_system.ChebyshevII`
    :py:class:`cbadc.analog_system.Cauer`



    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, N: int, Wn: float, rp: float):
        """Create a Chebyshev type I filter"""
        # State space order
        self.Wn = Wn
        self.rp = rp

        z, p, k = scipy.signal.iirfilter(
            N, Wn, rp, analog=True, btype="lowpass", ftype="cheby1", output="zpk"
        )
        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


class ChebyshevII(AnalogSystem):
    """A Chebyshev type II filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Chebyshev type II filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirfilter`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    N: `int`
        filter order
    Wn: (float, float)
        array containing the critical frequencies (low, high) of the filter.
        The frequencies are specified in rad/s.
    rs: float
        minimum attenutation in stopband. Specified in dB.


    Attributes
    ----------
    N : `int`
        state space order :math:`N`.
    N_tilde : `int`
        number of signal observations :math:`\\tilde{N}`.
    M : `int`
        number of digital control signals :math:`M`.
    M_tilde : `int`
        number of control signal observations :math:`\\tilde{M}`.
    L : `int`
        number of input signals :math:`L`.
    A : `array_like`, shape=(N, N)
        system matrix :math:`\mathbf{A}`.
    B : `array_like`, shape=(N, L)
        input matrix :math:`\mathbf{B}`.
    CT : `array_like`, shape=(N_tilde, N)
        signal observation matrix :math:`\mathbf{C}^\mathsf{T}`.
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    Wn : float
        critical frequency of filter.
    rs: float
        minimum attenuation in stop band (dB).

    See also
    --------
    :py:class:`cbadc.analog_system.ButterWorth`
    :py:class:`cbadc.analog_system.ChebyshevI`
    :py:class:`cbadc.analog_system.Cauer`



    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, N: int, Wn: float, rs: float):
        """Create a Chebyshev type II filter"""
        # State space order
        self.Wn = Wn
        self.rs = rs
        z, p, k = scipy.signal.iirfilter(
            N, Wn, rs=rs, analog=True, btype="lowpass", ftype="cheby2", output="zpk"
        )
        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


class Cauer(AnalogSystem):
    """A Cauer (elliptic) filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Cauer filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirfilter`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    N: `int`
        filter order
    Wn: (float, float)
        array containing the critical frequencies (low, high) of the filter.
        The frequencies are specified in rad/s.
    rp: float
        maximum ripple in passband. Specified in dB.
    rs: float
        minimum attenutation in stopband. Specified in dB.


    Attributes
    ----------
    N : `int`
        state space order :math:`N`.
    N_tilde : `int`
        number of signal observations :math:`\\tilde{N}`.
    M : `int`
        number of digital control signals :math:`M`.
    M_tilde : `int`
        number of control signal observations :math:`\\tilde{M}`.
    L : `int`
        number of input signals :math:`L`.
    A : `array_like`, shape=(N, N)
        system matrix :math:`\mathbf{A}`.
    B : `array_like`, shape=(N, L)
        input matrix :math:`\mathbf{B}`.
    CT : `array_like`, shape=(N_tilde, N)
        signal observation matrix :math:`\mathbf{C}^\mathsf{T}`.
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    Wn : float
        critical frequency of filter.
    rp: float
        maximal ripple in passband, specified in  (dB).
    rs: float
        minimum attenuation in stop band (dB).

    See also
    --------
    :py:class:`cbadc.analog_system.ButterWorth`
    :py:class:`cbadc.analog_system.ChebyshevI`
    :py:class:`cbadc.analog_system.ChebyshevII`



    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, N: int, Wn: float, rp: float, rs: float):
        """Create a Cauer filter"""
        # State space order
        self.Wn = Wn
        self.rp = rp
        self.rs = rs

        z, p, k = scipy.signal.iirfilter(
            N, Wn, rp, rs, analog=True, btype="lowpass", ftype="ellip", output="zpk"
        )
        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


class IIRDesign(AnalogSystem):
    """An analog signal designed using standard IIRDesign tools

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating IIR filters in an analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirdesign`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    wp, ws: `float or array_like`, shape=(2,)
        Passband and stopband edge frequencies. Possible values are scalars (for lowpass and highpass filters) or ranges (for bandpass and bandstop filters). For digital filters, these are in the same units as fs. By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency. For example:

        * Lowpass: wp = 0.2, ws = 0.3
        * Highpass: wp = 0.3, ws = 0.2
        * Bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]
        * Bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]

        wp and ws are angular frequencies (e.g., rad/s). Note, that for bandpass and bandstop filters passband must lie strictly inside stopband or vice versa.

    gpass: `float`
        The maximum loss in the passband (dB).

    gstop: `float`
        The minimum attenuation in the stopband (dB).

    ftype: `string`, `optional`
        IIR filter type, defaults to ellip. Complete list of choices:

        * Butterworth : ‘butter’
        * Chebyshev I : ‘cheby1’
        * Chebyshev II : ‘cheby2’
        * Cauer/elliptic: ‘ellip’
        * Bessel/Thomson: ‘bessel’


    Attributes
    ----------
    N : `int`
        state space order :math:`N`.
    N_tilde : `int`
        number of signal observations :math:`\\tilde{N}`.
    M : `int`
        number of digital control signals :math:`M`.
    M_tilde : `int`
        number of control signal observations :math:`\\tilde{M}`.
    L : `int`
        number of input signals :math:`L`.
    A : `array_like`, shape=(N, N)
        system matrix :math:`\mathbf{A}`.
    B : `array_like`, shape=(N, L)
        input matrix :math:`\mathbf{B}`.
    CT : `array_like`, shape=(N_tilde, N)
        signal observation matrix :math:`\mathbf{C}^\mathsf{T}`.
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.ticker
    >>> from cbadc.analog_system import IIRDesign
    >>> wp = 2 * np.pi * 1e3
    >>> ws = 2 * np.pi * 2e3
    >>> gpass = 0.1
    >>> gstop = 80
    >>> filter = IIRDesign(wp, ws, gpass, gstop)
    >>> f = np.logspace(1, 5)
    >>> w = 2 * np.pi * f
    >>> tf = filter.transfer_function_matrix(w)
    >>> fig, ax1 = plt.subplots()
    >>> _ = ax1.set_title('Analog filter frequency response')
    >>> _ = ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> _ = ax1.set_xlabel('Frequency [Hz]')
    >>> _ = ax1.semilogx(f, 20 * np.log10(np.abs(tf[0, 0, :])))
    >>> _ = ax1.grid()
    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(tf[0, 0, :]))
    >>> _ = ax2.plot(f, angles, 'g')
    >>> _ = ax2.set_ylabel('Angle (radians)', color='g')
    >>> _ = ax2.grid()
    >>> _ =ax2.axis('tight')
    >>> nticks = 8
    >>> _ = ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
    >>> _ = ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

    See also
    --------
    :py:class:`cbadc.analog_system.ButterWorth`
    :py:class:`cbadc.analog_system.ChebyshevI`
    :py:class:`cbadc.analog_system.ChebyshevII`
    :py:class:`cbadc.analog_system.Cauer`

    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, wp, ws, gpass, gstop, ftype="ellip"):
        """Create a IIR filter"""
        z, p, k = scipy.signal.iirdesign(
            wp, ws, gpass, gstop, analog=True, ftype=ftype, output="zpk"
        )
        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)
