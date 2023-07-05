"""Sinusoidal signals."""
from typing import Union, List
from ._analog_signal import _AnalogSignal
from .constant_signal import ConstantSignal
from sympy import cos
from mpmath import mp
import numpy as np


def _rotation_matrix(angle):
    """Get a rotation matrix.

    Parameters
    ----------
    angle : `float`
        the angle of rotation

    Returns
    -------
    `numpy.ndarray`
        a 2x2 rotation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def get_quadrature_signal_pair(
    amplitude: float,
    angular_frequency: float,
    in_phase: _AnalogSignal = ConstantSignal(1.0),
    quadrature: _AnalogSignal = ConstantSignal(0.0),
    phase: float = 0.0,
    offset: float = 0.0,
):
    """Get a quadrature pair of sinusoidal signals.

    Parameters
    ----------
    amplitude_modulation: :py:class:`cbadc.analog_signal._AnalogSignal`
        the amplitude modulation
    phase_modulation: :py:class:`cbadc.analog_signal._AnalogSignal`
        the phase modulation
    angular_frequency: `float`
        the angular frequency of the sinusoidal signal
    phase: `float`
        the phase of the sinusoidal signal
    offset: `float`
        the offset of the sinusoidal signal

    Returns
    -------
    `tuple` of :py:class:`cbadc.analog_signal._AnalogSignal`
        a quadrature pair of sinusoidal signals

    """

    return (
        QuadratureSignal2(amplitude, angular_frequency, True, [in_phase, quadrature]),
        QuadratureSignal2(amplitude, angular_frequency, False, [in_phase, quadrature]),
        # QuadratureSignal(
        #     amplitude_modulation, phase_modulation, angular_frequency, phase, offset
        # ),
        # ConstantSignal(0.0),
        # QuadratureSignal(
        #     amplitude_modulation,
        #     phase_modulation,
        #     angular_frequency,
        #     # phase + np.pi,
        #     phase - np.pi / 2,
        #     offset,
        # ),
    )


class QuadratureSignal(_AnalogSignal):
    """An analog continuous-time sinusoidal signal.

    Parameters
    ----------
    amplitude_modulation : :py:class`cbadc.analog_signal.AnalogSignal`
        the amplitude modulation signal
    phase_modulations: :py:class`cbadc.analog_signal.AnalogSignal`
        the phase modulation signal
    angular_frequency: `float`
        the angular frequency of the sinusoidal signal
    phase: `float`
        the phase of the sinusoidal signal
    offset: `float`
        the offset of the sinusoidal signal

    """

    amplitude: float
    angular_frequency: float
    phase: float
    offset: float

    def __init__(
        self,
        amplitude_modulation: _AnalogSignal,
        phase_modulation: _AnalogSignal,
        angular_frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
    ):
        super().__init__()
        self.amplitude_modulation = amplitude_modulation
        self.phase_modulation = phase_modulation
        self.angular_frequency = angular_frequency
        self.phase = phase
        self.offset = offset
        self._mpmath_dic = {
            'angular_frequency': mp.mpmathify(angular_frequency),
            'phase': mp.mpmathify(phase),
            'offset': mp.mpmathify(offset),
        }

    def __str__(self):
        return """
A quadrature signal.
        """

    def evaluate(self, t: float) -> float:
        """Evaluate the signal at time :math:`t`.

        Parameters
        ----------
        t : `float`
            the time instance for evaluation.

        Returns
        -------
        float
            The analog signal value
        """
        # in-phase component
        return (
            self.amplitude_modulation(t)
            * np.sin(self.angular_frequency * t + self.phase_modulation(t) + self.phase)
            + self.offset
        )

    def _mpmath(self, t: Union[mp.mpf, float]):
        t = mp.mpmathify(t)
        return (
            self.amplitude_modulation._mpmath(t)
            * mp.sin(
                self._mpmath_dic['angular_frequency'] * t
                + self.phase_modulation._mpmath(t)
                + self._mpmath_dic['phase']
            )
            + self._mpmath_dic['offset']
        )

    def symbolic(self) -> cos:
        """Returns as symbolic expression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            a sinusodial function
        """
        return (
            self.amplitude_modulation.symbolic()
            * cos(
                self.angular_frequency * self.t
                + self.phase
                + self.phase_modulation.symbolic()
            )
            + self.offset
        )


class QuadratureSignal2(_AnalogSignal):
    """An analog continuous-time sinusoidal signal.

    Parameters
    ----------
    amplitude_modulation : :py:class`cbadc.analog_signal.AnalogSignal`
        the amplitude modulation signal
    phase_modulations: :py:class`cbadc.analog_signal.AnalogSignal`
        the phase modulation signal
    angular_frequency: `float`
        the angular frequency of the sinusoidal signal
    phase: `float`
        the phase of the sinusoidal signal
    offset: `float`
        the offset of the sinusoidal signal

    """

    amplitude: float
    angular_frequency: float
    phase: float
    offset: float

    def __init__(
        self,
        amplitude: float,
        angular_frequency: float,
        in_phase: bool,
        i_q: List[_AnalogSignal],
    ):
        super().__init__()
        self.amplitude = amplitude
        self.angular_frequency = angular_frequency
        if in_phase:
            self.phase = np.array([1.0, 0.0])
        else:
            self.phase = np.array([0.0, 1.0])
        if len(i_q) != 2:
            raise ValueError('i_q must be a list of length 2.')
        self._iq = i_q

    def evaluate(self, t: float) -> float:
        """Evaluate the signal at time :math:`t`.

        Parameters
        ----------
        t : `float`
            the time instance for evaluation.

        Returns
        -------
        float
            The analog signal value
        """
        iq = np.array([iq.evaluate(t) for iq in self._iq])
        return self.amplitude * np.dot(
            self.phase,
            np.dot(_rotation_matrix(self.angular_frequency * t), iq),
        )
