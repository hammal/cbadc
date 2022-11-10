"""Sinusoidal signals."""
from typing import Union
from ._analog_signal import _AnalogSignal
from sympy import cos
from mpmath import mp
import numpy as np


def get_quadrature_signal_pair(
    amplitude_modulation: _AnalogSignal,
    phase_modulation: _AnalogSignal,
    angular_frequency: float = 1.0,
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
        QuadratureSignal(
            amplitude_modulation, phase_modulation, angular_frequency, phase, offset
        ),
        QuadratureSignal(
            amplitude_modulation,
            phase_modulation,
            angular_frequency,
            phase + np.pi / 4,
            offset,
        ),
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
            * np.cos(self.angular_frequency * t + self.phase_modulation(t) + self.phase)
            + self.offset
        )

    def _mpmath(self, t: Union[mp.mpf, float]):
        t = mp.mpmathify(t)
        return (
            self.amplitude_modulation._mpmath(t)
            * mp.cos(
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
