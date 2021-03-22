from cbadc.analog_system cimport AnalogSystem
from cbadc.digital_control cimport DigitalControl
from cbadc.analog_signal cimport AnalogSignal
import numpy as np
from scipy.integrate import solve_ivp
import math


cdef class StateSpaceSimulator(object):
    """Simulate the analog system and digital control interactions
    in the precense on analog signals.

    Parameters
    ----------
    analogSystem : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system
    digitalControl: :py:class:`cbadc.digital_control.DigitalControl`
        the digital control
    inputSignal : :py:class:`cbadc.analog_signal.AnalogSignal`
        the analog signal (or a derived class)
    Ts : `float`, optional
        specify a sampling rate at which we want to evaluate the systems
        , defaults to digitalContro.Ts. Note that this Ts must be smaller 
        than digitalControl.Ts. 
    t_stop : `float`, optional
        determines a stop time, defaults to :py:obj:`math.inf`



    Attributes
    ----------
    analogSystem : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system being simulated.
    digitalControl : :py:class:`cbadc.digital_control.DigitalControl`
        the digital control being simulated.
    t : `float`
        current time of simulator.
    Ts : `float` 
        sample rate of simulation.
    t_stop : `float`
        end time at which the generator raises :py:class:`StopIteration`.
    rtol, atol : `float`, `optional`
        Relative and absolute tolerances. The solver keeps the local error estimates less 
        than atol + rtol * abs(y). Effects the underlying solver as described in 
        :py:func:`scipy.integrate.solve_ivp`. Default to 1e-3 for rtol and 1e-6 for atol.
    max_step : `float`, `optional`
        Maximum allowed step size. Default is np.inf, i.e., the step size is not 
        bounded and determined solely by the solver. Effects the underlying solver as 
        described in :py:func:`scipy.integrate.solve_ivp`. Defaults to :py:obj:`math.inf`. 
    See also
    --------
    :py:class:`cbadc.analog_signal.AnalogSignal`
    :py:class:`cbadc.analog_system.AnalogSystem`
    :py:class:`cbadc.digital_control.DigitalControl`

    Examples
    --------
    >>> from cbadc import StateSpaceSimulator, Sinusodial, AnalogSystem, DigitalControl

    See also
    --------

    Yields
    ------
    `array_like`, shape=(M,), dtype=numpy.int8
    
    Raises
    ------
    str

    """
    cdef readonly AnalogSystem analog_system
    cdef readonly DigitalControl digital_control
    cdef dict __dict__
    cdef readonly double t, Ts, t_stop
    cdef double [:] _state_vector
    cdef double [:] _input_vector, _control_vector, _temp_state_vector
    cdef double _atol, _rtol, _max_step
        
    cdef _ordinary_differentail_function(self, t, y):
        cdef int l,
        for l in range(self.analog_system.L):
            self._input_vector[l] = self._is[l].evaluate(t)
        self._temp_state_vector = np.dot(self.analog_system.Gamma_tildeT, y)
        self._control_vector = self.digital_control.control_contribution(t, self._temp_state_vector)
        return np.asarray(self.analog_system.derivative(self._temp_state_vector, t, self._input_vector,self. _control_vector)).flatten()

    def __init__(self, 
            AnalogSystem analogSystem, 
            DigitalControl digitalControl, 
            inputSignal, 
            Ts=None,
            t_stop=math.inf,
            atol = 1e-6,
            rtol = 1e-3,
            max_step = math.inf,
        ):
        if analogSystem.L != len(inputSignal):
            raise """The analog system does not have as many inputs as in input
            list"""
        self.analog_system = analogSystem
        self.digital_control = digitalControl
        self._is = inputSignal
        self.t_stop = t_stop
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = self.digital_control.T
        if self.Ts > self.digital_control.T:
            raise f"Simulating with a sample period {self.Ts} that exceeds the control period of the digital control {self.digital_control.T}"
        self._state_vector = np.zeros(self.analog_system.N, dtype=np.double)
        self._temp_state_vector = np.zeros(self.analog_system.N, dtype=np.double)
        self._input_vector = np.zeros(self.analog_system.L, dtype=np.double)
        self._control_vector = np.zeros(self.analog_system.M, dtype=np.double)
        self._atol = atol # 1e-6
        self._rtol = rtol # 1e-6
        self._max_step = max_step # self.Ts / 10.
    
    def state_vector(self):
        """return current analog system state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Returns
        -------
        `array_like`, shape=(N,)
            returns the state vector :math:`\mathbf{x}(t)`
        """
        return self._state_vector[:]

    def __iter__(self):
        """Use simulator as an iterator
        """
        return self
    
    def __next__(self):
        """Computes the next control signal :math:`\mathbf{s}[k]`
        """
        cdef double t_end = self.t + self.Ts
        cdef double[2] t_span = (self.t, t_end)
        cdef double[1] t_eval = (t_end,)
        if t_end >= self.t_stop:
            raise StopIteration
        def f(t, x):
            return self._ordinary_differentail_function(t, x)
        sol = solve_ivp(f, t_span, self._state_vector, atol=self._atol, rtol=self._rtol, max_step=self._max_step)
        self._state_vector = sol.y[:,-1]
        self.t = t_end
        return self.digital_control.control_signal()

