"""
The digital estimator.
"""
#cython: language_level=3
from .offline_computations import care
from cbc.digital_control.digital_control cimport DigitalControl
from cbc.analog_signal.analog_signal cimport AnalogSignal
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
from numpy import dot as dot_product, eye, zeros, int8, double, roll, array

cdef class Filter:
    def __cinit__(self, AnalogSignal analogSystem, DigitalControl digitalControl, double eta2, int K1, int K2 = 0):
        """Initializes filter object by computing filter coefficents and allocation buffer memory.

        Args:
            analogSystem (:obj:`AnalogSystem`): an anlog system
            digitalControl (:obj:`DigitalControl`): a digital control
            eta2 (float): the bandwidth parameter.
            K1 (int): the batch size
            K2 (int): the lookahead size  
        """
        # Check inputs
        if (K1 < 1):
            raise "K1 must be a positive integer"
        if (K2 < 0):
            raise "K2 must be a non negative integer"

        self._M = analogSystem._M
        self._N = analogSystem._N
        self._L = analogSystem._L
        self._K1 = K1
        self._K2 = K2
        self._K3 = K1 + K2
        self.compute_filter_coefficients(analogSystem, digitalControl, eta2)
        self.allocate_memory()
        

    cdef void compute_filter_coefficients(self, AnalogSignal analogSystem, DigitalControl digitalControl, double eta2):
        # Compute filter coefficients
        A = analogSystem._A.transpose()
        B = analogSystem._C.transpose()
        Q = dot_product(analogSystem._B, analogSystem._B.transpose())
        R = eta2 * eye(analogSystem._N_tilde)
        # Solve care
        Vf, Vb = care(A, B, Q, R)
        cdef double T = digitalControl._Ts
        CCT = dot_product(analogSystem._C.transpose(),analogSystem._C)
        tempAf = analogSystem._A - dot_product(Vf,CCT) / eta2
        tempAb = analogSystem._A + dot_product(Vb,CCT) / eta2
        self._Af = expm(tempAf * T)
        self._Ab = expm(-tempAb * T)
        Gamma = analogSystem.Gamma()
        # Solve IVPs
        self._Bf = zeros((self._N, self._M))
        self._Bb = zeros((self._N, self._M))
        for m in range(self._M):
            derivative = lambda t, x: dot_product(tempAf, x) + dot_product(Gamma, digitalControl.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, T), zeros(self._N), t_eval=(T,)).y[:,0]
            derivative = lambda t, x: dot_product(tempAb, x) + dot_product(Gamma, digitalControl.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, T), zeros(self._N), t_eval=(T,)).y[:,0]
            for n in range (self._N):
                self._Bf[n, m] = solBf[n]
                self._Bb[n, m] = solBb[n]

        # Solve linear system of equations
        self._WT = solve(Vf + Vb, analogSystem._B).transpose()

    cdef void allocate_memory(self):
        # Allocate memory buffers
        self._control_signal = zeros((self._K3 + 1, self._M), dtype=int8)
        self._estimate = zeros((self._K1, self._L), dtype=double)
        self._control_signal_in_buffer = 0


    cdef int input(self, char [:] s):
        if (self._control_signal_in_buffer == self._K3 + 1):
            raise "Input buffer full. You must compute batch before adding more control signals"
        for m in range(self._M):
            self._control_signal[self._control_signal_in_buffer, m] = 2 * s[m] - 1
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer == (self._K3)

    cpdef double [:] output(self, int index):
        if (index < 0 or index >= self._K1):
            raise "index out of range"
        return self._estimate[index, :]
    
    cpdef int batch_size(self):
        return self._K1

    cpdef int lookahead(self):
        return self._K2

    cpdef int size(self):
        return self._K3

    cpdef void compute_batch(self):
        pass

