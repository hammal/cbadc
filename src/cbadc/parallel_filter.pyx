#cython: language_level=3
# distutils: language = c++
"""Parallel filter interface.
"""
from cbadc.offline_computations import care
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
import numpy as np
cimport numpy as np


cdef class ParallelFilter():

    def __cinit__(self, AnalogSystem analog_system, DigitalControl digital_control, double eta2, int K1, double Ts, int K2 = 0):
        """Initializes filter object by computing filter coefficents and allocation buffer memory.

        Args:
            analog_system (:obj:`AnalogSystem`): an anlog system
            digital_control (:obj:`DigitalControl`): a digital control
            eta2 (float): the bandwidth parameter.
            K1 (int): the batch size
            K2 (int): the lookahead size  
        """
        self._M = analog_system.M
        self._N = analog_system.N
        self._L = analog_system.L
        self._K1 = K1
        self._K2 = K2
        self._K3 = K1 + K2
        self._Ts = Ts
        self.compute_filter_coefficients(analog_system, digital_control, eta2)
        self._filter = new ParallelDigitalEstimator(
            &(self.forward_a[0]), &(self.backward_a[0]), 
            &(self.forward_b[0]), &(self.backward_b[0]), 
            &(self.forward_w[0]), &(self.backward_w[0]), 
            K1, 
            K2, 
            self._M, 
            self._N, 
            self._L
            )
        self._estimates = np.zeros((self._K1, self._L), dtype=np.double)
    
    cpdef void compute_batch(self):
        self._filter.compute_new_batch()
        for index in range(self._K1):
            self._filter.output(&self._estimates[index, 0])
        
    cdef void compute_filter_coefficients(self, AnalogSystem analog_system, DigitalControl digital_control, double eta2):
        # Compute filter coefficients
        A = np.array(analog_system.A).transpose()
        B = np.array(analog_system.CT).transpose()
        Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
        R = eta2 * np.eye(analog_system.N_tilde)
        # Solve care
        Vf, Vb = care(A, B, Q, R)
        cdef double T = self._Ts
        CCT = np.dot(np.array(analog_system.CT).transpose(), np.array(analog_system.CT))
        tempAf = analog_system.A - np.dot(Vf,CCT) / eta2
        tempAb = analog_system.A + np.dot(Vb,CCT) / eta2
        self._Af = expm(tempAf * T)
        self._Ab = expm(-tempAb * T)
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self._Bf = np.zeros((self._N, self._M))
        self._Bb = np.zeros((self._N, self._M))

        atol = 1e-200
        rtol = 1e-10
        max_step = T/1000.0
        for m in range(self._M):
            derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, T), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, T), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            for n in range (self._N):
                self._Bf[n, m] = solBf[n]
                self._Bb[n, m] = solBb[n]
        self._WT = np.solve(Vf + Vb, analog_system.B).transpose()

        # Parallelilize
        temp, Q_f = np.eig(self._Af)
        self.forward_a = np.array(temp, dtype=np.complex128, order='C')
        Q_f_inv = np.pinv(Q_f, rcond=1e-20)
        temp, Q_b = np.eig(self._Ab)
        self.backward_a = np.array(temp, dtype=np.complex128, order='C')
        Q_b_inv = np.pinv(Q_b, rcond=1e-20)

        self.forward_b = np.array(np.dot(Q_f_inv, self._Bf).flatten(), dtype=np.complex128, order='C')
        self.backward_b = np.array(np.dot(Q_b_inv, self._Bb).flatten(), dtype=np.complex128, order='C')

        self.forward_w = np.array(-np.dot(Q_f.transpose(), self._W).flatten(), dtype=np.complex128, order='C')
        self.backward_w = np.array(np.dot(Q_b.transpose(), self._W).flatten(), dtype=np.complex128, order='C')

    def __dealloc__(self):
        del self._filter


    def input(self, char [:] s):
        if (self._control_signal_in_buffer == (self._K3)):
            raise BaseException("Input buffer full. You must compute batch before adding more control signals")
        cdef int [:] temp = np.array(s, dtype=np.int8)
        self._filter.input(&temp[0])
        return self._filter.full_batch()

    def output(self, int index):
        if (index < 0 or index >= self._K1):
            raise BaseException("index out of range")
        return np.array(self._estimate[index, :], dtype=np.double)
    
    cpdef int batch_size(self):
        return self._K1 

    cpdef int lookahead(self):
        return self._K2

    cpdef int size(self):
        return self._K3