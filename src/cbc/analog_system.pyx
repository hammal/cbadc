"""
analog_system.py
====================================
The analog system of the control-bounded converter.
"""
#cython: language_level=3
import numpy as np


cdef class AnalogSystem:
    """The AnalogSystem class holds the analog system data parameters as well as
    provides related utility functions.

    :param a: the system matrix
    :type a: ndarray
        N x N dimensional numpy array of dtype numpy.float
    :param b: the input matrix
    :type b: ndarray
        N X L dimensional numpy array of dtype numpy.float
    :param c: the signal observation matrix
    :type c: ndarray
        N_tilde x N dimensional numpy array of dtype numpy.float
    :param gamma: the control input matrix
    :type gamma: ndarray
        N x M dimensional numpy array of dtype numpy.float
    :param gamma_tilde: the control observation matrix
    :type gamma_tilde: ndarray
        M x N dimensional numpy array of dtype numpy.float
    :param m: number of control inputs
    :type m: int
    :param n: number of states
    :type n: int
    :param l: numper of signal inputs
    :type l: int
    :param n_tilde: number of signal observations
    :type n_tilde: int
    """

    def __init__(self, A, B, C, Gamma, Gamma_tilde):
        """Initialize an analog system.

        :param A: the system matrix
        :type A: ndarray
        :param B: the input matrix
        :type B: ndarray
        :param C: the signal observation matrix
        :type C: ndarray
        :param Gamma: the control input matrix
        :type Gamma: ndarray
        :param Gamma_tilde: the control observation matrix
        :type Gamma_tilde: ndarray
        :raises InvalidAnalogSystemError: indicating errors in analog system
        matrices
        """
        self._A = np.array(A, dtype=np.double)    
        self._B = np.array(B, dtype=np.double)
        self._CT = np.array(C, dtype=np.double).transpose()
        self._Gamma = np.array(Gamma, dtype=np.double)
        self._Gamma_tildeT = np.array(Gamma_tilde, dtype=np.double)

        self._N = self._A.shape[0]    

        if self._A.shape[0] != self._A.shape[1]:
            raise InvalidAnalogSystemError(self, "system matrix not square")

        # ensure matrices
        if len(self._B.shape) == 1:
            self._B = self._B.reshape((self._N, 1))
        if len(self._CT.shape) == 1:
            self._CT = self._CT.reshape((1, self._N))

        self._M = self._Gamma.shape[1]
        self._M_tilde = self._Gamma_tildeT.shape[0]
        self._L = self._B.shape[1]
        self._N_tilde = self._CT.shape[0]

        self.temp_derivative = np.zeros(self._N,dtype=np.double)
        self.temp_y = np.zeros(self._N_tilde, dtype=np.double)
        self.temp_s_tilde = np.zeros(self._M_tilde, dtype=np.double)


        if self._B.shape[0] != self._A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with input matrix B."
            )

        if self._CT.shape[1] != self._A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with signal observation matrix C."
            )

        if self._Gamma.shape[0] != self._A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with control input matrix Gamma."
            )

        if self._Gamma_tildeT.shape[1] != self._A.shape[0]:
            raise InvalidAnalogSystemError(
                self,
                "N does not agree with control observation matrix Gamma_tilde.",
            )

        
    cpdef double [:] derivative(self, double [:] x, double t, double [:] u, double [:] s):
        """produces the state derivative :math:`\dot{\mathbf{x}}(t)` as a
        function of the state vector :math:`\mathbf{x}(t)`, the given time
        :math:`t`, the input signal u, and the control contribution s.

        :param x: state vector
        :type x: ndarray
            N dimensional
        :param t: time instance
        :type t: float
        :param u: input signal
        :type t: (float)=> ndarray
            function resulting in a L dimensional numpy array for each float
            argument.
        :param s: control contribution
        :type t: (float) => ndarray
            function resulting in a M dimensional numpy array for each float
            argument.
        :return: state derivative
        :rtype: ndarray
            N dimensional numpy array of dtype numpy.float
        """
        cdef int n, nn, l, m
        for n in range(self._N):
            self.temp_derivative[n] = 0
            for nn in range(self._N):
                self.temp_derivative[n] += self._A[n,nn] * x[nn]
            for l in range(self._L):
                self.temp_derivative[n] += self._B[n,l] * u[l]
            for m in range(self._M):
                self.temp_derivative[n] += self._Gamma[n,m] * s[m]
        return self.temp_derivative
    
    cpdef double [:] signal_output(self, double [:] x):
        cdef int n, n_tilde
        for n_tilde in range(self._N_tilde):
            self.temp_y[n_tilde] = 0
            for n in range(self._N):
                self.temp_y[n_tilde] += self._CT[n_tilde, n] * x[n]
        return self.temp_y
    
    cpdef double [:] control_output(self, double [:] x):
        cdef int n, m_tilde
        for m_tilde in range(self._M_tilde):
            self.temp_s_tilde[m_tilde] = 0
            for n in range(self._N):
                self.temp_s_tilde[m_tilde] = self._Gamma_tildeT[m_tilde, n] * x[n]
        return self.tem_s_tilde

    cdef complex [:,:] _atf(self, double _omega):
            return np.dot(
                np.linalg.inv(np.complex(0, _omega) * np.eye(self._N) - self._A),
                self._B,
            )

    def analog_transfer_function_matrix(self, double [:] omega):
        """produces the analog transfer function (ATF) matrix of the the analog system

        :param omega: angular frequencies specified in rad/s
        :type omega: ndarray
            K dimensional numpy array of angular frequencies to be computed
        :return: ATF matrix evaluated at the elements of omega
        :rtype: ndarray
            N X L X K dimensional numpy array of dtype numpy.complex
        """
        cdef int size = omega.size
        cdef complex [:,:,:] result =  np.zeros((self._N, self._L, size), dtype=np.complex)
        cdef int index
        for index in range(size):
            result[:, :, index] = self._atf(omega[index])
        return result

    def transfer_function(self, omega):
        """outputs the analog systems tranfer function matrix

        :param omega: angular frequencies specified in rad/s
        :type omega: ndarray
            K dimensional numpy array of angular frequencies to be computed
        :return: transfer function evaluated at the elements of omega
        :rtype: ndarray
            N_tilde x N x K dimensional numpy array of dtype numpy.complex
        """
        resp = np.einsum('ij,jkl', self._CT, self.analog_transfer_function_matrix(omega))
        return np.asarray(resp)

    def __str__(self):
        return f"The analog system is defined as:\nA: {self.A()},\nB: {self.B()},\nC: {self.C()},\nGamma: {self.Gamma()},\nand \nGamma_tilde: {self.Gamma_tilde()}"

    def A(self):
        return np.asarray(self._A)

    def B(self):
        return np.asarray(self._B)

    def C(self):
        return np.asarray(self._CT).transpose()
    
    def Gamma(self):
        return np.asarray(self._Gamma)
    
    def Gamma_tilde(self):
        return np.asarray(self._Gamma_tildeT).transpose()
    
    def N(self):
        return self._N
    
    def N_tilde(self):
        return self._N_tilde
    
    def M(self):
        return self._M

    def M_tilde(self):
        return self._M_tilde

    def L(self):
        return self._L
            


class InvalidAnalogSystemError(Exception):
    """Error when detecting faulty analog system specification

    :param system: an analog system object
    :type system: :class:`.AnalogSystem`
    :param message: error message
    :type message: string
    """

    def __init__(self, system, message):
        self.analog_system = system
        self.message = message
