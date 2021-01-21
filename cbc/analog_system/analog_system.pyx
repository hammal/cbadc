"""
analog_system.py
====================================
The analog system of the control-bounded converter.
"""
#cython: language_level=3
import numpy as np


cdef class AnalogSystem(object):
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
    cdef double [:,:] A
    cdef double [:,:] B
    cdef double [:,:] C
    cdef double [:,:] Gamma
    cdef double [:,:] Gamma_tilde
    cdef double [:] x
    cdef double [:] y
    cdef double [:] s_tilde
    cdef int M, N, L, N_tilde, M_tilde

    def __init__(self, A, B, C, Gamma, Gamma_tilde, x = None):
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
        if A:
            self.A = np.array(A, dtype=np.float)
        else:
            raise InvalidAnalogSystemError(self, "system matrix not specified")

        if B:
            self.B = np.array(B, dtype=np.float)
        else:
            raise InvalidAnalogSystemError(self, "input matrix not specified")

        if C:
            self.C = np.array(C, dtype=np.float)
        else:
            raise InvalidAnalogSystemError(self, "signal observation matrix not specified")
        
        if Gamma:
            self.Gamma = np.array(Gamma, dtype=np.float)
        else:
            raise InvalidAnalogSystemError(self, "control input matrix not specified")
        
        if Gamma_tilde:
            self.Gamma_tilde = np.array(Gamma_tilde, dtype=np.float)
        else:
            raise InvalidAnalogSystemError(self, "control observation matrix not specified")

        self.N = self.A.shape[0]

        if x:
            self.x = np.array(x, dtype=np.float)
        else:
            self.x = np.zeros(self.N,dtype=np.float)

        self.y = np.dot(self.C, self.x)
        self.s_tilde = np.dot(self.Gamma_tilde, self.x)

        if self.A.shape[0] != self.A.shape[1]:
            raise InvalidAnalogSystemError(self, "system matrix not square")

        if self.x.size != self.N:
            raise InvalidAnalogSystemError(self, "state vector not N sized")

        # ensure matrices
        if len(self.B.shape) == 1:
            self.B = self.B.reshape((self.N, 1))
        if len(self.C.shape) == 1:
            self.C = self.C.reshape((1, self.N))

        self.M = self.Gamma.shape[1]
        self.M_tilde = self.Gamma_tilde.shape[0]
        self.L = self.B.shape[1]
        self.N_tilde = self.C.shape[0]

        if self.B.shape[0] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with input matrix B."
            )

        if self.C.shape[1] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with signal observation matrix C."
            )

        if self.Gamma.shape[0] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with control input matrix Gamma."
            )

        if self.Gamma_tilde.shape[1] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self,
                "N does not agree with control observation matrix Gamma_tilde.",
            )

        
    cpdef double [:] derivative(self, double [:] x, double t, u, s):
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
        cdef double [:] uu = u(t)
        cdef double [:] ss = s(t) 
        for n in range(self.N):
            self.x[n] = 0
            for nn in range(self.N):
                self.x[n] += self.A[n,nn] * x[nn]
            for l in range(self.L):
                self.x[n] += self.B[n,l] * uu[l]
            for m in range(self.M):
                self.x[n] += self.Gamma[n,m] * ss[m]
        return self.x
    
    cpdef double [:] signal_output(self):
        cdef int n, n_tilde
        for n_tilde in range(self.N_tilde):
            self.y[n_tilde] = 0
            for n in range(self.N):
                self.y[n_tilde] += self.C[n_tilde, n] * self.x[n]
        return self.y
    
    cpdef double [:] control_output(self):
        cdef int n, m_tilde
        for m_tilde in range(self.M_tilde):
            self.s_tilde[m_tilde] = 0
            for n in range(self.N):
                self.s_tilde[m_tilde] = self.Gamma_tilde[m_tilde, n] * self.x[n]
        return self.s_tilde

    cdef complex [:] _atf(self, double _omega):
            return np.dot(
                np.linalg.inv(np.complex(0, _omega) * np.eye(self.N) - self.A),
                self.B,
            )

    def analog_transfer_function_matrix(self, double [:] omega):
        """produces the analog transfer function (ATF) matrix of the the analog system

        :param omega: angular frequencies specified in rad/s
        :type omega: ndarray
            K dimensional numpy array of angular frequencies to be computed
        :return: ATF matrix evaluated at the elements of omega
        :rtype: ndarray
            N X N X K dimensional numpy array of dtype numpy.complex
        """
        cdef int size = omega.size
        cdef complex [:,:,:] result =  np.zeros((self.N, self.N, size), dtype=np.complex)
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
        resp = np.dot(self.c, self.analog_transfer_function_matrix(omega))
        return resp

    def __str__(self):
        return (
            f"The analog system is defined as:\nA: {self.a},\nB: {self.b},\n"
            f"C: {self.c},\n and\nGamma: {self.gamma}",
        )


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
