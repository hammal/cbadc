"""
analog_system.py
====================================
The analog system of the control-bounded converter.
"""
import numpy as np


class AnalogSystem(object):
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
        self.a = np.array(A, dtype=np.float)
        self.b = np.array(B, dtype=np.float)
        self.c = np.array(C, dtype=np.float)
        self.gamma = np.array(Gamma, dtype=np.float)
        self.gamma_tilde = np.array(Gamma_tilde, dtype=np.float)

        self.n = self.a.shape[0]

        if self.a.shape[0] != self.a.shape[1]:
            raise InvalidAnalogSystemError(self, "system matrix not square")

        # ensure matrices
        if len(self.b.shape) == 1:
            self.b = self.b.reshape((self.n, 1))
        if len(self.c.shape) == 1:
            self.c = self.c.reshape((1, self.n))

        self.m = self.gamma.shape[1]
        self._l = self.b.shape[1]
        self.n_tilde = self.c.shape[0]

        if self.b.shape[0] != self.a.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with input matrix B."
            )

        if self.c.shape[1] != self.a.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with signal observation matrix C."
            )

        if self.gamma.shape[0] != self.a.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with control input matrix Gamma."
            )

        if self.gamma_tilde.shape[1] != self.a.shape[0]:
            raise InvalidAnalogSystemError(
                self,
                "N does not agree with control observation matrix Gamma_tilde.",
            )

        if self.gamma.shape[1] != self.gamma_tilde.shape[0]:
            raise InvalidAnalogSystemError(
                self,
                """M in control input matrix and control observation matrix do not
                agree.""",
            )

    def derivative(self, x, t, u, s):
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

        return np.array(
            np.dot(self.a, x) + np.dot(self.b, u(t)) + np.dot(self.gamma, s(t)),
            dtype=np.float,
        )

    def analog_transfer_function_matrix(self, omega):
        """produces the analog transfer function (ATF) matrix of the the analog system

        :param omega: angular frequencies specified in rad/s
        :type omega: ndarray
            K dimensional numpy array of angular frequencies to be computed
        :return: ATF matrix evaluated at the elements of omega
        :rtype: ndarray
            N X N X K dimensional numpy array of dtype numpy.complex
        """

        def atf(_omega):
            response = np.dot(
                np.linalg.inv(np.complex(0, _omega) * np.eye(self.n) - self.a),
                self.b,
            )
            return response

        result = np.zeros((self.n, self.n, omega.size), dtype=np.complex)
        for index, o in enumerate(omega):
            result[:, :, index] = atf(o)
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
        return f"""The analog system is defined as:\nA: {self.a},\nB:
        {self.b},\nC: {self.c},\n and\nGamma: {self.gamma}"""


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
