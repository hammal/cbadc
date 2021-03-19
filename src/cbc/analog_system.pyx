#cython: language_level=3
import numpy as np


cdef class AnalogSystem:
    """Represents an analog system.

    The AnalogSystem class represents an analog sytem goverened by the differential equations,

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`
    
    where we refer to :math:`\mathbf{A} \in \mathbb{R}^{N \\times N}` as the system matrix,
    :math:`\mathbf{B} \in \mathbb{R}^{N \\times L}` as the input matrix,
    and :math:`\mathbf{\Gamma} \in \mathbb{R}^{N \\times M}` is the control input matrix. Furthermore, 
    :math:`\mathbf{x}(t)\in\mathbb{R}^{N}` is the state vector of the system, :math:`\mathbf{u}(t)\in\mathbb{R}^{L}`
    is the vector valued, continuous-time, analog input signal, and :math:`\mathbf{s}(t)\in\mathbb{R}^{M}` is the
    vector valued control signal.

    The analog system also has two (possibly vector valued) outputs namely:

    * The control observation :math:`\\tilde{\mathbf{s}}(t)=\\tilde{\mathbf{\Gamma}}^T \mathbf{x}(t)` and
    * The signal observation :math:`\mathbf{y}(t) = \mathbf{C}^T \mathbf{x}(t)`

    where :math:`\\tilde{\mathbf{\Gamma}}^T\in\mathbb{R}^{\\tilde{M} \\times N}` is the control observation matrix 
    and :math:`\mathbf{C}^T\in\mathbb{R}^{\\tilde{N} \\times N}` is the signal observation matrix.

    Parameters
    ----------
    A : `array_like`, shape=(N, N)
        system matrix.
    B : `array_like`, shape=(N, L)
        input matrix.
    C : `array_like`, shape=(N, N_tilde)
        signal observation matrix.
    Gamma : `array_like`, shape=(N, M)
        control input matrix.
    Gamma_tilde : `array_like`, shape=(N, M_tilde)
        control observation matrix.


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
        signal observation matrix :math:`\mathbf{C}^T`.
    Gamma : `array_like`, shape=(N, M)
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : `array_like`, shape=(M_tilde, N)
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^T`.

    See also
    --------
    :py:class:`cbc.state_space_simulator.StateSpaceSimulator`

    Example
    -------
    >>> import cbc
    >>> print("This should do something")

    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, A, B, C, Gamma, Gamma_tilde):
        """Create an analog system.
        """

        self.A = np.array(A, dtype=np.double)    
        self.B = np.array(B, dtype=np.double)
        self.CT = np.array(C, dtype=np.double).transpose()
        self.Gamma = np.array(Gamma, dtype=np.double)
        self.Gamma_tildeT = np.array(Gamma_tilde, dtype=np.double).transpose()

        self.N = self.A.shape[0]    

        if self.A.shape[0] != self.A.shape[1]:
            raise InvalidAnalogSystemError(self, "system matrix not square")

        # ensure matrices
        if len(self.B.shape) == 1:
            self.B = self.B.reshape((self.N, 1))
        if len(self.CT.shape) == 1:
            self.CT = self.CT.reshape((1, self.N))

        self.M = self.Gamma.shape[1]
        self.M_tilde = self.Gamma_tildeT.shape[0]
        self.L = self.B.shape[1]
        self.N_tilde = self.CT.shape[0]

        self.temp_derivative = np.zeros(self.N,dtype=np.double)
        self.temp_y = np.zeros(self.N_tilde, dtype=np.double)
        self.temp_s_tilde = np.zeros(self.M_tilde, dtype=np.double)


        if self.B.shape[0] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with input matrix B."
            )

        if self.CT.shape[1] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with signal observation matrix C."
            )

        if self.Gamma.shape[0] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with control input matrix Gamma."
            )

        if self.Gamma_tildeT.shape[1] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self,
                "N does not agree with control observation matrix Gamma_tilde.",
            )

        
    cpdef double [:] derivative(self, double [:] x, double t, double [:] u, double [:] s):
        """Compute the derivative of the analog system.
        
        Specifically, produces the state derivative 
        
        :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`
        
        as a function of the state vector :math:`\mathbf{x}(t)`, the given time
        :math:`t`, the input signal value :math:`\mathbf{u}(t)`, and the
        control contribution value :math:`\mathbf{s}(t)`.

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector evaluated at time t.
        t : `float`
            the time t.
        u : `array_like`, shape=(L,)
            the input signal vector evaluated at time t.
        s : `array_like`, shape=(M,)
            the control contribution evaluated at time t.

        Returns
        -------
        `array_like`, shape=(N,)
            the derivative :math:`\dot{\mathbf{x}}(t)`.
        """
        cdef int n, nn, l, m
        for n in range(self.N):
            self.temp_derivative[n] = 0
            for nn in range(self.N):
                self.temp_derivative[n] += self.A[n,nn] * x[nn]
            for l in range(self.L):
                self.temp_derivative[n] += self.B[n,l] * u[l]
            for m in range(self.M):
                self.temp_derivative[n] += self.Gamma[n,m] * s[m]
        return self.temp_derivative
    
    cpdef double [:] signal_output(self, double [:] x):
        """Computes the signal observation for a given state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Specifically, returns

        :math:`\mathbf{y}(t)=\mathbf{C}^T \mathbf{x}(t)`

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector.
    
        Returns
        -------
        `array_like`, shape=(N_tilde,)
            the signal observation.

        """
        cdef int n, n_tilde
        for n_tilde in range(self.N_tilde):
            self.temp_y[n_tilde] = 0
            for n in range(self.N):
                self.temp_y[n_tilde] += self.CT[n_tilde, n] * x[n]
        return self.temp_y
    
    cpdef double [:] control_output(self, double [:] x):
        """Computes the control observation for a given state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Specifically, returns
        
        :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^T \mathbf{x}(t)`
    
        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector.
    
        Returns
        -------
        `array_like`, shape=(M_tilde,)
            the control observation.

        """
        cdef int n, m_tilde
        for m_tilde in range(self.M_tilde):
            self.temp_s_tilde[m_tilde] = 0
            for n in range(self.N):
                self.temp_s_tilde[m_tilde] = self.Gamma_tildeT[m_tilde, n] * x[n]
        return self.tem_s_tilde

    cdef complex [:,:] _atf(self, double _omega):
            return np.dot(
                np.linalg.inv(np.complex(0, _omega) * np.eye(self.N) - self.A),
                self.B,
            )

    def analog_transfer_function_matrix(self, double [:] omega):
        """Evaluates the analog transfer function (ATF) matrix at the angular frequencies of the omega array.

        Specifically, evaluates
        
        :math:`\mathbf{G}(\omega) = \\left(\mathbf{A} - i \omega \mathbf{I}_N\\right)^{-1} \mathbf{B}`

        for each angular frequency in omega where :math:`\mathbf{I}_N` represents
        a square identity matrix of the same dimensions as :math:`\mathbf{A}` and :math:`i=\sqrt{-1}`.


        Parameters
        ----------
        omega: `array_like`, shape=(:,)
            a array_like object containing the angular frequencies for evaluation.
        
        Returns
        -------
        `array_like`, shape=(N, L, K)
            the ATF matrix evaluated at K different angular frequencies.
 
        """
        cdef int size = omega.size
        cdef complex [:,:,:] result =  np.zeros((self.N, self.L, size), dtype=np.complex)
        cdef int index
        for index in range(size):
            result[:, :, index] = self._atf(omega[index])
        return result

    def transfer_function(self, omega):
        """Evaluate the analog signal transfer function at the angular frequencies of the omega array.

        Specifically, evaluates
        
        :math:`\mathbf{G}(\omega) = \mathbf{C}^T \\left(\mathbf{A} - i \omega \mathbf{I}_N\\right)^{-1} \mathbf{B}`

        for each angular frequency in omega where :math:`\mathbf{I}_N` represents
        a square identity matrix of the same dimensions as :math:`\mathbf{A}` and :math:`i=\sqrt{-1}`.
        
        Parameters
        ----------
        omega: `array_like`, shape=(:,)
            a array_like object containing the angular frequencies for evaluation.
        
        Returns
        -------
        `array_like`, shape=(N_tilde, L, K)
            the signal transfer function evaluated at K different angular frequencies.
        """
        resp = np.einsum('ij,jkl', self.CT, self.analog_transfer_function_matrix(omega))
        return np.asarray(resp)

    def __str__(self):
        return f"The analog system is defined as:\nA: {self.A},\nB: {self.B},\nC: {self.C},\nGamma: {self.Gamma},\nand \nGamma_tilde: {self.Gamma_tilde}"


class InvalidAnalogSystemError(Exception):
    """Error when detecting faulty analog system specification

    Parameters
    ----------
    system : :py:class:`cbc.analog_system.AnalogSystem`
        An analog system object
    message: str
        error message
    """

    def __init__(self, system, message):
        self.analog_system = system
        self.message = message
