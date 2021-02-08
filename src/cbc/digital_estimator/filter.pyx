"""
The digital estimator.
"""
#cython: language_level=3
from .offline_computations import care
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
from numpy import dot as dot_product, eye, zeros, int8, double, roll

cdef class Filter:
    def __init__(self, analogSystem, digitalControl, eta2, K1, K2):
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
        if (K2 < 1):
            raise "K2 must be a positive integer"
        # Compute filter coefficients
        A = analogSystem.A().transpose()
        B = analogSystem.C().transpose()
        Q = dot_product(analogSystem.B(), analogSystem.B().transpose())
        R = eta2 * eye(analogSystem.N_tilde())
        # Solve care
        Vf, Vb = care(A, B, Q, R)
        cdef double T = digitalControl.Ts()
        CCT = dot_product(analogSystem.C().transpose(),analogSystem.C())
        tempAf = analogSystem.A() - dot_product(Vf,CCT)/eta2
        tempAb = analogSystem.A() + dot_product(Vb,CCT)/eta2
        self._Af = expm(tempAf * T)
        self._Ab = expm(-tempAb * T)
        Gamma = analogSystem.Gamma()
        cdef int M = analogSystem.M()
        cdef int N = analogSystem.N()
        # Solve IVPs
        self._Bf = zeros((N, M))
        self._Bb = zeros((N, M))
        for m in range(M):
            derivative = lambda t, x: dot_product(tempAf, x) + dot_product(Gamma, digitalControl.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, T), zeros(N), t_eval=(T,)).y[:,0]
            derivative = lambda t, x: dot_product(tempAb, x) + dot_product(Gamma, digitalControl.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, T), zeros(N), t_eval=(T,)).y[:,0]
            for n in range (N):
                self._Bf[n, m] = solBf[n]
                self._Bb[n, m] = solBb[n]

        # Solve linear system of equations
        self._WT = solve(Vf + Vb, analogSystem.B()).transpose()

        # Allocate memory buffers
        cdef int L = analogSystem.L()
        self._K1 = K1
        self._K2 = K2
        self._K3 = K1 + K2
        self._control_signal = zeros((M, self._K3 + 1), dtype=int8)
        self._mean = zeros((N, self._K1 + 1), dtype=double)
        self._estimate = zeros((L, K1), dtype=double)
        self._control_signal_in_buffer = 0


    cdef int input(self, char [:] s):
        if (self._control_signal_in_buffer == self._K3 + 1):
            raise "Input buffer full. You must compute batch before adding more control signals"
        cdef int M = self.analogSystem.M()
        for m in range(M):
            self._control_signal[m, self._control_signal_in_buffer] = s[m]
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer == self._K3 + 1

    cdef double [:] output(self, int index):
        if (index < 0 or index > self._K1):
            raise "index out of range"
        return self._estimate[:,index]
    
    cdef void compute_batch(self):
        cdef int k1, k2, k3
        # check if ready to compute buffer
        if (self._control_signal_in_buffer < self.K3 + 1):
            raise "Control signal buffer not full"
        # compute lookahead
        for k1 in range(self._K3, self._K1, -1):
            self._mean[:, self._K1 + 1] = dot_product(self._Ab, self._mean[:, self._K1 + 1]) + dot_product(self._Bb, self._control_signal[:,k1])
        # compute forward recursion
        for k2 in range(1, self._K1 + 1):
            self._mean[:, k2] = - dot_product(self._Af, self._mean[:, k2 - 1]) - dot_product(self._Bf, self._control_signal[:, k2 - 1])
        # compute backward recursion and estimate
        for k3 in range(self._K1, 0, -1):
            temp = dot_product(self._Ab, self._mean[k3 + 1]) + dot_product(self._Bb, self._control_signal[:, k3])
            self._estimate[:, k3 - 1] = dot_product(self._WT, self._mean[k3] + temp)
            self._mean[k3] = temp
        # reset intital means
        self._mean[:, 0] = self._mean[:, self._K1 - 1]
        self._mean[:, self._K1 + 1] = 0
        # rotate buffer to make place for new control signals
        self._control_signal = roll(self._control_signal, -self._K1, axis=1)
        self._control_signal_in_buffer -= self._K1
        
    
    cpdef int batch_size(self):
        return self._K1

    cpdef int lookahead(self):
        return self._K2

    cpdef int size(self):
        return self._K3

