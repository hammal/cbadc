"""
The digital estimator.
"""
#cython: language_level=3
from cbc.offline_computations import care
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
from numpy import dot as dot_product, eye, zeros, int8, double, roll, array


cdef class MidPointFilter():

    def __cinit__(self, AnalogSystem analogSystem, DigitalControl digitalControl, double eta2, int K1, int K2 = 0):
        """Initializes filter object by computing filter coefficents and allocation buffer memory.

        Args:
            analogSystem (:obj:`AnalogSystem`): an anlog system
            digitalControl (:obj:`DigitalControl`): a digital control
            eta2 (float): the bandwidth parameter.
            K1 (int): the batch size
            K2 (int): the lookahead size  
        """
        self._M = analogSystem.M
        self._N = analogSystem.N
        self._L = analogSystem.L
        self._K1 = K1
        self._K2 = K2
        self._K3 = K1 + K2
        self.compute_filter_coefficients(analogSystem, digitalControl, eta2)
        self.allocate_memory_buffers()
    
    cpdef void compute_batch(self):
        cdef int k
        
        # check if ready to compute buffer
        if (self._control_signal_in_buffer < self._K3):
            raise "Control signal buffer not full"
        # compute lookahead
        # print(array(self._control_signal))
        for k in range(self._K3, 0, -1):
            self.temp_backward_mean = dot_product(self._Ab, self.temp_backward_mean) + dot_product(self._Bb, self._control_signal[k, :])
            temp_est = dot_product(self._WT, self.temp_backward_mean)
            if k < self._K1:
                for l in range(self._L):
                    self._estimate[k, l] = temp_est[l]
            self.temp_backward_mean = dot_product(self._Ab, self.temp_backward_mean) + dot_product(self._Bb, self._control_signal[k, :])
        for k in range(0, self._K1):
            self.temp_forward_mean = dot_product(self._Af, self.temp_forward_mean) + dot_product(self._Bf, self._control_signal[k, :])
            temp_est = -dot_product(self._WT, self.temp_forward_mean)
            for l in range(self._L):
                self._estimate[k, l] += temp_est[l]
            self.temp_forward_mean = dot_product(self._Af, self.temp_forward_mean) + dot_product(self._Bf, self._control_signal[k, :])
        for n in range(self._N):
            self.temp_backward_mean[n] = 0
            # temp_forward_mean is already set correctly.

        # rotate buffer to make place for new control signals
        self._control_signal = roll(self._control_signal, -self._K1, axis=0)
        self._control_signal_in_buffer -= self._K1
        
    cdef void compute_filter_coefficients(self, AnalogSystem analogSystem, DigitalControl digitalControl, double eta2):
        # Compute filter coefficients
        A = array(analogSystem.A).transpose()
        B = array(analogSystem.CT).transpose()
        Q = dot_product(array(analogSystem.B), array(analogSystem.B).transpose())
        R = eta2 * eye(analogSystem.N_tilde)
        # Solve care
        Vf, Vb = care(A, B, Q, R)
        cdef double T = digitalControl.T
        CCT = dot_product(array(analogSystem.CT).transpose(), array(analogSystem.CT))
        tempAf = analogSystem.A - dot_product(Vf, CCT) / eta2
        tempAb = analogSystem.A + dot_product(Vb, CCT) / eta2
        self._Af = expm(tempAf * T / 2.0)
        self._Ab = expm(-tempAb * T / 2.0)
        Gamma = array(analogSystem.Gamma)
        # Solve IVPs
        self._Bf = zeros((self._N, self._M))
        self._Bb = zeros((self._N, self._M))

        atol = 1e-200
        rtol = 1e-10
        max_step = T/1000.0
        for m in range(self._M):
            derivative = lambda t, x: dot_product(tempAf, x) + dot_product(Gamma, digitalControl.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, T / 2.0), zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            derivative = lambda t, x: - dot_product(tempAb, x) + dot_product(Gamma, digitalControl.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, T / 2.0), zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            for n in range (self._N):
                self._Bf[n, m] = solBf[n]
                self._Bb[n, m] = solBb[n]

        # Solve linear system of equations
        print(f"New Bf: {array(self._Bf)}")
        print(f"New Bb: {array(self._Bb)}")
        self._WT = solve(Vf + Vb, analogSystem.B).transpose()

    cdef void allocate_memory_buffers(self):
        # Allocate memory buffers
        self._control_signal = zeros((self._K3 + 1, self._M), dtype=int8)
        self._estimate = zeros((self._K1, self._L), dtype=double)
        self._control_signal_in_buffer = 0
        self.temp_forward_mean = zeros(self._N, dtype=double) 
        self.temp_backward_mean = zeros(self._N, dtype=double) 


    def input(self, char [:] s):
        if (self._control_signal_in_buffer == self._K3 + 1):
            raise "Input buffer full. You must compute batch before adding more control signals"
        for m in range(self._M):
            self._control_signal[self._control_signal_in_buffer, m] = 2 * s[m] - 1
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer == (self._K3)

    def output(self, int index):
        if (index < 0 or index >= self._K1):
            raise "index out of range"
        return array(self._estimate[index, :], dtype=double)
    
    cpdef int batch_size(self):
        return self._K1

    cpdef int lookahead(self):
        return self._K2

    cpdef int size(self):
        return self._K3