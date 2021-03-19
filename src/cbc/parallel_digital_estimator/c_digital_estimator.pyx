# distutils: language = c++
from cbc.offline_computations import care
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
from numpy import dot as dot_product, eye, zeros, int8, int32, double, roll, array
from numpy.linalg import eig, pinv
from libc.stdint cimport int8_t

cdef class C_Digital_Estimator():

    def __init__(self, analogSystem, digitalControl, eta2, K1, K2):
        self._iteration = 1
        self._M = analogSystem.M()
        self._N = analogSystem.N()
        self._L = analogSystem.L()
        self._K1 = K1
        self._K2 = K2
        self._K3 = K1 + K2
        self.compute_filter_coefficients(analogSystem, digitalControl, eta2)
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
        self.estimate = zeros((self._L,), dtype=double)


    cdef void compute_filter_coefficients(self, AnalogSystem analogSystem, DigitalControl digitalControl, double eta2):
        # Compute filter coefficients
        A = array(analogSystem._A).transpose()
        B = array(analogSystem._CT).transpose()
        Q = dot_product(analogSystem._B, array(analogSystem._B).transpose())
        R = eta2 * eye(analogSystem._N_tilde)
        # Solve care
        Vf, Vb = care(A, B, Q, R)
        cdef double T = digitalControl._Ts
        CCT = dot_product(array(analogSystem._CT).transpose(),array(analogSystem._CT))
        tempAf = analogSystem._A - dot_product(Vf,CCT) / eta2
        tempAb = analogSystem._A + dot_product(Vb,CCT) / eta2
        Af = expm(tempAf * T)
        Ab = expm(-tempAb * T)
        Gamma = analogSystem.Gamma()
        # Solve IVPs
        Bf = zeros((self._N, self._M))
        Bb = zeros((self._N, self._M))
        atol = 1e-200
        rtol = 1e-10
        max_step = T/1000.
        for m in range(self._M):
            derivative1 = lambda t, x: dot_product(tempAf, x) + dot_product(Gamma, digitalControl.impulse_response(m, t))
            solBf = solve_ivp(derivative1, (0, T), zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            derivative2 = lambda t, x: -dot_product(tempAb, x) + dot_product(Gamma, digitalControl.impulse_response(m, t))
            solBb = -solve_ivp(derivative2, (0, T), zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            for n in range (self._N):
                Bf[n, m] = solBf[n]
                Bb[n, m] = solBb[n]
        print(f"tempAf New: {tempAf}")
        print(f"tempAb New: {tempAb}")
        # print(f"New Parallel Bf: {array(Bf)}")
        # print(f"New Parallel Bb: {array(Bb)}")

        # Solve linear system of equations
        W = solve(Vf + Vb, analogSystem._B)

        # Parallelilize
        temp, Q_f = eig(Af)
        self.forward_a = array(temp, dtype=complex, order='C')
        Q_f_inv = pinv(Q_f, rcond=1e-20)
        temp, Q_b = eig(Ab)
        self.backward_a = array(temp, dtype=complex, order='C')
        Q_b_inv = pinv(Q_b, rcond=1e-20)

        self.forward_b = array(dot_product(Q_f_inv, Bf).flatten(), dtype=complex, order='C')
        self.backward_b = array(dot_product(Q_b_inv, Bb).flatten(), dtype=complex, order='C')

        self.forward_w = array(-dot_product(Q_f.transpose(), W).flatten(), dtype=complex, order='C')
        self.backward_w = array(dot_product(Q_b.transpose(), W).flatten(), dtype=complex, order='C')

        print(array(self.forward_a))
        # print(array(self.backward_a))
        # print(array(self.forward_b))
        # print(array(self.backward_b))
        # print(array(self.forward_w))
        # print(array(self.backward_w))

    def __dealloc__(self):
        del self._filter

    def output(self):
        self._iteration += 1
        self._filter.output(&self.estimate[0])
        return array(self.estimate, dtype=double, ndmin=1)
    
    def empty(self):
        return self._filter.empty_batch()

    def full(self):
        return self._filter.full_batch()

    def input(self, arg):
        cdef int [:] temp = array(arg, dtype=int32);
        # print("Input from python: ", array(temp))
        self._filter.input(&temp[0])
    
    def compute_batch(self):
        self._filter.compute_new_batch()