"""Tools to construct analog systems by means of combining other analog systems."""
import numpy as np
from typing import Dict, List, Tuple
from .analog_system import AnalogSystem
from cbadc.digital_control import MultiLevelDigitalControl
from cbadc.analog_signal.clock import Clock
import json


def chain(analog_systems: List[AnalogSystem]) -> AnalogSystem:
    """Construct an analog system by chaining several analog systems.

    The chaining is achieved by chainging :math:`\hat{N}` systems,
    parameterized by
    :math:`\mathbf{A}_1, \mathbf{B}_1, \mathbf{C}^\mathsf{T}_1, \mathbf{D}_1, \mathbf{\Gamma}_1, \\tilde{\mathbf{\Gamma}}_1, \\dots, \mathbf{A}_{\hat{N}}, \mathbf{B}_{\hat{N}}, \mathbf{C}^\mathsf{T}_{\hat{N}}, \mathbf{D}_{\hat{N}}, \mathbf{\Gamma}_{\hat{N}}, \\tilde{\mathbf{\Gamma}}_{\hat{N}}`,
    as

    :math:`\mathbf{A} = \\begin{pmatrix} \mathbf{A}_1 \\\ \mathbf{B}_2 \mathbf{C}_1^\mathsf{T} & \mathbf{A}_2 \\\  \mathbf{B}_3 \mathbf{D}_2 \mathbf{C}_1^\mathsf{T} & \mathbf{B}_3 \mathbf{C}_2^\mathsf{T} & \mathbf{A}_3 \\\ \mathbf{B}_4 \mathbf{D}_3 \mathbf{D}_2 \mathbf{C}_1^\mathsf{T} & \mathbf{B}_4\mathbf{D}_3\mathbf{C}_2^\mathsf{T} & \mathbf{B}_4 \mathbf{C}_3^\mathsf{T} & \mathbf{A}_4  \\\ \\vdots & \\vdots & \\vdots & \\vdots & \\ddots  \\end{pmatrix}`

    :math:`\mathbf{B} = \\begin{pmatrix} \mathbf{B}_1 \\\ \mathbf{B}_2 \mathbf{D}_1 \\\ \mathbf{B}_3 \mathbf{D}_2 \mathbf{D}_1 \\\ \mathbf{B}_4 \mathbf{D}_3 \mathbf{D}_2 \mathbf{D}_1 \\\ \\vdots \\end{pmatrix}`

    :math:`\mathbf{C}^\mathsf{T} = \\begin{pmatrix} \mathbf{D}_{\hat{N}}\\cdots \mathbf{D}_2 \mathbf{C}_1^\mathsf{T} & \mathbf{D}_{\hat{N}}\\cdots \mathbf{D}_3 \mathbf{C}_2^\mathsf{T} & \mathbf{D}_{\hat{N}} \\cdots \mathbf{D}_4 \mathbf{C}_3^{\mathsf{T}} & \\dots & \mathbf{D}_{\hat{N}} \mathbf{C}_{\hat{N}-1}^\mathsf{T} & \mathbf{C}_{\hat{N}}^\mathsf{T} \\end{pmatrix}`

    :math:`\mathbf{\Gamma} = \\begin{pmatrix} \ddots \\\ & \mathbf{\Gamma}_\ell  \\\ &  & \mathbf{\Gamma}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix}`

    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_\ell  \\\ &  & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix}`

    :math:`\mathbf{D} = \mathbf{D}_{\hat{N}} \mathbf{D}_{\hat{N}-1} \\dots \mathbf{D}_{1}`

    where :math:`N = \\sum_\ell N_\ell`, :math:`L = L_1`, and :math:`\\tilde{N} = N_{-1}`.
    Above the index :math:`-1` refers to the last index of the list.

    The systems in the list are chained in the order they are listed.

    Parameters
    ----------
    analog_systems: List[:py:class:`cbadc.analog_system.AnalogSystem`]
        a list of analog systems to be chained.

    Returns
    -------
    :py:class:`cbadc.analog_system.AnalogSystem`
        a new analog system
    """
    N: int = np.sum(np.array([analog_system.N for analog_system in analog_systems]))
    M: int = np.sum(np.array([analog_system.M for analog_system in analog_systems]))
    M_tilde: int = np.sum(
        np.array([analog_system.M_tilde for analog_system in analog_systems])
    )
    L: int = analog_systems[0].L

    A = np.zeros((N, N))
    B = np.zeros((N, L))
    CT = np.zeros((analog_systems[0].CT.shape[0], N))

    CT[
        : analog_systems[0].CT.shape[0], : analog_systems[0].CT.shape[1]
    ] = analog_systems[0].CT

    Gamma = np.zeros((N, M))
    Gamma_tilde = np.zeros((M_tilde, N))
    D = np.eye(analog_systems[0].N_tilde, analog_systems[0].L)

    n: int = 0
    m: int = 0
    m_tilde: int = 0
    for analog_system in analog_systems:
        n_end = n + analog_system.N
        m_end = m + analog_system.M
        m_tilde_end = m_tilde + analog_system.M_tilde

        A[n:n_end, :] = np.dot(analog_system.B, CT)
        A[n:n_end, n:n_end] = analog_system.A

        B[n:n_end, :] = np.dot(analog_system.B, D)

        D = np.dot(analog_system.D, D)

        CT = np.dot(analog_system.D, CT)
        CT[:, n:n_end] = analog_system.CT

        Gamma[n:n_end, m:m_end] = analog_system.Gamma

        Gamma_tilde[m_tilde:m_tilde_end, n:n_end] = analog_system.Gamma_tildeT

        n += analog_system.N
        m += analog_system.M
        m_tilde += analog_system.M_tilde
    return AnalogSystem(A, B, CT, Gamma, Gamma_tilde, D)


def stack(analog_systems: List[AnalogSystem]) -> AnalogSystem:
    """Construct an analog system by stacking several analog systems in parallel.

    The parallel stack is achieved by constructing

    :math:`\mathbf{A} = \\begin{pmatrix}\ddots \\\ & \mathbf{A}_\ell \\\ & &  \mathbf{A}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times N}`

    :math:`\mathbf{B} = \\begin{pmatrix} \ddots \\\ & \mathbf{B}_\ell \\\ & &  \mathbf{B}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times L}`

    :math:`\mathbf{C}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \mathbf{C}_\ell \\\ & &  \mathbf{C}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{\\tilde{N} \\times N}`

    :math:`\mathbf{\Gamma} = \\begin{pmatrix} \ddots \\\ & \mathbf{\Gamma}_\ell  \\\ &  & \mathbf{\Gamma}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times M}`

    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_\ell  \\\ &  & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{\\tilde{M} \\times N}`

    :math:`\mathbf{D} = \\begin{pmatrix}\\ddots \\\ & \mathbf{D}_\ell \\\ && \\ddots\\end{pmatrix}`

    where for :math:`n` systems
    :math:`L = \\sum_{\ell = 1}^n L_\ell`, :math:`M = \\sum_{\ell = 1}^n M_\ell`,
    :math:`\\tilde{M} = \\sum_{\ell = 1}^n \\tilde{M}_\ell`, :math:`N = \\sum_{\ell = 1}^n N_\ell`,
    and :math:`\\tilde{N} = \\sum_{\ell = 1}^n \\tilde{N}_\ell`.

    Parameters
    ----------
    analog_systems: List[:py:class:`cbadc.analog_system.AnalogSystem`]
        a list of analog systems to be chained.

    Returns
    -------
    :py:class:`cbadc.analog_system.AnalogSystem`
        a new analog system
    """
    N: int = np.sum(np.array([analog_system.N for analog_system in analog_systems]))
    M: int = np.sum(np.array([analog_system.M for analog_system in analog_systems]))
    M_tilde: int = np.sum(
        np.array([analog_system.M_tilde for analog_system in analog_systems])
    )
    L: int = np.sum(np.array([analog_system.L for analog_system in analog_systems]))
    N_tilde: int = np.sum(
        np.array([analog_system.N_tilde for analog_system in analog_systems])
    )

    A = np.zeros((N, N))
    B = np.zeros((N, L))
    CT = np.zeros((N_tilde, N))
    Gamma = np.zeros((N, M))
    Gamma_tilde = np.zeros((M_tilde, N))
    D = np.zeros((N_tilde, L))

    n: int = 0
    m: int = 0
    m_tilde: int = 0
    l: int = 0
    n_tilde = 0
    for analog_system in analog_systems:
        n_end = n + analog_system.N
        m_end = m + analog_system.M
        m_tilde_end = m_tilde + analog_system.M_tilde
        l_end = l + analog_system.L
        n_tilde_end = n_tilde + analog_system.N_tilde

        A[n:n_end, n:n_end] = analog_system.A
        B[n:n_end, l:l_end] = analog_system.B
        CT[n_tilde:n_tilde_end, n:n_end] = analog_system.CT
        Gamma[n:n_end, m:m_end] = analog_system.Gamma
        Gamma_tilde[m_tilde:m_tilde_end, n:n_end] = analog_system.Gamma_tildeT

        D[n_tilde:n_tilde_end, l:l_end] = analog_system.D

        l += analog_system.L
        n += analog_system.N
        n_tilde += analog_system.N_tilde
        m += analog_system.M
        m_tilde += analog_system.M_tilde
    return AnalogSystem(A, B, CT, Gamma, Gamma_tilde, D)


def zpk2abcd(z, p, k):
    """Convert zeros and poles into A, B, C, D matrix

    Futhermore, the transfer function is divided into
    sequences of products of Biquad filters.

    Specifically, for a transfer function

    :math:`k \\cdot \\frac{(s - z_N)\\dots(s - z_1)}{(s-p_N)\\dots(s-p_1)}`

    we partition the transfer function into biquadratic blocks as

    :math:`\\Pi_{\ell=1}^{N / 2} \\frac{Z(s)_\ell}{(s - p_{1,\ell})(s-p_{2,\ell})}`

    where
    :math:`Z(s)_\ell = \\begin{cases} 1 & \\text{if no zeros are specified} \\\  (s - z_{1,\ell})(s - z_{2,\ell}) & \\text{for real valued zeros}  \\\ (s-z_{1,\ell})(s-\\bar{z}_{1,\ell}) & \\text{for complex conjugate zero-pairs.} \\end{cases}`


    The poles are and zeros are sorted in the following order:

    1. Complex conjugate pairs
    2. Decreasing absolute magnitude

    """
    if len(z) > len(p) or len(p) < 1:
        raise Exception("Incorrect specification. can't have more zeros than poles.")

    # Sort poles and zeros
    p = _sort_by_complex_descending(p)
    if len(z) > 0:
        z = _sort_by_complex_descending(z)

    k_per_state = np.float64(np.power(np.float64(np.abs(k)), 1.0 / np.float64(len(p))))

    index = 0
    systems = []
    while index < len(p):
        D = np.array([[0.0]])
        if index + 1 < len(p):
            # Two poles
            A = np.zeros((2, 2))
            B = k_per_state**2 * np.array([[1.0], [0.0]])
            CT = np.zeros((1, 2))
            D = np.array([[0.0]])

            pole_1, pole_2 = p[index], p[index + 1]
            # If complex conjugate pole pairs
            if np.allclose(pole_1, np.conjugate(pole_2)):
                a1, b1 = np.real(pole_1), np.imag(pole_1)
                a2, b2 = np.real(pole_2), np.imag(pole_2)
                A = np.array([[a1, b1], [b2, a2]])
            else:
                if np.imag(pole_1) != 0 or np.imag(pole_2) != 0:
                    raise Exception("Can't have non-conjugate complex poles")
                A = np.array([[np.real(pole_1), 0], [1.0, np.real(pole_2)]])

            if index < len(z):
                zero_1 = z[index]
                if index + 1 < len(z):
                    # Two zeros left
                    zero_2 = z[index + 1]
                    y = np.array(
                        [
                            [zero_1 + zero_2 - A[0, 0] - A[1, 1]],
                            [zero_1 * zero_2 - A[0, 0] * A[1, 1] + A[0, 1] * A[1, 0]],
                        ]
                    )
                    if not np.allclose(np.imag(y), np.zeros(2)):
                        raise Exception("Can't have non-conjugate complex zeros")
                    M = np.array([[-1.0, 0], [-A[1, 1], A[1, 0]]])
                    sol = np.linalg.solve(M, np.real(y))
                    D = k_per_state**2 * np.array([[1.0]])
                    CT = np.array([[sol[0, 0], sol[1, 0]]])
                else:
                    # Single zero
                    if np.imag(zero_1) != 0:
                        raise Exception("Can't have non-conjugate complex zero")
                    c1 = 1.0
                    c2 = (A[1, 1] - np.real(zero_1)) / A[1, 0]
                    CT = np.array([[c1, c2]])
            else:
                # No zero
                #
                # gain
                CT = 1.0 / A[1, 0] * np.array([[0.0, 1.0]])

            index += 2
        else:
            # Only one pole and possibly zero left
            pole = p[index]
            if np.imag(pole) != 0:
                raise Exception("Can't have non-conjugate complex poles")
            A = np.array([[np.real(pole)]])
            B = np.array([[k_per_state]])
            CT = np.array([[1.0]])
            D = np.array([[0.0]])
            if index < len(z):
                zero = z[index]
                if np.imag(zero) != 0:
                    raise Exception("Cant have non-conjugate complex zeros")
                D[0, 0] = k_per_state
                CT[0, 0] = pole - np.real(zero)
            index += 1
        systems.append(AnalogSystem(A, B, CT, None, None, D))
    chained_system = chain(systems)
    return chained_system.A, chained_system.B, chained_system.CT, chained_system.D


def _sort_by_complex_descending(list: np.ndarray) -> np.ndarray:
    sorted_indexes = np.argsort(np.abs(np.imag(list)))[::-1]
    list = list[sorted_indexes]
    complex_indexes = np.imag(list) != 0
    number_of_complex_poles = np.sum(complex_indexes)
    if not _complex_conjugate_pairs(list[complex_indexes]):
        raise Exception("Not complex conjugate pairs")
    sorted = np.zeros_like(list)
    sorted[:number_of_complex_poles] = list[complex_indexes]
    list[number_of_complex_poles:] = list[complex_indexes != True]
    return list


def _complex_conjugate_pairs(list: np.ndarray) -> bool:
    index = 0
    while index + 1 < len(list):
        if not np.allclose(list[index], list[index + 1].conjugate()):
            return False
        index += 2
    return True


def sos2abcd(sos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform a series of biquad (second order systems (sos)) filters into
    their A,B,C,D state space model equivalent.

    Specifcally, for a filter with the transfer function

    :math:`\Pi_{\ell = 1}^{L} \\frac{b_{\ell,0}s^2 + b_{\ell,1} s + b_{\ell,2}}{s^2 + a_{\ell,1}s + a_{\ell,2}}`

    represented in a sos matrix

    :math:`\\text{sos} = \\begin{pmatrix} & & \\vdots  \\\ b_{\ell,0}, & b_{\ell,1}, & b_{\ell,2}, & 1, & a_{\ell,1}, & a_{\ell,2} \\\ & & \\vdots \\end{pmatrix}`

    is represented in a controllable canonical state space representation form

    :math:`\\begin{pmatrix} \dot{x}_{\ell, 1}(t) \\\ \dot{x}_{\ell,2}(t) \\end{pmatrix} = \\begin{pmatrix}1 , & 0 \\\ -a_{\ell,2}, & -a_{\ell,1} \\end{pmatrix} \\begin{pmatrix} x_{\ell,1}(t) \\\ x_{\ell,2}(t)\\end{pmatrix} + \\begin{pmatrix} 0 \\\ 1 \\end{pmatrix} u_\ell(t)`

    :math:`y_\ell(t) = \\begin{pmatrix}b_{\ell,2} - b_{\ell,0} * a_{\ell,2}, & b_{\ell,1} - b_{\ell,0}*a_{\ell,1} \\end{pmatrix} \\begin{pmatrix} x_{\ell,1}(t) \\\ x_{\ell,2}(t) \\end{pmatrix}  + \\begin{pmatrix}b_{\ell,0}\\end{pmatrix} u_\ell(t)`

    which are then chained together using :py:func:`cbadc.analog_system.chain`.

    Parameters
    ----------
    sos: np.ndarray, shape(N, 6)
        biquad equations

    Returns
    A: numpy.ndarray, shape=(2 * L, 2 * L)
        a joint state transition matrix.
    B: numpy.ndarray, shape=(2 * L, 1)
        a joint input matrix.
    C: numpy.ndarray, shape=(1, 2 * L)
        a joint signal observation matrix.
    D: numpy.ndarray, shape=(1, 1)
        The direct matrix.
    """
    biquadratic_analog_systems = []
    for row in range(sos.shape[0]):
        b_0 = sos[row, 0]
        b_1 = sos[row, 1]
        b_2 = sos[row, 2]
        a_0 = sos[row, 3]
        a_1 = sos[row, 4]
        a_2 = sos[row, 5]

        if a_0 != 1.0:
            b_0 /= a_0
            b_1 /= a_0
            b_2 /= a_0
            a_1 /= a_0
            a_2 /= a_0
            a_0 = 1.0

        A = np.array([[0.0, 1.0], [-a_2, -a_1]])
        B = np.array([[0.0], [1.0]])
        CT = np.array([[(b_2 - b_0 * a_2), (b_1 - b_0 * a_1)]])
        D = np.array([[b_0]])
        biquadratic_analog_systems.append(AnalogSystem(A, B, CT, None, None, D))
    chained_system = chain(biquadratic_analog_systems)
    return chained_system.A, chained_system.B, chained_system.CT, chained_system.D


def tf2abcd(
    b: np.ndarray, a: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform a transferfunctions into a controllable canonical state space form.

    Specifcally, for a filter with the transfer function

    :math:`\\frac{\Sigma_{\ell = 0}^{L} b_{\ell}s^\ell}{s^{L} + \Sigma_{\ell=0}^{L-1} a_{k} s^\ell}`

    is represented in a controllable canonical state space representation form

    :math:`\dot{\mathbf{x}}(t) = \\begin{pmatrix}0, & 1, & 0 \\\ & \\ddots & \ddots \\\ 0 & \\dots &  0, & 1 \\\ -a_{0}, & \\dots, &  & -a_{L-1} \\end{pmatrix} \mathbf{x}(t) + \\begin{pmatrix} 0 \\\ \\vdots \\\ 0 \\\ 1 \\end{pmatrix} u(t)`

    :math:`y_\ell(t) = \\begin{pmatrix}b_{0} - b_L a_0, & \\dots, & b_{L-1} - b_L a_{L-1} \\end{pmatrix} \mathbf{x}(t) + \\begin{pmatrix} b_L\\end{pmatrix} u(t)`

    Parameters
    ----------
    b: np.ndarray, shape=(L+1,)
        transferfunction nominator :math:`\\begin{pmatrix}b_{L}, & \\dots, & b_{0}\\end{pmatrix}`.
    a: np.ndarray, shape=(L,)
        transferfunction denominator :math:`\\begin{pmatrix}a_{L-1}, & \\dots, & a_{0}\\end{pmatrix}`.

    Returns
    A: numpy.ndarray, shape=(L, L)
        a joint state transition matrix.
    B: numpy.ndarray, shape=(L, 1)
        a joint input matrix.
    CT: numpy.ndarray, shape=(1,L)
        a joint signal observation matrix.
    D: numpy.ndarray, shape=(1,1)
        direct transition matrix.
    """
    if b.size != (a.size + 1) or len(b) > b.size or len(a) > a.size:
        raise Exception(f"a and b are not correctly configures with b={b} and a={a}")
    L = a.size
    A = np.zeros((a.size, a.size))
    B = np.zeros((a.size, 1))
    CT = np.zeros((1, a.size))
    D = np.zeros((1, 1))
    A[:-1, 1:] = np.eye(L - 1)
    A[-1, :] = -a[::-1]
    B[-1, 0] = 1.0
    CT[0, :] = (b[1::] - b[0] * a[::])[::-1]
    D[0, 0] = b[0]
    return A, B, CT, D


def ctsd2abc(ctsd_dict: Dict, T):
    """Build A, B and CT matrices from a dictionary describing a continuous-time delta-sigma modulator.

    Parameters
    ----------
    path: Dict
        Dictionary describing the continuous-time delta-sigma modulator. The dictionary is assumed to have the same format as the json files exported from `www.sigma-delta.de <www.sigma-delta.de>`_ . The coefficients are assumed to be normalized to the sampling frequency, and will be scaled according to the specified control period T.
    T: float
        Desired control period, used for scaling the coefficients of the system.

    Returns
    A: numpy.ndarray, shape=(N, N)
        state transition matrix.
    B: numpy.ndarray, shape=(N, 1)
        input matrix.
    CT: numpy.ndarray, shape=(N,N)
        signal observation matrix.
    """
    N = ctsd_dict['systemOptions']['systemOrder']
    coefficients = ctsd_dict['coefficient']
    fs = 1 / T
    A = np.zeros((N, N))
    B = np.zeros((N, 1))
    CT = np.eye(N)
    # Build A, B. Multiply by fs
    for n1 in range(N):
        B[n1, 0] = _get_coeff_value(coefficients, f'b{n1+1}') * fs
        if f'c{n1}' in coefficients:
            A[n1, n1 - 1] = _get_coeff_value(coefficients, f'c{n1}') * fs
        for n2 in range(N):
            if f'e{n2+1}{n1+1}' in coefficients:
                A[n1, n2] = _get_coeff_value(coefficients, f'e{n2+1}{n1+1}') * fs
            if f'd{n2+1}{n1+1}' in coefficients:
                A[n1, n2] = _get_coeff_value(coefficients, f'd{n2+1}{n1+1}') * fs
    return A, B, CT


def ctsd2gamma(ctsd_dict: Dict, T, dac_scale, local_control=True):
    """Build Gamma and Gamma_tildeT matrices from a dictionary describing a continuous-time delta-sigma modulator.
    Keep the scaling from the CTSD. Choose between local and global control, and the number of quuantizer levels.

    Parameters
    ----------
    path: Dict
        Dictionary describing the continuous-time delta-sigma modulator. The dictionary is assumed to have the same format as the json files exported from `www.sigma-delta.de <www.sigma-delta.de>`_ . The coefficients are assumed to be normalized to the sampling frequency, and will be scaled according to the specified control period T.
    T: float
        Desired control period, used for scaling the coefficients of the system.
    local_control: Bool
        Whether or not to use local control. With local control, the scaling is kept for the Gamma matrix, but Gamma_tildeT is set to the identity matrix

    Returns
    Gamma: numpy.ndarray, shape=(N, M)
        Control input matrix
    Gamma_tildeT: numpy.ndarray, shape=(M_tilde, N)
        Control observation matrix
    D_tilde: numpy.ndarray, shape=(1,1)
        signal observation matrix.
    """
    N = ctsd_dict['systemOptions']['systemOrder']
    M = N if local_control else 1
    M_tilde = M
    fs = 1 / T


    coefficients = ctsd_dict['coefficient']
    # Build Gamma, multiply by fs and dac_scale
    Gamma = np.zeros((N, M))
    for n1 in range(N):
        n2 = n1 if local_control else 0
        Gamma[n1, n2] = _get_coeff_value(coefficients, f'a{n1+1}') * fs * dac_scale
    # Build Gamma_tildeT and D_tilde
    if local_control:
        # Ignore observation from ctsd dict if local control
        D_tilde = np.zeros((M_tilde, 1))
        Gamma_tildeT = np.eye(M)
    else:
        # Try to pick last c value for gamma tilde, use 1 as default
        # Gamma_tilde is a pure scale factor, no integration
        Gamma_tildeT = np.zeros((M_tilde, N))
        Gamma_tildeT[0, -1] = _get_coeff_value(coefficients, f'c{N}', default=1)
        # Build D_tilde
        D_tilde = np.zeros((M_tilde, 1))
        D_tilde[-1, 0] = _get_coeff_value(coefficients, f'b{N+1}')

    return Gamma, Gamma_tildeT, D_tilde


def _get_coeff_value(coefficients: Dict, coeff_name: str, default: float=0):
    """ Return value if the coefficient exist and is active, else default """
    try:
        return coefficients[coeff_name]['fixedValue']
    except KeyError:
        return default


def ctsd2af(ctsd_dict: Dict, T, dac_scale, local_control=False, qlev=None):
    """Construct an analog system and a digital control based on a dictionary describing a continuous-time delta-sigma modulator.
    .
        Parameters
        ----------
        path: Dict
            Dictionary describing the continuous-time delta-sigma modulator. The dictionary is assumed to have the same format as the json files exported from `www.sigma-delta.de <www.sigma-delta.de>`_  The coefficients are assumed to be normalized to the sampling frequency, and will be scaled according to the specified control period T.
        T: float
            Desired control period, used for scaling the coefficients of the system.
        dac_scale: float
            Additional scaling to apply to the DAC coefficients.
        local_control: Bool
            Whether or not to use local control. Default: False
        qlev: Int
            Number of quantizer levels. If None, use the number of levels from the ctsd dictionary. Default: None

        Returns
        analog_system: :py:class:`cbadc.analog_system.analog_system.AnalogSystem`
            Analog System
        digital_control: :py:class:`cbadc.digital_control.DigitalControl`
            Digital Control
    """
    # Parse matrices
    A, B, CT = ctsd2abc(ctsd_dict, T)
    Gamma, Gamma_tildeT, D_tilde = ctsd2gamma(ctsd_dict, T, dac_scale, local_control)
    (M_tilde, _) = Gamma_tildeT.shape
    (_, M) = Gamma.shape
    # Init AS
    AS = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT, D_tilde=D_tilde)
    # Init DC
    Qlevels = ctsd_dict['quantizer']['level'] if qlev is None else qlev
    number_of_levels = [Qlevels for i in range(M_tilde)]
    DC = MultiLevelDigitalControl(Clock(T), M, number_of_levels=number_of_levels)
    return AS, DC

