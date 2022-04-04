"""Helper functions for synthesising sigma delta modulators as control-bounded converters."""
import numpy as np
from typing import Dict
from cbadc.analog_system import AnalogSystem
from cbadc.digital_control import MultiLevelDigitalControl
from cbadc.analog_signal.clock import Clock
from cbadc.analog_frontend import AnalogFrontend


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
    Gamma = np.zeros((N, M))
    Gamma_tildeT = np.zeros((M_tilde, N))
    for n1 in range(N):
        n2 = n1 if local_control else 0
        Gamma[n1, n2] = _get_coeff_value(coefficients, f'a{n1+1}') * fs * dac_scale

        # Gamma_tilde is a pure scale factor, no integration
        Gamma_tildeT[0, n1] = _get_coeff_value(coefficients, f'd{n1+1}{N+1}')

    # Build Gamma_tildeT and D_tilde
    # Try to pick last c value for gamma tilde, use 1 as default
    Gamma_tildeT[0, -1] = _get_coeff_value(coefficients, f'c{N}', default=1)
    # Build D_tilde
    D_tilde = np.zeros((M_tilde, 1))
    D_tilde[-1, 0] = _get_coeff_value(coefficients, f'b{N+1}')

    if local_control:
        # Ignore observation from ctsd dict if local control
        D_tilde = np.zeros((M_tilde, 1))
        Gamma_tildeT = np.eye(M)

    return Gamma, Gamma_tildeT, D_tilde


def _get_coeff_value(coefficients: Dict, coeff_name: str, default: float = 0):
    """Return value if the coefficient exist and is active, else default"""
    try:
        return coefficients[coeff_name]['fixedValue']
    except KeyError:
        return default


def ctsd_dict2af(
    ctsd_dict: Dict, T, dac_scale=1.0, local_control=False, qlev=None
) -> AnalogFrontend:
    """Construct an analog system and a digital control based on a dictionary describing a continuous-time delta-sigma modulator.

    Parameters
    ----------
    path: Dict
        Dictionary describing the continuous-time delta-sigma modulator. The dictionary is assumed to have the same format as the json files exported from `www.sigma-delta.de <www.sigma-delta.de>`_  The coefficients are assumed to be normalized to the sampling frequency, and will be scaled according to the specified control period T.
    T: float
        Desired control period, used for scaling the coefficients of the system.
    dac_scale: float, optional
        Additional scaling to apply to the DAC coefficients. Defaults: 1.0
    local_control: Bool, optional
        Whether or not to use local control. Default: False
    qlev: Int, optional
        Number of quantizer levels. If None, use the number of levels from the ctsd dictionary. Default: None

    Returns
    analog_system: :py:class:`cbadc.analog_frontend.AnalogFrontend`
        An analog frontend
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
    return AnalogFrontend(AS, DC)


def ctsd_abcd2af(abcd, T, qlev, dac_scale=1.0, local_control=False) -> AnalogFrontend:
    """Construct an analog system and a digital control from an ABCD state-space matrix. The format of the matrix must follow the conventions of the Richard Schreier's delta-sigma MATLAB Toolbox.
    Parameters
    ----------
    abcd: np.ndarray, shape=(N+1,N+2)
        ABCD state-space matrix describing the continuous-time modulator
    T: float
        Desired control period, used for scaling the coefficients of the system.
    qlev: Int
        Number of quantizer levels. If None, use the number of levels from the ctsd dictionary.
    dac_scale: float, optional
        Additional scaling to apply to the DAC coefficients. Default: 1.0
    local_control: Bool, optional
        Whether or not to use local control. Default: False

    Returns
    analog_system: :py:class:`cbadc.analog_frontend.AnalogFrontend`
        An analog frontend
    """
    ctsd_dict = _schreier_abcd_to_ulm_dict(abcd, qlev)
    return ctsd_dict2af(ctsd_dict, T, dac_scale, local_control, qlev)


def _schreier_abcd_to_ulm_dict(ABCD, qlev):
    """Convert ABCD matrix from schreier MATLAB toolbox to a dictionary with the format of the ULM www.sigma-delta.de"""
    N = ABCD.shape[0] - 1
    ctsd_dict = {}
    ctsd_dict['systemOptions'] = {}
    ctsd_dict['systemOptions']['systemOrder'] = N
    ctsd_dict['quantizer'] = {}
    ctsd_dict['quantizer']['level'] = qlev

    coefficients = {}
    min_val = 1e-8  # Ignore smaller values
    B = ABCD[:, N]  # Input
    A = ABCD[:, N + 1]  # Feedback
    C = ABCD[:, 0:N]  # State evolution

    # Input and feedback
    for n in range(N + 1):
        valB = B[n]
        if np.abs(valB) > min_val:
            name = f'b{n+1}'
            coefficients[name] = {}
            coefficients[name]['fixedValue'] = valB
            coefficients[name]['active'] = True
        valA = A[n]
        if np.abs(valA) > min_val:
            name = f'a{n+1}'
            coefficients[name] = {}
            coefficients[name]['fixedValue'] = valA
            coefficients[name]['active'] = True
    # States
    for m in range(N + 1):
        for n in range(N):
            val = C[m, n]
            if np.abs(val) > min_val:
                # Forward
                if (m - n) == 1:
                    name = f'c{n+1}'
                    coefficients[name] = {}
                    coefficients[name]['fixedValue'] = val
                    coefficients[name]['active'] = True
                elif (m - n) > 1:
                    name = f'd{n+1}{m+1}'
                    coefficients[name] = {}
                    coefficients[name]['fixedValue'] = val
                    coefficients[name]['active'] = True
                elif (m - n) < 0:
                    name = f'e{n+1}{m+1}'
                    coefficients[name] = {}
                    coefficients[name]['fixedValue'] = val
                    coefficients[name]['active'] = True
    ctsd_dict['coefficient'] = coefficients
    return ctsd_dict
