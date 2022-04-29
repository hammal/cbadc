import scipy.constants


def resistor_noise_voltage_source_model(R, T=3e2):
    """Johnson-Nyquist 4kTR noise as voltage source

    Parameters
    ----------
    R: `float`
        the resistor value in Ohms
    T: `float`, `optional`
        temperature expressed in Kelvin, defaults to 300.

    Returns
    -------
    : float
        the one-sided power spectral density (V^2/Hz)
    """
    return 4 * scipy.constants.Boltzmann * T * R


def resistor_noise_current_source_model(R, T=3e2):
    """Johnson-Nyquist 4kTR noise as current source

    Parameters
    ----------
    R: `float`
        the resistor value in Ohms
    T: `float`, `optional`
        temperature expressed in Kelvin, defaults to 300.

    Returns
    -------
    : float
        the one-sided power spectral density (I^2/Hz)
    """
    return resistor_noise_voltage_source_model(R, T) / (R**2)


def kTC_noise_voltage_source(C: float, T: float = 3e2):
    """PSD for KTC noise

    Parameters
    ----------
    C: `float`
        the capacitance value in Farads
    T: `float`, `optional`
        temperature expressed in Kelvin, defaults to 300.

    Returns
    -------
    : float
        the one-sided power spectral density (V^2/Hz)
    """
    return scipy.constants.Boltzmann * T / C
