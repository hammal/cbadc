import scipy.constants


def resistor_noise_voltage_source_model(R: float, T: float = 3e2):
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
        the one-sided power spectral density V^2/Hz (Voltage squared per Hertz)
    """
    return 4 * scipy.constants.Boltzmann * T * R


def resistor_sizing_voltage_source(PSD_per_bandwidth: float, T: float = 3e2):
    """Give a resistor value from a V^2/Hz specification

    Parameters
    ----------
    PSD_per_bandwidth: `float`
        the white noise PSD value expressed as V^2/Hz (Voltage squared per Hertz)
    T: `float`, `optional`
        temperature expressed in Kelvin, defaults to 300.
    """
    return PSD_per_bandwidth / (4 * scipy.constants.Boltzmann * T)


def resistor_noise_current_source_model(R: float, T: float = 3e2):
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
        the one-sided power spectral density I^2/Hz (Ampere squared over Hertz)
    """
    return resistor_noise_voltage_source_model(R, T) / (R**2)


def resistor_sizing_current_source(PSD_per_bandwidth: float, T: float = 3e2):
    """Give a resistor value from a I^2/Hz (Ampere squared over Hertz) specification

    Parameters
    ----------
    PSD_per_bandwidth: `float`
        the white noise PSD value expressed as I^2/Hz (Ampere squared over Hertz)
    T: `float`, `optional`
        temperature expressed in Kelvin, defaults to 300.
    """
    return 4 * scipy.constants.Boltzmann * T / PSD_per_bandwidth


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
        the one-sided power spectral density V^2/Hz (Voltage squared per Hertz)
    """
    return scipy.constants.Boltzmann * T / C


def capacitor_sizing_voltage_source(PSD_per_bandwidth: float, T: float = 3e2):
    """Give a capacitive value from a V^2/Hz (Voltage squared over Hertz) specification

    Parameters
    ----------
    PSD_per_bandwidth: `float`
        the white noise PSD value expressed as V^2/Hz (Voltage squared over Hertz)
    T: `float`, `optional`
        temperature expressed in Kelvin, defaults to 300.
    """
    return scipy.constants.Boltzmann * T / PSD_per_bandwidth
