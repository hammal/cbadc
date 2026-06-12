"""SNR measurement helpers for the data-aided readout.

The control-bounded readout is data-aided: a FIR is calibrated on a *known*
reference (white dither) and then estimates the input from the control signals.
Three complementary measurements cover the testbench needs:

* :func:`snr_tone`     -- leakage-free single-tone SNR by time-domain projection.
* :func:`snr_residual` -- band-averaged SNR against a known reference (the dither).
* :func:`snr_vs_frequency` -- SNR(f) shape from the reference / error spectra.

``snr_tone`` (projection) replaces a windowed FFT + peak pick, whose sidelobes
cap the readable SNR near ~30 dB; the projection is exact and reads past 120 dB.

Note that ``snr_tone`` and ``snr_vs_frequency`` are *distinct* metrics: the tone
projection divides the tone power by the *total* residual power, while SNR(f) is
a *local* power-spectral-density ratio. Use ``snr_residual`` for a single headline
number and ``snr_vs_frequency`` to see how reconstruction quality varies across
the band. For a valid SNR(f) the reference must be flat past the band of interest.
"""

import numpy as np
from scipy.signal import welch

from .digital_backend import decimate as _decimate

__all__ = ["decimate", "snr_tone", "snr_residual", "snr_vs_frequency"]


def decimate(x: np.ndarray, DSR: int, axis: int = 0, ftype: str = "iir", n: int = 9):
    """Anti-alias decimate by ``DSR`` (thin wrapper with testbench defaults)."""
    return _decimate(x, DSR, axis=axis, ftype=ftype, n=n)


def _flatten_trim(x: np.ndarray, trim: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    return x[trim : x.size - trim] if trim else x


def snr_tone(
    u_hat: np.ndarray, f_sig: float, fs: float, trim: int = 0, band: float = None
) -> float:
    """Leakage-free single-tone SNR via time-domain projection.

    Projects ``u_hat`` onto cos/sin at ``f_sig``; the residual is the noise +
    distortion. With ``band`` set, only the residual power below ``band`` [Hz]
    counts as noise (in-band SNR).

    Parameters
    ----------
    u_hat : numpy.ndarray
        the reconstructed signal (1-D, or squeezable to 1-D).
    f_sig : float
        the tone frequency [Hz].
    fs : float
        the sample rate of ``u_hat`` [Hz].
    trim : int, optional
        samples to drop from each end (filter edge transient), default 0.
    band : float, optional
        if given, integrate residual noise only below this frequency [Hz].

    Returns
    -------
    float
        SNR in dB.

    Examples
    --------
    >>> import numpy as np
    >>> fs, f = 1e3, 50.0
    >>> t = np.arange(1 << 14) / fs
    >>> rng = np.random.default_rng(0)
    >>> y = np.cos(2 * np.pi * f * t) + 1e-3 * rng.standard_normal(t.size)
    >>> bool(50.0 < snr_tone(y, f, fs) < 70.0)  # ~57 dB for a 1e-3 noise floor
    True
    """
    y = _flatten_trim(u_hat, trim)
    m = y.size
    t = np.arange(m) / fs
    c = np.cos(2 * np.pi * f_sig * t)
    s = np.sin(2 * np.pi * f_sig * t)
    a = 2.0 * np.mean(y * c)
    b = 2.0 * np.mean(y * s)
    resid = y - (a * c + b * s)
    sig_p = (a * a + b * b) / 2.0
    if band is not None:
        R = np.fft.rfft(resid)
        f = np.fft.rfftfreq(m, d=1.0 / fs)
        in_frac = np.sum(np.abs(R[f <= band]) ** 2) / np.sum(np.abs(R) ** 2)
        noise_p = np.mean(resid**2) * in_frac
    else:
        noise_p = np.mean(resid**2)
    return 10 * np.log10(sig_p / noise_p) if noise_p > 0 else np.inf


def snr_residual(u_hat: np.ndarray, u_ref: np.ndarray, trim: int = 0) -> float:
    """Broadband SNR against a known reference: var(ref) / var(u_hat - ref).

    Use with the calibration dither, where ``u_ref`` is the reference that was
    fitted -- no tone needed, the whole band is exercised at once.
    """
    a = _flatten_trim(u_hat, trim)
    b = _flatten_trim(u_ref, trim)
    err = a - b
    noise = np.var(err)
    return 10 * np.log10(np.var(b) / noise) if noise > 0 else np.inf


def snr_vs_frequency(
    u_hat: np.ndarray,
    u_ref: np.ndarray,
    fs: float,
    nperseg: int = 1 << 12,
    trim: int = 0,
):
    """SNR as a function of frequency from reference / error power spectra.

    With a white reference (dither) this directly shows how reconstruction
    quality varies across the band.

    Returns
    -------
    (f, snr_f) : (numpy.ndarray, numpy.ndarray)
        frequencies [Hz] and SNR(f) [dB].
    """
    a = _flatten_trim(u_hat, trim)
    b = _flatten_trim(u_ref, trim)
    err = a - b
    nperseg = min(nperseg, a.size)
    f, P_ref = welch(b, fs=fs, nperseg=nperseg)
    _, P_err = welch(err, fs=fs, nperseg=nperseg)
    snr_f = 10 * np.log10(P_ref / np.maximum(P_err, 1e-300))
    return f, snr_f
