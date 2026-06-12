"""SNR measurement helpers for the data-aided readout.

The control-bounded readout is data-aided: a FIR is calibrated on a *known*,
persistently-exciting reference (a full-scale random sequence) and then estimates
the input from the control signals. Three complementary measurements cover the
testbench needs:

* :func:`snr_tone`     -- leakage-free single-tone SNR by time-domain projection.
* :func:`snr_residual` -- band-averaged SNR against the known reference.
* :func:`snr_vs_frequency` -- SNR(f) shape from the reference / error spectra.

``snr_tone`` (projection) replaces a windowed FFT + peak pick, whose sidelobes
cap the readable SNR near ~30 dB; the projection is exact and reads past 120 dB.

Note that ``snr_tone`` and ``snr_vs_frequency`` are *distinct* metrics: the tone
projection divides the tone power by the *total* residual power, while SNR(f) is
a *local* power-spectral-density ratio. Use ``snr_residual`` for a single headline
number and ``snr_vs_frequency`` to see how reconstruction quality varies across
the band. For a valid SNR(f) the reference must be flat past the band of interest.

Coherent sampling
-----------------
``snr_tone`` is leakage-free, so it does *not* require coherent sampling. But any
*FFT-bin* method (a windowed spectrum, or the legacy
:meth:`AnalogFrontend.calculateSNR_from_fft`) does: an off-grid tone smears
across bins through the window and is undercounted, which can suppress the
reported SNR by tens of dB. When you plot a spectrum or use a bin method, place
the tone on an exact FFT bin with :func:`coherent_frequency`::

    f = coherent_frequency(f_target, fs, n)   # integer cycles in n samples
"""

from math import gcd

import numpy as np
from scipy.signal import welch

from .digital_backend import decimate as _decimate

__all__ = [
    "decimate",
    "coherent_frequency",
    "snr_tone",
    "snr_residual",
    "snr_vs_frequency",
]


def decimate(x: np.ndarray, DSR: int, axis: int = 0, ftype: str = "iir", n: int = 9):
    """Anti-alias decimate by ``DSR`` (thin wrapper with testbench defaults)."""
    return _decimate(x, DSR, axis=axis, ftype=ftype, n=n)


def coherent_frequency(f_target: float, fs: float, n: int) -> float:
    """Nearest frequency to ``f_target`` that is coherent in an ``n``-sample record.

    A coherent tone completes an integer number of cycles in the record, so it
    lands on a single FFT bin and produces no spectral leakage. The cycle count
    is nudged to be coprime with ``n`` where possible, so the samples visit
    distinct phases (avoiding a tone that trivially repeats).

    Parameters
    ----------
    f_target : float
        the desired tone frequency [Hz].
    fs : float
        the sample rate [Hz].
    n : int
        the number of samples in the record.

    Returns
    -------
    float
        a coherent frequency near ``f_target`` [Hz].

    Examples
    --------
    >>> import numpy as np
    >>> fs, n = 1e3, 1 << 12
    >>> f = coherent_frequency(50.0, fs, n)
    >>> cycles = f * n / fs            # integer number of cycles in the record
    >>> bool(np.isclose(cycles, round(cycles)))
    True
    """
    k = max(1, int(round(f_target * n / fs)))
    if gcd(k, n) != 1:
        for dk in range(1, max(2, k)):
            if k - dk >= 1 and gcd(k - dk, n) == 1:
                k -= dk
                break
            if gcd(k + dk, n) == 1:
                k += dk
                break
    return k * fs / n


def _as_columns(x: np.ndarray, trim: int) -> np.ndarray:
    """Drop ``trim`` samples from each end of axis 0 and return shape (m, C).

    Axis 0 is time; any trailing axes (L, J) become independent columns, so the
    measurement is per-channel / per-parallel-sequence rather than flattened.
    ``trim`` should be just the reconstruction transient (≈ the filter length),
    not a tuning knob -- the SNR is otherwise insensitive to it.
    """
    x = np.asarray(x)
    if trim:
        x = x[trim : x.shape[0] - trim]
    return x.reshape(x.shape[0], -1)


def snr_tone(
    u_hat: np.ndarray, f_sig: float, fs: float, trim: int = 0, band: float = None
) -> float:
    """Leakage-free single-tone SNR via time-domain projection.

    Projects ``u_hat`` onto cos/sin at ``f_sig``; the residual is the noise +
    distortion. With ``band`` set, only the residual power below ``band`` [Hz]
    counts as noise (in-band SNR).

    Unlike an FFT-bin method this is leakage-free and needs no coherent
    sampling, but if you also plot a spectrum pick ``f_sig`` with
    :func:`coherent_frequency` for a clean single-bin tone.

    Parameters
    ----------
    u_hat : numpy.ndarray
        the reconstructed signal; axis 0 is time, trailing axes (L, J) are
        measured independently and the result is averaged over them.
    f_sig : float
        the tone frequency [Hz].
    fs : float
        the sample rate of ``u_hat`` [Hz].
    trim : int, optional
        samples to drop from each end -- only the reconstruction transient
        (≈ the filter length); the result is otherwise insensitive to it.
    band : float, optional
        if given, count only residual noise below this frequency [Hz] (in-band
        SNR). The in-band fraction is estimated with Welch averaging, so it is
        robust to whether the tone is coherent with the record.

    Returns
    -------
    float
        SNR in dB (mean over columns if more than one).

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
    cols = _as_columns(u_hat, trim)
    m = cols.shape[0]
    t = np.arange(m) / fs
    c = np.cos(2 * np.pi * f_sig * t)
    s = np.sin(2 * np.pi * f_sig * t)
    snrs = []
    for y in cols.T:
        a = 2.0 * np.mean(y * c)
        b = 2.0 * np.mean(y * s)
        resid = y - (a * c + b * s)
        sig_p = (a * a + b * b) / 2.0
        if band is not None:
            fr, P = welch(resid, fs=fs, nperseg=min(1 << 12, m))
            in_frac = P[fr <= band].sum() / P.sum()
            noise_p = np.mean(resid**2) * in_frac
        else:
            noise_p = np.mean(resid**2)
        snrs.append(10 * np.log10(sig_p / noise_p) if noise_p > 0 else np.inf)
    return float(np.mean(snrs))


def snr_residual(u_hat: np.ndarray, u_ref: np.ndarray, trim: int = 0) -> float:
    """Broadband SNR against a known reference: var(ref) / var(u_hat - ref).

    Use with the calibration reference, where ``u_ref`` is the reference that was
    fitted -- no tone needed, the whole band is exercised at once. Trailing axes
    (L, J) are averaged over.
    """
    a = _as_columns(u_hat, trim)
    b = _as_columns(u_ref, trim)
    snrs = []
    for ai, bi in zip(a.T, b.T):
        noise = np.var(ai - bi)
        snrs.append(10 * np.log10(np.var(bi) / noise) if noise > 0 else np.inf)
    return float(np.mean(snrs))


def snr_vs_frequency(
    u_hat: np.ndarray,
    u_ref: np.ndarray,
    fs: float,
    nperseg: int = 1 << 12,
    trim: int = 0,
):
    """SNR as a function of frequency from reference / error power spectra.

    With a white reference this directly shows how reconstruction quality
    varies across the band.

    Trailing axes (L, J) are pooled into the spectral averaging.

    Returns
    -------
    (f, snr_f) : (numpy.ndarray, numpy.ndarray)
        frequencies [Hz] and SNR(f) [dB].
    """
    a = _as_columns(u_hat, trim)
    b = _as_columns(u_ref, trim)
    nperseg = min(nperseg, a.shape[0])
    # average the reference / error power spectra over all columns
    f, P_ref = welch(b, fs=fs, nperseg=nperseg, axis=0)
    _, P_err = welch(a - b, fs=fs, nperseg=nperseg, axis=0)
    P_ref = P_ref.mean(axis=-1)
    P_err = P_err.mean(axis=-1)
    snr_f = 10 * np.log10(P_ref / np.maximum(P_err, 1e-300))
    return f, snr_f
