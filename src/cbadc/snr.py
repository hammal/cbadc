"""Measure how much wanted signal survives reconstruction or conversion.

SNR is wanted-signal power divided by unwanted power. This module offers two
ways to identify the wanted signal: compare a reconstructed input with a known
reference, or measure a known sinusoidal tone in a converter output. The
reference methods (:func:`snr_residual`, :func:`snr_vs_frequency`) suit
broadband inputs. The tone methods (:func:`snr_tone`, :func:`fft_sndr`) suit
spectral tests; ``fft_sndr`` also separates harmonics and reports SNDR.

The file starts with record preparation: :func:`decimate` reduces the sample
rate of reconstructed signals, and :func:`coherent_frequency` places a test
tone on an FFT bin. Measurement functions follow. Finally,
:func:`simulate_snr` creates an amplitude sweep with :mod:`cbadc.simulation`
and measures the decisions, digital twin, and Wiener input estimate.
The reconstruction itself belongs to :mod:`cbadc.estimator`."""

from copy import copy, deepcopy
from math import gcd
from typing import NamedTuple, cast

import numpy as np
from scipy.signal import firwin, kaiserord, upfirdn, welch

from .frontend import DAC, Frontend, Samples, Sinusoid, StateSpace
from .simulation import simulate

# Bound numerically zero FFT residuals to a finite, float64-scale ratio.
_FFT_SNDR_CEILING_DB = 250.0
_FFT_RATIO_FLOOR = 10.0 ** (-_FFT_SNDR_CEILING_DB / 10.0)

__all__ = [
    "decimate",
    "decimation_transient",
    "coherent_frequency",
    "snr_tone",
    "fft_sndr",
    "FFTSndr",
    "snr_residual",
    "ResidualDiagnostics",
    "snr_vs_frequency",
    "measure_snr",
    "snr_sigma",
    "simulate_snr",
]


# Record preparation for the measurements below.
def _decimation_factors(DSR: int) -> list:
    """Split the downsampling ratio into anti-alias filter stages.

    Parameters
    ----------
    DSR : int
        Positive integer decimation ratio.

    Returns
    -------
    list of int
        Small per-stage decimation ratios whose product is ``DSR``.
    """
    factors = []
    for p in (2, 3, 5, 7):
        while DSR % p == 0:
            factors.append(p)
            DSR //= p
    return factors + ([DSR] if DSR > 1 else [])


def _stage_fir(r: int, pb: float, sb_final: float, atten_dB: float):
    """Build one anti-alias FIR stage for a decimated SNR measurement.

    Parameters
    ----------
    r : int
        Positive decimation factor for this stage.
    pb : float
        Passband edge normalized to the current Nyquist.
    sb_final : float
        Final stopband edge normalized to the current Nyquist.
    atten_dB : float
        Stopband attenuation in dB.

    Returns
    -------
    taps : ndarray of float, shape (tap_count,)
        Kaiser FIR coefficients.
    half : int
        Integer group delay in output samples.
    """
    sb = max(2.0 / r - sb_final, sb_final + 1e-3)
    ntaps, beta = kaiserord(atten_dB, max(sb - pb, 1e-6))
    half = int(np.ceil((ntaps - 1) / (2 * r)))  # integer group delay in output samples
    return firwin(2 * r * half + 1, 0.5 * (pb + sb), window=("kaiser", beta)), half


def decimation_transient(
    DSR: int, atten_dB: float = 160.0, pass_frac: float = 0.8, stop_frac: float = 1.0
) -> int:
    """Leading filtered samples to discard before measuring SNR.

    Parameters
    ----------
    DSR : int
        Positive integer decimation ratio.
    atten_dB : float, optional
        Kaiser stopband attenuation in dB.
        Defaults to ``160.0``.
    pass_frac : float, optional
        Passband edge as a fraction of final Nyquist.
        Defaults to ``0.8``.
    stop_frac : float, optional
        Stopband edge as a fraction of final Nyquist.
        Defaults to ``1.0``.

    Returns
    -------
    int
        Number of leading decimated samples affected by the record boundary.
    """
    if DSR < 2:
        return 0
    done, length = 1, 1
    for r in _decimation_factors(DSR):
        taps, _ = _stage_fir(
            r, pass_frac / DSR * done, stop_frac / DSR * done, atten_dB
        )
        length += (len(taps) - 1) * done
        done *= r
    return int(np.ceil(length / DSR))


def decimate(
    x: np.ndarray,
    DSR: int,
    axis: int = 0,
    atten_dB: float = 160.0,
    pass_frac: float = 0.8,
    stop_frac: float = 1.0,
) -> np.ndarray:
    """Anti-alias and downsample a record by ``DSR`` before measurement.

    Each stage is a Kaiser FIR with ``atten_dB`` stopband attenuation whose
    passband and stopband edges are ``pass_frac`` and ``stop_frac`` of the final
    Nyquist; the group delay is trimmed so the output is aligned with the input.

    Parameters
    ----------
    x : array_like, shape (size, ...) when axis is 0
        Sampled signal; other axes are independent channels.
    DSR : int
        Positive integer decimation ratio.
    axis : int, optional
        Axis containing clock samples.
        Defaults to ``0``.
    atten_dB : float, optional
        Kaiser stopband attenuation in dB.
        Defaults to ``160.0``.
    pass_frac : float, optional
        Passband edge as a fraction of final Nyquist.
        Defaults to ``0.8``.
    stop_frac : float, optional
        Stopband edge as a fraction of final Nyquist.
        Defaults to ``1.0``.

    Returns
    -------
    ndarray, shape with axis length size // DSR
        Filtered and downsampled signal, aligned to the input.
    """
    DSR = int(np.ceil(DSR))
    if DSR <= 1:
        return x
    x = np.moveaxis(np.asarray(x), axis, 0)
    shape = x.shape
    y = np.ascontiguousarray(x.reshape(shape[0], -1))
    done = 1
    for r in _decimation_factors(DSR):
        taps, half = _stage_fir(
            r, pass_frac / DSR * done, stop_frac / DSR * done, atten_dB
        )
        y = upfirdn(taps, y, up=1, down=r, axis=0)[half : half + y.shape[0] // r]
        done *= r
    n = shape[0] // DSR
    return np.moveaxis(y[:n].reshape((n,) + shape[1:]), 0, axis)


def coherent_frequency(f_target: float, fs: float, n: int) -> float:
    """Place a test tone on an FFT bin to avoid spectral leakage.

    Prefer a cycle count coprime to ``n`` near the target frequency.

    Parameters
    ----------
    f_target : float
        Desired tone frequency in Hz.
    fs : float
        Positive sampling frequency in Hz.
    n : int
        Positive FFT record length in samples.

    Returns
    -------
    float
        Nearest coherent FFT-bin frequency in Hz.
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


def _trim(x: np.ndarray, trim: int) -> np.ndarray:
    """Trim each time series and reshape to a two-dimensional array.

    Axis 0 is time; each combination of trailing indices remains a separate
    series. For example, ``(size, J, L)`` becomes ``(size - 2*trim, J*L)``.
    ``trim`` should be just the reconstruction transient (≈ the filter length),
    not a tuning knob -- the SNR is otherwise insensitive to it.

    Parameters
    ----------
    x : ndarray, shape (size, ...)
        Time samples with arbitrary trailing channel axes.
    trim : int
        Samples removed from each end of the record.

    Returns
    -------
    ndarray, shape (size - 2 * trim, number_of_series)
        Trimmed samples; number_of_series is the product of the trailing
        dimensions, or 1 for a one-dimensional input. No series are mixed.
    """
    x = np.asarray(x)
    if trim:
        x = x[trim : x.shape[0] - trim]
    if x.shape[0] < 1:
        raise ValueError(
            f"no samples left after trimming {trim} from each end; "
            "use a longer record or reduce trim."
        )
    return x.reshape(x.shape[0], -1)


def _welch_ref_err(a: np.ndarray, b: np.ndarray, fs: float, nperseg: int):
    """Reference / error power spectra averaged over time series.

    Returns ``(f, P_ref, P_err)`` with the spectra averaged over the series.
    When ``fs`` is 2.0 the frequency axis runs 0..1 (fraction of Nyquist).

    Parameters
    ----------
    a, b : ndarray of float, shape (size, number_of_series)
        Reconstruction and reference with matching shapes.
    fs : float
        Sampling frequency in Hz or 2 for normalized Nyquist units.
    nperseg : int
        Welch segment length in samples.

    Returns
    -------
    f, P_ref, P_err : ndarray of float, shape (frequency_bins,)
        Frequencies and pooled reference and error power spectra.
    """
    nperseg = min(nperseg, a.shape[0])
    f, P_ref = welch(b, fs=fs, nperseg=nperseg, axis=0)
    _, P_err = welch(a - b, fs=fs, nperseg=nperseg, axis=0)
    return f, P_ref.mean(axis=-1), P_err.mean(axis=-1)


def _band_mask(f: np.ndarray, band) -> np.ndarray:
    """Bins of ``f`` inside ``band`` (same units as ``f``). The DC bin is dropped
    when the lower edge is 0 (Welch's DC bin is biased and usually signal-free).

    ``band`` is a scalar upper edge ``f_hi`` -> ``[0, f_hi]`` or a pair
    ``(f_lo, f_hi)``.

    Parameters
    ----------
    f : ndarray of float, shape (frequency_bins,)
        Frequency axis.
    band : float or tuple of float, shape (2,)
        Upper edge or lower and upper edges in the units of ``f``.

    Returns
    -------
    ndarray of bool, shape (frequency_bins,)
        Mask selecting in-band spectral bins.
    """
    lo, hi = (0.0, band) if np.isscalar(band) else (band[0], band[1])
    mask = (f >= lo) & (f <= hi)
    if lo <= 0.0:
        mask &= f > 0.0
    if mask.sum() < 8:
        import warnings

        warnings.warn(
            f"sub-band [{lo}, {hi}] covers only {int(mask.sum())} spectral bins; "
            "use a larger nperseg or a wider band for a stable estimate.",
            stacklevel=2,
        )
    return mask


# Measurements from a known tone or a known reference.
def snr_tone(
    u_hat: np.ndarray, f_sig: float, fs: float, trim: int = 0, band: float = None
) -> float:
    """Leakage-free tone SNR from a cosine/sine/DC least-squares projection.

    Axis 0 is time; trailing axes are pooled. trim drops samples at both
    ends. band, when given in Hz, restricts residual noise power using Welch.
    Unlike fft_sndr, coherent sampling is unnecessary.

    Parameters
    ----------
    u_hat : array_like of float, shape (size, ...)
        Reconstructed samples; trailing axes are pooled.
    f_sig : float
        Tone frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    trim : int, optional
        Number of samples dropped from each record end.
        Defaults to ``0``.
    band : float, optional
        Upper noise-band edge in Hz.
        Defaults to ``None``.

    Returns
    -------
    float
        Pooled tone signal-to-noise ratio in dB.
    """
    u_hat = _trim(u_hat, trim)  # (size, number_of_series), e.g. (size, J*L)
    size = u_hat.shape[0]
    t = np.arange(size) / fs
    c = np.cos(2 * np.pi * f_sig * t)
    s = np.sin(2 * np.pi * f_sig * t)
    # Fit cosine, sine, and a constant to handle off-grid tones and remove DC.
    basis = np.stack([c, s, np.ones(size)], axis=1)  # (size, 3)
    theta, *_ = np.linalg.lstsq(basis, u_hat, rcond=None)  # (3, number_of_series)
    a, b = theta[0], theta[1]
    resid = u_hat - basis @ theta  # (size, number_of_series)
    sig_p = np.mean((a**2 + b**2) / 2.0)  # mean tone power over trailing axes
    if band is not None:
        fr, P = welch(resid, fs=fs, nperseg=min(1 << 12, size), axis=0)
        P = P.mean(axis=-1)  # pool the residual spectra over trailing axes
        in_frac = P[fr <= band].sum() / P.sum()
        noise_p = np.mean(resid**2) * in_frac  # pooled in-band noise power
    else:
        noise_p = np.mean(resid**2)
    return 10 * np.log10(sig_p / noise_p) if noise_p > 0 else np.inf


class FFTSndr(NamedTuple):
    """Coherent-FFT metrics and the unnormalised one-sided power spectrum.

    The dB values are scalars for one time series or arrays over trailing axes.

    Parameters
    ----------
    sndr_dB, snr_dB, sfdr_dBc : float or ndarray, shape (...)
        Spectral ratios for each batch member and channel, in dB or dBc.
    hd_dBc : ndarray, shape (H_max - 1, ...)
        Harmonic distortion levels in dBc.
    spectrum : ndarray, shape (size // 2 + 1, ...)
        Unnormalized one-sided power spectrum.

    Returns
    -------
    FFTSndr
        Tuple containing spectral ratios, harmonic levels, and the power spectrum.
    """

    sndr_dB: np.ndarray
    """signal-to-noise-and-distortion ratio: P_fund / Σ(all other in-band)."""
    snr_dB: np.ndarray
    """signal-to-noise ratio: harmonics excluded from the denominator."""
    sfdr_dBc: np.ndarray
    """spurious-free dynamic range: P_fund / largest single non-fundamental bin."""
    hd_dBc: np.ndarray
    """per-harmonic levels HD2..HD_Hmax [dBc], shape ``(H_max - 1, ...)``."""
    spectrum: np.ndarray
    """one-sided power spectrum ``|rfft|**2``, shape ``(size//2+1, ...)``."""


def fft_sndr(
    u_hat: np.ndarray,
    f_sig,
    fs: float,
    *,
    band=None,
    extra_bins: int = 2,
    n_dc: int = 3,
    H_max: int = 9,
) -> FFTSndr:
    """Coherent rectangular-FFT SNDR, SNR, SFDR and harmonic levels.

    Choose f_sig with coherent_frequency; an off-bin tone leaks into noise.
    Axis 0 is time; each batch member and channel is measured separately.
    f_sig may be scalar or one frequency per time series. band is an upper
    edge or (low, high) pair in Hz. extra_bins widens the fundamental; n_dc excludes DC/idle
    bins; H_max is the last harmonic reported. Aliased harmonics are included.

    Parameters
    ----------
    u_hat : array_like of float, shape (size, ...)
        Reconstructed samples; trailing axes are measured separately.
    f_sig : float or array_like of float, shape (number_of_series,)
        Coherent tone frequency in Hz, scalar or one per time series in
        flattened trailing-axis order, e.g. ``J*L`` for ``(size, J, L)``.
    fs : float
        Sampling frequency in Hz.
    band : float or tuple of float, shape (2,), optional
        Upper edge or lower and upper frequency edges in Hz.
        Defaults to ``None``.
    extra_bins : int, optional
        Number of FFT bins added on each side of the fundamental.
        Defaults to ``2``.
    n_dc : int, optional
        Number of low-frequency FFT bins excluded.
        Defaults to ``3``.
    H_max : int, optional
        Highest harmonic order reported.
        Defaults to ``9``.

    Returns
    -------
    FFTSndr
        SNDR, SNR, SFDR, harmonic levels, and spectrum for each batch member
        and channel, preserving the input's trailing shape.
    """
    x = np.asarray(u_hat)
    size = x.shape[0]
    u_hat = x.reshape(size, -1)  # (size, number_of_series), e.g. (size, J*L)
    number_of_series = u_hat.shape[1]
    f_sig = np.broadcast_to(
        np.asarray(f_sig, dtype=float).reshape(-1), (number_of_series,)
    )

    spec = np.abs(np.fft.rfft(u_hat, axis=0)) ** 2
    Nf = spec.shape[0]
    freqs = np.fft.rfftfreq(size, d=1.0 / fs)  # (Nf,)

    if band is None:
        in_band = np.ones(Nf, dtype=bool)
    else:
        lo, hi = (0.0, band) if np.isscalar(band) else (band[0], band[1])
        in_band = (freqs >= lo) & (freqs <= hi)
    in_band &= np.arange(Nf) > n_dc

    sndr = np.empty(number_of_series)
    snr = np.empty(number_of_series)
    sfdr = np.empty(number_of_series)
    hd = np.empty((H_max - 1, number_of_series))

    for c in range(number_of_series):
        P = spec[:, c]
        k = int(round(f_sig[c] * size / fs))
        fund = np.zeros(Nf, dtype=bool)
        lo_k = max(0, k - extra_bins)
        hi_k = min(Nf - 1, k + extra_bins)
        fund[lo_k : hi_k + 1] = True

        P_fund = P[fund & in_band].sum()
        harm = np.zeros(Nf, dtype=bool)
        hd_c = np.full(H_max - 1, -np.inf)
        for i, h in enumerate(range(2, H_max + 1)):
            kh = (h * k) % size
            b = min(kh, size - kh)  # fold aliased harmonic into the one-sided FFT
            bins = np.arange(max(0, b - 1), min(Nf - 1, b + 1) + 1)
            bins = bins[in_band[bins] & ~fund[bins]]
            harm[bins] = True
            ph = P[bins].sum()
            if P_fund > 0 and ph > 0:
                hd_c[i] = 10 * np.log10(ph / P_fund)

        other = in_band & ~fund  # everything in-band but the fundamental
        P_other = P[other].sum()
        P_noise = P[other & ~harm].sum()  # noise = in-band, non-fund, non-harm
        # Clamp numerical zero to the FFT ratio floor; zero fundamental is undefined.
        spur = P[other]
        if P_fund > 0.0:
            floor = P_fund * _FFT_RATIO_FLOOR
            sndr[c] = 10.0 * np.log10(P_fund / max(P_other, floor))
            snr[c] = 10.0 * np.log10(P_fund / max(P_noise, floor))
            sfdr[c] = 10.0 * np.log10(
                P_fund / max(spur.max() if spur.size else 0.0, floor)
            )
        else:
            sndr[c] = snr[c] = sfdr[c] = np.nan
        hd[:, c] = hd_c

    # Restore the batch and channel axes; one time series returns scalar ratios.
    tail = x.shape[1:]
    if tail == () or (len(tail) and int(np.prod(tail)) == 1):
        return FFTSndr(
            float(sndr[0]), float(snr[0]), float(sfdr[0]), hd[:, 0], spec[:, 0]
        )
    return FFTSndr(
        sndr.reshape(tail),
        snr.reshape(tail),
        sfdr.reshape(tail),
        hd.reshape((H_max - 1, *tail)),
        spec.reshape((Nf, *tail)),
    )


class ResidualDiagnostics(NamedTuple):
    """Pooled correlation, fitted gain and gain-invariant SNR in dB.

    For linear ratios, 1/SNR_res = (g - 1)**2 + g**2/SNR_rho.

    Parameters
    ----------
    rho : float
        Pooled normalized cross-correlation of reconstruction and reference.
    g : float
        Pooled best-fit linear gain.
    snr_rho : float
        Gain-invariant correlation SNR in dB.

    Returns
    -------
    ResidualDiagnostics
        Tuple containing correlation, fitted gain, and gain-invariant SNR.
    """

    rho: float
    """normalised cross-correlation between û and the reference (pooled)."""
    g: float
    """best-fit scalar gain ``g = <û, ref> / ||ref||**2`` (pooled)."""
    snr_rho: float
    """gain-invariant correlation SNR ``rho**2 / (1 - rho**2)`` in dB."""


def snr_residual(
    u_hat: np.ndarray,
    u_ref: np.ndarray,
    trim: int = 0,
    band=None,
    fs: float = None,
    nperseg: int = 1 << 12,
    return_diagnostics: bool = False,
):
    """Broadband SNR: reference variance / reconstruction-error variance.

    Axis 0 is time; full-band results are averaged over trailing axes.
    trim removes samples at both ends. band selects Welch spectral bins:
    an upper edge or (low, high) pair. With fs, edges are in Hz;
    otherwise fractions of Nyquist. nperseg controls Welch segments.
    return_diagnostics also returns pooled full-band correlation and gain.
    Use a held-out reference sequence.

    Parameters
    ----------
    u_hat, u_ref : array_like of float, shape (size, ...)
        Reconstruction and held-out reference with matching shapes.
    trim : int, optional
        Samples removed from each record end.
        Defaults to ``0``.
    band : float or tuple of float, shape (2,), optional
        Upper edge or lower and upper spectral edges.
        Defaults to ``None``.
    fs : float, optional
        Sampling frequency in Hz; when absent, band edges use Nyquist units.
        Defaults to ``None``.
    nperseg : int, optional
        Welch segment length in samples for sub-band measurements.
        Defaults to ``1 << 12``.
    return_diagnostics : bool, optional
        Also return pooled correlation, gain, and gain-invariant SNR.
        Defaults to ``False``.

    Returns
    -------
    snr_dB : float
        Residual SNR in dB.
    diagnostics : ResidualDiagnostics, optional
        Returned only when ``return_diagnostics`` is true.
    """
    a = _trim(u_hat, trim)
    b = _trim(u_ref, trim)
    if band is None:
        # Full band: average the time-domain variance ratios over trailing axes.
        snrs = []
        for ai, bi in zip(a.T, b.T):
            noise = np.var(ai - bi)
            snrs.append(10 * np.log10(np.var(bi) / noise) if noise > 0 else np.inf)
        snr_db = float(np.mean(snrs))
    else:
        # sub-band: integrate the pooled reference / error power spectra
        f, P_ref, P_err = _welch_ref_err(a, b, fs if fs is not None else 2.0, nperseg)
        mask = _band_mask(f, band)
        den = P_err[mask].sum()
        snr_db = 10 * np.log10(P_ref[mask].sum() / den) if den > 0 else np.inf
    if not return_diagnostics:
        return snr_db
    return snr_db, _residual_diagnostics(a, b)


def _residual_diagnostics(a: np.ndarray, b: np.ndarray) -> ResidualDiagnostics:
    """Pooled, mean-removed correlation and best-fit gain.

    Parameters
    ----------
    a, b : ndarray of float, shape (size, number_of_series)
        Reconstruction and reference with matching shapes.

    Returns
    -------
    ResidualDiagnostics
        Pooled correlation, fitted gain, and gain-invariant SNR.
    """
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    cross = float(np.sum(a * b))  # <û, ref>, pooled
    e_ref = float(np.sum(b * b))  # ||ref||**2, pooled
    e_hat = float(np.sum(a * a))  # ||û||**2, pooled
    g = cross / e_ref if e_ref > 0 else np.inf
    denom = e_hat * e_ref
    rho = cross / np.sqrt(denom) if denom > 0 else 0.0
    rho = float(np.clip(rho, -1.0, 1.0))
    one_minus = 1.0 - rho * rho
    snr_rho = 10 * np.log10(rho * rho / one_minus) if one_minus > 0 else np.inf
    return ResidualDiagnostics(rho=rho, g=g, snr_rho=snr_rho)


def snr_vs_frequency(
    u_hat: np.ndarray,
    u_ref: np.ndarray,
    fs: float,
    nperseg: int = 1 << 12,
    trim: int = 0,
):
    """Reference/error Welch PSD ratio in dB; trailing axes are pooled.

    Parameters
    ----------
    u_hat, u_ref : array_like of float, shape (size, ...)
        Reconstruction and reference with matching shapes.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Welch segment length in samples.
        Defaults to ``1 << 12``.
    trim : int, optional
        Samples removed from each record end.
        Defaults to ``0``.

    Returns
    -------
    f : ndarray of float, shape (frequency_bins,)
        Welch frequency axis in Hz.
    snr_dB : ndarray of float, shape (frequency_bins,)
        Reference-to-error PSD ratio in dB.
    """
    a = _trim(u_hat, trim)
    b = _trim(u_ref, trim)
    f, P_ref, P_err = _welch_ref_err(a, b, fs, nperseg)
    snr_f = 10 * np.log10(P_ref / np.maximum(P_err, 1e-300))
    return f, snr_f


def snr_sigma(n_eff: float, band=None, fs: float = None) -> float:
    """Approximate 1-sigma finite-record SNR uncertainty in dB.

    n_eff is the number of independent pooled samples. A band reduces the
    effective sample count by its fraction of Nyquist; specify fs for Hz
    edges, otherwise use fractions of Nyquist. Numerical saturation near
    the float64 SNR ceiling is excluded.

    Parameters
    ----------
    n_eff : float
        Effective number of independent pooled samples.
    band : float or tuple of float, shape (2,), optional
        Upper edge or lower and upper spectral edges.
        Defaults to ``None``.
    fs : float, optional
        Sampling frequency in Hz; when absent, edges use Nyquist units.
        Defaults to ``None``.

    Returns
    -------
    float
        Approximate one-sigma SNR uncertainty in dB.
    """
    n_eff = float(n_eff)
    if n_eff <= 0:
        return np.inf
    if band is not None:
        nyquist = (fs / 2.0) if fs is not None else 1.0
        lo, hi = (0.0, band) if np.isscalar(band) else band
        fraction = (min(hi, nyquist) - max(lo, 0.0)) / nyquist
        if fraction <= 0:
            return np.inf
        n_eff *= fraction
    return (10.0 / np.log(10.0)) * np.sqrt(2.0 / n_eff)


def measure_snr(
    u_hat: np.ndarray,
    u_ref: np.ndarray = None,
    *,
    method: str = "residual",
    band=None,
    fs: float = None,
    f_sig: float = None,
    trim: int = 0,
    nperseg: int = 1 << 12,
):
    """Dispatch to residual, tone or frequency-dependent SNR.

    method='residual' uses u_ref; 'tone' needs f_sig and fs; 'snr_f'
    returns (frequencies, SNR). band, trim, fs and nperseg are forwarded
    where applicable.

    Parameters
    ----------
    u_hat : array_like of float, shape (size, ...)
        Reconstructed sample record.
    u_ref : array_like of float, shape (size, ...), optional
        Held-out reference, required for residual and frequency methods.
        Defaults to ``None``.
    method : {"residual", "tone", "snr_f"}, optional
        Measurement to compute.
        Defaults to ``"residual"``.
    band : float or tuple of float, shape (2,), optional
        Upper edge or lower and upper spectral edges.
        Defaults to ``None``.
    fs : float, optional
        Sampling frequency in Hz, required for tone and frequency methods.
        Defaults to ``None``.
    f_sig : float, optional
        Tone frequency in Hz, required for the tone method.
        Defaults to ``None``.
    trim : int, optional
        Samples removed from each record end.
        Defaults to ``0``.
    nperseg : int, optional
        Welch segment length in samples.
        Defaults to ``1 << 12``.

    Returns
    -------
    float or tuple of ndarray
        Scalar SNR in dB for residual or tone methods; ``(frequencies, SNR)`` arrays for ``snr_f``.
    """
    if method == "residual":
        return snr_residual(u_hat, u_ref, trim=trim, band=band, fs=fs, nperseg=nperseg)
    if method == "tone":
        if f_sig is None or fs is None:
            raise ValueError("method='tone' requires f_sig and fs")
        return snr_tone(u_hat, f_sig, fs, trim=trim, band=band)
    if method == "snr_f":
        return snr_vs_frequency(u_hat, u_ref, fs, nperseg=nperseg, trim=trim)
    raise ValueError(f"unknown method {method!r}; use 'residual', 'tone', or 'snr_f'")


# End-to-end tone sweep of a Frontend.
def simulate_snr(
    af: Frontend,
    OSR: float,
    amp=None,
    *,
    full_scale: float,
    f: float | None = None,
    k: int = 13,
    warmup: int = 1024,
    readouts: tuple[str, ...] = ("v", "y_hat", "u_hat"),
    eta2: float | None = None,
    input_index: int = 0,
    readout_index: int = 0,
    engine: str = "numba",
    batch_size: int | None = None,
    seed: int | None = 98123591265830293457639481,
) -> dict:
    """Sweep a Frontend's input level and measure its in-band SNDR.

    Like DELSIG's ``simulateSNR``, sweep coherent sine amplitudes in dBFS.
    Return ``amp``, ``snr_v`` (decisions), ``snr_y_hat`` (raw twin), and
    ``snr_u_hat`` (Wiener input estimate), all from the same simulation.
    Each SNDR array has shape ``(J, amp.size)`` and includes harmonics.
    ``arguments`` captures every argument, including a frontend copy and the
    resolved tone, regularization, and seed. Replay with
    ``simulate_snr(**result["arguments"])``.
    Wiener reconstruction uses :meth:`Frontend.wiener_filter` with ``OSR``
    setting one nominal regularization shared over J, unless ``eta2`` is given.
    It requires K=1 and unfiltered DAC pulses contained in one clock period.
    Unsupported models can use ``readouts=("y_hat",)``.

    The frontend is deep-copied. Each batch member sees every amplitude: one
    simulation runs ``J * amp.size`` batch members by default, ordered first by
    member and then by amplitude. The ``J`` circuit maps are discretized once;
    amplitude changes only the sine source's initial state. State noise follows
    each member. Wiener coefficients are also computed once per circuit.
    ``batch_size`` optionally splits this expanded batch across runs. Measure
    ``2**k`` coherent samples with a periodic Hann window, below ``fs/(2*OSR)``.
    Discard ``warmup`` steps at both ends for the two-sided Wiener filter.
    Increase warmup to check transient convergence;
    no analytical startup correction is applied.

    Parameters
    ----------
    af : Frontend
        Converter to test; its selected input is replaced only in a copy.
    OSR : float
        Positive oversampling ratio defining the signal band.
    amp : array_like of float, shape (amplitudes,), optional
        Input levels in dBFS. Defaults to DELSIG's -120 to 0 dBFS sweep.
    full_scale : float
        Positive peak input amplitude represented by 0 dBFS, in input units.
    f : float, optional
        Test-tone frequency in Hz, rounded to a coherent FFT bin. Defaults
        to one-quarter of the signal-band edge.
    k : int, optional
        Log2 of the measured record length; defaults to 13.
    warmup : int, optional
        Clock steps discarded at each end of the record; defaults to 1024.
    readouts : tuple of str, optional
        Readouts to measure; defaults to ``("v", "y_hat", "u_hat")``.
        Select any nonempty subset for fewer measurements.
    eta2 : float, optional
        Wiener regularization variance in squared input units. Defaults to
        ``None``: derive it from OSR and batch member 0, as in
        :meth:`Frontend.wiener_filter`. Used only when measuring ``u_hat``.
    input_index : int, optional
        Input channel receiving the sinusoid and selected in ``u_hat``;
        defaults to 0.
    readout_index : int, optional
        Channel to measure in ``v`` and ``y_hat``; defaults to 0.
        For MASH, ``v`` measures this stage alone, without digital cancellation.
    engine : {"numba", "numpy"}, optional
        Simulation engine; defaults to ``"numba"``.
    batch_size : int, optional
        Maximum expanded batch width per simulation. Defaults to ``None``:
        simulate all ``J * amp.size`` batch members together.
    seed : int or None, optional
        Noise seed for the first run; later chunks use distinct seeds.
        Defaults to ``98123591265830293457639481``; ``None`` draws a fresh base
        seed, saved in ``arguments``. Each chunk uses base seed + batch start.

    Returns
    -------
    result : dict
        ``amp`` has shape (amplitudes,) and contains the tested dBFS levels.
        ``snr_v``, ``snr_y_hat``, and ``snr_u_hat`` have shape (J, amplitudes)
        and contain in-band SNDR in dB for the requested readouts. Entry [j, a]
        is member j at amp[a], even when J=1. Unrequested measurements are
        omitted. ``arguments`` is a dict of keyword arguments for replay;
        it stores settings and a frontend snapshot, not simulated records.
    """
    if not np.isfinite(OSR) or OSR <= 0:
        raise ValueError("OSR must be positive and finite")
    if not np.isfinite(full_scale) or full_scale <= 0:
        raise ValueError("full_scale must be positive and finite")
    if k < 4 or warmup < 0:
        raise ValueError("k must be at least 4 and warmup nonnegative")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive or None")
    if not 0 <= input_index < af.L:
        raise IndexError("input_index is outside the frontend's L inputs")
    if (
        not readouts
        or len(set(readouts)) != len(readouts)
        or any(name not in ("v", "y_hat", "u_hat") for name in readouts)
    ):
        raise ValueError(
            "readouts must select 'v', 'y_hat', or 'u_hat', without duplicates"
        )
    M_hat = 1 if af.C_hat is None else af.C_hat.shape[-2]
    if "y_hat" in readouts and not 0 <= readout_index < M_hat:
        raise IndexError("readout_index is outside the y_hat readouts")
    if "v" in readouts and not 0 <= readout_index < af.M:
        raise IndexError("readout_index is outside the v decisions")

    amp = (
        np.r_[np.arange(-120, -19, 10), -15, np.arange(-10, 1)]
        if amp is None
        else np.asarray(amp, dtype=float).reshape(-1)
    )
    if amp.size == 0 or not np.all(np.isfinite(amp)):
        raise ValueError("amp must contain finite dBFS levels")
    size, fs = 1 << k, af.fs
    band = fs / (2 * OSR)
    tone = coherent_frequency(band / 4 if f is None else f, fs, size)
    if not 2 < tone * size / fs and tone < band:
        raise ValueError("tone needs at least three FFT bins and must lie in-band")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)

    levels = full_scale * 10 ** (amp / 20)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    base = deepcopy(af)
    base_sources = list(cast(list, base.input))
    base_sources[input_index] = Sinusoid(1.0, tone, -np.pi / 2)
    base.input = base_sources
    base_dz = base.discretize()  # amplitude changes only the sine's x0
    wiener = base.wiener_filter(eta2=eta2, OSR=OSR) if "u_hat" in readouts else None

    def measure(
        members: np.ndarray, amplitudes: np.ndarray, start: int
    ) -> dict[str, np.ndarray]:
        work = deepcopy(af)
        seen: set[int] = set()

        # For any J-batched array X, expanded X[(j, a)] = X[j].
        def repeat_batch(array: np.ndarray) -> np.ndarray:
            return array if array.shape[0] == 1 else array[members % array.shape[0]]

        def repeat_system(system: StateSpace) -> None:
            if id(system) in seen:
                return
            seen.add(id(system))
            system.A, system.B, system.C, system.D = map(repeat_batch, system.abcd)
            system.x0 = repeat_batch(system.x0)
            if isinstance(system, Samples):
                system.values = repeat_batch(system.values.T).T
            if isinstance(system, DAC) and system.filter is not None:
                repeat_system(system.filter)

        repeat_system(work)
        if work.state_covariance is not None:
            work.state_covariance = repeat_batch(work.state_covariance)
        if work.C_hat is not None:
            work.C_hat = repeat_batch(np.asarray(work.C_hat))
        if work.D_hat is not None:
            work.D_hat = repeat_batch(np.asarray(work.D_hat))
        sources = list(cast(list, work.input))
        for i, source in enumerate(sources):
            if i != input_index:
                repeat_system(source)
        for pulse in cast(list, work.dac):
            repeat_system(pulse)
        sources[input_index] = Sinusoid(levels[amplitudes], tone, -np.pi / 2)
        work.input = sources
        dz = base_dz._repeat_sine_batch(work, members, levels[amplitudes], input_index)
        result = simulate(
            work,
            size + 2 * warmup,
            None,
            ("v", "y_hat"),
            engine,
            None,
            False,
            seed + start,
            discretization=dz,
        )
        signals = {"v": result["v"], "y_hat": result["y_hat"]}
        if wiener is not None:
            estimator = copy(wiener)
            if wiener.J_m > 1:
                for name in ("W", "A_f", "B_f", "A_b", "B_b"):
                    setattr(estimator, name, repeat_batch(getattr(wiener, name)))
            signals["u_hat"] = estimator(result["v"], method="state-space")
        # Hann spreads the coherent tone over three bins and DC over bins 0, 1.
        return {
            f"snr_{name}": np.atleast_1d(
                fft_sndr(
                    signals[name][
                        warmup : warmup + size,
                        input_index if name == "u_hat" else readout_index,
                    ]
                    * window[:, None],
                    tone,
                    fs,
                    band=band,
                    extra_bins=1,
                    n_dc=1,
                ).sndr_dB
            )
            for name in readouts
        }

    total = af.J * amp.size
    width = total if batch_size is None else batch_size
    members = np.repeat(np.arange(af.J), amp.size)
    amplitudes = np.tile(np.arange(amp.size), af.J)
    snr = {f"snr_{name}": np.empty(total) for name in readouts}
    for start in range(0, total, width):
        stop = min(start + width, total)
        for name, values in measure(
            members[start:stop], amplitudes[start:stop], start
        ).items():
            snr[name][start:stop] = values
    return {
        "amp": amp.copy(),
        **{name: values.reshape(af.J, amp.size) for name, values in snr.items()},
        "arguments": dict(
            af=deepcopy(af),
            OSR=OSR,
            amp=amp.copy(),
            full_scale=full_scale,
            f=tone,
            k=k,
            warmup=warmup,
            readouts=tuple(readouts),
            eta2=wiener.eta2 if wiener is not None else eta2,
            input_index=input_index,
            readout_index=readout_index,
            engine=engine,
            batch_size=batch_size,
            seed=seed,
        ),
    }
