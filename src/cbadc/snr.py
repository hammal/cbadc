"""SNR measurement helpers for the data-aided readout.

The control-bounded readout is data-aided: a FIR is calibrated on a *known*,
persistently-exciting reference (a full-scale random sequence) and then estimates
the input from the control signals. Three complementary measurements cover the
testbench needs:

* :func:`snr_tone`     -- leakage-free single-tone SNR by time-domain projection.
* :func:`snr_residual` -- broadband SNR against the known reference realisation.
* :func:`snr_vs_frequency` -- SNR(f) shape from the reference / error spectra.

``snr_residual`` is the broadband analog of the tone projection: where
``snr_tone`` correlates the reconstruction against a *known tone*, ``snr_residual``
correlates it against a *known full-band reference realisation* and reports
``var(ref) / var(û - ref)``. It is a **known-reference correlation** measurement.

``snr_tone`` (projection) replaces a windowed FFT + peak pick, whose sidelobes
cap the readable SNR near ~30 dB; the projection is exact and reads past 120 dB.

Note that ``snr_tone`` and ``snr_vs_frequency`` are *distinct* metrics: the tone
projection divides the tone power by the *total* residual power, while SNR(f) is
a *local* power-spectral-density ratio. Use ``snr_residual`` for a single headline
number and ``snr_vs_frequency`` to see how reconstruction quality varies across
the band. For a valid SNR(f) the reference must be flat past the band of interest.

Coherent vs. non-coherent
-------------------------
``snr_residual`` is *coherent in the estimation sense*: the reference
**realisation** (the actual sample sequence) is known and subtracted pointwise,
so it correlates û against that specific waveform. A genuinely *non-coherent*
measurement would know only the input **statistics / PSD** -- not the realisation --
and could not form ``û - ref``. So "leakage-free, no coherent *sampling* needed"
(true of ``snr_tone``) is **not** the same as "non-coherent": all three metrics
here rely on knowing either the tone or the reference realisation.

Practical limits
----------------
* **Float64 ceiling.** ``snr_residual`` is a variance ratio; with float64 the
  error variance cannot fall meaningfully below ~``1e-15`` of the signal
  variance, so the metric saturates near **~150 dB** regardless of the true SNR.
* **Finite-record uncertainty.** An SNR estimate from ``N_eff`` independent
  samples has a 1-sigma spread of roughly ``6.1 / sqrt(N_eff)`` dB; budget the
  record length accordingly before trusting the last few dB.
* **Held-out reference.** For a *trustworthy* number, evaluate on a **held-out**
  reference realisation (a fresh seed), not the calibration-training sequence:
  measuring on the training data is in-sample biased by ≈ ``K / N`` (filter taps
  over samples). This is a measurement recommendation only -- it does not change
  how calibration is performed.

Coherent sampling (FFT-bin methods)
----------------------------------
``snr_tone`` is leakage-free, so it does *not* require coherent sampling. But any
*FFT-bin* method (a windowed spectrum, or the legacy
:meth:`AnalogFrontend.calculateSNR_from_fft`) does: an off-grid tone smears
across bins through the window and is undercounted, which can suppress the
reported SNR by tens of dB. When you plot a spectrum or use a bin method, place
the tone on an exact FFT bin with :func:`coherent_frequency`::

    f = coherent_frequency(f_target, fs, n)   # integer cycles in n samples
"""

from math import gcd
from typing import NamedTuple

import numpy as np
from scipy.signal import welch

from .digital_backend import decimate as _decimate

__all__ = [
    "decimate",
    "coherent_frequency",
    "snr_tone",
    "snr_residual",
    "ResidualDiagnostics",
    "snr_vs_frequency",
    "measure_snr",
]


class ResidualDiagnostics(NamedTuple):
    """Correlation diagnostics for :func:`snr_residual` (all linear, not dB).

    The decomposition lets a user tell *why* ``snr_residual`` is low: a small
    ``rho`` means genuine noise (û and the reference are decorrelated), whereas
    ``rho ≈ 1`` with ``g != 1`` means the reconstruction is merely mis-scaled.
    The two effects combine through the exact identity (noise-to-signal form)::

        1 / SNR_res = (g - 1) ** 2 + g ** 2 / SNR_rho

    with ``SNR_res`` and ``SNR_rho`` taken as *linear* power ratios. ``snr_rho``
    (the dB form of ``SNR_rho = rho**2 / (1 - rho**2)``) is gain-invariant: it is
    the SNR you would read after the optimal scalar gain correction.
    """

    rho: float
    """normalised cross-correlation between û and the reference (pooled)."""
    g: float
    """best-fit scalar gain ``g = <û, ref> / ||ref||**2`` (pooled)."""
    snr_rho: float
    """gain-invariant correlation SNR ``rho**2 / (1 - rho**2)`` in dB."""


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
    if x.shape[0] < 1:
        raise ValueError(
            f"no samples left after trimming {trim} from each end; the record "
            "(or each parallel column) is too short -- use a longer record or "
            "fewer parallel sequences (J)."
        )
    return x.reshape(x.shape[0], -1)


def _welch_ref_err(a: np.ndarray, b: np.ndarray, fs: float, nperseg: int):
    """Pooled reference / error power spectra. ``a``, ``b`` are (m, C) columns.

    Returns ``(f, P_ref, P_err)`` with the spectra averaged over the columns.
    When ``fs`` is 2.0 the frequency axis runs 0..1 (fraction of Nyquist).
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
        the reconstructed signal; axis 0 is time. Trailing axes (L, J) are
        treated as independent realisations of the *same* tone and **pooled**:
        the noise estimate uses every sample across all columns, so J short
        sequences give the same SNR as one long record of the same total length
        (each column must still be long enough to resolve the in-band spectrum).
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
    cols = _as_columns(u_hat, trim)  # (m, C)
    m = cols.shape[0]
    t = np.arange(m) / fs
    c = np.cos(2 * np.pi * f_sig * t)
    s = np.sin(2 * np.pi * f_sig * t)
    # per-column tone projection (leakage-free), then pool the residual
    a = 2.0 * np.mean(cols * c[:, None], axis=0)  # (C,)
    b = 2.0 * np.mean(cols * s[:, None], axis=0)
    resid = cols - (a[None, :] * c[:, None] + b[None, :] * s[:, None])  # (m, C)
    sig_p = np.mean((a**2 + b**2) / 2.0)  # mean tone power over columns
    if band is not None:
        fr, P = welch(resid, fs=fs, nperseg=min(1 << 12, m), axis=0)
        P = P.mean(axis=-1)  # pool the residual spectra across columns
        in_frac = P[fr <= band].sum() / P.sum()
        noise_p = np.mean(resid**2) * in_frac  # pooled in-band noise power
    else:
        noise_p = np.mean(resid**2)
    return 10 * np.log10(sig_p / noise_p) if noise_p > 0 else np.inf


def snr_residual(
    u_hat: np.ndarray,
    u_ref: np.ndarray,
    trim: int = 0,
    band=None,
    fs: float = None,
    nperseg: int = 1 << 12,
    return_diagnostics: bool = False,
):
    """Broadband SNR against a known reference: var(ref) / var(u_hat - ref).

    The broadband analog of the tone projection :func:`snr_tone`: a
    *known-reference correlation* measurement. It correlates ``u_hat`` against the
    known reference **realisation** ``u_ref`` (the actual sample sequence), so it
    is coherent in the estimation sense -- the reference is subtracted pointwise,
    not merely matched in statistics. Use it with the calibration reference, where
    ``u_ref`` is the reference that was fitted -- no tone needed, the whole band is
    exercised at once. Trailing axes (L, J) are averaged over.

    .. note::

        Being a float64 variance ratio, the metric saturates near **~150 dB**.
        A finite record of ``N_eff`` samples gives an estimate with a 1-sigma
        spread of ≈ ``6.1 / sqrt(N_eff)`` dB. For a trustworthy number, evaluate
        on a **held-out** reference realisation (fresh seed), not the
        calibration-training one (in-sample bias ≈ ``K / N``). See the module
        docstring for the full discussion.

    Parameters
    ----------
    u_hat, u_ref : numpy.ndarray
        reconstruction and known reference; axis 0 is time, trailing axes pooled.
    trim : int, optional
        samples to drop from each end (the reconstruction transient only).
    band : float or (float, float), optional
        restrict the measurement to a sub-band (a spectral var ratio over the
        selected bins, by Parseval). A scalar is an upper edge ``[0, f_hi]``; a
        pair is a window ``[f_lo, f_hi]`` (use this to exclude the rolloff edge,
        where the converter SNR is worst and the full-band number is dominated).
        ``None`` (default) measures the full band via the exact time-domain
        variance ratio. Units follow ``fs``.
    fs : float, optional
        sample rate [Hz]. If given, ``band`` is in Hz; if omitted, ``band`` is a
        fraction of Nyquist (``f_Nyq == 1``).
    nperseg : int, optional
        Welch segment length for the sub-band path.
    return_diagnostics : bool, optional
        if ``True`` also return a :class:`ResidualDiagnostics` (``rho``, ``g``,
        ``snr_rho``) so a low SNR can be attributed to decorrelation (noise) vs.
        a mere scalar mis-gain. Computed on the full band (the diagnostics are a
        time-domain decomposition); ``band`` still selects the returned ``float``.

    Returns
    -------
    float
        SNR in dB (mean over columns if more than one).
    ResidualDiagnostics
        only if ``return_diagnostics`` -- the correlation decomposition.
    """
    a = _as_columns(u_hat, trim)
    b = _as_columns(u_ref, trim)
    if band is None:
        # full band: exact time-domain variance ratio (per column, averaged)
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
    """Correlation decomposition of ``snr_residual`` (a, b are (m, C) columns).

    Pooled over columns consistently with the full-band variance ratio: the
    cross-correlation, the reference energy and the reconstruction energy are
    summed over columns (mean-removed per column) before forming ``rho`` and the
    best-fit gain ``g = <û, ref> / ||ref||**2``. ``snr_rho`` is the gain-invariant
    SNR ``rho**2 / (1 - rho**2)`` in dB.
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
    f, P_ref, P_err = _welch_ref_err(a, b, fs, nperseg)
    snr_f = 10 * np.log10(P_ref / np.maximum(P_err, 1e-300))
    return f, snr_f


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
    """Single entry point for the SNR metrics, so callers select uniformly.

    Parameters
    ----------
    method : {"residual", "tone", "snr_f"}
        ``"residual"`` (default) -> :func:`snr_residual` (needs ``u_ref``);
        ``"tone"`` -> :func:`snr_tone` (needs ``f_sig`` and ``fs``);
        ``"snr_f"`` -> :func:`snr_vs_frequency` (returns ``(f, snr_f)``).
    band : float or (float, float), optional
        sub-band restriction; orthogonal to ``method`` (passed to residual/tone).
    fs, f_sig, trim, nperseg
        forwarded to the selected metric.
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
