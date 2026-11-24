"""Validate circuit thermal noise with independent ngspice source netlists.

The ordinary circuit netlist injects a factor of the total state covariance.
Here separate ngspice sources inject the resistor, OTA and op-amp currents,
so the physical-source comparison does not reuse that covariance factor.
"""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov

from cbadc import snr
from cbadc.circuit import ActiveRC, GmC
from cbadc.frontend import Frontend, Sinusoid
from cbadc.noise import k_B

from . import spice


def _open_loop(dt=2e-6):
    """A stable state with no DAC path; comparator decisions cannot affect x."""
    return Frontend([[-5e4]], [[0.0, 0.0]], [[1.0]], dt=dt)


def _circuit(realization, source, temperature, capacitance, dt=2e-6):
    af = _open_loop(dt)
    if realization == "gmc":
        return GmC(
            af,
            Cint=capacitance,
            C_design=1e-12,
            Ro=1e7 if source in ("resistor", "combined") else np.inf,
            gamma=1.5 if source in ("ota", "combined") else 0.0,
            temperature=temperature,
        )
    return ActiveRC(
        af,
        Cint=capacitance,
        C_design=1e-12,
        gm=1e-4,
        Ro=np.inf,
        Co=1e-10,
        omega_u=5e4,
        thermal_noise=source in ("resistor", "combined"),
        opamp_current_noise=5e-14 if source in ("current", "combined") else None,
        opamp_voltage_noise=2e-7 if source in ("voltage", "combined") else None,
        temperature=temperature,
    )


def _theory_variance(realization, source, temperature, capacitance):
    """Stationary Pxx from component spectra and the independently written A."""
    design, pole = 1e-12, 5e4
    if realization == "gmc":
        gm = design * pole
        ro = 1e7 if source in ("resistor", "combined") else np.inf
        gamma = 1.5 if source in ("ota", "combined") else 0.0
        decay = (gm + 1 / ro) / capacitance
        q = 2 * k_B * temperature * (gamma * gm + 1 / ro) / capacitance**2
        return q / (2 * decay)
    r_int, omega_t = 1 / (pole * design), 1e6
    A = np.array([[0, 1 / (r_int * capacitance)], [-omega_t, -omega_t]])
    resistor = (
        2 * k_B * temperature / r_int if source in ("resistor", "combined") else 0
    )
    current = (5e-14) ** 2 / 2 if source in ("current", "combined") else 0
    qx = (resistor + current) / capacitance**2
    qp = (omega_t * 2e-7) ** 2 / 2 if source in ("voltage", "combined") else 0
    return solve_continuous_lyapunov(A, -np.diag([qx, qp]))[0, 0]


def _physical_netlist(circuit, size, seed):
    """Replace aggregate ``F w`` with one independent source per component.

    A unit-density ngspice ``trnoise`` voltage has RMS ``1/sqrt(h)`` at
    interval ``h``. Each B row converts it to a current with coefficient
    ``sqrt(S_I/2)`` for one-sided current PSD ``S_I``. For op-amp voltage
    noise, ``Co * omega_t * e_n/sqrt(2)`` injects into the pole capacitor.
    """
    netlist = circuit.netlist(size, seed=seed)
    lines = [
        line
        for line in netlist.splitlines()
        if not (line.startswith("Vw") or line.startswith("Bwn"))
    ]
    h = circuit.dt / 32
    physical = []

    def add(label, node, current_density):
        if current_density > 0:
            physical.extend(
                (
                    f"V{label} n{label} 0 dc 0 trnoise({float(1 / np.sqrt(h))!r} {h!r} 0 0)",
                    f"B{label} 0 x{node} I = {float(current_density)!r}*v(n{label})",
                )
            )

    if isinstance(circuit, ActiveRC):
        if circuit.thermal_noise:
            add(
                "rint",
                0,
                np.sqrt(2 * k_B * circuit.temperature / circuit.R_int[0, 0, 0]),
            )
        add("iin", 0, float(circuit.opamp_current_noise[0, 0, 0]) / np.sqrt(2))
        en = float(circuit.opamp_voltage_noise[0, 0, 0])
        add("en", 1, float(circuit.gm[0, 0, 0]) * en / np.sqrt(2))
    else:
        gm = float(abs(circuit.gm_A[0, 0, 0]).sum() + abs(circuit.gm_B[0, 0, 0]).sum())
        add("ota", 0, np.sqrt(2 * k_B * circuit.temperature * circuit.gamma * gm))
        add("ro", 0, np.sqrt(2 * k_B * circuit.temperature / circuit.Ro[0, 0, 0]))
    index = next(i for i, line in enumerate(lines) if line.startswith(".ic "))
    lines[index:index] = physical
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("realization", "source"),
    [
        ("gmc", "ota"),
        ("gmc", "resistor"),
        ("gmc", "combined"),
        ("active_rc", "resistor"),
        ("active_rc", "current"),
        ("active_rc", "voltage"),
        ("active_rc", "combined"),
    ],
)
def test_open_loop_source_variance_matches_theory_and_ngspice(realization, source):
    """Each physical source alone, then their sum, gives the Lyapunov Pxx."""
    circuit = _circuit(realization, source, 300.0, 1e-12)
    theory = _theory_variance(realization, source, 300.0, 1e-12)
    size, warmup = 2048, 256
    x_cbadc = circuit.simulate(
        size, record="x", seed=17, engine="numpy", verbose=False
    )["x"][warmup:, 0, 0]
    tran = spice.run(_physical_netlist(circuit, size, 29))
    x_spice = spice.states(tran, circuit.N, circuit.dt, size)[warmup:, 0]
    assert np.var(x_cbadc) == pytest.approx(theory, rel=0.35)
    assert np.var(x_spice) == pytest.approx(theory, rel=0.5)


@pytest.mark.parametrize("realization", ["gmc", "active_rc"])
def test_temperature_capacitance_and_aggregate_netlist(realization):
    """Hotter/lower-C circuits grow noisier in both netlist constructions."""
    size, warmup = 2048, 256
    variances = []
    for temperature, capacitance in ((250.0, 2e-12), (400.0, 1e-12)):
        circuit = _circuit(realization, "combined", temperature, capacitance)
        physical = spice.run(_physical_netlist(circuit, size, 31))
        aggregate = spice.run(circuit.netlist(size, seed=37))
        x_physical = spice.states(physical, circuit.N, circuit.dt, size)[warmup:, 0]
        x_aggregate = spice.states(aggregate, circuit.N, circuit.dt, size)[warmup:, 0]
        theory = _theory_variance(realization, "combined", temperature, capacitance)
        assert np.var(x_physical) == pytest.approx(theory, rel=0.5)
        assert np.var(x_aggregate) == pytest.approx(theory, rel=0.5)
        variances.append(theory)
    assert variances[1] > variances[0]


@pytest.mark.parametrize("realization", ["gmc", "active_rc"])
def test_open_loop_variance_is_clock_step_independent(realization):
    """Physical noise bandwidth tracks dt; settled voltage variance does not."""
    observed = []
    for dt in (1e-6, 4e-6):
        circuit = _circuit(realization, "combined", 300.0, 1e-12, dt=dt)
        theory = _theory_variance(realization, "combined", 300.0, 1e-12)
        size = 2048
        tran = spice.run(_physical_netlist(circuit, size, 43))
        x = spice.states(tran, circuit.N, dt, size)[256:, 0]
        observed.append(np.var(x))
        assert observed[-1] == pytest.approx(theory, rel=0.5)
    assert observed[0] == pytest.approx(observed[1], rel=0.6)


def _loop_metrics(x, v, fs, tone, band):
    """State spread, bit transitions, integrated in-band PSD and FFT SNDR."""
    n = len(v)
    spectrum = np.abs(np.fft.rfft(v)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    signal_bin = round(tone * n / fs)
    noise_bins = (f > 3 * fs / n) & (f <= band)
    noise_bins[max(0, signal_bin - 2) : signal_bin + 3] = False
    return np.array(
        [
            np.var(x),
            np.mean(v[1:] != v[:-1]),
            2 * spectrum[noise_bins].sum() / n**2,
            snr.fft_sndr(v, tone, fs, band=band).sndr_dB,
        ]
    )


@pytest.mark.parametrize("temperature", [250.0, 400.0])
@pytest.mark.parametrize("capacitance", [2e-17, 5e-17])
@pytest.mark.parametrize("realization", ["gmc", "active_rc"])
def test_closed_loop_noise_statistics_across_seeds(
    realization, temperature, capacitance
):
    """Compare noisy SDM distributions with a quiet control at each T and C.

    Three independent seeds per solver exercise state variance, switching,
    integrated in-band output PSD and SNDR. Distinct random draws need not
    give matching bit streams; their distributions should agree instead.
    """
    dt, size, warmup = 1e-5, 2048, 256
    fs = 1 / dt
    tone, band = 13 * fs / (size - warmup), fs / 20
    ideal = Frontend(
        [[-0.2 / dt]],
        [[0.2 / dt, -0.5 / dt]],
        [[1.0]],
        dt=dt,
        input=Sinusoid(0.4, tone, -np.pi / 2),
    )
    component_args = dict(Cint=capacitance, C_design=3e-17, temperature=temperature)
    if realization == "gmc":
        noisy = GmC(ideal, **component_args)
        quiet = GmC(ideal, **component_args, thermal_noise=False)
    else:
        rc_args = dict(**component_args, gm=1e-7, Ro=np.inf, Co=1e-13, omega_u=5e4)
        noisy = ActiveRC(
            ideal,
            **rc_args,
            opamp_current_noise=2e-16,
            opamp_voltage_noise=5e-6,
        )
        quiet = ActiveRC(ideal, **rc_args, thermal_noise=False)

    def measure_cbadc(circuit, seed):
        result = circuit.simulate(
            size, record=("x", "v"), seed=seed, engine="numpy", verbose=False
        )
        return _loop_metrics(
            result["x"][warmup:, 0, 0],
            result["v"][warmup:, 0, 0],
            fs,
            tone,
            band,
        )

    def measure_spice(circuit, seed):
        tran = spice.run(circuit.netlist(size, seed=seed))
        x = spice.states(tran, circuit.N, dt, size)[warmup:, 0]
        v = spice.decisions(tran, circuit.N, circuit.M, dt, size)[warmup:, 0]
        return _loop_metrics(x, v, fs, tone, band)

    quiet_python = measure_cbadc(quiet, 1)
    quiet_spice = measure_spice(quiet, 1)
    assert np.all(np.isfinite(quiet_python))
    np.testing.assert_allclose(quiet_spice[0], quiet_python[0], rtol=0.03)
    assert quiet_spice[1] == pytest.approx(quiet_python[1], abs=0.005)
    np.testing.assert_allclose(quiet_spice[2], quiet_python[2], rtol=0.03)
    assert quiet_spice[3] == pytest.approx(quiet_python[3], abs=0.5)
    assert noisy.state_covariance is not None and quiet.state_covariance is None

    python_metrics = np.array([measure_cbadc(noisy, seed) for seed in (11, 29, 47)])
    spice_metrics = np.array([measure_spice(noisy, seed) for seed in (13, 31, 53)])
    assert np.all(np.isfinite(python_metrics))
    assert np.all(np.isfinite(spice_metrics))
    # Compare the median of independent realizations; the bit streams differ.
    py_median, sp_median = (
        np.median(python_metrics, axis=0),
        np.median(spice_metrics, axis=0),
    )
    np.testing.assert_allclose(sp_median[0], py_median[0], rtol=0.12)
    assert sp_median[1] == pytest.approx(py_median[1], abs=0.02)
    np.testing.assert_allclose(sp_median[2], py_median[2], rtol=0.5)
    assert sp_median[3] == pytest.approx(py_median[3], abs=2.0)
