from cbadc import (
    AnalogFrontend,
    GmC,
    ActiveRC,
    DigitalControl,
    Sinusoidal,
    AnalogSignal,
    delsig as ds,
    fom,
)
import numpy as np
from tests.fixture.chain_of_integrators import chain_of_integrators
from scipy.signal import StateSpace
import matplotlib.pyplot as plt
import time
import pytest


def test_initialization(chain_of_integrators):
    # N = chain_of_integrators["N"]
    M = chain_of_integrators["M"]
    dt = chain_of_integrators["dt"]
    A = chain_of_integrators["A"]
    B = chain_of_integrators["B"]
    C = chain_of_integrators["C"]
    D = chain_of_integrators["D"]
    analog_filter = StateSpace(A, B, C, D)
    digital_control = DigitalControl(M, dt)
    AnalogFrontend(analog_filter, digital_control)


def test_properties(chain_of_integrators):
    N = chain_of_integrators["N"]
    M = chain_of_integrators["M"]
    dt = chain_of_integrators["dt"]
    A = chain_of_integrators["A"]
    B = chain_of_integrators["B"]
    C = chain_of_integrators["C"]
    D = chain_of_integrators["D"]
    f = StateSpace(A, B, C, D)
    dc = DigitalControl(M, dt)
    af = AnalogFrontend(f, dc)

    assert af.M == M
    assert af.N == N
    assert af.dt == dt
    assert (af.A == A).all()
    assert (af.B == B).all()
    assert (af.C == C).all()
    assert (af.D == D).all()
    assert af.digital_control == dc
    assert af.analog_filter == f

    # test setter
    A = A + np.eye(N)
    af.A = A
    assert (af.A == A).all()

    # ensure error
    try:
        af.A = np.zeros((N, N - 1))
    except ValueError as e:
        print(e)
        pass

    try:
        af.B = np.zeros((N, 1))
    except ValueError as e:
        print(e)
        pass

    try:
        af.Co = np.zeros((1, N - 1))
    except ValueError as e:
        print(e)
        pass

    try:
        af.D = np.zeros((1, 1))
    except ValueError as e:
        print(e)
        pass


def test_constuctors():
    ENOB = 12
    N = 5
    Bw = 1e7
    coi, OSR = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)
    print(coi)
    lf, OSR = AnalogFrontend.leapfrog(ENOB=ENOB, N=N, BW=Bw)
    print(lf)
    # assert False


def test_discretize():
    ENOB = 14.0
    N = 3
    M = N
    Bw = 1e7
    af, OSR = AnalogFrontend.leapfrog(ENOB=ENOB, N=N, BW=Bw)
    assert not af.is_discrete_time
    afd = af.discretize(af.dt)
    assert afd.is_discrete_time

    alpha = 0 * np.ones(M)
    beta = 1.0 * np.ones(M)
    af.digital_control = DigitalControl(N, af.dt, alpha, beta, dac_waveform="rz")
    afd2 = af.discretize(af.dt)

    # test FIR DAC and multi-period dac waveforms
    beta = 2.9999 * np.ones(M)
    af.digital_control = DigitalControl(N, af.dt, alpha, beta)
    assert af.digital_control.delay_steps()[0] == 2
    afd3 = af.discretize(af.dt)

    # Test A matrix
    np.testing.assert_allclose(afd2.A, afd.A)
    np.testing.assert_allclose(afd3.A[:N, :N], afd.A)

    # Test B matrix
    B1 = afd3.B[:N, 1:]
    B2 = afd3.A[:N, N : N + M]
    B3 = afd3.A[:N, N + M :]
    print(B1)
    print(B2)
    print(B3)
    np.testing.assert_allclose(afd2.B, afd.B)
    np.testing.assert_allclose(afd3.B[:N, :], afd.B)

    # TODO Fix these something with ordering of the B matrix is wrong
    # possibly around 742 in analog frontend
    np.testing.assert_allclose(B2, B1)
    np.testing.assert_allclose(B3, B1)

    # Test C matrix
    np.testing.assert_allclose(afd2.Co, afd.Co)
    np.testing.assert_allclose(afd3.Co[:, :N], afd.Co)

    # Test D matrix
    np.testing.assert_allclose(afd2.D, afd.D)
    np.testing.assert_allclose(afd3.D, afd.D)


def test_tranfer_function():
    ENOB = 20.0
    N = 7
    Bw = 1e7
    # af = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)
    af, _ = AnalogFrontend.leapfrog(ENOB=ENOB, N=N, BW=Bw)
    # s = 2j * np.pi * np.linspace(1e6, 1e8, 1000)
    s = 2j * np.pi * np.logspace(6, 8, 1000)
    af.transfer_function(s)
    # check that indexing works.
    # ideally we would call the function above and index after
    # if we care for performance but this is a test so we can do this.
    for l in range(af.L):
        for m in range(af.M):
            w, h = af.transfer_function(s, input_index=l, output_index=m)
            plt.semilogx(
                np.abs(w) / (2 * np.pi),
                20 * np.log10(np.abs(h[:, 0, 0])),
                label=f"{l} -> {m}",
            )
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")

    plt.figure()
    w, h = af.transfer_function(s)
    plt.semilogx(
        np.abs(w) / (2 * np.pi), 20 * np.log10(np.abs(h[:, -1, 0])), label="Open Loop"
    )
    w, h = af.transfer_function(s, open_loop=False)
    plt.semilogx(
        np.abs(w) / (2 * np.pi), 20 * np.log10(np.abs(h[:, -1, 0])), label="Closed Loop"
    )
    plt.legend()
    plt.xlabel("frequency [hz]")
    plt.ylabel("Magnitude [dB]")

    plt.show()


def test_quadrature():
    ENOB = 14.0
    N = 6
    Bw = 1e7
    wp = 2 * np.pi * 1e9
    af, OSR = AnalogFrontend.leapfrog(ENOB=ENOB, N=N, BW=Bw)
    L = af.L
    quad = af.quadrate(wp)
    assert quad.N == 2 * N
    assert quad.M == 2 * N
    assert quad.dt == af.dt
    np.testing.assert_allclose(quad.A[:N, :N], af.A)
    np.testing.assert_allclose(quad.A[N:, N:], af.A)
    np.testing.assert_allclose(quad.A[N:, :N], wp * np.eye(N))
    np.testing.assert_allclose(quad.A[:N, N:], -wp * np.eye(N))
    np.testing.assert_allclose(quad.B[:N, : N + L], af.B)
    np.testing.assert_allclose(quad.B[N:, N + L :], af.B)
    np.testing.assert_allclose(quad.C[:N, :N], af.C)
    np.testing.assert_allclose(quad.C[N:, N:], af.C)
    np.testing.assert_allclose(quad.D[:N, : N + L], af.D)
    np.testing.assert_allclose(quad.D[N:, N + L :], af.D)


def test_global_control():
    ENOB = 14.0
    N = 3
    Bw = 1e7
    af, OSR = AnalogFrontend.leapfrog(ENOB=ENOB, N=N, BW=Bw)
    with pytest.raises(NotImplementedError) as excinfo:
        af.global_control()


def test_simulate_time():
    ENOB = 14.0
    N = 6
    Bw = 1e7
    af, OSR = AnalogFrontend.leapfrog(ENOB=ENOB, N=N, BW=Bw)
    size = 1 << 18
    afd = af.discretize(af.dt)
    start_time = time.time()
    afd.simulate(size)
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time} seconds")
    # assert False


# @pytest.mark.xfail(reason="Two different input models.")
def test_simulate():
    # analog_signal = AnalogSignal(np.array([offset]))
    af, OSR = AnalogFrontend.leapfrog(ENOB=14, N=6, BW=1e7)
    # af = AnalogFrontend.chain_of_integrators(ENOB=14.0, N=6, BW=1e7)
    amplitude = np.array([[0.25, 0]], dtype=float)
    freq = np.array([[af.fs / 128, af.fs / 64]], dtype=float)
    sinusoidal = Sinusoidal(amplitude, freq)
    af.analog_signal = sinusoidal

    size = 1 << 12
    start_time_full = time.time()
    full_sim = af.simulate(size, method="ode")
    end_time_full = time.time()
    full_sim_time = end_time_full - start_time_full

    start_time_dt = time.time()
    afd = af.discretize(af.dt)
    dt_sim = afd.simulate(size)
    end_time_dt = time.time()
    dt_sim_time = end_time_dt - start_time_dt

    start_time_sin = time.time()
    sin_sim = af.simulate(size, method="sin")
    end_time_sin = time.time()
    sin_sim_time = end_time_sin - start_time_sin

    print(f"Full simulation time: {full_sim_time} seconds")
    print(f"Discrete-time simulation time: {dt_sim_time} seconds")
    print(f"Sinusoidal simulation time: {sin_sim_time} seconds")

    for m in range(af.M):
        plt.figure()
        plt.title(f"s {m} freq")
        s_fft = np.fft.rfft(full_sim["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=af.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(s_fft)), label="Full")

        s_fft = np.fft.rfft(dt_sim["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=af.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(s_fft)), label="Discrete")

        s_fft = np.fft.rfft(sin_sim["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=af.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(s_fft)), label="Sinusoidal")
        plt.legend()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")

        plt.figure()
        plt.title(f"s {m}")
        # length = 200
        length = size
        plt.plot(full_sim["t"][:length], full_sim["s"][:length, m, 0], label="Full")
        plt.plot(dt_sim["t"][:length], dt_sim["s"][:length, m, 0], label="Discrete")
        plt.plot(sin_sim["t"][:length], sin_sim["s"][:length, m, 0], label="Sinusoidal")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("amplitude")

        # plt.figure()
        # plt.title(f"y {m}")
        # plt.plot(full_sim["t"][:length], full_sim["y"][:length, m, 0], label=f"Full")
        # plt.plot(dt_sim["t"][:length], dt_sim["y"][:length, m, 0], label=f"Discrete")
        # plt.plot(
        #     sin_sim["t"][:length], sin_sim["y"][:length, m, 0], label=f"Sinusoidal"
        # )
        # plt.legend()
        # plt.xlabel("Time [s]")
        # plt.ylabel("amplitude")

        plt.figure()
        plt.title(f"x {m}")
        plt.plot(full_sim["t"][:length], full_sim["x"][:length, m, 0], label="Full")
        plt.plot(dt_sim["t"][:length], dt_sim["x"][:length, m, 0], label="Discrete")
        plt.plot(sin_sim["t"][:length], sin_sim["x"][:length, m, 0], label="Sinusoidal")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("amplitude")

    # inputs
    plt.figure()
    plt.plot(full_sim["t"], full_sim["u"][:, 0], label="Full")
    plt.plot(dt_sim["t"], dt_sim["u"][:, 0], label="Discrete")
    plt.plot(sin_sim["t"], sin_sim["u"][:, 0], label="Sinusoidal")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Input")

    plt.show()
    # print(af)
    # print(afd)
    # assert False
    np.testing.assert_almost_equal(sin_sim["s"], dt_sim["s"])
    np.testing.assert_almost_equal(sin_sim["u"], dt_sim["u"])
    np.testing.assert_almost_equal(sin_sim["t"], dt_sim["t"])
    np.testing.assert_almost_equal(sin_sim["y"], dt_sim["y"])
    np.testing.assert_almost_equal(sin_sim["x"], dt_sim["x"])

    np.testing.assert_almost_equal(full_sim["s"], sin_sim["s"])
    np.testing.assert_almost_equal(full_sim["u"], sin_sim["u"])
    np.testing.assert_almost_equal(full_sim["t"], sin_sim["t"])
    np.testing.assert_almost_equal(full_sim["y"], sin_sim["y"])
    np.testing.assert_almost_equal(full_sim["x"], sin_sim["x"])


def test_ABDC():
    ENOB = 14.0
    N = 6
    Bw = 1e7
    af, _ = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)
    af.ABCD

    # assert False


def test_simulateDSM():
    OSR = 64
    H_inf = 1.5
    opt = 2
    N = 4
    H = ds.synthesizeNTF(N, OSR, opt, H_inf)
    args = ds.realizeNTF(H, form="CRFB")
    ABCD = ds.stuffABCD(*args)
    afd = AnalogFrontend.dtsdm(ABCD)
    size = 1 << 14

    amplitude = np.array([0.7], dtype=float)
    freq = np.array([1.0 / (2 * OSR)], dtype=float)
    sinusoidal = Sinusoidal(amplitude, freq)
    afd.analog_signal = sinusoidal

    dt_sim = afd.simulate(size)

    dt_simSDM = afd.simulateDSM(size)

    plt.figure("states")
    plt.plot(dt_sim["t"], dt_sim["x"][:, -1, 0], label="AFsim")
    plt.plot(dt_simSDM["t"], dt_simSDM["x"][:, -1, 0], label="DSSim")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Output")

    for m in range(afd.M):
        plt.figure("s")

        s_fft = np.fft.rfft(dt_sim["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=afd.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(s_fft)), label=f"AFsim {m}")

        s_fft = np.fft.rfft(dt_simSDM["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=afd.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(s_fft)), label=f"DSSim {m}")
        plt.legend()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")

        plt.figure("s_time")
        length = 200
        plt.plot(dt_sim["t"][:length], dt_sim["s"][:length, m, 0], label=f"AFsim {m}")
        plt.plot(
            dt_simSDM["t"][:length], dt_simSDM["s"][:length, m, 0], label=f"DSSim {m}"
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("s amplitude")

    # inputs
    plt.figure()
    plt.plot(dt_sim["t"], dt_sim["u"][:, 0], label="AFsim")
    plt.plot(dt_simSDM["t"], dt_simSDM["u"][:, 0], label="DSSim")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Input")

    plt.show()
    print(afd)
    np.testing.assert_almost_equal(dt_sim["s"], dt_simSDM["s"])
    np.testing.assert_almost_equal(dt_sim["u"], dt_simSDM["u"])
    np.testing.assert_almost_equal(dt_sim["t"], dt_simSDM["t"])
    np.testing.assert_almost_equal(dt_sim["y"], dt_simSDM["y"])
    np.testing.assert_almost_equal(dt_sim["x"], dt_simSDM["x"])


def test_calculateSNR_from_fft():
    lf, OSR = AnalogFrontend.leapfrog(ENOB=14, N=6, BW=1e7)
    amplitude = np.array([[1e0, 0.5, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]])
    size = 1 << 13
    warm_up = 1 << 7
    f = 0.5 / (OSR * 2)
    f = int(np.round(f * size))
    f -= 2
    freq = f / size * lf.fs * np.ones_like(amplitude)
    lf.analog_signal = Sinusoidal(amplitude, freq)

    sim = lf.simulate(size + warm_up)

    jomega_Bw = 1j * np.pi / (OSR * lf.dt)
    _, tf = lf.transfer_function(np.array([jomega_Bw]), input_index=0, output_index=-1)
    eta2 = np.abs(tf[0, 0, 0]) ** 2
    print(f"eta2 = {eta2}")
    wf = lf.wiener_filter(eta2)
    u_hat = wf.evaluate(sim["s"])[:, 0, :]
    hwfft = np.fft.fftshift(np.fft.fft(u_hat[warm_up:], axis=0), axes=0)
    in_band_bins = size // 2 + np.arange(3, np.round(size / (2 * OSR)) + 1, dtype=int)

    # shape = (size, J)
    fft_spec = hwfft[in_band_bins - 1, :]

    # shape = (J,)
    f_argmax = np.argmax(np.abs(fft_spec), axis=0)
    # shape = (n_dim, J)
    signal_bins = f_argmax[np.newaxis, :] + np.arange(-1, 2, dtype=int)[:, np.newaxis]

    noise_bins = np.arange(fft_spec.shape[0])[:, np.newaxis] * np.ones(
        (1, fft_spec.shape[1]), dtype=int
    )

    noise_bins = np.array(
        [
            np.setdiff1d(noise_bins[:, i], signal_bins[:, i])
            for i in range(amplitude.shape[1])
        ]
    ).T

    plt.figure()
    plt.title("FFT")
    for i, amp in enumerate(amplitude[0]):
        plt.plot(
            signal_bins[:, i],
            20 * np.log10(np.abs(fft_spec[signal_bins[:, i], i])),
            label=f"amp = {amp}, sig",
        )
        plt.plot(
            noise_bins[:, i],
            20 * np.log10(np.abs(fft_spec[noise_bins[:, i], i])),
            label=f"amp = {amp}, noise",
        )
    snr = 20 * np.log10(
        np.linalg.norm(fft_spec[signal_bins], axis=0)
        / np.linalg.norm(fft_spec[noise_bins], axis=0)
    )
    plt.plot(np.array([f - 1, f, f + 1]), [-20, 60, -20], label="f_sig")
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude [dB]")
    plt.legend()
    plt.grid(True)
    plt.show()

    snr_2 = lf.calculateSNR_from_fft(fft_spec)
    print(f"snr: {snr}")
    print(f"snr_2: {snr_2}")
    assert (snr_2 == snr[0, :]).all()


def test_simulateSNR():
    OSR = 32
    N = 4
    k = 16
    lf, _ = AnalogFrontend.leapfrog(OSR=OSR, N=N, BW=1e7)
    print(lf)
    start_time = time.time()
    snr_lf, amp_lf = lf.simulateSNR(OSR, k=k)
    end_time = time.time()

    ci, _ = AnalogFrontend.chain_of_integrators(OSR=OSR, N=N, BW=1e7)
    print(ci)
    snr_ci, amp_ci = ci.simulateSNR(OSR, k=k)

    nlev = 1 << 2

    H_inf_CRFB = 1.5
    H = ds.synthesizeNTF(N, OSR, 2, H_inf_CRFB)
    args = ds.realizeNTF(H, form="CRFB")
    ABCD = ds.stuffABCD(*args)
    ABCD, _, _ = ds.scaleABCD(ABCD, nlev=nlev)
    afd = AnalogFrontend.dtsdm(ABCD, quantization_levels=nlev)
    snr_dsm_CRFB, amp_crfb = afd.simulateSNR(OSR, k=k)

    # H_inf_CRFF = 1.5
    # H = ds.synthesizeNTF(N, OSR, 1, H_inf_CRFF)
    # args = ds.realizeNTF(H, form="CRFF")
    # ABCD = ds.stuffABCD(*args)
    # ABCD, _, _ = ds.scaleABCD(ABCD, nlev=nlev)
    # afd = AnalogFrontend.dtsdm(ABCD, quantization_levels=nlev)
    # snr_dsm_CRFF, amp_crff = afd.simulateSNR(OSR, k=16)

    plt.figure()
    plt.title(
        f"OSR = {OSR}, N = {N}, time = {end_time - start_time:0.1e} s, size = {1 << k} samples, for {amp_crfb.size} amplitudes"
    )
    plt.plot(amp_lf, snr_lf, "go", label="Leapfrog")
    plt.plot(amp_ci, snr_ci, "yo", label="Chain of Integrators")
    plt.plot(amp_crfb, snr_dsm_CRFB, "ro", label=f"DT-CRFB, H_inf = {H_inf_CRFB}")
    # plt.plot(amp_crff, snr_dsm_CRFF, "bo", label=f"DT-CRFF, H_inf_CRFF = {H_inf_CRFF}")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Input Level, dB")
    plt.ylabel("(SNR dB, ENOB)")
    yaxis = np.array([-40, -20, 0, 20, 40, 60, 80, 100])
    plt.yticks(yaxis, labels=[f"({y:.0f}, {fom.snr_to_enob(y):.0f})" for y in yaxis])
    plt.tight_layout()

    plt.show()
    # assert False


def test_GmC():
    ENOB = 14.0
    N = 3
    Bw = 1e7
    af, _ = AnalogFrontend.leapfrog(ENOB=ENOB, N=N, BW=Bw)
    Ro = np.ones(N) * 1e5
    Co = np.ones(N) * 1e-12
    gmc = GmC(af, Ro, Co)
    size = 1 << 12
    amplitude = np.array([1], dtype=float)
    freq = np.array([af.fs / 128], dtype=float)
    sinusoidal = Sinusoidal(amplitude, freq)
    af.analog_signal = sinusoidal
    gmc.analog_signal = sinusoidal

    af_sim = af.simulate(size)
    gmc_sim = gmc.simulate(size)

    plt.figure()
    plt.plot(af_sim["t"], af_sim["x"][:, -1, 0], label="AnalogFrontend")
    plt.plot(gmc_sim["t"], gmc_sim["x"][:, -1, 0], label="GmC")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Output")

    print(af_sim["x"].shape)
    print(gmc_sim["x"].shape)
    print(af_sim["s"].shape)
    print(gmc_sim["s"].shape)

    for m in range(af.M):
        plt.figure()
        af_fft = np.fft.rfft(af_sim["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=af.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(af_fft)), label=f"AnalogFrontend {m}")

        gmc_fft = np.fft.rfft(gmc_sim["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=gmc.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(gmc_fft)), label=f"GmC {m}")
        plt.legend()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")

        plt.figure()
        length = 200
        plt.plot(
            af_sim["t"][:length],
            af_sim["s"][:length, m, 0],
            label=f"AnalogFrontend {m}",
        )
        plt.plot(gmc_sim["t"][:length], gmc_sim["s"][:length, m, 0], label=f"GmC {m}")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("s amplitude")
    power = gmc.avg_power(gmc_sim["x"])
    print(power)
    print(af.A)
    print(gmc.A)

    plt.show()
    # assert False


def test_active_RC():
    ENOB = 14.0
    N = 5
    Bw = 1e7
    af, _ = AnalogFrontend.leapfrog(ENOB=ENOB, N=N, BW=Bw)
    Ro = np.ones(N) * 1e8
    Co = np.ones(N) * 1e-12
    Cint = np.ones(N) * 1e-12
    gm = 1e-6 * np.ones(N)
    gmc = ActiveRC(af, Cint, gm, Ro, Co)
    size = 1 << 12
    amplitude = np.array([1], dtype=float)
    freq = np.array([af.fs / 128], dtype=float)
    sinusoidal = Sinusoidal(amplitude, freq)
    af.analog_signal = sinusoidal
    gmc.analog_signal = sinusoidal

    print(gmc)
    af_sim = af.simulate(size)
    gmc_sim = gmc.simulate(size)

    plt.figure()
    plt.plot(af_sim["t"], af_sim["x"][:, -1, 0], label="AnalogFrontend")
    plt.plot(gmc_sim["t"], gmc_sim["x"][:, -1, 0], label="ActiveRC")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Output")

    for m in range(af.M):
        plt.figure()
        af_fft = np.fft.rfft(af_sim["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=af.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(af_fft)), label=f"AnalogFrontend {m}")

        gmc_fft = np.fft.rfft(gmc_sim["s"][:, m, 0])
        f = np.fft.rfftfreq(size, d=gmc.dt)
        plt.semilogx(f, 20 * np.log10(np.abs(gmc_fft)), label=f"ActiveRC {m}")
        plt.legend()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")

        plt.figure()
        length = 200
        plt.plot(
            af_sim["t"][:length],
            af_sim["s"][:length, m, 0],
            label=f"AnalogFrontend {m}",
        )
        plt.plot(
            gmc_sim["t"][:length], gmc_sim["s"][:length, m, 0], label=f"ActiveRC {m}"
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("s amplitude")
    # power = gmc.avg_power(gmc_sim["x"])
    # print(power)
    print(af.A)
    print(gmc.A)

    plt.show()
    assert False
