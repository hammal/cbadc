import numpy as np
import matplotlib.pyplot as plt
from cbadc import (
    WienerFilter,
    AnalogFrontend,
    Sinusoidal,
    AdaptiveFIRFilter,
    ZeroOrderHold,
)
from cbadc.digital_backend import decimate, BlackBoxEstimator
from scipy.signal import (
    dlti,
    firwin2,
    fftconvolve,
    convolve,
    resample,
    decimate as _decimate,
)
import pytest

ENOB = 12.0
N = 5
M = N
L = 1
Bw = 1e7


def test_wiener_initialization():

    af, _ = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)
    _, tf = af.transfer_function(
        np.array([2j * np.pi * Bw]), input_index=0, output_index=-1
    )
    eta2 = np.abs(tf[0, 0, 0]) ** 2
    wf = af.wiener_filter(eta2)

    assert wf._Af.shape == (N, N)
    assert wf._Ab.shape == (N, N)
    assert wf._Bf.shape == (N, M)
    assert wf._Bb.shape == (N, M)
    assert wf._eta2 == eta2
    assert wf._W.shape == (L, N)


def test_wiener_stf():
    af, _ = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)
    _, tf = af.transfer_function(
        np.array([2j * np.pi * Bw]), input_index=0, output_index=-1
    )
    eta2 = np.abs(tf[0, 0, 0]) ** 2
    wf = WienerFilter(af, eta2)
    jw = np.logspace(6, 8, 1000) * 2j * np.pi
    _, stf = wf.stf(jw)
    _, ntf = wf.ntf(jw)

    plt.figure()
    plt.semilogx(
        np.abs(jw) / (2 * np.pi), 20 * np.log10(np.abs(stf[:, 0, 0])), label="STF"
    )
    plt.semilogx(
        np.abs(jw) / (2 * np.pi), 20 * np.log10(np.abs(ntf[:, 0, 0])), label="NTF"
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    # plt.show()


def test_wiener_filter():
    af, _ = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)
    size = 1 << 16
    amplitude = np.array([[1]], dtype=float)
    freq = np.ones_like(amplitude) * 1e7
    sinusoidal = Sinusoidal(amplitude, freq)
    af.analog_signal = sinusoidal

    afd = af.discretize(af.dt)
    dt_sim = afd.simulate(size)
    ct_sim = af.simulate(size)

    _, tf = af.transfer_function(
        np.array([2j * np.pi * Bw]), input_index=0, output_index=-1
    )
    eta2 = np.abs(tf[0, 0, 0]) ** 2
    wf = af.wiener_filter(eta2)
    u_hat_dt = wf(dt_sim["v"])
    u_hat_ct = wf(ct_sim["v"])
    u = dt_sim["u"][:, 0]

    plt.figure()
    plt.plot(dt_sim["t"], u_hat_dt.squeeze(), label="u_hat (dt)")
    plt.plot(dt_sim["t"], u_hat_ct.squeeze(), label="u_hat (ct)")
    plt.plot(dt_sim["t"], u.squeeze(), label="u")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Output")

    plt.figure()
    plt.psd(u_hat_dt.flatten(), NFFT=1024, Fs=1 / af.dt, label="u_hat (dt)")
    plt.psd(u_hat_ct.flatten(), NFFT=1024, Fs=1 / af.dt, label="u_hat (ct)")
    plt.psd(u.flatten(), NFFT=1024, Fs=1 / af.dt, label="u")
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.title("Power Spectral Density")
    plt.xscale("log")

    # plt.show()


K = 1 << 7


def test_adaptive_fir_filter_initialization():
    af, _ = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)

    afir = AdaptiveFIRFilter(M, K, L, dtype=float, dt=af.dt, analog_frontend=af)

    # set coefficients
    afir.h = np.ones((L, M, K))

    with pytest.raises(ValueError) as excinfo:
        afir.h = np.ones((L, M, K + 1))  # This should raise an error

    assert afir.M == M
    assert afir.K == K
    assert afir.L == L

    h = afir.h
    for l in range(L):
        for m in range(M):
            print(h[l][m])
            assert isinstance(h[l][m], dlti)
            assert h[l][m].dt == af.dt
            assert h[l][m].num.size == K


def test_adaptive_fir_filter_calibration():
    af, _ = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)
    dsr = 1.0 / (2 * Bw * af.dt)
    print(dsr, int(dsr))
    size = 1 << 16
    # amplitude = np.array([1], dtype=float)
    # freq = np.array([1e7], dtype=float)
    # sinusoidal = Sinusoidal(amplitude, freq)
    uniform_reference = ZeroOrderHold.uniform_reference_signal(
        af.dt, np.array([-1]), np.array([1])
    )
    af.analog_signal = uniform_reference

    afd = af.discretize(af.dt)
    sim = afd.simulate(size)

    afir_lstsq = AdaptiveFIRFilter(M, K, L, dtype=float, dt=af.dt, analog_frontend=af)
    afir_lms = AdaptiveFIRFilter(M, K, L, dtype=float, dt=af.dt, analog_frontend=af)
    afir_rls = AdaptiveFIRFilter(M, K, L, dtype=float, dt=af.dt, analog_frontend=af)

    s, u = decimate(sim["v"], dsr), decimate(sim["u"], dsr)
    h0 = firwin2(K, [0.0, 1 / dsr, 1.0], [1.0, 1.0, 0.0])
    r = fftconvolve(u, h0[:, np.newaxis, np.newaxis], mode="valid")
    print(s.shape, r.shape)
    print(f"LSTSQ loss = {afir_lstsq.lstsq(s[:,:,0], r[:,:,0])}")
    print(
        f"LMS loss = {afir_lms.lms(
            s[:,:,0],
            r[:,:,0],
            batch_size=1 << 7,
            epochs=1 << 10,
            learning_rate=1e-2,
            momentum=0.97,
            verbose=True,
        )}"
    )
    print(
        f"RLS loss = {afir_rls.rls(s[:,:,0], r[:,:,0], epochs=1 << 0, delta=1e-2, lambda_=1e0 - 1e-6, verbose=True)}"
    )

    # jw = np.geomspace(Bw * 1e-2, Bw * 1e1, 1000) * 2j * np.pi
    jw = np.geomspace(0.5e-2, 0.5, 1000) * 2j * np.pi
    _, tf_lstsq = afir_lstsq.transfer_function(jw)
    # print(afir_lstsq.h)
    # print(tf_lstsq.shape)
    # print(tf_lstsq)

    afir_lstsq.plot_impulse_response()
    afir_lstsq.plot_amplitude_response(jw)
    afir_lms.plot_impulse_response()
    afir_lms.plot_amplitude_response(jw)
    afir_rls.plot_impulse_response()
    afir_rls.plot_amplitude_response(jw)

    # plt.show()

    # assert False


def test_adaptive_fir_impulse_response():
    af, _ = AnalogFrontend.chain_of_integrators(ENOB=ENOB, N=N, BW=Bw)
    dsr = int(1.0 / (2 * Bw * af.dt))
    print(dsr, int(dsr))
    size = 1 << 16
    # amplitude = np.array([1], dtype=float)
    # freq = np.array([1e7], dtype=float)
    uniform_reference = ZeroOrderHold.uniform_reference_signal(
        af.dt, np.array([-1]), np.array([1])
    )
    af.analog_signal = uniform_reference

    afd = af.discretize(af.dt)
    sim = afd.simulate(size)

    afir_lstsq = AdaptiveFIRFilter(M, K, L, dtype=float, dt=af.dt, analog_frontend=af)

    v, u = decimate(sim["v"], dsr), decimate(sim["u"], dsr)
    h0 = firwin2(K, [0.0, 1 / dsr, 1.0], [1.0, 1.0, 0.0])
    r = fftconvolve(u, h0[:, np.newaxis, np.newaxis], mode="valid")
    print(v.shape, r.shape)
    print(f"LSTSQ loss = {afir_lstsq.lstsq(v[:,:,0], r[:,:,0])}")

    afir_lstsq.plot_impulse_response()
    # plt.show()
    # assert False


def test_sanity_FIR_filter():
    K = 1 << 6
    M = 5
    L = 1
    J = 1 << 4
    size = 1 << 14
    noise_std = 1e-6

    afilter = AdaptiveFIRFilter(M=M, K=K, L=L, dtype=float, dt=1e-6)
    x = np.random.randn(size, M, J)
    rel_bw = 0.5 / (1 << 4)
    # h0.shape = (K, L)
    h0 = firwin2(
        K, [0, rel_bw, 1.0], [0, 1.0 / np.sqrt(2.0), 0.0], antisymmetric=True
    ).reshape((K, L))
    afilter.h0 = h0
    # y = np.random.randn(size - K + 1, L, J) * noise_std
    y = np.random.randn(size, L, J) * noise_std

    for j in range(J):
        for l in range(L):
            for m in range(M):
                y[:, l, j] += convolve(
                    x[:, m, j],
                    h0[:, l],
                    mode="same",
                ) * 10 ** (-m)

    loss = afilter.lstsq(x, y)
    print(f"SANITY CHECK: LSTSQ loss = {loss}")

    afilter.plot_impulse_response()

    fig, ax = plt.subplots(2)
    for m in range(M):
        for l in range(L):
            ax[0].plot(h0[:, l] * 10 ** (-m), label=f"ref h[{l},{m}]")
            ax[1].semilogy(np.abs(h0[:, l]) * 10 ** (-m), label=f"ref h[{l},{m}]")
    ax[0].legend()
    ax[1].legend()

    afilter.plot_amplitude_response(np.geomspace(1e-4, 0.5, 1000) * 2j * np.pi)
    plt.show()
    for m in range(M):
        for l in range(L):
            assert np.allclose(afilter._h[:, m, l], h0[:, l] * 10 ** (-m), atol=1e-8)


def test_sanity_FIR_filter_v2():
    K = 1 << 6
    Ksys = K >> 0
    M = 1
    L = 1
    J = 1 << 0
    size = 1 << 15
    noise_std = 1e-6

    afilter = AdaptiveFIRFilter(M=M, K=K, L=L, dtype=float, dt=1e-6)
    u = np.random.randn(size, L, J) * 1e-1
    # u = Sinusoidal(np.ones((L, J)), np.ones((L, J))).evaluate(np.arange(size) / (size - 1) * 2 * np.pi / 3).reshape((size, L, J))
    rel_bw = 0.5
    # h0.shape = (K, L)
    h0 = firwin2(
        Ksys, [0, rel_bw, 1.0], [0, 1.0 / np.sqrt(2.0), 0.0], antisymmetric=False
    ).reshape((Ksys, L))
    afilter.h0 = h0
    # h1 = np.zeros((K, L))
    # h1[K//2, 0] = 1.0
    # afilter.h0 = h1
    # y = np.random.randn(size + K + 1, L, J) * noise_std
    v = np.random.randn(size + Ksys - 1, M, J) * noise_std
    # v = np.random.randn(size , M, J) * noise_std

    for j in range(J):
        for l in range(L):
            for m in range(M):
                # v[:, m, j] += u[:, l, j] * 10 ** (-m)
                v[:, m, j] += convolve(
                    u[:, l, j],
                    h0[:, l] * 10 ** (-m),
                    mode="full",
                )

    plt.figure()
    plt.plot(u[:, 0, 0], label="u")
    for m in range(M):
        plt.plot(v[:, m, 0], label=f"v[{m}]")
    plt.legend()

    plt.figure()
    plt.psd(u[:, 0, 0].flatten(), NFFT=1024, Fs=1e6)
    for m in range(M):
        plt.psd(v[:, m, 0].flatten(), NFFT=1024, Fs=1e6)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.title("Power Spectral Density")
    plt.xscale("log")
    plt.legend()

    loss = afilter.lstsq(v, u[:, :, :])
    print(f"SANITY CHECK: LSTSQ loss = {loss}")

    afilter.plot_impulse_response()

    fig, ax = plt.subplots(2)
    for m in range(M):
        for l in range(L):
            ax[0].plot(h0[:, l] * 10 ** (-m), label=f"ref h[{l},{m}]")
            ax[1].semilogy(np.abs(h0[:, l]) * 10 ** (-m), label=f"ref h[{l},{m}]")
    ax[0].legend()
    ax[1].legend()

    afilter.plot_amplitude_response(np.geomspace(1e-4, 0.5, 1000) * 2j * np.pi)
    plt.show()
    for m in range(M):
        for l in range(L):
            assert np.allclose(afilter._h[:, m, l], h0[:, l] * 10 ** (-m), atol=1e-8)


def test_sanity_FIR_decimation_filter():
    K = 1 << 6
    M = 5
    L = 1
    J = 1 << 4
    size = 1 << 18
    noise_std = 1e-6
    DSR = 4

    afilter = AdaptiveFIRFilter(M=M, K=K, L=L, dtype=float, dt=1e-6)
    x = resample(np.random.randn(size // DSR, M, J), size, axis=0)
    x = np.random.randn(size, M, J)
    rel_bw = 1.0 / DSR
    # h0.shape = (K, L)
    h0 = firwin2(
        K, [0, rel_bw, 1.0], [0, 1.0 / np.sqrt(2.0), 0.0], antisymmetric=True
    ).reshape((K, L))
    afilter.h0 = h0
    # y = np.random.randn(size - K + 1, L, J) * noise_std
    y = np.random.randn(size, L, J) * noise_std

    for j in range(J):
        for l in range(L):
            for m in range(M):
                y[:, l, j] += convolve(
                    x[:, m, j],
                    h0[:, l],
                    mode="same",
                ) * 10 ** (-m)

    x_dec = decimate(x, DSR, axis=0)
    y_dec = decimate(y, DSR, axis=0)

    loss = afilter.lstsq(x_dec, y_dec)
    print(f"SANITY CHECK: LSTSQ loss = {loss}")

    afilter.plot_impulse_response()

    fig, ax = plt.subplots(2)
    for m in range(M):
        for l in range(L):
            ax[0].plot(h0[:, l] * 10 ** (-m), label=f"ref h[{l},{m}]")
            ax[1].semilogy(np.abs(h0[:, l]) * 10 ** (-m), label=f"ref h[{l},{m}]")
    ax[0].legend()
    ax[1].legend()

    afilter.plot_amplitude_response(np.geomspace(1e-4, 0.5, 1000) * 2j * np.pi)
    plt.show()
    for m in range(M):
        for l in range(L):
            assert np.allclose(afilter._h[:, m, l], h0[:, l] * 10 ** (-m), atol=1e-8)

    plt.figure()
    plt.psd(x_dec.flatten(), NFFT=1024, Fs=1e6, label="x decimated")
    for m in range(M):
        plt.psd(y_dec[:,].flatten(), NFFT=1024, Fs=1e6, label="y decimated")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.title("Power Spectral Density")
    plt.xscale("log")
    plt.legend()


def test_black_box_estimator_learn_from_analog_frontend():
    OSR = 16
    N = 5
    Bw = 1e7
    J = 2
    af, OSR = AnalogFrontend.chain_of_integrators(OSR=OSR, N=N, BW=Bw)
    af, OSR = AnalogFrontend.leapfrog(OSR=OSR, N=N, BW=Bw)

    size = 1 << 16
    OSR = np.floor(OSR).astype(int)
    DSR = OSR >> 0
    # DSR = 1
    print(f"Using DSR = {DSR} for BBE test.")
    rel_bw = 0.5 * DSR / OSR
    print(f"Using relative bandwidth = {rel_bw} for BBE test.")
    bbe = BlackBoxEstimator(af, DSR=DSR, K=1 << 7, seed=123456789, rel_bw=rel_bw)
    cal_res = bbe.learn_from_analog_frontend(
        DSR=DSR, max_amplitude=1.0, sim_size=size, rel_bw=rel_bw, J=J
    )
    print("Black Box Estimator learned parameters:")
    print(f"M = {bbe.M}, K = {bbe.K}, L = {bbe.L}")
    # assert bbe.M == M
    # assert bbe.K == 1 << 7
    # assert bbe.L == L

    fs = 1.0 / af.dt
    freq = fs
    while freq > Bw / 2:
        freq /= 2
    freq = Bw / 2
    amp = np.ones(1) * 1.0
    input_signal = Sinusoidal(amp, np.ones_like(amp) * freq)
    af.analog_signal = input_signal
    res = af.simulate(size)
    print("Simulation done.")

    dec_v_Fourier = decimate(res["v"], DSR, method="fft")
    dec_u_Fourier = decimate(res["u"], DSR, method="fft")

    dec_v_dec = decimate(res["v"], DSR, method="direct")
    dec_u_dec = decimate(res["u"], DSR, method="direct")

    dec_u_ref_Fourier = decimate(cal_res["u"], DSR, method="fft")
    dec_u_ref_dec = decimate(cal_res["u"], DSR, method="direct")

    u_hat_Fourier = bbe.convolve(dec_v_Fourier, method="fft")
    u_hat_dec = bbe.convolve(dec_v_dec, method="direct")

    print("Black Box Estimator convolution done.")

    plt.figure()
    plt.plot(dec_u_Fourier[:, 0], label="u Fourier")
    plt.plot(dec_u_dec[:, 0], label="u decimated")
    plt.plot(u_hat_Fourier[:, 0], label="u_hat Fourier")
    plt.plot(u_hat_dec[:, 0], label="u_hat decimated")
    plt.plot(dec_u_ref_Fourier[:, 0], "--", label="u ref Fourier")
    plt.plot(dec_u_ref_dec[:, 0], "--", label="u ref decimated")
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Output")
    plt.title("Black Box Estimator Output vs True Output")

    plt.figure()
    plt.psd(dec_u_Fourier[:, 0].flatten(), NFFT=1024, Fs=fs / DSR, label="u (Fourier)")
    plt.psd(dec_u_dec[:, 0].flatten(), NFFT=1024, Fs=fs / DSR, label="u (decimated)")

    plt.psd(
        u_hat_Fourier[:, 0].flatten(), NFFT=1024, Fs=fs / DSR, label="u_hat (Fourier)"
    )
    plt.psd(
        u_hat_dec[:, 0].flatten(), NFFT=1024, Fs=fs / DSR, label="u_hat (decimated)"
    )

    plt.psd(
        dec_u_ref_Fourier[:, 0].flatten(),
        NFFT=1024,
        Fs=fs / DSR,
        label="u ref (Fourier)",
    )
    plt.psd(
        dec_u_ref_dec[:, 0].flatten(), NFFT=1024, Fs=fs / DSR, label="u ref (decimated)"
    )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.title("Power Spectral Density")
    plt.xscale("log")
    plt.legend()

    bbe.plot_impulse_response()
    bbe.plot_amplitude_response(np.geomspace(1e-4, 0.5, 1000) * 2j * np.pi)
    plt.show()


def test_black_box_estimator_and_delta_analog_frontend():
    OSR = 10
    N = 10
    Bw = 1e7
    J = [1]
    # af_int, OSR = AnalogFrontend.chain_of_integrators(OSR=OSR, N=N, BW=Bw)
    af_lf, _ = AnalogFrontend.leapfrog(OSR=OSR, N=N, BW=Bw)
    af_lf_delta, _ = AnalogFrontend.leapfrog(OSR=OSR, N=N, BW=Bw, delta=0.5)

    fs = OSR * 2 * Bw
    freq = fs
    while freq > Bw / 2:
        freq /= 2
    # freq = Bw / 2
    amp = np.ones(1) * 1.0
    input_signal = Sinusoidal(amp, np.ones_like(amp) * freq)
    deltas = [10.0, 1.0, 0.1, 0.01]

    size = 1 << 18
    DSR = OSR >> 0
    # DSR = 1
    rel_bw = 0.5 * DSR / OSR
    for j in J:
        for delta in deltas:
            af, _ = AnalogFrontend.leapfrog(OSR=OSR, N=N, BW=Bw, delta=delta)
            print(f"Testing BBE with J={j} and Analog Frontend: {af}")
            print(f"Using relative bandwidth = {rel_bw} for BBE test.")
            bbe = BlackBoxEstimator(
                af, DSR=DSR, K=1 << 7, seed=123456789, rel_bw=rel_bw
            )
            cal_res = bbe.learn_from_analog_frontend(
                DSR=DSR, max_amplitude=1.0, sim_size=size, rel_bw=rel_bw, J=j
            )
            print("Black Box Estimator learned parameters:")
            print(f"M = {bbe.M}, K = {bbe.K}, L = {bbe.L}")
            # assert bbe.M == M
            # assert bbe.K == 1 << 7
            # assert bbe.L == L

            af.analog_signal = input_signal
            res = af.simulate(size)
            print("Simulation done.")

            dec_v_dec = decimate(res["v"], DSR, method="direct")
            u_hat_dec = bbe.convolve(dec_v_dec, method="direct")
            u_hat_spectrum = np.fft.rfft(
                u_hat_dec[:, 0, 0] * np.hanning(u_hat_dec.size)
            )
            snr = af.calculateSNR_from_fft(
                u_hat_spectrum[3 : u_hat_dec.size].reshape((-1, 1))
            )

            print("Black Box Estimator convolution done.")

            plt.figure("time")
            plt.plot(u_hat_dec[:, 0], label=f"d={delta:.0e}, J={j}, snr={snr[0]:.1f} dB u_hat")
            plt.legend()
            plt.xlabel("Samples")
            plt.ylabel("Output")
            plt.title("Black Box Estimator Output vs True Output")

            plt.figure("frequency")
            plt.title(f"BBE Test with J={j} and Analog Frontend: {af}")

            plt.psd(
                u_hat_dec[:, 0].flatten(),
                NFFT=1024,
                Fs=fs / DSR,
                label=f"d={delta:.0e}, J={j}, snr={snr[0]:.1f} dB u_hat",
            )
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD [dB/Hz]")
            plt.title("Power Spectral Density")
            plt.xscale("log")
            plt.legend()

            plt.figure("Last state histogram")
            plt.hist(res["x"][:, -1, 0], label=f"d={delta:.0e}, J={j}", bins=100, density=True)
            plt.legend()
            # plt.title(f"Last state histogram with J={j} and d={delta:.0e}")
            plt.xlabel("State value")
            plt.ylabel("Probability density")

            # bbe.plot_impulse_response()
            # bbe.plot_amplitude_response(np.geomspace(1e-4, 0.5, 1000) * 2j * np.pi)
    plt.show()
