import numpy as np
import matplotlib.pyplot as plt
from cbadc import (
    WienerFilter,
    AnalogFrontend,
    Sinusoidal,
    AdaptiveFIRFilter,
    ZeroOrderHold,
)
from cbadc.digital_backend import decimate
from scipy.signal import dlti, firwin2, fftconvolve
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
    u_hat_dt = wf(dt_sim["s"])
    u_hat_ct = wf(ct_sim["s"])
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

    s, u = decimate(sim["s"], dsr), decimate(sim["u"], dsr)
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

    s, u = decimate(sim["s"], dsr), decimate(sim["u"], dsr)
    h0 = firwin2(K, [0.0, 1 / dsr, 1.0], [1.0, 1.0, 0.0])
    r = fftconvolve(u, h0[:, np.newaxis, np.newaxis], mode="valid")
    print(s.shape, r.shape)
    print(f"LSTSQ loss = {afir_lstsq.lstsq(s[:,:,0], r[:,:,0])}")

    afir_lstsq.plot_impulse_response()
    # plt.show()
    # assert False
