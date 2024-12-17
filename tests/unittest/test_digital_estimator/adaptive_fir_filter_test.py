import pytest
import numpy as np
import cbadc
from scipy import signal


@pytest.fixture()
def setup():
    N = M = 6
    ENOB = 12
    BW = 1e6
    K = 1 << 5
    simulation_length = 1 << 14
    warm_up = 1 << 10
    analog_frontend = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW)
    T = analog_frontend.digital_control.clock.T
    fs = 1.0 / T
    OSR = fs / (2 * BW)
    OSR = int(np.ceil(OSR))
    DSR = OSR >> 0
    kappa_0 = 2e-1
    r_seq = kappa_0 * (2.0 * np.random.randint(0, 2, simulation_length + warm_up) - 1.0)
    r_signal = cbadc.analog_signal.ZeroOrderHold(r_seq, T)
    atol = 1e-15
    rtol = 1e-12
    simulator = cbadc.simulator.PreComputedControlSignalsSimulator(
        analog_frontend.analog_filter,
        analog_frontend.digital_control,
        [
            r_signal,
        ],
        atol=atol,
        rtol=rtol,
    )
    # simulate
    s = np.zeros((simulation_length, M))
    x = np.zeros((simulation_length, M))

    for i in cbadc.utilities.show_status(range(warm_up)):
        next(simulator)

    for i in cbadc.utilities.show_status(range(simulation_length)):
        s[i, :] = 2.0 * next(simulator) - 1.0
        x[i, :] = simulator.state_vector()
        r_seq[i + warm_up] = simulator.input_signals[0].evaluate(simulator.t)
    bw_rel = DSR / (2 * OSR)

    h0 = signal.firwin2(K, [0, bw_rel, bw_rel, 1], [1, 1 / np.sqrt(2), 0, 0])
    # Decimate and batch control signals and reference sequence
    s_decimated = cbadc.digital_estimator.decimate(s, DSR)
    s_batched = cbadc.digital_estimator.batch(s_decimated, K)

    r_decimated = cbadc.digital_estimator.decimate(r_seq[warm_up:], DSR)
    r_batched = cbadc.digital_estimator.batch(r_decimated, K)
    r_filtered = np.dot(r_batched, h0).reshape((1, -1))

    # Validate with test signal
    amplitude = 1e-0
    frequency = fs
    while frequency > BW / 8:
        frequency /= 2

    test_signal = cbadc.analog_signal.Sinusoidal(
        frequency=frequency, amplitude=amplitude, phase=0.0, offset=0.0
    )
    test_simulator = cbadc.simulator.PreComputedControlSignalsSimulator(
        analog_frontend.analog_filter,
        analog_frontend.digital_control,
        [
            test_signal,
        ],
        atol=atol,
        rtol=rtol,
    )
    # Simulate
    simulation_length = 1 << 14
    s_test = np.zeros((simulation_length, M))
    x_test = np.zeros((simulation_length, M))

    for i in cbadc.utilities.show_status(range(warm_up)):
        next(test_simulator)

    for i in cbadc.utilities.show_status(range(simulation_length)):
        s_test[i, :] = 2.0 * next(test_simulator) - 1.0
        x_test[i, :] = test_simulator.state_vector()

    s_test_decimated = cbadc.digital_estimator.decimate(s_test, DSR)
    s_test_batched = cbadc.digital_estimator.batch(s_test_decimated, K)

    return {
        "analog_frontend": analog_frontend,
        "s_decimated": s_decimated,
        "s_batched": s_batched,
        "r_decimated": r_decimated,
        "r_batched": r_batched,
        "r_filtered": r_filtered,
        "s_test_decimated": s_test_decimated,
        "s_test_batched": s_test_batched,
        "h0": h0,
        "K": K,
        "DSR": DSR,
        "fs": fs,
        "BW": BW,
        "ENOB": ENOB,
    }


def test_init_filter():
    M = 10
    K = 1 << 8
    L = 1
    cbadc.digital_estimator.AdaptiveFIRFilter(M, K, L)
    cbadc.digital_estimator.AdaptiveFIRFilter(M, K, L=2)
    cbadc.digital_estimator.AdaptiveFIRFilter(M, K, L=2, dtype=np.complex128)


def check_ENOB(f, psd, target_ENOB, fs, BW):
    signal_index = cbadc.utilities.find_sinusoidal(psd, 50)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[0:2] = False
    noise_index[f > BW] = False
    res = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=fs
    )
    SNR = 10 * np.log10(res["snr"])
    est_ENOB = cbadc.fom.snr_to_enob(SNR)

    if est_ENOB < target_ENOB:
        raise Exception(
            f"Estimated ENOB: {est_ENOB} is lower than target ENOB: {target_ENOB}"
        )


def test_calibration(setup):
    M = setup["analog_frontend"].analog_filter.M
    K = setup["K"]
    s_decimated = setup["s_decimated"]
    r_filtered = setup["r_filtered"]
    s_test_decimated = setup["s_test_decimated"]
    fs = setup["fs"]
    DSR = setup["DSR"]
    BW = setup["BW"]
    target_ENOB = setup["ENOB"]

    # Solve using least squares
    lstsq_estimator = cbadc.digital_estimator.AdaptiveFIRFilter(M, K, L=1)
    lstsq_estimator.lstsq(
        s_decimated,
        y=r_filtered,
    )

    # Solve using RLS
    epochs = 1 << 1
    delta = 1e-6
    lambda_ = 1e0 - 1e-12

    rls_estimator = cbadc.digital_estimator.AdaptiveFIRFilter(M, K, L=1)
    rls_estimator.rls(
        x=s_decimated,
        y=r_filtered.reshape((1, -1)),
        epochs=epochs,
        verbose=True,
        delta=delta,
        lambda_=lambda_,
    )

    # Solve using LMS
    epochs = 1 << 8
    batch_size = 1 << 1
    learning_rate = 1e-2
    momentum = 0.91

    lms_estimator = cbadc.digital_estimator.AdaptiveFIRFilter(M, K, L=1)
    lms_estimator.lms(
        x=s_decimated,
        y=r_filtered.reshape((1, -1)),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        shuffle=False,
        verbose=True,
    )

    lstsq_estimator.plot_bode()
    lstsq_estimator.plot_impulse_response()

    rls_estimator.plot_bode()
    rls_estimator.plot_impulse_response()

    lms_estimator.plot_bode()
    lms_estimator.plot_impulse_response()

    # Compute LSTSQ test estimate
    u_hat_lstsq = lstsq_estimator.predict(s_test_decimated).flatten()
    # Compute RLS test estimate
    u_hat_rls = rls_estimator.predict(s_test_decimated).flatten()
    # Compute LMS test estimate
    u_hat_lms = lms_estimator.predict(s_test_decimated).flatten()

    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_lstsq,
        fs=fs / DSR,
    )

    check_ENOB(f, psd, target_ENOB, fs, BW)

    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_lms,
        fs=fs / DSR,
    )

    check_ENOB(f, psd, target_ENOB, fs, BW)

    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_rls,
        fs=fs / DSR,
    )

    check_ENOB(f, psd, target_ENOB, fs, BW)
