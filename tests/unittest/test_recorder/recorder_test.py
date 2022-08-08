import cbadc
import numpy as np


def test_recorder():
    N = 6
    BW = 1e7
    ENOB = 14
    analog_frontend = cbadc.synthesis.get_leap_frog(
        N=N,
        BW=BW,
        ENOB=ENOB,
    )
    analog_system = analog_frontend.analog_system
    digital_control = analog_frontend.digital_control
    eta2 = 1e3
    K1 = 1 << 10
    K2 = K1
    input_signal = [cbadc.analog_signal.Sinusoidal(1e-1, BW / (1 << 6))]
    event_list = cbadc.simulation_event.out_of_bounds_factory(
        N, np.array([[-1, 1] for _ in range(N)]), "out-of-bounds"
    )
    simulator = cbadc.simulator.FullSimulator(
        analog_system, digital_control, input_signal, event_list=event_list
    )
    digital_estimator = cbadc.digital_estimator.BatchEstimator(
        analog_system, digital_control, eta2, K1, K2
    )

    recorder = cbadc.observer.Recorder('test', simulator, digital_estimator, 1 << 4)

    recorder.save(1 << 6)
    recorder.save(1 << 6)

    print(recorder)


def test_get_pandas():
    N = 6
    BW = 1e7
    ENOB = 14
    analog_frontend = cbadc.synthesis.get_leap_frog(
        N=N,
        BW=BW,
        ENOB=ENOB,
    )
    analog_system = analog_frontend.analog_system
    digital_control = analog_frontend.digital_control
    eta2 = 1e3
    K1 = 1 << 10
    K2 = K1
    input_signal = [cbadc.analog_signal.Sinusoidal(1e-1, BW / (1 << 6))]
    event_list = cbadc.simulation_event.out_of_bounds_factory(
        N, np.array([[-1, 1] for _ in range(N)]), "out-of-bounds"
    )
    simulator = cbadc.simulator.FullSimulator(
        analog_system, digital_control, input_signal, event_list=event_list
    )
    digital_estimator = cbadc.digital_estimator.BatchEstimator(
        analog_system, digital_control, eta2, K1, K2
    )

    recorder = cbadc.observer.Recorder('test', simulator, digital_estimator, 1 << 10)

    recorder.save(1 << 14)

    playback = cbadc.observer.Playback('test')
    print(playback._get_discrete_time_pandas_df())
    print(playback._get_discrete_time_pandas_df().info())
    print(playback._get_continuous_time_pandas_df())
    print(playback._get_continuous_time_pandas_df().info())
    # assert False
