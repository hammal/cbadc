import cbadc
import pytest
import os
import uuid
import numpy as np
import glob


@pytest.fixture()
def recorder_fixture():
    filename = f"{str(uuid.uuid4())}"

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

    _recorder = cbadc.observer.Recorder(filename, simulator, digital_estimator, 1 << 4)

    yield _recorder, filename

    filenames = glob.glob(f'{filename}*.data')
    filenames.append(filename)
    for filename in filenames:
        if os.path.exists(filename):
            print(f"removing file: {filename}")
            os.remove(filename)


def test_recorder(recorder_fixture):
    recorder, _ = recorder_fixture

    recorder.save(1 << 6)
    recorder.save(1 << 6)

    print(recorder)


def test_get_pandas(recorder_fixture):
    recorder, filename = recorder_fixture
    recorder.save(1 << 10)

    playback = cbadc.observer.Playback(filename)
    print(playback._get_discrete_time_pandas_df())
    print(playback._get_discrete_time_pandas_df().info())
    print(playback._get_continuous_time_pandas_df())
    print(playback._get_continuous_time_pandas_df().info())
    # assert False
