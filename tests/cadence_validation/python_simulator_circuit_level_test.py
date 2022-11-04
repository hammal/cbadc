import pytest
import cbadc
import cbadc
import copy
import numpy as np
import matplotlib.pyplot as plt

# Start-up cmd: pytest tests/cadence_validation/verilog-ams_test.py
DEBUG = True


@pytest.mark.parametrize(
    "N",
    [
        # pytest.param(2, id="N=2"),
        # pytest.param(3, id="N=3"),
        pytest.param(4, id="N=4"),
        # pytest.param(5, id="N=5"),
        # pytest.param(6, id="N=6"),
        # pytest.param(7, id="N=7"),
        pytest.param(7, id="N=8"),
    ],
)
@pytest.mark.parametrize(
    "ENOB",
    [
        # pytest.param(10, id="ENOB=10"),
        pytest.param(12, id="ENOB=12"),
        # pytest.param(14, id="ENOB=14"),
        pytest.param(16, id="ENOB=16"),
        # pytest.param(20, id="ENOB=20"),
        # pytest.param(23, id="ENOB=23"),
    ],
)
@pytest.mark.parametrize(
    "BW",
    [
        # pytest.param(1e0, id="BW=1Hz"),
        # # pytest.param(1e1, id="BW=10Hz"),
        # pytest.param(1e2, id="BW=100Hz"),
        # pytest.param(1e3, id="BW=1kHz"),
        pytest.param(1e4, id="BW=10kHz"),
        # pytest.param(1e5, id="BW=100kHz"),
        # pytest.param(1e6, id="BW=1MHz"),
        pytest.param(1e7, id="BW=10MHz"),
        # pytest.param(1e8, id="BW=100MHz"),
        # pytest.param(1e9, id="BW=1GHz"),
    ],
)
@pytest.mark.parametrize(
    "analog_system",
    [
        pytest.param('chain-of-integrators', id="chain_of_integrators_as"),
        pytest.param('leap_frog', id="leap_frog_as"),
    ],
)
@pytest.mark.parametrize(
    'eta2',
    [
        # pytest.param(1.0, id="eta2=1"),
        pytest.param('snr', id="eta2=ENOB")
    ],
)
@pytest.mark.parametrize(
    'analog_circuit_implementation',
    [
        pytest.param(cbadc.circuit_level.AnalogSystemStateSpaceEquations, id="ssm"),
        pytest.param(cbadc.circuit_level.AnalogSystemIdealOpAmp, id="ideal_opamp"),
        # pytest.param(
        #     cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp,
        #     id="first_order_pole_opamp",
        # ),
        # pytest.param(
        #     cbadc.circuit_level.AnalogSystemHigherOrderOpAmp,
        #     id="general_order_pole_opamp",
        # ),
    ],
)
def test_verilog_ams_in_cadence(
    N, ENOB, BW, analog_system, eta2, analog_circuit_implementation
):
    # Known limitations:
    if (
        ENOB > 12
        and analog_system == 'chain-of-integrators'
        and analog_circuit_implementation
        == cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp
    ):
        pytest.skip("Known limitation")

    # Instantiate analog frontend
    if analog_system == 'chain-of-integrators':
        xi = 6e-2
        AF = cbadc.synthesis.get_chain_of_integrator(
            N=N, ENOB=ENOB, BW=BW, xi=xi, finite_gain=True
        )
    elif analog_system == 'leap_frog':
        xi = 1e-1
        AF = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW, xi=xi)
    else:
        raise ValueError("Unknown analog system")
    C = 1e-12
    if analog_circuit_implementation == cbadc.circuit_level.AnalogSystemIdealOpAmp:
        verilog_analog_system = cbadc.circuit_level.AnalogSystemIdealOpAmp(
            analog_system=AF.analog_system, C=C
        )
    elif (
        analog_circuit_implementation
        == cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp
    ):
        A_DC = 2e3
        GBWP = 2 * np.pi * BW * A_DC
        verilog_analog_system = cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp(
            analog_system=AF.analog_system, C=C, A_DC=A_DC, GBWP=GBWP
        )
    elif (
        analog_circuit_implementation
        == cbadc.circuit_level.AnalogSystemStateSpaceEquations
    ):
        verilog_analog_system = cbadc.circuit_level.AnalogSystemStateSpaceEquations(
            analog_system=AF.analog_system
        )
    elif (
        analog_circuit_implementation
        == cbadc.circuit_level.AnalogSystemHigherOrderOpAmp
    ):
        amplifier_order = 2
        cutoff_freq = BW / 2
        pass_band_ripple = 3
        stop_band_ripple = 12
        # Set frequency characteristics of the op-amp
        amplifiers = [
            cbadc.analog_system.Cauer(
                amplifier_order, cutoff_freq, pass_band_ripple, stop_band_ripple
            )
            for _ in range(N)
        ]
        # Add amplification
        amplification = 1e3
        for amp in amplifiers:
            amp.B = -amplification * amp.B

        verilog_analog_system = cbadc.circuit_level.AnalogSystemHigherOrderOpAmp(
            analog_system=AF.analog_system, C=C, amplifiers=amplifiers
        )
    else:
        raise ValueError("Unknown analog_circuit_implementation")

    verilog_digital_control = cbadc.circuit_level.DigitalControl(
        copy.deepcopy(AF.digital_control)
    )

    verilog_analog_frontend = cbadc.circuit_level.AnalogFrontend(
        verilog_analog_system, verilog_digital_control
    )

    CLK = AF.digital_control.clock

    vdd = 1
    vi = vdd / 4
    f_clk = 1 / CLK.T
    fi = f_clk
    while fi > BW / 2:
        fi = fi / 2

    size = 1 << 15

    # Instantiate testbench and write to file
    VS = cbadc.analog_signal.Sinusoidal(vi, fi)
    TB = cbadc.circuit_level.TestBench(
        verilog_analog_frontend, VS, CLK, number_of_samples=size
    )

    if DEBUG:
        tb_filename = "verilog_debug_testbench.txt"
        TB.to_file(tb_filename)

    K1 = 1 << 10
    K2 = K1
    if eta2 == 'snr':
        eta2 = (
            np.linalg.norm(
                verilog_analog_system.analog_system.transfer_function_matrix(
                    np.array([2 * np.pi * BW])
                )
            )
            ** 2
        )

    digital_estimator = verilog_analog_frontend.get_estimator(
        cbadc.digital_estimator.BatchEstimator,
        eta2,
        K1,
        K2,
    )

    python_simulator = TB.get_simulator(
        cbadc.simulator.SimulatorType.pre_computed_numerical
    )

    digital_estimator(python_simulator)

    u_hat = np.zeros(size)

    for index in range(size):
        u_hat[index] = next(digital_estimator)

    u_hat_cut = u_hat[K1 + K2 :]

    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=1 / CLK.T, nperseg=u_hat_cut.size
    )

    if DEBUG:

        plt.title(f"Power spectral density:\nN={N},as={analog_system},ENOB={ENOB}")
        plt.semilogx(
            f,
            10 * np.log10(np.abs(psd)),
            label=f"Verilog-AMS, eta2={eta2}",
        )
        plt.xlabel('Hz')
        plt.ylabel('V^2 / Hz dB')
        plt.legend()
        plt.savefig('debug_psd.png')
        print("uhat:")
        print(u_hat)
        print(verilog_analog_system.analog_system.A)
        print(digital_estimator)

    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / CLK.T
    )
    est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)

    print(est_ENOB)
    assert est_ENOB >= ENOB
