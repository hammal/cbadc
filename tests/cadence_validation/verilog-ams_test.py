import pytest
import cbadc
from pade.utils import init_logger
from pade.spectre import simulate_netlist
from pade.psf_parser import PSFParser
import cbadc
import copy
import shlib
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
    ],
)
@pytest.mark.parametrize(
    "ENOB",
    [
        # pytest.param(10, id="ENOB=10"),
        pytest.param(12, id="ENOB=12"),
        # pytest.param(14, id="ENOB=14"),
        # pytest.param(16, id="ENOB=16"),
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
        # pytest.param(1e4, id="BW=10kHz"),
        # pytest.param(1e5, id="BW=100kHz"),
        pytest.param(1e6, id="BW=1MHz"),
        # pytest.param(1e7, id="BW=10MHz"),
        # pytest.param(1e8, id="BW=100MHz"),
        # pytest.param(1e9, id="BW=1GHz"),
    ],
)
@pytest.mark.parametrize(
    "analog_system",
    [
        # pytest.param('chain-of-integrators', id="chain_of_integrators_as"),
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
        # pytest.param(cbadc.circuit_level.AnalogSystemStateSpaceEquations, id="ssm"),
        pytest.param(cbadc.circuit_level.AnalogSystemIdealOpAmp, id="ideal_opamp"),
        # pytest.param(
        #     cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp,
        #     id="first_order_pole_opamp",
        # ),
        # pytest.param(
        #     cbadc.circuit_level.AnalogSystemStateSpaceOpAmp,
        #     id="general_order_pole_opamp",
        # ),
    ],
)
def test_verilog_ams_in_cadence(
    N, ENOB, BW, analog_system, eta2, analog_circuit_implementation
):
    # if N >= 12 and analog_system == 'leap_frog' and BW >= 1e8:
    #     pytest.skip("Known limitation")

    work_dir = shlib.to_path(__file__).parent
    logger = init_logger()
    # Instantiate analog frontend
    if analog_system == 'chain-of-integrators':
        AF = cbadc.synthesis.get_chain_of_integrator(N=N, ENOB=ENOB, BW=BW)
    elif analog_system == 'leap_frog':
        AF = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW)
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
        A_DC = 1e3
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
    f_clk = 1/CLK.T
    fi = f_clk
    while fi > BW/2:
        fi = fi/2

    # Instantiate testbench and write to file
    VS = cbadc.analog_signal.Sinusoidal(vi, fi)
    TB = cbadc.circuit_level.TestBench(verilog_analog_frontend, VS, CLK)
    tb_filename = "verilog_testbench.txt"
    TB.to_file(filename=tb_filename, path=work_dir)

    # Simulate
    simulate_netlist(logger, shlib.to_path(work_dir, tb_filename), work_dir=work_dir)

    # Parse the raw data file
    raw_data_dir = shlib.to_path(work_dir, 'simulation_output')
    parser = PSFParser(logger, raw_data_dir, 'tran')
    parser.parse()

    s_array = np.array(
        [parser.get_signal(f's_{index}', 'tran').trace for index in range(N)]
    ).transpose()
    simulated_control_signals = cbadc.simulator.NumpySimulator('', array=s_array)

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
        cbadc.digital_estimator.FIRFilter,
        eta2,
        K1,
        K2,
    )

    digital_estimator(simulated_control_signals)

    size = s_array.shape[0]
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
            # label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
        )
        plt.xlabel('Hz')
        plt.ylabel('V^2 / Hz dB')
        plt.legend()
        plt.savefig('debug_psd.png')
        print("s array:")
        print(s_array)
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

    assert est_ENOB >= ENOB
