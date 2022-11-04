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
if __name__ == '__main__':
    REWRITE_NETLIST = 1
    DEBUG = 0
    RERUN_SIM = 1

    N = 4
    ENOB = 12
    BW = 1e6
    analog_system = 'leap_frog'
    eta2 = 'snr'

    # if N >= 12 and analog_system == 'leap_frog' and BW >= 1e8:
    #     pytest.skip("Known limitation")

    work_dir = shlib.to_path(__file__).parent
    logger = init_logger()
    # Instantiate analog frontend
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

    A_DC = 1e3
    GBWP = 2 * np.pi * BW * A_DC
    # AF.analog_system.A *= -1
    # AF.analog_system.B *= -1
    # AF.analog_system.Gamma *= -1

    verilog_analog_system = cbadc.circuit.AnalogSystemFirstOrderPoleOpAmp(
        analog_system=AF.analog_system, C=C, A_DC=A_DC, GBWP=GBWP
    )

    verilog_digital_control = cbadc.circuit.DigitalControl(
        copy.deepcopy(AF.digital_control)
    )

    verilog_analog_frontend = cbadc.circuit.AnalogFrontend(
        verilog_analog_system, verilog_digital_control
    )

    CLK = AF.digital_control.clock
    CLK.tt = CLK.T * 1e-6

    vdd = 1
    vi = vdd / 2
    f_clk = 1 / CLK.T
    fi = f_clk
    while fi > BW / 2:
        fi = fi / 2

    size = 1 << 14

    # Instantiate testbench and write to file
    VS = cbadc.analog_signal.Sinusoidal(vi, fi, offset=vdd / 2)
    TB = cbadc.circuit.TestBench(
        verilog_analog_frontend, VS, CLK, number_of_samples=size
    )
    tb_filename = "verilog_testbench.txt"
    if REWRITE_NETLIST:
        TB.to_file(filename=tb_filename, path=work_dir)

    print(verilog_analog_system)

    # Simulate
    if RERUN_SIM:
        simulate_netlist(
            logger, shlib.to_path(work_dir, tb_filename), work_dir=work_dir
        )

    if DEBUG:
        logger.info('DEBUG active, terminating')
        quit()

    # Parse the raw data file
    raw_data_dir = shlib.to_path(work_dir, 'simulation_output')
    parser = PSFParser(logger, raw_data_dir, 'tran')
    parser.parse()

    s_array = np.array(
        [parser.get_signal(f's_{index}', 'tran').trace for index in range(N)]
    ).transpose()
    u_0 = parser.get_signal('u_0', 'tran').trace
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

    # size = s_array.shape[0]
    u_hat = np.zeros(size)
    for index in range(size):
        u_hat[index] = next(digital_estimator)
    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=1 / CLK.T, nperseg=u_hat_cut.size
    )

    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / CLK.T
    )

    est_SNR = cbadc.fom.snr_to_dB(fom['snr'])

    plt.figure()
    plt.title(f"Power spectral density:\nN={N},as={analog_system},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"Verilog-AMS, SNR={est_SNR}",
    )

    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    plt.savefig('debug_psd.png')
