import cbadc
import cbadc.datasets.chain_of_integrators
import numpy as np

chain_of_integrators_pre_simulations = {}
try:
    chain_of_integrators_pre_simulations = cbadc.utilities.pickle_load(
        'chain_of_integrators.pickle')
except FileNotFoundError:
    print("Starting a new dictionary.")

simulation_length = 1 << 34
file_size = 1 << 29

N = [i for i in range(1, 10)]
beta = 6250.
Rho = - np.linspace(0, beta, 10)
kappa = - 1

omega = 1.0 / (2 * beta)

amplitudes = np.logspace(-10, 0, 100)
frequencies = [omega / (2 * np.pi * OSR)
               for OSR in [1 << exponent for exponent in range(12)]]
phase = 0.0
offset = 0.0

for n in N:
    for rho in Rho:
        chain = cbadc.datasets.chain_of_integrators.ChainOfIntegrators(
            n,
            beta,
            rho,
            kappa
        )
        for amp in amplitudes:
            for freq in frequencies:
                print(
                    f"Simulating: N = {n}, rho = {rho}, amp = {amp}, freq = {freq}")
                ctrl_sig, simulator, size = chain.sin(amp, freq, phase, offset)
                byte_stream = cbadc.utilities.finite_iteration(cbadc.utilities.control_signal_2_byte_stream(
                    ctrl_sig, n), simulation_length)
                base_file_name = f"data/chain_of_integrators.N={n}_beta={beta}_rho={rho}_kappa={kappa}.amplitude={amp}_frequency={freq}_phase={phase}_offset={offset}.cbs"

                key = (n, beta, rho, kappa, 'sin', amp, freq, phase, offset)
                value = cbadc.utilities.write_byte_stream_to_files(
                    base_file_name, byte_stream, words_per_file=file_size)
                if key not in chain_of_integrators_pre_simulations:
                    chain_of_integrators_pre_simulations[key] = value


cbadc.utilities.pickle_dump(
    chain_of_integrators_pre_simulations, 'chain_of_integrators.pickle')
