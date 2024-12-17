import cProfile
from pstats import Stats
import sys


from cbadc.digital_estimator import BatchEstimator
from cbadc.digital_estimator._filter_coefficients import FilterComputationBackend
from cbadc.fom import snr_from_dB, enob_to_snr
from .fixtures import setup_filter


def run_script_under_test():
    N = 6
    ENOB = 12
    BW = 1e7
    analog_filter = "chain-of-integrators"
    # analog_filter = 'leap_frog'
    digital_control = "default"
    # digital_control = 'switch-cap'

    # solver_type = FilterComputationBackend.mpmath
    solver_type = FilterComputationBackend.numpy

    res = setup_filter(N, ENOB, BW, analog_filter, digital_control)

    K1 = 1 << 8
    K2 = K1

    eta2 = snr_from_dB(enob_to_snr(ENOB))

    BatchEstimator(
        res["analog_filter"],
        res["digital_control"],
        eta2,
        K1,
        K2,
        solver_type=solver_type,
    )


if __name__ == "__main__":
    # Profile script
    with cProfile.Profile() as pr:
        run_script_under_test()
    pr.dump_stats(f"{sys.argv[0].split('.')[0]}.prof")

    with open(f"{sys.argv[0].split('.')[0]}.txt", "w") as stream:
        stats = Stats(pr, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("time")
        stats.dump_stats(".prof_stats")
        stats.print_stats()
