from cbadc.synthesis import get_chain_of_integrator, get_leap_frog


def setup_filter(
    N, ENOB, BW, analog_filter, digital_control, excess_delay=0.0, local_feedback=False
):
    res = {"N": N, "ENOB": ENOB, "BW": BW}
    if analog_filter == "chain-of-integrators":
        analog_frontend = get_chain_of_integrator(
            N=N,
            BW=BW,
            ENOB=ENOB,
            digital_control=digital_control,
            excess_delay=excess_delay,
            local_feedback=local_feedback,
            xi=6e-3,
        )
        analog_filter = analog_frontend.analog_filter
        digital_control = analog_frontend.digital_control
        res["analog_filter"] = analog_filter
        res["digital_control"] = digital_control
    elif analog_filter == "leap_frog":
        analog_frontend = get_leap_frog(
            N=N,
            BW=BW,
            ENOB=ENOB,
            digital_control=digital_control,
            excess_delay=excess_delay,
            local_feedback=local_feedback,
            xi=9e-3,
        )
        analog_filter = analog_frontend.analog_filter
        digital_control = analog_frontend.digital_control
        res["analog_filter"] = analog_filter
        res["digital_control"] = digital_control
    return res


def pytest_generate_tests(metafunc):
    if all(
        fix in metafunc.fixturenames
        for fix in ("analog_filter_and_digital_control", "N", "BW", "ENOB")
    ):
        N = metafunc.config.getoption("N")
        BW = metafunc.config.getoption("BW")
        ENOB = metafunc.config.getoption("ENOB")
        analog_filter_and_digital_control_CI = get_chain_of_integrator(
            N=N, BW=BW, ENOB=ENOB
        )
        analog_filter_and_digital_control_LF = get_leap_frog(N=N, BW=BW, ENOB=ENOB)
        metafunc.parametrize(
            "analog_filter_and_digital_control",
            [
                (
                    analog_filter_and_digital_control_CI.analog_filter,
                    analog_filter_and_digital_control_CI.digital_control,
                ),
                (
                    analog_filter_and_digital_control_LF.analog_filter,
                    analog_filter_and_digital_control_LF.digital_control,
                ),
            ],
            ids=["chain-of-integrators", "leap-frog"],
        )
