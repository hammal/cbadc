from cbadc.analog_signal.impulse_responses import StepResponse, RCImpulseResponse
import sympy as sp


def test_step_response():
    t0 = 0.1
    analog_signal = StepResponse(t0)
    assert analog_signal.t0 == t0
    assert sp.Eq(sp.Symbol("t", real=True) - analog_signal.t, 0)
    sp.pprint(analog_signal.symbolic())


def test_RC_impulse_response():
    t0 = 0.5
    tau = 1e-3
    analog_signal = RCImpulseResponse(tau, t0)
    t = sp.Symbol("t", real=True)
    assert analog_signal.t0 == t0
    assert analog_signal.tau == tau
    sp.pprint(analog_signal.symbolic())
    sp.pprint(sp.Piecewise((sp.exp((t0 - t) / tau), t >= t0), (0, True)))
    # assert sp.Eq(
    #     analog_signal.symbolic() - sp.Piecewise((sp.exp((t0 - t)/tau), t >= t0), (0, True)),
    #     0
    # )
