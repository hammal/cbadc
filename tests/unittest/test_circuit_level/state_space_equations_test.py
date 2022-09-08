from tests.fixture.chain_of_integrators import get_simulator, chain_of_integrators
import cbadc.circuit_level.state_space_equations


def test_AnalogSystemModule(chain_of_integrators):
    analog_system_module = cbadc.circuit_level.state_space_equations.AnalogSystem(
        chain_of_integrators['system']
    )
    assert (
        analog_system_module.render()[0][-1]
        == """// analog_system\n// \n// Ports: vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4\n// \n// Parameters: \n// \n// Functional Description\n// \n// The analog system directly modeled using differential\n// equations.\n// \n// Specifically, we use the state space model equations\n// \n// ddt(x(t)) = A x(t) + B u(t) + Gamma s(t)\n// s_tilde(t) = Gamma_tildeT x(t)\n// \n// where\n// \n// x(t) = [x_0, x_1, x_2, x_3, x_4]^T\n// u(t) = [u_0]^T\n// s(t) = [s_0, s_1, s_2, s_3, s_4]^T\n// s_tilde(t) = [s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4]^T\n// \n// A ≈\n// [-6.25e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]\n// [6.25e+03, -6.25e+01, 0.00e+00, 0.00e+00, 0.00e+00]\n// [0.00e+00, 6.25e+03, -6.25e+01, 0.00e+00, 0.00e+00]\n// [0.00e+00, 0.00e+00, 6.25e+03, -6.25e+01, 0.00e+00]\n// [0.00e+00, 0.00e+00, 0.00e+00, 6.25e+03, -6.25e+01]\n// \n// B ≈\n// [6.25e+03]\n// [0.00e+00]\n// [0.00e+00]\n// [0.00e+00]\n// [0.00e+00]\n// \n// Gamma ≈\n// [-6.25e+03, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]\n// [-0.00e+00, -6.25e+03, -0.00e+00, -0.00e+00, -0.00e+00]\n// [-0.00e+00, -0.00e+00, -6.25e+03, -0.00e+00, -0.00e+00]\n// [-0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03, -0.00e+00]\n// [-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03]\n// \n// Gamma_tildeT ≈\n// [1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]\n// [0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]\n// [0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00]\n// [0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00]\n// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]\n//\nmodule analog_system(vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4);\n\n    input vdd; // positive supply\n    input vgd; // ground\n    input vsgd; // signal ground\n    input u_0;\n    input s_0;\n    input s_1;\n    input s_2;\n    input s_3;\n    input s_4;\n\n    output s_tilde_0;\n    output s_tilde_1;\n    output s_tilde_2;\n    output s_tilde_3;\n    output s_tilde_4;\n\n    electrical vdd; // positive supply\n    electrical vgd; // ground\n    electrical vsgd; // signal ground\n    electrical u_0;\n    electrical s_0;\n    electrical s_1;\n    electrical s_2;\n    electrical s_3;\n    electrical s_4;\n    electrical s_tilde_0;\n    electrical s_tilde_1;\n    electrical s_tilde_2;\n    electrical s_tilde_3;\n    electrical s_tilde_4;\n    electrical x_0;\n    electrical x_1;\n    electrical x_2;\n    electrical x_3;\n    electrical x_4;\n\n    analog begin\n        ddt(V(x_0, vsgd)) <+ -62.5*V(x_0, vsgd) -6250.0*V(s_0, vsgd) 6250.0*V(u_0, vsgd);\n        ddt(V(x_1, vsgd)) <+ 6250.0*V(x_0, vsgd) -62.5*V(x_1, vsgd) -6250.0*V(s_1, vsgd);\n        ddt(V(x_2, vsgd)) <+ 6250.0*V(x_1, vsgd) -62.5*V(x_2, vsgd) -6250.0*V(s_2, vsgd);\n        ddt(V(x_3, vsgd)) <+ 6250.0*V(x_2, vsgd) -62.5*V(x_3, vsgd) -6250.0*V(s_3, vsgd);\n        ddt(V(x_4, vsgd)) <+ 6250.0*V(x_3, vsgd) -62.5*V(x_4, vsgd) -6250.0*V(s_4, vsgd);\n        V(s_tilde_0, vsgd) <+ 1.0*V(x_0, vsgd);\n        V(s_tilde_1, vsgd) <+ 1.0*V(x_1, vsgd);\n        V(s_tilde_2, vsgd) <+ 1.0*V(x_2, vsgd);\n        V(s_tilde_3, vsgd) <+ 1.0*V(x_3, vsgd);\n        V(s_tilde_4, vsgd) <+ 1.0*V(x_4, vsgd);\n    end\n\nendmodule"""
    )


def test_AnalogSystemModule_module_comment(chain_of_integrators):
    analog_system_module = cbadc.circuit_level.state_space_equations.AnalogSystem(
        chain_of_integrators['system']
    )
    assert (
        str(analog_system_module)
        == """analog_system

Ports: vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4

Parameters:

Functional Description

The analog system directly modeled using differential
equations.

Specifically, we use the state space model equations

ddt(x(t)) = A x(t) + B u(t) + Gamma s(t)
s_tilde(t) = Gamma_tildeT x(t)

where

x(t) = [x_0, x_1, x_2, x_3, x_4]^T
u(t) = [u_0]^T
s(t) = [s_0, s_1, s_2, s_3, s_4]^T
s_tilde(t) = [s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4]^T

A ≈
[-6.25e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
[6.25e+03, -6.25e+01, 0.00e+00, 0.00e+00, 0.00e+00]
[0.00e+00, 6.25e+03, -6.25e+01, 0.00e+00, 0.00e+00]
[0.00e+00, 0.00e+00, 6.25e+03, -6.25e+01, 0.00e+00]
[0.00e+00, 0.00e+00, 0.00e+00, 6.25e+03, -6.25e+01]

B ≈
[6.25e+03]
[0.00e+00]
[0.00e+00]
[0.00e+00]
[0.00e+00]

Gamma ≈
[-6.25e+03, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]
[-0.00e+00, -6.25e+03, -0.00e+00, -0.00e+00, -0.00e+00]
[-0.00e+00, -0.00e+00, -6.25e+03, -0.00e+00, -0.00e+00]
[-0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03, -0.00e+00]
[-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03]

Gamma_tildeT ≈
[1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
[0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
[0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00]
[0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00]
[0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]"""
    )
