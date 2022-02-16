from tests.fixture.chain_of_integrators import get_simulator, chain_of_integrators
import pytest
import cbadc.circuit_level.state_space_equations


def test_AnalogSystemModule(chain_of_integrators):
    analog_system_module = cbadc.circuit_level.state_space_equations.AnalogSystem(
        chain_of_integrators['system']
    )
    assert (
        analog_system_module.render()[0][-1]
        == """// analog_system
// 
// Ports: vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4
// 
// Parameters: 
// 
// Functional Description
// 
// The analog system directly modeled using differential
// equations.
// 
// Specifically, we use the state space model equations
// 
// ddt(x(t)) = A x(t) + B u(t) + Gamma s(t)
// s_tilde(t) = Gamma_tildeT x(t)
// 
// where
// 
// x(t) = [x_0, x_1, x_2, x_3, x_4]^T
// u(t) = [u_0]^T
// s(t) = [s_0, s_1, s_2, s_3, s_4]^T
// s_tilde(t) = [s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4]^T
// 
// A ≈
// [-6.25e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [6.25e+03, -6.25e+01, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 6.25e+03, -6.25e+01, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 6.25e+03, -6.25e+01, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 6.25e+03, -6.25e+01]
// 
// B ≈
// [6.25e+03]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
// 
// Gamma ≈
// [-6.25e+03, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]
// [-0.00e+00, -6.25e+03, -0.00e+00, -0.00e+00, -0.00e+00]
// [-0.00e+00, -0.00e+00, -6.25e+03, -0.00e+00, -0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03, -0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03]
// 
// Gamma_tildeT ≈
// [1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]
//
module analog_system(vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4);

    input vdd; // positive supply
    input vgd; // ground
    input vsgd; // signal ground
    input u_0;
    input s_0;
    input s_1;
    input s_2;
    input s_3;
    input s_4;

    output s_tilde_0;
    output s_tilde_1;
    output s_tilde_2;
    output s_tilde_3;
    output s_tilde_4;


    analog begin
        ddt(V(x_0), sgd) <+ -62.5*V(x_0, sgd) -6250.0*V(s_0, sgd) 6250.0*V(u_0, sgd);
        ddt(V(x_1), sgd) <+ 6250.0*V(x_0, sgd) -62.5*V(x_1, sgd) -6250.0*V(s_1, sgd);
        ddt(V(x_2), sgd) <+ 6250.0*V(x_1, sgd) -62.5*V(x_2, sgd) -6250.0*V(s_2, sgd);
        ddt(V(x_3), sgd) <+ 6250.0*V(x_2, sgd) -62.5*V(x_3, sgd) -6250.0*V(s_3, sgd);
        ddt(V(x_4), sgd) <+ 6250.0*V(x_3, sgd) -62.5*V(x_4, sgd) -6250.0*V(s_4, sgd);
        V(s_tilde_0, sgd) <+ 1.0*V(x_0, sgd);
        V(s_tilde_1, sgd) <+ 1.0*V(x_1, sgd);
        V(s_tilde_2, sgd) <+ 1.0*V(x_2, sgd);
        V(s_tilde_3, sgd) <+ 1.0*V(x_3, sgd);
        V(s_tilde_4, sgd) <+ 1.0*V(x_4, sgd);
    end

endmodule"""
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
