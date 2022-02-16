from tests.fixture.chain_of_integrators import (
    get_simulator,
    chain_of_integrators,
    chain_of_integrators_op_amp,
    chain_of_integrators_op_amp_small,
)

# import cbadc.circuit_level
import cbadc.circuit_level.op_amp.op_amp
import cbadc.circuit_level.op_amp.resistor_network
import cbadc.circuit_level.op_amp.amplifier_configurations
import numpy as np
import cbadc.analog_system
import scipy.signal


def test_ideal_op_amp():

    op_amp = cbadc.circuit_level.op_amp.op_amp.IdealOpAmp("my_op_amp")
    assert (
        op_amp.render()[0][-1]
        == """// ideal_op_amp
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters:
//
// Functional Description:
//
// Ideal op-amp implementation.
//
module ideal_op_amp(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input


    analog begin
        V(out): V(p_in, n_in) == 0;
    end

endmodule"""
    )


def test_op_amp_with_first_order_pole():

    A_DC = 1e3
    omega_0 = 2 * np.pi * 1e3
    op_amp = cbadc.circuit_level.op_amp.op_amp.FirstOrderPoleOpAmp(
        "my_op_amp", A_DC, omega_0
    )
    assert (
        op_amp.render()[0][-1]
        == """// first_order_pole_op_amp
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: A_DC, omega_p
//
// Functional Description:
//
// Op-amp implementation including a
// first order pole.
//
// i.e.,
//
// ddt(V(out)) = A_DC * omega_p * (V(p_in) - V(n_in)) - omega_p * V(out)
//
module first_order_pole_op_amp(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real A_DC = 1000.0;
    parameter real omega_p = 6283.185307179586;


    analog begin
        V(out) <+ A_DC * laplace_qp(V(p_in, n_in), , {-omega_p, 0});
    end

endmodule"""
    )


def test_op_amp_with_finite_gain():

    A_DC = 1e3
    op_amp = cbadc.circuit_level.op_amp.op_amp.FiniteGainOpAmp("my_op_amp", A_DC)
    assert (
        op_amp.render()[0][-1]
        == """// finite_gain_op_amp
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: A_DC
//
// Functional Description:
//
// A finite gain op-amp implementation
// where
// V(out) = A_DC * (V(p_in) - V(n_in))
//
module finite_gain_op_amp(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real A_DC = 1000.0;


    analog begin
        V(out) <+  A_DC * V(p_in, n_in);
    end

endmodule"""
    )


def test_op_amp_integrator_ideal():

    C = 1e-12
    op_amp = cbadc.circuit_level.op_amp.amplifier_configurations.InvertingAmplifierCapacitiveFeedback(
        "my_op_amp", C, cbadc.circuit_level.op_amp.op_amp.IdealOpAmp
    )
    assert (
        op_amp.render()[0][-1]
        == """// inverting_amplifier_my_op_amp
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_my_op_amp(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    ideal_op_amp op_amp_my_op_amp (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule"""
    )


def test_op_amp_integrator_finite_gain():

    A_DC = 1e3
    C = 1e-12
    op_amp = cbadc.circuit_level.op_amp.amplifier_configurations.InvertingAmplifierCapacitiveFeedback(
        "my_op_amp", C, cbadc.circuit_level.op_amp.op_amp.FiniteGainOpAmp, A_DC=A_DC
    )
    assert (
        op_amp.render()[0][-1]
        == """// inverting_amplifier_my_op_amp
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_my_op_amp(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    finite_gain_op_amp op_amp_my_op_amp (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule"""
    )


def test_op_amp_integrator_first_order_pole():

    A_DC = 1e3
    omega_p = 2 * np.pi * 1e3
    C = 1e-12
    op_amp = cbadc.circuit_level.op_amp.amplifier_configurations.InvertingAmplifierCapacitiveFeedback(
        "my_op_amp",
        C,
        cbadc.circuit_level.op_amp.op_amp.FirstOrderPoleOpAmp,
        A_DC=A_DC,
        omega_p=omega_p,
    )
    assert (
        op_amp.render()[0][-1]
        == """// inverting_amplifier_my_op_amp
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_my_op_amp(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    first_order_pole_op_amp op_amp_my_op_amp (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule"""
    )


def test_resistor_network(chain_of_integrators):
    C = 1e-12
    G = chain_of_integrators['system'].A * C
    resistor_network_module = (
        cbadc.circuit_level.op_amp.resistor_network.ResistorNetwork(
            "module_name", "instance_name", G
        )
    )
    assert (
        resistor_network_module.render()[0][-1]
        == """// module_name
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [-1.60e+10, inf, inf, inf, inf] [in_0]
// [out_1] ≈ [1.60e+08, -1.60e+10, inf, inf, inf] [in_1]
// [out_2] ≈ [inf, 1.60e+08, -1.60e+10, inf, inf] [in_2]
// [out_3] ≈ [inf, inf, 1.60e+08, -1.60e+10, inf] [in_3]
// [out_4] ≈ [inf, inf, inf, 1.60e+08, -1.60e+10] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module module_name(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ -6.25e-11 * V(in_0,out_0);
        I(in_0, out_1) <+ 6.25e-09 * V(in_0,out_1);
        I(in_1, out_1) <+ -6.25e-11 * V(in_1,out_1);
        I(in_1, out_2) <+ 6.25e-09 * V(in_1,out_2);
        I(in_2, out_2) <+ -6.25e-11 * V(in_2,out_2);
        I(in_2, out_3) <+ 6.25e-09 * V(in_2,out_3);
        I(in_3, out_3) <+ -6.25e-11 * V(in_3,out_3);
        I(in_3, out_4) <+ 6.25e-09 * V(in_3,out_4);
        I(in_4, out_4) <+ -6.25e-11 * V(in_4,out_4);
    end

endmodule"""
    )


def test_analog_system_ideal_op_amp(chain_of_integrators_op_amp):
    C = 1e-12
    analog_system_module = cbadc.circuit_level.op_amp.AnalogSystemIdealOpAmp(
        chain_of_integrators_op_amp['system'], C
    )
    assert (
        (3 * "\n").join(analog_system_module.render()[0])
        == """// gamma_tildeT_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.00e+12, inf, inf, inf, inf] [in_0]
// [out_1] ≈ [inf, 1.00e+12, inf, inf, inf] [in_1]
// [out_2] ≈ [inf, inf, 1.00e+12, inf, inf] [in_2]
// [out_3] ≈ [inf, inf, inf, 1.00e+12, inf] [in_3]
// [out_4] ≈ [inf, inf, inf, inf, 1.00e+12] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module gamma_tildeT_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 1e-12 * V(in_0,out_0);
        I(in_1, out_1) <+ 1e-12 * V(in_1,out_1);
        I(in_2, out_2) <+ 1e-12 * V(in_2,out_2);
        I(in_3, out_3) <+ 1e-12 * V(in_3,out_3);
        I(in_4, out_4) <+ 1e-12 * V(in_4,out_4);
    end

endmodule


// gamma_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.60e+08, inf, inf, inf, inf] [in_0]
// [out_1] ≈ [inf, 1.60e+08, inf, inf, inf] [in_1]
// [out_2] ≈ [inf, inf, 1.60e+08, inf, inf] [in_2]
// [out_3] ≈ [inf, inf, inf, 1.60e+08, inf] [in_3]
// [out_4] ≈ [inf, inf, inf, inf, 1.60e+08] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module gamma_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 6.25e-09 * V(in_0,out_0);
        I(in_1, out_1) <+ 6.25e-09 * V(in_1,out_1);
        I(in_2, out_2) <+ 6.25e-09 * V(in_2,out_2);
        I(in_3, out_3) <+ 6.25e-09 * V(in_3,out_3);
        I(in_4, out_4) <+ 6.25e-09 * V(in_4,out_4);
    end

endmodule


// b_matrix
//
// Ports: in_0, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.60e+08] [in_0]
// [out_1] ≈ [-inf] [in_1]
// [out_2] ≈ [-inf] [in_2]
// [out_3] ≈ [-inf] [in_3]
// [out_4] ≈ [-inf] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module b_matrix(in_0, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 6.25e-09 * V(in_0,out_0);
    end

endmodule


// a_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [-inf, -inf, -inf, -inf, -inf] [in_0]
// [out_1] ≈ [1.60e+08, -inf, -inf, -inf, -inf] [in_1]
// [out_2] ≈ [-inf, 1.60e+08, -inf, -inf, -inf] [in_2]
// [out_3] ≈ [-inf, -inf, 1.60e+08, -inf, -inf] [in_3]
// [out_4] ≈ [-inf, -inf, -inf, 1.60e+08, -inf] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module a_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_1) <+ 6.25e-09 * V(in_0,out_1);
        I(in_1, out_2) <+ 6.25e-09 * V(in_1,out_2);
        I(in_2, out_3) <+ 6.25e-09 * V(in_2,out_3);
        I(in_3, out_4) <+ 6.25e-09 * V(in_3,out_4);
    end

endmodule


// inverting_amplifier_int_4
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_int_4(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    ideal_op_amp op_amp_int_4 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// inverting_amplifier_int_3
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_int_3(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    ideal_op_amp op_amp_int_3 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// inverting_amplifier_int_2
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_int_2(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    ideal_op_amp op_amp_int_2 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// inverting_amplifier_int_1
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_int_1(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    ideal_op_amp op_amp_int_1 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// ideal_op_amp
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters:
//
// Functional Description:
//
// Ideal op-amp implementation.
//
module ideal_op_amp(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input


    analog begin
        V(out): V(p_in, n_in) == 0;
    end

endmodule


// inverting_amplifier_int_0
//
// Ports: vdd, vgd, p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_int_0(vdd, vgd, p_in, n_in, out);

    input vdd; // positive supply
    input vgd; // ground
    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    ideal_op_amp op_amp_int_0 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// analog_system
//
// Ports: vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4
//
// Parameters:
//
// Functional Description
//
// An analog system enforcing the differential equations.
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
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [-6.25e+03, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, -6.25e+03, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, -6.25e+03, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, -6.25e+03, 0.00e+00]
//
// B ≈
// [-6.25e+03]
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
// CT ≈
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]
//
module analog_system(vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4);

    input vdd; // positive supply
    input vgd; // ground
    input vsgd; // signal ground
    input u_0; // input channel 0
    input s_0; // control signal 0
    input s_1; // control signal 1
    input s_2; // control signal 2
    input s_3; // control signal 3
    input s_4; // control signal 4

    output s_tilde_0; // control observation 0
    output s_tilde_1; // control observation 1
    output s_tilde_2; // control observation 2
    output s_tilde_3; // control observation 3
    output s_tilde_4; // control observation 4



    inverting_amplifier_int_0 int_0 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(vsgd),
            .n_in(vgd_0),
            .out(x_0)
    );

    inverting_amplifier_int_1 int_1 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(vsgd),
            .n_in(vgd_1),
            .out(x_1)
    );

    inverting_amplifier_int_2 int_2 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(vsgd),
            .n_in(vgd_2),
            .out(x_2)
    );

    inverting_amplifier_int_3 int_3 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(vsgd),
            .n_in(vgd_3),
            .out(x_3)
    );

    inverting_amplifier_int_4 int_4 (
            .vdd(vdd),
            .vgd(vgd),
            .p_in(vsgd),
            .n_in(vgd_4),
            .out(x_4)
    );

    a_matrix a_matrix_0 (
            .in_0(x_0),
            .in_1(x_1),
            .in_2(x_2),
            .in_3(x_3),
            .in_4(x_4),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    b_matrix b_matrix_0 (
            .in_0(u_0),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    gamma_matrix gamma_matrix_0 (
            .in_0(s_0),
            .in_1(s_1),
            .in_2(s_2),
            .in_3(s_3),
            .in_4(s_4),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    gamma_tildeT_matrix gamma_tildeT_matrix_0 (
            .in_0(x_0),
            .in_1(x_1),
            .in_2(x_2),
            .in_3(x_3),
            .in_4(x_4),
            .out_0(s_tilde_0),
            .out_1(s_tilde_1),
            .out_2(s_tilde_2),
            .out_3(s_tilde_3),
            .out_4(s_tilde_4)
    );

endmodule"""
    )


def test_analog_system_final_gain_op_amp(chain_of_integrators_op_amp):
    C = 1e-12
    A_DC = 1e2
    analog_system = chain_of_integrators_op_amp['system']
    analog_system_module = cbadc.circuit_level.op_amp.AnalogSystemFiniteGainOpAmp(
        chain_of_integrators_op_amp['system'], C, A_DC
    )
    assert (
        (3 * "\n").join(analog_system_module.render()[0])
        == """// gamma_tildeT_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.00e+12, inf, inf, inf, inf] [in_0]
// [out_1] ≈ [inf, 1.00e+12, inf, inf, inf] [in_1]
// [out_2] ≈ [inf, inf, 1.00e+12, inf, inf] [in_2]
// [out_3] ≈ [inf, inf, inf, 1.00e+12, inf] [in_3]
// [out_4] ≈ [inf, inf, inf, inf, 1.00e+12] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module gamma_tildeT_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 1e-12 * V(in_0,out_0);
        I(in_1, out_1) <+ 1e-12 * V(in_1,out_1);
        I(in_2, out_2) <+ 1e-12 * V(in_2,out_2);
        I(in_3, out_3) <+ 1e-12 * V(in_3,out_3);
        I(in_4, out_4) <+ 1e-12 * V(in_4,out_4);
    end

endmodule


// gamma_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.58e+08, inf, inf, inf, inf] [in_0]
// [out_1] ≈ [inf, 1.58e+08, inf, inf, inf] [in_1]
// [out_2] ≈ [inf, inf, 1.58e+08, inf, inf] [in_2]
// [out_3] ≈ [inf, inf, inf, 1.58e+08, inf] [in_3]
// [out_4] ≈ [inf, inf, inf, inf, 1.58e+08] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module gamma_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 6.3125e-09 * V(in_0,out_0);
        I(in_1, out_1) <+ 6.3125e-09 * V(in_1,out_1);
        I(in_2, out_2) <+ 6.3125e-09 * V(in_2,out_2);
        I(in_3, out_3) <+ 6.3125e-09 * V(in_3,out_3);
        I(in_4, out_4) <+ 6.3125e-09 * V(in_4,out_4);
    end

endmodule


// b_matrix
//
// Ports: in_0, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.58e+08] [in_0]
// [out_1] ≈ [-inf] [in_1]
// [out_2] ≈ [-inf] [in_2]
// [out_3] ≈ [-inf] [in_3]
// [out_4] ≈ [-inf] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module b_matrix(in_0, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 6.3125e-09 * V(in_0,out_0);
    end

endmodule


// a_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [-inf, -inf, -inf, -inf, -inf] [in_0]
// [out_1] ≈ [1.58e+08, -inf, -inf, -inf, -inf] [in_1]
// [out_2] ≈ [-inf, 1.58e+08, -inf, -inf, -inf] [in_2]
// [out_3] ≈ [-inf, -inf, 1.58e+08, -inf, -inf] [in_3]
// [out_4] ≈ [-inf, -inf, -inf, 1.58e+08, -inf] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module a_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_1) <+ 6.3125e-09 * V(in_0,out_1);
        I(in_1, out_2) <+ 6.3125e-09 * V(in_1,out_2);
        I(in_2, out_3) <+ 6.3125e-09 * V(in_2,out_3);
        I(in_3, out_4) <+ 6.3125e-09 * V(in_3,out_4);
    end

endmodule


// finite_gain_op_amp
//
// Ports: p_in, n_in, out
//
// Parameters: A_DC
//
// Functional Description:
//
// A finite gain op-amp implementation
// where
// V(out) = A_DC * (V(p_in) - V(n_in))
//
module finite_gain_op_amp(p_in, n_in, out);

    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real A_DC = 100.0;


    analog begin
        V(out) <+  A_DC * V(p_in, n_in);
    end

endmodule


// op_amp_integrator
//
// Ports: p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module op_amp_integrator(p_in, n_in, out);

    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    finite_gain_op_amp int_0_op_amp (
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// analog_system
//
// Ports: vdd, gnd, sgd, u_0, s_0, s_1, s_2, s_3, s_4, x_0, x_1, x_2, x_3, x_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4
//
// Parameters:
//
// Functional Description
//
// An analog system enforcing the differential equations.
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
// [-1.25e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [-6.25e+03, -1.25e+02, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, -6.25e+03, -1.25e+02, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, -6.25e+03, -1.25e+02, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, -6.25e+03, -1.25e+02]
//
// B ≈
// [-6.25e+03]
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
module analog_system(vdd, gnd, sgd, u_0, s_0, s_1, s_2, s_3, s_4, x_0, x_1, x_2, x_3, x_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4);

    input vdd; // positive supply
    input gnd; // ground
    input sgd; // signal ground
    input u_0; // input channel 0
    input s_0; // control signal 0
    input s_1; // control signal 1
    input s_2; // control signal 2
    input s_3; // control signal 3
    input s_4; // control signal 4

    output x_0; // state variable number 0
    output x_1; // state variable number 1
    output x_2; // state variable number 2
    output x_3; // state variable number 3
    output x_4; // state variable number 4
    output s_tilde_0; // control observation 0
    output s_tilde_1; // control observation 1
    output s_tilde_2; // control observation 2
    output s_tilde_3; // control observation 3
    output s_tilde_4; // control observation 4



    op_amp_integrator int_0 (
            .p_in(sgd),
            .n_in(vgd_0),
            .out(x_0)
    );

    op_amp_integrator int_1 (
            .p_in(sgd),
            .n_in(vgd_1),
            .out(x_1)
    );

    op_amp_integrator int_2 (
            .p_in(sgd),
            .n_in(vgd_2),
            .out(x_2)
    );

    op_amp_integrator int_3 (
            .p_in(sgd),
            .n_in(vgd_3),
            .out(x_3)
    );

    op_amp_integrator int_4 (
            .p_in(sgd),
            .n_in(vgd_4),
            .out(x_4)
    );

    a_matrix a_matrix_0 (
            .in_0(x_0),
            .in_1(x_1),
            .in_2(x_2),
            .in_3(x_3),
            .in_4(x_4),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    b_matrix b_matrix_0 (
            .in_0(u_0),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    gamma_matrix gamma_matrix_0 (
            .in_0(s_0),
            .in_1(s_1),
            .in_2(s_2),
            .in_3(s_3),
            .in_4(s_4),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    gamma_tildeT_matrix gamma_tildeT_matrix_0 (
            .in_0(x_0),
            .in_1(x_1),
            .in_2(x_2),
            .in_3(x_3),
            .in_4(x_4),
            .out_0(s_tilde_0),
            .out_1(s_tilde_1),
            .out_2(s_tilde_2),
            .out_3(s_tilde_3),
            .out_4(s_tilde_4)
    );

endmodule"""
    )


def test_analog_system_first_order_pole_op_amp(chain_of_integrators_op_amp):
    C = 1e-12
    A_DC = 1e2
    omega_p = 1e4
    analog_system_module = cbadc.circuit_level.op_amp.AnalogSystemFirstOrderPoleOpAmp(
        chain_of_integrators_op_amp['system'], C, A_DC, omega_p
    )
    assert (
        (3 * "\n").join(analog_system_module.render()[0])
        == """// gamma_tildeT_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.00e+12, inf, inf, inf, inf] [in_0]
// [out_1] ≈ [inf, 1.00e+12, inf, inf, inf] [in_1]
// [out_2] ≈ [inf, inf, 1.00e+12, inf, inf] [in_2]
// [out_3] ≈ [inf, inf, inf, 1.00e+12, inf] [in_3]
// [out_4] ≈ [inf, inf, inf, inf, 1.00e+12] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module gamma_tildeT_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 1e-12 * V(in_0,out_0);
        I(in_1, out_1) <+ 1e-12 * V(in_1,out_1);
        I(in_2, out_2) <+ 1e-12 * V(in_2,out_2);
        I(in_3, out_3) <+ 1e-12 * V(in_3,out_3);
        I(in_4, out_4) <+ 1e-12 * V(in_4,out_4);
    end

endmodule


// gamma_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.58e+08, inf, inf, inf, inf] [in_0]
// [out_1] ≈ [inf, 1.58e+08, inf, inf, inf] [in_1]
// [out_2] ≈ [inf, inf, 1.58e+08, inf, inf] [in_2]
// [out_3] ≈ [inf, inf, inf, 1.58e+08, inf] [in_3]
// [out_4] ≈ [inf, inf, inf, inf, 1.58e+08] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module gamma_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 6.3125e-09 * V(in_0,out_0);
        I(in_1, out_1) <+ 6.3125e-09 * V(in_1,out_1);
        I(in_2, out_2) <+ 6.3125e-09 * V(in_2,out_2);
        I(in_3, out_3) <+ 6.3125e-09 * V(in_3,out_3);
        I(in_4, out_4) <+ 6.3125e-09 * V(in_4,out_4);
    end

endmodule


// b_matrix
//
// Ports: in_0, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.58e+08] [in_0]
// [out_1] ≈ [-inf] [in_1]
// [out_2] ≈ [-inf] [in_2]
// [out_3] ≈ [-inf] [in_3]
// [out_4] ≈ [-inf] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module b_matrix(in_0, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_0) <+ 6.3125e-09 * V(in_0,out_0);
    end

endmodule


// a_matrix
//
// Ports: in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [-inf, -inf, -inf, -inf, -inf] [in_0]
// [out_1] ≈ [1.58e+08, -inf, -inf, -inf, -inf] [in_1]
// [out_2] ≈ [-inf, 1.58e+08, -inf, -inf, -inf] [in_2]
// [out_3] ≈ [-inf, -inf, 1.58e+08, -inf, -inf] [in_3]
// [out_4] ≈ [-inf, -inf, -inf, 1.58e+08, -inf] [in_4]
//
// note the resistors are specified by their resistive values in Ohms
//
module a_matrix(in_0, in_1, in_2, in_3, in_4, out_0, out_1, out_2, out_3, out_4);


    inout in_0;
    inout in_1;
    inout in_2;
    inout in_3;
    inout in_4;
    inout out_0;
    inout out_1;
    inout out_2;
    inout out_3;
    inout out_4;


    analog begin
        I(in_0, out_1) <+ 6.3125e-09 * V(in_0,out_1);
        I(in_1, out_2) <+ 6.3125e-09 * V(in_1,out_2);
        I(in_2, out_3) <+ 6.3125e-09 * V(in_2,out_3);
        I(in_3, out_4) <+ 6.3125e-09 * V(in_3,out_4);
    end

endmodule


// first_order_pole_op_amp
//
// Ports: p_in, n_in, out
//
// Parameters: A_DC, omega_p
//
// Functional Description:
//
// Op-amp implementation including a
// first order pole.
//
// i.e.,
//
// ddt(V(out)) = A_DC * omega_p * (V(p_in) - V(n_in)) - omega_p * V(out)
//
module first_order_pole_op_amp(p_in, n_in, out);

    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real A_DC = 100.0;
    parameter real omega_p = 10000.0;


    analog begin
        V(out) <+ A_DC * laplace_qp(V(p_in, n_in), , {-omega_p, 0});
    end

endmodule


// op_amp_integrator
//
// Ports: p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module op_amp_integrator(p_in, n_in, out);

    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    first_order_pole_op_amp int_0_op_amp (
            .p_in(p_in),
            .n_in(n_in),
            .out(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// analog_system
//
// Ports: vdd, gnd, sgd, u_0, s_0, s_1, s_2, s_3, s_4, x_0, x_1, x_2, x_3, x_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4
//
// Parameters:
//
// Functional Description
//
// An analog system enforcing the differential equations.
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
// [-1.01e+06, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.00e+04, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [-0.00e+00, -1.01e+06, -0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03, -1.00e+04, 0.00e+00, 0.00e+00, 0.00e+00]
// [-0.00e+00, -0.00e+00, -1.01e+06, -0.00e+00, -0.00e+00, 0.00e+00, -6.25e+03, -1.00e+04, 0.00e+00, 0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -1.01e+06, -0.00e+00, 0.00e+00, 0.00e+00, -6.25e+03, -1.00e+04, 0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.01e+06, 0.00e+00, 0.00e+00, 0.00e+00, -6.25e+03, -1.00e+04]
// [-1.00e+06, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.00e+04, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]
// [-0.00e+00, -1.00e+06, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.00e+04, -0.00e+00, -0.00e+00, -0.00e+00]
// [-0.00e+00, -0.00e+00, -1.00e+06, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.00e+04, -0.00e+00, -0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -1.00e+06, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.00e+04, -0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.00e+06, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.00e+04]
//
// B ≈
// [-6.25e+03]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
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
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
//
// Gamma_tildeT ≈
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]
// CT ≈
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]
//
module analog_system(vdd, gnd, sgd, u_0, s_0, s_1, s_2, s_3, s_4, x_0, x_1, x_2, x_3, x_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4);

    input vdd; // positive supply
    input gnd; // ground
    input sgd; // signal ground
    input u_0; // input channel 0
    input s_0; // control signal 0
    input s_1; // control signal 1
    input s_2; // control signal 2
    input s_3; // control signal 3
    input s_4; // control signal 4

    output x_0; // state variable number 0
    output x_1; // state variable number 1
    output x_2; // state variable number 2
    output x_3; // state variable number 3
    output x_4; // state variable number 4
    output s_tilde_0; // control observation 0
    output s_tilde_1; // control observation 1
    output s_tilde_2; // control observation 2
    output s_tilde_3; // control observation 3
    output s_tilde_4; // control observation 4



    op_amp_integrator int_0 (
            .p_in(sgd),
            .n_in(vgd_0),
            .out(x_0)
    );

    op_amp_integrator int_1 (
            .p_in(sgd),
            .n_in(vgd_1),
            .out(x_1)
    );

    op_amp_integrator int_2 (
            .p_in(sgd),
            .n_in(vgd_2),
            .out(x_2)
    );

    op_amp_integrator int_3 (
            .p_in(sgd),
            .n_in(vgd_3),
            .out(x_3)
    );

    op_amp_integrator int_4 (
            .p_in(sgd),
            .n_in(vgd_4),
            .out(x_4)
    );

    a_matrix a_matrix_0 (
            .in_0(x_0),
            .in_1(x_1),
            .in_2(x_2),
            .in_3(x_3),
            .in_4(x_4),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    b_matrix b_matrix_0 (
            .in_0(u_0),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    gamma_matrix gamma_matrix_0 (
            .in_0(s_0),
            .in_1(s_1),
            .in_2(s_2),
            .in_3(s_3),
            .in_4(s_4),
            .out_0(vgd_0),
            .out_1(vgd_1),
            .out_2(vgd_2),
            .out_3(vgd_3),
            .out_4(vgd_4)
    );

    gamma_tildeT_matrix gamma_tildeT_matrix_0 (
            .in_0(x_0),
            .in_1(x_1),
            .in_2(x_2),
            .in_3(x_3),
            .in_4(x_4),
            .out_0(s_tilde_0),
            .out_1(s_tilde_1),
            .out_2(s_tilde_2),
            .out_3(s_tilde_3),
            .out_4(s_tilde_4)
    );

endmodule"""
    )


def test_analog_system_n_th_order_pole_op_amp(chain_of_integrators_op_amp_small):
    C = 1e-12
    N = 3
    Wn = 1e3

    amplifier = cbadc.analog_system.filters.ButterWorth(N, Wn)
    amplifier.B = -1e3 * amplifier.B

    analog_system_module = cbadc.circuit_level.op_amp.AnalogSystemStateSpaceOpAmp(
        chain_of_integrators_op_amp_small['system'],
        C,
        [amplifier],
    )
    assert (
        (3 * "\n").join(analog_system_module.render()[0])
        == """// gamma_tildeT_matrix
//
// Ports: in_0, in_1, out_0, out_1
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [1.00e+12, inf] [in_0]
// [out_1] ≈ [inf, 1.00e+12] [in_1]
//
// note the resistors are specified by their resistive values in Ohms
//
module gamma_tildeT_matrix(in_0, in_1, out_0, out_1);


    inout in_0;
    inout in_1;
    inout out_0;
    inout out_1;


    analog begin
        I(in_0, out_0) <+ 1e-12 * V(in_0,out_0);
        I(in_1, out_1) <+ 1e-12 * V(in_1,out_1);
    end

endmodule


// gamma_matrix
//
// Ports: in_0, in_1, out_0, out_1
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [-1.80e+23, -inf] [in_0]
// [out_1] ≈ [-inf, -1.80e+23] [in_1]
//
// note the resistors are specified by their resistive values in Ohms
//
module gamma_matrix(in_0, in_1, out_0, out_1);


    inout in_0;
    inout in_1;
    inout out_0;
    inout out_1;


    analog begin
        I(in_0, out_0) <+ -5.5511151231257825e-24 * V(in_0,out_0);
        I(in_1, out_1) <+ -5.5511151231257825e-24 * V(in_1,out_1);
    end

endmodule


// b_matrix
//
// Ports: in_0, out_0, out_1
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [-1.80e+23] [in_0]
// [out_1] ≈ [-inf] [in_1]
//
// note the resistors are specified by their resistive values in Ohms
//
module b_matrix(in_0, out_0, out_1);


    inout in_0;
    inout out_0;
    inout out_1;


    analog begin
        I(in_0, out_0) <+ -5.5511151231257825e-24 * V(in_0,out_0);
    end

endmodule


// a_matrix
//
// Ports: in_0, in_1, out_0, out_1
//
// Parameters:
//
// Functional Description:
//
// Resistor network connecting inputs and outputs according to the following matrix
//
// [out_0] ≈ [-1.13e+28, -inf] [in_0]
// [out_1] ≈ [-1.80e+23, -1.13e+28] [in_1]
//
// note the resistors are specified by their resistive values in Ohms
//
module a_matrix(in_0, in_1, out_0, out_1);


    inout in_0;
    inout in_1;
    inout out_0;
    inout out_1;


    analog begin
        I(in_0, out_0) <+ -8.881784197001253e-29 * V(in_0,out_0);
        I(in_0, out_1) <+ -5.5511151231257825e-24 * V(in_0,out_1);
        I(in_1, out_1) <+ -8.881784197001253e-29 * V(in_1,out_1);
    end

endmodule


// op_amp_int_1
//
// Ports: u_0, y
//
// Parameters:
//
// Functional Description
//
// A linear state space system directly modeled using differential
// equations.
//
// Specifically,
//
// ddt(x(t)) = A x(t) + B u(t)
// y(t) = C^T x(t)
//
// where
//
// x(t) = [x_0, x_1, x_2]^T
// u(t) = [u_0]^T
// y(t) = [y]^T
//
// A ≈
// [-5.00e+02, -8.66e+02, 0.00e+00]
// [8.66e+02, -5.00e+02, 0.00e+00]
// [0.00e+00, 1.15e+00, -1.00e+03]
//
// B ≈
// [-1.00e+09]
// [-0.00e+00]
// [-0.00e+00]
//
//
// CT ≈
// [0.00e+00, 0.00e+00, 1.00e+00]
// D ≈
// [0.00e+00]
//
module op_amp_int_1(u_0, y);

    input u_0;

    output y; // Output


    analog begin
        ddt(V(x_0)) <+ -500.0000000000001*V(x_0) -866.0254037844386*V(x_1) -999999999.9999993*V(u_0);
        ddt(V(x_1)) <+ 866.0254037844386*V(x_0) -500.0000000000001*V(x_1);
        ddt(V(x_2)) <+ 1.1547005383792512*V(x_1) -1000.0*V(x_2);
        V(y_0) <+ 1.0*V(x_2);
    end

endmodule


// inverting_amplifier_int_1
//
// Ports: p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_int_1(p_in, n_in, out);

    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    op_amp_int_1 op_amp_int_1 (
            .u_0(n_in),
            .y(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// op_amp_int_0
//
// Ports: u_0, y
//
// Parameters:
//
// Functional Description
//
// A linear state space system directly modeled using differential
// equations.
//
// Specifically,
//
// ddt(x(t)) = A x(t) + B u(t)
// y(t) = C^T x(t)
//
// where
//
// x(t) = [x_0, x_1, x_2]^T
// u(t) = [u_0]^T
// y(t) = [y]^T
//
// A ≈
// [-5.00e+02, -8.66e+02, 0.00e+00]
// [8.66e+02, -5.00e+02, 0.00e+00]
// [0.00e+00, 1.15e+00, -1.00e+03]
//
// B ≈
// [-1.00e+09]
// [-0.00e+00]
// [-0.00e+00]
//
//
// CT ≈
// [0.00e+00, 0.00e+00, 1.00e+00]
// D ≈
// [0.00e+00]
//
module op_amp_int_0(u_0, y);

    input u_0;

    output y; // Output


    analog begin
        ddt(V(x_0)) <+ -500.0000000000001*V(x_0) -866.0254037844386*V(x_1) -999999999.9999993*V(u_0);
        ddt(V(x_1)) <+ 866.0254037844386*V(x_0) -500.0000000000001*V(x_1);
        ddt(V(x_2)) <+ 1.1547005383792512*V(x_1) -1000.0*V(x_2);
        V(y_0) <+ 1.0*V(x_2);
    end

endmodule


// inverting_amplifier_int_0
//
// Ports: p_in, n_in, out
//
// Parameters: C
//
// Functional Description:
//
// Op-amp integrator configuration where
// a capacitor is connected as negative feedback
// i.e., between the output and negative input
// of the op-amp.
//
// The resulting differential equations are
// C ddt(V(out, n_in)) = I(out, n_in)
//
module inverting_amplifier_int_0(p_in, n_in, out);

    input p_in; // positive input

    output out; // output

    inout n_in; // negative input

    parameter real C = 1e-12;



    op_amp_int_0 op_amp_int_0 (
            .u_0(n_in),
            .y(out)
    );

    analog begin
        ddt(V(out, n_in)) <+ I(out, n_in) / C;
    end

endmodule


// analog_system
//
// Ports: vdd, gnd, sgd, u_0, s_0, s_1, x_0, x_1, s_tilde_0, s_tilde_1
//
// Parameters:
//
// Functional Description
//
// An analog system enforcing the differential equations.
//
// ddt(x(t)) = A x(t) + B u(t) + Gamma s(t)
// s_tilde(t) = Gamma_tildeT x(t)
//
// where
//
// x(t) = [x_0, x_1]^T
// u(t) = [u_0]^T
// s(t) = [s_0, s_1]^T
// s_tilde(t) = [s_tilde_0, s_tilde_1]^T
//
// A ≈
// [-1.00e-01, 0.00e+00, 1.15e+00, -1.00e+03, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]
// [-1.00e+09, -5.00e+02, -8.66e+02, 0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]
// [0.00e+00, 8.66e+02, -5.00e+02, 0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]
// [0.00e+00, 0.00e+00, 1.15e+00, -1.00e+03, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03, -1.00e-01, 0.00e+00, 1.15e+00, -1.00e+03]
// [-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -1.00e+09, -5.00e+02, -8.66e+02, 0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, 0.00e+00, 8.66e+02, -5.00e+02, 0.00e+00]
// [-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, 0.00e+00, 0.00e+00, 1.15e+00, -1.00e+03]
//
// B ≈
// [-6.25e+03]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
// [0.00e+00]
//
// Gamma ≈
// [-6.25e+03, -0.00e+00]
// [0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00]
// [-0.00e+00, -6.25e+03]
// [0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00]
//
// Gamma_tildeT ≈
// [1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
// CT ≈
// [0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]
//
module analog_system(vdd, gnd, sgd, u_0, s_0, s_1, x_0, x_1, s_tilde_0, s_tilde_1);

    input vdd; // positive supply
    input gnd; // ground
    input sgd; // signal ground
    input u_0; // input channel 0
    input s_0; // control signal 0
    input s_1; // control signal 1

    output x_0; // state variable number 0
    output x_1; // state variable number 1
    output s_tilde_0; // control observation 0
    output s_tilde_1; // control observation 1



    inverting_amplifier_int_0 int_0 (
            .p_in(sgd),
            .n_in(vgd_0),
            .out(x_0)
    );

    inverting_amplifier_int_1 int_1 (
            .p_in(sgd),
            .n_in(vgd_1),
            .out(x_1)
    );

    a_matrix a_matrix_0 (
            .in_0(x_0),
            .in_1(x_1),
            .out_0(vgd_0),
            .out_1(vgd_1)
    );

    b_matrix b_matrix_0 (
            .in_0(u_0),
            .out_0(vgd_0),
            .out_1(vgd_1)
    );

    gamma_matrix gamma_matrix_0 (
            .in_0(s_0),
            .in_1(s_1),
            .out_0(vgd_0),
            .out_1(vgd_1)
    );

    gamma_tildeT_matrix gamma_tildeT_matrix_0 (
            .in_0(x_0),
            .in_1(x_1),
            .out_0(s_tilde_0),
            .out_1(s_tilde_1)
    );

endmodule"""
    )
