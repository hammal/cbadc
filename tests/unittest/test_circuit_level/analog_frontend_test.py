# def test_AnalogFrontend(get_simulator):
#     digital_control_module = cbadc.circuit_level.digital_control.DigitalControl(
#         get_simulator.digital_control
#     )
#     analog_system_module = cbadc.circuit_level.state_space_equations.AnalogSystem(
#         get_simulator.analog_system
#     )
#     analog_frontend_module = cbadc.circuit_level.analog_frontend.AnalogFrontend(
#         analog_system_module, digital_control_module
#     )
#     assert (
#         "\n\n\n".join(analog_frontend_module.render()[0])
#         == """// comparator
# //
# // Ports: vdd, vgd, vsgd, clk, s_tilde, s
# //
# // Parameters: dly, ttime
# //
# // Functional Description:
# //
# // A comparator implementation where
# // the output signal s(t) is updated at the
# // falling edge of the V(clk) signal depending
# // on the input signal V(s_tilde) is above or
# // below a given threshold.
# //
# // threshold determines the descision threshold.
# // Furthermore, dly and ttime specifies how quickly the
# // comparator can switch its output.
# //
# module comparator(vdd, vgd, vsgd, clk, s_tilde, s);

#     input vdd; // positive supply
#     input vgd; // ground
#     input vsgd; // signal ground
#     input clk; // clock signal
#     input s_tilde;

#     output s;

#     parameter real dly;
#     parameter real ttime = 10p;


#     analog initial begin
#         V(s) = 0;
#     end

#     analog begin
#         @(cross(V(clk) - V(sgd), -1)) begin
#         	if(V(s_tilde) > V(sgd))
#         		V(s, vgd) <+ V(vdd, vgd) * transition(1, dly, ttime);
#         	else
#         		V(s, vgd) <+ V(vdd, vgd) * transition(0, dly, ttime);
#         end
#     end

# endmodule


# // digital_control
# //
# // Ports: vdd, vgd, vsgd, clk, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4, s_0, s_1, s_2, s_3, s_4
# //
# // Parameters:
# //
# // Functional Description:
# //
# // A digital control which mainly connects
# // M comparators to the input and outputs of
# // the module itself.
# //
# module digital_control(vdd, vgd, vsgd, clk, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4, s_0, s_1, s_2, s_3, s_4);

#     input vdd; // positive supply
#     input vgd; // ground
#     input vsgd; // signal ground
#     input clk; // clock signal
#     input s_tilde_0;
#     input s_tilde_1;
#     input s_tilde_2;
#     input s_tilde_3;
#     input s_tilde_4;

#     output s_0;
#     output s_1;
#     output s_2;
#     output s_3;
#     output s_4;


#     comparator q_0 (
#             .vdd(vdd),
#             .vgd(vgd),
#             .vsgd(vsgd),
#             .clk(clk),
#             .s_tilde(s_tilde_0),
#             .s(s_0)
#     );

#     comparator q_1 (
#             .vdd(vdd),
#             .vgd(vgd),
#             .vsgd(vsgd),
#             .clk(clk),
#             .s_tilde(s_tilde_1),
#             .s(s_1)
#     );

#     comparator q_2 (
#             .vdd(vdd),
#             .vgd(vgd),
#             .vsgd(vsgd),
#             .clk(clk),
#             .s_tilde(s_tilde_2),
#             .s(s_2)
#     );

#     comparator q_3 (
#             .vdd(vdd),
#             .vgd(vgd),
#             .vsgd(vsgd),
#             .clk(clk),
#             .s_tilde(s_tilde_3),
#             .s(s_3)
#     );

#     comparator q_4 (
#             .vdd(vdd),
#             .vgd(vgd),
#             .vsgd(vsgd),
#             .clk(clk),
#             .s_tilde(s_tilde_4),
#             .s(s_4)
#     );

# endmodule


# // analog_system
# //
# // Ports: vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4
# //
# // Parameters:
# //
# // Functional Description
# //
# // The analog system directly modeled using differential
# // equations.
# //
# // Specifically, we use the state space model equations
# //
# // ddt(x(t)) = A x(t) + B u(t) + Gamma s(t)
# // s_tilde(t) = Gamma_tildeT x(t)
# //
# // where
# //
# // x(t) = [x_0, x_1, x_2, x_3, x_4]^T
# // u(t) = [u_0]^T
# // s(t) = [s_0, s_1, s_2, s_3, s_4]^T
# // s_tilde(t) = [s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4]^T
# //
# // A ≈
# // [-6.25e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
# // [6.25e+03, -6.25e+01, 0.00e+00, 0.00e+00, 0.00e+00]
# // [0.00e+00, 6.25e+03, -6.25e+01, 0.00e+00, 0.00e+00]
# // [0.00e+00, 0.00e+00, 6.25e+03, -6.25e+01, 0.00e+00]
# // [0.00e+00, 0.00e+00, 0.00e+00, 6.25e+03, -6.25e+01]
# //
# // B ≈
# // [6.25e+03]
# // [0.00e+00]
# // [0.00e+00]
# // [0.00e+00]
# // [0.00e+00]
# //
# // Gamma ≈
# // [-6.25e+03, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00]
# // [-0.00e+00, -6.25e+03, -0.00e+00, -0.00e+00, -0.00e+00]
# // [-0.00e+00, -0.00e+00, -6.25e+03, -0.00e+00, -0.00e+00]
# // [-0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03, -0.00e+00]
# // [-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -6.25e+03]
# //
# // Gamma_tildeT ≈
# // [1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
# // [0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
# // [0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00]
# // [0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00]
# // [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]
# //
# module analog_system(vdd, vgd, vsgd, u_0, s_0, s_1, s_2, s_3, s_4, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4);

#     input vdd; // positive supply
#     input vgd; // ground
#     input vsgd; // signal ground
#     input u_0;
#     input s_0;
#     input s_1;
#     input s_2;
#     input s_3;
#     input s_4;

#     output s_tilde_0;
#     output s_tilde_1;
#     output s_tilde_2;
#     output s_tilde_3;
#     output s_tilde_4;


#     analog begin
#         ddt(V(x_0), sgd) <+ -62.5*V(x_0, sgd) -6250.0*V(s_0, sgd) 6250.0*V(u_0, sgd);
#         ddt(V(x_1), sgd) <+ 6250.0*V(x_0, sgd) -62.5*V(x_1, sgd) -6250.0*V(s_1, sgd);
#         ddt(V(x_2), sgd) <+ 6250.0*V(x_1, sgd) -62.5*V(x_2, sgd) -6250.0*V(s_2, sgd);
#         ddt(V(x_3), sgd) <+ 6250.0*V(x_2, sgd) -62.5*V(x_3, sgd) -6250.0*V(s_3, sgd);
#         ddt(V(x_4), sgd) <+ 6250.0*V(x_3, sgd) -62.5*V(x_4, sgd) -6250.0*V(s_4, sgd);
#         V(s_tilde_0, sgd) <+ 1.0*V(x_0, sgd);
#         V(s_tilde_1, sgd) <+ 1.0*V(x_1, sgd);
#         V(s_tilde_2, sgd) <+ 1.0*V(x_2, sgd);
#         V(s_tilde_3, sgd) <+ 1.0*V(x_3, sgd);
#         V(s_tilde_4, sgd) <+ 1.0*V(x_4, sgd);
#     end

# endmodule


# // analog_frontend
# //
# // Ports: vdd, vgd, vsgd, clk, u_0, s_0, s_1, s_2, s_3, s_4
# //
# // Parameters:
# //
# // Functional Description:
# //
# // An analog frontend comparise of an analog-system
# // and digital control interfaced such that
# // control signals can be generated given a clock signal
# // and input stimuli.
# //
# module analog_frontend(vdd, vgd, vsgd, clk, u_0, s_0, s_1, s_2, s_3, s_4);

#     input vdd; // positive supply
#     input vgd; // ground
#     input vsgd; // signal ground
#     input clk; // clock signal
#     input u_0; // input channel 0

#     output s_0; // control signal 0
#     output s_1; // control signal 1
#     output s_2; // control signal 2
#     output s_3; // control signal 3
#     output s_4; // control signal 4


#     analog_system  (
#             .vdd(vdd),
#             .vgd(vgd),
#             .vsgd(vsgd),
#             .u_0(u_0),
#             .s_0(s_0),
#             .s_1(s_1),
#             .s_2(s_2),
#             .s_3(s_3),
#             .s_4(s_4),
#             .s_tilde_0(s_tilde_0),
#             .s_tilde_1(s_tilde_1),
#             .s_tilde_2(s_tilde_2),
#             .s_tilde_3(s_tilde_3),
#             .s_tilde_4(s_tilde_4)
#     );

#     digital_control  (
#             .vdd(vdd),
#             .vgd(vgd),
#             .vsgd(vsgd),
#             .clk(clk),
#             .s_tilde_0(s_tilde_0),
#             .s_tilde_1(s_tilde_1),
#             .s_tilde_2(s_tilde_2),
#             .s_tilde_3(s_tilde_3),
#             .s_tilde_4(s_tilde_4),
#             .s_0(s_0),
#             .s_1(s_1),
#             .s_2(s_2),
#             .s_3(s_3),
#             .s_4(s_4)
#     );

# endmodule"""
#     )
#     analog_frontend_module.to_file("chain_of_integrators.v")
