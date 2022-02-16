from tests.fixture.chain_of_integrators import get_simulator, chain_of_integrators
import cbadc.circuit_level.digital_control


def test_ComparatorModule():
    comparator = cbadc.circuit_level.digital_control.Comparator("my_comparator")
    assert (
        comparator.render()[0][-1]
        == """// comparator
// 
// Ports: vdd, vgd, vsgd, clk, s_tilde, s
// 
// Parameters: dly, ttime
// 
// Functional Description:
// 
// A comparator implementation where
// the output signal s(t) is updated at the
// falling edge of the V(clk) signal depending
// on the input signal V(s_tilde) is above or
// below a given threshold.
// 
// threshold determines the descision threshold.
// Furthermore, dly and ttime specifies how quickly the
// comparator can switch its output.
//
module comparator(vdd, vgd, vsgd, clk, s_tilde, s);

    input vdd; // positive supply
    input vgd; // ground
    input vsgd; // signal ground
    input clk; // clock signal
    input s_tilde;

    output s;

    parameter real dly;
    parameter real ttime = 10p;


    analog initial begin
        V(s) = 0;
    end

    analog begin
        @(cross(V(clk) - V(sgd), -1)) begin
        	if(V(s_tilde) > V(sgd))
        		V(s, vgd) <+ V(vdd, vgd) * transition(1, dly, ttime);
        	else
        		V(s, vgd) <+ V(vdd, vgd) * transition(0, dly, ttime);
        end
    end

endmodule"""
    )


def test_DigitalControlModule(get_simulator):
    digital_control_module = cbadc.circuit_level.digital_control.DigitalControl(
        get_simulator.digital_control
    )
    assert (
        digital_control_module.render()[0][-1]
        == """// digital_control
// 
// Ports: vdd, vgd, vsgd, clk, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4, s_0, s_1, s_2, s_3, s_4
// 
// Parameters: 
// 
// Functional Description:
// 
// A digital control which mainly connects
// M comparators to the input and outputs of
// the module itself.
//
module digital_control(vdd, vgd, vsgd, clk, s_tilde_0, s_tilde_1, s_tilde_2, s_tilde_3, s_tilde_4, s_0, s_1, s_2, s_3, s_4);

    input vdd; // positive supply
    input vgd; // ground
    input vsgd; // signal ground
    input clk; // clock signal
    input s_tilde_0;
    input s_tilde_1;
    input s_tilde_2;
    input s_tilde_3;
    input s_tilde_4;

    output s_0;
    output s_1;
    output s_2;
    output s_3;
    output s_4;



    comparator q_0 (
            .vdd(vdd),
            .vgd(vgd),
            .vsgd(vsgd),
            .clk(clk),
            .s_tilde(s_tilde_0),
            .s(s_0)
    );

    comparator q_1 (
            .vdd(vdd),
            .vgd(vgd),
            .vsgd(vsgd),
            .clk(clk),
            .s_tilde(s_tilde_1),
            .s(s_1)
    );

    comparator q_2 (
            .vdd(vdd),
            .vgd(vgd),
            .vsgd(vsgd),
            .clk(clk),
            .s_tilde(s_tilde_2),
            .s(s_2)
    );

    comparator q_3 (
            .vdd(vdd),
            .vgd(vgd),
            .vsgd(vsgd),
            .clk(clk),
            .s_tilde(s_tilde_3),
            .s(s_3)
    );

    comparator q_4 (
            .vdd(vdd),
            .vgd(vgd),
            .vsgd(vsgd),
            .clk(clk),
            .s_tilde(s_tilde_4),
            .s(s_4)
    );

endmodule"""
    )
