from cbadc.circuit import Terminal, SubCircuitElement
from cbadc.circuit.components.passives import Resistor
from cbadc.circuit.components.sources import (
    DCVoltageSource,
    PulseVoltageSource,
    SinusoidalVoltageSource,
)
from cbadc.circuit.components.cmos import (
    NMOS,
    PMOS,
    Inverter,
    RSLatch,
    StrongARMLatch,
    Comparator,
)
import os
import subprocess


def test_nmos_model_ngspice():
    nmos = NMOS(
        instance_name="M1",
        model_name="nmos1",
        W=1e-6,
        L=2e-7,
        VT0=0.5,
        KP=0.5,
        LAMBDA=0.5,
        RD=0.5,
        RS=0.5,
    )

    terminals = [Terminal("D"), Terminal("G"), Terminal("S"), Terminal("B")]
    subckt = SubCircuitElement("Xsub", "subckt", terminals)
    subckt.add(nmos)
    subckt.connects(
        (subckt["D"], nmos["D"]),
        (subckt["G"], nmos["G"]),
        (subckt["S"], nmos["S"]),
        (subckt["B"], nmos["B"]),
    )
    component_string = nmos.get_ngspice(subckt._internal_connections)
    print(component_string)
    assert component_string == "M1 D G S B nmos1 W=1e-06 L=2e-07 m=1"
    model_string = [model.get_ngspice() for model in subckt._get_model_set()][0]
    print(model_string)
    assert (
        model_string
        == """.model nmos1 NMOS
+ VT0=0.5
+ KP=0.5
+ GAMMA=0.45
+ PHI=0.9
+ LAMBDA=0.5
+ RD=0.5
+ RS=0.5
+ level=1
+ CBD=0.0
+ CBS=0.0
+ IS=0.0
+ PB=0.9
+ CGSO=0.0
+ CGDO=4e-10
+ CGBO=0.0
+ RSH=0.0
+ CJ=0.00056
+ MJ=0.45
+ CJSW=3.5e-12
+ MJSW=0.2
+ JS=1e-08
+ TOX=9e-09
+ NSUB=900000000000000.0
"""
    )


def test_pmos_model_ngspice():
    pmos = PMOS(
        instance_name="M1",
        model_name="pmos1",
        W=1e-6,
        L=2e-7,
        VT0=0.5,
        KP=0.5,
        LAMBDA=0.5,
        RD=0.5,
        RS=0.5,
    )

    terminals = [Terminal("D"), Terminal("G"), Terminal("S"), Terminal("B")]
    subckt = SubCircuitElement("Xsub", "subckt", terminals)
    subckt.add(pmos)
    subckt.connects(
        (subckt["D"], pmos["D"]),
        (subckt["G"], pmos["G"]),
        (subckt["S"], pmos["S"]),
        (subckt["B"], pmos["B"]),
    )
    component_string = pmos.get_ngspice(subckt._internal_connections)
    print(component_string)
    assert component_string == "M1 D G S B pmos1 W=1e-06 L=2e-07 m=1"
    model_string = [model.get_ngspice() for model in subckt._get_model_set()][0]
    print(model_string)
    assert (
        model_string
        == """.model pmos1 PMOS
+ VT0=0.5
+ KP=0.5
+ GAMMA=0.4
+ PHI=0.8
+ LAMBDA=0.5
+ RD=0.5
+ RS=0.5
+ level=1
+ CBD=0.0
+ CBS=0.0
+ IS=0.0
+ PB=0.9
+ CGSO=0.0
+ CGDO=3e-10
+ CGBO=0.0
+ RSH=0.0
+ CJ=0.00094
+ MJ=0.5
+ CJSW=3.2e-12
+ MJSW=0.3
+ JS=5e-09
+ TOX=9e-09
+ NSUB=500000000000000.0
"""
    )


def test_inverter_ngspice():
    transistor_unit_length = 0.5e-6

    nmos = NMOS(
        instance_name="M1",
        model_name="nmos1",
        W=2 * transistor_unit_length,
        L=transistor_unit_length,
    )
    pmos = PMOS(
        instance_name="M2",
        model_name="pmos1",
        W=2 * transistor_unit_length,
        L=transistor_unit_length,
    )
    terminals = [Terminal("VDD"), Terminal("GND"), Terminal("IN"), Terminal("OUT")]
    vdd = DCVoltageSource("Vdd", 1.8)

    inverter = Inverter("Xinv", "inv", nmos, pmos, m=2)
    testbench = SubCircuitElement("Xtest", "testbench", terminals)
    testbench.add(vdd)
    testbench.add(inverter)
    testbench.connects(
        (testbench["VDD"], vdd["P"]),
        (testbench["GND"], vdd["N"]),
        (testbench["IN"], inverter["IN"]),
        (testbench["OUT"], inverter["OUT"]),
        (inverter["VDD"], vdd["P"]),
        (inverter["GND"], vdd["N"]),
    )
    T = 1e-6
    size = 5

    amplitude = 0.9 / 2.0
    input_signal = SinusoidalVoltageSource(
        offset=0.9,
        amplitude=amplitude,
        frequency=1 / (3 * T),
        delay_time=0.0,
        phase=0.0,
        damping_factor=0.0,
        instance_name=f"Vin_0",
        ac_gain=0.9 / 2.0,
    )
    testbench.add(input_signal)
    testbench.connects(
        (testbench["GND"], input_signal[1]),
        (testbench["IN"], input_signal[0]),
    )

    component_string = inverter.get_ngspice(testbench._internal_connections)
    print(component_string)
    for subckt_definition in testbench.get_sub_circuit_definitions():
        print(subckt_definition)
        print("\n\n")

    testbench_name = "inverter_testbench.cir"
    testbench_spice = f"""{testbench_name}
.Global GND 0

{testbench.get_ngspice(testbench._internal_connections)}

* Subcircuits
{(2 * os.linesep).join(testbench.get_sub_circuit_definitions())}

* Models
{(1 * os.linesep).join([m.get_ngspice() for m in testbench._get_model_set()])}

.op
.tran {T} {size * T} UIC

.options reltol=0.0001
.options opts
.options warn=1
.options ACCT

.control
listing
run
plot V(IN) V(OUT)
.endc

.END
"""
    print(testbench_spice)
    with open(testbench_name, "w") as f:
        f.write(testbench_spice)
    subprocess.run(["ngspice", testbench_name])
    # assert False


def test_sr_latch():
    smalest_length = 0.5e-6
    mos = NMOS(
        instance_name="M1",
        model_name="nmos1",
        W=2 * smalest_length,
        L=smalest_length,
    )
    pmos = PMOS(
        instance_name="M2",
        model_name="pmos1",
        W=2 * smalest_length,
        L=smalest_length,
    )
    terminals = [
        Terminal("VDD"),
        Terminal("GND"),
        Terminal("SET"),
        Terminal("RESET"),
        Terminal("Q"),
        Terminal("Q_BAR"),
        Terminal("VCM"),
    ]
    rs_latch = RSLatch("Xrs", "rslatch", mos, pmos)
    testbench = SubCircuitElement("Xtest", "testbench", terminals)

    # Voltage supply
    vdd = DCVoltageSource("Vdd", 0.9)
    testbench.add(vdd)
    testbench.connects(
        (testbench["VDD"], vdd["P"]),
        (testbench["VCM"], vdd["N"]),
    )

    vss = DCVoltageSource("Vss", 0.9)
    testbench.add(vss)
    testbench.connects(
        (testbench["VCM"], vss["P"]),
        (testbench["GND"], vss["N"]),
    )

    # Latch
    testbench.add(rs_latch)
    testbench.connects(
        (testbench["VDD"], rs_latch["VDD"]),
        (testbench["GND"], rs_latch["GND"]),
        (testbench["SET"], rs_latch["SET"]),
        (testbench["RESET"], rs_latch["RESET"]),
        (testbench["Q"], rs_latch["Q"]),
        (testbench["Q_BAR"], rs_latch["Q_BAR"]),
    )

    T = 1e-6
    size = 5
    T_rise = T / 10.0
    T_fall = T / 10.0
    # Input signal clk
    amplitude = 0.9 / 2.0
    amplitude = 1e-1
    set_signal = SinusoidalVoltageSource(
        offset=0,
        amplitude=amplitude,
        frequency=1 / (3 * T),
        delay_time=0.0,
        phase=0.0,
        damping_factor=0.0,
        instance_name=f"Vset",
        ac_gain=0.9 / 2.0,
    )
    testbench.add(set_signal)
    testbench.connects(
        (testbench["SET"], set_signal[0]),
        (testbench["VCM"], set_signal[1]),
    )
    # set_signal = PulseVoltageSource("Vset", 0.0, 1.8, T / 3, T_rise, T_fall)
    # testbench.add(set_signal)
    # testbench.connects(
    #     (testbench["SET"], set_signal[0]),
    #     (testbench["GND"], set_signal[1]),
    # )

    reset_signal = SinusoidalVoltageSource(
        offset=0,
        amplitude=amplitude,
        frequency=1 / (3 * T),
        delay_time=0.0,
        phase=0.0,
        damping_factor=0.0,
        instance_name=f"Vreset",
        ac_gain=0.9 / 2.0,
    )
    testbench.add(reset_signal)
    testbench.connects(
        (testbench["RESET"], reset_signal[1]),
        (testbench["VCM"], reset_signal[0]),
    )
    # reset_signal = PulseVoltageSource("Vreset", 0.0, 1.8, 2 * T, T_rise, T_fall)
    # testbench.add(reset_signal)
    # testbench.connects(
    #     (testbench["RESET"], reset_signal[0]),
    #     (testbench["GND"], reset_signal[1]),
    # )

    # Print checks
    component_string = rs_latch.get_ngspice(testbench._internal_connections)
    print(component_string)
    assert component_string == "Xrs SET RESET Q Q_BAR VDD GND rslatch m=1"
    for subckt_definition in testbench.get_sub_circuit_definitions():
        print(subckt_definition)
        print("\n\n")
    testbench_name = "sr_latch_testbench.cir"
    testbench_spice = f"""{testbench_name}
.Global GND 0

{testbench.get_ngspice(testbench._internal_connections)}

* Subcircuits
{(2 * os.linesep).join(testbench.get_sub_circuit_definitions())}

* Models
{(1 * os.linesep).join([m.get_ngspice() for m in testbench._get_model_set()])}

.op
.tran {T} {size * T} UIC

.options reltol=0.0001
.options opts
.options warn=1
.options ACCT

.control
listing
run
plot V(SET) V(RESET)
plot V(Q) V(Q_BAR)
.endc

.END
"""
    print(testbench_spice)

    with open(testbench_name, "w") as f:
        f.write(testbench_spice)
    subprocess.run(["ngspice", testbench_name])
    # assert False


def test_strong_armed_latch():
    smalest_length = 0.5e-6
    mos = NMOS(
        instance_name="M1",
        model_name="nmos1",
        W=2 * smalest_length,
        L=smalest_length,
    )
    pmos = PMOS(
        instance_name="M2",
        model_name="pmos1",
        W=2 * smalest_length,
        L=smalest_length,
    )
    terminals = [
        Terminal("VDD"),
        Terminal("VCM"),
        Terminal("GND"),
        Terminal("CLK"),
        Terminal("VIN_0_P"),
        Terminal("VIN_0_N"),
        Terminal("VOUT_P"),
        Terminal("VOUT_N"),
    ]

    testbench = SubCircuitElement("Xtest", "testbench", terminals)

    vdd = DCVoltageSource("Vdd", 0.9)
    testbench.add(vdd)
    testbench.connects(
        (testbench["VDD"], vdd["P"]),
        (testbench["VCM"], vdd["N"]),
    )

    vss = DCVoltageSource("Vss", 0.9)
    testbench.add(vss)
    testbench.connects(
        (testbench["VCM"], vss["P"]),
        (testbench["GND"], vss["N"]),
    )

    T = 1e-6
    size = 6
    T_rise = T / 10.0
    T_fall = T / 10.0
    clock = PulseVoltageSource("Vclk", 0.0, 1.8, T, T_rise, T_fall)
    testbench.add(clock)
    testbench.connects(
        (testbench["CLK"], clock[0]),
        (testbench["GND"], clock[1]),
    )

    sal = StrongARMLatch("XSal", "strong_armed_latch", mos, pmos, [2.0])
    testbench.add(sal)
    testbench.connects(
        (testbench["VDD"], sal["VDD"]),
        (testbench["GND"], sal["GND"]),
        (testbench["CLK"], sal["CLK"]),
        (testbench["VIN_0_P"], sal["VIN_0_P"]),
        (testbench["VIN_0_N"], sal["VIN_0_N"]),
        (testbench["VOUT_P"], sal["VOUT_P"]),
        (testbench["VOUT_N"], sal["VOUT_N"]),
    )

    # Input signal
    amplitude = 0.9 / 2.0
    amplitude = 1e-1
    signal_0 = SinusoidalVoltageSource(
        offset=0,
        amplitude=amplitude,
        frequency=1 / (3 * T),
        delay_time=0.0,
        phase=0.0,
        damping_factor=0.0,
        instance_name=f"Vset",
        ac_gain=0.9 / 2.0,
    )
    testbench.add(signal_0)
    testbench.connects(
        (testbench["VIN_0_P"], signal_0[0]),
        (testbench["VCM"], signal_0[1]),
    )

    signal_1 = SinusoidalVoltageSource(
        offset=0,
        amplitude=amplitude,
        frequency=1 / (3 * T),
        delay_time=0.0,
        phase=0.0,
        damping_factor=0.0,
        instance_name=f"Vreset",
        ac_gain=0.9 / 2.0,
    )
    testbench.add(signal_1)
    testbench.connects(
        (testbench["VIN_0_N"], signal_1[1]),
        (testbench["VCM"], signal_1[0]),
    )

    component_string = sal.get_ngspice(testbench._internal_connections)
    print(component_string)
    assert (
        component_string
        == "Xsal VDD GND CLK VIN_0_P VIN_0_N VOUT_P VOUT_N strong_armed_latch m=1"
    )
    for subckt_definition in sal.get_sub_circuit_definitions():
        print(subckt_definition)
        print("\n\n")
    # assert False
    testbench_name = "sal_testbench.cir"
    testbench_spice = f"""{testbench_name}
.Global GND 0

{testbench.get_ngspice(testbench._internal_connections)}

* Subcircuits
{(2 * os.linesep).join(testbench.get_sub_circuit_definitions())}

* Models
{(1 * os.linesep).join([m.get_ngspice() for m in testbench._get_model_set()])}

.op
.tran {T} {size * T} UIC

.options reltol=0.0001
.options opts
.options warn=1
.options ACCT

.control
listing
run
plot V(CLK) V(VIN_0_P) V(VIN_0_N) V(VOUT_P) V(VOUT_N) V(xtest.xsal.P) V(xtest.xsal.Q)
plot V(CLK)
plot V(VOUT_P) V(VOUT_N)
plot V(xtest.xsal.P) V(xtest.xsal.Q)
.endc

.END
"""
    print(testbench_spice)
    with open(testbench_name, "w") as f:
        f.write(testbench_spice)
    subprocess.run(["ngspice", testbench_name])
    # assert False


def test_comparator_testbench():
    transistor_unit_length = 0.5e-6
    terminals = [
        Terminal("VDD"),
        Terminal("VCM"),
        Terminal("GND"),
        Terminal("CLK"),
        Terminal("VIN_0_P"),
        Terminal("VIN_0_N"),
        Terminal("VOUT_P"),
        Terminal("VOUT_N"),
    ]
    testbench = SubCircuitElement("Xtest", "subckt", terminals)

    vdd = DCVoltageSource("Vdd", 0.9)
    testbench.add(vdd)
    testbench.connects(
        (testbench["VDD"], vdd["P"]),
        (testbench["VCM"], vdd["N"]),
    )

    vss = DCVoltageSource("Vss", 0.9)
    testbench.add(vss)
    testbench.connects(
        (testbench["VCM"], vss["P"]),
        (testbench["GND"], vss["N"]),
    )

    # W=2e-6,
    # L=5e-6,
    mos = NMOS(
        instance_name="M1",
        model_name="nmos1",
        W=2 * transistor_unit_length,
        L=1 * transistor_unit_length,
    )
    pmos = PMOS(
        instance_name="M2",
        model_name="pmos1",
        W=2 * transistor_unit_length,
        L=1 * transistor_unit_length,
    )
    comp = Comparator("XCOMP", "comparator", mos, pmos, [2.0])
    testbench.add(comp)
    testbench.connects(
        (testbench["VDD"], comp["VDD"]),
        (testbench["GND"], comp["GND"]),
        (testbench["CLK"], comp["CLK"]),
        (testbench["VIN_0_P"], comp["VIN_0_P"]),
        (testbench["VIN_0_N"], comp["VIN_0_N"]),
        (testbench["VOUT_P"], comp["VOUT_P"]),
        (testbench["VOUT_N"], comp["VOUT_N"]),
    )

    T = 1e-6
    size = 6
    T_rise = T / 100.0
    T_fall = T / 100.0
    clock = PulseVoltageSource("Vclk", 0.0, 1.8, T, T_rise, T_fall)
    testbench.add(clock)
    testbench.connects(
        (testbench["CLK"], clock[0]),
        (testbench["GND"], clock[1]),
    )

    amplitude = 0.9 / 2.0
    amplitude = 1e-1
    input_signal = SinusoidalVoltageSource(
        offset=0,
        amplitude=amplitude,
        frequency=1 / (3 * T),
        delay_time=0.0,
        phase=0.0,
        damping_factor=0.0,
        instance_name=f"Vin_0",
        ac_gain=0.9 / 2.0,
    )
    testbench.add(input_signal)
    testbench.connects(
        (testbench["VIN_0_P"], input_signal[0]),
        (testbench["VCM"], input_signal[1]),
    )
    input_signal = SinusoidalVoltageSource(
        offset=0,
        amplitude=amplitude,
        frequency=1 / (3 * T),
        delay_time=0.0,
        phase=0.0,
        damping_factor=0.0,
        instance_name=f"Vin_1",
        ac_gain=0.9 / 2.0,
    )
    testbench.add(input_signal)
    testbench.connects(
        (testbench["VIN_0_N"], input_signal[1]),
        (testbench["VCM"], input_signal[0]),
    )

    component_string = comp.get_ngspice(comp._internal_connections)
    print(component_string)
    assert (
        component_string
        == "Xcomp VDD GND CLK VIN_0_P VIN_0_N VOUT_P VOUT_N comparator m=1"
    )
    for subckt_definition in comp.get_sub_circuit_definitions():
        print(subckt_definition)
        print("\n\n")
    # assert False

    testbench_name = "comparator_testbench.cir"
    testbench_spice = f"""{testbench_name}
.Global GND 0

{testbench.get_ngspice(testbench._internal_connections)}

* Subcircuits
{(2 * os.linesep).join(testbench.get_sub_circuit_definitions())}

* Models
{(1 * os.linesep).join([m.get_ngspice() for m in testbench._get_model_set()])}

.op
.tran {T} {size * T} {T/8*3} UIC

.options reltol=0.0001
.options opts
.options warn=1
.options ACCT

.control
listing
run
plot V(CLK)
plot V(CLK) V(VIN_0_P) V(VIN_0_N) V(VOUT_P) V(VOUT_N)
plot V(xtest.xcomp.X) V(xtest.xcomp.Y)
plot V(xtest.xcomp.xsal.P) V(xtest.xcomp.xsal.Q)
.endc

.END
"""
    print(testbench_spice)
    with open(testbench_name, "w") as f:
        f.write(testbench_spice)
    subprocess.run(["ngspice", testbench_name])
    # assert False
