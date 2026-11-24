"""Validate supported frontends against ngspice.

``GmC.netlist`` supplies the circuit: capacitor voltages are states,
transconductors and resistors implement the state equation, and XSPICE devices clock
binary comparator decisions into constant DAC pulses. Ngspice integrates the
circuit continuously; cbadc advances its discrete step map. ``spice`` reads
the transient, and ``bench`` compares decisions and states at clock steps.

This suite uses only frontend features that ``GmC.netlist`` represents.
"""
