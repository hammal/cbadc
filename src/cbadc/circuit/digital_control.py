from . import Terminal, SubCircuitElement, SPICE_VALUE
from .components.comparator import (
    DifferentialOutputClockedComparator,
)
from ..digital_control import DigitalControl as NominalDigitalControl
from ..digital_control.dither_control import DitherControl as NominalDitherControl
from ..digital_control.modulator import ModulatorControl as NominalModulatorControl
from ..analog_system import AnalogSystem
from .components.summer import DifferentialSummer
from .components.reference_source import ReferenceSource
from .components.comparator import DAC_Bridge, D_FLIP_FLOP, ADCBridgeAbsolute
from .components.analog_delay import AnalogDelay


class DigitalControl(SubCircuitElement):
    """The default digital control circuit.

    Parameters
    ----------
    instance_name : str
        The instance name of the digital control circuit.
    analog_system : AnalogSystem
        The analog system to which this digital control circuit belongs.
    digital_control : DigitalControl
        The digital control circuit to which this digital control circuit belongs.
    in_high : float
        The high input voltage of the comparator.
    in_low : float
        The low input voltage of the comparator.
    out_high : float
        The high output voltage of the comparator.
    out_low : float
        The low output voltage of the comparator.
    dither_offset : int
        The offset of the dither control circuit.

    """

    def __init__(
        self,
        instance_name: str,
        analog_system: AnalogSystem,
        digital_control: NominalDigitalControl,
        in_high: float,
        in_low: float,
        out_high: float,
        out_low: float,
        dither_offset: int = 0,
    ):
        super().__init__(
            instance_name,
            "digital_control",
            [Terminal("VSS"), Terminal("VDD"), Terminal("CLK"), Terminal("VCM")]
            + [Terminal(f"X{i}_P") for i in range(analog_system.N)]
            + [Terminal(f"X{i}_N") for i in range(analog_system.N)]
            + [Terminal(f"IN{i}_P") for i in range(analog_system.L)]
            + [Terminal(f"IN{i}_N") for i in range(analog_system.L)]
            + [Terminal(f"S{i}_P") for i in range(analog_system.M)]
            + [Terminal(f"S{i}_N") for i in range(analog_system.M)],
        )

        multi_input_comparator_names = [
            f"Xmic_{m}" for m in range(dither_offset, analog_system.M)
        ]

        self.add(
            *[
                MultiInputComparator(
                    multi_input_comparator_names[m - dither_offset],
                    analog_system,
                    digital_control,
                    m,
                    m - dither_offset,
                    in_high,
                    in_low,
                    out_high,
                    out_low,
                )
                for m in range(dither_offset, analog_system.M)
            ]
        )
        for m_tilde in range(analog_system.M_tilde):
            comparator: MultiInputComparator = self.__getattr__(
                multi_input_comparator_names[m_tilde]
            )
            self.connects(
                (self["CLK"], comparator["CLK"]),
                (self["VSS"], comparator["VSS"]),
                (self["VDD"], comparator["VDD"]),
                (self["VCM"], comparator["VCM"]),
            )
            # Connect States
            for n in range(analog_system.N):
                self.connects(
                    (self[f"X{n}_P"], comparator[f"X{n}_P"]),
                    (self[f"X{n}_N"], comparator[f"X{n}_N"]),
                )

            # Connect inputs
            for l in range(analog_system.L):
                self.connects(
                    (self[f"IN{l}_P"], comparator[f"IN{l}_P"]),
                    (self[f"IN{l}_N"], comparator[f"IN{l}_N"]),
                )

            # Connect outputs (control signals)
            for m in range(analog_system.M):
                self.connects(
                    (self[f"S{m}_P"], comparator[f"S{m}_P"]),
                    (self[f"S{m}_N"], comparator[f"S{m}_N"]),
                )


class MultiInputComparator(SubCircuitElement):
    """The multi-input comparator.

    Parameters
    ----------
    instance_name : str
        The instance name of the multi-input comparator.
    analog_system : AnalogSystem
        The analog system to which this multi-input comparator belongs.
    digital_control : DigitalControl
        The digital control circuit to which this multi-input comparator belongs.
    m : int
        The index of the multi-input comparator.
    m_tilde : int
        The index of the multi-input comparator in the dither control circuit.
    in_high : float
        The high input voltage of the comparator.
    in_low : float
        The low input voltage of the comparator.
    out_high : float
        The high output voltage of the comparator.
    out_low : float
        The low output voltage of the comparator.

    """

    in_low: float
    in_high: float
    out_low: float
    out_high: float
    out_undef: float

    def __init__(
        self,
        instance_name: str,
        analog_system: AnalogSystem,
        digital_control: NominalDigitalControl,
        m: int,
        m_tilde: int,
        in_high: float = 0.0,
        in_low: float = 0.0,
        out_high: float = 1.0,
        out_low: float = 0.0,
    ):
        STILDE = Terminal("STILDE")

        super().__init__(
            instance_name,
            f"multi_input_comparator_{m}",
            [Terminal("VSS"), Terminal("VDD"), Terminal("CLK"), Terminal("VCM")]
            + [Terminal(f"X{i}_P") for i in range(analog_system.N)]
            + [Terminal(f"X{i}_N") for i in range(analog_system.N)]
            + [Terminal(f"IN{i}_P") for i in range(analog_system.L)]
            + [Terminal(f"IN{i}_N") for i in range(analog_system.L)]
            + [Terminal(f"S{i}_P") for i in range(analog_system.M)]
            + [Terminal(f"S{i}_N") for i in range(analog_system.M)],
        )

        self._generate_differential_summer(
            analog_system, m_tilde, in_low, in_high, out_high, out_low, STILDE
        )
        self._generate_comparators(
            digital_control,
            m,
            in_high,
            in_low,
            out_high,
            out_low,
            STILDE,
        )

    def _generate_comparators(
        self,
        digital_control: NominalDigitalControl,
        m: int,
        in_high: float,
        in_low: float,
        out_high: float,
        out_low: float,
        STILDE: Terminal,
    ):
        self.Xcomp = DifferentialOutputClockedComparator(
            "Xcomp",
            "clocked_comparator",
            in_low,
            in_high,
            out_low,
            out_high,
            (out_high - out_low) / 2.0,
            clk_offset=0.0,
            t_fall=digital_control.clock.T / 100,
            t_rise=digital_control.clock.T / 100,
            fall_delay=digital_control.clock.T / 1000.0,
            rise_delay=digital_control.clock.T / 1000.0,
        )
        self.connects(
            (self["CLK"], self.Xcomp["CLK"]),
            (STILDE, self.Xcomp["IN"]),
            (self[f"S{m}_P"], self.Xcomp["OUT_P"]),
            (self[f"S{m}_N"], self.Xcomp["OUT_N"]),
            (self["VCM"], self.Xcomp["VCM"]),
        )

    def _generate_differential_summer(
        self,
        analog_system: AnalogSystem,
        m_tilde: int,
        in_low: float,
        in_high: float,
        out_high: float,
        out_low: float,
        STILDE: Terminal,
    ):
        # in_mid = (in_high - in_low) / 2.0 + in_low
        out_mid = (out_high - out_low) / 2.0 + out_low

        self.As = DifferentialSummer(
            "As",
            f"sum_{m_tilde}",
            number_of_inputs=analog_system.N + analog_system.L + analog_system.M,
            input_offset=[
                0.0 for _ in range(analog_system.N + analog_system.L + analog_system.M)
            ],
            input_gain=[gamma for gamma in analog_system.Gamma_tildeT[m_tilde, :]]
            + [b for b in analog_system.B_tilde[m_tilde, :]]
            + [a for a in analog_system.A_tilde[m_tilde, :]],
            output_offset=out_mid,
        )

        # Connect s_tilde
        self.connect(STILDE, self.As[-1])

        # Connect states X
        for n in range(analog_system.N):
            self.connects(
                (self[f"X{n}_P"], self.As[n]),
                (
                    self[f"X{n}_N"],
                    self.As[analog_system.N + analog_system.L + analog_system.M + n],
                ),
            )

        # Connect input U
        for l in range(analog_system.L):
            self.connects(
                (self[f"IN{l}_P"], self.As[analog_system.N + l]),
                (
                    self[f"IN{l}_N"],
                    self.As[
                        2 * analog_system.N + analog_system.L + analog_system.M + l
                    ],
                ),
            )

        # Connect S
        for m in range(analog_system.M):
            self.connects(
                (self[f"S{m}_P"], self.As[analog_system.N + analog_system.L + m]),
                (
                    self[f"S{m}_N"],
                    self.As[
                        2 * analog_system.N + 2 * analog_system.L + analog_system.M + m
                    ],
                ),
            )


class DitherControl(SubCircuitElement):
    """The DitherControl class is a subcircuit element that implements a dither control.

    Parameters
    ----------
    instance_name : str
        The name of the instance.
    analog_system : AnalogSystem
        The analog system.
    digital_control : NominalDigitalControl
        The digital control.
    in_high : float
        The high input voltage.
    in_low : float
        The low input voltage.
    out_high : float
        The high output voltage.
    out_low : float
        The low output voltage.
    clk_delay : float, optional
        The clock delay, by default 0.0
    t_rise : float, optional
        The rise time, by default 1e-15
    t_fall : float, optional
        The fall time, by default 1e-15
    rise_delay : float, optional
        The rise delay, by default 1e-15
    fall_delay : float, optional
        The fall delay, by default 1e-15
    set_delay : float, optional
        The set delay, by default 1e-15
    reset_delay : float, optional
        The reset delay, by default 1e-15
    input_load : float, optional
        The input load, by default 1e-15
    """

    out_low: float
    out_high: float
    out_undef: float

    def __init__(
        self,
        instance_name: str,
        analog_system: AnalogSystem,
        digital_control: NominalDitherControl,
        in_high: float,
        in_low: float,
        out_high: float,
        out_low: float,
        clk_delay: float = 0.0,
        t_rise: float = 1e-15,
        t_fall: float = 1e-15,
        rise_delay: float = 1e-15,
        fall_delay: float = 1e-15,
        set_delay: float = 1e-15,
        reset_delay: float = 1e-15,
        input_load: float = 1e-15,
    ):
        super().__init__(
            instance_name,
            "dither_control",
            [Terminal("VSS"), Terminal("VDD"), Terminal("CLK"), Terminal("VCM")]
            + [Terminal(f"X{i}_P") for i in range(analog_system.N)]
            + [Terminal(f"X{i}_N") for i in range(analog_system.N)]
            + [Terminal(f"IN{i}_P") for i in range(analog_system.L)]
            + [Terminal(f"IN{i}_N") for i in range(analog_system.L)]
            + [Terminal(f"S{i}_P") for i in range(analog_system.M)]
            + [Terminal(f"S{i}_N") for i in range(analog_system.M)],
        )

        self._generate_digital_control(
            analog_system,
            digital_control._deterministic_control,
            digital_control.number_of_random_control,
            in_high,
            in_low,
            out_high,
            out_low,
        )
        self._generate_dither_controls(
            analog_system,
            digital_control,
            out_low,
            out_high,
            (out_high + out_low) / 2.0,
            in_low,
            in_high,
            input_load,
            t_rise,
            t_fall,
            clk_delay,
            set_delay,
            reset_delay,
            rise_delay,
            fall_delay,
        )

    def _generate_dither_controls(
        self,
        analog_system: AnalogSystem,
        digital_control: NominalDitherControl,
        out_low: float,
        out_high: float,
        out_undef: float,
        in_low: float,
        in_high: float,
        input_load: float,
        t_rise: float,
        t_fall: float,
        clk_delay: float,
        set_delay: float,
        reset_delay: float,
        rise_delay: float,
        fall_delay: float,
    ):
        source = ReferenceSource(
            "Ars",
            "r_source",
            digital_control.number_of_random_control,
            f"dither_sequence_{hash(self)}.txt",
            digital_control._pseudo_random_sequence,
            time_step=digital_control.clock.T,
        )

        random_signal_terminals = [
            Terminal(f"RS_{m}") for m in range(digital_control.number_of_random_control)
        ]

        flip_flop = [
            D_FLIP_FLOP(
                f"Adffp_{m}",
                "dflip",
                clk_delay,
                set_delay=set_delay,
                reset_delay=reset_delay,
                ic=1,
                rise_delay=rise_delay,
                fall_delay=fall_delay,
            )
            for m in range(digital_control.number_of_random_control)
        ]

        clk_adc = ADCBridgeAbsolute(
            "Aclk_adc",
            "adc",
            in_low,
            in_high,
            rise_delay=rise_delay,
            fall_delay=fall_delay,
        )

        dac_p = [
            DAC_Bridge(
                f"Adac_p_{m}",
                "dac",
                out_low,
                out_high,
                out_undef,
                input_load,
                t_rise,
                t_fall,
            )
            for m in range(digital_control.number_of_random_control)
        ]
        dac_n = [
            DAC_Bridge(
                f"Adac_n_{m}",
                "dac",
                out_low,
                out_high,
                out_undef,
                input_load,
                t_rise,
                t_fall,
            )
            for m in range(digital_control.number_of_random_control)
        ]

        self.add(source, clk_adc, *flip_flop, *dac_p, *dac_n)

        self.connect(self["CLK"], clk_adc["IN"])
        for m in range(digital_control.number_of_random_control):
            self.connects(
                (random_signal_terminals[m], source[m]),
                (random_signal_terminals[m], flip_flop[m]["IN"]),
                (clk_adc["OUT"], flip_flop[m]["CLK"]),
                (flip_flop[m]["F_OUT_P"], dac_p[m]["IN"]),
                (flip_flop[m]["F_OUT_N"], dac_n[m]["IN"]),
                (self[f"S{m}_P"], dac_p[m]["OUT"]),
                (self[f"S{m}_N"], dac_n[m]["OUT"]),
            )

    def _generate_digital_control(
        self,
        analog_system: AnalogSystem,
        digital_control: NominalDigitalControl,
        dither_control_offset: int,
        in_high: float,
        in_low: float,
        out_high: float,
        out_low: float,
    ):
        if isinstance(digital_control, NominalDigitalControl):
            if digital_control._mulit_phase:
                self.Xdc = MultiPhaseDigitalControl(
                    "Xdc",
                    analog_system,
                    digital_control,
                    in_high,
                    in_low,
                    out_high,
                    out_low,
                    dither_offset=dither_control_offset,
                )
            else:
                self.Xdc = DigitalControl(
                    "Xdc",
                    analog_system,
                    digital_control,
                    in_high,
                    in_low,
                    out_high,
                    out_low,
                    dither_offset=dither_control_offset,
                )
        else:
            raise ValueError(f"Unknown digital control type {type(digital_control)}")

        self.connects(
            (self["VSS"], self.Xdc["VSS"]),
            (self["VDD"], self.Xdc["VDD"]),
            (self["CLK"], self.Xdc["CLK"]),
            (self["VCM"], self.Xdc["VCM"]),
        )

        # Connect States
        for n in range(analog_system.N):
            self.connects(
                (self[f"X{n}_P"], self.Xdc[f"X{n}_P"]),
                (self[f"X{n}_N"], self.Xdc[f"X{n}_N"]),
            )

        # Connect inputs
        for l in range(analog_system.L):
            self.connects(
                (self[f"IN{l}_P"], self.Xdc[f"IN{l}_P"]),
                (self[f"IN{l}_N"], self.Xdc[f"IN{l}_N"]),
            )

        # Connect outputs (control signals)
        for m in range(analog_system.M):
            self.connects(
                (self[f"S{m}_P"], self.Xdc[f"S{m}_P"]),
                (self[f"S{m}_N"], self.Xdc[f"S{m}_N"]),
            )


class MultiPhaseDigitalControl(SubCircuitElement):
    """The default digital control circuit.

    Parameters
    ----------
    instance_name : str
        The instance name of the digital control circuit.
    analog_system : AnalogSystem
        The analog system to which this digital control circuit belongs.
    digital_control : MultiPhaseDigitalControl
        The digital control circuit to which this digital control circuit belongs.
    in_high : float
        The high input voltage of the comparator.
    in_low : float
        The low input voltage of the comparator.
    out_high : float
        The high output voltage of the comparator.
    out_low : float
        The low output voltage of the comparator.
    dither_offset : int
        The offset of the dither control circuit.

    """

    def __init__(
        self,
        instance_name: str,
        analog_system: AnalogSystem,
        digital_control: NominalDigitalControl,
        in_high: float,
        in_low: float,
        out_high: float,
        out_low: float,
        dither_offset: int = 0,
    ):
        super().__init__(
            instance_name,
            "digital_control",
            [Terminal("VSS"), Terminal("VDD"), Terminal("CLK"), Terminal("VCM")]
            + [Terminal(f"X{i}_P") for i in range(analog_system.N)]
            + [Terminal(f"X{i}_N") for i in range(analog_system.N)]
            + [Terminal(f"IN{i}_P") for i in range(analog_system.L)]
            + [Terminal(f"IN{i}_N") for i in range(analog_system.L)]
            + [Terminal(f"S{i}_P") for i in range(analog_system.M)]
            + [Terminal(f"S{i}_N") for i in range(analog_system.M)],
        )

        multi_input_comparator_names = [
            f"Xmic_{m}" for m in range(dither_offset, analog_system.M)
        ]

        self.add(
            *[
                MultiInputComparator(
                    multi_input_comparator_names[m - dither_offset],
                    analog_system,
                    digital_control,
                    m,
                    m - dither_offset,
                    in_high,
                    in_low,
                    out_high,
                    out_low,
                )
                for m in range(dither_offset, analog_system.M)
            ]
        )

        phase_clock_names = [
            Terminal(f"CLK_{m}") for m in range(dither_offset, analog_system.M)
        ]
        delay_names = [f"A_delay_{m}" for m in range(dither_offset, analog_system.M)]

        self.add(
            *[
                AnalogDelay(
                    delay_names[m - dither_offset],
                    f"analog_delay_{m}",
                    digital_control._impulse_response[m].t0,
                )
                for m in range(dither_offset, analog_system.M)
            ]
        )

        for m_tilde in range(analog_system.M_tilde):
            comparator: MultiInputComparator = self.__getattr__(
                multi_input_comparator_names[m_tilde]
            )
            delay: AnalogDelay = self.__getattr__(delay_names[m_tilde])

            self.connects(
                (self["CLK"], delay["IN"]),
                (self["VDD"], delay["CNTRL"]),
                (phase_clock_names[m_tilde], delay["OUT"]),
                (phase_clock_names[m_tilde], comparator["CLK"]),
                (self["VSS"], comparator["VSS"]),
                (self["VDD"], comparator["VDD"]),
                (self["VCM"], comparator["VCM"]),
            )
            # Connect States
            for n in range(analog_system.N):
                self.connects(
                    (self[f"X{n}_P"], comparator[f"X{n}_P"]),
                    (self[f"X{n}_N"], comparator[f"X{n}_N"]),
                )

            # Connect inputs
            for l in range(analog_system.L):
                self.connects(
                    (self[f"IN{l}_P"], comparator[f"IN{l}_P"]),
                    (self[f"IN{l}_N"], comparator[f"IN{l}_N"]),
                )

            # Connect outputs (control signals)
            for m in range(analog_system.M):
                self.connects(
                    (self[f"S{m}_P"], comparator[f"S{m}_P"]),
                    (self[f"S{m}_N"], comparator[f"S{m}_N"]),
                )


class ModulatorDigitalControl(SubCircuitElement):
    """The default digital control circuit.

    Parameters
    ----------
    instance_name : str
        The instance name of the digital control circuit.
    analog_system : AnalogSystem
        The analog system to which this digital control circuit belongs.
    digital_control : MultiPhaseDigitalControl
        The digital control circuit to which this digital control circuit belongs.
    in_high : float
        The high input voltage of the comparator.
    in_low : float
        The low input voltage of the comparator.
    out_high : float
        The high output voltage of the comparator.
    out_low : float
        The low output voltage of the comparator.
    dither_offset : int
        The offset of the dither control circuit.

    """

    def __init__(
        self,
        instance_name: str,
        analog_system: AnalogSystem,
        digital_control: NominalModulatorControl,
        in_high: float,
        in_low: float,
        out_high: float,
        out_low: float,
        dither_offset: int = 0,
    ):
        super().__init__(
            instance_name,
            "digital_control",
            [Terminal("VSS"), Terminal("VDD"), Terminal("CLK"), Terminal("VCM")]
            + [Terminal(f"X{i}_P") for i in range(analog_system.N)]
            + [Terminal(f"X{i}_N") for i in range(analog_system.N)]
            + [Terminal(f"IN{i}_P") for i in range(analog_system.L)]
            + [Terminal(f"IN{i}_N") for i in range(analog_system.L)]
            + [Terminal(f"S{i}_P") for i in range(analog_system.M)]
            + [Terminal(f"S{i}_N") for i in range(analog_system.M)],
        )

        multi_input_comparator_names = [
            f"Xmic_{m}" for m in range(dither_offset, analog_system.M)
        ]

        self.add(
            *[
                MultiInputComparator(
                    multi_input_comparator_names[m - dither_offset],
                    analog_system,
                    digital_control,
                    m,
                    m - dither_offset,
                    in_high,
                    in_low,
                    out_high,
                    out_low,
                )
                for m in range(dither_offset, analog_system.M)
            ]
        )

        phase_clock_names = [
            Terminal(f"CLK_{m}") for m in range(dither_offset, analog_system.M)
        ]
        delay_names = [f"A_delay_{m}" for m in range(dither_offset, analog_system.M)]

        self.add(
            *[
                AnalogDelay(
                    delay_names[m - dither_offset],
                    f"analog_delay_{m}",
                    digital_control._phi_1[m],
                )
                for m in range(dither_offset, analog_system.M)
            ]
        )

        for m_tilde in range(analog_system.M_tilde):
            comparator: MultiInputComparator = self.__getattr__(
                multi_input_comparator_names[m_tilde]
            )
            delay: AnalogDelay = self.__getattr__(delay_names[m_tilde])

            self.connects(
                (self["CLK"], delay["IN"]),
                (self["VDD"], delay["CNTRL"]),
                (phase_clock_names[m_tilde], delay["OUT"]),
                (phase_clock_names[m_tilde], comparator["CLK"]),
                (self["VSS"], comparator["VSS"]),
                (self["VDD"], comparator["VDD"]),
                (self["VCM"], comparator["VCM"]),
            )
            # Connect States
            for n in range(analog_system.N):
                self.connects(
                    (self[f"X{n}_P"], comparator[f"X{n}_P"]),
                    (self[f"X{n}_N"], comparator[f"X{n}_N"]),
                )

            # Connect inputs
            for l in range(analog_system.L):
                self.connects(
                    (self[f"IN{l}_P"], comparator[f"IN{l}_P"]),
                    (self[f"IN{l}_N"], comparator[f"IN{l}_N"]),
                )

            # Connect outputs (control signals)
            for m in range(analog_system.M):
                self.connects(
                    (self[f"S{m}_P"], comparator[f"S{m}_P"]),
                    (self[f"S{m}_N"], comparator[f"S{m}_N"]),
                )
