"""analog frontends

An analog frontend consist of the combination of
an analog system and a digital control.

"""
from typing import List, Type, Union
from cbadc.circuit.module import Module, Wire, SubModules
from cbadc.circuit.observer import Observer
from cbadc.circuit.state_space_equations import AnalogSystem
from cbadc.circuit.analog_system import (
    # AnalogSystemFiniteGainOpAmp,
    AnalogSystemIdealOpAmp,
    # AnalogSystemStateSpaceOpAmp,
    AnalogSystemFirstOrderPoleOpAmp,
)
from cbadc.circuit.digital_control import DigitalControl, SwitchCapacitorControl
from cbadc.digital_estimator import (
    FIRFilter,
    BatchEstimator,
    IIRFilter,
    ParallelEstimator,
)
from cbadc.digital_estimator._filter_coefficients import FilterComputationBackend

_Analog_systems = Union[
    AnalogSystem,
    # AnalogSystemFiniteGainOpAmp,
    AnalogSystemFirstOrderPoleOpAmp,
    AnalogSystemIdealOpAmp,
    # AnalogSystemStateSpaceOpAmp,
]

_Digital_controls = Union[DigitalControl, SwitchCapacitorControl]


class AnalogFrontend(Module):
    """A combined analog system and digital control verilog-ams implementation

    Parameters
    ----------
    analog_system: :py:class`cbadc.circuit.AnalogSystem`
        an analog system verilog-ams module
    digital_control: :py:class`cbadc.circuit.DigitalControl`
        a digital control verilog-ams module
    save_all_variables: bool
        if True, all variables are saved to a csv file, if False only the control signals are saved.
    save_to_filename: str
        the filename to save the observations to.

    Attributes
    ----------
    analog_system: :py:class`cbadc.circuit.AnalogSystem`
        an analog system verilog-ams module
    digital_control: :py:class`cbadc.circuit.DigitalControl`
        a digital control verilog-ams module
    """

    analog_system: _Analog_systems
    digital_control: _Digital_controls
    inputs: List[Wire]
    outputs: List[Wire]

    def __init__(
        self,
        analog_system: _Analog_systems,
        digital_control: _Digital_controls,
        save_all_variables: bool = False,
        save_to_filename: str = "observations.csv",
    ) -> None:
        self.analog_system = analog_system
        self.digital_control = digital_control

        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        self._sgd = Wire("vsgd", True, False, True, comment="signal ground")
        self._clk = Wire("clk", True, False, True, comment="clock signal")
        u = [
            Wire(f"u_{l}", True, False, True, comment=f"input channel {l}")
            for l in range(analog_system.analog_system.L)
        ]
        self.inputs = [
            self._vdd,
            self._gnd,
            self._sgd,
            self._clk,
            *u,
        ]
        s = [
            Wire(f"s_{m}", False, True, True, comment=f"control signal {m}")
            for m in range(digital_control.digital_control.M)
        ]
        self.outputs = [*s]

        ports = [*self.inputs, *self.outputs]
        s_tilde = [
            Wire(
                f"s_tilde_{m_tilde}",
                False,
                False,
                True,
                comment=f"control observation {m_tilde}",
            )
            for m_tilde in range(digital_control.digital_control.M_tilde)
        ]
        nets = [
            *ports,
            *s_tilde,
        ]

        self.save_all_variables = save_all_variables

        if save_all_variables:
            save_variables = [*self.outputs, *s_tilde, *u]
        else:
            save_variables = self.outputs

        observer = Observer(save_variables, filename=save_to_filename)

        submodules = [
            SubModules(
                analog_system, [self._vdd, self._gnd, self._sgd, *u, *s, *s_tilde]
            ),
            SubModules(
                digital_control,
                [self._vdd, self._gnd, self._sgd, self._clk, *s_tilde, *s],
            ),
            SubModules(
                observer,
                [self._vdd, self._gnd, self._sgd, self._clk, *save_variables],
            ),
        ]

        super().__init__(
            "analog_frontend",
            nets,
            ports,
            submodules=submodules,
            filename="analog_frontend",
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "An analog frontend comparise of an analog-system",
            "and digital control interfaced such that",
            "control signals can be generated given a clock signal",
            "and input stimuli.",
        ]

    def get_estimator(
        self,
        DigitalEstimator: Union[
            Type[BatchEstimator],
            Type[FIRFilter],
            Type[IIRFilter],
            Type[ParallelEstimator],
        ],
        eta2: float,
        K1: int,
        K2: int = 0,
        stop_after_number_of_iterations: int = (1 << 63),
        Ts: float = None,
        mid_point: bool = False,
        downsample: int = 1,
        solver_type: FilterComputationBackend = FilterComputationBackend.mpmath,
    ):
        """Return an initialized digital estimator of the corresponding analog frontend

        This is a utility function to create an estimator with adapted filter coefficients
        as the analog system and digital control might have altered system descriptions due
        to implementation details

        Parameters
        ----------
        DigitalEstimator: :py:class:`cbadc.digital_estimator.BatchEstimator`, :py:class:`cbadc.digital_estimator.FIRFilter`, :py:class:`cbadc.digital_estimator.IIRFilter`, :py:class:`cbadc.digital_estimator.ParallelFilter`
            the class of the estimator to be returned. Note that this is not an instantiated
            object but the actual class.
        eta2 : `float`
            the :math:`\eta^2` parameter determines the bandwidth of the estimator.
        K1 : `int`
            batch size.
        K2 : `int`, `optional`
            lookahead size, defaults to 0.
        stop_after_number_of_iterations : `int`
            determine a max number of iterations by the iterator, defaults to
            :math:`2^{63}`.
        Ts: `float`, `optional`
            the sampling time, defaults to the time period of the digital control.
        mid_point: `bool`, `optional`
            set samples in between control updates, i.e., :math:`\hat{u}(kT + T/2)`
            , defaults to False.
        downsample: `int`, `optional`
            set a downsampling factor compared to the control signal rate,
            defaults to 1, i.e., no downsampling.
        solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
            determine which solver type to use when computing filter coefficients.

        """
        return DigitalEstimator(
            self.analog_system.analog_system,
            self.digital_control.digital_control,
            eta2,
            K1=K1,
            K2=K2,
            stop_after_number_of_iterations=stop_after_number_of_iterations,
            Ts=Ts,
            mid_point=mid_point,
            downsample=downsample,
            solver_type=solver_type,
        )
