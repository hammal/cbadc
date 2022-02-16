"""generic componets for building verilog-ams modules
"""
from typing import List, Tuple, Union
from jinja2 import Environment, PackageLoader, select_autoescape
from datetime import datetime
from ..__version__ import __version__
import os.path


class Parameter:
    """A verilog-ams parameter class

    Describes the necessary data to constitute a
    verilog-ams parameter.

    Note that parameter names are considered unique
    and are, by that principle, shared between modules/submodules.

    Parameters
    ----------
    name: `str`
        the name of the parameter.
    initial_value: Union[`float`, `int`, `str`, `None`], `optional`
        an initial value, defaults to `None`.
    real: `bool`, `optional`
        a bool determining if the parameter is real valued. If false
        the parameter will be assumed to be integer valued. Defafults
        to True.
    comment: `str`
        a descriptive comment.
    """

    real: bool
    name: str
    initial_value: Union[float, int, str, None]
    comment: str

    def __init__(
        self,
        name: str,
        initial_value: Union[float, int, str, None] = None,
        real: bool = True,
        comment: str = "",
    ):
        self.name = name
        self.initial_value = initial_value
        self.real = real
        self.comment = comment


class Wire:
    """A verilog-ams wire class

    This class holds the necessary data to
    consitute a verilog-ams net.

    Note that if the Wire is instantiating with both a
    negative input and output value, it will be considered an
    internal net to the module.

    Parameters
    ----------
    name: `str`
        the name of the wire/net.
    electrical: `bool`
        a bool determining if this is an electric wire.
    input: `bool`
        a bool determining if this net should be considered an input.
    output: `bool`
        a bool determining if this net should be considered an output.
    comment: `str`
        a descriptive comment.
    """

    name: str
    electrical: bool
    input: bool
    output: bool
    comment: str

    def __init__(
        self,
        name: str,
        input: bool,
        output: bool,
        electrical: bool = True,
        comment: str = "",
    ):
        self.name = name
        self.electrical = electrical
        self.input = input
        self.output = output
        self.comment = comment


class _SubModule:
    env: Environment
    module_name: str
    instance_name: str
    ports: List[Wire]
    nets: List[Wire]
    parameters: List[Parameter]
    analog_statements: List[str]
    analog_initial: List[str]
    description: List[str]
    _filename: str

    def __init__(
        self,
        module_name: str,
        nets: List[Wire],
        ports: List[Wire],
        instance_name='',
        parameters: List[Parameter] = [],
        analog_statements: List[str] = [],
        analog_initial: List[str] = [],
    ):
        self.env = Environment(
            loader=PackageLoader("cbadc", package_path="circuit_level/templates"),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.module_name = module_name
        self.instance_name = instance_name
        self.nets = nets
        self.ports = ports
        self._check_ports()
        self.parameters = parameters
        self.analog_statements = analog_statements
        self.analog_initial = analog_initial

    def _get_port(self, key: Wire):
        if key in self.ports:
            return self.ports[self.ports.index(key)]

    def _get_net(self, key: Wire):
        if key.name in [n.name for n in self.nets]:
            return key

    def _get_parameter(self, key: Parameter):
        if key.name in [p.name for p in self.parameters]:
            return key

    def __getitem__(self, key: Wire):
        return self._get_port(key)

    def _check_ports(self):
        for port in self.ports:
            if port not in self.nets:
                raise Exception("All ports must be specified among the nets")

    def _module_comment(self) -> List[str]:
        return [
            f"{self.module_name}",
            "",
            f"Ports: {', '.join([p.name for p in self.ports])}",
            "",
            f"Parameters: {', '.join([p.name for p in self.parameters])}",
        ]

    def render(self, rendered_modules=[]):
        template = self.env.get_template('module.vams')

        def input_filter(key: Wire):
            return key.input and not key.output

        def output_filter(key: Wire):
            return not key.input and key.output

        def inout_filter(key: Wire):
            return key.input and key.output

        def real_parameters(key: Parameter):
            return key.real

        def define_parameters(key: Parameter):
            if key.initial_value is not None:
                return f"{key.name} = {key.initial_value}"
            return key.name

        rendered_module_strings = template.render(
            {
                "module_name": self.module_name,
                "ports": [p for p in self.ports],
                "inputs": [n for n in filter(input_filter, self.ports)],
                "outputs": [n for n in filter(output_filter, self.ports)],
                "inouts": [n for n in filter(inout_filter, self.ports)],
                "real_parameters": [
                    p for p in filter(real_parameters, self.parameters)
                ],
                "int_parameters": [
                    p for p in filter(lambda x: not real_parameters(x), self.parameters)
                ],
                "electricals": [
                    w.name for w in filter(lambda x: x.electrical, self.nets)
                ],
                "analog_initial": self.analog_initial,
                "analog": self.analog_statements,
                "comment": self._module_comment(),
            }
        )
        return [rendered_module_strings], [*rendered_modules, self]

    def __str__(self):
        return "\n".join(self._module_comment())


class SubModules:
    """A circuit_level submodule

    Holds a module and the :py:class:`cbadc.circuit_level.Wire` which
    are used to connect it inside the module.

    Parameters
    ----------
    submodule: :py:class:`cbadc.circuit_level.Module`
        the module which constitutes the submodule.
    connections: List[ :py:class:`cbadc.circuit_level.Wire` ]
        a lis of connections, dictating how the ports of the submodule
        are connected to the parent module.

    Attributes
    ----------
    module: :py:class:`cbadc.circuit_level.Module`
        the module which constitutes the submodule.
    connections: List[:py:class:`cbadc.circuit_level.Wire`]
        a lis of connections, dictating how the ports of the submodule
        are connected to the parent module.
    parameters: List[:py:class:`cbadc.circuit_level.Parameter`]
        number of parameters shared between module and submodule.
    number_of_connections: `int`
        the number of connecting ports.

    """

    module: _SubModule
    connections: List[Wire]
    parameters: List[Parameter]
    number_of_connections: int

    def __init__(self, submodule: _SubModule, connections: List[Wire]) -> None:
        self.module = submodule
        self.connections = connections
        self.number_of_connections = len(self.connections)
        self.parameters = []


class Module(_SubModule):
    """A verilog-ams module

    Implements a python interface for specifying
    a verilog-ams module.

    Parameters
    ----------
    module_name: `str`
        name of the module
    nets: List[ :py:class:`cbadc.circuit_level.Wire` ]
        a list containting all internal/interfacing nets of
        the module.
    ports: List[ :py:class:`cbadc.circuit_level.Wire` ]
        the input/output nets of the module.
    instance_name: `str`, `optional`
        a specific name for this instance of the module. usefull when
        having multiple submodules of the same type embedded inside a
        parent module.
    parameters: List[ :py:class:`cbadc.circuit_level.Parameter` ]
        a list of parameters. Note that parameter names must be unique since
        modules/submodules with same names get connected.
    analog_statements: List[`str`]
        a list of analog statements defining the inner analog functions of the
        module.
    analog_initial: List[`str`]
        a list of analog initial statements.
    submodules: List[:py:class:`cbadc.circuit_level.Module`]
        a list of submodules to be included connected inside the module.
    description: List[`str`]
        a description of the verilog module.

    Attributes
    ----------
    env: :py:class:`jinja2.Environment`
        a jinja2 Environment instance used for rendering templates of verilog-ams
        modules.
    module_name: `str`
        the name of the module.
    instance_name: `str`, `optional`
        the name of this instance of the module. usefull when
        having multiple submodules of the same type embedded inside a
        parent module.
    ports: List[ :py:class:`cbadc.circuit_level.Wire` ]
        the input/output nets of the module.
    nets: List[ :py:class:`cbadc.circuit_level.Wire` ]
        the list containting all internal/interfacing nets of
        the module.
    parameters: List[ :py:class:`cbadc.circuit_level.Parameter` ]
        the list of parameters. Note that parameter names must be unique since
        modules/submodules with same names get connected.
    analog_statements: List[`str`]
        the list of analog statements defining the inner analog functions of the
        module.
    analog_initial: List[`str`]
        the list of analog initial statements.
    submodules: List[ :py:class:`cbadc.circuit_level.SubModule` ]
        the list of submodules to be included connected inside the module.
    description: List[`str`]
        the description of the verilog module.

    """

    def __init__(
        self,
        module_name: str,
        nets: List[Wire],
        ports: List[Wire],
        instance_name='',
        parameters: List[Parameter] = [],
        analog_statements: List[str] = [],
        analog_initial: List[str] = [],
        submodules: List[SubModules] = [],
        filename: str = "no_name.vams",
    ):
        super().__init__(
            module_name,
            nets,
            ports,
            instance_name,
            parameters,
            analog_statements,
            analog_initial,
        )
        if not filename.endswith('.vams'):
            filename = f"{filename}.vams"
        self._filename = filename
        self.submodules = submodules
        for s_module in self.submodules:
            if len(s_module.module.ports) != len(s_module.connections):
                raise Exception(
                    "The connections must match the number of submodules ports."
                )
            for port in s_module.connections:
                if self._get_net(port) is None:
                    raise Exception(
                        "Each connection of a submodule must exist in the parent module."
                    )
            for parameter in s_module.module.parameters:
                local_parameter = self._get_parameter(parameter)
                if local_parameter:
                    s_module.parameters.append(local_parameter)

    def to_file(self, filename: str = None) -> None:
        """Write this module to verilog-ams file

        Note that any submodules will also be included
        in the same file.

        Parameters
        ----------
        filename: `str`, `optional`
            the filename to write module definition to, defaults
            to no_name.
        """
        preamble = self.env.get_template('preamble.txt').render(
            {
                "datetime": datetime.isoformat(datetime.now()),
                "cbadc_version": __version__,
                "verilog_ams": True,
            }
        )
        res = "\n\n\n".join([preamble, *self.render()[0]])
        if filename:
            if not filename.endswith('.vams'):
                filename = f"{filename}.vams"
            self._filename = filename

        with open(os.path.abspath(self._filename), 'w') as f:
            f.write(res)

    def render(
        self, rendered_modules: List[_SubModule] = []
    ) -> Tuple[List[str], List[_SubModule]]:
        """Render the module into a verilog-ams compliant string

        Parameters
        ----------
        rendered_modules: List[ py:class:`cbadc.circuit_level.Module` ]
            a list of modules that have already been rendered. a tool use to exclude
            modules from the render if needed.

        Returns
        -------
        : Tuple[`str`, List[ :py:class:`cbadc.circuit_level.Module` ]]
            a rendered string and list including a flattended version of all
            rendered modules and submodules.
        """
        template = self.env.get_template('module.vams')
        rendered_modules_strings = []
        for submod in self.submodules:
            if submod.module.module_name not in [
                r.module_name for r in rendered_modules
            ]:
                render_string, rendered_modules = submod.module.render(rendered_modules)
                rendered_modules_strings = [*render_string, *rendered_modules_strings]

        def input_filter(key: Wire):
            return key.input and not key.output

        def output_filter(key: Wire):
            return not key.input and key.output

        def inout_filter(key: Wire):
            return key.input and key.output

        def real_parameters(key: Parameter):
            return key.real

        rendered_modules_strings.append(
            template.render(
                {
                    "module_name": self.module_name,
                    "ports": [p for p in self.ports],
                    "inputs": [n for n in filter(input_filter, self.ports)],
                    "outputs": [n for n in filter(output_filter, self.ports)],
                    "inouts": [n for n in filter(inout_filter, self.ports)],
                    "real_parameters": [
                        p for p in filter(real_parameters, self.parameters)
                    ],
                    "int_parameters": [
                        p
                        for p in filter(
                            lambda x: not real_parameters(x), self.parameters
                        )
                    ],
                    "electricals": [
                        w.name for w in filter(lambda x: x.electrical, self.nets)
                    ],
                    "analog_initial": self.analog_initial,
                    "submodules": self.submodules,
                    "analog": self.analog_statements,
                    "comment": self._module_comment(),
                }
            )
        )
        return rendered_modules_strings, [*rendered_modules, self]
