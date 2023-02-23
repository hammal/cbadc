from enum import Enum
from typing import Any, List, Dict, Set, Tuple, Union
from jinja2 import Environment, PackageLoader, select_autoescape
import itertools
import logging
import numpy as np
from os import path

logger = logging.getLogger(__name__)

_template_env = Environment(
    loader=PackageLoader("cbadc", package_path="circuit/templates"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


class SpiceDialect(Enum):
    ngspice = 'ngspice'
    specter = 'spectre'


class ComponentType(Enum):
    resistor = 'R'
    capacitor = 'C'
    inductor = 'L'
    voltage_source = ('V',)
    voltage_controlled_voltage_source = 'E'
    current_controlled_voltage_source = 'H'
    voltage_controlled_current_source = 'F'
    current_controlled_current_source = 'G'
    current_source = "I"
    switch = 'S'
    diode = 'D'
    mosfet = 'M'
    bipolar = 'Q'
    sub_circuit = 'X'
    xspice = 'A'


class Terminal:
    """A terminal is a connection point for a component

    Parameters
    ----------
    name : str, optional
        The name of the terminal
    """

    name: str
    id_iter = itertools.count(1)
    hidden: bool

    def __init__(self, name="", hidden=False):
        self.id = next(self.id_iter)
        if not name:
            name = f"{self.id}"
        self.name = name.upper()
        self.hidden = hidden

    def __str__(self):
        return self.name


class Ground(Terminal):
    """The ground terminal"""

    def __init__(self):
        self.name = '0'
        self.id = 0
        self.hidden = False


class Port:
    """A port is a collection of terminals that are connected together

    Parameters
    ----------
    terminals : Tuple[Terminal, Terminal]
        The terminals that are connected together
    """

    terminals: Set[Terminal]
    id_iter = itertools.count(1)
    name: str

    def __init__(self, terminals: Tuple[Terminal, Terminal]):
        self.id = next(self.id_iter)
        self.terminals = {*terminals}

    def add_terminal(self, terminal: Terminal):
        """Add a terminal to the port

        Parameters
        ----------
        terminal : Terminal
            The terminal to add
        """
        self.terminals.add(terminal)

    def merge(self, other):
        """Merge another port into this port

        Parameters
        ----------
        other : Port
            The port to merge into this port

        """
        if not isinstance(other, Port):
            raise TypeError("Can only merge Port objects")
        self.terminals = self.terminals.union(other.terminals)


class DeviceModel:
    """A device model

    Parameters
    ----------
    name : str
        The name of the model
    type_name : str
        The type name of the model (used for later instantiation)
    comments : List[str], optional
        Comments to add to the model, by default []
    **kwargs : Dict[str, str]
        The parameters of the model
    """

    ng_spice_model_name: str
    model_name: str
    parameters: Dict[str, str]
    comments: List[str]
    verilog_ams: bool

    def __init__(
        self,
        model_name: str,
        comments=[],
        **kwargs,
    ) -> None:
        self.model_name = model_name.lower()
        self.parameters = kwargs
        self.comments = comments
        self.verilog_ams = False

    def get_ngspice(self):
        raise NotImplementedError

    def get_spectre(self):
        raise NotImplementedError

    def get_verilog_ams(self):
        raise NotImplementedError

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, DeviceModel):
            return False
        return (self.ng_spice_model_name, self.model_name) == (
            __o.ng_spice_model_name,
            __o.model_name,
        )

    def __hash__(self) -> int:
        return hash((self.ng_spice_model_name, self.model_name))


class CircuitElement:
    """Circuit element base class

    Parameters
    ----------
    component_type : ComponentType
        The type of the component
    terminals : List[Terminal]
        The terminals of the component
    instance_name : str, optional
        The instance name of the component, by default ''
    comments : List[str], optional
        Comments to add to the component, by default []
    model : DeviceModel, optional
        The model to use for the component, by default None
    library : str, optional
        The library to use for the component, by default ''
    include : str, optional
        The include file to use for the component, by default ''
    """

    instance_name: str
    _terminals: List[Terminal]
    _terminal_lookup: Dict[str, int]
    _parameters_dict: Dict[str, str]
    _parameter_list: List[str]
    comments: List[str]
    model: DeviceModel

    def __init__(
        self,
        instance_name: str,
        terminals: List[Terminal],
        *args,
        comments: List[str] = [],
        model: DeviceModel = None,
        **kwargs,
    ):
        # Make sure instance name is of correct type and format
        if not instance_name or not isinstance(instance_name, str):
            raise ValueError("Instance name must be a non-empty string")
        self.instance_name = instance_name[0] + instance_name[1:].lower()

        #  Add terminals
        self._terminals = []
        self._terminal_lookup = {}
        self.add_terminals(terminals)

        # Make sure comments are of correct type
        if not isinstance(comments, list):
            raise ValueError("Comments must be a list")
        for comment in comments:
            if not isinstance(comment, str):
                raise ValueError("Comments must be of type str")
        self.comments = comments

        # Make sure model is of correct type
        if model and not isinstance(model, DeviceModel):
            raise ValueError("Model must be of type DeviceModel")
        self.model = model

        self._parameter_list = [str(arg) for arg in args]
        self._parameters_dict = {key: str(value) for key, value in kwargs.items()}

    def _get_terminal_names(self, connections: Dict[Terminal, Port]) -> List[str]:
        terminal_names = []
        for terminal in self._terminals:
            if terminal in connections:
                if connections[terminal].name:
                    terminal_names.append(connections[terminal].name)
                else:
                    terminal_names.append(f"{connections[terminal].id}")
            else:
                if not terminal.hidden:
                    logger.warning(f"Terminal {terminal} is not connected to anything")
                terminal_names.append(str(terminal))
        return terminal_names

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        """Get the ngspice call for the component"""
        raise NotImplementedError

    def get_spectre(self, connections: Dict[Terminal, Port]):
        """Get the spectre call for the component"""
        raise NotImplementedError

    def _get_model_set(self, verilog_ams=False) -> List[DeviceModel]:
        if self.model:
            return [self.model]
        return []

    def get_terminals(self) -> List[Terminal]:
        """Get the terminals of the component"""
        return self._terminals

    def add_terminal(self, terminal: Terminal, index: int = -1):
        """Add a terminal to the component

        Parameters
        ----------
        terminal : Terminal
            The terminal to add
        index : int, optional
            The index to add the terminal at, by default -1, i.e., at the end
        """
        if not isinstance(terminal, Terminal):
            raise ValueError("Terminal must be of type Terminal")
        if index == -1:
            if terminal.name in self._terminal_lookup:
                raise ValueError(f"Terminal {terminal.name} name already exists")
            self._terminal_lookup[terminal.name] = len(self._terminals)
            self._terminals.append(terminal)
        elif index >= 0 and index < len(self._terminals):
            self._terminals.insert(index, terminal)
            # Update terminal lookup
            for index, term in enumerate(self._terminals):
                self._terminal_lookup[term.name] = index
        else:
            raise ValueError("Index out of range")

    def __getitem__(self, key) -> Terminal:
        if isinstance(key, int):
            return self._terminals[key]
        elif isinstance(key, str):
            upper_key = key.upper()
            if upper_key not in self._terminal_lookup:
                raise ValueError(f"Terminal {upper_key} does not exist")
            return self._terminals[self._terminal_lookup[upper_key]]
        raise ValueError("Key must be of type int or str")

    def add_terminals(self, terminals: List[Terminal]):
        """Add multiple terminals to the component

        Parameters
        ----------
        terminals : List[Terminal]
            The terminals to add
        """
        if not isinstance(terminals, list):
            raise ValueError("Terminals must be a list")
        for terminal in terminals:
            if not isinstance(terminal, Terminal):
                raise ValueError("Terminals must be of type Terminal")
            self.add_terminal(terminal)

    def __str__(self):
        result = [
            "Circuit Element\n---------------",
            f"Name: {self.instance_name}",
            f"Terminals: {[str(x) for x in self._terminals]}",
            f"Parameters: {self._parameters_dict}",
            f"Comments: {self.comments}",
            "---------------",
        ]

        return "\n".join(result)


class SubCircuitElement(CircuitElement):
    """Subcircuit element class

    Parameters
    ----------
    terminals : List[Terminal]
        The terminals of the component
    subckt_name : str
        The name of the subcircuit
    instance_name : str, optional
        The instance name of the component, by default ''
    comments : List[str], optional
        Comments to add to the component, by default []
    """

    subckt_name: str
    _internal_connections: Dict[Terminal, Port]
    _subckts: Dict[str, CircuitElement]
    _subckt_names: List[str] = []

    def __init__(
        self,
        instance_name: str,
        subckt_name: str,
        terminals: List[Terminal],
        *args,
        comments: List[str] = [],
        **kwargs,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise ValueError("Instance name must be a non-empty string")

        # Make sure the instance name is valid
        if instance_name[0] != 'X':
            instance_name = f"X{instance_name}"

        super().__init__(
            instance_name,
            terminals,
            subckt_name,
            *args,
            comments=comments,
            **kwargs,
        )

        if not subckt_name or not isinstance(subckt_name, str):
            raise ValueError("Subcircuit name must be a non-empty string")
        self.subckt_name = subckt_name.lower()
        self._subckts = {}
        self._internal_connections = {Ground(): Port((Ground(), Ground()))}

    def get_ngspice(self, connections: Union[Dict[Terminal, Port], None] = None):
        connections = self._merge_connections_downstream(connections)
        return _template_env.get_template('ngspice/sub_circuit.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'subckt_name': self.subckt_name,
                'terminals': self._get_terminal_names(connections),
                'parameters': self._parameters_dict,
            }
        )

    def _merge_connections_downstream(
        self, connections: Union[Dict[Terminal, Port], None]
    ):
        if connections is None:
            connections = self._internal_connections
        if not isinstance(connections, dict):
            raise ValueError("Connections must be a Dict[Terminal, Port]")
        for key, value in connections.items():
            if not isinstance(key, Terminal):
                raise ValueError("Connections must be a Dict[Terminal, Port]")
            if not isinstance(value, Port):
                raise ValueError("Connections must be a Dict[Terminal, Port]")
        return connections

    def get_spectre(self, connections: Union[Dict[Terminal, Port], None] = None):
        connections = self._merge_connections_downstream(connections)
        return _template_env.get_template('spectre/sub_circuit.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'subckt_name': self.subckt_name,
                'terminals': self._get_terminal_names(connections),
                'parameters': self._parameters_dict,
            }
        )

    def get_sub_circuit_definitions(
        self, dialect: SpiceDialect = SpiceDialect.ngspice
    ) -> List[str]:
        """Get the spice definition of the subcircuit"""

        if dialect == SpiceDialect.ngspice:
            result = [
                _template_env.get_template(
                    "ngspice/sub_circuit_definition.cir.j2"
                ).render(
                    {
                        "subckt_name": self.subckt_name,
                        "terminals": self._get_terminal_names(
                            self._internal_connections
                        ),
                        "parameters": self._parameters_dict,
                        "sub_circuits": self._get_subckts(),
                        "connections": self._internal_connections,
                    }
                )
            ]

            for sub_circuit in self._get_subckts():
                if isinstance(sub_circuit, SubCircuitElement):
                    for sub_sub_circuit in sub_circuit.get_sub_circuit_definitions():
                        if sub_sub_circuit not in result:
                            result.append(sub_sub_circuit)
            return result
        elif dialect == SpiceDialect.specter:
            return self._get_spectre_sub_circuit_definition()

        raise ValueError(f"Unknown dialect {dialect}")

    def _get_spectre_sub_circuit_definition(self) -> List[str]:
        result = [
            _template_env.get_template("spectre/sub_circuit_definition.cir.j2").render(
                {
                    "subckt_name": self.subckt_name,
                    "terminals": self._get_terminal_names(self._internal_connections),
                    "parameters": self._parameters_dict,
                    "sub_circuits": self._get_subckts(),
                    "connections": self._internal_connections,
                }
            )
        ]

        for sub_circuit in self.subckt_components:
            if isinstance(sub_circuit, SubCircuitElement):
                result.extend(sub_circuit._get_spectre_sub_circuit_definition())
        return result

    def _get_model_set(self, verilog_ams=False) -> List[DeviceModel]:
        result_set = []
        for sub_circuit in self._get_subckts():
            candidates = sub_circuit._get_model_set(verilog_ams)
            for candidate in candidates:
                if candidate not in result_set:
                    result_set.append(candidate)
        return result_set

    def _get_subckts(self):
        return self._subckts.values()

    def __setattr__(self, __name: Any, __value: Any) -> None:
        if not isinstance(__name, str):
            raise ValueError("Subcircuit instance name must be a string")
        if isinstance(__value, CircuitElement):
            # Check if a new subcircuit is being added
            if __name not in self._subckts:
                if __name != __value.instance_name:
                    raise ValueError(
                        f"Instance name of {__value.instance_name} is not {__name}"
                    )
            else:
                # Replace the existing subcircuit
                if __name != __value.instance_name:
                    __value.instance_name = __name
                old_subckt = self._subckts[__name]
                for terminal in old_subckt.get_terminals():
                    self.connect(__value[terminal.name], terminal)
            self._subckts[__name] = __value
            return None
        return super().__setattr__(__name, __value)

    def __getattr__(self, __name: str) -> Any:
        if __name in self._subckts:
            return self._subckts[__name]
        raise AttributeError(f"Subcircuit {self.subckt_name} has no attribute {__name}")

    # def __getitem__(self, __name: Union[str, int]) -> Union[CircuitElement, Terminal]:
    #     if isinstance(__name, int) or __name in self._terminals:
    #         return super().__getitem__(__name)
    #     elif __name in self._subckts:
    #         return self._subckts[__name]
    #     raise KeyError(f"Subcircuit {self.subckt_name} has no element {__name}")

    # def __setitem__(self, __name: str, __value: Any) -> None:
    #     if isinstance(__value, CircuitElement):
    #         if __name is not __value.instance_name:
    #             raise ValueError(f"Instance name of {__value} is not {__name}")
    #         self._subckts[__name] = __value
    #         return None
    #     raise ValueError(
    #         f"Subcircuit {self.subckt_name} can only contain CircuitElements"
    #     )

    def add(self, *elements: CircuitElement):
        """Add elements to the subcircuit

        Parameters
        ----------
        elements : CircuitElement
            The elements to add
        """
        for element in elements:
            if not isinstance(element, CircuitElement):
                raise ValueError(f"Element {element} is not a CircuitElement")
            self.__setattr__(element.instance_name, element)

    def connects(self, *connections: tuple[Terminal, Terminal]):
        """Add connections to the subcircuit

        Parameters
        ----------
        args : list[tuple[Terminal, Terminal]]
            The connections to add
        """
        for conn in connections:
            self.connect(conn[0], conn[1])

    def connect(self, first: Terminal, second: Terminal, name=''):
        """Add a connection between two terminals

        Parameters
        ----------
        first : Terminal
            First terminal
        second : Terminal
            Second terminal
        name : str, optional
            Name of the connection, by default tries to inherit the name of the first named terminal

        Notes
        -----
        - If the terminals are already connected, the connection is not changed
        - If one of the terminals is already connected, the other terminal is added to the connection
        - If neither of the terminals is connected, a new connection is created
        - If both terminals are connected to different connections, the connections are merged
        - If both terminals are connected to the same connection, the connection is not changed
        - If one of the terminals has a name, the connection is named after the terminal with the name
        - If both terminals have a name, the connection is named after the first terminal
        - If neither of the terminals have a name, the connection is unnamed
        """
        if not isinstance(first, Terminal):
            raise ValueError(f"First terminal {first} is not a Terminal")
        if not isinstance(second, Terminal):
            raise ValueError(f"Second terminal {second} is not a Terminal")

        if first in self._internal_connections and second in self._internal_connections:
            self._internal_connections[first].merge(self._internal_connections[second])
            self._internal_connections[second] = self._internal_connections[first]
        elif (
            first in self._internal_connections
            and second not in self._internal_connections
        ):
            self._internal_connections[first].add_terminal(second)
            self._internal_connections[second] = self._internal_connections[first]
        elif (
            first not in self._internal_connections
            and second in self._internal_connections
        ):
            self._internal_connections[second].add_terminal(first)
            self._internal_connections[first] = self._internal_connections[second]
        else:
            self._internal_connections[first] = Port((first, second))
            self._internal_connections[second] = self._internal_connections[first]

        if name:
            self._internal_connections[first].name = name
        else:
            if first.name:
                self._internal_connections[first].name = first.name
            elif second.name:
                self._internal_connections[first].name = second.name
            else:
                self._internal_connections[first].name = ''

    def check_connections(self):
        """Check that all terminals are connected

        Raises
        ------
        ValueError
            If a terminal of the subckt is not connected
        """
        for component in self._get_subckts():
            if isinstance(component, SubCircuitElement):
                component.check_connections()
            for terminal in component.get_terminals():
                if terminal not in self._internal_connections and not terminal.hidden:
                    logger.warning(
                        f"Terminal {terminal} of component {component} not connected and not marked as hidden"
                    )
                    # raise ValueError(
                    #     f"Terminal {terminal} of component {component} not connected"
                    # )
        for terminal in self.get_terminals():
            if terminal not in self._internal_connections and not terminal.hidden:
                logger.warning(
                    f"Terminal {terminal} is not connected and not marked as hidden"
                )
                # raise ValueError(
                #     f"Terminal {terminal} named {self.instance_name} not connected"
                # )

    def _check_subckt_names(self) -> list[str]:
        # Lazy evaluation of subckt names
        if not self._subckt_names:
            self._subckt_names = [self.subckt_name]
            for component in self._get_subckts():
                if isinstance(component, SubCircuitElement):
                    self._subckt_names.extend(component._check_subckt_names())
        return self._subckt_names

    def check_subckt_names(self):
        """Check that all subcircuit names are unique

        Raises
        ------
        ValueError
            If a subcircuit name is not unique
        """
        for component in self._get_subckts():
            if isinstance(component, SubCircuitElement):
                component.check_subckt_names()
                if self.subckt_name in component._check_subckt_names():
                    raise ValueError(f"Subcircuit name {self.subckt_name} not unique")

    def connect_upstream(self):
        """Connect all terminals of the subcircuit to the parent circuit

        Notes
        -----
        - If a terminal is already connected, the connection is not changed
        - If a terminal is not connected, a new connection is created
        - If a terminal has a name, the connection is named after the terminal

        """
        for component in self._get_subckts():
            if isinstance(component, SubCircuitElement):
                component.connect_upstream()
            for terminal in component.get_terminals():
                if terminal not in self._internal_connections:
                    self._terminals.append(Terminal())
                    self.connect(terminal, self.terminals[-1])

    def __str__(self):
        return """
This Circuit Element is a SubCircuitElement with the following subcircuit components:
--------------------

""" + "\n\n\n".join(
            [str(x) for x in self.subckt_components]
        )


class NetlistElement(SubCircuitElement):
    """A SpiceNetlist is a SubCircuitElement that can be used to include a netlist in a circuit

    Parameters
    ----------
    netlist : str
        The netlist to include
    """

    _id_iter = itertools.count(1)

    def __init__(self, netlist: str):
        # check if netlist is a filename
        if netlist.endswith(('.txt', '.net', '.cir')):
            # if so read from the file
            with open(netlist, 'r') as f:
                netlist = f.read()

        # Parse netlist
        netlist_lines = netlist.splitlines()
        first_line = netlist_lines[0].split()

        # Check for .subckt statement
        if not first_line[0].lower() == '.subckt':
            raise ValueError(
                f'Netlist does not start with .subckt, but with {first_line[0]}'
            )

        # Check for .endc statement
        _endc_statement = False
        for line in netlist_lines[1:]:
            if line.split()[0].lower() == '.ends':
                _endc_statement = True
        if not _endc_statement:
            raise ValueError('Netlist does have a required .ends statement')

        # Extract subcircuit and terminal names.
        subckt_name = first_line[1].lower()
        terminals = [Terminal(term_name.upper()) for term_name in first_line[2:]]

        self._netlist = netlist

        super().__init__(
            f'X{next(self._id_iter)}',
            subckt_name,
            terminals,
        )

    def get_sub_circuit_definitions(
        self, dialect: SpiceDialect = SpiceDialect.ngspice
    ) -> List[str]:
        """Get the spice definition of the subcircuit"""
        return [self._netlist]

    def __str__(self):
        return f"""
This Circuit Element is a NetlistElement with the following definition:
--------------------

{self._netlist}

"""


# SPICE_VALUE = Union[float, str, int]
SPICE_VALUE = float
