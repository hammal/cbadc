from .. import CircuitElement, Port, Terminal, _template_env, SPICE_VALUE
from typing import Dict, Union


class Resistor(CircuitElement):
    """Resistor class

    Parameters
    ----------
    terminals : List[Terminal]
        The terminals of the resistor
    value : SPICE_VALUE
        The value of the resistor
    instance_name : str, optional
        The instance name of the resistor, by default None
    """

    r: SPICE_VALUE

    def __init__(
        self,
        instance_name: str,
        value: SPICE_VALUE,
        m: SPICE_VALUE = 1,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f"Expected str, got {type(instance_name)}")
        elif instance_name[0] != "R":
            instance_name = "R" + instance_name

        terminals = [Terminal(), Terminal()]
        super().__init__(
            instance_name,
            terminals,
            r=value,
            m=m,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template("ngspice/resistor.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._get_terminal_names(connections),
                "value": self._parameters_dict["r"],
                "m": self._parameters_dict["m"],
                "comments": self.comments,
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template("spectre/resistor.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._get_terminal_names(connections),
                "value": self._parameters_dict["r"],
                "comments": self.comments,
            }
        )


class Capacitor(CircuitElement):
    """Capacitor class

    Parameters
    ----------
    terminals : List[Terminal]
        The terminals of the capacitor
    value : SPICE_VALUE
        The value of the capacitor
    instance_name : str, optional
        The instance name of the capacitor, by default None
    """

    c: SPICE_VALUE

    def __init__(
        self,
        instance_name: str,
        value: SPICE_VALUE,
        m: SPICE_VALUE = 1,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f"Expected str, got {type(instance_name)}")
        elif instance_name[0] != "C":
            instance_name = "C" + instance_name

        terminals = [Terminal(), Terminal()]
        super().__init__(
            instance_name,
            terminals,
            c=value,
            m=m,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template("ngspice/capacitor.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._get_terminal_names(connections),
                "value": self._parameters_dict["c"],
                "m": self._parameters_dict["m"],
                "comments": self.comments,
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template("spectre/capacitor.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._get_terminal_names(connections),
                "value": self._parameters_dict["c"],
                "comments": self.comments,
            }
        )


class Inductor(CircuitElement):
    """Inductor class

    Parameters
    ----------
    terminals : List[Terminal]
        The terminals of the inductor
    value : SPICE_VALUE
        The value of the inductor
    instance_name : str, optional
        The instance name of the inductor, by default None
    """

    l: SPICE_VALUE

    def __init__(
        self,
        instance_name: str,
        value: SPICE_VALUE,
        m: SPICE_VALUE = 1,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f"Expected str, got {type(instance_name)}")
        elif instance_name[0] != "L":
            instance_name = "L" + instance_name

        terminals = [Terminal(), Terminal()]
        super().__init__(
            instance_name,
            terminals,
            l=value,
            m=m,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template("ngspice/inductor.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._get_terminal_names(connections),
                "value": self._parameters_dict["l"],
                "m": self._parameters_dict["m"],
                "comments": self.comments,
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template("spectre/inductor.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._get_terminal_names(connections),
                "value": self._parameters_dict["l"],
                "comments": self.comments,
            }
        )
