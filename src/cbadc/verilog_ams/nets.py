"""Defining verilog-ams nets

For more information see `this page <https://verilogams.com/refman/basics/wires.html>`_
"""
from typing import Union
from cbadc.verilog_ams.nature_and_disciplines import Current, Disciplines, Voltage
from cbadc.verilog_ams.variables import _Variables
from cbadc.verilog_ams.expressions import BaseExpression, Expression, Parenthesis
from .keywords import raise_exception_for_keyword
from .string_literals import end_line
from . import numbers as _numbers


class Wire(_Variables):
    """A universal connector component

    Each wire type component has a name and
    a definition associated with it.

    Parameters
    ----------
    name: `str`
        the name of the component.

    """

    def __init__(self, name: str, discipline: Disciplines):
        self._discipline = discipline
        super().__init__(name)
        self._definition = [
            f"wire {self._name};",
            f"{self._discipline.name} {self._name};"
        ]

    def discrete_time(self):
        """Check if discrete-time wire node.

        Returns
        -------
        : `bool`
            True if discrete-time node.
        """
        return (self._discipline.domain == 'discrete')


class Logic(Wire):
    """The logic describes a verilog-ams wire component.

    Parameters
    ----------
    name: `str`
        name of component.
    initial_value: `str`, `optional`
        an initial value, defaults to None.
    """

    def __init__(self, name: str, initial_value: Union[None, _numbers.Logic] = None):
        discipline = Disciplines(
            'logic',
            None,
            None,
            domain='discrete'
        )
        super().__init__(name, discipline)
        self._initial_value = initial_value
        raise_exception_for_keyword(str(self._initial_value))

        if self._initial_value:
            temp_def = self._definition[-1].split(end_line)
            self._definition[-1] = "".join([temp_def[0], " = ",
                                            str(self._initial_value), end_line])


class Electrical(Wire):
    """An electrical wire

    A continuous-time net.

    Parameters
    ----------
    name: `str`
        name of the component.
    """

    def __init__(self, name: str):
        discipline = Disciplines(
            'electrical',
            Voltage(),
            Current(),
            domain='continuous'
        )
        super().__init__(name, discipline)
        self.continuous_kernel = True


class Ground(Electrical):
    """A ground net

    Continuous-time ground net.

    Parameters
    ----------
    name: `str`, `optional`
        name of the component, defaults to gnd.
    """

    def __init__(self, name='gnd'):
        super().__init__(name)
        self._definition += ["ground" + " " + self._name + end_line]


class Branch(Wire):
    """A branch net connecting two nets under a commong name.

    Parameters
    ----------
    name: `str`
        name of the branch.
    net_1: :py:class:`cbadc.verilog_ams.nets.Net`
        the first net of the branch.
    net_2: :py:class:`cbadc.verilog_ams.nets.Net`
        the second net of the branch.
    """

    def __init__(self, name: str, net_1: Wire, net_2: Wire):
        self._name = name
        self._discipline = None
        self._definition = [
            "branch" + f" ({str(net_1)}, {str(net_2)}) " +
            self._name + end_line
        ]


class VoltageProbe(Expression):

    def __init__(self, node: Electrical):
        super().__init__(node)
        self.leaves = [
            BaseExpression("V", continuous_kernel=True),
            Parenthesis(self.leaves[0])
        ]


class CurrentProbe(VoltageProbe):

    def __init__(self, node: Electrical):
        super().__init__(node)
        self.leaves = [
            BaseExpression("I", continuous_kernel=True),
            Parenthesis(self.leaves[0])
        ]
