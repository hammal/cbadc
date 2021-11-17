"""Verilog am/s variable classes.

The verilog-ams number types. See `this page <https://verilogams.com/refman/basics/variables.html>`_ for further information.
"""
import re
from cbadc.verilog_ams.expressions import Expression
from cbadc.verilog_ams.keywords import raise_exception_for_keyword
from .numbers import Integer as _Integer
from .numbers import Real as _Real


class _Variables(Expression):
    """A variable

    Parameters
    ----------
    name: `str`
        the name of the component.

    """

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        if bool(re.search(r"[\s\t]", self._name)):
            raise BaseException(
                f"invalid name: name can't have whitespace characters within {name}")
        self._definition = []
        raise_exception_for_keyword(self._name)

    def definition(self) -> list[str]:
        """Generates a list of strings containing
        the definitions of the component.

        Returns
        -------
        : `list`[`str`]
            returns a list containing the definitions of the net.
        """
        return self._definition

    def name(self):
        """Returns the name of the component

        Returns
        -------
        : `str`
            name of component.
        """
        return self._name

    def __str__(self):
        return self.name()


class Integer(_Variables):
    """A integer variable

    Note that these can only be used
    in the discrete-time domain.

    Parameters
    ----------
    name: `str`
        name of the integer node.
    initial_value: `int`
        initial value, defaults to 0.

    """

    def __init__(self, name: str, initial_value=0):
        super().__init__(name)
        self.initial_value = _Integer(initial_value)
        self._definition = [f"integer {self} = {self.initial_value};"]


class Real(_Variables):
    """A real variable

    Note that these can be used
    in both the continuous-time and
    the discrete-time setting.

    Parameters
    ----------
    name: `str`
        name of the integer node.
    initial_value: `int`
        initial value, defaults to 0.
    continuous_time: `bool`
        determines if this is a continuous-time
        variable, defaults to True.

    """

    def __init__(self, name: str, initial_value=0, continuous_time=True):
        super().__init__(name)
        self.initial_value = _Real(initial_value)
        self.continuous_kernel = continuous_time
        if self.continuous_kernel:
            self._definition = [f"real {self};"]
        else:
            self._definition = [f"real {self} = {self.initial_value};"]
