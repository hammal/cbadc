"""Numbers classes.

The verilog-ams number types. See `this page <https://verilogams.com/refman/basics/numbers.html>`_ for further information.
"""
from typing import Union
from .keywords import raise_exception_for_keyword
import logging

logger = logging.getLogger(__name__)


class Number():

    def __init__(self, value):
        self._value = str(value)
        raise_exception_for_keyword(self._value)

    def __str__(self):
        return self._value


class Logic(Number):

    def __init__(self, value: str):
        self._logic_values = ['0', '1', 'x', 'z']
        if value not in self._logic_values:
            raise BaseException(f"{value} not in {self._logic_values}")
        super().__init__(value)


class Integer(Number):

    def __init__(self, value: int):
        if not isinstance(value, int):
            logger.warning(f"{value} not int")
        super(Integer, self).__init__(int(value))


class Real(Number):

    def __init__(self, value: float):
        super().__init__(float(value))


any_number = Union[Number, Logic, Integer, Real]
