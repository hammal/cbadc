"""Verilog am/s nature and disciple classes.
"""

from typing import Union


class Nature():

    def __init__(self, units: str, access: str, abstol: str, idt_nature: str) -> None:
        self.units = str(units)
        self.access = str(access)
        self.abstol = str(abstol)
        self.idt_nature = str(idt_nature)


class Voltage(Nature):

    def __init__(self, abstol: str = str(1e-6)) -> None:
        units = "V"
        access = "V"
        idt_nature = "Flux"
        self._abstol_override = "`VOLTAGE_ABSTOL"
        super().__init__(units, access, abstol, idt_nature)


class Current(Nature):

    def __init__(self, abstol: str = str(1e-12)) -> None:
        units = "A"
        access = "I"
        idt_nature = "Charge"
        self._abstol_override = "`CURRENT_ABSTOL"
        super().__init__(units, access, abstol, idt_nature)


class Disciplines():

    def __init__(self, name: str, potential: Union[Nature, None], flow: Union[Nature, None], domain: str) -> None:
        self.__domains = set(('discrete', 'continuous'))
        self.__discipline_names = set(
            ('logic', 'electrical', 'voltage', 'current'))

        if not (name in self.__discipline_names):
            raise BaseException(
                f"Non-valid discipline name {name} should be from set of {self.__discipline_names}.")
        if not (domain in self.__domains):
            raise BaseException(
                f"Non-valid domain name {domain} should be from set of {self.__domains}."
            )
        self.name = str(name)
        self.potential = potential
        self.flow = flow
        self.domain = str(domain)
