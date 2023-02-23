from typing import List

from cbadc.circuit import Terminal


def ngspice_vector_terminal_vector_scalar(terminals: List[str]) -> List[str]:
    inputs = ' '.join([f'{terminal}' for terminal in terminals[:-1]])
    return [f'[{inputs}]', terminals[-1]]


def ngspice_vector_terminal_vector_vector(terminals: List[str]) -> List[str]:
    inputs = ' '.join([f'{terminal}' for terminal in terminals[:-1]])
    return [f'[{inputs}]', f'[{terminals[-1]}]']
