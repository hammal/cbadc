"""The ngspice bench for circuits emitted by ``GmC.netlist``.

The tests cover binary comparator loops, constant DAC windows, noise,
single-frontends selected from a batch, and Leapfrog models. ``bench.py``
compares clocked decisions and states with cbadc's simulation.

Skipped as a whole when ``ngspice`` is not on ``PATH``. Not part of the unit
suite (``testpaths`` is ``tests/unittest``); run with ``make test-ngspice`` or
``scripts/test-performance ngspice``.
"""

import shutil

import pytest

HAS_NGSPICE = shutil.which("ngspice") is not None


def pytest_configure(config):
    config.addinivalue_line("markers", "ngspice: needs the ngspice binary")


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="ngspice is not on PATH")
    for item in items:
        item.add_marker(pytest.mark.ngspice)
        if not HAS_NGSPICE:
            item.add_marker(skip)
