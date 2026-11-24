"""The optional delta-sigma bridge must not block core imports."""

import subprocess
import sys


def _check(code):
    """Run each import check in a fresh interpreter, isolated from pytest imports."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_delsig_loads_on_demand():
    """Core import leaves the optional bridge unloaded until requested."""
    _check(
        "import cbadc, sys; "
        "assert 'cbadc.delsig' not in sys.modules; "
        "assert 'cbadc_deltasigma' not in sys.modules; "
        "from cbadc import delsig; "
        "import cbadc_deltasigma; "
        "assert delsig is cbadc.delsig; "
        "assert delsig.ds is cbadc_deltasigma; "
        "assert 'deltasigma' not in sys.modules"
    )


def test_missing_cbadc_deltasigma_explains_how_to_install():
    """A missing optional dependency must report the ``cbadc[delsig]`` extra."""
    _check(
        "import builtins, cbadc\n"
        "original = builtins.__import__\n"
        "def block(name, *args, **kwargs):\n"
        "    if name == 'cbadc_deltasigma':\n"
        "        raise ModuleNotFoundError(name='cbadc_deltasigma')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = block\n"
        "try:\n"
        "    from cbadc import delsig\n"
        "except ImportError as exc:\n"
        '    assert "cbadc[delsig]" in str(exc)\n'
        '    assert "cbadc-deltasigma" in str(exc)\n'
        "else:\n"
        "    raise AssertionError('bridge import unexpectedly succeeded')\n"
    )
