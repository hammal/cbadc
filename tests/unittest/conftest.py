"""Shared pytest fixtures for the unit suite."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Headless by default. scripts/ already exports MPLBACKEND=Agg, but a bare
# `pytest` invocation does not, and an interactive backend is both slower and a
# hard failure on a runner with no display.
matplotlib.use("Agg", force=False)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close every figure a test leaves open.

    `plt.figure()` registers the figure in a module-level registry that keeps it
    -- and everything plotted into it -- alive for the lifetime of the process.
    The suite creates ~40 figures and closed none of them, several holding
    records of 2**19 samples, so a worker's resident set grew monotonically as
    it went. That is what made the GitLab `unit` job exit 137 (OOMKilled) even
    at two xdist workers.
    """
    yield
    plt.close("all")
