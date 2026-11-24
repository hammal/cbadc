"""Speed and decision hashes of ``Frontend.simulate`` on fixed cases.

Run ``uv run python benchmark/simulate_bench.py`` before and after an engine change:
every case prints its median time per column-step and a hash of the decisions.
"""

import hashlib
import sys
import time

import numpy as np

from cbadc import Frontend, Sinusoid


def plant_batch(N: int, J: int) -> Frontend:
    """``J`` leap-frogs with a 0.1 % gain spread on ``A``: one matrix per column."""
    base, _ = Frontend.leapfrog(N=N, OSR=32, BW=1e6)
    gain = 1 + 1e-3 * np.random.default_rng(0).standard_normal(J)
    af = Frontend(base.A * gain[:, None, None, None], base.B, base.C, dt=base.dt)
    af.input = Sinusoid(0.5, 1e5)
    return af


def single(N: int) -> Frontend:
    af, _ = Frontend.leapfrog(N=N, OSR=32, BW=1e6)
    af.input = Sinusoid(0.5, 1e5)
    return af


def noisy_batch(J: int) -> Frontend:
    af, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
    af.input = Sinusoid(np.linspace(0.1, 0.6, J), 1e5)
    af.state_covariance = 1e-12 * np.eye(3)
    return af


CASES = [
    # name, frontend factory, size, simulate kwargs
    ("J=1 N=3", lambda: single(3), 1 << 20, {}),
    ("J=1 N=8", lambda: single(8), 1 << 20, {}),
    ("J=10000 N=3 plant MC", lambda: plant_batch(3, 10000), 1 << 16, {}),
    ("J=10000 N=8 plant MC", lambda: plant_batch(8, 10000), 1 << 16, {}),
    ("J=1 N=14 BLAS", lambda: single(14), 1 << 18, {}),
    ("J=64 noisy amplitudes", lambda: noisy_batch(64), 1 << 16, {}),
    ("J=11 threads=1 all records", lambda: plant_batch(3, 11), 1 << 16, dict(threads=1, record="all")),
]


def main(only: str = "") -> None:
    print(f"{'case':26s} {'ns/col-step':>12s} {'s':>8s}  sha1(v)")
    for name, make, size, kw in CASES:
        if only and only not in name:
            continue
        af = make()
        af.simulate(256, verbose=False, **kw)  # compile
        times, digest = [], None
        for _ in range(3):
            af = make()  # fresh rng state so noisy cases repeat
            t0 = time.perf_counter()
            v = af.simulate(size, verbose=False, **kw)["v"]
            times.append(time.perf_counter() - t0)
            digest = digest or hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()[:12]
        t = float(np.median(times))
        print(f"{name:26s} {t / (size * af.J) * 1e9:12.2f} {t:8.3f}  {digest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
