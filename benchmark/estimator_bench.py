"""Time the estimator recursion for shared and per-member coefficient matrices.

Run ``python benchmark/estimator_bench.py --baseline /tmp/estimator-before.py``
after saving the old module there. Without --baseline, time the current module.
The synthetic stable models use K=2, M=N, L=1 and int8 decisions. Timings include
coefficient preparation and output clearing, exclude synthesis and compilation,
and reuse the same output buffer to avoid allocator/alignment differences.
Report medians of five interleaved runs. Both directions must agree bitwise
with the baseline before a case is timed. Use --threads to control Numba workers.
"""

import argparse
import importlib.util
import sys
from time import perf_counter

import numba
import numpy as np

import cbadc.estimator as current


def main():
    """Run the fixed (N, J, log2(size)) workload matrix and print speedups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    modules = [current]
    if args.baseline:
        spec = importlib.util.spec_from_file_location(
            "cbadc._estimator_baseline", args.baseline
        )
        before = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = before
        spec.loader.exec_module(before)
        modules.insert(0, before)

    print(f"Numba {numba.__version__}, {numba.get_num_threads()} workers", flush=True)
    print(
        " N     J    size matrices       before ms   current ms   speedup", flush=True
    )
    cases = [
        (3, 1, 12),
        (7, 1, 18),
        (7, 4, 14),
        (7, 16, 16),
        (7, 64, 12),
        (7, 64, 16),
        (7, 256, 14),
        (12, 256, 14),
        (3, 1000, 14),
        (12, 64, 18),
    ]
    for N, J, power in cases:
        for shared in (True, False):
            if J == 1 and not shared:
                continue
            rng = np.random.default_rng(10)
            J_m = 1 if shared else J
            A = rng.normal(size=(J_m, 2, N, N))
            A *= 0.85 / np.maximum(np.linalg.norm(A, axis=(-2, -1), keepdims=True), 1)
            B = rng.normal(scale=0.1, size=(J_m, 2, N, N))
            W = rng.normal(size=(J_m, 2, 1, N))
            v = rng.integers(0, 2, size=(1 << power, N, J), dtype=np.int8) * 2 - 1
            out = np.empty((len(v), 1, J))

            def run(module, decisions, forward=True):
                target = out[: len(decisions)]
                target.fill(0)
                module._message(A, B, W, decisions, target, forward)
                return target

            for module in modules:
                run(module, v[:32])  # Compile each layout before timing.
            if len(modules) == 2:
                for forward in (True, False):
                    np.testing.assert_array_equal(
                        run(modules[0], v, forward).copy(), run(current, v, forward)
                    )
            times = {module: [] for module in modules}
            for repeat in range(5):
                for module in modules if repeat % 2 == 0 else modules[::-1]:
                    start = perf_counter()
                    run(module, v)
                    times[module].append(perf_counter() - start)
            old, new = np.median(times[modules[0]]), np.median(times[current])
            kind = "shared" if shared else "per member"
            print(
                f"{N:2d} {J:5d} {len(v):7d} {kind:10s} {old * 1e3:12.3f} {new * 1e3:12.3f} {old / new:8.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
