#!/usr/bin/env python3
"""Print the test files belonging to one shard.

Usage: ``_shard_tests.py i/N``  (i is 1-based, as GitLab's CI_NODE_INDEX is)

Files are packed into ``N`` bins longest-first, greedily onto whichever bin is
currently lightest. That is the standard LPT heuristic and it is within 4/3 of
optimal, which is far better than any static grouping and needs no plugin.

Durations come from ``tests/durations.json`` when it is present; regenerate it
with ``--update`` after a full run has written ``pytest-unittests.xml``. Missing
or unknown files fall back to the mean, so the split stays valid as tests are
added -- it just balances less well until the file is refreshed.

Sharding is at *file* granularity deliberately. The heaviest module memoises 64
stiff ODE solves across its 82 tests; splitting inside it would make every shard
recompute them and throw the memoisation away.
"""

from __future__ import annotations

import json
import pathlib
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_ROOT = ROOT / "tests" / "unittest"
DURATIONS = ROOT / "tests" / "durations.json"


def _discover() -> list[str]:
    return sorted(
        str(p.relative_to(ROOT)) for p in TEST_ROOT.rglob("*_test.py")
    )


def _update_durations() -> int:
    report = ROOT / "pytest-unittests.xml"
    if not report.exists():
        print(f"{report} not found; run the suite first", file=sys.stderr)
        return 1
    per: dict[str, float] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        # pytest's JUnit writer omits @file; derive the path from @classname,
        # which is the dotted module path.
        f = case.get("file")
        if not f:
            cls = case.get("classname", "")
            if not cls:
                continue
            f = cls.replace(".", "/") + ".py"
            # classname drops the leading directories pytest rooted at
            if not (ROOT / f).exists():
                cand = list(TEST_ROOT.rglob(pathlib.Path(f).name))
                if not cand:
                    continue
                f = str(cand[0].relative_to(ROOT))
        per[f] = per.get(f, 0.0) + float(case.get("time", 0.0))
    DURATIONS.write_text(json.dumps(dict(sorted(per.items())), indent=2) + "\n")
    print(f"wrote {DURATIONS} ({len(per)} files)", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--update":
        return _update_durations()
    if len(argv) != 2 or "/" not in argv[1]:
        print("usage: _shard_tests.py i/N  |  --update", file=sys.stderr)
        return 2

    index_s, total_s = argv[1].split("/", 1)
    index, total = int(index_s), int(total_s)
    if not (1 <= index <= total):
        print(f"shard {index}/{total} is out of range", file=sys.stderr)
        return 2

    files = _discover()
    if not files:
        print("no test files found", file=sys.stderr)
        return 1
    if total == 1:
        print(" ".join(files))
        return 0

    known: dict[str, float] = {}
    if DURATIONS.exists():
        known = json.loads(DURATIONS.read_text())
    default = (sum(known.values()) / len(known)) if known else 1.0

    # Longest-processing-time first: sort descending, drop each onto the
    # lightest bin. Ties broken by name so every runner computes the same split.
    weighted = sorted(files, key=lambda f: (-known.get(f, default), f))
    bins: list[list[str]] = [[] for _ in range(total)]
    loads = [0.0] * total
    for f in weighted:
        i = loads.index(min(loads))
        bins[i].append(f)
        loads[i] += known.get(f, default)

    print(" ".join(sorted(bins[index - 1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
