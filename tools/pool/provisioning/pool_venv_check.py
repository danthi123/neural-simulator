#!/usr/bin/env python3
"""Verify a pool venv provides exactly what requirements-pool.txt pins, and that each package imports.

Run with the venv's own interpreter, from the revision root (stdlib only, Python 3.10+; tools/pool_venv_deps.sh
pipes this file over ssh so revisions that predate it are checked the same way):

    .venv/bin/python tools/pool/provisioning/pool_venv_check.py .venv/pool-requirements.txt [--write-env FILE]

Checks, for every `name==version` line: the distribution is installed at that version, and every public
top-level module it ships imports. Then imports HARNESS_SUBMODULES (the in-process /api/brain-chat harness,
which needs httpx although starlette does not require it). Prints one JSON object; exit 1 on any failure.
--write-env also writes {distribution: version, python_major_minor} to FILE (the revision's
.pool_environment.json).
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import warnings
from importlib import metadata

# Keep equal to import_closure.SUBMODULE_EXTRA_IMPORTS's keys (tests/test_pool_venv_deps.py pins that).
HARNESS_SUBMODULES = ("starlette.testclient", "fastapi.testclient")

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(path: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.split(" #", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            match = _PIN.match(line)
            if not match or line[match.end():].strip():
                raise SystemExit(f"{path}:{number}: not a plain `name==version` pin: {raw.strip()!r}")
            pins[normalize(match.group(1))] = match.group(2)
    if not pins:
        raise SystemExit(f"{path}: no pins")
    return pins


def top_level_modules(dist: "str | metadata.Distribution") -> list[str]:
    """Public top-level import names a distribution ships, from its top_level.txt AND its RECORD file list.

    Not metadata.packages_distributions(): on Python 3.10 (the pool nodes) it reads top_level.txt only, which
    numpy/scipy/fastapi/pydantic/starlette/httpx wheels do not ship, so it reports them as shipping nothing."""
    if isinstance(dist, str):
        dist = metadata.distribution(dist)
    names: set[str] = set()
    top_level = dist.read_text("top_level.txt") or ""
    names.update(line.strip() for line in top_level.splitlines())
    for path in dist.files or []:
        parts = path.parts
        if not parts or parts[0] in {"..", "__pycache__"} or parts[0].endswith((".dist-info", ".egg-info", ".data")):
            continue
        if len(parts) > 1:
            names.add(parts[0])                         # a package directory
        elif parts[0].endswith(".py"):
            names.add(parts[0][:-3])                    # a single-file module
        elif parts[0].endswith((".so", ".pyd")):
            names.add(parts[0].split(".", 1)[0])        # a top-level extension module
    return sorted(n for n in names if n.isidentifier() and not n.startswith("_"))


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        return 2
    requirements, write_env = argv[0], None
    if len(argv) == 3 and argv[1] == "--write-env":
        write_env = argv[2]
    elif len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    pins = parse_pins(requirements)
    failures: list[str] = []
    versions: dict[str, str] = {}
    for name, wanted in pins.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            failures.append(f"{name}: not installed (want {wanted})")
            continue
        versions[name] = have
        if have != wanted:
            failures.append(f"{name}: installed {have}, pinned {wanted}")
        modules = top_level_modules(name)
        if not modules:
            failures.append(f"{name}: ships no importable top-level module")
        for module in modules:
            try:
                importlib.import_module(module)
            except Exception as exc:  # noqa: BLE001 -- report every failure, not just ImportError
                failures.append(f"{name}: import {module} failed: {type(exc).__name__}: {exc}")
    harness_warnings: list[str] = []
    for module in HARNESS_SUBMODULES:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                importlib.import_module(module)
            harness_warnings.extend(f"{module}: {w.category.__name__}: {w.message}" for w in caught)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"harness: import {module} failed: {type(exc).__name__}: {exc}")
    report = {"ok": not failures, "python": "%s.%s" % sys.version_info[:2], "versions": versions,
              "failures": failures, "warnings": sorted(set(harness_warnings))}
    print(json.dumps(report, sort_keys=True))
    if write_env and not failures:
        env = dict(versions, python_major_minor=report["python"])
        with open(write_env, "w", encoding="utf-8") as handle:
            json.dump(env, handle, sort_keys=True, separators=(",", ":"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
