# Capability: Lifelong learning
# Record: brain/research/2026-11-22-fresh-test-split-design.md
# Expected-fail: none (test helper: runs the pre-change code from git for flag-off byte-identity checks)
"""Test helper: export brain/ at the pre-change commit (BASE, the design commit before the TEST-v2/age-ladder code) into
a temp dir and run a small driver there and in the working tree, each in its own subprocess, so old and new code never
share a module object. LIFE_BASE_COMMIT overrides BASE; the tests skip when git or the commit is unavailable."""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

BASE = os.environ.get("LIFE_BASE_COMMIT", "4c9c0504f")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export_base(dst: str) -> str:
    """git archive BASE brain | tar -x into dst; -> dst (the PYTHONPATH root of the old code)."""
    os.makedirs(dst, exist_ok=True)
    try:
        arc = subprocess.run(["git", "-C", REPO, "archive", BASE, "brain"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", dst], input=arc, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"cannot export the pre-change tree {BASE}: {e}")
    return dst


def run_driver(root: str, code: str, args: list, timeout: int = 1800) -> str:
    """Run `code` (a Python script) with PYTHONPATH=root and cwd=root; -> stdout. Fails the test on a non-zero exit."""
    env = {**os.environ, "PYTHONPATH": root, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    p = subprocess.run([sys.executable, "-c", code, *args], cwd=root, env=env, capture_output=True, text=True,
                       timeout=timeout)
    assert p.returncode == 0, p.stderr[-4000:]
    return p.stdout


TIMING_KEYS = ("seconds", "live_seconds", "age_ladder_seconds")
RECORD_ONLY_KEYS = ("provenance",)   # life.provenance(): stack record (prereg test2 T1), absent from the pre-change code


def scrub(x):
    """A life result without wall-clock timings (the only fields that differ between two identical runs)."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if k not in TIMING_KEYS + RECORD_ONLY_KEYS}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return x


def scrub_md(text: str) -> str:
    """The .md without its wall-clock numbers ("<n> s" after a night or probe)."""
    return re.sub(r"(?<![\w.])\d+(?:\.\d+)? s(?![A-Za-z])", "<t> s", text)
