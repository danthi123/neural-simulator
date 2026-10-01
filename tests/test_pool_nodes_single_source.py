"""Guards the pool-node single source of truth (2026-09-30: pool40 retired indefinitely, tools/pool_nodes.conf
+ tools/pool_nodes.sh + tools/pool_nodes.py became the ONE place the default mini-PC pool node list lives).

WHY THIS EXISTS. Before this date ~20 scripts each hardcoded their own copy of "pool40 pool41 pool42", so
retiring pool40 meant hunting down every copy by hand -- and the next retirement/addition would hit the same
problem again. This test does not just check today's fix landed; it fails FUTURE drift too: any executable
(non-comment) line added under tools/ that hardcodes two-or-more "poolNN"-shaped hostnames in a row, instead of
reading tools/pool_nodes.conf (via tools/pool_nodes.sh or tools/pool_nodes.py), trips it.

A check whose own detector cannot be shown to fire is not trusted (project convention, see the verify-go
skill) -- test_detector_catches_a_hardcoded_list_and_ignores_legitimate_lines exercises the SAME regex the
repo-wide scan uses, both directions, before the repo-wide test relies on it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"

sys.path.insert(0, str(ROOT / "tools"))
import pool_nodes  # noqa: E402 -- single source of truth: tools/pool_nodes.conf

# The single-source-of-truth files themselves are exempt: pool_nodes.conf legitimately HOLDS the literal list
# (that is its entire job), pool_nodes.sh/pool_nodes.py legitimately quote it once in their own docstrings/
# comments as documentation of what they replaced.
EXEMPT_NAMES = {"pool_nodes.conf", "pool_nodes.sh", "pool_nodes.py"}

# Two or more "poolNN" hostnames back-to-back (whitespace- or comma-separated) is the shape of a hardcoded
# default node LIST (`NODES=(pool40 pool41 pool42)`, `"pool40 pool41 pool42"`, `["pool40", "pool41", "pool42"]`,
# `POOL_NODES = ("pool40", "pool41")`, ...). A single "poolNN" mention (an explicit CLI arg, a fixture, an AWS
# example alias like pool1) is NOT what this guards against -- those are legitimate and common.
NODE_LIST_RE = re.compile(r"pool\d+(?:\s*,\s*|\s+)pool\d+(?:(?:\s*,\s*|\s+)pool\d+)*")


def _offending_lines(tools_dir: Path) -> list[str]:
    """Every non-comment .sh/.py line under tools_dir that hardcodes a poolNN node list, as
    "relative/path:lineno: text" strings. Comments are stripped (whole-line '#...' and trailing '# ...')
    before matching, so a historical/explanatory mention of "pool40 pool41 pool42" in prose does not trip it --
    only code that would actually run for that literal list would.
    """
    offenders: list[str] = []
    for path in sorted(tools_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".sh", ".py"):
            continue
        if path.name in EXEMPT_NAMES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            hash_idx = raw.find("#")
            code = raw[:hash_idx] if hash_idx != -1 else raw
            if NODE_LIST_RE.search(code):
                try:
                    rel = path.relative_to(ROOT)
                except ValueError:
                    rel = path
                offenders.append(f"{rel}:{lineno}: {raw.strip()}")
    return offenders


def test_detector_catches_a_hardcoded_list_and_ignores_legitimate_lines(tmp_path):
    """The detector's own teeth, proven both directions before the repo-wide scan trusts it."""
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "offender.sh").write_text(
        "#!/usr/bin/env bash\n"
        "NODES=(pool40 pool41 pool42)\n"
    )
    assert _offending_lines(bad_dir), "the detector missed an obvious hardcoded 3-node list"

    good_dir = tmp_path / "good"
    good_dir.mkdir()
    (good_dir / "clean.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# historical note: this used to be pool40 pool41 pool42, now read from the conf file\n"
        'source "$ROOT/tools/pool_nodes.sh"\n'
        'NODES="${POOL_NODES:-$POOL_NODES_DEFAULT}"\n'
        'ssh -f -n pool41 "true"   # a single explicit node is fine\n'
    )
    assert not _offending_lines(good_dir), (
        "the detector false-positived on a comment-only mention and on legitimate single-node / "
        "conf-reading lines: " + repr(_offending_lines(good_dir))
    )


def test_no_tool_hardcodes_the_pool_node_list_instead_of_reading_the_conf_file():
    offenders = _offending_lines(TOOLS_DIR)
    assert not offenders, (
        "the following line(s) under tools/ hardcode a pool node list instead of sourcing "
        "tools/pool_nodes.conf (via tools/pool_nodes.sh or tools/pool_nodes.py) -- this is exactly the "
        "20-scripts-to-update problem pool40's 2026-09-30 retirement hit; fix the script to read the default "
        "from the conf file instead:\n  " + "\n  ".join(offenders)
    )


def test_conf_file_default_is_pool41_pool42_only():
    """Pins the CURRENT default (pool40 retired 2026-09-30) so a future edit to the conf file is a deliberate,
    reviewed change, not a silent drift -- update this alongside tools/pool_nodes.conf when a node is
    added/retired."""
    assert pool_nodes.default_nodes() == ["pool41", "pool42"]
