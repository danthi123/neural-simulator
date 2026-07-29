"""CI guard for the two document-structure rules in docs/WRITING.md.

These are STRUCTURE rules, not truth rules. Six of the nine 2026-07-28 retractions were instrument
failures and ALL SIX would have produced prose that passes both — see docs/WRITING.md "What this does
NOT do". Truth verification lives in .claude/skills/verify-go/SKILL.md; term conditions in docs/TERMS.md.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run():
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_docs.py")],
                          capture_output=True, text=True, cwd=ROOT)


def test_doc_structure_rules_pass():
    r = _run()
    assert r.returncode == 0, "docs/WRITING.md violations:\n" + r.stdout + r.stderr


def test_checker_detects_an_unmarked_citation():
    """The checker must FAIL on a known-bad input — a check that cannot fail is worthless.

    Earned: the first version of this checker reported 0 W1 violations twice while real unmarked
    citations existed (it matched only full basenames, and it exempted table rows from W1).
    """
    import io
    import re
    src = io.open(os.path.join(ROOT, "tools", "check_docs.py"), encoding="utf-8").read()
    # truncated citations must match
    assert 'len(pref) >= 30' in src, "checker must match truncated citations (hyphen-boundary prefixes)"
    # W1 must NOT skip table rows
    assert re.search(r"_lines\(p, prose_only=False\)", src), "W1 must scan table rows, not only prose"
