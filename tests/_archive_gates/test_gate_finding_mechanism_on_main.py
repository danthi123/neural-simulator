"""tests for tools/gates/finding_mechanism_on_main.py (CLASS FM).

Imports and exercises the REAL gate module the pre-commit registry calls -- same convention as
tests/test_seam_contracts.py (`import tools.gates.X as X_gate`), so these tests cannot drift from what the
hook actually runs. The module's own `selftest()` is the registry's own trust mechanism (a gate whose
selftest does not fail in the failing direction is treated as BROKEN); this file adds pytest-level coverage
on top so a regression here shows up as a normal test failure, not only at commit time.
"""
from __future__ import annotations

import os
import tempfile

import pytest

import tools.gates.finding_mechanism_on_main as fm_gate


# ---------------------------------------------------------------------------------------------------------
# the module's own selftest is the registry's trust mechanism -- pin it so a regression fails a normal run
# ---------------------------------------------------------------------------------------------------------
def test_registry_selftest_passes():
    problems = fm_gate.selftest()
    assert problems == [], "gate selftest reported problems: %r" % problems


def test_registry_discovers_this_gate_with_the_expected_contract():
    from tools.gates import discover
    hits = [t for t in discover() if t[0] == fm_gate.NAME]
    assert len(hits) == 1, "gates/__init__.discover() did not find exactly one %r module" % fm_gate.NAME
    name, mod, err = hits[0]
    assert err is None, "the registry reports this gate as broken: %s" % err
    assert mod.CLASS_ID == "FM"
    assert mod.BLOCKING is True


# ---------------------------------------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture()
def repo(tmp_path):
    return str(tmp_path)


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


GO_LIVE = "---\nstatus: live\nverdict: GO\n---\n\n# a GO finding\n\n"


# ---------------------------------------------------------------------------------------------------------
# THE FAILING DIRECTION FIRST: the actual defect this gate closes
# ---------------------------------------------------------------------------------------------------------
def test_a_go_finding_citing_a_flag_with_no_code_reference_is_blocked(repo):
    f = _write(repo, "research/findings/x.md", GO_LIVE + "Run with `BRAIN_TOTALLY_UNMERGED_FLAG=1`.\n")
    problems = fm_gate.check([f], root=repo)
    assert len(problems) == 1
    assert "BRAIN_TOTALLY_UNMERGED_FLAG" in problems[0]
    assert "x.md:" in problems[0].replace(os.sep, "/") or f in problems[0]


def test_an_assignment_form_citation_with_no_code_reference_is_also_blocked(repo):
    f = _write(repo, "research/findings/x.md", GO_LIVE + "The flag LB_TOTALLY_UNMERGED_PROBE=1 was set.\n")
    problems = fm_gate.check([f], root=repo)
    assert any("LB_TOTALLY_UNMERGED_PROBE" in p for p in problems)


def test_two_missing_flags_on_one_finding_both_fire(repo):
    f = _write(repo, "research/findings/x.md",
              GO_LIVE + "Uses `BRAIN_MISSING_ONE=1` and `LB_MISSING_TWO=1`.\n")
    problems = fm_gate.check([f], root=repo)
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "BRAIN_MISSING_ONE" in joined and "LB_MISSING_TWO" in joined


# ---------------------------------------------------------------------------------------------------------
# calibration: cases that must stay silent
# ---------------------------------------------------------------------------------------------------------
def test_a_flag_referenced_in_tracked_code_passes(repo):
    _write(repo, "sim/bridge.py", 'if os.environ.get("BRAIN_PRESENT") == "1":\n    pass\n')
    f = _write(repo, "research/findings/x.md", GO_LIVE + "Set `BRAIN_PRESENT=1`.\n")
    assert fm_gate.check([f], root=repo) == []


def test_a_flag_referenced_only_via_backtick_mention_in_code_passes(repo):
    _write(repo, "webapp/server.py", 'FLAG = os.environ.get("BRAIN_WIRED", "0") == "1"\n')
    f = _write(repo, "research/findings/x.md", GO_LIVE + "See `BRAIN_WIRED` for the wiring.\n")
    assert fm_gate.check([f], root=repo) == []


def test_a_flag_referenced_in_a_research_runner_passes(repo):
    _write(repo, "research/runners/_some_runner.py", 'X = os.environ.get("LB_SOME_PROBE")\n')
    f = _write(repo, "research/findings/x.md", GO_LIVE + "`LB_SOME_PROBE=1`\n")
    assert fm_gate.check([f], root=repo) == []


def test_a_flag_referenced_only_in_a_finding_not_in_code_still_fires(repo):
    """A mention in ANOTHER finding is not code -- research/findings/ is not one of the scanned dirs."""
    _write(repo, "research/findings/unrelated.md", "some other doc mentioning `BRAIN_MENTIONED_ELSEWHERE` too\n")
    f = _write(repo, "research/findings/x.md", GO_LIVE + "`BRAIN_MENTIONED_ELSEWHERE=1`\n")
    problems = fm_gate.check([f], root=repo)
    assert any("BRAIN_MENTIONED_ELSEWHERE" in p for p in problems)


def test_the_per_line_escape_clears_a_citation(repo):
    f = _write(repo, "research/findings/x.md",
              GO_LIVE + "Uses `BRAIN_DELIBERATELY_UNMERGED=1` "
                        "<!--flag-not-on-main: pre-registration for branch research/foo-->\n")
    assert fm_gate.check([f], root=repo) == []


def test_the_escape_is_per_line_not_per_file(repo):
    f = _write(repo, "research/findings/x.md",
              GO_LIVE + "Uses `BRAIN_ESCAPED=1` <!--flag-not-on-main: pre-registration-->\n"
                        "Also uses `BRAIN_NOT_ESCAPED=1` on a different line.\n")
    problems = fm_gate.check([f], root=repo)
    assert not any("BRAIN_ESCAPED" in p and "BRAIN_NOT_ESCAPED" not in p for p in problems)
    assert any("BRAIN_NOT_ESCAPED" in p for p in problems)
    assert not any("BRAIN_ESCAPED=" in p for p in problems)  # the escaped one never appears at all


@pytest.mark.parametrize("status", ["superseded", "retracted", "corrected", "void"])
def test_a_non_live_status_is_out_of_scope(repo, status):
    f = _write(repo, "research/findings/x.md",
              "---\nstatus: %s\nverdict: GO\n---\n\n# was a go\n\n`BRAIN_MISSING=1`\n" % status)
    assert fm_gate.check([f], root=repo) == []


def test_a_no_go_verdict_is_out_of_scope(repo):
    f = _write(repo, "research/findings/x.md",
              "---\nstatus: live\nverdict: NO-GO\n---\n\n# a negative\n\n`BRAIN_MISSING=1`\n")
    assert fm_gate.check([f], root=repo) == []


def test_a_hyphenated_go_verdict_is_in_scope(repo):
    """Real corpus shape: `verdict: SMOKE-GO (...)`, `verdict: WIRED-GO (...)` -- GO joined by a hyphen, not
    a standalone word. Must still be recognised as a positive verdict, distinct from NO-GO."""
    f = _write(repo, "research/findings/x.md",
              "---\nstatus: live\nverdict: WIRED-GO (default-off; reachable)\n---\n\n# ok\n\n`BRAIN_MISSING=1`\n")
    assert fm_gate.check([f], root=repo) != []


def test_a_finding_with_no_frontmatter_is_out_of_scope(repo):
    f = _write(repo, "research/findings/x.md", "# a legacy finding\n\n`BRAIN_MISSING=1`\n")
    assert fm_gate.check([f], root=repo) == []


def test_raw_artifact_paths_are_never_treated_as_findings(repo):
    f = _write(repo, "research/findings/raw/x.json", "{}")
    assert fm_gate.check([f], root=repo) == []


def test_empty_paths_list_returns_immediately(repo):
    assert fm_gate.check([], root=repo) == []


def test_a_glob_family_citation_is_not_treated_as_one_literal_flag(repo):
    """Real corpus shape: "...neither `BRAIN_GNW_STOP_TRIGGER_*` flag..." -- a wildcard family reference,
    not a claim about one specific flag with a code reference to check."""
    f = _write(repo, "research/findings/x.md", GO_LIVE + "Neither `BRAIN_GLOB_FAMILY_*` flag fires.\n")
    assert fm_gate.check([f], root=repo) == []


def test_standalone_audit_mode_scans_the_whole_corpus(repo):
    _write(repo, "research/findings/clean.md", GO_LIVE + "no flags here\n")
    _write(repo, "research/findings/dirty.md", GO_LIVE + "`BRAIN_AUDIT_MODE_MISSING=1`\n")
    problems = fm_gate.check(None, root=repo)
    assert any("BRAIN_AUDIT_MODE_MISSING" in p for p in problems)


# ---------------------------------------------------------------------------------------------------------
# the real corpus: this gate must not be BROKEN (crash) over the live findings directory, and the one
# real 2026-09-23 hit (the motivating case) must now read CLEAN since d69e7ddb merged the flag.
# ---------------------------------------------------------------------------------------------------------
def test_the_motivating_finding_is_clean_on_this_tree():
    rel = ("research/findings/"
          "2026-09-23-episodic-s100-cupy-repeats-7of7-loadbearing-robust-core-23.md")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = os.path.join(root, rel)
    if not os.path.exists(full):
        pytest.skip("motivating finding not present on this checkout")
    assert fm_gate.check([rel]) == [], (
        "the finding that motivated this gate should be CLEAN now that BRAIN_EPISODIC_STORE_VERIFY / "
        "LB_EPISODIC_DRIVE_PROBE are on main (merge d69e7ddb) -- if this fails, either the flag reference "
        "regressed or the gate's own scope/extraction logic has a new false positive")
