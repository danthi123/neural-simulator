"""Regression test for the review finding (2026-09-25, slotbinder-fast-teach-final fix round, MEDIUM #2):

Before this fix round, `research/runners/_slotbinder_production_gate.py` (its `run_arm` docstring, a code
comment, and the `--sparse-step` CLI help text) and `research/FAILURE_LOG.md`'s 2026-09-25 slotbinder row both
CITED "AMENDMENT 3" of `research/findings/2026-09-24-slotbinder-production-composer-gate-PREREG.md`, but that
prereg had only AMENDMENTS 1-2 -- the citation pointed at nothing. FAILURE_LOG's own parenthetical made it
worse: it named a DIFFERENT document (the sparse-step equivalence finding) as if THAT were where the step model
lived, instead of the prereg amendment. A `--sparse-step` gate or battery run would therefore not have been
registered in any prereg, violating this project's prereg-amendments-before-runs discipline (the same one
`gates/prereg_before_run.py` enforces for run ordering). The step-model artifact the amendment now cites was
also uncommitted.

This test pins: the amendment now exists as a real section in the prereg, the runner's citation is of a
document that actually has it, FAILURE_LOG's row no longer misattributes it to the wrong finding, and the
step-model artifact is committed.
"""
import os
import re

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREREG = os.path.join(_REPO, "research/findings/2026-09-24-slotbinder-production-composer-gate-PREREG.md")
RUNNER = os.path.join(_REPO, "research/runners/_slotbinder_production_gate.py")
FAILURE_LOG = os.path.join(_REPO, "research/FAILURE_LOG.md")
WRONG_DOC = "2026-09-25-slotbinder-event-driven-step-bit-identical-numpy.md"
STEP_MODEL_ARTIFACT = os.path.join(
    _REPO, "research/findings/raw/_slotbinder_sparse_step/step_model_seed7.json")


def _amendment_headings(path):
    return set(re.findall(r"^##\s+(AMENDMENT \d+)", open(path).read(), re.MULTILINE))


def test_prereg_has_an_amendment_3_section():
    headings = _amendment_headings(PREREG)
    assert "AMENDMENT 3" in headings, (
        f"{PREREG} has no '## AMENDMENT 3' heading -- every citation of it elsewhere is dangling")


def test_runner_still_cites_amendment_3_and_the_prereg_now_has_it():
    """The runner cites AMENDMENT 3 inline near the code it governs (not always with the prereg's own path
    alongside it); the load-bearing fact is that the ONE prereg governing this runner actually carries that
    amendment now."""
    src = open(RUNNER).read()
    assert "AMENDMENT 3" in src, "test assumption violated: the runner no longer cites AMENDMENT 3 anywhere"
    assert "AMENDMENT 3" in _amendment_headings(PREREG)


def test_failure_log_slotbinder_row_does_not_misattribute_amendment_3():
    """The 2026-09-25 slotbinder row must not cite the sparse-step equivalence finding as the home of
    AMENDMENT 3 -- that document has no AMENDMENT section; the prereg does."""
    log = open(FAILURE_LOG).read()
    rows = [ln for ln in log.splitlines() if "AMENDMENT 3" in ln and "slotbinder" in ln.lower()]
    assert rows, "test assumption violated: no FAILURE_LOG row cites AMENDMENT 3 for the slotbinder lane"
    for row in rows:
        assert WRONG_DOC not in row, (
            f"FAILURE_LOG row cites AMENDMENT 3 but also names {WRONG_DOC!r}, a document with no amendments -- "
            "this must cite the PREREG instead")
        assert os.path.basename(PREREG) in row, (
            "FAILURE_LOG's AMENDMENT 3 citation must name the prereg document that actually has that section")


def test_step_model_artifact_is_committed():
    assert os.path.exists(STEP_MODEL_ARTIFACT), (
        f"{STEP_MODEL_ARTIFACT} is cited by AMENDMENT 3 but is missing from the tree")
