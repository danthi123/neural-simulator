"""CLASS COV — a NOTICED failure mode that never became a gate.

WHY (owner, 2026-07-31: "are we automatically updating/adding to the gates once new failure modes are detected?").
The honest answer was NO. Every gate in this registry was added because I happened to notice a failure and
happened to act on it. That is the same dependency on memory the whole system exists to remove, one level up.

Noticing that something is a NEW CLASS is judgement and cannot be automated. CLOSING it can be:

    notice -> add one line to research/FAILURE_LOG.md -> this gate BLOCKS until that line names a gate,
    or explicitly declares NOT-GATEABLE with a reason.

It also checks the reverse direction, because a spec that drifts from the code is how docs/RETRACTED.md ended up
with one row against 21 retracted findings:
  * every class in docs/FAILURE_GATE_MATRIX.md that names a `gates/<module>` must have that module present;
  * every module in tools/gates/ must appear in the matrix.

WHAT IT CANNOT CATCH: a failure nobody wrote down. If the line is never added, nothing fires — the log is the
human input this gate operates on. It closes the "noticed but forgotten" hole, not the "never noticed" one.
"""
from __future__ import annotations

import glob
import os
import re
import tempfile

NAME = "coverage"
CLASS_ID = "COV"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOG = os.path.join(_ROOT, "research", "FAILURE_LOG.md")
_MATRIX = os.path.join(_ROOT, "docs", "FAILURE_GATE_MATRIX.md")


def _log_rows(text):
    rows = []
    for ln in text.split("\n"):
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # Require a REAL date in column 1. The first version excluded only the literal "date", so a header
        # like "| d | f | g |" parsed as DATA and its gate cell "g" resolved to nothing -- a false positive
        # caught by this gate's own selftest before it ever ran on the repo.
        if len(cells) < 3 or not re.match(r"^\d{4}-\d{2}-\d{2}$", cells[0]):
            continue
        rows.append(cells)
    return rows


def _check_text(log_text, modules):
    problems = []
    for cells in _log_rows(log_text):
        date, failure, gate = cells[0], cells[1], cells[2]
        if not gate:
            problems.append("FAILURE_LOG %s: '%s' names NO gate. Add one, or declare "
                            "'NOT-GATEABLE: <reason>'." % (date, failure[:70]))
            continue
        gate_clean = gate.strip('`').strip()
        if gate_clean.upper().startswith("NOT-GATEABLE"):
            if len(gate_clean) < len("NOT-GATEABLE:") + 15:
                problems.append("FAILURE_LOG %s: NOT-GATEABLE with no real reason given." % date)
            continue
        named = re.findall(r"`?([a-z_]+)`?", gate)
        if modules and not any(m in modules for m in named):
            # a gate elsewhere (hook, dispatcher, queue) is legitimate -- only flag when nothing resolves at all
            if not re.search(r"dispatcher|hook|queue|pre-commit|heartbeat|experiment|claim|biology|lab|tools/[\w.-]+\.(?:sh|py)", gate, re.I):
                problems.append("FAILURE_LOG %s: gate %r resolves to no module in tools/gates/ and names no "
                                "other enforcement point." % (date, gate[:50]))
    return problems


def check(paths):
    if paths is not None and len(paths) == 0:
        return []
    problems = []
    modules = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(_ROOT, "tools/gates/*.py"))
               if not p.endswith("__init__.py")}
    if not os.path.exists(_LOG):
        return ["research/FAILURE_LOG.md is missing — the noticed-failure log is where closure is enforced."]
    problems += _check_text(open(_LOG, errors="ignore").read(), modules)

    if os.path.exists(_MATRIX):
        mtx = open(_MATRIX, errors="ignore").read()
        for m in re.findall(r"`gates/([a-z_]+)`", mtx):
            if m == "__init__":
                continue
            if m not in modules:
                problems.append("MATRIX names `gates/%s` but tools/gates/%s.py does not exist." % (m, m))
        for m in sorted(modules):
            if m not in mtx:
                problems.append("tools/gates/%s.py exists but is absent from docs/FAILURE_GATE_MATRIX.md — the "
                                "spec and the code have drifted." % m)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: a logged failure with no gate MUST be caught."""
    bad = []
    mods = {"doc_type", "single_seed"}
    if not _check_text("| date | failure | gate |\n|---|---|---|\n| 2026-01-01 | something broke |  |\n", mods):
        bad.append("did NOT catch a logged failure with an EMPTY gate column")
    if not _check_text("| d | f | g |\n|---|---|---|\n| 2026-01-01 | x | NOT-GATEABLE: |\n", mods):
        bad.append("did NOT catch a NOT-GATEABLE with no reason")
    if _check_text("| d | f | g |\n|---|---|---|\n| 2026-01-01 | x | `doc_type` |\n", mods):
        bad.append("FALSE POSITIVE: flagged a row naming a real module")
    if _check_text("| d | f | g |\n|---|---|---|\n| 2026-01-01 | x | dispatcher exit-status |\n", mods):
        bad.append("FALSE POSITIVE: flagged a row naming a non-module enforcement point")
    with tempfile.TemporaryDirectory():
        pass
    return bad
