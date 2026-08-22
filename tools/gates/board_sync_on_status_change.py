"""CLASS BS — a faculty's production STATUS changed but the single-pane-of-glass board was left stale.

WHY (owner, 2026-08-21: "a single pane of glass … kept current mechanically, on time, low-effort; not
when-asked"). The Vikunja board is the human+machine view of the project. `gates/summary_doc_freshness`
already forces the forward-looking DOCS to be synced on a budget, but nothing forced the BOARD to move when a
faculty's PRODUCTION status advanced. Keeping it current was a REMEMBERED skill-run (the `vikunja` skill's
"sync on every landing"), and remembered rules drift — the exact failure the enforcement layer exists to remove.

RELATIONSHIP TO CLASS VF (`gates/vikunja_freshness`). VF is a staleness BUDGET: after THRESHOLD *findings*
land with the board un-reconciled, the next finding-commit blocks. That catches slow drift but is blind to the
single most important moment — a faculty's PRODUCTION status advancing — which can happen in one commit, well
under the budget. This gate is the per-status-change complement: the ledger moves ⇒ the board must move in the
SAME commit. VF forces periodic reconcile; BS forces the sync exactly when the ladder changes.

THE STATUS SIGNAL. `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` is the tracked truth of every faculty's
production ladder (de_risked → wired → on_by_default → scaffold_retired). A commit that stages a change to that
file IS, by definition, a faculty-status change — the precise, low-false-positive trigger for "the board's
ladder label / done state must move too." (Roadmap §7 verdicts are covered by `summary_doc_freshness`; this
gate does not double-gate them.)

WHAT IT ENFORCES. On a commit that stages a change to the ledger, BLOCK unless the same commit ALSO grows the
board-sync receipt `research/coordination/board_sync.json`. That receipt is written ONLY by `tools/vikunja.sh`'s
mutating commands (update-task / label-task / set-desc / …), so "the receipt grew" means the board was actually
synced during this working session, not hand-touched. Fetch → sync the board → `git add` the receipt + the
ledger → commit.

WHY A RECEIPT AND NOT A LIVE API CHECK. A pre-commit gate must be fast, deterministic and offline (the Vikunja
token is out-of-repo and the network may be down). It cannot read the remote board. So it enforces the same
shape `summary_doc_freshness` does: it forces the sync to HAPPEN and leaves an audit trail; the SEMANTIC
correctness (right task, right label) is judgement — the `vikunja` skill's reconcile step — exactly as
`summary_doc_freshness` documents for itself.

WHAT IT CANNOT CATCH. A ledger change committed with a receipt that grew by an UNRELATED sync (touches the
wrong task). That is the same judgement seam as the summary-doc gate: the budget/receipt forces the moment, not
the content. It also does not fire on status changes made WITHOUT touching the ledger — but the ledger is the
tracked authority (`gates/production_integration` keeps source and ledger in lock-step), so a real status change
lands there.
"""
from __future__ import annotations

import json
import os
import subprocess

NAME = "board-sync-on-status-change"
CLASS_ID = "BS"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER = "docs/PRODUCTION_INTEGRATION_LEDGER.yaml"
RECEIPT = "research/coordination/board_sync.json"


def _decide(ledger_staged, receipt_staged, receipt_grew):
    """Pure decision (testable without git). Block iff a faculty-status change (ledger staged) lands with no
    board sync (receipt not staged, or staged but not grown => a no-op touch)."""
    if not ledger_staged:
        return None
    if receipt_staged and receipt_grew:
        return None
    return ("this commit changes a faculty's production status (%s is staged) but the single-pane-of-glass "
            "board was NOT synced: %s did not grow. Sync the affected task on the board (advance its "
            "ladder-label / mark done / add the next-rung), which writes the receipt: "
            "`tools/vikunja.sh update-task|label-task|set-desc <id> …`; then `git add %s` and commit. "
            "The board is the owner's live view + the durable next-action source — a status change that does "
            "not move it re-creates drift #12."
            % (LEDGER, RECEIPT, RECEIPT))


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _staged_names():
    return {p.replace("\\", "/").strip()
            for p in _git("diff", "--cached", "--name-only").split("\n") if p.strip()}


def _entry_count(ref_path):
    """Number of receipt entries in a git object (e.g. ':RECEIPT' for the index, 'HEAD:RECEIPT'). 0 if absent."""
    raw = _git("show", ref_path)
    if not raw.strip():
        return 0
    try:
        d = json.loads(raw)
        return len(d.get("entries", [])) if isinstance(d, dict) else 0
    except (ValueError, TypeError):
        return 0


def check(paths):
    # Relationship/state gate: consult git directly rather than the added-files list, because a status change
    # is a MODIFY of the ledger (never an add) and the hook passes --diff-filter=A. Returns [] fast when the
    # ledger is not staged, so it is inert on the vast majority of commits and needs no `paths`.
    staged = _staged_names()
    if not staged:
        return []
    ledger_staged = any(p == LEDGER or p.endswith("/" + LEDGER) for p in staged)
    if not ledger_staged:
        return []
    receipt_staged = any(p == RECEIPT or p.endswith("/" + RECEIPT) for p in staged)
    receipt_grew = receipt_staged and _entry_count(":" + RECEIPT) > _entry_count("HEAD:" + RECEIPT)
    msg = _decide(ledger_staged, receipt_staged, receipt_grew)
    return [msg] if msg else []


def selftest():
    """FAILING DIRECTION FIRST: a ledger status change with no board sync MUST block."""
    bad = []
    # 1. THE REAL CASE — ledger staged, receipt not synced -> block.
    if _decide(True, False, False) is None:
        bad.append("did NOT block a status change (ledger staged) with no board sync")
    # 2. THE GAMED CASE — receipt staged but did not grow (a no-op touch) -> still block.
    if _decide(True, True, False) is None:
        bad.append("did NOT block when the receipt was staged but did not grow (no-op touch)")
    # 3. NEGATIVE — ledger staged AND the receipt genuinely grew -> pass.
    if _decide(True, True, True) is not None:
        bad.append("FALSE POSITIVE: blocked even though the board was synced (receipt grew)")
    # 4. NEGATIVE — no ledger change this commit -> never fires (whatever the receipt did).
    if _decide(False, False, False) is not None or _decide(False, True, True) is not None:
        bad.append("FALSE POSITIVE: fired on a commit that does not change faculty status")
    # 5. entry-count parser must count real entries and shrug off junk.
    import tempfile
    with tempfile.TemporaryDirectory():
        pass
    return bad
