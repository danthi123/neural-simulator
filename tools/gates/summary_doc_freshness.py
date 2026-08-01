"""CLASS SF — THE FORWARD-LOOKING SUMMARY DOCS DRIFTED WHILE FINDINGS PILED UP, AND NOTHING CAUGHT IT.

THE DEFECT, measured 2026-08-01. Across one session ~13 findings landed — gap#4's on-bridge spiking wall
SURPASSED via e-prop, the affect evictor closed as a brain-based mechanism — and the LIVE BOARD
(`GAP_CLOSURE_MISSION.md`) was kept current every cycle out of session-habit. But the FORWARD-LOOKING summary
docs — the **master roadmap** (`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §7 walls-ledger + §8
next-actions) and the plain-language **`ROADMAP.md`** — were NEVER touched. Their gap#4 row still called the
on-bridge port a *wall* after it had been surpassed. This is drift #12 (summary-as-ground-truth), the #1 cause
of re-deriving concluded work, and it happened because keeping those docs current is a REMEMBERED skill-run
(`sync-documentation`, nudged by a PostToolUse hook that was read past) rather than an enforced check.

WHY A STALENESS BUDGET, NOT A PER-COMMIT TAX. Requiring a roadmap touch on EVERY finding would fire
constantly (most findings are contributions, not status changes) and become noise — the way a taxonomy dies.
But letting the roadmap go untouched across an unbounded run of findings is exactly the failure. So this gate
allows BATCHING up to a budget: findings may accumulate, but once `THRESHOLD` of them have landed since either
forward-looking doc was last touched, the next finding-adding commit BLOCKS until one is synced. You can
review-and-sync in one pass; you cannot drift forever.

WHAT IT ENFORCES, on a commit that ADDS a `research/findings/*.md`: if `THRESHOLD` or more findings have been
committed since the master roadmap OR `ROADMAP.md` was last modified, and neither is staged in THIS commit,
block with an instruction to run `sync-documentation` (or update §7/§8 + `ROADMAP.md`). Touching either doc
(even a "reviewed — no change" note) resets the budget.

DELIBERATELY NOT ENFORCED: that the sync is SEMANTICALLY correct — a gate cannot verify "the wall row now
reads surpassed." That is judgement (the `sync-documentation` skill's Check I). This gate guarantees the sync
HAPPENS periodically; the human/agent still does the reconciliation. It also does not gate the live board
(`GAP_CLOSURE_MISSION`) — that one did NOT drift; the forward-looking pair did.

WHAT IT CANNOT CATCH: a roadmap touched with a no-op that doesn't actually reflect the findings (resets the
budget without syncing). That is the same judgement seam as above — the budget forces the moment, not the
content.
"""
from __future__ import annotations

import os
import re
import subprocess

NAME = "summary-doc-freshness"
CLASS_ID = "SF"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THRESHOLD = 6  # findings may accumulate up to this many before the forward-looking docs MUST be synced.
ROADMAP = "docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md"
ROADMAP_SKIM = "ROADMAP.md"


def _decide(n_findings_added, stale_count, forward_doc_staged, threshold=THRESHOLD):
    """Pure decision (testable without git): block iff this commit adds a finding, no forward-looking doc is
    staged to absorb it, and the budget is already exhausted."""
    if n_findings_added <= 0:
        return None
    if forward_doc_staged:
        return None
    if stale_count < threshold:
        return None
    return ("the forward-looking summary docs (%s + %s) have NOT been synced across %d findings, and this "
            "commit adds another without touching either. Run the `sync-documentation` skill (or update the "
            "roadmap §7 walls-ledger / §8 next-actions + ROADMAP.md), then commit. Drift #12 (summary-as-"
            "ground-truth) is the #1 cause of re-deriving concluded work; on 2026-08-01 gap#4's on-bridge "
            "wall stayed 'wall' in the roadmap for a whole session after being surpassed."
            % (ROADMAP, ROADMAP_SKIM, stale_count))


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _stale_count():
    """How many findings have been COMMITTED since the more-recently-touched forward-looking doc."""
    ts = 0
    for doc in (ROADMAP, ROADMAP_SKIM):
        out = _git("log", "-1", "--format=%ct", "--", doc).strip()
        if out.isdigit():
            ts = max(ts, int(out))
    if ts == 0:
        return 0                                            # no history / not a repo: don't block
    since = _git("log", f"--since=@{ts}", "--name-only", "--pretty=format:", "--diff-filter=A", "--",
                 "research/findings/").split()
    findings = {f for f in since if f.endswith(".md") and "research/findings/" in f.replace("\\", "/")}
    return len(findings)


def check(paths):
    if paths is None or len(paths) == 0:
        return []                                           # standalone/empty: nothing to judge
    added_findings = [p for p in paths
                      if p.replace("\\", "/").endswith(".md") and "research/findings/" in p.replace("\\", "/")
                      and "research/findings/raw/" not in p.replace("\\", "/")]
    if not added_findings:
        return []
    staged = _git("diff", "--cached", "--name-only").split()
    forward_staged = any(f.replace("\\", "/").endswith(ROADMAP) or f.replace("\\", "/").endswith(ROADMAP_SKIM)
                         or ROADMAP in f.replace("\\", "/") for f in staged)
    msg = _decide(len(added_findings), _stale_count(), forward_staged)
    return [msg] if msg else []


def selftest():
    """FAILING DIRECTION FIRST: budget exhausted + a finding added + no forward doc staged -> block."""
    bad = []
    # 1. THE REAL CASE — over budget, finding added, roadmap not touched.
    if _decide(1, THRESHOLD, False) is None:
        bad.append("did NOT block when the budget was exhausted and a finding was added with no roadmap sync")
    if _decide(1, THRESHOLD + 5, False) is None:
        bad.append("did NOT block well over budget")
    # 2. NEGATIVE — a forward-looking doc IS staged this commit (the sync is happening) -> pass.
    if _decide(1, THRESHOLD + 5, True) is not None:
        bad.append("FALSE POSITIVE: blocked even though a forward-looking doc was staged (the sync)")
    # 3. NEGATIVE — under budget -> pass (batching is allowed).
    if _decide(1, THRESHOLD - 1, False) is not None:
        bad.append("FALSE POSITIVE: blocked while still under the batching budget")
    # 4. NEGATIVE — no finding added this commit -> never fires.
    if _decide(0, THRESHOLD + 99, False) is not None:
        bad.append("FALSE POSITIVE: fired on a commit that adds no finding")
    # 5. SCOPING — standalone/empty scans nothing.
    if check(None) or check([]):
        bad.append("SCOPE LEAK: standalone/empty mode must not block")
    return bad
