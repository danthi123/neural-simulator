"""CLASS VF — THE VIKUNJA BOARD DRIFTED WHILE FINDINGS PILED UP, AND NOTHING CAUGHT IT.

THE DEFECT, measured 2026-08-20 (owner-flagged). Across one long session ~15 findings/integrations landed — the
whole SWR memory-replay arc, the two-organ workspace default-on flip, the D5 learn-through-use arc (steps 1-5) — and
the LIVE board (`GAP_CLOSURE_MISSION.md`), the roadmap docs, and the local task list were all kept current. But the
**Vikunja board** (https://vikunja.dant123.com) — the OWNER'S monitor of status/progress — was never reconciled: a
crux the owner tracked as open (#71) had been closed end-to-end, a landed default-on faculty (#76) still read open,
and the whole session's work was invisible on it. This is the SAME failure `summary_doc_freshness` (CLASS SF) fixes
for the roadmap docs, one level out: keeping the board current was a REMEMBERED skill-run (`vikunja`, "read at start,
sync on landing") rather than an enforced check. The doc sync was enforced and stayed fresh; the board sync was
remembered and drifted. This gate closes that asymmetry.

HOW (mirror of CLASS SF, staleness budget — NOT a per-commit tax). Requiring a board touch on EVERY finding would be
noise (most findings are contributions, not capability-status changes). So findings may BATCH: once `THRESHOLD` of
them have landed since the board was last reconciled, the next finding-adding commit BLOCKS until a reconcile is
recorded. The board is external (network + a token outside the repo), so a pre-commit gate cannot read its live
state — instead it checks a tracked LOCAL MARKER, `docs/.vikunja_sync`, which the `vikunja` skill STAMPS (and stages)
after reconciling the board. Touching the marker resets the budget; you review-and-sync in one pass, you cannot drift
forever.

WHAT IT CANNOT CATCH: a marker stamped without actually reconciling the board (resets the budget with no sync). Same
judgement seam as CLASS SF — the budget forces the MOMENT, not the content; the `vikunja` skill's reconcile step is
the judgement. It also does not gate the live board / roadmap (CLASS SF already does); this is only the external
Vikunja board.
"""
from __future__ import annotations

import os
import subprocess

NAME = "vikunja-freshness"
CLASS_ID = "VF"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THRESHOLD = 6            # findings may accumulate up to this many before the Vikunja board MUST be reconciled.
MARKER = "docs/.vikunja_sync"   # tracked stamp; the `vikunja` skill updates+stages it after reconciling the board.


def _decide(n_findings_added, stale_count, marker_staged, threshold=THRESHOLD):
    """Pure decision (testable without git): block iff this commit adds a finding, the Vikunja sync-marker is not
    staged to absorb it, and the batching budget is already exhausted."""
    if n_findings_added <= 0:
        return None
    if marker_staged:
        return None
    if stale_count < threshold:
        return None
    return ("the Vikunja board (the owner's monitor, https://vikunja.dant123.com) has NOT been reconciled across "
            "%d findings, and this commit adds another without stamping it. Run the `vikunja` skill: reconcile the "
            "board (mark shipped tasks done, add the new next-rungs, advance ladder labels), then record it by "
            "updating %s (the sync step stamps it). Earned 2026-08-20: the board drifted a whole session (a closed "
            "crux still read open) because the roadmap sync was enforced but the board sync was only remembered."
            % (stale_count, MARKER))


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _stale_count():
    """How many findings have been COMMITTED since the Vikunja sync-marker was last touched."""
    out = _git("log", "-1", "--format=%ct", "--", MARKER).strip()
    if not out.isdigit():
        return 0                                            # marker has no history yet / not a repo: don't block
    ts = int(out)
    since = _git("log", f"--since=@{ts}", "--name-only", "--pretty=format:", "--diff-filter=A", "--",
                 "research/findings/").split()
    findings = {f for f in since if f.endswith(".md") and "research/findings/" in f.replace("\\", "/")
                and "research/findings/raw/" not in f.replace("\\", "/")}
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
    marker_staged = any(f.replace("\\", "/").endswith(MARKER) or MARKER in f.replace("\\", "/") for f in staged)
    msg = _decide(len(added_findings), _stale_count(), marker_staged)
    return [msg] if msg else []


def selftest():
    """FAILING DIRECTION FIRST: budget exhausted + a finding added + marker not staged -> block."""
    bad = []
    # 1. THE REAL CASE — over budget, finding added, board not reconciled.
    if _decide(1, THRESHOLD, False) is None:
        bad.append("did NOT block when the budget was exhausted and a finding was added with no board sync")
    if _decide(1, THRESHOLD + 9, False) is None:
        bad.append("did NOT block well over budget")
    # 2. NEGATIVE — the marker IS staged this commit (the reconcile is being recorded) -> pass.
    if _decide(1, THRESHOLD + 9, True) is not None:
        bad.append("FALSE POSITIVE: blocked even though the sync-marker was staged (the reconcile)")
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
