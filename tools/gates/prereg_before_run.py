"""CLASS PR — a pre-registration committed TOGETHER WITH the run artifacts it governs.

EVIDENCE (2026-09-23). Three of five build lanes in one day came back from adversarial review with the same defect:
the PREREG file, the runner and the decisive seed-42 artifact all landed in ONE commit (curiosity 7ba619532,
perception 7a3bcccaa, and the D6 v3 prereg's earlier rounds). The artifact's provenance showed it was produced from
UNCOMMITTED prereg text, so git could not show the thresholds were written before the result was seen -- and in the
perception case the prereg's bands were demonstrably written after s42 was observed. docs/BUILD_LANE_CHECKLIST.md
already said "Pre-register in its own commit BEFORE any run it governs"; the rule was read and violated anyway.

THE GATE. At commit time: if the staged set ADDS a pre-registration document (a research/findings/ or docs/plans/
markdown whose filename contains PREREG, case-insensitive) AND stages any research/findings/raw/** data artifact
(anything except *.prov.json sidecars), BLOCK. Commit the prereg first, then run, then commit the artifacts.

ESCAPE (declared, visible in the document): a line `prereg-same-commit: <reason, >= 15 chars>` in the prereg -- e.g.
the artifacts are integrity smokes that no gate in the prereg reads, or a seed explicitly declared "observed before
registration, carries no pre-registered weight".

WHAT THIS GATE CANNOT CATCH.
  * A prereg committed first, then EDITED after the run (an amendment without an amendment log) -- that is a
    content judgement; the reviewer's job.
  * A run executed before the prereg commit but whose artifacts are committed later, separately: git order looks
    right while the thresholds may still have been fitted after seeing the data. Provenance timestamps vs the prereg
    commit time would catch it; not implemented here.
  * A pre-registration whose filename does not contain PREREG.
"""
from __future__ import annotations

import os
import re
import subprocess

NAME = "prereg_before_run"
CLASS_ID = "PR"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PREREG = re.compile(r"^(research/findings|docs/plans)/[^/]*prereg[^/]*\.md$", re.I)
_RAW = re.compile(r"^research/findings/raw/.+")
_ESCAPE = re.compile(r"^\s*[-*]?\s*prereg-same-commit:\s*(.{15,})$", re.I | re.M)


def _problems(added, staged, read):
    """added/staged: repo-relative paths; read(path) -> text. Pure, for the selftest."""
    preregs = [p for p in added if _PREREG.match(p)]
    raws = [p for p in staged if _RAW.match(p) and not p.endswith(".prov.json")]
    if not preregs or not raws:
        return []
    out = []
    for p in preregs:
        try:
            text = read(p)
        except OSError:
            text = ""
        if _ESCAPE.search(text):
            continue
        out.append("%s is ADDED in the same commit as %d run artifact(s) (e.g. %s). Commit the pre-registration "
                   "FIRST, then run, then commit the artifacts -- or declare `prereg-same-commit: <reason>` in it."
                   % (p, len(raws), raws[0]))
    return out


def _merged_in_unchanged(path):
    """True during a merge when `path`'s staged blob equals the incoming parent's (MERGE_HEAD) blob: the prereg is not
    new here, it arrives with its own history from the other branch (2026-09-24: merging main into a lane branch
    brought main's preregs + artifacts in as 'added' together and false-blocked the merge)."""
    def _git(*a):
        r = subprocess.run(["git", *a], cwd=_ROOT, capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    if not _git("rev-parse", "-q", "--verify", "MERGE_HEAD"):
        return False
    theirs = _git("rev-parse", "-q", "--verify", "MERGE_HEAD:" + path)
    staged = _git("rev-parse", "-q", "--verify", ":" + path)
    return bool(theirs) and theirs == staged


def check(paths):
    if paths is None or len(paths) == 0:
        return []                       # commit-scoped: nothing to say about the whole repo
    try:
        r = subprocess.run(["git", "diff", "--cached", "--name-status"], cwd=_ROOT, capture_output=True,
                           text=True, timeout=30)
    except Exception:
        return []
    added, staged = [], []
    for ln in r.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        st, path = parts[0], parts[-1]
        staged.append(path)
        if st.startswith("A"):
            added.append(path)
    added = [p for p in added if not _merged_in_unchanged(p)]
    return _problems(added, staged, lambda p: open(os.path.join(_ROOT, p), errors="ignore").read())


def selftest():
    """FAILING DIRECTION FIRST: a prereg added with a raw artifact MUST be caught."""
    bad = []
    txt = {"research/findings/2026-01-01-x-PREREG.md": "# prereg\nthresholds...\n",
           "docs/plans/2026-01-01-y-prereg.md": "# p\nprereg-same-commit: artifacts are integrity smokes only, no gate reads them\n"}
    rd = lambda p: txt[p]
    if not _problems(["research/findings/2026-01-01-x-PREREG.md"],
                     ["research/findings/2026-01-01-x-PREREG.md", "research/findings/raw/x/s42.json"], rd):
        bad.append("did NOT catch a prereg added together with a raw run artifact")
    if _problems(["research/findings/2026-01-01-x-PREREG.md"],
                 ["research/findings/2026-01-01-x-PREREG.md", "research/findings/raw/x/s42.json.prov.json"], rd):
        bad.append("FALSE POSITIVE: a provenance sidecar alone is not a run artifact")
    if _problems(["research/findings/2026-01-01-x-PREREG.md"], ["research/findings/2026-01-01-x-PREREG.md"], rd):
        bad.append("FALSE POSITIVE: a prereg committed on its own")
    if _problems([], ["research/findings/2026-01-01-x-PREREG.md", "research/findings/raw/x/s42.json"], rd):
        bad.append("FALSE POSITIVE: a MODIFIED (not added) prereg with artifacts (amendments are reviewed, not gated)")
    if _problems(["docs/plans/2026-01-01-y-prereg.md"],
                 ["docs/plans/2026-01-01-y-prereg.md", "research/findings/raw/y/s42.json"], rd):
        bad.append("FALSE POSITIVE: the declared prereg-same-commit escape was not honoured")
    return bad
