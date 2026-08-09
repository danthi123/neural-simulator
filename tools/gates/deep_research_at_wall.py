"""DR — DEEP RESEARCH (local record + EXTERNAL literature) NOT DONE WHEN HAMMERING A WALL.

THE DEFECT, recurring and owner-flagged more than once. When the project hits a wall, the reflex is to launch
lever after lever (mechanism after mechanism) WITHOUT the deep research that would either surface an existing
solution in our own record or a proven mechanism in the external literature. On 2026-08-09 the "teacher-loop
catastrophic forgetting" wall got FIVE mechanism levers (sleep-replay, spiking-engram store, replay-budget,
sparse-allocation, SHY) and a sixth queued — before any deep research. When the research was finally run it
took ONE query each to find: (a) the project ALREADY had a Complementary-Learning-Systems design and a
Phase-1.4 continual-learning mechanism at 103% retention, and that "replay caps at ~55%" was already
CHARACTERISED here (so the 0.55 plateau was re-derived, not novel); and (b) the external literature
(PS-SNN fixed-orthogonal-target pattern separation, EWC/Synaptic-Intelligence weight protection,
van de Ven generative replay) names the exact proven mechanisms the levers had been circling.

WHY THE EXISTING GATES DID NOT FIRE.
  - `corpus_check_required` (CC) is out of scope for cheap runs by design ("only fires on artifacts recording
    > 1 h of compute") — the whole forgetting arc was minute-long numpy de-risks, so CC never fired.
  - `refuted_mechanism_reproposal` (RM) catches re-proposing a REFUTED mechanism from a hand register; here the
    mechanisms were not "refuted", they were UN-RESEARCHED, and the existing SOLUTION (CLS/Phase-1.4) was not on
    the refuted register — it was simply never looked up.
  - `before_you_build.sh` records a LOCAL corpus check only. Nothing requires the EXTERNAL literature the owner
    specifically asks for, and every prior `.external_searches.jsonl` entry has an EMPTY `source`.
  - The "≥2 levers ⇒ research gate fires" line is PRINTED (before_you_build step 4), never enforced.

WHAT THIS GATE ENFORCES, regardless of compute cost. When a commit adds a mechanism-finding to a research LANE
that ALREADY carries >= DR_THRESHOLD findings dated within DR_WINDOW_DAYS (i.e. the wall is being hammered),
the commit must also carry evidence of a DEEP-research round in the same window: a `.external_searches.jsonl`
entry with a NON-EMPTY `source` (a real external paper / URL / author-year, not another internal cross-ref).
Missing it BLOCKS with the exact command to run. This makes the owner's standing rule mechanical: at a wall,
do the local-record check AND the external-literature check, and log a real source, BEFORE the next lever.

HOW TO CLEAR IT: `bash tools/deep_research.sh "<the wall in one line>"` — runs the local corpus check AND
prompts + records the external-literature search with a real source. (Or record one directly:
`bash tools/record_external_search.sh "<query>" "<source: paper/url/author-year>"`.)

WHAT IT CANNOT CATCH (left as judgement, not pretended away): an external source recorded but not READ; a wall
spread across several LANES so no single lane crosses the threshold (extend the grouping by mechanism-keyword
as walls teach us to); a genuinely-novel wall with no external literature (record `source: none-found — <why>`
which is itself a non-empty, honest source string and clears the gate while stating the search was done).
"""
from __future__ import annotations

import json
import os
import re
import time

NAME = "deep-research-at-wall"
CLASS_ID = "DR"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DR_THRESHOLD = 3          # findings in one lane within the window = "hammering a wall"
DR_WINDOW_DAYS = 3
_DAY = 86400.0

_FM = re.compile(r"^---\s*$(.*?)^---\s*$", re.M | re.S)


def _frontmatter(text):
    m = _FM.search(text)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            fm[k.strip().lower()] = v.strip().strip("'\"")
    return fm


def _date_epoch(s):
    """A frontmatter date:YYYY-MM-DD -> epoch seconds, or None."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(s or ""))
    if not m:
        return None
    try:
        return time.mktime((int(m.group(1)), int(m.group(2)), int(m.group(3)), 12, 0, 0, 0, 0, -1))
    except Exception:
        return None


def _findings_dir(root):
    return os.path.join(root, "research", "findings")


def _is_finding(path):
    p = path.replace("\\", "/")
    return "/research/findings/" in ("/" + p) and p.endswith(".md") and "/raw/" not in ("/" + p)


def _lane_and_date(fpath):
    try:
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            fm = _frontmatter(fh.read(4000))
    except OSError:
        return None, None
    if (fm.get("type") or "").strip() != "finding":
        return None, None
    return (fm.get("lane") or "").strip().lower() or None, _date_epoch(fm.get("date"))


def _lane_count_in_window(root, lane, center_epoch):
    """How many findings on disk share `lane` and are dated within DR_WINDOW_DAYS of center_epoch."""
    d = _findings_dir(root)
    n = 0
    if not os.path.isdir(d):
        return 0
    for name in os.listdir(d):
        if not name.endswith(".md"):
            continue
        fl, fe = _lane_and_date(os.path.join(d, name))
        if fl == lane and fe is not None and abs(fe - center_epoch) <= DR_WINDOW_DAYS * _DAY:
            n += 1
    return n


def _fresh_external_source(root, center_epoch, log=None):
    """True iff `.external_searches.jsonl` has an entry with a NON-EMPTY source within the window."""
    log = log or os.path.join(root, "research", "queue", ".external_searches.jsonl")
    if not os.path.exists(log):
        return False
    try:
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                src = str(rec.get("source") or "").strip()
                if not src:
                    continue
                ts = rec.get("ts") or rec.get("iso") or ""
                te = None
                m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(ts))
                if m:
                    te = _date_epoch(str(ts)[:10])
                if te is None or abs(te - center_epoch) <= DR_WINDOW_DAYS * _DAY:
                    return True
    except OSError:
        return False
    return False


def check(paths, root=_ROOT, log=None):
    problems = []
    staged = [p for p in paths if _is_finding(p)]
    if not staged:
        return problems
    # lanes touched by staged findings, with the newest staged date per lane as the window centre
    lanes = {}
    for p in staged:
        ap = p if os.path.isabs(p) else os.path.join(root, p)
        lane, dt = _lane_and_date(ap)
        if lane and dt is not None:
            lanes[lane] = max(lanes.get(lane, dt), dt)
    for lane, center in lanes.items():
        n = _lane_count_in_window(root, lane, center)
        if n >= DR_THRESHOLD and not _fresh_external_source(root, center, log=log):
            problems.append(
                "lane '%s' now has %d findings within %d days (repeated levers at a wall) but NO EXTERNAL-literature "
                "check with a real source is logged. Deep research (local record + external literature) is required "
                "before the next lever. Run:  bash tools/deep_research.sh \"<the wall in one line>\"  (or record a "
                "source: bash tools/record_external_search.sh \"<query>\" \"<paper/url/author-year>\"). "
                "The record already solving a wall, or a proven external mechanism, is one query away — that is the "
                "recurring, expensive miss this gate exists to stop." % (lane, n, DR_WINDOW_DAYS)
            )
    return problems


def selftest():
    """MUST fail in the failing direction: a hammered lane with no fresh external source -> a problem."""
    import tempfile
    out = []
    with tempfile.TemporaryDirectory() as d:
        fd = os.path.join(d, "research", "findings")
        qd = os.path.join(d, "research", "queue")
        os.makedirs(fd)
        os.makedirs(qd)
        now = time.strftime("%Y-%m-%d")
        staged = []
        for i in range(DR_THRESHOLD):
            fp = os.path.join(fd, "2026-08-09-wall-lever-%d.md" % i)
            with open(fp, "w") as fh:
                fh.write("---\ntype: finding\nstatus: contributing\ndate: %s\nmechanism: m-%d\nlane: the-wall\n---\n# lever %d\n" % (now, i, i))
            staged.append(fp)
        # (A) FAILING case: threshold met, no external source -> MUST return a problem
        probs = check(staged, root=d, log=os.path.join(qd, ".external_searches.jsonl"))
        if not probs:
            out.append("FAIL: hammered lane with no external source did not block")
        # (B) PASSING case: add a fresh external source -> MUST clear
        with open(os.path.join(qd, ".external_searches.jsonl"), "w") as fh:
            fh.write(json.dumps({"ts": now + "T00:00:00Z", "query": "the wall", "source": "PS-SNN Hu 2026 Sci Reports"}) + "\n")
        if check(staged, root=d, log=os.path.join(qd, ".external_searches.jsonl")):
            out.append("FAIL: a fresh real external source did not clear the gate")
        # (C) below threshold -> no problem
        if check(staged[:1], root=d, log=os.path.join(qd, ".external_searches.jsonl")):
            out.append("FAIL: a single lever (below threshold) fired the gate")
    return out
