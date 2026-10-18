#!/usr/bin/env python3
"""tools/smoke_first.py -- no FLIP preregistration is filed (or its gate-seed lines queued) without a dev-seed SMOKE
record: the shared validator behind `tools/gates/flip_smoke_first.py` (commit time) and `tools/gate_preflight.py check`
(queue time). RULING smoke-first-1018 (owner-approved 2026-10-18), research/coordination/LOCAL_LLM_RUNBOOK.md section 6.

WHY. The word-role flip's reply regression ("what does the cat eat" abstains with the flag on) was found only after the
full six-seed gate plus Amendment G (FAILURE_LOG 2026-10-17); a two-seed dev smoke of flag-on vs flag-off, scored with
`tools/answer_rate.py paired`, would have shown it in about an hour. The ruling was the rule until this existed; now a
flip prereg without the record cannot be committed, and its gate-seed lines cannot be queued.

WHICH FILE IS A FLIP PREREG (decision recorded 2026-10-18; chosen on the 134 `*prereg*.md` files of the corpus). The
author declares it; a conservative fallback catches the author who forgets.
  1. EXPLICIT (wins): a line `flip: yes` (also true / on / default-on / a `BRAIN_*` flag name or list) in the
     frontmatter or the body marks a flip prereg. `flip: no <reason>` marks a non-flip. When the fallback below would
     call the file a flip, `flip: no` must carry a reason of >= 20 characters; a bare `flip: no` on a file named
     `...-flip-...` is read as a flip (the drive-by escape the fallback exists to catch). Any other value is ignored.
  2. FALLBACK (no usable `flip:` line): the file name has a `flip` token (`2026-10-16-word-role-flip-PREREGISTRATION.md`,
     `...-d6-hebbian-write-flip-...`; a name with `amendment` in it is exempt, an amendment inherits the smoke of the
     prereg it amends, and may opt in with `flip: yes`), or the body has a flag table whose header names both
     `old default` and `new default`. Measured on the corpus: the name rule finds the 11 real flip preregs, the table
     rule the word-role flip; neither touches the production-default batteries, the phase-timing prereg or the
     stage exit gates, which the looser textual rules ("becomes the production default", a `flip` in the title)
     wrongly caught (4 and 1 false positives).

WHAT A FLIP PREREG MUST CARRY.
    smoke-record: <path>          a JSON that `python tools/answer_rate.py paired ... --seeds 7,8 --json` wrote, COMMITTED
                                  (or staged in the same commit; at queue time: present at the pinned revision). One line
                                  per record: the ruling asks for the packet's own probes AND the iw1/iw2 battery scripts.
    smoke-hurts-ruling: <KEY>     only when a record carries FLIP-HURTS-ANSWERS: KEY must be a RULING in section 6 of the
                                  runbook (`- <KEY>: ...` or `- <date> RULING <KEY> ...`), i.e. Claude answered the ASK.
A record is valid iff it parses; is answer_rate's report (`check: answer_rate`) in `mode: paired`; measured at least two
seeds, each a DEV seed (1-15, never 42/43/44/100/101/102), with no UNMEASURED seed left in it; its per-seed status and
`flags` agree with its own counts re-derived (a hand-edited record fails); and shows no FLIP-HURTS-ANSWERS (or the ruling).

GRANDFATHER. At commit time only a prereg ADDED by the commit is judged, so every existing file is untouched (the whole
current tree passes: `python tools/smoke_first.py scan`). At queue time a prereg is judged only when it was first added
on or after GRANDFATHER_DATE (git history at the pinned revision; the file-name date when history is unavailable; a
prereg with no readable date is grandfathered, never guessed at).

WHAT THIS CANNOT CATCH. A smoke that was run on the wrong probes (it checks the record's shape and the dev-seed rule,
not that its faculty or script is the flip's); a flip prereg that neither carries `flip:` nor has `flip` in its name; a
ruling key the author picked for a different decision (it checks the key exists, not what it ruled).

Stdlib only, Python 3.8+: `tools/gate_preflight.py` runs under the server's system python3.
  python3 tools/smoke_first.py check <prereg path> [--revision REV]   exit 0 ok / grandfathered / not a flip, 2 refused
  python3 tools/smoke_first.py scan                                    list every flip prereg in the tree and its status
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNBOOK_REL = "research/coordination/LOCAL_LLM_RUNBOOK.md"
GRANDFATHER_DATE = os.environ.get("SMOKE_FIRST_GRANDFATHER_DATE") or "2026-10-18"
GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = tuple(range(1, 16))
MIN_SEEDS = 2
MIN_REASON = 20
HURTS = "FLIP-HURTS-ANSWERS"
PREREG_RE = re.compile(r"^(research/findings|docs/plans)/[^/]*prereg[^/]*\.md$", re.I)

_LINE_RE = re.compile(r"^[ \t>*+-]*(?:\*\*)?(?P<key>flip|smoke-record|smoke-hurts-ruling)(?:\*\*)?:(?:\*\*)?[ \t]*"
                      r"(?P<val>.*?)[ \t]*$", re.I)
_YES_RE = re.compile(r"^(?:yes|true|on|default-?on|1|brain_[a-z0-9_]+)\b", re.I)
_NO_RE = re.compile(r"^(?:no|false|off|n/?a|none|0)\b[\s:;,.\-—]*(?P<reason>.*)$", re.I)
_NAME_FLIP_RE = re.compile(r"(?:^|[-_.])flip(?:[-_.]|$)", re.I)
_TABLE_RE = re.compile(r"^\|[^\n]*old[ _-]default[^\n]*new[ _-]default[^\n]*\|", re.I | re.M)


# ---------------------------------------------------------------- reading the prereg
def lines_of(text):
    """[(key, value)] for every `flip:` / `smoke-record:` / `smoke-hurts-ruling:` line (bullets, bold, quotes allowed)."""
    out = []
    for ln in text.splitlines():
        m = _LINE_RE.match(ln)
        if m:
            out.append((m.group("key").lower(), m.group("val")))
    return out


def is_flip(path, text):
    """(is_flip, why). See the module docstring for the rule and for why each conjunct is there."""
    base = os.path.basename(path)
    fallback = None
    if not re.search(r"amendment", base, re.I) and _NAME_FLIP_RE.search(base):
        fallback = "file name has a `flip` token"
    elif _TABLE_RE.search(text):
        fallback = "a flag table with `old default` and `new default` columns"
    yes = no = None
    for key, val in lines_of(text):
        if key != "flip":
            continue
        v = val.strip("`<>\"' ")
        if _YES_RE.match(v):
            yes = v
        else:
            m = _NO_RE.match(v)
            if m:
                no = m.group("reason").strip()
    if yes is not None:
        return True, "`flip: yes` declared"
    if no is not None:
        if fallback and len(no) < MIN_REASON:
            return True, ("`flip: no` has no reason (>= %d chars) but %s: read as a flip" % (MIN_REASON, fallback))
        return False, "`flip: no` declared"
    if fallback:
        return True, fallback
    return False, "not a flip prereg"


# ---------------------------------------------------------------- the runbook's rulings
def ruling_keys(runbook_text):
    """Keys of the RULING entries in section 6 of the runbook (`- <KEY>: ...` or `- <date> RULING <KEY> ...`)."""
    keys = set()
    in6 = False
    for ln in (runbook_text or "").splitlines():
        if ln.startswith("## "):
            in6 = bool(re.match(r"^##\s+(?:6\.\s*)?RULINGS\b", ln))
            continue
        if not in6:
            continue
        m = re.match(r"^-\s+(?:\d{4}-\d{2}-\d{2}\s+)?RULING\s+([A-Za-z0-9][A-Za-z0-9_.\-]*)", ln)
        if not m:
            m = re.match(r"^-\s+([A-Za-z0-9][A-Za-z0-9_.\-]*):", ln)
        if m:
            keys.add(m.group(1))
    return keys


# ---------------------------------------------------------------- one smoke record
def _rate(side):
    n, a = side.get("n"), side.get("answered")
    if not isinstance(n, int) or not isinstance(a, int) or isinstance(n, bool) or n < 0 or a < 0 or a > n:
        return None, n
    return (a / n if n else None), n


def judge_record(path, raw):
    """(problems, hurt_seeds) for one record `path` whose bytes are `raw`. Pure."""
    try:
        rec = json.loads(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw)
    except ValueError as e:
        return ["%s: not JSON (%s)" % (path, e)], []
    if not isinstance(rec, dict) or rec.get("check") != "answer_rate":
        return ["%s: not an answer_rate report (no `check: answer_rate`): write it with `python tools/answer_rate.py "
                "paired ... --seeds 7,8 --json`" % path], []
    mode = rec.get("mode")
    if mode != "paired":
        why = ("lacks `mode` (it predates answer_rate's `mode` field): regenerate it" if mode is None
               else "is mode %r, not paired" % mode)
        return ["%s: the record %s. The smoke is `tools/answer_rate.py paired ... --seeds 7,8 --json`." % (path, why)], []
    per_seed = rec.get("per_seed")
    if not isinstance(per_seed, dict) or not per_seed:
        return ["%s: no per_seed rows" % path], []
    probs, hurts, measured, unmeasured = [], [], [], []
    for key in sorted(per_seed, key=lambda k: (not str(k).isdigit(), int(k) if str(k).isdigit() else 0, str(k))):
        row = per_seed[key]
        if not str(key).isdigit():
            probs.append("%s: seed key %r is not a seed number" % (path, key))
            continue
        seed = int(key)
        if seed in GATE_SEEDS:
            probs.append("%s: seed %d is a GATE seed. A smoke runs on dev seeds only (1-15; use --seeds 7,8): a gate "
                         "seed read before its prereg is a verdict, not a smoke." % (path, seed))
            continue
        if seed not in DEV_SEEDS:
            probs.append("%s: seed %d is not a dev seed (dev seeds are 1-15)" % (path, seed))
            continue
        on, off = (row or {}).get("on") or {}, (row or {}).get("off") or {}
        r_on, n_on = _rate(on)
        r_off, n_off = _rate(off)
        if r_on is None or r_off is None:
            unmeasured.append(seed)
            if (row or {}).get("status") not in ("UNMEASURED",):
                probs.append("%s: seed %d has a side with no replies but status %r (hand-edited?)"
                             % (path, seed, (row or {}).get("status")))
            continue
        measured.append(seed)
        derived = HURTS if r_on < r_off else "ok"
        if (row or {}).get("status") != derived:
            probs.append("%s: seed %d says status %r but its own counts (on %d/%d, off %d/%d) read %r -- a hand-edited "
                         "record; regenerate it with answer_rate.py" % (path, seed, (row or {}).get("status"),
                                                                       on["answered"], n_on, off["answered"], n_off,
                                                                       derived))
        if derived == HURTS:
            hurts.append(seed)
    if unmeasured:
        probs.append("%s: seed(s) %s are UNMEASURED (a side has no reply). An unmeasured seed is not 'no harm': "
                     "re-run the missing arm or drop the seed from --seeds." % (path, ",".join(map(str, unmeasured))))
    if len(set(measured)) < MIN_SEEDS:
        probs.append("%s: measures %d dev seed(s); the smoke needs at least %d (the ruling names 7 and 8)"
                     % (path, len(set(measured)), MIN_SEEDS))
    flagged = HURTS in (rec.get("flags") or [])
    if flagged != bool(hurts) and not probs:
        probs.append("%s: its `flags` %s %s but its own counts %s -- a hand-edited record; regenerate it"
                     % (path, "carry" if flagged else "omit", HURTS, "show it" if hurts else "do not show it"))
    return probs, sorted(set(hurts))


# ---------------------------------------------------------------- one prereg
def _clean_path(val):
    return val.strip("`<>\"' ")


def judge_prereg(path, text, read_blob, runbook_text):
    """Pure. Problems for ONE flip prereg. read_blob(relpath) -> bytes of that path as committed/staged, or None."""
    vals = lines_of(text)
    recs = [_clean_path(v) for k, v in vals if k == "smoke-record"]
    hurts_key = next((_clean_path(v) for k, v in reversed(vals) if k == "smoke-hurts-ruling" and _clean_path(v)), "")
    if not any(recs):
        return ["%s is a flip prereg (%s) but has no `smoke-record:` line. Before filing it, run the dev-seed smoke "
                "(RULING smoke-first-1018): flag-on vs flag-off at dev seeds 7 and 8 on the packet probes and the "
                "iw1/iw2 battery scripts, scored with `python tools/answer_rate.py paired ... --seeds 7,8 --json`; "
                "commit the JSON and add `smoke-record: <path>` (one line per record)." % (path, is_flip(path, text)[1])]
    keys = ruling_keys(runbook_text)
    out = []
    for rel in recs:
        if not rel:
            out.append("%s: empty `smoke-record:` line" % path)
            continue
        if os.path.isabs(rel) or ".." in rel.split("/"):
            out.append("%s: smoke-record %r must be a path inside the repository" % (path, rel))
            continue
        raw = read_blob(rel)
        if raw is None:
            out.append("%s: smoke-record %s is not committed (commit it with, or before, this file)" % (path, rel))
            continue
        probs, hurts = judge_record(rel, raw)
        out += probs
        if hurts and not probs:
            if not hurts_key:
                out.append("%s: %s carries %s on seed(s) %s. Any HURTS turn is an ASK before the prereg: file the ASK, "
                           "and when Claude has ruled, cite it with `smoke-hurts-ruling: <KEY>`."
                           % (path, rel, HURTS, ",".join(map(str, hurts))))
            elif hurts_key not in keys:
                out.append("%s: `smoke-hurts-ruling: %s` is not a RULING in section 6 of %s (%s carries %s)"
                           % (path, hurts_key, RUNBOOK_REL, rel, HURTS))
    return out


def evaluate(path, text, read_blob, runbook_text):
    """(is_flip, why, problems) for one prereg file."""
    flip, why = is_flip(path, text)
    return flip, why, (judge_prereg(path, text, read_blob, runbook_text) if flip else [])


# ---------------------------------------------------------------- grandfathering
def name_date(path):
    m = re.match(r"^(\d{4}-\d{2}-\d{2})-", os.path.basename(path))
    return m.group(1) if m else None


_GIT_LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                      "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_PREFIX")


def _git(root, args, timeout=30):
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_VARS}      # find the repo from `-C root`
    try:
        r = subprocess.run(["git", "-C", root] + list(args), capture_output=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return 127, b""
    return r.returncode, r.stdout


def blob_at(root, rev, rel):
    rc, out = _git(root, ["cat-file", "blob", "%s:%s" % (rev, rel)])
    return out if rc == 0 else None


def added_date(root, rev, rel):
    """YYYY-MM-DD of the commit that first ADDED `rel` in the history of `rev`; the file-name date when git cannot say;
    None when neither is known."""
    rc, out = _git(root, ["log", "--diff-filter=A", "--format=%cI", rev, "--", rel])
    if rc == 0:
        dates = [ln.strip() for ln in out.decode("utf-8", "replace").splitlines() if ln.strip()]
        if dates:
            return dates[-1][:10]
    return name_date(rel)


def is_grandfathered(root, rev, rel):
    d = added_date(root, rev, rel)
    return d is None or d < GRANDFATHER_DATE


def read_runbook(root, rev=None):
    """The runbook as Claude last wrote it: the working tree first (rulings land on main), else the revision's."""
    p = os.path.join(root, RUNBOOK_REL)
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        pass
    raw = blob_at(root, rev or "HEAD", RUNBOOK_REL)
    return raw.decode("utf-8", "replace") if raw else ""


# ---------------------------------------------------------------- queue time
def check_queue(prereg, revision, root=None):
    """(ok, reason) for the prereg named by a gate-seed queue line, at the line's pinned revision. Fails open (ok) only
    when git cannot read the revision at all; every determinate violation is a refusal."""
    root = root or ROOT
    raw = blob_at(root, revision, prereg)
    if raw is None:
        return True, "smoke-first: %s unreadable at %s, not checked" % (prereg, revision)
    text = raw.decode("utf-8", "replace")
    flip, why = is_flip(prereg, text)
    if not flip:
        return True, "smoke-first: %s is not a flip prereg" % prereg
    if is_grandfathered(root, revision, prereg):
        return True, "smoke-first: %s predates %s (grandfathered)" % (prereg, GRANDFATHER_DATE)
    probs = judge_prereg(prereg, text, lambda rel: blob_at(root, revision, rel), read_runbook(root, revision))
    if probs:
        return False, ("flip prereg %s was added on or after %s and has no valid dev-seed smoke "
                       "(RULING smoke-first-1018): %s" % (prereg, GRANDFATHER_DATE, " | ".join(probs)))
    return True, "smoke-first: %s smoke record ok" % prereg


# ---------------------------------------------------------------- CLI
def _scan(root):
    rows = []
    for d in ("research/findings", "docs/plans"):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            rel = "%s/%s" % (d, fn)
            if not PREREG_RE.match(rel):
                continue
            with open(os.path.join(full, fn), encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            flip, why = is_flip(rel, text)
            if flip:
                rows.append((rel, why, name_date(rel)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    c = sub.add_parser("check")
    c.add_argument("prereg")
    c.add_argument("--revision", default="HEAD")
    sub.add_parser("scan")
    a = ap.parse_args(argv)
    if a.step == "scan":
        rows = _scan(ROOT)
        bad = 0
        for rel, why, d in rows:
            gf = d is None or d < GRANDFATHER_DATE
            bad += 0 if gf else 1
            print("%s  %-14s %s  (%s)" % ("grandfathered" if gf else "CHECKED      ", d or "no-date", rel, why))
        print("%d flip prereg(s) in the tree; %d would be checked (added on or after %s)"
              % (len(rows), bad, GRANDFATHER_DATE))
        return 0
    ok, why = check_queue(a.prereg, a.revision)
    print(("ok: " if ok else "REFUSED: ") + why, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
