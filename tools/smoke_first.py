#!/usr/bin/env python3
"""tools/smoke_first.py -- no FLIP preregistration is filed (or its gate-seed lines queued) without dev-seed SMOKE
records: the shared validator behind `tools/gates/flip_smoke_first.py` (commit time) and `tools/gate_preflight.py check`
(queue time). RULING smoke-first-1018 (owner-approved 2026-10-18), research/coordination/LOCAL_LLM_RUNBOOK.md section 6.

WHY. The word-role flip's reply regression ("what does the cat eat" abstains with the flag on) was found only after the
full six-seed gate plus Amendment G (FAILURE_LOG 2026-10-17); a two-seed dev smoke of flag-on vs flag-off, scored with
`tools/answer_rate.py paired`, would have shown it in about an hour. The ruling was the rule until this existed; now a
flip prereg without the records cannot be committed, and its gate-seed lines cannot be queued.

EVERY NEW PREREG DECLARES (CLASS BAR shape; review 2026-10-18). A prereg first added on or after GRANDFATHER_DATE must
carry exactly one of
    flip: yes                     (also true / on / default-on / a `BRAIN_*` flag name or list): a default-on candidate
    flip: no <reason>             a reason of at least 20 characters: it flips no default
and a silent prereg is refused, so a flip cannot slip through by saying nothing. Two fallbacks read a file as a flip
whatever it declares (`flip: no` cannot overrule them; rename the file or drop the table if it is no flip): the file name
has a `flip` / `flips` token (an `amendment` in the name exempts it from THIS rule only; it still has to declare), or the
body has a flag table whose header names both `old default` and `new default`. Measured on the 133 `*prereg*.md` files of
the corpus: the two fallbacks find the 11 real flip preregs and no other; looser textual rules ("becomes the production
default", `flip` in the title, a `BRAIN_*` line that says flip) wrongly caught 4-9 battery and exit-gate preregs.

WHAT A FLIP PREREG MUST CARRY.
    smoke-record: <path> [faculty=<dir>]   a JSON that `python tools/answer_rate.py paired ... --seeds 7,8 --json` wrote,
                                  COMMITTED (or staged in the same commit; at queue time: present at the pinned revision).
                                  AT LEAST TWO lines, one per record: the ruling asks for the packet's own probes AND the
                                  iw1/iw2 battery scripts. Keep the file OUTSIDE research/findings/raw (recommended
                                  `research/findings/smoke/`): under raw/ the prereg-before-run gate (PR) blocks a record
                                  staged beside its prereg.
    smoke-hurts-ruling: <KEY>     only when a record carries FLIP-HURTS-ANSWERS (any HURTS turn is an ASK before the prereg)
    smoke-exempt-ruling: <KEY>    the flip has no reply surface the smoke can read: Claude ruled so (replaces the records)
A KEY counts only if it is a RULING in section 6 of the runbook AS MAIN HOLDS IT (never the working tree or the index of
the checkout that is committing, so a prereg cannot write its own ruling), the entry is signed `(... Claude ...)`, and
the entry cites this prereg's file name or path (or, for a HURTS ruling, the record path). Entry styles read:
`- <KEY>: ...` and `- <date> RULING <KEY> ...`, with indented continuation lines.
A record is valid iff it parses and is answer_rate's report (`check: answer_rate`, `mode: paired`) as the CLI writes it:
`faculty`, `on_tag`, exactly one of `off_tag` / `off_arm` (never equal to `on_tag`), `evidence` (one `{file, sha256}` per
probe), and the per-seed counts; it measured BOTH dev seeds 7 and 8 (the ruling's seeds), any other seed a dev seed
(1-15, never 42/43/44/100/101/102), no UNMEASURED seed, at least MIN_PROBES probes per side, evidence entries matching the
counts and, where the file is committed at the revision, the file's sha256; its per-seed status and `flags` agree with the
counts re-derived (a hand-edited record fails); and it shows no FLIP-HURTS-ANSWERS (or the ruling). Across the records the
labels differ, no probe file is shared, and each record's faculty is named in the prereg (a record cannot be borrowed
from another flip).

GRANDFATHER (review 2026-10-18). The history of the file is followed through renames (path-limited `git log` plus `git diff-tree -M`). A prereg is
`new` when it, or its file-name date, is dated on or after GRANDFATHER_DATE (the LATER of the two: neither a forged
committer date nor a lying file name alone moves it). Otherwise it is `old`: it is judged only if it BECAME a flip after
the cutoff (its newest version before the cutoff was no flip: edited in place or renamed into a flip). A new prereg is
judged on every commit that touches it (added, modified or renamed), at the commit and at queue time. GRANDFATHER_DATE is
a constant; tests move it with monkeypatch.

WHAT THIS CANNOT CATCH. A record is checked for shape, internal consistency, the dev-seed rule and (when its files are
committed) its evidence hashes, not for authenticity: a forger who writes a self-consistent record with invented evidence
passes (the record's faculty, tags and probe files make that a deliberate act rather than a slip). A smoke run on the
wrong probes; a ruling signed `(Claude` by a hand that is not Claude's (main is the trust anchor); a hand that sets the
committer date AND names the file without a date.

Stdlib only, Python 3.8+: `tools/gate_preflight.py` runs under the server's system python3.
  python3 tools/smoke_first.py check <prereg path> [--revision REV]   exit 0 ok / grandfathered / not a flip, 2 refused
  python3 tools/smoke_first.py scan                                    list every flip prereg in the tree and its status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNBOOK_REL = "research/coordination/LOCAL_LLM_RUNBOOK.md"
GRANDFATHER_DATE = "2026-10-18"
GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = tuple(range(1, 16))
REQUIRED_SEEDS = (7, 8)
MIN_RECORDS = 2
MIN_PROBES = 2
MIN_REASON = 20
HURTS = "FLIP-HURTS-ANSWERS"
RECORD_DIR = "research/findings/smoke"
PREREG_RE = re.compile(r"^(research/findings|docs/plans)/[^/]*prereg[^/]*\.md$", re.I)

_KEYS = ("flip", "smoke-record", "smoke-hurts-ruling", "smoke-exempt-ruling")


def key_line_re(keys):
    """The declaration-line grammar (`key: value`, bullets, bold and quotes allowed), for any key set. Shared with
    tools/gates/host_decision_declared.py (CLASS HD) so the two gates read a declaration line the same way."""
    return re.compile(r"^[ \t>*+-]*(?:\*\*)?(?P<key>%s)(?:\*\*)?:(?:\*\*)?[ \t]*(?P<val>.*?)[ \t]*$"
                      % "|".join(re.escape(k) for k in keys), re.I)


_LINE_RE = key_line_re(_KEYS)
_YES_RE = re.compile(r"^(?:yes|true|on|default-?on|1|brain_[a-z0-9_]+)\b", re.I)
_NO_RE = re.compile(r"^(?:no|false|off|n/?a|none|0)\b[\s:;,.\-—]*(?P<reason>.*)$", re.I)
_NAME_FLIP_RE = re.compile(r"(?:^|[-_.])flips?(?:[-_.]|$)", re.I)
_TABLE_RE = re.compile(r"^\|[^\n]*old[ _-]default[^\n]*new[ _-]default[^\n]*\|", re.I | re.M)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SEED_RE = re.compile(r"^[0-9]{1,6}$")


# ---------------------------------------------------------------- reading the prereg
def lines_of(text):
    """[(key, value)] for every `flip:` / `smoke-*:` line (bullets, bold, quotes allowed)."""
    out = []
    for ln in text.splitlines():
        m = _LINE_RE.match(ln)
        if m:
            out.append((m.group("key").lower(), m.group("val")))
    return out


def _first_token(val):
    parts = val.strip().lstrip("`<>\"'*(").split()
    return parts[0].strip("`<>\"',;()*") if parts else ""


def _fallback(path, text):
    base = os.path.basename(path)
    if not re.search(r"amendment", base, re.I) and _NAME_FLIP_RE.search(base):
        return "the file name has a `flip` token"
    if _TABLE_RE.search(text):
        return "it has a flag table with `old default` and `new default` columns"
    return None


def declaration(text):
    """('yes', value) | ('no', reason) | (None, '') from the `flip:` lines (yes wins; an unrecognised value is ignored)."""
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
        return "yes", yes
    if no is not None:
        return "no", no
    return None, ""


def classify(path, text):
    """(kind, why). kind: 'flip' | 'nonflip' | 'undeclared'. See the module docstring."""
    fb = _fallback(path, text)
    d, val = declaration(text)
    if d == "yes":
        return "flip", "`flip: yes` declared"
    if fb:
        if d == "no":
            return "flip", ("it declares `flip: no` but %s: read as a flip (`flip: no` cannot overrule that; rename the "
                            "file or drop the table if it flips no default)" % fb)
        return "flip", fb
    if d == "no":
        if len(val) < MIN_REASON:
            return "undeclared", "`flip: no` needs a reason of at least %d characters" % MIN_REASON
        return "nonflip", "`flip: no` declared"
    return "undeclared", "no `flip: yes` / `flip: no <reason>` line"


def is_flip(path, text):
    """(is_flip, why)."""
    kind, why = classify(path, text)
    return kind == "flip", why


# ---------------------------------------------------------------- the runbook's rulings
def ruling_entries(runbook_text):
    """{KEY: entry text} for the Claude-signed RULING entries of section 6 (`- <KEY>: ...` or `- <date> RULING <KEY> ...`
    plus indented continuation lines). An entry without a `(... Claude ...)` signature is not a ruling here."""
    out = {}
    state = {"key": None, "lines": []}
    in6 = False

    def flush():
        key, body = state["key"], "\n".join(state["lines"])
        if key and key not in out and re.search(r"\([^()]*\bClaude\b", body):
            out[key] = body
        state["key"], state["lines"] = None, []

    for ln in (runbook_text or "").splitlines():
        if ln.startswith("## "):
            flush()
            in6 = bool(re.match(r"^##\s+(?:6\.\s*)?RULINGS\b", ln))
            continue
        if not in6:
            continue
        if re.match(r"^-\s", ln):
            flush()
            m = re.match(r"^-\s+(?:\d{4}-\d{2}-\d{2}\s+)?RULING\s+([A-Za-z0-9][A-Za-z0-9_.\-]*)", ln)
            if not m:
                m = re.match(r"^-\s+([A-Za-z0-9][A-Za-z0-9_.\-]*):", ln)
            if m:
                state["key"], state["lines"] = m.group(1), [ln]
        elif state["key"] and ln[:1] in (" ", "\t"):
            state["lines"].append(ln)
        else:
            flush()
    flush()
    return out


def ruling_keys(runbook_text):
    return set(ruling_entries(runbook_text))


def _ruling_ok(entries, key, anchors):
    """(ok, why) for a cited ruling KEY: a Claude-signed entry of section 6 that cites one of `anchors`."""
    if key not in entries:
        return False, "is not a Claude-signed RULING in section 6 of %s (as main holds it)" % RUNBOOK_REL
    body = entries[key]
    if not any(a and a in body for a in anchors):
        return False, ("is a RULING but does not cite this prereg's file name or path%s; a ruling must be about THIS "
                       "decision" % (" or the record path" if len(anchors) > 2 else ""))
    return True, ""


# ---------------------------------------------------------------- one smoke record
def _rate(side):
    if not isinstance(side, dict):
        return None, None
    n, a = side.get("n"), side.get("answered")
    if not isinstance(n, int) or not isinstance(a, int) or isinstance(n, bool) or isinstance(a, bool) \
            or n < 0 or a < 0 or a > n:
        return None, n
    return (a / n if n else None), n


def _safe_rel(rel):
    return isinstance(rel, str) and rel and not os.path.isabs(rel) and ".." not in rel.split("/")


def _judge_record_inner(path, raw, read_blob):
    try:
        rec = json.loads(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw)
    except ValueError as e:
        return ["%s: not JSON (%s)" % (path, e)], [], {}
    if not isinstance(rec, dict) or rec.get("check") != "answer_rate":
        return ["%s: not an answer_rate report (no `check: answer_rate`): write it with `python tools/answer_rate.py "
                "paired ... --seeds 7,8 --json`" % path], [], {}
    mode = rec.get("mode")
    if mode != "paired":
        why = ("lacks `mode` (it predates answer_rate's `mode` field): regenerate it" if mode is None
               else "is mode %r, not paired" % (mode,))
        return ["%s: the record %s. The smoke is `tools/answer_rate.py paired ... --seeds 7,8 --json`." % (path, why)], \
            [], {}
    per_seed = rec.get("per_seed")
    if not isinstance(per_seed, dict) or not per_seed:
        return ["%s: no per_seed rows" % path], [], {}
    probs, hurts, measured, unmeasured = [], [], [], []
    ev = rec.get("evidence")
    ev = ev if isinstance(ev, dict) else {}
    files = []
    # --- who made it: the CLI writes these; a hand-typed record lacks them
    faculty, on_tag = rec.get("faculty"), rec.get("on_tag")
    off_tag, off_arm = rec.get("off_tag"), rec.get("off_arm")
    label = rec.get("label")
    if not (isinstance(faculty, str) and faculty.strip() and isinstance(on_tag, str) and on_tag.strip()
            and isinstance(label, str) and label.strip()):
        probs.append("%s: no `faculty` / `on_tag` / `label`: not written by answer_rate.py paired; regenerate it" % path)
    if bool(off_tag) == bool(off_arm) or not isinstance(off_tag or off_arm, str):
        probs.append("%s: needs exactly one of `off_tag` / `off_arm` (the flag-off side); regenerate it with answer_rate.py"
                     % path)
    elif off_tag and off_tag == on_tag:
        probs.append("%s: on_tag and off_tag are both %r: the flip was compared with itself, which can never show a harm"
                     % (path, on_tag))
    if not isinstance(rec.get("argv"), list) or not str(rec.get("script", "")).endswith("answer_rate.py"):
        probs.append("%s: no `script` / `argv` of tools/answer_rate.py: not written by it; regenerate it" % path)
    if not isinstance(rec.get("flags", []), list):
        probs.append("%s: `flags` is not a list" % path)
        flags = []
    else:
        flags = rec.get("flags", [])
    for key in sorted(per_seed, key=lambda k: (not _SEED_RE.match(str(k)), int(k) if _SEED_RE.match(str(k)) else 0,
                                               str(k))):
        row = per_seed[key]
        if not _SEED_RE.match(str(key)):
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
        if not isinstance(row, dict):
            probs.append("%s: seed %d has a row that is not an object" % (path, seed))
            continue
        on, off = row.get("on"), row.get("off")
        r_on, n_on = _rate(on)
        r_off, n_off = _rate(off)
        if (on is not None and not isinstance(on, dict)) or (off is not None and not isinstance(off, dict)):
            probs.append("%s: seed %d has a side that is not an object" % (path, seed))
            continue
        if r_on is None or r_off is None:
            unmeasured.append(seed)
            if row.get("status") != "UNMEASURED":
                probs.append("%s: seed %d has a side with no replies but status %r (hand-edited?)"
                             % (path, seed, row.get("status")))
            continue
        measured.append(seed)
        derived = HURTS if r_on < r_off else "ok"
        if row.get("status") != derived:
            probs.append("%s: seed %d says status %r but its own counts (on %d/%d, off %d/%d) read %r -- a hand-edited "
                         "record; regenerate it with answer_rate.py" % (path, seed, row.get("status"), on["answered"],
                                                                       n_on, off["answered"], n_off, derived))
        if min(n_on, n_off) < MIN_PROBES:
            probs.append("%s: seed %d has %d probe(s) on one side; the smoke needs at least %d per side"
                         % (path, seed, min(n_on, n_off), MIN_PROBES))
        if derived == HURTS:
            hurts.append(seed)
        # --- the evidence: one {file, sha256} per probe, matching the counts and (when committed) the files
        e = ev.get(str(key))
        for side_name, side, n in (("on", on, n_on), ("off", off, n_off)):
            lst = e.get(side_name) if isinstance(e, dict) else None
            if not isinstance(lst, list) or len(lst) != n:
                probs.append("%s: seed %d `evidence.%s` does not list the %d probe file(s) of its counts; regenerate it "
                             "with answer_rate.py" % (path, seed, side_name, n))
                continue
            for item in lst:
                f, h = (item.get("file"), item.get("sha256")) if isinstance(item, dict) else (None, None)
                if not (isinstance(f, str) and f and isinstance(h, str) and _SHA_RE.match(h)):
                    probs.append("%s: seed %d `evidence.%s` has an entry without file + sha256" % (path, seed, side_name))
                    continue
                files.append(f)
                if read_blob is not None and _safe_rel(f):
                    blob = read_blob(f)
                    if blob is not None and hashlib.sha256(blob).hexdigest() != h:
                        probs.append("%s: evidence file %s differs from the sha256 the record states (the record was "
                                     "not made from this file)" % (path, f))
    if unmeasured:
        probs.append("%s: seed(s) %s are UNMEASURED (a side has no reply). An unmeasured seed is not 'no harm': "
                     "re-run the missing arm or drop the seed from --seeds." % (path, ",".join(map(str, unmeasured))))
    missing = [s for s in REQUIRED_SEEDS if s not in measured]
    if missing:
        probs.append("%s: dev seed(s) %s not measured; the smoke runs at dev seeds %s (RULING smoke-first-1018)"
                     % (path, ",".join(map(str, missing)), " and ".join(map(str, REQUIRED_SEEDS))))
    flagged = HURTS in flags
    if flagged != bool(hurts) and not probs:
        probs.append("%s: its `flags` %s %s but its own counts %s -- a hand-edited record; regenerate it"
                     % (path, "carry" if flagged else "omit", HURTS, "show it" if hurts else "do not show it"))
    if len(files) != len(set(files)):
        probs.append("%s: a probe file is listed twice (the same file on both sides or in two seeds)" % path)
    meta = {"label": label if isinstance(label, str) else "", "faculty": faculty if isinstance(faculty, str) else "",
            "files": set(files)}
    return probs, sorted(set(hurts)), meta


def judge_record_meta(path, raw, read_blob=None):
    """(problems, hurt_seeds, meta) for one record `path` whose bytes are `raw`. Pure; never raises."""
    try:
        return _judge_record_inner(path, raw, read_blob)
    except Exception as e:                                       # noqa: BLE001 -- a malformed record is a finding
        return ["%s: malformed record (%s: %s); regenerate it with answer_rate.py" % (path, type(e).__name__,
                                                                                    str(e)[:80])], [], {}


def judge_record(path, raw, read_blob=None):
    """(problems, hurt_seeds) for one record."""
    probs, hurts, _ = judge_record_meta(path, raw, read_blob)
    return probs, hurts


# ---------------------------------------------------------------- one prereg
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def judge_prereg(path, text, read_blob, runbook_text):
    """Pure. Problems for ONE flip prereg. read_blob(relpath) -> bytes of that path as committed/staged, or None.
    runbook_text = the runbook as MAIN holds it (see read_runbook)."""
    vals = lines_of(text)
    recs, empty = [], 0
    for k, v in vals:
        if k == "smoke-record":
            tok = _first_token(v)
            if not tok:
                empty += 1
            elif tok not in recs:
                recs.append(tok)
    hurts_key = next((_first_token(v) for k, v in reversed(vals) if k == "smoke-hurts-ruling" and _first_token(v)), "")
    exempt_key = next((_first_token(v) for k, v in reversed(vals) if k == "smoke-exempt-ruling" and _first_token(v)), "")
    entries = ruling_entries(runbook_text)
    base = os.path.basename(path)
    if exempt_key:
        ok, why = _ruling_ok(entries, exempt_key, [base, path])
        return [] if ok else ["%s: `smoke-exempt-ruling: %s` %s" % (path, exempt_key, why)]
    if not recs and not empty:
        return ["%s is a flip prereg (%s) but has no `smoke-record:` line. Before filing it, run the dev-seed smoke "
                "(RULING smoke-first-1018): flag-on vs flag-off at dev seeds 7 and 8 on the packet probes AND on the "
                "iw1/iw2 battery scripts, each scored with `python tools/answer_rate.py paired ... --seeds 7,8 --json "
                "> %s/<name>.json`; commit the JSONs (outside research/findings/raw) and add one `smoke-record: <path> "
                "faculty=<dir>` line per record. A flip with no reply surface needs a Claude ruling that cites this "
                "file, given as `smoke-exempt-ruling: <KEY>`." % (path, classify(path, text)[1], RECORD_DIR)]
    out = ["%s: empty `smoke-record:` line" % path] * empty
    metas = []
    for rel in recs:
        if not _safe_rel(rel):
            out.append("%s: smoke-record %r must be a path inside the repository" % (path, rel))
            continue
        raw = read_blob(rel)
        if raw is None:
            out.append("%s: smoke-record %s is not committed (commit it with, or before, this file)" % (path, rel))
            continue
        probs, hurts, meta = judge_record_meta(rel, raw, read_blob)
        out += probs
        if probs:
            continue
        metas.append((rel, hurts, meta))
        if hurts:
            if not hurts_key:
                out.append("%s: %s carries %s on seed(s) %s. Any HURTS turn is an ASK before the prereg: file the ASK, "
                           "and when Claude has ruled (citing this file name or the record path), cite it with "
                           "`smoke-hurts-ruling: <KEY>`." % (path, rel, HURTS, ",".join(map(str, hurts))))
            else:
                ok, why = _ruling_ok(entries, hurts_key, [base, path, rel])
                if not ok:
                    out.append("%s: `smoke-hurts-ruling: %s` %s (%s carries %s)" % (path, hurts_key, why, rel, HURTS))
    n_paths = len(recs)
    if n_paths < MIN_RECORDS:
        out.append("%s: cites %d smoke record(s); the ruling asks for the packet's own probes AND the iw1/iw2 battery "
                   "scripts, so at least %d distinct records (one `smoke-record:` line each)" % (path, n_paths, MIN_RECORDS))
    labels = [m["label"] for _, _, m in metas]
    for lab in sorted({x for x in labels if labels.count(x) > 1}):
        out.append("%s: two records share the label %r: they must be two different smokes" % (path, lab))
    seen = {}
    for rel, _, m in metas:
        for f in m["files"]:
            if f in seen and seen[f] != rel:
                out.append("%s: records %s and %s read the same probe file %s: they must be two different smokes"
                           % (path, seen[f], rel, f))
                break
            seen[f] = rel
    norm_text = _norm(text)
    for rel, _, m in metas:
        if m["faculty"] and _norm(m["faculty"]) not in norm_text:
            out.append("%s: record %s is for faculty %r, which this prereg never names. A record cannot be borrowed from "
                       "another flip: name the faculty in the prereg (e.g. `smoke-record: %s faculty=%s`)"
                       % (path, rel, m["faculty"], rel, m["faculty"]))
    return out


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


def file_history(root, rev, rel, max_renames=4):
    """[(sha, YYYY-MM-DD, path)] newest first, following up to `max_renames` renames; [] when git cannot say. Path-limited
    `git log` (about a second on this repo) plus one cheap `git diff-tree -M` on the oldest commit of each path to see
    whether it arrived by rename: `git log --follow` was measured at 4-30 s per file on this repo's 13k commits."""
    out, cur_rev, cur = [], rev, rel
    for _ in range(max_renames + 1):
        rc, o = _git(root, ["log", "--format=%H %cI", cur_rev, "--", cur], timeout=60)
        if rc != 0:
            return out
        entries = []
        for ln in o.decode("utf-8", "replace").splitlines():
            parts = ln.split()
            if len(parts) == 2:
                entries.append((parts[0], parts[1][:10], cur))
        if not entries:
            return out
        out += entries
        rc, o = _git(root, ["diff-tree", "-r", "-M", "--root", "--name-status", "--no-commit-id", entries[-1][0]])
        old = None
        if rc == 0:
            for ln in o.decode("utf-8", "replace").splitlines():
                f = ln.split("\t")
                if len(f) == 3 and f[0].startswith("R") and f[2] == cur:
                    old = f[1]
        if old is None:
            return out
        cur_rev, cur = entries[-1][0] + "^", old
    return out


def history_state(root, rev, rel, name_rel=None):
    """('new', None, None) | ('old', text_before_the_cutoff | None, path_then). 'new': the file (followed through renames,
    file_history)
    or its file-name date is dated on or after GRANDFATHER_DATE, the LATER of the two. 'old': it existed before the cutoff;
    text_before_the_cutoff is its newest version dated before the cutoff, None when git cannot say (never guessed)."""
    hist = file_history(root, rev, rel)
    nd = name_date(name_rel or rel) or ""            # the file's CURRENT name: a rename into a post-cutoff date is new
    added = hist[-1][1] if hist else ""
    eff = max(added, nd)
    if eff and eff >= GRANDFATHER_DATE:
        return "new", None, None
    if not hist:
        return "old", None, None
    before = next((h for h in hist if h[1] < GRANDFATHER_DATE), hist[-1])
    raw = blob_at(root, before[0], before[2])
    return "old", (raw.decode("utf-8", "replace") if raw is not None else None), before[2]


def decide(root, rev, rel, text, read_blob, runbook_text, force_new=False, history_rev=None, history_rel=None):
    """(action, why, problems) for ONE prereg file. action: 'skip' (not judged: not a flip, or grandfathered) | 'judged'.
    force_new: the file is ADDED by the commit being made. history_rev / history_rel: where to follow the history of a
    modified or renamed file (the parent commit and the OLD path); default rev / rel."""
    kind, why = classify(rel, text)
    if force_new:
        state, old, old_path = "new", None, None
    else:
        state, old, old_path = history_state(root, history_rev or rev, history_rel or rel, name_rel=rel)
    if state == "old":
        if old is None:
            return "skip", "predates %s (grandfathered: history unreadable, not guessed)" % GRANDFATHER_DATE, []
        if is_flip(old_path or rel, old)[0]:
            return "skip", "was a flip before %s (grandfathered)" % GRANDFATHER_DATE, []
        if kind != "flip":
            return "skip", "predates %s and is no flip" % GRANDFATHER_DATE, []
        why = "became a flip after %s (%s)" % (GRANDFATHER_DATE, why)
    elif kind == "undeclared":
        if not PREREG_RE.match(rel):
            return "skip", "not a pre-registration path", []
        return "judged", why, ["%s is a prereg first added on or after %s with no flip declaration (%s). Add `flip: yes` "
                               "(a default-on candidate; it then needs the smoke records) or `flip: no <reason of at "
                               "least %d characters>` (it flips no default). RULING smoke-first-1018."
                               % (rel, GRANDFATHER_DATE, why, MIN_REASON)]
    elif kind == "nonflip":
        return "skip", why, []
    return "judged", why, judge_prereg(rel, text, read_blob, runbook_text)


def read_runbook(root, rev=None):
    """The runbook as MAIN holds it (Claude writes rulings on main): never the working tree or the index of the checkout
    that is committing, so a prereg cannot carry its own ruling. Falls back to origin/main, then `rev`, then HEAD."""
    for ref in ("refs/heads/main", "refs/remotes/origin/main", rev or "HEAD", "HEAD"):
        raw = blob_at(root, ref, RUNBOOK_REL)
        if raw is not None:
            return raw.decode("utf-8", "replace")
    return ""


# ---------------------------------------------------------------- queue time
def check_queue(prereg, revision, root=None):
    """(ok, reason) for the prereg named by a gate-seed queue line, at the line's pinned revision. Fails open (ok) only
    when git cannot read the file at the revision at all; every determinate violation is a refusal."""
    root = root or ROOT
    raw = blob_at(root, revision, prereg)
    if raw is None:
        return True, "smoke-first: %s unreadable at %s, not checked" % (prereg, revision)
    text = raw.decode("utf-8", "replace")
    action, why, probs = decide(root, revision, prereg, text, lambda rel: blob_at(root, revision, rel),
                                read_runbook(root, revision))
    if probs:
        return False, ("prereg %s (%s) fails smoke-first (RULING smoke-first-1018): %s" % (prereg, why, " | ".join(probs)))
    return True, "smoke-first: %s %s" % (prereg, why if action == "skip" else "smoke records ok")


# ---------------------------------------------------------------- CLI
def _scan(root, with_state=True):
    """[(rel, why, name_date, state)] for every flip prereg of the working tree. The state needs git history (about a second
    per file); with_state=False skips it ('?')."""
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
                if with_state:
                    state, old, old_path = history_state(root, "HEAD", rel)
                    gf = state == "old" and (old is None or is_flip(old_path or rel, old)[0])
                    rows.append((rel, why, name_date(rel), "grandfathered" if gf else "CHECKED"))
                else:
                    rows.append((rel, why, name_date(rel), "?"))
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
        for rel, why, d, state in rows:
            bad += state == "CHECKED"
            print("%-13s %-10s %s  (%s)" % (state, d or "no-date", rel, why))
        print("%d flip prereg(s) in the tree; %d would be checked (new on or after %s)" % (len(rows), bad, GRANDFATHER_DATE))
        return 0
    ok, why = check_queue(a.prereg, a.revision)
    print(("ok: " if ok else "REFUSED: ") + why, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
