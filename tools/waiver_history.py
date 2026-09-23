#!/usr/bin/env python3
"""waiver_history.py — shared CLASSIFICATION + append-only BUDGET for the idle-compute escape hatches
(research/queue/.parallel_compute_waiver, read by tools/gates/compute_idle_persistent.py; research/queue/
.lane_waiver, read by tools/gates/lane_starvation.py).

THE LOOPHOLE THIS CLOSES (owner, 2026-09-23). `tools/parallel_audit.py` printed "⛔ UNDER-PARALLELIZED" for
~14.5 days straight, and neither blocking gate ever actually fired, because their waiver files were rewritten
each cycle with PROMISES -- "I will run the pool fanout as the dedicated NEXT step immediately after
harvesting..." -- that were not kept. The existing `_RATIONALISATION` regex in both gates already rejects a
PRIORITY/FOCUS excuse ("focused on the crux"), but a promise is a DIFFERENT shape of abuse: it names no
resource constraint at all, so it slid straight past a check built to catch prioritization language. Three
independent closes, all in this module so the two gates cannot drift on them:

 (1) CLASSIFY. A waiver must declare `CLASS: <one of VALID_CLASSES>` -- a fixed, resource-shaped vocabulary,
     not free prose. GAMING and OWNER-PAUSE are the pre-existing accepted-risk carve-outs (the owner's game
     time, or compute the owner explicitly reserved) and stay budget-EXEMPT below. RAM-CONTENTION and
     NO-READY-WORK are the two genuine per-lane blockers the old gates already asked for in their message
     text ("no ready de-risk for the pool: the cache build is blocked on X") -- now each must carry the
     MEASURED evidence (`avail_gb=` / `checked=`) that makes it falsifiable, not just asserted. Any waiver
     containing promise/intent language (will / next step / after / then / soon / going to / about to /
     shortly) is REJECTED regardless of its declared class -- a plan is not a resource constraint.
 (2) BUDGET. Every waiver READ (valid or not) is appended to the append-only `research/queue/
     .waiver_history.jsonl`. A non-GAMING/OWNER-PAUSE waiver is refused once the cumulative declared
     `waiver_max_h` of PRIOR valid non-exempt waivers in the trailing WAIVER_BUDGET_WINDOW_H hours already
     reaches WAIVER_BUDGET_H -- so even a genuinely-classified, evidence-carrying waiver cannot silently
     become a standing suspension by being renewed forever; eventually the resource has to run.
 (3) SURFACE. `describe()` renders "<CLASS> (age <N>m)" for `tools/parallel_audit.py` to print each cycle, so
     the escape hatch being OPEN is visible on the same line as the stall it is excusing, not just silent
     until a commit is attempted.

Dedup: `evaluate()` only appends a NEW history row when the waiver file's mtime differs from the last row this
(gate, path) pair recorded -- repeated `check()` calls against an UNCHANGED waiver file (every commit re-reads
it) must not inflate the budget; only a genuine rewrite (a renewal) counts as new consumption.
"""
from __future__ import annotations

import json
import os
import re
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(_ROOT, "research", "queue", ".waiver_history.jsonl")

# GAMING / OWNER-PAUSE: the owner's pre-existing accepted risk (compute_idle_persistent's own GAME_MODE
# carve-out is handled upstream in parallel_audit.py; this class exists for the cases the waiver FILE itself
# is the record, e.g. lane_starvation has no GAME_MODE branch of its own). RAM-CONTENTION / NO-READY-WORK are
# the two genuine per-lane blockers both gates' own messages already pointed at.
VALID_CLASSES = ("GAMING", "OWNER-PAUSE", "RAM-CONTENTION", "NO-READY-WORK")
BUDGET_EXEMPT_CLASSES = ("GAMING", "OWNER-PAUSE")

WAIVER_BUDGET_H = 6.0
WAIVER_BUDGET_WINDOW_H = 24.0

_CLASS_RE = re.compile(r"^\s*CLASS\s*[:=]\s*([A-Za-z-]+)\s*$", re.M)
# A promise/intent verb names a FUTURE action, never a CURRENT resource constraint -- the exact shape of the
# 14.5-day loophole ("will run the pool fanout ... immediately after harvesting ... as the dedicated next step").
_PROMISE_RE = re.compile(r"\b(will|next step|after\b|then\b|soon|going to|about to|shortly|shall)\b", re.I)
_AVAIL_GB_RE = re.compile(r"\bavail(?:_gb)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_CHECKED_RE = re.compile(r"\bchecked\s*[:=]\s*(.+)", re.I)


def parse_waiver(text):
    """Classify raw waiver-file text. Returns a dict:
      valid waiver  -> {"class": <CLASS>, "reason": <text, truncated>, "avail_gb"?: float, "checked"?: str}
      invalid       -> {"error": "<why, human-readable>"}
    Never raises -- a malformed waiver is data, not a crash."""
    text = (text or "").strip()
    if not text:
        return {"error": "empty waiver file"}
    m = _CLASS_RE.search(text)
    if not m:
        return {"error": "no 'CLASS: <%s>' line -- free-prose waivers are no longer accepted (the "
                          "2026-09-23 promise-language loophole)" % "|".join(VALID_CLASSES)}
    cls = m.group(1).upper()
    if cls not in VALID_CLASSES:
        return {"error": "CLASS %r is not one of %s" % (cls, ", ".join(VALID_CLASSES))}
    pm = _PROMISE_RE.search(text)
    if pm:
        return {"error": "promise/intent language %r detected -- REJECTED (a waiver states the CURRENT "
                          "resource constraint, never a plan to fill it later)" % pm.group(0)}
    out = {"class": cls, "reason": text[:400]}
    if cls == "RAM-CONTENTION":
        m2 = _AVAIL_GB_RE.search(text)
        if not m2:
            return {"error": "CLASS: RAM-CONTENTION must include the MEASURED 'avail_gb=<N>' at write time "
                              "(e.g. from `free -g`)"}
        out["avail_gb"] = float(m2.group(1))
    elif cls == "NO-READY-WORK":
        m3 = _CHECKED_RE.search(text)
        if not m3 or not m3.group(1).strip():
            return {"error": "CLASS: NO-READY-WORK must include 'checked=<what was checked>' (the concrete "
                              "search that came up empty, not an assertion)"}
        out["checked"] = m3.group(1).strip()[:300]
    return out


def _read_file(path, max_age_h, now_ts=None):
    """(text, mtime, age_h) for an existing, non-expired waiver file, else None."""
    if not os.path.exists(path):
        return None
    now_ts = now_ts if now_ts is not None else time.time()
    mtime = os.path.getmtime(path)
    age_h = (now_ts - mtime) / 3600.0
    if age_h > max_age_h:
        return None
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return None
    return text, mtime, age_h


def _last_recorded_mtime(gate_name, path, history_file):
    """The mtime of the most recent history row for this (gate, path), or None. Best-effort tail read (the
    file is append-only and small in practice); never raises."""
    if not os.path.exists(history_file):
        return None
    try:
        lines = open(history_file, errors="ignore").read().splitlines()[-500:]
    except OSError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("gate") == gate_name and row.get("path") == path:
            return row.get("mtime")
    return None


def record(gate_name, path, parsed, waiver_max_h, mtime, now_ts=None, history_file=HISTORY_FILE):
    """Append ONE row to the append-only history. Best-effort only -- a history-write failure must never
    block or crash the gate calling it (the same discipline parallel_state.persist uses)."""
    now_ts = now_ts if now_ts is not None else time.time()
    row = {
        "ts": now_ts, "gate": gate_name, "path": path, "mtime": mtime,
        "class": parsed.get("class"), "error": parsed.get("error"),
        "waiver_max_h": waiver_max_h,
    }
    try:
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def cumulative_waived_hours(window_h=WAIVER_BUDGET_WINDOW_H, now_ts=None,
                             exclude_classes=BUDGET_EXEMPT_CLASSES, history_file=HISTORY_FILE):
    """Sum of `waiver_max_h` for VALID, non-exempt-class history rows within the trailing `window_h` hours,
    GLOBAL across every gate (compute-idle-persistent and lane-starvation share one budget on purpose -- both
    are excuses for not launching ready work, and a per-gate budget would let renewing across the TWO waiver
    files double the real allowance). Missing/corrupt history -> 0.0 (fails OPEN: an absent history file must
    not itself become a block, the same staleness discipline `parallel_state.is_fresh` uses)."""
    now_ts = now_ts if now_ts is not None else time.time()
    if not os.path.exists(history_file):
        return 0.0
    cutoff = now_ts - window_h * 3600.0
    total = 0.0
    try:
        lines = open(history_file, errors="ignore").read().splitlines()
    except OSError:
        return 0.0
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("error"):
            continue                                    # a rejected waiver consumes no budget
        if row.get("ts", 0) < cutoff:
            continue
        if row.get("class") in exclude_classes:
            continue
        total += float(row.get("waiver_max_h") or 0.0)
    return total


def evaluate(gate_name, path, waiver_max_h, now_ts=None, budget_h=WAIVER_BUDGET_H,
             window_h=WAIVER_BUDGET_WINDOW_H, prior_cumulative_h=None, history_file=HISTORY_FILE):
    """The one call both gates make. Returns:
      {"active": False}                                                    -- no live waiver (missing/expired)
      {"active": True, "ok": False, "class": <or None>, "reject_reason": <str>, "age_h": float}
      {"active": True, "ok": True,  "class": <CLASS>, "age_h": float, "budget_h": <cumulative incl. this one>}

    `prior_cumulative_h` lets a caller/test inject the pre-this-waiver budget total directly (pure, no history
    file needed); omitted -> computed live from HISTORY_FILE. History is written (deduped by mtime) for every
    call that finds a live waiver file, valid or not, per the module docstring's point (2)."""
    now_ts = now_ts if now_ts is not None else time.time()
    found = _read_file(path, waiver_max_h, now_ts=now_ts)
    if found is None:
        return {"active": False}
    text, mtime, age_h = found
    parsed = parse_waiver(text)
    if mtime != _last_recorded_mtime(gate_name, path, history_file):
        record(gate_name, path, parsed, waiver_max_h, mtime, now_ts, history_file=history_file)
    if parsed.get("error"):
        return {"active": True, "ok": False, "class": None, "reject_reason": parsed["error"], "age_h": age_h}
    cls = parsed["class"]
    if cls in BUDGET_EXEMPT_CLASSES:
        return {"active": True, "ok": True, "class": cls, "age_h": age_h, "budget_h": None}
    prior = (cumulative_waived_hours(window_h=window_h, now_ts=now_ts, history_file=history_file)
             if prior_cumulative_h is None else prior_cumulative_h)
    if prior >= budget_h:
        return {"active": True, "ok": False, "class": cls,
                "reject_reason": "the %.1fh renewal budget for non-GAMING/OWNER-PAUSE waivers in the last "
                                  "%dh is EXHAUSTED (%.1fh already used, %s) -- a real resource constraint "
                                  "does not renew indefinitely; let the resource run, or waive GAMING/"
                                  "OWNER-PAUSE if that is the actual reason"
                                  % (budget_h, int(window_h), prior, parsed.get("reason", "")[:80]),
                "age_h": age_h}
    return {"active": True, "ok": True, "class": cls, "age_h": age_h, "budget_h": prior + waiver_max_h}


def describe(verdict):
    """One-line human string for parallel_audit.py's per-cycle surfacing. Empty string when no live waiver."""
    if not verdict.get("active"):
        return ""
    mins = int(verdict.get("age_h", 0.0) * 60)
    if verdict.get("ok"):
        return "waiver=%s (age %dm)" % (verdict.get("class"), mins)
    return "waiver=INVALID (age %dm): %s" % (mins, verdict.get("reject_reason", "")[:100])


def selftest():
    """FAILING DIRECTION FIRST: promise language and a missing CLASS must both be REJECTED."""
    bad = []
    if not parse_waiver("I will run the pool fanout as the dedicated next step after harvesting").get("error"):
        bad.append("did NOT reject promise/intent language with no CLASS at all")
    if not parse_waiver("CLASS: OWNER-PAUSE\nI will free this up soon").get("error"):
        bad.append("did NOT reject promise language even under a valid CLASS")
    if not parse_waiver("CLASS: NOTACLASS\nreason").get("error"):
        bad.append("did NOT reject an unknown CLASS token")
    if not parse_waiver("CLASS: RAM-CONTENTION\nreason: training is using it all").get("error"):
        bad.append("did NOT require avail_gb= on a RAM-CONTENTION waiver")
    if not parse_waiver("CLASS: NO-READY-WORK\nreason: nothing to run").get("error"):
        bad.append("did NOT require checked= on a NO-READY-WORK waiver")
    # NEGATIVES — genuinely well-formed waivers must NOT be rejected.
    ok1 = parse_waiver("CLASS: OWNER-PAUSE\nreason: owner reserved the GPU for a demo")
    if ok1.get("error"):
        bad.append("FALSE POSITIVE: rejected a well-formed OWNER-PAUSE waiver (%s)" % ok1.get("error"))
    ok2 = parse_waiver("CLASS: RAM-CONTENTION\navail_gb=4\nreason: training + desktop leave no margin")
    if ok2.get("error") or ok2.get("avail_gb") != 4.0:
        bad.append("FALSE POSITIVE/parse bug: a well-formed RAM-CONTENTION waiver was rejected or mis-parsed")
    ok3 = parse_waiver("CLASS: NO-READY-WORK\nchecked=lane_check.py, vikunja board, pool.queue -- all empty")
    if ok3.get("error"):
        bad.append("FALSE POSITIVE: rejected a well-formed NO-READY-WORK waiver")

    # evaluate(): exercise via a real temp file + a temp history file (NEVER the real
    # research/queue/.waiver_history.jsonl -- a selftest run must not pollute the real budget ledger) so
    # mtime-dedup + the budget check run end-to-end.
    import tempfile
    now = 3_000_000.0
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver")
        hist = os.path.join(td, "history.jsonl")
        with open(p, "w") as fh:
            fh.write("CLASS: NO-READY-WORK\nchecked=lane_check.py -- all 5 lanes served, pool.queue empty")
        os.utime(p, (now, now))
        # budget already exhausted -> a syntactically-valid non-exempt waiver must still be REJECTED.
        v_over = evaluate("selftest-gate", p, waiver_max_h=6, now_ts=now, budget_h=6.0,
                          prior_cumulative_h=6.0, history_file=hist)
        if v_over.get("ok"):
            bad.append("did NOT enforce the renewal budget: a NO-READY-WORK waiver passed with the 24h "
                       "budget already exhausted")
        # budget has room -> the SAME waiver must be accepted.
        v_under = evaluate("selftest-gate", p, waiver_max_h=6, now_ts=now, budget_h=6.0,
                           prior_cumulative_h=0.0, history_file=hist)
        if not v_under.get("ok"):
            bad.append("FALSE POSITIVE: a well-formed waiver with budget room was rejected (%s)"
                      % v_under.get("reject_reason"))
        # GAMING/OWNER-PAUSE must be budget-EXEMPT even with the budget nominally exhausted.
        with open(p, "w") as fh:
            fh.write("CLASS: OWNER-PAUSE\nreason: owner reserved the box")
        os.utime(p, (now + 1, now + 1))
        v_exempt = evaluate("selftest-gate", p, waiver_max_h=6, now_ts=now + 1, budget_h=6.0,
                            prior_cumulative_h=999.0, history_file=hist)
        if not v_exempt.get("ok"):
            bad.append("did NOT keep OWNER-PAUSE budget-exempt (the owner-gaming case must stay intact)")
        # missing/expired file -> inactive, never blocks.
        if evaluate("selftest-gate", os.path.join(td, "nope"), waiver_max_h=6, now_ts=now,
                    history_file=hist).get("active"):
            bad.append("FALSE POSITIVE: a MISSING waiver file read as active")
        # mtime-dedup: re-evaluating the SAME unchanged waiver twice must append only ONE history row.
        with open(p, "w") as fh:
            fh.write("CLASS: NO-READY-WORK\nchecked=dedup probe")
        os.utime(p, (now + 2, now + 2))
        evaluate("dedup-gate", p, waiver_max_h=6, now_ts=now + 2, history_file=hist)
        evaluate("dedup-gate", p, waiver_max_h=6, now_ts=now + 2.01, history_file=hist)
        n_dedup_rows = sum(1 for ln in open(hist).read().splitlines()
                           if json.loads(ln).get("gate") == "dedup-gate")
        if n_dedup_rows != 1:
            bad.append("mtime-dedup broken: an UNCHANGED waiver file produced %d history rows, not 1 "
                      "(would inflate the budget every commit)" % n_dedup_rows)
    return bad
