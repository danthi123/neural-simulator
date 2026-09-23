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
     not free prose. GAMING and OWNER-PAUSE are the owner's accepted-risk carve-outs and are budget-EXEMPT
     ONLY while the owner's own sentinel exists (see `_owner_reservation_evidence`). RAM-CONTENTION and
     NO-READY-WORK are the two genuine per-lane blockers the old gates already asked for in their message
     text -- each must carry the MEASURED evidence (`avail_gb=` / `checked=`) that makes it falsifiable.
     Promise/intent language and the old priority/focus vocabulary are REJECTED under every class.
 (2) BUDGET. Every waiver read is appended to the append-only `research/queue/.waiver_history.jsonl`. The
     budget charges each waiver EPISODE (one file path + one content hash + one mtime) ONCE, for the REAL
     time it was live -- never once per read, and never its declared cap. A non-exempt waiver is refused once
     the charged time of all episodes in the trailing WAIVER_BUDGET_WINDOW_H hours reaches WAIVER_BUDGET_H.
 (3) SURFACE. `describe()` renders "<CLASS> (age <N>m)" for `tools/parallel_audit.py` to print each cycle.

EPISODE ACCOUNTING (fix round 3, 2026-09-23). Fix round 2 moved an off-by-one instead of fixing it: every read
of an UNCHANGED waiver after the first summed that waiver's own earlier row (charged at its full 6h cap)
against it, so with WAIVER_MAX_H == WAIVER_BUDGET_H == 6 a genuine waiver was honoured on ONE read and then
rejected for the rest of its life. Charging real elapsed time removes the cap-vs-budget coincidence entirely:
  * an episode is keyed by (realpath, mtime, sha256 of content). Gate NAME is recorded but is NOT part of the
    key, so the heartbeat and the commit gate -- or any label -- reading one file cannot double-charge it;
  * an episode is charged from its start (its mtime) to the LAST moment it is known live: the next row on the
    same path (a rewrite supersedes it at the new file's mtime; a read that finds the file gone ends it), or
    the current read if it is being evaluated now; never past start + its waiver_max_h;
  * the time between two reads is charged only if the LATER read accepted it as non-exempt, so exempt
    (sentinel-backed) and rejected stretches cost nothing;
  * overlapping episodes on one path are UNIONED; different paths (the two waiver files) are SUMMED, because
    both waivers are excuses for not launching ready work and one global budget is the point.
So rewriting the file every 10 minutes is charged the CONTINUOUS time it was live (the renewal loophole the
budget exists for stays closed), while one genuine waiver read 100 times costs exactly its elapsed time.
A FUTURE mtime (`touch -d '+1 year'`) used to give a negative age that never expired -- it is now rejected.

Fails OPEN on a missing/corrupt history file (an absent ledger must not itself brick commits -- the same
staleness discipline `parallel_state.is_fresh` uses). Known residual: an agent that deletes the history file
resets the budget; the history is not tamper-evident.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shared_root():
    """The checkout whose persistent dispatchers/heartbeat actually consume the queues -- i.e. the repo's
    git-common-dir-based root, NOT necessarily `_ROOT` (this file's own worktree). Mirrors
    `gates/lane_starvation._shared_queue_root()` verbatim (SIM_QUEUE_ROOT override, then git-common-dir, then
    `_ROOT` as a last resort) so every waiver file, the history, the owner's GAME_MODE/GPU_PAUSE sentinels and
    `parallel_state.STATE_FILE` resolve to the SAME physical path from every worktree (`tools/gpu_queue.sh`
    resolves its own QDIR the same way)."""
    override = os.environ.get("SIM_QUEUE_ROOT")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=_ROOT, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        if common:
            return os.path.dirname(common)
    except Exception:
        pass
    return _ROOT


HISTORY_FILE = os.path.join(shared_root(), "research", "queue", ".waiver_history.jsonl")

VALID_CLASSES = ("GAMING", "OWNER-PAUSE", "RAM-CONTENTION", "NO-READY-WORK")
BUDGET_EXEMPT_CLASSES = ("GAMING", "OWNER-PAUSE")

WAIVER_BUDGET_H = 6.0
WAIVER_BUDGET_WINDOW_H = 24.0
_DEFAULT_MAX_H = 6.0
# mtime may lead the reader's clock by this much (clock skew, a write racing the read) before it is treated as
# a forged future date.
_FUTURE_SKEW_S = 300.0
# A repeat read of the same episode within this many seconds adds no history row (the budget is computed from
# the episode's timeline, so this bounds file growth without changing any charge by more than this).
_OBS_THROTTLE_S = 60.0

# Sentinels ONLY the owner's own tools write: `tools/game.sh on` writes GAME_MODE (and calls `gpu_queue.sh
# pause --now`, which writes GPU_PAUSE); both live under the SHARED checkout's research/queue. FIX ROUND 3: the
# fix-2 process-name heuristic (`proton|steam|...` anywhere in `ps -eo args`) is REMOVED -- this machine runs
# `python3 -m proton.vpn.daemon` permanently and idles the Steam client, so it made a self-declared OWNER-PAUSE
# budget-exempt at all times with no owner action behind it. A background process is not a reservation.
_GAME_SENTINELS = ("GAME_MODE", "GPU_PAUSE")

_CLASS_RE = re.compile(r"^\s*CLASS\s*[:=]\s*([A-Za-z-]+)\s*$", re.M)
# A promise/intent verb names a FUTURE action, never a CURRENT resource constraint -- the exact shape of the
# 14.5-day loophole ("will run the pool fanout ... immediately after harvesting ... as the dedicated next step").
_PROMISE_RE = re.compile(r"\b(will|next step|after\b|then\b|soon|going to|about to|shortly|shall)\b", re.I)
# The ORIGINAL rationalisation vocabulary (pre-refactor gates/lane_starvation._RATIONALISATION, 2026-07-31/
# 2026-08-01 abuse: "saturated with the gap#4 crux; deprioritized behind the crux"). Runs under EVERY class.
_RATIONALISATION_RE = re.compile(r"crux|priorit|focus|deprioriti|momentum|behind the|saturated with", re.I)
_AVAIL_GB_RE = re.compile(r"\bavail(?:_gb)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_CHECKED_RE = re.compile(r"\bchecked\s*[:=]\s*(.+)", re.I)


def _owner_reservation_evidence(root=None):
    """True only if the owner's GAME_MODE or GPU_PAUSE sentinel exists under the SHARED checkout's
    research/queue (where `tools/game.sh` / `tools/gpu_queue.sh pause` put it) -- never the evaluating
    worktree's own copy, and never anything read from the process list or the waiver text. `root` is
    injectable for tests; production callers pass nothing."""
    root = root or shared_root()
    return any(os.path.exists(os.path.join(root, "research", "queue", n)) for n in _GAME_SENTINELS)


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
    rm = _RATIONALISATION_RE.search(text)
    if rm:
        return {"error": "rationalisation language %r detected -- REJECTED (a waiver states the CURRENT "
                          "resource constraint, never a priority/focus excuse -- this is checked under every "
                          "CLASS, not just free prose with no CLASS line)" % rm.group(0)}
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


# ── history ledger ─────────────────────────────────────────────────────────────────────────────────────────

def _load_rows(history_file):
    if not os.path.exists(history_file):
        return []
    try:
        lines = open(history_file, errors="ignore").read().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("path") is not None and row.get("ts") is not None:
            rows.append(row)
    return rows


def _append(row, history_file):
    """Best-effort append -- a history-write failure must never block or crash the gate calling it."""
    try:
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def _present(row):
    return row.get("present", True) is not False


def _charged(row, exclude_classes=BUDGET_EXEMPT_CLASSES):
    """Did this observation ACCEPT the waiver as a non-exempt excuse? New rows carry `charged`; rows written
    by the fix-1/fix-2 code are interpreted from their error/exempt/class fields."""
    if not _present(row):
        return False
    if "charged" in row:
        return bool(row["charged"])
    if row.get("error"):
        return False
    exempt = row.get("exempt")
    if exempt is None:
        exempt = row.get("class") in exclude_classes
    return not exempt


def _ekey(row):
    return (row.get("mtime"), row.get("sha") or "")


def _charged_intervals_for_path(rows, now_ts, exclude_classes):
    """Charged [a, b] intervals for ONE path's rows (sorted by ts). See the module docstring for the rule."""
    out = []
    first_seen = {}
    for r in rows:
        if _present(r):
            first_seen.setdefault(_ekey(r), float(r["ts"]))

    def _bounds(r):
        start = min(float(r.get("mtime") or r["ts"]), first_seen[_ekey(r)])
        cap = start + 3600.0 * float(r.get("waiver_max_h") or _DEFAULT_MAX_H)
        return start, cap

    for i, r in enumerate(rows):
        if not _present(r):
            continue
        start, cap = _bounds(r)
        ts = float(r["ts"])
        prev = rows[i - 1] if i > 0 else None
        lead_from = start
        if prev is not None and _present(prev) and _ekey(prev) == _ekey(r):
            lead_from = float(prev["ts"])                     # the stretch since the last read of this episode
        if _charged(r, exclude_classes):
            out.append((max(lead_from, start), min(ts, cap)))
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if nxt is not None and (not _present(nxt) or _ekey(nxt) != _ekey(r)) and _charged(r, exclude_classes):
            n_ts = float(nxt["ts"])
            end = n_ts if not _present(nxt) else min(max(float(nxt.get("mtime") or n_ts), ts), n_ts)
            out.append((ts, min(end, cap)))                   # live until superseded / observed gone
    return [(a, b) for a, b in out if b > a]


def _union_len(intervals, lo, hi):
    clipped = sorted((max(a, lo), min(b, hi)) for a, b in intervals if min(b, hi) > max(a, lo))
    total, cur_a, cur_b = 0.0, None, None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def cumulative_waived_hours(window_h=WAIVER_BUDGET_WINDOW_H, now_ts=None,
                             exclude_classes=BUDGET_EXEMPT_CLASSES, history_file=HISTORY_FILE, live_row=None):
    """Charged waiver-hours inside the trailing `window_h` window, GLOBAL across both gates: per path the union
    of each episode's charged intervals, summed over paths. `live_row` is the observation being evaluated right
    now (not yet written), so a caller gets the budget INCLUDING the current episode's elapsed time without
    writing first. Missing/corrupt history -> only the live row counts (fails OPEN)."""
    now_ts = now_ts if now_ts is not None else time.time()
    rows = _load_rows(history_file)
    if live_row is not None:
        rows.append(live_row)
    by_path = {}
    for r in rows:
        by_path.setdefault(r["path"], []).append(r)
    lo = now_ts - window_h * 3600.0
    total_s = 0.0
    for prs in by_path.values():
        prs.sort(key=lambda r: float(r["ts"]))
        total_s += _union_len(_charged_intervals_for_path(prs, now_ts, exclude_classes), lo, now_ts)
    return total_s / 3600.0


def _last_row_for_path(rows, key_path):
    for r in reversed(rows):
        if r["path"] == key_path:
            return r
    return None


def _record_absence(key_path, now_ts, history_file):
    """A read that finds the file gone (or expired) CLOSES the open episode -- written only on the transition,
    so an absent waiver read every heartbeat adds one row, not one per read."""
    last = _last_row_for_path(_load_rows(history_file), key_path)
    if last is not None and _present(last):
        _append({"ts": now_ts, "path": key_path, "present": False}, history_file)


def evaluate(gate_name, path, waiver_max_h, now_ts=None, budget_h=WAIVER_BUDGET_H,
             window_h=WAIVER_BUDGET_WINDOW_H, prior_cumulative_h=None, history_file=HISTORY_FILE,
             evidence_check=None):
    """The one call both gates (and the heartbeat) make. Returns:
      {"active": False}                                                    -- no live waiver (missing/expired)
      {"active": True, "ok": False, "class": <or None>, "reject_reason": <str>, "age_h": float}
      {"active": True, "ok": True, "class": <CLASS>, "age_h": float, "budget_h": <charged h incl. this one>}
    (`budget_h` is None for a sentinel-backed exempt waiver.)

    `prior_cumulative_h` injects the charge of all OTHER episodes (tests); the current episode's own elapsed
    time is always added. `evidence_check` injects the owner-reservation check (tests); default is
    `_owner_reservation_evidence()` -- sentinel-only, shared root."""
    now_ts = now_ts if now_ts is not None else time.time()
    key_path = os.path.realpath(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _record_absence(key_path, now_ts, history_file)
        return {"active": False}
    age_h = (now_ts - mtime) / 3600.0
    if age_h > waiver_max_h:
        _record_absence(key_path, now_ts, history_file)
        return {"active": False}
    try:
        raw = open(path, "rb").read()
    except OSError:
        return {"active": False}
    text = raw.decode("utf-8", errors="ignore")
    sha = hashlib.sha256(raw).hexdigest()[:16]

    if mtime - now_ts > _FUTURE_SKEW_S:
        parsed = {"error": "the waiver's mtime is %.1fh in the FUTURE -- a forged date would never expire "
                           "(and would be charged a negative duration); REJECTED" % (-age_h)}
    else:
        parsed = parse_waiver(text)

    row = {"ts": now_ts, "gate": gate_name, "path": key_path, "mtime": mtime, "sha": sha,
           "class": parsed.get("class"), "error": parsed.get("error"), "waiver_max_h": waiver_max_h,
           "present": True, "exempt": False, "charged": False}
    verdict = None
    if parsed.get("error"):
        verdict = {"active": True, "ok": False, "class": None, "reject_reason": parsed["error"], "age_h": age_h}
    else:
        cls = parsed["class"]
        is_exempt = False
        if cls in BUDGET_EXEMPT_CLASSES:
            try:
                is_exempt = bool((evidence_check or _owner_reservation_evidence)())
            except Exception:
                is_exempt = False                  # fail CLOSED: no evidence == not exempt, never crash
        row["class"] = cls
        if is_exempt:
            row["exempt"] = True
            verdict = {"active": True, "ok": True, "class": cls, "age_h": age_h, "budget_h": None}
        else:
            live = dict(row, charged=True)
            if prior_cumulative_h is None:
                used = cumulative_waived_hours(window_h=window_h, now_ts=now_ts, history_file=history_file,
                                               live_row=live)
            else:
                start = min(mtime, now_ts)
                used = prior_cumulative_h + max(0.0, min(now_ts - start, waiver_max_h * 3600.0)) / 3600.0
            if used >= budget_h:
                reason = ("the %.1fh renewal budget for non-exempt waivers in the last %dh is EXHAUSTED "
                          "(%.2fh of live waiver time already charged, %s) -- a real resource constraint does "
                          "not renew indefinitely; let the resource run"
                          % (budget_h, int(window_h), used, parsed.get("reason", "")[:80]))
                if cls in BUDGET_EXEMPT_CLASSES:
                    reason = ("CLASS: %s claims an owner-reserved exemption but no GAME_MODE/GPU_PAUSE "
                              "sentinel exists under the shared research/queue (written by tools/game.sh / "
                              "tools/gpu_queue.sh pause) -- treated as a normal non-exempt waiver, and " % cls
                              ) + reason
                verdict = {"active": True, "ok": False, "class": cls, "reject_reason": reason, "age_h": age_h}
            else:
                row["charged"] = True
                verdict = {"active": True, "ok": True, "class": cls, "age_h": age_h, "budget_h": used}

    last = None
    for r in reversed(_load_rows(history_file)):
        if r["path"] == key_path:
            last = r
            break
    same_as_last = (last is not None and _present(last) and _ekey(last) == _ekey(row)
                    and bool(last.get("charged")) == row["charged"] and bool(last.get("exempt")) == row["exempt"]
                    and 0 <= now_ts - float(last["ts"]) < _OBS_THROTTLE_S)
    if not same_as_last:
        _append(row, history_file)
    return verdict


def describe(verdict):
    """One-line human string for parallel_audit.py's per-cycle surfacing. Empty string when no live waiver."""
    if not verdict.get("active"):
        return ""
    mins = int(verdict.get("age_h", 0.0) * 60)
    if verdict.get("ok"):
        return "waiver=%s (age %dm)" % (verdict.get("class"), mins)
    return "waiver=INVALID (age %dm): %s" % (mins, verdict.get("reject_reason", "")[:100])


def selftest():
    """FAILING DIRECTION FIRST: promise language and a missing CLASS must both be REJECTED; a budget that
    re-charges a re-read waiver, an escapable renewal budget, a future-dated waiver and a sentinel-free
    exemption must each be caught."""
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
    if not parse_waiver("CLASS: NO-READY-WORK\nchecked=focused on the crux, nothing else worth it").get("error"):
        bad.append("did NOT reject rationalisation language under a valid class")
    if parse_waiver("CLASS: RAM-CONTENTION\navail_gb=4\nreason: training + desktop leave no margin").get("error"):
        bad.append("FALSE POSITIVE: rejected a well-formed RAM-CONTENTION waiver")
    if parse_waiver("CLASS: NO-READY-WORK\nchecked=lane_check.py -- 5/5 lanes served, pool.queue empty").get("error"):
        bad.append("FALSE POSITIVE: rejected a well-formed NO-READY-WORK waiver")

    # evaluate(): a real temp file + temp history (NEVER the real ledger -- a selftest must not spend budget).
    import tempfile
    nrw = "CLASS: NO-READY-WORK\nchecked=lane_check.py -- all 5 lanes served, pool.queue empty"
    t0 = 3_000_000.0

    def _w(p, text, mt):
        with open(p, "w") as fh:
            fh.write(text)
        os.utime(p, (mt, mt))

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver")
        # (a) the fix-2 regression: the SAME unchanged waiver read repeatedly must stay valid (one episode).
        hist = os.path.join(td, "h1.jsonl")
        _w(p, nrw, t0)
        reads = [evaluate("selftest-gate", p, 6, now_ts=t0 + k * 900.0, history_file=hist) for k in range(8)]
        if not all(v.get("ok") for v in reads):
            bad.append("RE-READ DOUBLE-CHARGE: an unchanged waiver was rejected on a later read of the same "
                       "episode (fix-2's moved off-by-one)")
        used = cumulative_waived_hours(now_ts=t0 + 7 * 900.0, history_file=hist)
        if abs(used - 1.75) > 1e-6:
            bad.append("episode not charged its real elapsed time (expected 1.75h, got %.3f)" % used)
        # (b) FAILING DIRECTION: back-to-back renewals must hit the budget.
        hist = os.path.join(td, "h2.jsonl")
        caught = False
        for k in range(48):
            _w(p, nrw + "\nn=%d" % k, t0 + k * 600.0)
            if not evaluate("selftest-gate", p, 6, now_ts=t0 + k * 600.0, history_file=hist).get("ok"):
                caught = True
                break
        if not caught:
            bad.append("RENEWAL LOOPHOLE: rewriting the waiver every 10 min for 8h was never rejected")
        # (c) FAILING DIRECTION: a future-dated waiver must be rejected, not immortal.
        _w(p, nrw, t0 + 86400.0 * 365)
        if evaluate("selftest-gate", p, 6, now_ts=t0, history_file=os.path.join(td, "h3.jsonl")).get("ok"):
            bad.append("FUTURE-MTIME LOOPHOLE: a waiver dated a year ahead was accepted")
        # (d) exemption is sentinel-only.
        os.makedirs(os.path.join(td, "research", "queue"))
        if _owner_reservation_evidence(root=td):
            bad.append("owner-reservation evidence reported with NO sentinel present")
        _w(p, "CLASS: OWNER-PAUSE\nreason: owner reserved the box", t0)
        v_no = evaluate("selftest-gate", p, 6, now_ts=t0, prior_cumulative_h=6.0,
                        history_file=os.path.join(td, "h4.jsonl"), evidence_check=lambda: False)
        if v_no.get("ok"):
            bad.append("SELF-DECLARED EXEMPTION: an OWNER-PAUSE with no sentinel passed an exhausted budget")
        open(os.path.join(td, "research", "queue", "GAME_MODE"), "w").close()
        if not _owner_reservation_evidence(root=td):
            bad.append("FALSE NEGATIVE: the GAME_MODE sentinel was not recognised")
        v_yes = evaluate("selftest-gate", p, 6, now_ts=t0, prior_cumulative_h=999.0,
                         history_file=os.path.join(td, "h5.jsonl"), evidence_check=lambda: True)
        if not v_yes.get("ok") or v_yes.get("budget_h") is not None:
            bad.append("FALSE POSITIVE: a sentinel-backed OWNER-PAUSE was not budget-exempt")
        # (e) missing file -> inactive.
        if evaluate("selftest-gate", os.path.join(td, "nope"), 6, now_ts=t0,
                    history_file=os.path.join(td, "h6.jsonl")).get("active"):
            bad.append("FALSE POSITIVE: a MISSING waiver file read as active")
    return bad
