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
import subprocess
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shared_root():
    """The checkout whose persistent dispatchers/heartbeat actually consume the queues -- i.e. the repo's
    git-common-dir-based root, NOT necessarily `_ROOT` (this file's own worktree). Mirrors
    `gates/lane_starvation._shared_queue_root()` verbatim (SIM_QUEUE_ROOT override, then git-common-dir, then
    `_ROOT` as a last resort) so both waiver files resolve the SAME physical path from every worktree.

    REVIEW FIX (2026-09-23): `HISTORY_FILE` used to be pinned to `_ROOT` -- this module's OWN worktree -- so
    the budget was silently per-worktree, not the "GLOBAL across every gate" the docstring already claimed:
    an agent working in a fresh worktree read an empty history and got a fresh 6h budget every time, and two
    concurrent worktrees each independently believed they owned the whole budget. All git worktrees of one
    repo share ONE `.git` (git-common-dir) by design, so resolving through it gives every worktree the same
    physical history file without needing a hardcoded absolute path (`tools/parallel_audit.py`'s ROOT is
    hardcoded for this reason; this achieves the same result without hardcoding)."""
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

# GAMING / OWNER-PAUSE: the owner's pre-existing accepted risk (compute_idle_persistent's own GAME_MODE
# carve-out is handled upstream in parallel_audit.py; this class exists for the cases the waiver FILE itself
# is the record, e.g. lane_starvation has no GAME_MODE branch of its own). RAM-CONTENTION / NO-READY-WORK are
# the two genuine per-lane blockers both gates' own messages already pointed at.
VALID_CLASSES = ("GAMING", "OWNER-PAUSE", "RAM-CONTENTION", "NO-READY-WORK")
BUDGET_EXEMPT_CLASSES = ("GAMING", "OWNER-PAUSE")

WAIVER_BUDGET_H = 6.0
WAIVER_BUDGET_WINDOW_H = 24.0

# Sentinels an actual owner action creates -- `tools/game.sh on` writes GAME_MODE (and calls `gpu_queue.sh
# pause --now` underneath it, which writes GPU_PAUSE); the owner can also touch either by hand. Neither can be
# written by an agent's own waiver text.
_GAME_SENTINELS = ("GAME_MODE", "GPU_PAUSE")
# A real game/launcher process actually running is independent, falsifiable evidence too.
_GAME_PROCESS_RE = re.compile(r"\b(steam(?:\.sh)?|lutris|heroic|retroarch|proton|gamescope|steamwebhelper)\b",
                               re.I)

_CLASS_RE = re.compile(r"^\s*CLASS\s*[:=]\s*([A-Za-z-]+)\s*$", re.M)
# A promise/intent verb names a FUTURE action, never a CURRENT resource constraint -- the exact shape of the
# 14.5-day loophole ("will run the pool fanout ... immediately after harvesting ... as the dedicated next step").
_PROMISE_RE = re.compile(r"\b(will|next step|after\b|then\b|soon|going to|about to|shortly|shall)\b", re.I)
# The ORIGINAL rationalisation vocabulary (from the pre-refactor gates/lane_starvation._RATIONALISATION,
# 2026-07-31/2026-08-01 abuse: "saturated with the gap#4 crux; deprioritized behind the crux, not neglect").
# Classification alone (point 1 above) does not catch this on its own -- a `checked=`/`reason:` field under a
# perfectly valid CLASS can still smuggle a priority excuse ("CLASS: NO-READY-WORK\nchecked=focused on the
# crux, nothing else worth it") -- so this must run under EVERY class, not just free prose with no CLASS line.
_RATIONALISATION_RE = re.compile(r"crux|priorit|focus|deprioriti|momentum|behind the|saturated with", re.I)
_AVAIL_GB_RE = re.compile(r"\bavail(?:_gb)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_CHECKED_RE = re.compile(r"\bchecked\s*[:=]\s*(.+)", re.I)


def _owner_reservation_evidence(root=None, ps_text=None):
    """FALSIFIABLE evidence the owner actually reserved compute right NOW: a GAME_MODE/GPU_PAUSE sentinel file
    (written by `tools/game.sh on` / `tools/gpu_queue.sh pause`, or by the owner directly), or an actually-
    running game/launcher process. The 2026-09-23 loophole: GAMING/OWNER-PAUSE were budget-exempt on the
    strength of the waiver's OWN prose alone, so any agent could self-declare either class -- with no owner
    action behind it at all -- and get a free, unlimited-renewal exemption. The waiver text itself is never
    evidence; only something the agent cannot write is. `root`/`ps_text` are injectable for tests -- this must
    stay a pure function of its inputs so it can be exercised without a real GPU/desktop session."""
    root = root or _ROOT
    for name in _GAME_SENTINELS:
        if os.path.exists(os.path.join(root, "research", "queue", name)):
            return True
    if ps_text is None:
        try:
            ps_text = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True,
                                      timeout=10).stdout
        except Exception:
            ps_text = ""
    return bool(_GAME_PROCESS_RE.search(ps_text or ""))


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
    # Applies under EVERY class (2026-09-23 review fix): a valid CLASS + a required evidence field
    # (avail_gb=/checked=) does not exempt the free-text portions of the waiver from the same rationalisation
    # check free-prose waivers always faced.
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


def record(gate_name, path, parsed, waiver_max_h, mtime, now_ts=None, history_file=HISTORY_FILE, exempt=None):
    """Append ONE row to the append-only history. Best-effort only -- a history-write failure must never
    block or crash the gate calling it (the same discipline parallel_state.persist uses).

    `exempt` records whether THIS evaluation actually treated the row as budget-exempt (True/False) -- not
    merely whether its declared CLASS is one of BUDGET_EXEMPT_CLASSES. A GAMING/OWNER-PAUSE waiver with no
    owner-reservation evidence behind it is recorded with exempt=False, so `cumulative_waived_hours` charges it
    like any other class instead of re-deriving exemption from the class name alone (which would silently
    forget the evidence check ever ran). `exempt=None` for an error row is fine -- `cumulative_waived_hours`
    skips error rows before it ever looks at `exempt`."""
    now_ts = now_ts if now_ts is not None else time.time()
    row = {
        "ts": now_ts, "gate": gate_name, "path": path, "mtime": mtime,
        "class": parsed.get("class"), "error": parsed.get("error"),
        "waiver_max_h": waiver_max_h, "exempt": exempt,
    }
    try:
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def cumulative_waived_hours(window_h=WAIVER_BUDGET_WINDOW_H, now_ts=None,
                             exclude_classes=BUDGET_EXEMPT_CLASSES, history_file=HISTORY_FILE):
    """Sum of `waiver_max_h` for VALID, non-exempt history rows within the trailing `window_h` hours, GLOBAL
    across every gate (compute-idle-persistent and lane-starvation share one budget on purpose -- both are
    excuses for not launching ready work, and a per-gate budget would let renewing across the TWO waiver files
    double the real allowance). Missing/corrupt history -> 0.0 (fails OPEN: an absent history file must not
    itself become a block, the same staleness discipline `parallel_state.is_fresh` uses).

    A row's own recorded `exempt` flag is authoritative when present (it reflects whether THAT evaluation found
    owner-reservation evidence, per the 2026-09-23 review fix) -- `exclude_classes` is only a fallback for rows
    written before that flag existed, so old history does not need a migration."""
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
        exempt = row.get("exempt")
        if exempt is None:
            exempt = row.get("class") in exclude_classes    # back-compat: pre-evidence-gate history rows
        if exempt:
            continue
        total += float(row.get("waiver_max_h") or 0.0)
    return total


def evaluate(gate_name, path, waiver_max_h, now_ts=None, budget_h=WAIVER_BUDGET_H,
             window_h=WAIVER_BUDGET_WINDOW_H, prior_cumulative_h=None, history_file=HISTORY_FILE,
             evidence_check=None):
    """The one call both gates make. Returns:
      {"active": False}                                                    -- no live waiver (missing/expired)
      {"active": True, "ok": False, "class": <or None>, "reject_reason": <str>, "age_h": float}
      {"active": True, "ok": True,  "class": <CLASS>, "age_h": float, "budget_h": <cumulative incl. this one>}

    `prior_cumulative_h` lets a caller/test inject the pre-this-waiver budget total directly (pure, no history
    file needed); omitted -> computed live from HISTORY_FILE, BEFORE this waiver's own row is written (2026-
    09-23 review fix: recording first and THEN summing double-counted a fresh waiver against its own budget,
    an off-by-one that could reject the very first waiver ever written). History is written (deduped by mtime)
    for every call that finds a live waiver file, valid or not, per the module docstring's point (2).

    `evidence_check` lets a caller/test inject the owner-reservation check (pure, no disk/process access
    needed); omitted -> `_owner_reservation_evidence()`. GAMING/OWNER-PAUSE are budget-exempt ONLY when this
    returns True -- a self-declared GAMING/OWNER-PAUSE with no evidence behind it is treated as a normal
    non-exempt waiver (2026-09-23 review fix: the class alone used to be sufficient, so any agent could grant
    itself an unlimited-renewal exemption by simply writing the words)."""
    now_ts = now_ts if now_ts is not None else time.time()
    found = _read_file(path, waiver_max_h, now_ts=now_ts)
    if found is None:
        return {"active": False}
    text, mtime, age_h = found
    parsed = parse_waiver(text)
    is_new = mtime != _last_recorded_mtime(gate_name, path, history_file)

    if parsed.get("error"):
        if is_new:
            record(gate_name, path, parsed, waiver_max_h, mtime, now_ts, history_file=history_file, exempt=False)
        return {"active": True, "ok": False, "class": None, "reject_reason": parsed["error"], "age_h": age_h}

    cls = parsed["class"]
    is_exempt = False
    if cls in BUDGET_EXEMPT_CLASSES:
        check_fn = evidence_check or _owner_reservation_evidence
        try:
            is_exempt = bool(check_fn())
        except Exception:
            is_exempt = False        # fail CLOSED here: no evidence == not exempt, never crash the gate

    if is_exempt:
        if is_new:
            record(gate_name, path, parsed, waiver_max_h, mtime, now_ts, history_file=history_file, exempt=True)
        return {"active": True, "ok": True, "class": cls, "age_h": age_h, "budget_h": None}

    # Non-exempt path: a genuine per-lane blocker (RAM-CONTENTION/NO-READY-WORK), or a self-declared
    # GAMING/OWNER-PAUSE that found no owner-reservation evidence -- both go through the SAME renewal budget.
    # `prior` MUST be computed before this waiver's own row is recorded (the off-by-one fix): otherwise a
    # brand-new waiver would already see itself in the sum it is being checked against.
    prior = (cumulative_waived_hours(window_h=window_h, now_ts=now_ts, history_file=history_file)
             if prior_cumulative_h is None else prior_cumulative_h)
    if is_new:
        record(gate_name, path, parsed, waiver_max_h, mtime, now_ts, history_file=history_file, exempt=False)
    if prior >= budget_h:
        reject_reason = ("the %.1fh renewal budget for non-GAMING/OWNER-PAUSE waivers in the last "
                          "%dh is EXHAUSTED (%.1fh already used, %s) -- a real resource constraint "
                          "does not renew indefinitely; let the resource run, or waive GAMING/"
                          "OWNER-PAUSE if that is the actual reason"
                          % (budget_h, int(window_h), prior, parsed.get("reason", "")[:80]))
        if cls in BUDGET_EXEMPT_CLASSES:
            reject_reason = ("CLASS: %s claims an owner-reserved exemption but no evidence was found -- no "
                              "GAME_MODE/GPU_PAUSE sentinel and no game process running (the 2026-09-23 "
                              "self-declared-exemption loophole) -- treated as a normal non-exempt waiver, and "
                              % cls) + reject_reason
        return {"active": True, "ok": False, "class": cls, "reject_reason": reject_reason, "age_h": age_h}
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
        # GAMING/OWNER-PAUSE must be budget-EXEMPT even with the budget nominally exhausted -- BUT ONLY when
        # owner-reservation evidence is present (2026-09-23 review fix: the class name alone is NOT evidence).
        with open(p, "w") as fh:
            fh.write("CLASS: OWNER-PAUSE\nreason: owner reserved the box")
        os.utime(p, (now + 1, now + 1))
        v_exempt = evaluate("selftest-gate", p, waiver_max_h=6, now_ts=now + 1, budget_h=6.0,
                            prior_cumulative_h=999.0, history_file=hist, evidence_check=lambda: True)
        if not v_exempt.get("ok"):
            bad.append("did NOT keep OWNER-PAUSE budget-exempt (the owner-gaming case must stay intact)")
        # FAILING DIRECTION (the loophole itself): the SAME self-declared OWNER-PAUSE with NO owner-reservation
        # evidence (no GAME_MODE/GPU_PAUSE sentinel, no game process) must be treated as NON-exempt -- i.e. it
        # must be rejected once the budget is exhausted, exactly like any other class.
        with open(p, "w") as fh:
            fh.write("CLASS: OWNER-PAUSE\nreason: owner reserved the box")
        os.utime(p, (now + 1.5, now + 1.5))
        v_unevidenced = evaluate("selftest-gate", p, waiver_max_h=6, now_ts=now + 1.5, budget_h=6.0,
                                 prior_cumulative_h=999.0, history_file=hist, evidence_check=lambda: False)
        if v_unevidenced.get("ok"):
            bad.append("LOOPHOLE STILL OPEN: a self-declared OWNER-PAUSE with NO owner-reservation evidence "
                       "was still granted the budget-exempt exemption (the 2026-09-23 self-declared-exemption "
                       "loophole)")
        # ... and with budget ROOM, the same unevidenced OWNER-PAUSE must still be ACCEPTED (non-exempt is not
        # the same as invalid -- it just has to fit the renewal budget like anything else).
        with open(p, "w") as fh:
            fh.write("CLASS: OWNER-PAUSE\nreason: owner reserved the box")
        os.utime(p, (now + 1.7, now + 1.7))
        v_unevidenced_room = evaluate("selftest-gate", p, waiver_max_h=6, now_ts=now + 1.7, budget_h=6.0,
                                      prior_cumulative_h=0.0, history_file=hist, evidence_check=lambda: False)
        if not v_unevidenced_room.get("ok") or v_unevidenced_room.get("budget_h") is None:
            bad.append("FALSE POSITIVE / wrong accounting: an unevidenced OWNER-PAUSE with budget room was "
                       "rejected, or not charged against the budget (budget_h stayed None as if exempt)")

        # OFF-BY-ONE (2026-09-23 review fix): a FRESH (first-ever, no prior history) non-exempt waiver whose
        # own waiver_max_h alone equals the budget must NOT count itself when checking the PRIOR total -- it
        # must be ACCEPTED, computed against the REAL (empty) history file, not an injected prior_cumulative_h.
        hist_fresh = os.path.join(td, "history_fresh.jsonl")
        p_fresh = os.path.join(td, ".waiver_fresh")
        with open(p_fresh, "w") as fh:
            fh.write("CLASS: NO-READY-WORK\nchecked=first-ever waiver, empty history")
        os.utime(p_fresh, (now, now))
        v_fresh = evaluate("selftest-gate", p_fresh, waiver_max_h=6, now_ts=now, budget_h=6.0,
                           history_file=hist_fresh)     # prior_cumulative_h OMITTED -> computed live
        if not v_fresh.get("ok"):
            bad.append("OFF-BY-ONE STILL PRESENT: a waiver counted its OWN just-recorded row against its own "
                       "budget check and rejected the very first waiver ever written (budget_h=%s)"
                       % v_fresh.get("budget_h"))

        # RATIONALISATION VOCABULARY UNDER EVERY CLASS (2026-09-23 review fix): a well-formed CLASS +
        # required-evidence field must still be REJECTED if it smuggles the old priority/focus excuse.
        if not parse_waiver("CLASS: NO-READY-WORK\nchecked=focused on the crux, nothing else worth it").get("error"):
            bad.append("did NOT reject rationalisation language ('focused on the crux') under a valid "
                       "NO-READY-WORK class + a syntactically-valid checked= field")
        if not parse_waiver("CLASS: RAM-CONTENTION\navail_gb=4\nreason: deprioritized behind the crux").get("error"):
            bad.append("did NOT reject rationalisation language under a valid RAM-CONTENTION class")
        # NEGATIVE: a well-formed waiver containing none of the banned vocabulary must still pass.
        if parse_waiver("CLASS: NO-READY-WORK\nchecked=lane_check.py -- 5/5 lanes served, pool.queue empty").get("error"):
            bad.append("FALSE POSITIVE: the rationalisation check rejected a genuinely clean NO-READY-WORK waiver")

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
