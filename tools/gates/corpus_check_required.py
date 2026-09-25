"""CLASS CC — AN EXPENSIVE RUN WHOSE QUESTION WAS NEVER CHECKED AGAINST THE RECORD.

THE DEFECT, and it is the most expensive one measured on 2026-07-31. A nine-hour, eight-cell GPU crux was
launched against "does deep credit train on the on-bridge spiking forward". That question was already
answered:

  2026-07-07  the depth-2 spiking net does NOT train this task — all arms below chance, ALL SIX SEEDS,
              oracle 1.0. The identical signature the crux spent nine hours reproducing.
  2026-07-14  the cause, LOCATED: Izhikevich forward NOISE at full-task scale — not the rule, not epochs
              (300 epochs moved train 0.482 -> 0.497).
  2026-07-08  population coding swept K in {1,8,16}: no crossover.
  2026-07-12  the negative repeated at depth 2.

`tools/before_you_build.sh` returns all four in **0.63 seconds**. It was not run before launch. The 1-seed
replication was then written up as a new localisation, and only a post-hoc corpus check caught it.

WHY REPORTING WAS NOT ENOUGH, which is the whole argument for making this block. The heartbeat printed
"⛔ A FINDING WAS WRITTEN SINCE THE LAST SOURCE CHECK" roughly FIFTEEN times that day and was read past
every single time. That is the same shape as `lane_starvation`, where a true alarm ran for 194 minutes
unheeded and had to be made blocking. An alarm nobody acts on is not coverage.

EVERY OTHER GATE HERE LOOKS FOR A *WRONG* CLAIM. This one looks for a *REDUNDANT* one — and redundancy is
the more expensive failure, because a wrong claim gets caught downstream while a redundant one quietly
burns GPU-hours and produces a finding that reads perfectly well.

WHAT IT ENFORCES, on newly-added artifacts only: a run recording more than `MIN_COST_S` of compute must
carry evidence that the record was consulted — `corpus_check_fresh` in the artifact or its provenance
sidecar, stamped automatically by `research/runners/__init__` from the log that
`tools/before_you_build.sh` now writes. Cheap runs are exempt: the cost of re-deriving a two-minute smoke is
two minutes, and a gate that fires on those gets switched off.

FALLBACK (2026-09-25 incident: this gate blocked research/score-gap4-c26-0925 @ ed6758f61 despite a REAL
corpus check having been run at 01:38:31 before the pool jobs that produced the artifact were dispatched).
ROOT CAUSE: `tools/before_you_build.sh` wrote its log under the CALLER'S OWN `$PWD`, so a check run inside one
git worktree was invisible to a run launched from any other checkout of this repo (115 divergent logs existed
under `.claude/worktrees/*/research/queue/` the day this was found) — and a pool/GPU job commonly runs on a
node or an isolated revision checkout with no shared log reachable AT ALL. Fixed upstream (the log is now one
shared file per repo, and a queued job carries the freshest check in its own env — see
`tools/corpus_check_lib.sh` and `research/runners/__init__._corpus_check_state`), but an artifact produced
before that fix landed, or by a lane the propagation could not reach, still carries no stamp. For exactly that
case, and ONLY when the artifact+sidecar carry no stamp at all, this gate now ALSO accepts a corpus-check
entry found in the shared root log OR any `.claude/worktrees/*/research/queue/.corpus_checks.jsonl`, dated
BEFORE this run's own start (sidecar `started`, else the artifact's own, else `mtime(artifact) - elapsed`) and
within the same freshness window `corpus_check_fresh` uses. This is the SAME evidential strength as the direct
stamp, stated explicitly rather than implied: presence of *a* check is verified, never that its *topic*
matched this run's question (see "WHAT IT CANNOT CATCH" below, which already applies to the direct stamp and
now applies here too) — and a pass via this path is logged with the evidence file named, so it stays visible
rather than silently indistinguishable from a genuine in-artifact stamp.

WHAT IT CANNOT CATCH: a corpus check that was RUN and not READ. The check records that the question was
asked, never that the answer was understood — on 2026-07-31 the priors were one command away and the
failure was not looking, but a future failure could equally be looking and not reading. That is judgement,
and it is left as judgement rather than pretended away. The fallback above inherits this limitation exactly:
it confirms A check ran near THIS run's start, never that it was about the same question.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
import time

NAME = "corpus-check-required"
CLASS_ID = "CC"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ELAPSED_KEYS = ("elapsed_seconds", "elapsed_s", "elapsed", "runtime_seconds", "wall_seconds")
MIN_COST_S = 3600.0            # an hour. Below this, re-deriving costs less than the check's friction.
MAX_AGE_H = 24.0                # matches research/runners/__init__._corpus_check_state's default window.


def _find(obj, keys, depth=0):
    if not isinstance(obj, dict):
        return None
    for k, v in obj.items():
        if k.lower() in keys and v is not None and not isinstance(v, (dict, list)):
            return v
    if depth < 2:
        for v in obj.values():
            if isinstance(v, dict):
                got = _find(v, keys, depth + 1)
                if got is not None:
                    return got
    return None


def _sidecar(path):
    for sib in (path + ".prov.json", os.path.splitext(path)[0] + ".prov.json"):
        if os.path.exists(sib):
            try:
                return json.load(open(sib, errors="ignore"))
            except (OSError, ValueError):
                return None
    return None


def _shared_root():
    """The parent of git's *common* dir -- the ONE root every worktree of this repo shares -- mirroring
    tools/corpus_check_lib.sh's corpus_check_shared_log() and research/runners/__init__._shared_queue_root().
    Falls back to _ROOT outside a git checkout (nothing shared to find there either)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=_ROOT, capture_output=True, text=True, timeout=5,
        )
        common = out.stdout.strip()
        if out.returncode == 0 and common:
            return os.path.dirname(common)
    except Exception:
        pass
    return _ROOT


def _run_start_epoch(obj, side, path, elapsed):
    """Best-effort epoch seconds for when this run STARTED (not when the artifact was written) -- needed to
    judge whether a corpus-check log entry came before it. Prefers an explicit `started` timestamp (the
    provenance sidecar's, which research/runners/__init__ stamps at IMPORT time, before any of the run's own
    compute; else the artifact's own, for a runner that writes one directly), then a v2 sidecar's
    `started_utc_ns`, then falls back to `mtime(artifact) - elapsed` (the artifact is written at/after the run
    ends, so this approximates the start under the assumption mtime is close to completion)."""
    for src in (side, obj):
        if not isinstance(src, dict):
            continue
        started = src.get("started")
        if isinstance(started, str) and started:
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return time.mktime(time.strptime(started, fmt))
                except ValueError:
                    continue
        started_ns = src.get("started_utc_ns")
        if isinstance(started_ns, (int, float)):
            return started_ns / 1e9
    try:
        return os.path.getmtime(path) - elapsed
    except OSError:
        return None


def _fallback_evidence(run_start, max_age_h=MAX_AGE_H, shared_root=None):
    """Search the shared root log + every worktree's own log for a corpus-check entry dated BEFORE run_start
    and within max_age_h of it. Returns (evidence_log_path, entry_dict) or (None, None). `shared_root`
    overrides `_shared_root()` (test seam -- keeps selftest() isolated from the real repo's own logs)."""
    if run_start is None:
        return None, None
    root = shared_root if shared_root is not None else _shared_root()
    candidates = [os.path.join(root, "research", "queue", ".corpus_checks.jsonl")]
    candidates += sorted(glob.glob(os.path.join(root, ".claude", "worktrees", "*",
                                                 "research", "queue", ".corpus_checks.jsonl")))
    window_s = max_age_h * 3600.0
    for log in candidates:
        if not os.path.exists(log):
            continue
        try:
            with open(log, errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(entry, dict) or entry.get("when") is None:
                        continue
                    try:
                        when = float(entry["when"])
                    except (TypeError, ValueError):
                        continue
                    if when <= run_start and (run_start - when) <= window_s:
                        return log, entry
        except OSError:
            continue
    return None, None


def _check_one(path, rel=None, shared_root=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    if rel.endswith(".prov.json") or rel.endswith(".cmd.json"):
        return []
    try:
        obj = json.load(open(path, errors="ignore"))
    except (OSError, ValueError):
        return []
    if not isinstance(obj, dict):
        return []

    elapsed = _find(obj, ELAPSED_KEYS)
    try:
        elapsed = float(elapsed) if elapsed is not None else None
    except (TypeError, ValueError):
        elapsed = None
    if elapsed is None or elapsed <= MIN_COST_S:
        return []                                          # cheap or untimed: out of scope by design

    side = _sidecar(path)
    fresh = obj.get("corpus_check_fresh")
    if fresh is None:
        fresh = side.get("corpus_check_fresh") if isinstance(side, dict) else None
    if fresh:
        return []

    # FALLBACK (2026-09-25): no in-artifact/sidecar stamp -- before blocking, look for a corpus-check entry
    # recorded anywhere this repo keeps one, dated before this run started (see module docstring +
    # _fallback_evidence). A pass here is printed, not silent, and names its evidence.
    run_start = _run_start_epoch(obj, side, path, elapsed)
    ev_log, ev_entry = _fallback_evidence(run_start, MAX_AGE_H, shared_root)
    if ev_entry is not None:
        age_h = (run_start - float(ev_entry["when"])) / 3600.0
        print("  [corpus-check-required] %s: no in-artifact stamp, but %s records a check %.1fh before this "
              "run started (query: %r) -- accepted with the same evidential strength as a direct stamp "
              "(presence, not topic, is checked)." % (rel, ev_log, age_h, str(ev_entry.get("query", ""))[:120]))
        return []

    return ["%s: records %.1fh of compute with NO recent corpus check (`corpus_check_fresh` absent or false, "
            "in the artifact and its provenance sidecar, and no shared/worktree corpus-check log carries an "
            "entry before this run started either). Run `bash tools/before_you_build.sh \"<the "
            "question>\"` — it returns the priors in under a second. On 2026-07-31 a nine-hour eight-cell "
            "crux re-derived a SIX-SEED result banked three weeks earlier whose root cause was already "
            "located, and the heartbeat's advisory warning was read past ~15 times that day."
            % (rel, elapsed / 3600.0)]


def check(paths):
    if paths is None or len(paths) == 0:
        return []                                          # legacy predates the stamp; audited on touch
    problems = []
    for p in [x for x in paths if x.endswith(".json")]:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full, p)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: the expensive unchecked run, then everything that must NOT fire.

    Cases 1-2 and 10-13 pass an ISOLATED `shared_root` (an empty tmp dir, or one holding only the log the case
    itself wrote) so the new fallback (2026-09-25) cannot accidentally pick up a real entry from this actual
    repo's own `research/queue/.corpus_checks.jsonl` / `.claude/worktrees/*` and turn a case that must BLOCK
    into a silent pass -- a selftest whose verdict depends on what else happens to be logged on the machine
    running it is not a selftest."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, obj):
            p = os.path.join(d, name)
            json.dump(obj, open(p, "w"))
            return p

        empty_root = os.path.join(d, "_empty_shared_root")   # exists, but no .corpus_checks.jsonl under it
        os.makedirs(empty_root, exist_ok=True)

        # 1. THE REAL CASE: a 9-hour run with no corpus check anywhere.
        if not _check_one(w("a.json", {"elapsed_seconds": 9 * 3600}), "raw/a.json", shared_root=empty_root):
            bad.append("did NOT catch an expensive run with no corpus check")
        # 2. explicitly stale must fire too, not just absent.
        if not _check_one(w("b.json", {"elapsed_seconds": 9 * 3600, "corpus_check_fresh": False}), "raw/b.json",
                           shared_root=empty_root):
            bad.append("did NOT catch an expensive run whose corpus check was STALE")
        # 3. NEGATIVE CONTROL — a checked expensive run passes, or nobody can satisfy the gate.
        if _check_one(w("c.json", {"elapsed_seconds": 9 * 3600, "corpus_check_fresh": True}), "raw/c.json"):
            bad.append("FALSE POSITIVE: flagged an expensive run that DID check the corpus")
        # 4. NEGATIVE CONTROL — a CHEAP run is out of scope; re-deriving a smoke costs a smoke.
        if _check_one(w("d.json", {"elapsed_seconds": 120}), "raw/d.json"):
            bad.append("FALSE POSITIVE: flagged a cheap run")
        # 5. NEGATIVE CONTROL — an untimed artifact cannot be judged expensive.
        if _check_one(w("e.json", {"means": {"acc": 0.5}}), "raw/e.json"):
            bad.append("FALSE POSITIVE: flagged an artifact with no elapsed time")
        # 6. NEGATIVE CONTROL — the sidecar may carry the evidence instead of the artifact.
        p = w("f.json", {"elapsed_seconds": 9 * 3600})
        json.dump({"corpus_check_fresh": True}, open(p + ".prov.json", "w"))
        if _check_one(p, "raw/f.json"):
            bad.append("FALSE POSITIVE: ignored a provenance sidecar carrying the corpus check")
        # 7. NEGATIVE CONTROL — sidecars are evidence, not subjects.
        if _check_one(p + ".prov.json", "raw/f.json.prov.json"):
            bad.append("FALSE POSITIVE: audited a provenance sidecar as a result")
        # 8. SCOPING — standalone/empty scans nothing.
        if check(None) or check([]):
            bad.append("SCOPE LEAK: standalone/empty mode must not scan the legacy corpus")

        # ---- 2026-09-25 incident fix: the shared/worktree-log FALLBACK, isolated from the real repo -----------
        run_start = int(time.time()) - 3600                                       # this run "started" 1h ago
        started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start))

        def _log_with(root, rel_log, when, query="a real corpus check"):
            log_path = os.path.join(root, rel_log)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as fh:
                fh.write(json.dumps({"when": when, "query": query}) + "\n")
            return root

        # 9. THE INCIDENT ITSELF: no in-artifact stamp, but the SHARED ROOT log carries a real check shortly
        #    before this run started -- must be ACCEPTED, not blocked.
        root9 = os.path.join(d, "root9")
        _log_with(root9, "research/queue/.corpus_checks.jsonl", run_start - 1000, "gap4 transport ceiling BDSP clamp")
        if _check_one(w("g.json", {"elapsed_seconds": 9 * 3600, "started": started_str}), "raw/g.json",
                       shared_root=root9):
            bad.append("FALSE POSITIVE (regression on the incident itself): a real shared-log check dated "
                       "before this run's start was still blocked")
        # 10. The SAME evidence, but sitting only in a WORKTREE's own log -- must ALSO be accepted (this is the
        #     other half of root cause (a): a worktree's own check is real evidence too).
        root10 = os.path.join(d, "root10")
        _log_with(root10, os.path.join(".claude", "worktrees", "wf_x", "research", "queue",
                                        ".corpus_checks.jsonl"), run_start - 1000)
        if _check_one(w("h.json", {"elapsed_seconds": 9 * 3600, "started": started_str}), "raw/h.json",
                       shared_root=root10):
            bad.append("FALSE POSITIVE: a worktree-local corpus-check log was not treated as valid evidence")
        # 11. FAILING-DIRECTION CONTROL — a check logged AFTER this run already started must NOT count (a check
        #     run once the expensive compute was already under way proves nothing about whether it was
        #     consulted beforehand).
        root11 = os.path.join(d, "root11")
        _log_with(root11, "research/queue/.corpus_checks.jsonl", run_start + 500)
        if not _check_one(w("i.json", {"elapsed_seconds": 9 * 3600, "started": started_str}), "raw/i.json",
                           shared_root=root11):
            bad.append("FALSE POSITIVE: accepted a corpus-check entry logged AFTER the run had already started")
        # 12. FAILING-DIRECTION CONTROL — a check far older than the freshness window, even though it IS
        #     before the run, must NOT count (staleness applies to the fallback exactly as to the direct stamp).
        root12 = os.path.join(d, "root12")
        _log_with(root12, "research/queue/.corpus_checks.jsonl", run_start - int((MAX_AGE_H + 1) * 3600))
        if not _check_one(w("j.json", {"elapsed_seconds": 9 * 3600, "started": started_str}), "raw/j.json",
                           shared_root=root12):
            bad.append("FALSE POSITIVE: accepted a corpus-check entry older than the freshness window")
        # 13. NEGATIVE CONTROL — an unrelated, populated log tree with no qualifying entry must still BLOCK
        #     (proves 9/10 are not passing merely because SOME log file exists under shared_root).
        root13 = os.path.join(d, "root13")
        _log_with(root13, "research/queue/.corpus_checks.jsonl", run_start - int((MAX_AGE_H + 5) * 3600))
        _log_with(root13, os.path.join(".claude", "worktrees", "wf_y", "research", "queue",
                                        ".corpus_checks.jsonl"), run_start + 9999)
        if not _check_one(w("k.json", {"elapsed_seconds": 9 * 3600, "started": started_str}), "raw/k.json",
                           shared_root=root13):
            bad.append("did NOT catch an expensive run whose only logged checks are stale/after-the-fact")
    return bad
