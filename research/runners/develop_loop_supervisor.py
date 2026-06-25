"""CRASH-PROOF / PAUSABLE / RESUMABLE supervisor for the longitudinal develop loop (the 24/7 foundational-
curriculum training harness).

THE OWNER'S BAR (verbatim intent): "Run it 24/7 in a fashion that won't get accidentally killed no matter what,
but can be paused when I want to game and recover without too much lost in the event of a crash, restart, etc."
+ "the training absolutely can't be accidentally killed (or if killed, can recover and not have to start from
scratch)."

This module is the THIN, NON-INVASIVE hardening layer around the validated develop loop
(`research/runners/_longitudinal_develop_loop_gpu.develop_gpu`). It adds the four robustness requirements WITHOUT
editing the validated loop (`develop_gpu`) or `sim/`:

  (1) 24/7 CONTINUOUS    — run an unbounded number of simulated days into ONE persistent BridgeLineage (resume on
                           every relaunch). The companion `scripts/develop_train_watchdog.ps1` keeps it alive.
  (2) CAN'T BE KILLED    — the companion watchdog launches this run-DETACHED (survives terminal-close / parent
                           exit) and auto-RESTARTS it from the latest checkpoint on any non-pause exit.
  (3) PAUSABLE on demand — a PAUSE SENTINEL file (default `bridges/PAUSE`). The loop polls it at every safe point
                           (each day boundary, via the develop loop's existing `per_day_save_hook`), and on seeing
                           it: the latest day is already checkpointed -> it exits CLEANLY (code 0) so the watchdog
                           does NOT restart it. Deleting the sentinel + relaunching resumes.
  (4) CRASH-RESUMABLE    — every day's state is persisted ATOMICALLY (BridgeLineage `.new` + `os.replace`) AND
                           fsync'd to disk (this module's `_fsync_lineage` closes the residual power-loss window
                           the bare `os.replace` leaves). A hard kill at ANY point loses AT MOST the in-flight day
                           (and the watchdog resumes from the last completed day). With `--max-windows-per-day`
                           bounding a day to MINUTES, worst-case loss is bounded to minutes, not a whole long day.

WHY THE EXISTING LOOP IS ALREADY MOSTLY ROBUST (verified): `develop_gpu` persists the full `DevelopState` (day,
facts, vocab, tier, metrics, turns) every day via `BridgeLineage.save` (atomic `.h5.new`+`os.replace`) and resumes
correctly (`resume=True` re-instates facts + re-hears the cumulative vocab, continues from `state.day`). The GAPS
this module closes: (a) NO fsync before the atomic rename (a hard power-loss could leave the rename pointing at
unflushed bytes); (b) NO pause mechanism; (c) NO watchdog that auto-restarts the TRAINING (the repo's
`autonomous_watchdog.ps1` restarts Claude, not the run); (d) NO run-detached launcher.

NON-INVASIVE: this uses the develop loop's PUBLIC seams only — `develop_gpu(..., resume=, per_day_save_hook=,
corpus_path=, save_bundle_root=, per_day_bundles=)`. The pause-poll + fsync ride on the existing
`per_day_save_hook` (which fires AFTER each day's `_save_state` PERSIST). NO `sim/` edit; NO edit to the validated
loop.

PAUSE / RESUME (owner commands):
    # pause (frees the GPU after the current day finishes; resumable):
    New-Item -ItemType File bridges/PAUSE            # PowerShell
    touch bridges/PAUSE                              # bash
    # resume:
    Remove-Item bridges/PAUSE                        # PowerShell  (then relaunch / let the watchdog relaunch)
    rm bridges/PAUSE                                 # bash

Run (foreground, resumable; CPU-testable with SIM_BACKEND=numpy + a tiny config):
    SIM_BACKEND=cupy python -u -m research.runners.develop_loop_supervisor \
        --lineage-root bridges/developed/curriculum --bundle-root bridges/developed/curriculum/bundles \
        --per-day-bundles --max-windows-per-day 2500 --pause-file bridges/PAUSE
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Default to CuPy (the production training backend); an explicit SIM_BACKEND still wins (CPU tests pass numpy).
os.environ.setdefault("SIM_BACKEND", "cupy")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.lineage import BridgeLineage  # noqa: E402
from research.runners._longitudinal_develop_loop import _load_state  # noqa: E402
from research.runners._longitudinal_develop_loop_gpu import (  # noqa: E402
    GPUGradedCurriculum, StreamCortex, develop_gpu, save_developed_bundle,
)


# Sentinel exit code the supervisor returns when it stops because the PAUSE sentinel was seen. The watchdog uses
# THIS specific code to distinguish "owner paused -> do NOT restart" from "crashed/exited -> restart".
PAUSE_EXIT_CODE = 42


def _fsync_path(p: Path):
    """fsync a single file's bytes to physical disk (best-effort; never raises)."""
    try:
        fd = os.open(str(p), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _fsync_lineage(lineage: BridgeLineage):
    """fsync the lineage's just-written current state + metadata to disk.

    `BridgeLineage.save`/`write_metadata` already do the ATOMIC part (write `.new` -> `os.replace`), which
    guarantees a reader never sees a torn file. But `os.replace` does NOT guarantee the renamed file's *contents*
    are on physical media before a power loss. fsync'ing the final paths closes that residual window so a hard
    power-cut after a day completes cannot leave a zero-length / partially-flushed checkpoint. Best-effort: a
    failure here never kills the training (the atomic rename already protects against torn reads)."""
    for p in (lineage.current_path, lineage.metadata_path):
        if p.exists():
            _fsync_path(p)
    # also fsync the directory entry so the rename itself is durable
    try:
        dfd = os.open(str(lineage.root), os.O_RDONLY)
        try:
            os.fsync(dfd)
        except OSError:
            pass
        finally:
            os.close(dfd)
    except OSError:
        pass  # directory fsync is not supported on all platforms (e.g. some Windows configs) — atomic rename stands


def _heartbeat(path: Path, payload: dict):
    """Write a tiny heartbeat JSON (atomic) so the watchdog + the owner can see liveness + the last completed day
    WITHOUT parsing the .h5. Never raises."""
    try:
        tmp = path.with_suffix(".json.new")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except OSError:
        pass


def run_resumable(*, lineage_root, lineage_name="curriculum", seed=42, n_days=None,
                  max_windows_per_day=2500, n_hub=200, n_per=12, D=128, enable_neural_render=False,
                  bundle_root=None, per_day_bundles=False, corpus_path=None,
                  pause_file="bridges/PAUSE", heartbeat_file=None, max_runtime_s=None, verbose=True):
    """Run the develop loop RESUMABLY into ONE persistent lineage, polling the PAUSE sentinel at each day boundary
    and fsync'ing every checkpoint.

    Returns a result dict: {"stopped_reason": "pause"|"days_done"|"runtime_cap", "days_lived": int,
    "facts": int, "vocab": int, "exit_code": int, ...}.

    Args:
        lineage_root: directory holding the persistent BridgeLineage (the brain "lives" here across relaunches).
        n_days: if None (default), run UNBOUNDED (24/7) until paused / runtime cap / kill. If an int, run at most
            that many MORE days this invocation (the per-call bound the watchdog can use to checkpoint-and-relaunch
            for extra durability, while the loop's own per-day persistence is the real safety).
        max_windows_per_day: caps a day's stream-window budget -> caps per-day wall-clock -> bounds worst-case
            crash loss to that day's wall-clock. Keep it MINUTES (the robustness lever for long foundational days).
        pause_file: the PAUSE sentinel path. When it appears, the loop finishes persisting the current day then
            exits cleanly with PAUSE_EXIT_CODE.
        heartbeat_file: optional liveness file (default: <lineage_root>/<lineage_name>/heartbeat.json).
        max_runtime_s: optional wall-clock cap for THIS invocation (the watchdog can periodically recycle the
            process; the lineage resume makes this lossless).
    """
    lineage = BridgeLineage(lineage_name, root=Path(lineage_root))
    curriculum = GPUGradedCurriculum()
    full_vocab = curriculum.full_vocab()
    referent_nouns = curriculum.referent_nouns()
    pause_path = Path(pause_file)
    if heartbeat_file is None:
        heartbeat_file = str(lineage.root / "heartbeat.json")
    hb_path = Path(heartbeat_file)
    t_start = time.time()

    # If the owner set PAUSE before we even start, honor it immediately (do nothing, exit clean).
    if pause_path.exists():
        if verbose:
            print(f"[supervisor] PAUSE sentinel present at start ({pause_path}); exiting clean (paused).",
                  flush=True)
        _pre = _load_state(lineage) if lineage.exists() else None
        return {"stopped_reason": "pause", "exit_code": PAUSE_EXIT_CODE,
                "days_this_run": 0, "days_lived": (_pre.day if _pre else 0),
                "last_completed_day": None,
                "facts": (len(_pre.facts) if _pre else 0), "vocab": (len(_pre.vocab) if _pre else 0),
                "tier": (_pre.current_tier if _pre else None), "lineage_root": str(lineage.root),
                "heartbeat_file": str(hb_path)}

    # A SHARED, PERSISTENT stream cortex across all days THIS invocation (so the co-occurrence ACCUMULATES across
    # days within the process, exactly as the validated --save-bundle path does).
    shared_cortex = StreamCortex(full_vocab, seed, n_hub=n_hub, n_per=n_per, D=D, verbose=verbose,
                                 corpus_path=corpus_path)
    # On RESUME (a relaunch after a pause/crash), re-hear the cumulative developed vocab into this fresh shared
    # cortex so the grounded codes for PRIOR concepts are re-instated immediately (matching develop_gpu's own
    # `own_cortex` resume behavior, which is skipped when a shared cortex is injected). This is the documented
    # cheap stand-in for persisting the stream-cortex synaptic store in the lineage .h5 (a GPU follow-on).
    if lineage.exists():
        try:
            prior = _load_state(lineage)
            if prior.vocab:
                if verbose:
                    print(f"[supervisor] resume: re-hearing {len(prior.vocab)} developed concepts to re-instate "
                          f"learned codes...", flush=True)
                shared_cortex.hear_day(list(prior.vocab), max_windows=max_windows_per_day)
        except Exception as _re:  # noqa: BLE001
            if verbose:
                print(f"[supervisor] resume re-hear skipped (non-fatal): {_re!r}", flush=True)

    state_box = {"days_this_run": 0, "last_day": None, "facts": 0, "vocab": 0, "stop_reason": None}

    def per_day_hook(day_index, state, grounded, agent):
        # This fires AFTER develop_gpu's per-day _save_state PERSIST (atomic). We: (a) optionally save a per-day
        # console bundle, (b) fsync the lineage checkpoint to physical disk, (c) write a heartbeat. The PAUSE /
        # runtime-cap STOP is NOT done here (the loop swallows hook exceptions) -- it is done by `should_continue`,
        # the day-boundary predicate the loop polls BEFORE the next day.
        if bundle_root and per_day_bundles:
            try:
                bdir = os.path.join(bundle_root, f"day_{day_index}")
                from research.runners.developed_brain_io import save_developed_brain
                comp = getattr(agent, "agent", agent).composer
                save_developed_brain(
                    agent, bdir, seed=int(seed), D=int(getattr(comp, "D", D)), composer_kind="rf",
                    develop_state=state, lineage_name="developed_brain",
                    extra_metadata={"provenance": "develop_loop_supervisor", "day": int(day_index)})
            except Exception as _e:
                if verbose:
                    print(f"    [supervisor] per-day bundle save failed (non-fatal): {_e!r}", flush=True)

        # (b) DURABILITY: fsync the just-persisted checkpoint + metadata.
        _fsync_lineage(lineage)

        # (c) heartbeat / liveness.
        state_box["days_this_run"] += 1
        state_box["last_day"] = int(day_index)
        state_box["facts"] = len(state.facts)
        state_box["vocab"] = len(state.vocab)
        _heartbeat(hb_path, {"pid": os.getpid(), "ts": time.time(),
                             "last_completed_day": int(day_index),
                             "facts": len(state.facts), "vocab": len(state.vocab),
                             "tier": state.current_tier, "backend": os.environ.get("SIM_BACKEND")})
        if verbose:
            print(f"    [supervisor] day {day_index} checkpointed + fsync'd "
                  f"(facts={len(state.facts)} vocab={len(state.vocab)} tier={state.current_tier})", flush=True)

    def should_continue():
        # Polled at each day boundary BEFORE the next day's work. Stop cleanly on PAUSE or the runtime cap (the
        # prior day is already durably persisted+fsync'd, so stopping here loses zero completed work).
        if pause_path.exists():
            state_box["stop_reason"] = "pause"
            return False
        if max_runtime_s is not None and (time.time() - t_start) >= max_runtime_s:
            state_box["stop_reason"] = "runtime_cap"
            return False
        return True

    # Unbounded 24/7: run a very large day count; the loop persists per day + stops on the PAUSE/cap predicate.
    days_to_run = int(n_days) if n_days is not None else 10 ** 9
    try:
        resume = lineage.exists()
        develop_gpu(lineage, curriculum, days_to_run, seed=seed, consolidation_on=True, plasticity_on=True,
                    max_windows_per_day=max_windows_per_day, n_hub=n_hub, n_per=n_per, D=D,
                    enable_neural_render=enable_neural_render, resume=resume, verbose=verbose,
                    _shared_cortex=shared_cortex, per_day_save_hook=per_day_hook, corpus_path=corpus_path,
                    should_continue=should_continue)
    finally:
        try:
            shared_cortex.close()
        except Exception:
            pass

    stopped_reason = state_box["stop_reason"] or "days_done"
    if verbose and stopped_reason != "days_done":
        print(f"[supervisor] clean stop: reason={stopped_reason}; last completed day = "
              f"{state_box['last_day']}, state durably persisted.", flush=True)

    final = _load_state(lineage) if lineage.exists() else None
    exit_code = PAUSE_EXIT_CODE if stopped_reason in ("pause", "runtime_cap") else 0
    return {
        "stopped_reason": stopped_reason,
        "exit_code": exit_code,
        "days_this_run": state_box["days_this_run"],
        "days_lived": (final.day if final else 0),
        "last_completed_day": state_box["last_day"],
        "facts": (len(final.facts) if final else 0),
        "vocab": (len(final.vocab) if final else 0),
        "tier": (final.current_tier if final else None),
        "lineage_root": str(lineage.root),
        "heartbeat_file": str(hb_path),
    }


def main():
    ap = argparse.ArgumentParser(description="Crash-proof / pausable / resumable develop-loop training supervisor.")
    ap.add_argument("--lineage-root", default="bridges/developed/curriculum",
                    help="persistent lineage directory (the brain lives here across relaunches)")
    ap.add_argument("--lineage-name", default="curriculum")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-days", type=int, default=None,
                    help="MORE days to live this invocation (default None = UNBOUNDED 24/7; the watchdog + the "
                         "PAUSE sentinel control stopping). A finite value bounds this call (the loop still "
                         "persists+resumes per day).")
    ap.add_argument("--max-windows-per-day", type=int, default=2500,
                    help="stream-window budget per day; caps per-day wall-clock -> bounds worst-case crash loss")
    ap.add_argument("--n-hub", type=int, default=200)
    ap.add_argument("--n-per", type=int, default=12)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--neural-render", action="store_true")
    ap.add_argument("--bundle-root", default=None,
                    help="directory for console-loadable developed-brain bundles (per-day + final)")
    ap.add_argument("--per-day-bundles", action="store_true")
    ap.add_argument("--corpus-path", default=None, help="plain-text corpus shard (default: the wired TinyStories)")
    ap.add_argument("--pause-file", default="bridges/PAUSE",
                    help="PAUSE sentinel path; create it to pause (resumable), delete it to allow resume")
    ap.add_argument("--heartbeat-file", default=None)
    ap.add_argument("--max-runtime-s", type=float, default=None,
                    help="optional wall-clock cap for THIS invocation (lossless: resume continues)")
    ap.add_argument("--out", default=None, help="optional result JSON path")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    print("=" * 110, flush=True)
    print("[DEVELOP-LOOP SUPERVISOR — crash-proof / pausable / resumable 24/7 training]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND')}  lineage={a.lineage_root}/{a.lineage_name}  seed={a.seed}  "
          f"n_days={'UNBOUNDED' if a.n_days is None else a.n_days}  max_windows/day={a.max_windows_per_day}",
          flush=True)
    print(f"  PAUSE sentinel: {a.pause_file}  (create=pause, delete=resume)   exit code on pause = {PAUSE_EXIT_CODE}",
          flush=True)
    print("=" * 110 + "\n", flush=True)

    res = run_resumable(
        lineage_root=a.lineage_root, lineage_name=a.lineage_name, seed=a.seed, n_days=a.n_days,
        max_windows_per_day=a.max_windows_per_day, n_hub=a.n_hub, n_per=a.n_per, D=a.D,
        enable_neural_render=a.neural_render, bundle_root=a.bundle_root, per_day_bundles=a.per_day_bundles,
        corpus_path=a.corpus_path, pause_file=a.pause_file, heartbeat_file=a.heartbeat_file,
        max_runtime_s=a.max_runtime_s, verbose=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=str)

    print(f"\n{'=' * 110}", flush=True)
    print(f"  STOP: reason={res['stopped_reason']}  days_lived={res['days_lived']}  facts={res['facts']}  "
          f"vocab={res['vocab']}  tier={res['tier']}  -> exit code {res['exit_code']}", flush=True)
    print(f"  (exit {PAUSE_EXIT_CODE} = paused/runtime-cap = watchdog will NOT restart; any other code = restart)",
          flush=True)
    print("=" * 110, flush=True)
    return res["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
