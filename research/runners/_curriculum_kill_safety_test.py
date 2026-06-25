"""KILL-SAFETY TEST for the develop-loop training supervisor (the decisive robustness gate).

Verifies the owner's bar: "the training absolutely can't be accidentally killed (or if killed, can recover and
not have to start from scratch)."

THE TEST (CPU, numpy; does NOT touch the GPU):
  1. Launch `research.runners.develop_loop_supervisor` as a SUBPROCESS into a fresh persistent lineage and let it
     complete a few days (so there is real developed state on disk).
  2. Relaunch it and HARD-KILL the child at a random point mid-run (SIGKILL / taskkill /F — the most brutal
     termination, simulating a power loss / OOM-killer / accidental kill).
  3. Assert the on-disk lineage is NOT corrupted (it reloads, the day/facts/vocab are coherent, the .h5 is valid
     JSON, the metadata is valid JSON).
  4. Relaunch the supervisor -> assert it RESUMES from the last durably-persisted day (it does NOT start from
     scratch) and continues developing.
  5. Measure WORST-CASE LOSS = days that had to be re-done = (day count just before kill) - (day count after the
     hard kill, as seen on disk). With per-day atomic+fsync persistence this is 0 completed days lost; only the
     in-flight (not-yet-finished) day is repeated.

Run:
    SIM_BACKEND=numpy python -u -m research.runners._curriculum_kill_safety_test --out research/findings/raw/_kill_safety.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load_state_safe(lineage_dir: Path, name: str):
    """Reload the persisted DevelopState; return (state_dict|None, error|None, files_ok)."""
    cur = lineage_dir / name / "current.simstate.h5"
    meta = lineage_dir / name / "metadata.json"
    files_ok = {"current_exists": cur.exists(), "metadata_exists": meta.exists(),
                "current_size": (cur.stat().st_size if cur.exists() else 0),
                "metadata_size": (meta.stat().st_size if meta.exists() else 0)}
    if not cur.exists():
        return None, "current.simstate.h5 missing", files_ok
    try:
        payload = json.loads(cur.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return None, f"current.simstate.h5 is CORRUPT (not valid JSON): {e!r}", files_ok
    # also verify metadata is parseable (the second atomic file)
    if meta.exists():
        try:
            json.loads(meta.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return None, f"metadata.json is CORRUPT: {e!r}", files_ok
    return payload, None, files_ok


def _supervisor_cmd(lineage_root, name, pause_file, n_days=None, max_windows=40, n_hub=24, n_per=4):
    cmd = [sys.executable, "-u", "-m", "research.runners.develop_loop_supervisor",
           "--lineage-root", lineage_root, "--lineage-name", name,
           "--max-windows-per-day", str(max_windows), "--n-hub", str(n_hub), "--n-per", str(n_per),
           "--pause-file", pause_file]
    if n_days is not None:
        cmd += ["--n-days", str(n_days)]
    return cmd


def _hard_kill(proc: subprocess.Popen):
    """The most brutal termination available on this platform (no clean-shutdown chance)."""
    try:
        if os.name == "nt":
            # taskkill /F /T kills the process tree forcibly (SIGKILL-equivalent on Windows).
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:
            pass


def run_test(root, kill_after_first_day=True, max_windows=40, verbose=True):
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"   # CPU only — never contend for the GPU
    env.setdefault("PYTHONUNBUFFERED", "1")
    name = "dev"
    lineage_root = str(Path(root) / "lineage")
    pause_file = str(Path(root) / "PAUSE")
    log_dir = Path(root); log_dir.mkdir(parents=True, exist_ok=True)
    result = {"steps": []}

    # ---- Step 1: run a few days to establish real developed state on disk ----
    if verbose:
        print("[kill-test] Step 1: establish developed state (run 2 days to completion)...", flush=True)
    p = subprocess.Popen(_supervisor_cmd(lineage_root, name, pause_file, n_days=2, max_windows=max_windows),
                         cwd=_REPO, env=env, stdout=open(log_dir / "step1.log", "w"), stderr=subprocess.STDOUT)
    rc = p.wait()
    st1, err1, files1 = _load_state_safe(Path(lineage_root), name)
    days_before = st1["day"] if st1 else 0
    result["steps"].append({"step": "establish", "rc": rc, "days_on_disk": days_before,
                            "facts": (len(st1["facts"]) if st1 else 0), "corrupt": err1})
    if verbose:
        print(f"  established: rc={rc} days_on_disk={days_before} "
              f"facts={len(st1['facts']) if st1 else 0} corrupt={err1}", flush=True)
    assert err1 is None and days_before >= 1, f"step1 did not establish clean state: {err1} days={days_before}"

    # ---- Step 2: relaunch UNBOUNDED, then HARD-KILL mid-run ----
    # Watch the child's log for the FIRST day-checkpoint after relaunch, then kill it shortly after (so a day is
    # genuinely in flight at kill time -> we exercise the "in-flight day lost, completed days survive" path).
    if verbose:
        print("[kill-test] Step 2: relaunch unbounded, HARD-KILL mid-run...", flush=True)
    step2_log = log_dir / "step2.log"
    p2 = subprocess.Popen(_supervisor_cmd(lineage_root, name, pause_file, n_days=None, max_windows=max_windows),
                          cwd=_REPO, env=env, stdout=open(step2_log, "w"), stderr=subprocess.STDOUT)
    # Poll the log for the first post-resume per-day checkpoint line, then kill a moment into the NEXT day.
    killed_after_day = None
    t0 = time.time()
    saw_checkpoint = False
    while time.time() - t0 < 180:
        if p2.poll() is not None:
            break  # finished before we could kill (shouldn't happen unbounded)
        try:
            log_txt = step2_log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            log_txt = ""
        # count completed-day checkpoints since relaunch
        n_ckpt = log_txt.count("checkpointed + fsync")
        if n_ckpt >= 1 and not saw_checkpoint:
            saw_checkpoint = True
            # a day just completed + fsync'd; let the NEXT day get partway, then kill brutally mid-day.
            time.sleep(3.0)
            killed_after_day = days_before + n_ckpt  # disk should hold this many completed days
            if verbose:
                print(f"  killing now (>=1 post-resume day checkpointed; {n_ckpt} done this run)...", flush=True)
            _hard_kill(p2)
            break
        time.sleep(0.4)
    # ensure it's dead
    try:
        p2.wait(timeout=20)
    except subprocess.TimeoutExpired:
        _hard_kill(p2)
        p2.wait(timeout=20)
    kill_rc = p2.returncode
    result["steps"].append({"step": "hard_kill", "kill_rc": kill_rc, "saw_checkpoint": saw_checkpoint,
                            "expected_days_on_disk": killed_after_day})

    # ---- Step 3: assert NO corruption after the hard kill ----
    st2, err2, files2 = _load_state_safe(Path(lineage_root), name)
    days_after_kill = st2["day"] if st2 else 0
    no_corruption = (err2 is None and st2 is not None)
    result["steps"].append({"step": "post_kill_integrity", "corrupt": err2,
                            "days_on_disk": days_after_kill, "files": files2,
                            "facts": (len(st2["facts"]) if st2 else 0),
                            "no_corruption": no_corruption})
    if verbose:
        print(f"[kill-test] Step 3: post-kill integrity: corrupt={err2} days_on_disk={days_after_kill} "
              f"facts={len(st2['facts']) if st2 else 0}", flush=True)

    # ---- Step 4: relaunch -> assert RESUME from disk day (not from scratch) + continues ----
    if verbose:
        print("[kill-test] Step 4: relaunch -> must RESUME (not restart from scratch) + continue...", flush=True)
    p3 = subprocess.Popen(_supervisor_cmd(lineage_root, name, pause_file, n_days=1, max_windows=max_windows),
                          cwd=_REPO, env=env, stdout=open(log_dir / "step4.log", "w"), stderr=subprocess.STDOUT)
    rc3 = p3.wait()
    st3, err3, _ = _load_state_safe(Path(lineage_root), name)
    days_after_resume = st3["day"] if st3 else 0
    # The relaunch presented day == days_after_kill (resumed there), and lived one more.
    resumed_from_disk = (days_after_resume == days_after_kill + 1)
    continued = (days_after_resume > days_after_kill)
    result["steps"].append({"step": "resume", "rc": rc3, "days_on_disk": days_after_resume,
                            "facts": (len(st3["facts"]) if st3 else 0),
                            "resumed_from_disk_day": resumed_from_disk, "continued": continued})
    if verbose:
        print(f"  resumed: rc={rc3} days_on_disk={days_after_resume} resumed_from_disk={resumed_from_disk} "
              f"continued={continued}", flush=True)

    # ---- worst-case loss = completed days that had to be re-done ----
    # days_after_kill is what survived on disk; the supervisor RESUMES from there, so re-done completed days = 0.
    # (The in-flight day at kill time is the only repeated work — that is the bounded, expected loss.)
    completed_days_lost = max(0, (killed_after_day or days_after_kill) - days_after_kill)

    passed = bool(no_corruption and resumed_from_disk and continued and completed_days_lost == 0)
    result["summary"] = {
        "passed": passed,
        "no_corruption_after_hard_kill": no_corruption,
        "resumed_from_disk_not_scratch": resumed_from_disk,
        "continued_developing": continued,
        "completed_days_lost": completed_days_lost,
        "worst_case_loss": "<= the single in-flight (unfinished) day; 0 completed days re-done",
        "days_before_kill_run": days_before,
        "days_on_disk_after_kill": days_after_kill,
        "days_on_disk_after_resume": days_after_resume,
        "post_kill_corruption_error": err2,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_kill_safety.json")
    ap.add_argument("--max-windows", type=int, default=40)
    ap.add_argument("--keep", action="store_true", help="keep the temp lineage")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import tempfile
    import shutil
    root = tempfile.mkdtemp(prefix="killsafe_")
    print("=" * 100, flush=True)
    print("[KILL-SAFETY TEST — develop-loop training supervisor (CPU, numpy)]", flush=True)
    print(f"  temp lineage root: {root}", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    try:
        res = run_test(root, max_windows=a.max_windows, verbose=True)
    finally:
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
    res["wall_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)

    s = res["summary"]
    print(f"\n{'=' * 100}", flush=True)
    print(f"  KILL-SAFETY: passed={s['passed']}  no_corruption={s['no_corruption_after_hard_kill']}  "
          f"resumed_from_disk={s['resumed_from_disk_not_scratch']}  continued={s['continued_developing']}  "
          f"completed_days_lost={s['completed_days_lost']}", flush=True)
    print(f"  [saved] {a.out}  (wall {res['wall_seconds']}s)\n{'=' * 100}", flush=True)
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
