"""PAUSE-sentinel test for the develop-loop supervisor (mid-run pause + resume).

Confirms: (1) creating the PAUSE sentinel mid-run stops the supervisor CLEANLY at the next day boundary with exit
code 42 (NOT a runaway loop); (2) the state is durably persisted at the pause; (3) deleting the sentinel +
relaunching RESUMES from the paused day and continues. CPU-only (numpy); never contends for the GPU.

Run: SIM_BACKEND=numpy python -u -m research.runners._curriculum_pause_test --out research/findings/raw/_pause_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

PAUSE_EXIT_CODE = 42


def _load_day(lineage_dir: Path, name: str):
    cur = lineage_dir / name / "current.simstate.h5"
    if not cur.exists():
        return None
    try:
        return json.loads(cur.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _cmd(lineage_root, name, pause_file, n_days=None, max_windows=40, n_hub=24, n_per=4):
    c = [sys.executable, "-u", "-m", "research.runners.develop_loop_supervisor",
         "--lineage-root", lineage_root, "--lineage-name", name,
         "--max-windows-per-day", str(max_windows), "--n-hub", str(n_hub), "--n-per", str(n_per),
         "--pause-file", pause_file]
    if n_days is not None:
        c += ["--n-days", str(n_days)]
    return c


def run_test(root, max_windows=40, verbose=True):
    env = dict(os.environ); env["SIM_BACKEND"] = "numpy"; env.setdefault("PYTHONUNBUFFERED", "1")
    name = "dev"
    lineage_root = str(Path(root) / "lineage")
    pause_file = str(Path(root) / "PAUSE")
    log = Path(root) / "pause.log"; Path(root).mkdir(parents=True, exist_ok=True)
    res = {}

    # Launch UNBOUNDED (n_days=None) with a hard safety cap of 8 days so a pause REGRESSION cannot run away.
    if verbose:
        print("[pause-test] launch unbounded(cap 8); set PAUSE after the first day checkpoints...", flush=True)
    p = subprocess.Popen(_cmd(lineage_root, name, pause_file, n_days=8, max_windows=max_windows),
                         cwd=_REPO, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    set_pause = False
    t0 = time.time()
    while time.time() - t0 < 240:
        if p.poll() is not None:
            break
        try:
            txt = log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            txt = ""
        if (not set_pause) and txt.count("checkpointed + fsync") >= 1:
            Path(pause_file).touch()    # request pause mid-run (a day is now in flight)
            set_pause = True
            if verbose:
                print("  [pause-test] PAUSE sentinel set (mid second day).", flush=True)
        time.sleep(0.4)
    rc = p.wait(timeout=60)
    txt = log.read_text(encoding="utf-8", errors="ignore")
    days_at_pause = txt.count("checkpointed + fsync")
    paused_clean = (rc == PAUSE_EXIT_CODE)
    st_at_pause = _load_day(Path(lineage_root), name)
    disk_day_at_pause = st_at_pause["day"] if st_at_pause else 0
    res["pause"] = {"exit_code": rc, "clean_pause_exit_42": paused_clean,
                    "days_checkpointed_before_pause": days_at_pause,
                    "disk_day_at_pause": disk_day_at_pause,
                    "no_runaway": (days_at_pause <= 4),   # must NOT have run to the 8-day cap
                    "state_persisted": st_at_pause is not None}
    if verbose:
        print(f"  paused: exit={rc} (expect 42) days_checkpointed={days_at_pause} disk_day={disk_day_at_pause}",
              flush=True)

    # RESUME: delete sentinel, relaunch (+1 day) -> must continue from disk_day_at_pause.
    if verbose:
        print("[pause-test] delete sentinel + relaunch -> must RESUME from paused day...", flush=True)
    Path(pause_file).unlink(missing_ok=True)
    p2 = subprocess.Popen(_cmd(lineage_root, name, pause_file, n_days=1, max_windows=max_windows),
                          cwd=_REPO, env=env, stdout=open(Path(root) / "resume.log", "w"), stderr=subprocess.STDOUT)
    rc2 = p2.wait(timeout=240)
    st_after = _load_day(Path(lineage_root), name)
    disk_day_after = st_after["day"] if st_after else 0
    resumed = (disk_day_after == disk_day_at_pause + 1)
    res["resume"] = {"exit_code": rc2, "disk_day_after_resume": disk_day_after,
                     "resumed_from_paused_day": resumed}
    if verbose:
        print(f"  resumed: exit={rc2} disk_day_after={disk_day_after} resumed_from_paused_day={resumed}",
              flush=True)

    res["passed"] = bool(paused_clean and res["pause"]["no_runaway"] and res["pause"]["state_persisted"]
                         and resumed)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_pause_test.json")
    ap.add_argument("--max-windows", type=int, default=40)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import tempfile
    import shutil
    root = tempfile.mkdtemp(prefix="pausetest_")
    print("=" * 100, flush=True)
    print("[PAUSE-SENTINEL TEST — develop-loop supervisor (CPU, numpy)]", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    try:
        res = run_test(root, max_windows=a.max_windows, verbose=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    res["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\n{'=' * 100}", flush=True)
    print(f"  PAUSE TEST: passed={res['passed']}  clean_pause_exit_42={res['pause']['clean_pause_exit_42']}  "
          f"no_runaway={res['pause']['no_runaway']}  resumed={res['resume']['resumed_from_paused_day']}", flush=True)
    print(f"  [saved] {a.out}  (wall {res['wall_seconds']}s)\n{'=' * 100}", flush=True)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
