"""
SWR investigation status — quick health check during autonomous overnight.

Run on wakeup to see:
  * Which orchestrators are alive
  * What's currently in flight
  * Latest log activity per seed
  * Whether anything has crashed

Usage:
    python -m research.runners.swr_status
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "research" / "findings" / "raw" / "g11_bg"


def is_pid_alive(pid: int) -> bool:
    """Windows: ask tasklist."""
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=2,
        )
        return f'"{pid}"' in r.stdout or str(pid) in r.stdout
    except Exception:
        return False


def file_age_seconds(path: Path) -> float:
    if not path.exists():
        return -1
    return time.time() - path.stat().st_mtime


def fmt_age(s: float) -> str:
    if s < 0: return "—"
    if s < 60: return f"{int(s)}s"
    if s < 3600: return f"{int(s/60)}m"
    return f"{s/3600:.1f}h"


def main():
    print("=" * 70)
    print(f"SWR investigation status @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Orchestrator processes
    print("\n# Orchestrator processes")
    for label, pid_file in [
        ("Master 4-seed batch", RAW_DIR / "run_swr_remaining.master.orchestrator-pid"),
        ("Waiter (4-seed → H4 → H1 chain)", RAW_DIR / "wait_h4_h1.orchestrator-pid"),
    ]:
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                alive = is_pid_alive(pid)
                print(f"  {label}: PID {pid} {'ALIVE' if alive else 'DEAD'}")
            except Exception as e:
                print(f"  {label}: ERROR ({e})")
        else:
            print(f"  {label}: no PID file")

    # 2. Currently-running training subprocesses
    print("\n# Currently in-flight runs (via *.pid scan)")
    pid_files = sorted(RAW_DIR.glob("*.pid"))
    if not pid_files:
        print("  (no .pid files — chain may be between batches)")
    for pf in pid_files:
        if pf.name.endswith(".master.pid") or pf.name.endswith(".orchestrator-pid"):
            continue
        try:
            pid = int(pf.read_text().strip())
            alive = is_pid_alive(pid)
            log_file = pf.with_suffix(".log")
            log_age = file_age_seconds(log_file)
            tail = ""
            if log_file.exists():
                try:
                    text = log_file.read_text(errors="ignore")
                    lines = [ln for ln in text.splitlines() if ln.strip()]
                    tail = lines[-1] if lines else ""
                except Exception:
                    pass
            print(f"  {pf.stem}: PID {pid} {'ALIVE' if alive else 'DEAD'} "
                  f"(log {fmt_age(log_age)} ago)")
            if tail:
                print(f"    └─ {tail[:120]}")
        except Exception as e:
            print(f"  {pf.name}: ERROR ({e})")

    # 3. Master log states
    print("\n# Master logs")
    for log_path, label in [
        (RAW_DIR / "run_swr_remaining.master.log", "v2+SWR 4-seed batch"),
        (RAW_DIR / "wait_h4_h1.log",             "H4-then-H1 waiter"),
        (RAW_DIR / "run_h4_then_h1.super.log",   "H4-then-H1 super-orchestrator"),
        (RAW_DIR / "run_h4.master.log",          "H4 batch"),
        (RAW_DIR / "run_h1.master.log",          "H1 batch"),
    ]:
        if log_path.exists():
            try:
                text = log_path.read_text(errors="ignore").replace("\x00", "")
                lines = [ln for ln in text.splitlines() if ln.strip()]
                last = lines[-1] if lines else "(empty)"
                age = file_age_seconds(log_path)
                print(f"  {label} ({fmt_age(age)} ago):")
                print(f"    └─ {last[:120]}")
            except Exception as e:
                print(f"  {label}: ERROR ({e})")
        else:
            print(f"  {label}: not yet")

    # 4. Result files seen
    print("\n# Result files (text_eval_*.json)")
    patterns = {
        "v2+SWR (default)": "text_eval_v2_swr500_seed*.json",
        "H1 balanced":      "text_eval_h1_balanced_seed*.json",
        "H4 isolation":     "text_eval_h4_isolation_seed*.json",
    }
    seeds = [42, 43, 44, 100, 101, 102]
    for label, pat in patterns.items():
        existing = []
        for s in seeds:
            p = RAW_DIR / pat.replace("*", f"{s}")
            if p.exists():
                existing.append(s)
        print(f"  {label}: seeds {existing or '(none yet)'}")

    print()


if __name__ == "__main__":
    main()
