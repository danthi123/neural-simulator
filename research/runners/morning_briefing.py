"""Morning briefing — run after autonomous overnight to see what landed.

Aggregates the night's commits, findings docs, in-flight processes,
chain waiter status, and result aggregations into one summary.

Usage:
    python -m research.runners.morning_briefing

    # Custom since-time (defaults to last 12 hours):
    python -m research.runners.morning_briefing --since "2026-05-03 20:00"

    # Only summary (no details):
    python -m research.runners.morning_briefing --short
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _run(cmd: list[str], cwd: Path = None) -> str:
    """Run a command, return stdout (or empty string on error)."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=cwd, timeout=30, encoding="utf-8", errors="replace",
        )
        return out.stdout
    except Exception:
        return ""


def _git_recent_commits(since: str) -> list[str]:
    """List commits since the given timestamp."""
    txt = _run(["git", "log", "--oneline", f"--since={since}", "main"])
    if not txt:
        return []
    return [line for line in txt.strip().split("\n") if line]


def _findings_recent(since_dt: datetime) -> list[Path]:
    """List findings docs created/modified since the given datetime."""
    findings_dir = Path("research/findings")
    if not findings_dir.exists():
        return []
    out = []
    for p in findings_dir.glob("*.md"):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            if mtime >= since_dt:
                out.append(p)
        except OSError:
            continue
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def _python_processes() -> list[dict]:
    """List active python processes (Windows)."""
    txt = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-Process python -ErrorAction SilentlyContinue | "
        "Select-Object Id,ProcessName,@{Name='MB';Expression={[math]::Round($_.WS/1MB,0)}} | "
        "ConvertTo-Json"
    ])
    if not txt:
        return []
    try:
        data = json.loads(txt)
        # Single result is a dict; multiple results is a list
        if isinstance(data, dict):
            data = [data]
        return data
    except json.JSONDecodeError:
        return []


def _waiter_status() -> str:
    """Return last 10 lines of the wait_biology_then_decide log."""
    log = Path("research/findings/raw/g11_bg/wait_biology_then_decide.log")
    if not log.exists():
        return "(waiter log not found)"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").strip().split("\n")
        return "\n".join(lines[-10:])
    except OSError:
        return "(waiter log unreadable)"


def _chain_status() -> dict:
    """Detect where we are in the autonomous decision chain.

    Returns:
        {
            "stage": "minimal_iso_running" | "biology_sweep_running" |
                     "biology_sweep_done" | "A1_running" | "A1_done" |
                     "B1_running" | "B1_done" | "unknown",
            "verdict": "A" | "A_weak" | "B" | None,
            "next_step": short string describing what's expected next
        }
    """
    raw = Path("research/findings/raw/g11_bg")
    if not raw.exists():
        return {"stage": "unknown", "verdict": None,
                "next_step": "raw findings dir not found"}

    waiter_log = raw / "wait_biology_then_decide.log"
    biology_master = raw / "biology-sweep.master.log"
    biology_master_alt = raw / "run_biology_sweep.master.log"
    minbio_master = raw / "minimum-biology.master.log"
    sanity_log = raw / "eval_sanity_check.log"

    # Parse verdict if present
    verdict = None
    if waiter_log.exists():
        try:
            text = waiter_log.read_text(encoding="utf-8", errors="replace")
            for line in reversed(text.split("\n")):
                if line.startswith("VERDICT:"):
                    v = line.split(":", 1)[1].strip()
                    verdict = v if v != "unknown" else None
                    break
        except OSError:
            pass

    # Detect stage
    biology_master_text = ""
    for p in (biology_master, biology_master_alt):
        if p.exists():
            try:
                biology_master_text = p.read_text(encoding="utf-8",
                                                   errors="replace")
                break
            except OSError:
                pass

    biology_complete = (
        "BIOLOGY SWEEP COMPLETE" in biology_master_text or
        "biology-sweep COMPLETE" in biology_master_text or
        "ALL BATCHES COMPLETE" in biology_master_text
    )

    # Check minimal_iso done count
    minimal_iso_done = sum(1 for _ in raw.glob("minimal_iso_seed*.pid.done"))

    if minbio_master.exists():
        try:
            minbio_text = minbio_master.read_text(encoding="utf-8",
                                                    errors="replace")
            minbio_complete = "minimum-biology COMPLETE" in minbio_text
        except OSError:
            minbio_complete = False
        stage = "A1_done" if minbio_complete else "A1_running"
        next_step = ("Next: review minimum_biology results, propose A2 "
                     "(cascade reintroduction)" if minbio_complete
                     else "A1 (minimum biology dose-response) running")
    elif sanity_log.exists():
        stage = "B1_running"
        next_step = "B1 (eval sanity check) running"
        # Could also check for completion marker
    elif biology_complete and verdict:
        stage = "biology_sweep_done"
        next_step = (f"Verdict {verdict}; A1/B1 should be launching shortly "
                     f"(or has finished, check log)")
    elif biology_master_text:
        stage = "biology_sweep_running"
        next_step = "Biology sweep running; verdict TBD"
    elif minimal_iso_done >= 6:
        stage = "minimal_iso_done_pre_biology"
        next_step = "Minimal-iso done, biology sweep should launch"
    elif minimal_iso_done > 0:
        stage = "minimal_iso_running"
        next_step = (f"Minimal-iso batch ({minimal_iso_done}/6 done); biology "
                     "sweep auto-launches when batch 2 completes")
    else:
        stage = "unknown"
        next_step = "No clear chain stage detected"

    return {
        "stage": stage,
        "verdict": verdict,
        "next_step": next_step,
        "minimal_iso_done": minimal_iso_done,
        "biology_complete": biology_complete,
    }


def _experiment_progress(label_prefix: str = "minimal_iso") -> dict[str, str]:
    """For each {label_prefix}_seed*.log file, return last [PROGRESS] line."""
    out = {}
    raw_dir = Path("research/findings/raw/g11_bg")
    if not raw_dir.exists():
        return out
    for log in sorted(raw_dir.glob(f"{label_prefix}_seed*.log")):
        if log.name.endswith(".err"):
            continue
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
            progress = [l for l in text.split("\n") if "[PROGRESS]" in l]
            out[log.stem] = progress[-1] if progress else "(no progress yet)"
        except OSError:
            out[log.stem] = "(log unreadable)"
    return out


def _aggregator_verdict(config_name: str) -> str | None:
    """Run result_aggregator with the given config; return verdict line."""
    txt = _run([
        sys.executable, "-m", "research.result_aggregator",
        "--config", config_name,
    ])
    if not txt:
        return None
    for line in txt.split("\n"):
        if "**Headline:**" in line:
            return line.replace("**Headline:**", "").strip()
        if "Headline:" in line:
            return line.split("Headline:", 1)[1].strip()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", type=str, default=None,
                    help="Show commits/findings since this time. Default: 12h ago.")
    ap.add_argument("--short", action="store_true", default=False,
                    help="Brief summary only (no per-section details).")
    args = ap.parse_args()

    if args.since is None:
        since_dt = datetime.now() - timedelta(hours=12)
        args.since = since_dt.strftime("%Y-%m-%d %H:%M")
    else:
        try:
            since_dt = datetime.strptime(args.since, "%Y-%m-%d %H:%M")
        except ValueError:
            since_dt = datetime.now() - timedelta(hours=12)

    # Set stdout to UTF-8 if Windows console (best-effort; some environments
    # ignore this).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print(f"MORNING BRIEFING - {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Since: {args.since}")
    print("=" * 70)

    # 0. Chain status (one-glance current state)
    cs = _chain_status()
    print(f"\n## Autonomous chain status")
    print(f"  Stage: {cs['stage']}")
    print(f"  Minimal-iso done: {cs['minimal_iso_done']}/6")
    print(f"  Biology sweep complete: {cs['biology_complete']}")
    if cs['verdict']:
        print(f"  Verdict: {cs['verdict']} (A=aligned>=4/6, A_weak=2-3/6, B=0-1/6)")
    print(f"  Next: {cs['next_step']}")

    # 1. Git commits
    commits = _git_recent_commits(args.since)
    print(f"\n## Git commits ({len(commits)})")
    if commits:
        for c in commits[:30]:
            print(f"  {c}")
        if len(commits) > 30:
            print(f"  ... and {len(commits) - 30} more")
    else:
        print("  (none)")

    if not args.short:
        # 2. New/modified findings
        findings = _findings_recent(since_dt)
        print(f"\n## Findings docs created/modified ({len(findings)})")
        if findings:
            for p in findings[:15]:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                print(f"  {mtime:%H:%M} {p.name}")
            if len(findings) > 15:
                print(f"  ... and {len(findings) - 15} more")
        else:
            print("  (none)")

        # 3. Active python processes
        procs = _python_processes()
        print(f"\n## Active python processes ({len(procs)})")
        if procs:
            for p in procs:
                pid = p.get("Id", "?")
                mb = p.get("MB", "?")
                print(f"  PID {pid}  WS={mb}MB")
        else:
            print("  (none)")

        # 4. Experiment progress (in-flight)
        for prefix in ["minimal_iso", "biology"]:
            progress = _experiment_progress(prefix)
            if progress:
                print(f"\n## {prefix} runs in progress")
                for label, line in progress.items():
                    # extract just the JSON for compact display
                    if "[PROGRESS]" in line:
                        json_txt = line.split("[PROGRESS]", 1)[1].strip()
                        try:
                            j = json.loads(json_txt)
                            current = j.get("current", "?")
                            total = j.get("total", "?")
                            elapsed = j.get("elapsed_seconds", 0)
                            phase = j.get("phase", "?")
                            print(f"  {label}: {current}/{total} {phase} "
                                  f"({elapsed:.0f}s elapsed)")
                        except (json.JSONDecodeError, ValueError):
                            print(f"  {label}: {line[:100]}")
                    else:
                        print(f"  {label}: {line[:100]}")

        # 5. Waiter status
        print(f"\n## Decision-chain waiter status (last 10 lines)")
        for line in _waiter_status().split("\n"):
            print(f"  {line}")

    # 6. Aggregator verdicts (always shown, even in --short)
    print(f"\n## Aggregator verdicts")
    for cfg in ["biology", "minimum_biology", "sanity_check"]:
        v = _aggregator_verdict(cfg)
        if v:
            # Truncate long verdicts
            v_short = v[:120] + "..." if len(v) > 120 else v
            print(f"  {cfg}: {v_short}")
        else:
            print(f"  {cfg}: (no data or aggregator failed)")

    print("\n" + "=" * 70)
    print("Suggested next:")
    print("  - If biology aligned: review research/findings/2026-05-04-biology-sweep-results.md")
    print("  - If chain failed:    python -m research.experiment_runner experiments/biology_sweep.yaml")
    print("  - Full picture:       cat research/findings/2026-05-04-autonomous-overnight-arc.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
