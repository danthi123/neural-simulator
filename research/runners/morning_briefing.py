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
