#!/usr/bin/env python3
"""Re-arm reminder for the attention watcher (owner 2026-10-28).

The owner reboots the workstation to game about once a day, which ends this Claude session and the background
`tools/attention_watch.py --loop` with it; compaction and a fresh session lose it too, and nothing re-armed it unless the
owner asked. Wired on SessionStart and UserPromptSubmit (both add stdout to context): when no `attention_watch.py --loop`
process is alive on this host, it prints the one command to run before anything else. Silent when the watcher is live.
FAIL-OPEN: any error prints nothing and exits 0.
"""
import os
import subprocess
import sys


def main():
    try:
        r = subprocess.run(["pgrep", "-f", "attention_watch.py --loop"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return 0
        root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
        print("⛔ ATTENTION WATCHER NOT RUNNING (reboot, new session, compaction, or it fired and was not re-armed). "
              "Arm it BEFORE other work, as a Bash run_in_background call: "
              f"cd {root} && .venv/bin/python tools/attention_watch.py --loop "
              "(re-arm after handling each event; tools/attention_watch.py --once shows the current events).")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
