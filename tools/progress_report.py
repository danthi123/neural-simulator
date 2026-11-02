"""progress_report: the new core's progress in a few plain lines, from the repo alone (stdlib only).

Owner 2026-11-02: progress is what matters, and the board and reports must stay current without anyone remembering
to update them. One source, three readers: Hermes's phone report (sim-report.py appends this), the Claude session
start, and the board (`--board` rewrites one task, "New core: progress (auto-updated)", from the same lines; run on
the workstation by the user timer tools/progress-board.timer, since the board token lives there).

  python3 tools/progress_report.py [--repo PATH] [--board]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re
import subprocess

BOARD_TITLE = "New core: progress (auto-updated)"


def _git(repo, *args):
    try:
        return subprocess.run(["git", "-c", "safe.directory=" + repo, "-C", repo, *args], capture_output=True,
                              text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def next_milestone(repo, today):
    """The first row of the brain/ARCHITECTURE.md timeline table dated today or later."""
    try:
        text = open(os.path.join(repo, "brain/ARCHITECTURE.md"), encoding="utf-8").read()
    except OSError:
        return None
    for date, target in re.findall(r"^\| (\d{4}-\d{2}-\d{2}) \| (.+?) \|$", text, re.M):
        d = dt.date.fromisoformat(date)
        if d >= today:
            return d, target
    return None


def latest_transcript(repo):
    """(date, {arm: score}) of the newest weekly transcript: an explicit 'score' line per arm if present, else the
    count of ✅ marks under each '## <arm>' heading."""
    files = sorted(glob.glob(os.path.join(repo, "brain/eval/transcripts/*.md")))
    if not files:
        return None
    path = files[-1]
    arms, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^\*{0,2}score[^:]*\*{0,2}:\s*(.+)$", line.strip(), re.I)
        if m:
            arms["score"] = m.group(1).strip()
            continue
        h = re.match(r"^## (.+)$", line)
        if h:
            cur = h.group(1).strip()
            arms.setdefault(cur, 0)
        elif cur is not None and "✅" in line and isinstance(arms.get(cur), int):
            arms[cur] += 1
    return os.path.basename(path)[:-3], arms


def commit_share(repo, days=7):
    log = _git(repo, "log", f"--since={days}.days", "--name-only", "--pretty=format:@@")
    commits = [c for c in log.split("@@") if c.strip()]
    brain = sum(1 for c in commits if any(f.startswith("brain/") for f in c.split()))
    return brain, len(commits)


def lines(repo):
    today = dt.date.today()
    out = []
    ms = next_milestone(repo, today)
    if ms:
        out.append(f"Next milestone {ms[0]} ({(ms[0] - today).days} days): {ms[1]}")
    t = latest_transcript(repo)
    if t:
        age = (today - dt.date.fromisoformat(t[0][:10])).days if re.match(r"\d{4}-\d{2}-\d{2}", t[0]) else -1
        scores = ", ".join(f"{k} {v}" for k, v in t[1].items())
        out.append(f"Weekly conversation test {t[0]} ({age} days old): {scores}" + ("  STALE" if age > 7 else ""))
    else:
        out.append("Weekly conversation test: none yet")
    b, n = commit_share(repo)
    if n:
        out.append(f"Last 7 days: {b} of {n} commits touched the brain ({100 * b // n}%)")
    return out


def update_board(repo, text):
    """Rewrite the one auto-updated board task (create it once). Uses tools/vikunja.sh (token stays outside the repo)."""
    vk = os.path.join(repo, "tools/vikunja.sh")
    listing = subprocess.run(["bash", vk, "list-tasks", "2"], capture_output=True, text=True, timeout=60).stdout
    m = re.search(r"#(\d+) " + re.escape(BOARD_TITLE), listing)
    if m:
        tid = m.group(1)
    else:
        made = subprocess.run(["bash", vk, "create-task", "2", BOARD_TITLE, text, "5"], capture_output=True,
                              text=True, timeout=60).stdout
        tid = re.search(r"(\d+)", made).group(1)
        subprocess.run(["bash", vk, "label-task", tid, "8"], capture_output=True, timeout=60)
    subprocess.run(["bash", vk, "set-desc", tid, text], capture_output=True, timeout=60)
    return tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--board", action="store_true")
    a = ap.parse_args()
    out = lines(a.repo)
    print("\n".join(out))
    if a.board:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        tid = update_board(a.repo, f"Updated {stamp} (automatic). " + " | ".join(out) +
                           " (Ref: tools/progress_report.py; plan: brain/ARCHITECTURE.md §4 timeline.)")
        print(f"board task #{tid} updated")


if __name__ == "__main__":
    main()
