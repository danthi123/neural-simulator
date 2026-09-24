"""Every git hook under tools/githooks/ must be committed EXECUTABLE (mode 100755).

Git silently SKIPS a core.hooksPath hook that is not executable (it prints "hook was ignored because it's not set as
executable" and proceeds). tools/githooks/pre-merge-commit was committed 100644 on research/s06-hygiene-ledger-hook
through two review rounds; a reviewer proved end to end that a merge introducing a W2 violation went through
unblocked in a fresh clone. Nothing else pins the bit, so this test does.
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_tracked_githook_is_committed_executable():
    out = subprocess.run(["git", "ls-files", "-s", "tools/githooks/"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    rows = [ln.split(None, 3) for ln in out.splitlines() if ln.strip()]
    assert rows, "no hooks tracked under tools/githooks/"
    not_exec = [r[3] for r in rows if r[0] != "100755"]
    assert not_exec == [], "committed without the executable bit (git would silently skip them): %s" % not_exec
