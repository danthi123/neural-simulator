"""Tests for tools/pool_queue.sh `add`'s first-word-runnable gate (2026-09-25 review, fix round 3, LOW: "the
six 'A2 wiring seed N: ...' claims exited 127 -- bash tried to run 'A2' as a command"). A stray prose label
accidentally left at the head of a queued command dispatches, fails instantly on "command not found", and the
`&&`-chained real command after it never runs -- this gate refuses such a command at enqueue time.

Uses commands with NO `-m research.runners.X` pattern so `add`'s existing MOD-based local-import / remote-node
checks never engage (this is exactly what the real 'A2 wiring seed N: mem_gb=8 && cd ~/derisk-pool/revisions/
<sha> && ...' shape looked like structurally: env-assignment-and-cd chains around the eventual runner
invocation), keeping these tests fast and independent of any pool node / python environment.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"


def _run(cmd, extra_env=None, tmp_path=None):
    env = dict(os.environ)
    if tmp_path is not None:
        env.setdefault("POOL_QUEUE_PATH", str(tmp_path / "pool.queue"))
        env.setdefault("POOL_SSH_CONFIG", str(tmp_path / "no-such-pool-ssh-config"))
        env.setdefault("POOL_EXTRA_NODES_FILE", str(tmp_path / "no-such-extra-nodes"))
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(POOL_QUEUE), *cmd], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=30)


def test_add_refuses_a_bare_unresolvable_first_word(tmp_path):
    # The exact real-world shape: a prose label ("A2 wiring seed 1:") accidentally left at the head of the
    # command. Bash would try to run "A2" as a command and exit 127 before the real `&&`-chained work ever runs.
    res = _run(["add", "A2 wiring seed 1: mem_gb=8 && cd ~/derisk-pool/revisions/abc1234 && true",
                "--checked", "x"], tmp_path=tmp_path)
    assert res.returncode == 2
    assert "REFUSED" in res.stderr   # wording now comes from tools/queue_job_shape_check.sh (merged 2026-09-25)
    assert "A2" in res.stderr
    queue = tmp_path / "pool.queue"
    assert not queue.exists() or queue.read_text() == "", "the malformed job must never reach the queue"


def test_add_accepts_a_var_assignment_first_word(tmp_path):
    res = _run(["add", "mem_gb=8 && cd ~/derisk-pool/revisions/abc1234 && true", "--checked", "x"],
               tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout


def test_add_accepts_cd_as_the_first_word(tmp_path):
    res = _run(["add", "cd ~/derisk-pool/sim && true", "--checked", "x"], tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr


def test_add_accepts_a_path_like_first_word(tmp_path):
    res = _run(["add", ".venv/bin/python -c 'pass'", "--checked", "x"], tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr


def test_add_accepts_ssh_as_the_first_word(tmp_path):
    res = _run(["add", "ssh somehost true", "--checked", "x"], tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
