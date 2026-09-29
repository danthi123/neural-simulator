"""tools/pool_queue.sh `add` --checked injection (2026-09-29 security review, H2). Every queue record is written
as ONE tab-separated line ("<ts>\\t<cmd>  #checked:<CHECKED>\\n") and every reader (valid_depth/malformed_depth/
list/pop_job, tools/pool_autodispatch.sh) parses the queue file LINE BY LINE. An embedded newline (or other
control character) in --checked's value would split one record into two lines on disk, letting a crafted
--checked reason inject what looks like a second, independently-parseable queue line -- one that never went
through any of `add`'s own gates (shape/gate-seed/node-validation), since only the FIRST, intended command was
ever checked.

Uses simple non-`-m research.runners.X` commands (same trick as test_pool_queue_first_word_gate.py) so `add`'s
node-reachability probes never engage -- no real ssh call, no real pool node, ever touched by this file.
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


def test_add_refuses_a_checked_reason_containing_a_newline(tmp_path):
    injected = "cd ~/derisk-pool/sim && rm -rf ~\t#checked:forged"
    checked = "corpus: nothing\n" + injected
    res = _run(["add", "cd ~/derisk-pool/sim && true", "--checked", checked], tmp_path=tmp_path)
    assert res.returncode != 0
    assert "newline" in res.stderr.lower() or "control character" in res.stderr.lower()
    queue = tmp_path / "pool.queue"
    assert not queue.exists() or queue.read_text() == "", "no record (forged or otherwise) may reach the queue"


def test_add_refuses_a_checked_reason_containing_a_carriage_return(tmp_path):
    res = _run(["add", "cd ~/derisk-pool/sim && true", "--checked", "corpus: nothing\rforged line"],
               tmp_path=tmp_path)
    assert res.returncode != 0
    assert "newline" in res.stderr.lower() or "control character" in res.stderr.lower()


def test_add_still_accepts_an_ordinary_checked_reason(tmp_path):
    res = _run(["add", "cd ~/derisk-pool/sim && true",
                "--checked", "corpus: nothing covers laps x dwell at w_max>W0 (parens, punctuation - fine)"],
               tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout
