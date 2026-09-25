"""Tests for research/coordination/b2b_queue_next_wave.sh's check_queued_lines (2026-09-25 review LOW, of
research/b2b-torn-cells-redo: "check_queued_lines requires every queued .../_shards/b2b0924-base/ line to end
with the exact AMENDMENT 1 reason. While the redo lines are queued, --status or a wave run will report them as
'malformed (torn / wrong pin / wrong reason)' and exit 1, a false torn-line alarm.").

check_queued_lines scans research/queue/pool.queue for lines whose command contains "/_shards/<TAG>/" and refuses
(exit 1) unless each one starts with the pinned `cd .../revisions/<F> && ` prefix and ends with
`/lb.json  #checked:<reason>` for a RECOGNISED reason -- originally only Amendment 1's one fixed reason string.
A torn-cell redo line (Amendment 2) carries a per-cell reason (it names the specific fragment claim and host), so
it can never equal that one fixed string; the fix accepts any reason beginning with Amendment 2's own shared,
distinctive prefix instead of an exact match.

Runs the REAL script (`bash research/coordination/b2b_queue_next_wave.sh --status`) against a tmp POOL_QUEUE_PATH
-- never research/queue/pool.queue itself -- so this cannot touch the live queue. `--status` only reads the queue
and the two real, committed files (b2b0924_base_jobs.txt, b2b0924_corpus_sha256.tsv) it needs to start up; it
queues nothing.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "coordination" / "b2b_queue_next_wave.sh"
F = "a308f1e09babcc9ed096c3c8046d00040391368c"

AMENDMENT1_REASON = (
    "prereg research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md "
    "(F a308f1e09) AMENDMENT 1; B2b base; mem_gb=8"
)
REDO_REASON_D5 = (
    "prereg research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md "
    "(F a308f1e09) AMENDMENT 2; torn-cell redo of s43/d5-consolidate, different host than the pool2 fragment "
    "(research/findings/2026-09-25-dispatcher-fragment-jobs-audit.md); "
    "logged research/coordination/b2b0924_reruns.tsv; pool_node=pool41; mem_gb=8"
)


def _queue_line(reason: str, seed: str = "s99", faculty: str = "fake-faculty") -> str:
    epoch = int(time.time())
    job = (
        f"cd ~/derisk-pool/revisions/{F} && true "
        f"research/findings/raw/_load_bearing/_shards/b2b0924-base/{seed}/{faculty}/lb.json"
    )
    return f"{epoch}\t{job}  #checked:{reason}\n"


def _run_status(tmp_path, queue_text: str) -> subprocess.CompletedProcess:
    q = tmp_path / "pool.queue"
    q.write_text(queue_text, encoding="utf-8")
    env = dict(os.environ)
    env["POOL_QUEUE_PATH"] = str(q)
    return subprocess.run(
        ["bash", str(SCRIPT), "--status"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30
    )


def test_amendment1_line_alone_is_well_formed(tmp_path):
    res = _run_status(tmp_path, _queue_line(AMENDMENT1_REASON, "s99", "amendment1-ok"))
    assert res.returncode == 0, res.stderr
    assert "malformed" not in res.stderr


def test_amendment2_redo_line_is_accepted_not_flagged_as_malformed(tmp_path):
    """The exact defect this fix closes: before it, this redo line alone made --status exit 1."""
    res = _run_status(tmp_path, _queue_line(REDO_REASON_D5, "s43", "d5-consolidate"))
    assert res.returncode == 0, "redo line wrongly flagged malformed: %s" % res.stderr
    assert "malformed" not in res.stderr


def test_amendment1_and_redo_lines_together_are_both_accepted(tmp_path):
    text = _queue_line(AMENDMENT1_REASON, "s99", "amendment1-ok") + _queue_line(
        REDO_REASON_D5, "s43", "d5-consolidate"
    )
    res = _run_status(tmp_path, text)
    assert res.returncode == 0, res.stderr
    assert "well-formed: 2" in res.stdout


def test_a_genuinely_wrong_reason_is_still_flagged(tmp_path):
    """Mutation check: an unrecognised reason (neither Amendment 1's exact text nor Amendment 2's shared
    prefix) must still be refused -- the fix must not have loosened the check into accepting everything."""
    res = _run_status(tmp_path, _queue_line("some other reason entirely", "s99", "bad-reason"))
    assert res.returncode == 1
    assert "malformed" in res.stderr


def test_a_torn_line_missing_the_pinned_cd_prefix_is_still_flagged(tmp_path):
    """Mutation check: the head-prefix requirement (the original torn-line class this script guards against)
    must still apply even to a line that otherwise carries Amendment 2's reason."""
    epoch = int(time.time())
    job = "true research/findings/raw/_load_bearing/_shards/b2b0924-base/s43/d5-consolidate/lb.json"  # no `cd ...`
    text = f"{epoch}\t{job}  #checked:{REDO_REASON_D5}\n"
    res = _run_status(tmp_path, text)
    assert res.returncode == 1
    assert "malformed" in res.stderr


def test_a_redo_reason_for_a_different_amendment_2_cell_is_also_accepted(tmp_path):
    """The prefix match must not be hard-coded to one cell's exact suffix."""
    causal_reason = (
        "prereg research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md "
        "(F a308f1e09) AMENDMENT 2; torn-cell redo of s42/causal-whatif, different host than the pool1 fragment "
        "(research/findings/2026-09-25-dispatcher-fragment-jobs-audit.md); "
        "logged research/coordination/b2b0924_reruns.tsv; pool_node=pool42; mem_gb=8"
    )
    res = _run_status(tmp_path, _queue_line(causal_reason, "s42", "causal-whatif"))
    assert res.returncode == 0, res.stderr
