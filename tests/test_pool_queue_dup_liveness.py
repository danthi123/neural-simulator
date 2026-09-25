"""Subprocess-level tests for the pool.running half of tools/pool_queue.sh's duplicate guard.

THE BUG (measured 2026-09-25 07:20, see research/FAILURE_LOG.md and the "DUPLICATE GUARD" comment
in tools/pool_queue.sh's `add` case): a pool.running record is `date<TAB>node<TAB>job`, so `cut -f2-`
on it yields "<node>\\t<job>" -- never a bare command -- and the job field itself carries a
`POOL_CHECKED_REASON=<%q token> ` prefix the dispatcher's pop_job() adds, which the old
"#checked:"-stripping regex never touched either. The running-set half of the guard could therefore
NEVER match, so identical commands were re-queued and re-dispatched while an earlier claim was still
alive on a node.

The fix parses the record correctly, strips the POOL_CHECKED_REASON prefix (bash %q emits either a
backslash-escaped token or, for control characters, a $'...'-quoted one), and never trusts a text
match alone: a stopped/crashed job leaves its pool.running line behind, so a match is only a
CANDIDATE until job_liveness_on_node() confirms the claimed node's own /proc/*/environ still carries
that exact job's JOB_B64. These tests drive the real script (no internal function is imported/copied)
against a stubbed `ssh` on PATH, mirroring tests/test_pool_ssh_config_plumbing.py and
tests/test_pool_autodispatch_workflow.py's stub-binary approach -- no real network calls, no real
pool node, and the live queue/running files under research/queue/ are never touched (POOL_QUEUE_PATH
always points into tmp_path).
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"

CMD = "echo dedup_test_job_A"                 # deliberately has no `-m research.runners.` --
                                               # that keeps the MOD/argparse/node-reachability gates
                                               # (which are unrelated to this guard) out of the way,
                                               # so only the duplicate-guard's own ssh calls happen.
CHECKED = "test: exercising the pool.running duplicate-guard liveness check"


def _q_reason(reason: str) -> str:
    """The exact bash `printf '%q'` encoding pop_job() would produce for this reason -- shelling out
    to real bash rather than reimplementing %q in Python, so the test fixtures are byte-identical to
    what the live dispatcher writes."""
    out = subprocess.run(["bash", "-c", 'printf "%q" "$1"', "_", reason],
                          capture_output=True, text=True, timeout=10, check=True)
    return out.stdout


def _make_ssh_stub(tmp_path: Path) -> tuple[Path, Path]:
    """A stub `ssh` that logs its full argv and answers the two calls job_liveness_on_node makes:
    a bare reachability probe (`ssh ... <node> true`) and an environ-scan probe whose command text
    contains `JOB_B64=...`. Behaviour is controlled per-call via env vars (read by the stub at run
    time, so one stub file serves every test): UNREACHABLE_NODES (space-separated node names that
    fail EVERY call, reachability included) and ALIVE_NODE (the one node whose environ-scan reports
    a match; every other reachable node reports no match, i.e. DEAD)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    log.write_text("")
    stub = bin_dir / "ssh"
    stub.write_text(r"""#!/usr/bin/env bash
echo "$*" >> "$SSH_LOG"
node=""
skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  case "$a" in
    -F|-o) skip=1; continue ;;
    -n) continue ;;
    -*) continue ;;
    *) node="$a"; break ;;
  esac
done
for u in ${UNREACHABLE_NODES:-}; do
  [ "$u" = "$node" ] && exit 255
done
case "$*" in
  *"JOB_B64="*)
    [ "$node" = "${ALIVE_NODE:-}" ] && exit 0
    exit 1
    ;;
esac
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def _run(args: list[str], bin_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}:{full_env.get('PATH', '')}"
    full_env.update(env)
    return subprocess.run(["bash", str(POOL_QUEUE), *args], cwd=ROOT, env=full_env,
                           capture_output=True, text=True, timeout=30)


def _seed_running(tmp_path: Path, node: str, job_field: str, when: str = "2026-09-24 05:00:00") -> Path:
    running = tmp_path / "pool.running"
    running.write_text(f"{when}\t{node}\t{job_field}\n")
    return running


def _base_env(tmp_path: Path, ssh_log: Path, *, alive_node: str = "", unreachable: str = "") -> dict[str, str]:
    return {
        "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
        "SSH_LOG": str(ssh_log),
        "ALIVE_NODE": alive_node,
        "UNREACHABLE_NODES": unreachable,
    }


# (a) a running record whose command matches and the fake node reports ALIVE -> refused (exit 2).
def test_matching_running_record_alive_on_its_node_refuses(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = f"POOL_CHECKED_REASON={_q_reason(CHECKED)} {CMD}"
    _seed_running(tmp_path, "pool41", job_field)
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, _base_env(tmp_path, ssh_log, alive_node="pool41"))
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "RUNNING" in res.stderr
    assert "FORCE_DUP=1" in res.stderr
    # nothing was appended to the queue
    q = tmp_path / "pool.queue"
    assert q.read_text().strip() == ""


# (b) same match but the node reports DEAD (no matching JOB_B64 in its /proc) -> allowed, queued.
def test_matching_running_record_dead_on_its_node_is_queued(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = f"POOL_CHECKED_REASON={_q_reason(CHECKED)} {CMD}"
    _seed_running(tmp_path, "pool41", job_field)
    # alive_node="" (no default) or set to a different node -> pool41's environ-scan reports no match
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, _base_env(tmp_path, ssh_log, alive_node="pool99"))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "no longer alive" in res.stderr
    assert "queued" in res.stdout
    q = tmp_path / "pool.queue"
    assert CMD in q.read_text()


# (c) the claimed node cannot be reached -> refused (fail closed); FORCE_DUP=1 overrides.
def test_matching_running_record_unreachable_node_fails_closed(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = f"POOL_CHECKED_REASON={_q_reason(CHECKED)} {CMD}"
    _seed_running(tmp_path, "pool41", job_field)
    env = _base_env(tmp_path, ssh_log, unreachable="pool41")
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "could not be reached" in res.stderr or "UNREACH" in res.stderr.upper()
    q = tmp_path / "pool.queue"
    assert q.read_text().strip() == ""


def test_matching_running_record_unreachable_node_force_dup_overrides(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = f"POOL_CHECKED_REASON={_q_reason(CHECKED)} {CMD}"
    _seed_running(tmp_path, "pool41", job_field)
    env = _base_env(tmp_path, ssh_log, unreachable="pool41")
    env["FORCE_DUP"] = "1"
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 0, res.stdout + res.stderr
    q = tmp_path / "pool.queue"
    assert CMD in q.read_text()


# (d) a non-matching pool.running record must never trigger an ssh call at all.
def test_non_matching_running_record_never_triggers_ssh(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    other_job = f"POOL_CHECKED_REASON={_q_reason('unrelated reason')} echo some-completely-different-job"
    _seed_running(tmp_path, "pool41", other_job)
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, _base_env(tmp_path, ssh_log, alive_node="pool41"))
    assert res.returncode == 0, res.stdout + res.stderr
    assert ssh_log.read_text() == "", f"ssh was called for a non-matching record: {ssh_log.read_text()!r}"
    q = tmp_path / "pool.queue"
    assert CMD in q.read_text()


# (e) the %q prefix-stripping must handle a reason containing spaces, parens, commas and colons --
# the real D6 duplicate's reason text, reproduced verbatim.
D6_REASON = ("D6 N=2000 OOM root-cause found: pool_autodispatch.sh commit 3308e087c "
             "(2026-09-23 11:33, same day) + memory reservations added, re-verify before re-queue")


def test_prefix_stripping_handles_spaces_parens_commas_colons_in_the_reason(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    escaped = _q_reason(D6_REASON)
    # sanity: this reason really does %q-escape to something containing the tricky characters
    # unescaped (comma/colon) and escaped (space/paren) -- if bash's own %q behaviour ever changes
    # this assertion documents what we're actually testing against.
    assert "(2026-09-23" not in escaped or "\\(2026-09-23" in escaped
    job_field = f"POOL_CHECKED_REASON={escaped} {CMD}"
    _seed_running(tmp_path, "pool42", job_field)
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, _base_env(tmp_path, ssh_log, alive_node="pool42"))
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool42" in res.stderr
    assert "RUNNING" in res.stderr


# Sanity: the queue-side half of the guard (untouched logic) still refuses a plain in-queue duplicate,
# with no ssh call at all (no node is involved in a queue-vs-queue comparison).
def test_queue_side_duplicate_guard_untouched_and_ssh_free(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    q = tmp_path / "pool.queue"
    q.write_text(f"1700000000\t{CMD}  #checked:{CHECKED}\n")
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, _base_env(tmp_path, ssh_log))
    assert res.returncode == 2, res.stdout + res.stderr
    assert "already queued" in res.stderr
    assert ssh_log.read_text() == ""
