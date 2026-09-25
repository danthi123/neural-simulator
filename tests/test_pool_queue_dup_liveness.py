"""Subprocess-level tests for the pool.running half of tools/pool_queue.sh's duplicate guard.

THE ORIGINAL BUG (measured 2026-09-25 07:20, see research/FAILURE_LOG.md and the "DUPLICATE GUARD"
comment in tools/pool_queue.sh's `add` case): a pool.running record is `date<TAB>node<TAB>job`, so
`cut -f2-` on it yields "<node>\\t<job>" -- never a bare command -- and the job field itself carries a
`POOL_CHECKED_REASON=<%q token> ` prefix the dispatcher's pop_job() adds, which the old
"#checked:"-stripping regex never touched either. The running-set half of the guard could therefore
NEVER match, so identical commands were re-queued and re-dispatched while an earlier claim was still
alive on a node.

THE 2026-09-25 OPUS-REVIEW FIX ROUND (this file): the first liveness fix (`job_liveness_on_node`, two
ssh calls per node -- a bare reachability probe then a `grep -qxF` for one specific JOB_B64, verdict
cached by node) shipped with its own bugs, all closed here:
  1. HIGH fail-open -- the cache was keyed by NODE only, so a node carrying a DEAD older claim and a
     LIVE retry of the identical command got its liveness decided by whichever record was checked
     first; the other record silently reused that cached (possibly wrong) verdict.
  2. HIGH fail-open -- only the bare-reachability probe's failure ever mapped to UNREACH; a scan
     timeout (rc 124) or ssh transport error (rc 255) on the SECOND probe fell through to DEAD.
  3. MEDIUM -- a claim on a node no longer among the dispatcher's own targets (removed from
     .pool_extra_nodes, an idle-stopped AWS node) was refused FOREVER, with FORCE_DUP=1 (a blanket
     bypass of every check) as the only escape.
  4. LOW -- NEW_CMD / the running-record command / the queue-side command were `tr -s ' '`-squeezed
     but never trimmed, so a lone leading/trailing space defeated the comparison.
  7. LOW -- POOL_RUNNING_PATH (the env var pool_autodispatch.sh's own tests already use) was ignored;
     the running file was always derived from $Q.

The fix (`node_live_b64_set`, tools/pool_queue.sh) makes ONE ssh call per node that returns the node's
WHOLE set of live JOB_B64 values plus a fixed trailing sentinel line (POOL_LIVENESS_SCAN_OK); a
candidate record's own base64 is checked against that set locally (closes #1), and the sentinel must be
present in the captured output before ANY JOB_B64 line is trusted -- a dropped connection, a `timeout`
kill, a remote error, or a truncated scan all report UNREACH regardless of ssh's own exit status (closes
#2). A claim on a node outside `probe_nodes()` (or listed in POOL_DUP_ASSUME_DEAD_NODES) is treated as
retired/dead without contacting it, and does not affect any OTHER matching node's own ALIVE check
(closes #3). `trim()` closes #4. `POOL_RUNNING_PATH` is honoured (closes #7).

These tests drive the real script (no internal function is imported/copied) against a stubbed `ssh` on
PATH, mirroring tests/test_pool_ssh_config_plumbing.py and tests/test_pool_autodispatch_workflow.py's
stub-binary approach -- no real network calls, no real pool node, and the live queue/running files under
research/queue/ are never touched (POOL_QUEUE_PATH/POOL_RUNNING_PATH/POOL_EXTRA_NODES_FILE always point
into tmp_path).
"""
from __future__ import annotations

import base64
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
SENTINEL = "POOL_LIVENESS_SCAN_OK"


def _q_reason(reason: str) -> str:
    """The exact bash `printf '%q'` encoding pop_job() would produce for this reason -- shelling out
    to real bash rather than reimplementing %q in Python, so the test fixtures are byte-identical to
    what the live dispatcher writes."""
    out = subprocess.run(["bash", "-c", 'printf "%q" "$1"', "_", reason],
                          capture_output=True, text=True, timeout=10, check=True)
    return out.stdout


def _b64(text: str) -> str:
    """Byte-identical to what the script computes: printf '%s' "$rjob" | base64 -w0."""
    return base64.b64encode(text.encode()).decode("ascii")


def _make_ssh_stub(tmp_path: Path) -> tuple[Path, Path]:
    """A stub `ssh` matching node_live_b64_set's SINGLE-call protocol: it logs its full argv, finds
    the target node the same way the real remote-scan call addresses one, and then:
      - a node in UNREACHABLE_NODES: exits 255 with NO output (total connection failure).
      - a node in TIMEOUT_NODES: prints any $TIMEOUT_PARTIAL_<node> lines (simulating a scan that
        got partway through) but NEVER the trailing sentinel, then exits 124 (a `timeout` kill or a
        transport error mid-scan) -- the case node_live_b64_set must still map to UNREACH, since the
        real script trusts the sentinel's presence, not ssh's own exit status.
      - otherwise: prints "JOB_B64=<b64>" for every value in $ALIVE_B64_<node> (its live set), then
        the sentinel, then exits 0 -- a normal, complete scan.
    """
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
for t in ${TIMEOUT_NODES:-}; do
  if [ "$t" = "$node" ]; then
    eval "partial=\"\${TIMEOUT_PARTIAL_${node}:-}\""
    for b64 in $partial; do echo "JOB_B64=$b64"; done
    exit 124
  fi
done
eval "b64list=\"\${ALIVE_B64_${node}:-}\""
for b64 in $b64list; do
  echo "JOB_B64=$b64"
done
echo POOL_LIVENESS_SCAN_OK
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


def _seed_running(running_path: Path, node: str, job_field: str,
                   when: str = "2026-09-24 05:00:00", append: bool = False) -> Path:
    mode = "a" if append else "w"
    with open(running_path, mode) as fh:
        fh.write(f"{when}\t{node}\t{job_field}\n")
    return running_path


def _base_env(tmp_path: Path, ssh_log: Path, *, alive_b64: dict[str, list[str]] | None = None,
              unreachable: str = "", timeout: str = "", timeout_partial: dict[str, list[str]] | None = None,
              assume_dead: str = "", pool_nodes: str = "", running_path: Path | None = None,
              extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist-ssh-config"),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "does-not-exist-extra-nodes"),
        "SSH_LOG": str(ssh_log),
        "UNREACHABLE_NODES": unreachable,
        "TIMEOUT_NODES": timeout,
        "POOL_DUP_ASSUME_DEAD_NODES": assume_dead,
    }
    if pool_nodes:
        env["POOL_NODES"] = pool_nodes
    if running_path is not None:
        env["POOL_RUNNING_PATH"] = str(running_path)
    for node, b64s in (alive_b64 or {}).items():
        env[f"ALIVE_B64_{node}"] = " ".join(b64s)
    for node, b64s in (timeout_partial or {}).items():
        env[f"TIMEOUT_PARTIAL_{node}"] = " ".join(b64s)
    if extra:
        env.update(extra)
    return env


def _default_running_path(tmp_path: Path) -> Path:
    # ${Q%.queue}.running with Q = tmp_path/"pool.queue" -> tmp_path/"pool.running"
    return tmp_path / "pool.running"


def _job_field(reason: str = CHECKED, cmd: str = CMD) -> str:
    return f"POOL_CHECKED_REASON={_q_reason(reason)} {cmd}"


# ---------------------------------------------------------------------------
# (a) ALIVE -> refused.
def test_matching_running_record_alive_on_its_node_refuses(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field()
    _seed_running(_default_running_path(tmp_path), "pool41", job_field)
    b64 = _b64(job_field)
    env = _base_env(tmp_path, ssh_log, alive_b64={"pool41": [b64]})
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "RUNNING" in res.stderr
    assert "FORCE_DUP=1" in res.stderr
    assert (tmp_path / "pool.queue").read_text().strip() == ""


# (b) reachable, scan completes, but no matching JOB_B64 -> DEAD -> queued.
def test_matching_running_record_dead_on_its_node_is_queued(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field()
    _seed_running(_default_running_path(tmp_path), "pool41", job_field)
    env = _base_env(tmp_path, ssh_log, alive_b64={"pool41": []})   # node answers, no live JOB_B64s
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "no longer alive" in res.stderr
    assert "queued" in res.stdout
    assert CMD in (tmp_path / "pool.queue").read_text()


# (c) the claimed node cannot be reached at all -> UNREACH -> fails closed.
def test_matching_running_record_unreachable_node_fails_closed(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field()
    _seed_running(_default_running_path(tmp_path), "pool41", job_field)
    env = _base_env(tmp_path, ssh_log, unreachable="pool41")
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "could not be reached" in res.stderr or "UNREACH" in res.stderr.upper()
    assert (tmp_path / "pool.queue").read_text().strip() == ""


def test_matching_running_record_unreachable_node_force_dup_overrides(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field()
    _seed_running(_default_running_path(tmp_path), "pool41", job_field)
    env = _base_env(tmp_path, ssh_log, unreachable="pool41")
    env["FORCE_DUP"] = "1"
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "queueing despite" in res.stderr    # (issue #3) FORCE_DUP no longer prints "REFUSED"
    assert CMD in (tmp_path / "pool.queue").read_text()


# (d) issue #2 -- a scan that answers but is cut off BEFORE the sentinel (a `timeout` kill or a
# transport error mid-transfer) must fail CLOSED (UNREACH), exactly like total unreachability, never
# silently DEAD. This is the case the old two-probe design got wrong: only the bare-reachability
# probe's failure ever mapped to UNREACH.
def test_scan_truncated_before_sentinel_fails_closed_not_dead(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field()
    _seed_running(_default_running_path(tmp_path), "pool41", job_field)
    # the stub emits ONE JOB_B64 line (as if the scan started producing real matches) then dies with
    # rc 124 before ever printing the sentinel -- exercise this with a b64 that would NOT even be a
    # match, so a buggy implementation that ignored the missing sentinel and just fell through to
    # "DEAD" would queue (wrong); a correct one reports UNREACH and fails closed.
    env = _base_env(tmp_path, ssh_log, timeout="pool41",
                     timeout_partial={"pool41": ["not-a-real-match=="]})
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "could not be reached" in res.stderr or "UNREACH" in res.stderr.upper()
    assert (tmp_path / "pool.queue").read_text().strip() == ""


def test_scan_truncated_before_sentinel_force_dup_overrides(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field()
    _seed_running(_default_running_path(tmp_path), "pool41", job_field)
    env = _base_env(tmp_path, ssh_log, timeout="pool41")
    env["FORCE_DUP"] = "1"
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "queueing despite" in res.stderr
    assert CMD in (tmp_path / "pool.queue").read_text()


# (e) issue #1 -- cache conflation: pool41 carries TWO matching records, an older DEAD one and a
# newer LIVE retry (different POOL_CHECKED_REASON -> different JOB_B64). The dead record appears
# FIRST in file order -- exactly the shape that poisoned a per-node single-verdict cache built from
# whichever record was examined first. The node's live JOB_B64 SET must be checked per-record.
def test_dead_then_live_record_on_same_node_is_not_hidden_by_the_dead_one(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    running_path = _default_running_path(tmp_path)
    dead_field = _job_field(reason="first dispatch, now dead")
    live_field = _job_field(reason="retried dispatch, still running")
    _seed_running(running_path, "pool41", dead_field, when="2026-09-24 06:06:14")
    _seed_running(running_path, "pool41", live_field, when="2026-09-25 00:05:22", append=True)
    live_b64 = _b64(live_field)   # only the LIVE record's own b64 is present in the node's live set
    env = _base_env(tmp_path, ssh_log, alive_b64={"pool41": [live_b64]})
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "RUNNING" in res.stderr
    # one ssh call total, even though pool41 had two matching records (node_live_b64_set is called
    # once per node and cached, not once per record)
    calls = [ln for ln in ssh_log.read_text().splitlines() if ln.strip()]
    assert len(calls) == 1, calls


# (f) a non-matching pool.running record must never trigger an ssh call at all.
def test_non_matching_running_record_never_triggers_ssh(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    other_job = _job_field(reason="unrelated reason", cmd="echo some-completely-different-job")
    _seed_running(_default_running_path(tmp_path), "pool41", other_job)
    env = _base_env(tmp_path, ssh_log, alive_b64={"pool41": [_b64(other_job)]})
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert ssh_log.read_text() == "", f"ssh was called for a non-matching record: {ssh_log.read_text()!r}"
    assert CMD in (tmp_path / "pool.queue").read_text()


# (g) the %q prefix-stripping must handle a reason containing spaces, parens, commas and colons --
# the real D6 duplicate's reason text, reproduced verbatim.
D6_REASON = ("D6 N=2000 OOM root-cause found: pool_autodispatch.sh commit 3308e087c "
             "(2026-09-23 11:33, same day) + memory reservations added, re-verify before re-queue")


def test_prefix_stripping_handles_spaces_parens_commas_colons_in_the_reason(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field(reason=D6_REASON)
    escaped = _q_reason(D6_REASON)
    # sanity: this reason really does %q-escape to something containing the tricky characters
    # unescaped (comma/colon) and escaped (space/paren) -- if bash's own %q behaviour ever changes
    # this assertion documents what we're actually testing against.
    assert "(2026-09-23" not in escaped or "\\(2026-09-23" in escaped
    _seed_running(_default_running_path(tmp_path), "pool42", job_field)
    env = _base_env(tmp_path, ssh_log, alive_b64={"pool42": [_b64(job_field)]})
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool42" in res.stderr
    assert "RUNNING" in res.stderr


# Sanity: the queue-side half of the guard (untouched logic, aside from fix #4's trim) still refuses
# a plain in-queue duplicate, with no ssh call at all (no node is involved in a queue-vs-queue
# comparison).
def test_queue_side_duplicate_guard_untouched_and_ssh_free(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    q = tmp_path / "pool.queue"
    q.write_text(f"1700000000\t{CMD}  #checked:{CHECKED}\n")
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, _base_env(tmp_path, ssh_log))
    assert res.returncode == 2, res.stdout + res.stderr
    assert "already queued" in res.stderr
    assert ssh_log.read_text() == ""


# (h) issue #4 -- a lone trailing space on the QUEUE side must not defeat the comparison.
def test_queue_side_duplicate_detected_despite_incidental_trailing_space(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    q = tmp_path / "pool.queue"
    q.write_text(f"1700000000\t{CMD}  #checked:{CHECKED}\n")   # no trailing space on the stored cmd
    res = _run(["add", CMD + " ", "--checked", CHECKED], bin_dir, _base_env(tmp_path, ssh_log))
    assert res.returncode == 2, res.stdout + res.stderr
    assert "already queued" in res.stderr


# (i) issue #4 -- a lone trailing space on the RUNNING-record side must not defeat the comparison
# either (the match decides whether the record is even examined for liveness at all).
def test_running_side_duplicate_detected_despite_incidental_trailing_space(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field(cmd=CMD + " ")   # the stored command carries a trailing space
    _seed_running(_default_running_path(tmp_path), "pool41", job_field)
    env = _base_env(tmp_path, ssh_log, alive_b64={"pool41": [_b64(job_field)]})
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)   # add's own command has none
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "RUNNING" in res.stderr


# (j) issue #3 -- a claim on a node that is no longer a dispatch target at all (not in POOL_NODES,
# not in the extra-nodes file) is retired: treated as dead WITHOUT contacting it, and queued.
def test_claim_on_retired_node_is_queued_without_contacting_it(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field()
    _seed_running(_default_running_path(tmp_path), "poolstale", job_field)
    env = _base_env(tmp_path, ssh_log)   # default POOL_NODES = pool40 pool41 pool42; poolstale is not one
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "no longer a dispatch target" in res.stderr
    assert "poolstale" in res.stderr
    assert CMD in (tmp_path / "pool.queue").read_text()
    assert ssh_log.read_text() == "", f"a retired node must never be ssh'd: {ssh_log.read_text()!r}"


# (k) issue #3 -- the narrow POOL_DUP_ASSUME_DEAD_NODES override treats a still-registered node as
# dead without contacting it (an operator-known-dead node that IS still a probe target).
def test_assume_dead_override_is_queued_without_contacting_it(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    job_field = _job_field()
    _seed_running(_default_running_path(tmp_path), "pool41", job_field)
    env = _base_env(tmp_path, ssh_log, assume_dead="pool41")
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "POOL_DUP_ASSUME_DEAD_NODES" in res.stderr
    assert "pool41" in res.stderr
    assert CMD in (tmp_path / "pool.queue").read_text()
    assert ssh_log.read_text() == ""


# (l) issue #3 -- neither the retired-node path nor POOL_DUP_ASSUME_DEAD_NODES may skip the ALIVE
# check for a DIFFERENT matching node: a retired node and a genuinely alive node both carry a
# matching record -> must still refuse, citing the alive one, and must not ssh the retired node.
def test_retired_node_does_not_suppress_a_different_nodes_alive_check(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    running_path = _default_running_path(tmp_path)
    stale_field = _job_field(reason="stale claim on a retired node")
    live_field = _job_field(reason="live claim on a real node")
    _seed_running(running_path, "poolstale", stale_field)
    _seed_running(running_path, "pool41", live_field, append=True)
    env = _base_env(tmp_path, ssh_log, alive_b64={"pool41": [_b64(live_field)]})
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "RUNNING" in res.stderr
    assert "poolstale" not in ssh_log.read_text()


# (m) issue #7 -- POOL_RUNNING_PATH must be honoured, not silently ignored in favour of a path
# derived from $Q. Seed the record ONLY at a custom path that does not match ${Q%.queue}.running,
# and confirm it is still found via POOL_RUNNING_PATH.
def test_pool_running_path_env_var_is_honoured(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    custom_running = tmp_path / "elsewhere" / "custom.running"
    custom_running.parent.mkdir(parents=True)
    job_field = _job_field()
    _seed_running(custom_running, "pool41", job_field)
    # the Q-derived default path (tmp_path/"pool.running") is never created at all -- if the script
    # ignored POOL_RUNNING_PATH and fell back to the Q-derived path, [ -f "$RUNNING_FILE" ] would be
    # false and the running-side check would be skipped entirely (queued, wrongly).
    assert not _default_running_path(tmp_path).exists()
    env = _base_env(tmp_path, ssh_log, alive_b64={"pool41": [_b64(job_field)]}, running_path=custom_running)
    res = _run(["add", CMD, "--checked", CHECKED], bin_dir, env)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "pool41" in res.stderr
    assert "RUNNING" in res.stderr
