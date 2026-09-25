"""Subprocess-level tests for the repo-local pool ssh-config plumbing (tools/aws_pool_node.sh's
"AWS-AS-EXTRA-POOL-NODE" feature, 2026-09-23): every pool script that sshes/rsyncs to a node must build its
command with `-F <config>` when POOL_SSH_CONFIG names an existing file, and WITHOUT it (byte-identical to
before this feature) when the file is absent -- so pool40/41/42's existing behaviour never changes for anyone
who has not run `aws_pool_node.sh up`.

Uses a stubbed `ssh`/`rsync` on PATH (no real network calls), mirroring tests/test_aws_budget_guard_workflow.py's
stub-binary approach. Also covers the dispatcher's dynamic extra-node-list re-read (tools/pool_autodispatch.sh
re-reads .pool_extra_nodes every cycle, not once at startup).
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTODISPATCH = ROOT / "tools" / "pool_autodispatch.sh"
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"
POOL_SYNC = ROOT / "tools" / "pool_sync.sh"


def _make_ssh_stub(tmp_path: Path) -> tuple[Path, Path]:
    """A stub `ssh` that logs its argv and answers just enough to satisfy callers: the node_is_idle probe
    (nproc/load/proc-count/MemAvailable/max-job-gb, five numbers) and a bare reachability/`--help` check
    (exit 0)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    log.write_text("")
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
echo "8 0.10 0 20 0"
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def _make_rsync_stub(bin_dir: Path, tmp_path: Path) -> Path:
    log = tmp_path / "rsync.log"
    log.write_text("")
    stub = bin_dir / "rsync"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return log


def _run(script: Path, args: list[str], bin_dir: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}:{full_env.get('PATH', '')}"
    if env:
        full_env.update(env)
    return subprocess.run(["bash", str(script), *args], cwd=ROOT, env=full_env,
                           capture_output=True, text=True, timeout=30)


# --------------------------------------------------------------------------------- pool_autodispatch.sh

def test_node_idle_ssh_call_omits_dash_F_when_no_pool_ssh_config(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    missing_config = tmp_path / "does-not-exist"
    res = _run(AUTODISPATCH, ["--node-idle", "pool40"], bin_dir,
               {"POOL_QUEUE_PATH": str(tmp_path / "pool.queue"), "POOL_SSH_CONFIG": str(missing_config)})
    assert res.returncode in (0, 1), res.stderr   # idle/busy verdict, not a crash
    logged = ssh_log.read_text()
    assert "pool40" in logged
    assert " -F " not in f" {logged}"


def test_node_idle_ssh_call_includes_dash_F_when_pool_ssh_config_exists(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    config = tmp_path / "ssh_config"
    config.write_text("Include ~/.ssh/config\nHost pool1\n  HostName 10.0.0.5\n")
    res = _run(AUTODISPATCH, ["--node-idle", "pool1"], bin_dir,
               {"POOL_QUEUE_PATH": str(tmp_path / "pool.queue"), "POOL_SSH_CONFIG": str(config)})
    assert res.returncode in (0, 1), res.stderr
    logged = ssh_log.read_text()
    assert f"-F {config}" in logged
    assert "pool1" in logged


def test_extra_nodes_file_is_reread_fresh_each_call_not_cached_at_startup(tmp_path):
    # THE POINT (2026-09-23 build): aws_pool_node.sh up/down can add/remove a node with NO dispatcher restart
    # -- so the node list a running dispatch cycle uses must reflect the file's CURRENT contents, not whatever
    # it held when the (long-lived, systemd-managed) process started.
    extra = tmp_path / "extra_nodes"
    res_before = subprocess.run(
        ["bash", str(AUTODISPATCH), "--nodes-this-cycle"], cwd=ROOT,
        env={**os.environ, "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
             "POOL_EXTRA_NODES_FILE": str(extra)},
        capture_output=True, text=True, timeout=30,
    )
    assert res_before.stdout.strip() == "pool40 pool41 pool42"

    extra.write_text("# a comment, ignored\npool1\n\npool2\n")
    res_after = subprocess.run(
        ["bash", str(AUTODISPATCH), "--nodes-this-cycle"], cwd=ROOT,
        env={**os.environ, "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
             "POOL_EXTRA_NODES_FILE": str(extra)},
        capture_output=True, text=True, timeout=30,
    )
    assert res_after.stdout.strip() == "pool40 pool41 pool42 pool1 pool2"


def test_pool_nodes_env_default_unaffected_by_extra_nodes_when_not_present(tmp_path):
    # No .pool_extra_nodes at all -> the node list is EXACTLY what it was before this feature existed.
    res = subprocess.run(
        ["bash", str(AUTODISPATCH), "--nodes-this-cycle"], cwd=ROOT,
        env={**os.environ, "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
             "POOL_EXTRA_NODES_FILE": str(tmp_path / "does-not-exist")},
        capture_output=True, text=True, timeout=30,
    )
    assert res.stdout.strip() == "pool40 pool41 pool42"


# ------------------------------------------------------------------------------------------ pool_queue.sh

def test_probe_node_ssh_calls_omit_dash_F_when_no_pool_ssh_config(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    res = _run(POOL_QUEUE, ["--probe-node", "pool40", "research.runners.fake_mod"], bin_dir,
               {"POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
                "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist")})
    logged = ssh_log.read_text()
    assert "pool40" in logged
    assert " -F " not in f" {logged}"
    assert res.returncode in (0, 1)


def test_probe_node_ssh_calls_include_dash_F_when_pool_ssh_config_exists(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    config = tmp_path / "ssh_config"
    config.write_text("Include ~/.ssh/config\n")
    res = _run(POOL_QUEUE, ["--probe-node", "pool1", "research.runners.fake_mod"], bin_dir,
               {"POOL_QUEUE_PATH": str(tmp_path / "pool.queue"), "POOL_SSH_CONFIG": str(config)})
    logged = ssh_log.read_text()
    assert f"-F {config}" in logged
    assert "pool1" in logged
    assert res.returncode in (0, 1)


# ------------------------------------------------------------------------------------------- pool_sync.sh

def test_pool_sync_rsync_dash_e_omits_dash_F_when_no_pool_ssh_config(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    rsync_log = _make_rsync_stub(bin_dir, tmp_path)
    res = _run(POOL_SYNC, [], bin_dir,
               {"POOL_NODES": "pool40", "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist")})
    assert res.returncode == 0, res.stderr
    logged = rsync_log.read_text()
    assert "-e ssh -o BatchMode" in logged
    assert "-F" not in logged


def test_pool_sync_rsync_dash_e_includes_dash_F_when_pool_ssh_config_exists(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    rsync_log = _make_rsync_stub(bin_dir, tmp_path)
    config = tmp_path / "ssh_config"
    config.write_text("Include ~/.ssh/config\n")
    res = _run(POOL_SYNC, [], bin_dir,
               {"POOL_NODES": "pool1", "POOL_SSH_CONFIG": str(config)})
    assert res.returncode == 0, res.stderr
    logged = rsync_log.read_text()
    assert f"-e ssh -F {config} -o BatchMode" in logged


def test_pool_sync_default_node_list_grows_with_extra_nodes_file(tmp_path):
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    rsync_log = _make_rsync_stub(bin_dir, tmp_path)
    extra = tmp_path / "extra_nodes"
    extra.write_text("pool1\n")
    res = _run(POOL_SYNC, [], bin_dir,
               {"POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"), "POOL_EXTRA_NODES_FILE": str(extra)})
    assert res.returncode == 0, res.stderr
    assert "pool1" in res.stdout
    # An explicit POOL_NODES scopes the run and must NOT be widened by the extra-nodes file.
    res_scoped = _run(POOL_SYNC, [], bin_dir,
                       {"POOL_NODES": "pool40", "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
                        "POOL_EXTRA_NODES_FILE": str(extra)})
    assert res_scoped.returncode == 0, res_scoped.stderr
    assert "pool1" not in res_scoped.stdout


def test_pool_sync_isolated_revisions_ssh_call_does_not_double_the_ssh_binary(tmp_path):
    # REGRESSION (2026-09-23 fix round): `revs=$(... ssh $RSYNC_SSH "$N" ...)` ran as `ssh ssh -o ... <node> ...`
    # because $RSYNC_SSH is ALREADY the whole `-e`-style ssh invocation ("ssh -o BatchMode=yes ..."). Real ssh
    # fails that with "Could not resolve hostname ssh" (rc=255); the old test here only asserted on the RSYNC
    # log, so a stub `ssh` that answers ANY argv (as ours does) never caught it. Assert on the ssh stub's own
    # ARGV: the logged command line must never start with a literal "ssh" token (that would mean the ssh BINARY
    # was invoked with another "ssh" as its first argument -- i.e. doubled).
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    rsync_log = _make_rsync_stub(bin_dir, tmp_path)
    res = _run(POOL_SYNC, [], bin_dir,
               {"POOL_NODES": "pool40", "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist")})
    assert res.returncode == 0, res.stderr
    logged_lines = [ln for ln in ssh_log.read_text().splitlines() if ln.strip()]
    assert logged_lines, "expected at least one ssh call (the isolated-revisions 'ls -d' probe)"
    for ln in logged_lines:
        first_token = ln.split()[0]
        assert first_token != "ssh", f"ssh invoked with a literal 'ssh' as its own first argument: {ln!r}"


def _make_failing_ssh_rsync_stub(tmp_path: Path) -> Path:
    """ssh AND rsync both exit 255 unconditionally -- an unreachable node, or an instance stopped mid-cycle
    (aws_idle_stop.sh), from pool_sync's point of view. No log needed; only the exit status matters here."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("ssh", "rsync"):
        stub = bin_dir / name
        stub.write_text("#!/usr/bin/env bash\nexit 255\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def test_pool_sync_default_mode_still_exits_0_when_a_node_is_unreachable(tmp_path):
    # BASELINE (must NOT change): the systemd-timer default behaviour stays "best-effort, always exit 0" -- only
    # --strict/POOL_SYNC_STRICT=1 opts in to failing loudly.
    bin_dir = _make_failing_ssh_rsync_stub(tmp_path)
    res = _run(POOL_SYNC, [], bin_dir, {"POOL_NODES": "testnode", "POOL_SSH_CONFIG": str(tmp_path / "no-cfg")})
    assert res.returncode == 0, res.stderr
    assert "UNREACHABLE" in res.stdout


def test_pool_sync_strict_flag_exits_nonzero_when_the_node_is_unreachable(tmp_path):
    # HIGH (2026-09-23 fix round #2): `aws_pool_node.sh down` needs a way to tell "the pull actually worked" from
    # "nothing came back because the node was gone" -- the plain default above cannot distinguish them (always 0).
    bin_dir = _make_failing_ssh_rsync_stub(tmp_path)
    res = _run(POOL_SYNC, ["--strict", "--node", "testnode"], bin_dir,
               {"POOL_SSH_CONFIG": str(tmp_path / "no-cfg")})
    assert res.returncode == 1
    assert "STRICT" in (res.stdout + res.stderr)


def test_pool_sync_strict_env_var_is_equivalent_to_the_flag(tmp_path):
    bin_dir = _make_failing_ssh_rsync_stub(tmp_path)
    res = _run(POOL_SYNC, [], bin_dir,
               {"POOL_NODES": "testnode", "POOL_SSH_CONFIG": str(tmp_path / "no-cfg"), "POOL_SYNC_STRICT": "1"})
    assert res.returncode == 1


def test_pool_sync_strict_still_exits_0_when_the_node_is_actually_reachable(tmp_path):
    # Strict must not be "always fail" -- a genuinely successful pull still exits 0.
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    _make_rsync_stub(bin_dir, tmp_path)
    res = _run(POOL_SYNC, ["--strict"], bin_dir,
               {"POOL_NODES": "pool40", "POOL_SSH_CONFIG": str(tmp_path / "no-cfg")})
    assert res.returncode == 0, res.stderr


def test_pool_sync_strict_fails_when_a_per_revision_rsync_fails_but_the_main_pull_succeeds(tmp_path):
    # The main pull can succeed while an isolated-revision pull fails independently (e.g. that sub-path got
    # wedged/permission-denied on the node) -- strict mode must catch that too, not just total unreachability.
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    bin_dir_ssh = bin_dir / "ssh"
    # Override the generic stub: answer the node_is_idle-style probe normally, list ONE revision dir, but this
    # is the rsync stub (not ssh) that must fail for that revision's pull specifically.
    bin_dir_ssh.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
case "$*" in
  *"ls -d derisk-pool/revisions"*) echo "derisk-pool/revisions/abc1234/research/findings/raw"; exit 0 ;;
esac
echo "8 0.10 0 20 0"
exit 0
""")
    bin_dir_ssh.chmod(bin_dir_ssh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text("""#!/usr/bin/env bash
case "$*" in
  *"revisions/abc1234"*) exit 255 ;;
esac
exit 0
""")
    rsync_stub.chmod(rsync_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    res = _run(POOL_SYNC, ["--strict"], bin_dir, {"POOL_NODES": "pool40", "POOL_SSH_CONFIG": str(tmp_path / "no-cfg")})
    assert res.returncode == 1
    assert "rsync FAILED" in res.stdout


def test_pool_sync_survives_an_empty_or_comment_only_extra_nodes_file(tmp_path):
    # REGRESSION (2026-09-23 fix round): under `set -euo pipefail`, `_EXTRA=$(grep -vE ... | tr ...)` exits the
    # WHOLE SCRIPT with rc=1 and zero ssh/rsync calls whenever .pool_extra_nodes exists but is empty or holds
    # only comments (grep -v selects nothing, pipefail propagates its rc=1 through the assignment). This is
    # EXACTLY what `aws_pool_node.sh down` leaves behind, so every pool_sync after any up/down cycle died
    # silently. An empty file and a comment-only file must both leave pool_sync exit 0 and still sync the
    # default nodes.
    # NOTE: POOL_NODES must be UNSET (not e.g. "pool40") for this to exercise the buggy line at all -- it lives
    # behind `[ -z "${POOL_NODES:-}" ] &&`, which is exactly how a plain, unscoped `pool_sync.sh` call (the
    # regular pool40/41/42 sync path, e.g. from a cron/heartbeat) invokes it.
    bin_dir, ssh_log = _make_ssh_stub(tmp_path)
    rsync_log = _make_rsync_stub(bin_dir, tmp_path)
    for content in ("", "# just a comment\n", "\n\n"):
        extra = tmp_path / "extra_nodes"
        extra.write_text(content)
        res = _run(POOL_SYNC, [], bin_dir,
                   {"POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
                    "POOL_EXTRA_NODES_FILE": str(extra)})
        assert res.returncode == 0, f"content={content!r} stderr={res.stderr}"
        assert "pool40" in res.stdout


# ------------------------------------------------------------------ pool_sync.sh: stale-hostname auto-refresh

def _make_quiet_ssh_stub(tmp_path: Path):
    """A reachable `ssh` that answers EVERY call (including pool_sync's own isolated-revisions `ls -d` probe)
    with empty stdout and rc=0 -- i.e. "reachable, zero isolated revisions provisioned here", so the tests
    below can count rsync invocations exactly without `_make_ssh_stub`'s fixed node_is_idle-shaped stdout
    ("8 0.10 0 20 0") being mis-parsed as five bogus revision-directory names and multiplying rsync calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    log.write_text("")
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def _make_rsync_fail_then_succeed_stub(bin_dir: Path, tmp_path: Path, fail_times: int = 1):
    """A stub `rsync` that fails (rc=255, like an unreachable/stale-ip node) the first `fail_times` calls and
    succeeds every call after that -- so a test can prove pool_sync retried exactly once post-refresh without
    needing to fake a real ip change end-to-end (aws_pool_node.sh's OWN ip-rewrite correctness is covered
    directly in tests/test_aws_pool_node_workflow.py's `refresh`/`start` tests)."""
    counter = tmp_path / "rsync_calls"
    counter.write_text("0")
    log = tmp_path / "rsync.log"
    log.write_text("")
    stub = bin_dir / "rsync"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
n=$(cat "{counter}" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "{counter}"
if [ "$n" -le {fail_times} ]; then exit 255; fi
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return log, counter


def _make_refresh_aws_stub(bin_dir: Path, tmp_path: Path, ip: str = "9.9.9.9") -> Path:
    log = tmp_path / "aws.log"
    log.write_text("")
    stub = bin_dir / "aws"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
case "$*" in
  *"State.Name"*) echo running; exit 0 ;;
  *"PublicIpAddress"*) echo "{ip}"; exit 0 ;;
esac
echo ok
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return log


def test_pool_sync_refreshes_a_stale_aws_node_and_retries_once(tmp_path):
    # 2026-09-25, incident-driven: an AWS pool node's public ip changes every stop/start (no Elastic IP in this
    # feature), so a stale .pool_ssh_config Host block reads as plain UNREACHABLE to pool_sync -- the owner had
    # to fix pool1's entry by hand. A node with a research/queue/.aws_<name> state file gets ONE automatic
    # `aws_pool_node.sh refresh` + retry before being reported unreachable.
    bin_dir, ssh_log = _make_quiet_ssh_stub(tmp_path)
    rsync_log, counter = _make_rsync_fail_then_succeed_stub(bin_dir, tmp_path, fail_times=1)
    aws_log = _make_refresh_aws_stub(bin_dir, tmp_path, ip="9.9.9.9")

    aws_state_dir = tmp_path / "state"; aws_state_dir.mkdir()
    key = tmp_path / "key.pem"; key.write_text("fake key\n")
    (aws_state_dir / ".aws_pool1").write_text(f"instance=i-aaa\nregion=us-east-1\nkey={key}\nsg=sg-x\n")
    ssh_config = tmp_path / "pool_ssh_config"   # need not pre-exist; refresh creates/rewrites it

    res = _run(POOL_SYNC, [], bin_dir, {
        "POOL_NODES": "pool1",
        "POOL_SYNC_AWS_STATE_DIR": str(aws_state_dir),
        "POOL_SSH_CONFIG": str(ssh_config),
    })
    assert res.returncode == 0, res.stderr
    assert counter.read_text().strip() == "2", "expected exactly one retry after the refresh"
    assert "even after a Host-block refresh" not in res.stdout, "the retry should have succeeded"
    assert "State.Name" in aws_log.read_text() and "PublicIpAddress" in aws_log.read_text()
    assert "9.9.9.9" in ssh_config.read_text()   # the refresh really rewrote the Host block


def test_pool_sync_reports_unreachable_when_the_refresh_itself_cannot_help(tmp_path):
    # The node has an .aws_<name> state file, but the instance is NOT running (e.g. genuinely stopped) --
    # `refresh` correctly declines to rewrite anything, and pool_sync must still report UNREACHABLE (never
    # silently succeed, never loop).
    bin_dir, ssh_log = _make_quiet_ssh_stub(tmp_path)
    rsync_log, counter = _make_rsync_fail_then_succeed_stub(bin_dir, tmp_path, fail_times=99)
    aws_log = tmp_path / "aws.log"; aws_log.write_text("")
    aws_stub = bin_dir / "aws"
    aws_stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{aws_log}"
case "$*" in
  *"State.Name"*) echo stopped; exit 0 ;;
esac
echo ok
exit 0
""")
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    aws_state_dir = tmp_path / "state"; aws_state_dir.mkdir()
    key = tmp_path / "key.pem"; key.write_text("fake key\n")
    (aws_state_dir / ".aws_pool1").write_text(f"instance=i-aaa\nregion=us-east-1\nkey={key}\nsg=sg-x\n")

    res = _run(POOL_SYNC, [], bin_dir, {
        "POOL_NODES": "pool1",
        "POOL_SYNC_AWS_STATE_DIR": str(aws_state_dir),
        "POOL_SSH_CONFIG": str(tmp_path / "pool_ssh_config"),
    })
    assert res.returncode == 0, res.stderr   # plain mode never fails loudly (unchanged baseline)
    assert "UNREACHABLE" in res.stdout
    assert counter.read_text().strip() == "1", "refresh declined -> no retry attempt"


def test_pool_sync_never_attempts_refresh_for_a_minipc_node_without_an_aws_state_file(tmp_path):
    # pool40/41/42 have no research/queue/.aws_<name> state file -- this must stay BYTE-IDENTICAL to before
    # this feature: plain "UNREACHABLE (skipped)", no aws_pool_node.sh call at all (no `aws` binary needed on
    # PATH for this test to pass -- its absence would surface as a command-not-found if anything tried).
    bin_dir, ssh_log = _make_quiet_ssh_stub(tmp_path)
    rsync_log, counter = _make_rsync_fail_then_succeed_stub(bin_dir, tmp_path, fail_times=99)

    res = _run(POOL_SYNC, [], bin_dir, {
        "POOL_NODES": "pool40",
        "POOL_SYNC_AWS_STATE_DIR": str(tmp_path / "state"),   # dir does not even exist
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
    })
    assert res.returncode == 0, res.stderr
    assert "pool40: UNREACHABLE (skipped)" in res.stdout
    assert "Host-block refresh" not in res.stdout
    assert counter.read_text().strip() == "1"
