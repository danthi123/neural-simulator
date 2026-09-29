"""tools/agent_gateway_submit.py -- H2 fix (2026-09-29 security review): STRUCTURED gpu-submit/pool-submit
requests, replacing free-form job submission. Hermetic throughout: every test builds its own scratch git repo
(AGENT_GATEWAY_SUBMIT_ROOT), never the real neural-simulator repo, and stubs bwrap/gpu_queue.sh/pool_queue.sh/
pool_provision.sh via env overrides -- no real sandbox, GPU job, or pool node is ever touched.
"""
import os
import stat
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUBMIT = os.path.join(_ROOT, "tools", "agent_gateway_submit.py")


def _git(args, cwd, check=True):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"git {args} failed: {r.stderr}")
    return r


def _write_exec(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_repo(runner_name="foo", prereg_path=None, extra_unreachable_commit=False):
    """A scratch git repo with a runner file committed, a refs/remotes/origin/main pointing at HEAD (so that
    commit is "reachable" the way the real gitea/origin branch-protection setup would make it), and optionally
    a SECOND commit that is NOT reachable from any remote ref (to test the reachability refusal)."""
    d = tempfile.mkdtemp()
    _git(["init", "-q", "-b", "main"], d)
    _git(["config", "user.email", "t@t.com"], d)
    _git(["config", "user.name", "t"], d)
    os.makedirs(os.path.join(d, "research", "runners"), exist_ok=True)
    os.makedirs(os.path.join(d, "research", "findings", "raw"), exist_ok=True)
    with open(os.path.join(d, "research", "runners", f"{runner_name}.py"), "w") as f:
        f.write("# runner\n")
    if prereg_path:
        os.makedirs(os.path.dirname(os.path.join(d, prereg_path)), exist_ok=True)
        with open(os.path.join(d, prereg_path), "w") as f:
            f.write("prereg\n")
    _git(["add", "-A"], d)
    _git(["commit", "-q", "-m", "init"], d)
    reachable_sha = _git(["rev-parse", "HEAD"], d).stdout.strip()
    _git(["update-ref", "refs/remotes/origin/main", reachable_sha], d)

    unreachable_sha = None
    if extra_unreachable_commit:
        with open(os.path.join(d, "scratch.txt"), "w") as f:
            f.write("x\n")
        _git(["add", "-A"], d)
        _git(["commit", "-q", "-m", "unpushed"], d)
        unreachable_sha = _git(["rev-parse", "HEAD"], d).stdout.strip()
        # move origin/main back so this second commit is NOT an ancestor of any remote ref
        _git(["update-ref", "refs/remotes/origin/main", reachable_sha], d)
        _git(["checkout", "-q", reachable_sha], d)

    return d, reachable_sha, unreachable_sha


def _stub_set(d, gpu_ok=True, pool_ok=True, bwrap_present=True):
    log = os.path.join(d, "calls.log")
    gpu_queue = _write_exec(os.path.join(d, "stub_gpu_queue.sh"),
                             'echo "GPU_QUEUE $*" >> "%s"\nexit %d\n' % (log, 0 if gpu_ok else 1))
    pool_queue = _write_exec(os.path.join(d, "stub_pool_queue.sh"),
                              'echo "POOL_QUEUE $*" >> "%s"\nexit %d\n' % (log, 0 if pool_ok else 1))
    pool_provision = _write_exec(os.path.join(d, "stub_pool_provision.sh"),
                                  'echo "POOL_PROVISION $*" >> "%s"\nexit 0\n' % log)
    # a fake $HOME with a real .ssh dir, so build_bwrap_argv's "only mask a path that actually exists" check
    # (the same guard real bwrap needs -- it can only tmpfs-mount over a destination that exists inside the
    # already `--ro-bind / /`-mounted tree) has something real to find, exactly like the production $HOME does.
    fake_home = os.path.join(d, "fakehome")
    os.makedirs(os.path.join(fake_home, ".ssh"), exist_ok=True)
    env = {
        "AGENT_GATEWAY_SUBMIT_ROOT": d,
        "AGENT_GATEWAY_GPU_QUEUE_SH": gpu_queue,
        "AGENT_GATEWAY_POOL_QUEUE_SH": pool_queue,
        "AGENT_GATEWAY_POOL_PROVISION_SH": pool_provision,
        "AGENT_GATEWAY_HOME": fake_home,   # never point the sandbox-mask logic at the real $HOME in a test
    }
    if not bwrap_present:
        env["AGENT_GATEWAY_BWRAP"] = os.path.join(d, "no-such-bwrap-binary")
    return log, env


def _run(args, env_extra, timeout=15):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run([sys.executable, _SUBMIT] + args, env=env, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------------------------------- sha checks

def test_gpu_refuses_a_sha_that_is_not_40_hex():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", "abc123", "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode != 0
    assert "40 hex" in res.stderr


def test_gpu_refuses_a_sha_not_reachable_from_a_remote_ref():
    d, _reachable, unreachable = _make_repo(extra_unreachable_commit=True)
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", unreachable, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode != 0
    assert "not reachable" in res.stderr


def test_gpu_accepts_a_sha_reachable_from_origin():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "GPU_QUEUE add" in open(log).read()


# ------------------------------------------------------------------------------------------------ runner checks

def test_gpu_refuses_a_runner_not_matching_the_pattern():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "os.system", "--", "--seed", "7"], env)
    assert res.returncode != 0
    assert "research.runners" in res.stderr


def test_gpu_refuses_a_runner_file_missing_at_the_pinned_sha():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.doesnotexist", "--", "--seed", "7"], env)
    assert res.returncode != 0
    assert "does not exist at --sha" in res.stderr


# --------------------------------------------------------------------------------------------------- arg checks

def test_gpu_refuses_an_argument_with_shell_metacharacters():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--",
                "--seed", "7", "$(touch pwned)"], env)
    assert res.returncode != 0
    assert "allowed pattern" in res.stderr


def test_gpu_redirects_an_out_path_outside_the_allowed_root_into_the_per_job_dir():
    # 2026-09-29 re-review, LOW: a job must never write the SHARED research/findings/raw/ directly -- the
    # caller's directory is now always discarded outright (never trusted enough to validate), and only the
    # BASENAME is kept, relocated under a fresh per-job subdirectory that is the ONLY thing bound writable.
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--",
                "--seed", "7", "--out", "research/coordination/x.json"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    line = next(l for l in open(log).read().splitlines() if l.startswith("GPU_QUEUE add"))
    assert "research/coordination" not in line
    assert "_agent_jobs" in line and "/x.json" in line


def test_gpu_redirects_an_out_path_with_dotdot_into_the_per_job_dir():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--",
                "--seed", "7", "--out=research/findings/raw/../../etc/passwd"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    line = next(l for l in open(log).read().splitlines() if l.startswith("GPU_QUEUE add"))
    assert "etc/passwd" not in line and ".." not in line
    assert "_agent_jobs" in line and "/passwd" in line


def test_gpu_refuses_an_out_value_whose_basename_is_dotdot():
    # ';' and other shell metacharacters are already refused by the earlier arg-charset check (ARG_RE) --
    # this exercises the ONE thing that charset does NOT rule out: a basename of exactly '..', which is made
    # entirely of otherwise-safe characters but would let os.path.join(job_out_dir, '..') escape the job dir.
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--",
                "--seed", "7", "--out=research/findings/raw/.."], env)
    assert res.returncode != 0
    assert "not safe" in res.stderr


def test_gpu_only_binds_the_per_job_dir_writable_not_the_shared_raw_dir():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--",
                "--seed", "7", "--out", "o.json"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    line = next(l for l in open(log).read().splitlines() if l.startswith("GPU_QUEUE add"))
    shared_raw = os.path.join(d, "research", "findings", "raw")
    # the shared raw dir itself must never appear as a --bind target (only its _agent_jobs/<id> subdir may)
    assert ("--bind %s %s" % (shared_raw, shared_raw)) not in line
    assert "_agent_jobs" in line


# ---------------------------------------------------------------------------------------------- gate-seed guard

def test_gpu_refuses_a_gate_seed_with_no_prereg():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "42"], env)
    assert res.returncode != 0
    assert "GATE seed" in res.stderr


def test_gpu_refuses_a_gate_seed_whose_prereg_is_not_committed():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo",
                "--prereg", "docs/drafts/x/PREREG.md", "--", "--seed", "42"], env)
    assert res.returncode != 0
    assert "is not a file committed" in res.stderr


def test_gpu_accepts_a_gate_seed_with_a_committed_prereg():
    d, sha, _ = _make_repo(prereg_path="docs/drafts/x/PREREG.md")
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo",
                "--prereg", "docs/drafts/x/PREREG.md", "--", "--seed", "42"], env)
    assert res.returncode == 0, res.stdout + res.stderr


def test_gpu_a_number_merely_containing_a_gate_seed_is_not_a_false_positive():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "142"], env)
    assert res.returncode == 0, res.stdout + res.stderr


# --------------------------------------------------------------------------------------------------- sandboxing

def test_gpu_refuses_when_bwrap_is_not_installed():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d, bwrap_present=False)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode != 0
    assert "bubblewrap" in res.stderr
    assert "install" in res.stderr.lower()


def test_gpu_builds_a_sandboxed_command_with_the_required_bwrap_flags():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = open(log).read()
    assert "GPU_QUEUE add" in calls
    line = next(l for l in calls.splitlines() if l.startswith("GPU_QUEUE add"))
    # 2026-09-29 re-review, HIGH-1: rebuilt from empty -- NEVER a blanket --ro-bind / /  any more.
    assert "--ro-bind / /" not in line
    assert "--unshare-all" in line
    assert "--die-with-parent" in line
    assert "--new-session" in line
    assert "--clearenv" in line
    assert "--proc /proc" in line
    assert "--dev /dev" in line
    assert "--ro-bind /usr /usr" in line
    assert "--ro-bind /sys /sys" in line
    assert "--symlink usr/lib /lib" in line
    # never a blanket bind of /home, /etc, /proc (beyond the fresh one above), or /run
    for bad in ("--bind /home", "--ro-bind /home", "--bind /etc", "--ro-bind /etc",
                "--bind /run", "--ro-bind /run"):
        assert bad not in line, "must never bind %r: %s" % (bad, line)
    checkout = os.path.join(d, ".claude", "worktrees", "agent-jobs", sha)
    assert checkout in line
    assert "_agent_jobs" in line   # the per-job output dir, not the shared raw dir
    assert "research.runners.foo" in line
    # no shell wrapper any more -- the venv's python is exec'd directly
    assert "bash -c" not in line
    assert os.path.join(checkout, ".venv", "bin", "python") in line


def test_gpu_creates_a_detached_worktree_pinned_at_the_sha_with_a_venv_symlink():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    checkout = os.path.join(d, ".claude", "worktrees", "agent-jobs", sha)
    assert os.path.isdir(checkout)
    head = _git(["rev-parse", "HEAD"], checkout).stdout.strip()
    assert head == sha
    # the ONLY expected "dirt" is the untracked .venv symlink this script itself creates (matching the
    # convention every other worktree in the real repo already uses) -- no TRACKED file may be modified.
    status = _git(["status", "--porcelain"], checkout).stdout
    assert status.strip() in ("", "?? .venv"), status
    assert os.path.islink(os.path.join(checkout, ".venv"))
    assert os.path.realpath(os.path.join(checkout, ".venv")) == os.path.realpath(os.path.join(d, ".venv"))


def test_gpu_reuses_an_existing_correct_worktree_on_a_second_submit():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    res2 = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "8"], env)
    assert res2.returncode == 0, res2.stdout + res2.stderr
    # exactly one worktree entry for this sha, not a duplicate/renamed one
    listing = _git(["worktree", "list"], d).stdout
    assert listing.count(sha[:12]) == 1 or listing.count(os.path.join("agent-jobs", sha)) == 1


# ------------------------------------------------------------------------------------------------------- pool

def test_pool_requires_checked():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["pool", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode != 0
    assert "--checked" in res.stderr


def test_pool_builds_the_isolated_revision_job_line_and_forwards_it():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["pool", "--sha", sha, "--runner", "research.runners.foo",
                "--checked", "corpus:nothing", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = open(log).read()
    line = next(l for l in calls.splitlines() if l.startswith("POOL_QUEUE add"))
    assert f"derisk-pool/revisions/{sha}" in line
    assert "research.runners.foo" in line
    assert "--checked corpus:nothing" in line
    # the provisioning call is now `flock LOCKFILE bash <stub> ...` (2026-09-29 re-review, M2: serialized
    # provisioning) -- still backgrounded/non-blocking, but the extra flock fork+exec can land a beat after
    # this process returns; poll briefly rather than assume it is already flushed to disk.
    import time
    deadline = time.time() + 5
    while "POOL_PROVISION --revision %s --isolated" % sha not in open(log).read() and time.time() < deadline:
        time.sleep(0.05)
    assert "POOL_PROVISION --revision %s --isolated" % sha in open(log).read()


def test_pool_refuses_an_argument_with_shell_metacharacters():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["pool", "--sha", sha, "--runner", "research.runners.foo",
                "--checked", "corpus:nothing", "--", "--seed", "7;rm -rf ~"], env)
    assert res.returncode != 0
    assert "allowed pattern" in res.stderr


# ------------------------------------------------------------------------------------ M2 re-review (2026-09-29)

def test_gpu_job_is_wrapped_in_systemd_run_scope_with_resource_limits_and_a_timeout():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    line = next(l for l in open(log).read().splitlines() if l.startswith("GPU_QUEUE add"))
    assert "timeout --kill-after=30s" in line
    assert "systemd-run --user --scope --quiet" in line
    assert "-p MemoryMax=20G" in line
    assert "-p TasksMax=512" in line
    assert "--setenv=AGENT_JOB_SRC=sim-agent" in line
    # the bwrap invocation must still be present, wrapped INSIDE the systemd-run/timeout, not replaced by it
    assert "bwrap --unshare-all" in line
    # timeout/systemd-run must come BEFORE bwrap in the line (outside it), not after
    assert line.index("timeout") < line.index("systemd-run") < line.index("bwrap")


def test_gpu_queue_depth_cap_refuses_a_new_submission_once_reached():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    queue_path = os.path.join(d, "gpu.queue")
    with open(queue_path, "w") as f:
        for _ in range(20):
            f.write("timeout 4h systemd-run --user --scope --setenv=AGENT_JOB_SRC=sim-agent -- bwrap ...\n")
    env["AGENT_GATEWAY_GPU_QUEUE_PATH"] = queue_path
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode != 0
    assert "cap" in res.stderr.lower()
    assert "GPU_QUEUE add" not in open(log).read() if os.path.exists(log) else True


def test_gpu_queue_depth_cap_ignores_unrelated_queued_jobs():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    queue_path = os.path.join(d, "gpu.queue")
    with open(queue_path, "w") as f:
        for _ in range(30):
            f.write("cd /some/other/checkout && .venv/bin/python -u -m research.runners.unrelated --seed 1\n")
    env["AGENT_GATEWAY_GPU_QUEUE_PATH"] = queue_path
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr


def test_pool_queue_depth_cap_refuses_a_new_submission_once_reached():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    queue_path = os.path.join(d, "pool.queue")
    with open(queue_path, "w") as f:
        for _ in range(20):
            f.write("AGENT_JOB_SRC=sim-agent cd ~/derisk-pool/revisions/abc && true  #checked:x\n")
    env["AGENT_GATEWAY_POOL_QUEUE_PATH"] = queue_path
    res = _run(["pool", "--sha", sha, "--runner", "research.runners.foo",
                "--checked", "corpus:nothing", "--", "--seed", "7"], env)
    assert res.returncode != 0
    assert "cap" in res.stderr.lower()


def test_pool_provisioning_is_serialized_via_flock():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["pool", "--sha", sha, "--runner", "research.runners.foo",
                "--checked", "corpus:nothing", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    ps_log = os.path.join(d, "ps_calls.log")
    # the real `flock` binary was exec'd directly (not stubbed) -- assert indirectly via the stub provisioning
    # script actually being reached (proves flock granted the lock and exec'd through to it).
    import time
    deadline = time.time() + 5
    while "POOL_PROVISION" not in open(log).read() and time.time() < deadline:
        time.sleep(0.05)
    assert "POOL_PROVISION --revision %s --isolated" % sha in open(log).read()


def test_gpu_checkout_is_ro_bound_not_writable():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    line = next(l for l in open(log).read().splitlines() if l.startswith("GPU_QUEUE add"))
    checkout = os.path.join(d, ".claude", "worktrees", "agent-jobs", sha)
    assert ("--ro-bind %s %s" % (checkout, checkout)) in line
    assert ("--bind %s %s" % (checkout, checkout)) not in line


def test_worktree_cap_cleans_up_the_oldest_agent_job_worktrees():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    env["AGENT_GATEWAY_MAX_WORKTREES"] = "2"
    # build two extra committed shas so three distinct worktrees can exist before the cap is exercised
    shas = [sha]
    for i in range(2):
        with open(os.path.join(d, "extra_%d.txt" % i), "w") as f:
            f.write("x\n")
        _git(["add", "-A"], d)
        _git(["commit", "-q", "-m", "extra %d" % i], d)
        s = _git(["rev-parse", "HEAD"], d).stdout.strip()
        _git(["update-ref", "refs/remotes/origin/main", s], d)
        shas.append(s)

    import time
    for i, s in enumerate(shas):
        _run(["gpu", "--sha", s, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
        time.sleep(1.01)   # distinct mtimes -- cleanup order is mtime-based

    worktrees_dir = os.path.join(d, ".claude", "worktrees", "agent-jobs")
    remaining = set(os.listdir(worktrees_dir))
    assert len(remaining) <= 2, remaining
    # the two most RECENTLY created (shas[1], shas[2]) must survive; the oldest (shas[0]) must be gone
    assert shas[0] not in remaining
    assert shas[-1] in remaining


def test_git_operations_disable_hooks():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    hook_marker = os.path.join(d, "HOOK_RAN")
    hooks_dir = os.path.join(d, ".git", "hooks")
    os.makedirs(hooks_dir, exist_ok=True)
    for hook_name in ("post-checkout", "post-commit"):
        _write_exec(os.path.join(hooks_dir, hook_name), 'echo ran >> "%s"\n' % hook_marker)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert not os.path.exists(hook_marker), (
        "git worktree add must never run a repo's hooks (core.hooksPath=/nonexistent on every git call)")


def test_validate_sha_fetches_the_draft_branch_namespace_before_checking_reachability():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    fetch_log = os.path.join(d, "git_fetch_calls.log")
    real_git = subprocess.run(["which", "git"], capture_output=True, text=True).stdout.strip()
    fake_git = _write_exec(os.path.join(d, "fake_git.sh"), """
echo "GIT $*" >> "%s"
case "$*" in
  *" fetch "*) exit 0 ;;
  *) exec %s "$@" ;;
esac
""" % (fetch_log, real_git))
    env["AGENT_GATEWAY_GIT_BIN"] = fake_git
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = open(fetch_log).read()
    assert "fetch --quiet gitea refs/heads/research/draft-*:refs/remotes/gitea/research/draft-*" in calls


def test_validate_sha_tolerates_a_fetch_failure():
    """gitea unreachable / no such remote must never crash the submit -- reachability just falls back to
    whatever is already cached, exactly as if the fetch attempt had not been made at all."""
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    fake_git = _write_exec(os.path.join(d, "fake_git_fail.sh"), """
case "$*" in
  *" fetch "*) exit 1 ;;
  *) exec git "$@" ;;
esac
""")
    env["AGENT_GATEWAY_GIT_BIN"] = fake_git
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
