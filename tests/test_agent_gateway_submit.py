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


def test_gpu_refuses_an_out_path_outside_the_allowed_root():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--",
                "--seed", "7", "--out", "research/coordination/x.json"], env)
    assert res.returncode != 0
    assert "does not resolve under research/findings/raw" in res.stderr


def test_gpu_refuses_an_out_path_with_dotdot():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--",
                "--seed", "7", "--out=research/findings/raw/../../etc/passwd"], env)
    assert res.returncode != 0
    assert "does not resolve under research/findings/raw" in res.stderr or "contains '..'" in res.stderr


def test_gpu_accepts_an_out_path_inside_the_allowed_root():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--",
                "--seed", "7", "--out", "research/findings/raw/o.json"], env)
    assert res.returncode == 0, res.stdout + res.stderr


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
    assert "--ro-bind / /" in line
    assert "--unshare-net" in line
    assert "--die-with-parent" in line
    assert "--tmpfs" in line and ".ssh" in line
    checkout = os.path.join(d, ".claude", "worktrees", "agent-jobs", sha)
    assert checkout in line
    out_dir = os.path.join(d, "research", "findings", "raw")
    assert out_dir in line
    assert "bash -c" in line
    assert "research.runners.foo" in line


def test_gpu_creates_a_detached_worktree_pinned_at_the_sha():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["gpu", "--sha", sha, "--runner", "research.runners.foo", "--", "--seed", "7"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    checkout = os.path.join(d, ".claude", "worktrees", "agent-jobs", sha)
    assert os.path.isdir(checkout)
    head = _git(["rev-parse", "HEAD"], checkout).stdout.strip()
    assert head == sha
    status = _git(["status", "--porcelain"], checkout).stdout
    assert status == ""


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
    assert "POOL_PROVISION --revision %s --isolated" % sha in calls


def test_pool_refuses_an_argument_with_shell_metacharacters():
    d, sha, _ = _make_repo()
    log, env = _stub_set(d)
    res = _run(["pool", "--sha", sha, "--runner", "research.runners.foo",
                "--checked", "corpus:nothing", "--", "--seed", "7;rm -rf ~"], env)
    assert res.returncode != 0
    assert "allowed pattern" in res.stderr
