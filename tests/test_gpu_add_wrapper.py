"""tools/local_llm/gpu_add.sh: the ONLY way the local model may queue a GPU job (tools/gpu_queue.sh add itself
stays denied to it -- tools/local_llm/claude_local_settings.json, pinned by
tests/test_local_llm_permissions.py::test_raw_gpu_queue_add_stays_denied). Refuses a job that does not `cd` into
a COMMITTED, CLEAN git checkout, and refuses a GATE seed (42/43/44/100/101/102) with no committed --prereg.

Hermetic: every scratch git checkout lives in a tempdir this file creates and removes itself; `GPU_QUEUE_DIR` and
`GPU_ADD_QUEUE_SH` (a TEST-ONLY override this file adds nothing new for -- gpu_add.sh forwards to the real
tools/gpu_queue.sh, which already isolates on GPU_QUEUE_DIR) keep every enqueue in an isolated queue dir, never
the live research/queue/.
"""
import os
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GPU_ADD = os.path.join(_ROOT, "tools", "local_llm", "gpu_add.sh")


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True, text=True)


def _make_checkout(commit_prereg=None, dirty=False):
    """Returns a tempdir path holding a fresh git checkout, committed, with an optional --prereg-style file
    also committed at the same revision. `dirty=True` leaves a tracked file modified afterward."""
    d = tempfile.mkdtemp()
    _git(["init", "-q"], d)
    _git(["config", "user.email", "t@t.com"], d)
    _git(["config", "user.name", "t"], d)
    open(os.path.join(d, "README"), "w").write("x\n")
    if commit_prereg:
        os.makedirs(os.path.dirname(os.path.join(d, commit_prereg)), exist_ok=True)
        open(os.path.join(d, commit_prereg), "w").write("prereg\n")
    _git(["add", "-A"], d)
    _git(["commit", "-q", "-m", "init"], d)
    if dirty:
        open(os.path.join(d, "README"), "w").write("y\n")
    return d


def _run(cmd, qdir, timeout=10):
    env = dict(os.environ)
    env["GPU_QUEUE_DIR"] = qdir
    return subprocess.run(["bash", _GPU_ADD, cmd], env=env, capture_output=True, text=True, timeout=timeout)


def test_refuses_a_job_with_no_cd():
    with tempfile.TemporaryDirectory() as qdir:
        res = _run(".venv/bin/python -m research.runners.x --seed 7 --json raw/o.json", qdir)
        assert res.returncode != 0
        assert "cd <checkout>" in res.stderr
        assert not os.path.exists(os.path.join(qdir, "gpu.queue")) or open(os.path.join(qdir, "gpu.queue")).read() == ""


def test_refuses_a_cd_target_that_is_not_a_git_checkout():
    with tempfile.TemporaryDirectory() as qdir, tempfile.TemporaryDirectory() as notgit:
        res = _run("cd %s && echo hi" % notgit, qdir)
        assert res.returncode != 0
        assert "not a git checkout" in res.stderr


def test_refuses_a_dirty_checkout():
    with tempfile.TemporaryDirectory() as qdir:
        co = _make_checkout(dirty=True)
        try:
            res = _run("cd %s && .venv/bin/python -m research.runners.x --seed 7 --json raw/o.json" % co, qdir)
            assert res.returncode != 0
            assert "uncommitted changes" in res.stderr
        finally:
            subprocess.run(["rm", "-rf", co])


def test_allows_a_clean_checkout_with_a_dev_seed():
    with tempfile.TemporaryDirectory() as qdir:
        co = _make_checkout()
        try:
            res = _run("cd %s && .venv/bin/python -m research.runners.x --seed 7 --json raw/o.json" % co, qdir)
            assert res.returncode == 0, res.stdout + res.stderr
            queued = open(os.path.join(qdir, "gpu.queue")).read()
            assert co in queued and "--seed 7" in queued
        finally:
            subprocess.run(["rm", "-rf", co])


def test_refuses_a_gate_seed_with_no_prereg():
    with tempfile.TemporaryDirectory() as qdir:
        co = _make_checkout()
        try:
            res = _run("cd %s && .venv/bin/python -m research.runners.x --seed 42 --json raw/o.json" % co, qdir)
            assert res.returncode != 0
            assert "GATE seed" in res.stderr
            assert not os.path.exists(os.path.join(qdir, "gpu.queue")) or open(os.path.join(qdir, "gpu.queue")).read() == ""
        finally:
            subprocess.run(["rm", "-rf", co])


def test_refuses_a_gate_seeds_list_with_a_prereg_that_is_not_committed():
    with tempfile.TemporaryDirectory() as qdir:
        co = _make_checkout()
        try:
            res = _run(
                "cd %s && .venv/bin/python -m research.runners.x --seeds \"42 43 44 100 101 102\" "
                "--prereg docs/drafts/x/PREREG.md --json raw/o.json" % co, qdir)
            assert res.returncode != 0
            assert "not a file committed" in res.stderr
        finally:
            subprocess.run(["rm", "-rf", co])


def test_allows_a_gate_seed_with_a_committed_prereg_flag():
    with tempfile.TemporaryDirectory() as qdir:
        co = _make_checkout(commit_prereg="docs/drafts/x/PREREG.md")
        try:
            res = _run(
                "cd %s && .venv/bin/python -m research.runners.x --seed 100 "
                "--prereg docs/drafts/x/PREREG.md --json raw/o.json" % co, qdir)
            assert res.returncode == 0, res.stdout + res.stderr
        finally:
            subprocess.run(["rm", "-rf", co])


def test_allows_a_gate_seed_with_a_committed_prereg_env_assignment():
    """The same style as the existing `SIM_BACKEND=cupy ...` job-prefix convention."""
    with tempfile.TemporaryDirectory() as qdir:
        co = _make_checkout(commit_prereg="docs/drafts/x/PREREG.md")
        try:
            res = _run(
                "cd %s && PREREG=docs/drafts/x/PREREG.md .venv/bin/python -m research.runners.x "
                "--seed 101 --json raw/o.json" % co, qdir)
            assert res.returncode == 0, res.stdout + res.stderr
        finally:
            subprocess.run(["rm", "-rf", co])


def test_cd_path_is_never_evaluated_as_a_shell_command():
    """2026-09-29 security review, H2: `eval echo $cd_path` ran the caller-supplied path through a SECOND round
    of shell parsing. `$(id>marker)` is ONE whitespace-free token (survives the awk field-split that extracts
    cd_path intact), so under the old `eval` it executed `id`, redirecting its output to `marker` -- creating
    it -- rather than merely "expanding a variable". The path is not a real git checkout either way, so the job
    is refused regardless; the only question this test asks is whether the marker file got created."""
    with tempfile.TemporaryDirectory() as qdir, tempfile.TemporaryDirectory() as scratch:
        marker = os.path.join(scratch, "pwned")
        res = _run("cd $(id>%s) && echo hi" % marker, qdir)
        assert res.returncode != 0
        assert not os.path.exists(marker), "cd_path must never be evaluated as a shell command"


def test_a_number_that_merely_contains_a_gate_seed_is_not_a_false_positive():
    """--seed 142 must not be mistaken for gate seed 42 -- exact numeric match only."""
    with tempfile.TemporaryDirectory() as qdir:
        co = _make_checkout()
        try:
            res = _run("cd %s && .venv/bin/python -m research.runners.x --seed 142 --json raw/o.json" % co, qdir)
            assert res.returncode == 0, res.stdout + res.stderr
        finally:
            subprocess.run(["rm", "-rf", co])
