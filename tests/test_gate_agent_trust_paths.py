"""tools/gates/agent_trust_paths.py (class TP) -- 2026-09-30 security review, MEDIUM: everything the server runs as
the owner (docker group, sudo) or as root at boot is built from `src/`, which tracks GitHub main (the ctl forced
command, update.sh, the Dockerfile/entrypoint, the nft/systemd files); on the workstation, core.hooksPath and
.claude/settings.json hooks execute tools/githooks/** and .claude/** at the owner's next commit or session. The
agent's only route into main is a research/draft-* branch the owner merges, and nothing flagged diffs to those
paths. This gate blocks a merge (or squash merge) of commits reachable from a research/draft-* ref that changes a
trusted path, until the owner acknowledges the exact incoming sha; its CLI is the pre-merge review.

Each test builds a real scratch repository: main, and an agent branch fetched as refs/remotes/gitea/research/draft-*.
"""
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.gates import agent_trust_paths as gate  # noqa: E402

_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", repo, "-c", "core.hooksPath=/dev/null", *args], env=_ENV,
                          capture_output=True, text=True, check=check).stdout.strip()


def _write(repo, rel, text="x\n"):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _repo(tmp, agent_files, branch="research/draft-feature", remote_ref=True):
    repo = os.path.join(tmp, "r")
    os.makedirs(repo)
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "README.md", "base\n")
    _write(repo, "tools/githooks/pre-commit", "#!/bin/sh\nexit 0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", branch)
    for rel in agent_files:
        _write(repo, rel, "agent change\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "agent work")
    tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "docs/owner.md", "owner work on main\n")   # so the merge is a real (non-ff) merge
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "owner work")
    if remote_ref:   # how the owner sees it: fetched from gitea
        _git(repo, "update-ref", "refs/remotes/gitea/" + branch, tip)
        _git(repo, "branch", "-q", "-D", branch)
    return repo, tip


def _merge(repo, ref):
    _git(repo, "merge", "--no-ff", "--no-commit", ref, check=False)


def test_a_draft_merge_touching_a_trusted_path_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        repo, tip = _repo(tmp, ["tools/githooks/pre-commit", "research/notes.md"])
        _merge(repo, "gitea/research/draft-feature")
        probs = gate.check_repo(repo, ack="")
        assert len(probs) == 1, probs
        assert "tools/githooks/pre-commit" in probs[0] and "research/notes.md" not in probs[0]
        assert tip in probs[0] and "SIM_AGENT_TRUST_ACK" in probs[0]


@pytest.mark.parametrize("path", ["tools/agent_container/entrypoint.sh", "tools/agent_harness_ctl.sh",
                                  "tools/agent_gateway_submit.py", ".claude/settings.json",
                                  "tools/agent_container/server-stack/host/etc/nftables.d/sim-agent.nft",
                                  "tools/gates/agent_trust_paths.py"])
def test_every_trusted_path_family_is_covered(path):
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _repo(tmp, [path])
        _merge(repo, "gitea/research/draft-feature")
        assert gate.check_repo(repo, ack=""), path


def test_the_exact_incoming_sha_acknowledges_it_and_nothing_else_does():
    with tempfile.TemporaryDirectory() as tmp:
        repo, tip = _repo(tmp, ["tools/agent_harness_ctl.sh"])
        _merge(repo, "gitea/research/draft-feature")
        assert gate.check_repo(repo, ack=tip) == []
        assert gate.check_repo(repo, ack=tip[:12]) == []
        assert gate.check_repo(repo, ack="0" * 40), "an ack for another sha must not cover this one"
        assert gate.check_repo(repo, ack=tip[:6]), "a short prefix is not an acknowledgement"


def test_a_draft_merge_touching_only_ordinary_paths_passes():
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _repo(tmp, ["research/runners/_draft_x.py", "docs/drafts/x.md"])
        _merge(repo, "gitea/research/draft-feature")
        assert gate.check_repo(repo, ack="") == []


def test_an_owner_branch_touching_a_trusted_path_is_not_this_gates_business():
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _repo(tmp, ["tools/githooks/pre-commit"], branch="research/owner-topic")
        _merge(repo, "gitea/research/owner-topic")
        assert gate.check_repo(repo, ack="") == []


def test_a_local_draft_branch_counts_too_and_so_does_a_squash_merge():
    # a clean cherry-pick, a rebase and a fast-forward run no commit hook at all, so no hook-time gate can see them:
    # the pre-merge review CLI (below) is the control for those, and docs/AGENT_HARNESS_SETUP.md says so
    with tempfile.TemporaryDirectory() as tmp:
        repo, tip = _repo(tmp, ["tools/agent_container/Dockerfile"], remote_ref=False)
        _merge(repo, "research/draft-feature")
        assert gate.check_repo(repo, ack="")
        _git(repo, "merge", "--abort")
        _git(repo, "merge", "--squash", "research/draft-feature")
        probs = gate.check_repo(repo, ack="")
        assert probs and tip in probs[0], "a squash merge of an agent branch must be caught too"


def test_no_merge_in_progress_means_nothing_to_check():
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _repo(tmp, ["tools/githooks/pre-commit"])
        assert gate.check_repo(repo, ack="") == []


def test_the_review_cli_lists_trusted_changes_between_head_and_a_ref():
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _repo(tmp, ["tools/githooks/pre-commit", "research/notes.md"])
        r = subprocess.run([sys.executable, "-m", "tools.gates.agent_trust_paths", "--repo", repo,
                            "gitea/research/draft-feature"], cwd=ROOT, capture_output=True, text=True, env=_ENV)
        assert r.returncode == 1, r.stdout + r.stderr
        assert "tools/githooks/pre-commit" in r.stdout and "research/notes.md" not in r.stdout
        assert "git diff" in r.stdout
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _repo(tmp, ["research/notes.md"])
        r = subprocess.run([sys.executable, "-m", "tools.gates.agent_trust_paths", "--repo", repo,
                            "gitea/research/draft-feature"], cwd=ROOT, capture_output=True, text=True, env=_ENV)
        assert r.returncode == 0, r.stdout + r.stderr


def test_selftest_fails_in_the_failing_direction():
    assert gate.selftest() == []
    assert gate.BLOCKING is True
