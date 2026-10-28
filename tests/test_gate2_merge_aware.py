"""pre-commit GATE 2 (claims) is merge-aware: a merge is checked for what it adds, never for what main already carried.

2026-10-28: GATE 2 ran tools/claim_check.py on every findings doc ADDED relative to the FIRST parent, so a main-into-branch
merge (tools/agent_sync_branch.sh, every Hermes sync) re-checked all of main's new findings, and one stale citation on
main (research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md, a placeholder `--out` path added by an
amendment) blocked every branch sync. tools/merge_carried.py now drops, while MERGE_HEAD is present, a path whose staged
content is byte-identical to a merge parent that main already contains.

The scratch repos run the REAL hook code: the GATE 2 block is cut verbatim out of tools/githooks/pre-commit (only its
scratch output file is renamed) and installed as the scratch repo's pre-commit, next to the real pre-merge-commit, and
the real tools/claim_check.py and tools/merge_carried.py are committed into the scratch tree. Every merge is concluded
the way the repo's merge tools do it: `git merge --no-ff --no-commit` + `git commit`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import tools.merge_carried as mc  # noqa: E402

_STRIP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
          "GIT_CEILING_DIRECTORIES", "GIT_PREFIX")
ART = "research/findings/raw/x/ok.json"
GOOD = "---\nstatus: live\n---\n# good\n\nvalue 0.123456 read from `%s`\n" % ART
STALE = "---\nstatus: live\n---\n# stale\n\nvalue 0.123456 read from `research/findings/raw/x/missing.json`\n"
MAIN_DOC = "research/findings/2026-01-02-main-prereg.md"
LANE_DOC = "research/findings/2026-01-02-lane-finding.md"


def _env():
    env = dict(os.environ)
    for k in _STRIP:
        env.pop(k, None)
    return env


def _git(root, *args, check=True):
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=root, env=_env(), capture_output=True,
                       text=True, timeout=120)
    if check:
        assert r.returncode == 0, "git %s failed: %s %s" % (" ".join(args), r.stdout, r.stderr)
    return r


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


def _gate2_block():
    """The GATE 2 block of the REAL hook, verbatim, from its `# GATE 2` comment to the `# GATE 3` comment."""
    lines = open(os.path.join(_REPO, "tools", "githooks", "pre-commit"), encoding="utf-8").read().split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("# GATE 2"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("# GATE 3"))
    return "\n".join(lines[start:end])


def _hook_text(block, out):
    return ("#!/usr/bin/env bash\nset -uo pipefail\nROOT=$(git rev-parse --show-toplevel); export ROOT\n"
            "COMMON=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 1\n"
            "CANONICAL=$(dirname \"$COMMON\")\nPY=%s\n%s\nexit 0\n"
            % (sys.executable, block.replace("/tmp/_precommit_claims.out", out)))


def _make_repo(tmp_path, block=None):
    """main: the artifact + the two real tools, committed. Hooks: the cut GATE 2 block + the real pre-merge-commit."""
    root = os.path.realpath(str(tmp_path / "main"))
    os.makedirs(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    for rel in ("tools/claim_check.py", "tools/merge_carried.py"):
        os.makedirs(os.path.join(root, "tools"), exist_ok=True)
        shutil.copy(os.path.join(_REPO, rel), os.path.join(root, rel))
    hooks = os.path.join(root, "tools", "githooks")
    os.makedirs(hooks)
    with open(os.path.join(hooks, "pre-commit"), "w") as fh:
        fh.write(_hook_text(block if block is not None else _gate2_block(), str(tmp_path / "claims.out")))
    shutil.copy(os.path.join(_REPO, "tools", "githooks", "pre-merge-commit"), os.path.join(hooks, "pre-merge-commit"))
    for h in ("pre-commit", "pre-merge-commit"):
        os.chmod(os.path.join(hooks, h), 0o755)
    with open(os.path.join(root, ".git", "info", "exclude"), "a") as fh:
        fh.write("/tools/githooks/\n")
    _git(root, "config", "core.hooksPath", hooks)
    _write(root, ART, '{"v": 0.123456}')
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _blocked(r):
    return r.returncode != 0 and "COMMIT BLOCKED" in (r.stdout + r.stderr)


def _main_gains_a_stale_doc(root):
    """The real shape: added clean (GATE 2 passes it), then an AMENDMENT (a modify, which GATE 2 does not check)
    cites a file that does not exist. Main now carries a stale claim no gate saw."""
    _write(root, MAIN_DOC, GOOD)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "main: prereg")
    _write(root, MAIN_DOC, GOOD + STALE)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "main: amendment cites a placeholder")


@pytest.fixture()
def synced(tmp_path):
    """A lane worktree forked at base, then main gains a stale doc; returns (main_root, lane_root)."""
    root = _make_repo(tmp_path)
    lane = str(tmp_path / "lane")
    _git(root, "worktree", "add", "-q", "-b", "research/lane", lane)
    _write(lane, LANE_DOC, GOOD)
    _git(lane, "add", "-A")
    _git(lane, "commit", "-q", "-m", "lane: finding")
    _main_gains_a_stale_doc(root)
    return root, lane


# ------------------------------------------------------------------------------------------------------------------
# the three cases the fix is for
# ------------------------------------------------------------------------------------------------------------------
def test_sync_merge_of_main_with_a_stale_claim_on_main_passes(synced):
    root, lane = synced
    _git(lane, "merge", "--no-ff", "--no-commit", "main")
    r = _git(lane, "commit", "-m", "sync main", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "not re-checked: %s" % MAIN_DOC in r.stdout + r.stderr


def test_power_control_the_same_sync_blocks_without_the_merge_scope(tmp_path):
    """The pre-fix shape (the merge block disabled) blocks the same sync: the test above is not vacuous."""
    block = _gate2_block()
    assert '[ -f "$_MH" ]' in block
    root = _make_repo(tmp_path, block=block.replace('[ -f "$_MH" ]', "false"))
    lane = str(tmp_path / "lane")
    _git(root, "worktree", "add", "-q", "-b", "research/lane", lane)
    _main_gains_a_stale_doc(root)
    _git(lane, "merge", "--no-ff", "--no-commit", "main")
    assert _blocked(_git(lane, "commit", "-m", "sync main", check=False))


def test_merge_that_adds_a_new_stale_claim_still_blocks(synced):
    root, lane = synced
    _git(lane, "merge", "--no-ff", "--no-commit", "main")
    _write(lane, "research/findings/2026-01-03-written-in-the-merge.md", STALE)
    _git(lane, "add", "-A")
    r = _git(lane, "commit", "-m", "sync main + a new doc", check=False)
    assert _blocked(r)
    assert "2026-01-03-written-in-the-merge.md" in r.stdout + r.stderr


def test_merge_that_edits_mains_doc_is_checked_as_new_content(synced):
    """A conflict resolution or evil merge that rewrites main's doc matches no parent, so it is checked."""
    root, lane = synced
    _git(lane, "merge", "--no-ff", "--no-commit", "main")
    _write(lane, MAIN_DOC, STALE + "\nedited during the merge\n")
    _git(lane, "add", "-A")
    assert _blocked(_git(lane, "commit", "-m", "sync main, edited", check=False))


def test_normal_commit_is_unchanged(synced):
    root, lane = synced
    _write(lane, "research/findings/2026-01-04-stale.md", STALE)
    _git(lane, "add", "-A")
    r = _git(lane, "commit", "-m", "stale", check=False)
    assert _blocked(r) and "not re-checked" not in r.stdout + r.stderr
    _git(lane, "rm", "-q", "--cached", "research/findings/2026-01-04-stale.md")
    os.remove(os.path.join(lane, "research/findings/2026-01-04-stale.md"))
    _write(lane, "research/findings/2026-01-04-good.md", GOOD)
    _git(lane, "add", "-A")
    assert _git(lane, "commit", "-m", "good", check=False).returncode == 0


# ------------------------------------------------------------------------------------------------------------------
# what the rule must NOT exempt, and where main lives
# ------------------------------------------------------------------------------------------------------------------
def test_landing_a_branch_into_main_still_checks_the_branchs_new_docs(tmp_path):
    """land_claude_branch.sh / agent_merge.sh: the landing is where the gates run for work committed without them.
    The branch parent is not on main, so its new docs are checked (the literal 'identical to ANY parent' rule would
    have exempted them)."""
    root = _make_repo(tmp_path)
    _git(root, "checkout", "-q", "-b", "claude/cloud")
    _write(root, "research/findings/2026-01-05-cloud.md", STALE)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--no-verify", "-m", "made where no hook ran")
    _git(root, "checkout", "-q", "main")
    _write(root, "README", "main moves on\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "main moves")
    _git(root, "merge", "--no-ff", "--no-commit", "claude/cloud")
    assert _blocked(_git(root, "commit", "-m", "land", check=False))


def test_a_remote_main_counts_as_main(synced):
    """The workstation's local main is a stale follower; origin/main is the trunk it merges."""
    root, lane = synced
    tip = _git(root, "rev-parse", "main").stdout.strip()
    _git(root, "update-ref", "refs/remotes/origin/main", tip)
    _git(root, "update-ref", "refs/heads/main", _git(root, "rev-parse", "main~2").stdout.strip())
    _git(lane, "merge", "--no-ff", "--no-commit", "origin/main")
    assert _git(lane, "commit", "-m", "sync origin/main", check=False).returncode == 0


def test_merge_of_a_parent_main_does_not_contain_is_checked(synced, tmp_path):
    """Two branches, neither on main: the other branch's stale doc is checked, as before."""
    root, lane = synced
    other = str(tmp_path / "other")
    _git(root, "worktree", "add", "-q", "-b", "research/other", other, "main~2")
    _write(other, "research/findings/2026-01-06-other.md", STALE)
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "--no-verify", "-m", "other")
    _git(lane, "merge", "--no-ff", "--no-commit", "research/other")
    assert _blocked(_git(lane, "commit", "-m", "merge other", check=False))


# ------------------------------------------------------------------------------------------------------------------
# the helper itself
# ------------------------------------------------------------------------------------------------------------------
def test_helper_without_merge_head_returns_its_input(synced):
    root, lane = synced
    paths = [MAIN_DOC, LANE_DOC]
    assert mc.split(paths, root=lane) == (paths, [], "")


def test_helper_fails_closed_on_an_unreadable_merge(synced):
    root, lane = synced
    _git(lane, "merge", "--no-ff", "--no-commit", "main")
    mh = _git(lane, "rev-parse", "--path-format=absolute", "--git-path", "MERGE_HEAD").stdout.strip()
    assert mc.split([MAIN_DOC], root=lane)[0] == []          # readable: carried
    with open(mh, "w") as fh:
        fh.write("0" * 40 + "\n")                            # names no commit git has
    keep, carried, note = mc.split([MAIN_DOC], root=lane)
    assert keep == [MAIN_DOC] and carried == [] and "failing CLOSED" in note
    _git(lane, "merge", "--abort", check=False)


def test_hook_keeps_its_list_when_the_helper_crashes(tmp_path):
    """A non-zero helper exit must leave GATE 2's list unchanged (fail closed), not empty it."""
    root = _make_repo(tmp_path)
    lane = str(tmp_path / "lane")
    _git(root, "worktree", "add", "-q", "-b", "research/lane", lane)
    _main_gains_a_stale_doc(root)
    with open(os.path.join(lane, "tools", "merge_carried.py"), "w") as fh:
        fh.write("import sys\nsys.exit(3)\n")
    _git(lane, "merge", "--no-ff", "--no-commit", "main")
    assert _blocked(_git(lane, "commit", "-m", "sync main", check=False))
