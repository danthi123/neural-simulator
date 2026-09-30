"""tools/githooks/pre-merge-commit's own core.hooksPath self-check compared `$NORMALIZED_HOOKS` only against
`$ROOT` (via `git rev-parse --show-toplevel`) -- but in a LINKED WORKTREE, `$ROOT` is the worktree's own path,
while `core.hooksPath` is repo-common config shared by every worktree, typically configured as an absolute path
into the PRIMARY checkout. So a worktree merge with the correctly-configured shared hooksPath was refused with
"git config core.hooksPath is '<primary>/tools/githooks'. It should be either 'tools/githooks' (relative) or the
absolute path to tools/githooks." -- found 2026-09-29 landing host-share-baselines' merge of origin/main.

Fixed: also accept `$CANONICAL/tools/githooks` (`$CANONICAL` = dirname of `git rev-parse --git-common-dir`, the
primary checkout's root regardless of which worktree is running), and exec the canonical copy of pre-commit when
present. This test builds a REAL linked worktree off this repo (a lightweight, throwaway addition; removed in
teardown) and runs the hook script directly from it, verifying both directions: the pre-fix condition (ROOT-only
comparison) refuses, and the current script accepts."""
import os
import subprocess
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK_REL = "tools/githooks/pre-merge-commit"


def _run_hook(cwd, hooks_path, hook_text=None):
    """Run the hook (or a substituted script text, for the pre-fix-condition direction) from `cwd` with
    core.hooksPath temporarily set to `hooks_path`. Returns the completed process."""
    hook_abs = os.path.join(cwd, HOOK_REL)
    original = None
    if hook_text is not None:
        original = open(hook_abs).read()
        with open(hook_abs, "w") as f:
            f.write(hook_text)
        os.chmod(hook_abs, 0o755)
    old = subprocess.run(["git", "config", "--get", "core.hooksPath"], cwd=cwd, capture_output=True, text=True).stdout
    try:
        subprocess.run(["git", "config", "core.hooksPath", hooks_path], cwd=cwd, check=True)
        return subprocess.run(["bash", HOOK_REL], cwd=cwd, capture_output=True, text=True)
    finally:
        if original is not None:
            with open(hook_abs, "w") as f:
                f.write(original)
            os.chmod(hook_abs, 0o755)
        if old.strip():
            subprocess.run(["git", "config", "core.hooksPath", old.strip()], cwd=cwd, check=True)
        else:
            subprocess.run(["git", "config", "--unset", "core.hooksPath"], cwd=cwd)


@pytest.fixture
def linked_worktree():
    with tempfile.TemporaryDirectory() as d:
        wt = os.path.join(d, "wt")
        subprocess.run(["git", "worktree", "add", "--detach", wt, "HEAD"], cwd=ROOT, capture_output=True, text=True,
                       check=True)
        try:
            yield wt
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=ROOT, capture_output=True, text=True)


# The exact pre-fix condition (ROOT-only comparison), reconstructed from git history's known-bad line so this
# test does not depend on ever re-checking out an old blob.
_PRE_FIX_CHECK = '''#!/usr/bin/env bash
set -uo pipefail
ROOT=$(git rev-parse --show-toplevel); export ROOT
COMMON=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 1
CANONICAL=$(dirname "$COMMON")
HOOKS_PATH=$(git config core.hooksPath 2>/dev/null || echo "")
NORMALIZED_HOOKS="${HOOKS_PATH%/}"
if [ -z "$NORMALIZED_HOOKS" ] || ! { [ "$NORMALIZED_HOOKS" = "tools/githooks" ] || [ "$NORMALIZED_HOOKS" = "$ROOT/tools/githooks" ]; }; then
  echo "REFUSED: $HOOKS_PATH"
  exit 1
fi
exec "$ROOT/tools/githooks/pre-commit"
'''


def test_pre_fix_condition_refuses_a_worktree_with_the_shared_absolute_hookspath(linked_worktree):
    """Proves the bug is real and this harness reproduces it: the ROOT-only comparison rejects the primary
    checkout's absolute hooksPath when run from a linked worktree."""
    r = _run_hook(linked_worktree, os.path.join(ROOT, "tools/githooks"), hook_text=_PRE_FIX_CHECK)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSED" in r.stdout


def test_current_hook_accepts_the_shared_absolute_hookspath_from_a_worktree(linked_worktree):
    """The fix: the current script accepts the primary checkout's absolute hooksPath from a linked worktree.

    `git worktree add` checks out tools/githooks/pre-merge-commit from the COMMIT being checked out, not from
    this repo's own (possibly-uncommitted) working tree -- so the CURRENT script's text is passed explicitly via
    `hook_text`, exactly like the pre-fix-condition test above does for the known-bad text. This is what makes
    the test exercise this repo's actual current fix rather than whatever was last committed."""
    current_script = open(os.path.join(ROOT, HOOK_REL)).read()
    r = _run_hook(linked_worktree, os.path.join(ROOT, "tools/githooks"), hook_text=current_script)
    assert "REFUSED" not in r.stdout, r.stdout + r.stderr
    assert "pre-merge-commit: git config core.hooksPath" not in r.stdout, r.stdout + r.stderr


def test_current_hook_still_accepts_the_relative_form(linked_worktree):
    """The plain 'tools/githooks' relative form (git resolves it against $ROOT itself) must keep working."""
    current_script = open(os.path.join(ROOT, HOOK_REL)).read()
    r = _run_hook(linked_worktree, "tools/githooks", hook_text=current_script)
    assert "REFUSED" not in r.stdout
    assert "pre-merge-commit: git config core.hooksPath" not in r.stdout
