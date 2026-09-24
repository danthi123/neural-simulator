"""Tests for the protected-delete PreToolUse hook (.claude/hooks/guard_protected_delete.py).

Both directions are pinned, as for block_self_matching_kill.py: the guard must BLOCK the 2026-09-24 incident and its
variants, and must NOT block the everyday deletes and the text that merely DISCUSSES deleting (commit messages,
echo, grep, heredocs fed to cat), because a guard that cries wolf gets switched off.

    .venv/bin/python -m pytest tests/test_guard_protected_delete.py -q
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".claude", "hooks",
                    "guard_protected_delete.py")
_spec = importlib.util.spec_from_file_location("guard_protected_delete", HOOK)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


@pytest.fixture()
def home(tmp_path):
    h = tmp_path / "home" / "u"
    for d in (".claude/projects/-p/memory", "Projects/sim/.claude/worktrees/x", "Projects/sim/research",
              "Documents/empty", ".cache/pip"):
        (h / d).mkdir(parents=True)
    (h / ".claude/projects/-p/memory/note.md").write_text("x")
    (h / "notes.txt").write_text("x")
    return str(h)


def _check(cmd, home, cwd=None):
    return guard.analyze(cmd.replace("H/", home + "/").replace(" H ", " " + home + " "),
                         (cwd or "H").replace("H", home, 1), home=home, env={})


BLOCK = [
    # the incident itself, and the forms it could have taken
    "rm -rf H/Projects/sim/.claude/worktrees/scratch_s06_clone H/.claude/projects",
    "rm -rf ~/.claude",
    "rm -rf ~/.claude/projects",
    'rm -rf "$HOME/.claude/projects"',
    "rm -r ${HOME}/.claude/projects/-p",
    "rm -fr -- ~/.claude/",
    "sudo -u root rm -rf /mnt/vault/x",
    "timeout -k 5 10 rm -rf ~/.claude",
    "rm -rf --no-preserve-root /",
    "rm -rf /",
    "rm -rf /home",
    "rm -rf /mnt",
    "rm -rf /mnt/vault",
    "rm -rf ~",
    "rm -rf ~/",
    'rm -rf "$HOME"',
    "rm -rf ~/Projects",
    "rm -rf ~/Projects/sim",
    "rm -rf ~/Documents",
    "rm -rf ~/*",
    "rm -rf ~/.claude/*",
    "rm -rf ~/.claude.json",
    "rm H/.claude/projects/-p/memory/*.md",
    "rm -rf H/.claude/projects/-p/memory/note.md",
    # working-directory tricks, including a cd that fails (the agent's own story)
    "cd ~ && rm -rf .claude",
    "cd ~/.claude && rm -rf projects",
    "cd ~/Projects/sim && rm -rf .",
    'cd "$SOMEWHERE_UNSET" && rm -rf .',
    "cd $(git rev-parse --show-toplevel) && rm -rf .",
    # unset variables expand to nothing
    'rm -rf "$UNSET_A/$UNSET_B"',
    'REPO=$(git rev-parse --show-toplevel); rm -rf "$REPO"/*',
    "X=H/.claude; rm -rf \"$X/projects\"",
    # other delete verbs
    "find ~/.claude/projects -name '*.jsonl' -delete",
    "find ~/.claude -exec rm -rf {} +",
    "find ~ -name '*.pyc' -delete",
    "mv ~/.claude/projects /tmp/x",
    "rsync -a --delete /tmp/empty/ ~/.claude/",
    "git worktree remove --force H/Projects/sim",
    "trash-put ~/.claude",
    "gio trash ~/.claude",
    "shred -u ~/.claude.json",
    # loops, pipes, substitutions, nested shells
    'for d in ~/.claude/projects/*; do rm -rf "$d"; done',
    'for f in $(ls ~/.claude/projects); do rm -rf "$f"; done',
    "ls ~/.claude/projects | xargs rm -rf",
    'find ~/.claude -type f | while read f; do rm "$f"; done',
    "echo $(rm -rf ~/.claude)",
    "echo `rm -rf ~/.claude`",
    'echo "$(rm -rf ~/.claude)"',
    'bash -c "rm -rf ~/.claude/projects"',
    "sh -c 'cd ~ && rm -rf .claude'",
    "eval rm -rf ~/.claude",
    "bash <<'EOF'\nrm -rf ~/.claude/projects\nEOF",
    # inline code
    "python3 -c \"import shutil; shutil.rmtree('H/.claude/projects')\"",
    'python3 -c "import shutil; shutil.rmtree(\'$HOME/.claude/projects\')"',
    "python3 - <<'EOF'\nimport shutil\nshutil.rmtree('H/.claude/projects')\nEOF",
    "python3 - <<'EOF'\nimport shutil, pathlib\nshutil.rmtree(pathlib.Path.home() / '.claude')\nEOF",
    "node -e \"require('fs').rmSync('H/.claude', {recursive: true})\"",
    "perl -e 'use File::Path; rmtree(\"H/.claude\")'",
]

ALLOW = [
    # everyday deletes inside a project or scratch space
    "rm -rf H/Projects/sim/.claude/worktrees/scratch_s06_clone",
    "rm -rf .claude/worktrees/x",
    "rm -rf H/Projects/sim/research/tmp_x",
    "rm -rf /tmp/claude-1000/foo/*",
    'SP=/tmp/claude-1000/sp; for n in a b; do rm -rf "$SP/$n"; done',
    'D=$(mktemp -d); rm -rf "$D"',
    'D=$(mktemp -d); rm -rf "$D"/*',
    'rm -rf "${SCRATCH:?}/x"',
    "cd $(git rev-parse --show-toplevel) && rm -rf build",
    "find . -name '*.pyc' -delete",
    "find . -name '*.tmp' | while read f; do rm \"$f\"; done",
    "git ls-files -z | xargs -0 rm -f",
    "git worktree remove --force H/Projects/sim/.claude/worktrees/x",
    "rsync -a --delete src/ /tmp/dst/",
    "rm ~/notes.txt",
    "rmdir ~/Documents/empty",
    "rm -rf ~/.cache/pip",
    # the memory system's own maintenance: one named note, no -r, no wildcard
    "rm -f H/.claude/projects/-p/memory/note.md",
    "mv H/.claude/projects/-p/memory/note.md H/.claude/projects/-p/memory/note2.md",
    # text that DISCUSSES deleting is data, not a delete
    'git commit -m "incident: rm -rf ~/.claude/projects destroyed the history"',
    "git commit -q -F - <<'EOF'\nrm -rf ~/.claude/projects\nEOF",
    "cat > /tmp/x.md <<'EOF'\nrm -rf ~/.claude\nEOF",
    'echo "rm -rf ~/.claude"',
    "echo '$(rm -rf ~/.claude)'",
    'grep -rn "rm -rf" ~/.claude/projects',
    "ls -la ~/.claude && cat ~/.claude/settings.json",
    "python3 -c \"print('shutil.rmtree is dangerous')\"",
    "python3 - <<'EOF'\nimport os\nos.remove('/tmp/x')\nprint(open('H/Projects/sim/README.md').read())\nEOF",
    "cp -r ~/.claude/projects /tmp/backup",
]


@pytest.mark.parametrize("cmd", BLOCK)
def test_blocks(cmd, home):
    cwd = "H/Projects/sim"
    assert _check(cmd, home, cwd), "should BLOCK: %r" % cmd


@pytest.mark.parametrize("cmd", ALLOW)
def test_allows(cmd, home):
    cwd = "H/Projects/sim"
    assert not _check(cmd, home, cwd), "should ALLOW: %r -> %s" % (cmd, _check(cmd, home, cwd))


def test_relative_protected_name_from_home(home):
    assert _check("rm -rf .claude", home, cwd="H")


def test_a_failed_cd_is_judged_against_the_old_directory_too(home):
    # the review agent's own account: an earlier cd failed and the shell stayed where it was
    assert _check("cd /nonexistent/zzz; rm -rf .claude", home, cwd="H")


def test_worktree_relative_glob_is_fine_but_repo_root_glob_is_not(home):
    assert not _check("rm -rf ./*", home, cwd="H/Projects/sim/.claude/worktrees/x")
    assert _check("rm -rf ./*", home, cwd="H/Projects/sim")


def test_extra_protected_roots_from_env(home):
    reasons = guard.analyze("rm -rf /srv/data/x", "/", home=home, env={"CLAUDE_GUARD_EXTRA_PROTECTED": "/srv/data"})
    assert reasons


@pytest.mark.parametrize("cmd,code", [
    ("rm -rf %(h)s/.claude/projects", 2),
    ("rm -rf %(h)s/Projects/sim/.claude/worktrees/x", 0),
])
def test_hook_process_exit_codes(cmd, code, home):
    payload = {"tool_name": "Bash", "tool_input": {"command": cmd % {"h": home}}, "cwd": home}
    env = dict(os.environ, HOME=home)
    r = subprocess.run([sys.executable, HOOK], input=json.dumps(payload), text=True, capture_output=True, env=env)
    assert r.returncode == code, r.stderr
    if code == 2:
        assert "protected-delete guard" in r.stderr


def test_non_bash_and_malformed_payloads_pass(home):
    for raw in ("not json", json.dumps({"tool_name": "Write", "tool_input": {"file_path": "/x"}})):
        r = subprocess.run([sys.executable, HOOK], input=raw, text=True, capture_output=True)
        assert r.returncode == 0


def test_crude_fallback_catches_the_incident(home):
    assert guard.crude("rm -rf /tmp/a %s/.claude/projects" % home, home=home, env={})
    assert not guard.crude("ls %s/.claude/projects" % home, home=home, env={})
