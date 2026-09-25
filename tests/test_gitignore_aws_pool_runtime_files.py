"""tools/aws_pool_node.sh + tools/aws_stop_safety_lib.sh + tools/pool_autodispatch.sh's stale-refresh self-heal
write several runtime files into research/queue/ alongside the two already-gitignored ones
(.pool_ssh_config/.pool_extra_nodes). 2026-09-25 review, LOW: "the new runtime files are not gitignored" --
`.pool_ssh_config.bak` in particular is a FULL COPY of the ssh config (IPs + IdentityFile paths), which
.gitignore's own `research/queue/.pool_ssh_config` line deliberately excludes but does not cover.

`git check-ignore` is the actual mechanism (not a text search of .gitignore, which would not catch a pattern
that fails to match) -- each case here is a real path git is asked to classify.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_ignored(rel_path: str) -> bool:
    res = subprocess.run(["git", "check-ignore", "-q", rel_path], cwd=ROOT)
    return res.returncode == 0


def test_pool_ssh_config_backup_is_gitignored():
    assert _is_ignored("research/queue/.pool_ssh_config.bak")


def test_pool_ssh_config_lock_is_gitignored():
    assert _is_ignored("research/queue/.pool_ssh_config.lock")


def test_pool_extra_nodes_tmp_is_gitignored():
    assert _is_ignored("research/queue/.pool_extra_nodes.tmp")


def test_pause_dispatch_extra_nodes_tmp_files_are_gitignored():
    assert _is_ignored("research/queue/.extra_nodes.aB3xY9")


def test_host_block_tmp_files_are_gitignored():
    assert _is_ignored("research/queue/.host_block.aB3xY9")


def test_pool_stale_refresh_mark_dir_is_gitignored():
    assert _is_ignored("research/queue/.pool_stale_refresh/pool1")
