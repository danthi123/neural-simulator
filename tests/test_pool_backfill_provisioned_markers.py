"""Tests for tools/pool_backfill_provisioned_markers.sh (2026-09-23 fix round #3).

THE DEFECT this migrates: pool_autodispatch.sh's revision_available() started requiring a `.provisioned_ok`
completion marker (fix round #2) that only a NEW pool_provision.sh --isolated run writes. Every revision already
on pool40/41/42 lacks it, so once that check merges to main, every job pinned to one of those revisions is
skipped on every node forever. This script backfills the marker onto a legacy revision directory, but ONLY after
independently re-verifying it with the same checks pool_provision.sh itself ends a successful provision with.

NO real ssh/AWS is used: a stubbed `ssh` on PATH executes the script's remote payload locally (via `bash -c`)
against a fake $HOME standing in for the "node" -- the payload is pure bash/python doing local file checks, so
this exercises the REAL remote script byte-for-byte, not a re-typed copy of its logic.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

from tools.pool.provisioning.source_manifest import write_manifest

ROOT = Path(__file__).resolve().parents[1]
BACKFILL = ROOT / "tools" / "pool_backfill_provisioned_markers.sh"
MARKER_SOURCE = ROOT / "tools" / "pool" / "provisioning" / "source_manifest.py"


def _make_ssh_stub_running_locally(tmp_path: Path, fake_home: Path):
    """A stub `ssh` that runs the LAST argv element (the remote command string, e.g. "bash -s -- '0'") via a
    real `bash -c`, with $HOME overridden to `fake_home` and stdin inherited (so the caller's heredoc -- the
    script's own REMOTE_BACKFILL_SCRIPT -- reaches the innermost `bash -s`) -- i.e. it actually RUNS the real
    remote payload, just against a local directory standing in for the node's home, instead of a real ssh
    session. `-F`/`-o` options and the node name (all but the last argv element) are ignored, matching what a
    real ssh call would do with them (routing/auth), never touched by the payload itself."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "ssh.log"
    log.write_text("")
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
cmd="${{@: -1}}"
HOME="{fake_home}" bash -c "$cmd"
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def _build_revision_dir(rev_dir: Path, *, imports_ok: bool, tamper_after_manifest: bool = False) -> None:
    """A minimal, self-contained fake revision directory: a couple of small "source" files, a REAL manifest
    (via the actual source_manifest.py, never re-typed) covering them, and a stub `.venv/bin/python` that fakes
    ONLY the numpy/scipy/sim/webapp import check (`-c ...`) -- every other invocation (the source_manifest.py
    verify subcommand itself) execs the REAL interpreter running this test, so that check is genuinely exercised,
    not stubbed away."""
    (rev_dir / "fake_src").mkdir(parents=True)
    (rev_dir / "fake_src" / "hello.py").write_text("VALUE = 1\n", encoding="ascii")
    (rev_dir / "tools" / "pool" / "provisioning").mkdir(parents=True)
    (rev_dir / "tools" / "pool" / "provisioning" / "source_manifest.py").write_text(
        MARKER_SOURCE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest_path = rev_dir / ".source_manifest.sha256"
    write_manifest(rev_dir, manifest_path)

    if tamper_after_manifest:
        (rev_dir / "fake_src" / "hello.py").write_text("VALUE = 2  # tampered after manifest\n", encoding="ascii")

    venv_bin = rev_dir / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    python_stub = venv_bin / "python"
    ok_marker = ".fake_imports_ok"
    if imports_ok:
        (rev_dir / ok_marker).write_text("", encoding="ascii")
    python_stub.write_text(f"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then
  [ -f "{ok_marker}" ] && exit 0 || exit 1
fi
exec "{sys.executable}" "$@"
""")
    python_stub.chmod(python_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(args, bin_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
           "POOL_SSH_CONFIG": str(bin_dir / "does-not-exist")}
    return subprocess.run(["bash", str(BACKFILL), *args], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=60)


def test_backfill_marks_a_legacy_revision_that_passes_every_check(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    good = fake_home / "derisk-pool" / "revisions" / "good1234"
    _build_revision_dir(good, imports_ok=True)
    bin_dir, ssh_log = _make_ssh_stub_running_locally(tmp_path, fake_home)

    res = _run(["testnode"], bin_dir)
    assert res.returncode == 0, res.stderr
    assert (good / ".provisioned_ok").exists(), res.stdout + res.stderr
    assert "MARKED good1234" in res.stdout


def test_backfill_never_marks_a_revision_that_fails_the_import_check(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    bad = fake_home / "derisk-pool" / "revisions" / "badimport1"
    _build_revision_dir(bad, imports_ok=False)
    bin_dir, ssh_log = _make_ssh_stub_running_locally(tmp_path, fake_home)

    res = _run(["testnode"], bin_dir)
    assert res.returncode == 0, res.stderr
    assert not (bad / ".provisioned_ok").exists()
    assert "SKIP badimport1" in res.stdout
    assert "import check failed" in res.stdout


def test_backfill_never_marks_a_revision_whose_files_no_longer_match_its_manifest(tmp_path: Path) -> None:
    # Verification must be GENUINE, not just "a manifest file exists" -- a revision whose content was modified
    # (or partially synced) AFTER its manifest was written must not be marked ready to dispatch to.
    fake_home = tmp_path / "home"
    tampered = fake_home / "derisk-pool" / "revisions" / "tampered1"
    _build_revision_dir(tampered, imports_ok=True, tamper_after_manifest=True)
    bin_dir, ssh_log = _make_ssh_stub_running_locally(tmp_path, fake_home)

    res = _run(["testnode"], bin_dir)
    assert res.returncode == 0, res.stderr
    assert not (tampered / ".provisioned_ok").exists()
    assert "SKIP tampered1" in res.stdout
    assert "source file verify" in res.stdout


def test_backfill_leaves_an_already_marked_revision_untouched(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    already = fake_home / "derisk-pool" / "revisions" / "already1"
    _build_revision_dir(already, imports_ok=True)
    (already / ".provisioned_ok").write_text("", encoding="ascii")
    mtime_before = (already / ".provisioned_ok").stat().st_mtime
    bin_dir, ssh_log = _make_ssh_stub_running_locally(tmp_path, fake_home)

    res = _run(["testnode"], bin_dir)
    assert res.returncode == 0, res.stderr
    assert (already / ".provisioned_ok").stat().st_mtime == mtime_before
    assert "already1" not in res.stdout   # never even mentioned -- skipped before any check ran


def test_backfill_dry_run_reports_without_writing_the_marker(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    good = fake_home / "derisk-pool" / "revisions" / "good1234"
    _build_revision_dir(good, imports_ok=True)
    bin_dir, ssh_log = _make_ssh_stub_running_locally(tmp_path, fake_home)

    res = _run(["--dry-run", "testnode"], bin_dir)
    assert res.returncode == 0, res.stderr
    assert not (good / ".provisioned_ok").exists()
    assert "WOULD-MARK good1234" in res.stdout


def test_backfill_mirrors_the_dispatcher_marker_predicate(tmp_path: Path) -> None:
    # The written marker must be exactly what tools/pool_revision_marker.sh's revision_marker_probe_cmd (the ONE
    # predicate both pool_autodispatch.sh's revision_available() and pool_queue.sh's `add` gate call) checks for
    # -- proving this script actually closes the gap, not just writes A file.
    fake_home = tmp_path / "home"
    good = fake_home / "derisk-pool" / "revisions" / "good1234"
    _build_revision_dir(good, imports_ok=True)
    bin_dir, ssh_log = _make_ssh_stub_running_locally(tmp_path, fake_home)
    _run(["testnode"], bin_dir)

    marker_check = subprocess.run(
        ["bash", "-c",
         f'source {ROOT}/tools/pool_revision_marker.sh; '
         f'eval "$(revision_marker_probe_cmd derisk-pool/revisions/good1234)"'],
        env={**os.environ, "HOME": str(fake_home)}, capture_output=True, text=True,
    )
    assert marker_check.returncode == 0, marker_check.stderr
