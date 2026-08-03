#!/usr/bin/env python3
"""Check or install the repository-controlled RAG hook workflow."""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

try:
    from .rag_paths import resolve_paths
except ImportError:  # direct script execution
    from rag_paths import resolve_paths


EXPECTED_HOOKS_PATH = "tools/githooks"
EXPECTED_SCHEMA = "repo-relative-v1"
PERIODIC_SERVICE = "sim-rag-autoupdate.service"
PERIODIC_TIMER = "sim-rag-autoupdate.timer"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _user_systemd_dir() -> Path:
    return Path(
        os.environ.get(
            "XDG_CONFIG_HOME", Path.home() / ".config"
        )
    ) / "systemd" / "user"


def _installed_helper() -> Path:
    return Path.home() / ".local" / "libexec" / "sim-rag-autoupdate"


def periodic_status(repo: Path) -> list[tuple[str, bool, str]]:
    """Report whether non-Git catalog changes have a live refresh trigger."""
    unit_dir = _user_systemd_dir()
    helper = _installed_helper()
    systemctl = shutil.which("systemctl")
    files_ready = all(
        (unit_dir / name).is_file() for name in (PERIODIC_SERVICE, PERIODIC_TIMER)
    ) and os.access(helper, os.X_OK)
    repository_helper = repo / "tools" / "rag" / "periodic_update.sh"
    helper_current = bool(
        helper.is_file()
        and repository_helper.is_file()
        and filecmp.cmp(helper, repository_helper, shallow=False)
    )
    enabled = active = False
    detail = "systemctl unavailable"
    if systemctl:
        enabled_result = subprocess.run(
            [systemctl, "--user", "is-enabled", PERIODIC_TIMER],
            capture_output=True,
            text=True,
        )
        active_result = subprocess.run(
            [systemctl, "--user", "is-active", PERIODIC_TIMER],
            capture_output=True,
            text=True,
        )
        enabled = enabled_result.returncode == 0
        active = active_result.returncode == 0
        detail = f"enabled={enabled_result.stdout.strip() or 'no'}, active={active_result.stdout.strip() or 'no'}"
    return [
        ("periodic-refresh-files", files_ready, str(unit_dir)),
        (
            "periodic-refresh-helper-current",
            helper_current,
            f"installed={helper}, repository={repository_helper}",
        ),
        ("periodic-refresh-timer", bool(systemctl) and enabled and active, detail),
    ]


def workflow_status(
    repo: Path, *, include_periodic: bool = False
) -> list[tuple[str, bool, str]]:
    paths = resolve_paths(repo)
    configured = _git(repo, "config", "--get", "core.hooksPath")
    hook_dir = repo / EXPECTED_HOOKS_PATH
    schema_path = paths.rag_root / ".rag_schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8")).get(
            "document_id_schema"
        )
    except Exception:
        schema = None
    checks = [
        (
            "hooks-path",
            configured == EXPECTED_HOOKS_PATH,
            configured or "unset",
        ),
        (
            "pre-commit-executable",
            os.access(hook_dir / "pre-commit", os.X_OK),
            str(hook_dir / "pre-commit"),
        ),
        (
            "post-commit-executable",
            os.access(hook_dir / "post-commit", os.X_OK),
            str(hook_dir / "post-commit"),
        ),
        (
            "engine-python",
            os.access(paths.engine_python, os.X_OK),
            str(paths.engine_python),
        ),
        (
            "rag-python",
            os.access(paths.rag_python, os.X_OK),
            str(paths.rag_python),
        ),
        ("full-index", paths.full_index.is_dir(), str(paths.full_index)),
        ("catalog", paths.catalog.is_dir(), str(paths.catalog)),
        (
            "index-schema",
            schema == EXPECTED_SCHEMA,
            schema or "missing/legacy; run tools/rag/update_indexes.py --rebuild",
        ),
    ]
    if include_periodic:
        checks.extend(periodic_status(repo))
    return checks


def install(repo: Path) -> None:
    _git(repo, "config", "core.hooksPath", EXPECTED_HOOKS_PATH)
    for name in ("pre-commit", "post-commit"):
        path = repo / EXPECTED_HOOKS_PATH / name
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    paths = resolve_paths(repo)
    helper = _installed_helper()
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / "tools" / "rag" / "periodic_update.sh", helper)
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    unit_dir = _user_systemd_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    canonical = str(paths.common_repo).replace("\\", "\\\\").replace('"', '\\"')
    (unit_dir / PERIODIC_SERVICE).write_text(
        "[Unit]\n"
        "Description=Refresh the simulated-brain RAG corpus\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        f'Environment="SIM_CANONICAL_REPO={canonical}"\n'
        f"ExecStart={helper}\n",
        encoding="utf-8",
    )
    (unit_dir / PERIODIC_TIMER).write_text(
        "[Unit]\n"
        "Description=Check the simulated-brain RAG corpus for changes\n\n"
        "[Timer]\n"
        "OnBootSec=2min\n"
        "OnUnitActiveSec=5min\n"
        "RandomizedDelaySec=30s\n"
        "Persistent=true\n"
        f"Unit={PERIODIC_SERVICE}\n\n"
        "[Install]\n"
        "WantedBy=timers.target\n",
        encoding="utf-8",
    )
    systemctl = shutil.which("systemctl")
    if not systemctl:
        raise RuntimeError("systemctl is required to install the periodic RAG refresh")
    subprocess.run([systemctl, "--user", "daemon-reload"], check=True)
    subprocess.run(
        [systemctl, "--user", "enable", "--now", PERIODIC_TIMER], check=True
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if args.install:
        install(repo)
    checks = workflow_status(repo, include_periodic=True)
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'BLOCKED'}] {name}: {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print("RAG_WORKFLOW_BLOCKED: " + ", ".join(failed))
        return 1
    print("RAG_WORKFLOW_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
