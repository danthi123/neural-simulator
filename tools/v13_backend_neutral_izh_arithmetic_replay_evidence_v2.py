#!/usr/bin/env python3
"""Freeze and receipt V13 strict-arithmetic replay v2 evidence."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from research.runners import _v13_backend_neutral_izh_arithmetic_replay_v2 as replay
from tools import v13_backend_neutral_izh_arithmetic_replay_evidence as common


ROOT = common.ROOT
ACTIONS = common.ACTIONS
OUTPUT_DIR = replay.OUTPUT_DIRECTORY
SCHEMA = "v13-backend-neutral-izh-arithmetic-replay-command-v2"
FINAL_MANIFEST_SCHEMA = (
    "v13-backend-neutral-izh-arithmetic-replay-evidence-manifest-v2"
)
PROTOCOL = common.EvidenceProtocol(
    replay_protocol=replay.PROTOCOL,
    command_schema=SCHEMA,
    final_manifest_schema=FINAL_MANIFEST_SCHEMA,
    output_directory=OUTPUT_DIR,
)
EvidenceError = common.EvidenceError
execution_receipt = common.execution_receipt
shared = common.shared


def freeze_source_manifest(
    *, root: Path, revision: str, out: str | Path,
) -> dict[str, Any]:
    return common.freeze_source_manifest(
        root=root, revision=revision, out=out, protocol=PROTOCOL,
    )


def _paths() -> dict[str, str]:
    return common._paths(PROTOCOL)


def _inner_command(
    *, root: Path, action: str, revision: str, paths: dict[str, str], python: str,
) -> list[str]:
    return common._inner_command(
        root=root, action=action, revision=revision, paths=paths, python=python,
        protocol=PROTOCOL,
    )


def emit_command(
    *, root: Path, action: str, revision: str, host: str, device: str,
    out: str | Path, python: str = sys.executable,
) -> dict[str, Any]:
    return common.emit_command(
        root=root, action=action, revision=revision, host=host, device=device,
        out=out, python=python, protocol=PROTOCOL,
    )


def _expected_comparison_argv(
    *, root: Path, artifact_file: Path, artifact: dict[str, Any], python: str,
) -> list[str]:
    return common._expected_comparison_argv(
        root=root, artifact_file=artifact_file, artifact=artifact, python=python,
        protocol=PROTOCOL,
    )


def finalize_evidence(
    *, root: Path, artifact_path: str | Path, receipt_path: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    return common.finalize_evidence(
        root=root, artifact_path=artifact_path, receipt_path=receipt_path, out=out,
        protocol=PROTOCOL,
    )


def main(argv: list[str] | None = None) -> int:
    return common.main(argv, protocol=PROTOCOL)


if __name__ == "__main__":
    raise SystemExit(main())
