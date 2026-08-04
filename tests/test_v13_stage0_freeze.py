from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import v13_stage0_controller as controller
from tools import v13_stage0_freeze as freeze


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_manifest_is_sorted_create_only_and_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    for relative, value in (("b.py", "b\n"), ("a.py", "a\n")):
        (root / relative).write_text(value)
    monkeypatch.setattr(controller, "_required_candidate_source_paths", lambda _root: ("b.py", "a.py"))
    monkeypatch.setattr(freeze, "_head", lambda _root: "a" * 40)
    monkeypatch.setattr(freeze, "_run_git", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        controller, "_revision_file_digest",
        lambda source, revision, relative: _digest(source / relative),
    )
    target = root / "source.sha256"

    result = freeze.freeze_source_manifest(root=root, emit=target)

    assert result["file_count"] == 2
    assert target.read_text().splitlines() == [
        f"{_digest(root / 'a.py')}  a.py",
        f"{_digest(root / 'b.py')}  b.py",
    ]
    with pytest.raises(freeze.FreezeError, match="overwrite"):
        freeze.freeze_source_manifest(root=root, emit=target)


def test_source_manifest_rejects_dirty_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "source.py").write_text("source\n")
    monkeypatch.setattr(controller, "_required_candidate_source_paths", lambda _root: ("source.py",))
    monkeypatch.setattr(freeze, "_run_git", lambda *_args, **_kwargs: " M source.py\n")
    with pytest.raises(freeze.FreezeError, match="dirty or untracked"):
        freeze.freeze_source_manifest(root=tmp_path, emit=tmp_path / "source.sha256")


def test_artifact_destinations_are_explicit_and_unique() -> None:
    assert freeze.CORRECTION_ID == "v13-stage0-process-correction-v6"
    assert freeze.ARTIFACT_ROOT.endswith("stage0_process_correction_v6")
    paths = freeze._artifact_paths()
    assert set(paths) == {
        "calibration_numpy", "calibration_cupy", "calibration_selection",
        "replication_numpy", "replication_cupy", "held_out_cupy",
        "held_out_numpy", "performance_baseline", "performance_candidate",
        "final_stage0",
    }
    assert len(paths) == len(set(paths.values()))
    assert all(path.startswith(freeze.ARTIFACT_ROOT + "/") for path in paths.values())


def test_config_output_summary_does_not_expose_seed_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "candidate_source_revision": "a" * 40,
        "candidate_source_manifest": {"file_count": 12},
        "sha256": "b" * 64,
        "seeds": {"calibration": 123, "replication": 456, "held_out": 789},
    }
    monkeypatch.setattr(freeze, "build_config", lambda **_kwargs: config)
    monkeypatch.setattr(controller, "load_config", lambda *_args, **_kwargs: config)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    source = tmp_path / "source.sha256"
    source.write_text("source\n")
    target = tmp_path / "config.json"

    result = freeze.freeze_config(
        root=tmp_path, source_manifest=source, python=python, emit=target,
    )

    assert json.loads(target.read_text())["seeds"] == config["seeds"]
    rendered = json.dumps(result)
    assert "123" not in rendered and "456" not in rendered and "789" not in rendered
    with pytest.raises(freeze.FreezeError, match="overwrite"):
        freeze.freeze_config(
            root=tmp_path, source_manifest=source, python=python, emit=target,
        )
