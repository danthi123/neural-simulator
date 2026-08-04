from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/pool_regression_bundle_v1.sh"


def _stage_gitless_copy(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "pool-copy"
    (root / "tools").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "tools/pool_regression_bundle_v1.sh")
    log = root / "python-calls.jsonl"
    fake_python = root / ".venv/bin/python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

log = Path(os.environ["POOL_BUNDLE_TEST_LOG"])
record = {
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "env": {
        key: os.environ.get(key)
        for key in (
            "LC_ALL",
            "TZ",
            "PYTHONHASHSEED",
            "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "SIM_BACKEND",
        )
    },
}
with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\\n")
fail_on = os.environ.get("POOL_BUNDLE_TEST_FAIL_ON")
if fail_on and fail_on in " ".join(sys.argv[1:]):
    raise SystemExit(17)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    assert not (root / ".git").exists()
    return root, log


def _run(root: Path, log: Path, *args: str, fail_on: str = "") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["POOL_BUNDLE_TEST_LOG"] = str(log)
    env["POOL_BUNDLE_TEST_FAIL_ON"] = fail_on
    return subprocess.run(
        ["bash", str(root / "tools/pool_regression_bundle_v1.sh"), *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _calls(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_runs_fixed_read_only_cpu_checks_in_gitless_pool_copy(tmp_path: Path) -> None:
    root, log = _stage_gitless_copy(tmp_path)

    result = _run(root, log)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = _calls(log)
    assert calls[0]["argv"] == ["tools/check_docs.py"]
    assert calls[1]["argv"] == [
        "-m",
        "pytest",
        "-q",
        "-x",
        "-p",
        "no:cacheprovider",
        "tests/test_pool_regression_bundle_v1.py",
        "tests/test_experiment_automation_lifecycle.py",
        "tests/test_doc_rules.py",
        "tests/test_v13_stage0_controller.py",
        "tests/test_v13_stage0_manifest.py",
    ]
    assert all(call["cwd"] == str(root) for call in calls)
    assert all(
        call["env"]
        == {
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONHASHSEED": "0",
            "SIM_BACKEND": "numpy",
            "TZ": "UTC",
        }
        for call in calls
    )
    assert not (root / "research").exists()
    assert not (root / ".pytest_cache").exists()
    assert "[pool-regression-v1] PASS" in result.stdout


def test_rejects_every_argument_before_invoking_python(tmp_path: Path) -> None:
    root, log = _stage_gitless_copy(tmp_path)

    result = _run(root, log, "--seed", "123")

    assert result.returncode == 2
    assert "accepts no arguments or seeds" in result.stderr
    assert _calls(log) == []


def test_fails_fast_when_document_check_fails(tmp_path: Path) -> None:
    root, log = _stage_gitless_copy(tmp_path)

    result = _run(root, log, fail_on="tools/check_docs.py")

    assert result.returncode == 17
    assert [call["argv"] for call in _calls(log)] == [["tools/check_docs.py"]]
    assert "[pool-regression-v1] PASS" not in result.stdout


def test_requires_the_deployed_pool_virtual_environment(tmp_path: Path) -> None:
    root, log = _stage_gitless_copy(tmp_path)
    (root / ".venv/bin/python").unlink()

    result = _run(root, log)

    assert result.returncode == 1
    assert "missing deployed interpreter" in result.stderr
    assert _calls(log) == []
