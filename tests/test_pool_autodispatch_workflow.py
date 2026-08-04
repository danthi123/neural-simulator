from __future__ import annotations

import base64
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = ROOT / "tools" / "pool_autodispatch.sh"
WORKFLOW = ROOT / "tools" / "workflow_check.sh"


def run_bash(script: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=True,
    )


def test_remote_wrapper_records_multiline_job_as_one_v2_row(tmp_path: Path) -> None:
    remote_root = tmp_path / "derisk-pool" / "sim"
    remote_root.mkdir(parents=True)
    queue = tmp_path / "pool.queue"
    job = "printf 'alpha\\nbeta\\tgamma\\n'\nexit 7"
    rendered = run_bash(
        DISPATCHER,
        "--render-remote-command",
        job,
        env={"POOL_QUEUE_PATH": str(queue), "HOME": str(tmp_path)},
    ).stdout

    subprocess.run(
        ["bash", "-c", rendered],
        env={**os.environ, "HOME": str(tmp_path)},
        text=True,
        check=True,
    )
    status = remote_root / "job_status.log"
    for _ in range(100):
        if status.exists() and status.read_text():
            break
        time.sleep(0.02)

    rows = status.read_text().splitlines()
    assert len(rows) == 1
    version, epoch, rc, payload = rows[0].split("\t")
    assert version == "v2"
    assert epoch.isdigit()
    assert rc == "7"
    assert base64.b64decode(payload).decode() == job


def test_status_classifier_rejects_malformed_and_stale_rows(tmp_path: Path) -> None:
    now = 2_000_000_000
    recent_job = "pytest -q tests/test_example.py"
    stale_job = "pytest -q tests/test_old.py"
    log = tmp_path / "job_status.log"
    log.write_text(
        "test_name\tand\tnot\n"
        f"v2\t{now - 30}\t4\t{base64.b64encode(recent_job.encode()).decode()}\n"
        f"v2\t{now - 7200}\t2\t{base64.b64encode(stale_job.encode()).decode()}\n"
        "not-a-version\t1\t00:00:00\tbad\n"
    )

    result = run_bash(
        WORKFLOW,
        "--classify-pool-status",
        str(log),
        str(tmp_path),
        str(now),
        "3600",
    )

    assert result.stdout == "C\t4\tpytest -q tests/test_example.py\n"


def test_status_classifier_distinguishes_written_artifact(tmp_path: Path) -> None:
    now = 2_000_000_000
    relative_out = "research/findings/raw/result.json"
    artifact = tmp_path / relative_out
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    job = f"runner --out {relative_out}"
    payload = base64.b64encode(job.encode()).decode()
    log = tmp_path / "job_status.log"
    log.write_text(f"v2\t{now - 5}\t1\t{payload}\n")

    result = run_bash(
        WORKFLOW,
        "--classify-pool-status",
        str(log),
        str(tmp_path),
        str(now),
        "3600",
    )

    assert result.stdout == f"V\t1\t{relative_out}\n"


def test_queue_health_excludes_dispatcher_stale_records(tmp_path: Path) -> None:
    now = 2_000_000_000
    queue = tmp_path / "pool.queue"
    queue.write_text(
        "# staged work\n"
        f"{now - 10}\trecent command #checked:test\n"
        f"{now + 10}\tfuture clock-skew command #checked:test\n"
        f"{now - 50_000}\tstale command #checked:test\n"
        "command-only malformed row\n"
    )

    result = run_bash(
        WORKFLOW,
        "--queue-health",
        str(queue),
        str(now),
        "43200",
    )

    assert result.stdout == "2\t1\t1\t3\n"
