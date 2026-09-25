"""Subprocess-level tests for tools/status.sh -- the single read-only handoff-status snapshot (see
research/coordination/LOCAL_LLM_RUNBOOK.md). Runs the REAL script against stubbed `aws`/`systemctl`/`ssh`
binaries and isolated GPU/pool queue dirs (never the shared production queue), so this never touches the real
GPU queue, the real pool nodes, or a real AWS account. `tools/gpu_queue.sh`/`tools/pool_queue.sh` are run as
themselves (not stubbed) against the isolated dirs, exactly like tests/test_gpu_queue_status_exit.py already
does -- real nvidia-smi is used if present (a read-only query; this sandbox has one), matching that test's
convention.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "status.sh"


def _make_stub(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _bin_dir(tmp_path: Path, *, llm_active: bool = False, aws_json: str | None = None) -> Path:
    """A stub-binary dir: `aws` answers describe-instances with `aws_json` (default: no instances), `systemctl`
    reports local-llm active/inactive per `llm_active` (still via the REAL is-active exit-code convention: 0 for
    active, 3 for inactive -- this is what caught the real double-"inactive"-line bug during development)."""
    d = tmp_path / "stubbin"
    d.mkdir(exist_ok=True)
    aws_json = aws_json if aws_json is not None else '{"Reservations": []}'
    _make_stub(d / "aws", f"""
if [ "$1" = "ec2" ] && [ "$2" = "describe-instances" ]; then
  echo '{aws_json}'
  exit 0
fi
exit 1
""")
    if llm_active:
        _make_stub(d / "systemctl", 'if [ "$2" = "is-active" ]; then echo active; exit 0; fi; exit 1')
    else:
        _make_stub(d / "systemctl", 'if [ "$2" = "is-active" ]; then echo inactive; exit 3; fi; exit 1')
    return d


def _unreachable_ssh(tmp_path: Path) -> Path:
    p = tmp_path / "fake_ssh"
    _make_stub(p, 'exit 255')  # every node "unreachable" -- no real network, deterministic
    return p


def _run_status(tmp_path: Path, *, llm_active: bool = False, aws_json: str | None = None,
                 ssh_bin: Path | None = None, extra_env: dict | None = None):
    stub_dir = _bin_dir(tmp_path, llm_active=llm_active, aws_json=aws_json)
    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}:{env.get('PATH', '')}"
    env["GPU_QUEUE_DIR"] = str(tmp_path / "gpuq")
    env["POOL_QUEUE_PATH"] = str(tmp_path / "pool.queue")
    (tmp_path / "pool.queue").touch()
    env["POOL_SSH_CONFIG"] = str(tmp_path / "no_such_ssh_config")  # -> no -F flag added
    env["POOL_NODES"] = "fakenode1 fakenode2"
    env["STATUS_SSH"] = str(ssh_bin or _unreachable_ssh(tmp_path))
    env["AWS_BUDGET_LOG"] = str(tmp_path / "aws_budget.log")
    env["AWS_SPEND_LEDGER"] = str(tmp_path / "aws_ledger.json")
    env["HANDOFF_BATTERIES_TSV"] = str(tmp_path / "batteries.tsv")
    # ABSOLUTE glob (os.path.join(root, abs_glob) == abs_glob, ignoring --root): status.sh always resolves
    # --root to its OWN script location (the real repo checkout), not this test's cwd, so a repo-relative glob
    # here would count files in the real repo instead of this test's isolated tmp_path/raw/ tree.
    (tmp_path / "batteries.tsv").write_text(
        "name\traw_glob\texpected_rows\tharvest_cmd\tfinding_template\n"
        f"fi\t{tmp_path}/raw/_fi/**/*.json\t2\techo hi\ttmpl.md\n",
        encoding="utf-8",
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(SCRIPT)], cwd=str(tmp_path), env=env,
                           capture_output=True, text=True, timeout=60)


def test_status_exits_zero_even_when_it_has_a_failure_to_report(tmp_path):
    """Pins the exact bug found + fixed while building this script: a bare `[ "$ANY" -eq 0 ] && echo ...` as the
    LAST statement returns the TEST's own exit status (1) whenever there WAS something to report (ANY=1) -- so
    the more informative the report, the more likely a caller sees this read-only script as "failed". This must
    stay non-vacuous: it fabricates a real recent GPU failure so ANY=1 is actually exercised (mutating the
    trailing `if ...; fi; exit 0` back to `[ "$ANY" -eq 0 ] && echo ...` makes this fail, confirmed by hand)."""
    import datetime

    gpuq = tmp_path / "gpuq"
    gpuq.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%F %T")
    (gpuq / "gpu_queue.log").write_text(f"{ts} DONE(rc=1): some fake failing job\n", encoding="utf-8")
    res = _run_status(tmp_path, aws_json=json.dumps({
        "Reservations": [{"Instances": [{"InstanceId": "i-fake0001", "InstanceType": "t3.micro",
                                          "State": {"Name": "running"},
                                          "LaunchTime": "2026-09-25T00:00:00+00:00",
                                          "Tags": [{"Key": "Project", "Value": "neural-simulator"}]}]}]
    }))
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "DONE(rc=1): some fake failing job" in res.stdout


def test_status_output_is_at_most_40_lines(tmp_path):
    res = _run_status(tmp_path)
    assert res.returncode == 0, res.stderr
    lines = res.stdout.rstrip("\n").split("\n")
    assert len(lines) <= 40, f"status.sh printed {len(lines)} lines (budget is 40):\n{res.stdout}"


def test_status_never_writes_to_the_repo(tmp_path):
    """Read-only means read-only: nothing new appears in the checkout `status.sh` was run from."""
    before = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=30).stdout
    _run_status(tmp_path)
    after = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=30).stdout
    assert before == after


def test_status_reports_all_pool_nodes_unreachable(tmp_path):
    res = _run_status(tmp_path)
    assert res.returncode == 0, res.stderr
    assert "fakenode1: unreachable" in res.stdout
    assert "fakenode2: unreachable" in res.stdout


def test_status_local_llm_active_prints_exactly_one_line_no_duplicate(tmp_path):
    """Regression for the real bug caught manually: `cmd || echo inactive` doubled the line because
    `systemctl --user is-active` prints a state word on BOTH a zero and non-zero exit."""
    res = _run_status(tmp_path, llm_active=True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.count("local-llm: active") == 1
    assert "inactive" not in res.stdout.split("-- local-llm --")[1].split("-- Batteries --")[0]


def test_status_local_llm_inactive_prints_exactly_one_line_no_duplicate(tmp_path):
    res = _run_status(tmp_path, llm_active=False)
    assert res.returncode == 0, res.stderr
    section = res.stdout.split("-- local-llm --")[1].split("-- Batteries --")[0]
    assert section.count("inactive") == 1


def test_status_shows_battery_waiting_below_expected_rows(tmp_path):
    (tmp_path / "raw" / "_fi").mkdir(parents=True)
    (tmp_path / "raw" / "_fi" / "s1.json").write_text("{}", encoding="utf-8")
    res = _run_status(tmp_path)
    assert res.returncode == 0, res.stderr
    assert "fi" in res.stdout and "1/2" in res.stdout and "WAITING" in res.stdout


def test_status_shows_battery_ready_when_rows_complete(tmp_path):
    (tmp_path / "raw" / "_fi").mkdir(parents=True)
    (tmp_path / "raw" / "_fi" / "s1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "raw" / "_fi" / "s2.json").write_text("{}", encoding="utf-8")
    res = _run_status(tmp_path)
    assert res.returncode == 0, res.stderr
    assert "1/2" not in res.stdout
    assert "fi" in res.stdout and "2/2" in res.stdout and "READY" in res.stdout


def test_status_recent_failures_section_reports_none_when_clean(tmp_path):
    res = _run_status(tmp_path)
    assert res.returncode == 0, res.stderr
    section = res.stdout.split("-- Recent failures (24h) --")[1]
    assert "(none)" in section


def test_status_recent_failures_section_surfaces_a_recent_gpu_failure(tmp_path):
    """The GPU log line format is `%F %T DONE(rc=<n>): <job>`, written by tools/gpu_queue.sh's daemon loop --
    reproduced verbatim here rather than re-deriving a fresh format for the test fixture."""
    import datetime

    gpuq = tmp_path / "gpuq"
    gpuq.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    ts = now.strftime("%F %T")
    (gpuq / "gpu_queue.log").write_text(f"{ts} DONE(rc=1): some fake failing job\n", encoding="utf-8")
    res = _run_status(tmp_path)
    assert res.returncode == 0, res.stderr
    assert "DONE(rc=1): some fake failing job" in res.stdout


def test_status_ignores_a_gpu_failure_older_than_24h(tmp_path):
    import datetime

    gpuq = tmp_path / "gpuq"
    gpuq.mkdir(parents=True, exist_ok=True)
    old = datetime.datetime.now() - datetime.timedelta(days=3)
    ts = old.strftime("%F %T")
    (gpuq / "gpu_queue.log").write_text(f"{ts} DONE(rc=1): a stale failing job from days ago\n", encoding="utf-8")
    res = _run_status(tmp_path)
    assert res.returncode == 0, res.stderr
    assert "a stale failing job from days ago" not in res.stdout
    section = res.stdout.split("-- Recent failures (24h) --")[1]
    assert "(none)" in section
