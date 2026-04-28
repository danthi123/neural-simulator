"""Smoke tests for the webapp/server.py FastAPI app.

Tests the surface without booting a real subprocess — uses FastAPI's
TestClient to hit the in-process app. The launcher endpoint is tested
with a no-op subprocess (unset path).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")  # FastAPI's TestClient depends on httpx
    from fastapi.testclient import TestClient
    from webapp.server import app
    return TestClient(app)


def test_info_endpoint(client):
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert "flagship" in data["presets"]
    assert "flagship_with_cheat5" in data["presets"]
    assert data["phase"].startswith("1")


def test_runs_listing(client):
    res = client.get("/api/runs")
    assert res.status_code == 200
    data = res.json()
    assert "runs" in data
    assert "count" in data
    assert isinstance(data["count"], int)
    # If there are runs, each has the summary fields we render.
    for r in data["runs"][:3]:
        assert "name" in r
        assert "n_phases" in r
        assert "final_qs" in r


def test_findings_listing(client):
    res = client.get("/api/findings")
    assert res.status_code == 200
    data = res.json()
    assert "findings" in data
    # We have a lot of findings (60+); expect at least some.
    assert data["count"] > 0


def test_finding_detail_path_traversal_rejected(client):
    """Path-traversal must not return file content. Either the routing
    layer normalizes `..` away (404) or our handler rejects it (400);
    both are safe outcomes."""
    for name in ("..", "..%2Fserver.py", "..%5Cwebapp%5Cserver.py", ".%2E"):
        res = client.get(f"/api/findings/{name}")
        assert res.status_code in (400, 404), (
            f"path-traversal name={name!r} returned {res.status_code}; "
            "must be 400 or 404"
        )


def test_run_detail_path_traversal_rejected(client):
    for name in ("..", "..%2F..%2Fwebapp%2Fserver.py", "%2E%2E"):
        res = client.get(f"/api/runs/{name}")
        assert res.status_code in (400, 404), (
            f"path-traversal name={name!r} returned {res.status_code}; "
            "must be 400 or 404"
        )


def test_finding_detail_not_found(client):
    res = client.get("/api/findings/this-finding-does-not-exist.md")
    assert res.status_code == 404


def test_index_html(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.text
    assert "<title>" in body
    assert "Neural Simulator" in body


def test_static_assets_served(client):
    res = client.get("/static/style.css")
    assert res.status_code == 200
    assert "background" in res.text  # something CSS-like


def test_world_tab_assets_served(client):
    """Phase 2 — world.js loads + index has the World tab markup."""
    res = client.get("/static/world.js")
    assert res.status_code == 200
    assert "setupWorldTab" in res.text  # exported function

    res = client.get("/")
    assert res.status_code == 200
    body = res.text
    assert 'data-tab="world"' in body, "World tab nav button must be present"
    assert 'id="world-canvas"' in body, "World canvas element must be present"


def test_launch_unknown_preset_rejected(client):
    res = client.post("/api/runs/launch", json={"preset": "no_such_preset", "seed": 42})
    assert res.status_code == 400


def test_launch_status_unknown_id(client):
    res = client.get("/api/runs/launch/nonexistent_id")
    assert res.status_code == 404


def test_active_launches_listing(client):
    """Phase 2.5: GET /api/runs/launch lists in-flight runs (empty by default)."""
    res = client.get("/api/runs/launch")
    assert res.status_code == 200
    data = res.json()
    assert "runs" in data
    assert "count" in data


def test_progress_line_parser():
    """Parser converts runner stdout into ProgressEvent."""
    from webapp.server import _try_parse_progress
    line = "[g11 seed=42] step 800/1800  pos=(6,1)  goal=(1,6)  recent_dist=7.58  actions= 21N/ 46E/ 20S/ 13W"
    ev = _try_parse_progress(line, 0.0)
    assert ev is not None
    assert (ev.step, ev.total) == (800, 1800)
    assert ev.pos == (6, 1)
    assert ev.goal == (1, 6)
    assert ev.recent_dist == 7.58


def test_progress_line_parser_rejects_non_progress():
    from webapp.server import _try_parse_progress
    for bad in [
        "",
        "random output",
        "[g11 seed=42] curriculum phase 1: cortex_to_d1 plastic",
        "[g11 seed=42] step 800/1800",  # missing pos/goal
    ]:
        assert _try_parse_progress(bad, 0.0) is None, f"unexpectedly parsed: {bad!r}"


def test_experiments_endpoint_groups_runs(client):
    """/api/experiments groups runs by filename suffix and returns
    per-group aggregates. Must include at least the well-known
    flagship/sensedonly experiment if findings/raw is populated."""
    res = client.get("/api/experiments")
    assert res.status_code == 200
    data = res.json()
    assert "experiments" in data
    assert "count" in data
    assert isinstance(data["experiments"], list)

    # Each experiment row has the expected schema.
    if data["experiments"]:
        e = data["experiments"][0]
        for key in ("experiment", "n_seeds", "n_complete",
                    "mean_sum", "std_sum", "min_sum", "max_sum", "runs"):
            assert key in e, f"missing key {key} in experiment row"
        # n_complete <= n_seeds
        assert e["n_complete"] <= e["n_seeds"]
        # std_sum is None or float
        assert e["std_sum"] is None or isinstance(e["std_sum"], (int, float))


def test_detect_experiment():
    """Filename → experiment-name parsing matches frontend's detectExperiment."""
    from webapp.server import _detect_experiment
    cases = [
        ("g11_seed42.json", "default"),
        ("g11_seed42_v3lateral.json", "v3lateral"),
        ("g11_seed100_sensedonly.json", "sensedonly"),
        ("g11_seed44_cheat5v2.json", "cheat5v2"),
        ("g11_seed101_2goal_partialfreeze.json", "2goal_partialfreeze"),
        ("not_a_g11_file.json", "(other)"),
    ]
    for fname, expected in cases:
        actual = _detect_experiment(fname)
        assert actual == expected, f"{fname}: expected {expected!r}, got {actual!r}"


def test_interactive_presets_exposed(client):
    """The interactive_* presets must be in the available list — they're
    what wire the click-to-control flow in the World tab."""
    res = client.get("/api/info")
    assert res.status_code == 200
    presets = res.json()["presets"]
    assert "interactive_flagship" in presets
    assert "interactive_baseline" in presets


def test_control_endpoint_404_for_unknown_run(client):
    res = client.post("/api/runs/launch/no_such_run/control", json={"paused": True})
    assert res.status_code == 404


def test_control_endpoint_400_for_non_interactive_run(client):
    """Posting control to a non-interactive run must fail cleanly with 400,
    not silently corrupt state. We simulate this by wiring up a fake
    LaunchedRun without a control_file."""
    from webapp.server import LaunchedRun, launched_runs
    fake_id = "test_noninteractive"
    launched_runs[fake_id] = LaunchedRun(
        run_id=fake_id, cmd=[], started_at=0.0, control_file=None,
    )
    try:
        res = client.post(f"/api/runs/launch/{fake_id}/control", json={"paused": True})
        assert res.status_code == 400
    finally:
        launched_runs.pop(fake_id, None)


def test_kill_endpoint_404_for_unknown_run(client):
    res = client.post("/api/runs/launch/no_such_run/kill")
    assert res.status_code == 404


def test_trash_lifecycle_roundtrip(client, monkeypatch, tmp_path):
    """End-to-end trash lifecycle: trash a fake run, see it in /api/runs/trash/list,
    restore it, see it gone from trash and back as a regular run."""
    import webapp.server as srv
    fake_runs = tmp_path / "raw"
    fake_runs.mkdir()
    fake_trash = fake_runs / ".trash"
    fake_trash.mkdir()
    monkeypatch.setattr(srv, "RAW_RUNS_DIR", fake_runs)
    monkeypatch.setattr(srv, "TRASH_DIR", fake_trash)

    # Create a fake run JSON
    sample = fake_runs / "g11_seed42_test.json"
    sample.write_text(json.dumps({
        "seed": 42, "n_steps": 100, "phase_stats": [{
            "phase": 0, "step_start": 0, "step_end": 100, "goal": [6, 6],
            "mean_distance": 1.5, "final_quarter_mean_distance": 1.4,
        }],
    }))

    # Trash it
    res = client.post("/api/runs/trash", json={"names": ["g11_seed42_test.json"]})
    assert res.status_code == 200
    body = res.json()
    assert body["n_trashed"] == 1
    assert "g11_seed42_test.json" in body["trashed"]

    # Confirm gone from regular runs list
    res = client.get("/api/runs")
    names = [r["name"] for r in res.json()["runs"]]
    assert "g11_seed42_test.json" not in names

    # Confirm in trash list
    res = client.get("/api/runs/trash/list")
    assert res.status_code == 200
    trashed = res.json()["trashed"]
    assert len(trashed) == 1
    trash_filename = trashed[0]["trash_filename"]
    assert trashed[0]["original_name"] == "g11_seed42_test.json"

    # Restore it
    res = client.post("/api/runs/trash/restore", json={"trash_filenames": [trash_filename]})
    assert res.status_code == 200
    assert res.json()["n_restored"] == 1

    # Back in regular runs list
    res = client.get("/api/runs")
    names = [r["name"] for r in res.json()["runs"]]
    assert "g11_seed42_test.json" in names

    # Trash list is empty
    res = client.get("/api/runs/trash/list")
    assert res.json()["count"] == 0


def test_trash_purge_permanently_deletes(client, monkeypatch, tmp_path):
    """After purge, the trashed file is gone from disk."""
    import webapp.server as srv
    fake_runs = tmp_path / "raw"
    fake_runs.mkdir()
    fake_trash = fake_runs / ".trash"
    fake_trash.mkdir()
    monkeypatch.setattr(srv, "RAW_RUNS_DIR", fake_runs)
    monkeypatch.setattr(srv, "TRASH_DIR", fake_trash)

    (fake_runs / "g11_seed99_test.json").write_text("{}")

    # Trash and capture name
    client.post("/api/runs/trash", json={"names": ["g11_seed99_test.json"]})
    trashed = client.get("/api/runs/trash/list").json()["trashed"]
    assert len(trashed) == 1
    trash_filename = trashed[0]["trash_filename"]

    # Purge
    res = client.post("/api/runs/trash/purge", json={"trash_filenames": [trash_filename]})
    assert res.status_code == 200
    assert res.json()["n_purged"] == 1

    # Disk: file is actually gone
    assert not (fake_trash / trash_filename).exists()
    assert not (fake_runs / "g11_seed99_test.json").exists()


def test_trash_incomplete_picks_up_empty_phase_stats(client, monkeypatch, tmp_path):
    """trash/incomplete should mass-trash runs without complete phase_stats."""
    import webapp.server as srv
    fake_runs = tmp_path / "raw"
    fake_runs.mkdir()
    fake_trash = fake_runs / ".trash"
    fake_trash.mkdir()
    monkeypatch.setattr(srv, "RAW_RUNS_DIR", fake_runs)
    monkeypatch.setattr(srv, "TRASH_DIR", fake_trash)

    # Two complete + two incomplete + one malformed
    (fake_runs / "complete_a.json").write_text(json.dumps({
        "seed": 1, "phase_stats": [{"final_quarter_mean_distance": 2.0}],
    }))
    (fake_runs / "complete_b.json").write_text(json.dumps({
        "seed": 2, "phase_stats": [{"final_quarter_mean_distance": 1.5}],
    }))
    (fake_runs / "incomplete_a.json").write_text(json.dumps({
        "seed": 3, "phase_stats": [],
    }))
    (fake_runs / "incomplete_b.json").write_text(json.dumps({
        "seed": 4, "phase_stats": [{"goal": [1, 2]}],  # no finalQ
    }))
    (fake_runs / "malformed.json").write_text("not json {")

    res = client.post("/api/runs/trash/incomplete")
    assert res.status_code == 200
    trashed = res.json()["trashed"]
    assert "incomplete_a.json" in trashed
    assert "incomplete_b.json" in trashed
    assert "malformed.json" in trashed
    assert "complete_a.json" not in trashed
    assert "complete_b.json" not in trashed


def test_kill_endpoint_already_done(client):
    """Killing a run that has no live process returns 200 with status
    'already_done', not an error. Lets the UI safely call kill on stale rows."""
    from webapp.server import LaunchedRun, launched_runs
    fake_id = "test_killed_already"
    launched_runs[fake_id] = LaunchedRun(
        run_id=fake_id, cmd=[], started_at=0.0, proc=None, returncode=0,
    )
    try:
        res = client.post(f"/api/runs/launch/{fake_id}/kill")
        assert res.status_code == 200
        assert res.json()["status"] == "already_done"
    finally:
        launched_runs.pop(fake_id, None)


def test_sidecar_404_when_missing(client):
    """Re-run sidecar lookup returns 404 when sidecar file doesn't exist."""
    res = client.get("/api/runs/this_run_has_no_sidecar.json/sidecar")
    assert res.status_code == 404


def test_sidecar_path_traversal_blocked(client):
    res = client.get("/api/runs/..%2Fserver.py/sidecar")
    assert res.status_code in (400, 404)


def test_launch_returns_200_not_500(client, monkeypatch):
    """Regression for: launch_run was sync `def`, dispatched to a worker
    thread with no running event loop, asyncio.create_task raised
    RuntimeError, endpoint returned 500 — but the subprocess HAD already
    been spawned (orphaning it). Bug existed since Phase 1; uncaught
    because earlier tests only checked endpoint structure, never an actual
    POST that gets past Popen.

    This test mocks subprocess.Popen + asyncio.create_task so it doesn't
    touch the GPU but DOES go through the full request-handler code path
    where the bug lived."""
    from unittest.mock import MagicMock
    import webapp.server as srv

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.return_value = b""
    fake_proc.terminate = MagicMock()
    fake_proc.kill = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)
    fake_proc.returncode = None
    # Real int pid required so the sidecar JSON serialization works.
    fake_proc.pid = 12345

    monkeypatch.setattr(srv.subprocess, "Popen", lambda *a, **kw: fake_proc)
    # asyncio.create_task needs a running loop. The TestClient does run a
    # loop, so we don't replace it — but we DO want to verify our handler
    # is async so the loop is reachable from inside it.
    def fake_create_task(coro):
        # Close the coroutine to suppress the unawaited-coroutine warning,
        # then return a mock task. We're not actually streaming stdout here.
        coro.close()
        return MagicMock()

    monkeypatch.setattr(srv.asyncio, "create_task", fake_create_task)

    try:
        res = client.post("/api/runs/launch", json={
            "preset": "smoke", "seed": 42, "extra_args": [],
        })
        assert res.status_code == 200, (
            f"launch returned {res.status_code} (probably the sync-def regression "
            f"is back). Body: {res.text[:300]}"
        )
        data = res.json()
        assert "run_id" in data
        assert "ws_url" in data
        # The subprocess constructor was called once with the runner cmd.
        assert fake_proc.terminate.call_count == 0  # not yet
    finally:
        # Cleanup: drop the fake from launched_runs
        for rid in list(srv.launched_runs.keys()):
            if srv.launched_runs[rid].proc is fake_proc:
                srv.launched_runs.pop(rid)


def test_launch_writes_sidecar_for_rerun(client, monkeypatch, tmp_path):
    """Verify the sidecar `.cmd.json` is written on launch so the Re-run
    button works for runs originating from this dashboard."""
    from unittest.mock import MagicMock
    import webapp.server as srv

    # Redirect run output dir to tmp so we don't pollute findings/raw
    fake_runs_dir = tmp_path / "raw"
    fake_runs_dir.mkdir()
    monkeypatch.setattr(srv, "RAW_RUNS_DIR", fake_runs_dir)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.stdout = MagicMock()
    fake_proc.pid = 54321  # real int for JSON-serializable sidecar
    monkeypatch.setattr(srv.subprocess, "Popen", lambda *a, **kw: fake_proc)
    monkeypatch.setattr(srv.asyncio, "create_task", lambda coro: (coro.close(), MagicMock())[1])

    res = client.post("/api/runs/launch", json={
        "preset": "smoke", "seed": 99, "extra_args": ["--my-flag"],
    })
    assert res.status_code == 200
    body = res.json()
    out_path = Path(body["out_path"])
    sidecar = out_path.with_suffix(".cmd.json")
    assert sidecar.exists(), f"sidecar should be written next to {out_path}"
    sidecar_data = json.loads(sidecar.read_text())
    assert sidecar_data["preset"] == "smoke"
    assert sidecar_data["seed"] == 99
    assert "--my-flag" in sidecar_data["extra_args"]

    # Cleanup
    for rid in list(srv.launched_runs.keys()):
        if srv.launched_runs[rid].proc is fake_proc:
            srv.launched_runs.pop(rid)


def test_control_endpoint_writes_state(client, tmp_path):
    """When a run IS interactive, posting control writes to the control file
    and returns the merged state."""
    from webapp.server import LaunchedRun, launched_runs
    cf = tmp_path / "ctrl.json"
    cf.write_text("{}")
    fake_id = "test_interactive"
    launched_runs[fake_id] = LaunchedRun(
        run_id=fake_id, cmd=[], started_at=0.0, control_file=str(cf),
    )
    try:
        # First update — set goal
        res = client.post(f"/api/runs/launch/{fake_id}/control",
                          json={"goal": [3, 5]})
        assert res.status_code == 200
        body = res.json()
        assert body["state"]["goal"] == [3, 5]

        # Second update — pause, but goal should be preserved (merge semantics)
        res = client.post(f"/api/runs/launch/{fake_id}/control",
                          json={"paused": True})
        assert res.status_code == 200
        merged = res.json()["state"]
        assert merged["paused"] is True
        assert merged["goal"] == [3, 5]

        # File should match
        on_disk = json.loads(cf.read_text())
        assert on_disk == merged
    finally:
        launched_runs.pop(fake_id, None)
