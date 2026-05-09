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


def test_launch_skips_live_mode_flags_for_overridden_runners(client):
    """Regression test 2026-05-07: launches with non-g11_bg_runner runners
    (chat_demo, chat_continual_demo, phase_1_3/4/5, tier_2_3, text_io_*)
    must NOT inject --interactive-control-file or --progress-print-interval.
    Those runners reject these flags as 'unrecognized arguments' and the
    subprocess fails before producing output.

    See: webapp/server.py supports_live_mode = preset not in PRESET_RUNNERS.
    """
    presets_without_live_mode = [
        "chat_demo",
        "chat_continual_demo",
        "chat_synonym_demo",
        "consolidation_synonym",
        "consolidation_synonym_smoke",
        "consolidation_synonym_medium",
        "consolidation_synonym_medium_strict",
        "consolidation_synonym_12word_medium",
        "consolidation_synonym_12word_scaled_medium",
        "consolidation_synonym_16word_scaled_medium",
        "phase_1_4_forgetting",
        "phase_1_3_consolidation",
        "phase_1_5_unified",
        "phase_1_5_unified_scaled",
        "tier_2_3_phrases",
        "text_io_v2_smoke",
    ]
    for preset in presets_without_live_mode:
        res = client.post(
            "/api/runs/launch",
            json={"preset": preset, "seed": 999, "extra_args": []},
        )
        assert res.status_code == 200, f"{preset} launch failed: {res.text}"
        cmd = res.json()["cmd"]
        assert "--interactive-control-file" not in cmd, (
            f"{preset} got --interactive-control-file (runner doesn't accept it)"
        )
        assert "--progress-print-interval" not in cmd, (
            f"{preset} got --progress-print-interval (runner doesn't accept it)"
        )
        # Cleanup: kill the spawned subprocess so the test is hermetic.
        run_id = res.json()["run_id"]
        client.post(f"/api/runs/launch/{run_id}/kill")


def test_launch_keeps_live_mode_flags_for_g11_runners(client):
    """Regression test 2026-05-07: g11_bg_runner presets MUST still receive
    live-mode flags. Smoke + flagship + interactive_* are the canonical
    g11 presets — verifies the gate isn't over-eager."""
    res = client.post(
        "/api/runs/launch",
        json={"preset": "smoke", "seed": 999, "extra_args": []},
    )
    assert res.status_code == 200, res.text
    cmd = res.json()["cmd"]
    assert "--interactive-control-file" in cmd, (
        "smoke (g11_bg_runner preset) lost live-mode flag injection"
    )
    assert "--progress-print-interval" in cmd, (
        "smoke (g11_bg_runner preset) lost progress-print-interval injection"
    )
    run_id = res.json()["run_id"]
    client.post(f"/api/runs/launch/{run_id}/kill")


def test_active_launches_listing(client):
    """Phase 2.5: GET /api/runs/launch lists in-flight runs (empty by default)."""
    res = client.get("/api/runs/launch")
    assert res.status_code == 200
    data = res.json()
    assert "runs" in data
    assert "count" in data


def test_elapsed_sec_freezes_on_done_runs(client):
    """Regression for the 'elapsed counter ticks even after run completes'
    bug: when a run is no longer running but has no finished_at set
    (e.g. drain_log task crashed or never started), the API endpoint must
    lazy-set finished_at so subsequent calls return a stable elapsed_sec.
    """
    import time
    from webapp import server as srv
    # Inject a synthetic LaunchedRun that's already done but missing finished_at.
    run_id = "test_elapsed_freeze_synthetic"
    started = time.time() - 10.0  # 10 seconds ago
    fake = srv.LaunchedRun(
        run_id=run_id, cmd=[], started_at=started,
        proc=None, returncode=0, log_file=None, pid=None,
    )
    assert fake.finished_at is None, "test setup expects finished_at unset"
    srv.launched_runs[run_id] = fake
    try:
        # First call: lazy-sets finished_at. elapsed_sec ~ 10s.
        res1 = client.get("/api/runs/launch")
        assert res1.status_code == 200
        run1 = next(r for r in res1.json()["runs"] if r["run_id"] == run_id)
        elapsed1 = run1["elapsed_sec"]
        assert run1["running"] is False
        # Second call ~50ms later: must return the SAME elapsed_sec (frozen).
        time.sleep(0.05)
        res2 = client.get("/api/runs/launch")
        run2 = next(r for r in res2.json()["runs"] if r["run_id"] == run_id)
        # Allow tiny jitter from log mtime if log_file existed; we set log_file=None
        # so the fallback is time.time(). Both calls should produce the same value
        # because finished_at is now set on the in-memory object.
        assert abs(run2["elapsed_sec"] - elapsed1) < 0.01, (
            f"elapsed_sec ticked: {elapsed1:.3f} -> {run2['elapsed_sec']:.3f}"
        )
        # Same regression on /api/runs/launch/{id}.
        res3 = client.get(f"/api/runs/launch/{run_id}")
        assert res3.status_code == 200
        elapsed3 = res3.json()["elapsed_sec"]
        time.sleep(0.05)
        res4 = client.get(f"/api/runs/launch/{run_id}")
        assert abs(res4.json()["elapsed_sec"] - elapsed3) < 0.01
    finally:
        srv.launched_runs.pop(run_id, None)


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
        "[g11 seed=42] curriculum phase 1: corticostriatal plastic",
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
        # Legacy g11_seed-prefix style
        ("g11_seed42.json", "default"),
        ("g11_seed42_v3lateral.json", "v3lateral"),
        ("g11_seed100_sensedonly.json", "sensedonly"),
        ("g11_seed44_cheat5v2.json", "cheat5v2"),
        ("g11_seed101_2goal_partialfreeze.json", "2goal_partialfreeze"),
        # Modern seed-suffix naming (2026-05-01)
        ("clusterG_Gfv2nmda_seed100.json", "clusterG_Gfv2nmda"),
        ("k_v2_stress_16x16_seed42.json", "k_v2_stress_16x16"),
        ("text_eval_R5_delta_seed42.json", "text_eval_R5_delta"),
        ("no_heuristic_16x16_seed44.json", "no_heuristic_16x16"),
        ("stress_24x24_seed43.json", "stress_24x24"),
        # Smoke test files (no seed)
        ("clusterF_smoke.json", "clusterF_smoke"),
        ("text_eval_smoke.json", "text_eval_smoke"),
        # Truly unknown
        ("not_a_g11_file.json", "(other)"),
        ("random_other_thing.json", "(other)"),
    ]
    for fname, expected in cases:
        actual = _detect_experiment(fname)
        assert actual == expected, f"{fname}: expected {expected!r}, got {actual!r}"


def test_new_presets_exposed(client):
    """2026-05-01: G v2.5 + K v2 + text-io presets must be exposed via API."""
    res = client.get("/api/info")
    assert res.status_code == 200
    presets = res.json()["presets"]
    assert "flagship_g_v25" in presets, "G v2.5 NMDA flagship preset missing"
    assert "flagship_k_v2_visual" in presets, "K v2 visual cortex preset missing"
    assert "flagship_k_v2_24x24" in presets, "K v2 24x24 preset missing"
    assert "interactive_k_v2_visual" in presets, "interactive K v2 preset missing"


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


# ─────────────────────────────────────────────────────────────────────────
# Capability status endpoint (added 2026-05-09)
# ─────────────────────────────────────────────────────────────────────────


def test_capability_status_returns_json(client):
    """The /api/capability-status endpoint should return parseable JSON
    matching the documented shape (headline + pillars + capacity_rule +
    phase_status). Backed by webapp/capability_status.json."""
    res = client.get("/api/capability-status")
    assert res.status_code == 200
    data = res.json()
    # Top-level keys present
    assert "as_of" in data
    assert "headline" in data
    assert "pillars" in data
    assert "capacity_rule" in data
    assert "phase_status" in data


def test_capability_status_headline_shape(client):
    """Headline should have tier + result + finding_doc fields so the UI
    can render the headline card without conditionals on missing keys."""
    res = client.get("/api/capability-status")
    data = res.json()
    headline = data.get("headline")
    if headline is None:
        pytest.skip("no headline configured in capability_status.json")
    assert "tier" in headline
    assert "result" in headline
    assert "finding_doc" in headline


def test_capability_status_pillars_have_status(client):
    """Each pillar should have name + status + metric. Status is one of
    VALIDATED/BOUNDARY/PREDICTED/NEGATIVE so the UI can color-code badges."""
    res = client.get("/api/capability-status")
    data = res.json()
    pillars = data.get("pillars") or []
    assert len(pillars) >= 1, "expect at least one empirical pillar documented"
    valid_statuses = {"VALIDATED", "BOUNDARY", "PREDICTED", "NEGATIVE"}
    for p in pillars:
        assert "name" in p
        assert "status" in p
        assert p["status"] in valid_statuses, f"unknown pillar status: {p['status']}"
        assert "metric" in p


def test_capability_status_capacity_rule_table(client):
    """Capacity rule should have a numerical table the UI can render."""
    res = client.get("/api/capability-status")
    data = res.json()
    rule = data.get("capacity_rule")
    if rule is None:
        pytest.skip("no capacity rule configured")
    assert "rule" in rule
    rows = rule.get("rows") or []
    assert len(rows) >= 1
    for r in rows:
        # Each row needs the columns the UI table renders
        for k in ("vocab", "subpops", "n_motor", "neurons_per_subpop", "status"):
            assert k in r, f"capacity rule row missing key: {k}"


def test_capability_status_phase_status(client):
    """Phase status should at least name the active phase so the UI's
    'Active:' line is never blank."""
    res = client.get("/api/capability-status")
    data = res.json()
    ps = data.get("phase_status")
    if ps is None:
        pytest.skip("no phase_status configured")
    assert "active" in ps and ps["active"], "active phase must be non-empty"


def test_parse_log_progress_continual_eval_partial(tmp_path):
    """continual_eval_suite logs: 2 benchmarks completed, 3rd in flight.

    User-reported bug 2026-05-09: 'It shows 0% · (no progress markers yet)'
    on a Phase 1.5 multi-seed run. The runner emits human-readable
    benchmark markers; _parse_log_progress was only matching navigation-
    runner step lines.
    """
    from webapp.server import _parse_log_progress

    log = tmp_path / "ces.log"
    log.write_text(
        "boot logging...\n"
        "\n--- Running benchmark: sequential_expansion ---\n"
        "  Pre-silence: 53%\n"
        "  Post-silence: 51%\n"
        "  [OK] sequential_expansion: score=0.95 pass=True (1830s)\n"
        "\n--- Running benchmark: retention_over_time ---\n"
        "    Train done (980s)\n"
        "    Pre-silence: 52.0%\n"
        "    Post-silence: 45.0%\n"
        "  [OK] retention_over_time: score=0.87 pass=True (1911s)\n"
        "\n--- Running benchmark: interference ---\n"
        "  [INT] Train interleaved 8-word vocab\n"
        "  ... still running ...\n",
        encoding="utf-8",
    )
    p = _parse_log_progress(log)
    assert p is not None
    assert p["kind"] == "continual_eval"
    assert p["n_completed"] == 2  # sequential_expansion + retention_over_time
    assert p["n_started"] == 3    # +interference now running
    assert p["current_benchmark"] == "interference"
    # Fraction = (2 + 0.5) / 4 = 0.625 (interference is "half done")
    assert abs(p["fraction"] - 0.625) < 0.01
    # Per-benchmark results surfaced for the panel
    names = [r["name"] for r in p["completed_results"]]
    assert names == ["sequential_expansion", "retention_over_time"]
    assert p["completed_results"][1]["score"] == 0.87
    assert p["completed_results"][1]["pass"] is True


def test_parse_log_progress_continual_eval_all_done(tmp_path):
    """Final completion: all 4 benchmarks done, fraction = 1.0."""
    from webapp.server import _parse_log_progress

    log = tmp_path / "ces.log"
    body = []
    for name in ("sequential_expansion", "retention_over_time",
                 "interference", "long_tail"):
        body.append(f"\n--- Running benchmark: {name} ---\n")
        body.append(f"  [OK] {name}: score=0.85 pass=True (1500s)\n")
    log.write_text("".join(body), encoding="utf-8")

    p = _parse_log_progress(log)
    assert p["kind"] == "continual_eval"
    assert p["n_completed"] == 4
    assert p["n_started"] == 4
    assert p["fraction"] == 1.0


def test_parse_log_progress_continual_eval_handles_failed_benchmark(tmp_path):
    """[X] (failed) benchmark counts as a completed benchmark for progress."""
    from webapp.server import _parse_log_progress

    log = tmp_path / "ces.log"
    log.write_text(
        "\n--- Running benchmark: sequential_expansion ---\n"
        "  [OK] sequential_expansion: score=0.95 pass=True (1830s)\n"
        "\n--- Running benchmark: retention_over_time ---\n"
        "  [X] retention_over_time: score=0.65 pass=False (1900s)\n"
        "\n--- Running benchmark: interference ---\n"
        "  ... in flight ...\n",
        encoding="utf-8",
    )
    p = _parse_log_progress(log)
    assert p["kind"] == "continual_eval"
    assert p["n_completed"] == 2
    assert p["n_started"] == 3
    assert p["current_benchmark"] == "interference"
    # Failed benchmark still appears in completed_results with pass=False
    failed = [r for r in p["completed_results"] if not r["pass"]]
    assert len(failed) == 1
    assert failed[0]["name"] == "retention_over_time"


def test_inflight_includes_webapp_launched_runs(client, tmp_path):
    """Bug fix 2026-05-09: /api/inflight must merge launched_runs (POST
    /api/runs/launch) with the PID-file scan, otherwise webapp-launched
    runs are invisible in both the Home in-flight panel and the Runs
    tab "Live runs" panel."""
    from webapp.server import launched_runs, LaunchedRun
    fake_id = "test_inflight_merge_xyz"
    log_path = tmp_path / "fake_run.log"
    log_path.write_text("mock log\n", encoding="utf-8")
    out_path = tmp_path / "fake_run.json"  # doesn't exist yet (still running)
    fake_run = LaunchedRun(
        run_id=fake_id,
        cmd=["python", "-m", "fake.runner"],
        started_at=0.0,
        proc=None,        # no real process
        pid=None,         # treat as not-alive (poll returns False, pid None)
        log_file=str(log_path),
        out_path=str(out_path),
    )
    launched_runs[fake_id] = fake_run
    try:
        res = client.get("/api/inflight")
        assert res.status_code == 200
        data = res.json()
        # Find the entry for our fake run
        webapp_entries = [r for r in data["inflight"]
                          if r.get("source") == "webapp_launch"
                          and r.get("run_id") == fake_id]
        assert len(webapp_entries) == 1, (
            f"webapp-launched run not in /api/inflight; "
            f"got entries: {data['inflight']}")
        entry = webapp_entries[0]
        assert entry["name"] == "fake_run"  # from out_path.stem
        assert entry["log_file"] == "fake_run.log"
        assert entry["completed"] is False
        # alive = False because proc=None and pid=None
        assert entry["alive"] is False
    finally:
        launched_runs.pop(fake_id, None)


def test_inflight_dedup_webapp_run_with_pid_file(client, tmp_path, monkeypatch):
    """If a webapp-launched run somehow also has a PID file with the
    same PID, dedup so the same run isn't rendered twice in the panel."""
    from webapp.server import launched_runs, LaunchedRun
    import webapp.server as srv

    # Stub the PID-file scan dir so we don't pollute real run files
    fake_raw_dir = tmp_path / "raw"
    fake_raw_dir.mkdir()
    fake_pid = 999999  # very unlikely to collide with real processes
    pid_file = fake_raw_dir / "shared_pid_run.pid"
    pid_file.write_text(str(fake_pid))
    monkeypatch.setattr(srv, "RAW_RUNS_DIR", fake_raw_dir)

    fake_id = "dedup_test_id"
    fake_run = LaunchedRun(
        run_id=fake_id,
        cmd=["python", "-m", "fake.runner"],
        started_at=0.0,
        proc=None,
        pid=fake_pid,  # SAME pid as the PID file
        log_file=None,
        out_path=None,
    )
    launched_runs[fake_id] = fake_run
    try:
        res = client.get("/api/inflight")
        data = res.json()
        # Should see exactly ONE entry for this PID (from the PID file scan,
        # NOT the webapp_launch entry — dedup skips the launched_runs side)
        matching = [r for r in data["inflight"] if r.get("pid") == fake_pid]
        assert len(matching) == 1
        # And it should NOT have source=webapp_launch (PID file wins)
        assert matching[0].get("source") != "webapp_launch"
    finally:
        launched_runs.pop(fake_id, None)


def test_capability_status_handles_missing_file(client, monkeypatch, tmp_path):
    """If capability_status.json is missing, the endpoint should return a
    stub with _warning rather than 500-ing — the dashboard should still
    render on a fresh checkout."""
    # Re-import inside the test so we can monkey-patch Path resolution
    import webapp.server as srv
    real_resolve = srv.Path

    # Point the endpoint at a temp dir without the JSON file
    fake_static = tmp_path / "static"
    fake_static.mkdir()
    fake_server_dir = tmp_path
    monkeypatch.setattr(srv, "Path",
                        lambda *a, **k: real_resolve(*a, **k))
    # Easier path: temporarily move the real JSON aside
    real_path = real_resolve(srv.__file__).parent / "capability_status.json"
    backup = None
    if real_path.exists():
        backup = real_path.read_bytes()
        real_path.unlink()
    try:
        res = client.get("/api/capability-status")
        assert res.status_code == 200
        data = res.json()
        assert "_warning" in data
        assert data["headline"] is None
    finally:
        if backup is not None:
            real_path.write_bytes(backup)
