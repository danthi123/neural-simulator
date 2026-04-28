"""Smoke tests for the webapp/server.py FastAPI app.

Tests the surface without booting a real subprocess — uses FastAPI's
TestClient to hit the in-process app. The launcher endpoint is tested
with a no-op subprocess (unset path).
"""
from __future__ import annotations

import os
import sys

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
