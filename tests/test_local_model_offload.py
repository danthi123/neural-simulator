"""Fail-closed contract tests for the optional local drafting worker."""
from __future__ import annotations

from pathlib import Path

from tools import local_model_offload as offload


def _config(tmp_path: Path, *, enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "base_url": "http://127.0.0.1:8020/v1",
        "expected_model_ids": ["qwen3.6-27b"],
        "timeout_seconds": 1,
        "max_output_tokens": 20,
        "temperature": 0.2,
        "gpu_index": 0,
        "lease_path": str(tmp_path / "gpu.lock"),
        "allowed_tasks": ["documentation", "research_synthesis", "catalog_triage"],
        "forbidden_tasks": ["code_edit", "experiment_design", "gate_decision", "scientific_conclusion"],
    }


def test_disabled_worker_does_not_probe_or_claim_completion(tmp_path: Path) -> None:
    result = offload.run_task("documentation", "draft", _config(tmp_path, enabled=False))
    assert result["status"] == "disabled"
    assert "response" not in result


def test_unavailable_service_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    def unavailable(*args, **kwargs):
        raise offload.OffloadError("endpoint down")

    result = offload.run_task("documentation", "draft", _config(tmp_path), request_json=unavailable)
    assert result["status"] == "unavailable"
    assert result["reason"] == "endpoint down"
    assert "response" not in result


def test_malformed_model_listing_is_rejected() -> None:
    def malformed(*args, **kwargs):
        return {"data": {"id": "qwen3.6-27b"}}

    try:
        offload.probe_service(_config(Path("/tmp")), request_json=malformed)
    except offload.OffloadError as exc:
        assert "list-valued data" in str(exc)
    else:
        raise AssertionError("malformed model listing was accepted")


def test_forbidden_task_is_rejected_before_endpoint_access(tmp_path: Path) -> None:
    called = False

    def should_not_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("endpoint should not be queried")

    result = offload.run_task("gate_decision", "approve", _config(tmp_path), request_json=should_not_call)
    assert result["status"] == "rejected"
    assert not called


def test_gpu_lease_contention_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"
    first = offload.acquire_gpu_lease(path)
    assert first is not None
    try:
        assert offload.acquire_gpu_lease(path) is None
    finally:
        offload.release_gpu_lease(first)


def test_success_records_identity_hashes_and_gpu_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def fake_http(url, *, method, payload=None, timeout):
        if method == "GET":
            return {"data": [{"id": "qwen3.6-27b"}]}
        assert payload["model"] == "qwen3.6-27b"
        return {"choices": [{"message": {"content": "provisional draft"}}]}

    result = offload.run_task(
        "documentation",
        "draft this",
        config,
        request_json=fake_http,
        gpu_state=lambda index: {
            "index": index,
            "utilization_percent": 4,
            "memory_used_mib": 18000,
            "memory_total_mib": 24576,
        },
    )
    assert result["status"] == "completed"
    assert result["model"] == "qwen3.6-27b"
    assert result["response_sha256"] == offload.sha256_text("provisional draft")
    assert result["prompt_sha256"] == offload.sha256_text("draft this")
    assert result["gpu"]["index"] == 0
    assert result["review_required"] is True


def test_lane_dispatch_and_service_wrapper_share_lease_contract() -> None:
    root = Path(__file__).parents[1]
    lane_text = (root / "tools" / "lane_dispatch.sh").read_text(encoding="utf-8")
    service_text = (root / "tools" / "local_model_service.sh").read_text(encoding="utf-8")
    assert "SIM_GPU_LEASE_PATH" in lane_text
    assert "flock -n 8" in lane_text
    assert "SIM_GPU_LEASE_PATH" in service_text
    assert "flock -n 9" in service_text
