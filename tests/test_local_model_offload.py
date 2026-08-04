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
        "ownership_path": str(tmp_path / "service-owner.json"),
        "queue_directory": str(tmp_path / "queue"),
        "service_command": ["fake-compose", "up"],
        "service_stop_command": ["fake-compose", "down"],
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

    result = offload.run_task(
        "documentation",
        "draft",
        _config(tmp_path),
        request_json=unavailable,
        ownership_check=lambda config: {"state": "service_owned"},
    )
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
        ownership_check=lambda supplied: {"state": "service_owned", "owner_pid": 123},
    )
    assert result["status"] == "completed"
    assert result["model"] == "qwen3.6-27b"
    assert result["response_sha256"] == offload.sha256_text("provisional draft")
    assert result["prompt_sha256"] == offload.sha256_text("draft this")
    assert result["gpu"]["index"] == 0
    assert result["review_required"] is True
    assert result["owner"]["owner_pid"] == 123


def test_lane_dispatch_and_service_wrapper_share_lease_contract() -> None:
    root = Path(__file__).parents[1]
    lane_text = (root / "tools" / "lane_dispatch.sh").read_text(encoding="utf-8")
    service_text = (root / "tools" / "local_model_service.sh").read_text(encoding="utf-8")
    assert "SIM_GPU_LEASE_PATH" in lane_text
    assert "flock -n 8" in lane_text
    assert "SIM_GPU_LEASE_PATH" in service_text
    assert "flock -n 9" in service_text


def test_enqueue_is_durable_and_does_not_access_endpoint_or_gpu(tmp_path: Path) -> None:
    config = _config(tmp_path)
    envelope = offload.enqueue_task("documentation", "Summarize reviewed notes.", config)

    queued = tmp_path / "queue" / "pending" / f"{envelope['task_id']}.json"
    assert queued.exists()
    assert envelope["status"] == "queued"
    assert envelope["review_required"] is True
    assert envelope["prompt_sha256"] == offload.sha256_text("Summarize reviewed notes.")
    status = offload.queue_status(config)
    assert status["pending"] == [envelope["task_id"]]
    assert status["running"] == []


def test_enqueue_rejects_decision_task_before_writing_queue(tmp_path: Path) -> None:
    config = _config(tmp_path)
    try:
        offload.enqueue_task("experiment_design", "Choose the next experiment.", config)
    except offload.OffloadError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("forbidden task was queued")
    assert offload.queue_status(config)["pending"] == []


def test_retryable_attempt_preserves_task_and_writes_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    envelope = offload.enqueue_task("catalog_triage", "Rank these references.", config)

    def busy(task, prompt, supplied_config):
        assert task == "catalog_triage"
        assert prompt == "Rank these references."
        assert supplied_config is config
        return {"schema": offload.SCHEMA, "status": "gpu_busy", "review_required": True}

    receipt = offload.process_next(config, runner=busy)
    status = offload.queue_status(config)
    assert receipt["status"] == "retryable"
    assert receipt["result"]["status"] == "gpu_busy"
    assert status["pending"] == [envelope["task_id"]]
    assert status["running"] == []
    assert status["receipt_count"] == 1


def test_completed_attempt_requires_review_and_clears_claim(tmp_path: Path) -> None:
    config = _config(tmp_path)
    envelope = offload.enqueue_task("research_synthesis", "Organize reviewed notes.", config)

    def completed(task, prompt, supplied_config):
        return {
            "schema": offload.SCHEMA,
            "status": "completed",
            "response": "provisional draft",
            "review_required": True,
        }

    receipt = offload.process_next(config, runner=completed)
    status = offload.queue_status(config)
    assert receipt["status"] == "awaiting_review"
    assert receipt["task_id"] == envelope["task_id"]
    assert receipt["review_required"] is True
    assert status["pending"] == []
    assert status["running"] == []
    assert status["receipt_count"] == 1


def test_recover_returns_interrupted_claim_to_pending(tmp_path: Path) -> None:
    config = _config(tmp_path)
    envelope = offload.enqueue_task("documentation", "Draft a summary.", config)
    paths = offload._prepare_queue(offload.queue_directory(config))
    pending = paths["pending"] / f"{envelope['task_id']}.json"
    running = paths["running"] / pending.name
    pending.replace(running)

    result = offload.recover_running(config)
    assert result["task_ids"] == [envelope["task_id"]]
    assert offload.queue_status(config)["pending"] == [envelope["task_id"]]


def test_direct_request_without_verified_owner_never_touches_endpoint(tmp_path: Path) -> None:
    called = False

    def should_not_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsafe request reached endpoint")

    result = offload.run_task("documentation", "draft", _config(tmp_path), request_json=should_not_call)
    assert result["status"] == "unsafe_service"
    assert "ownership" in result["reason"]
    assert not called


def test_legitimate_broker_owner_allows_request_without_reacquiring_lease(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lease, record = offload.claim_service_ownership(config)

    def fake_http(url, *, method, payload=None, timeout):
        if method == "GET":
            return {"data": [{"id": "qwen3.6-27b"}]}
        return {"choices": [{"message": {"content": "provisional"}}]}

    try:
        assert offload.acquire_gpu_lease(Path(config["lease_path"])) is None
        result = offload.run_task(
            "documentation",
            "draft",
            config,
            request_json=fake_http,
            gpu_state=lambda index: {"index": index},
        )
        assert result["status"] == "completed"
        assert result["owner"]["ownership_token"] == record["ownership_token"]
    finally:
        offload.clear_service_ownership(config, record)
        offload.release_gpu_lease(lease)


def test_owner_record_without_matching_kernel_lock_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lease, record = offload.claim_service_ownership(config)
    try:
        try:
            offload.verify_service_ownership(config, find_lock_owner=lambda path: record["owner_pid"] + 1)
        except offload.OffloadError as exc:
            assert "does not own" in str(exc)
        else:
            raise AssertionError("ownership record without matching lock was accepted")
    finally:
        offload.clear_service_ownership(config, record)
        offload.release_gpu_lease(lease)


def test_pid_reuse_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lease, record = offload.claim_service_ownership(config)
    try:
        try:
            offload.verify_service_ownership(
                config,
                identify_process=lambda pid: {"pid": pid, "start_ticks": "different", "boot_id": record["boot_id"]},
                find_lock_owner=lambda path: record["owner_pid"],
            )
        except offload.OffloadError as exc:
            assert "identity is stale" in str(exc)
        else:
            raise AssertionError("reused PID identity was accepted")
    finally:
        offload.clear_service_ownership(config, record)
        offload.release_gpu_lease(lease)


def test_service_startup_fails_before_spawn_when_experiment_owns_lease(tmp_path: Path) -> None:
    config = _config(tmp_path)
    experiment = offload.acquire_gpu_lease(Path(config["lease_path"]))
    spawned = False

    def should_not_spawn(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("service spawned during experiment lease")

    try:
        try:
            offload.run_service_broker(config, popen=should_not_spawn)
        except offload.OffloadError as exc:
            assert "lease is busy" in str(exc)
        else:
            raise AssertionError("service broker started during experiment lease")
        assert not spawned
        assert not Path(config["ownership_path"]).exists()
    finally:
        offload.release_gpu_lease(experiment)


def test_stale_owner_recovers_only_after_lease_is_free(tmp_path: Path) -> None:
    config = _config(tmp_path)
    owner_path = Path(config["ownership_path"])
    offload.write_result(owner_path, {"schema": offload.OWNER_SCHEMA, "state": "service_owned", "owner_pid": 999999})
    experiment = offload.acquire_gpu_lease(Path(config["lease_path"]))
    try:
        blocked = offload.recover_service_ownership(config)
        assert blocked["status"] == "blocked_busy"
        assert owner_path.exists()
    finally:
        offload.release_gpu_lease(experiment)
    recovered = offload.recover_service_ownership(config)
    assert recovered["status"] == "stale_recovered"
    assert not owner_path.exists()


def test_broker_clears_authorization_then_stops_service_before_releasing_lease(tmp_path: Path) -> None:
    config = _config(tmp_path)
    events = []

    class FakeProcess:
        def wait(self):
            assert offload.verify_service_ownership(config)["state"] == "service_owned"
            events.append("service_running")
            return 0

    def fake_popen(command, *, pass_fds):
        assert command == config["service_command"]
        assert len(pass_fds) == 1
        assert pass_fds[0] >= 0
        events.append("spawn")
        return FakeProcess()

    def fake_stop(command, *, check, timeout):
        assert command == config["service_stop_command"]
        assert not Path(config["ownership_path"]).exists()
        assert offload.acquire_gpu_lease(Path(config["lease_path"])) is None
        events.append("stop_while_locked")

    result = offload.run_service_broker(config, popen=fake_popen, run_command=fake_stop)
    assert result["status"] == "service_stopped"
    assert events == ["spawn", "service_running", "stop_while_locked"]
    available = offload.acquire_gpu_lease(Path(config["lease_path"]))
    assert available is not None
    offload.release_gpu_lease(available)
