#!/usr/bin/env python3
"""Fail-closed, low-risk task offload to the local OpenAI-compatible model.

The local model is a draft assistant, not a scientific authority. It may
write reviewable documentation/research drafts, but it cannot edit source,
choose experiments, approve gates, or silently fall back to a hosted model.
The GPU lease is shared with ``tools/lane_dispatch.sh gpu`` and the foreground
local-model service wrapper so model requests cannot race queued experiments.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping
import uuid
from urllib.request import Request, urlopen


SCHEMA = "local-model-offload-v1"
TASK_SCHEMA = "local-model-offload-task-v1"
RECEIPT_SCHEMA = "local-model-offload-receipt-v1"
OWNER_SCHEMA = "local-model-service-owner-v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "local_model_offload.json"
USER_AGENT = "neural-simulator-local-model-offload/1.0"


class OffloadError(RuntimeError):
    """Raised when a local-model contract check fails."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OffloadError(f"configuration is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OffloadError("configuration must be a JSON object")
    required = {
        "enabled",
        "base_url",
        "expected_model_ids",
        "timeout_seconds",
        "max_output_tokens",
        "temperature",
        "gpu_index",
        "lease_path",
        "ownership_path",
        "queue_directory",
        "service_command",
        "service_stop_command",
        "allowed_tasks",
        "forbidden_tasks",
    }
    missing = sorted(required - set(value))
    if missing:
        raise OffloadError(f"configuration is missing fields: {', '.join(missing)}")
    if not isinstance(value["expected_model_ids"], list) or not value["expected_model_ids"]:
        raise OffloadError("expected_model_ids must be a non-empty list")
    if not isinstance(value["allowed_tasks"], list) or not isinstance(value["forbidden_tasks"], list):
        raise OffloadError("allowed_tasks and forbidden_tasks must be lists")
    if not isinstance(value["queue_directory"], str) or not value["queue_directory"].strip():
        raise OffloadError("queue_directory must be a non-empty path")
    if not isinstance(value["ownership_path"], str) or not value["ownership_path"].strip():
        raise OffloadError("ownership_path must be a non-empty path")
    for field in ("service_command", "service_stop_command"):
        command = value[field]
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise OffloadError(f"{field} must be a non-empty string list")
    return value


def task_eligibility(task: str, config: Mapping[str, Any]) -> tuple[bool, str]:
    """Classify a task without contacting the endpoint or inspecting the GPU."""
    forbidden = {str(item) for item in config["forbidden_tasks"]}
    allowed = {str(item) for item in config["allowed_tasks"]}
    if task in forbidden:
        return False, f"task type is forbidden: {task}"
    if task not in allowed:
        return False, f"task type is not allowlisted: {task}"
    return True, "task type is allowlisted for provisional local drafting"


def queue_directory(config: Mapping[str, Any]) -> Path:
    return Path(str(config["queue_directory"])).expanduser()


def _queue_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "pending": root / "pending",
        "running": root / "running",
        "receipts": root / "receipts",
        "lock": root / ".queue.lock",
    }


def _prepare_queue(root: Path) -> dict[str, Path]:
    paths = _queue_paths(root)
    for name in ("pending", "running", "receipts"):
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OffloadError(f"JSON artifact is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OffloadError(f"JSON artifact is not an object: {path}")
    return value


def enqueue_task(task: str, prompt: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Durably stage an eligible draft task without using the model or GPU."""
    eligible, reason = task_eligibility(task, config)
    if not eligible:
        raise OffloadError(reason)
    if not prompt.strip():
        raise OffloadError("prompt must not be empty")
    created = utc_now()
    task_id = f"{created.replace(':', '').replace('+00:00', 'Z')}-{uuid.uuid4().hex[:12]}"
    envelope = {
        "schema": TASK_SCHEMA,
        "status": "queued",
        "task_id": task_id,
        "task": task,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "created_at": created,
        "eligibility": reason,
        "review_required": True,
        "review_policy": (
            "A human or higher-capability model must review output before it informs "
            "scientific claims, experiment design, gate decisions, or code changes."
        ),
    }
    paths = _prepare_queue(queue_directory(config))
    write_result(paths["pending"] / f"{task_id}.json", envelope)
    return envelope


def queue_status(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = _prepare_queue(queue_directory(config))
    return {
        "schema": TASK_SCHEMA,
        "status": "available",
        "queue_directory": str(paths["root"]),
        "pending": sorted(path.stem for path in paths["pending"].glob("*.json")),
        "running": sorted(path.stem for path in paths["running"].glob("*.json")),
        "receipt_count": len(list(paths["receipts"].glob("*.json"))),
    }


def recover_running(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return interrupted claims to pending; never invokes the local model."""
    paths = _prepare_queue(queue_directory(config))
    recovered: list[str] = []
    with paths["lock"].open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for source in sorted(paths["running"].glob("*.json")):
            destination = paths["pending"] / source.name
            if destination.exists():
                raise OffloadError(f"cannot recover duplicate task: {source.stem}")
            os.replace(source, destination)
            recovered.append(source.stem)
    return {"schema": TASK_SCHEMA, "status": "recovered", "task_ids": recovered}


def _claim_next(paths: Mapping[str, Path]) -> tuple[Path, dict[str, Any]] | None:
    with paths["lock"].open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        pending = sorted(paths["pending"].glob("*.json"))
        if not pending:
            return None
        source = pending[0]
        destination = paths["running"] / source.name
        os.replace(source, destination)
    envelope = _load_object(destination)
    if envelope.get("schema") != TASK_SCHEMA or envelope.get("task_id") != destination.stem:
        raise OffloadError(f"claimed task has invalid identity: {destination}")
    prompt = envelope.get("prompt")
    if not isinstance(prompt, str) or sha256_text(prompt) != envelope.get("prompt_sha256"):
        raise OffloadError(f"claimed task prompt hash does not match: {destination}")
    return destination, envelope


def process_next(
    config: Mapping[str, Any],
    *,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Process one durable task and preserve a review receipt or retryable claim."""
    # The late fallback avoids coupling module import order to the default.
    if runner is None:
        runner = run_task
    paths = _prepare_queue(queue_directory(config))
    claim = _claim_next(paths)
    if claim is None:
        return {"schema": RECEIPT_SCHEMA, "status": "queue_empty", "finished_at": utc_now()}
    claim_path, envelope = claim
    result = runner(str(envelope["task"]), str(envelope["prompt"]), config)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "awaiting_review" if result.get("status") == "completed" else "retryable",
        "task_id": envelope["task_id"],
        "task": envelope["task"],
        "prompt_sha256": envelope["prompt_sha256"],
        "review_required": True,
        "result": result,
        "recorded_at": utc_now(),
    }
    receipt_name = f"{envelope['task_id']}-{uuid.uuid4().hex[:12]}.json"
    write_result(paths["receipts"] / receipt_name, receipt)
    if result.get("status") == "completed":
        claim_path.unlink()
    else:
        destination = paths["pending"] / claim_path.name
        if destination.exists():
            raise OffloadError(f"cannot requeue duplicate task: {claim_path.stem}")
        os.replace(claim_path, destination)
    return receipt


def _url(config: Mapping[str, Any], suffix: str) -> str:
    return f"{str(config['base_url']).rstrip('/')}/{suffix.lstrip('/')}"


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float,
) -> Mapping[str, Any]:
    body = None
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if payload is not None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except Exception as exc:
        raise OffloadError(f"local endpoint request failed: {exc}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OffloadError(f"local endpoint returned invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise OffloadError("local endpoint returned a non-object JSON value")
    return value


def probe_service(
    config: Mapping[str, Any],
    *,
    request_json: Callable[..., Mapping[str, Any]] = http_json,
) -> dict[str, Any]:
    payload = request_json(
        _url(config, "/models"),
        method="GET",
        timeout=float(config["timeout_seconds"]),
    )
    models = payload.get("data")
    if not isinstance(models, list):
        raise OffloadError("/v1/models response has no list-valued data field")
    model_ids = [
        item.get("id")
        for item in models
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    ]
    expected = {str(item) for item in config["expected_model_ids"]}
    selected = next((model_id for model_id in model_ids if model_id in expected), None)
    if selected is None:
        raise OffloadError(
            "expected local model is not served; "
            f"expected={sorted(expected)!r} served={model_ids!r}"
        )
    return {
        "status": "available",
        "endpoint": _url(config, "/models"),
        "served_model_ids": model_ids,
        "model_id": selected,
        "observed_at": utc_now(),
    }


def gpu_snapshot(gpu_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={int(gpu_index)}",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise OffloadError(f"GPU state could not be verified: {exc}") from exc
    if completed.returncode != 0:
        raise OffloadError(f"GPU state could not be verified: {completed.stderr.strip()}")
    fields = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(fields) != 4:
        raise OffloadError(f"GPU state output is malformed: {completed.stdout.strip()!r}")
    try:
        index, utilization, memory_used, memory_total = (int(part) for part in fields)
    except ValueError as exc:
        raise OffloadError(f"GPU state contains non-numeric values: {fields!r}") from exc
    return {
        "index": index,
        "utilization_percent": utilization,
        "memory_used_mib": memory_used,
        "memory_total_mib": memory_total,
        "observed_at": utc_now(),
    }


def acquire_gpu_lease(path: Path):
    """Return a held non-blocking advisory lock, or ``None`` when busy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def release_gpu_lease(handle) -> None:
    if handle is None:
        return
    with suppress(OSError):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    with suppress(OSError):
        handle.close()


def ownership_path(config: Mapping[str, Any]) -> Path:
    return Path(str(config["ownership_path"])).expanduser()


def process_identity(pid: int) -> dict[str, Any]:
    """Return identity fields that distinguish a live process from PID reuse."""
    try:
        stat_text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OffloadError(f"service owner process is not live: pid={pid}: {exc}") from exc
    close = stat_text.rfind(")")
    fields = stat_text[close + 2 :].split() if close >= 0 else []
    if len(fields) <= 19:
        raise OffloadError(f"service owner process identity is malformed: pid={pid}")
    return {"pid": int(pid), "start_ticks": fields[19], "boot_id": boot_id}


def lease_owner_pid(path: Path) -> int | None:
    """Read the kernel FLOCK owner for this exact lease inode."""
    try:
        stat = path.stat()
        lines = Path("/proc/locks").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OffloadError(f"GPU lease ownership could not be inspected: {exc}") from exc
    identity = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    inode_suffix = f":{stat.st_ino}"
    inode_owners: set[int] = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 6 or fields[1] != "FLOCK" or fields[3] != "WRITE":
            continue
        if fields[5] == identity:
            try:
                return int(fields[4])
            except ValueError as exc:
                raise OffloadError(f"GPU lease owner PID is malformed: {line!r}") from exc
        # Bind-mounted test/runtime directories can expose a translated device
        # number in /proc/locks. In that case require one unambiguous owner for
        # the same inode; the ownership record also binds the canonical path.
        if fields[5].endswith(inode_suffix):
            try:
                inode_owners.add(int(fields[4]))
            except ValueError as exc:
                raise OffloadError(f"GPU lease owner PID is malformed: {line!r}") from exc
    return next(iter(inode_owners)) if len(inode_owners) == 1 else None


def verify_service_ownership(
    config: Mapping[str, Any],
    *,
    identify_process: Callable[[int], dict[str, Any]] = process_identity,
    find_lock_owner: Callable[[Path], int | None] = lease_owner_pid,
) -> dict[str, Any]:
    """Authorize requests only for the live broker that owns the kernel lease."""
    path = ownership_path(config)
    try:
        record = _load_object(path)
    except OffloadError as exc:
        raise OffloadError(f"unsafe local service ownership: {exc}") from exc
    lease_path = Path(str(config["lease_path"])).expanduser().resolve()
    try:
        lease_inode = lease_path.stat().st_ino
    except OSError as exc:
        raise OffloadError(f"unsafe local service ownership: lease file is unavailable: {exc}") from exc
    required = {
        "schema": OWNER_SCHEMA,
        "state": "service_owned",
        "lease_path": str(lease_path),
        "base_url": str(config["base_url"]).rstrip("/"),
        "lease_inode": lease_inode,
    }
    for field, expected in required.items():
        if record.get(field) != expected:
            raise OffloadError(f"unsafe local service ownership: {field} does not match")
    pid = record.get("owner_pid")
    if not isinstance(pid, int) or pid <= 0:
        raise OffloadError("unsafe local service ownership: owner_pid is invalid")
    live = identify_process(pid)
    if record.get("owner_start_ticks") != live.get("start_ticks") or record.get("boot_id") != live.get("boot_id"):
        raise OffloadError("unsafe local service ownership: owner process identity is stale")
    if find_lock_owner(lease_path) != pid:
        raise OffloadError("unsafe local service ownership: broker does not own the GPU lease")
    return record


def claim_service_ownership(
    config: Mapping[str, Any],
    *,
    owner_pid: int | None = None,
    identify_process: Callable[[int], dict[str, Any]] = process_identity,
) -> tuple[Any, dict[str, Any]]:
    """Acquire the exclusive experiment lease and publish broker ownership."""
    lease_path = Path(str(config["lease_path"])).expanduser().resolve()
    lease = acquire_gpu_lease(lease_path)
    if lease is None:
        raise OffloadError("GPU lease is busy; an experiment or service already owns GPU 0")
    try:
        pid = os.getpid() if owner_pid is None else int(owner_pid)
        identity = identify_process(pid)
        record = {
            "schema": OWNER_SCHEMA,
            "state": "service_owned",
            "owner_pid": pid,
            "owner_start_ticks": identity["start_ticks"],
            "boot_id": identity["boot_id"],
            "lease_path": str(lease_path),
            "base_url": str(config["base_url"]).rstrip("/"),
            "lease_inode": lease_path.stat().st_ino,
            "ownership_token": uuid.uuid4().hex,
            "acquired_at": utc_now(),
        }
        write_result(ownership_path(config), record)
        return lease, record
    except Exception:
        release_gpu_lease(lease)
        raise


def clear_service_ownership(config: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """Remove only the ownership record created by this broker instance."""
    path = ownership_path(config)
    if not path.exists():
        return
    current = _load_object(path)
    if current.get("ownership_token") != record.get("ownership_token"):
        raise OffloadError("refusing to remove a different service owner's record")
    path.unlink()


def recover_service_ownership(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove stale metadata only when no process owns the GPU lease."""
    path = ownership_path(config)
    if not path.exists():
        return {"schema": OWNER_SCHEMA, "status": "absent"}
    try:
        record = verify_service_ownership(config)
    except OffloadError as invalid:
        lease = acquire_gpu_lease(Path(str(config["lease_path"])).expanduser().resolve())
        if lease is None:
            return {
                "schema": OWNER_SCHEMA,
                "status": "blocked_busy",
                "reason": f"stale or invalid ownership cannot be recovered while lease is busy: {invalid}",
            }
        try:
            path.unlink()
        finally:
            release_gpu_lease(lease)
        return {"schema": OWNER_SCHEMA, "status": "stale_recovered", "reason": str(invalid)}
    return {"schema": OWNER_SCHEMA, "status": "active", "owner": record}


def run_service_broker(
    config: Mapping[str, Any],
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    run_command: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Own the lease for the complete foreground service lifetime."""
    if not bool(config.get("enabled")):
        raise OffloadError("local-model offload is disabled by configuration")
    lease, record = claim_service_ownership(config)
    process = None
    try:
        # The foreground service child inherits the same open-file description.
        # If the broker is killed, the kernel lease remains held until that
        # child exits, preventing an experiment from racing an orphan service.
        process = popen(list(config["service_command"]), pass_fds=(lease.fileno(),))
        return_code = int(process.wait())
        return {
            "schema": OWNER_SCHEMA,
            "status": "service_stopped" if return_code == 0 else "service_failed",
            "return_code": return_code,
            "owner": record,
        }
    finally:
        # Deny new requests first, stop the endpoint fully, then release GPU 0.
        try:
            clear_service_ownership(config, record)
        finally:
            try:
                if process is not None:
                    with suppress(OSError, subprocess.SubprocessError):
                        run_command(list(config["service_stop_command"]), check=False, timeout=60)
            finally:
                release_gpu_lease(lease)


def request_completion(
    config: Mapping[str, Any],
    model_id: str,
    prompt: str,
    *,
    request_json: Callable[..., Mapping[str, Any]] = http_json,
) -> str:
    payload = request_json(
        _url(config, "/chat/completions"),
        method="POST",
        timeout=float(config["timeout_seconds"]),
        payload={
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a local drafting assistant. Return a clearly provisional draft. "
                        "Do not make unsupported scientific claims, approve gates, or edit code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(config["temperature"]),
            "max_tokens": int(config["max_output_tokens"]),
        },
    )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise OffloadError("completion response has no usable choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise OffloadError("completion response has no textual message content")
    content = message["content"].strip()
    if not content:
        raise OffloadError("completion response was empty")
    return content


def run_task(
    task: str,
    prompt: str,
    config: Mapping[str, Any],
    *,
    request_json: Callable[..., Mapping[str, Any]] = http_json,
    gpu_state: Callable[[int], dict[str, Any]] = gpu_snapshot,
    ownership_check: Callable[[Mapping[str, Any]], dict[str, Any]] = verify_service_ownership,
) -> dict[str, Any]:
    started = utc_now()
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "task": task,
        "started_at": started,
        "finished_at": None,
        "prompt_sha256": sha256_text(prompt),
        "review_required": True,
    }
    if not bool(config.get("enabled")):
        base.update({"status": "disabled", "reason": "local-model offload is disabled by configuration"})
        base["finished_at"] = utc_now()
        return base
    eligible, reason = task_eligibility(task, config)
    if not eligible:
        base.update({"status": "rejected", "reason": reason})
        base["finished_at"] = utc_now()
        return base

    try:
        try:
            owner = ownership_check(config)
        except OffloadError as exc:
            base.update({"status": "unsafe_service", "reason": str(exc)})
            return base
        try:
            service = probe_service(config, request_json=request_json)
        except OffloadError as exc:
            base.update({"status": "unavailable", "reason": str(exc), "owner": owner})
            return base
        try:
            gpu = gpu_state(int(config["gpu_index"]))
            content = request_completion(config, service["model_id"], prompt, request_json=request_json)
        except OffloadError as exc:
            base.update({"status": "failed", "reason": str(exc), "service": service})
            return base
        base.update({
            "status": "completed",
            "owner": owner,
            "service": service,
            "gpu": gpu,
            "model": service["model_id"],
            "endpoint": _url(config, "/chat/completions"),
            "response": content,
            "response_sha256": sha256_text(content),
        })
        return base
    finally:
        base["finished_at"] = utc_now()


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _emit(result: Mapping[str, Any], output: Path | None) -> None:
    if output is not None:
        write_result(output, result)
    print(json.dumps(dict(result), indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("SIM_LOCAL_MODEL_CONFIG", DEFAULT_CONFIG)))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe", help="verify broker ownership, then probe the configured endpoint")
    sub.add_parser("broker", help="own the GPU lease while running the configured model service")
    sub.add_parser("owner-status", help="verify the live service owner's process and kernel lock")
    sub.add_parser("recover-owner", help="remove stale ownership metadata when the GPU lease is free")
    sub.add_parser("list", help="show durable local-model queue state without using the GPU")
    sub.add_parser("recover", help="return interrupted task claims to the pending queue")
    enqueue = sub.add_parser("enqueue", help="durably queue one eligible task without using the GPU")
    enqueue.add_argument("--task", required=True)
    enqueue_source = enqueue.add_mutually_exclusive_group(required=True)
    enqueue_source.add_argument("--input", type=Path)
    enqueue_source.add_argument("--prompt")
    enqueue.add_argument("--output", type=Path)
    process = sub.add_parser("process-next", help="process one queued task when the GPU lane is available")
    process.add_argument("--output", type=Path)
    run = sub.add_parser("run", help="run one allowlisted draft task")
    run.add_argument("--task", required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--prompt")
    run.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.command == "probe":
            owner = verify_service_ownership(config)
            result = probe_service(config)
            result["owner"] = owner
            _emit(result, None)
            return 0
        if args.command == "broker":
            result = run_service_broker(config)
            _emit(result, None)
            return 0 if result["status"] == "service_stopped" else 1
        if args.command == "owner-status":
            _emit({"schema": OWNER_SCHEMA, "status": "active", "owner": verify_service_ownership(config)}, None)
            return 0
        if args.command == "recover-owner":
            result = recover_service_ownership(config)
            _emit(result, None)
            return 1 if result["status"] == "blocked_busy" else 0
        if args.command == "list":
            _emit(queue_status(config), None)
            return 0
        if args.command == "recover":
            _emit(recover_running(config), None)
            return 0
        if args.command == "process-next":
            result = process_next(config)
            _emit(result, args.output)
            return 0 if result["status"] in {"awaiting_review", "queue_empty"} else 1
        if args.prompt is not None:
            prompt = args.prompt
        else:
            try:
                prompt = args.input.read_text(encoding="utf-8")
            except OSError as exc:
                raise OffloadError(f"input is unreadable: {args.input}: {exc}") from exc
        if args.command == "enqueue":
            result = enqueue_task(args.task, prompt, config)
            _emit(result, args.output)
            return 0
        result = run_task(args.task, prompt, config)
        _emit(result, args.output)
        return 0 if result["status"] == "completed" else 1
    except OffloadError as exc:
        result = {"schema": SCHEMA, "status": "failed", "reason": str(exc), "finished_at": utc_now()}
        _emit(result, getattr(args, "output", None))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
