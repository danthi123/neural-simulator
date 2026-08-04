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
from urllib.request import Request, urlopen


SCHEMA = "local-model-offload-v1"
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
    return value


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
    if task in {str(item) for item in config["forbidden_tasks"]}:
        base.update({"status": "rejected", "reason": f"task type is forbidden: {task}"})
        base["finished_at"] = utc_now()
        return base
    if task not in {str(item) for item in config["allowed_tasks"]}:
        base.update({"status": "rejected", "reason": f"task type is not allowlisted: {task}"})
        base["finished_at"] = utc_now()
        return base

    lease = None
    try:
        try:
            service = probe_service(config, request_json=request_json)
        except OffloadError as exc:
            base.update({"status": "unavailable", "reason": str(exc)})
            return base
        lease = acquire_gpu_lease(Path(str(config["lease_path"])))
        if lease is None:
            base.update({"status": "gpu_busy", "reason": "shared GPU lease is held by another workload"})
            return base
        try:
            gpu = gpu_state(int(config["gpu_index"]))
            content = request_completion(config, service["model_id"], prompt, request_json=request_json)
        except OffloadError as exc:
            base.update({"status": "failed", "reason": str(exc), "service": service})
            return base
        base.update({
            "status": "completed",
            "service": service,
            "gpu": gpu,
            "model": service["model_id"],
            "endpoint": _url(config, "/chat/completions"),
            "response": content,
            "response_sha256": sha256_text(content),
        })
        return base
    finally:
        release_gpu_lease(lease)
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
    sub.add_parser("probe", help="probe the configured local endpoint")
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
            result = probe_service(config)
            _emit(result, None)
            return 0
        if args.prompt is not None:
            prompt = args.prompt
        else:
            try:
                prompt = args.input.read_text(encoding="utf-8")
            except OSError as exc:
                raise OffloadError(f"input is unreadable: {args.input}: {exc}") from exc
        result = run_task(args.task, prompt, config)
        _emit(result, args.output)
        return 0 if result["status"] == "completed" else 1
    except OffloadError as exc:
        result = {"schema": SCHEMA, "status": "failed", "reason": str(exc), "finished_at": utc_now()}
        _emit(result, getattr(args, "output", None))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
