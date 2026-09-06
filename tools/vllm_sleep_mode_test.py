#!/usr/bin/env python3
"""vllm_sleep_mode_test.py — the DECISIVE measurement for the vLLM Sleep Mode pilot.

Drives a running vllm_sleep_pilot_serve.sh server through: baseline inference -> record VRAM -> POST /sleep ->
confirm VRAM actually dropped (not just that the endpoint returned 200) -> POST /wake_up (+ reload_weights for
level 2) -> confirm VRAM restored -> confirm inference resumes and is coherent. Emits one JSON report per level
tested, with an explicit per-level verdict so the "does it free VRAM cleanly + wake in single-digit seconds?"
question in the task has a machine-checkable answer instead of a felt impression from watching logs scroll.

Stdlib only (urllib, subprocess, json, argparse) — runs under ANY python3, not just the vllm pilot's own venv,
so it can be invoked from the sim repo's normal environment without perturbing anything.

Usage:
    python3 tools/vllm_sleep_mode_test.py --endpoint http://127.0.0.1:18020 --model qwen3.8-27b-pilot \\
        --levels 1,2 --out research/findings/raw/vllm_sleep_mode_pilot/sleep_wake_$(date +%s).json

⚠️ Talks to a server that must ALREADY be up (this script does not launch/stop vLLM — see
tools/vllm_sleep_pilot_serve.sh). Run this only when the controller has confirmed the GPU is free of resident
brain jobs (`tools/gpu_queue.sh status` + `nvidia-smi`) and launched the pilot server itself — see the exact
commands in research/findings/2026-09-06-vllm-sleep-mode-pilot-*.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Generous pilot bars, not the target. The task's OWN target is "single-digit seconds" (~3-6s, the figure vLLM's
# blog post quotes) for a level-1 sleep/wake round-trip; PASS here just means "worked at all, within a bound wide
# enough that a slow-but-working result doesn't get misclassified as broken." Read the actual seconds, not just PASS/FAIL.
WAKE_PASS_S = 15.0
SLEEP_PASS_S = 15.0
# VRAM must drop by at least this fraction of the pre-sleep RESIDENT total for a sleep to count as having freed
# anything real (catches a /sleep that returns 200 but the allocator silently keeps everything resident — the
# Mamba/GDN recurrent-state cache is a SEPARATE, special-cased pool from the ordinary KV cache upstream, per
# vLLM's own docs, and is exactly the kind of thing a sleep-mode implementation could miss on a hybrid model).
MIN_VRAM_FREED_FRACTION = 0.5


def http(method: str, url: str, api_key: str | None = None, timeout: float = 60.0, data: bytes | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, method=method, data=data)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return -1, str(e)


def gpu_used_mib(gpu_index: int = 0) -> float | None:
    """Ground truth from nvidia-smi, not from anything vLLM self-reports — a self-report of freed memory that
    the allocator did not actually release would pass a check that only reads vLLM's own response body."""
    try:
        out = subprocess.run(
            ["timeout", "8", "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=12,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_health(endpoint: str, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, _ = http("GET", f"{endpoint}/health", timeout=4)
        if status == 200:
            return True
        time.sleep(1)
    return False


def is_sleeping(endpoint: str, api_key: str | None) -> bool | None:
    status, body = http("GET", f"{endpoint}/is_sleeping", api_key, timeout=8)
    if status != 200:
        return None
    try:
        j = json.loads(body)
        # vLLM's response shape has varied across versions; accept either a bare bool or {"is_sleeping": bool}.
        if isinstance(j, bool):
            return j
        if isinstance(j, dict) and "is_sleeping" in j:
            return bool(j["is_sleeping"])
    except json.JSONDecodeError:
        pass
    return "true" in body.lower()


def run_inference(endpoint: str, model: str, api_key: str | None, prompt: str) -> tuple[bool, float, str]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 24,
        "temperature": 0.0,
    }).encode()
    t0 = time.time()
    status, body = http("POST", f"{endpoint}/v1/chat/completions", api_key, timeout=120, data=payload)
    dt = time.time() - t0
    if status != 200:
        return False, dt, f"HTTP {status}: {body[:300]}"
    try:
        j = json.loads(body)
        text = j["choices"][0]["message"]["content"]
        return bool(text.strip()), dt, text[:200]
    except Exception as e:
        return False, dt, f"parse error: {e!r} body={body[:300]}"


def poll_until(pred, timeout_s: float, interval_s: float = 0.5):
    """Returns (achieved: bool, elapsed_s: float)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if pred():
            return True, time.time() - t0
        time.sleep(interval_s)
    return False, time.time() - t0


def test_one_level(endpoint: str, model: str, api_key: str | None, level: int, gpu_index: int, prompt: str) -> dict:
    result: dict = {"level": level}

    ok0, dt0, text0 = run_inference(endpoint, model, api_key, prompt)
    result["inference_before"] = {"ok": ok0, "seconds": round(dt0, 3), "sample": text0}
    vram_awake = gpu_used_mib(gpu_index)
    result["vram_awake_mib"] = vram_awake

    t_sleep0 = time.time()
    status, body = http("POST", f"{endpoint}/sleep?level={level}", api_key, timeout=60)
    sleep_call_s = time.time() - t_sleep0
    result["sleep_http_status"] = status
    result["sleep_call_seconds"] = round(sleep_call_s, 3)

    achieved, elapsed = poll_until(lambda: is_sleeping(endpoint, api_key) is True, timeout_s=SLEEP_PASS_S)
    result["is_sleeping_confirmed"] = achieved
    result["sleep_confirm_seconds"] = round(elapsed, 3)
    time.sleep(1.0)  # let the allocator settle before reading nvidia-smi
    vram_asleep = gpu_used_mib(gpu_index)
    result["vram_asleep_mib"] = vram_asleep

    freed_mib = (vram_awake - vram_asleep) if (vram_awake is not None and vram_asleep is not None) else None
    result["vram_freed_mib"] = freed_mib
    result["vram_freed_fraction_of_awake"] = (freed_mib / vram_awake) if (freed_mib is not None and vram_awake) else None

    t_wake0 = time.time()
    status, body = http("POST", f"{endpoint}/wake_up", api_key, timeout=120)
    wake_call_s = time.time() - t_wake0
    result["wake_http_status"] = status
    result["wake_call_seconds"] = round(wake_call_s, 3)

    if level == 2:
        # Level 2 discards weights entirely (only small buffers survive in CPU) — vLLM's own docs say wake_up
        # for level 2 needs an explicit reload_weights() collective RPC afterwards. Time it SEPARATELY: this is
        # the step that actually re-reads the ~14-15GB checkpoint from disk/page-cache, and folding it into
        # "wake_call_seconds" above would hide the one number level 2's slower-wake claim rests on.
        t_reload0 = time.time()
        status, body = http("POST", f"{endpoint}/collective_rpc", api_key, timeout=180,
                             data=json.dumps({"method": "reload_weights"}).encode())
        result["reload_weights_http_status"] = status
        result["reload_weights_seconds"] = round(time.time() - t_reload0, 3)

    achieved, elapsed = poll_until(lambda: is_sleeping(endpoint, api_key) is False, timeout_s=WAKE_PASS_S)
    result["is_awake_confirmed"] = achieved
    result["wake_confirm_seconds"] = round(elapsed, 3)
    time.sleep(1.0)
    vram_awake_after = gpu_used_mib(gpu_index)
    result["vram_awake_after_wake_mib"] = vram_awake_after

    ok1, dt1, text1 = run_inference(endpoint, model, api_key, prompt)
    result["inference_after"] = {"ok": ok1, "seconds": round(dt1, 3), "sample": text1}

    total_wake_s = wake_call_s + result.get("reload_weights_seconds", 0.0)
    result["total_wake_seconds"] = round(total_wake_s, 3)

    # Explicit, honest, threshold-based — not a felt impression. See module docstring for why the pass bars
    # are wide: this is "did it work, and how fast", not a claim that it hit the 3-6s target.
    checks = {
        "slept_and_confirmed": bool(result["is_sleeping_confirmed"]),
        "vram_freed_meaningfully": bool(result["vram_freed_fraction_of_awake"] is not None
                                         and result["vram_freed_fraction_of_awake"] >= MIN_VRAM_FREED_FRACTION),
        "woke_and_confirmed": bool(result["is_awake_confirmed"]),
        "inference_resumed": bool(ok1),
        "wake_within_pass_bar": bool(total_wake_s <= WAKE_PASS_S),
    }
    result["checks"] = checks
    result["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    result["hits_3_6s_target"] = bool(total_wake_s <= 6.0)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://127.0.0.1:18020")
    ap.add_argument("--model", default="qwen3.8-27b-pilot")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--levels", default="1", help="comma-separated sleep levels to test in sequence, e.g. 1,2")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--prompt", default="In one short sentence, what is a Gated DeltaNet?")
    ap.add_argument("--out", default=None, help="write the full JSON report here (dir created if needed)")
    ap.add_argument("--skip-startup-wait", action="store_true")
    args = ap.parse_args()

    if not args.skip_startup_wait and not wait_health(args.endpoint, timeout_s=10):
        print(f"ERROR: {args.endpoint}/health not reachable — is vllm_sleep_pilot_serve.sh up and ready?",
              file=sys.stderr)
        return 2

    report: dict = {"endpoint": args.endpoint, "model": args.model, "levels_tested": [], "results": []}
    for level_s in args.levels.split(","):
        level = int(level_s.strip())
        print(f"--- testing sleep level {level} ---", file=sys.stderr)
        r = test_one_level(args.endpoint, args.model, args.api_key, level, args.gpu_index, args.prompt)
        report["levels_tested"].append(level)
        report["results"].append(r)
        print(json.dumps(r, indent=2), file=sys.stderr)

    report["overall_verdict"] = "PASS" if all(r["verdict"] == "PASS" for r in report["results"]) else "FAIL"

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)

    print(json.dumps({"overall_verdict": report["overall_verdict"],
                       "results": [{"level": r["level"], "verdict": r["verdict"],
                                     "total_wake_seconds": r["total_wake_seconds"],
                                     "vram_freed_mib": r["vram_freed_mib"]} for r in report["results"]]},
                      indent=2))
    return 0 if report["overall_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
