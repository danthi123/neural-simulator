#!/usr/bin/env python3
"""chat_latency_probe.py — HTTP-level latency probe for `/api/brain-chat` (A7, warm-server-prewarm,
2026-09-24; plan step S18).

WHAT IT MEASURES (client-side wall clock only; host-lifecycle tool, touches no cognition):
  - cold_first_turn_ms: the FIRST turn against a session name never seen before (pays whatever
    lazy build cost the server has not already paid — a fresh brain build, and on a GPU host the
    ~58s Qwen-0.5B model load if BRAIN_PREWARM/the existing qwen startup-warm has not already run).
  - warm p50 / p95 over N subsequent turns on an already-built session.
  - the "Qwen generation span" ISOLATED per turn: each scripted message is sent TWICE, once at
    `--renderer` (the renderer under test, e.g. 'qwen') and once at renderer='raw' (the brain's own
    answer with NO LLM rendering step at all) on a second, otherwise-identical scripted session.
    The per-turn wall-clock DELTA (target − raw) is reported as the renderer's own isolated cost —
    this is legitimate because the only thing that differs between the two sessions is which
    renderer object turns the SAME recalled fact into surface text; the delta is not itself timed
    inside the server (no new response field was added — see the PREWARM finding this tool
    produced, research/findings/2026-09-24-brain-prewarm-scratch-session-plasticity-leak.md, for why
    a server-side per-phase timestamp is a SEPARATE, larger change (S01's
    `_prod_chat_phase_timing.py`), not this tool's job).

USAGE
    python tools/chat_latency_probe.py --url http://127.0.0.1:8000 \\
        --brain tiny-demo --renderer qwen --n-turns 10 --out research/findings/raw/_prewarm/latency.json
    # numpy/stub smoke (no GPU needed):
    python tools/chat_latency_probe.py --url http://127.0.0.1:8000 --renderer stub --n-turns 5

Exits 0 and writes the JSON report even if some turns 400 (the server errors are recorded, not
hidden) — a hard exit(1) only for a transport failure (server unreachable) on the FIRST turn.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - urllib fallback keeps this tool dependency-light
    requests = None
    import urllib.request
    import urllib.error


# A short, varied scripted conversation over the shipped tiny-demo fixture's own facts
# (research/runners/brain_chat_tui.py::_build_tiny_demo — "brain use spikes" / "dog chase cat" /
# "cat eat fish"), so every turn is a real gate-hit + verified render, not a trivial abstain.
_DEFAULT_MESSAGES = [
    "what does the dog chase?",
    "what does it eat?",
    "what does the brain use?",
    "what does the brain learn?",
    "what does the brain store?",
    "what does the cat eat?",
    "what does the dog chase?",
    "tell me more",
    "what does it eat?",
    "what does the brain use?",
]


def _post(url: str, payload: dict, timeout: float) -> tuple[float, int, dict | None, str | None]:
    """POST payload to url; returns (elapsed_seconds, status_code, json_body_or_None, error_or_None)."""
    t0 = time.perf_counter()
    if requests is not None:
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            dt = time.perf_counter() - t0
            try:
                body = r.json()
            except Exception:
                body = None
            return dt, r.status_code, body, None
        except Exception as e:
            return time.perf_counter() - t0, 0, None, f"{type(e).__name__}: {e}"
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                dt = time.perf_counter() - t0
                body = json.loads(resp.read().decode("utf-8"))
                return dt, resp.status, body, None
        except urllib.error.HTTPError as e:
            dt = time.perf_counter() - t0
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                body = None
            return dt, e.code, body, None
        except Exception as e:
            return time.perf_counter() - t0, 0, None, f"{type(e).__name__}: {e}"


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _run_session(url: str, session: str, brain: str, renderer: str, messages: list[str],
                  timeout: float) -> list[dict[str, Any]]:
    """Run `messages` as ONE scripted conversation against `session` (fresh each call — a UUID
    suffix keeps two probe runs from colliding on the same server). Returns one record per turn."""
    endpoint = url.rstrip("/") + "/api/brain-chat"
    out = []
    for i, msg in enumerate(messages):
        payload = {"session": session, "message": msg, "brain": brain, "renderer": renderer}
        dt, code, body, err = _post(endpoint, payload, timeout)
        rec = {
            "turn": i, "message": msg, "elapsed_s": dt, "status_code": code,
            "error": err,
            "answer": (body or {}).get("answer") if isinstance(body, dict) else None,
            "abstained": (body or {}).get("abstained") if isinstance(body, dict) else None,
        }
        out.append(rec)
        if err is not None and i == 0:
            raise RuntimeError(f"first turn to {endpoint!r} failed transport-level: {err}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="webapp base URL")
    ap.add_argument("--brain", default="tiny-demo")
    ap.add_argument("--renderer", default="qwen", help="renderer under test ('qwen'/'stub'/'raw')")
    ap.add_argument("--n-turns", type=int, default=len(_DEFAULT_MESSAGES),
                     help="number of scripted turns (cycles _DEFAULT_MESSAGES if larger)")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request HTTP timeout (s)")
    ap.add_argument("--skip-generation-isolation", action="store_true",
                     help="skip the paired renderer='raw' session (halves request count; no isolated span)")
    ap.add_argument("--session-prefix", default="latprobe")
    ap.add_argument("--out", default=None, help="write the JSON report here (else stdout)")
    args = ap.parse_args(argv)

    messages = [_DEFAULT_MESSAGES[i % len(_DEFAULT_MESSAGES)] for i in range(args.n_turns)]
    run_id = uuid.uuid4().hex[:8]

    target_session = f"{args.session_prefix}_{args.renderer}_{run_id}"
    t0 = time.time()
    target_turns = _run_session(args.url, target_session, args.brain, args.renderer, messages, args.timeout)

    raw_turns = None
    if not args.skip_generation_isolation and args.renderer != "raw":
        raw_session = f"{args.session_prefix}_raw_{run_id}"
        raw_turns = _run_session(args.url, raw_session, args.brain, "raw", messages, args.timeout)

    target_times = [t["elapsed_s"] for t in target_turns if t["status_code"] == 200]
    cold_first_turn_s = target_turns[0]["elapsed_s"] if target_turns else None
    warm_times = target_times[1:] if len(target_times) > 1 else []

    report: dict[str, Any] = {
        "tool": "tools/chat_latency_probe.py",
        "generated_at": time.time(),
        "args": vars(args),
        "run_id": run_id,
        "n_turns_requested": args.n_turns,
        "n_turns_ok": len(target_times),
        "n_turns_failed": len(target_turns) - len(target_times),
        "cold_first_turn_s": cold_first_turn_s,
        "warm_p50_s": _percentile(warm_times, 0.50),
        "warm_p95_s": _percentile(warm_times, 0.95),
        "warm_mean_s": statistics.mean(warm_times) if warm_times else None,
        "wall_clock_total_s": time.time() - t0,
        "target_turns": target_turns,
    }

    if raw_turns is not None:
        raw_times = [t["elapsed_s"] for t in raw_turns if t["status_code"] == 200]
        raw_warm = raw_times[1:] if len(raw_times) > 1 else []
        # per-turn isolated span: only over turns BOTH sides answered 200 (index-aligned — both
        # sessions run the identical scripted `messages` list in the same order).
        spans = []
        for a, b in zip(target_turns[1:], raw_turns[1:]):  # skip turn 0 (cold build cost dominates)
            if a["status_code"] == 200 and b["status_code"] == 200:
                spans.append(a["elapsed_s"] - b["elapsed_s"])
        report["raw_baseline"] = {
            "warm_p50_s": _percentile(raw_warm, 0.50),
            "warm_p95_s": _percentile(raw_warm, 0.95),
            "raw_turns": raw_turns,
        }
        report["isolated_generation_span"] = {
            "note": ("target_turn_elapsed_s - raw_turn_elapsed_s, index-aligned, turn 0 excluded "
                     "(cold-build cost dominates turn 0 on both sides and would swamp the span)"),
            "per_turn_s": spans,
            "p50_s": _percentile(spans, 0.50),
            "p95_s": _percentile(spans, 0.95),
        }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"[chat_latency_probe] wrote {out_path} "
              f"(cold={report['cold_first_turn_s']}, warm_p50={report['warm_p50_s']}, "
              f"warm_p95={report['warm_p95_s']})", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
