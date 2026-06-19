"""Iteration harness for #5 place-code SPARSIFY (2026-06-19).

Goal: make the self-organized spiking `place` code SPARSE (<=~10% active, cos(near,far)<0.3)
so the STEP-2 value-train produces a GRADED V (near >> far) -> delta gap > host-Gaussian 1.3.

It captures the EXACT deployed kwargs of the NEGATIVE-repro config by monkeypatching
`run_moving_goal_episode` inside `main()`, then re-invokes the real function with our
sparsity-lever overrides applied. Runs `stage_a_smoke` (fast, ~9s, sparsity+cos) by default;
pass --stage-b to additionally run the full value-train delta probe (~95s).

Usage (numpy CPU, deterministic):
  SIM_BACKEND=numpy python -m research.runners._n5_place_sparsify_probe \
      --overrides place_sensors_to_place_weight=14,place_fs_open_during_selforg=1 [--stage-b]
"""
import os
import sys
import json
import argparse
import importlib

import research.runners.g11_bg_runner as g


def _parse_overrides(s):
    out = {}
    if not s:
        return out
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip(); v = v.strip()
        # type coercion: int, float, then str
        try:
            if "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            else:
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", type=str, default="",
                    help="comma-sep k=v overrides applied to the captured deployed kwargs")
    ap.add_argument("--stage-b", action="store_true",
                    help="run the full STEP-2 value-train delta probe (~95s) instead of stage-A only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--value-train-trials", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    cli, _ = ap.parse_known_args()

    overrides = _parse_overrides(cli.overrides)

    captured = {}

    real_fn = g.run_moving_goal_episode

    def _intercept(*args, **kwargs):
        captured.update(kwargs)
        # don't actually run the deployed path; we re-invoke ourselves below.
        return {"_intercepted": True}

    # Build the deployed kwargs by running main() with the NEGATIVE-repro argv,
    # intercepting the call so nothing heavy runs.
    g.run_moving_goal_episode = _intercept
    argv = [
        "g11", "--moving-goal", "--goal-schedule", "multi", "--deterministic",
        "--enable-neural-critic", "--spiking-reward-us", "--enable-critic-homeostasis",
        "--enable-critic-fs-inhibition", "--critic-fs-weight", "16",
        "--neural-place-selforg", "--deterministic-selforg",
        ("--stage-b-smoke" if cli.stage_b else "--stage-a-smoke"),
        "--value-train-trials", str(cli.value_train_trials),
        "--seed", str(cli.seed),
        "--no-emit-webapp-sidecar",
    ]
    saved_argv = sys.argv
    try:
        sys.argv = argv
        g.main()
    finally:
        sys.argv = saved_argv
        g.run_moving_goal_episode = real_fn

    if not captured:
        print("ERROR: failed to capture deployed kwargs", flush=True)
        sys.exit(2)

    # Apply our sparsity-lever overrides.
    for k, v in overrides.items():
        if k not in captured:
            print(f"[probe] WARN override key not in deployed kwargs: {k} (adding anyway)", flush=True)
        captured[k] = v

    print("=" * 72, flush=True)
    print(f"[probe] seed={cli.seed} stage={'B' if cli.stage_b else 'A'} overrides={overrides}", flush=True)
    print("=" * 72, flush=True)

    result = real_fn(**captured)

    if cli.out:
        os.makedirs(os.path.dirname(cli.out), exist_ok=True)
        # strip non-JSON-friendly objects
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items() if not (isinstance(k, str) and k.startswith("_"))}
            if isinstance(o, (list, tuple)):
                return [_clean(v) for v in o]
            return o
        with open(cli.out, "w") as f:
            json.dump({"overrides": overrides, "result": _clean(result)}, f, indent=2, default=str)
        print(f"[probe] wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
