"""Stage-1 parser-collision DIAGNOSTIC (root-cause, not a fix).

The Stage-1 no-regression gate crashed at `agent.hear("dog chase cat")` with `KeyError: 'action'`. `hear()` does an
UNSAFE `roles["action"]`, and `parse_on_slices` builds `roles` as a dict-comprehension keyed by the DECODED role
(nav_conv_merged_bridge.py:163) — so if two of the three positions decode to the SAME role, a key goes missing and the
unsafe access crashes. The decode is a dt=1.0 WTA read (`role_of_on_slices`), and the parser is purely POSITION-based
(word identity never drives the read), so a collision is a read-out tie, NOT a weight/wiring change.

This probe gathers the decisive evidence WITHOUT a fix and WITHOUT nav episodes (fast):
  - build the merged bridge gen-OFF and gen-ON at the same seed;
  - on each, decode every position of BOTH "dog chase cat" (the failing sentence) and "dog go north" (the shipped-test
    sentence), printing per-role accumulated rates + the winner + whether the 3 positions collide;
  - 3 trials per (sentence, settle) to test determinism, at the default settle (reset=60, test=80) and a long settle
    (reset=200, test=160) to test whether more pre-read quiescence resolves a marginal tie.

Verdict logic (printed): is the collision gen-SPECIFIC (only gen-ON) or pre-existing (both)? deterministic or flaky?
fixed by longer settle (→ residual-state / tie) or structural (→ persists)?

Run: SIM_BACKEND=cupy python -m research.runners._unified_stage1_parser_diag --seed 42
"""
import argparse
import json
import time

import numpy as np

from sim.backend import get_backend, to_host
from research.runners.nav_conv_merged_bridge import (
    build_merged_nav_conv_bridge, _step_reset, ROLES,
)

SENTENCES = [["dog", "chase", "cat"], ["dog", "go", "north"]]
SETTLES = [("default", 60, 80), ("long", 200, 160)]
DRIVE = 2500.0


def role_of_with_rates(bridge, conj_arr, role_arr, position, voice, test_steps, drive, reset):
    """role_of_on_slices, but also returns the per-role accumulated rates (so we can see the margin)."""
    xp, _ = get_backend()
    n = bridge.core_config.num_neurons
    k = position * 2 + (0 if voice in (0, "active") else 1)
    _step_reset(bridge, reset)
    cur = xp.zeros(n, dtype=xp.float32)
    cur[conj_arr[k]] = drive
    bridge.cp_external_input_current[:] = cur
    rates = {r: 0.0 for r in ROLES}
    for _ in range(test_steps):
        bridge._run_one_simulation_step()
        for r in ROLES:
            rates[r] += float(to_host(bridge.cp_firing_states[role_arr[r]].astype(xp.float64).mean()))
    bridge.cp_external_input_current[:] = 0.0
    winner = max(rates, key=rates.get)
    return winner, rates


def parse_diag(bridge, conj_arr, role_arr, words, test_steps, reset, voice="active"):
    per_pos = []
    for pos in range(3):
        winner, rates = role_of_with_rates(bridge, conj_arr, role_arr, pos, voice, test_steps, DRIVE, reset)
        per_pos.append((pos, words[pos], winner, rates))
    decoded = [p[2] for p in per_pos]
    collision = len(set(decoded)) < 3
    roles = {p[2]: p[1] for p in per_pos}     # the EXACT dict parse_on_slices builds (may drop a key)
    return per_pos, decoded, collision, roles


def run_one_bridge(seed, gen_flag):
    print(f"\n[diag] ===== building merged bridge gen={'ON' if gen_flag else 'OFF'} (seed {seed}) =====", flush=True)
    t0 = time.time()
    bridge, h = build_merged_nav_conv_bridge(
        seed=seed, co_resident_rf=True, co_resident_perception=True, enable_spiking_wta_readout=True,
        co_resident_generalization=gen_flag)
    conj_arr, role_arr = h["conj_arr"], h["role_arr"]
    print(f"[diag] built in {time.time()-t0:.0f}s | num_neurons={bridge.core_config.num_neurons}", flush=True)

    out = {"gen": gen_flag, "results": []}
    for words in SENTENCES:
        for label, reset, test_steps in SETTLES:
            trials = []
            for trial in range(3):
                per_pos, decoded, collision, roles = parse_diag(bridge, conj_arr, role_arr, words, test_steps, reset)
                ratestr = " | ".join(
                    f"pos{pos}({w}) -> {win}  [" + ",".join(f"{r}:{rates[r]:.2f}" for r in ROLES) + "]"
                    for pos, w, win, rates in per_pos)
                col = "COLLISION" if collision else "ok"
                has_action = "action" in roles
                print(f"[diag] gen={'ON ' if gen_flag else 'OFF'} '{' '.join(words)}' settle={label:<7} "
                      f"trial{trial}: decoded={decoded} {col} action_key={has_action}", flush=True)
                print(f"         {ratestr}", flush=True)
                trials.append({"decoded": decoded, "collision": collision, "has_action": has_action,
                               "rates": [{r: round(rates[r], 3) for r in ROLES} for _, _, _, rates in per_pos]})
            out["results"].append({"sentence": " ".join(words), "settle": label, "reset": reset,
                                   "test_steps": test_steps, "trials": trials})
    del bridge
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/_unified_stage1_parser_diag.json")
    args = ap.parse_args()

    _, backend = get_backend()
    print(f"[diag] backend={backend} seed={args.seed}", flush=True)
    results = [run_one_bridge(args.seed, gen_flag) for gen_flag in (False, True)]

    # ── verdict synthesis ──
    def any_collision(res):
        return any(t["collision"] for r in res["results"] for t in r["trials"])

    def collision_by(res, sentence, settle):
        rows = [r for r in res["results"] if r["sentence"] == sentence and r["settle"] == settle]
        return any(t["collision"] for r in rows for t in r["trials"]) if rows else None

    gen_off, gen_on = results[0], results[1]
    chase = "dog chase cat"
    summary = {
        "seed": args.seed, "backend": backend,
        "gen_off_any_collision": any_collision(gen_off),
        "gen_on_any_collision": any_collision(gen_on),
        "chase_gen_off_default": collision_by(gen_off, chase, "default"),
        "chase_gen_on_default": collision_by(gen_on, chase, "default"),
        "chase_gen_off_long": collision_by(gen_off, chase, "long"),
        "chase_gen_on_long": collision_by(gen_on, chase, "long"),
    }
    # interpret
    gen_specific = bool(summary["chase_gen_on_default"]) and not bool(summary["chase_gen_off_default"])
    long_fixes = bool(summary["chase_gen_on_default"]) and not bool(summary["chase_gen_on_long"])
    summary["interpretation"] = {
        "gen_specific": gen_specific,
        "long_settle_resolves": long_fixes,
        "note": ("gen-specific residual/tie -> longer pre-read quiescence is the principled fix"
                 if (gen_specific and long_fixes) else
                 "pre-existing dt=1.0 read tie (not gen-specific)" if not gen_specific else
                 "structural (longer settle does not resolve) -> investigate weights/wiring"),
    }
    print(f"\n[diag] SUMMARY {json.dumps(summary, indent=2)}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"[diag] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
