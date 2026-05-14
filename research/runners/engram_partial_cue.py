"""Partial-cue recall test for engram-composition.

Tests whether engram tagging produces Marr-like pattern completion:
encode "go_north" engram, then drive ONLY the verb_pool subset of the
tag (not the full tag). Measure motor pool firing.

If motor_N fires preferentially via natural bridge dynamics, the
mechanism is pattern completion (D.13 + D.14 combined) — driving
the verb fragment of an episodic memory reactivates the bound motor
through the bridge's existing recurrent connectivity.

If motor_N doesn't fire preferentially, the engram requires full
stimulation to retrieve — episodic memory exists but no pattern
completion. Still useful (engram-based composition validates), but
less biologically powerful.
"""
from __future__ import annotations
import argparse
import json
import time

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_engram_demo import encode_compose_pair


def stimulate_subset_and_measure(bridge, tag_neurons, region_to_measure,
                                    drive_pA=1500.0, stim_steps=100):
    """Drive a SUBSET of tag neurons (those in region_to_measure exclusion)
    and measure motor firing.
    """
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager

    # Get neurons in each motor pool
    motor_pools = ["motor_N", "motor_E", "motor_S", "motor_W"]
    motor_indices = {p: set(rm.indices(p)) for p in motor_pools}
    motor_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                   for p in motor_pools}

    # Partition tag neurons: those in motor regions vs those not in motor
    # (=> in verb_pool/noun_pool/adjective_pool/etc)
    tag_host = to_host(tag_neurons)
    motor_neuron_set = set()
    for ms in motor_indices.values():
        motor_neuron_set |= ms
    non_motor_tag_neurons = [int(n) for n in tag_host
                              if int(n) not in motor_neuron_set]
    non_motor_arr = cp.asarray(non_motor_tag_neurons, dtype=cp.int64)

    if len(non_motor_tag_neurons) == 0:
        return {"error": "all tag neurons were in motor regions"}, 0

    spike_counts = {p: 0 for p in motor_pools}
    n_total = bridge.cp_external_input_current.shape[0]

    # Reset bridge state
    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()

    # Drive ONLY the non-motor subset of the engram tag
    for _ in range(stim_steps):
        ext = cp.zeros(n_total, dtype=cp.float32)
        ext[non_motor_arr] = drive_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            for p, arr in motor_arrs.items():
                spike_counts[p] += int(firing[arr].sum())

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    rates = {p: spike_counts[p] / (stim_steps * len(motor_arrs[p]))
              for p in motor_pools}
    return rates, len(non_motor_tag_neurons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--compose-pairs", type=str,
                    default="go:north,come:south,stop:west,look:east")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-pA", type=float, default=1500.0)
    p.add_argument("--stim-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = [tuple(s.strip().split(":")) for s in args.compose_pairs.split(",")]

    print(f"=== engram_partial_cue (seed={args.seed}) ===")
    print(f"  Tests pattern-completion: drive ONLY non-motor subset of engram")
    print(f"  If motor still fires for TRUE pair, that's Marr-style completion")
    print()

    bridge = cpd.build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        enable_adjective=True,
        weak_dynamics=True,
        enable_direct_verb_to_motor=True,
        verbose=False,
    )
    bridge.load_checkpoint(args.load_bridge)

    region_filter_full = (
        [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
        + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
        + [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]]
        + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
    )

    # Phase 1: ENCODE engrams
    print("[ENCODE] Tagging compose pairs as engrams...")
    tag_names = []
    for verb, motor in pairs:
        tag_name = f"{verb}_{motor}"
        tag_names.append(tag_name)
        encode_compose_pair(
            bridge, verb, motor, tag_name,
            encoding_steps=args.encoding_steps,
            drive_pA=200.0,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            region_filter=region_filter_full,
            top_k=args.top_k,
            verbose=True,
        )
    print()

    # Phase 2: PARTIAL-CUE RECALL — drive non-motor subset only
    print("[PARTIAL-CUE] Stimulating ONLY non-motor portion of tag")
    print(f"  {'tag':18s} {'n_non_motor':12s} {'motor_N':10s} {'motor_E':10s} {'motor_S':10s} {'motor_W':10s}")
    results = []
    n_pass = 0
    for tag_name, (verb, motor) in zip(tag_names, pairs):
        indices_gpu = bridge.get_engram_tag_indices(tag_name)
        rates, n_non = stimulate_subset_and_measure(
            bridge, indices_gpu, motor,
            drive_pA=args.drive_pA, stim_steps=args.stim_steps,
        )
        if isinstance(rates, dict) and "error" in rates:
            print(f"  {tag_name:18s} ERROR: {rates['error']}")
            continue
        target_pool = _WORD_TO_POOL[motor]
        target_rate = rates[target_pool]
        off = max(v for k, v in rates.items() if k != target_pool)
        passed = target_rate > off
        if passed:
            n_pass += 1
        marker = "PASS" if passed else "FAIL"
        print(f"  {tag_name:18s} {n_non:12d} {rates['motor_N']:.3f}     "
              f"{rates['motor_E']:.3f}     {rates['motor_S']:.3f}     "
              f"{rates['motor_W']:.3f}    [{marker}]")
        results.append({
            "tag": tag_name, "verb": verb, "motor": motor,
            "n_non_motor": n_non,
            "rates": rates,
            "target_rate": target_rate, "max_off": off,
            "passed": passed, "ratio": target_rate / max(off, 0.001),
        })

    print()
    print(f"[VERDICT] {n_pass}/{len(pairs)} partial-cue recalls PASS "
          f"(motor fires from non-motor subset stimulation)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed, "load_bridge": args.load_bridge,
                "encoding_steps": args.encoding_steps,
                "top_k": args.top_k,
                "drive_pA": args.drive_pA,
                "stim_steps": args.stim_steps,
                "results": results,
                "n_pass": n_pass,
                "n_total": len(pairs),
            }, f, indent=2, default=str)
        print(f"[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
