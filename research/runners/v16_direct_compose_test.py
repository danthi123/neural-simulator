"""Architectural test: drive verb_pool DIRECTLY (bypassing Phase 1
lang_input -> verb_pool pathway) and measure motor pool firing.

This isolates the v16 verb_pool -> motor compositional pathway from
the Phase 1 binding bottleneck. If TRUE mapping is reliably ranked
1/24 on this test, the v16 + compose-training architecture works at
the pathway level — the BOUNDARY result on lang_input-driven tests
is purely a Phase 1 binding artifact.
"""
from __future__ import annotations
import argparse
import itertools
import json

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import (
    _WORD_TO_IDX, _WORD_TO_POOL,
)


def drive_pool_directly_measure_motors(bridge, verb_pool_name,
                                          drive_pA=1500.0, stim_steps=100):
    """Drive a verb_pool with strong external current, measure each
    motor pool's spike count over the stimulus window."""
    from sim.backend import get_backend
    cp, _ = get_backend()

    rm = bridge.region_manager
    verb_idx = list(rm.indices(verb_pool_name))
    verb_arr_gpu = cp.asarray(verb_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    motor_pools = ["motor_N", "motor_E", "motor_S", "motor_W"]
    motor_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                   for p in motor_pools}

    spike_counts = {p: 0 for p in motor_pools}

    # Reset bridge external input + free run briefly
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()

    # Drive verb_pool, count motor spikes
    for _ in range(stim_steps):
        ext = cp.zeros(n_total, dtype=cp.float32)
        ext[verb_arr_gpu] = drive_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        # Count spikes in motor pools (use cp_firing_states or recent fired)
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            for p, arr in motor_arrs.items():
                spike_counts[p] += int(firing[arr].sum())

    # Normalize to per-neuron rate (spikes / step / neuron)
    rates = {p: spike_counts[p] / (stim_steps * len(motor_arrs[p]))
              for p in motor_pools}
    return rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--drive-pA", type=float, default=1500.0)
    p.add_argument("--stim-steps", type=int, default=100)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

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

    print(f"[DIRECT TEST] Driving verb pools directly, measuring motor firing")
    print(f"  Bridge: {args.load_bridge}")
    print(f"  Drive: {args.drive_pA} pA for {args.stim_steps} steps")
    print()

    # Drive each verb pool, measure all motor pools
    verb_pools = ["verb_pool_GO", "verb_pool_COME", "verb_pool_STOP",
                  "verb_pool_LOOK"]
    verb_words = ["go", "come", "stop", "look"]
    motor_words = ["north", "south", "west", "east"]
    true_mapping = {"go": "north", "come": "south", "stop": "west", "look": "east"}

    firing = {}
    print(f"  {'verb_pool':18s} {'motor_N':10s} {'motor_E':10s} {'motor_S':10s} {'motor_W':10s}")
    for verb_word, verb_pool in zip(verb_words, verb_pools):
        rates = drive_pool_directly_measure_motors(
            bridge, verb_pool, drive_pA=args.drive_pA,
            stim_steps=args.stim_steps,
        )
        firing[verb_word] = rates
        print(f"  {verb_pool:18s} {rates['motor_N']:.3f}      "
              f"{rates['motor_E']:.3f}      {rates['motor_S']:.3f}      "
              f"{rates['motor_W']:.3f}")

    # PASS test: for each verb, does it preferentially fire its trained motor?
    print()
    n_pass = 0
    for verb in verb_words:
        target_motor = _WORD_TO_POOL[true_mapping[verb]]
        rates = firing[verb]
        target_rate = rates[target_motor]
        max_off = max(v for k, v in rates.items() if k != target_motor)
        passed = target_rate > max_off
        if passed:
            n_pass += 1
        marker = "PASS" if passed else "FAIL"
        ratio = target_rate / max(max_off, 0.001)
        print(f"  verb_pool {verb:6s} -> {target_motor:10s}  "
              f"target={target_rate:.3f}  off={max_off:.3f}  "
              f"ratio={ratio:.2f}x  [{marker}]")
    print()
    print(f"[VERDICT] {n_pass}/{len(verb_words)} compose pairs PASS (direct-drive test)")

    # Anti-cheat: 24 permutations
    print()
    print(f"[ANTI-CHEAT] 24 permutations")
    perm_results = []
    for motor_perm in itertools.permutations(motor_words):
        mapping = list(zip(verb_words, motor_perm))
        passes = 0
        for verb, motor in mapping:
            tgt = _WORD_TO_POOL[motor]
            rates = firing[verb]
            target_rate = rates[tgt]
            max_off = max(v for k, v in rates.items() if k != tgt)
            if target_rate > max_off:
                passes += 1
        is_true = all(motor == true_mapping[verb] for verb, motor in mapping)
        perm_results.append({"mapping": mapping, "n_pass": passes,
                              "is_true": is_true})
    perm_results.sort(key=lambda r: -r["n_pass"])
    true_rank = next(i for i, r in enumerate(perm_results, start=1)
                      if r["is_true"])
    true_pass = next(r["n_pass"] for r in perm_results if r["is_true"])
    print(f"  Top permutations:")
    for rank, r in enumerate(perm_results[:5], start=1):
        m = ", ".join(f"{v}->{m}" for v, m in r["mapping"])
        tag = "** TRUE **" if r["is_true"] else ""
        print(f"    {rank:3d} {m:50s} {r['n_pass']}/4 {tag}")
    print()
    print(f"  TRUE mapping rank: {true_rank}/24 (chance=12.5/24)")
    print(f"  TRUE pass: {true_pass}/4")
    print(f"  Best perm pass: {perm_results[0]['n_pass']}/4")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "bridge": args.load_bridge,
                "seed": args.seed,
                "drive_pA": args.drive_pA,
                "stim_steps": args.stim_steps,
                "firing": firing,
                "n_pass": n_pass,
                "true_rank": true_rank,
                "true_pass": true_pass,
                "max_pass": perm_results[0]["n_pass"],
            }, f, indent=2, default=str)
        print(f"[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
