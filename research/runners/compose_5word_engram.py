"""5-word phrase composition at v17: verb + adj1 + adj2 + noun + motor.

Tests 5-way conjunction engrams like "walk big red apple east".
With v17's 28-word vocab, we have enough adjectives to do dual-adj
phrases.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

# Patch v17 vocab into v1 module before any compose-related import
import research.runners.compose_engram_demo_v2  # noqa: F401 (triggers patch)

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from sim.text_embeddings import orthogonal_drive_pattern


def encode_5way(bridge, verb, adj1, adj2, noun, motor, tag_name,
                  encoding_steps=200, drive_pA=200.0, sparsity=0.03,
                  n_lang_input=4096, n_words_for_orthogonal=28,
                  region_filter=None, top_k=400, motor_teacher_pA=750.0,
                  verbose=True):
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    drives = [orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[w], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    ) for w in [verb, adj1, adj2, noun, motor]]
    combined_gpu = cp.asarray(sum(drives), dtype=cp.float32)

    lang_arr_gpu = cp.asarray(
        list(rm.indices("language_input")), dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    if motor_teacher_pA > 0:
        motor_target_idx = list(rm.indices(_WORD_TO_POOL[motor]))
        motor_target_arr_gpu = cp.asarray(motor_target_idx, dtype=cp.int64)

    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr_gpu] = combined_gpu
        if motor_teacher_pA > 0:
            ext[motor_target_arr_gpu] = motor_teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    stats = bridge.commit_engram_tag(tag_name, top_k=top_k,
                                       region_filter=region_filter)
    if verbose:
        print(f"  [{tag_name}] tagged {stats['n_tagged']} neurons")
    return stats


def recall_5way(bridge, tag_name, target_pools_to_check,
                  drive_pA=1500.0, stim_steps=100):
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    target_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                    for p in target_pools_to_check}
    spike_counts = {p: 0 for p in target_pools_to_check}

    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()

    bridge.stimulate_tag(tag_name, drive_pA=drive_pA, additive=False)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            for p, arr in target_arrs.items():
                spike_counts[p] += int(firing[arr].sum())

    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    rates = {p: spike_counts[p] / (stim_steps * len(target_arrs[p]))
              for p in target_pools_to_check}
    return rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=4096)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--quintets", type=str,
                    default="walk:big:red:apple:north,run:small:blue:tree:east,eat:hot:fast:dog:south,sleep:cold:slow:bird:west",
                    help="verb:adj1:adj2:noun:motor")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=400)
    p.add_argument("--motor-teacher-pA", type=float, default=750.0)
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.03)
    p.add_argument("--n-words-for-orthogonal", type=int, default=28)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    quintets = []
    for q in args.quintets.split(","):
        parts = q.strip().split(":")
        if len(parts) != 5:
            print(f"ERROR: need 5 words per phrase, got {q}")
            return
        quintets.append(tuple(parts))

    print(f"=== compose_5word_engram (seed={args.seed}) ===")
    for q in quintets:
        print(f"  {q}")
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

    region_filter = (
        [f"verb_pool_{v.upper()}" for v in
         ["GO", "COME", "STOP", "LOOK", "WALK", "RUN", "EAT", "SLEEP"]]
        + [f"noun_pool_{n.upper()}" for n in
            ["APPLE", "RIVER", "DOG", "CAT", "TREE", "BIRD", "SUN", "MOON"]]
        + [f"adjective_pool_{a.upper()}" for a in
            ["BIG", "SMALL", "HOT", "COLD", "RED", "BLUE", "FAST", "SLOW"]]
        + [f"motor_{m}" for m in ["N", "E", "S", "W"]]
    )

    print("[ENCODE] 5-way conjunction engrams")
    for verb, adj1, adj2, noun, motor in quintets:
        tag = f"{verb}_{adj1}_{adj2}_{noun}_{motor}"
        encode_5way(bridge, verb, adj1, adj2, noun, motor, tag,
                     encoding_steps=args.encoding_steps,
                     drive_pA=200.0, sparsity=args.sparsity,
                     n_lang_input=args.n_lang_input,
                     n_words_for_orthogonal=args.n_words_for_orthogonal,
                     region_filter=region_filter,
                     top_k=args.top_k,
                     motor_teacher_pA=args.motor_teacher_pA,
                     verbose=True)

    print()
    print("[RECALL] Stimulate each tag, check ALL 5 pools fire")
    n_motor = 0
    n_full = 0
    results = []
    for verb, adj1, adj2, noun, motor in quintets:
        tag = f"{verb}_{adj1}_{adj2}_{noun}_{motor}"
        target_motor = _WORD_TO_POOL[motor]
        target_verb = _WORD_TO_POOL[verb]
        target_adj1 = _WORD_TO_POOL[adj1]
        target_adj2 = _WORD_TO_POOL[adj2]
        target_noun = _WORD_TO_POOL[noun]

        # All 28 output pools + 4 motors = 32 pools to check
        all_pools = region_filter
        rates = recall_5way(bridge, tag, all_pools,
                              drive_pA=args.recall_stim_pA,
                              stim_steps=args.recall_steps)

        m_t = rates[target_motor]
        m_o = max(v for k, v in rates.items()
                    if k.startswith("motor_") and k != target_motor)
        v_t = rates[target_verb]
        v_o = max(v for k, v in rates.items()
                    if k.startswith("verb_pool_") and k != target_verb)
        # Both adj targets — check each against off (other adj pools)
        a1_t = rates[target_adj1]
        a1_o = max(v for k, v in rates.items()
                     if k.startswith("adjective_pool_")
                     and k not in (target_adj1, target_adj2))
        a2_t = rates[target_adj2]
        a2_o = max(v for k, v in rates.items()
                     if k.startswith("adjective_pool_")
                     and k not in (target_adj1, target_adj2))
        n_t = rates[target_noun]
        n_o = max(v for k, v in rates.items()
                    if k.startswith("noun_pool_") and k != target_noun)

        motor_pass = m_t > m_o
        full_pass = motor_pass and v_t > v_o and a1_t > a1_o and a2_t > a2_o and n_t > n_o
        if motor_pass:
            n_motor += 1
        if full_pass:
            n_full += 1
        marker = "FULL" if full_pass else ("MOTOR" if motor_pass else "FAIL")
        print(f"  {tag[:50]:50s} M={m_t:.2f}/{m_o:.2f} V={v_t:.2f}/{v_o:.2f} "
              f"A1={a1_t:.2f}/{a1_o:.2f} A2={a2_t:.2f}/{a2_o:.2f} "
              f"N={n_t:.2f}/{n_o:.2f} [{marker}]")
        results.append({
            "tag": tag, "rates": rates,
            "motor_pass": motor_pass, "full_pass": full_pass,
        })

    print()
    print(f"[VERDICT] Motor: {n_motor}/{len(quintets)}, "
          f"Full 5-way: {n_full}/{len(quintets)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seed": args.seed, "quintets": quintets,
                        "n_motor_pass": n_motor, "n_full_pass": n_full,
                        "n_total": len(quintets), "results": results},
                       f, indent=2, default=str)


if __name__ == "__main__":
    main()
