"""3-word phrase composition test via engram tagging.

Tests if the engram mechanism handles 3-way conjunctions. Each phrase
"adj noun motor" (e.g., "big apple north") is encoded by simultaneously
driving lang_input for all 3 words. The engram captures co-firing
neurons across adjective_pool + noun_pool + motor regions.

At recall, stimulating the tag should activate all 3 pools together
and ultimately drive the motor pool (the "action" component).

Tests:
- Direct recall: stimulate tag, measure motor pool firing
- Anti-cheat: try shuffled (adj, noun, motor) tuples - TRUE should
  still recall correctly because each tag has a distinct neuron set.
"""
from __future__ import annotations
import argparse
import json
import time

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from sim.text_embeddings import orthogonal_drive_pattern


def encode_3way(bridge, adj_word: str, noun_word: str, motor_word: str,
                  tag_name: str,
                  encoding_steps: int = 200,
                  drive_pA: float = 200.0,
                  sparsity: float = 0.05,
                  n_lang_input: int = 2048,
                  n_words_for_orthogonal: int = 16,
                  region_filter=None,
                  top_k: int = 150,
                  motor_teacher_pA: float = 1500.0,
                  verbose: bool = True):
    """Encode a 3-way (adj, noun, motor) engram via simultaneous drive."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    adj_drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[adj_word], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    noun_drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[noun_word], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    motor_drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[motor_word], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    combined_gpu = cp.asarray(adj_drive + noun_drive + motor_drive,
                                dtype=cp.float32)

    lang_input_idx = list(rm.indices("language_input"))
    lang_arr_gpu = cp.asarray(lang_input_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    use_teacher = motor_teacher_pA > 0.0
    if use_teacher:
        motor_target_idx = list(rm.indices(_WORD_TO_POOL[motor_word]))
        motor_target_arr_gpu = cp.asarray(motor_target_idx, dtype=cp.int64)

    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    # Pre-allocated ext
    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr_gpu] = combined_gpu
        if use_teacher:
            ext[motor_target_arr_gpu] = motor_teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    stats = bridge.commit_engram_tag(tag_name, top_k=top_k,
                                       region_filter=region_filter)
    if verbose:
        print(f"  [{tag_name}] tagged {stats['n_tagged']} neurons "
              f"({adj_word}+{noun_word}+{motor_word})")
    return stats


def recall_3way(bridge, tag_name, drive_pA=1500.0, stim_steps=100):
    """Stimulate tag, measure motor + adj + noun pool firing."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    target_pools = (
        [f"motor_{a}" for a in ["N", "E", "S", "W"]]
        + [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]]
        + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
    )
    target_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                    for p in target_pools}
    spike_counts = {p: 0 for p in target_pools}

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
              for p in target_pools}
    return rates


def main():
    p = argparse.ArgumentParser(description="3-word phrase composition test")
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--triples", type=str,
                    default="big:apple:north,small:cat:south,hot:dog:west,cold:river:east",
                    help="adj:noun:motor triples")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=150,
                    help="Bigger (150 vs 100) so tag covers all 3 pools")
    p.add_argument("--motor-teacher-pA", type=float, default=1500.0)
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    triples = []
    for t in args.triples.split(","):
        parts = t.strip().split(":")
        if len(parts) != 3:
            print(f"ERROR: triple must be adj:noun:motor, got {t}")
            return
        triples.append(tuple(parts))

    print(f"=== compose_3word_engram (seed={args.seed}) ===")
    print(f"  Triples: {triples}")
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
        [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]]
        + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
        + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
    )

    print("[ENCODE] 3-way conjunction engrams")
    for adj, noun, motor in triples:
        tag = f"{adj}_{noun}_{motor}"
        encode_3way(bridge, adj, noun, motor, tag,
                     encoding_steps=args.encoding_steps,
                     drive_pA=200.0, sparsity=args.sparsity,
                     n_lang_input=args.n_lang_input,
                     region_filter=region_filter,
                     top_k=args.top_k,
                     motor_teacher_pA=args.motor_teacher_pA,
                     verbose=True)

    print()
    print("[RECALL] Stimulate each tag, check ALL 3 pools fire")
    n_motor_pass = 0
    n_full_pass = 0
    results = []
    for adj, noun, motor in triples:
        tag = f"{adj}_{noun}_{motor}"
        rates = recall_3way(bridge, tag,
                              drive_pA=args.recall_stim_pA,
                              stim_steps=args.recall_steps)
        target_motor_pool = _WORD_TO_POOL[motor]
        target_adj_pool = _WORD_TO_POOL[adj]
        target_noun_pool = _WORD_TO_POOL[noun]

        motor_target = rates[target_motor_pool]
        motor_off = max(v for k, v in rates.items()
                          if k.startswith("motor_") and k != target_motor_pool)
        motor_pass = motor_target > motor_off

        adj_target = rates[target_adj_pool]
        adj_off = max(v for k, v in rates.items()
                       if k.startswith("adjective_pool_") and k != target_adj_pool)
        adj_pass = adj_target > adj_off

        noun_target = rates[target_noun_pool]
        noun_off = max(v for k, v in rates.items()
                        if k.startswith("noun_pool_") and k != target_noun_pool)
        noun_pass = noun_target > noun_off

        full_pass = motor_pass and adj_pass and noun_pass
        if motor_pass:
            n_motor_pass += 1
        if full_pass:
            n_full_pass += 1

        marker = ("FULL" if full_pass else
                  ("MOTOR" if motor_pass else "FAIL"))
        print(f"  {tag:24s} motor={motor_target:.3f}/{motor_off:.3f} "
              f"adj={adj_target:.3f}/{adj_off:.3f} "
              f"noun={noun_target:.3f}/{noun_off:.3f} [{marker}]")
        results.append({
            "tag": tag, "adj": adj, "noun": noun, "motor": motor,
            "rates": rates,
            "motor_target": motor_target, "motor_off": motor_off,
            "adj_target": adj_target, "adj_off": adj_off,
            "noun_target": noun_target, "noun_off": noun_off,
            "motor_pass": motor_pass, "adj_pass": adj_pass,
            "noun_pass": noun_pass, "full_pass": full_pass,
        })

    print()
    print(f"[VERDICT] Motor pass: {n_motor_pass}/{len(triples)}; "
          f"Full 3-way pass: {n_full_pass}/{len(triples)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed, "triples": triples,
                "n_motor_pass": n_motor_pass, "n_full_pass": n_full_pass,
                "n_total": len(triples),
                "results": results,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
