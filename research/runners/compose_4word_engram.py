"""4-word phrase composition test: verb + adj + noun + motor.

Each phrase like "go big apple north" encoded as a single engram.
Tests whether the engram mechanism scales to 4-way conjunctions.
"""
from __future__ import annotations
import argparse
import json

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from sim.text_embeddings import orthogonal_drive_pattern


def encode_4way(bridge, verb_word, adj_word, noun_word, motor_word, tag_name,
                  encoding_steps=200, drive_pA=200.0, sparsity=0.05,
                  n_lang_input=2048, n_words_for_orthogonal=16,
                  region_filter=None, top_k=300,
                  motor_teacher_pA=750.0, verbose=True):
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    drives = [orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[w], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    ) for w in [verb_word, adj_word, noun_word, motor_word]]
    combined_gpu = cp.asarray(sum(drives), dtype=cp.float32)

    lang_input_idx = list(rm.indices("language_input"))
    lang_arr_gpu = cp.asarray(lang_input_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    if motor_teacher_pA > 0:
        motor_target_idx = list(rm.indices(_WORD_TO_POOL[motor_word]))
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


def recall_4way(bridge, tag_name, drive_pA=1500.0, stim_steps=100):
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    target_pools = (
        [f"motor_{a}" for a in ["N", "E", "S", "W"]]
        + [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
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
    p = argparse.ArgumentParser(description="4-word phrase composition test")
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--quads", type=str,
                    default="go:big:apple:north,come:small:cat:south,stop:hot:dog:west,look:cold:river:east")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=300)
    p.add_argument("--motor-teacher-pA", type=float, default=750.0)
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    quads = []
    for q in args.quads.split(","):
        parts = q.strip().split(":")
        if len(parts) != 4:
            print(f"ERROR: quad must be verb:adj:noun:motor, got {q}")
            return
        quads.append(tuple(parts))

    print(f"=== compose_4word_engram (seed={args.seed}) ===")
    print(f"  Quads: {quads}")
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
        [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
        + [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]]
        + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
        + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
    )

    print("[ENCODE] 4-way conjunction engrams")
    for verb, adj, noun, motor in quads:
        tag = f"{verb}_{adj}_{noun}_{motor}"
        encode_4way(bridge, verb, adj, noun, motor, tag,
                     encoding_steps=args.encoding_steps,
                     drive_pA=200.0, sparsity=args.sparsity,
                     n_lang_input=args.n_lang_input,
                     region_filter=region_filter,
                     top_k=args.top_k,
                     motor_teacher_pA=args.motor_teacher_pA,
                     verbose=True)

    print()
    print("[RECALL] Stimulate each tag, check ALL 4 pools fire")
    n_motor = 0
    n_full = 0
    results = []
    for verb, adj, noun, motor in quads:
        tag = f"{verb}_{adj}_{noun}_{motor}"
        rates = recall_4way(bridge, tag,
                              drive_pA=args.recall_stim_pA,
                              stim_steps=args.recall_steps)
        target = {
            "motor": _WORD_TO_POOL[motor],
            "verb": _WORD_TO_POOL[verb],
            "adj": _WORD_TO_POOL[adj],
            "noun": _WORD_TO_POOL[noun],
        }

        def kind_check(kind_prefix):
            tgt_pool = target[kind_prefix if kind_prefix in target else 'motor']
            return rates[tgt_pool], max(v for k, v in rates.items()
                                          if k.startswith(kind_prefix)
                                          and k != tgt_pool)

        m_t, m_o = rates[target["motor"]], max(v for k, v in rates.items()
                                                  if k.startswith("motor_")
                                                  and k != target["motor"])
        v_t, v_o = rates[target["verb"]], max(v for k, v in rates.items()
                                                if k.startswith("verb_pool_")
                                                and k != target["verb"])
        a_t, a_o = rates[target["adj"]], max(v for k, v in rates.items()
                                                if k.startswith("adjective_pool_")
                                                and k != target["adj"])
        n_t, n_o = rates[target["noun"]], max(v for k, v in rates.items()
                                                if k.startswith("noun_pool_")
                                                and k != target["noun"])

        motor_pass = m_t > m_o
        full_pass = motor_pass and (v_t > v_o) and (a_t > a_o) and (n_t > n_o)

        if motor_pass:
            n_motor += 1
        if full_pass:
            n_full += 1

        marker = "FULL" if full_pass else ("MOTOR" if motor_pass else "FAIL")
        print(f"  {tag:32s} M={m_t:.2f}/{m_o:.2f} V={v_t:.2f}/{v_o:.2f} "
              f"A={a_t:.2f}/{a_o:.2f} N={n_t:.2f}/{n_o:.2f} [{marker}]")
        results.append({
            "tag": tag, "rates": rates,
            "motor_pass": motor_pass, "full_pass": full_pass,
        })

    print()
    print(f"[VERDICT] Motor: {n_motor}/{len(quads)}, "
          f"Full 4-way: {n_full}/{len(quads)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seed": args.seed, "quads": quads,
                        "n_motor_pass": n_motor, "n_full_pass": n_full,
                        "n_total": len(quads),
                        "results": results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
