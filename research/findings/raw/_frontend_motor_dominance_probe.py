"""Front-end wall mechanism + cheap-fix probe (28-word bridge, no retrain).

The cheating audit quantified the front-end wall: 28-word pool-label recognition is 0.571 (vs 0.812 at 16
words). The v17 finding diagnosed the cause as MOTOR-POOL DOMINANCE: 4 motor pools have ~150 concentrated
lang_input weights each vs 24 concept pools with ~60 spread weights, so the motor pools win the argmax for
many concept words. THIS probe tests, at INFERENCE (no retrain), whether DOWN-WEIGHTING the motor pools'
accumulated activity before the argmax lifts pool-label -- distinguishing a CHEAP inference-time fix from a
needs-retrain (owner-strategic) one.

Sweep a motor down-weight factor f in {1.0, 0.5, 0.25, 0.0}; for each, recompute pool-label (argmax over
pools with motor-pool rates scaled by f). If pool-label LIFTS as f drops -> motor dominance is the mechanism
AND a simple inference-time rebalance is a cheap partial fix. If it does NOT lift -> the wall is deeper than
readout dominance (a real retrain/redesign is needed). Also reports, per word, whether the correct pool is
the top NON-MOTOR pool (the cleanest mechanism read).

Reuse-by-import (concept_pool_demo_v2 patches vocab to 28 words; load the existing _v17 bridge). load_checkpoint
validates architecture. No protected-module change; no autograd.

Run (GPU): python -m research.findings.raw._frontend_motor_dominance_probe
"""
from __future__ import annotations
import os
import numpy as np

import research.runners.concept_pool_demo_v2 as v2          # MUST import first: patches vocab to 28 words
import research.runners.concept_pool_demo as cpd            # vocab now patched
from sim.backend import get_backend, to_host

CKPT = "research/findings/raw/_v17_28word_seed42.simstate.h5"
N_LANG = 2048
SPARSITY = 0.03
DRIVE_PA = 200.0
RESET, STIM = 50, 100


def main():
    if not os.path.exists(CKPT):
        print(f"CANNOT-CONCLUDE: {CKPT} not found (run the 28-word training first)", flush=True); return
    from sim.text_embeddings import orthogonal_drive_pattern
    xp, backend = get_backend()
    print(f"=== front-end motor-dominance probe (28 words, backend={backend}) ===", flush=True)

    words = (list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB)
             + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(words)}
    word_to_pool = {}
    for w, v in cpd.DIRECTION_VOCAB.items(): word_to_pool[w] = f"motor_{v}"
    for w, v in cpd.NOUN_VOCAB.items(): word_to_pool[w] = f"noun_pool_{v}"
    for w, v in cpd.VERB_VOCAB.items(): word_to_pool[w] = f"verb_pool_{v}"
    for w, v in cpd.ADJECTIVE_VOCAB.items(): word_to_pool[w] = f"adjective_pool_{v}"
    pools = ([f"motor_{v}" for v in cpd.DIRECTION_VOCAB.values()]
             + [f"noun_pool_{v}" for v in cpd.NOUN_VOCAB.values()]
             + [f"verb_pool_{v}" for v in cpd.VERB_VOCAB.values()]
             + [f"adjective_pool_{v}" for v in cpd.ADJECTIVE_VOCAB.values()])
    motor_pools = set(f"motor_{v}" for v in cpd.DIRECTION_VOCAB.values())

    bridge = cpd.build_concept_bridge(seed=42, n_lang_input=N_LANG, n_per_pool=200, n_fs_per_pool=24,
                                      enable_adjective=True, weak_dynamics=True, verbose=False)
    bridge.load_checkpoint(CKPT)
    rm = bridge.region_manager
    pool_slices = {}; all_idx = []
    for p in pools:
        idx = list(rm.indices(p)); pool_slices[p] = (len(all_idx), len(all_idx) + len(idx)); all_idx += idx
    all_arr = xp.asarray(all_idx, dtype=xp.int64)
    lang_arr = xp.asarray(list(rm.indices("language_input")), dtype=xp.int64)

    def capture(word):
        drive = orthogonal_drive_pattern(cue_idx=word_to_idx[word], n_cues=len(words), n_neurons=N_LANG,
                                         drive_max_pA=DRIVE_PA, sparsity=SPARSITY)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[lang_arr] = xp.asarray(drive, dtype=xp.float32)
        acc = xp.zeros(len(all_idx), dtype=xp.float64)
        for _ in range(STIM):
            bridge._run_one_simulation_step()
            acc += bridge.cp_firing_states[all_arr].astype(xp.float64)
        bridge.cp_external_input_current[:] = 0.0
        a = to_host(acc) / STIM
        return {p: float(a[s:e].mean()) for p, (s, e) in pool_slices.items()}

    rates_per_word = {w: capture(w) for w in words}

    # mechanism read 1: for NON-motor (concept) words, is the correct pool the top NON-MOTOR pool?
    concept_words = [w for w in words if word_to_pool[w] not in motor_pools]
    top_nonmotor_ok = 0
    for w in concept_words:
        r = rates_per_word[w]
        nonmotor = {p: v for p, v in r.items() if p not in motor_pools}
        if max(nonmotor, key=nonmotor.get) == word_to_pool[w]:
            top_nonmotor_ok += 1
    print(f"  among CONCEPT pools only, correct concept word wins: {top_nonmotor_ok}/{len(concept_words)} "
          f"= {top_nonmotor_ok/len(concept_words):.3f}", flush=True)

    # mechanism read 2: sweep motor down-weight factor f, recompute full pool-label
    def pool_label_at(f):
        ok = 0
        for w in words:
            r = {p: (v * f if p in motor_pools else v) for p, v in rates_per_word[w].items()}
            if max(r, key=r.get) == word_to_pool[w]:
                ok += 1
        return ok / len(words)

    print("  motor-downweight sweep (pool-label over ALL words):", flush=True)
    sweep = {}
    for f in [1.0, 0.5, 0.25, 0.1, 0.0]:
        sweep[f] = pool_label_at(f)
        print(f"    f={f:.2f}  pool-label={sweep[f]:.3f}", flush=True)
    base = sweep[1.0]
    best_f_acc = max(sweep[f] for f in [0.5, 0.25, 0.1, 0.0])
    print(f"\nRESULT: baseline pool-label {base:.3f}; best motor-downweighted {best_f_acc:.3f}; "
          f"concept-only-among-concepts {top_nonmotor_ok/len(concept_words):.3f}", flush=True)
    if best_f_acc >= base + 0.10:
        print("VERDICT: MOTOR-DOMINANCE CONFIRMED + CHEAP INFERENCE FIX -- down-weighting the motor pools at "
              "readout lifts pool-label notably. The 28-word front-end wall is substantially a readout-"
              "dominance issue; a simple pool-balance (inference or a light retrain) is a cheap partial fix.",
              flush=True)
    elif top_nonmotor_ok/len(concept_words) >= 0.80:
        print("VERDICT: motor dominance is real (concept words win among concept pools) but the full-vocab "
              "fix needs balancing motor vs concept (the concept words lose ONLY to motors). Cheap at "
              "readout for concept-only queries; a balanced retrain for unified recognition.", flush=True)
    else:
        print("VERDICT: NOT just motor dominance -- down-weighting motors does not recover pool-label and the "
              "concept words don't cleanly win among concept pools. The 28-word wall is deeper (real "
              "retrain/redesign needed), consistent with the owner-strategic framing.", flush=True)


if __name__ == "__main__":
    main()
