"""Capture multi-sample per-neuron concept activity from the 28-word bridge -> npz cache.

Input data for the two cheap-first representation-learning gates (expansion+Hebbian; e-prop). Each of the
28 words is driven M times (OU noise -> distinct samples); we record the full per-neuron concept-pool
activity (the substrate's LEARNED concept representation) per sample, so downstream analyses can measure
representation separability (nearest-neighbor classification) and test whether a learned expansion lifts it
past the 54%-concept-wins wall -- all on CPU from the cache, no GPU re-runs.

Run (GPU): python -m research.findings.raw._capture_28concept_activity --m-samples 16
Output: research/findings/raw/_28concept_activity_seed42.npz  (X [N, D], y [N], words, pools, pool_of_word)
Reuse-by-import (concept_pool_demo_v2 patches vocab to 28 words; load the existing _v17 bridge). No
protected-module change; no autograd.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.runners.concept_pool_demo_v2 as v2          # patches vocab to 28 words
import research.runners.concept_pool_demo as cpd
from sim.backend import get_backend, to_host

CKPT = "research/findings/raw/_v17_28word_seed42.simstate.h5"
N_LANG = 2048
SPARSITY = 0.03
DRIVE_PA = 200.0
RESET, STIM = 50, 100
OUT = "research/findings/raw/_28concept_activity_seed42.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-samples", type=int, default=16, help="captures per word (OU noise -> variation)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if not os.path.exists(CKPT):
        print(f"CANNOT-CONCLUDE: {CKPT} not found", flush=True); return
    from sim.text_embeddings import orthogonal_drive_pattern
    xp, backend = get_backend()
    print(f"=== capture 28-concept activity (M={a.m_samples}, backend={backend}) ===", flush=True)

    words = (list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB)
             + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(words)}
    word_to_pool = {}
    for w, vv in cpd.DIRECTION_VOCAB.items(): word_to_pool[w] = f"motor_{vv}"
    for w, vv in cpd.NOUN_VOCAB.items(): word_to_pool[w] = f"noun_pool_{vv}"
    for w, vv in cpd.VERB_VOCAB.items(): word_to_pool[w] = f"verb_pool_{vv}"
    for w, vv in cpd.ADJECTIVE_VOCAB.items(): word_to_pool[w] = f"adjective_pool_{vv}"
    pools = ([f"motor_{vv}" for vv in cpd.DIRECTION_VOCAB.values()]
             + [f"noun_pool_{vv}" for vv in cpd.NOUN_VOCAB.values()]
             + [f"verb_pool_{vv}" for vv in cpd.VERB_VOCAB.values()]
             + [f"adjective_pool_{vv}" for vv in cpd.ADJECTIVE_VOCAB.values()])

    bridge = cpd.build_concept_bridge(seed=a.seed, n_lang_input=N_LANG, n_per_pool=200, n_fs_per_pool=24,
                                      enable_adjective=True, weak_dynamics=True, verbose=False)
    bridge.load_checkpoint(CKPT)
    rm = bridge.region_manager
    all_idx = []
    for p in pools:
        all_idx += list(rm.indices(p))
    all_arr = xp.asarray(all_idx, dtype=xp.int64)
    lang_arr = xp.asarray(list(rm.indices("language_input")), dtype=xp.int64)
    D = len(all_idx)
    print(f"  {len(words)} words, {len(pools)} pools, D={D} per-neuron concept code", flush=True)

    def reset_to_rest():
        # thorough reset so each capture is independent (no state drift across 448 sequential captures)
        if bridge.cp_izh_vr is not None:
            bridge.cp_membrane_potential_v[:] = bridge.cp_izh_vr
        if bridge.cp_recovery_variable_u is not None:
            bridge.cp_recovery_variable_u[:] = 0.0
        bridge.cp_conductance_g_e[:] = 0.0
        bridge.cp_conductance_g_i[:] = 0.0
        bridge.cp_firing_states[:] = False
        if getattr(bridge, "cp_prev_firing_states", None) is not None:
            bridge.cp_prev_firing_states[:] = False

    def capture_once(word):
        drive = orthogonal_drive_pattern(cue_idx=word_to_idx[word], n_cues=len(words), n_neurons=N_LANG,
                                         drive_max_pA=DRIVE_PA, sparsity=SPARSITY)
        bridge.cp_external_input_current[:] = 0.0
        reset_to_rest()
        for _ in range(RESET):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[lang_arr] = xp.asarray(drive, dtype=xp.float32)
        acc = xp.zeros(D, dtype=xp.float64)
        for _ in range(STIM):
            bridge._run_one_simulation_step()
            acc += bridge.cp_firing_states[all_arr].astype(xp.float64)
        bridge.cp_external_input_current[:] = 0.0
        return to_host(acc) / STIM

    X = np.zeros((len(words) * a.m_samples, D), dtype=np.float32)
    y = np.zeros(len(words) * a.m_samples, dtype=np.int64)
    k = 0
    for wi, w in enumerate(words):
        for _ in range(a.m_samples):
            X[k] = capture_once(w); y[k] = wi; k += 1
        if (wi + 1) % 7 == 0:
            print(f"  captured {wi+1}/{len(words)} words", flush=True)
    pool_of_word = np.array([pools.index(word_to_pool[w]) for w in words], dtype=np.int64)
    np.savez_compressed(OUT, X=X, y=y, words=np.array(words), pools=np.array(pools),
                        pool_of_word=pool_of_word, m_samples=a.m_samples)
    print(f"  SAVED {OUT}: X{X.shape}, {len(words)} classes x {a.m_samples} samples", flush=True)


if __name__ == "__main__":
    main()
