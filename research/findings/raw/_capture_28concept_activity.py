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

import importlib
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
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--out", type=str, default=OUT)
    ap.add_argument("--sparsity", type=float, default=SPARSITY)
    ap.add_argument("--n-lang", type=int, default=N_LANG)
    ap.add_argument("--vocab-mod", type=str, default="research.runners.concept_pool_demo_v2",
                    help="vocab monkey-patch module (v2=28 words, v3=64 words)")
    a = ap.parse_args()
    importlib.import_module(a.vocab_mod)   # patches cpd's vocab dicts (28 or 64 words)
    n_lang = a.n_lang
    ckpt = a.ckpt
    if not os.path.exists(ckpt):
        print(f"CANNOT-CONCLUDE: {ckpt} not found", flush=True); return
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

    bridge = cpd.build_concept_bridge(seed=a.seed, n_lang_input=n_lang, n_per_pool=200, n_fs_per_pool=24,
                                      enable_adjective=True, weak_dynamics=True, verbose=False)
    bridge.load_checkpoint(ckpt)
    rm = bridge.region_manager

    # Snapshot the clean loaded state. Restore it at the START of each round so EVERY round is a faithful
    # capture from the clean checkpoint state (round 0 reproduces the front-end probe's pool-argmax 0.571);
    # OU noise still varies the samples. Without this, the bridge state drifts after round 0 (0.571 -> 0.3).
    _snap = {
        "v": bridge.cp_membrane_potential_v.copy(),
        "u": (bridge.cp_recovery_variable_u.copy() if bridge.cp_recovery_variable_u is not None else None),
        "ge": bridge.cp_conductance_g_e.copy(),
        "gi": bridge.cp_conductance_g_i.copy(),
        "fire": bridge.cp_firing_states.copy(),
    }

    def restore_snap():
        bridge.cp_membrane_potential_v[:] = _snap["v"]
        if _snap["u"] is not None:
            bridge.cp_recovery_variable_u[:] = _snap["u"]
        bridge.cp_conductance_g_e[:] = _snap["ge"]
        bridge.cp_conductance_g_i[:] = _snap["gi"]
        bridge.cp_firing_states[:] = _snap["fire"]
    all_idx = []
    for p in pools:
        all_idx += list(rm.indices(p))
    all_arr = xp.asarray(all_idx, dtype=xp.int64)
    lang_arr = xp.asarray(list(rm.indices("language_input")), dtype=xp.int64)
    D = len(all_idx)
    print(f"  {len(words)} words, {len(pools)} pools, D={D} per-neuron concept code", flush=True)

    def capture_once(word):
        # WARM continuation (no cold reset -- matches the front-end probe that gets pool-argmax 0.571);
        # samples are INTERLEAVED round-robin in main() so no word is driven 16x in a row (avoids the
        # adaptation/saturation that collapsed pool-argmax to ~0.25 in the 16-in-a-row version).
        drive = orthogonal_drive_pattern(cue_idx=word_to_idx[word], n_cues=len(words), n_neurons=n_lang,
                                         drive_max_pA=DRIVE_PA, sparsity=a.sparsity)
        bridge.cp_external_input_current[:] = 0.0
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
    for rnd in range(a.m_samples):                 # INTERLEAVED round-robin: each word once per round
        restore_snap()                             # each round starts from the clean loaded state (faithful)
        for wi, w in enumerate(words):
            X[k] = capture_once(w); y[k] = wi; k += 1
        print(f"  round {rnd+1}/{a.m_samples} done", flush=True)
    pool_of_word = np.array([pools.index(word_to_pool[w]) for w in words], dtype=np.int64)
    np.savez_compressed(a.out, X=X, y=y, words=np.array(words), pools=np.array(pools),
                        pool_of_word=pool_of_word, m_samples=a.m_samples)
    print(f"  SAVED {a.out}: X{X.shape}, {len(words)} classes x {a.m_samples} samples", flush=True)


if __name__ == "__main__":
    main()
