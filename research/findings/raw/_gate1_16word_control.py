"""GATE 1 decisive CONTROL: reproduce the internal map's 16-word "NN >> pool-argmax (lossy readout)" with MY
OWN capture+decode pipeline. If my pipeline shows 16-word full-code NN clearly beats pool-argmax (the
internal map: 16-concept 100% NN-identifiable though pool-argmax recognition ~81%), then my pipeline is VALID
and the 28-word result (NN <= pool-argmax) is a genuine 16->28 transition (the readout escape does not
extend). If my pipeline ALSO shows low NN at 16 words, my measurement is broken and neither conclusion holds.

Captures the 16-word _learned16 bridge (4 motor + 4 noun + 4 verb + 4 adj) with the same snapshot-restore /
interleaved methodology, then the same fair head-to-head (k-avg pool-argmax vs full-code leave-one-out NN).

Reuse-by-import (plain concept_pool_demo -- NO v2 patch -> 16 words). No protected-module change; no autograd.
Run (GPU): python -m research.findings.raw._gate1_16word_control
"""
from __future__ import annotations
import os
import numpy as np

import research.runners.concept_pool_demo as cpd          # NO v2 -> 16-word vocab
from sim.backend import get_backend, to_host

CKPT = "research/findings/raw/_learned16_seed42.simstate.h5"
N_LANG = 2048; SPARSITY = 0.05; DRIVE_PA = 200.0; RESET, STIM = 50, 100; M = 16


def _mc(A):
    A = A - A.mean(axis=1, keepdims=True)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def main():
    if not os.path.exists(CKPT):
        print(f"CANNOT-CONCLUDE: {CKPT} not found", flush=True); return
    from sim.text_embeddings import orthogonal_drive_pattern
    xp, backend = get_backend()
    words = (list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB)
             + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    w2i = {w: i for i, w in enumerate(words)}
    pools = ([f"motor_{v}" for v in cpd.DIRECTION_VOCAB.values()]
             + [f"noun_pool_{v}" for v in cpd.NOUN_VOCAB.values()]
             + [f"verb_pool_{v}" for v in cpd.VERB_VOCAB.values()]
             + [f"adjective_pool_{v}" for v in cpd.ADJECTIVE_VOCAB.values()])
    w2pool = {}
    for w, v in cpd.DIRECTION_VOCAB.items(): w2pool[w] = f"motor_{v}"
    for w, v in cpd.NOUN_VOCAB.items(): w2pool[w] = f"noun_pool_{v}"
    for w, v in cpd.VERB_VOCAB.items(): w2pool[w] = f"verb_pool_{v}"
    for w, v in cpd.ADJECTIVE_VOCAB.items(): w2pool[w] = f"adjective_pool_{v}"
    pw = np.array([pools.index(w2pool[w]) for w in words])
    nw = len(words)
    print(f"=== GATE 1 CONTROL: 16-word (backend={backend}) ===", flush=True)

    bridge = cpd.build_concept_bridge(seed=42, n_lang_input=N_LANG, n_per_pool=200, n_fs_per_pool=24,
                                      enable_adjective=True, weak_dynamics=True, verbose=False)
    bridge.load_checkpoint(CKPT)
    rm = bridge.region_manager
    all_idx = []
    for p in pools:
        all_idx += list(rm.indices(p))
    aa = xp.asarray(all_idx, dtype=xp.int64); la = xp.asarray(list(rm.indices("language_input")), dtype=xp.int64)
    D = len(all_idx)
    snap = {"v": bridge.cp_membrane_potential_v.copy(),
            "u": (bridge.cp_recovery_variable_u.copy() if bridge.cp_recovery_variable_u is not None else None),
            "ge": bridge.cp_conductance_g_e.copy(), "gi": bridge.cp_conductance_g_i.copy(),
            "fire": bridge.cp_firing_states.copy()}

    def restore():
        bridge.cp_membrane_potential_v[:] = snap["v"]
        if snap["u"] is not None: bridge.cp_recovery_variable_u[:] = snap["u"]
        bridge.cp_conductance_g_e[:] = snap["ge"]; bridge.cp_conductance_g_i[:] = snap["gi"]
        bridge.cp_firing_states[:] = snap["fire"]

    def cap(word):
        dr = orthogonal_drive_pattern(cue_idx=w2i[word], n_cues=nw, n_neurons=N_LANG,
                                      drive_max_pA=DRIVE_PA, sparsity=SPARSITY)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET): bridge._run_one_simulation_step()
        bridge.cp_external_input_current[la] = xp.asarray(dr, dtype=xp.float32)
        acc = xp.zeros(D, dtype=xp.float64)
        for _ in range(STIM):
            bridge._run_one_simulation_step(); acc += bridge.cp_firing_states[aa].astype(xp.float64)
        bridge.cp_external_input_current[:] = 0.0
        return to_host(acc) / STIM

    X = np.zeros((nw * M, D)); y = np.zeros(nw * M, dtype=np.int64); k = 0
    for rnd in range(M):
        restore()
        for wi, w in enumerate(words):
            X[k] = cap(w); y[k] = wi; k += 1
    print(f"  captured {nw} words x {M}; D={D}", flush=True)

    for kk in [1, 4, 8]:
        ng = M // kk
        codes = []; lab = []
        for c in range(nw):
            xs = X[y == c]
            for g in range(ng):
                codes.append(xs[g*kk:(g+1)*kk].mean(0)); lab.append(c)
        codes = np.array(codes); lab = np.array(lab)
        pa = float(np.mean(codes.reshape(len(codes), nw, 200).mean(2).argmax(1) == pw[lab]))
        cn = _mc(codes); ok = 0
        for i in range(len(codes)):
            cents = np.stack([cn[(lab == c) & (np.arange(len(codes)) != i)].mean(0) for c in range(nw)])
            cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
            ok += int((cn[i] @ cents.T).argmax() == lab[i])
        nn = ok / len(codes)
        verdict = "NN WINS (lossy readout)" if nn > pa + 0.05 else "tie/NN-not-better"
        print(f"  k={kk:2d}-avg: pool-argmax {pa:.3f}  NN(full-code) {nn:.3f}  -> {verdict}", flush=True)
    print("\nINTERPRET: if 16-word NN clearly > pool-argmax here, the pipeline is VALID and the 28-word "
          "NN<=pool-argmax is a genuine readout-escape failure at scale (-> representation limit at 28). If "
          "16-word NN is ALSO not better, the NN measurement is the limiter, not the representation.",
          flush=True)


if __name__ == "__main__":
    main()
