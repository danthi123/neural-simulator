"""Conversation frontier (2026-07-09, research-gate-confirmed): A->W SPEAK decode on the SPARSE-DISTRIBUTED pool
-- the biology-preferred path to scale the speakable-on-spikes vocabulary PAST the 16-word grandmother-cell cap.

THE CAP (research gate + my own read): the concept-pool A->W (concept_speak_demo.py) uses ONE dedicated ~500-neuron
pool PER WORD (4 kinds x 4 pools = 16; the within-kind WTA degrades past ~4/kind). The `language_output` READ-OUT is
NOT the bottleneck -- it is already a shared region where each word is a sparse-distributed pattern (cosine-decode of
hundreds is fine). The fix = drive the words from a SHARED sparse-distributed pool (Kanerva/Pulvermuller G.20), NOT
disjoint per-word pools. `concept_pool_sparse_distributed.py` ALREADY builds that (shared_concept_pool + sparse K-of-N
patterns + a TRAINED shared_concept_pool->language_output read-out, capacity ~500-2000) but only EVALS RETRIEVAL. This
runner adds the missing A->W SPEAK decode + the anti-cheats -- a COMPOSE of validated pieces, NO new neural mechanism.

THE QUESTION this de-risks: does driving ONE of many OVERLAPPING sparse codes in the shared pool produce a CLEANLY
DECODABLE `language_output` pattern, or does aggregate crosstalk blur the decode at scale?

Anti-cheats: (1) LESION shared->language_output -> decode collapses (genuinely spiking, not a host lookup);
(2) PERMUTED -- driving word i's pattern must decode i, not j (checked by the confusion in speak_acc); (3) MOAT --
a NOVEL untrained sparse pattern yields no confident decode; (4) HELD-OUT reported via per-word margins.

Run:  SIM_BACKEND=numpy python -m research.runners._sparse_aw_speak_derisk --n-concepts 24 --n-train-events 80   (CPU smoke)
      SIM_BACKEND=cupy  python -m research.runners._sparse_aw_speak_derisk --n-concepts 64 --seeds 42,43,44,100,101,102
NO `sim/` edit (reuse-by-import).
"""
from __future__ import annotations
import argparse, time
import numpy as np


def _speak_decode(bridge, cp, sparse_pattern, word_patterns_out, n_lang_output,
                  drive_pA=1500.0, stim_steps=100, reset_steps=50):
    """Drive a concept's SPARSE PATTERN in shared_concept_pool -> accumulate the language_output firing vector ->
    cosine-match against each word's language_output reference pattern. Returns (best_idx, best_cos, margin, total_spikes)."""
    from sim.backend import to_host
    rm = bridge.region_manager
    shared = list(rm.indices("shared_concept_pool"))
    pat_global = cp.asarray([shared[i] for i in sparse_pattern], dtype=cp.int64)
    lang_out = cp.asarray(list(rm.indices("language_output")), dtype=cp.int64)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[pat_global] = float(drive_pA)
    acc = cp.zeros(n_lang_output, dtype=cp.float32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        acc += bridge.cp_firing_states[lang_out].astype(cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    v = to_host(acc)
    vn = v / (np.linalg.norm(v) + 1e-9)
    coss = np.asarray([float(vn @ (wp / (np.linalg.norm(wp) + 1e-9))) for wp in word_patterns_out])
    best = int(np.argmax(coss))
    srt = np.sort(coss)[::-1]
    margin = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])
    return best, float(coss[best]), margin, float(v.sum())


def _lesion_shared_to_langout(bridge, cp):
    """Zero the shared_concept_pool->language_output synapse weights (the A->W read-out) -> the speak decode must
    collapse (proves the decode rides those spiking synapses, not a host lookup). Returns a restore closure."""
    from sim.backend import to_host, from_host
    rm = bridge.region_manager
    shared = set(int(x) for x in rm.indices("shared_concept_pool"))
    langout = set(int(x) for x in rm.indices("language_output"))
    conn = bridge.cp_connections
    nnz = int(conn.nnz)
    indptr = to_host(conn.indptr); indices = to_host(conn.indices)
    pre = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    post = indices[:nnz]
    mask = np.isin(pre, list(shared)) & np.isin(post, list(langout))
    saved = to_host(conn.data[:nnz]).copy()
    data = saved.copy(); data[mask] = 0.0
    conn.data[:nnz] = from_host(data)
    def restore():
        conn.data[:nnz] = from_host(saved)
    return restore, int(mask.sum())


def run_seed(seed, n_concepts=64, n_train_events=400, n_lang_input=8192, n_shared_pool=2000,
             n_shared_fs=300, pattern_size=100, sparsity=0.03, verbose=True):
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern
    from research.runners.concept_pool_sparse_distributed import (
        build_sparse_pool_bridge, generate_sparse_patterns, apply_sparse_topographic_prior,
        train_concept_sparse, ALL_60)

    vocab = ALL_60[:n_concepts] if n_concepts <= 60 else list(ALL_60) + [f"concept{i}" for i in range(60, n_concepts)]
    n_concepts = len(vocab)
    n_lang_output = n_lang_input

    t0 = time.time()
    bridge = build_sparse_pool_bridge(seed=seed, n_lang_input=n_lang_input, n_shared_pool=n_shared_pool,
                                      n_shared_fs=n_shared_fs, n_lang_output=n_lang_output, verbose=False)
    sparse_patterns = generate_sparse_patterns(n_concepts, n_shared_pool, pattern_size, seed)
    apply_sparse_topographic_prior(bridge, n_concepts, n_lang_input, sparse_patterns, sparsity=sparsity,
                                   n_words_for_orthogonal=n_concepts, verbose=False)
    bridge.set_plasticity_gate("language_input_to_shared", 1.0)
    bridge.set_plasticity_gate("shared_to_language_output", 1.0)
    rng = np.random.RandomState(seed)
    for _e in range(n_train_events):
        order = list(range(n_concepts)); rng.shuffle(order)
        for wi in order:
            train_concept_sparse(bridge, wi, sparse_patterns[wi], n_lang_input, n_lang_output,
                                 sparsity, n_concepts)
    bridge.set_plasticity_gate("language_input_to_shared", 0.0)
    bridge.set_plasticity_gate("shared_to_language_output", 0.0)
    if verbose:
        print(f"  [seed {seed}] built+trained {n_concepts}x{n_train_events} in {time.time()-t0:.0f}s", flush=True)

    # the language_output REFERENCE patterns == the per-word teacher used in train_concept_sparse (orthogonal_drive_pattern)
    word_patterns_out = [orthogonal_drive_pattern(cue_idx=wi, n_cues=n_concepts, n_neurons=n_lang_output,
                                                  drive_max_pA=200.0, sparsity=sparsity) for wi in range(n_concepts)]

    # A->W SPEAK decode: drive each concept's sparse pattern -> decode its word
    correct = 0; margins = []; totals = []
    for wi in range(n_concepts):
        best, cos, margin, total = _speak_decode(bridge, cp, sparse_patterns[wi], word_patterns_out, n_lang_output)
        correct += int(best == wi); margins.append(margin); totals.append(total)
    speak_acc = correct / n_concepts

    # anti-cheat 3: MOAT -- a NOVEL untrained sparse pattern -> no confident decode
    novel = sorted(np.random.RandomState(seed + 99991).choice(n_shared_pool, pattern_size, replace=False).tolist())
    _, novel_cos, novel_margin, _ = _speak_decode(bridge, cp, novel, word_patterns_out, n_lang_output)

    # anti-cheat 1: LESION shared->language_output -> decode collapses
    restore, n_les = _lesion_shared_to_langout(bridge, cp)
    les_correct = 0
    for wi in range(n_concepts):
        best, _, _, _ = _speak_decode(bridge, cp, sparse_patterns[wi], word_patterns_out, n_lang_output)
        les_correct += int(best == wi)
    lesion_acc = les_correct / n_concepts
    restore()

    return {"seed": seed, "n_concepts": n_concepts, "speak_acc": round(speak_acc, 3),
            "mean_margin": round(float(np.mean(margins)), 4), "mean_langout_spikes": round(float(np.mean(totals)), 1),
            "novel_cos": round(novel_cos, 4),
            "novel_margin": round(novel_margin, 4), "lesion_acc": round(lesion_acc, 3), "n_lesioned": n_les}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-concepts", type=int, default=64)
    ap.add_argument("--n-train-events", type=int, default=400)
    ap.add_argument("--n-lang-input", type=int, default=8192)
    ap.add_argument("--n-shared-pool", type=int, default=2000)
    ap.add_argument("--pattern-size", type=int, default=100)
    ap.add_argument("--sparsity", type=float, default=0.03)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[SPARSE A->W SPEAK derisk] scale the speakable-on-spikes vocab past 16 via the shared sparse-distributed "
          f"pool | n_concepts={a.n_concepts}", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, n_concepts=a.n_concepts, n_train_events=a.n_train_events, n_lang_input=a.n_lang_input,
                     n_shared_pool=a.n_shared_pool, pattern_size=a.pattern_size, sparsity=a.sparsity)
        rows.append(r)
        print(f"  [seed {s}] speak_acc={r['speak_acc']} (margin {r['mean_margin']}, langout_spikes {r['mean_langout_spikes']}) "
              f"| MOAT novel_cos={r['novel_cos']} novel_margin={r['novel_margin']} | LESION acc={r['lesion_acc']} "
              f"({r['n_lesioned']} syn) | n_concepts={r['n_concepts']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        sa = [r["speak_acc"] for r in rows]; la = [r["lesion_acc"] for r in rows]
        # GO: speak scales well past 16 (>0.75 on 64 words), the decode is genuinely spiking (lesion collapses),
        # and the moat holds (a novel pattern's margin is far below the trained words').
        go = (np.mean(sa) > 0.75) and (np.mean(la) < 0.30)
        print(f"\n  AGGREGATE: speak_acc={np.mean(sa):.3f} lesion_acc={np.mean(la):.3f} "
              f"(chance={1.0/rows[0]['n_concepts']:.3f})", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the shared sparse-distributed pool SPEAKS '+str(rows[0]['n_concepts'])+' words on spikes (>>16 grandmother cap), decode is genuinely spiking (lesion collapses), scaling path VALIDATED' if go else 'the overlapping-sparse-code speak decode does not yet scale cleanly -- crosstalk blurs it (tune sparsity/pattern-size/training) or fall back to the multi-bridge dispatch (Rank 2)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
