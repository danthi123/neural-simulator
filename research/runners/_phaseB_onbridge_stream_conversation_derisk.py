"""CYCLE 96 — the FULL on-bridge conversational capability on STREAM-LEARNED codes: does the who/what SVO
recall + the no-confab abstention moat (CYCLE 90) work on the codes the SPIKING BRIDGE learned from the
corpus stream (CYCLE 95/capstone)?

This closes the loop. CYCLE 90 proved the conversational pipeline (multi-role HRR binding -> who/what recall
-> the no-confab moat) works on PPMI codes. CYCLE 95 proved the spiking bridge LEARNS the cortex codes from
the stream (population Hebbian co-occurrence, reaching host fidelity). This runner composes them: it
stream-learns the codes ON the bridge (reuse-by-import of the capstone's bridge + stream), then runs the
EXACT CYCLE-90 HRR who/what + abstention pipeline on those on-bridge-learned codes.

GATES (multi-seed, vs the CYCLE-90 PPMI baseline recall ~0.9 / abstain 1.0 / gap >0.1):
  recall      : who/what recall on PRESENT facts >= 0.70 (binding works on the stream-learned codes).
  no_confab   : ABSENT (verb,object) queries ABSTAIN -- zero false-accepts (the moat must NOT weaken).
  familiarity : present-match >> absent-match (a clean separable gap).
Anti-cheat: the gate is set a-priori (not tuned on the test); permuted-fact-free by construction (absent
queries are genuinely absent); the codes are LEARNED on the bridge from the stream (not curated/PPMI).

Reuse-by-import: build_stream_bridge + load_token_stream (capstone), hrr_bind/unbind (CYCLE 90), the taxonomy.
GPU (CuPy) for the stream-learning; the HRR pipeline is numpy.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_stream_conversation_derisk --seeds 42 --n-per 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
from research.runners._phaseB_onbridge_stream_cortex_derisk import (  # noqa: E402
    build_stream_bridge, load_token_stream, double_center, WINDOW)
from research.runners._phaseB_assembled_pipeline_ppmi_derisk import hrr_bind, hrr_unbind, _cos  # noqa: E402
from research.runners.option_c_real_cooccurrence_derisk import TAXONOMY_8x8, taxonomy_to_vocab_categories  # noqa: E402
from research.runners.option_c_stageB_fair_test import STOPLIST  # noqa: E402

N_FACTS = 8
GATE = 0.25      # a-priori conjunctive-cue gate (CYCLE 90): present min ~0.4, absent min ~0.1; midpoint


def stream_learn_codes(seed, stories, vocab, cat_ids, a):
    """Stream the corpus window-by-window into the population bridge; the bridge's Hebbian synapses learn the
    co-occurrence M; return the per-concept normalized code (log-double-centred population block-mean).
    Optional codes cache (--codes-npy): if the file exists, load the codes (skip the ~9-min re-stream); else
    stream + save. Lets the (instant) HRR re-tests reuse a single stream-learning."""
    cache = getattr(a, "codes_npy", None)
    if cache:
        cpath = cache.replace("SEED", str(seed))
        if os.path.exists(cpath):
            print(f"  [codes cache] loading stream-learned codes from {cpath} (skipping the re-stream)", flush=True)
            return np.load(cpath), 0, 0.0
    rng = np.random.RandomState(seed)
    targets = list(vocab); target_set = set(targets); Nt = len(targets)
    n_hub, n_per = a.n_hub, a.n_per
    gfreq = Counter()
    for toks in stories:
        gfreq.update(toks)
    hubs = [w for w, _ in gfreq.most_common() if w not in STOPLIST and w not in target_set][:n_hub]
    hub_idx = {w: i for i, w in enumerate(hubs)}
    tgt_row = {w: i for i, w in enumerate(targets)}
    keep = target_set | set(hubs)
    bridge, hub_region, tgt_region = build_stream_bridge(Nt, n_hub, n_per, seed)
    xp = bridge._cp if hasattr(bridge, "_cp") else None
    n_hub_neurons, n_tgt_neurons = n_hub * n_per, Nt * n_per

    def present_window(tgt_ids, hub_ids):
        hub_full = np.zeros(n_hub_neurons, np.float32)
        tgt_full = np.zeros(n_tgt_neurons, np.float32)
        for h in hub_ids:
            hub_full[h * n_per:(h + 1) * n_per] = a.hub_scale
        for t in tgt_ids:
            tgt_full[t * n_per:(t + 1) * n_per] = a.tgt_scale
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[hub_region] = xp.asarray(hub_full) if xp is not None else hub_full
        bridge.cp_external_input_current[tgt_region] = xp.asarray(tgt_full) if xp is not None else tgt_full
        for _ in range(a.window_steps):
            bridge._run_one_simulation_step()

    story_order = rng.permutation(len(stories)); n_windows = 0; t0 = time.time()
    for si in story_order:
        if n_windows >= a.max_windows:
            break
        kept = [t for t in stories[si] if t in keep]
        for c in range(len(kept)):
            lo, hi = max(0, c - WINDOW), min(len(kept), c + WINDOW + 1)
            win = kept[lo:hi]
            tgt_ids = [tgt_row[w] for w in win if w in target_set]
            hub_ids = [hub_idx[w] for w in win if w in hub_idx]
            if tgt_ids and hub_ids:
                present_window(tgt_ids, hub_ids)
                n_windows += 1
                if n_windows >= a.max_windows:
                    break
    bridge.cp_external_input_current[:] = 0.0
    W = np.asarray(to_host(bridge.cp_connections.todense())).astype(np.float64)
    blk = W[np.ix_(hub_region, tgt_region)].reshape(n_hub, n_per, Nt, n_per).mean(axis=(1, 3))
    code = double_center(np.log1p(blk.T * 100.0))                # (Nt, n_hub) stream-learned concept codes
    code = code / (np.linalg.norm(code, axis=1, keepdims=True) + 1e-12)
    if cache:
        cpath = cache.replace("SEED", str(seed))
        np.save(cpath, code)
        print(f"  [codes cache] saved stream-learned codes to {cpath}", flush=True)
    return code, n_windows, time.time() - t0


def run_conversation(codes, labels, seed):
    """The EXACT CYCLE-90 HRR who-Q&A + no-confab pipeline, on whatever per-concept codes arrive."""
    Nc, D = codes.shape
    rng = np.random.default_rng(seed * 17 + 3)
    R_a = rng.standard_normal(D) / np.sqrt(D)
    R_v = rng.standard_normal(D) / np.sqrt(D)
    R_o = rng.standard_normal(D) / np.sqrt(D)
    facts = []
    for _ in range(N_FACTS):
        i, j, k = rng.choice(Nc, 3, replace=False)
        facts.append((int(i), int(j), int(k)))
    bound = np.array([hrr_bind(R_a, codes[i]) + hrr_bind(R_v, codes[j]) + hrr_bind(R_o, codes[k])
                      for i, j, k in facts])

    def cue_match(verb, obj):
        scores = []
        for F in bound:
            mv = _cos(hrr_unbind(F, R_v), codes)[verb]
            mo = _cos(hrr_unbind(F, R_o), codes)[obj]
            scores.append(min(mv, mo))
        scores = np.array(scores)
        return int(np.argmax(scores)), float(scores.max())

    recall_ok, within_cat_err, conf_present = 0, 0, []
    for (i, j, k), F in zip(facts, bound):
        bf, conf = cue_match(j, k); conf_present.append(conf)
        if conf >= GATE:
            pred = int(np.argmax(_cos(hrr_unbind(bound[bf], R_a), codes)))
            recall_ok += int(pred == i)
            if pred != i and labels[pred] == labels[i]:
                within_cat_err += 1
    recall = recall_ok / N_FACTS
    stored = {(j, k) for _, j, k in facts}
    fa, n_absent, conf_absent, tries = 0, 0, [], 0
    while n_absent < N_FACTS and tries < 2000:
        tries += 1
        v, o = int(rng.integers(Nc)), int(rng.integers(Nc))
        if (v, o) in stored or v == o:
            continue
        n_absent += 1
        _, conf = cue_match(v, o); conf_absent.append(conf)
        fa += int(conf >= GATE)
    abstain = 1.0 - fa / max(n_absent, 1)
    return {"recall": recall, "abstain": abstain, "false_accept": fa, "within_cat_err": within_cat_err,
            "conf_present": float(np.mean(conf_present)), "conf_absent": float(np.mean(conf_absent))}


def run_seed(seed, stories, vocab, cat_ids, a):
    labels = np.asarray(cat_ids)
    codes, n_windows, secs = stream_learn_codes(seed, stories, vocab, cat_ids, a)
    r = run_conversation(codes, labels, seed)
    gap = r["conf_present"] - r["conf_absent"]
    print(f"\n[on-bridge stream conversation seed {seed}] {codes.shape[0]} concepts x {codes.shape[1]}D | "
          f"{n_windows} stream windows ({secs:.0f}s) | n_per={a.n_per}", flush=True)
    print(f"  who-Q&A recall (present): {r['recall']:.2f} (within-cat {r['within_cat_err']}/{N_FACTS}) | "
          f"no-confab abstain {r['abstain']:.2f} (false-accepts {r['false_accept']}) | "
          f"familiarity gap present {r['conf_present']:+.3f} vs absent {r['conf_absent']:+.3f}", flush=True)
    r.update({"seed": seed, "gap": gap, "n_windows": n_windows})
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-hub", type=int, default=300)
    p.add_argument("--n-per", type=int, default=16)
    p.add_argument("--window-steps", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=30000)
    p.add_argument("--hub-scale", type=float, default=250.0)
    p.add_argument("--tgt-scale", type=float, default=1200.0)
    p.add_argument("--codes-npy", default=None,
                   help="cache path (use SEED as a placeholder) — stream once + save, reload to skip re-streaming")
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[on-bridge stream CONVERSATION] seeds={seeds} n_per={a.n_per} -- does who/what recall + the no-confab "
          f"moat work on codes the SPIKING BRIDGE learned from the corpus stream?", flush=True)
    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    stories = load_token_stream()
    print(f"  loaded {len(stories)} stories; vocab {len(vocab)} concepts", flush=True)
    rows = [run_seed(s, stories, vocab, cat_ids, a) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    recall, abstain, cp, ca = m("recall"), m("abstain"), m("conf_present"), m("conf_absent")
    fa = sum(r["false_accept"] for r in rows); gap = cp - ca
    print(f"\n{'='*98}\n  MEAN ({len(seeds)} seeds): who-Q&A recall {recall:.2f} | no-confab abstain {abstain:.2f} "
          f"(total false-accepts {fa}) | familiarity gap present {cp:+.3f} vs absent {ca:+.3f}", flush=True)
    print(f"{'='*98}", flush=True)
    if recall >= 0.70 and abstain >= 0.95 and gap >= 0.10:
        print(f"  GO: the full conversational capability runs on STREAM-LEARNED on-bridge codes -- who-Q&A recall "
              f"{recall:.2f}, the no-confab moat HOLDS (abstain {abstain:.2f}, {fa} false-accepts), clean "
              f"familiarity gap (present {cp:+.3f} >> absent {ca:+.3f}). ==> the biology-faithful stream cortex "
              f"learned ON the spiking substrate carries the binding + recall + abstention end-to-end.", flush=True)
    elif recall >= 0.70 and gap >= 0.10:
        print(f"  PARTIAL (recall + gap OK, gate placement): recall {recall:.2f}, present {cp:+.3f} vs absent "
              f"{ca:+.3f} separable but abstain {abstain:.2f} at gate {GATE} -- set the gate between them.", flush=True)
    else:
        print(f"  NEGATIVE/PARTIAL: recall {recall:.2f}, abstain {abstain:.2f}, gap {gap:+.3f} -- the stream-learned "
              f"codes' fidelity or structure limits the binding/abstention; raise n_per/max_windows or inspect.",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"recall": recall, "abstain": abstain, "false_accepts": fa, "conf_present": cp, "conf_absent": ca,
           "gap": gap, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_onbridge_stream_conversation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
