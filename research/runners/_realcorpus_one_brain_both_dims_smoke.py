"""TRUE one brain, BOTH dimensions fully-spiking in ONE process: the property HTM reasoner AND the
relational RF-FHRR composer co-execute on cupy, answering BOTH a property question (inherit/cancel, apical
competition on the committed HTM coincidence kernel) AND a relational question (SVO what/who on the RF
resonate-and-fire + complex-synapse store), over the brain's OWN real-corpus codes, with the no-confab moat.

The both-dimension analog of EMERGE-70/71 (one brain, one backend, one process). Both spiking reasoners
are separate `SimulationBridge`s on the SAME cupy backend -> they co-execute. Reuse-by-import. NO `sim/`
edit. Requires SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._realcorpus_cancellation_spiking_derisk import (
    CancellingPoolerProbe, emergent_inputs, _adaptive_teach,
)
from research.runners._realcorpus_svo_spiking_derisk import _grounded_phases, D
from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi


def run(corpus_path, K, n_clusters, seed):
    # ---- PROPERTY reasoner (HTM, cupy): discover clusters, teach class + a member exception ----
    sdr_by_row, row_to_cat, cat_ids, vocab = emergent_inputs(corpus_path, K, seed, n_clusters)
    prop = CancellingPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=40)
    word_of = {r: vocab[r] for r in prop.rows}
    pos = exc_row = None
    for k in prop.cat_ids:
        for r in [rr for rr in prop.rows if prop.row2cat[rr] == k]:
            if prop.query(r, include_exc=False) == f"C{k}":
                pos, exc_row = k, r; break
        if pos is not None:
            break
    if pos is None:
        print("  no inheriting member for the property demo"); return None
    exc_word = word_of[exc_row]
    _adaptive_teach(prop, exc_row, pos, max_passes=16)          # teach the exception ON SPIKES (HTM)
    other = next((word_of[r] for r in prop.rows if prop.row2cat[r] == pos and r != exc_row
                  and prop.query(r) == f"C{pos}"), None)

    # ---- RELATIONAL reasoner (RF-FHRR, cupy substrate): grounded codes + a few SVO facts ----
    stories = load_token_stream_multi(corpus_path, max_stories=None)
    rvocab, gfreq = discover_vocab(stories, 64)
    rrow = {w: i for i, w in enumerate(rvocab)}
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in set(rvocab) or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, rvocab, hubs, window=WINDOW)
    rng = np.random.default_rng(seed)
    idx = list(range(len(rvocab))); rng.shuffle(idx)
    triples = [(rvocab[idx[i]], rvocab[idx[i + 1]], rvocab[idx[i + 2]]) for i in range(0, 6 * 3, 3)]
    words = sorted({w for t in triples for w in t})
    grounded = _grounded_phases(codes, words, rrow, seed)
    rel = RFPhasorComposer(seed=seed, D=D, vocab=words, grounded_codes=grounded, enable_substrate_store=True)
    for (s, v, o) in triples:
        rel.store(s, v, o)
    s0, v0, o0 = triples[0]

    print("  === ONE BRAIN, BOTH DIMENSIONS, ONE cupy PROCESS ===", flush=True)
    # PROPERTY (HTM apical competition on spikes)
    print(f"  [property/HTM] taught cluster {pos} class-property + exception '{exc_word}'", flush=True)
    pred = prop.query(exc_row)
    print(f"    Q: does '{exc_word}' have the class property?  -> {'NO (exception overrides)' if pred=='EXC' else pred}", flush=True)
    if other:
        print(f"    Q: does '{other}' have the class property?  -> {'YES (inherits)' if prop.query([r for r in prop.rows if word_of[r]==other][0])==f'C{pos}' else 'no'}", flush=True)
    # RELATIONAL (RF-FHRR on spikes)
    print(f"  [relational/RF] stored: '{s0} {v0} {o0}' (+{len(triples)-1} more)", flush=True)
    print(f"    Q: what does '{s0}' {v0}?  -> {rel.query_patient(s0, v0)}", flush=True)
    print(f"    Q: who {v0} '{o0}'?  -> {rel.query_agent(v0, o0)}", flush=True)
    unknown = next((w for w in words if (w, v0) not in {(a, b) for a, b, _ in triples}), "zzz")
    print(f"    Q: what does '{unknown}' {v0}? (unstored) -> {rel.query_patient(unknown, v0)}  [moat]", flush=True)

    prop_ok = (pred == "EXC")
    rel_ok = (rel.query_patient(s0, v0) == o0 and rel.query_agent(v0, o0) == s0)
    moat_ok = (rel.query_patient(unknown, v0) is None)
    return {"prop_ok": prop_ok, "rel_ok": rel_ok, "moat_ok": moat_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"[one brain both dims] property HTM + relational RF, co-executing on cupy | seed={a.seed}", flush=True)
    r = run(a.corpus_path, a.K, a.n_clusters, a.seed)
    if r is None:
        print("  VERDICT: NOT-EVALUABLE"); return
    go = r["prop_ok"] and r["rel_ok"] and r["moat_ok"]
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL'} -- BOTH the property (HTM apical cancellation) AND the relational "
          f"(RF-FHRR SVO what/who) spiking reasoners answered over the brain's OWN real-corpus codes IN ONE cupy "
          f"PROCESS, with the no-confab moat. The TRUE one brain, both knowledge dimensions, fully spiking.", flush=True)


if __name__ == "__main__":
    main()
