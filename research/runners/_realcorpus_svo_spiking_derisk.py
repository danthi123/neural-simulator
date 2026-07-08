"""RELATIONAL/SVO on the SPIKING substrate (de-risk): store + answer SVO facts over the brain's OWN
real-corpus codes via the validated `RFPhasorComposer` (spiking resonate-and-fire + complex synapses),
NOT the numpy FHRR reference (CYCLE 988). The fully-spiking realization of the relational dimension --
matching the property dimension's spiking HTM realization.

The composer's `grounded_codes={word: phase}` lets us feed the brain's real-corpus co-occurrence codes as
the concept phases; `store(agent, action, patient)` / `query_patient(agent, action)` /
`query_agent(action, patient)` run the bind/unbind on spikes; the no-confab moat is built in (abstain when
no stored fact matches the cue). Cheap-first: does the relational algebra hold on the SPIKING substrate over
real-corpus codes? Anti-cheats: unstored -> abstain (moat); permuted (wrong-verb cue) -> abstain.
Requires SIM_BACKEND=cupy (the RF substrate). Reuse-by-import. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi

D = 64


def _grounded_phases(codes, words, row_of, seed):
    """Map each concept's real-corpus code -> a D-dim phase vector in [0,1] (a fixed random complex
    projection -> angle), the composer's grounded-code format."""
    rng = np.random.default_rng(seed * 131 + 7)
    NF = codes.shape[1]
    Mr = rng.standard_normal((D, NF)); Mi = rng.standard_normal((D, NF))
    out = {}
    for w in words:
        c = codes[row_of[w]]
        proj = (Mr @ c) + 1j * (Mi @ c)
        out[w] = (np.angle(proj) / (2 * np.pi)) % 1.0          # phase in [0,1]
    return out


def run_seed(seed, stories, K, n_facts=8, substrate=False):
    vocab, gfreq = discover_vocab(stories, K)
    row_of = {w: i for i, w in enumerate(vocab)}
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in set(vocab) or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    rng = np.random.default_rng(seed)
    idx = list(range(len(vocab))); rng.shuffle(idx)
    triples = [(vocab[idx[i]], vocab[idx[i + 1]], vocab[idx[i + 2]]) for i in range(0, n_facts * 3, 3)]
    words = sorted({w for t in triples for w in t})
    grounded = _grounded_phases(codes, words, row_of, seed)

    comp = RFPhasorComposer(seed=seed, D=D, vocab=words, grounded_codes=grounded,
                            enable_substrate_store=substrate)   # substrate=True -> RF complex-synapse store (spiking)
    for (s, v, o) in triples:
        comp.store(s, v, o)

    # ANSWER what (patient) + who (agent); MOAT on an unstored cue; PERMUTED (wrong verb) -> abstain
    what_ok = sum(int(comp.query_patient(s, v) == o) for (s, v, o) in triples)
    who_ok = sum(int(comp.query_agent(v, o) == s) for (s, v, o) in triples)
    used = {(s, v) for (s, v, o) in triples}
    unstored, tries = [], 0
    while len(unstored) < len(triples) and tries < 500:
        s, v = rng.choice(words), rng.choice(words)
        if (s, v) not in used:
            unstored.append((s, v))
        tries += 1
    moat_abstain = sum(int(comp.query_patient(s, v) is None) for (s, v) in unstored)
    perm_ok = 0
    for (s, v, o) in triples:
        wrong = rng.choice([vv for (_, vv, _) in triples if vv != v] or [v])
        perm_ok += int(comp.query_patient(s, wrong) == o)
    n = len(triples)
    return {"seed": seed, "n_facts": n, "n_words": len(words),
            "what_acc": what_ok / n, "who_acc": who_ok / n,
            "moat_abstain": moat_abstain / max(1, len(unstored)), "permuted_acc": perm_ok / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-facts", type=int, default=8)
    ap.add_argument("--substrate", action="store_true", help="RF complex-synapse store on the spiking substrate (cupy)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[SVO on SPIKING substrate] RFPhasorComposer + real-corpus grounded codes | K={a.K} D={D} "
          f"{'SUBSTRATE-store (RF complex synapses)' if a.substrate else 'numpy-KB store'}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a.n_facts, substrate=a.substrate)
        recs.append(r)
        print(f"  [seed {s}] what_acc={r['what_acc']:.3f} who_acc={r['who_acc']:.3f} | "
              f"MOAT abstain={r['moat_abstain']:.3f} | permuted={r['permuted_acc']:.3f} (words={r['n_words']})",
              flush=True)

    def m(k): return float(np.mean([r[k] for r in recs]))
    what, who, moat, perm = m("what_acc"), m("who_acc"), m("moat_abstain"), m("permuted_acc")
    what_ok = all(r["what_acc"] > 0.75 for r in recs)
    who_ok = all(r["who_acc"] > 0.75 for r in recs)
    moat_ok = all(r["moat_abstain"] > 0.9 for r in recs)
    perm_ok = all(r["what_acc"] - r["permuted_acc"] > 0.4 for r in recs)
    go = what_ok and who_ok and moat_ok and perm_ok
    print(f"\n  AGGREGATE ({len(recs)} seeds): what={what:.3f} who={who:.3f} MOAT={moat:.3f} permuted={perm:.3f}", flush=True)
    print(f"  what>0.75={what_ok} who>0.75={who_ok} moat>0.9={moat_ok} beats_perm={perm_ok}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'NEGATIVE'} -- relational SVO (what + who) over the brain's OWN real-corpus "
          f"codes runs ON THE SPIKING SUBSTRATE (RFPhasorComposer), with the no-confab moat.", flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
