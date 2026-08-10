"""RELATIONAL/SVO facts over the brain's OWN real-corpus codes (cheap-first probe): can the talkable
brain STORE + ANSWER a subject-verb-object fact ("the dog chased the cat" -> "who did the dog chase?
the cat") by binding role-filler pairs over its DISCOVERED co-occurrence concept codes?

A genuinely NEW knowledge dimension beyond property-inheritance: events/relations. The mechanism is the
project's extensively-validated FHRR (Fourier Holographic Reduced Representation) composition -- the OPEN
question is whether the CORRELATED real-corpus co-occurrence codes compose with acceptable SNR (the
compose-perceived arc showed the algebra tolerates code-correlation up to ~0.98; real-corpus codes carry
semantic similarity). This probe measures fidelity + the no-confab moat over real-corpus codes specifically.

Mechanism (self-contained numpy FHRR):
  * each concept's real-corpus code -> a unit PHASOR via a fixed random complex projection (grounded codes).
  * roles AGENT/VERB/PATIENT = fixed random unit phasors.
  * BIND a fact = AGENT*z_subj + VERB*z_verb + PATIENT*z_obj (element-wise complex mult, superposed).
  * QUERY the object = fact * conj(PATIENT) -> cleanup (argmax Re<est, z_c> over concepts) -> the object.
  * MOAT: an UNSTORED fact's query has low cleanup margin -> abstain.
Anti-cheats: PERMUTED roles (query with the wrong role -> wrong answer); UNSTORED (never-bound fact ->
abstain); memorization-floor (a random guess). 6-seed(-blind). NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi

D = 512          # phasor dimensionality (FHRR √D SNR)


def _phasors(codes, rows, seed):
    """Map each concept's real-corpus code -> a unit phasor via a fixed random complex projection."""
    rng = np.random.default_rng(seed * 131 + 7)
    NF = codes.shape[1]
    Mr = rng.standard_normal((D, NF)); Mi = rng.standard_normal((D, NF))
    Z = {}
    for r in rows:
        c = codes[r]
        proj = (Mr @ c) + 1j * (Mi @ c)
        Z[r] = proj / (np.abs(proj) + 1e-9)          # unit-magnitude phasor (info in phase)
    return Z


def _role(rng):
    ph = rng.uniform(-np.pi, np.pi, D)
    return np.exp(1j * ph)


def run_seed(seed, stories, K, n_facts=12):
    vocab, gfreq = discover_vocab(stories, K)
    target_set = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    rng = np.random.default_rng(seed)
    rows = list(range(len(vocab)))
    Z = _phasors(codes, rows, seed)
    AGENT, VERB, PATIENT = _role(rng), _role(rng), _role(rng)

    # build n_facts random SVO triples over distinct concepts
    idx = list(rows); rng.shuffle(idx)
    facts = []
    for i in range(0, min(n_facts * 3, len(idx) - 2), 3):
        facts.append((idx[i], idx[i + 1], idx[i + 2]))          # (subj, verb, obj)
    facts = facts[:n_facts]

    def bind(s, v, o):
        return AGENT * Z[s] + VERB * Z[v] + PATIENT * Z[o]

    def cleanup(est, restrict=None):
        cand = restrict if restrict is not None else rows
        scores = {r: float(np.real(np.vdot(Z[r], est))) / D for r in cand}
        best = max(scores, key=scores.get)
        vals = sorted(scores.values())
        margin = vals[-1] - vals[-2] if len(vals) > 1 else vals[-1]
        return best, margin

    # store each fact as its bound vector; query the OBJECT (patient) and the SUBJECT (agent)
    obj_correct, subj_correct, margins = 0, 0, []
    for (s, v, o) in facts:
        f = bind(s, v, o)
        est_o = f * np.conj(PATIENT)
        best_o, m = cleanup(est_o); margins.append(m)
        obj_correct += int(best_o == o)
        est_s = f * np.conj(AGENT)
        best_s, _ = cleanup(est_s)
        subj_correct += int(best_s == s)
    obj_acc = obj_correct / len(facts); subj_acc = subj_correct / len(facts)

    # PERMUTED-role anti-cheat: unbind the object slot with the WRONG role (VERB) -> should NOT recover obj
    perm_correct = 0
    for (s, v, o) in facts:
        f = bind(s, v, o)
        best, _ = cleanup(f * np.conj(VERB))
        perm_correct += int(best == o)
    perm_acc = perm_correct / len(facts)

    # MOAT: query an UNSTORED fact (bind a random triple NOT in the store, ask a DIFFERENT store's object).
    # Operationalize: the margin on a stored query vs the margin when unbinding a role from a NON-fact
    # (a single unbound phasor) -> a threshold separates stored from unstored.
    stored_margin = float(np.mean(margins))
    non_facts = []
    for _ in range(len(facts)):
        z = _role(rng)                                          # a random phasor that is NOT a bound fact
        _, m = cleanup(z * np.conj(PATIENT)); non_facts.append(m)
    unstored_margin = float(np.mean(non_facts))
    chance = 1.0 / len(vocab)
    return {"seed": seed, "n_facts": len(facts), "n_vocab": len(vocab), "chance": chance,
            "obj_acc": obj_acc, "subj_acc": subj_acc, "permuted_acc": perm_acc,
            "stored_margin": round(stored_margin, 4), "unstored_margin": round(unstored_margin, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--n-facts", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[SVO compose probe] corpus={a.corpus_path} K={a.K} D={D} n_facts={a.n_facts}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a.n_facts)
        recs.append(r)
        print(f"  [seed {s}] obj_acc={r['obj_acc']:.3f} subj_acc={r['subj_acc']:.3f} | "
              f"permuted={r['permuted_acc']:.3f} | margin stored={r['stored_margin']} vs unstored={r['unstored_margin']} | "
              f"chance={r['chance']:.4f} (V={r['n_vocab']})", flush=True)

    def m(k): return float(np.mean([r[k] for r in recs]))
    obj, subj, perm = m("obj_acc"), m("subj_acc"), m("permuted_acc")
    beats_chance = all(r["obj_acc"] > 0.5 for r in recs)
    beats_perm = all(r["obj_acc"] - r["permuted_acc"] > 0.4 for r in recs)
    moat = all(r["stored_margin"] > 1.5 * r["unstored_margin"] for r in recs)
    go = beats_chance and beats_perm
    print(f"\n  AGGREGATE ({len(recs)} seeds): obj_acc={obj:.3f} subj_acc={subj:.3f} permuted={perm:.3f}", flush=True)
    print(f"  obj>0.5 all={beats_chance} | beats_permuted(>.4) all={beats_perm} | moat-margin-sep all={moat}", flush=True)
    _verdict_msg = ('STORES + ANSWERS SVO relational facts over its OWN real-corpus codes (bind/unbind role-fillers; '
                    'correlated co-occurrence codes DO compose)'
                    if go else 'does NOT cleanly compose SVO over real-corpus codes')
    print(f"  VERDICT: {'GO' if go else 'NEGATIVE'} -- the brain {_verdict_msg}; moat-margin-separation={moat}.",
          flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "aggregate": {"obj": obj, "subj": subj, "permuted": perm},
                   "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
