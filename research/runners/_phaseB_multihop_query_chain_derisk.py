"""Multi-hop reasoning cheap-first DE-RISK — the role-structured pointer-chase over separately-stored facts.

Per the scoping (2026-06-17-multihop-reasoning-multiturn-dialogue-scoping.md, Option 1): multi-hop reasoning is
ITERATED single-hop retrieval where each hop's cleaned-up output becomes the next hop's cue. The production
composer already does ONE validated hop (query_patient: match (agent, action), read patient, abstain on miss).
A `query_chain` is that hop iterated — re-discretizing via cleanup between hops, so error does NOT integrate
multiplicatively. This probe falsifies/maps it on the production RFPhasorComposer.

WHY THE ANTI-CHEATS ARE LOAD-BEARING. A "90% transitive inference" result was CLAIMED then RETRACTED in this
project (2026-05-14) as a leaky spreading-activation artifact (2nd-degree co-occurrence neighbours, no role
structure). So this probe is built around the controls that defeat that trap. A multi-hop number any control
defeats is NOT a result.

THE RELATIONAL CHASE vs THE SPREADING TRAP. Each concept that is an agent in the chain is ALSO given a
DISTRACTOR fact with a different action (e.g. "cat play ball"), so its raw co-occurrence neighbourhood is
polluted and a spreading baseline cannot uniquely reach the chain endpoint — but the RELATIONAL chase
(follow the EAT relation: match agent AND action) stays unique. That gap is the whole test.

GATE (>=3 seeds): GO = 2-hop accuracy >= 0.90 on HELD-OUT chains (premises stored, the 2-hop composition never
a direct fact) AND >= spreading-floor + 0.5 AND permuted-relation -> chance AND lesion -> chance AND the moat
holds at every hop. BOUNDARY = 2-hop GO but 3-hop falls below 0.5 at D=128 and recovers only at higher D (a
precisely-mapped SNR depth limit -- still a deliverable). NEGATIVE = 2-hop <= spreading floor at all D, or
permuted-relation does NOT collapse (reading co-occurrence, not relations).

Run (CPU/numpy fast path; the spiking-cleanup parity is already established multi-seed):
  SIM_BACKEND=numpy python -m research.runners._phaseB_multihop_query_chain_derisk --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer

# A food-web: each chain is c0 --eat--> c1 --eat--> c2 --eat--> ... (predator eats prey eats ...). 8 chains x
# length 5 = 40 chain concepts. Each chain concept that is an agent ALSO gets a distractor fact (a different
# action + a distractor object) so its raw co-occurrence neighbourhood is polluted -> spreading can't chase.
CHAINS = [
    ["dog", "cat", "mouse", "bug", "leaf"],
    ["lion", "deer", "grass", "seed", "soil"],
    ["hawk", "snake", "frog", "fly", "pollen"],
    ["bear", "fish", "worm", "algae", "rock"],
    ["wolf", "rabbit", "clover", "dew", "sand"],
    ["owl", "vole", "moth", "nectar", "petal"],
    ["shark", "tuna", "squid", "krill", "plankton"],
    ["eagle", "trout", "beetle", "moss", "bark"],
]
DISTRACTOR_OBJS = ["ball", "box", "bell", "kite", "drum", "rope", "cup", "ring"]
EAT, PLAY, SEE = "eat", "play", "see"


def build_vocab():
    words = set([EAT, PLAY, SEE]) | set(DISTRACTOR_OBJS)
    for ch in CHAINS:
        words |= set(ch)
    return sorted(words)


def store_facts(composer, chains, permute_relation=False, rng=None, distractor_rng=None):
    """Store each chain as separate (agent eat patient) facts, PLUS dense distractors: each agent also
    play/see a couple of OTHER-chain concepts (DIFFERENT relations). The distractors pollute the relation-blind
    co-occurrence graph -- making a 2-hop co-occurrence neighbour AMBIGUOUS -- while leaving the functional EAT
    relation a clean unique chain. That gap is what separates the relational chase from leaky spreading.

    With permute_relation, scramble which PATIENT is bound to each (agent, eat) -- destroying the relational
    structure while keeping the concept set fixed (the permuted-relation anti-cheat). Returns
    (relational_edges, cooccurrence_edges)."""
    edges = []
    for ch in chains:
        for a, p in zip(ch[:-1], ch[1:]):
            edges.append((a, p))
    if permute_relation:
        patients = [p for _, p in edges]
        rng.shuffle(patients)
        edges = [(a, patients[i]) for i, (a, _) in enumerate(edges)]
    for a, p in edges:
        composer.store(a, EAT, p)
    all_concepts = [c for ch in chains for c in ch]
    cooc = list(edges)                                   # co-occurrence includes the relational edges
    agents = [a for a, _ in edges]
    for a in agents:
        others = [c for c in all_concepts if c != a]
        for act in (PLAY, SEE):                          # two distractor relations per agent
            o = others[int(distractor_rng.integers(len(others)))]
            composer.store(a, act, o)
            cooc.append((a, o))
    return edges, cooc


def query_chain(composer, cue, actions, lesion_rng=None, all_concepts=None):
    """The relational pointer-chase: x <- query_patient(x, action) per hop; abstain (None) on any miss.

    lesion_rng (anti-cheat 4): if set, replace the cleaned hop output with a RANDOM concept before the next hop
    (severs the between-hop re-cue) -> the chain must collapse to chance."""
    x = cue
    for a in actions:
        nxt = composer.query_patient(x, a)
        if nxt is None:
            return None
        if lesion_rng is not None:
            nxt = all_concepts[int(lesion_rng.integers(len(all_concepts)))]
        x = nxt
    return x


def spreading_predict(cooc, cue, k, all_concepts):
    """The memorization-floor baseline: leaky co-occurrence spreading (NO role structure). Undirected
    co-occurrence adjacency over ALL facts (relational + distractor), k diffusion steps, argmax excluding the
    cue. This is the mechanism the retracted result rode -- the chase must beat it."""
    idx = {c: i for i, c in enumerate(all_concepts)}
    n = len(all_concepts)
    A = np.zeros((n, n))
    for a, p in cooc:
        if a in idx and p in idx:
            A[idx[a], idx[p]] = 1.0
            A[idx[p], idx[a]] = 1.0
    A = A / (A.sum(1, keepdims=True) + 1e-12)
    p = np.zeros(n); p[idx[cue]] = 1.0
    for _ in range(k):
        p = A @ p
    p[idx[cue]] = -np.inf
    return all_concepts[int(np.argmax(p))]


def run_seed(seed, D, max_hops=4):
    vocab = build_vocab()
    rng = np.random.default_rng(seed)
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    edges, cooc = store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]

    # held-out k-hop queries: cue = chain[0], expected = chain[k]. The k-hop composition is NEVER a direct fact.
    out = {"seed": seed, "D": D, "hops": {}}
    for k in range(1, max_hops + 1):
        chase_ok = spread_ok = lesion_ok = 0
        tot = 0
        for ch in CHAINS:
            if len(ch) <= k:
                continue
            cue, want = ch[0], ch[k]
            tot += 1
            pred = query_chain(composer, cue, [EAT] * k)
            chase_ok += int(pred == want)
            spred = spreading_predict(cooc, cue, k, all_concepts)
            spread_ok += int(spred == want)
            lpred = query_chain(composer, cue, [EAT] * k, lesion_rng=np.random.default_rng(seed * 7 + k),
                                all_concepts=all_concepts)
            lesion_ok += int(lpred == want)
        out["hops"][k] = {"n": tot, "chase": chase_ok / tot, "spread": spread_ok / tot,
                          "lesion": lesion_ok / tot}

    # permuted-relation control: re-store under scrambled patient assignment -> 2-hop chase must collapse.
    comp_perm = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_facts(comp_perm, CHAINS, permute_relation=True, rng=np.random.default_rng(seed * 101 + 5),
                distractor_rng=np.random.default_rng(seed * 53 + 1))
    perm_ok = perm_tot = 0
    for ch in CHAINS:
        if len(ch) <= 2:
            continue
        perm_tot += 1
        perm_ok += int(query_chain(comp_perm, ch[0], [EAT, EAT]) == ch[2])
    out["perm_2hop"] = perm_ok / perm_tot

    # moat anti-cheat: an unstored cue, and a chain queried past its end, must abstain (None).
    unstored = query_chain(composer, "ball", [EAT, EAT])          # ball is never an agent
    overrun = query_chain(composer, CHAINS[0][0], [EAT] * (len(CHAINS[0]) + 2))  # past the chain end
    out["moat_unstored_abstains"] = unstored is None
    out["moat_overrun_abstains"] = overrun is None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--out", default="research/findings/raw/_phaseB_multihop_query_chain.json")
    a = ap.parse_args()

    n_concepts = len({c for ch in CHAINS for c in ch})
    chance = 1.0 / n_concepts
    print(f"[multihop query-chain de-risk] {len(CHAINS)} chains | {n_concepts} concepts | chance {chance:.3f}\n"
          "  GATE: 2-hop chase >=0.90 held-out, >= spread+0.5, permuted->chance, lesion->chance, moat holds.\n",
          flush=True)

    results = []
    for D in a.dims:
        for seed in a.seeds:
            r = run_seed(seed, D)
            results.append(r)
            h = r["hops"]
            hop_str = " | ".join(f"{k}h chase {h[k]['chase']:.2f}/spread {h[k]['spread']:.2f}/lesion {h[k]['lesion']:.2f}"
                                 for k in sorted(h))
            print(f"  [D={D} seed={seed}] {hop_str} || perm-2h {r['perm_2hop']:.2f} | "
                  f"moat {'ok' if (r['moat_unstored_abstains'] and r['moat_overrun_abstains']) else 'X'}",
                  flush=True)

    # aggregate at production D=128
    def agg(D, k, key):
        vals = [r["hops"][k][key] for r in results if r["D"] == D and k in r["hops"]]
        return float(np.mean(vals)) if vals else float("nan")
    d0 = a.dims[0]
    chase2 = agg(d0, 2, "chase"); spread2 = agg(d0, 2, "spread"); lesion2 = agg(d0, 2, "lesion")
    perm2 = float(np.mean([r["perm_2hop"] for r in results if r["D"] == d0]))
    moat_ok = all(r["moat_unstored_abstains"] and r["moat_overrun_abstains"] for r in results)
    # depth where chase crosses 0.5 at D=128
    cross = None
    for k in sorted(results[0]["hops"]):
        if agg(d0, k, "chase") >= 0.5:
            cross = k
    go = (chase2 >= 0.90 and chase2 >= spread2 + 0.5 and perm2 <= 2 * chance and lesion2 <= 2 * chance and moat_ok)
    chase3 = agg(d0, 3, "chase")
    # does a higher D recover 3-hop?
    chase3_hi = max([agg(D, 3, "chase") for D in a.dims if not np.isnan(agg(D, 3, "chase"))], default=float("nan"))
    boundary = (chase2 >= 0.90 and chase3 < 0.5 and chase3_hi >= 0.5 and perm2 <= 2 * chance and moat_ok)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"chance": chance, "d0": d0, "chase2": chase2, "spread2": spread2, "lesion2": lesion2,
                   "perm2": perm2, "chase3": chase3, "chase3_hi": chase3_hi, "moat_ok": moat_ok,
                   "cross_below_0.5_at_depth": cross, "results": results}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if go:
        print(f"  GO: the role-structured pointer-chase reasons multi-hop on the production composer — 2-hop "
              f"held-out {chase2:.2f} (vs spreading floor {spread2:.2f}, lesion {lesion2:.2f}, permuted {perm2:.2f}, "
              f"chance {chance:.2f}), moat intact at every hop. Genuine relational chaining (NOT co-occurrence "
              f"spreading — the controls collapse). Accuracy stays >=0.5 through depth {cross}.", flush=True)
    elif boundary:
        print(f"  BOUNDARY: 2-hop GO ({chase2:.2f}) but 3-hop falls to {chase3:.2f} at D={d0} and recovers only at "
              f"higher D ({chase3_hi:.2f}) — a precisely-mapped SNR depth limit (reachable to 2 hops at production "
              f"D; deeper chains need higher D). Controls collapse (perm {perm2:.2f}); moat {moat_ok}. Deliverable.",
              flush=True)
    elif not moat_ok:
        print("  MOAT_BREACH (HARD STOP): a chain accepted an unstored cue or over-ran its end — the no-confab "
              "guarantee failed across hops; investigate before anything else.", flush=True)
    else:
        print(f"  NEGATIVE: 2-hop chase {chase2:.2f} vs spreading floor {spread2:.2f} (gap {chase2-spread2:+.2f}) / "
              f"permuted {perm2:.2f} — the role-structured chase does not beat leaky spreading, or permutation does "
              "not collapse it (reading co-occurrence, not relations). Multi-hop is the next genuine wall; the "
              "recommendation flips to a factorised relational code (TEM, Option 4) as a research program.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
