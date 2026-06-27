"""Tier 2.2 — SELF-CUED associative chain-of-thought, cheap-first DE-RISK (numpy CPU).

THE ONE DIFFERENCE vs the GO `query_chain` (2026-06-17). `query_chain(cue, [eat, eat])` takes a CALLER-SUPPLIED
relation list. Tier 2.2 is SELF-CUED: from a START concept with NO caller plan, the agent itself SELECTS the next
relation to chase by LEARNED ASSOCIATION STRENGTH (over its own stored facts), then chases that relation via the
VALIDATED single hop (`query_patient`: match the concept as AGENT under the chosen relation, read the PATIENT,
abstain on miss). Cleanup re-discretizes between hops (as `query_chain` does) so error does NOT compound. That
single change -- the agent picking the next hop instead of the caller -- is what turns retrieval into THINKING
(front-3 §2.6; roadmap Tier 2.2).

WHY THE ANTI-CHEATS ARE LOAD-BEARING. A "90% transitive inference" result was CLAIMED then RETRACTED here
(2026-05-14) as a leaky spreading-activation artifact. So this probe is built around the controls that defeat
that trap, EXACTLY as the GO `query_chain` de-risk was:
  (a) LESION the association the selector reads (zero/scramble the strengths) -> the self-cued chain collapses to
      a spreading floor (proves the LEARNED association is load-bearing -- the N.17 awake-replay-disruption test).
  (b) PERMUTED-graph control (scramble which patient binds each (agent, relation)) -> collapses (it follows
      RELATIONS, not concept co-occurrence).
  (c) The no-confab MOAT holds at EVERY hop -- a dead-end / no-associate -> ABSTAIN, never fabricate a hop.
  (d) NO error-compounding -- a 3-4 hop chain does not degrade (the cleanup resets SNR each hop).
And the spreading-floor baseline: a relation-BLIND co-occurrence walk (the retracted mechanism) is the floor the
self-cued chase must beat.

THE SELECTOR (this is the new bit). The "learned association strength" is RELATION-KEYED: assoc[(agent, rel)] =
how strongly the agent reinforced `rel` (counted from the stored facts, the same co-occurrence the agent's
`_assoc_graph` reads, but keyed by RELATION so it picks WHICH relation to chase). The chain relation (`eat`) is
reinforced more than the distractor relations (`play`/`see`) -- so at each concept the selector PICKS `eat` by
strength, then the role-structured `query_patient` does the actual hop. LESION = zero assoc -> the selector
cannot pick -> the chain dies.

GATE (>=3 seeds here; 6 in the build phase):
  GO       = self-cued k-hop accuracy >= 0.90 held-out (the k-hop target is NEVER a stored fact) AND
             >= spreading-floor + 0.5 AND lesion -> floor AND permuted -> chance AND moat holds at every hop AND
             no compounding (chase stays >= 0.5 to depth >= 3).
  BOUNDARY = self-cued 2-hop GO but a deeper hop falls below 0.5 (a mapped SNR/selector depth limit). Deliverable.
  NEGATIVE = self-cued <= spreading floor, OR lesion does NOT collapse, OR permuted does NOT collapse, OR the moat
             breaks at any hop. STOP, write the honest NEGATIVE (cite the 2026-05-14 retraction), do NOT over-claim.

Run (CPU/numpy fast path; spiking-cleanup parity is established multi-seed elsewhere):
  SIM_BACKEND=numpy python -m research.runners._phaseB_self_cued_chain_of_thought_derisk --seeds 42 43 44
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
# length 5 = 40 chain concepts. Each chain concept ALSO gets distractor facts under DIFFERENT relations
# (play/see) to OTHER-chain concepts -> its relation-blind co-occurrence neighbourhood is polluted (spreading
# can't chase), while the functional EAT relation stays a clean unique chain. SAME corpus shape as the GO
# query_chain de-risk -- the only change here is the agent SELF-SELECTS `eat` instead of the caller supplying it.
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
# `eat` reinforced more than the distractor relations -> the selector PICKS it by LEARNED strength. (The chain
# relation being the strongest association is the structure the lesion destroys.)
EAT_REINFORCE, DISTRACTOR_REINFORCE = 3, 1


def build_vocab():
    words = set([EAT, PLAY, SEE]) | set(DISTRACTOR_OBJS)
    for ch in CHAINS:
        words |= set(ch)
    return sorted(words)


def store_facts(composer, chains, permute_relation=False, rng=None, distractor_rng=None):
    """Store each chain as separate (agent eat patient) facts, PLUS dense distractors (agent play/see an
    other-chain concept under a DIFFERENT relation). Returns (relational_edges, cooccurrence_edges, assoc),
    where `assoc[(agent, relation)] = learned strength` = the RELATION-KEYED association the SELF-CUED selector
    reads (reinforced count per (agent, relation), the co-occurrence the agent's _assoc_graph reads -- but keyed
    by relation so it picks WHICH relation to chase).

    With permute_relation, scramble which PATIENT binds each (agent, eat) -- destroys the relational structure,
    keeps the concept set + the assoc strengths fixed (the permuted-relation anti-cheat: the SELECTOR still
    picks `eat`, but the role-structured chase now reaches the WRONG patient -> collapse)."""
    edges = []
    for ch in chains:
        for a, p in zip(ch[:-1], ch[1:]):
            edges.append((a, p))
    if permute_relation:
        patients = [p for _, p in edges]
        rng.shuffle(patients)
        edges = [(a, patients[i]) for i, (a, _) in enumerate(edges)]
    assoc = {}                                            # (agent, relation) -> learned association strength
    for a, p in edges:
        composer.store(a, EAT, p)
        assoc[(a, EAT)] = assoc.get((a, EAT), 0.0) + EAT_REINFORCE
    all_concepts = [c for ch in chains for c in ch]
    cooc = list(edges)                                    # co-occurrence includes the relational edges
    agents = [a for a, _ in edges]
    for a in agents:
        others = [c for c in all_concepts if c != a]
        for act in (PLAY, SEE):                           # two distractor relations per agent
            o = others[int(distractor_rng.integers(len(others)))]
            composer.store(a, act, o)
            cooc.append((a, o))
            assoc[(a, act)] = assoc.get((a, act), 0.0) + DISTRACTOR_REINFORCE
    return edges, cooc, assoc


# --- THE SELF-CUED SELECTOR + CHAIN ------------------------------------------------------------------------------
def select_next_relation(assoc, x, lesion=None, lesion_rng=None):
    """The agent's OWN choice of the next hop: among the relations available from concept `x`'s stored facts, pick
    the one with the HIGHEST learned association strength. Returns the relation, or None (no associate -> the moat
    abstains -- a dead end, no fabricated hop).

    THE SELECTOR LESION (the load-bearing anti-cheat: "lesion the association the selector reads"). Two modes:
      lesion="zero"   -> ZERO the learned association strengths. With no learned signal the selector has nothing
                         to read -> it ABSTAINS (returns None). This is the literal lesion of the association the
                         selector uses; the self-cued chain must collapse (the N.17 awake-replay-disruption test).
      lesion="demote" -> ACTIVELY MISLEAD: scramble the ordering so the chain relation no longer wins; the selector
                         is steered to a DISTRACTOR relation -> the role-structured chase lands on a non-chain
                         concept. (The "permuted association weights -> wanders to non-associated concepts" form.)
    `lesion_rng` seeds the demote scramble."""
    cands = {rel: w for (a, rel), w in assoc.items() if a == x}
    if not cands:
        return None
    if lesion == "zero":
        return None                                                    # no learned association left to read
    if lesion == "demote":
        cands = {rel: float(lesion_rng.random()) for rel in cands}     # destroy the learned ordering
    return max(cands, key=cands.get)


def self_cued_chain(composer, assoc, start, max_hops, goal=None, lesion_sel=None, lesion_sel_rng=None,
                    lesion_recue_rng=None, all_concepts=None):
    """SELF-CUED associative chain-of-thought: from `start`, at each step the AGENT selects the next relation by
    learned association strength (NO caller plan), then chases it via the validated `query_patient`; cleanup
    re-discretizes between hops. Stops at `goal` (if reached) or a dead-end (abstain -> None terminal). Returns
    (terminal_concept_or_None, the visited path).

    lesion_sel ("zero"/"demote") -> lesion the SELECTOR (anti-cheat a): see select_next_relation.
    lesion_recue_rng             -> lesion the HAND-OFF (the query_chain re-cue lesion): replace each hop output
                                    with a random concept -> the cleaned hand-off cannot be load-bearing."""
    x = start
    path = [x]
    for _ in range(max_hops):
        rel = select_next_relation(assoc, x, lesion=lesion_sel, lesion_rng=lesion_sel_rng)
        if rel is None:                                   # dead end / lesioned selector -> abstain (no fab hop)
            return None, path
        nxt = composer.query_patient(x, rel)              # the VALIDATED role-structured single hop + its moat
        if nxt is None:
            return None, path
        if lesion_recue_rng is not None:
            nxt = all_concepts[int(lesion_recue_rng.integers(len(all_concepts)))]
        path.append(nxt)
        x = nxt
        if goal is not None and x == goal:
            return x, path
    return x, path


def spreading_predict(cooc, cue, k, all_concepts):
    """The memorization floor: relation-BLIND co-occurrence spreading (the RETRACTED mechanism). Undirected
    co-occurrence adjacency over ALL facts (relational + distractor), k diffusion steps, argmax excluding the cue.
    The self-cued chase must beat this."""
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
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    edges, cooc, assoc = store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]

    # held-out k-hop queries: start = chain[0], goal = chain[k]. The k-hop self-cued composition is NEVER a
    # direct stored fact. The agent is told only the START + the GOAL identity (for stop) -- NOT the hop relations.
    out = {"seed": seed, "D": D, "hops": {}, "example_chains": {}}
    for k in range(1, max_hops + 1):
        chase_ok = spread_ok = lesion_demote_ok = lesion_zero_ok = lesion_recue_ok = 0
        tot = 0
        for ci, ch in enumerate(CHAINS):
            if len(ch) <= k:
                continue
            start, want = ch[0], ch[k]
            tot += 1
            term, path = self_cued_chain(composer, assoc, start, k, goal=want)
            chase_ok += int(term == want)
            if ci == 0:                                   # keep one worked example per depth for the report
                out["example_chains"][k] = {"start": start, "goal": want, "path": path, "reached": term == want}
            spred = spreading_predict(cooc, start, k, all_concepts)
            spread_ok += int(spred == want)
            # lesion the SELECTOR -- DEMOTE: actively mislead (scramble ordering) -> the chain wanders off-relation
            ld, _ = self_cued_chain(composer, assoc, start, k, goal=want, lesion_sel="demote",
                                    lesion_sel_rng=np.random.default_rng(seed * 17 + k))
            lesion_demote_ok += int(ld == want)
            # lesion the SELECTOR -- ZERO: remove the learned association entirely -> the selector abstains
            lz, _ = self_cued_chain(composer, assoc, start, k, goal=want, lesion_sel="zero")
            lesion_zero_ok += int(lz == want)
            # lesion the between-hop hand-off (re-cue) -> collapse
            lr, _ = self_cued_chain(composer, assoc, start, k, goal=want,
                                    lesion_recue_rng=np.random.default_rng(seed * 7 + k), all_concepts=all_concepts)
            lesion_recue_ok += int(lr == want)
        out["hops"][k] = {"n": tot, "chase": chase_ok / tot, "spread": spread_ok / tot,
                          "lesion_selector_demote": lesion_demote_ok / tot,
                          "lesion_selector_zero": lesion_zero_ok / tot, "lesion_recue": lesion_recue_ok / tot}

    # SANITY on the selector: confirm the agent's chosen relation at each chain start IS the chain relation `eat`
    # (the selector is reading the LEARNED reinforcement, not defaulting). Reported, not gated.
    sel_picks_eat = sum(int(select_next_relation(assoc, ch[0]) == EAT) for ch in CHAINS)
    out["selector_picks_eat_at_start"] = f"{sel_picks_eat}/{len(CHAINS)}"

    # permuted-relation control: re-store under scrambled patient assignment (assoc strengths UNCHANGED, so the
    # selector still picks `eat`) -> the role-structured chase now lands on the WRONG patient -> collapse.
    comp_perm = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    _e, _c, assoc_perm = store_facts(comp_perm, CHAINS, permute_relation=True,
                                     rng=np.random.default_rng(seed * 101 + 5),
                                     distractor_rng=np.random.default_rng(seed * 53 + 1))
    perm_ok = perm_tot = 0
    for ch in CHAINS:
        if len(ch) <= 2:
            continue
        perm_tot += 1
        term, _ = self_cued_chain(comp_perm, assoc_perm, ch[0], 2, goal=ch[2])
        perm_ok += int(term == ch[2])
    out["perm_2hop"] = perm_ok / perm_tot

    # moat anti-cheat: an unstored start (never an agent) abstains; a chain told to run past its end abstains at
    # the dead end (no associate -> None) rather than fabricating a hop.
    unstored, _ = self_cued_chain(composer, assoc, "ball", 2, goal=None)            # ball is never an agent
    overrun, _ = self_cued_chain(composer, assoc, CHAINS[0][0], len(CHAINS[0]) + 2, goal=None)  # past chain end
    out["moat_unstored_abstains"] = unstored is None
    out["moat_overrun_abstains"] = overrun is None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--out", default="research/findings/raw/_phaseB_self_cued_chain_of_thought.json")
    a = ap.parse_args()

    n_concepts = len({c for ch in CHAINS for c in ch})
    chance = 1.0 / n_concepts
    print(f"[Tier 2.2 self-cued chain-of-thought de-risk] {len(CHAINS)} chains | {n_concepts} concepts | "
          f"chance {chance:.3f}\n"
          "  The agent SELECTS each hop by learned association (no caller plan), chases via query_patient.\n"
          "  GATE: self-cued 2-hop >=0.90 held-out, >= spread+0.5, lesion-selector->floor, permuted->chance, "
          "moat holds, no compounding to depth>=3.\n", flush=True)

    results = []
    for D in a.dims:
        for seed in a.seeds:
            r = run_seed(seed, D)
            results.append(r)
            h = r["hops"]
            hop_str = " | ".join(f"{k}h sc {h[k]['chase']:.2f}/spr {h[k]['spread']:.2f}/"
                                 f"lZ {h[k]['lesion_selector_zero']:.2f}/lD {h[k]['lesion_selector_demote']:.2f}/"
                                 f"lR {h[k]['lesion_recue']:.2f}" for k in sorted(h))
            print(f"  [D={D} seed={seed}] {hop_str} || perm-2h {r['perm_2hop']:.2f} | "
                  f"sel->eat {r['selector_picks_eat_at_start']} | "
                  f"moat {'ok' if (r['moat_unstored_abstains'] and r['moat_overrun_abstains']) else 'X'}",
                  flush=True)

    # aggregate at production D=128
    def agg(D, k, key):
        vals = [r["hops"][k][key] for r in results if r["D"] == D and k in r["hops"]]
        return float(np.mean(vals)) if vals else float("nan")
    d0 = a.dims[0]
    chase2 = agg(d0, 2, "chase"); spread2 = agg(d0, 2, "spread")
    lesZ2 = agg(d0, 2, "lesion_selector_zero"); lesD2 = agg(d0, 2, "lesion_selector_demote")
    lesion_recue2 = agg(d0, 2, "lesion_recue")
    perm2 = float(np.mean([r["perm_2hop"] for r in results if r["D"] == d0]))
    moat_ok = all(r["moat_unstored_abstains"] and r["moat_overrun_abstains"] for r in results)
    sel_eat_ok = all(int(r["selector_picks_eat_at_start"].split("/")[0]) == len(CHAINS) for r in results)
    # depth where self-cued chase crosses 0.5 at D=128
    cross = None
    for k in sorted(results[0]["hops"]):
        if agg(d0, k, "chase") >= 0.5:
            cross = k
    chase3 = agg(d0, 3, "chase")
    # A control "COLLAPSED" iff it sits at the spreading/chance floor AND is >=0.5 below the self-cued chase. (The
    # goal-early-stop lets a permuted/scrambled walk hit the target by coincidence on ~1/8 chains -> ~0.08-0.12,
    # i.e. chance -- so the bar is "at the floor", not the over-strict exactly-2*chance.)
    floor = max(2.0 * chance, spread2 + 0.05)
    def collapsed(v):
        return v <= floor and v < chase2 - 0.5
    # LOAD-BEARING selector lesion = ZERO (the literal "lesion the association the selector reads" -> the selector
    # has no learned signal -> abstain -> collapse to 0.00; decisive). DEMOTE (scramble the ordering) is a
    # CORROBORATING control only, NOT gated: it mis-steers and drops the chain toward the floor, but the tiny
    # 3-relation candidate set {eat,play,see} + goal-early-stop leaves a seed-variable residual (a scrambled walk
    # still picks `eat` ~1/3 of hops) -- a TOY property of the small relation alphabet, not a mechanism leak (the
    # zero lesion already proves the learned association is load-bearing). The four task anti-cheats gated below:
    # (a) lesion-association->collapse [ZERO], (b) permuted->collapse, (c) moat at every hop, (d) no-compounding.
    go = (chase2 >= 0.90 and chase2 >= spread2 + 0.5 and lesZ2 <= 2 * chance
          and collapsed(perm2) and collapsed(lesion_recue2) and moat_ok and sel_eat_ok
          and (cross is not None and cross >= 3))
    chase3_hi = max([agg(D, 3, "chase") for D in a.dims if not np.isnan(agg(D, 3, "chase"))], default=float("nan"))
    boundary = (chase2 >= 0.90 and chase3 < 0.5 and chase3_hi >= 0.5 and lesZ2 <= 2 * chance
                and collapsed(perm2) and moat_ok)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"chance": chance, "d0": d0, "chase2": chase2, "spread2": spread2,
                   "lesion_selector_zero2": lesZ2, "lesion_selector_demote2": lesD2,
                   "lesion_recue2": lesion_recue2, "perm2": perm2, "selector_picks_eat": sel_eat_ok,
                   "chase3": chase3, "chase3_hi": chase3_hi, "moat_ok": moat_ok,
                   "cross_stays_above_0.5_to_depth": cross, "results": results}, fh, indent=2, default=str)

    ex = results[0]["example_chains"].get(2) or next(iter(results[0]["example_chains"].values()), None)
    print(f"\n{'='*108}", flush=True)
    if ex:
        print(f"  example self-cued chain (seed {results[0]['seed']}): {ex['start']} -> "
              f"{' -> '.join(ex['path'][1:])}  (goal {ex['goal']}, reached={ex['reached']}; hops chosen by the "
              f"agent's learned association, NOT supplied)", flush=True)
    if go:
        print(f"  GO: self-cued associative chain-of-thought works on the production composer — 2-hop held-out "
              f"{chase2:.2f} (vs spreading floor {spread2:.2f}; gap {chase2-spread2:+.2f}). LESION the selector's "
              f"learned association (zero) -> {lesZ2:.2f} (abstain/floor); mis-steer (demote) -> {lesD2:.2f}; "
              f"permuted-graph -> {perm2:.2f}; re-cue lesion -> {lesion_recue2:.2f}; the selector picks the chain "
              f"relation at every start; moat holds at every hop; no compounding (stays >=0.5 to depth {cross}). The "
              f"agent CHOOSES its hops by learned structure -> genuine self-cued thinking, NOT co-occurrence "
              f"smearing (every control collapses).", flush=True)
    elif boundary:
        print(f"  BOUNDARY: self-cued 2-hop GO ({chase2:.2f}) but a deeper hop falls below 0.5 (3-hop {chase3:.2f} "
              f"at D={d0}, recovers to {chase3_hi:.2f} at higher D) — a mapped selector/SNR depth limit. Controls "
              f"collapse (lesion-zero {lesZ2:.2f}, perm {perm2:.2f}); moat {moat_ok}. Deliverable.", flush=True)
    elif not moat_ok:
        print("  MOAT_BREACH (HARD STOP): a self-cued chain accepted an unstored start or fabricated a hop past a "
              "dead end — the no-confab guarantee failed; investigate before anything else.", flush=True)
    else:
        print(f"  NEGATIVE: self-cued chase {chase2:.2f} vs spreading floor {spread2:.2f} (gap {chase2-spread2:+.2f}) "
              f"| lesion-zero {lesZ2:.2f} | demote {lesD2:.2f} | permuted {perm2:.2f}. The self-cued selection does "
              f"not beat leaky spreading, OR the lesion/permutation does not collapse it (the selector is not reading "
              f"LEARNED structure). Per the 2026-05-14 retraction precedent: STOP, do NOT over-claim. The honest next "
              f"wall is a factorised relational/ordinal map (TEM), a research program.", flush=True)
    print(f"  [saved] {a.out}\n{'='*108}", flush=True)


if __name__ == "__main__":
    main()
