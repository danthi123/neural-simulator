"""Unified-agent benchmark -- the CONVERGE-not-add measurement (2026-06-04).

The bottleneck is no longer a missing mechanism; it is fragmentation + the absence of an honest end-to-end
measurement. The NestedCompositionAgent (research/runners/nested_composition_agent.py) ALREADY unifies the
validated pieces -- store/compose facts whose slots are structured entities, who/what Q&A, abstention on the
unknown, and dialogue planning -- on the phasor FHRR substrate (the de-risked unified-substrate candidate).

This module is the BENCHMARK HARNESS, not a new mechanism: it drives that one agent through a FROZEN
conversational test set at the real 320-concept scale, multi-seed, and reports a per-category pass-rate AND an
explicit boundary report -- so the project has a truthful "this is what the brain-analogue does and where it
ceilings", instead of scattered per-demo wins.

Two modes:
  - constructed (default): the agent's own phasor codes -- formalizes the 48%->100% composition claim into a
    rigorous frozen multi-seed benchmark with honest boundary reporting.
  - grounded: codes recalled from grounded word cues by PhasorAssociativeMemory (online-bounded STDP). The
    honest extension (cheat-removal backlog item #4 -- ungrounded codes): does grounded-cue recall fidelity at
    320 concepts degrade composition? Reported either way.

Reuse-by-import only; no protected-module edits. All phasor/composition code is numpy/CPU by design.

  python -m research.runners.unified_agent_benchmark               # full 5-seed constructed
  python -m research.runners.unified_agent_benchmark --quick       # 2-seed smoke
  python -m research.runners.unified_agent_benchmark --mode grounded
"""
from __future__ import annotations
import argparse
import json

from research.runners.nested_composition_agent import NestedCompositionAgent, Clause

# --- 320-concept vocabulary: readable CORE (the fact-bearing tokens) + generated FILLER (codebook difficulty).
# The frozen facts reference only core words (readable transcripts, the honesty guard); the filler pads each
# codebook to its target size so the cleanup/resonator faces the genuine 320-concept difficulty.
CORE_NOUNS = ["dog", "cat", "ball", "bird", "river", "child", "man", "woman", "bear", "fish",
              "lion", "wolf", "horse", "mouse", "rabbit", "apple", "bread", "stone", "leaf", "star",
              "snake", "frog", "duck", "goat", "book", "car", "tree", "house", "cloud", "hill"]
CORE_VERBS = ["chase", "hold", "see", "eat", "want", "find", "give", "take", "watch", "carry",
              "push", "pull", "throw", "catch", "follow"]
CORE_ADJS = ["big", "small", "red", "cold", "fast", "hot", "blue", "green", "old", "soft", "hard", "young"]

N_NOUN, N_VERB, N_ADJ = 200, 60, 60   # -> 320 distinct concepts total


def build_vocab(n_noun=N_NOUN, n_verb=N_VERB, n_adj=N_ADJ):
    nouns = CORE_NOUNS + [f"noun_{i}" for i in range(max(0, n_noun - len(CORE_NOUNS)))]
    verbs = CORE_VERBS + [f"verb_{i}" for i in range(max(0, n_verb - len(CORE_VERBS)))]
    adjs = CORE_ADJS + [f"adj_{i}" for i in range(max(0, n_adj - len(CORE_ADJS)))]
    return nouns, verbs, adjs


# --- FROZEN conversational test set. Every (agent, action) key is globally unique so query_patient is
# well-defined; every who-query (action, flat-patient) key is unique among flat facts. Categories are the
# composition depths the substrate is claimed to handle.
FACTS_FLAT = [
    ("dog", "chase", "cat"), ("child", "hold", "ball"), ("man", "see", "bird"),
    ("woman", "eat", "apple"), ("bear", "catch", "fish"), ("wolf", "follow", "horse"),
    ("mouse", "find", "bread"), ("bird", "watch", "river"),
]
FACTS_1ATTR = [
    ("cat", "want", ("red", "ball")), ("dog", "carry", ("big", "stone")),
    ("child", "throw", ("small", "leaf")), ("man", "give", ("cold", "bread")),
    ("horse", "pull", ("old", "car")), ("frog", "see", ("green", "snake")),
]
FACTS_2ATTR = [
    ("woman", "hold", (("big", "red"), "ball")), ("lion", "chase", (("small", "fast"), "rabbit")),
    ("man", "find", (("old", "soft"), "book")), ("child", "want", (("big", "blue"), "star")),
    ("bear", "push", (("hard", "cold"), "stone")),
]
FACTS_CLAUSE = [          # depth-1: the patient is one embedded clause (flat or attributed inner argument)
    ("duck", "see", Clause("cat", "chase", "bird")),
    ("woman", "watch", Clause("dog", "eat", ("red", "apple"))),
    ("goat", "hold", Clause("bear", "catch", "fish")),
    ("snake", "follow", Clause("cat", "chase", ("cold", "river"))),
    ("rabbit", "find", Clause("wolf", "push", "horse")),
]
FACTS_CLAUSE2 = [        # depth-2: the embedded clause's OWN patient is itself a clause (clause-in-clause) --
                          # the documented robust-depth boundary (~2, "occasionally costs a seed", needs D>=2048)
    ("dog", "see", Clause("cat", "chase", Clause("bird", "eat", "leaf"))),
    ("man", "watch", Clause("wolf", "follow", Clause("mouse", "find", "bread"))),
    ("child", "see", Clause("frog", "catch", Clause("duck", "hold", "fish"))),
]
CATEGORIES = [("flat", FACTS_FLAT), ("1-attribute", FACTS_1ATTR),
              ("2-attribute", FACTS_2ATTR), ("clause-depth1", FACTS_CLAUSE),
              ("clause-depth2", FACTS_CLAUSE2)]
ALL_FACTS = FACTS_FLAT + FACTS_1ATTR + FACTS_2ATTR + FACTS_CLAUSE + FACTS_CLAUSE2

# who-queries: (action, flat-patient) -> expected agent
WHO_QUERIES = [("chase", "cat", "dog"), ("hold", "ball", "child"), ("see", "bird", "man"),
               ("eat", "apple", "woman"), ("catch", "fish", "bear"), ("find", "bread", "mouse")]
# abstention probes: (agent, action) that is NOT a stored key -> expect None (no confabulation). Each token IS
# in-vocabulary; only the PAIR is unstored -- the hard abstention case (the agent and action both exist, just
# never together), so a pass is genuine no-confabulation, not an out-of-vocab reject.
ABSTAIN_QUERIES = [("river", "chase"), ("apple", "eat"), ("star", "hold"), ("dog", "watch"),
                   ("cat", "follow"), ("bird", "eat")]


def _build_grounded_codes(nouns, verbs, adjs, D, seed, n_input, cleanup=False):
    """Learn a phasor code per token via PhasorAssociativeMemory (grounded-cue -> code STDP), then return the
    per-token code used for composition.

    cleanup=False (raw grounded): the noisy recall READOUT itself (angle(W @ cue)) -- the harshest test, which
      conflates perception noise with composition (the resonator drowns in it).
    cleanup=True (pattern completion / CA3 autoassociator): snap the noisy readout to the nearest CLEAN concept
      ATTRACTOR and compose on THAT -- the biological architecture (perception is noisy; the cortical concept
      representation is a stable attractor; composition operates on the concept, not the raw sensory readout).
      The honest residual cost is recall MIS-identification (a wrong-but-clean attractor), reported as id_acc.

    Returns (codes_by_token, mean_recall_confidence, identification_accuracy_or_None)."""
    import numpy as np
    from research.runners.phasor_associative_memory import PhasorAssociativeMemory
    mem = PhasorAssociativeMemory(n_input=n_input, D=D, seed=seed)
    toks = list(nouns) + list(verbs) + list(adjs)
    for t in toks:
        mem.learn(t)
    codes, confs, n_id_correct = {}, [], 0
    for t in toks:
        pred = mem._readout(mem._cue(t)[0])           # the grounded recall (imperfect)
        confs.append(mem._best(pred)[1])
        if cleanup:
            tok = mem._best(pred)[0]                   # nearest clean attractor (pattern completion; no threshold)
            n_id_correct += (tok == t)
            codes[t] = mem.codes[tok]                  # compose on the consolidated concept code, not raw readout
        else:
            codes[t] = pred                            # compose on the raw noisy sensory readout
    return codes, float(np.mean(confs)), (n_id_correct / len(toks) if cleanup else None)


def run_seed(seed, D=2048, mode="constructed", n_input=512, n_noun=N_NOUN, n_verb=N_VERB, n_adj=N_ADJ):
    """Run the frozen conversational test set on one seed; return per-category + who + abstain pass counts."""
    nouns, verbs, adjs = build_vocab(n_noun, n_verb, n_adj)
    ext, recall_conf, id_acc = None, None, None
    if mode in ("grounded", "grounded-cleanup"):
        ext, recall_conf, id_acc = _build_grounded_codes(nouns, verbs, adjs, D, seed, n_input,
                                                         cleanup=(mode == "grounded-cleanup"))
    agent = NestedCompositionAgent(nouns, verbs, adjs, D=D, seed=seed, external_codes=ext)
    for ag, ac, pa in ALL_FACTS:
        agent.learn(ag, ac, pa)

    res = {"seed": seed, "recall_conf": recall_conf, "id_acc": id_acc, "categories": {}, "wrong": []}
    for name, facts in CATEGORIES:
        ok = 0
        for ag, ac, pa in facts:
            got = agent.query_patient(ag, ac)
            want = pa if isinstance(pa, str) else agent._render_filler(pa)
            if got == want:
                ok += 1
            else:
                res["wrong"].append({"q": f"what does {ag} {ac}?", "got": got, "want": want, "cat": name})
        res["categories"][name] = [ok, len(facts)]

    who_ok = 0
    for ac, pn, want in WHO_QUERIES:
        got = agent.query_agent(ac, pn)
        if got == want:
            who_ok += 1
        else:
            res["wrong"].append({"q": f"who {ac} {pn}?", "got": got, "want": want, "cat": "who"})
    res["categories"]["who-query"] = [who_ok, len(WHO_QUERIES)]

    abstain_ok = 0
    for ag, ac in ABSTAIN_QUERIES:
        got = agent.query_patient(ag, ac)
        if got is None:
            abstain_ok += 1
        else:
            res["wrong"].append({"q": f"what does {ag} {ac}? [should abstain]", "got": got,
                                 "want": None, "cat": "abstain"})
    res["categories"]["abstain"] = [abstain_ok, len(ABSTAIN_QUERIES)]
    return res


def aggregate(seed_results):
    """Per-category pass-rate across seeds + overall + the boundary report (which categories ceiling)."""
    cats = list(seed_results[0]["categories"].keys())
    agg = {}
    for c in cats:
        ok = sum(r["categories"][c][0] for r in seed_results)
        tot = sum(r["categories"][c][1] for r in seed_results)
        agg[c] = [ok, tot, ok / tot if tot else 0.0]
    grand_ok = sum(v[0] for v in agg.values())
    grand_tot = sum(v[1] for v in agg.values())
    return agg, grand_ok, grand_tot


def main():
    ap = argparse.ArgumentParser(description="Unified-agent benchmark: one agent, 320 concepts, frozen test set.")
    ap.add_argument("--mode", choices=["constructed", "grounded", "grounded-cleanup"], default="constructed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--quick", action="store_true", help="2-seed smoke")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--n-input", type=int, default=512, help="grounded-mode cue dimension")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42, 43] if args.quick else args.seeds

    n_concepts = N_NOUN + N_VERB + N_ADJ
    print(f"=== unified-agent benchmark | mode={args.mode} | {n_concepts} concepts "
          f"({N_NOUN}n+{N_VERB}v+{N_ADJ}a) | D={args.D} | seeds={seeds} ===\n", flush=True)
    print(f"  frozen test set: {len(ALL_FACTS)} facts "
          f"({len(FACTS_FLAT)} flat, {len(FACTS_1ATTR)} 1-attr, {len(FACTS_2ATTR)} 2-attr, "
          f"{len(FACTS_CLAUSE)} clause-depth1, {len(FACTS_CLAUSE2)} clause-depth2) "
          f"+ {len(WHO_QUERIES)} who + {len(ABSTAIN_QUERIES)} abstain\n", flush=True)

    seed_results = []
    for s in seeds:
        r = run_seed(s, D=args.D, mode=args.mode, n_input=args.n_input)
        seed_results.append(r)
        line = "  ".join(f"{c}={r['categories'][c][0]}/{r['categories'][c][1]}" for c in r["categories"])
        rc = f" | recall_conf={r['recall_conf']:.2f}" if r["recall_conf"] is not None else ""
        ida = f" id_acc={r['id_acc']:.2f}" if r.get("id_acc") is not None else ""
        print(f"  seed {s}:  {line}{rc}{ida}", flush=True)

    agg, gok, gtot = aggregate(seed_results)
    print("\n  --- per-category pass-rate (multi-seed) ---", flush=True)
    for c, (ok, tot, rate) in agg.items():
        print(f"    {c:<16} {ok:>3}/{tot:<3} = {rate*100:5.1f}%", flush=True)
    print(f"\n  OVERALL: {gok}/{gtot} = {gok/gtot*100:.1f}%  ({len(seeds)} seeds)", flush=True)

    print("\n  --- boundary report (honest ceiling) ---", flush=True)
    ceil = [c for c, (ok, tot, rate) in agg.items() if rate < 0.999 and c not in ("abstain", "who-query")]
    perfect = [c for c, (ok, tot, rate) in agg.items() if rate >= 0.999]
    print(f"    robust (100% multi-seed): {', '.join(perfect) if perfect else '(none)'}", flush=True)
    print(f"    ceilings below 100%:      {', '.join(ceil) if ceil else '(none)'}", flush=True)
    if seed_results[0]["wrong"]:
        print("    representative misses (seed-1):", flush=True)
        for w in seed_results[0]["wrong"][:6]:
            print(f"      [{w['cat']}] {w['q']}  got={w['got']!r}  want={w['want']!r}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": args.mode, "seeds": seeds, "D": args.D, "n_concepts": n_concepts,
                       "aggregate": agg, "overall": [gok, gtot], "per_seed": seed_results}, f, indent=2)
        print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
