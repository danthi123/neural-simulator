"""Generation-NOVELTY + K-capacity CEILING-PROBE -- the categorical-LLM-gap probe (scoping §6, the decisive sweep).

Question this MEASURES (converts ASSERTED -> MEASURED): "the spiking composer cannot freely GENERATE novel language,
only RETRIEVE stored facts." Per `research/findings/2026-06-22-conversational-scaling-vs-dendritic-scoping.md` §6
sweep 4 (the most important): store N facts, then ask the agent to render/describe/affirm, and measure

  (a) DISTINCT-GENERATED / STORED ratio -- across many generation+emission calls, does the agent EVER produce a
      sentence (an SVO triple) that is NOT one of the stored facts?  Expected ratio of NOVEL triples = 0.
  (b) HELD-OUT NOVEL-TRIPLE check -- a set of in-vocabulary SVO triples that were NEVER stored. Assert the agent
      either ABSTAINS or returns only stored content (never the held-out triple) under describe / what_does /
      who_does / is_it_true. Expected novel-composition score = 0 (it can only retrieve, never generate novel).

A 0 novel-composition score QUANTITATIVELY CONFIRMS the categorical free-generation gap -- the single cleanest piece
of evidence for the owner that the LLM gap is categorical, not scale.

ALSO (§6 sweep 1, cheap): the K-CAPACITY ceiling -- store K in {8,16,32,64,128} facts and measure recall accuracy +
the moat separation (unused-cue abstention) vs K. The genuinely-new number is the per-bridge NEURON-budget cap.

ANTI-CHEATS (carried from the vocab_ceiling_probe harness, NOT re-tuned on the test):
  - ABSTENTION FLOOR: a substantial set of UNSTORED (agent, action) cues must return None (the no-confab moat). A
    drop here is a HARD FAIL -- if the moat is broken, a "novel" emission could just be a confabulation, so the
    novelty number is only meaningful WHILE the moat holds.
  - SHUFFLED-FACT / permuted control: re-query who/what with WRONG (cue, filler) pairings -> ~0 false hits.

Strict REUSE-BY-IMPORT of `vocab_ceiling_probe` (vocab builder + agent builder + the matrix's anti-cheat structure)
and `rf_phasor_composer` (the composer API). NO sim/ edit -- this is a runner. The RF composer self-generates phasor
codes from `seed`, so NO retrain / NO trained bridge load is needed (a word-list is all it consumes).

GPU: the Hebbian parser + dlPFC are GPU-validated. Run with SIM_BACKEND=cupy.

Usage:
  SIM_BACKEND=cupy python -m research.runners.generation_novelty_probe --seed 42 --V 64 --D 128 --N-facts 32 \
      --k-sweep 8,16,32,64,128 --out research/findings/raw/_gen_novelty_s42.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from research.runners.vocab_ceiling_probe import _build_vocab, _build_agent


def _canon(triple):
    """A canonical (agent, action, patient) string key for set membership. patient is the surface string the
    composer would emit (a plain word here -- we only use flat SVO facts for the novelty sweep so 'generated' and
    'stored' are directly comparable as 3-word triples)."""
    return f"{triple[0]} {triple[1]} {triple[2]}"


def _build_disjoint_facts(words, n_facts, n_heldout, rng):
    """Carve the vocab into THREE disjoint word pools (agents / actions / patients) and draw `n_facts` STORED flat
    SVO triples + `n_heldout` HELD-OUT triples that share the SAME pools (so the held-out triples are fully
    IN-VOCABULARY and structurally identical to stored ones -- the ONLY difference is they were never stored).

    Disjoint pools guarantee: (i) every (agent, action) cue is unique (no accidental cue collision), and (ii) a
    held-out triple's (agent, action) was never a stored cue, so a correct system MUST abstain on it. Returns
    (stored_facts, heldout_facts) as lists of (a, ac, p) tuples, all distinct."""
    V = len(words)
    # need 3 disjoint pools big enough to draw n_facts + n_heldout distinct triples from each role pool
    need = n_facts + n_heldout
    third = V // 3
    assert third >= need, (f"V={V} too small: each role pool has {third} words but need {need} distinct "
                           f"(n_facts={n_facts} + n_heldout={n_heldout}). Use a larger --V.")
    agents = list(words[0:third])
    actions = list(words[third:2 * third])
    patients = list(words[2 * third:3 * third])
    # distinct row indices for stored, then DIFFERENT rows for held-out (no overlap of (agent,action) cues)
    perm = rng.permutation(third)
    stored_rows = perm[:n_facts]
    heldout_rows = perm[n_facts:n_facts + n_heldout]
    stored = [(agents[i], actions[i], patients[i]) for i in stored_rows]
    heldout = [(agents[i], actions[i], patients[i]) for i in heldout_rows]
    return stored, heldout


def _store_flat(agent_obj, facts):
    for a, ac, p in facts:
        agent_obj.hear(f"{a} {ac} {p}")


# ============================== SWEEP 4: GENERATION NOVELTY (the decisive probe) ==============================
def run_generation_novelty(words, seed, D, n_facts, n_heldout):
    """Store n_facts flat SVO facts, then measure (a) distinct-generated/stored ratio + novel count, and
    (b) the held-out novel-triple score. Carries the abstention-floor + shuffled-fact anti-cheats."""
    rng = np.random.default_rng((seed * 2654435761) % (2 ** 32))
    stored, heldout = _build_disjoint_facts(words, n_facts, n_heldout, rng)
    stored_set = {_canon(f) for f in stored}
    heldout_set = {_canon(f) for f in heldout}
    assert stored_set.isdisjoint(heldout_set)

    agent = _build_agent(len(words), seed, D, words)
    _store_flat(agent, stored)

    # ---- (a) DISTINCT-GENERATED / STORED + NOVEL count ----
    # Collect EVERY distinct sentence the agent emits across a broad battery of generation+emission calls:
    #   - describe(a) for every word in the vocab (stored agents -> a sentence; unstored -> None [moat])
    #   - render_fact direct (== describe) for every stored agent
    #   - what_does(a, ac) reconstructed into a full triple for every stored cue AND a sample of unstored cues
    generated = set()           # all non-None emitted SVO triples (full 3-word strings)
    describe_emissions = 0
    for w in words:
        s = agent.describe(w)            # generation: render a stored sentence whose agent == w, else None
        if s is not None:
            describe_emissions += 1
            generated.add(s)
    # what_does over stored cues -> rebuild the full triple "a ac p" from the returned patient
    for a, ac, _p in stored:
        pt = agent.what_does(a, ac)
        if pt is not None:
            generated.add(f"{a} {ac} {pt}")
    # what_does over a sample of UNSTORED cues (cross of stored agents x stored actions, off the stored diagonal)
    stored_cues = {(a, ac) for a, ac, _ in stored}
    n_cross = 0
    for a, _ac0, _ in stored:
        for _a1, ac, _ in stored:
            if (a, ac) in stored_cues:
                continue
            pt = agent.what_does(a, ac)
            if pt is not None:
                generated.add(f"{a} {ac} {pt}")
            n_cross += 1
            if n_cross >= 200:
                break
        if n_cross >= 200:
            break

    novel = sorted(generated - stored_set)          # emitted sentences that were NEVER stored
    distinct_generated = len(generated)
    distinct_over_stored = distinct_generated / max(1, len(stored_set))

    # ---- (b) HELD-OUT NOVEL-TRIPLE check ----
    # For each held-out (a, ac, p) triple (in-vocab, never stored): the agent must NOT produce it and MUST abstain.
    produced_heldout = 0          # times the agent emitted/affirmed a held-out triple (the novel-composition score)
    abstain_what = abstain_who = abstain_yesno = abstain_describe = 0
    heldout_detail = []
    for a, ac, p in heldout:
        wd = agent.what_does(a, ac)            # must be None (cue never stored)
        wh = agent.who_does(ac, p)             # must be None
        yn = agent.is_it_true(a, ac, p)        # must be 'unknown'
        ds = agent.describe(a)                 # must be None (agent never stored)
        abstain_what += int(wd is None)
        abstain_who += int(wh is None)
        abstain_yesno += int(yn == "unknown")
        abstain_describe += int(ds is None)
        # did the agent PRODUCE this exact held-out triple anywhere?
        emitted_triple = (wd == p) or (wh == a) or (yn in ("yes", "no")) or (ds == _canon((a, ac, p)))
        produced_heldout += int(emitted_triple)
        # also: is the held-out triple anywhere in the full generated set?
        in_generated = _canon((a, ac, p)) in generated
        produced_heldout += int(in_generated and not emitted_triple)  # don't double-count
        heldout_detail.append({"triple": _canon((a, ac, p)), "what_does": wd, "who_does": wh,
                               "is_it_true": yn, "describe": ds, "in_generated": in_generated})
    novel_composition_score = produced_heldout / max(1, len(heldout))   # expected 0.0

    # ---- ANTI-CHEAT 1: ABSTENTION FLOOR (20 unstored cues) ----
    n_ab = 20
    okab, att_ab, guard, conf = 0, 0, 0, []
    while att_ab < n_ab and guard < 100000:
        guard += 1
        a2, ac2 = (str(x) for x in rng.choice(words, size=2, replace=False))
        if (a2, ac2) in stored_cues:
            continue
        att_ab += 1
        got = agent.what_does(a2, ac2)
        okab += int(got is None)
        if got is not None and len(conf) < 5:
            conf.append({"cue": [a2, ac2], "confabulated": got})
    abstention = {"correct": okab, "attempted": att_ab, "rate": okab / max(1, att_ab), "confabulations": conf}

    # ---- ANTI-CHEAT 2: SHUFFLED-FACT / PERMUTED CONTROL (off-diagonal who-queries -> ~0 hits) ----
    sh_hits, sh_att = 0, 0
    A = [f[0] for f in stored]; AC = [f[1] for f in stored]; P = [f[2] for f in stored]
    n_shuf = min(len(stored), 12)   # cap so it stays cheap at large N
    for i in range(n_shuf):
        for j in range(n_shuf):
            if i == j:
                continue
            sh_att += 1
            sh_hits += int(agent.who_does(AC[i], P[j]) == A[i])   # (action_i, patient_j) was never stored
    shuffled = {"false_hits": sh_hits, "attempted": sh_att}

    return {
        "n_facts_stored": len(stored), "n_heldout": len(heldout),
        "distinct_generated": distinct_generated, "n_stored": len(stored_set),
        "distinct_generated_over_stored_ratio": distinct_over_stored,
        "n_novel_generated": len(novel), "novel_examples": novel[:10],
        "describe_emissions": describe_emissions,
        "novel_composition_score": novel_composition_score,
        "produced_heldout_count": produced_heldout,
        "heldout_abstention": {"what_does_None": abstain_what, "who_does_None": abstain_who,
                               "is_it_true_unknown": abstain_yesno, "describe_None": abstain_describe,
                               "attempted": len(heldout)},
        "heldout_detail": heldout_detail[:8],
        "abstention_floor": abstention, "shuffled_control": shuffled,
    }


# ============================== SWEEP 1: K-CAPACITY (cheap) ==============================
def run_k_capacity(words, seed, D, K_values, n_heldout_for_moat=8):
    """For each K, store K flat facts and measure recall (what_does over all K stored cues) + moat separation
    (abstention over unstored cues). Each K builds a FRESH agent (so K is the only load variable)."""
    out = {}
    for K in K_values:
        rng = np.random.default_rng((seed * 2654435761 + K) % (2 ** 32))
        third = len(words) // 3
        if third < K + n_heldout_for_moat:
            out[str(K)] = {"skipped": f"V too small for K={K} (third={third} < {K + n_heldout_for_moat})"}
            continue
        stored, heldout = _build_disjoint_facts(words, K, n_heldout_for_moat, rng)
        agent = _build_agent(len(words), seed, D, words)
        _store_flat(agent, stored)
        # recall: every stored cue returns its true patient
        rec_ok = 0
        for a, ac, p in stored:
            rec_ok += int(agent.what_does(a, ac) == p)
        # moat: held-out (never-stored) cues abstain
        moat_ok = 0
        for a, ac, p in heldout:
            moat_ok += int(agent.what_does(a, ac) is None)
        out[str(K)] = {
            "K": K, "recall_correct": rec_ok, "recall_attempted": len(stored),
            "recall_rate": rec_ok / max(1, len(stored)),
            "moat_abstain_correct": moat_ok, "moat_attempted": len(heldout),
            "moat_rate": moat_ok / max(1, len(heldout)),
        }
        print(f"[gen-novelty] K-capacity K={K:4d}: recall {rec_ok}/{len(stored)} "
              f"({rec_ok / max(1, len(stored)):.2f})  moat-abstain {moat_ok}/{len(heldout)} "
              f"({moat_ok / max(1, len(heldout)):.2f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--V", type=int, default=64, help="vocabulary size (needs V/3 >= N_facts + n_heldout)")
    ap.add_argument("--D", type=int, default=128, help="phasor dimension (production default 128)")
    ap.add_argument("--N-facts", type=int, default=32, help="facts stored for the novelty sweep")
    ap.add_argument("--n-heldout", type=int, default=16, help="held-out (never-stored) in-vocab triples")
    ap.add_argument("--k-sweep", type=str, default="8,16,32,64,128",
                    help="comma-separated K values for the K-capacity sweep ('' to skip)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sim.backend import get_backend, is_gpu_backend
    _, backend_name = get_backend()
    print(f"[gen-novelty] backend={backend_name} gpu={is_gpu_backend()}  seed={args.seed} V={args.V} "
          f"D={args.D} N_facts={args.N_facts} n_heldout={args.n_heldout}")

    words = _build_vocab(args.V)

    # --- SWEEP 4: generation novelty (the decisive number) ---
    print(f"\n[gen-novelty] ===== SWEEP 4: GENERATION NOVELTY  (store {args.N_facts}, held-out {args.n_heldout}) =====")
    gen = run_generation_novelty(words, args.seed, args.D, args.N_facts, args.n_heldout)
    print(f"  distinct-generated / stored ratio : {gen['distinct_generated_over_stored_ratio']:.3f}  "
          f"({gen['distinct_generated']} distinct emitted / {gen['n_stored']} stored)")
    print(f"  NOVEL generated (never stored)    : {gen['n_novel_generated']}   <-- expect 0")
    if gen["novel_examples"]:
        print(f"    novel examples: {gen['novel_examples']}")
    print(f"  novel-composition score (held-out): {gen['novel_composition_score']:.3f}  "
          f"({gen['produced_heldout_count']}/{gen['n_heldout']})   <-- expect 0.000")
    hab = gen["heldout_abstention"]
    print(f"  held-out abstention: what_does None {hab['what_does_None']}/{hab['attempted']}, "
          f"who_does None {hab['who_does_None']}/{hab['attempted']}, "
          f"is_it_true unknown {hab['is_it_true_unknown']}/{hab['attempted']}, "
          f"describe None {hab['describe_None']}/{hab['attempted']}")
    ab = gen["abstention_floor"]
    print(f"  ABSTENTION FLOOR (moat): {ab['correct']}/{ab['attempted']} ({ab['rate']:.2f})   <-- must be ~1.00")
    sc = gen["shuffled_control"]
    print(f"  shuffled-control false_hits: {sc['false_hits']}/{sc['attempted']}  (must be ~0)")

    # --- SWEEP 1: K-capacity ---
    kcap = {}
    if args.k_sweep.strip():
        K_values = [int(x) for x in args.k_sweep.split(",")]
        print(f"\n[gen-novelty] ===== SWEEP 1: K-CAPACITY  (K in {K_values}) =====")
        kcap = run_k_capacity(words, args.seed, args.D, K_values)

    # --- VERDICT ---
    moat_pass = ab["correct"] == ab["attempted"]
    shuffle_pass = sc["false_hits"] == 0
    novelty_zero = gen["n_novel_generated"] == 0 and gen["novel_composition_score"] == 0.0
    if moat_pass and shuffle_pass and novelty_zero:
        verdict = "CATEGORICAL_GAP_CONFIRMED"   # 0 novel content; moat + shuffle intact -> the LLM gap is categorical
    elif moat_pass and shuffle_pass and not novelty_zero:
        verdict = "GENERATES_NOVEL"             # surprising: the composer DID emit novel content (report honestly)
    else:
        verdict = "ANTICHEAT_FAIL"              # moat or shuffle broke -> the novelty number is not trustworthy
    print(f"\n[gen-novelty] VERDICT: {verdict}  "
          f"novelty_zero={novelty_zero}  moat={'PASS' if moat_pass else 'FAIL'}  "
          f"shuffled={'PASS' if shuffle_pass else 'FAIL'}")

    out = {
        "probe": "generation_novelty", "seed": args.seed, "V": args.V, "D": args.D,
        "N_facts": args.N_facts, "n_heldout": args.n_heldout, "backend": backend_name,
        "generation_novelty": gen, "k_capacity": kcap,
        "verdict": verdict, "novelty_zero": novelty_zero,
        "moat_pass": moat_pass, "shuffled_pass": shuffle_pass,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[gen-novelty] wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
