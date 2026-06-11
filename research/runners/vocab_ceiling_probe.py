"""Conversational vocab + capability ceiling characterization -- the CHEAP-FIRST probe (overnight Thread A).

Question: does the FULL consolidated conversational agent's capability matrix HOLD at a larger vocabulary
(V = 64 -> 128 -> 320) than its V=16 default? This green-lights or kills the V=128->320 multi-seed sweep.

What it runs: the WHOLE agent loop -- BrainConversationalAgent (the Hebbian parser bridge + the FHRR-on-bridge
RFPhasorComposer + the dlPFC spiking dialogue planner) -- on a V-word vocabulary, asserting every capability in the
matrix as PASS/FAIL counts:

  - who/what Q&A          : comprehend (parse) + store SVO facts; who_does / what_does retrieve the right role.
  - ABSTENTION (no-confab moat): query an UNSTORED (agent, action) cue -> MUST return None. The hardest + most
                             load-bearing bar at scale (confabulation is the key failure mode); reported separately.
  - negation / yes-no    : a bound AFFIRM/NEGATE polarity tag -> is_it_true -> yes / no / unknown.
  - embedded clause      : hear_clause_fact (patient = a nested SVO clause) + what_does -> the rendered clause.
  - one-attribute        : an (adjective, noun) attribute bind ("big apple") + query.
  - two-attribute        : a (("adj1","adj2"), noun) bind ("big hot apple"); the documented K=5 boundary of the +-1
                             scheme. The FHRR substrate lifts it via a D-dial; we record the result at the agent's
                             default D and (optionally) sweep D for the min-D-at-V curve.
  - generation (describe): a full sentence reconstructed from spiking memory; None on an unknown subject.
  - dialogue (elaborate) : the dlPFC spiking content-selection brings up an on-topic associate; None if unconnected.

ANTI-CHEATS (so a high score is not trivially inflated):
  - ABSTENTION FLOOR: abstention must stay ~100% -- a drop here is a HARD FAIL regardless of the other rows.
  - SHUFFLED-FACT / permuted control: re-query who/what with a RANDOM permutation of (agent,action)->patient
    pairings; correct answers must collapse to ~chance (a system that "echoes" the most-recent/most-frequent filler
    would still score high here -- this catches that, the analogue of the 2026-05-03 permuted-label control).

Vocabulary: the RF composer self-generates a deterministic phasor code per word from `seed` and uses ONLY the word
SET (it ignores any code values), so a V-word LIST is all that's required -- no external code cache. We take the
first V words of the curated G.20 320-word list (g20_vocab_spec_320.ALL_WORDS_64); the fixed fact set is drawn FROM
that list so its concepts are guaranteed in-vocab. The parser is vocabulary-agnostic (it assigns roles by word
position x voice), so the same trained parser serves any V.

NOTE (honest caveat, stated in the findings doc verbatim): the composer is a PRINCIPLED IDEALIZATION -- an
exact-inverse VSA algebra that demands decorrelated full-precision codes. A clean pass at V is the algebra working
at V, NOT evidence the substrate became "more brain-like". The genuinely-new signal this probe produces is (i)
whether the algebra still holds through the FULL agent loop (parser hand-off + every capability) at V>16, and (ii)
the per-capability DEGRADATION MAP (which capability breaks first, at what V, at what D) -- the spec Step 3 (the
learned cortex) inherits.

GPU: the Hebbian parser + the dlPFC are GPU-validated. Run with SIM_BACKEND=cupy. NumPy is an import-check path only.

Usage:
  SIM_BACKEND=cupy python -m research.runners.vocab_ceiling_probe --V 64 --seed 42 --D 128 \
      --out research/findings/raw/_vocab_ceiling_V64_s42.json
  # two-attribute min-D sweep at this V (optional):
  SIM_BACKEND=cupy python -m research.runners.vocab_ceiling_probe --V 64 --seed 42 --D 128 \
      --two-attr-D-sweep 128,256,512
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.rf_phasor_composer import RFPhasorComposer, Clause


def _build_vocab(V):
    """First V words of the curated G.20 320-word list. (Word identity is irrelevant to the RF composer -- it makes
    its own random codes -- but real curated words read honestly and are globally unique by construction.)"""
    from research.runners.g20_vocab_spec_320 import ALL_WORDS_64
    assert V <= len(ALL_WORDS_64), f"V={V} exceeds the 320-word curated list"
    return list(ALL_WORDS_64[:V])


def _build_agent(V, seed, D, words, enable_spiking_cleanup=False):
    """The FULL agent on V words at dimension D. We construct the RFPhasorComposer EXPLICITLY (to control D -- the
    agent hardcodes D=128 otherwise) and inject it, so the agent routes parser -> this composer. The parser is built
    inside the agent (a real ~126-neuron Hebbian bridge)."""
    rf = RFPhasorComposer(seed=seed, D=D, vocab=words, period=200,
                          enable_spiking_cleanup=enable_spiking_cleanup)
    agent = BrainConversationalAgent(seed=seed, composer=rf)
    return agent


def _mixed_fact_set(words):
    """A fixed mixed fact set drawn from the FRONT of the vocab so the concepts are guaranteed in-vocab. Reuses the
    V=16 capability-matrix structure (flat / one-attribute / clause / negated) verbatim, just relabeled onto the
    vocab's own first words. Returns (flat_facts, attr_fact, clause_fact, neg_facts, two_attr_fact) plus the set of
    stored (agent, action) cues for the abstention control.

    Layout (indices into `words`):
      flat:   (0,1,2) (3,4,5) (6,7,8) (9,10,11)         -- 4 plain SVO facts
      attr:   (12,13,(14,15))                            -- one-attribute: w12 w13 (w14 w15)  ['adj noun']
      clause: (16,17, Clause(18,19,20))                  -- embedded clause: w16 w17 (w18 w19 w20)
      neg:    (21,22,23, AFFIRM) (24,25,26, NEGATE)      -- negation/yes-no
      2attr:  (27,28,((29,30),31))                       -- two-attribute: w27 w28 (w29 w30 w31)
    All distinct (agent, action) cues; all words < 32 so they fit the smallest V we test (>=32)."""
    w = words
    flat = [(w[0], w[1], w[2]), (w[3], w[4], w[5]), (w[6], w[7], w[8]), (w[9], w[10], w[11])]
    attr = (w[12], w[13], (w[14], w[15]))                              # ('adj','noun') one-attribute
    clause = (w[16], w[17], Clause(w[18], w[19], w[20]))              # nested SVO
    neg = [(w[21], w[22], w[23], "AFFIRM"), (w[24], w[25], w[26], "NEGATE")]
    two_attr = (w[27], w[28], ((w[29], w[30]), w[31]))               # (('adj1','adj2'),'noun')
    return flat, attr, clause, neg, two_attr


def _run_matrix(agent, words, two_attr_fact=None):
    """Store the mixed fact set through the agent loop and assert every capability. Returns a per-capability dict of
    {correct, attempted} plus the abstention-floor + shuffled-fact control results."""
    flat, attr, clause, neg, default_two = _mixed_fact_set(words)
    two_attr = two_attr_fact if two_attr_fact is not None else default_two

    # --- comprehend + store via the parser (hear), or structurally for the clause (nested input parsing is future
    # work, so the clause is supplied via hear_clause_fact, exactly as the V=16 test does). ---
    for a, ac, p in flat:
        agent.hear(f"{a} {ac} {p}")
    # one-attribute: stored structurally (the parser handles flat 3-word SVO; an attributed patient is a structure)
    agent.hear_clause_fact(attr[0], attr[1], attr[2])
    # embedded clause
    agent.hear_clause_fact(clause[0], clause[1], clause[2])
    # negation (hear with polarity)
    for a, ac, p, pol in neg:
        agent.hear(f"{a} {ac} {p}", polarity=pol)
    # two-attribute (stored structurally)
    agent.hear_clause_fact(two_attr[0], two_attr[1], two_attr[2])

    res = {}

    # who / what Q&A on the 4 flat facts (the action disambiguates)
    okw = oka = 0
    for a, ac, p in flat:
        okw += int(agent.what_does(a, ac) == p)
        oka += int(agent.who_does(ac, p) == a)
    res["what_qa"] = {"correct": okw, "attempted": len(flat)}
    res["who_qa"] = {"correct": oka, "attempted": len(flat)}

    # one-attribute: 'adj noun' both decoded
    adj, noun = attr[2]
    res["one_attribute"] = {
        "correct": int(agent.what_does(attr[0], attr[1]) == f"{adj} {noun}"),
        "attempted": 1, "expected": f"{adj} {noun}", "got": agent.what_does(attr[0], attr[1]),
    }

    # embedded clause: the nested SVO renders
    cl = clause[2]
    expect_clause = f"{cl.agent} {cl.action} {cl.patient}"
    got_clause = agent.what_does(clause[0], clause[1])
    res["embedded_clause"] = {"correct": int(got_clause == expect_clause), "attempted": 1,
                              "expected": expect_clause, "got": got_clause}

    # negation / yes-no: yes on AFFIRM, no on NEGATE, unknown on an unstored full-SVO cue
    okn = 0
    okn += int(agent.is_it_true(neg[0][0], neg[0][1], neg[0][2]) == "yes")
    okn += int(agent.is_it_true(neg[1][0], neg[1][1], neg[1][2]) == "no")
    # unknown cue: a full SVO that matches no stored fact (swap the patient of the AFFIRM fact)
    okn += int(agent.is_it_true(neg[0][0], neg[0][1], neg[1][2]) == "unknown")
    res["negation_yesno"] = {"correct": okn, "attempted": 3}

    # two-attribute: both adjectives + the noun decoded (set-equality, order-free)
    adj1, adj2 = two_attr[2][0]
    noun2 = two_attr[2][1]
    got_2a = agent.what_does(two_attr[0], two_attr[1])
    correct_2a = int(got_2a is not None and set(got_2a.split()) == {adj1, adj2, noun2})
    res["two_attribute"] = {"correct": correct_2a, "attempted": 1,
                            "expected_set": sorted({adj1, adj2, noun2}), "got": got_2a}

    # generation (describe): a flat fact's agent renders its full sentence; None on an unknown subject.
    # use flat[0] (a unique agent among the flat facts -- guaranteed: distinct first words).
    a0, ac0, p0 = flat[0]
    res["generation"] = {
        "correct": int(agent.describe(a0) == f"{a0} {ac0} {p0}"),
        "attempted": 1, "expected": f"{a0} {ac0} {p0}", "got": agent.describe(a0),
    }

    # dialogue planning (elaborate): an on-topic associate from the agent's own graph; None on an unconnected topic.
    graph = agent._assoc_graph()
    # pick a connected topic that actually has neighbors (flat[0]'s agent co-occurs with its action + patient)
    neighbors = set(graph.get(a0, {}))
    assoc = agent.elaborate(a0)
    res["dialogue"] = {"correct": int(assoc in neighbors and len(neighbors) > 0), "attempted": 1,
                       "topic": a0, "got": assoc, "neighbors": sorted(neighbors)}

    # ===== ANTI-CHEAT 1: ABSTENTION FLOOR =====
    # For a SUBSTANTIAL set of UNSTORED (agent, action) cues, what_does MUST return None. The no-confab moat is the
    # load-bearing bar at scale, so we sample 20 unstored cues (not just len(flat)) -- confabulation is the key
    # failure mode and a thin sample under-tests it. Cues are pairs of vocab words NOT used as a stored cue.
    stored_cues = {(a, ac) for a, ac, *_ in flat} | {(attr[0], attr[1]), (clause[0], clause[1]),
                                                      (neg[0][0], neg[0][1]), (neg[1][0], neg[1][1]),
                                                      (two_attr[0], two_attr[1])}
    rng = np.random.default_rng(hash(("abstain", tuple(words))) % (2 ** 32))
    n_abstain = 20
    okab, attempted_ab, guard, conf_examples = 0, 0, 0, []
    while attempted_ab < n_abstain and guard < 100000:
        guard += 1
        a2, ac2 = (str(x) for x in rng.choice(words, size=2, replace=False))
        if (a2, ac2) in stored_cues:
            continue
        attempted_ab += 1
        got = agent.what_does(a2, ac2)
        okab += int(got is None)
        if got is not None and len(conf_examples) < 5:
            conf_examples.append({"cue": [a2, ac2], "confabulated": got})
    res["abstention"] = {"correct": okab, "attempted": attempted_ab, "confabulations": conf_examples}

    # ===== ANTI-CHEAT 2: SHUFFLED-FACT / PERMUTED CONTROL =====
    # Re-query who/what with WRONG (cue, filler) pairings of the flat facts -- every off-diagonal (i != j) pair, so
    # the control is exhaustive (not one random permutation). The cue (action_i, patient_j) was never stored, so a
    # correct retrieval system returns None / not-agent_i. We count how many such queries RETURN the true agent_i
    # anyway -- which must be ~0. A high count means the system echoes a frequent/recent filler (the analogue of the
    # 2026-05-03 permuted-label control that caught the text-IO artifact).
    agents = [f[0] for f in flat]
    actions = [f[1] for f in flat]
    patients = [f[2] for f in flat]
    shuffled_hits, shuffled_attempted = 0, 0
    for i in range(len(flat)):
        for j in range(len(flat)):
            if i == j:
                continue
            shuffled_attempted += 1
            # who <action_i> <patient_j>? -- (action_i, patient_j) is an unstored pair -> must NOT return agent_i
            got = agent.who_does(actions[i], patients[j])
            shuffled_hits += int(got == agents[i])
    res["shuffled_control"] = {"false_hits": shuffled_hits, "attempted": shuffled_attempted}

    return res


def _capability_rows():
    return ["what_qa", "who_qa", "one_attribute", "embedded_clause", "negation_yesno",
            "two_attribute", "generation", "dialogue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=64, help="vocabulary size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=128, help="phasor dimension (production default 128)")
    ap.add_argument("--two-attr-D-sweep", type=str, default=None,
                    help="comma-separated D values to sweep two-attribute at this V (e.g. 128,256,512)")
    ap.add_argument("--spiking-cleanup", action="store_true",
                    help="route cleanup through the fully-on-bridge spiking path (selection in spikes)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sim.backend import get_backend, is_gpu_backend
    _, backend_name = get_backend()
    print(f"[vocab-ceiling] backend={backend_name} gpu={is_gpu_backend()}  V={args.V} seed={args.seed} D={args.D}")

    words = _build_vocab(args.V)
    agent = _build_agent(args.V, args.seed, args.D, words, enable_spiking_cleanup=args.spiking_cleanup)
    res = _run_matrix(agent, words)

    # optional two-attribute min-D sweep at this V (fresh agent per D so the two-attribute fact is the only load var)
    two_attr_sweep = {}
    if args.two_attr_D_sweep:
        Ds = [int(x) for x in args.two_attr_D_sweep.split(",")]
        flat, attr, clause, neg, two_attr = _mixed_fact_set(words)
        adj1, adj2 = two_attr[2][0]
        noun2 = two_attr[2][1]
        for Dv in Ds:
            ag2 = _build_agent(args.V, args.seed, Dv, words)
            ag2.hear_clause_fact(two_attr[0], two_attr[1], two_attr[2])
            ag2.hear(f"{flat[0][0]} {flat[0][1]} {flat[0][2]}")     # a flat distractor fact
            got = ag2.what_does(two_attr[0], two_attr[1])
            ok = int(got is not None and set(got.split()) == {adj1, adj2, noun2})
            two_attr_sweep[str(Dv)] = {"correct": ok, "got": got}
            print(f"[vocab-ceiling] two-attr V={args.V} D={Dv}: {'PASS' if ok else 'FAIL'} (got={got})")

    # --- print the per-capability matrix ---
    print(f"\n[vocab-ceiling] ===== capability matrix  V={args.V}  D={args.D}  seed={args.seed} =====")
    for cap in _capability_rows():
        r = res[cap]
        rate = r["correct"] / r["attempted"] if r["attempted"] else 0.0
        print(f"  {cap:18s} {r['correct']}/{r['attempted']}  ({rate:.2f})")
    ab = res["abstention"]
    ab_rate = ab["correct"] / ab["attempted"] if ab["attempted"] else 0.0
    print(f"  {'ABSTENTION (moat)':18s} {ab['correct']}/{ab['attempted']}  ({ab_rate:.2f})   <-- must be ~1.00")
    sc = res["shuffled_control"]
    print(f"  shuffled-control false_hits {sc['false_hits']}/{sc['attempted']}  (must be ~0)")

    # --- verdict ---
    core_caps = _capability_rows()
    all_core_pass = all(res[c]["correct"] == res[c]["attempted"] for c in core_caps)
    abstain_pass = ab["correct"] == ab["attempted"]
    shuffle_pass = sc["false_hits"] == 0
    if abstain_pass and all_core_pass and shuffle_pass:
        verdict = "GO"
    elif abstain_pass and shuffle_pass:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"
    failing = [c for c in core_caps if res[c]["correct"] != res[c]["attempted"]]
    print(f"\n[vocab-ceiling] VERDICT: {verdict}  abstention={'PASS' if abstain_pass else 'FAIL'}  "
          f"shuffled={'PASS' if shuffle_pass else 'FAIL'}  failing_caps={failing}")

    out = {
        "probe": "vocab_ceiling", "V": args.V, "seed": args.seed, "D": args.D,
        "backend": backend_name, "spiking_cleanup": bool(args.spiking_cleanup),
        "matrix": res, "two_attr_D_sweep": two_attr_sweep,
        "verdict": verdict, "abstention_pass": abstain_pass, "shuffled_pass": shuffle_pass,
        "failing_caps": failing,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[vocab-ceiling] wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
