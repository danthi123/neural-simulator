"""D3 EVENT QA -> the LIVE MultiTurnAgent (the RANK-3 QA DEPLOYED end-to-end): a real `MultiTurnAgent` ANSWERS
"what does HE eat?" over a running discourse -- resolving the coref-DEEP pronoun via its D3 event register, then querying
its OWN fact-store KB. The unification, on the production agent.

THE FLOW (all on the real agent object):
  1. TEACH separate knowledge:  "dog eat worm. cat eat ball. bird eat fish. ..."   -> the composer KB (the fact store)
  2. RESET the event register (clear the running event; the KB persists).
  3. HEAR a deep-coref discourse: "dog chase cat. he chase fish. he chase bird. he chase worm."  -> the D3 event register
     composes the running event: agent = DOG (set@0, persisted through the 'he' run -- coref-DEEP, NOT last-mentioned).
  4. ASK "what does HE eat?"  -> `what_does_agent_now("eat")`: resolve 'he' -> the running agent (dog, via the register)
     -> query the KB (dog eat ?) -> "worm".

The answer needs BOTH the running event (to resolve 'he' to the deep agent -- the chase-discourse never says who "he" is
as an entity) AND the fact store (the eat-KB, taught separately, never in the chase-discourse). A FLAT-FACT retriever
("he" unresolved -> no KB entry) and a RECENCY resolver ('he' = the last-mentioned -> the WRONG agent's eat-fact) both
FAIL. Mirrors the event-agent wire (`_d3_event_agent_derisk --wire`, who_agent_now) but now the answer is a PROPERTY
keyed by the composed referent = the situation-model x fact-store QA DEPLOYED.

ANTI-CHEATS (6-seed): (a) the LIVE agent's EVENT-QA >> FLAT-FACT (unresolved 'he') and >> RECENCY (last-mentioned's
eat-fact); (b) the eat-KB is SEPARATE (never in the chase-discourse) -> the answer is not readable from the discourse;
(c) `--spiking` maintains the running event on two FS-WTA slots (the whole resolve on spikes). Reuse-by-import
(`D3EventRegister` + `MultiTurnAgent.what_does_agent_now`); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_qa_wire_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_agent_derisk import D3EventRegister
from research.runners.multi_turn_agent import MultiTurnAgent


SCENARIOS = [   # (subject, object) clauses; verb = "chase". The agent is set then PERSISTS via 'he' -> coref-DEEP.
    [("dog", "cat"), ("he", "fish"), ("he", "bird"), ("he", "worm")],       # agent = dog (set@0, persists)
    [("cat", "worm"), ("bird", "ball"), ("he", "dog"), ("he", "fish")],     # agent = bird (set@1, persists)
    [("worm", "cat"), ("he", "ball"), ("it", "dog"), ("he", "fish")],       # promote@2: agent <- prev patient (worm->? )
    [("bird", "dog"), ("he", "cat"), ("he", "worm"), ("he", "ball")],       # agent = bird (set@0, persists)
]


def _true_agent(facts, idx, coref, promote):
    a = p = None
    for (s, o) in facts:
        if s in coref:
            p = idx[o]                          # agent persists
        elif s in promote:
            a = p; p = idx[o]
        else:
            a = idx[s]; p = idx[o]
    return a


def run_seed_qa(seed, spiking):
    referents = ["dog", "cat", "fish", "bird", "worm", "ball"]
    idx = {r: i for i, r in enumerate(referents)}
    coref = ("he", "she", "they"); promote = ("it",)
    # the eat-KB: a derangement (food[r] != r), distinct per agent so a wrong referent -> a wrong eat-answer
    food = np.random.RandomState(seed + 55).permutation(len(referents))
    for i in range(len(food)):
        if food[i] == i:
            food[(i + 1) % len(food)], food[i] = food[i], food[(i + 1) % len(food)]
    food_word = {referents[i]: referents[int(food[i])] for i in range(len(referents))}

    vocab = {w: None for w in (referents + ["chase", "eat"])}
    register = D3EventRegister(referents, seed=seed, spiking=spiking)
    agent = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                           event_register=register, enable_neural_render=False)
    # 1. TEACH the eat-KB (separate knowledge; folds into the register too, but we reset before the discourse)
    for r in referents:
        agent.hear(f"{r} eat {food_word[r]}")

    ev_ok = flat_ok = rec_ok = tot = 0
    for facts in SCENARIOS:
        register.reset()                                            # 2. clear the running event (KB persists)
        for (s, o) in facts:                                        # 3. HEAR the deep-coref chase discourse
            agent.hear(f"{s} chase {o}")
        ta = _true_agent(facts, idx, coref, promote)
        true_food = food_word[referents[ta]]                        # the correct answer to "what does he eat?"
        # 4. the LIVE agent's EVENT-QA: resolve 'he' via the running event register, then query the eat-KB
        ev_ans = agent.what_does_agent_now("eat")
        # FLAT-FACT baseline: the last literal subject ('he' -> unresolved -> no entity -> no KB answer)
        last_s = facts[-1][0]; flat_ans = food_word.get(last_s) if last_s not in coref + promote else None
        # RECENCY baseline: the last-mentioned entity (last object) -> its eat-fact (wrong agent)
        rec_ans = food_word[facts[-1][1]]
        ev_ok += int(ev_ans == true_food); flat_ok += int(flat_ans == true_food); rec_ok += int(rec_ans == true_food); tot += 1
    m = max(tot, 1)
    return {"seed": seed, "spiking": spiking, "EVENT_QA": round(ev_ok / m, 3),
            "FLAT_FACT_QA": round(flat_ok / m, 3), "RECENCY_QA": round(rec_ok / m, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--spiking", action="store_true", help="maintain the running event ON SPIKES (two FS-WTA slots)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    _sp = " [SPIKING event]" if a.spiking else ""
    print(f"[D3 EVENT QA -> LIVE MultiTurnAgent]{_sp} the DEPLOYED agent answers 'what does HE eat?' -- resolve the coref-DEEP pronoun via its D3 event register, then query its OWN eat-KB", flush=True)
    rows = []
    for s in seeds:
        r = run_seed_qa(s, a.spiking); rows.append(r)
        print(f"  [seed {s}] LIVE-agent EVENT-QA={r['EVENT_QA']} || FLAT-FACT(unresolved 'he')={r['FLAT_FACT_QA']} || RECENCY(last-mentioned's fact)={r['RECENCY_QA']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        ev, flat, rec = _m("EVENT_QA"), _m("FLAT_FACT_QA"), _m("RECENCY_QA")
        go = (ev > 0.75) and (ev - flat > 0.3) and (ev - rec > 0.3)
        print(f"\n  AGGREGATE (wire QA): LIVE-agent EVENT-QA={ev:.3f} | FLAT-FACT={flat:.3f} | RECENCY={rec:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the DEPLOYED MultiTurnAgent ANSWERS what-does-HE-eat ('+format(ev,'.2f')+') by resolving the coref-DEEP pronoun via its D3 event register THEN querying its own eat-KB (taught separately, never in the chase-discourse), where a FLAT-FACT retriever (unresolved he) FAILS ('+format(flat,'.2f')+') and a RECENCY resolver (the last-mentioned entity s eat-fact) FAILS ('+format(rec,'.2f')+') -> the situation-model x fact-store QA is DEPLOYED on the real agent (what_does_agent_now, additive)'+(' ON SPIKES' if a.spiking else '')+' = the anti-RAG payoff: the brain answers a question about the running discourse from the composed meaning, not a retrieved/last-mentioned fact' if go else 'the live-agent QA wire did not clearly beat flat-fact + recency (read the gaps)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
