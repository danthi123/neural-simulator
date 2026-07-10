"""D3 EVENT QA -> MULTI-TURN coherence: the composed running EVENT persists ACROSS conversational turns and SURVIVES
intervening questions, so a turn-2 pronoun resolves to a turn-1 referent. The deployed "talk to the brain" property.

The single-turn QA rungs answered "what does HE eat?" over ONE discourse. Real conversation is multi-turn: the running
meaning must carry across turn boundaries, and asking a question must not disturb it. THIS rung tests both on the LIVE
`MultiTurnAgent`:

    TURN 1:  "dog chase cat."   "he chase fish."          -> running event: agent=dog (coref-DEEP), patient=fish
    ASK:     "what does he eat?"  -> dog's eat-fact                     (a QUESTION between turns)
    TURN 2:  "he chase bird."   "it flee worm."           -> coref carries DOG across the boundary; then PROMOTE
                                                              (agent <- bird, the prev patient), patient=worm
    ASK:     "what does he eat?"  -> BIRD's eat-fact       (the answer CHANGED with the composed event)

The turn-2 "he" has NO antecedent inside turn 2 -- it can only resolve through state carried from turn 1. And the
intervening question must leave the event untouched (query-invariance).

ANTI-CHEATS (6-seed): (a) multi-turn QA >> the RESET control (register reset at each turn boundary -> the turn-2 coref
loses its antecedent); (b) >> RECENCY (the last-mentioned entity's eat-fact); (c) >> FLAT-FACT (the last literal subject
is a pronoun -> unresolved); (d) QUERY-INVARIANCE: the running (agent, patient) is byte-identical before/after a
question (asserted every query); (e) `--spiking` maintains the event on two FS-WTA slots. Reuse-by-import
(`D3EventRegister` + `MultiTurnAgent.what_does_agent_now`); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_multiturn_qa_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_agent_derisk import D3EventRegister
from research.runners.multi_turn_agent import MultiTurnAgent


COREF = ("he", "she", "they"); PROMOTE = ("it",)


def make_conversation(rng, referents, n_turns=3, clauses_per_turn=2):
    """A multi-turn conversation. Turn 0 opens with an INTRODUCE; EVERY LATER TURN OPENS WITH A COREF/PROMOTE, so the
    turn's first clause can only resolve through state carried across the boundary.

    Returns turns, truth (the true agent after each turn), and CROSS (per turn: does the correct answer REQUIRE state
    carried across the turn boundary?). CROSS is computed by re-simulating the turn from a RESET state: if the resulting
    agent differs from the true agent, the query genuinely depends on the prior turn. This is the load-bearing subset --
    a turn whose LAST clause is an intro (names the agent) or a promote (binds the in-turn observed patient) needs no
    cross-boundary state, and would dilute the metric."""
    turns = []; truth = []; cross = []
    a = p = None
    for t in range(n_turns):
        clauses = []; ops = []
        for c in range(clauses_per_turn):
            o = referents[rng.randint(len(referents))]
            first_of_turn = (c == 0)
            if t == 0 and c == 0:
                op = "intro"                                  # the conversation must SET an agent
            elif first_of_turn:
                op = "coref" if rng.rand() < 0.6 else "promote"   # cross-boundary dependency (no in-turn antecedent)
            else:
                r = rng.rand()
                op = "coref" if r < 0.5 else ("promote" if r < 0.75 else "intro")
            if op == "coref":
                s = "he"
            elif op == "promote":
                s = "it"; a = p
            else:
                s = referents[rng.randint(len(referents))]; a = s
            p = o
            clauses.append((s, "chase" if op != "promote" else "flee", o)); ops.append((op, s, o))
        # re-simulate this turn from the RESET state (what the register would hold with no carry-over)
        a_r = p_r = referents[0]                              # ident slot == referents[0]
        for (op, s, o) in ops:
            if op == "promote":
                a_r = p_r
            elif op == "intro":
                a_r = s
            p_r = o
        turns.append(clauses); truth.append(a); cross.append(a_r != a)
    return turns, truth, cross


def run_seed(seed, spiking, n_convs=40):
    referents = ["dog", "cat", "fish", "bird", "worm", "ball"]
    rng = np.random.RandomState(seed + 3)
    food = np.random.RandomState(seed + 55).permutation(len(referents))
    for i in range(len(food)):
        if food[i] == i:
            food[(i + 1) % len(food)], food[i] = food[i], food[(i + 1) % len(food)]
    food_word = {referents[i]: referents[int(food[i])] for i in range(len(referents))}

    vocab = {w: None for w in (referents + ["chase", "flee", "eat"])}

    def build():
        reg = D3EventRegister(referents, seed=seed, spiking=spiking)
        ag = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                            event_register=reg, enable_neural_render=False)
        for r in referents:                                   # TEACH the eat-KB (separate knowledge)
            ag.hear(f"{r} eat {food_word[r]}")
        reg.reset()                                           # clear the running event; the KB persists
        return ag, reg

    agent, register = build()
    agent_rst, register_rst = build()                          # the RESET control (fresh event each turn)

    mt_ok = rst_ok = rec_ok = flat_ok = tot = 0
    x_mt = x_rst = x_rec = x_flat = x_tot = 0                  # the CROSS-TURN subset (the load-bearing one)
    invariance_ok = True
    for _ in range(n_convs):
        turns, truth, cross = make_conversation(rng, referents)
        register.reset(); register_rst.reset()
        for ti, clauses in enumerate(turns):
            if ti > 0:
                register_rst.reset()                           # CONTROL: the event does not carry across the boundary
            for (s, v, o) in clauses:
                agent.hear(f"{s} {v} {o}"); agent_rst.hear(f"{s} {v} {o}")
            true_food = food_word[truth[ti]]
            # QUERY-INVARIANCE: the running event must be untouched by asking
            before = (register.who_agent(), register.who_patient())
            ans = agent.what_does_agent_now("eat")             # the LIVE multi-turn answer
            after = (register.who_agent(), register.who_patient())
            invariance_ok &= (before == after)
            ans_rst = agent_rst.what_does_agent_now("eat")
            last_s, _, last_o = clauses[-1]
            flat = food_word.get(last_s) if last_s not in COREF + PROMOTE else None
            ok = int(ans == true_food); ok_r = int(ans_rst == true_food)
            ok_rec = int(food_word[last_o] == true_food); ok_f = int(flat == true_food)
            mt_ok += ok; rst_ok += ok_r; rec_ok += ok_rec; flat_ok += ok_f; tot += 1
            if cross[ti]:                                      # this query REQUIRES state carried across the boundary
                x_mt += ok; x_rst += ok_r; x_rec += ok_rec; x_flat += ok_f; x_tot += 1
    m = max(tot, 1); xm = max(x_tot, 1)
    return {"seed": seed, "spiking": spiking,
            # the LOAD-BEARING subset: queries whose correct answer requires the event carried across a turn boundary
            "XTURN_QA": round(x_mt / xm, 3), "XTURN_reset": round(x_rst / xm, 3),
            "XTURN_recency": round(x_rec / xm, 3), "XTURN_flat": round(x_flat / xm, 3), "n_xturn": x_tot,
            # the diluted aggregate (reported, not gated: turns ending in an intro/promote need no cross-turn state)
            "ALL_QA": round(mt_ok / m, 3), "ALL_reset": round(rst_ok / m, 3), "ALL_recency": round(rec_ok / m, 3),
            "query_invariance": bool(invariance_ok), "n_queries": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--spiking", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    _sp = " [SPIKING event]" if a.spiking else ""
    print(f"[D3 EVENT MULTI-TURN QA]{_sp} the running EVENT persists ACROSS turns + survives intervening questions; a turn-2 pronoun resolves to a turn-1 referent", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.spiking); rows.append(r)
        print(f"  [seed {s}] CROSS-TURN QA={r['XTURN_QA']} (n={r['n_xturn']}) || reset={r['XTURN_reset']} | recency={r['XTURN_recency']} | flat-fact={r['XTURN_flat']} "
              f"|| [diluted all-query: {r['ALL_QA']} vs reset {r['ALL_reset']}] | query-invariance={r['query_invariance']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        mt, rst, rec, flat = _m("XTURN_QA"), _m("XTURN_reset"), _m("XTURN_recency"), _m("XTURN_flat")
        allq, allr = _m("ALL_QA"), _m("ALL_reset")
        inv = all(r["query_invariance"] for r in rows)
        # GATE: XTURN_reset is ~0 BY CONSTRUCTION (the cross-turn subset is DEFINED as "the reset-simulated agent differs"),
        # so it is a CONSISTENCY CHECK, never a gate term (a tautological gate metric is exactly the defect this project's
        # own adversarial audit caught before). The gate rides on the INDEPENDENT floors: recency + flat-fact. The honest
        # reset evidence is the ALL-QUERY comparison (ALL_QA vs ALL_reset), which is not definitionally rigged.
        go = (mt > 0.75) and (mt - rec > 0.3) and (mt - flat > 0.3) and (allq - allr > 0.2) and inv
        print(f"\n  AGGREGATE -- LOAD-BEARING cross-turn subset: QA={mt:.3f} | recency={rec:.3f} | flat-fact={flat:.3f} | query-invariance={inv}", flush=True)
        print(f"  (reset on the cross-turn subset={rst:.3f} -- ~0 BY CONSTRUCTION, a consistency check, NOT a gate term)", flush=True)
        print(f"  (all-query aggregate -- the non-rigged reset evidence: QA={allq:.3f} vs reset={allr:.3f}; diluted because a turn ending in an intro/promote needs no cross-turn state)", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the composed running EVENT persists ACROSS conversational turns and SURVIVES intervening questions: on the LOAD-BEARING subset (queries whose correct answer REQUIRES state carried across the turn boundary) the deployed agent answers what-does-HE-eat from the running event ('+format(mt,'.2f')+') where the INDEPENDENT floors RECENCY ('+format(rec,'.2f')+') and FLAT-FACT ('+format(flat,'.2f')+') both FAIL; resetting the event at each boundary drops the all-query answer '+format(allq,'.2f')+'->'+format(allr,'.2f')+' (the non-rigged reset evidence); and asking a question leaves the running event byte-identical (query-invariance) -> multi-turn discourse coherence on the deployed brain'+(' ON SPIKES' if a.spiking else '') if go else 'the cross-turn QA did not clearly beat its INDEPENDENT floors (read XTURN vs recency/flat + the all-query reset drop + query-invariance)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
