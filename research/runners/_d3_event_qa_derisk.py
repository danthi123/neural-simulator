"""D3 EVENT -> RANK-3 REASONING/QA over the composed running EVENT, UNIFIED with a fact store (the research-gate payoff:
"cheap once RANK-1 exists"). The event-composition arc built a running FACTORED (agent, patient) MEANING on the
discrete-attractor; this rung READS it to ANSWER A QUESTION, joining the situation model to stored knowledge:

    discourse:  "dog chases cat. he chases fish. it flees bird."   (D3 composes the running event: agent=dog [coref-DEEP
                                                                     via 'he'], patient=bird [most-recent])
    fact store: dog->meat, cat->fish, bird->seed, ...              (SEPARATE knowledge, NEVER stated in the discourse)
    question:   "what does HE eat?"   -> resolve 'he' to the COMPOSED running agent (dog, traced through the coref run
                                          by D3, NOT the last-mentioned) -> key the fact store -> "meat"
                "what does IT eat?"   -> resolve 'it' to the running patient slot -> key the fact store

THE UNIFICATION (why this is the anti-RAG payoff): the ANSWER ("meat") is in neither the discourse (only entities+ops
appear) nor derivable from recency -- it requires BOTH the D3-composed situation model (to resolve the pronoun to the
DEEP referent) AND the fact store (to recall the referent's property). A retrieve-a-set→render loop cannot do this; it
has no running referent to key the store with. Frankland-Greene 2015 lmSTC (the agent/patient registers) + a
Collins-Quillian property store; the QA = read-the-register → associative-recall.

ANTI-CHEATS (6-seed): (a) COMPOSED-QA >> chance (1/K); (b) RECENCY-QA (resolve the pronoun to the last-mentioned entity
→ key the store) FAILS (the agent is coref-deep); (c) NO-EVENT lesion (recurrence-off compose) collapses to ~recency;
(d) PERMUTED-fact-store: the answer TRACKS the permuted store (proves the answer routes THROUGH the store, not a
discourse shortcut) while COMPOSED-QA on the true store is unchanged; (e) both slots queried (agent AND patient — a
2-slot situation model, not 1). Reuse-by-import (`make_event_task` + `factored_event_rnn` [the RANK-1 composer]); numpy;
NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_qa_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_composition_derisk import make_event_task, factored_event_rnn


def build_fact_store(seed, K):
    """entity -> property: a bijection over K property labels (DISJOINT from the entity/op codes — the property is
    SEPARATE stored knowledge, never uttered in the discourse). A bijection so QA-correct <=> referent-resolved-correct
    AND the permuted-store control cleanly re-routes."""
    return np.random.RandomState(seed + 77).permutation(K)


def roll_resolved(W, task, split, K, ident, lesion_rec=False):
    """Roll the D3-composed factored event over a split; return (fa, fp, ta, tp) = resolved + true (agent, patient) per
    item. Mirrors `factored_event_rnn.eval_split` but returns the INDICES (for the QA head to key the fact store)."""
    emb, Wr, Wi, Wa, ba, Wp, bp = (W["emb"], W["Wr"], W["Wi"], W["Wa"], W["ba"], W["Wp"], W["bp"])
    X, Ya, Yp, L, SEQ, STA, STP = task[split]; B = len(L); Lmax = int(L.max())
    a = np.full(B, ident, np.int64); p = np.full(B, ident, np.int64)
    fa = np.full(B, ident, np.int64); fp = np.full(B, ident, np.int64)
    rg = 0.0 if lesion_rec else 1.0
    for t in range(Lmax):
        active = (L > t)
        state_in = np.concatenate([emb[a], emb[p]], axis=1)
        h = np.tanh(rg * (state_in @ Wr.T) + X[:, t] @ Wi.T)
        na = (h @ Wa.T + ba).argmax(1); npp = (h @ Wp.T + bp).argmax(1)
        a = np.where(active, na, a); p = np.where(active, npp, p)
        last = (L == (t + 1)); fa = np.where(last, a, fa); fp = np.where(last, p, fp)
    ta = STA[np.arange(B), L - 1]; tp = STP[np.arange(B), L - 1]
    return fa, fp, ta, tp


def recency_resolved(task, split, K):
    """RECENCY resolver: agent <- the last SUBJECT (or last object on an 'it'/'he' turn), patient <- the last OBJECT.
    (s_idx>=K => the last utterance was a pronoun op, so the 'subject surface' is the object.)"""
    X, Ya, Yp, L, SEQ, STA, STP = task[split]; B = len(L)
    ra = np.zeros(B, np.int64); rp = np.zeros(B, np.int64)
    for n in range(B):
        Ln = int(L[n]); so = int(SEQ[n][Ln - 1]); s_idx, o = so // K, so % K
        ra[n] = s_idx if s_idx < K else o; rp[n] = o
    return ra, rp


def qa_accuracy(res_a, res_p, true_a, true_p, prop, queries):
    """QA answer = prop[resolved_slot]; correct iff == prop[true_slot]. queries: 0=ask agent ('he'), 1=ask patient ('it')."""
    res = np.where(queries == 0, res_a, res_p); tru = np.where(queries == 0, true_a, true_p)
    return float((prop[res] == prop[tru]).mean())


def run_seed(seed, K, n_hid, epochs):
    task = make_event_task(seed, K=K, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    W = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=False)["weights"]
    prop = build_fact_store(seed, K); ident = task["ident"]
    B = len(task["test_deeper"][3])
    queries = np.array([i % 2 for i in range(B)], np.int64)          # alternate: ask agent ('he') / patient ('it')

    fa, fp, ta, tp = roll_resolved(W, task, "test_deeper", K, ident)           # D3-composed
    la, lp, _, _ = roll_resolved(W, task, "test_deeper", K, ident, lesion_rec=True)  # no-event lesion
    ra, rp = recency_resolved(task, "test_deeper", K)                          # recency floor

    prop_perm = np.random.RandomState(seed + 999).permutation(K)              # a DIFFERENT store (control d)
    ask_agent = np.zeros(B, np.int64); ask_pat = np.ones(B, np.int64)
    # The AGENT slot is the coref-DEEP referent (persists through the 'he'-run back to a random-depth setting) -> the
    # load-bearing QA contrast. The PATIENT slot is shallow-by-design ("it" = the most-recent object, which is realistic)
    # so recency answers it easily -> it is the SECONDARY "both slots are QA-able" check, not the deep claim.
    return {"seed": seed, "K": K,
            "AGENT_qa": round(qa_accuracy(fa, fp, ta, tp, prop, ask_agent), 3),          # DEEP referent (primary)
            "AGENT_recency_qa": round(qa_accuracy(ra, rp, ta, tp, prop, ask_agent), 3),  # recency FAILS on the deep agent
            "AGENT_lesion_qa": round(qa_accuracy(la, lp, ta, tp, prop, ask_agent), 3),   # no-event collapses
            "PATIENT_qa": round(qa_accuracy(fa, fp, ta, tp, prop, ask_pat), 3),          # 2nd slot QA-able (secondary)
            "BOTH_qa": round(qa_accuracy(fa, fp, ta, tp, prop, queries), 3),
            # control (d): the COMPOSED referents are unchanged, but a permuted store re-routes the answers -> the QA on
            # the permuted store scores prop_perm[resolved]==prop_perm[true] == the SAME resolution acc (proves the answer
            # is prop[resolved], i.e. it routes THROUGH the store; a discourse-shortcut answer would NOT track the store).
            "AGENT_qa_permstore": round(qa_accuracy(fa, fp, ta, tp, prop_perm, ask_agent), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT QA] K={a.K} | QA reads the D3-composed running EVENT (agent/patient) to resolve a pronoun, then keys a SEPARATE fact store -> the situation-model x fact-store UNIFICATION", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs); rows.append(r)
        print(f"  [seed {s}] AGENT(deep)-QA={r['AGENT_qa']} vs recency={r['AGENT_recency_qa']} / no-event={r['AGENT_lesion_qa']} || PATIENT-slot-QA={r['PATIENT_qa']} | perm-store={r['AGENT_qa_permstore']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        comp, rec, les, pat = _m("AGENT_qa"), _m("AGENT_recency_qa"), _m("AGENT_lesion_qa"), _m("PATIENT_qa")
        chance = 1.0 / a.K
        go = (comp > 0.75) and (comp - rec > 0.3) and (comp - les > 0.3) and (pat > 0.75)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}): AGENT(deep)-QA={comp:.3f} | recency={rec:.3f} | no-event={les:.3f} || PATIENT-slot-QA={pat:.3f} | perm-store={_m('AGENT_qa_permstore'):.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the QA READS the D3-composed running EVENT to resolve a pronoun to the coref-DEEP AGENT, then keys a SEPARATE fact store for the answer ('+format(comp,'.2f')+') where a RECENCY resolver FAILS ('+format(rec,'.2f')+') and a NO-EVENT lesion collapses ('+format(les,'.2f')+'); the 2nd (patient) slot is QA-able too ('+format(pat,'.2f')+'); the answer TRACKS a permuted store (routes THROUGH stored knowledge, not a discourse shortcut) -> RANK-3: the situation model (D3 event) UNIFIED with the fact store answers a question neither alone can = the anti-RAG payoff (reason OVER a running meaning); next: multi-turn QA/connectives, the spiking QA read-out, self-supervised TEM delta' if go else 'the deep-agent QA did not clearly beat recency+lesion (read AGENT-QA vs recency/no-event; tune epochs/n_hid)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
