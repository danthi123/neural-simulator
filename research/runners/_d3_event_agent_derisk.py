"""D3 EVENT COMPOSITION — the DEPLOYMENT: a conversational agent ANSWERS who/what from the running EVENT (the anti-RAG
payoff). The event-composition rungs proved the discrete-attractor maintains a running FACTORED (agent, patient) MEANING
(deep, weak-supervisable, spiking). THIS deploys it as a conversational interface: an `EventDiscourseAgent` HEARS a
multi-clause discourse (relational events incl. AGENT-COREF "he ..."), maintains the running event ON SPIKES (two FS-WTA
slots), and ANSWERS a query — "who is the agent now?" / "who is the patient now?" — from the COMPOSED event. On a deep-
coref discourse ("the dog chased the cat. he chased the fish. he chased the bird.") the agent resolves "who is chasing
now?" -> DOG (the deep-tracked agent, persisted through the coref run), where the two mechanisms the current
conversational stack uses FAIL: a FLAT-FACT retriever answers the last clause's literal subject ("he", unresolved) and a
RECENCY resolver answers the last-mentioned entity (the last object) -> neither composes the running event.

⇒ this is the anti-RAG demonstration: the agent answers about the COMPOSED MEANING it maintains across the discourse, not
a retrieved/last-mentioned fact. Mirrors the anaphora live-agent wire (D3's composed focus drove the real biased
competition); here D3's composed EVENT drives the who/what answer.

ANTI-CHEATS: (a) the agent's EVENT answer >> the FLAT-FACT baseline (last clause's literal subject — "he" unresolved) and
>> the RECENCY baseline (last-mentioned) on deep-coref discourses; (b) held-out-DEEPER discourses (longer than training);
(c) fully SPIKING (the running event on two FS-WTA slots); (d) multi-seed dev+blind. Reuse-by-import (`make_event_task`
+ `factored_event_rnn` + `build_fswta_score_bridge`/`fswta_drive`); numpy backend; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_agent_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_composition_derisk import make_event_task, factored_event_rnn
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive


class EventDiscourseAgent:
    """Maintains a running FACTORED (agent, patient) EVENT across a heard discourse and answers who/what from it. The
    transition is the rate-learned `factored_event_rnn` δ; the re-discretization is ON SPIKES (two FS-WTA slots) when
    `spiking=True`. `hear(code)` folds one utterance into the running event; `who_agent()`/`who_patient()` read it."""
    def __init__(self, W, K, ident, sb_a=None, sb_p=None, settle=25):
        self.W = W; self.K = K; self.ident = ident; self.a = ident; self.p = ident
        self.sb_a = sb_a; self.sb_p = sb_p; self.settle = settle

    def reset(self):
        self.a = self.ident; self.p = self.ident

    def hear(self, code):
        emb, Wr, Wi = self.W["emb"], self.W["Wr"], self.W["Wi"]; Wa, ba, Wp, bp = self.W["Wa"], self.W["ba"], self.W["Wp"], self.W["bp"]
        h = np.tanh(np.concatenate([emb[self.a], emb[self.p]]) @ Wr.T + code @ Wi.T)
        a_scores = h @ Wa.T + ba; p_scores = h @ Wp.T + bp
        if self.sb_a is None:                                     # host-argmax re-discretization
            self.a = int(np.argmax(a_scores)); self.p = int(np.argmax(p_scores))
        else:                                                     # SPIKING FS-WTA re-discretization (two slots)
            _, acc_a = fswta_drive(self.sb_a, self.K, a_scores, settle=self.settle)
            _, acc_p = fswta_drive(self.sb_p, self.K, p_scores, settle=self.settle)
            self.a = int(np.argmax(acc_a)) if acc_a.max() > 0 else self.ident
            self.p = int(np.argmax(acc_p)) if acc_p.max() > 0 else self.ident

    def who_agent(self):
        return self.a

    def who_patient(self):
        return self.p


def run_seed(seed, K, n_hid, epochs, spiking, settle):
    task = make_event_task(seed, K=K, n_per_len=2000, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    W = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=False)["weights"]
    sb_a = build_fswta_score_bridge(seed=seed, K=K) if spiking else None
    sb_p = build_fswta_score_bridge(seed=seed + 7, K=K) if spiking else None
    agent = EventDiscourseAgent(W, K, task["ident"], sb_a=sb_a, sb_p=sb_p, settle=settle)

    X, Ya, Yp, L, SEQ, STA, STP = task["test_deeper"]
    rng = np.random.RandomState(seed + 5)
    idx = rng.choice(len(L), min(60, len(L)), replace=False)
    ev_agent_ok = flat_agent_ok = recency_agent_ok = ev_patient_ok = tot = 0
    for n in idx:
        agent.reset()
        for t in range(int(L[n])):
            agent.hear(X[n, t])                                   # the agent HEARS each clause -> running event
        true_a = int(STA[n, int(L[n]) - 1]); true_p = int(STP[n, int(L[n]) - 1])
        # the agent's EVENT answer:
        ev_agent_ok += int(agent.who_agent() == true_a); ev_patient_ok += int(agent.who_patient() == true_p)
        # FLAT-FACT baseline: answer the last clause's LITERAL subject (s_idx = SEQ//K); on a coref/promote it is the
        # marker (>=K) = UNRESOLVED -> cannot name an entity (counts as wrong):
        s_last = int(SEQ[n][int(L[n]) - 1]) // K
        flat_agent_ok += int(s_last < K and s_last == true_a)
        # RECENCY baseline: answer the most-recently-mentioned entity (the last object):
        o_last = int(SEQ[n][int(L[n]) - 1]) % K
        recency_agent_ok += int(o_last == true_a)
        tot += 1
    m = max(tot, 1)
    return {"seed": seed, "K": K, "spiking": spiking,
            "EVENT_who_agent": round(ev_agent_ok / m, 3), "EVENT_who_patient": round(ev_patient_ok / m, 3),
            "FLAT_FACT_agent": round(flat_agent_ok / m, 3), "RECENCY_agent": round(recency_agent_ok / m, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--spiking", action="store_true", help="maintain the running event ON SPIKES (two FS-WTA slots)")
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    _sp = " [SPIKING event]" if a.spiking else ""
    print(f"[D3 EVENT AGENT]{_sp} K={a.K} | a conversational agent ANSWERS 'who is the agent/patient now?' from the running EVENT it maintains across a discourse (vs flat-fact + recency)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs, a.spiking, a.settle); rows.append(r)
        print(f"  [seed {s}] EVENT who-agent={r['EVENT_who_agent']} (who-patient={r['EVENT_who_patient']}) || "
              f"FLAT-FACT(last subject)={r['FLAT_FACT_agent']} || RECENCY(last entity)={r['RECENCY_agent']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        ev, flat, rec = _m("EVENT_who_agent"), _m("FLAT_FACT_agent"), _m("RECENCY_agent")
        go = (ev > 0.75) and (ev - flat > 0.3) and (ev - rec > 0.3)
        print(f"\n  AGGREGATE (K={a.K}): EVENT who-agent={ev:.3f} | FLAT-FACT(last subject)={flat:.3f} | RECENCY(last entity)={rec:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the conversational agent ANSWERS who-is-the-agent-now from the running EVENT it maintains across the discourse ('+format(ev,'.2f')+') where a FLAT-FACT retriever (the last clause literal subject = an UNRESOLVED coref marker) FAILS ('+format(flat,'.2f')+') and a RECENCY resolver (the last-mentioned entity) FAILS ('+format(rec,'.2f')+') -> the agent answers about the COMPOSED MEANING it tracks, NOT a retrieved/last-mentioned fact = the anti-RAG payoff DEPLOYED'+(' ON SPIKES' if a.spiking else '')+'; next: fold the running-event register into the production MultiTurnAgent alongside the flat-fact store' if go else 'the event-agent answer did not clearly beat flat-fact + recency (read the gaps; tune epochs/n_hid)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
