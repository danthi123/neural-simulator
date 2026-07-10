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
from research.runners.multi_turn_agent import MultiTurnAgent


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


class D3EventRegister:
    """The PRODUCTION-WIRE adapter: maintains the running FACTORED (agent, patient) EVENT over the SVO facts a real
    `MultiTurnAgent` hears. `observe(subject_word, object_word)` maps the raw words to the D3 event encoding (an entity
    subject = INTRODUCE; 'he'/'she'/'they' = AGENT-COREF, the agent persists; 'it' = PROMOTE) and folds it via the D3
    discrete-attractor (host, or spiking FS-WTA). `who_agent()`/`who_patient()` return the composed referent NAMES.
    Mirrors the anaphora `D3CenteringFocusSource` adapter -> plugs into `MultiTurnAgent(event_register=...)`."""
    def __init__(self, referents, seed=42, spiking=False, n_hid=192, epochs=60,
                 coref_words=("he", "she", "they"), promote_words=("it",)):
        self.referents = list(referents); self.ref2idx = {r: i for i, r in enumerate(referents)}
        K = len(referents)
        task = make_event_task(seed, K=K, n_per_len=2000, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
        W = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=False)["weights"]
        self.ent = task["ent"]; self.HE = task["HE"]; self.IT = task["IT"]; self.op_intro = task["op_intro"]
        self.coref = set(coref_words); self.promote = set(promote_words)
        sb_a = build_fswta_score_bridge(seed=seed, K=K) if spiking else None
        sb_p = build_fswta_score_bridge(seed=seed + 7, K=K) if spiking else None
        self.agent = EventDiscourseAgent(W, K, task["ident"], sb_a=sb_a, sb_p=sb_p)

    def reset(self):
        self.agent.reset()

    def is_pronoun_subject(self, word):
        """True if the subject is a coref/promote marker ('he'/'it') the flat-fact composer cannot store as an entity."""
        w = (word or "").lower()
        return w in self.coref or w in self.promote

    def observe(self, subject_word, object_word):
        o = self.ref2idx.get(object_word)
        if o is None:
            return                                                # unknown patient -> skip (moat: no confabulation)
        sw = (subject_word or "").lower()
        if sw in self.coref:                                      # "he ..." -> AGENT-COREF (the agent persists, goes deep)
            subj_code = self.HE
        elif sw in self.promote:                                  # "it ..." -> PROMOTE (the patient promotes to agent)
            subj_code = self.IT
        else:
            s = self.ref2idx.get(sw)
            if s is None:
                return
            subj_code = self.ent[s]                               # entity subject -> INTRODUCE
        code = np.concatenate([subj_code, self.ent[o], self.op_intro]).astype(np.float32)
        self.agent.hear(code)                                     # fold via the D3 transition (host / spiking FS-WTA)

    def who_agent(self):
        return self.referents[self.agent.who_agent()]

    def who_patient(self):
        return self.referents[self.agent.who_patient()]


def run_seed_wire(seed, spiking):
    """The PRODUCTION WIRE: a REAL `MultiTurnAgent` with a `D3EventRegister` hears a deep-coref discourse; the agent
    ANSWERS who_agent_now() from the running event where the flat-fact + recency baselines fail. Mirrors the anaphora
    live-agent wire (a real agent object, the D3 adapter plugged into an additive hook)."""
    referents = ["dog", "cat", "fish", "bird", "worm", "ball"]
    vocab = {w: None for w in (referents + ["chase"])}
    register = D3EventRegister(referents, seed=seed, spiking=spiking)
    agent = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                           event_register=register, enable_neural_render=False)
    idx = {r: i for i, r in enumerate(referents)}
    coref = ("he", "she", "they"); promote = ("it",)
    # DEEP discourses: the agent is set at a VARYING depth (intro/promote) then PERSISTS via 'he' corefs -> the final
    # agent is deep (not the last-mentioned, not the last literal subject "he").
    scenarios = [
        [("dog", "cat"), ("he", "fish"), ("he", "bird"), ("he", "worm")],          # agent = dog (set@0, persists)
        [("cat", "worm"), ("bird", "ball"), ("he", "dog"), ("he", "fish")],        # agent = bird (set@1, persists)
        [("worm", "cat"), ("he", "ball"), ("it", "dog"), ("he", "fish")],          # promote@2: agent<-prev patient
        [("bird", "dog"), ("he", "cat"), ("he", "worm"), ("he", "ball")],          # agent = bird (set@0, persists)
    ]
    ev_ok = flat_ok = rec_ok = tot = 0
    for facts in scenarios:
        register.reset()
        a = p = None
        for (s, o) in facts:
            agent.hear(f"{s} chase {o}")                          # the REAL agent hears each clause
            # roll the ground-truth event (the op semantics):
            if s in coref:
                a = a; p = idx[o]
            elif s in promote:
                a = p; p = idx[o]
            else:
                a = idx[s]; p = idx[o]
        true_agent = referents[a]
        ev = agent.who_agent_now()                               # the agent's answer FROM the running event
        last_s = facts[-1][0]
        flat = last_s if last_s not in coref + promote else None  # flat-fact: the last literal subject (coref -> None)
        recency = facts[-1][1]                                    # the last-mentioned entity (last object)
        ev_ok += int(ev == true_agent); flat_ok += int(flat == true_agent); rec_ok += int(recency == true_agent); tot += 1
    m = max(tot, 1)
    return {"seed": seed, "spiking": spiking, "EVENT_who_agent": round(ev_ok / m, 3),
            "FLAT_FACT_agent": round(flat_ok / m, 3), "RECENCY_agent": round(rec_ok / m, 3)}


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
    ap.add_argument("--wire", action="store_true", help="the PRODUCTION WIRE: a real MultiTurnAgent + D3EventRegister")
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    _sp = " [SPIKING event]" if a.spiking else ""
    if a.wire:
        print(f"[D3 EVENT -> LIVE MultiTurnAgent]{_sp} the DEPLOYED agent answers who_agent_now() from a D3EventRegister maintaining the running event over the SVO facts it hears", flush=True)
        rows = []
        for s in seeds:
            r = run_seed_wire(s, a.spiking); rows.append(r)
            print(f"  [seed {s}] LIVE-agent EVENT who-agent={r['EVENT_who_agent']} || FLAT-FACT(last subject)={r['FLAT_FACT_agent']} || RECENCY={r['RECENCY_agent']}", flush=True)
        if a.json and rows:
            import json
            json.dump(rows, open(a.json, "w"), indent=1)
        if rows:
            def _m(k): return float(np.mean([r[k] for r in rows]))
            ev, flat, rec = _m("EVENT_who_agent"), _m("FLAT_FACT_agent"), _m("RECENCY_agent")
            go = (ev > 0.75) and (ev - flat > 0.3) and (ev - rec > 0.3)
            print(f"\n  AGGREGATE (wire): LIVE-agent EVENT who-agent={ev:.3f} | FLAT-FACT={flat:.3f} | RECENCY={rec:.3f}", flush=True)
            print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the DEPLOYED MultiTurnAgent answers who_agent_now() from a D3EventRegister maintaining the running event over the SVO facts it hears ('+format(ev,'.2f')+') where the FLAT-FACT retriever (the last literal subject = an unresolved coref) FAILS ('+format(flat,'.2f')+') and RECENCY FAILS ('+format(rec,'.2f')+') -> the running-event register is WIRED into the production agent (additive default-off event_register hook) alongside its flat-fact store = the anti-RAG payoff on the real agent'+(' ON SPIKES' if a.spiking else '') if go else 'the live-agent wire did not clearly beat flat-fact + recency (read the gaps)'}. NO sim/ edit.", flush=True)
        return
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
