"""D3 EVENT — THE CAPSTONE: the LIVE `MultiTurnAgent` answers "what does HE eat?" over a running event whose transition
was **NEVER GIVEN A STATE LABEL** and which **RUNS ON SPIKES**. Emergent + spiking + deployed + QA, end-to-end.

The pieces, each already 6-seed GO:
  * the transition delta learned from an agent-emission cross-entropy ALONE, no (agent,patient) label  (selfsup rung)
  * that delta executed on a spiking one-of-K FS-WTA Izhikevich attractor                              (selfsup-spiking)
  * the deployed QA: resolve the coref-DEEP pronoun via the event register, then query the agent's own KB  (QA wire)
THIS rung fuses them: `SelfSupEventRegister` (a drop-in for `D3EventRegister`) whose delta is self-supervised and whose
slot is maintained on spikes, plugged into the real agent's additive `event_register` hook.

THE LABEL-FREE NAMING PROBLEM (and its honest solution). The emergent slot is a PERMUTATION of entity identity -- the
register must map slot -> referent NAME to answer. Fitting that map with the true-agent labels would smuggle the
supervision straight back in. Instead: **INTRODUCE clauses NAME the agent in the observable utterance** ("dog chase
cat"), so the slot->name read-out is fitted from (slot-state-after-an-INTRODUCE, the named subject) pairs alone -- all
observable, zero hidden labels. The whole register is therefore label-free: an emergent delta plus a read-out learned
from what the brain hears. (Biologically: a downstream region learning to read the slot.)

ANTI-CHEATS (6-seed): (a) the LIVE agent's EVENT-QA >> FLAT-FACT (unresolved 'he') and >> RECENCY (the last-mentioned
entity's eat-fact); (b) an EMISSION-SEVERED register (delta trained with the agent->emission link cut) COLLAPSES through
the identical deployment -- so the deployed answer rides a LEARNED delta, not a generic attractor; (c) the eat-KB is
separate knowledge, never uttered in the chase-discourse; (d) `--spiking` maintains the slot on the FS-WTA substrate.
Reuse-by-import (`_d3_event_selfsup_derisk` + `build_fswta_score_bridge`/`fswta_drive` + `MultiTurnAgent`); numpy;
NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_selfsup_capstone_derisk --seeds 42 --spiking
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_derisk import (
    make_selfsup_event_task, train_selfsup, linear_probe, INTRO)
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive
from research.runners.multi_turn_agent import MultiTurnAgent

COREF_W = ("he", "she", "they"); PROMOTE_W = ("it",)


def _sm(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def fit_label_free_slot_names(task, W, K, n_seq=1500):
    """Learn slot -> entity NAME from INTRODUCE clauses ONLY (the subject is named in the observable utterance).
    Zero hidden labels. Returns perm[slot] = entity index."""
    emb, Wr, Wi, Wa, ba = W["emb"], W["Wr"], W["Wi"], W["Wa"], W["ba"]
    S, SI, O, E, OP, Ls, T = task["train"]
    X, Y = [], []
    for n in range(min(n_seq, len(Ls))):
        sa = np.zeros(K, np.float32); sa[task["ident"]] = 1.0
        sp = np.zeros(K, np.float32); sp[task["ident"]] = 1.0
        for t in range(int(Ls[n])):
            h = np.tanh(np.concatenate([sa @ emb, sp @ emb]) @ Wr.T + S[n, t] @ Wi.T)
            sa = _sm(h @ Wa.T + ba)
            if OP[n, t] == INTRO:                    # the utterance NAMED the agent -> an observable (state, name) pair
                oh = np.zeros(K, np.float32); oh[int(np.argmax(sa))] = 1.0
                X.append(oh); Y.append(int(SI[n, t]))
            sp = np.zeros(K, np.float32); sp[int(O[n, t])] = 1.0
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64)
    eye = np.eye(K, dtype=np.float32)
    pred = linear_probe(X, Y, eye, np.arange(K), K)   # read the name of each pure slot
    return pred                                        # perm[slot] = entity idx


class SelfSupEventRegister:
    """Drop-in for `D3EventRegister`, but the transition delta is SELF-SUPERVISED (no state label) and the running agent
    slot is re-discretized ON SPIKES (FS-WTA). `who_agent()` names the slot via the label-free read-out."""

    def __init__(self, referents, seed=42, spiking=True, n_hid=128, epochs=40, theta_peak=3.0,
                 settle=25, fs_inh=9.0, random_emit=False):
        self.referents = list(referents); self.ref2idx = {r: i for i, r in enumerate(referents)}
        K = len(referents); self.K = K
        task = make_selfsup_event_task(seed, K=K, theta_peak=theta_peak)
        roll = train_selfsup(task, seed=seed, n_hid=n_hid, epochs=epochs, random_emit=random_emit)
        self.W = roll.W; self.task = task
        self.ent, self.HE, self.IT = task["ent"], task["HE"], task["IT"]
        self.ident = task["ident"]
        self.perm = fit_label_free_slot_names(task, self.W, K)      # slot -> entity name (label-free)
        self.sb = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh) if spiking else None
        self.settle = settle
        self.a = self.ident; self.p = self.ident

    def reset(self):
        self.a = self.ident; self.p = self.ident

    def is_pronoun_subject(self, word):
        w = (word or "").lower()
        return w in COREF_W or w in PROMOTE_W

    def observe(self, subject_word, object_word):
        o = self.ref2idx.get(object_word)
        if o is None:
            return                                                   # unknown patient -> skip (moat: no confabulation)
        sw = (subject_word or "").lower()
        if sw in COREF_W:
            code = self.HE
        elif sw in PROMOTE_W:
            code = self.IT
        else:
            s = self.ref2idx.get(sw)
            if s is None:
                return
            code = self.ent[s]
        emb, Wr, Wi, Wa, ba = self.W["emb"], self.W["Wr"], self.W["Wi"], self.W["Wa"], self.W["ba"]
        h = np.tanh(np.concatenate([emb[self.a], emb[self.p]]) @ Wr.T + code.astype(np.float32) @ Wi.T)
        la = h @ Wa.T + ba                                           # the SELF-SUPERVISED transition's agent scores
        if self.sb is None:
            self.a = int(np.argmax(la))
        else:                                                        # re-discretize ON SPIKES (one-of-K FS-WTA)
            _, acc = fswta_drive(self.sb, self.K, la, settle=self.settle)
            self.a = int(np.argmax(acc)) if acc.max() > 0 else self.ident
        self.p = o                                                   # the patient is the observed object

    def who_agent(self):
        return self.referents[int(self.perm[self.a])]                # name the emergent slot (label-free read-out)

    def who_patient(self):
        return self.referents[self.p]


def make_scenarios(rng, referents, n=40, lengths=(4, 5, 6), p_coref=0.5, p_promote=0.25):
    """RANDOM deep discourses (not 4 hand-picked ones): the first clause INTRODUCES, later clauses coref/promote/intro.
    A hand-picked handful gives 0.25-resolution and makes both the estimate and the severed control unreadable -- the
    emergent register has a genuine ~0.87 error rate, so it needs a proper sample."""
    out = []
    for _ in range(n):
        L = int(rng.choice(lengths)); facts = []
        for t in range(L):
            o = referents[rng.randint(len(referents))]
            r = rng.rand()
            if t == 0:
                s = referents[rng.randint(len(referents))]
            elif r < p_coref:
                s = "he"
            elif r < p_coref + p_promote:
                s = "it"
            else:
                s = referents[rng.randint(len(referents))]
            facts.append((s, o))
        out.append(facts)
    return out


def _true_agent(facts, idx):
    a = p = None
    for (s, o) in facts:
        if s in COREF_W:
            pass
        elif s in PROMOTE_W:
            a = p
        else:
            a = idx[s]
        p = idx[o]
    return a


def run_seed(seed, spiking):
    referents = ["dog", "cat", "fish", "bird", "worm", "ball"]
    idx = {r: i for i, r in enumerate(referents)}
    food = np.random.RandomState(seed + 55).permutation(len(referents))
    for i in range(len(food)):
        if food[i] == i:
            food[(i + 1) % len(food)], food[i] = food[i], food[(i + 1) % len(food)]
    food_word = {referents[i]: referents[int(food[i])] for i in range(len(referents))}
    vocab = {w: None for w in (referents + ["chase", "eat"])}

    def build(random_emit=False):
        reg = SelfSupEventRegister(referents, seed=seed, spiking=spiking, random_emit=random_emit)
        ag = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                            event_register=reg, enable_neural_render=False)
        for r in referents:
            ag.hear(f"{r} eat {food_word[r]}")                        # TEACH the eat-KB (separate knowledge)
        reg.reset()
        return ag, reg

    agent, register = build()
    agent_sev, register_sev = build(random_emit=True)                 # EMISSION-SEVERED control (delta learned nothing)
    scen = make_scenarios(np.random.RandomState(seed + 17), referents)

    ev_ok = sev_ok = flat_ok = rec_ok = tot = 0
    for facts in scen:
        register.reset(); register_sev.reset()
        for (s, o) in facts:
            agent.hear(f"{s} chase {o}"); agent_sev.hear(f"{s} chase {o}")
        true_food = food_word[referents[_true_agent(facts, idx)]]
        ev_ok += int(agent.what_does_agent_now("eat") == true_food)
        sev_ok += int(agent_sev.what_does_agent_now("eat") == true_food)
        last_s, last_o = facts[-1]
        flat = food_word.get(last_s) if last_s not in COREF_W + PROMOTE_W else None
        flat_ok += int(flat == true_food); rec_ok += int(food_word[last_o] == true_food); tot += 1
    m = max(tot, 1)
    return {"seed": seed, "spiking": spiking, "CAPSTONE_QA": round(ev_ok / m, 3),
            "emission_severed_QA": round(sev_ok / m, 3), "FLAT_FACT": round(flat_ok / m, 3),
            "RECENCY": round(rec_ok / m, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--spiking", action="store_true", help="maintain the emergent slot on the FS-WTA substrate")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    _sp = " [SPIKING slot]" if a.spiking else ""
    print(f"[D3 EVENT CAPSTONE]{_sp} the LIVE MultiTurnAgent answers 'what does HE eat?' over a running event whose delta was NEVER given a state label", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.spiking); rows.append(r)
        print(f"  [seed {s}] CAPSTONE-QA={r['CAPSTONE_QA']} || emission-severed register={r['emission_severed_QA']} | FLAT-FACT={r['FLAT_FACT']} | RECENCY={r['RECENCY']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        ev, sev, flat, rec = _m("CAPSTONE_QA"), _m("emission_severed_QA"), _m("FLAT_FACT"), _m("RECENCY")
        go = (ev > 0.75) and (ev - sev > 0.3) and (ev - flat > 0.3) and (ev - rec > 0.3)
        print(f"\n  AGGREGATE: CAPSTONE-QA={ev:.3f} | emission-severed register={sev:.3f} | FLAT-FACT={flat:.3f} | RECENCY={rec:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the DEPLOYED MultiTurnAgent answers what-does-HE-eat ('+format(ev,'.2f')+') over a running event whose transition delta was learned from agent-emission prediction ALONE (NO (agent,patient) state label anywhere) and whose slot is named by a read-out fitted only on INTRODUCE clauses (observable), maintained'+(' ON SPIKES (FS-WTA one-of-K attractor)' if a.spiking else ' host-side')+'; an EMISSION-SEVERED register COLLAPSES through the identical deployment ('+format(sev,'.2f')+') so the answer rides a LEARNED delta, and FLAT-FACT ('+format(flat,'.2f')+') + RECENCY ('+format(rec,'.2f')+') both FAIL -> EMERGENT + SPIKING + DEPLOYED + QA, end-to-end: the brain answers a question about the running discourse from a composed meaning it was never taught to represent' if go else 'the capstone did not clearly beat its controls (read CAPSTONE vs emission-severed/flat/recency)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
