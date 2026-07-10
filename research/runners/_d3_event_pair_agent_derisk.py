"""D3 EVENT PAIR -> the LIVE MultiTurnAgent: the deployed brain answers "who was doing it BEFORE?" -- a question a
single-event register STRUCTURALLY CANNOT answer.

The connectives rungs showed a discourse connective marks an EVENT BOUNDARY that SHIFTS the running event into a
previous slot (rate 6-seed GO; four spiking FS-WTA slots 6-seed GO). THIS deploys it: `PairEventRegister` is a drop-in
for `D3EventRegister` that holds (a_curr, p_curr | a_prev, p_prev) on the spiking substrate, and `MultiTurnAgent` gains
`who_agent_before()`.

    "dog chase cat."  "he chase fish."          -> current event: agent=dog
    "THEN bird chase worm."                     -> the connective SHIFTS dog's event into the prior slot
    "he chase ball."                            -> current agent=bird (coref), prior agent=dog (HELD)
    ASK "who is doing it now?"    -> bird
    ASK "who was doing it BEFORE?" -> dog        <- a single-event register overwrote dog and cannot answer at all

The agent's `hear` strips a leading discourse connective ("then"/"but"/"meanwhile") and calls `mark_boundary()` on any
register that supports it -- single-event registers simply lack the method and are unaffected (backward-compatible).

ANTI-CHEATS (6-seed): (a) the LIVE agent's who_agent_before() >> a SINGLE-EVENT register (`D3EventRegister`), which
returns None -- structurally incapable, not merely weaker; (b) >> RECENCY (the most-recently-mentioned entity);
(c) >> "answer the CURRENT agent" (the naive conflation of now and before); (d) the CURRENT-event answer stays correct
(holding a prior event must not cost the present); (e) `--spiking` maintains all four slots on FS-WTA attractors.
Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_pair_agent_derisk --seeds 42 --spiking
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_connective_derisk import make_connective_task, multislot_rnn
from research.runners._d3_event_agent_derisk import D3EventRegister
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive
from research.runners.multi_turn_agent import MultiTurnAgent

COREF_W = ("he", "she", "they"); PROMOTE_W = ("it",); CONNECTIVES = ("then", "but", "meanwhile")


class PairEventRegister:
    """Holds a PAIR of composed events (a_curr, p_curr | a_prev, p_prev). A discourse connective marks an EVENT BOUNDARY
    that SHIFTS the current event into the previous slot instead of overwriting it. Drop-in for `D3EventRegister`,
    plus `mark_boundary()` and `who_agent_prev()`."""

    def __init__(self, referents, seed=42, spiking=True, n_hid=192, epochs=50, settle=25, fs_inh=9.0):
        self.referents = list(referents); self.ref2idx = {r: i for i, r in enumerate(referents)}
        K = len(referents); self.K = K
        task = make_connective_task(seed, K=K)
        self.W = multislot_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, n_slots=4)["weights"]
        self.ent, self.HE, self.IT = task["ent"], task["HE"], task["IT"]
        self.BND, self.NOB = task["BND"], task["NOB"]
        self.ident = task["ident"]; self.settle = settle
        self.bridges = [build_fswta_score_bridge(seed=seed + 3 * k, K=K, fs_to_exc=fs_inh) for k in range(4)] if spiking else None
        self.slots = [self.ident] * 4
        self._boundary = False

    def reset(self):
        self.slots = [self.ident] * 4; self._boundary = False

    def mark_boundary(self):
        """A discourse connective was heard: the NEXT clause opens a new event and shifts the current one back."""
        self._boundary = True

    def is_pronoun_subject(self, word):
        w = (word or "").lower()
        return w in COREF_W or w in PROMOTE_W

    def observe(self, subject_word, object_word):
        o = self.ref2idx.get(object_word)
        if o is None:
            return                                                # unknown patient -> skip (moat: no confabulation)
        sw = (subject_word or "").lower()
        if sw in COREF_W:
            sub = self.HE
        elif sw in PROMOTE_W:
            sub = self.IT
        else:
            s = self.ref2idx.get(sw)
            if s is None:
                return
            sub = self.ent[s]
        mk = self.BND if self._boundary else self.NOB
        self._boundary = False
        code = np.concatenate([mk, sub, self.ent[o]]).astype(np.float32)
        emb, Wr, Wi, Ws, bs = self.W["emb"], self.W["Wr"], self.W["Wi"], self.W["Ws"], self.W["bs"]
        h = np.tanh(np.concatenate([emb[s_] for s_ in self.slots]) @ Wr.T + code @ Wi.T)
        outs = []
        for k in range(4):
            sc = h @ Ws[k].T + bs[k]
            if self.bridges is None:
                outs.append(int(np.argmax(sc)))
            else:                                                 # NORMALIZE the drive (f-I saturation fix), then spike
                d = np.maximum(sc, 0.0); mx = d.max()
                d = d / (mx + 1e-9) if mx > 0 else d
                _, acc = fswta_drive(self.bridges[k], self.K, d, settle=self.settle)
                outs.append(int(np.argmax(acc)) if acc.max() > 0 else self.ident)
        self.slots = outs

    def who_agent(self):
        return self.referents[self.slots[0]]

    def who_patient(self):
        return self.referents[self.slots[1]]

    def who_agent_prev(self):
        return self.referents[self.slots[2]]


def make_discourse(rng, referents, n_clause=6, p_boundary=0.28, p_coref=0.5, p_promote=0.2):
    """A discourse with connective-marked event boundaries. Returns clauses (with the connective attached) + the true
    (a_curr, a_prev) at the end."""
    idx = {r: i for i, r in enumerate(referents)}
    ac = pc = ap = 0
    out = []
    for t in range(n_clause):
        o = referents[rng.randint(len(referents))]
        boundary = (t > 0) and (rng.rand() < p_boundary)
        if boundary:
            ap = ac                                               # SHIFT (p_prev tracked implicitly)
            s = referents[rng.randint(len(referents))]; ac = idx[s]
            out.append(f"{CONNECTIVES[rng.randint(len(CONNECTIVES))]} {s} chase {o}")
        else:
            r = rng.rand()
            if t == 0 or r >= p_coref + p_promote:
                s = referents[rng.randint(len(referents))]; ac = idx[s]
            elif r < p_coref:
                s = "he"
            else:
                s = "it"; ac = pc
            out.append(f"{s} chase {o}")
        pc = idx[o]
    return out, ac, ap


def run_seed(seed, spiking, n_disc=30):
    referents = ["dog", "cat", "fish", "bird", "worm", "ball"]
    vocab = {w: None for w in (referents + ["chase"])}
    rng = np.random.RandomState(seed + 11)

    pair_reg = PairEventRegister(referents, seed=seed, spiking=spiking)
    pair_agent = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                                event_register=pair_reg, enable_neural_render=False)
    single_reg = D3EventRegister(referents, seed=seed, spiking=spiking)      # STRUCTURALLY INCAPABLE control
    single_agent = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                                  event_register=single_reg, enable_neural_render=False)

    before_ok = single_ok = rec_ok = curr_as_before_ok = now_ok = tot = 0
    for _ in range(n_disc):
        pair_reg.reset(); single_reg.reset()
        clauses, true_now, true_before = make_discourse(rng, referents)
        for c in clauses:
            pair_agent.hear(c); single_agent.hear(c)
        tb = referents[true_before]; tn = referents[true_now]
        before_ok += int(pair_agent.who_agent_before() == tb)
        single_ok += int(single_agent.who_agent_before() == tb)       # None -> structurally cannot answer
        now_ok += int(pair_agent.who_agent_now() == tn)
        curr_as_before_ok += int(pair_agent.who_agent_now() == tb)    # naive: conflate now with before
        last_obj = clauses[-1].split()[-1]
        rec_ok += int(last_obj == tb)
        tot += 1
    m = max(tot, 1)
    return {"seed": seed, "spiking": spiking, "BEFORE_pair": round(before_ok / m, 3),
            "BEFORE_single_event": round(single_ok / m, 3), "BEFORE_recency": round(rec_ok / m, 3),
            "BEFORE_naive_current": round(curr_as_before_ok / m, 3), "NOW_pair": round(now_ok / m, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--spiking", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    _sp = " [SPIKING pair]" if a.spiking else ""
    print(f"[D3 EVENT PAIR -> LIVE MultiTurnAgent]{_sp} the deployed brain answers 'who was doing it BEFORE?' across a discourse connective", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.spiking); rows.append(r)
        print(f"  [seed {s}] BEFORE (pair register)={r['BEFORE_pair']} || single-event register={r['BEFORE_single_event']} | recency={r['BEFORE_recency']} | "
              f"naive-'current'={r['BEFORE_naive_current']} || NOW (current event) still correct={r['NOW_pair']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        bp, bs_, br, bn, nw = _m("BEFORE_pair"), _m("BEFORE_single_event"), _m("BEFORE_recency"), _m("BEFORE_naive_current"), _m("NOW_pair")
        go = (bp > 0.7) and (bp - bs_ > 0.5) and (bp - br > 0.4) and (bp - bn > 0.4) and (nw > 0.7)
        print(f"\n  AGGREGATE: BEFORE (pair)={bp:.3f} | single-event register={bs_:.3f} | recency={br:.3f} | naive-'current'={bn:.3f} || NOW={nw:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the DEPLOYED MultiTurnAgent answers who-was-doing-it-BEFORE ('+format(bp,'.2f')+') across a discourse connective, holding the prior event while continuing to track the present ('+format(nw,'.2f')+'), where a SINGLE-EVENT register CANNOT ANSWER AT ALL ('+format(bs_,'.2f')+' -- it overwrote the prior event), RECENCY fails ('+format(br,'.2f')+') and naively answering the CURRENT agent fails ('+format(bn,'.2f')+')'+(' -- all four event slots maintained ON SPIKES' if a.spiking else '')+' -> the conversational payoff of the connectives arc: the brain relates two composed meanings and can be ASKED about either' if go else 'the deployed pair register did not clearly answer BEFORE (read BEFORE vs single-event/recency/naive)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
