"""D3 SELF-SUPERVISED EVENT PAIR -> the LIVE MultiTurnAgent: the deployed brain answers "who was doing it BEFORE?" from
a pair of events whose transition delta was **NEVER GIVEN A STATE LABEL**.

The labelled pair register answers BEFORE at 0.93 (`_d3_event_pair_agent_derisk`). Removing the labels is the master
directive. Two rungs supply the pieces:
  * forward prediction does NOT teach the held slot -- **REPLAY (retrodiction) does** (held-slot decode 0.597 at gamma=3,
    79% of the one-emission ceiling; `2026-07-10-D3-event-pair-selfsup-NEGATIVE-then-replay-mechanism.md`)
  * the emergent slot is a PERMUTATION of entity identity, so a deployed register must NAME it

THE LABEL-FREE NAMING OF **BOTH** SLOTS (and a real job for the refuted hypothesis).
  * `a_curr` is named from INTRODUCE clauses: the subject is SPOKEN, so (slot-state-after-an-introduce, named subject)
    is an observable pair. Zero labels. (Same trick as the single-event capstone.)
  * `a_prev` cannot be named that way -- nothing ever speaks it. But a **RETURN clause (the discourse pop)** copies
    `a_curr <- a_prev`, so the ALREADY-CALIBRATED current read-out *reads aloud* whatever the prior slot was holding.
    Fitting `perm_prev` from (a_prev state BEFORE a return, name decoded from a_curr AFTER it) is therefore also
    label-free.

    ⇒ The discourse pop is NOT what teaches the held slot (that hypothesis was refuted by its own control: replay
      without pops scores as well or better). But it IS what lets the brain **name what it is holding**. A discourse
      habit of returning to the prior protagonist is a read-out calibrator, not a teacher.

ANTI-CHEATS (6-seed): (a) the LIVE agent's who_agent_before() >> a SINGLE-EVENT register (0.0 -- structurally cannot);
(b) >> RECENCY and >> naive "answer the current agent"; (c) a REPLAY-ABLATED register (delta trained by prediction
alone) COLLAPSES -- the deployed BEFORE answer rides the replay mechanism; (d) the CURRENT answer stays usable.
Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_selfsup_pair_agent_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_pair_derisk import (
    make_pair_task, train_pair_selfsup, INTRO, RETURN, BOUND)
from research.runners._d3_event_agent_derisk import D3EventRegister
from research.runners.multi_turn_agent import MultiTurnAgent

COREF_W = ("he", "she", "they"); PROMOTE_W = ("it",); CONNECTIVES = ("then", "but", "meanwhile")


def _sm(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def _fit_perm(X, Y, K, epochs=300, lr=0.5):
    """Fit a linear slot->name read-out and return perm[slot] = entity index (the pair runner's `linear_probe` returns
    accuracy, not predictions, so the permutation is read out here)."""
    if len(X) == 0:
        return np.arange(K)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64)
    rng = np.random.RandomState(0)
    W = (rng.randn(K, K) * 0.1).astype(np.float32); b = np.zeros(K, np.float32)
    eye = np.eye(K, dtype=np.float32); n = len(Y)
    for _ in range(epochs):
        s = _sm(X @ W.T + b); d = (s - eye[Y]) / n
        W -= lr * (d.T @ X); b -= lr * d.sum(0)
    return (eye @ W.T + b).argmax(1)          # the name each pure slot decodes to


def fit_label_free_names(task, W, K, n_seq=1200):
    """Fit BOTH slot->name read-outs with zero hidden labels.
      perm_curr : from INTRODUCE clauses (the subject is spoken).
      perm_prev : from RETURN clauses (a_curr <- a_prev, so the calibrated current read-out names the held slot)."""
    emb, Wr, Wi, Wc, bc, Wp, bp = (W["emb"], W["Wr"], W["Wi"], W["Wc"], W["bc"], W["Wp"], W["bp"])
    X, OBJ, EMIT, L, AC, AP, PE, PC = task["train"]
    ident = task["ident"]
    OPS = task["ops_train"]; SID = task["sid_train"]
    Xc, Yc = [], []
    prev_states, ret_at = [], []
    for n in range(min(n_seq, len(L))):
        sc = np.zeros(K, np.float32); sc[ident] = 1.0
        sp = np.zeros(K, np.float32); sp[ident] = 1.0
        pat = np.zeros(K, np.float32); pat[ident] = 1.0
        for t in range(int(L[n])):
            h = np.tanh(np.concatenate([sc @ emb, sp @ emb, pat @ emb]) @ Wr.T + X[n, t] @ Wi.T)
            nc = _sm(h @ Wc.T + bc); npv = _sm(h @ Wp.T + bp)
            if OPS[n, t] == INTRO and SID[n, t] >= 0:              # the utterance NAMED the agent
                oh = np.zeros(K, np.float32); oh[int(np.argmax(nc))] = 1.0
                Xc.append(oh); Yc.append(int(SID[n, t]))
            if OPS[n, t] == RETURN:                                # the pop reads the held slot ALOUD
                prev_states.append(sp.copy())                      # a_prev BEFORE the return
                ret_at.append(int(np.argmax(nc)))                  # a_curr slot AFTER it (== the held agent)
            sc, sp = nc, npv
            pat = np.zeros(K, np.float32); pat[int(OBJ[n, t])] = 1.0
    perm_curr = _fit_perm(Xc, Yc, K)                               # slot -> name, from SPOKEN introduce subjects
    if prev_states:                                                # name the prior slot via the calibrated current one
        Xp = np.zeros((len(prev_states), K), np.float32)
        for i, s_ in enumerate(prev_states):
            Xp[i, int(np.argmax(s_))] = 1.0
        Yp = np.asarray([perm_curr[c] for c in ret_at], np.int64)  # the pop READS ALOUD what the prior slot holds
        perm_prev = _fit_perm(Xp, Yp, K)
    else:
        perm_prev = np.arange(K)
    return perm_curr, perm_prev


class SelfSupPairRegister:
    """A pair register whose transition delta was learned from an agent-emission cross-entropy + REPLAY, with NO
    (agent,patient) state label. Both slot->name read-outs are fitted label-free."""

    def __init__(self, referents, seed=42, n_hid=128, epochs=40, gamma=3.0, replay=True):
        self.referents = list(referents); self.ref2idx = {r: i for i, r in enumerate(referents)}
        K = len(referents); self.K = K
        task = make_pair_task(seed, K=K)
        roll = train_pair_selfsup(task, seed=seed, n_hid=n_hid, epochs=epochs, replay=replay, gamma=gamma)
        self.W = roll.W
        self.ent, self.marks = task["ent"], task["marks"]
        self.ident = task["ident"]
        self.perm_curr, self.perm_prev = fit_label_free_names(task, self.W, K)
        self.sc = np.zeros(K, np.float32); self.sc[self.ident] = 1.0
        self.sp = np.zeros(K, np.float32); self.sp[self.ident] = 1.0
        self.pat = np.zeros(K, np.float32); self.pat[self.ident] = 1.0
        self._boundary = False

    def reset(self):
        self.sc = np.zeros(self.K, np.float32); self.sc[self.ident] = 1.0
        self.sp = np.zeros(self.K, np.float32); self.sp[self.ident] = 1.0
        self.pat = np.zeros(self.K, np.float32); self.pat[self.ident] = 1.0
        self._boundary = False

    def mark_boundary(self):
        self._boundary = True

    def is_pronoun_subject(self, word):
        w = (word or "").lower()
        return w in COREF_W or w in PROMOTE_W

    def observe(self, subject_word, object_word):
        o = self.ref2idx.get(object_word)
        if o is None:
            return
        sw = (subject_word or "").lower()
        if sw in COREF_W:
            sub = self.marks["HE"]
        elif sw in PROMOTE_W:
            sub = self.marks["IT"]
        else:
            s = self.ref2idx.get(sw)
            if s is None:
                return
            sub = self.ent[s]
        # A connective + a NAMED subject opens a NEW event (BND). A connective + a PRONOUN subject is a DISCOURSE POP
        # (RET: a_curr <- a_prev, "meanwhile HE chased the ball"). Deploying with only BND was a distribution shift --
        # the delta was trained with RET clauses at 20%, and RET is also what calibrates the prior-slot read-out.
        if self._boundary:
            mk = self.marks["RET"] if (sw in COREF_W or sw in PROMOTE_W) else self.marks["BND"]
        else:
            mk = self.marks["NOB"]
        self._boundary = False
        code = np.concatenate([mk, sub, self.ent[o]]).astype(np.float32)
        emb, Wr, Wi, Wc, bc, Wp, bp = (self.W["emb"], self.W["Wr"], self.W["Wi"],
                                       self.W["Wc"], self.W["bc"], self.W["Wp"], self.W["bp"])
        h = np.tanh(np.concatenate([self.sc @ emb, self.sp @ emb, self.pat @ emb]) @ Wr.T + code @ Wi.T)
        self.sc = _sm(h @ Wc.T + bc); self.sp = _sm(h @ Wp.T + bp)
        self.pat = np.zeros(self.K, np.float32); self.pat[o] = 1.0

    def who_agent(self):
        return self.referents[int(self.perm_curr[int(np.argmax(self.sc))])]

    def who_patient(self):
        return self.referents[int(np.argmax(self.pat))]

    def who_agent_prev(self):
        return self.referents[int(self.perm_prev[int(np.argmax(self.sp))])]


def make_discourse(rng, referents, n_clause=7, p_boundary=0.25, p_return=0.15, p_coref=0.5, p_promote=0.2):
    """Matches the TRAINING discourse statistics: a connective + a named subject opens a new event (boundary); a
    connective + a pronoun is a discourse POP (return to the prior protagonist)."""
    idx = {r: i for i, r in enumerate(referents)}
    ac = pc = ap = 0
    has_prev = False
    out = []
    for t in range(n_clause):
        o = referents[rng.randint(len(referents))]
        boundary = (t > 0) and (rng.rand() < p_boundary)
        pop = (not boundary) and has_prev and (t > 0) and (rng.rand() < p_return)
        if pop:                                                   # "meanwhile he chase X" -> a_curr <- a_prev
            ac = ap
            out.append(f"{CONNECTIVES[rng.randint(len(CONNECTIVES))]} he chase {o}")
            pc = idx[o]
            continue
        if boundary:
            ap = ac; has_prev = True
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


def run_seed(seed, n_disc=30):
    referents = ["dog", "cat", "fish", "bird", "worm", "ball"]
    vocab = {w: None for w in (referents + ["chase"])}
    rng = np.random.RandomState(seed + 11)

    reg = SelfSupPairRegister(referents, seed=seed, replay=True, gamma=3.0)
    agent = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                           event_register=reg, enable_neural_render=False)
    abl = SelfSupPairRegister(referents, seed=seed, replay=False)          # REPLAY-ABLATED (prediction alone)
    agent_abl = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                               event_register=abl, enable_neural_render=False)
    single = D3EventRegister(referents, seed=seed, spiking=False)          # structurally incapable
    agent_single = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                                  event_register=single, enable_neural_render=False)

    ok = ok_abl = ok_single = ok_rec = ok_naive = ok_now = tot = 0
    tried = 0
    while tot < n_disc and tried < n_disc * 20:
        tried += 1
        clauses, true_now, true_before = make_discourse(rng, referents)
        if true_before == true_now or true_before == 0:
            continue                                              # INFORMATIVE only: a real prior event, != the current
        reg.reset(); abl.reset(); single.reset()
        for c in clauses:
            agent.hear(c); agent_abl.hear(c); agent_single.hear(c)
        tb = referents[true_before]; tn = referents[true_now]
        ok += int(agent.who_agent_before() == tb)
        ok_abl += int(agent_abl.who_agent_before() == tb)
        ok_single += int(agent_single.who_agent_before() == tb)
        ok_now += int(agent.who_agent_now() == tn)
        ok_naive += int(agent.who_agent_now() == tb)
        ok_rec += int(clauses[-1].split()[-1] == tb)
        tot += 1
    m = max(tot, 1)
    return {"seed": seed, "BEFORE_selfsup_replay": round(ok / m, 3), "BEFORE_replay_ablated": round(ok_abl / m, 3),
            "BEFORE_single_event": round(ok_single / m, 3), "BEFORE_recency": round(ok_rec / m, 3),
            "BEFORE_naive_current": round(ok_naive / m, 3), "NOW_selfsup": round(ok_now / m, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print("[D3 SELF-SUP EVENT PAIR -> LIVE MultiTurnAgent] the deployed brain answers 'who was doing it BEFORE?' from a pair whose delta was NEVER given a state label", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s); rows.append(r)
        print(f"  [seed {s}] BEFORE (self-sup + replay)={r['BEFORE_selfsup_replay']} || replay-ABLATED={r['BEFORE_replay_ablated']} | "
              f"single-event={r['BEFORE_single_event']} | recency={r['BEFORE_recency']} | naive-current={r['BEFORE_naive_current']} || NOW={r['NOW_selfsup']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        bp, ba, bs_, br, bn, nw = (_m("BEFORE_selfsup_replay"), _m("BEFORE_replay_ablated"), _m("BEFORE_single_event"),
                                   _m("BEFORE_recency"), _m("BEFORE_naive_current"), _m("NOW_selfsup"))
        go = (bp > 0.45) and (bp - ba > 0.2) and (bp - br > 0.25) and (bp - bn > 0.25)
        print(f"\n  AGGREGATE: BEFORE (self-sup + replay)={bp:.3f} | replay-ABLATED={ba:.3f} | single-event={bs_:.3f} | recency={br:.3f} | naive-current={bn:.3f} || NOW={nw:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the DEPLOYED agent answers who-was-doing-it-BEFORE ('+format(bp,'.2f')+') from a pair of composed events whose transition delta was NEVER given a state label, with BOTH slot names learned label-free (a_curr from spoken INTRODUCE subjects; a_prev from RETURN clauses, where the discourse pop makes the calibrated current read-out say aloud what the prior slot holds). A REPLAY-ABLATED register (prediction alone) collapses to '+format(ba,'.2f')+' -- the deployed answer rides the replay mechanism -- while a SINGLE-EVENT register cannot answer at all ('+format(bs_,'.2f')+'), and recency ('+format(br,'.2f')+') + naive-current ('+format(bn,'.2f')+') both fail. The discourse pop is NOT the teacher (refuted) but IS the read-out calibrator' if go else 'the deployed self-supervised pair did not clearly answer BEFORE (read vs replay-ablated / recency / naive)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
