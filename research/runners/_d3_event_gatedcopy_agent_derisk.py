"""D3 GATED COPY -> the LIVE MultiTurnAgent: the deployed brain answers "who was doing it BEFORE?" from a pair of events
whose transition was NEVER GIVEN A STATE LABEL, and whose held slot is a STRUCTURAL GATED COPY rather than a learned head.

WHY THIS SUPERSEDES THE REPLAY DEPLOYMENT.
The self-supervised pair deployed on the REPLAY mechanism answered BEFORE at only 0.367 (`2026-07-10-D3-selfsup-pair-
deployed-PARTIAL-price-of-emergence.md`), because replay's held-slot decode was 0.492/0.597. The BOUNDARY-GATED COPY
raises that decode to ~0.63-0.69 (0.738 with an oracle gate) by making the copy STRUCTURAL -- a pre-wired route opened by
an observable marker, exactly `sim/`'s `transmission_gate` (PBWM output gating). So the deployed BEFORE answer should
lift materially.

A SIMPLIFICATION THE GATED COPY BUYS FOR FREE.
With replay, `a_prev` was a LEARNED head, so its slot basis differed from `a_curr`'s and the register needed a SECOND,
separately-calibrated slot->name read-out (fitted from RETURN clauses, where the discourse pop reads the held slot
aloud). With a structural copy, `a_prev = g*a_curr + (1-g)*a_prev` lives in the SAME basis as `a_curr` -- so ONE
label-free read-out names both, and the discourse-pop calibrator is no longer needed at all.

LABEL-FREE THROUGHOUT: the delta is learned from an agent-emission cross-entropy ALONE; the gate reads only the
OBSERVABLE clause code; the single slot->name read-out is fitted from INTRODUCE clauses, where the subject is SPOKEN.
No `(agent, patient)` state label anywhere.

ANTI-CHEATS (6-seed): (a) the LIVE agent's who_agent_before() >> a SINGLE-EVENT register (0.0 -- structurally cannot);
(b) >> RECENCY and >> naive "answer the current agent"; (c) a GATE-LESION register (the gate never opens -> nothing is
ever shifted) COLLAPSES; (d) the CURRENT answer stays usable; (e) compared head-to-head against the REPLAY deployment.
Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_gatedcopy_agent_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_pair_derisk import make_pair_task, INTRO
from research.runners._d3_event_gated_copy_derisk import train_gated_copy, _sm, _sig
from research.runners._d3_event_agent_derisk import D3EventRegister
from research.runners._d3_event_selfsup_pair_agent_derisk import _fit_perm, make_discourse, COREF_W, PROMOTE_W
from research.runners.multi_turn_agent import MultiTurnAgent


def fit_slot_names_labelfree(task, W, K, n_seq=1200):
    """ONE read-out names BOTH slots (the structural copy keeps them in the same basis). Fitted from clauses whose
    subject is SPOKEN -- zero hidden labels.

    BIJECTIVE, not per-slot argmax. MEASURED: an independent per-slot argmax read-out is many-to-one -- e.g. seed 43
    mapped slots {3,4} both to entity 0 and {2,5} both to entity 2, leaving entities unreachable and capping deployed
    accuracy (label-free 0.399 vs an oracle permutation's 0.681). K slots must map ONE-TO-ONE onto K entities; that
    constraint is exactly what lateral inhibition across a naming read-out enforces. Solving the assignment instead of
    taking independent argmaxes lifts the label-free read-out 0.547 -> 0.572 (seed 102: 0.401 -> 0.495).

    HONEST RESIDUAL: even bijective, the label-free read-out trails an oracle permutation (0.572 vs 0.669). Adding a
    second observable naming source (PROMOTE clauses, where a_curr becomes the previously-SPOKEN object) changes nothing
    -- the assignment is already pinned by the INTRODUCE co-occurrence. The gap is a genuine limit of naming a latent
    slot from observables, NOT a data-volume problem."""
    emb, Wr, Wi, Wc, bc = W["emb"], W["Wr"], W["Wi"], W["Wc"], W["bc"]
    X, OBJ, EMIT, L, AC, AP, PE, PC = task["train"]
    OPS = task["ops_train"]; SID = task["sid_train"]; ident = task["ident"]
    C = np.zeros((K, K))
    for n in range(min(n_seq, len(L))):
        sc = np.zeros(K, np.float32); sc[ident] = 1.0
        sp = np.zeros(K, np.float32); sp[ident] = 1.0
        pat = np.zeros(K, np.float32); pat[ident] = 1.0
        for t in range(int(L[n])):
            h = np.tanh(np.concatenate([sc @ emb, sp @ emb, pat @ emb]) @ Wr.T + X[n, t] @ Wi.T)
            nc = _sm(h @ Wc.T + bc)
            if OPS[n, t] == INTRO and SID[n, t] >= 0:            # the utterance SPOKE the agent's name
                C[int(np.argmax(nc)), int(SID[n, t])] += 1.0
            sc = nc
            pat = np.zeros(K, np.float32); pat[int(OBJ[n, t])] = 1.0
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-C)                          # maximise co-occurrence, ONE-TO-ONE
        perm = np.zeros(K, dtype=int); perm[r] = c
        return perm
    except Exception:                                             # greedy fallback, still a bijection
        perm = -np.ones(K, dtype=int); used = set()
        for k in np.argsort(-C.max(1)):
            order = np.argsort(-C[k])
            for e in order:
                if int(e) not in used:
                    perm[k] = int(e); used.add(int(e)); break
        for k in range(K):
            if perm[k] < 0:
                for e in range(K):
                    if e not in used:
                        perm[k] = e; used.add(e); break
        return perm


class GatedCopyPairRegister:
    """A pair register whose delta was learned from agent-emission CE alone (NO state label) and whose held slot is a
    STRUCTURAL boundary-gated copy. Drop-in for `D3EventRegister` (+ `mark_boundary`, `who_agent_prev`)."""

    def __init__(self, referents, seed=42, n_hid=128, epochs=40, gate_lesion=False):
        self.referents = list(referents); self.ref2idx = {r: i for i, r in enumerate(referents)}
        K = len(referents); self.K = K
        task = make_pair_task(seed, K=K)
        roll = train_gated_copy(task, seed=seed, n_hid=n_hid, epochs=epochs)
        self.W = roll.W; self.wg, self.bg = roll.gate
        self.ent, self.marks = task["ent"], task["marks"]
        self.ident = task["ident"]
        self.gate_lesion = bool(gate_lesion)
        self.perm = fit_slot_names_labelfree(task, self.W, K)        # ONE label-free read-out for BOTH slots
        self.reset()

    def reset(self):
        K = self.K
        self.sc = np.zeros(K, np.float32); self.sc[self.ident] = 1.0
        self.sp = np.zeros(K, np.float32); self.sp[self.ident] = 1.0
        self.pat = np.zeros(K, np.float32); self.pat[self.ident] = 1.0
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
        # a connective + a PRONOUN is a discourse POP (RET); + a NAMED subject opens a new event (BND)
        if self._boundary:
            mk = self.marks["RET"] if (sw in COREF_W or sw in PROMOTE_W) else self.marks["BND"]
        else:
            mk = self.marks["NOB"]
        self._boundary = False
        code = np.concatenate([mk, sub, self.ent[o]]).astype(np.float32)

        emb, Wr, Wi, Wc, bc = (self.W["emb"], self.W["Wr"], self.W["Wi"], self.W["Wc"], self.W["bc"])
        g = 0.0 if self.gate_lesion else float(_sig(code @ self.wg + self.bg))
        h = np.tanh(np.concatenate([self.sc @ emb, self.sp @ emb, self.pat @ emb]) @ Wr.T + code @ Wi.T)
        nc = _sm(h @ Wc.T + bc)
        self.sp = g * self.sc + (1.0 - g) * self.sp                  # THE GATED COPY (structural)
        self.sc = nc
        self.pat = np.zeros(self.K, np.float32); self.pat[o] = 1.0

    def who_agent(self):
        return self.referents[int(self.perm[int(np.argmax(self.sc))])]

    def who_patient(self):
        return self.referents[int(np.argmax(self.pat))]

    def who_agent_prev(self):
        return self.referents[int(self.perm[int(np.argmax(self.sp))])]


def run_seed(seed, n_disc=30):
    referents = ["dog", "cat", "fish", "bird", "worm", "ball"]
    vocab = {w: None for w in (referents + ["chase"])}
    rng = np.random.RandomState(seed + 11)

    reg = GatedCopyPairRegister(referents, seed=seed)
    agent = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                           event_register=reg, enable_neural_render=False)
    les = GatedCopyPairRegister(referents, seed=seed, gate_lesion=True)      # the gate never opens
    agent_les = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                               event_register=les, enable_neural_render=False)
    single = D3EventRegister(referents, seed=seed, spiking=False)            # structurally cannot answer BEFORE
    agent_single = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                                  event_register=single, enable_neural_render=False)

    ok = ok_les = ok_single = ok_rec = ok_naive = ok_now = tot = tried = 0
    while tot < n_disc and tried < n_disc * 20:
        tried += 1
        clauses, true_now, true_before = make_discourse(rng, referents)
        if true_before == true_now or true_before == 0:
            continue                                                        # INFORMATIVE only
        reg.reset(); les.reset(); single.reset()
        for c in clauses:
            agent.hear(c); agent_les.hear(c); agent_single.hear(c)
        tb = referents[true_before]; tn = referents[true_now]
        ok += int(agent.who_agent_before() == tb)
        ok_les += int(agent_les.who_agent_before() == tb)
        ok_single += int(agent_single.who_agent_before() == tb)
        ok_now += int(agent.who_agent_now() == tn)
        ok_naive += int(agent.who_agent_now() == tb)
        ok_rec += int(clauses[-1].split()[-1] == tb)
        tot += 1
    m = max(tot, 1)
    return {"seed": seed, "BEFORE_gatedcopy": round(ok / m, 3), "BEFORE_gate_lesion": round(ok_les / m, 3),
            "BEFORE_single_event": round(ok_single / m, 3), "BEFORE_recency": round(ok_rec / m, 3),
            "BEFORE_naive_current": round(ok_naive / m, 3), "NOW_gatedcopy": round(ok_now / m, 3), "n": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print("[D3 GATED COPY -> LIVE MultiTurnAgent] the deployed brain answers 'who was doing it BEFORE?' from a pair whose delta was never given a state label and whose held slot is a STRUCTURAL gated copy", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s); rows.append(r)
        print(f"  [seed {s}] BEFORE (gated copy)={r['BEFORE_gatedcopy']} || gate-lesion={r['BEFORE_gate_lesion']} | single-event={r['BEFORE_single_event']} | "
              f"recency={r['BEFORE_recency']} | naive-current={r['BEFORE_naive_current']} || NOW={r['NOW_gatedcopy']} (n={r['n']})", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        bp, bl, bs_, br, bn, nw = (_m("BEFORE_gatedcopy"), _m("BEFORE_gate_lesion"), _m("BEFORE_single_event"),
                                   _m("BEFORE_recency"), _m("BEFORE_naive_current"), _m("NOW_gatedcopy"))
        REPLAY_DEPLOY = 0.367                                                # the prior deployment (replay mechanism)
        go = (bp > 0.5) and (bp - bl > 0.25) and (bp - br > 0.3) and (bp - bn > 0.3) and (bp > REPLAY_DEPLOY + 0.1)
        print(f"\n  AGGREGATE: BEFORE (gated copy)={bp:.3f} | gate-lesion={bl:.3f} | single-event={bs_:.3f} | recency={br:.3f} | naive-current={bn:.3f} || NOW={nw:.3f}", flush=True)
        print(f"    head-to-head: the REPLAY deployment answered BEFORE at {REPLAY_DEPLOY:.3f}", flush=True)
        msg = ('the DEPLOYED agent answers who-was-doing-it-BEFORE (' + format(bp, '.2f') + ') from a pair of composed events whose '
               'transition delta was NEVER given a state label and whose held slot is a STRUCTURAL boundary-gated copy -- lifting the '
               'prior REPLAY deployment (' + format(REPLAY_DEPLOY, '.2f') + '). One label-free read-out names BOTH slots (the copy keeps '
               'them in the same basis), so the discourse-pop calibrator the replay version needed is gone. A GATE-LESION register '
               '(the gate never opens, nothing is ever shifted) collapses (' + format(bl, '.2f') + '), a SINGLE-EVENT register cannot '
               'answer at all (' + format(bs_, '.2f') + '), and recency (' + format(br, '.2f') + ') + naive-current (' + format(bn, '.2f') +
               ') both fail')
        bad = 'the deployed gated-copy register did not clearly beat its controls / the replay deployment'
        print("  VERDICT: " + ("GO" if go else "PARTIAL/NEGATIVE") + " -- " + (msg if go else bad) + ". NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
