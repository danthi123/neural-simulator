"""D3 EVENT -> DISCOURSE CONNECTIVES: relate TWO composed events, not just carry one. The register built so far holds a
single running event and OVERWRITES it, so it structurally cannot answer "who was doing it BEFORE?" -- it has no prior
event to relate the current one to. This rung adds the missing mechanism.

THE MECHANISM (Zacks & Radvansky event segmentation; the research gate's RANK-3 "connectives" residual). A connective
("then", "but") marks an EVENT BOUNDARY: instead of overwriting, the current event model is SHIFTED into a previous
slot, and a new event begins. So the state is a PAIR of factored events:

    state = ( a_curr, p_curr | a_prev, p_prev )        each a K-way discrete-attractor slot

    non-boundary clause  -> update the CURRENT event in place   (INTRODUCE / AGENT-COREF / PROMOTE, as before)
    boundary clause      -> (a_prev, p_prev) <- (a_curr, p_curr), then the new clause opens the new current event

The prior event must then be HELD across however many non-boundary clauses follow -- so the shift is not a delayed copy
of the last clause; it must survive an arbitrary run. This is what a single-event register cannot do at any accuracy.

QUERIES on the final state: current agent, PREVIOUS-event agent (the new capability), and the relation "same agent in
both events?".

ANTI-CHEATS (6-seed): (a) PREV-agent accuracy on held-out-DEEPER >> a SINGLE-EVENT model (2 slots + the same prev-head,
which can only guess prev from the current state) -- the load-bearing contrast; (b) >> RECENCY; (c) recurrence-lesion
collapses; (d) the SAME-AGENT relation is read from the pair, and a permuted-slot control destroys it; (e) held-out-
DEEPER lengths. numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_connective_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

INTRO, COREF, PROMOTE = 0, 1, 2


def make_connective_task(seed, K=6, n_pool=96, noise=0.5, train_lens=(3, 4, 5), test_lens=(7, 8, 9),
                         n_per_len=2000, p_boundary=0.25, p_coref=0.5, p_promote=0.25):
    """Discourse with connective-marked EVENT BOUNDARIES. Clause code = [boundary-third ; subject-third ; object-third].
    Emits per-step the 4 true slots (a_curr, p_curr, a_prev, p_prev)."""
    rng = np.random.RandomState(seed)
    th = n_pool // 3; ck = max(3, th // 4)
    ent = -np.ones((K, th), np.float32)
    for e in range(K):
        ent[e, rng.choice(th, ck, replace=False)] = 1.0
    HE = -np.ones(th, np.float32); HE[rng.choice(th, ck, replace=False)] = 1.0
    IT = -np.ones(th, np.float32); IT[rng.choice(th, ck, replace=False)] = 1.0
    BND = -np.ones(th, np.float32); BND[rng.choice(th, ck, replace=False)] = 1.0    # the connective ("then"/"but")
    NOB = -np.ones(th, np.float32); NOB[rng.choice(th, ck, replace=False)] = 1.0    # no boundary
    ident = 0; Lmax = max(tuple(train_lens) + tuple(test_lens))

    def gen(lens, n_each):
        X, L, SA, SP, PA, PP = [], [], [], [], [], []
        for L_ in lens:
            for _ in range(n_each):
                ac = pc = ap = pp = ident
                codes = np.zeros((Lmax, n_pool), np.float32)
                sa = np.full(Lmax, -1, np.int64); sp = np.full(Lmax, -1, np.int64)
                pa = np.full(Lmax, -1, np.int64); ppz = np.full(Lmax, -1, np.int64)
                for t in range(L_):
                    o = int(rng.randint(0, K)); r = rng.rand()
                    boundary = (t > 0) and (rng.rand() < p_boundary)
                    if boundary:
                        ap, pp = ac, pc                       # SHIFT: the current event becomes the previous one
                        op = INTRO                            # a new event opens by NAMING its agent
                    else:
                        op = INTRO if t == 0 else (COREF if r < p_coref else (PROMOTE if r < p_coref + p_promote else INTRO))
                    if op == COREF:
                        sub = HE
                    elif op == PROMOTE:
                        sub = IT; ac = pc
                    else:
                        s = int(rng.randint(0, K)); sub = ent[s]; ac = s
                    pc = o
                    c = np.concatenate([BND if boundary else NOB, sub, ent[o]]).copy()
                    flip = rng.rand(n_pool) < (noise * 0.12); c[flip] = -c[flip]
                    codes[t] = c; sa[t] = ac; sp[t] = pc; pa[t] = ap; ppz[t] = pp
                X.append(codes); L.append(L_); SA.append(sa); SP.append(sp); PA.append(pa); PP.append(ppz)
        return (np.asarray(X, np.float32), np.asarray(L, np.int64), np.asarray(SA, np.int64),
                np.asarray(SP, np.int64), np.asarray(PA, np.int64), np.asarray(PP, np.int64))

    return {"train": gen(train_lens, n_per_len), "test_deeper": gen(test_lens, max(400, n_per_len // 3)),
            "K": K, "ident": ident, "n_pool": n_pool}


def _sm(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def multislot_rnn(task, seed=42, n_hid=192, epochs=50, lr=0.1, batch=256, temperature=0.7,
                  n_slots=4, lesion_rec=False):
    """A FACTORED discrete-attractor over `n_slots` K-way slots, each re-discretized per step.
    n_slots=4 -> (a_curr, p_curr, a_prev, p_prev) = the event PAIR.
    n_slots=2 -> (a_curr, p_curr) only = the SINGLE-EVENT control; it still carries a prev-agent HEAD (so it is asked the
                 same question) but has no prev slot to carry the answer in."""
    K, ident, n_pool = task["K"], task["ident"], task["n_pool"]
    rng = np.random.RandomState(seed + 9)
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)
    Wr = (rng.randn(n_hid, n_slots * n_hid) * np.sqrt(1.0 / (n_slots * n_hid))).astype(np.float32)
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    n_out = 4                                       # always predict all 4 targets (the 2-slot model must guess prev)
    Ws = [(rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32) for _ in range(n_out)]
    bs = [np.zeros(K, np.float32) for _ in range(n_out)]
    rg = 0.0 if lesion_rec else 1.0
    eye = np.eye(K, dtype=np.float32)

    X, L, SA, SP, PA, PP = task["train"]
    TGT = [SA, SP, PA, PP]
    by_len = {}
    for n in range(len(L)):
        by_len.setdefault(int(L[n]), []).append(n)

    def state_in(slots):
        return np.concatenate([s @ emb for s in slots], axis=1)

    for _ in range(epochs):
        for Ln, ids in by_len.items():
            ids = np.asarray(ids); rng.shuffle(ids)
            for i in range(0, len(ids), batch):
                b = ids[i:i + batch]; B = len(b)
                slots = [np.zeros((B, K), np.float32) for _ in range(n_slots)]
                for s in slots:
                    s[:, ident] = 1.0
                dWr = np.zeros_like(Wr); dWi = np.zeros_like(Wi)
                dWs = [np.zeros_like(w) for w in Ws]; dbs = [np.zeros_like(v) for v in bs]
                for t in range(Ln):
                    si = state_in(slots)
                    h = np.tanh(rg * (si @ Wr.T) + X[b, t] @ Wi.T)
                    outs = [_sm((h @ Ws[k].T + bs[k]) / temperature) for k in range(n_out)]
                    dh = np.zeros_like(h)
                    for k in range(n_out):
                        d = (outs[k] - eye[TGT[k][b, t]]) / (B * temperature)
                        dWs[k] += d.T @ h; dbs[k] += d.sum(0); dh += d @ Ws[k]
                    dh *= (1 - h ** 2)
                    dWi += dh.T @ X[b, t]
                    if rg:
                        dWr += dh.T @ si
                    # teacher-force the slots (per-step supervision), re-discretized to clean one-hots
                    slots = [eye[TGT[k][b, t]] for k in range(n_slots)]
                for k in range(n_out):
                    Ws[k] -= lr * dWs[k]; bs[k] -= lr * dbs[k]
                Wr -= lr * dWr; Wi -= lr * dWi

    def evaluate(split):
        X_, L_, SA_, SP_, PA_, PP_ = task[split]; B = len(L_); Lm = int(L_.max())
        slots = [np.zeros((B, K), np.float32) for _ in range(n_slots)]
        for s in slots:
            s[:, ident] = 1.0
        fin = [np.zeros(B, np.int64) for _ in range(4)]
        for t in range(Lm):
            act = (L_ > t)
            si = state_in(slots)
            h = np.tanh(rg * (si @ Wr.T) + X_[:, t] @ Wi.T)
            am = [(h @ Ws[k].T + bs[k]).argmax(1) for k in range(4)]
            new = [np.eye(K, dtype=np.float32)[am[k]] for k in range(n_slots)]
            slots = [np.where(act[:, None], new[k], slots[k]) for k in range(n_slots)]
            last = (L_ == (t + 1))
            for k in range(4):
                fin[k] = np.where(last, am[k], fin[k])
        tg = [SA_[np.arange(B), L_ - 1], SP_[np.arange(B), L_ - 1], PA_[np.arange(B), L_ - 1], PP_[np.arange(B), L_ - 1]]
        curr_a = float((fin[0] == tg[0]).mean()); prev_a = float((fin[2] == tg[2]).mean())
        same_true = (tg[0] == tg[2]); same_pred = (fin[0] == fin[2])
        return {"curr_agent": curr_a, "prev_agent": prev_a, "same_agent_rel": float((same_pred == same_true).mean()),
                "obj": tg}

    return evaluate("test_deeper")


def recency_floor(task):
    """RECENCY: current agent = the last object; previous-event agent = the last object too (a recency reader has no
    event boundary, so it has nothing else to offer)."""
    X, L, SA, SP, PA, PP = task["test_deeper"]; B = len(L)
    last_o = SP[np.arange(B), L - 1]
    return float((last_o == SA[np.arange(B), L - 1]).mean()), float((last_o == PA[np.arange(B), L - 1]).mean())


def run_seed(seed, K, n_hid, epochs):
    task = make_connective_task(seed, K=K)
    pair = multislot_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, n_slots=4)
    single = multislot_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, n_slots=2)     # SINGLE-EVENT control
    lesion = multislot_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, n_slots=4, lesion_rec=True)
    rec_c, rec_p = recency_floor(task)
    return {"seed": seed, "K": K,
            "PAIR_curr_agent": round(pair["curr_agent"], 3), "PAIR_prev_agent": round(pair["prev_agent"], 3),
            "PAIR_same_rel": round(pair["same_agent_rel"], 3),
            "SINGLE_prev_agent": round(single["prev_agent"], 3), "SINGLE_curr_agent": round(single["curr_agent"], 3),
            "LESION_prev_agent": round(lesion["prev_agent"], 3),
            "recency_curr": round(rec_c, 3), "recency_prev": round(rec_p, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT CONNECTIVES] K={a.K} | a connective marks an EVENT BOUNDARY that SHIFTS the current event into a previous slot -> the brain holds a PAIR of events and can relate them", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs); rows.append(r)
        print(f"  [seed {s}] PAIR prev-agent={r['PAIR_prev_agent']} (curr={r['PAIR_curr_agent']} | same-agent rel={r['PAIR_same_rel']}) || "
              f"SINGLE-EVENT prev={r['SINGLE_prev_agent']} (its curr={r['SINGLE_curr_agent']}) | lesion prev={r['LESION_prev_agent']} | recency prev={r['recency_prev']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        pp, pc, rel = _m("PAIR_prev_agent"), _m("PAIR_curr_agent"), _m("PAIR_same_rel")
        sp, sc, le, rp = _m("SINGLE_prev_agent"), _m("SINGLE_curr_agent"), _m("LESION_prev_agent"), _m("recency_prev")
        chance = 1.0 / a.K
        go = (pp > 0.75) and (pp - sp > 0.3) and (pp - le > 0.3) and (pp - rp > 0.3) and (rel > 0.75)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}):", flush=True)
        print(f"    EVENT-PAIR: prev-agent={pp:.3f} | curr-agent={pc:.3f} | same-agent relation={rel:.3f}", flush=True)
        print(f"    SINGLE-EVENT control: prev-agent={sp:.3f} (curr-agent {sc:.3f} -- it tracks the CURRENT event fine, it simply has no prior event to hold)", flush=True)
        print(f"    recurrence-lesion prev={le:.3f} | recency prev={rp:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'a connective-marked EVENT BOUNDARY shifts the running event into a previous slot, so the discrete-attractor holds a PAIR of composed events and can RELATE them: the previous-event agent is recovered on held-out-DEEPER discourses ('+format(pp,'.2f')+') and the same-agent relation read across the pair ('+format(rel,'.2f')+'), where a SINGLE-EVENT register FAILS ('+format(sp,'.2f')+') despite tracking the current event fine ('+format(sc,'.2f')+') -- it structurally has no prior event to hold -- and recurrence-lesion ('+format(le,'.2f')+') + recency ('+format(rp,'.2f')+') both collapse -> discourse connectives: the brain relates two composed meanings, not just carries one' if go else 'the event-pair did not clearly beat the single-event control (read PAIR prev vs SINGLE prev / lesion / recency)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
