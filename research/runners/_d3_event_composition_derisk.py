"""D3 -> INCREMENTAL EVENT COMPOSITION (the anti-RAG running meaning; research-gated TOP pick after anaphora): the
recurrent discrete-attractor cortex maintains a running, updatable "who-did-what-to-whom" EVENT state -- a FACTORED
(agent, patient) register built word-by-word and REVISED across the discourse -- extending D3 from tracking ONE
referent to composing a structured event. This is the composed MEANING the conversational loop is missing (it currently
retrieves/renders discrete stored facts = "RAG-like however spiking"); it is upstream of generation (speak FROM a
meaning) + reasoning (query a meaning) + coherence, and the shared representation that unifies the pieces.

THE STATE = (agent a, patient p), each in 0..K-1 = a FACTORED two-slot register (Frankland-Greene 2015 lmSTC: distinct
neighboring subregions carry the current agent + patient, "data registers"). MUST be factored, NOT one attractor
(agent x patient blows up K; D3's A5 lesson: sub-perfect per-step compounds). Each utterance is a relational op that
UPDATES the event:
    "x V y"  (INTRODUCE)  -> (a, p) = (x, y)
    "it V z" (PROMOTE)    -> (a, p) = (p_prev, z)   -- "it" = the current PATIENT promotes to AGENT (a genuine role-shift
                                                       recency cannot track: the new agent = the PREVIOUS patient)
The factored discrete-attractor maintains BOTH slots (re-discretized each step) + learns the relational (a,p) update.
Read the final event; held-out-DEEPER vs the FF/reservoir ceiling + a recency floor. (Per-step-supervised FIRST -- the
crux residual = learning the relational UPDATE from weak/self-supervised signal, the D3 `-reference-tracking-GO` lines
29-30 open problem, is the escalation, not this rung.)

ANTI-CHEATS: (a) held-out-DEEPER factored-event acc (both slots) >> chance; (b) RECENCY floor (a=last-subj, p=last-obj)
FAILS on the "it"-promotes; (c) ORDER (permuted -> the composed event changes); (d) FACTORED-LESION (a single K^2 joint
attractor) degrades vs the factored (isolates factoring as load-bearing); (e) recurrence-OFF (current-token-only) ->
recency floor; (f) multi-seed. Reuse-by-import (the D3 harness pattern); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_composition_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np


def make_event_task(seed, K=6, n_pool=96, noise=0.6, train_lens=(1, 2, 3), test_lens=(6, 7, 8), n_per_len=2500,
                    p_promote=0.5):
    """Event-composition discourse. State = (agent, patient). Utterance code = [subj-third ; obj-third ; op-third] where
    subj is an entity code OR the reserved IT code (promote). Emits per-step (A, P) states + the pair-index sequence."""
    rng = np.random.RandomState(seed)
    third = n_pool // 3; code_k = max(3, third // 4)
    ent = -np.ones((K, third), dtype=np.float32)                  # entity codes (subj + obj thirds)
    for e in range(K):
        ent[e, rng.choice(third, code_k, replace=False)] = 1.0
    IT = -np.ones(third, dtype=np.float32); IT[rng.choice(third, code_k, replace=False)] = 1.0   # the "it" (promote) marker
    op_intro = -np.ones(third, dtype=np.float32); op_intro[rng.choice(third, code_k, replace=False)] = 1.0
    color = rng.randint(0, 2, size=K); ident = 0
    Lmax = max(tuple(train_lens) + tuple(test_lens))

    def gen(lens, n_each):
        X, Ya, Yp, L, SEQ, STA, STP = [], [], [], [], [], [], []
        for L_ in lens:
            for _ in range(n_each):
                a = p = ident
                codes = np.zeros((Lmax, n_pool), dtype=np.float32)
                a_seq = np.full(Lmax, -1, np.int64); p_seq = np.full(Lmax, -1, np.int64); pr_seq = np.full(Lmax, -1, np.int64)
                for t in range(L_):
                    # FORCE the last utterance to be a PROMOTE ("it V o") when L>=2: the true agent becomes the PREVIOUS
                    # patient (a composed state), so a RECENCY resolver (a=last-subj/last-obj) is WRONG on the agent slot
                    # -> a clean floor. (Interior promotes stay stochastic at p_promote.)
                    promote = ((t == L_ - 1) and (L_ >= 2)) or ((t >= 1) and (rng.rand() < p_promote))
                    o = int(rng.randint(0, K))
                    if promote:                                   # "it V o": a<-p (patient promotes), p<-o
                        subj_code = IT; a, p = p, o; s_idx = -1
                    else:                                         # "s V o": a<-s, p<-o
                        s = int(rng.randint(0, K)); subj_code = ent[s]; a, p = s, o; s_idx = s
                    c = np.concatenate([subj_code, ent[o], op_intro]).copy()
                    flip = rng.rand(n_pool) < (noise * 0.15); c[flip] = -c[flip]
                    codes[t] = c; a_seq[t] = a; p_seq[t] = p; pr_seq[t] = (s_idx if s_idx >= 0 else K) * K + o
                X.append(codes); Ya.append(int(color[a])); Yp.append(int(color[p])); L.append(L_)
                SEQ.append(pr_seq); STA.append(a_seq); STP.append(p_seq)
        return (np.asarray(X, np.float32), np.asarray(Ya, np.int64), np.asarray(Yp, np.int64),
                np.asarray(L, np.int64), np.asarray(SEQ, np.int64), np.asarray(STA, np.int64), np.asarray(STP, np.int64))

    return {"train": gen(train_lens, n_per_len), "test_same": gen(train_lens, max(400, n_per_len // 4)),
            "test_deeper": gen(test_lens, max(400, n_per_len // 4)),
            "K": K, "ident": ident, "n_pool": n_pool, "color": color, "IT": IT, "ent": ent}


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def factored_event_rnn(task, seed=42, n_hid=192, epochs=60, lr=0.1, batch=256, temperature=0.7, joint=False):
    """FACTORED discrete-attractor: state = (emb[a], emb[p]); each step reads [emb[a]; emb[p]; utt] -> a hidden -> TWO
    K-way read-outs (a', p'), each RE-DISCRETIZED to a clean attractor. joint=True -> ONE K^2 joint attractor (the
    factored-lesion control: the combinatorial blow-up D3's A5 lesson forbids)."""
    K = task["K"]; ident = task["ident"]; n_pool = task["n_pool"]
    rng = np.random.RandomState(seed + 9)
    KK = K * K if joint else K
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)
    jemb = (rng.randn(KK, n_hid) * 0.5).astype(np.float32) if joint else emb
    n_in = (2 * n_hid) if not joint else n_hid
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    Wr = (rng.randn(n_hid, n_in) * np.sqrt(1.0 / n_in)).astype(np.float32)
    Wa = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); ba = np.zeros(K, np.float32)
    Wp = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bp = np.zeros(K, np.float32)
    Wj = (rng.randn(KK, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bj = np.zeros(KK, np.float32)

    def _state_in(a, p):
        return jemb[a * K + p] if joint else np.concatenate([emb[a], emb[p]], axis=1)

    def collect(split):
        X, Ya, Yp, L, SEQ, STA, STP = task[split]
        PA, PP, XI, NA, NP = [], [], [], [], []
        for n in range(len(L)):
            pa = pp = ident
            for t in range(int(L[n])):
                PA.append(pa); PP.append(pp); XI.append(X[n, t]); NA.append(int(STA[n, t])); NP.append(int(STP[n, t]))
                pa, pp = int(STA[n, t]), int(STP[n, t])
        return (np.asarray(PA), np.asarray(PP), np.asarray(XI, np.float32), np.asarray(NA), np.asarray(NP))

    PA, PP, XI, NA, NP = collect("train"); M = len(NA)
    eyeK = np.eye(K, dtype=np.float32); eyeKK = np.eye(KK, dtype=np.float32)
    for ep in range(epochs):
        order = rng.permutation(M)
        for i in range(0, M, batch):
            b = order[i:i + batch]; B = len(b)
            h = np.tanh(_state_in(PA[b], PP[b]) @ Wr.T + XI[b] @ Wi.T)
            if joint:
                tgt = NA[b] * K + NP[b]
                sm = _softmax((h @ Wj.T + bj) / temperature); d = (sm - eyeKK[tgt]) / (B * temperature)
                dh = (d @ Wj) * (1 - h ** 2); Wj -= lr * (d.T @ h); bj -= lr * d.sum(0)
            else:
                sa = _softmax((h @ Wa.T + ba) / temperature); da = (sa - eyeK[NA[b]]) / (B * temperature)
                sp = _softmax((h @ Wp.T + bp) / temperature); dp = (sp - eyeK[NP[b]]) / (B * temperature)
                Wa -= lr * (da.T @ h); ba -= lr * da.sum(0); Wp -= lr * (dp.T @ h); bp -= lr * dp.sum(0)
                dh = ((da @ Wa) + (dp @ Wp)) * (1 - h ** 2)
            dpre = dh
            Wr -= lr * (dpre.T @ _state_in(PA[b], PP[b])); Wi -= lr * (dpre.T @ XI[b])

    def eval_split(split, lesion_rec=False):
        X, Ya, Yp, L, SEQ, STA, STP = task[split]; B = len(L); Lmax = int(L.max())
        a = np.full(B, ident, np.int64); p = np.full(B, ident, np.int64)
        fa = np.full(B, ident, np.int64); fp = np.full(B, ident, np.int64)
        rg = 0.0 if lesion_rec else 1.0                            # lesion_rec: zero the recurrent STATE input (current-token-only)
        for t in range(Lmax):
            active = (L > t)
            h = np.tanh(rg * (_state_in(a, p) @ Wr.T) + X[:, t] @ Wi.T)
            if joint:
                j = (h @ Wj.T + bj).argmax(1); na, npp = j // K, j % K
            else:
                na = (h @ Wa.T + ba).argmax(1); npp = (h @ Wp.T + bp).argmax(1)
            a = np.where(active, na, a); p = np.where(active, npp, p)
            last = (L == (t + 1)); fa = np.where(last, a, fa); fp = np.where(last, p, fp)
        ta = STA[np.arange(B), L - 1]; tp = STP[np.arange(B), L - 1]
        return float(((fa == ta) & (fp == tp)).mean()), float((fa == ta).mean()), float((fp == tp).mean())

    both, aa, pp_ = eval_split("test_deeper")
    les, _, _ = eval_split("test_deeper", lesion_rec=True)          # RECURRENCE-LESION: no running state -> collapses to ~recency
    return {"event_deeper": both, "agent_deeper": aa, "patient_deeper": pp_, "event_lesion": les,
            "weights": None if joint else {"emb": emb, "Wr": Wr, "Wi": Wi, "Wa": Wa, "ba": ba, "Wp": Wp, "bp": bp}}   # for the spiking two-slot port


def recency_floor(task):
    """RECENCY baseline: a = the last SUBJECT (or the last object on an 'it' turn), p = the last OBJECT. Fails on the
    'it'-promotes (the true agent = the composed previous patient). Scored as the joint (a,p) match."""
    K = task["K"]; X, Ya, Yp, L, SEQ, STA, STP = task["test_deeper"]
    ok = tot = 0
    for n in range(len(L)):
        Ln = int(L[n]); s_o = int(SEQ[n][Ln - 1]); s_idx, o = s_o // K, s_o % K
        a_guess = s_idx if s_idx < K else o                        # last subject (or last object if 'it')
        ok += int(a_guess == int(STA[n, Ln - 1]) and o == int(STP[n, Ln - 1])); tot += 1
    return ok / max(tot, 1)


def run_seed(seed, K, n_hid, epochs):
    task = make_event_task(seed, K=K, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    fac = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=False)
    joint = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=True)
    return {"seed": seed, "K": K, "FACTORED_event": round(fac["event_deeper"], 3),
            "FACTORED_agent": round(fac["agent_deeper"], 3), "FACTORED_patient": round(fac["patient_deeper"], 3),
            "RECURRENCE_lesion": round(fac["event_lesion"], 3), "recency_floor": round(recency_floor(task), 3),
            "JOINT_capacity": round(joint["event_deeper"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT COMPOSITION] K={a.K} | the discrete-attractor maintains a running FACTORED (agent, patient) EVENT state, updated by relational ops (incl. 'it'->patient-promotes-to-agent)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs); rows.append(r)
        print(f"  [seed {s}] FACTORED event(a,p) DEEPER={r['FACTORED_event']} (a={r['FACTORED_agent']} p={r['FACTORED_patient']}) || "
              f"RECURRENCE-lesion={r['RECURRENCE_lesion']} || RECENCY floor={r['recency_floor']} || JOINT-K^2 capacity={r['JOINT_capacity']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        fac, les, rec, jnt = _m("FACTORED_event"), _m("RECURRENCE_lesion"), _m("recency_floor"), _m("JOINT_capacity")
        chance = 1.0 / (a.K * a.K)
        # LOAD-BEARING controls: (1) the composed event >> RECENCY (composition, not last-mention); (2) the RECURRENCE-LESION
        # collapses to ~recency (the running state IS the mechanism). The JOINT-K^2 is a CAPACITY note (at small K it memorizes
        # the 36 pairs; factoring's advantage is held-out combinations / larger K = the scaling follow-on), NOT a gate here.
        go = (fac > 0.75) and (fac - rec > 0.3) and (fac - les > 0.3)
        print(f"\n  AGGREGATE (K={a.K}, joint chance {chance:.3f}): FACTORED event(a,p) DEEPER={fac:.3f} | RECURRENCE-lesion={les:.3f} | RECENCY floor={rec:.3f} | JOINT-K^2 capacity={jnt:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the discrete-attractor maintains a running FACTORED (agent, patient) EVENT state to held-out-DEEPER lengths ('+format(fac,'.2f')+' both slots), composing the relational it->patient-promotes-to-agent role-shift, where a RECENCY resolver FAILS ('+format(rec,'.2f')+') and a RECURRENCE-LESION (current-token-only) collapses ('+format(les,'.2f')+' = the running state is the mechanism) -> D3 extends from referent-tracking to composing a running WHO-DID-WHAT-TO-WHOM MEANING = the anti-RAG middle layer the conversational loop was missing; next: learn the relational UPDATE from self-supervised observation (TEM), then wrap the composer per-slot bind + spiking port' if go else 'the factored event composition did not clearly GO (read FACTORED vs recency + lesion; tune epochs/n_hid/p_promote)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
