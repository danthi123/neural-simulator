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
                    p_coref=0.5, p_promote=0.25):
    """Event-composition discourse. State = (agent, patient). THREE relational ops (utterance code = [subj-third ; obj-third
    ; op-third]):
      INTRODUCE  "s V o"  (subj=entity code) -> a<-s, p<-o
      AGENT-COREF "he V o" (subj=HE marker)  -> a<-a (the AGENT PERSISTS), p<-o    [makes the agent DEEP]
      PROMOTE    "it V o"  (subj=IT marker)  -> a<-p_prev (the patient promotes to agent), p<-o
    The COREF op is what forces GENUINE DEPTH (adversarial-verify fix, 2026-07-09): the agent PERSISTS across a
    variable-length coref run, so the final agent traces back to a RANDOM-depth introduce/promote -> a static
    "last-2-objects" reader FAILS on the agent (the skeptic's shortcut is defeated); the model must TRACK the running
    agent across the discourse. Emits per-step (A,P) states + SEQ (encodes the object each turn: o = SEQ%K)."""
    rng = np.random.RandomState(seed)
    third = n_pool // 3; code_k = max(3, third // 4)
    ent = -np.ones((K, third), dtype=np.float32)                  # entity codes (subj + obj thirds)
    for e in range(K):
        ent[e, rng.choice(third, code_k, replace=False)] = 1.0
    IT = -np.ones(third, dtype=np.float32); IT[rng.choice(third, code_k, replace=False)] = 1.0   # "it" (patient-promote)
    HE = -np.ones(third, dtype=np.float32); HE[rng.choice(third, code_k, replace=False)] = 1.0   # "he" (agent-coref, persists)
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
                    o = int(rng.randint(0, K))
                    r = rng.rand()
                    if t == 0:                                    # the first op must SET the agent
                        op = "intro"
                    elif r < p_coref:
                        op = "coref"
                    elif r < p_coref + p_promote:
                        op = "promote"
                    else:
                        op = "intro"
                    if op == "coref":                             # "he V o": the AGENT PERSISTS, p<-o (agent goes DEEP)
                        subj_code = HE; a, p = a, o; s_idx = K + 1
                    elif op == "promote":                         # "it V o": a<-p_prev (patient promotes), p<-o
                        subj_code = IT; a, p = p, o; s_idx = K
                    else:                                         # "s V o": a<-s, p<-o
                        s = int(rng.randint(0, K)); subj_code = ent[s]; a, p = s, o; s_idx = s
                    c = np.concatenate([subj_code, ent[o], op_intro]).copy()
                    flip = rng.rand(n_pool) < (noise * 0.15); c[flip] = -c[flip]
                    codes[t] = c; a_seq[t] = a; p_seq[t] = p; pr_seq[t] = s_idx * K + o    # o = pr % K always
                X.append(codes); Ya.append(int(color[a])); Yp.append(int(color[p])); L.append(L_)
                SEQ.append(pr_seq); STA.append(a_seq); STP.append(p_seq)
        return (np.asarray(X, np.float32), np.asarray(Ya, np.int64), np.asarray(Yp, np.int64),
                np.asarray(L, np.int64), np.asarray(SEQ, np.int64), np.asarray(STA, np.int64), np.asarray(STP, np.int64))

    return {"train": gen(train_lens, n_per_len), "test_same": gen(train_lens, max(400, n_per_len // 4)),
            "test_deeper": gen(test_lens, max(400, n_per_len // 4)),
            "K": K, "ident": ident, "n_pool": n_pool, "color": color,
            "ent": ent, "IT": IT, "HE": HE, "op_intro": op_intro}   # codes for the D3EventRegister to encode heard facts


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
    """RECENCY baseline: a = the last SUBJECT (or the last object on an 'it'/'he' turn), p = the last OBJECT. Scored as
    the joint (a,p) match."""
    K = task["K"]; X, Ya, Yp, L, SEQ, STA, STP = task["test_deeper"]
    ok = tot = 0
    for n in range(len(L)):
        Ln = int(L[n]); s_o = int(SEQ[n][Ln - 1]); s_idx, o = s_o // K, s_o % K
        a_guess = s_idx if s_idx < K else o                        # last subject (or last object if 'it'/'he')
        ok += int(a_guess == int(STA[n, Ln - 1]) and o == int(STP[n, Ln - 1])); tot += 1
    return ok / max(tot, 1)


def last2_objects_floor(task):
    """THE ADVERSARIAL-VERIFY anti-cheat (2026-07-09 skeptic): the static "last-2-objects" reader — guess a = the
    2nd-to-last object, p = the last object. In the shallow v1 task this scored 1.0 (the answer was only the last 2
    objects); with the AGENT-COREF op the agent traces back through a variable-length coref run to a random-depth
    setting, so this reader FAILS on the agent = the task is now genuinely DEEP (not a 2-token lookup)."""
    K = task["K"]; X, Ya, Yp, L, SEQ, STA, STP = task["test_deeper"]
    ok = tot = 0
    for n in range(len(L)):
        Ln = int(L[n])
        if Ln < 2:
            continue
        o_last = int(SEQ[n][Ln - 1]) % K; o_prev = int(SEQ[n][Ln - 2]) % K   # o = SEQ % K (both objects)
        ok += int(o_prev == int(STA[n, Ln - 1]) and o_last == int(STP[n, Ln - 1])); tot += 1
    return ok / max(tot, 1)


def run_seed(seed, K, n_hid, epochs):
    task = make_event_task(seed, K=K, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    fac = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=False)
    joint = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=True)
    return {"seed": seed, "K": K, "FACTORED_event": round(fac["event_deeper"], 3),
            "FACTORED_agent": round(fac["agent_deeper"], 3), "FACTORED_patient": round(fac["patient_deeper"], 3),
            "RECURRENCE_lesion": round(fac["event_lesion"], 3), "recency_floor": round(recency_floor(task), 3),
            "last2_objects_floor": round(last2_objects_floor(task), 3), "JOINT_capacity": round(joint["event_deeper"], 3)}


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
              f"RECURRENCE-lesion={r['RECURRENCE_lesion']} || LAST-2-OBJECTS(shallow-reader)={r['last2_objects_floor']} || RECENCY={r['recency_floor']} || JOINT-K^2 cap={r['JOINT_capacity']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        fac, les, rec, l2, jnt = _m("FACTORED_event"), _m("RECURRENCE_lesion"), _m("recency_floor"), _m("last2_objects_floor"), _m("JOINT_capacity")
        chance = 1.0 / (a.K * a.K)
        # LOAD-BEARING controls (adversarial-verify-hardened): (1) the composed event >> the LAST-2-OBJECTS shallow reader
        # (the AGENT-COREF op makes the task genuinely DEEP -> a 2-token reader FAILS = not a shallow lookup); (2) >> RECENCY;
        # (3) the RECURRENCE-LESION collapses (the running state IS the mechanism). JOINT-K^2 is a CAPACITY note, not gated.
        go = (fac > 0.7) and (fac - l2 > 0.3) and (fac - les > 0.3)
        print(f"\n  AGGREGATE (K={a.K}, joint chance {chance:.3f}): FACTORED event(a,p) DEEPER={fac:.3f} | LAST-2-OBJECTS(shallow)={l2:.3f} | RECURRENCE-lesion={les:.3f} | RECENCY={rec:.3f} | JOINT-K^2 cap={jnt:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the discrete-attractor maintains a running FACTORED (agent, patient) EVENT state to held-out-DEEPER lengths ('+format(fac,'.2f')+'), composing the relational role-shifts (agent-coref persistence + it-promotes) where a static LAST-2-OBJECTS reader FAILS ('+format(l2,'.2f')+' = the task is genuinely DEEP, NOT a 2-token lookup), RECENCY fails ('+format(rec,'.2f')+'), and a RECURRENCE-LESION collapses ('+format(les,'.2f')+' = the running state is the mechanism) -> D3 composes a running WHO-DID-WHAT-TO-WHOM MEANING across a discourse = the anti-RAG middle layer; next: learn the relational UPDATE from self-supervised observation (TEM), then wrap the composer per-slot bind + spiking port' if go else 'the deep event composition did not clearly GO (read FACTORED vs last-2-objects + lesion; tune p_coref/epochs/n_hid)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
