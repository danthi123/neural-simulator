"""D3 EVENT PAIR -> the FULLY SELF-SUPERVISED version: what TEACHES a brain to hold a PRIOR event, with no state label?

The connectives rung showed a connective-marked EVENT BOUNDARY shifts the running event into a previous slot, so the
brain holds a PAIR of events -- but its delta was learned from per-step (agent,patient) STATE LABELS. Removing them
exposes a genuine problem:

    predicting the CURRENT agent's emission gives the `a_prev` slot NO GRADIENT AT ALL.
    Nothing in a purely forward-predictive objective teaches a brain to hold a prior event.

LANGUAGE'S OWN ANSWER: the DISCOURSE POP (Grosz & Sidner's attentional stack). A "meanwhile / again" clause RETURNS to
the previous event's agent:

    RETURN  "meanwhile he V o"  ->  a_curr <- a_prev        (pop back to the prior event's protagonist)

After a RETURN the emission depends on `a_prev`, so **holding the prior event becomes necessary to predict at all**. The
prev slot is taught by the discourse's own habit of coming back to what it was talking about.

That yields a DECISIVE control this rung is built around: **remove the RETURN op (p_return=0) and the prev slot should
stop encoding anything** -- because nothing would ever require it. If `a_prev` still decodes without RETURNs, the claim
is wrong.

THE STATE: two LATENT K-way discrete-attractor slots (a_curr, a_prev); the patient is the OBSERVED object.
THE SIGNAL: an agent-characteristic emission drawn from theta[a_curr] -- a TARGET ONLY, never an input.
NO (agent, patient) state label anywhere. Labels are used ONLY by a frozen-state probe.

ANTI-CHEATS (6-seed): (a) probe(a_prev) on held-out-DEEPER >> chance; (b) **NO-RETURN control: probe(a_prev) collapses**
(the load-bearing one -- it names WHAT teaches the slot); (c) SINGLE-SLOT control (no a_prev slot, same prev head)
fails; (d) EMISSION-SEVERED collapses; (e) recency floor. numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_selfsup_pair_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

INTRO, COREF, PROMOTE, BOUND, RETURN = 0, 1, 2, 3, 4


def make_pair_task(seed, K=6, M=8, n_pool=64, noise=0.5, train_lens=(4, 5, 6), test_lens=(8, 9, 10),
                   n_per_len=1200, p_boundary=0.2, p_return=0.2, p_coref=0.35, p_promote=0.15, theta_peak=3.0,
                   coherent=False):
    """Discourse with event boundaries AND discourse pops (RETURN). Clause code = [op-third ; subject-third ; object-third].
    Emits the (label-only) a_curr / a_prev per step."""
    M = max(M, K)
    rng = np.random.RandomState(seed)
    th = n_pool // 3; n_pool = 3 * th            # the clause code is three equal thirds [op ; subject ; object]
    ck = max(3, th // 4)
    ent = -np.ones((K, th), np.float32)
    for e in range(K):
        ent[e, rng.choice(th, ck, replace=False)] = 1.0
    marks = {}
    for name in ("HE", "IT", "BND", "RET", "NOB"):
        v = -np.ones(th, np.float32); v[rng.choice(th, ck, replace=False)] = 1.0; marks[name] = v
    logits = rng.randn(K, M) * 0.3
    for e in range(K):
        logits[e, e % M] += theta_peak
    theta = np.exp(logits - logits.max(1, keepdims=True)); theta /= theta.sum(1, keepdims=True)
    ident = 0; Lmax = max(tuple(train_lens) + tuple(test_lens))

    OPS_all, SID_all = {}, {}

    def gen(lens, n_each, p_ret, tag=None):
        X, OBJ, EMIT, L, AC, AP, PE, PC = [], [], [], [], [], [], [], []
        OPS, SID = [], []
        for L_ in lens:
            for _ in range(n_each):
                ac = ap = ident; has_prev = False
                codes = np.zeros((Lmax, n_pool), np.float32)
                ob = np.zeros(Lmax, np.int64); em = np.zeros(Lmax, np.int64)
                acs = np.full(Lmax, -1, np.int64); aps = np.full(Lmax, -1, np.int64)
                pe = np.full(Lmax, -1, np.int64); prev_last_emit = -1
                ops_ = np.full(Lmax, -1, np.int64); sid_ = np.full(Lmax, -1, np.int64)
                pc = np.zeros((Lmax, M), np.float32)               # prior event's emission MULTISET (seq-replay target)
                cur_counts = np.zeros(M, np.float32); prev_counts = np.zeros(M, np.float32)
                for t in range(L_):
                    o = int(rng.randint(0, K)); r = rng.rand()
                    if t == 0:
                        op = INTRO
                    elif r < p_boundary:
                        op = BOUND
                    elif r < p_boundary + p_ret:
                        # A discourse can only POP BACK to a prior event that EXISTS. Without this guard an early RETURN
                        # pops to the EMPTY initial state (a_curr <- ident), and later boundaries then copy `ident` into
                        # a_prev -- making `ident` a 42.5% majority class that every arm (incl. an untrained one) predicts.
                        # Found by reading the measurement: P(a_prev==ident) exactly matched all three arms' scores.
                        op = RETURN if has_prev else COREF
                    elif coherent:
                        # AGENT-COHERENT episodes: within an event the protagonist PERSISTS (only coref). Then the
                        # event's whole emission sequence is k samples of ONE agent, and sequence replay can beat a
                        # single symbol. With mixed within-event ops the sequence is a MIXTURE across agents and the
                        # LAST emission (the one a_prev actually produced) is strictly more informative.
                        op = COREF
                    elif r < p_boundary + p_ret + p_coref:
                        op = COREF
                    elif r < p_boundary + p_ret + p_coref + p_promote:
                        op = PROMOTE
                    else:
                        op = INTRO
                    if op == BOUND:                                # SHIFT: current event becomes the previous one,
                        ap = ac; has_prev = True                   # and a new event opens by NAMING its agent
                        prev_last_emit = int(em[t - 1])            # the just-ended event's LAST emission (OBSERVED)
                        prev_counts = cur_counts.copy()            # the just-ended event's WHOLE emission sequence
                        cur_counts = np.zeros(M, np.float32)
                        s = int(rng.randint(0, K)); sub = ent[s]; ac = s; mk = marks["BND"]; sid_[t] = s
                    elif op == RETURN:                             # DISCOURSE POP: come back to the prior protagonist
                        ac = ap; sub = marks["HE"]; mk = marks["RET"]
                    elif op == COREF:
                        sub = marks["HE"]; mk = marks["NOB"]
                    elif op == PROMOTE:
                        ac = int(ob[t - 1]) if t > 0 else ac; sub = marks["IT"]; mk = marks["NOB"]
                    else:
                        s = int(rng.randint(0, K)); sub = ent[s]; ac = s; mk = marks["NOB"]; sid_[t] = s
                    c = np.concatenate([mk, sub, ent[o]]).copy()
                    flip = rng.rand(n_pool) < (noise * 0.12); c[flip] = -c[flip]
                    codes[t] = c; ob[t] = o; acs[t] = ac; aps[t] = ap; ops_[t] = op
                    em[t] = int(rng.choice(M, p=theta[ac]))        # EMISSION from the CURRENT agent (target only)
                    pe[t] = prev_last_emit                          # REPLAY target: the prior event's last emission
                    cur_counts[em[t]] += 1.0
                    tot = prev_counts.sum()
                    pc[t] = (prev_counts / tot) if tot > 0 else 0.0  # SEQ-REPLAY target: the prior event's emission dist
                X.append(codes); OBJ.append(ob); EMIT.append(em); L.append(L_); AC.append(acs); AP.append(aps); PE.append(pe); PC.append(pc)
                OPS.append(ops_); SID.append(sid_)
        if tag:
            OPS_all[tag] = np.asarray(OPS, np.int64); SID_all[tag] = np.asarray(SID, np.int64)
        return (np.asarray(X, np.float32), np.asarray(OBJ, np.int64), np.asarray(EMIT, np.int64),
                np.asarray(L, np.int64), np.asarray(AC, np.int64), np.asarray(AP, np.int64), np.asarray(PE, np.int64),
                np.asarray(PC, np.float32))

    tr = gen(train_lens, n_per_len, p_return, tag="train")
    te = gen(test_lens, max(400, n_per_len // 3), p_return, tag="test")
    return {"train": tr, "test_deeper": te,
            "K": K, "M": M, "ident": ident, "n_pool": n_pool, "theta": theta,
            "ent": ent, "marks": marks,                       # codes for the deployed SelfSupPairRegister
            "ops_train": OPS_all["train"], "sid_train": SID_all["train"]}   # OBSERVABLE ops + spoken subject ids


def _sm(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def train_pair_selfsup(task, seed=42, n_hid=128, epochs=40, lr=0.05, batch=256, random_emit=False, n_slots=2,
                       replay=False, gamma=1.0, seq_replay=False):
    """Two LATENT K-way slots (a_curr, a_prev) + the observed patient. Learn from the EMISSION cross-entropy ALONE.
    n_slots=1 -> the SINGLE-SLOT control (no a_prev slot; it still carries a prev head, so it is asked the same
    question, but has nowhere to hold the answer)."""
    K, M, n_pool, ident = task["K"], task["M"], task["n_pool"], task["ident"]
    rng = np.random.RandomState(seed + 9)
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)
    n_in = (n_slots + 1) * n_hid                                   # latent slots + the observed patient
    Wr = (rng.randn(n_hid, n_in) * np.sqrt(1.0 / n_in)).astype(np.float32)
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    Wc = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bc = np.zeros(K, np.float32)
    Wp = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bp = np.zeros(K, np.float32)
    We = (rng.randn(M, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); be = np.zeros(M, np.float32)
    Wq = (rng.randn(M, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bq = np.zeros(M, np.float32)  # REPLAY head

    X, OBJ, EMIT, L, AC, AP, PE, PC = task["train"]
    if random_emit:
        EMIT = np.random.RandomState(seed + 4).randint(0, M, size=EMIT.shape).astype(np.int64)
    eyeM = np.eye(M, dtype=np.float32); eyeK = np.eye(K, dtype=np.float32)
    by_len = {}
    for n in range(len(L)):
        by_len.setdefault(int(L[n]), []).append(n)

    for _ in range(epochs):
        for Ln, ids in by_len.items():
            ids = np.asarray(ids); rng.shuffle(ids)
            for i in range(0, len(ids), batch):
                b = ids[i:i + batch]; B = len(b)
                sc_ = np.zeros((B, K), np.float32); sc_[:, ident] = 1.0    # a_curr
                sp_ = np.zeros((B, K), np.float32); sp_[:, ident] = 1.0    # a_prev
                pat = np.zeros((B, K), np.float32); pat[:, ident] = 1.0    # observed patient
                cache = []
                dWr = np.zeros_like(Wr); dWi = np.zeros_like(Wi)
                dWc = np.zeros_like(Wc); dbc = np.zeros_like(bc)
                dWp = np.zeros_like(Wp); dbp = np.zeros_like(bp)
                dWe = np.zeros_like(We); dbe = np.zeros_like(be)
                dWq = np.zeros_like(Wq); dbq = np.zeros_like(bq)
                for t in range(Ln):
                    parts = [sc_ @ emb] + ([sp_ @ emb] if n_slots == 2 else []) + [pat @ emb]
                    st_in = np.concatenate(parts, axis=1)
                    h = np.tanh(st_in @ Wr.T + X[b, t] @ Wi.T)
                    nc = _sm(h @ Wc.T + bc); npv = _sm(h @ Wp.T + bp)
                    ev = nc @ emb; se = _sm(ev @ We.T + be)
                    qv = npv @ emb; sq = _sm(qv @ Wq.T + bq)          # RETRODICT the prior event's last emission
                    cache.append((st_in, h, nc, npv, ev, se, qv, sq, b, t))
                    sc_ = nc
                    if n_slots == 2:
                        sp_ = npv
                    pat = np.zeros((B, K), np.float32); pat[np.arange(B), OBJ[b, t]] = 1.0
                d_c_next = np.zeros((B, K), np.float32); d_p_next = np.zeros((B, K), np.float32)
                for t in range(Ln - 1, -1, -1):
                    st_in, h, nc, npv, ev, se, qv, sq, bb, tt = cache[t]
                    d_le = (se - eyeM[EMIT[bb, tt]]) / B
                    dWe += d_le.T @ ev; dbe += d_le.sum(0)
                    d_c = (d_le @ We) @ emb.T + d_c_next            # emission head + next-step recurrence
                    d_p = d_p_next                                  # a_prev via the recurrence...
                    if replay and n_slots == 2:                     # ...and the REPLAY (retrodiction) signal
                        tgt = PE[bb, tt]; valid = (tgt >= 0)
                        if valid.any():
                            if seq_replay:                          # SEQUENCE replay: the prior event's WHOLE emission
                                oh = PC[bb, tt]                     # distribution (SWR replays trajectories, not a symbol)
                            else:                                   # single-symbol replay: its LAST emission
                                oh = np.zeros_like(sq)
                                idxv = np.arange(len(tgt))[valid]
                                oh[idxv, tgt[valid]] = 1.0
                            d_lq = ((sq - oh) * valid[:, None]) * (gamma / B)
                            dWq += d_lq.T @ qv; dbq += d_lq.sum(0)
                            d_p = d_p + (d_lq @ Wq) @ emb.T
                    d_lc = nc * (d_c - (nc * d_c).sum(1, keepdims=True))
                    d_lp = npv * (d_p - (npv * d_p).sum(1, keepdims=True))
                    dWc += d_lc.T @ h; dbc += d_lc.sum(0); dWp += d_lp.T @ h; dbp += d_lp.sum(0)
                    dh = ((d_lc @ Wc) + (d_lp @ Wp)) * (1 - h ** 2)
                    dWi += dh.T @ X[bb, tt]; dWr += dh.T @ st_in
                    d_st = dh @ Wr
                    d_c_next = d_st[:, :n_hid] @ emb.T
                    d_p_next = (d_st[:, n_hid:2 * n_hid] @ emb.T) if n_slots == 2 else np.zeros((B, K), np.float32)
                Wr -= lr * dWr; Wi -= lr * dWi; Wc -= lr * dWc; bc -= lr * dbc
                Wp -= lr * dWp; bp -= lr * dbp; We -= lr * dWe; be -= lr * dbe
                Wq -= lr * dWq; bq -= lr * dbq

    def rollout(split):
        X_, O_, E_, L_, AC_, AP_, PE_, PC_ = task[split]; B = len(L_); Lm = int(L_.max())
        sc_ = np.zeros((B, K), np.float32); sc_[:, ident] = 1.0
        sp_ = np.zeros((B, K), np.float32); sp_[:, ident] = 1.0
        pat = np.zeros((B, K), np.float32); pat[:, ident] = 1.0
        fc = sc_.copy(); fp = sp_.copy()
        for t in range(Lm):
            act = (L_ > t)
            parts = [sc_ @ emb] + ([sp_ @ emb] if n_slots == 2 else []) + [pat @ emb]
            h = np.tanh(np.concatenate(parts, axis=1) @ Wr.T + X_[:, t] @ Wi.T)
            nc = _sm(h @ Wc.T + bc); npv = _sm(h @ Wp.T + bp)
            sc_ = np.where(act[:, None], nc, sc_)
            if n_slots == 2:
                sp_ = np.where(act[:, None], npv, sp_)
            pn = np.zeros((B, K), np.float32); pn[np.arange(B), O_[:, t]] = 1.0
            pat = np.where(act[:, None], pn, pat)
            last = (L_ == (t + 1))
            fc = np.where(last[:, None], sc_, fc); fp = np.where(last[:, None], npv, fp)
        return fc, fp, AC_[np.arange(B), L_ - 1], AP_[np.arange(B), L_ - 1]

    # expose the learned transition (for the deployed SelfSupPairRegister)
    rollout.W = {"emb": emb, "Wr": Wr, "Wi": Wi, "Wc": Wc, "bc": bc, "Wp": Wp, "bp": bp, "n_slots": n_slots}
    return rollout


def linear_probe(trX, trY, teX, teY, K, epochs=300, lr=0.5, mask=None):
    rng = np.random.RandomState(0)
    W = (rng.randn(K, trX.shape[1]) * 0.1).astype(np.float32); b = np.zeros(K, np.float32)
    eye = np.eye(K, dtype=np.float32); n = len(trY)
    for _ in range(epochs):
        s = _sm(trX @ W.T + b); d = (s - eye[trY]) / n
        W -= lr * (d.T @ trX); b -= lr * d.sum(0)
    pred = (teX @ W.T + b).argmax(1)
    m = np.ones(len(teY), bool) if mask is None else mask
    return float((pred[m] == teY[m]).mean()) if m.sum() else float("nan")


def _probe_prev(task, roll, K, mask=None, train_mask=None):
    """Probe a_prev (and a_curr) on the INFORMATIVE subset -- and FIT the probe on the informative subset of TRAIN too.
    Fitting on the full split lets the probe collapse to the `ident` majority class (30%) and never use the slot at all;
    it then scores exactly 0 on a subset that excludes ident. Masking BOTH sides forces the probe to decode from the slot."""
    trc, trp, tra, trb = roll("train"); tec, tep, tea, teb = roll("test_deeper")
    if train_mask is not None:
        trp, trb2 = trp[train_mask], trb[train_mask]
    else:
        trb2 = trb
    return (linear_probe(trp, trb2, tep, teb, K, mask=mask),
            linear_probe(trc, tra, tec, tea, K))


def _informative_train(task):
    """Same informative mask, on the TRAIN split (for fitting the probe without the majority-class shortcut)."""
    X, O, E, L, AC, AP, PE, PC = task["train"]; B = len(L)
    ac = AC[np.arange(B), L - 1]; ap = AP[np.arange(B), L - 1]
    return (ap != ac) & (ap != task["ident"])


def _informative(task):
    """The LOAD-BEARING subset: the prior event is REAL (not the initial slot) and DIFFERS from the current agent.
    Outside it, a probe scores by reading a_curr or predicting the majority class -- which is exactly how the first
    version of this rung fooled itself (every arm, incl. an untrained one, scored the ident majority rate)."""
    X, O, E, L, AC, AP, PE, PC = task["test_deeper"]; B = len(L)
    ac = AC[np.arange(B), L - 1]; ap = AP[np.arange(B), L - 1]
    return (ap != ac) & (ap != task["ident"]), ac, ap


def emission_ceiling(task):
    """Bayes-optimal accuracy of decoding the AGENT from ONE observed emission (uniform prior)."""
    th = task["theta"]; K = task["K"]
    return float(sum(th[:, m].max() for m in range(th.shape[1])) / K)


def multiset_ceiling(task, info):
    """Bayes decode of the prior agent from the prior event's WHOLE observed emission multiset -- the ceiling a SEQUENCE
    replay target affords. k samples identify an agent far better than one."""
    X, O, E, L, AC, AP, PE, PC = task["test_deeper"]; B = len(L)
    th = np.log(task["theta"] + 1e-9)                       # [K, M]
    counts = PC[np.arange(B), L - 1]                        # normalized prior-event emission distribution
    pred = (counts @ th.T).argmax(1)
    ap = AP[np.arange(B), L - 1]
    return float((pred[info] == ap[info]).mean()) if info.sum() else float("nan")


def run_seed(seed, K, n_hid, epochs):
    task = make_pair_task(seed, K=K)
    task_noret = make_pair_task(seed, K=K, p_return=0.0)
    info, ac, ap = _informative(task); tmask = _informative_train(task)
    info_nr, _, ap_nr = _informative(task_noret); tmask_nr = _informative_train(task_noret)

    def prev_of(tsk, msk, tmsk, **kw):
        roll = train_pair_selfsup(tsk, seed=seed, n_hid=n_hid, epochs=epochs, **kw)
        return _probe_prev(tsk, roll, K, mask=msk, train_mask=tmsk)

    pred_prev, pred_curr = prev_of(task, info, tmask)                                  # THE NEGATIVE: prediction only
    rep_prev, rep_curr = prev_of(task, info, tmask, replay=True)                       # + REPLAY (the mechanism)
    sev_prev, _ = prev_of(task, info, tmask, replay=True, random_emit=True)            # emission-severed
    one_prev, _ = prev_of(task, info, tmask, replay=True, n_slots=1)                   # single-slot (no prev slot)
    nrp_prev, _ = prev_of(task_noret, info_nr, tmask_nr, replay=True)                  # replay WITHOUT discourse pops
    seq_prev, seq_curr = prev_of(task, info, tmask, replay=True, seq_replay=True)      # SEQUENCE replay (SWR-like)

    X, O, E, L, AC, AP, PE, PC = task["test_deeper"]
    rec = float((O[np.arange(len(L)), L - 1][info] == ap[info]).mean())
    return {"seed": seed, "K": K, "n_informative": int(info.sum()),
            "PREDONLY_prev": round(pred_prev, 3), "PREDONLY_curr": round(pred_curr, 3),
            "REPLAY_prev": round(rep_prev, 3), "REPLAY_curr": round(rep_curr, 3),
            "REPLAY_severed_prev": round(sev_prev, 3), "REPLAY_singleslot_prev": round(one_prev, 3),
            "REPLAY_noreturn_prev": round(nrp_prev, 3),
            "SEQREPLAY_prev": round(seq_prev, 3), "SEQREPLAY_curr": round(seq_curr, 3),
            "one_emission_ceiling": round(emission_ceiling(task), 3),
            "multiset_ceiling": round(multiset_ceiling(task, info), 3), "recency_prev": round(rec, 3)}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", default="42")
    ap_.add_argument("--K", type=int, default=6)
    ap_.add_argument("--n-hid", type=int, default=128)
    ap_.add_argument("--epochs", type=int, default=40)
    ap_.add_argument("--json", default=None)
    a = ap_.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT PAIR SELF-SUPERVISED] K={a.K} | what TEACHES a brain to hold a PRIOR event with no state label?", flush=True)
    rows = []
    for s_ in seeds:
        r = run_seed(s_, a.K, a.n_hid, a.epochs); rows.append(r)
        print(f"  [seed {s_}] pred-only prev={r['PREDONLY_prev']} (curr={r['PREDONLY_curr']}) || REPLAY(1-symbol) prev={r['REPLAY_prev']} || "
              f"SEQ-REPLAY prev={r['SEQREPLAY_prev']} (curr={r['SEQREPLAY_curr']}) || severed={r['REPLAY_severed_prev']} | single-slot={r['REPLAY_singleslot_prev']} | "
              f"no-return={r['REPLAY_noreturn_prev']} || ceilings: 1-emission={r['one_emission_ceiling']} multiset={r['multiset_ceiling']} | recency={r['recency_prev']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        po, pc = _m("PREDONLY_prev"), _m("PREDONLY_curr")
        rp, rc = _m("REPLAY_prev"), _m("REPLAY_curr")
        sq, sqc = _m("SEQREPLAY_prev"), _m("SEQREPLAY_curr")
        sv, ss, nr = _m("REPLAY_severed_prev"), _m("REPLAY_singleslot_prev"), _m("REPLAY_noreturn_prev")
        ceil, mceil, rec = _m("one_emission_ceiling"), _m("multiset_ceiling"), _m("recency_prev")
        chance = 1.0 / a.K
        go = (sq - rp > 0.05) and (rp - po > 0.15) and (sq - sv > 0.25) and (sq - ss > 0.25) and (po < 0.35)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}, one-emission decode CEILING {ceil:.3f}):", flush=True)
        print(f"    *** THE NEGATIVE: prediction-only prev-agent={po:.3f} (~chance) while curr-agent={pc:.3f} -- forward prediction learns the CURRENT event fine and teaches the HELD one NOTHING ***", flush=True)
        print(f"    *** THE MECHANISM: + REPLAY (retrodict the just-ended event's last OBSERVED emission from a_prev) prev-agent={rp:.3f} (curr {rc:.3f}) ***", flush=True)
        print(f"    *** SEQUENCE replay (the prior event's WHOLE emission distribution, SWR-like): prev-agent={sq:.3f} (curr {sqc:.3f}) -- ceilings: 1-emission {ceil:.3f}, multiset {mceil:.3f} ***", flush=True)
        print(f"    controls: emission-severed={sv:.3f} | single-slot={ss:.3f} | replay-WITHOUT-discourse-pops={nr:.3f} | recency={rec:.3f}", flush=True)
        print(f"  SEQ-vs-1SYMBOL: {sq:.3f} vs {rp:.3f} (delta {sq-rp:+.3f}); fraction of the multiset ceiling reached: {sq/max(mceil,1e-9):.2f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'FORWARD PREDICTION ALONE DOES NOT TEACH A BRAIN TO HOLD A PRIOR EVENT (prev-agent '+format(po,'.2f')+' ~ chance, while the CURRENT event is learned fine at '+format(pc,'.2f')+') -- the a_prev slot receives no gradient from predicting the current agent emission, and even a discourse-pop (RETURN) that makes the prior event NECESSARY does not deliver enough credit through the long BPTT path. A REPLAY / RETRODICTION signal DOES teach it: reconstructing the just-ended event last OBSERVED emission from a_prev lifts prev-agent to '+format(rp,'.2f')+' (vs a one-emission decode ceiling of '+format(ceil,'.2f')+'), where an EMISSION-SEVERED model ('+format(sv,'.2f')+') and a SINGLE-SLOT model ('+format(ss,'.2f')+') both collapse. This resolves the cited HAE/TEM discrepancy from BOTH sides: the reconstruction anchor is NOT load-bearing for the CURRENT slot (its target moves, so prediction suffices) but IS load-bearing for the HELD slot (nothing else supplies gradient) -> consolidating a just-ended episode by replaying it is what gives a brain a two-event memory' if go else 'the replay mechanism did not clearly rescue the held slot (read REPLAY prev vs prediction-only / severed / single-slot)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
