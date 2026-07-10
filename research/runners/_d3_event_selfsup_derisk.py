"""D3 EVENT -> the FULLY SELF-SUPERVISED transition (the genuinely-open crux): learn the event delta with NO
(agent, patient) STATE LABEL anywhere. Every prior event rung learned delta from a host state label (per-step, or
end-state-only in the weak-supervision rung). A real brain is never TOLD who the agent is -- the running state must FALL
OUT of predicting what it hears. This removes the last host supervision in the event composition (the master-directive
"emergent, developed from experience" requirement).

THE SELF-SUPERVISED SIGNAL (TEM / Whittington 2020 factorization).
Each clause EMITS a symbol drawn from the CURRENT AGENT's distribution theta[a_t] (an agent-characteristic emission --
"how this entity sounds"). The emission is a **TARGET ONLY, NEVER AN INPUT** (audited), so the model cannot read the
agent off the current observation: to predict the emission it must MAINTAIN the running agent --
    INTRODUCE "s V o"    -> the subject NAMES the agent      (the slot is SET)
    AGENT-COREF "he V o" -> the utterance does NOT name it   (the slot must PERSIST -- deep)
    PROMOTE  "it V o"    -> the agent becomes the prev PATIENT (the slot must BIND the observed object)
Learning signal = the emission cross-entropy ALONE.

WHY PREDICTION-ALONE SUFFICES HERE (the cited-literature discrepancy, ADJUDICATED by adversarial verification).
The project's TEM/HAE read says "loss = L_rec + gamma*L_pred; prediction-alone collapses to identity, so the
reconstruction anchor is load-bearing." This runner uses prediction ALONE, no anchor, and it works. The reason is NOT
the K-way bottleneck (a skeptic REFUTED that: handing the model a copy path -- feeding the previous emission as an input
-- makes the probe RISE to 0.96, not collapse, so the identity solution simply does not exist here). The real reason:
**the emission target MOVES across the discourse** -- introduce/promote switch the agent, while coref leaves the SUBJECT
agent-independent but the EMISSION agent-dependent -- so there is no static input->target map to collapse onto. The
target itself requires memory, which makes prediction alone a sufficient self-supervised signal in this setting.

EVAL = a frozen-state linear PROBE (state -> agent identity) trained on TRAIN and read out on held-out-DEEPER. The
identity labels are used ONLY to READ what the unsupervised state encodes -- never to learn delta (standard
representation probing; biologically, a downstream region learning to read the slot).

CONTROLS (hardened after 3 adversarial skeptics; see the finding for the numbers they overturned):
  * HONEST LABEL-FREE FLOOR = `last-named-subject` (latch the most recent INTRODUCE subject, ignore coref/promote;
    reads only the SUBJ codes, zero labels). This is the floor the claim must beat -- NOT `recency` (last-mentioned
    object), which a skeptic showed is a WEAK strawman (0.167 vs last-named-subject's ~0.58). The genuine edge over it
    is PROMOTE-BINDING (~half of finals), which last-named-subject structurally misses.
  * FAIR RESERVOIR = a proper echo-state network (large random recurrent hidden + ridge readout). The old `untrained`
    arm was a DEGENERATE reservoir (it probed only the collapsed K-dim softmax slot of a random-init net, 0.24) and
    inflated the learning margin ~3x. A fair ESN scores ~0.69 overall -- but COLLAPSES TO CHANCE on deep coref, which
    is exactly where the trained model holds. That depth-resolved contrast is the real evidence.
  * EMISSION-SEVERED = `random_emit` (emissions independent of the agent). NOTE: this lands on the same number as the
    untrained arm (~0.26, state-cosine 0.84) -- they are ONE control axis, not two, and are reported as such.
  * NO-RECURRENCE = zero the recurrent state input (a fair single-variable ablation: with rg=0, d h/d Wr = 0, so
    dropping dWr is mathematically correct; Wi/Wa still train).
  * DEPTH-CONDITIONED read-out: the aggregate is shallow-dominated (~half of finals have the agent set at the LAST
    clause), so "coref-DEEP" is only earned on the depth>=3 subset, reported separately.

SCOPE (skeptic-established): requires M >= K (theta peaks on e % M, so K > M ALIASES agents -- auto-enforced below).
Scaling K needs proportional capacity (K=10 wants n_hid 256 / epochs >= 80). Robust to emission noise (~0.87 at
49%-modal) and to deep coref (p_coref=0.8 -> 0.88).

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_selfsup_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

INTRO, COREF, PROMOTE = 0, 1, 2


def make_selfsup_event_task(seed, K=6, M=8, n_pool=48, noise=0.5, train_lens=(2, 3, 4), test_lens=(6, 7, 8),
                            n_per_len=1500, p_coref=0.5, p_promote=0.25, theta_peak=3.0):
    """Event stream + an agent-characteristic EMISSION (the self-supervised target; NEVER an input).
    Emits SUBJ codes, SUBJ_ID (entity idx for INTRODUCE else -1), OBJ ids, EMIT ids, OPS, and the (label-only) agent."""
    M = max(M, K)                       # SCOPE FIX: theta peaks on e % M -> K > M would ALIAS agents (skeptic-found bug)
    rng = np.random.RandomState(seed)
    ent = -np.ones((K, n_pool), dtype=np.float32); code_k = max(3, n_pool // 6)
    for e in range(K):
        ent[e, rng.choice(n_pool, code_k, replace=False)] = 1.0
    HE = -np.ones(n_pool, np.float32); HE[rng.choice(n_pool, code_k, replace=False)] = 1.0
    IT = -np.ones(n_pool, np.float32); IT[rng.choice(n_pool, code_k, replace=False)] = 1.0
    logits = rng.randn(K, M) * 0.3
    for e in range(K):
        logits[e, e % M] += theta_peak
    theta = np.exp(logits - logits.max(1, keepdims=True)); theta /= theta.sum(1, keepdims=True)
    purity = float(np.mean([theta[e, e % M] for e in range(K)]))   # P(emission == the agent's modal symbol)
    ident = 0; Lmax = max(tuple(train_lens) + tuple(test_lens))

    def gen(lens, n_each):
        SUBJ, SID, OBJ, EMIT, OPS, L, TA = [], [], [], [], [], [], []
        for L_ in lens:
            for _ in range(n_each):
                a = p = ident
                sc = np.zeros((Lmax, n_pool), np.float32)
                sid = np.full(Lmax, -1, np.int64); ob = np.zeros(Lmax, np.int64)
                em = np.zeros(Lmax, np.int64); op_ = np.full(Lmax, -1, np.int64); ta = np.full(Lmax, -1, np.int64)
                for t in range(L_):
                    o = int(rng.randint(0, K)); r = rng.rand()
                    op = INTRO if t == 0 else (COREF if r < p_coref else (PROMOTE if r < p_coref + p_promote else INTRO))
                    if op == COREF:
                        code = HE                                    # agent PERSISTS (deep; not named)
                    elif op == PROMOTE:
                        code = IT; a = p                             # agent <- the previous (observed) patient
                    else:
                        s = int(rng.randint(0, K)); code = ent[s]; a = s; sid[t] = s   # the subject NAMES the agent
                    p = o
                    c = code.copy(); flip = rng.rand(n_pool) < (noise * 0.12); c[flip] = -c[flip]
                    sc[t] = c; ob[t] = o; ta[t] = a; op_[t] = op
                    em[t] = int(rng.choice(M, p=theta[a]))           # EMISSION: drawn from the CURRENT agent
                SUBJ.append(sc); SID.append(sid); OBJ.append(ob); EMIT.append(em); OPS.append(op_); L.append(L_); TA.append(ta)
        return (np.asarray(SUBJ, np.float32), np.asarray(SID, np.int64), np.asarray(OBJ, np.int64),
                np.asarray(EMIT, np.int64), np.asarray(OPS, np.int64), np.asarray(L, np.int64), np.asarray(TA, np.int64))

    return {"train": gen(train_lens, n_per_len), "test_deeper": gen(test_lens, max(400, n_per_len // 3)),
            "K": K, "M": M, "ident": ident, "n_pool": n_pool, "theta": theta, "emission_purity": round(purity, 3)}


def _sm(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def train_selfsup(task, seed=42, n_hid=128, epochs=40, lr=0.05, batch=256, random_emit=False, no_recurrence=False):
    """Learn delta from the EMISSION cross-entropy ALONE (no state label). The K-way slot `sa` is the ONLY recurrent
    state; the emission is a target, never an input."""
    K, M, n_pool, ident = task["K"], task["M"], task["n_pool"], task["ident"]
    rng = np.random.RandomState(seed + 9)
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)              # FIXED entity codes
    Wr = (rng.randn(n_hid, 2 * n_hid) * np.sqrt(1.0 / (2 * n_hid))).astype(np.float32)
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    Wa = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); ba = np.zeros(K, np.float32)
    We = (rng.randn(M, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); be = np.zeros(M, np.float32)
    rg = 0.0 if no_recurrence else 1.0

    SUBJ, SID, OBJ, EMIT, OPS, L, TA = task["train"]
    if random_emit:            # CONTROL: emissions INDEPENDENT of the agent (the link is severed)
        EMIT = np.random.RandomState(seed + 4).randint(0, M, size=EMIT.shape).astype(np.int64)
    eyeM = np.eye(M, dtype=np.float32)
    by_len = {}
    for n in range(len(L)):
        by_len.setdefault(int(L[n]), []).append(n)

    for _ in range(epochs):
        for Ln, ids in by_len.items():
            ids = np.asarray(ids); rng.shuffle(ids)
            for i in range(0, len(ids), batch):
                b = ids[i:i + batch]; B = len(b)
                sa = np.zeros((B, K), np.float32); sa[:, ident] = 1.0
                sp = np.zeros((B, K), np.float32); sp[:, ident] = 1.0
                cache = []
                dWr = np.zeros_like(Wr); dWi = np.zeros_like(Wi); dWa = np.zeros_like(Wa); dba = np.zeros_like(ba)
                dWe = np.zeros_like(We); dbe = np.zeros_like(be)
                for t in range(Ln):
                    st_in = np.concatenate([sa @ emb, sp @ emb], axis=1)
                    h = np.tanh(rg * (st_in @ Wr.T) + SUBJ[b, t] @ Wi.T)
                    sa_new = _sm(h @ Wa.T + ba)
                    ev = sa_new @ emb; se = _sm(ev @ We.T + be)
                    cache.append((st_in, h, sa_new, ev, se, b, t))
                    sa = sa_new
                    sp = np.zeros((B, K), np.float32); sp[np.arange(B), OBJ[b, t]] = 1.0   # patient = the OBSERVED object
                d_sa_next = np.zeros((B, K), np.float32)
                for t in range(Ln - 1, -1, -1):
                    st_in, h, sa_new, ev, se, bb, tt = cache[t]
                    d_le = (se - eyeM[EMIT[bb, tt]]) / B
                    dWe += d_le.T @ ev; dbe += d_le.sum(0)
                    d_sa = (d_le @ We) @ emb.T + d_sa_next
                    d_la = sa_new * (d_sa - (sa_new * d_sa).sum(1, keepdims=True))
                    dWa += d_la.T @ h; dba += d_la.sum(0)
                    dh = (d_la @ Wa) * (1 - h ** 2)
                    dWi += dh.T @ SUBJ[bb, tt]
                    if rg:
                        dWr += dh.T @ st_in
                        d_sa_next = (dh @ Wr)[:, :h.shape[1]] @ emb.T
                    else:
                        d_sa_next = np.zeros((B, K), np.float32)
                Wr -= lr * dWr; Wi -= lr * dWi; Wa -= lr * dWa; ba -= lr * dba; We -= lr * dWe; be -= lr * dbe

    def rollout(split):
        S, SI, O, E, OP, Ls, T = task[split]; B = len(Ls); Lm = int(Ls.max())
        sa = np.zeros((B, K), np.float32); sa[:, ident] = 1.0
        sp = np.zeros((B, K), np.float32); sp[:, ident] = 1.0
        fsa = sa.copy()
        for t in range(Lm):
            act = (Ls > t)
            st_in = np.concatenate([sa @ emb, sp @ emb], axis=1)
            h = np.tanh(rg * (st_in @ Wr.T) + S[:, t] @ Wi.T)
            sa_new = _sm(h @ Wa.T + ba)
            sa = np.where(act[:, None], sa_new, sa)
            spn = np.zeros((B, K), np.float32); spn[np.arange(B), O[:, t]] = 1.0
            sp = np.where(act[:, None], spn, sp)
            last = (Ls == (t + 1)); fsa = np.where(last[:, None], sa, fsa)
        return fsa, T[np.arange(B), Ls - 1]

    return rollout


def fair_reservoir(task, seed=42, n_res=512, rho=0.9, ridge=1e-3):
    """A FAIR echo-state-network floor (the skeptic's fix): a LARGE random recurrent hidden state over the same inputs
    (subject code + onehot previous object) with a trained ridge read-out. This -- not the degenerate K-dim random-init
    slot -- is the honest 'architecture without learned delta' baseline."""
    K, n_pool, ident = task["K"], task["n_pool"], task["ident"]
    rng = np.random.RandomState(seed + 77)
    W = rng.randn(n_res, n_res) / np.sqrt(n_res)
    W *= rho / max(np.abs(np.linalg.eigvals(W)).max(), 1e-9)
    Win = (rng.randn(n_res, n_pool + K) * 0.5).astype(np.float32)
    W = W.astype(np.float32)

    def states(split):
        S, SI, O, E, OP, Ls, T = task[split]; B = len(Ls); Lm = int(Ls.max())
        h = np.zeros((B, n_res), np.float32)
        prev = np.zeros((B, K), np.float32); prev[:, ident] = 1.0
        fh = h.copy()
        for t in range(Lm):
            act = (Ls > t)
            u = np.concatenate([S[:, t], prev], axis=1)
            hn = np.tanh(h @ W.T + u @ Win.T)
            h = np.where(act[:, None], hn, h)
            pn = np.zeros((B, K), np.float32); pn[np.arange(B), O[:, t]] = 1.0
            prev = np.where(act[:, None], pn, prev)
            last = (Ls == (t + 1)); fh = np.where(last[:, None], h, fh)
        return fh, T[np.arange(B), Ls - 1]

    Htr, ytr = states("train"); Hte, yte = states("test_deeper")
    Y = np.eye(K, dtype=np.float32)[ytr]
    Wout = np.linalg.solve(Htr.T @ Htr + ridge * np.eye(n_res, dtype=np.float32), Htr.T @ Y)
    return (Hte @ Wout).argmax(1), yte


def last_named_subject_floor(task, split):
    """The HONEST LABEL-FREE floor: latch the most recent INTRODUCE subject and ignore coref/promote. Reads only the
    observable subject stream, uses zero labels. Structurally MISSES promote-binding (the agent becomes the observed
    patient) -- which is exactly the model's genuine edge."""
    S, SI, O, E, OP, Ls, T = task[split]; B = len(Ls)
    pred = np.zeros(B, np.int64)
    for n in range(B):
        cur = task["ident"]
        for t in range(int(Ls[n])):
            if OP[n, t] == INTRO:
                cur = int(SI[n, t])
        pred[n] = cur
    return pred, T[np.arange(B), Ls - 1]


def final_depth_and_promote(task, split):
    """Per item: the number of TRAILING coref clauses at the end (how DEEP the final agent's persistence is), and
    whether the final agent was set by a PROMOTE (bound from the observed patient)."""
    S, SI, O, E, OP, Ls, T = task[split]; B = len(Ls)
    depth = np.zeros(B, np.int64); promo = np.zeros(B, bool)
    for n in range(B):
        Ln = int(Ls[n]); d = 0; t = Ln - 1
        while t >= 0 and OP[n, t] == COREF:
            d += 1; t -= 1
        depth[n] = d; promo[n] = (t >= 0 and OP[n, t] == PROMOTE)
    return depth, promo


def linear_probe(train_X, train_y, test_X, test_y, K, epochs=300, lr=0.5):
    """Frozen-state linear probe: read the AGENT identity out of the unsupervised slot. Labels used ONLY here."""
    rng = np.random.RandomState(0)
    W = (rng.randn(K, train_X.shape[1]) * 0.1).astype(np.float32); b = np.zeros(K, np.float32)
    eye = np.eye(K, dtype=np.float32); n = len(train_y)
    for _ in range(epochs):
        s = _sm(train_X @ W.T + b); d = (s - eye[train_y]) / n
        W -= lr * (d.T @ train_X); b -= lr * d.sum(0)
    return (test_X @ W.T + b).argmax(1)


def run_seed(seed, K, n_hid, epochs, theta_peak, p_coref):
    task = make_selfsup_event_task(seed, K=K, theta_peak=theta_peak, p_coref=p_coref)
    depth, promo = final_depth_and_promote(task, "test_deeper")
    deep = depth >= 3                                                   # the genuinely coref-DEEP subset

    def probe_of(**kw):
        roll = train_selfsup(task, seed=seed, n_hid=n_hid, **kw)
        trX, trY = roll("train"); teX, teY = roll("test_deeper")
        return linear_probe(trX, trY, teX, teY, K), teY

    p_ss, y = probe_of(epochs=epochs)
    p_re, _ = probe_of(epochs=epochs, random_emit=True)                 # emission-severed (== untrained arm; ONE axis)
    p_nr, _ = probe_of(epochs=epochs, no_recurrence=True)
    p_res, _ = fair_reservoir(task, seed=seed)                          # the FAIR architecture floor
    p_lns, _ = last_named_subject_floor(task, "test_deeper")            # the HONEST label-free floor
    _, _, O, _, _, Ls, _ = task["test_deeper"]
    p_rec = O[np.arange(len(Ls)), Ls - 1]                               # the weak recency strawman (reference only)

    def acc(p, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        return round(float((p[m] == y[m]).mean()), 3) if m.sum() else float("nan")

    return {"seed": seed, "K": K, "emission_purity": task["emission_purity"],
            "SELFSUP": acc(p_ss), "SELFSUP_deep": acc(p_ss, deep), "SELFSUP_promote": acc(p_ss, promo),
            "fair_reservoir": acc(p_res), "fair_reservoir_deep": acc(p_res, deep),
            "last_named_subject": acc(p_lns), "last_named_subject_promote": acc(p_lns, promo),
            "emission_severed": acc(p_re), "no_recurrence": acc(p_nr), "recency_weak_ref": acc(p_rec),
            "frac_deep": round(float(deep.mean()), 3), "frac_promote": round(float(promo.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--theta-peak", type=float, default=3.0)
    ap.add_argument("--p-coref", type=float, default=0.5)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT SELF-SUPERVISED] K={a.K} | learn the event delta from the AGENT-EMISSION prediction ALONE -- NO (agent,patient) state label anywhere; probe the frozen slot for the deep agent", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs, a.theta_peak, a.p_coref); rows.append(r)
        print(f"  [seed {s}] SELF-SUP={r['SELFSUP']} (deep>=3: {r['SELFSUP_deep']} | promote: {r['SELFSUP_promote']}) || "
              f"honest floor last-named-subj={r['last_named_subject']} (promote: {r['last_named_subject_promote']}) | "
              f"fair-reservoir={r['fair_reservoir']} (deep: {r['fair_reservoir_deep']}) || emission-severed={r['emission_severed']} | "
              f"no-recurrence={r['no_recurrence']} | [weak-ref recency={r['recency_weak_ref']}] | purity={r['emission_purity']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        ss, ssd, ssp = _m("SELFSUP"), _m("SELFSUP_deep"), _m("SELFSUP_promote")
        lns, lnsp = _m("last_named_subject"), _m("last_named_subject_promote")
        res, resd = _m("fair_reservoir"), _m("fair_reservoir_deep")
        sev, nr = _m("emission_severed"), _m("no_recurrence")
        chance = 1.0 / a.K
        # GATE (hardened by 3 adversarial skeptics): beat the HONEST label-free floor, not the weak recency strawman;
        # and win DECISIVELY on the coref-DEEP subset where the fair reservoir collapses. Emission-severed +
        # no-recurrence must both collapse. (emission-severed and the old `untrained` arm are ONE axis, not two.)
        go = (ss > 0.75) and (ss - lns > 0.25) and (ssd > 0.65) and (ssd - resd > 0.3) and (ss - sev > 0.3) and (ss - nr > 0.3)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}, emission purity {_m('emission_purity'):.3f}, frac deep>=3 {_m('frac_deep'):.3f}, frac promote {_m('frac_promote'):.3f}):", flush=True)
        print(f"    SELF-SUP={ss:.3f}  (coref-DEEP>=3: {ssd:.3f} | promote-bound: {ssp:.3f})", flush=True)
        print(f"    honest label-free floor `last-named-subject`={lns:.3f} (promote-bound: {lnsp:.3f} <- it structurally MISSES promote-binding)", flush=True)
        print(f"    FAIR reservoir (ESN+ridge)={res:.3f}  (coref-DEEP>=3: {resd:.3f} <- collapses exactly where the claim lives)", flush=True)
        print(f"    emission-severed={sev:.3f} | no-recurrence={nr:.3f} | [weak-ref recency={_m('recency_weak_ref'):.3f}]", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the event transition delta is learned FULLY SELF-SUPERVISED (agent-emission prediction ALONE, NO state label anywhere): the frozen K-way slot decodes the running agent at '+format(ss,'.2f')+' on held-out-DEEPER, and on the genuinely coref-DEEP subset (>=3 trailing corefs) it holds '+format(ssd,'.2f')+' while a FAIR echo-state reservoir COLLAPSES to '+format(resd,'.2f')+'; it beats the HONEST label-free floor last-named-subject ('+format(lns,'.2f')+') with the edge concentrated in PROMOTE-BINDING (self-sup '+format(ssp,'.2f')+' vs floor '+format(lnsp,'.2f')+', which structurally misses it); severing the agent->emission link ('+format(sev,'.2f')+') and removing recurrence ('+format(nr,'.2f')+') both collapse -> the running who-did-what-to-whom MEANING EMERGES from predicting what the brain hears, no host label = the last host supervision in the event composition is REMOVED' if go else 'the self-supervised delta did not clearly beat its HARDENED controls (read SELF-SUP vs last-named-subject + the deep-subset contrast vs the fair reservoir)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
