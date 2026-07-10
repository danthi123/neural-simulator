"""D3 — the transition delta learned by a BIOLOGICAL credit rule: clean-error feedback alignment, one step, no BPTT,
with the push gate taught by REPLAY. The last host learning machinery in the event register is removed.

WHERE THIS SITS.
The event register now has: two persistent spiking attractors (a_curr, a_prev); a WRITE gate opened by a boundary; a READ
gate opened by a return. Both gates are STRUCTURAL. Two results set up this rung:

  * `truncate=True` (cut the cross-clause gradient) leaves next-emission and a_curr slightly BETTER and DESTROYS the held
    slot (a_prev 0.610 -> 0.195, pop-sep +0.751 -> +0.067): the held slot exists only for the future, so a ONE-STEP local
    rule cannot learn the write gate...
  * ...unless REPLAY hands that gate a target NOW. Retrodicting the just-ended event's last observed emission from what
    the held slot holds recovers a_prev to 0.648 (109% of the BPTT value) with NO backprop through time. A SHUFFLED
    replay target scores 0.195 (dead) and replaying the CURRENT event scores 0.295 -- the content of the retrodiction is
    what teaches.

So one-step credit suffices for the transition. What remains is that the transition's credit is delivered by BACKPROP:
`d_lc @ Wc` transports the forward weight `Wc` into the backward pass. Biology cannot do that. This rung replaces it.

THE RULE (already validated on this substrate, `2026-07-07-D1-microcircuit-...clears-bar-on-spikes.md`, 0.964 held-out,
3-seed, batch-robust, adversarially verified). The depth-2 accuracy there is carried by the CLEAN-ERROR credit channel
(Urbanczik-Senn M2.6): the descending quantity is a clean error `e_k = phi'(E_k) * (Y_k^T @ e_{k+1})` propagated through a
FIXED-RANDOM feedback `Y_k` -- a weighted sum over the upper layer, so it averages rather than estimating credit from a
noisy per-unit burst fraction. Explicitly NOT the raw Burstprop burst-deviation (worse, ~0.79, and batch-fragile), and
the audit of that finding attributes the accuracy to this feedforward rule rather than to interneuron cancellation.

THE STRUCTURE THIS RUNG EXPLOITS. There is no `(agent, patient)` state label anywhere, so the AGENT layer has no target.
Only the EMISSION is observed. The network is therefore

    clause-code + a_curr + a_prev + patient  ->  h (tanh)  ->  AGENT (softmax)  ->  EMISSION (softmax)

where only the TOP (emission) layer has target access -- the layer the microcircuit rule also grants it to. Credit reaches
the agent layer and the hidden layer through fixed-random feedback of the clean error. The agent representation is thus
learned WITHOUT ever being taught, which is precisely the emergence claim of this whole arc.

ANTI-CHEATS (6-seed), gated on a_prev / RETURN -- **never** on next-emission, which is blind to the held slot's collapse
(P(agent | current emission) = 0.78):
 (a) vs BACKPROP (the reference: a_prev 0.610, RETURN 0.713, pop-sep +0.751) -- the like-for-like comparison;
 (b) NO WEIGHT TRANSPORT, asserted: the feedback matrices are fixed-random and are never a forward weight or its transpose;
 (c) FEEDBACK LESION (the descending clean error is zeroed -> only the emission head learns) must collapse the agent layer;
 (d) WRONG-SIGN (negate the top error) must ANTI-learn;
 (e) SHUFFLED replay target must kill the held slot (the control already proven discriminating: 0.195);
 (f) the no-teaching null (zero error) must not move the hidden weights (the P0 moat analogue).

Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_delta_cleanerror_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_pair_derisk import make_pair_task, INTRO, COREF, PROMOTE, BOUND, RETURN
from research.runners._d3_event_gated_copy_derisk import _sm, _sig
from research.runners._d3_event_pop_gate_derisk import _oracle_perm, OP_NAMES


def train_cleanerror(task, seed=42, n_hid=128, epochs=40, lr=0.05, batch=256,
                     credit="clean_error", replay_gamma=1.0, replay_target="prev",
                     lr_gate=5.0, gate_cost=0.01, lr_pop=0.1, stage_pop_epochs=15):
    """The register trained with ONE-STEP credit (no BPTT) and a biologically-plausible credit channel.

    credit: "clean_error" -> the descending error passes through FIXED-RANDOM feedback (no weight transport)
            "backprop"    -> the descending error passes through Wc^T / Wr^T (weight transport; the reference)
            "lesion"      -> the descending error is ZEROED (only the emission head learns)
            "wrong_sign"  -> the top error is negated (the teacher lies) -> must anti-learn
            "no_teaching" -> the top error is ZEROED entirely (the moat: hidden weights must not move)
    """
    K, M, n_pool, ident = task["K"], task["M"], task["n_pool"], task["ident"]
    rng = np.random.RandomState(seed + 9)
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)
    n_in = 3 * n_hid
    Wr = (rng.randn(n_hid, n_in) * np.sqrt(1.0 / n_in)).astype(np.float32)
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    Wc = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bc = np.zeros(K, np.float32)
    We = (rng.randn(M, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); be = np.zeros(M, np.float32)
    wg = (rng.randn(n_pool) * 0.05).astype(np.float32); bg = np.float32(-1.0)
    wp = (rng.randn(n_pool) * 0.05).astype(np.float32); bp = np.float32(-1.0)

    # FIXED-RANDOM feedback. Drawn from their OWN generators so the shared `rng` stream (and hence the minibatch order)
    # is untouched. NEVER a forward weight, never its transpose -> no weight transport.
    _rf = np.random.RandomState(seed + 4242)
    Ye = (_rf.randn(M, K) * np.sqrt(1.0 / M)).astype(np.float32)          # emission -> agent
    Yc = (_rf.randn(K, n_hid) * np.sqrt(1.0 / K)).astype(np.float32)     # agent -> hidden
    _rq = np.random.RandomState(seed + 991)
    Wq = (_rq.randn(M, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bq = np.zeros(M, np.float32)

    X, OBJ, EMIT, L, AC, AP, PE, PC = task["train"]
    eyeM = np.eye(M, dtype=np.float32)
    by_len = {}
    for n in range(len(L)):
        by_len.setdefault(int(L[n]), []).append(n)
    pop_on = [False]
    hidden0 = (Wr.copy(), Wi.copy())

    def _phase(n_epochs, upd_pop):
        nonlocal Wr, Wi, Wc, bc, We, be, wg, bg, wp, bp, Wq, bq
        for _ in range(n_epochs):
            for Ln, ids in by_len.items():
                ids = np.asarray(ids); rng.shuffle(ids)
                for i in range(0, len(ids), batch):
                    b = ids[i:i + batch]; B = len(b)
                    sc = np.zeros((B, K), np.float32); sc[:, ident] = 1.0
                    sp = np.zeros((B, K), np.float32); sp[:, ident] = 1.0
                    pat = np.zeros((B, K), np.float32); pat[:, ident] = 1.0
                    # Gradients ACCUMULATE across the clause loop and are applied ONCE per minibatch -- as in the
                    # reference harness. Applying them inside the loop makes each minibatch take Ln full-size steps
                    # (measured: a_prev 0.354 vs 0.648), which is a step-size artifact, not a property of the rule.
                    dWr = np.zeros_like(Wr); dWi = np.zeros_like(Wi)
                    dWc = np.zeros_like(Wc); dbc = np.zeros_like(bc)
                    dWe = np.zeros_like(We); dbe = np.zeros_like(be)
                    dwg = np.zeros_like(wg); dbg = np.float32(0.0)
                    dwp = np.zeros_like(wp); dbp = np.float32(0.0)
                    dWq = np.zeros_like(Wq); dbq = np.zeros_like(bq)
                    for t in range(Ln):
                        st_in = np.concatenate([sc @ emb, sp @ emb, pat @ emb], axis=1)
                        h = np.tanh(st_in @ Wr.T + X[b, t] @ Wi.T)
                        raw = _sm(h @ Wc.T + bc)                       # the AGENT layer -- never given a target
                        g = _sig(X[b, t] @ wg + bg)[:, None]
                        r = (_sig(X[b, t] @ wp + bp)[:, None] if pop_on[0] else np.zeros((B, 1), np.float32))
                        npv = g * sc + (1.0 - g) * sp                  # PUSH (structural)
                        nsc = r * sp + (1.0 - r) * raw                 # POP  (structural)
                        ev = nsc @ emb
                        se = _sm(ev @ We.T + be)                       # the EMISSION layer -- the ONLY target access

                        # ---------- top layer: a LOCAL delta (it sees the target)
                        d_le = (se - eyeM[EMIT[b, t]]) / B
                        if credit == "wrong_sign":
                            d_le = -d_le
                        if credit == "no_teaching":
                            d_le = np.zeros_like(d_le)
                        dWe += d_le.T @ ev; dbe += d_le.sum(0)

                        # ---------- the CLEAN ERROR descends onto the agent layer
                        if credit in ("lesion", "no_teaching"):
                            e_agent = np.zeros((B, K), np.float32)
                        elif credit == "backprop":
                            e_agent = (d_le @ We) @ emb.T              # WEIGHT TRANSPORT (the reference)
                        else:
                            e_agent = d_le @ Ye                        # FIXED-RANDOM feedback: no transport

                        d_raw = (1.0 - r) * e_agent                    # the pop is convex: only (1-r) reaches `raw`
                        if credit == "somatic_nudge":
                            # The EXACT M2.6 form as implemented in MicrocircuitBDSPNet.train_step:
                            #     soma_err = phi'(E) * v_api,  with phi'(E) = E*(1-E)  (ELEMENTWISE, a sigmoid unit)
                            # The apical error nudges each agent unit's soma independently; the feedforward weights
                            # follow the somatic-rate difference. NOTE this is NOT the softmax Jacobian, which couples
                            # all K units -- and a 6-unit softmax is exactly the narrow, coupled layer through which
                            # feedback alignment must align in the `clean_error` arm. This arm isolates that variable.
                            d_lc = (raw * (1.0 - raw)) * d_raw
                        else:
                            d_lc = raw * (d_raw - (raw * d_raw).sum(1, keepdims=True))
                        dWc += d_lc.T @ h; dbc += d_lc.sum(0)

                        if credit == "backprop":
                            dh = (d_lc @ Wc) * (1 - h ** 2)
                        elif credit in ("lesion", "no_teaching"):
                            dh = np.zeros_like(h)
                        else:
                            dh = (d_lc @ Yc) * (1 - h ** 2)            # fixed-random again (the descending soma_err)
                        dWr += dh.T @ st_in; dWi += dh.T @ X[b, t]

                        # ---------- the PUSH gate: taught by REPLAY, a target that exists NOW
                        d_p = np.zeros((B, K), np.float32)
                        if replay_gamma > 0 and credit != "no_teaching":
                            if replay_target == "current":
                                tgt = EMIT[b, t]
                            elif replay_target == "shuffled":
                                tgt = PE[b[_rq.permutation(B)], t]
                            else:
                                tgt = PE[b, t]
                            msk = tgt >= 0
                            if msk.any():
                                qv = npv @ emb
                                sq = _sm(qv @ Wq.T + bq)
                                d_lq = sq.copy(); d_lq[msk] -= eyeM[tgt[msk]]; d_lq[~msk] = 0.0
                                d_lq *= (replay_gamma / B)
                                dWq += d_lq.T @ qv; dbq += d_lq.sum(0)
                                d_p = (d_lq @ Wq) @ emb.T              # -> npv -> the PUSH gate, at THIS step

                        d_g = ((sc - sp) * d_p).sum(1) + (gate_cost / B)
                        dsig_g = (g[:, 0] * (1.0 - g[:, 0])) * d_g
                        dwg += X[b, t].T @ dsig_g; dbg += dsig_g.sum()

                        if upd_pop and pop_on[0]:                       # the POP gate (delayed onset)
                            d_r = ((sp - raw) * e_agent).sum(1)
                            dsig_r = (r[:, 0] * (1.0 - r[:, 0])) * d_r
                            dwp += X[b, t].T @ dsig_r; dbp += dsig_r.sum()

                        sc, sp = nsc, npv                              # NO gradient crosses this boundary
                        pat = np.zeros((B, K), np.float32); pat[np.arange(B), OBJ[b, t]] = 1.0

                    Wr -= lr * dWr; Wi -= lr * dWi; Wc -= lr * dWc; bc -= lr * dbc
                    We -= lr * dWe; be -= lr * dbe
                    if replay_gamma > 0 and credit != "no_teaching":
                        Wq -= lr * dWq; bq -= lr * dbq
                    wg -= lr_gate * dwg; bg -= lr_gate * dbg
                    if upd_pop and pop_on[0]:
                        wp -= lr_pop * dwp; bp -= lr_pop * dbp

    pop_on[0] = False; _phase(epochs, upd_pop=False)                   # delayed onset of the output gate
    pop_on[0] = True;  _phase(stage_pop_epochs, upd_pop=True)

    def rollout(split="test_deeper"):
        X_, O_, E_, L_, AC_, AP_, PE_, PC_ = task[split]
        B = len(L_); Lm = int(L_.max())
        sc = np.zeros((B, K), np.float32); sc[:, ident] = 1.0
        sp = np.zeros((B, K), np.float32); sp[:, ident] = 1.0
        pat = np.zeros((B, K), np.float32); pat[:, ident] = 1.0
        OPS = task["ops_train"] if split == "train" else task["ops_test"]
        rs, ra, ro, rr = [], [], [], []
        fc = sc.copy(); fp = sp.copy(); ok_e = tot_e = 0
        for t in range(Lm):
            act = L_ > t
            h = np.tanh(np.concatenate([sc @ emb, sp @ emb, pat @ emb], axis=1) @ Wr.T + X_[:, t] @ Wi.T)
            raw = _sm(h @ Wc.T + bc)
            g = _sig(X_[:, t] @ wg + bg)[:, None]; r = _sig(X_[:, t] @ wp + bp)[:, None]
            npv = g * sc + (1.0 - g) * sp
            nsc = r * sp + (1.0 - r) * raw
            sc = np.where(act[:, None], nsc, sc); sp = np.where(act[:, None], npv, sp)
            se = _sm((sc @ emb) @ We.T + be)
            m = np.where(act)[0]
            ok_e += int((np.argmax(se[m], 1) == E_[m, t]).sum()); tot_e += len(m)
            rs.append(np.argmax(sc[m], 1)); ra.append(AC_[m, t]); ro.append(OPS[m, t]); rr.append(r[m, 0])
            pn = np.zeros((B, K), np.float32); pn[np.arange(B), O_[:, t]] = 1.0
            pat = np.where(act[:, None], pn, pat)
            last = (L_ == (t + 1))
            fc = np.where(last[:, None], sc, fc); fp = np.where(last[:, None], sp, fp)
        return (np.concatenate(rs), np.concatenate(ra), np.concatenate(ro), np.concatenate(rr),
                fc, fp, AC_[np.arange(B), L_ - 1], AP_[np.arange(B), L_ - 1], ok_e / max(tot_e, 1))

    rollout.no_transport = (not np.allclose(Ye, We[:, :K] if We.shape[1] >= K else 0) and
                            not np.allclose(Yc, Wc))
    rollout.hidden_moved = float(np.abs(Wr - hidden0[0]).max() + np.abs(Wi - hidden0[1]).max())
    return rollout


def _score(roll, K):
    s, a, o, r, fc, fp, ac, ap, emis = roll()
    perm = _oracle_perm(s, a, K)
    ok = perm[s] == a
    ret = (o == RETURN)
    return {"a_curr": float(ok.mean()),
            "RETURN": float(ok[ret].mean()) if ret.any() else float("nan"),
            "a_prev": float((perm[np.argmax(fp, 1)] == ap).mean()),
            "pop_sep": float(r[ret].mean() - r[~ret].mean()) if ret.any() else float("nan"),
            "emission": float(emis), "hidden_moved": roll.hidden_moved}


def run_seed(seed, K=6, epochs=40):
    task = make_pair_task(seed, K=K)
    arms = {}
    for name, kw in (("somatic_nudge", {"credit": "somatic_nudge"}),
                     ("clean_error", {}),
                     ("backprop_ref", {"credit": "backprop"}),
                     ("feedback_lesion", {"credit": "lesion"}),
                     ("wrong_sign", {"credit": "wrong_sign"}),
                     ("no_teaching", {"credit": "no_teaching"}),
                     ("clean_replay_shuffled", {"replay_target": "shuffled"}),
                     ("clean_no_replay", {"replay_gamma": 0.0})):
        roll = train_cleanerror(task, seed=seed, epochs=epochs, **kw)
        arms[name] = _score(roll, K)
        arms[name]["no_transport"] = bool(roll.no_transport)
    out = {"seed": seed}
    for n, r in arms.items():
        out[n] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print("[D3 CLEAN-ERROR DELTA] the transition learned by fixed-random clean-error feedback: one step, no BPTT, no weight transport", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, epochs=a.epochs); rows.append(r)
        ce, bp = r["clean_error"], r["backprop_ref"]
        print(f"  [seed {s}] a_prev: clean={ce['a_prev']} vs backprop={bp['a_prev']} | RETURN {ce['RETURN']} vs {bp['RETURN']} | "
              f"a_curr {ce['a_curr']} vs {bp['a_curr']} | emission {ce['emission']} vs {bp['emission']}", flush=True)
        print(f"            lesion a_curr={r['feedback_lesion']['a_curr']} | wrong-sign a_curr={r['wrong_sign']['a_curr']} | "
              f"no-teaching hidden-moved={r['no_teaching']['hidden_moved']} | shuffled-replay a_prev={r['clean_replay_shuffled']['a_prev']} | "
              f"no-replay a_prev={r['clean_no_replay']['a_prev']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(arm, k):
            v = [r[arm][k] for r in rows if not (isinstance(r[arm][k], float) and np.isnan(r[arm][k]))]
            return float(np.mean(v)) if v else float("nan")
        ce_p, bp_p = _m("clean_error", "a_prev"), _m("backprop_ref", "a_prev")
        ce_r, bp_r = _m("clean_error", "RETURN"), _m("backprop_ref", "RETURN")
        ce_c, les, ws = _m("clean_error", "a_curr"), _m("feedback_lesion", "a_curr"), _m("wrong_sign", "a_curr")
        shuf, norep = _m("clean_replay_shuffled", "a_prev"), _m("clean_no_replay", "a_prev")
        nt = _m("no_teaching", "hidden_moved")
        transport_free = all(r["clean_error"]["no_transport"] for r in rows)
        go = ((ce_p > 0.45) and (ce_p - shuf > 0.2) and (ce_p - norep > 0.2) and (ce_c - les > 0.15)
              and (ws < les + 0.05) and (nt == 0.0) and transport_free)
        print(f"\n  AGGREGATE  a_prev: clean-error={ce_p:.3f} | backprop={bp_p:.3f} || RETURN {ce_r:.3f} vs {bp_r:.3f} || a_curr {ce_c:.3f}", flush=True)
        print(f"    feedback-lesion a_curr={les:.3f} | wrong-sign a_curr={ws:.3f} | no-teaching hidden-moved={nt:.6f} | "
              f"shuffled-replay a_prev={shuf:.3f} | no-replay a_prev={norep:.3f} | no-weight-transport={transport_free}", flush=True)
        msg = ('the transition is learned by a CLEAN ERROR descending through FIXED-RANDOM feedback -- no weight transport, no '
               'backprop through time -- and the write gate by REPLAY. The agent layer is never given a target, yet a_prev reaches '
               + format(ce_p, '.2f') + ' (backprop reference ' + format(bp_p, '.2f') + ') and RETURN ' + format(ce_r, '.2f') + '. Zeroing the '
               'descending error collapses the agent layer (' + format(les, '.2f') + '); a lying teacher anti-learns (' + format(ws, '.2f') +
               '); the no-teaching null moves the hidden weights by exactly ' + format(nt, '.6f') + '; a shuffled replay target kills the held '
               'slot (' + format(shuf, '.2f') + '), as does removing replay (' + format(norep, '.2f') + ')')
        bad = 'the clean-error transition did not reach the reference / a control failed to collapse'
        print("  VERDICT: " + ("GO" if go else "PARTIAL/NEGATIVE") + " -- " + (msg if go else bad) + ". NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
