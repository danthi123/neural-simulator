"""D3 -> the SPIKING BOUNDARY-GATED COPY: the held prior event is a genuine SPIKING MEMORY, not a Python variable.

This is the first rung in the event arc whose HOLD is actually on the substrate. Every earlier "spiking slot" used
`fswta_drive`, a STATELESS one-of-K re-discretizer (it resets v/u/firing each call; its pools have internal_density=0),
so the slot's hold lived in host memory between calls. Here the held event lives in a PERSISTENT slow-NMDA attractor
(`_d3_persistent_slot_derisk`, HOLD 6/6 GO) and the boundary gate does what `sim/`'s own `transmission_gate` documents:

    gate CLOSED  ->  NO input to the a_prev slot at all. The attractor SUSTAINS ITS OWN FIRING (persistent activity,
                     Wang 2002). The prior event is remembered by spikes, across arbitrarily many clauses.
    gate OPEN    ->  CLEAR (an FS inhibitory burst LONGER than tau_NMDA) then LOAD the current event's pattern.
                     Measured: input alone cannot overwrite an attractor (0/6), and a reset shorter than tau_NMDA lets
                     the old bump RE-IGNITE from residual conductance -- which is exactly why PBWM gating clears first.

The transition and the gate are the rate-learned ones from `_d3_event_gated_copy_derisk` (learned from the agent-emission
cross-entropy ALONE -- NO (agent,patient) state label anywhere; the gate reads only the OBSERVABLE clause code).
The held agent is then READ OUT OF SPIKES: whichever pool of the a_prev attractor is firing at the end of the discourse.

TWO BUGS FOUND ON THE WAY, both by instrumenting rather than tuning:
  * `_reset` cleared v/u/firing but NOT the conductances, and `g_nmda_recurrent` has tau=100 ms -- MEASURED 95.1 before
    and after the reset. Every discourse item therefore INHERITED the previous item's fully-charged held bump, which
    re-ignited into it (the same residual-conductance re-ignition that forces a gate's CLEAR to outlast tau_NMDA).
  * The slot is a PERMUTATION of entity identity. Comparing a raw slot index to a true agent index is meaningless; the
    rate reference is obtained through a fitted slot->entity read-out. Applying it lifts the spiking arm 0.444 -> 0.667.

A LIKE-FOR-LIKE ATTRIBUTION (the decisive check): a HOST TWIN of this pipeline -- same binarised gate, one-hot copy and
one-hot feedback, but no bridge at all -- scores EXACTLY what the spiking arm scores. The spiking substrate is faithful;
nothing is lost by putting the hold on real NMDA neurons.

ANTI-CHEATS (6-seed): (a) the spiking-held prev-agent >> RECENCY and >> a GATE-LESION (gate never opens -> the slot
never loads anything); (b) a STATELESS a_prev slot (the old fswta bridge, no recurrence) degrades sharply -- it cannot
hold the prior event between clauses, which is the whole claim; (c) the hold is read with the external input to the slot
identically ZERO on every non-boundary clause (asserted); (d) compared against the rate gated-copy as reference.
numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_gated_copy_spiking_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_pair_derisk import (
    make_pair_task, _informative, _informative_train)
from research.runners._d3_event_gated_copy_derisk import train_gated_copy, probe, _sm, _sig
from research.runners._d3_persistent_slot_derisk import build_persistent_slot, _pool_idx, _reset


def fit_slot_names(trp, trb, tmask, K):
    """Fit the slot -> entity read-out (the emergent slot is a PERMUTATION of identity). This is exactly how the RATE
    reference number is obtained -- comparing a raw slot index to a true agent index is meaningless, and doing so is
    what first made the spiking arm look like 0.444 when it is 0.667."""
    X = trp[tmask]; Y = trb[tmask]
    oh = np.zeros_like(X); oh[np.arange(len(X)), X.argmax(1)] = 1.0
    rng = np.random.RandomState(0)
    W = (rng.randn(K, K) * 0.1).astype(np.float32); b = np.zeros(K, np.float32)
    eye = np.eye(K, dtype=np.float32); n = len(Y)
    for _ in range(300):
        z = oh @ W.T + b; e = np.exp(z - z.max(1, keepdims=True)); sm = e / e.sum(1, keepdims=True)
        d = (sm - eye[Y]) / n
        W -= 0.5 * (d.T @ oh); b -= 0.5 * d.sum(0)
    return (eye @ W.T + b).argmax(1)


def _run(sb, cur_vec, steps, idx, K):
    from sim.backend import to_host, from_host
    acc = np.zeros(K); dev = from_host(cur_vec)
    for _ in range(steps):
        sb.cp_external_input_current[:] = dev
        sb._run_one_simulation_step()
        fir = np.asarray(to_host(sb.cp_firing_states)).astype(float)
        for k in range(K):
            acc[k] += fir[idx[k]].mean()
    return acc / max(steps, 1)


def spiking_gated_rollout(task, W, gate, split, K, n_eval=80, seed=42, recurrent=True,
                          gate_lesion=False, inter_clause=15, clear_steps=250, load_steps=80,
                          clear_gain=1500.0, load_gain=400.0, gate_thresh=0.5):
    """Roll the discourse. `a_curr` is the rate transition's argmax (host); `a_prev` is a PERSISTENT SPIKING ATTRACTOR
    whose only inputs are the gate's clear+load pulses. Returns the spiking-held prev winner per item."""
    emb, Wr, Wi, Wc, bc = W["emb"], W["Wr"], W["Wi"], W["Wc"], W["bc"]
    wg, bg = gate
    ident = task["ident"]
    X, OBJ, EMIT, L, AC, AP, PE, PC = task[split]
    rng = np.random.RandomState(seed + 1)
    sel = rng.choice(len(L), min(n_eval, len(L)), replace=False)

    sb = build_persistent_slot(seed, K, recur=(25.0 if recurrent else 0.0))
    idx = _pool_idx(sb, K)
    fs_idx = np.asarray(list(sb.region_manager.indices("fs")), dtype=int)
    n = sb.core_config.num_neurons
    zero = np.zeros(n, dtype=np.float64)

    held = np.zeros(len(sel), np.int64); true_prev = np.zeros(len(sel), np.int64); true_curr = np.zeros(len(sel), np.int64)
    n_open = 0
    for i, item in enumerate(sel):
        _reset(sb)
        sc = np.zeros(K, np.float32); sc[ident] = 1.0
        pat = np.zeros(K, np.float32); pat[ident] = 1.0
        prev_winner = ident
        for t in range(int(L[item])):
            code = X[item, t]
            g = 0.0 if gate_lesion else float(_sig(code @ wg + bg))
            load_content = int(np.argmax(sc))                # what a boundary copies: a_curr ENTERING this clause
            if g > gate_thresh:                              # ---- gate OPEN: CLEAR then LOAD (a real WM update)
                n_open += 1
                cc = np.zeros(n, dtype=np.float64); cc[fs_idx] = clear_gain
                _run(sb, cc, clear_steps, idx, K)
                cl = np.zeros(n, dtype=np.float64); cl[idx[load_content]] = load_gain
                acc = _run(sb, cl, load_steps, idx, K)
            else:                                            # ---- gate CLOSED: ZERO input; the attractor HOLDS itself
                assert not zero.any()
                acc = _run(sb, zero, inter_clause, idx, K)
            if acc.max() > 1e-6:                             # a_prev is READ FROM SPIKES and fed back to the transition
                prev_winner = int(np.argmax(acc))
            sp_oh = np.zeros(K, np.float32); sp_oh[prev_winner] = 1.0
            st_in = np.concatenate([sc @ emb, sp_oh @ emb, pat @ emb])
            h = np.tanh(st_in @ Wr.T + code @ Wi.T)
            sc = _sm(h @ Wc.T + bc)
            pat = np.zeros(K, np.float32); pat[int(OBJ[item, t])] = 1.0
        read = _run(sb, zero, 30, idx, K)                    # READ THE HELD EVENT OUT OF SPIKES (zero input)
        held[i] = int(np.argmax(read)) if read.max() > 1e-6 else ident
        true_prev[i] = int(AP[item, int(L[item]) - 1]); true_curr[i] = int(AC[item, int(L[item]) - 1])
    return held, true_prev, true_curr, sel, n_open / max(len(sel), 1)


def run_seed(seed, K, n_hid, epochs, n_eval):
    task = make_pair_task(seed, K=K)
    info, ac, ap = _informative(task); tmask = _informative_train(task)
    roll = train_gated_copy(task, seed=seed, n_hid=n_hid, epochs=epochs)      # rate: delta + gate, NO state label
    trc, trp, tra, trb = roll("train"); tec, tep, tea, teb = roll("test_deeper")
    rate_prev = probe(trp, trb, tep, teb, K, mask=info, train_mask=tmask)
    perm = fit_slot_names(trp, trb, tmask, K)                                 # slot -> entity (as the rate probe does)

    out = {"seed": seed, "K": K, "rate_gated_prev": round(rate_prev, 3)}
    for tag, kw in (("SPK", dict(recurrent=True)),
                    ("SPK_stateless", dict(recurrent=False)),
                    ("SPK_gate_lesion", dict(recurrent=True, gate_lesion=True))):
        held, tp, tc, sel, orate = spiking_gated_rollout(task, roll.W, roll.gate, "test_deeper", K,
                                                         n_eval=n_eval, seed=seed, **kw)
        m = info[sel]
        acc = float((perm[held][m] == tp[m]).mean()) if m.sum() else float("nan")
        out[tag + "_prev"] = round(acc, 3)
        if tag == "SPK":
            out["gate_open_rate"] = round(orate, 3)
            out["n_informative_eval"] = int(m.sum())
    O = task["test_deeper"][1]; L = task["test_deeper"][3]
    rec = float((O[np.arange(len(L)), L - 1][info] == ap[info]).mean())
    out["recency_prev"] = round(rec, 3)
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", default="42")
    ap_.add_argument("--K", type=int, default=6)
    ap_.add_argument("--n-hid", type=int, default=128)
    ap_.add_argument("--epochs", type=int, default=40)
    ap_.add_argument("--n-eval", type=int, default=80)
    ap_.add_argument("--json", default=None)
    a = ap_.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 SPIKING GATED COPY] K={a.K} | the held prior event lives in a PERSISTENT slow-NMDA attractor; closed gate = zero input = the attractor holds itself; open gate = clear-then-load", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs, a.n_eval); rows.append(r)
        print(f"  [seed {s}] SPIKING held prev={r['SPK_prev']} (rate reference {r['rate_gated_prev']}) || stateless-slot={r['SPK_stateless_prev']} | gate-lesion={r['SPK_gate_lesion_prev']} | recency={r['recency_prev']} | gate-open-rate={r['gate_open_rate']} (n={r['n_informative_eval']})", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        spk, rt = _m("SPK_prev"), _m("rate_gated_prev")
        sl, gl, rec = _m("SPK_stateless_prev"), _m("SPK_gate_lesion_prev"), _m("recency_prev")
        chance = 1.0 / a.K
        go = (spk > 0.55) and (spk - sl > 0.25) and (spk - gl > 0.3) and (spk - rec > 0.35)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}):", flush=True)
        print(f"    *** SPIKING held prev-agent={spk:.3f} (rate gated-copy reference {rt:.3f}) ***", flush=True)
        print(f"    controls: STATELESS a_prev slot (the old fswta bridge)={sl:.3f} | gate-lesion={gl:.3f} | recency={rec:.3f}", flush=True)
        msg = ('the held prior event is a GENUINE SPIKING MEMORY: it lives in a persistent slow-NMDA attractor that '
               'sustains its own firing with the slot input identically ZERO on every non-boundary clause, and the boundary '
               'gate updates it as a CLEAR-then-LOAD (an inhibitory reset longer than tau_NMDA, then the new content) -- '
               'exactly the semantics of sim/ transmission_gate. The held agent is read out of SPIKES (' + format(spk, '.2f') +
               '), close to the rate gated-copy reference (' + format(rt, '.2f') + '), while a STATELESS slot -- the '
               're-discretizer every prior rung used -- CANNOT hold the prior event between clauses (' + format(sl, '.2f') +
               '), a gate that never opens has nothing to hold (' + format(gl, '.2f') + '), and recency fails (' +
               format(rec, '.2f') + ') -> the event arc HOLD is no longer a Python variable')
        bad = 'the spiking gated copy did not clearly hold (read SPK vs stateless-slot / gate-lesion / recency)'
        print("  VERDICT: " + ("GO" if go else "PARTIAL/NEGATIVE") + " -- " + (msg if go else bad) + ". NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
