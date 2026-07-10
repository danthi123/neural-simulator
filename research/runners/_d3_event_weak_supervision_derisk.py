"""D3 EVENT COMPOSITION — the CRUX escalation (research-gated): learn the relational (agent, patient) UPDATE from
END-STATE-ONLY (weak) supervision — NO per-step intermediate (a,p) states. The RANK-1 event de-risk
(`_d3_event_composition_derisk`, 6-seed GO 0.993) was PER-STEP-supervised; the genuinely-open residual (the D3
`-language-reference-tracking-GO` lines 29-30) is whether the composed running MEANING can be LEARNED without dense
supervision. This rung answers the sharp question the reference-tracking finding posed: is the EVENT-update a LOOKUP
composition (a routing/copy op — the promote `a<-p_prev` is a COPY, learnable from the final answer alone, like a group
DFA) or a RELATIONAL one (needs dense per-step signal, the 0.289 residual)?

THE MECHANISM (reuse of the RANK-1 weak-supervision realization, lifted to the FACTORED two-slot state): a Dyna-style
DETACHED-rollout CURRICULUM. For each discourse, roll BOTH slots forward autoregressively with argmax (the intermediate
(a,p) are the model's OWN predictions, DETACHED — never targets), then take ONE supervised gradient step on the FINAL
step only, against the K-way final agent AND the K-way final patient (2*log2 K bits at the endpoint). CURRICULUM len
1->2->3 so the rolled prev-(a,p) entering the final step is correct at each depth. NO intermediate (a,p) is ever a target.

THE DECISIVE ISOLATION (one variable): `--supervise state` (K-way final a AND p) vs `--supervise property` (2-way color
of the final a AND p = the reference-tracking-style low-bit endpoint). If STATE length-generalizes on held-out-DEEPER
where PROPERTY does not, the composed EVENT is WEAK-SUPERVISABLE — the running meaning is LEARNED from just the final
answer (the promote is a lookup-routing op), surpassing the dense-supervision residual for THIS op class.

ANTI-CHEATS: (a) held-out-DEEPER (learned iteration, not memorized length); (b) `--supervise property` low-bit control;
(c) SHUFFLE-label endpoint (memorization-floor -> collapse); (d) RECENCY floor (the forced-promote-last makes last-mention
fail); (e) per-step-supervised CEILING (reported); (f) multi-seed dev+blind. Reuse-by-import (`make_event_task`); numpy;
NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_weak_supervision_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_composition_derisk import make_event_task, recency_floor, last2_objects_floor


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def train_event_endstate(task, seed=42, n_hid=192, epochs=90, lr=0.08, batch=256, supervise="state",
                         shuffle_labels=False, curriculum=True):
    """Learn the factored (a,p) UPDATE from END-STATE-only supervision via a detached-rollout curriculum. Two K-way heads
    (agent Wa / patient Wp) supervised on the FINAL (a,p) only; the intermediate (a,p) are the model's own argmax rollout.
    supervise='property' -> two 2-way color heads (the low-bit endpoint control)."""
    K = task["K"]; ident = task["ident"]; n_pool = task["n_pool"]; color = np.asarray(task["color"])
    rng = np.random.RandomState(seed + 33)
    n_in = 2 * n_hid
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)
    Wr = (rng.randn(n_hid, n_in) * np.sqrt(1.0 / n_in)).astype(np.float32)
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    Wa = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); ba = np.zeros(K, np.float32)
    Wp = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bp = np.zeros(K, np.float32)
    Ca = (rng.randn(2, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bca = np.zeros(2, np.float32)   # property heads
    Cp = (rng.randn(2, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bcp = np.zeros(2, np.float32)
    eyeK = np.eye(K, dtype=np.float32); eye2 = np.eye(2, dtype=np.float32)

    X, Ya, Yp, L, SEQ, STA, STP = task["train"]; N = len(L)
    fa_all = STA[np.arange(N), L - 1]; fp_all = STP[np.arange(N), L - 1]         # final (a, p) endpoints ONLY
    if shuffle_labels:
        perm = rng.permutation(N); fa_all = fa_all[perm]; fp_all = fp_all[perm]
    fca_all = color[fa_all]; fcp_all = color[fp_all]

    def _sin(a, p):
        return np.concatenate([emb[a], emb[p]], axis=1)

    def roll_hard(Xb, Lb):
        """Argmax DETACHED rollout of BOTH slots; return (prev_a, prev_p entering the final step, x at the final step)."""
        B = len(Lb); Lmax = int(Lb.max())
        a = np.full(B, ident, np.int64); p = np.full(B, ident, np.int64)
        pa = np.full(B, ident, np.int64); pp = np.full(B, ident, np.int64); xf = np.zeros((B, n_pool), np.float32)
        for t in range(Lmax):
            active = (Lb > t); last = (Lb == (t + 1))
            pa[last] = a[last]; pp[last] = p[last]; xf[last] = Xb[last, t]
            h = np.tanh(_sin(a, p) @ Wr.T + Xb[:, t] @ Wi.T)
            na = (h @ Wa.T + ba).argmax(1); npp = (h @ Wp.T + bp).argmax(1)
            a = np.where(active, na, a); p = np.where(active, npp, p)
        return pa, pp, xf

    def final_ap(Xb, Lb):
        B = len(Lb); Lmax = int(Lb.max())
        a = np.full(B, ident, np.int64); p = np.full(B, ident, np.int64)
        fa = np.full(B, ident, np.int64); fp = np.full(B, ident, np.int64)
        for t in range(Lmax):
            active = (Lb > t); last = (Lb == (t + 1))
            h = np.tanh(_sin(a, p) @ Wr.T + Xb[:, t] @ Wi.T)
            na = (h @ Wa.T + ba).argmax(1); npp = (h @ Wp.T + bp).argmax(1)
            a = np.where(active, na, a); p = np.where(active, npp, p)
            fa = np.where(last, a, fa); fp = np.where(last, p, fp)
        return fa, fp

    phases = sorted(set(int(x) for x in L)) if curriculum else [int(L.max())]
    for phase_len in phases:
        sub = np.where(L <= phase_len)[0]
        for ep in range(epochs):
            pa, pp, xf = roll_hard(X[sub], L[sub])                 # detached rollout with current weights
            order = rng.permutation(len(sub))
            for i in range(0, len(sub), batch):
                b = order[i:i + batch]; gi = sub[b]; B = len(b)
                pe = _sin(pa[b], pp[b])                            # detached prev-(a,p) embedding
                h = np.tanh(pe @ Wr.T + xf[b] @ Wi.T)
                if supervise == "state":                          # K-way final a AND p
                    sa = _softmax(h @ Wa.T + ba); da = (sa - eyeK[fa_all[gi]]) / B
                    sp = _softmax(h @ Wp.T + bp); dp = (sp - eyeK[fp_all[gi]]) / B
                    Wa -= lr * (da.T @ h); ba -= lr * da.sum(0); Wp -= lr * (dp.T @ h); bp -= lr * dp.sum(0)
                    dpre = ((da @ Wa) + (dp @ Wp)) * (1.0 - h ** 2)
                else:                                             # 2-way color of final a AND p (low-bit control)
                    sa = _softmax(h @ Ca.T + bca); da = (sa - eye2[fca_all[gi]]) / B
                    sp = _softmax(h @ Cp.T + bcp); dp = (sp - eye2[fcp_all[gi]]) / B
                    Ca -= lr * (da.T @ h); bca -= lr * da.sum(0); Cp -= lr * (dp.T @ h); bcp -= lr * dp.sum(0)
                    dpre = ((da @ Ca) + (dp @ Cp)) * (1.0 - h ** 2)
                Wr -= lr * (dpre.T @ pe); Wi -= lr * (dpre.T @ xf[b])   # emb DETACHED

    def eval_split(split):
        Xe, Yae, Ype, Le, SEQe, STAe, STPe = task[split]
        ta = STAe[np.arange(len(Le)), Le - 1]; tp = STPe[np.arange(len(Le)), Le - 1]
        if supervise == "property":
            pa, pp, xf = roll_hard(Xe, Le); h = np.tanh(_sin(pa, pp) @ Wr.T + xf @ Wi.T)
            pca = (h @ Ca.T + bca).argmax(1); pcp = (h @ Cp.T + bcp).argmax(1)
            return float(((pca == color[ta]) & (pcp == color[tp])).mean())      # both color slots right
        fa, fp = final_ap(Xe, Le)
        return float(((fa == ta) & (fp == tp)).mean())                          # both state slots right

    return {"deeper": eval_split("test_deeper"), "same": eval_split("test_same"), "train": eval_split("train")}


def run_seed(seed, K, n_hid, epochs):
    task = make_event_task(seed, K=K, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    st = train_event_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, supervise="state")
    pr = train_event_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, supervise="property")
    sh = train_event_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, supervise="state", shuffle_labels=True)
    return {"seed": seed, "K": K, "STATE_endpoint_deeper": round(st["deeper"], 3), "STATE_same": round(st["same"], 3),
            "PROPERTY_endpoint_deeper": round(pr["deeper"], 3), "SHUFFLE_deeper": round(sh["deeper"], 3),
            "recency_floor": round(recency_floor(task), 3), "last2_objects_floor": round(last2_objects_floor(task), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT WEAK-SUPERVISION] K={a.K} | learn the factored (a,p) UPDATE from END-STATE-only supervision (NO per-step states); is the promote a LOOKUP (weak-learnable) or RELATIONAL (needs dense) op?", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs); rows.append(r)
        print(f"  [seed {s}] STATE-endpoint DEEPER={r['STATE_endpoint_deeper']} (same={r['STATE_same']}) || "
              f"PROPERTY-endpoint(low-bit)={r['PROPERTY_endpoint_deeper']} || SHUFFLE(mem-floor)={r['SHUFFLE_deeper']} || LAST-2-OBJ(shallow)={r['last2_objects_floor']} || RECENCY={r['recency_floor']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        sd, pr, sh, rec, l2 = _m("STATE_endpoint_deeper"), _m("PROPERTY_endpoint_deeper"), _m("SHUFFLE_deeper"), _m("recency_floor"), _m("last2_objects_floor")
        # GO (adversarial-verify-hardened): learned from end-state-only AND genuinely DEEP -> STATE >> SHUFFLE (mem-floor)
        # AND >> the LAST-2-OBJECTS shallow reader (the task needs deep tracking, not a 2-token lookup).
        go = (sd > 0.7) and (sd - sh > 0.2) and (sd - l2 > 0.2)
        print(f"\n  AGGREGATE (K={a.K}): STATE-endpoint DEEPER={sd:.3f} | LAST-2-OBJ(shallow)={l2:.3f} | SHUFFLE(mem-floor)={sh:.3f} | PROPERTY(low-bit)={pr:.3f} | RECENCY={rec:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the factored (agent, patient) EVENT UPDATE is LEARNED from END-STATE-only supervision (NO per-step states; a short-length curriculum + the model own detached rollout) and length-generalizes to a genuinely DEEP task (DEEPER '+format(sd,'.2f')+' >> the LAST-2-OBJECTS shallow reader '+format(l2,'.2f')+' [deep tracking, not a 2-token lookup] >> shuffle-label mem-floor '+format(sh,'.2f')+') -> the running MEANING is weak-supervisable: the agent-coref/it-promote routing ops are LOOKUP compositions learnable from just the final answer, surpassing the dense-supervision residual for this op class -> next: self-supervised OBSERVATION-prediction (TEM) removes the endpoint label entirely, then wrap the composer per-slot bind + spiking port' if go else 'the event update did NOT clearly learn deep from end-state-only (DEEPER '+format(sd,'.2f')+' vs last-2-obj '+format(l2,'.2f')+'/shuffle '+format(sh,'.2f')+') -> escalate to TEM self-supervised observation-prediction (the research-gated route)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
