"""D3 — the WEAK-SUPERVISION LEARNING with a SPIKING-FORWARD transition (fully-spiking-forward weak learning). RANK-1
(`_d3_weak_supervision_derisk`) learned delta from END-STATE-only supervision + curriculum through a RATE (tanh) hidden;
rung-2 (`_d3_spiking_transition_derisk`) showed a spiking LIF hidden REPRESENTS delta (surrogate grad, but on teacher-
forced triples). THIS composes them: the transition's hidden is a SPIKING LIF pool (rate-coded, T steps, surrogate
gradient), and it is trained from WEAK (end-state-only) supervision via the Dyna-style DETACHED-rollout CURRICULUM (roll
with the LIF-argmax, supervise ONLY the final K-way state). So the transition FORWARD is spiking THROUGHOUT the weak-
supervision learning (not just at execution) -> the delta is learned-on-a-spiking-forward from weak supervision, and
length-generalizes.

(Honest scope: the surrogate-gradient BACKWARD is still host BPTT -- a biologically-plausible LOCAL learning rule is the
separate deep wall, EMERGE-6..8's 5x-confirmed dead-end; this rung makes the FORWARD spiking during learning, matching
rung-2's "transition on spikes" sense.)

ANTI-CHEATS: (a) STATE (weak-sup, spiking-forward) held-out-DEEPER >> chance; (b) PROPERTY-endpoint (1-bit = rung-3)
control fails (the reframe); (c) SHUFFLE-label collapse (genuine learning); (d) multi-seed. Reuse-by-import
(`make_group_task` + `sim.surrogate_grad`); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_spiking_weak_learning_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, build_group


def lif_rate(F, W1, T=16, leak=0.9, thr=1.0, want_grad=False):
    """Rate-coded LIF hidden: static drive F@W1.T -> T hard-reset LIF steps -> rate = mean spikes (+ surrogate factor)."""
    from sim.surrogate_grad import atan_surrogate
    drive = np.asarray(F, dtype=np.float32) @ W1.T
    B, n_hid = drive.shape
    v = np.zeros((B, n_hid), np.float32); acc = np.zeros((B, n_hid), np.float32)
    s = np.zeros((B, n_hid), np.float32); sg = np.zeros((B, n_hid), np.float32)
    for _ in range(T):
        v = leak * v * (1.0 - s) + drive
        s = (v >= thr).astype(np.float32); acc += s
        if want_grad:
            sg += atan_surrogate(v - thr)
    return acc / T, (sg / T if want_grad else None)


def train_spiking_weak(task, seed=42, n_hid=192, T=16, epochs=80, lr=0.05, batch=128,
                       supervise="state", shuffle_labels=False, curriculum=True):
    K = task["K"]; ident = task["ident"]; color = np.asarray(task["color"]); n_pool = task["n_pool"]
    rng = np.random.RandomState(seed + 33)
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)          # fixed attractor prototypes
    n_in = n_hid + n_pool
    W1 = (rng.randn(n_hid, n_in) * np.sqrt(2.0 / n_in)).astype(np.float32)
    W2 = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); b2 = np.zeros(K, dtype=np.float32)
    Wp = (rng.randn(2, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bp = np.zeros(2, dtype=np.float32)
    eyeK = np.eye(K, dtype=np.float32); eye2 = np.eye(2, dtype=np.float32)

    Xtr, ytr, Ltr, _, Str = task["train"]; N = len(Ltr)
    fin_state_all = Str[np.arange(N), Ltr - 1]
    if shuffle_labels:
        fin_state_all = fin_state_all[rng.permutation(N)]
    fin_prop_all = color[fin_state_all]

    def feat(prev_idx, x):
        return np.concatenate([emb[prev_idx], x], axis=1)         # [B, n_in]

    def roll_hard(Xb, Lb):
        """Argmax DETACHED rollout THROUGH THE SPIKING LIF; return (prev-state entering the final step, x at final step)."""
        B = len(Lb); Lmax = int(Lb.max())
        cur = np.full(B, ident, dtype=np.int64)
        prev_fin = np.full(B, ident, dtype=np.int64); x_fin = np.zeros((B, n_pool), dtype=np.float32)
        for t in range(Lmax):
            active = (Lb > t); last = (Lb == (t + 1))
            prev_fin[last] = cur[last]; x_fin[last] = Xb[last, t]
            rate, _ = lif_rate(feat(cur, Xb[:, t]), W1, T)
            nxt = (rate @ W2.T + b2).argmax(1)
            cur = np.where(active, nxt, cur)
        return prev_fin, x_fin

    phases = sorted(set(int(x) for x in Ltr)) if curriculum else [int(Ltr.max())]
    for phase_len in phases:
        sub = np.where(Ltr <= phase_len)[0]
        for ep in range(epochs):
            prev_fin, x_fin = roll_hard(Xtr[sub], Ltr[sub])       # roll with CURRENT spiking transition (detached)
            order = rng.permutation(len(sub))
            for i in range(0, len(sub), batch):
                b = order[i:i + batch]; gi = sub[b]; B = len(b)
                F = feat(prev_fin[b], x_fin[b])
                rate, sg = lif_rate(F, W1, T, want_grad=True)     # SPIKING forward (+ surrogate for the backward)
                if supervise == "state":
                    sm = _softmax(rate @ W2.T + b2); d = (sm - eyeK[fin_state_all[gi]]) / B
                    dW2 = d.T @ rate; db2 = d.sum(0); drate = d @ W2
                    W2 -= lr * dW2; b2 -= lr * db2
                else:
                    sm = _softmax(rate @ Wp.T + bp); d = (sm - eye2[fin_prop_all[gi]]) / B
                    dWp = d.T @ rate; dbp = d.sum(0); drate = d @ Wp
                    Wp -= lr * dWp; bp -= lr * dbp
                dpre = drate * sg                                 # surrogate through the LIF threshold
                W1 -= lr * (dpre.T @ F)

    def eval_split(split):
        Xe, ye, Le, _, Se = task[split]
        prev_fin, x_fin = roll_hard(Xe, Le)
        rate, _ = lif_rate(feat(prev_fin, x_fin), W1, T)
        fin_state = (rate @ W2.T + b2).argmax(1)
        if supervise == "property":
            prop = ((rate @ Wp.T + bp).argmax(1) == ye).mean()
        else:
            prop = (color[fin_state] == ye).mean()
        return {"prop": float(prop), "state_track": float((fin_state == Se[np.arange(len(Le)), Le - 1]).mean())}

    return {"same": eval_split("test_same"), "deeper": eval_split("test_deeper"), "train": eval_split("train")}


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def run_seed(group, seed, n_hid, epochs, n_pool):
    task = make_group_task(group, seed, n_pool=n_pool, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    st = train_spiking_weak(task, seed=seed, n_hid=n_hid, epochs=epochs, supervise="state")
    pr = train_spiking_weak(task, seed=seed, n_hid=n_hid, epochs=epochs, supervise="property")
    sh = train_spiking_weak(task, seed=seed, n_hid=n_hid, epochs=epochs, supervise="state", shuffle_labels=True)
    return {"seed": seed, "SPKFWD_STATE_deeper": round(st["deeper"]["prop"], 3), "SPKFWD_STATE_track": round(st["deeper"]["state_track"], 3),
            "PROPERTY_deeper": round(pr["deeper"]["prop"], 3), "SHUFFLE_deeper": round(sh["deeper"]["prop"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    n_pool = 256 if a.group == "A5" else 64
    print(f"[D3 SPIKING-FORWARD weak learning] {a.group} | delta learned from END-STATE-only supervision THROUGH a spiking LIF hidden (surrogate grad)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(a.group, s, a.n_hid, a.epochs, n_pool)
        rows.append(r)
        print(f"  [seed {s}] SPIKING-FORWARD weak-learn STATE deeper={r['SPKFWD_STATE_deeper']} (track={r['SPKFWD_STATE_track']}) || "
              f"PROPERTY(=rung3)={r['PROPERTY_deeper']} || SHUFFLE={r['SHUFFLE_deeper']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        sd, pr, sh = _m("SPKFWD_STATE_deeper"), _m("PROPERTY_deeper"), _m("SHUFFLE_deeper")
        go = (sd > 0.75) and (sd - pr > 0.15) and (sd - sh > 0.15)
        print(f"\n  AGGREGATE ({a.group}): SPIKING-FORWARD weak-learn STATE deeper={sd:.3f} | PROPERTY(rung3)={pr:.3f} | SHUFFLE={sh:.3f} (chance 0.5)", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the transition delta is LEARNED from END-STATE-only supervision THROUGH a spiking LIF hidden (surrogate grad) and length-generalizes (DEEPER '+format(sd,'.2f')+' >> property '+format(pr,'.2f')+' >> shuffle '+format(sh,'.2f')+') -> the weak-supervision transition learning has a SPIKING FORWARD throughout (matching rung-2 on-spikes, now from weak supervision)' if go else 'the spiking-forward weak learning did not clearly beat the controls (tune epochs/T/n_hid)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
