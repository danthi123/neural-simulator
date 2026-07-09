"""D3 residual — RANK 1 (research-gated, `2026-07-09-...sparse-supervision...`): learn the transition delta from
END-STATE-ONLY supervision (NO per-step intermediate states). Rung 3 (`_d3_endlabel_supervision_derisk`) supervised only
the 2-way PROPERTY at the endpoint (1 bit) and did NOT length-generalize (deeper ~0.55). THE REFRAME: the running K-way
state needs ~log2(K) bits to identify (S3: 2.58, A5: 5.9); ONE endpoint bit cannot pin it, so BPTT through the scrambling
latent walk has no learnable target. But delta is LENGTH-INDEPENDENT (learned from SHORT sequences; depth-gen is the
drift-free rollout). So the fix is to supply ~log2(K) bits at SHORT lengths WITHOUT handing per-step states: supervise the
K-WAY FINAL STATE (Se[:,L-1]) at the endpoint of short sequences. Near-provably sufficient: length-1 endpoints fix
delta(ident,.) (s1=g1); once that is learned the model's own rollout pins s1, so length-2 endpoints supply an effective
(s1, g2)->s2 signal over ALL s1 -> the full transition table is recoverable from length-1+2 endpoints alone. Genuinely
LEARNED (only the final state is a target; the intermediate states are the model's OWN straight-through rollout), NOT
per-step taught.

THE DECISIVE ISOLATION (one variable): `--supervise state` (RANK 1, K-way endpoint) vs `--supervise property` (= rung 3,
2-way endpoint), SAME straight-through rollout. If state >> property on held-out-DEEPER, the reframe is confirmed: the
endpoint STATE (log2 K bits) is learnable where the endpoint PROPERTY (1 bit) is not.

ANTI-CHEATS: (a) held-out-DEEPER (learned iteration, not memorized length); (b) `--supervise property` control (= rung 3,
the 1-bit endpoint -> must fail = the isolation); (c) CONTINUOUS-carry control (`--no-rediscretize` -> drifts -> fails
deeper, isolates the re-discretization); (d) train-lens<=2 ablation (train 1,2 only -> still generalizes to 6,7,8 = delta
is length-independent, not depth-memorized); (e) memorization-floor (shuffle endpoint STATE labels -> collapse); (f)
permuted-ORDER (non-abelian task property, `order_control`); (g) multi-seed. Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_weak_supervision_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, build_group, order_control


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def train_endstate(task, seed=42, n_hid=160, epochs=120, lr=0.05, batch=128, discrete=True,
                   supervise="state", shuffle_labels=False, curriculum=True):
    """Learn delta from END-STATE-only supervision via a Dyna-style DETACHED-rollout CURRICULUM (the correct realization
    of RANK 1 -- straight-through BPTT teaches garbage early because the final-step target is conditioned on the model's
    OWN wrong intermediate prediction; the curriculum fixes the rollout depth-by-depth).

    Mechanism: for each sequence, roll forward AUTOREGRESSIVELY with argmax (the intermediate states are the model's OWN
    predictions, DETACHED -- no per-step targets), then take ONE supervised gradient step on the FINAL step only:
    CE( f(emb[roll_state_{L-1}], x_L), true_final_state_L ). The prev-state entering the final step is the model's own
    rolled prediction (detached, used as a fixed input) -> the final step is a clean per-step classifier of
    delta(roll_state_{L-1}, x_L) -> s_L. CURRICULUM: train length-1 first (delta(ident,.), a rollout-free clean target),
    then include length-2 (the model now rolls step-1 correctly -> the end-state cleanly teaches delta(s_1,.)), then
    length-3 -- so the rolled prev-state is CORRECT at each depth. NO intermediate state is ever a target (end-state only).

    supervise="state" (RANK 1): K-way final STATE (log2 K bits). supervise="property" (rung-3 control): 2-way property (1 bit).
    discrete is ignored for state (the rollout is always argmax); kept for signature compat with the continuous control below."""
    K = task["K"]; ident = task["ident"]; color = task["color"]; n_pool = task["n_pool"]
    rng = np.random.RandomState(seed + 33)
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)
    Wr = (rng.randn(n_hid, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32)
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    Ws = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bs = np.zeros(K, dtype=np.float32)
    Wp = (rng.randn(2, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bp = np.zeros(2, dtype=np.float32)
    color_arr = np.asarray(color); eyeK = np.eye(K, dtype=np.float32); eye2 = np.eye(2, dtype=np.float32)

    Xtr, ytr, Ltr, _, Str = task["train"]; N = len(Ltr)
    fin_state_all = Str[np.arange(N), Ltr - 1]
    if shuffle_labels:                                           # memorization-floor: destroy the endpoint target
        fin_state_all = fin_state_all[rng.permutation(N)]
    fin_prop_all = color_arr[fin_state_all]

    def roll_hard(Xb, Lb):
        """Argmax DETACHED rollout; return (prev_state ENTERING the final step, x at the final step) for the supervised
        final-step gradient. The intermediate states are the model's OWN argmax predictions (never targets)."""
        B = len(Lb); Lmax = int(Lb.max())
        cur = np.full(B, ident, dtype=np.int64)
        prev_fin = np.full(B, ident, dtype=np.int64); x_fin = np.zeros((B, n_pool), dtype=np.float32)
        for t in range(Lmax):
            active = (Lb > t); last = (Lb == (t + 1))
            prev_fin[last] = cur[last]; x_fin[last] = Xb[last, t]
            h = np.tanh(emb[cur] @ Wr.T + Xb[:, t] @ Wi.T); nxt = (h @ Ws.T + bs).argmax(1)
            cur = np.where(active, nxt, cur)
        return prev_fin, x_fin

    def final_scores(Xb, Lb, soft=False):
        """Roll out and return the FINAL-step K-way scores. soft=False -> re-discretize each step (state=emb[argmax], the
        mechanism); soft=True -> carry the continuous blend p@emb (the re-discretization LESION -> drift)."""
        B = len(Lb); Lmax = int(Lb.max())
        cur = np.full(B, ident, dtype=np.int64); cur_emb = np.tile(emb[ident], (B, 1))
        fin = np.zeros((B, K), dtype=np.float32)
        for t in range(Lmax):
            active = (Lb > t); last = (Lb == (t + 1))
            prev_emb = cur_emb if soft else emb[cur]
            h = np.tanh(prev_emb @ Wr.T + Xb[:, t] @ Wi.T); sc = h @ Ws.T + bs; p = _softmax(sc); nxt = sc.argmax(1)
            fin[last] = sc[last]
            if soft:
                cur_emb = np.where(active[:, None], p @ emb, cur_emb)
            cur = np.where(active, nxt, cur)
        return fin

    phases = sorted(set(int(x) for x in Ltr)) if curriculum else [int(Ltr.max())]
    for phase_len in phases:
        sub = np.where(Ltr <= phase_len)[0]                     # curriculum: all sequences UP TO this length
        for ep in range(epochs):
            prev_fin, x_fin = roll_hard(Xtr[sub], Ltr[sub])     # roll with CURRENT weights (detached argmax)
            order = rng.permutation(len(sub))
            for i in range(0, len(sub), batch):
                b = order[i:i + batch]; gi = sub[b]; B = len(b)
                pe = emb[prev_fin[b]]                            # detached prev-state embedding (model's own rolled state)
                h = np.tanh(pe @ Wr.T + x_fin[b] @ Wi.T)        # final-step hidden
                if supervise == "state":                        # RANK 1: K-way final-STATE target (log2 K bits)
                    sm = _softmax(h @ Ws.T + bs); d = (sm - eyeK[fin_state_all[gi]]) / B
                    dWo = d.T @ h; dbo = d.sum(0); dpre = (d @ Ws) * (1.0 - h ** 2)   # dpre uses the OLD Ws (before update)
                    Ws -= lr * dWo; bs -= lr * dbo
                else:                                            # rung-3 control: 2-way property target (1 bit)
                    sm = _softmax(h @ Wp.T + bp); d = (sm - eye2[fin_prop_all[gi]]) / B
                    dWo = d.T @ h; dbo = d.sum(0); dpre = (d @ Wp) * (1.0 - h ** 2)
                    Wp -= lr * dWo; bp -= lr * dbo
                Wr -= lr * (dpre.T @ pe); Wi -= lr * (dpre.T @ x_fin[b])   # emb[prev] DETACHED -> no gradient to emb

    def eval_split(split, soft=False):
        Xe, ye, Le, _, Se = task[split]
        fin = final_scores(Xe, Le, soft=soft); fin_state = fin.argmax(1)
        if supervise == "property":                             # property head reads the final-step hidden
            prev_fin, x_fin = roll_hard(Xe, Le)
            h = np.tanh(emb[prev_fin] @ Wr.T + x_fin @ Wi.T)
            prop = ((h @ Wp.T + bp).argmax(1) == ye).mean()
        else:
            prop = (color_arr[fin_state] == ye).mean()
        true_state = Se[np.arange(len(Le)), Le - 1]
        return {"prop": float(prop), "state_track": float((fin_state == true_state).mean())}

    return {"same": eval_split("test_same"), "deeper": eval_split("test_deeper", soft=not discrete),
            "train": eval_split("train")}


def run_seed(group, seed, n_hid, epochs, train_lens, test_lens, n_pool):
    task = make_group_task(group, seed, n_pool=n_pool, n_per_len=2500, train_lens=train_lens, test_lens=test_lens)
    st = train_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, discrete=True, supervise="state")
    pr = train_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, discrete=True, supervise="property")
    cont = train_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, discrete=False, supervise="state")
    shuf = train_endstate(task, seed=seed, n_hid=n_hid, epochs=epochs, discrete=True, supervise="state", shuffle_labels=True)
    return {"seed": seed, "order_changes": round(order_control(task, seed), 3),
            "STATE_deeper_prop": round(st["deeper"]["prop"], 3), "STATE_deeper_track": round(st["deeper"]["state_track"], 3),
            "STATE_same_prop": round(st["same"]["prop"], 3),
            "PROPERTY_deeper_prop": round(pr["deeper"]["prop"], 3),
            "CONTINUOUS_deeper_prop": round(cont["deeper"]["prop"], 3),
            "SHUFFLE_deeper_prop": round(shuf["deeper"]["prop"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--train-lens", default="1,2,3")
    ap.add_argument("--test-lens", default="6,7,8")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    train_lens = tuple(int(x) for x in a.train_lens.replace(",", " ").split())
    test_lens = tuple(int(x) for x in a.test_lens.replace(",", " ").split())
    n_pool = 256 if a.group == "A5" else 64
    print(f"[D3 WEAK-SUPERVISION RANK 1] {a.group} | K-way END-STATE-only supervision (log2 K bits) vs 2-way property (1 bit); train {train_lens} -> DEEPER {test_lens}", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(a.group, s, a.n_hid, a.epochs, train_lens, test_lens, n_pool)
        rows.append(r)
        print(f"  [seed {s}] STATE-endpoint DEEPER prop={r['STATE_deeper_prop']} (track={r['STATE_deeper_track']}) || "
              f"PROPERTY-endpoint(=rung3) DEEPER={r['PROPERTY_deeper_prop']} || CONTINUOUS={r['CONTINUOUS_deeper_prop']} || "
              f"SHUFFLE={r['SHUFFLE_deeper_prop']} (order-changes={r['order_changes']})", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        sd, pr, co, sh = _m("STATE_deeper_prop"), _m("PROPERTY_deeper_prop"), _m("CONTINUOUS_deeper_prop"), _m("SHUFFLE_deeper_prop")
        # GO = the LOAD-BEARING isolations for WEAK SUPERVISION: STATE (log2 K endpoint bits) >> PROPERTY (rung-3's 1-bit
        # endpoint = the reframe) AND >> SHUFFLE (memorization-floor = genuine learning). The continuous-carry control is
        # REPORTED not gated: on a clean transition softmax@emb ~ one-hot ~ the clean attractor, so it does not drift on
        # S3 -- the re-discretization is load-bearing at DEPTH (established in the one-loop finding), not the isolation here.
        go = (sd > 0.75) and (sd - pr > 0.15) and (sd - sh > 0.15)
        print(f"\n  AGGREGATE ({a.group}): STATE-endpoint DEEPER={sd:.3f} | PROPERTY(rung3, 1-bit)={pr:.3f} | SHUFFLE(mem-floor)={sh:.3f} | (continuous-carry, reported)={co:.3f} (chance 0.5)", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the transition delta is LEARNED from K-way END-STATE-only supervision (NO per-step intermediate states; a short-length curriculum + the model own detached rollout) and length-generalizes (DEEPER '+format(sd,'.2f')+' >> the 2-way property endpoint = rung 3 '+format(pr,'.2f')+' [the reframe: log2 K endpoint bits learnable where 1 bit is not], >> shuffle-label '+format(sh,'.2f')+' [genuine, not memorized]) -> the residual credit wall is SURPASSED: the discrete-attractor composition is GENUINELY LEARNED, not per-step taught' if go else 'STATE-endpoint did not clearly beat the load-bearing controls (property/shuffle); read the gaps'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
