"""D3 spiking port, rung 2: the TRANSITION on-spikes. Rung 1 realized the RE-DISCRETIZATION on the Izhikevich bridge;
the transition (delta: state x input -> next-state scores) was still the rate-learned tanh-hidden weights. The cheap-
first design test (`discrete_attractor_rnn(linear=True)` -> 0.584) proved the transition NEEDS a nonlinear hidden
layer (the group-mult DFA is not linearly separable). THIS realizes that hidden layer as a SPIKING LIF pool: the
transition is a rate-coded feedforward SNN (`[emb[state]; input_code]` -> W1 -> a LIF HIDDEN pool [T steps, hard-reset
LIF, rate = mean spikes] -> W2 -> K next-state scores), TRAINED THROUGH the spiking nonlinearity by SURROGATE GRADIENT
(sim/surrogate_grad atan). If the spiking-hidden transition learns the group-mult DFA (step-delta ~1.0), the WHOLE step
(transition + the rung-1 re-discretization) is spiking -> the simulated recurrent language cortex step is fully on-substrate.

ANTI-CHEATS: (a) step-delta (teacher-forced transition acc) >> chance == the rate tanh baseline; (b) a LINEAR-hidden
(no spiking nonlinearity, rate=identity) control must FAIL (isolates the nonlinearity as load-bearing); (c) multi-seed.
Reuse-by-import (`_d3_group_composition_derisk` triples + `sim.surrogate_grad`); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_spiking_transition_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, build_group


def _emb(K, n_hid, seed):
    return (np.random.RandomState(seed + 9).randn(K, n_hid) * 0.5).astype(np.float32)


def collect_triples(task, split, emb):
    """(input feature = [emb[prev_state]; input_code]) -> next_state, teacher-forced over the split's sequences."""
    Xe, ye, Le, _, Se = task[split]; ident = task["ident"]
    feats, nxt = [], []
    for n in range(len(Le)):
        prev = ident
        for t in range(int(Le[n])):
            feats.append(np.concatenate([emb[prev], Xe[n, t]])); nxt.append(int(Se[n, t])); prev = int(Se[n, t])
    return np.asarray(feats, dtype=np.float32), np.asarray(nxt, dtype=np.int64)


def spiking_transition(task, seed=42, n_hid=192, T=16, leak=0.9, thr=1.0, epochs=25, lr=0.05, batch=256, spiking=True):
    """Rate-coded feedforward LIF SNN transition, surrogate-gradient trained. spiking=False -> rate=identity(pre-act)
    (a LINEAR-hidden control: no spiking nonlinearity -> must fail on the non-linearly-separable DFA)."""
    from sim.surrogate_grad import atan_surrogate
    K = task["K"]; emb = _emb(K, n_hid, seed)
    n_in = n_hid + task["n_pool"]
    rng = np.random.RandomState(seed + 21)
    W1 = (rng.randn(n_hid, n_in) * np.sqrt(2.0 / n_in)).astype(np.float32)
    W2 = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); b2 = np.zeros(K, dtype=np.float32)
    Ftr, Ntr = collect_triples(task, "train", emb)
    Nn = len(Ntr)

    def lif_rate(F, want_grad=False):
        """Run the LIF hidden pool for T steps on static input F@W1.T; return rate = mean spikes (+ a surrogate factor
        for the backward through the threshold). Hard-reset LIF: v = leak*v*(1-s) + drive."""
        drive = F @ W1.T                                          # [B, n_hid] constant per-step drive
        B = F.shape[0]
        v = np.zeros((B, n_hid), dtype=np.float32); acc = np.zeros((B, n_hid), dtype=np.float32)
        sg = np.zeros((B, n_hid), dtype=np.float32); s = np.zeros((B, n_hid), dtype=np.float32)
        if not spiking:
            return drive, np.ones_like(drive)                    # PURE-LINEAR control (identity, NO nonlinearity ->
            #                                                      the 2 linear layers compose to linear -> must FAIL on
            #                                                      the non-linearly-separable group-mult DFA)
        for _ in range(T):
            v = leak * v * (1.0 - s) + drive
            s = (v >= thr).astype(np.float32); acc += s
            if want_grad:
                sg += atan_surrogate(v - thr)                     # accumulate the surrogate d(spike)/d(v) over steps
        rate = acc / T
        return rate, (sg / T)

    for ep in range(epochs):
        order = rng.permutation(Nn)
        for i in range(0, Nn, batch):
            bi = order[i:i + batch]; F = Ftr[bi]; y = Ntr[bi]; B = len(bi)
            rate, sgrad = lif_rate(F, want_grad=True)
            logits = rate @ W2.T + b2
            ex = np.exp(logits - logits.max(1, keepdims=True)); sm = ex / ex.sum(1, keepdims=True)
            d = sm.copy(); d[np.arange(B), y] -= 1.0; d /= B
            dW2 = d.T @ rate; db2 = d.sum(0)
            drate = d @ W2                                        # [B, n_hid]
            dpre = drate * sgrad                                  # surrogate through the LIF threshold
            dW1 = dpre.T @ F
            W2 -= lr * dW2; b2 -= lr * db2; W1 -= lr * dW1

    def step_acc(split):
        F, N = collect_triples(task, split, emb)
        rate, _ = lif_rate(F, want_grad=False)
        return float(((rate @ W2.T + b2).argmax(1) == N).mean())

    return {"step_delta_train": step_acc("train"), "step_delta_same": step_acc("test_same"),
            "step_delta_deeper": step_acc("test_deeper"),
            "weights": {"emb": emb, "W1": W1, "W2": W2, "b2": b2, "T": T, "leak": leak, "thr": thr, "n_hid": n_hid}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-per-len", type=int, default=None, help="triples-per-length (default A5=8000 / else 1500)")
    ap.add_argument("--n-pool", type=int, default=None, help="pool code width (default A5=256 / else 64)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    np_pool = a.n_pool if a.n_pool is not None else (256 if a.group == "A5" else 64)
    nperlen = a.n_per_len if a.n_per_len is not None else (8000 if a.group == "A5" else 1500)
    print(f"[D3 spiking TRANSITION] {a.group} | the group-mult delta learned THROUGH a spiking LIF hidden pool (surrogate grad)", flush=True)
    rows = []
    print(f"  config: n_hid={a.n_hid} epochs={a.epochs} n_pool={np_pool} n_per_len={nperlen}", flush=True)
    for s in seeds:
        task = make_group_task(a.group, s, n_pool=np_pool, n_per_len=nperlen, train_lens=(1, 2, 3), test_lens=(4, 5, 6))
        spk = spiking_transition(task, seed=s, n_hid=a.n_hid, epochs=a.epochs, spiking=True)
        lin = spiking_transition(task, seed=s, n_hid=a.n_hid, epochs=a.epochs, spiking=False)
        rows.append({"seed": s, "spk": spk, "lin": lin})
        print(f"  [seed {s}] SPIKING-hidden step-delta: train={spk['step_delta_train']:.3f} same={spk['step_delta_same']:.3f} "
              f"deeper={spk['step_delta_deeper']:.3f} || LINEAR-hidden control: train={lin['step_delta_train']:.3f}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        sd = float(np.mean([r["spk"]["step_delta_same"] for r in rows]))
        ld = float(np.mean([r["lin"]["step_delta_train"] for r in rows]))
        Kg = build_group(a.group)[1].shape[0]
        go = (sd > 0.90) and (sd - ld > 0.15)
        print(f"\n  AGGREGATE ({a.group}): SPIKING-hidden step-delta(same)={sd:.3f} | LINEAR-hidden(train)={ld:.3f} (chance={1.0/Kg:.3f})", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the group-mult DFA transition is learned THROUGH a spiking LIF hidden pool (surrogate grad, step-delta '+format(sd,'.2f')+' >> the linear-hidden control) -> composed with rung-1 re-discretization, the WHOLE recurrent step is spiking = the simulated recurrent language cortex step fully on-substrate; next: wire transition+re-discretization into one on-bridge loop + reduce supervision' if go else 'the spiking-hidden transition did not learn cleanly (tune T/thr/leak/lr/epochs; the LIF rate-code must resolve the nonlinear delta)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
