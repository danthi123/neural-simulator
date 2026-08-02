"""gap#4 crux DEPTH-RESCUE MECHANISM probe (STANDALONE; imports the shared runner, does NOT edit it).

THE QUESTION (task-independent -- this is the point). The rate result: fixed-random feedback-alignment (FA) credit
becomes MISALIGNED with the true gradient as depth grows (FA angle degrades with depth), while KP-learned
(Kolen-Pollack) feedback REALIGNS (Y -> W^T in direction), so KP rescues depth where fixed-FA collapses. We could NOT
build a clean obligatory-depth-3 ACCURACY task on spikes (parity=unoptimizable, hierarchical=memorization-shortcut).
So we measure the mechanism DIRECTLY via ALIGNMENT, which is a property of the CREDIT PATH, not the task's
depth-requirement: does fixed-FA's per-layer credit-vs-true-gradient alignment degrade with depth while KP's holds, ON
THE SPIKING LIF SNN?

METHOD. For each N_hidden in {2,3,4} and seed, on the depth-2 XOR->threshold task (matched config: hidden 32, T 24,
epochs 200, lr 0.05):
  * train arm FIXED-FA (Y_list fixed-random) and arm KP (Y_list KP-updated) via _train_snn_arm (reused, unedited)
  * on a fresh HELD batch, run ONE forward pass; from that SAME forward state compute
      (a) the TRUE per-hidden-layer weight gradient via backward_unroll_xp (surrogate-BPTT) = the reference "correct
          credit direction",
      (b) the DELIVERED per-hidden-layer weight gradient via _chained_fa_grads (the arm's FA/KP credit; kp_cfg=None so
          Y is used as-learned, not further updated).
    Interface-matched: SAME layers, SAME forward activations fs, SAME output_grad og. og built exactly as the trainer:
    og = repeat((softmax(logits) - onehot(y)) / T, T).
  * per hidden layer li, ALIGNMENT[li] = cosine( flatten(true_grad[li]), flatten(delivered_grad[li]) ) in [-1,1].

LAYER INDEXING: sizes = [n_in] + [hidden]*N + [k]; layers[0..N-1] = hidden, layers[N] = output. Hidden li=0 is the
DEEPEST hidden (furthest from output, distance N); hidden li=N-1 is the TOP hidden (reads the output error directly,
distance 1). FA is predicted to degrade at the DEEPEST layer as N grows.

DECISIVE READ:
  * KP-DEPTH-RESCUE CONFIRMED-ON-SPIKES if fixed-FA deepest-layer alignment DROPS toward 0 as N 2->3->4 while KP
    deepest-layer alignment STAYS clearly higher (gap WIDENS with depth).
  * SURROGATE-ATTENUATION WALL if BOTH FA and KP alignment collapse toward 0 at N>=3 (sigma' vanishes through hops
    regardless of feedback) -- names the next mechanism, an honest informative negative.
  * MEASUREMENT-BUG if even N=2 TOP-hidden alignment is low for both arms -- debug the interface first.

Run (numpy CPU):
    SIM_BACKEND=numpy python /tmp/claude-1000/-home-dant123-Projects-sim/\
87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/fa_kp_alignment_probe.py
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import sys
import time

_REPO = "/home/dant123/Projects/sim"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

# ---- reuse-by-import (NO edit) : the task + the transport-free chained-FA/KP credit + the trainer ----
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import (  # noqa: E402
    make_task_xor, _chained_fa_grads, _train_snn_arm)
# ---- reuse-by-import : the SAME forward/eval helpers the trainer uses ----
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _softmax, _forward_logits, _accuracy)
# ---- reuse-by-import : the TRUE surrogate-BPTT gradient = the reference correct credit direction ----
from sim.bptt_snn_gpu import backward_unroll_xp  # noqa: E402


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _delivered_output_grad(logits, y, T):
    """The EXACT output error the trainer builds: og = repeat((softmax(logits) - onehot(y)) / T, T)."""
    p = _softmax(logits)
    delta = p.copy()
    delta[np.arange(len(y)), y] -= 1.0
    return np.repeat((delta / T)[None, :, :], T, axis=0).astype(np.float64)


def measure_arm(mode, Xtr, ytr, Xte, yte, sizes, N, T, epochs, lr, lr_fa, in_gain, seed, batch_eval):
    """Train `mode` arm, then measure per-hidden-layer credit-vs-true-gradient alignment on a fresh held batch."""
    layers, Y_list = _train_snn_arm(Xtr, ytr, sizes, T, epochs, lr, lr_fa, in_gain, seed, mode)
    tr_acc = _accuracy(Xtr, ytr, layers, T, in_gain)
    te_acc = _accuracy(Xte, yte, layers, T, in_gain)

    # fresh HELD batch -> ONE forward pass -> the SAME forward state both credit paths read
    rng = np.random.default_rng(seed + 99)
    bidx = rng.permutation(len(Xte))[:batch_eval]
    Xb, yb = Xte[bidx], yte[bidx]
    logits, fs, inp = _forward_logits(Xb, layers, T, in_gain)
    og = _delivered_output_grad(logits, yb, T)
    alpha_leak = layers[0].leak

    # (a) TRUE surrogate-BPTT credit  (b) DELIVERED FA/KP credit -- SAME layers, fs, og. kp_cfg=None: use Y as-learned.
    true_wg, _ = backward_unroll_xp(inp, layers, fs, og, alpha=2.0, xp=np)
    deliv_wg = _chained_fa_grads(inp, layers, fs, og, Y_list, alpha_leak, alpha_surr=2.0,
                                 sigma_norm=True, train_hidden=True, kp_cfg=None, lr=0.0)

    # per-hidden-layer alignment (hidden layers are li = 0..N-1; li=0 deepest, li=N-1 top). Output layer li=N reported too.
    align_hidden = [_cosine(true_wg[li], deliv_wg[li]) for li in range(N)]
    align_output = _cosine(true_wg[N], deliv_wg[N])
    return {"mode": mode, "train_acc": tr_acc, "held_acc": te_acc,
            "align_hidden": align_hidden, "align_output": align_output}


def run(N, seed, hidden=32, T=24, epochs=200, lr=0.05, in_gain=1.0, batch_eval=256):
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_xor(seed)
    k = meta["k_classes"]; n_in = Xtr.shape[1]
    sizes = [n_in] + [hidden] * N + [k]
    lr_fa = lr
    fa = measure_arm("chained_fa", Xtr, ytr, Xte, yte, sizes, N, T, epochs, lr, lr_fa, in_gain, seed, batch_eval)
    kp = measure_arm("chained_fa_kp", Xtr, ytr, Xte, yte, sizes, N, T, epochs, lr, lr_fa, in_gain, seed, batch_eval)
    return {"N": N, "seed": seed, "sizes": sizes, "fa": fa, "kp": kp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--batch-eval", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    for N in args.n_list:
        for sd in args.seeds:
            r = run(N, sd, hidden=args.hidden, T=args.timesteps, epochs=args.epochs, lr=args.lr,
                    in_gain=args.in_gain, batch_eval=args.batch_eval)
            rows.append(r)
            fa, kp = r["fa"], r["kp"]
            print(f"[N={N} seed={sd}] "
                  f"FA held={fa['held_acc']:.3f} tr={fa['train_acc']:.3f} | "
                  f"KP held={kp['held_acc']:.3f} tr={kp['train_acc']:.3f}", flush=True)
            print(f"    FA align hidden(deep->top) {['%+.3f' % a for a in fa['align_hidden']]} out {fa['align_output']:+.3f}")
            print(f"    KP align hidden(deep->top) {['%+.3f' % a for a in kp['align_hidden']]} out {kp['align_output']:+.3f}",
                  flush=True)

    # ---------- summary tables (mean over seeds) ----------
    def mean_over_seeds(N, arm, layer_selector):
        vals = []
        for r in rows:
            if r["N"] != N:
                continue
            ah = r[arm]["align_hidden"]
            vals.append(layer_selector(ah, N))
        vals = [v for v in vals if v == v]  # drop nan
        return float(np.mean(vals)) if vals else float("nan")

    deepest = lambda ah, N: ah[0]        # li=0 = furthest from output
    top = lambda ah, N: ah[N - 1]        # li=N-1 = reads output error directly

    print("\n" + "=" * 92)
    print("ALIGNMENT TABLE  (cosine of delivered-credit vs true surrogate-BPTT gradient, mean over seeds "
          + str(args.seeds) + ")")
    print("=" * 92)
    print(f"{'N_hidden':>9} | {'arm':>9} | {'DEEPEST hidden (li=0)':>22} | {'TOP hidden (li=N-1)':>20}")
    print("-" * 92)
    for N in args.n_list:
        for arm, label in (("fa", "fixed-FA"), ("kp", "KP")):
            d = mean_over_seeds(N, arm, deepest)
            t = mean_over_seeds(N, arm, top)
            print(f"{N:>9} | {label:>9} | {d:>+22.3f} | {t:>+20.3f}")
    print("-" * 92)

    print("\nPER-LAYER (mean over seeds), deepest(li=0) -> top(li=N-1):")
    for N in args.n_list:
        for arm, label in (("fa", "fixed-FA"), ("kp", "KP")):
            per = []
            for li in range(N):
                vals = [r[arm]["align_hidden"][li] for r in rows if r["N"] == N]
                vals = [v for v in vals if v == v]
                per.append(float(np.mean(vals)) if vals else float("nan"))
            print(f"  N={N} {label:>9}: " + "  ".join("li%d=%+.3f" % (li, per[li]) for li in range(N)))

    # ---------- decisive verdict ----------
    print("\n" + "=" * 92)
    fa_deep = {N: mean_over_seeds(N, "fa", deepest) for N in args.n_list}
    kp_deep = {N: mean_over_seeds(N, "kp", deepest) for N in args.n_list}
    fa_top2 = mean_over_seeds(2, "fa", top) if 2 in args.n_list else float("nan")
    kp_top2 = mean_over_seeds(2, "kp", top) if 2 in args.n_list else float("nan")
    print(f"DEEPEST-layer alignment vs depth:  fixed-FA {[ '%+.3f'%fa_deep[N] for N in args.n_list ]}  "
          f"KP {[ '%+.3f'%kp_deep[N] for N in args.n_list ]}")
    print(f"N=2 TOP-hidden alignment (sanity, should be high both arms): FA {fa_top2:+.3f}  KP {kp_top2:+.3f}")

    Nmax = max(args.n_list)
    fa_lo = fa_deep.get(Nmax, float("nan"))
    kp_hi = kp_deep.get(Nmax, float("nan"))
    gap_widens = (2 in args.n_list) and (Nmax > 2) and \
        ((kp_deep[Nmax] - fa_deep[Nmax]) > (kp_deep[2] - fa_deep[2]) + 0.05)
    fa_degrades = (2 in args.n_list) and (Nmax > 2) and (fa_deep[Nmax] < fa_deep[2] - 0.10)
    _deep_Ns = [N for N in args.n_list if N >= 3]
    both_collapse = bool(_deep_Ns) and all(
        (abs(fa_deep[N]) < 0.10 and abs(kp_deep[N]) < 0.10) for N in _deep_Ns)
    meas_bug = (2 in args.n_list) and (fa_top2 < 0.30 and kp_top2 < 0.30)

    if meas_bug:
        verdict = ("MEASUREMENT-BUG: N=2 TOP-hidden alignment is LOW for both arms -> the credit-path interface is "
                   "wrong; debug before reading depth.")
    elif fa_degrades and kp_hi > fa_lo + 0.10 and gap_widens:
        verdict = ("KP-DEPTH-RESCUE CONFIRMED-ON-SPIKES: fixed-FA deepest-layer alignment DROPS with depth while KP "
                   "deepest-layer alignment stays clearly higher and the gap WIDENS -> the rate depth-rescue signature "
                   "reproduced mechanistically on the spiking LIF SNN.")
    elif both_collapse:
        verdict = ("SURROGATE-ATTENUATION WALL: BOTH fixed-FA and KP deepest-layer alignment collapse toward 0 at "
                   "N>=3 -> sigma'(v-theta) vanishes through hops regardless of feedback; NAMES the next mechanism "
                   "(per-layer credit normalization / e-prop temporal credit). Honest informative negative.")
    else:
        verdict = ("MIXED / INCONCLUSIVE: neither a clean KP rescue nor a symmetric collapse. Read the per-layer "
                   "table -- report the numbers as-is.")
    print("VERDICT: " + verdict)
    print("=" * 92)
    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
