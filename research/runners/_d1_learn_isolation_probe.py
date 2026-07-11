"""D1 LEARN-ISOLATION PROBE (read-only diagnostic) -- WHERE does the on-bridge spiking BDSP credit path break?

CONTEXT: `OnBridgeBDSPNet` (in `_d1_onbridge_learn_to_accuracy_derisk.py`) does NOT learn -- train accuracy stuck at
chance -- EVEN in a confirmed bursting regime (couple_soma=True, soma_g=500, B_apical ~ 0.5). This probe ISOLATES
the break with three orthogonal checks, each printing a crisp VERDICT line. It builds the EXACT failing config
(hidden=60, couple_soma=True, soma_g=500.0, hidden_bias=20.0, output_bias=20.0, bdsp_lr=0.03, fwd_wmean=40.0,
bdsp_w_max=200.0). REUSE-BY-IMPORT ONLY -- NO edit to sim/ (the protected engine) and no edit to the derisk file.

THREE MODES (each vs chance 0.5 + the numpy 2-layer OVERFIT CEILING trained+eval'd on the same balanced set):

  readout_sanity  -- THE most fundamental check: does the spike-count readout even reflect the hidden->output
                     WEIGHTS?  Hand-overwrite hidden->output in cp_connections (pool 0 large, pool 1 zero), read
                     the balanced set -> expect ~all predicted class 0.  FLIP (pool 1 large, pool 0 zero) -> expect
                     ~all class 1.  If the flip is NOT followed the eval metric cannot reflect learned weights,
                     which by itself explains the non-learning.  DECISIVE.

  single_layer    -- freeze input->hidden (a FIXED random projection) and train ONLY hidden->output with BDSP.
                     If one trainable spiking layer on a random projection learns, the hidden->output credit is
                     fine and the problem is DEPTH / credit-propagation.  If flat, the hidden->output credit itself
                     is broken.

  settled_credit  -- test the TRANSIENT hypothesis (B lags E, so the first per-step dw is spurious LTD).  Per
                     sample: settle E/B/Pbar with the apical ON but learning OFF, THEN apply a few learning steps
                     from the SETTLED B/E.  Compare train accuracy to the standard per-step train_epoch on the SAME
                     set.  If settled >> per-step, the transient IS the contaminant.

RUN (OPENBLAS_NUM_THREADS=1 SIM_BACKEND=numpy):
  readout_sanity : python -u -m research.runners._d1_learn_isolation_probe --mode readout_sanity --n-train 16
  single_layer   : python -u -m research.runners._d1_learn_isolation_probe --mode single_layer   --n-train 16 --epochs 60
  settled_credit : python -u -m research.runners._d1_learn_isolation_probe --mode settled_credit --n-train 16 --epochs 60
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
# tiny matmuls + a small bridge -> one BLAS thread (oversubscription is far slower).
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# Reuse-by-import: the exact failing net, the task loader, and the numpy backprop oracle.
from research.runners._d1_onbridge_learn_to_accuracy_derisk import (  # noqa: E402
    OnBridgeBDSPNet,
    _load_task,
    _numpy_oracle_heldout,
)

# The EXACT construct of the failing experiment (verified against OnBridgeBDSPNet.__init__ signature).
NET_KW = dict(
    hidden=60, couple_soma=True, soma_g=500.0, hidden_bias=20.0, output_bias=20.0,
    bdsp_lr=0.03, fwd_wmean=40.0, bdsp_w_max=200.0,
)
CHANCE = 0.5  # by construction the subset is class-balanced


# ============================================================================================================
# helpers
# ============================================================================================================
def _build_net(seed, n_bits):
    return OnBridgeBDSPNet(seed=seed, n_bits=n_bits, **NET_KW)


def _balanced_subset(X, y, k, seed):
    """Equal class-0 / class-1 rows so chance is exactly 0.5 (task-required)."""
    X = np.asarray(X); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    half = min(k // 2, len(idx0), len(idx1))
    sel = np.concatenate([idx0[:half], idx1[:half]])
    rng.shuffle(sel)
    return X[sel], y[sel]


def _overfit_ceiling(n_bits, Xbal, ybal, seed):
    """numpy 2-layer backprop MLP TRAINED AND EVALUATED on the SAME balanced set = the ~1.0 overfit ceiling."""
    return _numpy_oracle_heldout(n_bits, 60, Xbal, ybal, Xbal, ybal,
                                 epochs=500, lr=0.5, batch=64, seed=seed)


def _read_data(net):
    from sim.backend import to_host
    return np.asarray(to_host(net.sb.cp_connections.data), dtype=np.float64).copy()


def _write_data(net, data_host):
    """Write the full (nnz-length) weight vector back into cp_connections.data (the forward-pass weights).
    coo.row/coo.col (and net.mask_*) are index-aligned to cp_connections.data, so masked writes hit the right
    synapses."""
    from sim.backend import from_host
    net.sb.cp_connections.data[:] = from_host(np.asarray(data_host, dtype=np.float32))


def _top_error(net, readout, y):
    """The clean top error e = onehot - softmax(readout), exactly as OnBridgeBDSPNet.train_epoch computes it."""
    z = readout - readout.max(); ez = np.exp(z); p = ez / ez.sum()
    onehot = np.zeros(net.n_classes); onehot[int(y)] = 1.0
    return onehot - p


def _freeze_in2hid(net):
    """Freeze the input->hidden pathway (a FIXED random projection). This net has NO plastic=False synapses, so
    cp_synapse_plastic_mask is None; allocate an nnz-length boolean mask that is True everywhere EXCEPT the
    input->hidden synapses. The committed BDSP update (bridge.py:7264) does `new_w = where(mask, new_w, cur_w)`,
    so mask==False keeps those weights verbatim -> the input->hidden layer stays a fixed random projection while
    hidden->output learns."""
    from sim.backend import from_host
    nnz = int(net.sb.cp_connections.nnz)
    mask = np.ones(nnz, dtype=bool)
    mask[net.mask_in2hid] = False          # net.mask_in2hid is an nnz-length boolean over the cached COO
    net.sb.cp_synapse_plastic_mask = from_host(mask)


# ============================================================================================================
# MODE 1 -- readout_sanity: does the spike-count readout reflect the hidden->output weights?
# ============================================================================================================
def mode_readout_sanity(seed, n_bits, Xbal, settle, big_w):
    net = _build_net(seed, n_bits)
    m_hid2out = net.mask_hid2out                                   # row in hidden AND col in output (nnz-aligned)
    m_c0 = m_hid2out & np.isin(net._coo_col, net.class_idx[0])     # -> output pool for class 0
    m_c1 = m_hid2out & np.isin(net._coo_col, net.class_idx[1])     # -> output pool for class 1

    def set_pool(big_class):
        d = _read_data(net)
        d[m_c0] = big_w if big_class == 0 else 0.0
        d[m_c1] = big_w if big_class == 1 else 0.0
        _write_data(net, d)

    def eval_flip(big_class):
        set_pool(big_class)
        big, other = [], []                                       # per-sample big-pool / zeroed-pool spike counts
        preds = []
        for x in Xbal:
            r = net._readout(x, settle)                           # learning + apical OFF inside _readout
            preds.append(int(np.argmax(r)))
            big.append(float(r[big_class])); other.append(float(r[1 - big_class]))
        big = np.asarray(big); other = np.asarray(other)
        frac_pred = float(np.mean(np.asarray(preds) == big_class))   # raw argmax (ties break toward class 0)
        driven = big != other                                    # samples where the hidden layer drove SOME output
        n_degenerate = int(np.sum(~driven))                      # both pools tied (usually both 0 = hidden-silent)
        # readout FOLLOWS the weights iff, whenever there is any signal, the large-weight pool out-fires the zeroed one
        follow_driven = float(np.mean(big[driven] > other[driven])) if driven.any() else 0.0
        return dict(big_mean=float(big.mean()), other_mean=float(other.mean()), frac_pred=frac_pred,
                    follow_driven=follow_driven, n_driven=int(driven.sum()), n_degenerate=n_degenerate)

    f0 = eval_flip(0); f1 = eval_flip(1)
    # tie-robust: the readout MECHANISM works iff, on every sample that produced output, the big pool won both flips.
    works = (f0["n_driven"] > 0 and f0["follow_driven"] >= 0.85) and (f1["n_driven"] > 0 and f1["follow_driven"] >= 0.85)
    n_degen = f0["n_degenerate"] + f1["n_degenerate"]

    print(f"  pool-0 large : mean counts [big {f0['big_mean']:.2f}, zeroed {f0['other_mean']:.2f}]  "
          f"-> fraction predicted class 0 = {f0['frac_pred']:.3f}  (big-pool wins on {f0['n_driven']}/"
          f"{len(Xbal)} driven samples; {f0['n_degenerate']} hidden-silent tie(s))", flush=True)
    print(f"  pool-1 large : mean counts [big {f1['big_mean']:.2f}, zeroed {f1['other_mean']:.2f}]  "
          f"-> fraction predicted class 1 = {f1['frac_pred']:.3f}  (big-pool wins on {f1['n_driven']}/"
          f"{len(Xbal)} driven samples; {f1['n_degenerate']} hidden-silent tie(s))", flush=True)
    if works:
        verdict = ("READOUT WORKS -- whenever the hidden layer drives the output, the spike-count readout follows "
                   "the hidden->output weights (the large-weight pool out-fires the zeroed pool on every driven "
                   "sample, both flips). The eval metric CAN reflect learned weights, so non-learning lies UPSTREAM "
                   "(credit / plasticity into hidden->output), not in the readout.")
        if n_degen > 0:
            verdict += (" NOTE: some inputs left the output SILENT (hidden fired nothing into output; per-flip "
                        f"hidden-silent counts {f0['n_degenerate']} and {f1['n_degenerate']} of {len(Xbal)}) -> a "
                        "separate DRIVE/DEPTH signal (those inputs give the learner no output signal to read/train "
                        "on); probe with --mode single_layer.")
    else:
        verdict = ("READOUT BROKEN -- on a driven sample the large-weight pool did NOT out-fire the zeroed pool "
                   "(follow_driven pool-0 %.2f / pool-1 %.2f). The eval metric cannot reflect learned hidden->output "
                   "weights, which by itself explains the stuck-at-chance training. CRITICAL FINDING."
                   % (f0["follow_driven"], f1["follow_driven"]))
    print(f"\n  VERDICT [readout_sanity]: {verdict}", flush=True)
    return works


# ============================================================================================================
# MODE 2 -- single_layer: freeze input->hidden (random projection), train ONLY hidden->output with BDSP.
# ============================================================================================================
def mode_single_layer(seed, n_bits, Xbal, ybal, settle, epochs, teach_steps):
    net = _build_net(seed, n_bits)
    _freeze_in2hid(net)
    w_ih0, w_ho0 = net.pathway_weight_sums()
    for ep in range(epochs):
        net.train_epoch(Xbal, ybal, "bdsp", settle, teach_steps, seed + 1000 * ep + 7)
    w_ih1, w_ho1 = net.pathway_weight_sums()
    acc = net.accuracy(Xbal, ybal, settle)
    dw_ih = abs(w_ih1 - w_ih0); dw_ho = abs(w_ho1 - w_ho0)
    learns = acc > CHANCE + 0.15

    print(f"  input->hidden FROZEN (dw {dw_ih:.4f}, should be ~0)  |  hidden->output TRAINABLE (dw {dw_ho:.4f})",
          flush=True)
    print(f"  train accuracy {acc:.3f}  vs chance {CHANCE:.3f}", flush=True)
    verdict = ("SINGLE LAYER LEARNS -- one trainable spiking layer on a fixed random projection learns "
               f"(acc {acc:.3f} > chance+0.15). The hidden->output BDSP credit WORKS; the break is DEPTH / "
               "credit-propagation into the hidden layer (input->hidden)." if learns else
               "SINGLE LAYER FLAT -- even with input->hidden frozen as a random projection, training only "
               f"hidden->output stays at chance (acc {acc:.3f}). The hidden->output BDSP credit ITSELF is broken "
               f"(check: hidden->output moved dw {dw_ho:.4f}).")
    print(f"\n  VERDICT [single_layer]: {verdict}", flush=True)
    return learns


# ============================================================================================================
# MODE 3 -- settled_credit: settle E/B/Pbar (apical on, learning off), THEN learn from the settled B/E.
# ============================================================================================================
def _settled_train_epoch(net, X, y, settle, teach_on, shuffle_seed):
    rng = np.random.default_rng(shuffle_seed)
    for i in rng.permutation(len(X)):
        r = net._readout(X[i], settle)                     # forward read (learning frozen inside _readout)
        e = _top_error(net, r, y[i])
        # SETTLE: apical ON, learning OFF -> let E/B/Pbar reach steady state (no spurious transient dw).
        net._reset_membrane()
        net._set_input_drive(X[i])
        net._set_apical(e)
        net.cfg.bdsp_learning_rate = 0.0
        net._run(settle, accumulate_out=False)
        # TEACH from SETTLED B/E: learning ON for a few steps only.
        net.cfg.bdsp_learning_rate = net._bdsp_lr
        net._run(teach_on, accumulate_out=False)
        net._set_apical(None)
        net.cfg.bdsp_learning_rate = 0.0


def mode_settled_credit(seed, n_bits, Xbal, ybal, settle, epochs, teach_on, teach_steps):
    # settled arm
    net_s = _build_net(seed, n_bits)
    for ep in range(epochs):
        _settled_train_epoch(net_s, Xbal, ybal, settle, teach_on, seed + 1000 * ep + 7)
    acc_settled = net_s.accuracy(Xbal, ybal, settle)
    # standard per-step arm (fresh net, same seed/init)
    net_p = _build_net(seed, n_bits)
    for ep in range(epochs):
        net_p.train_epoch(Xbal, ybal, "bdsp", settle, teach_steps, seed + 1000 * ep + 7)
    acc_perstep = net_p.accuracy(Xbal, ybal, settle)
    helps = (acc_settled > acc_perstep + 0.10) and (acc_settled > CHANCE + 0.10)

    print(f"  SETTLED-credit train acc {acc_settled:.3f}  |  standard per-step train acc {acc_perstep:.3f}  "
          f"|  chance {CHANCE:.3f}", flush=True)
    verdict = ("SETTLED-CREDIT HELPS -- settling E/B/Pbar before applying dw beats the per-step update "
               f"({acc_settled:.3f} >> {acc_perstep:.3f}); the B-lags-E transient (spurious early LTD) IS the "
               "contaminant of the per-step rule." if helps else
               f"SETTLED-CREDIT NO DIFFERENT -- settling first ({acc_settled:.3f}) does not beat per-step "
               f"({acc_perstep:.3f}); the transient is NOT the (main) contaminant -- look elsewhere "
               "(readout / credit path).")
    print(f"\n  VERDICT [settled_credit]: {verdict}", flush=True)
    return helps


# ============================================================================================================
def main():
    ap = argparse.ArgumentParser(description="D1 on-bridge BDSP learn-isolation probe (read-only).")
    ap.add_argument("--mode", choices=["readout_sanity", "single_layer", "settled_credit"], required=True)
    ap.add_argument("--n-train", type=int, default=16, help="balanced subset size (equal class-0/1 rows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settle", type=int, default=50, help="read/settle steps per sample")
    ap.add_argument("--epochs", type=int, default=60, help="training epochs (single_layer / settled_credit)")
    ap.add_argument("--teach-steps", type=int, default=10, help="learning-ON steps in the standard per-step teach")
    ap.add_argument("--teach-on-steps", type=int, default=5, help="learning-ON steps AFTER settle (settled_credit)")
    ap.add_argument("--big-w", type=float, default=200.0, help="the large hand-set weight (readout_sanity)")
    ap.add_argument("--task", default="emerge1", help="task (default emerge1 = the exact D1 depth-2 task)")
    a = ap.parse_args()

    (Xtr, ytr), (Xte, yte), n_bits = _load_task(a.task, a.seed, 4)
    Xbal, ybal = _balanced_subset(Xtr, ytr, a.n_train, a.seed + 11)
    ceiling = _overfit_ceiling(n_bits, Xbal, ybal, a.seed)

    print("=" * 100, flush=True)
    print(f"[d1-learn-isolation] mode={a.mode}  task={a.task}  seed={a.seed}  n_bits={n_bits}", flush=True)
    print(f"[d1-learn-isolation] balanced n_train={len(Xbal)}  chance={CHANCE:.3f}  "
          f"numpy 2-layer OVERFIT ceiling (same set)={ceiling:.3f}  (expect ~1.0)", flush=True)
    print("=" * 100, flush=True)

    if a.mode == "readout_sanity":
        mode_readout_sanity(a.seed, n_bits, Xbal, a.settle, a.big_w)
    elif a.mode == "single_layer":
        mode_single_layer(a.seed, n_bits, Xbal, ybal, a.settle, a.epochs, a.teach_steps)
    elif a.mode == "settled_credit":
        mode_settled_credit(a.seed, n_bits, Xbal, ybal, a.settle, a.epochs, a.teach_on_steps, a.teach_steps)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
