"""gap#4 rank-1 escape — HYPOTHESIS TEST (numpy, cheapest-first): does a FIXED Marr-Albus sparse coincidence EXPANSION
make the semantic-inheritance task HELD-OUT-LINEARLY separable? The research gate's decisive prediction: the gap#4 forward
adds ZERO linearly-usable structure (hidden ho-lin 0.284 == input ho-lin 0.284) because every lever attacked
compression/drive/credit/readout, never the missing NONLINEAR EXPANSION. A fixed coincidence expansion (each of N_COL
columns samples SAMP features, fires if >= ACT_TH active) is a random nonlinear kernel that should LINEARIZE the task.
GO iff codon ho-LINEAR rises materially above the INPUT linear ceiling 0.284 (and the failing H2 0.34), toward the
0.988 mlp ceiling, on >=5/6 seeds. If numpy confirms, escalate to the SPIKING EMERGE-35 codon bridge.
Anti-cheats (research-gate-specified): (1) winning readout must be LINEAR; (2) same-dim NON-expanding coincidence must
NOT lift (isolates expansion from the coincidence nonlinearity); (3) permuted-features -> chance; (4) HELD-OUT only,
binarization threshold fit on TRAIN only (no leakage)."""
import os, sys, time
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance


def _sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_lin(X, y, k, iters=600, lr=0.5, l2=3e-3):
    n, d = X.shape; W = np.zeros((d, k)); b = np.zeros(k); Y = np.eye(k)[y]
    for _ in range(iters):
        P = _sm(X @ W + b); g = (P - Y) / n; W -= lr * (X.T @ g + l2 * W); b -= lr * g.sum(0)
    return lambda Z: np.argmax(Z @ W + b, 1)


def fit_mlp(X, y, k, h=64, iters=1500, lr=0.2, l2=1e-3, seed=0):
    n, d = X.shape; rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d, h)) / np.sqrt(d); b1 = np.zeros(h)
    W2 = rng.standard_normal((h, k)) / np.sqrt(h); b2 = np.zeros(k); Y = np.eye(k)[y]
    for _ in range(iters):
        Z1 = X @ W1 + b1; A1 = np.maximum(Z1, 0.0); P = _sm(A1 @ W2 + b2)
        gO = (P - Y) / n; gW2 = A1.T @ gO + l2 * W2; gb2 = gO.sum(0)
        gZ1 = (gO @ W2.T) * (Z1 > 0); gW1 = X.T @ gZ1 + l2 * W1; gb1 = gZ1.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2

    def f(Z):
        A = np.maximum(Z @ W1 + b1, 0.0); return np.argmax(A @ W2 + b2, 1)
    return f


def _acc(f, X, y):
    return float(np.mean(f(X) == y))


def binarize(Xtr, Xall, topk):
    """per-input top-k active features (no cross-input leakage). Returns bool arrays."""
    def tk(X):
        th = np.sort(X, axis=1)[:, -topk][:, None]
        return (X >= th)
    return tk(Xtr), tk(Xall)


def codon(active, W_sample, act_th):
    """active: (n, n_in) bool; W_sample: (N_COL, n_in) bool (each column's sampled features). -> (n, N_COL) float codon."""
    counts = active.astype(np.float64) @ W_sample.T.astype(np.float64)   # how many sampled features are active per column
    return (counts >= act_th).astype(np.float64)


SEEDS = (42, 43, 44, 100, 101, 102)
SAMP, ACT_TH, TOPK, N_COL = 3, 2, 4, 200        # coincidence: 3-of-7 sampled, fire>=2; top-4 active; 200 columns
ROWS = ["INPUT", "RANDFEAT-ReLU expand", "RANDFEAT non-expand(ctrl)", "RANDFEAT permuted(ctrl)",
        "CODON expand", "CODON permuted(ctrl)"]
agg = {r: {"lin": [], "mlp": []} for r in ROWS}
for SEED in SEEDS:
    t0 = time.time()
    (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(
        SEED, n_super=12, n_members=8, held_per_super=3, n_prop=2, n_obs=16, member_id_dim=3, noise=0.02)
    n_in = Xtr.shape[1]; k = meta["k_classes"]; inh = idx["inh_idx"]
    srng = np.random.default_rng(SEED * 13 + 1); keep = srng.permutation(len(Xtr))[:96]
    Xb, yb = Xtr[keep], ytr[keep]; Xh, yh = Xte[inh], yte[inh]
    rng = np.random.default_rng(SEED * 991 + 7)
    perm = rng.permutation(n_in)
    # --- RANDOM-FEATURE (ReLU) expansion: the continuous nonlinear-kernel instance ---
    Wr = rng.standard_normal((n_in, N_COL)) / np.sqrt(n_in); br = rng.standard_normal(N_COL) * 0.5
    Wr_ne = rng.standard_normal((n_in, n_in)) / np.sqrt(n_in); br_ne = rng.standard_normal(n_in) * 0.5
    relu = lambda X, W, b: np.maximum(X @ W + b, 0.0)
    # --- COINCIDENCE codon (EMERGE-35 instance) on top-k binarized input ---
    Ws = np.zeros((N_COL, n_in), bool)
    for c in range(N_COL):
        Ws[c, rng.choice(n_in, SAMP, replace=False)] = True
    _, ab_all = binarize(Xb, np.vstack([Xb, Xh]), TOPK)
    ab_b, ab_h = ab_all[:len(Xb)], ab_all[len(Xb):]
    reps = {
        "INPUT": (Xb, Xh),
        "RANDFEAT-ReLU expand": (relu(Xb, Wr, br), relu(Xh, Wr, br)),
        "RANDFEAT non-expand(ctrl)": (relu(Xb, Wr_ne, br_ne), relu(Xh, Wr_ne, br_ne)),
        "RANDFEAT permuted(ctrl)": (relu(Xb[:, perm], Wr, br), relu(Xh[:, perm], Wr, br)),
        "CODON expand": (codon(ab_b, Ws, ACT_TH), codon(ab_h, Ws, ACT_TH)),
        "CODON permuted(ctrl)": (codon(ab_b[:, perm], Ws, ACT_TH), codon(ab_h[:, perm], Ws, ACT_TH)),
    }
    for r, (Rtr, Rho) in reps.items():
        fl = fit_lin(Rtr, yb, k); fm = fit_mlp(Rtr, yb, k, seed=SEED)
        agg[r]["lin"].append(_acc(fl, Rho, yh)); agg[r]["mlp"].append(_acc(fm, Rho, yh))
    # VALID anti-cheat: LABEL-SHUFFLE on the expander (train on shuffled labels -> held-out MUST be chance)
    ysh = rng.permutation(yb); Re_tr, Re_ho = reps["RANDFEAT-ReLU expand"]
    flsh = fit_lin(Re_tr, ysh, k); fmsh = fit_mlp(Re_tr, ysh, k, seed=SEED)
    agg.setdefault("RANDFEAT label-shuffle(ctrl)", {"lin": [], "mlp": []})
    agg["RANDFEAT label-shuffle(ctrl)"]["lin"].append(_acc(flsh, Re_ho, yh))
    agg["RANDFEAT label-shuffle(ctrl)"]["mlp"].append(_acc(fmsh, Re_ho, yh))
    print(f"  seed {SEED} ({time.time()-t0:.0f}s) n_in={n_in} N_COL={N_COL} codon_spars={reps['CODON expand'][0].mean():.3f} chance={1.0/k:.2f} n_ho={len(inh)}", flush=True)

print("\n===== gap#4 EXPANSION held-out (mean over 6 seeds) — GO metric = ho-LINEAR rises off input 0.284 =====", flush=True)
for r in ROWS + ["RANDFEAT label-shuffle(ctrl)"]:
    L = np.array(agg[r]["lin"]); M = np.array(agg[r]["mlp"])
    print(f"  {r:26s} ho-lin {L.mean():.3f}+/-{L.std():.3f}   ho-mlp {M.mean():.3f}+/-{M.std():.3f}", flush=True)
for exp, ne in [("RANDFEAT-ReLU expand", "RANDFEAT non-expand(ctrl)")]:
    Lc = np.array(agg[exp]["lin"]); Ln = np.array(agg[ne]["lin"])
    print(f"\n{exp}: ho-lin per-seed {[round(x,3) for x in Lc]} ({int(np.sum(Lc>0.40))}/6 > 0.40); "
          f"expand {Lc.mean():.3f} vs non-expand {Ln.mean():.3f} ({Lc.mean()-Ln.mean():+.3f})", flush=True)
print("\nCODON-NUMPY DONE", flush=True)
