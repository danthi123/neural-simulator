"""gap#4 -> gap#1 BRIDGE: does the transport-free BIOLOGICAL deep-credit rule train a GENERATIVE
NEXT-TOKEN sequence model to accuracy, BEATING a frozen reservoir?  ("the brain's own Broca needs a
credit rule, not backprop.")

This REUSES the gap#4 enabler's credit machinery BY IMPORT -- `BdspNet` (a `sim.dendritic_mlp.DendriticMLP`
subclass with the FAITHFUL on-bridge BDSP rule + feedback-alignment + a frozen-RESERVOIR arm, all in one
`train_step`) is imported byte-for-byte from `_gap4_bdsp_faithful_credit_derisk`; the credit rule is NOT
reimplemented here.  The only thing added is a SEQUENCE WRAPPER: a fixed-context-window autoregressive
char-level next-token task over a real corpus stream (tinystories).  Input = one-hot of the previous k
chars; target = the next char.  The model is generative (roll it out to sample) and every hidden layer is
the same sparse-binary "spike raster" the enabler used (frac<1 -> top-k active), so the SAME
"gap-grows-with-spiking-sparsity" test transfers directly.

The reservoir baseline in a sequence context is exactly the classic echo-state / reservoir-computing
setup (frozen random hidden, only the readout trained) -- the credit-INDEPENDENT sequence baseline.

GO gate (read the runner's OWN printed VERDICT line, not a lifted field):
  credit (BDSP or FA) beats the reservoir on HELD-OUT next-token accuracy, at every sparsity, all 6 seeds
  (42 43 44 100 101 102), AND the credit>reservoir gap GROWS as spiking gets sparser (mirrors the enabler
  +0.155 dense -> +0.299 at 5%-active), AND the shuffled-target anti-cheat collapses to the unigram floor.

Anti-cheats (ALL WIRED AND INVOKED in the run path -- a control written-but-never-called is the #1 silent
failure; each is asserted-live below):
  (1) RESERVOIR baseline (frozen hidden, readout-only) -- run as an arm every frac.
  (2) SHUFFLED-TARGET control -- the best-credit arm trained with the next-char labels permuted (breaks the
      input->target map); evaluated on the REAL held-out; must collapse to ~unigram accuracy.
  (3) SEED-BUG GUARD -- everything is seeded via numpy default_rng(seed) + BdspNet(seed=); there is NO
      SimulationBridge / cfg here, so the 2026-07-17 unseeded-substrate confound cannot touch it.  A live
      self-check builds two nets at the same seed and asserts byte-identical weights (seed DOES control the
      substrate), and asserts a different seed changes them.  (We NEVER read cfg.actual_seed_used.)

numpy CPU (SIM_BACKEND=numpy by default; pass SIM_BACKEND=cupy to force GPU).  Small enough to CPU-smoke.
NO sim/ edit (reuse-by-import only).  ASCII only.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")  # match the enabler; caller may override with SIM_BACKEND=cupy

import argparse
import json
import time

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
# --- REUSE-BY-IMPORT: the credit rule (reservoir / fa_linear / fa_coinc / bdsp_nocoinc / bdsp) is the
#     enabler's, unmodified.  We do NOT redefine train_step here. ---
from research.runners._gap4_bdsp_faithful_credit_derisk import BdspNet, _sparsify  # noqa: E402,F401

xp, _ = get_backend()

# Core arms compared for the GO gate.  "credit" = best of {fa_linear, bdsp}; "reservoir" = credit-independent.
CORE_ARMS = ["reservoir", "fa_linear", "bdsp"]
CREDIT_ARMS = ["fa_linear", "bdsp"]


# ----------------------------------------------------------------------------- data / sequence wrapper
def _load_char_ids(corpus_path, max_chars, vocab=None):
    """Read a lowercase char stream, map to a compact vocab of ids. Returns (ids, vocab, itos)."""
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(max_chars)
    if vocab is None:
        # frozen, sorted -> deterministic index assignment (vocab is NOT seed-dependent)
        vocab = {c: i for i, c in enumerate(sorted(set(text)))}
    itos = {i: c for c, i in vocab.items()}
    ids = np.frombuffer(bytes([vocab[c] if c in vocab else 0 for c in text]), dtype=np.uint8) \
        if len(vocab) <= 256 else np.array([vocab.get(c, 0) for c in text], dtype=np.int64)
    return np.asarray(ids, dtype=np.int64), vocab, itos


def _make_windows(ids_seg, k, V, n, rng):
    """Sample n windows from a contiguous char segment. X = one-hot of previous k chars (flattened),
    y = next char id.  Windows are drawn from ONE segment so train/test never share chars (held-out)."""
    hi = len(ids_seg) - k - 1
    if hi <= 0:
        raise ValueError("segment too short for window k=%d" % k)
    n = min(n, hi)
    starts = rng.choice(hi, size=n, replace=False)
    X = np.zeros((n, k * V), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    for r, s in enumerate(starts):
        w = ids_seg[s:s + k]
        X[r, np.arange(k) * V + w] = 1.0
        y[r] = ids_seg[s + k]
    return X, y


def build_sequence_task(a, seed):
    """A held-out next-token dataset: split the raw char stream into a train segment and a DISJOINT test
    segment (a gap between them blocks boundary leakage), then sample windows within each."""
    ids, vocab, itos = _load_char_ids(a.corpus, a.max_chars)
    V = len(vocab)
    rng = np.random.default_rng(seed)
    split = int(0.8 * len(ids))
    gap = a.window + 8  # >k so no test window can peek into train chars
    train_ids, test_ids = ids[:split], ids[split + gap:]
    Xtr, ytr = _make_windows(train_ids, a.window, V, a.n_train, rng)
    Xte, yte = _make_windows(test_ids, a.window, V, a.n_test, rng)
    # unigram floor: best constant predictor = the most frequent next-char in TRAIN, scored on TEST.
    counts = np.bincount(ytr, minlength=V)
    unigram_pred = int(np.argmax(counts))
    unigram_acc = float(np.mean(yte == unigram_pred))
    return Xtr, ytr, Xte, yte, V, vocab, itos, unigram_acc


# ----------------------------------------------------------------------------- metrics
def _ppl(net, X, y):
    """Held-out perplexity = exp(mean NLL). Reads the net's forward; does NOT touch the credit rule."""
    _, _, lg = net._forward(xp.asarray(X, float))
    p = BdspNet._softmax(lg)
    yy = xp.asarray(y)
    nll = -xp.log(p[xp.arange(len(yy)), yy] + 1e-12).mean()
    return float(to_host(xp.exp(nll)))


def _train_eval(mode, Xtr, ytr, Xte, yte, sizes, seed, frac, a, shuffle_targets=False):
    net = BdspNet(sizes, seed=seed, frac=frac, p0=a.p0, beta=a.beta)
    rng = np.random.default_rng(seed * 131 + 7)
    ytr_used = ytr
    if shuffle_targets:
        # ANTI-CHEAT (2): permute the next-char labels -> input->target map is destroyed.
        ytr_used = ytr[rng.permutation(len(ytr))]
    for _ in range(a.epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), a.batch):
            idx = order[s:s + a.batch]
            net.train_step(Xtr[idx], ytr_used[idx], mode, a.lr)
    return net.accuracy(Xte, yte), _ppl(net, Xte, yte)


# ----------------------------------------------------------------------------- seed-bug guard (LIVE)
def _seed_guard(sizes, seed, frac, a):
    """ANTI-CHEAT (3): prove the seed controls the substrate (the 2026-07-17 gotcha). Two nets at the SAME
    seed must be byte-identical; a DIFFERENT seed must differ. Returns a dict; raises on failure."""
    n1 = BdspNet(sizes, seed=seed, frac=frac, p0=a.p0, beta=a.beta)
    n2 = BdspNet(sizes, seed=seed, frac=frac, p0=a.p0, beta=a.beta)
    n3 = BdspNet(sizes, seed=seed + 1, frac=frac, p0=a.p0, beta=a.beta)
    same = bool(to_host(xp.all(n1.W[0] == n2.W[0])) and to_host(xp.all(n1.B[0] == n2.B[0])))
    diff = bool(to_host(xp.any(n1.W[0] != n3.W[0])))
    if not (same and diff):
        raise AssertionError("SEED GUARD FAILED: same-seed identical=%s diff-seed changes=%s" % (same, diff))
    return dict(same_seed_identical=same, diff_seed_changes=diff)


# ----------------------------------------------------------------------------- per-seed run
def one_seed(seed, a):
    Xtr, ytr, Xte, yte, V, vocab, itos, unigram_acc = build_sequence_task(a, seed)
    sizes = [a.window * V] + [a.hidden] * a.depth + [V]
    guard = _seed_guard(sizes, seed, a.fracs[0], a)  # INVOKED, live, per seed
    print(f"  [seed {seed}] V={V} sizes={sizes} unigram_floor={unigram_acc:.3f} "
          f"seed_guard(same={guard['same_seed_identical']},diff={guard['diff_seed_changes']})")
    rows = []
    for frac in a.fracs:
        acc, ppl = {}, {}
        for m in CORE_ARMS:
            acc[m], ppl[m] = _train_eval(m, Xtr, ytr, Xte, yte, sizes, seed, frac, a)
        # ANTI-CHEAT (2) INVOKED: best-credit arm on SHUFFLED targets (only at the sparsest frac to save time
        # unless --shuffle-all-fracs). Must collapse toward the unigram floor.
        shuf_acc = shuf_ppl = None
        if a.shuffle_all_fracs or frac == a.fracs[-1]:
            shuf_acc, shuf_ppl = _train_eval("bdsp", Xtr, ytr, Xte, yte, sizes, seed, frac, a,
                                             shuffle_targets=True)
        best_credit = max(acc[m] for m in CREDIT_ARMS)
        rows.append(dict(frac=frac, unigram=unigram_acc,
                         **{f"acc_{k}": round(float(v), 4) for k, v in acc.items()},
                         **{f"ppl_{k}": round(float(v), 3) for k, v in ppl.items()},
                         acc_bdsp_shuffled=(round(float(shuf_acc), 4) if shuf_acc is not None else None),
                         ppl_bdsp_shuffled=(round(float(shuf_ppl), 3) if shuf_ppl is not None else None),
                         best_credit=round(float(best_credit), 4),
                         gap=round(float(best_credit - acc["reservoir"]), 4)))
        sh = f" shuf={shuf_acc:.3f}" if shuf_acc is not None else ""
        print(f"  [seed {seed}] frac={frac:.2f}: RES={acc['reservoir']:.3f} fa_lin={acc['fa_linear']:.3f} "
              f"bdsp={acc['bdsp']:.3f} best_credit={best_credit:.3f} gap={best_credit - acc['reservoir']:+.3f}"
              f"{sh} | ppl RES={ppl['reservoir']:.2f} best={min(ppl['fa_linear'], ppl['bdsp']):.2f}")
    return dict(seed=seed, V=V, sizes=sizes, unigram_acc=unigram_acc, seed_guard=guard, rows=rows)


# ----------------------------------------------------------------------------- verdict (the runner's OWN)
def verdict(per, a):
    seeds = [p["seed"] for p in per]
    n = len(seeds)
    fracs = a.fracs
    # mean over seeds per frac
    agg = []
    for i, frac in enumerate(fracs):
        res = float(np.mean([p["rows"][i]["acc_reservoir"] for p in per]))
        best = float(np.mean([p["rows"][i]["best_credit"] for p in per]))
        # per-seed credit>reservoir at this frac
        n_win = sum(1 for p in per if p["rows"][i]["best_credit"] > p["rows"][i]["acc_reservoir"] + a.margin)
        agg.append(dict(frac=frac, reservoir=round(res, 4), best_credit=round(best, 4),
                        gap=round(best - res, 4), seeds_credit_gt_res=f"{n_win}/{n}"))
    # GO conditions
    credit_gt_res_all_frac = all(g["best_credit"] > g["reservoir"] for g in agg)
    all_seeds_win_sparse = all(
        (sum(1 for p in per if p["rows"][i]["best_credit"] > p["rows"][i]["acc_reservoir"] + a.margin) == n)
        for i in range(len(fracs)))
    gap_dense = agg[0]["gap"]
    gap_sparse = agg[-1]["gap"]
    gap_grows = gap_sparse > gap_dense
    # shuffle collapse: shuffled-bdsp acc <= unigram_floor + slack, AND well below best_credit, every seed (at
    # the fracs where it was run).
    shuf_ok = True
    for p in per:
        for i, frac in enumerate(fracs):
            sa = p["rows"][i]["acc_bdsp_shuffled"]
            if sa is None:
                continue
            uni = p["rows"][i]["unigram"]
            bc = p["rows"][i]["best_credit"]
            if not (sa <= uni + a.shuffle_slack and sa < bc - a.margin):
                shuf_ok = False
    conds = dict(credit_gt_res_all_frac=credit_gt_res_all_frac,
                 all_seeds_win_sparse=all_seeds_win_sparse,
                 gap_grows_with_sparsity=gap_grows,
                 shuffle_collapses=shuf_ok)
    is_go = (n >= 6) and all(conds.values())
    tag = "GO" if is_go else ("SMOKE (n<6): not a GO/NO-GO verdict" if n < 6 else "NO-GO")
    return dict(tag=tag, is_go=is_go, n_seeds=n, agg=agg, conds=conds,
                gap_dense=gap_dense, gap_sparse=gap_sparse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--max-chars", type=int, default=2_000_000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.1, 0.05])
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.03)   # the enabler's validated lr (NOT the 0.3 dense-artifact default)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--shuffle-slack", type=float, default=0.03)
    ap.add_argument("--shuffle-all-fracs", action="store_true")
    ap.add_argument("--out", default="research/findings/raw/gap4_seq/seq_deep_credit.json")
    a = ap.parse_args()
    _, backend = get_backend()
    knobs = vars(a).copy()
    print(f"[gap4->gap1] deep-credit trains a NEXT-TOKEN sequence model vs a frozen reservoir | "
          f"corpus={a.corpus} window={a.window} hidden={a.hidden} depth={a.depth} fracs={a.fracs} "
          f"lr={a.lr} epochs={a.epochs} n_train={a.n_train} seeds={a.seeds} backend={backend}")
    t0 = time.time()
    per = [one_seed(s, a) for s in a.seeds]
    v = verdict(per, a)
    print("[gap4->gap1] SUMMARY (mean over seeds):")
    for g in v["agg"]:
        print(f"  frac={g['frac']:.2f}: RES={g['reservoir']:.3f} best_credit={g['best_credit']:.3f} "
              f"gap={g['gap']:+.3f} seeds_credit>res={g['seeds_credit_gt_res']}")
    print(f"[gap4->gap1] gap_dense={v['gap_dense']:+.3f} gap_sparse={v['gap_sparse']:+.3f} "
          f"(gap grows with sparsity: {v['conds']['gap_grows_with_sparsity']})")
    # THE runner's OWN verdict line (the GO gate parses THIS, not a lifted field):
    print(f"[gap4->gap1] VERDICT: {v['tag']} | credit>res all-frac:{v['conds']['credit_gt_res_all_frac']} "
          f"| all-seeds-win-sparse:{v['conds']['all_seeds_win_sparse']} "
          f"| gap-grows:{v['conds']['gap_grows_with_sparsity']} "
          f"| shuffle-collapses:{v['conds']['shuffle_collapses']}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(knobs=knobs, backend=backend, elapsed_s=round(time.time() - t0, 1),
                       verdict=v, per=per), f, indent=2)
    print(f"[gap4->gap1] wrote {a.out} ({round(time.time() - t0, 1)}s)")


if __name__ == "__main__":
    main()
