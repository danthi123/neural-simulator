"""EMERGENCE-BAR de-risk, step 3 (the STRONG, master-directive-aligned close): does the reservoir-generation ladder's
Rung-3 generalization ride codes whose category structure was DISCOVERED from REAL PERCEPTION -- objects SEEN through the
project's Gabor/V1 visual front end -- with NO hand category label, NO hand feature pool, and NO symbolic token? Step 1
removed the hand class-LABEL but used disjoint hand feature pools; step 2 (corpus co-occurrence) was scale-limited on the
toy. Step 3 grounds the category in PERCEPTION: each animal is rendered as an object SHAPE, encoded through the real
retina->V1 Gabor bank (`encode_v1`); same-category shapes overlap in V1 features (EMERGE-34: within-cat ~0.25, cross-cat
~0.00 -- a strong signal the perception PRESERVES), surfaced by a fixed random codon (F.12; step 1 showed the codon, not a
learned pooler, is the honest surfacer) into the reservoir input. The reservoir + one-step-local-delta read-out are
byte-identical. NO `sim/` edit, NO BPTT, NO deep credit. Reuse-by-import: EMERGE-34's shape/Gabor/V1 machinery + the
Rung-3 grammar/reservoir. "The brain LEARNS what a category IS by looking, then reasons about it."

ANTI-CHEATS (single variable = where the category comes from): scramble = PER-IMAGE PIXEL SCRAMBLE (destroys the within-
category VISUAL similarity -> no category -> COLLAPSE; isolates the VISUAL shape as the cause); onehot = no category
block; untrained = frozen read-out. METRIC: `heldagent_cat_acc` (a held-out perceived animal as AGENT -> correct category
action). GO: main >> scramble (the visual similarity is load-bearing) on all 6 seeds. CPU numpy.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

import research.runners._emerge_reservoir_lm_rung3_systematic_generation_derisk as r3
from research.runners._emerge34_perception_grounded_emergence_derisk import (
    build_shape_set, build_gabor_response_matrix, encode_v1)

OUT = Path("research/findings/raw/_reslm_perception_grounded.json")

N_EX = 9             # exemplars per visual category (>= 6 train + 3 held animals/category)
T_ACTIVE = 20        # top-T active V1 cells = each object's perception feature
N_COL = 80
K = 12
V = r3.V
D_CODE = N_COL + V
ACTION_POS = 3

_GABOR = None
def _gabor():
    global _GABOR
    if _GABOR is None:
        _GABOR = build_gabor_response_matrix()
    return _GABOR


def perception_features(seed, scramble):
    """Render the Rung-3 animals as object SHAPES, SEE them through the real Gabor/V1 front end, return a sparse binary
       perception vector per animal (top-T active V1 cells). scramble = per-image pixel permutation (destroys visual
       similarity). Maps PRED animals -> visual category 0's exemplars, PREY -> category 1's, deterministically."""
    rng = np.random.default_rng(seed)
    imgs, labels, _ = build_shape_set(n_categories=2, n_exemplars=N_EX, rng=rng)
    imgs = np.asarray(imgs)
    if scramble:
        r = np.random.default_rng(seed * 7 + 1)
        imgs = np.stack([im.flatten()[r.permutation(im.size)].reshape(im.shape) for im in imgs])
    Vv = encode_v1(imgs, _gabor())                                             # object x V1-feature responses
    NF = Vv.shape[1]
    cat_imgs = {0: [i for i in range(len(labels)) if labels[i] == 0],
                1: [i for i in range(len(labels)) if labels[i] == 1]}
    feat = {}
    for ci, cat in enumerate(("PRED", "PREY")):
        animals = r3.CAT_ANIMALS[cat]["train"] + r3.CAT_ANIMALS[cat]["held"]
        for j, a in enumerate(animals):
            v1 = Vv[cat_imgs[ci][j]]                                           # this animal's SEEN shape -> V1 response
            top = np.argsort(-v1)[:T_ACTIVE]
            x = np.zeros(NF); x[top] = 1.0                                     # sparse perception code (top-T active V1 cells)
            feat[a] = x
    return feat, NF


def build_codes(seed, scramble):
    feat, NF = perception_features(seed, scramble)
    Wc = np.random.default_rng(seed).uniform(0.45, 0.55, (N_COL, NF))          # FIXED random codon (F.12; no learning)
    return {a: sorted(np.argsort(-((Wc > 0.5) @ feat[a]))[:K].tolist()) for a in feat}


def word_code(w, codes, use_block):
    v = np.zeros(D_CODE); v[N_COL + r3.WORD_IDX[w]] = 1.0
    if use_block and w in r3.ANIMAL_CAT:
        for c in codes[w]:
            v[c] = 1.0
    return v


def encode(sent, codes, use_block):
    return np.asarray([word_code(w, codes, use_block) for w in sent])


def cum_feat(res, prefix, codes, use_block):
    return res.per_token_states(encode(prefix, codes, use_block), feature="running_cumulative")[ACTION_POS - 1]


def _train(feats, tgts, ncls, epochs, lr, seed):
    X = np.array(feats); mean = X.mean(0); std = X.std(0) + 1e-6
    Xn = np.concatenate([(X - mean) / std, np.ones((len(X), 1))], 1)
    W = np.zeros((ncls, Xn.shape[1])); rng = np.random.default_rng(seed * 13 + 1); idx = list(range(len(Xn)))
    Ws = np.zeros_like(W); na = 0; burn = epochs // 2
    for ep in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            z = W @ Xn[i]; z = z - z.max(); p = np.exp(z); p /= p.sum()
            t = np.zeros(ncls); t[tgts[i]] = 1.0; W += lr * np.outer(t - p, Xn[i])
        if ep >= burn:
            Ws += W; na += 1
    return (Ws / na if na else W), mean, std


def run_arm(seed, arm, epochs, lr, n_pool):
    scramble = (arm == "scramble"); use_block = (arm != "onehot")
    codes = build_codes(seed, scramble)
    res = r3.ReservoirStates(D_CODE, seed=seed, n=n_pool)
    feats = [cum_feat(res, s[:3], codes, use_block) for s in r3.TRAIN_SENTS]
    tgts = [r3.WORD_IDX[s[3]] for s in r3.TRAIN_SENTS]
    if arm == "untrained":
        W = np.zeros((V, len(feats[0]) + 1)); m = np.zeros(len(feats[0])); sd = np.ones(len(feats[0]))
    else:
        W, m, sd = _train(feats, tgts, V, epochs, lr, seed)

    def pred(prefix):
        f = cum_feat(res, prefix, codes, use_block); x = np.concatenate([(f - m) / sd, [1.0]])
        return r3.WORDS[int(np.argmax(W @ x))]
    ok = tot = 0
    for prefix, true_cat in r3.HELD_PREFIXES:
        p = pred(prefix); ok += int(p in r3.ACTION_CAT and r3.ACTION_CAT[p] == true_cat); tot += 1
    return {"arm": arm, "heldagent_cat_acc": ok / tot}


ARMS = ["main", "scramble", "onehot", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()
    train_tokens = {w for s in r3.TRAIN_SENTS for w in s}
    for prefix, _ in r3.HELD_PREFIXES:
        assert prefix[0] not in train_tokens, f"held agent {prefix[0]} leaked"

    t0 = time.time(); per_seed = {}
    for seed in args.seeds:
        rb = {}
        for arm in ARMS:
            try:
                rb[arm] = run_arm(seed, arm, args.epochs, args.lr, args.n_pool)
            except Exception as e:
                rb[arm] = {"arm": arm, "error": f"{e}", "trace": traceback.format_exc()}
            r = rb[arm]
            print(f"[seed {seed}] {arm:10s} " + (f"heldagent={r.get('heldagent_cat_acc'):.3f}" if "error" not in r else r["error"]), flush=True)
        per_seed[seed] = rb

    def agg(arm):
        vals = [per_seed[s][arm]["heldagent_cat_acc"] for s in args.seeds if "error" not in per_seed[s][arm]]
        return float(np.mean(vals)) if vals else None
    aggregate = {arm: agg(arm) for arm in ARMS}
    per_seed_go = []
    for s in args.seeds:
        rb = per_seed[s]
        if any("error" in rb[a] for a in ARMS):
            per_seed_go.append(False); continue
        m = rb["main"]["heldagent_cat_acc"]; sc = rb["scramble"]["heldagent_cat_acc"]
        per_seed_go.append(bool(m >= 0.75 and (m - sc) >= 0.25))
    n_go = int(sum(per_seed_go))

    out = {"runner": "_emerge_reservoir_lm_perception_grounded_codes_derisk", "seeds": args.seeds,
           "n_ex": N_EX, "t_active": T_ACTIVE, "n_col": N_COL, "k": K,
           "per_seed": {str(s): per_seed[s] for s in args.seeds}, "aggregate": aggregate,
           "per_seed_go": per_seed_go, "n_go": n_go, "n_seeds": len(args.seeds), "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nAGG main={aggregate['main']} scramble={aggregate['scramble']} onehot={aggregate['onehot']} "
          f"GO {n_go}/{len(args.seeds)} ({out['elapsed_s']}s) -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
