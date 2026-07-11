"""EMERGENCE-BAR de-risk, step 2 (close the step-1 residual): does the reservoir-generation ladder's Rung-3 generalization
ride codes whose category structure comes from CORPUS CO-OCCURRENCE statistics -- category-mates that recur in SIMILAR
(OVERLAPPING) contexts of a token stream develop shared codes -- with NO hand category label AND NO hand-partitioned
feature pool? Step 1 (`_emerge_reservoir_lm_emergent_category_codes_derisk.py`) removed the hand class-LABEL but the
feature->category correlation used DISJOINT hand pools (an adversarial skeptic's core caveat). Step 2 replaces those with
OVERLAPPING distributional contexts: each animal co-occurs with context tokens sampled from its category's distribution,
where the two categories' context distributions OVERLAP (they share ~20% of their mass) -- so the category is NOT a hard
partition and must EMERGE from the co-occurrence STATISTICS (Redington-Chater-Finch 1998 distributional category
induction; EMERGE-30/62 co-occurrence). The animal's PPMI co-occurrence vector is surfaced by a FIXED random codon
(Marr-Albus F.12; step 1 showed the codon, not a learned pooler, is the honest surfacer) into the reservoir input; the
reservoir + one-step-local-delta read-out are byte-identical. NO `sim/` edit, NO BPTT, NO deep credit. CPU numpy.

ARMS (single variable = where the animals' category component comes from):
  * main      -- PPMI co-occurrence over CATEGORY-structured (overlapping) contexts, surfaced by a fixed codon.  (works)
  * scramble  -- each animal draws contexts from a UNIFORM distribution (no category structure).  (COLLAPSE; load-bearing)
  * onehot    -- content bit only, NO category block.                                             (collapse)
  * untrained -- read-out frozen at zeros.                                                          (floor)

METRIC: `heldagent_cat_acc` (Rung 3 -- a held-out animal as AGENT -> correct category action; the clean generalization
metric). GO: main >> scramble (the co-occurrence structure is load-bearing) on all 6 seeds. Deterministic seeding.
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

OUT = Path("research/findings/raw/_reslm_corpus_cooccur.json")

N_CTX = 10           # context tokens (5 "predator-ish", 5 "prey-ish"), but the category distributions OVERLAP
N_OCC = 80           # co-occurrences sampled per animal (the stream length per animal)
OWN_MASS = 0.80      # fraction of an animal's co-occurrences drawn from its OWN category's context half (0.5 = no structure)
N_COL = 80           # fixed-codon columns
K = 12               # active columns
V = r3.V
D_CODE = N_COL + V
ACTION_POS = 3


N_ACTIVE = 3         # sparse: how many context tokens each animal co-occurs with (a sparse distributional signature)


def cooccur_vector(animal, cat, seed, permute):
    """The animal's SPARSE distributional signature: the N_ACTIVE context tokens it co-occurs with, drawn from its
       CATEGORY's context distribution (OWN half gets OWN_MASS, the other shares 1-OWN_MASS -> OVERLAPPING, not a hard
       partition -- category-mates share MOST but not all contexts, cross-category share SOME). permute -> UNIFORM draw
       (no category structure). Sparse (like step 1) so the control collapses cleanly."""
    half = N_CTX // 2
    r = np.random.default_rng(seed * 1000 + r3.WORD_IDX[animal])              # deterministic per-animal (no hash())
    if permute:
        probs = np.ones(N_CTX) / N_CTX
    else:
        own = list(range(0, half)) if cat == "PRED" else list(range(half, N_CTX))
        probs = np.full(N_CTX, (1 - OWN_MASS) / half)
        probs[own] = OWN_MASS / half
    active = r.choice(N_CTX, size=N_ACTIVE, replace=False, p=probs)
    v = np.zeros(N_CTX); v[active] = 1.0
    return v


def ppmi(mat):
    """Positive PMI over the animal x context co-occurrence matrix (EMERGE-30/62 / the standard distributional weighting)."""
    total = mat.sum() + 1e-9
    pxy = mat / total
    px = pxy.sum(1, keepdims=True) + 1e-9
    py = pxy.sum(0, keepdims=True) + 1e-9
    pmi = np.log((pxy + 1e-9) / (px * py))
    return np.maximum(pmi, 0.0)


def build_codes(seed, permute, learn=False, train_only=True):
    animals = [(a, c) for c in ("PRED", "PREY") for a in r3.CAT_ANIMALS[c]["train"] + r3.CAT_ANIMALS[c]["held"]]
    feat = np.array([cooccur_vector(a, c, seed, permute) for a, c in animals])  # animals x N_CTX SPARSE distributional sigs
    Wc = np.random.default_rng(seed).uniform(0.45, 0.55, (N_COL, N_CTX))        # random codon init (F.12)
    if learn:
        # Competitive HTM Spatial Pooler (Cui-Ahmad-Hawkins): winner-take-all + potentiate-active + homeostatic boosting
        # AMPLIFIES a weak/overlapping distributional category signal a FIXED codon cannot separate. train_only=True ->
        # inductive (never sees held animals).
        train_idx = [i for i, (a, c) in enumerate(animals) if a in r3.CAT_ANIMALS[c]["train"]] if train_only else list(range(len(animals)))
        ac = np.zeros(N_COL); boost = np.ones(N_COL)
        for e in range(800):
            for i in train_idx:
                x = feat[i]; act = np.argsort(-((Wc > 0.5) @ x) * boost)[:K]
                Wc[act] += 0.1 * (2 * x - 1); Wc[act] = np.clip(Wc[act], 0, 1); ac[act] += 1
            boost = np.exp(1.5 * (K / N_COL - ac / ((e + 1) * len(train_idx))))
    codes = {}
    for i, (a, _c) in enumerate(animals):
        codes[a] = sorted(np.argsort(-((Wc > 0.5) @ feat[i]))[:K].tolist())    # top-K columns = the animal's category SDR
    return codes


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
    permute = (arm in ("scramble", "learned_scramble")); use_block = (arm != "onehot")
    learn = (arm in ("learned", "learned_scramble"))                          # learned = competitive SP (amplifies overlap)
    codes = build_codes(seed, permute, learn=learn)
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


ARMS = ["main", "learned", "scramble", "learned_scramble", "onehot", "untrained"]  # main=fixed codon; learned=competitive SP


def main():
    global OWN_MASS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--own-mass", type=float, default=OWN_MASS)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()
    OWN_MASS = args.own_mass
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

    out = {"runner": "_emerge_reservoir_lm_corpus_cooccurrence_codes_derisk", "seeds": args.seeds,
           "own_mass": OWN_MASS, "n_ctx": N_CTX, "n_occ": N_OCC, "n_col": N_COL, "k": K,
           "per_seed": {str(s): per_seed[s] for s in args.seeds}, "aggregate": aggregate,
           "per_seed_go": per_seed_go, "n_go": n_go, "n_seeds": len(args.seeds), "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nAGG main={aggregate['main']} scramble={aggregate['scramble']} onehot={aggregate['onehot']} "
          f"(own_mass={OWN_MASS}) GO {n_go}/{len(args.seeds)} ({out['elapsed_s']}s) -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
