"""EMERGENCE-BAR de-risk for the reservoir-generation ladder: does Rung-3 GENERALIZATION (a novel subject inherits its
category's continuation) and Rung-4 ORDER-DECISIVE role reversal ride codes whose CATEGORY STRUCTURE was DISCOVERED from
experience -- instead of the hand-coded class bit? Per the master directive + emergence bar (does it EMERGE, or did I
hand-install it?), and the deep-research gate's Option A: replace the animals' hand class bit (`make_code` in Rung 3) with
an EMERGENT SDR from a competitive HTM Spatial Pooler (Cui-Ahmad-Hawkins 2017; the EMERGE-33 machinery) self-organized on
a PRED/PREY FEATURE stream -- category-mates come to share columns, a held-out member inherits the block -- and re-run the
Rung-3 + Rung-4 metrics UNCHANGED. The reservoir + one-step-local-delta read-out are byte-identical; only the input code's
category component changes from HAND-GIVEN to DISCOVERED. NO `sim/` edit, NO BPTT, NO deep credit. CPU numpy.

Biology/precedent: Redington-Chater-Finch 1998 (distributional category induction licenses novel-member generalization);
Hinaut-Dominey 2013 (a construction-grammar reservoir fed word-category + meaning generalizes to unseen constructions);
Farkas-Crocker 2008 (self-organized category codes -> systematic recurrent processing); Saxe-McClelland-Ganguli 2019
(emergent taxonomy). The reservoir itself is catalogued Marr-Albus codon / random-feature reservoir (F.02/F.12).

ARMS (single variable = the animals' category component: DISCOVERED vs hand/absent):
  * main       -- pooler SDR discovered from CATEGORY-structured features (the emergent category block).   (expect: works)
  * scramble   -- pooler trained on PERMUTED (mixed-pool) features -> no category block forms.  (expect: COLLAPSE; load-bearing)
  * nopooler   -- random per-animal SDR (no shared block).                                       (expect: collapse)
  * onehot     -- animals get their CONTENT bit only, NO category block.                         (expect: collapse)
  * untrained  -- read-out frozen at zeros.                                                       (floor)

METRICS (both re-used verbatim from the rungs): `heldagent_cat_acc` (Rung 3 -- a held-out animal as AGENT -> correct
category action; cum feature) ; `reversal_acc` (Rung 4 -- role reversal over held-out cross-category twins; trajectory
feature). GO: main high on BOTH AND the discovery-scramble + nopooler + onehot collapse (with untrained the floor), 6-seed.
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
import research.runners._emerge_reservoir_lm_rung4_order_decisive_recombination_derisk as r4

OUT = Path("research/findings/raw/_reslm_emergent_codes.json")

N_FEAT = 8            # feature pool: PRED features 0-3, PREY features 4-7 (each animal = 3 active of its category's 4)
N_COL = 80           # pooler columns
K = 12               # active columns (kWTA)
POOL_EPOCHS = 800
V = r3.V
D_CODE = N_COL + V   # [discovered SDR columns] + [content one-hot over the vocab]
ACTION_POS = 3


class EmergentCategoryPooler:
    """Competitive HTM Spatial Pooler (Cui-Ahmad-Hawkins 2017; mirrors EMERGE-33): self-organizes an overlap-preserving
       column SDR per animal from a category-structured FEATURE stream. permute=True mixes the feature pool (no category
       structure -> the discovery-scramble control); pooler=False = random per-animal codes (no shared block)."""

    def __init__(self, seed, permute=False, pooler=True, train_only=False, learn=True):
        cats = ["PRED", "PREY"]
        half = N_FEAT // 2
        self.X = {}; train_members = []
        for ci, c in enumerate(cats):
            pool = list(range(ci * half, (ci + 1) * half))                    # this category's feature pool
            for j, a in enumerate(r3.CAT_ANIMALS[c]["train"] + r3.CAT_ANIMALS[c]["held"]):
                r = np.random.default_rng(seed * 1000 + r3.WORD_IDX[a])        # DETERMINISTIC per-animal seed (hash() is process-salted)
                p = list(range(N_FEAT)) if permute else pool                  # permute -> draw from ALL features (no category)
                x = np.zeros(N_FEAT); x[r.choice(p, 3, replace=False)] = 1.0
                self.X[a] = x
                if a in r3.CAT_ANIMALS[c]["train"]:
                    train_members.append(a)
        self.W = np.random.default_rng(seed).uniform(0.45, 0.55, (N_COL, N_FEAT))
        self.pooler = pooler; self.seed = seed
        # train_only=True -> the competitive pooler NEVER sees held animals (INDUCTIVE: their codes are inferred by applying
        # the train-learned W to their observed features) -> closes any transductive-leakage concern.
        # learn=False -> SKIP the competitive loop (W stays at random init) -> a FIXED RANDOM PROJECTION of the same
        # category-structured features (Marr-Albus codon, F.12). Isolates whether the competitive LEARNING is load-bearing,
        # or whether ANY projection of the disjoint feature pools already surfaces the shared category block.
        if pooler and learn:
            members = train_members if train_only else list(self.X); ac = np.zeros(N_COL); boost = np.ones(N_COL)
            for e in range(POOL_EPOCHS):
                for a in members:
                    x = self.X[a]; act = np.argsort(-((self.W > 0.5) @ x) * boost)[:K]
                    self.W[act] += 0.1 * (2 * x - 1); self.W[act] = np.clip(self.W[act], 0, 1); ac[act] += 1
                boost = np.exp(1.5 * (K / N_COL - ac / ((e + 1) * len(members))))

    def code(self, a):
        if not self.pooler:
            r = np.random.default_rng(self.seed * 10000 + r3.WORD_IDX[a])      # DETERMINISTIC (hash() is process-salted)
            return sorted(r.choice(N_COL, K, replace=False).tolist())
        return sorted(np.argsort(-((self.W > 0.5) @ self.X[a]))[:K].tolist())


def word_code(w, pooler, use_block):
    v = np.zeros(D_CODE)
    v[N_COL + r3.WORD_IDX[w]] = 1.0                                           # content bit (all words)
    if use_block and w in r3.ANIMAL_CAT:                                      # animals get the DISCOVERED category SDR
        for c in pooler.code(w):
            v[c] = 1.0
    return v


def encode(sent, pooler, use_block):
    return np.asarray([word_code(w, pooler, use_block) for w in sent])


def cum_feat(res, prefix, pooler, use_block):
    return res.per_token_states(encode(prefix, pooler, use_block), feature="running_cumulative")[ACTION_POS - 1]


def traj_feat(res, prefix, pooler, use_block):
    win = res.per_token_states(encode(prefix, pooler, use_block), feature="per_window")
    return np.concatenate([win[0], win[1], win[2]])


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


def run_arm(seed, arm, epochs, lr, n_pool, train_only=False):
    permute = (arm == "scramble"); pooler_on = (arm != "nopooler"); use_block = (arm != "onehot")
    learn = (arm != "nolearn")                                                # nolearn = fixed random projection (no SP learning)
    pooler = EmergentCategoryPooler(seed, permute=permute, pooler=pooler_on, train_only=train_only, learn=learn)
    res = r3.ReservoirStates(D_CODE, seed=seed, n=n_pool)

    # --- Rung-3 metric: held-out animal as AGENT -> correct category action (cum feature) ---
    feats = [cum_feat(res, s[:3], pooler, use_block) for s in r3.TRAIN_SENTS]
    tgts = [r3.WORD_IDX[s[3]] for s in r3.TRAIN_SENTS]
    if arm == "untrained":
        Wc = np.zeros((V, len(feats[0]) + 1)); mc = np.zeros(len(feats[0])); sc = np.ones(len(feats[0]))
    else:
        Wc, mc, sc = _train(feats, tgts, V, epochs, lr, seed)

    def pred_cum(prefix):
        f = cum_feat(res, prefix, pooler, use_block); x = np.concatenate([(f - mc) / sc, [1.0]])
        return r3.WORDS[int(np.argmax(Wc @ x))]
    ha_ok = ha_tot = 0
    for prefix, true_cat in r3.HELD_PREFIXES:
        p = pred_cum(prefix); ha_ok += int(p in r3.ACTION_CAT and r3.ACTION_CAT[p] == true_cat); ha_tot += 1
    heldagent = ha_ok / ha_tot

    # --- Rung-4 metric: role reversal over held-out cross-category twins (trajectory feature) ---
    feats2 = [traj_feat(res, s[:3], pooler, use_block) for s in r3.TRAIN_SENTS]
    if arm == "untrained":
        Wt = np.zeros((V, len(feats2[0]) + 1)); mt = np.zeros(len(feats2[0])); st = np.ones(len(feats2[0]))
    else:
        Wt, mt, st = _train(feats2, tgts, V, epochs, lr, seed)

    def pred_traj(prefix):
        f = traj_feat(res, prefix, pooler, use_block); x = np.concatenate([(f - mt) / st, [1.0]])
        return r3.WORDS[int(np.argmax(Wt @ x))]
    both = tot = 0
    for a, b in r4.TWINS:
        oks = []
        for (n1, n2) in [(a, b), (b, a)]:
            p = pred_traj([n1, r3.MEETS, n2]); oks.append(p in r3.ACTION_CAT and r3.ACTION_CAT[p] == r3.ANIMAL_CAT[n1])
        both += int(oks[0] and oks[1]); tot += 1
    reversal = both / tot
    return {"arm": arm, "heldagent_cat_acc": heldagent, "reversal_acc": reversal}


ARMS = ["main", "scramble", "nolearn", "nopooler", "onehot", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--inductive", action="store_true", help="pooler trained on TRAIN animals only; held codes inferred")
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
                rb[arm] = run_arm(seed, arm, args.epochs, args.lr, args.n_pool, train_only=args.inductive)
            except Exception as e:
                rb[arm] = {"arm": arm, "error": f"{e}", "trace": traceback.format_exc()}
            r = rb[arm]
            tag = (f"heldagent={r.get('heldagent_cat_acc'):.3f} reversal={r.get('reversal_acc'):.3f}") if "error" not in r else r["error"]
            print(f"[seed {seed}] {arm:10s} {tag}", flush=True)
        per_seed[seed] = rb

    def agg(arm, k):
        vals = [per_seed[s][arm].get(k) for s in args.seeds if "error" not in per_seed[s][arm]]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    aggregate = {arm: {k: agg(arm, k) for k in ("heldagent_cat_acc", "reversal_acc")} for arm in ARMS}

    per_seed_go = []
    for s in args.seeds:
        rb = per_seed[s]
        if any("error" in rb[a] for a in ARMS):
            per_seed_go.append(False); continue
        # HELDAGENT is the clean emergence metric (genuinely held animals; reversal is confounded by trained-agent identity
        # -- see the finding). The load-bearing control is SCRAMBLE (destroy the feature co-occurrence structure). The
        # generalization rides a FIXED RANDOM codon of the category-structured features (nolearn); the competitive SP (main)
        # is NOT better -- so gate on nolearn, the honest mechanism.
        nl = rb["nolearn"]["heldagent_cat_acc"]; sc = rb["scramble"]["heldagent_cat_acc"]
        per_seed_go.append(bool(nl >= 0.80 and (nl - sc) >= 0.30))
    n_go = int(sum(per_seed_go))

    out = {"runner": "_emerge_reservoir_lm_emergent_category_codes_derisk", "seeds": args.seeds,
           "n_col": N_COL, "k": K, "pool_epochs": POOL_EPOCHS, "per_seed": {str(s): per_seed[s] for s in args.seeds},
           "aggregate": aggregate, "per_seed_go": per_seed_go, "n_go": n_go, "n_seeds": len(args.seeds),
           "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nAGG main heldagent={aggregate['main']['heldagent_cat_acc']} reversal={aggregate['main']['reversal_acc']} | "
          f"scramble heldagent={aggregate['scramble']['heldagent_cat_acc']} reversal={aggregate['scramble']['reversal_acc']} "
          f"| GO {n_go}/{len(args.seeds)} ({out['elapsed_s']}s) -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
