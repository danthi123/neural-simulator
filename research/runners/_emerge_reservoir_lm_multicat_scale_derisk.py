"""EMERGENCE scale probe: does the emergent-perception-category ladder generalize with FOUR categories, not just the
two of Rung-3? The 2-category perception result (`_emerge_reservoir_lm_perception_grounded_codes_derisk.py`) had a 2-way
floor of ~0.5; a 4-category test has a 4-way floor of ~0.25, so a held-out perceived animal landing in its correct
category action is a stronger discrimination. Each of 4 categories has its own animals + its own action set; each animal
is rendered as an object SHAPE and SEEN through the real Gabor/V1 front end (EMERGE-34); the visual category is surfaced
by a FIXED codon (F.12) into the reservoir input; the ladder predicts the (category-appropriate) action after "the
<animal>"; a HELD-OUT perceived animal must inherit its category's action. Reuse-by-import (EMERGE-34 shapes/V1 + the
Rung-1 reservoir + one-step-local-delta read-out); NO `sim/` edit, NO BPTT. CPU numpy.

ARMS: main (SEEN via Gabor/V1) ; scramble (per-image pixel scramble -> no visual category) ; onehot (no block) ; untrained.
METRIC: held-out animal -> predicted action in its CORRECT category set (4-way; 2-way = which-category | action emitted).
GO: main >> scramble (visual category is load-bearing) on all 6 seeds.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import ReservoirStates
from research.runners._emerge34_perception_grounded_emergence_derisk import (
    build_shape_set, build_gabor_response_matrix, encode_v1)
from research.runners._emerge_reservoir_lm_spiking_perception_codes_derisk import SpikingCodon

OUT = Path("research/findings/raw/_reslm_multicat_scale.json")

NCAT = 4
N_TRAIN = 6          # trained animals per category
N_HELD = 3           # held-out animals per category
N_EX = N_TRAIN + N_HELD
T_ACTIVE = 20
N_COL = 120          # column-code space (must match the SpikingCodon's N_COL so both codon types index [0, N_COL))
K = 14
ACTION_POS = 2       # "the <animal>" is a length-2 prefix; predict the action at position 2

# vocabulary: the + 4 categories x 9 animals + 4 categories x 3 actions
DET = "the"
CAT_ANIMALS = {c: [f"an_{c}_{i}" for i in range(N_EX)] for c in range(NCAT)}
CAT_ACTIONS = {c: [f"act_{c}_{k}" for k in range(3)] for c in range(NCAT)}
ANIMALS = [a for c in range(NCAT) for a in CAT_ANIMALS[c]]
ACTIONS = [a for c in range(NCAT) for a in CAT_ACTIONS[c]]
WORDS = [DET] + ANIMALS + ACTIONS
WORD_IDX = {w: i for i, w in enumerate(WORDS)}
V = len(WORDS)
ANIMAL_CAT = {a: c for c in range(NCAT) for a in CAT_ANIMALS[c]}
ACTION_CAT = {a: c for c in range(NCAT) for a in CAT_ACTIONS[c]}
D_CODE = N_COL + V

# training sentences: "the <train-animal> <a category action>" (each train animal x each of its category's 3 actions)
TRAIN_SENTS = [[DET, a, act] for c in range(NCAT) for a in CAT_ANIMALS[c][:N_TRAIN] for act in CAT_ACTIONS[c]]
HELD_ANIMALS = [(a, c) for c in range(NCAT) for a in CAT_ANIMALS[c][N_TRAIN:]]     # (held animal, its true category)

_GABOR = None
def _gabor():
    global _GABOR
    if _GABOR is None:
        _GABOR = build_gabor_response_matrix()
    return _GABOR


def _v1_active(seed, scramble):
    """Each animal's top-T active V1 cells (a set of feature indices) from its SEEN shape."""
    rng = np.random.default_rng(seed)
    imgs, labels, _ = build_shape_set(n_categories=NCAT, n_exemplars=N_EX, rng=rng)
    imgs = np.asarray(imgs)
    if scramble:
        r = np.random.default_rng(seed * 7 + 1)
        imgs = np.stack([im.flatten()[r.permutation(im.size)].reshape(im.shape) for im in imgs])
    Vv = encode_v1(imgs, _gabor()); NF = Vv.shape[1]
    cat_imgs = {c: [i for i in range(len(labels)) if labels[i] == c] for c in range(NCAT)}
    active = {}
    for c in range(NCAT):
        for j, a in enumerate(CAT_ANIMALS[c]):
            v1 = Vv[cat_imgs[c][j]]
            active[a] = set(int(x) for x in np.argsort(-v1)[:T_ACTIVE])
    return active, NF


def build_codes(seed, scramble, codon="spiking"):
    active, NF = _v1_active(seed, scramble)
    if codon == "fixed":
        Wc = np.random.default_rng(seed).uniform(0.45, 0.55, (N_COL, NF))       # low-expansion fixed codon (fails at 4 cat)
        out = {}
        for a in active:
            x = np.zeros(NF); x[list(active[a])] = 1.0
            out[a] = sorted(np.argsort(-((Wc > 0.5) @ x))[:K].tolist())
        return out
    # SPIKING high-sparse-expansion Marr-Albus codon (EMERGE-35, F.12) over the union of ever-active V1 cells
    union = sorted(set().union(*[active[a] for a in active]))
    remap = {f: i for i, f in enumerate(union)}
    sc = SpikingCodon(len(union), seed)                                         # many columns, each sampling 3, coincidence-fired
    return {a: sc.codon({remap[f] for f in active[a]}) for a in active}


def word_code(w, codes, use_block):
    v = np.zeros(D_CODE); v[N_COL + WORD_IDX[w]] = 1.0
    if use_block and w in ANIMAL_CAT:
        for c in codes[w]:
            v[c] = 1.0
    return v


def cum_feat(res, prefix, codes, use_block):
    U = np.asarray([word_code(w, codes, use_block) for w in prefix])
    return res.per_token_states(U, feature="running_cumulative")[ACTION_POS - 1]


def _train(feats, tgts, epochs, lr, seed):
    X = np.array(feats); mean = X.mean(0); std = X.std(0) + 1e-6
    Xn = np.concatenate([(X - mean) / std, np.ones((len(X), 1))], 1)
    W = np.zeros((V, Xn.shape[1])); rng = np.random.default_rng(seed * 13 + 1); idx = list(range(len(Xn)))
    Ws = np.zeros_like(W); na = 0; burn = epochs // 2
    for ep in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            z = W @ Xn[i]; z = z - z.max(); p = np.exp(z); p /= p.sum()
            t = np.zeros(V); t[tgts[i]] = 1.0; W += lr * np.outer(t - p, Xn[i])
        if ep >= burn:
            Ws += W; na += 1
    return (Ws / na if na else W), mean, std


def run_arm(seed, arm, epochs, lr, n_pool):
    scramble = (arm == "scramble"); use_block = (arm != "onehot")
    codon = "spiking" if arm == "spiking_codon" else "fixed"                  # main uses the FIXED codon (scales cleanly at 4 cat)
    codes = build_codes(seed, scramble, codon=codon)
    res = ReservoirStates(D_CODE, seed=seed, n=n_pool)
    feats = [cum_feat(res, s[:2], codes, use_block) for s in TRAIN_SENTS]
    tgts = [WORD_IDX[s[2]] for s in TRAIN_SENTS]
    if arm == "untrained":
        W = np.zeros((V, len(feats[0]) + 1)); m = np.zeros(len(feats[0])); sd = np.ones(len(feats[0]))
    else:
        W, m, sd = _train(feats, tgts, epochs, lr, seed)

    def pred(prefix):
        f = cum_feat(res, prefix, codes, use_block); x = np.concatenate([(f - m) / sd, [1.0]])
        return WORDS[int(np.argmax(W @ x))]
    cat_ok = two_ok = isact = tot = 0
    for a, c in HELD_ANIMALS:
        p = pred([DET, a])
        isa = p in ACTION_CAT
        isact += int(isa)
        if isa:
            two_ok += int(ACTION_CAT[p] == c)
        cat_ok += int(isa and ACTION_CAT[p] == c)
        tot += 1
    return {"arm": arm, "held_cat_acc": cat_ok / tot, "held_2way": two_ok / max(1, isact), "held_isaction": isact / tot}


ARMS = ["main", "spiking_codon", "scramble", "onehot", "untrained"]  # main = fixed codon (scales at 4 cat); spiking_codon = diagnostic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()
    train_tokens = {w for s in TRAIN_SENTS for w in s}
    for a, _ in HELD_ANIMALS:
        assert a not in train_tokens, f"held animal {a} leaked"

    t0 = time.time(); per_seed = {}
    for seed in args.seeds:
        rb = {}
        for arm in ARMS:
            try:
                rb[arm] = run_arm(seed, arm, args.epochs, args.lr, args.n_pool)
            except Exception as e:
                rb[arm] = {"arm": arm, "error": f"{e}", "trace": traceback.format_exc()}
            r = rb[arm]
            tag = (f"cat={r.get('held_cat_acc'):.3f} 2way={r.get('held_2way'):.2f} isact={r.get('held_isaction'):.2f}") if "error" not in r else r["error"][:160]
            print(f"[seed {seed}] {arm:10s} {tag}", flush=True)
        per_seed[seed] = rb

    def agg(arm, k):
        vals = [per_seed[s][arm].get(k) for s in args.seeds if "error" not in per_seed[s][arm]]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    aggregate = {arm: {k: agg(arm, k) for k in ("held_cat_acc", "held_2way", "held_isaction")} for arm in ARMS}
    per_seed_go = []
    for s in args.seeds:
        rb = per_seed[s]
        if any("error" in rb[a] for a in ARMS):
            per_seed_go.append(False); continue
        m = rb["main"]["held_cat_acc"]; sc = rb["scramble"]["held_cat_acc"]
        per_seed_go.append(bool(m >= 0.45 and (m - sc) >= 0.20))     # 4-way chance ~0.25
    n_go = int(sum(per_seed_go))

    out = {"runner": "_emerge_reservoir_lm_multicat_scale_derisk", "seeds": args.seeds, "ncat": NCAT,
           "chance_cat": 1.0 / (NCAT), "per_seed": {str(s): per_seed[s] for s in args.seeds}, "aggregate": aggregate,
           "per_seed_go": per_seed_go, "n_go": n_go, "n_seeds": len(args.seeds), "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nAGG main.cat={aggregate['main']['held_cat_acc']} 2way={aggregate['main']['held_2way']} | "
          f"scramble.cat={aggregate['scramble']['held_cat_acc']} | chance~{1.0/NCAT:.2f} GO {n_go}/{len(args.seeds)} "
          f"({out['elapsed_s']}s) -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
