"""2026-07-15 — TEST A (research gate `2026-07-15-beyond-ngram-wall-...`): does a FIXED binding STRUCTURE + learned/observed
codes crack SYSTEMATIC compositional EXTRAPOLATION where every LEARNER (deep e-prop, and even the ideal BPTT oracle in
our record) sits at chance?

THE TASK (scale-confound-FREE by construction: held-out combos have ZERO surface count -> n-grams/memorization fail):
  intent[cat,qt] = argmax_k  W_intent[k] . (a[cat] (X) b[qt])        # (X) = XOR-in-+-1 = the +-1 Hadamard product
  a[cat], b[qt] in {+-1}^nb are the RULE factors; W_intent fixed random -> intent is LINEARLY readable FROM a(X)b.
OBSERVED codes carry the factors: cat_code = [a[cat] ; fixed-random +-1 padding], q_code = [b[qt] ; padding] (dim D).
Hold out ~30% of (cat,qt) COMBINATIONS entirely (each cat + each qt still attested in OTHER cells).

THE DISCRIMINATOR:
  - FIXED +-1 BIND (cat_code (.) q_code) computes a(X)b in the first nb dims for ANY combo incl. held-out -> a LINEAR
    read-out trained on ATTESTED combos EXTRAPOLATES (the structure supplies the combination).
  - A LEARNER (deep e-prop on [cat_code;q_code]) must LEARN to combine the two factors -> memorizes attested (cat,qt),
    fails held-out (our record: deep e-prop 0.264 chance, BPTT oracle also fails).
  - LINEAR on [cat_code;q_code] concat cannot represent the XOR combination at all.
GATE (6-seed): fixedbind held-out >> learner AND >> linear-concat AND >> memfloor(1-NN); permuted->chance;
  POSITIVE CONTROL = fixedbind on an EASY single-factor task reproduces ~1.0 (the fixed bind isn't magic, it matches the
  task structure). GO => systematicity is ARCHITECTURAL STRUCTURE the point-neuron substrate ALREADY hosts (the FHRR
  composer) -> route effort to 'learn to USE a fixed bind/store', not to a richer learner and not to scale.

Run: SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m research.runners._fixedbind_systematicity_derisk
"""
import os, sys, json, argparse
import numpy as np
from numpy.linalg import norm, solve

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._deep_eprop_binder_bundling_derisk import _train_snn, score_snn, standardize, T

N_CAT = 7
N_QTYPE = 7
N_INTENT = 4
NB = 8           # rule-factor bits (a[cat], b[qt] in {+-1}^NB)
D_PAD = 4        # per-factor identity/padding dims (small: enough to distinguish, not enough to memorize under strong lam)
LAM_BIND = 8.0   # strong ridge regularization on the bind -> use the generalizable a(X)b signal, not the per-combo padding noise
SPARS = 0.5


def _pm(rng, n):
    return np.where(rng.random(n) < 0.5, -1.0, 1.0)


def build_task(seed, easy=False, holdout_frac=0.3):
    rng = np.random.default_rng(seed * 2711 + 13)
    a = np.stack([_pm(rng, NB) for _ in range(N_CAT)])        # rule factor per category
    b = np.stack([_pm(rng, NB) for _ in range(N_QTYPE)])      # rule factor per qtype
    W_int = rng.standard_normal((N_INTENT, NB))               # fixed linear map: a(X)b -> intent logits
    cat_pad = np.stack([_pm(rng, D_PAD) for _ in range(N_CAT)])
    q_pad = np.stack([_pm(rng, D_PAD) for _ in range(N_QTYPE)])
    cat_code = np.concatenate([a, cat_pad], axis=1)           # observed code carries a[cat] + identity padding
    q_code = np.concatenate([b, q_pad], axis=1)               # carries b[qt] + padding
    D = NB + D_PAD

    def intent_of(c, q):
        if easy:
            return int((a[c][0] > 0) * 2 + (a[c][1] > 0))    # single-factor -> pos control
        xb = a[c] * b[q]                                      # a (X) b = +-1 Hadamard = XOR of the factor bits
        return int((xb[0] > 0) * 2 + (xb[1] > 0))            # intent = the first two XOR-bits -> LINEARLY separable from a(X)b (oracle ~1.0)

    # hold out ~holdout_frac of (cat,qt) CELLS entirely (each cat + qt still attested elsewhere)
    cells = [(c, q) for c in range(N_CAT) for q in range(N_QTYPE)]
    rng.shuffle(cells)
    held = set()
    for (c, q) in cells:
        if len(held) >= int(holdout_frac * len(cells)):
            break
        tc = sum(1 for (c2, q2) in cells if c2 == c and (c2, q2) not in held) - 1
        tq = sum(1 for (c2, q2) in cells if q2 == q and (c2, q2) not in held) - 1
        if tc >= 2 and tq >= 2:
            held.add((c, q))
    return a, b, cat_code, q_code, intent_of, held, D


def _bind(cat_code, q_code):
    return cat_code * q_code                                  # +-1 Hadamard -> first NB dims = a (X) b; padding = noise


def _ridge(Phi_tr, y_tr, Phi_ev, K, lam=1.0):
    Y = np.zeros((len(y_tr), K)); Y[np.arange(len(y_tr)), y_tr] = 1.0
    W = solve(Phi_tr.T @ Phi_tr + lam * np.eye(Phi_tr.shape[1]), Phi_tr.T @ Y)
    return np.argmax(Phi_ev @ W, axis=1)


def _dataset(cat_code, q_code, intent_of, held, n_per=40, seed=0):
    rng = np.random.default_rng(seed + 999)
    X, y, is_held = [], [], []
    for c in range(N_CAT):
        for q in range(N_QTYPE):
            for _ in range(n_per):                            # replicate (deterministic codes -> tests generalization, not noise)
                X.append((c, q)); y.append(intent_of(c, q)); is_held.append((c, q) in held)
    return X, np.array(y), np.array(is_held, bool)


def run_one(seed, easy=False, hidden=48, epochs=140):
    a, b, cat_code, q_code, intent_of, held, D = build_task(seed, easy=easy)
    cells, y, is_held = _dataset(cat_code, q_code, intent_of, held, n_per=1, seed=seed)  # 1 per cell (codes deterministic)
    tr = ~is_held
    out = {"seed": seed, "easy": easy, "chance": round(1.0 / N_INTENT, 4),
           "n_train_cells": int(tr.sum()), "n_held_cells": int(is_held.sum())}
    CAT = np.array([cat_code[c] for (c, q) in cells]); Q = np.array([q_code[q] for (c, q) in cells])
    # ORACLE (posctrl): ridge on the TRUE a(X)b (clean rule variable, no padding) -> is the a(X)b->intent map RECOVERABLE
    # + generalizable from the attested combos at all? (the ceiling the fixed bind aims to reach)
    AB = np.array([a[c] * b[q] for (c, q) in cells])
    ABtr, ABev = standardize(AB[tr], AB)
    po = _ridge(ABtr, y[tr], ABev, N_INTENT, lam=1.0)
    out["oracle_held"] = round(float(np.mean(po[is_held] == y[is_held])), 4)
    # ARM 1: FIXED +-1 BIND + strongly-regularized linear ridge (the structural combination over OBSERVED codes)
    B = np.array([_bind(cat_code[c], q_code[q]) for (c, q) in cells])
    Btr, Bev = standardize(B[tr], B)
    pb = _ridge(Btr, y[tr], Bev, N_INTENT, lam=LAM_BIND)
    out["fixedbind_train"] = round(float(np.mean(pb[tr] == y[tr])), 4)
    out["fixedbind_held"] = round(float(np.mean(pb[is_held] == y[is_held])), 4)
    # ARM 2: LINEAR on [cat;q] concat (can't represent the XOR combination)
    C = np.concatenate([CAT, Q], axis=1); Ctr, Cev = standardize(C[tr], C)
    pc = _ridge(Ctr, y[tr], Cev, N_INTENT)
    out["linear_concat_held"] = round(float(np.mean(pc[is_held] == y[is_held])), 4)
    # ARM 3: DEEP e-prop LEARNER on [cat;q] concat (must learn to combine -> memorizes attested, fails held-out)
    lay = _train_snn(Ctr, y[tr], [C.shape[1], hidden, hidden, N_INTENT], T, epochs, 0.05, 1.0, seed, credit_mode="eprop")
    out["learner_train"], out["learner_held"], _ = score_snn(lay, Cev, y, is_held, 1.0)
    out["learner_train"] = round(out["learner_train"], 4); out["learner_held"] = round(out["learner_held"], 4)
    # ANTI-CHEAT memfloor: 1-NN on the raw concat (held-out combo has no same-combo neighbor)
    hi = np.where(is_held)[0]

    def nn(i):
        d = [norm(C[i] - C[j]) for j in np.where(tr)[0]]; return y[np.where(tr)[0][int(np.argmin(d))]]
    out["memfloor_held"] = round(float(np.mean([nn(i) == y[i] for i in hi])), 4) if len(hi) else 0.0
    # ANTI-CHEAT permuted: shuffle intent labels on train -> fixedbind held collapses to chance
    rp = np.random.default_rng(seed + 7); yp = y.copy(); yp[tr] = y[tr][rp.permutation(int(tr.sum()))]
    pbp = _ridge(Btr, yp[tr], Bev, N_INTENT, lam=LAM_BIND)
    out["fixedbind_permuted_held"] = round(float(np.mean(pbp[is_held] == y[is_held])), 4)
    out["GO"] = bool(out["oracle_held"] > 0.85                       # posctrl: the a(X)b->intent map IS recoverable+generalizable
                     and out["fixedbind_held"] > 0.6
                     and out["fixedbind_held"] > out["learner_held"] + 0.15
                     and out["fixedbind_held"] > out["linear_concat_held"] + 0.15
                     and out["fixedbind_held"] > out["memfloor_held"] + 0.15
                     and out["fixedbind_permuted_held"] < out["chance"] + 0.15)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--out", default="research/findings/raw/_fixedbind_systematicity.json")
    a = ap.parse_args()
    rows = [run_one(s, easy=False, epochs=a.epochs) for s in a.seeds]
    for r in rows:
        print(f"[fixedbind s{r['seed']}] chance={r['chance']} || FIXEDBIND held={r['fixedbind_held']:.3f} (train {r['fixedbind_train']:.3f}) "
              f"| oracle(posctrl)={r['oracle_held']:.3f} | learner held={r['learner_held']:.3f} (train {r['learner_train']:.3f}) "
              f"| linear-concat={r['linear_concat_held']:.3f} | memfloor={r['memfloor_held']:.3f} | permuted={r['fixedbind_permuted_held']:.3f} "
              f"|| {'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    print(f"[fixedbind] {ngo}/{len(rows)} GO (fixed bind EXTRAPOLATES held-out compositions >> learner + linear + memfloor; permuted collapses; posctrl~1.0)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
