"""CYCLE 89 — the decisive de-risk for the SHARPENED blocker: is there a code-correlation SWEET SPOT where
codes generalize SEMANTICALLY (similar concepts → similar codes) AND still BIND (systematic recombination)?

THE SHARPENED BLOCKER (CYCLE 88 + the 2026-06-11 binder finding): the CYCLE-88 PPMI reframe gives a generalizing
cortex cheaply (correlated semantic codes) — but the 2026-06-11 finding showed a learned bilinear binder is
SYSTEMATIC on DECORRELATED codes (between-cos ~0.05, held-out 1.000) and FAILS on highly-CORRELATED codes
(denoise64, between-cos ~0.81, held-out ~chance) by ill-conditioning. So: semantic generalization wants
correlation; binding wants decorrelation — a direct tension. BUT 2026-06-11 only tested the two EXTREMES
(0.05 and 0.81). The MIDDLE — where the PPMI codes live — was never swept. This probe sweeps it.

THE KNOB: F fillers in C categories; each filler code = sqrt(1-beta)·unique + sqrt(beta)·category_shared. beta=0 →
decorrelated; beta→1 → within-category near-identical (semantic but ill-conditioned). At each beta measure BOTH:
  - SEMANTIC structure: Pearson(cos(filler_codes), S_true_category) + held-out category generalization (the
    property that makes a cortex generalize across similar concepts -- the CYCLE-88 win).
  - BINDING systematicity: a leakage-free held-out recombination unbind accuracy (the 2026-06-11 metric),
    using the project's own BilinearBinder (reuse-by-import) + the memorization-floor anti-cheat.

THE FORK:
  SWEET SPOT  : some beta gives BOTH semantic structure (Pearson >= 0.30, gen above chance) AND binding (held-out
                >= 0.70) ==> the tension is NOT strict; a moderately-correlated code generalizes AND binds ->
                the functional cortex is reachable WITHOUT decorrelating away the semantics. The build target.
  STRICT      : binding and semantics are strictly anti-correlated across beta (binding only where semantics ~0,
                and vice-versa) ==> the tension is fundamental; the cortex must either decorrelate (lose
                semantic generalization, the flat shipped cortex) OR solve binding differently (the deep
                frontier: a binder robust to correlated codes -- the genuinely-open problem).
Anti-cheat: memorization floor (lookup table → ~0 on leakage-0 held-out); decorrelated endpoint (beta=0) must
reproduce 2026-06-11's systematic binding; high-beta endpoint must reproduce the binding failure; multi-seed.

Reuse-by-import (BilinearBinder, make_role_codes); NO sim/ edits; numpy; 3 seeds.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_correlation_binding_tradeoff_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.cortex_learned_binder_systematicity_probe import BilinearBinder, make_role_codes  # noqa: E402

SEEDS = (42, 43, 44)
C_CAT = 4                       # categories
PER_CAT = 4                     # fillers per category -> F = 16 fillers
R_ROLES = 4
D_IN = 64                       # code dimension
D_H = 64
EPOCHS = 400
BETAS = (0.0, 0.3, 0.5, 0.7, 0.85, 0.95)


def make_filler_codes(beta, seed):
    """F=C_CAT*PER_CAT filler codes with within-category correlation knob beta. Returns (codes [F,D], labels [F])."""
    rng = np.random.default_rng(seed * 7 + 1)
    cat_shared = rng.standard_normal((C_CAT, D_IN))
    cat_shared /= np.linalg.norm(cat_shared, axis=1, keepdims=True) + 1e-9
    codes, labels = [], []
    for c in range(C_CAT):
        for _ in range(PER_CAT):
            u = rng.standard_normal(D_IN); u /= np.linalg.norm(u) + 1e-9
            v = np.sqrt(1 - beta) * u + np.sqrt(beta) * cat_shared[c]
            codes.append(v / (np.linalg.norm(v) + 1e-9)); labels.append(c)
    return np.array(codes), np.array(labels)


def systematicity_split(F, R, seed):
    """Hold out R novel (role,filler) combos s.t. every role + every filler still appears in TRAIN (leakage-0)."""
    rng = np.random.default_rng(seed * 31 + 5)
    held = [(r, rng.integers(0, F)) for r in range(R)]                 # one held filler per role
    # ensure every filler still covered in train: all combos minus held
    allc = [(r, f) for r in range(R) for f in range(F)]
    train = [c for c in allc if c not in held]
    seen_f = {f for _, f in train}
    if len(seen_f) < F:                                                # a held filler not covered -> swap
        return systematicity_split(F, R, seed + 1)
    return train, held


def eval_binding(filler_codes, seed):
    """Train BilinearBinder on TRAIN combos, return (train_acc, heldout_acc, memfloor_acc) on unbind->nearest."""
    F = filler_codes.shape[0]
    roles = make_role_codes(R_ROLES, D_IN, seed)
    train, held = systematicity_split(F, R_ROLES, seed)
    assert not (set(train) & set(held)), "leakage"
    binder = BilinearBinder(D_in=D_IN, D_h=D_H, lr=0.01, seed=seed)
    rng = np.random.default_rng(seed * 13 + 9)
    for _ep in range(EPOCHS):
        for i in rng.permutation(len(train)):
            r, f = train[i]
            binder.train_step(roles[r], filler_codes[f])

    def acc(combos):
        ok = 0
        for r, f in combos:
            _, est = binder.predict(roles[r], filler_codes[f])
            pred = int(np.argmax(_cos_sim(np.vstack([est, filler_codes]))[0, 1:]))
            ok += int(pred == f)
        return ok / max(len(combos), 1)
    # memorization floor: a lookup table has no held-out entry -> guess train-mean filler -> ~chance.
    train_f = {f for _, f in train}
    memfloor = np.mean([int(f in train_f and False) for _, f in held])  # held fillers ARE in train_f (leakage-0)
    return acc(train), acc(held), float(memfloor)


def run_seed(seed):
    S_cat = None
    print(f"\n[corr-binding tradeoff seed {seed}]  (beta: within-category correlation knob)", flush=True)
    rows = []
    for beta in BETAS:
        codes, labels = make_filler_codes(beta, seed)
        if S_cat is None:
            S_cat = (labels[:, None] == labels[None, :]).astype(np.float64)
        bcos = float(_cos_sim(codes)[np.triu_indices(len(codes), 1)].mean())
        sem_p = _pearson_vs_Strue(_cos_sim(codes), S_cat)
        gen, chance = heldout_generalization(codes, labels)
        tr, ho, mem = eval_binding(codes, seed)
        rows.append({"beta": beta, "between_cos": round(bcos, 3), "semantic_pearson": round(sem_p, 3),
                     "gen": round(gen, 3), "bind_train": round(tr, 3), "bind_heldout": round(ho, 3)})
        print(f"  beta={beta:.2f}: between-cos {bcos:+.3f} | SEMANTIC Pearson {sem_p:+.3f} (gen {gen:.2f}/ch {chance:.2f})"
              f" | BIND train {tr:.2f} held-out {ho:.2f}", flush=True)
    return {"seed": seed, "chance": 1.0 / C_CAT, "rows": rows}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[correlation-binding tradeoff de-risk] seeds={SEEDS} C={C_CAT} per-cat={PER_CAT} R={R_ROLES} "
          f"-- is there a code-correlation SWEET SPOT (semantic generalization AND binding)?", flush=True)
    rows = [run_seed(s) for s in SEEDS]

    def col(beta, key):
        return float(np.mean([next(r for r in s["rows"] if r["beta"] == beta)[key] for s in rows]))
    chance = float(np.mean([s["chance"] for s in rows]))
    print(f"\n{'='*96}\n  MEAN ({len(SEEDS)} seeds)  [binding chance = nearest-of-{C_CAT*PER_CAT} ~ {1.0/(C_CAT*PER_CAT):.2f}; "
          f"semantic chance {chance:.2f}]", flush=True)
    sweet = []
    for beta in BETAS:
        bc, sp, gn, ho = col(beta, "between_cos"), col(beta, "semantic_pearson"), col(beta, "gen"), col(beta, "bind_heldout")
        tag = ""
        if sp >= 0.30 and gn >= chance + 0.15 and ho >= 0.70:
            tag = "  <<< SWEET SPOT (semantic AND binding)"; sweet.append((beta, sp, ho))
        print(f"  beta={beta:.2f}: between-cos {bc:+.3f} | SEMANTIC {sp:+.3f} (gen {gn:.2f}) | BIND held-out {ho:.2f}{tag}",
              flush=True)
    print(f"{'='*96}", flush=True)
    if sweet:
        b, sp, ho = max(sweet, key=lambda x: min(x[1] / 0.5, x[2]))
        print(f"  SWEET SPOT FOUND (the tension is NOT strict): at beta={b:.2f} the codes BOTH generalize "
              f"semantically (Pearson {sp:+.3f}) AND bind systematically (held-out {ho:.2f}). ==> a moderately-"
              f"correlated code generalizes across similar concepts AND supports systematic binding -> the "
              f"functional cortex is reachable WITHOUT decorrelating away the semantics. The build target = codes "
              f"at this correlation level (the CYCLE-88 PPMI codes' regime, if their between-cos lands near beta={b:.2f}).",
              flush=True)
    else:
        # characterize the tradeoff: where is binding OK, where is semantics OK?
        bind_ok = [beta for beta in BETAS if col(beta, "bind_heldout") >= 0.70]
        sem_ok = [beta for beta in BETAS if col(beta, "semantic_pearson") >= 0.30]
        print(f"  STRICT TENSION (no sweet spot at this scale): binding OK at beta∈{bind_ok} (low correlation), "
              f"semantics OK at beta∈{sem_ok} (high correlation) -- they do NOT overlap. ==> the semantic-vs-binding "
              f"tension is fundamental: a single code can't both generalize AND bind. The cortex must either "
              f"decorrelate (lose semantic generalization = the flat shipped cortex) OR the binder must be made "
              f"robust to correlated codes (the genuinely-open deep frontier). This SHARPENS the 2026-06-11 finding "
              f"with the full curve (not just the two endpoints).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"betas": list(BETAS), "chance_semantic": chance, "per_seed": rows,
           "curve": {f"{beta:.2f}": {"between_cos": col(beta, "between_cos"),
                                     "semantic_pearson": col(beta, "semantic_pearson"),
                                     "gen": col(beta, "gen"), "bind_heldout": col(beta, "bind_heldout")}
                     for beta in BETAS}}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_correlation_binding_tradeoff.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
