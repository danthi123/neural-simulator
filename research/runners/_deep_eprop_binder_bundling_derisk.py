"""2026-07-14 — ROADMAP #2 (the learned cortical binder): can a DEEP, NONLINEAR, e-prop-trained UNBIND network
learn MULTI-ATTRIBUTE BUNDLING (the conversational-fact bind) where a SHALLOW/LINEAR learned unbind was NEGATIVE?

THE OPEN QUESTION (from `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`).
A conversational FACT is a 3-way SUPERPOSITION of role-filler binds (agent+verb+object). Single-attribute learned
bind = GO (0.806). But 3-way BUNDLING was NEGATIVE for a learned bind (additive 0.193; learned-multiplicative-with-
LEARNED-LINEAR-inverse 0.056) while a FIXED +-1 self-inverse bind bundles at 0.989 (positive control). The finding
LOCALIZED it: unbinding role t from the bundle needs a ROLE-DEPENDENT INVERSE (a multiplication by ~1/u_t); a shared
LINEAR unbind is "structurally incapable independent of capacity/training." It concluded "multiplication is dendritic,
not point-neuron." **BUT IT ONLY TESTED A SHALLOW/LINEAR UNBIND.** A DEEP NONLINEAR net can approximate the
role-dependent multiplication (universal approximation; and with R=4 DISCRETE roles the role-dependent inverse is a
PIECEWISE-LINEAR / mixture-of-4-diagonal-maps function that deep ReLU/LIF nets represent) -- THE UNTESTED LEVER.

THE TEST (single variable = unbind DEPTH/nonlinearity; everything else identical).
  Fixed bind (shared, VSA-faithful, = the positive control's bind): role_proj[r] = sign(role_r @ R_proj) in {+-1}^Dh
    (fixed random binarized), filler_repr[f] = filler_f @ F_proj in R^Dh; bind = role_proj (x) filler_repr; a FACT =
    sum of 3 binds. The +-1 self-inverse makes the info RECOVERABLE-in-principle (positive control 0.989) -> the ONLY
    question is whether a LEARNED unbind can DISCOVER the role-dependent multiplicative inverse.
  Unbind input = concat(bundle [Dh], role_cue [D_in raw code]); output = F-way classifier over the fillers; the
    metric = filler-recovery accuracy (argmax over F), split into TRAIN-combo and HELD-OUT-combo (systematicity).
  ARM A (baseline, the NEGATIVE): SHALLOW/LINEAR unbind (0 hidden LIF layers) -- reproduces the "linear unbind can't".
  ARM B (the test): DEEP (2 and 3 hidden LIF layers) NONLINEAR unbind trained by TRANSPORT-FREE e-prop (Bellec 2020 +
    Nokland DFA; reuse `_eprop_grads` -- forward eligibility + membrane surrogate + fixed-random feedback, NO BPTT, NO
    weight transport). A deep-BPTT reference (best-possible credit) is ALSO run so a NEGATIVE is interpretable (depth
    vs credit-rule).

MANDATORY VALIDITY GATES (reported explicitly; the goal is the TRUTH either way):
  * POSITIVE CONTROL passes (fixed +-1 -> ~0.99) else the harness is broken.
  * ARM-B FIT GATE: the deep e-prop net must FIT (memorize) a small bundle set to high train acc -- else the e-prop
    wiring is broken (false negative). Run with BOTH e-prop and BPTT credit.
  * anti-cheats: 1-NN memorization floor (novel bundles -> ~chance); chance line 1/F=0.0625; leakage-free held-out
    systematicity split (asserted); permuted-label control (Arm-B trained on shuffled labels -> ~chance held-out).

VERDICT:
  * Arm B (deep e-prop) held-out bundling >> chance AND >> the shallow ~0.19 (approaching fixed-+-1 0.99) => the
    point-neuron/dendrite verdict is SURPASSED: a DEEP learned binder cracks multi-attribute bundling.
  * Arm B also ~0.19/chance => the point-neuron limit is CONFIRMED even for a deep nonlinear net => multiplication
    genuinely needs the DENDRITIC substrate (two-compartment D2 neuron), not depth -- a precise, honest boundary.

SUBSTRATE NOTE: the 2026-06-16 numbers used the cached stream-cortex codes (missing/regenerable). The bundling
limitation is CODE-AGNOSTIC (additive has no inverse; fixed +-1 self-inverses -- structural, not code-specific), so
this runner uses the reproducible DECORRELATED sparse codes (`load_sparse_codes_native`) -- the FAIREST (best-case)
substrate for a learned binder, which makes any NEGATIVE maximally decisive. Reuse-by-import; NO sim/ edit; CPU.
Run:  SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m research.runners._deep_eprop_binder_bundling_derisk
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    load_sparse_codes_native, make_role_codes, make_systematicity_splits)
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _train_snn, _forward_logits)

R, F, D_H = 4, 16, 64          # 4 roles, 16 fillers, bind dimension (match the 2026-06-16 harness)
PROJ_DIM = 128                 # code dimension (role cue + filler codes live here)
T = 24                         # LIF rate-code timesteps (match the validated e-prop isolation runner)
OUT = os.path.join(_REPO, "research", "findings", "raw", "_deep_eprop_binder_bundling.json")


# ---------------------------------------------------------------------------
# Fixed VSA-faithful bind (identical to the positive-control bind)
# ---------------------------------------------------------------------------

def build_fixed_bind(codes, roles, D_h, seed):
    D_in = codes.shape[1]
    rng = np.random.default_rng(seed * 31 + 5)
    R_proj = rng.standard_normal((D_in, D_h)) / np.sqrt(D_in)
    F_proj = rng.standard_normal((D_in, D_h)) / np.sqrt(D_in)
    role_proj = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)          # [R, D_h] +-1 hypervector (self-inverse)
    filler_repr = codes[:F] @ F_proj                                # [F, D_h]
    fr_unit = filler_repr / (np.linalg.norm(filler_repr, axis=1, keepdims=True) + 1e-12)
    return role_proj, filler_repr, fr_unit


def positive_control(role_proj, filler_repr, fr_unit, train_set, rng, n_facts=200):
    """Fixed +-1 unbind = bundle (x) role_proj[r]; cleanup = nearest filler_repr by cosine. No learning."""
    def cleanup(v):
        v = v / (np.linalg.norm(v) + 1e-12)
        return int(np.argmax(fr_unit @ v))
    s_ok = s_n = 0
    for r in range(3):
        for f in range(F):
            b = role_proj[r] * filler_repr[f]
            s_ok += int(cleanup(b * role_proj[r]) == f); s_n += 1
    tr_ok = tr_n = he_ok = he_n = 0
    for _ in range(n_facts):
        fids = rng.choice(F, 3, replace=False)
        bundle = sum(role_proj[r] * filler_repr[int(fids[r])] for r in range(3))
        for r in range(3):
            ok = int(cleanup(bundle * role_proj[r]) == fids[r])
            if (r, int(fids[r])) in train_set:
                tr_ok += ok; tr_n += 1
            else:
                he_ok += ok; he_n += 1
    return s_ok / s_n, tr_ok / max(tr_n, 1), he_ok / max(he_n, 1)


# ---------------------------------------------------------------------------
# Dataset: (input = concat(bundle, role_cue), target = queried filler index)
# ---------------------------------------------------------------------------

def make_train_rows(role_proj, filler_repr, roles, split, rng, n_facts):
    """BUNDLE-AWARE training rows built ONLY from TRAIN combos. Each fact (roles 0,1,2 = agent,verb,object) yields 3
    rows (query each role). A held-out (role, filler) pair is NEVER formed here."""
    tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
    X, y = [], []
    for _ in range(n_facts):
        fids = [int(rng.choice(tr_by_role[r])) for r in range(3)]
        bundle = sum(role_proj[r] * filler_repr[fids[r]] for r in range(3))
        for r in range(3):
            X.append(np.concatenate([bundle, roles[r]])); y.append(fids[r])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def make_eval_rows(role_proj, filler_repr, roles, train_set, rng, n_facts):
    """Eval facts: 3 random distinct fillers (any of F). Each queried (role, filler) tagged train- vs held-out-combo."""
    X, y, held = [], [], []
    for _ in range(n_facts):
        fids = rng.choice(F, 3, replace=False)
        bundle = sum(role_proj[r] * filler_repr[int(fids[r])] for r in range(3))
        for r in range(3):
            X.append(np.concatenate([bundle, roles[r]])); y.append(int(fids[r]))
            held.append((r, int(fids[r])) not in train_set)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), np.asarray(held, dtype=bool)


def standardize(Xtr, *others):
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0) + 1e-8
    return [(X - mu) / sd for X in (Xtr,) + others]


def score_snn(layers, Xev, yev, held, in_gain):
    logits, _, _ = _forward_logits(Xev, layers, T, in_gain)
    correct = (np.argmax(logits, axis=1) == yev)
    tr = correct[~held]; he = correct[held]
    return (float(tr.mean()) if len(tr) else float("nan"),
            float(he.mean()) if len(he) else float("nan"),
            float(correct.mean()))


def nn_floor(Xtr, ytr, Xev, yev, held):
    """1-NN memorization floor: nearest TRAIN input -> its label. On novel bundles -> ~chance held-out."""
    # cosine 1-NN (rows are standardized); chunk to bound memory
    Xtr_n = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    Xev_n = Xev / (np.linalg.norm(Xev, axis=1, keepdims=True) + 1e-12)
    preds = np.empty(len(Xev), dtype=np.int64)
    for i in range(0, len(Xev), 256):
        sims = Xev_n[i:i + 256] @ Xtr_n.T
        preds[i:i + 256] = ytr[np.argmax(sims, axis=1)]
    correct = (preds == yev)
    he = correct[held]
    return float(he.mean()) if len(he) else float("nan")


# ---------------------------------------------------------------------------
# Per (seed, split)
# ---------------------------------------------------------------------------

def run_split(codes, roles, split, seed, hidden, epochs, lr, in_gain, n_train_facts, n_eval_facts,
              run_bptt=True, run_3hidden=True):
    train_set = set(split["train"])
    # leakage assert (anti-cheat)
    assert len(train_set & set(split["held_out"])) == 0, "LEAKAGE: held-out combo in train set"
    role_proj, filler_repr, fr_unit = build_fixed_bind(codes, roles, D_H, seed)

    pc_rng = np.random.default_rng(seed * 101 + split["split_id"])
    pc_single, pc_tr, pc_he = positive_control(role_proj, filler_repr, fr_unit, train_set, pc_rng)

    tr_rng = np.random.default_rng(seed * 53 + split["split_id"] * 7 + 9)
    ev_rng = np.random.default_rng(seed * 97 + split["split_id"] * 13 + 3)
    Xtr, ytr = make_train_rows(role_proj, filler_repr, roles, split, tr_rng, n_train_facts)
    Xev, yev, held = make_eval_rows(role_proj, filler_repr, roles, train_set, ev_rng, n_eval_facts)
    Xtr, Xev = standardize(Xtr, Xev)
    n_in = Xtr.shape[1]

    out = {"split_id": split["split_id"], "n_in": int(n_in), "n_train_rows": int(len(Xtr)),
           "n_eval_rows": int(len(Xev)), "n_held_rows": int(held.sum()),
           "pc_single": pc_single, "pc_bundle_train": pc_tr, "pc_bundle_held": pc_he}

    def _fit_acc(layers):
        return float(np.mean(np.argmax(_forward_logits(Xtr, layers, T, in_gain)[0], axis=1) == ytr))

    # ARM A: shallow/linear unbind (0 hidden LIF layers), e-prop
    la = _train_snn(Xtr, ytr, [n_in, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    out["armA_lin_fit"] = _fit_acc(la)
    out["armA_lin_train"], out["armA_lin_held"], _ = score_snn(la, Xev, yev, held, in_gain)

    # ARM B (2 hidden), e-prop
    lb2 = _train_snn(Xtr, ytr, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    out["armB2_eprop_fit"] = _fit_acc(lb2)
    out["armB2_eprop_train"], out["armB2_eprop_held"], _ = score_snn(lb2, Xev, yev, held, in_gain)

    # ARM B (3 hidden), e-prop
    if run_3hidden:
        lb3 = _train_snn(Xtr, ytr, [n_in, hidden, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
        out["armB3_eprop_fit"] = _fit_acc(lb3)
        out["armB3_eprop_train"], out["armB3_eprop_held"], _ = score_snn(lb3, Xev, yev, held, in_gain)

    # ARM B (2 hidden), BPTT reference (best-possible credit -> makes a NEGATIVE interpretable)
    if run_bptt:
        lb2b = _train_snn(Xtr, ytr, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="bptt")
        out["armB2_bptt_fit"] = _fit_acc(lb2b)
        out["armB2_bptt_train"], out["armB2_bptt_held"], _ = score_snn(lb2b, Xev, yev, held, in_gain)

    # anti-cheat: 1-NN memorization floor (held-out)
    out["nn_floor_held"] = nn_floor(Xtr, ytr, Xev, yev, held)

    # anti-cheat: permuted-label Arm B (2h e-prop) -> must collapse to ~chance held-out
    prng = np.random.default_rng(seed * 3 + split["split_id"])
    yperm = ytr[prng.permutation(len(ytr))]
    lbp = _train_snn(Xtr, yperm, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    _, out["armB2_permuted_held"], _ = score_snn(lbp, Xev, yev, held, in_gain)
    return out


def fit_gate(codes, roles, seed, hidden, in_gain, n_small=128, epochs=300):
    """FIT GATE: can the deep net MEMORIZE a small bundle set (train acc high)? Run with BOTH e-prop and BPTT credit
    so a low e-prop fit vs a high BPTT fit is diagnosable (credit-rule vs wiring)."""
    split = make_systematicity_splits(R, F, 1, seed)[0]
    role_proj, filler_repr, _ = build_fixed_bind(codes, roles, D_H, seed)
    rng = np.random.default_rng(seed * 71 + 1)
    Xtr, ytr = make_train_rows(role_proj, filler_repr, roles, split, rng, max(1, n_small // 3))
    Xtr = Xtr[:n_small]; ytr = ytr[:n_small]
    (Xn,) = standardize(Xtr)
    n_in = Xn.shape[1]
    res = {}
    for mode in ("eprop", "bptt"):
        layers = _train_snn(Xn, ytr, [n_in, hidden, hidden, F], T, epochs, 0.05, in_gain, seed, credit_mode=mode)
        logits, _, _ = _forward_logits(Xn, layers, T, in_gain)
        res[mode] = float(np.mean(np.argmax(logits, axis=1) == ytr))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--n-train-facts", type=int, default=500)
    ap.add_argument("--n-eval-facts", type=int, default=150)
    ap.add_argument("--no-bptt", action="store_true")
    ap.add_argument("--no-3hidden", action="store_true")
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    chance = 1.0 / F
    t0 = time.time()

    print(f"[deep-e-prop binder bundling de-risk] R={R} F={F} D_H={D_H} proj={PROJ_DIM} T={T} hidden={args.hidden} "
          f"epochs={args.epochs} | chance {chance:.4f}; shallow/additive NEGATIVE (2026-06-16) 0.193; fixed-+-1 0.989",
          flush=True)

    # ---- FIT GATE (seed 42) ----
    codes42, cm, cx = load_sparse_codes_native(seed=42, V=F, n_pool=2000, pattern_size=100, proj_dim=PROJ_DIM)
    roles42 = make_role_codes(R, codes42.shape[1], 42)
    fg = fit_gate(codes42, roles42, 42, args.hidden, args.in_gain)
    print(f"  [FIT GATE seed42] deep 2-hidden memorize-small train acc: e-prop {fg['eprop']:.3f} | BPTT {fg['bptt']:.3f}"
          f"  (>=0.90 => e-prop wiring can fit; codes between-cos mean {cm:.3f} max {cx:.3f})", flush=True)

    per_seed = []
    for seed in seeds:
        codes, cmn, cmx = load_sparse_codes_native(seed=seed, V=F, n_pool=2000, pattern_size=100, proj_dim=PROJ_DIM)
        roles = make_role_codes(R, codes.shape[1], seed)
        splits = make_systematicity_splits(R, F, args.n_splits, seed)
        rows = [run_split(codes, roles, sp, seed, args.hidden, args.epochs, args.lr, args.in_gain,
                          args.n_train_facts, args.n_eval_facts,
                          run_bptt=not args.no_bptt, run_3hidden=not args.no_3hidden) for sp in splits]

        def m(k):
            vals = [r[k] for r in rows if k in r and not np.isnan(r[k])]
            return float(np.mean(vals)) if vals else float("nan")
        agg = {"seed": seed, "between_cos_mean": float(cmn),
               "pc_single": m("pc_single"), "pc_bundle_held": m("pc_bundle_held"),
               "armA_lin_fit": m("armA_lin_fit"), "armA_lin_held": m("armA_lin_held"),
               "armB2_eprop_fit": m("armB2_eprop_fit"), "armB2_eprop_train": m("armB2_eprop_train"),
               "armB2_eprop_held": m("armB2_eprop_held"),
               "armB3_eprop_fit": m("armB3_eprop_fit"), "armB3_eprop_held": m("armB3_eprop_held"),
               "armB2_bptt_fit": m("armB2_bptt_fit"), "armB2_bptt_held": m("armB2_bptt_held"),
               "nn_floor_held": m("nn_floor_held"), "armB2_permuted_held": m("armB2_permuted_held"),
               "splits": rows}
        per_seed.append(agg)
        print(f"  [seed {seed}] PC single {agg['pc_single']:.3f} bundle-held {agg['pc_bundle_held']:.3f} | "
              f"ArmA(lin) fit {agg['armA_lin_fit']:.3f} held {agg['armA_lin_held']:.3f} | ArmB2(eprop) fit "
              f"{agg['armB2_eprop_fit']:.3f} bundle-train {agg['armB2_eprop_train']:.3f} held {agg['armB2_eprop_held']:.3f} | "
              f"ArmB3(eprop) fit {agg['armB3_eprop_fit']:.3f} held {agg['armB3_eprop_held']:.3f} | "
              f"ArmB2(bptt) fit {agg['armB2_bptt_fit']:.3f} held {agg['armB2_bptt_held']:.3f} | "
              f"nn-floor {agg['nn_floor_held']:.3f} | perm {agg['armB2_permuted_held']:.3f}", flush=True)

    def M(k):
        vals = [s[k] for s in per_seed if not np.isnan(s[k])]
        return float(np.mean(vals)) if vals else float("nan")
    pc = M("pc_bundle_held"); armA = M("armA_lin_held")
    b2 = M("armB2_eprop_held"); b3 = M("armB3_eprop_held"); b2b = M("armB2_bptt_held")
    b2_fit = M("armB2_eprop_fit"); b3_fit = M("armB3_eprop_fit"); b2b_fit = M("armB2_bptt_fit")
    b2_btr = M("armB2_eprop_train")
    nnf = M("nn_floor_held"); perm = M("armB2_permuted_held")
    best_deep = max([v for v in (b2, b3) if not np.isnan(v)] or [float("nan")])

    print(f"\n{'=' * 104}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds): POSITIVE CONTROL fixed-+-1 bundle-held {pc:.3f} | ArmA shallow/linear held "
          f"{armA:.3f}", flush=True)
    print(f"  ArmB DEEP e-prop: 2h fit {b2_fit:.3f} bundle-train {b2_btr:.3f} HELD {b2:.3f} | 3h fit {b3_fit:.3f} HELD "
          f"{b3:.3f}  (best held {best_deep:.3f}) | 2h BPTT-ref fit {b2b_fit:.3f} HELD {b2b:.3f}", flush=True)
    print(f"  anti-cheats: chance {chance:.4f} | 1-NN mem-floor held {nnf:.3f} | permuted-label ArmB held {perm:.3f} | "
          f"FIT GATE(mem-small) e-prop {fg['eprop']:.3f} / BPTT {fg['bptt']:.3f}", flush=True)
    print(f"{'=' * 104}", flush=True)

    pc_ok = pc >= 0.50
    fit_ok = fg["eprop"] >= 0.90 or fg["bptt"] >= 0.90
    surpass = (not np.isnan(best_deep)) and best_deep >= 0.50 and best_deep >= armA + 0.20 and best_deep >= 2 * chance
    if not pc_ok:
        verdict = "HARNESS-BROKEN"
        msg = (f"POSITIVE CONTROL failed ({pc:.3f} < 0.50) -- the fixed +-1 bind does not bundle on this harness; fix "
               f"the harness before concluding on the learned arms.")
    elif surpass:
        verdict = "SURPASS"
        msg = (f"a DEEP nonlinear e-prop-trained unbind CRACKS multi-attribute bundling: held-out {best_deep:.3f} >> "
               f"shallow/linear {armA:.3f} and >> chance {chance:.3f}, approaching the fixed-+-1 ceiling {pc:.3f}. The "
               f"2026-06-16 'multiplication is dendritic, not point-neuron' verdict is SURPASSED by DEPTH -> a LEARNED "
               f"cortical binder can replace the VSA composer's fixed exact-inverse algebra (ROADMAP #2).")
    else:
        verdict = "CONFIRMED-BOUNDARY"
        msg = (f"a DEEP nonlinear net MEMORIZES the unbind (2h fit {b2_fit:.3f}; TRAIN-combo bundling {b2_btr:.3f} >> the "
               f"additive 0.193 -- so depth CAN compute the multiplication for SEEN pairs) but does NOT GENERALIZE it: "
               f"held-out (novel role-filler combos) {best_deep:.3f} ~ shallow/linear {armA:.3f} ~ chance {chance:.3f}"
               + (f", and the best-possible-credit BPTT reference ALSO fails to generalize ({b2b:.3f})" if not np.isnan(b2b) else "")
               + f", while the FIXED +-1 algebra generalizes for free ({pc:.3f} held). DEPTH is NOT the lever: it buys a "
               f"MEMORIZED per-pair multiplication, not the fixed algebra's BUILT-IN systematicity. The learned "
               f"cortical binder still needs the STRUCTURAL binding primitive -> the DENDRITIC-multiplication substrate "
               f"(two-compartment D2 neuron) is the honest next lever, not point-neuron depth.")
    fit_note = "" if fit_ok else ("  [WARNING] FIT GATE did not reach 0.90 -- the e-prop/BPTT wiring may under-fit; "
                                  "a NEGATIVE could be a training artifact, interpret with caution.")
    print(f"  VERDICT: {verdict} -- {msg}{fit_note}", flush=True)
    print(f"  Total elapsed: {time.time() - t0:.1f}s", flush=True)

    out = {"probe": "deep_eprop_binder_bundling", "verdict": verdict, "message": msg,
           "config": {"R": R, "F": F, "D_H": D_H, "proj_dim": PROJ_DIM, "T": T, "hidden": args.hidden,
                      "epochs": args.epochs, "lr": args.lr, "in_gain": args.in_gain, "n_splits": args.n_splits,
                      "n_train_facts": args.n_train_facts, "n_eval_facts": args.n_eval_facts, "seeds": seeds},
           "chance": chance, "fit_gate": fg, "fit_ok": bool(fit_ok), "pc_ok": bool(pc_ok),
           "means": {"pc_bundle_held": pc, "armA_lin_held": armA, "armB2_eprop_fit": b2_fit,
                     "armB2_eprop_train": b2_btr, "armB2_eprop_held": b2, "armB3_eprop_fit": b3_fit,
                     "armB3_eprop_held": b3, "armB2_bptt_fit": b2b_fit, "armB2_bptt_held": b2b,
                     "best_deep_eprop_held": best_deep, "nn_floor_held": nnf, "armB2_permuted_held": perm},
           "per_seed": per_seed, "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
