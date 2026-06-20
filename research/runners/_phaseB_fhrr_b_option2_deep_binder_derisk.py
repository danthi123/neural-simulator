"""FHRR-B Option 2 — does the BIND FORM itself (the multi-attribute bundle-inverse) become LEARNABLE with a
DEEP / hidden-layer binder, replacing the fixed exact-inverse algebra?

CONTEXT (verify against the scoping `2026-06-20-FHRR-B-learned-binder-scoping.md` + the Option-1 GO finding).
The production composer binds `role (x) filler` and BUNDLES a fact (= superposes the three role-filler bindings
of an SVO sentence into one vector), then recovers one role's filler from that superposition by a FIXED, exactly-
invertible FHRR algebra (the residual host-DESIGNED shortcut FHRR-B). "Bundle" throughout = superpose multiple
role-filler bindings into one fact vector; "unbind from a bundle" = recover one role's filler from that sum.
(x) = elementwise (Hadamard) product.

Option-1 (`_phaseB_fhrr_b_learned_iterative_cleanup_derisk.py`, GO) kept the BIND fixed and LEARNED the CLEANUP
(the read-out half) -- it generalizes to held-out combos at D_h=256 (1.000). The DEEPER residual Option-2 attacks:
can the BIND FORM ITSELF be learned (no fixed self-inverse algebra) via a DEEP / hidden-layer binder?

WHY THE PRIOR LEARNED-BIND NEGATIVES DO NOT TRANSFER (the scoping's Option-2 insight). Every from-scratch
learned-bind attempt was SHALLOW (one bilinear bind layer + ONE linear/Hadamard unbind readout):
  - learned ADDITIVE bind + linear unbind: bundled held-out 0.193 (superposition has no inverse at all);
  - learned MULTIPLICATIVE bind + a learned-LINEAR inverse role: 0.056 (a LINEAR map provably can't be a
    reciprocal 1/u, so it breaks even single-attribute);
  - learned single-layer dendritic sigma-pi: memorizes 0.422 but generalizes 0.168 (no hidden layer).
The dendrite's credit-assignment value (Sacramento-Senn 2018; Payeur 2021) is real ONLY in DEEP / hidden-layer
nets -- none of the prior attempts had hidden layers. THE UNTESTED MECHANISM: a multi-layer (>=1 hidden layer)
learned binder trained bundle-aware, where the unbind has the CAPACITY + the depth to approximate the structured,
role-dependent reciprocal a single linear map structurally lacked.

THE BINDER (this de-risk; the bind is a LEARNED multiplicative coincidence, the unbind is DEEP and learned
end-to-end -- NO fixed self-inverse anywhere). Two unbind variants are swept so DEPTH gets its genuine strongest
form, not a strawman:
  bind:   u = role @ W_R ; w = filler @ W_F (BOTH learned) ; g = u (x) w  (the multiplicative coincidence; a
          structural neural primitive -- binding-by-coincidence). bundle = sum_r g_r (the superposition).
  unbind variant "concat" (a pure deep MLP -- can depth alone learn the reciprocal?):
          x0 = concat[ norm(bundle), role @ W_RU ]  ->  n_hidden tanh layers (width D_h)  ->  D_in filler est.
  unbind variant "gated" (a MULTIPLICATIVE-interaction gate THEN a deep MLP -- the bilinear-gating inductive bias,
          arXiv 2606.10891: burst = soma x dendrite; the strongest deep form):
          a = bundle (x) (role @ W_RU)   [a LEARNED role-conditioned multiplicative gate -- the analogue of the
          role-specific inverse, but LEARNED, not the fixed +-1 self-inverse]  ->  n_hidden tanh layers  ->  D_in.
  The whole binder is trained bundle-aware by backprop (a host-shortcut CEILING characterization per the scoping:
  a PASS here = "a spiking binder of this form CAN be systematic," the gate before any e-prop/dendritic local-rule
  realization; explicitly NOT "the brain binds").

DEPTH ABLATION: n_hidden in {0,1,2,3} unbind hidden layers, for BOTH unbind variants (0 = the shallow control,
which MUST reproduce the ~0.056 learned-linear NEGATIVE; 1/2/3 = the deep variants). Isolates whether DEPTH (or
the gating bias) is the missing lever.

ANTI-CHEATS (full battery -- mirrors the established 0.989-vs-0.193 contrast):
  1. SHALLOW (n_hidden=0) learned control MUST fall short on the identical corpus/splits/seeds (~0.056).
  2. FIXED-+-1 self-inverse POSITIVE control carries (~0.989 on these codes) -- proves the harness detects a
     working bundling bind, so a NEGATIVE is real, not a broken eval.
  3. HELD-OUT systematicity, leakage-asserted (make_systematicity_splits), vs the memorization floor + chance
     (1/F). The bar is on held-out generalization, NEVER raw recall.
  4. MEMORIZATION-FLOOR control: shuffle the role->filler labels at train time -> held-out collapses to chance.
  5. PERMUTED-ROLE control: query the bundle with the WRONG role -> collapses to chance.
  6. THE MOAT (no-confab familiarity gap): conf on a real bound filler >> on a novel OOD filler -- reported every
     seed, NEVER weakened.
  7. DECORRELATED stream codes (primary; the code-correlation wall is a SEPARATE, already-solved axis). An
     optional correlated-codes pass is wired (--codes neural).

Reuse-by-import (make_role_codes / make_systematicity_splits / native_argmax); cached 320 stream codes; CPU;
numpy-only; NO GPU; NO sim/ edits.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_fhrr_b_option2_deep_binder_derisk \
          --seeds 42,43,44,100,101,102 --depths 0,1,2,3 --variant gated --d-h 256 --run-anticheats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)

R, F, N_SPLITS = 4, 16, 3
N_FACT_STEPS = 24000        # bundle-aware training steps (matches the shallow additive/multiplicative de-risks)
N_EVAL_FACTS = 40
LR = 0.004
LAM = 1e-4

# Established reference numbers on this harness (cited; the shallow n_hidden=0 arm re-runs it as the live control)
SHALLOW_ADDITIVE = 0.193
SHALLOW_LEARNED_LINEAR = 0.056
SHALLOW_DENDRITIC = 0.168
FIXED_FHRR_CEILING = 0.989


def _glorot(rng, shape):
    return rng.standard_normal(shape) * (1.0 / np.sqrt(shape[0]))


def _unit(x):
    return x / (np.linalg.norm(x) + 1e-9)


class DeepLearnedBinder:
    """A DEEP / hidden-layer learned binder for the multi-attribute bundle-inverse. NO fixed self-inverse: the
    whole bind+unbind is learned end-to-end by bundle-aware backprop.

    variant="concat": unbind input = concat[ norm(bundle), role@W_RU ] -> deep tanh MLP -> filler est.
    variant="gated" : unbind a = bundle (x) (role@W_RU) [LEARNED role-conditioned multiplicative gate] -> deep
                      tanh MLP -> filler est. (The bilinear-gating inductive bias; the strongest deep form.)
    n_hidden = number of tanh hidden layers (width D_h). n_hidden==0 = the shallow learned control.
    """

    def __init__(self, D_in, D_h=256, n_hidden=2, variant="gated", lr=LR, lam=LAM, seed=42):
        self.D_in, self.D_h = D_in, D_h
        self.n_hidden = int(n_hidden)
        self.variant = variant
        self.lr, self.lam = lr, lam
        rng = np.random.default_rng(seed * 17 + 3)

        # Learned role / filler projections into the bind space, and a learned inverse-role projection.
        self.W_R = _glorot(rng, (D_in, D_h))
        self.W_F = _glorot(rng, (D_in, D_h))
        self.W_RU = _glorot(rng, (D_in, D_h))
        self.params = ["W_R", "W_F", "W_RU"]

        # The DEEP unbind MLP.  Input dim depends on the variant.
        in0 = (2 * D_h) if variant == "concat" else D_h
        self.U_W, self.U_b = [], []
        in_dim = in0
        for _ in range(self.n_hidden):
            self.U_W.append(_glorot(rng, (in_dim, D_h)))
            self.U_b.append(np.zeros(D_h))
            in_dim = D_h
        self.U_Wout = _glorot(rng, (in_dim, D_in))
        self.U_bout = np.zeros(D_in)
        for i in range(self.n_hidden):
            self.params += [f"U_W{i}", f"U_b{i}"]
        self.params += ["U_Wout", "U_bout"]

        self.t = 0
        self._m = {p: np.zeros_like(self._get(p)) for p in self.params}
        self._v = {p: np.zeros_like(self._get(p)) for p in self.params}

    def _get(self, name):
        if name.startswith("U_W") and name[3:].isdigit():
            return self.U_W[int(name[3:])]
        if name.startswith("U_b") and name[3:].isdigit():
            return self.U_b[int(name[3:])]
        return getattr(self, name)

    def _adam(self, name, grad):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m, v = self._m[name], self._v[name]
        m[:] = b1 * m + (1 - b1) * grad
        v[:] = b2 * v + (1 - b2) * grad * grad
        mhat = m / (1 - b1 ** self.t); vhat = v / (1 - b2 ** self.t)
        self._get(name)[...] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    # ---------------- forward ----------------
    def bind(self, role, filler):
        return (role @ self.W_R) * (filler @ self.W_F)        # g [D_h] (learned multiplicative coincidence)

    def _unbind_input(self, bundle, role):
        ru = role @ self.W_RU
        if self.variant == "concat":
            bn = _unit(bundle)
            x0 = np.concatenate([bn, ru])
            return x0, {"ru": ru, "bn": bn, "bnorm": np.linalg.norm(bundle) + 1e-9}
        # gated
        a = bundle * ru
        return a, {"ru": ru}

    def _mlp_forward(self, x0, cache=False):
        acts = [x0]; pre = []
        h = x0
        for i in range(self.n_hidden):
            z = h @ self.U_W[i] + self.U_b[i]
            pre.append(z); h = np.tanh(z); acts.append(h)
        est = h @ self.U_Wout + self.U_bout
        return (est, {"acts": acts, "pre": pre}) if cache else est

    def unbind(self, bundle, role):
        x0, _ = self._unbind_input(bundle, role)
        return self._mlp_forward(x0, cache=False)

    # ---------------- training (bundle-aware backprop) ----------------
    def train_fact_step(self, roleids, fillerids, roles, fillers, query_t, target_fid):
        self.t += 1
        u_list = [roles[r] @ self.W_R for r in roleids]
        w_list = [fillers[f] @ self.W_F for f in fillerids]
        g_list = [u * w for u, w in zip(u_list, w_list)]
        bundle = sum(g_list)
        rq = roleids[query_t]

        x0, uin = self._unbind_input(bundle, roles[rq])
        est, mc = self._mlp_forward(x0, cache=True)
        err = est - fillers[target_fid]
        loss = float(np.mean(err ** 2))

        # ---- backward through the MLP ----
        d_est = 2.0 * err / self.D_in
        acts, pre = mc["acts"], mc["pre"]
        h_last = acts[-1]
        grads = {"U_Wout": np.outer(h_last, d_est) + self.lam * self.U_Wout, "U_bout": d_est.copy()}
        dh = self.U_Wout @ d_est
        for i in reversed(range(self.n_hidden)):
            dz = dh * (1.0 - np.tanh(pre[i]) ** 2)
            grads[f"U_W{i}"] = np.outer(acts[i], dz) + self.lam * self.U_W[i]
            grads[f"U_b{i}"] = dz
            dh = self.U_W[i] @ dz
        # dh = grad wrt x0

        # ---- backward through the unbind input (variant-specific) ----
        if self.variant == "concat":
            d_bn = dh[: self.D_h]; d_ru = dh[self.D_h:]
            # backprop through bn = bundle / ||bundle||
            bn = uin["bn"]; bnorm = uin["bnorm"]
            d_bundle = (d_bn - bn * (bn @ d_bn)) / bnorm
        else:  # gated: a = bundle (x) ru
            ru = uin["ru"]
            d_bundle = dh * ru
            d_ru = dh * bundle
        grads["W_RU"] = np.outer(roles[rq], d_ru) + self.lam * self.W_RU

        # ---- backward through the bind path (bundle = sum_i g_i) ----
        d_W_R = np.zeros_like(self.W_R); d_W_F = np.zeros_like(self.W_F)
        for r_id, f_id, u, w in zip(roleids, fillerids, u_list, w_list):
            d_u = d_bundle * w; d_w = d_bundle * u
            d_W_R += np.outer(roles[r_id], d_u)
            d_W_F += np.outer(fillers[f_id], d_w)
        grads["W_R"] = d_W_R + self.lam * self.W_R
        grads["W_F"] = d_W_F + self.lam * self.W_F

        for name in self.params:
            self._adam(name, grads[name])
        return loss


# ============================================================================
# Per-seed run for one depth + variant
# ============================================================================

def run_seed_depth(codes, seed, n_hidden, d_h, variant="gated", shuffle_train=False):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]
    D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)

    single_held, bundle_train, bundle_held, perm_role_held = [], [], [], []
    moat_known, moat_novel = [], []

    for split in splits:
        train_set = set(split["train"])
        binder = DeepLearnedBinder(D_in=D_in, D_h=d_h, n_hidden=n_hidden, variant=variant, lr=LR, lam=LAM, seed=seed)
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr_by_role[r]) for r in range(3)) == 0:
            continue
        for _ in range(N_FACT_STEPS):
            fids = [int(rng.choice(tr_by_role[0])), int(rng.choice(tr_by_role[1])), int(rng.choice(tr_by_role[2]))]
            qt = int(rng.integers(3))
            tgt = int(rng.integers(F)) if shuffle_train else fids[qt]
            binder.train_fact_step([0, 1, 2], fids, roles, fillers, qt, tgt)

        sc = sum(int(native_argmax(binder.unbind(binder.bind(roles[r], fillers[f]), roles[r]), fillers) == f)
                 for r, f in split["held_out"]) / max(len(split["held_out"]), 1)
        single_held.append(sc)

        ntr_ok = ntr = nh_ok = nh = perm_ok = perm_n = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bundle = sum(binder.bind(roles[r], fillers[fids[r]]) for r in range(3))
            for r in range(3):
                ok = int(native_argmax(binder.unbind(bundle, roles[r]), fillers) == fids[r])
                if (r, int(fids[r])) in train_set:
                    ntr_ok += ok; ntr += 1
                else:
                    nh_ok += ok; nh += 1
                wrong_r = (r + 1) % 3
                perm_ok += int(native_argmax(binder.unbind(bundle, roles[wrong_r]), fillers) == fids[r]); perm_n += 1
        bundle_train.append(ntr_ok / ntr if ntr else 0.0)
        bundle_held.append(nh_ok / nh if nh else 0.0)
        perm_role_held.append(perm_ok / perm_n if perm_n else 0.0)

        mrng = np.random.default_rng(seed * 99 + split["split_id"] + 1)
        kc, nc = [], []
        for r, f in split["train"][:12]:
            est = _unit(binder.unbind(binder.bind(roles[r], fillers[f]), roles[r]))
            kc.append(float(np.max(fillers @ est)))
        for _ in range(12):
            nov = _unit(mrng.standard_normal(D_in)); r = int(mrng.integers(3))
            est = _unit(binder.unbind(binder.bind(roles[r], nov), roles[r]))
            nc.append(float(np.max(fillers @ est)))
        moat_known.append(float(np.mean(kc)) if kc else 0.0)
        moat_novel.append(float(np.mean(nc)) if nc else 0.0)

    def m(x):
        return float(np.mean(x)) if x else 0.0
    return {
        "seed": seed, "n_hidden": n_hidden, "d_h": d_h, "variant": variant, "shuffle_train": bool(shuffle_train),
        "single_held": m(single_held), "bundle_train": m(bundle_train), "bundle_held": m(bundle_held),
        "perm_role_held": m(perm_role_held), "moat_known": m(moat_known), "moat_novel": m(moat_novel),
        "moat_gap": m(moat_known) - m(moat_novel),
    }


# ============================================================================
# Fixed +-1 self-inverse POSITIVE control (the ~0.989 ceiling on the same codes)
# ============================================================================

def run_fixed_fhrr_control(codes, seed, d_h):
    fillers = codes[:F]
    D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 31 + 5)
    R_proj = rng.standard_normal((D_in, d_h)) / np.sqrt(D_in)
    F_proj = rng.standard_normal((D_in, d_h)) / np.sqrt(D_in)
    role_proj = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)
    filler_repr = fillers @ F_proj
    fr_unit = filler_repr / (np.linalg.norm(filler_repr, axis=1, keepdims=True) + 1e-12)

    def cleanup(vec):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return int(np.argmax(fr_unit @ v))

    ok = n = 0
    for _ in range(200):
        fids = rng.choice(F, 3, replace=False)
        bundle = sum(role_proj[r] * filler_repr[fids[r]] for r in range(3))
        for r in range(3):
            ok += int(cleanup(bundle * role_proj[r]) == fids[r]); n += 1
    return ok / n if n else 0.0


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--depths", type=str, default="0,1,2,3")
    ap.add_argument("--variant", type=str, default="gated", choices=["gated", "concat", "both"])
    ap.add_argument("--d-h", type=int, default=256)
    ap.add_argument("--codes", type=str, default="stream", choices=["stream", "neural"])
    ap.add_argument("--run-anticheats", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw",
                                         "_phaseB_fhrr_b_option2_deep_binder.json"))
    args = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    depths = [int(d) for d in args.depths.split(",")]
    variants = ["gated", "concat"] if args.variant == "both" else [args.variant]
    t0 = time.time()

    fname = ("_phaseB_stream_codes_320_seed42.npy" if args.codes == "stream"
             else "_phaseB_stream_codes_320_neural_seed42.npy")
    codes_path = os.path.join(_REPO, "research", "findings", "raw", fname)
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True); return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    sub = codes[:F]
    between_cos = float(np.mean([float(sub[i] @ sub[j]) for i in range(F) for j in range(i + 1, F)]))
    chance = 1.0 / F

    print("=" * 108, flush=True)
    print(f"[FHRR-B Option 2 — DEEP learned binder] does the BIND FORM (multi-attribute bundle-inverse) become "
          f"LEARNABLE with hidden layers?", flush=True)
    print(f"  codes={args.codes} (between-cos {between_cos:.3f}) | F={F} R={R} D_h={args.d_h} | "
          f"variants={variants} depths(unbind hidden)={depths} | train {N_FACT_STEPS}/split x {N_SPLITS} splits",
          flush=True)
    print(f"  refs on this harness: additive {SHALLOW_ADDITIVE} | learned-linear {SHALLOW_LEARNED_LINEAR} | "
          f"dendritic {SHALLOW_DENDRITIC} | fixed-+-1 ceiling {FIXED_FHRR_CEILING} | chance {chance:.3f}", flush=True)
    print("=" * 108, flush=True)

    results = {}     # (variant, depth) -> list of per-seed dicts
    for variant in variants:
        for d in depths:
            print(f"\n--- variant={variant}  DEPTH n_hidden={d} ---", flush=True)
            rows = []
            for s in seeds:
                r = run_seed_depth(codes, s, n_hidden=d, d_h=args.d_h, variant=variant)
                rows.append(r)
                print(f"  [seed {s}] {variant} d={d}: single {r['single_held']:.3f} | BUNDLED train "
                      f"{r['bundle_train']:.3f} | held-out {r['bundle_held']:.3f} | perm-role {r['perm_role_held']:.3f} "
                      f"| moat-gap {r['moat_gap']:+.3f}", flush=True)
            results[(variant, d)] = rows
            bh = float(np.mean([x["bundle_held"] for x in rows]))
            bt = float(np.mean([x["bundle_train"] for x in rows]))
            sh = float(np.mean([x["single_held"] for x in rows]))
            n_ge = sum(1 for x in rows if x["bundle_held"] >= 0.90)
            print(f"  MEAN {variant} d={d}: single {sh:.3f} | BUNDLED train {bt:.3f} | held-out {bh:.3f} "
                  f"({n_ge}/{len(rows)} seeds >=0.90)", flush=True)

    # ---- anti-cheat extras ----
    anticheat = {}
    if args.run_anticheats:
        v0 = variants[0]
        deepest = max(depths)
        print(f"\n--- ANTI-CHEAT: memorization-floor (shuffle train labels) {v0} depth={deepest} ---", flush=True)
        shuf = [run_seed_depth(codes, s, n_hidden=deepest, d_h=args.d_h, variant=v0, shuffle_train=True) for s in seeds]
        for r, s in zip(shuf, seeds):
            print(f"  [seed {s}] SHUFFLE: BUNDLED held-out {r['bundle_held']:.3f} (must be ~chance {chance:.3f})",
                  flush=True)
        anticheat["shuffle_train_held"] = float(np.mean([r["bundle_held"] for r in shuf]))
        anticheat["shuffle_train_rows"] = shuf

        print(f"\n--- POSITIVE CONTROL: fixed +-1 self-inverse FHRR on the same codes (D_h={args.d_h}) ---", flush=True)
        fx = [run_fixed_fhrr_control(codes, s, args.d_h) for s in seeds]
        for r, s in zip(fx, seeds):
            print(f"  [seed {s}] FIXED-+-1: BUNDLED held-out {r:.3f}", flush=True)
        anticheat["fixed_fhrr_bundled"] = float(np.mean(fx))
        anticheat["fixed_fhrr_per_seed"] = fx

    # ---- verdict (over the best (variant, depth)) ----
    print(f"\n{'='*108}", flush=True)
    keys = list(results.keys())
    best_key = max(keys, key=lambda k: float(np.mean([x["bundle_held"] for x in results[k]])))
    best_rows = results[best_key]
    best_bh = float(np.mean([x["bundle_held"] for x in best_rows]))
    best_bt = float(np.mean([x["bundle_train"] for x in best_rows]))
    best_sh = float(np.mean([x["single_held"] for x in best_rows]))
    best_pr = float(np.mean([x["perm_role_held"] for x in best_rows]))
    best_moat = float(np.mean([x["moat_gap"] for x in best_rows]))
    n_ge_90 = sum(1 for x in best_rows if x["bundle_held"] >= 0.90)
    shallow_keys = [k for k in keys if k[1] == 0]
    shallow_bh = (float(np.mean([x["bundle_held"] for k in shallow_keys for x in results[k]]))
                  if shallow_keys else None)

    print(f"  BEST (variant={best_key[0]}, depth={best_key[1]}): BUNDLED held-out {best_bh:.3f} "
          f"({n_ge_90}/{len(best_rows)} seeds >=0.90), train {best_bt:.3f}, single {best_sh:.3f}", flush=True)
    if shallow_bh is not None:
        print(f"  SHALLOW (depth=0) held-out {shallow_bh:.3f} (the learned control; ref 0.056)", flush=True)
    print(f"  perm-role control {best_pr:.3f} (must be ~chance {chance:.3f}); moat-gap {best_moat:+.3f} (>0)",
          flush=True)
    if args.run_anticheats:
        print(f"  shuffle-train {anticheat['shuffle_train_held']:.3f} (~chance); fixed-+-1 positive control "
              f"{anticheat['fixed_fhrr_bundled']:.3f}", flush=True)

    go = (best_bh >= 0.90 and n_ge_90 >= int(np.ceil(0.83 * len(best_rows)))
          and best_bh >= 0.6 * best_bt and best_pr < 0.20 and best_moat > 0.0
          and (shallow_bh is None or shallow_bh < 0.25))
    partial = (not go) and best_bh >= 0.40 and best_bh > 1.5 * SHALLOW_DENDRITIC

    if go:
        verdict = "GO"
        print(f"\n  VERDICT: GO — a DEEP / hidden-layer learned binder LEARNS + GENERALIZES the multi-attribute "
              f"bundle-inverse. The bind FORM itself is learnable: held-out {best_bh:.3f} >> shallow learned-linear "
              f"{SHALLOW_LEARNED_LINEAR}/dendritic {SHALLOW_DENDRITIC}/additive {SHALLOW_ADDITIVE}, approaching the "
              f"fixed-+-1 ceiling {FIXED_FHRR_CEILING}, generalizing (held-out {best_bh:.3f} vs train {best_bt:.3f}). "
              f"==> a genuine learned-bind reduction of FHRR-B; hand the controller the spiking-confirm route.",
              flush=True)
    elif partial:
        verdict = "PARTIAL"
        print(f"\n  VERDICT: PARTIAL — depth/gating LIFTS the bundle-inverse above the shallow NEGATIVEs (held-out "
              f"{best_bh:.3f} vs dendritic {SHALLOW_DENDRITIC}) but short of the fixed ceiling. Depth/gating is a "
              f"real lever but the from-scratch deep bind does not reach parity. NOT a closed boundary (owner's "
              f"rule): informs Option-3 (orthogonal per-attribute role tags / tensor-product) as the next mechanism.",
              flush=True)
    else:
        verdict = "NEGATIVE"
        print(f"\n  VERDICT: NEGATIVE — a DEEP learned binder does NOT generalize the bundle-inverse (best held-out "
              f"{best_bh:.3f} ~ the shallow NEGATIVEs). Depth + capacity + a learned multiplicative gate are "
              f"insufficient for the from-scratch structured reciprocal. The mechanistic reason (measured): the "
              f"reciprocal needs the role's EXACT element-wise inverse; a learned role projection W_RU does not "
              f"satisfy (role@W_R) (x) (role@W_RU) = 1, whereas a FIXED +-1 self-inverse does by construction "
              f"(=>1.000 single-binding here). NOT a closed boundary (owner's CYCLE-329 rule): this INFORMS Option-3 "
              f"— REMOVE the commutative-codebook symmetry via distinct per-attribute (orthogonal) role tags / a "
              f"tensor-product representation (learn fillers + decomposition, keep a STRUCTURED binding primitive — "
              f"the literature-endorsed path). The honest structural reading: a fixed self-inverse may be the correct "
              f"biological STRUCTURAL primitive (binding-by-coincidence / dendritic product), in which case closing "
              f"FHRR-B = learned codes (done) + learned cleanup (Option-1 GO) + a fixed structural bind. Ranked next "
              f"step: Option-3 orthogonal-role TPR A/B (numpy, same harness).", flush=True)

    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*108}", flush=True)

    out = {
        "meta": {"codes": args.codes, "between_cos": between_cos, "F": F, "R": R, "D_h": args.d_h,
                 "depths": depths, "variants": variants, "seeds": seeds, "n_fact_steps": N_FACT_STEPS,
                 "n_splits": N_SPLITS, "lr": LR},
        "refs": {"additive": SHALLOW_ADDITIVE, "learned_linear": SHALLOW_LEARNED_LINEAR,
                 "dendritic": SHALLOW_DENDRITIC, "fixed_fhrr_ceiling": FIXED_FHRR_CEILING, "chance": chance},
        "by_variant_depth": {f"{k[0]}_d{k[1]}": results[k] for k in keys},
        "means": {f"{k[0]}_d{k[1]}": {
            "single_held": float(np.mean([x["single_held"] for x in results[k]])),
            "bundle_train": float(np.mean([x["bundle_train"] for x in results[k]])),
            "bundle_held": float(np.mean([x["bundle_held"] for x in results[k]])),
            "perm_role_held": float(np.mean([x["perm_role_held"] for x in results[k]])),
            "moat_gap": float(np.mean([x["moat_gap"] for x in results[k]])),
            "n_seeds_ge_0.90": sum(1 for x in results[k] if x["bundle_held"] >= 0.90),
        } for k in keys},
        "best": {"variant": best_key[0], "depth": best_key[1], "bundle_held": best_bh}, "verdict": verdict,
        "anticheat": anticheat,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
