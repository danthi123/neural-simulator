"""CYCLE 103 — POSITIVE CONTROL for the bundled de-risks: does a FIXED VSA bind bundle on THIS eval harness?

The learned binders FAILED on 3-way bundles (additive 0.193; the learned-multiplicative-with-LEARNED-inverse
collapsed even on single bindings, 0.000 -- a LINEAR learned inverse cannot approximate the reciprocal). Before
concluding "a learned bind can't bundle," rule out the INSTRUMENT: can a FIXED, VSA-faithful bind (the production
mechanism) recover each role's filler from a 3-way superposition on THIS exact harness (the systematicity
splits + the bundled eval loop)? Real VSA uses a FIXED inverse, not a learned one: a role binarized to a +-1
hypervector is its OWN inverse under the elementwise (Hadamard) product.

Fixed +-1 FHRR: role_proj[r] = sign(role @ R_proj) in {+-1}^D_h (a fixed random projection, binarized);
filler_repr[f] = filler @ F_proj in R^D_h; bind = role_proj (x) filler_repr; bundle = sum of binds; unbind(bundle,
r) = bundle (x) role_proj[r] (the +-1 self-inverse) ~ filler_repr + crosstalk; cleanup = nearest filler_repr by
cosine. NO training -- it binds any (role, filler) by construction.

GATE (3 seeds): bundled recall >> chance (0.062) AND >> the learned NEGATIVE (0.193). GO => the harness DETECTS a
working bundling bind, so the learned NEGATIVEs are REAL (the additive point-neuron bind genuinely can't bundle;
the naive learned multiplicative inverse is broken) -- and bundling's validated mechanism is the FIXED algebra
(production composer, V=320). The genuine open arc = a VSA-FAITHFUL learned bind (FIXED inverse + LEARNED
embeddings + dendritic multiplication). NEGATIVE => the bundled eval is too hard for anything; revisit the eval.

Reuse-by-import (the systematicity protocol); cached 320 stream codes; CPU; no GPU; no sim/.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_fixed_fhrr_bundled_control
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

from research.runners.cortex_learned_binder_systematicity_probe import make_role_codes  # noqa: E402

R, F, D_H = 4, 16, 64
N_EVAL_FACTS = 200


def run_seed(codes, seed):
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    rng = np.random.default_rng(seed * 31 + 5)
    R_proj = rng.standard_normal((D_in, D_H)) / np.sqrt(D_in)
    F_proj = rng.standard_normal((D_in, D_H)) / np.sqrt(D_in)
    role_proj = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)          # [R, D_h] +-1 hypervector (self-inverse)
    filler_repr = fillers @ F_proj                                  # [F, D_h]
    fr_unit = filler_repr / (np.linalg.norm(filler_repr, axis=1, keepdims=True) + 1e-12)

    def cleanup(vec):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return int(np.argmax(fr_unit @ v))

    single_ok = single_n = 0
    for r in range(3):
        for f in range(F):
            bound = role_proj[r] * filler_repr[f]
            single_ok += int(cleanup(bound * role_proj[r]) == f); single_n += 1
    bun_ok = bun_n = 0
    for _ in range(N_EVAL_FACTS):
        fids = rng.choice(F, 3, replace=False)
        bundle = sum(role_proj[r] * filler_repr[fids[r]] for r in range(3))
        for r in range(3):
            bun_ok += int(cleanup(bundle * role_proj[r]) == fids[r]); bun_n += 1
    sb = single_ok / single_n
    bb = bun_ok / bun_n
    print(f"  [seed {seed}] FIXED +-1 FHRR: single-binding {sb:.3f} | BUNDLED (3-way) {bb:.3f}", flush=True)
    return {"seed": seed, "single": sb, "bundled": bb}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[fixed-FHRR bundled POSITIVE CONTROL] does a FIXED +-1 VSA bind bundle on THIS harness? "
          f"(learned additive bundled 0.193; chance 0.062)", flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    sb, bb = m("single"), m("bundled")
    chance = 1.0 / F
    print(f"\n{'='*92}\n  MEAN (3 seeds): FIXED +-1 FHRR single-binding {sb:.3f} | BUNDLED (3-way) {bb:.3f} | "
          f"chance {chance:.3f} (learned additive bundled was 0.193)", flush=True)
    print(f"{'='*92}", flush=True)
    if bb >= 0.50:
        print(f"  GO (control valid): the FIXED +-1 FHRR bind BUNDLES on this harness -- 3-way recall {bb:.3f} "
              f">> chance {chance:.3f} >> the learned additive NEGATIVE (0.193). The eval DETECTS a working "
              f"bundling bind, so the learned NEGATIVEs are REAL: the additive point-neuron bind genuinely can't "
              f"bundle, and the naive LEARNED multiplicative inverse is broken (a linear map can't be a "
              f"reciprocal). Bundling's validated mechanism = the FIXED algebra (production composer, V=320). The "
              f"genuine open arc = a VSA-FAITHFUL learned bind (FIXED inverse + LEARNED embeddings + dendritic "
              f"multiplication).", flush=True)
    elif bb >= 0.25:
        print(f"  PARTIAL: fixed FHRR bundles only weakly here ({bb:.3f}) -- D_h={D_H} / F={F} may stress the +-1 "
              f"capacity at 3-way; the harness is marginal, interpret the learned NEGATIVEs with that caveat.",
              flush=True)
    else:
        print(f"  HARNESS SUSPECT: even fixed +-1 FHRR fails to bundle here ({bb:.3f}) -- the bundled eval is too "
              f"hard for ANY bind at this D_h/F; revisit the eval before concluding on the learned binders.",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"single": sb, "bundled": bb, "chance": chance, "learned_additive_bundled": 0.193, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_fixed_fhrr_bundled_control.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
