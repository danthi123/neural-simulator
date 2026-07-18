"""KEYSTONE gap #2 — de-risk #2: the SELF-ORGANIZING binder (delta-rule error-correcting fast-weight write).

2026-07-17, per `research/findings/2026-07-17-keystone-binder-research-gate.md` (ranked #2). #1 (the fixed
coincidence-product) bundles CORRELATED codes 0.873 but keeps a HAND-SET ±1 self-inverse + conj-inverse tie —
the scaffold the emergence bar rejects. This tests the genuinely self-organizing replacement:

  bind a fact = write K (role-key, filler-value) pairs into a per-fact fast-weight matrix W.
  DELTA rule (error-correcting): for each (k, v):  W += beta * (v - W @ k) k^T   (erase-then-write)
  plain-HEBBIAN control:                            W += v k^T                   (accumulates crosstalk)
  unbind role t:  v_est = W @ k_t  -> cleanup (nearest concept).

Why this is on the emergence bar (vs #1's fixed algebra): the write is a LOCAL rule (no backprop, no weight
transport), the read is just `W k` (NO hand-set conjugate-inverse), role keys are developmental random draws, and
filler values are the LEARNED stream-cortex codes. The delta rule's erase-then-write is exactly the mechanism by
which DeltaNet beats vanilla additive fast-weights on non-orthogonal keys — and the committed Urbanczik-Senn clean-
error channel IS a delta-rule instance, so this ports to spikes without new credit machinery.

GO bar: delta-rule held bundling top-1 on CORRELATED codes >= 0.80 AND >= the plain-Hebbian arm (delta is
load-bearing), degrading gracefully with K. Anti-cheats (all mandatory):
 - PERMUTED-ROLE: unbind by a WRONG role key -> must collapse toward chance (the read is genuinely role-addressed,
   not reading residual code-similarity between correlated fillers). This is the artifact that killed prior
   concept-concept claims; STRICT top-1 exact-concept scoring + permuted-role jointly rule it out.
 - CHANCE line 1/F, reported.
 - DECORRELATED-code control: the SAME op on decorrelated codes must ALSO work (the win is the OP, not correlation).

CPU/numpy, 3-seed rate rung (spike rung via fused_coincidence_plateau is the follow-on if GO).
"""

import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, native_argmax)

F = 64            # fillers (concepts)
R = 3             # roles per fact (SVO: agent/verb/patient)
N_FACTS = 400     # random facts evaluated per seed
BETA = 1.0        # delta step (1.0 = exact erase-then-write per key)


def _norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def store_fact(D, roleids, fillerids, roles, fillers, delta):
    """Build a per-fact fast-weight W (D x D) by writing each (role-key, filler-value) pair."""
    W = np.zeros((D, D))
    for r, f in zip(roleids, fillerids):
        k = roles[r]; v = fillers[f]
        if delta:
            W += BETA * np.outer(v - W @ k, k)     # error-correcting erase-then-write
        else:
            W += np.outer(v, k)                    # plain Hebbian outer product
    return W


def run_seed(codes, seed, delta, permute_role=False, decorrelate=False):
    rng = np.random.default_rng(seed * 101 + 7)
    if decorrelate:
        # decorrelated control: random near-orthogonal filler codes, same dim/count
        fillers = _norm_rows(rng.standard_normal((F, codes.shape[1])))
    else:
        fillers = _norm_rows(codes[:F].astype(np.float64))
    D = fillers.shape[1]
    roles = _norm_rows(make_role_codes(R, D, seed).astype(np.float64))
    n_ok = n = 0
    for _ in range(N_FACTS):
        fids = rng.choice(F, R, replace=False)
        W = store_fact(D, list(range(R)), list(fids), roles, fillers, delta)
        for r in range(R):
            key = roles[(r + 1) % R] if permute_role else roles[r]   # WRONG key under permute
            est = W @ key
            pred = native_argmax(est, fillers)
            n_ok += int(pred == fids[r]); n += 1
    return n_ok / n if n else 0.0


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True); return
    codes = np.load(codes_path)
    nrm = _norm_rows(codes[:F].astype(np.float64))
    mean_cos = float(np.abs((nrm @ nrm.T)[~np.eye(F, dtype=bool)]).mean())
    print(f"[delta-rule bind de-risk] fillers={F} roles={R} facts/seed={N_FACTS} | "
          f"filler mean|cos|={mean_cos:.3f} (CORRELATED) | chance={1.0/F:.3f}", flush=True)

    seeds = (42, 43, 44)
    rows = []
    for s in seeds:
        delta = run_seed(codes, s, delta=True)
        hebb = run_seed(codes, s, delta=False)
        perm = run_seed(codes, s, delta=True, permute_role=True)      # anti-cheat
        deco = run_seed(codes, s, delta=True, decorrelate=True)       # control
        rows.append({"seed": s, "delta": delta, "hebb": hebb, "permuted_role": perm, "decorrelated": deco})
        print(f"  [seed {s}] DELTA {delta:.3f} | hebbian {hebb:.3f} | "
              f"permuted-role {perm:.3f} (must ~chance) | decorrelated-ctrl {deco:.3f}", flush=True)

    md = {k: float(np.mean([r[k] for r in rows])) for k in ("delta", "hebb", "permuted_role", "decorrelated")}
    chance = 1.0 / F
    go = (md["delta"] >= 0.80 and md["delta"] >= md["hebb"] and md["permuted_role"] <= 2.5 * chance
          and md["decorrelated"] >= 0.80)
    print("=" * 92)
    print(f"  MEAN(3): DELTA {md['delta']:.3f} | hebbian {md['hebb']:.3f} | permuted-role {md['permuted_role']:.3f} "
          f"| decorrelated {md['decorrelated']:.3f} | chance {chance:.3f}")
    print(f"  {'GO' if go else 'BOUNDARY'}: delta>=0.80 ({md['delta']>=0.80}) & delta>=hebbian "
          f"({md['delta']>=md['hebb']}) & permuted-role~chance ({md['permuted_role']<=2.5*chance}) & "
          f"decorrelated-works ({md['decorrelated']>=0.80})")
    print(f"  Total elapsed: {time.time()-t0:.1f}s")
    out = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_deltarule_bind_bundled.json")
    json.dump({"rows": rows, "mean": md, "chance": chance, "go": go, "filler_mean_cos": mean_cos}, open(out, "w"), indent=2)
    print(f"  [saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
