"""CYCLE 82 — "learn naturally over time", judged by FUNCTION (retention), in the VIABLE (separable) regime.

The CYCLE-81 answer: on the RAW corpus the point-neuron SM caps at ~+0.35 (the locality wall); the viable regime
is SEPARABLE/curated concepts (SM +0.93). The owner's "learn naturally over time" question, asked in the regime
where the brain-rule WORKS, becomes two concrete, untested, function-grounded questions:

  Q1 STABILITY-OVER-TIME : on the RAW corpus the online SM OVER-TRAINS (peak ~+0.35 -> degrades to ~+0.26). Does
                           the SAME degrade hit the SEPARABLE regime, or is separable STABLE over long continual
                           training? (If separable degrades too, even the shipped curated cortex needs
                           early-stopping/consolidation -- a real finding for the product.)
  Q2 CATASTROPHIC-FORGET : the project's CORE continual-learning premise. Train the online SM on a FIRST set of
                           categories; then CONTINUE training on a SECOND, DISJOINT set WITHOUT revisiting the
                           first. Does the first set's structure SURVIVE (retention >= bar) or collapse? An
                           interleaved (all-at-once) arm is the no-forgetting upper bound.

This is "learn naturally over time" measured by RETENTION (function), not by Pearson-to-an-offline-PCA-target
(the project's "validate a signal by its function" standard). The SM update equations are the validated
`learn_simmatch` rule (Oja feedforward + anti-Hebbian fixed-point lateral), inlined ONLY to thread the (W, M)
state across continual phases (learn_simmatch re-initializes each call). Byte-faithful to the validated rule.

THE FORK:
  STABLE+NO-FORGET : separable stays at its +0.9 plateau over long training AND continual-disjoint training
                     retains the first set (>= 0.80x) ==> natural learning over time WORKS in the viable regime;
                     ships with the curated cortex. A clean, constructive positive for the owner.
  DEGRADE / FORGET : separable also over-trains OR continual-disjoint catastrophically forgets ==> even the
                     viable regime needs the project's consolidation/replay machinery (interleaved replay, the
                     SWR mechanism) to learn-over-time -- a concrete, honest next build (NOT the dendritic wall).

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_continual_separable_derisk
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
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, effective_rank,
)
from research.runners.learned_graded_cortex_fair_test import ppmi_matrix  # noqa: E402

SEEDS = (42, 43, 44)
K = 64
# the VALIDATED stable learn_simmatch defaults (the +0.93-on-synthetic config): equal rates 0.01, settle 30.
# (CYCLE 81 showed the fast-lateral ratio does nothing on the raw corpus; here we use the validated stable
# config that reaches +0.93 on separable, the regime under test.)
LR_FF, LR_M, SETTLE = 0.01, 0.01, 30
M_BOUND = 5.0     # numerical guard: keep |M| bounded so the damped settle stays stable (bounded inhibitory
                  # weights -- biologically standard; the anti-Hebbian de-risk used the same guard). The raw
                  # corpus never reached it; the strong separable structure can, hence the explicit cap.


def _settle(ff, M, settle_steps):
    y = np.zeros(M.shape[0])
    for _ in range(settle_steps):
        y = 0.5 * y + 0.5 * (ff - M @ y)
    return y


def sm_epoch(W, M, Xn, idx, rng, lr_ff=LR_FF, lr_m=LR_M, settle=SETTLE):
    """One epoch of the VALIDATED learn_simmatch rule over the given concept indices, mutating (W, M) in place
    (inlined ONLY to thread state across continual phases -- equations byte-faithful to learn_simmatch:189-192,
    plus a bounded-M numerical guard for the strong separable structure)."""
    for i in rng.permutation(idx):
        x = Xn[i]
        y = _settle(W @ x, M, settle)
        W += lr_ff * (np.outer(y, x) - (y ** 2)[:, None] * W)
        dM = np.outer(y, y) - M
        np.fill_diagonal(dM, 0.0)
        M += lr_m * dM
        nrm = np.linalg.norm(M)
        if nrm > M_BOUND:
            M *= M_BOUND / nrm


def read_block(W, M, Xn, idx, S_true, settle=SETTLE):
    """Pearson(cos(codes[idx]), S_true[idx,idx]) -- the internal structure of a concept subset."""
    codes = np.array([_settle(W @ Xn[i], M, settle) for i in idx])
    if not np.all(np.isfinite(codes)):
        return float("nan"), float("nan")                     # diverged -> surfaced honestly, not silently 0
    Sb = S_true[np.ix_(idx, idx)]
    return _pearson_vs_Strue(_cos_sim(codes), Sb), effective_rank(codes)


def run_seed(seed):
    C, labels, S_true, _ = build_concept_hub_counts(8, 8, 200, 12, 40.0, 4.0, 0.3, seed)
    Nc, H = C.shape
    labels = np.asarray(labels)
    Xp = ppmi_matrix(C, 0.75)                                  # the validated +0.93 input (host-matched PPMI)
    Xn = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-9)
    rng = np.random.RandomState(seed * 104729 + 3)
    all_idx = np.arange(Nc)
    cats = np.unique(labels)
    phase1_cats, phase2_cats = cats[:4], cats[4:]
    idx1 = np.where(np.isin(labels, phase1_cats))[0]
    idx2 = np.where(np.isin(labels, phase2_cats))[0]

    # ---- Q1: stability over long continual training on ALL concepts (track the trajectory) ----
    W = rng.randn(K, H) * 0.1; M = np.zeros((K, K))
    traj = []
    for ep in range(400):
        sm_epoch(W, M, Xn, all_idx, rng)
        if (ep + 1) % 80 == 0:
            p, r = read_block(W, M, Xn, all_idx, S_true)
            traj.append((ep + 1, round(p, 3), round(r, 1)))
    peak = max(p for _, p, _ in traj); end = traj[-1][1]
    stable = end >= peak - 0.03

    # ---- Q2: catastrophic forgetting -- train phase 1, then phase 2 (disjoint), re-read phase 1 ----
    Wc = rng.randn(K, H) * 0.1; Mc = np.zeros((K, K))
    for _ in range(150):
        sm_epoch(Wc, Mc, Xn, idx1, rng)
    p1_pre, _ = read_block(Wc, Mc, Xn, idx1, S_true)           # phase-1 structure after phase-1 training
    for _ in range(150):
        sm_epoch(Wc, Mc, Xn, idx2, rng)                        # CONTINUE on phase 2 only (no phase-1 revisit)
    p1_post, _ = read_block(Wc, Mc, Xn, idx1, S_true)          # phase-1 structure after phase-2 training
    p2_post, _ = read_block(Wc, Mc, Xn, idx2, S_true)          # did phase 2 learn?
    retention = p1_post / p1_pre if p1_pre > 1e-6 else 0.0

    # ---- control: INTERLEAVED (all 64 shuffled) = the no-forgetting upper bound ----
    Wi = rng.randn(K, H) * 0.1; Mi = np.zeros((K, K))
    for _ in range(150):
        sm_epoch(Wi, Mi, Xn, all_idx, rng)
    p1_inter, _ = read_block(Wi, Mi, Xn, idx1, S_true)

    print(f"\n[continual seed {seed}] {Nc} separable concepts (8 cat x 8), {H} hubs", flush=True)
    print(f"  Q1 stability(all,400ep): traj {traj}  peak={peak:+.3f} end={end:+.3f}  "
          f"[{'STABLE' if stable else 'DEGRADES'}]", flush=True)
    print(f"  Q2 forgetting: phase1 pre={p1_pre:+.3f} -> post-phase2={p1_post:+.3f}  retention={retention:.2f}  "
          f"(phase2 learned={p2_post:+.3f}; interleaved-control phase1={p1_inter:+.3f})", flush=True)
    return {"seed": seed, "peak": peak, "end": end, "stable": stable, "p1_pre": p1_pre, "p1_post": p1_post,
            "retention": retention, "p2_post": p2_post, "p1_interleaved": p1_inter, "traj": traj}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[continual separable de-risk] seeds={SEEDS} -- does the online SM learn STABLY over time + retain old "
          f"concepts when new ones are added, in the VIABLE separable regime?", flush=True)
    rows = [run_seed(s) for s in SEEDS]

    def m(key):
        return float(np.mean([r[key] for r in rows]))
    peak, end = m("peak"), m("end")
    ret = m("retention"); p1_post, p1_inter = m("p1_post"), m("p1_interleaved"); p2 = m("p2_post")
    all_stable = all(r["stable"] for r in rows)
    print(f"\n{'='*96}\n  MEAN ({len(SEEDS)} seeds): Q1 peak {peak:+.3f} -> end {end:+.3f} "
          f"[{'ALL STABLE' if all_stable else 'DEGRADES'}]", flush=True)
    print(f"  Q2 forgetting: retention {ret:.2f} (phase1 post-phase2 {p1_post:+.3f} vs interleaved {p1_inter:+.3f}); "
          f"phase2 learned {p2:+.3f}", flush=True)
    print(f"{'='*96}", flush=True)
    no_forget = ret >= 0.80
    if all_stable and no_forget:
        print(f"  STABLE+NO-FORGET: the online SM in the viable separable regime is STABLE over long training "
              f"(peak {peak:+.3f} ~ end {end:+.3f}) AND retains the first concept-set after continual-disjoint "
              f"training (retention {ret:.2f} >= 0.80). ==> natural learning over time WORKS in the regime where "
              f"the brain-rule works; ships with the curated 2,048-concept cortex. A clean constructive positive.",
              flush=True)
    elif not all_stable and no_forget:
        print(f"  OVER-TRAINS (but no-forget): separable also degrades over long training (peak {peak:+.3f} -> end "
              f"{end:+.3f}) though continual-disjoint retains (ret {ret:.2f}). ==> even the viable regime needs "
              f"early-stopping / homeostatic stabilization over time -- a concrete product finding (NOT the wall).",
              flush=True)
    elif all_stable and not no_forget:
        print(f"  CATASTROPHIC-FORGET: separable is stable in batch but continual-DISJOINT training forgets the "
              f"first set (retention {ret:.2f} < 0.80 vs interleaved {p1_inter:+.3f}). ==> the viable regime needs "
              f"the project's CONSOLIDATION/REPLAY machinery (interleaved SWR replay) to learn-over-time without "
              f"forgetting -- a concrete, honest next build (the CLS/replay path the project already validated).",
              flush=True)
    else:
        print(f"  DEGRADE+FORGET: separable both over-trains AND forgets ==> natural-over-time needs BOTH stability "
              f"AND replay-consolidation even in the viable regime; the project's homeostasis + SWR machinery is "
              f"the concrete next build.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"q1_peak": peak, "q1_end": end, "q1_all_stable": all_stable, "q2_retention": ret,
           "q2_phase1_post": p1_post, "q2_phase1_interleaved": p1_inter, "q2_phase2_learned": p2, "per_seed": rows}
    raw_dir = os.path.join(_REPO, "research", "findings", "raw")
    path = os.path.join(raw_dir, "_phaseB_continual_separable.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
