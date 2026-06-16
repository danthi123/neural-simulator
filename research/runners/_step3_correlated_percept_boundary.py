"""Step-3 boundary map — how much CORRELATED (shared-structure) percept does the fixed-grounded-code compose tolerate?

The scaled de-risk validated the FLAT-DISTINCT regime (orthogonal/disjoint percepts — faithful to the deployed nav
perception render + the V=320 flat-distinct tier): live `cortex_it` rate -> grounded phasor -> compose, GO to 32
objects. The honest open frontier is the SEMANTICALLY-CORRELATED regime (similar objects share code structure ->
generalize across similar concepts), which the CLAUDE.md step-3 fork assigns to the deferred dendritic / option-B
rewrite. THIS run MAPS that boundary quantitatively: sweep a shared-common-mode fraction alpha in each object's
percept (alpha=0 = orthogonal; alpha=1 = all objects identical) at CONSTANT total drive, and measure where the
grounded-code compose degrades. The alpha at which clean compose falls below the gate = exactly how much correlated
structure the fixed point-neuron grounded-code approach tolerates before the dendritic frontier is required.

This is an honest-negative-mapping run (the project's "negatives map the limit" standard): it is EXPECTED to
degrade as alpha rises (the composer's clean algebra demands decorrelated codes — the documented common-mode /
opponency wall). The DELIVERABLE is the degradation curve (the tolerance threshold), not a GO.

  SIM_BACKEND=cupy python -m research.runners._step3_correlated_percept_boundary --alphas 0.0 0.25 0.5 0.75 0.9
No sim/ edit; reuse-by-import (the (B) probe's live cortex_it bridge).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host
from sim.text_embeddings import orthogonal_drive_pattern

from research.runners.funcint_perception_to_memory_probe import (
    N_OBJECTS, N_CORTEX_IT, PERCEPT_SPARSITY, PERCEPT_DRIVE_PA, build_probe_bridge,
)

D = 2048
RATE_READ_STEPS = 80
SETTLE_STEPS = 20


def _projection(n_in, seed):
    rng = np.random.default_rng(seed * 5077 + 11)
    return (rng.standard_normal((D, n_in)) + 1j * rng.standard_normal((D, n_in))).astype(np.complex128)


def _to_phasor(rate_vec, proj):
    z = proj @ rate_vec.astype(np.complex128)
    return np.exp(1j * np.angle(z))


def _role(seed):
    rng = np.random.default_rng(seed * 13 + 3)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, D))


def _cleanup(query, codebook):
    sims = [float(np.abs(np.vdot(c, query)) / D) for c in codebook]
    return int(np.argmax(sims))


def _correlated_drive(obj_idx, n_it, alpha, shared_band):
    """object percept = (1-alpha)*unique_orthogonal_band + alpha*shared_common_band, at ~constant total magnitude.
    alpha=0 -> fully unique (orthogonal); alpha=1 -> all objects identical (the shared band). The shared band is the
    SAME for every object -> a controlled common-mode / semantic-overlap component."""
    unique = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS, n_neurons=n_it,
                                      drive_max_pA=PERCEPT_DRIVE_PA, sparsity=PERCEPT_SPARSITY).astype(np.float64)
    return (1.0 - alpha) * unique + alpha * shared_band


def read_rate(bridge, it_indices, obj_idx, alpha, shared_band):
    xp, _ = get_backend()
    n_it = int(it_indices.size)
    drive = _correlated_drive(obj_idx, n_it, alpha, shared_band)
    drive_dev = xp.asarray(drive, dtype=xp.float32)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    counts = np.zeros(n_it, dtype=np.float64)
    for _ in range(RATE_READ_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[it_indices] = drive_dev
        bridge._run_one_simulation_step()
        counts += np.asarray(to_host(bridge.cp_firing_states[it_indices])).astype(np.float64)
    bridge.cp_external_input_current[:] = 0.0
    return counts / RATE_READ_STEPS


def run_seed_alpha(bridge, it_indices, seed, alpha):
    n_it = int(it_indices.size)
    # a fixed shared common-mode band (a sparse pattern common to ALL objects -> the correlated component).
    srng = np.random.default_rng(seed * 31 + 5)
    shared_band = np.zeros(n_it)
    n_active = max(3, int(PERCEPT_SPARSITY * n_it))
    shared_band[srng.choice(n_it, n_active, replace=False)] = PERCEPT_DRIVE_PA
    proj = _projection(n_it, seed)
    codebook = [_to_phasor(read_rate(bridge, it_indices, i, alpha, shared_band), proj) for i in range(N_OBJECTS)]
    # mean pairwise code similarity (the actual induced correlation, for the curve).
    sims = [float(np.abs(np.vdot(codebook[a], codebook[b])) / D)
            for a in range(N_OBJECTS) for b in range(N_OBJECTS) if a != b]
    mean_code_sim = float(np.mean(sims))
    R_AGENT, R_PATIENT = _role(seed + 1), _role(seed + 2)

    pairs = [(a, b) for a in range(N_OBJECTS) for b in range(N_OBJECTS) if a != b]
    rng = np.random.default_rng(seed * 99 + 7)
    rng.shuffle(pairs)
    memorized, held_out = pairs[:len(pairs) // 2], pairs[len(pairs) // 2:]
    mem_store = [(R_AGENT * codebook[ai] + R_PATIENT * codebook[bi], ai, bi) for (ai, bi) in memorized]

    def _mem_recall(fact, role):
        best, bk = -1.0, 0
        for k, (f, a, b) in enumerate(mem_store):
            c = float(np.abs(np.vdot(f, fact)) / D)
            if c > best:
                best, bk = c, k
        return mem_store[bk][1] if role == "agent" else mem_store[bk][2]

    clean_ok = mem_ok = 0
    for (ai, bi) in held_out:
        fact = R_AGENT * codebook[ai] + R_PATIENT * codebook[bi]
        clean_ok += int(_cleanup(fact * np.conj(R_AGENT), codebook) == ai) + \
                    int(_cleanup(fact * np.conj(R_PATIENT), codebook) == bi)
        mem_ok += int(_mem_recall(fact, "agent") == ai) + int(_mem_recall(fact, "patient") == bi)
    clean = clean_ok / (2 * len(held_out))
    floor = mem_ok / (2 * len(held_out))
    return clean, floor, mean_code_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default="research/findings/raw/_step3_correlated_percept_boundary.json")
    args = ap.parse_args()
    _, backend = get_backend()
    print(f"[step3-corr-boundary] backend={backend} — how much correlated percept does grounded-code compose "
          f"tolerate? (alpha=0 orthogonal .. 1 identical)", flush=True)
    blocks = []
    for alpha in args.alphas:
        cleans, floors, sims = [], [], []
        for s in args.seeds:
            bridge, handles = build_probe_bridge(s)
            c, fl, cs = run_seed_alpha(bridge, handles["it_indices"], s, alpha)
            cleans.append(c); floors.append(fl); sims.append(cs)
        mc, mf, ms = float(np.mean(cleans)), float(np.mean(floors)), float(np.mean(sims))
        ok = all(c >= 0.90 for c in cleans)
        print(f"  alpha={alpha:.2f}  code-sim {ms:.3f}  held-out clean {mc:.3f} (per-seed {[round(c,2) for c in cleans]})"
              f"  floor {mf:.3f}  {'TOLERATED' if ok else 'DEGRADED'}", flush=True)
        blocks.append({"alpha": alpha, "mean_code_sim": ms, "mean_clean": mc, "mean_floor": mf,
                       "per_seed_clean": cleans, "tolerated": ok})
    tol = [b["alpha"] for b in blocks if b["tolerated"]]
    boundary = max(tol) if tol else None
    print(f"\n  TOLERANCE BOUNDARY: grounded-code compose holds (clean>=0.90 all seeds) up to alpha="
          f"{boundary if boundary is not None else 'NONE (degrades even at alpha=0?)'}"
          f"  -> beyond this, semantically-correlated codes need the dendritic / option-B frontier (CLAUDE.md fork).",
          flush=True)
    with open(args.out, "w") as f:
        json.dump({"backend": backend, "boundary_alpha": boundary, "blocks": blocks}, f, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
