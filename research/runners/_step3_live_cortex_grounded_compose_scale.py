"""Step-3 SCALED stress — does the LIVE cortex_it RATE-derived grounded-code COMPOSE hold as the object
vocabulary grows (cleanup over the codebook becomes non-trivial)?

The cheap-first GO (`_step3_live_cortex_grounded_compose_probe.py`) showed the MECHANISM works — a live spiking
`cortex_it` rate code -> fixed projection -> phasor -> compose, with held-out (never-composed) facts recovering FAR
above a recall baseline (held-out 1.000 vs mem-floor 0.500). BUT that was 4 objects (chance 0.25), where cleanup
over 4 codes is trivial — the 1.000 absolute is a ceiling artifact, NOT a scaled validation.

THIS run is the honest stress the cheap-first demanded: scale the object vocabulary to 8/16/32 (chance 0.125 ->
0.031) on the SAME 256-neuron cortex_it substrate, and ask whether the live spiking rate code keeps the grounded
phasor codes separable enough to (a) clean up and (b) GENERALIZE (held-out >> recall floor). A degradation curve
localizes the limit to the rate code's capacity at fixed neuron count (-> population code / more cortex_it neurons /
a learned projection) — an honest map either way.

Percepts use an ADAPTIVE sparsity so n_active < stride = 256/n_objects (the codes stay separable percepts, isolating
the rate->phasor->cleanup question from a drive-overlap confound). The corrupt test re-reads a live noisy percept;
it is SAMPLED to K pairs (logged) to bound bridge stepping. Clean + mem-floor run on ALL held-out pairs (pure
algebra, no stepping).

GATE (per n_objects, multi-seed): GO if held-out clean >= 0.90 AND held-out corrupt >= 0.80 AND clean >= floor+0.30
on every seed. GPU (`SIM_BACKEND=cupy`) — the user's standing preference for decisive/scaled runs.
  SIM_BACKEND=cupy python -m research.runners._step3_live_cortex_grounded_compose_scale --n-objects 8 16 32
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.backend import get_backend, to_host
from sim.text_embeddings import orthogonal_drive_pattern

from research.runners.funcint_perception_to_memory_probe import (
    N_CORTEX_IT, PERCEPT_DRIVE_PA, build_probe_bridge,
)

D = 2048
RATE_READ_STEPS = 80
SETTLE_STEPS = 20
CORRUPT_SAMPLE = 48          # sampled held-out pairs for the (live, expensive) corrupted-percept compose test.


def _adaptive_sparsity(n_objects):
    """n_active < stride = 256/n_objects so the orthogonal percepts stay separable (isolates rate->phasor->cleanup
    from a drive-overlap confound). Use ~half the stride, floored at 3 active neurons."""
    stride = N_CORTEX_IT / n_objects
    n_active = max(3, int(0.5 * stride))
    return n_active / N_CORTEX_IT, n_active


def _projection(n_in, seed=42):
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


def read_rate(bridge, it_indices, obj_idx, n_objects, sparsity, corrupt_rng=None):
    """LIVE cortex_it spiking firing-rate read for `obj_idx` under an n_objects-orthogonal percept (optionally
    CORRUPTED). Returns a host float vector (256,) = the grounded rate code."""
    xp, _ = get_backend()
    n_it = int(it_indices.size)
    drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=n_objects, n_neurons=n_it,
                                     drive_max_pA=PERCEPT_DRIVE_PA, sparsity=sparsity).astype(np.float64)
    if corrupt_rng is not None:
        drive = drive + corrupt_rng.normal(0.0, 0.35 * PERCEPT_DRIVE_PA, size=drive.shape)
        drop = corrupt_rng.random(drive.shape) < 0.15
        drive[drop] = 0.0
        drive = np.maximum(drive, 0.0)
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


def run_seed(seed, n_objects):
    bridge, handles = build_probe_bridge(seed)
    it_indices = handles["it_indices"]
    sparsity, n_active = _adaptive_sparsity(n_objects)
    proj = _projection(N_CORTEX_IT, seed)
    codebook = [_to_phasor(read_rate(bridge, it_indices, i, n_objects, sparsity), proj) for i in range(n_objects)]
    R_AGENT, R_PATIENT = _role(seed + 1), _role(seed + 2)
    crng = np.random.default_rng(seed * 99 + 7)

    pairs = [(a, b) for a in range(n_objects) for b in range(n_objects) if a != b]
    crng.shuffle(pairs)
    n_mem = len(pairs) // 2
    memorized, held_out = pairs[:n_mem], pairs[n_mem:]

    mem_store = [(R_AGENT * codebook[ai] + R_PATIENT * codebook[bi], ai, bi) for (ai, bi) in memorized]

    def _mem_recall(fact, role):
        best, bk = -1.0, 0
        for k, (f, a, b) in enumerate(mem_store):
            c = float(np.abs(np.vdot(f, fact)) / D)
            if c > best:
                best, bk = c, k
        return mem_store[bk][1] if role == "agent" else mem_store[bk][2]

    # clean compose + mem-floor on ALL held-out pairs (pure algebra, no bridge stepping).
    clean_ok = mem_ok = 0
    for (ai, bi) in held_out:
        fact = R_AGENT * codebook[ai] + R_PATIENT * codebook[bi]
        ra = _cleanup(fact * np.conj(R_AGENT), codebook)
        rb = _cleanup(fact * np.conj(R_PATIENT), codebook)
        clean_ok += int(ra == ai) + int(rb == bi)
        mem_ok += int(_mem_recall(fact, "agent") == ai) + int(_mem_recall(fact, "patient") == bi)
    clean_acc = clean_ok / (2 * len(held_out))
    mem_floor = mem_ok / (2 * len(held_out))

    # corrupt compose on a SAMPLED subset (live noisy re-read of the agent slot) — bounds bridge stepping.
    sample = held_out if len(held_out) <= CORRUPT_SAMPLE else [held_out[i] for i in
              np.random.default_rng(seed * 7 + 1).choice(len(held_out), CORRUPT_SAMPLE, replace=False)]
    corrupt_ok = 0
    for (ai, bi) in sample:
        a_noisy = _to_phasor(read_rate(bridge, it_indices, ai, n_objects, sparsity, corrupt_rng=crng), proj)
        fact_c = R_AGENT * a_noisy + R_PATIENT * codebook[bi]
        corrupt_ok += int(_cleanup(fact_c * np.conj(R_AGENT), codebook) == ai)
    corrupt_acc = corrupt_ok / len(sample)

    chance = 1.0 / n_objects
    print(f"    [seed {seed} n_obj={n_objects} n_active={n_active}] held-out compose clean {clean_acc:.3f} "
          f"corrupt {corrupt_acc:.3f} (n={len(sample)}) | mem-floor {mem_floor:.3f} (chance {chance:.3f})",
          flush=True)
    return {"seed": seed, "n_objects": n_objects, "n_active": n_active, "clean": clean_acc,
            "corrupt": corrupt_acc, "mem_floor": mem_floor, "chance": chance,
            "n_held_out": len(held_out), "n_corrupt_sample": len(sample)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-objects", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default="research/findings/raw/_step3_live_cortex_grounded_compose_scale.json")
    args = ap.parse_args()
    _, backend = get_backend()
    print(f"[step3-scale] backend={backend} — does live cortex_it RATE-grounded compose hold as vocab grows?",
          flush=True)
    blocks = []
    for n in args.n_objects:
        print(f"  n_objects={n}:", flush=True)
        rows = [run_seed(s, n) for s in args.seeds]
        clean = float(np.mean([r["clean"] for r in rows]))
        corrupt = float(np.mean([r["corrupt"] for r in rows]))
        floor = float(np.mean([r["mem_floor"] for r in rows]))
        go = (all(r["clean"] >= 0.90 for r in rows) and all(r["corrupt"] >= 0.80 for r in rows)
              and all(r["clean"] >= r["mem_floor"] + 0.30 for r in rows))
        partial = (clean >= 0.80 and clean >= floor + 0.30)
        verdict = "GO" if go else ("PARTIAL" if partial else "NEGATIVE")
        print(f"    => n_obj={n}: MEAN clean {clean:.3f} corrupt {corrupt:.3f} mem-floor {floor:.3f} "
              f"(chance {rows[0]['chance']:.3f})  [{verdict}]", flush=True)
        blocks.append({"n_objects": n, "verdict": verdict, "mean_clean": clean, "mean_corrupt": corrupt,
                       "mean_floor": floor, "chance": rows[0]["chance"], "per_seed": rows})
    print(f"\n  SCALING CURVE: " + "  ".join(f"n{b['n_objects']}={b['mean_clean']:.2f}/{b['mean_corrupt']:.2f}"
          f"(floor {b['mean_floor']:.2f})[{b['verdict']}]" for b in blocks), flush=True)
    overall = "GO" if all(b["verdict"] == "GO" for b in blocks) else (
        "PARTIAL" if any(b["verdict"] in ("GO", "PARTIAL") for b in blocks) else "NEGATIVE")
    print(f"  OVERALL: {overall} (clean/corrupt; floor in parens; GO needs clean>=.90 corrupt>=.80 clean>=floor+.30)",
          flush=True)
    with open(args.out, "w") as f:
        json.dump({"overall": overall, "backend": backend, "blocks": blocks}, f, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    raise SystemExit(0 if overall == "GO" else (2 if overall == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
