"""Unify the two validated brain-analogue threads: the GENUINE-SPIKES unified agent (spiking_unified_agent,
Orchard-2023 phasor populations) running on REAL-V1-SENSORY-GROUNDED concept codes (sim/visual_cortex.py Gabor
bank + ventral-hierarchy decorrelation -- the #4 agent-integration recipe that matched constructed codes 92.3%
in the numpy agent).

This is the most complete brain-analogue conversational artifact so far: fact memory + who/what Q&A + abstention
+ 1/2-attribute composition + embedded clauses, all in spiking-phasor populations, on concept codes derived from
a real biological V1 receptive-field bank rather than random/constructed codes. The numpy result (constructed
parity with decorrelation) sets the expectation; this measures whether the SPIKING substrate reproduces it on
grounded codes. Reported either way (honest -- the spiking quantization to integer spike steps may cost some).

Reuse-by-import only (the grounding pipeline + the benchmark's frozen test set + the validated spiking agent,
which gained a backward-compatible `external_phases` hook). numpy/CPU substrate; the resonator uses the GPU when
present.

  SIM_BACKEND=numpy python -m research.runners.spiking_unified_agent_grounded            # 2-seed
  python -m research.runners.spiking_unified_agent_grounded --seeds 42 43 44 --no-decorrelate
"""
from __future__ import annotations
import argparse
import numpy as np

import research.runners.unified_agent_visual_grounded as uvg
from research.runners._visual_grounding_probe import _v1_matrix
from research.runners.unified_agent_visual_grounded import _v1_codes_for_tokens, _decorrelate
from research.runners.unified_agent_benchmark import build_vocab, aggregate
from research.runners.spiking_unified_agent import run_core_benchmark

D = 2048


def _grounded_phases(v1_codes, n_v1, seed):
    """Project the (decorrelated) V1 codes to D phases in [0,1) -- the spiking agent's external_phases format
    (phases_to_spikes expects [0,1)). angle of a complex projection is uniform-random per code (grounded only
    through the inter-code structure of the V1 features)."""
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((D, n_v1)) + 1j * rng.standard_normal((D, n_v1))
    Z = proj @ v1_codes.T                       # (D, N)
    return np.mod(np.angle(Z) / (2.0 * np.pi), 1.0)   # (D, N) in [0,1)


def main():
    ap = argparse.ArgumentParser(description="Spiking unified agent on real-V1-grounded codes.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--no-decorrelate", action="store_true",
                    help="skip the ventral-hierarchy decorrelation (expect attribute composition to collapse)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    decorr = not args.no_decorrelate
    uvg.STIMULUS_MODE = "tiled"

    nouns, verbs, adjs = build_vocab()
    tokens = list(nouns) + list(verbs) + list(adjs)
    print(f"=== SPIKING unified agent on real-V1-GROUNDED codes | {len(tokens)} concepts | D={D} | "
          f"decorrelate={decorr} | seeds={args.seeds} ===\n", flush=True)
    W, n_v1 = _v1_matrix()
    v1_codes = _v1_codes_for_tokens(tokens, W)        # real Gabor V1 response per concept (seed-independent)
    if decorr:
        v1_codes = _decorrelate(v1_codes)

    seed_results = []
    for s in args.seeds:
        ph = _grounded_phases(v1_codes, n_v1, s)
        ext = {t: ph[:, i] for i, t in enumerate(tokens)}
        res, wrong = run_core_benchmark(n_dim=D, seed=s, n_noun=200, n_verb=60, n_adj=60, external_phases=ext)
        seed_results.append({"seed": s, "categories": res, "wrong": wrong})
        line = "  ".join(f"{c}={res[c][0]}/{res[c][1]}" for c in res)
        print(f"  seed {s}:  {line}", flush=True)

    agg, gok, gtot = aggregate(seed_results)
    print("\n  --- per-category pass-rate (multi-seed, SPIKING + V1-grounded) ---", flush=True)
    for c, (ok, tot, rate) in agg.items():
        print(f"    {c:<16} {ok:>3}/{tot:<3} = {rate*100:5.1f}%", flush=True)
    print(f"\n  OVERALL: {gok}/{gtot} = {gok/gtot*100:.1f}%  ({len(args.seeds)} seeds)", flush=True)
    print("  (numpy agent on the SAME grounded+decorrelated codes = 92.3%, 6-category core 100%, constructed parity)",
          flush=True)

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({"mode": "spiking-visual-grounded", "decorrelate": decorr, "seeds": args.seeds,
                       "D": D, "aggregate": agg, "overall": [gok, gtot], "per_seed": seed_results}, f, indent=2)
        print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
