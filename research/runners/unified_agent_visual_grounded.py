"""#4 agent integration: run the FULL unified-agent benchmark on Gabor-V1-grounded concept codes.

The separability + composition probes showed real V1 Gabor features give usable, composable grounded codes for a
handful of visual concepts. This closes the agent-integration follow-up: assign each of the 320 benchmark
concepts a DISTINCT synthetic visual stimulus, run it through the REAL biological V1 Gabor receptive-field bank
(sim/visual_cortex.py), convert the V1 response to a phasor code (a deterministic function of the sensory
features -> grounded), and feed those as the agent's `external_codes`. Then run the SAME frozen conversational
test set the constructed/word-cue benchmark uses (flat / 1-attr / 2-attr / clause / who / abstain), multi-seed.

Honest framing: the grounding PIPELINE is real V1 (Gabor simple cells); the per-concept stimuli are synthetic
distinct visual textures (we have no natural images for abstract words -- the embodied-cognition limit). The
question this answers: does real sensory grounding feed the WHOLE agent at 320-concept scale, or does visual-space
crowding (many stimuli in one feature space) overlap the codes enough to degrade composition? Reported either way,
side-by-side with the constructed baseline.

Reuse-by-import only; no protected-module edits.

  SIM_BACKEND=numpy python -m research.runners.unified_agent_visual_grounded            # 3-seed
  SIM_BACKEND=numpy python -m research.runners.unified_agent_visual_grounded --quick    # 2-seed smoke
"""
from __future__ import annotations
import argparse
import json
import math

import numpy as np

from research.runners._visual_grounding_probe import _v1_matrix, render_bar, render_spot, _v1_code
from research.runners.unified_agent_benchmark import (
    build_vocab, ALL_FACTS, CATEGORIES, WHO_QUERIES, ABSTAIN_QUERIES, aggregate,
)
from research.runners.nested_composition_agent import NestedCompositionAgent

D = 2048
N_ELEMENTS = 5          # Gabor elements per concept's visual texture (diversity vs crowding sweet spot)
STIMULUS_MODE = "tiled"  # "tiled" = distinct localized patches (separable); "texture" = overlapping random (crowded)


def _render_patch(cx, cy, theta, halflen=4.0, thickness=1.4, size=32):
    """A SHORT oriented Gabor bar localized at (cx,cy) -> drives a distinct cluster of V1 cells (position+orient)."""
    img = np.zeros((2, size, size), dtype=np.float32)
    ct, st = math.cos(theta), math.sin(theta)
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            perp = abs(-dx * st + dy * ct)
            along = abs(dx * ct + dy * st)
            if perp < thickness and along < halflen:
                img[0, y, x] = 1.0
    return img


def generate_stimulus(idx, size=32):
    """A distinct synthetic visual stimulus for concept `idx`. Deterministic per idx (the sensory input is fixed
    -> grounded). Two modes:
      tiled   -- 2 short oriented patches at idx-determined DISTINCT positions+orientations, tiling the visual
                 field so different concepts excite different V1 (position x orientation) cells -> separable codes.
      texture -- N_ELEMENTS overlapping random bars/spots -> codes crowd the V1 feature space (the honest
                 crowding stress that collapses composition)."""
    if STIMULUS_MODE == "texture":
        rng = np.random.default_rng(1000 + idx)
        img = np.zeros((2, size, size), dtype=np.float32)
        for _ in range(N_ELEMENTS):
            if rng.integers(0, 2) == 0:
                img = img + render_bar(float(rng.uniform(0, math.pi)), size=size, thickness=1.4,
                                       shift=(float(rng.uniform(-9, 9)), float(rng.uniform(-9, 9))))
            else:
                img = img + render_spot(float(rng.uniform(6, size - 6)), float(rng.uniform(6, size - 6)),
                                        size=size, r=float(rng.uniform(2.0, 4.0)))
        return np.clip(img, 0.0, 1.0)

    # tiled: spread concepts over an 18x18 position grid x 8 orientations; a 2nd patch (offset) adds richness
    # while keeping each concept's (position, orientation) signature distinct.
    g = 18
    margin = 4.0
    span = size - 2 * margin
    px, py = idx % g, (idx // g) % g
    cx = margin + (px + 0.5) * span / g
    cy = margin + (py + 0.5) * span / g
    th = (idx % 8) * math.pi / 8.0
    img = _render_patch(cx, cy, th, size=size)
    # second patch diagonally offset with a different orientation -> richer, still idx-distinct
    cx2 = margin + ((px + 5) % g + 0.5) * span / g
    cy2 = margin + ((py + 7) % g + 0.5) * span / g
    img = img + _render_patch(cx2, cy2, ((idx // 8) % 8) * math.pi / 8.0, size=size)
    return np.clip(img, 0.0, 1.0)


def _v1_codes_for_tokens(tokens, W):
    """Real V1 Gabor response (unit-normalized) per concept; fixed (seed-independent)."""
    return np.stack([_v1_code(W, generate_stimulus(i)) for i in range(len(tokens))])   # (N, 8192)


def _decorrelate(v1_codes):
    """ZCA-whiten the concept codes so they become mutually low-coherence (orthonormal in their span). A
    biologically-motivated stand-in for the DECORRELATION the ventral visual hierarchy performs (V1->V2->V4->IT;
    efficient coding / redundancy reduction -- Atick-Redlich 1992, Olshausen-Field 1996). A single V1 Gabor layer
    leaves high MAX coherence (near-duplicate stimuli) that the composition resonator can't factor; the hierarchy's
    decorrelation is what yields composition-ready codes. The whitened code is still a fixed linear function of the
    V1 response (grounded)."""
    X = v1_codes - v1_codes.mean(0, keepdims=True)
    G = X @ X.T
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 1e-6, None)
    G_inv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.T
    Xw = G_inv_sqrt @ X                                  # rows orthonormal (decorrelated)
    return Xw / (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-12)


def _projection(n_v1, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((D, n_v1)) + 1j * rng.standard_normal((D, n_v1)))


def _grounded_phasor_codes(tokens, v1_codes, proj):
    """V1 features -> phasor code (FHRR) via a fixed complex projection. Grounded: a deterministic function of the
    sensory features. NOTE the agent's external_codes contract: it does np.exp(1j*ext[token]), so ext[token] must
    be the real PHASE ANGLES (not the complex phasor) -- return angle(Z)."""
    Z = proj @ v1_codes.T               # (D, N) complex
    angles = np.angle(Z)                # (D, N) real phases -- the agent re-exponentiates these
    return {t: angles[:, i] for i, t in enumerate(tokens)}


def run_seed_visual(seed, v1_codes, W, n_v1, tokens):
    proj = _projection(n_v1, seed)
    ext = _grounded_phasor_codes(tokens, v1_codes, proj)
    nouns, verbs, adjs = build_vocab()
    agent = NestedCompositionAgent(nouns, verbs, adjs, D=D, seed=seed, external_codes=ext)
    for ag, ac, pa in ALL_FACTS:
        agent.learn(ag, ac, pa)

    res = {"seed": seed, "categories": {}, "wrong": []}
    for name, facts in CATEGORIES:
        ok = 0
        for ag, ac, pa in facts:
            got = agent.query_patient(ag, ac)
            want = pa if isinstance(pa, str) else agent._render_filler(pa)
            ok += int(got == want)
            if got != want:
                res["wrong"].append({"q": f"what does {ag} {ac}?", "got": got, "want": want, "cat": name})
        res["categories"][name] = [ok, len(facts)]
    who_ok = sum(int(agent.query_agent(ac, pn) == want) for ac, pn, want in WHO_QUERIES)
    res["categories"]["who-query"] = [who_ok, len(WHO_QUERIES)]
    abstain_ok = sum(int(agent.query_patient(ag, ac) is None) for ag, ac in ABSTAIN_QUERIES)
    res["categories"]["abstain"] = [abstain_ok, len(ABSTAIN_QUERIES)]
    return res


def main():
    ap = argparse.ArgumentParser(description="Unified-agent benchmark on Gabor-V1-grounded codes.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--quick", action="store_true", help="2-seed smoke")
    ap.add_argument("--stimulus-mode", choices=["tiled", "texture"], default="tiled",
                    help="tiled=distinct localized patches (separable); texture=overlapping random (crowded)")
    ap.add_argument("--decorrelate", action="store_true",
                    help="ZCA-whiten the V1 codes (ventral-hierarchy decorrelation stand-in) before grounding")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42, 43] if args.quick else args.seeds
    global STIMULUS_MODE
    STIMULUS_MODE = args.stimulus_mode

    nouns, verbs, adjs = build_vocab()
    tokens = list(nouns) + list(verbs) + list(adjs)
    print(f"=== unified-agent benchmark | GROUNDED via real V1 Gabor | {len(tokens)} concepts | D={D} | "
          f"seeds={seeds} ===\n", flush=True)
    W, n_v1 = _v1_matrix()
    v1_codes = _v1_codes_for_tokens(tokens, W)
    if args.decorrelate:
        v1_codes = _decorrelate(v1_codes)
        print("  [decorrelate] ZCA-whitened V1 codes (ventral-hierarchy decorrelation stand-in)", flush=True)

    # report the grounded V1-code separability at 320 (the crowding question)
    C = v1_codes @ v1_codes.T
    off = C[~np.eye(len(tokens), dtype=bool)]
    print(f"  320-concept V1 grounding separability: mean cosine {off.mean():.3f}, max {off.max():.3f}\n", flush=True)

    seed_results = []
    for s in seeds:
        r = run_seed_visual(s, v1_codes, W, n_v1, tokens)
        seed_results.append(r)
        line = "  ".join(f"{c}={r['categories'][c][0]}/{r['categories'][c][1]}" for c in r["categories"])
        print(f"  seed {s}:  {line}", flush=True)

    agg, gok, gtot = aggregate(seed_results)
    print("\n  --- per-category pass-rate (multi-seed, V1-grounded) ---", flush=True)
    for c, (ok, tot, rate) in agg.items():
        print(f"    {c:<16} {ok:>3}/{tot:<3} = {rate*100:5.1f}%", flush=True)
    print(f"\n  OVERALL: {gok}/{gtot} = {gok/gtot*100:.1f}%  ({len(seeds)} seeds)", flush=True)
    print("  (compare constructed baseline: robust 6-category core 100%, clause-depth2 the documented ceiling)",
          flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "visual-grounded", "seeds": seeds, "D": D, "n_concepts": len(tokens),
                       "v1_separability": [float(off.mean()), float(off.max())],
                       "aggregate": agg, "overall": [gok, gtot], "per_seed": seed_results}, f, indent=2)
        print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
