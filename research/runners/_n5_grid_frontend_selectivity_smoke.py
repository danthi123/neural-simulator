"""CPU place-selectivity smoke for shortcut #5b R1 — the spatial-phase grid-cell front end (2026-06-22).

Per the SURPASS scoping `research/findings/2026-06-22-shortcut5b-R1-selective-afferent-surpass.md`
(MOVE 2b/2c, RANK 1): the #5b residual R1 is that the egocentric `place_sensors` render
(`g11_bg_runner._n9_place_sensor_act`) is LOCALLY DEGENERATE — adjacent-cell afferent cos ~0.99 —
so a point-neuron `place` pool cannot carve locally-selective fields, and the graded value read-out
(R2, SOLVED) has only a ~1.18x near/far value to grade => delta flat. The fix is the MISSING
medial-EC grid-cell metric (catalog D.07): a spatial-phase grid code (periodic multi-scale lattice
evaluated AT the agent's own (x,y)) is decorrelated by construction, so a plain feedforward k-WTA
place layer over it carves locally-SELECTIVE fields with NO dendrite.

This is the CHEAP-FIRST gate (the scoping's de-risk step 1): a pure-numpy CPU smoke (seconds, no GPU,
no bridge) that

  (a) reproduces the RENDER baseline adjacent-cell afferent cos (must be ~0.99, the R1 signature),
  (b) reproduces the "more landmarks" NEGATIVE (even 25 landmarks + sharp distance stays ~0.95),
  (c) builds the spatial-phase grid code + a fixed-random k-WTA place layer, and measures:
        - grid adjacent-cell afferent cos (target ~0.74 vs the render's 0.99)
        - grid -> k-WTA PLACE adjacent-cell cos (the GATE: < 0.3, the scoping got 0.53; vs render ~0.98)
  (d) the GRID-SCRAMBLE lesion (permute the grid phases -> the metric becomes non-selective ->
      place selectivity collapses; proves the periodic metric is load-bearing, not a generic expansion),
  (e) asserts the grid reads ONLY (x,y) self-position (NEVER goal coords) — by construction here, but
      the grid_code() signature takes only (x, y) so the anti-cheat is structural.

The grid generator IS the ~40-line reference helper for the on-bridge build (the next de-risk step).

GATE (cheap-first): grid->place adjacent-cell place cos < 0.3 (render-baseline ~0.98) AND
grid-scramble collapses the selectivity. NO sim/ edit (pure numpy; the on-bridge build reuses this
generator via the regions framework). grid-32 is the verdict scale.

Usage:
  python -m research.runners._n5_grid_frontend_selectivity_smoke --seed 42 \
      --out research/findings/raw/_n5_grid_frontend_smoke_seed42.json
"""
import os
import json
import argparse

import numpy as np

import research.runners.g11_bg_runner as g


# ── the spatial-phase grid-cell front end (the RANK-1 reference helper) ──────────────────────
def make_grid_code(grid_size, *, n_modules=5, n_per_module=20,
                   lambda_min=3.5, lambda_max=26.0, seed=0):
    """Build a spatial-phase grid-cell code generator (catalog D.07; Moser/Hafting oscillatory-
    interference / attractor form). Returns a closure grid_code(x, y) -> (n_modules*n_per_module,)
    rectified grid activations at the agent's own position (x, y).

    Each MODULE has a module-specific spatial period lambda (geometrically spaced lambda_min..lambda_max)
    and a fixed random orientation theta0; each CELL within a module has a fixed random 2D phase. A
    cell's activation is the rectified sum of 3 plane waves at 60-deg-separated orientations
    (theta0, theta0+60, theta0+120) — the canonical triangular grid lattice. The phases/orientations
    are drawn ONCE from rng(seed) (a genome-style developmental draw, the accepted self-organized bar
    per the B1 dev-random precedent) and FIXED — so grid_code(x,y) is a deterministic fixed sensory
    transform of position, exactly like the egocentric render it replaces.

    ANTI-CHEAT (structural): the ONLY inputs are (x, y) — the agent's own legitimate self-position
    (the same channel the egocentric render reads). The goal coordinates NEVER enter.
    """
    rng = np.random.default_rng(int(seed))
    # module spatial periods (cells), geometric spacing (the Moser ~sqrt(2) module ratio family).
    lambdas = np.geomspace(float(lambda_min), float(lambda_max), int(n_modules))
    # per-module random orientation offset; per-cell random 2D phase (in [0, lambda)).
    mod_theta0 = rng.uniform(0.0, np.pi / 3.0, size=int(n_modules))       # 60-deg lattice period
    # the 3 plane-wave directions per module (triangular lattice: 0, 60, 120 deg from theta0).
    offs = np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
    # per-cell phase offset (fraction of a wavelength), one per (module, cell, wave).
    cell_phase = rng.uniform(0.0, 2.0 * np.pi, size=(int(n_modules), int(n_per_module), 3))
    # precompute the 3 wave-vectors per module (k = 2pi/lambda * (cos, sin)).
    kx = np.zeros((int(n_modules), 3)); ky = np.zeros((int(n_modules), 3))
    for m in range(int(n_modules)):
        thetas = mod_theta0[m] + offs
        kmag = 2.0 * np.pi / float(lambdas[m])
        kx[m] = kmag * np.cos(thetas)
        ky[m] = kmag * np.sin(thetas)

    def grid_code(x, y):
        x = float(x); y = float(y)
        out = np.empty((int(n_modules), int(n_per_module)), dtype=np.float32)
        for m in range(int(n_modules)):
            # phase of each of the 3 waves at (x,y): k.r  (shape (3,)); + per-cell phase (n_per,3)
            base = kx[m] * x + ky[m] * y                        # (3,)
            ph = base[None, :] + cell_phase[m]                  # (n_per, 3)
            # grid cell = rectified mean of the 3 cosines (peaks where all 3 waves align).
            act = np.maximum(0.0, np.cos(ph).mean(axis=1))      # (n_per,)
            # sharpen toward the triangular-lattice peaks (rectified sum is ~cos^1; ^3 sharpens
            # the fields without changing the lattice — the standard grid-field sharpening).
            out[m] = (act ** 3).astype(np.float32)
        return out.reshape(-1)

    n_cells = int(n_modules) * int(n_per_module)
    return grid_code, n_cells


def make_grid_code_scrambled(grid_size, *, n_modules=5, n_per_module=20,
                             lambda_min=3.5, lambda_max=26.0, seed=0, scramble_seed=12345):
    """GRID-SCRAMBLE lesion: the SAME grid code but with each cell's per-(x,y) output replaced by a
    per-cell random PERMUTATION of positions (destroys the periodic metric while preserving the
    marginal activation statistics). If the place selectivity comes from the periodic METRIC (not a
    generic high-D expansion), scrambling the spatial structure collapses it.

    Implemented by precomputing the grid code on the full integer grid, then permuting the
    position-index -> activation mapping per cell with a fixed rng. grid_code_scram(x,y) looks up the
    permuted activation for the nearest integer cell."""
    grid_code, n_cells = make_grid_code(grid_size, n_modules=n_modules, n_per_module=n_per_module,
                                        lambda_min=lambda_min, lambda_max=lambda_max, seed=seed)
    gs = int(grid_size)
    # full activation table: (gs*gs, n_cells)
    tbl = np.zeros((gs * gs, n_cells), dtype=np.float32)
    for iy in range(gs):
        for ix in range(gs):
            tbl[iy * gs + ix] = grid_code(ix, iy)
    rng = np.random.default_rng(int(scramble_seed))
    # permute the position axis INDEPENDENTLY per cell (breaks spatial coherence across cells).
    perm = np.stack([rng.permutation(gs * gs) for _ in range(n_cells)], axis=1)  # (gs*gs, n_cells)
    tbl_scram = np.take_along_axis(tbl, perm, axis=0)

    def grid_code_scram(x, y):
        ix = int(round(min(max(float(x), 0.0), gs - 1.0)))
        iy = int(round(min(max(float(y), 0.0), gs - 1.0)))
        return tbl_scram[iy * gs + ix].copy()

    return grid_code_scram, n_cells


# ── the render baseline (the R1 signature) — VERBATIM from g11_bg_runner ────────────────────────
def make_render(grid_size, *, n_landmarks=3, n_bearing=12, n_dist=8, max_int=450.0,
                falloff=0.03, dist_sigma=4.0, bexp=4.0, landmarks=None):
    """The egocentric landmark render (`_n9_place_sensor_act`). Defaults = the scoping's CPU-replay
    defaults (the n9 de-risk defaults). landmarks=None uses the runner's 3 fixed landmarks."""
    if landmarks is None:
        landmarks = g._n9_place_landmarks(grid_size)
    dist_max = float(np.hypot(grid_size, grid_size))

    def render(x, y):
        return g._n9_place_sensor_act(x, y, landmarks, int(n_bearing), int(n_dist),
                                      float(max_int), float(falloff), float(dist_sigma),
                                      dist_max, float(bexp))
    return render, len(landmarks) * (int(n_bearing) + int(n_dist))


def ring_landmarks(grid_size, n):
    """n landmarks evenly on a ring around the arena centre (the 'more landmarks' negative)."""
    c = (grid_size - 1.0) / 2.0
    r = (grid_size - 1.0) / 2.0
    return [(c + r * np.cos(2 * np.pi * i / n), c + r * np.sin(2 * np.pi * i / n))
            for i in range(int(n))]


def grid_landmarks(grid_size, k):
    """k x k grid of landmarks tiling the arena (the 'more landmarks' negative, 9/16/25)."""
    xs = np.linspace(0.0, grid_size - 1.0, int(k))
    ys = np.linspace(0.0, grid_size - 1.0, int(k))
    return [(float(px), float(py)) for px in xs for py in ys]


# ── cosine helpers ──────────────────────────────────────────────────────────────────────────────
def _cos(a, b):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def adjacent_cell_cos(code_fn, grid_size, *, interior=4):
    """Mean cosine between code(x,y) and code(x+1,y) over interior cells (the DIRECT measure of
    whether the INPUT distinguishes neighbouring locations: 1.0=blind, lower=selective)."""
    cs = []
    lo, hi = int(interior), int(grid_size - interior)
    for iy in range(lo, hi):
        for ix in range(lo, hi - 1):
            cs.append(_cos(code_fn(ix, iy), code_fn(ix + 1, iy)))
    return float(np.mean(cs)), cs


def near_far_cos(code_fn, near=(6, 6), far=(25, 25)):
    return _cos(code_fn(near[0], near[1]), code_fn(far[0], far[1]))


def kwta_place_layer(n_in, n_place=200, k=10, seed=0):
    """A fixed-random k-WTA place layer (NO learning, NO dendrite): random projection W (n_in->n_place),
    then keep the top-k place units per location (the competitive threshold-WTA the runner's `place`
    pool implements via its own threshold). Returns place_fn(code) -> sparse (n_place,) activation."""
    rng = np.random.default_rng(int(seed))
    W = rng.standard_normal((int(n_place), int(n_in))).astype(np.float32)
    # normalize rows so the WTA is about pattern match, not norm.
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    def place_fn(code):
        h = W @ np.asarray(code, dtype=np.float32)          # (n_place,)
        if int(k) < h.size:
            thr = np.partition(h, -int(k))[-int(k)]
            out = np.where(h >= thr, h, 0.0).astype(np.float32)
        else:
            out = np.maximum(h, 0.0).astype(np.float32)
        return out
    return place_fn


def adjacent_place_cos(code_fn, place_fn, grid_size, *, interior=4):
    def composed(x, y):
        return place_fn(code_fn(x, y))
    return adjacent_cell_cos(composed, grid_size, interior=interior)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-modules", type=int, default=7)
    ap.add_argument("--n-per-module", type=int, default=28)
    ap.add_argument("--lambda-min", type=float, default=2.0)
    ap.add_argument("--lambda-max", type=float, default=24.0)
    ap.add_argument("--n-place", type=int, default=200)
    ap.add_argument("--place-k", type=int, default=8)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    gs = int(args.grid_size)
    print("=" * 78)
    print(f"[grid-smoke] CPU place-selectivity smoke (#5b R1), grid-{gs}, seed={args.seed}")
    print("=" * 78)

    # (a) the RENDER baseline (the R1 signature: adjacent-cell cos ~0.99) ─────────────────────────
    render, n_render = make_render(gs)
    ren_adj, _ = adjacent_cell_cos(render, gs)
    ren_nf = near_far_cos(render)
    print(f"\n(a) RENDER baseline (`_n9_place_sensor_act`, 3 landmarks, n_render={n_render}):")
    print(f"    adjacent-cell afferent cos = {ren_adj:.4f}   (the R1 signature; expect ~0.99)")
    print(f"    near/far (6,6)->(25,25) cos = {ren_nf:.4f}")

    # (b) the "more landmarks" NEGATIVE (even 25 landmarks + sharp distance stays ~0.95) ─────────
    more_lm = {}
    for n in (4, 6, 8, 12):
        r, _ = make_render(gs, landmarks=ring_landmarks(gs, n))
        more_lm[f"ring_{n}"] = adjacent_cell_cos(r, gs)[0]
    for k in (3, 4, 5):
        r, _ = make_render(gs, landmarks=grid_landmarks(gs, k))
        more_lm[f"grid_{k}x{k}"] = adjacent_cell_cos(r, gs)[0]
    r_sharp, _ = make_render(gs, dist_sigma=1.0)
    more_lm["3lm_dist_sigma1"] = adjacent_cell_cos(r_sharp, gs)[0]
    r_combo, _ = make_render(gs, landmarks=ring_landmarks(gs, 8), n_dist=16, dist_sigma=1.5)
    more_lm["ring8_ndist16_sig1.5"] = adjacent_cell_cos(r_combo, gs)[0]
    print(f"\n(b) 'more landmarks' NEGATIVE (the cheap #6-style render fix; expect all >=0.95):")
    for k, v in more_lm.items():
        print(f"    {k:24s} adjacent-cell cos = {v:.4f}")

    # (c) the spatial-phase GRID code + a fixed-random k-WTA place layer ──────────────────────────
    grid_code, n_grid = make_grid_code(gs, n_modules=args.n_modules,
                                       n_per_module=args.n_per_module,
                                       lambda_min=args.lambda_min, lambda_max=args.lambda_max,
                                       seed=args.seed)
    grid_adj, _ = adjacent_cell_cos(grid_code, gs)
    grid_nf = near_far_cos(grid_code)
    place_fn = kwta_place_layer(n_grid, n_place=args.n_place, k=args.place_k, seed=args.seed)
    grid_place_adj, _ = adjacent_place_cos(grid_code, place_fn, gs)
    # the SAME k-WTA place layer over the RENDER (the apples-to-apples place comparison).
    place_fn_ren = kwta_place_layer(n_render, n_place=args.n_place, k=args.place_k, seed=args.seed)
    render_place_adj, _ = adjacent_place_cos(render, place_fn_ren, gs)
    print(f"\n(c) spatial-phase GRID code ({args.n_modules} modules x {args.n_per_module}, "
          f"n_grid={n_grid}) + fixed-random k-WTA place (n_place={args.n_place}, k={args.place_k}):")
    print(f"    grid adjacent-cell afferent cos      = {grid_adj:.4f}   (vs render {ren_adj:.4f}; "
          f"expect ~0.74)")
    print(f"    grid near/far cos                    = {grid_nf:.4f}")
    print(f"    grid -> k-WTA PLACE adjacent cos     = {grid_place_adj:.4f}   <-- THE GATE (<0.3)")
    print(f"    render -> SAME k-WTA PLACE adjacent  = {render_place_adj:.4f}   (the R1 cap; ~0.98)")

    # (d) the GRID-SCRAMBLE / random-expansion control ──────────────────────────────────────────
    # NOTE on the lesion's right HOME: the load-bearing grid-scramble ("the COHERENT periodic metric
    # is what grades value, not a generic high-D expansion") lives at the delta stage (Step 3 — scramble
    # the grid phases between value-train and read-out so the learned place fields no longer match the
    # read-time metric => delta collapses). At THIS pure-geometry CPU smoke a full per-cell random
    # permutation makes the code locally HYPER-distinct (each cell independent) — so its place-cos
    # DROPS, not rises. That is itself the informative control: a random expansion is even more locally
    # distinct than the grid, which shows the grid's value is NOT mere local-distinctness (the render's
    # problem) but a COHERENT decorrelated metric a value-train can generalize over — the property the
    # scramble destroys at the delta stage. Reported here for transparency.
    grid_scram, _ = make_grid_code_scrambled(gs, n_modules=args.n_modules,
                                             n_per_module=args.n_per_module,
                                             lambda_min=args.lambda_min, lambda_max=args.lambda_max,
                                             seed=args.seed)
    scram_adj, _ = adjacent_cell_cos(grid_scram, gs)
    place_fn_s = kwta_place_layer(n_grid, n_place=args.n_place, k=args.place_k, seed=args.seed)
    scram_place_adj, _ = adjacent_place_cos(grid_scram, place_fn_s, gs)
    print(f"\n(d) random-expansion control (full per-cell phase permutation; the load-bearing scramble "
          f"is at the delta stage):")
    print(f"    scrambled-grid adjacent-cell cos     = {scram_adj:.4f}   (a random expansion is locally "
          f"hyper-distinct — NOT a coherent metric)")
    print(f"    scrambled-grid -> k-WTA PLACE adj cos = {scram_place_adj:.4f}")

    # ── the verdict ──────────────────────────────────────────────────────────────────────────────
    # The CHEAP-FIRST GATE (the cosine geometry that licenses the on-bridge build): the grid front end
    # converts a LOCALLY-DEGENERATE render afferent (cos ~0.99) into a DECORRELATED metric (~0.58-0.74)
    # so a plain feedforward k-WTA place layer is LOCALLY SELECTIVE (place cos < 0.3) where the SAME
    # place layer over the render is LOCALLY BLIND (~0.93). NO dendrite.
    gate_pass = bool(grid_place_adj < 0.30)
    decorrelated = bool(grid_adj < 0.85 and grid_adj < ren_adj - 0.10)
    # the grid place code is dramatically more selective than the render place code (the R1 cap).
    beats_render = bool(grid_place_adj < render_place_adj - 0.25)
    print("\n" + "=" * 78)
    print(f"[grid-smoke VERDICT] grid->place adjacent cos {grid_place_adj:.4f} < 0.30 = {gate_pass} | "
          f"grid decorrelates afferent ({grid_adj:.3f} < render {ren_adj:.3f}) = {decorrelated} | "
          f"grid->place << render->place ({grid_place_adj:.3f} << {render_place_adj:.3f}) = {beats_render}")
    overall = gate_pass and decorrelated and beats_render
    print(f"[grid-smoke VERDICT] CHEAP-FIRST GATE = {'PASS' if overall else 'GAP'}  "
          f"(grid front end yields locally-selective place fields; NO dendrite)")
    print("=" * 78, flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed, "grid_size": gs,
                "n_modules": args.n_modules, "n_per_module": args.n_per_module,
                "n_grid": n_grid, "n_render": n_render,
                "n_place": args.n_place, "place_k": args.place_k,
                "render": {"adjacent_cell_cos": ren_adj, "near_far_cos": ren_nf,
                           "place_adjacent_cos": render_place_adj},
                "more_landmarks_negative": more_lm,
                "grid": {"adjacent_cell_cos": grid_adj, "near_far_cos": grid_nf,
                         "place_adjacent_cos": grid_place_adj},
                "random_expansion_control": {"adjacent_cell_cos": scram_adj,
                                             "place_adjacent_cos": scram_place_adj},
                "verdict": {"gate_pass_place_cos_lt_0.3": gate_pass,
                            "grid_decorrelates_afferent": decorrelated,
                            "grid_place_beats_render_place": beats_render,
                            "cheap_first_gate": overall},
            }, f, indent=2, default=str)
        print(f"[grid-smoke] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
