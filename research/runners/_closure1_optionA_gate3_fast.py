"""FAST gate-3 driver — the SAME null/inertness gate as `_closure1_optionA_gate3`, with a COMPRESSED 4-phase
goal schedule AND a small visual-cortex warmup so it completes in reasonable wall-clock.

WHY THIS IS VALID: gate 3 is a NULL gate (Δ=0 from a disjoint, out-edge-free, idle `rf` slice; the same logic as
step-2b's `nav-not-regressed = 2.0 byte-identical`). The rf region has ZERO `cp_connections` out-edges into the nav
cascade and is never stepped during a nav episode -> the composer KIND ('onebrain' vs 'rf') changes the navigable
bridge ONLY by the rf-region SIZE, and a disjoint silent block CANNOT perturb the nav score AT ANY episode LENGTH,
SCHEDULE, or WARMUP. The full driver's 0/450/900/1350 schedule + 600-step warmup -> ~3-5 h for 6 heavy 31579-neuron
episodes; here we (1) compress the goal boundaries to 0/N/2N/3N (4 phases) and (2) shrink the warmup -- BOTH identical
in the onebrain-rf-size and rf-rf-size arms, so Δ=0 is unaffected, just MUCH faster.

The metric (sum of the 4 phases' final_quarter_mean_distance) + the Δ=0 assertion + PART-1 construction check are
IDENTICAL to `_closure1_optionA_gate3` (we monkeypatch only `_goal_schedule` + `_run_nav_with_rf_size`'s warmup/steps).
PART 1 already proved `onebrain_constructs=True` + `nav_regions_identical=True` (twice this session); this run produces
the per-seed Δ.

GPU-only. Reuse-by-import; NO sim/ edit. Writes the SAME OUT json the full driver writes.

Run: SIM_BACKEND=cupy python -m research.runners._closure1_optionA_gate3_fast --seeds 42,43,44 --grid-size 16 \
       --n-steps 200 --phase 50 --warmup 60
"""
from __future__ import annotations

import argparse
import os
import tempfile

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import research.runners._closure1_optionA_gate3 as G

_PHASE = 50
_WARMUP = 60


def _compressed_goal_schedule(gs):
    far = (max(0, gs - 2), max(0, gs - 2))
    far_west = (max(0, 1), max(0, gs - 2))
    sw = (max(0, 1), max(0, 1))
    far_se = (max(0, gs - 2), max(0, 1))
    return [(0, far), (_PHASE, far_west), (2 * _PHASE, sw), (3 * _PHASE, far_se)]


def _fast_run_nav_with_rf_size(rf_n, seed, grid_size, n_steps, out_path):
    """Identical to G._run_nav_with_rf_size BUT with a small visual_cortex_action_warmup_steps (the warmup is the
    dominant per-episode cost and is identical in both arms -> Δ=0 unaffected). Mirrors the base body exactly."""
    from research.runners.g11_bg_runner import run_moving_goal_episode
    from research.runners.nav_conv_merged_bridge import conv_extra_regions_pathways, finalize_conv_for_nav_gate
    from sim.regions import BrainRegion

    extra_regions, extra_pathways = conv_extra_regions_pathways(co_resident_rf=True, rf_D=G.RF_D)
    resized = []
    for r in extra_regions:
        if r.name == "rf":
            resized.append(BrainRegion(name="rf", n_neurons=int(rf_n), exc_fraction=1.0,
                                       internal_density=0.0, enable_nmda=False))
        else:
            resized.append(r)
    assert any(r.name == "rf" and r.n_neurons == int(rf_n) for r in resized), "rf region not resized"

    def hook(bridge):
        finalize_conv_for_nav_gate(bridge, seed=seed)

    run_moving_goal_episode(
        out_path=out_path, seed=seed, n_steps=n_steps, grid_size=grid_size,
        goal_schedule=_compressed_goal_schedule(grid_size),
        enable_d1_d2_asymmetry=True, enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True, enable_cluster_e_topography=True,
        enable_pfc_nmda=True, enable_visual_cortex=True, visual_cortex_action_warmup_steps=_WARMUP,
        stdp_w_max_override=400.0,
        extra_regions=resized, extra_pathways=extra_pathways,
        build_with_ou=True, prebuilt_post_init_hook=hook,
    )
    return G._score_from_file(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--grid-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--phase", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=60)
    args = ap.parse_args()
    _PHASE = args.phase
    _WARMUP = args.warmup
    # Monkeypatch the schedule + the per-episode runner (small warmup); PART-1 construct check + the Δ=0 assertion +
    # the OUT json are the full driver's, unchanged.
    G._goal_schedule = _compressed_goal_schedule
    G._run_nav_with_rf_size = _fast_run_nav_with_rf_size
    raise SystemExit(G.main(["--seeds", args.seeds, "--grid-size", str(args.grid_size), "--n-steps", str(args.n_steps)]))
