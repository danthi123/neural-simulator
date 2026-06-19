"""STEP 2a navigation gate (a) — one flagship navigation run, WITH or WITHOUT the conversational populations.

Runs `run_moving_goal_episode` at the G v2.5 + K v2-style recipe (grid-32, multi-goal, the closed-loop BG
clusters + visual cortex), at stdp_w_max=400 (the 5a clip mitigation). `--with-conv` appends the parser +
dlPFC regions and the index-based conv-finalization hook (the merged bridge); without it, the standalone
navigation runs. The merged-vs-standalone navigation score across seeds IS gate (a): the conversational
populations are frozen + disjoint from navigation, so the score should be within run-to-run noise.

Determinism (CUBLAS_WORKSPACE_CONFIG) is set at module top, BEFORE any CuPy import, so both the merged and
standalone runs are reproducible (matching the standalone flagship's --deterministic).
"""
import os

# MUST precede any CuPy import (g11_bg_runner imports CuPy when imported inside main()).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse


def main():
    ap = argparse.ArgumentParser(description="STEP 2a nav gate (a): one flagship nav run, +/- conv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-conv", action="store_true", help="merged bridge (append parser+dlPFC + the hook)")
    ap.add_argument("--co-resident-rf", action="store_true",
                    help="STEP 2b: also append the `rf` composer region (nav-not-regressed-with-rf gate)")
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--out", type=str, required=True)
    # Roadmap #4 (fully-spiking motor read-out): which spiking pool drives the action selection.
    #   motor      = host argmax over motor_X spike counts (the deployed default / the N6 cheat).
    #   thal       = host argmax over the cleanly-selective thalamus (biologized SOURCE, best historically).
    #   spiking_wta= the ACCUMULATE-THEN-COMMIT layer: each commit_X burst threshold-crossing IS the
    #                decision (the host argmax merely OBSERVES which commit pool bursted -> a tie-break of
    #                last resort). Combine with --urgency-max-pa 180 (the recommended Cisek collapsing bound)
    #                so even a weak late release crosses the bound (eliminates the silent-commit fallback).
    # All three already wired through run_moving_goal_episode; this just exposes them on the merged-gate runner.
    ap.add_argument("--readout-source", choices=["motor", "thal", "spiking_wta"], default="motor",
                    help="action-selection read-out (roadmap #4: spiking_wta = the commit-burst IS the decision)")
    ap.add_argument("--urgency-max-pa", type=float, default=0.0,
                    help="Cisek collapsing-bound urgency peak pA (spiking_wta only; RECOMMENDED 180)")
    args = ap.parse_args()

    from research.runners.g11_bg_runner import run_moving_goal_episode

    # The "multi" 4-phase goal schedule, replicated from the CLI (g11_bg_runner.py:8581-8586): 4 corners,
    # transitions at 0/450/900/1350 (the documented multi-goal benchmark).
    gs = args.grid_size
    far = (max(0, gs - 2), max(0, gs - 2))
    far_west = (max(0, 1), max(0, gs - 2))
    sw = (max(0, 1), max(0, 1))
    far_se = (max(0, gs - 2), max(0, 1))
    goal_schedule = [(0, far), (450, far_west), (900, sw), (1350, far_se)]

    # The flagship recipe (the confident CLI->kwarg subset; the merged-vs-standalone delta is the gate, so the
    # exact flag set matters less than using the SAME flags for both — both branches use these).
    kw = dict(
        out_path=args.out, seed=args.seed, n_steps=args.n_steps, grid_size=args.grid_size,
        goal_schedule=goal_schedule,
        enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc_nmda=True,
        enable_visual_cortex=True,
        visual_cortex_action_warmup_steps=600,
        stdp_w_max_override=400.0,
        # Roadmap #4: the read-out source + the collapsing-bound urgency (both already threaded through
        # run_moving_goal_episode). Default motor = the deployed host-argmax read-out (the historical gate).
        readout_source=args.readout_source,
        urgency_max_pA=args.urgency_max_pa,
    )
    if args.with_conv:
        from research.runners.nav_conv_merged_bridge import (
            conv_extra_regions_pathways, finalize_conv_for_nav_gate,
        )
        extra_regions, extra_pathways = conv_extra_regions_pathways(co_resident_rf=args.co_resident_rf)

        def hook(bridge):
            finalize_conv_for_nav_gate(bridge, seed=args.seed)

        kw.update(extra_regions=extra_regions, extra_pathways=extra_pathways,
                  build_with_ou=True, prebuilt_post_init_hook=hook)
        _rf = " + rf" if args.co_resident_rf else ""
        print(f"[nav-gate(a)] seed={args.seed} MERGED (nav+parser+dlPFC{_rf}) -> {args.out}", flush=True)
    else:
        print(f"[nav-gate(a)] seed={args.seed} STANDALONE nav -> {args.out}", flush=True)

    run_moving_goal_episode(**kw)
    print(f"[nav-gate(a)] seed={args.seed} {'MERGED' if args.with_conv else 'STANDALONE'} DONE", flush=True)


if __name__ == "__main__":
    main()
