"""Phase-0 de-risk — is the spiking-SC merged-nav NO-GO an OPEN reentrant loop (closable) or a deeper wall?

Pre-registered by `research/findings/2026-06-20-nav-reward-value-loop-deep-research.md`. That doc's central
claim is: the SC-deploy NO-GO (`2026-06-19-nav-spiking-sc-deploy-NO-GO.md`, actor goes silent, ~58x host) is a
LOOP-STABILITY / OPEN-LOOP failure, because the reentrant `thal_X -> cortex_X` self-sustain arc
(`enable_cluster_a_closed_loop`, catalog A.05) was OFF in the failing config, so the open cascade had nothing to
keep the actor firing once the host heuristic drive was removed (`heuristic_strength=0`).

This probe trust-but-verifies that load-bearing premise EMPIRICALLY (CPU/numpy, minutes), and runs the A/B the
doc recommends. CRITICAL CHECK FIRST: the NO-GO was produced by `_nav_gate_merged_run.py --with-conv --spiking-sc`,
which sets `enable_cluster_a_closed_loop=True` in its base kw (line 96). So the premise that the arc is OFF may be
WRONG. We measure the actual `thal_X -> cortex_X` synapse count in the built bridge for each arm, then run a short
episode and read `motor_counts` + `gate_score`.

Arms (all = the failing `--spiking-sc` merged config, only the closed-loop flag differs):
  - closed_on  : enable_cluster_a_closed_loop=True  (the ACTUAL merged-gate default that produced the NO-GO)
  - closed_off : enable_cluster_a_closed_loop=False (the doc's claimed failing config; the lesion / inverse test)

Reads, per arm:
  - n_thal_to_cortex_syn : the reentrant arc synapse count (confirms ON vs OFF structurally)
  - motor_sustain        : fraction of logged steps with any motor firing (the actor-silence signature)
  - late_motor_sustain   : same over the SECOND HALF (post-warmup) — the doc's "drops to ~zero after warmup"
  - total_motor_spikes   : sum over the episode
  - gate_score           : sum of per-phase final-quarter mean distance (lower=better; the nav metric)

Interpretation:
  - If closed_on ALREADY has the arc and the actor ALREADY sustains in a short smoke -> the doc's premise is wrong
    (the loop was closed); the 58x has a different cause -> report honest correction.
  - If closed_on has the arc but the actor STILL goes silent -> loop closure alone is not the fix (escalate).
  - If closed_off (arc absent) goes silent and closed_on (arc present) sustains -> the arc is load-bearing
    (the doc's mechanism, just mis-attributed to the deployed config) -> the lesion confirms loop-stability.

NO sim/ edit. Reuse-by-import of run_moving_goal_episode + the merged conv hook. CPU/numpy first.
"""
import os

# MUST precede any CuPy import (g11_bg_runner imports the backend when imported).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json

import numpy as np


def _count_thal_to_cortex(bridge):
    """Count cp_connections synapses whose pre is in any thal_X and post is in the matching cortex_X.

    Returns (total_thal_to_cortex_matching, per_action_dict). 'Matching' = action-specific thal_X->cortex_X
    (the catalog-A.05 reentrant arc is action-specific, density 0.5 weight 5.0)."""
    from sim.backend import to_host

    rm = bridge.region_manager
    csr = bridge.cp_connections
    indptr_h = to_host(csr.indptr)
    post_h = to_host(csr.indices).astype(np.int64)
    nnz = int(post_h.shape[0])
    pre_h = np.zeros(nnz, dtype=np.int64)
    for r in range(int(csr.shape[0])):
        pre_h[int(indptr_h[r]):int(indptr_h[r + 1])] = r

    actions = ["N", "E", "S", "W"]
    per_action = {}
    total = 0
    region_names = set(rm.region_indices_dict().keys())
    for a in actions:
        tname, cname = f"thal_{a}", f"cortex_{a}"
        if tname not in region_names or cname not in region_names:
            per_action[a] = None
            continue
        t_idx = np.asarray(rm.indices(tname), dtype=np.int64)
        c_idx = np.asarray(rm.indices(cname), dtype=np.int64)
        m = np.isin(pre_h, t_idx) & np.isin(post_h, c_idx)
        n = int(m.sum())
        per_action[a] = n
        total += n
    return total, per_action


def run_arm(closed_loop: bool, seed: int, n_steps: int, grid_size: int, with_conv: bool, out_dir: str):
    from research.runners.g11_bg_runner import run_moving_goal_episode

    gs = grid_size
    far = (max(0, gs - 2), max(0, gs - 2))
    far_west = (max(0, 1), max(0, gs - 2))
    sw = (max(0, 1), max(0, 1))
    far_se = (max(0, gs - 2), max(0, 1))
    goal_schedule = [(0, far), (450, far_west), (900, sw), (1350, far_se)]

    out_path = os.path.join(out_dir, f"loopclose_{'on' if closed_loop else 'off'}_seed{seed}.json")
    box = {}

    kw = dict(
        out_path=out_path, seed=seed, n_steps=n_steps, grid_size=grid_size,
        goal_schedule=goal_schedule,
        enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=closed_loop,   # <-- the A/B variable
        enable_cluster_e_topography=True,
        enable_pfc_nmda=True,
        enable_visual_cortex=True,
        visual_cortex_action_warmup_steps=min(100, max(1, n_steps // 2)),
        stdp_w_max_override=400.0,
        # the failing --spiking-sc arm (verbatim from _nav_gate_merged_run.py)
        enable_spiking_sc=True,
        enable_spiking_sc_approach=True,
        spiking_reward_us=True,
        enable_neural_critic=True,
        spiking_snc=True,
        heuristic_strength=0.0,
    )

    if with_conv:
        from research.runners.nav_conv_merged_bridge import (
            conv_extra_regions_pathways, finalize_conv_for_nav_gate,
        )
        extra_regions, extra_pathways = conv_extra_regions_pathways()

        def hook(bridge):
            finalize_conv_for_nav_gate(bridge, seed=seed)
            box["n_thal_cortex"], box["per_action"] = _count_thal_to_cortex(bridge)
            box["n_regions"] = len(bridge.core_config.brain_regions)
            box["n_neurons"] = int(bridge.core_config.num_neurons)

        kw.update(extra_regions=extra_regions, extra_pathways=extra_pathways,
                  build_with_ou=True, prebuilt_post_init_hook=hook)
    else:
        def hook(bridge):
            box["n_thal_cortex"], box["per_action"] = _count_thal_to_cortex(bridge)
            box["n_regions"] = len(bridge.core_config.brain_regions)
            box["n_neurons"] = int(bridge.core_config.num_neurons)

        kw.update(prebuilt_post_init_hook=hook)

    # the SC merged op-point (the env values from the deploy prep), set in-process
    os.environ.setdefault("SC_RET_SC", "160")
    os.environ.setdefault("SC_REC", "12")
    os.environ.setdefault("SC_RET_DRIVE", "3500")
    os.environ.setdefault("SC_ROS_US", "40")

    print(f"[loop-closure] arm=closed_{'on' if closed_loop else 'off'} seed={seed} grid={grid_size} "
          f"n_steps={n_steps} with_conv={with_conv}", flush=True)
    run_moving_goal_episode(**kw)

    # read the episode JSON for motor_counts + gate_score
    with open(out_path) as f:
        results = json.load(f)
    mlog = np.asarray(results.get("motor_counts", []), dtype=float)   # (n_logged, 4)
    gate = results.get("gate_score", None)
    if gate is None:
        # derive from phase_stats if not top-level
        ps = results.get("phase_stats", [])
        gate = float(sum(p.get("final_quarter_mean_distance", 0.0) for p in ps)) if ps else None

    if mlog.size:
        any_fire = (mlog.sum(axis=1) > 0)
        motor_sustain = float(any_fire.mean())
        half = len(any_fire) // 2
        late_sustain = float(any_fire[half:].mean()) if half < len(any_fire) else float("nan")
        total_spikes = float(mlog.sum())
        n_logged = int(mlog.shape[0])
    else:
        motor_sustain = late_sustain = total_spikes = float("nan")
        n_logged = 0

    summary = {
        "arm": f"closed_{'on' if closed_loop else 'off'}",
        "closed_loop": closed_loop,
        "seed": seed,
        "grid_size": grid_size,
        "n_steps": n_steps,
        "with_conv": with_conv,
        "n_thal_to_cortex_syn": box.get("n_thal_cortex"),
        "thal_to_cortex_per_action": box.get("per_action"),
        "n_regions": box.get("n_regions"),
        "n_neurons": box.get("n_neurons"),
        "n_motor_log_steps": n_logged,
        "motor_sustain_frac": motor_sustain,
        "late_motor_sustain_frac": late_sustain,
        "total_motor_spikes": total_spikes,
        "gate_score": gate,
        "episode_json": out_path,
    }
    print(f"[loop-closure] arm=closed_{'on' if closed_loop else 'off'}: "
          f"thal->cortex syn={summary['n_thal_to_cortex_syn']} "
          f"motor_sustain={motor_sustain:.3f} late={late_sustain:.3f} "
          f"total_spikes={total_spikes:.0f} gate={gate}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Phase-0 nav loop-closure de-risk (CPU smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=120)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--with-conv", action="store_true",
                    help="merged bridge (the NO-GO config). Off = standalone nav SC (faster CPU smoke).")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/nav_gate_2a/loopclose_summary.json")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    for closed in (True, False):
        s = run_arm(closed, args.seed, args.n_steps, args.grid_size, args.with_conv, out_dir)
        summaries.append(s)

    # the A/B verdict line
    on = next(x for x in summaries if x["closed_loop"])
    off = next(x for x in summaries if not x["closed_loop"])
    verdict = {
        "premise_doc_claimed_arc_OFF_in_failing_config": True,
        "arc_present_in_closed_on": (on["n_thal_to_cortex_syn"] or 0) > 0,
        "arc_present_in_closed_off": (off["n_thal_to_cortex_syn"] or 0) > 0,
        "closed_on_motor_sustain": on["motor_sustain_frac"],
        "closed_off_motor_sustain": off["motor_sustain_frac"],
        "closed_on_late_sustain": on["late_motor_sustain_frac"],
        "closed_off_late_sustain": off["late_motor_sustain_frac"],
        "closed_on_gate": on["gate_score"],
        "closed_off_gate": off["gate_score"],
    }
    out = {"arms": summaries, "verdict": verdict}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[loop-closure] ===== A/B VERDICT =====", flush=True)
    for k, v in verdict.items():
        print(f"  {k}: {v}", flush=True)
    print(f"[loop-closure] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
