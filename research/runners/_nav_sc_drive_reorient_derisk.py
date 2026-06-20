"""One-shot de-risk — is the spiking-SC merged-nav re-orient-after-goal-change failure a DRIVE-STRENGTH
gap (closable by raising sc_map->cortex_X) or a deeper re-TARGETING / operating-point floor?

Pre-registered by `research/findings/2026-06-20-nav-loop-closure-derisk.md` (commits cdb2603d/bcb45d38/a45629e2).
That doc LOCALIZED the spiking-SC NO-GO (~58x host, actor partly silent at faithful grid-32): it is NOT the
(already-closed) reentrant thal->cortex arc, and NOT a dendritic credit-assignment wall. The load-bearing gap is
the **SC orienting drive `sc_map -> cortex_X` (weight SC_CORTEX_W, default 18) being too weak to replace the ~150 pA
host Manhattan heuristic as the actor's drive**. Symptom: the agent navigates EARLY goals fine (dist 40->4.6 by
step 400) but CANNOT reliably RE-ORIENT after a goal change (the SC bump's drive into cortex is too weak to override
the stale winner / cross the commit bound for the new cardinal).

THIS PROBE is the ONE informed shot: STRENGTHEN that drive (sweep SC_CORTEX_W over ~3 levels toward the host's ~150 pA
equivalent) and measure whether the per-phase RE-ORIENT QUALITY (final-quarter mean distance AFTER each goal change,
phases 1..3) recovers toward the host-heuristic positive control.

Arms (the EXACT failing --spiking-sc merged kwargs; only SC_CORTEX_W differs across the sweep):
  - host          : the host-heuristic POSITIVE control (heuristic_strength default, NO spiking SC) -- it re-orients.
  - sc_w<LEVEL>   : the spiking-SC arm at sc_map->cortex drive = LEVEL (e.g. 18 current / 60 mid / 150 strong).

Reads, per arm:
  - per-phase finalQ   : final_quarter_mean_distance for EACH phase (phase 0 = early acquisition; 1..3 = post-change
                         re-orient). The PRIMARY metric is the post-change phases.
  - gate_score         : sum of per-phase finalQ (lower = better).
  - first_quarter      : per-phase first-quarter mean distance (re-adapt SPEED).
  - motor_sustain/late : actor firing-presence (overall + 2nd half) -- does a stronger drive keep the actor firing
                         through the re-orient (vs the partial-silence the NO-GO showed)?

Verdict logic:
  GO (drive-strength CONVERTS #6): a stronger SC_CORTEX_W restores robust re-orient -- the post-change finalQ of the
    sc arm approaches the host control (and late_sustain stops collapsing). #6 (SC orienting) converts; the spiking SC
    can replace the host heuristic.
  HONEST NEGATIVE (drive-strength is NOT enough): stronger drive does NOT fix re-orient. Report WHICH:
    - re-TARGETING gap : the SC stays locked on the OLD goal's cardinal even at strong drive (post-change finalQ stays
      high / the action distribution keeps favoring the old phase's winner) -- a re-targeting problem, not a drive gap.
    - operating-point FLOOR : a strong drive SATURATES / destabilizes (gate gets WORSE at the strong level, or the
      sweep is non-monotone with the strong arm regressing) -- the operating-point floor.
  Either way: the host heuristic stays the documented scaffold; the spiking SC is validated for early-goal orienting
  only, and #6 closes as a CHARACTERIZED honest-negative. Do NOT escalate into a multi-knob search.

Anti-cheats:
  - host-heuristic POSITIVE control (it re-orients) anchors the sc-arm degradation.
  - the drive SWEEP itself (monotone toward host? or saturating/regressing?) distinguishes drive-gap from op-floor.
  - the per-goal-phase split (phase 0 early-goal vs phases 1..3 post-change) -- the localized symptom is the split.
  - perception is NOT stripped (enable_visual_cortex stays on; warmup honored).

NO sim/ edit. SC_CORTEX_W is an existing env knob (g11_bg_runner.py:4433); the sweep is env-only. GPU
(SIM_BACKEND=cupy) -- the numpy path is bug-blocked for this neural-critic config.
"""
import os

# MUST precede any CuPy import (g11_bg_runner imports the backend when imported).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# the SC merged het-off op-point (the de-risk's merged-tuned values; same as the NO-GO / loop-closure probe).
os.environ.setdefault("SC_RET_SC", "160")
os.environ.setdefault("SC_REC", "12")
os.environ.setdefault("SC_RET_DRIVE", "3500")
os.environ.setdefault("SC_ROS_US", "40")

import argparse
import json

import numpy as np


def _goal_schedule(gs):
    far = (max(0, gs - 2), max(0, gs - 2))
    far_west = (max(0, 1), max(0, gs - 2))
    sw = (max(0, 1), max(0, 1))
    far_se = (max(0, gs - 2), max(0, 1))
    return [(0, far), (450, far_west), (900, sw), (1350, far_se)]


def run_arm(arm_name, sc_cortex_w, seed, n_steps, grid_size, warmup_steps, out_dir, with_conv):
    """arm_name: 'host' (positive control, no SC) or 'sc' (spiking-SC at drive sc_cortex_w)."""
    from research.runners.g11_bg_runner import run_moving_goal_episode

    out_path = os.path.join(out_dir, f"scdrive_{arm_name}_seed{seed}.json")

    kw = dict(
        out_path=out_path, seed=seed, n_steps=n_steps, grid_size=grid_size,
        goal_schedule=_goal_schedule(grid_size),
        enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,   # the reentrant arc (ON in the NO-GO; loop-closure de-risk settled it)
        enable_cluster_e_topography=True,
        enable_pfc_nmda=True,
        enable_visual_cortex=True,            # perception NOT stripped (anti-cheat)
        visual_cortex_action_warmup_steps=warmup_steps,
        stdp_w_max_override=400.0,
    )

    if arm_name == "sc":
        # the EXACT failing --spiking-sc merged kwargs (verbatim from _nav_gate_merged_run.py); heuristic OFF.
        kw.update(
            enable_spiking_sc=True,
            enable_spiking_sc_approach=True,
            spiking_reward_us=True,
            enable_neural_critic=True,
            spiking_snc=True,
            heuristic_strength=0.0,
        )
        os.environ["SC_CORTEX_W"] = str(float(sc_cortex_w))   # <-- THE SWEEP VARIABLE
    else:
        # host-heuristic positive control: the documented host Manhattan orienting + host reward; NO spiking SC.
        os.environ.pop("SC_CORTEX_W", None)

    if with_conv:
        from research.runners.nav_conv_merged_bridge import (
            conv_extra_regions_pathways, finalize_conv_for_nav_gate,
        )
        extra_regions, extra_pathways = conv_extra_regions_pathways()

        def hook(bridge):
            finalize_conv_for_nav_gate(bridge, seed=seed)

        kw.update(extra_regions=extra_regions, extra_pathways=extra_pathways,
                  build_with_ou=True, prebuilt_post_init_hook=hook)

    print(f"[sc-drive] arm={arm_name} sc_cortex_w={sc_cortex_w if arm_name=='sc' else 'n/a'} "
          f"seed={seed} grid={grid_size} n_steps={n_steps} warmup={warmup_steps} with_conv={with_conv}",
          flush=True)
    run_moving_goal_episode(**kw)

    with open(out_path) as f:
        results = json.load(f)

    ps = results.get("phase_stats", [])
    per_phase_finalQ = [float(p.get("final_quarter_mean_distance", float("nan"))) for p in ps]
    per_phase_firstQ = [float(p.get("first_quarter_mean_distance", float("nan"))) for p in ps]
    per_phase_goal = [list(p.get("goal", [])) for p in ps]
    per_phase_actions = [list(p.get("action_counts", [])) for p in ps]
    gate = results.get("gate_score")
    if gate is None:
        gate = float(sum(per_phase_finalQ)) if per_phase_finalQ else None
    # the post-change re-orient metric: phases 1.. (exclude phase 0 = the initial acquisition)
    post_change_finalQ = per_phase_finalQ[1:] if len(per_phase_finalQ) > 1 else []
    post_change_sum = float(np.nansum(post_change_finalQ)) if post_change_finalQ else float("nan")

    mlog = np.asarray(results.get("motor_counts", []), dtype=float)
    if mlog.size:
        any_fire = (mlog.sum(axis=1) > 0)
        motor_sustain = float(any_fire.mean())
        half = len(any_fire) // 2
        late_sustain = float(any_fire[half:].mean()) if half < len(any_fire) else float("nan")
    else:
        motor_sustain = late_sustain = float("nan")

    summary = {
        "arm": arm_name,
        "sc_cortex_w": (float(sc_cortex_w) if arm_name == "sc" else None),
        "seed": seed, "grid_size": grid_size, "n_steps": n_steps, "warmup_steps": warmup_steps,
        "with_conv": with_conv,
        "per_phase_finalQ": per_phase_finalQ,
        "per_phase_firstQ": per_phase_firstQ,
        "per_phase_goal": per_phase_goal,
        "per_phase_action_counts": per_phase_actions,
        "phase0_finalQ": (per_phase_finalQ[0] if per_phase_finalQ else float("nan")),
        "post_change_finalQ": post_change_finalQ,
        "post_change_finalQ_sum": post_change_sum,
        "gate_score": gate,
        "motor_sustain_frac": motor_sustain,
        "late_motor_sustain_frac": late_sustain,
        "episode_json": out_path,
    }
    print(f"[sc-drive] arm={arm_name} w={summary['sc_cortex_w']}: "
          f"phase0_finalQ={summary['phase0_finalQ']:.3f} "
          f"post_change_finalQ={['%.3f' % x for x in post_change_finalQ]} "
          f"(sum {post_change_sum:.3f}) gate={gate} "
          f"late_sustain={late_sustain:.3f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="One-shot nav SC->cortex drive-strength re-orient de-risk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=480)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="visual_cortex_action_warmup_steps. Default = min(100, n_steps//2). NO-GO grid-32 used 600.")
    ap.add_argument("--sc-drive-levels", type=str, default="18,60,150",
                    help="comma list of sc_map->cortex drive strengths to sweep (default 18,60,150 = current/mid/strong~host).")
    ap.add_argument("--with-conv", action="store_true",
                    help="merged bridge (the NO-GO config). Off = standalone nav SC (faster smoke).")
    ap.add_argument("--no-host", action="store_true", help="skip the host positive control (sweep-only).")
    ap.add_argument("--out", type=str, default="research/findings/raw/nav_gate_2a/scdrive_summary.json")
    args = ap.parse_args()

    warmup = args.warmup_steps if args.warmup_steps is not None else min(100, max(1, args.n_steps // 2))
    levels = [float(x) for x in args.sc_drive_levels.split(",") if x.strip()]
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    # host positive control FIRST (the anchor)
    if not args.no_host:
        summaries.append(run_arm("host", None, args.seed, args.n_steps, args.grid_size, warmup, out_dir, args.with_conv))
    # the SC drive sweep
    for w in levels:
        s = run_arm("sc", w, args.seed, args.n_steps, args.grid_size, warmup, out_dir, args.with_conv)
        s["arm"] = f"sc_w{w:g}"
        summaries.append(s)

    # verdict synthesis
    host = next((x for x in summaries if x["arm"] == "host"), None)
    sc_arms = [x for x in summaries if x["arm"].startswith("sc")]
    sc_arms_sorted = sorted(sc_arms, key=lambda x: x["sc_cortex_w"])

    host_post = host["post_change_finalQ_sum"] if host else None
    host_phase0 = host["phase0_finalQ"] if host else None
    sweep_post = [(x["sc_cortex_w"], x["post_change_finalQ_sum"]) for x in sc_arms_sorted]
    sweep_gate = [(x["sc_cortex_w"], x["gate_score"]) for x in sc_arms_sorted]
    sweep_late = [(x["sc_cortex_w"], x["late_motor_sustain_frac"]) for x in sc_arms_sorted]

    # monotone toward host? (does post-change finalQ DECREASE as drive increases?)
    post_vals = [v for _, v in sweep_post]
    improves_with_drive = (len(post_vals) >= 2 and post_vals[-1] < post_vals[0])
    best_sc_post = min(post_vals) if post_vals else None
    # ratio of best sc post-change to host post-change (>= ~1.0 means approaches host)
    ratio_best_to_host = (host_post / best_sc_post) if (host_post and best_sc_post and best_sc_post > 0) else None

    verdict = {
        "host_phase0_finalQ": host_phase0,
        "host_post_change_finalQ_sum": host_post,
        "sweep_post_change_finalQ_sum": sweep_post,
        "sweep_gate": sweep_gate,
        "sweep_late_sustain": sweep_late,
        "post_change_improves_with_drive": improves_with_drive,
        "best_sc_post_change": best_sc_post,
        "host_over_best_sc_ratio": ratio_best_to_host,
        "NOTE": ("GO if the strong-drive sc arm's post_change_finalQ approaches host (ratio ~1) AND late_sustain "
                 "recovers; HONEST-NEGATIVE if it does not -- then classify: re-TARGETING (stays high / action "
                 "counts favor the old phase winner) vs op-FLOOR (strong arm REGRESSES / non-monotone)."),
    }
    out = {"arms": summaries, "verdict": verdict}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[sc-drive] ===== DRIVE-SWEEP RE-ORIENT VERDICT =====", flush=True)
    for k, v in verdict.items():
        print(f"  {k}: {v}", flush=True)
    print(f"[sc-drive] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
