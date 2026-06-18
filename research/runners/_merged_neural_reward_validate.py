"""TRUE-ONE-BRAIN roadmap #2 — 6-seed VALIDATION: the NEURAL reward `r` on the MERGED bridge.

The build (research/findings/2026-06-18-merged-neural-reward-SCOPE-GO.md): make the merged "one brain" nav episode
source its reward `r` SYNAPTICALLY from `sc_rostral->reward_us` firing (the N5 SC proximity / goal-salience approach
reward), retiring the host Manhattan/sign(delta-ecc) formula. Together with the already-committed value-train (learned
V, commit 6fe74bc5), delta = r - V becomes FULLY synaptic on the one brain.

This runner runs the RPE battery + the nav-not-regressed A/B + the moat, multi-seed. GO bar:
  * GRADED   : corr(proximity, reward_us) <= -0.5 (and the gradient SPREADS across eccentricities).
  * BURST    : SNc reward burst >= 1.3x tonic, driven by reward_us FIRING (not a host write).
  * LESION   : sever sc_rostral->reward_us -> the reward / SNc-burst COLLAPSES (3 clean seeds; decisive anti-cheat
               that r is the synaptic SC proximity, not a re-hidden host scalar).
  * NAV      : the merged nav score with the neural reward vs the host reward (an honest regression IS the
               deliverable — the gridworld is orient-solvable, so the reward may not be behaviorally load-bearing).
  * MOAT     : MergedNavConvAgent.what_does('dog','go')=='north' AND what_does('river','look') is None (every build).

Runner-only: the g11_bg_runner.py:7140 reward-routing fix (approach_n5 -> sc_rostral) + the het-off SC operating
point env vars (SC_RET_SC/SC_REC/SC_ROS_US/SC_RET_DRIVE), default-preserving (env unset => standalone byte-identical).

    SIM_BACKEND=cupy python research/runners/_merged_neural_reward_validate.py --seeds 42 43 44 100 101 102
    SIM_BACKEND=numpy python research/runners/_merged_neural_reward_validate.py --smoke   # tiny CPU smoke
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sim.backend import get_backend, to_host

# The het-off merged-tuned SC operating point (the de-risk's validated weights). Set BEFORE the episode runs so the
# env-var overrides at the install_spiking_sc_wiring call site (g11_bg_runner.py) pick them up. The RPE battery
# (_merged_neural_reward_scope_derisk.run) sets its OWN weights post-init directly, so these env vars only affect the
# nav-not-regressed episode path.
SC_OP = dict(SC_RET_SC="160.0", SC_REC="12.0", SC_ROS_US="40.0", SC_RET_DRIVE="3500.0")

xp, BACKEND = get_backend()


def _set_sc_op(on):
    for k, v in SC_OP.items():
        if on:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ── (A) RPE battery + moat: reuse the validated de-risk harness (drives sc_retina directly, measures reward_us/snc) ──
from research.runners._merged_neural_reward_scope_derisk import run as rpe_run, composition_and_moat


def rpe_battery(seed, hold=60):
    """Intact RPE battery (graded proximity reward + SNc burst). Returns the de-risk's result dict."""
    return rpe_run(seed=seed, hold=hold, lesion=False, tag=f"INTACT s{seed}", quiet=True)


def lesion_battery(seed, hold=60):
    """Lesion RPE battery (sc_rostral->reward_us zeroed). Returns the de-risk's result dict."""
    return rpe_run(seed=seed, hold=hold, lesion=True, tag=f"LESION s{seed}", quiet=True)


def moat_check(seed):
    """MergedNavConvAgent moat on the SC-reward bridge: a known fact resolves; an unstored cue abstains -> None."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    agent = MergedNavConvAgent(seed=seed, co_resident_nav_critic=True, nav_critic_spiking_sc=True)
    agent.hear("dog go north")
    resolves = agent.what_does("dog", "go")
    abstains = agent.what_does("river", "look")
    ok = (resolves == "north") and (abstains is None)
    return ok, resolves, abstains


# ── (B) nav-not-regressed: run the merged nav episode with the NEURAL reward vs the HOST reward ──
def _merged_nav_score(seed, n_steps, grid_size, neural_reward, out_dir):
    """Run the navigation episode on the MERGED nav+conv bridge with the FULL nav critic + SC chain, sourcing the
    reward either NEURALLY (neural_reward=True: sc_rostral->reward_us carries r; the het-off SC op-point is set) or
    from the HOST (neural_reward=False: enable_spiking_sc_approach OFF -> g11_bg_runner.py:7154 host reward_us write).
    Returns (mean_distance_overall, gate_score=sum(final_quarter_mean_distance), n_steps_at_goal)."""
    from research.runners.g11_bg_runner import run_moving_goal_episode
    from research.runners.nav_conv_merged_bridge import (
        conv_extra_regions_pathways, finalize_conv_for_nav_gate,
    )
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"nav_{'neural' if neural_reward else 'host'}_seed{seed}.json")
    extra_regions, extra_pathways = conv_extra_regions_pathways()

    def hook(bridge):
        finalize_conv_for_nav_gate(bridge, seed=seed)

    _set_sc_op(neural_reward)   # the het-off SC op-point is only needed when the SC carries the reward
    try:
        results = run_moving_goal_episode(
            out_path=out, seed=seed, n_steps=n_steps, grid_size=grid_size,
            # the FULL nav critic (learned V) -> delta=r-V; with neural_reward, r is also synaptic (the whole point).
            spiking_snc=True, enable_neural_critic=True, spiking_reward_us=True,
            enable_critic_homeostasis=True,
            # the SC chain (vision hierarchy + sc_retina/sc_map/sc_fs/sc_rostral). enable_spiking_sc_approach gates
            # WHETHER reward_us is driven by the SC (neural) or the host write (g11_bg_runner.py:7140 branch).
            enable_visual_cortex=True, enable_spiking_sc=True,
            enable_spiking_sc_approach=bool(neural_reward),
            visual_cortex_action_warmup_steps=min(100, max(1, n_steps // 2)),
            stdp_w_max_override=400.0,
            extra_regions=extra_regions, extra_pathways=extra_pathways,
            build_with_ou=True, prebuilt_post_init_hook=hook,
        )
    finally:
        _set_sc_op(False)
    mean_d = float(results["mean_distance_overall"])
    gate = float(sum(p["final_quarter_mean_distance"] for p in results["phase_stats"]))
    at_goal = int(results.get("n_steps_at_goal", 0))
    return mean_d, gate, at_goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--lesion-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-steps", type=int, default=900, help="nav-not-regressed episode length (per condition)")
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--nav-seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="seeds for the nav-not-regressed A/B (expensive; default 3)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU smoke: 1 seed, short episode")
    ap.add_argument("--out", default="research/findings/raw/merged_neural_reward_validate.json")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [42]
        args.lesion_seeds = [42]
        args.nav_seeds = [42]
        args.n_steps = 120

    print("=" * 92)
    print(f"TRUE-ONE-BRAIN #2 VALIDATION — the NEURAL reward on the MERGED bridge   backend={BACKEND}")
    print("=" * 92)

    report = dict(backend=BACKEND, seeds=args.seeds, rpe={}, lesion={}, moat={}, nav={})

    # ── (A) RPE battery + moat, per seed ──
    print("\n[A] RPE battery (graded proximity reward + SNc burst) + moat, per seed")
    print(f"{'seed':>5} | {'corr(ecc,reward_us)':>19} | {'close/far rew_us Hz':>20} | "
          f"{'SNc burst x':>11} | {'moat':>5}")
    for s in args.seeds:
        r = rpe_battery(s)
        mo_ok, resolves, abstains = moat_check(s)
        burst_x = r["close_snc"] / max(1e-9, r["base_snc"])
        report["rpe"][s] = dict(corr_us=r["corr_us"], close_us=r["close_us"], far_us=r["far_us"],
                                base_snc=r["base_snc"], close_snc=r["close_snc"], burst_x=burst_x,
                                eccs=r["eccs"], us_rs=r["us_rs"], snc_rs=r["snc_rs"])
        report["moat"][s] = dict(ok=mo_ok, resolves=resolves, abstains=abstains)
        print(f"{s:>5} | {r['corr_us']:>19.3f} | {r['close_us']:>8.1f}/{r['far_us']:<11.1f} | "
              f"{burst_x:>11.3f} | {str(mo_ok):>5}")

    # ── (B) lesion (sc_rostral->reward_us zeroed) collapses the reward, clean seeds ──
    print("\n[B] LESION (sever sc_rostral->reward_us) collapses the reward")
    print(f"{'seed':>5} | {'intact close rew_us':>19} | {'lesion close rew_us':>19} | {'collapses?':>10}")
    for s in args.lesion_seeds:
        intact = report["rpe"].get(s) or rpe_battery(s)
        les = lesion_battery(s)
        intact_close = intact["close_us"] if isinstance(intact, dict) else intact["close_us"]
        collapses = les["close_us"] < max(2.0, intact_close * 0.5)
        report["lesion"][s] = dict(intact_close=intact_close, lesion_close=les["close_us"], collapses=collapses)
        print(f"{s:>5} | {intact_close:>19.1f} | {les['close_us']:>19.1f} | {str(collapses):>10}")

    # ── (C) nav-not-regressed: neural reward vs host reward ──
    print(f"\n[C] nav-not-regressed (n_steps={args.n_steps}/cond, grid={args.grid_size}): neural reward vs host reward")
    print(f"{'seed':>5} | {'host meanD':>10} {'host gate':>10} {'host@goal':>10} | "
          f"{'neural meanD':>12} {'neural gate':>12} {'neural@goal':>12} | {'dMeanD':>8}")
    out_dir = "research/findings/raw/merged_neural_reward_nav"
    for s in args.nav_seeds:
        host_d, host_gate, host_goal = _merged_nav_score(s, args.n_steps, args.grid_size, False, out_dir)
        neur_d, neur_gate, neur_goal = _merged_nav_score(s, args.n_steps, args.grid_size, True, out_dir)
        d_mean = neur_d - host_d   # lower meanD = better; positive = neural regressed
        report["nav"][s] = dict(host_meanD=host_d, host_gate=host_gate, host_at_goal=host_goal,
                                neural_meanD=neur_d, neural_gate=neur_gate, neural_at_goal=neur_goal,
                                delta_meanD=d_mean)
        print(f"{s:>5} | {host_d:>10.3f} {host_gate:>10.3f} {host_goal:>10} | "
              f"{neur_d:>12.3f} {neur_gate:>12.3f} {neur_goal:>12} | {d_mean:>+8.3f}")

    # ── VERDICT ──
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    graded = [report["rpe"][s]["corr_us"] <= -0.5 for s in args.seeds]
    bursts = [report["rpe"][s]["burst_x"] >= 1.3 for s in args.seeds]
    moats = [report["moat"][s]["ok"] for s in args.seeds]
    lesions = [report["lesion"][s]["collapses"] for s in args.lesion_seeds]
    n_graded = sum(graded); n_burst = sum(bursts); n_moat = sum(moats); n_les = sum(lesions)
    mean_corr = float(np.mean([report["rpe"][s]["corr_us"] for s in args.seeds]))
    mean_burst = float(np.mean([report["rpe"][s]["burst_x"] for s in args.seeds]))
    print(f"GRADED  (corr <= -0.5): {n_graded}/{len(args.seeds)}   mean corr={mean_corr:.3f}")
    print(f"BURST   (>= 1.3x)     : {n_burst}/{len(args.seeds)}   mean burst={mean_burst:.3f}x")
    print(f"LESION  (collapses)   : {n_les}/{len(args.lesion_seeds)}")
    print(f"MOAT    (intact)      : {n_moat}/{len(args.seeds)}")
    if report["nav"]:
        mean_dmean = float(np.mean([report["nav"][s]["delta_meanD"] for s in args.nav_seeds]))
        print(f"NAV     delta-meanD (neural-host), mean over {len(args.nav_seeds)} seeds: {mean_dmean:+.3f} "
              f"(positive => neural reward regressed nav; the gridworld is orient-solvable so this may be expected)")
        report["nav_mean_delta_meanD"] = mean_dmean
    report["summary"] = dict(n_graded=n_graded, n_burst=n_burst, n_lesion=n_les, n_moat=n_moat,
                             n_seeds=len(args.seeds), n_lesion_seeds=len(args.lesion_seeds),
                             mean_corr=mean_corr, mean_burst=mean_burst)

    rpe_go = (n_graded == len(args.seeds)) and (n_burst == len(args.seeds)) and (n_les == len(args.lesion_seeds))
    moat_ok = (n_moat == len(args.seeds))
    if rpe_go and moat_ok:
        print("\nVERDICT: GO — the reward `r` is sourced SYNAPTICALLY (sc_rostral->reward_us); graded by proximity, "
              "the SNc bursts on it, the lesion collapses it, the moat is intact. delta=r-V is FULLY synaptic on the "
              "one brain. (Report the nav delta honestly — a regression IS a finding, the orient-solvable caveat.)")
        verdict = "GO"
    else:
        print("\nVERDICT: BOUNDARY / honest-negative — see the per-seed numbers. An honest negative (neural reward "
              "sourced but a gate misses, or nav regresses) IS the deliverable.")
        verdict = "BOUNDARY"
    report["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[validate] wrote {args.out}")
    return rpe_go and moat_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
