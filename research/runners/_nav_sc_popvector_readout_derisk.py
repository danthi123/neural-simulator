"""#6 SC orienting read-out BUILD-test — does the population-VECTOR read-out (+ bump-mass divisive
normalization) make the spiking-SC arm's action distribution TRACK the goal and RE-ORIENT after a
goal change, where the deployed half-plane LINEAR-RAMP read-out is stuck-N (the 2026-06-20 NEGATIVE)?

Prescribed by `research/findings/2026-06-20-nav-readout-geometry-deep-research.md` (Option A): the
deployed sc_map->cortex_X read-out (a signed half-plane ramp, g11_bg_runner.py:262-263) is an
UN-normalized weighted SUM (mass-coding, position-INVARIANT). The fix is the SC's canonical
population-VECTOR decode (each site's preferred-direction cosine projection) + the Carandini-Heeger
bump-mass divisive normalization on the four cortex_X pools (input_divisive_norm) + the existing #4
competitive WTA ring (the --spiking-sc config already routes through readout_source="spiking_wta").
All point-neuron; NO sim/ edit (a runner read-out-weight formula + an existing-primitive flag).

Arms (the EXACT failing --spiking-sc merged-het-off kwargs; only the read-out geometry differs):
  - host             : the host-heuristic POSITIVE control (host Manhattan orienting + host reward,
                       NO spiking SC) -- it re-orients (centroid+argmax = a position decode).
  - sc_ramp          : the spiking-SC arm with the DEPLOYED half-plane ramp read-out (= the NEGATIVE).
  - sc_popvector     : the spiking-SC arm with the population-VECTOR read-out + cortex_X divisive norm
                       (the #6 BUILD).
  - sc_popvector_scr : the population-vector arm with the retinotopy SCRAMBLE LESION (SC_SCRAMBLE=1) --
                       MUST collapse (proves the orienting is carried by the RETINOTOPIC decode, not a
                       non-retinotopic leak / a cascade prior). The mandatory anti-cheat lesion.

Reads, per arm:
  - per-phase finalQ        : final_quarter_mean_distance for EACH phase (phase 0 = initial
                              acquisition; phases 1..3 = post-goal-change re-orient). PRIMARY metric.
  - per-phase action_counts : (N,E,S,W) per phase -- the DECISIVE read: must TRACK the goal (W-heavy
                              for a west goal, E-heavy for an east goal), not stuck-N.
  - post_change_finalQ_sum  : sum of phases 1..3 finalQ (the re-orient metric the NEGATIVE was ~73x
                              worse on).
  - motor late_sustain      : actor firing-presence in the 2nd half (the NEGATIVE had ~0.40).

Verdict logic:
  GO (#6 CONVERTS): the sc_popvector arm's per-phase action distribution TRACKS the goal (the
    dominant cardinal SHIFTS toward the goal's bearing across phases, not stuck-N) AND its
    post_change_finalQ approaches the host control materially (vs the ramp arm's ~73x gap). The
    spiking SC orienting is now properly biologized; the host heuristic is no longer needed to
    re-orient. The SCRAMBLE lesion MUST collapse (else the "tracking" is a non-retinotopic artifact).
  HONEST NEGATIVE: if even the population-vector + divnorm read-out cannot track/re-orient, report the
    residual crisply (still under-selective? a deeper issue?). Do NOT loosen anything; do NOT
    config-search beyond the prescribed A+B.

Anti-cheats (all from the deep-research + the build's own lesion):
  - host-heuristic POSITIVE control (centroid+argmax = a position decode) anchors the SC arm.
  - the re-orient-after-change metric (phases 1..3, NOT static hold).
  - the per-phase action distribution (the datum that diagnosed the NEGATIVE).
  - the retinotopy-scramble LESION (sc_popvector_scr MUST regress to chance).
  - matched drive (sc_ramp and sc_popvector both at the SAME SC_CORTEX_W -- attribute to the GEOMETRY,
    not a covert drive increase).
  - perception NOT stripped (enable_visual_cortex on; warmup honored).

NO sim/ edit. GPU (SIM_BACKEND=cupy) -- the numpy path is bug-blocked for this neural-critic config.
"""
import os

# MUST precede any CuPy import (g11_bg_runner imports the backend when imported).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# the SC merged het-off op-point (the de-risk's merged-tuned values; same as the NO-GO drive sweep).
os.environ.setdefault("SC_RET_SC", "160")
os.environ.setdefault("SC_REC", "12")
os.environ.setdefault("SC_RET_DRIVE", "3500")
os.environ.setdefault("SC_ROS_US", "40")

import argparse
import json

import numpy as np

ACTION_NAMES = ["N", "E", "S", "W"]


def _goal_schedule(gs):
    far = (max(0, gs - 2), max(0, gs - 2))        # phase 0: NE corner
    far_west = (max(0, 1), max(0, gs - 2))        # phase 1: NW corner (goal moves WEST)
    sw = (max(0, 1), max(0, 1))                    # phase 2: SW corner (goal moves SOUTH)
    far_se = (max(0, gs - 2), max(0, 1))          # phase 3: SE corner (goal moves EAST)
    return [(0, far), (450, far_west), (900, sw), (1350, far_se)]


def _action_frac(counts):
    """(N,E,S,W) counts -> dict of fractions + the dominant cardinal."""
    c = np.asarray(counts, dtype=float)
    tot = c.sum()
    if tot <= 0:
        return {a: 0.0 for a in ACTION_NAMES}, None
    frac = {ACTION_NAMES[i]: float(c[i] / tot) for i in range(4)}
    dom = ACTION_NAMES[int(np.argmax(c))]
    return frac, dom


def run_arm(arm_name, seed, n_steps, grid_size, warmup_steps, out_dir, with_conv,
            sc_cortex_w, divnorm_sigma, divnorm_gain, cortex_wta=False,
            cortex_fs_weight=8.0, cortex_fs_n=5, fix1=False, fix2=False,
            tie_break_eps=0, fix3=False, opponent_axis_eps=0,
            fixA=False, sel_divnorm_sigma=1.0, sel_divnorm_gain=1.0,
            fixB=False, sel_opponent_weight=12.0, sel_crossaxis_weight=0.0):
    """arm_name in {'host','sc_ramp','sc_popvector','sc_popvector_scr'}.

    cortex_wta (the R1 'sharpen-earlier' next mechanism after the Option-A+B HONEST NEGATIVE): add
    per-cardinal FS interneuron WTA competition DIRECTLY between cortex_N/E/S/W
    (enable_cortex_lateral_inhibition) so the small position-correct SC pop-vector margin can win
    BEFORE the cascade N-bias swamps it at the downstream sel_X ring (the scoping's Option-B failure
    remedy). Pure point-neuron (LIF FS cross-inhibition). Default False = byte-identical.

    fix1 (Cascade North-bias FIX 1): tie-aware STOCHASTIC action read-out -- a K-way tie among the
    sel_X/commit_X accumulators is broken by a uniform draw, NOT the N-first max() ordering (which
    deterministically resolves [40,40,40,40] ties to N). Removes the host tie-break shortcut.
    fix2 (Cascade North-bias FIX 2): per-region homeostasis on the four sel_X pools -- baseline
    equalization at the selection stage so the all-saturate-at-40 ties are reduced at the source.
    fix3 (Cascade North-bias FIX 3): OPPONENT-AXIS push-pull read-out -- decide by a signed margin per
    opponent axis (N-S, E-W) so a faint position-correct 1-D surplus becomes a decisive axis winner,
    not a sub-threshold 4-way contender. Stacks on FIX 1 (genuine both-axes ties fall through to the
    FIX-1 tie-break draw). The scoping's prescribed remedy for the margin-SNR residual left by FIX 1."""
    from research.runners.g11_bg_runner import run_moving_goal_episode

    out_path = os.path.join(out_dir, f"scpv_{arm_name}_seed{seed}.json")

    kw = dict(
        out_path=out_path, seed=seed, n_steps=n_steps, grid_size=grid_size,
        goal_schedule=_goal_schedule(grid_size),
        enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc_nmda=True,
        enable_visual_cortex=True,            # perception NOT stripped (anti-cheat)
        visual_cortex_action_warmup_steps=warmup_steps,
        stdp_w_max_override=400.0,
    )
    if cortex_wta:
        kw["enable_cortex_lateral_inhibition"] = True   # R1: inter-cardinal cortex WTA
        kw["fs_to_cortex_weight"] = float(cortex_fs_weight)   # R1 escalation: WTA strength
        kw["n_cortex_fs_per_action"] = int(cortex_fs_n)
    if fix1:
        kw["sc_tie_break_stochastic"] = True            # FIX 1: tie-aware stochastic read-out
        kw["sc_tie_break_eps"] = int(tie_break_eps)
    if fix2:
        kw["sc_sel_homeostasis"] = True                 # FIX 2: per-pool baseline equalization
    if fix3:
        kw["sc_opponent_axis"] = True                   # FIX 3: opponent-axis push-pull read-out
        kw["sc_opponent_axis_eps"] = int(opponent_axis_eps)
    if fixA:
        kw["sc_sel_divnorm"] = True                     # FIX A: divisive norm at the sel_X input
        kw["sc_sel_divnorm_sigma"] = float(sel_divnorm_sigma)
        kw["sc_sel_divnorm_gain"] = float(sel_divnorm_gain)
    if fixB:
        kw["enable_sel_opponent_pair"] = True           # FIX B: opponent-pair the sel accumulators
        kw["sel_opponent_weight"] = float(sel_opponent_weight)
        kw["sel_crossaxis_weight"] = float(sel_crossaxis_weight)

    # reset the per-run SC env knobs so a prior arm doesn't leak.
    os.environ.pop("SC_CORTEX_W", None)
    os.environ.pop("SC_SCRAMBLE", None)
    os.environ.pop("SC_POPVECTOR", None)

    if arm_name == "host":
        # host-heuristic positive control: the documented host Manhattan orienting + host reward.
        pass
    else:
        # the EXACT failing --spiking-sc merged kwargs; heuristic OFF; matched drive across SC arms.
        kw.update(
            enable_spiking_sc=True,
            enable_spiking_sc_approach=True,
            spiking_reward_us=True,
            enable_neural_critic=True,
            spiking_snc=True,
            heuristic_strength=0.0,
        )
        os.environ["SC_CORTEX_W"] = str(float(sc_cortex_w))   # MATCHED drive across SC arms
        if arm_name in ("sc_popvector", "sc_popvector_scr"):
            kw.update(
                sc_popvector_readout=True,
                sc_popvector_divnorm_sigma=float(divnorm_sigma),
                sc_popvector_divnorm_gain=float(divnorm_gain),
            )
        if arm_name == "sc_popvector_scr":
            os.environ["SC_SCRAMBLE"] = "1"    # the retinotopy LESION anti-cheat

    if with_conv:
        from research.runners.nav_conv_merged_bridge import (
            conv_extra_regions_pathways, finalize_conv_for_nav_gate,
        )
        extra_regions, extra_pathways = conv_extra_regions_pathways()

        def hook(bridge):
            finalize_conv_for_nav_gate(bridge, seed=seed)

        kw.update(extra_regions=extra_regions, extra_pathways=extra_pathways,
                  build_with_ou=True, prebuilt_post_init_hook=hook)

    print(f"[sc-pv] arm={arm_name} seed={seed} grid={grid_size} n_steps={n_steps} "
          f"warmup={warmup_steps} scw={sc_cortex_w} sigma={divnorm_sigma} gain={divnorm_gain} "
          f"with_conv={with_conv}", flush=True)
    run_moving_goal_episode(**kw)

    # clean up the env so it doesn't leak to the next arm.
    os.environ.pop("SC_CORTEX_W", None)
    os.environ.pop("SC_SCRAMBLE", None)
    os.environ.pop("SC_POPVECTOR", None)

    with open(out_path) as f:
        results = json.load(f)

    ps = results.get("phase_stats", [])
    per_phase_finalQ = [float(p.get("final_quarter_mean_distance", float("nan"))) for p in ps]
    per_phase_goal = [list(p.get("goal", [])) for p in ps]
    per_phase_actions = [list(p.get("action_counts", [])) for p in ps]
    per_phase_frac, per_phase_dom = [], []
    for c in per_phase_actions:
        f_, d_ = _action_frac(c)
        per_phase_frac.append({k: round(v, 3) for k, v in f_.items()})
        per_phase_dom.append(d_)
    gate = results.get("gate_score")
    if gate is None:
        gate = float(np.nansum(per_phase_finalQ)) if per_phase_finalQ else None
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

    # Per-stage N-S / E-W surplus (the FIX-A surplus-shrink check). Aggregate the per-step per-cardinal
    # counts (N,E,S,W) over the whole run; report the common-mode N-S surplus (absolute + percent) at each
    # cascade stage. A real FIX A SHRINKS sel_counts/commit_counts N-S toward 0 (the decisive gate).
    def _stage_surplus(key):
        log = results.get(key, [])
        if not log:
            return None
        arr = np.asarray(log, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 4:
            return None
        tot = arr.sum(axis=0)
        N, E, S, W = (float(tot[0]), float(tot[1]), float(tot[2]), float(tot[3]))
        ns = N - S
        ew = E - W
        denom = max(1.0, (N + S) / 2.0)
        return {"N": N, "E": E, "S": S, "W": W,
                "NS_surplus": ns, "NS_pct": round(100.0 * ns / denom, 2),
                "EW_surplus": ew}
    stage_surplus = {k: _stage_surplus(k)
                     for k in ("thal_counts", "sel_counts", "commit_counts", "motor_counts")}

    summary = {
        "arm": arm_name, "seed": seed, "grid_size": grid_size, "n_steps": n_steps,
        "warmup_steps": warmup_steps, "with_conv": with_conv,
        "fix1_tie_break": bool(results.get("sc_tie_break_stochastic", False)),
        "fix2_sel_homeostasis": bool(results.get("sc_sel_homeostasis", False)),
        "fix3_opponent_axis": bool(results.get("sc_opponent_axis", False)),
        "fixA_sel_divnorm": bool(results.get("sc_sel_divnorm", False)),
        "fixA_sel_divnorm_sigma": float(results.get("sc_sel_divnorm_sigma", 1.0)),
        "fixA_sel_divnorm_gain": float(results.get("sc_sel_divnorm_gain", 1.0)),
        "fixB_sel_opponent_pair": bool(results.get("sc_sel_opponent_pair", False)),
        "fixB_sel_opponent_weight": float(results.get("sel_opponent_weight", 12.0)),
        "fixB_sel_crossaxis_weight": float(results.get("sel_crossaxis_weight", 0.0)),
        "stage_surplus": stage_surplus,
        "tie_break_count": int(results.get("tie_break_count", 0)),
        "decision_total": int(results.get("decision_total", 0)),
        "tie_break_fraction": float(results.get("tie_break_fraction", 0.0)),
        "opponent_axis_count": int(results.get("opponent_axis_count", 0)),
        "opponent_axis_fraction": float(results.get("opponent_axis_fraction", 0.0)),
        "sc_cortex_w": (float(sc_cortex_w) if arm_name != "host" else None),
        "divnorm_sigma": (float(divnorm_sigma) if arm_name in ("sc_popvector", "sc_popvector_scr") else None),
        "divnorm_gain": (float(divnorm_gain) if arm_name in ("sc_popvector", "sc_popvector_scr") else None),
        "per_phase_finalQ": per_phase_finalQ,
        "per_phase_goal": per_phase_goal,
        "per_phase_action_counts": per_phase_actions,
        "per_phase_action_frac": per_phase_frac,
        "per_phase_dominant_cardinal": per_phase_dom,
        "phase0_finalQ": (per_phase_finalQ[0] if per_phase_finalQ else float("nan")),
        "post_change_finalQ": post_change_finalQ,
        "post_change_finalQ_sum": post_change_sum,
        "gate_score": gate,
        "motor_sustain_frac": motor_sustain,
        "late_motor_sustain_frac": late_sustain,
        "episode_json": out_path,
    }
    print(f"[sc-pv] arm={arm_name}: phase0_finalQ={summary['phase0_finalQ']:.3f} "
          f"post_change_finalQ={['%.3f' % x for x in post_change_finalQ]} (sum {post_change_sum:.3f}) "
          f"gate={gate} late_sustain={late_sustain:.3f}", flush=True)
    print(f"[sc-pv] arm={arm_name} per-phase dominant cardinal: {per_phase_dom}", flush=True)
    print(f"[sc-pv] arm={arm_name} per-phase action frac: {per_phase_frac}", flush=True)
    print(f"[sc-pv] arm={arm_name} FIX1={summary['fix1_tie_break']} FIX2={summary['fix2_sel_homeostasis']} "
          f"FIX3={summary['fix3_opponent_axis']} FIXA={summary['fixA_sel_divnorm']} "
          f"(sigma={summary['fixA_sel_divnorm_sigma']} gain={summary['fixA_sel_divnorm_gain']}) "
          f"tie_break_fraction={summary['tie_break_fraction']:.4f} "
          f"({summary['tie_break_count']}/{summary['decision_total']})", flush=True)
    _ss = summary.get("stage_surplus") or {}
    for _stg in ("thal_counts", "sel_counts", "commit_counts", "motor_counts"):
        _v = _ss.get(_stg)
        if _v:
            print(f"[sc-pv] arm={arm_name} {_stg:14s} N-S={_v['NS_surplus']:+.0f} ({_v['NS_pct']:+.1f}%) "
                  f"E-W={_v['EW_surplus']:+.0f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="#6 SC population-vector read-out re-orient BUILD-test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=480)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="visual_cortex_action_warmup_steps. Default = min(100, n_steps//2). NEGATIVE grid-32 used 600.")
    ap.add_argument("--sc-cortex-w", type=float, default=18.0,
                    help="MATCHED sc_map->cortex drive across the SC arms (default 18 = the deployed/NEGATIVE level).")
    ap.add_argument("--divnorm-sigma", type=float, default=1.0)
    ap.add_argument("--divnorm-gain", type=float, default=1.0)
    ap.add_argument("--cortex-wta", action="store_true",
                    help="R1 next mechanism: add inter-cardinal FS WTA between cortex_N/E/S/W "
                         "(enable_cortex_lateral_inhibition) to sharpen the SC margin EARLIER, "
                         "before the sel_X ring's N-bias swamping. Applies to the SC arms.")
    ap.add_argument("--cortex-fs-weight", type=float, default=8.0,
                    help="R1 escalation: fs_to_cortex inhibitory weight (builder default 8). "
                         "Raise to make the inter-cardinal WTA STRONGER. Only with --cortex-wta.")
    ap.add_argument("--cortex-fs-n", type=int, default=5,
                    help="R1 escalation: n_cortex_fs_per_action (builder default 5). Only with --cortex-wta.")
    ap.add_argument("--fix1", action="store_true",
                    help="Cascade North-bias FIX 1: tie-aware STOCHASTIC action read-out (break K-way "
                         "ties by a uniform draw, NOT the N-first max() ordering). Applies to the SC arms.")
    ap.add_argument("--fix2", action="store_true",
                    help="Cascade North-bias FIX 2: per-region homeostasis on the four sel_X pools "
                         "(baseline equalization at the selection stage). Applies to the SC arms. "
                         "Stack with --fix1 if FIX 1 alone leaves residual bias.")
    ap.add_argument("--tie-break-eps", type=int, default=0,
                    help="FIX 1 tie tolerance (counts): an action is 'tied' if its count >= leader - eps.")
    ap.add_argument("--fix3", action="store_true",
                    help="Cascade North-bias FIX 3: OPPONENT-AXIS push-pull read-out (decide by a signed "
                         "margin per opponent axis N-S / E-W). The scoping's remedy for the margin-SNR "
                         "residual; stack with --fix1. Applies to the SC arms.")
    ap.add_argument("--opponent-axis-eps", type=int, default=0,
                    help="FIX 3 axis tie tolerance (counts): both axes tie if |axis margin| <= eps -> fall "
                         "through to the FIX-1 tie-break.")
    ap.add_argument("--fixA", action="store_true",
                    help="Cascade-accumulator FIX A: DIVISIVE NORMALIZATION at the sel_X accumulator INPUT "
                         "(divide each sel_X by sigma + gain*mean over the four sel pools -> common-mode "
                         "rejection BEFORE the Wang-2002 amplification). The scoping's rank-1 remedy; stack "
                         "with --fix1. Applies to the SC arms.")
    ap.add_argument("--sel-divnorm-sigma", type=float, default=1.0,
                    help="FIX A semi-saturation sigma on the sel_X divisive pool.")
    ap.add_argument("--sel-divnorm-gain", type=float, default=1.0,
                    help="FIX A divisive strength on the four-sel mean term.")
    ap.add_argument("--fixB", action="store_true",
                    help="Cascade-accumulator FIX B: OPPONENT-PAIR the sel accumulators (N<->S, E<->W "
                         "integrate the DIFFERENCE via balanced sel_FS axis-partner inhibition -> the "
                         "common-mode N-S offset cancels structurally; Bogacz 2006). The scoping's rank-2 "
                         "remedy if FIX A over-flattens/under-shrinks. Stack with --fix1. Applies to the SC arms.")
    ap.add_argument("--sel-opponent-weight", type=float, default=12.0,
                    help="FIX B: strong balanced sel_FS_X -> axis-partner inhibitory weight.")
    ap.add_argument("--sel-crossaxis-weight", type=float, default=0.0,
                    help="FIX B: weak/zero cross-axis sel_FS_X -> non-partner inhibitory weight.")
    ap.add_argument("--with-conv", action="store_true",
                    help="merged bridge (the NEGATIVE config). Off = standalone nav SC (faster smoke).")
    ap.add_argument("--no-host", action="store_true", help="skip the host positive control.")
    ap.add_argument("--no-scramble", action="store_true", help="skip the scramble lesion arm (faster smoke).")
    ap.add_argument("--arms", type=str, default=None,
                    help="comma list to restrict arms (host,sc_ramp,sc_popvector,sc_popvector_scr).")
    ap.add_argument("--out", type=str, default="research/findings/raw/nav_gate_2a/scpv_summary.json")
    args = ap.parse_args()

    warmup = args.warmup_steps if args.warmup_steps is not None else min(100, max(1, args.n_steps // 2))
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        arms = ["host", "sc_ramp", "sc_popvector", "sc_popvector_scr"]
        if args.no_host:
            arms = [a for a in arms if a != "host"]
        if args.no_scramble:
            arms = [a for a in arms if a != "sc_popvector_scr"]

    summaries = []
    for arm in arms:
        summaries.append(run_arm(arm, args.seed, args.n_steps, args.grid_size, warmup,
                                 out_dir, args.with_conv, args.sc_cortex_w,
                                 args.divnorm_sigma, args.divnorm_gain,
                                 cortex_wta=args.cortex_wta,
                                 cortex_fs_weight=args.cortex_fs_weight,
                                 cortex_fs_n=args.cortex_fs_n,
                                 fix1=args.fix1, fix2=args.fix2,
                                 tie_break_eps=args.tie_break_eps,
                                 fix3=args.fix3,
                                 opponent_axis_eps=args.opponent_axis_eps,
                                 fixA=args.fixA,
                                 sel_divnorm_sigma=args.sel_divnorm_sigma,
                                 sel_divnorm_gain=args.sel_divnorm_gain,
                                 fixB=args.fixB,
                                 sel_opponent_weight=args.sel_opponent_weight,
                                 sel_crossaxis_weight=args.sel_crossaxis_weight))

    by = {s["arm"]: s for s in summaries}
    host = by.get("host")
    ramp = by.get("sc_ramp")
    pv = by.get("sc_popvector")
    scr = by.get("sc_popvector_scr")

    # GOAL-TRACKING test: does the dominant cardinal SHIFT toward the goal's bearing across phases?
    # The schedule's goals: phase0 NE, phase1 NW(west), phase2 SW(south), phase3 SE(east). A goal-
    # tracking read-out should NOT output the SAME cardinal every phase (the stuck-N NEGATIVE
    # signature). We report (a) #distinct dominant cardinals across phases, (b) whether any phase is
    # W-dominant (the far-west goal) and any is E-dominant (the SE goal) -- the host's signature.
    def _track(s):
        if not s:
            return None
        doms = s.get("per_phase_dominant_cardinal", [])
        n_distinct = len(set(d for d in doms if d is not None))
        has_W = any(d == "W" for d in doms)
        has_E = any(d == "E" for d in doms)
        return {"dominant_per_phase": doms, "n_distinct_dominant": n_distinct,
                "has_W_dominant_phase": has_W, "has_E_dominant_phase": has_E,
                "tracks_goal": (n_distinct >= 2 and (has_W or has_E))}

    verdict = {
        "host_phase0_finalQ": (host["phase0_finalQ"] if host else None),
        "host_post_change_finalQ_sum": (host["post_change_finalQ_sum"] if host else None),
        "ramp_post_change_finalQ_sum": (ramp["post_change_finalQ_sum"] if ramp else None),
        "popvector_post_change_finalQ_sum": (pv["post_change_finalQ_sum"] if pv else None),
        "scramble_post_change_finalQ_sum": (scr["post_change_finalQ_sum"] if scr else None),
        "host_tracking": _track(host),
        "ramp_tracking": _track(ramp),
        "popvector_tracking": _track(pv),
        "scramble_tracking": _track(scr),
        "ramp_late_sustain": (ramp["late_motor_sustain_frac"] if ramp else None),
        "popvector_late_sustain": (pv["late_motor_sustain_frac"] if pv else None),
        "popvector_fix1": (pv["fix1_tie_break"] if pv else None),
        "popvector_fix2": (pv["fix2_sel_homeostasis"] if pv else None),
        "popvector_fix3": (pv["fix3_opponent_axis"] if pv else None),
        "popvector_fixA": (pv["fixA_sel_divnorm"] if pv else None),
        "popvector_stage_surplus": (pv["stage_surplus"] if pv else None),
        "ramp_stage_surplus": (ramp["stage_surplus"] if ramp else None),
        "scramble_stage_surplus": (scr["stage_surplus"] if scr else None),
        "popvector_tie_break_fraction": (pv["tie_break_fraction"] if pv else None),
        "popvector_opponent_axis_fraction": (pv["opponent_axis_fraction"] if pv else None),
        "scramble_tie_break_fraction": (scr["tie_break_fraction"] if scr else None),
        "scramble_opponent_axis_fraction": (scr["opponent_axis_fraction"] if scr else None),
    }
    # host_over_popvector ratio on the post-change re-orient (>= ~1 means popvector approaches host).
    if host and pv:
        hp = host["post_change_finalQ_sum"]
        pp = pv["post_change_finalQ_sum"]
        verdict["host_over_popvector_post_ratio"] = (float(hp / pp) if (hp and pp and pp > 0) else None)
        verdict["popvector_over_ramp_post_improvement"] = (
            float(ramp["post_change_finalQ_sum"] / pp)
            if (ramp and pp and pp > 0 and ramp.get("post_change_finalQ_sum")) else None)
    verdict["NOTE"] = (
        "GO (#6 CONVERTS) if popvector_tracking.tracks_goal is True AND popvector_post_change "
        "approaches host (host_over_popvector_post_ratio ~1, vs the ramp's ~0.01) AND the SCRAMBLE "
        "lesion collapses (scramble_tracking.tracks_goal False / much worse finalQ). HONEST NEGATIVE "
        "otherwise -- report whether popvector still does NOT track (under-selective) or a deeper issue."
    )

    out = {"arms": summaries, "verdict": verdict}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[sc-pv] ===== POPULATION-VECTOR READ-OUT RE-ORIENT VERDICT =====", flush=True)
    for k, v in verdict.items():
        print(f"  {k}: {v}", flush=True)
    print(f"[sc-pv] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
