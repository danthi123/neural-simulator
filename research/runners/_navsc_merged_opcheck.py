"""STEP-0 co-residence OP-POINT check for the spiking superior-colliculus (N1 orienting
+ N5 approach-reward) DEPLOY onto the merged "one brain" bridge.

Pre-registered by `research/findings/2026-06-19-tier2-nav-spikeification-scoping.md` §4 Step 0.

WHY THIS EXISTS (the single most likely spurious-NEGATIVE cause):
the standalone spiking SC is 6-seed GO (`2026-06-10-N1-spiking-superior-colliculus-CLOSED.md`)
but the MERGED bridge runs heterogeneity OFF + global homeostasis OFF for nav/conv determinism.
The companion limbic lift (`2026-06-18-merged-limbic-core-lift.md`) found the SNc's effective
synaptic response is ~6-10x WEAKER co-resident regardless of het — i.e. a standalone-tuned
spiking organ STARVES on the merged bridge. The SC `sc_map` bump faces the SAME risk: at the
standalone weights (w_ret_sc=80 / w_sc_rec=6 / drive=2500) the de-risk reported `sc_map` fires
~2 Hz and `reward_us` never crosses threshold. The de-risk's merged-tuned op-point is
SC_RET_SC=160 / SC_REC=12 / drive=3500 / sc_rostral->reward_us=40. This probe BUILDS the merged
bridge with the SC slice co-resident, installs the SC wiring at the promoted op-point, drives the
egocentric retina on hand-set (agent, goal) renders, and asserts the bump is ALIVE (not starved):

  (a) the `sc_map` Mexican-hat WTA bump FIRES — peak-site rate >> background rate;
  (b) the orienting cardinal BY FIRING (winning cortex_X pool) matches the host
      `sc_orienting_cardinal_from_image` on >= 7/8 positions;
  (c) `reward_us` crosses threshold AND corr(eccentricity, reward_us-rate) < -0.6
      (closer goal => more reward_us firing).

This is the `sc_map_orienting_probe.py` / `sc_n5_rpe_probe.py` falsifiers re-run on the *merged*
bridge (the standalone probes build their OWN tiny bridge — they cannot catch the co-residence
starvation). Cheap (CPU/numpy or tiny GPU smoke); it GATES the expensive 6-seed A/B.

If the bump is starved at the promoted op-point, this probe is the place to identify the knob
(SC_RET_SC / SC_REC / SC_RET_DRIVE / SC_ROS_US) that restores it.

Run (CPU smoke — numpy backend, no env needed; the merged op-point is set in-process):
    SIM_BACKEND=numpy python -m research.runners._navsc_merged_opcheck --seed 42
Or a single-seed GPU smoke (faster build):
    SIM_BACKEND=cupy   python -m research.runners._navsc_merged_opcheck --seed 42
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sim.backend import get_backend, to_host
from sim.visual_cortex import image_to_retina_drive, render_gridworld_to_image
from research.runners.g11_bg_runner import (
    render_egocentric_goal,
    install_spiking_sc_wiring,
    sc_orienting_cardinal_from_image,
    ACTION_NAMES,
)
from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge

xp, BACKEND = get_backend()

# Geometry of the deployed nav SC chain (g11_bg_runner.py:2488 build, default visual_image_size=32).
IMG = 32
GRID = 8                    # the merged smoke runs grid 8 (nav_on_merged_smoke); the egocentric render is grid-agnostic.

# The merged-tuned op-point (the de-risk's promoted values, 2026-06-18-merged-limbic-core-lift.md
# §3 + the env-var defaults in g11_bg_runner.py:4435-4459/:6736). These STARVE-fix the het-off bump.
MERGED_OP = dict(w_ret_sc=160.0, w_sc_rec=12.0, w_sc_cortex=18.0, ret_drive=3500.0, ros_us=40.0)
# The standalone op-point (for the A/B contrast — does the merged op-point actually help?).
STANDALONE_OP = dict(w_ret_sc=80.0, w_sc_rec=6.0, w_sc_cortex=18.0, ret_drive=2500.0, ros_us=14.0)

# F1 orienting cases (agent, goal): all four cardinals + diagonals (dominant-axis), grid-8 frame.
F1_CASES = [
    ((4, 4), (4, 6)),   # goal N
    ((4, 4), (4, 2)),   # goal S
    ((4, 4), (6, 4)),   # goal E
    ((4, 4), (2, 4)),   # goal W
    ((4, 4), (6, 5)),   # NE, E-dominant
    ((4, 4), (5, 6)),   # NE, N-dominant
    ((4, 4), (2, 3)),   # SW
    ((4, 4), (1, 4)),   # far W
]
# F2/N5 proximity cases: a single goal at (4,4), the agent at varied eccentricity. corr(ecc, reward_us)
# should be strongly negative (closer goal => bigger central bump => more sc_rostral => more reward_us).
N5_CASES = [
    ((4, 4), (4, 4)),   # on goal: ecc 0
    ((3, 4), (4, 4)),   # ecc 1
    ((2, 4), (4, 4)),   # ecc 2
    ((1, 4), (4, 4)),   # ecc 3
    ((4, 1), (4, 4)),   # ecc 3 (other axis)
    ((6, 6), (4, 4)),   # ecc ~2.83 diag
]


def _gidx(bridge, name):
    return np.asarray(list(bridge.region_manager.indices(name)), dtype=np.int64)


def _build_merged_with_sc(seed, op):
    """Build the merged nav+conv bridge with the spiking SC co-resident, install the SC wiring at
    the given op-point, and set the resting nav config (OU off, like the episode). Returns the bridge.

    nav_critic_spiking_sc=True forwards enable_visual_cortex + enable_spiking_sc + enable_spiking_sc_approach
    + spiking_reward_us=True (the build wires reward_us + the sc_rostral->reward_us pathway). The post-init
    SC wiring (retinotopy + Mexican-hat-quadrant pooling + sc_map->sc_rostral) is NOT done by the builder
    (it lives in run_moving_goal_episode's post-init hook), so we install it here at the promoted op-point —
    exactly what the deploy will do via finalize_conv_for_nav_gate's sibling hook."""
    bridge, handles = build_merged_nav_conv_bridge(
        seed=seed, co_resident_nav_critic=True, nav_critic_spiking_sc=True)
    # The promoted merged op-point: retina->sc_map=160, recurrent=12, sc_map->cortex_X pooling=18.
    install_spiking_sc_wiring(bridge, visual_image_size=IMG,
                              w_ret_sc=op["w_ret_sc"], w_sc_rec=op["w_sc_rec"],
                              w_sc_cortex=op["w_sc_cortex"], scramble=False, verbose=True)
    # Boost sc_rostral->reward_us to the merged op-point (the build declares 14.0; de-risk used 40.0).
    rm = bridge.region_manager
    _ros = np.asarray(list(rm.indices("sc_rostral")), dtype=np.int64)
    _us = np.asarray(list(rm.indices("reward_us")), dtype=np.int64)
    _pre = np.repeat(_ros, _us.shape[0]).astype(np.int64)
    _post = np.tile(_us, _ros.shape[0]).astype(np.int64)
    bridge.set_pathway_weights("sc_rostral_to_reward_us", _pre, _post,
                               np.full(_pre.size, float(op["ros_us"]), np.float32), add_missing=True)
    # Resting nav config (the episode runs OU OFF; the parser pass already trained + restored).
    bridge.core_config.enable_ou_process = False
    # Capture a clean resting state so each presentation starts identically (the probe-clean-reset discipline).
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge


def _hard_reset(bridge):
    bridge.cp_membrane_potential_v[:] = bridge._rest_v
    bridge.cp_recovery_variable_u[:] = bridge._rest_u
    bridge.cp_conductance_g_e[:] = 0.0
    bridge.cp_conductance_g_i[:] = 0.0
    if getattr(bridge, "cp_conductance_g_nmda", None) is not None:
        bridge.cp_conductance_g_nmda[:] = 0.0
    bridge.cp_firing_states[:] = False
    bridge.cp_refractory_timers[:] = 0
    bridge.cp_external_input_current[:] = 0.0


def _present(bridge, agent, goal, op, n_steps=160, warm=30):
    """Drive the SC's egocentric eye with the (agent, goal) render and read the per-region firing.
    Returns (sc_map peak-site count, sc_map mean-site count, cortex_X counts dict, reward_us rate)."""
    _hard_reset(bridge)
    sc_ret = _gidx(bridge, "sc_retina")
    sc_map = _gidx(bridge, "sc_map")
    rus = _gidx(bridge, "reward_us")
    ctx = {a: _gidx(bridge, f"cortex_{a}") for a in ACTION_NAMES}
    ego = render_egocentric_goal((int(agent[0]), int(agent[1])), (int(goal[0]), int(goal[1])), image_size=IMG)
    egd = image_to_retina_drive(ego, drive_max_pA=float(op["ret_drive"]))
    sc_ret_dev = xp.asarray(sc_ret)
    egd_dev = xp.asarray(egd, dtype=xp.float32)
    for _ in range(3):
        bridge._run_one_simulation_step()
    # Per-site sc_map spike accumulation + cortex + reward_us.
    sc_counts = np.zeros(sc_map.shape[0], dtype=np.int64)
    cc = {a: 0 for a in ACTION_NAMES}
    rus_count = 0
    m = 0
    sc_map_dev = xp.asarray(sc_map)
    ctx_dev = {a: xp.asarray(ctx[a]) for a in ACTION_NAMES}
    rus_dev = xp.asarray(rus)
    for t in range(n_steps):
        # Re-assert the egocentric SC drive each step (the deploy drives it every nav step).
        bridge.cp_external_input_current[sc_ret_dev] = egd_dev
        bridge._run_one_simulation_step()
        if t >= warm:
            fs = bridge.cp_firing_states
            sc_counts += to_host(fs[sc_map_dev]).astype(np.int64)
            for a in ACTION_NAMES:
                cc[a] += int(to_host(fs[ctx_dev[a]]).sum())
            rus_count += int(to_host(fs[rus_dev]).sum())
            m += 1
    m = max(m, 1)
    peak = int(sc_counts.max())
    mean = float(sc_counts.mean())
    rus_rate = rus_count / float(m * max(1, rus.shape[0])) * 1000.0   # Hz (dt=1ms)
    sc_peak_hz = peak / float(m) * 1000.0
    sc_mean_hz = mean / float(m) * 1000.0
    return sc_peak_hz, sc_mean_hz, cc, rus_rate


def _cardinal(cc):
    if max(cc.values()) == 0:
        return None
    top = sorted(cc.values())
    if len(top) >= 2 and top[-1] == top[-2] and top[-1] > 0:
        return "TIE"
    return max(cc, key=lambda a: cc[a])


def _ecc(agent, goal):
    return float(((goal[0] - agent[0]) ** 2 + (goal[1] - agent[1]) ** 2) ** 0.5)


def run_check(seed, op, label):
    print(f"\n================ STEP-0 SC op-check on the MERGED bridge: {label} "
          f"(seed {seed}, backend {BACKEND}) ================")
    print(f"  op-point: w_ret_sc={op['w_ret_sc']} w_sc_rec={op['w_sc_rec']} "
          f"sc_map->cortex_X={op['w_sc_cortex']} ret_drive={op['ret_drive']} sc_rostral->reward_us={op['ros_us']}")
    bridge = _build_merged_with_sc(seed, op)
    rm = bridge.region_manager
    print(f"  merged bridge: {len(bridge.core_config.brain_regions)} regions, "
          f"{int(bridge.core_config.num_neurons)} neurons; SC slice present: "
          f"{'sc_map' in rm.region_indices_dict()}, sc_rostral: {'sc_rostral' in rm.region_indices_dict()}, "
          f"reward_us: {'reward_us' in rm.region_indices_dict()}")

    # --- (a) bump alive + (b) orienting cardinal vs host ---
    print("\n[a+b — bump + N1 orienting]  the sc_map bump must FIRE; the winning cortex_X (BY FIRING) match host")
    print(f"{'agent':>8} {'goal':>8} | {'host':>5} | {'sc_map peak/mean Hz':>20} | {'cortex N/E/S/W':>16} | {'SC':>5} | match")
    n1_ok = n1_tot = 0
    peak_hz_all = []
    mean_hz_all = []
    for agent, goal in F1_CASES:
        host = sc_orienting_cardinal_from_image(
            render_gridworld_to_image(agent_pos=agent, goal_pos=goal, grid_size=GRID, image_size=IMG))
        pk, mn, cc, _ = _present(bridge, agent, goal, op)
        sc = _cardinal(cc)
        ok = (sc == host) and (host is not None)
        n1_tot += 1
        n1_ok += int(ok)
        peak_hz_all.append(pk)
        mean_hz_all.append(mn)
        fired = "/".join(str(cc[a]) for a in ACTION_NAMES)
        print(f"{str(agent):>8} {str(goal):>8} | {str(host):>5} | {pk:8.1f}/{mn:7.1f} | "
              f"{fired:>16} | {str(sc):>5} | {'OK' if ok else 'x'}")
    peak_med = float(np.median(peak_hz_all))
    mean_med = float(np.median(mean_hz_all))
    bump_ratio = peak_med / max(mean_med, 1e-6)

    # --- (c) N5 reward_us proximity ---
    print("\n[c — N5 approach-reward]  reward_us must cross threshold; corr(eccentricity, reward_us-rate) < -0.6")
    print(f"{'agent':>8} {'goal':>8} | {'ecc':>5} | {'sc_map peak Hz':>14} | {'reward_us Hz':>12}")
    eccs, rus_rates = [], []
    for agent, goal in N5_CASES:
        e = _ecc(agent, goal)
        pk, mn, cc, rus_rate = _present(bridge, agent, goal, op)
        eccs.append(e)
        rus_rates.append(rus_rate)
        print(f"{str(agent):>8} {str(goal):>8} | {e:5.2f} | {pk:14.1f} | {rus_rate:12.2f}")
    eccs = np.asarray(eccs)
    rus_rates = np.asarray(rus_rates)
    rus_max = float(rus_rates.max())
    if rus_rates.std() > 1e-9 and eccs.std() > 1e-9:
        corr = float(np.corrcoef(eccs, rus_rates)[0, 1])
    else:
        corr = float("nan")

    # --- verdict ---
    print(f"\n  (a) bump:  peak {peak_med:.1f} Hz (median), mean {mean_med:.1f} Hz, peak/mean ratio {bump_ratio:.2f}")
    print(f"  (b) N1 orienting cardinal match: {n1_ok}/{n1_tot}")
    print(f"  (c) reward_us max {rus_max:.2f} Hz, corr(ecc, reward_us) = {corr:.3f}")
    bump_alive = (peak_med >= 20.0) and (bump_ratio >= 3.0)
    n1_pass = n1_ok >= n1_tot - 1
    n5_pass = (rus_max >= 5.0) and (not np.isnan(corr)) and (corr < -0.6)
    print(f"\n  STARVATION CHECK: bump_alive={bump_alive} (peak>=20Hz AND peak/mean>=3x)  "
          f"N1_match={n1_pass} (>=7/8)  N5_reward={n5_pass} (reward_us>=5Hz AND corr<-0.6)")
    return dict(label=label, op=op, peak_med=peak_med, mean_med=mean_med, bump_ratio=bump_ratio,
                n1_ok=n1_ok, n1_tot=n1_tot, rus_max=rus_max, corr=corr,
                bump_alive=bool(bump_alive), n1_pass=bool(n1_pass), n5_pass=bool(n5_pass))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--also-standalone-op", action="store_true",
                    help="also run the STANDALONE op-point (80/6/2500/14) for the A/B contrast (slower)")
    args = ap.parse_args()

    print("SPIKING-SC MERGED-BRIDGE STEP-0 OP-POINT CHECK (co-residence starvation falsifier)")
    print("GATE: at the merged op-point (160/12/3500/40), the sc_map bump must FIRE (peak>=20Hz, peak/mean>=3x),")
    print("      orienting must match host >=7/8, and reward_us must cross threshold with corr(ecc,reward_us)<-0.6.")
    print("      If STARVED at the merged op-point, identify the knob that restores it.")

    res_merged = run_check(args.seed, MERGED_OP, "MERGED op-point (promoted)")
    res_standalone = None
    if args.also_standalone_op:
        res_standalone = run_check(args.seed, STANDALONE_OP, "STANDALONE op-point (contrast)")

    print("\n================ STEP-0 VERDICT ================")
    alive = res_merged["bump_alive"]
    print(f"MERGED op-point: bump_alive={res_merged['bump_alive']}  N1={res_merged['n1_ok']}/{res_merged['n1_tot']}  "
          f"reward_us_max={res_merged['rus_max']:.2f}Hz  corr={res_merged['corr']:.3f}")
    if res_standalone is not None:
        print(f"STANDALONE op-point: bump_alive={res_standalone['bump_alive']}  "
              f"N1={res_standalone['n1_ok']}/{res_standalone['n1_tot']}  "
              f"reward_us_max={res_standalone['rus_max']:.2f}Hz  corr={res_standalone['corr']:.3f}")
    if alive and res_merged["n1_pass"]:
        print("VERDICT: BUMP ALIVE at the merged op-point — the SC is NOT starved co-resident. "
              "The merged op-point should be promoted to the merged builder default. GO to the Step-1 6-seed A/B.")
    elif alive and not res_merged["n1_pass"]:
        print("VERDICT: bump fires but orienting < 7/8 — the bump is alive but the cortex pooling needs a "
              "re-tune (SC_CORTEX_W sweep on the merged bridge). NOT starved; an integration-strength tune.")
    else:
        print("VERDICT: STARVED at the merged op-point — the sc_map bump does not fire (peak<20Hz or peak/mean<3x). "
              "Raise SC_RET_SC / SC_RET_DRIVE further OR add per-region enable_homeostasis=True to the SC slice "
              "(the limbic-lift fix). This would be the spurious-NEGATIVE cause; fix BEFORE the 6-seed A/B.")


if __name__ == "__main__":
    main()
