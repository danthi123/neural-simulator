"""TRUE-ONE-BRAIN roadmap #2 — CHEAP-FIRST SCOPE + DE-RISK: the NEURAL reward `r` on the MERGED bridge.

Goal of #2 (research/findings/2026-06-18-full-spikeification-shared-substrate-roadmap.md §3 #2): make the merged
"one brain" nav episode source its reward `r` SYNAPTICALLY (from `sc_rostral->reward_us` firing — the N5 SC
proximity/goal-salience reward), retiring the host Manhattan/sign(delta-ecc) formula (g11_bg_runner.py:6901-6946
computes `reward`, then :7148 drives `reward_us` with `reward_us_drive_pa * max(0, reward)` = the HOST scalar).
Together with the already-committed value-train (learned V, commit 6fe74bc5), δ=r-V becomes FULLY synaptic.

THIS FILE IS THE CHEAP-FIRST DE-RISK (numpy/CPU). It does NOT do the full 6-seed route+build. It answers:
  (1) COMPOSITION — does the spiking SC chain (sc_retina/sc_map/sc_fs/sc_rostral) + reward_us + snc COMPOSE on the
      merged bridge (co-resident with parser/dlPFC/composer/limbic-critic), no region/index collision? Moat intact?
  (2) DE-RISK GATE — driving the merged bridge's sc_retina with the rendered gridworld at varying agent->goal
      proximities, does `sc_rostral->reward_us` source a GRADED proximity reward (reward_us + snc firing tracks
      proximity)? The decisive anti-cheat LESION: zero sc_rostral->reward_us -> the reward/SNc-burst must vanish
      (proves the reward IS the synaptic SC proximity, not a re-hidden host scalar).

The measurement reuses the VALIDATED sc_n5_rpe_probe.py mechanism (corr(distance,SNc)=-0.99, lesion collapses) but
builds it CO-RESIDENT on the merged nav+conv bridge via the additive `nav_critic_spiking_sc` kwarg (default-off,
byte-preserved). The SC chain is self-contained on its own sc_retina (does NOT need enable_visual_cortex). The
sc_map->sc_rostral proximity readout + retina->sc_map retinotopy are wired POST-INIT by install_spiking_sc_wiring.

    SIM_BACKEND=numpy python research/runners/_merged_neural_reward_scope_derisk.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sim.backend import get_backend, to_host
from sim.visual_cortex import image_to_retina_drive
from research.runners.g11_bg_runner import (
    install_spiking_sc_wiring, render_egocentric_goal,
)
from research.runners.nav_conv_merged_bridge import (
    build_merged_nav_conv_bridge, MergedNavConvAgent,
)

xp, BACKEND = get_backend()
IMG = 32          # visual_image_size (matches build_bg_brain_regions default)
SNC_TONIC = 220.0  # the SNc tonic drive (sc_n5_rpe_probe.py operating point)


def _gi(bridge, name):
    return np.asarray(list(bridge.region_manager.indices(name)), dtype=np.int64)


# MERGED-BRIDGE OPERATING POINT (het-off). The standalone sc_n5_rpe_probe.py weights are too weak on the
# heterogeneity-OFF merged bridge (the documented "standalone-tuned organ fires ~6-10x weaker co-resident"
# boundary, finding 2026-06-18-merged-limbic-core-lift.md): at the default w_ret_sc=80/w_sc_rec=6/ros_us=14 +
# 2500pA, sc_map fires ~2Hz and reward_us never crosses threshold. These stronger weights restore the chain
# het-off (sc_rostral fires graded, reward_us bursts when close, SNc bursts 2.7x). The 6-seed build tunes this
# operating point (or adds per-region homeostasis on sc_map/sc_rostral — the established merge-lift fix).
RETINA_DRIVE = 3500.0
W_RET_SC = 160.0
W_SC_REC = 12.0
W_ROS_US = 40.0


def _snc_reward_rate(bridge, snc_idx, us_idx, hold, image=None, ret_idx=None):
    """Drive the SNc tonic + (optionally) the SC's sc_retina with `image` (the neural US via the SC chain), run
    `hold` steps, return (snc_rate_Hz, reward_us_rate_Hz). r is sourced PURELY from the SC -> reward_us -> snc
    synaptic chain (no host current onto reward_us/snc beyond the SNc tonic pacemaker)."""
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[xp.asarray(snc_idx)] = xp.float32(SNC_TONIC)
    if image is not None and ret_idx is not None:
        d = image_to_retina_drive(image, drive_max_pA=RETINA_DRIVE)
        bridge.cp_external_input_current[xp.asarray(ret_idx)] = xp.asarray(d, dtype=xp.float32)
    snc_tot = 0
    us_tot = 0
    for _ in range(hold):
        bridge._run_one_simulation_step()
        snc_tot += int(to_host(bridge.cp_firing_states[xp.asarray(snc_idx)].sum()))
        us_tot += int(to_host(bridge.cp_firing_states[xp.asarray(us_idx)].sum()))
    snc_hz = snc_tot / len(snc_idx) / (hold * 1e-3)
    us_hz = us_tot / len(us_idx) / (hold * 1e-3)
    return snc_hz, us_hz


def run(seed=42, hold=60, grid_size=8, lesion=False, tag="INTACT", quiet=False):
    """Build the merged bridge with the spiking SC reward chain co-resident, wire the SC, run the proximity RPE
    battery. Returns the per-eccentricity (snc, reward_us) rates + the proximity correlation."""
    bridge, h = build_merged_nav_conv_bridge(
        seed=seed, co_resident_nav_critic=True, nav_critic_spiking_sc=True)
    # wire the SC post-init (retina->sc_map retinotopy + sc_map recurrent + sc_map->cortex_NESW + sc_map->sc_rostral)
    # at the merged-bridge-tuned operating point (het-off needs stronger drive than the standalone probe).
    n_sc = install_spiking_sc_wiring(bridge, visual_image_size=IMG, w_ret_sc=W_RET_SC, w_sc_rec=W_SC_REC,
                                     verbose=False)
    # set the sc_rostral->reward_us weight to the het-off operating point (the build's declared 14.0 is too weak
    # co-resident). lesion=True ZEROES it instead (the decisive reward anti-cheat: no SC drive -> no reward burst).
    rm = bridge.region_manager
    ros = _gi(bridge, "sc_rostral")
    us = _gi(bridge, "reward_us")
    pre = np.repeat(ros, us.shape[0]).astype(np.int64)
    post = np.tile(us, ros.shape[0]).astype(np.int64)
    w_ros_us = 0.0 if lesion else W_ROS_US
    bridge.set_pathway_weights("sc_rostral_to_reward_us", pre, post,
                               np.full(pre.size, w_ros_us, np.float32), add_missing=True)

    snc_idx = _gi(bridge, "snc")
    us_idx = _gi(bridge, "reward_us")
    ret_idx = _gi(bridge, "sc_retina")
    goal = (grid_size - 1, grid_size // 2)   # e.g. (7,4) on an 8-grid

    # settle
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
    base_snc, base_us = _snc_reward_rate(bridge, snc_idx, us_idx, hold)   # tonic baseline, no image

    if not quiet:
        print(f"\n== {tag} (seed {seed}, {n_sc} SC synapses) ==  tonic SNc={base_snc:.1f}Hz  reward_us={base_us:.1f}Hz")
        print(f"{'agent (ecc)':>14} | {'SNc Hz':>7} | {'reward_us Hz':>12} | burst?")
    eccs, snc_rs, us_rs = [], [], []
    # agent x sweeps toward the goal: ecc = |goal_x - ax| decreasing -> proximity increasing
    for ax in (goal[0] - 7, goal[0] - 6, goal[0] - 5, goal[0] - 4, goal[0] - 2):
        ax = int(np.clip(ax, 0, grid_size - 1))
        ecc = abs(goal[0] - ax)
        img = render_egocentric_goal((ax, goal[1]), goal, image_size=IMG)
        snc_hz, us_hz = _snc_reward_rate(bridge, snc_idx, us_idx, hold, image=img, ret_idx=ret_idx)
        eccs.append(ecc); snc_rs.append(snc_hz); us_rs.append(us_hz)
        if not quiet:
            burst = snc_hz > base_snc * 1.15
            print(f"{str((ax, goal[1]))+' ('+str(ecc)+')':>14} | {snc_hz:6.1f} | {us_hz:11.1f} | {'YES' if burst else 'no'}")
    # graded: closer (smaller ecc) -> bigger reward_us + SNc. corr(ecc, rate) should be NEGATIVE.
    corr_snc = float(np.corrcoef(eccs, snc_rs)[0, 1]) if np.std(snc_rs) > 1e-9 else 0.0
    corr_us = float(np.corrcoef(eccs, us_rs)[0, 1]) if np.std(us_rs) > 1e-9 else 0.0
    close_us = us_rs[-1]      # ecc 2 (closest in the sweep)
    far_us = us_rs[0]         # ecc 7 (farthest)
    close_snc = snc_rs[-1]
    if not quiet:
        print(f"  corr(ecc, reward_us)={corr_us:.2f}  corr(ecc, SNc)={corr_snc:.2f}  "
              f"reward_us close/far = {close_us:.1f}/{far_us:.1f}Hz")
    return dict(base_snc=base_snc, base_us=base_us, eccs=eccs, snc_rs=snc_rs, us_rs=us_rs,
                corr_snc=corr_snc, corr_us=corr_us, close_us=close_us, far_us=far_us,
                close_snc=close_snc, n_sc=n_sc)


def composition_and_moat(seed=42):
    """(1) the spiking SC chain composes on the merged bridge (all regions present, neuron-count union ok);
    (2) the no-confab moat is intact on the bridge WITH the SC chain (a known fact resolves; an unstored cue
    abstains -> None). Returns (compose_ok, moat_ok, info)."""
    print(f"[compose+moat] building MergedNavConvAgent(co_resident_nav_critic=True, nav_critic_spiking_sc=True) "
          f"seed={seed} ...")
    agent = MergedNavConvAgent(seed=seed, co_resident_nav_critic=True, nav_critic_spiking_sc=True)
    bridge = agent._merged_bridge
    rm = bridge.region_manager
    names = set(rm.region_indices_dict())
    sc_regions = ("sc_retina", "sc_map", "sc_fs", "sc_rostral", "reward_us", "snc")
    conv_regions = ("parse_conj", "parse_role", "cortex_ctx", "dlpfc_wm")
    nav_regions = ("cortex_N", "striosome_value")
    compose_ok = all(r in names for r in sc_regions + conv_regions + nav_regions)
    missing = [r for r in sc_regions + conv_regions + nav_regions if r not in names]
    print(f"[compose+moat] regions present: SC={[r for r in sc_regions if r in names]}")
    print(f"[compose+moat] conv={[r for r in conv_regions if r in names]} nav={[r for r in nav_regions if r in names]}")
    if missing:
        print(f"[compose+moat] MISSING regions: {missing}")

    # the no-confab MOAT: a known fact resolves; an unstored cue abstains. Teach the agent one SVO fact first.
    agent.hear("dog go north")
    resolves = agent.what_does("dog", "go")        # expect 'north'
    abstains = agent.what_does("river", "look")    # expect None (no fact -> no confabulation)
    moat_ok = (resolves == "north") and (abstains is None)
    print(f"[compose+moat] moat: what_does('dog','go')={resolves!r} (want 'north'); "
          f"what_does('river','look')={abstains!r} (want None) -> moat_ok={moat_ok}")
    n_neurons = int(bridge.core_config.num_neurons)
    print(f"[compose+moat] merged bridge: {len(bridge.core_config.brain_regions)} regions, {n_neurons} neurons, "
          f"{int(bridge.cp_connections.nnz)} synapses")
    return compose_ok, moat_ok, dict(n_neurons=n_neurons, missing=missing, resolves=resolves, abstains=abstains)


def main():
    print("=" * 88)
    print("TRUE-ONE-BRAIN #2 CHEAP-FIRST SCOPE + DE-RISK — the NEURAL reward on the MERGED bridge")
    print(f"backend={BACKEND}")
    print("=" * 88)

    # (1) COMPOSITION + MOAT
    compose_ok, moat_ok, info = composition_and_moat(seed=42)

    # (2) DE-RISK GATE: graded proximity reward (INTACT) + lesion collapse
    intact = run(seed=42, tag="INTACT (sc_rostral->reward_us LIVE)")
    lesion = run(seed=42, lesion=True, tag="LESION (sc_rostral->reward_us ZEROED)")

    # gates
    graded = (intact["corr_us"] <= -0.5) and (intact["close_us"] > intact["far_us"] + 1.0)
    burst_close = intact["close_snc"] > intact["base_snc"] * 1.15
    # lesion must collapse the reward_us drive: the close-proximity reward_us drops to ~its lesioned baseline
    # (no SC drive => reward_us only fires from residual). Decisive: lesioned close reward_us << intact close.
    lesion_collapses = lesion["close_us"] < max(2.0, intact["close_us"] * 0.5)

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"(1) COMPOSITION : SC chain co-resides on the merged bridge = {compose_ok}  "
          f"({info['n_neurons']} neurons; missing={info['missing']})")
    print(f"    MOAT intact : known fact resolves + unstored cue abstains = {moat_ok}")
    print(f"(2) GRADED PROXIMITY reward (reward_us): corr(ecc,reward_us)={intact['corr_us']:.2f} (<=-0.5) "
          f"AND close>far ({intact['close_us']:.1f}>{intact['far_us']:.1f}) = {graded}")
    print(f"    SNc bursts on close goal: {burst_close} (close SNc {intact['close_snc']:.1f} vs tonic "
          f"{intact['base_snc']:.1f})")
    print(f"    LESION collapses reward: intact close reward_us {intact['close_us']:.1f}Hz -> lesioned "
          f"{lesion['close_us']:.1f}Hz = {lesion_collapses}")

    overall = compose_ok and moat_ok and graded and lesion_collapses
    if overall:
        print("\nVERDICT: GO (cheap-first) — the spiking SC reward chain COMPOSES co-resident on the merged "
              "one brain (moat intact), and `sc_rostral->reward_us` sources a GRADED proximity reward that "
              "the SNc bursts on; the lesion collapses it (the reward IS the synaptic SC proximity, not a "
              "re-hidden host scalar). Proceed to the route+6-seed build: thread the SC-reward flags through "
              "run_moving_goal_episode's merged nav gate + replace g11_bg_runner.py:7148 with the sc_rostral "
              "branch (zero the host reward_us drive; let the SC fire reward_us).")
    else:
        print(f"\nVERDICT: NOT-YET / BOUNDARY — compose={compose_ok} moat={moat_ok} graded={graded} "
              f"lesion={lesion_collapses}. Report the boundary (an honest scope finding is the deliverable).")
    return overall


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
