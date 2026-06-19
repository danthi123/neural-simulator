"""G11: Basal ganglia action selection module.

Phase B follow-up to the silent-motor trap arc (Sessions G/H/I, all NEGATIVE).
The trap was diagnosed (V6) as a *reservoir-state bias problem* — random
hidden->motor weights on a shared reservoir naturally favor whichever motor
the input pattern happens to align with. Argmax + reservoir bias = lock-in.

Phase B fix (architectural): replace the shared-reservoir + argmax-readout
with a per-action basal-ganglia cascade. Each motor has its own dedicated
D1 MSN pool, D2 MSN pool, GPi, thalamus, and motor populations. Lateral
inhibition between motor populations provides structural winner-take-all
(no shared spike count to bias).

Architecture:
    cortex ─-> str_D1[N,E,S,W]    str_D2[N,E,S,W]
                  │                     │
            direct pathway       indirect pathway
                  v                     v
              GPi[N,E,S,W] <-── STN <-── GPe[N,E,S,W]
                  │
                  v (disinhibition)
              thal[N,E,S,W]
                  │
                  v
              motor[N,E,S,W]   (lateral inhibition between)

DA modulation: midbrain DA neurons (A9 SNc / A10 VTA, collapsed in this
model) project to all striatal pools. DA enhances the direct pathway
(D1-class receptor, Gs-coupled, LTP-biased) and suppresses the indirect
pathway (D2-class receptor, Gi-coupled, LTD-biased). Per Kandel ch 43.

Built on validated Phase A presets:
- IZH2007_STRIATAL_MSN_D1 / D2 (rest=-80 mV down-state, fires when driven)
- IZH2007_GPE_PACEMAKER, IZH2007_GPI_OUTPUT (high tonic rates)
- IZH2007_STN_BURST (autonomous + scales with input)
- IZH2007_THALAMIC_RELAY (tonic mode)
- IZH2007_RS_CORTICAL_PYRAMIDAL, IZH2007_FS_CORTICAL_INTERNEURON (cortex)
- IZH2007_DOPAMINE (slow tonic + phasic)

Reference: Frank 2005 J Neurosci; Schroll & Hamker 2013 Front Comp Neurosci.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

# ───────────────────────────────────────────────────────────────────────
# Phase 0 (N9 nav-deployment, 2026-06-09): pin cuBLAS determinism
# UNCONDITIONALLY at the very top, BEFORE any cupy/numpy/sim import, so the
# place-code self-organization under enable_neural_critic is reproducible
# (the place-code self-org / value-LTP loop is sensitive to cuBLAS GEMM
# non-determinism; the pin MUST precede `import cupy`). This is a no-op for
# the flagship (no behaviour change — it only fixes the GEMM workspace), and
# `setdefault` leaves any externally-set value intact.
# ───────────────────────────────────────────────────────────────────────
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ───────────────────────────────────────────────────────────────────────
# CUDA determinism (must be set BEFORE cupy/cuBLAS init).
# Triggered by --deterministic flag in argv. Tightens seed-to-seed noise
# floor (per the 2026-04-29 finding that A+E single-goal det gave
# 3.31 +/- 0.74 vs documented 4.08 +/- 0.49 — same code, +/-3-5 noise
# without determinism). ~10-30% slowdown.
# ───────────────────────────────────────────────────────────────────────
if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
N_ACTIONS = 4

# N9 neural place-code self-org (2026-06-09): number of fixed landmarks the egocentric
# place-sensor render uses (matches the de-risk default_landmarks: 3 corner/edge beacons).
# place_sensors size = N_PLACE_LANDMARKS * (n_place_bearing + n_place_dist).
N_PLACE_LANDMARKS = 3


def _n9_place_landmarks(grid_size):
    """The 3 fixed landmark positions for the N9 egocentric place render (de-risk
    default_landmarks: two bottom corners + a top-middle beacon). Diverse bearings/
    distances so the self-org place code can carve distinct fields per (x,y)."""
    g = float(grid_size) - 1.0
    return [(0.0, 0.0), (g, 0.0), (g / 2.0, g)]


def _n9_place_sensor_act(x, y, landmarks, n_bearing, n_dist, max_int, falloff,
                         dist_sigma, dist_max, bexp):
    """Egocentric landmark sensor render (the N9 legitimate body-sensing channel; VERBATIM
    from n9_place_graded_critic_stage2_derisk.landmark_sensor_act). (x,y) enters the brain
    ONLY here (the position-leak boundary). Per landmark: n_bearing cosine-tuned bearing
    sensors (intensity * cos_align**bexp, distance-attenuated) + n_dist distance-tuned
    Gaussians. Returns the concatenated (len(landmarks)*(n_bearing+n_dist),) vector."""
    blocks = []
    bpx = np.cos(2.0 * np.pi * np.arange(n_bearing) / n_bearing)
    bpy = np.sin(2.0 * np.pi * np.arange(n_bearing) / n_bearing)
    dist_centers = np.linspace(0.0, dist_max, n_dist)
    for (lx, ly) in landmarks:
        dx = float(lx - x); dy = float(ly - y)
        d = (dx * dx + dy * dy) ** 0.5
        if d < 1e-6:
            bear = np.full(n_bearing, max_int, dtype=np.float32)
            dist = np.full(n_dist, max_int, dtype=np.float32)
        else:
            bx = dx / d; by = dy / d
            intensity = max_int / (1.0 + falloff * d)
            cos_align = np.maximum(0.0, bpx * bx + bpy * by)
            bear = (intensity * (cos_align ** bexp)).astype(np.float32)
            dist = (max_int * np.exp(-(d - dist_centers) ** 2 / (2.0 * dist_sigma ** 2))).astype(np.float32)
        blocks.append(bear.astype(np.float32))
        blocks.append(dist.astype(np.float32))
    return np.concatenate(blocks).astype(np.float32)


def sc_orienting_cardinal_from_image(image):
    """Innate superior-colliculus orienting reflex: read the goal's RETINAL
    direction from the rendered retinotopic image ALONE — no (gx,gy)/(x,y)
    coordinate access. Biology: the SC computes a retinotopic salience map and
    orients toward the salient target (Kandel 6e Ch 35; catalog A.07/H.25); the
    de-risk for the navigation perceptual cold-start
    (research/findings/2026-06-07-perceptual-bootstrap-deep-research.md).

    sim/visual_cortex.render_gridworld_to_image paints the agent as the bright
    ON blob (~1.0/0.7) and the goal as a dimmer ON blob (~0.5). This reads both
    blob CENTROIDS from the pixels and returns the cardinal of the goal's offset
    from the agent (= the goal's eccentricity on a retina centred on the agent).
    Returns one of "N"/"E"/"S"/"W", or None if either blob is absent (e.g. the
    agent is already on the goal).

    ANTI-CHEAT: the only argument is the rendered image array; coordinates never
    enter this function. Image convention (matches render_gridworld_to_image):
    channel 0 = ON; larger pixel-x = +grid-x = East; larger pixel-y = +grid-y =
    North (so the cardinal mapping matches the heuristic's gx/gy comparison).
    """
    on = np.asarray(image)[0]
    agent_ys, agent_xs = np.where(on >= 0.65)               # bright agent blob
    goal_ys, goal_xs = np.where((on >= 0.35) & (on < 0.65))  # dimmer goal blob
    if agent_ys.size == 0 or goal_ys.size == 0:
        return None
    dx = float(goal_xs.mean() - agent_xs.mean())   # +x = East
    dy = float(goal_ys.mean() - agent_ys.mean())   # +y = North
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return None  # essentially co-located → no orienting target
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def sc_salience_offset_from_image(image, grid_size=8, image_size=32):
    """Continuous goal offset (dx, dy) in GRID-CELL units, read from the rendered
    retinotopic image ALONE (no coordinates) — the position-PRESERVING "where"
    signal for the Rank-2 learned dorsal/PPC read-out (Rank 2 of
    research/findings/2026-06-07-perceptual-bootstrap-deep-research.md).

    Sibling of sc_orienting_cardinal_from_image: same agent/goal blob-centroid
    read, but returns the GRADED offset (= the goal's retinal eccentricity in
    cells) rather than a cardinal. (dx, dy) matches the coordinate (gx-x, gy-y)
    convention (+x=East/+gx, +y=North/+gy) so it plugs straight into the existing
    sensory Gaussian-bump code. Returns None if either blob is absent.

    ANTI-CHEAT: the only data input is the image array; coordinates never enter.
    """
    on = np.asarray(image)[0]
    agent_ys, agent_xs = np.where(on >= 0.65)
    goal_ys, goal_xs = np.where((on >= 0.35) & (on < 0.65))
    if agent_ys.size == 0 or goal_ys.size == 0:
        return None
    ppc = max(1.0, float(image_size) / float(grid_size))  # pixels per grid cell
    dx = float(goal_xs.mean() - agent_xs.mean()) / ppc     # +x = East  = +gx
    dy = float(goal_ys.mean() - agent_ys.mean()) / ppc     # +y = North = +gy
    return (dx, dy)


def render_egocentric_goal(agent, goal, image_size=32, ppc=4, radius=2):
    """ENVIRONMENT render (legitimate, channel-1 of the BRAIN-BASED-ONLY bar): the world
    from the agent's EYE — the goal as a dim ON blob at its bearing (goal - agent) relative
    to the foveal centre. Egocentric (the agent does not see its own eye), so a single blob
    the spiking superior colliculus localises directly. Same (2,H,W) ON/OFF convention as
    render_gridworld_to_image. De-risked in sc_map_orienting_probe.py."""
    img = np.zeros((2, image_size, image_size), dtype=np.float32)
    c = image_size // 2
    gx = int(round(c + (goal[0] - agent[0]) * ppc))   # +x = East
    gy = int(round(c + (goal[1] - agent[1]) * ppc))   # +y = North
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = gx + dx, gy + dy
            if 0 <= px < image_size and 0 <= py < image_size:
                img[0, py, px] = max(img[0, py, px], 0.5)
    return img


def install_spiking_sc_wiring(bridge, visual_image_size=32, w_ret_sc=80.0,
                              w_sc_rec=6.0, w_sc_cortex=1.0, scramble=False, verbose=False):
    """Post-init explicit wiring for the spiking superior colliculus (N1), de-risked in
    sc_map_orienting_probe.py. Installs (via set_pathway_weights, add_missing): retina(ON)->
    sc_map retinotopic pooling (2x2 RF), sc_map short-range recurrent excitation, and
    sc_map->cortex_{N,E,S,W} weighted-quadrant pooling (the orienting read-out: the winning
    cortex pool BY FIRING = the cardinal). The sc_map<->sc_fs Mexican-hat is framework-built
    (declared with real density so sc_fs is inhibitory). Must run AFTER _initialize_simulation_data.
    Returns the count of synapses installed."""
    rm = bridge.region_manager
    IMGW = int(visual_image_size)
    SCN = IMGW // 2                       # sc sheet side (32 -> 16)
    ret0 = int(list(rm.indices("sc_retina"))[0])   # the SC's own egocentric eye
    sc0 = int(list(rm.indices("sc_map"))[0])
    ctx0 = {a: int(list(rm.indices(f"cortex_{a}"))[0]) for a in ACTION_NAMES}
    n_ctx = {a: len(list(rm.indices(f"cortex_{a}"))) for a in ACTION_NAMES}
    sc_center = (SCN - 1) / 2.0
    sc_idx = lambda sy, sx: sy * SCN + sx
    ret_on = lambda py, px: py * IMGW + px           # ON channel = first IMGW*IMGW indices

    # Anti-cheat lesion: scramble=True permutes the sc-site target assignment (destroys
    # retinotopy). A scrambled-retinotopy nav must REGRESS (the de-risk lesion) -> proves the
    # orienting is genuinely carried by the retinotopic map, not a non-retinotopic leak.
    sc_targets = list(range(SCN * SCN))
    if scramble:
        sc_targets = [int(v) for v in np.random.default_rng(12345).permutation(SCN * SCN)]

    n_installed = 0
    # 1) retina(ON) -> sc_map retinotopic (each sc site pools its 2x2 ON block)
    pre, post, w = [], [], []
    for sy in range(SCN):
        for sx in range(SCN):
            tgt = sc_targets[sc_idx(sy, sx)]
            for a in (0, 1):
                for b in (0, 1):
                    pre.append(ret0 + ret_on(2 * sy + a, 2 * sx + b))
                    post.append(sc0 + tgt)
                    w.append(w_ret_sc)
    n_installed += bridge.set_pathway_weights("retina_to_sc_map",
        np.asarray(pre, np.int64), np.asarray(post, np.int64),
        np.asarray(w, np.float32), add_missing=True)
    # 2) sc_map short-range recurrent excitation (radius 1)
    pre, post, w = [], [], []
    for sy in range(SCN):
        for sx in range(SCN):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < SCN and 0 <= nx < SCN and (dy, dx) != (0, 0):
                        pre.append(sc0 + sc_idx(sy, sx))
                        post.append(sc0 + sc_idx(ny, nx))
                        w.append(w_sc_rec)
    n_installed += bridge.set_pathway_weights("sc_map_recurrent",
        np.asarray(pre, np.int64), np.asarray(post, np.int64),
        np.asarray(w, np.float32), add_missing=True)
    # 3) sc_map -> cortex_{N,E,S,W} weighted-quadrant pooling (+sx=East, +sy=North)
    for a in ACTION_NAMES:
        pre, post, w = [], [], []
        for sy in range(SCN):
            for sx in range(SCN):
                ddx, ddy = sx - sc_center, sy - sc_center
                wv = {"E": max(0.0, ddx), "W": max(0.0, -ddx),
                      "N": max(0.0, ddy), "S": max(0.0, -ddy)}[a]
                if wv <= 0.0:
                    continue
                for d in range(n_ctx[a]):
                    pre.append(sc0 + sc_idx(sy, sx))
                    post.append(ctx0[a] + d)
                    w.append(w_sc_cortex * wv)
        if pre:
            n_installed += bridge.set_pathway_weights(f"sc_map_to_cortex_{a}",
                np.asarray(pre, np.int64), np.asarray(post, np.int64),
                np.asarray(w, np.float32), add_missing=True)
    # N5 Option C: sc_map -> sc_rostral foveal-CENTRE pooling (broad Gaussian) so sc_rostral
    # fires graded with how central the bump is (= how small the goal's eccentricity is). The
    # temporal-difference (sc_rostral - lagged) downstream = "the goal got closer". Only when
    # the N5 approach circuit is present.
    try:
        ros_list = list(rm.indices("sc_rostral"))
    except Exception:
        ros_list = []
    if ros_list:
        ros0 = int(ros_list[0]); n_ros = len(ros_list)
        sig = 5.0
        pre, post, w = [], [], []
        for sy in range(SCN):
            for sx in range(SCN):
                r2 = (sx - sc_center) ** 2 + (sy - sc_center) ** 2
                wv = float(np.exp(-r2 / (2 * sig * sig)))
                if wv <= 0.02:
                    continue
                for d in range(n_ros):
                    pre.append(sc0 + sc_idx(sy, sx)); post.append(ros0 + d)
                    w.append(20.0 * wv)         # strong: the rostral readout must fire robustly
        if pre:
            n_installed += bridge.set_pathway_weights("sc_map_to_sc_rostral",
                np.asarray(pre, np.int64), np.asarray(post, np.int64),
                np.asarray(w, np.float32), add_missing=True)
    if verbose:
        print(f"[g11] spiking SC: installed {n_installed} synapses "
              f"(retina->sc_map {SCN}x{SCN} + recurrent + sc_map->cortex_NESW"
              f"{' + sc_map->sc_rostral (N5)' if ros_list else ''})", flush=True)
    return n_installed


def build_bg_brain_regions(
    n_cortex: int = 100,
    n_striatum_per_action: int = 50,
    n_gpe_per_action: int = 10,
    n_gpe_arky_per_action: int = 4,  # R3.7: arkypallidal (PV-) subpool
    n_str_striosome_per_action: int = 8,  # R3.11: striosome (patch) subpool
    # ── Spiking-SNc actor-critic Stage B (2026-06-08): neural value critic ──
    # When True, add a dedicated `striosome_value` GABAergic critic population
    # driven by the PERCEIVED state (cortex_it, the ventral-stream object code —
    # never coordinates), with a PLASTIC cortex_it->striosome_value pathway
    # (gate "value_input") trained by the SAME SNc-derived dopamine delta the
    # actor uses, so the striosome firing comes to encode the expected value V.
    # Its inhibition reaches the SNc through the slow GABA_B/GIRK K+ conductance
    # (E_K=-90mV, receptor="gaba_b") so the value subtraction r-V happens at the
    # SNc MEMBRANE (sign-correct + strong on the KCC2-lacking depolarized SNc),
    # NOT a host arithmetic term. Only meaningful with spiking_snc=True; when set,
    # the host _V_scaffold subtraction is DROPPED (the subtraction is now neural).
    # Validated CPU de-risk: research/findings/2026-06-08-gabab-girk-stageB-derisk-GO.md.
    enable_neural_critic: bool = False,
    n_striosome_value: int = 80,            # critic pool size (GABAergic MSN-D1)
    critic_cortex_it_to_value_weight: float = 3.0,   # plastic afferent init weight (STDP grows V)
    critic_cortex_it_to_value_density: float = 0.6,
    critic_value_to_snc_weight: float = 10.0,        # GABA_B inhibitory critic->SNc weight (de-risk PASS value)
    critic_value_to_snc_density: float = 0.5,
    spiking_reward_us: bool = False,        # reward_us (PPN-like) excitatory US->SNc afferent: spiking reward burst, drops the host write
    n_reward_us: int = 40,                  # US/reward afferent pool size (C.33: 30-50)
    reward_us_to_snc_weight: float = 50.0,  # reward_us->snc excitatory weight (Pavlovian de-risk: moderate SNc burst the critic GABA_B can subtract)
    reward_us_to_snc_density: float = 0.6,
    # ── 2026-06-09 VALIDATED redesign (navfaithful-afferent-critic-homeostasis PASS) ──
    # The critic afferent is a DEDICATED DENSE `vs_place_context` region (a grid-32-tuned
    # Gaussian place code so 30-80 cells fire/location), NOT the SPARSE actor
    # `sensor_place_readout` (~1-3 cells) which provably can't fire the MSN critic. Per-region
    # homeostasis (intrinsic homeostatic plasticity; the committed 89b8d909 sim/ edit) is set
    # on BOTH `vs_place_context` AND `striosome_value` (GLOBAL cfg.enable_homeostasis stays
    # OFF — deterministic regime preserved). This is the exact wiring the de-risk PASSED 3/3
    # (snc_stageb_critic_probe_navfaithful.py --afferent-homeostasis): critic ~1.3-1.5 Hz,
    # place code sharply graded (~59 Hz near vs 0 Hz far), GABA_B value subtraction opens.
    n_vs_place_context: int = 200,           # dense dedicated place-context afferent (feeds ONLY the critic)
    vs_place_to_value_weight: float = 0.2,   # vs_place_context->striosome_value plastic INIT weight (de-risk PASS: STDP grows 0.20->0.58)
    vs_place_to_value_density: float = 0.5,
    enable_critic_homeostasis: bool = False, # per-region homeostasis on vs_place_context + striosome_value
    # ── 2026-06-09 CONVERGENT-EXCITATION UP-STATE (the faithful, homeostasis-free critic-firing
    #    mechanism; research/findings/2026-06-09-N9-faithful-value-cell-design.md Option A) ──
    # Splits the critic afferent into TWO DISTINCT regions (the RegionManager keys pathways by
    # (from,to), sim/regions.py:537, so two pathways from ONE region COLLIDE — hence two regions):
    #   A1  `vs_place_drive`   -> striosome_value : DENSE, NON-plastic, many weak synapses summing
    #       PAST the MSN-D1 ~339 pA rheobase at the goal => the B.02 convergent-excitation up-state.
    #       Fires the cell from init (breaks the LTP bootstrap structurally — no homeostasis needed).
    #   A2  `vs_place_context` -> striosome_value : the existing SPARSE PLASTIC value-learning arm.
    # BOTH rendered with the SAME grid-32 Gaussian place code each nav step (the runner injects the
    # drive into BOTH when enable_convergent_upstate=True). Default OFF => byte-identical to the
    # single-afferent path; the A1 region/pathway are simply not added.
    #
    # CuPy DE-RISK (2026-06-09, n9_convergent_upstate_derisk.py, 3 seeds):
    #   FIRE 3/3, LEARNS-V(LTP) 3/3, ACTOR-NOT-PERTURBED 3/3  — but PLACE-GRADED(near>=3x far) 0/3.
    # The dense NON-plastic A1 up-state is POSITION-BLIND: it fires the critic wherever a place bump
    # exists (its per-location rate is set by which afferent cells happen to wire onto the critic,
    # NOT by the goal), so the trained critic NEAR/FAR ratio caps ~1.2-1.4 (A2's learned LTP adds
    # NEAR selectivity but cannot suppress the FAR up-state floor). Option B (per-region NMDA on the
    # critic) deepens the up-state at BOTH locations and makes grading WORSE. The SNc value-
    # subtraction gap was only 2/3 at one cherry-picked operating point. HONEST NEGATIVE on a
    # value-of-LOCATION (design Option D): the up-state fires + the value LEARNS, but the spatial
    # selectivity that makes it a value CRITIC needs a richer/self-organized place code. Shipped
    # default-OFF as documented infrastructure; NOT wired into any flagship config.
    enable_convergent_upstate: bool = False,
    vs_place_drive_to_value_weight: float = 28.0,   # A1 dense NON-plastic up-state weight (de-risk: ~28 fires the corner goal >=5Hz)
    vs_place_drive_to_value_density: float = 0.8,   # A1 dense convergence (many weak synapses, not one giant)
    # ── 2026-06-09 N9 NEURAL PLACE-CODE SELF-ORG (nav deployment of the VALIDATED de-risk
    #    research/runners/n9_place_graded_critic_stage2_derisk.py; design
    #    docs/plans/2026-06-09-N9-nav-deployment-design.md). When neural_place_selforg=True
    #    (only meaningful WITH enable_neural_critic), the host-Gaussian `vs_place_context`
    #    place code (a BRAIN-BASED-ONLY shortcut) is REPLACED by a SELF-ORGANIZED spiking place
    #    code: a dedicated `place_sensors` region (the legitimate egocentric bearing/distance
    #    landmark render — the body-sensing channel; (x,y) enters the brain ONLY here) drives a
    #    `place` pool (IZH2007_HIPPO_PYRAMIDAL) through a PLASTIC competitive pathway (gate
    #    `landmark_to_place`); an FS-PING pool (`place_fs`) reciprocally wired to `place` re-times
    #    the sparse ensemble into a coincident gamma volley (the FS->place arm gated by
    #    `place_fs_gate`, held CLOSED during self-org for clean threshold-WTA -> sparse DISTINCT
    #    fields, OPENED for the volley read-out). `place->striosome_value` is a Route-D
    #    coincidence_detector so the volley fires the MSN critic that the sparse-async code can't.
    #    Default OFF => the enable_neural_critic path is byte-identical (the host vs_place_context).
    #    HARD-GATES enable_convergent_upstate OFF (the position-blind A1 floor caps grading ~1.2x).
    neural_place_selforg: bool = False,
    n_place: int = 200,                     # the self-org place pool (alias of n_vs_place_context size)
    n_place_fs: int = 24,                   # FS-PING interneuron pool (~10-20% of n_place)
    place_sensors_to_place_weight: float = 28.0,   # landmark_sensors->place plastic init (de-risk lm_to_place_weight)
    place_sensors_to_place_density: float = 0.5,
    place_sensors_to_place_jitter: float = 0.6,
    enable_critic_fs_inhibition: bool = False,  # place_fs->striosome_value GABA_A (spiking critic rate-clamp; root fix vs GIRK cap)
    critic_fs_weight: float = 16.0,         # place_fs->striosome_value inhibitory weight (de-risk sweet spot: critic 126->8 Hz physiological, gap 3.75x graded)
    critic_fs_density: float = 0.6,         # place_fs->striosome_value density (pooled perisomatic FS shunt)
    place_fs_weight: float = 16.0,          # place->place_fs (FS-PING excitation; de-risk value)
    place_fs_density: float = 0.4,
    fs_to_place_weight: float = 8.0,        # place_fs->place GABA_A (de-risk value)
    fs_to_place_density: float = 0.4,
    coincidence_threshold: int = 12,        # Route-D readout K (de-risk readout_weighted_k ~12)
    coincidence_train_k: float = 4.0,       # Route-D TRAIN count K (de-risk coincidence_k; MUST be >1)
    coincidence_plateau: float = 80.0,      # Route-D plateau strength (de-risk value)
    # N9 place-sensors egocentric render params (the legitimate sensory channel; de-risk canon).
    n_place_bearing: int = 12,              # bearing sensors per landmark (de-risk n_bearing)
    n_place_dist: int = 8,                  # distance sensors per landmark (de-risk n_dist)
    enable_cluster_a_closed_loop: bool = False,  # Cluster A: hyperdirect + thal->cortex
    n_gpi_per_action: int = 10,
    n_stn: int = 20,
    n_thal_per_action: int = 10,
    n_motor_per_action: int = 10,
    n_motor_fs_per_action: int = 5,
    n_dopamine: int = 10,
    enable_motor_lateral_inhibition: bool = False,
    # WTA defaults validated 2026-04-25 on probe_bg_wta_ambiguous: under equal
    # cortex_N/cortex_E drive, asymmetry 1.06x → 1.77x with these weights.
    # Lower values (10/5) leave FS pool subthreshold and inhibition is dead.
    motor_to_fs_weight: float = 50.0,
    fs_to_motor_weight: float = 20.0,
    # Thalamic reticular-nucleus (TRN)-style lateral inhibition between thal
    # relay pools (2026-06-06, N8+N6). Under genuine GPi->thal disinhibition the
    # released action's thal is the cleanest selection signal, but during a
    # plastic multi-goal run cumulative D1 plasticity leaks several thals at
    # once (ties). TRN provides reciprocal inhibition between thalamic relay
    # nuclei (Pinault 2004; Crabtree 2018) — a biological WTA on the relay that
    # suppresses the non-winner thals so the readout_source="thal" argmax sees a
    # single clean winner. Modeled like the motor WTA: thal_X -> thal_FS_X
    # (excitatory) and thal_FS_X -> thal_Y!=X (inhibitory). Default OFF.
    enable_thal_lateral_inhibition: bool = False,
    n_thal_fs_per_action: int = 5,
    thal_to_fs_weight: float = 50.0,
    thal_fs_to_thal_weight: float = 20.0,
    # Spiking action-selection WTA readout (2026-06-06, N6 biologization).
    # A DEDICATED, READ-ONLY selection layer that biologizes the host argmax:
    # four sel_X excitatory pools driven feed-forward by the cleanly-selective
    # thal_X (strong thal_to_sel_weight), competing among themselves via
    # sel_FS_X GABAergic interneurons (sel_X -> sel_FS_X exc; sel_FS_X ->
    # sel_Y!=X inh). The DECISION emerges from this spiking competition (the
    # winner is which sel_X fires); the host then merely OBSERVES which pool
    # won, instead of computing an argmax over raw rates. Critically this layer
    # has NO back-projection to thal — it does NOT perturb the thal->motor
    # cascade or the navigation dynamics (unlike enable_thal_lateral_inhibition,
    # which put the competition ON the relay and corrupted the forward signal,
    # scoring 20.0). Drive the competition from the strong clean thalamus, not
    # the weak motor counts (which scored 14.7). Enabled when
    # readout_source="spiking_wta". Cortical soft-WTA microcircuit (Douglas-
    # Martin 2004; Rutishauser-Douglas-Slotine 2011 lateral-inhibition WTA).
    enable_spiking_wta_readout: bool = False,
    n_sel_per_action: int = 20,
    n_sel_fs_per_action: int = 10,
    thal_to_sel_weight: float = 30.0,    # thal_X -> sel_X (feed-forward EVIDENCE; modest, not saturating)
    sel_to_sel_fs_weight: float = 20.0,  # sel_X -> sel_FS_X (drives the competing interneuron)
    sel_fs_to_sel_weight: float = 5.0,   # sel_FS_X -> sel_Y!=X (GENTLE cross-pool suppression; symmetric over-inhibition is unstable)
    # ACCUMULATE-THEN-COMMIT (2026-06-06, N6 fix). The gain-0 sel_X soft-WTA
    # above is a PASSIVE INSTANTANEOUS COMPARATOR (internal_density=0,
    # exc_weight_mean=0) — it cannot manufacture a winner from the weak released
    # thalamus (the deep-research finding 2026-06-06-action-selection-readout-
    # deep-research.md). The brain commits decisions in TWO STAGES:
    #   (1) ACCUMULATE: each sel_X gets NMDA-SLOW recurrent self-excitation
    #       (sel_recurrent_density>0, sel_recurrent_weight>0, soft-WTA gain
    #       alpha<1 — STABLE attractor per Rutishauser-Douglas-Slotine 2011, NOT
    #       alpha>1 unstable). The recurrence amplifies + integrates the weak
    #       thal drive over the readout window (network tau = tau_syn/|1-w_rec|;
    #       Wang 2002 Neuron slow-reverberation decision attractor). sel_X is
    #       NMDA-enabled via the per-region cp_nmda_neuron_mask (the same
    #       mechanism enable_pfc_nmda uses) so the integration time constant is
    #       biological (NMDA tau_decay=100ms), not AMPA-fast.
    #   (2) COMMIT: a downstream burst pool commit_X (superior-colliculus /
    #       saccade-generator analogue, H.24/H.25) is held silent by a tonically
    #       firing commit_OPN omnipause pool (constant external drive). Only when
    #       sel_X ramps past threshold does sel_X -> commit_X overcome the tonic
    #       inhibition and commit_X fires ALL-OR-NONE (Lo-Wang 2006 Nat Neurosci
    #       SC threshold; Stine-Shadlen 2023 LIP-accumulate/SC-commit). The host
    #       reads which commit_X burst — a thresholded spiking event, NOT an
    #       argmax over graded rates. All additive, read-only, NO sim/ edit.
    sel_recurrent_density: float = 0.5,   # sel_X internal recurrence density (Wang attractor)
    sel_recurrent_weight: float = 1.0,    # sel_X -> sel_X NMDA-slow gain (soft-WTA alpha<1)
    enable_commit_burst: bool = True,     # build the commit_X / commit_OPN burst stage
    n_commit_per_action: int = 20,        # neurons per commit_X burst pool
    n_commit_opn: int = 20,               # neurons in the shared omnipause pool
    sel_to_commit_weight: float = 22.0,   # sel_X -> commit_X (the winning ramp fires the burst)
    commit_recurrent_density: float = 0.5,  # commit_X internal recurrence (all-or-none burst)
    commit_recurrent_weight: float = 0.6,   # commit_X -> commit_X (burst regeneration; low to avoid rebound-bursting)
    opn_to_commit_weight: float = 10.0,   # commit_OPN -> commit_X tonic inhibition
    # OPN tonic default 0: the commit_X burst pool is gated by the sel_X ramp +
    # its own intrinsic IZH threshold (commit fires all-or-none only when its
    # sel_X has ramped high enough — the deep-research finding's documented
    # "minimal variant"). A CONSTANT commit_OPN drive (the textbook SC/OPN gate,
    # H.24) induces SYNCHRONIZED REBOUND BURSTING across all commit pools on this
    # rate-coded substrate (the symmetric-inhibition instability — Rutishauser):
    # 500pA -> all commit fire (rebound); 200pA -> none fire. No constant middle
    # exists, so the structurally-faithful OPN is left available but OFF by
    # default; opt in with --commit-opn-tonic-pa for experiments.
    commit_opn_tonic_pA: float = 0.0,
    # Distributed motor coding (2026-05-02). Adds excitatory cross-coupling
    # between motor pools at ADJACENT cardinal directions (N↔E, E↔S, S↔W,
    # W↔N — 90° angular distance). Opposite directions (N↔S, E↔W) get NO
    # coupling. Models real M1's overlapping somatotopy (Penfield 1937
    # homunculus has fuzzy boundaries) and Pulvermüller's distributed
    # action-word coding (1999/2005). Weight is small (~0.5) to soften
    # the labeled-line architecture without dissolving pool selectivity.
    # Hypothesis: 28.5% W→A ceiling is partly due to rigid 4-pool argmax;
    # smoother tuning may extract more signal.
    enable_motor_cross_coupling: bool = False,
    motor_cross_coupling_weight: float = 0.5,
    motor_cross_coupling_density: float = 0.3,
    # Full distributed motor pool (Pulvermüller G.20, 2026-05-02). Replaces
    # 4 separate motor_X pools (10 neurons each, total 40) with 8 sub-pools
    # at 45° angular intervals (5 neurons each, total 40 — same neuron count).
    # See docs/plans/2026-05-02-distributed-motor-pool-design.md.
    # Sub-pools: motor_pop_E (0°), motor_pop_NE (45°), motor_pop_N (90°),
    #   motor_pop_NW (135°), motor_pop_W (180°), motor_pop_SW (225°),
    #   motor_pop_S (270°), motor_pop_SE (315°).
    # Cosine-tuned pathways: each thal_X / cortex_X drives motor_pop_θ
    # with weight scaled by max(0, cos(θ_X - θ)).
    # Action selection / W->A eval: population vector decoding
    # (Georgopoulos 1986).
    # Default OFF for backwards compat. Incompatible with
    # enable_motor_lateral_inhibition (FS inhibition assumes 4 pools).
    enable_distributed_motor_pop: bool = False,
    n_motor_pop_per_subpool: int = 5,  # 8 sub-pools × 5 = 40 (matches default)
    # Cortex-level WTA (Phase B follow-up to plastic-input-layer cold-start).
    # Adds per-pool FS interneurons that mediate cross-pool inhibition.
    # Mirrors motor WTA pattern. Goal: enforce one-cortex-pool-wins regardless
    # of how noisy the input drive is. Lets hippocampus / sensory plastic layers
    # add drive on top of heuristic without washing out cascade selectivity.
    enable_cortex_lateral_inhibition: bool = False,
    n_cortex_fs_per_action: int = 5,
    # Scaled down 2.5x from motor WTA values: cortex pools are 25 neurons each
    # (vs 10 for motor), so density=1.0 gives 2.5x more synapses. Compensating
    # keeps total drive into/from FS comparable to motor case.
    cortex_to_fs_weight: float = 20.0,
    fs_to_cortex_weight: float = 8.0,
    # Real perception (option #3 in Phase B follow-up): replace heuristic
    # cortex drive with a learned sensory→cortex mapping. Adds a 49-neuron
    # sensory layer tuned to (dx, dy) ∈ [-3, 3]² relative-position pairs.
    # Plastic sensory→cortex pathways must learn position-to-action mapping
    # via STDP+reward.
    enable_learned_perception: bool = False,
    n_sensory: int = 49,  # 7×7 grid of (dx, dy)-tuned neurons
    sensory_to_cortex_weight: float = 10.0,
    # Hippocampal module (option #1 in Phase B follow-up): adds place cells and
    # goal cells, both Gaussian-tuned (sparse). Plastic place+goal → cortex
    # pathways let the agent learn spatial→action associations. Replaces
    # heuristic cortex drive when enabled. Sparse encoding (σ=0.5) avoids
    # cascade saturation that broke earlier dense-encoding attempts.
    enable_hippocampus: bool = False,
    n_hippocampus_per_layer: int = 64,  # 8×8 grid place + 8×8 grid goal cells
    hippocampus_to_cortex_weight: float = 10.0,
    # Working memory in PFC (Item 3, 2026-04-27).
    # Adds a prefrontal cortex region with recurrent internal connectivity
    # to support persistent activity (working memory). Real PFC neurons
    # show sustained firing across delay periods to maintain task-relevant
    # information. With this region, goal_cells project to PFC (plastic),
    # PFC has dense recurrent connectivity (plastic), PFC projects to
    # cortex (plastic). Tests whether PFC can hold goal info across delays.
    enable_pfc: bool = False,
    n_pfc: int = 60,
    pfc_internal_density: float = 0.2,  # recurrent connectivity for persistence
    goal_to_pfc_weight: float = 8.0,
    pfc_to_cortex_weight: float = 8.0,
    # Cluster G v2 (2026-05-01): when True, the dlpfc_wm region gets
    # BrainRegion.enable_nmda=True so NMDA-mediated bistability applies
    # ONLY to PFC neurons, not globally. Composes with cfg.enable_nmda
    # via the bridge's cp_nmda_neuron_mask. Recommended over global NMDA
    # when stacking with hippocampus / cerebellum / etc.
    pfc_enable_nmda: bool = False,
    # Cheat #5: BG cross-projections (2026-04-27).
    # Default: cortex_X → str_D1_X only (same-action). Real biology has
    # cross-projections (cortex_E might also project weakly to str_D1_W,
    # learnable). With cross-projections enabled, all 16 cortex×D1 pairs
    # exist, but with cross-projections starting weak. Plasticity should
    # learn to weaken/strengthen them appropriately.
    enable_bg_cross_projections: bool = False,
    cross_projection_weight: float = 5.0,  # weak vs same-action 25.0
    cross_projection_density: float = 1.0,  # 1.0 = dense (24 cross-pathways); 0.25 = patch-matrix-like (6 of 24)
    cross_projection_topology_seed: int = 0,  # deterministic pathway selection when density < 1.0
    # Goal-beacon perception (Item 1 Stage 1, 2026-04-27 skeleton).
    # Replaces direct (gx, gy) goal access with beacon sensors that detect
    # beacon strength + direction (modeling biological cue perception).
    # Skeleton only — full wiring in trial loop deferred to next session.
    # See docs/plans/2026-04-27-perception-arc-plan.md for the full plan.
    enable_beacon_perception: bool = False,
    n_beacon_sensors: int = 8,  # 8 directional sensors (cardinal + diagonal)
    beacon_to_goal_weight: float = 8.0,
    # Landmark perception (Item 1 Stage 2, 2026-04-27).
    # Adds landmark_sensors region perceiving a FIXED-position landmark
    # (typically grid center). Used to self-organize place_cells via
    # plastic landmark_sensors → place_cells pathway. With a known fixed
    # landmark at L and 8 directional sensors, the (distance, bearing)
    # to L uniquely identifies agent position — place cells can learn to
    # fire at specific positions based on this multi-cell sensor pattern.
    enable_landmarks: bool = False,
    n_landmark_sensors: int = 8,
    landmark_to_place_weight: float = 8.0,
    # v3 (2026-04-28): MSN cross-pool lateral inhibition. Real BG sharpens
    # action selection via GABAergic collaterals between MSNs (within and
    # between action pools), striatal FS interneurons, and pallidal
    # center-surround. v3 adds the cross-pool MSN→MSN piece (the simplest
    # and most impactful). Without this, cross-projections (cheat #5)
    # corrupt the cascade because there's nothing to suppress cross-talk.
    # Static (plastic=False). MSN regions are GABAergic (exc_fraction=0.05)
    # so the projection is inhibitory.
    enable_bg_lateral_inhibition: bool = False,
    lateral_inhibition_density: float = 0.3,
    lateral_inhibition_weight: float = 2.0,
    # Cluster B.2 (2026-04-28): striatal fast-spiking interneurons.
    # Real BG striatum has ~1% PV-positive FSIs that provide fast convergent
    # GABAergic broadcast inhibition. Different from v3 MSN-MSN lateral
    # (slower, more local) — FSIs broadcast indiscriminately on a
    # millisecond timescale to bias which action's MSN pool wins.
    # Per-action FS pool receives same-action cortex drive, then inhibits
    # ALL striatal MSN pools (D1+D2, every action including same-action).
    # All FS pathways plastic=False (static gating, not plastic).
    # NOTE: kwargs are prefixed `cortex_to_str_fs_*` / `str_fs_to_msn_*` to
    # avoid collision with the cortex-WTA `cortex_to_fs_weight` (line 84)
    # and `fs_to_cortex_weight` (line 85) — different microcircuit.
    enable_striatal_fsis: bool = False,
    n_striatal_fs_per_action: int = 5,
    cortex_to_str_fs_weight: float = 30.0,
    # Cluster B.2 retune (2026-04-28 evening): initial guess of 8.0 caused
    # over-suppression — winner pool got suppressed by 35% (12.8 Hz drop)
    # while loser only got 1.6 Hz drop. With density=1.0 and 4 FS source
    # pools, effective inhibition was 32 (vs v3 lateral inhibition ~7).
    # Lowering to 2.0 → effective ~8, comparable to v3 lateral.
    str_fs_to_msn_weight: float = 2.0,
    # Cluster D v1 (2026-04-29): hippocampus trisynaptic loop.
    # Adds 5 regions (ec, dg, dg_pv_basket, ca3, ca1) and ~10 pathways implementing
    # the canonical Cajal trisynaptic loop:
    #   sensory + landmark_sensors -> ec
    #   ec -> dg (perforant path), ec -> dg_pv_basket (FFi recruitment)
    #   dg_pv_basket -> dg (strong feedforward inhibition for sparsity)
    #   ec -> ca1 (direct cortical bypass)
    #   dg -> ca3 (mossy fibers; sparse but strong)
    #   ca3 -> ca3 (recurrent autoassociator; via region.internal_density)
    #   ca3 -> ca1 (Schaffer collaterals)
    #   ca1 -> place_cells (readout into existing perception arc, when
    #     enable_hippocampus is on; otherwise CA1 still exists but its
    #     readout pathway into place_cells is omitted).
    # Composition: ADDS to existing perception arc; does NOT replace
    # place_cells/goal_cells regions or landmark_sensors -> place_cells.
    # See docs/plans/2026-04-29-cluster-d-hippocampus-design.md.
    enable_cluster_d_hippocampus: bool = False,
    # Cluster D v2 (2026-04-30): SWR-gated CA3 plasticity for offline cleanup.
    # When True (REQUIRES enable_cluster_d_hippocampus=True):
    #   - CA3 region's implicit internal_density is set to 0
    #   - An explicit ca3 -> ca3 RegionPathway is added with
    #     plasticity_gate="ca3_swr_burst", letting the runner gate STDP
    #     on the CA3 recurrent autoassociator on a per-step basis (open
    #     during sharp-wave-ripple bursts; suppressed otherwise during sleep).
    # See docs/plans/2026-04-30-cluster-d-v2-swr-design.md.
    enable_cluster_d_v2_swr: bool = False,
    # Cluster E v1 (2026-04-29): topographic maps + distance-dependent
    # connection probability. When enabled:
    #   - cortex_X / str_D1_X / str_D2_X regions get 2D coordinates anchored
    #     to a corner of the unit square (N=(0.5,1.0), E=(1.0,0.5),
    #     S=(0.5,0.0), W=(0.0,0.5)).
    #   - cortex_X -> str_D1_X / str_D2_X pathways are sampled with
    #     Gaussian-weighted probability (sigma=0.3 by default).
    # Default off — backward compatible.
    # See docs/plans/2026-04-29-cluster-e-topographic-maps-design.md.
    enable_cluster_e_topography: bool = False,
    cluster_e_distance_sigma: float = 0.3,
    # Cluster F v1: Marr-Albus-Ito cerebellar microcircuit. Adds 11 regions
    # (mossy_state, granule, purkinje_{N,E,S,W}, dcn_aip_{N,E,S,W},
    # inferior_olive) and ~25 pathways implementing state -> mossy -> granule
    # PF -> Purkinje -> DCN -inhibitory-> motor + IO -> Purkinje teaching.
    # Composes with Cluster A (closed BG loop): cerebellar DCN provides
    # additive contribution to motor pools alongside thal_X drive. v1 uses
    # reward-modulated STDP on PF->PC; full CF-gated LTD deferred to v2.
    # Default off — backward compatible.
    # See docs/plans/2026-04-29-cluster-f-cerebellum-design.md.
    enable_cluster_f_cerebellum: bool = False,
    # Number of cerebellar granule cells. Default 250 implements Marr's
    # sparse-expansion code at ~3-5% activity in our reduced model. Real
    # cerebellum has ~50M granule cells per hemisphere with ~150K
    # parallel-fiber inputs per Purkinje cell. The 250-cell setup breaks
    # Albus 1971's anti-Hebbian LTD calibration (F v2 NO-GO 2026-04-30).
    # Scaling experiment 2026-04-30: n_granule=1000-5000 tests whether
    # F v2 becomes viable at closer-to-biological scale.
    n_granule: int = 250,
    # Cluster K v1 (2026-05-01): visual cortex hierarchy.
    # Adds retina (32x32 ON/OFF) → V1_simple (Hubel & Wiesel 1962 simple cells,
    # orientation-tuned via Gabor RF) → V1_complex (phase-pooled) → V2 → IT
    # (Felleman & Van Essen 1991 ventral-stream hierarchy). For v1, regions
    # are built but image rendering + drive injection happen outside this
    # function (in the runner step loop). Default off — backward compatible.
    # See sim/visual_cortex.py and docs/plans/2026-05-01-cluster-k-visual-cortex-hierarchy.md.
    # Sizes are reduced from the visual_cortex.py defaults to keep the
    # gridworld model tractable: 8 orient × 2 freq × 8x8 pos = 1024 V1
    # simple, vs 8192 in the full module.
    enable_visual_cortex: bool = False,
    visual_n_orientations: int = 8,
    visual_n_frequencies: int = 2,
    visual_n_positions_per_dim: int = 8,
    visual_image_size: int = 32,  # retina spatial dim (32x32 pixels)
    visual_n_v2: int = 256,
    visual_n_it: int = 64,
    # Cluster K v2 (2026-05-01): IT → cortex_X action-selection density
    visual_it_to_cortex_density: float = 0.5,
    # Spiking superior colliculus (N1 orienting; 2026-06-10). A retinotopic SC sheet
    # (sc_map) + Mexican-hat surround (sc_fs) that, fed the egocentric retinal image,
    # produces the orienting cardinal as cortex_{N,E,S,W} firing — the spiking
    # replacement for the host sc_orienting_cardinal_from_image reflex (N1). Requires
    # enable_visual_cortex (uses the retina). De-risked: 2026-06-10-N1-N5-spiking-SC-derisk-RESULT.md.
    enable_spiking_sc: bool = False,
    n_spiking_sc_fs: int = 12,
    # N5 Option C (2026-06-10): the neural approach-reward. A slow-channel temporal-difference
    # of the SC bump's rostral-ward motion (sc_rostral - sc_rostral_slow via gaba_b) -> approach
    # -> reward_us, replacing the host sign(delta eccentricity). Requires enable_spiking_sc +
    # spiking_reward_us. De-risked: sc_approach_td_probe.py.
    enable_spiking_sc_approach: bool = False,
    # Text I/O (2026-05-01)
    enable_text_io: bool = False,
    text_n_input_neurons: int = 256,
    text_n_output_neurons: int = 256,
    text_input_to_pfc_density: float = 0.20,
    text_input_to_pfc_weight: float = 2.0,
    text_input_to_cortex_density: float = 0.20,
    text_it_to_output_density: float = 0.20,
    # Non-zero default init for language-to-cortex (per Kandel ch 53,
    # developmental pruning starts from dense connectivity)
    text_input_to_cortex_weight: float = 2.0,
    text_input_to_cortex_jitter: float = 0.5,
    # PFC-bypass: direct language_input → motor_X (Kandel ch 60 anatomy)
    text_input_to_motor_density: float = 0.30,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    # Readout pathway initial weights (2026-05-02 fix). Default 0.0 was
    # the original design (STDP grows from scratch) but with weak training
    # signal, growth doesn't happen — the pathways stayed at floor in the
    # 100-ep Hebbian-off test. Small non-zero init lets STDP both LTP
    # correct pairings and LTD wrong ones. Biology source: real cortical
    # synapses have spontaneous baseline weights, not absolute zero.
    text_cortex_to_output_weight: float = 0.0,
    text_it_to_output_weight: float = 0.0,
    text_cortex_to_output_jitter: float = 0.0,
    text_it_to_output_jitter: float = 0.0,
):
    """Returns list of BrainRegion + list of RegionPathway for the BG circuit.

    When `enable_motor_lateral_inhibition=True`, adds 4 motor_FS_X regions
    (FS interneuron sub-pools, exc_fraction=0.0) plus pathways:
      - motor_X → motor_FS_X (excitatory; motor's own activity drives its FS)
      - motor_FS_X → motor_Y for Y != X (inhibitory; FS suppresses other motors)
    This creates standard cortical winner-take-all microcircuit dynamics:
    when motor_X fires, motor_FS_X fires, suppressing motor_{Y,Z,W}.
    """
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    # Cluster D v2 requires v1 — there's no CA3 region to gate without it.
    if enable_cluster_d_v2_swr and not enable_cluster_d_hippocampus:
        raise ValueError(
            "enable_cluster_d_v2_swr=True requires enable_cluster_d_hippocampus=True "
            "(cluster D v1 builds the CA3 region that v2 gates). Either enable v1 "
            "or disable v2."
        )

    regions = []
    pathways = []

    # Hippocampal module (opt-in): place + goal cells with sparse Gaussian tuning.
    # Place cells encode agent (x, y), goal cells encode goal (gx, gy). Both
    # project plastically to all 4 cortex pools so the agent can learn
    # (place, goal) → action associations via STDP+reward.
    # Sparse encoding (σ=0.5 in runner): only 1-3 cells fire per position →
    # avoids cascade saturation that broke previous dense sensory encoding.
    if enable_hippocampus:
        regions.append(BrainRegion(
            name="sensor_place_readout",
            n_neurons=n_hippocampus_per_layer,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="ppc_goal_input",
            n_neurons=n_hippocampus_per_layer,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))

    # Goal-beacon perception (Item 1 Stage 1 skeleton, 2026-04-27). Replaces
    # direct (gx, gy) goal access with directional beacon sensors. Each sensor
    # has a preferred bearing; activation is proportional to beacon intensity
    # × cosine alignment with sensor direction. Plastic beacon → goal_cells
    # pathway lets goal_cells learn to integrate sensor patterns into spatial
    # representations. Full trial-loop wiring deferred to next session.
    if enable_beacon_perception:
        regions.append(BrainRegion(
            name="beacon_sensors",
            n_neurons=n_beacon_sensors,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Landmark perception (Item 1 Stage 2, 2026-04-27). Fixed-position
    # landmark with 8 directional sensors. Plastic landmark_sensors →
    # place_cells pathway lets place cells self-organize from the unique
    # (distance, bearing) pattern at each agent position. Replaces direct
    # (x, y) place cell access with biologically-grounded localization.
    if enable_landmarks:
        regions.append(BrainRegion(
            name="landmark_sensors",
            n_neurons=n_landmark_sensors,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # PFC working memory (Item 3, 2026-04-27): recurrent prefrontal region.
    # Internal density > 0 enables recurrent connections that can sustain
    # activity across delay periods (persistent activity / attractor dynamics).
    # PFC pyramidal preset has biophysical features for sustained firing.
    if enable_pfc:
        regions.append(BrainRegion(
            name="dlpfc_wm",
            n_neurons=n_pfc,
            exc_fraction=0.8,
            internal_density=pfc_internal_density,
            exc_weight_mean=2.0,  # moderate self-excitation for persistence
            inh_weight_mean=4.0,
            weight_jitter=0.2,
            plastic_internal=True,  # plastic recurrence supports learning
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
            # IZH2007_HIPPO_PYRAMIDAL works for PFC-style dynamics; can switch
            # to dedicated PFC preset (HH_PFC_PYRAMIDAL) for full biophysics.
            # Cluster G v2: tag PFC for NMDA-mediated bistability (Wang 2002)
            # only when pfc_enable_nmda is set. Other regions keep enable_nmda=False
            # so global cfg.enable_nmda only activates NMDA dynamics here.
            enable_nmda=bool(pfc_enable_nmda),
        ))

    # Sensory layer (opt-in): position-tuned input neurons feeding cortex.
    # Replaces heuristic cortex drive when enable_learned_perception=True.
    # Each sensory neuron is tuned to a relative-position (dx, dy) ∈ [-3, 3]².
    # 7×7 grid = 49 neurons. The runner sets per-step drive based on goal offset.
    if enable_learned_perception:
        regions.append(BrainRegion(
            name="sensory",
            n_neurons=n_sensory,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Cortex (input layer for goal-directed signals).
    # Split into per-action pools so different inputs preferentially activate
    # different actions. This is a phenomenological substitute for what
    # learning would produce: differential cortex→striatum weights.
    # Cluster C v2 (2026-04-29): action_index stamped on action-specific
    # regions so cp_synapse_action_tag can resolve per-synapse DA targeting.
    # Cluster E v1 (2026-04-29): topographic 2D coordinates per action
    # corner of unit square when enable_cluster_e_topography is on.
    n_cortex_per_action = n_cortex // N_ACTIONS
    # Cardinal-direction corners of the unit square (Cluster E v1).
    _action_corner = {
        "N": (0.5, 1.0),
        "E": (1.0, 0.5),
        "S": (0.5, 0.0),
        "W": (0.0, 0.5),
    }
    _topo_kw = (
        {"coordinate_dim": 2, "coordinate_extent": (1.0, 1.0)}
        if enable_cluster_e_topography
        else {}
    )
    # cortex_{N,E,S,W}: per-action motor-cortex (M1-equivalent) pools.
    # Anatomy: regular-spiking pyramidal neurons (RS preset). The "cortex_"
    # prefix is project shorthand; biologically these stand in for primary
    # motor cortex columns wired in topographic action channels (cf.
    # Cluster E catalog, Kandel 6e Ch 38). Each pool drives the
    # corresponding striatal D1/D2 channel (cortex -> str_d1_X / str_d2_X).
    for action_idx, action in enumerate(ACTION_NAMES):
        kw = dict(_topo_kw)
        if enable_cluster_e_topography:
            kw["coordinate_center"] = _action_corner[action]
        regions.append(BrainRegion(
            name=f"cortex_{action}",
            n_neurons=n_cortex_per_action,
            exc_fraction=1.0,  # All excitatory for cortex inputs
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            action_index=action_idx,
            # Cluster G v2.5: cortical pyramidals naturally express NMDA
            # receptors (Wang 2002 calibration applies). Enable when
            # pfc_enable_nmda is set so cortex_X + dlpfc_wm both get NMDA-
            # mediated bistability while hippocampus/cerebellum stay AMPA-only.
            enable_nmda=bool(pfc_enable_nmda),
            **kw,
        ))

    # Cortex WTA microcircuit (opt-in). Per-pool FS interneurons that mediate
    # cross-pool inhibition: cortex_X drives cortex_FS_X, which inhibits
    # cortex_{Y,Z,W}. Standard cortical WTA pattern, mirror of motor WTA.
    # Goal: enforce clean pool selectivity even when plastic input layers
    # (hippocampus, learned-perception) add noisy drive across all 4 pools.
    if enable_cortex_lateral_inhibition:
        for action_idx, action in enumerate(ACTION_NAMES):
            regions.append(BrainRegion(
                name=f"cortex_FS_{action}",
                n_neurons=n_cortex_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory → outgoing synapses are inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                action_index=action_idx,
            ))

    # Per-action striatal pools (D1 direct, D2 indirect).
    # internal_density=0 (no lateral inhibition) initially — MSNs need
    # strong cortex drive to escape the down-state and lateral inhibition
    # makes that even harder. Add it back later if action selection needs
    # sharpening.
    for action_idx, action in enumerate(ACTION_NAMES):
        # Striatal MSNs: ECl ~−60 mV (PBR-160 ch 6, gramicidin perforated patch).
        # IPSPs are shunting near rest, hyperpolarizing only near AP threshold.
        # Cluster E v1: same per-action corner as cortex for topographic
        # cortex_X -> str_D{1,2}_X mapping.
        msn_kw = dict(_topo_kw)
        if enable_cluster_e_topography:
            msn_kw["coordinate_center"] = _action_corner[action]
        regions.append(BrainRegion(
            name=f"str_D1_{action}",
            n_neurons=n_striatum_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,
            action_index=action_idx,
            **msn_kw,
        ))
        regions.append(BrainRegion(
            name=f"str_D2_{action}",
            n_neurons=n_striatum_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D2.name,
            syn_reversal_potential_i_override=-60.0,
            action_index=action_idx,
            **msn_kw,
        ))

    # Cluster B.2 (2026-04-28): striatal fast-spiking interneurons (FSIs).
    # ~1% of striatal cells; PV-positive; broadcast inhibition. One small
    # str_PV_FSI_{N,E,S,W}: per-action striatal fast-spiking interneurons.
    # Strict naming: this is the **PV-FSI** class (parvalbumin-positive
    # fast-spiking) — one of EIGHT distinct striatal GABAergic interneuron
    # classes catalogued in Tepper-2018 (the others are NPY-LTS, NPY-NGF,
    # CR, TH/THIN, FAI, SABI, plus the cholinergic ChI/TAN). The "str_FS"
    # prefix in this codebase models PV-FSI specifically — it is NOT a
    # generic "all striatal interneurons" pool. The class is named "FS"
    # for its short-AP / high-rate firing (Tepper 2018 ch 8). Catalog
    # ref: TK-2017 ch 8; Tepper 2018 §"Functional Significance".
    # FS pool per action, all GABAergic (exc_fraction=0.0) so the outgoing
    # synapses are auto-derived inhibitory by the bridge. No internal
    # recurrence: FSIs just receive cortex drive and broadcast to all MSNs.
    if enable_striatal_fsis:
        for action_idx, action in enumerate(ACTION_NAMES):
            regions.append(BrainRegion(
                name=f"str_PV_FSI_{action}",
                n_neurons=n_striatal_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory → outgoing synapses are inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                action_index=action_idx,
            ))

    # Per-action BG output (GPe / GPi)
    # R3.7 (2026-04-29): GPe is split into PV+ (prototypic) and PV-
    # (arkypallidal) subpools per Mallet 2008 / Kita 2007 (PBR-160 ch 7).
    # gpe_X = prototypic (PV+), forming the canonical GPe -> STN/GPi/SNr
    # projection. gpe_arky_X = arkypallidal (PV-), forming the
    # GPe -> striatum feedback (broadcasts onto FSIs, "stop-signal"
    # role per Mallet 2012). Sizes: PV+ at the original n_gpe_per_action
    # (10), PV- at n_gpe_arky_per_action (4) — consistent with Kita's
    # observation that PV-negative cells form ~1/3 of GPe.
    for action_idx, action in enumerate(ACTION_NAMES):
        regions.append(BrainRegion(
            name=f"gpe_{action}",  # prototypic (PV+); existing alias preserved
            n_neurons=n_gpe_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPE_PACEMAKER.name,
            action_index=action_idx,
        ))
        regions.append(BrainRegion(
            name=f"gpe_arky_{action}",  # arkypallidal (PV-); R3.7 new pool
            n_neurons=n_gpe_arky_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPE_PACEMAKER.name,
            action_index=action_idx,
        ))
        # gpi_{N,E,S,W}: BG-output complex per action (GPi/SNr in primates;
        # predominantly SNr in rodents — internal-pallidal cells are sparse
        # in rats/mice and SNr carries most output-nucleus work). Tonic
        # 40-80 Hz GABAergic projection neurons. Disinhibition via direct
        # pathway (D1 MSN -> GPi/SNr) is the canonical "go" mechanism.
        # Catalog refs: Kandel 6e Ch 38 p 935-943; PBR-160 ch 9 Deniau.
        regions.append(BrainRegion(
            name=f"gpi_{action}",
            n_neurons=n_gpi_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
            action_index=action_idx,
        ))
        # R3.11 (2026-04-29): striosome (patch) compartment.
        # Per PBR-160 ch 9 / ch 11: striosomes are D1-MSN-rich patches
        # that project to BOTH SNc (canonical, drives DA) and SNr (gpi)
        # in addition to the matrix-pathway. The patch/matrix split
        # aligns with SNc/SNr at the output level. Real input is limbic
        # (vmPFC, amygdala, ventral hippocampus); we use cortex_X as a
        # placeholder until a limbic source is added (Cluster O work).
        # E_inh override -60 mV is inherited via the same MSN-class
        # convention applied to str_D1/D2.
        regions.append(BrainRegion(
            name=f"str_striosome_{action}",
            n_neurons=n_str_striosome_per_action,
            exc_fraction=0.05,  # MSN is GABAergic with sparse glutamatergic spillover
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,  # MSN GABA_A reversal (R1.1)
            action_index=action_idx,
        ))

    # Single STN (excitatory, projects diffusely to all GPi)
    regions.append(BrainRegion(
        name="stn",
        n_neurons=n_stn,
        exc_fraction=1.0,  # STN is glutamatergic (excitatory)
        internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_STN_BURST.name,
    ))

    # Per-action thalamic relay + motor cortex
    for action_idx, action in enumerate(ACTION_NAMES):
        regions.append(BrainRegion(
            name=f"thal_{action}",
            n_neurons=n_thal_per_action,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_THALAMIC_RELAY.name,
            action_index=action_idx,
        ))
        # Default: 4 labeled motor pools (motor_N/E/S/W). Skipped when
        # distributed-motor-pop is enabled (replaced with 8 sub-pools below).
        if not enable_distributed_motor_pop:
            regions.append(BrainRegion(
                name=f"motor_{action}",
                n_neurons=n_motor_per_action,
                exc_fraction=1.0,
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                action_index=action_idx,
                # Cluster G v2.5: motor cortex pyramidals also express NMDA;
                # included for consistency with cortex_X enable_nmda.
                enable_nmda=bool(pfc_enable_nmda),
            ))

    # Distributed motor pool (Pulvermüller G.20, 2026-05-02). 8 sub-pools at
    # 45° angular intervals, n_motor_pop_per_subpool neurons each.
    # See docs/plans/2026-05-02-distributed-motor-pool-design.md
    if enable_distributed_motor_pop:
        # Sub-pool angles in degrees: E, NE, N, NW, W, SW, S, SE
        for theta_deg, suffix in [
            (0, "E"), (45, "NE"), (90, "N"), (135, "NW"),
            (180, "W"), (225, "SW"), (270, "S"), (315, "SE"),
        ]:
            regions.append(BrainRegion(
                name=f"motor_pop_{suffix}",
                n_neurons=n_motor_pop_per_subpool,
                exc_fraction=1.0,
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                action_index=None,  # Not action-specific in the labeled-line sense
                enable_nmda=bool(pfc_enable_nmda),
            ))

    # SNc dopamine neurons (single pool, broadcasts via neuromodulator subsystem).
    # Anatomy note: this region is the project's A9-equivalent — SNc
    # dopaminergic neurons that drive nigrostriatal projections. The
    # mesolimbic A10/VTA → NAc/PFC arms are NOT separately modeled; the
    # single `snc` pool collapses A9 + A10 into one broadcast modulator.
    # With Cluster C v2 (`--enable-compartmentalized-da`), per-action DA
    # channels (dopamine_{N,E,S,W}) decompose this into per-action
    # targeting, though still A9-typed. The transmitter (`dopamine`
    # neuromodulator) keeps its canonical chemistry name; only the
    # *region* renamed from "dopamine" → "snc" 2026-04-29 (Wave-1 #3).
    # Catalog refs: Kandel 6e Ch 11 (DA system); PBR-160 ch 11 (Tepper & Lee).
    # SNc DA neurons lack KCC2 → ECl ~−55 mV (PBR-160 ch 11). GABA_A is
    # depolarizing or even excitatory at rest in adult SNc; override the
    # cortical-pyramidal default of −75 mV.
    regions.append(BrainRegion(
        name="snc",
        n_neurons=n_dopamine,
        exc_fraction=1.0,
        internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
        syn_reversal_potential_i_override=-55.0,
    ))

    # reward_us — the SPIKING unconditioned-stimulus (US) afferent that DRIVES the SNc reward burst
    # (2026-06-10, the spiking-ification of the host SNc reward write; research
    # 2026-06-10-N9-spiking-reward-and-critic-normalization, catalog C.33 PPN sensory+reward->DA).
    # Biology: DA neurons do NOT compute reward internally — they are DRIVEN to burst by an EXCITATORY
    # afferent (PPN/PBN glutamate; Watabe-Uchida 2012 inputome) that carries the PRIMARY (perceived)
    # reward signal. Today the runner writes `snc += snc_reward_gain*max(0,reward)` DIRECTLY onto the
    # DA cell (a number -> DA current with NO neuron between = the shortcut). With spiking_reward_us,
    # this PPN-like population receives the PERCEIVED reward (the coord-free N5 approach signal, a
    # sensory drive like the place/retina injection) and FIRES into the SNc, so the reward burst is
    # produced by a NEURON's synapse (US->VTA), and the whole δ=r−V is synaptic (r from reward_us
    # excitation, V from the striosome GABA_B). plastic=False: the unconditioned US->DA reflex arc is
    # innate (only the cue->prediction learning is plastic, and that lives in the actor/critic).
    # Default OFF => byte-equivalent (region/pathway absent). RS pyramidal = closest excitable relay.
    if spiking_reward_us:
        regions.append(BrainRegion(
            name="reward_us", n_neurons=int(n_reward_us), exc_fraction=1.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Spiking-SNc actor-critic Stage B (2026-06-08): a dedicated neural value
    # critic. `striosome_value` is the striosome/patch state-value population
    # (Houk-Adams-Barto 1995 / catalog C.30: striosome-patch = critic V(s)).
    # FULLY GABAergic (MSNs are ~100% inhibitory) so its projection SUBTRACTS at
    # the SNc; no internal recurrence so V is a graded value readout that scales
    # with the learned cortex_it->striosome_value weight (not a WTA gate). This
    # is Option A of 2026-06-08-spiking-snc-stageB-striosome-critic-research.md
    # (a dedicated region — cleaner + more additive than re-purposing the four
    # per-action str_striosome_* pools, which are Q(s,a)-shaped and action-cortex
    # driven). Mirrors the validated CPU de-risk's striosome_value recipe.
    if enable_neural_critic and neural_place_selforg:
        # ═══ N9 NEURAL PLACE-CODE SELF-ORG afferent (2026-06-09 nav deployment of the de-risk
        #     n9_place_graded_critic_stage2_derisk._build). REPLACES the host-Gaussian
        #     vs_place_context (a BRAIN-BASED-ONLY shortcut). Three regions: ═══
        # (1) place_sensors — the legitimate egocentric landmark sensors (bearing+distance render,
        #     driven externally each nav step; (x,y) enters the brain ONLY here). EXC stub.
        _n_place_sensors = int(N_PLACE_LANDMARKS) * (int(n_place_bearing) + int(n_place_dist))
        regions.append(BrainRegion(
            name="place_sensors", n_neurons=_n_place_sensors, exc_fraction=1.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        # (2) place — the SELF-ORGANIZING place pool (hippocampal pyramidal; competition = the
        #     cell's own threshold WTA). NO per-region homeostasis (anti-cheat: it fires from the
        #     LEARNED synaptic current, not a threshold collapse). Mirrors the de-risk `place`.
        regions.append(BrainRegion(
            name="place", n_neurons=int(n_place), exc_fraction=1.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        # (3) place_fs — the FS-PING gamma synchronizer (reciprocally wired to `place` below). The
        #     gamma EMERGES from neurons+synapses (CORTEX_GAMMA_FS_NETWORK pattern); location-BLIND
        #     (it sets WHEN the active place cells fire, the place code selects WHICH) so the
        #     distinctness is preserved, not densified. Mirrors the de-risk `place_fs`.
        regions.append(BrainRegion(
            name="place_fs", n_neurons=int(n_place_fs), exc_fraction=0.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
        ))
        # The MSN-D1 value critic. NMDA ON (the Route-D coincidence plateau reuses the Mg2+-block
        # kernel — the per-region NMDA mask restricts it to this slice). Optional per-region INTRINSIC
        # homeostasis (committed sim/ edit 89b8d909; Turrigiano/Desai) via --enable-critic-homeostasis:
        # the place-code self-org is CuPy-non-deterministic (transpose-SpMV atomic scatter, research
        # 2026-06-10-N9-placecode-reproducibility-robustness-research.md), so the volley strength —
        # hence the critic rate — varies 28-118 Hz run-to-run, and a single GABA_B prop can't serve
        # both (28→arithmetic, 118→clamp). Intrinsic homeostasis defends a TARGET critic rate by
        # adapting the cell's OWN threshold to the volley strength → normalizes the readout across
        # draws so the draw stops mattering (the research's primary lever B1). Applied ONLY to the
        # critic here (NOT the self-org `place` pool, whose threshold-WTA is the competition — adding
        # it there would densify/blur the code; the volley, not a weak linear afferent, supplies the
        # critic's drive, so critic-only is the path-appropriate form — unlike the vs_place_context
        # linear-afferent path where critic-only failed). Default OFF (byte-equivalent). Anti-cheat:
        # the threshold is the cell's own (intrinsic, neural), and grading near≫far must SURVIVE
        # (Stage-B re-asserts it; a 0-drive far can't cross even a lowered threshold).
        regions.append(BrainRegion(
            name="striosome_value", n_neurons=n_striosome_value,
            exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,
            enable_nmda=True,
            enable_homeostasis=bool(enable_critic_homeostasis),
        ))
    elif enable_neural_critic:
        # The DEDICATED DENSE place-context afferent (2026-06-09 VALIDATED redesign). A
        # grid-32-tuned Gaussian place code drive-injected each nav step (wide sigma => 30-80
        # cells fire/location, the convergent-excitation up-state the SPARSE actor place code
        # ~1-3 cells cannot deliver). RS_CORTICAL_PYRAMIDAL (excitable) + per-region homeostasis
        # so it reaches a firing range under the deterministic regime. Feeds ONLY the critic
        # (no edge to the actor cortex — actor-not-perturbed gate-4 of the de-risk: ratio 1.000).
        # Mirrors snc_stageb_critic_probe_navfaithful.py:_build_navfaithful_bridge vs_place_context.
        regions.append(BrainRegion(
            name="vs_place_context",
            n_neurons=n_vs_place_context,
            exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            # Per-region homeostasis on the AFFERENT (committed sim/ edit 89b8d909). The forensic
            # showed the global-homeostasis "fix" lifted V by firing the afferent harder (lowering
            # its threshold); per-region homeostasis on the afferent reproduces that faithfully.
            # Place-blindness risk did NOT materialize (de-risk gate-5: ~59 Hz near vs 0 Hz far) —
            # a cell with ~0 drive at FAR can't cross even a lowered threshold. GLOBAL stays OFF.
            enable_homeostasis=bool(enable_critic_homeostasis),
        ))
        if enable_convergent_upstate:
            # A1 — the convergent-excitation UP-STATE drive (B.02; design Option A). A DISTINCT
            # dense afferent region (a second pathway from vs_place_context would COLLIDE in the
            # RegionManager's (from,to) key, sim/regions.py:537). Drive-injected with the SAME
            # grid-32 Gaussian place code as vs_place_context each nav step. Its dense, NON-plastic
            # vs_place_drive->striosome_value pathway (below) sums past the MSN-D1 rheobase at the
            # goal so the critic is in a location-gated up-state from init (breaks the LTP bootstrap
            # WITHOUT the homeostasis threshold-collapse). RS_CORTICAL_PYRAMIDAL like vs_place_context.
            regions.append(BrainRegion(
                name="vs_place_drive",
                n_neurons=n_vs_place_context,
                exc_fraction=1.0, internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
        regions.append(BrainRegion(
            name="striosome_value",
            n_neurons=n_striosome_value,
            exc_fraction=0.0,            # fully GABAergic: a pure inhibitory value
            internal_density=0.0,       # graded readout, not winner-take-all
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
            syn_reversal_potential_i_override=-60.0,  # MSN GABA_A reversal
            # Per-region homeostasis on the CRITIC (committed sim/ edit 89b8d909). Intrinsic
            # homeostatic plasticity lets the under-active MSN-D1 reach a firing range from its
            # place afferent. GLOBAL cfg.enable_homeostasis stays False (deterministic regime
            # preserved). The actor regions DO NOT set this — only the critic gets the mask.
            enable_homeostasis=bool(enable_critic_homeostasis),
        ))

    # Cluster D v1 (2026-04-29): hippocampus trisynaptic loop.
    # Five new regions implementing the canonical Cajal loop. See
    # docs/plans/2026-04-29-cluster-d-hippocampus-design.md.
    #   ec (entorhinal cortex stub) — receives sensory + landmark, projects
    #     to DG, CA1; bridges perception to hippocampus proper.
    #   dg (dentate gyrus) — pattern separation via FFi-driven sparsity;
    #     internal_density=0 (no recurrence — DG granule cells fire sparsely).
    #   dg_pv_basket — fast-spiking interneurons providing strong feedforward
    #     inhibition (exc_fraction=0.0 → outputs auto-derived inhibitory).
    #   ca3 — pattern completion; recurrent autoassociator core
    #     (internal_density=0.30 generates the dense recurrent collaterals).
    #   ca1 — readout integrating direct EC input + CA3 output; projects
    #     into existing place_cells region when enable_hippocampus is on.
    if enable_cluster_d_hippocampus:
        regions.append(BrainRegion(
            name="ec",
            n_neurons=80,
            exc_fraction=0.8,
            internal_density=0.05,
            exc_weight_mean=0.3, inh_weight_mean=0.8,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="dg",
            n_neurons=200,
            exc_fraction=0.95,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="dg_pv_basket",
            n_neurons=60,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
        ))
        # Cluster D v2 SWR-cleanup: when enabled, the CA3 self-loop is
        # pulled out of the implicit `internal_density` mechanism (which
        # has no plasticity_gate hook) and rewired below as an explicit
        # ca3 -> ca3 pathway with `plasticity_gate="ca3_swr_burst"`. That
        # lets the runner gate STDP on the recurrent autoassociator on a
        # per-step basis (open during ripple bursts; suppressed otherwise
        # during sleep). plastic_internal stays True for symmetry but is
        # a no-op once internal_density is 0.
        ca3_internal_density = 0.0 if enable_cluster_d_v2_swr else 0.30
        regions.append(BrainRegion(
            name="ca3",
            n_neurons=100,
            exc_fraction=0.85,
            internal_density=ca3_internal_density,
            exc_weight_mean=1.5, inh_weight_mean=2.0,
            weight_jitter=0.2, plastic_internal=True,  # recurrent CA3 plasticity
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="ca1",
            n_neurons=120,
            exc_fraction=0.85,
            internal_density=0.05,
            exc_weight_mean=0.3, inh_weight_mean=0.8,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))

    # Cluster F v1 (2026-04-29): Marr-Albus-Ito cerebellar microcircuit.
    # Five region types per the catalog (F.01-F.06):
    #   mossy_state     — single MF input pool (v2 splits into 3 streams F.03)
    #   granule         — sparse expansion code, ~3-5% active (Marr §3, Albus §IV.A)
    #   purkinje_X      — per-action PC pool; tonic 30-80 Hz; PF input modulates rate
    #   dcn_aip_X       — per-action AIP-equivalent; tonic 40 Hz; PC pause -> disinhibition
    #   inferior_olive  — sparse ~1 Hz; CF teaching signal (v1 driven by Δd>0 trigger)
    # Per-action structure (X in {N,E,S,W}) mirrors the BG cascade for clean
    # composition with Cluster A. The granule->purkinje pathway is the
    # learning site (PF->PC plasticity).
    if enable_cluster_f_cerebellum:
        regions.append(BrainRegion(
            name="mossy_state",
            n_neurons=60,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="granule",
            n_neurons=n_granule,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            # Granule cells are small and fire briefly. RS preset is fine for v1
            # (sparse expansion code is determined by topology, not intrinsic dynamics).
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        for action in ACTION_NAMES:
            # Per-action Purkinje pool. v1 uses FS-style preset for high
            # firing rate (PCs fire 30-80 Hz); proper HH_CEREBELLAR_PURKINJE
            # preset would be more accurate but requires HH dt scaling.
            regions.append(BrainRegion(
                name=f"purkinje_{action}",
                n_neurons=60,
                exc_fraction=0.0,  # PCs are GABAergic onto DCN (output is inhibitory)
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
            ))
            # Per-action DCN (AIP-equivalent). Tonic firing 40 Hz; PC inhibition
            # silences this pool, releasing the motor drive. exc_fraction=1.0
            # because DCN -> motor projection is excitatory.
            regions.append(BrainRegion(
                name=f"dcn_aip_{action}",
                n_neurons=30,
                exc_fraction=1.0,
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
        regions.append(BrainRegion(
            name="inferior_olive",
            n_neurons=20,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # ---- Pathways (cross-region projections) ----

    # Sensory → cortex (LEARNING site for perception, opt-in).
    # Plastic; agent learns position-to-action mapping via STDP + reward.
    # Each sensory neuron projects to all 4 cortex pools; learning shapes
    # which sensory patterns drive which cortex action pool.
    # Tagged with plasticity_gate="sensory_to_cortex" so curriculum can
    # stage perceptual learning: frozen during cortex warmup, thawed
    # during phase 2 to learn position→action mapping with the heuristic
    # as teacher (additive, not mutually exclusive).
    if enable_learned_perception:
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="sensory", to_region=f"cortex_{action}",
                density=1.0, weight_mean=sensory_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="sensory_to_cortex",
            ))

    # Hippocampus → cortex (LEARNING site, opt-in).
    # Plastic; agent learns (place, goal) → action via STDP + reward.
    # Place cells provide spatial context (where am I), goal cells provide
    # task context (where do I want to be). Together they should learn
    # the full position-action mapping.
    # Tagged with plasticity_gate="place_goal_to_cortex" so runners can
    # implement curriculum: freeze during cortex-warmup, thaw later.
    if enable_hippocampus:
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="sensor_place_readout", to_region=f"cortex_{action}",
                density=1.0, weight_mean=hippocampus_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="place_goal_to_cortex",
            ))
            pathways.append(RegionPathway(
                from_region="ppc_goal_input", to_region=f"cortex_{action}",
                density=1.0, weight_mean=hippocampus_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="place_goal_to_cortex",
            ))

    # Beacon perception pathway (Item 1 Stage 1 skeleton, 2026-04-27).
    # Beacon sensors → goal_cells: tagged plasticity_gate="beacon_to_goal"
    # for curriculum-staged learning. With curriculum, this pathway is frozen
    # during cortex warmup (when heuristic provides selectivity) and thawed
    # in phase 2 to learn beacon-pattern → goal-cell-position mapping.
    # NOTE: full trial-loop wiring (driving beacon_sensors based on beacon
    # position) is deferred to next session. Currently the region exists
    # but isn't driven, so enable_beacon_perception is a no-op until the
    # trial loop is updated.
    if enable_beacon_perception and enable_hippocampus:
        pathways.append(RegionPathway(
            from_region="beacon_sensors", to_region="ppc_goal_input",
            density=1.0, weight_mean=beacon_to_goal_weight,
            weight_jitter=0.2, plastic=True,
            plasticity_gate="beacon_to_goal",
        ))

    # Landmark → place cells pathway (Item 1 Stage 2, 2026-04-27).
    # Plastic; place cells self-organize from landmark sensor patterns.
    # Each unique (distance, bearing) to landmark gives a unique sensor
    # activation pattern, so place cells learn to fire at specific positions.
    if enable_landmarks and enable_hippocampus:
        pathways.append(RegionPathway(
            from_region="landmark_sensors", to_region="sensor_place_readout",
            density=1.0, weight_mean=landmark_to_place_weight,
            weight_jitter=0.2, plastic=True,
            plasticity_gate="landmark_to_place",
        ))

    # PFC working memory pathways (Item 3, 2026-04-27):
    #   goal_cells → PFC: goal info enters working memory
    #   PFC → cortex_X: PFC drives cortex selection across delays
    # Both tagged with plasticity_gate="dlpfc_wm_pathways" so curriculum can
    # stage PFC learning. Internal PFC connectivity is plastic_internal=True
    # for recurrent learning (gated by "dlpfc_wm_recurrent" if needed).
    if enable_pfc:
        if enable_hippocampus:
            # goal_cells → PFC for working memory of goal
            pathways.append(RegionPathway(
                from_region="ppc_goal_input", to_region="dlpfc_wm",
                density=0.5, weight_mean=goal_to_pfc_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="dlpfc_wm_pathways",
            ))
        # PFC → cortex (action selection driven by working memory)
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="dlpfc_wm", to_region=f"cortex_{action}",
                density=0.5, weight_mean=pfc_to_cortex_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="dlpfc_wm_pathways",
            ))

    # Cortex -> striatum (LEARNING site).
    # Each cortex_X projects strongly to its corresponding str_D1_X / str_D2_X
    # AND (if enable_bg_cross_projections) weakly to other actions' striatum.
    # Same-action paths are tagged with plasticity_gate="corticostriatal" so the
    # curriculum can freeze cortex→striatum once mature.
    # Cross-projections are tagged with plasticity_gate="corticostriatal_cross"
    # (separate gate, 2026-04-28) so the curriculum can stage them
    # independently — keep them frozen during phase 1+2 (don't accumulate
    # phase-0 motor bias), thaw post-goal-change in phase 3 so STDP+reward
    # can shape cross-action routing symmetrically.
    # Patch-matrix sparsity (2026-04-28, option 2): if cross_projection_density < 1.0,
    # randomly skip cross-pathways at build time to mirror real BG patch-matrix
    # anatomy (~10-25% cross-projection density). Selection is deterministic
    # given cross_projection_topology_seed so reruns reproduce the same topology.
    import random as _random
    _topology_rng = _random.Random(cross_projection_topology_seed)
    _all_cross_pairs = [(c, s) for c in ACTION_NAMES for s in ACTION_NAMES if c != s]
    _n_keep = max(0, int(round(len(_all_cross_pairs) * cross_projection_density)))
    _selected_cross = set(_topology_rng.sample(_all_cross_pairs, _n_keep))

    # R3.5 (2026-04-29): cortex->MSN density tightened to 0.20 (was 1.0)
    # per Bolam-2000 / Kincaid 1998 (catalog ref). At our scale (25 cortex
    # x 50 MSN per pool) density 0.20 ~ 5 cortex inputs per MSN, ~10 MSN
    # targets per cortex axon — matches "sparse + decorrelated" biological
    # convergence. Original density=1.0 was anatomically dense (every
    # cortex neuron synapsing every MSN). Re-tunable via runner kwarg
    # cortex_to_msn_density if needed; weight_mean kept at 25.0 to
    # maintain net excitatory drive given the sparser fan-in.
    cortex_to_msn_density_same = float(locals().get("cortex_to_msn_density_same_override", 0.20))
    cortex_to_msn_density_cross = 0.10  # sparser still per Bolam
    # R3.5 follow-up (2026-04-29 morning diagnostic): density 1.0 -> 0.20 reduced
    # cortex->MSN drive ~5x, which empirically silenced motor pools (1798/1800
    # trials all-zero motor counts at seed 42). To preserve effective drive
    # while honoring Bolam-2000 "few synapses per pair" biology, scale weight
    # inversely with density. Original (density=1.0, weight=25) -> default scaled
    # weight at density=0.20 is 25/0.2 = 125. Override via cortex_to_msn_weight_override
    # kwarg if needed.
    if cortex_to_msn_density_same < 1.0:
        # Scale weight to compensate density reduction. Original (density=1.0, weight=25)
        # gives 25 weight-units per cortex-MSN pair on average. After R3.5's density=0.20,
        # naive weight=25 gives 5 weight-units (5x weaker drive). Compensating gives
        # weight = 25 / density = 125 at density 0.20.
        cortex_to_msn_weight_same = 25.0 / cortex_to_msn_density_same
    else:
        cortex_to_msn_weight_same = 25.0
    cortex_to_msn_weight_same = float(locals().get("cortex_to_msn_weight_same_override", cortex_to_msn_weight_same))
    # When using sparse density (post-R3.5 default 0.20), scale weight to recover drive.
    # Setting cortex_to_msn_weight_same_override=25.0 reverts to the broken
    # weak-cascade behavior; setting density_same=1.0 reverts to pre-R3.5.
    # Cluster E v1 (2026-04-29): when topography is on, cortex_X -> str_D{1,2}_X
    # pathways carry distance_sigma so connections are Gaussian-weighted by
    # 2D corner distance. Same-action pairs share the same corner (distance=0,
    # full density); cross-action pairs are 1.0 unit apart (heavily attenuated
    # at sigma=0.3). Falls back to uniform Bernoulli when the flag is off.
    _cluster_e_sigma = (
        float(cluster_e_distance_sigma)
        if enable_cluster_e_topography
        else None
    )
    for cortex_action in ACTION_NAMES:
        for str_action in ACTION_NAMES:
            same = (cortex_action == str_action)
            if same:
                density = cortex_to_msn_density_same
                weight = cortex_to_msn_weight_same
                gate = "corticostriatal"
            elif enable_bg_cross_projections and (cortex_action, str_action) in _selected_cross:
                density = cortex_to_msn_density_cross
                weight = cross_projection_weight
                gate = "corticostriatal_cross"
            else:
                continue
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_D1_{str_action}",
                density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
                plasticity_gate=gate,
                distance_sigma=_cluster_e_sigma,
            ))
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_D2_{str_action}",
                density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
                plasticity_gate=gate,
                distance_sigma=_cluster_e_sigma,
            ))

    # v3 (2026-04-28): MSN cross-pool lateral inhibition.
    # Adds str_D1_X → str_D1_Y and str_D2_X → str_D2_Y for X != Y. MSNs are
    # GABAergic (exc_fraction=0.05), so these projections IS inhibitory —
    # firing in pool X suppresses firing in pool Y, sharpening action
    # selection. Real BG has GABAergic MSN collaterals plus FS interneurons
    # for stronger feed-forward inhibition. v3 covers the MSN-collateral
    # piece. FS interneurons + pallidal center-surround are v3.5 if needed.
    # Static (plastic=False): lateral inhibition is a structural feature.
    # 4 cortex actions × 3 cross targets × 2 (D1/D2) = 24 new pathways.
    if enable_bg_lateral_inhibition:
        for src_action in ACTION_NAMES:
            for dst_action in ACTION_NAMES:
                if src_action == dst_action:
                    continue
                for d_type in ("D1", "D2"):
                    pathways.append(RegionPathway(
                        from_region=f"str_{d_type}_{src_action}",
                        to_region=f"str_{d_type}_{dst_action}",
                        density=lateral_inhibition_density,
                        weight_mean=lateral_inhibition_weight,
                        weight_jitter=0.2,
                        plastic=False,
                    ))

    # Cluster B.2 (2026-04-28, R1.2 rewire 2026-04-29): striatal FSI pathways.
    # (a) cortex_X → str_PV_FSI_X (excitatory, dense, plastic=False, same-action only).
    #     FS pool gets driven only by its same-action cortex pool.
    # (b) str_PV_FSI_X → str_D{1,2}_Y for X != Y ONLY (cross-action feedforward
    #     inhibition; auto-derived inhibitory because str_FS regions have
    #     exc_fraction=0.0). 4 FS × 3 cross D-pool × 2 D-types = 24 paths.
    #
    # Biological grounding (Tepper-2018 pp 8–9; Tepper, Koós & Wilson, TK-2017
    # pp 161–163): paired-recording studies show MSN→MSN collaterals deliver
    # only ~0.5 mV unitary IPSPs at 14–25% connection probability with high
    # failure rates and short-term depression — i.e., MSN-MSN lateral
    # inhibition is functionally weak. By contrast, FSI→MSN feedforward
    # IPSPs are significantly larger and more reliable, and FSIs preferentially
    # innervate MSNs of OTHER action channels (cross-action). This makes the
    # FSI cross-action projection the dominant biological substrate for the
    # striatal WTA microcircuit. The previous (R1.1) within-action broadcast
    # was anatomically inaccurate; we now restrict FS_X to MSN_Y for Y != X.
    # The v3 `--bg-lateral-inhibition` MSN→MSN flag is now redundant with
    # this cross-action FSI WTA but is kept opt-in for backward compatibility.
    if enable_striatal_fsis:
        # (a) cortex_X → str_PV_FSI_X (excitatory drive, same-action)
        for cortex_action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_PV_FSI_{cortex_action}",
                density=1.0,
                weight_mean=cortex_to_str_fs_weight,
                weight_jitter=0.2,
                plastic=False,
            ))
        # (b) str_PV_FSI_X → str_D{1,2}_Y for X != Y only (cross-action WTA;
        # FSIs do NOT inhibit their own action's MSN pool).
        for fs_action in ACTION_NAMES:
            for str_action in ACTION_NAMES:
                if fs_action == str_action:
                    continue  # skip within-action — FSIs target other channels
                for d_type in ("D1", "D2"):
                    pathways.append(RegionPathway(
                        from_region=f"str_PV_FSI_{fs_action}",
                        to_region=f"str_{d_type}_{str_action}",
                        density=1.0,  # dense within-pool
                        weight_mean=str_fs_to_msn_weight,
                        weight_jitter=0.2,
                        plastic=False,
                    ))

    # Direct pathway: D1 -> GPi (inhibitory). Strong weight needed to overcome
    # GPi tonic firing (~30-75 Hz baseline).
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"str_D1_{action}", to_region=f"gpi_{action}",
            density=1.0, weight_mean=15.0, weight_jitter=0.2, plastic=False,
        ))

    # Indirect pathway: D2 -> GPe (PV+) -> STN -> GPi
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"str_D2_{action}", to_region=f"gpe_{action}",
            density=0.6, weight_mean=2.5, weight_jitter=0.2, plastic=False,
        ))
        pathways.append(RegionPathway(
            from_region=f"gpe_{action}", to_region="stn",
            density=0.3, weight_mean=1.5, weight_jitter=0.2, plastic=False,
        ))

    # R3.7 (2026-04-29): arkypallidal (PV-) GPe subpool. D2 also drives
    # arky cells; arky projects back to striatal FSIs broadcasting a
    # "stop signal" (Mallet 2012). Per Kita 2007 / Tepper-2018, PV-
    # cells rarely collateralize to STN/GPi -- their canonical target
    # is the striatum. Modeling as broadcast to all str_PV_FSI_Y so a single
    # action's D2 activation can feedback-inhibit the entire striatal
    # FSI population, halting ongoing motor commitments.
    if enable_striatal_fsis:  # arky->FSI requires FSI population
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"str_D2_{action}", to_region=f"gpe_arky_{action}",
                density=0.5, weight_mean=2.0, weight_jitter=0.2, plastic=False,
            ))
            for fs_action in ACTION_NAMES:
                pathways.append(RegionPathway(
                    from_region=f"gpe_arky_{action}", to_region=f"str_PV_FSI_{fs_action}",
                    density=0.3, weight_mean=1.5, weight_jitter=0.2, plastic=False,
                ))
    else:
        # Without FSI population, arky has no striatal target. Still
        # receive D2 input so dynamics are correct; outputs are dropped.
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"str_D2_{action}", to_region=f"gpe_arky_{action}",
                density=0.5, weight_mean=2.0, weight_jitter=0.2, plastic=False,
            ))

    # STN -> all GPi (diffuse excitation; this is the "hyperdirect"-like
    # contribution that biases against premature action selection)
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="stn", to_region=f"gpi_{action}",
            density=0.4, weight_mean=1.0, weight_jitter=0.2, plastic=False,
        ))

    # GPi -> thalamus (inhibitory). Strong weight + density needed so
    # GPi tonic firing fully suppresses thal, AND so D1-mediated GPi
    # silence cleanly releases the gate.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"gpi_{action}", to_region=f"thal_{action}",
            density=1.0, weight_mean=8.0, weight_jitter=0.2, plastic=False,
        ))

    # R3.11 (2026-04-29): striosome (patch) pathways.
    # cortex_X -> str_striosome_X: placeholder for limbic input (vmPFC/amygdala/
    # ventral hippocampus per PBR-160 ch 9). Plastic so patch can learn
    # cortical-to-patch mapping. Same density as matrix per Bolam.
    # str_striosome_X -> snc: canonical striosome->SNc projection driving
    # phasic DA (Tepper & Lee PBR-160 ch 11 p 191).
    # str_striosome_X -> gpi_X: secondary striosome->SNr projection (PBR-160
    # ch 9 Deniau p 160 — striosomes contribute substantial direct input
    # to SNr in addition to the canonical SNc target). Smaller weight
    # than matrix's str_D1->gpi to reflect minor contribution.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"cortex_{action}", to_region=f"str_striosome_{action}",
            density=cortex_to_msn_density_same, weight_mean=cortex_to_msn_weight_same,
            weight_jitter=0.2, plastic=True, plasticity_gate="corticostriatal",
        ))
        pathways.append(RegionPathway(
            from_region=f"str_striosome_{action}", to_region="snc",
            density=0.4, weight_mean=2.5, weight_jitter=0.2, plastic=False,
        ))
        pathways.append(RegionPathway(
            from_region=f"str_striosome_{action}", to_region=f"gpi_{action}",
            density=0.3, weight_mean=1.5, weight_jitter=0.2, plastic=False,
        ))

    # Spiking-SNc actor-critic Stage B (2026-06-08): the neural value critic's
    # two pathways. (1) PERCEIVED STATE -> critic (plastic, gate "value_input"),
    # trained by the SNc-derived dopamine delta via the existing three-factor
    # pipeline so the critic LEARNS V(s). (2) critic -> SNc, GABAergic, routed
    # through the slow GABA_B/GIRK K+ conductance (receptor="gaba_b", E_K=-90mV)
    # so V is SUBTRACTED at the SNc membrane (the host _V_scaffold term is then
    # dropped in the reward block). See the Stage-B research doc Option A.
    #
    # AFFERENT RE-POINTED 2026-06-08 (redesign per
    # 2026-06-08-nav-neural-value-critic-redesign-research.md): the original
    # afferent was the ventral object code `cortex_it` — which is (a) the
    # position-INVARIANT "what" stream (cannot encode a value-of-LOCATION) and
    # (b) NEVER fires in nav (it_mean=0 over 16k steps; the smoke NEGATIVE).
    # The biology is unambiguous (catalog C.30 / B.07; Houk-Adams-Barto 1995;
    # Lansink 2009; van der Meer & Redish 2009): the striosome/patch critic for
    # a SPATIAL value reads the hippocampal PLACE code (dorsal "where" stream)
    # via the hippocampus -> ventral-striatum projection, NOT IT. The faithful +
    # ACTIVE + position-SENSITIVE afferent already in this runner is
    # `sensor_place_readout` (the Gaussian place-cell readout driven every nav
    # step from the agent's (x,y)), enabled by --enable-place-goal-readout.
    # Anti-cheat: the afferent is a perceived-position POPULATION code (a place
    # code, not a coordinate handed to a formula); it must NOT be a
    # coordinate/goal-cell region.
    if enable_neural_critic and neural_place_selforg:
        # ═══ N9 NEURAL PLACE-CODE SELF-ORG pathways (mirror the de-risk _build). The afferent is
        #     the self-organized spiking `place` pool (NOT a host Gaussian). Anti-cheat provenance:
        #     place fires ONLY from place_sensors (an egocentric POPULATION sense), never a
        #     coordinate/goal-cell region. ═══
        _critic_afferent = "place"
        # (1) place_sensors -> place : PLASTIC competitive (the Hartley-Burgess self-org pathway,
        #     gate `landmark_to_place`; opened during STEP-1 self-org, then FROZEN).
        pathways.append(RegionPathway(
            from_region="place_sensors", to_region="place",
            density=float(place_sensors_to_place_density),
            weight_mean=float(place_sensors_to_place_weight),
            weight_jitter=float(place_sensors_to_place_jitter), plastic=True,
            plasticity_gate="landmark_to_place",
        ))
        # (2) FS-PING reciprocal: place -> place_fs (excite the FS) + place_fs -> place (GABA_A,
        #     transmission_gate `place_fs_gate` so it can be held CLOSED during self-org for clean
        #     threshold-WTA -> sparse DISTINCT fields, and OPENED for the volley read-out/nav).
        pathways.append(RegionPathway(
            from_region="place", to_region="place_fs",
            density=float(place_fs_density), weight_mean=float(place_fs_weight),
            weight_jitter=0.2, plastic=False,
        ))
        pathways.append(RegionPathway(
            from_region="place_fs", to_region="place",
            density=float(fs_to_place_density), weight_mean=float(fs_to_place_weight),
            weight_jitter=0.2, plastic=False,
            transmission_gate="place_fs_gate",
        ))
        # (2b) FS-PING -> CRITIC feedforward inhibition (2026-06-10, the SPIKING root fix for the
        #      over-firing critic, research 2026-06-10-N9-spiking-reward-and-critic-normalization).
        #      The MSN-D1 critic over-fires ~125 Hz on hot place-code draws (unphysiological; MSNs
        #      fire 1-20 Hz) -> it over-clamps the SNc -> binary delta. The biological brake is
        #      perisomatic FS-PV feedforward inhibition (catalog B.06, Lee 2017): `place_fs` already
        #      gamma-synchronizes the volley and SCALES with the volley size, so a hotter draw recruits
        #      MORE FS -> MORE inhibition -> a DIVISIVE-leaning clamp that holds the critic in a
        #      physiological rate band across draws (a SPIKING normalization, not the GIRK-cap masking
        #      at the SNc). GABA_A (place_fs is inhibitory, exc_fraction=0). Grading of V is carried by
        #      the WEIGHTED coincidence plateau (read-out), NOT by dividing the all-or-none plateau
        #      (the honest hybrid: FS clamps the rate, the weighted plateau grades the value). Default
        #      OFF => byte-equivalent. Held open via `critic_fs_gate` (always-on default).
        if enable_critic_fs_inhibition:
            pathways.append(RegionPathway(
                from_region="place_fs", to_region="striosome_value",
                density=float(critic_fs_density), weight_mean=float(critic_fs_weight),
                weight_jitter=0.2, plastic=False,
                transmission_gate="critic_fs_gate",
            ))
        # (3) place -> striosome_value : PLASTIC, DA-delta-gated (gate `value_input`), Route-D
        #     coincidence_detector so the FS-PING-synchronized volley fires the MSN critic that the
        #     sparse-async code cannot. STILL plastic + DA-gated so it GRADES + LEARNS V.
        pathways.append(RegionPathway(
            from_region="place", to_region="striosome_value",
            density=float(vs_place_to_value_density),
            weight_mean=float(vs_place_to_value_weight),
            weight_jitter=float(0.2), plastic=True, plasticity_gate="value_input",
            coincidence_detector=True,
        ))
    elif enable_neural_critic:
        # 2026-06-09 VALIDATED redesign: the critic afferent is the DEDICATED DENSE
        # `vs_place_context` (built above), drive-injected each nav step with the agent's
        # perceived (x,y) as a grid-32 Gaussian place code. This is the dorsal 'where' /
        # hippocampal place stream biology uses for spatial value (Houk-Adams-Barto 1995,
        # Lansink 2009), but a DEDICATED dense version (not the SPARSE actor
        # `sensor_place_readout` ~1-3 cells, which provably can't fire the MSN critic at ANY
        # weight — the 2026-06-08 calibration NEGATIVE). The dense afferent is self-contained
        # (its own region + drive injection), so --enable-place-goal-readout is no longer a
        # hard requirement for the critic itself (the flagship still enables it for the actor).
        # Anti-cheat provenance: vs_place_context is a perceived-POSITION population code (a
        # place code rendered from (x,y), NOT a coordinate handed to a formula), never a
        # coordinate/goal-cell region.
        _critic_afferent = "vs_place_context"
        assert _critic_afferent not in ("goal_cells", "ppc_goal_input"), (
            "neural-critic anti-cheat: the value critic must read the perceived "
            "place code, not a coordinate/goal-cell region; got "
            f"{_critic_afferent!r}."
        )
        if enable_convergent_upstate:
            # A1 — UP-STATE arm: dense, NON-plastic convergent excitation (the B.02 pre-wired
            # corticostriatal up-state drive). Many weak per-synapse weights summing PAST the
            # ~339 pA rheobase at the goal (NOT one giant synapse — the convergence is via
            # density x n_presynaptic, the per-synapse weight stays moderate). plastic=False so it
            # does NOT learn (it is the innate up-state, escaping the bootstrap structurally).
            pathways.append(RegionPathway(
                from_region="vs_place_drive", to_region="striosome_value",
                density=float(vs_place_drive_to_value_density),
                weight_mean=float(vs_place_drive_to_value_weight),
                weight_jitter=0.5, plastic=False,
            ))
        # A2 (or the sole afferent when enable_convergent_upstate=False) — the PLASTIC value
        # learner. DA-delta-gated STDP sculpts V(s) on top of the (A1-fired, when enabled) cell.
        pathways.append(RegionPathway(
            from_region=_critic_afferent, to_region="striosome_value",
            density=float(vs_place_to_value_density),
            weight_mean=float(vs_place_to_value_weight),
            weight_jitter=0.5, plastic=True, plasticity_gate="value_input",
        ))
    if enable_neural_critic:
        # critic -> SNc GABA_B subtraction. transmission_gate="critic_snc_window"
        # lets the runner OPEN this route only for a ~1-tau LEAD window into the
        # reward evaluation (the de-risk's value-leads-reward constraint:
        # d0416fc3 — the GABA_B must pre-build ~100-150 ms BEFORE reward and must
        # NOT integrate across the whole dwell, else the far-V also gets canceled
        # / the SNc flatlines). Default gate value is 1.0 (always-on, additive,
        # zero overhead) so this is byte-equivalent if the runner never sets it.
        pathways.append(RegionPathway(
            from_region="striosome_value", to_region="snc",
            density=float(critic_value_to_snc_density),
            weight_mean=float(critic_value_to_snc_weight),
            weight_jitter=0.2, plastic=False,
            receptor="gaba_b",   # slow GIRK K+ subtraction onto the depolarized SNc
            transmission_gate="critic_snc_window",
        ))

    # reward_us -> snc : the EXCITATORY US/reward afferent (PPN->VTA glutamate, C.33). When
    # spiking_reward_us, the SNc reward burst (the `r` term) is produced by `reward_us` FIRING into
    # the SNc, NOT a host current write -> δ=r−V is fully synaptic (r=excitation, V=GABA_B). plastic=
    # False (innate US->DA reflex). weight tuned so a full US volley ~= the old snc_reward_gain drive.
    if spiking_reward_us:
        pathways.append(RegionPathway(
            from_region="reward_us", to_region="snc",
            density=float(reward_us_to_snc_density),
            weight_mean=float(reward_us_to_snc_weight),
            weight_jitter=0.2, plastic=False,
        ))

    # R3.10 (2026-04-29): GPi/SNr -> snc collateral disinhibition
    # (PBR-160 ch 11 Tepper & Lee pp 192-193, 199; Tepper et al. 1995).
    # SNr GABA neurons project to SNc DA neurons via axon collaterals;
    # the major in-vivo drive of spontaneous DA burst firing is the
    # SNr -> SNc disinhibition (when D1-mediated SNr silencing releases
    # tonic GABA suppression of DA cells, DA neurons burst). Combined
    # with R1.1 (E_inh = -55 mV on the snc region, since SNc lacks KCC2),
    # this gives a biologically grounded substrate for phasic DA without
    # external injection. NOTE: in our cascade we conflate SNr with GPi
    # (both GABAergic BG output nuclei); this is the standard rodent vs
    # primate naming difference rather than a separate population.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"gpi_{action}", to_region="snc",
            density=0.3, weight_mean=2.0, weight_jitter=0.2, plastic=False,
        ))

    # Thalamus -> motor cortex (excitatory). Very strong weight needed
    # because thal pool is small (10 cells) and we need ~50 Hz motor output
    # from ~24 Hz thal input.
    if not enable_distributed_motor_pop:
        # Default: per-action labeled-line pathway.
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"thal_{action}", to_region=f"motor_{action}",
                density=1.0, weight_mean=20.0, weight_jitter=0.2, plastic=False,
            ))
    else:
        # Distributed motor pool: cosine-tuned thal_X -> motor_pop_θ.
        # Each thal_X drives all 8 motor_pop sub-pools but with weight scaled
        # by max(0, cos(θ_X - θ)). Adjacent sub-pools (45° away) get 0.707x;
        # perpendicular (90°) get 0; opposite get 0 (cosine clamped negative).
        import math as _math
        ACTION_THETA_DEG = {"N": 90, "E": 0, "S": 270, "W": 180}
        SUBPOOL_THETA = [
            (0, "E"), (45, "NE"), (90, "N"), (135, "NW"),
            (180, "W"), (225, "SW"), (270, "S"), (315, "SE"),
        ]
        for action in ACTION_NAMES:
            theta_x = ACTION_THETA_DEG[action]
            for theta_y, suffix in SUBPOOL_THETA:
                # Angular distance (signed) — wrap to [-180, 180]
                d = ((theta_x - theta_y + 180) % 360) - 180
                cos_w = _math.cos(_math.radians(d))
                if cos_w <= 0.01:  # Skip pathways with negligible weight
                    continue
                # Strong base weight (20.0) scaled by cosine tuning.
                pathways.append(RegionPathway(
                    from_region=f"thal_{action}",
                    to_region=f"motor_pop_{suffix}",
                    density=1.0,
                    weight_mean=20.0 * cos_w,
                    weight_jitter=0.2,
                    plastic=False,
                ))

    # Cluster A (2026-04-29): closed BG loop.
    # (a) Hyperdirect pathway: cortex_X -> stn (Nambu 2002). ~30% of cortex
    #     pyramids project directly to STN, bypassing striatum. Sparse
    #     excitatory drive provides a fast global "stop" signal that
    #     biases against premature action commitment when multiple
    #     cortex pools fire simultaneously. Static (plastic=False) since
    #     anatomical projection is genetically specified, not learned.
    # (b) Thalamo-cortical feedback: thal_X -> cortex_X. Closes the
    #     cortex -> BG -> thal -> cortex loop. Action-specific (not
    #     cross-action) per VA/VL topographic organization. Provides the
    #     post-synaptic activity that lets STDP shape useful cross-action
    #     weights (the "teaching signal" missing for cross-projection
    #     learning per CLAUDE.md cheat-5 reframe). Static.
    if enable_cluster_a_closed_loop:
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"cortex_{action}", to_region="stn",
                density=0.10, weight_mean=3.0, weight_jitter=0.2,
                plastic=False,
            ))
            pathways.append(RegionPathway(
                from_region=f"thal_{action}", to_region=f"cortex_{action}",
                density=0.50, weight_mean=5.0, weight_jitter=0.2,
                plastic=False,
            ))

    # ---- Motor lateral inhibition (opt-in) ----
    # FS interneuron sub-pool per motor pool. Each motor_X drives its own
    # motor_FS_X (excitatory), which in turn inhibits the other 3 motor pools.
    # This implements the cortical WTA microcircuit: when motor_X fires,
    # motor_FS_X fires, suppressing motor_{Y,Z,W}. Combined with BG gating,
    # this should sharpen action selection in cases where multiple cortex
    # pools drive simultaneously (currently the dominant random-fallback case).
    if enable_motor_lateral_inhibition:
        for action_idx, action in enumerate(ACTION_NAMES):
            regions.append(BrainRegion(
                name=f"motor_FS_{action}",
                n_neurons=n_motor_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory → outgoing synapses are inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                action_index=action_idx,
            ))

        # motor_X → motor_FS_X (excitatory drive — motor's own activity drives its FS)
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"motor_{action}", to_region=f"motor_FS_{action}",
                density=1.0, weight_mean=motor_to_fs_weight, weight_jitter=0.2,
                plastic=False,
            ))

        # motor_FS_X → motor_Y for Y != X (inhibitory cross-pool suppression)
        for src_action in ACTION_NAMES:
            for tgt_action in ACTION_NAMES:
                if src_action == tgt_action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"motor_FS_{src_action}", to_region=f"motor_{tgt_action}",
                    density=1.0, weight_mean=fs_to_motor_weight, weight_jitter=0.2,
                    plastic=False,
                ))

    # ---- Thalamic reticular-nucleus (TRN) lateral inhibition (opt-in,
    #      2026-06-06, N8+N6) ----
    # A biological WTA on the thalamic RELAY (the cleanest genuine-disinhibition
    # selection signal). Under genuine GPi->thal disinhibition + a plastic
    # multi-goal run, cumulative D1 plasticity partially releases several thal
    # pools at once -> the readout_source="thal" argmax ties. The thalamic
    # reticular nucleus (TRN) provides reciprocal GABAergic inhibition between
    # relay nuclei (Pinault 2004; Crabtree 2018; Halassa 2017) — it sharpens the
    # released winner and silences the leaked losers, so the readout sees ONE
    # clean winner. Same microcircuit shape as the motor WTA, applied one stage
    # upstream where the signal is strong. Combine with --readout-source thal.
    if enable_thal_lateral_inhibition:
        for action_idx, action in enumerate(ACTION_NAMES):
            regions.append(BrainRegion(
                name=f"thal_FS_{action}",
                n_neurons=n_thal_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory (TRN is GABAergic)
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                action_index=action_idx,
            ))
        # thal_X → thal_FS_X (excitatory: relay collaterals drive TRN)
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"thal_{action}", to_region=f"thal_FS_{action}",
                density=1.0, weight_mean=thal_to_fs_weight, weight_jitter=0.2,
                plastic=False,
            ))
        # thal_FS_X → thal_Y for Y != X (inhibitory cross-pool suppression)
        for src_action in ACTION_NAMES:
            for tgt_action in ACTION_NAMES:
                if src_action == tgt_action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"thal_FS_{src_action}", to_region=f"thal_{tgt_action}",
                    density=1.0, weight_mean=thal_fs_to_thal_weight, weight_jitter=0.2,
                    plastic=False,
                ))

    # ---- Spiking action-selection WTA readout (opt-in, 2026-06-06, N6) ----
    # A DEDICATED, READ-ONLY selection layer that biologizes the host argmax.
    # The N6 residual was that action selection, even reading the clean
    # thalamus, was still a host-side argmax over raw rates. Here the decision
    # instead EMERGES from a spiking competition:
    #   thal_X --(thal_to_sel_weight, exc, feed-forward)--> sel_X
    #   sel_X  --(sel_to_sel_fs_weight, exc)--------------> sel_FS_X
    #   sel_FS_X --(sel_fs_to_sel_weight, inh)------------> sel_Y (Y != X)
    # The selected action's thal (the cleanest genuine-disinhibition signal,
    # only that pool released) drives its sel_X decisively above threshold;
    # sel_X recruits sel_FS_X, which silences the other three sel pools — a
    # cortical soft-WTA (Douglas-Martin 2004; Rutishauser-Douglas-Slotine
    # 2011). The host then OBSERVES which sel_X fired (the winner of the
    # competition), not an argmax over rates.
    #
    # Why this differs from enable_thal_lateral_inhibition (which scored 20.0):
    # that put the competition ON the thalamic relay (thal_FS_X -| thal_Y),
    # corrupting the SAME thal_X signal that drives the thal->motor cascade and
    # navigation. The sel layer is a pure readout tap — it reads thal but never
    # projects back, so the forward dynamics are byte-identical to thal-readout.
    # Why it differs from the motor WTA (14.7): that drove the competition from
    # the WEAK motor counts (one synapse past the clean thal); here the
    # competition is driven by the STRONG clean thalamus.
    if enable_spiking_wta_readout:
        for action_idx, action in enumerate(ACTION_NAMES):
            # ACCUMULATE stage. Excitatory selection pool (the competitor whose
            # ramping firing = accumulated evidence). NMDA-SLOW recurrent
            # self-excitation (sel_recurrent_density / sel_recurrent_weight)
            # turns the gain-0 passive comparator into a Wang-2002 amplifying
            # integrator: a small consistent thal drive is re-excited + integrated
            # over the readout window to a committed bound. enable_nmda=True puts
            # this slice (and only this slice, unless --enable-pfc-nmda is also on)
            # in the bridge's cp_nmda_neuron_mask so the recurrence is NMDA-slow
            # (tau_decay=100ms), the biological integration constant. Soft-WTA
            # gain alpha<1 (Rutishauser-Douglas-Slotine 2011) — tuned to ramp/hold
            # but not self-ignite without thalamic drive.
            regions.append(BrainRegion(
                name=f"sel_{action}",
                n_neurons=n_sel_per_action,
                exc_fraction=1.0,
                internal_density=sel_recurrent_density,
                exc_weight_mean=sel_recurrent_weight, inh_weight_mean=0.0,
                weight_jitter=0.2, plastic_internal=False,
                enable_nmda=True,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                action_index=action_idx,
            ))
            # Inhibitory interneuron (GABAergic, mediates structured cross-pool
            # WTA competition — Rutishauser selective inhibition, not a symmetric
            # blanket: sel_FS_X is driven only by sel_X and inhibits only sel_Y!=X).
            regions.append(BrainRegion(
                name=f"sel_FS_{action}",
                n_neurons=n_sel_fs_per_action,
                exc_fraction=0.0,  # all-inhibitory → outgoing synapses are inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                action_index=action_idx,
            ))
            if enable_commit_burst:
                # COMMIT stage. Burst pool (SC / saccade-generator EBN analogue,
                # H.24/H.25). Held silent by the tonic commit_OPN; fires ALL-OR-
                # NONE only when sel_X ramps past threshold (Lo-Wang 2006 SC
                # threshold; Stine-Shadlen 2023). Its own recurrence regenerates
                # the burst once triggered (decisive commit, not a graded rate).
                regions.append(BrainRegion(
                    name=f"commit_{action}",
                    n_neurons=n_commit_per_action,
                    exc_fraction=1.0,
                    internal_density=commit_recurrent_density,
                    exc_weight_mean=commit_recurrent_weight, inh_weight_mean=0.0,
                    weight_jitter=0.2, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                    action_index=action_idx,
                ))
        if enable_commit_burst:
            # Shared omnipause pool (OPN, H.24). Tonically driven (commit_opn_
            # tonic_pA set on cp_external_input_current at setup) → fires
            # continuously → inhibits every commit_X equally, holding them all
            # below threshold until a sel_X accumulator wins.
            regions.append(BrainRegion(
                name="commit_OPN",
                n_neurons=n_commit_opn,
                exc_fraction=0.0,  # all-inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
            ))
        # thal_X → sel_X (excitatory feed-forward: the clean relay drives its
        # selection pool). READ-ONLY: there is no sel_X → thal projection.
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"thal_{action}", to_region=f"sel_{action}",
                density=1.0, weight_mean=thal_to_sel_weight, weight_jitter=0.2,
                plastic=False,
            ))
        # sel_X → sel_FS_X (excitatory: a winning sel pool recruits its
        # interneuron, which then suppresses the losers).
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"sel_{action}", to_region=f"sel_FS_{action}",
                density=1.0, weight_mean=sel_to_sel_fs_weight, weight_jitter=0.2,
                plastic=False,
            ))
        # sel_FS_X → sel_Y for Y != X (inhibitory cross-pool suppression).
        for src_action in ACTION_NAMES:
            for tgt_action in ACTION_NAMES:
                if src_action == tgt_action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"sel_FS_{src_action}", to_region=f"sel_{tgt_action}",
                    density=1.0, weight_mean=sel_fs_to_sel_weight, weight_jitter=0.2,
                    plastic=False,
                ))
        if enable_commit_burst:
            # sel_X → commit_X (excitatory: the accumulator drives its burst pool;
            # weight tuned so only a RAMPED sel_X overcomes the OPN tonic inhibition).
            for action in ACTION_NAMES:
                pathways.append(RegionPathway(
                    from_region=f"sel_{action}", to_region=f"commit_{action}",
                    density=1.0, weight_mean=sel_to_commit_weight, weight_jitter=0.2,
                    plastic=False,
                ))
            # commit_OPN → commit_X (inhibitory: tonic omnipause suppression of all
            # burst pools until the accumulator wins).
            for action in ACTION_NAMES:
                pathways.append(RegionPathway(
                    from_region="commit_OPN", to_region=f"commit_{action}",
                    density=1.0, weight_mean=opn_to_commit_weight, weight_jitter=0.2,
                    plastic=False,
                ))

    # ---- Motor cross-coupling (opt-in, 2026-05-02) ----
    # Models distributed/overlapping somatotopy in M1 (Penfield 1937 fuzzy
    # boundaries; Pulvermüller 1999 distributed action-word neurons).
    # Adds excitatory connections between motor pools at adjacent cardinal
    # directions (90° angular distance: N↔E, E↔S, S↔W, W↔N).
    # Opposite directions (N↔S, E↔W at 180°) get NO coupling — they're
    # antagonistic, like agonist/antagonist muscle pairs in real motor
    # control.
    # Hypothesis: rigid 4-pool argmax architecture is the bottleneck for
    # text I/O accuracy. Softer cross-pool tuning lets motor population
    # encode direction more like real M1 (continuous tuning curves) than
    # discrete labeled lines.
    if enable_motor_cross_coupling:
        # Adjacent direction pairs (90° apart). N↔E adjacent, etc.
        ADJACENT_PAIRS = [
            ("N", "E"), ("E", "N"),
            ("E", "S"), ("S", "E"),
            ("S", "W"), ("W", "S"),
            ("W", "N"), ("N", "W"),
        ]
        for src, tgt in ADJACENT_PAIRS:
            pathways.append(RegionPathway(
                from_region=f"motor_{src}", to_region=f"motor_{tgt}",
                density=motor_cross_coupling_density,
                weight_mean=motor_cross_coupling_weight,
                weight_jitter=0.2,
                plastic=False,  # Static — represents inherited tuning structure
            ))

    # Cortex WTA pathways (opt-in). Mirror of motor WTA structure.
    if enable_cortex_lateral_inhibition:
        # cortex_X → cortex_FS_X (excitatory: cortex pool drives its FS)
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"cortex_{action}", to_region=f"cortex_FS_{action}",
                density=1.0, weight_mean=cortex_to_fs_weight, weight_jitter=0.2,
                plastic=False,
            ))
        # cortex_FS_X → cortex_Y for Y != X (inhibitory: FS suppresses other pools)
        for src_action in ACTION_NAMES:
            for tgt_action in ACTION_NAMES:
                if src_action == tgt_action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"cortex_FS_{src_action}", to_region=f"cortex_{tgt_action}",
                    density=1.0, weight_mean=fs_to_cortex_weight, weight_jitter=0.2,
                    plastic=False,
                ))

    # ---- Cluster D v1 (2026-04-29): hippocampus trisynaptic loop pathways ----
    # See docs/plans/2026-04-29-cluster-d-hippocampus-design.md.
    # Pathways added when --enable-cluster-d-hippocampus is on:
    #   sensory -> ec (perceptual entry; only if --learned-perception)
    #   landmark_sensors -> ec (only if --landmarks; landmark_sensors region
    #     only exists in that case)
    #   ec -> dg (perforant path; main excitatory drive to DG)
    #   ec -> dg_pv_basket (FFi recruitment)
    #   dg_pv_basket -> dg (strong feedforward inhibition for sparsity)
    #   ec -> ca1 (direct cortical bypass)
    #   dg -> ca3 (mossy fibers; sparse but strong)
    #   ca3 -> ca3 (recurrent autoassociator — handled by region.internal_density)
    #   ca3 -> ca1 (Schaffer collaterals)
    #   ca1 -> place_cells (readout; only if --hippocampus, since place_cells
    #     region only exists then; coexists with landmark_sensors->place_cells)
    if enable_cluster_d_hippocampus:
        # sensory -> ec (only when learned-perception layer exists)
        if enable_learned_perception:
            pathways.append(RegionPathway(
                from_region="sensory", to_region="ec",
                density=0.40, weight_mean=4.0, weight_jitter=0.2,
                plastic=True, plasticity_gate="sensory_to_ec",
            ))
        # landmark_sensors -> ec (only when landmark_sensors region exists)
        if enable_landmarks:
            pathways.append(RegionPathway(
                from_region="landmark_sensors", to_region="ec",
                density=0.40, weight_mean=4.0, weight_jitter=0.2,
                plastic=True, plasticity_gate="sensory_to_ec",
            ))
        # ec -> dg (perforant path)
        pathways.append(RegionPathway(
            from_region="ec", to_region="dg",
            density=0.40, weight_mean=6.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ec_to_dg",
        ))
        # ec -> dg_pv_basket (FFi recruitment, static)
        pathways.append(RegionPathway(
            from_region="ec", to_region="dg_pv_basket",
            density=0.40, weight_mean=5.0, weight_jitter=0.2,
            plastic=False,
        ))
        # dg_pv_basket -> dg (strong feedforward inhibition; static)
        pathways.append(RegionPathway(
            from_region="dg_pv_basket", to_region="dg",
            density=1.00, weight_mean=6.0, weight_jitter=0.2,
            plastic=False,
        ))
        # ec -> ca1 (direct cortical bypass)
        pathways.append(RegionPathway(
            from_region="ec", to_region="ca1",
            density=0.30, weight_mean=3.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ec_to_ca1",
        ))
        # dg -> ca3 (mossy fibers)
        pathways.append(RegionPathway(
            from_region="dg", to_region="ca3",
            density=0.10, weight_mean=8.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="dg_to_ca3",
        ))
        # ca3 -> ca3 recurrent: by default handled via ca3
        # region.internal_density=0.30. With v2 on, the ca3 region's
        # internal_density was zeroed above and we add an explicit
        # plastic self-pathway here, gated so the runner can flip
        # plasticity on only during ripple-burst windows.
        if enable_cluster_d_v2_swr:
            pathways.append(RegionPathway(
                from_region="ca3", to_region="ca3",
                density=0.30, weight_mean=1.5, weight_jitter=0.2,
                plastic=True, plasticity_gate="ca3_swr_burst",
            ))
        # ca3 -> ca1 (Schaffer collaterals)
        pathways.append(RegionPathway(
            from_region="ca3", to_region="ca1",
            density=0.30, weight_mean=4.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ca3_to_ca1",
        ))
        # ca1 -> place_cells: only when --hippocampus is on (place_cells region
        # only exists in that case). Coexists with landmark_sensors->place_cells.
        if enable_hippocampus:
            pathways.append(RegionPathway(
                from_region="ca1", to_region="sensor_place_readout",
                density=0.50, weight_mean=5.0, weight_jitter=0.2,
                plastic=False,
            ))

    # Cluster F v1 pathways (2026-04-29). Marr-Albus forward path + IO teaching.
    # Total: ~25 pathways across the cerebellar microcircuit.
    if enable_cluster_f_cerebellum:
        # State input -> mossy_state. Drive mossy fibers from existing place /
        # goal-vector regions when available; fall back to cortex_X if neither
        # plastic-perception flag is on. v1 uses a simple union of available
        # state-bearing sources to keep the cerebellum learning regardless
        # of which other clusters are enabled.
        _state_sources = []
        if enable_hippocampus:
            _state_sources.append("sensor_place_readout")
            _state_sources.append("ppc_goal_input")
        if enable_learned_perception:
            _state_sources.append("sensory")
        if not _state_sources:
            # Bare-cerebellum mode (no other input flags): pull from cortex
            # pools as proxy state; not biologically pure but lets the
            # cerebellum still receive SOMETHING during smoke tests.
            for action in ACTION_NAMES:
                _state_sources.append(f"cortex_{action}")
        for src in _state_sources:
            pathways.append(RegionPathway(
                from_region=src, to_region="mossy_state",
                density=0.5, weight_mean=4.0, weight_jitter=0.2,
                plastic=False,
            ))
        # mossy_state -> granule: sparse expansion (Marr's codon coding).
        # Density 0.05 means each granule receives ~3 mossy inputs (matches
        # Marr's "4-5 claws per granule" prediction).
        pathways.append(RegionPathway(
            from_region="mossy_state", to_region="granule",
            density=0.05, weight_mean=8.0, weight_jitter=0.2,
            plastic=False,
        ))
        # granule -> purkinje_X (parallel fiber, all-to-all density 0.30,
        # plastic). THIS IS THE LEARNING SITE. v1 uses reward-modulated STDP
        # via the existing infrastructure, tagged with "cerebellum_pf_pc"
        # gate so curriculum can stage cerebellar learning. Initial weight
        # 1.0 is small so PCs aren't dominated by PF drive at start; learning
        # shapes which granule patterns drive which PC pool.
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="granule", to_region=f"purkinje_{action}",
                density=0.30, weight_mean=1.0, weight_jitter=0.3,
                plastic=True, plasticity_gate="cerebellum_pf_pc",
            ))
        # purkinje_X -> dcn_aip_X (same-action only, INHIBITORY; PCs are
        # GABAergic). High weight (15.0) so PC firing strongly silences DCN.
        # plastic=False in v1 (Mauk's two-site plasticity deferred to v2).
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"purkinje_{action}", to_region=f"dcn_aip_{action}",
                density=0.5, weight_mean=15.0, weight_jitter=0.2,
                plastic=False,
            ))
        # dcn_aip_X -> motor_X (same-action only, EXCITATORY; additive
        # contribution alongside thal_X drive). Weight 8.0 keeps the
        # cerebellar contribution comparable to the BG drive without
        # overwhelming it. plastic=False.
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"dcn_aip_{action}", to_region=f"motor_{action}",
                density=0.3, weight_mean=8.0, weight_jitter=0.2,
                plastic=False,
            ))
        # inferior_olive -> purkinje_X (climbing fiber, sparse 1:few; v1
        # doesn't model the strict 1:1 PC:CF ratio). High weight (50.0) so
        # each CF event evokes a strong PC complex spike. v1 uses the
        # existing reward-modulation path: when the runner injects current
        # to inferior_olive on a Δd>0 step, IO neurons fire, the resulting
        # CF + recent PF coactivation registers in the eligibility trace,
        # and a negative reward signal at that moment yields LTD-like
        # weight changes on the active PF→PC synapses.
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="inferior_olive", to_region=f"purkinje_{action}",
                density=0.05, weight_mean=50.0, weight_jitter=0.2,
                plastic=False,
            ))

    # ─── Cluster K v1: visual cortex hierarchy (Hubel & Wiesel 1962, Felleman
    # & Van Essen 1991). Retina is driven externally by the runner via image
    # rendering + cp_external_input_current. V1_simple receives sparse Gabor-
    # initialized weights post-build via apply_v1_gabor_weights().
    # V1_simple → V1_complex pools per-orientation (phase invariance).
    # V2 → IT learn via STDP. v1 does NOT yet wire IT → cortex_X — feeding
    # the visual stream into action selection requires separate validation
    # and is deferred to v2.
    if enable_visual_cortex:
        n_retina = 2 * visual_image_size * visual_image_size  # 2*32*32 = 2048
        n_v1_simple = (visual_n_orientations * visual_n_frequencies
                       * visual_n_positions_per_dim * visual_n_positions_per_dim)
        n_v1_complex = (visual_n_orientations
                        * visual_n_positions_per_dim * visual_n_positions_per_dim)
        n_v2 = visual_n_v2
        n_it = visual_n_it

        regions.append(BrainRegion(
            name="retina",
            n_neurons=n_retina,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="cortex_v1_simple",
            n_neurons=n_v1_simple,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="cortex_v1_complex",
            n_neurons=n_v1_complex,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="cortex_v2",
            n_neurons=n_v2,
            exc_fraction=0.8,
            internal_density=0.05,
            exc_weight_mean=2.0, inh_weight_mean=4.0,
            weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="cortex_it",
            n_neurons=n_it,
            exc_fraction=0.8,
            internal_density=0.10,
            exc_weight_mean=2.0, inh_weight_mean=4.0,
            weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

        # Spiking superior colliculus (N1 orienting; de-risked 2026-06-10). A retinotopic
        # sheet sc_map (visual_image_size//2 per side) + a Mexican-hat surround sc_fs. Fed
        # the egocentric retinal image (in the nav loop), it forms a single activity bump at
        # the goal's retinal site; sc_map -> cortex_{N,E,S,W} pooling reads the orienting
        # cardinal BY FIRING (the spiking replacement for sc_orienting_cardinal_from_image).
        if enable_spiking_sc:
            n_sc_side = visual_image_size // 2          # 32 -> 16
            n_sc_map = n_sc_side * n_sc_side            # 256
            # The SC's OWN egocentric eye (separate from the allocentric `retina` the
            # visual cortex / N5 reward / learned-perception use), driven STRONG in the
            # nav loop so the SC forms a robust bump without over-driving the visual cortex.
            regions.append(BrainRegion(
                name="sc_retina", n_neurons=2 * visual_image_size * visual_image_size,
                exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
            regions.append(BrainRegion(
                name="sc_map", n_neurons=n_sc_map, exc_fraction=1.0,
                internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
            regions.append(BrainRegion(
                name="sc_fs", n_neurons=int(n_spiking_sc_fs), exc_fraction=0.0,
                internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
            ))
            # Mexican-hat surround. CRITICAL (de-risk gotcha): these must be declared with
            # REAL density so inject_explicit_wiring marks sc_fs INHIBITORY (sets the
            # per-neuron trait mask). A density-0 + set_pathway_weights route leaves the mask
            # unset -> sc_fs synapses act EXCITATORY and drive the whole map (not a bump).
            pathways.append(RegionPathway(
                from_region="sc_map", to_region="sc_fs",
                density=0.5, weight_mean=4.0, weight_jitter=0.1, plastic=False,
            ))
            pathways.append(RegionPathway(
                from_region="sc_fs", to_region="sc_map",
                density=0.8, weight_mean=2.0, weight_jitter=0.1, plastic=False,
            ))

            # N5: the neural reward = the SC bump's PROXIMITY (goal-salience) signal. sc_rostral
            # pools the sc_map CENTRE (wired post-init, broad Gaussian) so it fires graded with
            # how central/close the goal is -> drives reward_us, replacing the host sign(delta ecc).
            # The temporal-difference is left to the dopamine RPE (delta = r - V, the N9 critic) --
            # the correct + more-biological actor-critic factorization. VALIDATED by the proper
            # dopamine-RPE test (sc_n5_rpe_probe.py: neural reward -> burst on close, monotone
            # corr -0.99, omission dip; lesion+omission anti-cheats confirm load-bearing). The
            # earlier slow-channel temporal-difference circuit (sc_rostral_slow/approach_n5) was
            # dropped: a compound nmda_slow+gaba_b lag (~2.5 nav-steps) + a global-gaba_b-tau
            # collision with the N9 critic. See 2026-06-10-N5-proper-reward-RPE-test-design.md.
            if enable_spiking_sc_approach:
                regions.append(BrainRegion(
                    name="sc_rostral", n_neurons=24, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name))
                if spiking_reward_us:   # sc_rostral (proximity) -> reward_us = the neural reward r
                    pathways.append(RegionPathway(
                        from_region="sc_rostral", to_region="reward_us",
                        density=0.6, weight_mean=14.0, weight_jitter=0.1, plastic=False))

        # retina → V1_simple. Plastic so STDP can refine weights from
        # whatever Gabor init we apply post-build (or from random init in
        # v1 minimal mode). Tagged so the runner can freeze it after a
        # critical-period developmental phase.
        pathways.append(RegionPathway(
            from_region="retina", to_region="cortex_v1_simple",
            density=0.05,           # sparse: Gabor RF is local, not all-to-all
            weight_mean=0.5, weight_jitter=0.5,
            plastic=True,
            plasticity_gate="visual_cortex_v1",
        ))
        # V1_simple → V1_complex: phase pooling (max across frequency + phase
        # within each orientation × position). Implemented as a wide fixed
        # pathway; the bridge averages activity, so this approximates max-
        # pooling at the rate level. plastic=False to lock the pooling.
        pathways.append(RegionPathway(
            from_region="cortex_v1_simple", to_region="cortex_v1_complex",
            density=visual_n_frequencies / float(n_v1_simple),  # roughly N_freq cells per complex cell
            weight_mean=2.0, weight_jitter=0.0,
            plastic=False,
        ))
        # V1_complex → V2: ventral stream. Plastic so V2 learns higher-order
        # features (combinations of orientations/positions).
        pathways.append(RegionPathway(
            from_region="cortex_v1_complex", to_region="cortex_v2",
            density=0.10, weight_mean=1.0, weight_jitter=0.5,
            plastic=True,
            plasticity_gate="visual_cortex_v2",
        ))
        # V2 → IT: object/category-level. Plastic.
        pathways.append(RegionPathway(
            from_region="cortex_v2", to_region="cortex_it",
            density=0.20, weight_mean=1.5, weight_jitter=0.5,
            plastic=True,
            plasticity_gate="visual_cortex_it",
        ))
        # IT → cortex_{N,E,S,W} action selection (Cluster K v2, 2026-05-01).
        # Initialized at weight_mean=0.0 to avoid disrupting cascade dynamics
        # before the visual cortex has learned anything. STDP+reward grow
        # weights from zero post-warmup. Plasticity gate
        # "visual_cortex_action" can be opened (set to 1.0) by the runner
        # after a critical-period warmup, mimicking real visuomotor
        # development where V1/V2/IT mature first then visuomotor wiring
        # follows. weight_jitter=0.0 keeps every synapse at exactly 0
        # weight at init.
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="cortex_it", to_region=f"cortex_{action}",
                density=visual_it_to_cortex_density,
                weight_mean=0.0,  # zero init — STDP+reward grows post-warmup
                weight_jitter=0.0,
                plastic=True,
                plasticity_gate="visual_cortex_action",
            ))

    # ─── Text I/O regions (2026-05-01). Wernicke-area-like input region
    # receives token embeddings; Broca-area-like output region produces
    # action-driving + visualizable activity. Both plastic recurrent.
    # See sim/text_embeddings.py and docs/plans/2026-05-01-text-interaction-design.md.
    if enable_text_io:
        regions.append(BrainRegion(
            name="language_input",
            n_neurons=text_n_input_neurons,
            exc_fraction=0.8,
            internal_density=0.05,
            exc_weight_mean=2.0, inh_weight_mean=4.0,
            weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="language_output",
            n_neurons=text_n_output_neurons,
            exc_fraction=0.8,
            internal_density=0.10,
            exc_weight_mean=2.0, inh_weight_mean=4.0,
            weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

        # language_input → PFC (so words enter working memory)
        # Only if PFC region exists
        if enable_pfc:
            pathways.append(RegionPathway(
                from_region="language_input", to_region="dlpfc_wm",
                density=text_input_to_pfc_density,
                weight_mean=text_input_to_pfc_weight,
                weight_jitter=0.5,
                plastic=True,
                plasticity_gate="language_input_to_pfc",
            ))

        # language_input → cortex_X (word-to-action learning).
        # Per Kandel ch 53 (developmental pruning), the brain starts with
        # DENSE connectivity that gets pruned via experience, not zero
        # weights that grow. Non-zero init lets each token's pattern have
        # SOME initial differential drive to cortex_X; STDP then refines
        # which connections to strengthen vs. weaken. This counteracts
        # the cascade's structural N-bias (cortex_N fires 2x more at init,
        # so language with zero weights can't compete).
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="language_input", to_region=f"cortex_{action}",
                density=text_input_to_cortex_density,
                weight_mean=text_input_to_cortex_weight,  # non-zero default
                weight_jitter=text_input_to_cortex_jitter,
                plastic=True,
                plasticity_gate="language_input_to_cortex",
            ))

        # IT → language_output (image-to-word learning).
        # Only when visual cortex is also enabled — without IT there's
        # no upstream signal to drive the readout.
        # 2026-05-02: small non-zero init via text_it_to_output_weight kwarg
        # (default 0.0 preserves prior behavior; pass >0 to seed STDP).
        if enable_visual_cortex:
            pathways.append(RegionPathway(
                from_region="cortex_it", to_region="language_output",
                density=text_it_to_output_density,
                weight_mean=text_it_to_output_weight,
                weight_jitter=text_it_to_output_jitter,
                plastic=True,
                plasticity_gate="it_to_language_output",
            ))

        # cortex_X → language_output (action verbalization).
        # Lets the agent "say what it just did" — STDP+reward grows
        # weights when the supervisor clamps the appropriate word output
        # while a cortex_X is active.
        # 2026-05-02: small non-zero init via text_cortex_to_output_weight
        # kwarg (default 0.0 preserves prior behavior).
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"cortex_{action}", to_region="language_output",
                density=0.10,
                weight_mean=text_cortex_to_output_weight,
                weight_jitter=text_cortex_to_output_jitter,
                plastic=True,
                plasticity_gate="cortex_to_language_output",
            ))

        # ─── PFC-bypass: language_input → motor_X DIRECT ───
        # Biology source: Kandel ch 60 + Geschwind disconnection model.
        # Real anatomy: Wernicke's (auditory comprehension) → arcuate
        # fasciculus → Broca's (motor planning) → primary motor cortex.
        # Our cortex_X is more like cingulate/parietal action-selection,
        # which is BG-cascade biased (cortex_N dominates from cluster A/E
        # feedback). To bypass cascade bias for instructed action, we
        # provide a direct language_input → motor_X pathway.
        # Plastic, gated separately so the regime can disable cortex
        # involvement and force PFC bypass.
        if not enable_distributed_motor_pop:
            for action in ACTION_NAMES:
                pathways.append(RegionPathway(
                    from_region="language_input", to_region=f"motor_{action}",
                    density=text_input_to_motor_density,
                    weight_mean=text_input_to_motor_weight,
                    weight_jitter=text_input_to_motor_jitter,
                    plastic=True,
                    plasticity_gate="language_input_to_motor",
                ))
        else:
            # Distributed motor pool: language_input projects to ALL 8
            # sub-pools with PLASTIC weights. STDP+reward sculpts which
            # neurons fire for which token. Initial weight is uniform
            # across sub-pools (token-token contrast emerges via training).
            # Same total density as labeled-line: 0.30 × 4 pools = 1.20
            # spread across 8 sub-pools = 0.15 per sub-pool to match.
            SUBPOOL_SUFFIXES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
            per_subpool_density = text_input_to_motor_density * 4.0 / 8.0
            for suffix in SUBPOOL_SUFFIXES:
                pathways.append(RegionPathway(
                    from_region="language_input",
                    to_region=f"motor_pop_{suffix}",
                    density=per_subpool_density,
                    weight_mean=text_input_to_motor_weight,
                    weight_jitter=text_input_to_motor_jitter,
                    plastic=True,
                    plasticity_gate="language_input_to_motor",
                ))

    return regions, pathways


def _warn_motor_lateral_inhibition_deprecated(value: bool) -> bool:
    """Emit a one-time DeprecationWarning if --motor-lateral-inhibition was
    used. The flag is NEGATIVE on cheat-5 (2026-04-26 evaluation) and the
    biology is wrong (real motor-pool WTA = spinal Renshaw, not cortical-FS).
    Slated for removal in a future cleanup."""
    if value:
        import warnings
        warnings.warn(
            "--motor-lateral-inhibition is DEPRECATED (NEGATIVE on cheat-5 "
            "evaluation; biology is wrong — real motor-pool WTA is spinal "
            "Renshaw inhibition per Kandel ch 35, not cortical-FS-like "
            "inhibition). Slated for removal in a future cleanup. If you "
            "need motor-WTA dynamics, plan to use spinal Renshaw modeling "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return value


def _position_to_cortex_drive(x, y, n_cortex_per_action, grid_size,
                                rate_peak=400.0, rate_floor=50.0, sigma=1.5):
    """Map (x,y) position to per-action cortex drive amplitudes.

    Each action's cortex pool gets a baseline + position-dependent component.
    For now: uniform baseline drive to all 4 cortex pools (the differential
    selectivity comes from learning the cortex→striatum weights, not from
    input encoding).

    Returns a dict {action: drive_pA}.
    """
    # Simple encoding: drive ALL cortex pools uniformly with a position-
    # dependent total amplitude. The cortex→striatum learning has to
    # discover which action is right for each position.
    return {a: rate_peak for a in ACTION_NAMES}


# Plasticity gates we expect to find on the runner's pathways. Pretraining
# thaws all of these; absence means a runner-side typo in plasticity_gate=
# (or a flag that doesn't add the pathway). Error early before GPU work.
_PRETRAINING_THAWED_GATES = (
    "corticostriatal",
    "sensory_to_cortex",
    "place_goal_to_cortex",
    "beacon_to_goal",
    "landmark_to_place",
    "dlpfc_wm_pathways",
    "corticostriatal_cross",
)


def _sample_pretraining_goal(rng, grid_size, start_pos, prev_goal):
    """Uniform random (gx, gy) on the grid with Manhattan >= 3 from start_pos
    and != prev_goal. Re-samples on rejection. The grid is small enough
    (8x8 → 16 valid cells given start (1,1)) that rejection sampling is
    trivially fast."""
    sx, sy = start_pos
    while True:
        gx = rng.randrange(grid_size)
        gy = rng.randrange(grid_size)
        if abs(gx - sx) + abs(gy - sy) < 3:
            continue
        if prev_goal is not None and (gx, gy) == prev_goal:
            continue
        return (gx, gy)


def _run_pretraining_phase(
    bridge,
    cfg,
    regions,
    n_goals: int,
    steps_per_goal: int,
    grid_size: int,
    start_pos,
    seed: int,
    enable_bg_cross_projections: bool = True,
    verbose: bool = True,
) -> dict:
    """Critical-period analog. Thaws ALL declared plasticity gates and runs
    the agent through n_goals random goals for steps_per_goal trials each.

    Returns a summary dict: {n_trials, n_goal_changes, cross_weights_mean,
    cross_weights_std}. See docs/plans/2026-04-28-cheat5-v4-design.md."""
    available = set(bridge.list_plasticity_gates())
    missing = [g for g in _PRETRAINING_THAWED_GATES
               if g not in available
               and _gate_required(g, regions,
                                  enable_bg_cross_projections=enable_bg_cross_projections)]
    if missing:
        raise KeyError(
            f"_run_pretraining_phase: gate(s) not declared on any pathway: "
            f"{missing!r}. Available: {sorted(available)!r}. "
            f"Either spell-check the gate name in build_bg_brain_regions, "
            f"or enable the flag that adds the pathway."
        )

    # Thaw every gate that IS declared. Gates not declared (e.g. learned
    # perception is off, so sensory_to_cortex doesn't exist) are silently
    # skipped — the corresponding pathway just isn't there.
    for gate in _PRETRAINING_THAWED_GATES:
        if gate in available:
            bridge.set_plasticity_gate(gate, 1.0)

    if verbose:
        print(f"[g11 seed={seed}] pretraining: all {len(available)} declared gates "
              f"thawed to 1.0; running {n_goals} goals × {steps_per_goal} steps each",
              flush=True)

    # Capture cross-projection synapse indices once (constant after build) so
    # we can compute weight stats at the end of pretraining. Empty if the
    # gate isn't declared (e.g. --bg-cross-projections off).
    cross_indices_cpu = []
    if "corticostriatal_cross" in getattr(bridge, "_plasticity_gate_to_synapses", {}):
        cross_indices_cpu = list(bridge._plasticity_gate_to_synapses["corticostriatal_cross"])

    # Early-out: zero goals → nothing to drive. Useful for tests that only
    # exercise the gate-thaw / signature path. Fall through to the summary.
    if n_goals == 0 or steps_per_goal == 0:
        return {
            "n_trials": 0,
            "n_goal_changes": 0,
            "cross_weights_mean": float("nan"),
            "cross_weights_std": float("nan"),
        }

    # Imports kept inside the helper to match the file's existing style and
    # avoid touching the top-of-file import block (Task 5 constraint).
    import random
    import numpy as np
    import cupy as cp

    # Reconstruct GPU-index arrays for the regions we drive in the inner
    # loop. The eval loop pre-caches these in run_moving_goal_episode; we
    # rebuild here so the helper stays self-contained (no extra kwargs).
    region_indices_cp = {}
    for r in regions:
        idx = list(bridge.region_manager.indices(r.name))
        if idx:
            region_indices_cp[r.name] = cp.asarray(idx, dtype=cp.int64)
    motor_idx_per_action = {
        a: region_indices_cp[f"motor_{a}"] for a in ACTION_NAMES
    }

    # Stimulus / readout window (mirrors eval). The fused dynamics
    # accumulate in 0.5 ms ticks, so 100 ms = 200 sub-steps.
    STIMULUS_MS = 100.0
    READOUT_START_MS = 30.0
    READOUT_END_MS = 100.0
    n_stim_steps = int(STIMULUS_MS / cfg.dt_ms)
    readout_start = int(READOUT_START_MS / cfg.dt_ms)
    readout_end = int(READOUT_END_MS / cfg.dt_ms)
    reward_hold_steps = 10  # matches eval default

    # Action geometry (must mirror ACTION_DELTAS in run_moving_goal_episode)
    ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

    # Lock baseline tonic drives once. The eval loop re-sets these every
    # trial as a defensive measure; for pretraining we accept the drift
    # tradeoff in exchange for simpler code and equivalent biology (basal
    # ganglia tonic drives are biologically slow-varying).
    bridge.cp_external_input_current[:] = 0.0
    for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
    for rn in [f"gpe_arky_{a}" for a in ACTION_NAMES]:
        if rn in region_indices_cp:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(120.0)
    for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(110.0)
    for rn in ["stn", "snc"]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
    for rn in [f"thal_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(300.0)

    rng = random.Random(seed * 7919)  # deterministic, distinct from eval RNGs
    # Action-selection RNG must NOT collide with the eval loop's per-step
    # RNG seeds (which use seed*10000 + step). Use a different prime offset.
    action_rng = np.random.default_rng(seed * 13_417)

    prev_goal = None
    n_goal_changes = 0
    trial_counter = 0
    x, y = start_pos

    HEURISTIC_DRIVE_PA = cp.float32(800.0)

    for goal_idx in range(n_goals):
        gx, gy = _sample_pretraining_goal(rng, grid_size, start_pos, prev_goal)
        prev_goal = (gx, gy)
        n_goal_changes += 1
        if verbose:
            print(f"[g11 seed={seed}] pretraining goal {goal_idx + 1}/{n_goals}: "
                  f"({gx},{gy})", flush=True)

        # Reset agent to start at each new pretraining-goal episode
        x, y = start_pos

        for trial in range(steps_per_goal):
            # ── Heuristic cortex drive: directly drive cortex_X for each
            # goal-relative direction. Pretraining always uses the
            # heuristic — no opt-in perception modes here. The point is to
            # evolve weights under varied goals using the simplest possible
            # input pathway.
            #
            # Zero cortex pools first so the prior trial's drive doesn't
            # leak across direction transitions.
            for a in ACTION_NAMES:
                bridge.cp_external_input_current[region_indices_cp[f"cortex_{a}"]] = cp.float32(0.0)
            if gy > y:
                bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = HEURISTIC_DRIVE_PA
            if gx > x:
                bridge.cp_external_input_current[region_indices_cp["cortex_E"]] = HEURISTIC_DRIVE_PA
            if gy < y:
                bridge.cp_external_input_current[region_indices_cp["cortex_S"]] = HEURISTIC_DRIVE_PA
            if gx < x:
                bridge.cp_external_input_current[region_indices_cp["cortex_W"]] = HEURISTIC_DRIVE_PA

            # ── Run stimulus window and tally motor spikes
            motor_counts = {a: 0 for a in ACTION_NAMES}
            bridge.core_config.current_reward_signal = 0.0
            for s in range(n_stim_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                bridge.runtime_state.current_time_ms = (
                    bridge.runtime_state.current_time_step * cfg.dt_ms
                )
                if readout_start <= s < readout_end:
                    firing = bridge.cp_firing_states.get().astype(bool)
                    for a in ACTION_NAMES:
                        motor_counts[a] += int(firing[motor_idx_per_action[a].get()].sum())

            # ── Argmax action selection (random if all silent)
            if max(motor_counts.values()) > 0:
                action_idx = max(range(N_ACTIONS),
                                 key=lambda i: motor_counts[ACTION_NAMES[i]])
            else:
                action_idx = int(action_rng.integers(0, N_ACTIONS))

            # ── Position update + reward (Manhattan-delta only; sensed
            # reward is an eval-time refinement and adds no value during
            # pretraining where we just want weight evolution)
            dist_before = abs(x - gx) + abs(y - gy)
            dxa, dya = ACTION_DELTAS[action_idx]
            new_x = int(np.clip(x + dxa, 0, grid_size - 1))
            new_y = int(np.clip(y + dya, 0, grid_size - 1))
            x, y = new_x, new_y
            dist_after = abs(x - gx) + abs(y - gy)

            if dist_after < dist_before:
                reward = 1.0
            elif dist_after > dist_before:
                reward = -1.0
            else:
                reward = 0.0

            # ── Reward signal hold: drive plasticity for reward_hold_steps
            # extra sim ticks. This is the actual learning step — STDP
            # eligibility built up during the stimulus window gets
            # converted to weight updates here.
            if abs(reward) > 0:
                bridge.core_config.current_reward_signal = float(reward)
                for _ in range(reward_hold_steps):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                    bridge.runtime_state.current_time_ms = (
                        bridge.runtime_state.current_time_step * cfg.dt_ms
                    )
                bridge.core_config.current_reward_signal = 0.0

            # Structural pruning (cheat-5 option-1, 2026-04-28). Only fires
            # during pretraining when enable_structural_pruning is on. Restricted
            # to cross-projection synapses so we don't sparsify the same-action
            # corticostriatal routing. cp_eligibility_trace is allocated at capacity
            # (which can exceed nnz to leave room for structural plasticity), so
            # we slice it down to nnz before handing to update_pruning.
            if cfg.enable_structural_pruning and bridge.cp_synapse_alive is not None:
                cross_idx_list = bridge._plasticity_gate_to_synapses.get("corticostriatal_cross")
                if cross_idx_list:
                    nnz = int(bridge.cp_connections.nnz)
                    bridge.update_pruning(
                        eligibility_trace=bridge.cp_eligibility_trace[:nnz],
                        reward_signal=reward,
                        prunable_indices=cp.asarray(list(cross_idx_list), dtype=cp.int64),
                    )

            trial_counter += 1

    # ── Cross-projection weight summary
    if cross_indices_cpu:
        cross_w = bridge.cp_connections.data[cp.asarray(cross_indices_cpu)].get()
        if np.isnan(cross_w).any():
            raise RuntimeError(
                "pretraining produced NaN cross-projection weights — likely STDP "
                "instability. Lower learning rate or shorten "
                "pretraining_steps_per_goal."
            )
        cross_mean = float(cross_w.mean())
        cross_std = float(cross_w.std())
    else:
        cross_mean = float("nan")
        cross_std = float("nan")

    if verbose:
        print(f"[g11 seed={seed}] pretraining complete: {trial_counter} trials, "
              f"{n_goal_changes} goal changes; cross weights mean={cross_mean:.3f} "
              f"std={cross_std:.3f} -> handing off to eval (curriculum will freeze "
              f"corticostriatal_cross)", flush=True)

    return {
        "n_trials": trial_counter,
        "n_goal_changes": n_goal_changes,
        "cross_weights_mean": cross_mean,
        "cross_weights_std": cross_std,
    }


def _gate_required(name: str, regions, enable_bg_cross_projections: bool = True) -> bool:
    """Return True iff the gate must exist regardless of which flags are on.

    `regions` is accepted for forward-compatibility — Task 3 will inspect it
    to derive the full required-set from the active flag combination.
    Currently unused; we hard-code the gates known to always exist or whose
    presence is gated by a known flag.

    `enable_bg_cross_projections` softens the bg_cross_projections requirement:
    when False, that gate is not expected (--bg-cross-projections is off, so
    the pathway isn't built). Pretraining still runs but won't shape any
    cross-projection weights — Task 7 emits a warning at that path.
    """
    if name == "corticostriatal":
        return True
    if name == "corticostriatal_cross":
        return enable_bg_cross_projections
    return False


def run_moving_goal_episode(
    out_path: str,
    seed: int = 42,
    n_steps: int = 1800,
    grid_size: int = 8,
    start_pos=(1, 1),
    goal_pos=(6, 6),
    goal_schedule=None,
    n_hippocampus_per_layer: int = 64,  # default 8×8 grid; should be roughly grid_size²
    sensory_to_cortex_weight: float = 10.0,
    hippocampus_to_cortex_weight: float = 10.0,
    enable_pfc: bool = False,
    n_pfc: int = 60,
    pfc_internal_density: float = 0.2,
    goal_to_pfc_weight: float = 8.0,
    pfc_to_cortex_weight: float = 8.0,
    # Cluster G v1 (2026-05-01): Wang 2002 NMDA-mediated PFC working memory.
    # When True, enables global NMDA with elevated 0.5 NMDA:AMPA ratio
    # (Wang 2002 calibration for PFC pyramidals). Combined with --enable-pfc,
    # gives the dlpfc_wm region true persistent activity for delayed-
    # response tasks. NOTE: NMDA is currently a global cfg flag, so this
    # affects all regions, not just PFC. Future work: per-region NMDA
    # ratio override. See docs/plans/2026-05-01-cluster-g-pfc-wm-wang2002.md.
    enable_pfc_nmda: bool = False,
    enable_bg_cross_projections: bool = False,
    cross_projection_weight: float = 5.0,
    cross_projection_density: float = 1.0,
    cross_projection_topology_seed: int = 0,
    # v3 (2026-04-28) — see build_bg_brain_regions docstring.
    enable_bg_lateral_inhibition: bool = False,
    lateral_inhibition_density: float = 0.3,
    lateral_inhibition_weight: float = 2.0,
    # Interactive runtime control (2026-04-28). When set to a writable JSON
    # file path, the runner polls the file at the start of each trial and
    # applies the contents:
    #   { "paused": bool, "goal": [gx, gy] | null, "inject_reward": float | null }
    # - paused: blocks the trial loop until cleared
    # - goal: overrides the scheduled goal (persistent until set again)
    # - inject_reward: one-shot additive reward applied this trial; runner
    #   clears it back to null after consuming
    # Used by the webapp's World-tab live mode for click-to-teleport-goal,
    # pause/resume, and reward-injection. Default None = no polling, no
    # behavior change (ie. fully backwards compatible).
    interactive_control_file: str = None,
    # Progress print frequency (steps). Default 100 keeps validation runs
    # quiet; webapp interactive runs override to 1 so the dashboard's live
    # mode can animate per-step instead of jumping every 100 steps.
    progress_print_interval: int = 100,
    # Optional throttle (ms) between trials. Lets a human watch the agent
    # learn in real time without GPU saturation outpacing the eye. Default
    # 0 = full speed.
    trial_sleep_ms: float = 0.0,
    enable_beacon_perception: bool = False,
    n_beacon_sensors: int = 8,
    beacon_to_goal_weight: float = 8.0,
    beacon_max_intensity: float = 600.0,  # peak sensor drive (pA) when on top of beacon
    beacon_falloff: float = 1.0,           # intensity = peak / (1 + falloff*distance)
    beacon_replaces_goal: bool = False,    # if True, beacon→goal_cells is the ONLY goal info (true Stage 1 test)
    # Landmark perception (Item 1 Stage 2, 2026-04-27).
    enable_landmarks: bool = False,
    n_landmark_sensors: int = 8,
    landmark_to_place_weight: float = 8.0,
    landmark_position: tuple = None,  # default to grid center
    landmark_max_intensity: float = 600.0,
    landmark_falloff: float = 1.0,
    landmarks_replace_place: bool = False,
    # Cheat #4: sensed reward (2026-04-27).
    # Default reward = +1 if Manhattan distance decreased, -1 if increased.
    # This computes from raw (gx, gy, x, y) coordinates — a cheat. Sensed
    # reward instead computes reward from beacon-intensity GRADIENT
    # (intensity_after - intensity_before): the agent "feels warmer" as it
    # approaches and "cooler" as it retreats. Same information content as
    # distance-based, but operates on the agent's perceptual signal.
    enable_sensed_reward: bool = False,
    # N5 (2026-06-08): coordinate-FREE perceived-approach reward. reward = sign of
    # the DECREASE in the goal's retinal eccentricity (the image-sourced offset
    # magnitude the reflex already reads from pixels) — appetitive/incentive-
    # salience approach reward (Schultz reward-fn-2; Berridge wanting; phototaxis).
    # Use with --enable-visual-cortex. Takes precedence over sensed/Manhattan reward.
    perceived_approach_reward: bool = False,
    # Homeostatic agent hook (2026-06-17). Default None = byte-identical (no
    # behavior change for any existing caller; guarded below). When set to a
    # callable, it is invoked once per trial AFTER the natural reward is
    # finalized, as: gated_reward, new_goal = homeostatic_hook(reward, x, y,
    # gx, gy, step, dist_after). The hook lets a drive (e.g. a self-generated
    # hunger signal) GATE the reward (reward *= hunger) and relocate the goal
    # (food) on an "eat" event (dist_after == 0), so the validated BG-cascade +
    # value-critic learner can be reused for a homeostatic agent WITHOUT a fork
    # or any re-derivation of the tuned drive/readout/reward loop. Returning
    # new_goal != None reassigns (gx, gy) and logs a goal change. See
    # research/runners/_homeostatic_g11bg_reuse_probe.py.
    homeostatic_hook=None,
    # Hidden-goal (Morris-water-maze analogue) diagnostic (2026-06-19). Default
    # False = byte-identical to every existing caller. When True, the goal's
    # coordinates are NOT fed into the brain anywhere: the ppc_goal_input goal
    # drive (gx,gy → goal cells) is zeroed each step (the place drive (x,y →
    # sensor_place_readout) stays — own-position self-knowledge is legitimate).
    # Combined with --heuristic-strength 0 (no goal-direction teacher) and no
    # cue-reflex / SC-orienting / learned-perception, the ONLY goal-related
    # signal reaching the agent is the SCALAR reward (distance-decreased → +1).
    # The agent must therefore learn the goal's location via reward → value →
    # dopamine → corticostriatal STDP — i.e. the spiking reward/value/dopamine
    # limbic core must be BEHAVIORALLY LOAD-BEARING. This is the harder task the
    # 2026-06-19 scoping flagged: on the visible/orient-solvable gridworld the
    # limbic core is GREEN_INERT (validated but inert). See the load-bearing
    # diagnostic finding. NO sim/ edit; suppression is a runner-side drive zero.
    hidden_goal: bool = False,
    # Reward lesion (the load-bearing anti-cheat, 2026-06-19). Default False =
    # byte-identical. When True, the natural reward is FORCED to 0 every step
    # (after the natural computation), so NO learning signal reaches dopamine /
    # the value critic / corticostriatal plasticity. The owner standard
    # (validate_signal_by_its_function): the reward lesion must collapse the
    # BEHAVIOR (the nav score), not merely the SNc/reward-pop firing. If the
    # agent still solves the hidden goal with reward lesioned, the task is not
    # reward-load-bearing. Applied AFTER manual_reward_injection / the
    # homeostatic hook so it is an unconditional clamp.
    lesion_reward: bool = False,
    # Cue-following reflex (Item 1 Stage 3, 2026-04-27).
    # Replaces the heuristic with a hand-tuned innate reflex that computes
    # cortex drive from beacon sensor activations. Models a real animal's
    # "approach attractive cue" reflex (e.g., phototaxis). The reflex is
    # non-plastic — it represents innate sensorimotor wiring like vestibular
    # reflexes or looming detection. Plastic layers (sensory, hippo, beacon
    # → goal_cells) layer on top to refine the behavior.
    enable_cue_reflex: bool = False,
    cue_reflex_strength: float = 800.0,  # peak reflex drive matching heuristic
    cue_reflex_replaces_heuristic: bool = False,  # if True, heuristic disabled when reflex on
    # N1 de-risk (2026-06-07): innate SC orienting reflex — image-sourced (NO
    # coords), the biological replacement for the coordinate heuristic-teacher.
    sc_orienting_reflex: bool = False,
    sc_reflex_strength: float = 800.0,  # SC orienting push (pA), matches heuristic
    # Rank 2 (2026-06-07): the DURABLE learned dorsal/PPC read-out — drive the
    # learned sensory population from the IMAGE salience offset (position-
    # preserving, NO coords); the innate SC reflex teaches then weans.
    learned_perception_from_vision: bool = False,
    sc_reflex_wean_start: int = -1,   # step to begin weaning the reflex (-1 = never)
    sc_reflex_wean_steps: int = 1500,  # linear ramp-to-zero window
    # Rank 2 tuning (2026-06-07): supervised motor-teacher (feedback-error-
    # learning). When > 0, the reflex drives its chosen target cortex pool at
    # THIS strength (a clean supervised label for the sensory→cortex STDP)
    # instead of the movement-reflex strength. 0 = off (plain reflex).
    sensory_cortex_teacher_pA: float = 0.0,
    learning_rate: float = 0.01,
    reward_eligibility_tau_ms: float = 500.0,
    reward_hold_steps: int = 10,
    verbose: bool = True,
    enable_motor_lateral_inhibition: bool = False,
    enable_cortex_lateral_inhibition: bool = False,
    enable_per_action_da_targeting: bool = False,
    enable_adaptive_per_action_da: bool = False,
    adaptive_da_ema_decay: float = 0.9,  # ~tau=10 trials (used for positive reward)
    adaptive_da_ema_decay_negative: float = None,  # if set, separate decay for negative reward (faster = quicker exploration trigger)
    enable_learned_perception: bool = False,
    sensory_drive_max_pA: float = 600.0,
    sensory_drive_sigma: float = 1.5,
    enable_hippocampus: bool = False,
    hippocampus_drive_max_pA: float = 600.0,
    hippocampus_drive_sigma: float = 0.5,  # narrower → sparser firing → 1-3 cells per position
    # Informed init for learned perception: bias initial sensory->cortex_X weights
    # by alignment between sensor's preferred (dx,dy) and action X's direction
    # vector. Solves cold-start failure (random init produces no asymmetry, no
    # learning signal). Plasticity then refines the prior rather than discovers.
    enable_learned_perception_informed_init: bool = False,
    informed_init_alpha: float = 8.0,  # sharper positive-only prior; aligned ~ 24.5 weight (heuristic-equivalent)
    # DA-gated WTA: when both --motor-lateral-inhibition and adaptive DA are on,
    # scale FS→motor inhibition weight per-trial by gating_strength (reward EMA).
    # Implements the user's "DA gate" concept: when winning, WTA strong (commit);
    # when losing, WTA relaxes (explore via reduced inhibition).
    enable_da_gated_wta: bool = False,
    # RPE-scaled reward (NE-like surprise amplification):
    # delivered_reward = reward + alpha * (reward - reward_ema)
    # When reward is unexpectedly negative (after positive EMA), the prediction
    # error is large and amplified — fast adaptation. Expected outcomes get
    # muted. Real biology: DA encodes RPE not raw reward (Schultz 1997).
    enable_rpe_scaled_reward: bool = False,
    rpe_scale_alpha: float = 1.0,  # 1.0 means: delivered = 2*reward - ema
    # N9 step 1 (2026-06-08): the dopamine signal IS the reward-PREDICTION-ERROR
    # delta = r - V (V = reward_ema, the learned Rescorla-Wagner critic). Converts
    # the actor-only / raw-reward DA into an actor-CRITIC RPE (Schultz 1998).
    # With --perceived-approach-reward (N5) the whole RPE loop is coordinate-free.
    rpe_dopamine: bool = False,
    # ── Spiking-SNc actor-critic Stage A (2026-06-08) ──────────────────────
    # docs/plans/2026-06-08-spiking-snc-actor-critic-design.md.
    # When True, the dopamine reward-prediction error is computed by the
    # FIRING of the spiking `snc` pool (IZH2007_DOPAMINE, the previously-silent
    # placeholder region), NOT a host formula. Each reward step the SNc pool is
    # driven by three additive external currents:
    #   I_snc = snc_tonic_pa + snc_reward_gain*max(0, r) - snc_value_gain*V
    # so its windowed firing rate encodes delta = r - V (burst above tonic on
    # +RPE, tonic at 0, DIP below tonic on -RPE). The DA broadcast is then
    # produced FROM that firing via the new `from_region_firing_signed`
    # neuromodulator rule (the only protected sim/ edit) -> the bridge's existing
    # da_signal = get_concentration("dopamine") - baseline path consumes it.
    # STAGE A: value V is the host reward_ema scaffold (the inhibitory drive is
    # proportional to the host R̄). Stage B's neural critic is NOT in this runner.
    # Supersedes --rpe-dopamine (host formula) and owns the `dopamine` modulator
    # vs --enable-tonic-da (precedence guard). Default OFF (additive).
    spiking_snc: bool = False,
    snc_tonic_pa: float = 220.0,       # tonic pacemaker drive -> mid-range SNc rate (headroom to dip)
    snc_reward_gain: float = 400.0,    # k_r: excitatory reward afferent gain (pA per unit r)
    snc_value_gain: float = 400.0,     # k_v: inhibitory value (striosome) drive gain (pA per unit V)
    snc_da_sensitivity: float = 8.0,   # signed-rule sensitivity (firing-rate deviation -> DA conc)
    # ── Spiking-SNc actor-critic Stage B (2026-06-08): NEURAL value critic ──
    # Only meaningful with spiking_snc=True. When True, a dedicated GABAergic
    # `striosome_value` critic (built in build_bg_brain_regions) learns V(s) from
    # the perceived state (cortex_it) via the SNc-derived dopamine delta, and
    # SUBTRACTS V at the SNc membrane through the slow GABA_B/GIRK K+ conductance
    # (cfg.enable_gabab; the critic->snc pathway is receptor="gaba_b"). The host
    # _V_scaffold term in the reward block is then DROPPED — the r-V subtraction
    # is NEURAL, not host arithmetic (the BRAIN-BASED-ONLY completion of Stage B).
    # When False, Stage A behavior is byte-unchanged (host _V_scaffold subtraction).
    # Validated CPU de-risk: research/findings/2026-06-08-gabab-girk-stageB-derisk-GO.md.
    enable_neural_critic: bool = False,
    # ── 2026-06-10 spiking US/reward -> SNc (drops the host SNc reward write) ──
    # With spiking_reward_us, a PPN-like `reward_us` population receives the PERCEIVED reward (N5's
    # coord-free approach signal, a sensory drive) and FIRES into the SNc, so the reward burst is a
    # NEURON's synapse (US->VTA), not a host current write. STRONGLY recommend --perceived-approach-
    # reward so the US rides on pixels (coord-free); else a WARN (the reward is the coord default).
    spiking_reward_us: bool = False,
    n_reward_us: int = 40,
    reward_us_to_snc_weight: float = 50.0,
    reward_us_to_snc_density: float = 0.6,
    reward_us_drive_pa: float = 250.0,      # US-afferent drive (Pavlovian de-risk: reward_us ~66Hz, SNc burst 266 vs tonic 57, V subtracts 266->86)
    # ── Critic drive calibration (2026-06-08, runner-side; diagnosed by
    #    research/findings/raw/g11_bg/_placecritic_diag*.py). The smoke found the
    #    MSN-D1 striosome_value critic NEVER FIRED in nav. Root cause (decisive):
    #    (1) the MSN-D1 preset's depolarized rheobase (~700 pA) is built for the
    #        cortically-driven up-state; the SPARSE sensor_place_readout place code
    #        (HIPPO_PYRAMIDAL, ~3-8 Hz) cannot supply that convergent current, so the
    #        afferent route can't fire it at ANY weight (verified to w=25) — and even
    #        a 600 pA teacher gave 0.4 Hz. A more EXCITABLE critic type
    #        (IZH2007_RS_CORTICAL_PYRAMIDAL) DOES fire from the place code (5-20 Hz,
    #        graded by drive x weight) — diag4.
    #    (2) a reward-window TEACHER current on the critic is COUNTER-productive: it
    #        drives post >> pre so STDP sees post-before-pre LTD and the place->value
    #        weight COLLAPSES (diag6). The place-driven RS critic gives clean LTP.
    #    (3) at grid_size=32 the default place sigma=0.5 (tuned for the 8x8 grid,
    #        cell spacing 1) is far narrower than the 4.43 cell spacing -> the place
    #        code is near-silent at most positions. A modest widening makes it a real
    #        population bump so a value-of-LOCATION can be carved (diag6: sigma>=1.5 +
    #        RS critic + afferent w>=8 -> V rises, V(near)>V(far), location-selective
    #        weight growth, teacher-FREE).
    #    Calibration = RS critic + raised afferent weight + NO teacher (+ the place
    #    sigma widened via --hippocampus-drive-sigma at the run level). All runner-
    #    side; the MSN-D1 default is preserved when the override is None.
    critic_neuron_type: str = None,            # override striosome_value izh type (None=keep MSN-D1)
    critic_afferent_weight: float = 3.0,       # (legacy 2026-06-08) sensor_place_readout->value weight
    critic_afferent_density: float = 0.6,
    # ── 2026-06-09 VALIDATED redesign (navfaithful-afferent-critic-homeostasis PASS) ──
    # The critic afferent is now the DEDICATED DENSE `vs_place_context` (grid-32 place code,
    # 30-80 cells/location), fired into a useful range by per-region homeostasis on BOTH the
    # afferent AND the MSN-D1 critic (GLOBAL homeostasis stays OFF). This SUPERSEDES the
    # 2026-06-08 RS-critic-type + raised-sensor-afferent-weight calibration above (which the
    # de-risk arc showed could not, on its own, fire the MSN critic from the SPARSE actor place
    # code under the deterministic regime). With this on, --critic-neuron-type stays None (the
    # critic keeps its MSN-D1 default; homeostasis — not the type swap — fires it).
    enable_critic_homeostasis: bool = False,   # per-region homeostasis on vs_place_context + critic
    n_vs_place_context: int = 200,             # dense dedicated place-context afferent size
    vs_place_to_value_weight: float = 0.2,     # vs_place_context->striosome_value plastic INIT weight (STDP grows V up)
    vs_place_to_value_density: float = 0.5,
    # ── 2026-06-09 CONVERGENT-EXCITATION UP-STATE (homeostasis-free critic firing; Option A). ──
    # Adds a DISTINCT dense NON-plastic `vs_place_drive` afferent (the B.02 up-state arm) alongside
    # `vs_place_context` (the plastic value learner). When on, the runner injects the SAME grid-32
    # place code into BOTH each nav step. CuPy 3-seed de-risk: FIRE/LEARNS/ACTOR pass but PLACE-
    # GRADED FAILS (the dense up-state is position-blind) — HONEST NEGATIVE; shipped default-OFF as
    # documented infrastructure (byte-identical when off). See build_bg_brain_regions + the finding
    # 2026-06-09-N9-convergent-upstate-derisk.md.
    enable_convergent_upstate: bool = False,
    vs_place_drive_to_value_weight: float = 28.0,
    vs_place_drive_to_value_density: float = 0.8,
    # ── 2026-06-09 N9 NEURAL PLACE-CODE SELF-ORG (nav deployment of the validated de-risk
    #    n9_place_graded_critic_stage2_derisk; design docs/plans/2026-06-09-N9-nav-deployment-
    #    design.md). When True (only meaningful WITH enable_neural_critic), the host-Gaussian
    #    vs_place_context place code is REPLACED by a SELF-ORGANIZED spiking place code
    #    (place_sensors -> place [+ FS-PING place_fs] -> striosome_value coincidence critic).
    #    The STEP-1 self-org runs in _run_critic_warmup; the per-step host Gaussian injection is
    #    NOT used in this path (place_sensors egocentric render is the only place input). HARD-
    #    GATES enable_convergent_upstate OFF (the position-blind A1 floor caps grading ~1.2x).
    #    Default OFF => the enable_neural_critic path is byte-identical (host vs_place_context).
    neural_place_selforg: bool = False,
    deterministic_selforg: bool = False,  # toggle cfg.deterministic_transpose_matvec during STEP-1 self-org (reproducible place code)
    n_place: int = 200,
    n_place_fs: int = 24,
    place_sensors_to_place_weight: float = 28.0,
    place_sensors_to_place_density: float = 0.5,
    place_sensors_to_place_jitter: float = 0.6,
    enable_critic_fs_inhibition: bool = False,  # place_fs->striosome_value GABA_A: spiking critic rate-clamp (root fix vs GIRK cap masking)
    critic_fs_weight: float = 16.0,
    critic_fs_density: float = 0.6,
    place_fs_weight: float = 16.0,
    place_fs_density: float = 0.4,
    fs_to_place_weight: float = 8.0,
    fs_to_place_density: float = 0.4,
    coincidence_threshold: int = 12,    # Route-D readout K (de-risk readout_weighted_k ~12)
    coincidence_train_k: float = 4.0,   # Route-D TRAIN count K (de-risk coincidence_k)
    coincidence_plateau: float = 80.0,  # Route-D plateau strength (de-risk value)
    n_place_bearing: int = 12,          # bearing sensors/landmark (de-risk n_bearing)
    n_place_dist: int = 8,              # distance sensors/landmark (de-risk n_dist)
    selforg_steps: int = 2000,          # total STEP-1 self-org sweep steps (de-risk validated)
    selforg_n_positions: int = 40,      # # agent positions swept during self-org
    reward_delay_steps: int = 8,        # online: hold place active before the SNc burst (Yagishita)
    place_sensor_max_intensity: float = 450.0,  # de-risk max_intensity
    place_sensor_falloff: float = 0.03,         # de-risk falloff
    place_sensor_dist_sigma: float = 4.0,       # de-risk dist_sigma
    place_sensor_bexp: float = 4.0,             # de-risk bexp
    stage_a_smoke: bool = False,        # N9 Stage-A cheap-first probe (FIRE/GRADED/ACTOR), exit pre-nav
    # ----- N9 Phase 3 STEP-2 value-training (pair-then-reward warm-up) + Stage-B smoke -----
    # (2026-06-10) Mirrors the VALIDATED de-risk STEP-2 (n9_place_graded_critic_stage2_derisk
    # run_seed, --pair-then-reward): on the FROZEN self-organized place fields, open the critic
    # arm (gate `value_input`) and run de-risk-style pair_then_reward trials at the scheduled
    # goal(s) so DA-gated STDP grows the NEAR place->striosome_value synapses (V learns). Each
    # trial: ITI floor (SNc tonic, no place, zero eligibility) -> PAIR (place + SNc TONIC,
    # pair_steps -> up-state + SILENT eligibility) -> REWARD (place + SNc BURST after
    # reward_delay_steps -> DA AFTER the pairing -> converts eligibility, the Yagishita timing)
    # -> reset g_gabab + SNc membrane. A sub-threshold phase-locked critic TEACHER
    # (critic_teacher_pa, de-risk --critic-teacher-pa 300) on striosome_value during the PAIR
    # phase ONLY (removed after) makes the weak-drive place volley fire the critic phase-locked.
    # Then `value_input` is frozen for the read-out / nav. Default value_train_trials=0 => OFF
    # (byte-equivalent; the place fields self-organize but V is never trained). Only meaningful
    # WITH neural_place_selforg; the legacy vs_place_context warm-up is unchanged.
    value_train_trials: int = 0,        # N9 STEP-2: pair-then-reward value-training trials per goal (de-risk 40)
    value_train_pair_steps: int = 100,  # PAIR-phase length (de-risk pair_steps; >= the up-state warm-up)
    value_train_hold_steps: int = 40,   # ITI / REWARD sub-phase length (de-risk hold_steps)
    critic_teacher_pa: float = 300.0,   # sub-threshold phase-locked teacher on striosome_value during PAIR (de-risk 300)
    value_train_stdp_w_max: float = 0.0,  # critic soft-bound ceiling DURING value-train (de-risk 40; 0=no override, keep nav's 150)
    stage_b_smoke: bool = False,        # N9 Stage-B probe (LEARNS-V / CRITIC FIRE+GRADE / GABA_B gap+lesion), exit pre-nav
    # ----- Critic value-acquisition WARM-UP (2026-06-09, the deadlock-breaker) -----
    # The 1800-step nav left the MSN-D1 critic SILENT (striov_rate_log all-zero, weight
    # frozen at 0.20). Forensic root cause (NOT the brief's "homeostasis too slow"): a
    # LTP-bootstrap DEADLOCK. At the init weight 0.20 the afferent fires ~12 Hz but the
    # critic cannot cross MSN-D1 threshold at ANY threshold value (threshold-sweep: 0 Hz
    # down to -54 mV; the per-region homeostasis lowers the threshold ~1 mV in 1800 steps,
    # nowhere near enough). With no critic spike there is no STDP post-event -> eligibility
    # stays exactly 0 -> reward cannot grow the weight -> the critic stays silent forever.
    # The de-risk PASSES only because it runs 40 CONCENTRATED reward-paired drives at ONE
    # location (the value-leads-reward protocol), which fires the critic enough to seed LTP
    # and grow the weight 0.20->0.58; a free-moving agent never gets that concentration.
    # Speeding up homeostasis (brief option 1) is NEGATIVE: a faster afferent adapt-rate
    # HOMOGENIZES the place code (place-selectivity ratio collapses to ~0.4, the gate-5
    # failure mode). Raising the init weight (option 2) gives a FLAT, non-graded V.
    # The faithful fix (brief option 3) is this WARM-UP: before the nav loop, run
    # `critic_warmup_trials` de-risk-style reward-paired drives at the scheduled goal
    # location(s) at the BASELINE homeostasis rate (which preserves place-selectivity, as
    # the de-risk's gate-5 PASS shows). This is the value system maturing on the rewarding
    # locations before the test (latent learning / pre-exposure) — the LTP, the critic
    # firing, the GABA_B subtraction are ALL neural; only the agent placement + reward
    # delivery is environment/body scaffolding (BRAIN-BASED-ONLY compliant). Validated in
    # isolation: 20 trials -> V_goal 2.08 Hz > V_far 1.25 Hz (place-graded, deadlock broken).
    # Default 0 => OFF (byte-equivalent to the pre-warmup runner; the smoke's silent critic).
    critic_warmup_trials: int = 0,
    critic_warmup_hold_steps: int = 40,        # steps per warm-up sub-phase (de-risk used 40)
    critic_warmup_all_goals: bool = True,      # warm up at EVERY scheduled goal (multi-goal)
                                               # vs only the first goal
    # Stage 2 windowed GABA_B (2026-06-08 redesign): gate the striosome_value->snc
    # GABA_B current to a bounded LEAD window into each reward evaluation so the
    # slow conductance pre-builds ~1 tau before reward but does NOT integrate
    # across a long dwell (the de-risk's >=200 ms over-suppression boundary,
    # d0416fc3). Default False => Stage 1: gate held OPEN continuously (isolate
    # "does it run + learn" before adding windowing).
    enable_critic_window: bool = False,
    critic_lead_steps: int = 120,        # bounded OPEN-window length (steps; dt=1ms)
    # GABA_B (GIRK) propagation strength onto the SNc (2026-06-10). The default 0.02 was tuned for a
    # ~20-30 Hz critic (seed 42); seeds 43/44 draw STRONGER place fields -> the critic fires 120 Hz ->
    # the slow GIRK over-accumulates and CLAMPS the SNc to 0 at BOTH near AND far (the differential
    # delta=r-V is hidden by saturation; the lesion still shows near<far). Lowering the propagation
    # de-saturates the GABA_B so the graded near<far subtraction is VISIBLE across the seed range
    # (Eshel-2015 arithmetic shift, not the all-or-none clamp). 0=keep the cfg default (0.02).
    critic_gabab_propagation: float = 0.0,
    critic_gabab_max: float = 0.0,   # cap g_gabab (finite GIRK channels) so a hot critic can't fully clamp the SNc (graded δ); 0=no cap
    # Surprise-boosted learning rate: when |RPE| is high (unexpected outcome),
    # temporarily boost reward_learning_rate. Models NE-like fast meta-modulation.
    enable_surprise_lr_boost: bool = False,
    surprise_lr_alpha: float = 2.0,  # max boost factor: 1 + alpha * |RPE|
    # Curriculum learning (Option B from plastic-input-layer arc):
    # In phase 1 (steps 0..curriculum_warmup_steps), suppress hippocampus drive
    # so the heuristic+WTA builds up cortex→D1 selectivity in isolation. Then
    # in phase 2, enable hippo drive — hippo plastic weights learn given that
    # cortex→D1 is already mature.
    #
    # Stage 3 (2026-04-27): real curriculum uses bridge plasticity_gate
    # infrastructure — cortex→D1 frozen at warmup, hippo→cortex thawed.
    # Stage 5 (2026-04-27): ramp_steps>0 enables smooth critical-period
    # closure: gate values interpolate linearly from phase-1 to phase-2
    # values over `ramp_steps` centered on warmup_steps. Biologically
    # grounded: real critical periods close gradually via PV interneuron
    # maturation (~weeks), not as instantaneous step functions. Smoother
    # transition reduces variance from abrupt cascade disruption.
    enable_curriculum: bool = False,
    curriculum_warmup_steps: int = 600,  # phase 1 length: cortex→D1 builds without hippo noise
    curriculum_ramp_steps: int = 0,      # 0 = abrupt step; >0 = smooth ramp window
    # Stage 5 (2026-04-27): partial freeze allows cortex→D1 to keep
    # adapting at reduced rate during phase 2. 0.0 = full freeze (default,
    # cortex locked); 1.0 = no freeze (combo A). Intermediate values let
    # cortex slowly track changing reward landscape while hippo learns
    # primary input mapping. Biologically: cortical plasticity doesn't
    # halt absolutely with maturation — it slows but persists, especially
    # under top-down attention or unexpected reward (DA-modulated).
    curriculum_phase2_cortex_gain: float = 0.0,
    curriculum_phase2_hippo_gain: float = 1.0,
    # Cheat #5 closure (2026-04-28): cross-projections (cortex_X → str_D1_Y / str_D2_Y
    # for X != Y) are tagged with a separate plasticity gate "corticostriatal_cross"
    # so the curriculum can stage them later than same-action pathways. The
    # naive approach (cross-projections on same gate as same-action) failed
    # 2026-04-27 because phase-0 motor activations reinforced cross-projections
    # to all D1 pools, locking in N/E motor bias before goal change.
    # Phase 3 thaws cross-projections AFTER goal change, when the agent has
    # experienced both regimes and STDP+reward can shape cross-action routing
    # symmetrically. -1 = stay frozen forever (default for safety).
    bg_cross_thaw_step: int = -1,
    # Plasticity gain for bg_cross_projections in phase 3. 1.0 = full plastic,
    # 0.5 = half-rate (slower than same-action), 0.0 = stay frozen.
    bg_cross_phase3_gain: float = 0.5,
    # ─── v4 (2026-04-28): developmental pretraining ────────────────────
    # Run a critical-period analog before the standard eval: N random
    # goals × M trials per goal with all plasticity gates open. At the
    # transition, the existing curriculum init naturally freezes
    # bg_cross_projections (line 1220 of this file). See
    # docs/plans/2026-04-28-cheat5-v4-design.md.
    enable_developmental_pretraining: bool = False,
    pretraining_n_goals: int = 10,
    pretraining_steps_per_goal: int = 3000,
    enable_structural_pruning: bool = False,
    # Cluster B.1 (2026-04-28): D1/D2 plasticity asymmetry — D2-targeting
    # synapses' weight updates flip sign vs D1. Default off.
    enable_d1_d2_asymmetry: bool = False,
    # Cluster B.2 (2026-04-28): striatal fast-spiking interneurons —
    # 4 str_PV_FSI_X pools providing broadcast inhibition to all D1/D2 MSN
    # pools. Default off. See
    # docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md.
    enable_striatal_fsis: bool = False,
    # Cluster B.3 (2026-04-28): cholinergic interneurons (TANs). Adds an
    # acetylcholine neuromodulator with the `pause_on_reward` rule that
    # transiently drops corticostriatal plasticity_window_gate on salient
    # reward events. Default off. See
    # docs/plans/2026-04-28-cluster-b3-tans-implementation.md.
    enable_tans: bool = False,
    enable_bg_neuropeptides: bool = False,  # R3.6: D1/D2 neuropeptide arms
    enable_cluster_a_closed_loop: bool = False,  # Cluster A: hyperdirect + thal->cortex
    enable_tonic_da: bool = False,  # Cluster C v1: dopamine as a real neuromodulator
    enable_compartmentalized_da: bool = False,  # Cluster C v2: per-action DA channels
    enable_cluster_d_hippocampus: bool = False,  # Cluster D v1: trisynaptic loop (ec+dg+ca3+ca1)
    enable_cluster_d_v2_swr: bool = False,  # Cluster D v2: SWR-gated CA3 plasticity (REQUIRES v1)
    enable_cluster_e_topography: bool = False,  # Cluster E v1: 2D coords + Gaussian-weighted cortex->striatum
    cluster_e_distance_sigma: float = 0.3,
    enable_cluster_f_cerebellum: bool = False,  # Cluster F v1: Marr-Albus cerebellar microcircuit
    n_granule: int = 250,  # Cerebellar granule cells (scaling test for F v2)
    # Cluster K v1 (2026-05-01): visual cortex hierarchy.
    # Adds retina (32x32 ON/OFF) → V1_simple → V1_complex → V2 → IT regions.
    # When True, the env step loop renders the gridworld as a 32x32 image and
    # drives the retina each step (before the stim window). v1 does NOT yet
    # wire IT → cortex_X for action selection — visual stream runs alongside
    # existing perception (heuristic / beacon / hippocampus / etc.) without
    # affecting motor output. Future v2: gated IT → cortex_X with curriculum.
    enable_visual_cortex: bool = False,
    visual_n_orientations: int = 8,
    visual_n_frequencies: int = 2,
    visual_n_positions_per_dim: int = 8,
    visual_image_size: int = 32,
    visual_n_v2: int = 256,
    visual_n_it: int = 64,
    visual_drive_max_pA: float = 200.0,
    # Spiking superior colliculus (N1 orienting; 2026-06-10)
    enable_spiking_sc: bool = False,
    n_spiking_sc_fs: int = 12,
    enable_spiking_sc_approach: bool = False,   # N5 Option C (neural approach-reward)
    # Cluster K v2 (2026-05-01)
    visual_receptive_field_radius: int = 4,
    visual_v1_weight_scale: float = 10.0,
    visual_it_to_cortex_density: float = 0.5,
    # Steps before IT -> cortex_X gate opens. Mimics critical-period
    # closure: V1/V2/IT mature first, then visuomotor wiring follows.
    # 0 = open from start (no critical period); -1 = stay closed forever
    # (visual cortex passive observer).
    visual_cortex_action_warmup_steps: int = 600,
    # Text I/O (2026-05-01): language_input + language_output regions for
    # bidirectional text training and dialogue. Driven externally via
    # bridge.set_token_drive() and read via bridge.read_language_output().
    # See sim/text_embeddings.py and docs/plans/2026-05-01-text-interaction-design.md.
    enable_text_io: bool = False,
    text_n_input_neurons: int = 256,
    text_n_output_neurons: int = 256,
    text_input_to_pfc_density: float = 0.20,
    text_input_to_pfc_weight: float = 2.0,
    text_input_to_cortex_density: float = 0.20,
    text_it_to_output_density: float = 0.20,
    # Tier 2.2 (2026-05-06): embodied-language during navigation.
    # Pulvermüller somatotopic semantics applied to the navigating
    # agent. When agent executes action a, drive language_input[word(a)] +
    # language_output[word(a)] simultaneously → STDP at lang↔motor
    # pathways binds word to action via embodied co-firing. When
    # agent perceives goal (within N cells), drive language_input["goal"] +
    # language_output["goal"] → STDP at lang↔IT pathways binds word
    # to visual concept. Same paradigm as Tier 1+2.1 but applied to
    # the navigating agent's perception/action stream. Requires
    # enable_text_io=True for the language regions.
    embodied_language: bool = False,
    embodied_language_drive_pA: float = 80.0,  # Lower than Tier 1's 200pA
                                                # because nav has competing
                                                # retina+BG drives. 200 was
                                                # disruptive (smoke result).
    embodied_language_goal_radius: int = 3,
    embodied_language_every_n_steps: int = 5,  # Sporadic drive (not every
                                                # step) — real biology pairs
                                                # language with experience
                                                # episodically, not at every
                                                # microsecond. Drive once
                                                # per N steps.
    embodied_language_warmup_steps: int = 600,  # Skip language drive until
                                                # nav has converged. Real
                                                # children hear words during
                                                # intentional action, not
                                                # random flailing. Bind only
                                                # to "successful" actions.
    # Cluster F v2 (2026-04-30): CF-gated anti-Hebbian LTD per Albus 1971
    # §IV.C eq.4. v1 used the global reward signal for PF→PC plasticity
    # (cerebellum and BG learned redundantly from the same signal). v2
    # decouples: cerebellum_pf_pc synapses see -1.0 only when IO is active
    # (CF event), 0.0 otherwise — global reward propagates only to non-
    # cerebellum synapses. Per Albus, cerebellum should ONLY weaken on
    # CF events, never strengthen on positive reward. Requires
    # enable_cluster_f_cerebellum=True. Default OFF.
    enable_cluster_f_v2: bool = False,
    # Structural-pruning hyperparameters (cheat-5 option-1, 2026-04-28).
    # Defaults match CoreSimConfig but can be overridden from the runner's
    # CLI / kwargs to tune the pruning aggressiveness for short pretraining
    # windows (e.g. smoke tests). None preserves the cfg default.
    pruning_alpha: float = None,
    pruning_threshold: float = None,
    pruning_weight_floor: float = None,
    # Heuristic decay (Stage 6, 2026-04-27): scales the heuristic cortex
    # drive (800 pA per aligned pool) by this factor. Default 1.0 keeps
    # full heuristic. Set to 0.0 to disable heuristic entirely (tests
    # whether learned hippo weights alone can navigate). Useful for
    # validating that hippo actually learned something vs. just being
    # along for the ride.
    heuristic_strength: float = 1.0,
    # Step at which heuristic_strength changes from heuristic_strength to
    # post_curriculum_heuristic_strength. -1 = no change (default).
    heuristic_decay_after_step: int = -1,
    post_curriculum_heuristic_strength: float = 0.0,
    # Critical-period developmental scaffold (N1, 2026-06-06). Instead of the
    # abrupt step-down of heuristic_decay_after_step, this LINEARLY ramps the
    # effective heuristic_strength from its base value down to 0 over the
    # window [heuristic_wean_start, heuristic_wean_start + heuristic_wean_steps],
    # then holds at 0. Biology: an innate scaffold (the heuristic teacher)
    # bootstraps the learned IT->cortex_X mapping during an early critical
    # period, then fades — the deployed weaned agent navigates from genuinely-
    # learned perception with NO heuristic. heuristic_wean_start = -1 (default)
    # disables the wean (unchanged behavior). When enabled, it takes precedence
    # over heuristic_decay_after_step.
    heuristic_wean_start: int = -1,
    heuristic_wean_steps: int = 1500,
    # ADAPTIVE / activity-gated weaning (N1, 2026-06-06). A FIXED critical-period
    # clock (heuristic_wean_start) is NOT robust across seeds — the post-wean hold
    # is seed-dependent and non-monotonic (more teaching can HURT; see
    # research/findings/2026-06-06-N1-critical-period-scaffold-TRACTABLE.md). Real
    # critical periods close when the circuit is READY (Hensch — activity/
    # maturation-gated), not on a clock. With heuristic_wean_adaptive=True the
    # runner PROBES readiness online: every wean_probe_every steps it turns the
    # heuristic OFF for a short probe window (wean_probe_window steps) and measures
    # the agent's mean distance to goal during the probe. If the learned mapping
    # navigates self-sufficiently (mean probe distance <= wean_probe_threshold),
    # it COMMITS — weans the heuristic off permanently by ramping to 0 over
    # heuristic_wean_steps from the commit step. Otherwise the heuristic turns back
    # ON (keep teaching) until the next probe. This adapts to each seed's sweet
    # spot and avoids over-training. Takes precedence over the fixed-clock wean
    # (heuristic_wean_start) when both are set. Default OFF (unchanged behavior).
    heuristic_wean_adaptive: bool = False,
    wean_probe_every: int = 500,
    wean_probe_window: int = 200,
    wean_probe_threshold: float = 2.5,
    # Sleep-replay memory consolidation (Stage 7, 2026-04-27).
    # During sleep phases: no external goal, hippo cells fire in random
    # replay patterns (modeling NREM sharp-wave ripples), corticostriatal
    # is thawed (consolidation), hippo_to_cortex is frozen (preserve
    # learned weights). The replayed hippo signal drives cortex pools
    # via the learned hippo→cortex weights, and STDP between cortex_X
    # and D1_X consolidates the pattern into the cortex→D1 cascade.
    # After sleep, the cortex→D1 weights should encode hippo's learned
    # mapping, enabling navigation with reduced hippo dependency.
    # Biologically: episodic→semantic memory consolidation during NREM.
    # -1 = no sleep replay (default).
    sleep_replay_after_step: int = -1,
    sleep_replay_steps: int = 300,
    sleep_replay_rate_hz: float = 200.0,  # high rate (sharp-wave ripples)
    # NREM/REM stages (Item 7, 2026-04-27). When sleep_nrem_rem_alternate=True,
    # the sleep period alternates between NREM (trajectory replay, slow ripples)
    # and REM (random replay, faster). NREM cycle dominates first half, REM
    # second half, modeling sleep-stage progression.
    sleep_nrem_rem_alternate: bool = False,
    # Reverse-order trajectory replay during NREM (2026-04-30). Real CA1/CA3
    # ripples replay trajectories in reverse time order during NREM (Foster
    # & Wilson 2006, Diba & Buzsaki 2007). When enabled, the runner indexes
    # the successful_trajectories buffer from newest-to-oldest by sleep step
    # index instead of random sampling. Biologically grounded as TD-style
    # backward credit assignment. Default off — backward compatible.
    enable_reverse_replay: bool = False,
    # Hindsight Experience Replay (Andrychowicz 2017). Logs
    # (old_pos, current_pos) tuples to successful_trajectories every
    # `her_lag_steps`, treating the achieved position as if it had been
    # the goal. Provides hindsight credit assignment for sparse-goal
    # generalization. Default off.
    enable_her: bool = False,
    # Recency-weighted replay (2026-04-30): exponential bias toward newest
    # successful_trajectories during NREM. Addresses the "stale replay"
    # bottleneck flagged in SCIENCE_ROADMAP §4.7 (older entries are from
    # goals that no longer apply). Default off.
    enable_recency_weighted_replay: bool = False,
    # 2026-04-30 probe: when True, heuristic drives only ONE cortex pool
    # (random choice among manhattan-reducing directions) instead of all
    # valid directions. Matches g11_bg_replicated_runner's heuristic.
    # Investigating whether this is the source of the replicated-vs-single
    # discrepancy.
    heuristic_single_pool: bool = False,
    # PFC Stage 2: delayed-response test. Silence goal_cells during a delay
    # window to test whether PFC maintains goal info via persistent activity.
    # If PFC works as working memory, agent should still navigate toward goal
    # during the silence period (PFC remembers). Without PFC, agent should
    # drift (no goal info available).
    goal_silence_after_step: int = -1,
    goal_silence_duration: int = 0,
    # N8 cheat conversion (2026-06-06): genuine GPi->thalamus disinhibition.
    # The default tonic regime drives thal_X with a direct 300 pA current (N8)
    # and gpi_X with only 110 pA — the thalamic relay is externally PACED, so
    # the BG cascade's output gate is short-circuited (selection happens
    # upstream but the thalamus it should gate is paced regardless). When
    # genuine_thal_disinhibition=True, the thalamic relay is RELEASED by a real
    # direct-pathway cascade instead: GPi is a strong tonic pacemaker
    # (genuine_gpi_tonic_pA) that silences thal_X by default; the selected
    # action's cortex drive -> D1 -> (GABA) GPi silence -> thal_X DISINHIBITED.
    # thal_X carries only a tonic excitation (genuine_thal_tonic_pA) it can
    # express ONLY when its GPi is released. The cortex->D1->gpi->thal->motor
    # pathways already exist at the validated weight scale (D1->GPi=15,
    # GPi->thal=8); this flag only changes the gpi/thal DRIVES and removes the
    # direct thal pacing. Ported from gated_compose_bg_genuine_demo.py
    # (Logiaco-Abbott-Escola 2021; Kandel ch 38 direct-pathway "go").
    # Default OFF = the tonic-drive cheat stays runnable as the CONTROL.
    genuine_thal_disinhibition: bool = False,
    genuine_gpi_tonic_pA: float = 1000.0,   # tonic GPi pacemaker drive (silences thal by default)
    genuine_thal_tonic_pA: float = 900.0,   # tonic thalamic excitation (expressed only when GPi releases)
    # N6 readout-source (2026-06-06): which spiking pool the host argmax reads
    # for action selection. "motor" = legacy (host argmax over motor_X spike
    # counts; the N6 cheat). "thal" = read the cleanly-selective THALAMUS
    # (argmax over thal_X spike counts). Under genuine GPi->thal disinhibition
    # the thalamus is the cleanest, strongest selection signal (only the
    # released action's thal fires, others ~0); the thal->motor labeled-line
    # amplification is too weak for a reliable motor-count argmax over a noisy
    # multi-goal run. Reading thal is the combined N8+N6 fix's cheap test:
    # "is the weak motor SIGNAL the whole problem?". Default "motor"
    # (backward-compatible). See research/findings/2026-06-06-N8N6-*.
    # DEFAULT-ON 2026-06-19 (CYCLE 235): the fully-spiking commit-burst decision is now the LIBRARY default
    # (the merged "one brain" navigates fully-spiking) — validated 6-seed grid-32 at 1.16x host with 100%
    # commit-burst (2026-06-19-spiking-decision-default-on-GO.md). "motor"/"thal" = the opt-in host-argmax
    # ORACLE; the CLI --readout-source still defaults to "motor" so the documented standalone benchmarks
    # reproduce unchanged. The tuned levers below (sel_recurrent_weight 0.3 + n_sel/n_commit 40) are the
    # cost-reduction winners and are inert under "motor"/"thal" (the sel/commit layer is only built for spiking_wta).
    readout_source: str = "spiking_wta",
    # TRN-style thalamic lateral inhibition (2026-06-06, N8+N6) — a biological
    # WTA on the thalamic relay (the clean genuine-disinhibition signal). See
    # build_bg_brain_regions for the mechanism. Combine with readout_source="thal".
    enable_thal_lateral_inhibition: bool = False,
    n_thal_fs_per_action: int = 5,
    thal_to_fs_weight: float = 50.0,
    # Spiking action-selection WTA readout (2026-06-06, N6 biologization). A
    # dedicated read-only sel_X / sel_FS_X selection layer driven by the clean
    # thalamus; the action decision EMERGES from the spiking competition rather
    # than a host argmax. Built when readout_source="spiking_wta". See
    # build_bg_brain_regions for the mechanism + why it differs from the prior
    # (failed) motor/TRN WTAs.
    n_sel_per_action: int = 40,   # DEFAULT-ON 2026-06-19: N-scaled accumulator pool (finite-size-noise lever, 20->40 -> 1.16x host)
    n_sel_fs_per_action: int = 10,
    thal_to_sel_weight: float = 30.0,
    sel_to_sel_fs_weight: float = 20.0,
    sel_fs_to_sel_weight: float = 5.0,
    # Accumulate-then-commit readout (2026-06-06, N6 fix). See
    # build_bg_brain_regions for the full mechanism (Wang-2002 NMDA recurrent
    # accumulator on sel_X + Lo-Wang/SC commit_X burst stage). These tune the
    # recurrent gain (alpha<1 soft-WTA), the commit threshold, and the OPN
    # tonic inhibition. Active only when readout_source="spiking_wta".
    sel_recurrent_density: float = 0.5,
    sel_recurrent_weight: float = 0.3,   # DEFAULT-ON 2026-06-19: leak/forgetting (Usher-McClelland; 1.0->0.3 cuts the cross-trial NMDA hysteresis = the dominant cost)
    enable_commit_burst: bool = True,
    n_commit_per_action: int = 40,   # DEFAULT-ON 2026-06-19: N-scaled commit pool (grown in lockstep with n_sel_per_action)
    n_commit_opn: int = 20,
    sel_to_commit_weight: float = 22.0,
    commit_recurrent_density: float = 0.5,
    commit_recurrent_weight: float = 0.6,
    opn_to_commit_weight: float = 10.0,
    commit_opn_tonic_pA: float = 0.0,
    # Reset the accumulator (sel_X + commit_X NMDA + fast conductances) at the
    # START of each trial. Motivation: the NMDA-slow accumulator (tau_decay=100ms)
    # persists ~one full inter-trial, so at goal-change boundaries the previous
    # trial's winner lingers (working-memory hysteresis). HOWEVER, empirically
    # (grid-8 multi-goal seed 42) the reset is NET NEGATIVE: zeroing the NMDA
    # state each trial removes the carried-over drive that helps the burst fire,
    # so commit goes silent on ~55% of trials (vs ~34% un-reset) and the score
    # WORSENS (6.93 reset vs 4.71 un-reset). The cross-trial persistence is a
    # smaller cost than the lost ramp. So default FALSE; the persistence (a
    # working-memory latch, biologically real) is kept. Opt in with
    # --reset-accumulator for the goal-change-hysteresis ablation. NO sim/ edit.
    reset_accumulator_each_trial: bool = False,
    # N6 refinement 1 (2026-06-06): LOSER-ONLY accumulator reset. Zero the NMDA +
    # fast conductances on every sel_X (+commit_X) pool EXCEPT the previous trial's
    # selected action each trial. Surgical hysteresis removal: the winner's
    # working-memory latch persists (fast re-ramp when the goal is stable) while the
    # three losers integrate fresh evidence, so at a goal change the stale old winner
    # decays naturally instead of out-competing the new thal evidence. Unlike the
    # naive ALL-reset (reset_accumulator_each_trial, NET NEGATIVE → 6.93) the eventual
    # winner is only ever reset while it is a loser, so its eventual ramp is never
    # zeroed. Mutually exclusive with reset_accumulator_each_trial (the all-reset
    # takes precedence if both set). Active only under readout_source="spiking_wta".
    # NO sim/ edit. (Cisek trial-wise re-baselining of the losing options.)
    # EMPIRICAL (grid-8 multi-goal seed 42, 2026-06-06): NET NEGATIVE alone (SUM
    # 5.58 vs 4.71 baseline — it preserves the PREVIOUS-decision winner's latch, so
    # at a goal change the stale winner is kept while the new winner — among the
    # freshly-reset losers — integrates slower; random fallback jumps to 25.7%) and
    # adds no net lift combined with urgency (4.35 combined vs 4.08 urgency-alone:
    # phase 2 regresses 1.43→2.02). Kept opt-in; NOT recommended.
    reset_losers_only: bool = False,
    # N6 refinement 2 (2026-06-06): CISEK URGENCY / collapsing decision bound. Peak
    # (pA) of a ramping action-INDEPENDENT urgency current injected into all sel_X
    # over the readout window (0 at readout_start → urgency_max_pA at readout_end).
    # The effective commit bound collapses with elapsed time so even a weak
    # late-phase release crosses within the 100ms window → the commit bursts
    # (eliminates the silent-commit → host-argmax-residual fallback). Same drive for
    # every pool → no action bias; only the time-to-cross shrinks (Cisek 2009;
    # Thura-Cisek 2014; Lo-Wang 2006 DA-modulated bound). Default 0 = OFF. Active
    # only under readout_source="spiking_wta". NO sim/ edit.
    # EMPIRICAL (grid-8 multi-goal seed 42, 2026-06-06): the BEST refinement. At
    # urgency_max_pA=180 the cheat-5 SUM improves 4.71→4.08, the random fallback is
    # nearly eliminated (25%→1.4%), thal-winner alignment jumps 80%→94.8% and commit
    # separation 15×→49× (the decision is MORE decisive + accurate, not a quiet
    # argmax lean). Phases 0-1 reach the host-argmax reference (0.60/0.50); the
    # residual cost stays in the goal-change phases 2-3 (cross-trial NMDA hysteresis,
    # which the loser-only reset does not fix). RECOMMENDED value: 180.
    urgency_max_pA: float = 180.0,   # DEFAULT-ON 2026-06-19: the Cisek collapsing bound (REQUIRED for spiking_wta -> 100% commit-burst, no silent-commit fallback)
    thal_fs_to_thal_weight: float = 20.0,
    # ── Live brain-activity streaming (frontend-revamp Phase 1, 2026-06-08) ──
    # docs/plans/2026-06-08-frontend-revamp-design.md §3.4. Default OFF: when
    # emit_activity is False, the RegionActivityProbe is never constructed and
    # emit_activity() is never called, so the step loop is byte-identical and
    # every multi-seed / determinism / science run is unaffected (zero overhead).
    # When True, build the probe ONCE and emit a throttled [ACTIVITY] {json}
    # line every `emit_activity_every` steps (fire-and-forget to stdout -> the
    # run's .log; the webapp tails it). The probe is a tiny host-side per-region
    # mean-firing reduction (~30 floats), NOT per-neuron, so it never bottlenecks
    # the sim. Requires the brain-region framework (region_manager) — always on
    # for the BG nav cascade.
    emit_activity: bool = False,
    emit_activity_every: int = 5,
    # ── STEP 2a merge integration: additive override for the STDP soft-bound
    # ceiling (cfg.stdp_w_max). Default None = the existing computed value
    # (max(30, 25/0.20*1.2) = 150), so the standalone nav path is BYTE-EQUIVALENT
    # when this is unused. The merge passes 400 (the 5a clip mitigation — raised
    # above the ~300 frozen parser role-route so the ungated reward clip cannot
    # move it). The nav-gate (a) check verifies the soft-bound nav actor
    # (cortex->D1) does NOT over-grow when the ceiling is raised. NO sim/ edit.
    stdp_w_max_override=None,
    # STEP 2a merge integration (additive; ALL default no-op => standalone nav BYTE-EQUIVALENT).
    # extra_regions/extra_pathways: conversational regions/pathways appended to the nav lists so they share
    #   ONE bridge. build_with_ou: build with OU on so the parser train pass's OU per-neuron state is allocated
    #   at init (a runtime toggle does NOT allocate it). prebuilt_post_init_hook(bridge): called AFTER the
    #   Gabor/SC post-init wiring (which rebuilds the CSR) and BEFORE the episode loop, to train+freeze the
    #   conversational populations on the merged bridge. See research/runners/nav_conv_merged_bridge.py.
    extra_regions=None,
    extra_pathways=None,
    prebuilt_post_init_hook=None,
    build_with_ou=False,
):
    """Phase B acid test: run BG circuit on G9-style moving-goal scenario.

    If the BG architecture dissolves the silent-motor trap (which V1-V7
    runner-side interventions all failed to do), phase 1 finalQ should
    drop substantially below the G9 baseline of 6.74.
    """
    # v4 (2026-04-28): conflict check. v4 keeps cross-projections frozen
    # during eval; v3.1 thaws them at bg_cross_thaw_step. Both at once is
    # meaningless. Fail loud instead of silent priority resolution.
    if enable_developmental_pretraining and bg_cross_thaw_step >= 0:
        raise ValueError(
            "--developmental-pretraining (v4) is incompatible with "
            "--bg-cross-thaw-step (v3.1). v4 keeps cross-projections frozen "
            "throughout eval; v3.1 thaws them mid-eval. Use one or the other, "
            f"not both. Got bg_cross_thaw_step={bg_cross_thaw_step}."
        )
    if enable_developmental_pretraining and not enable_bg_cross_projections:
        print(
            "[g11 warning] --developmental-pretraining without "
            "--enable-corticostriatal-cross: pretraining will run but won't shape any "
            "corticostriatal_cross gate (no cross pathways exist). Did you "
            "mean to also pass --enable-corticostriatal-cross "
            "(or its legacy alias --bg-cross-projections)?",
            flush=True,
        )
    import cupy as cp
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel

    if goal_schedule is None:
        goal_schedule = [(0, tuple(goal_pos))]
    goal_schedule_sorted = sorted(
        [(int(s), tuple(g)) for s, g in goal_schedule], key=lambda t: t[0]
    )

    # N6 readout-source normalization. Validated up-front (before wiring) so
    # the builder can conditionally add the spiking-WTA selection layer.
    _readout_source = str(readout_source).lower()
    if _readout_source not in ("motor", "thal", "spiking_wta"):
        raise ValueError(
            "readout_source must be 'motor', 'thal', or 'spiking_wta', "
            f"got {readout_source!r}")

    # N9 nav-deployment HARD-GATE (design §4 BRAIN-BASED-ONLY audit): when the neural place-code
    # self-org path is active, the position-blind convergent up-state (A1 floor) is DROPPED — the
    # de-risk found it caps value-grading ~1.2x. enable_convergent_upstate is forced OFF for the
    # rest of run_g11 (build + warm-up + nav drive) when neural_place_selforg is on.
    _neural_place_selforg = bool(neural_place_selforg and enable_neural_critic)
    if _neural_place_selforg and enable_convergent_upstate:
        enable_convergent_upstate = False
        if verbose:
            print("[g11] N9 neural_place_selforg ON -> enable_convergent_upstate HARD-GATED OFF "
                  "(the position-blind A1 floor caps grading ~1.2x).", flush=True)
    # Spiking US -> SNc coord-free guard: the reward_us afferent rides on the PERCEIVED reward.
    # WITHOUT --perceived-approach-reward, `reward` is the coord-touched default -> the spiking US is
    # then driven by a coordinate-derived signal (a documented residual shortcut, NOT coord-free).
    if spiking_reward_us and not perceived_approach_reward and verbose:
        print("[g11] WARN: --spiking-reward-us WITHOUT --perceived-approach-reward -> the US rides "
              "on the COORD-based reward (not coord-free). Add --perceived-approach-reward for the "
              "fully-biologized (coord-free) reward chain.", flush=True)

    regions, pathways = build_bg_brain_regions(
        n_cortex=100,  # 25 per action — keeps D1 firing in physiological range (~75 Hz)
        enable_motor_lateral_inhibition=enable_motor_lateral_inhibition,
        enable_thal_lateral_inhibition=enable_thal_lateral_inhibition,
        n_thal_fs_per_action=n_thal_fs_per_action,
        thal_to_fs_weight=thal_to_fs_weight,
        thal_fs_to_thal_weight=thal_fs_to_thal_weight,
        enable_spiking_wta_readout=(_readout_source == "spiking_wta"),
        n_sel_per_action=n_sel_per_action,
        n_sel_fs_per_action=n_sel_fs_per_action,
        thal_to_sel_weight=thal_to_sel_weight,
        sel_to_sel_fs_weight=sel_to_sel_fs_weight,
        sel_fs_to_sel_weight=sel_fs_to_sel_weight,
        sel_recurrent_density=sel_recurrent_density,
        sel_recurrent_weight=sel_recurrent_weight,
        enable_commit_burst=enable_commit_burst,
        n_commit_per_action=n_commit_per_action,
        n_commit_opn=n_commit_opn,
        sel_to_commit_weight=sel_to_commit_weight,
        commit_recurrent_density=commit_recurrent_density,
        commit_recurrent_weight=commit_recurrent_weight,
        opn_to_commit_weight=opn_to_commit_weight,
        enable_cortex_lateral_inhibition=enable_cortex_lateral_inhibition,
        enable_learned_perception=enable_learned_perception,
        enable_hippocampus=enable_hippocampus,
        n_hippocampus_per_layer=n_hippocampus_per_layer,
        sensory_to_cortex_weight=sensory_to_cortex_weight,
        hippocampus_to_cortex_weight=hippocampus_to_cortex_weight,
        enable_pfc=enable_pfc,
        n_pfc=n_pfc,
        pfc_internal_density=pfc_internal_density,
        goal_to_pfc_weight=goal_to_pfc_weight,
        pfc_to_cortex_weight=pfc_to_cortex_weight,
        pfc_enable_nmda=enable_pfc_nmda,
        enable_bg_cross_projections=enable_bg_cross_projections,
        cross_projection_weight=cross_projection_weight,
        cross_projection_density=cross_projection_density,
        cross_projection_topology_seed=cross_projection_topology_seed,
        enable_bg_lateral_inhibition=enable_bg_lateral_inhibition,
        lateral_inhibition_density=lateral_inhibition_density,
        lateral_inhibition_weight=lateral_inhibition_weight,
        enable_striatal_fsis=enable_striatal_fsis,
        enable_cluster_a_closed_loop=enable_cluster_a_closed_loop,
        enable_cluster_d_hippocampus=enable_cluster_d_hippocampus,
        enable_cluster_d_v2_swr=enable_cluster_d_v2_swr,
        enable_cluster_e_topography=enable_cluster_e_topography,
        cluster_e_distance_sigma=cluster_e_distance_sigma,
        enable_cluster_f_cerebellum=enable_cluster_f_cerebellum,
        n_granule=n_granule,
        enable_visual_cortex=enable_visual_cortex,
        enable_spiking_sc=enable_spiking_sc,
        n_spiking_sc_fs=n_spiking_sc_fs,
        enable_spiking_sc_approach=enable_spiking_sc_approach,
        # Spiking-SNc actor-critic Stage B: the neural value critic (2026-06-08).
        enable_neural_critic=enable_neural_critic,
        spiking_reward_us=spiking_reward_us,
        n_reward_us=n_reward_us,
        reward_us_to_snc_weight=reward_us_to_snc_weight,
        reward_us_to_snc_density=reward_us_to_snc_density,
        # N9 NEURAL PLACE-CODE SELF-ORG (2026-06-09 nav deployment) — the self-organized spiking
        # place code afferent + FS-PING + coincidence critic (replaces the host vs_place_context).
        neural_place_selforg=_neural_place_selforg,
        n_place=n_place,
        n_place_fs=n_place_fs,
        place_sensors_to_place_weight=place_sensors_to_place_weight,
        place_sensors_to_place_density=place_sensors_to_place_density,
        place_sensors_to_place_jitter=place_sensors_to_place_jitter,
        enable_critic_fs_inhibition=enable_critic_fs_inhibition,
        critic_fs_weight=critic_fs_weight,
        critic_fs_density=critic_fs_density,
        place_fs_weight=place_fs_weight,
        place_fs_density=place_fs_density,
        fs_to_place_weight=fs_to_place_weight,
        fs_to_place_density=fs_to_place_density,
        coincidence_threshold=coincidence_threshold,
        coincidence_train_k=coincidence_train_k,
        coincidence_plateau=coincidence_plateau,
        n_place_bearing=n_place_bearing,
        n_place_dist=n_place_dist,
        # Critic drive calibration (2026-06-08): raised place-afferent weight so the
        # learned value-of-location is well-graded (the MSN-D1->RS type swap is applied
        # to the returned region below, after build, to keep the build signature stable).
        critic_cortex_it_to_value_weight=critic_afferent_weight,
        critic_cortex_it_to_value_density=critic_afferent_density,
        # 2026-06-09 VALIDATED redesign: the dense dedicated afferent + per-region homeostasis.
        enable_critic_homeostasis=enable_critic_homeostasis,
        n_vs_place_context=n_vs_place_context,
        vs_place_to_value_weight=vs_place_to_value_weight,
        vs_place_to_value_density=vs_place_to_value_density,
        enable_convergent_upstate=enable_convergent_upstate,
        vs_place_drive_to_value_weight=vs_place_drive_to_value_weight,
        vs_place_drive_to_value_density=vs_place_drive_to_value_density,
        visual_n_orientations=visual_n_orientations,
        visual_n_frequencies=visual_n_frequencies,
        visual_n_positions_per_dim=visual_n_positions_per_dim,
        visual_image_size=visual_image_size,
        visual_n_v2=visual_n_v2,
        visual_n_it=visual_n_it,
        visual_it_to_cortex_density=visual_it_to_cortex_density,
        enable_beacon_perception=enable_beacon_perception,
        n_beacon_sensors=n_beacon_sensors,
        beacon_to_goal_weight=beacon_to_goal_weight,
        enable_landmarks=enable_landmarks,
        n_landmark_sensors=n_landmark_sensors,
        landmark_to_place_weight=landmark_to_place_weight,
        enable_text_io=enable_text_io,
        text_n_input_neurons=text_n_input_neurons,
        text_n_output_neurons=text_n_output_neurons,
        text_input_to_pfc_density=text_input_to_pfc_density,
        text_input_to_pfc_weight=text_input_to_pfc_weight,
        text_input_to_cortex_density=text_input_to_cortex_density,
        text_it_to_output_density=text_it_to_output_density,
    )

    # Critic neuron-type calibration (2026-06-08, runner-side; NO sim/ edit). The
    # smoke found the MSN-D1 striosome_value critic silent in nav: its depolarized
    # rheobase (~700 pA) is built for the cortical up-state and the sparse place
    # code (sensor_place_readout, ~3-8 Hz) can't reach it through the afferent at
    # ANY weight. A more excitable type fires from the place code (diag4: RS gives
    # 5-20 Hz, graded by drive x weight) and carves a teacher-free value-of-location
    # (diag6). Applied by mutating the returned BrainRegion BEFORE the bridge is
    # built (the build signature is untouched). None => keep the MSN-D1 default
    # (byte-equivalent to the prior behavior).
    if enable_neural_critic and critic_neuron_type:
        for _r in regions:
            if _r.name == "striosome_value":
                _r.izh_neuron_type = str(critic_neuron_type)
                break

    # Pre-compute sensory neuron preferred (dx, dy) — 7x7 grid covering [-3, 3]²
    if enable_learned_perception:
        sensory_pref = []
        for iy in range(7):
            for ix in range(7):
                sensory_pref.append((ix - 3, iy - 3))  # dx, dy ∈ [-3, 3]
        sensory_pref_dx = np.array([p[0] for p in sensory_pref], dtype=np.float32)
        sensory_pref_dy = np.array([p[1] for p in sensory_pref], dtype=np.float32)
    else:
        sensory_pref_dx = None
        sensory_pref_dy = None

    # Pre-compute hippocampal cell preferred (x, y) — covering full grid.
    # Layout: square grid of side = ceil(sqrt(n_hippocampus_per_layer)) with
    # cells spaced to span the full grid range. For 8×8 grid with 64 cells,
    # one cell per position. For 16×16 grid with 256 cells, also one per
    # position. For mismatched cases, cells space out uniformly.
    if enable_hippocampus:
        side = int(round(n_hippocampus_per_layer ** 0.5))
        scale = (grid_size - 1) / max(1, side - 1) if side > 1 else 1.0
        hippo_pref_x = np.array([(i % side) * scale for i in range(n_hippocampus_per_layer)], dtype=np.float32)
        hippo_pref_y = np.array([(i // side) * scale for i in range(n_hippocampus_per_layer)], dtype=np.float32)
    else:
        hippo_pref_x = None
        hippo_pref_y = None

    # Pre-compute the DEDICATED DENSE `vs_place_context` afferent's preferred (x,y) tiling +
    # its wide place-code sigma (2026-06-09 VALIDATED redesign). Mirrors the de-risk probe's
    # _grid_prefs: a near-square sub-grid of side=round(sqrt(N)) tiling [0,grid_size)^2,
    # padded/truncated to exactly N cells. The wide sigma (grid_size/8 => 4.0 at grid-32, the
    # de-risk's validated value) makes 30-80 cells fire per location (the convergent-excitation
    # up-state), UNLIKE the actor's narrow hippocampus_drive_sigma=0.5 (~1-3 cells). Drive-
    # injected each nav step (see the nav loop). Built only when the neural critic is on.
    if enable_neural_critic:
        _vs_side = int(round(n_vs_place_context ** 0.5))
        _vs_xs = np.linspace(0.0, grid_size - 1.0, _vs_side, dtype=np.float32)
        _vs_ys = np.linspace(0.0, grid_size - 1.0, _vs_side, dtype=np.float32)
        _vs_gx, _vs_gy = np.meshgrid(_vs_xs, _vs_ys)
        _vs_px = _vs_gx.ravel(); _vs_py = _vs_gy.ravel()
        if _vs_px.size < n_vs_place_context:
            _reps = int(np.ceil(n_vs_place_context / max(_vs_px.size, 1)))
            _vs_px = np.tile(_vs_px, _reps)[:n_vs_place_context]
            _vs_py = np.tile(_vs_py, _reps)[:n_vs_place_context]
        vs_place_pref_x = _vs_px[:n_vs_place_context].copy()
        vs_place_pref_y = _vs_py[:n_vs_place_context].copy()
        vs_place_sigma = float(grid_size) / 8.0       # 4.0 at grid-32 (de-risk validated)
        vs_place_drive_max_pA = 800.0                 # de-risk validated drive
    else:
        vs_place_pref_x = None
        vs_place_pref_y = None
        vs_place_sigma = None
        vs_place_drive_max_pA = None

    # ── N9 place-sensor egocentric render precompute (the legitimate body-sensing channel). The 3
    #    fixed landmarks + dist_max are grid-tuned; the per-step render is _n9_place_sensor_act
    #    (VERBATIM the de-risk landmark_sensor_act). (x,y) enters the brain ONLY through this. ──
    if _neural_place_selforg:
        _n9_landmarks = _n9_place_landmarks(grid_size)
        _n9_dist_max = float(grid_size) * 1.42        # de-risk dist_max (diag of the grid)

        def _n9_render(px, py):
            return _n9_place_sensor_act(
                px, py, _n9_landmarks, int(n_place_bearing), int(n_place_dist),
                float(place_sensor_max_intensity), float(place_sensor_falloff),
                float(place_sensor_dist_sigma), _n9_dist_max, float(place_sensor_bexp))
    else:
        _n9_landmarks = None
        _n9_dist_max = None
        _n9_render = None

    # Pre-compute beacon sensor preferred directions (Item 1 Stage 1).
    # Sensors evenly distributed in 2D — for n=8: N, NE, E, SE, S, SW, W, NW.
    # Each sensor responds maximally when beacon is in its preferred direction
    # (cosine alignment), with intensity falling off with distance.
    # Models biological directional cue detection (e.g., bilateral hearing
    # estimating sound source direction from intensity differences).
    if enable_beacon_perception:
        beacon_pref_x = np.zeros(n_beacon_sensors, dtype=np.float32)
        beacon_pref_y = np.zeros(n_beacon_sensors, dtype=np.float32)
        for i in range(n_beacon_sensors):
            angle = 2.0 * np.pi * i / n_beacon_sensors
            beacon_pref_x[i] = np.cos(angle)
            beacon_pref_y[i] = np.sin(angle)
    else:
        beacon_pref_x = None
        beacon_pref_y = None

    # Pre-compute landmark sensor preferred directions (Item 1 Stage 2).
    # Same structure as beacon sensors; landmark is at fixed position.
    if enable_landmarks:
        landmark_pref_x = np.zeros(n_landmark_sensors, dtype=np.float32)
        landmark_pref_y = np.zeros(n_landmark_sensors, dtype=np.float32)
        for i in range(n_landmark_sensors):
            angle = 2.0 * np.pi * i / n_landmark_sensors
            landmark_pref_x[i] = np.cos(angle)
            landmark_pref_y[i] = np.sin(angle)
        # Default landmark position: grid center
        if landmark_position is None:
            landmark_position = (grid_size / 2.0, grid_size / 2.0)
    else:
        landmark_pref_x = None
        landmark_pref_y = None

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = float(learning_rate)
    cfg.reward_eligibility_tau_ms = float(reward_eligibility_tau_ms)
    # cortex->D1 weight_mean needs w_max above that or soft-bound STDP collapses it.
    # When R3.5's density reduction triggers weight scaling (e.g. weight=125 at
    # density=0.20), w_max must be ABOVE that — otherwise LTP events drive weights
    # negative, collapsing the cascade silently. Recompute the post-R3.5 weight
    # locally (mirrors build_bg_brain_regions logic).
    _ctx_msn_density = 0.20  # R3.5 default
    _ctx_msn_weight = (25.0 / _ctx_msn_density) if _ctx_msn_density < 1.0 else 25.0
    cfg.stdp_w_max = max(30.0, _ctx_msn_weight * 1.2)
    # STEP 2a merge integration: optional additive override of the STDP soft-bound
    # ceiling. Default None => keep the computed value above (byte-equivalent
    # standalone). The merge passes 400 (5a clip mitigation); the nav-gate (a)
    # check uses --stdp-w-max 400 to verify the soft-bound actor does not over-grow.
    if stdp_w_max_override is not None:
        cfg.stdp_w_max = float(stdp_w_max_override)
    # STEP 2a merge: append the conversational regions/pathways (defaults None => no-op; standalone unchanged).
    # Mutate IN PLACE so cfg.brain_regions / cfg.region_pathways (already bound to these list objects above)
    # see the additions before _initialize_simulation_data reads them.
    if extra_regions:
        regions.extend(extra_regions)
    if extra_pathways:
        pathways.extend(extra_pathways)
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    # STEP 2a merge: build with OU on so the parser train pass's OU per-neuron state is ALLOCATED at init
    # (a runtime toggle does not allocate it). The post-init hook sets OU off for the nav episode after the
    # parser pass. Default False => standalone nav byte-equivalent (OU stays off).
    cfg.enable_ou_process = bool(build_with_ou)
    cfg.ou_std_current_pA = 20.0 if build_with_ou else cfg.ou_std_current_pA
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False  # keep synapse count fixed (per-action DA mask depends on it)
    # Spiking-SNc actor-critic Stage B (2026-06-08): the neural value critic's
    # critic->SNc projection (built with receptor="gaba_b") subtracts V via the
    # slow GABA_B/GIRK K+ conductance (E_K=-90mV) — strong + sign-correct on the
    # KCC2-lacking depolarized SNc, where weak GABA_A failed. Default OFF =>
    # byte-identical when the critic is not enabled (the protected sim/ GABA_B
    # support, commit a7370d49, is inert unless cfg.enable_gabab is set AND a
    # pathway is tagged receptor="gaba_b"). Note: cfg.stdp_w_max above (=150) is
    # already far above the critic's working weight range, so the soft-bound STDP
    # collapse (CLAUDE.md gotcha) cannot clip V — no extra calibration needed.
    if enable_neural_critic:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = 150.0
        # PHYSIOLOGICAL operating point 0.02 (NOT the 0.105 default), per the
        # de-risk PASS (commit d0416fc3): at 0.105 the slow GABA_B conductance
        # over-accumulates to g~50-170 nS and FLATLINES the SNc at -87mV / 0 Hz
        # -> no live burst to discriminate (the prior smoke's "SNc silenced"
        # failure mode). 0.02 settles g~10-20, restoring baseline 1-3 Hz +
        # burst 50-80 Hz so the value subtraction is visible, not annihilating.
        cfg.gabab_propagation_strength = (float(critic_gabab_propagation)
                                          if critic_gabab_propagation and critic_gabab_propagation > 0
                                          else 0.02)
        # Brain-based GIRK saturation cap (finite channels) so a hot critic can't fully clamp the
        # SNc -> graded online δ at any critic rate (the nav-A/B honest-negative fix). 0 = no cap.
        cfg.gabab_conductance_max = float(critic_gabab_max)
    # (N5 now uses a plain proximity reward sc_rostral->reward_us; no slow channels needed --
    #  the temporal-difference is the dopamine RPE's job. enable_gabab is set by the neural
    #  critic above when present.)
    if _neural_place_selforg:
        # N9 Route-D coincidence read-out (the landed b980070a dendritic plateau): the FS-PING
        # gamma volley on `place` is read by the place->striosome_value coincidence_detector so the
        # synchronized packet fires the MSN critic the sparse-async code cannot. Mirrors the de-risk
        # _build (volley arm). NMDA ON (the per-region mask restricts the Mg2+-block kernel to the
        # critic slice, which carries enable_nmda=True; the actor regions are NMDA-free). At BUILD
        # the plateau is the strong COUNT form (coincidence_k_threshold in COUNT units = the train
        # K) so it bootstraps the post-spike that drives DA-gated LTP; cfg.coincidence_weighted_drive
        # stays False (the WEIGHTED Poirazi-Mel readout that GRADES with the learned weight is a
        # READ-OUT-only toggle, applied in Phase 3 value-learning, not here).
        cfg.enable_coincidence_detection = True
        cfg.coincidence_k_threshold = float(coincidence_train_k)
        cfg.coincidence_gain = 2.0
        cfg.coincidence_plateau_strength = float(coincidence_plateau)
        cfg.coincidence_weighted_drive = False
        cfg.enable_nmda = True   # per-region mask -> NMDA only on the critic (enable_nmda=True there)
    cfg.enable_structural_pruning = enable_structural_pruning
    cfg.enable_d1_d2_asymmetry = enable_d1_d2_asymmetry
    # Cluster G v1 (2026-05-01): Wang 2002 NMDA-mediated PFC working memory.
    # NMDA is global (affects all regions); ratio elevated to PFC-typical 0.5
    # per Wang 2002. Future work: per-region NMDA ratio override for
    # biologically-correct PFC-only NMDA dominance.
    if enable_pfc_nmda:
        cfg.enable_nmda = True
        cfg.nmda_ratio = 0.5  # Wang 2002 PFC calibration (default 0.4)
        # nmda_tau_decay (100 ms) and nmda_tau_rise (3 ms) keep their
        # CoreSimConfig defaults — already match Wang 2002.
    # N6 accumulate-then-commit (2026-06-06): the sel_X accumulator pools carry
    # BrainRegion.enable_nmda=True (set in build_bg_brain_regions). Turning on
    # global cfg.enable_nmda activates the dual-exponential NMDA conductance; the
    # bridge's per-region cp_nmda_neuron_mask then restricts the NMDA CURRENT to
    # the sel_X slice only (and dlpfc/cortex too if --enable-pfc-nmda is also on).
    # This gives the recurrent self-excitation the slow Wang-2002 integration
    # time constant (nmda_tau_decay=100ms) needed to ramp the weak thalamic drive
    # to a committed bound. Without this the sel_X recurrence would be AMPA-fast.
    if _readout_source == "spiking_wta":
        cfg.enable_nmda = True
        if not enable_pfc_nmda:
            cfg.nmda_ratio = 0.5  # PFC-typical NMDA dominance for the accumulator
    # Cluster B.3 (2026-04-28): cholinergic TANs. Turn the neuromod
    # subsystem ON cumulatively (no other flag in this runner enables it
    # today, but `|=` keeps it future-proof if one starts to) and append
    # the default acetylcholine config to whatever the cfg already has.
    if enable_tans:
        from sim.neuromodulators import _default_acetylcholine_tan_config
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_acetylcholine_tan_config()
        ]
    # R3.6 (2026-04-29): D1/D2 neuropeptide arms — dynorphin (D1, KOR
    # plasticity-rate brake), substance P (D1, NK-1 ACh boost), enkephalin
    # (D2, DOR plasticity-rate boost). All three opt-in together.
    if enable_bg_neuropeptides:
        from sim.neuromodulators import (
            _default_dynorphin_config,
            _default_substance_p_config,
            _default_enkephalin_config,
        )
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_dynorphin_config(),
            _default_substance_p_config(),
            _default_enkephalin_config(),
        ]
    # Cluster C v1 (2026-04-29): tonic dopamine via neuromodulator framework.
    # Replaces signed-scalar reward modulation with a real DA concentration
    # (tonic baseline + phasic activation/depression). Unlocks B.3 ACh
    # window-gating (which is otherwise a no-op without tonic DA-driven
    # plasticity to gate). Composes with --enable-tans and
    # --enable-bg-neuropeptides.
    #
    # Spiking-SNc actor-critic Stage A (2026-06-08): the `dopamine` modulator's
    # production rule reads the SNc pool's FIRING (from_region_firing_signed),
    # so the DA broadcast IS the spiking reward-prediction error. It OWNS the
    # `dopamine` modulator: --enable-tonic-da (which registers the from_reward
    # `_default_dopamine_config`) is skipped when spiking_snc is set (two
    # `dopamine` modulators would be a config error), exactly mirroring the
    # existing tonic-vs-compartmentalized precedence below. Mutually exclusive
    # with --enable-compartmentalized-da (per-action channels are a different DA
    # decomposition; combining them is undefined in v1).
    if spiking_snc and enable_compartmentalized_da:
        raise ValueError(
            "--spiking-snc and --enable-compartmentalized-da are mutually "
            "exclusive: the spiking SNc owns a single global `dopamine` "
            "modulator (from_region_firing_signed over the snc pool), while "
            "compartmentalized DA registers 4 per-action channels. Pick one."
        )
    if spiking_snc:
        from sim.neuromodulators import (
            NeuromodulatorConfig, ModulatorTarget, ProductionRule,
        )
        cfg.enable_neuromodulator_subsystem = True
        # SNc tonic firing FRACTION threshold: at this rate the signed rule
        # nets ~0 production so the DA concentration sits at baseline (RPE=0).
        # The pool is paced to ~mid-range by snc_tonic_pa; the threshold is the
        # firing FRACTION (0..1) the windowed-rate EMA settles to at tonic.
        # Calibrated empirically by the Pavlovian probe (snc_pavlovian_probe.py).
        snc_tonic_firing_fraction = 0.30
        # Inline (no new sim/ symbol per the design's recommendation): the same
        # NeuromodulatorConfig the factory would build. baseline/decay match
        # _default_dopamine_config so ACh window-gating (--enable-tans) composes.
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            NeuromodulatorConfig(
                name="dopamine",
                baseline=0.5,
                decay_tau_ms=200.0,
                concentration_min=0.0,
                concentration_max=2.0,
                targets=[
                    ModulatorTarget(
                        target_type="plasticity_rate", scope="all",
                        sensitivity=+1.0,
                    ),
                ],
                production_rules=[
                    ProductionRule(
                        rule_type="from_region_firing_signed",
                        sensitivity=float(snc_da_sensitivity),
                        threshold=float(snc_tonic_firing_fraction),
                        window_ms=200.0,
                        source_regions=["snc"],
                    ),
                ],
            )
        ]
        if verbose:
            print(f"[g11 seed={seed}] Spiking-SNc Stage A: dopamine modulator "
                  f"= from_region_firing_signed over ['snc'] "
                  f"(tonic={snc_tonic_pa}pA, k_r={snc_reward_gain}, "
                  f"k_v={snc_value_gain}); RPE is the SNc FIRING. "
                  f"V = host reward_ema (Stage-A scaffold).")

    # Precedence: when both --enable-tonic-da and --enable-compartmentalized-da
    # are set, only the per-action channels are registered (the global
    # `dopamine` modulator would double-count with the per-synapse path).
    # Also skipped when --spiking-snc owns the `dopamine` modulator (above).
    if enable_tonic_da and not enable_compartmentalized_da and not spiking_snc:
        from sim.neuromodulators import _default_dopamine_config
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_dopamine_config()
        ]

    # Cluster C v2 (2026-04-29): compartmentalized DA — per-action channels.
    # Registers 4 modulators (dopamine_N, dopamine_E, dopamine_S, dopamine_W),
    # each targeting only synapses with matching action_index via
    # scope='action:{idx}'. Production rule: from_action_specific_reward
    # gates concentration update by last_selected_action. Implies tonic-DA
    # at the per-action level (the single global dopamine modulator is NOT
    # registered when this flag is on).
    # See docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md.
    if enable_compartmentalized_da:
        from sim.neuromodulators import _default_per_action_dopamine_config
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_per_action_dopamine_config(action, idx)
            for idx, action in enumerate(ACTION_NAMES)
        ]
        if verbose:
            print(f"[g11 seed={seed}] Cluster C v2 compartmentalized DA: "
                  f"4 modulators registered "
                  f"(dopamine_{{{','.join(ACTION_NAMES)}}})")
    if pruning_alpha is not None:
        cfg.pruning_alpha = float(pruning_alpha)
    if pruning_threshold is not None:
        cfg.pruning_threshold = float(pruning_threshold)
    if pruning_weight_floor is not None:
        cfg.pruning_weight_floor = float(pruning_weight_floor)

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Cluster K v2 (2026-05-01): apply Gabor pre-init to V1 simple cells
    # so the visual cortex starts with biology-correct orientation tuning
    # rather than random weights. Must happen AFTER bridge init (CSR exists)
    # but BEFORE region_indices_cp is built since the call may grow nnz.
    # Also freeze IT -> cortex_X gate at 0 so the visual stream doesn't
    # disrupt motor selection during the critical period.
    if enable_visual_cortex:
        from sim.visual_cortex import apply_v1_gabor_weights
        n_gabor = apply_v1_gabor_weights(
            bridge,
            n_orientations=visual_n_orientations,
            n_frequencies=visual_n_frequencies,
            n_positions_per_dim=visual_n_positions_per_dim,
            retina_size=visual_image_size,
            receptive_field_radius=visual_receptive_field_radius,
            weight_scale=visual_v1_weight_scale,
        )
        if verbose:
            print(f"[g11 seed={seed}] Cluster K v2: applied {n_gabor} Gabor "
                  f"weights to retina -> cortex_v1_simple", flush=True)
        # Freeze IT -> cortex_X until critical-period close (warmup)
        try:
            bridge.set_plasticity_gate("visual_cortex_action", 0.0)
            if verbose:
                print(f"[g11 seed={seed}] Cluster K v2: visual_cortex_action "
                      f"gate frozen until warmup", flush=True)
        except KeyError:
            pass  # No IT -> cortex_X synapses if visual cortex regions absent

        # Spiking superior colliculus (N1): install the retinotopic retina->sc_map +
        # recurrent + sc_map->cortex_{N,E,S,W} quadrant-pooling wiring (the Mexican-hat
        # sc_map<->sc_fs is framework-built). De-risked in sc_map_orienting_probe.py.
        if enable_spiking_sc:
            # w_sc_cortex (the sc_map->cortex_X pooling strength) sets how hard the SC bump
            # biases action selection vs the BG cascade + OU noise. The host reflex injects
            # ~150 pA; the synaptic pooling must be strong enough to match. Sweepable via the
            # SC_CORTEX_W env var (tuning the integration-vs-isolation gap, A/B 2026-06-10).
            # Default 18.0 = the single-seed best (A/B 2026-06-10: SC/host 0.899 at w=18 — the
            # spiking SC BEATS the host reflex; non-monotonic, w=40 over-dominates). 6-seed
            # validates multi-seed. Override via SC_CORTEX_W.
            _scw = float(os.environ.get("SC_CORTEX_W", "18.0"))
            _scramble = os.environ.get("SC_SCRAMBLE", "0") == "1"   # anti-cheat lesion
            # TRUE-ONE-BRAIN #2 het-off operating point: the standalone-tuned w_ret_sc=80/w_sc_rec=6
            # STARVE the SC bump on the heterogeneity-OFF merged bridge (the documented "standalone
            # organ fires ~6-10x weaker co-resident" boundary, 2026-06-18-merged-limbic-core-lift.md):
            # sc_map fires ~2Hz and reward_us never crosses threshold. SC_RET_SC / SC_REC override the
            # retina->sc_map and sc_map recurrent weights; the de-risk's merged-tuned op-point is
            # 160/12 (corr(ecc,reward_us)=-0.81, SNc burst 1.45x, lesion collapses). Env-var-gated so
            # the standalone-nav default (env unset => 80/6) is BYTE-IDENTICAL.
            _w_ret_sc = float(os.environ.get("SC_RET_SC", "80.0"))
            _w_sc_rec = float(os.environ.get("SC_REC", "6.0"))
            install_spiking_sc_wiring(bridge, visual_image_size=visual_image_size,
                                      w_ret_sc=_w_ret_sc, w_sc_rec=_w_sc_rec,
                                      w_sc_cortex=_scw, scramble=_scramble, verbose=verbose)
            # TRUE-ONE-BRAIN #2: boost sc_rostral->reward_us to the het-off op-point (the build's
            # declared 14.0 is too weak co-resident; the de-risk used 40.0). Default unset => leave the
            # built 14.0 (byte-identical). Only meaningful with enable_spiking_sc_approach (sc_rostral
            # + the sc_rostral->reward_us pathway exist).
            _ros_us = os.environ.get("SC_ROS_US", "")
            if _ros_us and enable_spiking_sc_approach and "sc_rostral" in [r.name for r in regions] \
                    and spiking_reward_us:
                try:
                    rm_sc = bridge.region_manager
                    _ros = np.asarray(list(rm_sc.indices("sc_rostral")), dtype=np.int64)
                    _us = np.asarray(list(rm_sc.indices("reward_us")), dtype=np.int64)
                    _pre = np.repeat(_ros, _us.shape[0]).astype(np.int64)
                    _post = np.tile(_us, _ros.shape[0]).astype(np.int64)
                    bridge.set_pathway_weights(
                        "sc_rostral_to_reward_us", _pre, _post,
                        np.full(_pre.size, float(_ros_us), np.float32), add_missing=True)
                    if verbose:
                        print(f"[g11 seed={seed}] N5 reward: boosted sc_rostral->reward_us to "
                              f"{float(_ros_us)} ({_pre.size} synapses)", flush=True)
                except Exception as _e:
                    if verbose:
                        print(f"[g11 seed={seed}] N5 reward: sc_rostral->reward_us boost skipped ({_e})", flush=True)

    # Tier 2.2 (2026-05-06): open language plasticity gates for embodied
    # language training during nav. Same set of gates that were declared
    # in build_bg_brain_regions when enable_text_io=True.
    if embodied_language and enable_text_io:
        for gate_name in ("language_input_to_cortex",
                          "language_input_to_motor",
                          "it_to_language_output",
                          "cortex_to_language_output",
                          "language_input_to_pfc"):
            try:
                bridge.set_plasticity_gate(gate_name, 1.0)
                if verbose:
                    print(f"[g11 seed={seed}] embodied-language: gate "
                          f"'{gate_name}' open for nav training", flush=True)
            except KeyError:
                pass  # Gate may not exist depending on enable_pfc etc.

    # STEP 2a merge: the conv-finalization hook runs AFTER the Gabor/SC post-init wiring (which rebuilt the
    # CSR via set_pathway_weights(add_missing=True), staling earlier gate-index maps) and BEFORE the episode
    # loop -- it trains+freezes the conversational populations on the merged bridge BY INDEX. Default None =>
    # no-op (standalone nav byte-equivalent).
    if prebuilt_post_init_hook is not None:
        prebuilt_post_init_hook(bridge)

    # Pre-cache region indices (cupy arrays for fast per-step indexing)
    region_indices_cp = {}
    for r in regions:
        idx = list(bridge.region_manager.indices(r.name))
        if idx:
            region_indices_cp[r.name] = cp.asarray(idx, dtype=cp.int64)
    motor_idx_per_action = {
        a: region_indices_cp[f"motor_{a}"] for a in ACTION_NAMES
    }
    # N6 readout sources — host index arrays precomputed once (the legacy
    # readout called .get() on the cupy index every substep inside the
    # readout window). motor_X is the legacy source; thal_X is the
    # cleanly-selective genuine-disinhibition source (readout_source="thal");
    # sel_X is the spiking-WTA selection layer (readout_source="spiking_wta").
    motor_idx_host = {a: motor_idx_per_action[a].get() for a in ACTION_NAMES}
    thal_idx_host = {
        a: region_indices_cp[f"thal_{a}"].get()
        for a in ACTION_NAMES
        if f"thal_{a}" in region_indices_cp
    }
    sel_idx_host = {
        a: region_indices_cp[f"sel_{a}"].get()
        for a in ACTION_NAMES
        if f"sel_{a}" in region_indices_cp
    }
    # N6 accumulate-then-commit: the commit_X burst pools are the ACTUAL spiking
    # decision readout when enable_commit_burst is on (the host reads which
    # commit_X bursts past threshold). sel_X is the accumulator (logged as a guard).
    commit_idx_host = {
        a: region_indices_cp[f"commit_{a}"].get()
        for a in ACTION_NAMES
        if f"commit_{a}" in region_indices_cp
    }
    _use_commit = (_readout_source == "spiking_wta"
                   and enable_commit_burst
                   and len(commit_idx_host) == N_ACTIONS)
    # (_readout_source already normalized + validated above, before wiring.)
    if _readout_source == "thal" and len(thal_idx_host) != N_ACTIONS:
        raise ValueError(
            "readout_source='thal' requires per-action thal_X regions; "
            f"found {sorted(thal_idx_host)}")
    if _readout_source == "spiking_wta" and len(sel_idx_host) != N_ACTIONS:
        raise ValueError(
            "readout_source='spiking_wta' requires per-action sel_X regions; "
            f"found {sorted(sel_idx_host)}")
    if _readout_source == "spiking_wta" and enable_commit_burst and len(commit_idx_host) != N_ACTIONS:
        raise ValueError(
            "readout_source='spiking_wta' with enable_commit_burst requires "
            f"per-action commit_X regions; found {sorted(commit_idx_host)}")
    # N6 accumulate-then-commit: precompute the combined sel_X (+ commit_X) index
    # slice (cupy) for the per-trial accumulator reset (zero NMDA conductance +
    # reset membrane to rest each trial so each decision integrates fresh thalamic
    # evidence — see reset_accumulator_each_trial docstring).
    _accum_reset_idx_cp = None
    # N6 refinement 1 (2026-06-06): LOSER-ONLY reset. Per-action sel_X (+commit_X)
    # index slice so the previous trial's WINNER keeps its carried NMDA drive (the
    # legit working-memory latch when the goal is stable) while the three LOSERS are
    # zeroed each trial. At a goal change the new winner is among the freshly-cleared
    # losers (clean fresh integration of the new thal evidence) and the OLD winner —
    # no longer fed by thal — decays naturally instead of contaminating the race
    # (Cisek trial-wise re-baselining of the losing options). Surgical alternative
    # to the naive ALL-reset (which zeroed the eventual winner's drive too → silent
    # commit → WORSE). Built whenever spiking_wta is active so the per-trial code can
    # use it under either reset_accumulator_each_trial OR reset_losers_only.
    _accum_reset_idx_per_action_cp = {}
    if _readout_source == "spiking_wta":
        for a in ACTION_NAMES:
            _names_a = [f"sel_{a}"]
            if enable_commit_burst:
                _names_a.append(f"commit_{a}")
            _idx_a = []
            for nm in _names_a:
                if nm in region_indices_cp:
                    _idx_a.extend(region_indices_cp[nm].get().tolist())
            if _idx_a:
                _accum_reset_idx_per_action_cp[a] = cp.asarray(
                    sorted(set(_idx_a)), dtype=cp.int64)
    # N6 refinement 2 (2026-06-06): CISEK URGENCY / collapsing bound. Combined
    # sel_X index slice (all four accumulator pools). A ramping action-INDEPENDENT
    # urgency current is injected into this slice over the readout window so the
    # effective commit bound collapses with elapsed time: even on a weak late-phase
    # release the winning accumulator (the one ALSO receiving thal evidence) crosses
    # the bound within the 100ms window → its commit bursts (eliminates the
    # silent-commit → host-argmax-residual fallback). The urgency is the same for
    # every pool so it does NOT bias WHICH action wins — it only lowers the
    # threshold over time (Cisek 2009; Thura-Cisek 2014; Lo-Wang 2006 DA-modulated
    # bound). NO sim/ edit.
    _sel_all_idx_cp = None
    if _readout_source == "spiking_wta":
        _sel_idx_all = []
        for a in ACTION_NAMES:
            nm = f"sel_{a}"
            if nm in region_indices_cp:
                _sel_idx_all.extend(region_indices_cp[nm].get().tolist())
        if _sel_idx_all:
            _sel_all_idx_cp = cp.asarray(sorted(set(_sel_idx_all)), dtype=cp.int64)
    if _readout_source == "spiking_wta" and reset_accumulator_each_trial:
        _acc_names = [f"sel_{a}" for a in ACTION_NAMES]
        if enable_commit_burst:
            _acc_names += [f"commit_{a}" for a in ACTION_NAMES]
        _acc_idx = []
        for nm in _acc_names:
            if nm in region_indices_cp:
                _acc_idx.extend(region_indices_cp[nm].get().tolist())
        if _acc_idx:
            _accum_reset_idx_cp = cp.asarray(sorted(set(_acc_idx)), dtype=cp.int64)

    # Per-action DA targeting: pre-compute synapse-post-action mask.
    # For each plastic cortex→str_D1_X synapse, mark which action X it serves.
    # Per-trial: scale eligibility on synapses where post is in str_D1_Y (Y != selected)
    # by (1 - gating_strength), where gating_strength is either fixed at 1.0 (hard
    # mode) or adapted from recent reward stability (adaptive mode).
    # Adaptive: tracks reward EMA; high positive EMA → strength=1 (commit, exploit);
    # low/negative EMA → strength=0 (explore, broadcast credit).
    # We restrict to D1 (direct path); D2 (indirect) keeps broadcast learning.
    use_da_targeting = enable_per_action_da_targeting or enable_adaptive_per_action_da
    if use_da_targeting:
        coo = bridge.cp_connections.tocoo()
        post_neurons_cp = coo.col  # cupy int64
        n_synapses = int(post_neurons_cp.size)
        synapse_post_action = cp.full(n_synapses, -1, dtype=cp.int8)
        for action_idx_setup, action_name in enumerate(ACTION_NAMES):
            d1_indices = region_indices_cp[f"str_D1_{action_name}"]
            mask_d1 = cp.isin(post_neurons_cp, d1_indices)
            synapse_post_action[mask_d1] = action_idx_setup
        # Cache: per-action mask of "synapses NOT going to action X's D1 pool"
        # (used to zero eligibility before reward hold).
        d1_synapse_other_action_masks = {}
        for action_idx_setup, action_name in enumerate(ACTION_NAMES):
            # Mask = is a D1 synapse AND post-action != this action
            other_d1 = (synapse_post_action >= 0) & (synapse_post_action != action_idx_setup)
            d1_synapse_other_action_masks[action_idx_setup] = other_d1
        if verbose:
            n_d1_synapses = int((synapse_post_action >= 0).sum().get())
            mode = "adaptive" if enable_adaptive_per_action_da else "hard"
            print(f"[g11 seed={seed}] per-action DA ({mode}): "
                  f"{n_d1_synapses} synapses are cortex->D1 (will be selectively gated)")
    else:
        d1_synapse_other_action_masks = None

    # Adaptive DA state — reward EMA in [-1, +1]
    reward_ema = 0.0
    da_strength_log = []  # log per-trial gating strength for analysis

    # Spiking-SNc Stage A (2026-06-08): host index array for the snc pool (used
    # to sum cp_firing_states over the reward-hold window) + per-trial SNc spike
    # log (the spiking RPE readout, surfaced in the output JSON for diagnostics).
    _snc_idx_host = None
    snc_rate_log = []
    if spiking_snc and "snc" in region_indices_cp:
        _snc_idx_host = region_indices_cp["snc"].get()

    # Spiking-SNc Stage B (2026-06-08): neural-critic instrumentation. Host index
    # for the striosome_value pool (to read its firing during reward windows =
    # the learned value V), plus a reader for the plastic cortex_it->striosome_value
    # weight so we can confirm the critic LEARNS (the weight grows from its init
    # and V tracks expected reward). The smoke gate inspects these.
    _striov_idx_host = None
    striov_rate_log = []          # per-trial striosome_value spike count (reward window)
    critic_weight_initial = None  # mean cortex_it->striosome_value weight at start

    # AFFERENT 2026-06-09 VALIDATED redesign: the value critic reads the DEDICATED DENSE
    # `vs_place_context` place-context code (drive-injected each nav step), NOT the SPARSE
    # actor `sensor_place_readout` (which can't fire the MSN critic). The weight reader +
    # the instrumentation below follow this afferent. N9 nav deployment: under
    # neural_place_selforg the afferent is the SELF-ORGANIZED spiking `place` pool instead.
    _critic_afferent_region = "place" if _neural_place_selforg else "vs_place_context"

    def _mean_critic_weight():
        """Mean weight of the <afferent>->striosome_value edges in the CSR.
        In this bridge's cp_connections, rows=PRE(source), cols=POST(target)
        (verified: the afferent->striosome_value pathway matches rows in the
        afferent, cols in striosome_value). Vectorized via np.isin (the CSR has
        ~140k synapses — a per-edge Python generator is far too slow). Returns
        None if the critic isn't built or no edges match."""
        if not enable_neural_critic:
            return None
        if ("striosome_value" not in region_indices_cp
                or _critic_afferent_region not in region_indices_cp):
            return None
        try:
            pre = region_indices_cp[_critic_afferent_region].get()  # source rows
            post = region_indices_cp["striosome_value"].get()    # target cols
            coo = bridge.cp_connections.tocoo()
            rows = coo.row.get() if hasattr(coo.row, "get") else np.asarray(coo.row)
            cols = coo.col.get() if hasattr(coo.col, "get") else np.asarray(coo.col)
            data = coo.data.get() if hasattr(coo.data, "get") else np.asarray(coo.data)
            m = np.isin(rows, pre) & np.isin(cols, post)
            if not m.any():
                # Fallback to the opposite orientation (robust to CSR convention).
                m = np.isin(rows, post) & np.isin(cols, pre)
            return float(data[m].mean()) if m.any() else None
        except Exception:
            return None

    if enable_neural_critic and "striosome_value" in region_indices_cp:
        _striov_idx_host = region_indices_cp["striosome_value"].get()
        critic_weight_initial = _mean_critic_weight()
        if verbose:
            _gate_mode = (
                f"WINDOWED (lead<= {critic_lead_steps} steps, flushed otherwise)"
                if enable_critic_window else "OPEN (continuous, Stage 1)"
            )
            print(f"[g11 seed={seed}] Spiking-SNc Stage B: NEURAL value critic ON "
                  f"(striosome_value n={len(_striov_idx_host)}, "
                  f"{_critic_afferent_region}->value w0={critic_weight_initial}); "
                  f"r-V subtracted via GABA_B/GIRK (E_K=-90mV, prop=0.02); "
                  f"critic->SNc gate = {_gate_mode}. Host _V_scaffold DROPPED.")

    # Stage 2 windowed-GABA_B state. The `critic_snc_window` transmission gate
    # (declared on the striosome_value->snc pathway) scales that route's
    # effective synaptic CURRENT = the g_gabab increment. Stage 1 (default,
    # enable_critic_window=False): hold it OPEN (1.0) for the whole run, so the
    # slow conductance integrates continuously (isolate "does it run + learn").
    # Stage 2 (enable_critic_window=True): a rolling sawtooth that OPENS the gate
    # for a bounded <=critic_lead_steps lead (g_gabab pre-builds ~1 tau into the
    # reward), then CLOSES it for an equal flush phase so the conductance decays
    # and cannot integrate across a long dwell (the de-risk >=200 ms
    # over-suppression boundary, d0416fc3). The gate is FORCE-OPEN through each
    # reward block regardless of phase so the subtraction is live at reward.
    _critic_gate_known = (
        enable_neural_critic
        and "critic_snc_window" in getattr(bridge, "_transmission_gate_to_synapses", {})
    )
    _critic_open_counter = 0
    if _critic_gate_known:
        # Stage 1: explicit OPEN (also the inject default, so byte-equivalent).
        bridge.set_transmission_gate("critic_snc_window", 1.0)

    # DA-gated WTA: pre-compute FS->motor synapse indices and save baseline weights.
    # Per-trial we'll scale these weights by gating_strength to make WTA adaptive.
    # Cluster F v2: cache cerebellum_pf_pc synapse indices for the per-synapse
    # reward override path. When enabled, these synapses get the CF-gated
    # signal (-1.0 on CF event, 0.0 otherwise) instead of the global reward.
    cerebellum_pf_pc_indices = None
    cerebellum_pf_pc_mask = None  # GPU bool array
    if enable_cluster_f_v2 and enable_cluster_f_cerebellum:
        gate_to_syns = getattr(bridge, "_plasticity_gate_to_synapses", {})
        cere_idx_list = gate_to_syns.get("cerebellum_pf_pc")
        if cere_idx_list:
            cerebellum_pf_pc_indices = cp.asarray(np.asarray(cere_idx_list, dtype=np.int64))
            actual_nnz = bridge.cp_connections.nnz
            cerebellum_pf_pc_mask = cp.zeros(actual_nnz, dtype=cp.bool_)
            cerebellum_pf_pc_mask[cerebellum_pf_pc_indices] = True
            if verbose:
                print(f"[g11 seed={seed}] Cluster F v2 enabled: "
                      f"{len(cere_idx_list)} cerebellum_pf_pc synapses tagged for CF-gated LTD",
                      flush=True)
        elif verbose:
            print(f"[g11 seed={seed}] WARNING: --enable-cluster-f-v2 set but no "
                  f"cerebellum_pf_pc gate found. Did you forget --enable-cluster-f-cerebellum?",
                  flush=True)

    fs_to_motor_indices = None
    fs_to_motor_baseline_weights = None
    if enable_da_gated_wta and enable_motor_lateral_inhibition:
        # All FS pre-neurons (across 4 actions); all motor post-neurons (across 4 actions)
        fs_indices_all = []
        motor_indices_all = []
        for action in ACTION_NAMES:
            fs_indices_all.extend(region_indices_cp[f"motor_FS_{action}"].get().tolist())
            motor_indices_all.extend(region_indices_cp[f"motor_{action}"].get().tolist())
        fs_set = set(fs_indices_all)
        motor_set = set(motor_indices_all)
        # Find synapse indices where pre in fs_set AND post in motor_set
        coo = bridge.cp_connections.tocoo()
        rows = coo.row.get(); cols = coo.col.get()
        # CSR convention: assume cp_connections[i, j] means i->j (pre->post)
        # We pick the orientation that gives non-zero count.
        mask_a = np.array([r in fs_set and c in motor_set for r, c in zip(rows, cols)])
        mask_b = np.array([c in fs_set and r in motor_set for r, c in zip(rows, cols)])
        if mask_a.sum() > mask_b.sum():
            chosen_mask = mask_a
            convention = "row=pre, col=post"
        else:
            chosen_mask = mask_b
            convention = "row=post, col=pre"
        fs_to_motor_indices = cp.asarray(np.where(chosen_mask)[0], dtype=cp.int64)
        # Snapshot baseline weights (constant since FS->motor is plastic=False)
        fs_to_motor_baseline_weights = bridge.cp_connections.data[fs_to_motor_indices].copy()
        if verbose:
            print(f"[g11 seed={seed}] DA-gated WTA: {int(chosen_mask.sum())} FS->motor synapses "
                  f"({convention}), will scale by gating_strength per trial")

    # Informed initialization for learned perception: bias initial sensory->cortex_X
    # weights by alignment between sensor's preferred (dx, dy) and action X's
    # direction vector. Solves the cold-start problem identified in
    # research/findings/2026-04-26-learned-perception-cold-start-fail.md.
    if (enable_learned_perception
            and enable_learned_perception_informed_init
            and sensory_pref_dx is not None):
        # Action direction vectors (N, E, S, W) — must match ACTION_DELTAS
        action_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        sensory_indices_list = list(bridge.region_manager.indices("sensory"))
        sensory_set = set(sensory_indices_list)
        sensory_idx_to_pos = {n: i for i, n in enumerate(sensory_indices_list)}
        coo = bridge.cp_connections.tocoo()
        rows_np = coo.row.get(); cols_np = coo.col.get()
        n_modified = 0
        # CSR convention here: rows are pre, cols are post (verified by FS->motor logic above)
        for action_idx, action_name in enumerate(ACTION_NAMES):
            cortex_X_set = set(bridge.region_manager.indices(f"cortex_{action_name}"))
            ax, ay = action_dirs[action_idx]
            # Find synapse indices where pre is in sensory and post is in cortex_X
            new_weights = []
            target_indices = []
            for syn_idx in range(rows_np.size):
                pre = int(rows_np[syn_idx])
                post = int(cols_np[syn_idx])
                if pre in sensory_set and post in cortex_X_set:
                    sensor_layer_idx = sensory_idx_to_pos[pre]
                    dx_pref = float(sensory_pref_dx[sensor_layer_idx])
                    dy_pref = float(sensory_pref_dy[sensor_layer_idx])
                    # Alignment: dot product of sensor's preferred direction with action's direction
                    alignment = dx_pref * ax + dy_pref * ay  # ranges roughly [-3, +3]
                    # SHARP prior: only positive alignment contributes meaningfully.
                    # Orthogonal/anti-aligned sensors get near-zero weight so they don't
                    # drive cortex_X (avoiding cascade saturation across all 4 pools).
                    # Aligned sensors get strong weight (up to ~25 = matches heuristic 800pA equivalent).
                    positive_alignment = max(0.0, alignment)
                    new_w = max(0.5, 0.5 + informed_init_alpha * positive_alignment)
                    new_weights.append(new_w)
                    target_indices.append(syn_idx)
            if target_indices:
                idx_cp = cp.asarray(target_indices, dtype=cp.int64)
                w_cp = cp.asarray(new_weights, dtype=cp.float32)
                bridge.cp_connections.data[idx_cp] = w_cp
                n_modified += len(target_indices)
        if verbose:
            print(f"[g11 seed={seed}] learned perception (informed init): "
                  f"rewrote {n_modified} sensory->cortex weights with directional prior "
                  f"(alpha={informed_init_alpha})")

    # Setup baseline tonic drives that don't change between steps.
    # N8 conversion: under genuine_thal_disinhibition, GPi becomes a strong
    # tonic pacemaker (genuine_gpi_tonic_pA) and thalamus carries only a tonic
    # excitation (genuine_thal_tonic_pA) expressed when GPi is released — NO
    # direct 300 pA thal pacing. Otherwise the legacy tonic cheat (gpi 110,
    # thal 300) is used. GPe/STN/SNc drives are unchanged in both (N9, out of
    # scope for this conversion).
    _gpi_tonic = cp.float32(genuine_gpi_tonic_pA if genuine_thal_disinhibition else 110.0)
    _thal_tonic = cp.float32(genuine_thal_tonic_pA if genuine_thal_disinhibition else 300.0)
    bridge.cp_external_input_current[:] = 0.0
    for region_name in [f"gpe_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(150.0)
    for region_name in [f"gpe_arky_{a}" for a in ACTION_NAMES]:
        if region_name in region_indices_cp:
            bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(120.0)
    for region_name in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = _gpi_tonic
    for region_name in ["stn", "snc"]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = cp.float32(150.0)
    for region_name in [f"thal_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[region_name]] = _thal_tonic
    # N6 commit stage: keep commit_OPN tonically firing (omnipause) so it holds
    # every commit_X burst pool below threshold until a sel_X accumulator ramps
    # past the bound (Lo-Wang 2006 SC threshold; H.24 OPN->EBN). Constant drive,
    # set once like the GPi pacemaker above.
    if _use_commit and "commit_OPN" in region_indices_cp:
        bridge.cp_external_input_current[region_indices_cp["commit_OPN"]] = cp.float32(commit_opn_tonic_pA)
    if genuine_thal_disinhibition and verbose:
        print(f"[g11 seed={seed}] N8 GENUINE thalamic disinhibition ON: "
              f"gpi_tonic={genuine_gpi_tonic_pA:.0f} pA (pacemaker), "
              f"thal_tonic={genuine_thal_tonic_pA:.0f} pA (released by D1-|GPi-|thal); "
              f"NO direct thal pacing.", flush=True)

    # Action deltas
    ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W
    n_motor_per_action = sum(1 for r in regions if r.name.startswith("motor_")) * 0  # placeholder
    # Number of neurons in each motor pool (all same)
    n_motor_pop = next(r.n_neurons for r in regions if r.name.startswith("motor_"))

    x, y = start_pos
    current_schedule_idx = 0
    gx, gy = goal_schedule_sorted[0][1]

    def manhattan(px, py):
        return abs(px - gx) + abs(py - gy)

    trajectory = [(x, y)]
    goal_log = [(gx, gy)]
    motor_counts_log = []
    thal_counts_log = []  # N6 thal-readout firing guard (per-step thal_X counts)
    sel_counts_log = []   # N6 spiking-WTA firing guard (per-step sel_X counts)
    commit_counts_log = []  # N6 accumulate-then-commit guard (per-trial commit_X burst counts)
    # N6 guard: count which arm of the fallback chain each trial took. "primary" =
    # the commit burst fired (the genuine spiking decision); "fallback" = silent
    # commit → sel-lean argmax residual; "random" = both silent. A GO drives
    # fallback+random toward 0 (the commit fires reliably).
    _decision_path_counts = {"primary": 0, "fallback": 0, "random": 0}
    # Per-step accumulation/commit traces for a few sample trials (the guard the
    # task asks for: winner's sel ramps + its commit bursts while losers stay low).
    _GUARD_SAMPLE_TRIALS = {0, 1, 2, 60, 120}
    accum_trace_log = {}  # step -> {"sel": [[per-action per-substep]], "commit": [[...]]}
    action_log = []
    reward_log = []
    distance_log = [manhattan(x, y)]
    goal_change_steps = []

    # ----- ADAPTIVE / activity-gated heuristic weaning state (N1) -----
    # Phase machine: "teaching" (heuristic full strength, occasionally interrupted
    # by OFF probe windows) -> "committed" (heuristic permanently weaning to 0 by
    # ramping over heuristic_wean_steps from adaptive_commit_step).
    adaptive_phase = "teaching"
    adaptive_probe_active = False          # currently inside an OFF probe window
    adaptive_probe_start_step = -1         # step at which the active probe began
    adaptive_commit_step = -1              # step at which the wean committed (-1 = not yet)
    adaptive_probe_history = []            # list of dicts: {probe_start, probe_end, mean_dist, committed}
    # First probe begins at the first multiple of wean_probe_every that is > 0
    # (i.e. after at least one teaching block). Guard against degenerate configs.
    _wpe = max(1, int(wean_probe_every))
    _wpw = max(1, int(wean_probe_window))

    STIMULUS_MS = 100.0
    READOUT_START_MS = 30.0
    READOUT_END_MS = 100.0
    n_stim_steps = int(STIMULUS_MS / cfg.dt_ms)
    readout_start = int(READOUT_START_MS / cfg.dt_ms)
    readout_end = int(READOUT_END_MS / cfg.dt_ms)

    cortex_idx_per_action = {
        a: region_indices_cp[f"cortex_{a}"] for a in ACTION_NAMES
    }

    if verbose:
        print(f"[g11 seed={seed}] BG circuit: {len(regions)} regions, "
              f"{cfg.num_neurons} neurons, {bridge.cp_connections.nnz} synapses",
              flush=True)

    # Sleep-replay trajectory log: stores (x, y, gx, gy) tuples from
    # waking trials where the agent successfully approached goal
    # (reward > 0). During sleep, these are replayed instead of random
    # patterns, modeling biological hippocampal replay of successful
    # episodic memories.
    # Bounded to recent ~200 entries so sleep replays mostly the
    # current-goal patterns, not stale patterns from earlier goals
    # (which can bias consolidation toward old goal directions).
    # Biologically: hippocampal trace decay ensures replay reflects
    # recent experience, not arbitrary old episodes.
    successful_trajectories: List = []
    SUCCESSFUL_TRAJ_MAX = 200
    # HER lag buffer: stores (x, y) from `her_lag_steps` ago so we can
    # construct hindsight tuples (old_pos, current_pos_as_goal). 50 steps
    # is the reach distance on an 8x8 grid (max Manhattan ≈ 14, ~3 steps
    # per goal change typical, so ~50-step lookahead spans a meaningful
    # chunk of trajectory).
    her_lag_buffer = []
    her_lag_steps = 50

    # Curriculum: real plasticity gating (Stage 3, 2026-04-27).
    # The hippo→cortex pathways are tagged "place_goal_to_cortex" and cortex→D1/D2
    # are tagged "corticostriatal" in build_bg_brain_regions. We use these gates
    # to implement true developmental staging:
    #   Phase 1 (warmup): cortex→D1 plastic, hippo→cortex frozen
    #     → cortex builds correct cortex→D1 mapping under heuristic alone
    #   Phase 2 (mature): cortex→D1 frozen, hippo→cortex plastic
    #     → hippo learns place→action given that cascade is locked-in
    # This addresses the architectural ceiling identified in the 6-NEGATIVE
    # plastic-input-layer arc: the cascade depends on a single clean cortex
    # input source. By staging plasticity, we let cortex selectivity
    # establish itself, then add the plastic input layer with the cascade
    # protected against further drift.
    #
    # Stage 5 ramping: when ramp_steps>0, transitions are smooth (linear
    # interpolation of gate values over ramp window centered on warmup).
    # This matches biology — critical periods close gradually via PV
    # maturation, not as step functions — and reduces variance from
    # abrupt cascade disruption.
    # Curriculum gates: corticostriatal, hippo_to_cortex, sensory_to_cortex,
    # beacon_to_goal. In phase 1, all input layers (hippo, sensory, beacon→goal)
    # are frozen and only corticostriatal is plastic. Cortex builds D1 mapping
    # under the heuristic teacher. In phase 2, corticostriatal freezes and the
    # input layers thaw, learning their mappings with cortex as the locked target.

    # v4 developmental pretraining (2026-04-28). Runs only if enabled.
    # Inserted BEFORE curriculum init so the init's phase-1 gate values
    # naturally freeze bg_cross_projections at eval start (line 1220).
    pretraining_summary = None
    if enable_developmental_pretraining:
        pretraining_summary = _run_pretraining_phase(
            bridge=bridge, cfg=cfg, regions=regions,
            n_goals=pretraining_n_goals,
            steps_per_goal=pretraining_steps_per_goal,
            grid_size=grid_size, start_pos=start_pos,
            seed=seed,
            enable_bg_cross_projections=enable_bg_cross_projections,
            verbose=verbose,
        )

    available_gates = bridge.list_plasticity_gates() if enable_curriculum else []
    has_hippo_gate = enable_curriculum and "place_goal_to_cortex" in available_gates
    has_cortex_gate = enable_curriculum and "corticostriatal" in available_gates
    has_sensory_gate = enable_curriculum and "sensory_to_cortex" in available_gates
    has_beacon_gate = enable_curriculum and "beacon_to_goal" in available_gates
    has_landmark_gate = enable_curriculum and "landmark_to_place" in available_gates
    has_bg_cross_gate = enable_curriculum and "corticostriatal_cross" in available_gates

    # Cluster D v2: cache the SWR gate availability + the CA3 indices used
    # to compute population firing rate every step. Gate availability is
    # checked against bridge.list_plasticity_gates() which enumerates the
    # gates that build_wiring_plan registered for this run. CA3 indices
    # come from the region manager. Runtime per-step cost: one CuPy
    # `cp_firing_states[ca3_indices].sum()` reduction.
    has_swr_gate = (
        enable_cluster_d_v2_swr
        and "ca3_swr_burst" in (bridge.list_plasticity_gates() or [])
    )
    ca3_indices_cp = None
    if has_swr_gate:
        try:
            _ca3_idx = list(bridge.region_manager.indices("ca3"))
            if _ca3_idx:
                ca3_indices_cp = cp.asarray(_ca3_idx, dtype=cp.int64)
        except (KeyError, AttributeError):
            ca3_indices_cp = None
        if ca3_indices_cp is None:
            has_swr_gate = False  # CA3 region not allocated; skip gating
    ca3_rate_history: deque = deque(maxlen=40)
    swr_burst_count = 0      # number of steps where v2 burst was detected
    swr_sleep_steps = 0      # number of sleep steps where v2 gate was active
    bg_cross_thawed = False  # tracks the phase-3 thaw event for verbose logging
    if enable_curriculum:
        # Phase 1: input plasticity OFF, corticostriatal plasticity ON,
        # bg_cross_projections OFF (stays off until phase 3 if configured)
        if has_hippo_gate:
            bridge.set_plasticity_gate("place_goal_to_cortex", 0.0)
        if has_sensory_gate:
            bridge.set_plasticity_gate("sensory_to_cortex", 0.0)
        if has_beacon_gate:
            bridge.set_plasticity_gate("beacon_to_goal", 0.0)
        if has_landmark_gate:
            bridge.set_plasticity_gate("landmark_to_place", 0.0)
        if has_cortex_gate:
            bridge.set_plasticity_gate("corticostriatal", 1.0)
        if has_bg_cross_gate:
            bridge.set_plasticity_gate("corticostriatal_cross", 0.0)
        if verbose:
            ramp_msg = (f", ramp={curriculum_ramp_steps}" if curriculum_ramp_steps > 0
                       else " (abrupt)")
            gates_msg = ", ".join(filter(None, [
                "place_goal_to_cortex" if has_hippo_gate else None,
                "sensory_to_cortex" if has_sensory_gate else None,
            ]))
            print(f"[g11 seed={seed}] curriculum phase 1: corticostriatal plastic, "
                  f"input gates frozen [{gates_msg}]{ramp_msg}", flush=True)
    last_logged_phase = 1  # for verbose phase-2 announcement on first ramp tick

    def _curriculum_gate_values(step_idx):
        """Return (cortex_gate, hippo_gate) for the given step under the
        current curriculum schedule. Linear ramp centered on warmup boundary
        when ramp_steps > 0; abrupt step otherwise.

        Phase 1 values: cortex=1.0, hippo=0.0 (cortex plastic, hippo frozen)
        Phase 2 values: cortex=curriculum_phase2_cortex_gain (default 0.0),
                        hippo=curriculum_phase2_hippo_gain (default 1.0).
        Partial-freeze configs (e.g. cortex=0.3) let cortex slowly track
        changing reward landscape while hippo learns the primary input
        mapping — biologically: cortical plasticity slows but persists.
        """
        c_phase1, h_phase1 = 1.0, 0.0
        c_phase2 = curriculum_phase2_cortex_gain
        h_phase2 = curriculum_phase2_hippo_gain
        if curriculum_ramp_steps <= 0:
            # Abrupt: phase 1 until warmup, phase 2 after
            if step_idx < curriculum_warmup_steps:
                return c_phase1, h_phase1
            return c_phase2, h_phase2
        # Smooth: ramp over [warmup - half, warmup + half]
        half = curriculum_ramp_steps // 2
        ramp_start = curriculum_warmup_steps - half
        ramp_end = curriculum_warmup_steps + (curriculum_ramp_steps - half)
        if step_idx < ramp_start:
            return c_phase1, h_phase1
        if step_idx >= ramp_end:
            return c_phase2, h_phase2
        # In ramp window: linear interpolation between phase 1 and phase 2 values
        progress = (step_idx - ramp_start) / float(curriculum_ramp_steps)
        c_val = c_phase1 + (c_phase2 - c_phase1) * progress
        h_val = h_phase1 + (h_phase2 - h_phase1) * progress
        return c_val, h_val

    # Live brain-activity probe (frontend-revamp Phase 1). Built ONCE here so
    # the per-step loop only pays the (throttled) reduction when --emit-activity
    # is set. Default off => _activity_probe stays None => zero overhead, the
    # loop is byte-identical to a science run.
    _activity_probe = None
    _activity_emit = None
    if emit_activity:
        try:
            from sim.activity_probe import RegionActivityProbe
            from sim.progress import emit_activity as _activity_emit
            _activity_probe = RegionActivityProbe(bridge)
            if verbose:
                print(
                    f"[g11 seed={seed}] activity streaming ON: "
                    f"{_activity_probe.n_regions} regions, "
                    f"{_activity_probe.n_pathways} pathways, "
                    f"every {emit_activity_every} steps",
                    flush=True,
                )
        except Exception as _ap_err:
            # Never let activity instrumentation break a run.
            print(f"[g11 seed={seed}] activity probe init failed ({_ap_err}); "
                  f"continuing without it", flush=True)
            _activity_probe = None

    # ═══════════════════════════════════════════════════════════════════════════════════
    # N9 STEP-1 PLACE-CODE SELF-ORGANIZATION (2026-06-09 nav deployment of the de-risk
    # n9_place_graded_critic_stage2_derisk run_seed STEP-1). Before the nav loop, self-organize
    # the spiking `place` fields from the egocentric `place_sensors` render, then FREEZE them
    # (a stable afferent for the value critic). BRAIN-BASED-ONLY: the only host scaffolding is
    # the agent-placement sweep (the environment) + the sensory render; the place fields emerge
    # from neurons + synapses (competitive threshold-WTA).
    # ═══════════════════════════════════════════════════════════════════════════════════
    def _n9_step(n):
        for _ in range(int(n)):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * cfg.dt_ms)

    def _n9_selforg_positions():
        """A near-square sub-grid of selforg_n_positions agent positions tiling the arena
        (diverse egocentric sensor patterns so the place code carves distinct fields)."""
        k = max(1, int(round(float(selforg_n_positions) ** 0.5)))
        xs = np.linspace(0.0, grid_size - 1.0, k)
        ys = np.linspace(0.0, grid_size - 1.0, k)
        return [(float(px), float(py)) for px in xs for py in ys]

    def _n9_place_ensemble(px, py, *, n_meas=80):
        """Per-cell spike-count vector of the `place` pool at (px,py) (learning frozen)."""
        p_idx = region_indices_cp["place"]
        n = int(p_idx.size) if hasattr(p_idx, "size") else len(p_idx)
        act = _n9_render(px, py)
        bridge.cp_external_input_current[:] = cp.float32(0.0)
        bridge.cp_external_input_current[region_indices_cp["place_sensors"]] = cp.asarray(act, dtype=cp.float32)
        counts = cp.zeros(n, dtype=cp.float32)
        saved = bridge.core_config.reward_learning_rate
        bridge.core_config.reward_learning_rate = 0.0
        for _ in range(int(n_meas)):
            _n9_step(1)
            counts += bridge.cp_firing_states[p_idx].astype(cp.float32)
        bridge.core_config.reward_learning_rate = saved
        bridge.cp_external_input_current[:] = cp.float32(0.0)
        return counts.get() if hasattr(counts, "get") else np.asarray(counts)

    def _run_place_selforg():
        if not (_neural_place_selforg and "place" in region_indices_cp
                and "place_sensors" in region_indices_cp):
            return None
        t_so = time.time()
        # Optional DETERMINISTIC self-org (2026-06-10): the place code is CuPy-non-deterministic
        # because the per-step Wᵀ@fired is a transpose SpMV (csc atomic scatter; research
        # 2026-06-10-N9-placecode-reproducibility-robustness-research.md). Toggle the (default-off,
        # byte-identity-proven) cfg.deterministic_transpose_matvec ON for the self-org so the SAME
        # seed draws the SAME place code (the anti-cheat R-A: byte-identical place code across two
        # invocations). Restored after self-org (bounds the per-step .tocsr() cost to STEP-1).
        _saved_detmv = getattr(bridge.core_config, "deterministic_transpose_matvec", False)
        if deterministic_selforg:
            bridge.core_config.deterministic_transpose_matvec = True
        # STEP-1 gate config: open the competitive landmark->place learning, freeze the value
        # arm, and hold the FS-PING inhibition CLOSED (clean threshold-WTA -> sparse, DISTINCT
        # fields; the de-risk's Stage-1 regime).
        # SPARSIFY (2026-06-19, #5): the FS-PING can instead be held OPEN during self-org so the
        # fields are carved WITH recurrent feedback inhibition (the canonical Hartley-Burgess
        # place-field formation mechanism) -> sparser, better-separated learned weights that match
        # the FS-open read-out regime the value-train uses. Env-gated, default-off (byte-identical
        # when unset): N5_SPARSIFY_FS_DURING_SELFORG=1.
        _sparsify_fs_selforg = bool(int(os.environ.get("N5_SPARSIFY_FS_DURING_SELFORG", "0")))
        bridge.set_plasticity_gate("landmark_to_place", 1.0)
        bridge.set_plasticity_gate("value_input", 0.0)
        bridge.set_transmission_gate("place_fs_gate", 1.0 if _sparsify_fs_selforg else 0.0)
        positions = _n9_selforg_positions()
        steps_each = max(1, int(selforg_steps) // max(1, len(positions)))
        _rng = np.random.default_rng(seed)
        order = list(positions); _rng.shuffle(order)
        ps_idx = region_indices_cp["place_sensors"]
        for (px, py) in order:
            # brief silent gap (clears lingering depolarization) then drive the sensors.
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            _n9_step(20)
            bridge.cp_external_input_current[ps_idx] = cp.asarray(_n9_render(px, py), dtype=cp.float32)
            _n9_step(steps_each)
        # FREEZE the place fields (stable afferent) + OPEN the FS-PING for the volley read-out/nav.
        bridge.set_plasticity_gate("landmark_to_place", 0.0)
        bridge.set_transmission_gate("place_fs_gate", 1.0)
        bridge.cp_external_input_current[:] = cp.float32(0.0)
        # Place-code provenance (the de-risk Stage-1 gates, abridged): diff-location cosine.
        _goal0 = goal_schedule_sorted[0][1]
        _far = (float(grid_size) - 1.0 - float(_goal0[0]), float(grid_size) - 1.0 - float(_goal0[1]))
        ens_goal = _n9_place_ensemble(float(_goal0[0]), float(_goal0[1]))
        ens_far = _n9_place_ensemble(_far[0], _far[1])
        _na = float(np.linalg.norm(ens_goal)); _nb = float(np.linalg.norm(ens_far))
        diff_cos = float(np.dot(ens_goal, ens_far) / (_na * _nb)) if (_na > 0 and _nb > 0) else 1.0
        sparsity = float(0.5 * (np.mean(ens_goal > 0) + np.mean(ens_far > 0)))
        bridge.core_config.deterministic_transpose_matvec = _saved_detmv   # restore (bound cost to STEP-1)
        if verbose:
            print(f"[g11 seed={seed}] N9 STEP-1 place self-org done ({time.time()-t_so:.0f}s): "
                  f"diff-loc cos={diff_cos:.6f} sparsity={sparsity:.6f} (place fields FROZEN"
                  f"{'; DETERMINISTIC' if deterministic_selforg else ''})", flush=True)
        return dict(diff_cos=diff_cos, sparsity=sparsity)

    def _run_stage_a_smoke():
        """N9 Stage-A cheap-first probe (design §Stage A): FIRE / PLACE-GRADED / ACTOR-NOT-
        PERTURBED. Drives place_sensors at the goal vs a far cell after self-org. Does NOT
        proceed to the nav loop (caller exits).

        NOTE on the FIRE gate: the de-risk's `place` code is INTENTIONALLY SPARSE + low-rate
        (~3.65% sparsity, ~1-2 Hz; verified the de-risk's own `place` pool is 0.4-1.3 Hz at the
        goal both at init AND after its STEP-1 self-org). So the "place >=5 Hz" reading of the
        design's FIRE gate is a SPARSE-CODE rate that the de-risk never hits either — the load-
        bearing FIRE gate in the de-risk is the CRITIC (striosome_value) >=5 Hz, which fires via
        the FS-PING coincidence VOLLEY through the LEARNED place->value weight AFTER STEP-2 value-
        training (a Phase 3 deliverable, out of Phase 1-2 scope). This probe therefore reports the
        place rate/sparsity INFORMATIONALLY and gates Phase 1-2 on PLACE-GRADED (distinct fields)
        + ACTOR-NOT-PERTURBED (the structural-isolation guarantee); the critic rate is reported
        pre-value-training as a baseline."""
        p_idx = region_indices_cp["place"]
        c_idx = region_indices_cp["striosome_value"]
        n_place_n = int(p_idx.size) if hasattr(p_idx, "size") else len(p_idx)
        n_crit_n = int(c_idx.size) if hasattr(c_idx, "size") else len(c_idx)
        _goal0 = goal_schedule_sorted[0][1]
        _far = (float(grid_size) - 1.0 - float(_goal0[0]), float(grid_size) - 1.0 - float(_goal0[1]))

        def _pool_rate(idx, npool, px, py, *, n_meas=120, warmup=30):
            act = _n9_render(px, py)
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            bridge.cp_external_input_current[region_indices_cp["place_sensors"]] = cp.asarray(act, dtype=cp.float32)
            saved = bridge.core_config.reward_learning_rate
            bridge.core_config.reward_learning_rate = 0.0
            spk = 0; m = 0
            for t in range(int(n_meas)):
                _n9_step(1)
                if t >= warmup:
                    spk += int(bridge.cp_firing_states[idx].sum()); m += 1
            bridge.core_config.reward_learning_rate = saved
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            return spk / max(npool, 1) / max(m * 1e-3, 1e-9)

        # place rate (informational; sparse-by-design) + critic rate (the real FIRE target, pre-V-train)
        rate_near = _pool_rate(p_idx, n_place_n, float(_goal0[0]), float(_goal0[1]))
        rate_far = _pool_rate(p_idx, n_place_n, _far[0], _far[1])
        crit_near = _pool_rate(c_idx, n_crit_n, float(_goal0[0]), float(_goal0[1]))
        # (b) PLACE-GRADED: near-vs-far place ensemble cosine (LOW => distinct fields). The
        #     de-risk reports place_diff_cos ~0.06-0.12; "distinct" (<=~0.2) is the PASS, and a
        #     lower value is better (the 'diff' = 1-cos is reported for the design's >=0.05 read).
        ens_near = _n9_place_ensemble(float(_goal0[0]), float(_goal0[1]))
        ens_far = _n9_place_ensemble(_far[0], _far[1])
        _na = float(np.linalg.norm(ens_near)); _nb = float(np.linalg.norm(ens_far))
        diff_cos = float(np.dot(ens_near, ens_far) / (_na * _nb)) if (_na > 0 and _nb > 0) else 1.0
        sparsity = float(0.5 * (np.mean(ens_near > 0) + np.mean(ens_far > 0)))
        # (c) ACTOR-NOT-PERTURBED: the added regions feed ONLY the critic (no edge to the actor
        #     cortex/motor). Measure the actor's mean cortex firing over a short free-running
        #     window WITH the critic regions present; the byte-level guard is that all N9 edges
        #     are place_sensors/place/place_fs/striosome_value/snc — none touch cortex_X/motor_X.
        #     A direct twin (critic-absent) rebuild is out of scope for the cheap probe; instead
        #     we assert the structural isolation + report the actor's live cortex rate as a sanity
        #     baseline (the nav-A/B in Stage C is the quantitative actor-not-perturbed test).
        _cortex_idx = []
        for a in ACTION_NAMES:
            ci = region_indices_cp.get(f"cortex_{a}")
            if ci is not None:
                _cortex_idx.append(ci.get() if hasattr(ci, "get") else np.asarray(ci))
        cortex_rate = float("nan")
        if _cortex_idx:
            cx = cp.asarray(np.concatenate(_cortex_idx), dtype=cp.int64)
            ncx = int(cx.size)
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            # drive place_sensors at the goal (critic active) while the actor free-runs.
            bridge.cp_external_input_current[region_indices_cp["place_sensors"]] = (
                cp.asarray(_n9_render(float(_goal0[0]), float(_goal0[1])), dtype=cp.float32))
            saved = bridge.core_config.reward_learning_rate
            bridge.core_config.reward_learning_rate = 0.0
            spk = 0; m = 0
            for t in range(120):
                _n9_step(1)
                if t >= 40:
                    spk += int(bridge.cp_firing_states[cx].sum()); m += 1
            bridge.core_config.reward_learning_rate = saved
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            cortex_rate = spk / max(ncx, 1) / max(m * 1e-3, 1e-9)
        # Structural actor-isolation assert (the load-bearing actor-not-perturbed guarantee).
        _n9_targets = set()
        for pw in pathways:
            if pw.from_region in ("place_sensors", "place", "place_fs", "striosome_value"):
                _n9_targets.add(pw.to_region)
        _actor_touched = {t for t in _n9_targets
                          if t.startswith("cortex_") or t.startswith("motor_")
                          or t.startswith("str_") or t.startswith("thal_")}
        actor_isolated = (len(_actor_touched) == 0)
        # Phase 1-2 gates: PLACE-GRADED (distinct fields) + ACTOR-ISOLATED. FIRE (critic >=5Hz)
        # is the Phase-3 value-training gate; reported here pre-training as a baseline.
        place_fires = bool(rate_near >= 5.0)        # informational (sparse-by-design; de-risk ~1Hz too)
        critic_fires = bool(crit_near >= 5.0)       # the real FIRE gate (needs Phase-3 value-train)
        graded = bool(diff_cos <= 0.2)
        print("=" * 72)
        print(f"[g11 seed={seed}] N9 STAGE-A SMOKE (place self-org @ nav scale):")
        print(f"  place sparsity={sparsity:.3f} (SPARSE-by-design; de-risk ~0.037)")
        print(f"  (a) place rate (info) : place@near={rate_near:.2f}Hz place@far={rate_far:.2f}Hz "
              f"(>=5Hz => {place_fires}; sparse code is ~1-2Hz, like the de-risk)")
        print(f"      critic FIRE gate  : critic@near={crit_near:.2f}Hz (>=5Hz => {critic_fires}; "
              f"needs Phase-3 value-train to grade)")
        print(f"  (b) PLACE-GRADED [P1-2]: near/far ensemble cos={diff_cos:.3f} "
              f"(diff={1.0-diff_cos:.3f}; distinct<=0.2 => {graded})")
        print(f"  (c) ACTOR-NOT-PERTURBED[P1-2]: cortex_live={cortex_rate:.2f}Hz; N9-edges-touch-actor="
              f"{sorted(_actor_touched) if _actor_touched else 'NONE'} (isolated => {actor_isolated})")
        print(f"  PHASE-1-2 VERDICT: PLACE-GRADED={graded} ACTOR-ISOLATED={actor_isolated} "
              f"(critic-FIRE deferred to Phase 3)")
        print("=" * 72, flush=True)
        return dict(place_rate_near=rate_near, place_rate_far=rate_far, place_sparsity=sparsity,
                    critic_rate_near=crit_near, place_diff_cos=diff_cos,
                    cortex_live_rate=cortex_rate, actor_isolated=actor_isolated,
                    place_fires=place_fires, critic_fires=critic_fires, graded=graded)

    # ═══════════════════════════════════════════════════════════════════════════════════
    # N9 PHASE 3 — STEP-2 value-training (pair-then-reward warm-up) + Stage-B smoke.
    # (2026-06-10) Ports the VALIDATED de-risk STEP-2 (n9_place_graded_critic_stage2_derisk
    # run_seed --pair-then-reward) onto the FROZEN self-organized nav-scale place fields. All
    # neural: the place ensemble fires the coincidence critic, the SNc reward burst is the
    # teacher/US (DA), DA-gated three-factor STDP grows V; only agent placement + reward
    # delivery is environment/body scaffolding (BRAIN-BASED-ONLY). Guarded by neural_place_selforg.
    # ═══════════════════════════════════════════════════════════════════════════════════

    def _n9_far_of(gxgy):
        """The point-reflected 'far' location for a goal (mirror across the arena centre) —
        the de-risk's far convention used in _run_place_selforg / _run_stage_a_smoke."""
        return (float(grid_size) - 1.0 - float(gxgy[0]), float(grid_size) - 1.0 - float(gxgy[1]))

    def _n9_active_place_set(px, py, *, n_meas=80):
        """The set of GLOBAL `place` neuron indices that fire at (px,py) (the location's
        active ensemble core, for per-location LTP tracking; de-risk active_set)."""
        p_idx = region_indices_cp["place"]
        p_host = p_idx.get() if hasattr(p_idx, "get") else np.asarray(p_idx)
        ens = _n9_place_ensemble(float(px), float(py), n_meas=n_meas)
        return set(int(p_host[i]) for i in np.where(np.asarray(ens) > 0)[0])

    def _n9_mean_place_to_value_w(pre_subset):
        """Mean place->striosome_value weight over a SUBSET of place pre-neurons (de-risk _mean_w).
        Used for LEARNS-V (near vs far). Vectorized over the CSR via np.isin."""
        if ("striosome_value" not in region_indices_cp or "place" not in region_indices_cp):
            return 0.0
        post = region_indices_cp["striosome_value"]
        post = post.get() if hasattr(post, "get") else np.asarray(post)
        pre = np.asarray(sorted(int(i) for i in pre_subset), dtype=np.int64)
        if pre.size == 0:
            return 0.0
        coo = bridge.cp_connections.tocoo()
        rows = coo.row.get() if hasattr(coo.row, "get") else np.asarray(coo.row)
        cols = coo.col.get() if hasattr(coo.col, "get") else np.asarray(coo.col)
        data = coo.data.get() if hasattr(coo.data, "get") else np.asarray(coo.data)
        m = np.isin(rows, pre) & np.isin(cols, post)
        if not m.any():
            m = np.isin(rows, post) & np.isin(cols, pre)
        return float(data[m].mean()) if m.any() else 0.0

    def _n9_reset_snc_subtraction_state():
        """Clear the SLOW GABA_B/GIRK conductance + reset the SNc membrane/recovery before a
        calibration / value-train / test phase (de-risk _reset_snc_subtraction_state). The
        residual-conductance fix: g_gabab (tau=150ms => ~150x summation) otherwise carries a
        standing GIRK current from a hard-firing phase into the next, hyperpolarizing the SNc
        so the DA calibration / RPE reads ~0. Zeroing it (+ the SNc V/u) makes each phase clean."""
        s_idx = region_indices_cp.get("snc")
        if s_idx is None:
            return
        if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
            bridge.cp_conductance_g_gabab[:] = cp.float32(0.0)
        if (getattr(bridge, "cp_membrane_potential_v", None) is not None
                and getattr(bridge, "cp_izh_vr", None) is not None):
            bridge.cp_membrane_potential_v[s_idx] = bridge.cp_izh_vr[s_idx]
        if getattr(bridge, "cp_recovery_variable_u", None) is not None:
            bridge.cp_recovery_variable_u[s_idx] = cp.float32(0.0)

    def _n9_reset_critic_read_state(*, gap_steps=80):
        """Clear the critic's SLOW Route-D coincidence (NMDA-spike) plateau conductance + GABA_B +
        reset the critic & SNc membrane/recovery, then run a brief SILENT gap, so each near/far read
        starts from the SAME clean state (the de-risk inter-trial discipline). Without it the plateau
        (tau~80ms) and the up-state CARRY OVER from the previous read -> the second read is contaminated
        by the first (order-dependent, false grading). Brain-faithful: a real inter-trial silent gap."""
        c_idx = region_indices_cp.get("striosome_value")
        for _g in ("cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise"):
            _arr = getattr(bridge, _g, None)
            if _arr is not None:
                _arr[:] = cp.float32(0.0)
        if (c_idx is not None and getattr(bridge, "cp_membrane_potential_v", None) is not None
                and getattr(bridge, "cp_izh_vr", None) is not None):
            bridge.cp_membrane_potential_v[c_idx] = bridge.cp_izh_vr[c_idx]
            if getattr(bridge, "cp_recovery_variable_u", None) is not None:
                bridge.cp_recovery_variable_u[c_idx] = cp.float32(0.0)
        _n9_reset_snc_subtraction_state()   # also clears g_gabab + resets SNc
        bridge.cp_external_input_current[:] = cp.float32(0.0)
        if gap_steps > 0:
            _n9_step(int(gap_steps))

    def _n9_find_da_rule():
        """The dopamine from_region_firing_signed production rule (whose threshold gates DA-LTP)."""
        try:
            for _nm in (bridge.neuromodulator_manager._configs
                        if bridge.neuromodulator_manager is not None else []):
                if _nm.name == "dopamine" and _nm.production_rules:
                    for _pr in _nm.production_rules:
                        if _pr.rule_type == "from_region_firing_signed":
                            return _pr
        except Exception:
            pass
        return None

    def _n9_calibrate_da_threshold(da_rule, *, n_steps=300):
        """Set the DA production threshold to the SNc tonic firing FRACTION under snc_tonic_pa,
        so a phasic reward burst -> DA>baseline -> three-factor LTP (de-risk _calibrate_da).
        Returns the measured tonic fraction. Resets the residual GIRK first (clean read)."""
        s_idx = region_indices_cp["snc"]
        n_snc = int(s_idx.size) if hasattr(s_idx, "size") else len(s_idx)
        _n9_reset_snc_subtraction_state()
        bridge.cp_external_input_current[:] = cp.float32(0.0)
        bridge.cp_external_input_current[s_idx] = cp.float32(snc_tonic_pa)
        frac = 0.0; m = 0
        for i in range(int(n_steps)):
            _n9_step(1)
            if i >= n_steps // 2:
                frac += float(bridge.cp_firing_states[s_idx].sum()) / max(n_snc, 1); m += 1
        tf = frac / max(m, 1)
        if da_rule is not None:
            da_rule.threshold = float(tf)
        bridge.cp_external_input_current[:] = cp.float32(0.0)
        if not (0.005 <= tf <= 0.15) and verbose:
            print(f"[g11 seed={seed}]   [CALIB-WARN] SNc tonic_frac={tf:.4f} outside sane band "
                  f"[0.005,0.15] @ {snc_tonic_pa:.0f}pA ({tf*1000:.0f}Hz) -> DA threshold may "
                  f"mis-set (residual-GIRK / drive).", flush=True)
        return tf

    def _run_place_value_training():
        """N9 STEP-2: pair-then-reward value-training on the FROZEN place fields (de-risk
        run_seed --pair-then-reward loop, VERBATIM protocol). Opens `value_input`, trains the
        place->striosome_value V at the scheduled goal(s) via DA-gated STDP, then freezes it."""
        if not (_neural_place_selforg and value_train_trials > 0
                and "place" in region_indices_cp
                and "striosome_value" in region_indices_cp
                and "snc" in region_indices_cp):
            return None
        t_vt = time.time()
        ps_idx = region_indices_cp["place_sensors"]
        c_idx = region_indices_cp["striosome_value"]
        s_idx = region_indices_cp["snc"]
        n_crit = int(c_idx.size) if hasattr(c_idx, "size") else len(c_idx)
        # The goal locations to value (every distinct scheduled goal, or just the first).
        goals = [g for _, g in goal_schedule_sorted] if critic_warmup_all_goals \
            else [goal_schedule_sorted[0][1]]
        seen = set(); goal_list = []
        for g in goals:
            if g not in seen:
                seen.add(g); goal_list.append(g)
        # Provenance: near/far active place sets + their pre-train weights (gate LEARNS-V).
        near0 = goal_list[0]
        near_set = _n9_active_place_set(near0[0], near0[1])
        far0 = _n9_far_of(near0)
        far_set = _n9_active_place_set(far0[0], far0[1]) - near_set
        w_near_pre = _n9_mean_place_to_value_w(near_set)
        w_far_pre = _n9_mean_place_to_value_w(far_set)
        # DA-threshold calibration (the value-train phase drives the SNc tonic + a reward burst;
        # the nav's hardcoded 0.30 wouldn't let the burst cross threshold -> no LTP gate). Restore
        # the nav's threshold afterwards so the nav loop's SNc RPE dynamics are byte-unchanged.
        da_rule = _n9_find_da_rule()
        saved_da_threshold = (da_rule.threshold if da_rule is not None else None)
        tonic_frac = _n9_calibrate_da_threshold(da_rule)
        saved_reward = bridge.core_config.current_reward_signal
        # CRITIC weight ceiling (2026-06-10): the nav's global stdp_w_max (=150, sized for the
        # actor's cortex->D1 ~125) lets the place->value soft-bound LTP run to w_near~90 -> the MSN
        # critic SATURATES (depolarization block; GRADE inverts as the denser far ensemble out-fires
        # the over-driven near) and the over-strong critic over-CLAMPS the SNc GABA_B to 0. The
        # de-risk validated stdp_w_max=40 -> w_near settles in the graded ~3-6 range. Apply the
        # de-risk ceiling DURING value-train ONLY: the actor is undriven/quiescent here (only
        # place_sensors + snc are driven, OU off), so its cortex->D1 sees NO STDP pre-post events
        # and is NOT collapsed by the lower soft-bound; restored after for the nav loop. value_train_
        # stdp_w_max<=0 => no override (byte-equivalent to the prior behavior).
        saved_w_max = bridge.core_config.stdp_w_max
        if value_train_stdp_w_max and value_train_stdp_w_max > 0:
            bridge.core_config.stdp_w_max = float(value_train_stdp_w_max)
        bridge.set_plasticity_gate("value_input", 1.0)        # open the critic arm
        bridge.set_plasticity_gate("landmark_to_place", 0.0)  # place fields stay FROZEN
        bridge.set_transmission_gate("place_fs_gate", 1.0)    # FS-PING volley ON for the read
        # NOTE: FS->critic inhibition (critic_fs_gate) is held ON THROUGHOUT (value-train AND
        # read-out/nav) so the critic learns AND reads V in ONE consistent CLAMPED regime — the
        # (B) de-risk GO (gating it OFF for value-train learns a STRONGER V that then over-fires the
        # UNGATED read-out -> binary δ; the self-consistent clamped regime grades cleanly).
        pair_steps = int(value_train_pair_steps)
        hold = int(value_train_hold_steps)
        rdelay = int(reward_delay_steps)
        near_v_first = None; near_v_last = None
        for (wgx, wgy) in goal_list:
            place_act = cp.asarray(_n9_render(float(wgx), float(wgy)), dtype=cp.float32)
            for _t in range(int(value_train_trials)):
                # (1) ITI floor: SNc tonic, no place drive, then zero eligibility (de-risk ITI).
                bridge.core_config.current_reward_signal = 0.0
                bridge.cp_external_input_current[:] = cp.float32(0.0)
                bridge.cp_external_input_current[s_idx] = cp.float32(snc_tonic_pa)
                _n9_step(hold)
                if bridge.cp_eligibility_trace is not None:
                    bridge.cp_eligibility_trace[:] = cp.float32(0.0)
                # (2) PAIR: place drive ON + SNc at TONIC (DA baseline) for pair_steps. The
                #     bistable MSN-D1 climbs into the up-state; place-pre x critic-post STDP lays
                #     a clean, net-positive, SILENT eligibility trace (DA-baseline => ~0 weight
                #     converted, so the trace ACCUMULATES). The sub-threshold phase-locked teacher
                #     (critic_teacher_pa) on striosome_value fires the weak-drive volley
                #     phase-locked; it is present ONLY in PAIR (removed at REWARD + read-out).
                bridge.core_config.current_reward_signal = 0.0
                bridge.cp_external_input_current[:] = cp.float32(0.0)
                bridge.cp_external_input_current[ps_idx] = place_act
                if critic_teacher_pa > 0.0:
                    bridge.cp_external_input_current[c_idx] = cp.float32(critic_teacher_pa)
                bridge.cp_external_input_current[s_idx] = cp.float32(snc_tonic_pa)
                spk = 0; n_meas = 0
                for _ in range(pair_steps):
                    _n9_step(1)
                    spk += int(bridge.cp_firing_states[c_idx].sum()); n_meas += 1
                # (3) REWARD: place STILL ON (cell holds the up-state), teacher REMOVED; after an
                #     ~reward_delay_steps lag (DA rises AFTER the pairing — Yagishita), inject the
                #     SNc BURST -> DA>baseline -> converts the accumulated eligibility (robust LTP).
                bridge.core_config.current_reward_signal = 1.0
                bridge.cp_external_input_current[:] = cp.float32(0.0)
                bridge.cp_external_input_current[ps_idx] = place_act
                bridge.cp_external_input_current[s_idx] = cp.float32(snc_tonic_pa)
                if rdelay > 0:
                    _n9_step(rdelay)                          # pairing-then-DA lag (place still on)
                bridge.cp_external_input_current[s_idx] = cp.float32(snc_tonic_pa + snc_reward_gain)
                for _ in range(hold):
                    _n9_step(1)
                    spk += int(bridge.cp_firing_states[c_idx].sum()); n_meas += 1
                # (4) reset the SLOW GABA_B/GIRK + SNc membrane (de-risk per-trial reset).
                _n9_reset_snc_subtraction_state()
                _v = spk / max(n_crit, 1) / max(n_meas * 1e-3, 1e-9)
                if near0 == (wgx, wgy):
                    if near_v_first is None:
                        near_v_first = _v
                    near_v_last = _v
                if os.environ.get("VALUE_TRAIN_DEBUG") and (_t < 3 or _t % 10 == 0
                                                            or _t == int(value_train_trials) - 1):
                    wn = _n9_mean_place_to_value_w(near_set)
                    wf = _n9_mean_place_to_value_w(far_set)
                    da = (bridge.neuromodulator_manager.get_concentration("dopamine")
                          if bridge.neuromodulator_manager is not None else float("nan"))
                    print(f"[VALUE_TRAIN goal=({wgx},{wgy}) t={_t:02d}] V={_v:6.2f}Hz "
                          f"w_near={wn:.3f} w_far={wf:.3f} (near/far {wn/max(wf,1e-6):.2f}) "
                          f"DA={da:.3f}", flush=True)
        bridge.set_plasticity_gate("value_input", 0.0)        # FREEZE V for the read-out / nav
        bridge.core_config.current_reward_signal = saved_reward
        bridge.cp_external_input_current[:] = cp.float32(0.0)
        if bridge.cp_eligibility_trace is not None:
            bridge.cp_eligibility_trace[:] = cp.float32(0.0)
        _n9_reset_snc_subtraction_state()
        # RESTORE the actor's soft-bound ceiling + the nav's hardcoded DA threshold so the nav loop
        # (SNc RPE + actor cortex->D1) is byte-unchanged.
        bridge.core_config.stdp_w_max = float(saved_w_max)
        if da_rule is not None and saved_da_threshold is not None:
            da_rule.threshold = float(saved_da_threshold)
        w_near_post = _n9_mean_place_to_value_w(near_set)
        w_far_post = _n9_mean_place_to_value_w(far_set)
        stats = dict(
            n_goals=len(goal_list), value_train_trials=int(value_train_trials),
            tonic_frac=float(tonic_frac), critic_teacher_pa=float(critic_teacher_pa),
            near_set_size=len(near_set), far_set_size=len(far_set),
            w_near_pre=float(w_near_pre), w_near_post=float(w_near_post),
            w_far_pre=float(w_far_pre), w_far_post=float(w_far_post),
            w_near_over_far_post=float(w_near_post / max(w_far_post, 1e-6)),
            near_v_first_hz=(float(near_v_first) if near_v_first is not None else None),
            near_v_last_hz=(float(near_v_last) if near_v_last is not None else None),
            # cache the near/far sets for the Stage-B smoke (so it doesn't re-derive them).
            _near_set=near_set, _far_set=far_set, near0=near0, far0=far0)
        if verbose:
            print(f"[g11 seed={seed}] N9 STEP-2 value-training done ({time.time()-t_vt:.0f}s): "
                  f"{value_train_trials} pair-then-reward trials x {len(goal_list)} goal(s); "
                  f"w_near {w_near_pre:.3f}->{w_near_post:.3f} w_far {w_far_pre:.3f}->{w_far_post:.3f} "
                  f"(near/far {stats['w_near_over_far_post']:.2f}); V(near) "
                  f"{near_v_first}->{near_v_last}Hz (place fields FROZEN; V FROZEN)", flush=True)
        return stats

    def _n9_critic_boundary_diag(near0, far0, c_idx, ps_idx):
        """N9 deploy diagnostic (env N9_DIAG): localize WHERE the critic read breaks at nav scale.
        For near AND far, drive place_sensors over a read window and log the component-boundary
        signals (systematic-debugging Phase 1):
          - place  : `place` pool rate (is the self-org code firing a volley at all?)
          - g_coinc: max Route-D plateau conductance on the critic (>0 => the FS-PING-synchronized
                     volley delivered c_i>=K coincident spikes and triggered the supralinear
                     plateau; ~0 => c_i<K, the volley is NOT coincident at nav scale)
          - V(crit): critic membrane (rest ~ -79.6mV => no current integrated; climbing => drive arrives)
          - critic : critic firing rate.
        Assumes the caller has toggled the WEIGHTED plateau read-out ON (matches the real read)."""
        p_idx = region_indices_cp.get("place")
        gco = getattr(bridge, "cp_conductance_g_coincidence", None)
        n_crit_n = int(c_idx.size) if hasattr(c_idx, "size") else len(c_idx)
        n_place_n = (int(p_idx.size) if (p_idx is not None and hasattr(p_idx, "size"))
                     else (len(p_idx) if p_idx is not None else 0))
        _sv_wd = bool(getattr(bridge.core_config, "coincidence_weighted_drive", False))
        _sv_kth = float(getattr(bridge.core_config, "coincidence_k_threshold", 4.0))

        def _measure(px, py):
            _n9_reset_critic_read_state()   # clean state per read (no carryover between near/far)
            act = cp.asarray(_n9_render(float(px), float(py)), dtype=cp.float32)
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            bridge.cp_external_input_current[ps_idx] = act
            saved = bridge.core_config.reward_learning_rate
            bridge.core_config.reward_learning_rate = 0.0
            pspk = 0; cspk = 0; m = 0
            gco_max = 0.0; v_max = -1e9; v_sum = 0.0
            for t in range(120):
                _n9_step(1)
                if t >= 30:
                    if p_idx is not None:
                        pspk += int(bridge.cp_firing_states[p_idx].sum())
                    cspk += int(bridge.cp_firing_states[c_idx].sum())
                    if gco is not None:
                        gco_max = max(gco_max, float(bridge.cp_conductance_g_coincidence[c_idx].max()))
                    _vc = bridge.cp_membrane_potential_v[c_idx]
                    v_max = max(v_max, float(_vc.max())); v_sum += float(_vc.mean())
                    m += 1
            bridge.core_config.reward_learning_rate = saved
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            return (pspk / max(n_place_n, 1) / max(m * 1e-3, 1e-9),
                    gco_max, v_sum / max(m, 1), v_max,
                    cspk / max(n_crit_n, 1) / max(m * 1e-3, 1e-9))

        # COUNT form (weight-blind; the train-time plateau): does the FS-PING volley deliver
        # c_i>=K_train coincident spikes at nav scale AT ALL? + WEIGHTED form (the read-out grading).
        for form, wd, kth in (("COUNT", False, float(coincidence_train_k)),
                              ("WGHTD", True, float(coincidence_threshold))):
            bridge.core_config.coincidence_weighted_drive = wd
            bridge.core_config.coincidence_k_threshold = kth
            for tag, (px, py) in (("near", near0), ("far", far0)):
                ph, gx, vm, vmax, ch = _measure(px, py)
                print(f"  [N9_DIAG {form} {tag}] place={ph:.2f}Hz  g_coinc(crit)max={gx:.3f}  "
                      f"V(crit) mean={vm:.1f}mV max={vmax:.1f}mV  critic={ch:.2f}Hz", flush=True)
        bridge.core_config.coincidence_weighted_drive = _sv_wd
        bridge.core_config.coincidence_k_threshold = _sv_kth

    def _run_stage_b_smoke(vt_stats):
        """N9 Stage-B probe (design §Stage B): the load-bearing critic gate after STEP-1+STEP-2.
          LEARNS-V        : near place->striosome_value weight >= 1.5x far.
          CRITIC FIRE+GRADE: with the WEIGHTED plateau toggled ON at read-out
                             (coincidence_weighted_drive=True, k_threshold=coincidence_threshold),
                             drive place_sensors at the goal -> striosome_value >=5Hz AND near >=3x far.
          GABA_B gap (d=r-V): place@goal + SNc burst -> snc rate (predicted); place@far + SNc burst
                             -> snc rate (unpredicted); predicted < unpredicted (learned V subtracts).
                             + LESION control: zero g_gabab mask -> gap -> ~1.0.
        Reports all numbers honestly; does NOT proceed to the nav loop (caller exits)."""
        c_idx = region_indices_cp["striosome_value"]
        s_idx = region_indices_cp["snc"]
        ps_idx = region_indices_cp["place_sensors"]
        n_crit = int(c_idx.size) if hasattr(c_idx, "size") else len(c_idx)
        n_snc = int(s_idx.size) if hasattr(s_idx, "size") else len(s_idx)
        near0 = vt_stats["near0"] if vt_stats else goal_schedule_sorted[0][1]
        far0 = vt_stats["far0"] if vt_stats else _n9_far_of(near0)
        near_set = vt_stats["_near_set"] if vt_stats else _n9_active_place_set(near0[0], near0[1])
        far_set = (vt_stats["_far_set"] if vt_stats
                   else _n9_active_place_set(far0[0], far0[1]) - near_set)

        # ── LEARNS-V ──
        w_near = _n9_mean_place_to_value_w(near_set)
        w_far = _n9_mean_place_to_value_w(far_set)
        learns_v = bool(w_near >= 1.5 * max(w_far, 1e-6))

        # ── CRITIC FIRE + GRADE (the real FIRE gate): WEIGHTED-plateau read-out toggle ──
        def _critic_rate(px, py, *, n_meas=120, warmup=30):
            _n9_reset_critic_read_state()   # clean state per read (no near<->far plateau carryover)
            act = cp.asarray(_n9_render(float(px), float(py)), dtype=cp.float32)
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            bridge.cp_external_input_current[ps_idx] = act
            saved = bridge.core_config.reward_learning_rate
            bridge.core_config.reward_learning_rate = 0.0
            spk = 0; m = 0
            for t in range(int(n_meas)):
                _n9_step(1)
                if t >= warmup:
                    spk += int(bridge.cp_firing_states[c_idx].sum()); m += 1
            bridge.core_config.reward_learning_rate = saved
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            return spk / max(n_crit, 1) / max(m * 1e-3, 1e-9)

        # READ-OUT toggle to the Poirazi-Mel WEIGHTED subunit so the LEARNED w_near (grown) fires
        # the critic while the unlearned w_far (~init) cannot -> grading from the weight itself.
        # k_threshold swaps to WEIGHT units (coincidence_threshold). Restored after the read.
        _saved_wd = bool(getattr(bridge.core_config, "coincidence_weighted_drive", False))
        _saved_kth = float(getattr(bridge.core_config, "coincidence_k_threshold", 4.0))
        bridge.core_config.coincidence_weighted_drive = True
        bridge.core_config.coincidence_k_threshold = float(coincidence_threshold)
        crit_near = _critic_rate(near0[0], near0[1])
        crit_far = _critic_rate(far0[0], far0[1])
        if os.environ.get("N9_DIAG"):
            _n9_critic_boundary_diag(near0, far0, c_idx, ps_idx)
        bridge.core_config.coincidence_weighted_drive = _saved_wd
        bridge.core_config.coincidence_k_threshold = _saved_kth
        crit_grade_ratio = crit_near / max(crit_far, 1e-3)
        critic_fire = bool(crit_near >= 5.0)
        critic_grade = bool(crit_near >= 5.0 and crit_grade_ratio >= 3.0)

        # ── GABA_B gap (delta = r - V): predicted(NEAR) < unpredicted(FAR) + lesion control ──
        def _snc_burst_rate(px, py):
            _n9_reset_critic_read_state()   # clean critic plateau + GABA_B + SNc per read
            act = cp.asarray(_n9_render(float(px), float(py)), dtype=cp.float32)
            # LEAD: place drive (critic fires -> GABA_B onto SNc) BEFORE the reward burst.
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            bridge.cp_external_input_current[ps_idx] = act
            bridge.cp_external_input_current[s_idx] = cp.float32(snc_tonic_pa)
            _n9_step(int(critic_lead_steps))
            # REWARD burst (place still on). When spiking_reward_us, the `r` term is produced by the
            # reward_us US afferent FIRING into the SNc (the spiking reward), NOT a host write -> this
            # measures the FULLY-SPIKING δ=r−V (reward_us r + critic GABA_B V) in the real config.
            if spiking_reward_us and "reward_us" in region_indices_cp:
                bridge.cp_external_input_current[s_idx] = cp.float32(snc_tonic_pa)
                bridge.cp_external_input_current[region_indices_cp["reward_us"]] = cp.float32(reward_us_drive_pa)
            else:
                bridge.cp_external_input_current[s_idx] = cp.float32(snc_tonic_pa + snc_reward_gain)
            saved = bridge.core_config.reward_learning_rate
            bridge.core_config.reward_learning_rate = 0.0
            spk = 0
            for _ in range(int(value_train_hold_steps)):
                _n9_step(1)
                spk += int(bridge.cp_firing_states[s_idx].sum())
            bridge.core_config.reward_learning_rate = saved
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            return spk / max(n_snc, 1) / max(int(value_train_hold_steps) * 1e-3, 1e-9)

        # The gate-2e read needs the WEIGHTED plateau too (so the critic fires DIFFERENTIALLY
        # near>>far during the LEAD -> a differential GABA_B). Toggle around the gap reads.
        bridge.core_config.coincidence_weighted_drive = True
        bridge.core_config.coincidence_k_threshold = float(coincidence_threshold)
        snc_pred = _snc_burst_rate(near0[0], near0[1])     # predicted (NEAR; V subtracts)
        snc_unpred = _snc_burst_rate(far0[0], far0[1])     # unpredicted (FAR; no V)
        bridge.core_config.coincidence_weighted_drive = _saved_wd
        bridge.core_config.coincidence_k_threshold = _saved_kth
        gap_ratio = snc_unpred / max(snc_pred, 1e-6)
        gabab_gap = bool(snc_unpred > 1.30 * max(snc_pred, 1e-6))

        # LESION control: zero the GABA_B mask -> the predicted/unpredicted gap must vanish (~1.0).
        n_cut = 0
        snc_pred_les = float("nan"); snc_unpred_les = float("nan"); gap_les = float("nan")
        m_mask = getattr(bridge, "cp_gabab_synapse_mask", None)
        if m_mask is not None:
            _saved_mask = m_mask.copy()
            n_cut = int((m_mask.get() if hasattr(m_mask, "get") else np.asarray(m_mask)).sum())
            bridge.cp_gabab_synapse_mask = cp.zeros_like(m_mask)
            if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
                bridge.cp_conductance_g_gabab[:] = cp.float32(0.0)
            bridge.core_config.coincidence_weighted_drive = True
            bridge.core_config.coincidence_k_threshold = float(coincidence_threshold)
            snc_pred_les = _snc_burst_rate(near0[0], near0[1])
            snc_unpred_les = _snc_burst_rate(far0[0], far0[1])
            bridge.core_config.coincidence_weighted_drive = _saved_wd
            bridge.core_config.coincidence_k_threshold = _saved_kth
            gap_les = snc_unpred_les / max(snc_pred_les, 1e-6)
            bridge.cp_gabab_synapse_mask = _saved_mask   # restore (clean nav, though we exit)
            if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
                bridge.cp_conductance_g_gabab[:] = cp.float32(0.0)
        lesion_collapses = bool(m_mask is not None and gap_les <= 1.15)

        print("=" * 72)
        print(f"[g11 seed={seed}] N9 STAGE-B SMOKE (value-learning @ nav scale):")
        print(f"  near={tuple(round(c,1) for c in near0)} far={tuple(round(c,1) for c in far0)} "
              f"(near_set={len(near_set)} far_set={len(far_set)} place cells)")
        print(f"  [LEARNS-V]  w_near={w_near:.3f} w_far={w_far:.3f} (near/far "
              f"{w_near/max(w_far,1e-6):.2f}; >=1.5x => {learns_v})")
        print(f"  [CRITIC FIRE+GRADE] (weighted plateau, k={coincidence_threshold}) "
              f"critic@near={crit_near:.2f}Hz critic@far={crit_far:.2f}Hz "
              f"(>=5Hz & near>=3xfar; fire={critic_fire} grade={critic_grade})")
        print(f"  [GABA_B gap d=r-V]  predicted(NEAR)={snc_pred:.2f}Hz unpredicted(FAR)={snc_unpred:.2f}Hz "
              f"gap={gap_ratio:.2f} (unpred>1.3x pred => {gabab_gap})")
        if m_mask is not None:
            print(f"  [LESION control]    zeroed {n_cut} GABA_B synapses -> "
                  f"pred={snc_pred_les:.2f}Hz unpred={snc_unpred_les:.2f}Hz gap={gap_les:.2f} "
                  f"(collapses to ~1.0 => {lesion_collapses})")
        else:
            print(f"  [LESION control]    no GABA_B mask present (skipped)")
        _crit_at_nav = bool(critic_fire and critic_grade)
        print(f"  STAGE-B VERDICT: LEARNS-V={learns_v} CRITIC-FIRE+GRADE={_crit_at_nav} "
              f"GABA_B-gap={gabab_gap} lesion-collapses={lesion_collapses}")
        if not _crit_at_nav:
            print(f"  [HONEST] critic does NOT fire+grade at nav scale "
                  f"(critic@near={crit_near:.2f}Hz, near/far={crit_grade_ratio:.2f}) — "
                  f"design §Risk 'place-code self-org at nav scale' / 'actor-critic interaction'.")
        print("=" * 72, flush=True)
        return dict(
            near=list(near0), far=list(far0),
            near_set_size=len(near_set), far_set_size=len(far_set),
            w_near=float(w_near), w_far=float(w_far),
            w_near_over_far=float(w_near / max(w_far, 1e-6)), learns_v=learns_v,
            crit_near_hz=float(crit_near), crit_far_hz=float(crit_far),
            crit_grade_ratio=float(crit_grade_ratio),
            critic_fire=critic_fire, critic_grade=critic_grade,
            critic_fires_and_grades_at_nav=_crit_at_nav,
            snc_predicted_near_hz=float(snc_pred), snc_unpredicted_far_hz=float(snc_unpred),
            snc_gap_ratio=float(gap_ratio), gabab_gap=gabab_gap,
            lesion_n_cut=int(n_cut),
            snc_pred_lesion_hz=(float(snc_pred_les) if m_mask is not None else None),
            snc_unpred_lesion_hz=(float(snc_unpred_les) if m_mask is not None else None),
            lesion_gap_ratio=(float(gap_les) if m_mask is not None else None),
            lesion_collapses=lesion_collapses)

    # ===== CRITIC VALUE-ACQUISITION WARM-UP (2026-06-09 deadlock-breaker) =====
    # Before the nav loop, seed the MSN-D1 value critic with the de-risk's VALIDATED
    # value-leads-reward protocol at the scheduled goal location(s), at the BASELINE
    # homeostasis rate (which preserves place-selectivity — the de-risk gate-5 PASS).
    # This breaks the LTP-bootstrap deadlock the 1800-step nav can't (the critic must
    # fire to seed STDP eligibility, but at the init weight it can't fire from a
    # free-moving agent's brief per-location visits). All neural (afferent fires ->
    # critic fires -> SNc bursts -> DA -> three-factor LTP grows vs_place->value);
    # only the agent placement + reward delivery is environment/body scaffolding.
    def _run_critic_warmup():
        if not (enable_neural_critic and critic_warmup_trials > 0
                and "vs_place_context" in region_indices_cp
                and "striosome_value" in region_indices_cp
                and "snc" in region_indices_cp):
            return None
        aff_idx = region_indices_cp["vs_place_context"]
        # A1 up-state arm (convergent-upstate, opt-in): the warm-up must also drive vs_place_drive
        # so the critic FIRES (gives the A2 plastic synapses a post-spike to pair with).
        drive_idx = (region_indices_cp["vs_place_drive"]
                     if (enable_convergent_upstate and "vs_place_drive" in region_indices_cp) else None)
        snc_idx = region_indices_cp["snc"]
        # The goal locations to value: every scheduled goal (multi-goal) or just the
        # first. Dedupe preserving order so each distinct goal is warmed once.
        goals = [g for _, g in goal_schedule_sorted] if critic_warmup_all_goals \
            else [goal_schedule_sorted[0][1]]
        seen = set(); goal_list = []
        for g in goals:
            if g not in seen:
                seen.add(g); goal_list.append(g)
        w_pre = _mean_critic_weight()
        hold = int(critic_warmup_hold_steps)
        saved_reward = bridge.core_config.current_reward_signal
        # --- DA-threshold calibration (the de-risk's _calibrate_da_threshold) ---
        # CRITICAL: the nav HARDCODES the dopamine production-rule threshold at the SNc
        # tonic firing FRACTION 0.30, tuned for the nav loop's SNc RPE dynamics. But the
        # warm-up drives the SNc differently (tonic floor + a reward burst), and at 0.30
        # the burst does NOT cross threshold -> DA DECAYS instead of rising -> the three-
        # factor LTP is never gated -> the weight stays frozen (forensic: warm-up trials
        # at threshold 0.30 gave crit_spk=0, DA 0.35->0.15, weight unchanged). The de-risk
        # CALIBRATES the threshold to the SNc's measured tonic fraction (~0.02) so a reward
        # burst -> DA>baseline -> LTP. Replicate that HERE for the warm-up, then RESTORE the
        # nav's 0.30 so the nav loop's SNc RPE is byte-unchanged. Find the dopamine rule.
        _da_rule = None
        try:
            for _nm in (bridge.neuromodulator_manager._configs
                        if bridge.neuromodulator_manager is not None else []):
                if _nm.name == "dopamine" and _nm.production_rules:
                    for _pr in _nm.production_rules:
                        if _pr.rule_type == "from_region_firing_signed":
                            _da_rule = _pr; break
                if _da_rule is not None:
                    break
        except Exception:
            _da_rule = None
        _saved_da_threshold = (_da_rule.threshold if _da_rule is not None else None)
        if _da_rule is not None:
            # Measure the SNc tonic firing fraction under the warm-up's tonic drive (300
            # steps, average over the back half — exactly _calibrate_da_threshold).
            _n_snc = int(snc_idx.size) if hasattr(snc_idx, "size") else len(snc_idx)
            bridge.core_config.current_reward_signal = 0.0
            bridge.cp_external_input_current[:] = cp.float32(0.0)
            bridge.cp_external_input_current[snc_idx] = cp.float32(snc_tonic_pa)
            _frac_sum = 0.0; _m = 0
            for _i in range(300):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                bridge.runtime_state.current_time_ms = (
                    bridge.runtime_state.current_time_step * cfg.dt_ms)
                if _i >= 150:
                    _frac_sum += float(bridge.cp_firing_states[snc_idx].sum()) / max(_n_snc, 1)
                    _m += 1
            _tonic_frac = _frac_sum / max(_m, 1)
            _da_rule.threshold = float(_tonic_frac)
        # WARMUP_DRIVE_MULT (env, default 1.0): a diagnostic knob for the warm-up's afferent
        # drive strength. The 2026-06-09 forensic showed that even 10x drive (g_exc 0.5 >
        # the de-risk's firing 0.35) did NOT fire the MSN-D1 critic in the deployed nav
        # bridge (the critic's membrane plateaus at ~-79.6 mV where the byte-identical de-risk
        # critic integrates to -71 mV on the same g_exc — the unresolved nav-bridge blocker).
        _drive_mult = float(os.environ.get("WARMUP_DRIVE_MULT", "1.0"))
        for (wgx, wgy) in goal_list:
            # the (x,y)->dense place-code drive, the SAME rendering the nav loop uses.
            vs_dsq = (vs_place_pref_x - float(wgx)) ** 2 + (vs_place_pref_y - float(wgy)) ** 2
            vs_drive = (_drive_mult * vs_place_drive_max_pA) * np.exp(-vs_dsq / (2.0 * vs_place_sigma ** 2))
            vs_drive_cp = cp.asarray(vs_drive, dtype=cp.float32)
            for _t in range(int(critic_warmup_trials)):
                # (1) ITI floor: SNc tonic only, no place drive, no reward -> clears
                #     any lingering depolarization (de-risk's inter-trial interval).
                bridge.core_config.current_reward_signal = 0.0
                bridge.cp_external_input_current[:] = cp.float32(0.0)
                bridge.cp_external_input_current[snc_idx] = cp.float32(snc_tonic_pa)
                for _ in range(hold):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                    bridge.runtime_state.current_time_ms = (
                        bridge.runtime_state.current_time_step * cfg.dt_ms)
                # (2) clear eligibility (de-risk clears between trials so each trial's
                #     LTP reflects THIS location's coincidence, not a carry-over).
                if bridge.cp_eligibility_trace is not None:
                    bridge.cp_eligibility_trace[:] = cp.float32(0.0)
                # (3) LEARN: drive the place code at the goal + a reward burst on the
                #     SNc (the value-leads-reward pairing -> DA -> three-factor LTP).
                bridge.core_config.current_reward_signal = 1.0
                bridge.cp_external_input_current[:] = cp.float32(0.0)
                bridge.cp_external_input_current[aff_idx] = vs_drive_cp
                if drive_idx is not None:
                    bridge.cp_external_input_current[drive_idx] = vs_drive_cp   # A1 up-state arm
                bridge.cp_external_input_current[snc_idx] = cp.float32(snc_tonic_pa + snc_reward_gain)
                _wu_crit_spk = 0
                for _ in range(hold):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                    bridge.runtime_state.current_time_ms = (
                        bridge.runtime_state.current_time_step * cfg.dt_ms)
                    if os.environ.get("WARMUP_DEBUG"):
                        _wu_crit_spk += int(bridge.cp_firing_states[
                            region_indices_cp["striosome_value"]].sum())
                if os.environ.get("WARMUP_DEBUG") and (_t < 2 or _t == int(critic_warmup_trials) - 1):
                    _da = (bridge.neuromodulator_manager.get_concentration("dopamine")
                           if bridge.neuromodulator_manager is not None else float("nan"))
                    print(f"[WARMUP_DEBUG goal=({wgx},{wgy}) trial={_t}] crit_spk={_wu_crit_spk} "
                          f"DA={_da:.3f} w={_mean_critic_weight():.4f}", flush=True)
        # restore pre-warmup transient state for a clean nav start.
        bridge.core_config.current_reward_signal = saved_reward
        bridge.cp_external_input_current[:] = cp.float32(0.0)
        if bridge.cp_eligibility_trace is not None:
            bridge.cp_eligibility_trace[:] = cp.float32(0.0)
        # RESTORE the nav's hardcoded DA threshold (0.30) so the nav loop's SNc RPE
        # dynamics are byte-unchanged — the warm-up's calibrated threshold was ONLY for
        # the warm-up's LTP gating.
        if _da_rule is not None and _saved_da_threshold is not None:
            _da_rule.threshold = float(_saved_da_threshold)
        w_post = _mean_critic_weight()
        return (w_pre, w_post, len(goal_list))

    # N9 STEP-1: self-organize the place fields BEFORE the value warm-up / nav (the value critic
    # reads the FROZEN place code). Only runs under neural_place_selforg.
    _selforg_stats = _run_place_selforg()

    # N9 Stage-A cheap-first probe: measure FIRE / PLACE-GRADED / ACTOR-NOT-PERTURBED and exit
    # BEFORE the nav loop (the design's cheap-first integration de-risk). Stage A is pre-STEP-2
    # (it gates the place-code self-org alone); STEP-2 value-training + Stage B come after.
    if stage_a_smoke and _neural_place_selforg:
        _stage_a = _run_stage_a_smoke()
        if _stage_a is not None:
            _stage_a["place_selforg"] = _selforg_stats
        return {"stage_a_smoke": _stage_a, "selforg": _selforg_stats}

    # N9 STEP-2 (Phase 3): pair-then-reward value-training on the FROZEN place fields. Grows the
    # place->striosome_value V via DA-gated STDP, then freezes it. Only under neural_place_selforg
    # (value_train_trials>0); a no-op otherwise (byte-equivalent — the legacy warm-up is below).
    _value_train_stats = _run_place_value_training()

    # N9 Stage-B smoke (the load-bearing critic gate): LEARNS-V / CRITIC FIRE+GRADE / GABA_B gap
    # + lesion, after STEP-1+STEP-2. Exits BEFORE the nav loop (the design's Stage B de-risk).
    if stage_b_smoke and _neural_place_selforg:
        _stage_b = _run_stage_b_smoke(_value_train_stats)
        # drop the cached index-set objects (not JSON-friendly) from the value-train stats blob.
        _vt_clean = ({k: v for k, v in _value_train_stats.items() if not k.startswith("_")}
                     if _value_train_stats is not None else None)
        return {"stage_b_smoke": _stage_b, "value_train": _vt_clean,
                "selforg": _selforg_stats}

    _warmup_stats = _run_critic_warmup()
    if _warmup_stats is not None and verbose:
        _wp, _wq, _ng = _warmup_stats
        print(f"[g11 seed={seed}] CRITIC WARM-UP: {critic_warmup_trials} reward-paired "
              f"trials x {_ng} goal(s) -> vs_place->value weight {_wp:.4f} -> {_wq:.4f} "
              f"(deadlock-breaker; baseline homeostasis preserves place-selectivity)", flush=True)

    t0 = time.time()
    # Track current gating_strength (used for DA-gated WTA across the whole trial,
    # not just the reward-hold sub-step). Initialized to 1.0 (full WTA on first trial
    # before any reward feedback exists).
    current_gating_strength = 1.0
    visual_cortex_action_gate_opened = False
    for step in range(n_steps):
        # ----- Stage 2 windowed critic->SNc GABA_B gate (2026-06-08 redesign) -----
        # Managed at the NAV-STEP granularity (one nav step = n_stim_steps sub-steps
        # = ~100 ms ≈ 0.67 GABA_B tau). In WINDOWED mode, run a sawtooth: OPEN the
        # gate for a bounded lead of ~critic_lead_steps (converted to whole nav
        # steps), then CLOSE it for an equal flush phase so g_gabab decays and
        # cannot integrate across a long multi-step dwell (the de-risk's >=200 ms
        # over-suppression boundary, d0416fc3). The reward block below FORCE-OPENS
        # the gate for its hold loop regardless of phase (the subtraction must be
        # live at reward). Stage 1 (enable_critic_window=False): the gate was set
        # OPEN once before the loop and is never touched here -> continuous.
        if _critic_gate_known and enable_critic_window:
            # lead length in whole nav steps (>=1); equal-length flush phase.
            _lead_nav = max(1, int(round(critic_lead_steps / max(1, n_stim_steps))))
            _phase = _critic_open_counter % (2 * _lead_nav)
            if _phase < _lead_nav:
                bridge.set_transmission_gate("critic_snc_window", 1.0)   # pre-build lead
            else:
                bridge.set_transmission_gate("critic_snc_window", 0.0)   # flush (decay)
            _critic_open_counter += 1
        # ----- ADAPTIVE / activity-gated heuristic weaning scheduler (N1) -----
        # Run BEFORE the per-step h_strength decision below. While in the
        # "teaching" phase, periodically open an OFF probe window (heuristic
        # silenced) to measure whether the learned IT->cortex mapping can
        # navigate self-sufficiently. If a probe shows readiness (mean distance
        # over the window <= wean_probe_threshold), COMMIT the wean; otherwise
        # resume teaching until the next probe.
        if heuristic_wean_adaptive and adaptive_phase == "teaching":
            if adaptive_probe_active:
                # Are we at the first step AFTER the active probe window? If so,
                # evaluate the distances recorded during the window (the last
                # _wpw entries of distance_log are exactly the probe-window steps).
                if step >= adaptive_probe_start_step + _wpw:
                    probe_dists = distance_log[-_wpw:]
                    mean_probe_dist = float(np.mean(probe_dists)) if probe_dists else float("inf")
                    committed = mean_probe_dist <= float(wean_probe_threshold)
                    adaptive_probe_history.append({
                        "probe_start": int(adaptive_probe_start_step),
                        "probe_end": int(step),
                        "mean_dist": mean_probe_dist,
                        "committed": bool(committed),
                    })
                    adaptive_probe_active = False
                    if committed:
                        adaptive_phase = "committed"
                        adaptive_commit_step = step
                        if verbose:
                            print(f"[g11 seed={seed}] step {step}: ADAPTIVE WEAN "
                                  f"COMMITTED (probe mean dist {mean_probe_dist:.2f} "
                                  f"<= threshold {wean_probe_threshold}); ramping "
                                  f"heuristic to 0 over {heuristic_wean_steps} steps",
                                  flush=True)
                    else:
                        if verbose:
                            print(f"[g11 seed={seed}] step {step}: adaptive probe "
                                  f"NOT ready (mean dist {mean_probe_dist:.2f} > "
                                  f"threshold {wean_probe_threshold}); resume teaching",
                                  flush=True)
            else:
                # Open a new probe window every _wpe steps (first probe at step _wpe).
                if step > 0 and step % _wpe == 0:
                    adaptive_probe_active = True
                    adaptive_probe_start_step = step
                    if verbose:
                        print(f"[g11 seed={seed}] step {step}: adaptive readiness "
                              f"PROBE start (heuristic OFF for {_wpw} steps)",
                              flush=True)

        # Cluster K v2 visual cortex critical-period close: open the
        # IT -> cortex_X gate at the configured warmup step. Mimics real
        # visuomotor development: V1/V2/IT mature first (sensory critical
        # period), then visuomotor wiring matures via STDP+reward.
        if (enable_visual_cortex
                and not visual_cortex_action_gate_opened
                and visual_cortex_action_warmup_steps >= 0
                and step >= visual_cortex_action_warmup_steps):
            try:
                bridge.set_plasticity_gate("visual_cortex_action", 1.0)
                visual_cortex_action_gate_opened = True
                if verbose:
                    print(f"[g11 seed={seed}] step {step}: Cluster K v2 "
                          f"visual_cortex_action gate OPENED (warmup="
                          f"{visual_cortex_action_warmup_steps})", flush=True)
            except KeyError:
                pass  # Gate not present (no IT -> cortex synapses)

        # Curriculum gate update — for ramp mode, update every step during
        # the ramp window; for abrupt mode, only at the warmup boundary.
        # Sensory and hippo input layers share phase-2 gain (they're peer
        # input pathways being thawed together).
        if enable_curriculum and (has_cortex_gate or has_hippo_gate or has_sensory_gate):
            target_cortex, target_hippo = _curriculum_gate_values(step)
            target_sensory = target_hippo  # input layers transition together
            if curriculum_ramp_steps > 0:
                if has_cortex_gate:
                    bridge.set_plasticity_gate("corticostriatal", float(target_cortex))
                if has_hippo_gate:
                    bridge.set_plasticity_gate("place_goal_to_cortex", float(target_hippo))
                if has_sensory_gate:
                    bridge.set_plasticity_gate("sensory_to_cortex", float(target_sensory))
                if has_beacon_gate:
                    bridge.set_plasticity_gate("beacon_to_goal", float(target_sensory))
                if has_landmark_gate:
                    bridge.set_plasticity_gate("landmark_to_place", float(target_sensory))
                if (last_logged_phase == 1 and target_hippo > 0.0):
                    last_logged_phase = 2
                    if verbose:
                        print(f"[g11 seed={seed}] step {step}: CURRICULUM RAMP "
                              f"BEGINNING (cortex {target_cortex:.2f}, inputs {target_hippo:.2f})",
                              flush=True)
            else:
                if last_logged_phase == 1 and step >= curriculum_warmup_steps:
                    last_logged_phase = 2
                    if has_cortex_gate:
                        bridge.set_plasticity_gate("corticostriatal", float(curriculum_phase2_cortex_gain))
                    if has_hippo_gate:
                        bridge.set_plasticity_gate("place_goal_to_cortex", float(curriculum_phase2_hippo_gain))
                    if has_sensory_gate:
                        bridge.set_plasticity_gate("sensory_to_cortex", float(curriculum_phase2_hippo_gain))
                    if has_beacon_gate:
                        bridge.set_plasticity_gate("beacon_to_goal", float(curriculum_phase2_hippo_gain))
                    if has_landmark_gate:
                        bridge.set_plasticity_gate("landmark_to_place", float(curriculum_phase2_hippo_gain))
                    if verbose:
                        print(f"[g11 seed={seed}] step {step}: CURRICULUM PHASE 2 -- "
                              f"corticostriatal={curriculum_phase2_cortex_gain:.2f}, "
                              f"inputs={curriculum_phase2_hippo_gain:.2f}", flush=True)

        # Phase 3 (Cheat #5 closure, 2026-04-28): thaw bg_cross_projections.
        # Cross-projection cortex_X → str_D1_Y / str_D2_Y pathways stay frozen
        # through phases 1 and 2 (so they don't accumulate phase-0 motor bias),
        # then thaw at bg_cross_thaw_step. By this point the agent has typically
        # experienced both pre- and post-goal-change regimes (default thaw=1200
        # is ~300 steps after the default goal change at 900), so STDP+reward
        # can shape cross-action routing symmetrically rather than locking in
        # phase-0 winners.
        if (
            has_bg_cross_gate and not bg_cross_thawed
            and bg_cross_thaw_step >= 0 and step >= bg_cross_thaw_step
        ):
            bridge.set_plasticity_gate("corticostriatal_cross", float(bg_cross_phase3_gain))
            bg_cross_thawed = True
            if verbose:
                print(f"[g11 seed={seed}] step {step}: CURRICULUM PHASE 3 -- "
                      f"bg_cross_projections gain={bg_cross_phase3_gain:.2f}",
                      flush=True)

        # Sleep-replay phase (Stage 7, 2026-04-27): biological memory consolidation.
        # During sleep, hippo cells fire in random replay patterns (sharp-wave ripples),
        # corticostriatal is thawed (consolidation), hippo_to_cortex is frozen.
        # Hippo's already-learned weights drive cortex via existing connections;
        # STDP between cortex and D1 then consolidates the pattern.
        in_sleep = (sleep_replay_after_step >= 0
                   and step >= sleep_replay_after_step
                   and step < sleep_replay_after_step + sleep_replay_steps)
        if in_sleep:
            # Set gates for consolidation: corticostriatal plastic, hippo_to_cortex frozen
            if has_cortex_gate:
                bridge.set_plasticity_gate("corticostriatal", 1.0)
            if has_hippo_gate:
                bridge.set_plasticity_gate("place_goal_to_cortex", 0.0)
            if has_sensory_gate:
                bridge.set_plasticity_gate("sensory_to_cortex", 0.0)
            # Mark phase entry for verbose output
            if step == sleep_replay_after_step and verbose:
                print(f"[g11 seed={seed}] step {step}: ENTERING SLEEP REPLAY "
                      f"(corticostriatal=1, hippo/sensory frozen, replay rate={sleep_replay_rate_hz:.0f}Hz)",
                      flush=True)

        # Cluster D v2: SWR-gated CA3 plasticity. During sleep, suppress
        # CA3 recurrent STDP except during sharp-wave-ripple bursts. Detect
        # bursts by population firing rate spike (μ + 2σ over ~200ms window).
        # During wake, keep the gate fully open so v1 behavior is preserved.
        # NOTE: the actual CA3 drive injection happens AFTER the global
        # `cp_external_input_current[:] = 0` reset further down (alongside
        # the sleep replay drive). Here we only handle the gate-flipping
        # decision based on last step's firing rate.
        if has_swr_gate:
            # Scheduled SWR window mechanism: every `swr_window_period`-th
            # sleep env step is a ripple window (gate=1.0); all others
            # baseline (gate=0.1). Wake always 1.0. See
            # `_swr_gate_value_scheduled` docstring for biological grounding.
            sleep_step_idx = step - sleep_replay_after_step if in_sleep else 0
            swr_gate = _swr_gate_value_scheduled(in_sleep, sleep_step_idx, period=7)
            bridge.set_plasticity_gate("ca3_swr_burst", swr_gate)
            if in_sleep:
                swr_sleep_steps += 1
                if swr_gate >= 0.99:
                    swr_burst_count += 1
        elif sleep_replay_after_step >= 0 and step == sleep_replay_after_step + sleep_replay_steps and verbose:
            print(f"[g11 seed={seed}] step {step}: EXITING SLEEP REPLAY",
                  flush=True)
            # Restore phase-2 gates
            if has_cortex_gate:
                bridge.set_plasticity_gate("corticostriatal", float(curriculum_phase2_cortex_gain))
            if has_hippo_gate:
                bridge.set_plasticity_gate("place_goal_to_cortex", float(curriculum_phase2_hippo_gain))

        # Interactive runtime control (2026-04-28). Polls a JSON file every
        # trial for paused / goal / inject_reward overrides from the webapp.
        # See webapp/static/world.js for the click-to-control wiring.
        manual_reward_injection = 0.0
        if interactive_control_file:
            try:
                with open(interactive_control_file) as _cf:
                    _ctrl = json.load(_cf)
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                _ctrl = {}
            # Pause loop — block while paused, re-reading the file periodically
            while _ctrl.get("paused"):
                time.sleep(0.1)
                try:
                    with open(interactive_control_file) as _cf:
                        _ctrl = json.load(_cf)
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    break
            # Goal override (persistent until set again)
            _new_goal = _ctrl.get("goal")
            if _new_goal is not None and len(_new_goal) == 2:
                _ng = (int(_new_goal[0]), int(_new_goal[1]))
                if (gx, gy) != _ng:
                    gx, gy = _ng
                    goal_change_steps.append(step)
                    if verbose:
                        print(f"[g11 seed={seed}] step {step}: INTERACTIVE GOAL "
                              f"-> ({gx}, {gy})", flush=True)
            # One-shot reward injection (consumed by clearing the field)
            _inj = _ctrl.get("inject_reward")
            if _inj is not None:
                manual_reward_injection = float(_inj)
                _ctrl["inject_reward"] = None
                try:
                    with open(interactive_control_file, "w") as _cf:
                        json.dump(_ctrl, _cf)
                except OSError:
                    pass

        # Goal change (scheduled)
        while (current_schedule_idx + 1 < len(goal_schedule_sorted)
               and step >= goal_schedule_sorted[current_schedule_idx + 1][0]):
            current_schedule_idx += 1
            gx, gy = goal_schedule_sorted[current_schedule_idx][1]
            goal_change_steps.append(step)
            if verbose:
                print(f"[g11 seed={seed}] step {step}: GOAL CHANGED to ({gx}, {gy})",
                      flush=True)

        # DA-gated WTA: scale FS->motor synapse weights by current gating_strength.
        # When gating=1 (winning, exploit), full WTA. When gating=0 (losing,
        # explore), WTA disabled (no inhibition). Updated AFTER each trial's
        # reward feedback below.
        if fs_to_motor_indices is not None:
            bridge.cp_connections.data[fs_to_motor_indices] = (
                fs_to_motor_baseline_weights * cp.float32(current_gating_strength)
            )

        dist_before = manhattan(x, y)

        # Sensory input encoding: drive cortex pools based on position.
        # SIMPLE HEURISTIC: drive each cortex_X pool with strength inversely
        # proportional to current direction's distance to goal. This is a
        # phenomenological "goal-direction signal" — what the agent's
        # higher cortex would compute given knowledge of the goal.
        # The BG circuit then has to produce a clean motor output.
        # NOTE: this DOESN'T let the BG demonstrate "discovery" — but it
        # does test whether the BG's per-action structure dissolves the
        # silent-motor trap on phase change.
        # RE-SET ALL baseline drives every trial (defensive against any drift).
        # N8 conversion: genuine disinhibition uses a strong GPi pacemaker +
        # tonic-only thal excitation (no direct thal pacing); see the one-time
        # setup block above and the function-signature docstring.
        bridge.cp_external_input_current[:] = 0.0
        for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
        for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = _gpi_tonic
        for rn in ["stn", "snc"]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
        # Spiking-SNc Stage A (2026-06-08): override the snc tonic floor with
        # the calibrated snc_tonic_pa (the generic 150 pA above is for the
        # silent placeholder). This holds the pool at its spontaneous rate
        # OUTSIDE the reward window so there is headroom to DIP on -RPE; the
        # reward window then layers I_reward - I_value on top (see reward block).
        if spiking_snc and "snc" in region_indices_cp:
            bridge.cp_external_input_current[region_indices_cp["snc"]] = (
                cp.float32(snc_tonic_pa)
            )
        for rn in [f"thal_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_indices_cp[rn]] = _thal_tonic
        # N6 commit stage: RE-set the commit_OPN tonic drive every trial (the
        # `cp_external_input_current[:] = 0.0` above wipes the one-time setup, so
        # without this the omnipause pool goes silent after trial 0 and the commit
        # burst pools are no longer held below threshold). Keeps the omnipause
        # tonically firing so commit_X stays gated until a sel_X accumulator wins.
        if _use_commit and "commit_OPN" in region_indices_cp:
            bridge.cp_external_input_current[region_indices_cp["commit_OPN"]] = cp.float32(commit_opn_tonic_pA)
        # N6 accumulate-then-commit: RESET the accumulator each trial. Zero the
        # NMDA-slow conductance (the integrator state, tau=100ms) plus the fast
        # excitatory/inhibitory conductances on the sel_X (+ commit_X) slice so
        # each decision integrates FRESH thalamic evidence rather than carrying
        # the previous trial's winner across the inter-trial (the working-memory
        # hysteresis that mis-commits at goal-change boundaries). The 30ms
        # pre-readout window lets membrane settle to rest; only conductance state
        # is zeroed (no IZH-specific membrane poke). NO sim/ edit.
        if _accum_reset_idx_cp is not None:
            if getattr(bridge, "cp_conductance_g_nmda", None) is not None:
                bridge.cp_conductance_g_nmda[_accum_reset_idx_cp] = 0.0
                bridge.cp_conductance_g_nmda_rise[_accum_reset_idx_cp] = 0.0
            bridge.cp_conductance_g_e[_accum_reset_idx_cp] = 0.0
            bridge.cp_conductance_g_i[_accum_reset_idx_cp] = 0.0
        # N6 refinement 1 (2026-06-06): LOSER-ONLY reset. Zero the NMDA + fast
        # conductances on every sel_X (+commit_X) pool EXCEPT the previous trial's
        # selected action (the carry-winner). Surgical hysteresis removal: the
        # winner's latch persists (fast re-ramp when the goal is stable), but the
        # three losers integrate FRESH evidence each trial. At a goal change the new
        # winner is among the freshly-cleared losers (clean) and the stale old winner
        # — now contradicted by thal — decays naturally over the 100ms window instead
        # of out-competing the new evidence (the all-reset's lost-drive penalty is
        # avoided because the eventual winner is only ever reset while it is a loser).
        # Mutually exclusive with the all-reset above. NO sim/ edit.
        if (reset_losers_only and _accum_reset_idx_per_action_cp
                and not reset_accumulator_each_trial):
            _carry_winner = action_log[-1] if action_log else None
            for _ai, _an in enumerate(ACTION_NAMES):
                if _ai == _carry_winner:
                    continue  # keep the winner's carried drive (the WM latch)
                _ridx = _accum_reset_idx_per_action_cp.get(_an)
                if _ridx is None:
                    continue
                if getattr(bridge, "cp_conductance_g_nmda", None) is not None:
                    bridge.cp_conductance_g_nmda[_ridx] = 0.0
                    bridge.cp_conductance_g_nmda_rise[_ridx] = 0.0
                bridge.cp_conductance_g_e[_ridx] = 0.0
                bridge.cp_conductance_g_i[_ridx] = 0.0
        # Cluster F (cerebellum) baseline drives. Inferior olive baseline
        # gives ~1 Hz spontaneous firing (Hesslow & Yeo 2002 §"Afferent
        # Systems" p 99); CF burst on negative-reward step is set below
        # after reward computation. DCN baseline gives tonic 40 Hz output
        # (so PC silence releases motor drive). Purkinje baseline drives
        # tonic simple-spike firing (~30-80 Hz) per F.01 Cerminara & Rawson.
        if enable_cluster_f_cerebellum:
            bridge.cp_external_input_current[region_indices_cp["inferior_olive"]] = cp.float32(80.0)
            for a in ACTION_NAMES:
                bridge.cp_external_input_current[region_indices_cp[f"dcn_aip_{a}"]] = cp.float32(180.0)
                bridge.cp_external_input_current[region_indices_cp[f"purkinje_{a}"]] = cp.float32(120.0)
        # Cortex drives — both heuristic AND learned perception can be active
        # simultaneously (additive). The heuristic represents innate
        # sensorimotor primitives; the sensory layer learns refined
        # position→action mappings on top. With curriculum, the sensory
        # layer learns via STDP+reward using the heuristic as teacher.
        # Heuristic cortex drive: directly drive cortex_X for each goal-relative direction.
        # Heuristic strength can decay post-curriculum to test pure-learned navigation.
        # During sleep replay: heuristic disabled so consolidation runs purely
        # on hippo-driven cortex activity.
        # During goal_silence (PFC Stage 2): also silence heuristic to test
        # whether PFC + already-learned input layers maintain navigation.
        in_goal_silence_step = (goal_silence_after_step >= 0
                                and step >= goal_silence_after_step
                                and step < goal_silence_after_step + goal_silence_duration)
        if in_sleep or in_goal_silence_step:
            h_strength = 0.0
        elif heuristic_wean_adaptive:
            # ADAPTIVE / activity-gated weaning (N1): the scheduler at the top of
            # the loop drives the phase machine. During the "teaching" phase the
            # heuristic is full strength EXCEPT inside an OFF readiness-probe
            # window (heuristic silenced to measure self-sufficiency). Once the
            # wean has committed, ramp from full strength to 0 over
            # heuristic_wean_steps from adaptive_commit_step, then hold at 0.
            if adaptive_phase == "committed":
                if step >= adaptive_commit_step + heuristic_wean_steps:
                    h_strength = 0.0
                else:
                    _wean_frac = (step - adaptive_commit_step) / float(max(1, heuristic_wean_steps))
                    h_strength = heuristic_strength * (1.0 - _wean_frac)
            elif adaptive_probe_active:
                # OFF probe window: silence the heuristic so we measure whether
                # the learned mapping navigates on its own.
                h_strength = 0.0
            else:
                h_strength = heuristic_strength
        elif heuristic_wean_start >= 0:
            # Critical-period developmental scaffold (N1): base strength during
            # the critical period (step < wean_start), linear ramp to 0 over
            # [wean_start, wean_start + wean_steps], then 0 forever after.
            if step < heuristic_wean_start:
                h_strength = heuristic_strength
            elif step >= heuristic_wean_start + heuristic_wean_steps:
                h_strength = 0.0
            else:
                # Linear ramp factor 1.0 -> 0.0 across the wean window.
                _wean_frac = (step - heuristic_wean_start) / float(max(1, heuristic_wean_steps))
                h_strength = heuristic_strength * (1.0 - _wean_frac)
        elif heuristic_decay_after_step >= 0 and step >= heuristic_decay_after_step:
            h_strength = post_curriculum_heuristic_strength
        elif enable_cue_reflex and cue_reflex_replaces_heuristic:
            # Stage 3: reflex replaces heuristic. The reflex below computes
            # cortex drive from beacon sensor activations instead of (gx,gy).
            h_strength = 0.0
        else:
            h_strength = heuristic_strength
        h_drive = cp.float32(800.0 * h_strength)
        if h_strength > 0:
            if heuristic_single_pool:
                # Replicated-runner-style: drive ONE cortex pool only (chosen
                # randomly among the directions that would shrink Manhattan).
                # 2026-04-30 probe: investigating whether multi-pool heuristic
                # is what makes single runner ~2x worse than replicated.
                cands = []
                if gy > y: cands.append("N")
                if gx > x: cands.append("E")
                if gy < y: cands.append("S")
                if gx < x: cands.append("W")
                if cands:
                    pick = cands[np.random.randint(0, len(cands))]
                    bridge.cp_external_input_current[region_indices_cp[f"cortex_{pick}"]] = h_drive
            else:
                # Original multi-pool: drive every cortex pool whose direction
                # reduces Manhattan distance. For diagonal goals, this drives
                # 2 pools simultaneously, forcing BG arbitration.
                if gy > y:
                    bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = h_drive
                if gx > x:
                    bridge.cp_external_input_current[region_indices_cp["cortex_E"]] = h_drive
                if gy < y:
                    bridge.cp_external_input_current[region_indices_cp["cortex_S"]] = h_drive
                if gx < x:
                    bridge.cp_external_input_current[region_indices_cp["cortex_W"]] = h_drive

        # Cue-following reflex (Item 1 Stage 3, 2026-04-27).
        # Innate reflex: computes cortex drive from beacon sensor activations
        # instead of from raw (gx, gy) coordinates. Each cortex pool gets
        # drive proportional to the integrated beacon strength in its
        # preferred cardinal direction. Models "approach attractive cue"
        # reflex like phototaxis. Non-plastic (innate sensorimotor wiring).
        # Direction-normalized: reflex strength is independent of beacon
        # distance (real biological reflexes operate on direction once
        # stimulus is detected, not on absolute intensity).
        if enable_cue_reflex and enable_beacon_perception and not (in_sleep or in_goal_silence_step):
            bdx = float(gx - x); bdy = float(gy - y)
            distance = (bdx * bdx + bdy * bdy) ** 0.5
            if distance > 1e-6:
                bearing_x = bdx / distance
                bearing_y = bdy / distance
                # Direction-only sensor pattern: cosine alignment, half-rectified
                sensor_dir = np.maximum(0.0, beacon_pref_x * bearing_x + beacon_pref_y * bearing_y)
                # Normalize so total activation sums to 1 (direction representation)
                total = sensor_dir.sum() + 1e-6
                sensor_norm = sensor_dir / total
                # Each cortex pool integrates sensors aligned with its cardinal direction
                drive_N = float(np.sum(sensor_norm * np.maximum(0, beacon_pref_y)))
                drive_E = float(np.sum(sensor_norm * np.maximum(0, beacon_pref_x)))
                drive_S = float(np.sum(sensor_norm * np.maximum(0, -beacon_pref_y)))
                drive_W = float(np.sum(sensor_norm * np.maximum(0, -beacon_pref_x)))
                # Scale to match heuristic strength regardless of distance
                # (the reflex is "go this direction at full strength" once
                # the cue direction is detected, like phototaxis)
                if drive_N > 1e-3:
                    bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = cp.float32(drive_N * cue_reflex_strength)
                if drive_E > 1e-3:
                    bridge.cp_external_input_current[region_indices_cp["cortex_E"]] = cp.float32(drive_E * cue_reflex_strength)
                if drive_S > 1e-3:
                    bridge.cp_external_input_current[region_indices_cp["cortex_S"]] = cp.float32(drive_S * cue_reflex_strength)
                if drive_W > 1e-3:
                    bridge.cp_external_input_current[region_indices_cp["cortex_W"]] = cp.float32(drive_W * cue_reflex_strength)
        # Sensory layer drive (opt-in, additive on top of heuristic).
        # Each sensory neuron i has preferred (dx_i, dy_i); rate = max * exp(-d²/2σ²)
        # The sensory→cortex pathway is plastic — agent learns mapping via STDP+reward.
        if enable_learned_perception and not learned_perception_from_vision:
            dx = float(gx - x)
            dy = float(gy - y)
            dx_clip = max(-3.0, min(3.0, dx))
            dy_clip = max(-3.0, min(3.0, dy))
            d_sq = (sensory_pref_dx - dx_clip) ** 2 + (sensory_pref_dy - dy_clip) ** 2
            sensory_drive = sensory_drive_max_pA * np.exp(-d_sq / (2.0 * sensory_drive_sigma ** 2))
            bridge.cp_external_input_current[region_indices_cp["sensory"]] = cp.asarray(sensory_drive, dtype=cp.float32)

        # Hippocampus drive (ADDITIVE on top of heuristic — provides plastic memory).
        # Real biology: hippocampus augments cortex, doesn't replace it. Place + goal
        # cells learn (place, goal) → action associations via STDP+reward, providing
        # additional cortex drive that should reinforce the correct action over training.
        # Curriculum gate: during the warmup phase, suppress hippo drive so the
        # heuristic (+WTA if enabled) builds up cortex→D1 selectivity in isolation.
        # After the warmup, hippo drive turns on and learns via STDP+reward.
        # SLEEP REPLAY: drive place + goal cells to simulate sharp-wave
        # ripples. The replayed pattern, via existing learned hippo→cortex
        # weights, drives cortex pools, which then strengthens cortex→D1
        # weights via STDP (corticostriatal thawed).
        # Trajectory replay (preferred): sample from successful_trajectories
        # log (built during wake from positive-reward steps). Models
        # biological replay of episodic memories. Falls back to random
        # patterns if no trajectories logged yet.
        # NREM/REM (Item 7): if sleep_nrem_rem_alternate, first half of sleep
        # is NREM-style (trajectory replay, biological consolidation), second
        # half is REM-style (random patterns, less structured).
        if in_sleep and enable_hippocampus:
            sleep_progress = (step - sleep_replay_after_step) / max(1, sleep_replay_steps)
            in_rem_phase = sleep_nrem_rem_alternate and sleep_progress >= 0.5
            if successful_trajectories and not in_rem_phase:
                # NREM: trajectory replay from logged successful steps
                if enable_reverse_replay:
                    # Reverse-order replay (Foster & Wilson 2006, Diba & Buzsaki 2007):
                    # during NREM ripples, real CA1/CA3 replay trajectories in reverse
                    # time order — last-position-before-goal replayed first, working
                    # backward to start. Biologically grounded as TD-style backward
                    # credit assignment: the goal "sends signal back" through the
                    # trajectory. Implementation: walk successful_trajectories from
                    # newest to oldest, indexing by sleep progress.
                    n_traj = len(successful_trajectories)
                    sleep_step_idx = step - sleep_replay_after_step
                    # Map sleep_step_idx to a position in successful_trajectories:
                    # idx 0 -> newest, idx (n_traj-1) -> oldest. Cycle through if
                    # sleep window is longer than the trajectory buffer.
                    traj_idx = (n_traj - 1) - (sleep_step_idx % n_traj)
                    replay_x, replay_y, replay_gx, replay_gy = successful_trajectories[traj_idx]
                elif enable_recency_weighted_replay:
                    # Recency-weighted replay (2026-04-30): bias sampling toward
                    # the newest trajectories with exponential weighting:
                    # P(idx) ∝ exp((idx - 0) / tau). Newest = highest probability.
                    # Tau set so the oldest entry is weighted ~e^(-3) ≈ 5% relative
                    # to the newest. Addresses the SCIENCE_ROADMAP §4.7 note that
                    # "stale trajectory replay doesn't help" — older trajectories
                    # were sampled from goals that no longer apply.
                    n_traj = len(successful_trajectories)
                    tau = max(1.0, n_traj / 3.0)
                    weights = np.exp((np.arange(n_traj) - (n_traj - 1)) / tau)
                    weights /= weights.sum()
                    idx = int(np.random.choice(n_traj, p=weights))
                    replay_x, replay_y, replay_gx, replay_gy = successful_trajectories[idx]
                else:
                    # Forward random sampling (original behavior).
                    idx = int(np.random.randint(0, len(successful_trajectories)))
                    replay_x, replay_y, replay_gx, replay_gy = successful_trajectories[idx]
                replay_x = float(replay_x); replay_y = float(replay_y)
                replay_gx = float(replay_gx); replay_gy = float(replay_gy)
            else:
                # REM (or fallback): random patterns, less structured
                replay_x = float(np.random.randint(0, grid_size))
                replay_y = float(np.random.randint(0, grid_size))
                replay_gx = float(np.random.randint(0, grid_size))
                replay_gy = float(np.random.randint(0, grid_size))
            place_dsq = (hippo_pref_x - replay_x) ** 2 + (hippo_pref_y - replay_y) ** 2
            place_drive = hippocampus_drive_max_pA * np.exp(-place_dsq / (2.0 * hippocampus_drive_sigma ** 2))
            bridge.cp_external_input_current[region_indices_cp["sensor_place_readout"]] = cp.asarray(place_drive, dtype=cp.float32)
            goal_dsq = (hippo_pref_x - replay_gx) ** 2 + (hippo_pref_y - replay_gy) ** 2
            goal_drive = hippocampus_drive_max_pA * np.exp(-goal_dsq / (2.0 * hippocampus_drive_sigma ** 2))
            if hidden_goal:
                goal_drive = goal_drive * 0.0  # hidden-goal: no goal coords into the brain
            bridge.cp_external_input_current[region_indices_cp["ppc_goal_input"]] = cp.asarray(goal_drive, dtype=cp.float32)
            # Cluster D v2: also drive CA3 directly. The existing replay
            # injects into sensor_place_readout / ppc_goal_input but neither
            # has a path to CA3 in v1's wiring, so the autoassociator stays
            # silent during sleep and bursts never fire. Sparse Poisson kick
            # (~5-10% of CA3 active per step at 220 pA) gives the recurrent
            # network an excitation source to amplify; bursts emerge from
            # intrinsic CA3 dynamics on top of this drive.
            # Cluster D v2 baseline drive: keep CA3 at modest depolarization
            # during sleep so the autoassociator has activity to consolidate.
            # Below the rheobase for sustained firing (verified ~220 pA is
            # sub-threshold for IZH2007_HIPPO_PYRAMIDAL in our setup); the
            # actual ripple-window drive is added by the dg→ca3 Schaffer
            # input which fires when the existing replay drive activates EC.
            # No cheats: we don't artificially blow up CA3 to force bursts.
            if has_swr_gate:
                n_ca3 = len(ca3_indices_cp)
                kick_mask = cp.random.random(n_ca3) < 0.05
                ca3_drive = cp.where(kick_mask, 60.0, 0.0).astype(cp.float32)
                bridge.cp_external_input_current[ca3_indices_cp] = ca3_drive
            hippo_active = False  # skip the normal-flow hippo drive below
        else:
            hippo_active = enable_hippocampus and (
                not enable_curriculum or step >= curriculum_warmup_steps
            )
        if hippo_active:
            if enable_landmarks and landmarks_replace_place:
                # Stage 2: don't drive place_cells directly. They get input
                # only via the plastic landmark_sensors → place_cells pathway.
                pass
            else:
                place_dsq = (hippo_pref_x - float(x)) ** 2 + (hippo_pref_y - float(y)) ** 2
                place_drive = hippocampus_drive_max_pA * np.exp(-place_dsq / (2.0 * hippocampus_drive_sigma ** 2))
                bridge.cp_external_input_current[region_indices_cp["sensor_place_readout"]] = cp.asarray(place_drive, dtype=cp.float32)
            # Goal cells silencing test (PFC Stage 2): during the silence
            # window, goal_cells are forced to 0 — tests whether PFC working
            # memory holds the goal info during the delay.
            in_goal_silence = (goal_silence_after_step >= 0
                              and step >= goal_silence_after_step
                              and step < goal_silence_after_step + goal_silence_duration)
            if in_goal_silence:
                bridge.cp_external_input_current[region_indices_cp["ppc_goal_input"]] = cp.float32(0.0)
            elif enable_beacon_perception and beacon_replaces_goal:
                # Replace mode: don't drive goal_cells directly. The
                # beacon → goal_cells pathway must learn to drive them
                # from sensor patterns.
                pass  # goal_cells gets only the plastic beacon→goal drive
            elif hidden_goal:
                # Hidden-goal diagnostic: the goal's coordinates must NOT enter
                # the brain. Zero the goal-cell drive (the place drive above
                # stays — own position is legitimate). The agent must learn the
                # goal location from the scalar reward alone (limbic core).
                bridge.cp_external_input_current[region_indices_cp["ppc_goal_input"]] = cp.float32(0.0)
            else:
                goal_dsq = (hippo_pref_x - float(gx)) ** 2 + (hippo_pref_y - float(gy)) ** 2
                goal_drive = hippocampus_drive_max_pA * np.exp(-goal_dsq / (2.0 * hippocampus_drive_sigma ** 2))
                bridge.cp_external_input_current[region_indices_cp["ppc_goal_input"]] = cp.asarray(goal_drive, dtype=cp.float32)
        elif enable_hippocampus:
            # Curriculum phase 1: keep hippo neurons silent (zero drive) so they
            # don't fire and don't accumulate STDP eligibility. Cortex→D1 trains
            # without hippo noise.
            bridge.cp_external_input_current[region_indices_cp["sensor_place_readout"]] = cp.float32(0.0)
            bridge.cp_external_input_current[region_indices_cp["ppc_goal_input"]] = cp.float32(0.0)

        # === DEDICATED DENSE value-critic afferent drive (2026-06-09 VALIDATED redesign) ===
        # Render the agent's perceived (x,y) into the `vs_place_context` place code EACH nav step
        # (a grid-32 Gaussian over the cells' preferred (x,y), WIDE sigma => 30-80 cells fire/
        # location — the convergent-excitation up-state the SPARSE actor place code cannot
        # deliver). This is legitimate sensory rendering under BRAIN-BASED-ONLY (the same
        # mechanism the actor place code uses); the critic's value computation downstream is all
        # neural. Independent of enable_hippocampus / landmarks_replace_place (the critic afferent
        # is its own region). Zeroed during sleep (no live position to value). Mirrors the de-risk
        # probe's grid_place_code_drive; the critic LEARNS V via the three-factor pipeline.
        if enable_neural_critic and "vs_place_context" in region_indices_cp:
            if in_sleep:
                bridge.cp_external_input_current[region_indices_cp["vs_place_context"]] = cp.float32(0.0)
                if enable_convergent_upstate and "vs_place_drive" in region_indices_cp:
                    bridge.cp_external_input_current[region_indices_cp["vs_place_drive"]] = cp.float32(0.0)
            else:
                vs_dsq = (vs_place_pref_x - float(x)) ** 2 + (vs_place_pref_y - float(y)) ** 2
                vs_drive = vs_place_drive_max_pA * np.exp(-vs_dsq / (2.0 * vs_place_sigma ** 2))
                vs_drive_cp_step = cp.asarray(vs_drive, dtype=cp.float32)
                bridge.cp_external_input_current[region_indices_cp["vs_place_context"]] = vs_drive_cp_step
                # A1 up-state arm: SAME place code into vs_place_drive (the convergent up-state
                # afferent). The dense NON-plastic vs_place_drive->striosome_value pathway fires the
                # critic into the up-state from this drive (2026-06-09 convergent-upstate, opt-in).
                if enable_convergent_upstate and "vs_place_drive" in region_indices_cp:
                    bridge.cp_external_input_current[region_indices_cp["vs_place_drive"]] = vs_drive_cp_step

        # === N9 NEURAL PLACE-CODE afferent drive (2026-06-09 nav deployment) ===
        # In the neural_place_selforg path the host-Gaussian vs_place_context is NOT built; the
        # critic afferent is the self-organized spiking `place` pool, which fires from the EGOCENTRIC
        # `place_sensors` render (the ONLY place input — the host place computation is gone). (x,y)
        # enters the brain ONLY through this legitimate sensory render. Zeroed during sleep.
        if _neural_place_selforg and "place_sensors" in region_indices_cp:
            ps_idx_step = region_indices_cp["place_sensors"]
            if in_sleep:
                bridge.cp_external_input_current[ps_idx_step] = cp.float32(0.0)
            else:
                _ps_act = _n9_render(float(x), float(y))
                bridge.cp_external_input_current[ps_idx_step] = cp.asarray(_ps_act, dtype=cp.float32)

        # Landmark perception drive (Item 1 Stage 2, 2026-04-27).
        # Drives landmark_sensors based on agent's bearing+distance to a
        # FIXED landmark position. Each unique (distance, bearing) gives a
        # unique sensor activation pattern, so place_cells can self-organize
        # to fire at specific positions via the plastic landmark→place pathway.
        if enable_landmarks:
            in_goal_silence_step_lm = (goal_silence_after_step >= 0
                                       and step >= goal_silence_after_step
                                       and step < goal_silence_after_step + goal_silence_duration)
            if in_sleep or in_goal_silence_step_lm:
                bridge.cp_external_input_current[region_indices_cp["landmark_sensors"]] = cp.float32(0.0)
            else:
                lx, ly = landmark_position
                ldx = float(lx - x); ldy = float(ly - y)
                ldist = (ldx * ldx + ldy * ldy) ** 0.5
                if ldist < 1e-6:
                    sensor_act = np.full(n_landmark_sensors, landmark_max_intensity, dtype=np.float32)
                else:
                    bearing_x = ldx / ldist
                    bearing_y = ldy / ldist
                    intensity = landmark_max_intensity / (1.0 + landmark_falloff * ldist)
                    cos_alignment = landmark_pref_x * bearing_x + landmark_pref_y * bearing_y
                    sensor_act = intensity * np.maximum(0.0, cos_alignment)
                bridge.cp_external_input_current[region_indices_cp["landmark_sensors"]] = (
                    cp.asarray(sensor_act, dtype=cp.float32)
                )

        # Beacon perception drive (Item 1 Stage 1, 2026-04-27).
        # The beacon emits intensity that falls off with distance from goal.
        # Each sensor has a preferred direction; activation is intensity ×
        # max(0, cosine_alignment) — modeling biological directional cue
        # detection (e.g., bilateral hearing inferring sound source direction).
        # During goal silence (PFC Stage 2 test) and sleep, beacon is also
        # silenced — these tests assume no external goal info available.
        if enable_beacon_perception:
            in_goal_silence_step = (goal_silence_after_step >= 0
                                    and step >= goal_silence_after_step
                                    and step < goal_silence_after_step + goal_silence_duration)
            if in_sleep or in_goal_silence_step:
                bridge.cp_external_input_current[region_indices_cp["beacon_sensors"]] = cp.float32(0.0)
            else:
                # Compute beacon-to-agent vector
                bdx = float(gx - x)
                bdy = float(gy - y)
                distance = (bdx * bdx + bdy * bdy) ** 0.5
                if distance < 1e-6:
                    # On top of beacon: all sensors max
                    sensor_act = np.full(n_beacon_sensors,
                                         beacon_max_intensity,
                                         dtype=np.float32)
                else:
                    bearing_x = bdx / distance
                    bearing_y = bdy / distance
                    intensity = beacon_max_intensity / (1.0 + beacon_falloff * distance)
                    cos_alignment = beacon_pref_x * bearing_x + beacon_pref_y * bearing_y
                    sensor_act = intensity * np.maximum(0.0, cos_alignment)
                bridge.cp_external_input_current[region_indices_cp["beacon_sensors"]] = (
                    cp.asarray(sensor_act, dtype=cp.float32)
                )

        # Cluster K v1 retina drive (2026-05-01).
        # Render the gridworld as a 32x32 ON/OFF image and inject as input
        # current to the retina region. This activates the V1 → V2 → IT
        # ventral stream alongside other perception. v1 doesn't yet wire
        # IT → cortex_X — the visual cortex runs but doesn't influence
        # action selection. Future v2: gated IT → cortex_X with curriculum.
        if enable_visual_cortex:
            from sim.visual_cortex import (
                render_gridworld_to_image,
                image_to_retina_drive,
            )
            in_goal_silence_step_vc = (
                goal_silence_after_step >= 0
                and step >= goal_silence_after_step
                and step < goal_silence_after_step + goal_silence_duration
            )
            if in_sleep or in_goal_silence_step_vc:
                # Sleep / goal-silence: blank retina (no visual input)
                bridge.cp_external_input_current[region_indices_cp["retina"]] = cp.float32(0.0)
            else:
                img = render_gridworld_to_image(
                    agent_pos=(int(x), int(y)),
                    goal_pos=(int(gx), int(gy)),
                    grid_size=int(grid_size),
                    image_size=int(visual_image_size),
                )
                drive = image_to_retina_drive(img, drive_max_pA=float(visual_drive_max_pA))
                bridge.cp_external_input_current[region_indices_cp["retina"]] = (
                    cp.asarray(drive, dtype=cp.float32)
                )
                # Spiking SC (N1, 2026-06-10): drive the SC's OWN egocentric eye STRONG so
                # it forms a robust orienting bump; the framework-wired sc_map->cortex_X
                # pooling then biases action selection SYNAPTICALLY (the spiking replacement
                # for the host sc_orienting_cardinal_from_image current injection). The main
                # `retina` stays allocentric for the visual cortex / N5 reward. Drive strength
                # 2500 pA matches the de-risk operating point (tunable for the 6-seed A/B).
                if enable_spiking_sc and "sc_retina" in region_indices_cp:
                    _ego = render_egocentric_goal((int(x), int(y)), (int(gx), int(gy)),
                                                  image_size=int(visual_image_size))
                    # TRUE-ONE-BRAIN #2 het-off op-point: 2500 pA STARVES the SC bump on the het-off
                    # merged bridge; the de-risk used 3500. SC_RET_DRIVE overrides; default unset =>
                    # 2500 (byte-identical to the standalone nav).
                    _ret_drive = float(os.environ.get("SC_RET_DRIVE", "2500.0"))
                    _egd = image_to_retina_drive(_ego, drive_max_pA=_ret_drive)
                    bridge.cp_external_input_current[region_indices_cp["sc_retina"]] = (
                        cp.asarray(_egd, dtype=cp.float32))
                # Innate superior-colliculus orienting reflex (N1 de-risk,
                # 2026-06-07): read the goal's retinal direction from THIS
                # rendered image alone (no coords) and inject an orienting push
                # into the matching cortex pool — the biological replacement for
                # the coordinate heuristic-teacher. Naturally gated to awake,
                # non-goal-silence steps (this branch). Anti-cheat:
                # sc_orienting_cardinal_from_image sees only `img` (pixels).
                if sc_orienting_reflex:
                    # Rank 2 wean: the innate reflex TEACHES, then fades to zero
                    # over [wean_start, wean_start+wean_steps] as the learned
                    # circuit matures (developmental scaffold; -1 = never wean).
                    # Rank 2 tuning: with a supervised motor-teacher
                    # (sensory_cortex_teacher_pA > 0, feedback-error-learning),
                    # drive the chosen target pool at the STRONG teacher strength
                    # (a clean supervised label) instead of the reflex strength.
                    _base_strength = (sensory_cortex_teacher_pA
                                      if sensory_cortex_teacher_pA > 0
                                      else sc_reflex_strength)
                    _sc_eff = _base_strength
                    if sc_reflex_wean_start >= 0:
                        if step >= sc_reflex_wean_start + sc_reflex_wean_steps:
                            _sc_eff = 0.0
                        elif step >= sc_reflex_wean_start:
                            _wf = (step - sc_reflex_wean_start) / float(max(1, sc_reflex_wean_steps))
                            _sc_eff = _base_strength * (1.0 - _wf)
                    if _sc_eff > 0:
                        _sc_card = sc_orienting_cardinal_from_image(img)
                        if _sc_card is not None:
                            bridge.cp_external_input_current[
                                region_indices_cp[f"cortex_{_sc_card}"]
                            ] = cp.float32(_sc_eff)
                # Rank 2 (2026-06-07): drive the LEARNED sensory population from
                # the IMAGE salience offset (position-PRESERVING "where" signal,
                # NO coords), so the plastic sensory→cortex_X learns a durable
                # where→action mapping (the thing the position-invariant
                # IT→cortex_X could not). The coord-sourced sensory drive is gated
                # OFF when this flag is set. Anti-cheat: sc_salience_offset_from_image
                # sees only `img`.
                if (enable_learned_perception and learned_perception_from_vision
                        and sensory_pref_dx is not None):
                    _off = sc_salience_offset_from_image(
                        img, grid_size=int(grid_size), image_size=int(visual_image_size))
                    if _off is not None:
                        _vdx = max(-3.0, min(3.0, _off[0]))
                        _vdy = max(-3.0, min(3.0, _off[1]))
                        _vd_sq = (sensory_pref_dx - _vdx) ** 2 + (sensory_pref_dy - _vdy) ** 2
                        _vsens = sensory_drive_max_pA * np.exp(-_vd_sq / (2.0 * sensory_drive_sigma ** 2))
                        bridge.cp_external_input_current[region_indices_cp["sensory"]] = (
                            cp.asarray(_vsens, dtype=cp.float32))

        # Tier 2.2 (2026-05-06): embodied-language during nav. Drive
        # language regions simultaneously with the agent's perception/
        # action stream so STDP at lang↔motor and lang↔IT pathways
        # binds words to embodied concepts via Pulvermüller-style
        # somatotopic Hebbian co-firing.
        # Drive sporadically (every Nth step) and at moderate amplitude
        # (80pA vs nav's 100-200pA from retina + BG) so language drive
        # supplements rather than dominates nav.
        if (embodied_language and enable_text_io and not in_sleep
                and step >= int(embodied_language_warmup_steps)
                and step % max(1, int(embodied_language_every_n_steps)) == 0):
            from sim.text_embeddings import vocab_to_drive_pattern
            lang_in_indices_cp = region_indices_cp.get("language_input")
            lang_out_indices_cp = region_indices_cp.get("language_output")
            if lang_in_indices_cp is not None and lang_out_indices_cp is not None:
                # Action labeling (A→W direction): use the previously-
                # executed action as the teacher signal. At step t we
                # know action_(t-1) — agent just moved that direction.
                # During step t's forward-prop, motor pool of action_(t-1)
                # has lingering activity from the previous step (NMDA tau)
                # plus current spontaneous baseline. Drive language with
                # the corresponding word.
                if step > 0 and len(action_log) > 0:
                    prev_action_idx = action_log[-1]
                    if 0 <= prev_action_idx < 4:
                        action_letter = "NESW"[prev_action_idx]
                        word = {"N": "north", "E": "east",
                                "S": "south", "W": "west"}[action_letter]
                        n_lang_in = int(lang_in_indices_cp.size)
                        n_lang_out = int(lang_out_indices_cp.size)
                        in_drive = vocab_to_drive_pattern(
                            word, n_neurons=n_lang_in,
                            drive_max_pA=float(embodied_language_drive_pA),
                            sparsity=0.1,
                        )
                        out_drive = vocab_to_drive_pattern(
                            word, n_neurons=n_lang_out,
                            drive_max_pA=float(embodied_language_drive_pA),
                            sparsity=0.1,
                        )
                        bridge.cp_external_input_current[lang_in_indices_cp] = (
                            cp.asarray(in_drive, dtype=cp.float32)
                        )
                        bridge.cp_external_input_current[lang_out_indices_cp] = (
                            cp.asarray(out_drive, dtype=cp.float32)
                        )

                # Goal perception (W→I direction): if agent is within
                # goal radius, drive language_input["goal"] +
                # language_output["goal"]. Same paradigm: co-active
                # language + IT (which is firing on visual goal cue) →
                # STDP binds word to visual concept.
                dist_to_goal = abs(int(x) - int(gx)) + abs(int(y) - int(gy))
                if dist_to_goal <= int(embodied_language_goal_radius):
                    n_lang_in = int(lang_in_indices_cp.size)
                    n_lang_out = int(lang_out_indices_cp.size)
                    g_in = vocab_to_drive_pattern(
                        "goal", n_neurons=n_lang_in,
                        drive_max_pA=float(embodied_language_drive_pA),
                        sparsity=0.1,
                    )
                    g_out = vocab_to_drive_pattern(
                        "goal", n_neurons=n_lang_out,
                        drive_max_pA=float(embodied_language_drive_pA),
                        sparsity=0.1,
                    )
                    # Add to existing language drive (if action drive
                    # already set above, this combines via max)
                    cur_in = bridge.cp_external_input_current[lang_in_indices_cp]
                    cur_out = bridge.cp_external_input_current[lang_out_indices_cp]
                    bridge.cp_external_input_current[lang_in_indices_cp] = (
                        cp.maximum(cur_in, cp.asarray(g_in, dtype=cp.float32))
                    )
                    bridge.cp_external_input_current[lang_out_indices_cp] = (
                        cp.maximum(cur_out, cp.asarray(g_out, dtype=cp.float32))
                    )

        # Run stimulus window and tally motor (and, for readout_source="thal",
        # thalamus; "spiking_wta", the sel_X selection layer) spike counts over
        # the readout window.
        motor_counts = {a: 0 for a in ACTION_NAMES}
        thal_counts = {a: 0 for a in ACTION_NAMES}
        sel_counts = {a: 0 for a in ACTION_NAMES}
        commit_counts = {a: 0 for a in ACTION_NAMES}
        _read_thal = (_readout_source == "thal")
        _read_sel = (_readout_source == "spiking_wta")
        # When the spiking-WTA accumulator reads sel/commit, ALSO tally the
        # upstream thal_X it is fed by (the cleanly-selective input) so the guard
        # can confirm the accumulator's winner matches the thalamic winner it
        # integrates. Cheap (4 sums/substep) and only over the readout window.
        _read_thal_guard = _read_sel and len(thal_idx_host) == N_ACTIONS
        bridge.core_config.current_reward_signal = 0.0
        _capture = step in _GUARD_SAMPLE_TRIALS
        if _capture:
            _trace_sel = []     # per-substep [per-action sel spike count]
            _trace_commit = []  # per-substep [per-action commit spike count]
        for s in range(n_stim_steps):
            # N6 refinement 2: CISEK URGENCY ramp. Inject a growing
            # action-independent baseline into ALL sel_X over the readout window so
            # the effective commit bound collapses with elapsed time. Linear ramp
            # from 0 at readout_start to urgency_max_pA at readout_end. Same for
            # every pool → no action bias; only the time-to-cross shrinks, so a weak
            # late-phase winner still bursts within the window (kills silent-commit).
            if (urgency_max_pA > 0.0 and _sel_all_idx_cp is not None
                    and readout_start <= s < readout_end):
                _u_frac = (s - readout_start) / max(1, (readout_end - readout_start))
                bridge.cp_external_input_current[_sel_all_idx_cp] = cp.float32(
                    urgency_max_pA * _u_frac)
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * cfg.dt_ms
            )
            # Capture the FULL-window per-substep sel/commit traces for the guard
            # (sample trials only) so the accumulation ramp + commit burst onset
            # are visible, not just the windowed totals.
            if _capture and _read_sel:
                firing_g = bridge.cp_firing_states.get().astype(bool)
                _trace_sel.append([int(firing_g[sel_idx_host[a]].sum()) for a in ACTION_NAMES])
                if _use_commit:
                    _trace_commit.append([int(firing_g[commit_idx_host[a]].sum()) for a in ACTION_NAMES])
            if readout_start <= s < readout_end:
                if not (_capture and _read_sel):
                    firing = bridge.cp_firing_states.get().astype(bool)
                else:
                    firing = firing_g
                for a in ACTION_NAMES:
                    motor_counts[a] += int(firing[motor_idx_host[a]].sum())
                    if _read_thal or _read_thal_guard:
                        thal_counts[a] += int(firing[thal_idx_host[a]].sum())
                    if _read_sel:
                        sel_counts[a] += int(firing[sel_idx_host[a]].sum())
                        if _use_commit:
                            commit_counts[a] += int(firing[commit_idx_host[a]].sum())

        motor_counts_log.append([motor_counts[a] for a in ACTION_NAMES])
        if _read_thal or _read_thal_guard:
            thal_counts_log.append([thal_counts[a] for a in ACTION_NAMES])
        if _read_sel:
            sel_counts_log.append([sel_counts[a] for a in ACTION_NAMES])
            if _use_commit:
                commit_counts_log.append([commit_counts[a] for a in ACTION_NAMES])
        if _capture and _read_sel:
            accum_trace_log[step] = {
                "sel": _trace_sel,
                "commit": _trace_commit if _use_commit else [],
            }

        # Action selection (N6 readout). Default reads the motor pool (legacy,
        # the host-argmax cheat). readout_source="thal" reads the cleanly-
        # selective thalamus (genuine GPi->thal disinhibition releases only the
        # selected action's thal; the thal->motor amplification is too weak for
        # a reliable motor-count argmax over a noisy run). readout_source=
        # "spiking_wta" reads the ACCUMULATE-THEN-COMMIT layer: each sel_X is a
        # Wang-2002 NMDA-recurrent accumulator that ramps the weak clean thalamus
        # to a bound; the downstream commit_X burst pool fires ALL-OR-NONE only
        # when its sel_X crosses threshold (gated by the sel->commit drive +
        # commit's intrinsic threshold; the commit_OPN omnipause gate is OFF by
        # default — a constant drive rebound-bursts on this rate substrate; see
        # commit_opn_tonic_pA docstring). Lo-Wang 2006 SC. The DECISION is the
        # commit_X threshold CROSSING — read which commit_X burst (the spiking
        # termination event), NOT an argmax of graded sel rates. When commit is
        # disabled (a #2 Rutishauser soft-WTA ablation), fall back to the sel_X argmax.
        # The host argmax below merely OBSERVES which commit pool bursted; under a
        # decisive commit the loser counts are ~0, so it is a tie-break of last
        # resort, not the selection mechanism.
        #
        # Fallback chain for accumulate-then-commit (_use_commit): the all-or-none
        # commit_X burst is the PRIMARY decision (the threshold crossing). On a
        # SUB-THRESHOLD trial (no commit crosses — e.g. a brief/weak release where
        # the winner's sel ramp didn't reach the burst bound), the decision falls
        # back to the sel_X ACCUMULATOR's leading pool: the accumulator still
        # carries a graded "lean" toward the winning action (it is selective:
        # winner ~12 vs runner-up ~0.1), so reading its argmax is the biological
        # provisional commitment (Shadlen affordance / Stine 2023: the accumulator
        # keeps a candidate even when the SC burst hasn't fired) — vastly better
        # than a random guess. Random is the last resort only when BOTH the burst
        # AND the accumulator are fully silent (a genuinely undriven trial).
        if _use_commit:
            _primary = commit_counts
            _fallback = sel_counts
        elif _read_sel:
            _primary = sel_counts
            _fallback = None
        elif _read_thal:
            _primary = thal_counts
            _fallback = None
        else:
            _primary = motor_counts
            _fallback = None
        if max(_primary.values()) > 0:
            action_idx = max(range(N_ACTIONS), key=lambda i: _primary[ACTION_NAMES[i]])
            _decision_path = "primary"  # commit burst fired (the spiking decision)
        elif _fallback is not None and max(_fallback.values()) > 0:
            action_idx = max(range(N_ACTIONS), key=lambda i: _fallback[ACTION_NAMES[i]])
            _decision_path = "fallback"  # silent commit → sel-lean (argmax residual)
        else:
            action_idx = int(np.random.default_rng(seed * 10000 + step).integers(0, N_ACTIONS))
            _decision_path = "random"  # both silent
        # N6 guard: track which arm of the fallback chain made each decision so the
        # commit-fire-rate / silent-commit-fallback-rate can be reported (a GO needs
        # the commit firing reliably, not quietly leaning on the argmax fallback).
        if _use_commit:
            _decision_path_counts[_decision_path] += 1
        action_log.append(action_idx)
        # Cluster C v2 (2026-04-29): expose selected action so per-action DA
        # production rules can fire only for the matching channel.
        bridge.core_config.last_selected_action = int(action_idx)

        dx, dy = ACTION_DELTAS[action_idx]
        # During sleep, agent does not move (consolidation phase, no behavior)
        if in_sleep:
            new_x, new_y = x, y
        else:
            new_x = int(np.clip(x + dx, 0, grid_size - 1))
            new_y = int(np.clip(y + dy, 0, grid_size - 1))
        dist_after = manhattan(new_x, new_y)
        x, y = new_x, new_y
        trajectory.append((x, y))
        goal_log.append((gx, gy))
        distance_log.append(dist_after)

        # Reward computation. Default uses Manhattan distance change (cheat:
        # uses raw (gx, gy)). Sensed reward instead uses beacon-intensity
        # gradient (the agent "feels warmer" as it approaches), which operates
        # on the perceptual signal — biologically grounded.
        if perceived_approach_reward:
            # N5 (2026-06-08): coordinate-FREE perceived-approach reward. The agent is
            # reinforced for perceiving the goal get CLOSER in its visual field — reward
            # = sign of the decrease in the goal's retinal ECCENTRICITY (the image-sourced
            # offset magnitude). Appetitive/incentive-salience approach reward (Schultz
            # reward-fn-2; Berridge wanting; phototaxis-like). ANTI-CHEAT: the reward LOGIC
            # reads only sc_salience_offset_from_image (pixels; coords never enter it); the
            # render of the agent's visual input uses goal_pos = N2 (the world's visible
            # goal, a defensible perception, NOT a coordinate fed to the reward).
            from sim.visual_cortex import render_gridworld_to_image as _rgi
            _old = trajectory[-2] if len(trajectory) >= 2 else (int(x), int(y))
            _img_b = _rgi(agent_pos=(int(_old[0]), int(_old[1])), goal_pos=(int(gx), int(gy)),
                          grid_size=int(grid_size), image_size=int(visual_image_size))
            _img_a = _rgi(agent_pos=(int(x), int(y)), goal_pos=(int(gx), int(gy)),
                          grid_size=int(grid_size), image_size=int(visual_image_size))
            _ob = sc_salience_offset_from_image(_img_b, grid_size=int(grid_size), image_size=int(visual_image_size))
            _oa = sc_salience_offset_from_image(_img_a, grid_size=int(grid_size), image_size=int(visual_image_size))
            _eb = (_ob[0] ** 2 + _ob[1] ** 2) ** 0.5 if _ob is not None else 0.0
            _ea = (_oa[0] ** 2 + _oa[1] ** 2) ** 0.5 if _oa is not None else 0.0  # None = on goal = ecc 0
            if _ea < _eb - 1e-6:
                reward = 1.0
            elif _ea > _eb + 1e-6:
                reward = -1.0
            else:
                reward = 0.0
        elif enable_sensed_reward and enable_beacon_perception:
            # Compute beacon intensity at old vs new position
            d_before = float(((gx - (x - dx)) ** 2 + (gy - (y - dy)) ** 2) ** 0.5) if not in_sleep else 0.0
            d_after = float(((gx - x) ** 2 + (gy - y) ** 2) ** 0.5)
            intensity_before = beacon_max_intensity / (1.0 + beacon_falloff * d_before)
            intensity_after = beacon_max_intensity / (1.0 + beacon_falloff * d_after)
            intensity_diff = intensity_after - intensity_before
            # Threshold to avoid noise; sign-only output
            if intensity_diff > 1e-3:
                reward = 1.0
            elif intensity_diff < -1e-3:
                reward = -1.0
            else:
                reward = 0.0
        else:
            if dist_after < dist_before:
                reward = 1.0
            elif dist_after > dist_before:
                reward = -1.0
            else:
                reward = 0.0
        # Interactive reward injection (2026-04-28): additive on top of
        # the natural reward. Lets the user "click +reward" from the webapp
        # to test conditioning / exploration in real time.
        if manual_reward_injection != 0.0:
            reward = float(reward) + manual_reward_injection
            if verbose:
                print(f"[g11 seed={seed}] step {step}: INTERACTIVE REWARD "
                      f"injection {manual_reward_injection:+.2f} -> reward={reward:+.2f}",
                      flush=True)
        # Homeostatic agent hook (2026-06-17). Default None = byte-identical:
        # `None is not None` is False, so this is a no-op for every existing
        # caller. When set, the hook gates the reward by a self-generated drive
        # (reward *= hunger) and may relocate the goal (food) on an eat event.
        # Reusing the validated learner for a homeostatic agent, no fork.
        if homeostatic_hook is not None:
            reward, _homeo_new_goal = homeostatic_hook(
                float(reward), int(x), int(y), int(gx), int(gy), int(step), int(dist_after))
            reward = float(reward)
            if _homeo_new_goal is not None:
                gx, gy = int(_homeo_new_goal[0]), int(_homeo_new_goal[1])
                goal_change_steps.append(step)
        # Reward lesion (load-bearing anti-cheat, 2026-06-19): unconditional clamp
        # to 0 so no learning signal reaches dopamine / value critic / corticostriatal
        # plasticity. Must collapse the BEHAVIOR if the reward is load-bearing.
        if lesion_reward:
            reward = 0.0
        reward_log.append(float(reward))

        # Cluster F v1: climbing-fiber teaching signal. When the just-completed
        # action increased Manhattan distance (reward < 0), bump inferior_olive
        # drive to evoke a CF burst that propagates to PCs as complex spikes.
        # The next bridge.step() will see this elevated drive; combined with
        # recent PF activity in the eligibility trace and the active negative
        # reward, this yields LTD-like weight changes on the active PF→PC
        # synapses. v2 will add proper CF-gated LTD with explicit anti-
        # Hebbian rule rather than relying on the existing reward-modulation
        # path. See docs/plans/2026-04-29-cluster-f-cerebellum-design.md.
        if enable_cluster_f_cerebellum and reward < 0:
            bridge.cp_external_input_current[region_indices_cp["inferior_olive"]] = cp.float32(450.0)

        # Log successful (place, goal) tuples during wake for sleep-replay.
        # When reward > 0 (agent moved toward goal), the (place_before, goal)
        # pairing is biologically meaningful and should be replayed during
        # sleep for memory consolidation. Only logged during wake (not sleep).
        if reward > 0 and not in_sleep:
            successful_trajectories.append((x, y, gx, gy))
            if len(successful_trajectories) > SUCCESSFUL_TRAJ_MAX:
                # Drop oldest to keep memory bounded
                successful_trajectories.pop(0)

        # HER (Hindsight Experience Replay, Andrychowicz 2017): also log
        # (place, position-N-steps-later) tuples as if the achieved later
        # position were the goal. Provides hindsight credit assignment:
        # "this trajectory leading to position X would have been optimal
        # IF X had been the goal." Generalizes spatial knowledge across
        # goals; biological correlate is mental simulation/imagination.
        # Default off — backward compatible.
        if enable_her and not in_sleep:
            # Append a hindsight tuple where the goal is the agent's position
            # k steps in the future (after the trajectory has actually visited
            # that position). To do this we lag-buffer the wake trajectory and
            # append (x_old, y_old, x_now, y_now) when the lag fires.
            her_lag_buffer.append((x, y))
            if len(her_lag_buffer) > her_lag_steps:
                old_x, old_y = her_lag_buffer.pop(0)
                # Skip degenerate cases where position didn't change
                if (old_x, old_y) != (x, y):
                    successful_trajectories.append((old_x, old_y, x, y))
                    if len(successful_trajectories) > SUCCESSFUL_TRAJ_MAX:
                        successful_trajectories.pop(0)

        if abs(reward) > 0:
            # Capture EMA BEFORE update (= the agent's prediction at this step)
            reward_ema_pre = reward_ema
            # Update reward EMA (used by adaptive DA mode).
            # If asymmetric decay is configured, use faster decay for negative
            # reward (quicker exploration trigger on goal change / policy break).
            # Models phasic DA biology: dips on negative RPE faster than ramps
            # up on positive (Schultz 1998).
            if reward < 0 and adaptive_da_ema_decay_negative is not None:
                _decay = adaptive_da_ema_decay_negative
            else:
                _decay = adaptive_da_ema_decay
            reward_ema = _decay * reward_ema + (1 - _decay) * float(reward)

            # Compute gating strength for per-action DA targeting:
            #   hard:     always 1.0 (full gating)
            #   adaptive: scales linearly from reward_ema in [-1, +1] to strength in [0, 1]
            #             reward_ema=+1 (consistently winning) → strength=1.0 (full gating, exploit)
            #             reward_ema=-1 (consistently losing)  → strength=0.0 (no gating, explore)
            if enable_adaptive_per_action_da:
                gating_strength = max(0.0, min(1.0, (reward_ema + 1.0) / 2.0))
            elif enable_per_action_da_targeting:
                gating_strength = 1.0
            else:
                gating_strength = 0.0
            da_strength_log.append(float(gating_strength))
            # Cache for next trial's WTA scaling
            current_gating_strength = float(gating_strength)

            # Apply per-action DA: scale eligibility on non-selected pathways by (1 - strength)
            if (gating_strength > 0
                    and d1_synapse_other_action_masks is not None
                    and bridge.cp_eligibility_trace is not None):
                actual_nnz = bridge.cp_connections.nnz
                other_mask = d1_synapse_other_action_masks[action_idx][:actual_nnz]
                scale = float(1.0 - gating_strength)
                # Scale eligibility on non-selected pathways
                trace = bridge.cp_eligibility_trace[:actual_nnz]
                trace[other_mask] = trace[other_mask] * scale

            # RPE-scaled reward (opt-in): amplify surprise (= deviation from expectation)
            # Uses reward_ema_pre (the agent's prediction BEFORE this trial's reward).
            rpe = float(reward) - reward_ema_pre
            if spiking_snc:
                # Spiking-SNc Stage A (2026-06-08): the RPE is NOT a host scalar.
                # It is computed by the FIRING of the snc pool, which is driven
                # below (I_snc current) and read out as the `dopamine`
                # concentration via from_region_firing_signed. So
                # current_reward_signal stays 0 — nothing downstream reads a
                # stale host RPE (the legacy scalar path is bypassed, not used).
                # --spiking-snc SUPERSEDES --rpe-dopamine (mutually exclusive
                # semantics; both would be "the RPE", and spiking wins).
                delivered_reward = 0.0
            elif rpe_dopamine:
                # N9 step 1: the DA signal IS the reward-prediction-error delta = r - V
                # (V = reward_ema_pre, the learned Rescorla-Wagner critic) — an actor-CRITIC
                # RPE, not raw reward (Schultz 1998; catalog C.22/C.28/C.30). With N5
                # (--perceived-approach-reward) r is coord-free, so the whole RPE loop is.
                # Reuses the already-present reward_ema critic; NO sim/ edit.
                delivered_reward = rpe
            elif enable_rpe_scaled_reward:
                delivered_reward = float(reward) + rpe_scale_alpha * rpe
            else:
                delivered_reward = float(reward)
            bridge.core_config.current_reward_signal = delivered_reward

            # Cluster F v2 (2026-04-30): CF-gated LTD per Albus 1971 §IV.C eq.4.
            # Decouples cerebellum_pf_pc plasticity from the global reward signal.
            # PF→PC synapses see -1.0 only when IO is active (CF event = reward<0
            # in our task model), 0.0 otherwise. Non-cerebellum synapses see the
            # delivered_reward as before. The bridge's reward modulation step
            # uses cp_per_synapse_reward_override when set, replacing the scalar.
            if cerebellum_pf_pc_mask is not None:
                actual_nnz = bridge.cp_connections.nnz
                # Per-synapse override array: default = global reward
                override = cp.full(actual_nnz, delivered_reward, dtype=cp.float32)
                # Cerebellum synapses get CF-gated signal
                cf_signal = -1.0 if delivered_reward < 0 else 0.0
                override[cerebellum_pf_pc_mask[:actual_nnz]] = cf_signal
                bridge.cp_per_synapse_reward_override = override
            elif bridge.cp_per_synapse_reward_override is not None:
                # Defensive: clear stale override if v2 wasn't actually wired up
                bridge.cp_per_synapse_reward_override = None

            # Spiking-SNc Stage A (2026-06-08): drive the snc pool's external
            # current so its windowed firing rate encodes delta = r - V.
            #   I_snc = I_tonic + k_r * max(0, r) - k_v * V
            # The pool integrates these opposing currents and FIRES the RPE:
            # reward above value -> burst (rate > tonic); reward below value ->
            # DIP (rate < tonic). V is the host reward_ema_pre scaffold for
            # Stage A (the ONLY host use of the value; Stage B replaces it with
            # a neural striosome critic). max(0, ·) makes the inhibitory drive
            # the EXPECTED appetitive value (a negative expectation must not
            # flip the inhibition into excitation). The snc tonic at :4042
            # (=150 pA generic) is OVERRIDDEN here with the calibrated
            # snc_tonic_pa; since cp_external_input_current is NOT reset inside
            # the reward-hold loop, this write persists across all hold steps.
            # NO sim/ edit — pure cp_external_input_current write, the same
            # mechanism every other region uses.
            #
            # Stage B (2026-06-08, --enable-neural-critic): the host _V_scaffold
            # subtraction is DROPPED. The value V is now SUBTRACTED at the SNc
            # membrane by the NEURAL critic's GABA_B/GIRK inhibition (the
            # striosome_value -> snc pathway, receptor="gaba_b", driven by the
            # perceived state through a plastic, dopamine-delta-trained afferent).
            # I_snc carries only tonic + reward; the brain does the r - V
            # subtraction (BRAIN-BASED-ONLY). The striosome critic LEARNS V via
            # the existing three-factor pipeline (the same SNc-derived da_signal
            # the actor uses), so as the cue->value weight grows, the GABA_B
            # current cancels the reward drive and the SNc burst shrinks toward
            # the predicted level — all neural, no host value read.
            # Windowed mode: FORCE the critic->SNc GABA_B gate OPEN through the
            # reward-hold loop (regardless of the sawtooth phase above) so the
            # value subtraction is live exactly at reward delivery — the de-risk's
            # value-leads-reward window covers the run-up (the just-elapsed nav
            # integration) PLUS the reward hold. (No-op in Stage 1: already open.)
            if _critic_gate_known and enable_critic_window:
                bridge.set_transmission_gate("critic_snc_window", 1.0)
            if spiking_snc and "snc" in region_indices_cp:
                if enable_neural_critic:
                    if spiking_reward_us and "reward_us" in region_indices_cp:
                        # The reward burst `r` is produced by reward_us FIRING into the SNc (the
                        # spiking US->VTA glutamatergic afferent), NOT a host current write. Drive
                        # reward_us with the PERCEIVED reward (coord-free with N5) so it fires through
                        # the reward-hold -> the SNc bursts SYNAPTICALLY; the striosome GABA_B then
                        # subtracts V at the membrane. The whole δ=r−V is now neural (r = reward_us
                        # excitation, V = critic GABA_B). _I_snc carries ONLY tonic.
                        if enable_spiking_sc_approach and "sc_rostral" in region_indices_cp:
                            # N5 (TRUE-ONE-BRAIN #2): reward_us is driven SYNAPTICALLY by the
                            # sc_rostral PROXIMITY pool (the SC bump's goal-salience / how
                            # central+close the goal is), NOT the host sign(delta ecc). Zero the
                            # host write; the sc_rostral -> reward_us pathway (built at :2541-2544)
                            # carries the reward r (the whole r term is now neural: SC retina ->
                            # sc_map -> sc_rostral -> reward_us -> SNc; the temporal-difference
                            # delta=r-V is left to the dopamine RPE critic). NOTE: the older
                            # approach_n5 (slow-channel TD) region was dropped per :2532, so this
                            # branch previously checked a region that never exists -> the host write
                            # at the `else` always won (latent bug). Now sc_rostral carries r.
                            bridge.cp_external_input_current[region_indices_cp["reward_us"]] = cp.float32(0.0)
                        else:
                            bridge.cp_external_input_current[region_indices_cp["reward_us"]] = (
                                cp.float32(float(reward_us_drive_pa) * max(0.0, float(reward))))
                        _I_snc = float(snc_tonic_pa)
                    else:
                        _I_snc = (
                            float(snc_tonic_pa)
                            + float(snc_reward_gain) * max(0.0, float(reward))
                            # NO host -k_v*V term: the striosome_value GABA_B
                            # inhibition subtracts V at the membrane.
                        )
                else:
                    _V_scaffold = max(0.0, float(reward_ema_pre))
                    _I_snc = (
                        float(snc_tonic_pa)
                        + float(snc_reward_gain) * max(0.0, float(reward))
                        - float(snc_value_gain) * _V_scaffold
                    )
                bridge.cp_external_input_current[region_indices_cp["snc"]] = (
                    cp.float32(_I_snc)
                )

            # Surprise-boosted learning rate (opt-in): NE-like fast meta-modulation.
            # When |RPE| is high, temporarily boost reward_learning_rate. Restored
            # after reward hold. Decoupled from per-action DA gating mechanism.
            base_lr = float(learning_rate)
            if enable_surprise_lr_boost:
                surprise = abs(rpe)
                bridge.core_config.reward_learning_rate = base_lr * (1.0 + surprise_lr_alpha * surprise)

            # Accumulate SNc spikes over the reward-hold window (the spiking RPE
            # READOUT — measured from cp_firing_states, not a formula). Logged
            # for diagnostics / the calibration harness. Stage B also accumulates
            # the striosome_value (critic) firing = the learned value V.
            _snc_spikes_this_trial = 0
            _striov_spikes_this_trial = 0
            for _ in range(reward_hold_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                bridge.runtime_state.current_time_ms = (
                    bridge.runtime_state.current_time_step * cfg.dt_ms
                )
                if spiking_snc and _snc_idx_host is not None:
                    _snc_spikes_this_trial += int(
                        bridge.cp_firing_states[_snc_idx_host].sum()
                    )
                if _striov_idx_host is not None:
                    _striov_spikes_this_trial += int(
                        bridge.cp_firing_states[_striov_idx_host].sum()
                    )
            if spiking_snc:
                snc_rate_log.append(_snc_spikes_this_trial)
            if _striov_idx_host is not None:
                striov_rate_log.append(_striov_spikes_this_trial)
            bridge.core_config.current_reward_signal = 0.0
            # Restore base reward_learning_rate (in case surprise-boosted)
            if enable_surprise_lr_boost:
                bridge.core_config.reward_learning_rate = base_lr
            # Silence the spiking US afferent between reward deliveries (it fires ONLY during the
            # reward-hold above), so reward_us doesn't spuriously drive the SNc during the next nav
            # integration. (The host-write path zeroes nothing here because it writes the SNc directly.)
            if spiking_reward_us and "reward_us" in region_indices_cp:
                bridge.cp_external_input_current[region_indices_cp["reward_us"]] = cp.float32(0.0)

        if verbose and progress_print_interval > 0 and (step + 1) % progress_print_interval == 0:
            recent_dist = float(np.mean(distance_log[-100:]))
            # Per-step action + reward surfaced for live-mode HUD (parsed by
            # webapp ProgressEvent regex). action_log[step] is the action just
            # taken at this step; reward_log[step] is the reward observed.
            _last_action_idx = action_log[step] if step < len(action_log) else -1
            _action_letter = "NESW"[_last_action_idx] if 0 <= _last_action_idx < 4 else "?"
            _last_reward = float(reward_log[step]) if step < len(reward_log) else 0.0
            print(f"[g11 seed={seed}] step {step+1}/{n_steps}  pos=({x},{y})  "
                  f"goal=({gx},{gy})  recent_dist={recent_dist:.2f}  "
                  f"action={_action_letter}  reward={_last_reward:+.2f}  "
                  f"actions={action_log[-100:].count(0):>3d}N/{action_log[-100:].count(1):>3d}E/"
                  f"{action_log[-100:].count(2):>3d}S/{action_log[-100:].count(3):>3d}W",
                  flush=True)
            # Tier-1 universal progress event
            try:
                from sim.progress import emit_progress
                emit_progress(
                    "step", step + 1, n_steps,
                    phase=f"seed={seed}", unit="steps",
                    pos=[int(x), int(y)], goal=[int(gx), int(gy)],
                    recent_dist=round(float(recent_dist), 2),
                    action=_action_letter, reward=round(_last_reward, 2),
                )
            except Exception:
                pass

        # Live brain-activity frame (frontend-revamp Phase 1). Independent
        # throttle from progress: emit at most every `emit_activity_every`
        # steps so it stays ~5-30 Hz of sim-time. Fire-and-forget: the probe
        # does one host-side per-region reduction and emit_activity() prints a
        # stdout line and returns — it NEVER waits on a reader, so the sim is
        # never blocked by the viz. Whole block is a no-op when --emit-activity
        # is off (_activity_probe is None).
        if _activity_probe is not None and emit_activity_every > 0 \
                and (step + 1) % emit_activity_every == 0:
            try:
                _regions, _flux = _activity_probe.sample(bridge)
                _activity_emit(
                    bridge.runtime_state.current_time_ms,
                    _regions, _flux,
                    step=step + 1, seed=seed,
                )
            except Exception:
                # Activity emission must never crash a run.
                pass

        # Optional throttle for human-watchable speed in interactive mode.
        if trial_sleep_ms > 0:
            time.sleep(trial_sleep_ms / 1000.0)

    elapsed = time.time() - t0
    dist_arr = np.asarray(distance_log[1:])
    quarters = [float(dist_arr[i*len(dist_arr)//4:(i+1)*len(dist_arr)//4].mean())
                for i in range(4)]

    # Per-phase stats
    phase_stats = []
    phase_boundaries = [0] + goal_change_steps + [n_steps]
    for phase_idx in range(len(phase_boundaries) - 1):
        p_start = phase_boundaries[phase_idx]
        p_end = phase_boundaries[phase_idx + 1]
        p_dist = dist_arr[p_start:p_end]
        p_actions = action_log[p_start:p_end]
        if len(p_dist) == 0:
            continue
        p_goal = goal_log[p_start + 1] if p_start + 1 < len(goal_log) else goal_log[-1]
        phase_stats.append({
            "phase": phase_idx,
            "step_start": p_start, "step_end": p_end,
            "goal": list(p_goal),
            "mean_distance": float(p_dist.mean()),
            "final_quarter_mean_distance": float(p_dist[len(p_dist)*3//4:].mean())
                if len(p_dist) >= 4 else float(p_dist.mean()),
            # Adaptation-speed metric (2026-04-30): mean Manhattan distance
            # over the FIRST quarter of the phase, after the goal change.
            # final_quarter measures asymptotic skill; first_quarter measures
            # how quickly the agent re-adapts. Useful for testing whether
            # mechanisms (replay, fast-credit-assignment) help adaptation
            # vs steady-state navigation. Both shipped so post-hoc analyses
            # don't need to recompute from distance_log.
            "first_quarter_mean_distance": float(p_dist[:len(p_dist)//4].mean())
                if len(p_dist) >= 4 else float(p_dist.mean()),
            "n_steps_at_goal": int((p_dist == 0).sum()),
            "n_steps": len(p_dist),
            "action_counts": [int((np.asarray(p_actions) == a).sum())
                              for a in range(N_ACTIONS)],
        })

    # ── STEP 2a merge-gate (a) probe: the max cortex_X->str_D1_X actor weight
    #    after the run. Mirrors _mean_critic_weight's CSR read, restricted to the
    #    four same-action cortex->D1 pathways (the navigation ACTOR). Used by the
    #    nav-gate (a) check to verify that raising cfg.stdp_w_max (150->400, the 5a
    #    clip mitigation) does NOT let the soft-bound actor over-grow toward 400.
    #    Always computed (cheap, ~one CSR isin), stored in the results JSON;
    #    PRINTED only when --stdp-w-max was passed (so default runs are unchanged).
    def _actor_max_weight():
        try:
            pre_list, post_list = [], []
            for a in ACTION_NAMES:
                if f"cortex_{a}" in region_indices_cp and f"str_D1_{a}" in region_indices_cp:
                    pre_list.append(region_indices_cp[f"cortex_{a}"].get())
                    post_list.append(region_indices_cp[f"str_D1_{a}"].get())
            if not pre_list:
                return None
            pre = np.concatenate(pre_list)
            post = np.concatenate(post_list)
            coo = bridge.cp_connections.tocoo()
            rows = coo.row.get() if hasattr(coo.row, "get") else np.asarray(coo.row)
            cols = coo.col.get() if hasattr(coo.col, "get") else np.asarray(coo.col)
            data = coo.data.get() if hasattr(coo.data, "get") else np.asarray(coo.data)
            m = np.isin(rows, pre) & np.isin(cols, post)
            if not m.any():
                m = np.isin(rows, post) & np.isin(cols, pre)
            return float(data[m].max()) if m.any() else None
        except Exception:
            return None

    _actor_w_max = _actor_max_weight()
    if stdp_w_max_override is not None:
        _gate_score = sum(p["final_quarter_mean_distance"] for p in phase_stats)
        print(f"[g11 seed={seed}] NAV-GATE(a) stdp_w_max={cfg.stdp_w_max:.1f}  "
              f"sum_finalQ={_gate_score:.4f}  mean_distance_overall={float(dist_arr.mean()):.4f}  "
              f"actor_max_cortex_to_D1_weight={_actor_w_max}", flush=True)

    results = {
        "seed": seed, "n_steps": n_steps, "grid_size": grid_size,
        "start_pos": list(start_pos), "goal_pos": list(goal_pos),
        "stdp_w_max": float(cfg.stdp_w_max),
        "actor_max_cortex_to_D1_weight": _actor_w_max,
        "goal_schedule": [[s, list(g)] for s, g in goal_schedule_sorted],
        "goal_change_steps": goal_change_steps,
        "phase_stats": phase_stats,
        "reward_learning_rate": learning_rate,
        "trajectory": trajectory, "goal_log": goal_log,
        "motor_counts": motor_counts_log,
        "thal_counts": thal_counts_log,
        "sel_counts": sel_counts_log,
        "commit_counts": commit_counts_log,
        "accum_trace": accum_trace_log,
        "use_commit_readout": bool(_use_commit),
        "decision_path_counts": dict(_decision_path_counts),
        "reset_losers_only": bool(reset_losers_only),
        "urgency_max_pA": float(urgency_max_pA),
        "readout_source": _readout_source,
        # N1 adaptive weaning instrumentation. adaptive_wean_commit_step is -1 if
        # adaptive weaning was off or never committed; otherwise the step at which
        # the readiness probe passed and the permanent wean began. probe_history
        # is the sequence of probe windows ({probe_start, probe_end, mean_dist,
        # committed}) showing readiness rising over time.
        "heuristic_wean_adaptive": bool(heuristic_wean_adaptive),
        "adaptive_wean_commit_step": int(adaptive_commit_step),
        "adaptive_wean_probe_history": adaptive_probe_history,
        "adaptive_wean_probe_params": {
            "every": int(wean_probe_every),
            "window": int(wean_probe_window),
            "threshold": float(wean_probe_threshold),
            "wean_steps": int(heuristic_wean_steps),
        },
        "action_log": action_log, "reward_log": reward_log,
        "distance_log": distance_log,
        # Spiking-SNc Stage A (2026-06-08): per-trial SNc spike count over the
        # reward-hold window (the spiking reward-prediction error, read from
        # cp_firing_states). Empty unless --spiking-snc. V = host reward_ema
        # SCAFFOLD at Stage A (NOT a neural critic — honestly labeled).
        "snc_rate_log": snc_rate_log,
        "spiking_snc": bool(spiking_snc),
        "snc_value_source": (
            "neural_striosome_gabab" if (spiking_snc and enable_neural_critic)
            else "host_reward_ema_scaffold" if spiking_snc
            else None
        ),
        # Spiking-SNc Stage B (2026-06-08): the NEURAL value critic. V is the
        # striosome_value firing (per-trial spike count over the reward window),
        # subtracted at the SNc membrane via GABA_B/GIRK. critic_weight_* track
        # the plastic cortex_it->striosome_value weight learning (the smoke gate:
        # the weight should GROW from its init and striov_rate_log should track V).
        "enable_neural_critic": bool(enable_neural_critic),
        "critic_afferent": (_critic_afferent_region if enable_neural_critic else None),
        "critic_gabab_propagation_strength": (0.02 if enable_neural_critic else None),
        # 2026-06-09 VALIDATED redesign facts (the smoke / Stage-0 replication check reads these).
        "enable_critic_homeostasis": bool(enable_critic_homeostasis and enable_neural_critic),
        "global_homeostasis_off": (not cfg.enable_homeostasis),   # MUST be True (deterministic regime)
        "per_region_homeostasis_mask_set": bool(
            getattr(bridge, "cp_homeostasis_neuron_mask", None) is not None),
        "n_vs_place_context": (int(n_vs_place_context) if enable_neural_critic else None),
        "enable_critic_window": bool(enable_critic_window and enable_neural_critic),
        "critic_lead_steps": (int(critic_lead_steps) if (enable_critic_window and enable_neural_critic) else None),
        "striov_rate_log": striov_rate_log,
        "critic_weight_initial": critic_weight_initial,
        "critic_weight_final": (_mean_critic_weight() if enable_neural_critic else None),
        # Critic value-acquisition warm-up (2026-06-09 deadlock-breaker) facts.
        "critic_warmup_trials": int(critic_warmup_trials),
        "critic_warmup_weight_pre": (float(_warmup_stats[0]) if _warmup_stats and _warmup_stats[0] is not None else None),
        "critic_warmup_weight_post": (float(_warmup_stats[1]) if _warmup_stats and _warmup_stats[1] is not None else None),
        "critic_warmup_n_goals": (int(_warmup_stats[2]) if _warmup_stats else 0),
        "mean_distance_overall": float(dist_arr.mean()),
        "mean_distance_quarters": quarters,
        "n_steps_at_goal": int((dist_arr == 0).sum()),
        "elapsed_seconds": elapsed,
        # Cluster D v2 (SWR replay) instrumentation. swr_sleep_steps is
        # 0 if v2 was off or no sleep phase ran; swr_burst_count is the
        # number of those steps where the gate was thawed by a detected
        # CA3 population burst. Healthy v2 run: burst rate ~5-15% of
        # sleep steps. <1% means the autoassociator never bursts (raise
        # replay drive). >40% means everything is "a burst" (tighten σ
        # threshold or extend history window).
        "swr_burst_count": swr_burst_count,
        "swr_sleep_steps": swr_sleep_steps,
        "swr_burst_fraction": (
            float(swr_burst_count) / swr_sleep_steps if swr_sleep_steps > 0 else 0.0
        ),
    }
    # Tier 2.2: post-nav language eval suite. Runs only when embodied
    # language was active during nav. Tests:
    #   W->A: drive direction word → motor pool fires correctly
    #   I->W: present visual scene → language_output emits direction word
    if embodied_language and enable_text_io:
        try:
            from research.runners.text_eval import (
                evaluate_word_to_action, evaluate_image_to_word,
            )
            # Freeze plasticity for eval phase
            for gate_name in ("language_input_to_cortex",
                              "language_input_to_motor",
                              "it_to_language_output",
                              "cortex_to_language_output",
                              "language_input_to_pfc"):
                try:
                    bridge.set_plasticity_gate(gate_name, 0.0)
                except KeyError:
                    pass
            if verbose:
                print(f"\n[g11 seed={seed}] Tier 2.2 EVAL: word -> action",
                      flush=True)
            wa_result = evaluate_word_to_action(
                bridge, n_trials_per_word=25, stim_steps_per_trial=100,
                n_reset_steps=50, token_sparsity=0.1, verbose=False,
            )
            results["tier22_word_to_action"] = wa_result
            if verbose:
                print(f"  W->A accuracy: {wa_result['accuracy']:.1%}",
                      flush=True)
            if verbose:
                print(f"\n[g11 seed={seed}] Tier 2.2 EVAL: image -> word",
                      flush=True)
            iw_result = evaluate_image_to_word(
                bridge, n_trials=100, grid_size=int(grid_size),
                stim_steps_per_trial=200, drive_pA=200.0, seed=seed,
                verbose=False,
            )
            results["tier22_image_to_word"] = iw_result
            if verbose:
                print(f"  I->W accuracy: {iw_result['accuracy']:.1%}",
                      flush=True)
        except Exception as e:
            if verbose:
                print(f"[g11 seed={seed}] Tier 2.2 eval failed: {e}",
                      flush=True)
            results["tier22_eval_error"] = str(e)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\n[g11 seed={seed}] DONE in {elapsed:.0f}s. "
              f"Phase stats:")
        for p in phase_stats:
            print(f"  phase {p['phase']} goal={p['goal']} "
                  f"meanD={p['mean_distance']:.2f} "
                  f"finalQ={p['final_quarter_mean_distance']:.2f} "
                  f"actions={p['action_counts']}")

    return results


def _ca3_burst_active(current_rate_hz: float, history) -> bool:
    """Detect a CA3 population burst (sharp-wave-ripple proxy).

    `current_rate_hz` is this step's CA3 mean firing rate. `history`
    is a `collections.deque` of recent rate samples (caller-owned;
    typical maxlen=40 ≈ 200ms at dt=5ms).

    Returns True when current_rate exceeds μ + 2σ of the recent
    history. Requires at least 10 prior samples to compute meaningful
    statistics; before that, returns False (and the caller should still
    push the current sample so the history fills up). σ is floored at
    1e-6 to avoid division by zero on flat signals — flat signals
    therefore cannot trigger a burst (μ + 2*0 = μ; current == μ won't
    cross the threshold).
    """
    history.append(current_rate_hz)
    if len(history) < 10:
        return False
    # Compute mu/sigma over the history *including* the current sample.
    # Including it doesn't bias the burst detection meaningfully because
    # the burst is a 1–2 step transient against a 40-sample window.
    n = len(history)
    mu = sum(history) / n
    var = sum((x - mu) ** 2 for x in history) / n
    sigma = max(var ** 0.5, 1e-6)
    return current_rate_hz > mu + 2.0 * sigma


def _swr_gate_value(in_sleep: bool, current_rate_hz: float, history) -> float:
    """Compute the plasticity gate value for the `ca3_swr_burst` gate this
    step using endogenous burst detection.

    During wake (`in_sleep=False`), the gate is always fully open (1.0)
    so cluster D v1's normal CA3 recurrent plasticity is unchanged.

    During sleep, the gate sits at a low baseline (0.1) suppressing
    most STDP, except during sharp-wave-ripple bursts (detected via
    `_ca3_burst_active`), when it temporarily opens to 1.0.

    NOTE: in our reduced ~100-neuron CA3 with 0.30 recurrent density and
    weight_mean=1.5, endogenous bursts do not reliably fire under the
    standard sleep-replay drive (verified empirically 2026-04-30: even
    220 pA into 10 CA3 neurons leaves V_mean at rest -65; only 1500 pA
    into all 100 produces firing). For the actual v2 eval the runner
    falls back to `_swr_gate_value_scheduled` which imposes SWR windows
    on a fixed schedule. This function is kept for unit-test coverage
    of the burst detector and as a future hook if CA3 dynamics become
    self-sustaining.
    """
    if not in_sleep:
        return 1.0
    if _ca3_burst_active(current_rate_hz, history):
        return 1.0
    return 0.1


def _swr_gate_value_scheduled(
    in_sleep: bool, sleep_step_index: int, period: int = 7
) -> float:
    """Compute the plasticity gate value for the `ca3_swr_burst` gate
    using a SCHEDULED ripple-window mechanism.

    Real cerebral SWR events are sparse and brief (~1/sec, ~100 ms each
    during NREM = ~10-15% duty cycle). This helper implements the same
    temporal restriction without requiring endogenous CA3 bursts: every
    `period`-th sleep env step is treated as a ripple window with the
    gate fully open (1.0); all other sleep steps gate at 0.1.

    Default period=7 → 14% duty cycle, matching biological NREM SWR rate.
    During wake, always 1.0 (v1 behavior preserved).

    The hypothesis under test is unchanged from the design doc: TEMPORAL
    RESTRICTION of plasticity windows during offline consolidation
    selectively reinforces structured replay events while suppressing
    reinforcement of constant-drive noise. The mechanism just imposes
    the timing externally rather than detecting it endogenously.
    """
    if not in_sleep:
        return 1.0
    return 1.0 if (sleep_step_index % period == 0) else 0.1


def _emit_webapp_sidecar_and_redirect_stdout(args) -> None:
    """Redirect stdout/stderr to a log file under webapp/runtime/ AND
    write a sidecar matching the webapp's launch format so the
    dashboard's Live-picker orphan-scan discovers this run and supports
    attach (live progress + trajectory replay) as if it had been
    launched via the webapp.

    Why dup2 rather than open(...).write: cupy / cuDNN / our own
    `print()` calls all write to file descriptor 1 / 2 directly. A
    Python-level `sys.stdout = ...` reassignment doesn't catch those.
    dup2 redirects at the OS level so every subsequent write — Python
    and native — goes to the log file.

    Sidecar fields mirror webapp/server.py launch_run sidecar so the
    same orphan-recovery code path handles both flavors.
    """
    import os as _os
    import sys as _sys
    import time as _time
    import uuid as _uuid
    import json as _json
    from pathlib import Path as _Path
    run_id = _uuid.uuid4().hex[:12]
    repo_root = _Path(__file__).resolve().parents[2]
    runtime_dir = repo_root / "webapp" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / f"run_{run_id}.log"
    log_handle = open(log_path, "w", buffering=1)  # line-buffered
    _os.dup2(log_handle.fileno(), 1)  # stdout
    _os.dup2(log_handle.fileno(), 2)  # stderr
    # Resolve the eventual out path so the sidecar lives next to it
    out_path = args.out or f"research/findings/raw/g11_bg/g11_seed{args.seed}.json"
    if not _os.path.isabs(out_path):
        out_path = str((repo_root / out_path).resolve())
    sidecar_path = _Path(out_path).with_suffix(".cmd.json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "run_id": run_id,
        "preset": "g11_bg_runner",
        "seed": args.seed,
        "extra_args": [a for a in _sys.argv[1:] if a != "--emit-webapp-sidecar"],
        "deterministic": getattr(args, "deterministic", False),
        "cmd": [_sys.executable, "-m", "research.runners.g11_bg_runner", *_sys.argv[1:]],
        "pid": _os.getpid(),
        "log_file": str(log_path),
        "control_file": getattr(args, "interactive_control_file", None),
        "out_path": out_path,
        "started_at": _time.time(),
        "runner_kind": "single",
    }
    sidecar_path.write_text(_json.dumps(sidecar, indent=2))
    print(f"[g11_bg_runner] webapp sidecar: {sidecar_path}")
    print(f"[g11_bg_runner] log: {log_path}")
    print(f"[g11_bg_runner] run_id={run_id} pid={_os.getpid()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: build + 50 steps at rest")
    ap.add_argument("--probe-action", type=str, default=None,
                    choices=ACTION_NAMES,
                    help="Drive cortex toward this action and measure motor output")
    ap.add_argument("--moving-goal", action="store_true",
                    help="Run G9-style moving-goal scenario (Phase B.T6 acid test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--grid-size", type=int, default=8,
                    help="Side length of square gridworld (default 8). Larger grids stress-test the architecture.")
    ap.add_argument("--n-hippocampus-per-layer", type=int, default=64,
                    help="Number of place + goal cells per layer (should be ~grid_size² for one cell per position; default 64 = 8×8).")
    ap.add_argument("--sensory-to-cortex-weight", type=float, default=10.0,
                    help="Initial mean weight for sensory→cortex pathway (default 10). Higher values let input layer drive cortex more strongly during phase 2.")
    ap.add_argument("--hippocampus-to-cortex-weight", type=float, default=10.0,
                    help="Initial mean weight for hippocampus→cortex pathway (default 10). Higher = stronger plastic input contribution.")
    # Canonical: --enable-dlpfc-wm. The implementation is a single recurrent
    # attractor modeling dlPFC working-memory persistent activity (catalog
    # G.06 / G.08), not the whole prefrontal cortex (dlPFC + vmPFC + OFC + ACC).
    # Legacy --pfc kept as alias for one release cycle (2026-04-29 Wave-1 #2).
    ap.add_argument("--enable-dlpfc-wm", "--pfc", action="store_true",
                    dest="pfc",
                    help="Enable a dlPFC working-memory module (one recurrent "
                         "attractor pool implementing persistent activity, NOT "
                         "the whole prefrontal cortex). Catalog G.06 / G.08.")
    ap.add_argument("--n-dlpfc-wm", "--n-pfc", type=int, default=60,
                    dest="n_pfc",
                    help="Number of dlPFC working-memory neurons (default 60).")
    ap.add_argument("--pfc-internal-density", type=float, default=0.2,
                    help="PFC recurrent connection density (default 0.2; higher = more persistent activity).")
    ap.add_argument("--goal-to-pfc-weight", type=float, default=8.0)
    ap.add_argument("--pfc-to-cortex-weight", type=float, default=8.0)
    ap.add_argument("--enable-pfc-nmda", action="store_true",
                    help="Cluster G v1 (Wang 2002, 2026-05-01): NMDA-mediated "
                         "recurrent excitation for PFC working memory. "
                         "Globally enables NMDA with elevated 0.5 NMDA:AMPA "
                         "ratio (PFC pyramidal calibration). Combined with "
                         "--enable-dlpfc-wm, gives true persistent activity "
                         "for delayed-response tasks. Default off.")
    ap.add_argument("--beacon-perception", action="store_true",
                    help="Item 1 Stage 1: enable beacon_sensors region with directional tuning. Sensors are driven each step based on perceived beacon strength + bearing.")
    ap.add_argument("--n-beacon-sensors", type=int, default=8,
                    help="Number of beacon sensors (default 8 = cardinal+diagonal).")
    ap.add_argument("--beacon-to-goal-weight", type=float, default=8.0)
    ap.add_argument("--beacon-max-intensity", type=float, default=600.0,
                    help="Peak sensor drive (pA) when on top of beacon (default 600).")
    ap.add_argument("--beacon-falloff", type=float, default=1.0,
                    help="Intensity = peak / (1 + falloff*distance). Higher = faster falloff.")
    ap.add_argument("--beacon-replaces-goal", action="store_true",
                    help="If set, beacon → goal_cells is the ONLY goal info source (true perception test). Otherwise beacon adds info on top of direct goal_cells drive.")
    ap.add_argument("--cue-reflex", action="store_true",
                    help="Item 1 Stage 3: cue-following reflex computes cortex drive from beacon sensors (innate sensorimotor wiring like phototaxis). Augments heuristic by default; use --cue-reflex-replaces-heuristic to fully replace.")
    ap.add_argument("--cue-reflex-strength", type=float, default=800.0,
                    help="Peak reflex drive (pA) — matches heuristic strength by default.")
    ap.add_argument("--cue-reflex-replaces-heuristic", action="store_true",
                    help="If set with --cue-reflex, the heuristic is fully disabled; only reflex provides cortex drive.")
    ap.add_argument("--sc-orienting-reflex", action="store_true",
                    help="N1 de-risk: innate superior-colliculus orienting reflex. Reads the goal's retinal direction from the rendered image ALONE (no coords) and pushes the matching cortex pool — the biological replacement for the coordinate heuristic-teacher. Requires --enable-visual-cortex; run with --heuristic-strength 0 so the reflex (not the heuristic) provides the drive.")
    ap.add_argument("--sc-reflex-strength", type=float, default=800.0,
                    help="SC orienting push (pA) — matches heuristic strength by default.")
    ap.add_argument("--learned-perception-from-vision", action="store_true",
                    help="Rank 2: drive the learned sensory population (--enable-learned-perception) from the IMAGE salience offset (position-preserving 'where' signal, NO coords) instead of (gx-x,gy-y). The plastic sensory->cortex learns a durable where->action mapping; the SC reflex teaches it. Requires --enable-visual-cortex --enable-learned-perception.")
    ap.add_argument("--sc-reflex-wean-start", type=int, default=-1,
                    help="Rank 2: step to begin weaning the SC orienting reflex to zero (-1 = never). The innate reflex teaches, then fades as the learned circuit matures (developmental scaffold).")
    ap.add_argument("--sc-reflex-wean-steps", type=int, default=1500,
                    help="Rank 2: linear ramp-to-zero window for the SC reflex wean.")
    ap.add_argument("--sensory-cortex-teacher-pA", type=float, default=0.0,
                    help="Rank 2 tuning: supervised motor-teacher (feedback-error-learning). When >0, the SC reflex drives its chosen target cortex pool at THIS strength (a clean supervised label for the sensory->cortex STDP) instead of the movement-reflex strength. Tightens the learned read-out if the learning rule (reward-STDP coarseness) is the precision bottleneck. 0 = off.")
    # Canonical name: --enable-landmark-sensor (the implementation is a sensor
    # abstraction, not landmark-cell biology). --landmarks is the legacy alias
    # kept for one release cycle (2026-04-29 Wave-1 rename).
    ap.add_argument("--enable-landmark-sensor", "--landmarks", action="store_true",
                    dest="enable_landmark_sensor",
                    help="Item 1 Stage 2: enable fixed-position landmark with directional sensors. Plastic landmark_sensors → place_cells pathway lets place cells self-organize from sensor patterns.")
    ap.add_argument("--n-landmark-sensors", type=int, default=8)
    ap.add_argument("--landmark-to-place-weight", type=float, default=8.0)
    ap.add_argument("--landmark-x", type=float, default=None,
                    help="Landmark x position (default = grid_size/2)")
    ap.add_argument("--landmark-y", type=float, default=None,
                    help="Landmark y position (default = grid_size/2)")
    ap.add_argument("--landmark-max-intensity", type=float, default=600.0)
    ap.add_argument("--landmark-falloff", type=float, default=1.0)
    ap.add_argument("--landmarks-replace-place", action="store_true",
                    help="If set, place_cells receive ONLY landmark-derived input (no direct (x,y) cheat). True Stage 2 perception test.")
    ap.add_argument("--sensed-reward", action="store_true",
                    help="Cheat #4: compute reward from beacon-intensity gradient (sensed signal) instead of Manhattan distance change (cheat). Requires --beacon-perception.")
    ap.add_argument("--perceived-approach-reward", action="store_true",
                    help="N5: coordinate-FREE perceived-approach reward. reward = sign(decrease in the goal's retinal eccentricity), read from the rendered image via sc_salience_offset_from_image (pixels only; coords never enter the reward logic). Replaces the Manhattan/sensed coordinate reward; the agent is reinforced for perceiving the goal getting closer (Schultz reward-fn-2 / Berridge wanting / phototaxis). Use with --enable-visual-cortex.")
    # Canonical: --enable-corticostriatal-cross (specifies cortex→striatum
    # cross-action, not BG-internal cross). Legacy --bg-cross-projections kept
    # as alias for one release cycle (2026-04-29 Wave-2 rename #19).
    # Currently NEGATIVE on cheat-5 evaluation; on hold pending biology buildout.
    ap.add_argument("--enable-corticostriatal-cross", "--bg-cross-projections",
                    action="store_true", dest="bg_cross_projections",
                    help="Cheat #5: enable cortex × str_D1 cross-projections (e.g. cortex_E → str_D1_W) at weak initial weight. Plasticity learns the right cross-strengths instead of hand-coded same-action-only.")
    ap.add_argument("--cross-projection-weight", type=float, default=5.0,
                    help="Initial weight for BG cross-projections (default 5.0 vs 25.0 same-action).")
    ap.add_argument("--cross-projection-density", type=float, default=1.0,
                    help="Cheat-5 option 2: pathway-level density of cross-projections at build time. "
                         "1.0=dense (24 cross-pathways, current default); 0.25=patch-matrix-like (6 of 24).")
    ap.add_argument("--cross-projection-topology-seed", type=int, default=0,
                    help="Cheat-5 option 2: deterministic RNG seed for which cross-pathways survive when density<1.0. "
                         "Vary independently from --seed to test topology-conditional reproducibility.")
    # Canonical: --enable-msn-lateral-inhibition (specifies MSN-MSN, not BG-wide).
    # Legacy --bg-lateral-inhibition kept as alias for one release cycle
    # (2026-04-29 Wave-1 rename #8). Note: catalog B.04 supplemental flags
    # this implementation as anatomically backwards — real cross-pool WTA in
    # striatum is FSI feedforward, not MSN-MSN feedback (Wilson 2007 PBR-160
    # ch 6). Kept as v3 default per 2026-04-28 evaluation; future biology
    # buildout should replace with FSI-mediated form.
    ap.add_argument("--enable-msn-lateral-inhibition", "--bg-lateral-inhibition",
                    action="store_true", dest="bg_lateral_inhibition",
                    help="v3 (2026-04-28): add MSN cross-pool lateral inhibition (24 GABAergic pathways). Sharpens action selection regardless of cheat #5; required prerequisite for cross-projection closure.")
    ap.add_argument("--lateral-inhibition-density", type=float, default=0.3,
                    help="Density of MSN cross-pool inhibitory pathways (default 0.3).")
    ap.add_argument("--lateral-inhibition-weight", type=float, default=2.0,
                    help="Weight of MSN cross-pool inhibitory connections (default 2.0).")
    ap.add_argument("--interactive-control-file", type=str, default=None,
                    help="If set, runner polls this JSON file at the start of "
                         "each trial for runtime control: paused (bool), "
                         "goal ([gx, gy] override, persistent), inject_reward "
                         "(one-shot additive). Used by webapp World-tab live "
                         "mode for click-to-teleport-goal etc.")
    ap.add_argument("--progress-print-interval", type=int, default=100,
                    help="Print a progress line every N steps (default 100). "
                         "Webapp interactive mode sets this to 1 for per-step "
                         "live animation.")
    ap.add_argument("--trial-sleep-ms", type=float, default=0.0,
                    help="Sleep this many ms between trials (default 0 = full "
                         "speed). Use 50-200 to watch the agent learn at "
                         "human-readable speed in interactive mode.")
    # Live brain-activity streaming (frontend-revamp Phase 1, 2026-06-08).
    ap.add_argument("--emit-activity", action="store_true",
                    help="Stream live per-region brain activity as throttled "
                         "[ACTIVITY] {json} stdout lines (the webapp Brain tab "
                         "consumes them). Default OFF — when off there is zero "
                         "per-step overhead and the run is byte-identical to a "
                         "science run. Fire-and-forget: never blocks the sim.")
    ap.add_argument("--emit-activity-every", type=int, default=5,
                    help="Emit an [ACTIVITY] frame every N steps (default 5). "
                         "Decoupled from the sim step rate so the stream stays "
                         "~5-30 Hz of sim-time regardless of step speed. Only "
                         "used when --emit-activity is set.")
    ap.add_argument("--bg-cross-thaw-step", type=int, default=-1,
                    help="Cheat #5 closure (2026-04-28): step at which bg_cross_projections "
                         "gate thaws to its phase-3 value. -1 = stay frozen. Recommended 1200 "
                         "for default 1800-step moving-goal episodes (~300 steps after goal "
                         "change at step 900). Requires --bg-cross-projections + --curriculum.")
    ap.add_argument("--bg-cross-phase3-gain", type=float, default=0.5,
                    help="Plasticity gain for bg_cross_projections in phase 3. 1.0 = full plastic, "
                         "0.5 = half-rate (slower than same-action; default), 0.0 = stay frozen.")
    # v4 (2026-04-28): developmental pretraining
    ap.add_argument("--developmental-pretraining", action="store_true",
                    help="v4 cheat-5 closure: run a critical-period analog "
                         "(all plasticity gates open) on N random goals before "
                         "the standard eval. Cross-projections freeze at eval "
                         "start. Requires --bg-cross-projections.")
    ap.add_argument("--pretraining-n-goals", type=int, default=10,
                    help="Number of random goal positions during pretraining (default 10).")
    ap.add_argument("--pretraining-steps-per-goal", type=int, default=3000,
                    help="Trials per pretraining goal (default 3000). 10x3000=30K "
                         "default total; reduce for tier-2 smoke (e.g. 1000) or "
                         "tier-1 wiring check (e.g. 1 goal x 1000).")
    ap.add_argument("--enable-structural-pruning", action="store_true",
                    help="Cheat-5 option 1: experience-dependent synapse pruning during "
                         "pretraining. Synapses with negative survival score AND low weight "
                         "get permanently eliminated. See "
                         "docs/plans/2026-04-28-structural-plasticity-design.md.")
    ap.add_argument("--enable-d1-d2-asymmetry", action="store_true",
                    help="Cluster B.1: D1/D2 plasticity asymmetry — D2-targeting "
                         "synapses' weight updates flip sign vs D1. See "
                         "docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md.")
    # Canonical: --enable-striatal-pv-fsi (specifies PV+ FSI; per Tepper-2018
    # this is one of EIGHT distinct striatal GABAergic interneuron classes —
    # NPY-LTS, NPY-NGF, CR, TH/THIN, FAI, SABI, ChI/TAN are NOT modeled).
    # Legacy --enable-striatal-fsis kept as alias for one release cycle
    # (2026-04-29 Wave-1 rename #9). Region naming: str_PV_FSI_X (canonical)
    # with str_PV_FSI_X retained as a region-name alias via RegionManager.
    ap.add_argument("--enable-striatal-pv-fsi", "--enable-striatal-fsis",
                    action="store_true", dest="enable_striatal_fsis",
                    help="Cluster B.2: striatal PV-FSI fast-spiking interneurons "
                         "(broadcast inhibition). One of 8 striatal GABAergic "
                         "interneuron classes per Tepper-2018; the others are NOT "
                         "modeled. See "
                         "docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md.")
    # Canonical: --enable-msn-co-release (more specific — D1 co-releases
    # dynorphin + substance P with GABA, D2 co-releases enkephalin with GABA).
    # Legacy --enable-bg-neuropeptides kept as alias for one release cycle
    # (2026-04-29 Wave-2 rename #25).
    ap.add_argument("--enable-msn-co-release", "--enable-bg-neuropeptides",
                    action="store_true", dest="enable_bg_neuropeptides",
                    help="R3.6 (2026-04-29): D1/D2 neuropeptide co-release. "
                         "Registers dynorphin (D1, KOR plasticity-rate brake), "
                         "substance P (D1, NK-1 ACh boost), and enkephalin "
                         "(D2, DOR plasticity-rate boost) neuromodulators. "
                         "Per PBR-160 ch 16 McGinty.")
    ap.add_argument("--enable-cluster-a-closed-loop", action="store_true",
                    help="Cluster A (2026-04-29): closed BG loop. Adds "
                         "cortex_X -> stn (hyperdirect, sparse) and "
                         "thal_X -> cortex_X (action-specific feedback). "
                         "Provides the teaching signal missing for "
                         "cross-projection learning. See "
                         "docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md.")
    ap.add_argument("--genuine-thal-disinhibition", action="store_true",
                    help="N8 cheat conversion (2026-06-06): release the thalamic "
                         "relay via a genuine GPi->thalamus DISINHIBITION cascade "
                         "instead of the default tonic 300 pA thal drive. GPi "
                         "becomes a strong tonic pacemaker (--genuine-gpi-tonic-pa, "
                         "default 1000) that silences thal_X by default; the "
                         "selected action's cortex drive -> D1 -> (GABA) GPi "
                         "silence -> thal_X disinhibited. thal carries only a tonic "
                         "excitation (--genuine-thal-tonic-pa, default 600) "
                         "expressed when released. Pathways already wired "
                         "(D1->GPi=15, GPi->thal=8). Ported from "
                         "gated_compose_bg_genuine_demo.py. Default OFF keeps the "
                         "tonic-drive cheat as the control.")
    ap.add_argument("--genuine-gpi-tonic-pa", type=float, default=1000.0,
                    help="Tonic GPi pacemaker drive (pA) when "
                         "--genuine-thal-disinhibition is set (default 1000).")
    ap.add_argument("--genuine-thal-tonic-pa", type=float, default=900.0,
                    help="Tonic thalamic excitation (pA, expressed only when GPi "
                         "releases the relay) when --genuine-thal-disinhibition is "
                         "set (default 900).")
    ap.add_argument("--readout-source", choices=["motor", "thal", "spiking_wta"],
                    default="motor",
                    help="N6 readout source: how action selection is read. "
                         "'motor' (default) = legacy host-argmax over motor_X "
                         "spike counts (the N6 cheat). 'thal' = host-argmax over "
                         "the cleanly-selective THALAMUS spike counts (biologizes "
                         "the SIGNAL SOURCE, still a host argmax). 'spiking_wta' = "
                         "biologize the DECISION: a dedicated read-only sel_X / "
                         "sel_FS_X selection layer driven feed-forward by the "
                         "clean thalamus competes via lateral inhibition, and the "
                         "winning (firing) sel_X IS the action — the selection "
                         "emerges from a spiking competition, not a host argmax "
                         "over rates. The sel layer has no back-projection to "
                         "thal, so the thal->motor cascade / navigation dynamics "
                         "are unperturbed. Combine with "
                         "--genuine-thal-disinhibition. See "
                         "research/findings/2026-06-06-N6-spiking-wta-readout-*.")
    ap.add_argument("--n-sel-per-action", type=int, default=20,
                    help="Neurons per sel_X selection pool (spiking_wta readout).")
    ap.add_argument("--n-sel-fs-per-action", type=int, default=10,
                    help="Neurons per sel_FS_X interneuron pool (spiking_wta).")
    ap.add_argument("--thal-to-sel-weight", type=float, default=30.0,
                    help="thal_X -> sel_X excitatory feed-forward EVIDENCE weight "
                         "(spiking_wta; modest so the accumulator integrates "
                         "rather than instantly saturating).")
    ap.add_argument("--sel-to-sel-fs-weight", type=float, default=20.0,
                    help="sel_X -> sel_FS_X excitatory weight (spiking_wta; "
                         "recruits the interneuron that suppresses the losers).")
    ap.add_argument("--sel-fs-to-sel-weight", type=float, default=5.0,
                    help="sel_FS_X -> sel_Y!=X inhibitory cross-pool weight "
                         "(spiking_wta; GENTLE — symmetric over-inhibition is "
                         "unstable per Rutishauser-Douglas-Slotine).")
    # Accumulate-then-commit readout (N6 fix, 2026-06-06).
    ap.add_argument("--sel-recurrent-density", type=float, default=0.5,
                    help="sel_X internal recurrent self-excitation density "
                         "(spiking_wta accumulate-then-commit; Wang-2002 NMDA "
                         "attractor. 0.0 reverts to the passive comparator).")
    ap.add_argument("--sel-recurrent-weight", type=float, default=1.0,
                    help="sel_X -> sel_X NMDA-slow recurrent gain (soft-WTA "
                         "alpha<1; amplifies+integrates the weak thal drive).")
    ap.add_argument("--no-commit-burst", dest="enable_commit_burst",
                    action="store_false",
                    help="Disable the commit_X / commit_OPN burst stage (ablation "
                         "to a #2 Rutishauser instantaneous soft-WTA; read sel_X "
                         "argmax instead of the commit threshold crossing).")
    ap.set_defaults(enable_commit_burst=True)
    ap.add_argument("--n-commit-per-action", type=int, default=20,
                    help="Neurons per commit_X burst pool.")
    ap.add_argument("--n-commit-opn", type=int, default=20,
                    help="Neurons in the shared commit_OPN omnipause pool.")
    ap.add_argument("--sel-to-commit-weight", type=float, default=22.0,
                    help="sel_X -> commit_X weight (the winning sel_X ramp fires "
                         "the all-or-none burst; the decision threshold).")
    ap.add_argument("--commit-recurrent-density", type=float, default=0.5,
                    help="commit_X internal recurrence density (all-or-none burst).")
    ap.add_argument("--commit-recurrent-weight", type=float, default=0.6,
                    help="commit_X -> commit_X recurrent gain (burst regeneration; "
                         "low to avoid OPN-driven rebound bursting).")
    ap.add_argument("--opn-to-commit-weight", type=float, default=10.0,
                    help="commit_OPN -> commit_X tonic inhibition weight (gentle "
                         "gate; symmetric crushing inhibition causes rebound bursts).")
    ap.add_argument("--commit-opn-tonic-pa", type=float, default=0.0,
                    help="Constant drive (pA) that keeps commit_OPN tonically "
                         "firing (the omnipause baseline; H.24). Default 0 = OFF: "
                         "a constant drive induces synchronized rebound bursting "
                         "across commit pools on this rate-coded substrate, so the "
                         "commit burst is gated by the sel_X ramp + intrinsic "
                         "threshold instead (the minimal-variant commit).")
    ap.add_argument("--reset-accumulator", dest="reset_accumulator_each_trial",
                    action="store_true",
                    help="Zero the sel_X/commit_X NMDA+conductance state each trial "
                         "(spiking_wta) so each decision integrates fresh evidence. "
                         "Default OFF: empirically NET NEGATIVE on grid-8 multi-goal "
                         "(removes carried-over drive -> commit goes silent ~55%% of "
                         "trials -> score worsens 4.71->6.93). The cross-trial "
                         "persistence (a working-memory latch) is kept by default.")
    ap.set_defaults(reset_accumulator_each_trial=False)
    ap.add_argument("--reset-losers-only", dest="reset_losers_only",
                    action="store_true",
                    help="N6 refinement 1: LOSER-ONLY accumulator reset (spiking_wta). "
                         "Zero the sel_X/commit_X NMDA+conductance state each trial on "
                         "every pool EXCEPT the previous trial's selected action — the "
                         "winner's working-memory latch persists (fast re-ramp when the "
                         "goal is stable) while the losers integrate fresh evidence, so "
                         "at a goal change the stale old winner decays instead of "
                         "out-competing the new thal evidence. Surgical alternative to "
                         "the naive --reset-accumulator (which zeroed the eventual "
                         "winner too → silent commit → worse). Cisek trial-wise "
                         "re-baselining of the losing options.")
    ap.set_defaults(reset_losers_only=False)
    ap.add_argument("--urgency-max-pa", type=float, default=0.0,
                    help="N6 refinement 2: CISEK URGENCY / collapsing bound "
                         "(spiking_wta). Peak (pA) of a ramping action-independent "
                         "urgency current injected into all sel_X over the readout "
                         "window (0 at readout_start → this at readout_end). The "
                         "effective commit bound collapses with elapsed time so a weak "
                         "late-phase winner still bursts within the 100ms window "
                         "(eliminates the silent-commit → argmax-residual fallback). "
                         "Same drive for every pool → no action bias. Default 0 = OFF. "
                         "(Cisek 2009; Thura-Cisek 2014; Lo-Wang DA-modulated bound.)")
    ap.add_argument("--enable-thal-lateral-inhibition", action="store_true",
                    help="N8+N6 (2026-06-06): TRN-style lateral inhibition "
                         "between thalamic relay pools. A biological WTA on the "
                         "thalamic relay (the clean genuine-disinhibition "
                         "signal) — the thalamic reticular nucleus reciprocally "
                         "inhibits relay nuclei (Pinault 2004), sharpening the "
                         "released winner and silencing leaked losers so a "
                         "--readout-source thal argmax sees one clean winner. "
                         "Adds thal_FS_X pools + thal_X->thal_FS_X (exc) + "
                         "thal_FS_X->thal_Y!=X (inh). Combine with "
                         "--readout-source thal + --genuine-thal-disinhibition.")
    ap.add_argument("--thal-to-fs-weight", type=float, default=50.0,
                    help="thal_X -> thal_FS_X excitatory weight for the TRN WTA "
                         "(default 50, mirrors the motor WTA).")
    ap.add_argument("--thal-fs-to-thal-weight", type=float, default=20.0,
                    help="thal_FS_X -> thal_Y!=X inhibitory weight for the TRN "
                         "WTA (default 20, mirrors the motor WTA).")
    ap.add_argument("--learning-rate", type=float, default=0.01,
                    help="reward_learning_rate for STDP/reward modulation "
                         "(default 0.01). Set 0.0 to freeze plasticity (diagnostic: "
                         "isolate dynamics from cumulative weight changes).")
    ap.add_argument("--stdp-w-max", type=float, default=None,
                    help="STEP 2a merge integration: override the STDP soft-bound "
                         "ceiling cfg.stdp_w_max. Default None = the computed value "
                         "(150). The merge uses 400 (the 5a clip mitigation above the "
                         "~300 frozen parser role-route). The nav-gate (a) check runs "
                         "the flagship at --stdp-w-max 400 vs the default 150 to verify "
                         "the soft-bound nav actor (cortex->D1) does not over-grow.")
    ap.add_argument("--enable-tonic-da", action="store_true",
                    help="Cluster C v1 (2026-04-29): replace signed-scalar "
                         "reward modulation with a real `dopamine` "
                         "neuromodulator (tonic baseline + phasic "
                         "activation/depression). Unlocks B.3 TANs by "
                         "providing tonic DA-driven plasticity for ACh "
                         "to gate. See "
                         "docs/plans/2026-04-29-cluster-c-tonic-da-design.md.")
    ap.add_argument("--enable-compartmentalized-da", action="store_true",
                    help="Cluster C v2 (2026-04-29): replace single-channel "
                         "DA with 4 per-action DA modulators "
                         "(dopamine_{N,E,S,W}). Each targets only synapses "
                         "with matching action_index; production rule fires "
                         "only when last_selected_action matches. Implies "
                         "tonic DA at the per-action level (the global "
                         "`dopamine` modulator is NOT registered when this "
                         "flag is on, even if --enable-tonic-da is set). "
                         "See docs/plans/2026-04-29-cluster-c-v2-"
                         "compartmentalized-da-design.md.")
    ap.add_argument("--enable-cluster-d-hippocampus", action="store_true",
                    help="Cluster D v1 (2026-04-29): hippocampus trisynaptic "
                         "loop. Adds 5 regions (ec, dg, dg_pv_basket, ca3, ca1) and "
                         "~10 pathways implementing the canonical Cajal loop "
                         "(EC -> DG -> CA3 -> CA1 + EC -> CA1 direct + CA3 "
                         "recurrent autoassociator). Composes with --hippocampus "
                         "(adds ca1 -> place_cells readout) and --landmarks "
                         "(adds landmark_sensors -> ec). See "
                         "docs/plans/2026-04-29-cluster-d-hippocampus-design.md.")
    ap.add_argument("--enable-cluster-d-v2-swr", action="store_true",
                    help="Cluster D v2 (2026-04-30): SWR-gated CA3 plasticity "
                         "for offline cleanup. REQUIRES --enable-cluster-d-"
                         "hippocampus. Replaces CA3's implicit recurrent "
                         "autoassociator with an explicit ca3 -> ca3 pathway "
                         "tagged with the `ca3_swr_burst` plasticity gate; "
                         "the runner detects population bursts in CA3 during "
                         "sleep replay and only thaws plasticity during burst "
                         "windows. See "
                         "docs/plans/2026-04-30-cluster-d-v2-swr-design.md.")
    ap.add_argument("--enable-cluster-e-topography", action="store_true",
                    help="Cluster E v1 (2026-04-29): topographic maps + "
                         "distance-dependent connection probability. "
                         "cortex_X / str_D1_X / str_D2_X regions get 2D coords "
                         "anchored to corners of unit square (N=(0.5,1.0), "
                         "E=(1.0,0.5), S=(0.5,0.0), W=(0.0,0.5)); cortex_X -> "
                         "str_D{1,2}_Y pathways are sampled with Gaussian-"
                         "weighted probability (sigma=0.3 default, set via "
                         "--cluster-e-distance-sigma). See "
                         "docs/plans/2026-04-29-cluster-e-topographic-maps-design.md.")
    ap.add_argument("--cluster-e-distance-sigma", type=float, default=0.3,
                    help="Cluster E v1 Gaussian-kernel sigma for distance-"
                         "weighted cortex -> striatum connectivity. Default 0.3 "
                         "(at corner-to-corner distance ~1.0, cross-action prob "
                         "drops to ~0.4%% of same-action). Larger -> looser "
                         "spatial selectivity.")
    ap.add_argument("--n-granule", type=int, default=250,
                    help="Cerebellar granule cell count. Default 250 implements "
                         "Marr's sparse-expansion code in our reduced model. "
                         "Real cerebellum has ~50M granule cells per hemisphere "
                         "with ~150K parallel-fiber inputs per Purkinje cell. "
                         "Scaling experiment 2026-04-30: 1000-5000 tests "
                         "whether F v2 (Albus 1971 anti-Hebbian LTD) becomes "
                         "viable at closer-to-biological scale.")
    ap.add_argument("--enable-cluster-f-cerebellum", action="store_true",
                    help="Cluster F v1 (2026-04-29): Marr-Albus-Ito cerebellar "
                         "microcircuit. Adds 11 regions (mossy_state, granule, "
                         "purkinje_{N,E,S,W}, dcn_aip_{N,E,S,W}, "
                         "inferior_olive) and ~25 pathways implementing the "
                         "MF -> GC -> PF -> PC -> DCN -> motor forward path "
                         "plus IO -> PC climbing-fiber teaching signal. "
                         "DCN_aip_X provides additive contribution to motor_X "
                         "alongside thal_X drive. The granule->purkinje_X "
                         "pathway is the learning site (gate "
                         "'cerebellum_pf_pc'). v1 uses reward-modulated STDP "
                         "via the existing infrastructure; full CF-gated LTD "
                         "deferred to v2. See "
                         "docs/plans/2026-04-29-cluster-f-cerebellum-design.md.")
    ap.add_argument("--enable-cluster-f-v2", action="store_true",
                    help="Cluster F v2 (2026-04-30): CF-gated anti-Hebbian LTD "
                         "per Albus 1971 §IV.C eq.4. Decouples cerebellum_pf_pc "
                         "plasticity from the global reward signal — PF→PC "
                         "synapses see -1.0 ONLY when IO is active (CF event), "
                         "0.0 otherwise. Per Albus, cerebellum should weaken "
                         "PF synapses on error events but never strengthen "
                         "on positive reward. Requires "
                         "--enable-cluster-f-cerebellum. See "
                         "research/findings/2026-04-29-cluster-f-results.md.")
    ap.add_argument("--enable-tans", action="store_true",
                    help="Cluster B.3: cholinergic interneurons (TANs). Adds "
                         "an acetylcholine_tan neuromodulator (the striatal-TAN-"
                         "specific ACh source) that pauses on reward and gates "
                         "corticostriatal plasticity windows. See "
                         "docs/plans/2026-04-28-cluster-b3-tans-implementation.md.")
    ap.add_argument("--enable-visual-cortex", action="store_true",
                    help="Cluster K v2 (2026-05-01): visual cortex hierarchy "
                         "(Hubel-Wiesel 1962, Felleman & Van Essen 1991). "
                         "Adds retina (32x32 ON/OFF) -> V1_simple (Gabor pre-"
                         "init via apply_v1_gabor_weights, 1024 cells) -> "
                         "V1_complex (512, phase-pooled) -> V2 (256, plastic) "
                         "-> IT (64, plastic) -> cortex_{N,E,S,W} (action "
                         "selection, plastic, gated visual_cortex_action). "
                         "Env step loop renders the gridworld as a 32x32 image "
                         "each step and drives the retina. The IT -> cortex "
                         "pathway is initialized at zero weight and frozen "
                         "until --visual-cortex-action-warmup-steps; STDP+"
                         "reward then grows the visuomotor weights from zero. "
                         "Mimics real visual development (sensory critical "
                         "period -> visuomotor maturation). Compose with or "
                         "without --heuristic-single-pool / perception arc.")
    ap.add_argument("--enable-spiking-sc", action="store_true",
                    help="Spiking superior colliculus (N1 orienting; 2026-06-10). A "
                         "retinotopic sc_map (16x16) + Mexican-hat sc_fs surround that, fed "
                         "the egocentric retinal image, forms an activity bump at the goal's "
                         "retinal site; sc_map -> cortex_{N,E,S,W} pooling reads the orienting "
                         "cardinal BY NEURON FIRING -- the spiking replacement for the host "
                         "sc_orienting_cardinal_from_image reflex. Requires --enable-visual-cortex. "
                         "De-risked: 2026-06-10-N1-N5-spiking-SC-derisk-RESULT.md (N1 8/8, lesion-confirmed).")
    ap.add_argument("--enable-spiking-sc-approach", action="store_true",
                    help="N5 neural reward (2026-06-10). The SC bump's PROXIMITY/goal-salience "
                         "signal (sc_rostral, pooling the sc_map centre) drives reward_us, replacing "
                         "the host sign(delta eccentricity); the temporal-difference is left to the "
                         "dopamine RPE (delta=r-V, N9). Requires --enable-spiking-sc + --spiking-reward-us. "
                         "VALIDATED: sc_n5_rpe_probe.py (neural reward -> graded dopamine RPE, "
                         "lesion+omission confirmed).")
    ap.add_argument("--visual-cortex-action-warmup-steps", type=int, default=600,
                    help="Cluster K v2: steps before the IT -> cortex_X "
                         "plasticity gate opens. Default 600. 0 = open from "
                         "start (no critical period); -1 = stay closed forever "
                         "(visual cortex passive observer, doesn't drive "
                         "action).")
    ap.add_argument("--visual-v1-weight-scale", type=float, default=10.0,
                    help="Cluster K v2: multiplier on Gabor weights when "
                         "applied to retina -> V1_simple. Default 10.0. The "
                         "Gabor cosine values are in [-1, 1]; weight_scale=10 "
                         "gives roughly 10pA per active pixel, comparable to "
                         "other plastic pathways.")
    ap.add_argument("--visual-image-size", type=int, default=32,
                    help="Retina spatial dimension (default 32, gives 32x32 "
                         "image = 1024 pixels per channel × 2 channels = 2048 "
                         "retina neurons). MUST be >= grid_size or "
                         "render_gridworld_to_image will fail with "
                         "pixels_per_cell=0. For grid_size > 32, set this to "
                         "match grid_size (e.g. --grid-size 64 --visual-image-size 64).")
    ap.add_argument("--enable-text-io", action="store_true", default=False,
                    help="Enable text I/O regions (language_input + "
                    "language_output) for bidirectional text training. "
                    "Required for --embodied-language.")
    ap.add_argument("--embodied-language", action="store_true", default=False,
                    help="Tier 2.2 (2026-05-06): drive language regions "
                    "during navigation in sync with agent's perception/action. "
                    "When agent executes action a, drive language_input + "
                    "language_output with word(a). When agent perceives goal, "
                    "drive language with 'goal'. STDP at lang↔motor and "
                    "lang↔IT pathways binds words to embodied concepts via "
                    "Pulvermüller somatotopic semantics. Requires "
                    "--enable-text-io.")
    ap.add_argument("--embodied-language-drive-pA", type=float, default=80.0,
                    help="Drive amplitude for language regions during "
                    "embodied training (default 80pA — moderate, "
                    "supplements rather than dominates nav). Tier 1 used "
                    "200pA but in isolation; here nav has 100-200pA "
                    "retina + BG inputs running concurrently.")
    ap.add_argument("--embodied-language-goal-radius", type=int, default=3,
                    help="Manhattan distance threshold within which agent "
                    "is considered to 'perceive' the goal (drives 'goal' "
                    "word teacher). Default 3.")
    ap.add_argument("--embodied-language-every-n-steps", type=int, default=5,
                    help="Drive language regions every N steps (default 5). "
                    "Sporadic — biology pairs language with experience "
                    "episodically, not at every microsecond.")
    ap.add_argument("--embodied-language-warmup-steps", type=int, default=600,
                    help="Skip embodied-language until step N (default 600). "
                    "Lets nav converge first; language binds to 'intentional' "
                    "actions, not random walk. Mirrors child language "
                    "acquisition: words are heard during competent action.")
    ap.add_argument("--pruning-alpha", type=float, default=None,
                    help="Cheat-5 option-1 pruning rate. Default: cfg.pruning_alpha (0.001 = conservative). "
                         "Try 0.05 for a 5K-trial pretraining smoke; 0.005 for 30K validation.")
    ap.add_argument("--pruning-threshold", type=float, default=None,
                    help="Cheat-5 option-1: survival score below which pruning is eligible. Default: -1.0.")
    ap.add_argument("--pruning-weight-floor", type=float, default=None,
                    help="Cheat-5 option-1: weight below which pruning is eligible. Default: 1.0.")
    ap.add_argument("--out", type=str, default=None)
    # DEPRECATED 2026-04-29 (Wave-1 rename master plan #11). NEGATIVE on
    # cheat-5 evaluation; biology is wrong (real motor-pool WTA is via spinal
    # Renshaw cells / reciprocal inhibition per Kandel ch 35, not cortical-FS-
    # like inhibition). Slated for removal in a future cleanup. The
    # motor_FS_X regions and motor_X→motor_FS_X→motor_Y plumbing remain for
    # archival reproducibility of 2026-04-26 findings.
    ap.add_argument("--motor-lateral-inhibition", "--enable-motor-pool-wta",
                    action="store_true", dest="motor_lateral_inhibition",
                    help="DEPRECATED (NEGATIVE on cheat-5; slated for removal). "
                         "Enable FS-mediated motor pool lateral inhibition "
                         "(WTA microcircuit). Real motor-pool WTA biology is "
                         "spinal Renshaw, not cortical-FS-like inhibition.")
    # Canonical: --enable-m1-pv-basket. Implementation is per-pool FS+ basket
    # cells (cortical PV+ basket biology, Kandel ch 17). Legacy --cortex-wta
    # kept as alias for one release cycle (2026-04-29 Wave-2 rename #24).
    # NB: cortex_FS_X regions remain on the legacy name pending #23 (Wave-2;
    # paired with broader cortical interneuron taxonomy expansion).
    ap.add_argument("--enable-m1-pv-basket", "--cortex-wta", action="store_true",
                    dest="cortex_wta",
                    help="Enable M1-level PV+ basket-cell WTA: per-pool FS interneurons enforce one-cortex-pool-wins. Tools plastic input layers (place_goal_readout, learned-perception) to coexist with heuristic.")
    ap.add_argument("--per-action-da", action="store_true",
                    help="Enable per-action dopamine targeting (hard): reward only credits chosen action's cortex->D1 synapses")
    ap.add_argument("--adaptive-da", action="store_true",
                    help="Enable ADAPTIVE per-action DA: gating strength scales with recent reward EMA (low reward -> broadcast)")
    ap.add_argument("--adaptive-da-ema-decay", type=float, default=0.9,
                    help="EMA decay for adaptive DA (default 0.9, tau~10 trials; lower = faster reaction)")
    ap.add_argument("--adaptive-da-ema-decay-negative", type=float, default=None,
                    help="Separate (faster) EMA decay for negative reward (asymmetric ramp; biologically: phasic DA dip)")
    ap.add_argument("--learned-perception", action="store_true",
                    help="Enable learned sensory->cortex mapping (49-neuron sensory layer, plastic to cortex)")
    ap.add_argument("--informed-init", action="store_true",
                    help="Bias initial sensory->cortex weights by directional alignment (requires --learned-perception)")
    ap.add_argument("--informed-init-alpha", type=float, default=8.0,
                    help="Strength of positive-only directional prior (default 8.0; aligned weight ~24.5, orthogonal ~0.5)")
    # Canonical: --enable-place-goal-readout. The flag adds two abstract
    # sensor-driven regions (sensor_place_readout, ppc_goal_input). Per
    # glossary: the readout cells are NOT canonical allocentric place cells
    # (sensor-driven, not allocentric per O'Keefe & Nadel 1978 criteria);
    # the goal-encoding cells are anatomically PPC-like, not hippocampal.
    # Legacy --hippocampus kept as alias for one release cycle (2026-04-29
    # Wave-1 renames #4/#5/#6). For canonical hippocampus biology, use
    # --enable-cluster-d-hippocampus (DG/CA3/CA1 trisynaptic pathway).
    ap.add_argument("--enable-place-goal-readout", "--hippocampus",
                    action="store_true", dest="hippocampus",
                    help="Enable place-goal readout module: 64 sensor-driven "
                         "place readout cells (sensor_place_readout) + 64 "
                         "goal-vector cells (ppc_goal_input) with sparse "
                         "Gaussian tuning, plastic to cortex. NOT canonical "
                         "allocentric place cells; for that use "
                         "--enable-cluster-d-hippocampus.")
    ap.add_argument("--da-gated-wta", action="store_true",
                    help="Scale motor FS->motor inhibition by reward-EMA gating_strength (the 'DA gate'). Requires --motor-lateral-inhibition + --adaptive-da")
    ap.add_argument("--goal-schedule", type=str, default="default",
                    help="'default' = (6,6) -> (1,6) at step 300. 'multi' = 4 goal changes across the corners.")
    ap.add_argument("--deterministic", action="store_true",
                    help="Set CUBLAS_WORKSPACE_CONFIG=:4096:8 BEFORE cupy import for "
                         "deterministic cuBLAS algos. Tightens seed-to-seed noise "
                         "(2026-04-29 result: A+E det single-goal 3.31 +/- 0.74 vs "
                         "non-det 7.28 +/- 1.76 multi-goal). ~10-30% slowdown. "
                         "Note: this flag is read at module-import time (top of file), "
                         "not parsed here — argparse just suppresses 'unrecognized arg'.")
    ap.add_argument("--rpe-scaled-reward", action="store_true",
                    help="Scale reward by prediction error: delivered = reward + alpha * (reward - reward_ema). Surprise gets amplified.")
    ap.add_argument("--rpe-dopamine", action="store_true",
                    help="N9 (actor-critic): the dopamine signal IS the reward-prediction-error delta = r - V (V = reward_ema, the learned Rescorla-Wagner critic), not raw reward. Converts actor-only/scalar-DA -> actor-critic RPE (Schultz 1998; catalog C.22/C.28/C.30). Combine with --perceived-approach-reward (N5) for a fully coordinate-free RPE loop.")
    ap.add_argument("--rpe-alpha", type=float, default=1.0)
    # ── Spiking-SNc actor-critic Stage A (2026-06-08) ──────────────────────
    ap.add_argument("--spiking-snc", action="store_true",
                    help="Stage A (spiking SNc): the dopamine reward-prediction "
                         "error is computed by the FIRING of the spiking `snc` "
                         "pool (IZH2007_DOPAMINE), NOT a host formula. Each "
                         "reward step the snc pool is driven by "
                         "I_snc = tonic + k_r*max(0,r) - k_v*V so its windowed "
                         "rate encodes delta = r - V (burst on +RPE, dip on "
                         "-RPE); the DA broadcast is produced from that firing "
                         "via from_region_firing_signed (the one protected "
                         "sim/ edit). V = host reward_ema (Stage-A SCAFFOLD; "
                         "Stage B's neural striosome critic is separate). "
                         "SUPERSEDES --rpe-dopamine; owns the `dopamine` "
                         "modulator vs --enable-tonic-da; mutually exclusive "
                         "with --enable-compartmentalized-da. See "
                         "docs/plans/2026-06-08-spiking-snc-actor-critic-design.md.")
    ap.add_argument("--snc-tonic-pa", type=float, default=220.0,
                    help="Spiking SNc tonic pacemaker drive (pA) holding the "
                         "pool at its spontaneous rate so there is headroom to "
                         "DIP on negative RPE. Grace & Bunney 1984; "
                         "enums.py:665. Default 220.")
    ap.add_argument("--snc-reward-gain", type=float, default=400.0,
                    help="Spiking SNc excitatory reward afferent gain k_r (pA "
                         "per unit r). Reward above zero depolarizes SNc -> "
                         "burst. Default 400.")
    ap.add_argument("--snc-value-gain", type=float, default=400.0,
                    help="Spiking SNc inhibitory value (striosome) drive gain "
                         "k_v (pA per unit V). Prediction suppresses the DA "
                         "burst (expected reward elicits no burst). Default 400.")
    ap.add_argument("--snc-da-sensitivity", type=float, default=8.0,
                    help="from_region_firing_signed sensitivity: how strongly "
                         "the SNc firing-rate deviation from tonic maps to the "
                         "dopamine concentration deviation from baseline. "
                         "Default 8.")
    ap.add_argument("--enable-neural-critic", action="store_true",
                    help="Stage B (NEURAL value critic): replace the host "
                         "_V_scaffold (reward_ema) value with a spiking "
                         "striosome_value critic. AFFERENT (re-pointed 2026-06-08 "
                         "redesign): the PLACE code `sensor_place_readout` (the "
                         "dorsal/where hippocampal place stream, requires "
                         "--enable-place-goal-readout) — NOT the old ventral "
                         "`cortex_it` (which was position-INVARIANT + inactive in "
                         "nav). Its sensor_place_readout->striosome_value afferent "
                         "is PLASTIC and trained by the SAME SNc-derived dopamine "
                         "delta the actor uses (so it learns a value-of-LOCATION), "
                         "and it SUBTRACTS V at the SNc membrane through the slow "
                         "GABA_B/GIRK K+ conductance (E_K=-90mV, prop=0.02) — the "
                         "brain does the r-V subtraction, not host arithmetic "
                         "(BRAIN-BASED-ONLY completion of Stage B). Only "
                         "meaningful with --spiking-snc; the host _V_scaffold "
                         "term is dropped when set. Validated CPU de-risk: "
                         "research/findings/2026-06-08-gabab-girk-stageB-derisk-GO.md "
                         "+ place-code de-risk d0416fc3.")
    ap.add_argument("--spiking-reward-us", action="store_true",
                    help="SPIKING reward delivery: a PPN-like `reward_us` population (excitatory "
                         "US->VTA afferent, catalog C.33) FIRES the SNc reward burst instead of the "
                         "host snc_reward_gain*max(0,reward) write -> the whole δ=r−V is synaptic "
                         "(r=reward_us excitation, V=critic GABA_B). reward_us is driven by the "
                         "PERCEIVED reward (use --perceived-approach-reward so it's coord-free). "
                         "Requires --spiking-snc --enable-neural-critic. Default OFF=byte-equivalent.")
    ap.add_argument("--n-reward-us", type=int, default=40)
    ap.add_argument("--reward-us-to-snc-weight", type=float, default=50.0)
    ap.add_argument("--reward-us-to-snc-density", type=float, default=0.6)
    ap.add_argument("--reward-us-drive-pa", type=float, default=250.0,
                    help="US-afferent drive current onto reward_us when the perceived reward is +1.")
    ap.add_argument("--critic-window", action="store_true",
                    help="Stage 2 (windowed GABA_B): gate the "
                         "striosome_value->snc GABA_B current to a bounded LEAD "
                         "window into each reward evaluation (transmission_gate "
                         "'critic_snc_window'), CLOSED otherwise. Per the "
                         "place-code de-risk (d0416fc3) the slow GABA_B must "
                         "pre-build ~1 tau (~100-150 ms) BEFORE reward and must "
                         "NOT integrate across the whole dwell (>=200 ms "
                         "over-suppresses: the far-V also gets canceled / the SNc "
                         "flatlines). With this OFF (default, Stage 1) the gate is "
                         "held OPEN continuously (no windowing) to first isolate "
                         "'does it run + learn'. Requires --enable-neural-critic.")
    ap.add_argument("--critic-lead-steps", type=int, default=120,
                    help="When --critic-window: the bounded OPEN-window length in "
                         "steps (dt=1 ms => ms). The critic->SNc GABA_B gate is "
                         "OPEN for at most this many consecutive steps leading "
                         "into a reward, CLOSED otherwise so g_gabab decays and "
                         "cannot integrate across a long dwell. ~1 GABA_B tau "
                         "(150 ms); the de-risk sweet spot was 100-150. Default "
                         "120.")
    ap.add_argument("--critic-gabab-propagation", type=float, default=0.0,
                    help="GABA_B(GIRK) propagation strength onto the SNc (default 0.02). Lower it "
                         "(e.g. 0.006-0.010) to de-saturate the SNc when the critic fires hard "
                         "(strong-place-field seeds, ~120 Hz, over-clamp BOTH near+far to 0). 0=keep 0.02.")
    ap.add_argument("--critic-gabab-max", type=float, default=0.0,
                    help="Brain-based GIRK saturation cap on g_gabab (finite channels) so a HOT critic "
                         "can't over-accumulate it and fully clamp the SNc -> GRADED online δ at any "
                         "critic rate (the nav-A/B honest-negative fix; rate-robust unlike a fixed prop). "
                         "Try ~20-30. 0=no cap (default, byte-identical).")
    ap.add_argument("--critic-neuron-type", type=str, default=None,
                    help="Override the striosome_value critic's Izhikevich type "
                         "(2026-06-08 calibration). Default None keeps the MSN-D1 "
                         "preset; but the smoke showed the MSN-D1's depolarized "
                         "rheobase (~700 pA) can't be reached by the SPARSE "
                         "sensor_place_readout place code (~3-8 Hz) through the "
                         "afferent at ANY weight, so the critic stayed silent. "
                         "Pass IZH2007_RS_CORTICAL_PYRAMIDAL: a more excitable type "
                         "that DOES fire from the place code (5-20 Hz, graded by "
                         "drive x weight) and carves a teacher-free value-of-"
                         "location (research/findings/raw/g11_bg/_placecritic_diag*).")
    ap.add_argument("--critic-afferent-weight", type=float, default=3.0,
                    help="Init weight of the sensor_place_readout->striosome_value "
                         "plastic afferent (the value-critic input). Default 3.0. "
                         "Raise (~12-15) so the learned value-of-location is well-"
                         "graded once the place code fires an excitable critic "
                         "(--critic-neuron-type). stdp_w_max headroom is ensured in "
                         "run_g11. NOTE a reward-window TEACHER current was tried "
                         "and REJECTED: it drives post>>pre so STDP sees post-before-"
                         "pre LTD and the weight COLLAPSES (diag6); the place-driven "
                         "excitable critic gives clean LTP teacher-free.")
    ap.add_argument("--enable-critic-homeostasis", action="store_true",
                    help="2026-06-09 VALIDATED redesign (navfaithful-afferent-critic-"
                         "homeostasis PASS 3/3): add a DEDICATED DENSE `vs_place_context` "
                         "afferent (grid-32 Gaussian place code, 30-80 cells/location, "
                         "drive-injected each nav step) feeding ONLY the critic, AND set "
                         "per-region homeostasis (committed sim/ edit 89b8d909) on BOTH "
                         "vs_place_context AND the MSN-D1 striosome_value critic (GLOBAL "
                         "cfg.enable_homeostasis stays OFF -> deterministic regime "
                         "preserved). The de-risk fired the critic ~1.3-1.5 Hz, kept the "
                         "place code sharply graded (~59 Hz near vs 0 Hz far — no place-"
                         "blindness), and opened the GABA_B value subtraction. SUPERSEDES "
                         "the 2026-06-08 RS-critic-type calibration: with this on, leave "
                         "--critic-neuron-type unset (homeostasis fires the MSN-D1, not a "
                         "type swap). Requires --enable-neural-critic --spiking-snc.")
    ap.add_argument("--n-vs-place-context", type=int, default=200,
                    help="Size of the dense dedicated value-critic place afferent "
                         "(--enable-critic-homeostasis). Default 200 (the de-risk value).")
    ap.add_argument("--vs-place-to-value-weight", type=float, default=0.2,
                    help="INIT weight of the vs_place_context->striosome_value plastic "
                         "afferent (--enable-critic-homeostasis). Default 0.2 (the de-risk PASS "
                         "value); STDP grows the location-selective value UP from here (the "
                         "de-risk grew w_near 0.20->0.58). NOT a large init — the critic LEARNS V.")
    ap.add_argument("--vs-place-to-value-density", type=float, default=0.5,
                    help="Density of the vs_place_context->striosome_value afferent "
                         "(--enable-critic-homeostasis). Default 0.5 (de-risk value).")
    ap.add_argument("--enable-convergent-upstate", action="store_true",
                    help="CONVERGENT-EXCITATION UP-STATE (2026-06-09, homeostasis-free critic firing; "
                         "design Option A). Adds a DISTINCT dense NON-plastic `vs_place_drive` afferent "
                         "(the B.02 up-state arm) alongside `vs_place_context` (the plastic learner); "
                         "the runner injects the SAME place code into BOTH each nav step. CuPy 3-seed "
                         "de-risk: FIRE/LEARNS/ACTOR pass but PLACE-GRADED FAILS (the dense up-state is "
                         "position-blind) -> HONEST NEGATIVE; opt-in only, NOT in any flagship config. "
                         "See research/findings/2026-06-09-N9-convergent-upstate-derisk.md.")
    ap.add_argument("--vs-place-drive-to-value-weight", type=float, default=28.0,
                    help="A1 (vs_place_drive->striosome_value) dense NON-plastic up-state weight "
                         "(many weak synapses summing past the ~339 pA rheobase). De-risk: ~28 fires "
                         "a corner goal >=5 Hz. Only with --enable-convergent-upstate.")
    ap.add_argument("--vs-place-drive-to-value-density", type=float, default=0.8,
                    help="A1 up-state afferent density (dense convergence, not one giant synapse). "
                         "Default 0.8. Only with --enable-convergent-upstate.")
    # ── N9 NEURAL PLACE-CODE SELF-ORG (2026-06-09 nav deployment of the validated de-risk) ──
    ap.add_argument("--neural-place-selforg", action="store_true",
                    help="N9 NEURAL place-code self-org: REPLACE the host-Gaussian vs_place_context "
                         "with a SELF-ORGANIZED spiking place code (place_sensors egocentric render "
                         "-> `place` pool + FS-PING `place_fs` -> striosome_value coincidence critic; "
                         "the de-risk n9_place_graded_critic_stage2_derisk ported to nav). The place "
                         "fields self-organize in a STEP-1 warm-up; the host place injection is NOT "
                         "used. HARD-GATES --enable-convergent-upstate OFF. BRAIN-BASED-ONLY: (x,y) "
                         "enters only via the egocentric landmark render. Requires --enable-neural-critic.")
    ap.add_argument("--deterministic-selforg", action="store_true",
                    help="Toggle cfg.deterministic_transpose_matvec (default-off, byte-identity-proven "
                         "sim/ flag) ON during the STEP-1 place self-org so the SAME seed draws the SAME "
                         "place code (fixes the cusparse transpose-SpMV non-determinism; anti-cheat R-A). "
                         "Restored after self-org. Requires --neural-place-selforg.")
    ap.add_argument("--n-place", type=int, default=200, help="N9 self-org place pool size.")
    ap.add_argument("--n-place-fs", type=int, default=24, help="N9 FS-PING interneuron pool size.")
    ap.add_argument("--enable-critic-fs-inhibition", action="store_true",
                    help="SPIKING root fix for the over-firing (~125 Hz) value critic: add a "
                         "place_fs -> striosome_value GABA_A feedforward-inhibition pathway (the FS-PING "
                         "pool, which scales with the volley size, divisively clamps the MSN critic into "
                         "a physiological rate band across draws) instead of masking the over-clamp "
                         "downstream with the GIRK cap. Grading of V comes from the WEIGHTED plateau.")
    ap.add_argument("--critic-fs-weight", type=float, default=16.0,
                    help="place_fs -> striosome_value inhibitory weight (tune in the de-risk).")
    ap.add_argument("--critic-fs-density", type=float, default=0.6)
    ap.add_argument("--place-sensors-to-place-weight", type=float, default=28.0)
    ap.add_argument("--place-sensors-to-place-density", type=float, default=0.5)
    ap.add_argument("--place-sensors-to-place-jitter", type=float, default=0.6)
    ap.add_argument("--place-fs-weight", type=float, default=16.0)
    ap.add_argument("--place-fs-density", type=float, default=0.4)
    ap.add_argument("--fs-to-place-weight", type=float, default=8.0)
    ap.add_argument("--fs-to-place-density", type=float, default=0.4)
    ap.add_argument("--coincidence-threshold", type=int, default=12,
                    help="N9 Route-D READOUT K (weight units; the weighted plateau threshold).")
    ap.add_argument("--coincidence-train-k", type=float, default=4.0,
                    help="N9 Route-D TRAIN count K (count units; bootstraps the post-spike). MUST be >1.")
    ap.add_argument("--coincidence-plateau", type=float, default=80.0)
    ap.add_argument("--n-place-bearing", type=int, default=12,
                    help="N9 egocentric bearing sensors per landmark.")
    ap.add_argument("--n-place-dist", type=int, default=8,
                    help="N9 egocentric distance sensors per landmark.")
    ap.add_argument("--selforg-steps", type=int, default=2000,
                    help="N9 STEP-1 total self-org sweep steps.")
    ap.add_argument("--selforg-n-positions", type=int, default=40,
                    help="N9 STEP-1 number of agent positions swept.")
    ap.add_argument("--reward-delay-steps", type=int, default=8,
                    help="N9 online: hold `place` active this many steps before the SNc burst (Yagishita pairing-then-DA).")
    ap.add_argument("--stage-a-smoke", action="store_true",
                    help="N9 Stage-A cheap-first probe: after self-org, measure FIRE / PLACE-GRADED / "
                         "ACTOR-NOT-PERTURBED and exit BEFORE the nav loop. Requires --neural-place-selforg.")
    # ── N9 Phase 3: STEP-2 pair-then-reward value-training + Stage-B smoke ──
    ap.add_argument("--value-train-trials", type=int, default=0,
                    help="N9 STEP-2 (Phase 3): pair-then-reward value-training trials per scheduled goal "
                         "on the FROZEN self-org place fields (de-risk --pair-then-reward, ~40). Each trial: "
                         "ITI floor + zero eligibility -> PAIR (place + SNc tonic, --value-train-pair-steps) "
                         "-> REWARD (place + SNc burst after --reward-delay-steps) -> reset GABA_B/SNc. Grows "
                         "place->striosome_value V via DA-gated STDP, then freezes it. Default 0 = OFF "
                         "(byte-equivalent). Requires --neural-place-selforg + --enable-neural-critic.")
    ap.add_argument("--value-train-pair-steps", type=int, default=100,
                    help="N9 STEP-2 PAIR-phase length (steps; de-risk pair_steps=100; >= the up-state warm-up "
                         "so the bistable MSN-D1 reaches its up-state + lays a silent eligibility trace).")
    ap.add_argument("--value-train-hold-steps", type=int, default=40,
                    help="N9 STEP-2 ITI / REWARD sub-phase length (steps; de-risk hold_steps=40).")
    ap.add_argument("--critic-teacher-pa", type=float, default=300.0,
                    help="N9 STEP-2 sub-threshold phase-locked TEACHER current (pA) on striosome_value during "
                         "the PAIR phase ONLY (de-risk --critic-teacher-pa 300); removed at REWARD + read-out. "
                         "Draws the weak-drive place volley into firing phase-locked so net LTP forms. 0=off.")
    ap.add_argument("--value-train-stdp-w-max", type=float, default=0.0,
                    help="N9 STEP-2 critic soft-bound ceiling DURING value-train ONLY (de-risk validated 40). "
                         "The nav's global stdp_w_max=150 (sized for the actor cortex->D1 ~125) over-grows the "
                         "place->value weight to w_near~90 -> critic saturates (GRADE inverts) + GABA_B over-clamps "
                         "the SNc. 40 keeps w_near in the graded ~3-6 range. The actor is quiescent during value-"
                         "train so its weights are untouched; restored to the nav ceiling after. 0=no override.")
    ap.add_argument("--stage-b-smoke", action="store_true",
                    help="N9 Stage-B probe (the load-bearing critic gate): after STEP-1 self-org + STEP-2 "
                         "value-training, measure LEARNS-V / CRITIC FIRE+GRADE (weighted plateau) / GABA_B "
                         "gap (delta=r-V) + lesion control, and exit BEFORE the nav loop. Requires "
                         "--neural-place-selforg (+ --value-train-trials>0 for a trained V).")
    ap.add_argument("--critic-warmup-trials", type=int, default=0,
                    help="CRITIC VALUE-ACQUISITION WARM-UP (2026-06-09 deadlock-breaker). "
                         "Before the nav loop, run N de-risk-style reward-paired drives "
                         "(ITI floor -> clear eligibility -> place-code-at-goal + SNc reward "
                         "burst) at the scheduled goal location(s), at the BASELINE "
                         "homeostasis rate. The 1800-step nav leaves the MSN-D1 critic SILENT "
                         "(weight frozen at init): a LTP-bootstrap deadlock — the critic must "
                         "fire to seed STDP, but at the init weight it can't fire from a "
                         "free-moving agent's brief per-location visits, and faster homeostasis "
                         "(option 1) goes place-blind. This warm-up seeds the LTP (validated in "
                         "isolation: 20 trials -> V_goal > V_far, place-graded). All neural "
                         "(afferent->critic->SNc->DA->three-factor LTP); only agent placement + "
                         "reward delivery is environment/body scaffolding. Default 0 = OFF "
                         "(byte-equivalent). Requires --enable-neural-critic. Try 20-40.")
    ap.add_argument("--critic-warmup-hold-steps", type=int, default=40,
                    help="Steps per warm-up sub-phase (ITI floor / LEARN). Default 40 "
                         "(the de-risk's hold_steps).")
    ap.add_argument("--no-critic-warmup-all-goals", action="store_true",
                    help="Warm up the critic ONLY at the first scheduled goal (default warms "
                         "at EVERY distinct scheduled goal so V is graded for each multi-goal "
                         "epoch).")
    ap.add_argument("--hippocampus-drive-sigma", type=float, default=None,
                    help="Override the place/goal Gaussian drive sigma (cells per "
                         "bump). Default None keeps 0.5 (tuned for the 8x8 grid). "
                         "At grid_size=32 the 0.5 code is far narrower than the "
                         "~4.43 cell spacing -> near-silent at most positions; a "
                         "modest widening (1.5-2.5) makes it a real population bump "
                         "so a value-of-LOCATION can be carved by the neural critic "
                         "(diag6). Affects the actor's place/goal drive too, so "
                         "validate nav when changing it.")
    ap.add_argument("--hippocampus-drive-max-pa", type=float, default=None,
                    help="Override the place/goal Gaussian drive PEAK pA. Default "
                         "None keeps 600. The deterministic nav (OU off) place code "
                         "fires <1 Hz at 600 pA — too sparse to drive ANY striatal "
                         "value critic (diag4/7/8: even an RS critic needs ~1200+ pA "
                         "to fire from place). Raise (~1200-1500) to fire the critic, "
                         "BUT this doubles the actor's place+goal drive too, so "
                         "validate nav. (Pairs with --critic-neuron-type + "
                         "--critic-afferent-weight + --hippocampus-drive-sigma.)")
    ap.add_argument("--surprise-lr-boost", action="store_true",
                    help="Boost reward_learning_rate when |RPE| is high (NE-like fast meta-modulation)")
    ap.add_argument("--surprise-lr-alpha", type=float, default=2.0)
    ap.add_argument("--curriculum", action="store_true",
                    help="Curriculum learning: suppress hippocampus drive for first N steps (cortex→D1 builds without hippo noise), then enable. Requires --hippocampus.")
    ap.add_argument("--curriculum-warmup-steps", type=int, default=600,
                    help="Steps to keep hippo silent at start of curriculum (default 600).")
    ap.add_argument("--curriculum-ramp-steps", type=int, default=0,
                    help="Smooth gate ramp window centered on warmup boundary (default 0 = abrupt step). Biologically grounded: critical periods close gradually via PV maturation.")
    ap.add_argument("--curriculum-phase2-cortex-gain", type=float, default=0.0,
                    help="Phase 2 plasticity gain for cortex→D1 (default 0.0 = full freeze). Biologically: cortical plasticity slows but doesn't fully halt.")
    ap.add_argument("--curriculum-phase2-hippo-gain", type=float, default=1.0,
                    help="Phase 2 plasticity gain for hippo→cortex (default 1.0 = full plasticity).")
    ap.add_argument("--heuristic-strength", type=float, default=1.0,
                    help="Heuristic cortex drive strength multiplier (default 1.0). 0.0 disables heuristic.")
    ap.add_argument("--hidden-goal", action="store_true",
                    help="Hidden-goal (Morris-water-maze analogue) diagnostic: the goal's coordinates are NOT fed "
                         "into the brain (ppc_goal_input goal drive zeroed; the own-position place drive stays). "
                         "Use with --heuristic-strength 0 + no cue-reflex/SC-orienting so the agent must learn the "
                         "goal location from the scalar reward alone (the limbic core must be load-bearing).")
    ap.add_argument("--lesion-reward", action="store_true",
                    help="Reward lesion (load-bearing anti-cheat): force reward=0 every step so no learning signal "
                         "reaches dopamine / value critic / corticostriatal plasticity. Must collapse the nav score "
                         "if the reward is behaviorally load-bearing.")
    ap.add_argument("--heuristic-decay-after-step", type=int, default=-1,
                    help="Step after which heuristic_strength changes to --post-curriculum-heuristic-strength (default -1 = no decay).")
    ap.add_argument("--post-curriculum-heuristic-strength", type=float, default=0.0,
                    help="Heuristic strength after decay step (default 0.0 = full off).")
    ap.add_argument("--heuristic-wean-start", type=int, default=-1,
                    help="N1 critical-period scaffold: step at which the heuristic begins linearly weaning to 0 "
                         "(default -1 = no wean). Takes precedence over --heuristic-decay-after-step. The heuristic "
                         "teaches IT->cortex during [0, wean-start], then fades over --heuristic-wean-steps, then is off.")
    ap.add_argument("--heuristic-wean-steps", type=int, default=1500,
                    help="N1 critical-period scaffold: number of steps over which the heuristic linearly ramps from "
                         "full strength to 0 (default 1500). Active for both --heuristic-wean-start and "
                         "--heuristic-wean-adaptive (ramps from the adaptive commit step).")
    ap.add_argument("--heuristic-wean-adaptive", action="store_true",
                    help="N1 ADAPTIVE / activity-gated weaning: instead of a fixed critical-period clock, PROBE "
                         "readiness online — every --wean-probe-every steps silence the heuristic for "
                         "--wean-probe-window steps and measure mean distance to goal; if <= --wean-probe-threshold "
                         "(learned mapping is self-sufficient) COMMIT the wean (ramp to 0 over --heuristic-wean-steps), "
                         "else keep teaching. Robust across seeds where a fixed clock is not. Takes precedence over "
                         "--heuristic-wean-start.")
    ap.add_argument("--wean-probe-every", type=int, default=500,
                    help="Adaptive weaning: probe readiness every N steps (default 500).")
    ap.add_argument("--wean-probe-window", type=int, default=200,
                    help="Adaptive weaning: length (steps) of each OFF readiness-probe window (default 200).")
    ap.add_argument("--wean-probe-threshold", type=float, default=2.5,
                    help="Adaptive weaning: commit the wean when the probe-window mean distance to goal is "
                         "<= this value (default 2.5 — the learned mapping navigates without the heuristic).")
    ap.add_argument("--sleep-replay-after-step", type=int, default=-1,
                    help="Step at which to enter sleep-replay phase (default -1 = no sleep). During sleep, hippo replays random place/goal patterns, corticostriatal thaws for consolidation.")
    ap.add_argument("--sleep-replay-steps", type=int, default=300,
                    help="Number of steps in sleep-replay phase.")
    ap.add_argument("--sleep-replay-rate-hz", type=float, default=200.0,
                    help="Replay drive rate (Hz) — biologically: sharp-wave ripples ~150-250Hz.")
    ap.add_argument("--sleep-nrem-rem-alternate", action="store_true",
                    help="Alternate between NREM (trajectory replay, first half) and REM (random replay, second half) during sleep.")
    ap.add_argument("--enable-reverse-replay", action="store_true",
                    help="Reverse-order trajectory replay during NREM "
                         "(Foster & Wilson 2006). Replays successful "
                         "trajectories newest-to-oldest by sleep step index, "
                         "modeling TD-style backward credit assignment via "
                         "sharp-wave ripples. Composes with "
                         "--enable-cluster-d-hippocampus and "
                         "--enable-cluster-d-v2-swr.")
    ap.add_argument("--enable-her", action="store_true",
                    help="Hindsight Experience Replay (Andrychowicz 2017): "
                         "log (old_pos, current_pos) tuples to "
                         "successful_trajectories every 50 steps, treating "
                         "the achieved position as if it had been the goal. "
                         "Provides hindsight credit assignment for sparse-goal "
                         "generalization. Composes with sleep replay; the "
                         "expanded buffer feeds the existing replay drive.")
    ap.add_argument("--heuristic-single-pool", action="store_true",
                    help="Probe flag: heuristic drives ONE cortex pool "
                         "(replicated-style) instead of all manhattan-reducing "
                         "directions. Investigating cross-runner discrepancy.")
    ap.add_argument("--enable-recency-weighted-replay", action="store_true",
                    help="Recency-weighted sleep replay sampling: bias toward "
                         "newest successful_trajectories with exponential "
                         "weighting (tau = n_traj/3). Addresses the "
                         "stale-replay bottleneck for multi-goal tasks where "
                         "older entries are from goals that no longer apply. "
                         "Mutually exclusive with --enable-reverse-replay.")
    ap.add_argument("--goal-silence-after-step", type=int, default=-1,
                    help="PFC Stage 2 delayed-response test: silence goal_cells AND heuristic at this step. PFC working memory should maintain goal info.")
    ap.add_argument("--goal-silence-duration", type=int, default=0,
                    help="How long to keep goal_cells/heuristic silenced.")
    # Webapp discovery: when this runner is launched directly via the
    # terminal, the dashboard's Live picker discovers the run via the
    # sidecar + redirected stdout. ON by default since 2026-04-30; pass
    # --no-emit-webapp-sidecar to opt out (e.g. headless eval batches
    # that don't want webapp/runtime/ files to accumulate).
    ap.add_argument("--emit-webapp-sidecar", action="store_true", default=True,
                    help="(default ON 2026-04-30) Redirect stdout to "
                         "webapp/runtime/run_<id>.log and write a sidecar "
                         "so the dashboard's Live picker discovers + can "
                         "attach to this run.")
    ap.add_argument("--no-emit-webapp-sidecar", action="store_false",
                    dest="emit_webapp_sidecar",
                    help="Disable webapp sidecar emission. Use for headless "
                         "eval batches where the webapp/runtime/ log files "
                         "would just accumulate without ever being viewed.")
    args = ap.parse_args()

    if args.emit_webapp_sidecar:
        _emit_webapp_sidecar_and_redirect_stdout(args)

    if args.moving_goal:
        out_path = args.out or f"research/findings/raw/g11_bg/g11_seed{args.seed}.json"
        # Scale goal positions to grid size — keeps relative spacing the same
        # so the same task structure works at any grid scale. Defaults are
        # ~75% and ~12% of grid extent (matches the 8×8 (6,6) and (1,6)).
        gs = args.grid_size
        far = (max(0, gs - 2), max(0, gs - 2))            # was (6, 6)
        far_west = (max(0, 1), max(0, gs - 2))            # was (1, 6)
        sw = (max(0, 1), max(0, 1))                        # was (1, 1)
        far_se = (max(0, gs - 2), max(0, 1))              # was (6, 1)
        if args.goal_schedule == "multi":
            goal_schedule = [(0, far), (450, far_west), (900, sw), (1350, far_se)]
        elif args.goal_schedule == "single":
            # 2026-06-06 perceptual-bootstrap gauge: ONE fixed goal for the
            # whole episode. Use with a long --n-steps so a zero-init
            # perception→action pathway (e.g. visual-cortex IT→cortex_X) has
            # time + many goal-reaches to bootstrap from reward + exploration
            # WITHOUT the coordinate heuristic. The cleanest "can perception
            # learn to navigate at all?" test (no goal-change generalization).
            goal_schedule = [(0, far)]
        elif args.goal_schedule == "generalize":
            # Rank 2 generalization test (2026-06-08): train on ONE goal through
            # the reflex wean (0-3000 on `far`; reflex teaches + weans @2000-3000),
            # then change to THREE NEW goals AFTER the wean (3000/4000/5000) with
            # the reflex teacher OFF. Tests whether the LEARNED goal-agnostic
            # (dx,dy)->action map (position-preserving, image-sourced) navigates to
            # goals it was NEVER taught on. Use with --n-steps 6000 +
            # --sc-reflex-wean-start 2000 --sc-reflex-wean-steps 1000.
            goal_schedule = [(0, far), (3000, far_west), (4000, sw), (5000, far_se)]
        elif args.goal_schedule == "generalize2":
            # Rank 2 generalization test v2 (2026-06-08): train (reflex teaches) on
            # ALL FOUR CORNERS rotating (0-3000, so the learned (dx,dy)->action map
            # covers the full offset/direction space — the fix for the single-goal
            # `generalize` failure where training on ONE goal only covered offsets
            # toward it), wean @2000-3000, then test on THREE NEW NON-CORNER goals
            # (3000/4000/5000) with the reflex OFF. Use --n-steps 6000 +
            # --sc-reflex-wean-start 2000 --sc-reflex-wean-steps 1000.
            mid_top = (max(0, gs // 2), max(0, gs - 2))
            mid_left = (max(0, 1), max(0, gs // 2))
            mid_right = (max(0, gs - 2), max(0, gs // 2))
            goal_schedule = [(0, far), (700, far_west), (1400, sw), (2100, far_se),
                             (3000, mid_top), (4000, mid_left), (5000, mid_right)]
        elif args.goal_schedule == "curriculum":
            flip = max(1200, args.curriculum_warmup_steps + 600)
            goal_schedule = [(0, far), (flip, far_west)]
        elif args.goal_schedule == "random":
            # Harder benchmark (2026-04-30): 4 phases × 450 steps, but goals
            # are sampled uniformly at random per phase (excluding start
            # position). NOTE empirically: random is actually EASIER than
            # the fixed-corner `multi` schedule because corner goals have
            # ~10 Manhattan from start (1,1) while random uniform averages
            # ~5.5. Kept for reference; not the harder benchmark.
            rng = np.random.default_rng(args.seed)
            goal_schedule = [(0, far)]
            for phase_start in (450, 900, 1350):
                while True:
                    gx = int(rng.integers(0, gs))
                    gy = int(rng.integers(0, gs))
                    if (gx, gy) != (1, 1) and (gx, gy) != goal_schedule[-1][1]:
                        break
                goal_schedule.append((phase_start, (gx, gy)))
        elif args.goal_schedule == "multi-fast":
            # Harder benchmark (2026-04-30): same 4 corner goals as multi,
            # but transitions every 225 steps instead of 450 — agent has
            # half the adaptation budget per phase. Total still 1800 steps
            # (8 phases of 225, cycling through the 4 corners twice).
            seq = [far, far_west, sw, far_se]
            goal_schedule = []
            for i in range(8):
                goal_schedule.append((i * 225, seq[i % 4]))
        elif args.goal_schedule == "random-far":
            # Harder benchmark (2026-04-30): random goals constrained to
            # be at least Manhattan-8 from the previous goal (or start
            # pos for phase 0). Forces long transitions like the corner
            # goals do, but with novel positions each phase.
            rng = np.random.default_rng(args.seed)
            prev = (1, 1)  # start pos for phase 0
            goal_schedule = []
            for phase_start in (0, 450, 900, 1350):
                attempts = 0
                while True:
                    attempts += 1
                    gx = int(rng.integers(0, gs))
                    gy = int(rng.integers(0, gs))
                    manhattan = abs(gx - prev[0]) + abs(gy - prev[1])
                    if manhattan >= 8 and (gx, gy) != prev:
                        break
                    if attempts > 1000:
                        gx, gy = (gs - 2, gs - 2)  # fallback
                        break
                goal_schedule.append((phase_start, (gx, gy)))
                prev = (gx, gy)
        else:
            goal_schedule = [(0, far), (300, far_west)]
        run_moving_goal_episode(
            out_path=out_path,
            seed=args.seed,
            n_steps=args.n_steps,
            grid_size=args.grid_size,
            n_hippocampus_per_layer=args.n_hippocampus_per_layer,
            # Place/goal drive sigma: keep the 0.5 default unless overridden
            # (2026-06-08 critic calibration — widen for a real population bump at
            # large grids so the neural value critic can carve a value-of-location).
            hippocampus_drive_sigma=(args.hippocampus_drive_sigma
                                     if args.hippocampus_drive_sigma is not None
                                     else 0.5),
            hippocampus_drive_max_pA=(args.hippocampus_drive_max_pa
                                      if args.hippocampus_drive_max_pa is not None
                                      else 600.0),
            sensory_to_cortex_weight=args.sensory_to_cortex_weight,
            hippocampus_to_cortex_weight=args.hippocampus_to_cortex_weight,
            enable_pfc=args.pfc,
            n_pfc=args.n_pfc,
            pfc_internal_density=args.pfc_internal_density,
            goal_to_pfc_weight=args.goal_to_pfc_weight,
            pfc_to_cortex_weight=args.pfc_to_cortex_weight,
            enable_pfc_nmda=args.enable_pfc_nmda,
            enable_beacon_perception=args.beacon_perception,
            n_beacon_sensors=args.n_beacon_sensors,
            beacon_to_goal_weight=args.beacon_to_goal_weight,
            beacon_max_intensity=args.beacon_max_intensity,
            beacon_falloff=args.beacon_falloff,
            beacon_replaces_goal=args.beacon_replaces_goal,
            enable_cue_reflex=args.cue_reflex,
            cue_reflex_strength=args.cue_reflex_strength,
            cue_reflex_replaces_heuristic=args.cue_reflex_replaces_heuristic,
            sc_orienting_reflex=args.sc_orienting_reflex,
            sc_reflex_strength=args.sc_reflex_strength,
            learned_perception_from_vision=args.learned_perception_from_vision,
            sc_reflex_wean_start=args.sc_reflex_wean_start,
            sc_reflex_wean_steps=args.sc_reflex_wean_steps,
            sensory_cortex_teacher_pA=args.sensory_cortex_teacher_pA,
            enable_landmarks=args.enable_landmark_sensor,
            n_landmark_sensors=args.n_landmark_sensors,
            landmark_to_place_weight=args.landmark_to_place_weight,
            landmark_position=(args.landmark_x, args.landmark_y) if args.landmark_x is not None and args.landmark_y is not None else None,
            landmark_max_intensity=args.landmark_max_intensity,
            landmark_falloff=args.landmark_falloff,
            landmarks_replace_place=args.landmarks_replace_place,
            enable_sensed_reward=args.sensed_reward,
            perceived_approach_reward=args.perceived_approach_reward,
            enable_bg_cross_projections=args.bg_cross_projections,
            cross_projection_weight=args.cross_projection_weight,
            cross_projection_density=args.cross_projection_density,
            cross_projection_topology_seed=args.cross_projection_topology_seed,
            bg_cross_thaw_step=args.bg_cross_thaw_step,
            bg_cross_phase3_gain=args.bg_cross_phase3_gain,
            enable_bg_lateral_inhibition=args.bg_lateral_inhibition,
            enable_developmental_pretraining=args.developmental_pretraining,
            pretraining_n_goals=args.pretraining_n_goals,
            pretraining_steps_per_goal=args.pretraining_steps_per_goal,
            enable_structural_pruning=args.enable_structural_pruning,
            enable_d1_d2_asymmetry=args.enable_d1_d2_asymmetry,
            enable_striatal_fsis=args.enable_striatal_fsis,
            enable_tans=args.enable_tans,
            enable_bg_neuropeptides=args.enable_bg_neuropeptides,
            enable_cluster_a_closed_loop=args.enable_cluster_a_closed_loop,
            enable_tonic_da=args.enable_tonic_da,
            enable_compartmentalized_da=args.enable_compartmentalized_da,
            enable_cluster_d_hippocampus=args.enable_cluster_d_hippocampus,
            enable_cluster_d_v2_swr=args.enable_cluster_d_v2_swr,
            enable_cluster_e_topography=args.enable_cluster_e_topography,
            enable_cluster_f_cerebellum=args.enable_cluster_f_cerebellum,
            enable_cluster_f_v2=args.enable_cluster_f_v2,
            n_granule=args.n_granule,
            enable_visual_cortex=args.enable_visual_cortex,
            enable_spiking_sc=args.enable_spiking_sc,
            enable_spiking_sc_approach=args.enable_spiking_sc_approach,
            visual_cortex_action_warmup_steps=args.visual_cortex_action_warmup_steps,
            visual_v1_weight_scale=args.visual_v1_weight_scale,
            visual_image_size=args.visual_image_size,
            enable_text_io=args.enable_text_io,
            embodied_language=args.embodied_language,
            embodied_language_drive_pA=args.embodied_language_drive_pA,
            embodied_language_goal_radius=args.embodied_language_goal_radius,
            embodied_language_every_n_steps=args.embodied_language_every_n_steps,
            embodied_language_warmup_steps=args.embodied_language_warmup_steps,
            cluster_e_distance_sigma=args.cluster_e_distance_sigma,
            pruning_alpha=args.pruning_alpha,
            pruning_threshold=args.pruning_threshold,
            pruning_weight_floor=args.pruning_weight_floor,
            lateral_inhibition_density=args.lateral_inhibition_density,
            lateral_inhibition_weight=args.lateral_inhibition_weight,
            interactive_control_file=args.interactive_control_file,
            progress_print_interval=args.progress_print_interval,
            trial_sleep_ms=args.trial_sleep_ms,
            emit_activity=args.emit_activity,
            emit_activity_every=args.emit_activity_every,
            goal_schedule=goal_schedule,
            enable_motor_lateral_inhibition=_warn_motor_lateral_inhibition_deprecated(args.motor_lateral_inhibition),
            enable_cortex_lateral_inhibition=args.cortex_wta,
            enable_per_action_da_targeting=args.per_action_da,
            enable_adaptive_per_action_da=args.adaptive_da,
            adaptive_da_ema_decay=args.adaptive_da_ema_decay,
            adaptive_da_ema_decay_negative=args.adaptive_da_ema_decay_negative,
            enable_learned_perception=args.learned_perception,
            enable_learned_perception_informed_init=args.informed_init,
            informed_init_alpha=args.informed_init_alpha,
            enable_hippocampus=args.hippocampus,
            enable_da_gated_wta=args.da_gated_wta,
            enable_rpe_scaled_reward=args.rpe_scaled_reward,
            rpe_dopamine=args.rpe_dopamine,
            rpe_scale_alpha=args.rpe_alpha,
            spiking_snc=args.spiking_snc,
            snc_tonic_pa=args.snc_tonic_pa,
            snc_reward_gain=args.snc_reward_gain,
            snc_value_gain=args.snc_value_gain,
            snc_da_sensitivity=args.snc_da_sensitivity,
            enable_neural_critic=args.enable_neural_critic,
            spiking_reward_us=args.spiking_reward_us,
            n_reward_us=args.n_reward_us,
            reward_us_to_snc_weight=args.reward_us_to_snc_weight,
            reward_us_to_snc_density=args.reward_us_to_snc_density,
            reward_us_drive_pa=args.reward_us_drive_pa,
            enable_critic_window=args.critic_window,
            critic_lead_steps=args.critic_lead_steps,
            critic_gabab_propagation=args.critic_gabab_propagation,
            critic_gabab_max=args.critic_gabab_max,
            critic_neuron_type=args.critic_neuron_type,
            critic_afferent_weight=args.critic_afferent_weight,
            enable_critic_homeostasis=args.enable_critic_homeostasis,
            n_vs_place_context=args.n_vs_place_context,
            vs_place_to_value_weight=args.vs_place_to_value_weight,
            vs_place_to_value_density=args.vs_place_to_value_density,
            enable_convergent_upstate=args.enable_convergent_upstate,
            vs_place_drive_to_value_weight=args.vs_place_drive_to_value_weight,
            vs_place_drive_to_value_density=args.vs_place_drive_to_value_density,
            # N9 neural place-code self-org (2026-06-09 nav deployment).
            neural_place_selforg=args.neural_place_selforg,
            deterministic_selforg=args.deterministic_selforg,
            enable_critic_fs_inhibition=args.enable_critic_fs_inhibition,
            critic_fs_weight=args.critic_fs_weight,
            critic_fs_density=args.critic_fs_density,
            n_place=args.n_place,
            n_place_fs=args.n_place_fs,
            place_sensors_to_place_weight=args.place_sensors_to_place_weight,
            place_sensors_to_place_density=args.place_sensors_to_place_density,
            place_sensors_to_place_jitter=args.place_sensors_to_place_jitter,
            place_fs_weight=args.place_fs_weight,
            place_fs_density=args.place_fs_density,
            fs_to_place_weight=args.fs_to_place_weight,
            fs_to_place_density=args.fs_to_place_density,
            coincidence_threshold=args.coincidence_threshold,
            coincidence_train_k=args.coincidence_train_k,
            coincidence_plateau=args.coincidence_plateau,
            n_place_bearing=args.n_place_bearing,
            n_place_dist=args.n_place_dist,
            selforg_steps=args.selforg_steps,
            selforg_n_positions=args.selforg_n_positions,
            reward_delay_steps=args.reward_delay_steps,
            stage_a_smoke=args.stage_a_smoke,
            value_train_trials=args.value_train_trials,
            value_train_pair_steps=args.value_train_pair_steps,
            value_train_hold_steps=args.value_train_hold_steps,
            critic_teacher_pa=args.critic_teacher_pa,
            value_train_stdp_w_max=args.value_train_stdp_w_max,
            stage_b_smoke=args.stage_b_smoke,
            critic_warmup_trials=args.critic_warmup_trials,
            critic_warmup_hold_steps=args.critic_warmup_hold_steps,
            critic_warmup_all_goals=not args.no_critic_warmup_all_goals,
            enable_surprise_lr_boost=args.surprise_lr_boost,
            surprise_lr_alpha=args.surprise_lr_alpha,
            enable_curriculum=args.curriculum,
            curriculum_warmup_steps=args.curriculum_warmup_steps,
            curriculum_ramp_steps=args.curriculum_ramp_steps,
            curriculum_phase2_cortex_gain=args.curriculum_phase2_cortex_gain,
            curriculum_phase2_hippo_gain=args.curriculum_phase2_hippo_gain,
            heuristic_strength=args.heuristic_strength,
            hidden_goal=args.hidden_goal,
            lesion_reward=args.lesion_reward,
            heuristic_decay_after_step=args.heuristic_decay_after_step,
            post_curriculum_heuristic_strength=args.post_curriculum_heuristic_strength,
            heuristic_wean_start=args.heuristic_wean_start,
            heuristic_wean_steps=args.heuristic_wean_steps,
            heuristic_wean_adaptive=args.heuristic_wean_adaptive,
            wean_probe_every=args.wean_probe_every,
            wean_probe_window=args.wean_probe_window,
            wean_probe_threshold=args.wean_probe_threshold,
            sleep_replay_after_step=args.sleep_replay_after_step,
            sleep_replay_steps=args.sleep_replay_steps,
            sleep_replay_rate_hz=args.sleep_replay_rate_hz,
            sleep_nrem_rem_alternate=args.sleep_nrem_rem_alternate,
            enable_reverse_replay=args.enable_reverse_replay,
            enable_her=args.enable_her,
            enable_recency_weighted_replay=args.enable_recency_weighted_replay,
            heuristic_single_pool=args.heuristic_single_pool,
            goal_silence_after_step=args.goal_silence_after_step,
            goal_silence_duration=args.goal_silence_duration,
            genuine_thal_disinhibition=args.genuine_thal_disinhibition,
            genuine_gpi_tonic_pA=args.genuine_gpi_tonic_pa,
            genuine_thal_tonic_pA=args.genuine_thal_tonic_pa,
            readout_source=args.readout_source,
            enable_thal_lateral_inhibition=args.enable_thal_lateral_inhibition,
            thal_to_fs_weight=args.thal_to_fs_weight,
            thal_fs_to_thal_weight=args.thal_fs_to_thal_weight,
            n_sel_per_action=args.n_sel_per_action,
            n_sel_fs_per_action=args.n_sel_fs_per_action,
            thal_to_sel_weight=args.thal_to_sel_weight,
            sel_to_sel_fs_weight=args.sel_to_sel_fs_weight,
            sel_fs_to_sel_weight=args.sel_fs_to_sel_weight,
            sel_recurrent_density=args.sel_recurrent_density,
            sel_recurrent_weight=args.sel_recurrent_weight,
            enable_commit_burst=args.enable_commit_burst,
            n_commit_per_action=args.n_commit_per_action,
            n_commit_opn=args.n_commit_opn,
            sel_to_commit_weight=args.sel_to_commit_weight,
            commit_recurrent_density=args.commit_recurrent_density,
            commit_recurrent_weight=args.commit_recurrent_weight,
            opn_to_commit_weight=args.opn_to_commit_weight,
            commit_opn_tonic_pA=args.commit_opn_tonic_pa,
            reset_accumulator_each_trial=args.reset_accumulator_each_trial,
            reset_losers_only=args.reset_losers_only,
            urgency_max_pA=args.urgency_max_pa,
            learning_rate=args.learning_rate,
            stdp_w_max_override=args.stdp_w_max,
        )
        return 0

    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    import cupy as cp

    print(f"\n{'='*72}")
    print(f"  G11 BG Action Selection Module -- Smoke Test")
    print(f"{'='*72}\n", flush=True)

    regions, pathways = build_bg_brain_regions()
    n_total = sum(r.n_neurons for r in regions)
    print(f"  Built {len(regions)} regions with {n_total} total neurons")
    print(f"  Built {len(pathways)} pathways")
    print()

    # Verify no name collisions
    names = [r.name for r in regions]
    assert len(set(names)) == len(names), "Region name collision!"

    cfg = CoreSimConfig()
    cfg.num_neurons = 0  # Set by region framework
    cfg.dt_ms = 1.0
    cfg.seed = int(args.seed)
    cfg.num_traits = 1  # Force single neuron type per region
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = False  # Smoke test: no plasticity
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False

    # Cluster C v2 (2026-04-29) smoke compatibility: register per-action DA
    # modulators if --enable-compartmentalized-da is set. Smoke run will
    # exercise the registration path; reward modulation is disabled so the
    # DA signal is not actually consumed but the array allocations and
    # registration are validated.
    if args.enable_compartmentalized_da:
        from sim.neuromodulators import _default_per_action_dopamine_config
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = list(cfg.neuromodulators) + [
            _default_per_action_dopamine_config(action, idx)
            for idx, action in enumerate(ACTION_NAMES)
        ]
        print(f"  Cluster C v2: registered {len(ACTION_NAMES)} per-action DA modulators "
              f"(dopamine_{{{','.join(ACTION_NAMES)}}})")

    print(f"  Initializing bridge...", flush=True)
    t0 = time.time()
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    elapsed = time.time() - t0
    print(f"  Bridge initialized in {elapsed:.1f}s", flush=True)
    print(f"  Total neurons: {cfg.num_neurons}")
    print(f"  Total synapses: {bridge.cp_connections.nnz}")

    if not args.smoke and not args.probe_action:
        return 0

    # Quick 30-step smoke run with no input — should show GPe/GPi tonic firing
    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0
    n_steps = 50
    n_motor_total = sum(r.n_neurons for r in regions if r.name.startswith("motor_"))

    spike_counts = np.zeros(cfg.num_neurons, dtype=np.int32)
    print(f"\n  Running {n_steps} steps with no input (rest dynamics)...", flush=True)
    for s in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        firing = bridge.cp_firing_states.get().astype(np.int32)
        spike_counts += firing

    # Per-region firing rate
    print(f"\n  Per-region firing rates (Hz over {n_steps}ms with no input):")
    for r in regions:
        idx = bridge.region_manager.indices(r.name)
        rate_hz = spike_counts[list(idx)].sum() / r.n_neurons / (n_steps * cfg.dt_ms / 1000.0)
        print(f"    {r.name:<24s} ({r.izh_neuron_type or 'default':<32s}): {rate_hz:.1f} Hz")

    print(f"\n  Smoke test PASSED -- {len(regions)} regions, "
          f"{bridge.cp_connections.nnz} synapses initialized cleanly.")

    # ---- Phase B.T4 / T5: action selection probe ----
    if args.probe_action:
        print(f"\n{'='*72}")
        print(f"  Action selection probe: drive cortex -> {args.probe_action} pathway")
        print(f"{'='*72}\n", flush=True)

        # Inject strong current into a SUBSET of cortex neurons. The cortex->D1/D2
        # weights are random — so the input pattern preferentially activates
        # whichever D1/D2 happens to have stronger weights from these inputs.
        # For a clean probe, manually override: inject ONLY into cortex neurons
        # whose hash maps to the target action.
        # Apply tonic baseline drive to BG output nuclei (mimics intrinsic
        # depolarizing conductance that makes real GPe/GPi/STN autonomously
        # fire 30-80 Hz). Without this, our Izh presets sit at rest because
        # Izh doesn't model intrinsic Ca pacemaker currents.
        bridge.cp_external_input_current[:] = 0.0
        # Per-region tonic drive levels:
        for region_name in [f"gpe_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(150.0)
        for region_name in [f"gpi_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                # Lower baseline for GPi → easier to silence by D1 inhibition
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(110.0)
        for region_name in ["stn", "snc"]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(150.0)
        # Thalamus baseline drive — set such that GPi inhibition (when active)
        # keeps thal silent, AND when GPi drops to 0 (D1 suppression),
        # thal fires actively.
        for region_name in [f"thal_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(300.0)

        # Drive ONLY the target action's cortex pool
        cortex_idx = list(bridge.region_manager.indices(f"cortex_{args.probe_action}"))
        cortex_cp = cp.asarray(cortex_idx, dtype=cp.int64)

        bridge.runtime_state.current_time_step = 0
        bridge.runtime_state.current_time_ms = 0.0

        drive_pA = 800.0
        n_probe_steps = 500
        target_cortex = cortex_idx
        spike_counts = np.zeros(cfg.num_neurons, dtype=np.int32)
        for s in range(n_probe_steps):
            bridge.cp_external_input_current[cortex_cp] = cp.float32(drive_pA)
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            firing = bridge.cp_firing_states.get().astype(np.int32)
            spike_counts += firing

        # Per-region firing rate
        print(f"  Driving {len(target_cortex)}/{len(cortex_idx)} cortex neurons "
              f"with {drive_pA} pA for {n_probe_steps}ms")
        print(f"\n  Per-region firing rates over {n_probe_steps}ms:")
        ordered_groups = [f"cortex_{a}" for a in ACTION_NAMES]
        for a in ACTION_NAMES:
            ordered_groups += [f"str_D1_{a}", f"str_D2_{a}", f"gpe_{a}",
                                f"gpi_{a}", f"thal_{a}", f"motor_{a}"]
        ordered_groups += ["stn", "snc"]
        for region_name in ordered_groups:
            r = next((reg for reg in regions if reg.name == region_name), None)
            if r is None:
                continue
            idx = bridge.region_manager.indices(r.name)
            if not idx:
                continue
            rate_hz = spike_counts[list(idx)].sum() / r.n_neurons / (n_probe_steps / 1000.0)
            marker = " <-" if (region_name.endswith(f"_{args.probe_action}") and
                              region_name.startswith(("str_D1_", "thal_", "motor_"))) else ""
            print(f"    {r.name:<15s} {rate_hz:>6.1f} Hz{marker}")

        # Quick check: did the right motor pop fire most?
        motor_rates = {}
        for a in ACTION_NAMES:
            idx = bridge.region_manager.indices(f"motor_{a}")
            n = len(idx)
            r = spike_counts[list(idx)].sum() / max(n, 1) / (n_probe_steps / 1000.0)
            motor_rates[a] = r
        winner = max(motor_rates, key=motor_rates.get)
        print(f"\n  Motor rates: {motor_rates}")
        print(f"  Winner: {winner}  (target: {args.probe_action})")
        if winner == args.probe_action and motor_rates[winner] > 5:
            print(f"  [OK] BG circuit selected the correct motor")
        else:
            print(f"  -> BG circuit did not produce a clean winner (rates may be too low/noisy)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
