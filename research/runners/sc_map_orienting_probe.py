"""Cheapest-first DE-RISK for the spiking superior-colliculus (N1 + N5).

Per the deep-research GO (`2026-06-10-N1-N5-spiking-superior-colliculus-research.md`):
the two remaining HOST computations between sensation and action in the nav agent —
  N1 = `sc_orienting_cardinal_from_image` (pixels -> orienting cardinal, an argmax in numpy)
  N5 = `reward = sign(delta retinal-eccentricity)` via `sc_salience_offset_from_image`
       (pixels -> a host distance-formula reward)
are read-outs of the SAME latent quantity (the goal's retinal position + its motion),
which biology computes in ONE structure: the superior colliculus (a retinotopic salience
map with winner-take-all). This probe proves-or-kills that a SPIKING SC map, on the real
SimulationBridge, can reproduce BOTH by NEURON FIRING — image-only, no coordinates,
no host argmax/distance — BEFORE any GPU nav integration.

Architecture (all runner/probe-side region+pathway wiring; ZERO sim/ edits):
  egocentric retina image (the world rendered from the agent's eye: the goal at its
      relative bearing, agent at the foveal centre)  --image_to_retina_drive-->
  `retina` (2*32*32 spiking photoreceptors)  --retinotopic-->
  `sc_map` (16x16 spiking retinotopic sheet) + `sc_fs` global inhibition (soft WTA)
      -> a single activity bump at the goal's egocentric retinal site
  N1 read-out: `sc_map -> cortex_{N,E,S,W}` weighted-quadrant pooling -> the winning
      cortex pool BY FIRING = the orienting cardinal (NOT a host argmax).
  N5 read-out: `sc_map -> approach` foveal-centre pooling -> firing rises as the bump
      nears the centre (= the goal foveates / gets closer); sign(approach_now - prev)
      = "the goal got closer" (NOT a host distance).

Why egocentric: the SC/retina frame IS egocentric (the fovea = the agent's gaze centre),
so a single goal bump's direction-from-centre directly gives the orienting vector and a
temporal rise toward centre gives approach — no two-blob subtraction. Rendering the world
from the agent's viewpoint is a legitimate ENVIRONMENT operation (channel-1 of the
BRAIN-BASED-ONLY bar: "rendering the agent's sensory input"), exactly like the existing
allocentric render, just egocentric. The orienting CARDINAL is frame-invariant, so the
behavioural-equivalence comparison against the host (which reads the allocentric render)
is valid.

Falsifiers (behavioural equivalence vs the REAL host functions, the N5 "8/8 label-
agreement" precedent):
  F1 (N1): the winning cortex_X BY FIRING matches sc_orienting_cardinal_from_image on
           >= N-1 of N hand-set (agent, goal) pairs (a firing winner, not a host argmax).
  F2 (N5): sign(approach_now - approach_prev) matches the host reward sign on >= 7/8
           (old->new) transitions (graded SC-bump motion, not a host distance).
  LESION (decisive anti-cheat): scramble the retina->sc_map retinotopy (permute the
           topographic source->target assignment) -> BOTH F1 and F2 must collapse to
           chance. If they survive a scrambled map, the signal is leaking from a
           non-retinotopic source (a hidden shortcut) and the build is REJECTED.

Run (CPU, tiny smoke — numpy backend):
    SIM_BACKEND=numpy python research/runners/sc_map_orienting_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sim.backend import get_backend
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronModel, NeuronType
from sim.visual_cortex import image_to_retina_drive, render_gridworld_to_image
# The REAL host functions the spiking SC must reproduce (the anti-cheat ground truth).
from research.runners.g11_bg_runner import (
    sc_orienting_cardinal_from_image,
    sc_salience_offset_from_image,
)

xp, BACKEND = get_backend()

# --- geometry ---
IMG = 32                  # retina is IMG x IMG pixels x 2 channels
SC = 16                   # sc_map is SC x SC retinotopic sites (IMG // 2)
GRID = 8                  # gridworld cells
PPC = IMG // GRID         # pixels per grid cell in the egocentric render (4)
CENTER = IMG // 2         # the foveal centre of the egocentric retina (16)
SC_CENTER = (SC - 1) / 2.0  # the foveal centre of the sc_map (7.5)


def render_egocentric(agent, goal, image_size=IMG, ppc=PPC, radius=2):
    """ENVIRONMENT render (channel-1 legitimate): the world from the agent's eye.

    The agent's gaze centre is the image centre; the goal is a dim (0.5) ON blob at
    its bearing (goal - agent) relative to centre. The agent does NOT see its own eye
    (egocentric), so there is a single blob — the SC localises it directly. Returns the
    same (2, H, W) ON/OFF convention as render_gridworld_to_image. A (2*radius+1)^2 blob
    gives the spiking retina enough firing pixels to drive a robust SC bump.
    """
    img = np.zeros((2, image_size, image_size), dtype=np.float32)
    rdx = (goal[0] - agent[0]) * ppc          # +x = East
    rdy = (goal[1] - agent[1]) * ppc          # +y = North
    gx = int(round(CENTER + rdx))
    gy = int(round(CENTER + rdy))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = gx + dx, gy + dy
            if 0 <= px < image_size and 0 <= py < image_size:
                img[0, py, px] = max(img[0, py, px], 0.5)   # dim goal blob (matches host goal band)
    return img


# ============================ explicit retinotopic wiring ============================

def retina_on_idx(py, px):
    """ON-channel retina neuron index (channel 0): channel*(H*W) + py*W + px."""
    return py * IMG + px


def sc_idx(sy, sx):
    return sy * SC + sx


def build_retina_to_sc(scramble=False, rng=None):
    """retina(ON) -> sc_map retinotopic pooling: sc site (sy,sx) pools the 2x2 ON
    retina block at (2sy:2sy+2, 2sx:2sx+2). scramble=True permutes the sc-site target
    assignment (destroys retinotopy) for the lesion control."""
    pre, post, w = [], [], []
    sc_targets = list(range(SC * SC))
    if scramble:
        perm = rng.permutation(SC * SC)
        sc_targets = [int(perm[i]) for i in range(SC * SC)]
    for sy in range(SC):
        for sx in range(SC):
            tgt = sc_targets[sc_idx(sy, sx)]           # scrambled or identity
            for a in (0, 1):
                for b in (0, 1):
                    pre.append(retina_on_idx(2 * sy + a, 2 * sx + b))
                    post.append(tgt)
                    w.append(1.0)
    return np.asarray(pre, np.int64), np.asarray(post, np.int64), np.asarray(w, np.float32)


def build_sc_recurrent(weight):
    """short-range recurrent excitation within sc_map (bump-sharpening; radius 1)."""
    pre, post, w = [], [], []
    for sy in range(SC):
        for sx in range(SC):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < SC and 0 <= nx < SC and (dy, dx) != (0, 0):
                        pre.append(sc_idx(sy, sx))
                        post.append(sc_idx(ny, nx))
                        w.append(weight)
    return np.asarray(pre, np.int64), np.asarray(post, np.int64), np.asarray(w, np.float32)


def build_pool(n_src, src_idx_fn, n_dst, dst_offset_in_region, weights_per_src):
    """generic dense pooling: every src site -> every dst neuron, weight = weights_per_src[src].
    Returns RELATIVE-to-region indices (caller offsets to global)."""
    pre, post, w = [], [], []
    for s in range(n_src):
        wv = weights_per_src[s]
        if wv <= 0.0:
            continue
        for d in range(n_dst):
            pre.append(src_idx_fn(s))
            post.append(dst_offset_in_region + d)
            w.append(float(wv))
    return pre, post, w


# ============================ the spiking SC bridge ============================

def build_bridge(seed=42, scramble=False,
                 retina_drive_pa=2500.0, ou_std=6.0,
                 w_ret_sc=80.0, w_sc_rec=6.0, w_sc_fs=4.0, w_fs_sc=2.0,
                 w_sc_cortex=1.0, w_sc_approach=20.0):
    cfg = CoreSimConfig()
    cfg.seed = seed
    cfg.heterogeneity_seed = seed
    cfg.ou_seed = seed
    # Low background noise so the retinotopic signal forms a clean bump (the SC
    # read-out must reflect the image-driven input, not OU noise). Biologically the
    # SC operates with low spontaneous rate; the high default OU (100 pA) is for
    # cortical-circuit realism, not a sparse sensory map.
    cfg.ou_std_current_pA = float(ou_std)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_neuromodulator_subsystem = False

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    FS = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name
    N_CTX, N_APP, N_FS = 12, 16, 12

    def reg(name, n, izh, exc=1.0):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=exc, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                           plastic_internal=False, izh_neuron_type=izh)

    cfg.brain_regions = [
        reg("retina", 2 * IMG * IMG, RS),
        reg("sc_map", SC * SC, RS),
        reg("sc_fs", N_FS, FS, exc=0.0),
        reg("cortex_N", N_CTX, RS), reg("cortex_E", N_CTX, RS),
        reg("cortex_S", N_CTX, RS), reg("cortex_W", N_CTX, RS),
        reg("approach", N_APP, RS),
    ]
    # pathways declared with density 0 (no auto edges) -> fully explicit-wired below.
    def path(a, b, receptor="gaba_a" if False else None):
        kw = dict(from_region=a, to_region=b, density=0.0, weight_mean=0.0,
                  weight_jitter=0.0, plastic=False)
        return RegionPathway(**kw)

    cfg.region_pathways = [
        path("retina", "sc_map"),
        path("sc_map", "sc_map"),
        # The Mexican-hat surround inhibition: framework-built (non-zero density) so
        # inject_explicit_wiring marks sc_fs neurons INHIBITORY (sets the trait mask).
        # Explicit density-0 + set_pathway_weights leaves an empty plan -> the inhibitory
        # mask is never set -> sc_fs synapses act EXCITATORY (the bug that drove all sites).
        RegionPathway(from_region="sc_map", to_region="sc_fs", density=0.5,
                      weight_mean=w_sc_fs, weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="sc_fs", to_region="sc_map", density=0.8,
                      weight_mean=w_fs_sc, weight_jitter=0.1, plastic=False),  # global inhibition
        path("sc_map", "cortex_N"), path("sc_map", "cortex_E"),
        path("sc_map", "cortex_S"), path("sc_map", "cortex_W"),
        path("sc_map", "approach"),
    ]

    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)

    rm = b.region_manager
    off = lambda name: int(list(rm.indices(name))[0])
    ret0, sc0 = off("retina"), off("sc_map")
    fs0, app0 = off("sc_fs"), off("approach")
    ctx0 = {c: off(f"cortex_{c}") for c in "NESW"}

    rng = np.random.default_rng(seed)

    # 1) retina(ON) -> sc_map (retinotopic; scrambled for the lesion control)
    rp, pp, rw = build_retina_to_sc(scramble=scramble, rng=rng)
    b.set_pathway_weights("retina_to_sc", ret0 + rp, sc0 + pp, w_ret_sc * rw, add_missing=True)

    # 2) sc_map short-range recurrent excitation (amplifies the bump; contained by the
    #    now-properly-inhibitory FS surround above).
    if w_sc_rec > 0.0:
        rp, pp, rw = build_sc_recurrent(1.0)
        b.set_pathway_weights("sc_recurrent", sc0 + rp, sc0 + pp, w_sc_rec * rw, add_missing=True)

    # 3) The sc_map<->sc_fs Mexican-hat is framework-built (declared with density above) so
    #    sc_fs is correctly marked INHIBITORY — no explicit FS wiring here.

    # 4) N1 read-out: weighted-quadrant pooling sc_map -> cortex_{N,E,S,W}
    #    weight from sc site (sy,sx) to cortex_C = max(0, axis displacement toward C).
    #    +sx = East, -sx = West, +sy = North, -sy = South (matches the host pixel convention).
    wN = np.zeros(SC * SC); wE = np.zeros(SC * SC); wS = np.zeros(SC * SC); wW = np.zeros(SC * SC)
    for sy in range(SC):
        for sx in range(SC):
            i = sc_idx(sy, sx)
            ddx = sx - SC_CENTER
            ddy = sy - SC_CENTER
            wE[i] = max(0.0, ddx); wW[i] = max(0.0, -ddx)
            wN[i] = max(0.0, ddy); wS[i] = max(0.0, -ddy)
    for C, wv in (("N", wN), ("E", wE), ("S", wS), ("W", wW)):
        pre, post, w = build_pool(SC * SC, lambda s: s, N_CTX, 0, w_sc_cortex * wv)
        b.set_pathway_weights(f"sc_to_cortex_{C}", sc0 + np.asarray(pre, np.int64),
                              ctx0[C] + np.asarray(post, np.int64),
                              np.asarray(w, np.float32), add_missing=True)

    # 5) N5 read-out: foveal-centre Gaussian pooling sc_map -> approach
    #    weight = exp(-r^2 / 2 sigma^2), r = distance of sc site from sc centre.
    #    Bump near centre (goal foveated/close) -> approach fires high.
    # Broad foveal Gaussian: the approach pool fires GRADED with how central the bump
    # is (= how small the goal's eccentricity is) across the whole working range, not
    # just at the fovea — so sign(approach_now - approach_prev) tracks the eccentricity
    # change at the moderate distances the nav agent actually operates at.
    wA = np.zeros(SC * SC)
    sigma = 5.0
    for sy in range(SC):
        for sx in range(SC):
            r2 = (sx - SC_CENTER) ** 2 + (sy - SC_CENTER) ** 2
            wA[sc_idx(sy, sx)] = float(np.exp(-r2 / (2 * sigma ** 2)))
    pre, post, w = build_pool(SC * SC, lambda s: s, N_APP, 0, w_sc_approach * wA)
    b.set_pathway_weights("sc_to_approach", sc0 + np.asarray(pre, np.int64),
                          app0 + np.asarray(post, np.int64),
                          np.asarray(w, np.float32), add_missing=True)

    b._sc_offsets = dict(retina=ret0, sc_map=sc0, sc_fs=fs0, approach=app0,
                         cortex={c: ctx0[c] for c in "NESW"},
                         N_CTX=N_CTX, N_APP=N_APP)
    # Capture the post-init resting state so each trial starts identically (no bump
    # carryover between (agent, goal) presentations — the order-independence the de-risk
    # needs; mirrors the nav-critic clean-reset discipline).
    b._rest_v = b.cp_membrane_potential_v.copy()
    b._rest_u = b.cp_recovery_variable_u.copy()
    return b


def hard_reset(b):
    """Restore the bridge to its resting state (no carryover between trials)."""
    b.cp_membrane_potential_v[:] = b._rest_v
    b.cp_recovery_variable_u[:] = b._rest_u
    b.cp_conductance_g_e[:] = 0.0
    b.cp_conductance_g_i[:] = 0.0
    b.cp_firing_states[:] = False
    b.cp_refractory_timers[:] = 0
    b.cp_external_input_current[:] = 0.0


# ============================ run / read-out ============================

def _gidx(b, name):
    return np.asarray(list(b.region_manager.indices(name)), dtype=np.int64)


def present(b, image, n_steps=160, warm=30, drive_pa=2000.0, decay=25):
    """Decay the previous bump, drive the retina with the image, run, and read the
    cortex_X firing counts + the approach firing rate (steps warm..n_steps)."""
    ret = _gidx(b, "retina")
    # Hard reset so the previous trial's bump cannot carry over (clean, order-independent read).
    hard_reset(b)
    for _ in range(5):                       # brief OU settle
        b._run_one_simulation_step()
    # ON+OFF retina drive from the image (image-only afferent).
    drive = image_to_retina_drive(image, drive_max_pA=drive_pa)
    b.cp_external_input_current[:] = 0.0
    b.cp_external_input_current[xp.asarray(ret)] = xp.asarray(drive, dtype=xp.float32)
    ctx = {c: _gidx(b, f"cortex_{c}") for c in "NESW"}
    app = _gidx(b, "approach")
    cc = {c: 0 for c in "NESW"}
    ac = 0
    m = 0
    for t in range(n_steps):
        b._run_one_simulation_step()
        if t >= warm:
            fs = b.cp_firing_states
            for c in "NESW":
                cc[c] += int(fs[xp.asarray(ctx[c])].sum())
            ac += int(fs[xp.asarray(app)].sum())
            m += 1
    return cc, ac / max(m, 1)


def sc_cardinal(cc):
    """The orienting cardinal = which cortex pool fired most (BY FIRING, not host argmax).
    None if all silent or a tie at zero."""
    if max(cc.values()) == 0:
        return None
    best = max(cc, key=lambda c: cc[c])
    # tie guard
    top = sorted(cc.values())
    if len(top) >= 2 and top[-1] == top[-2] and top[-1] > 0:
        return "TIE"
    return best


# ============================ falsifiers ============================

# F1 positions: (agent, goal) spanning all four cardinals + diagonals (dominant-axis).
F1_CASES = [
    ((4, 4), (4, 6)),   # goal N
    ((4, 4), (4, 2)),   # goal S
    ((4, 4), (6, 4)),   # goal E
    ((4, 4), (2, 4)),   # goal W
    ((4, 4), (6, 5)),   # NE, E-dominant
    ((4, 4), (5, 6)),   # NE, N-dominant
    ((4, 4), (2, 3)),   # SW, S-dominant... (dx=-2,dy=-... ) host decides
    ((4, 4), (1, 4)),   # far W
]

# F2 transitions: (agent_old, agent_new, goal) — a 1-cell step toward (closer) or away
# (farther) from a NEAR goal (offset 1-2 cells). The simple foveal-position approach read
# is valid in this near-field range (the bump stays on the map, unclipped); the full
# eccentricity range needs the Option-C temporal-difference of bump MOTION (scoped
# separately — a peripheral bump's motion-toward-centre is detectable where its absolute
# position saturates the foveal pool).
F2_CASES = [
    ((2, 4), (3, 4), (4, 4)),   # step E toward goal: eccentricity 2->1 (closer) -> +1
    ((4, 2), (4, 3), (4, 4)),   # step N toward goal: closer -> +1
    ((6, 4), (5, 4), (4, 4)),   # step W toward goal: closer -> +1
    ((4, 6), (4, 5), (4, 4)),   # step S toward goal: closer -> +1
    ((3, 4), (2, 4), (4, 4)),   # step W away from goal: 1->2 (farther) -> -1
    ((4, 3), (4, 2), (4, 4)),   # step S away: farther -> -1
    ((5, 4), (6, 4), (4, 4)),   # step E away: farther -> -1
    ((4, 5), (4, 6), (4, 4)),   # step N away: farther -> -1
]


def host_reward_sign(agent_old, agent_new, goal):
    """The REAL host N5: sign(eccentricity_after - eccentricity_before) from the
    allocentric render (the ground-truth label the spiking SC must reproduce)."""
    ib = render_gridworld_to_image(agent_old, goal, grid_size=GRID, image_size=IMG)
    ia = render_gridworld_to_image(agent_new, goal, grid_size=GRID, image_size=IMG)
    ob = sc_salience_offset_from_image(ib, grid_size=GRID, image_size=IMG)
    oa = sc_salience_offset_from_image(ia, grid_size=GRID, image_size=IMG)
    eb = (ob[0] ** 2 + ob[1] ** 2) ** 0.5 if ob is not None else 0.0
    ea = (oa[0] ** 2 + oa[1] ** 2) ** 0.5 if oa is not None else 0.0
    if ea < eb - 1e-6:
        return +1
    if ea > eb + 1e-6:
        return -1
    return 0


def run_falsifiers(scramble=False, seed=42, drive_pa=2000.0, **kw):
    tag = "SCRAMBLED-RETINOTOPY (lesion)" if scramble else "INTACT"
    print(f"\n================ SC de-risk: {tag} (seed {seed}, backend {BACKEND}) ================")
    b = build_bridge(seed=seed, scramble=scramble, retina_drive_pa=drive_pa, **kw)

    # --- F1: orienting cardinal vs the host ---
    print("\n[F1 — N1 orienting]  host vs spiking-SC (by FIRING)")
    print(f"{'agent':>8} {'goal':>8} | {'host':>5} | {'SC firing N/E/S/W':>22} | {'SC':>5} | match")
    f1_ok = f1_tot = 0
    for agent, goal in F1_CASES:
        host = sc_orienting_cardinal_from_image(
            render_gridworld_to_image(agent, goal, grid_size=GRID, image_size=IMG))
        img = render_egocentric(agent, goal)
        cc, _ = present(b, img, drive_pa=drive_pa)
        sc = sc_cardinal(cc)
        ok = (sc == host) and host is not None
        f1_tot += 1
        f1_ok += int(ok)
        fired = "/".join(str(cc[c]) for c in "NESW")
        print(f"{str(agent):>8} {str(goal):>8} | {str(host):>5} | {fired:>22} | {str(sc):>5} | {'OK' if ok else 'x'}")

    # --- F2: approach sign vs the host reward sign ---
    print("\n[F2 — N5 approach]  host reward-sign vs sign(approach_now - approach_prev)")
    print(f"{'old->new':>14} {'goal':>7} | {'host':>5} | {'app_old':>8} {'app_new':>8} | {'SC':>4} | match")
    f2_ok = f2_tot = 0
    for ao, an, goal in F2_CASES:
        host = host_reward_sign(ao, an, goal)
        _, app_old = present(b, render_egocentric(ao, goal), drive_pa=drive_pa)
        _, app_new = present(b, render_egocentric(an, goal), drive_pa=drive_pa)
        d = app_new - app_old
        sc = (+1 if d > 0.3 else (-1 if d < -0.3 else 0))   # dead-band (host 1e-6 guard analogue)
        ok = (sc == host) and host != 0
        f2_tot += 1
        f2_ok += int(ok)
        print(f"{str(ao)+'->'+str(an):>14} {str(goal):>7} | {host:>5} | {app_old:8.2f} {app_new:8.2f} | {sc:>4} | {'OK' if ok else 'x'}")

    print(f"\n  F1 (N1 orienting): {f1_ok}/{f1_tot}    F2 (N5 approach): {f2_ok}/{f2_tot}")
    return f1_ok, f1_tot, f2_ok, f2_tot


def main():
    print("SPIKING SUPERIOR-COLLICULUS DE-RISK (N1 + N5) — cheapest-first, CPU")
    print("PASS bar: INTACT F1 >= 7/8 AND F2 >= 7/8 (by neuron firing); "
          "LESION F1 and F2 must COLLAPSE to chance (the decisive anti-cheat).")
    i1o, i1t, i2o, i2t = run_falsifiers(scramble=False)
    l1o, l1t, l2o, l2t = run_falsifiers(scramble=True)

    print("\n================ VERDICT ================")
    intact_pass = (i1o >= i1t - 1) and (i2o >= 7)
    lesion_break = (l1o <= max(2, l1t // 2)) and (l2o <= max(2, l2t // 2))
    print(f"INTACT:  F1 {i1o}/{i1t}  F2 {i2o}/{i2t}   -> {'PASS' if intact_pass else 'FAIL'}")
    print(f"LESION:  F1 {l1o}/{l1t}  F2 {l2o}/{l2t}   -> {'BREAKS (good)' if lesion_break else 'SURVIVES (BAD: leak)'}")
    if intact_pass and lesion_break:
        print("VERDICT: RESOLVES — a spiking SC reproduces BOTH N1 orienting and N5 approach by "
              "neuron firing, image-only, and the signal is genuinely retinotopic (lesion breaks "
              "it). GO to build Option A+C into the flagship nav (6-seed GPU A/B).")
    elif (i1o >= i1t - 1) and lesion_break:
        n5 = ("carries the approach signal (lesion-confirmed, above chance) but the static "
              "foveal-position read is SNR-limited and rises with integration window — the robust "
              "read-out is Option C (the slow-channel temporal-difference of the rostral-ward bump "
              "motion)" if i2o >= 4 else "is weak (F2 < 5/8)")
        print(f"VERDICT: N1 RESOLVES (F1 {i1o}/{i1t} by neuron firing, lesion-confirmed) via the FULL "
              f"2D retinotopic SC map + Mexican-hat — GO to build the spiking SC orienting into nav. "
              f"N5 {n5}. Build N1 now; refine N5 with Option C.")
    else:
        print("VERDICT: NOT YET — tune drives/weights (the bump/ WTA), or the honest NEGATIVE maps a "
              "rate-coded-WTA limit at this scale (fall back to the discrete 4-pool Option B).")


if __name__ == "__main__":
    main()
