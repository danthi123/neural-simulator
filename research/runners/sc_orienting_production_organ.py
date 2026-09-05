"""SPIKING SUPERIOR COLLICULUS — reflexive visual ORIENTING / saliency ("where"), wired as a
PRODUCTION visuomotor organ (N1, de-risked 2026-06-10, 6-seed CLOSED GO).

The owner's canonical "build a cheap visuomotor consumer" case. The nav agent's orienting was a HOST
pixel-reader (`sc_orienting_cardinal_from_image`: pixels -> argmax -> cardinal, a numpy computation
between sensation and action = a BRAIN-BASED-ONLY-bar shortcut). N1 replaced it with a SPIKING
superior colliculus: an egocentric retinal image -> a 16x16 retinotopic spiking sheet (`sc_map`) with
a Mexican-hat surround (`sc_fs`) winner-take-all -> a topographic `sc_map -> cortex_{N,E,S,W}` pooling
whose winning pool BY FIRING is the orienting cardinal. 6-seed nav A/B: SC/host 0.883 (12% better,
5/6 seeds win); the scrambled-retinotopy lesion regresses 2.4x
(`research/findings/2026-06-10-N1-spiking-superior-colliculus-CLOSED.md`).

This module packages that spiking SC as a STANDALONE, process-shared ORGAN and drives a tiny EMBODIED
visuomotor loop (the consumer, in `_sc_orienting_production_organ_verify.py`): render the world from the
agent's eye (an off-centre salient goal blob), drive `sc_retina`, step the bridge, let the `sc_map`
Mexican-hat WTA form a single saliency bump at the blob's retinotopic site, and read the winning
`cortex_{N,E,S,W}` pool BY FIRING as the orienting cardinal -> the BODY moves the agent one cell in that
cardinal. So the SPIKING SC (not the host `--sc-orienting-reflex` scaffold) drives orienting.

REUSE-BY-IMPORT (NO `sim/` edit): the wiring is `install_spiking_sc_wiring` from `g11_bg_runner` verbatim
(the SAME production machinery the nav runner uses behind `--enable-spiking-sc`); the render is
`render_egocentric_goal`; the OFF-path scaffold / ground-truth is `sc_orienting_cardinal_from_image`.
The organ only ADDS the minimal region scaffold those functions expect (`sc_retina`, `sc_map`, `sc_fs`,
`cortex_{N,E,S,W}` + the framework-built Mexican-hat) — exactly the subset `build_bg_brain_regions`
builds when `enable_spiking_sc=True` — then calls the production wiring on it.

BRAIN-BASED: the orienting cardinal is a `cp_firing_states[cortex_X]` READ off a topographic spiking
read-out of a retinotopic WTA sheet — no host argmax/coordinate enters the decision. The only host
boundaries are the two the bar permits: (1) the ENVIRONMENT render (the world from the agent's eye ->
the retinal image the neural retina receives) and (2) the BODY (the agent moving one cell by which
cortex pool fired). The `sc_map` firing-peak SITE is also reported as a saliency INSTRUMENT (a "where"
read-out), but it is NOT the decision — the cardinal is the cortex-pool firing (the load-bearing read).

ADDITIVE + DEFAULT-ON (flipped 2026-09-05, scaffold-retirement backlog rank 24): gated behind
`BRAIN_SPIKING_SC_ORIENT`, default ON now that the 6-seed flip-soak GATE passes fresh
(`research/runners/_sc_orienting_flip_soak.py`: INTACT correct-cardinal min 1.000 >= 0.80, LESION max
0.333 <= 0.45; INTACT embodied reach min 1.000 >= 0.80, LESION max 0.125 <= 0.50 — see
research/findings/2026-09-05-enable-spiking-sc-split-verdict-organ-flag-GO-library-default-reverted.md).
`sc_orient_enabled()` currently has
NO caller anywhere in the tree (this organ + its verify/soak scripts read `lesion=` directly, not this
flag) — it wires into NO existing production path either way, so flipping the default changes NOTHING
observable TODAY; it sets the correct default for whenever a future consumer is built, so it inherits the
spiking SC rather than silently defaulting to the host-reflex scaffold. `BRAIN_SPIKING_SC_ORIENT=0` is the
explicit escape back to the host reflex `sc_orienting_cardinal_from_image` (the scaffold the spiking SC
replaces) for any future caller that wants the oracle/comparator arm.

LESION-LOAD-BEARING (the faculty's OWN oracle): `SC_SCRAMBLE=1` permutes the `sc_retina -> sc_map`
retinotopic target assignment (the de-risk's decisive anti-cheat). The image-only afferent (the
`sc_retina` drive) is UNCHANGED — only the retinotopic sheet's topography is destroyed — so the `sc_map`
bump no longer sits at the blob's true retinotopic site, the cortex pooling reads a DECOUPLED cardinal,
and the correct-cardinal rate collapses to chance (the nav's 2.4x regression analogue). This proves the
retinotopic SPIKING sheet carries the orienting target, not a re-hidden host read.

FUNCTIONAL, NOT phenomenal: this measures + reports an orienting/saliency CORRELATE (a retinotopic bump
+ a cardinal read-out). It makes NO claim of visual experience.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests/CPU smoke).
"""
from __future__ import annotations

import os

import numpy as np

# --- reuse-by-import the de-risked PRODUCTION SC machinery (NO reinvention, NO sim/ edit) ---
from research.runners.g11_bg_runner import (
    install_spiking_sc_wiring,       # the production retina->sc_map->cortex_NESW wiring (+ scramble lesion)
    render_egocentric_goal,          # the ENVIRONMENT render (the world from the agent's eye)
    sc_orienting_cardinal_from_image,  # the OFF-path host reflex / ground-truth (the scaffold N1 replaces)
    ACTION_NAMES,                    # ["N", "E", "S", "W"]
)
from sim.visual_cortex import image_to_retina_drive

# ── geometry (matches the de-risk + the nav runner) ───────────────────────────────────────────────
IMG = 32                 # sc_retina is IMG x IMG x 2 channels (ON/OFF)
SC = IMG // 2            # sc_map is SC x SC retinotopic sites (16)
N_CTX = 12              # per-cardinal cortex pool size (the read-out pools)
N_FS = 12              # Mexican-hat surround interneurons

# de-risked operating point (the sc_map_orienting_probe op-point that scored 8/8 orienting, CPU):
# low background OU so the retinotopic signal forms a clean bump; strong eye drive; the ramp read-out.
OU_STD = 6.0
RET_DRIVE_PA = 2500.0    # sc_retina eye drive (matches the nav loop's SC_RET_DRIVE default)
W_RET_SC = 80.0          # retina -> sc_map retinotopic weight
W_SC_REC = 6.0           # sc_map short-range recurrent excitation
W_SC_CORTEX = 1.0        # sc_map -> cortex_X pooling gain (ramp read-out; probe op-point)
WARM_STEPS = 30          # OU settle before reading (drop the transient)
READ_STEPS = 160         # steps the bump is read over (matches the probe's present() window)
SETTLE_STEPS = 5         # brief settle after the hard reset


def sc_orient_enabled() -> bool:
    """DEFAULT-OFF. `BRAIN_SPIKING_SC_ORIENT` in {1,true,yes,on} -> the spiking SC drives orienting.
    Unset / {0,false,no,off} -> disabled (the consumer falls back to the host reflex scaffold)."""
    v = os.environ.get("BRAIN_SPIKING_SC_ORIENT")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def sc_orient_scrambled() -> bool:
    """`SC_SCRAMBLE=1` -> the load-bearing lesion (permute the sc_retina->sc_map retinotopy). The
    faculty's OWN de-risk anti-cheat: the image-only afferent is unchanged; only the retinotopic
    assignment is destroyed -> the orienting decouples from the blob location."""
    return os.environ.get("SC_SCRAMBLE", "0").strip().lower() in ("1", "true", "yes", "on")


def true_bearing_cardinal(agent, goal):
    """The dominant-axis cardinal of (goal - agent), in the render's convention (+x=East, +y=North).
    A SCORING oracle only (the ground-truth the spiking read-out is graded against) — it is NOT part
    of the brain path (the brain never sees coordinates). Returns None when co-located."""
    ddx = float(goal[0] - agent[0])   # +x = East
    ddy = float(goal[1] - agent[1])   # +y = North
    if abs(ddx) < 0.5 and abs(ddy) < 0.5:
        return None
    if abs(ddx) >= abs(ddy):
        return "E" if ddx > 0 else "W"
    return "N" if ddy > 0 else "S"


class SpikingSCOrientingOrgan:
    """A process-shared spiking superior colliculus. Built ONCE (lazily): the minimal region scaffold
    (`sc_retina`/`sc_map`/`sc_fs`/`cortex_{N,E,S,W}` + the framework-built Mexican-hat) that the
    production `install_spiking_sc_wiring` expects, then the production wiring itself. Each `orient`
    renders the world from the agent's eye, drives `sc_retina`, steps the bridge, and reads the winning
    `cortex_X` pool BY FIRING (the orienting cardinal). A lesioned twin (scrambled retinotopy) is built
    on demand for the load-bearing check.

    `scramble` at construction runs the WHOLE organ lesioned (the intended production path reads
    `SC_SCRAMBLE` at build). `orient(..., lesion=True)` reads the on-demand scrambled twin for an
    intact-vs-lesion A/B within one process."""

    def __init__(self, seed: int = 42, scramble: bool | None = None,
                 log_polar: bool = False):
        self.seed = int(seed)
        # default the base organ's scramble from the env oracle (production reads SC_SCRAMBLE at build)
        self.scramble = sc_orient_scrambled() if scramble is None else bool(scramble)
        self.log_polar = bool(log_polar)
        self._built = False
        self.bridge = self.cfg = self.xp = None
        self.idx_ctx = None       # {cardinal: cupy int index array}
        self.idx_sc = None        # sc_map global indices (for the saliency-peak instrument)
        self.snap0 = None         # clean post-init resting state (per-orient reset)
        self.les = None           # lazily-built lesioned twin (scrambled retinotopy)

    # ── construction ──────────────────────────────────────────────────────────────────────────────
    def _build_one(self, scramble: bool):
        """Build the minimal spiking-SC bridge and install the production wiring. Returns
        (bridge, cfg, xp, idx_ctx, idx_sc, snap0)."""
        from sim.backend import get_backend
        from sim.bridge import SimulationBridge
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.regions import BrainRegion, RegionPathway
        from sim.enums import NeuronModel, NeuronType
        xp, _ = get_backend()

        cfg = CoreSimConfig()
        cfg.seed = self.seed
        cfg.heterogeneity_seed = self.seed
        cfg.ou_seed = self.seed
        # Low background OU so the retinotopic input forms a clean bump (the SC read-out must reflect
        # the image-driven input, not OU noise) — the de-risk operating point (the SC runs at a low
        # spontaneous rate; the high default OU is for cortical-circuit realism, not a sparse map).
        cfg.ou_std_current_pA = float(OU_STD)
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

        def reg(name, n, izh, exc=1.0):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=exc, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                               plastic_internal=False, izh_neuron_type=izh)

        # EXACTLY the subset build_bg_brain_regions builds for enable_spiking_sc (g11 L2761-2790):
        # the SC's own egocentric eye + the retinotopic sheet + the Mexican-hat FS + the 4 read-out pools.
        cfg.brain_regions = [
            reg("sc_retina", 2 * IMG * IMG, RS),
            reg("sc_map", SC * SC, RS),
            reg("sc_fs", N_FS, FS, exc=0.0),
            reg("cortex_N", N_CTX, RS), reg("cortex_E", N_CTX, RS),
            reg("cortex_S", N_CTX, RS), reg("cortex_W", N_CTX, RS),
        ]
        # The Mexican-hat surround is FRAMEWORK-built (declared with REAL density) so
        # inject_explicit_wiring marks sc_fs INHIBITORY (the de-risk gotcha: a density-0 +
        # set_pathway_weights route leaves the mask unset -> sc_fs acts EXCITATORY and drives the
        # whole map). install_spiking_sc_wiring does the rest post-init (retina->sc_map, recurrent,
        # sc_map->cortex_NESW). Same declaration as g11 build_bg_brain_regions.
        cfg.region_pathways = [
            RegionPathway(from_region="sc_map", to_region="sc_fs",
                          density=0.5, weight_mean=4.0, weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="sc_fs", to_region="sc_map",
                          density=0.8, weight_mean=2.0, weight_jitter=0.1, plastic=False),
        ]

        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)

        # the PRODUCTION wiring (verbatim from g11 — the same call the nav runner makes).
        install_spiking_sc_wiring(b, visual_image_size=IMG, w_ret_sc=W_RET_SC,
                                  w_sc_rec=W_SC_REC, w_sc_cortex=W_SC_CORTEX,
                                  scramble=bool(scramble), popvector=False, verbose=False)

        rm = b.region_manager
        idx_ctx = {c: xp.asarray(list(rm.indices(f"cortex_{c}")), dtype=xp.int64) for c in ACTION_NAMES}
        idx_sc = xp.asarray(list(rm.indices("sc_map")), dtype=xp.int64)
        # capture the clean post-init resting state so each orient starts identically (no bump carryover
        # between presentations — the order-independence the read-out needs; mirrors the de-risk).
        snap0 = dict(v=b.cp_membrane_potential_v.copy(), u=b.cp_recovery_variable_u.copy())
        # THE READ-ISOLATION FIX (2026-09-02, board #150's ~29-runner follow-up audit): `_hard_reset` already
        # restored `cp_refractory_timers` but never `cp_prev_firing_states` (a HARD firing gate independent of
        # membrane potential) or the homeostatic pair (`cp_neuron_activity_ema` / `cp_neuron_firing_thresholds` —
        # this organ's CoreSimConfig defaults `enable_homeostasis=True` and is never overridden, so these are
        # NOT config-inert here, unlike several sibling runners). A repeat-read diagnostic (same (agent,goal)
        # presented twice in a row) found a small but real leak (cortex_N count 15 vs 16 spikes, sc_total_spikes
        # 185 vs 184 — ~0.5-1% relative); the fix snapshots the true post-settle rest value for the two arrays
        # that need one and zeroes the two hard-gate arrays whose true-rest value is unambiguous.
        for nm in ("cp_neuron_activity_ema", "cp_neuron_firing_thresholds"):
            arr = getattr(b, nm, None)
            snap0[nm] = arr.copy() if arr is not None else None
        return b, cfg, xp, idx_ctx, idx_sc, snap0

    def ensure_built(self):
        if self._built:
            return
        (self.bridge, self.cfg, self.xp, self.idx_ctx, self.idx_sc,
         self.snap0) = self._build_one(scramble=self.scramble)
        self._built = True

    def _ensure_les(self):
        """Lazily build the scrambled-retinotopy twin (the load-bearing lesion). The image-only afferent
        is identical; only the sc_retina->sc_map topography is permuted."""
        if self.les is None:
            b, c, xp, idx_ctx, idx_sc, snap0 = self._build_one(scramble=True)
            self.les = dict(bridge=b, cfg=c, xp=xp, idx_ctx=idx_ctx, idx_sc=idx_sc, snap0=snap0)
        return self.les

    # ── the read-out ──────────────────────────────────────────────────────────────────────────────
    def _hard_reset(self, bridge, snap0):
        """Restore the resting state (no bump carryover between presentations)."""
        bridge.cp_membrane_potential_v[:] = snap0["v"]
        bridge.cp_recovery_variable_u[:] = snap0["u"]
        bridge.cp_conductance_g_e[:] = 0.0
        bridge.cp_conductance_g_i[:] = 0.0
        bridge.cp_firing_states[:] = False
        bridge.cp_refractory_timers[:] = 0
        # THE READ-ISOLATION FIX: the other 2 C2 hard-gate/homeostatic arrays (see `_build_one`'s snapshot
        # comment above). `cp_prev_firing_states` has an unambiguous False true-rest value; the homeostatic
        # pair is restored only from the actual snapshot (never guessed).
        if getattr(bridge, "cp_prev_firing_states", None) is not None:
            bridge.cp_prev_firing_states[:] = False
        for nm in ("cp_neuron_activity_ema", "cp_neuron_firing_thresholds"):
            arr = getattr(bridge, nm, None)
            snap_val = snap0.get(nm)
            if arr is not None and snap_val is not None:
                arr[:] = snap_val
        bridge.cp_external_input_current[:] = 0.0

    def _orient_on(self, bridge, xp, idx_ctx, idx_sc, snap0, agent, goal):
        """Drive the SC eye with the egocentric render, step, and read the cortex-pool firing counts +
        the sc_map peak site. Returns (cortex_counts, sc_peak_site, sc_total_spikes)."""
        rm = bridge.region_manager
        ret_idx = xp.asarray(list(rm.indices("sc_retina")), dtype=xp.int64)
        self._hard_reset(bridge, snap0)
        for _ in range(SETTLE_STEPS):
            bridge._run_one_simulation_step()
        # ENVIRONMENT render (channel-1 legit): the world from the agent's eye, a dim ON blob at the
        # goal's bearing. image_to_retina_drive -> the sc_retina afferent (ON+OFF), image-only.
        img = render_egocentric_goal((int(agent[0]), int(agent[1])),
                                     (int(goal[0]), int(goal[1])),
                                     image_size=IMG, log_polar=self.log_polar)
        drive = image_to_retina_drive(img, drive_max_pA=RET_DRIVE_PA)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[ret_idx] = xp.asarray(drive, dtype=xp.float32)

        counts = {c: 0 for c in ACTION_NAMES}
        sc_accum = None
        n_read = 0
        for t in range(READ_STEPS):
            bridge._run_one_simulation_step()
            if t >= WARM_STEPS:
                fs = bridge.cp_firing_states
                for c in ACTION_NAMES:
                    counts[c] += int(fs[idx_ctx[c]].sum())
                sc_fire = fs[idx_sc].astype(xp.float32)
                sc_accum = sc_fire if sc_accum is None else (sc_accum + sc_fire)
                n_read += 1
        sc_total = float(sc_accum.sum()) if sc_accum is not None else 0.0
        # the sc_map firing-peak SITE (the saliency "where" instrument; NOT the decision)
        if sc_accum is not None and float(sc_accum.max()) > 0.0:
            peak = int(np.asarray(sc_accum.argmax() if xp is np else sc_accum.argmax().get()))
            sc_peak = (peak // SC, peak % SC)   # (sy, sx)
        else:
            sc_peak = None
        return counts, sc_peak, sc_total

    @staticmethod
    def _cardinal_from_counts(counts):
        """The orienting cardinal = which cortex pool fired MOST (BY FIRING, not a host argmax over an
        external quantity — this is the spiking read-out's own winner). None if all silent; 'TIE' on a
        nonzero tie."""
        mx = max(counts.values())
        if mx == 0:
            return None
        top = sorted(counts.values())
        if len(top) >= 2 and top[-1] == top[-2] and top[-1] > 0:
            return "TIE"
        return max(counts, key=lambda c: counts[c])

    def orient(self, agent, goal, lesion: bool = False) -> dict:
        """Render the world from the agent's eye, drive the spiking SC, and return the orienting cardinal
        read off the winning cortex pool BY FIRING, plus the cortex firing counts, the sc_map saliency
        peak, and the total sc_map spikes (the bump strength). `lesion=True` reads the scrambled-
        retinotopy twin (the load-bearing anti-cheat: the cardinal decouples from the blob location)."""
        self.ensure_built()
        if lesion:
            st = self._ensure_les()
            counts, sc_peak, sc_total = self._orient_on(
                st["bridge"], st["xp"], st["idx_ctx"], st["idx_sc"], st["snap0"], agent, goal)
        else:
            counts, sc_peak, sc_total = self._orient_on(
                self.bridge, self.xp, self.idx_ctx, self.idx_sc, self.snap0, agent, goal)
        return {"on": True, "lesioned": bool(lesion or self.scramble),
                "cardinal": self._cardinal_from_counts(counts),
                "cortex_counts": counts, "sc_peak_site": sc_peak,
                "sc_total_spikes": float(sc_total)}


_ORGAN: SpikingSCOrientingOrgan | None = None


def get_organ(seed: int = 42, scramble: bool | None = None,
              log_polar: bool = False) -> SpikingSCOrientingOrgan:
    """The process-shared spiking SC orienting organ (built once on first use). A fresh `scramble`/seed
    request rebuilds (the verify/soak build several)."""
    global _ORGAN
    if (_ORGAN is None or _ORGAN.seed != int(seed)
            or (scramble is not None and _ORGAN.scramble != bool(scramble))
            or _ORGAN.log_polar != bool(log_polar)):
        _ORGAN = SpikingSCOrientingOrgan(seed=seed, scramble=scramble, log_polar=log_polar)
    return _ORGAN


def host_reflex_cardinal(agent, goal, image_size: int = IMG):
    """The OFF-path scaffold: the host pixel-reader the spiking SC replaces. Reads the goal's retinal
    direction from the ALLOCENTRIC render (the same signal the nav `--sc-orienting-reflex` uses). Used
    by the consumer when BRAIN_SPIKING_SC_ORIENT is OFF, and as the ground-truth comparator."""
    from sim.visual_cortex import render_gridworld_to_image
    # allocentric render on a grid that contains both points (the agent bright blob + goal dim blob);
    # grid 16 fits the verify's CENTER=(8,8) and its within-FOV targets. The host reads both blob
    # centroids from the pixels (no coordinates) — the scaffold the spiking SC replaces.
    img = render_gridworld_to_image((int(agent[0]), int(agent[1])),
                                    (int(goal[0]), int(goal[1])),
                                    grid_size=16, image_size=image_size)
    return sc_orienting_cardinal_from_image(img)
