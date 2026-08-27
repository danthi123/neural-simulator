"""PROSPECTIVE MEMORY (faculty-map Tier-2) -- a spiking INTENTION LATCH + BA10-style CUE-MONITOR de-risk.

THE MISSING RUNG. A genuine conversant remembers to do something LATER: "remind me to X when Y comes up."
That is an INTENTION held ACROSS intervening turns that fires when its CUE appears -- and NOT before, NOT on
a wrong cue. This is distinct from within-turn working memory (the item you are currently manipulating): the
prospective intention must survive a stretch of UNRELATED distractor turns and then be RELEASED by a specific
external cue. Nobody in this repo owns it. The two-part mechanism (Burgess/Gilbert rostral-PFC/BA10 prospective
memory; Gollwitzer implementation intentions "when X, I will do Y"):
  (1) a PFC INTENTION LATCH -- sustained recurrent (attractor) activity that HOLDS the deferred intention
      through N intervening turns of distractor input;
  (2) a BA10 CUE-MONITOR -- a coincidence detector that RELEASES the intention (fires it) only when the
      current-turn cue assembly matches the intention's cue, and stays silent otherwise.

REUSE (sustained-attractor latch). The project already has a VALIDATED spiking persistent-attractor WM: the
cortex_ctx<->dlpfc_wm loop with per-concept outer-product attractors (content_selection_spiking.
SpikingLoopContextBuffer / biased_competition_buffer, 6-seed GO). Its concept assemblies self-sustain on real
resonate/Izhikevich spikes and hold a SET of >=3 concepts at once. We reuse that loop verbatim as the intention
LATCH (the intention = a self-sustaining assembly; distractors = other self-sustaining assemblies that provide
genuine competing WM load). We ADD the cue-monitor as the biased_competition sel-pool idiom used the other way:
per action a small NMDA-recurrent accumulator region `rel_A` (soft-WTA, never self-ignites) that receives
FEEDFORWARD EVIDENCE from BOTH the latched intention's cortex assembly AND the cue's cortex assembly. Neither
input alone crosses its ramp threshold; only the COINCIDENCE (intention held AND cue present) ramps rel_A over
threshold -> the intention FIRES (is released/executed). This is a spiking AND, computed by neurons/synapses.

BRAIN-BASED (BRAIN-BASED-ONLY standard). Latch (attractor persistence), cue-monitoring (the coincidence
integration), and release (the accumulator crossing threshold) are all done by spiking neurons + synapses; every
read is `cp_firing_states`. HOST-SCAFFOLD, FLAGGED: the INTENTION CONTENT binding -- WHICH cue releases WHICH
action -- is INSTALLED synaptically (cue_A.cortex->rel_A and act_A.cortex->rel_A outer-product edges), exactly
like every attractor in SpikingLoopContextBuffer ("learning them with the correct rule is the documented next
step"). So the *mechanism* (hold-across-turns + cue-gated release) is brain-based; the *content* (this cue -> this
action) is host-wired in this de-risk. The named follow-on LEARNS the binding via Hebbian encoding at intention-
formation time (Gollwitzer's "forming the implementation intention" = one-shot potentiation of cue->action).

PRE-REGISTERED GO GATE (6 seeds; read the runner's OWN printed verdict, do NOT lift a field). Over the seeds,
per condition, the mechanism must show ALL of:
  * FIRE-ON-CUE     : after N intervening distractor turns, presenting the RIGHT cue fires the correct action
                      (rel_correct rate >= FIRE_THR), for both act_A/cue_A and act_B/cue_B.
  * PERSISTENCE     : the intention's held assembly stays above HOLD_FLOOR across ALL N intervening turns
                      (the latch survives the distractors).
  * NO-FIRE-BEFORE  : rel_correct stays <= SILENT_MAX on every intervening turn (silent until the cue).
  * NO-FIRE-WRONGCUE: presenting the WRONG cue does NOT fire (rel <= SILENT_MAX) -- the monitor is cue-specific.
  * NO-INTENTION    : with NO intention ever latched, the cue alone does NOT fire (rel <= SILENT_MAX) -- the fire
                      is gated by the held intention, not the cue alone (anti-cheat, the coincidence is real).
  * LATCH-LESION    : zero the latch (verify held-rate collapses <= LESION_HELD_MAX AT MEASUREMENT), then the
                      cue does NOT fire (rel <= SILENT_MAX) -- the intention is FORGOTTEN.
  * SEPARATION      : FIRE rate >= SEP_RATIO x the largest silent rate (a real margin, not a hair).
GO iff >= GO_MIN_SEEDS / n_seeds seeds pass every clause. A miss NAMES the residual (weights / N / thresholds) as
the next single-variable de-risk; do NOT force GO.

reuse-by-import; NO sim/ edit (additive runner; the mechanism is built from public bridge APIs). Run:
  SIM_BACKEND=numpy python -m research.runners._pmem_intention_latch_derisk --smoke      # 1 seed, N=3, fast
  SIM_BACKEND=numpy python -m research.runners._pmem_intention_latch_derisk --derisk     # 6 seeds, N=5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# CPU-first: default to numpy (small nets); must be set before any sim import.
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to   # noqa: E402  (attribution of the latch-lesion control)
from tools.verdict import Verdict        # noqa: E402  (preconditions travel with the verdict)

OUT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_intention_latch.json")

# --------------------------------------------------------------------------------------------------------
# PRE-REGISTERED / FROZEN GO thresholds (recorded into the output JSON; the verdict reads these).
# Rates are per-neuron firing fractions over the read window (0..1).
# --------------------------------------------------------------------------------------------------------
FIRE_THR = 0.20          # rel_correct on the RIGHT cue must reach at least this (the intention FIRES)
SILENT_MAX = 0.06        # any "must stay silent" rel read must be at/below this
HOLD_FLOOR = 0.05        # the intention's held assembly must stay above this on every intervening turn
LESION_HELD_MAX = 0.02   # after lesion, the held assembly must have collapsed to at most this (lesion HOLDS)
SEP_RATIO = 2.5          # FIRE rate must be >= this x the largest silent rate in the trial (a real margin)
GO_MIN_SEEDS_FRAC = 5.0 / 6.0   # >= this fraction of seeds must pass every clause


# --------------------------------------------------------------------------------------------------------
# The spiking prospective-memory substrate: the validated cortex<->dlpfc attractor LOOP as the intention
# LATCH, plus per-action NMDA-recurrent accumulator regions `rel_A` as the cue-monitor COINCIDENCE detectors.
# --------------------------------------------------------------------------------------------------------
class ProspectiveMemory:
    """Intention latch (persistent attractor) + BA10 cue-monitor (coincidence-gated release), all spiking.

    Assemblies (disjoint cortex_ctx slices):
      * act_X  : an intention/action assembly WITH a cortex<->dlpfc attractor -> self-sustains = the LATCH.
      * dist_i : a distractor assembly WITH an attractor -> real competing WM load on intervening turns.
      * cue_X  : a cue assembly with NO attractor -> present only while externally driven (a transient cue).
    Per action X: a small NMDA-recurrent accumulator region `rel_X` (soft-WTA; ramps under evidence, never
    self-ignites) receiving FEEDFORWARD from act_X.cortex (the held intention) AND cue_X.cortex (the cue). The
    coincidence (both firing) ramps rel_X over threshold = the intention is RELEASED/FIRED.
    """

    def __init__(self, actions, distractors, n=800, pattern_size=40, attractor_weight=50.0,
                 n_rel=60, rel_recurrent_weight=0.10, rel_recurrent_density=0.5,
                 hold_to_rel_weight=3.2, cue_to_rel_weight=4.2, rel_bias_pA=-1050.0,
                 seed=42, verbose=False, shared=None):
        import sim.backend as B
        from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion, RegionPathway
        from sim.enums import NeuronType
        self.B = B
        self.xp, _ = B.get_backend()
        self.actions = list(actions)
        self.distractors = list(distractors)
        self._psize = pattern_size
        self._n_rel = n_rel
        self._rel_bias_pA = float(rel_bias_pA)   # tonic hyperpolarizing bias on the rel readout (models tonic
                                                 # inhibition setting the operating point) -> a single ff input
                                                 # stays sub-rheobase; only the COINCIDENCE crosses = a real AND.
        # shared -> a one-brain-merge MergedPool: ADOPT its bridge slice instead of building a private bridge, and
        # let the pool's explicit_wiring_fn install the attractor + cue-monitor edges (build-time, both the merged
        # and coresident arms identical). shared=None -> the ORIGINAL standalone path, byte-identical: this whole
        # __init__ is purely ADDITIVE (the shared branch is skipped, every install runs exactly as before).
        self._shared = shared

        # concepts that get a self-sustaining loop attractor: the intentions + the distractors.
        self._attractor_concepts = list(self.actions) + list(self.distractors)
        # cue assemblies get an index slice but NO attractor (transient input only).
        self._cue_names = [f"cue_{a}" for a in self.actions]
        all_cortex_assemblies = self._attractor_concepts + self._cue_names

        if shared is None:
            def loop_reg(name):
                return BrainRegion(name=name, n_neurons=n, exc_fraction=0.8, internal_density=0.0,
                                   exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                                   plastic_internal=False,
                                   izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=True)

            regions = [loop_reg("cortex_ctx"), loop_reg("dlpfc_wm")]
            for a in self.actions:
                # rel_X : soft NMDA-recurrent accumulator (the sel-pool idiom, used as a coincidence detector).
                # alpha<1 soft-WTA -> ramps/holds under converging evidence, never self-ignites from zero.
                regions.append(BrainRegion(
                    name=f"rel_{a}", n_neurons=n_rel, exc_fraction=1.0,
                    internal_density=rel_recurrent_density, exc_weight_mean=rel_recurrent_weight,
                    inh_weight_mean=0.0, weight_jitter=0.2, plastic_internal=False, enable_nmda=True,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name))

            cfg = CoreSimConfig()
            cfg.enable_brain_region_framework = True
            cfg.brain_regions = regions
            pathways = [
                # loop pathways seeded at weight 0 (as build_loop_wm_bridge with loop_weight=0): a non-empty CSR so
                # the per-concept attractors + the cue-monitor edges can be installed via set_pathway_weights.
                RegionPathway(from_region="cortex_ctx", to_region="dlpfc_wm", density=0.05,
                              weight_mean=0.0, weight_jitter=0.2, plastic=False),
                RegionPathway(from_region="dlpfc_wm", to_region="cortex_ctx", density=0.05,
                              weight_mean=0.0, weight_jitter=0.2, plastic=False),
            ]
            for a in self.actions:
                # seed a 0-weight cortex_ctx -> rel_X pathway so the CSR has the rows; the SPECIFIC evidence edges
                # (act_X.cortex->rel_X, cue_X.cortex->rel_X) are then installed at real weight via add_missing.
                pathways.append(RegionPathway(from_region="cortex_ctx", to_region=f"rel_{a}", density=0.05,
                                              weight_mean=0.0, weight_jitter=0.2, plastic=False))
            cfg.region_pathways = pathways
            cfg.dt_ms = 0.5
            cfg.seed = seed
            cfg.enable_nmda = True
            cfg.enable_ou_process = False          # quiet clean hold (the validated multi-concept-WM config)
            cfg.enable_structural_plasticity = False
            cfg.enable_hebbian_learning = False
            cfg.enable_short_term_plasticity = False
            cfg.stdp_w_max = 60.0
            cfg.fast_spike_reset = True
            bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                      runtime_state=RuntimeState(), gpu_config=GPUConfig())
            bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
            bridge._initialize_simulation_data(called_from_playback_init=False)
            self.bridge = bridge
        else:
            # SHARED: adopt the pool's already-built bridge slice. cortex_ctx/dlpfc_wm/rel_* live on the pool at the
            # SAME names/sizes (the descriptor's spec_fn reuses THIS class's region build), and the attractor +
            # cue-monitor edges are installed by the pool's explicit_wiring_fn (pmem_explicit_wiring in the merge
            # framework -- build-time, both arms identical). This branch NEVER builds a bridge, edits sim/, or steps
            # the substrate: it only discovers index maps + edge references. The homeostat/plateau calibration
            # (subclasses) then re-homes onto this slice inside the pool's per-sequence read_isolation guard.
            shared.ensure_built()
            self.bridge = shared.bridge
            if int(np.asarray(self.bridge.region_manager.indices("cortex_ctx")).size) != n:
                raise ValueError(f"shared pool cortex_ctx size != n={n} (config drift in the pmem descriptor)")

        rm = self.bridge.region_manager
        bridge = self.bridge
        cidx = np.asarray(rm.indices("cortex_ctx"))
        didx = np.asarray(rm.indices("dlpfc_wm"))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        need = len(all_cortex_assemblies) * pattern_size
        if need > n:
            raise ValueError(f"n={n} too small for {len(all_cortex_assemblies)} assemblies x {pattern_size}")

        self._cpat = {}   # cortex assembly indices per name (device)
        self._dpat = {}   # dlpfc assembly indices per attractor-concept (device)
        # install the per-concept outer-product attractors (intentions + distractors), batched into ONE
        # c2d + ONE d2c call (each set_pathway_weights rebuilds the whole CSR; batching => 2 rebuilds total).
        c2d_pre, c2d_post, d2c_pre, d2c_post = [], [], [], []
        for i, name in enumerate(all_cortex_assemblies):
            p = perm[i * pattern_size:(i + 1) * pattern_size]
            self._cpat[name] = self.xp.asarray(cidx[p])
            if name in self._attractor_concepts:
                dpat = didx[p]
                self._dpat[name] = self.xp.asarray(dpat)
                cpat = cidx[p]
                c2d_pre.append(np.repeat(cpat, pattern_size)); c2d_post.append(np.tile(dpat, pattern_size))
                d2c_pre.append(np.repeat(dpat, pattern_size)); d2c_post.append(np.tile(cpat, pattern_size))
        c2d_pre = np.concatenate(c2d_pre).astype(np.int64); c2d_post = np.concatenate(c2d_post).astype(np.int64)
        d2c_pre = np.concatenate(d2c_pre).astype(np.int64); d2c_post = np.concatenate(d2c_post).astype(np.int64)
        ww = np.full(c2d_pre.size, np.float32(attractor_weight), np.float32)
        if shared is None:
            # standalone: install the attractor edges here. (SHARED: the pool's explicit_wiring_fn installed the
            # SAME edges at build time via the SAME perm/weights, so this slice already carries them.)
            bridge.set_pathway_weights("c2d", pre_indices=c2d_pre, post_indices=c2d_post, weights=ww, add_missing=True)
            bridge.set_pathway_weights("d2c", pre_indices=d2c_pre, post_indices=d2c_post, weights=ww, add_missing=True)

        # store the ATTRACTOR EDGES per attractor-concept so the latch-lesion can zero exactly its edges.
        self._attr_edges = {}
        for name in self._attractor_concepts:
            cp = self.B.to_host(self._cpat[name]).astype(np.int64)
            dp = self.B.to_host(self._dpat[name]).astype(np.int64)
            self._attr_edges[name] = (np.repeat(cp, pattern_size), np.tile(dp, pattern_size),
                                      np.repeat(dp, pattern_size), np.tile(cp, pattern_size))

        # install the cue-monitor COINCIDENCE edges: per action X, BOTH act_X.cortex->rel_X and
        # cue_X.cortex->rel_X (feedforward evidence). Tuned so neither alone crosses rel_X's ramp threshold.
        self._rel_idx = {a: np.asarray(rm.indices(f"rel_{a}"), dtype=np.int64) for a in self.actions}
        self._rel_all = self.xp.asarray(np.concatenate([self._rel_idx[a] for a in self.actions]))
        pre_all, post_all, w_all = [], [], []
        for a in self.actions:
            relX = self._rel_idx[a]
            actc = self.B.to_host(self._cpat[a]).astype(np.int64)           # the latched intention's cortex assembly
            cuec = self.B.to_host(self._cpat[f"cue_{a}"]).astype(np.int64)  # the cue's cortex assembly
            pre_all.append(np.repeat(actc, relX.size)); post_all.append(np.tile(relX, actc.size))
            w_all.append(np.full(actc.size * relX.size, np.float32(hold_to_rel_weight), np.float32))
            pre_all.append(np.repeat(cuec, relX.size)); post_all.append(np.tile(relX, cuec.size))
            w_all.append(np.full(cuec.size * relX.size, np.float32(cue_to_rel_weight), np.float32))
        if shared is None:
            # standalone: install the cue-monitor coincidence edges. (SHARED: installed by explicit_wiring_fn.)
            bridge.set_pathway_weights("cue_monitor", pre_indices=np.concatenate(pre_all),
                                       post_indices=np.concatenate(post_all), weights=np.concatenate(w_all),
                                       add_missing=True)
        if verbose:
            print(f"[pmem] latch loop n={n}, {len(self.actions)} actions, {len(self.distractors)} distractors, "
                  f"rel/action={n_rel}, hold_w={hold_to_rel_weight}, cue_w={cue_to_rel_weight}", flush=True)

    def _step(self, drive_idx=None, drive_pA=0.0):
        """One simulation step with the tonic rel bias ALWAYS applied (so the rel readout's operating point is
        the same in every phase -- write, hold, cue). Optionally drive one assembly. All rel firing is thus a
        genuine coincidence read against a fixed threshold offset."""
        cur = self.bridge.cp_external_input_current
        cur[:] = 0.0
        if drive_idx is not None:
            cur[drive_idx] = np.float32(drive_pA)
        cur[self._rel_all] = np.float32(self._rel_bias_pA)   # tonic hyperpolarizing bias on the rel readout
        self.bridge._run_one_simulation_step()

    # ---- holding: write a concept into WM (drive its assembly; the attractor sustains it) ----
    def _write(self, name, drive_pA=2500.0, stim=40, settle=15):
        drv = self._cpat[name]
        for _ in range(stim):
            self._step(drive_idx=drv, drive_pA=drive_pA)
        for _ in range(settle):
            self._step()

    def encode_intention(self, action):
        """Form the deferred intention: latch the action assembly (it self-sustains = the held intention)."""
        self._write(action)

    def intervening_turn(self, distractor):
        """One intervening conversational turn of UNRELATED content: write a distractor (real competing WM
        load), then read the held intentions + the rel monitors (cue ABSENT) over a no-drive window."""
        self._write(distractor)
        return self._read(window=20, cue=None)

    def _read(self, window=20, cue=None):
        """Run `window` steps (optionally driving one cue assembly) and return per-action rel firing rate and
        per-attractor-concept held firing rate, all from cp_firing_states. Cue present => the coincidence with
        any latched intention can ramp rel; cue None => cue-absent read (must stay silent unless nothing gates)."""
        cue_idx = self._cpat[cue] if cue is not None else None
        rel_acc = {a: 0.0 for a in self.actions}
        held_acc = {c: 0.0 for c in self._attractor_concepts}
        for _ in range(window):
            self._step(drive_idx=cue_idx, drive_pA=2500.0)
            fs = self.bridge.cp_firing_states
            for a in self.actions:
                rel_acc[a] += float(self.B.to_host(fs[self._rel_idx[a]]).sum())
            for c in self._attractor_concepts:
                held_acc[c] += float(self.B.to_host(fs[self._cpat[c]]).sum())
        rel = {a: rel_acc[a] / (self._n_rel * window) for a in self.actions}
        held = {c: held_acc[c] / (self._psize * window) for c in self._attractor_concepts}
        return {"rel": rel, "held": held}

    def present_cue(self, cue_action, window=30):
        """Present a cue (drive cue_{cue_action}.cortex): the cue-monitor coincidence with a latched matching
        intention ramps its rel accumulator over threshold = the intention FIRES."""
        return self._read(window=window, cue=f"cue_{cue_action}")

    def lesion_latch(self, action):
        """LATCH LESION: zero the intention's attractor edges (c2d+d2c) AND clear the network firing/membrane so
        the held assembly collapses and cannot re-sustain. The intention is destroyed at the substrate level."""
        c2d_pre, c2d_post, d2c_pre, d2c_post = self._attr_edges[action]
        z1 = np.zeros(c2d_pre.size, np.float32); z2 = np.zeros(d2c_pre.size, np.float32)
        self.bridge.set_pathway_weights("c2d", pre_indices=c2d_pre, post_indices=c2d_post, weights=z1, add_missing=False)
        self.bridge.set_pathway_weights("d2c", pre_indices=d2c_pre, post_indices=d2c_post, weights=z2, add_missing=False)
        self._reset_dynamics()

    def _reset_dynamics(self):
        """Clear membrane/recovery/firing/conductance/in-flight synaptic state so a lesion (or a fresh trial)
        starts from rest (mirrors SpikingSpreadingController._reset_wm)."""
        b = self.bridge
        for a in ("cp_firing_states", "cp_prev_firing_states"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = False
        for a in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda",
                  "cp_conductance_g_nmda_rise", "cp_refractory_timers", "cp_synapse_pulse_timers",
                  "cp_synapse_pulse_progress"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = 0
        if getattr(b, "cp_izh_vr", None) is not None and b.cp_membrane_potential_v is not None:
            b.cp_membrane_potential_v[:] = b.cp_izh_vr
        if getattr(b, "cp_recovery_variable_u", None) is not None:
            b.cp_recovery_variable_u[:] = 0.0


# --------------------------------------------------------------------------------------------------------
# One seed: run every condition on FRESH bridges (each condition builds its own PM so trials never contaminate).
# --------------------------------------------------------------------------------------------------------
def _new_pm(seed, n_distractors, **kw):
    actions = ["A", "B"]
    distractors = [f"d{i}" for i in range(n_distractors)]
    return ProspectiveMemory(actions, distractors, seed=seed, **kw)


def run_seed(seed, N, n_distractors, verbose=False, **kw):
    dists = [f"d{i}" for i in range(n_distractors)]
    inter = [dists[i % len(dists)] for i in range(N)]   # the N intervening-turn distractor sequence

    # --- condition 1: FIRE-ON-CUE (act_A, cue_A) + PERSISTENCE + NO-FIRE-BEFORE ---
    pm = _new_pm(seed, n_distractors, verbose=verbose, **kw)
    pm.encode_intention("A")
    held_trace, relbefore_trace = [], []
    for d in inter:
        r = pm.intervening_turn(d)
        held_trace.append(r["held"]["A"])
        relbefore_trace.append(r["rel"]["A"])
    cue_read = pm.present_cue("A")
    fireA = {"rel_A_on_cueA": cue_read["rel"]["A"], "rel_B_on_cueA": cue_read["rel"]["B"],
             "held_A_trace": held_trace, "rel_A_before_trace": relbefore_trace}

    # --- condition 2: symmetric FIRE-ON-CUE for act_B, cue_B (the monitor is action-specific both ways) ---
    pm = _new_pm(seed, n_distractors, verbose=False, **kw)
    pm.encode_intention("B")
    for d in inter:
        pm.intervening_turn(d)
    cue_read = pm.present_cue("B")
    fireB = {"rel_B_on_cueB": cue_read["rel"]["B"], "rel_A_on_cueB": cue_read["rel"]["A"]}

    # --- condition 3: WRONG-CUE (intention A latched, present cue_B) -> must NOT fire ---
    pm = _new_pm(seed, n_distractors, verbose=False, **kw)
    pm.encode_intention("A")
    for d in inter:
        pm.intervening_turn(d)
    wrong_read = pm.present_cue("B")
    wrongcue = {"rel_A_on_cueB": wrong_read["rel"]["A"], "rel_B_on_cueB": wrong_read["rel"]["B"]}

    # --- condition 4: NO-INTENTION anti-cheat (never latch; present cue_A) -> must NOT fire ---
    pm = _new_pm(seed, n_distractors, verbose=False, **kw)
    for d in inter:
        pm.intervening_turn(d)
    noint_read = pm.present_cue("A")
    noint = {"rel_A_on_cueA_no_intention": noint_read["rel"]["A"]}

    # --- condition 5: LATCH-LESION (latch A, hold N turns, zero the latch, present cue_A) -> forgotten ---
    pm = _new_pm(seed, n_distractors, verbose=False, **kw)
    pm.encode_intention("A")
    for d in inter:
        pm.intervening_turn(d)
    pre_lesion_held = pm._read(window=20, cue=None)["held"]["A"]
    pm.lesion_latch("A")
    post_lesion_held = pm._read(window=20, cue=None)["held"]["A"]   # verify the lesion HOLDS at measurement
    lesion_read = pm.present_cue("A")
    lesion = {"held_A_pre_lesion": pre_lesion_held, "held_A_post_lesion": post_lesion_held,
              "rel_A_on_cueA_lesioned": lesion_read["rel"]["A"]}

    # ---- per-seed pass evaluation (every clause) ----
    rel_before_max = max(fireA["rel_A_before_trace"]) if fireA["rel_A_before_trace"] else 0.0
    silent_pool = [
        rel_before_max,                             # no-fire-before
        fireA["rel_B_on_cueA"],                     # wrong action not fired on cue_A
        fireB["rel_A_on_cueB"],                     # wrong action not fired on cue_B
        wrongcue["rel_A_on_cueB"], wrongcue["rel_B_on_cueB"],  # wrong-cue silent
        noint["rel_A_on_cueA_no_intention"],        # no-intention silent
        lesion["rel_A_on_cueA_lesioned"],           # lesion silent
    ]
    max_silent = max(silent_pool)
    fire_min = min(fireA["rel_A_on_cueA"], fireB["rel_B_on_cueB"])
    held_min = min(held_trace) if held_trace else 0.0

    clauses = {
        "fire_on_cue": fire_min >= FIRE_THR,
        "persistence": held_min >= HOLD_FLOOR,
        "no_fire_before": rel_before_max <= SILENT_MAX,
        "no_fire_wrongcue": max(wrongcue["rel_A_on_cueB"], wrongcue["rel_B_on_cueB"]) <= SILENT_MAX,
        "no_intention_silent": noint["rel_A_on_cueA_no_intention"] <= SILENT_MAX,
        "lesion_holds": post_lesion_held <= LESION_HELD_MAX,
        "lesion_forgets": lesion["rel_A_on_cueA_lesioned"] <= SILENT_MAX,
        # separation: the FIRE must beat the largest silent read by the ratio AND clear the silent ceiling
        # itself -- so a degenerate near-zero fire (rel that never ramped) cannot "separate" from near-zero
        # silence. This makes a genuine release-amplitude failure (e.g. a hypo-excitable pool) fail honestly.
        "separation": (fire_min >= SEP_RATIO * max(max_silent, 1e-6)) and (fire_min >= SILENT_MAX),
    }
    passed = all(clauses.values())
    return {"seed": seed, "N": N, "passed": bool(passed), "clauses": clauses,
            "max_silent": max_silent, "fire_min": fire_min, "held_min": held_min,
            "fireA": fireA, "fireB": fireB, "wrongcue": wrongcue, "noint": noint, "lesion": lesion}


def _derisk(seeds, N, n_distractors, smoke=False, **kw):
    tag = "SMOKE" if smoke else "DE-RISK"
    print(f"PROSPECTIVE MEMORY [{tag}] -- spiking intention LATCH + BA10 cue-MONITOR; {len(seeds)} seed(s), "
          f"N={N} intervening turns, {n_distractors} distractors", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = run_seed(s, N, n_distractors, **kw); per.append(d)
            c = d["clauses"]
            fails = " ".join(k for k, v in c.items() if not v) or "ALL-PASS"
            print(f"  [seed {s}] pass={d['passed']} | fireA={d['fireA']['rel_A_on_cueA']:.3f} "
                  f"fireB={d['fireB']['rel_B_on_cueB']:.3f} | max_silent={d['max_silent']:.3f} "
                  f"held_min={d['held_min']:.3f} | lesion_held {d['lesion']['held_A_pre_lesion']:.3f}->"
                  f"{d['lesion']['held_A_post_lesion']:.3f} rel_les={d['lesion']['rel_A_on_cueA_lesioned']:.3f} "
                  f"| {fails}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        n_pass = sum(int(p["passed"]) for p in per)
        min_seeds = int(np.ceil(GO_MIN_SEEDS_FRAC * len(seeds)))
        go = bool(n_pass >= min_seeds) and not smoke
        agg = {clause: sum(int(p["clauses"][clause]) for p in per) for clause in per[0]["clauses"]}
        mean_fire = float(np.mean([min(p["fireA"]["rel_A_on_cueA"], p["fireB"]["rel_B_on_cueB"]) for p in per]))
        mean_silent = float(np.mean([p["max_silent"] for p in per]))

        # ATTRIBUTION (tools.lab): of the release on the correct cue, what fraction is owned by the INTACT
        # latch vs. what leaks through the cue path after the latch is lesioned? A high fraction = the fire is
        # genuinely gated by the held intention (the coincidence), not a cue passthrough.
        mean_fireA_intact = float(np.mean([p["fireA"]["rel_A_on_cueA"] for p in per]))
        mean_rel_lesioned = float(np.mean([p["lesion"]["rel_A_on_cueA_lesioned"] for p in per]))
        latch_share = attributable_to("latch on release (rel_A on cue_A: intact vs latch-lesioned)",
                                       mean_fireA_intact, mean_rel_lesioned)
        # PRECONDITIONS that must hold for the verdict to be interpretable (tools.verdict.Verdict): the
        # INSTRUMENT is valid (persistence + the silence controls + the lesion actually collapsing the latch).
        # The fire-amplitude clause is the SUBSTANTIVE result, not a precondition.
        mean_pre_les = float(np.mean([p["lesion"]["held_A_pre_lesion"] for p in per]))
        mean_post_les = float(np.mean([p["lesion"]["held_A_post_lesion"] for p in per]))
        vd = Verdict("pmem_intention_latch")
        vd.require("persistence: held survives all N intervening turns (per-seed)", agg["persistence"],
                   expect=lambda x: x == len(seeds))
        vd.require("no-fire-before: silent on every intervening turn (per-seed)", agg["no_fire_before"],
                   expect=lambda x: x == len(seeds))
        vd.require("wrong-cue: monitor stays silent (per-seed)", agg["no_fire_wrongcue"],
                   expect=lambda x: x == len(seeds))
        vd.require("no-intention: cue alone cannot fire (per-seed)", agg["no_intention_silent"],
                   expect=lambda x: x == len(seeds))
        vd.reaches("latch-lesion collapses the held assembly", mean_pre_les, mean_post_les)
        vd.control("release: intact vs latch-lesioned", mean_fireA_intact, mean_rel_lesioned,
                   min_separation=0.05)
        vd.disabled("STDP / Hebbian / STP / OU-noise",
                    "clean-hold WM config; the release readout is gated by a CONSTANT tonic bias — a proxy "
                    "for tonic inhibition / intrinsic-plasticity gain control (the named per-pool homeostatic "
                    "residual behind the fire-amplitude spread)")
        decided = vd.decide(go)

        if smoke:
            verdict = (f"SMOKE OK -- the mechanism RUNS end-to-end and every condition is live/measured "
                       f"({n_pass}/{len(seeds)} seed passed all clauses; fire~{mean_fire:.3f} vs "
                       f"max-silent~{mean_silent:.3f}). Not a GO claim; run --derisk for the 6-seed verdict.")
        elif go:
            verdict = (
                f"GO -- prospective memory works on spikes. A PFC intention LATCH (a self-sustaining cortex<->"
                f"dlpfc attractor assembly) HOLDS a deferred intention through N={N} intervening distractor turns "
                f"(held stays above the floor), and a BA10-style spiking CUE-MONITOR (an NMDA-recurrent "
                f"coincidence accumulator fed by BOTH the held-intention and the cue assemblies) RELEASES the "
                f"intention ONLY when the RIGHT cue appears: fire~{mean_fire:.3f} on the correct cue vs "
                f"max-silent~{mean_silent:.3f} across (a) every intervening turn before the cue, (b) the WRONG "
                f"cue, (c) NO intention ever latched (the cue alone cannot fire -> the coincidence is real, not "
                f"a cue passthrough), and (d) a LATCH-LESION (zero the attractor -> the held assembly collapses "
                f"at measurement -> the intention is FORGOTTEN and the cue does nothing). {n_pass}/{len(seeds)} "
                f"seeds pass every clause. All reads are cp_firing_states. HOST-SCAFFOLD (flagged): the "
                f"cue->action CONTENT binding is installed synaptically (like every SpikingLoopContextBuffer "
                f"attractor); the mechanism (hold-across-turns + cue-gated release) is brain-based -- the "
                f"follow-on LEARNS the binding via one-shot Hebbian potentiation at intention-formation.")
        else:
            fails = {clause: agg[clause] for clause in agg if agg[clause] < len(seeds)}
            verdict = (f"BOUNDARY -- {n_pass}/{len(seeds)} seeds passed all clauses (need {min_seeds}). "
                       f"per-clause pass counts: {agg}. fire~{mean_fire:.3f} vs max-silent~{mean_silent:.3f}. "
                       f"The failing clause(s) {sorted(fails)} NAME the residual (rel weights / N / thresholds / "
                       f"attractor persistence) as the next single-variable de-risk. Do NOT force GO.")
    else:
        go = False; verdict = f"ERROR -- {err}"; agg = mean_fire = mean_silent = n_pass = None
        decided = latch_share = None

    summary = {
        "probe": "pmem_intention_latch", "verdict": verdict, "go": bool(go) if err is None else False,
        "task": ("prospective memory: a spiking PFC intention LATCH (self-sustaining cortex<->dlpfc attractor) "
                 "holds a deferred intention across N intervening distractor turns; a BA10-style spiking "
                 "cue-MONITOR (NMDA-recurrent coincidence accumulator fed by the held-intention AND the cue) "
                 "releases it ONLY on the right cue; anti-cheats: no-fire-before, wrong-cue, no-intention; "
                 "lesion: zero the latch -> forgotten. All reads cp_firing_states. CPU 6-seed."),
        "gate": {"FIRE_THR": FIRE_THR, "SILENT_MAX": SILENT_MAX, "HOLD_FLOOR": HOLD_FLOOR,
                 "LESION_HELD_MAX": LESION_HELD_MAX, "SEP_RATIO": SEP_RATIO,
                 "GO_MIN_SEEDS_FRAC": GO_MIN_SEEDS_FRAC},
        "N_intervening": N, "n_distractors": n_distractors, "seeds": list(seeds),
        "n_pass": n_pass, "per_clause_pass_counts": agg,
        "mean_fire": mean_fire, "mean_max_silent": mean_silent,
        "latch_share_of_release": latch_share,   # attributable_to: fraction of the release owned by the intact latch
        "preconditions": (decided or {}).get("preconditions"),   # instrument preconditions travel with the verdict
        "disabled_processes": (decided or {}).get("disabled_processes"),
        "verdict_status": (decided or {}).get("status"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_seed": per,
        "HOST_SCAFFOLD_NOTE": ("BRAIN-BASED: the latch (attractor persistence), the cue-monitoring (coincidence "
                               "integration) and the release (accumulator crossing threshold) are all spiking; "
                               "every read is cp_firing_states. FLAGGED host scaffold: the cue->action CONTENT "
                               "binding (which cue releases which action) is installed synaptically "
                               "(outer-product edges), exactly as every SpikingLoopContextBuffer attractor is. "
                               "The mechanism is brain-based; the content binding's LEARNED version (one-shot "
                               "Hebbian potentiation of cue->action at intention-formation, Gollwitzer "
                               "implementation-intention) is the named follow-on."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + "=" * 118, flush=True)
    print(f"[pmem] VERDICT: {verdict}", flush=True)
    print(f"[pmem] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and (go or smoke)) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=None,
                    help="single-seed convenience (overrides --seeds); the substrate seed -> cfg.seed")
    ap.add_argument("--N", type=int, default=5, help="number of intervening distractor turns")
    ap.add_argument("--n-distractors", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="1 seed, N=3 -- proves it RUNS + measures every condition")
    ap.add_argument("--derisk", action="store_true")
    # mechanism knobs (for single-variable residual de-risks)
    ap.add_argument("--hold-to-rel-weight", type=float, default=3.2)
    ap.add_argument("--cue-to-rel-weight", type=float, default=4.2)
    ap.add_argument("--rel-recurrent-weight", type=float, default=0.10)
    ap.add_argument("--rel-bias-pA", type=float, default=-1050.0)
    ap.add_argument("--n-rel", type=int, default=60)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--pattern-size", type=int, default=40)
    a = ap.parse_args()

    seeds = [a.seed] if a.seed is not None else a.seeds
    kw = dict(hold_to_rel_weight=a.hold_to_rel_weight, cue_to_rel_weight=a.cue_to_rel_weight,
              rel_recurrent_weight=a.rel_recurrent_weight, rel_bias_pA=a.rel_bias_pA,
              n_rel=a.n_rel, n=a.n, pattern_size=a.pattern_size)
    if a.smoke:
        return _derisk([seeds[0]], N=3, n_distractors=min(3, a.n_distractors), smoke=True, verbose=True, **kw)
    return _derisk(seeds, N=a.N, n_distractors=a.n_distractors, smoke=False, **kw)


if __name__ == "__main__":
    raise SystemExit(main())
