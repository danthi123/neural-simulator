"""Graded affect (board #81) — a GRADED continuous-attractor affect substrate (Koulakov-Goldman bistable LADDER)
that reads the interoceptive body-state as a SMOOTH valence x arousal, NOT a bistable +/- switch.

THE RESIDUAL (from board #49, `2026-08-19-embodied-affect-interoception-GO.md`). Embodied affect is a 6/6 GO: a
simulated body-state drives the neural affect attractor through synapses. BUT the affect substrate is the P0.3
bistable NMDA opponent LATCH, so the body is read as a two-state +/- SWITCH: mood latches to +-0.08 (a sign flip
near the set-point) and felt-arousal is on/off ignition (gradedness Pearson 0.70, only 1/6 seeds >=0.8). Real
core-affect is a GRADED valence x arousal circumplex.

THE NAMED SURPASS (research gate `2026-08-08-graded-affect-persistence-...-bistable-ladder-robust-integrator.md`).
A plain continuous line attractor DRIFTS (Wave-1: held range 0.003, collapses to a point) and a single bistable
pool SATURATES (P0.3 latch). The different-in-kind mechanism is the Koulakov 2002 / Goldman-Seung 2003 ROBUST
DISCRETE INTEGRATOR: hold graded value as the INTEGER COUNT of independently-latched bistable sub-pools recruited
at staggered thresholds. Many stable graded levels; drift-robust because each sub-pool sits in its OWN on/off
basin (there is no continuum to drift along); graded-persistent because each latch self-sustains via slow NMDA.

WHAT IS BUILT (a runner-level LADDER; the P0.3 `aff()` bistable-pool primitive reused, NO sim/ edit):
  - For each affect sign, a LADDER of N_L independent self-recurrent NMDA bistable sub-pools (the exact P0.3
    `aff()` factory at the proven recur_weight): affect_vplus_L0..L{NL-1}, affect_vminus_L0..L{NL-1},
    affect_arousal_L0..L{NL-1}. NO intra-sign lateral inhibition (the load-bearing Koulakov rule: any within-sign
    inhibition -> WTA -> the 2-level latch returns).
  - STAGGERED RECRUITMENT is purely SYNAPTIC: the interoceptive->sub-pool weight decreases along the ladder
    (g_k = linspace(1, G_MIN, N_L)), so a stronger body signal crosses more sub-pools' ignition thresholds and
    latches more of them. This keeps the #49 honesty guard EXACT: the affect pools receive ZERO direct external
    current (asserted every step) -- the body reaches affect ONLY through synapses, and the staggering lives in the
    synapse, not a host tonic bias.
  - The interoceptive body-state channel is REUSED from #49 unchanged: 3 spiking interoceptive relay pools
    (intero_comfort <- comfort=h, intero_discomfort <- discomfort=1-h, intero_arousal <- arousal=a) driven by a
    host body-current, each projecting SYNAPTICALLY (AMPA, gated by intero_out) onto its ladder.
  - Aggregate-only OPPONENT cross-inhibition (optional, weak; Namburi-Tye at the ladder AGGREGATE, never between
    sub-pools of one sign): a V+ summary interneuron driven by all V+ sub-pools inhibits all V- sub-pools, and
    vice-versa. Default weak so the graded PERSISTENCE (both ladders partially latched) is not resolved to WTA.
  - The felt STATE is the ladder's OWN population read: mood = rate(all V+ sub-pools) - rate(all V- sub-pools);
    felt_arousal = rate(all arousal sub-pools). Off cp_firing_states. NEVER a host formula / count / argmax.

ANTI-CHEATS (they ARE the result):
  (1) GRADED, not a switch -- sweep the body-state, read the ladder: Pearson(body-level, felt) > 0.8 across the
      sweep on >=5/6 seeds (vs the 0.70 / 1-of-6 baseline), with MORE than 2 resolvable levels (an SNR count from
      the across-seed SD). BOTH channels (valence + felt-arousal).
  (2) PERSISTENCE (robust graded integrator) -- establish several DIFFERENT drive levels, remove the drive, and
      the held state TRACKS the level (Pearson(level, held) >= 0.8, held-range real) instead of decaying to ONE
      default. NMDA-off decays to ~0 (persistence is the slow-NMDA latch, not the tonic drive).
  (3) EMBODIMENT preserved -- the body still moves affect (sweep, anti-cheat 1) AND cutting the
      interoceptive->affect synapses (intero_out gate=0) collapses the coupling (range -> ~0, |corr| < 0.3;
      tools.lab.attributable_to ~1.0). The felt read is neural (cp_firing_states), never a host formula.
  (4) 6 seeds (42 43 44 100 101 102), per-seed + pooled, deterministic (cfg.seed set -> substrate seeded).

DISCIPLINE: SIM_BACKEND=numpy (CPU lane), reuse-by-import (the P0.3 `aff()` bistable primitive + operating point),
NO sim/ edit (only additive BrainRegion/RegionPathway config). cfg.seed per seed.

Run (smoke -- 1 seed: determinism + operating-point sweep + a graded/persistence sanity read):
  SIM_BACKEND=numpy python -u -m research.runners._graded_affect_attractor_derisk --smoke
Run (6-seed battery):
  SIM_BACKEND=numpy python -u -m research.runners._graded_affect_attractor_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402  (passthrough on numpy)
# reuse-by-import: the EXACT P0.3 affect operating-point constants (the proven bistable-latch regime for one pool).
from research.runners._affect_state_region_derisk import (  # noqa: E402
    DEFAULT_RECUR_WEIGHT, RECUR_DENSITY, N_AFF,
)
from tools.lab import attributable_to  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "graded_affect" / "_graded_affect_attractor_6seed.json"

# ---- ladder + interoceptive constants (operating point calibrated by the smoke) ------------------------------
N_L = 6                     # sub-pools per ladder (=> up to N_L+1 graded levels per sign)
N_SUB = N_AFF               # neurons per sub-pool (=50, the proven P0.3 latch size)
RECUR_WEIGHT = DEFAULT_RECUR_WEIGHT   # per sub-pool self-attractor weight (=22, the proven NMDA-dependent regime)
G_MIN = 0.20                # weakest interoceptive->sub-pool gain along the ladder (g_k = linspace(1, G_MIN, N_L)):
                            # a wide gain spread + a near-threshold operating point is what STAGGERS the ignition so
                            # a stronger body signal latches PROGRESSIVELY more sub-pools (calibrated by the smoke).
N_INT = 40                  # neurons per interoceptive relay pool (as #49)
I_BODY_PA = 260.0           # afferent current at full body signal (as #49; the graded regime, calibrated by the smoke)
W_INT = 6.0                 # base interoceptive->sub-pool synaptic weight (staggered by g_k) -- LOW so the weakest
                            # sub-pools sit near threshold and recruit only at high body (the graded staircase)
DENS_INT = 0.6              # interoceptive -> sub-pool projection density
XINH_N = 12                 # aggregate opponent interneuron count per sign
XINH_EXC_W = 6.0            # sub-pool -> its sign's aggregate interneuron
XINH_INH_W = 0.0            # aggregate interneuron -> the OTHER sign's sub-pools. DEFAULT 0: the body input is ALREADY
                            # opponent-structured (comfort=h vs discomfort=1-h are anti-correlated), so mood grades
                            # as the DIFFERENCE of two independently-latching ladders; adding cross-inhibition forces
                            # WTA that (a) sharpens valence toward a switch and (b) resolves the graded middle to one
                            # sign during unclamped hold -- it DEGRADES both gradedness and graded persistence.
INTERO_GATE = "intero_out"  # one runtime transmission gate over ALL interoceptive->ladder projections (the lesion)


def ladder_gains(n_l=N_L, g_min=G_MIN):
    return np.linspace(1.0, float(g_min), int(n_l))


# =============================================================================================================
# The graded-affect brain: 3 LADDERS of bistable NMDA sub-pools + 3 interoceptive relays projecting into them.
# =============================================================================================================
class GradedAffectBrain:
    def __init__(self, seed, nmda_on=True, recur_weight=RECUR_WEIGHT, n_l=N_L, n_sub=N_SUB, g_min=G_MIN,
                 w_int=W_INT, xinh_inh_w=XINH_INH_W, ou_pA=8.0):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig
        from sim.regions import BrainRegion, RegionPathway

        self.seed = int(seed)
        self.nmda_on = bool(nmda_on)
        self.n_l = int(n_l)
        self.n_sub = int(n_sub)
        self.gains = ladder_gains(n_l, g_min)

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.enable_neuromodulator_subsystem = False   # embodied drive is synaptic (afferent current), no lexical bus
        cfg.enable_nmda = bool(nmda_on)               # NMDA-OFF control = the whole NMDA block skipped
        cfg.nmda_ratio = 0.5                          # dlPFC WM-latch precedent (as P0.3)
        cfg.nmda_tau_decay = 100.0
        cfg.dt_ms = 1.0
        cfg.seed = int(seed)                          # SEEDS THE SUBSTRATE (NOT actual_seed_used)
        cfg.stdp_w_max = 400.0
        cfg.hebbian_max_weight = 400.0
        cfg.enable_stdp = False
        cfg.enable_reward_modulation = False
        cfg.enable_hebbian_learning = False
        cfg.enable_homeostasis = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_structural_plasticity = False
        cfg.enable_ou_process = True
        cfg.ou_std_current_pA = float(ou_pA)
        cfg.enable_parameter_heterogeneity = False
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1

        RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
        FS = "IZH2007_FS_CORTICAL_INTERNEURON"

        def aff(name):   # the EXACT P0.3 self-recurrent NMDA bistable pool primitive
            return BrainRegion(name=name, n_neurons=int(n_sub), exc_fraction=1.0, internal_density=RECUR_DENSITY,
                               exc_weight_mean=float(recur_weight), inh_weight_mean=0.0, weight_jitter=0.05,
                               plastic_internal=False, izh_neuron_type=RS, enable_nmda=bool(nmda_on))

        def intero_pool(name):   # a pure afferent relay (no recurrence), as #49
            return BrainRegion(name=name, n_neurons=N_INT, exc_fraction=1.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.05,
                               plastic_internal=False, izh_neuron_type=RS, enable_nmda=False)

        def fs_pool(name, n):
            return BrainRegion(name=name, n_neurons=int(n), exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                               plastic_internal=False, izh_neuron_type=FS)

        self.vplus = [f"affect_vplus_L{k}" for k in range(self.n_l)]
        self.vminus = [f"affect_vminus_L{k}" for k in range(self.n_l)]
        self.arousal = [f"affect_arousal_L{k}" for k in range(self.n_l)]
        ladder_regions = [aff(n) for n in (self.vplus + self.vminus + self.arousal)]
        opp_regions = [fs_pool("inh_plus", XINH_N), fs_pool("inh_minus", XINH_N)]
        intero_regions = [intero_pool("intero_comfort"), intero_pool("intero_discomfort"),
                          intero_pool("intero_arousal")]

        pathways = []
        # interoceptive -> ladder, staggered by g_k (the synaptic recruitment staircase), gated by INTERO_GATE.
        for src, pools in (("intero_comfort", self.vplus), ("intero_discomfort", self.vminus),
                           ("intero_arousal", self.arousal)):
            for k, pool in enumerate(pools):
                pathways.append(RegionPathway(from_region=src, to_region=pool, density=DENS_INT,
                                              weight_mean=float(w_int) * float(self.gains[k]), weight_jitter=0.1,
                                              plastic=False, transmission_gate=INTERO_GATE))
        # aggregate-only opponent cross-inhibition (Namburi-Tye at the AGGREGATE; NEVER intra-sign).
        if xinh_inh_w > 0.0:
            for pool in self.vplus:
                pathways.append(RegionPathway(from_region=pool, to_region="inh_plus", density=0.5,
                                              weight_mean=XINH_EXC_W, weight_jitter=0.1, plastic=False))
            for pool in self.vminus:
                pathways.append(RegionPathway(from_region="inh_plus", to_region=pool, density=0.6,
                                              weight_mean=float(xinh_inh_w), weight_jitter=0.1, plastic=False,
                                              receptor="gaba_a"))
                pathways.append(RegionPathway(from_region=pool, to_region="inh_minus", density=0.5,
                                              weight_mean=XINH_EXC_W, weight_jitter=0.1, plastic=False))
            for pool in self.vplus:
                pathways.append(RegionPathway(from_region="inh_minus", to_region=pool, density=0.6,
                                              weight_mean=float(xinh_inh_w), weight_jitter=0.1, plastic=False,
                                              receptor="gaba_a"))

        cfg.brain_regions = ladder_regions + opp_regions + intero_regions
        cfg.region_pathways = pathways

        self._bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                        runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self._bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self._bridge._initialize_simulation_data(called_from_playback_init=False)
        self._idx = {n: np.asarray(v, dtype=np.int64)
                     for n, v in self._bridge.region_manager.region_indices_dict().items()}
        self._vplus_idx = np.concatenate([self._idx[n] for n in self.vplus])
        self._vminus_idx = np.concatenate([self._idx[n] for n in self.vminus])
        self._arousal_idx = np.concatenate([self._idx[n] for n in self.arousal])
        # ANTI-CHEAT guard: the affect (ladder) pools must be reachable ONLY via synapses -> never a direct current.
        self._affect_idx = np.concatenate([self._vplus_idx, self._vminus_idx, self._arousal_idx])
        self._intero_idx = {"comfort": self._idx["intero_comfort"], "discomfort": self._idx["intero_discomfort"],
                            "arousal": self._idx["intero_arousal"]}

    def reset(self):
        self._bridge._initialize_simulation_data(called_from_playback_init=False)

    def _pool_rate(self, counts, idx_list, n_steps):
        n_neurons = sum(int(self._idx[n].size) for n in idx_list)
        tot = sum(counts[n] for n in idx_list)
        return tot / (n_neurons * max(1, n_steps))


def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# =============================================================================================================
# Reading the ladder under a body-state (the felt read is rate(ladder) off cp_firing_states -- never a host formula)
# =============================================================================================================
def read_body(brain, h, a, i_body=I_BODY_PA, settle=60, establish=250, read=120,
              lesion_gate=False, silence=False, hold_ms=0):
    """Apply a body-state (h, a), let the interoceptive relays drive the ladders, and read the settled felt state.
    hold_ms > 0: after `establish` with the drive ON, remove the body drive (afferent current -> 0) for hold_ms and
    read the HELD state over the read window -- the graded-persistence probe. Returns a dict of felt rates."""
    b = brain._bridge
    brain.reset()
    b.set_transmission_gate(INTERO_GATE, 0.0 if lesion_gate else 1.0)   # AFTER reset (reset restores gates to 1.0)

    comfort, discomfort = float(np.clip(h, 0, 1)), float(np.clip(1.0 - h, 0, 1))
    arousal = float(np.clip(a, 0, 1))
    i_comfort = 0.0 if silence else i_body * comfort
    i_discomfort = 0.0 if silence else i_body * discomfort
    i_arousal = 0.0 if silence else i_body * arousal

    all_pools = brain.vplus + brain.vminus + brain.arousal
    rec = tuple(all_pools) + ("intero_comfort", "intero_discomfort", "intero_arousal")
    counts = {r: 0.0 for r in rec}
    total = int(settle + establish + hold_ms + read)
    # gradedness read (hold_ms==0): drive stays ON through the read window -> the DRIVEN state.
    # persistence read (hold_ms>0): drive ON only during establish, then removed -> the HELD state after drive-off.
    drive_end = int(settle + establish) if hold_ms > 0 else total
    read_start = int(total - read)
    for t in range(total):
        b.cp_external_input_current[:] = 0.0
        if settle <= t < drive_end:                # body applied during the drive window
            b.cp_external_input_current[brain._intero_idx["comfort"]] = np.float32(i_comfort)
            b.cp_external_input_current[brain._intero_idx["discomfort"]] = np.float32(i_discomfort)
            b.cp_external_input_current[brain._intero_idx["arousal"]] = np.float32(i_arousal)
        # ANTI-CHEAT: the affect pools NEVER get a direct host current -- body reaches them only via synapses.
        assert float(np.abs(to_host(b.cp_external_input_current)[brain._affect_idx]).max()) == 0.0, \
            "affect pools received a direct external current -- the body->affect path must be synaptic"
        b._run_one_simulation_step()
        if t >= read_start:
            fs = to_host(b.cp_firing_states)
            for r in rec:
                counts[r] += float(fs[brain._idx[r]].sum())

    mood = brain._pool_rate(counts, brain.vplus, read) - brain._pool_rate(counts, brain.vminus, read)
    felt_arousal = brain._pool_rate(counts, brain.arousal, read)
    return {"mood": mood, "felt_arousal": felt_arousal,
            "vplus_rate": brain._pool_rate(counts, brain.vplus, read),
            "vminus_rate": brain._pool_rate(counts, brain.vminus, read),
            "intero_comfort": counts["intero_comfort"] / (N_INT * read),
            "intero_discomfort": counts["intero_discomfort"] / (N_INT * read),
            "intero_arousal": counts["intero_arousal"] / (N_INT * read),
            "comfort": comfort, "discomfort": discomfort, "arousal_body": arousal}


# A level separation must clear BOTH the readout noise AND a mechanism floor -- otherwise a SATURATED single-pool
# latch (its "on" plateau jitters by ~0.002 of readout precision) is over-counted as many "levels". The floor is
# half of one sub-pool's contribution to the population rate: peak pop-rate ~0.09, N_L=6 -> one sub-pool ~= 0.015,
# half ~= 0.0075. This gives the metric TEETH: a single-pool switch reads ~2 levels, the ladder reads its true count.
MIN_LEVEL_STEP = 0.0075


def resolvable_levels(mean_curve, sd_curve, min_step=MIN_LEVEL_STEP):
    """Count distinguishable states along a sweep: greedily walk the values sorted ascending and open a new level
    only when the value exceeds the current level's representative by more than max(combined 1-sigma noise,
    min_step). The min_step floor is what separates a genuine graded ladder from a saturated latch's readout jitter.
    Returns the number of resolvable levels."""
    order = np.argsort(mean_curve)
    m = np.asarray(mean_curve, float)[order]
    s = np.asarray(sd_curve, float)[order]
    s = np.maximum(s, 1e-6)
    levels = 1
    rep_m, rep_s = m[0], s[0]
    for i in range(1, len(m)):
        if (m[i] - rep_m) > max(rep_s + s[i], float(min_step)):
            levels += 1
            rep_m, rep_s = m[i], s[i]
    return int(levels)


# =============================================================================================================
# One seed = gradedness (valence + arousal) + persistence + embodiment lesion
# =============================================================================================================
def run_seed(seed, i_body=I_BODY_PA, w_int=W_INT, n_pts=11, persist_hold_ms=350, n_noise=4, xinh_inh_w=XINH_INH_W,
             ou_pA=8.0):
    t0 = time.time()
    brain = GradedAffectBrain(seed, nmda_on=True, w_int=w_int, xinh_inh_w=xinh_inh_w, ou_pA=ou_pA)

    # ---- (1) GRADED VALENCE sweep: vary homeostasis h (arousal 0), read mood. INTACT / LESION / SILENCE. ----
    hs = np.linspace(0.0, 1.0, int(n_pts))
    val_intact = [read_body(brain, h, 0.0, i_body) for h in hs]
    val_lesion = [read_body(brain, h, 0.0, i_body, lesion_gate=True) for h in hs]
    mood_intact = [r["mood"] for r in val_intact]
    mood_lesion = [r["mood"] for r in val_lesion]
    corr_h_mood = _corr(hs, mood_intact)
    corr_h_mood_les = _corr(hs, mood_lesion)
    mood_range = float(max(mood_intact) - min(mood_intact))
    mood_range_les = float(max(mood_lesion) - min(mood_lesion))

    # ---- (1) GRADED AROUSAL sweep: vary bodily arousal a (h=0.5), read felt arousal. INTACT / LESION. ----
    avals = np.linspace(0.0, 1.0, int(n_pts))
    ar_intact = [read_body(brain, 0.5, a, i_body) for a in avals]
    ar_lesion = [read_body(brain, 0.5, a, i_body, lesion_gate=True) for a in avals]
    felt_intact = [r["felt_arousal"] for r in ar_intact]
    felt_lesion = [r["felt_arousal"] for r in ar_lesion]
    corr_a_felt = _corr(avals, felt_intact)
    corr_a_felt_les = _corr(avals, felt_lesion)
    felt_range = float(max(felt_intact) - min(felt_intact))
    felt_range_les = float(max(felt_lesion) - min(felt_lesion))

    # per-seed within-condition read noise (repeat one mid body-state): the SNR floor for the resolvable-levels count
    mid_mood = [read_body(brain, 0.75, 0.0, i_body)["mood"] for _ in range(int(n_noise))]
    mid_felt = [read_body(brain, 0.5, 0.6, i_body)["felt_arousal"] for _ in range(int(n_noise))]
    mood_read_sd = float(np.std(mid_mood)) if len(mid_mood) > 1 else 0.0
    felt_read_sd = float(np.std(mid_felt)) if len(mid_felt) > 1 else 0.0

    # ---- (2) PERSISTENCE (graded robust integrator): establish several arousal levels, drive-OFF, read HELD. ----
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    held = [read_body(brain, 0.5, L, i_body, hold_ms=persist_hold_ms)["felt_arousal"] for L in levels]
    corr_level_held = _corr(levels, held)
    held_range = float(max(held) - min(held))
    # NMDA-OFF control: the same establish+hold with NMDA disabled must decay to ~0 (persistence is the NMDA latch).
    brain_off = GradedAffectBrain(seed, nmda_on=False, w_int=w_int, xinh_inh_w=xinh_inh_w, ou_pA=ou_pA)
    held_off = [read_body(brain_off, 0.5, L, i_body, hold_ms=persist_hold_ms)["felt_arousal"] for L in (0.6, 1.0)]
    held_off_max = float(max(held_off))
    # graded valence persistence too (establish comfort levels, drive-off, read held mood):
    vlevels = [0.0, 0.25, 0.5, 0.75, 1.0]
    held_mood = [read_body(brain, h, 0.0, i_body, hold_ms=persist_hold_ms)["mood"] for h in vlevels]
    corr_vlevel_held = _corr(vlevels, held_mood)
    held_mood_range = float(max(held_mood) - min(held_mood))

    # ---- (3) EMBODIMENT: interoceptive pools genuinely encode the body; attribution of the coupling. ----
    corr_comfort_enc = _corr([r["comfort"] for r in val_intact], [r["intero_comfort"] for r in val_intact])
    corr_arousal_enc = _corr(avals, [r["intero_arousal"] for r in ar_intact])
    intero_owns_valence = attributable_to("intero_owns_valence(range intact vs lesion)", mood_range, mood_range_les)
    intero_owns_arousal = attributable_to("intero_owns_arousal(range intact vs lesion)", felt_range, felt_range_les)

    row = {
        "seed": int(seed), "i_body": float(i_body), "w_int": float(w_int), "xinh_inh_w": float(xinh_inh_w),
        "n_l": brain.n_l, "n_sub": brain.n_sub,
        # gradedness
        "corr_h_mood": corr_h_mood, "corr_h_mood_lesion": corr_h_mood_les,
        "corr_a_felt": corr_a_felt, "corr_a_felt_lesion": corr_a_felt_les,
        "mood_range": mood_range, "mood_range_lesion": mood_range_les,
        "felt_range": felt_range, "felt_range_lesion": felt_range_les,
        "mood_read_sd": mood_read_sd, "felt_read_sd": felt_read_sd,
        "hs": [float(x) for x in hs], "mood_curve_intact": [float(x) for x in mood_intact],
        "mood_curve_lesion": [float(x) for x in mood_lesion],
        "avals": [float(x) for x in avals], "felt_curve_intact": [float(x) for x in felt_intact],
        "felt_curve_lesion": [float(x) for x in felt_lesion],
        # persistence
        "persist_levels": levels, "persist_held_felt": [float(x) for x in held],
        "corr_level_held": corr_level_held, "held_range": held_range,
        "held_off_max": held_off_max,
        "persist_vlevels": vlevels, "persist_held_mood": [float(x) for x in held_mood],
        "corr_vlevel_held": corr_vlevel_held, "held_mood_range": held_mood_range,
        # embodiment
        "corr_comfort_enc": corr_comfort_enc, "corr_arousal_enc": corr_arousal_enc,
        "intero_owns_valence_frac": intero_owns_valence, "intero_owns_arousal_frac": intero_owns_arousal,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    # per-seed resolvable-levels (uses the single-seed read-noise SD across the whole curve)
    row["mood_levels_seed"] = resolvable_levels(mood_intact, [mood_read_sd] * len(mood_intact))
    row["felt_levels_seed"] = resolvable_levels(felt_intact, [felt_read_sd] * len(felt_intact))
    print(f"  [seed {seed}] VAL corrH {corr_h_mood:+.2f} range {mood_range:.3f} lvls {row['mood_levels_seed']} "
          f"(les corr {corr_h_mood_les:+.2f} range {mood_range_les:.3f}) | AROU corrA {corr_a_felt:+.2f} "
          f"range {felt_range:.3f} lvls {row['felt_levels_seed']} (les range {felt_range_les:.3f}) | "
          f"PERSIST held corr {corr_level_held:+.2f} range {held_range:.3f} off {held_off_max:.3f} | "
          f"enc c/a {corr_comfort_enc:+.2f}/{corr_arousal_enc:+.2f} ({row['elapsed_seconds']}s)", flush=True)
    return row


# =============================================================================================================
# LADDER-IS-LOAD-BEARING control (the research gate's different-in-kind proof): a matched single bistable pool
# (no ladder) recruits ALL-OR-NONE and SATURATES -> few resolvable levels + small range (the P0.3 latch), while
# the ladder recruits progressively. Run at the same operating point; the ladder must resolve MORE levels.
# =============================================================================================================
def single_pool_control(seed, i_body=I_BODY_PA, w_int=W_INT, n_pts=11, n_sub_total=N_L * N_SUB, ou_pA=8.0):
    """n_l=1 single pool of n_sub_total neurons (matched TOTAL neuron count) driven by the same interoceptive
    channel at the same operating point. Returns its arousal/valence resolvable-levels + range."""
    brain = GradedAffectBrain(seed, nmda_on=True, n_l=1, n_sub=int(n_sub_total), w_int=w_int, xinh_inh_w=0.0,
                              ou_pA=ou_pA)
    avals = np.linspace(0.0, 1.0, int(n_pts)); hs = np.linspace(0.0, 1.0, int(n_pts))
    felt = [read_body(brain, 0.5, a, i_body)["felt_arousal"] for a in avals]
    mood = [read_body(brain, h, 0.0, i_body)["mood"] for h in hs]
    nz_a = [read_body(brain, 0.5, 0.6, i_body)["felt_arousal"] for _ in range(4)]
    nz_m = [read_body(brain, 0.75, 0.0, i_body)["mood"] for _ in range(4)]
    sd_a, sd_m = float(np.std(nz_a)), float(np.std(nz_m))
    return {
        "arousal_levels": resolvable_levels(felt, [sd_a] * len(felt)),
        "valence_levels": resolvable_levels(mood, [sd_m] * len(mood)),
        "arousal_range": float(max(felt) - min(felt)), "valence_range": float(max(mood) - min(mood)),
        "corr_a_felt": _corr(avals, felt), "corr_h_mood": _corr(hs, mood),
        "felt_curve": [float(x) for x in felt], "mood_curve": [float(x) for x in mood],
    }


# =============================================================================================================
# SMOKE — determinism + operating-point sweep (i_body x w_int x xinh) on one seed
# =============================================================================================================
def _threshold_hash(seed):
    brain = GradedAffectBrain(seed)
    th = to_host(brain._bridge.cp_neuron_firing_thresholds)
    return np.asarray(th, float).tobytes()


def run_smoke(seed, i_bodies, w_ints, xinhs):
    print(f"[graded-affect SMOKE] seed={seed} — determinism + operating point (i_body x w_int x xinh)", flush=True)
    det_ok = (_threshold_hash(seed) == _threshold_hash(seed))
    print(f"  determinism: two builds at one seed -> {'IDENTICAL (seeded)' if det_ok else 'DIFFER (BUG)'}", flush=True)
    print(f"  {'i_body':>7} {'w_int':>6} {'xinh':>5} | {'corrA':>6} {'aLvls':>5} {'aRange':>7} | "
          f"{'corrH':>6} {'vLvls':>5} | {'held_corr':>9} {'held_off':>8} | verdict", flush=True)
    rows, chosen = [], None
    for ib in i_bodies:
        for w in w_ints:
            for xi in xinhs:
                r = run_seed(seed, i_body=ib, w_int=w, n_pts=9, n_noise=3, xinh_inh_w=xi)
                ok = (r["corr_a_felt"] >= 0.8 and r["felt_levels_seed"] > 2 and r["corr_h_mood"] >= 0.8
                      and r["mood_levels_seed"] > 2 and r["corr_level_held"] >= 0.6 and r["held_off_max"] < 0.02)
                print(f"  {ib:>7.0f} {w:>6.1f} {xi:>5.1f} | {r['corr_a_felt']:>+6.2f} {r['felt_levels_seed']:>5d} "
                      f"{r['felt_range']:>7.3f} | {r['corr_h_mood']:>+6.2f} {r['mood_levels_seed']:>5d} | "
                      f"{r['corr_level_held']:>+9.2f} {r['held_off_max']:>8.3f} | {'GOOD' if ok else '-'}", flush=True)
                rows.append({"i_body": ib, "w_int": w, "xinh": xi, "ok": bool(ok),
                             **{k: r[k] for k in ("corr_a_felt", "felt_levels_seed", "corr_h_mood",
                                                  "mood_levels_seed", "corr_level_held", "held_off_max",
                                                  "felt_range", "mood_range")}})
                if ok and chosen is None:
                    chosen = (ib, w, xi)
    if chosen is None:
        best = max(rows, key=lambda r: (r["corr_a_felt"] + r["corr_h_mood"] + 0.1 * r["felt_levels_seed"]))
        chosen = (best["i_body"], best["w_int"], best["xinh"])
        print(f"  [smoke] no operating point cleanly passed; best at {chosen}", flush=True)
    else:
        print(f"  [smoke] operating point: i_body={chosen[0]} w_int={chosen[1]} xinh={chosen[2]}", flush=True)
    return {"determinism_ok": bool(det_ok), "chosen_i_body": float(chosen[0]), "chosen_w_int": float(chosen[1]),
            "chosen_xinh": float(chosen[2]), "sweep": rows}


# =============================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--i-body", type=float, default=I_BODY_PA)
    ap.add_argument("--w-int", type=float, default=W_INT)
    ap.add_argument("--xinh", type=float, default=XINH_INH_W)
    ap.add_argument("--ou-pA", type=float, default=8.0, help="OU background noise (pA) -> cfg.ou_std_current_pA")
    ap.add_argument("--n-pts", type=int, default=11)
    ap.add_argument("--sweep-i-body", type=float, nargs="+", default=[260.0])
    ap.add_argument("--sweep-w-int", type=float, nargs="+", default=[5.0, 6.0, 7.0])
    ap.add_argument("--sweep-xinh", type=float, nargs="+", default=[0.0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.time()
    if a.smoke:
        smoke = run_smoke(a.seeds[0], a.sweep_i_body, a.sweep_w_int, a.sweep_xinh)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        outp = str(a.out).replace(".json", "_smoke.json")
        Path(outp).write_text(json.dumps(smoke, indent=2, default=str))
        print(f"[graded-affect SMOKE] wrote {outp} ({round(time.time()-t0,1)}s)", flush=True)
        return 0

    print(f"[graded-affect] 6-seed battery @ i_body={a.i_body} w_int={a.w_int} xinh={a.xinh} "
          f"(N_L={N_L}, N_SUB={N_SUB}, recur={RECUR_WEIGHT})", flush=True)
    determinism_ok = (_threshold_hash(a.seeds[0]) == _threshold_hash(a.seeds[0]))
    rows = [run_seed(s, i_body=a.i_body, w_int=a.w_int, n_pts=a.n_pts, xinh_inh_w=a.xinh, ou_pA=a.ou_pA)
            for s in a.seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    # POOLED resolvable-levels: use the ACROSS-SEED SD at each sweep point (the honest 6-seed noise floor).
    mood_curves = np.array([r["mood_curve_intact"] for r in rows])          # (6, n_pts)
    felt_curves = np.array([r["felt_curve_intact"] for r in rows])
    mood_mean, mood_sd = mood_curves.mean(0), mood_curves.std(0)
    felt_mean, felt_sd = felt_curves.mean(0), felt_curves.std(0)
    mood_levels_pooled = resolvable_levels(mood_mean, mood_sd)
    felt_levels_pooled = resolvable_levels(felt_mean, felt_sd)

    # LADDER-IS-LOAD-BEARING control (seed 42): a matched single bistable pool must resolve FEWER levels (it
    # saturates all-or-none -> the P0.3 switch) than the ladder.
    print("[graded-affect] single-pool control (matched total N, no ladder) ...", flush=True)
    ctrl = single_pool_control(a.seeds[0], i_body=a.i_body, w_int=a.w_int, n_pts=a.n_pts, ou_pA=a.ou_pA)
    print(f"  single-pool: AROU levels {ctrl['arousal_levels']} range {ctrl['arousal_range']:.3f} "
          f"(corr {ctrl['corr_a_felt']:+.2f}) | VAL levels {ctrl['valence_levels']} range "
          f"{ctrl['valence_range']:.3f} (corr {ctrl['corr_h_mood']:+.2f})", flush=True)

    n_val_graded = sum(1 for r in rows if r["corr_h_mood"] >= 0.8 and r["mood_levels_seed"] > 2)
    n_aro_graded = sum(1 for r in rows if r["corr_a_felt"] >= 0.8 and r["felt_levels_seed"] > 2)
    n_persist = sum(1 for r in rows if r["corr_level_held"] >= 0.8 and r["held_range"] >= 0.02)
    n_persist_off = sum(1 for r in rows if r["held_off_max"] < 0.02)
    n_lesion_val = sum(1 for r in rows if abs(r["corr_h_mood_lesion"]) < 0.3
                       and r["mood_range_lesion"] <= 0.25 * r["mood_range"] + 1e-9)
    n_lesion_aro = sum(1 for r in rows if r["felt_range_lesion"] <= 0.25 * r["felt_range"] + 1e-9)
    n_enc = sum(1 for r in rows if r["corr_comfort_enc"] >= 0.9 and r["corr_arousal_enc"] >= 0.9)

    ns = len(rows)
    # GATES (>=5/6 for the graded headline; the rest 6/6):
    agg = {
        "valence_graded_5of6(corrH>=0.8 & >2 levels)": n_val_graded >= 5,
        "arousal_graded_5of6(corrA>=0.8 & >2 levels)": n_aro_graded >= 5,
        "pooled_mood_levels>2": mood_levels_pooled > 2,
        "pooled_felt_levels>2": felt_levels_pooled > 2,
        "graded_persistence_5of6(held tracks level, range real)": n_persist >= 5,
        "persistence_is_nmda(off decays ~0, 6/6)": n_persist_off == ns,
        "lesion_decouples_valence_6of6": n_lesion_val == ns,
        "lesion_collapses_arousal_6of6": n_lesion_aro == ns,
        "intero_encodes_body_6of6": n_enc == ns,
        "ladder_resolves_more_than_single_pool(arou)": felt_levels_pooled > ctrl["arousal_levels"],
        "ladder_resolves_more_than_single_pool(val)": mood_levels_pooled > ctrl["valence_levels"],
    }
    preconditions = [
        {"kind": "require", "name": "substrate_seeded(cfg.seed; identical thresholds on rebuild)", "ok": determinism_ok},
        {"kind": "require", "name": "all_requested_seeds_ran(n==6)", "ok": bool(ns == len(a.seeds) == 6)},
        {"kind": "require", "name": "felt_read_is_neural(rate off cp_firing_states, not a host formula)", "ok": True},
        {"kind": "require", "name": "body_reaches_affect_only_via_synapses(runtime assert held every step)", "ok": True},
        {"kind": "require", "name": "numpy_spiking_backend", "ok": os.environ.get("SIM_BACKEND", "") == "numpy"},
    ]
    go = all(agg.values()) and all(p["ok"] for p in preconditions)

    means = {k: m(k) for k in ("corr_h_mood", "corr_a_felt", "mood_range", "felt_range", "corr_h_mood_lesion",
                               "mood_range_lesion", "felt_range_lesion", "corr_level_held", "held_range",
                               "held_off_max", "corr_vlevel_held", "held_mood_range", "intero_owns_valence_frac",
                               "intero_owns_arousal_frac", "corr_comfort_enc", "corr_arousal_enc")}
    means["mood_levels_pooled"] = mood_levels_pooled
    means["felt_levels_pooled"] = felt_levels_pooled
    means["mood_levels_seed_mean"] = m("mood_levels_seed")
    means["felt_levels_seed_mean"] = m("felt_levels_seed")

    baseline = "baseline (#49 P0.3 latch): mood a 2-state +-0.08 switch; felt-arousal on/off (Pearson 0.70, 1/6 seeds >=0.8)"
    if go:
        verdict = (f"GO ({ns}-seed) — a GRADED bistable-LADDER affect substrate reads the interoceptive body-state "
                   f"as a SMOOTH valence x arousal, not a +/- switch. VALENCE: Pearson(h,mood) {means['corr_h_mood']:+.2f} "
                   f"({n_val_graded}/{ns} seeds >=0.8), {mood_levels_pooled} pooled resolvable levels (range "
                   f"{means['mood_range']:.3f}). AROUSAL: Pearson(a,felt) {means['corr_a_felt']:+.2f} "
                   f"({n_aro_graded}/{ns} seeds >=0.8 vs the 1/6 baseline), {felt_levels_pooled} pooled levels. "
                   f"PERSISTENCE: the held state TRACKS the drive level (Pearson {means['corr_level_held']:+.2f}, "
                   f"held-range {means['held_range']:.3f}) and decays to ~0 with NMDA-off ({means['held_off_max']:.3f}) "
                   f"— a graded robust integrator, {n_persist}/{ns} seeds. EMBODIMENT: cutting the "
                   f"interoceptive->affect synapses collapses the coupling (valence range {means['mood_range']:.3f}->"
                   f"{means['mood_range_lesion']:.3f}, |corr|->{abs(means['corr_h_mood_lesion']):.2f}; intero owns "
                   f"{means['intero_owns_valence_frac']*100:.0f}%/{means['intero_owns_arousal_frac']*100:.0f}% "
                   f"val/arou) while the interoceptive pools still encode the body (corr "
                   f"{means['corr_comfort_enc']:+.2f}/{means['corr_arousal_enc']:+.2f}). Felt read = rate(ladder) off "
                   f"cp_firing_states; body reaches affect only via synapses (asserted). numpy-CPU; NO sim/ edit. "
                   f"{baseline}.")
    else:
        miss = [k for k, v in agg.items() if not v]
        verdict = (f"PARTIAL/BOUNDARY ({ns}-seed) — FAILED {miss}. VALENCE corrH {means['corr_h_mood']:+.2f} "
                   f"({n_val_graded}/{ns} graded), {mood_levels_pooled} pooled levels; AROUSAL corrA "
                   f"{means['corr_a_felt']:+.2f} ({n_aro_graded}/{ns} graded), {felt_levels_pooled} pooled levels; "
                   f"PERSIST held corr {means['corr_level_held']:+.2f} range {means['held_range']:.3f} off "
                   f"{means['held_off_max']:.3f} ({n_persist}/{ns}); lesion val {n_lesion_val}/{ns} arou "
                   f"{n_lesion_aro}/{ns}; enc {n_enc}/{ns}. {baseline}.")

    summary = {
        "probe": "graded_affect_attractor (board #81)", "verdict": verdict, "GO": bool(go),
        "preconditions": preconditions, "aggregate_checks": agg,
        "n_seeds": ns, "n_valence_graded": n_val_graded, "n_arousal_graded": n_aro_graded,
        "n_persist": n_persist, "n_persist_nmda_off_decays": n_persist_off,
        "n_lesion_valence": n_lesion_val, "n_lesion_arousal": n_lesion_aro, "n_intero_encodes": n_enc,
        "mood_levels_pooled": mood_levels_pooled, "felt_levels_pooled": felt_levels_pooled,
        "single_pool_control": ctrl,
        "baseline_note": baseline, "means": means, "per_seed": rows,
        "config": {"seeds": a.seeds, "i_body_pA": a.i_body, "w_int": a.w_int, "xinh_inh_w": a.xinh, "ou_pA": a.ou_pA,
                   "N_L": N_L, "N_SUB": N_SUB, "recur_weight": RECUR_WEIGHT, "g_min": G_MIN, "dens_int": DENS_INT,
                   "gains": [float(x) for x in ladder_gains()]},
        "mechanism": "A Koulakov-2002/Goldman-2003 robust DISCRETE integrator: per affect sign, a LADDER of N_L "
                     "independent self-recurrent NMDA bistable sub-pools (the P0.3 aff() primitive) with NO "
                     "intra-sign inhibition; staggered SYNAPTIC recruitment (intero->sub-pool gain decreasing along "
                     "the ladder) so a stronger body signal latches more sub-pools; graded value = the count of "
                     "latched sub-pools = population firing rate. Interoceptive relays (comfort/discomfort/arousal) "
                     "drive the ladders synaptically (gated by intero_out). Felt = rate(V+ ladder)-rate(V- ladder) "
                     "and rate(arousal ladder), off cp_firing_states. Aggregate-only opponent cross-inhibition "
                     "(weak). Body variables host; everything from the afferent current on is neurons/synapses.",
        "HONEST_NOTE": "numpy-CPU (real spiking Izhikevich bridge). Body-state variables (h, a) are HOST (the body "
                       "boundary, like the world); the de-risk is the body->AFFECT mapping being a graded ladder of "
                       "synaptically-recruited bistable pools. Gradedness is QUANTIZED (N_L+1 levels) by design "
                       "(Koulakov: drift-robustness is bought with resolution) -- a robust graded staircase, not a "
                       "smooth continuum; the resolvable-levels count reports the achieved resolution. Staggering is "
                       "synaptic (the affect pools get 0 direct external current, asserted), so it cannot inject "
                       "host persistence. Bounded first slice: 2 body axes, open-loop sweep (no body dynamics).",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[graded-affect] VERDICT: {verdict}", flush=True)
    print(f"[graded-affect] GO={go} | wrote {a.out} ({summary['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
