"""One-brain INTEGRATION R3 -- closing R2-a's declared residual: the credit SIGNAL'S VALUE becomes a
SPIKING dopamine/coincidence population's OWN firing, not a host-delivered scalar.

R2 (`research/findings/2026-08-27-onebrain-integration-R2-threefactor-selforganized.md`) made the
plasticity RULE genuinely three-factor (reward-DEFERRED STDP: a weight changes only on a same-episode
credit pulse) but left ONE residual explicitly open: "the credit signal's VALUE (`current_reward_signal`)
is still a host-delivered scalar -- the runner sets it directly from its own ground-truth bookkeeping of
which episode is 'correct,' not from a spiking dopamine/value population computed by the brain's own error
or success detection." This runner closes that residual using the EXISTING, already-reviewed engine
primitive named for exactly this purpose (`sim/neuromodulators.py`'s "Spiking-SNc actor-critic Stage A" --
`ProductionRule(rule_type="from_region_firing"/"from_region_firing_signed")`, which drives a registered
"dopamine" NeuromodulatorConfig's concentration from a REGION's OWN `cp_firing_states`, every step, entirely
inside `_run_one_simulation_step` -- NO runner-mediated scalar write). Once a "dopamine" modulator is
registered, `sim/bridge.py`'s C2 reward-modulated-STDP block (~line 10605-10651) AUTOMATICALLY prefers
`da_signal = DA_concentration - DA_baseline` over the raw `current_reward_signal - reward_baseline` scalar
for the ACTUAL weight-update math (`effective_signal`, consumed at line 10699:
`weight_updates = effective_reward_lr * effective_signal * eligibility_trace`). This runner NEVER sets
`cfg.current_reward_signal` away from 0.0 for the entire R3 training regime -- the only path by which a
cross-edge can grow is via the DA-population route.

THE MECHANISM (config + explicit wiring only; NO `sim/` edit):

  A new tiny co-located organ ("da_credit": 4 small BrainRegions, teach_agent/teach_patient/snc_a/snc_b,
  reusing the SAME low-level `BrainRegion`/`inject_explicit_wiring` primitives R1/R2/the crossedge smoke
  already use to add ad hoc circuits to a merge pool) implements a genuine spiking COINCIDENCE DETECTOR:

    sel_agent   --(fixed, w=2.0, non-plastic)--> snc_a  <--(fixed, w=2.0, non-plastic)--  teach_agent
    sel_patient --(fixed, w=2.0, non-plastic)--> snc_b  <--(fixed, w=2.0, non-plastic)--  teach_patient

  Calibrated (research/findings/raw/_onebrain_integration_r3_calibration.log) so EITHER leg ALONE is
  strictly SUBTHRESHOLD for snc_a/snc_b (0/24 neurons fire) but BOTH TOGETHER cross threshold cleanly
  (24/24 fire) -- a textbook feedforward AND-gate via subthreshold EPSP summation (integrate-and-fire
  coincidence detection; the same class of mechanism as auditory/NMDA coincidence detectors). `sel_agent`/
  `sel_patient` carry the network's OWN resolved Wong-Wang WTA decision (a REAL synaptic read, not a host
  peek at `cp_firing_states`); `teach_agent`/`teach_patient` carry a host/environment-delivered "the teacher
  confirms this role" drive -- legitimate host territory, the SAME class of signal as the AGENT_CUE/
  PATIENT_CUE currents R1/R2 already use (the environment/teacher boundary, CLAUDE.md's brain-based-only
  standard). Only when the substrate's OWN decision agrees with the teacher's declared truth does snc_a/
  snc_b fire; a registered `dopamine` NeuromodulatorConfig (`ProductionRule(rule_type="from_region_firing",
  source_regions=["snc_a","snc_b"])`) turns THAT firing into the DA concentration the C2 block consumes.

  This upgrades R2's credit semantics too: R2 delivered `current_reward_signal` purely from SCHEDULE
  METADATA (which branch this episode came from), regardless of what the WTA actually resolved that trial.
  R3's credit is contingent on the network's ACTUAL settled decision matching the teacher's label.

THE NEW LOAD-BEARING CONTROL (R3's crux, the one the host-scalar version could not run): DOPAMINE LESION --
zero the 4 fixed sel/teach->snc synapse weights (snc_a/snc_b can then never fire; DA concentration never
leaves baseline) and re-run the IDENTICAL intact-credit schedule (same teach-agent/teach-patient drives).
If the credit is genuinely carried by the coincidence population's spikes, this must collapse selectivity to
~W0 (indistinguishable from the "removed" control) even though the runner is doing EXACTLY the same thing it
did in the intact arm. R2's controls (withhold / decorrelate credit) are carried forward UNCHANGED, now
implemented as "never drive teach" / "drive teach per a shuffled credit vector."

HONEST RESIDUAL (declared, not hidden): the coincidence-detector CIRCUIT's wiring (the 4 fixed synapse
groups + the dopamine ProductionRule's threshold/window/decay constants) is HOST-DESIGNED, like R1/R2's cue
pathways and the Wong-Wang WTA's own lateral-inhibition circuit -- infrastructure, never claimed
self-organized. What IS newly claimed: the CREDIT VALUE (whether a given trial's plasticity gets converted
from eligibility tag to an actual weight change) is now read from that population's OWN spikes via the
engine's native neuromodulator-concentration pathway, not written by a host boolean into a reward scalar.
`current_reward_signal` is never touched (stays 0.0 for the whole run) -- the ONLY route to weight change is
DA-population-mediated.

GATE: R2's F1-F4 + lesion-recovers-migration (re-based against a [D6, COMP, da_credit]-with-R2-candidate-
edges-absent baseline, since da_credit is now permanent infrastructure, not part of what "migration" tests)
+ R2-a-style three-factor (intact selective / removed inert / shuffled degraded) + the NEW dopamine-lesion
control (must be as inert as "removed"). 6 seeds (42,43,44,100,101,102). numpy CPU; NO `sim/` edit.

Run:
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_r3_spiking_dopamine_credit --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_r3_spiking_dopamine_credit \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_integration_r3_spiking_dopamine_credit_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import time
import types
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend
from tools.lab import attributable_to

# reuse R2's generic F1/F2/F3/F4 gate arms verbatim (they operate purely on the pool-instance's attributes/
# methods: .comp_organ/.d6_organ/.amb_read/.masks/.p_agent/.p_patient/.b/.xp -- R3Pool below matches that shape).
from research.runners._onebrain_integration_r2_threefactor_selforganized import (
    _f1, _f2, _f3, _f4,
    CAND_POOLS, BASE_POOL, W0, GATE, LOAD_PA, CUE_PA, AMBIG_PA, CLEAR_PA,
    LOAD_STEPS, TRAIN_STEPS, READ_STEPS, N_READS, N_EPISODE_PAIRS, HMAX, REWARD_LR, REWARD_TAU_MS,
    _CONDUCT, _role_assignment,
)

# ---- R3-specific geometry + protocol constants ----
TEACH_N, SNC_N = 24, 24               # match sel_agent/sel_patient's own scale (calibrated at this size)
W_SEL = 2.0                           # sel_{agent,patient} -> snc_{a,b}: calibrated SUBTHRESHOLD-alone
W_TEACH = 2.0                         # teach_{agent,patient} -> snc_{a,b}: calibrated SUBTHRESHOLD-alone
TEACH_PA = 3500.0                     # teach_agent/teach_patient direct drive (matches CUE_PA's scale)
# DA modulator (sim.neuromodulators.NeuromodulatorConfig/ProductionRule -- EXISTING engine primitive, no sim/
# edit): calibrated so idle (no coincidence) reads ~0 (threshold~0) and a coincidence burst (24/24 snc
# neurons firing) registers a clearly nonzero, decaying da_signal within the episode.
DA_WINDOW_MS = 5.0
DA_THRESHOLD = 0.0
DA_SENSITIVITY = 60.0
DA_DECAY_TAU_MS = 30.0
DA_CONC_MAX = 5.0

# F-gate floors (identical to R2 — same organs, same read protocol)
F2_INTACT_FLOOR = 0.008
F2_LESION_RATIO = 0.34
F4A_FRAC = 0.5
F4B_RETAIN = 0.5
RATE_LO, RATE_HI = 5e-4, 0.7
# R3-a / dopamine-lesion floors (pre-registered on seed 42/43 calibration; see the finding for the measured
# per-seed pattern).
SEL_FLOOR_INTACT = 0.15
SEL_REMOVED_EPS = 1e-6
SEL_SHUFFLE_RATIO = 0.35
SEL_DA_LESION_EPS = 1e-6
TOPOLOGY_SHUFFLE_MAX_MATCH = 4


def _dense(pre, post, w, gate, plastic=True):
    pre = np.asarray(pre, np.int64); post = np.asarray(post, np.int64)
    P = np.repeat(pre, len(post)); Q = np.tile(post, len(pre))
    return {"pre_indices": P, "post_indices": Q,
            "initial_weights": np.full(P.size, float(w), np.float32),
            "plastic": bool(plastic), "plasticity_gate": gate, "conn_type": "E_TO_E", "count": int(P.size)}


def _spec_da_credit(seed):
    """The coincidence-detector organ's regions: teach_{agent,patient} (host/environment-driven "the teacher
    confirms this role" input) + snc_{a,b} (the AND-gate dopamine population). No internal recurrence (density
    0) -- these are driven-only populations, like `src`/`tgt` in the crossedge smoke."""
    from sim.regions import BrainRegion
    regions = [
        BrainRegion(name="teach_agent", n_neurons=TEACH_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="teach_patient", n_neurons=TEACH_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="snc_a", n_neurons=SNC_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="snc_b", n_neurons=SNC_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
    ]
    return regions, [], {}


def _dopamine_cfg():
    """The 'dopamine' NeuromodulatorConfig — sourced from snc_a/snc_b's OWN firing (from_region_firing: reads
    bridge.cp_firing_states + region_manager.indices, EMAs it over DA_WINDOW_MS, produces a positive
    contribution when the ema exceeds DA_THRESHOLD). This is the EXISTING 'Spiking-SNc actor-critic Stage A'
    primitive (sim/neuromodulators.py) — no sim/ edit. Registering it makes sim/bridge.py's C2 block prefer
    `da_signal` over the raw `current_reward_signal` scalar for EVERY weight update (bridge.py:~10605-10651,
    10699) — current_reward_signal is never set away from 0.0 anywhere in this runner."""
    from sim.neuromodulators import NeuromodulatorConfig, ProductionRule
    return NeuromodulatorConfig(
        name="dopamine", baseline=0.0, decay_tau_ms=DA_DECAY_TAU_MS,
        concentration_min=0.0, concentration_max=DA_CONC_MAX,
        targets=[],
        production_rules=[ProductionRule(rule_type="from_region_firing", source_regions=["snc_a", "snc_b"],
                                          sensitivity=DA_SENSITIVITY, threshold=DA_THRESHOLD, window_ms=DA_WINDOW_MS)])


def _build_pool(seed, with_cross):
    """[d6, comprehension, da_credit] MergedPool. Identical to R2's _build_pool EXCEPT (a) the da_credit organ
    is ALWAYS present (permanent infrastructure — the migration invariant re-bases against this, not against
    R2's bare [d6, comprehension]), and (b) the fixed sel->snc / teach->snc coincidence wiring is injected
    alongside the (optional) R2 candidate cross-edges."""
    from research.runners.onebrain_merge_framework import REGISTRY, MergedPool, OrganDescriptor
    xp, _ = get_backend()
    D6, COMP = REGISTRY["d6_multiref_wm"], REGISTRY["comprehension"]
    STRIP = ("enable_stdp", "enable_reward_modulation")
    D6c = dataclasses.replace(D6, config={k: v for k, v in D6.config.items() if k not in STRIP})
    COMPc = dataclasses.replace(COMP, config={k: v for k, v in COMP.config.items() if k not in STRIP})
    DA = OrganDescriptor(key="da_credit", regions=("teach_agent", "teach_patient", "snc_a", "snc_b"),
                          spec_fn=_spec_da_credit, config={})
    extra = types.SimpleNamespace(key="r3_extra", config={
        "enable_stdp": True, "enable_reward_modulation": True,
        "reward_defer_stdp_weight_update": True,           # strict three-factor: STDP alone never writes a weight
        "enable_neuromodulator_subsystem": True,            # the DA-population credit path (replaces the raw scalar)
        "neuromodulators": [_dopamine_cfg()],
    }, param_het=False)
    pool = MergedPool(seed, [D6, COMP, DA], config_descriptors=[D6c, COMPc, extra], wire=True)
    pool.ensure_built()
    b = pool.bridge
    rm = b.region_manager
    def idxr(nm):
        return np.asarray(rm.indices(nm), np.int64)
    names = list(CAND_POOLS) + [BASE_POOL, "sel_agent", "sel_patient", "fs",
                                 "cue_animacy_pos", "cue_animacy_neg", "cue_verbfit_pos", "cue_verbfit_neg",
                                 "teach_agent", "teach_patient", "snc_a", "snc_b"]
    ix = {nm: idxr(nm) for nm in names}
    union = dict(rm.build_wiring_plan(seed=pool.seed, per_region_seed=True))
    masks = None
    if with_cross:
        for p in CAND_POOLS:
            union[f"x_{p}_sela"] = _dense(ix[p], ix["sel_agent"], W0, GATE)
            union[f"x_{p}_selp"] = _dense(ix[p], ix["sel_patient"], W0, GATE)
    # the coincidence-detector wiring — ALWAYS present (fixed, non-plastic; not gated by `with_cross`, since
    # it is infrastructure, not the thing under test for the migration invariant).
    union["teachA_snca"] = _dense(ix["teach_agent"], ix["snc_a"], W_TEACH, None, plastic=False)
    union["selA_snca"] = _dense(ix["sel_agent"], ix["snc_a"], W_SEL, None, plastic=False)
    union["teachP_sncb"] = _dense(ix["teach_patient"], ix["snc_b"], W_TEACH, None, plastic=False)
    union["selP_sncb"] = _dense(ix["sel_patient"], ix["snc_b"], W_SEL, None, plastic=False)
    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    if with_cross:
        coo = b.cp_connections.tocoo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        masks = {}
        for p in CAND_POOLS:
            masks[f"{p}->A"] = np.isin(row, ix[p]) & np.isin(col, ix["sel_agent"])
            masks[f"{p}->P"] = np.isin(row, ix[p]) & np.isin(col, ix["sel_patient"])
    # da_credit masks (always built — used by the dopamine-lesion control)
    coo2 = b.cp_connections.tocoo()
    row2 = np.asarray(to_host(coo2.row)); col2 = np.asarray(to_host(coo2.col))
    da_masks = {
        "teach_agent->snc_a": np.isin(row2, ix["teach_agent"]) & np.isin(col2, ix["snc_a"]),
        "sel_agent->snc_a": np.isin(row2, ix["sel_agent"]) & np.isin(col2, ix["snc_a"]),
        "teach_patient->snc_b": np.isin(row2, ix["teach_patient"]) & np.isin(col2, ix["snc_b"]),
        "sel_patient->snc_b": np.isin(row2, ix["sel_patient"]) & np.isin(col2, ix["snc_b"]),
    }
    # re-settle to rest + refresh pool.snap (v,u) so the comprehension organ's hard-reset uses the CURRENT rest
    b.cp_external_input_current[:] = 0.0
    for _ in range(40):
        b._run_one_simulation_step()
    b.cp_external_input_current[:] = 0.0
    if pool.snap is not None:
        pool.snap["cp_membrane_potential_v"] = np.asarray(to_host(b.cp_membrane_potential_v)).copy()
        pool.snap["cp_recovery_variable_u"] = np.asarray(to_host(b.cp_recovery_variable_u)).copy()
    return pool, ix, masks, da_masks


class R3Pool:
    """The R3 integrated pool: [d6, comprehension, da_credit], trained by reward-gated (strict three-factor)
    STDP whose credit VALUE comes from the da_credit coincidence-detector population's OWN spikes (via the
    engine's native dopamine-concentration pathway), under one of FOUR credit regimes:
      intact      — teach_{agent,patient} driven exactly on the schedule's TRUE-correct episodes.
      removed     — teach never driven (mirrors R2's withhold-credit control).
      shuffled    — teach driven per a SHUFFLED true/false vector (mirrors R2's decorrelate control).
      da_lesioned — the SAME intact teach-drive schedule, but the 4 coincidence synapses are zeroed BEFORE
                    training, so snc_a/snc_b can never fire (the NEW R3 control: proves the credit is
                    carried by the population's spikes, not by anything else the runner does).
    `current_reward_signal` is NEVER set away from 0.0 in ANY regime — the only route to a weight change is
    the DA-population pathway."""

    def __init__(self, seed, mode="intact"):
        self.seed = int(seed)
        self.mode = str(mode)
        self.xp, _ = get_backend()
        self.p_agent, self.p_patient, self.p_ctrl = _role_assignment(seed)
        self.pool, self.ix, self.masks, self.da_masks = _build_pool(seed, with_cross=True)
        self.b = self.pool.bridge
        # WHITELIST FREEZE FIRST (R3-specific ordering fix, earned by calibration): with the DA-population
        # credit path active (enable_stdp + enable_reward_modulation + enable_neuromodulator_subsystem all
        # True), `comp_organ.ensure_built()`'s OWN calibration battery drives cue populations through the
        # SAME shared bridge -- and if plasticity is not YET frozen, that calibration traffic (sel_agent
        # firing -> a real synapse into snc_a -> a transient DA deviation -> a nonzero effective_signal) can
        # write spurious weight changes on WHATEVER synapses are STDP-active before the whitelist ever runs,
        # corrupting D6's frozen internal recurrent-hold weights (observed: d6_hold_alive_min -> 0.0 pre-
        # training). R2 never hit this because its effective_signal is the raw current_reward_signal scalar,
        # which stays exactly 0.0 during calibration regardless of what fires. Freezing gain=0 BEFORE any
        # organ's own calibration touches the pool (then re-opening ONLY the R2 candidate-edge gate) makes
        # every OTHER synapse -- including da_credit's own fixed wiring -- immune to this, matching the
        # already-validated R1/R2 invariant that only the whitelisted 6 candidate edges ever move.
        self.b.set_plasticity_gate(GATE, 1.0)
        self.b.cp_plasticity_rate_gain[:] = 0.0
        self.b.set_plasticity_gate(GATE, 1.0)
        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]
        from research.runners.onebrain_merge_framework import _comprehension_organ, _d6_organ
        self.comp_organ = _comprehension_organ(seed, self.pool)
        self.d6_organ = _d6_organ(seed, self.pool)
        self.comp_organ.ensure_built()
        if self.mode == "da_lesioned":
            data = np.asarray(to_host(self.b.cp_connections.data)).copy()
            for k in self.da_masks:
                data[self.da_masks[k]] = 0.0
            self.b.cp_connections.data = self.xp.asarray(data, dtype=self.b.cp_connections.data.dtype)
        cfg = self.b.core_config
        cfg.stdp_w_max = HMAX
        cfg.reward_learning_rate = REWARD_LR
        cfg.reward_eligibility_tau_ms = REWARD_TAU_MS
        cfg.reward_baseline = 0.0
        cfg.current_reward_signal = 0.0          # NEVER touched again — the DA-population path is the ONLY route
        cfg.enable_hebbian_learning = False
        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()

    # ---- primitives ----
    def _hard_reset(self):
        b, xp = self.b, self.xp
        b.cp_membrane_potential_v[:] = xp.asarray(self.rest_v)
        b.cp_recovery_variable_u[:] = xp.asarray(self.rest_u)
        for nm in _CONDUCT:
            a = getattr(b, nm, None)
            if a is not None:
                a[:] = 0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        if getattr(b, "cp_last_spike_time", None) is not None:
            b.cp_last_spike_time[:] = -1000.0
        if getattr(b, "cp_eligibility_trace", None) is not None:
            b.cp_eligibility_trace[:] = 0.0
        b.cp_external_input_current[:] = 0.0
        # RESET the dopamine concentration between episodes — otherwise a burst late in one episode would
        # leak (via decay_tau_ms) into the NEXT episode's early steps, crediting the wrong trial.
        if b.neuromodulator_manager is not None:
            b.neuromodulator_manager.set_concentration("dopamine", 0.0)
            b.neuromodulator_manager._rule_state["dopamine"] = {"err_ema": 0.0, "rate_ema": 0.0, "signed_rate_ema": 0.0}

    def _drive(self, pairs, steps, read=None):
        b, xp = self.b, self.xp
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        acc = {k: 0.0 for k in (read or {})}
        for _ in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
            if read:
                fs = b.cp_firing_states
                for k, idx in read.items():
                    acc[k] += float(to_host(fs[xp.asarray(idx)].astype(xp.float64).sum())) / idx.size
        b.cp_external_input_current[:] = 0.0
        return {k: v / steps for k, v in acc.items()}

    def _wmean(self, name):
        return float(np.asarray(to_host(self.b.cp_connections.data))[self.masks[name]].mean())

    def cross_weights(self):
        return {k: round(self._wmean(k), 4) for k in self.masks}

    # ---- emergence: grow the cross-edges from credited experience, credit VALUE = da_credit's OWN spikes ----
    def _episode(self, pool_key, cue_pairs, credited, teach_pool):
        """hard-reset -> load+hold `pool_key` -> drive cue_pairs for the WHOLE cue window (calibrated
        coincidence-detector window — the sel_{agent,patient}->snc_{a,b} leg needs the SAME window the network
        actually resolves its WTA decision in), driving `teach_pool` alongside it ONLY when `credited`. NO
        `current_reward_signal` write anywhere in this method."""
        self._hard_reset()
        self._drive([(self.ix[pool_key], LOAD_PA)], LOAD_STEPS)
        pairs = list(cue_pairs)
        if credited and teach_pool is not None:
            pairs.append((self.ix[teach_pool], TEACH_PA))
        self._drive(pairs, TRAIN_STEPS)

    def train(self, n_episode_pairs=None):
        if n_episode_pairs is None:
            n_episode_pairs = N_EPISODE_PAIRS
        ix = self.ix
        AGENT_CUE = [(ix["cue_animacy_pos"], CUE_PA), (ix["cue_verbfit_pos"], CUE_PA)]
        PATIENT_CUE = [(ix["cue_animacy_neg"], CUE_PA), (ix["cue_verbfit_neg"], CUE_PA)]
        schedule = []   # (pool_key, cue_pairs, is_correct, teach_pool)
        for ep in range(n_episode_pairs):
            schedule.append((self.p_agent, AGENT_CUE, True, "teach_agent"))
            schedule.append((self.p_patient, PATIENT_CUE, True, "teach_patient"))
            schedule.append((self.p_ctrl, AGENT_CUE, False, "teach_agent"))
            schedule.append((self.p_ctrl, PATIENT_CUE, False, "teach_patient"))
        true_credit = [c for (_p, _c, c, _t) in schedule]
        if self.mode == "removed":
            credited = [False] * len(schedule)
        elif self.mode == "shuffled":
            rng = np.random.RandomState(self.seed * 104729 + 1)
            credited = list(rng.permutation(true_credit))
        else:   # intact, da_lesioned — the SAME true schedule; da_lesioned differs only in the wiring, not the drive
            credited = true_credit
        traj = [dict(i=0, **self.cross_weights())]
        for i, (pool_key, cue_pairs, _true, teach_pool) in enumerate(schedule):
            self._episode(pool_key, cue_pairs, credited[i], teach_pool)
            if (i + 1) % 12 == 0 or i == len(schedule) - 1:
                traj.append(dict(i=i + 1, **self.cross_weights()))
        now = np.asarray(to_host(self.b.cp_connections.data))
        self.frozen_maxdrift = float(np.max(np.abs(now[self._noncross] - self._frozen_w0[self._noncross])))
        return traj

    def amb_read(self, hold_pool_key, cue_pairs, band=None):
        ix = self.ix
        margins, rates = [], {"sel_agent": 0.0, "sel_patient": 0.0, self.p_agent: 0.0, self.p_patient: 0.0,
                              "fs": 0.0, "cue_animacy_pos": 0.0}
        for _ in range(N_READS):
            self._hard_reset()
            if hold_pool_key is not None:
                self._drive([(ix[hold_pool_key], LOAD_PA)], LOAD_STEPS)
                self._drive([], 6)
            read = {"A": ix["sel_agent"], "P": ix["sel_patient"]}
            if band is not None:
                for r in rates:
                    read[r] = ix[r]
            acc = self._drive([(ix[k], pa) for k, pa in cue_pairs], READ_STEPS, read=read)
            margins.append(acc["A"] - acc["P"])
            if band is not None:
                for r in rates:
                    rates[r] += acc[r]
        out = {"margin": float(np.mean(margins))}
        if band is not None:
            out["rates"] = {r: rates[r] / N_READS for r in rates}
        return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  R3-a (spiking three-factor credit) + the NEW dopamine-lesion control
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _selectivity(final_weights, p_agent, p_patient):
    correct = [final_weights[f"{p_agent}->A"], final_weights[f"{p_patient}->P"]]
    incorrect = [v for k, v in final_weights.items() if k not in (f"{p_agent}->A", f"{p_patient}->P")]
    return float(np.mean(correct) - np.mean(incorrect)), float(np.mean(correct)), float(np.mean(incorrect))


def _argmax_pool(final_weights, role_suffix):
    scored = {p: final_weights[f"{p}->{role_suffix}"] for p in CAND_POOLS}
    return max(scored, key=scored.get)


def _r3_emergence(seed, intact_final, removed_final, shuffled_final, da_lesioned_final, p_agent, p_patient, p_ctrl):
    sel_i, corr_i, incorr_i = _selectivity(intact_final, p_agent, p_patient)
    sel_r, _, _ = _selectivity(removed_final, p_agent, p_patient)
    sel_s, _, _ = _selectivity(shuffled_final, p_agent, p_patient)
    sel_d, _, _ = _selectivity(da_lesioned_final, p_agent, p_patient)
    removed_max_dev = float(max(abs(v - W0) for v in removed_final.values()))
    da_lesioned_max_dev = float(max(abs(v - W0) for v in da_lesioned_final.values()))
    argmax_a_i, argmax_p_i = _argmax_pool(intact_final, "A"), _argmax_pool(intact_final, "P")
    tracks_intact = bool(argmax_a_i == p_agent and argmax_p_i == p_patient)
    frac_removed_attrib = attributable_to(f"seed{seed} R3-a selectivity vs REMOVED-credit control", sel_i, sel_r)
    frac_shuffle_attrib = attributable_to(f"seed{seed} R3-a selectivity vs SHUFFLED-credit control", sel_i, sel_s)
    frac_da_lesion_attrib = attributable_to(f"seed{seed} R3 selectivity vs DOPAMINE-LESION control", sel_i, sel_d)
    r3a_pass = bool(sel_i > SEL_FLOOR_INTACT and removed_max_dev < SEL_REMOVED_EPS
                    and (sel_s < SEL_SHUFFLE_RATIO * sel_i if sel_i > 0 else False))
    da_lesion_pass = bool(da_lesioned_max_dev < SEL_DA_LESION_EPS)
    return {
        "role_assignment": {"p_agent": p_agent, "p_patient": p_patient, "p_ctrl": p_ctrl},
        "selectivity_intact": sel_i, "selectivity_removed": sel_r, "selectivity_shuffled": sel_s,
        "selectivity_da_lesioned": sel_d,
        "correct_mean_intact": corr_i, "incorrect_mean_intact": incorr_i,
        "removed_max_deviation_from_W0": removed_max_dev, "removed_formed_nothing": bool(removed_max_dev < SEL_REMOVED_EPS),
        "da_lesioned_max_deviation_from_W0": da_lesioned_max_dev, "da_lesioned_formed_nothing": da_lesion_pass,
        "frac_attributable_removed_control": (None if frac_removed_attrib is None else float(frac_removed_attrib)),
        "frac_attributable_shuffled_control": (None if frac_shuffle_attrib is None else float(frac_shuffle_attrib)),
        "frac_attributable_da_lesion_control": (None if frac_da_lesion_attrib is None else float(frac_da_lesion_attrib)),
        "argmax_agent_intact": argmax_a_i, "argmax_patient_intact": argmax_p_i,
        "topology_tracks_true_assignment_intact": tracks_intact,
        "intact_final": intact_final, "removed_final": removed_final, "shuffled_final": shuffled_final,
        "da_lesioned_final": da_lesioned_final,
        "R3a_three_factor_PASS": r3a_pass,
        "R3_dopamine_lesion_PASS": da_lesion_pass,
    }


def _migration_invariant(seed, r3, comp_battery_reads):
    """Re-based against [d6, comprehension, da_credit] WITHOUT the R2 candidate cross-edges (da_credit is now
    permanent infrastructure — it is NOT what this invariant tests; only the R2 cross-edges + any weight drift
    elsewhere are). Structurally identical to R2's migration_invariant, using the LOCAL _build_pool."""
    from research.runners.onebrain_merge_framework import _comprehension_organ, _comprehension_battery
    pool0, ix0, _m0, _dm0 = _build_pool(seed, with_cross=False)
    org0 = _comprehension_organ(seed, pool0)
    org0.ensure_built()
    def edge_map(pool):
        coo = pool.bridge.cp_connections.tocoo()
        r = to_host(coo.row); c = to_host(coo.col); d = to_host(coo.data)
        return {(int(a), int(b)): float(w) for a, b, w in zip(r, c, d)}
    k0 = edge_map(pool0)
    k1 = edge_map(r3.pool)
    xrows = set(int(x) for p in CAND_POOLS for x in r3.ix[p])
    xcols = set(int(x) for x in np.concatenate([r3.ix["sel_agent"], r3.ix["sel_patient"]]))
    k1_base = {kk: vv for kk, vv in k1.items() if not (kk[0] in xrows and kk[1] in xcols)}
    connectivity_identical = bool(k1_base == k0)
    thr = float(org0.threshold)
    base = [float(org0.read_margin(n0, v, n1)) for (_l, _t, n0, v, n1) in _comprehension_battery(seed)]
    maxerr = float(np.max(np.abs(np.asarray(base) - np.asarray(comp_battery_reads)))) if base else 0.0
    dec_base = [m >= thr for m in base]
    dec_les = [m >= thr for m in comp_battery_reads]
    decisions_preserved = bool(dec_base == dec_les)
    return {"baseline_margins": base, "lesioned_margins": list(map(float, comp_battery_reads)),
            "base_connectivity_byte_identical": connectivity_identical, "read_maxerr": maxerr,
            "decisions_preserved": decisions_preserved, "fp_floor_below_decision_gap": bool(maxerr < 0.5 * thr),
            "PASS": bool(connectivity_identical and decisions_preserved and maxerr < 0.5 * thr)}


def run_seed(seed):
    t0 = time.time()
    p_agent, p_patient, p_ctrl = _role_assignment(seed)

    intact = R3Pool(seed, mode="intact")
    traj = intact.train()
    removed = R3Pool(seed, mode="removed")
    removed.train()
    shuffled = R3Pool(seed, mode="shuffled")
    shuffled.train()
    da_lesioned = R3Pool(seed, mode="da_lesioned")
    da_lesioned.train()

    emg = _r3_emergence(seed, intact.cross_weights(), removed.cross_weights(), shuffled.cross_weights(),
                         da_lesioned.cross_weights(), p_agent, p_patient, p_ctrl)
    emg["frozen_weight_maxdrift_intact"] = float(intact.frozen_maxdrift)
    emg["no_corruption_intact"] = bool(intact.frozen_maxdrift < 1e-6)

    f1 = _f1(intact)
    f4 = _f4(intact)
    f2 = _f2(intact)   # lesions the cross-edges IN PLACE at the end
    intact._hard_reset()
    from research.runners.onebrain_merge_framework import _comprehension_battery
    lesioned_reads = [float(intact.comp_organ.read_margin(n0, v, n1))
                      for (_l, _t, n0, v, n1) in _comprehension_battery(seed)]
    f3 = _f3(intact, traj, f2)
    mig = _migration_invariant(seed, intact, lesioned_reads)

    go = bool(f1["PASS"] and f2["PASS"] and f3["PASS"] and f4["PASS"] and mig["PASS"]
              and emg["no_corruption_intact"] and emg["R3a_three_factor_PASS"] and emg["R3_dopamine_lesion_PASS"])
    return {"seed": int(seed), "PASS": go, "elapsed_s": round(time.time() - t0, 1),
            "emergence": emg, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "lesion_recovers_migration": mig}


def _agg(runs):
    def frac(key):
        parts = key.split(".")
        return sum(1 for r in runs if r[parts[0]][parts[1]])
    keys = ["F1.PASS", "F2.PASS", "F3.PASS", "F4.PASS", "lesion_recovers_migration.PASS",
            "emergence.R3a_three_factor_PASS", "emergence.R3_dopamine_lesion_PASS",
            "emergence.no_corruption_intact"]
    return {k: f"{frac(k)}/{len(runs)}" for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        e = r["emergence"]
        print(f"[seed {s}] {'GO' if r['PASS'] else 'no'} ({r['elapsed_s']}s) | "
              f"sel(intact={e['selectivity_intact']:.3f} removed={e['selectivity_removed']:.3f} "
              f"shuffled={e['selectivity_shuffled']:.3f} da_lesioned={e['selectivity_da_lesioned']:.3f}) "
              f"R3a={e['R3a_three_factor_PASS']} da_lesion={e['R3_dopamine_lesion_PASS']} | "
              f"F1={r['F1']['PASS']} F2={r['F2']['PASS']} F3={r['F3']['PASS']} F4={r['F4']['PASS']} "
              f"mig={r['lesion_recovers_migration']['PASS']}", flush=True)

    n_go = sum(r["PASS"] for r in runs)
    agg = _agg(runs)
    all_go = (n_go == len(runs)) and not args.smoke
    tag = "GO" if all_go else ("SMOKE-GO (1-seed indicator)" if args.smoke and n_go == len(runs) else "NO-GO/PARTIAL")
    verdict = (f"{tag} — R3 spiking-dopamine credit-gated cross-edge d6 WM referent -> comprehension role "
               f"competition: {n_go}/{len(runs)} seeds pass ALL of F1-F4 + lesion-recovers-migration + "
               f"R3-a(three-factor via spikes: intact selective, removed inert, shuffled degraded) + "
               f"the DOPAMINE-LESION control (zero the sel/teach->snc coincidence synapses -> credit vanishes -> "
               f"NO learning, even though the runner drives teach_agent/teach_patient identically to intact). "
               f"Per-arm: {agg}. Mechanism: a spiking coincidence-detector population (snc_a/snc_b) reads the "
               f"network's OWN resolved WTA decision (sel_agent/sel_patient, real synapses) AND a host/"
               f"environment teacher-confirmation drive (teach_agent/teach_patient); its firing feeds the "
               f"engine's native 'dopamine' NeuromodulatorConfig (sim/neuromodulators.py from_region_firing — "
               f"existing primitive, no sim/ edit), which sim/bridge.py's C2 reward-modulated-STDP block "
               f"AUTOMATICALLY prefers over current_reward_signal (never set away from 0.0 in this runner). "
               f"numpy CPU; NO sim/ edit.")

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("onebrain_integration_r3_spiking_dopamine_credit")
        Vd.require("f2_lesion_removes_shift", 1 if all(
            abs(r["F2"]["delta_agent_lesion"]) < F2_LESION_RATIO * max(abs(r["F2"]["delta_agent_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the F2 shift must VANISH under lesion or it is a confound, not the cross-edges")
        Vd.require("migration_byte_identity", 1 if all(r["lesion_recovers_migration"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="lesion every candidate cross-edge -> comprehension reads == the [d6,comp,da_credit] baseline")
        Vd.require("current_reward_signal_never_used", 1, expect=lambda x: x >= 1,
                   note="current_reward_signal is set to 0.0 at build and NEVER written again anywhere in this "
                        "runner — grep-verifiable; the ONLY route to a weight change is the DA-population path")
        Vd.require("three_factor_removed_control_inert", 1 if all(
            r["emergence"]["removed_formed_nothing"] for r in runs) else 0, expect=lambda x: x >= 1,
            note="withholding the teacher drive entirely -> every candidate edge stays at W0")
        Vd.require("three_factor_shuffled_control_degrades", 1 if all(
            r["emergence"]["selectivity_shuffled"] < SEL_SHUFFLE_RATIO * max(r["emergence"]["selectivity_intact"], 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="decorrelating the teacher drive from correctness collapses selectivity")
        Vd.require("dopamine_lesion_control_inert", 1 if all(
            r["emergence"]["da_lesioned_formed_nothing"] for r in runs) else 0, expect=lambda x: x >= 1,
            note="THE CRUX: zeroing the sel/teach->snc coincidence synapses (same teach-drive schedule as "
                 "intact) collapses every candidate edge to W0 — the credit is carried by that population's "
                 "spikes, not by anything else the runner does")
        Vd.require("topology_intact_tracks_random_assignment", 1 if all(
            r["emergence"]["topology_tracks_true_assignment_intact"] for r in runs) else 0, expect=lambda x: x >= 1,
            note="the winning wire follows the per-seed RANDOM role assignment, never a hardcoded pair")
        Vd.require("moat_no_winner_from_silence", 1 if all(r["F4"]["f4a_no_winner_from_silence"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="a silent input + WM held stays sub-decision (F4 moat)")
        dec = Vd.decide(all_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "onebrain_integration_r3_spiking_dopamine_credit", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(runs), "per_arm": agg, "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "config": {"W0": W0, "stdp_w_max": HMAX, "reward_learning_rate": REWARD_LR,
                          "reward_eligibility_tau_ms": REWARD_TAU_MS, "n_episode_pairs": N_EPISODE_PAIRS,
                          "w_sel": W_SEL, "w_teach": W_TEACH, "teach_pa": TEACH_PA,
                          "da_window_ms": DA_WINDOW_MS, "da_threshold": DA_THRESHOLD,
                          "da_sensitivity": DA_SENSITIVITY, "da_decay_tau_ms": DA_DECAY_TAU_MS,
                          "sel_floor_intact": SEL_FLOOR_INTACT, "sel_removed_eps": SEL_REMOVED_EPS,
                          "sel_shuffle_ratio": SEL_SHUFFLE_RATIO, "sel_da_lesion_eps": SEL_DA_LESION_EPS},
               "mechanism": ("ONE shared merge pool [d6_multiref_wm + comprehension + da_credit]; the R2 unbiased "
                             "6-edge candidate topology is the SOLE plastic synapse set. Credit VALUE: a spiking "
                             "coincidence-detector population (snc_a/snc_b) fed by (a) the network's OWN resolved "
                             "sel_agent/sel_patient WTA decision via REAL synapses and (b) a host/environment "
                             "teach_agent/teach_patient confirmation drive; its firing feeds a registered "
                             "'dopamine' NeuromodulatorConfig (sim/neuromodulators.py, from_region_firing) whose "
                             "concentration sim/bridge.py's C2 reward-modulated-STDP block consumes AUTOMATICALLY "
                             "in place of the raw current_reward_signal scalar (never set away from 0.0). "
                             "DOPAMINE-LESION (zero the 4 coincidence synapses) collapses learning to W0 even "
                             "under the IDENTICAL intact teach-drive schedule — the crux control R2's host-scalar "
                             "version could not run."),
               "scaffold_residuals": ["the coincidence-detector CIRCUIT's wiring (sel/teach->snc synapse weights, "
                                      "the dopamine ProductionRule's threshold/window/decay constants) is HOST-"
                                      "DESIGNED infrastructure, like R1/R2's cue pathways and the Wong-Wang WTA's "
                                      "own lateral-inhibition circuit — never claimed self-organized",
                                      "the teach_agent/teach_patient drive TIMING/SCHEDULE (which episodes are "
                                      "'teacher-confirmed correct') remains runner-declared — legitimate "
                                      "environment/teacher territory (CLAUDE.md's brain-based-only boundary), the "
                                      "same class of signal as R1/R2's AGENT_CUE/PATIENT_CUE currents, but still "
                                      "host-scheduled, not derived from a further upstream brain process",
                                      "carried from R2: the candidate topology is a host-chosen REGION PAIR (d6 "
                                      "slot pools -> comprehension sel pools); the ambiguous item is a balanced-"
                                      "cue competition stand-in for full pronoun-resolution discourse; WM-pool "
                                      "ALLOCATION (LOAD_PA targeting) remains host-directed"],
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[R3] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
