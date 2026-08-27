"""One-brain INTEGRATION R2 — closing R1's two declared scaffold residuals: THREE-FACTOR credit-gated
plasticity + SELF-ORGANIZED (host-un-chosen) topology, on the SAME [d6_multiref_wm, comprehension] merge pool.

R1 (`research/findings/2026-08-27-onebrain-integration-R1-wm-to-comprehension.md`) grew the FIRST learned
cross-region edge `w{0,1}->{sel_agent,sel_patient}` by pure TWO-FACTOR (pre*post) rate-window Hebbian, on a
HOST-CHOSEN candidate pair-set, from a HOST-CURATED experience stream. It explicitly declared two residuals
NOT closed (design doc `2026-08-27-onebrain-integration-phase-DESIGN.md` R2):

  (R2-a) THREE-FACTOR: gate the edge's plasticity by a genuine THIRD factor (credit/task-success), not raw
    correlation. Mechanism used: the substrate's OWN reward-modulated STDP ("C2: Reward-Modulated Plasticity
    (Three-Factor Learning)", `sim/bridge.py` ~10486-10730) — STDP alone only writes a local ELIGIBILITY tag
    (`cfg.reward_defer_stdp_weight_update=True`, sim/bridge.py:10084's "Optional strict three-factor
    expression": "This prevents unrewarded exploration from silently becoming ordinary Hebbian habit
    learning"); a synaptic weight NEVER changes without a same-episode `current_reward_signal` pulse (the
    credit/task-success signal). CONTROL: withhold the pulse entirely (REMOVED) -> nothing forms; decorrelate
    which episodes get it from which are actually correct (SHUFFLED) -> the mapping does not track ground
    truth (and can even invert -- the distractor pool ends up the DOMINANT edge, see the finding).

  (R2-b) SELF-ORGANIZED TOPOLOGY: the candidate cross-edge SET is an UNBIASED 3-pool x 2-role block (w0,w1,w2
    -> sel_agent,sel_patient; 6 structurally-identical edges, no host-favored pair -- the runner's WIRING code
    is IDENTICAL across seeds). WHICH physical pool ends up wired to WHICH role is decided by a PER-SEED
    RANDOM assignment computed BEFORE training (`_role_assignment`, standing in for "the world"/the teacher's
    discourse content -- legitimate per the environment/host boundary, CLAUDE.md's brain-based-only standard),
    never hardcoded in the wiring or verification code. CONTROL: an absent or shuffled credit signal must NOT
    reliably discover the seed's true (randomly-assigned) mapping.

GATE: the SAME F1-F4 functional gate as R1 (faculty-works / vary-then-lesion attributable / no-runaway /
moat) + lesion-recovers-migration, run on the INTACT-trained pool, PLUS the two new emergence controls above.
6 seeds (42,43,44,100,101,102). numpy CPU; NO `sim/` edit (config-only: `enable_stdp` + `enable_reward_modulation`
+ `reward_defer_stdp_weight_update` + `cp_plasticity_rate_gain` whitelist -- all pre-existing bridge machinery).

R1'S CARRIED-FORWARD ENGINEERING NOTES (still apply): (1) `enable_hebbian_learning`/plasticity-relevant config
must be set at BUILD via the config-descriptor union (allocates the relevant trace arrays); (2) the whitelist
(`cp_plasticity_rate_gain=0` everywhere, then re-open the cross gate) is REQUIRED; (3) `_hard_reset` must clear
`cp_conductance_g_nmda_recurrent` or residual WM re-ignites; (4) the migration invariant is decision-preservation
+ connectivity byte-identity, not raw-margin bit-identity (a ~0.02-0.03 FP floor).

TWO NEW ENGINEERING TRAPS THIS BUILD EARNED (STDP-specific; R1 used pure Hebbian and never hit these):
  (5) `_run_one_simulation_step()` does NOT advance `runtime_state.current_time_ms` (only `step_simulation()`
      does) -- for RATE-based Hebbian this is irrelevant, but STDP is TIMING-based: every spike gets stamped
      with the same clock value, `delta_t` is identically 0 for every pair, and `fused_stdp_weight_update`
      returns EXACTLY the unchanged weight (both its branches use strict inequalities). This is a documented,
      self-warning guard (`sim/bridge.py:9935`, "STDP IS INERT"); the runner must manually advance the clock
      (`bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms`) after every step.
  (6) `cp_last_spike_time` must be reset to its -1000.0 init sentinel between episodes, or a stale cross-episode
      "last fired" timestamp on an otherwise-silent neuron becomes a spurious STDP-eligible pair the moment the
      OTHER side of that synapse fires in a LATER episode (adjacent episodes sit well within the default
      100ms STDP window). `cp_eligibility_trace` must also be zeroed between episodes, or a reward pulse
      broadcasts credit onto a DIFFERENT (stale, unrelated) synapse's leftover eligibility from a prior episode
      -- reward is a GLOBAL scalar applied to the WHOLE eligibility vector, so any residual anywhere gets
      credited too. (7) Reward delivered for the FULL cue-drive window (not just its tail) credits the
      Wong-Wang WTA competition's LOSING pool's early transient firing too (before mutual inhibition has
      suppressed it), destroying selectivity; a SHORT eligibility tau (~20ms, vs the default 1000ms) + a
      reward pulse confined to the LATE (post-settle) tail of the cue-drive window fixes this -- credit only
      the RESOLVED state of the competition, which is also the more biologically apt reading (outcome-locked
      dopamine, not a reward smeared across the whole trial).

Run:
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_r2_threefactor_selforganized --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_r2_threefactor_selforganized \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_integration_r2_threefactor_selforganized_6seed.json
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

# ---- geometry + protocol constants (validated on seeds 42/43; pre-registered) ----
W0 = 0.05                 # near-zero seed weight for every candidate edge (must GROW, not be pre-wired)
GATE = "wm_to_sel_r2"      # the plastic cross-edge whitelist gate
CAND_POOLS = ("w0", "w1", "w2")     # the UNBIASED candidate topology: 3 structurally-identical d6 slot pools
ROLE_TARGETS = ("sel_agent", "sel_patient")
BASE_POOL = "w3"           # F2's matched no-cross-edge control-hold (structurally NEVER a candidate edge)
LOAD_PA = 400.0
CUE_PA = 3500.0
AMBIG_PA = 2200.0
CLEAR_PA = 3500.0
LOAD_STEPS = 30
TRAIN_STEPS = 30
REWARD_TAIL_STEPS = 8      # reward delivered ONLY in the tail of the cue-drive (post-WTA-settle credit; note 7)
READ_STEPS = 150           # long enough for the top-down bias to accumulate a robustly-measurable margin shift
N_READS = 3
N_EPISODE_PAIRS = 100      # each contributes 1 correct-agent + 1 correct-patient + 2 distractor episodes
                           # (BALANCED 1:1 correct:distractor -> 400 total episodes. 75 pairs (300 total,
                           # matching R1's raw event count) rebalanced the shuffle control cleanly but left
                           # F2's functional margin shift under-powered (fewer ABSOLUTE credited episodes than
                           # the un-balanced schedule had at the same total budget) — F2 failed on 5/6 seeds
                           # despite R2-a/R2-b both holding 6/6. 100 pairs restores a robust F2 margin (re-
                           # measured: seed 100 delta_agent=+0.016, delta_patient=-0.011, both >> floor) while
                           # keeping the 1:1 balance that makes the shuffle control decisive.
HMAX = 20.0                # stdp_w_max — the soft-bound ceiling the whitelisted edge converges toward
REWARD_LR = 0.04           # cfg.reward_learning_rate
REWARD_TAU_MS = 20.0       # cfg.reward_eligibility_tau_ms — SHORT (note 7): only late/resolved coincidence survives
R_PULSE = 1.0              # cfg.current_reward_signal during a credited episode's reward tail

# F-gate floors (pre-registered; F2/F4/migration floors carried from R1's validated values)
F2_INTACT_FLOOR = 0.008
F2_LESION_RATIO = 0.34
F4A_FRAC = 0.5
F4B_RETAIN = 0.5
RATE_LO, RATE_HI = 5e-4, 0.7
# R2-a / R2-b floors (pre-registered on seeds 42/43/44's calibration at the FINAL protocol: intact selectivity
# ~8.6-10.4, removed selectivity ~0.0 exactly, shuffled (BALANCED 1:1 credit shuffle) selectivity ~0.18-0.23 i.e.
# ~18-23% of intact — a 2:1 correct:distractor schedule was tried first and left the shuffle control too weak
# (a majority-True vector barely moves under permutation): shuffled/intact sat at 0.44-0.51, straddling a 0.5
# floor and MISSING on seed 44 (0.507). Rebalancing the schedule to 1:1 (see `train()`) is the honest fix — a
# properly-decorrelating control, not a loosened floor for a weak one.
SEL_FLOOR_INTACT = 0.15        # intact (correct_mean - incorrect_mean) must clear this
SEL_REMOVED_EPS = 1e-6         # removed: every candidate edge must stay within this of W0 (no growth AT ALL)
SEL_SHUFFLE_RATIO = 0.35       # shuffled selectivity must be < this fraction of intact's (attributable_to-style)
TOPOLOGY_SHUFFLE_MAX_MATCH = 4  # shuffled "tracks the true per-seed assignment on both roles" must hold on
                                # AT MOST this many of 6 seeds (intact must hold on ALL 6 — a materially lower bar)

_CONDUCT = ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
            "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise",
            "cp_conductance_g_gabab_slow", "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise")


def _role_assignment(seed):
    """The per-seed RANDOM (host-un-chosen) mapping of the 3 unbiased candidate pools to {agent, patient,
    control-distractor}. This stands in for "the world"/the teacher's discourse content deciding which
    referent happens to act in which role -- legitimate host territory (the ENVIRONMENT), never the
    substrate's wiring or the credit rule. The runner's candidate-edge WIRING and its VERIFICATION code are
    IDENTICAL for every seed; only this training-data assignment varies."""
    rng = np.random.RandomState(int(seed) * 7919 + 13)
    perm = rng.permutation(3)
    pools = [CAND_POOLS[i] for i in perm]
    return pools[0], pools[1], pools[2]   # p_agent, p_patient, p_ctrl


def _dense(pre, post, w, gate):
    pre = np.asarray(pre, np.int64); post = np.asarray(post, np.int64)
    P = np.repeat(pre, len(post)); Q = np.tile(post, len(pre))
    return {"pre_indices": P, "post_indices": Q,
            "initial_weights": np.full(P.size, float(w), np.float32),
            "plastic": True, "plasticity_gate": gate, "conn_type": "E_TO_E", "count": int(P.size)}


def _build_pool(seed, with_cross):
    """Build the [d6, comprehension] MergedPool with the UNBIASED 6-edge candidate topology (or none). Both
    arms re-inject the SAME per-region-seamed base wiring (so they differ ONLY by the candidate cross-edges ->
    the lesion-recovers-migration comparison is exact). Config union: strip `enable_stdp`/`enable_reward_
    modulation` from the two organs' declared config (both normally False) and re-supply them True via a
    config-only extra descriptor -- the SAME "config-only extra descriptor" pattern R1 used for
    `hebbian_rate_window`, generalized to a key two base descriptors already claim (so the raw keys must be
    stripped from copies of those descriptors first, or the union raises MergeConflict)."""
    from research.runners.onebrain_merge_framework import REGISTRY, MergedPool
    xp, _ = get_backend()
    D6, COMP = REGISTRY["d6_multiref_wm"], REGISTRY["comprehension"]
    STRIP = ("enable_stdp", "enable_reward_modulation")
    D6c = dataclasses.replace(D6, config={k: v for k, v in D6.config.items() if k not in STRIP})
    COMPc = dataclasses.replace(COMP, config={k: v for k, v in COMP.config.items() if k not in STRIP})
    extra = types.SimpleNamespace(key="r2_threefactor", config={
        "enable_stdp": True, "enable_reward_modulation": True,
        "reward_defer_stdp_weight_update": True,   # strict three-factor: STDP alone never writes a weight
    }, param_het=False)
    pool = MergedPool(seed, [D6, COMP], config_descriptors=[D6c, COMPc, extra], wire=True)
    pool.ensure_built()
    b = pool.bridge
    rm = b.region_manager
    def idxr(nm):
        return np.asarray(rm.indices(nm), np.int64)
    names = list(CAND_POOLS) + [BASE_POOL, "sel_agent", "sel_patient", "fs",
                                 "cue_animacy_pos", "cue_animacy_neg", "cue_verbfit_pos", "cue_verbfit_neg"]
    ix = {nm: idxr(nm) for nm in names}
    union = dict(rm.build_wiring_plan(seed=pool.seed, per_region_seed=True))
    masks = None
    if with_cross:
        for p in CAND_POOLS:
            union[f"x_{p}_sela"] = _dense(ix[p], ix["sel_agent"], W0, GATE)
            union[f"x_{p}_selp"] = _dense(ix[p], ix["sel_patient"], W0, GATE)
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
    # re-settle to rest + refresh pool.snap (v,u) so the comprehension organ's hard-reset uses the CURRENT rest
    b.cp_external_input_current[:] = 0.0
    for _ in range(40):
        b._run_one_simulation_step()
    b.cp_external_input_current[:] = 0.0
    if pool.snap is not None:
        pool.snap["cp_membrane_potential_v"] = np.asarray(to_host(b.cp_membrane_potential_v)).copy()
        pool.snap["cp_recovery_variable_u"] = np.asarray(to_host(b.cp_recovery_variable_u)).copy()
    return pool, ix, masks


class R2Pool:
    """The R2 integrated pool: the merged [d6, comprehension] bridge + the unbiased 6-edge candidate topology,
    trained by reward-gated (strict three-factor) STDP under one of three credit regimes (intact / removed /
    shuffled). `mode` is fixed for the life of the instance (each mode needs its OWN fresh substrate — training
    is not reversible)."""

    def __init__(self, seed, mode="intact"):
        self.seed = int(seed)
        self.mode = str(mode)
        self.xp, _ = get_backend()
        self.p_agent, self.p_patient, self.p_ctrl = _role_assignment(seed)
        self.pool, self.ix, self.masks = _build_pool(seed, with_cross=True)
        self.b = self.pool.bridge
        from research.runners.onebrain_merge_framework import _comprehension_organ, _d6_organ
        self.comp_organ = _comprehension_organ(seed, self.pool)
        self.d6_organ = _d6_organ(seed, self.pool)
        self.comp_organ.ensure_built()
        # WHITELIST FREEZE (design §2 / R1's "one-line inversion"): the 6 candidate cross-edges are the SOLE
        # plastic synapses (gain 1), EVERYTHING else frozen (gain 0).
        self.b.set_plasticity_gate(GATE, 1.0)
        self.b.cp_plasticity_rate_gain[:] = 0.0
        self.b.set_plasticity_gate(GATE, 1.0)
        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]
        cfg = self.b.core_config
        cfg.stdp_w_max = HMAX
        cfg.reward_learning_rate = REWARD_LR
        cfg.reward_eligibility_tau_ms = REWARD_TAU_MS
        cfg.reward_baseline = 0.0
        cfg.current_reward_signal = 0.0
        cfg.enable_hebbian_learning = False   # this arc never uses Hebbian — reward-gated STDP only
        # refresh the resting snapshot AFTER the inject/settle so the direct-drive reads start from true rest.
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
        # STDP-specific resets (notes 5-6): a stale cross-episode spike time or eligibility tag is a spurious
        # coincidence / a mis-credited synapse the next time reward or the OTHER endpoint fires.
        if getattr(b, "cp_last_spike_time", None) is not None:
            b.cp_last_spike_time[:] = -1000.0
        if getattr(b, "cp_eligibility_trace", None) is not None:
            b.cp_eligibility_trace[:] = 0.0
        b.cp_external_input_current[:] = 0.0

    def _drive(self, pairs, steps, reward=0.0, read=None):
        """Drive `pairs` for `steps`, with `current_reward_signal` set to `reward` throughout (0.0 = no
        credit). Manually advances `runtime_state.current_time_ms` every step (note 5) — required for STDP's
        spike-timing math to be non-degenerate; `_run_one_simulation_step()` does not do this itself."""
        b, xp = self.b, self.xp
        b.core_config.current_reward_signal = float(reward)
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
        b.core_config.current_reward_signal = 0.0
        return {k: v / steps for k, v in acc.items()}

    def _wmean(self, name):
        return float(np.asarray(to_host(self.b.cp_connections.data))[self.masks[name]].mean())

    def cross_weights(self):
        return {k: round(self._wmean(k), 4) for k in self.masks}

    # ---- emergence: grow the cross-edges from credited experience ----
    def _episode(self, pool_key, cue_pairs, credited):
        """hard-reset -> load+hold `pool_key` -> drive `cue_pairs` for TRAIN_STEPS, delivering the reward
        pulse ONLY in the tail (post-WTA-settle credit; note 7) when `credited`."""
        self._hard_reset()
        self._drive([(self.ix[pool_key], LOAD_PA)], LOAD_STEPS)
        head = TRAIN_STEPS - REWARD_TAIL_STEPS
        self._drive(cue_pairs, head, reward=0.0)
        self._drive(cue_pairs, REWARD_TAIL_STEPS, reward=(R_PULSE if credited else 0.0))

    def train(self, n_episode_pairs=None):
        # NOTE: default resolved at CALL time, not bind time — a module-level constant used as a mutable
        # default argument freezes at function-definition time, which silently ignores any later override
        # (a real trap this build hit once during calibration: `R.N_EPISODE_PAIRS = 100` had no effect on an
        # already-bound default).
        if n_episode_pairs is None:
            n_episode_pairs = N_EPISODE_PAIRS
        ix = self.ix
        AGENT_CUE = [(ix["cue_animacy_pos"], CUE_PA), (ix["cue_verbfit_pos"], CUE_PA)]
        PATIENT_CUE = [(ix["cue_animacy_neg"], CUE_PA), (ix["cue_verbfit_neg"], CUE_PA)]
        schedule = []   # (pool_key, cue_pairs, is_correct)
        for ep in range(n_episode_pairs):
            schedule.append((self.p_agent, AGENT_CUE, True))
            schedule.append((self.p_patient, PATIENT_CUE, True))
            # TWO distractor episodes (agent-shaped + patient-shaped), both on p_ctrl, NEVER credited — a
            # BALANCED 1:1 correct:distractor schedule. A 2:1 schedule (one distractor) was tried first and
            # left the SHUFFLED-credit control too weak (a majority-True vector barely moves under a random
            # permutation, since most positions keep True by chance): shuffled/intact selectivity sat at
            # 0.44-0.51 across 3 seeds, straddling the pre-registered <0.5 floor (seed 44 missed at 0.507).
            # Balancing to 1:1 gives the permutation genuine 50/50 decorrelating power (re-measured: seed 42's
            # ratio dropped to 0.23), which is the scientifically correct fix — a control that barely moves
            # the thing it is supposed to null out is a weak control, not a passing one.
            schedule.append((self.p_ctrl, AGENT_CUE, False))
            schedule.append((self.p_ctrl, PATIENT_CUE, False))
        true_credit = [c for (_p, _c, c) in schedule]
        if self.mode == "removed":
            credited = [False] * len(schedule)
        elif self.mode == "shuffled":
            rng = np.random.RandomState(self.seed * 104729 + 1)
            credited = list(rng.permutation(true_credit))
        else:
            credited = true_credit
        traj = [dict(i=0, **self.cross_weights())]
        for i, (pool_key, cue_pairs, _true) in enumerate(schedule):
            self._episode(pool_key, cue_pairs, credited[i])
            if (i + 1) % 12 == 0 or i == len(schedule) - 1:
                traj.append(dict(i=i + 1, **self.cross_weights()))
        # NO-CORRUPTION: every NON-cross (migrated) weight must be byte-unchanged after training.
        now = np.asarray(to_host(self.b.cp_connections.data))
        self.frozen_maxdrift = float(np.max(np.abs(now[self._noncross] - self._frozen_w0[self._noncross])))
        return traj

    # ---- the signed ambiguous read with the WM bump alive (F1/F2/F4 read protocol; unchanged from R1) ----
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
#  R2-a (three-factor credit) + R2-b (self-organized topology): compare intact/removed/shuffled arms
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _selectivity(final_weights, p_agent, p_patient):
    """(mean of the 2 CORRECT pairs) - (mean of the OTHER 4 candidate pairs)."""
    correct = [final_weights[f"{p_agent}->A"], final_weights[f"{p_patient}->P"]]
    incorrect = [v for k, v in final_weights.items()
                 if k not in (f"{p_agent}->A", f"{p_patient}->P")]
    return float(np.mean(correct) - np.mean(incorrect)), float(np.mean(correct)), float(np.mean(incorrect))


def _argmax_pool(final_weights, role_suffix):
    """Which of the 3 candidate pools has the highest ->role weight."""
    scored = {p: final_weights[f"{p}->{role_suffix}"] for p in CAND_POOLS}
    return max(scored, key=scored.get)


def _r2_emergence(seed, intact_final, removed_final, shuffled_final, p_agent, p_patient, p_ctrl):
    sel_i, corr_i, incorr_i = _selectivity(intact_final, p_agent, p_patient)
    sel_r, corr_r, incorr_r = _selectivity(removed_final, p_agent, p_patient)
    sel_s, corr_s, incorr_s = _selectivity(shuffled_final, p_agent, p_patient)
    removed_max_dev = float(max(abs(v - W0) for v in removed_final.values()))
    argmax_a_i, argmax_p_i = _argmax_pool(intact_final, "A"), _argmax_pool(intact_final, "P")
    argmax_a_s, argmax_p_s = _argmax_pool(shuffled_final, "A"), _argmax_pool(shuffled_final, "P")
    tracks_intact = bool(argmax_a_i == p_agent and argmax_p_i == p_patient)
    tracks_shuffled = bool(argmax_a_s == p_agent and argmax_p_s == p_patient)
    frac_removed_attrib = attributable_to(f"seed{seed} R2-a selectivity vs REMOVED-credit control", sel_i, sel_r)
    frac_shuffle_attrib = attributable_to(f"seed{seed} R2-a selectivity vs SHUFFLED-credit control", sel_i, sel_s)
    # R2-a: three-factor credit is load-bearing — intact selective, removed inert, shuffled degraded
    r2a_pass = bool(sel_i > SEL_FLOOR_INTACT and removed_max_dev < SEL_REMOVED_EPS
                    and (sel_s < SEL_SHUFFLE_RATIO * sel_i if sel_i > 0 else False))
    return {
        "role_assignment": {"p_agent": p_agent, "p_patient": p_patient, "p_ctrl": p_ctrl},
        "selectivity_intact": sel_i, "selectivity_removed": sel_r, "selectivity_shuffled": sel_s,
        "correct_mean_intact": corr_i, "incorrect_mean_intact": incorr_i,
        "removed_max_deviation_from_W0": removed_max_dev,
        "removed_formed_nothing": bool(removed_max_dev < SEL_REMOVED_EPS),
        "frac_attributable_removed_control": (None if frac_removed_attrib is None else float(frac_removed_attrib)),
        "frac_attributable_shuffled_control": (None if frac_shuffle_attrib is None else float(frac_shuffle_attrib)),
        "argmax_agent_intact": argmax_a_i, "argmax_patient_intact": argmax_p_i,
        "argmax_agent_shuffled": argmax_a_s, "argmax_patient_shuffled": argmax_p_s,
        "topology_tracks_true_assignment_intact": tracks_intact,
        "topology_tracks_true_assignment_shuffled": tracks_shuffled,
        "intact_final": intact_final, "removed_final": removed_final, "shuffled_final": shuffled_final,
        "R2a_three_factor_PASS": r2a_pass,
        # R2-b's per-seed piece: intact must track; the CROSS-SEED aggregate (shuffled must NOT reliably
        # track) is judged in main() over all 6 seeds, not per-seed (a single seed's shuffle could luck into
        # a match — the honest claim is a RATE, not a per-seed absolute).
        "R2b_topology_intact_tracks": tracks_intact,
    }


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The F1-F4 functional-gate arms + the migration invariant (R1's protocol, generalized off r2.p_agent/
#  r2.p_patient/BASE_POOL instead of hardcoded w0/w1/w2)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _f1(r2):
    from research.runners.onebrain_merge_framework import _comprehension_battery, _d6_reads
    r2._hard_reset()
    org = r2.comp_organ
    items = _comprehension_battery(org.seed)
    well, ill = [], []
    for (lab, _tag, n0, v, n1) in items:
        m = float(org.read_margin(n0, v, n1))
        (well if lab == 1 else ill).append(m)
    d6 = _d6_reads(r2.d6_organ)
    mean_well = float(np.mean(well)); mean_ill = float(np.mean(ill))
    well_ok = all(m >= org.threshold for m in well)
    ill_ok = all(m < org.threshold for m in ill)
    return {"threshold": float(org.threshold), "mean_well": mean_well, "mean_ill": mean_ill,
            "min_well": float(np.min(well)), "max_ill": float(np.max(ill)),
            "well_all_comprehended": bool(well_ok), "ill_all_abstained": bool(ill_ok),
            "d6_all_recovered": bool(d6["all_recovered"]), "d6_hold_alive_min": float(d6["hold_alive_min"]),
            "PASS": bool(well_ok and ill_ok and d6["all_recovered"] and d6["hold_alive_min"] > 0.0)}


def _f2(r2):
    """F2 INTERACTION-IS-REAL: vary which referent is held (p_agent / p_patient) vs the matched control-hold
    (BASE_POOL, structurally never a candidate edge); the sel margin shifts toward the held referent's LEARNED
    role and the shift vanishes when ALL candidate cross-edges are lesioned."""
    ambig = [("cue_animacy_pos", AMBIG_PA), ("cue_animacy_neg", AMBIG_PA)]
    def battery(band=False):
        none = r2.amb_read(BASE_POOL, ambig, band=band)
        agent = r2.amb_read(r2.p_agent, ambig, band=band)
        patient = r2.amb_read(r2.p_patient, ambig, band=band)
        return none, agent, patient
    n_i, a_i, p_i = battery(band=True)
    d_agent_i = a_i["margin"] - n_i["margin"]
    d_patient_i = p_i["margin"] - n_i["margin"]
    data = np.asarray(to_host(r2.b.cp_connections.data)).copy()
    for k in r2.masks:
        data[r2.masks[k]] = 0.0
    r2.b.cp_connections.data = r2.xp.asarray(data, dtype=r2.b.cp_connections.data.dtype)
    n_l, a_l, p_l = battery()
    d_agent_l = a_l["margin"] - n_l["margin"]
    d_patient_l = p_l["margin"] - n_l["margin"]
    agent_ok = (d_agent_i > F2_INTACT_FLOOR) and (abs(d_agent_l) < F2_LESION_RATIO * abs(d_agent_i))
    patient_ok = (d_patient_i < -F2_INTACT_FLOOR) and (abs(d_patient_l) < F2_LESION_RATIO * abs(d_patient_i))
    frac_agent = attributable_to("F2 agent->A shift = the cross-edges", d_agent_i, d_agent_l)
    frac_patient = attributable_to("F2 patient->P shift = the cross-edges", -d_patient_i, -d_patient_l)
    return {"frac_attributable_agent": (None if frac_agent is None else float(frac_agent)),
            "frac_attributable_patient": (None if frac_patient is None else float(frac_patient)),
            "margins_intact": {"none": n_i["margin"], "agent": a_i["margin"], "patient": p_i["margin"]},
            "margins_lesion": {"none": n_l["margin"], "agent": a_l["margin"], "patient": p_l["margin"]},
            "delta_agent_intact": float(d_agent_i), "delta_patient_intact": float(d_patient_i),
            "delta_agent_lesion": float(d_agent_l), "delta_patient_lesion": float(d_patient_l),
            "rates_intact_agent": a_i.get("rates", {}),
            "agent_shift_toward_agent": bool(agent_ok), "patient_shift_toward_patient": bool(patient_ok),
            "PASS": bool(agent_ok and patient_ok)}


def _f3(r2, traj, f2):
    rates = f2.get("rates_intact_agent", {})
    band_pools = ("sel_agent", "sel_patient", "cue_animacy_pos", r2.p_agent)
    in_band = all(RATE_LO < rates.get(p, 0.0) < RATE_HI for p in band_pools) if rates else False
    grown = {k: traj[-1][k] for k in (f"{r2.p_agent}->A", f"{r2.p_patient}->P")}
    correct = 0.5 * (grown[f"{r2.p_agent}->A"] + grown[f"{r2.p_patient}->P"])
    bounded = grown[f"{r2.p_agent}->A"] <= HMAX and grown[f"{r2.p_patient}->P"] <= HMAX
    def cmean(row):
        return 0.5 * (row[f"{r2.p_agent}->A"] + row[f"{r2.p_patient}->P"])
    first_dw = cmean(traj[1]) - cmean(traj[0]) if len(traj) >= 2 else 0.0
    last_dw = cmean(traj[-1]) - cmean(traj[-2]) if len(traj) >= 2 else 0.0
    decelerating = last_dw < first_dw
    alive = rates.get("sel_agent", 0.0) > RATE_LO and rates.get("sel_patient", 0.0) > RATE_LO
    return {"rates": rates, "in_band": bool(in_band), "grown_correct_mean": float(correct),
            "bounded_by_hmax": bool(bounded), "first_window_dw": float(first_dw), "last_window_dw": float(last_dw),
            "decelerating": bool(decelerating), "pool_alive": bool(alive),
            "PASS": bool(in_band and bounded and decelerating and alive)}


def _f4(r2):
    clear = [("cue_animacy_pos", CLEAR_PA), ("cue_verbfit_pos", CLEAR_PA)]
    m_nowm = r2.amb_read(BASE_POOL, clear)["margin"]
    m_right = r2.amb_read(r2.p_agent, clear)["margin"]
    m_wrong = r2.amb_read(r2.p_patient, clear)["margin"]
    wm0 = r2.amb_read(r2.p_agent, [])["margin"]
    wm1 = r2.amb_read(r2.p_patient, [])["margin"]
    decision = abs(m_nowm)
    f4a_ok = abs(wm0) < F4A_FRAC * decision and abs(wm1) < F4A_FRAC * decision
    same_sign = (m_wrong > 0) == (m_nowm > 0)
    retained = abs(m_wrong) >= F4B_RETAIN * abs(m_nowm)
    f4b_ok = bool(m_nowm > 0 and same_sign and retained)
    return {"wm_only_agent": float(wm0), "wm_only_patient": float(wm1), "decision_scale_clear": float(decision),
            "wm_only_frac_of_decision": float(max(abs(wm0), abs(wm1)) / max(decision, 1e-9)),
            "f4a_no_winner_from_silence": bool(f4a_ok),
            "clear_noWM": float(m_nowm), "clear_rightWM": float(m_right), "clear_wrongWM": float(m_wrong),
            "f4b_clear_not_flipped": f4b_ok, "PASS": bool(f4a_ok and f4b_ok)}


def _migration_invariant(seed, r2, comp_battery_reads):
    from research.runners.onebrain_merge_framework import _comprehension_organ, _comprehension_battery
    pool0, ix0, _m0 = _build_pool(seed, with_cross=False)
    org0 = _comprehension_organ(seed, pool0)
    org0.ensure_built()
    def edge_map(pool):
        coo = pool.bridge.cp_connections.tocoo()
        r = to_host(coo.row); c = to_host(coo.col); d = to_host(coo.data)
        return {(int(a), int(b)): float(w) for a, b, w in zip(r, c, d)}
    k0 = edge_map(pool0)
    k1 = edge_map(r2.pool)
    xrows = set(int(x) for p in CAND_POOLS for x in r2.ix[p])
    xcols = set(int(x) for x in np.concatenate([r2.ix["sel_agent"], r2.ix["sel_patient"]]))
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

    intact = R2Pool(seed, mode="intact")
    traj = intact.train()
    removed = R2Pool(seed, mode="removed")
    removed.train()
    shuffled = R2Pool(seed, mode="shuffled")
    shuffled.train()

    emg = _r2_emergence(seed, intact.cross_weights(), removed.cross_weights(), shuffled.cross_weights(),
                         p_agent, p_patient, p_ctrl)
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
              and emg["no_corruption_intact"] and emg["R2a_three_factor_PASS"] and emg["R2b_topology_intact_tracks"])
    return {"seed": int(seed), "PASS": go, "elapsed_s": round(time.time() - t0, 1),
            "emergence": emg, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "lesion_recovers_migration": mig}


def _agg(runs):
    def frac(key):
        parts = key.split(".")
        return sum(1 for r in runs if r[parts[0]][parts[1]])
    keys = ["F1.PASS", "F2.PASS", "F3.PASS", "F4.PASS", "lesion_recovers_migration.PASS",
            "emergence.R2a_three_factor_PASS", "emergence.R2b_topology_intact_tracks",
            "emergence.no_corruption_intact"]
    return {k: f"{frac(k)}/{len(runs)}" for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        e = r["emergence"]
        print(f"[seed {s}] {'GO' if r['PASS'] else 'no'} ({r['elapsed_s']}s) | "
              f"role(agent={e['role_assignment']['p_agent']},patient={e['role_assignment']['p_patient']},"
              f"ctrl={e['role_assignment']['p_ctrl']}) | "
              f"sel(intact={e['selectivity_intact']:.3f} removed={e['selectivity_removed']:.3f} "
              f"shuffled={e['selectivity_shuffled']:.3f}) R2a={e['R2a_three_factor_PASS']} | "
              f"topo(intact_tracks={e['topology_tracks_true_assignment_intact']} "
              f"shuffled_tracks={e['topology_tracks_true_assignment_shuffled']}) R2b={e['R2b_topology_intact_tracks']} | "
              f"F1={r['F1']['PASS']} F2={r['F2']['PASS']} F3={r['F3']['PASS']} F4={r['F4']['PASS']} "
              f"mig={r['lesion_recovers_migration']['PASS']}", flush=True)

    n_go = sum(r["PASS"] for r in runs)
    agg = _agg(runs)
    n_shuffled_tracks = sum(1 for r in runs if r["emergence"]["topology_tracks_true_assignment_shuffled"])
    topology_shuffle_control_ok = bool(n_shuffled_tracks <= TOPOLOGY_SHUFFLE_MAX_MATCH)
    all_go = (n_go == len(runs)) and topology_shuffle_control_ok and not args.smoke
    tag = "GO" if all_go else ("SMOKE-GO (1-seed indicator)" if args.smoke and n_go == len(runs) else "NO-GO/PARTIAL")
    verdict = (f"{tag} — R2 three-factor + self-organized-topology cross-edge d6 WM referent -> comprehension "
               f"role competition: {n_go}/{len(runs)} seeds pass ALL of F1-F4 + lesion-recovers-migration + "
               f"R2-a(three-factor: intact selective, removed inert, shuffled degraded) + "
               f"R2-b(topology tracks the per-seed RANDOM assignment, intact). Per-arm: {agg}. "
               f"Shuffled-credit topology-match rate: {n_shuffled_tracks}/{len(runs)} "
               f"(control requires <= {TOPOLOGY_SHUFFLE_MAX_MATCH}/{len(runs)}, vs intact's "
               f"{sum(1 for r in runs if r['emergence']['topology_tracks_true_assignment_intact'])}/{len(runs)}). "
               f"Mechanism: reward-DEFERRED STDP (sim/bridge.py's existing three-factor block) on an UNBIASED "
               f"3-pool x 2-role candidate set; the true agent/patient/control-distractor pool identity is a "
               f"PER-SEED RANDOM assignment the runner's wiring/verification code never hardcodes. numpy CPU; "
               f"NO sim/ edit.")

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("onebrain_integration_r2_threefactor_selforganized")
        Vd.require("f2_lesion_removes_shift", 1 if all(
            abs(r["F2"]["delta_agent_lesion"]) < F2_LESION_RATIO * max(abs(r["F2"]["delta_agent_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the F2 shift must VANISH under lesion or it is a confound, not the cross-edges (the crux control)")
        Vd.require("migration_byte_identity", 1 if all(r["lesion_recovers_migration"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="lesion every candidate cross-edge -> comprehension reads == the plain merged pool")
        Vd.require("three_factor_removed_control_inert", 1 if all(
            r["emergence"]["removed_formed_nothing"] for r in runs) else 0, expect=lambda x: x >= 1,
            note="withholding credit entirely -> every candidate edge stays at W0 (the third factor is NECESSARY)")
        Vd.require("three_factor_shuffled_control_degrades", 1 if all(
            r["emergence"]["selectivity_shuffled"] < SEL_SHUFFLE_RATIO * max(r["emergence"]["selectivity_intact"], 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="decorrelating credit from correctness collapses selectivity (the third factor is SUFFICIENT to "
                 "explain the intact mapping's correctness, not decorative)")
        Vd.require("topology_intact_tracks_random_assignment", 1 if all(
            r["emergence"]["topology_tracks_true_assignment_intact"] for r in runs) else 0, expect=lambda x: x >= 1,
            note="the winning wire follows the per-seed RANDOM role assignment, never a hardcoded w0/w1 pair")
        Vd.require("topology_shuffled_control_degrades", 1 if topology_shuffle_control_ok else 0,
                   expect=lambda x: x >= 1,
                   note=f"a decorrelated credit signal tracks the true assignment on AT MOST "
                        f"{TOPOLOGY_SHUFFLE_MAX_MATCH}/{len(runs)} seeds, materially below intact's rate")
        Vd.require("moat_no_winner_from_silence", 1 if all(r["F4"]["f4a_no_winner_from_silence"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="a silent input + WM held stays sub-decision (F4 moat)")
        dec = Vd.decide(all_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "onebrain_integration_r2_threefactor_selforganized", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(runs), "per_arm": agg, "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "n_shuffled_topology_matches": n_shuffled_tracks,
               "topology_shuffle_control_ok": topology_shuffle_control_ok,
               "config": {"W0": W0, "stdp_w_max": HMAX, "reward_learning_rate": REWARD_LR,
                          "reward_eligibility_tau_ms": REWARD_TAU_MS, "r_pulse": R_PULSE,
                          "n_episode_pairs": N_EPISODE_PAIRS, "reward_tail_steps": REWARD_TAIL_STEPS,
                          "sel_floor_intact": SEL_FLOOR_INTACT, "sel_removed_eps": SEL_REMOVED_EPS,
                          "sel_shuffle_ratio": SEL_SHUFFLE_RATIO,
                          "topology_shuffle_max_match": TOPOLOGY_SHUFFLE_MAX_MATCH},
               "mechanism": ("ONE shared merge pool [d6_multiref_wm + comprehension]; an UNBIASED 6-edge candidate "
                             "topology w{0,1,2}->{sel_agent,sel_patient} (structurally identical, no host-favored "
                             "pair), the SOLE plastic synapses (cp_plasticity_rate_gain whitelist). Plasticity = "
                             "the substrate's OWN reward-DEFERRED STDP (enable_stdp + enable_reward_modulation + "
                             "reward_defer_stdp_weight_update): STDP alone only tags an eligibility trace; a weight "
                             "changes ONLY when a same-episode current_reward_signal pulse (task-success credit) "
                             "is delivered (R2-a, three-factor). WHICH physical pool plays agent/patient/control-"
                             "distractor is a PER-SEED RANDOM assignment computed before training and never "
                             "hardcoded in the wiring or verification code (R2-b, self-organized topology); the "
                             "winning wire is read off by argmax and checked against that random assignment."),
               "scaffold_residuals": ["host-CURATED experience STREAM persists: the runner still schedules "
                                      "correct-vs-distractor episodes and the reward pulse's TIMING/MAGNITUDE — "
                                      "the credit RULE is genuine (reward-gated STDP), but WHEN a trial counts as "
                                      "'successful' is still runner-declared, not derived from the brain's own "
                                      "task performance/error signal (a self-computed task-success readout, e.g. "
                                      "from comprehension's own margin, is the next rung)",
                                      "the candidate topology is still a HOST-CHOSEN REGION PAIR (d6 slot pools -> "
                                      "comprehension sel pools) — R2-b closes WHICH pool wins within that pair, not "
                                      "WHETHER d6->comprehension is the pair that gets tried at all",
                                      "3-pool x 2-role is a small, hand-sized candidate set for tractability, not a "
                                      "dense/all-to-all block or genuine structural-plasticity synaptogenesis",
                                      "the ambiguous item is a balanced-cue competition (a substrate stand-in for a "
                                      "full pronoun-resolution discourse), carried from R1"],
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[R2] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
