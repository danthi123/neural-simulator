"""One-brain INTEGRATION — a LEARNED cross-edge SURPRISE (prediction-error) -> episodic/provenance ENCODING gate
("does this get committed as noteworthy/self-generated"): audit rung "(5) surprise->episodic gate"
(`research/findings/2026-08-27-onebrain-completeness-audit.md` sec 4 step 5), built DECLARATIVELY on the
`CrossEdge`/`merge_organs(cross_edges=)` framework (`research/findings/2026-08-27-declarative-cross-edges-
framework-GO.md`), mirroring the R1 (`_onebrain_integration_r1_wm_comprehension.py`) / R4
(`_onebrain_integration_r4_selfschema_provenance.py`) learned-cross-edge template verbatim in SHAPE.

HONEST SCOPE SUBSTITUTION (verified BEFORE building, not discovered after) -- the literal ask was
"surprise -> the EPISODIC (D5) organ", but `d5_episodic` is NOT migration-ready: it is a `GROUP_A_DEFERRED` entry
in `onebrain_merge_framework.py` ("Heavy own-pool -- a ~2000-neuron CA3 with two-compartment apical dendritic-dAP
+ slow-NMDA reverberation + BTSP formation. Group-C own-pool + apical/NMDA-slow seam"), and the completeness
audit's own prioritized roadmap step 5 ("R3 (DESIGN) surprise->episodic/provenance ENCODING gate ... Depends on
step 6") explicitly gates the FULL d5_episodic pairing on step 6 ("Migrate d5_episodic (Group-C own-pool seam)"
-- a separate, heavy, independent migration lane, NOT a tiny de-risk: the 2026-08-12 D5 finding measured a
SINGLE BTSP store at ~510s on numpy@2000 neurons, and this de-risk's GO bar needs dozens of encode events x 6
seeds x anti-cheat -- hours to tens of hours, incompatible with the CPU/numpy "tiny" budget this arc was given.
R4 hit the identical fork one arc earlier and substituted `source_provenance` for the same reason (see its
module docstring); this run makes the SAME substitution, EXPLICITLY, for the SAME rung. The audit's own step-5
title names BOTH acceptable targets for this exact biological claim ("surprise->episodic/PROVENANCE ENCODING
gate") -- `source_provenance`'s `prov_generated` pool (the substrate's own "this is being tagged as internally-
generated/noteworthy" ENCODING read-out, board #129, already GROUP_A-migrated + already proven as an R4 cross-
edge TARGET) is the audit-sanctioned, currently-buildable half of the SAME rung, not an unrelated substitution.
STAND-IN, DECLARED: `prov_generated` firing is used here as an ENCODING-COMMITMENT proxy (a pool that lights up
when new content is being freshly tagged/flagged), NOT a literal CA3 autobiographical memory trace -- the full
"does topic X's hippocampal assembly form more readily when surprising" claim rides the d5_episodic Group-C
migration (step 6), a named follow-on, not this arc.

THE BIOLOGY (Lisman & Grace 2005 VTA-hippocampal novelty loop; Kafkas & Montaldi 2018 novelty/memory-strength):
novelty/prediction-error GATES what gets encoded -- a mismatch (surprise) signal should preferentially bias a
co-temporal, genuinely ambiguous encoding-context read toward "this is being freshly/actively encoded"
(GENERATED), exactly the source-monitoring coupling R4 already validated for self_schema's authorship signal,
now driven by SURPRISE'S OWN prediction-error firing instead.

THE MECHANISM (emergence-compliant; NO sim/ edit):
  * ONE shared spiking bridge holds BOTH organs' regions: SURPRISE (the D2 expectation-violation circuit,
    `research.runners._spiking_expectation_rpe_derisk.build_expectation_circuit` via the framework's `SURPRISE`
    OrganDescriptor -- cue -(FIXED topographic 0.8)-> patient_expected(FS/GABA_A) ; patient_asserted -(FIXED
    exc)-> surprise ; patient_expected -(FIXED inh)-> surprise -- ALL THREE pathways are FIXED/block-diagonal at
    BUILD time, not Hebbian-trained, so surprise's own mismatch detector needs no training phase) + SOURCE_
    PROVENANCE (episode/ctx_*/prov_perceived/prov_generated opponent trace, board #129, UNCHANGED from R4).
    `merge_organs([SURPRISE_LITE, source_provenance], wire=True, cross_edges=[...])`.
  * CONFIG RECONCILIATION (found + fixed BEFORE the smoke, not after): the registered `SURPRISE` descriptor's
    `_POOL1_CONFIG` sets `enable_hebbian_learning=True` (needed by `worldmodel`, SURPRISE's usual pool-1
    co-resident, for ITS OWN learned prediction pathway) -- a direct `MergeConflict` against
    `source_provenance`'s `enable_hebbian_learning=False` requirement (R4's exact hazard, `hebbian_rate_window`
    hijacking source_provenance's own prov_learn/content_learn edges onto an uncalibrated rule). SURPRISE's OWN
    circuit does NOT need live Hebbian learning (all 3 of its pathways are fixed block-diagonal installs, see
    above), so `SURPRISE_LITE = dataclasses.replace(SURPRISE, config={**SURPRISE.config,
    "enable_hebbian_learning": False, "hebbian_rate_window": False})` reuses the IDENTICAL spec_fn/post_build/
    idx_fn (byte-identical surprise circuit) with a build-time-safe config, resolving the conflict without
    touching source_provenance's descriptor at all. `enable_hebbian_learning` is flipped LIVE (True) only around
    the cross-edge's own training window (R1/R4's pattern), with the STANDARD (non-rate-window) symmetric rule
    R4's fix established.
  * ONE plastic cross-edge SURPRISE.surprise -> source_provenance.prov_generated, declared as a `CrossEdge` row
    (init_weight=0.05, the SOLE plastic synapse via `MergedPool.apply_cross_edge_freeze()`'s whitelist
    inversion). It GROWS by the substrate's OWN standard same-step Hebbian rule over episodes that co-drive a
    CONTRADICT (mismatch) trial on the surprise circuit -- cue block c (the recall probe) + patient_asserted
    block c' != c (the false assertion), which the FIXED wiring above turns into "surprise" firing SPECIFICALLY
    in block c' -- with source_provenance's `ctx_generated` line (a FIXED, non-plastic pathway to
    `prov_generated`, R4's own de-risk constant), so `prov_generated` reliably co-fires with the block-c' slice
    of `surprise`, Hebbian-binding that slice's edges to `prov_generated`.
  * ANTI-CHEAT: RANDOM per-seed surprise assignment. Which two (of 8 trained) concepts play "cue" (c) and "false
    assertion" (c') -- i.e. WHICH 24-neuron block of the 288-neuron `surprise` region will actually fire during
    training -- is drawn from a seed-keyed RNG, independently per seed (NOT concept 0 vs 1 every time). The
    de-risk requires growth to land on WHICHEVER block was randomly assigned this seed's "surprise" role, and
    the OTHER 11 (never-mismatched) concept blocks' edges into `prov_generated` to stay at the ~0.05 seed value
    -- proving the coupling tracks the SURPRISE POPULATION'S activity (a general mismatch detector), not a
    memorized/hardcoded concept identity. This is the same-class control as R1's "mismatched w0->sel_patient /
    w1->sel_agent pairs stay at 0.05" negative control, generalized to a randomized assignment.

THE FUNCTIONAL GATE (6 seeds 42,43,44,100,101,102):
  F1 FACULTY-STILL-WORKS: source_provenance's OWN 8-item battery keeps its pre-registered floor; SURPRISE's own
     CONFIRM-vs-CONTRADICT discrimination on THIS seed's randomly-assigned block pair stays clean (contradict >>
     confirm), both with the cross-edge present (no interference).
  F2 INTERACTION-IS-REAL (the crux): on a FRESH, genuinely-ambiguous content pattern (dual-context encoded, R4's
     exact protocol, reused by import), co-driving the trained CONTRADICT trial (surprise fires) during that
     item's recall shifts the signed margin (rate_generated - rate_perceived) toward GENERATED vs a no-hold
     baseline. LESIONING the cross-edge collapses the shift. attributable_to ~1.0.
  F3 NO-RUNAWAY: physiological firing band; the trained-block cross weight converges (bounded, decelerating).
  F4 MOAT/HONESTY: (a) the surprise-inducing trial held with NO content drive at all stays sub-decision (no
     confabulated provenance from bias alone); (b) a CLEAR, already-correctly-encoded battery item is NOT
     flipped by a co-occurring (wrong-context) surprise hold.
  + EMERGENCE: the trained block's weight grows from ~0.05 by the substrate's own Hebbian rule; every OTHER
     (non-participating) concept block's edges stay at ~0.05 (the anti-cheat / specificity read); zero
     corruption of any non-cross synapse.
  + LESION-RECOVERS-MIGRATION: with the cross-edge lesioned, both organs' base connectivity + own battery reads
     are byte-identical (within the FP-layout floor) to the plain (no-cross-edge) merged pool.

DE-RISK ONLY -- no production wiring, no `sim/` edit, no default flip. Additive (NO import cycle into `server.py`
or any production organ). numpy CPU throughout.

Run:
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_surprise_episodic_crossedge --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_surprise_episodic_crossedge \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only -- never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend
from tools.lab import attributable_to

# ---- geometry + protocol constants (validated on seed 42; pre-registered) ----
W0 = 0.05                        # near-zero seed weight (the edge must GROW, not be pre-wired)
GATE = "surprise_to_provgen"     # the single plastic cross-edge gate
CUE_PA = 2000.0                   # RECALIBRATED (not R1/R4's "hold constant" class -- a genuine operating-point
                                   # fix, found + fixed BEFORE the smoke, verified via a raw-rate sweep, never a
                                   # floor-tuning game): the registered SURPRISE descriptor never declares
                                   # `enable_homeostasis`, so it silently inherits source_provenance's REQUIRED
                                   # `enable_homeostasis=False` in the config union (no MergeConflict is raised --
                                   # only an EXPLICIT two-value clash raises). SURPRISE's own de-risk
                                   # (`_spiking_expectation_rpe_derisk`/`_one_brain_merge_2organ_derisk`) was
                                   # calibrated at CoreSimConfig's class default `enable_homeostasis=True` (the
                                   # threshold EMA continuously re-centers `cue`'s operating point); with it
                                   # forced OFF by this pairing, `cue`'s STATIC heterogeneous thresholds fire
                                   # ~2x less readily at 600pA (275->144 spikes/60 steps, measured), starving
                                   # `patient_expected`'s recall of drive entirely (0.00 Hz, no discrimination
                                   # possible). 2000pA restores a reliable CONFIRM~0.00Hz / CONTRADICT>0Hz
                                   # separation (swept 600-2000pA x 60-150 pre-steps on seed 42 before any F-gate
                                   # floor was touched -- 2000pA is the first value where `patient_expected`
                                   # recalls in BOTH conditions at all, matching the qualitative CONFIRM-cancels /
                                   # CONTRADICT-fires shape the base module documents).
CTX_DRIVE_PA = 2500.0            # source_provenance's OWN encoding-context drive (R4's de-risk constant)
EPISODE_DRIVE_PA = 2500.0        # source_provenance's OWN content drive (R4's de-risk constant)
TRAIN_STEPS = 60          # matches the calibrated CONTRADICT measurement window (the 600->2000pA sweep found
                          # surprise's separation at hold=60; a shorter training window under-exposes the edge
                          # to genuine co-activity, measured: 30 steps x 40 episodes grew the edge only to 0.22)
N_EPISODES = 150          # calibrated up from 40/60 (measured: 60 episodes converges the trained block to only
                          # ~1.19, giving an F2 signal (~0.004-0.005) comparable in magnitude to a small, fixed,
                          # non-cross-edge read residual this pairing's multi-step spiking read carries (~0.003 --
                          # the documented "layout-mediated coupling" class, `deterministic_transpose_matvec`'s
                          # own docstring: "a SPIKING-DYNAMICS read integrated over hundreds of steps AMPLIFIES a
                          # single-ULP per-step delta"; measured stable across N_READS=3->8, i.e. NOT sampling
                          # noise). Rather than chase an engine-level fix (out of this de-risk's scope), the
                          # R1/R4 precedent is followed: train the edge further toward its bound so the genuine
                          # signal clearly dominates the fixed-magnitude residual.
RECALL_STEPS = 100
N_READS = 8                      # averaged reads per condition (denoise) -- higher than R1/R4's 3: this pairing's
                                  # signal is smaller-magnitude (a weaker, more diluted concept-block-scoped
                                  # edge), so more averaging is needed to separate it from read noise
HMAX = 20.0                      # hebbian_max_weight -- calibrated on seed 42's smoke (F3: the soft bound the
                                  # trained block converges toward). Raised from an initial 6.0 (R4's own value)
                                  # once F2's genuine signal was found to need more headroom than R4's pairing did
                                  # (this pairing's read carries a small fixed non-cross-edge residual the edge's
                                  # effect must clearly dominate, see N_EPISODES' note) -- F4's moat check still
                                  # gates any overshoot exactly as it did for R4.
N_AMBIG_PASSES = 2                # interleaved perceived/generated encode passes for the fresh ambiguous item
PRE_STEPS = 60                    # prediction pre-phase (cue alone) before the assertion volley -- lets the
                                   # slow GABA_B subtractive prediction settle first (predictive-coding /
                                   # mismatch-negativity; `_spiking_expectation_rpe_derisk.measure_conditions`)
CROSS_EDGE_LR = 0.15              # SCALING fix (not a floor-tuning game), applied 2026-08-27 continuing the
                                   # prior seed-42 WIP session: F2's vary-then-lesion crux (delta_intact) came
                                   # in short of its pre-registered F2_INTACT_FLOOR at the original 0.05 rate.
                                   # That session's dose-response check found delta_intact scaling with the
                                   # rate at fixed N_EPISODES/HMAX -- i.e. the cross-edge hadn't grown far
                                   # enough toward its bound within the training budget, NOT a ceiling on the
                                   # mechanism -- and identified 0.15 as clearing the floor. This run verifies
                                   # that at 0.15 across the full 6-seed set (still inside F3's bounded/
                                   # decelerating convergence check). This is the ONLY change from the WIP
                                   # runner's calibration; N_EPISODES/HMAX/CUE_PA/floors are all unchanged.

# F-gate floors (pre-registered before the 6-seed run; calibrated on seed 42's smoke)
F1_SEP_RATIO = 5.0         # surprise's own CONTRADICT rate must exceed CONFIRM rate by this factor (min)
F2_INTACT_FLOOR = 0.010    # signed margin (rate_generated - rate_perceived) the surprise-hold must move, intact
F2_LESION_RATIO = 0.34     # lesion |delta| must be < this * intact |delta| (the shift is edge-caused)
F4A_FRAC = 0.5             # silence-only |margin| must be < this fraction of a genuine decision
F4B_RETAIN = 0.5           # a clear item keeps >= this fraction of its margin under a WRONG surprise hold
RATE_LO, RATE_HI = 5e-4, 0.7   # physiological firing band during the held reads
OTHER_BLOCK_DRIFT_MAX = 0.03   # the anti-cheat floor: non-participating concept blocks must stay near W0

_CONDUCT = ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
            "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise",
            "cp_conductance_g_gabab_slow", "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise")

# THE READ-ISOLATION FIX (2026-09-02, Port A -- routes to the SAME 4 arrays the framework's own
# `MergedPool._PER_NEURON_STATE` (`onebrain_merge_framework.py:246-250`) already lists, ported verbatim from the
# C2 fix `research/runners/_crossedge_surprise_metacog_derisk.py` `_EXTRA_RESET_ARRAYS`/`_rest_extra`, which
# diagnosed this exact bug class: `_hard_reset` restores membrane/recovery/conductances/firing/Hebbian-trace but
# NOT these 4, so residual state from whichever condition/episode ran immediately before a read leaks into the
# next -- ORDER-dependent, not a genuine substrate difference. Audit:
# `research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md` (FW-2: this runner's F2
# lesion control fails 5/6 seeds with the leak's magnitude the SAME ORDER as the `delta_lesion` it corrupts).
#   * cp_refractory_timers / cp_prev_firing_states -- HARD firing gates (int32 countdown / bool), independent of
#     membrane potential; a neuron mid-refractory at the end of one read/episode stays gated at the START of the
#     next even though v/u were reset.
#   * cp_neuron_activity_ema / cp_neuron_firing_thresholds -- the homeostatic per-neuron EMA + adaptive threshold
#     (`sim/bridge.py` fused_homeostasis_update); update is participation-gated (fired-or-driven neurons only),
#     so it silently drifts on whichever neurons the immediately-prior read/episode drove, never on the rest.
_EXTRA_RESET_ARRAYS = ("cp_refractory_timers", "cp_prev_firing_states",
                       "cp_neuron_activity_ema", "cp_neuron_firing_thresholds")


def _assign_blocks(seed, n_trained):
    """ANTI-CHEAT: RANDOM per-seed assignment of which two trained concepts play cue (c) / false-assertion (c').
    A seed-keyed RNG independent of every other seeded draw in this module (distinct offset)."""
    rng = np.random.default_rng(int(seed) * 104729 + 17)
    c = int(rng.integers(0, n_trained))
    cp = int(rng.integers(0, n_trained - 1))
    if cp >= c:
        cp += 1
    return c, cp


def _build_pool(seed):
    """Build the [SURPRISE_LITE, source_provenance] MergedPool via the DECLARATIVE cross_edges= framework param
    (CrossEdge + merge_organs), mirroring `_onebrain_declarative_crossedge_r1_repro.DeclarativeR1Pool` verbatim
    in STRUCTURE: one merge_organs(...) call with the cross-edge already unioned into the wire=True inject, no
    hand-typed second re-inject."""
    from research.runners.onebrain_merge_framework import REGISTRY, CrossEdge, merge_organs
    SURPRISE = REGISTRY["surprise"]
    SP = REGISTRY["source_provenance"]
    # Config reconciliation (see module docstring): SURPRISE's registered config forces
    # enable_hebbian_learning=True (needed only by worldmodel, its usual pool-1 partner) -- a direct MergeConflict
    # against source_provenance's enable_hebbian_learning=False. SURPRISE's OWN circuit is 100% fixed/block-
    # diagonal (no live Hebbian pathway of its own), so this override is behavior-preserving for SURPRISE and
    # required for SP -- verified empirically by F1's own-battery + own-discrimination checks below.
    SURPRISE_LITE = dataclasses.replace(
        SURPRISE, config={**SURPRISE.config, "enable_hebbian_learning": False, "hebbian_rate_window": False})
    CROSS_EDGES = [
        CrossEdge(key=GATE, source_key="surprise", source_region="surprise",
                 target_key="source_provenance", target_region="prov_generated",
                 init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True),
    ]
    pool = merge_organs([SURPRISE_LITE, SP], seed=seed, config_descriptors=[SURPRISE_LITE, SP],
                        wire=True, cross_edges=CROSS_EDGES)
    return pool


def _build_pool_plain(seed):
    """The no-cross-edge baseline pool (same two organs, same config, cross_edges=None) for the migration
    invariant -- structurally identical to `_build_pool` minus the declared edge."""
    from research.runners.onebrain_merge_framework import REGISTRY, merge_organs
    SURPRISE = REGISTRY["surprise"]
    SP = REGISTRY["source_provenance"]
    SURPRISE_LITE = dataclasses.replace(
        SURPRISE, config={**SURPRISE.config, "enable_hebbian_learning": False, "hebbian_rate_window": False})
    return merge_organs([SURPRISE_LITE, SP], seed=seed, config_descriptors=[SURPRISE_LITE, SP], wire=True)


class SurpriseEpisodicPool:
    """The integrated pool: the merged [surprise, source_provenance] bridge + the DECLARATIVE learned
    surprise->prov_generated cross-edge, with the direct-drive train + read protocol the F-gate consumes.
    Structure mirrors R4Pool (`_onebrain_integration_r4_selfschema_provenance.R4Pool`) exactly."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.xp, _ = get_backend()
        self.pool = _build_pool(seed)
        self.pool.ensure_built()
        self.b = self.pool.bridge
        rm = self.b.region_manager

        def idxr(nm):
            return np.asarray(rm.indices(nm), np.int64)

        self.ix = {nm: idxr(nm) for nm in ("cue", "patient_expected", "patient_asserted", "surprise",
                                           "episode", "content_readout", "ctx_perceived", "ctx_generated",
                                           "prov_perceived", "prov_generated", "inh_perceived", "inh_generated")}
        meta = self.pool.meta["surprise"]
        self.blk = int(meta["blk"]); self.n_trained = int(meta["n_trained"]); self.n_novel = int(meta["n_novel"])
        self.cue_c, self.assert_cp = _assign_blocks(seed, self.n_trained)   # THIS SEED's random block pair

        coo = self.b.cp_connections.tocoo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        surprise_idx = self.ix["surprise"]
        trained_block = surprise_idx[self.assert_cp * self.blk:(self.assert_cp + 1) * self.blk]
        post = self.ix["prov_generated"]
        full_mask = np.isin(row, surprise_idx) & np.isin(col, post)
        trained_mask = np.isin(row, trained_block) & np.isin(col, post)
        assert int(full_mask.sum()) > 0, "declarative cross-edge is EMPTY -- the CrossEdge did not wire"
        self.masks = {"surprise->provgen": full_mask,
                     "trained_block->provgen": trained_mask,
                     "other_blocks->provgen": full_mask & ~trained_mask}

        from research.runners.onebrain_merge_framework import _source_prov_organ
        self.sp_organ = _source_prov_organ(seed, self.pool)
        self.sp_organ.ensure_built()          # runs its OWN build-time Hebbian encode of the 8-item battery

        # THE DECLARATIVE WHITELIST FREEZE -- cp_plasticity_rate_gain=0 everywhere then re-open ONLY GATE.
        self.pool.apply_cross_edge_freeze()

        # the fresh AMBIGUOUS content pattern (disjoint from the 8-item battery), dual-context encoded so its
        # prov_perceived/prov_generated traces land near-tied (R4's exact protocol, reused verbatim below).
        self.ambig_pattern = self._make_ambiguous_pattern()
        self._encode_ambiguous()

        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        self._noncross &= ~self.masks["surprise->provgen"]
        # standard (non-rate-window) Hebbian hyperparameters for OUR cross-edge's training window.
        for kk, vv in dict(hebbian_symmetric=True, hebbian_learning_rate=CROSS_EDGE_LR, hebbian_max_weight=HMAX,
                           hebbian_min_weight=0.0, hebbian_weight_decay=0.0).items():
            setattr(self.b.core_config, kk, vv)

        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()
        # THE READ-ISOLATION FIX (2026-09-02 -- see module-level `_EXTRA_RESET_ARRAYS` comment): snapshot the
        # SAME true-rest baseline for every OTHER per-neuron array `_run_one_simulation_step` mutates that this
        # runner's ORIGINAL `_hard_reset` never restored (refractory timers, prev-firing-state, the homeostatic
        # activity EMA / adaptive threshold). Taken ONCE here, immediately after the true-rest settle, restored
        # on EVERY `_hard_reset` call below.
        self._rest_extra = {}
        for nm in _EXTRA_RESET_ARRAYS:
            arr = getattr(self.b, nm, None)
            self._rest_extra[nm] = np.asarray(to_host(arr)).copy() if arr is not None else None

    # ---- primitives (byte-identical shape to R4Pool) ----
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
        if getattr(b, "cp_hebb_coactivity_trace", None) is not None:
            b.cp_hebb_coactivity_trace[:] = 0.0
        # THE READ-ISOLATION FIX: restore every array in `_EXTRA_RESET_ARRAYS` to the TRUE rest snapshot taken in
        # __init__. Without this, whichever condition/episode ran immediately before a read leaks its residual
        # refractory/homeostatic state into the next one -- an ORDER-dependent bias (the diagnosed direct cause
        # of F2's own lesion control failing 5/6 seeds: `delta_lesion` was picking up this leak, not a genuine
        # cross-edge-independent residual), not a block-identity effect.
        for nm in _EXTRA_RESET_ARRAYS:
            val = self._rest_extra.get(nm)
            if val is not None:
                getattr(b, nm)[:] = xp.asarray(val)
        b.cp_external_input_current[:] = 0.0

    def _drive(self, pairs, steps, learn=False, read=None, pre_pairs=None, pre_steps=0):
        """Optionally run a PREDICTION pre-phase (pre_pairs, pre_steps -- e.g. cue alone, so the slow GABA_B
        subtractive prediction is already settled before the assertion volley arrives; predictive-coding /
        mismatch-negativity, the validated `_spiking_expectation_rpe_derisk.measure_conditions` protocol), then
        the measured phase (pairs, steps). Without the pre-phase CONFIRM's inhibition never catches up within a
        single window and surprise stops discriminating mismatch (caught empirically on the seed-42 smoke:
        F1 separation collapsed to exactly 1.0x)."""
        b, xp = self.b, self.xp
        b.core_config.enable_hebbian_learning = False
        if pre_pairs is not None and pre_steps > 0:
            precur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
            for idx, pa in pre_pairs:
                precur[xp.asarray(idx)] = xp.float32(pa)
            for _ in range(pre_steps):
                b.cp_external_input_current[:] = precur
                b._run_one_simulation_step()
        b.core_config.enable_hebbian_learning = bool(learn)
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        acc = {k: 0.0 for k in (read or {})}
        for _ in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            if read:
                fs = b.cp_firing_states
                for k, idx in read.items():
                    acc[k] += float(to_host(fs[xp.asarray(idx)].astype(xp.float64).sum())) / idx.size
        b.cp_external_input_current[:] = 0.0
        b.core_config.enable_hebbian_learning = False
        return {k: v / steps for k, v in acc.items()}

    def _wmean(self, name="trained_block->provgen"):
        data = np.asarray(to_host(self.b.cp_connections.data))
        m = self.masks[name]
        return float(data[m].mean()) if m.any() else float("nan")

    def cross_weights(self):
        return {k: round(self._wmean(k), 4) for k in self.masks}

    # ---- the surprise-inducing CONTRADICT trial for THIS seed's random block pair ----
    def _cue_idx(self, concept):
        return self.ix["cue"][concept * self.blk:(concept + 1) * self.blk]

    def _cue_pre_pairs(self):
        """The prediction pre-phase drive: cue alone (this seed's cue_c), so patient_expected's recall + the
        slow GABA_B subtractive prediction are already settled BEFORE the assertion volley (mismatch-negativity
        protocol) -- required for surprise to discriminate CONFIRM vs CONTRADICT at all."""
        return [(self._cue_idx(self.cue_c), CUE_PA)]

    def _contradict_pairs(self):
        ix = self.ix
        assert_idx = ix["patient_asserted"][self.assert_cp * self.blk:(self.assert_cp + 1) * self.blk]
        return [(self._cue_idx(self.cue_c), CUE_PA), (assert_idx, CUE_PA)]

    def _confirm_pairs(self):
        ix = self.ix
        assert_idx = ix["patient_asserted"][self.cue_c * self.blk:(self.cue_c + 1) * self.blk]
        return [(self._cue_idx(self.cue_c), CUE_PA), (assert_idx, CUE_PA)]

    # ---- the fresh, genuinely-ambiguous content pattern (dual-context encoded; R4's exact protocol) ----
    def _make_ambiguous_pattern(self):
        from research.runners._laneC_source_provenance_opponent_derisk import (
            make_paired_patterns, EP_PATTERN, N_EPISODE)
        pats = make_paired_patterns(self.seed)
        used = set()
        for prov in ("perceived", "generated"):
            for arr in pats[prov]:
                used.update(int(x) for x in arr.tolist())
        rng = np.random.default_rng(int(self.seed) * 997 + 3)
        free = [j for j in range(N_EPISODE) if j not in used]
        return np.sort(rng.choice(free, size=EP_PATTERN, replace=False)).astype(np.int64)

    def _encode_ambiguous(self):
        """Balanced dual-context encode of the fresh ambiguous pattern (R4's exact protocol, reused verbatim:
        see `_onebrain_integration_r4_selfschema_provenance.R4Pool._encode_ambiguous` for the full rationale,
        including the hebbian_max_weight save/set/restore that avoids clipping the already-trained battery)."""
        from research.runners._laneC_source_provenance_opponent_derisk import HEBB_LR, HEBB_WMAX
        b = self.b
        cc = b.core_config
        saved_gain = np.asarray(to_host(b.cp_plasticity_rate_gain)).copy()
        saved = {k: getattr(cc, k) for k in (
            "enable_hebbian_learning", "hebbian_learning_rate", "hebbian_max_weight",
            "hebbian_min_weight", "hebbian_weight_decay", "hebbian_symmetric")}
        b.cp_plasticity_rate_gain[:] = 0.0
        b.set_plasticity_gate("prov_learn", 1.0)
        b.set_plasticity_gate("content_learn", 1.0)
        cc.enable_hebbian_learning = True
        cc.hebbian_learning_rate = float(HEBB_LR)
        cc.hebbian_max_weight = float(HEBB_WMAX)
        cc.hebbian_min_weight = 0.0
        cc.hebbian_weight_decay = 0.0
        cc.hebbian_symmetric = True
        try:
            for _ in range(N_AMBIG_PASSES):
                self.sp_organ.brain.encode(self.ambig_pattern, "perceived", learning=True)
                self.sp_organ.brain.encode(self.ambig_pattern, "generated", learning=True)
        finally:
            b.set_plasticity_gate("prov_learn", 0.0)
            b.set_plasticity_gate("content_learn", 0.0)
            for k, v in saved.items():
                setattr(cc, k, v)
            b.cp_plasticity_rate_gain[:] = self.xp.asarray(saved_gain)

    # ---- emergence: grow the cross-edge from experience ----
    def train(self, n_episodes=N_EPISODES):
        traj = [dict(ep=0, w=self._wmean("trained_block->provgen"), w_other=self._wmean("other_blocks->provgen"))]
        for ep in range(n_episodes):
            self._hard_reset()
            drives = self._contradict_pairs() + [(self.ix["ctx_generated"], CTX_DRIVE_PA)]
            self._drive(drives, TRAIN_STEPS, learn=True,
                       pre_pairs=self._cue_pre_pairs(), pre_steps=PRE_STEPS)
            if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                traj.append(dict(ep=ep + 1, w=self._wmean("trained_block->provgen"),
                                 w_other=self._wmean("other_blocks->provgen")))
        self.b.core_config.enable_hebbian_learning = False
        now = np.asarray(to_host(self.b.cp_connections.data))
        self.frozen_maxdrift = float(np.max(np.abs(now[self._noncross] - self._frozen_w0[self._noncross])))
        return traj

    # ---- the signed ambiguous-item read with an optional surprise (CONTRADICT) hold ----
    def amb_read(self, hold_surprise, band=None):
        ix = self.ix
        ep_idx = ix["episode"][self.ambig_pattern]
        margins, rates = [], {"prov_generated": 0.0, "prov_perceived": 0.0, "surprise": 0.0, "ctx_generated": 0.0}
        for _ in range(N_READS):
            self._hard_reset()
            pairs = [(ep_idx, EPISODE_DRIVE_PA)]
            pre, pre_steps = None, 0
            if hold_surprise:
                pairs = pairs + self._contradict_pairs()
                pre, pre_steps = self._cue_pre_pairs(), PRE_STEPS
            read = {"gen": ix["prov_generated"], "perc": ix["prov_perceived"]}
            if band is not None:
                for r in rates:
                    read[r] = ix[r]
            acc = self._drive(pairs, RECALL_STEPS, read=read, pre_pairs=pre, pre_steps=pre_steps)
            margins.append(acc["gen"] - acc["perc"])
            if band is not None:
                for r in rates:
                    rates[r] += acc.get(r, 0.0)
        out = {"margin": float(np.mean(margins))}
        if band is not None:
            out["rates"] = {r: rates[r] / N_READS for r in rates}
        return out


def _selftest_read_isolation(seed=42):
    """FAILS-IN-THE-FAILING-DIRECTION guard for the 2026-09-02 read-isolation fix (Port A: `_EXTRA_RESET_ARRAYS`
    restored in `_hard_reset`). On a ZEROED-MECHANISM pool -- untrained (no `.train()` call) AND the cross-edge
    lesioned (zeroed) -- NOTHING should differ between two back-to-back identical reads of the same fresh
    ambiguous item, so two consecutive `amb_read(hold_surprise=False)` calls must be BITWISE identical.

    Verified against the PRE-fix code during this port (temporarily reverting `_EXTRA_RESET_ARRAYS`'s restore in
    `_hard_reset`): this assertion FAILS then -- the first call's internal N_READS=8 hard-resets leak residual
    refractory/homeostatic state into the second call's reads, exactly the order-dependent bias the C2 finding's
    own instrumentation table documents (`research/findings/2026-09-02-c2-metacog-read-isolation-fix-GO.md`).
    With the fix, both calls start from the identical restored true-rest snapshot every time -> byte-identical."""
    sep = SurpriseEpisodicPool(seed)
    data = np.asarray(to_host(sep.b.cp_connections.data)).copy()
    data[sep.masks["surprise->provgen"]] = 0.0     # lesion: the cross-edge is the only thing that could differ
    sep.b.cp_connections.data = sep.xp.asarray(data, dtype=sep.b.cp_connections.data.dtype)
    r1 = sep.amb_read(False)
    r2 = sep.amb_read(False)
    ok = bool(r1["margin"] == r2["margin"])
    return {"seed": int(seed), "margin_1": r1["margin"], "margin_2": r2["margin"],
            "diff": abs(r1["margin"] - r2["margin"]), "PASS": ok}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The four functional-gate arms + the migration invariant + the emergence read
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _f1(sep):
    """F1 FACULTY-STILL-WORKS: source_provenance's OWN 8-item battery keeps its floor; SURPRISE's own
    CONFIRM-vs-CONTRADICT discrimination on this seed's random block pair stays clean. Edge present."""
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS, D_FLOOR
    sep._hard_reset()
    ds, accs = [], []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            pat = sep.sp_organ.patterns[prov][i]
            rec = sep.sp_organ.brain.recall(pat)
            rp, rg = rec["rate_perceived"], rec["rate_generated"]
            margin = rp - rg
            winner = "perceived" if margin >= 0 else "generated"
            d_perc = margin / (rp + rg + 1e-9)
            d_true = d_perc if prov == "perceived" else -d_perc
            ds.append(d_true); accs.append(winner == prov)
    min_d = float(np.min(ds)); acc = float(np.mean(accs))
    battery_ok = bool(acc >= 0.999 and min_d >= D_FLOOR)

    sep._hard_reset()
    confirm = sep._drive(sep._confirm_pairs(), RECALL_STEPS, read={"surprise": sep.ix["surprise"]},
                         pre_pairs=sep._cue_pre_pairs(), pre_steps=PRE_STEPS)["surprise"]
    sep._hard_reset()
    contradict = sep._drive(sep._contradict_pairs(), RECALL_STEPS, read={"surprise": sep.ix["surprise"]},
                            pre_pairs=sep._cue_pre_pairs(), pre_steps=PRE_STEPS)["surprise"]
    disc_ok = bool(contradict > F1_SEP_RATIO * max(confirm, 1e-6))
    return {"battery_acc": acc, "battery_min_d": min_d, "battery_ok": battery_ok,
            "surprise_confirm_rate": float(confirm), "surprise_contradict_rate": float(contradict),
            "surprise_sep_ratio": float(contradict / max(confirm, 1e-6)), "discrimination_ok": disc_ok,
            "cue_concept": sep.cue_c, "assert_concept": sep.assert_cp,
            "PASS": bool(battery_ok and disc_ok)}


def _f2(sep):
    """F2 INTERACTION-IS-REAL: on the fresh AMBIGUOUS item, holding the trained CONTRADICT trial (surprise fires)
    shifts the signed margin toward GENERATED vs no-hold baseline, intact; the shift vanishes when the
    cross-edge is lesioned."""
    base_i = sep.amb_read(False, band=True)
    held_i = sep.amb_read(True)
    d_i = held_i["margin"] - base_i["margin"]
    data = np.asarray(to_host(sep.b.cp_connections.data)).copy()
    data[sep.masks["surprise->provgen"]] = 0.0
    sep.b.cp_connections.data = sep.xp.asarray(data, dtype=sep.b.cp_connections.data.dtype)
    base_l = sep.amb_read(False)
    held_l = sep.amb_read(True)
    d_l = held_l["margin"] - base_l["margin"]
    shift_ok = (d_i > F2_INTACT_FLOOR) and (abs(d_l) < F2_LESION_RATIO * abs(d_i))
    frac = attributable_to("F2 surprise-hold shift toward GENERATED = the cross-edge", d_i, d_l)
    return {"frac_attributable": (None if frac is None else float(frac)),
            "margin_base_intact": base_i["margin"], "margin_held_intact": held_i["margin"],
            "margin_base_lesion": base_l["margin"], "margin_held_lesion": held_l["margin"],
            "delta_intact": float(d_i), "delta_lesion": float(d_l),
            "rates_base_intact": base_i.get("rates", {}),
            "shift_toward_generated": bool(shift_ok), "PASS": bool(shift_ok)}


def _f3(sep, traj, f2):
    """F3 NO-RUNAWAY: firing band during the base-intact read, trained-block weight converges."""
    rates = f2.get("rates_base_intact", {})
    band_pools = ("prov_generated", "prov_perceived")
    in_band = all(RATE_LO < rates.get(p, 0.0) < RATE_HI for p in band_pools) if rates else False
    grown = traj[-1]["w"]
    bounded = grown <= HMAX
    first_dw = traj[1]["w"] - traj[0]["w"] if len(traj) >= 2 else 0.0
    last_dw = traj[-1]["w"] - traj[-2]["w"] if len(traj) >= 2 else 0.0
    decelerating = last_dw < first_dw
    alive = rates.get("prov_generated", 0.0) > RATE_LO or rates.get("prov_perceived", 0.0) > RATE_LO
    return {"rates": rates, "in_band": bool(in_band), "grown": float(grown), "bounded_by_hmax": bool(bounded),
            "first_window_dw": float(first_dw), "last_window_dw": float(last_dw),
            "decelerating": bool(decelerating), "pool_alive": bool(alive),
            "PASS": bool(in_band and bounded and decelerating and alive)}


def _f4(sep):
    """F4 MOAT/HONESTY. (a) surprise-hold with NO content drive stays SUB-DECISION; (b) a CLEAR battery item is
    NOT flipped by a co-occurring (wrong-context) surprise hold. Run on the INTACT edge, BEFORE F2's lesion."""
    ix = sep.ix
    sep._hard_reset()
    silent = sep._drive(sep._contradict_pairs(), RECALL_STEPS,
                        read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]},
                        pre_pairs=sep._cue_pre_pairs(), pre_steps=PRE_STEPS)
    silence_margin = silent["gen"] - silent["perc"]
    amb_base = sep.amb_read(False)["margin"]
    clear_pat = sep.sp_organ.patterns["perceived"][0]
    ep_idx = ix["episode"][np.asarray(clear_pat, np.int64)]
    sep._hard_reset()
    clear_nohold = sep._drive([(ep_idx, EPISODE_DRIVE_PA)], RECALL_STEPS,
                              read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]})
    m_nohold = clear_nohold["gen"] - clear_nohold["perc"]
    decision = max(abs(amb_base), abs(m_nohold), 1e-9)
    f4a_ok = bool(abs(silence_margin) < F4A_FRAC * decision)
    sep._hard_reset()
    clear_held = sep._drive([(ep_idx, EPISODE_DRIVE_PA)] + sep._contradict_pairs(), RECALL_STEPS,
                            read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]},
                            pre_pairs=sep._cue_pre_pairs(), pre_steps=PRE_STEPS)
    m_wrong = clear_held["gen"] - clear_held["perc"]
    same_sign = (m_wrong < 0) == (m_nohold < 0)
    retained = abs(m_wrong) >= F4B_RETAIN * abs(m_nohold)
    f4b_ok = bool(m_nohold < 0 and same_sign and retained)
    return {"silence_margin": float(silence_margin), "decision_scale": float(decision),
            "silence_frac_of_decision": float(abs(silence_margin) / decision),
            "f4a_no_winner_from_silence": f4a_ok,
            "clear_nohold": float(m_nohold), "clear_wrong_hold": float(m_wrong),
            "f4b_clear_not_flipped": f4b_ok, "PASS": bool(f4a_ok and f4b_ok)}


def _emergence(traj, frozen_maxdrift):
    g = traj[-1]["w"]; g_other = traj[-1]["w_other"]
    grew = g > 5 * W0
    specific = bool(abs(g_other - W0) < OTHER_BLOCK_DRIFT_MAX)   # anti-cheat: non-participating blocks stay ~W0
    return {"trajectory": traj, "final_weight_trained_block": float(g), "final_weight_other_blocks": float(g_other),
            "grew_from_near_zero": bool(grew), "other_blocks_stayed_near_seed": specific,
            "frozen_weight_maxdrift": float(frozen_maxdrift), "no_corruption": bool(frozen_maxdrift < 1e-6),
            "PASS": bool(grew and specific and frozen_maxdrift < 1e-6)}


def _migration_invariant(seed, sep, sp_battery_lesioned, surprise_reads_lesioned):
    """LESION-RECOVERS-MIGRATION: with the cross-edge lesioned, (1) base connectivity is BYTE-IDENTICAL to the
    plain no-cross-edge merged pool, and (2) both organs' own reads match within the FP-layout floor."""
    from research.runners.onebrain_merge_framework import _source_prov_organ
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS
    pool0 = _build_pool_plain(seed)
    pool0.ensure_built()
    sp0 = _source_prov_organ(seed, pool0); sp0.ensure_built()

    def edge_map(pool):
        coo = pool.bridge.cp_connections.tocoo()
        r = to_host(coo.row); c = to_host(coo.col); d = to_host(coo.data)
        return {(int(a), int(b)): float(w) for a, b, w in zip(r, c, d)}
    k0 = edge_map(pool0)
    k1 = edge_map(sep.pool)
    xmask = sep.masks["surprise->provgen"]
    coo1 = sep.b.cp_connections.tocoo()
    r1 = to_host(coo1.row); c1 = to_host(coo1.col)
    xpairs = set(zip((int(x) for x in r1[xmask]), (int(x) for x in c1[xmask])))
    k1_base = {kk: vv for kk, vv in k1.items() if kk not in xpairs}
    struct_identical = bool(set(k1_base.keys()) == set(k0.keys()))
    base = []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            rec = sp0.brain.recall(sp0.patterns[prov][i])
            base.append(rec["rate_perceived"] - rec["rate_generated"])
    maxerr = float(np.max(np.abs(np.asarray(base) - np.asarray(sp_battery_lesioned)))) if base else 0.0

    rm0 = pool0.bridge.region_manager
    idx0 = {nm: np.asarray(rm0.indices(nm), np.int64) for nm in ("cue", "patient_asserted", "surprise")}
    blk = sep.blk
    b0 = pool0.bridge
    b0.cp_external_input_current[:] = 0.0
    for _ in range(40):
        b0._run_one_simulation_step()
    rest_v0 = np.asarray(to_host(b0.cp_membrane_potential_v)).copy()
    rest_u0 = np.asarray(to_host(b0.cp_recovery_variable_u)).copy()
    # THE READ-ISOLATION FIX, ported to this SECOND bespoke reset too (this closure inlines the same pre-fix
    # `_hard_reset` shape on a separate bridge `b0`; left unfixed it would compare a leak-free `sep`-side lesioned
    # read against a still-leaky `b0`-side read here, an avoidable inconsistency -- same _EXTRA_RESET_ARRAYS).
    rest_extra0 = {}
    for _nm in _EXTRA_RESET_ARRAYS:
        _arr = getattr(b0, _nm, None)
        rest_extra0[_nm] = np.asarray(to_host(_arr)).copy() if _arr is not None else None

    def read_surprise0(cue_c, assert_c, xp):
        b0.cp_membrane_potential_v[:] = xp.asarray(rest_v0)
        b0.cp_recovery_variable_u[:] = xp.asarray(rest_u0)
        for nm in _CONDUCT:
            a = getattr(b0, nm, None)
            if a is not None:
                a[:] = 0
        if getattr(b0, "cp_firing_states", None) is not None:
            b0.cp_firing_states[:] = False
        for _nm in _EXTRA_RESET_ARRAYS:
            _val = rest_extra0.get(_nm)
            if _val is not None:
                getattr(b0, _nm)[:] = xp.asarray(_val)
        b0.cp_external_input_current[:] = 0.0
        precur = xp.zeros(b0.core_config.num_neurons, dtype=xp.float32)
        precur[xp.asarray(idx0["cue"][cue_c * blk:(cue_c + 1) * blk])] = xp.float32(CUE_PA)
        for _ in range(PRE_STEPS):
            b0.cp_external_input_current[:] = precur
            b0._run_one_simulation_step()
        cur = xp.zeros(b0.core_config.num_neurons, dtype=xp.float32)
        cur[xp.asarray(idx0["cue"][cue_c * blk:(cue_c + 1) * blk])] = xp.float32(CUE_PA)
        cur[xp.asarray(idx0["patient_asserted"][assert_c * blk:(assert_c + 1) * blk])] = xp.float32(CUE_PA)
        acc = 0.0
        for _ in range(RECALL_STEPS):
            b0.cp_external_input_current[:] = cur
            b0._run_one_simulation_step()
            acc += float(to_host(b0.cp_firing_states[xp.asarray(idx0["surprise"])].astype(xp.float64).sum())) \
                / idx0["surprise"].size
        b0.cp_external_input_current[:] = 0.0
        return acc / RECALL_STEPS
    confirm0 = read_surprise0(sep.cue_c, sep.cue_c, sep.xp)
    contradict0 = read_surprise0(sep.cue_c, sep.assert_cp, sep.xp)
    surp_err = float(max(abs(confirm0 - surprise_reads_lesioned[0]), abs(contradict0 - surprise_reads_lesioned[1])))
    return {"base_connectivity_structurally_identical": struct_identical,
            "sp_battery_maxerr": maxerr, "surprise_reads_maxerr": surp_err,
            "PASS": bool(struct_identical and maxerr < 0.05 and surp_err < 0.05)}


def run_seed(seed):
    t0 = time.time()
    sep = SurpriseEpisodicPool(seed)
    traj = sep.train()
    emg = _emergence(traj, sep.frozen_maxdrift)
    f1 = _f1(sep)
    f4 = _f4(sep)                                   # F4 BEFORE F2 (F2 lesions the edge in place)
    f2 = _f2(sep)                                   # F2 lesions the cross-edge at its end
    sep._hard_reset()
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS
    sp_les = []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            rec = sep.sp_organ.brain.recall(sep.sp_organ.patterns[prov][i])
            sp_les.append(rec["rate_perceived"] - rec["rate_generated"])
    sep._hard_reset()
    confirm_les = sep._drive(sep._confirm_pairs(), RECALL_STEPS, read={"surprise": sep.ix["surprise"]},
                             pre_pairs=sep._cue_pre_pairs(), pre_steps=PRE_STEPS)["surprise"]
    sep._hard_reset()
    contradict_les = sep._drive(sep._contradict_pairs(), RECALL_STEPS, read={"surprise": sep.ix["surprise"]},
                                pre_pairs=sep._cue_pre_pairs(), pre_steps=PRE_STEPS)["surprise"]
    f3 = _f3(sep, traj, f2)
    mig = _migration_invariant(seed, sep, sp_les, (confirm_les, contradict_les))
    go = bool(f1["PASS"] and f2["PASS"] and f3["PASS"] and f4["PASS"] and emg["PASS"] and mig["PASS"])
    return {"seed": int(seed), "PASS": go, "elapsed_s": round(time.time() - t0, 1),
            "cue_concept": sep.cue_c, "assert_concept": sep.assert_cp,
            "emergence": emg, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "lesion_recovers_migration": mig}


def _agg(runs):
    def frac(key):
        return sum(1 for r in runs if r[key.split(".")[0]][key.split(".")[1]]) if "." in key else 0
    keys = ["F1.PASS", "F2.PASS", "F3.PASS", "F4.PASS", "emergence.PASS", "lesion_recovers_migration.PASS"]
    return {k: f"{frac(k)}/{len(runs)}" for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--selftest", action="store_true",
                    help="read-isolation fails-in-failing-direction guard only (no F-gate run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        r = _selftest_read_isolation()
        print(f"[selftest read_isolation] seed={r['seed']} margin_1={r['margin_1']!r} margin_2={r['margin_2']!r} "
              f"diff={r['diff']!r} {'PASS' if r['PASS'] else 'FAIL'}", flush=True)
        assert r["PASS"], (
            "READ-ISOLATION REGRESSION: two identical consecutive reads on a zeroed-mechanism pool are not "
            "bitwise identical -- _EXTRA_RESET_ARRAYS is missing or not restored in _hard_reset")
        return 0

    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        f2 = r["F2"]
        print(f"[seed {s}] {'GO' if r['PASS'] else 'no'} ({r['elapsed_s']}s) block(c={r['cue_concept']},"
              f"c'={r['assert_concept']}) | "
              f"emerge w={r['emergence']['final_weight_trained_block']:.2f} "
              f"w_other={r['emergence']['final_weight_other_blocks']:.3f} "
              f"grew={r['emergence']['grew_from_near_zero']} specific={r['emergence']['other_blocks_stayed_near_seed']} | "
              f"F1 battery(acc={r['F1']['battery_acc']:.3f},min_d={r['F1']['battery_min_d']:.3f})="
              f"{r['F1']['battery_ok']} disc(sep={r['F1']['surprise_sep_ratio']:.1f}x)={r['F1']['discrimination_ok']} "
              f"F1={r['F1']['PASS']} | "
              f"F2 delta={f2['delta_intact']:+.4f}(les {f2['delta_lesion']:+.4f}) frac={f2['frac_attributable']} "
              f"={f2['PASS']} | F3={r['F3']['PASS']} F4={r['F4']['PASS']} mig={r['lesion_recovers_migration']['PASS']}",
              flush=True)

    n_go = sum(r["PASS"] for r in runs)
    agg = _agg(runs)
    all_go_raw = (n_go == len(runs)) and not args.smoke

    # THE VERDICT MUST BE EARNED, NOT MERELY TALLIED (tools/verdict.py; tools/gates/verdict_preconditions.py).
    # Compute the preconditions BEFORE composing the human-readable tag/verdict string -- the original WIP
    # computed `all_go`/`tag`/`verdict` from the raw F1-F4 pass counts FIRST and only appended `preconditions`
    # afterward, so a failing precondition (here: f2_lesion_removes_shift, the crux's own internal validity
    # check) never reached the tag at all. That is the EXACT "affect eviction" bug this module's own docstring
    # names ("the runner COMPUTED arm_valid=False on 3/3 seeds and printed NO-GO anyway") -- found and fixed
    # here, not re-derived: `Vd.decide()` already returns UNDEFINED whenever any precondition fails; this
    # runner just was not reading that field.
    dec, preconditions = None, []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("onebrain_integration_surprise_episodic_crossedge")
        Vd.require("f2_lesion_removes_shift", 1 if all(
            abs(r["F2"]["delta_lesion"]) < F2_LESION_RATIO * max(abs(r["F2"]["delta_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the F2 shift must VANISH under lesion or it is a confound, not the cross-edge (the crux control)")
        Vd.require("migration_byte_identity", 1 if all(r["lesion_recovers_migration"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="lesion the cross-edge -> both organs' own reads == the plain merged pool")
        Vd.require("emergence_grew_from_near_zero", 1 if all(r["emergence"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="the trained block grows from ~0.05 by Hebbian co-activity, not hand-set; other blocks stay put")
        Vd.require("moat_no_winner_from_silence", 1 if all(r["F4"]["f4a_no_winner_from_silence"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="surprise-held + no content drive stays sub-decision (F4 moat)")
        Vd.require("anti_cheat_random_assignment", 1 if len(set((r["cue_concept"], r["assert_concept"])
                   for r in runs)) > 1 else 0, expect=lambda x: x >= 1,
                   note="the per-seed block pair must actually vary (not the same hardcoded pair every seed)")
        dec = Vd.decide(all_go_raw, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    verdict_status = dec.get("status") if dec else None       # GO | NO-GO | UNDEFINED | None (helper unavailable)
    all_go = all_go_raw if dec is None else bool(dec.get("go"))
    if verdict_status == "UNDEFINED":
        tag = "UNDEFINED"
    elif args.smoke:
        tag = "SMOKE-GO (1-seed indicator)" if n_go == len(runs) else "NO-GO/PARTIAL"
    else:
        tag = "GO" if all_go_raw else "NO-GO/PARTIAL"
    verdict = (f"{tag} -- surprise (D2 prediction-error) -> source_provenance encoding-gate "
               f"(the audit-sanctioned 'episodic/provenance' half of the surprise->episodic rung, substituting "
               f"for the still-Group-C-deferred d5_episodic): {n_go}/{len(runs)} seeds pass ALL of F1(faculty-"
               f"still-works) + F2(vary-then-lesion) + F3(no-runaway) + F4(moat) + emergence(LEARNED, near-zero "
               f"start, anti-cheat-specific) + lesion-recovers-migration. Per-arm: {agg}. The cross-edge GROWS "
               f"from near-zero (0.05) by the substrate's OWN standard Hebbian rule on WHICHEVER concept-block "
               f"was RANDOMLY assigned this seed's surprise-inducing role (anti-cheat), while the 11 other, "
               f"never-mismatched concept blocks' edges stay at the seed value. Co-driving a genuine mismatch "
               f"(CONTRADICT) trial on the surprise circuit during a fresh, ambiguous provenance item's recall "
               f"shifts the signed margin toward GENERATED, and the shift VANISHES on lesion (load-bearing). "
               f"The moat holds (no decision from silence; a clear item is not flipped by a wrong hold). "
               f"numpy CPU; NO sim/ edit; declarative CrossEdge on merge_organs (no bespoke re-inject)."
               + (f" UNDEFINED, NOT a validated negative: {len(dec.get('undefined_reasons', []))} precondition(s) "
                  f"unmet -- {'; '.join(dec.get('undefined_reasons', []))}. The raw pass-tally alone would have "
                  f"read {n_go}/{len(runs)} ({'GO' if all_go_raw else 'NO-GO/PARTIAL'}), but the F2 lesion control "
                  f"itself does not cleanly hold on every seed, so that tally is not evidence either way."
                  if verdict_status == "UNDEFINED" else ""))

    payload = {"probe": "onebrain_integration_surprise_episodic_crossedge", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(runs), "per_arm": agg, "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "config": {"W0": W0, "cross_edge_hebbian_lr": CROSS_EDGE_LR, "hebbian_max_weight": HMAX,
                          "n_episodes": N_EPISODES,
                          "cue_pa": CUE_PA, "ctx_drive_pa": CTX_DRIVE_PA, "episode_drive_pa": EPISODE_DRIVE_PA,
                          "recall_steps": RECALL_STEPS, "n_reads": N_READS,
                          "f2_intact_floor": F2_INTACT_FLOOR, "f2_lesion_ratio": F2_LESION_RATIO,
                          "f4a_frac": F4A_FRAC, "f4b_retain": F4B_RETAIN, "f1_sep_ratio": F1_SEP_RATIO,
                          "other_block_drift_max": OTHER_BLOCK_DRIFT_MAX},
               "mechanism": ("ONE shared merge pool [surprise (D2 expectation-violation circuit) + "
                             "source_provenance] via the DECLARATIVE CrossEdge/merge_organs(cross_edges=) "
                             "framework; a SINGLE plastic cross-edge surprise->prov_generated seeded ~0.05, the "
                             "SOLE plastic synapse (apply_cross_edge_freeze whitelist inversion), GROWN by the "
                             "substrate's standard Hebbian rule over episodes co-driving a genuine CONTRADICT "
                             "(mismatch) trial on a RANDOMLY per-seed assigned concept-block pair with source_"
                             "provenance's (fixed, non-plastic) ctx_generated line. A fresh content pattern is "
                             "dual-context encoded to create a genuinely AMBIGUOUS provenance item for F2."),
               "scope_substitution": ("The literal ask (surprise -> the D5 episodic organ) is blocked by "
                                      "d5_episodic remaining GROUP_A_DEFERRED (Group-C heavy CA3+apical-dAP+BTSP "
                                      "own-pool seam; a single BTSP store measured ~510s on numpy@2000 neurons, "
                                      "2026-08-12 finding) -- the completeness audit's own roadmap gates the full "
                                      "pairing on migrating d5_episodic first (step 6), a separate heavy lane, "
                                      "not a tiny de-risk. source_provenance's prov_generated encoding-gate is "
                                      "the audit's OWN named alternative target for this exact rung ('surprise->"
                                      "episodic/PROVENANCE ENCODING gate') and is already GROUP_A-migrated + "
                                      "already validated as a cross-edge TARGET by R4 -- this run builds THAT "
                                      "half now, honestly declared, mirroring R4's own d5_episodic->self_schema "
                                      "substitution one arc earlier."),
               "scaffold_residuals": ["prov_generated firing is an ENCODING-COMMITMENT PROXY, not a literal CA3 "
                                      "autobiographical memory trace -- the full d5_episodic pairing rides the "
                                      "Group-C migration (step 6), a named follow-on",
                                      "host-chosen cross-edge TOPOLOGY (surprise -> prov_generated only)",
                                      "host-curated training schedule (co-driving a CONTRADICT trial + "
                                      "ctx_generated directly, not via an organic dialogue turn) -- same class of "
                                      "scaffold-residual as R1/R4's host-curated schedules",
                                      "two-factor Hebbian (no reward/dopamine gating here)",
                                      "the ambiguous item is a balanced-dual-context construction, a substrate "
                                      "stand-in for a genuinely uncertain real memory"],
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[SURP->EPISODIC] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
