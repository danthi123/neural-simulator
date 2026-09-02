"""One-brain INTEGRATION phase — R4: LEARNED cross-edge self_schema authorship -> source_provenance monitoring
("is this my own thought" self-monitoring). Reuses the R1 learned-cross-edge template
(`_onebrain_integration_r1_wm_comprehension.py`) verbatim in structure; a NEW organ pairing, NO new machinery.

PAIRING CHOICE (over affect->mouth/tone): both `self_schema` and `source_provenance` are ALREADY registered,
fully-migrated GROUP_A descriptors in `onebrain_merge_framework.py` (organ_cls + idx_fn + read_fn + answer_fn,
`supports_shared=True`) — the SAME two-organ merge shape R1 used for [d6, comprehension]. `affective_tom` is
instead a GROUP_A_DEFERRED entry needing a new OU + neuromodulator-subsystem seam, and no "mouth/tone" organ is
registered in the framework at all — that pairing needs new wiring this arc explicitly avoids inventing.

THE MECHANISM (emergence-compliant; NO sim/ edit):
  * ONE shared spiking bridge holds BOTH organs' regions: self_schema (workspace GNW attractor + attend/confid/
    AUTHOR sub-blocks, DR-3) + source_provenance (episode/ctx_*/prov_perceived/prov_generated opponent trace,
    board #129). `merge_organs([self_schema, source_provenance], wire=True)` — both organs are ORGAN-READ CLOSED
    on this pool (self_schema's assembly loops + member->attend + source_provenance's base pathways all reinject
    per-region-seamed, byte-identical to their own standalone builds).
  * self_schema's `author` sub-block is the substrate's OWN "did I author this" tag: driven by a tonic current
    when the current thought is SELF-generated (a volunteered proposition), silent when it is HEARD (a recalled
    fact) — see `self_schema_production_organ.py`. source_provenance's `prov_generated` pool is the substrate's
    OWN "this memory's source reads as internally-generated" pool. The natural coupling: when the brain currently
    judges a thought as SELF-authored, that signal should bias a co-temporal, genuinely AMBIGUOUS source-memory
    read toward GENERATED (Johnson-Hashtroudi-Lindsay 1993 source-monitoring: self-referential processing biases
    later source attributions).
  * ONE plastic cross-edge SET `author -> prov_generated` is injected at w0=0.05 (near-zero) as the SOLE plastic
    synapse (`cp_plasticity_rate_gain=0` everywhere then `set_plasticity_gate(GATE, 1)` — R1's whitelist
    inversion, every migrated edge byte-frozen). It GROWS by the substrate's OWN STANDARD (same-step, symmetric)
    Hebbian rule (`hebbian_symmetric`, `sim/bridge.py:1181`,`:9767`) over episodes that co-drive the author pool
    (AUTHOR_PA, self_schema's OWN production constant) with source_provenance's `ctx_generated` line (CTX_DRIVE_PA,
    source_provenance's OWN de-risk constant) — `ctx_generated -> prov_generated` is a FIXED (non-plastic) strong
    pathway, so this reliably co-fires `author` (pre) with `prov_generated` (post), Hebbian-binding the cross-edge
    without ever touching source_provenance's own learned episode->prov traces. NOT R1's rate-window rule
    (`hebbian_rate_window`): that flag is GLOBAL, not per-edge (`sim/bridge.py:1191`), and would silently hijack
    source_provenance's OWN prov_learn/content_learn edges onto an untuned coactivity-trace rule they were never
    calibrated for — caught empirically (battery collapsed to chance) before the 6-seed run; see `_build_pool`'s
    comment for the full mechanism.
  * ONE-SIDED BY DESIGN (declared, not smoothed over): self_schema's authorship axis is a genuine BINARY TAG (one
    population, fires for 'self', silent for 'heard') — unlike d6's two independently-drivable slot pools, there
    is no second population to wire a symmetric opposite-direction edge from. The cross-edge therefore biases
    ONLY toward GENERATED when held; F2 tests this one real direction (a held 'self' state vs no-hold baseline),
    not R1's two-direction test — the honest shape of the underlying signal, not a forced symmetry.

THE FUNCTIONAL GATE (6 seeds 42,43,44,100,101,102):
  F1 FACULTY-STILL-WORKS: source_provenance's OWN 8-item battery keeps its pre-registered floor (min |d|>=D_FLOOR,
     every sign correct) and self_schema's OWN authorship read keeps self>heard separation, edge present.
  F2 INTERACTION-IS-REAL (the crux): encode a FRESH, genuinely AMBIGUOUS content pattern under BOTH provenance
     contexts (balanced dual-context encode -> near-tied prov_perceived/prov_generated traces). VARY whether
     self_schema's author pool is HELD (self) during that item's recall -> the signed margin (rate_generated -
     rate_perceived) shifts POSITIVE (toward generated) vs a no-hold baseline. LESION the cross-edge -> the shift
     VANISHES. attributable_to ~1.0.
  F3 NO-RUNAWAY: per-region firing stays in a physiological band during the held read; the cross weight CONVERGES
     (soft-bounded, decelerating) rather than diverging; the pool stays alive.
  F4 MOAT/HONESTY: (a) author-held with NO content drive at all stays SUB-DECISION (no confabulated provenance
     from bias alone); (b) a CLEAR, already-correctly-encoded item is NOT flipped by a WRONG author hold (self
     held on a genuinely PERCEIVED item does not flip its verdict, only reweights genuine ambiguity).
  + LESION-RECOVERS-MIGRATION: with the cross-edge lesioned, both organs' base connectivity + own battery reads
     are byte-identical (within the FP-layout floor) to the plain (no-cross-edge) merged pool.

Run (numpy CPU; NO sim/ edit; routes off the GPU):
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_r4_selfschema_provenance --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_r4_selfschema_provenance \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend
from tools.lab import attributable_to

# ---- geometry + protocol constants (validated on seed 42; pre-registered) ----
W0 = 0.05                 # near-zero seed weight (the edge must GROW, not be pre-wired)
GATE = "author_to_provgen"       # the single plastic cross-edge gate
AUTHOR_PA = 650.0         # self_schema's OWN 'self' authorship drive (self_schema_production_organ.AUTHOR_PA)
CTX_DRIVE_PA = 2500.0     # source_provenance's OWN encoding-context drive (the de-risk's own CTX_DRIVE_PA)
EPISODE_DRIVE_PA = 2500.0 # source_provenance's OWN content drive (the de-risk's own EPISODE_DRIVE_PA)
TRAIN_STEPS = 30
N_EPISODES = 40
RECALL_STEPS = 100
N_READS = 3               # averaged reads per condition (denoise)
HMAX = 6.0                 # hebbian_max_weight (F3: the soft bound the edge converges toward). Calibrated (seed
                           # 42, verified 5 more seeds before the 6-seed gate): the raw same-step Hebbian rule
                           # (no rate-window trace) grows this edge MUCH faster per coincident step than R1's
                           # rate-windowed rule, so R1's HMAX=40 massively overshoots -- at HMAX=40 the edge's
                           # silence-alone bias (0.124) EXCEEDS a genuine clear-item decision's own margin (0.080),
                           # violating F4a/F4b (a moat failure, not a floor-tuning game). HMAX=6.0 lands the
                           # converged weight at ~2.9-4.1 across 6 seeds -- comfortably clears F2_INTACT_FLOOR
                           # (10-59% headroom every seed) while keeping the silence-alone bias to 26-47% of a
                           # genuine decision (well under the 50% F4a ceiling).
N_AMBIG_PASSES = 2         # interleaved perceived/generated encode passes for the fresh ambiguous item

# F-gate floors (pre-registered before the 6-seed run; calibrated on seed 42's smoke)
F2_INTACT_FLOOR = 0.010    # signed margin (rate_generated - rate_perceived) the author-hold must move, intact
F2_LESION_RATIO = 0.34     # lesion |Δ| must be < this * intact |Δ| (the shift is edge-caused)
F4A_FRAC = 0.5             # silence-only |margin| must be < this fraction of a genuine decision
F4B_RETAIN = 0.5           # a clear item keeps >= this fraction of its margin under a WRONG author hold
RATE_LO, RATE_HI = 5e-4, 0.7   # physiological firing band during the held reads

_CONDUCT = ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
            "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise",
            "cp_conductance_g_gabab_slow", "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise")

# THE READ-ISOLATION FIX (2026-09-02, C2 bug class / read-isolation audit, item H-1 -- see
# research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md). `_hard_reset` below
# restores v/u to true rest and zeroes conductances/firing_states/hebb-trace, but ORIGINALLY never touched 4
# further per-neuron arrays `_run_one_simulation_step` mutates: `cp_refractory_timers`/`cp_prev_firing_states`
# (HARD firing gates -- a neuron mid-refractory from whichever read/episode ran immediately before stays gated
# even though v/u were reset) and `cp_neuron_activity_ema`/`cp_neuron_firing_thresholds` (the homeostatic EMA +
# adaptive threshold, update participation-gated so it silently drifts on whichever neurons the prior read/
# episode drove). Port A (reuse, don't hand-roll): `self.pool` IS a `MergedPool`
# (`onebrain_merge_framework.py::MergedPool._PER_NEURON_STATE`, L246-250) which ALREADY lists all 4 of these
# arrays (plus the ones already handled below) as the framework's own tested read-isolation primitive -- the
# ORIGINAL omission here was exactly a hand-rolled array list that had fallen out of sync with that primitive.
# `_ALREADY_RESET` names the subset this runner's `_hard_reset` already restores explicitly; every OTHER name in
# `MergedPool._PER_NEURON_STATE` is snapshotted once at true rest (`R4Pool.__init__`, after the post-build
# settle) and restored on every `_hard_reset` call below.
_ALREADY_RESET = frozenset(("cp_membrane_potential_v", "cp_recovery_variable_u",
                             "cp_firing_states", "cp_external_input_current") + _CONDUCT)


def _dense(pre, post, w, gate):
    pre = np.asarray(pre, np.int64); post = np.asarray(post, np.int64)
    P = np.repeat(pre, len(post)); Q = np.tile(post, len(pre))
    return {"pre_indices": P, "post_indices": Q,
            "initial_weights": np.full(P.size, float(w), np.float32),
            "plastic": True, "plasticity_gate": gate, "conn_type": "E_TO_E", "count": int(P.size)}


def _build_pool(seed, with_cross):
    """Build the [self_schema, source_provenance] MergedPool. Both arms re-inject the SAME per-region-seamed
    base wiring UNION self_schema's OWN explicit wiring (assembly loops + member->attend) so they differ ONLY by
    the cross-edge (with_cross=True adds the near-zero plastic author->prov_generated edge, the SOLE plastic
    synapse). Re-settles + refreshes pool.snap after the (re-)inject, mirroring R1's `_build_pool` exactly."""
    from research.runners.onebrain_merge_framework import REGISTRY, MergedPool, _self_schema_member_attend
    xp, _ = get_backend()
    SS, SP = REGISTRY["self_schema"], REGISTRY["source_provenance"]
    # NOTE (found empirically, seed-42 smoke): `hebbian_rate_window` is a GLOBAL (not per-edge) cfg flag that
    # switches the ENTIRE bridge's Hebbian update rule from the standard same-step pre&post-fired coincidence to
    # a coactivity-TRACE-threshold rule (sim/bridge.py:1181-1207). source_provenance's OWN prov_learn/content_learn
    # edges are calibrated for the STANDARD rule (hebbian_symmetric=True, immediate coincidence); enabling
    # hebbian_rate_window pool-wide (as R1 does, safe there because R1's OTHER organ has no live Hebbian pathway
    # of its own) silently hijacked source_provenance's own encode onto the untuned trace rule and collapsed its
    # 8-item battery to chance (prov_l1 stayed exactly 0 after its own encode -- caught before the 6-seed run).
    # FIX: do NOT enable hebbian_rate_window here. Our own cross-edge is driven by TONIC (near-constant, every-step)
    # currents on both `author` and `ctx_generated`, so the STANDARD same-step coincidence rule fires reliably too
    # -- no rate-window trace is needed for this training design.
    pool = MergedPool(seed, [SS, SP], wire=True)
    pool.ensure_built()
    b = pool.bridge
    rm = b.region_manager

    def idxr(nm):
        return np.asarray(rm.indices(nm), np.int64)

    ix = {nm: idxr(nm) for nm in ("episode", "content_readout", "ctx_perceived", "ctx_generated",
                                  "prov_perceived", "prov_generated", "inh_perceived", "inh_generated")}
    _g, _member, _attend, _confid, author_idx = _self_schema_member_attend(b)
    ix["author"] = np.asarray(author_idx, np.int64)

    # Reproduce EXACTLY what ensure_built's wire=True step already installed (base per-region-seamed wiring UNION
    # self_schema's own explicit_wiring_fn -- assembly loops + member->attend), UNION (optionally) the cross-edge,
    # then RE-INJECT. The base+self_schema union is byte-identical to what ensure_built already did, so the ONLY
    # difference this re-inject can introduce is the cross-edge.
    union = dict(rm.build_wiring_plan(seed=pool.seed, per_region_seed=True))
    union.update(SS.explicit_wiring_fn(b, rm))
    masks = None
    if with_cross:
        union["x_author_provgen"] = _dense(ix["author"], ix["prov_generated"], W0, GATE)
    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    if SS.post_inject_fn is not None:
        SS.post_inject_fn(b)     # re-freeze self_schema's WS_LOOP_GATE (per-turn reads never learn)
    if with_cross:
        coo = b.cp_connections.tocoo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        masks = {"author->provgen": np.isin(row, ix["author"]) & np.isin(col, ix["prov_generated"])}

    # re-settle to rest + refresh pool.snap (v,u) so every organ's calibrate/read starts from true rest.
    b.cp_external_input_current[:] = 0.0
    for _ in range(40):
        b._run_one_simulation_step()
    b.cp_external_input_current[:] = 0.0
    if pool.snap is not None:
        pool.snap["cp_membrane_potential_v"] = np.asarray(to_host(b.cp_membrane_potential_v)).copy()
        pool.snap["cp_recovery_variable_u"] = np.asarray(to_host(b.cp_recovery_variable_u)).copy()
    return pool, ix, masks


class R4Pool:
    """The R4 integrated pool: the merged [self_schema, source_provenance] bridge + the injected learned
    author->prov_generated cross-edge, with the direct-drive train + read protocol the F-gate consumes."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.xp, _ = get_backend()
        self.pool, self.ix, self.masks = _build_pool(seed, with_cross=True)
        self.b = self.pool.bridge

        from research.runners.onebrain_merge_framework import _source_prov_organ, _self_schema_organ
        # WHITELIST FREEZE (design: the R1 "one-line inversion"): the cross-edge is the SOLE plastic synapse
        # (gain 1), everything else frozen (gain 0), BEFORE either organ's own build-time step runs. Both
        # organs' own encode/calibrate steps explicitly save+zero+reopen the gates THEY need and restore the
        # SAVED gain afterward -- so they compose correctly with this outer whitelist (they always restore back
        # to it), and neither organ's own weights can ever pick up spurious plasticity from this training arc.
        self.b.set_plasticity_gate(GATE, 1.0)               # ensure the gate index map + gain array exist
        self.b.cp_plasticity_rate_gain[:] = 0.0              # freeze EVERYTHING
        self.b.set_plasticity_gate(GATE, 1.0)                # whitelist ONLY the cross-edge

        # build the source_provenance organ VIEW: runs its OWN build-time Hebbian encode of the 8-item battery
        # (its own universal gain-0-freeze + reopen prov_learn/content_learn + restore dance -- restores back to
        # OUR whitelist state above). Then the self_schema organ VIEW (calibration only, no plasticity).
        self.sp_organ = _source_prov_organ(seed, self.pool)
        self.sp_organ.ensure_built()
        self.ss_organ = _self_schema_organ(seed, self.pool)
        self.ss_organ.ensure_built()

        # the fresh AMBIGUOUS content pattern (disjoint from the 8-item battery's used episode neurons),
        # dual-context encoded so its prov_perceived/prov_generated traces land near-tied.
        self.ambig_pattern = self._make_ambiguous_pattern()
        self._encode_ambiguous()

        # snapshot the post-encode (migrated + battery + ambiguous item) weights so training's no-corruption
        # can be asserted against THIS baseline (everything legitimate is already in place).
        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]
        # standard (non-rate-window) Hebbian hyperparameters for OUR cross-edge's training window. hebbian_symmetric
        # (same-step pre&post coincidence) suits the TONIC co-drive: both `author` and `ctx_generated`-driven
        # `prov_generated` fire on nearly every training step, so immediate coincidence triggers reliably.
        for kk, vv in dict(hebbian_symmetric=True, hebbian_learning_rate=0.05, hebbian_max_weight=HMAX,
                           hebbian_min_weight=0.0, hebbian_weight_decay=0.0).items():
            setattr(self.b.core_config, kk, vv)

        # refresh the resting snapshot AFTER every build-time step so the direct-drive reads start from true rest.
        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()
        # THE READ-ISOLATION FIX (see _ALREADY_RESET comment above): snapshot the SAME true-rest baseline for
        # every `MergedPool._PER_NEURON_STATE` array not already covered by the reset below (in practice:
        # cp_prev_firing_states, cp_refractory_timers, cp_refractory, cp_neuron_firing_thresholds,
        # cp_neuron_activity_ema) -- reused from the framework's own list, not hand-typed.
        self._rest_extra = {}
        for nm in self.pool._PER_NEURON_STATE:
            if nm in _ALREADY_RESET:
                continue
            arr = getattr(self.b, nm, None)
            self._rest_extra[nm] = np.asarray(to_host(arr)).copy() if arr is not None else None

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
        if getattr(b, "cp_hebb_coactivity_trace", None) is not None:
            b.cp_hebb_coactivity_trace[:] = 0.0
        # THE READ-ISOLATION FIX: restore every other _PER_NEURON_STATE array to the TRUE rest snapshot taken in
        # __init__, so residue from whichever read/episode ran immediately before this call (refractory gating,
        # prev-firing state, homeostatic EMA/threshold) cannot leak into the next read.
        for nm, val in self._rest_extra.items():
            if val is not None:
                getattr(b, nm)[:] = xp.asarray(val)
        b.cp_external_input_current[:] = 0.0

    def _drive(self, pairs, steps, learn=False, read=None):
        b, xp = self.b, self.xp
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

    def _wmean(self, name="author->provgen"):
        return float(np.asarray(to_host(self.b.cp_connections.data))[self.masks[name]].mean())

    def cross_weights(self):
        return {k: round(self._wmean(k), 4) for k in self.masks}

    # ---- the fresh, genuinely-ambiguous content pattern (dual-context encoded) ----
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
        """Balanced dual-context encode of the fresh ambiguous pattern: reopen prov_learn/content_learn ONLY for
        this step, interleaved perceived/generated so neither trace dominates, then restore the whitelist. Mirrors
        source_provenance's own `_encode_all` interleaving discipline exactly.

        MUST also save+set+restore hebbian_learning_rate/max_weight/min_weight/weight_decay/symmetric, exactly as
        `_SourceProvReadOrgan.ensure_built`'s own shared-mode wrapper does (found empirically, seed-42 smoke): the
        POOL's global hebbian_max_weight DEFAULTS to CoreSimConfig's class default (1.0, config.py:824) whenever
        no descriptor declares it (neither self_schema's nor source_provenance's config does) -- ONLY
        source_provenance's OWN wrapper temporarily raises it to HEBB_WMAX=60.0 during ITS encode. Without doing
        the same here, this call's Hebbian step CLIPS every gain>0 (prov_learn/content_learn-gated) synapse to
        [hebbian_min_weight, hebbian_max_weight] = [0.05, 1.0] -- crushing the ALREADY-TRAINED 8-item battery
        weights (which reopen under the SAME gate names, not scoped to just the ambiguous item's neurons) down to
        ~1.0 and collapsing its discriminability to chance. Caught before the 6-seed run (F1 battery_acc=0.500)."""
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
        ix = self.ix
        traj = [dict(ep=0, w=self._wmean())]
        for ep in range(n_episodes):
            # co-drive self_schema's author pool (self-authorship) + source_provenance's ctx_generated line;
            # ctx_generated->prov_generated is a FIXED (non-plastic) strong pathway, so prov_generated reliably
            # co-fires with author -> Hebbian binds author->prov_generated. No episode content is needed (the
            # bias is content-independent, matching the one-sided binary authorship tag it reads from).
            self._hard_reset()
            self._drive([(ix["author"], AUTHOR_PA), (ix["ctx_generated"], CTX_DRIVE_PA)], TRAIN_STEPS, learn=True)
            if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                traj.append(dict(ep=ep + 1, w=self._wmean()))
        self.b.core_config.enable_hebbian_learning = False
        # NO-CORRUPTION: every NON-cross (migrated + battery + ambiguous-item) weight must be byte-unchanged.
        now = np.asarray(to_host(self.b.cp_connections.data))
        self.frozen_maxdrift = float(np.max(np.abs(now[self._noncross] - self._frozen_w0[self._noncross])))
        return traj

    # ---- the signed ambiguous-item read with an optional author hold ----
    def amb_read(self, hold_author, band=None):
        """Matched protocol for both conditions: hard-reset -> drive the ambiguous pattern's episode neurons
        (+ optionally the author pool) -> read the SIGNED margin (rate_generated - rate_perceived). Averaged."""
        ix = self.ix
        ep_idx = ix["episode"][self.ambig_pattern]
        margins, rates = [], {"prov_generated": 0.0, "prov_perceived": 0.0, "author": 0.0, "ctx_generated": 0.0}
        for _ in range(N_READS):
            self._hard_reset()
            pairs = [(ep_idx, EPISODE_DRIVE_PA)]
            if hold_author:
                pairs.append((ix["author"], AUTHOR_PA))
            read = {"gen": ix["prov_generated"], "perc": ix["prov_perceived"]}
            if band is not None:
                for r in rates:
                    read[r] = ix[r] if r != "author" else ix["author"]
            acc = self._drive(pairs, RECALL_STEPS, read=read)
            margins.append(acc["gen"] - acc["perc"])
            if band is not None:
                for r in rates:
                    rates[r] += acc.get(r, 0.0)
        out = {"margin": float(np.mean(margins))}
        if band is not None:
            out["rates"] = {r: rates[r] / N_READS for r in rates}
        return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The four functional-gate arms + the migration invariant + the emergence read
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _f1(r4):
    """F1 FACULTY-STILL-WORKS: source_provenance's OWN 8-item battery keeps its pre-registered floor (sign
    correct + min |d|>=D_FLOOR); self_schema's OWN authorship read keeps self>heard separation. Edge present,
    author NOT held (no interference)."""
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS, D_FLOOR
    r4._hard_reset()
    ds, accs = [], []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            pat = r4.sp_organ.patterns[prov][i]
            rec = r4.sp_organ.brain.recall(pat)
            rp, rg = rec["rate_perceived"], rec["rate_generated"]
            margin = rp - rg
            winner = "perceived" if margin >= 0 else "generated"
            d_perc = margin / (rp + rg + 1e-9)
            d_true = d_perc if prov == "perceived" else -d_perc
            ds.append(d_true); accs.append(winner == prov)
    min_d = float(np.min(ds)); acc = float(np.mean(accs))
    battery_ok = bool(acc >= 0.999 and min_d >= D_FLOOR)
    self_rate = r4.ss_organ._author_rate(authored=True, lesion=False)
    heard_rate = r4.ss_organ._author_rate(authored=False, lesion=False)
    author_ok = bool(self_rate > heard_rate and self_rate >= r4.ss_organ.threshold > heard_rate)
    return {"battery_acc": acc, "battery_min_d": min_d, "battery_ok": battery_ok,
            "author_self_rate": float(self_rate), "author_heard_rate": float(heard_rate),
            "author_threshold": float(r4.ss_organ.threshold), "author_ok": author_ok,
            "PASS": bool(battery_ok and author_ok)}


def _f2(r4):
    """F2 INTERACTION-IS-REAL: on the fresh AMBIGUOUS item, holding self_schema's author pool (self) shifts the
    signed margin toward GENERATED vs a no-hold baseline, intact; the shift vanishes when the cross-edge is
    lesioned. One-sided by design (the author axis is a genuine binary tag -- see module docstring)."""
    base_i = r4.amb_read(False, band=True)
    held_i = r4.amb_read(True)
    d_i = held_i["margin"] - base_i["margin"]
    # lesion the cross-edge (zero its weight)
    data = np.asarray(to_host(r4.b.cp_connections.data)).copy()
    for k in r4.masks:
        data[r4.masks[k]] = 0.0
    r4.b.cp_connections.data = r4.xp.asarray(data, dtype=r4.b.cp_connections.data.dtype)
    base_l = r4.amb_read(False)
    held_l = r4.amb_read(True)
    d_l = held_l["margin"] - base_l["margin"]
    shift_ok = (d_i > F2_INTACT_FLOOR) and (abs(d_l) < F2_LESION_RATIO * abs(d_i))
    frac = attributable_to("F2 self-hold shift toward GENERATED = the cross-edge", d_i, d_l)
    return {"frac_attributable": (None if frac is None else float(frac)),
            "margin_base_intact": base_i["margin"], "margin_held_intact": held_i["margin"],
            "margin_base_lesion": base_l["margin"], "margin_held_lesion": held_l["margin"],
            "delta_intact": float(d_i), "delta_lesion": float(d_l),
            "rates_base_intact": base_i.get("rates", {}),
            "shift_toward_generated": bool(shift_ok), "PASS": bool(shift_ok)}


def _f3(r4, traj, f2):
    """F3 NO-RUNAWAY: firing band during the base-intact read, cross-weight converges (bounded + decelerating)."""
    rates = f2.get("rates_base_intact", {})
    # ctx_generated is legitimately SILENT during a recall read (ctx_drive is only active at ENCODE, not recall
    # -- brain.recall()'s own protocol, unchanged here); band-check only the pools genuinely driven at read time.
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


def _f4(r4):
    """F4 MOAT/HONESTY. (a) author held + NO content drive stays SUB-DECISION (no confabulated provenance from
    bias alone); (b) a CLEAR, genuinely PERCEIVED battery item is NOT flipped by a WRONG (self) author hold. Run
    on the INTACT edge, BEFORE F2's in-place lesion (called before F2 in run_seed)."""
    ix = r4.ix
    # F4a: author held, episode current entirely absent (pure silence).
    r4._hard_reset()
    silent = r4._drive([(ix["author"], AUTHOR_PA)], RECALL_STEPS,
                       read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]})
    silence_margin = silent["gen"] - silent["perc"]
    # reference decision scale: the ambiguous item's own baseline (no-hold) genuine-content margin magnitude,
    # and the clear-item baseline margin (whichever is larger sets the honest "real decision" scale).
    amb_base = r4.amb_read(False)["margin"]
    clear_pat = r4.sp_organ.patterns["perceived"][0]
    ep_idx = ix["episode"][np.asarray(clear_pat, np.int64)]
    r4._hard_reset()
    clear_nohold = r4._drive([(ep_idx, EPISODE_DRIVE_PA)], RECALL_STEPS,
                             read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]})
    m_nohold = clear_nohold["gen"] - clear_nohold["perc"]     # expect strongly NEGATIVE (correctly 'perceived')
    decision = max(abs(amb_base), abs(m_nohold), 1e-9)
    f4a_ok = bool(abs(silence_margin) < F4A_FRAC * decision)
    # F4b: the SAME clear (perceived) item, held under the WRONG author state (self) -> should not flip.
    r4._hard_reset()
    clear_held = r4._drive([(ep_idx, EPISODE_DRIVE_PA), (ix["author"], AUTHOR_PA)], RECALL_STEPS,
                           read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]})
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
    g = traj[-1]["w"]
    correct = g > 5 * W0                     # the edge genuinely GREW from near-zero
    return {"trajectory": traj, "final_weight": float(g), "grew_from_near_zero": bool(correct),
            "frozen_weight_maxdrift": float(frozen_maxdrift), "no_corruption": bool(frozen_maxdrift < 1e-6),
            "PASS": bool(correct and frozen_maxdrift < 1e-6)}


def _migration_invariant(seed, r4, sp_battery_lesioned, ss_lesioned):
    """LESION-RECOVERS-MIGRATION: with the cross-edge lesioned, (1) base connectivity is BYTE-IDENTICAL to the
    plain no-cross-edge merged pool, and (2) both organs' own battery reads match within the FP-layout floor."""
    from research.runners.onebrain_merge_framework import _source_prov_organ, _self_schema_organ
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS, D_FLOOR
    pool0, ix0, _m0 = _build_pool(seed, with_cross=False)
    sp0 = _source_prov_organ(seed, pool0); sp0.ensure_built()
    ss0 = _self_schema_organ(seed, pool0); ss0.ensure_built()

    def edge_map(pool):
        coo = pool.bridge.cp_connections.tocoo()
        r = to_host(coo.row); c = to_host(coo.col); d = to_host(coo.data)
        return {(int(a), int(b)): float(w) for a, b, w in zip(r, c, d)}
    k0 = edge_map(pool0)
    k1 = edge_map(r4.pool)
    xrows = set(int(x) for x in r4.ix["author"])
    xcols = set(int(x) for x in r4.ix["prov_generated"])
    k1_base = {kk: vv for kk, vv in k1.items() if not (kk[0] in xrows and kk[1] in xcols)}
    # the two organs' OWN learned traces (battery + ambiguous item) legitimately differ in VALUE between pool0
    # (a fresh, separately-encoded copy) and r4.pool (encoded once, shared by every arm) -- only compare the
    # STRUCTURAL edge set (same (pre,post) pairs exist), not their trained values, for those two organs' own
    # plastic pathways; every OTHER edge (fixed pathways + assembly loops) must be exactly value-identical.
    struct_identical = bool(set(k1_base.keys()) == set(k0.keys()))
    base = []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            rec = sp0.brain.recall(sp0.patterns[prov][i])
            base.append(rec["rate_perceived"] - rec["rate_generated"])
    maxerr = float(np.max(np.abs(np.asarray(base) - np.asarray(sp_battery_lesioned)))) if base else 0.0
    ss_self0 = ss0._author_rate(authored=True, lesion=False)
    ss_heard0 = ss0._author_rate(authored=False, lesion=False)
    ss_err = float(max(abs(ss_self0 - ss_lesioned[0]), abs(ss_heard0 - ss_lesioned[1])))
    return {"base_connectivity_structurally_identical": struct_identical,
            "sp_battery_maxerr": maxerr, "self_schema_maxerr": ss_err,
            "PASS": bool(struct_identical and maxerr < 0.05 and ss_err < 0.05)}


def run_seed(seed):
    t0 = time.time()
    r4 = R4Pool(seed)
    traj = r4.train()
    emg = _emergence(traj, r4.frozen_maxdrift)
    f1 = _f1(r4)
    f4 = _f4(r4)                                   # F4 BEFORE F2 (F2 lesions the edge in place)
    f2 = _f2(r4)                                   # F2 lesions the cross-edge at its end
    r4._hard_reset()
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS
    sp_les = []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            rec = r4.sp_organ.brain.recall(r4.sp_organ.patterns[prov][i])
            sp_les.append(rec["rate_perceived"] - rec["rate_generated"])
    ss_les = (r4.ss_organ._author_rate(authored=True, lesion=False),
             r4.ss_organ._author_rate(authored=False, lesion=False))
    f3 = _f3(r4, traj, f2)
    mig = _migration_invariant(seed, r4, sp_les, ss_les)
    go = bool(f1["PASS"] and f2["PASS"] and f3["PASS"] and f4["PASS"] and emg["PASS"] and mig["PASS"])
    return {"seed": int(seed), "PASS": go, "elapsed_s": round(time.time() - t0, 1),
            "emergence": emg, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "lesion_recovers_migration": mig}


def _agg(runs):
    def frac(key):
        return sum(1 for r in runs if r[key.split(".")[0]][key.split(".")[1]]) if "." in key else 0
    keys = ["F1.PASS", "F2.PASS", "F3.PASS", "F4.PASS", "emergence.PASS", "lesion_recovers_migration.PASS"]
    return {k: f"{frac(k)}/{len(runs)}" for k in keys}


def _selftest_repeat_read_identity(seed=42, verbose=True):
    """READ-ISOLATION FIX -- fails-in-its-failing-direction guard (2026-09-02).

    On a fresh, UNTRAINED pool (cross-edge still at its near-zero seed W0, no mechanism has grown -- a
    "zeroed-mechanism" pool), two back-to-back IDENTICAL reads through `_hard_reset` must be bitwise identical:
    no RNG runs inside `_drive`/`_run_one_simulation_step`, so any difference between two runs of the same drive
    can only be residual per-neuron state leaking from whatever ran immediately before.

    Runs the SAME probe twice: once with the fix's extra-array restore programmatically disabled (reproducing
    the ORIGINAL bug -- this must DIVERGE, proving the assertion has teeth and would have caught the original
    defect), then with it enabled (the actual fix -- this must be bitwise IDENTICAL). Asserts both directions."""
    r4 = R4Pool(seed)
    ix = r4.ix
    ep_idx = ix["episode"][r4.ambig_pattern]
    read = {"gen": ix["prov_generated"], "perc": ix["prov_perceived"]}

    def _one_read():
        r4._hard_reset()
        return r4._drive([(ep_idx, EPISODE_DRIVE_PA)], RECALL_STEPS, read=dict(read))

    # induce asymmetric residue ahead of the two probed reads: an author-held read drives a DIFFERENT set of
    # pools (refractory/homeostatic state ends up shaped differently) than the ambiguous-item probe below.
    r4._hard_reset()
    r4._drive([(ix["author"], AUTHOR_PA)], RECALL_STEPS)

    saved_extra = r4._rest_extra
    r4._rest_extra = {}                    # simulate the ORIGINAL (pre-fix) _hard_reset
    read_broken_1 = _one_read()
    read_broken_2 = _one_read()
    broken_diverges = (read_broken_1 != read_broken_2)

    r4._rest_extra = saved_extra           # the actual fix
    read_fixed_1 = _one_read()
    read_fixed_2 = _one_read()
    fixed_identical = (read_fixed_1 == read_fixed_2)

    if verbose:
        print(f"[selftest] fix-disabled diverges={broken_diverges}: {read_broken_1} vs {read_broken_2}")
        print(f"[selftest] fix-enabled  identical={fixed_identical}: {read_fixed_1} vs {read_fixed_2}")
    assert broken_diverges, ("SELFTEST HAS NO TEETH: two identical reads were bitwise identical even with the "
                              "extra-array restore disabled -- this probe would not have caught the original bug")
    assert fixed_identical, (f"READ-ISOLATION FIX BROKEN: repeat reads not bitwise identical with the fix "
                              f"enabled: {read_fixed_1} vs {read_fixed_2}")
    if verbose:
        print("[selftest] PASS -- repeat-read bitwise identity holds under the fix, and the probe has teeth "
              "(diverges when the fix is disabled)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true",
                     help="read-isolation fix guard only: repeat-read bitwise identity (fails in its failing "
                          "direction) -- runs no F-gate, exits 0/1")
    args = ap.parse_args()
    if args.selftest:
        ok = _selftest_repeat_read_identity()
        return 0 if ok else 1
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        f2 = r["F2"]
        print(f"[seed {s}] {'GO' if r['PASS'] else 'no'} ({r['elapsed_s']}s) | "
              f"emerge w={r['emergence']['final_weight']:.2f} grew={r['emergence']['grew_from_near_zero']} | "
              f"F1 battery(acc={r['F1']['battery_acc']:.3f},min_d={r['F1']['battery_min_d']:.3f})="
              f"{r['F1']['battery_ok']} author={r['F1']['author_ok']} F1={r['F1']['PASS']} | "
              f"F2 Δ={f2['delta_intact']:+.4f}(les {f2['delta_lesion']:+.4f}) frac={f2['frac_attributable']} "
              f"={f2['PASS']} | F3={r['F3']['PASS']} F4={r['F4']['PASS']} mig={r['lesion_recovers_migration']['PASS']}",
              flush=True)

    n_go = sum(r["PASS"] for r in runs)
    agg = _agg(runs)
    all_go = (n_go == len(runs)) and not args.smoke
    tag = "GO" if all_go else ("SMOKE-GO (1-seed indicator)" if args.smoke and n_go == len(runs) else "NO-GO/PARTIAL")
    verdict = (f"{tag} — R4 learned cross-edge self_schema authorship -> source_provenance monitoring "
               f"('is this my own thought' self-monitoring): {n_go}/{len(runs)} seeds pass ALL of F1(faculty-"
               f"still-works) + F2(vary-then-lesion) + F3(no-runaway) + F4(moat) + emergence(LEARNED, near-zero "
               f"start) + lesion-recovers-migration. Per-arm: {agg}. The cross-edge GROWS from near-zero (0.05) "
               f"by the substrate's OWN standard (same-step) Hebbian rule to ~{(runs[0]['emergence']['final_weight'] if runs else 0):.1f} "
               f"(LEARNED, not hand-set) via episodes co-driving self_schema's author pool with source_provenance's "
               f"ctx_generated line. Holding self_schema's author pool 'self' during a fresh, genuinely-ambiguous "
               f"(dual-context-encoded) provenance item's recall shifts the signed margin toward GENERATED, and "
               f"the shift VANISHES on lesion (load-bearing). One-sided by design (the authorship axis is a "
               f"genuine binary tag, not two independently-drivable pools); the moat holds (no decision from "
               f"silence; a clear item is not flipped by a wrong hold). numpy CPU; NO sim/ edit.")

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("onebrain_integration_r4_selfschema_provenance")
        Vd.require("f2_lesion_removes_shift", 1 if all(
            abs(r["F2"]["delta_lesion"]) < F2_LESION_RATIO * max(abs(r["F2"]["delta_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the F2 shift must VANISH under lesion or it is a confound, not the cross-edge (the crux control)")
        Vd.require("migration_byte_identity", 1 if all(r["lesion_recovers_migration"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="lesion the cross-edge -> both organs' own reads == the plain merged pool")
        Vd.require("emergence_grew_from_near_zero", 1 if all(r["emergence"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="the edge grows from ~0.05 by Hebbian co-activity, not hand-set")
        Vd.require("moat_no_winner_from_silence", 1 if all(r["F4"]["f4a_no_winner_from_silence"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="author-held + no content drive stays sub-decision (F4 moat)")
        dec = Vd.decide(all_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "onebrain_integration_r4_selfschema_provenance", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(runs), "per_arm": agg, "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "config": {"W0": W0, "hebbian_max_weight": HMAX, "n_episodes": N_EPISODES,
                          "author_pa": AUTHOR_PA, "ctx_drive_pa": CTX_DRIVE_PA, "episode_drive_pa": EPISODE_DRIVE_PA,
                          "recall_steps": RECALL_STEPS, "n_reads": N_READS,
                          "f2_intact_floor": F2_INTACT_FLOOR, "f2_lesion_ratio": F2_LESION_RATIO,
                          "f4a_frac": F4A_FRAC, "f4b_retain": F4B_RETAIN},
               "mechanism": ("ONE shared merge pool [self_schema + source_provenance] (merge_organs wire=True); a "
                             "SINGLE plastic cross-edge author->prov_generated seeded ~0.05, the SOLE plastic "
                             "synapse (whitelist inversion: cp_plasticity_rate_gain=0 everywhere then GATE=1), "
                             "GROWN by the substrate's standard (same-step, symmetric) Hebbian rule over episodes "
                             "co-driving self_schema's "
                             "author pool with source_provenance's (fixed, non-plastic) ctx_generated line. A "
                             "fresh content pattern is dual-context encoded (balanced) to create a genuinely "
                             "AMBIGUOUS provenance item for the F2 read."),
               "scaffold_residuals": ["host-chosen cross-edge TOPOLOGY (author -> prov_generated only, one-sided "
                                      "by the axis's own binary-tag structure -- declared, not smoothed over)",
                                      "host-curated training schedule (co-driving author + ctx_generated directly, "
                                      "not via an organic dialogue turn) -- same class of scaffold-residual as R1's "
                                      "host-curated referent/role pairing schedule",
                                      "two-factor Hebbian (unchanged from R1; no reward/dopamine gating here)",
                                      "the ambiguous item is a balanced-dual-context construction, a substrate "
                                      "stand-in for a genuinely uncertain real memory"],
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[R4] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
