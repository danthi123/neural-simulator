"""EMERGENT NEURAL-WTA FRESH-SLOT ALLOCATION -- replace the multi-slot WM's HOST free-counter + HOST teaching-clamp with
an emergent, entity-specific neural WTA whose competition threshold SELF-CALIBRATES.

THE RESIDUAL (named by BOTH banked lanes -- do NOT re-derive; read the record):
  * `_multi_slot_binding_derisk.py` (banked multi-slot WM GO) uses `persistent_entity_binder` = a HOST HebbianBinder that
    assigns each entity a stable LOCAL slot index, and `MultiSlotHold.write(reg, local)` HOST-drives exactly that pool
    (the "post" of the bind). BOTH the free-counter (which fresh slot a NEW entity takes) and the teaching-clamp (driving
    that slot) are HOST. Lane b (spiking-STP-bind) and lane c (multi-slot) both named this as THE remaining shortcut.
  * RUNG6e (`2026-07-13-RUNG6e-...-freshslot-allocation-is-the-subproblem.md`) isolated the EXACT open sub-problem: on the
    real bridge, with equal barcode drive "a winner emerges but it is NOISE-picked (not entity-specific)"; "Hebbian
    specificity needs a clean high-rate winner"; the delicate WTA "can't yet deliver a clean, entity-specific, high-rate
    winner" (measured selectivity ~0.31 blur -- reproduced here in the smoke).
  * rungB1b (`2026-07-04-rungB1b-neural-role-wta-GO.md`, BANKED GO) proves the substrate CAN do entity-specific neural
    WTA read from the GATE the WTA opened, never a host argmax over the logits. We honour that read discipline: the
    winner is the pool the spiking attractor LATCHED (persistent activity, drive removed), read from SPIKES; selectivity
    is reported to prove it is a clean one-of-K, not a soft argmax over a blur.
  * The keystone-slot-binder research gate (2026-07-17) named the fix: a SELF-CALIBRATING competition threshold
    (HTM boosting / BCM sliding threshold / Turrigiano synaptic scaling / adaptive-theta) -- "what makes the WTA fair
    WITHOUT a hand-set cut." The lever-3 competitive stabilizer (feedback inhibition + divisive normalisation) is the
    reusable structural-competition mechanism.

THE MECHANISM (all runner-side host math on the substrate's OWN populations; NO sim/ edit -- SAME accepted scope as the
lever-3 stabilizer, whose 2-layer competition is host math with the on-substrate spiking realisation the named next rung):
  Substrate = ONE bank of K D3 slow-NMDA persistent-activity attractor pools sharing ONE FS pool
  (`build_persistent_slot(seed, K)` -- the SAME multi-slot substrate, R=1 bank; the allocation residual is WITHIN a bank:
  entity -> local slot). A NEW entity's barcode drives ALL K pools through a fixed developmental-random projection
  P (K x code_dim); the graded, ENTITY-SPECIFIC drive d = P @ barcode is injected as external current (the barcode->pool
  synaptic drive -- the SAME host-computed-drive residual `write()` already uses; the WTA SELECTION is neural, read from
  spikes). Two SELF-CALIBRATING competition mechanisms then resolve the blur into a clean, entity-specific, fresh-slot
  winner -- neither ever names a pool; both only set SCALARS / per-pool excitability:
    (A) ADAPTIVE COMPETITION THRESHOLD (the "fair WTA without a hand-set cut"). A pooled subtractive feedback inhibition
        `inh` common to all pools is raised/lowered by a homeostatic controller that reads only the COUNT of active pools
        (a population statistic -- Carandini-Heeger divisive/subtractive normalisation; the lever-3 feedback-inhibition
        motif) and drives it toward exactly ONE active pool: inh <- inh + inh_lr*(n_active - 1). The right inhibition
        level for a clean one-of-K DEPENDS on the drive magnitude (which varies per entity + occupancy), so no single
        HAND-SET cut works everywhere -- the adaptive threshold TRACKS the operating point (CLAUDE.md's deepest lesson:
        the operating point is implicit; a hand-set bound is the proxy that fails -> RUNG6e's noise-picked blur). The
        winner (highest total drive) survives as inh rises -> a clean high-rate latch. LOAD-BEARING for entity-specificity.
    (B) ADAPTIVE OCCUPANCY EXCITABILITY (HTM boosting / Turrigiano homeostatic intrinsic plasticity). A per-pool
        excitability `boost_k = -boost_beta * used_k` where used_k integrates pool k's own recent winning activity: a pool
        that just latched is homeostatically DEPRESSED, steering the NEXT novel entity to a FREE pool. Occupancy lives in
        a neural excitability trace, NOT a host free-counter -- the WTA still re-decides which pool. LOAD-BEARING for
        distinct-slot allocation (collision).
  BIND (retrieve) = a content-agnostic Hebbian fast weight W (K x code_dim): W[winner] += barcode/||barcode|| once the
  spiking WTA opens the winner. Re-presenting the entity drives the pools with W @ barcode -> the bound pool wins the SAME
  self-calibrating WTA -> retrieve, read from SPIKES. No host free-counter, no host teaching-clamp picks the slot.

ARMS (all required):
  * emergent_wta        -- the candidate: projection drive + (A)+(B) self-calibration + spiking WTA + Hebbian bind.
  * host_free_counter   -- the banked reference (host free-counter picks the fresh slot; host teaching-clamp drives it;
                           same spiking latch + spikes read). The ceiling to match on downstream recall.
  * lesion_selfcalib    -- the candidate with the SELF-CALIBRATION OFF (fixed hand-set inhibition + no occupancy boost)
                           -> MUST degrade to NOISE-PICKED (RUNG6e's blur): the load-bearing proof.
  * noise_picked_null   -- WTA with NO entity-conditioning (uniform drive replaces the projection) -> allocation at
                           chance (1/K): what "not entity-specific" looks like.

METRICS (per arm, held-out NOVEL entities minted at test):
  * entity_specific  = same-entity re-presentation -> SAME slot. Present ONE entity into an empty bank T times with
                       DIFFERENT noise realisations; mode-fraction of its winner slot, averaged over entities. The direct
                       RUNG6e noise-picked test. GO: >= 0.90 (null ~ 1/K).
  * collision        = distinct entities -> distinct slots. One sequential pass of R novel entities into an empty bank;
                       fraction landing on an already-claimed slot. GO: <= 0.10 (host = 0 by construction).
  * recall           = downstream: re-present each entity (retrieve) -> spiking WTA latch -> read slot from SPIKES ->
                       deref (ent_of_slot) -> agreeing verb. Compare to the host_free_counter ceiling.
  * selectivity      = winner_rate/total_rate of the LATCHED read (drive removed) -> a clean one-of-K (read from spikes),
                       not a soft argmax over the blur. Anti-cheat.

GO (6-seed: 42 43 44 100 101 102): emergent_wta entity_specific >= 0.90 AND collision <= 0.10 AND recall >= host - 0.05
(and >= 0.85), with the self-calibration LOAD-BEARING (lesion_selfcalib re-collapses to noise-picked: entity_specific
drops >= 0.30 below the candidate AND toward chance) and the null at chance -- NO host free-counter. HONEST-NEGATIVE is
first-class: if the emergent winner stays noise-picked (RUNG6e's wall), report exactly what the self-calibrating
competition could / couldn't fix, per component.

SIM_BACKEND=numpy (sub-200-neuron D3 loops are launch-bound: CPU faster). Reuse-by-import; NO sim/ edit. cfg.seed seeds
the substrate (verified: build twice at one seed -> identical firing thresholds).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._var_bind_emergent_wta_allocation_derisk --seeds 42 --smoke
6-seed decisive (fan across processes, then --merge-from):
  for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m \
    research.runners._var_bind_emergent_wta_allocation_derisk --seeds $s \
    --out research/findings/raw/_emergent_wta_alloc/seed_$s.json & done ; wait
  SIM_BACKEND=numpy python -m research.runners._var_bind_emergent_wta_allocation_derisk \
    --merge-from research/findings/raw/_emergent_wta_alloc/seed_*.json \
    --out research/findings/raw/_emergent_wta_alloc/emergent_wta_alloc_6seed.json
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

# reuse-by-import: the VERIFIED D3 spiking slow-NMDA persistent-activity HOLD slot (the multi-slot substrate, R=1 bank);
# the sparse developmental-random barcode mint (the same one the multi-slot runner uses).
from research.runners._d3_persistent_slot_derisk import build_persistent_slot, _pool_idx, _reset
from research.runners._novel_referent_hebbian_fastweight_derisk import _mint_codes, _DIM

try:
    from tools.lab import lever, attributable_to, void_if
except Exception:  # tools.lab optional at import time; the runner still runs
    def lever(name, before, after, required=True, continuous=None):
        print(f"  LEVER {name}: {before} -> {after}"); return before != after
    def attributable_to(label, t, c, warn_below=0.5):
        print(f"  attributable_to {label}: t={t} c={c}"); return None
    def void_if(cond, reason):
        if cond:
            print(f"  VOID: {reason}")
        return bool(cond)

OUT = Path("research/findings/raw/_emergent_wta_alloc/emergent_wta_alloc.json")


# ====================================================================================================================
# THE EMERGENT SELF-CALIBRATING NEURAL-WTA SLOT ALLOCATOR
#   one bank of K D3 slow-NMDA attractor pools sharing one FS; a barcode->pool projection injects graded entity-specific
#   drive; an ADAPTIVE competition threshold (subtractive feedback inhibition, homeostatic to one winner) + an ADAPTIVE
#   occupancy excitability (HTM boost) resolve a clean, entity-specific, fresh-slot latch; a Hebbian fast weight binds it.
#   The winner is READ FROM SPIKES (the latched pool, drive removed) -- never a host argmax over the drive logits.
# ====================================================================================================================
class EmergentSlotAllocator:
    def __init__(self, seed, K, code_dim=_DIM, recur=25.0, drive_gain=400.0, noise_pA=60.0,
                 self_calib=True, entity_cond=True,
                 inh_high_frac=0.95, inh_step_frac=0.08, settle_rounds=16, round_steps=6, consolidate_steps=16,
                 hold_steps=18, clear_steps=250, active_frac=0.5, boost_beta=380.0,
                 fixed_inh_frac=0.45, retrieve_theta=0.35):
        from sim.backend import to_host, from_host
        self._to_host, self._from_host = to_host, from_host
        self.seed = int(seed); self.K = int(K); self.code_dim = int(code_dim)
        self.sb = build_persistent_slot(seed, self.K, recur=recur)
        self.idx = _pool_idx(self.sb, self.K)
        self.pool_neurons = np.concatenate([self.idx[k] for k in range(self.K)])
        self.n = self.sb.core_config.num_neurons
        self.drive_gain, self.noise_pA = float(drive_gain), float(noise_pA)
        self.self_calib, self.entity_cond = bool(self_calib), bool(entity_cond)
        self.inh_high = float(inh_high_frac) * self.drive_gain     # start: all pools suppressed
        self.inh_step = float(inh_step_frac) * self.drive_gain     # release step per round
        self.settle_rounds, self.round_steps = int(settle_rounds), int(round_steps)
        self.consolidate_steps, self.hold_steps = int(consolidate_steps), int(hold_steps)
        self.clear_steps = int(clear_steps)
        self.active_frac, self.boost_beta = float(active_frac), float(boost_beta)
        self.fixed_inh = float(fixed_inh_frac) * self.drive_gain   # the lesion's HAND-SET cut
        self.retrieve_theta = float(retrieve_theta)
        # fixed developmental-random barcode->pool projection (seeded, entity-specific by construction)
        self.P = np.random.default_rng(seed + 101).normal(0.0, 1.0, (self.K, self.code_dim)) / np.sqrt(self.code_dim)
        # neural state carried ACROSS entities within an allocation episode (occupancy = homeostatic excitability trace)
        self.used = np.zeros(self.K)                       # accumulated winning activity per pool (HTM duty proxy)
        self.W = np.zeros((self.K, self.code_dim))         # Hebbian fast weight (barcode -> pool) for retrieve

    # ---- reset the WHOLE episode (empty bank): attractor + occupancy trace + fast weights ----
    def reset_episode(self):
        _reset(self.sb); self.used[:] = 0.0; self.W[:] = 0.0

    def _boost(self):
        # HTM / Turrigiano homeostatic occupancy excitability: recently-won pools depressed (steer new entities to free
        # pools). OFF (== the lesion / the null unless entity_cond needs it) when self_calib is False.
        if not self.self_calib:
            return np.zeros(self.K)
        return -self.boost_beta * self.used

    def _drive(self, barcode):
        if not self.entity_cond:                            # noise-picked NULL: uniform drive, no entity conditioning
            return np.full(self.K, 0.5)
        b = barcode / (np.linalg.norm(barcode) + 1e-9)
        d = self.P @ b
        return (d - d.min()) / (d.max() - d.min() + 1e-9)   # per-entity graded drive in [0,1]

    def _run(self, base_cur, steps, noise_rng):
        """Advance the substrate `steps` steps with external current `base_cur` (+ per-step noise on the pool neurons);
        return the per-pool mean firing rate over the window."""
        rates = np.zeros(self.K)
        for _ in range(steps):
            cur = base_cur.copy()
            if self.noise_pA > 0.0 and noise_rng is not None:
                cur[self.pool_neurons] += noise_rng.normal(0.0, self.noise_pA, self.pool_neurons.shape[0])
            self.sb.cp_external_input_current[:] = self._from_host(cur)
            self.sb._run_one_simulation_step()
            fir = np.asarray(self._to_host(self.sb.cp_firing_states)).astype(float)
            for k in range(self.K):
                rates[k] += fir[self.idx[k]].mean()
        return rates / max(steps, 1)

    def _compete_and_latch(self, per_pool_drive, noise_rng, clear_first=True, bias=None):
        """Inject a graded per-pool drive; the SELF-CALIBRATING competition threshold resolves it to ONE winner; then
        remove the drive and read the LATCHED winner from SPIKES. Returns (winner_or_-1, selectivity, hold_max,
        n_active_final, inh_final).

        SELF-CALIBRATING competition = a DOWN-RAMP (release-of-inhibition) search: a pooled subtractive inhibition
        common to all pools starts HIGH (every pool silent) and is RELEASED step by step; the FIRST pool to escape (the
        one with the highest total drive = the entity-specific projection winner, shaped by the recurrent + heterogeneous
        substrate) is the winner, and the ramp STOPS the moment exactly one pool is active -> a clean, high-rate,
        entity-specific latch WITHOUT a hand-set cut (the right release level DEPENDS on the drive margin, which varies
        per entity + occupancy -> no single fixed cut works; the adaptive ramp tracks it). The controller reads only the
        COUNT of active pools (a population statistic; Carandini-Heeger / the lever-3 feedback-inhibition motif) -- it
        NEVER selects a pool. LESION (self_calib=False): the inhibition is FROZEN at the HAND-SET fixed_inh (no ramp) ->
        the RUNG6e blur is not resolved -> a noise-picked winner."""
        if clear_first:                                     # clear residual NMDA so a prior latch does not re-ignite
            cc = np.zeros(self.n); cc[np.asarray(list(self.sb.region_manager.indices("fs")), dtype=int)] = 1500.0
            self._run(cc, self.clear_steps, None)            # >tau_NMDA (D3: 2.5*tau erases; else the old bump re-ignites)

        bvec = np.zeros(self.K) if bias is None else bias   # per-pool occupancy excitability (HTM boost; <=0 for used)

        def _base(inh):
            base = np.zeros(self.n)
            eff = self.drive_gain * per_pool_drive + bvec - inh   # drive + occupancy boost - pooled inhibition
            for k in range(self.K):
                base[self.idx[k]] = eff[k]
            return base

        inh = self.inh_high if self.self_calib else self.fixed_inh
        n_active = 0
        for _ in range(self.settle_rounds):
            rates = self._run(_base(inh), self.round_steps, noise_rng)
            mx = rates.max()
            n_active = int((rates > self.active_frac * mx).sum()) if mx > 1e-6 else 0
            if not self.self_calib:
                continue                                    # LESION: a frozen hand-set cut, no ramp -> keep the SAME
                                                            # drive budget as the candidate (fair) but never adapt
            if n_active == 0:
                inh -= self.inh_step                        # release inhibition: let the highest-drive pool escape
            elif n_active > 1:
                inh += 0.5 * self.inh_step                  # too many escaped -> tighten back toward one winner
            else:
                break                                       # exactly one winner found
        # consolidate the winner at the found cut (build up slow-NMDA) then HOLD with the drive removed
        self._run(_base(inh), self.consolidate_steps, noise_rng)
        hold = self._run(np.zeros(self.n), self.hold_steps, noise_rng)   # drive removed: the winner LATCHES (persistent)
        mx = hold.max()
        winner = int(np.argmax(hold)) if mx > 1e-6 else -1
        selectivity = float(mx / (hold.sum() + 1e-9)) if hold.sum() > 0 else 0.0
        return winner, selectivity, float(mx), n_active, float(inh)

    # ---- ALLOCATE a NEW entity to a fresh slot (emergent) ----
    def allocate(self, barcode, noise_rng, clear=True):
        d = self._drive(barcode)
        w, sel, alive, na, inh = self._compete_and_latch(d, noise_rng, clear_first=clear, bias=self._boost())
        if w >= 0:
            self.used[w] += 1.0                             # homeostatic occupancy update (this pool just won)
            b = barcode / (np.linalg.norm(barcode) + 1e-9)
            self.W[w] += b                                  # one-shot Hebbian bind (for retrieve)
        return w, sel, alive, na, inh

    # ---- RETRIEVE a bound entity: drive the pools with the Hebbian-potentiated projection; the bound pool wins ----
    def retrieve(self, barcode, noise_rng):
        if not self.entity_cond:                            # NULL: no entity conditioning -> drive uniform (ignore W)
            w, sel, alive, na, inh = self._compete_and_latch(np.full(self.K, 0.5), noise_rng)
            return w, sel, alive
        b = barcode / (np.linalg.norm(barcode) + 1e-9)
        match = self.W @ b
        if float(match.max()) > self.retrieve_theta:        # familiar: the potentiated weight drives the bound pool
            d = np.clip(match / (match.max() + 1e-9), 0.0, 1.0)
            w, sel, alive, na, inh = self._compete_and_latch(d, noise_rng)
            return w, sel, alive
        return self.allocate(barcode, noise_rng)[:3]        # novel: allocate a fresh slot


# ====================================================================================================================
# THE HOST FREE-COUNTER + TEACHING-CLAMP REFERENCE (the banked mechanism -- the ceiling to match)
#   `free_counter` assigns each NEW entity the next free slot (host); `write(slot)` HOST-drives exactly that pool (the
#   teaching-clamp) -> latch -> read from spikes -> deref. collision = 0 and entity_specific = 1.0 by construction.
# ====================================================================================================================
class HostFreeCounterReference:
    def __init__(self, seed, K, recur=25.0, drive_gain=400.0, noise_pA=60.0, load_steps=30, hold_steps=18):
        from sim.backend import to_host, from_host
        self._to_host, self._from_host = to_host, from_host
        self.seed, self.K = int(seed), int(K)
        self.sb = build_persistent_slot(seed, self.K, recur=recur)
        self.idx = _pool_idx(self.sb, self.K)
        self.pool_neurons = np.concatenate([self.idx[k] for k in range(self.K)])
        self.n = self.sb.core_config.num_neurons
        self.drive_gain, self.noise_pA = float(drive_gain), float(noise_pA)
        self.load_steps, self.hold_steps = int(load_steps), int(hold_steps)
        self.free = 0; self.slot_of_ent = {}               # THE HOST FREE-COUNTER (what this de-risk replaces)

    def reset_episode(self):
        _reset(self.sb); self.free = 0; self.slot_of_ent = {}

    def _run(self, base_cur, steps, noise_rng):
        rates = np.zeros(self.K)
        for _ in range(steps):
            cur = base_cur.copy()
            if self.noise_pA > 0.0 and noise_rng is not None:
                cur[self.pool_neurons] += noise_rng.normal(0.0, self.noise_pA, self.pool_neurons.shape[0])
            self.sb.cp_external_input_current[:] = self._from_host(cur)
            self.sb._run_one_simulation_step()
            fir = np.asarray(self._to_host(self.sb.cp_firing_states)).astype(float)
            for k in range(self.K):
                rates[k] += fir[self.idx[k]].mean()
        return rates / max(steps, 1)

    def _clamp_latch(self, slot, noise_rng):
        cc = np.zeros(self.n); cc[np.asarray(list(self.sb.region_manager.indices("fs")), dtype=int)] = 1500.0
        self._run(cc, 250, None)                             # >tau_NMDA erase (parity with the emergent clear)
        base = np.zeros(self.n); base[self.idx[slot]] = self.drive_gain       # HOST teaching-clamp drives exactly `slot`
        self._run(base, self.load_steps, noise_rng)
        hold = self._run(np.zeros(self.n), self.hold_steps, noise_rng)
        mx = hold.max(); w = int(np.argmax(hold)) if mx > 1e-6 else -1
        sel = float(mx / (hold.sum() + 1e-9)) if hold.sum() > 0 else 0.0
        return w, sel, float(mx)

    def allocate(self, ent_id, noise_rng):
        if ent_id not in self.slot_of_ent:                 # HOST free-counter: next free slot
            self.slot_of_ent[ent_id] = min(self.free, self.K - 1); self.free = min(self.free + 1, self.K)
        slot = self.slot_of_ent[ent_id]
        w, sel, alive = self._clamp_latch(slot, noise_rng)
        return w, sel, alive

    def retrieve(self, ent_id, noise_rng):
        slot = self.slot_of_ent.get(ent_id, 0)
        return self._clamp_latch(slot, noise_rng)


# ====================================================================================================================
# METRICS
# ====================================================================================================================
def sequential_allocation(alloc, codes, order, noise_seed):
    """One sequential pass of the entity codebook (in presentation `order`) into an EMPTY bank. Fresh-slot allocation is
    inherently occupancy-based; the ENTITY-SPECIFIC content is (i) distinct entities -> distinct slots (collision) and
    (ii) a stable retrievable address per entity. Returns (winners_in_order, sels, collision, slot_of_ent, ent_of_slot).
    Collision = fraction of presentations landing on an already-claimed slot (RUNG6e's distinct=False failure)."""
    alloc.reset_episode()
    winners, sels = [], []; claimed = set(); collisions = 0; slot_of_ent = {}; ent_of_slot = {}
    nr = np.random.default_rng(noise_seed)
    for e in order:
        w, sel, _al, _na, _inh = alloc.allocate(codes[e], nr)
        winners.append(w); sels.append(sel)
        if w in claimed:
            collisions += 1
        claimed.add(w); slot_of_ent[e] = w; ent_of_slot[w] = e   # last-writer-wins (a collision overwrites the deref)
    return winners, sels, collisions / max(1, len(order)), slot_of_ent, ent_of_slot


def retrieve_consistency(alloc, codes, slot_of_ent, order, T, base_noise_seed):
    """SAME-ENTITY -> SAME SLOT on re-presentation: after allocation, re-present each entity T times (different noise)
    via retrieve; fraction of retrievals that return the entity's ALLOCATED slot. The direct RUNG6e noise-picked test at
    re-presentation (noise-picked -> the address is not stable -> low). Reuses the post-allocation allocator (its Hebbian
    fast weights + latched map)."""
    ok = 0; tot = 0
    for e in order:
        for t in range(T):
            nr = np.random.default_rng(base_noise_seed + 1000 * e + t)
            w, _sel, _al = alloc.retrieve(codes[e], nr)
            ok += int(w == slot_of_ent.get(e, -999)); tot += 1
    return ok / max(1, tot)


def host_retrieve_consistency(ref, order, T, base_noise_seed):
    ok = 0; tot = 0
    for e in order:
        for t in range(T):
            nr = np.random.default_rng(base_noise_seed + 1000 * e + t)
            w, _sel, _al = ref.retrieve(e, nr)
            ok += int(w == ref.slot_of_ent.get(e, -999)); tot += 1
    return ok / max(1, tot)


def recall_emergent(alloc, codes, ent_of_slot, verb_of, order, noise_seed):
    """Downstream: re-present each entity (retrieve) -> spiking WTA latch -> read slot from SPIKES -> deref -> verb."""
    ok = 0; nr = np.random.default_rng(noise_seed + 777)
    for e in order:
        w, _sel, _al = alloc.retrieve(codes[e], nr)
        pred_ent = ent_of_slot.get(w, -1)
        ok += int(verb_of.get(pred_ent, -2) == verb_of.get(e, -1))
    return ok / max(1, len(order))


def recall_host(ref, ent_of_slot, verb_of, order, noise_seed):
    ok = 0; nr = np.random.default_rng(noise_seed + 777)
    for e in order:
        w, _sel, _al = ref.retrieve(e, nr)
        pred_ent = ent_of_slot.get(w, -1)
        ok += int(verb_of.get(pred_ent, -2) == verb_of.get(e, -1))
    return ok / max(1, len(order))


def alloc_noise_stability(alloc, codes, order, seedA, seedB):
    """Run the SAME-ORDER sequential allocation twice with DIFFERENT noise; fraction of entities that land on the SAME
    slot. Noise-robust allocation (candidate) -> ~1.0; NOISE-PICKED allocation (the lesion / RUNG6e) -> low. The direct
    demonstration that the winner is not noise-determined."""
    _wa, _sa, _ca, mapA, _ea = sequential_allocation(alloc, codes, order, seedA)
    _wb, _sb, _cb, mapB, _eb = sequential_allocation(alloc, codes, order, seedB)
    same = sum(1 for e in order if mapA.get(e) == mapB.get(e))
    return same / max(1, len(order))


# ====================================================================================================================
# One seed
# ====================================================================================================================
def run_point(seed, K, N, T, drive_gain, noise_pA, boost_beta, fixed_inh_frac):
    # held-out NOVEL entities minted at TEST (the allocator has NO per-entity params -> novel by construction; minted
    # from a disjoint stream to make the held-out-novel discipline explicit)
    codes = _mint_codes(np.random.default_rng(seed + 900), N)     # N distinct sparse barcodes
    verb_of = {e: e for e in range(N)}                            # each entity agrees with a distinct verb; chance 1/N
    chance_slot = 1.0 / K
    order = list(range(N))                                        # canonical presentation order
    sA, sB = seed + 31, seed + 61                                 # two noise realisations (allocation noise-stability)

    common = dict(K=K, drive_gain=drive_gain, noise_pA=noise_pA, boost_beta=boost_beta, fixed_inh_frac=fixed_inh_frac)
    # build each arm's allocator ONCE (the bridge build is the expensive part); reset_episode reuses it
    arms = {"emergent": EmergentSlotAllocator(seed, self_calib=True, entity_cond=True, **common),
            "lesion": EmergentSlotAllocator(seed, self_calib=False, entity_cond=True, **common),
            "null": EmergentSlotAllocator(seed, self_calib=True, entity_cond=False, **common)}
    out = {"seed": seed, "K": K, "N": N, "chance_slot": chance_slot, "verb_chance": 1.0 / N}

    for tag, alloc in arms.items():
        # (1) sequential allocation (canonical, noise A): collision + selectivity + the emergent entity<->slot map + W
        winners, sels, coll, soe, eos = sequential_allocation(alloc, codes, order, sA)
        # (2) same-entity -> same slot on RE-PRESENTATION (retrieve, T noise reps); uses the post-alloc W + map
        retr = retrieve_consistency(alloc, codes, soe, order, T, seed + 5)
        # (3) downstream recall (retrieve -> spikes -> deref -> verb)
        rec = recall_emergent(alloc, codes, eos, verb_of, order, sA)
        # (4) allocation noise-stability (2 same-order runs, different noise) -- RESETS W (do LAST for this arm)
        nstab = alloc_noise_stability(alloc, codes, order, sA, sB)
        out.update({f"{tag}_collision": coll, f"{tag}_selectivity": float(np.mean(sels)),
                    f"{tag}_retrieve": retr, f"{tag}_recall": rec, f"{tag}_noise_stability": nstab,
                    f"{tag}_winners": winners, f"{tag}_n_distinct": len(set(winners))})

    # ---- host free-counter + teaching-clamp reference (the ceiling) ----
    ref = HostFreeCounterReference(seed, K, drive_gain=drive_gain, noise_pA=noise_pA)
    win_h, sel_h, coll_h, soe_h, eos_h = _host_sequential(ref, order, sA)
    out.update({"host_collision": coll_h, "host_selectivity": float(np.mean(sel_h)),
                "host_winners": win_h, "host_n_distinct": len(set(win_h))})
    out["host_retrieve"] = host_retrieve_consistency(ref, order, T, seed + 5)
    # host_recall + host noise_stability RESET ref -> reinstate the map first, and run noise_stability LAST
    _host_sequential(ref, order, sA)                              # reinstate ref.slot_of_ent (retrieve_consistency mutated it)
    out["host_recall"] = recall_host(ref, eos_h, verb_of, order, sA)
    out["host_retrieve"] = out["host_retrieve"]                   # (kept)
    # host noise-stability (deterministic slot choice -> ~1.0), measured honestly; resets ref -> last
    _wa, _sa, _ca, mA, _ea = _host_sequential(ref, order, sA)
    _wb, _sb, _cb, mB, _eb = _host_sequential(ref, order, sB)
    out["host_noise_stability"] = sum(1 for e in order if mA.get(e) == mB.get(e)) / max(1, len(order))
    return out


def _host_sequential(ref, order, noise_seed):
    ref.reset_episode()
    winners, sels = [], []; claimed = set(); coll = 0; nr = np.random.default_rng(noise_seed)
    for e in order:
        w, sel, _al = ref.allocate(e, nr); winners.append(w); sels.append(sel)
        if w in claimed:
            coll += 1
        claimed.add(w)
    eos = {w: e for e, w in ref.slot_of_ent.items()}
    return winners, sels, coll / max(1, len(order)), ref.slot_of_ent.copy(), eos


def _agg(per, key):
    return float(np.mean([p[key] for p in per]))


def _aggmin(per, key):
    return float(np.min([p[key] for p in per]))


def agg(per):
    arms = ("emergent", "lesion", "null", "host")
    metrics = ("collision", "selectivity", "retrieve", "recall", "noise_stability")
    a = {f"{arm}_{m}": _agg(per, f"{arm}_{m}") for arm in arms for m in metrics}
    a["emergent_retrieve_min"] = _aggmin(per, "emergent_retrieve")
    a["emergent_collision_max"] = float(np.max([p["emergent_collision"] for p in per]))
    a["emergent_recall_min"] = _aggmin(per, "emergent_recall")
    a["emergent_selectivity_min"] = _aggmin(per, "emergent_selectivity")
    a["emergent_noise_stability_min"] = _aggmin(per, "emergent_noise_stability")
    a.update({"K": per[0]["K"], "N": per[0]["N"], "chance_slot": per[0]["chance_slot"],
              "verb_chance": per[0]["verb_chance"], "per_seed": per})
    return a


def build_verdict(a, smoke):
    chance = a["chance_slot"]; vchance = a["verb_chance"]
    em_re, em_col, em_rec = a["emergent_retrieve"], a["emergent_collision"], a["emergent_recall"]
    le_rec, nu_rec = a["lesion_recall"], a["null_recall"]
    host_rec, host_re = a["host_recall"], a["host_retrieve"]

    # same-entity -> same slot (retrieve) is meaningful only when the allocation is DISTINCT (collision low): under a
    # collapse-to-one-pool, retrieve is trivially 1.0. So entity-specificity REQUIRES retrieve high AND collision low.
    entity_specific_ok = em_re >= 0.90 and a["emergent_retrieve_min"] >= 0.75
    collision_ok = em_col <= 0.10 and a["emergent_collision_max"] <= 0.20
    recall_ok = em_rec >= host_rec - 0.05 and em_rec >= 0.85
    selectivity_ok = a["emergent_selectivity"] >= 0.60
    # self-calibration LOAD-BEARING: removing it (fixed hand-set threshold + no occupancy) COLLAPSES the downstream
    # outcome -- recall drops >= 0.30 AND the allocation degrades (collision up or selectivity down: the RUNG6e blur)
    lesion_bearing = ((em_rec - le_rec) >= 0.30) and (a["lesion_collision"] >= em_col + 0.20
                                                      or a["lesion_selectivity"] <= a["emergent_selectivity"] - 0.20)
    # null (no entity-conditioning) collapses: with the competition ON but the barcode REMOVED the allocation carries no
    # entity information (the pools fill by intrinsic excitability, not identity) -> downstream recall falls to verb-chance
    # (the slot no longer names the entity). This is the load-bearing proof that the barcode-conditioning is what makes
    # the allocation ENTITY-specific (not the competition alone).
    null_at_chance = nu_rec <= vchance + 0.15 and (a["emergent_recall"] - nu_rec) >= 0.30
    core = bool(entity_specific_ok and collision_ok and recall_ok and lesion_bearing and null_at_chance
                and selectivity_ok)
    go = bool(core and not smoke)

    common = (f"emergent same-entity->same-slot(retrieve) {em_re:.3f}[min {a['emergent_retrieve_min']:.3f}] "
              f"(host {host_re:.3f}) | collision {em_col:.3f}[max {a['emergent_collision_max']:.3f}] "
              f"(host {a['host_collision']:.3f}) | recall {em_rec:.3f} (host {host_rec:.3f}) | selectivity "
              f"{a['emergent_selectivity']:.3f} (read from SPIKES; host {a['host_selectivity']:.3f}) | alloc-noise-stability "
              f"{a['emergent_noise_stability']:.3f} || LESION-self-calib recall {le_rec:.3f} collision "
              f"{a['lesion_collision']:.3f} selectivity {a['lesion_selectivity']:.3f} (bearing={lesion_bearing}) | "
              f"NULL(no entity-cond) recall {nu_rec:.3f} collision {a['null_collision']:.3f} (verb-chance {vchance:.3f}, "
              f"collapsed={null_at_chance}). NO host free-counter (occupancy = a homeostatic excitability trace; the "
              f"winner is the pool the spiking attractor LATCHED, read from spikes).")
    smoketag = "" if not smoke else " (1-seed SMOKE indicator; run the 6-seed sweep)"

    if core:
        tag = "GO" if go else "SMOKE-GO"
        return (f"EMERGENT-WTA-ALLOCATION {tag}{smoketag} -- an emergent neural WTA with a SELF-CALIBRATING competition "
                f"threshold (a down-ramp release-of-inhibition -> one clean winner + HTM occupancy boost) replaces the "
                f"HOST free-counter + teaching-clamp: a NEW entity's barcode claims a FRESH slot ENTITY-SPECIFICALLY "
                f"(same entity -> same slot on re-presentation {em_re:.3f}, distinct entities -> distinct slots: collision "
                f"{em_col:.3f}), matching the host reference on downstream recall ({em_rec:.3f} vs {host_rec:.3f}). The "
                f"self-calibration is LOAD-BEARING (fixed hand-set threshold re-collapses: recall {le_rec:.3f}, collision "
                f"{a['lesion_collision']:.3f}, selectivity {a['lesion_selectivity']:.3f}), the null (no "
                f"entity-conditioning) collapses to verb-chance (recall {nu_rec:.3f}). {common} CAVEAT: the "
                f"barcode->pool projection + the release-of-inhibition/boost controller are HOST math on the substrate's "
                f"own pools (the SAME accepted scope as the lever-3 stabilizer; the WTA SELECTION + read are "
                f"neural/spikes); the on-substrate spiking DA-gated / lateral-inhibitory realisation is the named next "
                f"rung. Reuse-by-import; NO sim/ edit."), go, core

    # honest negative -- isolate precisely what the self-calibration could / couldn't fix
    miss = []
    if not entity_specific_ok:
        miss.append(f"same-entity->same-slot(retrieve) {em_re:.3f}[min {a['emergent_retrieve_min']:.3f}] not >=0.90")
    if not collision_ok:
        miss.append(f"collision {em_col:.3f}[max {a['emergent_collision_max']:.3f}] not <=0.10")
    if not recall_ok:
        miss.append(f"recall {em_rec:.3f} not within 0.05 of host {host_rec:.3f} / >=0.85")
    if not selectivity_ok:
        miss.append(f"selectivity {a['emergent_selectivity']:.3f} not >=0.60 (winner still a blur, not clean one-of-K)")
    if not lesion_bearing:
        miss.append(f"self-calibration NOT load-bearing (lesion recall {le_rec:.3f} / collision {a['lesion_collision']:.3f} "
                    f"/ selectivity {a['lesion_selectivity']:.3f} did not re-collapse) -> the outcome is coming from the "
                    f"projection alone, not the competition")
    if not null_at_chance:
        miss.append(f"null recall {nu_rec:.3f} / collision {a['null_collision']:.3f} not collapsed (verb-chance "
                    f"{vchance:.3f})")
    return (f"EMERGENT-WTA-ALLOCATION HONEST-NEGATIVE (first-class){smoketag} -- did not clear the 6-seed bar: "
            + "; ".join(miss) + f". {common} PRECISELY: if selectivity stayed low the self-calibrating competition did "
            f"NOT resolve RUNG6e's blur into a clean high-rate winner (raise settle-rounds / lower inh-step, or the "
            f"on-substrate lateral-inhibition WTA is needed); if selectivity is high but retrieve/noise-stability is low "
            f"the winner is CLEAN but noise-picked (the projection differential is below noise -> a stronger barcode->pool "
            f"code); if collision is the miss the occupancy boost is too weak (raise boost_beta). NO sim/ edit."), False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--K", type=int, default=8, help="# attractor pools (slots) in the bank; chance = 1/K")
    ap.add_argument("--N", type=int, default=6, help="# entities to allocate (<=K); each agrees with a distinct verb")
    ap.add_argument("--T", type=int, default=8, help="# noise realisations for the retrieve-consistency test")
    ap.add_argument("--drive-gain", type=float, default=400.0)
    ap.add_argument("--noise-pA", type=float, default=60.0)
    ap.add_argument("--boost-beta", type=float, default=380.0, help="HTM occupancy-excitability depression per win (pA)")
    ap.add_argument("--fixed-inh-frac", type=float, default=0.45,
                    help="the HAND-SET inhibition (fraction of drive_gain) the lesion is frozen at")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge-from", nargs="+", default=None, help="MERGE mode: aggregate per-seed artifacts (each a "
                    "single-seed run of THIS runner) through the SAME verdict code -> the 6 seeds run in PARALLEL.")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.merge_from:
        per = []
        for pth in a.merge_from:
            d = json.loads(Path(pth).read_text())
            per.extend(d.get("point", {}).get("per_seed", []))
        seeds = sorted(set(p["seed"] for p in per))
        a.seeds = seeds
        smoke = len(seeds) < 6
        backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
        point = agg(per)
        t0 = time.time(); err = None
    else:
        smoke = a.smoke or len(a.seeds) < 6
        backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
        print(f"backend={backend} device={device} | K={a.K} N={a.N} T={a.T} chance_slot={1.0/a.K:.3f} | "
              f"drive_gain={a.drive_gain} noise={a.noise_pA} boost_beta={a.boost_beta} fixed_inh_frac={a.fixed_inh_frac} "
              f"| seeds={a.seeds} smoke={smoke}", flush=True)
        t0 = time.time(); err = None; per = []
        try:
            for s in a.seeds:
                p = run_point(s, a.K, a.N, a.T, a.drive_gain, a.noise_pA, a.boost_beta, a.fixed_inh_frac)
                per.append(p)
                print(f"  [seed {s}] retrieve(same-entity->same-slot): em {p['emergent_retrieve']:.3f} le "
                      f"{p['lesion_retrieve']:.3f} null {p['null_retrieve']:.3f} host {p['host_retrieve']:.3f} | "
                      f"collision: em {p['emergent_collision']:.3f} le {p['lesion_collision']:.3f} null "
                      f"{p['null_collision']:.3f} host {p['host_collision']:.3f} | selectivity: em "
                      f"{p['emergent_selectivity']:.3f} le {p['lesion_selectivity']:.3f} | noise-stab: em "
                      f"{p['emergent_noise_stability']:.3f} le {p['lesion_noise_stability']:.3f} | recall: em "
                      f"{p['emergent_recall']:.3f} host {p['host_recall']:.3f} | winners em {p['emergent_winners']} "
                      f"(distinct {p['emergent_n_distinct']}) le {p['lesion_winners']}", flush=True)
        except Exception as e:
            err = repr(e); traceback.print_exc()
        point = agg(per) if per else None

    verdict = None; go = False; core = False
    if point is not None:
        verdict, go, core = build_verdict(point, smoke)
        print(f"\n[emergent-wta-alloc] {verdict}", flush=True)
    elif err is not None:
        verdict = f"ERROR -- {err}"

    # ---- earned verdict preconditions (validity travels with the verdict) ----
    preconditions = []
    try:
        from tools.verdict import Verdict
        if point is not None:
            chance = point["chance_slot"]
            Vd = Verdict("var_bind_emergent_wta_allocation", chance=chance)
            Vd.require("host_reference_retrieve_stable", round(point["host_retrieve"], 4),
                       expect=lambda x: x >= 0.90,
                       note="the host free-counter reference retrieves a stable address (~1.0) -> a real ceiling")
            Vd.require("host_reference_recall_above_chance", round(point["host_recall"], 4),
                       expect=lambda x: x >= point["verb_chance"] + 0.30,
                       note="the host teaching-clamp downstream recall clears verb-chance -> the recall target exists")
            Vd.require("null_recall_at_verb_chance", round(point["null_recall"], 4),
                       expect=lambda x: x <= point["verb_chance"] + 0.15,
                       note="the no-entity-conditioning null carries no identity -> downstream recall falls to verb-chance")
            Vd.control("selfcalib_differs_from_lesion_recall", treatment=point["emergent_recall"],
                       control=point["lesion_recall"], min_separation=0.05,
                       note="validity: the self-calibrated arm's recall must differ from the fixed-threshold lesion's "
                            "(retrieve alone is degenerate: it recovers the bound slot even under a collided allocation)")
            dec = Vd.decide(go, verbose=False)
            preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "var_bind_emergent_wta_allocation", "verdict": verdict, "go": bool(go), "core": bool(core),
               "backend": backend, "sim_backend": backend, "device": device, "smoke": smoke, "cost_acknowledged": True,
               "preconditions": preconditions,
               "mechanism": "one bank of K D3 slow-NMDA persistent-activity attractor pools sharing ONE FS "
                            "(build_persistent_slot, the multi-slot substrate R=1); a NEW entity's barcode drives ALL "
                            "pools through a fixed developmental-random projection P (K x code_dim) -> graded "
                            "entity-specific external current; a SELF-CALIBRATING competition threshold (adaptive "
                            "subtractive pooled feedback inhibition, homeostatic to ONE active pool -- Carandini-Heeger "
                            "normalisation / the lever-3 feedback-inhibition motif) resolves the blur into a clean "
                            "high-rate winner; an adaptive HTM/Turrigiano occupancy excitability (recently-won pools "
                            "depressed) steers the next entity to a FREE pool; a content-agnostic Hebbian fast weight "
                            "binds the winner for retrieve. The winner is READ FROM SPIKES (the latched pool, drive "
                            "removed), never a host argmax over the drive; the controller sets only a SCALAR inhibition "
                            "from the population active-count, never a pool. NO host free-counter, NO host teaching-clamp.",
               "task": "fresh-slot allocation of N held-out NOVEL entities into a K-pool bank; arms: emergent_wta / "
                       "host_free_counter (ceiling) / lesion_selfcalib (fixed hand-set threshold -> noise-picked) / "
                       "noise_picked_null (no entity-conditioning -> chance); metrics: entity_specific (same-entity -> "
                       "same slot, the RUNG6e noise-picked test), collision (distinct -> distinct), downstream recall "
                       "(retrieve -> spikes -> deref -> verb), selectivity (clean one-of-K from spikes)",
               "seeds": a.seeds, "config": {"K": a.K, "N": a.N, "T": a.T, "drive_gain": a.drive_gain,
               "noise_pA": a.noise_pA, "boost_beta": a.boost_beta, "fixed_inh_frac": a.fixed_inh_frac,
               "chance_slot": (point["chance_slot"] if point else 1.0 / a.K)},
               "point": point, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "the barcode->pool projection + the feedback-inhibition/occupancy controller are HOST math "
                              "on the substrate's OWN pools (the SAME accepted scope as the banked lever-3 competitive "
                              "stabilizer and the multi-slot write()): the WTA SELECTION and the read are neural (spikes), "
                              "the controller sets only a scalar inhibition from a population active-COUNT (never a pool), "
                              "and occupancy is a homeostatic excitability trace (not a host free-counter). The "
                              "on-substrate spiking lateral-inhibitory / DA-gated realisation is the named next rung "
                              "(RUNG6e's hard region-framework WTA engineering). 1-seed is a SMOKE indicator; the 6-seed "
                              "sweep is decisive. Honest-negative (the emergent winner staying noise-picked) is a "
                              "first-class deliverable that isolates what the self-calibrating competition could/couldn't "
                              "fix. NO sim/ edit."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[emergent-wta-alloc] VERDICT: {verdict}", flush=True)
    print(f"[emergent-wta-alloc] go={go}  wrote {a.out}\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
