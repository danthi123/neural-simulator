"""ONE-BRAIN MERGE — ORGAN-READ verify: the follow-on rung to `_onebrain_twopool_merge_derisk.py` (Vikunja #171).

THE GAP THE PRIOR RUNG LEFT OPEN. `_onebrain_twopool_merge_derisk.py` (6/6 GO,
`2026-08-27-onebrain-twopool-merge-substrate-byte-identity-6seed-GO.md`) proved SUBSTRATE-INIT byte-identity only:
every per-neuron init array (thresholds, v, u, the 8 Izhikevich params, the 2 gate masks) is byte-identical for the
D2 surprise + E2 world-model + E1 metacog + D pragmatic organs merged onto ONE `SimulationBridge` vs each organ
co-resident alone on the same superset config. It explicitly did NOT run the organ READ pipelines
(`SurpriseProductionOrgan.judge`, `WorldModelProductionOrgan.expectation`, `MetacogProductionOrgan.judge`,
`PragmaticProductionOrgan.interpret`) nor the post-build topographic wiring (surprise's block-diagonal, metacog's
self-recurrent assembly loops) that those reads depend on.

THIS RUNNER closes that gap: it builds the FULL 4-organ merged substrate (wiring included), runs each organ's REAL
production read/judge pipeline on it, and compares every read + a battery of rendered chat-answer strings against
running the SAME 4 organs on TODAY'S PRODUCTION TWO POOLS — `onebrain_merge_production.MergedSubstrate` (pool #1:
surprise + world-model, DEFAULT-ON `BRAIN_ONEBRAIN_MERGE`) and `onebrain_merge_production2.MergedSubstrate2` (pool
#2: metacog + pragmatic, DEFAULT-ON `BRAIN_ONEBRAIN_MERGE2`) — using the SHIPPED, UNMODIFIED organ classes for both
arms (only the injected `shared=` substrate differs).

THE WIRING THIS RUNNER ADDS (post substrate-build, before any organ trains/reads):
  1. `rm.build_wiring_plan(seed, per_region_seed=True)` (the framework-declared pathways, order-invariant) UNION
     metacog's K dense self-recurrent assembly loops (`_build_assembly_loop_population`, the GNW Rung-1 primitive
     `build_metacog_bridge` and pool #2's `MergedSubstrate2` both use) -> `bridge.inject_explicit_wiring(...)`
     (this REBUILDS `cp_connections` wholesale, exactly as `build_metacog_bridge`/`build_rsa_bridge`/pool #2 do) ->
     `bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)` (freeze the loop, matching every existing metacog builder).
  2. Surprise's post-build TOPOGRAPHIC masking (`_install_block_diagonal` x3: cue->patient_expected,
     patient_asserted->surprise, patient_expected->surprise), applied AFTER (1) rebuilds the dense edges wholesale
     -- exactly the order `build_expectation_circuit` and pool #1's `MergedSubstrate.ensure_built()` use.

THE NAMED CROSS-REGION INTERACTION (surprise's shared Hebbian vs pool-2's frozen edges), handled by the SAME
primitive the parser-on-pool merge used (`onebrain_merge_production._bind_parser_onto_pool`, CONFLICT A): a
PERMANENT per-synapse `cp_plasticity_rate_gain = 0.0` on every edge with BOTH endpoints inside a metacog/pragmatic
region. The gated potentiate/decay/clip in `sim/bridge.py`'s Hebbian block (`_run_one_simulation_step`, the
rate-window branch) already read this array and leave a gain-0 synapse VERBATIM; gain=1.0 elsewhere is
byte-identical to the ungated scalar path. This protects TWO distinct hazards, not one: (a) surprise/world-model's
shared Hebbian TRAINING (their own `train_expectation`/`train_transition`, which the base runner already showed is
config-global) never touches pool-2's weights; (b) metacog/pragmatic's OWN reads drive REAL evidence-elicited
firing on the shared bridge -- unlike their standalone/pool-#2 builds (`enable_hebbian_learning=False` globally),
the 4-organ superset needs it GLOBALLY ON for surprise, so WITHOUT the freeze every metacog/pragmatic read would
also Hebbian-nudge its OWN edges (workspace<->workspace_fs, the assembly loop, item<->item_fs) -- non-reproducible
drift the shipped organs never exhibit. `gain0_freeze_check` verifies BOTH the STRUCTURAL gain array (0.0 exactly
over pool-2 edges, 1.0 elsewhere) and the OUTCOME (pool-2 edge weights byte-identical before vs after every organ
has trained/read on the shared bridge).

READ ISOLATION, extended from pool #1's 2-organ primitive to 4 organs. `onebrain_merge_production.MergedSubstrate.
read_isolation(active)` (snapshot the full per-neuron state, let `active`'s slice evolve, restore every OTHER
region's slice) already protects surprise vs world-model from each other's spontaneous-firing footprint during a
long stepping burst (homeostatically-silenced FS neurons can fire near-rest). `MergedSubstrate4` extends the SAME
mechanism to name-key on 4 organs; `SurpriseProductionOrgan`/`WorldModelProductionOrgan` invoke it internally
(unmodified). `MetacogProductionOrgan`/`PragmaticProductionOrgan` do NOT call it internally (pool #2 alone never
needed it -- neither organ there carries per-region homeostasis, so there is no threshold-drift hazard within pool
#2), so THIS RUNNER wraps their read batteries in `read_isolation("metacog"/"pragmatic")` externally -- a new
hazard specific to co-residing with surprise/world-model's per-region-homeostasis-ON regions.

SCOPE. A DE-RISK, NOT a production flip: no `sim/` edit, no change to any of the four shipped `..._production_organ.
py` files or to `onebrain_merge_production{,2}.py`. The 4-organ single-pool substrate lives ONLY in this file and
`_onebrain_twopool_merge_derisk.py` (reused, not modified). Flipping production to the single pool is the named
NEXT rung.

Run (CPU, bit-exact):
    SIM_BACKEND=numpy python -m research.runners._onebrain_twopool_organread_verify \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_twopool_organread_6seed.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np

from research.runners._onebrain_twopool_merge_derisk import (
    ALL_ORGANS, SURPRISE_REGIONS, WORLDMODEL_REGIONS, METACOG_REGIONS, PRAGMATIC_REGIONS,
    _SURPRISE_KW, _WORLDMODEL_KW, build_pool, byte_identity, _host, _region_indices,
)
from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit, _install_block_diagonal,
)
from research.runners._affective_world_model_derisk import build_world_model_circuit
from research.runners._second_order_metacog_monitor_derisk import (
    ASSEMBLY_SIZE, K_CLASSES, WS_LOOP_GATE, DEFAULT_ATTRACTOR_WEIGHT,
)
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, SETTLE_STEPS,
)

from research.runners.surprise_production_organ import SurpriseProductionOrgan, surprise_notice
from research.runners.worldmodel_production_organ import WorldModelProductionOrgan, worldmodel_surprise_notice
from research.runners.metacog_production_organ import MetacogProductionOrgan, hedge_prefix
from research.runners.pragmatic_production_organ import PragmaticProductionOrgan, pragmatic_notice

from research.runners.onebrain_merge_production import MergedSubstrate as MergedSubstrate1
from research.runners.onebrain_merge_production2 import MergedSubstrate2

from research.runners._onebrain_merge_rung1_verify import _surprise_reads, _worldmodel_reads, _max_delta

_ALL_REGIONS_BY_ORGAN = {
    "surprise": SURPRISE_REGIONS, "worldmodel": WORLDMODEL_REGIONS,
    "metacog": METACOG_REGIONS, "pragmatic": PRAGMATIC_REGIONS,
}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  MergedSubstrate4 — the 4-organ single pool this rung verifies (build_pool's config superset,
#  PLUS the post-build wiring the base runner declared out of scope, PLUS the pool-2 gain-0 freeze).
# ─────────────────────────────────────────────────────────────────────────────────────────────
class MergedSubstrate4:
    """ONE `SimulationBridge` holding all 4 organs' regions, fully wired + trained-ready, exposing BOTH pool #1's
    `read_isolation` API (surprise/world-model's own code calls it) and pool #2's `.snap` / `.metacog_idx()` /
    `.pragmatic_item_dev()` API (metacog/pragmatic's own code reads these). Built ONCE (lazily)."""

    _PER_NEURON_STATE = (
        "cp_membrane_potential_v", "cp_recovery_variable_u",
        "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
        "cp_firing_states", "cp_prev_firing_states", "cp_refractory_timers", "cp_refractory",
        "cp_neuron_firing_thresholds", "cp_neuron_activity_ema", "cp_external_input_current",
    )

    def __init__(self, seed: int = 42, freeze_pool2: bool = True):
        self.seed = int(seed)
        self.freeze_pool2 = bool(freeze_pool2)   # False only for the discriminator (never in production use)
        self.bridge = self.cfg = self.xp = None
        self.meta_surprise = self.meta_worldmodel = None
        self.snap = None                          # the pool-2-style full quiescent snapshot
        self._pool2_frozen_idx = None              # bool mask over cp_connections nnz (pool-2-internal edges)
        self._keep_mask_cache = {}
        self._built = False

    def ensure_built(self):
        if self._built:
            return
        from sim.backend import get_backend
        xp, _ = get_backend()

        # SPEC EXTRACTION (reuse-by-import, throwaway -- mirrors onebrain_merge_production.MergedSubstrate exactly):
        # pull the meta dicts (blk / W_exc / W_inh / n_states ...) each organ's own read path needs.
        _brS, _cfgS, metaS = build_expectation_circuit(self.seed, per_region_thresh=True, **_SURPRISE_KW)
        _brW, _cfgW, metaW = build_world_model_circuit(self.seed, **_WORLDMODEL_KW)
        self.meta_surprise = metaS
        self.meta_worldmodel = metaW

        bridge, cfg = build_pool(self.seed, ALL_ORGANS)   # the validated per-region-seam superset (6/6 GO)
        rm = bridge.region_manager

        # ── POST-BUILD WIRING (the gap this runner closes: NOT run by the base merge de-risk). ──
        union = dict(rm.build_wiring_plan(seed=int(self.seed), per_region_seed=True))
        ws = _region_indices(bridge, "workspace")
        for k in range(K_CLASSES):
            member = ws[k * ASSEMBLY_SIZE:(k + 1) * ASSEMBLY_SIZE]
            union[f"loop_{k}"] = _build_assembly_loop_population(member, float(DEFAULT_ATTRACTOR_WEIGHT))
        inh = []
        for region in rm.regions():
            inh.extend(rm.inhibitory_indices(region.name))
        bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
        bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

        blk = metaS["blk"]
        _install_block_diagonal(bridge, "patient_asserted", "surprise", blk, metaS["W_exc"])
        _install_block_diagonal(bridge, "patient_expected", "surprise", blk, metaS["W_inh"])
        _install_block_diagonal(bridge, "cue", "patient_expected", blk, float(_SURPRISE_KW["cue_to_expected_weight"]))

        # ── THE NAMED CROSS-REGION INTERACTION: permanent per-synapse gain-0 on every pool-2-internal edge. ──
        pool2_names = set(METACOG_REGIONS) | set(PRAGMATIC_REGIONS)
        pool2_idx = set()
        for name in pool2_names:
            pool2_idx |= set(int(i) for i in _region_indices(bridge, name))
        pool2_idx_arr = np.asarray(sorted(pool2_idx), dtype=np.int64)
        coo = bridge.cp_connections.tocoo()
        row = np.asarray(_host(coo.row)); col = np.asarray(_host(coo.col))
        row_in = np.isin(row, pool2_idx_arr); col_in = np.isin(col, pool2_idx_arr)
        cross = row_in ^ col_in
        if bool(cross.any()):
            raise AssertionError(f"{int(cross.sum())} unexpected cross-organ synapse(s) touch pool-2 regions "
                                 f"(exactly one endpoint in metacog/pragmatic) -- the no-cross-synapse premise broke")
        in_pool2 = row_in & col_in
        self._pool2_frozen_idx = in_pool2
        if self.freeze_pool2:
            nnz = int(bridge.cp_connections.nnz)
            if bridge.cp_plasticity_rate_gain is None:
                bridge.cp_plasticity_rate_gain = xp.ones(nnz, dtype=xp.float32)
            gain_host = np.asarray(_host(bridge.cp_plasticity_rate_gain)).copy()
            gain_host[in_pool2] = 0.0
            bridge.cp_plasticity_rate_gain = xp.asarray(gain_host, dtype=xp.float32)

        # ── resting snapshot (surprise/world-model's own `_hard_reset`). ──
        bridge._rest_v = bridge.cp_membrane_potential_v.copy()
        bridge._rest_u = bridge.cp_recovery_variable_u.copy()

        # ── settle + full quiescent snapshot (metacog/pragmatic's own `_restore_state` protocol). ──
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(SETTLE_STEPS):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
        self.snap = _snapshot_state(bridge, xp)

        self.bridge = bridge
        self.cfg = cfg
        self.xp = xp
        self._built = True

    # ── pool #1 API (surprise / world-model's own code calls these) ──
    def surprise_idx_map(self):
        self.ensure_built()
        return {n: self.xp.asarray(_region_indices(self.bridge, n)) for n in SURPRISE_REGIONS}

    def worldmodel_idx_map(self):
        self.ensure_built()
        return {n: self.xp.asarray(_region_indices(self.bridge, n)) for n in WORLDMODEL_REGIONS}

    # ── pool #2 API (metacog / pragmatic's own code calls these) ──
    def metacog_idx(self):
        self.ensure_built()
        rm = self.bridge.region_manager
        xp = self.xp
        ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
        fs = np.asarray(rm.indices("workspace_fs"), dtype=np.int64)
        meta = np.asarray(rm.indices("meta_schema"), dtype=np.int64)
        member_idx = {k: ws[k * ASSEMBLY_SIZE:(k + 1) * ASSEMBLY_SIZE] for k in range(K_CLASSES)}
        from research.runners._second_order_metacog_monitor_derisk import META_SIZE
        meta_sub = META_SIZE // K_CLASSES
        meta_member_idx = {k: meta[k * meta_sub:(k + 1) * meta_sub] for k in range(K_CLASSES)}
        return {
            "member_dev": {k: xp.asarray(v) for k, v in member_idx.items()},
            "meta_dev": xp.asarray(meta),
            "meta_member_dev": {k: xp.asarray(v) for k, v in meta_member_idx.items()},
            "fs_dev": xp.asarray(fs),
            "confidence_read": "balance",
        }

    def pragmatic_item_dev(self):
        self.ensure_built()
        from research.runners._recursive_tom_rsa_derisk import RSA_ITEM_SIZE
        rm = self.bridge.region_manager
        xp = self.xp
        base = np.asarray(rm.indices("item"), dtype=np.int64)
        return {i: xp.asarray(base[i * RSA_ITEM_SIZE:(i + 1) * RSA_ITEM_SIZE]) for i in range(3)}

    # ── one_pool sanity (mirrors _onebrain_merge_rung1_verify's check, extended to 4 organs) ──
    def one_pool(self) -> bool:
        self.ensure_built()
        n_all = int(self.bridge.cp_membrane_potential_v.shape[0])
        sizes = {r: int(len(_region_indices(self.bridge, r)))
                for regs in _ALL_REGIONS_BY_ORGAN.values() for r in regs}
        maxes = [int(_host(_region_indices(self.bridge, r)).max())
                for regs in _ALL_REGIONS_BY_ORGAN.values() for r in regs]
        return bool(n_all >= sum(sizes.values()) and all(m < n_all for m in maxes))

    # ── read isolation, extended from pool #1's 2-organ primitive to all 4 organs. ──
    def _keep_mask(self, active: str):
        cache = self._keep_mask_cache
        if active not in cache:
            self.ensure_built()
            regions = _ALL_REGIONS_BY_ORGAN[active]
            n = int(self.bridge.cp_membrane_potential_v.shape[0])
            mask = self.xp.zeros(n, dtype=bool)
            for r in regions:
                mask[self.xp.asarray(_region_indices(self.bridge, r))] = True
            cache[active] = mask
        return cache[active]

    @contextlib.contextmanager
    def read_isolation(self, active: str):
        """Identical mechanism to `onebrain_merge_production.MergedSubstrate.read_isolation`, name-keyed over all
        4 organs: snapshot the full per-neuron state, let `active`'s slice evolve, restore every OTHER organ's
        slice at the end -- so a long stepping burst by one organ leaves no footprint on the other three."""
        b = self.bridge
        snaps = []
        for name in self._PER_NEURON_STATE:
            arr = getattr(b, name, None)
            snaps.append(None if arr is None else arr.copy())
        try:
            yield
        finally:
            keep = self._keep_mask(active)
            for name, snap in zip(self._PER_NEURON_STATE, snaps):
                if snap is None:
                    continue
                cur = getattr(b, name)
                setattr(b, name, self.xp.where(keep, cur, snap))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Read batteries (metacog / pragmatic; surprise / world-model reuse rung1_verify's).
# ─────────────────────────────────────────────────────────────────────────────────────────────
_METACOG_BATTERY = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
_PRAGMATIC_BATTERY = ("some", "all", "none")


def _metacog_reads(organ: MetacogProductionOrgan) -> dict:
    organ.ensure_built()
    out = {}
    c = organ.calib
    for k in ("mean_hi", "min_hi", "mean_lo", "max_lo"):
        out[f"calib.{k}"] = float(c[k])
    out["threshold"] = float(organ.threshold)
    for e in _METACOG_BATTERY:
        j = organ.judge(e)
        out[f"judge[{e:.2f}].balance"] = float(j["balance"])
        out[f"judge[{e:.2f}].confident"] = float(bool(j["confident"]))
    return out


def _pragmatic_reads(organ: PragmaticProductionOrgan) -> dict:
    organ.ensure_built()
    out = {}
    for u in _PRAGMATIC_BATTERY:
        r = organ.interpret(u)
        for i, s in enumerate(r["states"]):
            out[f"interpret[{u}].belief[{s}]"] = float(r["belief"][i])
        out[f"interpret[{u}].margin"] = float(r["implicature_margin"])
        out[f"interpret[{u}].represented"] = float(bool(r["implicature_represented"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Per-seed verification.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed: int, verbose: bool = True) -> dict:
    # ── BASELINE: today's production -- TWO separate pools, the shipped, unmodified organ classes. ──
    base1 = MergedSubstrate1(seed=seed, organs=("surprise", "worldmodel"))
    base2 = MergedSubstrate2(seed=seed, organs=("metacog", "pragmatic"))
    b_surprise = SurpriseProductionOrgan(seed=seed, shared=base1)
    b_worldmodel = WorldModelProductionOrgan(seed=seed, shared=base1)
    b_metacog = MetacogProductionOrgan(seed=seed, shared=base2)
    b_pragmatic = PragmaticProductionOrgan(seed=seed, shared=base2)
    b_reads = {
        "surprise": _surprise_reads(b_surprise), "worldmodel": _worldmodel_reads(b_worldmodel),
        "metacog": _metacog_reads(b_metacog), "pragmatic": _pragmatic_reads(b_pragmatic),
    }

    # ── MERGED: all 4 organs on ONE shared bridge, the shipped organ classes, our new substrate injected. ──
    merged4 = MergedSubstrate4(seed=seed)
    m_surprise = SurpriseProductionOrgan(seed=seed, shared=merged4)
    m_worldmodel = WorldModelProductionOrgan(seed=seed, shared=merged4)
    m_metacog = MetacogProductionOrgan(seed=seed, shared=merged4)
    m_pragmatic = PragmaticProductionOrgan(seed=seed, shared=merged4)

    merged4.ensure_built()
    pool2_mask = _host(merged4._pool2_frozen_idx)
    gain_before = np.asarray(_host(merged4.bridge.cp_connections.data)).astype(np.float64).copy()
    weight_before_pool2 = gain_before[pool2_mask].copy()
    gain_arr = np.asarray(_host(merged4.bridge.cp_plasticity_rate_gain))
    gain0_structural_ok = bool(np.all(gain_arr[pool2_mask] == 0.0) and np.all(gain_arr[~pool2_mask] == 1.0))

    m_reads = {
        "surprise": _surprise_reads(m_surprise),        # self-wraps via shared.read_isolation("surprise")
        "worldmodel": _worldmodel_reads(m_worldmodel),  # self-wraps via shared.read_isolation("worldmodel")
    }
    with merged4.read_isolation("metacog"):
        m_reads["metacog"] = _metacog_reads(m_metacog)
    with merged4.read_isolation("pragmatic"):
        m_reads["pragmatic"] = _pragmatic_reads(m_pragmatic)

    weight_after_pool2 = np.asarray(_host(merged4.bridge.cp_connections.data)).astype(np.float64)[pool2_mask]
    gain0_maxerr = float(np.max(np.abs(weight_after_pool2 - weight_before_pool2))) if weight_before_pool2.size else 0.0
    gain0_bit_frozen = bool(gain0_maxerr == 0.0)

    # ── BYTE-IDENTITY of the production reads, per organ. ──
    byte = {}
    for organ_name in ALL_ORGANS:
        d, k, missing = _max_delta(m_reads[organ_name], b_reads[organ_name])
        byte[organ_name] = {"maxdelta": d, "worst_key": k, "missing_keys": missing,
                            "byte_identical": bool(d == 0.0 and not missing)}
    byte_ok = all(byte[o]["byte_identical"] for o in ALL_ORGANS)

    # ── ANSWER PRESERVATION: the rendered chat-answer strings, not just the numeric reads. ──
    b_notice = surprise_notice("agent", "acts", "alpha")
    m_notice = surprise_notice("agent", "acts", "alpha")               # pure function of its args -> trivially equal
    b_j = b_surprise.judge("agent", "acts", "beta", "gamma")
    m_j = m_surprise.judge("agent", "acts", "beta", "gamma")
    surprise_answer_same = bool(b_j["surprised"] == m_j["surprised"])

    b_e = b_worldmodel.expectation(+1); m_e = m_worldmodel.expectation(+1)
    b_wm_notice = worldmodel_surprise_notice(b_e["pred_sign"])
    m_wm_notice = worldmodel_surprise_notice(m_e["pred_sign"])
    worldmodel_answer_same = bool(b_e["pred_sign"] == m_e["pred_sign"] and b_wm_notice == m_wm_notice)

    b_hedge = hedge_prefix(); m_hedge = hedge_prefix()                  # pure constant -> trivially equal
    b_conf = b_metacog.judge(0.2)["confident"]; m_conf = m_metacog.judge(0.2)["confident"]
    metacog_answer_same = bool(b_conf == m_conf and b_hedge == m_hedge)

    b_interp = b_pragmatic.interpret("some"); m_interp = m_pragmatic.interpret("some")
    b_pn = pragmatic_notice(b_interp); m_pn = pragmatic_notice(m_interp)
    pragmatic_answer_same = bool(b_interp["enriched_interpretation"] == m_interp["enriched_interpretation"]
                                 and b_pn == m_pn)

    answers_same = {"surprise": surprise_answer_same, "worldmodel": worldmodel_answer_same,
                    "metacog": metacog_answer_same, "pragmatic": pragmatic_answer_same}
    answers_ok = all(answers_same.values())

    one_pool = merged4.one_pool()
    n_all = int(merged4.bridge.cp_membrane_potential_v.shape[0])

    # ── the base runner's INIT-level legacy discriminator, carried forward (non-vacuousness at the init layer). ──
    leg = byte_identity(seed, legacy=True)
    legacy_diverges = any(leg[o]["maxerr"] > 0.0 for o in ALL_ORGANS)

    go = bool(byte_ok and answers_ok and gain0_structural_ok and gain0_bit_frozen
              and legacy_diverges and one_pool)

    res = {
        "seed": seed, "n_all_neurons": n_all, "one_pool": one_pool,
        "byte_identity": byte, "byte_ok": byte_ok,
        "answers_same": answers_same, "answers_ok": answers_ok,
        "gain0_structural_ok": gain0_structural_ok, "gain0_n_frozen_edges": int(pool2_mask.sum()),
        "gain0_maxerr_after_full_lifecycle": gain0_maxerr, "gain0_bit_frozen": gain0_bit_frozen,
        "legacy_discriminator": {o: leg[o]["maxerr"] for o in ALL_ORGANS}, "legacy_diverges": legacy_diverges,
        "GO": go,
    }
    if verbose:
        print(f"  [seed {seed}] pool={one_pool}(N={n_all}) | "
              f"byte_ok={byte_ok} " + " ".join(f"{o}={byte[o]['maxdelta']:.2e}" for o in ALL_ORGANS) + " | "
              f"answers_ok={answers_ok} | gain0 struct={gain0_structural_ok} "
              f"frozen(n={int(pool2_mask.sum())})={gain0_bit_frozen}(d={gain0_maxerr:.2e}) | "
              f"legacy_diverges={legacy_diverges} -> {'GO' if go else 'NO-GO'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    print("=== ONE-BRAIN MERGE — ORGAN-READ verify: 4 organs' REAL read pipelines on ONE shared bridge ===")
    print("    vs today's production TWO pools (onebrain_merge_production{,2}.py, shipped organ classes unmodified)")
    results = [run_seed(s) for s in seeds]
    n = len(results)
    n_go = sum(r["GO"] for r in results)
    n_byte = sum(r["byte_ok"] for r in results)
    n_answers = sum(r["answers_ok"] for r in results)
    n_gain0 = sum(r["gain0_structural_ok"] and r["gain0_bit_frozen"] for r in results)
    n_legacy = sum(r["legacy_diverges"] for r in results)
    verdict = "GO" if n_go == n and n > 0 else "NO-GO"
    print("\n=== VERDICT ===")
    print(f"  organ-read byte-identity (all 4 organs):  {n_byte}/{n}")
    print(f"  chat-answer preservation (all 4 organs):  {n_answers}/{n}")
    print(f"  gain-0 freeze holds (structural + bit-frozen after full lifecycle): {n_gain0}/{n}")
    print(f"  legacy discriminator diverges (not vacuous): {n_legacy}/{n}")
    print(f"  FULL ORGAN-READ MERGE: {n_go}/{n}  ->  {verdict}")
    from tools.verdict import Verdict
    v = Verdict("one-brain two-pool ORGAN-READ byte-identity (4 organs, merged vs two production pools)")
    v.require("all 4 organs' reads byte-identical vs the two pools, every seed", n_byte, expect=n)
    v.require("chat-answers preserved for all 4 organs, every seed", n_answers, expect=n)
    v.require("gain-0 freeze holds pool-2 bit-frozen after the full lifecycle, every seed", n_gain0, expect=n)
    v.require("legacy discriminator diverges (byte-identity not vacuous), every seed", n_legacy, expect=n)
    decided = v.decide(go=(n_go == n and n > 0))
    payload = {"mode": "onebrain_twopool_organread_verify", "organs": list(ALL_ORGANS),
               "n_seeds": n, "results": results, "n_go": n_go, "n_byte_ok": n_byte,
               "n_answers_ok": n_answers, "n_gain0_ok": n_gain0, "n_legacy_diverges": n_legacy,
               "verdict": verdict}
    payload.update(decided)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
