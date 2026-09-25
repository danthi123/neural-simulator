"""Subprocess-isolated scenario for tests/test_chat_time_plasticity_local_freeze.py.

Mirrors tests/_plastic_mask_stdp_scenario.py's own reasoning for running in its OWN process: `sim/bridge.py`
resolves numpy-vs-cupy ONCE at module import, so whichever backend a sibling test file happened to force earlier
in the same pytest session would otherwise leak here. `SIM_BACKEND=numpy` is set (by the test, in this
subprocess's env) before `sim.bridge` is ever imported in this process.

WHAT THIS MEASURES (research/findings/2026-09-24-chat-time-plasticity-audit-*.md): `WorldModelProductionOrgan`
and `SurpriseProductionOrgan` used to freeze their OWN trained pathway by setting `cfg.enable_hebbian_learning =
False` on the SHARED wave3-pool cfg object -- a bridge-WIDE kill switch that silently freezes every co-resident
organ's Hebbian pathway too, not just the caller's own. `BRAIN_WORLDMODEL_LOCAL_FREEZE=1` / `BRAIN_SURPRISE_
LOCAL_FREEZE=1` (both default-OFF) replace that with a NAMED per-pathway `set_plasticity_gate(..., 0.0)` on each
organ's own trained pathway, leaving `cfg.enable_hebbian_learning` process-wide TRUE.

This scenario builds a 3-organ merged pool (surprise + world-model + a synthetic UNGATED "probe" pathway
standing in for "some other faculty's chat-time-plastic pathway") under `--mode off` (today's shipped default)
and `--mode on` (both new flags set), trains surprise+world-model exactly as `webapp/server.py`'s warmup does,
then drives real co-activity on the probe pathway and re-reads all three pathways' weights. Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _to_host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _probe_spec(seed):
    """A minimal 2-region, ONE-pathway organ standing in for "some other faculty's chat-time-plastic pathway":
    plain excitatory populations, one UNGATED `plastic=True` pathway, no plasticity_gate at all -- exactly the
    shape curiosity/self_schema/pragmatic/etc.'s pathways would take IF one of them ever wired an ongoing
    chat-time Hebbian pathway onto the shared pool (none does today -- see the finding's per-faculty table)."""
    from sim.regions import BrainRegion, RegionPathway
    regions = [
        BrainRegion(name="probe_a", n_neurons=12, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type="IZH2007_RS_CORTICAL_PYRAMIDAL"),
        BrainRegion(name="probe_b", n_neurons=12, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type="IZH2007_RS_CORTICAL_PYRAMIDAL"),
    ]
    pathways = [
        RegionPathway(from_region="probe_a", to_region="probe_b", density=1.0,
                      weight_mean=0.5, weight_jitter=0.0, plastic=True),   # UNGATED, unlike surprise/worldmodel
    ]
    return regions, pathways, {}


def _pathway_dw(bridge, w_before, src, dst):
    """max|dw| over every synapse from region `src` to region `dst` (host-side, robust to CSR row/col
    orientation -- checks both directions the way `_install_block_diagonal` does)."""
    import numpy as np
    src_idx = set(int(i) for i in bridge.region_manager.indices(src))
    dst_idx = set(int(i) for i in bridge.region_manager.indices(dst))
    coo = bridge.cp_connections.tocoo()
    row = np.asarray(_to_host(coo.row)); col = np.asarray(_to_host(coo.col))
    data_now = np.asarray(_to_host(bridge.cp_connections.data))[:bridge.cp_connections.nnz]
    mask = np.array([(r in src_idx and c in dst_idx) or (r in dst_idx and c in src_idx)
                     for r, c in zip(row.tolist(), col.tolist())])
    if not mask.any():
        return None, 0
    dw = np.abs(data_now[mask] - w_before[mask])
    return float(dw.max()), int(mask.sum())


def run(mode: str, seed: int = 42) -> dict:
    on = (mode == "on")
    os.environ["BRAIN_WORLDMODEL_LOCAL_FREEZE"] = "1" if on else "0"
    os.environ["BRAIN_SURPRISE_LOCAL_FREEZE"] = "1" if on else "0"

    import numpy as np
    from research.runners.onebrain_merge_framework import merge_organs, REGISTRY, OrganDescriptor
    from research.runners.surprise_production_organ import SurpriseProductionOrgan
    from research.runners.worldmodel_production_organ import WorldModelProductionOrgan
    from research.runners._affective_world_model_derisk import WORLDMODEL_FREEZE_GATE
    from research.runners._spiking_expectation_rpe_derisk import SURPRISE_FREEZE_GATE

    probe = OrganDescriptor(key="probe", regions=("probe_a", "probe_b"), spec_fn=_probe_spec, config={})
    seed = int(seed)
    pool = merge_organs([REGISTRY["surprise"], REGISTRY["worldmodel"], probe], seed, wire=True)
    bridge = pool.bridge

    out = {"mode": mode, "seed": seed}
    out["gates_before_organ_build"] = sorted(bridge.list_plasticity_gates())

    # TRAIN surprise + world-model on the shared pool -- exactly what webapp/server.py's warmup does
    # (_get_surprise_organ().ensure_built() then _get_worldmodel_organ().ensure_built()).
    surp = SurpriseProductionOrgan(seed=seed, shared=pool)
    surp.ensure_built()
    wm = WorldModelProductionOrgan(seed=seed, shared=pool)
    wm.ensure_built()

    out["enable_hebbian_learning_after_both_organs_built"] = bool(bridge.core_config.enable_hebbian_learning)
    gates_after = sorted(bridge.list_plasticity_gates())
    out["gates_after_organ_build"] = gates_after
    out["worldmodel_gate_value"] = (float(bridge.get_plasticity_gate_value(WORLDMODEL_FREEZE_GATE))
                                    if WORLDMODEL_FREEZE_GATE in gates_after else None)
    out["surprise_gate_value"] = (float(bridge.get_plasticity_gate_value(SURPRISE_FREEZE_GATE))
                                  if SURPRISE_FREEZE_GATE in gates_after else None)

    # ANSWER-PRESERVATION (the 6-seed pre-registration's no-regression arm): both organs' own functional
    # reads must be IDENTICAL between the old global-kill mechanism and the new local-gate mechanism -- the
    # fix changes HOW the freeze is applied, never WHAT either organ reports to its caller.
    out["surprise_answer"] = surp.judge("dog", "chase", "cat", "bone")
    out["worldmodel_answer"] = {"expectation_pos": wm.expectation(1), "expectation_neg": wm.expectation(-1),
                                "surprise_confirm": wm.read_surprise(1, 1), "surprise_violate": wm.read_surprise(1, -1)}

    # Snapshot every pathway's weights POST-TRAINING (the "chat has started" baseline), then drive real
    # co-activity on the probe pathway ONLY (an unrelated faculty's turn) for a few hundred steps -- exactly
    # the kind of incidental co-firing a live chat turn produces on whatever is UNGATED on the shared bridge.
    nnz = bridge.cp_connections.nnz
    w0 = np.asarray(_to_host(bridge.cp_connections.data))[:nnz].copy()
    probe_a = bridge.region_manager.indices("probe_a")
    probe_b = bridge.region_manager.indices("probe_b")
    for _ in range(300):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[probe_a] = np.float32(700.0)
        bridge.cp_external_input_current[probe_b] = np.float32(700.0)   # co-fire pre+post -> Hebbian-eligible
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0

    probe_dw, probe_n = _pathway_dw(bridge, w0, "probe_a", "probe_b")
    wm_dw, wm_n = _pathway_dw(bridge, w0, "state", "pred_pos")
    wm_dw2, _ = _pathway_dw(bridge, w0, "state", "pred_neg")
    su_dw, su_n = _pathway_dw(bridge, w0, "cue", "patient_expected")

    out["probe_pathway"] = {"max_dw": probe_dw, "n_synapses": probe_n}
    out["worldmodel_own_pathway"] = {"max_dw": max(wm_dw or 0.0, wm_dw2 or 0.0), "n_synapses": wm_n}
    out["surprise_own_pathway"] = {"max_dw": su_dw, "n_synapses": su_n}
    return out


def run_fallback_pool_check(seed: int = 42) -> dict:
    """Coordinator review round 2 (2026-09-24): with `BRAIN_ONEBRAIN_WAVE3_POOL=0` AND `BRAIN_ONEBRAIN_SINGLE_
    POOL=0` (both default-ON; setting both is required to actually reach the LEGACY pool below -- WAVE3_POOL=0
    alone still falls through to `onebrain_single_pool_production.get_single_pool`, which builds via
    `onebrain_merge_framework.merge_organs` + the SAME `_surprise_spec`/`_worldmodel_spec` this fix already
    threads `local_freeze_gate` into, so it does NOT crash), `get_organ()` falls back further, to the legacy
    `onebrain_merge_production.MergedSubstrate` (pool #1, surprise+world-model only) -- built via its OWN inline
    `build_expectation_circuit(...)`/`build_world_model_circuit(...)` calls, which never pass `local_freeze_gate`.
    With `BRAIN_WORLDMODEL_LOCAL_FREEZE=1`/`BRAIN_SURPRISE_LOCAL_FREEZE=1` also set, `_build_one` used to call
    `bridge.set_plasticity_gate(<GATE>, 0.0)` unconditionally and crash with `KeyError` (the gate was never
    declared on this bridge's wiring plan). `_freeze_local_or_fallback` (in both organ modules) now checks
    `<GATE> in bridge.list_plasticity_gates()` first and falls back to the pre-fix `cfg.enable_hebbian_learning
    = False` with a logged `RuntimeWarning` instead of crashing. This function builds BOTH organs through the
    real `get_organ()` entry point under this exact flag combination and reports whether it crashed, whether the
    fallback warning fired, and what state the bridge ended up in."""
    seed = int(seed)
    os.environ["BRAIN_ONEBRAIN_WAVE3_POOL"] = "0"
    os.environ["BRAIN_ONEBRAIN_SINGLE_POOL"] = "0"
    os.environ["BRAIN_WORLDMODEL_LOCAL_FREEZE"] = "1"
    os.environ["BRAIN_SURPRISE_LOCAL_FREEZE"] = "1"

    import warnings as _warnings
    out = {"seed": seed, "crashed": False, "error": None, "warnings": []}
    try:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            from research.runners.onebrain_merge_production import get_merged_substrate
            pool_before = get_merged_substrate(seed)   # confirms which pool this combination actually reaches
            out["reached_pool"] = type(pool_before).__name__

            from research.runners.surprise_production_organ import get_organ as get_surprise_organ
            from research.runners.worldmodel_production_organ import get_organ as get_worldmodel_organ
            surp = get_surprise_organ(seed=seed)
            surp.ensure_built()
            wm = get_worldmodel_organ(seed=seed)
            wm.ensure_built()
            out["warnings"] = [str(w.message) for w in caught]
    except Exception as e:
        out["crashed"] = True
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out["surprise_bridge_is_worldmodel_bridge"] = bool(surp.bridge is wm._st["bridge"])
    out["enable_hebbian_learning_after_build"] = bool(surp.bridge.core_config.enable_hebbian_learning)
    out["gates_present"] = sorted(surp.bridge.list_plasticity_gates())
    # both organs still read correctly on the fallback (global-kill) path -- the crash fix must not make the
    # organ itself non-functional, only change HOW it freezes.
    out["surprise_answer"] = surp.judge("dog", "chase", "cat", "bone")
    out["worldmodel_answer"] = wm.expectation(1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "on", "fallback-pool"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.mode == "fallback-pool":
        result = run_fallback_pool_check(seed=a.seed)
    else:
        result = run(a.mode, seed=a.seed)
    print(json.dumps(result, default=str))
    sys.exit(0)
