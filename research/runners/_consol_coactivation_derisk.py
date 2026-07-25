"""Consolidation de-risk — CORE potentiation check (1-seed smoke). The A1 failure: CA3-only replay -> concept pools
never fire -> the plastic ca1->concept/slot wire stays frozen at ~0.01. THE FIX (co-activation): drive the CA3 tag AND
reinstate the fact's concept pools during replay -> post-spikes -> STDP potentiates the wire. GO(this smoke) iff the
ca1->comp_attr wire potentiates OFF its init with coactivate=ON and stays ~frozen with coactivate=OFF."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
from types import SimpleNamespace
import numpy as np
from research.runners.nmda_compositional_consolidation import (
    build_substrate, train_phase1, encode_facts_with_reinstatement, coactivation_replay,
    _mean_gate_weight, CONSOLIDATED_FACTS, _try_tgate, _try_pgate)
from sim.backend import get_backend, to_host


def slot_ignition(bridge, tags, n_slots):
    """After consolidation: cue each fact's tag, read which comp_attr slot ignites. SELECTIVE iff fact i -> slot i."""
    cp, _ = get_backend(); rm = bridge.region_manager
    _try_tgate(bridge, "nmda_attractor", 1.0)                # attractor ON (hold)
    _try_pgate(bridge, "ca1_to_comp_attr", 1.0)              # tag -> ca1 -> slot route open
    slot_arr = {s: list(rm.indices(f"comp_attr_{s}")) for s in range(n_slots)}
    rows = []
    for i, tag in enumerate(tags):
        bridge.cp_external_input_current[:] = 0.0            # reset: quiet so any latch decays
        for _ in range(60):
            bridge._run_one_simulation_step()
        bridge.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        cnt = {s: 0 for s in range(n_slots)}
        for _ in range(80):
            bridge._run_one_simulation_step()
            fs = to_host(bridge.cp_firing_states)
            for s in range(n_slots):
                cnt[s] += int(fs[slot_arr[s]].sum())
        try:
            bridge.clear_tag_drive(tag)
        except Exception:
            pass
        top = max(cnt, key=cnt.get)
        rows.append((i, top, cnt[top], cnt))
    return rows

ARGS = dict(ca1_concept_density=0.25, ca1_concept_weight=0.0, nmda_self_weight=12.0, nmda_self_density=0.15,
            nmda_recurrent_ratio=0.6, cross_pool_density=0.10, stdp_w_max=8.0, enable_global_nmda=False,
            enable_hebbian=True, skip_nmda_additions=True,   # Option-1: no weak-pool self-loops; use the dedicated region
            comp_attractor_slots=len(CONSOLIDATED_FACTS), comp_attractor_n_per=120, comp_self_weight=12.0, comp_wta_weight=5.0)


def one(coactivate, seed=42, cycles=100, n_events=40):
    t0 = time.time()
    b = build_substrate(seed, SimpleNamespace(**ARGS))
    print(f"    [stage] built ({time.time()-t0:.0f}s)", flush=True)
    # SKIP train_phase1 for the potentiation MECHANISM check: co-activation drives the concept pools DIRECTLY by index
    # (they fire regardless of word->pool training), and encode uses teacher-drive. Phase-1 is only needed for the
    # FUNCTIONAL recall test (use a cached substrate there, per the research).
    tags, dims = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    print(f"    [stage] encoded ({time.time()-t0:.0f}s)", flush=True)
    w_ca1_concept_0 = _mean_gate_weight(b, "ca1_to_concept_pool")
    w_ca1_slot_0 = _mean_gate_weight(b, "ca1_to_comp_attr")
    w_pool_slot_0 = _mean_gate_weight(b, "concept_to_comp_attr")
    coactivation_replay(b, CONSOLIDATED_FACTS, tags, cycles, seed, coactivate=coactivate, attractor_on=True)
    w_ca1_slot_1 = _mean_gate_weight(b, "ca1_to_comp_attr")
    # FUNCTIONAL test: does each fact's TAG now ignite its dedicated slot (fact i -> slot i)?
    rows = slot_ignition(b, tags, len(CONSOLIDATED_FACTS))
    sel = sum(1 for (i, top, cnt, _) in rows if top == i and cnt > 0)
    ign = sum(1 for (i, top, cnt, _) in rows if cnt > 0)
    print(f"  [coactivate={coactivate}] ca1->slot {w_ca1_slot_0:.4f}->{w_ca1_slot_1:.4f} (Δ{w_ca1_slot_1-w_ca1_slot_0:+.4f}) | "
          f"ignition {ign}/{len(rows)} SELECTIVE(fact i->slot i) {sel}/{len(rows)} | slots={[(i,top,c) for (i,top,c,_) in rows]} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return sel, ign, w_ca1_slot_1 - w_ca1_slot_0


print("Consolidation FUNCTIONAL smoke (seed 42): after CO-ACTIVATION replay, does each fact's tag IGNITE its dedicated "
      "attractor slot SELECTIVELY (fact i -> slot i)? vs no-co-activation control (should not consolidate).", flush=True)
selON, ignON, dON = one(True)
selOFF, ignOFF, dOFF = one(False)
print(f"\n  SELECTIVE ignition: coactivate-ON {selON}/{len(CONSOLIDATED_FACTS)}  vs  OFF {selOFF}/{len(CONSOLIDATED_FACTS)}", flush=True)
go = selON >= 2 and selOFF <= 1
print(f"  -> {'FUNCTIONAL GO (co-activation consolidates selective slot ignition; control does not)' if go else 'partial -> tune (cycles/drive/WTA/slot-strength)'}", flush=True)
print("CONSOL-COACT-SMOKE DONE", flush=True)
