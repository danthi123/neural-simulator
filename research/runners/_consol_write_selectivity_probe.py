"""Consolidation NEXT-(b) probe: is the unseparated ca1->slot c_drive caused by the co-activation write running with
the ATTRACTOR ON (recurrent spread + weak WTA make ALL slots fire when slot_i is driven -> STDP potentiates ca1_i->ALL
slots, not ca1_i->slot_i)? Test the fix: run the co-activation WRITE with attractor_on=False (no recurrent spread ->
only the externally-driven slot_i fires -> ca1_i->slot_i potentiates SELECTIVELY), then read with the attractor on.

GO(indicator) = attractor-OFF write gives c_drive own/other ratio > 1.5 (separation) where attractor-ON write gives
~1.0. This is the ROOT-cause fix upstream of the plateau; if it separates, the dendritic plateau (already de-risked) has
structure to route on. Reuses the sweep driver's cdrive_probe + slot_ignition. Seed-42 indicator; 6-seed if GO.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_write_selectivity_probe
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("SIM_BACKEND", "cupy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay, _mean_gate_weight, CONSOLIDATED_FACTS)
from research.runners._consol_dendritic_opsweep import cdrive_probe, slot_ignition, BASE, N

# a mid operating point (dendritic plateau on; the write-mode is the variable under test)
OP = dict(comp_dendritic=True, comp_wta_weight=20.0, comp_k_thresh=3.0, comp_self_regen=0.10, comp_kir_g=3.0)


def one(write_attractor_on, seed=42, cycles=40):
    a = dict(BASE); a.update(OP)
    b = build_substrate(seed, SimpleNamespace(**a))
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    w0 = _mean_gate_weight(b, "ca1_to_comp_attr")
    coactivation_replay(b, CONSOLIDATED_FACTS, tags, cycles, seed,
                        coactivate=True, attractor_on=write_attractor_on, slot_drive_pA=1400.0)
    w1 = _mean_gate_weight(b, "ca1_to_comp_attr")
    cd = cdrive_probe(b, tags)
    ig = slot_ignition(b, tags)
    return cd, ig, w1 - w0


print(f"WRITE-SELECTIVITY PROBE (seed 42, op={OP}) — does an attractor-OFF write separate c_drive?", flush=True)
for mode in (True, False):
    t0 = time.time()
    cd, ig, dw = one(mode)
    label = "attractor-ON write (the sweep's mode)" if mode else "attractor-OFF write (the NEXT-b FIX)"
    print(f"  [{label}] c_drive mean_ratio={cd['mean_ratio']} n_separated={cd['n_separated']}/{N} | "
          f"per-fact ratios={[r['ratio'] for r in cd['rows']]} | selective_ignition={ig['selective']}/{N} | "
          f"dw={dw:+.4f} ({time.time()-t0:.0f}s)", flush=True)
print("VERDICT: attractor-OFF ratio >1.5 while attractor-ON ~1.0 => the write-spread is the root cause, the fix is the "
      "attractor-off write (6-seed next). If both ~1.0 => deeper (line/bump attractor / a different selective write).",
      flush=True)
print("WRITE-SELECTIVITY-PROBE DONE", flush=True)
