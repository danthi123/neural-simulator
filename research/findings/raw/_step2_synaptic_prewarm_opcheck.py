"""Confirm — through the SHIPPED `UnifiedBrainBridge._op_synaptic` (with the gate pre-warm) — that the
parser-opened role gate reads ~1.0 during the READOUT window (not flickering at the EMA threshold as it did
before the fix). This is the directive's verification: "confirm the gate is ~1.0 during the readout AFTER
your pre-warm".

It wraps `bridge._run_one_simulation_step` to record the patient gate value on every step, then runs the
real `u.hear_synaptic("apple south come")` (the seed-42 fact whose patient 'come' regressed). The recorded
trace is segmented into the per-word ops; for the PATIENT op the readout-window gate values must be ~1.0.
Also re-confirms the seed-42 6-fact what/who recall through the shipped path == the Python path (6/6).

Run on CuPy/GPU (do NOT pin numpy).

    python -m research.findings.raw._step2_synaptic_prewarm_opcheck
"""
from __future__ import annotations

import numpy as np

from research.runners.unified_brain_bridge import UnifiedBrainBridge
from sim.backend import get_backend, to_host

SEED = 42; PROJ_DIM = 2048


def main():
    u = UnifiedBrainBridge(seed=SEED, proj_dim=PROJ_DIM, enable_synaptic_route=True)
    bridge = u.bridge
    patient_gate = bridge._transmission_gate_to_synapses.get("role_route_patient")

    trace = []
    orig_step = bridge._run_one_simulation_step

    def _wrapped_step(*a, **k):
        r = orig_step(*a, **k)
        if patient_gate is not None and bridge.cp_transmission_gain is not None:
            trace.append(float(to_host(bridge.cp_transmission_gain[patient_gate]).mean()))
        return r

    bridge._run_one_simulation_step = _wrapped_step
    try:
        u.kb = []
        u.hear_synaptic("apple south come")   # patient 'come' — the seed-42 regressor
    finally:
        bridge._run_one_simulation_step = orig_step

    # The trace covers 3 words. Each word op = RESET_STEPS(20) + pre-window(<=60) + readout(150). The PATIENT
    # is the 3rd word (position 2). We locate its readout window: the LAST contiguous run of 150 steps where
    # the gate is held (coupling paused). Simplest robust check: the final 150 recorded steps are the patient
    # readout (the op ends with the readout window, and after it we restore + zero the gate but that is outside
    # the run loop). Report the gate stats over that window.
    readout = np.asarray(trace[-150:]) if len(trace) >= 150 else np.asarray(trace)
    below = int(np.sum(readout < 0.99))
    print(f"[opcheck] PATIENT readout window (last 150 steps via shipped _op_synaptic): "
          f"gate mean={readout.mean():.3f} min={readout.min():.3f} gate<0.99 on {below}/{len(readout)} steps",
          flush=True)
    if below == 0:
        print("[opcheck] PASS: the parser-opened gate is held at ~1.0 for the WHOLE patient readout window "
              "(was ~102/150 below 0.99 before the pre-warm fix).", flush=True)
    else:
        print(f"[opcheck] gate still below 0.99 on {below}/150 readout steps — pre-warm/hold not fully "
              "effective; investigate.", flush=True)

    # Re-confirm recall parity on all 6 seed-42 facts through the shipped path (what/who) vs Python.
    rng = np.random.default_rng(SEED + 1)
    facts = [tuple(str(x) for x in rng.choice(u.words, size=3, replace=False)) for _ in range(6)]
    syn_what = syn_who = py_what = py_who = 0
    for a, ac, p in facts:
        u.kb = []
        u.hear_synaptic(f"{a} {ac} {p}")
        syn_what += int(u.query_patient(a, ac) == p); syn_who += int(u.query_agent(ac, p) == a)
    for a, ac, p in facts:
        u.kb = []
        roles = u.parse(f"{a} {ac} {p}")
        u.store(roles["agent"], roles["action"], roles["patient"])
        py_what += int(u.query_patient(a, ac) == p); py_who += int(u.query_agent(ac, p) == a)
    print(f"[opcheck] seed-42 recall via shipped path: synaptic what={syn_what}/6 who={syn_who}/6 | "
          f"python what={py_what}/6 who={py_who}/6", flush=True)


if __name__ == "__main__":
    main()
