"""Confirm the gate is the SOLE bottleneck: if the role-route gate is held OPEN throughout the readout
(rather than flickering at the EMA threshold), does the composer role bank fire at the Python path's rate
(~0.125) and does the seed-42 patient ('come') readout decode correctly?

Three readout protocols on seed 42, D=2048, patient='come':
  (A) BASELINE synaptic: gate coupled (flickers), 150-step readout — the current behavior.
  (B) PRE-WARM + COUPLED: drive parser conj for a pre-window (no accumulation), then 150-step readout with
      the coupling still live (gate still re-evaluated every step → still flickers).
  (C) PRE-WARM + HELD-OPEN: drive parser conj for a pre-window so the parser FIRES and (via the coupling)
      LATCHES the gate open, then for the readout window REMOVE the gate coupling so the parser-opened gate
      PERSISTS (comprehension done → compose with the gate as comprehension set it), 150-step readout.

For each, print the role_on bank rate and the decoded patient word (cleanup over the full vocab). This
isolates whether holding the parser-opened gate open during composition restores the Python role-bank rate
and fixes the decode. Diagnostic only — not the shipped op.

    python -m research.findings.raw._step2_synaptic_gatehold_diag
"""
from __future__ import annotations

import numpy as np

from research.runners.unified_brain_bridge import UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES, ROLE_SRC_DRIVE_PA
from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE, RESET_STEPS
from sim.backend import get_backend, to_host

SEED = 42; PROJ_DIM = 2048; POS = 2


def _drive(u, xp, bridge, comp, idx, conj_k, fon, foff):
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[u.parser.conj_arr[conj_k]] = u.parser.drive
    for r in SYNAPTIC_ROUTE_ROLES:
        cur[u._role_src[r]] = ROLE_SRC_DRIVE_PA
    cur[idx["fill_on"]] = xp.asarray(fon.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(foff.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = comp.coinc_bias
    return cur


def _store_one_synaptic(u, sentence, prewarm=0, hold_open=False):
    """Store ONE flat fact via a (modified) synaptic protocol with optional pre-warm / hold-open, return the
    role_on bank mean rate on the PATIENT word's op (for the rate read-out) by re-implementing _op_synaptic
    inline with the protocol knobs."""
    from research.runners.core_sim_composition import onoff as _onoff
    xp, _ = get_backend()
    bridge = u.bridge; comp = u.composer; idx = comp.idx
    words = sentence.split(); v = 0
    bound_on = np.zeros(comp.D); bound_off = np.zeros(comp.D)
    fact = {}
    patient_role_on_rate = None
    for pos in range(3):
        word = words[pos]; k = pos * 2 + v
        role = u.parser.role_of(pos, "active")
        fact[role] = word
        c_on, c_off = _onoff(comp.concepts[word])
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        # reset
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
        cur = _drive(u, xp, bridge, comp, idx, k, fon, foff)
        bridge.cp_external_input_current[:] = cur
        # pre-warm (no accumulation) — lets the parser fire and the coupling latch the gate open
        for _ in range(prewarm):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
        # optionally hold the parser-opened gate open during readout by removing the coupling for this window
        saved = None
        if hold_open:
            saved = bridge._gate_couplings
            bridge._gate_couplings = []   # stop per-step re-evaluation → the latched gate value persists
        acc = {b: xp.zeros(comp.D, dtype=xp.float64) for b in ("A", "B", "C", "D")}
        ron = 0.0
        for _ in range(comp.run_steps):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
            for b in ("A", "B", "C", "D"):
                acc[b] += bridge.cp_firing_states[idx[b]].astype(xp.float64)
            ron += float(bridge.cp_firing_states[idx["role_on"]].astype(xp.float64).mean())
        if hold_open:
            bridge._gate_couplings = saved
            # re-close the gates we may have left open (restore the closed default for the next op)
            for r in SYNAPTIC_ROUTE_ROLES:
                bridge.set_transmission_gate(f"role_route_{r}", 0.0)
                cpl = next((c for c in bridge._gate_couplings if c["gate_name"] == f"role_route_{r}"), None)
                if cpl is not None:
                    cpl["ema"] = 0.0; cpl["last_value"] = None
        bridge.cp_external_input_current[:] = 0.0
        rates = {b: to_host(acc[b]) / comp.run_steps for b in ("A", "B", "C", "D")}
        bound_on += rates["A"] + rates["B"]; bound_off += rates["C"] + rates["D"]
        if role == "patient":
            patient_role_on_rate = ron / comp.run_steps
    comp.kb.append((fact, onoff(bound_on - bound_off)))
    return patient_role_on_rate


def main():
    u = UnifiedBrainBridge(seed=SEED, proj_dim=PROJ_DIM, enable_synaptic_route=True)
    # the three seed-42 facts whose patient is 'come' (from the diagnosis): use a representative one
    rng = np.random.default_rng(SEED + 1)
    facts = [tuple(str(x) for x in rng.choice(u.words, size=3, replace=False)) for _ in range(6)]
    come_facts = [f for f in facts if f[2] == "come"]
    print(f"[gatehold] seed-42 facts with patient='come': {come_facts}", flush=True)
    test = come_facts[0] if come_facts else facts[0]
    a, ac, p = test
    sent = f"{a} {ac} {p}"
    print(f"[gatehold] test fact: {sent!r} (expect query_patient({a!r},{ac!r}) == {p!r})", flush=True)

    for label, prewarm, hold in (("A baseline", 0, False), ("B prewarm120 coupled", 120, False),
                                 ("C prewarm120 held-open", 120, True)):
        u.kb = []
        ron = _store_one_synaptic(u, sent, prewarm=prewarm, hold_open=hold)
        decoded = u.query_patient(a, ac)
        ok = "OK" if decoded == p else "WRONG"
        print(f"[gatehold] {label:<26} patient role_on rate={ron:.3f}  decoded={decoded!r}  [{ok}]", flush=True)

    print("[gatehold] (Python path role_on reference ~0.125; if C reaches ~0.125 and decodes OK, "
          "holding the parser-opened gate open during composition is the fix.)", flush=True)


if __name__ == "__main__":
    main()
