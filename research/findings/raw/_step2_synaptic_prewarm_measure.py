"""Measure the gate pre-warm length needed to resolve the seed-42 patient regression.

Question: in the synaptic op (`_op_synaptic`), the parser-coupled transmission gate `role_route_<role>`
opens via an EMA (alpha 0.3, threshold 0.05) of the parser role ensemble's firing. After the per-op
`RESET_STEPS` of zero input, the gate EMA has decayed to ~0 → during the readout window the gate spends
~102/150 steps below 0.99 (the `_step2_synaptic_gate_opcheck.py` finding), so the composer's role bank
fires at ~1/7 the Python path's rate. The faithful fix (controller-approved) is to PRE-WARM the gate: drive
the parser conjunction (+ role_src pools) for a PRE-WINDOW long enough for the gate EMA to cross threshold
and the gate to read open_value (1.0), WITHOUT accumulating the coincidence banks, THEN run the readout
window with the gate already open.

This probe measures, on seed 42, D=2048, for the failing patient word 'come':
  (a) how many pre-window steps until the gate first reaches 1.0 and how many until it is STABLY 1.0;
  (b) the gate trajectory + role-bank firing rate over a subsequent readout window AFTER an N-step pre-warm,
      for several candidate N — confirming the readout window now sees the gate ~1.0 throughout and the
      role-bank rate matches the Python path's direct-role rate (~0.125).

It does NOT change `_op_synaptic`; it drives the same pools by hand to find the right pre-window length so
the implementation uses a safe value. Run on CuPy/GPU (do NOT pin numpy).

    python -m research.findings.raw._step2_synaptic_prewarm_measure
"""
from __future__ import annotations

import numpy as np

from research.runners.unified_brain_bridge import UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES, ROLE_SRC_DRIVE_PA
from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE, RESET_STEPS
from sim.backend import get_backend, to_host

SEED = 42
PROJ_DIM = 2048
WORD = "come"          # the patient that fails on seed 42
POS = 2                # patient position (active voice)


def _setup_drive(u, xp, bridge, comp, idx, conj_k, fon, foff):
    """Build the full synaptic-op drive current (parser conjunction + all role_src + fill + coinc bias)."""
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[u.parser.conj_arr[conj_k]] = u.parser.drive
    for r in SYNAPTIC_ROUTE_ROLES:
        cur[u._role_src[r]] = ROLE_SRC_DRIVE_PA
    cur[idx["fill_on"]] = xp.asarray(fon.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(foff.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = comp.coinc_bias
    return cur


def main():
    u = UnifiedBrainBridge(seed=SEED, proj_dim=PROJ_DIM, enable_synaptic_route=True)
    xp, _ = get_backend()
    bridge = u.bridge; comp = u.composer; idx = comp.idx
    k = POS * 2 + 0       # patient conjunction, active voice
    role = u.parser.role_of(POS, "active")
    print(f"[prewarm] parser routes position {POS} (active) -> role {role!r}", flush=True)

    c_on, c_off = onoff(comp.concepts[WORD])
    fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
    gate_syn = bridge._transmission_gate_to_synapses.get(f"role_route_{role}")

    def gate_value():
        if gate_syn is None or bridge.cp_transmission_gain is None:
            return None
        return float(to_host(bridge.cp_transmission_gain[gate_syn]).mean())

    # ── (a) Reset (gate decays to ~0), then drive and watch the gate ramp; find first-open + stable-open. ──
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
    print(f"[prewarm] gate AFTER reset (pre-drive) = {gate_value()}", flush=True)

    cur = _setup_drive(u, xp, bridge, comp, idx, k, fon, foff)
    bridge.cp_external_input_current[:] = cur

    first_open = None
    stable_open = None
    traj = []
    MAXW = 220
    for step in range(MAXW):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        gv = gate_value()
        traj.append(gv)
        if gv is not None and gv >= 0.99 and first_open is None:
            first_open = step
        # stable = open and stays open for the rest of a 20-step lookahead (cheap: check last 20 all open)
        if (stable_open is None and gv is not None and gv >= 0.99
                and len(traj) >= 20 and all(g is not None and g >= 0.99 for g in traj[-20:])):
            stable_open = step - 19
    bridge.cp_external_input_current[:] = 0.0

    n_open = sum(1 for g in traj if g is not None and g >= 0.99)
    print(f"[prewarm] over {MAXW} continuous drive steps: gate first reached 1.0 at step={first_open}, "
          f"first STABLY-open (20-step) at step={stable_open}, open on {n_open}/{MAXW} steps", flush=True)

    # ── (b) For candidate pre-warm lengths, reset → pre-warm N (no accumulation) → readout 150 (measure). ──
    print("[prewarm] candidate pre-warm sweep (reset -> pre-warm N -> 150-step readout):", flush=True)
    print(f"[prewarm]   {'N':>5}  {'gate<0.99 in readout':>22}  {'gate mean':>10}  {'role_on rate':>12}", flush=True)
    for N in (0, 30, 60, 80, 100, 120, 150):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
        cur = _setup_drive(u, xp, bridge, comp, idx, k, fon, foff)
        bridge.cp_external_input_current[:] = cur
        # pre-warm N steps (NO accumulation)
        for _ in range(N):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
        # readout window (measure gate + role bank); current stays applied
        below = 0; gsum = 0.0; ron = 0.0
        for _ in range(comp.run_steps):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
            gv = gate_value()
            if gv is not None:
                gsum += gv
                if gv < 0.99:
                    below += 1
            ron += float(bridge.cp_firing_states[idx["role_on"]].astype(xp.float64).mean())
        bridge.cp_external_input_current[:] = 0.0
        ron /= comp.run_steps; gmean = gsum / comp.run_steps
        print(f"[prewarm]   {N:>5}  {below:>5}/{comp.run_steps:<16}  {gmean:>10.3f}  {ron:>12.3f}", flush=True)

    # Python-path reference role-bank rate (direct role current, no gate) for comparison.
    from research.runners.core_sim_composition import ROLE_DRIVE
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
    rv = comp.roles[role]
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[idx["role_on"]] = xp.asarray((rv > 0).astype(np.float32) * ROLE_DRIVE)
    cur[idx["role_off"]] = xp.asarray((rv < 0).astype(np.float32) * ROLE_DRIVE)
    cur[idx["fill_on"]] = xp.asarray(fon.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(foff.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = comp.coinc_bias
    bridge.cp_external_input_current[:] = cur
    pron = 0.0
    for _ in range(comp.run_steps):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        pron += float(bridge.cp_firing_states[idx["role_on"]].astype(xp.float64).mean())
    bridge.cp_external_input_current[:] = 0.0
    print(f"[prewarm] PYTHON path role_on rate (direct role current, no gate) = {pron / comp.run_steps:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
