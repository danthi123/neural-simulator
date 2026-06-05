"""Diagnose WHY the gate flickers under continuous parser drive (the pre-warm sweep showed the gate is never
stably open). Print, step by step over a 120-step continuous drive of the patient conjunction on seed 42:
the parser role ensemble firing rate, the gate EMA, and the gate value — to see whether the EMA oscillates
around the 0.05 threshold (bursty parser firing) so the gate flickers 1.0<->0.0.

    python -m research.findings.raw._step2_synaptic_prewarm_diag
"""
from __future__ import annotations

import numpy as np

from research.runners.unified_brain_bridge import UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES, ROLE_SRC_DRIVE_PA
from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE, RESET_STEPS
from sim.backend import get_backend, to_host

SEED = 42; PROJ_DIM = 2048; WORD = "come"; POS = 2


def main():
    u = UnifiedBrainBridge(seed=SEED, proj_dim=PROJ_DIM, enable_synaptic_route=True)
    xp, _ = get_backend()
    bridge = u.bridge; comp = u.composer; idx = comp.idx
    k = POS * 2 + 0
    role = u.parser.role_of(POS, "active")
    role_ens = u.parser.role_arr[role]          # the parser's role ensemble (control pool for the gate)
    c_on, c_off = onoff(comp.concepts[WORD])
    fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
    gate_syn = bridge._transmission_gate_to_synapses.get(f"role_route_{role}")

    # find the registered coupling so we can read its EMA
    coupling = next((c for c in bridge._gate_couplings if c["gate_name"] == f"role_route_{role}"), None)
    print(f"[diag] role={role} threshold={coupling['threshold']} alpha={coupling['alpha']} "
          f"open_value={coupling['open_value']} ensemble_size={len(role_ens)}", flush=True)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()

    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[u.parser.conj_arr[k]] = u.parser.drive
    for r in SYNAPTIC_ROUTE_ROLES:
        cur[u._role_src[r]] = ROLE_SRC_DRIVE_PA
    cur[idx["fill_on"]] = xp.asarray(fon.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(foff.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = comp.coinc_bias
    bridge.cp_external_input_current[:] = cur

    rates = []; emas = []; gates = []
    for step in range(120):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        rate = float(to_host(bridge.cp_firing_states[role_ens]).mean())
        gv = float(to_host(bridge.cp_transmission_gain[gate_syn]).mean()) if gate_syn is not None else None
        rates.append(rate); emas.append(coupling["ema"]); gates.append(gv)
    bridge.cp_external_input_current[:] = 0.0

    print(f"[diag] role-ensemble rate over 120 steps: mean={np.mean(rates):.4f} "
          f"min={np.min(rates):.4f} max={np.max(rates):.4f} frac_nonzero={np.mean(np.asarray(rates) > 0):.3f}",
          flush=True)
    print(f"[diag] gate-EMA over 120 steps: mean={np.mean(emas):.4f} min={np.min(emas):.4f} "
          f"max={np.max(emas):.4f} frac>=thr={np.mean(np.asarray(emas) >= coupling['threshold']):.3f}",
          flush=True)
    print(f"[diag] gate value frac>=0.99: {np.mean(np.asarray([g for g in gates if g is not None]) >= 0.99):.3f}",
          flush=True)
    # show the first 60 steps as a compact trace (rate / ema / gate)
    print("[diag] step: rate  ema   gate  (first 60)", flush=True)
    for s in range(0, 60):
        print(f"[diag]  {s:>3}: {rates[s]:.3f} {emas[s]:.3f} {gates[s]:.1f}", flush=True)


if __name__ == "__main__":
    main()
