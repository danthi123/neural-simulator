"""Confirm the operating-point gap behind the seed-42 patient regression: does the GATED role route deliver
LESS effective role-bank current than the Python path's DIRECT role current, because the transmission gate
ramps from 0 via its EMA over the readout window?

On seed 42, D=2048, synaptic route: for the patient role of one failing fact ('apple south come' -> patient
'come'), drive the parser PATIENT conjunction + all role_src pools and read (a) the transmission-gate value
trajectory for `role_route_patient` over the window and (b) the composer role_ON/role_OFF firing rate. Compare
to the Python path, which sets role_ON/role_OFF current to ROLE_DRIVE (2500) directly with no gate. A lower
gated role-bank rate (especially early in the window) is the mechanism: the gate EMA warm-up costs role drive,
thinning the already-razor (between-cos 0.81) patient cleanup margin.

    python -m research.findings.raw._step2_synaptic_gate_opcheck
"""
from __future__ import annotations

import numpy as np

from research.runners.unified_brain_bridge import UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES, ROLE_SRC_DRIVE_PA
from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE, RESET_STEPS, ROLE_DRIVE
from sim.backend import get_backend, to_host

SEED = 42
PROJ_DIM = 2048
WORD = "come"          # the patient that fails on seed 42
POS = 2                # patient position (active voice)


def main():
    u = UnifiedBrainBridge(seed=SEED, proj_dim=PROJ_DIM, enable_synaptic_route=True)
    xp, _ = get_backend()
    bridge = u.bridge; comp = u.composer; idx = comp.idx
    k = POS * 2 + 0       # patient conjunction, active voice

    role = u.parser.role_of(POS, "active")
    print(f"[opcheck] parser routes position {POS} (active) -> role {role!r}", flush=True)

    c_on, c_off = onoff(comp.concepts[WORD])
    fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)

    # ---- SYNAPTIC path: drive parser conjunction (opens gate) + all role_src; read gate + role-bank rate ----
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

    gate_syn = bridge._transmission_gate_to_synapses.get(f"role_route_{role}")   # the gate's synapse indices
    gate_traj = []
    syn_role_on = syn_role_off = 0.0
    for _ in range(comp.run_steps):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        if gate_syn is not None and bridge.cp_transmission_gain is not None:
            gate_traj.append(float(to_host(bridge.cp_transmission_gain[gate_syn]).mean()))
        else:
            gate_traj.append(None)
        syn_role_on += float(bridge.cp_firing_states[idx["role_on"]].astype(xp.float64).mean())
        syn_role_off += float(bridge.cp_firing_states[idx["role_off"]].astype(xp.float64).mean())
    bridge.cp_external_input_current[:] = 0.0
    syn_role_on /= comp.run_steps; syn_role_off /= comp.run_steps

    # ---- PYTHON path: drive role_ON/OFF DIRECTLY at ROLE_DRIVE (no gate); read role-bank rate ----
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
    py_role_on = py_role_off = 0.0
    for _ in range(comp.run_steps):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        py_role_on += float(bridge.cp_firing_states[idx["role_on"]].astype(xp.float64).mean())
        py_role_off += float(bridge.cp_firing_states[idx["role_off"]].astype(xp.float64).mean())
    bridge.cp_external_input_current[:] = 0.0
    py_role_on /= comp.run_steps; py_role_off /= comp.run_steps

    valid_traj = [g for g in gate_traj if g is not None]
    if valid_traj:
        print(f"[opcheck] gate role_route_{role} value: first={valid_traj[0]:.3f} "
              f"step5={valid_traj[4] if len(valid_traj) > 4 else float('nan'):.3f} "
              f"final={valid_traj[-1]:.3f} mean={np.mean(valid_traj):.3f}", flush=True)
        n_below = sum(g < 0.99 for g in valid_traj)
        print(f"[opcheck] gate < 0.99 on {n_below}/{len(valid_traj)} steps (warm-up ramp)", flush=True)
    else:
        print("[opcheck] gate trajectory unavailable on this bridge build (no _transmission_gain_by_name)",
              flush=True)
    print(f"[opcheck] role-bank mean firing rate: SYNAPTIC on={syn_role_on:.3f} off={syn_role_off:.3f} | "
          f"PYTHON on={py_role_on:.3f} off={py_role_off:.3f}", flush=True)
    print(f"[opcheck] role-bank drive deficit (python - synaptic): "
          f"on={py_role_on - syn_role_on:+.3f} off={py_role_off - syn_role_off:+.3f}", flush=True)


if __name__ == "__main__":
    main()
