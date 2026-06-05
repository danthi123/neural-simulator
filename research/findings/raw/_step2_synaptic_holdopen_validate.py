"""Validate the held-open pre-warm across ALL seed-42 facts (the 3 'come' patients that regressed + the
others), with the production pre-window protocol = "warm the parser conjunction until the gate reads open
(parser has opened it), up to a cap, then hold the parser-opened gate during the readout window".

For each of the 6 seed-42 facts, store via the held-open synaptic protocol and check query_patient and
query_agent, comparing to the Python path. Confirms (a) the 3 come-patients now decode, (b) no other fact
regresses, (c) the gate reads ~1.0 during the readout (opcheck-style).

    python -m research.findings.raw._step2_synaptic_holdopen_validate
"""
from __future__ import annotations

import numpy as np

from research.runners.unified_brain_bridge import UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES, ROLE_SRC_DRIVE_PA
from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE, RESET_STEPS
from sim.backend import get_backend, to_host

SEED = 42; PROJ_DIM = 2048
PREWARM_CAP = 60          # max pre-window steps to wait for the parser to open the gate


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


def _op_holdopen(u, conj_k, fon, foff, report_gate=False):
    xp, _ = get_backend()
    bridge = u.bridge; comp = u.composer; idx = comp.idx
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
    cur = _drive(u, xp, bridge, comp, idx, conj_k, fon, foff)
    bridge.cp_external_input_current[:] = cur
    # PRE-WARM: run until every coupled gate has been opened by the parser's firing (cap PREWARM_CAP).
    role_gate_names = [f"role_route_{r}" for r in SYNAPTIC_ROUTE_ROLES]
    def any_open():
        for gn in role_gate_names:
            syn = bridge._transmission_gate_to_synapses.get(gn)
            if syn is not None and float(to_host(bridge.cp_transmission_gain[syn]).mean()) >= 0.99:
                return True
        return False
    pre = 0
    while pre < PREWARM_CAP:
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        pre += 1
        if any_open():
            break
    # HOLD the parser-opened gate during readout: remove the coupling so it is not re-evaluated.
    saved = bridge._gate_couplings
    bridge._gate_couplings = []
    gate_below = 0; gsum = 0.0
    acc = {b: xp.zeros(comp.D, dtype=xp.float64) for b in ("A", "B", "C", "D")}
    role_gate = bridge._transmission_gate_to_synapses.get("role_route_patient")
    for _ in range(comp.run_steps):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        for b in ("A", "B", "C", "D"):
            acc[b] += bridge.cp_firing_states[idx[b]].astype(xp.float64)
        if report_gate and role_gate is not None:
            gv = float(to_host(bridge.cp_transmission_gain[role_gate]).mean())
            gsum += gv
            if gv < 0.99:
                gate_below += 1
    bridge._gate_couplings = saved
    for r in SYNAPTIC_ROUTE_ROLES:                       # restore the closed default for the next op
        bridge.set_transmission_gate(f"role_route_{r}", 0.0)
        cpl = next((c for c in bridge._gate_couplings if c["gate_name"] == f"role_route_{r}"), None)
        if cpl is not None:
            cpl["ema"] = 0.0; cpl["last_value"] = None
    bridge.cp_external_input_current[:] = 0.0
    rates = {b: to_host(acc[b]) / comp.run_steps for b in ("A", "B", "C", "D")}
    out = (rates["A"] + rates["B"], rates["C"] + rates["D"])
    if report_gate:
        return out, pre, gate_below, (gsum / comp.run_steps if comp.run_steps else 0.0)
    return out, pre, None, None


def _store_holdopen(u, sentence, report_patient_gate=False):
    xp, _ = get_backend()
    comp = u.composer
    words = sentence.split(); v = 0
    bound_on = np.zeros(comp.D); bound_off = np.zeros(comp.D)
    fact = {}
    info = None
    for pos in range(3):
        word = words[pos]; k = pos * 2 + v
        role = u.parser.role_of(pos, "active")
        fact[role] = word
        c_on, c_off = onoff(comp.concepts[word])
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        (o, f), pre, gb, gm = _op_holdopen(u, k, fon, foff,
                                           report_gate=(report_patient_gate and role == "patient"))
        bound_on += o; bound_off += f
        if report_patient_gate and role == "patient":
            info = (pre, gb, gm)
    comp.kb.append((fact, onoff(bound_on - bound_off)))
    return info


def main():
    u = UnifiedBrainBridge(seed=SEED, proj_dim=PROJ_DIM, enable_synaptic_route=True)
    rng = np.random.default_rng(SEED + 1)
    facts = [tuple(str(x) for x in rng.choice(u.words, size=3, replace=False)) for _ in range(6)]
    print(f"[holdopen] seed-42 facts: {facts}", flush=True)

    what_ok = who_ok = 0
    for i, (a, ac, p) in enumerate(facts):
        u.kb = []
        info = _store_holdopen(u, f"{a} {ac} {p}", report_patient_gate=True)
        dw = u.query_patient(a, ac); da = u.query_agent(ac, p)
        wok = (dw == p); aok = (da == a)
        what_ok += int(wok); who_ok += int(aok)
        pre, gb, gm = info if info else (None, None, None)
        print(f"[holdopen] fact{i} {a:>6} {ac:>6} {p:>6} | what={dw!r}{'' if wok else ' X'} "
              f"who={da!r}{'' if aok else ' X'} | patient pre={pre} gate<0.99={gb}/150 gate_mean={gm:.3f}",
              flush=True)
    print(f"[holdopen] TOTAL what={what_ok}/6 who={who_ok}/6  (Python path is 6/6 both; "
          f"target: what within +-1 of 6 => >=5, ideally 6)", flush=True)


if __name__ == "__main__":
    main()
