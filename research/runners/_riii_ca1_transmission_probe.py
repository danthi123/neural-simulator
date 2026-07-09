"""R-iii ca1-transmission ISOLATION probe (CYCLE 1076 follow-on): the SWR gen-replay Rung 1 found ca1 does NOT fire
from the completed CA3 assembly even with a 15x Schaffer boost + a synchronous burst (ca1_fire stuck ~5-7 = noise
floor). Isolate the variable: drive a fixed set of CA3 cells DIRECTLY + HARD (no formation, no completion) with the
Schaffer ca3->ca1 boosted, and read ca1. If ca1 fires >> baseline -> transmission works and the recall's CA3 firing
was the issue; if ca1 ~ baseline -> the block is ca1's own rheobase / feedback inhibition / a closed gate. Reuse-by-
import of _build + _scale_pathway. NO `sim/` edit."""
from __future__ import annotations
import argparse
import numpy as np
from research.runners._riii_ca3_coincidence_completion_derisk import _build
from research.runners._riii_swr_generative_replay_derisk import _scale_pathway


def run(seed=42, n_ca3=500, n_drive=40, drive_pA=3000.0, schaffer_boost=15.0, steps=60):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = _build(seed, n_ca3=n_ca3, ca3_density=0.5, ca3w=6.0, coincidence=False, weighted=True, train=False)
    rm = bridge.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    ca1_idx = np.asarray(list(rm.indices("ca1")), dtype=np.int64)
    n_sch = _scale_pathway(bridge, cp, ca3_idx, ca1_idx, schaffer_boost)
    drv = cp.asarray(ca3_idx[:n_drive], dtype=cp.int64)
    ca1 = cp.asarray(ca1_idx, dtype=cp.int64)
    ca3d = cp.asarray(ca3_idx[:n_drive], dtype=cp.int64)

    def _measure(drive_on):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()
        ca1_acc = 0.0; ca3_acc = 0.0
        for _ in range(steps):
            bridge.cp_external_input_current[:] = 0.0
            if drive_on:
                bridge.cp_external_input_current[drv] = float(drive_pA)
            bridge._run_one_simulation_step()
            ca1_acc += float(cp.sum(bridge.cp_firing_states[ca1]))
            ca3_acc += float(cp.sum(bridge.cp_firing_states[ca3d]))
        bridge.cp_external_input_current[:] = 0.0
        return ca1_acc, ca3_acc

    base_ca1, _ = _measure(False)
    drive_ca1, drive_ca3 = _measure(True)
    print(f"[ca1-transmission probe] seed {seed} n_drive={n_drive} drive={drive_pA}pA schaffer x{schaffer_boost} "
          f"({n_sch} edges)", flush=True)
    print(f"  CA3 driven-cells fired = {drive_ca3:.0f} spikes over {steps} steps (is the source firing?)", flush=True)
    print(f"  ca1 baseline (no drive) = {base_ca1:.0f} | ca1 WITH CA3 drive = {drive_ca1:.0f}", flush=True)
    verdict = ("TRANSMISSION OK -- ca1 fires from a hard direct CA3 drive; the Rung-1 issue is the recall's CA3 firing being too weak/brief"
               if drive_ca1 > 3 * (base_ca1 + 1) else
               "ca1 BLOCKED -- even a hard direct CA3 drive does not fire ca1; the block is ca1 rheobase / feedback inhibition / a closed transmission gate (NOT the Schaffer weight or the drive strength)")
    print(f"  VERDICT: {verdict}", flush=True)
    return {"base_ca1": base_ca1, "drive_ca1": drive_ca1, "drive_ca3": drive_ca3, "n_sch": n_sch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-drive", type=int, default=40)
    ap.add_argument("--drive-pA", type=float, default=3000.0)
    ap.add_argument("--schaffer-boost", type=float, default=15.0)
    a = ap.parse_args()
    run(seed=a.seed, n_drive=a.n_drive, drive_pA=a.drive_pA, schaffer_boost=a.schaffer_boost)


if __name__ == "__main__":
    main()
