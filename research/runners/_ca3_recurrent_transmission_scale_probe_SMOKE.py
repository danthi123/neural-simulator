"""MINIMAL CA3 recurrent transmission instrument (NEW runner, no sim/ edit).

Confirms/refutes the 2026-07-08 claim that the ca3->ca3 recurrents are
"functionally silent / ~1000x too weak / weight-invariant"
(2026-07-08-riii-CORRECTION-ca3-recurrents-functionally-silent-not-point-neuron-limit.md),
which the 2026-08-10 parallel-derisk batch re-cited as the episodic-recall
sim-internals wall.

Method (direct-delivery probe, the class the 2026-07-17 finding used):
  - Build the biological brain (numpy) with a chosen ca3->ca3 recurrent WEIGHT.
  - Freeze plasticity (weights cannot drift during the probe).
  - Drive a fixed CA3 DRIVER assembly with strong external current so it fires.
  - Measure the g_e and Vm DELIVERED to the NON-driven CA3 recurrent TARGETS
    (targets receive ZERO external current, so their g_e comes ONLY from the
    ca3->ca3 recurrent path from the drivers).
  - Sweep weight in {0, 1, 10, 100, 1000}.

Skeptical controls (specificity / no-lever):
  - weight=0  -> targets should receive ~0 g_e and ~0 mV (the recurrent path is
    the source; if a "silent" reading is a floor artifact, w=0 pins the floor).
  - MONOTONE SCALING of delivered g_e with weight refutes "weight-invariant";
    a delivered mV that is NOT ~1000x too weak refutes "functionally silent".

Cheapest decisive test: single seed, ~1400-neuron numpy net, one process, <~2 min.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np


def _build(seed, ca3w, n_lang=384, n_ec=160, n_dg=300, n_ca3=150, n_ca1=120, ca3_density=0.30):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang, n_motor_per_action=16, n_motor_fs_per_action=4, enable_motor_fs=True,
        enable_language_output=True, n_lang_output=n_lang, enable_hippocampus_consolidation=True,
        n_ec=n_ec, n_dg=n_dg, n_ca3=n_ca3, n_ca1=n_ca1, ca3_recurrent_density=ca3_density,
        ca3_recurrent_weight=float(ca3w))   # train=True path: weight IS applied at construction
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0; cfg.seed = seed; cfg.enable_nmda = True
    # FREEZE plasticity so the swept weight cannot drift during the probe.
    cfg.enable_structural_plasticity = False; cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = max(10.0, 2.5 * max(1.0, ca3w)); cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _mean_abs_ca3_recurrent_weight(bridge, ca3_idx):
    """Verify the swept weight is actually applied to the ca3->ca3 block."""
    try:
        C = bridge.cp_connections
        try:
            import scipy.sparse as sp
            C = sp.csr_matrix(C)
        except Exception:
            pass
        idx = np.asarray(ca3_idx, dtype=np.int64)
        blk = C[idx, :][:, idx]
        data = np.asarray(blk.data, dtype=np.float64)
        data = data[np.abs(data) > 0]
        if data.size == 0:
            return 0.0, 0
        return float(np.mean(np.abs(data))), int(data.size)
    except Exception as e:  # pragma: no cover - diagnostic only
        return float("nan"), -1


def run_weight(seed, ca3w, drive_frac=0.40, drive_pA=350.0, reset_steps=25, drive_steps=45):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = _build(seed, ca3w)
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3"))
    n_ca3 = len(ca3_idx)
    n_drive = max(1, int(round(drive_frac * n_ca3)))
    drivers = np.asarray(ca3_idx[:n_drive], dtype=np.int64)
    targets = np.asarray(ca3_idx[n_drive:], dtype=np.int64)
    drv = cp.asarray(drivers, dtype=cp.int64)
    tgt = cp.asarray(targets, dtype=cp.int64)

    meanw, nsyn = _mean_abs_ca3_recurrent_weight(bridge, ca3_idx)

    # Reset transients (no drive).
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    rest_vm = float(to_host(cp.mean(bridge.cp_membrane_potential_v[tgt])))

    # Drive the driver assembly; measure delivery to the non-driven targets.
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[drv] = cp.float32(drive_pA)

    n_t = int(targets.size)
    tgt_max_vm = cp.full(n_t, -1e9, dtype=cp.float32)   # per-target peak Vm over the window
    tgt_ever_fired = cp.zeros(n_t, dtype=cp.bool_)
    peak_mean_ge = 0.0
    driver_spikes = 0.0
    target_spikes = 0.0
    for _ in range(drive_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        vm_t = bridge.cp_membrane_potential_v[tgt]
        ge_t = bridge.cp_conductance_g_e[tgt]
        tgt_max_vm = cp.maximum(tgt_max_vm, vm_t)
        fired_t = bridge.cp_firing_states[tgt].astype(cp.bool_)
        tgt_ever_fired = tgt_ever_fired | fired_t
        target_spikes += float(to_host(cp.sum(bridge.cp_firing_states[tgt].astype(cp.float32))))
        driver_spikes += float(to_host(cp.sum(bridge.cp_firing_states[drv].astype(cp.float32))))
        peak_mean_ge = max(peak_mean_ge, float(to_host(cp.mean(ge_t))))
    bridge.cp_external_input_current[drv] = 0.0

    ever = to_host(tgt_ever_fired).astype(bool)
    maxvm = to_host(tgt_max_vm).astype(np.float64)
    non_fired = ~ever
    # Subthreshold delivered depolarization = peak Vm of targets that NEVER fired, minus rest.
    if non_fired.any():
        dvm_subthresh = float(np.mean(maxvm[non_fired]) - rest_vm)
        dvm_subthresh_max = float(np.max(maxvm[non_fired]) - rest_vm)
    else:
        dvm_subthresh = float("nan"); dvm_subthresh_max = float("nan")
    recruited_frac = float(ever.mean())
    return {
        "ca3_weight": float(ca3w),
        "mean_abs_recurrent_w": meanw,
        "n_ca3_recurrent_syn": nsyn,
        "n_drivers": int(drivers.size), "n_targets": int(targets.size),
        "driver_spikes": driver_spikes, "target_spikes": target_spikes,
        "rest_vm": rest_vm,
        "peak_mean_target_g_e": peak_mean_ge,
        "delivered_dVm_subthresh_mean": dvm_subthresh,
        "delivered_dVm_subthresh_max": dvm_subthresh_max,
        "recruited_frac_of_targets": recruited_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=str, default="0,1,10,100,1000")
    ap.add_argument("--drive-pA", type=float, default=350.0)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    weights = [float(x) for x in a.weights.split(",") if x.strip() != ""]
    t0 = time.time()
    rows = []
    print(f"[CA3 recurrent transmission scale probe] seed={a.seed} weights={weights} drive_pA={a.drive_pA}", flush=True)
    for w in weights:
        r = run_weight(a.seed, w, drive_pA=a.drive_pA)
        rows.append(r)
        print(f"  w={w:>7.1f} | mean|w|={r['mean_abs_recurrent_w']:.3f} nsyn={r['n_ca3_recurrent_syn']} "
              f"| driver_spk={r['driver_spikes']:.0f} tgt_spk={r['target_spikes']:.0f} "
              f"| peak g_e(tgt)={r['peak_mean_target_g_e']:.4f} "
              f"| delivered dVm(sub)={r['delivered_dVm_subthresh_mean']:.3f} mV (max {r['delivered_dVm_subthresh_max']:.3f}) "
              f"| recruited={r['recruited_frac_of_targets']*100:.1f}%", flush=True)

    # Verdict logic: transmission SCALES (not silent/weight-invariant) if delivered g_e
    # to non-driven targets rises monotonically and w=0 pins the floor near zero.
    by_w = {r["ca3_weight"]: r for r in rows}
    ge = [(r["ca3_weight"], r["peak_mean_target_g_e"]) for r in rows]
    ge.sort()
    floor = by_w.get(0.0, {}).get("peak_mean_target_g_e", None)
    monotone = all(ge[i][1] <= ge[i + 1][1] + 1e-6 for i in range(len(ge) - 1))
    span = (ge[-1][1] / max(ge[0][1], 1e-9)) if len(ge) >= 2 else float("nan")
    print("\n  === VERDICT ===", flush=True)
    print(f"  peak g_e monotone-increasing with weight: {monotone}", flush=True)
    if floor is not None:
        print(f"  w=0 floor peak g_e = {floor:.5f} (specificity control: should be ~0)", flush=True)
    print(f"  g_e span (max weight / min weight): {span:.1f}x", flush=True)
    print(f"  => 'weight-invariant / functionally silent' is {'REFUTED' if (monotone and span > 5) else 'NOT refuted by this probe'}", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"seed": a.seed, "rows": rows, "monotone": monotone, "span": span, "w0_floor_ge": floor}, f, indent=2)
        print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
