"""Quick CPU diagnostic for the D2 Phase-2 forward-pass NEGATIVE: localize WHERE the structure is lost
(hub layer vs readout) and whether the per-hub EMAs/gains differentiate common vs category hubs at read
time. Reuses the Phase-2 runner's bridge + presentation. SIM_BACKEND=numpy, seconds."""
from __future__ import annotations
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners.dendritic_d1_learn_graded_structure_derisk import build_concept_hub_counts, _cos_sim, _pearson_vs_Strue
from research.runners.dendritic_cortex_forward_codes_derisk import _build_cortex_bridge, _present


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seed = 42
    C, labels, S_true, _ = build_concept_hub_counts(8, 8, 200, 12, 40.0, 4.0, 0.3, seed)
    Nc, n_hub = C.shape
    n_common = 200  # the first n_common hubs are the COMMON (high-freq) ones; the rest are category-signal
    raw = "--raw" in sys.argv
    drive_scale = float([a for a in sys.argv if a.replace('.', '').isdigit()][0]) if any(
        a.replace('.', '').isdigit() for a in sys.argv) else (3.0 if raw else 20.0)
    # RAW count drive (the common-mode-dominated regime where the gain MATTERS) vs PRESENCE (binary).
    C_drive = C if raw else (C > 1.5).astype(np.float64)
    window, settle, warmup, alpha, sigma = 20, 6, 8, 0.002, 0.05
    print(f"[drive={'RAW counts' if raw else 'presence'} scale={drive_scale} alpha={alpha} warmup={warmup}]")

    bridge, hub_idx, ro_idx = _build_cortex_bridge(n_hub, 200, seed, True, sigma, alpha, 0.1)
    hub_idx = np.asarray(hub_idx); ro_idx = np.asarray(ro_idx)

    # warm-up (converge the per-hub EMAs to the marginals)
    for _ in range(warmup):
        for i in range(Nc):
            _present(bridge, hub_idx, ro_idx, C_drive[i], drive_scale, window, settle)

    # --- inspect the per-hub EMA + gain after warm-up (common vs category hubs) ---
    ema = np.asarray(bridge.cp_dendritic_source_activity)[hub_idx]
    gain = sigma / (sigma + ema)
    common_ema, cat_ema = ema[:n_common].mean(), ema[n_common:].mean()
    common_gain, cat_gain = gain[:n_common].mean(), gain[n_common:].mean()
    print(f"[per-hub EMA after warm-up]  common={common_ema:.4f}  category={cat_ema:.4f}  "
          f"(want common >> category for marginal normalization)")
    print(f"[per-hub GAIN sigma/(sigma+ema)] common={common_gain:.4f}  category={cat_gain:.4f}  "
          f"(want common << category)")

    # --- hub FIRING profile per concept (is the hub layer's firing structured by category?) ---
    # read hub firing directly (drive each concept, read hub region firing)
    hub_codes = np.zeros((Nc, hub_idx.size))
    ro_codes = np.zeros((Nc, ro_idx.size))
    for i in range(Nc):
        # read hub firing
        import numpy as _np
        bridge.cp_external_input_current[:] = 0.0
        drv = np.zeros(int(bridge.cp_membrane_potential_v.shape[0]))
        drv[hub_idx] = C_drive[i] * drive_scale
        if hasattr(bridge, "_cp") and bridge._cp is not None:
            bridge.cp_external_input_current[hub_idx] = bridge._cp.asarray(drv[hub_idx].astype(np.float32))
        else:
            bridge.cp_external_input_current[hub_idx] = drv[hub_idx].astype(np.float32)
        hacc = np.zeros(hub_idx.size); racc = np.zeros(ro_idx.size); ns = 0
        for t in range(settle + window):
            bridge._run_one_simulation_step()
            if t >= settle:
                hacc += np.asarray(bridge.cp_firing_states)[hub_idx].astype(np.float64)
                racc += np.asarray(bridge.cp_conductance_g_e)[ro_idx].astype(np.float64)
                ns += 1
        bridge.cp_external_input_current[:] = 0.0
        hub_codes[i] = hacc / max(1, ns)
        ro_codes[i] = racc / max(1, ns)

    # structure of the hub-firing profiles, the gain-weighted hub profiles, and the readout codes
    hub_pearson = _pearson_vs_Strue(_cos_sim(hub_codes), S_true)
    gainw = hub_codes * gain[None, :]                      # what the readout SHOULD project
    gainw_pearson = _pearson_vs_Strue(_cos_sim(gainw), S_true)
    raw_drive_pearson = _pearson_vs_Strue(_cos_sim(C_drive), S_true)
    gainw_drive = C_drive * gain[None, :]
    gainw_drive_pearson = _pearson_vs_Strue(_cos_sim(gainw_drive), S_true)
    ro_pearson = _pearson_vs_Strue(_cos_sim(ro_codes), S_true)
    print(f"[presence-drive pattern cosine] Pearson(sim,S_true)={raw_drive_pearson:+.3f}  "
          f"(gain-weighted DRIVE {gainw_drive_pearson:+.3f})")
    print(f"[HUB firing profile]            Pearson={hub_pearson:+.3f}  "
          f"(gain-weighted HUB firing {gainw_pearson:+.3f} = what a faithful readout would project)")
    print(f"[READOUT g_e code]              Pearson={ro_pearson:+.3f}")
    print("\nLOCALIZATION:")
    if common_ema <= cat_ema + 1e-4:
        print("  -> the per-hub EMA does NOT differentiate common vs category (marginal normalization "
              "broken: EMA tracks the current concept, not the marginal) => fix the EMA/warm-up.")
    elif gainw_drive_pearson < raw_drive_pearson + 0.1:
        print("  -> the gain-weighted DRIVE doesn't beat the raw drive => the gain values aren't recovering "
              "structure even in the ideal (analog) case => the presence-drive code itself is the issue.")
    elif hub_pearson < gainw_drive_pearson - 0.1:
        print("  -> the HUB LAYER firing loses the structure (spiking threshold/saturation) => fix the hub "
              "encoding so firing tracks the drive.")
    elif ro_pearson < hub_pearson - 0.1:
        print("  -> the READOUT projection/dynamics scramble the structure the hub layer carried => fix the "
              "readout (attractor/cleanup or a structure-preserving read).")
    else:
        print("  -> structure is weak at every stage; reconsider the toy/drive calibration.")


if __name__ == "__main__":
    main()
