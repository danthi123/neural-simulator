"""De-risk 5b (EDITED version) — validate the sliced-RF-ops co-residence edit.

After the protected `sim/bridge.py` edit (optional `neuron_mask` on `rf_kick` + `_rf_advance_one`), this
proves the two PASS criteria for strict RF co-residence on one bridge:

  (1) CORRECTNESS — an RF op run on a MASKED slice of an Izhikevich bridge reproduces the result of the same
      op on a standalone RESONATE_AND_FIRE bridge (the mask does not corrupt the RF dynamics for the masked
      neurons). Tested by free-resonate phase recovery: kick known phases onto the RF slice, resonate, read
      back -> must equal the standalone bridge's read-back exactly.

  (2) ISOLATION — a co-resident Izhikevich (navigation) slice's `v`/`u` are BYTE-IDENTICAL across the RF op
      (the masked RF ops never touch the navigation slice). This is the thing the as-is path could not give
      (one Izhikevich step destroys a phasor; here, conversely, the RF op must not touch the Izhikevich slice).

The complex-synapse FHRR bind/unbind path is already validated on 100%-RF bridges by the 18 verbatim-passing
conversational tests (mask=None); the mask only changes the write-back, which free-resonate exercises directly.

Run on GPU (CuPy).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.backend import get_backend, to_host
from research.runners.rf_phasor_composer import _build_rf_bridge
from research.runners.derisk_unification_5b_rf_izh_coexistence import build_izh_bridge


def main():
    ap = argparse.ArgumentParser(description="Unification de-risk 5b (EDITED): sliced-RF-ops co-residence")
    ap.add_argument("--n-izh", type=int, default=24, help="navigation (Izhikevich) slice size")
    ap.add_argument("--n-rf", type=int, default=8, help="RF (composer) slice size")
    ap.add_argument("--period", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/derisk_5b_edited.json")
    args = ap.parse_args()

    xp, backend = get_backend()
    I, R, P = args.n_izh, args.n_rf, args.period
    N = I + R
    print(f"[5b-edited] backend={backend} N={N} (izh={I} + rf={R}) period={P} seed={args.seed}")

    # known phases for the RF slice
    phases = (np.arange(R) + 0.5) / R          # 1/2R, 3/2R, ... (avoid exact 0 boundary)
    kick_R = np.exp(2j * np.pi * phases)

    # --- (A) standalone RF reference ---
    ref = _build_rf_bridge(R, seed=args.seed)
    ref.rf_kick(kick_R, period=P, lam=0.0)
    ref.rf_resonate_steps(P + 8)
    standalone_phases = ref.rf_read_phases()    # length R

    # --- (B) mixed Izhikevich bridge with a masked RF slice [I:I+R] ---
    mix = build_izh_bridge(N, seed=args.seed)
    # give the Izhikevich (navigation) slice genuine dynamics state: drive it + step the Izhikevich loop.
    izh_idx = xp.asarray(np.arange(0, I), dtype=xp.int64)
    for _ in range(20):
        mix.cp_external_input_current[:] = 0.0
        mix.cp_external_input_current[izh_idx] = 600.0
        mix._run_one_simulation_step()           # Izhikevich over all N (RF slice gets garbage; re-kicked below)
    mix.cp_external_input_current[:] = 0.0
    izh_v_before = to_host(mix.cp_membrane_potential_v[:I]).copy()
    izh_u_before = to_host(mix.cp_recovery_variable_u[:I]).copy()

    # the RF op on the masked slice (the composer's per-op pattern, but sliced onto the shared bridge)
    mask = np.zeros(N, dtype=bool)
    mask[I:I + R] = True
    full_kick = np.zeros(N, dtype=np.complex128)
    full_kick[I:I + R] = kick_R
    mix.rf_kick(full_kick, period=P, lam=0.0, neuron_mask=mask)
    mix.rf_resonate_steps(P + 8)                 # masked _rf_advance_one: only the RF slice advances
    mixed_phases = mix.rf_read_phases()[I:I + R]

    izh_v_after = to_host(mix.cp_membrane_potential_v[:I])
    izh_u_after = to_host(mix.cp_recovery_variable_u[:I])

    # --- verdict ---
    phase_dev = float(np.max(np.abs(mixed_phases - standalone_phases)))
    phase_match = phase_dev <= 1e-6
    izh_isolated = bool(np.array_equal(izh_v_before, izh_v_after)
                        and np.array_equal(izh_u_before, izh_u_after))
    # also: the RF slice's v should now be off-rest (it ran RF), confirming the op actually ran
    rf_ran = float(np.max(np.abs(to_host(mix.cp_membrane_potential_v[I:I + R])))) > 1e-6

    passed = phase_match and izh_isolated and rf_ran

    print("\n=== De-risk 5b (EDITED) verdict ===")
    print(f"(1) RF-on-slice phases == standalone : {phase_match}  (max|d phase| = {phase_dev:.3e})")
    print(f"(2) Izhikevich slice byte-isolated   : {izh_isolated}  (v+u identical across the RF op)")
    print(f"    RF op actually ran on the slice  : {rf_ran}")
    print(f"\n[5b-edited] {'PASS' if passed else 'FAIL'} — the sliced-RF-ops edit gives strict RF co-residence: "
          f"the RF op on the masked slice matches a standalone RF bridge AND leaves the Izhikevich slice untouched.")

    result = {
        "derisk": "5b_edited_sliced_rf_ops",
        "backend": backend, "n_izh": I, "n_rf": R, "period": P, "seed": args.seed,
        "phase_dev_max": phase_dev, "phase_match": phase_match,
        "izh_isolated": izh_isolated, "rf_ran": rf_ran, "pass": passed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[5b-edited] wrote {args.out}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
