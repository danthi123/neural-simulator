"""In-substrate GRADED coincidence gating -- the second primitive for the bind.

The compositional bind preserves filler MAGNITUDE: bound[i] = role[i]*filler[i] with
role in {+1,-1}, so bound[i] = +-filler[i] keeps the graded filler value. The substrate
concept codes are graded (real-valued firing-rate deviations), not binary. So the
coincidence operator must be GRADED: when the role gates it ON, the coinc firing rate
must scale with the filler drive (monotonic); when the role is OFF, the coinc must stay
silent regardless of filler level.

This is multiplicative gain-gating (role multiplies filler) -- well-documented biology
(attentional gain, dendritic gating). Reuses the validated coincidence build()
(role/filler/coinc, identity wiring, tonic bias) from _insubstrate_coincidence_probe.

FROZEN read: GRADED-GATING WORKS if, at the operating point (w=320, bias=-1000),
(a) role-ON coinc rate increases monotonically with filler level (Spearman ~1),
(b) role-OFF coinc rate stays ~0 (<= 0.01) at every filler level, and
(c) the role-ON rate at max filler is clearly above the role-OFF rate (gating ratio
high). Then the spiking bind preserves graded filler magnitude -> cleanup works.

VALIDATED 2026-05-31 (RTX 3090, CuPy), seed 42, w=320 bias=-1000:
  filler level  0.00 0.25 0.50 0.75 1.00
  role-ON  rate 0.000 0.011 0.028 0.040 0.048   (Spearman 1.000 -- monotonic)
  role-OFF rate 0.000 0.000 0.000 0.000 0.000   (perfect gating; ratio ~inf)
The coincidence operator is a clean multiplicative gate: role gates, filler provides
graded drive, coinc rate ~ filler magnitude when gated, silent when not. The spiking
bind preserves graded filler magnitude -> cleanup works.

stdlib+numpy + the project bridge; no protected-module modification.
"""
from __future__ import annotations
import numpy as np

from sim.backend import get_backend, to_host
from research.findings.raw._insubstrate_coincidence_probe import build, N, DRIVE_PA, RESET_STEPS, RUN_STEPS


def measure_graded(bridge, role, fill, coinc, role_on, fill_idx, fill_level, xp, coinc_bias):
    """Drive role neurons (binary, full) in role_on; drive filler neurons in fill_idx at
    fill_level*DRIVE_PA (graded); hold coinc_bias on coinc. Return mean coinc rate over coinc
    neurons whose index is in fill_idx (the gated dims)."""
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    coinc_arr = xp.asarray(coinc, dtype=xp.int64)
    if role_on:
        r_idx = xp.asarray([role[i] for i in role_on], dtype=xp.int64)
        bridge.cp_external_input_current[r_idx] = DRIVE_PA
    if fill_idx:
        f_idx = xp.asarray([fill[i] for i in fill_idx], dtype=xp.int64)
        bridge.cp_external_input_current[f_idx] = fill_level * DRIVE_PA   # GRADED filler drive
    if coinc_bias != 0.0:
        bridge.cp_external_input_current[coinc_arr] = coinc_bias
    c = xp.zeros(N, dtype=xp.float64)
    for _ in range(RUN_STEPS):
        bridge._run_one_simulation_step()
        c += bridge.cp_firing_states[coinc_arr].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    rate = to_host(c) / RUN_STEPS
    return float(np.mean(rate[fill_idx]))   # coinc rate at the gated dims


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / d) if d > 1e-9 else 0.0


def main():
    xp, backend = get_backend()
    W, BIAS = 320.0, -1000.0      # validated AND operating point (perfect single rejection)
    print(f"=== in-substrate GRADED coincidence gating (backend={backend}, N={N}, "
          f"w={W}, bias={BIAS}) ===", flush=True)
    # gated dims = a block where BOTH role and filler can be active; we vary filler level.
    gated = list(range(N // 4, N // 2))     # role always ON here; sweep filler level
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    on_rates, off_rates = [], []
    for lvl in levels:
        # role ON: role active on gated dims; filler at lvl
        b1, r1, f1, c1 = build(42, W)
        on = measure_graded(b1, r1, f1, c1, gated, gated, lvl, xp, BIAS); del b1
        # role OFF: role NOT driven; filler at lvl (control -- should stay silent)
        b2, r2, f2, c2 = build(42, W)
        off = measure_graded(b2, r2, f2, c2, [], gated, lvl, xp, BIAS); del b2
        on_rates.append(on); off_rates.append(off)
        print(f"  filler={lvl:.2f} | role-ON coinc={on:.3f}   role-OFF coinc={off:.3f}", flush=True)

    on_rates = np.array(on_rates); off_rates = np.array(off_rates)
    rho = spearman(np.array(levels), on_rates)
    off_max = float(off_rates.max())
    ratio = on_rates[-1] / (off_rates[-1] + 1e-9)
    mono = rho >= 0.9
    gated_off = off_max <= 0.01
    sep = on_rates[-1] >= 0.03 and ratio >= 5.0
    print(f"\nSpearman(filler-level, role-ON rate)={rho:.3f} (mono>=0.9: {mono})", flush=True)
    print(f"role-OFF max rate={off_max:.3f} (gated-off<=0.01: {gated_off})", flush=True)
    print(f"gating ratio (ON/OFF at max filler)={ratio:.1f} (sep>=5 & ON>=0.03: {sep})", flush=True)
    if mono and gated_off and sep:
        print("VERDICT: GRADED-GATING WORKS -- coinc rate scales with filler when role gates ON, "
              "silent when role OFF. The spiking bind preserves graded filler magnitude.", flush=True)
    else:
        print("VERDICT: needs tuning -- adjust w/bias/drive so the role-ON regime is graded "
              "(rate ~ filler) while role-OFF stays silent.", flush=True)


if __name__ == "__main__":
    main()
