"""B1 ON-BRIDGE V1 orientation self-org -- BCM SLIDING-THRESHOLD VARIANT.

THE WALL (research/findings/2026-08-14-b1-v1-selforg-onbridge-operating-point-BOUNDARY.md)
-------------------------------------------------------------------------------------------
  The on-bridge rate-Hebbian rule is POTENTIATION-ONLY above a FIXED coactivity threshold. On
  ON/OFF-split full-field gratings, averaged over random orientation/phase every ON and OFF
  synapse sees equal co-activation, so both potentiate to nearly identical values and the SIGNED
  ON-OFF receptive field cancels => OSI ~ 0 (measured COMMON-MODE CONVERGENCE, op-point-verified
  active-sparse, robust to fixed FS inhibition and subtractive normalization). The rule is fully
  exercised; it simply has no opponency to break.

THE MECHANISM (this variant): BCM sliding metaplastic threshold
---------------------------------------------------------------
  Bienenstock-Cooper-Munro 1982; Cooper-Intrator 2004. Add the missing INPUT-SPECIFIC DEPRESSION
  via a PER-POSTSYNAPTIC-CELL sliding threshold theta_M = <y^2> (running average of postsynaptic
  activity squared):  dw_ij = gain * x_j * y_i * (y_i - theta_M_i).
    * y_i > theta_M => LTP (the cell fired above its recent average -> strengthen the co-active input)
    * y_i < theta_M => LTD (the cell fired below -> DEPRESS the co-active input)
    * theta_M ~ <y^2> grows superlinearly with activity -> runaway potentiation is self-limited.
  A cell that (by random init) fires strongly for its preferred phase/orientation potentiates the
  co-active ON/OFF pixels there, and DEPRESSES the pixels co-active at the anti-preferred
  (contrast-reversed) phase where it fires weakly. Over development W_ON and W_OFF become spatially
  anti-correlated => a signed oriented RF. This is the classic mechanism that makes Hebbian RF
  development stable + selective, and it is exactly the input-specific depression the potentiation-
  only rule lacks -> the named fix for the common-mode boundary.

  IMPLEMENTATION: additive, guarded, default-OFF substrate primitive (sim/bridge.py + sim/config.py,
  byte-identical when hebbian_bcm=0). This runner REUSES the base on-bridge runner's build / develop /
  RF-read / OSI / RSA / controls / operating-point instrument by import, and only turns BCM on via the
  env passthrough (HEBB_BCM etc.) + a BCM-appropriate operating point (synaptic scaling OFF, since
  theta_M is itself the competitive/homeostatic normalization; no global weight decay).

THE PARTIAL (research/findings/2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md)
--------------------------------------------------------------------------------------------
  BCM decisively breaks the common mode (osi_post_frac 0.173 mean, ~62x the potentiation-only
  control) but is SEED-VARIABLE: only 3/6 seeds clear the +0.15 margin over BOTH controls (2 of
  them by ~0.32); osi_post_frac splits BIMODALLY (~0.33 strong mode vs ~0.03 weak mode) -- "the
  classic BCM/Hebbian INITIAL-CONDITION dependence" per that finding. k-WTA/fixed-lateral-inhibition
  was separately tried as a companion competition mechanism and 6-seed NO-GO'd (theorem-grounded:
  a diagonal gain-control op cannot rotate away the ON/OFF common mode's off-diagonal correlation --
  commit fa89d09b4); that lever is NOT retried here.

THE HARDENING LEVER (this file, --warmup-steps): a pre-BCM homeostatic-scaling warm-up
----------------------------------------------------------------------------------------
  See `homeostatic_warmup()` in the base runner for the full mechanism + citation (Turrigiano &
  Nelson 2004). One-line version: BCM's LTP/LTD split only produces a genuine stimulus-driven
  symmetry break when a cell's postsynaptic response starts in a workable dynamic range around
  theta_M; a cell whose RANDOM initial weight norm is by chance too small or too large starts
  outside that range and either never escapes LTD or re-saturates via runaway LTP -- an accident of
  initialization, not of the stimulus, that plausibly explains the observed bimodality. This lever
  adds an OPTIONAL pre-development phase (`--warmup-steps N`, 0 = OFF = byte-identical) that runs
  the bridge's OWN Turrigiano multiplicative synaptic-scaling mechanism with Hebbian/BCM learning
  FROZEN, equalizing each cell's overall firing-rate operating point BEFORE oriented BCM development
  begins (scaling rescales a cell's synapses UNIFORMLY, so it changes gain only, never the relative
  RF pattern BCM will read). Applied identically to the learn AND shuffle-control bridges (matched
  treatment). NOT k-WTA, NOT lateral inhibition, NOT LGN whitening.

GO BAR (unchanged, the base runner's; the spec's pre-registered margin)
-----------------------------------------------------------------------
  osi_post_frac must clear BOTH freeze (pre-random) AND shuffle by +0.15, on >= 2/3 seeds, with the
  developmental operating point active-sparse (else VOID, not scored). RSA-to-host-Gabor secondary.
  ANTI-CHEATS carried verbatim from the base runner: isotropic RF support (no orientation), random
  init (any orientation must be learned), the Gabor bank never applied, PRE/POST + SHUFFLE controls.
  INSTRUMENT CHECK: the freeze (pre) and shuffle arms must genuinely differ from the learned arm; if
  learn==freeze the test is void (the base runner reports all three).

Run (the PARTIAL, unchanged, warmup OFF):
  SIM_BACKEND=cupy python -u -m research.runners._b1_v1_selforg_bcm_derisk \
      --seeds 42 43 44 45 46 47 --dev-steps 40000 --bcm-gain 200 \
      --out research/findings/raw/_b1_v1_selforg_bcm_6seed.json

Run (the HARDENING LEVER, warmup ON):
  SIM_BACKEND=cupy python -u -m research.runners._b1_v1_selforg_bcm_derisk \
      --seeds 42 43 44 100 101 102 --dev-steps 40000 --bcm-gain 800 --bcm-pre-floor 0.002 \
      --warmup-steps 4000 \
      --out research/findings/raw/_b1_v1_selforg_bcm_warmup_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import: the base on-bridge runner supplies EVERYTHING (build_v1_bridge reads HEBB_BCM from
# env; run_seed builds both the learn + shuffle bridges through it, so setting env once covers both).
from research.runners._b1_v1_selforg_onbridge_derisk import run_seed  # noqa: E402
from sim.visual_cortex import (  # noqa: E402
    N_ORIENTATIONS, N_FREQUENCIES, V1_POSITIONS_PER_DIM, RETINA_SIZE,
)


def _phase2_pass(r):
    """Pre-registered per-seed GO: op-point active-sparse AND osi_post_frac >= 0.50 AND
    osi_post_frac >= max(pre_random, shuffle) + 0.20 (the strict lift-over-both-controls)."""
    op = r["osi"]["post_learned"]["frac_gt0_5"]
    pre = r["osi"]["pre_random"]["frac_gt0_5"]
    shuf = r["osi"]["shuffle_ctrl"]["frac_gt0_5"]
    return bool(r["op_point_ok"] and op >= 0.50 and op >= max(pre, shuf) + 0.20)


def _margin_pass(r):
    """The task's pre-registered margin: osi_post clears BOTH freeze (pre) AND shuffle by +0.15,
    op-point verified. Softer than _phase2_pass (no absolute 0.50 floor) -> a PARTIAL indicator."""
    op = r["osi"]["post_learned"]["frac_gt0_5"]
    pre = r["osi"]["pre_random"]["frac_gt0_5"]
    shuf = r["osi"]["shuffle_ctrl"]["frac_gt0_5"]
    return bool(r["op_point_ok"] and op >= pre + 0.15 and op >= shuf + 0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--n-orient", type=int, default=N_ORIENTATIONS)
    ap.add_argument("--n-freq", type=int, default=N_FREQUENCIES)
    ap.add_argument("--n-pos", type=int, default=V1_POSITIONS_PER_DIM)
    ap.add_argument("--retina-size", type=int, default=RETINA_SIZE)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--dev-steps", type=int, default=40000)
    ap.add_argument("--present-steps", type=int, default=40)
    ap.add_argument("--drive-pA", type=float, default=1200.0)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--read-steps", type=int, default=15)
    ap.add_argument("--init-weight-mean", type=float, default=30.0)
    ap.add_argument("--init-weight-jitter", type=float, default=7.0)
    # hebb_lr is UNUSED by the BCM branch (BCM has its own gain); kept for build_v1_bridge signature.
    ap.add_argument("--hebb-lr", type=float, default=0.05)
    ap.add_argument("--hebb-decay", type=float, default=0.0,
                    help="global weight decay; 0 for BCM (theta_M self-limits; decay only dilutes the signed RF)")
    ap.add_argument("--hebb-max", type=float, default=70.0)
    ap.add_argument("--coact-decay", type=float, default=0.85)
    ap.add_argument("--coact-thresh", type=float, default=0.03)
    # BCM: synaptic scaling OFF -- theta_M IS the competitive/homeostatic normalization; a second
    # multiplicative scaler double-regulates. Firing-rate op-point is held by threshold homeostasis below.
    ap.add_argument("--syn-scaling", type=int, default=0)
    ap.add_argument("--syn-scaling-rate", type=float, default=0.02)
    ap.add_argument("--n-inh", type=int, default=0)
    ap.add_argument("--inh-exc-w", type=float, default=6.0)
    ap.add_argument("--inh-inh-w", type=float, default=12.0)
    ap.add_argument("--inh-density", type=float, default=0.25)
    ap.add_argument("--homeo-target", type=float, default=0.012)
    ap.add_argument("--homeo-ema-alpha", type=float, default=0.01)
    ap.add_argument("--homeo-adapt-rate", type=float, default=0.004)
    ap.add_argument("--rule", type=str, default="hebbian", choices=["hebbian"],
                    help="BCM rides the rate-window Hebbian branch (rule=hebbian); STDP path is untouched")
    # --- BCM hyperparameters (pushed to the base build_v1_bridge via env passthrough) ---
    ap.add_argument("--bcm-gain", type=float, default=200.0,
                    help="BCM gain (multiplies phi=x*y*(y-theta_M)); 0 => OFF => the potentiation-only control")
    ap.add_argument("--bcm-theta-alpha", type=float, default=0.001,
                    help="EMA rate of the sliding threshold theta_M=<y^2> (slow vs the coactivity trace)")
    ap.add_argument("--bcm-pre-floor", type=float, default=0.02,
                    help="presynaptic-activity floor: only x_j>floor synapses change (the x_j gate)")
    # --- SEED-VARIANCE HARDENING LEVER (2026-08-27): pre-BCM homeostatic-scaling warm-up ---
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="0 (default) = OFF = byte-identical to the 2026-08-26 PARTIAL. >0 runs an "
                         "additional pre-development phase (Hebbian/BCM frozen, Turrigiano synaptic "
                         "scaling forced ON) that equalizes each V1 cell's firing-rate operating "
                         "point BEFORE oriented BCM development begins -- see homeostatic_warmup() "
                         "in the base on-bridge runner for the full mechanism + citation.")
    ap.add_argument("--dev-active-lo", type=float, default=0.005)
    ap.add_argument("--dev-active-hi", type=float, default=0.05)
    ap.add_argument("--n-categories", type=int, default=4)
    ap.add_argument("--n-exemplars", type=int, default=4)
    ap.add_argument("--n-orient-dec", type=int, default=8)
    ap.add_argument("--n-orient-ex", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_b1_v1_selforg_bcm_6seed.json")
    a = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "cupy")
    # BCM ON via the base runner's env passthrough (build_v1_bridge reads these for BOTH bridges).
    os.environ["HEBB_BCM"] = str(a.bcm_gain)
    os.environ["HEBB_BCM_THETA_ALPHA"] = str(a.bcm_theta_alpha)
    os.environ["HEBB_BCM_PRE_FLOOR"] = str(a.bcm_pre_floor)

    print(f"[B1 on-bridge V1 self-org -- BCM] seeds={a.seeds} dev_steps={a.dev_steps} "
          f"bcm_gain={a.bcm_gain} theta_alpha={a.bcm_theta_alpha} pre_floor={a.bcm_pre_floor} "
          f"warmup_steps={a.warmup_steps} "
          f"arch={a.n_orient}x{a.n_freq}x{a.n_pos}x{a.n_pos} radius={a.radius}", flush=True)

    per_seed = []
    for s in a.seeds:
        r = run_seed(s, a)
        per_seed.append(r)
        print(json.dumps(r, indent=2), flush=True)

    def col(f):
        return [f(r) for r in per_seed]

    op_ok = [bool(r["op_point_ok"]) for r in per_seed]
    n_op_ok = sum(op_ok)
    n_seeds = len(per_seed)
    n_phase2 = sum(_phase2_pass(r) for r in per_seed)     # strict GO (>=0.50 absolute)
    n_margin = sum(_margin_pass(r) for r in per_seed)     # +0.15 lift over BOTH controls

    # ---- operating-point-aware overall verdict ----
    if n_op_ok == 0:
        overall = "VOID"                                   # dead-forward: rule never fairly exercised
    elif n_phase2 >= max(2, (2 * n_seeds + 2) // 3):
        overall = "GO"                                     # cleared the strict pre-registered GO on >=2/3
    elif n_margin >= max(2, (2 * n_seeds + 2) // 3):
        overall = "PARTIAL"                                # cleared the +0.15 margin (real lift) but not 0.50
    elif n_op_ok == n_seeds:
        overall = "BOUNDARY"                               # active-sparse, below the margin -> mapped residual
    else:
        overall = "BOUNDARY-PARTIAL"

    summary = dict(
        overall_verdict=overall,
        mechanism="bcm-sliding-threshold" if a.warmup_steps == 0 else "bcm-sliding-threshold+homeostatic-warmup",
        bcm=dict(gain=a.bcm_gain, theta_alpha=a.bcm_theta_alpha, pre_floor=a.bcm_pre_floor),
        warmup_steps=a.warmup_steps,
        seeds=a.seeds,
        per_seed_verdicts=[r["verdict"] for r in per_seed],
        op_point_ok=op_ok,
        n_op_point_verified=n_op_ok,
        n_phase2_go_seeds=n_phase2,
        n_margin_pass_seeds=n_margin,
        dev_firing_fraction=[r["dev_firing_fraction"] for r in per_seed],
        dev_firing_fraction_mean=round(float(np.mean(col(lambda r: r["dev_firing_fraction"]))), 5),
        dev_active_band=[a.dev_active_lo, a.dev_active_hi],
        osi_pre_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["pre_random"]["frac_gt0_5"]))), 4),
        osi_post_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["post_learned"]["frac_gt0_5"]))), 4),
        osi_shuffle_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["shuffle_ctrl"]["frac_gt0_5"]))), 4),
        osi_post_frac_per_seed=col(lambda r: r["osi"]["post_learned"]["frac_gt0_5"]),
        osi_pre_frac_per_seed=col(lambda r: r["osi"]["pre_random"]["frac_gt0_5"]),
        osi_shuffle_frac_per_seed=col(lambda r: r["osi"]["shuffle_ctrl"]["frac_gt0_5"]),
        osi_post_mean_mean=round(float(np.mean(col(lambda r: r["osi"]["post_learned"]["mean"]))), 4),
        rsa_vs_host_mean=round(float(np.mean(col(lambda r: r["geometry"]["v1_firing_post"]["rsa_vs_host"]))), 4),
        margin_mean=round(float(np.mean(col(lambda r: r["geometry"]["v1_firing_post"]["margin"]))), 4),
        orient_decode_mean=round(float(np.mean(col(lambda r: r["geometry"]["v1_firing_post"]["orient_decode"]))), 4),
        host_decode_mean=round(float(np.mean(col(lambda r: r["geometry"]["host_reference"]["orient_decode"]))), 4),
        v1_firing_rate_mean=round(float(np.mean(col(lambda r: r["v1_firing_rate"]))), 4),
        weight_diagnosis=[r["weight_diagnosis"] for r in per_seed],
    )

    out = dict(summary=summary, per_seed=per_seed)
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 90, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[written] {outp}", flush=True)


if __name__ == "__main__":
    main()
