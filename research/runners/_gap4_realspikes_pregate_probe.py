"""gap#4 REAL-SPIKES pre-gate probe (the cheapest-first, most-decisive step of the on-bridge spiking port).

The unsupervised movable-plateau rule is the ONLY positive gap#4 credit signal (rate 5/6, dcs +0.139) -- but it
reads the plateau via `_prime_from_winners`: reset-soma + HOLD winner features as BOOLEANS in cp_prev_firing_states
with ZERO input current. Features never integrate current or cross threshold -> that is a rate/analytic STAND-IN, not
real spikes. The named next lever: does it hold when the read is a REAL spiking forward pass?

Finding 2026-07-10 (D1) found the on-bridge spiking forward pass DEGENERATE (hidden fires input-INDEPENDENTLY, near-
silent) -- the boundary that gates any real-spikes hidden-credit test -- but CORRECTED it to a config/operating-point
fix (AMPA propagates input-dependently; NMDA saturates), NOT a wall. And the reset-read was introduced to fix a
reproducibility-0.07 collapse, so a naive real-spikes read RE-OPENS it: reproducibility is LOAD-BEARING.

THIS PROBE (no 6-seed sweep, no credit comparison -- clear the boundary FIRST): replace the boolean-hold read with a
REAL spiking forward pass (drive the active FEATURE neurons via cp_external_input_current, integrate a real window, let
them SPIKE, propagate through the coincidence pathway to the columns' real plateau) and measure the PRE-GATE:
  (i)  INPUT-DEPENDENT firing: across inheritance inputs, feature spike-count std > 0 AND column plateau-margin std > 0
       (columns respond to the input, not tonic-pinned/near-silent).
  (ii) REPRODUCIBILITY: two identical presentations -> plateau-codon correlation >= 0.8 (the reset-read's 0.07 risk).
If the pre-gate FAILS, clearing the AMPA operating point IS the deliverable -- no credit-vs-frozen number is
interpretable until (i)+(ii) pass. Reuses PlasticPlateauExpander's bridge (the exact 5/6 substrate). NO sim/ edit.
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.runners._gap4_plastic_plateau_credit_derisk import (
    PlasticPlateauExpander, topk_active, FLOOR, TOPK, make_task_semantic_inheritance)
from research.runners._emerge12_stageB2_bridge_tm_derisk import reset_soma, _clear_apical
from sim.backend import to_host as _host


def _realspikes_read(exp, active_feats, drive_pa, n_steps, fw_scale):
    """A REAL spiking forward pass on the exact 5/6 bridge (replaces `_prime_from_winners`'s boolean hold).
    Drive the active FEATURE neurons with input current for n_steps; features integrate + SPIKE; their real firings
    drive the coincidence pathway to the columns' plateau. Returns (feature_spike_counts, column_margin, column_spike
    _counts). fw_scale temporarily scales the forward weights to the AMPA operating point (fw~40 vs the reservoir 0.35)
    WITHOUT persisting (restored after) -- an operating-point knob, not a weight edit."""
    b = exp.b; xp = b.xp if hasattr(b, "xp") else np
    n = int(b.core_config.num_neurons)
    saved = None
    if fw_scale != 1.0:
        saved = exp._get_data()
        exp._set_data(saved * fw_scale)
    reset_soma(b); _clear_apical(b)
    inp = np.zeros(n, np.float32)
    if len(active_feats):
        inp[exp.ci[np.asarray(list(active_feats), int)]] = float(drive_pa)   # drive FEATURES with real current
    feat_sp = np.zeros(exp.NF); col_sp = np.zeros(exp.NC)
    inp_x = xp.asarray(inp)
    for _ in range(int(n_steps)):
        b.cp_external_input_current[:] = inp_x
        b._run_one_simulation_step()
        fired = np.asarray(_host(b.cp_firing_states)).astype(np.float64)
        feat_sp += fired[exp.ci[:exp.NF]]
        col_sp += fired[exp.ci[exp.NF:exp.NF + exp.NC]]
    vap = getattr(b, "cp_v_apical", None)
    margin = (np.maximum(0.0, np.asarray(_host(vap))[exp.ci][exp.NF:exp.NF + exp.NC] - FLOOR)
              if vap is not None else np.zeros(exp.NC))
    if saved is not None:
        exp._set_data(saved)                                 # restore the reservoir weights (operating-point knob only)
    return feat_sp, margin, col_sp


def run_probe(seed, n_col, drive_pa, n_steps, fw_scale, n_inputs, task_kwargs):
    (Xtr, ytr, _), _, meta, _ = make_task_semantic_inheritance(seed, **task_kwargs)
    n_feat = Xtr.shape[1]
    af = topk_active(Xtr[:n_inputs], TOPK)
    exp = PlasticPlateauExpander(n_feat, n_col, seed)        # the EXACT 5/6 bridge (frozen reservoir init)
    # --- (i) INPUT-DEPENDENT firing: read each input, collect feature-spike + column-margin vectors ---
    feat_sps, margins, col_sps = [], [], []
    for a in af:
        fs, mg, cs = _realspikes_read(exp, a, drive_pa, n_steps, fw_scale)
        feat_sps.append(fs); margins.append(mg); col_sps.append(cs)
    feat_sps = np.asarray(feat_sps); margins = np.asarray(margins); col_sps = np.asarray(col_sps)
    # per-neuron/per-column std ACROSS inputs (input-dependence) -> mean over units
    feat_spike_std = float(np.mean(np.std(feat_sps, axis=0)))
    col_margin_std = float(np.mean(np.std(margins, axis=0)))
    col_spike_std = float(np.mean(np.std(col_sps, axis=0)))
    mean_feat_rate = float(np.mean(feat_sps) / max(1, n_steps))
    mean_col_margin = float(np.mean(margins))
    frac_cols_active = float(np.mean(margins.max(0) > 1e-6))
    # --- (ii) REPRODUCIBILITY: two identical presentations of the FIRST input -> codon correlation ---
    fs1, mg1, _ = _realspikes_read(exp, af[0], drive_pa, n_steps, fw_scale)
    fs2, mg2, _ = _realspikes_read(exp, af[0], drive_pa, n_steps, fw_scale)
    if np.std(mg1) > 1e-9 and np.std(mg2) > 1e-9:
        repro = float(np.corrcoef(mg1, mg2)[0, 1])
    else:
        repro = 1.0 if np.allclose(mg1, mg2) else 0.0
    # PRE-GATE
    input_dependent = bool(feat_spike_std > 1e-6 and col_margin_std > 1e-6)
    reliable = bool(repro >= 0.8)
    passed = bool(input_dependent and reliable)
    return {
        "seed": seed, "drive_pa": drive_pa, "n_steps": n_steps, "fw_scale": fw_scale, "n_inputs": n_inputs,
        "feat_spike_std_across_inputs": round(feat_spike_std, 5),
        "col_margin_std_across_inputs": round(col_margin_std, 5),
        "col_spike_std_across_inputs": round(col_spike_std, 5),
        "mean_feat_rate_per_step": round(mean_feat_rate, 5), "mean_col_margin": round(mean_col_margin, 5),
        "frac_cols_ever_active": round(frac_cols_active, 4),
        "reproducibility_codon_corr": round(repro, 4),
        "input_dependent_firing": input_dependent, "reliable_repro_ge_0p8": reliable,
        "PREGATE_PASS": passed,
    }


def main():
    ap = argparse.ArgumentParser(description="gap#4 real-spikes forward-pass + reproducibility PRE-GATE probe.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-col", type=int, default=200)
    ap.add_argument("--drive-pa", type=float, default=1200.0, help="input current to the feature neurons (AMPA op-point)")
    ap.add_argument("--n-steps", type=int, default=30, help="real integration window (ms)")
    ap.add_argument("--fw-scale", type=float, default=1.0, help="temporary forward-weight scale to the AMPA op-point")
    ap.add_argument("--n-inputs", type=int, default=12)
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super, n_prop=a.n_prop,
                       member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
    t0 = time.time()
    r = run_probe(a.seed, a.n_col, a.drive_pa, a.n_steps, a.fw_scale, a.n_inputs, task_kwargs)
    r["elapsed_seconds"] = round(time.time() - t0, 1); r["backend"] = os.environ.get("SIM_BACKEND")
    print("\n" + "=" * 100, flush=True)
    print(f"[realspikes-pregate] seed={r['seed']} drive={r['drive_pa']} steps={r['n_steps']} fw_scale={r['fw_scale']}",
          flush=True)
    print(f"  (i)  input-dependent: feat_spike_std={r['feat_spike_std_across_inputs']} "
          f"col_margin_std={r['col_margin_std_across_inputs']} -> {r['input_dependent_firing']}", flush=True)
    print(f"       mean_feat_rate/step={r['mean_feat_rate_per_step']} mean_col_margin={r['mean_col_margin']} "
          f"frac_cols_active={r['frac_cols_ever_active']}", flush=True)
    print(f"  (ii) reproducibility (codon corr)={r['reproducibility_codon_corr']} -> {r['reliable_repro_ge_0p8']}",
          flush=True)
    print(f"  ==> PRE-GATE {'PASS' if r['PREGATE_PASS'] else 'FAIL'}", flush=True)
    print("=" * 100, flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2, default=str))
        print(f"[realspikes-pregate] wrote {a.out}", flush=True)
    return 0 if r["PREGATE_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
