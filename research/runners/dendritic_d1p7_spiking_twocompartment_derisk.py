"""DENDRITIC de-risk D1.7 -- Phase 0 of the D2 build: a FULL SPIKING two-compartment neuron.
(the last NON-protected gate before the protected on-bridge NeuronModel edit)

WHY THIS RUNNER EXISTS
======================
D1 (algebraic gain), D1.5 (finite spike read), D1.6 (conductance-membrane) all kept the soma's output as a
RATE. The one risk the ladder hasn't faced: a genuine SPIKING SOMA -- a leaky integrate-and-fire (LIF)
with a hard threshold + reset -- whose threshold could CLIP the low-drive category-signal hubs below
firing, destroying the very signal the dendrite exposed. This runner closes that gap with a full
two-compartment spiking unit per hub-compartment:
  - DENDRITE: conductance-based shunting gain (D1.6 steady state), V_dend = x / (g_leak + g_inh).
  - SOMA: a LIF driven by V_dend; spike-count over a window = the code. (g_inh per-hub = dendritic;
    g_inh single global = the point-neuron control.)
The code is read from SPIKES (counts), not the rate. GATE: does the per-compartment advantage survive the
somatic threshold, across a threshold sweep, WHILE the point-neuron control fails at the same thresholds?

THE KEY MECHANISM (why the threshold should NOT clip the signal): the per-hub gain divides each hub by its
own frequency, so after normalization the rare CATEGORY hubs become MORE salient than the dominant common
hubs (a within-category concept's category hub: drive ~ lam_sig/(g_leak + small_marginal) is LARGE; common
hub: lam_common/(g_leak + lam_common) ~ 1). The somatic threshold then clips the (now-small, normalized)
common mode + the out-of-category background, leaving the category hubs to fire -- the threshold HELPS. The
point-neuron's single global gain leaves the common mode dominant, so its threshold clips the (small)
category hubs and passes only the common mode -> no structure. This runner tests that prediction in spikes.

Pure numpy, OFF-bridge, NO sim/ edits, multi-seed, reuse-by-import from D1/D1.6. If it SURVIVES, Phase 1
(the protected two-compartment NeuronModel on the bridge) is warranted; if it COLLAPSES, surface the
NEGATIVE (the spiking soma breaks it) BEFORE any protected edit.

Run (CPU/numpy; multi-seed):
  python -u -m research.runners.dendritic_d1p7_spiking_twocompartment_derisk \
      --seeds 42,43,44 --out research/findings/raw/_dendritic_d1p7_multiseed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, learn_perhub_gains, learn_global_gain,
    _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def lif_spike_counts(drive, threshold, leak, T, dt, noise, seed):
    """Leaky integrate-and-fire soma per (concept, hub) driven by the constant dendritic drive `drive`
    [Nc x H]. dU/dt = -leak*U + drive (+ membrane noise); spike when U > threshold, reset to 0. Returns the
    spike-count code over T steps."""
    rng = np.random.RandomState((seed * 40503 + int(threshold * 1000) + 7) % (2**31))
    Nc, H = drive.shape
    U = np.zeros((Nc, H), dtype=np.float64)
    counts = np.zeros((Nc, H), dtype=np.float64)
    for t in range(T):
        U = U + dt * (-leak * U + drive)
        if noise > 0:
            U = U + rng.normal(0.0, noise, size=U.shape) * np.sqrt(dt)
        spiked = U > threshold
        counts += spiked
        U = np.where(spiked, 0.0, U)
    return counts


def _eval_codes(codes, labels, S_true):
    pear = _pearson_vs_Strue(_cos_sim(codes), S_true)
    gen, chance = heldout_generalization(codes, labels)
    rank = effective_rank(codes)
    off = float(_cos_sim(codes)[np.triu_indices(codes.shape[0], 1)].mean())
    silent = float(np.mean(codes.sum(1) == 0))   # fraction of concepts that fired no spikes at all
    return {"pearson": pear, "gen": gen, "chance": chance, "eff_rank": rank, "offdiag": off, "silent": silent}


def run_seed(seed, args):
    print(f"\n{'='*92}\n  DENDRITIC D1.7 -- SPIKING TWO-COMPARTMENT (seed {seed})\n{'='*92}", flush=True)
    C, labels, S_true, hub_freq = build_concept_hub_counts(
        args.n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
        args.lam_common, args.lam_sig, args.lam_bg, seed)
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(args.host_svd, min(C.shape) - 1), alpha=args.host_alpha)
    host_pearson, _, _, _ = score(host_sim, labels)

    # learned shunting conductances (D1 online local rule)
    g_hub, _ = learn_perhub_gains(C, args.epochs, args.eta, seed)
    g_glob = learn_global_gain(C, args.epochs, args.eta, seed)
    # dendritic steady-state drive (D1.6 conductance shunting, settled form) feeding the soma
    dend_drive = C / (args.g_leak + g_hub[None, :])
    pn_drive = C / (args.g_leak + g_glob)

    print(f"  host ceiling={host_pearson:+.3f}; LIF soma leak={args.leak} T={args.T} dt={args.dt} "
          f"noise={args.noise}; threshold sweep {args.thresholds}", flush=True)
    print(f"  {'thresh':>7} | {'DEND pear':>9} {'DEND gen':>8} {'DEND silent':>11} | "
          f"{'PN pear':>8} {'PN gen':>7} {'PN silent':>9}", flush=True)
    sweep = []
    for th in args.thresholds:
        d = lif_spike_counts(dend_drive, th, args.leak, args.T, args.dt, args.noise, seed)
        p = lif_spike_counts(pn_drive, th, args.leak, args.T, args.dt, args.noise, seed)
        de = _eval_codes(d, labels, S_true)
        pe = _eval_codes(p, labels, S_true)
        # reproducibility: a second independent spike run of the dendritic drive
        d2 = lif_spike_counts(dend_drive, th, args.leak, args.T, args.dt, args.noise, seed + 9991)
        n1 = np.linalg.norm(d, axis=1); n2 = np.linalg.norm(d2, axis=1)
        repro = float(np.mean(np.sum(d * d2, 1) / (n1 * n2 + 1e-12)))
        row = {"threshold": th, "dend": de, "pn": pe, "dend_repro": repro,
               "dend_struct": de["pearson"], "pn_struct": pe["pearson"]}
        sweep.append(row)
        print(f"  {th:>7.2f} | {de['pearson']:>+9.3f} {de['gen']:>8.3f} {de['silent']:>11.2f} | "
              f"{pe['pearson']:>+8.3f} {pe['gen']:>7.3f} {pe['silent']:>9.2f}", flush=True)

    chance = sweep[0]["dend"]["chance"]
    # the headline: is there a threshold where the dendritic clears the structure bar + reproducibility,
    # the point-neuron fails, and the dendritic is not mostly-silent (the threshold didn't clip the signal)?
    ok_rows = [r for r in sweep
               if r["dend_struct"] >= args.structure_bar
               and r["dend_repro"] >= args.repro_bar
               and abs(r["pn_struct"]) <= args.pn_fail_bar
               and r["dend"]["silent"] <= args.max_silent
               and (r["dend"]["gen"] - r["pn"]["gen"]) >= args.gen_contrast_margin]
    best = max(ok_rows, key=lambda r: r["dend_struct"]) if ok_rows else None
    survives = best is not None and host_pearson >= args.host_bar
    if best is not None:
        print(f"  [seed {seed}] BEST surviving threshold={best['threshold']:.2f}: dend struct "
              f"{best['dend_struct']:+.3f} (repro {best['dend_repro']:.3f}, silent {best['dend']['silent']:.2f}) "
              f"vs PN {best['pn_struct']:+.3f} -> SURVIVES={survives}", flush=True)
    else:
        print(f"  [seed {seed}] NO threshold satisfies the gate -> SURVIVES=False", flush=True)
    return {"seed": seed, "host_ceiling_pearson": host_pearson, "host_carries": bool(host_pearson >= args.host_bar),
            "chance": chance, "sweep": sweep, "best": best, "survives": bool(survives)}


def decide_verdict(per_seed, seeds, args):
    survive_all = all(per_seed[str(s)]["survives"] for s in seeds)
    host_all = all(per_seed[str(s)]["host_carries"] for s in seeds)
    bests = [per_seed[str(s)]["best"] for s in seeds if per_seed[str(s)]["best"] is not None]
    dmean = float(np.mean([b["dend_struct"] for b in bests])) if bests else None
    pmean = float(np.mean([b["pn_struct"] for b in bests])) if bests else None
    if not host_all:
        verdict, why = "NEGATIVE_miscalibrated", "host ceiling did not carry -> re-tune the toy."
    elif survive_all:
        verdict = "SURVIVES"
        why = (f"a genuine SPIKING LIF soma preserves the per-compartment advantage: at a common threshold "
               f"the dendritic spike-count code recovers the structure (mean {dmean:+.3f}, reproducible, not "
               f"clipped-silent) and beats the point-neuron ({pmean:+.3f}) which the same threshold leaves "
               f"common-mode-dominated, all seeds. The somatic threshold does NOT clip the signal (the "
               f"per-hub normalization makes the category hubs salient) -> Phase 0 PASSES; the protected "
               f"on-bridge two-compartment build (Phase 1) is warranted.")
    else:
        verdict = "COLLAPSES"
        why = (f"the spiking soma breaks the advantage on some seed (no threshold clears structure + "
               f"reproducibility + point-neuron-fails + not-clipped together) -> a decision-relevant Phase-0 "
               f"NEGATIVE: the protected on-bridge build should NOT start until the somatic-threshold clip is "
               f"resolved.")
    return verdict, why, {"survive_all": survive_all, "host_all": host_all,
                          "dend_struct_at_best_mean": dmean, "pn_struct_at_best_mean": pmean,
                          "best_thresholds": [b["threshold"] for b in bests]}


def main():
    p = argparse.ArgumentParser(description="Dendritic D1.7 (D2 Phase 0): a full spiking two-compartment "
                                            "neuron -- does the advantage survive a spiking soma?")
    p.add_argument("--seeds", default="42,43,44")
    # toy (D1 calibrated operating point)
    p.add_argument("--n-cat", type=int, default=8); p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--n-common", type=int, default=200); p.add_argument("--n-sig-per-cat", type=int, default=12)
    p.add_argument("--lam-common", type=float, default=40.0); p.add_argument("--lam-sig", type=float, default=4.0)
    p.add_argument("--lam-bg", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--host-svd", type=int, default=50); p.add_argument("--host-alpha", type=float, default=0.75)
    p.add_argument("--g-leak", type=float, default=1.0, help="dendritic leak conductance (divisive constant)")
    # LIF soma
    p.add_argument("--leak", type=float, default=1.0, help="somatic LIF leak")
    p.add_argument("--T", type=int, default=200, help="spike-count window (steps)")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.02, help="somatic membrane noise")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
    # bars
    p.add_argument("--structure-bar", type=float, default=0.30); p.add_argument("--pn-fail-bar", type=float, default=0.12)
    p.add_argument("--host-bar", type=float, default=0.30); p.add_argument("--repro-bar", type=float, default=0.90)
    p.add_argument("--gen-contrast-margin", type=float, default=0.30); p.add_argument("--max-silent", type=float, default=0.10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print(f"[dendritic D1.7 / D2 Phase 0] seeds={seeds}  question: does the per-compartment advantage "
          f"survive a genuine spiking LIF soma (threshold could clip the signal)?", flush=True)
    per_seed = {str(s): run_seed(s, args) for s in seeds}
    verdict, why, detail = decide_verdict(per_seed, seeds, args)
    print(f"\n{'='*92}\n  D1.7 (D2 Phase 0) VERDICT: {verdict}\n  {why}", flush=True)
    print(f"  ladder: DEND@best-threshold {detail['dend_struct_at_best_mean']}  vs  PN "
          f"{detail['pn_struct_at_best_mean']}  (best thresholds {detail['best_thresholds']})", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n{'='*92}\n", flush=True)

    out = {"verdict": verdict, "why": why, "detail": detail, "seeds": seeds, "config": vars(args),
           "per_seed": per_seed,
           "note": ("D2 Phase 0: a full spiking two-compartment numpy neuron (dendritic shunting -> LIF "
                    "soma, spike-count code). The last non-protected gate before the protected on-bridge "
                    "two-compartment NeuronModel edit (Phase 1)."),
           "elapsed_total_s": time.time() - t0}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_dendritic_d1p7_multiseed_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
