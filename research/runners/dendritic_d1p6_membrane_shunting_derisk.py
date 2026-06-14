"""DENDRITIC de-risk D1.6 -- does the advantage hold under REAL conductance-based membrane dynamics?
(the last cheap, non-gated rung before the owner-gated D2 on-substrate build)

WHY THIS RUNNER EXISTS
======================
D1 showed a per-compartment ALGEBRAIC divisive gain (r_h = x_h/(sigma+g_h)) recovers category structure a
single-soma point neuron cannot (GO, multi-seed). D1.5 showed it survives a finite Poisson spike read
(the spiking-noise floor). The remaining gap to the on-substrate D2 build is whether the divisive gain
holds when realized by a BIOPHYSICALLY FAITHFUL mechanism -- **conductance-based shunting inhibition** on a
leaky membrane, settled over time, with membrane noise -- rather than an algebraic divide. Shunting
inhibition IS the biophysics of divisive gain: a leaky membrane V with an inhibitory conductance g_inh,
   dV/dt = -g_leak*V + I_exc - g_inh*V  (+ membrane noise),
settles to V = I_exc / (g_leak + g_inh) -- DIVISIVE. Delivering g_inh PER COMPARTMENT (per hub) gives the
per-input normalization; a single SOMATIC g_inh (the point neuron) gives only a global gain. This runner
asks: under the settled, noisy, conductance-based membrane, does the per-compartment shunting recover the
structure the global-shunting point-neuron cannot? If yes, the biophysical mechanism D2 would build is
validated at the toy level; if the membrane dynamics/noise break it (cf. the phase-coding "needs high-Q"
lesson), that is a decision-relevant D2 caveat.

Pure numpy, OFF-bridge, NO sim/ edits, multi-seed, reuse-by-import from D1. DIAGNOSTIC, not a deliverable.

Run (CPU/numpy; multi-seed):
  python -u -m research.runners.dendritic_d1p6_membrane_shunting_derisk \
      --seeds 42,43,44 --out research/findings/raw/_dendritic_d1p6_multiseed.json
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

# reuse D1's data + gain-learning + metrics VERBATIM
from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, learn_perhub_gains, learn_global_gain,
    _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def settle_membrane(C, g_inh, g_leak, steps, dt, noise, seed, per_hub=True):
    """Settle a leaky conductance-based membrane with SHUNTING inhibition for every concept's hub-drive.
    dV/dt = -g_leak*V + x - g_inh*V + membrane_noise. Steady state V = x/(g_leak + g_inh) (divisive).
    g_inh is a per-hub vector (dendritic, per-compartment) or a scalar (point-neuron, somatic).
    Returns the settled membrane codes [Nc x H] (mean of the last few steps, with per-step noise)."""
    rng = np.random.RandomState((seed * 2246822519 + (7 if per_hub else 13)) % (2**31))
    Nc, H = C.shape
    if per_hub:
        gshunt = g_inh[None, :]            # [1 x H] per-compartment
    else:
        gshunt = float(g_inh)              # scalar somatic
    # EXPONENTIAL EULER (exact + unconditionally stable for the linear conductance membrane): the total
    # conductance g_tot = g_leak + g_inh can be large (g_inh ~ hub marginals up to ~40), so forward Euler
    # would be unstable; exp-Euler relaxes toward the steady state V_ss = x / g_tot each step.
    g_tot = g_leak + gshunt                # per-hub vector or scalar
    V_ss = C / g_tot                       # divisive steady state (the per-compartment gain)
    decay = np.exp(-g_tot * dt)            # broadcast over hubs
    V = np.zeros((Nc, H), dtype=np.float64)
    tail = []
    for t in range(steps):
        V = V_ss + (V - V_ss) * decay
        if noise > 0:
            V = V + rng.normal(0.0, noise, size=V.shape) * np.sqrt(dt)
        V = np.maximum(V, 0.0)             # rectified membrane (firing-rate proxy)
        if t >= steps - 4:
            tail.append(V.copy())
    code = np.mean(tail, axis=0) if tail else V
    # convergence diagnostic: relative change over the last two recorded steps
    conv = float(np.linalg.norm(tail[-1] - tail[-2]) / (np.linalg.norm(tail[-1]) + 1e-12)) if len(tail) >= 2 else 1.0
    return code, conv


def run_seed(seed, args):
    print(f"\n{'='*88}\n  DENDRITIC D1.6 -- CONDUCTANCE SHUNTING MEMBRANE (seed {seed})\n{'='*88}", flush=True)
    C, labels, S_true, hub_freq = build_concept_hub_counts(
        args.n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
        args.lam_common, args.lam_sig, args.lam_bg, seed)
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(args.host_svd, min(C.shape) - 1), alpha=args.host_alpha)
    host_pearson, _, _, _ = score(host_sim, labels)

    # learn the inhibitory conductances (same local online rule as D1; g_inh ~ hub marginal)
    g_hub, _ = learn_perhub_gains(C, args.epochs, args.eta, seed)
    g_glob = learn_global_gain(C, args.epochs, args.eta, seed)

    # settle the conductance-based shunting membrane (per-compartment vs global), with membrane noise
    dend_code, dend_conv = settle_membrane(C, g_hub, args.g_leak, args.steps, args.dt, args.noise,
                                           seed, per_hub=True)
    pn_code, pn_conv = settle_membrane(C, g_glob, args.g_leak, args.steps, args.dt, args.noise,
                                       seed, per_hub=False)
    dend_pearson = _pearson_vs_Strue(_cos_sim(dend_code), S_true)
    pn_pearson = _pearson_vs_Strue(_cos_sim(pn_code), S_true)
    dend_gen, chance = heldout_generalization(dend_code, labels)
    pn_gen, _ = heldout_generalization(pn_code, labels)
    dend_rank = effective_rank(dend_code)
    dend_off = float(_cos_sim(dend_code)[np.triu_indices(C.shape[0], 1)].mean())

    # reproducibility: two independent noisy membrane settles of the same drive
    d2, _ = settle_membrane(C, g_hub, args.g_leak, args.steps, args.dt, args.noise, seed + 1, per_hub=True)
    n1 = np.linalg.norm(dend_code, axis=1); n2 = np.linalg.norm(d2, axis=1)
    dend_repro = float(np.mean(np.sum(dend_code * d2, 1) / (n1 * n2 + 1e-12)))

    print(f"  host ceiling (clean counts)={host_pearson:+.3f}; membrane: g_leak={args.g_leak} "
          f"steps={args.steps} dt={args.dt} noise={args.noise}", flush=True)
    print(f"  [DENDRITIC per-compartment shunting] Pearson={dend_pearson:+.3f}  gen={dend_gen:.3f} "
          f"(chance {chance:.3f})  repro={dend_repro:.3f}  eff-rank={dend_rank:.1f}  off={dend_off:+.3f}  "
          f"(settle conv={dend_conv:.1e})", flush=True)
    print(f"  [POINT-NEURON global shunting] Pearson={pn_pearson:+.3f}  gen={pn_gen:.3f}  "
          f"(settle conv={pn_conv:.1e})", flush=True)

    point_neuron_fails = abs(pn_pearson) <= args.pn_fail_bar
    host_carries = host_pearson >= args.host_bar
    structure = (dend_pearson >= args.structure_bar) and point_neuron_fails and host_carries
    generalize = (dend_gen > chance + args.gen_margin) and (dend_gen - pn_gen >= args.gen_contrast_margin)
    settled = dend_conv <= args.conv_bar
    gates = {"structure_contrast": bool(structure), "point_neuron_fails": bool(point_neuron_fails),
             "host_ceiling_carries": bool(host_carries), "generalize_contrast": bool(generalize),
             "membrane_settles": bool(settled), "not_collapsed": bool(dend_off < 0.95 and dend_rank > 1.5)}
    print(f"  [seed {seed} gates] {gates}", flush=True)
    return {"seed": seed, "host_ceiling_pearson": host_pearson, "chance": chance,
            "dendritic": {"pearson": dend_pearson, "generalization": dend_gen, "repro": dend_repro,
                          "eff_rank": dend_rank, "settle_conv": dend_conv},
            "point_neuron": {"pearson": pn_pearson, "generalization": pn_gen, "settle_conv": pn_conv},
            "gates": gates}


def decide_verdict(per_seed, seeds, args):
    def allg(k):
        return all(per_seed[str(s)]["gates"][k] for s in seeds)
    structure = allg("structure_contrast"); pn_fails = allg("point_neuron_fails")
    host_ok = allg("host_ceiling_carries"); generalize = allg("generalize_contrast")
    settles = allg("membrane_settles")
    dmean = float(np.mean([per_seed[str(s)]["dendritic"]["pearson"] for s in seeds]))
    pmean = float(np.mean([per_seed[str(s)]["point_neuron"]["pearson"] for s in seeds]))
    if not host_ok:
        verdict, why = "NEGATIVE_miscalibrated", "host ceiling did not carry -> re-tune the toy."
    elif not pn_fails:
        verdict, why = "NEGATIVE_miscalibrated", f"point-neuron (global shunting) did not fail ({pmean:+.3f})."
    elif structure and settles and generalize:
        verdict = "SURVIVES"
        why = (f"under REAL conductance-based shunting-inhibition membrane dynamics (settled + noisy), the "
               f"per-compartment shunting recovers the structure (mean {dmean:+.3f}) and generalizes WHILE "
               f"the global-shunting point-neuron fails ({pmean:+.3f}), all seeds, membrane converged. The "
               f"biophysical divisive-normalization mechanism D2 would build is validated at the toy level "
               f"-> the D2 case is maximally de-risked cheaply; the months-scale build remains owner-gated.")
    elif structure and settles:
        verdict = "SURVIVES_structure_only"
        why = (f"structure recovers under the membrane ({dmean:+.3f} vs point-neuron {pmean:+.3f}, settled) "
               f"but the generalization contrast did not clear -> real but partial; characterize.")
    else:
        verdict = "COLLAPSES"
        why = (f"the membrane dynamics/noise break the per-compartment advantage (dendritic {dmean:+.3f} vs "
               f"point-neuron {pmean:+.3f}, or did not settle) -> a decision-relevant D2 caveat: the "
               f"biophysical realization needs care (cf. the phase-coding 'needs high-Q resonator' lesson).")
    return verdict, why, {"dendritic_pearson_mean": dmean, "point_neuron_pearson_mean": pmean,
                          "structure_all": structure, "pn_fails_all": pn_fails, "settles_all": settles,
                          "generalize_all": generalize}


def main():
    p = argparse.ArgumentParser(description="Dendritic D1.6: per-compartment advantage under conductance "
                                            "shunting membrane dynamics?")
    p.add_argument("--seeds", default="42,43,44")
    # toy (D1's calibrated operating point)
    p.add_argument("--n-cat", type=int, default=8); p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--n-common", type=int, default=200); p.add_argument("--n-sig-per-cat", type=int, default=12)
    p.add_argument("--lam-common", type=float, default=40.0); p.add_argument("--lam-sig", type=float, default=4.0)
    p.add_argument("--lam-bg", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--host-svd", type=int, default=50); p.add_argument("--host-alpha", type=float, default=0.75)
    # membrane
    p.add_argument("--g-leak", type=float, default=1.0, help="leak conductance (the divisive semi-saturation)")
    p.add_argument("--steps", type=int, default=60, help="membrane settle steps")
    p.add_argument("--dt", type=float, default=0.2, help="membrane integration step")
    p.add_argument("--noise", type=float, default=0.05, help="per-step membrane noise std")
    # bars
    p.add_argument("--structure-bar", type=float, default=0.30); p.add_argument("--pn-fail-bar", type=float, default=0.12)
    p.add_argument("--host-bar", type=float, default=0.30); p.add_argument("--gen-margin", type=float, default=0.05)
    p.add_argument("--gen-contrast-margin", type=float, default=0.30); p.add_argument("--conv-bar", type=float, default=0.05)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print(f"[dendritic D1.6] seeds={seeds}  question: does the per-compartment advantage hold under real "
          f"conductance-based shunting-inhibition membrane dynamics?", flush=True)
    per_seed = {str(s): run_seed(s, args) for s in seeds}
    verdict, why, detail = decide_verdict(per_seed, seeds, args)
    print(f"\n{'='*88}\n  D1.6 VERDICT: {verdict}\n  {why}", flush=True)
    print(f"  ladder: DENDRITIC membrane {detail['dendritic_pearson_mean']:+.3f}  vs  POINT-NEURON "
          f"{detail['point_neuron_pearson_mean']:+.3f}  (structure all seeds: {detail['structure_all']})", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n{'='*88}\n", flush=True)

    out = {"verdict": verdict, "why": why, "detail": detail, "seeds": seeds, "config": vars(args),
           "per_seed": per_seed,
           "note": ("DIAGNOSTIC: realizes the per-compartment divisive gain as conductance-based shunting "
                    "inhibition on a settled, noisy leaky membrane (the biophysics D2 would build). NO sim/ "
                    "edits. Last cheap rung before the owner-gated D2."),
           "elapsed_total_s": time.time() - t0}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_dendritic_d1p6_multiseed_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
