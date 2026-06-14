"""DENDRITIC de-risk D1.5 -- does the per-compartment advantage SURVIVE the spiking-noise floor?
(the cheap follow-on to D1 that retires the biggest D2 risk BEFORE the owner commits months)

WHY THIS RUNNER EXISTS
======================
D1 (`dendritic_d1_learn_graded_structure_derisk.py`) showed GO multi-seed: a dendritic per-compartment
gain recovers category structure (mean Pearson +0.845) that a single-soma point-neuron control cannot
(+0.052) -- but at the RATE level (clean residuals, reproducibility 0.999). The five prior point-neuron
NEGATIVEs all died at the SPIKING-noise floor: the input-driven signal sank below the spike noise (the
project's reproducibility gate: same input + sigma=0.1 noise -> output cosine >= 0.9, which killed the DG
kWTA and fixed-expansion attempts). D1.5 asks the load-bearing follow-on:

    When each concept's residual is READ as a finite Poisson SPIKE COUNT (the genuine spiking realism,
    swept from clean to noisy), does the DENDRITIC per-compartment structure recovery SURVIVE down to the
    noise floor where the POINT-NEURON cannot? I.e. does the per-compartment gain raise the category
    signal's SNR enough to clear the spike-noise floor that buried the five prior attempts?

The mechanism hypothesis: the per-hub gain down-weights the dominant high-frequency COMMON hubs, so in the
spike read the category-signal hubs get relatively MORE of the spike budget -> the structure survives a
lower spike budget (more noise) than the point-neuron, whose signal stays buried under the common-hub
spikes. If the dendritic advantage SURVIVES the noise floor, the D2 on-substrate build's biggest risk is
retired cheaply; if it COLLAPSES at the floor (like the five priors), that is a critical caveat that
changes the D2 recommendation.

Pure numpy, OFF-bridge, NO sim/ edits, multi-seed, reuse-by-import from D1. DIAGNOSTIC, not a deliverable.

THE SPIKE READ
==============
For a concept's residual rate-profile r (>=0): normalize to a rate density, draw a Poisson spike count per
hub with expected total = `spike_budget` (n_h ~ Poisson(spike_budget * r_h / sum_h r_h)). The spike-count
vector is the concept's CODE. Small spike_budget = few spikes = high relative noise (the regime that killed
the priors); large = clean (-> D1's rate result). Reproducibility = cosine of two INDEPENDENT spike reads
of the same concept. Structure recovery = Pearson(cos(spike_codes), S_true).

GATES (per the D1 discipline; multi-seed):
  SURVIVES -- at the LOWEST spike budget where the DENDRITIC reproducibility >= 0.90 (the project floor),
              the dendritic structure recovery is still >= +0.30 AND exceeds the point-neuron's by >= 0.30,
              WHILE the point-neuron structure is ~0 (<= 0.12). I.e. the per-compartment SNR boost clears
              the floor the point neuron cannot.
  COLLAPSES -- the dendritic structure recovery falls below +0.30 by the time reproducibility reaches 0.90
              (the signal sinks below the spike noise like the five priors) -> the rate-level GO does NOT
              carry to spikes; a critical D2 caveat.

ANTI-CHEATS: point-neuron-must-stay-failed at every budget; host PPMI+SVD ceiling (on clean counts) carries;
S_true a-priori; the dendritic advantage must be the per-hub gain (D1's lesion already proved this -- here
we additionally report the point-neuron at the SAME budget so the contrast is at matched noise).

Run (CPU/numpy; multi-seed):
  python -u -m research.runners.dendritic_d1p5_spiking_noise_derisk \
      --seeds 42,43,44 --out research/findings/raw/_dendritic_d1p5_multiseed.json
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

# reuse D1's data + learners + metrics VERBATIM (no reimplementation)
from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, learn_perhub_gains, learn_global_gain,
    perhub_residual, global_residual, _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def spike_read(residual, spike_budget, rng):
    """Read each concept's non-negative residual rate-profile as a Poisson SPIKE COUNT vector with expected
    total = spike_budget. Returns the [Nc x H] spike-count codes. Small budget = noisy."""
    r = np.maximum(residual, 0.0)
    row = r.sum(1, keepdims=True)
    p = r / (row + 1e-12)                      # per-concept rate density
    lam = spike_budget * p                     # expected spikes per hub
    return rng.poisson(lam).astype(np.float64)


def _struct_and_repro(residual, spike_budget, labels, S_true, seed, tag):
    """Two independent spike reads of the residual at `spike_budget`: structure recovery (Pearson vs S_true)
    + reproducibility (cosine of the two reads). Returns (struct, repro, gen)."""
    tag_salt = {"dend": 101, "pn": 202}.get(tag, 303)  # deterministic (Python hash() is per-process salted)
    rng = np.random.RandomState((seed * 2654435761 + tag_salt) % (2**31))
    code1 = spike_read(residual, spike_budget, rng)
    code2 = spike_read(residual, spike_budget, rng)
    struct = _pearson_vs_Strue(_cos_sim(code1), S_true)
    # reproducibility = mean per-concept cosine between the two independent reads
    n1 = np.linalg.norm(code1, axis=1); n2 = np.linalg.norm(code2, axis=1)
    repro = float(np.mean(np.sum(code1 * code2, 1) / (n1 * n2 + 1e-12)))
    gen, chance = heldout_generalization(code1, labels)
    return struct, repro, gen, chance


def run_seed(seed, args):
    print(f"\n{'='*88}\n  DENDRITIC D1.5 -- SPIKING-NOISE ROBUSTNESS (seed {seed})\n{'='*88}", flush=True)
    C, labels, S_true, hub_freq = build_concept_hub_counts(
        args.n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
        args.lam_common, args.lam_sig, args.lam_bg, seed)
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(args.host_svd, min(C.shape) - 1), alpha=args.host_alpha)
    host_pearson, _, _, _ = score(host_sim, labels)

    # learn the gains (clean rate learn, as in D1) then form the residual rate-profiles
    g_hub, _ = learn_perhub_gains(C, args.epochs, args.eta, seed)
    g_glob = learn_global_gain(C, args.epochs, args.eta, seed)
    dend_res = perhub_residual(C, g_hub, args.sigma)
    pn_res = global_residual(C, g_glob, args.sigma)

    print(f"  host PPMI ceiling (clean counts)={host_pearson:+.3f}; sweeping spike budget "
          f"{args.budgets}", flush=True)
    print(f"  {'budget':>8} | {'DEND struct':>11} {'DEND repro':>10} {'DEND gen':>8} | "
          f"{'PN struct':>9} {'PN repro':>8}", flush=True)
    sweep = []
    for b in args.budgets:
        ds, dr, dg, chance = _struct_and_repro(dend_res, b, labels, S_true, seed, "dend")
        ps, pr, pg, _ = _struct_and_repro(pn_res, b, labels, S_true, seed, "pn")
        sweep.append({"budget": b, "dend_struct": ds, "dend_repro": dr, "dend_gen": dg,
                      "pn_struct": ps, "pn_repro": pr, "pn_gen": pg, "chance": chance})
        print(f"  {b:>8} | {ds:>+11.3f} {dr:>10.3f} {dg:>8.3f} | {ps:>+9.3f} {pr:>8.3f}", flush=True)

    # the headline: at the LOWEST budget where dendritic repro >= repro_bar, is the dendritic structure
    # still >= structure_bar AND >> point-neuron, while point-neuron stays ~0?
    survive_rows = [r for r in sweep if r["dend_repro"] >= args.repro_bar]
    floor_row = min(survive_rows, key=lambda r: r["budget"]) if survive_rows else None
    if floor_row is not None:
        survives = (floor_row["dend_struct"] >= args.structure_bar
                    and (floor_row["dend_struct"] - floor_row["pn_struct"]) >= args.contrast_bar
                    and abs(floor_row["pn_struct"]) <= args.pn_fail_bar)
    else:
        survives = False
    host_carries = host_pearson >= args.host_bar
    print(f"  [seed {seed}] floor budget (dend repro>={args.repro_bar}): "
          f"{floor_row['budget'] if floor_row else 'NONE'}; "
          f"dend struct there={floor_row['dend_struct'] if floor_row else float('nan'):+.3f} vs PN "
          f"{floor_row['pn_struct'] if floor_row else float('nan'):+.3f} -> SURVIVES={survives}", flush=True)
    return {"seed": seed, "host_ceiling_pearson": host_pearson, "host_carries": bool(host_carries),
            "sweep": sweep, "floor_row": floor_row, "survives": bool(survives), "chance": sweep[0]["chance"]}


def decide_verdict(per_seed, seeds, args):
    survive_all = all(per_seed[str(s)]["survives"] for s in seeds)
    host_all = all(per_seed[str(s)]["host_carries"] for s in seeds)
    floors = [per_seed[str(s)]["floor_row"]["budget"] for s in seeds
              if per_seed[str(s)]["floor_row"] is not None]
    dstructs = [per_seed[str(s)]["floor_row"]["dend_struct"] for s in seeds
                if per_seed[str(s)]["floor_row"] is not None]
    pstructs = [per_seed[str(s)]["floor_row"]["pn_struct"] for s in seeds
                if per_seed[str(s)]["floor_row"] is not None]
    if not host_all:
        verdict = "NEGATIVE_miscalibrated"
        why = "the host ceiling did not carry on clean counts -> re-tune the toy before trusting D1.5."
    elif survive_all:
        verdict = "SURVIVES"
        why = (f"at the spiking-noise floor (dendritic reproducibility >= {args.repro_bar}), the dendritic "
               f"per-compartment structure recovery holds (mean {np.mean(dstructs):+.3f}) and exceeds the "
               f"point-neuron (mean {np.mean(pstructs):+.3f}) by the required margin, all seeds. The "
               f"per-compartment SNR boost clears the spike-noise floor that buried the five prior "
               f"point-neuron attempts -> D2's biggest risk (does the rate-level GO carry to spikes?) is "
               f"retired at the toy level; the D2 build case strengthens.")
    else:
        verdict = "COLLAPSES"
        why = (f"the dendritic structure recovery falls below the bar by the time reproducibility reaches "
               f"{args.repro_bar} on some seed -> the rate-level GO does NOT carry to the spike-noise floor "
               f"(the signal sinks below the spike noise like the five priors). A critical D2 caveat: the "
               f"on-substrate build must show the spiking two-compartment unit clears this floor before the "
               f"months-scale commit is justified.")
    return verdict, why, {"survive_all": survive_all, "host_all": host_all,
                          "floor_budgets": floors, "dend_struct_at_floor_mean": float(np.mean(dstructs)) if dstructs else None,
                          "pn_struct_at_floor_mean": float(np.mean(pstructs)) if pstructs else None}


def main():
    p = argparse.ArgumentParser(description="Dendritic D1.5: does the per-compartment advantage survive "
                                            "the spiking-noise floor?")
    p.add_argument("--seeds", default="42,43,44")
    # toy config (match D1's calibrated operating point)
    p.add_argument("--n-cat", type=int, default=8)
    p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--n-common", type=int, default=200)
    p.add_argument("--n-sig-per-cat", type=int, default=12)
    p.add_argument("--lam-common", type=float, default=40.0)
    p.add_argument("--lam-sig", type=float, default=4.0)
    p.add_argument("--lam-bg", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--host-svd", type=int, default=50)
    p.add_argument("--host-alpha", type=float, default=0.75)
    # the spike-budget sweep (expected total spikes per concept read; small = noisy)
    p.add_argument("--budgets", type=int, nargs="+",
                   default=[50, 100, 200, 400, 800, 1600, 3200, 6400])
    # gate bars
    p.add_argument("--structure-bar", type=float, default=0.30)
    p.add_argument("--contrast-bar", type=float, default=0.30)
    p.add_argument("--pn-fail-bar", type=float, default=0.12)
    p.add_argument("--repro-bar", type=float, default=0.90)
    p.add_argument("--host-bar", type=float, default=0.30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print(f"[dendritic D1.5] seeds={seeds}  question: does the per-compartment advantage survive the "
          f"spiking-noise floor that killed the 5 prior attempts?", flush=True)
    per_seed = {str(s): run_seed(s, args) for s in seeds}
    verdict, why, detail = decide_verdict(per_seed, seeds, args)
    print(f"\n{'='*88}\n  D1.5 VERDICT: {verdict}\n  {why}", flush=True)
    print(f"  floor budgets per seed: {detail['floor_budgets']}; dend struct at floor "
          f"{detail['dend_struct_at_floor_mean']} vs PN {detail['pn_struct_at_floor_mean']}", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n{'='*88}\n", flush=True)

    out = {"verdict": verdict, "why": why, "detail": detail, "seeds": seeds, "config": vars(args),
           "per_seed": per_seed,
           "note": ("DIAGNOSTIC follow-on to D1: tests whether the rate-level per-compartment advantage "
                    "survives a finite Poisson spike read (the spiking-noise floor that killed the five "
                    "prior point-neuron NEGATIVEs). NO sim/ edits. Informs the D2 go/no-go."),
           "elapsed_total_s": time.time() - t0}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_dendritic_d1p5_multiseed_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
