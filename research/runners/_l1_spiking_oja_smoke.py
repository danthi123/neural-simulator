"""L1 spiking de-risk: does the +0.48 learned structure SURVIVE the input being delivered as SPIKES?

The rate-level GO (`_l1_oja_validated.py`): Oja online local PCA on TRUE PPMI input reaches +0.481 (92% of
the offline +0.523, beats host +0.323), full battery clean. The single highest-variance unknown before the
weeks-scale SPIKING build is THIS project's hardest recurring theme -- the rate->spike loss. This smoke
mirrors the trusted D1.5->D1.7 numpy spiking ladder: deliver the PPMI input as POISSON SPIKE COUNTS (the
cortex sees noisy spike trains, not exact rates), re-sampled every epoch (realistic trial-to-trial noise)
and at TEST time (generalization to fresh spike noise), and ask whether the Oja learner still extracts the
category structure. Sweeps the SPIKE BUDGET (expected spikes per unit PPMI) -- the wide-dynamic-range
coding axis: high budget -> low Poisson noise -> approach the rate ceiling; low budget -> the threshold-
silencing regime. CONSERVATIVE staging to avoid a misleading negative: the LEARNING math stays exact-Oja
(driven by the spike-count vectors), so this isolates the INPUT-spiking gap (Q-input) cleanly; the
learning-spiking gap (spike-timing Hebbian) is the follow-on if this survives.

ARMS (multi-seed): Oja on spike-count PPMI at each budget; faithful-saturating control (must fail);
permuted-similarity (must ~0). Reference: rate Oja +0.481, offline +0.523.

GATE: at a reasonable spike budget, spiking-Oja Pearson >= 0.70 * rate-Oja AND >= 0.30, generalizes,
controls clean -> the structure SURVIVES spikes -> the spiking build is de-risked on the input axis.
If it needs an unrealistic budget / never recovers -> flags the rate->spike cost BEFORE the weeks-commit.
CPU/numpy, seed-independent corpus -> build once. NO sim/ edits.

Run: python -u -m research.runners._l1_spiking_oja_smoke --n-hub 2000 --seeds 42,43,44
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import (  # noqa: E402
    build_real_corpus, ppmi_matrix, pca_lowrank_sim,
)
from research.runners._l1_centered_online_pca_probe import center_cols  # noqa: E402


def _poisson_spikes(lam, rng):
    return rng.poisson(np.maximum(lam, 0.0)).astype(np.float64)


def oja_spiking(Xppmi_c, k, epochs, lr, spike_gain, seed, resample=True):
    """Oja's subspace rule driven by POISSON SPIKE-COUNT inputs. lam = max(PPMI,0)*spike_gain (expected
    counts); inputs re-sampled per epoch (realistic spike noise). Learning math = exact Oja on the noisy
    spike-count vectors (isolates the input-spiking gap). Center is applied to the EXPECTED input
    (subtractive-inhibition EMA of the mean drive); spikes are non-negative, the centering is on the rate."""
    rng = np.random.RandomState(seed * 6151 + 5)
    Nc, H = Xppmi_c.shape
    # restore a non-negative rate from the centered PPMI for Poisson sampling, keep the centering as a
    # subtractive bias applied AFTER sampling (the brain realizes mean-removal as subtractive inhibition).
    Xppmi = Xppmi_c - Xppmi_c.min()  # shift to non-negative rates for Poisson; structure preserved
    col_mean_rate = Xppmi.mean(0, keepdims=True)
    W = rng.randn(k, H) * 0.1
    base = _poisson_spikes(Xppmi * spike_gain, rng)  # one fixed sample if not resample
    order = np.arange(Nc)
    for _ in range(epochs):
        S = _poisson_spikes(Xppmi * spike_gain, rng) if resample else base
        S = S - col_mean_rate * spike_gain            # subtractive inhibition (common-mode removal)
        Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
        rng.shuffle(order)
        for i in order:
            x = Sn[i]
            y = W @ x
            W += lr * (np.outer(y, x) - np.outer(y, y) @ W)
    # read codes from a FRESH spike sample (test-time noise)
    St = _poisson_spikes(Xppmi * spike_gain, rng) - col_mean_rate * spike_gain
    Stn = St / (np.linalg.norm(St, axis=1, keepdims=True) + 1e-9)
    mean_spikes = float(_poisson_spikes(Xppmi * spike_gain, rng).sum(1).mean())
    return (W @ Stn.T).T, mean_spikes


def faithful_saturating_spiking(Xppmi_c, k, epochs, lr, spike_gain, seed):
    rng = np.random.RandomState(seed * 7919 + 1)
    Xppmi = Xppmi_c - Xppmi_c.min()
    Nc, H = Xppmi.shape
    W = rng.randn(k, H) * 0.01
    order = np.arange(Nc)
    for _ in range(epochs):
        S = _poisson_spikes(Xppmi * spike_gain, rng)
        Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
        rng.shuffle(order)
        for i in order:
            x = Sn[i]; y = W @ x
            W += lr * np.outer(y, x)
            np.clip(W, -5.0, 5.0, out=W)
    St = _poisson_spikes(Xppmi * spike_gain, rng)
    Stn = St / (np.linalg.norm(St, axis=1, keepdims=True) + 1e-9)
    return (W @ Stn.T).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-hub", type=int, default=2000)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--host-alpha", type=float, default=0.75)
    p.add_argument("--budgets", default="1,3,10,30,100", help="spike_gain values (expected spikes per PPMI unit)")
    p.add_argument("--out", default="research/findings/raw/_l1_spiking_oja_smoke.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [float(b) for b in args.budgets.split(",")]
    t0 = time.time()

    C, labels, S_true = build_real_corpus(42, args.n_hub)
    Xppmi = ppmi_matrix(C, args.host_alpha); Xppmi_c = center_cols(Xppmi)
    offline = _pearson_vs_Strue(pca_lowrank_sim(Xppmi, args.k), S_true)
    print(f"[L1 spiking-Oja smoke] {C.shape[0]} concepts x {C.shape[1]} hubs; rate-Oja ceiling +0.481, "
          f"offline {offline:+.3f}", flush=True)

    results = []
    best = None
    for g in budgets:
        ps, gens, sats, perms, spk = [], [], [], [], []
        for s in seeds:
            codes, mean_spikes = oja_spiking(Xppmi_c, args.k, args.epochs, args.lr, g, s)
            ps.append(_pearson_vs_Strue(_cos_sim(codes), S_true))
            gg, ch = heldout_generalization(codes, labels)
            gens.append(gg); spk.append(mean_spikes)
            satc = faithful_saturating_spiking(Xppmi_c, args.k, args.epochs, 0.02, g, s)
            sats.append(_pearson_vs_Strue(_cos_sim(satc), S_true))
            rng = np.random.RandomState(s * 2718281 + 1); perm = rng.permutation(labels)
            S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
            perms.append(_pearson_vs_Strue(_cos_sim(codes), S_perm))
        rec = {"spike_gain": g, "mean_spikes_per_concept": float(np.mean(spk)),
               "pearson_mean": float(np.mean(ps)), "pearson_seeds": ps, "gen_mean": float(np.mean(gens)),
               "sat_mean": float(np.mean(sats)), "perm_mean": float(np.mean(perms)), "chance": ch}
        results.append(rec)
        if best is None or rec["pearson_mean"] > best["pearson_mean"]:
            best = rec
        print(f"  [budget gain={g:6.1f}  ~{rec['mean_spikes_per_concept']:7.0f} spk/concept] "
              f"Oja-spiking Pearson={rec['pearson_mean']:+.3f} {['%+.3f'%x for x in ps]}  gen={rec['gen_mean']:.3f}  "
              f"| sat-ctrl {rec['sat_mean']:+.3f}  perm {rec['perm_mean']:+.3f}", flush=True)

    frac = best["pearson_mean"] / 0.481
    clean = (abs(best["perm_mean"]) <= 0.15) and (abs(best["sat_mean"]) <= 0.15)
    if best["pearson_mean"] >= 0.70 * 0.481 and best["pearson_mean"] >= 0.30 and best["gen_mean"] > best["chance"] + 0.10 and clean:
        verdict = "SPIKING_INPUT_SURVIVES_GO"
        why = (f"at spike budget gain={best['spike_gain']} (~{best['mean_spikes_per_concept']:.0f} spikes/concept) "
               f"spiking-Oja reaches {best['pearson_mean']:+.3f} = {frac:.0%} of the rate ceiling (+0.481), "
               f"generalizes {best['gen_mean']:.3f}, controls clean (sat {best['sat_mean']:+.3f}, perm "
               f"{best['perm_mean']:+.3f}) -> the learned structure SURVIVES the input being spikes; the "
               f"spiking build is de-risked on the INPUT axis (the learning-spiking gap is the follow-on).")
    elif best["pearson_mean"] >= 0.30 and clean:
        verdict = "SPIKING_INPUT_PARTIAL"
        why = (f"spiking-Oja best {best['pearson_mean']:+.3f} ({frac:.0%} of rate +0.481) at "
               f"~{best['mean_spikes_per_concept']:.0f} spikes/concept -> partial survival; the rate->spike "
               f"cost is real but not fatal -> note the budget needed; the spiking build needs the spike count.")
    else:
        verdict = "SPIKING_INPUT_COLLAPSES"
        why = (f"spiking-Oja never clears +0.30 (best {best['pearson_mean']:+.3f} at gain "
               f"{best['spike_gain']}) -> the input-spiking (Poisson) noise destroys the structure -> the "
               f"rate->spike loss is the wall on the input axis; flag BEFORE the weeks-scale commit.")
    print(f"\n{'='*92}\n  SPIKING SMOKE VERDICT: {verdict}\n  {why}\n{'='*92}", flush=True)
    print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    out = {"verdict": verdict, "why": why, "best": best, "rate_ceiling": 0.481, "offline_pca": offline,
           "results": results, "seeds": seeds, "n_hub": args.n_hub, "k": args.k}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
