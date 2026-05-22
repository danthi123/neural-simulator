"""Activity-level integration with attractor-denoised symbols --
biologization shortcut 2, deeper form, on the REAL substrate.

The activity-level integration decisive run was a NEGATIVE: deriving the
composition symbol directly from a single raw substrate-activity
observation fails, because the substrate's per-neuron activity has a
measured trial-to-trial coefficient of variation of about 1.6 -- far
too noisy (integrated 0.33-0.42, composition-only 0.36-0.42, against the
0.80 bar).

Biologization shortcut 3 built an attractor network (the Threshold
Phasor Associative Memory) that denoises a noisy phasor by settling it
toward a clean fixed point. Shortcut 2's deeper form puts that machinery
in front of the symbol: derive a noisy symbol from the substrate
activity, then SETTLE it through an attractor whose fixed points are the
consolidated concept representations, and use the settled (denoised)
phasor as the grounded symbol.

The cheap-first probe (attractor_grounded_symbol_probe.py) found this
REACHABLE in an independent-noise model -- at the measured-substrate
noise level the attractor recovered the correct concept 99.6% of the
time. But that probe models independent per-component phase noise; the
real substrate noise may be correlated, which would not average out.
This runner is the genuine test: it reuses the REAL captured substrate
activity (the activity_level_integration cache -- the same per-neuron
observations, with whatever correlated structure they have) and inserts
the attractor-denoising stage.

Pipeline (per seed; reuses the cached real activity, no new GPU run):
1. Load the cached per-neuron substrate activity (M observations/word).
2. Derive a phasor symbol from each activity observation (the same
   fixed random projection the activity-level integration used).
3. Build an attractor whose fixed points are the consolidated concept
   symbols (the deriver applied to the K-averaged activity per word).
4. SETTLE each derived noisy symbol through the attractor -> the
   denoised, grounded symbol.
5. Compose the denoised symbols through the spiking composition layer;
   measure against the frozen 0.80 bar.

The ONLY change from the activity-level integration runner is the
attractor-denoising stage in step 4 -- so the integrated accuracy is
directly comparable to that runner's NEGATIVE (0.33-0.42).

PRE-REGISTERED reading (fixed; never tuned):
- If integrated multi-seed mean >= 0.80 at loads {2,3,5}: attractor
  denoising rescues the activity-grounded symbol on the real substrate
  -- shortcut 2's deeper form works.
- If it stays below 0.80: even attractor denoising cannot rescue the
  real substrate activity (the real noise is too severe or too
  correlated to average out); the honest ceiling is that a grounded
  symbol needs a cleaner substrate representation, which routes to
  improving the recognition substrate itself.

Reuse-by-import only: the activity-level integration cache + deriver,
the spiking composition layer, the attractor -- all byte-unchanged. No
protected/frozen/moat module modified. No automatic differentiation.
Plain ASCII.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.activity_level_integration import (
    capture_seed, make_deriver, CACHE_DIR, N_DIM, LOADS, N_TRIALS, BAR,
    M_OBS, K_VOCAB, DERIV_SEED, SEEDS,
)
from research.runners.unified_per_regime_monitor_runner import (
    _direct_pool_target,
)
from research.runners.spiking_phasor_fhrr import (
    phases_to_spikes, phase_similarity,
)
from research.runners.resonate_fire_fhrr import (
    ResonateFireFHRR, ResonateFireTPAM,
    ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
)


def phasor_to_spikes(z, t_steps):
    """A settled complex phasor state -> the spike train it encodes."""
    return phases_to_spikes(np.mod(np.angle(z) / (2.0 * np.pi), 1.0), t_steps)


def run_one_seed(seed, m_obs, k_vocab, n_trials):
    """Activity-level integration with attractor-denoised symbols for
    one substrate seed (reuses the cached real activity)."""
    print(f"\n--- seed {seed} ---")
    cache_path = os.path.join(CACHE_DIR, f"full_seed{seed}.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"missing activity cache {cache_path}; run "
            f"activity_level_integration.py (full) first")
    obs, clean, _slices, _all_pools, words = capture_seed(
        seed, cache_path, m_obs)

    d_act = obs[words[0]].shape[1]
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    t_steps = net.t_steps

    # Consolidated concept symbols = the deriver applied to the
    # K-averaged activity per word. These are the attractor fixed
    # points (the grounded, consolidated concept representations).
    consolidated = {w: deriver(obs[w][:k_vocab].mean(axis=0)) for w in words}
    attractor = ResonateFireTPAM([consolidated[w] for w in words])

    def denoise(word, obs_idx):
        """Derive a noisy symbol from one activity observation, then
        settle it through the attractor -> the denoised grounded
        symbol (as a spike train). Uses the resonate-and-fire transfer
        in closed form (fast=True) -- biologization step 1 validated
        the genuine time-stepped rf_resonate realizes exactly this
        transfer (primitive error ~0.002; settle-level equivalence
        confirmed: 0 recognition mismatches over a 16-concept check)."""
        noisy = deriver(obs[word][obs_idx])
        z, _ = attractor.settle_annealed(noisy, ANNEAL_THETA_LOW,
                                         ANNEAL_THETA_HIGH, ANNEAL_ITERS,
                                         fast=True)
        return phasor_to_spikes(z, t_steps)

    cue_words = [w for w in words
                 if _direct_pool_target(w).startswith(("noun_pool_",
                                                        "verb_pool_"))]
    filler_words = [w for w in words
                    if _direct_pool_target(w).startswith("adjective_pool_")]
    # Clean-up vocabulary: the consolidated concept symbols.
    vocab = {fw: consolidated[fw] for fw in filler_words}

    qrng = np.random.default_rng(seed + 1)

    def true_pool(word):
        return _direct_pool_target(word)

    per_load = {}
    for load in LOADS:
        n_int_correct = n_int_total = 0
        n_comp_correct = n_comp_total = 0
        for _ in range(n_trials):
            cues = list(qrng.choice(cue_words, size=load, replace=False))
            fills = list(qrng.choice(filler_words, size=load, replace=True))
            facts = []
            for (c, f) in zip(cues, fills):
                facts.append((c, f, int(qrng.integers(m_obs)),
                              int(qrng.integers(m_obs)),
                              int(qrng.integers(m_obs))))
            composite = net.encode([
                (denoise(c, ci_enc), denoise(f, fi_enc))
                for (c, f, ci_enc, fi_enc, ci_qry) in facts])
            for (c, f, ci_enc, fi_enc, ci_qry) in facts:
                recovered = net.query(composite, denoise(c, ci_qry))
                sims = {fw: phase_similarity(recovered, vocab[fw])
                        for fw in filler_words}
                best = max(sims, key=sims.get)
                hit = (true_pool(best) == true_pool(f))
                n_int_correct += int(hit)
                n_int_total += 1
                if clean[c][ci_enc] and clean[f][fi_enc] and clean[c][ci_qry]:
                    n_comp_correct += int(hit)
                    n_comp_total += 1
        int_acc = n_int_correct / n_int_total
        comp_acc = (n_comp_correct / n_comp_total) if n_comp_total else float("nan")
        per_load[load] = {
            "integrated_accuracy": int_acc,
            "composition_only_accuracy": comp_acc,
            "n_composition_only": n_comp_total,
        }
        print(f"  L={load}: integrated acc={int_acc:.4f} | "
              f"composition-only acc={comp_acc:.4f} (n={n_comp_total})")
    return {"seed": seed, "activity_dim": int(d_act), "per_load": per_load}


def main():
    print("=== activity-level integration with ATTRACTOR-DENOISED symbols "
          "(shortcut 2, deeper form, real substrate) ===")
    print(f"seeds {SEEDS}; FHRR N_dim={N_DIM}; loads={LOADS}; bar={BAR}; "
          f"obs/word={M_OBS}; vocab-K={K_VOCAB}; reuses the real activity "
          f"cache")

    seed_results = [run_one_seed(s, M_OBS, K_VOCAB, N_TRIALS) for s in SEEDS]

    print(f"\n=== MULTI-SEED AGGREGATE ===")
    agg = {}
    all_pass = True
    for load in LOADS:
        int_accs = [r["per_load"][load]["integrated_accuracy"]
                    for r in seed_results]
        comp_accs = [r["per_load"][load]["composition_only_accuracy"]
                     for r in seed_results]
        mean_int = float(np.mean(int_accs))
        mean_comp = float(np.mean([c for c in comp_accs if c == c]))
        agg[load] = {"mean_integrated": mean_int,
                     "per_seed_integrated": int_accs,
                     "mean_composition_only": mean_comp}
        if mean_int < BAR:
            all_pass = False
        print(f"  L={load}: integrated per-seed="
              f"{['%.3f' % a for a in int_accs]} mean={mean_int:.4f} "
              f"({'>=' if mean_int >= BAR else '<'} {BAR}) | "
              f"composition-only mean={mean_comp:.4f}")

    print(f"\n=== VERDICT ===")
    if all_pass:
        verdict = "ATTRACTOR_GROUNDED_PASS"
        print("  Attractor denoising rescues the activity-grounded symbol "
              "on the REAL substrate: the integrated pipeline clears the "
              "frozen 0.80 bar multi-seed mean at all loads. Shortcut 2's "
              "deeper form works -- the grounded symbol is the attractor "
              "fixed point, no oracle lookup.")
    else:
        verdict = "ATTRACTOR_DENOISING_INSUFFICIENT"
        print("  Even attractor denoising does not lift the real-substrate "
              "integrated accuracy to 0.80. The honest ceiling: a grounded "
              "symbol needs a cleaner substrate representation; routes to "
              "improving the recognition substrate itself.")

    out = {
        "seeds": SEEDS, "n_dim": N_DIM, "loads": LOADS, "n_trials": N_TRIALS,
        "bar": BAR, "m_obs": M_OBS, "k_vocab": K_VOCAB,
        "anneal": [ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS],
        "per_seed": seed_results,
        "aggregate": {str(k): v for k, v in agg.items()},
        "verdict": verdict,
    }
    out_path = "research/findings/raw/activity_level_integration_attractor.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
