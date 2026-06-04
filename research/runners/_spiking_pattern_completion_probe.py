"""Cheap-first de-risk for the spiking unified agent (backlog #1): does the spiking cleanup perform the
(b)-required PATTERN COMPLETION in genuine spikes?

The (b) result (2026-06-04, unified-agent benchmark) made pattern completion a REQUIRED component: composition
works only on clean concept codes, so a noisy grounded readout must be snapped to its nearest clean concept
attractor (an autoassociative cleanup) BEFORE composing. The spiking substrate `spiking_phasor_fhrr.cleanup()`
is exactly that primitive (winner-take-all to the nearest stored attractor by spike-phase similarity). This
probe confirms it recovers a NOISE-CORRUPTED phasor code to the correct attractor across a noise sweep — the
spiking analogue of the numpy id_acc curve.

Pre-registered verdict (frozen-bar discipline):
  - RECOVERY: at a noise level that substantially corrupts the code (mean self-similarity of the noisy code to
    its own attractor degraded to ~0.6), spiking cleanup recovery >= 0.95 (pattern completion works in spikes).
  - GRACEFUL DEGRADATION (anti-cheat): recovery must FALL toward chance as noise -> full randomization (it is
    NOT magically always-correct; if it were, the test would be trivial / the codes too separable to be a real
    test). i.e. recovery at the highest noise < recovery at the lowest noise.

Reuse-by-import of the validated spiking substrate; no protected-module edits. numpy/CPU.

  python -m research.runners._spiking_pattern_completion_probe
"""
from __future__ import annotations
import json

import numpy as np

from research.runners.spiking_phasor_fhrr import (
    CYCLE_STEPS, phases_to_spikes, spikes_to_phases, phase_similarity, cleanup)

N_VOCAB = 32          # clean concept attractors
N_DIM = 512
SEED = 42
N_TRIALS = 20         # noisy presentations per attractor per noise level
SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45]   # phase-noise std (fraction of a cycle)
ABSTAIN = -1.0        # no abstention for the identification test (always pick nearest)


def _noisy(symbol, sigma, rng):
    """A grounded-readout analogue: jitter each phasor neuron's spike phase by Gaussian(0, sigma)."""
    ph = spikes_to_phases(symbol) + rng.normal(0.0, sigma, size=symbol.shape[0])
    return phases_to_spikes(np.mod(ph, 1.0))


def run(seed=SEED):
    rng = np.random.default_rng(seed)
    vocab = [phases_to_spikes(rng.uniform(0, 1, size=N_DIM)) for _ in range(N_VOCAB)]
    rows = {}
    for sigma in SIGMAS:
        n_correct, n_total, self_sims = 0, 0, []
        for i, attractor in enumerate(vocab):
            for _ in range(N_TRIALS):
                noisy = _noisy(attractor, sigma, rng)
                self_sims.append(phase_similarity(noisy, attractor))   # how corrupted the code is
                idx, _ = cleanup(noisy, vocab, ABSTAIN)                 # spiking pattern completion
                n_correct += (idx == i)
                n_total += 1
        rows[sigma] = {"recovery": n_correct / n_total,
                       "mean_self_similarity": float(np.mean(self_sims))}
    return rows


def main():
    print("=== spiking pattern-completion de-risk (backlog #1 cheap-first) ===", flush=True)
    print(f"  {N_VOCAB} attractors, N_dim={N_DIM}, cycle={CYCLE_STEPS}, {N_TRIALS} noisy/attractor/level\n", flush=True)
    rows = run()
    print("  sigma  mean_self_sim  recovery", flush=True)
    for s, r in rows.items():
        print(f"  {s:<5}  {r['mean_self_similarity']:>11.3f}  {r['recovery']*100:>6.1f}%", flush=True)

    # pre-registered verdict
    # RECOVERY: find the noise level whose corrupted-code similarity is closest to ~0.6 and require recovery>=0.95
    target = min(rows.items(), key=lambda kv: abs(kv[1]["mean_self_similarity"] - 0.6))
    recovery_ok = target[1]["recovery"] >= 0.95
    # GRACEFUL DEGRADATION: recovery at max noise strictly below recovery at min noise
    lo = rows[SIGMAS[0]]["recovery"]
    hi = rows[SIGMAS[-1]]["recovery"]
    degrades_ok = hi < lo
    verdict = "RESOLVES" if (recovery_ok and degrades_ok) else "DOES_NOT_RESOLVE"

    print(f"\n  RECOVERY @ corrupted-sim~0.6 (sigma={target[0]}, sim={target[1]['mean_self_similarity']:.3f}): "
          f"{target[1]['recovery']*100:.1f}%  ({'>=' if recovery_ok else '<'} 95% bar)", flush=True)
    print(f"  GRACEFUL DEGRADATION: recovery {lo*100:.0f}% (clean) -> {hi*100:.0f}% (max noise)  "
          f"{'(degrades, anti-cheat ok)' if degrades_ok else '(does NOT degrade -- test too easy)'}", flush=True)
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    if verdict == "RESOLVES":
        print("  Spiking cleanup performs the (b)-required pattern completion: it recovers a substantially-", flush=True)
        print("  corrupted phasor code to the correct attractor, and degrades gracefully toward chance under", flush=True)
        print("  full randomization. The load-bearing new component of the spiking unified agent is de-risked.", flush=True)

    with open("research/findings/raw/spiking_pattern_completion_probe.json", "w") as f:
        json.dump({"vocab": N_VOCAB, "n_dim": N_DIM, "rows": {str(k): v for k, v in rows.items()},
                   "recovery_ok": recovery_ok, "degrades_ok": degrades_ok, "verdict": verdict}, f, indent=2)
    print("\n  wrote research/findings/raw/spiking_pattern_completion_probe.json", flush=True)
    return 0 if verdict == "RESOLVES" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
