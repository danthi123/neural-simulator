"""FHRR abstention probe: can a spiking-phasor FHRR composition layer
PRESERVE the project's no-confabulation property?

Design context: docs/plans/2026-05-22-phase-coded-VSA-composition-design.md

The project's distinctive contribution is trustworthy memory that
ABSTAINS ("I don't know") on ungroundable queries rather than
confabulating. FHRR clean-up, as Orchard & Jarvis describe it, is an
argmax over the vocabulary -- it ALWAYS returns the nearest vocabulary
vector, even for a garbage input. By construction that is
confabulation, not abstention.

But FHRR exposes a natural abstention signal: the clean-up SIMILARITY.
A groundable query (cue a fact that was encoded) yields a recovered
vector close to the bound filler -> high top-similarity. An
ungroundable query (cue a symbol with no encoded fact) yields a noise
vector -> low top-similarity to every vocabulary item. If groundable
and ungroundable top-similarities are cleanly separable, a fixed
threshold IS an abstention moat: emit the clean-up answer above the
threshold, abstain below it.

This probe measures that separation under realistic spike-timing
jitter. It tests the project-distinctive open question for the
phase-coded VSA arc: does FHRR composition preserve no-confabulation?

PRE-REGISTERED decision rule (fixed; never tuned):
- PASS: a single fixed similarity threshold classifies groundable
  (-> answer) vs ungroundable (-> abstain) queries at >= 95% accuracy
  at loads {2,3,5} under biological-precision jitter (sigma=0.05).
  -> FHRR composition can carry an abstention moat; the no-confab
  property is preservable. Strong green light for the bridge build.
- NEGATIVE: the distributions overlap (no threshold reaches 95%)
  -> FHRR clean-up confabulates and routes to a moat-design question
  before the bridge build.

Standalone numpy. ENGINEERING ceiling-clarification, non-load-bearing.
No protected/frozen/moat module touched. No autograd.
"""
from __future__ import annotations

import json
import sys

import numpy as np

N_CUES = 8
N_FILLERS = 8
LOADS = [2, 3, 5]
DIM = 512
JITTER = 0.05          # biological-precision spike-timing jitter
T_STEPS = 1000
N_TRIALS = 400
SEED = 42
ACC_BAR = 0.95         # pre-registered separation-accuracy bar


def random_phases(n, rng):
    return rng.uniform(0.0, 1.0, size=n)


def spike_realize(phase, t_steps, jitter_sigma, rng):
    t = phase * t_steps + rng.normal(0.0, jitter_sigma * t_steps, size=phase.shape)
    return np.mod(np.round(t) / t_steps, 1.0)


def bind(a, b):
    return np.mod(a + b, 1.0)


def unbind(a, b):
    return np.mod(a - b, 1.0)


def bundle(phase_list):
    z = np.sum([np.exp(2j * np.pi * p) for p in phase_list], axis=0)
    return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)


def similarity(a, b):
    return float(np.mean(np.cos(2.0 * np.pi * (a - b))))


def cleanup_top_similarity(recovered, vocab):
    """The clean-up's top-similarity -- the abstention signal."""
    return max(similarity(recovered, v) for v in vocab)


def run_load(load, n_trials, rng):
    """Collect top-similarity for groundable + ungroundable queries."""
    groundable_top = []
    ungroundable_top = []
    for _ in range(n_trials):
        cues = [random_phases(DIM, rng) for _ in range(N_CUES)]
        fillers = [random_phases(DIM, rng) for _ in range(N_FILLERS)]
        cue_idx = list(rng.choice(N_CUES, size=load, replace=False))
        fill_idx = list(rng.choice(N_FILLERS, size=load, replace=True))
        facts = list(zip(cue_idx, fill_idx))
        bound = []
        for (c, f) in facts:
            cue_sp = spike_realize(cues[c], T_STEPS, JITTER, rng)
            fill_sp = spike_realize(fillers[f], T_STEPS, JITTER, rng)
            bound.append(spike_realize(bind(cue_sp, fill_sp), T_STEPS, JITTER, rng))
        composite = spike_realize(bundle(bound), T_STEPS, JITTER, rng)
        # groundable: the encoded cues
        for c in cue_idx:
            cue_sp = spike_realize(cues[c], T_STEPS, JITTER, rng)
            rec = spike_realize(unbind(composite, cue_sp), T_STEPS, JITTER, rng)
            groundable_top.append(cleanup_top_similarity(rec, fillers))
        # ungroundable: the cues NOT in any fact
        for c in range(N_CUES):
            if c in cue_idx:
                continue
            cue_sp = spike_realize(cues[c], T_STEPS, JITTER, rng)
            rec = spike_realize(unbind(composite, cue_sp), T_STEPS, JITTER, rng)
            ungroundable_top.append(cleanup_top_similarity(rec, fillers))
    return np.array(groundable_top), np.array(ungroundable_top)


def best_threshold_accuracy(g, u):
    """Best single fixed threshold separating groundable (>= thr ->
    answer) from ungroundable (< thr -> abstain); return (thr, acc)."""
    cand = np.unique(np.concatenate([g, u]))
    best_acc, best_thr = 0.0, 0.0
    for thr in cand:
        acc = (np.mean(g >= thr) * len(g) + np.mean(u < thr) * len(u)) / (len(g) + len(u))
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, best_acc


def main():
    print("=== FHRR abstention probe (no-confabulation preservation) ===")
    print(f"vocab {N_CUES}x{N_FILLERS}; loads={LOADS}; dim={DIM}; "
          f"jitter sigma={JITTER}; trials={N_TRIALS}; acc bar={ACC_BAR}")
    rng = np.random.default_rng(SEED)

    per_load = {}
    all_pass = True
    for load in LOADS:
        g, u = run_load(load, N_TRIALS, rng)
        thr, acc = best_threshold_accuracy(g, u)
        per_load[load] = {
            "groundable_mean": float(np.mean(g)),
            "groundable_min": float(np.min(g)),
            "ungroundable_mean": float(np.mean(u)),
            "ungroundable_max": float(np.max(u)),
            "best_threshold": thr,
            "separation_accuracy": acc,
            "n_groundable": int(len(g)),
            "n_ungroundable": int(len(u)),
        }
        if acc < ACC_BAR:
            all_pass = False
        print(f"  L={load}: groundable sim mean={np.mean(g):.3f} "
              f"min={np.min(g):.3f} | ungroundable sim mean={np.mean(u):.3f} "
              f"max={np.max(u):.3f} | best-threshold sep acc={acc:.4f}")

    print(f"\n=== VERDICT ===")
    if all_pass:
        verdict = "ABSTENTION_PRESERVABLE"
        print(f"  A fixed similarity threshold separates groundable from "
              f"ungroundable at >= {ACC_BAR} accuracy at all loads {LOADS} "
              f"under biological-precision jitter.")
        print(f"  --> ABSTENTION PRESERVABLE: FHRR composition can carry a "
              f"no-confabulation moat (answer above threshold, abstain "
              f"below). Strong green light for the biological-scale bridge "
              f"integration.")
    else:
        verdict = "ABSTENTION_FRAGILE"
        print(f"  The groundable / ungroundable similarity distributions "
              f"overlap -- no fixed threshold reaches {ACC_BAR} at some load.")
        print(f"  --> ABSTENTION FRAGILE: FHRR clean-up confabulates; routes "
              f"to a moat-design question before the bridge build.")

    out = {
        "n_cues": N_CUES, "n_fillers": N_FILLERS, "loads": LOADS, "dim": DIM,
        "jitter": JITTER, "t_steps": T_STEPS, "n_trials": N_TRIALS,
        "acc_bar": ACC_BAR, "seed": SEED,
        "per_load": {str(k): v for k, v in per_load.items()},
        "verdict": verdict,
    }
    with open("research/findings/raw/fhrr_abstention_probe.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/fhrr_abstention_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
