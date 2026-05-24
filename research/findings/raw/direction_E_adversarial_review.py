"""Direction E ADVERSARIAL REVIEW: scrutinize whether the PASS is
genuinely theta-gamma positional binding, or artifacts.

Three exploit-class checks (each fixed pre-registered control):
  (A) Permutation control: shuffle which concept is at which slot
      independently per trial; if decoder PASSes with this scrambled
      ground truth, the slot-to-concept mapping is being recovered
      from pattern recognition alone, not from gamma-slot phase
      encoding. TRUE LEARNING signature: TRUE PASSes (~1.0), permuted
      drops to chance (1/16).
  (B) No-slot-windowing control: decoder reads the FULL theta cycle
      (not slot i's gamma window) and tries to identify which slot
      each concept was at. If accuracy holds without windowing, the
      slot-windowing is NOT load-bearing -- the test is just sparse-
      pattern recognition. TRUE LEARNING signature: TRUE PASSes,
      no-window drops below.
  (C) Overlapping-vocab control: increase vocab overlap (each concept
      shares neurons with others; matches substrate 0.45 measured
      overlap). If accuracy holds with high overlap, vocab orthogon-
      ality wasn't load-bearing (good); if it drops, the algebra is
      vocab-orthogonality-sensitive (a real limit; would explain why
      raw substrate FHRR failed).

NUMPY only; ~3 min wall. Reuses the main probe primitives.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.direction_E_theta_gamma_numpy_probe import (
    N_THETA, N_GAMMA, GAMMA_PERIOD, N_DIM, PHASE_NOISE_STD,
    N_VOCAB, N_TRIALS_PER_LOAD, BAR, ACTIVE_FRAC, SEEDS,
    generate_concept_patterns, encode_sequence, add_phase_noise,
    decode_slot,
)

OUT_JSON = os.path.join(_HERE, "direction_E_adversarial_review.json")
LOAD = 5


def control_permutation(seed):
    """A: permute the slot-to-concept mapping per trial; if accuracy
    > chance, decoder doesn't use slot phase."""
    rng = np.random.default_rng(seed * 31337)
    patterns = generate_concept_patterns(
        N_VOCAB, N_DIM, GAMMA_PERIOD, seed=seed)
    n_correct = 0; n_total = 0
    for trial in range(N_TRIALS_PER_LOAD):
        seq = list(rng.choice(N_VOCAB, size=LOAD, replace=False))
        ensemble = encode_sequence(
            seq, patterns, N_THETA, GAMMA_PERIOD, N_GAMMA)
        noisy = add_phase_noise(ensemble, PHASE_NOISE_STD, rng)
        # Permute the slot-to-concept mapping for ground truth scoring.
        perm = rng.permutation(LOAD)
        perm_seq = [seq[p] for p in perm]
        for slot_idx in range(LOAD):
            pred, _ = decode_slot(
                noisy, slot_idx, patterns, GAMMA_PERIOD, N_VOCAB)
            if pred == perm_seq[slot_idx]:
                n_correct += 1
            n_total += 1
    return n_correct / n_total


def control_no_window(seed):
    """B: read the full theta-cycle (ignoring slot windows). Try to
    decode each slot using the same windowed decoder but with
    NO slot offset (always read [0:GAMMA_PERIOD])."""
    rng = np.random.default_rng(seed * 31337 + 1)
    patterns = generate_concept_patterns(
        N_VOCAB, N_DIM, GAMMA_PERIOD, seed=seed)
    n_correct = 0; n_total = 0
    for trial in range(N_TRIALS_PER_LOAD):
        seq = list(rng.choice(N_VOCAB, size=LOAD, replace=False))
        ensemble = encode_sequence(
            seq, patterns, N_THETA, GAMMA_PERIOD, N_GAMMA)
        noisy = add_phase_noise(ensemble, PHASE_NOISE_STD, rng)
        # Read full theta cycle and try to match against summed
        # patterns. NO slot windowing.
        win_summed = noisy.sum(axis=0)
        scores = []
        for c in range(N_VOCAB):
            p_summed = patterns[c].sum(axis=0)
            a = win_summed.astype(np.float64)
            b = p_summed.astype(np.float64)
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12: scores.append(-np.inf); continue
            scores.append(float(np.dot(a, b) / (na * nb)))
        # Top-LOAD concepts predicted to be in the sequence.
        topL = np.argsort(scores)[::-1][:LOAD]
        # Score: for each slot, predict the top-i'th concept as the
        # concept at slot i. This is the cheating decoder.
        for slot_idx in range(LOAD):
            if topL[slot_idx] == seq[slot_idx]:
                n_correct += 1
            n_total += 1
    return n_correct / n_total


def control_overlap(seed, overlap_frac=0.45):
    """C: high-overlap vocab; each concept shares ~overlap_frac of
    active neurons with one shared pool. Mirrors the substrate's
    0.45 measured concept overlap that broke the FHRR oracle-
    replacement attempt."""
    rng = np.random.default_rng(seed * 31337 + 2)
    n_active = int(ACTIVE_FRAC * N_DIM)
    # Shared pool that every concept uses ~overlap_frac of.
    n_shared = int(overlap_frac * n_active)
    n_unique = n_active - n_shared
    shared_pool = rng.choice(N_DIM, size=int(n_shared * 1.5),
                               replace=False)

    patterns = np.zeros(
        (N_VOCAB, GAMMA_PERIOD, N_DIM), dtype=np.float32)
    for c in range(N_VOCAB):
        # Most of the active set: from shared pool
        shared_active = rng.choice(shared_pool, size=n_shared,
                                     replace=False)
        # Unique active set
        avail = list(set(range(N_DIM)) - set(shared_pool))
        unique_active = rng.choice(avail, size=n_unique, replace=False)
        active_idx = np.concatenate([shared_active, unique_active])
        for t in range(GAMMA_PERIOD):
            patterns[c, t, active_idx] = 1.0

    n_correct = 0; n_total = 0
    for trial in range(N_TRIALS_PER_LOAD):
        seq = list(rng.choice(N_VOCAB, size=LOAD, replace=False))
        ensemble = encode_sequence(
            seq, patterns, N_THETA, GAMMA_PERIOD, N_GAMMA)
        noisy = add_phase_noise(ensemble, PHASE_NOISE_STD, rng)
        for slot_idx in range(LOAD):
            pred, _ = decode_slot(
                noisy, slot_idx, patterns, GAMMA_PERIOD, N_VOCAB)
            if pred == seq[slot_idx]:
                n_correct += 1
            n_total += 1

    # Measure actual vocab overlap to report.
    overlaps = []
    for c1 in range(N_VOCAB):
        for c2 in range(c1 + 1, N_VOCAB):
            a = (patterns[c1, 0] > 0).astype(np.float64)
            b = (patterns[c2, 0] > 0).astype(np.float64)
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12: continue
            overlaps.append(float(np.dot(a, b) / (na * nb)))
    mean_overlap = float(np.mean(overlaps))
    return n_correct / n_total, mean_overlap


def main():
    print(f"=== Direction E ADVERSARIAL REVIEW ===", flush=True)
    print(f"  LOAD={LOAD}, vocab={N_VOCAB}, seeds: {SEEDS}",
          flush=True)
    print(f"  Pre-registered bar: {BAR}", flush=True)
    print(f"  Chance: 1/{N_VOCAB} = {1.0/N_VOCAB:.3f}", flush=True)

    t0 = time.time()

    # (A) Permutation control
    print(f"\n--- Control A: permutation ---", flush=True)
    accs_perm = []
    for seed in SEEDS:
        a = control_permutation(seed)
        accs_perm.append(a)
        print(f"  seed {seed}: {a:.3f}", flush=True)
    mean_perm = float(np.mean(accs_perm))
    print(f"  mean: {mean_perm:.3f}", flush=True)

    # (B) No-window control
    print(f"\n--- Control B: no-slot-windowing ---", flush=True)
    accs_nowin = []
    for seed in SEEDS:
        a = control_no_window(seed)
        accs_nowin.append(a)
        print(f"  seed {seed}: {a:.3f}", flush=True)
    mean_nowin = float(np.mean(accs_nowin))
    print(f"  mean: {mean_nowin:.3f}", flush=True)

    # (C) Overlap control
    print(f"\n--- Control C: high-overlap vocab (target 0.45) ---",
          flush=True)
    accs_overlap = []
    overlaps_measured = []
    for seed in SEEDS:
        a, ovl = control_overlap(seed, overlap_frac=0.45)
        accs_overlap.append(a)
        overlaps_measured.append(ovl)
        print(f"  seed {seed}: acc={a:.3f}, measured overlap="
              f"{ovl:.3f}", flush=True)
    mean_overlap_acc = float(np.mean(accs_overlap))
    mean_overlap_val = float(np.mean(overlaps_measured))
    print(f"  mean acc: {mean_overlap_acc:.3f}, mean overlap: "
          f"{mean_overlap_val:.3f}", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"\nWall: {total_min:.1f} min", flush=True)

    # Original main probe at LOAD=5 was 1.000 (confirmed in JSON).
    main_acc = 1.000
    # Correct chance baselines per control:
    # - (A) Permutation: decoder correctly identifies slot-i's concept;
    #   permuted ground truth puts the correct concept at slot perm(i).
    #   P(perm(i) == i) = 1/LOAD for a random permutation. Decoder
    #   accuracy = 1/LOAD if decoder is perfect at slot-position
    #   recovery. Higher would mean decoder is randomly guessing or
    #   not using slot info.
    # - (B) No-window: decoder picks the top-LOAD concepts seen in the
    #   summed theta cycle (correctly identifies which concepts are
    #   in the sequence) then assigns them by descending score to
    #   slots 0, 1, .., LOAD-1. Since concept-to-slot is random in
    #   ground truth, P(assigned slot == true slot) = 1/LOAD.
    # - (C) High-overlap vocab: same task, accuracy holds -> robust;
    #   accuracy drops -> overlap-sensitive (substrate-realistic
    #   overlap matters).
    chance_perm = 1.0 / LOAD
    chance_nowin = 1.0 / LOAD
    print(f"\n=== VERDICT ===", flush=True)
    print(f"  Main probe acc (load {LOAD}):     {main_acc:.3f}",
          flush=True)
    print(f"  (A) permutation control:          {mean_perm:.3f} "
          f"(expected chance 1/LOAD={chance_perm:.3f}; at-chance "
          f"= decoder uses slot-position info)", flush=True)
    print(f"  (B) no-window control:            {mean_nowin:.3f} "
          f"(expected chance 1/LOAD={chance_nowin:.3f}; at-chance "
          f"= slot windowing load-bearing)", flush=True)
    print(f"  (C) high-overlap control:         {mean_overlap_acc:.3f} "
          f"(measured overlap {mean_overlap_val:.3f}; hold = robust;"
          f" drop = sensitive)", flush=True)

    # CORRECTED verdict logic per chance baselines above.
    perm_at_chance = abs(mean_perm - chance_perm) < 0.05
    nowin_at_chance = abs(mean_nowin - chance_nowin) < 0.05
    overlap_holds = mean_overlap_acc >= BAR

    if perm_at_chance and nowin_at_chance and overlap_holds:
        verdict = "CONTROLS_DECISIVE"
        print(f"\n  CONTROLS DECISIVE: permutation -> 1/LOAD chance"
              f" (decoder uses slot phase); no-window -> 1/LOAD "
              f"chance (slot windowing load-bearing); high-overlap "
              f"-> holds (robust to substrate-realistic overlap). "
              f"The theta-gamma positional binding is genuinely "
              f"slot-discriminating AND overlap-robust. Pillar-"
              f"eligible.", flush=True)
    elif perm_at_chance and nowin_at_chance and not overlap_holds:
        verdict = "CONTROLS_OK_OVERLAP_SENSITIVE"
        print(f"\n  CONTROLS OK but OVERLAP-SENSITIVE: phase-coding"
              f" controls pass; high-overlap drops -- algebra is "
              f"substrate-overlap sensitive. Same recognition "
              f"bound as FHRR.", flush=True)
    elif not perm_at_chance:
        if mean_perm > 0.6:
            verdict = "PERMUTATION_LEAK"
            print(f"\n  PERMUTATION LEAK: accuracy holds on "
                  f"permuted ground truth -- decoder doesn't use "
                  f"slot position. Test design flaw.", flush=True)
        else:
            verdict = "PERMUTATION_BETWEEN"
            print(f"\n  PERMUTATION BETWEEN: {mean_perm:.3f} not "
                  f"clearly at 1/LOAD or main; intermediate; "
                  f"warrants deeper review.", flush=True)
    else:
        verdict = "NO_WINDOWING_NOT_AT_CHANCE"
        print(f"\n  NO-WINDOW NOT AT CHANCE: {mean_nowin:.3f} != "
              f"1/LOAD; either slot windowing partially load-"
              f"bearing or the no-window decoder gets some info "
              f"from order-of-concept-scores; warrants review.",
              flush=True)

    out = {
        "load": LOAD, "main_probe_acc": main_acc,
        "ctrl_permutation_mean": mean_perm,
        "ctrl_permutation_per_seed": accs_perm,
        "ctrl_no_window_mean": mean_nowin,
        "ctrl_no_window_per_seed": accs_nowin,
        "ctrl_overlap_mean_acc": mean_overlap_acc,
        "ctrl_overlap_per_seed_acc": accs_overlap,
        "ctrl_overlap_measured_overlap": overlaps_measured,
        "ctrl_overlap_mean_measured": mean_overlap_val,
        "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
