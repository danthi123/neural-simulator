"""Theta-gamma mode-unification: cheap-first numpy probe.

Tests whether the FHRR phase-coded vector-symbolic algebra supports
BOTH order-bearing AND order-invariant readout from the SAME encoded
code at usable accuracy, where the encoding represents an ordered
K-item sequence by binding each item to its gamma-slot position
phasor and bundling.

The catalog-documented Lisman-Idiart N.16 mechanism (see design doc
`docs/plans/2026-05-23-theta-gamma-mode-unification-design.md`) the
owner explicitly flagged on 2026-05-19 as "never built" and load-
bearing for the conversational path. The same cheap-first pattern
the FHRR-biologization arc used (numpy algebra probe first; spiking
implementation second if the algebra works).

PRE-REGISTERED reading (fixed; never tuned):
- PASS: BOTH order-bearing AND order-invariant multi-seed-mean >=
  the frozen 0.80 bar at every load {2, 3, 5}. The algebra supports
  unified bidirectional readout from one code. A spiking biologized
  implementation is the next pre-registered step (subject to a
  fresh dedicated adversarial review of this probe before any
  claim).
- NEGATIVE_ORDER_BEARING_ONLY: order-bearing PASSes, order-invariant
  misses. Per-position unbinding works; marginal scoring does not.
- NEGATIVE_ORDER_INVARIANT_ONLY: order-invariant PASSes, order-
  bearing misses. Marginal scoring works; per-position unbinding
  crosstalks too much.
- NEGATIVE_BOTH: neither mode clears the bar at this dim/vocab.

Pure numpy; no GPU; no spiking; no protected/frozen/moat module
modified; no automatic differentiation. Reuse-by-import-style only
for self-contained correctness -- the FHRR primitives are inline
because they are textbook (random_phasor, bind = elementwise
complex multiply, unbind = multiply by conjugate, bundle = sum,
cleanup = real inner product argmax). Plain ASCII.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# =====================================================================
# Pre-registered constants (fixed; never tuned). The frozen 0.80
# compositional bar is the SAME bar the vocab-scaling thread used.
# =====================================================================
N_DIM = 512                  # FHRR phasor dimension (matches vocab-scaling)
N_VOCAB = 32                 # vocabulary size (mid-tier between 16, 64)
LOADS = [2, 3, 5]            # standard compositional loads
N_TRIALS = 200               # trials per load per seed
N_GAMMA_SLOTS = 7            # 7 gamma slots per theta cycle (Lisman-Idiart)
SEEDS = [42, 43, 44]         # multi-seed
BAR = 0.80                   # frozen compositional bar


# ---------------------------------------------------------------------
# Textbook FHRR primitives (pure complex phasors; numpy).
# ---------------------------------------------------------------------
def random_phasor(n_dim, rng):
    """Return a random unit-magnitude complex phasor (each component
    is exp(i * uniform-random-phase))."""
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_dim)
    return np.exp(1j * phases)


def bind(a, b):
    """Bind = elementwise phase addition = elementwise complex
    multiply. The standard FHRR binding."""
    return a * b


def unbind(c, b):
    """Unbind = elementwise phase subtraction = multiply by complex
    conjugate. The standard FHRR unbinding."""
    return c * np.conj(b)


def bundle(phasors):
    """Bundle = sum of phasors (the standard FHRR bundling). Not
    normalized -- the magnitude shrinks with load, which is what
    drives the load-capacity ceiling."""
    return np.sum(phasors, axis=0)


def similarity(u, v):
    """FHRR similarity = real inner product / dimension (cosine-like
    on unit-magnitude phasors). For nearest-match queries."""
    return float(np.real(np.dot(np.conj(u), v))) / float(u.size)


def nearest_match(query, vocab_phasors):
    """Return the index of the vocabulary phasor most similar to the
    query (argmax over similarity)."""
    return int(np.argmax([similarity(query, vp)
                           for vp in vocab_phasors]))


# ---------------------------------------------------------------------
# The mode-unification probe core.
# ---------------------------------------------------------------------
def build_vocab_and_positions(seed, n_vocab, n_slots, n_dim):
    """Build a deterministic per-seed vocabulary + position phasor
    set. Positions are FIXED across all trials per seed; they
    represent the gamma-slot phasors in one theta cycle."""
    rng = np.random.default_rng(seed)
    vocab = [random_phasor(n_dim, rng) for _ in range(n_vocab)]
    positions = [random_phasor(n_dim, rng) for _ in range(n_slots)]
    return vocab, positions


def encode_sequence(items_idx, vocab, positions):
    """Encode an ordered K-item sequence as the bundle of
    (item, position) bindings. items_idx[k] = index of the item at
    gamma-slot k."""
    return bundle([bind(vocab[items_idx[k]], positions[k])
                   for k in range(len(items_idx))])


def order_bearing_readout(C, positions, vocab, K):
    """For each of the first K positions, unbind it from C and
    nearest-match to the vocabulary. Returns the K-tuple of
    recovered item indices (in slot order)."""
    return tuple(nearest_match(unbind(C, positions[k]), vocab)
                 for k in range(K))


def order_invariant_readout(C, positions, vocab, K):
    """Score each vocabulary item by sum over slots of the
    inner-product of (unbind C at slot) with the item. Return the
    top-K items by score, sorted by index (so we can compare to the
    encoded set without order)."""
    scores = []
    for w_idx, w in enumerate(vocab):
        score = sum(similarity(unbind(C, positions[k]), w)
                    for k in range(K))
        scores.append((score, w_idx))
    scores.sort(reverse=True)   # highest score first
    topK_idx = sorted(scores[k][1] for k in range(K))
    return tuple(topK_idx)


def run_one_seed(seed, n_vocab, n_slots, n_dim, loads, n_trials):
    """Per-seed: for each load, run n_trials random sequences,
    measure order-bearing accuracy (exact match of the K-tuple) and
    order-invariant accuracy (exact match of the unordered set)."""
    vocab, positions = build_vocab_and_positions(
        seed, n_vocab, n_slots, n_dim)
    rng = np.random.default_rng(seed + 7)
    per_load = {}
    for load in loads:
        assert load <= n_slots, (
            f"load {load} exceeds gamma slots {n_slots}")
        ob_ok = oi_ok = 0
        for _ in range(n_trials):
            # Sample a random ORDERED sequence of K distinct items.
            items_idx = tuple(int(x) for x in rng.choice(
                n_vocab, size=load, replace=False))
            C = encode_sequence(items_idx, vocab, positions)
            ob = order_bearing_readout(C, positions, vocab, load)
            oi = order_invariant_readout(C, positions, vocab, load)
            if ob == items_idx:
                ob_ok += 1
            if oi == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / n_trials,
            "order_invariant_accuracy": oi_ok / n_trials,
            "n_trials": n_trials,
        }
    return per_load


def main():
    print("=== theta-gamma mode-unification: cheap-first numpy probe ===",
          flush=True)
    print(f"FHRR algebra: N_dim={N_DIM}, vocab={N_VOCAB}, gamma slots="
          f"{N_GAMMA_SLOTS}, loads={LOADS}, trials/load={N_TRIALS}",
          flush=True)
    print(f"seeds={SEEDS}; frozen bar={BAR}", flush=True)

    seed_results = {}
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        per_load = run_one_seed(seed, N_VOCAB, N_GAMMA_SLOTS, N_DIM,
                                LOADS, N_TRIALS)
        seed_results[seed] = per_load
        for load in LOADS:
            e = per_load[load]
            print(f"  L={load}: order-bearing="
                  f"{e['order_bearing_accuracy']:.4f}  "
                  f"order-invariant={e['order_invariant_accuracy']:.4f}",
                  flush=True)

    # Multi-seed aggregate.
    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    print(f"          order-bearing       order-invariant", flush=True)
    agg = {}
    ob_all_pass = True
    oi_all_pass = True
    for load in LOADS:
        ob = [seed_results[s][load]["order_bearing_accuracy"]
              for s in SEEDS]
        oi = [seed_results[s][load]["order_invariant_accuracy"]
              for s in SEEDS]
        ob_mean = float(np.mean(ob))
        oi_mean = float(np.mean(oi))
        agg[load] = {
            "order_bearing_mean": ob_mean,
            "order_bearing_per_seed": ob,
            "order_invariant_mean": oi_mean,
            "order_invariant_per_seed": oi,
        }
        if ob_mean < BAR:
            ob_all_pass = False
        if oi_mean < BAR:
            oi_all_pass = False
        print(f"  L={load}:  {ob_mean:.4f} "
              f"({'>=' if ob_mean >= BAR else '<'}{BAR})        "
              f"{oi_mean:.4f} ({'>=' if oi_mean >= BAR else '<'}{BAR})",
              flush=True)

    # Pre-registered reading.
    print(f"\n=== VERDICT ===", flush=True)
    if ob_all_pass and oi_all_pass:
        verdict = "MODE_UNIFICATION_PASS"
        print("  BOTH order-bearing AND order-invariant clear the frozen "
              "0.80 bar multi-seed at every tested load. The FHRR algebra "
              "supports unified bidirectional readout from one code -- "
              "the catalog-documented Lisman-Idiart N.16 mechanism is "
              "algebraically realisable on the project's chosen "
              "compositional substrate. Subject to a fresh dedicated "
              "adversarial review on this probe before any capability "
              "claim. The spiking biologized implementation is the next "
              "pre-registered step.", flush=True)
    elif ob_all_pass and not oi_all_pass:
        verdict = "NEGATIVE_ORDER_BEARING_ONLY"
        print("  Order-bearing PASSes; order-invariant misses. The "
              "per-position unbind recovers ordered items but the "
              "marginal scoring across positions does not recover the "
              "unordered set. The unification claim fails on the "
              "order-invariant side.", flush=True)
    elif oi_all_pass and not ob_all_pass:
        verdict = "NEGATIVE_ORDER_INVARIANT_ONLY"
        print("  Order-invariant PASSes; order-bearing misses. The "
              "marginal scoring recovers the unordered set but per-"
              "position unbinding crosstalks too much. The unification "
              "claim fails on the order-bearing side.", flush=True)
    else:
        verdict = "NEGATIVE_BOTH"
        print("  Neither order-bearing nor order-invariant clears the "
              "frozen 0.80 bar at this dim/vocab. The FHRR algebra at "
              f"N_dim={N_DIM} does not support unified bidirectional "
              "readout for these loads.", flush=True)

    out = {
        "n_dim": N_DIM, "n_vocab": N_VOCAB, "n_gamma_slots": N_GAMMA_SLOTS,
        "loads": LOADS, "n_trials": N_TRIALS, "seeds": list(SEEDS),
        "bar": BAR,
        "per_seed": {str(s): {str(l): v for l, v in d.items()}
                     for s, d in seed_results.items()},
        "aggregate": {str(l): v for l, v in agg.items()},
        "verdict": verdict,
    }
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "theta_gamma_mode_unification_probe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
