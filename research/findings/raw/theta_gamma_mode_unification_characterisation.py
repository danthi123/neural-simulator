"""Theta-gamma mode-unification: capacity-edge / noise-robustness /
vocab-scaling characterisation.

Three cheap-first follow-up probes to the algebra-PASS, run as one
comprehensive characterisation. Each probe holds two of {load, noise
level, vocab size} fixed at the algebra-PASS defaults and sweeps the
third, measuring multi-seed where BOTH readouts (order-bearing AND
order-invariant) clear the frozen 0.80 bar. Together they map the
algebra's mode-unification capacity boundary and inform the
biologized spiking implementation's design budget.

PRE-REGISTERED reading (fixed; never tuned):
- Per probe per condition: PASS iff BOTH readouts multi-seed-mean >=
  0.80 (the same frozen bar throughout).
- For each probe, the highest value with BOTH readouts PASSing is
  the "ceiling" for that axis; the lowest value with EITHER readout
  missing is the "first miss"; either-or both is reported.
- The combined characterisation gives the algebra's mode-unification
  capacity envelope on the three axes (load, noise, vocab) at the
  fixed FHRR phasor dimension N_dim=512.

Pure numpy; no GPU; no spiking; no protected/frozen/moat module
modified; no autograd. FHRR primitives reused from the algebra
probe by import. Plain ASCII.
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

# The textbook FHRR primitives + the readout helpers from the
# algebra-PASS probe, byte-unchanged via reuse.
from research.findings.raw.theta_gamma_mode_unification_probe import (
    random_phasor, bind, unbind, bundle, similarity, nearest_match,
    build_vocab_and_positions, encode_sequence,
    order_bearing_readout, order_invariant_readout,
    BAR, N_DIM, N_GAMMA_SLOTS,
)

SEEDS = [42, 43, 44]
N_TRIALS = 200

# --- Sweep axes (fixed pre-registered) ------------------------------
# (a) Capacity-edge: hold vocab=32, noise=0, sweep load.
CAPACITY_VOCAB = 32
CAPACITY_LOADS = [2, 3, 5, 7]    # bounded by N_GAMMA_SLOTS = 7

# (b) Noise-robustness: hold vocab=32, load=5, sweep noise std on the
# encoded code C. Substrate-realistic noise envelope from the FHRR-
# biologization arc: pattern-separation + mean-centring gives CV
# around 0.1; raw spiking activity is around CV 1.6. Sweep covers
# both regimes.
NOISE_VOCAB = 32
NOISE_LOAD = 5
NOISE_STDS = [0.00, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]

# (c) Vocab-scaling: hold load=5, noise=0, sweep vocabulary size.
VOCAB_LOAD = 5
VOCAB_SIZES = [32, 64, 128, 256]


def _add_noise(C, std, rng):
    """Add Gaussian noise to the complex code C at the given standard
    deviation. Independent real + imaginary noise."""
    if std <= 0.0:
        return C
    noise = (rng.normal(0.0, std, size=C.shape) +
             1j * rng.normal(0.0, std, size=C.shape))
    return C + noise


def _run_cell(seed, n_vocab, n_dim, load, noise_std, n_trials,
              n_slots=N_GAMMA_SLOTS):
    """Run n_trials at one (seed, vocab, load, noise) cell. Returns
    (order_bearing_accuracy, order_invariant_accuracy)."""
    vocab, positions = build_vocab_and_positions(
        seed, n_vocab, n_slots, n_dim)
    sample_rng = np.random.default_rng(seed + 7)
    noise_rng = np.random.default_rng(seed + 13)
    ob_ok = oi_ok = 0
    for _ in range(n_trials):
        items_idx = tuple(int(x) for x in sample_rng.choice(
            n_vocab, size=load, replace=False))
        C = encode_sequence(items_idx, vocab, positions)
        C = _add_noise(C, noise_std, noise_rng)
        ob = order_bearing_readout(C, positions, vocab, load)
        oi = order_invariant_readout(C, positions, vocab, load)
        if ob == items_idx:
            ob_ok += 1
        if oi == tuple(sorted(items_idx)):
            oi_ok += 1
    return ob_ok / n_trials, oi_ok / n_trials


def sweep_capacity():
    print("\n=== (a) CAPACITY-EDGE SWEEP "
          "(vocab=32, noise=0, sweep load) ===", flush=True)
    print(f"{'load':>5}  {'order-bearing mean':>20}  "
          f"{'order-invariant mean':>22}", flush=True)
    table = {}
    for load in CAPACITY_LOADS:
        obs = []; ois = []
        for seed in SEEDS:
            ob, oi = _run_cell(seed, CAPACITY_VOCAB, N_DIM, load,
                               0.0, N_TRIALS)
            obs.append(ob); ois.append(oi)
        ob_m = float(np.mean(obs)); oi_m = float(np.mean(ois))
        table[load] = {"order_bearing_mean": ob_m,
                       "order_invariant_mean": oi_m,
                       "order_bearing_per_seed": obs,
                       "order_invariant_per_seed": ois}
        ob_tag = ">=" if ob_m >= BAR else "<"
        oi_tag = ">=" if oi_m >= BAR else "<"
        print(f"  L={load:>2}  {ob_m:>14.4f} {ob_tag}{BAR}  "
              f"{oi_m:>16.4f} {oi_tag}{BAR}", flush=True)
    return table


def sweep_noise():
    print(f"\n=== (b) NOISE-ROBUSTNESS SWEEP "
          f"(vocab=32, load=5, sweep noise std) ===", flush=True)
    print(f"{'noise':>7}  {'order-bearing mean':>20}  "
          f"{'order-invariant mean':>22}", flush=True)
    table = {}
    for noise in NOISE_STDS:
        obs = []; ois = []
        for seed in SEEDS:
            ob, oi = _run_cell(seed, NOISE_VOCAB, N_DIM, NOISE_LOAD,
                               noise, N_TRIALS)
            obs.append(ob); ois.append(oi)
        ob_m = float(np.mean(obs)); oi_m = float(np.mean(ois))
        table[noise] = {"order_bearing_mean": ob_m,
                        "order_invariant_mean": oi_m,
                        "order_bearing_per_seed": obs,
                        "order_invariant_per_seed": ois}
        ob_tag = ">=" if ob_m >= BAR else "<"
        oi_tag = ">=" if oi_m >= BAR else "<"
        print(f"  std={noise:>4.2f}  {ob_m:>14.4f} {ob_tag}{BAR}  "
              f"{oi_m:>16.4f} {oi_tag}{BAR}", flush=True)
    return table


def sweep_vocab():
    print(f"\n=== (c) VOCAB-SCALING SWEEP "
          f"(load=5, noise=0, sweep vocab) ===", flush=True)
    print(f"{'vocab':>5}  {'order-bearing mean':>20}  "
          f"{'order-invariant mean':>22}", flush=True)
    table = {}
    for vocab in VOCAB_SIZES:
        obs = []; ois = []
        for seed in SEEDS:
            ob, oi = _run_cell(seed, vocab, N_DIM, VOCAB_LOAD,
                               0.0, N_TRIALS)
            obs.append(ob); ois.append(oi)
        ob_m = float(np.mean(obs)); oi_m = float(np.mean(ois))
        table[vocab] = {"order_bearing_mean": ob_m,
                        "order_invariant_mean": oi_m,
                        "order_bearing_per_seed": obs,
                        "order_invariant_per_seed": ois}
        ob_tag = ">=" if ob_m >= BAR else "<"
        oi_tag = ">=" if oi_m >= BAR else "<"
        print(f"  V={vocab:>3}  {ob_m:>14.4f} {ob_tag}{BAR}  "
              f"{oi_m:>16.4f} {oi_tag}{BAR}", flush=True)
    return table


def _summarise_axis(table, axis_name):
    """For one sweep axis, find the last value where both readouts
    PASS, and the first value where either misses."""
    keys = sorted(table.keys())
    last_both = None
    first_miss = None
    first_miss_which = None
    for k in keys:
        ob_pass = table[k]["order_bearing_mean"] >= BAR
        oi_pass = table[k]["order_invariant_mean"] >= BAR
        if ob_pass and oi_pass:
            last_both = k
        else:
            if first_miss is None:
                first_miss = k
                first_miss_which = []
                if not ob_pass:
                    first_miss_which.append("order_bearing")
                if not oi_pass:
                    first_miss_which.append("order_invariant")
    return {"axis": axis_name,
            "last_value_both_PASS": last_both,
            "first_value_any_miss": first_miss,
            "first_miss_readouts": first_miss_which}


def main():
    print("=== theta-gamma mode-unification: "
          "comprehensive characterisation probe ===", flush=True)
    print(f"FHRR algebra: N_dim={N_DIM}, gamma slots="
          f"{N_GAMMA_SLOTS}, seeds={SEEDS}, trials/cell={N_TRIALS}",
          flush=True)
    print(f"frozen bar={BAR}", flush=True)

    capacity = sweep_capacity()
    noise = sweep_noise()
    vocab = sweep_vocab()

    print(f"\n=== ALGEBRA MODE-UNIFICATION CAPACITY ENVELOPE ===",
          flush=True)
    cap_s = _summarise_axis(capacity, "load (vocab=32, noise=0)")
    noise_s = _summarise_axis(noise, "noise std (vocab=32, load=5)")
    vocab_s = _summarise_axis(vocab, "vocab (load=5, noise=0)")
    for s in (cap_s, noise_s, vocab_s):
        print(f"  axis {s['axis']}:", flush=True)
        print(f"    last value both readouts PASS: "
              f"{s['last_value_both_PASS']}", flush=True)
        print(f"    first value any miss: "
              f"{s['first_value_any_miss']} ({s['first_miss_readouts']})",
              flush=True)

    out = {
        "n_dim": N_DIM, "n_gamma_slots": N_GAMMA_SLOTS,
        "seeds": list(SEEDS), "n_trials": N_TRIALS, "bar": BAR,
        "capacity_sweep_vocab": CAPACITY_VOCAB,
        "noise_sweep_vocab": NOISE_VOCAB, "noise_sweep_load": NOISE_LOAD,
        "vocab_sweep_load": VOCAB_LOAD,
        "capacity_sweep": {str(k): v for k, v in capacity.items()},
        "noise_sweep": {str(k): v for k, v in noise.items()},
        "vocab_sweep": {str(k): v for k, v in vocab.items()},
        "envelope_summary": {
            "capacity": cap_s,
            "noise": noise_s,
            "vocab": vocab_s,
        },
    }
    out_path = os.path.join(
        _HERE, "theta_gamma_mode_unification_characterisation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
