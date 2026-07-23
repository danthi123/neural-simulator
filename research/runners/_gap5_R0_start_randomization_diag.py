"""gap#5 candidate-#3 R0 DIAGNOSTIC (research-gate step, ~1 min).

Tests the reframe: does the numpy gamma-WTA GO (1.000 forward) ride start=A (assembly 0),
i.e. are the STORED WEIGHTS near-symmetric with the forward direction supplied by the ignition
point, NOT by weight asymmetry?

Reproduces ARM_B (gamma-WTA + post-fire silence) with start=0 (the GO) vs start=random over the
SAME extracted W. Expected (analytically ~0.33 for n_mem=3): start=random collapses 1.000 -> ~0.33
=> PROVES the GO rode start=A + validates the pivot to asymmetric-weight ENCODING.
"""
import numpy as np
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence, SEQ_CFG
from research.runners._gap5_gamma_wta_replay_derisk import _extract_W, _replay, _forward_frac


def _arm_start(W, n_mem, seed, noise, n_trials, randomize_start):
    rng = np.random.default_rng(int(seed) * 7919 + 1)  # self_avoid arm (matches _arm_stats seed for self_avoid=True)
    fr = []
    for _ in range(n_trials):
        start = int(rng.integers(n_mem)) if randomize_start else 0
        fr.append(_forward_frac(_replay(W, n_mem, True, rng, noise, start=start)))
    fr = np.asarray(fr)
    return dict(mean=float(fr.mean()), full=float((fr >= 0.999).mean()))


def main():
    seeds = [42, 43, 44]
    n_mem, noise, n_trials = 3, 8.0, 400
    cfg = dict(SEQ_CFG)
    cfg["n_mem"] = n_mem; cfg["within_events"] = 30; cfg["within_refresh"] = 8
    cfg["chain_fwd"] = 24; cfg["chain_rev"] = 0; cfg["rank1_encode"] = True; cfg["overlap_draw"] = False
    print(f"[R0] start-randomization control on the numpy gamma-WTA isolation (n_mem={n_mem}, noise={noise}, trials={n_trials})")
    print(f"[R0] REFRAME PREDICTION: start=0 -> ~1.000 forward (the GO);  start=random -> ~0.33 (chance) if weights are symmetric.\n")
    for s in seeds:
        prep = _prepare_sequence(s, cfg)
        W = _extract_W(prep, n_mem)
        adj_fwd = float(np.mean([W[i, i + 1] for i in range(n_mem - 1)]))
        adj_rev = float(np.mean([W[i + 1, i] for i in range(n_mem - 1)]))
        s0 = _arm_start(W, n_mem, s, noise, n_trials, randomize_start=False)
        sr = _arm_start(W, n_mem, s, noise, n_trials, randomize_start=True)
        verdict = "RODE-START (symmetric weights)" if (s0["mean"] > 0.8 and sr["mean"] < 0.5) else "?"
        print(f"  seed {s}: adj_fwd={adj_fwd:.1f} adj_rev={adj_rev:.1f} asym={adj_fwd - adj_rev:+.2f} "
              f"| start=0 fwd={s0['mean']:.3f} full={s0['full']:.3f} "
              f"| start=RANDOM fwd={sr['mean']:.3f} full={sr['full']:.3f} => {verdict}")


if __name__ == "__main__":
    main()
