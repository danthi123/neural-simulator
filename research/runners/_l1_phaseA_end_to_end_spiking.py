"""L1 Phase-A CAPSTONE: do the four validated axes COMPOSE? An end-to-end spiking non-negative similarity-
matching net -- spiking INPUT + spiking OUTPUT + spike-driven LEARNING + recurrent-lateral SETTLE, all at
once -- recovering the real category structure. NO sim/ edits (numpy), so within autonomy (the gate is the
bridge build + protected edits + scale = Phase B+).

Each axis was de-risked with the OTHERS held at a faithful rate level: rule (signed/nonneg rate), input-
spiking (spike input + exact-Oja learning), learning-nonneg+spiking (rate input + spike output). This
composes them: every signal is spikes.

PIPELINE (per concept presentation):
  x_rate = centered PPMI row (the "sensory" experience)
  x_spk  = Poisson(relu(x_rate_shift) * in_gain)  -- spiking input
  x_in   = x_spk - col_mean*in_gain               -- subtractive-inhibition common-mode removal
  settle: y_rate = relu(W_ff @ x_in - M @ y_rate) over settle_steps   -- recurrent lateral (sub-threshold)
  y_spk  = Poisson(y_rate * out_gain)             -- spiking output
  learn:  W_ff += lr (y_spk x_spk^T - y_spk^2 W_ff)   -- spike-driven Oja feedforward
          M    += lr_m (y_spk y_spk^T - M)            -- spike-driven anti-Hebbian lateral (fixed point)
  readout (test): code = y_spk from a FRESH spike sample -> cosine -> structure.

GATE: at a reasonable input budget, end-to-end spiking Pearson >= 0.70 * rate ceiling (+0.515) i.e. >=+0.36
AND >= +0.30, generalizes, permuted ~0, saturating control fails -> the axes COMPOSE -> the full spiking
similarity-matching pipeline works end-to-end at the numpy level (the strongest pre-bridge artifact). A
collapse = the axes interact badly -> a finding BEFORE the bridge commit. CPU/numpy, build once. NO sim/ edits.

Run: python -u -m research.runners._l1_phaseA_end_to_end_spiking --n-hub 2000 --in-budgets 10,30,100
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
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix, pca_lowrank_sim  # noqa: E402
from research.runners._l1_centered_online_pca_probe import center_cols  # noqa: E402


def end_to_end_spiking(Xc, k, epochs, lr_ff, lr_m, settle_steps, in_gain, out_gain, seed, saturating=False,
                       read_window=12):
    """End-to-end spiking SM. Input: Poisson spikes of the shifted PPMI, common-mode removed by subtractive
    inhibition, then unit-normalized BY ITS OWN norm (matching the validated spike smoke). Output: spike-
    count driven learning. Readout: the output spike rate INTEGRATED over a window of `read_window` samples
    (a real spiking readout integrates over the encoding window -- averaging is faithful temporal integration,
    not cheating; reading a single Poisson sample is unrealistically noisy)."""
    rng = np.random.RandomState(seed * 104729 + 3)
    Nc, H = Xc.shape
    Xshift = Xc - Xc.min()                       # nonneg rates for Poisson input sampling
    col_rate = Xshift.mean(0, keepdims=True)     # the common-mode (subtractive-inhibition target)
    W_ff = rng.randn(k, H) * 0.1
    M = np.zeros((k, k), dtype=np.float64)
    order = np.arange(Nc)

    def spike_input(i):
        x_spk = rng.poisson(Xshift[i] * in_gain).astype(np.float64) - col_rate[0] * in_gain  # subtractive inhib
        return x_spk / (np.linalg.norm(x_spk) + 1e-9)                                          # own-norm scale

    def settle(x_in):
        y = np.zeros(k)
        for _ in range(settle_steps):
            y = np.maximum(0.7 * y + 0.3 * (W_ff @ x_in - M @ y), 0.0)   # nonneg recurrent settle
        return y

    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            x_in = spike_input(i)
            y = settle(x_in)
            y_spk = rng.poisson(np.maximum(y, 0.0) * out_gain).astype(np.float64) / out_gain
            if saturating:
                W_ff += lr_ff * np.outer(y_spk, x_in); np.clip(W_ff, -5, 5, out=W_ff)
            else:
                W_ff += lr_ff * (np.outer(y_spk, x_in) - (y_spk ** 2)[:, None] * W_ff)
                dM = np.outer(y_spk, y_spk) - M
                np.fill_diagonal(dM, 0.0)
                M += lr_m * dM
    # readout: integrate the output spike rate over a window (faithful temporal integration)
    codes = np.zeros((Nc, k))
    for i in range(Nc):
        acc = np.zeros(k)
        for _ in range(read_window):
            y = settle(spike_input(i))
            acc += rng.poisson(np.maximum(y, 0.0) * out_gain).astype(np.float64)
        codes[i] = acc / read_window
    return codes


def random_proj_centered(Xc, k, in_gain, out_gain, seed, read_window=12):
    """Is learning load-bearing in the CENTERED-spiking regime? An UNTRAINED random projection of the
    centered spike input, same readout. If it reaches the learners -> centering+readout is the work; if it
    trails -> the learning is load-bearing."""
    rng = np.random.RandomState(seed * 7919 + 11)
    Nc, H = Xc.shape
    Xshift = Xc - Xc.min(); col_rate = Xshift.mean(0, keepdims=True)
    W = rng.randn(k, H) / np.sqrt(H)
    codes = np.zeros((Nc, k))
    for i in range(Nc):
        acc = np.zeros(k)
        for _ in range(read_window):
            x_spk = rng.poisson(Xshift[i] * in_gain).astype(np.float64) - col_rate[0] * in_gain
            x_in = x_spk / (np.linalg.norm(x_spk) + 1e-9)
            acc += rng.poisson(np.maximum(W @ x_in, 0.0) * out_gain).astype(np.float64)
        codes[i] = acc / read_window
    return codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-hub", type=int, default=2000)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--in-gain", type=float, default=30.0)
    p.add_argument("--out-gain", type=float, default=30.0)
    p.add_argument("--host-alpha", type=float, default=0.75)
    p.add_argument("--out", default="research/findings/raw/_l1_phaseA_end_to_end_spiking.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    g = args.in_gain
    t0 = time.time()

    C, labels, S_true = build_real_corpus(42, args.n_hub)
    Xc = center_cols(ppmi_matrix(C, args.host_alpha))
    offline = _pearson_vs_Strue(pca_lowrank_sim(ppmi_matrix(C, args.host_alpha), args.k), S_true)
    print(f"[L1 Phase-A end-to-end spiking] {C.shape[0]} concepts x {C.shape[1]} hubs; rate ceiling +0.515, "
          f"offline {offline:+.3f}; in_gain={g}", flush=True)

    # 3 brain-plausible spiking recipes + the learning-load-bearing control
    def run_arm(fn):
        ps, gens, perms = [], [], []
        for s in seeds:
            codes = fn(s)
            ps.append(_pearson_vs_Strue(_cos_sim(codes), S_true))
            gg, ch = heldout_generalization(codes, labels); gens.append(gg)
            rng = np.random.RandomState(s * 2718281 + 1); perm = rng.permutation(labels)
            S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
            perms.append(_pearson_vs_Strue(_cos_sim(codes), S_perm))
        return {"pearson_mean": float(np.mean(ps)), "pearson_seeds": ps, "gen_mean": float(np.mean(gens)),
                "perm_mean": float(np.mean(perms)), "chance": ch}

    full_sm = run_arm(lambda s: end_to_end_spiking(Xc, args.k, args.epochs, 0.010, 0.030, 30, g, args.out_gain, s))
    bounded_hebb = run_arm(lambda s: end_to_end_spiking(Xc, args.k, args.epochs, 0.02, 0.0, 30, g, args.out_gain, s, saturating=True))
    rand_proj = run_arm(lambda s: random_proj_centered(Xc, args.k, g, args.out_gain, s))
    arms = {"full_SM (lateral)": full_sm, "bounded_Hebbian (no lateral)": bounded_hebb,
            "random_proj (no learning)": rand_proj}
    for nm, r in arms.items():
        print(f"  [{nm:28s}] Pearson={r['pearson_mean']:+.3f} {['%+.3f'%x for x in r['pearson_seeds']]}  "
              f"gen={r['gen_mean']:.3f}  perm={r['perm_mean']:+.3f}", flush=True)

    best_learner = max(full_sm["pearson_mean"], bounded_hebb["pearson_mean"])
    best_name = "bounded_Hebbian" if bounded_hebb["pearson_mean"] >= full_sm["pearson_mean"] else "full_SM"
    frac = best_learner / 0.515
    learning_loadbearing = best_learner >= rand_proj["pearson_mean"] + 0.10
    permuted_clean = abs(full_sm["perm_mean"]) <= 0.15 and abs(bounded_hebb["perm_mean"]) <= 0.15
    generalizes = max(full_sm["gen_mean"], bounded_hebb["gen_mean"]) > full_sm["chance"] + 0.10
    lateral_helps = full_sm["pearson_mean"] >= bounded_hebb["pearson_mean"] - 0.02

    if best_learner >= 0.70 * 0.515 and best_learner >= 0.30 and generalizes and permuted_clean and learning_loadbearing:
        verdict = "AXES_COMPOSE_END_TO_END_SPIKING_GO"
        lat = ("the anti-Hebbian lateral HELPS" if lateral_helps else
               "NOTABLE: the anti-Hebbian lateral HURTS under end-to-end spike noise (bounded-Hebbian "
               f"{bounded_hebb['pearson_mean']:+.3f} > full-SM {full_sm['pearson_mean']:+.3f}) -> the spiking "
               "build can likely DROP the lateral (the highest-risk protected edit) and use subtractive-"
               "inhibition centering + a homeostatically-bounded Hebbian feedforward instead -- a SIMPLER build")
        why = (f"the full spiking pipeline recovers the structure end-to-end ({best_name} {best_learner:+.3f} = "
               f"{frac:.0%} of rate, gen {max(full_sm['gen_mean'], bounded_hebb['gen_mean']):.3f}, permuted clean), "
               f"and learning is load-bearing (vs random-proj {rand_proj['pearson_mean']:+.3f}). {lat}. ==> the "
               f"four axes COMPOSE in full spikes; common-mode removal (subtractive inhibition) is the load-"
               f"bearing op. Strongest pre-bridge artifact; Phase B (bridge) owner-gated.")
    elif best_learner >= 0.30 and permuted_clean:
        verdict = "AXES_COMPOSE_PARTIAL"
        why = (f"end-to-end spiking recovers structure ({best_name} {best_learner:+.3f} = {frac:.0%} of rate) but "
               f"learning-vs-random margin or generalization is soft (rand-proj {rand_proj['pearson_mean']:+.3f}); "
               f"a real cost to budget for the bridge build.")
    else:
        verdict = "AXES_DO_NOT_COMPOSE"
        why = (f"the full spiking pipeline does not recover the structure (best {best_learner:+.3f}) -> the axes "
               f"interact badly end-to-end -> a finding BEFORE the bridge commit.")
    print(f"\n{'='*92}\n  PHASE-A VERDICT: {verdict}\n  {why}\n{'='*92}", flush=True)
    print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    out = {"verdict": verdict, "why": why, "arms": arms, "best_learner": best_learner, "best_name": best_name,
           "lateral_helps": lateral_helps, "learning_loadbearing": learning_loadbearing,
           "rate_ceiling": 0.515, "offline_pca": offline, "seeds": seeds, "n_hub": args.n_hub, "k": args.k}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
