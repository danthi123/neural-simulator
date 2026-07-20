"""gap#4 CREDIT-vs-FORWARD diagnostic (the decisive open control from the 2026-07-20 tuning-boundary finding).

The on-bridge spiking BDSP net reaches ~0.56 on cleanxor while the numpy-backprop oracle (SAME in->hid->out arch)
reaches ~0.99 — is that a CREDIT boundary (BDSP fails to learn useful weights) or a FORWARD/READOUT boundary (the
spiking net cannot EXPRESS the function even in principle)?

RESERVOIR CONTROL (clean, no oracle-weight-install confound): freeze input->hidden at its RANDOM init (NO BDSP), read
each sample's hidden firing pattern (baseline-subtracted to defeat the hidden_bias swamp), and train a numpy readout on
those RANDOM-hidden features.
  - reservoir readout HIGH (>> input-linear floor, ~oracle): the spiking forward + a random hidden layer is EXPRESSIVE
    (reservoir computing works on this substrate) => the function IS representable => the boundary is that BDSP fails to
    learn a useful readout/features (a CREDIT/learning boundary).
  - reservoir readout LOW (~ input-linear floor): the spiking hidden representation itself is too weak => a
    FORWARD/READOUT boundary, independent of credit.
NO sim/ edit (reuse-by-import; reads public arrays)."""
import sys; sys.path.insert(0, "/home/dant123/Projects/sim")
import os; os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json
import numpy as np
from research.runners._d1_onbridge_learn_to_accuracy_derisk import (
    OnBridgeBDSPNet, _load_task, _numpy_oracle_heldout, _numpy_singlelayer_floor)


def _hidden_features(net, X, settle_steps, differential=True):
    """Per-sample hidden firing-rate vector (random reservoir, apical OFF, NO learning). differential subtracts the
    all-low-input baseline so the INPUT-DEPENDENT hidden modulation is read, not the hidden_bias-swamped common rate."""
    base = None
    if differential:
        net._reset_membrane(); net._set_apical(None); net.cfg.bdsp_learning_rate = 0.0
        net._set_input_drive(np.zeros(net.n_bits)); net._run(settle_steps, accumulate_hidden=True)
        base = net._last_hid_rate.copy()
    feats = []
    for x in X:
        net._reset_membrane(); net._set_apical(None); net.cfg.bdsp_learning_rate = 0.0
        net._set_input_drive(np.asarray(x)); net._run(settle_steps, accumulate_hidden=True)
        h = net._last_hid_rate.copy()
        feats.append(h - base if base is not None else h)
    return np.asarray(feats)


def _ridge_readout_acc(Ftr, ytr, Fte, yte, n_classes, ridge=1.0):
    """Train a one-vs-rest ridge (least-squares) readout on features; held-out argmax accuracy."""
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-6
    Ztr = np.concatenate([(Ftr - mu) / sd, np.ones((len(Ftr), 1))], 1)
    Zte = np.concatenate([(Fte - mu) / sd, np.ones((len(Fte), 1))], 1)
    Y = np.eye(n_classes)[ytr]                                    # one-hot targets
    W = np.linalg.solve(Ztr.T @ Ztr + ridge * np.eye(Ztr.shape[1]), Ztr.T @ Y)
    pred = (Zte @ W).argmax(1)
    return float((pred == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cleanxor")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--in-pop", type=int, default=2)
    ap.add_argument("--pool-out", type=int, default=6)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--fwd-wmean", type=float, default=6.0)
    ap.add_argument("--fwd-wjit", type=float, default=0.5)
    ap.add_argument("--in-hi", type=float, default=750.0)
    ap.add_argument("--in-lo", type=float, default=40.0)
    ap.add_argument("--hidden-bias", type=float, default=520.0)
    ap.add_argument("--output-bias", type=float, default=520.0)
    ap.add_argument("--train-subset", type=int, default=400)
    ap.add_argument("--test-subset", type=int, default=200)
    ap.add_argument("--json", default="research/findings/raw/_gap4_credit_vs_forward.json")
    a = ap.parse_args()

    # a minimal args namespace for the net constructor (reservoir: no BDSP training, so credit knobs are inert)
    class NS: pass
    results = []
    for seed in a.seeds:
        (Xtr, ytr), (Xte, yte), n_bits = _load_task(a.task, seed, 4)
        n_classes = int(max(ytr.max(), yte.max())) + 1
        # subsets (match the on-bridge arm's speed subset)
        rng = np.random.default_rng(seed + 11)
        itr = rng.permutation(len(Xtr))[:a.train_subset]; ite = np.random.default_rng(seed + 12).permutation(len(Xte))[:a.test_subset]
        Xtr_b, ytr_b, Xte_b, yte_b = Xtr[itr], ytr[itr], Xte[ite], yte[ite]

        oracle = _numpy_oracle_heldout(n_bits, max(a.hidden, 32), Xtr, ytr, Xte, yte, 500, 0.5, 64, seed)
        floor = _numpy_singlelayer_floor(n_bits, Xtr, ytr, Xte, yte, 500, 0.5, 64, seed)

        # RESERVOIR net: random init, NO BDSP training (couple_soma irrelevant; feedback fixed)
        net = OnBridgeBDSPNet(seed, n_bits, hidden=a.hidden, in_pop=a.in_pop, pool_out=a.pool_out,
                              microcircuit=True, fwd_wmean=a.fwd_wmean, fwd_wjit=a.fwd_wjit,
                              in_hi=a.in_hi, in_lo=a.in_lo, hidden_bias=a.hidden_bias, output_bias=a.output_bias,
                              couple_soma=False, soma_g=0.0)
        Ftr = _hidden_features(net, Xtr_b, a.settle_steps, differential=True)
        Fte = _hidden_features(net, Xte_b, a.settle_steps, differential=True)
        # also the RAW (non-differential) hidden read, and the input-bit linear readout, for comparison
        Ftr_raw = _hidden_features(net, Xtr_b, a.settle_steps, differential=False)
        Fte_raw = _hidden_features(net, Xte_b, a.settle_steps, differential=False)

        res_diff = _ridge_readout_acc(Ftr, ytr_b, Fte, yte_b, n_classes)
        res_raw = _ridge_readout_acc(Ftr_raw, ytr_b, Fte_raw, yte_b, n_classes)
        inp_lin = _ridge_readout_acc(Xtr_b.astype(float), ytr_b, Xte_b.astype(float), yte_b, n_classes)
        act = float((Ftr != 0).mean())
        r = {"seed": seed, "task": a.task, "oracle": oracle, "floor": floor,
             "reservoir_hidden_diff": res_diff, "reservoir_hidden_raw": res_raw, "input_linear": inp_lin,
             "hidden_feat_active_frac": act}
        results.append(r)
        print(f"[seed {seed}] {a.task}: oracle={oracle:.3f} floor={floor:.3f} input-linear={inp_lin:.3f} | "
              f"RESERVOIR hidden(diff)={res_diff:.3f} hidden(raw)={res_raw:.3f} | hid-feat-active={act:.2f}", flush=True)

    # verdict
    md = np.mean([r["reservoir_hidden_diff"] for r in results])
    mi = np.mean([r["input_linear"] for r in results])
    mo = np.mean([r["oracle"] for r in results])
    print(f"\n[VERDICT] mean reservoir-hidden(diff)={md:.3f} vs input-linear-floor={mi:.3f} vs oracle={mo:.3f}")
    if md > mi + 0.08 and md > 0.75:
        print("  => FORWARD IS EXPRESSIVE (random-hidden reservoir separates the classes >> input-linear) "
              "=> the ~0.56 on-bridge floor is a CREDIT/LEARNING boundary, NOT a forward/readout limit.")
    elif md <= mi + 0.05:
        print("  => the random-hidden reservoir ~= input-linear floor => the spiking HIDDEN representation is the limit "
              "=> a FORWARD/READOUT boundary (independent of credit).")
    else:
        print("  => INTERMEDIATE: the hidden adds some but not oracle-level separability; inspect.")
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(results, open(a.json, "w"), indent=2)
    print(f"-> {a.json}")


if __name__ == "__main__":
    main()
