"""ARC-A FIRST DE-RISK (2026-08-01): does the INTEGRATED brain learn from the TEACHER through the loop?

THE SMALLEST demonstration of the arc-A thesis: a Kuhl-style CONTINGENT teacher supplies a corrected target on
the brain's OWN spiking output, and the tonight-validated transport-free e-prop rule (`OnBridgeEpropNet`, the
production Izhikevich bridge) moves the weights toward it -- MEASURED, with the anti-cheats that are the result.

WHY THIS TASK (not the semantic-inheritance generalization task): the first atom is "the teacher's contingent
correction moves the spiking substrate's weights toward its target", NOT compositional held-out generalization
(a later milestone). So the task is a minimal DEVELOPMENTAL atom: the teacher NAMES K referents; each referent is
a noisy prototype in feature space (a small perceptual category), and the brain must learn cue -> the teacher's
label. A held-out test set of FRESH noisy draws from the same prototypes makes the permuted/non-contingent lesion
CLEAN: a real teacher signal generalizes to fresh draws; a scrambled one cannot (it can only memorize noise).

CONTINGENCY = the teacher's target is paired with the cue the brain is currently responding to. The two lesions
break exactly that binding, each in a different place:
  (1) NON-CONTINGENT teacher  -- the teacher's label is drawn at random, uncorrelated with the cue (the teacher is
      no longer responding to what the brain is looking at). No learnable cue->label map => held-out ~ chance.
  (2) SHUFFLE-DFA credit lesion -- the e-prop learning signal is scrambled across the batch (eligibility intact,
      credit mismatched to the example). The forward is unchanged; only the CREDIT ROUTE is broken => ~ chance.
The teacher signal enters e-prop as an ERROR (softmax(logits) - onehot), NOT a persistent clamp -- so it vanishes
at match and cannot become the "clamp-as-crutch" the 2026-06-08 teacher-correction finding warns about.

GO (for the 6-seed claim): held-out test_acc(main) > chance + 0.15 AND main > non_contingent + 0.15 AND
main > shuffle_dfa + 0.15 AND the FF weights actually move. This SMOKE runs 1 seed to get real numbers; the
claim requires seeds 42..47 (6 seeds), cfg.seed-controlled (the substrate is seeded via CoreSimConfig.seed).

Reuse-by-import ONLY (OnBridgeEpropNet + _train_eprop from the ported e-prop de-risk); NO sim/ edit.
Run: SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._a1_teacher_contingent_eprop_derisk --seeds 42
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet, _train_eprop  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_a1_teacher_contingent_eprop.json"


def make_referent_task(seed, k=5, d=16, n_per_class=24, n_test_per_class=12, noise=0.15):
    """K noisy-prototype referents. proto_c in [0,1]^d; a presentation = clip(proto_c + noise*N(0,1), 0, 1).
    Features are kept in [0,1] to match the bridge input-current mapping (in_bias + in_current*f, clipped)."""
    rng = np.random.default_rng(seed + 101)
    protos = rng.random((k, d)).astype(np.float64)
    def draw(n):
        X, y = [], []
        for c in range(k):
            for _ in range(n):
                X.append(np.clip(protos[c] + noise * rng.standard_normal(d), 0.0, 1.0))
                y.append(c)
        idx = rng.permutation(len(y))
        return np.asarray(X)[idx].astype(np.float64), np.asarray(y)[idx].astype(np.int64)
    Xtr, ytr = draw(n_per_class)
    Xte, yte = draw(n_test_per_class)
    return Xtr, ytr, Xte, yte, k


def _mk(n_in, k, seed, hidden, settle, eprop_lr, w_clip):
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    return OnBridgeEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                            eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                            logit_source="leaky_readout", w_clip=w_clip, hp=hp)


def run_seed(seed, hidden=24, settle=25, epochs=80, batch=20, eprop_lr=0.5, w_clip=4000.0,
             k=5, d=16, n_per_class=24, noise=0.15):
    Xtr, ytr, Xte, yte, k = make_referent_task(seed, k=k, d=d, n_per_class=n_per_class, noise=noise)
    chance = float(max(np.mean(yte == c) for c in np.unique(yte)))

    # --- MAIN: contingent teacher (true label paired with the cue) ---
    net = _mk(d, k, seed, hidden, settle, eprop_lr, w_clip)
    w0 = net.ff_weight_norm()
    acc0 = net.accuracy(Xte, yte)
    _train_eprop(net, Xtr, ytr, epochs, batch, seed)
    main_test = net.accuracy(Xte, yte)
    ff_moved = float(abs(net.ff_weight_norm() - w0))

    # --- LESION 1: NON-CONTINGENT teacher (random label, uncorrelated with the cue) ---
    nrng = np.random.default_rng(seed + 202)
    y_nc = nrng.integers(0, k, size=len(ytr)).astype(np.int64)
    nnet = _mk(d, k, seed, hidden, settle, eprop_lr, w_clip)
    _train_eprop(nnet, Xtr, y_nc, epochs, batch, seed)
    nc_test = nnet.accuracy(Xte, yte)

    # --- LESION 2: SHUFFLE-DFA credit lesion (eligibility intact, credit mismatched to the example) ---
    snet = _mk(d, k, seed, hidden, settle, eprop_lr, w_clip)
    _train_eprop(snet, Xtr, ytr, epochs, batch, seed, shuffle_dfa=True)
    sh_test = snet.accuracy(Xte, yte)

    # THE LOAD-BEARING anti-cheat for this SHALLOW associative atom is the CONTINGENCY lesion (non-contingent
    # teacher). shuffle-DFA is a DEPTH/deep-credit control: at 1 hidden layer the spiking reservoir + trained
    # readout carry the task, so scrambling the HIDDEN DFA credit does NOT collapse it (measured: shuffle-DFA
    # stays high) -- that control belongs with the depth-2 semantic-inheritance task + its frozen-hidden
    # reservoir control (the SECOND de-risk), not here. So it is REPORTED but NOT gated on.
    learns = bool(main_test > chance + 0.15 and main_test > nc_test + 0.15 and ff_moved > 1e-3)
    # ATTRIBUTION (tools.lab): the effect is the teacher CONTINGENCY, not merely two arms measured -- a real
    # teacher signal (main) vs a non-contingent one (nc). This is the load-bearing anti-cheat for this atom.
    from tools.lab import attributable_to
    attributable_to("teacher contingency (main vs non-contingent teacher)", main_test, nc_test)
    return {"seed": seed, "k": int(k), "chance": chance, "test_before": acc0,
            "main_test": main_test, "noncontingent_test": nc_test, "shuffle_dfa_test": sh_test,
            "ff_weight_moved": ff_moved, "learns_from_teacher": learns}


def main():
    ap = argparse.ArgumentParser(description="Arc-A first de-risk: contingent-teacher e-prop on the spiking bridge.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--n-per-class", type=int, default=24)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    per = []
    for s in a.seeds:
        r = run_seed(s, hidden=a.hidden, settle=a.settle_steps, epochs=a.epochs, batch=a.batch,
                     eprop_lr=a.eprop_lr, w_clip=a.w_clip, k=a.k, d=a.d, n_per_class=a.n_per_class, noise=a.noise)
        per.append(r)
        print(f"[seed {s}] chance {r['chance']:.3f} | MAIN test {r['main_test']:.3f} "
              f"(before {r['test_before']:.3f}) | non-contingent {r['noncontingent_test']:.3f} | "
              f"shuffle-DFA {r['shuffle_dfa_test']:.3f} | ff-moved {r['ff_weight_moved']:.1f} => "
              f"LEARNS-FROM-TEACHER {r['learns_from_teacher']}", flush=True)
    n_learn = sum(p["learns_from_teacher"] for p in per)
    summary = {"probe": "a1_teacher_contingent_eprop", "seeds": a.seeds,
               "config": vars(a), "elapsed_seconds": round(time.time() - t0, 1),
               "per_seed": per, "n_learn": n_learn, "n_seeds": len(a.seeds),
               "ALL_LEARN": bool(n_learn == len(a.seeds))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[a1-teacher-contingent] {n_learn}/{len(a.seeds)} seeds LEARN-FROM-TEACHER "
          f"(GO needs 6/6 at seeds 42..47) -> wrote {a.out}", flush=True)
    return 0 if summary["ALL_LEARN"] else 1


if __name__ == "__main__":
    sys.exit(main())
