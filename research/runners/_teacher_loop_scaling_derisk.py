"""TEACHER-LOOP SCALING DE-RISK (2026-08-08): ground the "1 fact -> thousands of facts" time estimate with
MEASUREMENT. Reuses the corrective-acquisition e-prop machinery (the brain learns a fact by its OWN plasticity --
OnBridgeEpropNet FF weights moved by the transport-free e-prop rule; NO host store-write) and asks the SCALING
question the single-fact de-risk could not: teach N DISTINCT facts SEQUENTIALLY into ONE brain and measure
  (a) per-fact wall-clock throughput on cupy/3090,
  (b) the CATASTROPHIC-FORGETTING retention curve -- after teaching all facts so far, re-test EVERY earlier fact,
  (c) capacity = accuracy vs N.

THE ATOM, SCALED. Each fact i is a DISTINCT referent->patient mapping: a fresh noisy perceptual prototype ref_i
(the a1 small-perceptual-category world, host = sensory render) mapped to a distinct patient class i. The brain is
ONE OnBridgeEpropNet with a k=N_max-way leaky readout; teaching fact i = the contingent corrective e-prop update
(softmax(logits)-onehot(i)) on fresh ref_i draws. This is the SAME acquisition path as the single-fact de-risk,
now driven as a sequential curriculum.

CATASTROPHIC FORGETTING, MADE MEASURABLE. Sequential single-class corrective updates share ONE readout, so
teaching fact i pushes the readout toward class i and can suppress the weights that carried earlier classes -- the
classic CF failure mode. We measure it directly, WITHOUT a separate task-A/task-B protocol, by recording per fact:
  * acquire_acc[i]  = held-out accuracy of fact i measured IMMEDIATELY after teaching it (did it ever learn?),
  * retention at each milestone N in {1,5,10,20,40}: re-test ALL facts 0..N-1 -> fraction still correctly argmaxed.
The gap between acquire_acc (high) and end-retention (low) is the forgetting: LEARNED-then-LOST, not never-learned.
That gap is the honest diagnostic for WHAT SETS THE PACE -- if per-fact acquisition is fast and clean but retention
collapses as N grows, then BREADTH is blocked by forgetting (a mechanism prerequisite), not by throughput.

INTERLEAVED UPPER BOUND (control, --interleaved-n). Train a FRESH net on the first M facts INTERLEAVED (pooled,
shuffled) for the same per-fact epoch budget. If interleaved retains what sequential forgets, the wall is the
sequential-curriculum training regime (forgetting), not net capacity or the e-prop mechanism -- the decisive
sequential-vs-interleaved contrast for the "what sets the pace" verdict.

HONEST SCOPE. Single-seed SMOKE-scale (as instructed). We report the MEASURED per-fact throughput + the forgetting
curve, and EXTRAPOLATE to ~1000 facts with every number labelled measured-vs-extrapolated. A 3090-vs-AWS analysis:
a faster/bigger GPU speeds per-fact throughput by ~X, but the curriculum is SEQUENTIAL so cloud parallelism does
NOT speed teaching -- it speeds parallel de-risk EXPERIMENTS (independent seeds/configs), never the one brain's
one-fact-after-another schedule.

DISCIPLINE: reuse-by-import (OnBridgeEpropNet + train_batch from the a1-GO port; ReferentEnv + _predict_conf from
the single-fact corrective-acquire de-risk). NO sim/ edit. cfg.seed via the seed= arg the net passes to
CoreSimConfig.seed (NOT actual_seed_used). SIM_BACKEND=cupy (the 3090) by default; caller may override to numpy for
a plumbing smoke. Readout-norm (the leaky-readout homeostatic input scaling) is fit ONCE over the whole referent
world BEFORE the curriculum -- refitting per-fact would itself distort earlier facts' readout (an instrument choice,
declared; see the caveat).

RUN (3090, single-seed smoke as instructed):
  SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_scaling_derisk --seed 42 \
      --milestones 1 5 10 20 40 --n-max 40 --epochs 60 --n-draws 48 --interleaved-n 10 \
      --out research/findings/raw/teacher_loop_scaling_s42.json
  PLUMBING SMOKE (fast, numpy): ... SIM_BACKEND=numpy ... --n-max 3 --milestones 1 3 --epochs 8 --settle-steps 12 --interleaved-n 0
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")   # the 3090 by default; caller may set numpy for a plumbing smoke
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
# reuse-by-import: the a1-GO transport-free e-prop substrate (the brain's OWN plasticity) + the single-fact
# corrective-acquire world/read helpers. NO sim/ edit.
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet  # noqa: E402
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv, _predict_conf  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_scaling.json"

N_ACT = 2          # keep the action-context one-hot width the single-fact atom used (action fixed = index 0)
ACT_IDX = 0        # the single teaching action ("eats"); the fact is the referent->patient mapping


def _mk_net(n_in, k, seed, hidden=24, settle=25, eprop_lr=0.5, w_clip=4000.0):
    """The a1-GO OnBridgeEpropNet build (transport-free e-prop; the FF weights are the SOLE learner). Same hp as
    the single-fact corrective-acquire de-risk."""
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    return OnBridgeEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                            eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                            logit_source="leaky_readout", w_clip=w_clip, hp=hp)


def _feat(env, referent):
    """The brain's input = percept(referent) concat action-context one-hot (action fixed at ACT_IDX)."""
    a = np.zeros(N_ACT, dtype=np.float64); a[ACT_IDX] = 1.0
    return np.concatenate([env.draw(referent), a]).astype(np.float64)


def _fit_readout_norm_world(net, env, referents, seed, per_ref=6):
    """Fit the leaky-readout per-neuron eligibility mean/std ONCE over the WHOLE referent world (input statistics,
    NOT labels -- a homeostatic input scaling). Declared instrument choice: refitting per-fact would retune the
    shared readout normalization to the current referent and itself distort earlier facts' readout."""
    feats = []
    rng = np.random.default_rng(seed + 909)
    order = list(referents)
    for _ in range(per_ref):
        for r in order:
            feats.append(_feat(env, r))
    R = np.array([net._readout_elig(net._forward_record(feats[i])[0]) for i in range(len(feats))])
    net._r_mu = R.mean(axis=0)
    net._r_sigma = R.std(axis=0) + 1e-6
    _ = rng


def _teach_fact(net, X, y, epochs, batch, rng):
    """One contingent corrective-acquisition episode = the port's train_batch loop, WITHOUT the per-call
    readout-norm refit (fit once globally). e-prop moves the brain's OWN FF weights; NO host store-write."""
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_batch(X[b], y[b])


def _fact_acc(net, env, referent, cls, n=40):
    """Held-out generalization: n FRESH noisy draws of `referent` -> fraction the net argmaxes to `cls`."""
    correct = 0
    for _ in range(n):
        c, _conf = _predict_conf(net, _feat(env, referent))
        correct += int(c == cls)
    return correct / n


def _corrective_batch(env, referent, cls, n_draws):
    """n corrective micro-turns: fresh noisy `referent` draws (the cue) paired with the teacher's target class."""
    X = np.asarray([_feat(env, referent) for _ in range(n_draws)], dtype=np.float64)
    y = np.full(n_draws, int(cls), dtype=np.int64)
    return X, y


def run(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
        test_n, interleaved_n, checkpoint=None):
    """checkpoint(partial_result_dict) is called after every milestone AND after the interleaved control so an
    interrupted run (the 2026-08-08 kill at fact 39/40 lost the whole artifact) preserves its measured curve."""
    K = int(n_max)                                   # one class per fact (referent_i -> class i); chance = 1/K
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))

    # ---- the world: N_max distinct referents (fresh noisy perceptual prototypes) ----
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)                                 # instantiate the world's referents (host = sensory environment)

    # ---- ONE brain; readout-norm fit ONCE over the whole world (declared instrument choice) ----
    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    _fit_readout_norm_world(net, env, referents, seed)
    rng = np.random.default_rng(seed + 777)

    per_fact_seconds = []
    acquire_acc = []                                 # accuracy of fact i measured IMMEDIATELY after teaching it
    ff_norm0 = net.ff_weight_norm()
    retention_curve = {}                             # milestone N -> {mean_retained, per_fact_acc[list], n_recalled}
    milestone_wall = {}
    interleaved = None

    def _snapshot(partial=True):
        pfs = per_fact_seconds or [0.0]
        return {
            "seed": seed, "backend": os.environ.get("SIM_BACKEND"), "K_classes": K, "chance": chance,
            "n_max": n_max, "milestones": milestones, "partial": bool(partial),
            "facts_taught_so_far": len(per_fact_seconds),
            "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "interleaved_n": interleaved_n},
            "throughput": {"mean_seconds_per_fact": float(np.mean(pfs)),
                           "median_seconds_per_fact": float(np.median(pfs)),
                           "min_seconds_per_fact": float(np.min(pfs)), "max_seconds_per_fact": float(np.max(pfs)),
                           "per_fact_seconds": [round(x, 2) for x in per_fact_seconds]},
            "acquire_acc_immediate": [float(a) for a in acquire_acc],
            "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
            "retention_curve": retention_curve, "milestone_cumulative_wall_seconds": milestone_wall,
            "interleaved_control": interleaved,
        }

    t_run0 = time.time()
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)
        t0 = time.time()
        _teach_fact(net, X, y, epochs, batch, rng)
        dt = time.time() - t0
        per_fact_seconds.append(dt)
        acc_i = _fact_acc(net, env, r, i, n=test_n)  # immediate held-out acquisition
        acquire_acc.append(acc_i)
        N = i + 1
        print(f"[fact {N}/{n_max}] ref={r}->class{i} taught in {dt:.1f}s | immediate held-out acc {acc_i:.2f} "
              f"(chance {chance:.3f}) | cum {time.time()-t_run0:.0f}s", flush=True)
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention_curve[str(N)] = {
                "mean_retained_acc": float(np.mean(accs)),
                "frac_recalled": float(n_recalled / N),        # fraction of the N taught facts still correctly recalled
                "n_recalled": n_recalled, "N": N,
                "most_recent_fact_acc": float(accs[-1]),        # the fact just taught (recency)
                "oldest_fact_acc": float(accs[0]),              # fact 0 (the most-overwritten)
                "per_fact_acc": [float(a) for a in accs],
            }
            milestone_wall[str(N)] = round(time.time() - t_run0, 1)
            print(f"  >> MILESTONE N={N}: mean-retained {np.mean(accs):.3f} | frac-recalled {n_recalled}/{N} "
                  f"| oldest(fact0) {accs[0]:.2f} | newest {accs[-1]:.2f}", flush=True)
            if checkpoint is not None:
                checkpoint(_snapshot(partial=True))

    ff_moved_total = float(abs(net.ff_weight_norm() - ff_norm0))
    seq_wall = round(time.time() - t_run0, 1)

    # ---- INTERLEAVED UPPER BOUND (control): fresh net, first M facts pooled+shuffled, same per-fact epoch budget ----
    M = int(interleaved_n)
    if M and M >= 2:
        M = min(M, n_max)
        inet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
        _fit_readout_norm_world(inet, env, referents[:M], seed)
        irng = np.random.default_rng(seed + 4242)
        Xs, ys = [], []
        for j in range(M):
            Xj, yj = _corrective_batch(env, referents[j], j, n_draws)
            Xs.append(Xj); ys.append(yj)
        Xall = np.concatenate(Xs, 0); yall = np.concatenate(ys, 0)
        t0 = time.time()
        # SAME per-fact epoch budget as sequential (epochs passes over EACH fact's draws) => epochs passes over the
        # pooled set gives each fact `epochs` exposures too, matched compute per fact.
        _teach_fact(inet, Xall, yall, epochs, batch, irng)
        i_dt = time.time() - t0
        iaccs = [_fact_acc(inet, env, referents[j], j, n=test_n) for j in range(M)]
        i_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in iaccs))
        interleaved = {
            "M": M, "wall_seconds": round(i_dt, 1),
            "mean_retained_acc": float(np.mean(iaccs)),
            "frac_recalled": float(i_recalled / M), "n_recalled": i_recalled,
            "per_fact_acc": [float(a) for a in iaccs],
        }
        print(f"[interleaved M={M}] {i_dt:.1f}s | mean {np.mean(iaccs):.3f} | recalled {i_recalled}/{M}", flush=True)
        if checkpoint is not None:
            checkpoint(_snapshot(partial=True))

    mean_s = float(np.mean(per_fact_seconds)); med_s = float(np.median(per_fact_seconds))
    result = {
        "seed": seed, "backend": os.environ.get("SIM_BACKEND"), "K_classes": K, "chance": chance,
        "n_max": n_max, "milestones": milestones,
        "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                   "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                   "test_n": test_n, "interleaved_n": interleaved_n},
        "throughput": {"mean_seconds_per_fact": mean_s, "median_seconds_per_fact": med_s,
                       "min_seconds_per_fact": float(np.min(per_fact_seconds)),
                       "max_seconds_per_fact": float(np.max(per_fact_seconds)),
                       "per_fact_seconds": [round(x, 2) for x in per_fact_seconds],
                       "sequential_curriculum_wall_seconds": seq_wall},
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)),
        "retention_curve": retention_curve, "milestone_cumulative_wall_seconds": milestone_wall,
        "ff_weight_moved_total": ff_moved_total,
        "interleaved_control": interleaved,
    }
    return result


def _extrapolate(result, target=1000):
    """Honest extrapolation to `target` facts. Every number labelled measured-vs-extrapolated."""
    mean_s = result["throughput"]["mean_seconds_per_fact"]
    rc = result["retention_curve"]
    Ns = sorted(int(k) for k in rc)
    biggest = Ns[-1] if Ns else None
    frac_at_biggest = rc[str(biggest)]["frac_recalled"] if biggest else None
    naive_seconds = mean_s * target            # sequential: no parallelism speedup (one brain, one fact at a time)
    return {
        "target_facts": target,
        "measured_mean_seconds_per_fact": mean_s,
        "extrapolated_wall_seconds_as_is": naive_seconds,
        "extrapolated_wall_hours_as_is": naive_seconds / 3600.0,
        "extrapolated_wall_days_as_is": naive_seconds / 86400.0,
        "retention_at_largest_measured_N": {"N": biggest, "frac_recalled": frac_at_biggest},
        "breadth_blocked_by_forgetting": (frac_at_biggest is not None and frac_at_biggest < 0.5),
        "note": ("Throughput extrapolation assumes per-fact cost stays ~constant (it grows slowly with K via the "
                 "readout width; measured range in throughput.per_fact_seconds). If retention collapses "
                 "(breadth_blocked_by_forgetting), the wall-clock number is MEANINGLESS as-is: you would 'teach' "
                 "1000 facts but retain a handful -- a forgetting-mitigation mechanism (replay/consolidation, "
                 "sparse/gated readout allocation, or per-fact protected subspaces) is a PREREQUISITE before the "
                 "throughput number describes real breadth."),
    }


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop SCALING de-risk: sequentially teach N facts into one "
                                             "brain; measure per-fact throughput, catastrophic-forgetting retention, "
                                             "capacity; extrapolate to ~1000 with a 3090-vs-AWS analysis.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=40)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10, 20, 40])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=48)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--interleaved-n", type=int, default=10,
                    help="M: interleaved upper-bound control on the first M facts (0 to skip)")
    ap.add_argument("--extrapolate-to", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint(partial_result):
        # write a partial artifact after every milestone / interleaved so a kill cannot lose the measured curve.
        try:
            Path(a.out).write_text(json.dumps(
                {"probe": "teacher_loop_scaling", "seed": a.seed, "backend": os.environ.get("SIM_BACKEND"),
                 "single_seed_smoke": True, "partial": True, "elapsed_seconds": round(time.time() - t0, 1),
                 "result": partial_result,
                 "extrapolation": _extrapolate(partial_result, target=a.extrapolate_to)}, indent=2, default=str))
        except Exception as _e:
            print(f"[warn] checkpoint write failed ({type(_e).__name__}: {_e})", flush=True)

    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.interleaved_n, checkpoint=_checkpoint)
    extrap = _extrapolate(result, target=a.extrapolate_to)
    summary = {"probe": "teacher_loop_scaling", "seed": a.seed, "backend": os.environ.get("SIM_BACKEND"),
               "single_seed_smoke": True, "elapsed_seconds": round(time.time() - t0, 1),
               "result": result, "extrapolation": extrap}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    tp = result["throughput"]
    print(f"[teacher-loop-scaling] mean {tp['mean_seconds_per_fact']:.1f}s/fact (median {tp['median_seconds_per_fact']:.1f}s) "
          f"| seq-wall {tp['sequential_curriculum_wall_seconds']:.0f}s", flush=True)
    for N in sorted(int(k) for k in result["retention_curve"]):
        rcN = result["retention_curve"][str(N)]
        print(f"    retention N={N}: frac-recalled {rcN['n_recalled']}/{N} ({rcN['frac_recalled']:.2f}) | "
              f"mean-acc {rcN['mean_retained_acc']:.3f} | oldest {rcN['oldest_fact_acc']:.2f} newest {rcN['most_recent_fact_acc']:.2f}", flush=True)
    print(f"    extrapolate {extrap['target_facts']} facts as-is: {extrap['extrapolated_wall_hours_as_is']:.1f} h "
          f"({extrap['extrapolated_wall_days_as_is']:.2f} d) | breadth-blocked-by-forgetting "
          f"{extrap['breadth_blocked_by_forgetting']}", flush=True)
    print(f"[teacher-loop-scaling] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
