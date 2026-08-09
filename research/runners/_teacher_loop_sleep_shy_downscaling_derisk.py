"""TEACHER-LOOP SLEEP SYNAPTIC DOWNSCALING (SHY / Tononi-Cirelli) DE-RISK (2026-08-09): close the sleep-replay
RETENTION PLATEAU by adding the COMPANION PROCESS biology runs alongside replay.

THE RE-DIAGNOSED WALL this attacks (main 8d2510d3, adversarially verified). Sleep-replay consolidation
(_teacher_loop_sleep_replay_consolidation_derisk.py) is self-generated + robust (replay 0.55 vs no-replay 0.13,
6-seed) but PLATEAUS at frac_recalled ~ 0.55, well below the interleaved ceiling (8/10 = 0.80). Two honest
negatives with teeth already RULED OUT the store and the amount of replay: a spiking-attractor engram TIES the
host-mean engram (+0.017 -- the mean is the sufficient statistic in this unimodal world) and 16x-64x more replay
work is FLAT (+0.00). So the 0.55->0.80 residual is NOT the store and NOT the replay budget -- it is a COMPANION
PROCESS the real sleeping brain runs ALONGSIDE replay. The learner is ONE OnBridgeEpropNet (spiking Izhikevich,
transport-free e-prop) reading a SHARED leaky-readout -- the interference site. Replay potentiates; nothing
renormalizes; the shared readout accumulates recency-inflated columns that dominate the argmax.

THE COMPANION PROCESS: SLEEP SYNAPTIC DOWNSCALING (the Synaptic Homeostasis Hypothesis, Tononi & Cirelli 2003,
2014; Vyazovskiy 2008 measured net synaptic depression across sleep; de Vivo 2017 / Diering 2017 ultrastructural
+ molecular confirmation). Wake POTENTIATES synapses net-up; SLEEP RENORMALIZES them net-down, PROPORTIONALLY, so
total synaptic weight returns to a homeostatic setpoint WHILE PRESERVING RELATIVE DIFFERENCES. Replay+downscaling
are two halves of ONE offline process: replay selectively RE-POTENTIATES the synapses carrying the consolidated
(important) facts, and a global/homeostatic DOWNSCALING renormalizes ALL readout synapses toward a target total
weight -- so the shared readout does not saturate, no single (recency-inflated) class column dominates, and older
facts survive relative to the noise floor. Net: replay+downscaling TOGETHER consolidate; downscaling ALONE (no
replay) just forgets (nothing re-potentiates the old facts before the renormalization).

WHERE IT ACTS (brain-based, additive, NO sim/ edit). SHY renormalizes the brain's OWN shared readout synapses --
the last FF pathway H_last->out, the weights that live in net.br.cp_connections.data[net._data_idx_flat[-1]] (the
Bellec-2020 leaky readout e-prop trains). It is a HOMEOSTATIC RENORMALIZATION of those synapses, NOT a host
bookkeeping reset: per POSTSYNAPTIC output neuron (per class column, exactly the SHY axis -- each neuron
renormalizes its own incoming total synaptic weight), multiplicatively downscale the column toward a common target
total weight, DOWNSCALE ONLY (sleep depresses; potentiation is replay's job). This preserves the relative pattern
WITHIN a class column (which hidden units carry it) while equalizing the magnitude ACROSS classes, removing the
recency-magnitude bias that the shared readout accumulates.

  WHY per-POSTSYNAPTIC-NEURON and not a single global scalar: a global scalar on the WHOLE readout is ARGMAX-INERT
  (logit_k = r.W[:,k]; scaling all of W by c scales every logit by c, argmax unchanged). SHY is per-neuron (each
  neuron renormalizes ITS OWN incoming synapses), so it differentially scales the columns -- the recency-inflated
  column comes down relative to the quiet older ones. The `global` mode below is included as an explicit INERT
  control to prove the per-neuron axis is what carries the effect.

ARMS (same net build / seed / per-fact WAKE budget / replay budget -- the ONLY difference is the sleep phase):
  * noreplay   = the scaling baseline (no sleep phase at all) -> frac_recalled ~ 1/N (the CF wall).
  * replay     = REPLAY-ONLY, the 0.55 plateau this attacks (self-replay consolidate, no downscaling).
  * replay_shy = REPLAY + SHY DOWNSCALING (the treatment): the companion process alongside replay.
  * shy_only   = SHY DOWNSCALING but NO replay (downscaling alone) -> ~no-replay floor (it is not the mechanism
                 alone: renormalizing without re-potentiating the old facts just forgets).
  * scramble   = REPLAY + SHY with the engram LABELS shuffled (stored CONTENT lesioned, identical compute) ->
                 ~chance: the retention rise needs the STORED ENGRAM CONTENT (self-generated), not the extra steps.

TEETH (single-seed SMOKE; 6-seed command below):
  (a) RETENTION RISES vs the 0.55 replay-only baseline: replay_shy frac_recalled@N > replay frac_recalled@N
      (same net/seed/epochs), rising toward the interleaved 0.80 ceiling.
  (b) SHY IS LOAD-BEARING: remove SHY (== the `replay` arm) -> retention drops back to the ~0.55 plateau. The
      replay_shy - replay gap IS the companion process's contribution.
  (c) DOWNSCALING IS NOT THE MECHANISM ALONE: shy_only (no replay) ~ noreplay floor (renormalizing without
      re-potentiating just forgets).
  (d) SELF-GENERATED CONTENT: scramble (content-lesioned, identical replay+SHY compute) forgets to ~chance ->
      the rise comes from the brain's OWN stored engram content, not extra training and not the teacher.
  (e) IMMEDIATE ACQUISITION STAYS PERFECT: acquire_acc (held-out, measured right after WAKE teaching each fact,
      BEFORE the sleep phase), in replay_shy, stays ~1.0.
  grep-verify TEACHER/WORLD ABSENT during the sleep phase (replay is self-generated, SHY touches only the readout):
      grep -n 'env' research/runners/_teacher_loop_sleep_shy_downscaling_derisk.py | grep -i draw   # -> only WAKE
      the sleep block calls _self_replay_consolidate(net, hippo, ...) [no env] + _shy_downscale(net, ...) [weights].

HONEST-NEGATIVE IS FIRST-CLASS. If SHY does NOT lift replay_shy above the replay plateau, that is a teeth-bearing
negative that NARROWS the cause: the 0.55->0.80 residual is then learner-side interference INTERNAL to the e-prop
readout dynamics (not a missing renormalization), and the next lead is the readout's gradient/competition rule.

DISCIPLINE: reuse-by-import (Hippocampus + _self_replay_consolidate from the sleep-replay consolidation de-risk;
_mk_net / _feat / _fit_readout_norm_world / _teach_fact / _fact_acc / _corrective_batch / N_ACT + ReferentEnv from
the scaling / corrective-acquire de-risks). NO sim/ edit -- SHY operates on the public cp_connections.data (the
brain's own synapses). cfg.seed via the seed= the net passes to CoreSimConfig.seed (NOT actual_seed_used).
SIM_BACKEND=cupy (3090) by default; numpy for a plumbing smoke. tools.lab attribution + a Verdict preconditions
block.

RUN (3090, single-seed smoke as instructed):
  SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_sleep_shy_downscaling_derisk --seed 42 \
      --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
      --shy-frac 1.0 --out research/findings/raw/teacher_loop_sleep_shy_s42.json
  6-SEED (GO needs the retention rise 6/6 at 42..47), run one seed per process in parallel:
    for s in 42 43 44 45 46 47; do SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_sleep_shy_downscaling_derisk --seed $s \
      --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
      --shy-frac 1.0 --out research/findings/raw/teacher_loop_sleep_shy_s$s.json & done; wait
  PLUMBING SMOKE (fast, numpy): ... SIM_BACKEND=numpy ... --n-max 3 --milestones 1 3 --epochs 8 \
      --replay-epochs 6 --replay-per-fact 12 --settle-steps 12 --test-n 20
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
# reuse-by-import: the sleep-replay HIPPOCAMPUS + the self-replay consolidation (self-generated, teacher/world
# absent) + the teacher-loop scaling machinery (net build, world features, readout-norm fit, per-fact teaching,
# held-out accuracy, corrective batch) + ReferentEnv. NO sim/ edit, NO re-derivation of the baseline.
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import (  # noqa: E402
    Hippocampus, _self_replay_consolidate,
)
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _mk_net, _feat, _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_sleep_shy.json"


# ============================ SLEEP SYNAPTIC DOWNSCALING (SHY / Tononi-Cirelli) ============================
def _readout_W(net):
    """Read the shared leaky-readout weights (H_last->out, the interference site) as an (n_pre_phys, K_phys) host
    matrix, plus the (xp array module, data buffer, flat index) needed to write them back. These are the brain's
    OWN synapses (cp_connections.data) -- the same weights e-prop moves and _logits_from reads."""
    from sim.backend import to_host
    xp = net._xp
    data = net.br.cp_connections.data
    idx = net._data_idx_flat[-1]
    W = np.asarray(to_host(data[idx]), dtype=np.float64).reshape(net.sizes_phys[-2], net.sizes_phys[-1])
    return W, xp, data, idx


def _shy_downscale(net, shy_frac=1.0, mode="percol", reference="median", floor_norm=1e-6):
    """SLEEP SYNAPTIC DOWNSCALING (SHY). Homeostatic renormalization of the brain's OWN shared readout synapses
    toward a common target total weight -- DOWNSCALE ONLY (sleep depresses; potentiation is replay's job).

      mode="percol"  (the SHY mechanism): per POSTSYNAPTIC output neuron (per class column -- the SHY axis, each
                     neuron renormalizes ITS OWN incoming total), multiplicatively scale the column DOWN toward a
                     common target = shy_frac * reference-column-norm. Columns already at/below target untouched.
                     Preserves the WITHIN-column pattern (which hidden units carry the fact) while equalizing the
                     magnitude ACROSS classes -> removes the recency-magnitude bias on the shared readout.
      mode="global"  (INERT control): one scalar over the WHOLE readout toward shy_frac*total. Argmax-invariant by
                     construction (proves the per-neuron axis is what carries the effect, not mere shrinkage).

    reference in {"median","mean","min"}: the homeostatic setpoint the columns renormalize toward (median = the
    typical column, robust to the one inflated recency column). Returns a small dict of what it did (diagnostics)."""
    W, xp, data, idx = _readout_W(net)
    col_norm = np.sqrt((W * W).sum(axis=0))                      # per output-neuron incoming L2 weight (K,)
    active = col_norm > floor_norm
    if not active.any():
        return {"applied": False, "reason": "all readout columns below floor"}
    ref_vals = col_norm[active]
    ref = {"median": np.median, "mean": np.mean, "min": np.min}[reference](ref_vals)
    target = float(shy_frac) * float(ref)
    if mode == "global":
        tot = float(np.sqrt((W * W).sum()))
        if tot <= floor_norm:
            return {"applied": False, "reason": "total readout weight below floor"}
        # renormalize the WHOLE readout toward shy_frac*total (argmax-inert control): scale <=1 only.
        scale_scalar = min(1.0, (float(shy_frac) * tot) / tot)  # == shy_frac if <1 else 1
        Wn = W * scale_scalar
        col_scale = np.full(W.shape[1], scale_scalar, dtype=np.float64)
    else:  # percol -- the SHY mechanism
        col_scale = np.ones(W.shape[1], dtype=np.float64)
        # DOWNSCALE ONLY: columns whose incoming total exceeds the target are renormalized down toward it.
        over = active & (col_norm > target) & (target > floor_norm)
        col_scale[over] = target / col_norm[over]
        Wn = W * col_scale[None, :]
    data[idx] = xp.asarray(Wn.astype(np.float32).ravel())
    return {"applied": True, "mode": mode, "reference": reference, "target_norm": target,
            "col_norm_before_min": float(col_norm.min()), "col_norm_before_max": float(col_norm.max()),
            "col_norm_before_mean": float(col_norm.mean()),
            "col_scale_min": float(col_scale.min()), "col_scale_mean": float(col_scale.mean()),
            "n_columns_downscaled": int((col_scale < 1.0 - 1e-9).sum())}


# ================================= one arm of the sequential curriculum =================================
def _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws,
             milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance,
             shy_frac, shy_mode, shy_reference):
    """Teach the referents SEQUENTIALLY into ONE brain. arm in {noreplay, replay, replay_shy, shy_only, scramble}.
    The WAKE phase is identical across arms; only the SLEEP phase differs:
        noreplay   : (no sleep phase)
        replay     : self-replay consolidate
        replay_shy : self-replay consolidate + SHY downscaling  (the companion process)
        shy_only   : SHY downscaling only (no replay)
        scramble   : self-replay consolidate (labels shuffled) + SHY downscaling  (content lesioned)
    """
    do_replay = arm in ("replay", "replay_shy", "scramble")
    do_shy = arm in ("replay_shy", "shy_only", "scramble")
    scramble = arm == "scramble"

    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    _fit_readout_norm_world(net, env, referents, seed)     # readout-norm fit ONCE over the world (as the baseline)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)          # brain-owned RNG for consolidation shuffling
    hippo = Hippocampus(seed, replay_noise=replay_noise)

    acquire_acc = []        # immediate held-out acc right after teaching each fact, BEFORE the sleep phase (teeth e)
    retention = {}
    shy_diag = []           # per-fact SHY diagnostics (what the downscaling did)
    for i, r in enumerate(referents):
        # --- WAKE: teacher teaches fact i from the world (env draws are legitimate: the sensory environment) ---
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acq = _fact_acc(net, env, r, i, n=test_n)          # IMMEDIATE acquisition (teeth e) -- before any sleep
        acquire_acc.append(acq)
        hippo.encode(X, i)                                  # the hippocampus captures the engram of this episode
        # --- SLEEP: replay (self-generated) and/or SHY downscaling (readout renormalization); noreplay SKIPS both --
        if do_replay:
            _self_replay_consolidate(net, hippo, replay_epochs, batch, brain_rng, replay_per_fact, scramble=scramble)
        if do_shy:
            d = _shy_downscale(net, shy_frac=shy_frac, mode=shy_mode, reference=shy_reference)
            d["after_fact"] = i + 1
            shy_diag.append(d)
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
            }
    return {
        "arm": arm,
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention,
        "shy_diag_last": (shy_diag[-1] if shy_diag else None),
    }


def run(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
        test_n, replay_epochs, replay_per_fact, replay_noise, shy_frac, shy_mode, shy_reference, arms_to_run):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))

    # ONE world, shared across arms so the comparison is like-for-like (same referents, same seed).
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        # fresh env draw-stream per arm (reset the env RNG) so each arm sees the SAME teaching percepts.
        env.rng = np.random.default_rng(seed + 101)
        arms[arm] = _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr,
                             w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise,
                             chance, shy_frac, shy_mode, shy_reference)
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {arm}] done in {arms[arm]['wall_seconds']:.0f}s | immediate-acq "
              f"{arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}", flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "replay_epochs": replay_epochs, "replay_per_fact": replay_per_fact,
                       "replay_noise": replay_noise, "shy_frac": shy_frac, "shy_mode": shy_mode,
                       "shy_reference": shy_reference},
            "arms": arms}


def _verdict(result):
    """Emit a Verdict preconditions block + the GO decision. The DECISIVE comparison is replay_shy vs replay (the
    0.55 replay-only plateau). TEETH:
      (a) retention rises: replay_shy frac_recalled > replay frac_recalled at the largest N.
      (b) SHY load-bearing: replay (SHY removed) sits at the ~0.55 plateau, below replay_shy by >0.10.
      (c) downscaling not the mechanism alone: shy_only ~ noreplay floor (shy_only <= noreplay + 0.10).
      (d) self-generated content: scramble (content-lesioned, same replay+SHY compute) forgets
          (scramble <= replay_shy - 0.15 AND scramble near chance).
      (e) immediate acquisition stays perfect in replay_shy (mean immediate acq >= 0.9)."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    A = result["arms"]
    rc = {a: A[a]["retention_curve"] for a in A}
    big = max((int(k) for k in rc["replay_shy"]), default=None)
    key = str(big)

    def _fr(arm):
        return rc[arm][key]["frac_recalled"] if (arm in rc and key in rc[arm]) else float("nan")

    replay_frac = _fr("replay")            # the 0.55 replay-only plateau this attacks
    shy_frac_r = _fr("replay_shy")         # replay + SHY (the treatment)
    noreplay_frac = _fr("noreplay")
    shyonly_frac = _fr("shy_only")
    scramble_frac = _fr("scramble")
    shy_acq = A["replay_shy"]["mean_acquire_acc_immediate"]
    chance = result["chance"]

    # the effect is the SHY companion process (replay_shy vs replay), and the STORED CONTENT (replay_shy vs scramble).
    attributable_to("SHY downscaling (replay_shy vs replay-only plateau)", shy_frac_r, replay_frac)
    attributable_to("stored engram content (replay_shy vs scramble, same compute)", shy_frac_r, scramble_frac)

    v = Verdict("teacher-loop sleep SHY downscaling", chance=chance)
    v.reaches("(a) retention RISES vs the 0.55 replay-only plateau", before=replay_frac, after=shy_frac_r)
    v.control("(b) SHY is load-bearing (replay_shy vs replay-only)", treatment=shy_frac_r,
              control=replay_frac, min_separation=0.10)
    v.require("(c) downscaling ALONE is not the mechanism (shy_only ~ noreplay floor)",
              shyonly_frac <= noreplay_frac + 0.10, expect=True,
              note=f"shy_only {shyonly_frac:.2f} vs noreplay {noreplay_frac:.2f}")
    v.control("(d) self-generated content (replay_shy vs scramble)", treatment=shy_frac_r,
              control=scramble_frac, min_separation=0.15)
    v.floor("(e) immediate acquisition stays perfect (replay_shy)", shy_acq, floor=0.9)

    go = (shy_frac_r > replay_frac + 0.10 and shyonly_frac <= noreplay_frac + 0.10
          and shy_frac_r > scramble_frac + 0.15 and shy_acq >= 0.9)
    decision = v.decide(go=go)
    return {
        "largest_N": big,
        "replay_only_plateau_frac": replay_frac, "replay_shy_frac": shy_frac_r,
        "noreplay_frac": noreplay_frac, "shy_only_frac": shyonly_frac, "scramble_frac": scramble_frac,
        "replay_shy_immediate_acq": shy_acq,
        "shy_rise_over_replay_plateau": float(shy_frac_r - replay_frac),
        "self_generation_margin_shy_minus_scramble": float(shy_frac_r - scramble_frac),
        **decision,
    }


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop SLEEP SYNAPTIC DOWNSCALING (SHY): add the companion "
                                             "process biology runs alongside replay -- renormalize the shared "
                                             "readout so replay stops plateauing at 0.55.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40, help="per-fact WAKE teaching epochs")
    ap.add_argument("--replay-epochs", type=int, default=24, help="offline SLEEP consolidation epochs over the store")
    ap.add_argument("--replay-per-fact", type=int, default=16, help="self-generated replay draws per stored engram")
    ap.add_argument("--replay-noise", type=float, default=0.10, help="brain-owned variability on the replayed engram")
    ap.add_argument("--shy-frac", type=float, default=1.0,
                    help="SHY target = shy_frac * reference-column-norm (downscale-only; <1 renormalizes harder)")
    ap.add_argument("--shy-mode", choices=["percol", "global"], default="percol",
                    help="percol = the SHY mechanism (per postsynaptic neuron); global = argmax-inert control")
    ap.add_argument("--shy-reference", choices=["median", "mean", "min"], default="median",
                    help="the homeostatic setpoint the columns renormalize toward")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--arms", nargs="+",
                    default=["noreplay", "replay", "replay_shy", "shy_only", "scramble"],
                    help="which arms to run (default = all 5)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact, a.replay_noise,
                 a.shy_frac, a.shy_mode, a.shy_reference, a.arms)
    verdict = _verdict(result) if set(("noreplay", "replay", "replay_shy", "shy_only", "scramble")).issubset(a.arms) \
        else {"status": "PARTIAL", "note": "not all 5 arms run; verdict needs the full arm set", "arms_run": a.arms}
    summary = {"probe": "teacher_loop_sleep_shy_downscaling", "seed": a.seed,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 100, flush=True)
    if "status" in verdict and "largest_N" in verdict:
        print(f"[sleep-SHY] seed {a.seed} @ N={verdict['largest_N']}: "
              f"NOREPLAY {verdict['noreplay_frac']:.2f} | REPLAY-only {verdict['replay_only_plateau_frac']:.2f} | "
              f"REPLAY+SHY {verdict['replay_shy_frac']:.2f} | SHY-only {verdict['shy_only_frac']:.2f} | "
              f"SCRAMBLE {verdict['scramble_frac']:.2f} (chance {result['chance']:.2f})", flush=True)
        print(f"[sleep-SHY] rise(SHY over replay plateau) {verdict['shy_rise_over_replay_plateau']:+.2f} | "
              f"self-gen margin {verdict['self_generation_margin_shy_minus_scramble']:+.2f} | "
              f"REPLAY+SHY immediate-acq {verdict['replay_shy_immediate_acq']:.3f} | VERDICT {verdict['status']}",
              flush=True)
    for arm in a.arms:
        rc = result["arms"][arm]["retention_curve"]
        line = " ".join(f"N={k}:{rc[k]['n_recalled']}/{k}({rc[k]['frac_recalled']:.2f})" for k in sorted(rc, key=int))
        print(f"    {arm:11s}: {line}", flush=True)
    print(f"[sleep-SHY] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if verdict.get("status") == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
