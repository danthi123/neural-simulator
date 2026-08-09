"""TEACHER-LOOP SLEEP-REPLAY CONSOLIDATION DE-RISK (2026-08-08): attack the BREADTH crux -- catastrophic
forgetting in the sequential teacher-loop -- with the brain's OWN hippocampal->cortical systems-consolidation
mechanism: after the teacher teaches fact N, run an OFFLINE consolidation phase in which the brain SELF-REPLAYS
its earlier stored facts from its OWN hippocampal engram store (teacher ABSENT, world/env ABSENT) and
re-consolidates them into the shared leaky-readout via the same e-prop rule, interleaved with the new fact.

THE MEASURED BASELINE this attacks (research/runners/_teacher_loop_scaling_derisk.py, finding fcdc2fd2):
sequential single-class corrective e-prop on ONE shared leaky-readout retains ~1 fact (frac_recalled ~ 1/N;
each fact learned perfectly at ~0.995 immediate, then OVERWRITTEN). The INTERLEAVED control (the teacher
re-presents old facts alongside the new one) retains 8/10 at N=10 on the SAME net -> capacity is adequate; the
failure is sequential INTERFERENCE on the shared readout. The mitigation must raise SEQUENTIAL retention toward
that 8/10 ceiling WITHOUT the teacher re-presenting old facts (that would just BE the interleaved crutch).

BRAIN-BASED SELF-GENERATION (the load-bearing distinction). The replayed patterns are SELF-GENERATED from the
brain's OWN store, NOT the teacher/world re-presenting the percept:
  * ENCODE (wake, teacher present): teaching fact i draws fresh noisy percepts from the world (env.draw -- the
    legitimate sensory environment) and moves the cortical readout by e-prop. At the SAME time the HIPPOCAMPUS
    captures a compressed engram of what the brain experienced = the MEAN of the percepts it saw (a lossy
    one-shot trace) tagged with the taught class. This is the brain's own memory, formed during the episode.
  * REPLAY (sleep, teacher + world ABSENT): the offline consolidation phase reactivates each stored engram and
    GENERATES a replay pattern from it -- engram + brain-owned internally-generated variability (a generative
    replay from the hippocampal prototype, a separate brain RNG, NEVER env's true prototype or env's noise
    process). These self-generated old-fact patterns are interleaved with the new fact and re-consolidated into
    the cortical readout by the SAME e-prop rule. `_self_replay_consolidate` takes ONLY the hippocampal store --
    it has NO `env` parameter and never calls env.draw (grep-verifiable: the teacher/world is absent).

This is Marr/McClelland hippocampal->cortical systems consolidation: the hippocampus replays recent experience
offline so the neocortex interleaves and consolidates it WITHOUT re-experiencing -- the standard biological
answer to catastrophic interference (McClelland/McNaughton/O'Reilly 1995; Wilson/McNaughton 1994 replay). The
replay pattern is the brain's, not the teacher's. Reuses the teacher-loop scaling machinery unchanged
(reuse-by-import) and follows the hippocampal-replay-of-stored-engrams pattern established by the
replay-cortical-consolidation gate line (v1..v3); that gate is a distinct spiking substrate, cited as the
biological pattern, not imported (its scientific verdict path is retired).

THREE ARMS, same net build / seed / per-fact teaching budget (so the ONLY difference is the consolidation phase):
  * NOREPLAY  = the scaling baseline: teach each fact, NO consolidation -> frac_recalled ~ 1/N (the wall).
  * REPLAY    = teach each fact, THEN offline self-replay-consolidate the whole hippocampus (teacher+env absent).
  * SCRAMBLE  = REPLAY's IDENTICAL extra consolidation compute, but the engram labels are SHUFFLED (the store's
                CONTENT is lesioned; the replay reactivates the wrong memory). Isolates the STORED CONTENT from
                the mere extra gradient steps -- if retention needs the store, SCRAMBLE forgets like NOREPLAY.

TEETH (single-seed SMOKE; 6-seed command below):
  (a) RETENTION RISES: REPLAY frac_recalled at N > NOREPLAY frac_recalled at N (the decisive comparison, same
      net/epochs), rising toward the interleaved 8/10 ceiling.
  (b) LOAD-BEARING: NOREPLAY (== lesion the consolidation phase) forgets -> the replay phase is what carries it.
  (c) IMMEDIATE ACQUISITION STAYS PERFECT: acquire_acc measured right after teaching each fact (BEFORE
      consolidation), in the REPLAY arm, stays ~1.0 -- consolidation must not break learning the new fact.
  (d) STORE IS THE SOURCE (self-generated + load-bearing): SCRAMBLE (replay the store's content lesioned, same
      compute) forgets like NOREPLAY -> the retention rise comes from the STORED ENGRAM CONTENT, not extra
      training and not the teacher. Lesion the store's content -> forgetting returns.
  grep-verify TEACHER/WORLD ABSENT during consolidation:
      grep -n 'def _self_replay_consolidate' research/runners/_teacher_loop_sleep_replay_consolidation_derisk.py
      -> the function signature has no `env`; `grep -n env` inside its body is empty.

DISCIPLINE: reuse-by-import (_mk_net / _feat / _fit_readout_norm_world / _teach_fact / _fact_acc /
_corrective_batch + ReferentEnv from the scaling + corrective-acquire de-risks). NO sim/ edit. cfg.seed via the
seed= the net passes to CoreSimConfig.seed (NOT actual_seed_used). SIM_BACKEND=cupy (3090) by default; numpy for
a plumbing smoke. tools.lab attribution + a Verdict preconditions block. Single-seed SMOKE as instructed.

RUN (3090, single-seed smoke as instructed):
  SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_sleep_replay_consolidation_derisk --seed 42 \
      --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --n-draws 32 \
      --out research/findings/raw/teacher_loop_sleep_replay_s42.json
  6-SEED (GO needs the retention rise 6/6 at 42..47), run one seed per process in parallel:
    for s in 42 43 44 45 46 47; do SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_sleep_replay_consolidation_derisk --seed $s \
      --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --n-draws 32 \
      --out research/findings/raw/teacher_loop_sleep_replay_s$s.json & done; wait
  PLUMBING SMOKE (fast, numpy): ... SIM_BACKEND=numpy ... --n-max 3 --milestones 1 3 --epochs 8 \
      --replay-epochs 6 --n-draws 12 --settle-steps 12 --test-n 20
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
# reuse-by-import: the teacher-loop SCALING machinery (net build, world features, readout-norm fit, per-fact
# teaching, held-out accuracy, corrective batch) + ReferentEnv. NO sim/ edit, NO re-derivation of the baseline.
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _mk_net, _feat, _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_sleep_replay.json"


# =============================== the HIPPOCAMPUS: the brain's OWN engram store ===============================
class Hippocampus:
    """A fast one-shot store of compressed engrams. During the wake episode (teacher present) it CAPTURES a
    lossy trace of what the brain experienced -- the MEAN of the percepts it saw for a fact, tagged with the
    taught class. During sleep it SELF-GENERATES a replay pattern from each stored engram (engram + brain-owned
    internally-generated variability). This is the brain's memory, NOT the teacher/world re-presenting: the store
    holds only the compressed prototype the brain encoded, and generates its own variability at replay time from
    a brain-owned RNG -- it never has env's true prototype or env's noise process, and replay never touches env."""

    def __init__(self, seed, replay_noise=0.10):
        self.engrams = []              # list of (engram_vector, class_label) -- the brain's own captured traces
        self.rng = np.random.default_rng(seed + 5150)   # a BRAIN-owned RNG (distinct from env's; self-generated)
        self.replay_noise = float(replay_noise)

    def encode(self, X_experienced, cls):
        """Wake capture: store the compressed engram = mean of the percepts the brain experienced for this fact.
        X_experienced are the SAME inputs the cortical readout was just trained on (the episode), so the engram
        is the brain's own trace of the episode -- lossy (a single prototype, not the raw draws)."""
        engram = np.asarray(X_experienced, dtype=np.float64).mean(axis=0)
        self.engrams.append((engram, int(cls)))

    def generate_replay(self, per_fact, scramble_labels=False):
        """Sleep self-generation: reactivate each stored engram and GENERATE per_fact replay draws from it using
        the brain's OWN variability (engram + brain_noise*N(0,1)) -- a generative replay from the hippocampal
        prototype. NO env, NO teacher: the pattern is reconstructed from the store, not re-presented by the world.
        scramble_labels (the lesion) shuffles which class each engram is replayed AS -> the store's CONTENT is
        corrupted while the compute is identical."""
        if not self.engrams:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
        labels = [c for _e, c in self.engrams]
        if scramble_labels:
            labels = list(self.rng.permutation(labels))
        Xs, ys = [], []
        for (engram, _c), lab in zip(self.engrams, labels):
            for _ in range(per_fact):
                draw = engram + self.replay_noise * self.rng.standard_normal(engram.shape[0])
                Xs.append(draw)
                ys.append(int(lab))
        return np.asarray(Xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def _self_replay_consolidate(net, hippocampus, epochs, batch, brain_rng, per_fact, scramble=False):
    """OFFLINE consolidation (SLEEP). Teacher ABSENT, world/env ABSENT -- this function has NO `env` parameter and
    never calls env.draw (grep-verify). The brain SELF-REPLAYS its stored engrams (generated internally by the
    hippocampus) interleaved (pooled + shuffled) and re-consolidates ALL of them into the shared leaky-readout via
    the SAME e-prop rule (`_teach_fact` == the port's train_batch loop). This interleaves old + new offline, the
    biological answer to catastrophic interference, without re-experiencing anything."""
    Xr, yr = hippocampus.generate_replay(per_fact, scramble_labels=scramble)
    if len(Xr) == 0:
        return
    _teach_fact(net, Xr, yr, epochs, batch, brain_rng)   # pooled + shuffled inside _teach_fact -> interleaved


# ================================= one arm of the sequential curriculum =================================
def _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws,
             milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance):
    """Teach the referents SEQUENTIALLY into ONE brain. arm in {noreplay, replay, scramble}. For replay/scramble,
    after teaching each fact run the offline self-replay consolidation over the hippocampus so far."""
    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    _fit_readout_norm_world(net, env, referents, seed)     # readout-norm fit ONCE over the world (as the baseline)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)          # brain-owned RNG for consolidation shuffling
    hippo = Hippocampus(seed, replay_noise=replay_noise)
    do_replay = arm in ("replay", "scramble")
    scramble = arm == "scramble"

    acquire_acc = []        # immediate held-out acc right after teaching each fact, BEFORE consolidation (teeth c)
    retention = {}          # milestone N -> {frac_recalled, n_recalled, mean_retained_acc, oldest, newest, per_fact}
    for i, r in enumerate(referents):
        # --- WAKE: teacher teaches fact i from the world (env draws are legitimate: the sensory environment) ---
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acq = _fact_acc(net, env, r, i, n=test_n)          # IMMEDIATE acquisition (teeth c) -- before any replay
        acquire_acc.append(acq)
        hippo.encode(X, i)                                  # the hippocampus captures the engram of this episode
        # --- SLEEP: offline self-replay consolidation (teacher + world ABSENT); noreplay arm SKIPS this ---
        if do_replay:
            _self_replay_consolidate(net, hippo, replay_epochs, batch, brain_rng, replay_per_fact, scramble=scramble)
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
    }


def run(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
        test_n, replay_epochs, replay_per_fact, replay_noise):
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
    for arm in ("noreplay", "replay", "scramble"):
        t0 = time.time()
        # fresh env draw-stream per arm (reset the env RNG) so each arm sees the SAME teaching percepts.
        env.rng = np.random.default_rng(seed + 101)
        arms[arm] = _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr,
                             w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance)
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
                       "replay_noise": replay_noise},
            "arms": arms}


def _verdict(result):
    """Emit a Verdict preconditions block + the GO decision. TEETH:
      (a) retention rises: REPLAY frac_recalled > NOREPLAY at the largest N.
      (b) load-bearing: NOREPLAY (== consolidation lesioned) forgets (frac < 0.5).
      (c) immediate acquisition stays perfect in REPLAY (mean immediate acq >= 0.9).
      (d) store is the source: SCRAMBLE (content-lesioned, same compute) does NOT match REPLAY's rise --
          scramble_frac <= noreplay_frac + margin (forgetting returns) AND replay_frac > scramble_frac + 0.15."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    rc = {a: result["arms"][a]["retention_curve"] for a in result["arms"]}
    big = max((int(k) for k in rc["replay"]), default=None)
    key = str(big)
    noreplay_frac = rc["noreplay"][key]["frac_recalled"]
    replay_frac = rc["replay"][key]["frac_recalled"]
    scramble_frac = rc["scramble"][key]["frac_recalled"]
    replay_acq = result["arms"]["replay"]["mean_acquire_acc_immediate"]
    chance = result["chance"]

    # the effect is the STORED ENGRAM CONTENT (replay vs scramble), not extra compute.
    attributable_to("self-replay content (replay vs scramble, same compute)", replay_frac, scramble_frac)
    attributable_to("consolidation phase (replay vs no-replay)", replay_frac, noreplay_frac)

    v = Verdict("teacher-loop sleep-replay consolidation", chance=chance)
    v.reaches("(a) retention RISES vs no-replay sequential", before=noreplay_frac, after=replay_frac)
    v.require("(b) no-replay forgets (consolidation load-bearing)", noreplay_frac < 0.5, expect=True,
              note=f"noreplay frac_recalled@N={big} = {noreplay_frac:.2f}")
    v.floor("(c) immediate acquisition stays perfect (REPLAY)", replay_acq, floor=0.9)
    v.control("(d) store CONTENT is the source (replay vs scramble)", treatment=replay_frac,
              control=scramble_frac, min_separation=0.15)
    v.require("(d') scramble forgets like no-replay (content-lesioned)", scramble_frac <= noreplay_frac + 0.10,
              expect=True, note=f"scramble {scramble_frac:.2f} vs noreplay {noreplay_frac:.2f}")
    go = (replay_frac > noreplay_frac and noreplay_frac < 0.5 and replay_acq >= 0.9
          and replay_frac > scramble_frac + 0.15 and scramble_frac <= noreplay_frac + 0.10)
    decision = v.decide(go=go)
    return {
        "largest_N": big,
        "noreplay_frac_recalled": noreplay_frac, "replay_frac_recalled": replay_frac,
        "scramble_frac_recalled": scramble_frac, "replay_immediate_acq": replay_acq,
        "retention_rise_replay_minus_noreplay": float(replay_frac - noreplay_frac),
        "self_generation_margin_replay_minus_scramble": float(replay_frac - scramble_frac),
        **decision,
    }


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop SLEEP-REPLAY consolidation: self-replay the brain's "
                                             "own hippocampal engrams offline (teacher absent) to beat catastrophic "
                                             "forgetting in the sequential teacher-loop.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40, help="per-fact WAKE teaching epochs")
    ap.add_argument("--replay-epochs", type=int, default=24, help="offline SLEEP consolidation epochs over the store")
    ap.add_argument("--replay-per-fact", type=int, default=16, help="self-generated replay draws per stored engram")
    ap.add_argument("--replay-noise", type=float, default=0.10, help="brain-owned variability on the replayed engram")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact, a.replay_noise)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_sleep_replay_consolidation", "seed": a.seed,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 100, flush=True)
    print(f"[sleep-replay] seed {a.seed} @ N={verdict['largest_N']}: "
          f"NOREPLAY frac-recalled {verdict['noreplay_frac_recalled']:.2f} | "
          f"REPLAY {verdict['replay_frac_recalled']:.2f} | SCRAMBLE {verdict['scramble_frac_recalled']:.2f} "
          f"(chance {result['chance']:.2f})", flush=True)
    print(f"[sleep-replay] rise(replay-noreplay) {verdict['retention_rise_replay_minus_noreplay']:+.2f} | "
          f"self-gen margin(replay-scramble) {verdict['self_generation_margin_replay_minus_scramble']:+.2f} | "
          f"REPLAY immediate-acq {verdict['replay_immediate_acq']:.3f} | VERDICT {verdict['status']}", flush=True)
    for arm in ("noreplay", "replay", "scramble"):
        rc = result["arms"][arm]["retention_curve"]
        line = " ".join(f"N={k}:{rc[k]['n_recalled']}/{k}({rc[k]['frac_recalled']:.2f})" for k in sorted(rc, key=int))
        print(f"    {arm:9s}: {line}", flush=True)
    print(f"[sleep-replay] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
