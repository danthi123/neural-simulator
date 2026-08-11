"""TEACHER-LOOP INTERLEAVED GENERATIVE REPLAY -- van de Ven "brain-inspired replay" (2026-08-11).

WHY THIS LEVER (the record's own open frontier). The continual-retention arc has RESOLVED N=20 (self_replay
0.45 -> 0.742 de-clamp -> 0.967 capacity) but the CAPACITY resolution SLIPS at N=100 (grown 0.97 -> 0.91 -> 0.73;
`7ee36d66b`), and the board flags **acquisition-at-scale as "UPSTREAM OF ALL"**: the shared leaky-readout can no
longer cleanly ACQUIRE the newest fact once it is crowded (immediate-acq 0.95 -> 0.82 @ N=100). Replay-SCHEDULING is
exhausted: fixed-k prioritized replay is NEGATIVE at scale (`777fcb0d`, N=50 0.627 vs full 0.893 -- a fixed budget
over individual facts loses coverage), CLS bounded two-store NEGATIVE (`0c7531785`), EWC weight-protect REFUTED
(`e50f5d45a`), sparse-gated readout NEGATIVE. The one BOUNDED replay SOURCE that reaches flat-store retention is the
non-forgetting generator (`generative_v2`, matches flat 0.958 @ N=20).

THE UN-TRIED MOVE. Both the base sleep-replay consolidation AND generative_v2 teach each new fact **ALONE** first
(`_teach_fact(net, X_new, ...)`) and only THEN replay the store POST-HOC (`_teach_fact(net, X_regen, ...)`). Teaching
the new fact alone lets its shared-readout row over-commit (it drives every input toward the new class); post-hoc
replay must then UNDO that damage, and at scale a bounded replay budget cannot fully undo it -> retention degrades.
van de Ven, Siegelmann & Tolias 2020 (Nat Commun, doi:10.1038/s41467-020-17866-2) INTERLEAVE the generator's
regenerations of old tasks WITH the new task DURING training, so the shared readout never over-commits in the first
place. That specific interleaving -- the canonical brain-inspired-replay mechanism -- has NOT been run here. It
attacks the "acquisition-at-scale" wall directly, where post-hoc replay leaves it intact.

THE CLEAN A/B (exactly matched: same generator, same data, same TOTAL gradient steps -- ONLY the ordering differs).
For each new fact i: draw fresh world percepts X_new (n_draws); keep the FIXED non-forgetting generator current
(`gen.learn_fact`); build the old-fact replay pool from the generator's OWN regenerations of classes 0..i-1
(teacher/world ABSENT for old facts). Then consolidate the FIXED slow cortex --
  * INTERLEAVED_GR (TREATMENT, van de Ven): one shuffled pass over the multiset [X_new x epochs] + [X_regen x
    replay_epochs] -> every batch mixes new + regenerated-old; the new fact is NEVER taught in isolation.
  * POSTHOC_GR (CONTROL = the established generative_v2 / sleep-replay ordering): teach X_new alone for `epochs`,
    THEN replay X_regen for `replay_epochs`. IDENTICAL data, IDENTICAL total gradient steps -- only all-new-then-
    all-old instead of mixed. So a difference is the INTERLEAVING, not compute/data/capacity/generator quality.
  * SELF_REPLAY (BASELINE = the replay-cap to beat): teach X_new alone, then self-replay the Hippocampus
    mean-prototype store (NOT the generator). This is the record's replay-cap baseline (~0.71 @ N=50 fixed reservoir).
  * SCRAMBLE (CONTENT LESION of interleaved): interleaved, but the old regenerations are LABEL-SHUFFLED -> the
    stored content is corrupted while the interleaved compute is identical -> forgetting must return.

TEETH / GO (largest N, at a scale where post-hoc degrades so interleaving CAN show; N=20 saturates and is not the test):
  (PRIMARY, the novel lever) INTERLEAVED_GR frac-recalled @ largest N > POSTHOC_GR by a margin, 6/6 seeds -> the
     interleaving is load-bearing (van de Ven brain-inspired replay).
  (TASK GO) INTERLEAVED_GR > SELF_REPLAY (the replay-cap baseline) by a margin -> beats the record's cap.
  (CONTENT) INTERLEAVED_GR > SCRAMBLE by a margin AND SCRAMBLE ~ SELF_REPLAY -> the win is the stored regenerated
     CONTENT, not the interleaved compute.
  (ACQUISITION, the upstream test) INTERLEAVED_GR immediate-acq-at-scale >= POSTHOC_GR -> interleaving keeps the new
     fact learnable when the readout is crowded (the "upstream of all" wall).
  If INTERLEAVED_GR ~ POSTHOC_GR -> HONEST NEGATIVE: on this substrate interleaving adds nothing over post-hoc
     generative replay (the residual is acquisition CAPACITY / storage compression, not replay ORDERING) -- reported
     with the numbers + the named next lever (a COMPRESSING generator for sub-linear storage).

ANTI-CHEATS (each a REAL test): generator holds 0 raw patterns (regenerates, `_stored_raw_patterns==0`); its
trained-param count is CONSTANT in N; the slow-cortex consolidation of OLD facts reads ONLY generator regenerations
(never env / the teacher / a true-engram ruler); de-clamped bdsp_wmax=1e9 (the -6/+6 clamp silences the reservoir,
bound-trap `8ca014ff2`); cfg.seed byte-identical substrate across two builds; git diff main -- sim/ empty (all
runner-side). SCRAMBLE isolates content from compute.

DISCIPLINE: reuse-by-import (GenerativeReplayNetV2 non-forgetting generator; _build_slow_cortex fixed de-clamped
cortex + _assert_byte_identical_substrate + _git_sim_diff_empty from the CLS two-store de-risk; Hippocampus +
_self_replay_consolidate from the sleep-replay de-risk; _teach_fact/_fact_acc/_corrective_batch/N_ACT from the
scaling de-risk; ReferentEnv from corrective-acquire). NO sim/ edit. SIM_BACKEND=numpy (tiny launch-bound net).

RUN (single-seed smoke, numpy; N=50 = the scale where post-hoc degrades so the lever can show):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_interleaved_generative_replay_derisk --seed 42 \
      --n-max 50 --milestones 25 50 --slow-hidden 120 --gen-hidden 96 --gen-k 96 \
      --out research/findings/raw/teacher_loop_interleaved_gr_s42.json
  6-SEED (self-sweep, one aggregate; GO needs interleaved > posthoc 6/6):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_interleaved_generative_replay_derisk --seeds 42 43 44 45 46 47 \
      --n-max 50 --milestones 25 50 --slow-hidden 120 --gen-hidden 96 --gen-k 96 \
      --out research/findings/raw/teacher_loop_interleaved_gr.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # tiny launch-bound net -> CPU faster
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
# reuse-by-import: the non-forgetting generator + the fixed de-clamped slow cortex + arm primitives + anti-cheats. NO sim/ edit.
from research.runners._teacher_loop_generative_replay_v2_derisk import GenerativeReplayNetV2  # noqa: E402
from research.runners._teacher_loop_generative_replay_derisk import _cos  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _build_slow_cortex, _assert_byte_identical_substrate, _git_sim_diff_empty,
)
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import (  # noqa: E402
    Hippocampus, _self_replay_consolidate,
)
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_interleaved_gr.json"


def _regen_pool(gen, past_classes, per_fact, brain_rng, replay_noise, scramble=False):
    """The OLD-fact replay pool = the generator's OWN neural regenerations of classes 0..i-1 (teacher/world absent),
    each drawn `per_fact` times with brain-owned variability. SCRAMBLE shuffles the labels (content lesion, same
    compute). Returns (Xr, yr)."""
    past = list(past_classes)
    if not past:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    labels = list(past)
    if scramble:
        labels = list(brain_rng.permutation(labels))
    Xr, yr = [], []
    for j, lab in zip(past, labels):
        eg = gen.regenerate(j)                                    # NEURAL regeneration (query -> spiking -> readout)
        for _ in range(per_fact):
            Xr.append(eg + replay_noise * brain_rng.standard_normal(eg.shape[0]))
            yr.append(int(lab))
    return np.asarray(Xr, dtype=np.float64), np.asarray(yr, dtype=np.int64)


def _run_arm(arm, seed, referents, env, K, n_in, slow_hidden, gen_hidden, gen_k, settle, epochs, batch, eprop_lr,
             w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance, bdsp_wmax,
             gen_settle, gen_epochs, gen_lr, gen_tol, gen_max_epochs, gen_check_every, gen_new_mult):
    """arm in {interleaved_gr, posthoc_gr, self_replay, scramble}. Same fixed slow cortex, same seed/env, same
    per-fact data + TOTAL gradient steps. interleaved/posthoc/scramble drive consolidation from the FIXED
    non-forgetting generator; self_replay drives it from the Hippocampus mean-prototype store (the replay cap)."""
    net, slow_active0 = _build_slow_cortex(n_in, K, seed, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax,
                                           env, referents)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)
    gen_rng = np.random.default_rng(seed + 999)
    mix_rng = np.random.default_rng(seed + 606)

    use_gen = arm in ("interleaved_gr", "posthoc_gr", "scramble")
    interleave = arm in ("interleaved_gr", "scramble")
    scramble = arm == "scramble"

    gen = None
    hippo = None
    gen_param_trace = []
    if use_gen:
        gen = GenerativeReplayNetV2(int(gen_k), n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip,
                                    bdsp_wmax=bdsp_wmax, conv_tol=gen_tol, conv_max_epochs=gen_max_epochs,
                                    conv_check_every=gen_check_every, new_mult=gen_new_mult)
        gen.fit_query_norm()
    else:
        hippo = Hippocampus(seed, replay_noise=replay_noise)

    acquire_acc, slow_active_trace, acq_vs_N = [], [], {}
    retention = {}
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)             # WAKE: fresh world percepts of the NEW fact
        if use_gen:
            engram_i = np.asarray(X, dtype=np.float64).mean(axis=0)
            gen.learn_fact(i, engram_i, range(i), gen_epochs, batch, gen_rng)   # keep the FIXED generator current
            Xr, yr = _regen_pool(gen, range(i), replay_per_fact, brain_rng, replay_noise, scramble=scramble)
            if interleave:
                # van de Ven: ONE shuffled pass over [X_new x epochs] + [X_regen x replay_epochs] -> every batch
                # mixes new + regenerated-old. Matched total steps: epochs*n_draws + replay_epochs*|Xr|.
                parts_X = [np.repeat(X, epochs, axis=0)]
                parts_y = [np.repeat(np.asarray(y), epochs, axis=0)]
                if len(Xr):
                    parts_X.append(np.repeat(Xr, replay_epochs, axis=0))
                    parts_y.append(np.repeat(yr, replay_epochs, axis=0))
                Xmix = np.concatenate(parts_X, axis=0)
                ymix = np.concatenate(parts_y, axis=0)
                _teach_fact(net, Xmix, ymix, 1, batch, mix_rng)  # 1 pass; _teach_fact shuffles new+old together
            else:
                # post-hoc (the established ordering): all-new first, then all-regenerated-old. SAME data + steps.
                _teach_fact(net, X, y, epochs, batch, teach_rng)
                if len(Xr):
                    _teach_fact(net, Xr, yr, replay_epochs, batch, brain_rng)
            gen_param_trace.append((i + 1, gen.trained_param_count()))
        else:
            # self_replay baseline: teach the new fact alone, then self-replay the mean-prototype store.
            _teach_fact(net, X, y, epochs, batch, teach_rng)
            hippo.encode(X, i)
            _self_replay_consolidate(net, hippo, replay_epochs, batch, brain_rng, replay_per_fact, scramble=False)

        acq = _fact_acc(net, env, r, i, n=test_n)                # immediate acquisition of the JUST-taught fact
        acquire_acc.append(acq)
        slow_active_trace.append(int(net.n_active))
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "slow_reservoir_active": int(net.n_active),
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "immediate_acq_of_this_fact": float(acq),        # the "acquisition-at-scale" witness at each milestone
                "per_fact_acc": [float(a) for a in accs],
            }
            acq_vs_N[str(N)] = float(acq)
    out = {
        "arm": arm,
        "slow_reservoir_active_start": int(slow_active0),
        "slow_reservoir_active_constant": bool(len(set(slow_active_trace)) == 1),
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "immediate_acq_vs_N": acq_vs_N,
        "retention_curve": retention,
    }
    if use_gen:
        out["generator_trained_params"] = int(gen.trained_param_count())
        out["generator_stored_raw_patterns"] = int(gen._stored_raw_patterns)
        out["generator_param_constant_across_N"] = bool(len({p for _n, p in gen_param_trace}) <= 1)
        out["generator_param_trace"] = [[int(n), int(p)] for n, p in gen_param_trace]
        out["consolidation_used_ruler"] = bool(getattr(gen, "_used_ruler_in_consolidation", False))
    return out


def run(seed, n_max, milestones, slow_hidden, gen_hidden, gen_k, settle, epochs, batch, eprop_lr, w_clip, n_draws,
        d_p, noise, test_n, replay_epochs, replay_per_fact, replay_noise, gen_settle, gen_epochs, gen_lr, gen_tol,
        gen_max_epochs, gen_check_every, gen_new_mult, arms_to_run, bdsp_wmax):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    gen_k = int(gen_k) if gen_k and int(gen_k) > 0 else K
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    referents = [f"ref{i}" for i in range(n_max)]

    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, K, seed, slow_hidden, settle, eprop_lr,
                                                               w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        env = ReferentEnv(seed, d_p=d_p, noise=noise)            # fresh env per arm (identical referents + draw stream)
        for r in referents:
            env.proto(r)
        env.rng = np.random.default_rng(seed + 101)
        arms[arm] = _run_arm(arm, seed, referents, env, K, n_in, slow_hidden, gen_hidden, gen_k, settle, epochs,
                             batch, eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact,
                             replay_noise, chance, bdsp_wmax, gen_settle, gen_epochs, gen_lr, gen_tol, gen_max_epochs,
                             gen_check_every, gen_new_mult)
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {arm:15s}] {arms[arm]['wall_seconds']:.0f}s | immediate-acq "
              f"{arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}", flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "slow_hidden": int(slow_hidden), "gen_hidden": int(gen_hidden), "gen_k_query_width": int(gen_k),
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"slow_hidden": slow_hidden, "gen_hidden": gen_hidden, "gen_k": gen_k, "settle_steps": settle,
                       "epochs": epochs, "batch": batch, "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws,
                       "d_p": d_p, "noise": noise, "test_n": test_n, "replay_epochs": replay_epochs,
                       "replay_per_fact": replay_per_fact, "replay_noise": replay_noise, "gen_settle": gen_settle,
                       "gen_epochs": gen_epochs, "gen_lr": gen_lr, "gen_tol": gen_tol, "gen_max_epochs": gen_max_epochs,
                       "gen_check_every": gen_check_every, "gen_new_mult": gen_new_mult, "bdsp_wmax": bdsp_wmax,
                       "frozen_hidden": True},
            "arms": arms}


def _verdict(result):
    """Verdict + GO. PRIMARY (the novel lever): interleaved_gr frac-recalled @ largest N > posthoc_gr by a margin.
    TASK GO: interleaved_gr > self_replay (the replay-cap baseline). CONTENT: interleaved_gr > scramble AND
    scramble ~ self_replay. ACQUISITION: interleaved immediate-acq >= posthoc (the upstream test). Anti-cheats asserted.
    If interleaved ~ posthoc -> HONEST NEGATIVE (interleaving adds nothing over post-hoc generative replay)."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    arms = result["arms"]
    chance = result["chance"]

    def frac_at(arm, N):
        return arms.get(arm, {}).get("retention_curve", {}).get(str(N), {}).get("frac_recalled", float("nan"))

    def big_of(arm):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return max((int(k) for k in rc), default=None)

    big = big_of("interleaved_gr") or big_of("posthoc_gr") or big_of("self_replay")

    inter_f = frac_at("interleaved_gr", big)
    post_f = frac_at("posthoc_gr", big)
    self_f = frac_at("self_replay", big)
    scr_f = frac_at("scramble", big)
    inter_acq = arms.get("interleaved_gr", {}).get("mean_acquire_acc_immediate", float("nan"))
    post_acq = arms.get("posthoc_gr", {}).get("mean_acquire_acc_immediate", float("nan"))
    inter_acq_bigN = arms.get("interleaved_gr", {}).get("immediate_acq_vs_N", {}).get(str(big), float("nan"))
    post_acq_bigN = arms.get("posthoc_gr", {}).get("immediate_acq_vs_N", {}).get(str(big), float("nan"))

    garm = arms.get("interleaved_gr", {})
    param_constant = bool(garm.get("generator_param_constant_across_N", False))
    not_buffer = bool(garm.get("generator_stored_raw_patterns", 1) == 0)
    no_ruler = bool(not garm.get("consolidation_used_ruler", True))
    slow_constant = bool(garm.get("slow_reservoir_active_constant", False))

    if "interleaved_gr" not in arms or "posthoc_gr" not in arms:
        return {"largest_N": big, "interleaved_gr_frac_recalled": inter_f, "posthoc_gr_frac_recalled": post_f,
                "self_replay_frac_recalled": self_f, "scramble_frac_recalled": scr_f, "status": "PARTIAL"}

    attributable_to("interleaving old regenerations INTO acquisition (interleaved vs post-hoc)", inter_f, post_f)
    if not np.isnan(self_f):
        attributable_to("generative replay beats the self-replay cap (interleaved vs self_replay)", inter_f, self_f)
    if not np.isnan(scr_f):
        attributable_to("stored regenerated CONTENT (interleaved vs scramble)", inter_f, scr_f)

    v = Verdict("teacher-loop interleaved generative replay (van de Ven brain-inspired replay)", chance=chance)
    v.reaches("(PRIMARY) interleaving RAISES retention vs post-hoc generative replay", before=post_f, after=inter_f)
    v.require("(PRIMARY') interleaved_gr > posthoc_gr + 0.05", (inter_f > post_f + 0.05), expect=True,
              note=f"interleaved {inter_f:.2f} vs posthoc {post_f:.2f} @ N={big}")
    if not np.isnan(self_f):
        v.require("(TASK-GO) interleaved_gr > self_replay cap + 0.10", (inter_f > self_f + 0.10), expect=True,
                  note=f"interleaved {inter_f:.2f} vs self_replay {self_f:.2f} @ N={big}")
    if not np.isnan(scr_f):
        v.control("(CONTENT) stored regenerated content is the source (interleaved vs scramble)", treatment=inter_f,
                  control=scr_f, min_separation=0.10)
    if not (np.isnan(inter_acq_bigN) or np.isnan(post_acq_bigN)):
        v.require("(ACQUISITION) interleaved keeps the newest fact learnable at scale (>= post-hoc)",
                  inter_acq_bigN >= post_acq_bigN - 1e-9, expect=True,
                  note=f"immediate-acq@N={big}: interleaved {inter_acq_bigN:.2f} vs posthoc {post_acq_bigN:.2f}")
    v.floor("(acq) interleaved immediate acquisition stays high", inter_acq, floor=0.80)
    v.require("(FIXED) generator trained-param count CONSTANT across N", param_constant, expect=True,
              note=f"trace {garm.get('generator_param_trace')}")
    v.require("(FIXED') generator stores 0 raw patterns (genuinely generative)", not_buffer, expect=True)
    v.require("(ANTI-CHEAT) consolidation NEVER read the true-engram ruler", no_ruler, expect=True)
    v.require("(decoupled) slow reservoir CONSTANT across the curriculum (not a growing reservoir)", slow_constant,
              expect=True)
    v.require("(seed) substrate byte-identical across two builds at one seed",
              bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    go = (inter_f > post_f + 0.05 and inter_acq >= 0.80 and param_constant and not_buffer and no_ruler
          and slow_constant and bool(result["substrate_byte_identical"]) and bool(result["sim_diff_empty"]))
    if not np.isnan(self_f):
        go = go and (inter_f > self_f + 0.10)
    if not np.isnan(scr_f):
        go = go and (inter_f > scr_f + 0.10)
    if not (np.isnan(inter_acq_bigN) or np.isnan(post_acq_bigN)):
        go = go and (inter_acq_bigN >= post_acq_bigN - 1e-9)
    decision = v.decide(go=go)

    return {
        "largest_N": big,
        "interleaved_gr_frac_recalled": inter_f, "posthoc_gr_frac_recalled": post_f,
        "self_replay_frac_recalled": self_f, "scramble_frac_recalled": scr_f,
        "interleaved_immediate_acq": inter_acq, "posthoc_immediate_acq": post_acq,
        "interleaved_minus_posthoc": (float(inter_f - post_f) if not (np.isnan(inter_f) or np.isnan(post_f)) else None),
        "interleaved_minus_self_replay": (float(inter_f - self_f) if not (np.isnan(inter_f) or np.isnan(self_f)) else None),
        "interleaved_minus_scramble": (float(inter_f - scr_f) if not (np.isnan(inter_f) or np.isnan(scr_f)) else None),
        "immediate_acq_at_bigN": {"interleaved": inter_acq_bigN, "posthoc": post_acq_bigN},
        "retention_vs_N": {str(N): {"interleaved_gr": frac_at("interleaved_gr", N),
                                    "posthoc_gr": frac_at("posthoc_gr", N),
                                    "self_replay": frac_at("self_replay", N),
                                    "scramble": frac_at("scramble", N)} for N in result["milestones"]},
        "generator_trained_params": garm.get("generator_trained_params"),
        "generator_param_constant_across_N": param_constant,
        "generator_stored_raw_patterns": garm.get("generator_stored_raw_patterns"),
        "consolidation_used_ruler": garm.get("consolidation_used_ruler"),
        "slow_reservoir_constant": slow_constant,
        "substrate_byte_identical": result["substrate_byte_identical"], "sim_diff_empty": result["sim_diff_empty"],
        **decision,
    }


def _one_seed(a, seed, arms_to_run):
    result = run(seed, a.n_max, a.milestones, a.slow_hidden, a.gen_hidden, a.gen_k, a.settle_steps, a.epochs, a.batch,
                 a.eprop_lr, a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact,
                 a.replay_noise, a.gen_settle, a.gen_epochs, a.gen_lr, a.gen_tol, a.gen_max_epochs, a.gen_check_every,
                 a.gen_new_mult, arms_to_run, a.bdsp_wmax)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop INTERLEAVED generative replay (van de Ven brain-inspired "
                                             "replay): interleave the non-forgetting generator's regenerations of old "
                                             "facts INTO new-fact acquisition, vs the established post-hoc replay "
                                             "ordering, at matched data + gradient steps.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--n-max", type=int, default=50)
    ap.add_argument("--milestones", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--slow-hidden", type=int, default=120, help="the FIXED slow-cortex reservoir (never grown)")
    ap.add_argument("--gen-hidden", type=int, default=96, help="the FIXED generator reservoir (H_gen; constant in N)")
    ap.add_argument("--gen-k", type=int, default=96, help="FIXED class-query width (constant in N; 0 => =n_max)")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20, help="new-fact wake epochs (== post-hoc new-teach passes)")
    ap.add_argument("--replay-epochs", type=int, default=12, help="old-fact replay passes (matched across arms)")
    ap.add_argument("--replay-per-fact", type=int, default=8)
    ap.add_argument("--replay-noise", type=float, default=0.10)
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-epochs", type=int, default=16)
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--gen-tol", type=float, default=0.05, help="generator train-to-convergence tolerance")
    ap.add_argument("--gen-max-epochs", type=int, default=120)
    ap.add_argument("--gen-check-every", type=int, default=4)
    ap.add_argument("--gen-new-mult", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9)
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["interleaved_gr", "posthoc_gr", "self_replay", "scramble"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    arms_to_run = list(a.arms)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  (n_max={a.n_max}, slow_H={a.slow_hidden}, gen_H={a.gen_hidden}, "
              f"gen_k={a.gen_k})\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, arms_to_run)
        summary = {"probe": "teacher_loop_interleaved_generative_replay", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "arms_run": arms_to_run, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        print(f"[interleaved-gr] seed {s} @ N={rv.get('largest_N')}: "
              f"INTERLEAVED {rv.get('interleaved_gr_frac_recalled', float('nan')):.2f} | "
              f"POSTHOC {rv.get('posthoc_gr_frac_recalled', float('nan')):.2f} | "
              f"SELF_REPLAY {rv.get('self_replay_frac_recalled', float('nan')):.2f} | "
              f"SCRAMBLE {rv.get('scramble_frac_recalled', float('nan')):.2f} (chance {result['chance']:.2f})", flush=True)
        rvn = rv.get("retention_vs_N", {})
        for N in result["milestones"]:
            d = rvn.get(str(N), {})
            print(f"    N={N:3d}: interleaved {d.get('interleaved_gr', float('nan')):.2f} | "
                  f"posthoc {d.get('posthoc_gr', float('nan')):.2f} | self {d.get('self_replay', float('nan')):.2f} | "
                  f"scramble {d.get('scramble', float('nan')):.2f}", flush=True)
        print(f"[interleaved-gr] inter-post {rv.get('interleaved_minus_posthoc')} | inter-self "
              f"{rv.get('interleaved_minus_self_replay')} | inter-scr {rv.get('interleaved_minus_scramble')} | "
              f"immediate-acq@bigN {rv.get('immediate_acq_at_bigN')} | acq {rv.get('interleaved_immediate_acq', float('nan')):.3f}", flush=True)
        print(f"[interleaved-gr] gen-params {rv.get('generator_trained_params')} const-in-N "
              f"{rv.get('generator_param_constant_across_N')} stored-raw {rv.get('generator_stored_raw_patterns')} "
              f"used-ruler {rv.get('consolidation_used_ruler')} | byte-ident {rv.get('substrate_byte_identical')} | "
              f"sim-clean {rv.get('sim_diff_empty')} | VERDICT {rv.get('status')}", flush=True)
        print(f"[interleaved-gr] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        inter = [p["verdict"].get("interleaved_gr_frac_recalled", float("nan")) for p in per_seed]
        post = [p["verdict"].get("posthoc_gr_frac_recalled", float("nan")) for p in per_seed]
        selfr = [p["verdict"].get("self_replay_frac_recalled", float("nan")) for p in per_seed]
        scr = [p["verdict"].get("scramble_frac_recalled", float("nan")) for p in per_seed]
        agg = {"probe": "teacher_loop_interleaved_generative_replay_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "n_max": a.n_max, "go_count": go_n, "n_seeds": len(seeds),
               "interleaved_gr_frac_mean": float(np.nanmean(inter)), "posthoc_gr_frac_mean": float(np.nanmean(post)),
               "self_replay_frac_mean": float(np.nanmean(selfr)), "scramble_frac_mean": float(np.nanmean(scr)),
               "interleaved_minus_posthoc_mean": float(np.nanmean(np.array(inter) - np.array(post))),
               "interleaved_minus_self_replay_mean": float(np.nanmean(np.array(inter) - np.array(selfr))),
               "per_seed": per_seed}
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[interleaved-gr AGG] GO {go_n}/{len(seeds)} | interleaved {np.nanmean(inter):.2f} vs posthoc "
              f"{np.nanmean(post):.2f} vs self_replay {np.nanmean(selfr):.2f} vs scramble {np.nanmean(scr):.2f} | "
              f"inter-post {np.nanmean(np.array(inter) - np.array(post)):+.2f} | wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
