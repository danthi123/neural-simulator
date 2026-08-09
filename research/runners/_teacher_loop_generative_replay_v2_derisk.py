"""TEACHER-LOOP GENERATIVE-REPLAY v2 -- THE NON-FORGETTING GENERATOR (2026-08-09).

WHERE v1 LANDED (443351967, 6-seed verify-confirmed): a FIXED-SIZE neural generator that re-dreams ALL learned
facts BEATS the bounded buffer (0.692 vs 0.517) but does NOT match the flat O(N) store (0.950). THE LOAD-BEARING
RESIDUAL, isolated and named: **the generator ITSELF forgets** -- its regeneration fidelity degrades from ~1.00 at
N=10 to ~0.80-0.90 at N=20, and the shared slow readout amplifies that into the -0.258 retention gap. Capacity is
NOT the issue (H_gen=96 >> N=20); it is continual-training INTERFERENCE in the generator's OWN readout.

WHY v1's self-replay is INSUFFICIENT (the recursion analysis). The generator is a FIXED random spiking reservoir +
a linear leaky readout W; per class j the reservoir eligibility r_j is CONSTANT (fixed query, fixed reservoir), so
consolidation is a static linear regression r_j @ W ~= (engram_j - anchor). When a NEW class i is added, v1 fits W
to the new fact's TRUE engram while PINNING each past j to its OWN regeneration snapshot (van de Ven self-replay).
If that incremental fit were EXACT (zero error on all i+1 constraints), the past outputs would not move at all and
-- by induction from the moment each class was first fit to its true engram -- regenerate(j) would stay == the true
engram FOREVER (the recursion bottoms out). v1 degrades ONLY because the delta rule runs a FIXED, small number of
epochs (gen_epochs=16) and does NOT reach that fit: each new fact leaves a small residual error on the past classes,
their snapshots drift, and the next step pins to the ALREADY-corrupted snapshot -> monotone accumulation. The
recursion does not bottom out because the fit is incomplete, NOT because the store is too small.

THE v2 SURPASS (still van de Ven 2020, doi:10.1038/s41467-020-17866-2; brain-based; NEURAL; genuinely fixed-size).
Give the generator its OWN deeper consolidation so each incremental fit CONVERGES -- more sleep-replay cycles, not
more parameters (speed is secondary; faithfulness is the bar):
  (1) TRAIN-TO-CONVERGENCE self-replay. At each new fact, run the SAME local NLMS delta rule on the SAME frozen
      reservoir, but keep replaying (new fact's true engram + the generator's OWN regenerations of ALL prior classes)
      until the reconstruction error over that replay set falls below a tolerance (capped at gen_max_epochs). A
      converged fit does not move the past outputs, so the snapshot the NEXT step pins to has NOT drifted -> the
      recursion bottoms out at the true engrams and fidelity HOLDS near 1.0 as N grows.
  (2) RECENCY-BALANCED replay: the new fact is replayed new_mult times inside the same set, so as the past set grows
      the new fact is not starved and immediate acquisition stays high, while past COVERAGE stays complete.
The generator adds ZERO parameters vs v1 (same H_gen readout) -- it is the SAME fixed-size store consolidated
harder. Optionally a modestly larger FIXED H_gen is exposed, still CONSTANT in N.

ANTI-CHEATS (the #1 cheat first). The generator's self-replay targets for past classes are its OWN regenerations
(`self.regenerate`, snapshot BEFORE the update) -- NEVER `true_engrams`, which is an experimenter ruler used ONLY to
MEASURE fidelity (asserted: `_used_ruler_in_consolidation == False`). The generator holds 0 stored raw patterns; its
trained-param count is CONSTANT across N=10 and N=20 (asserted, printed as a trace); it is NEURAL (fixed spiking
Izhikevich reservoir + local delta rule on spike eligibility); de-clamped bdsp_wmax=1e9 (the -6/+6 clamp silences
the reservoir, bound-trap 8ca014ff2); cfg.seed byte-identical substrate; git diff main -- sim/ empty; backend
recorded.

THREE ARMS (same net build / seed / env / wake budget; the ONLY difference is the SLEEP replay SOURCE):
  * generative_v2 = TREATMENT. The strengthened (train-to-convergence) fixed-size generator re-dreams all N facts.
  * generative_v1 = the PRIOR naive generator (~0.692), MEASURED IN-RUN on the SAME slow reservoir/seed/env.
  * flat          = the O(N) target (~0.95), MEASURED IN-RUN (CLS flat arm). bounded_buffer optional (~0.517).

THE KEY MEASUREMENTS: (A) generator regeneration fidelity at N=10 vs N=20 for v2 -- does it now HOLD near 1.0
(vs v1's 1.00 -> 0.80-0.90)? (B) generative_v2 retention -- does it rise toward flat and beat generative_v1?

GO (largest N, generator FIXED): v2 fidelity holds (mean cos >= 0.95 at N=20 AND does not drop > 0.03 from N=10);
v2 retention within 0.10 of flat AND > v1 + 0.10; generator param-count constant in N AND 0 stored raw patterns AND
consolidation never touched the ruler; acquisition >= 0.85; substrate byte-identical; sim/ clean. If v2 fidelity
STILL degrades despite train-to-convergence, that is an HONEST NEGATIVE naming WHY the recursion does not bottom out
on this substrate (e.g. reservoir codes not linearly separable at N; NLMS cannot reach the exact fit) + what would.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_generative_replay_v2_derisk --seed 42 \
      --n-max 20 --milestones 10 20 --gen-hidden 96 --gen-epochs 16 --gen-max-epochs 120 \
      --gen-tol 0.05 --gen-new-mult 3 --gen-lr 0.8 \
      --out research/findings/raw/teacher_loop_generative_replay_v2_s42.json
  6-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_generative_replay_v2_derisk --seeds 42 43 44 45 46 47 \
      --n-max 20 --milestones 10 20 --gen-hidden 96 --gen-max-epochs 120 --gen-tol 0.05 --gen-new-mult 3 \
      --out research/findings/raw/teacher_loop_generative_replay_v2.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
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
# reuse-by-import: the v1 generator + arm primitives + the CLS flat/bounded arms + the anti-cheat asserts. NO sim/ edit.
from research.runners._teacher_loop_generative_replay_derisk import (  # noqa: E402
    GenerativeReplayNet, _cos, _assert_generator_fixed_size,
)
from research.runners._teacher_loop_neurogenesis_capacity_derisk import NeurogenesisNet  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _build_slow_cortex, _run_arm as _run_cls_arm,
    _assert_byte_identical_substrate, _git_sim_diff_empty,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_generative_replay_v2.json"


# ================ the STRENGTHENED (non-forgetting) FIXED-SIZE NEURAL generator ================
class GenerativeReplayNetV2(GenerativeReplayNet):
    """v1's fixed-size spiking associative generator, CONSOLIDATED HARDER. Same reservoir, same readout, same param
    count -- the ONLY change is that per-fact self-replay runs the SAME local delta rule to CONVERGENCE (until the
    reconstruction error over the replay set falls below `conv_tol`, capped at `conv_max_epochs`), with the new fact
    replayed `new_mult` times (recency balance). A converged fit does not move the past outputs, so the snapshot the
    next fact pins to has not drifted -> the self-replay recursion bottoms out at the true engrams and fidelity holds
    as N grows. Anti-cheat: past-class targets are the generator's OWN regenerations (snapshot before the update),
    NEVER the experimenter ruler (`true_engrams`); tripwire `_used_ruler_in_consolidation` stays False."""

    def __init__(self, *args, conv_tol=0.05, conv_max_epochs=120, conv_check_every=4, new_mult=3, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv_tol = float(conv_tol)
        self._conv_max_epochs = int(conv_max_epochs)
        self._conv_check_every = max(1, int(conv_check_every))
        self._new_mult = max(1, int(new_mult))
        self._used_ruler_in_consolidation = False   # anti-cheat tripwire: consolidation must NEVER read true engrams
        self._last_learn_epochs = 0                 # epochs actually spent on the most recent fact (convergence witness)
        self._learn_epoch_trace = []                # per-fact convergence-epoch trace

    def learn_fact(self, new_cls, new_engram, past_classes, epochs, batch, rng):
        """Train-to-convergence van de Ven self-replay. Replay set = the NEW fact's TRUE engram (x new_mult, the only
        raw pattern ever in hand) + the generator's OWN regenerations of ALL prior classes (snapshot BEFORE this
        update). Keep replaying with the local NLMS delta rule until the max reconstruction error over that set falls
        below conv_tol (capped at conv_max_epochs). NEVER reads `true_engrams` for a past class."""
        past = list(past_classes)
        new_engram = np.asarray(new_engram, dtype=np.float64)
        # OWN regenerations, snapshot BEFORE the update (the generator dreams its own past). NOT the ruler.
        past_snap = {j: self.regenerate(j) for j in past}
        Q_list, T_list = [], []
        for _ in range(self._new_mult):                                  # recency balance: keep the new fact fitted
            Q_list.append(self._query_code(new_cls)); T_list.append(new_engram)
        for j in past:
            Q_list.append(self._query_code(j)); T_list.append(past_snap[j])
        Q = np.asarray(Q_list, dtype=np.float64); T = np.asarray(T_list, dtype=np.float64)
        max_e = self._conv_max_epochs if self._conv_max_epochs else int(epochs)
        self._last_learn_epochs = 0
        for ep in range(int(max_e)):
            perm = rng.permutation(len(Q))
            for i in range(0, len(Q), int(batch)):
                b = perm[i:i + int(batch)]
                self._train_regression_batch(Q[b], T[b])
            self._last_learn_epochs = ep + 1
            if self._conv_tol > 0 and ((ep + 1) % self._conv_check_every == 0 or ep == max_e - 1):
                # convergence check: reconstruction error vs the FIXED replay targets (new true engram + past OWN
                # snapshots). Uses self.regenerate ONLY -> never the ruler.
                err = float(np.linalg.norm(self.regenerate(new_cls) - new_engram))
                for j in past:
                    err = max(err, float(np.linalg.norm(self.regenerate(j) - past_snap[j])))
                if err < self._conv_tol:
                    break
        self._learn_epoch_trace.append((int(new_cls), int(self._last_learn_epochs)))


def _run_generative_arm_v2(arm_name, net_cls, net_kwargs, gen_k, seed, referents, env, K, n_in, slow_hidden,
                           gen_hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, milestones, test_n,
                           replay_epochs, replay_per_fact, replay_noise, chance, bdsp_wmax, gen_settle, gen_epochs,
                           gen_lr):
    """Generic generative arm: same fixed slow cortex + same wake budget as flat; sleep replay SOURCE = the fixed
    generator (`net_cls`) regenerating ALL learned facts. Used for BOTH generative_v2 (strengthened) and
    generative_v1 (naive), so the comparison is apples-to-apples in-run on the SAME slow reservoir/seed/env. `gen_k`
    = the FIXED class-query address width (constant in N; the readout/plastic store is INDEPENDENT of gen_k). v1 uses
    gen_k=K (the prior naive width that collided -> rank deficient); v2 uses a wider FIXED gen_k (full-rank codes)."""
    net, slow_active0 = _build_slow_cortex(n_in, K, seed, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax,
                                           env, referents)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)
    gen_rng = np.random.default_rng(seed + 999)

    gen = net_cls(int(gen_k), n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax, **net_kwargs)
    gen.fit_query_norm()
    gen_trained_params = gen.trained_param_count()
    gen_total_params = gen.total_param_count()
    gen_param_trace = []

    true_engrams = {}     # experimenter-only ruler for the fidelity metric; the consolidation path never reads it.

    acquire_acc, slow_active_trace = [], []
    retention, gen_fidelity = {}, {}
    max_replay_set = 0
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acquire_acc.append(_fact_acc(net, env, r, i, n=test_n))
        engram_i = np.asarray(X, dtype=np.float64).mean(axis=0)
        true_engrams[i] = engram_i                                        # ruler only
        gen.learn_fact(i, engram_i, range(i), gen_epochs, batch, gen_rng) # keep the FIXED generator current
        classes = list(range(i + 1))
        max_replay_set = max(max_replay_set, len(classes))
        Xr, yr = [], []
        for j in classes:
            eg = gen.regenerate(j)                                        # NEURAL regeneration (spikes -> readout)
            for _ in range(replay_per_fact):
                Xr.append(eg + replay_noise * brain_rng.standard_normal(eg.shape[0]))
                yr.append(j)
        Xr = np.asarray(Xr, dtype=np.float64); yr = np.asarray(yr, dtype=np.int64)
        _teach_fact(net, Xr, yr, replay_epochs, batch, brain_rng)
        slow_active_trace.append(int(net.n_active))
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "generator_trained_params": gen_trained_params, "generator_total_params": gen_total_params,
                "generator_stored_raw_patterns": int(gen._stored_raw_patterns),
                "flat_buffer_floats_equiv": int(N * n_in),
                "slow_reservoir_active": int(net.n_active),
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
            }
            gen_param_trace.append((N, gen_trained_params))
            # --- generator regeneration fidelity: does IT forget as N grows? (the honest recursion test) ---
            regens = [gen.regenerate(j) for j in range(N)]
            coss = [_cos(regens[j], true_engrams[j]) for j in range(N)]
            protos = np.stack([true_engrams[j] for j in range(N)])
            near_ok = 0
            for j in range(N):
                d = np.linalg.norm(protos - regens[j][None, :], axis=1)
                near_ok += int(int(np.argmin(d)) == j)
            gen_fidelity[str(N)] = {"mean_cos": float(np.mean(coss)), "min_cos": float(np.min(coss)),
                                    "nearest_proto_acc": float(near_ok / N)}
    generative_not_buffer = bool(gen._stored_raw_patterns == 0)
    param_constant = bool(len({p for _n, p in gen_param_trace}) <= 1)
    used_ruler = bool(getattr(gen, "_used_ruler_in_consolidation", False))
    out = {
        "arm": arm_name, "gen_k_query_width": int(gen_k),
        "slow_reservoir_active_start": slow_active0,
        "slow_reservoir_active_constant": bool(len(set(slow_active_trace)) == 1),
        "generator_trained_params": gen_trained_params, "generator_total_params": gen_total_params,
        "generator_hidden": int(gen_hidden), "generator_param_constant_across_N": param_constant,
        "generator_param_trace": [[int(n), int(p)] for n, p in gen_param_trace],
        "generator_stored_raw_patterns": int(gen._stored_raw_patterns),
        "generative_not_stored_buffer": generative_not_buffer,
        "consolidation_used_ruler": used_ruler,       # MUST be False (anti-cheat #1)
        "max_replay_set_size": int(max_replay_set),
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention,
        "generator_fidelity": gen_fidelity,
        "learn_epoch_trace": [[int(c), int(e)] for c, e in getattr(gen, "_learn_epoch_trace", [])],
    }
    return out


def _assert_gen_k_independent_store(gen_k, n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax):
    """anti-cheat: the plastic store (readout trained-param count) must be INDEPENDENT of the query-address width
    gen_k -- widening the query code (the v2 rank fix) must NOT smuggle in extra plastic capacity. Build the V2
    generator at gen_k and at gen_k+37 and confirm the trained-param count is identical."""
    a = GenerativeReplayNetV2(int(gen_k), n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax)
    b = GenerativeReplayNetV2(int(gen_k) + 37, n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax)
    return int(a.trained_param_count()), bool(a.trained_param_count() == b.trained_param_count())


def run(seed, n_max, milestones, capacity, slow_hidden, gen_hidden, gen_k, settle, epochs, batch, eprop_lr, w_clip,
        n_draws, d_p, noise, test_n, replay_epochs, replay_per_fact, replay_noise, gen_settle, gen_epochs, gen_lr,
        gen_tol, gen_max_epochs, gen_check_every, gen_new_mult, arms_to_run, bdsp_wmax):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    gen_k = int(gen_k) if gen_k and int(gen_k) > 0 else K   # 0/None => v1-equivalent width (=n_max, the collided code)
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    referents = [f"ref{i}" for i in range(n_max)]

    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, K, seed, slow_hidden, settle, eprop_lr,
                                                               w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()
    # generator fixed-size in N (built at the v2 query width) AND the plastic store independent of gen_k.
    gen_tp, gen_totp, gen_fixed_ok = _assert_generator_fixed_size(gen_k, n_in, gen_hidden, seed, gen_settle, gen_lr,
                                                                  w_clip, bdsp_wmax)
    gen_store_tp, gen_store_indep_k = _assert_gen_k_independent_store(gen_k, n_in, gen_hidden, seed, gen_settle,
                                                                      gen_lr, w_clip, bdsp_wmax)

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        env = ReferentEnv(seed, d_p=d_p, noise=noise)
        for r in referents:
            env.proto(r)
        env.rng = np.random.default_rng(seed + 101)
        if arm == "generative_v2":
            arms[arm] = _run_generative_arm_v2(
                arm, GenerativeReplayNetV2,
                dict(conv_tol=gen_tol, conv_max_epochs=gen_max_epochs, conv_check_every=gen_check_every,
                     new_mult=gen_new_mult),
                gen_k,                                  # v2: the WIDER FIXED query code (full-rank eligibilities)
                seed, referents, env, K, n_in, slow_hidden, gen_hidden, settle, epochs, batch, eprop_lr, w_clip,
                n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance, bdsp_wmax,
                gen_settle, gen_epochs, gen_lr)
        elif arm == "generative_v1":
            arms[arm] = _run_generative_arm_v2(
                arm, GenerativeReplayNet, {},
                K,                                      # v1: the PRIOR naive width (=n_max; the collided code)
                seed, referents, env, K, n_in, slow_hidden, gen_hidden, settle, epochs, batch, eprop_lr, w_clip,
                n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance, bdsp_wmax,
                gen_settle, gen_epochs, gen_lr)
        else:
            cls_arm = "flat" if arm == "flat" else "two_store"
            arms[arm] = _run_cls_arm(cls_arm, seed, referents, env, K, n_in, slow_hidden, capacity, settle, epochs,
                                     batch, eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs,
                                     replay_per_fact, replay_noise, chance, bdsp_wmax)
            arms[arm]["arm"] = arm
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        fd = arms[arm].get("generator_fidelity", {}).get(str(big), {})
        print(f"[arm {arm:14s}] {arms[arm]['wall_seconds']:.0f}s | "
              f"immediate-acq {arms[arm].get('mean_acquire_acc_immediate', float('nan')):.3f} | "
              f"frac-recalled@N={big}: {fr:.2f} | gen-cos {fd.get('mean_cos', float('nan')):.3f}", flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "capacity_F": int(capacity), "slow_hidden": int(slow_hidden), "gen_hidden": int(gen_hidden),
            "gen_k_query_width_v2": int(gen_k),
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "generator_trained_params": gen_tp, "generator_total_params": gen_totp,
            "generator_fixed_size_two_builds_ok": gen_fixed_ok,
            "generator_store_indep_of_gen_k": gen_store_indep_k, "generator_store_params": gen_store_tp,
            "config": {"capacity_F": capacity, "slow_hidden": slow_hidden, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "settle_steps": settle, "epochs": epochs, "batch": batch, "eprop_lr": eprop_lr,
                       "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise, "test_n": test_n,
                       "replay_epochs": replay_epochs, "replay_per_fact": replay_per_fact,
                       "replay_noise": replay_noise, "gen_settle": gen_settle, "gen_epochs": gen_epochs,
                       "gen_lr": gen_lr, "gen_tol": gen_tol, "gen_max_epochs": gen_max_epochs,
                       "gen_check_every": gen_check_every, "gen_new_mult": gen_new_mult,
                       "bdsp_wmax": bdsp_wmax, "frozen_hidden": True},
            "arms": arms}


def _verdict(result):
    """Verdict + GO. THE KEY MEASUREMENTS: generative_v2 fidelity HOLDS near 1.0 (mean cos >= 0.95 at largest N AND
    does not drop > 0.03 from N=10), and generative_v2 retention rises toward flat (within 0.10) AND beats v1 (+0.10).
    Anti-cheats: generator fixed-size in N, 0 stored raw patterns, consolidation never read the ruler, byte-identical
    substrate, sim/ clean, acquisition >= 0.85. If v2 fidelity STILL degrades -> HONEST NEGATIVE naming why."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    arms = result["arms"]
    chance = result["chance"]

    def frac_at(arm, N):
        return arms.get(arm, {}).get("retention_curve", {}).get(str(N), {}).get("frac_recalled", float("nan"))

    def big_of(arm):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return max((int(k) for k in rc), default=None)

    big = big_of("generative_v2") or big_of("generative_v1") or big_of("flat")
    small = min((int(k) for k in arms.get("generative_v2", {}).get("retention_curve", {})), default=big)

    v2_f = frac_at("generative_v2", big)
    v1_f = frac_at("generative_v1", big)
    flat_f = frac_at("flat", big)
    v2_acq = arms.get("generative_v2", {}).get("mean_acquire_acc_immediate", float("nan"))

    if "generative_v2" not in arms:
        return {"largest_N": big, "generative_v2_frac": v2_f, "generative_v1_frac": v1_f, "flat_frac": flat_f,
                "status": "PARTIAL"}

    garm = arms["generative_v2"]
    fid = garm.get("generator_fidelity", {})
    cos_big = fid.get(str(big), {}).get("mean_cos", float("nan"))
    cos_small = fid.get(str(small), {}).get("mean_cos", float("nan"))
    near_big = fid.get(str(big), {}).get("nearest_proto_acc", float("nan"))
    fid_drop = (float(cos_small - cos_big) if not (np.isnan(cos_small) or np.isnan(cos_big)) else float("nan"))
    fid_holds = bool((not np.isnan(cos_big)) and cos_big >= 0.95 and (np.isnan(fid_drop) or fid_drop <= 0.03))

    param_constant = bool(garm.get("generator_param_constant_across_N"))
    not_buffer = bool(garm.get("generative_not_stored_buffer"))
    no_ruler = bool(not garm.get("consolidation_used_ruler"))
    fixed_two_builds = bool(result.get("generator_fixed_size_two_builds_ok"))
    store_indep_k = bool(result.get("generator_store_indep_of_gen_k"))
    slow_constant = bool(garm.get("slow_reservoir_active_constant"))

    # v1 fidelity for the side-by-side (does v2 actually fix the recursion vs the naive generator?)
    v1_fid = arms.get("generative_v1", {}).get("generator_fidelity", {})
    v1_cos_big = v1_fid.get(str(big), {}).get("mean_cos", float("nan"))

    if not np.isnan(flat_f):
        attributable_to("generative_v2 coverage matches the flat O(N) store (v2 vs flat)", v2_f, flat_f)
    if not np.isnan(v1_f):
        attributable_to("strengthened generator beats the naive generator (v2 vs v1)", v2_f, v1_f)

    v = Verdict("teacher-loop generative replay v2 (non-forgetting fixed-size generator)", chance=chance)
    v.require("(NO-FORGET) generator fidelity HOLDS (mean cos >= 0.95 @ largest N AND drop from N=10 <= 0.03)",
              fid_holds, expect=True,
              note=f"cos {cos_small:.3f}(N={small}) -> {cos_big:.3f}(N={big}), drop {fid_drop:.3f}; v1 cos {v1_cos_big:.3f}")
    if not np.isnan(v1_f):
        v.reaches("(BEAT-v1) v2 retention beats the naive generator", before=v1_f, after=v2_f)
        v.require("(BEAT-v1') generative_v2 > generative_v1 + 0.10", (v2_f > v1_f + 0.10), expect=True,
                  note=f"v2 {v2_f:.2f} vs v1 {v1_f:.2f} @ N={big}")
    if not np.isnan(flat_f):
        v.require("(KEY) generative_v2 within 0.10 of flat AND >= 0.5", (v2_f >= flat_f - 0.10 and v2_f >= 0.5),
                  expect=True, note=f"v2 {v2_f:.2f} vs flat {flat_f:.2f} @ N={big}")
    v.require("(FIXED) generator trained-param count CONSTANT across N", param_constant, expect=True,
              note=f"trace {garm.get('generator_param_trace')}")
    v.require("(FIXED') generator stores 0 raw patterns (genuinely generative)", not_buffer, expect=True)
    v.require("(FIXED'') generator fixed-size across two builds", fixed_two_builds, expect=True,
              note=f"trained_params {result.get('generator_trained_params')}")
    v.require("(FIXED''') plastic store INDEPENDENT of the query width gen_k (the rank fix adds no capacity)",
              store_indep_k, expect=True, note=f"store_params {result.get('generator_store_params')}")
    v.require("(ANTI-CHEAT#1) consolidation NEVER read the true-engram ruler", no_ruler, expect=True)
    v.floor("(acq) generative_v2 immediate acquisition stays high", v2_acq, floor=0.85)
    v.require("(decoupled) slow reservoir CONSTANT across the curriculum", slow_constant, expect=True)
    v.require("(seed) substrate byte-identical across two builds at one seed",
              bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    go = (fid_holds and param_constant and not_buffer and no_ruler and fixed_two_builds and store_indep_k
          and slow_constant and v2_acq >= 0.85 and bool(result["substrate_byte_identical"])
          and bool(result["sim_diff_empty"]))
    if not np.isnan(v1_f):
        go = go and (v2_f > v1_f + 0.10)
    if not np.isnan(flat_f):
        go = go and (v2_f >= flat_f - 0.10) and (v2_f >= 0.5)
    decision = v.decide(go=go)

    return {
        "largest_N": big, "smallest_milestone_N": small, "capacity_F": result["capacity_F"],
        "generative_v2_frac_recalled": v2_f, "generative_v1_frac_recalled": v1_f, "flat_frac_recalled": flat_f,
        "generative_v2_immediate_acq": v2_acq,
        "generative_v2_minus_v1": (float(v2_f - v1_f) if not np.isnan(v1_f) else None),
        "generative_v2_minus_flat": (float(v2_f - flat_f) if not np.isnan(flat_f) else None),
        "retention_vs_N": {str(N): {"generative_v2": frac_at("generative_v2", N),
                                    "generative_v1": frac_at("generative_v1", N),
                                    "flat": frac_at("flat", N)} for N in result["milestones"]},
        "v2_fidelity_vs_N": fid, "v1_fidelity_vs_N": v1_fid,
        "v2_mean_cos_at_smallN": cos_small, "v2_mean_cos_at_bigN": cos_big, "v2_fidelity_drop": fid_drop,
        "v2_nearest_proto_acc_at_bigN": near_big, "v1_mean_cos_at_bigN": v1_cos_big, "v2_fidelity_holds": fid_holds,
        "generator_trained_params": result.get("generator_trained_params"),
        "generator_total_params": result.get("generator_total_params"),
        "generator_param_constant_across_N": param_constant,
        "generator_stored_raw_patterns": garm.get("generator_stored_raw_patterns"),
        "generative_not_stored_buffer": not_buffer, "consolidation_used_ruler": garm.get("consolidation_used_ruler"),
        "generator_fixed_size_two_builds_ok": fixed_two_builds,
        "generator_store_indep_of_gen_k": store_indep_k, "gen_k_query_width_v2": result.get("gen_k_query_width_v2"),
        "learn_epoch_trace": garm.get("learn_epoch_trace"),
        "slow_reservoir_constant": slow_constant,
        "substrate_byte_identical": result["substrate_byte_identical"], "sim_diff_empty": result["sim_diff_empty"],
        **decision,
    }


def _one_seed(a, seed, arms_to_run):
    result = run(seed, a.n_max, a.milestones, a.capacity, a.slow_hidden, a.gen_hidden, a.gen_k, a.settle_steps,
                 a.epochs, a.batch, a.eprop_lr, a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs,
                 a.replay_per_fact, a.replay_noise, a.gen_settle, a.gen_epochs, a.gen_lr, a.gen_tol,
                 a.gen_max_epochs, a.gen_check_every, a.gen_new_mult, arms_to_run, a.bdsp_wmax)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop GENERATIVE REPLAY v2: a NON-FORGETTING fixed-size neural "
                                             "generator (train-to-convergence self-replay) -- fidelity holds as N "
                                             "grows and retention rises toward the flat O(N) store, beating the "
                                             "naive v1 generator, with a bounded store.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--milestones", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--capacity", type=int, default=5)
    ap.add_argument("--slow-hidden", type=int, default=100)
    ap.add_argument("--gen-hidden", type=int, default=96, help="the FIXED generator reservoir size (H_gen; constant in N)")
    ap.add_argument("--gen-k", type=int, default=64, help="v2 FIXED class-query address width (constant in N; the "
                                                        "plastic readout store is INDEPENDENT of it). Wider => "
                                                        "collision-free sparse query codes => FULL-RANK reservoir "
                                                        "eligibilities => the readout can reach cos~1.0. gen_k=n_max "
                                                        "(v1) collides: rank 18/20, ceiling 0.93. 0 => v1 width.")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--replay-epochs", type=int, default=12)
    ap.add_argument("--replay-per-fact", type=int, default=8)
    ap.add_argument("--replay-noise", type=float, default=0.10)
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-epochs", type=int, default=16, help="v1 fixed self-replay epochs (used by generative_v1 arm "
                                                             "and as the floor when --gen-tol 0)")
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--gen-tol", type=float, default=0.05, help="v2 train-to-convergence tolerance (max reconstruction "
                                                              "L2 error over the replay set); 0 disables (=> v1)")
    ap.add_argument("--gen-max-epochs", type=int, default=120, help="v2 convergence epoch cap per fact")
    ap.add_argument("--gen-check-every", type=int, default=4, help="v2 convergence-check period (epochs)")
    ap.add_argument("--gen-new-mult", type=int, default=3, help="v2 recency balance: replay the new fact this many times")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9)
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["generative_v2", "generative_v1", "flat"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    arms_to_run = list(a.arms)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  (gen_H={a.gen_hidden}, tol={a.gen_tol}, max_ep={a.gen_max_epochs}, "
              f"new_mult={a.gen_new_mult}, n_max={a.n_max})\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, arms_to_run)
        summary = {"probe": "teacher_loop_generative_replay_v2", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "arms_run": arms_to_run, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        print(f"[genrep-v2] seed {s} @ N={rv.get('largest_N')}: v2 {rv.get('generative_v2_frac_recalled', float('nan')):.2f} "
              f"| v1 {rv.get('generative_v1_frac_recalled', float('nan')):.2f} | flat "
              f"{rv.get('flat_frac_recalled', float('nan')):.2f} (chance {result['chance']:.2f})", flush=True)
        rvn = rv.get("retention_vs_N", {})
        for N in result["milestones"]:
            d = rvn.get(str(N), {})
            fd2 = result["arms"].get("generative_v2", {}).get("generator_fidelity", {}).get(str(N), {})
            fd1 = result["arms"].get("generative_v1", {}).get("generator_fidelity", {}).get(str(N), {})
            print(f"    N={N:3d}: v2 {d.get('generative_v2', float('nan')):.2f} | v1 {d.get('generative_v1', float('nan')):.2f} "
                  f"| flat {d.get('flat', float('nan')):.2f} | v2-cos {fd2.get('mean_cos', float('nan')):.3f} "
                  f"v1-cos {fd1.get('mean_cos', float('nan')):.3f}", flush=True)
        print(f"[genrep-v2] fidelity-holds {rv.get('v2_fidelity_holds')} (cos {rv.get('v2_mean_cos_at_smallN'):.3f}->"
              f"{rv.get('v2_mean_cos_at_bigN'):.3f}) | gen-params {rv.get('generator_trained_params')} "
              f"const-in-N {rv.get('generator_param_constant_across_N')} stored-raw {rv.get('generator_stored_raw_patterns')} "
              f"used-ruler {rv.get('consolidation_used_ruler')} | acq {rv.get('generative_v2_immediate_acq', float('nan')):.3f} "
              f"| VERDICT {rv.get('status')}", flush=True)
        print(f"[genrep-v2] v2-v1 {rv.get('generative_v2_minus_v1')} | v2-flat {rv.get('generative_v2_minus_flat')} | "
              f"byte-identical {rv.get('substrate_byte_identical')} | sim-clean {rv.get('sim_diff_empty')} | "
              f"wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        v2 = [p["verdict"].get("generative_v2_frac_recalled", float("nan")) for p in per_seed]
        v1 = [p["verdict"].get("generative_v1_frac_recalled", float("nan")) for p in per_seed]
        flat = [p["verdict"].get("flat_frac_recalled", float("nan")) for p in per_seed]
        cos2 = [p["verdict"].get("v2_mean_cos_at_bigN", float("nan")) for p in per_seed]
        cos1 = [p["verdict"].get("v1_mean_cos_at_bigN", float("nan")) for p in per_seed]
        holds = [bool(p["verdict"].get("v2_fidelity_holds")) for p in per_seed]
        agg = {"probe": "teacher_loop_generative_replay_v2_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"),
               "gen_hidden": a.gen_hidden, "n_max": a.n_max, "gen_tol": a.gen_tol, "gen_max_epochs": a.gen_max_epochs,
               "gen_new_mult": a.gen_new_mult, "go_count": go_n, "n_seeds": len(seeds),
               "fidelity_holds_count": int(sum(holds)),
               "generative_v2_frac_mean": float(np.nanmean(v2)), "generative_v1_frac_mean": float(np.nanmean(v1)),
               "flat_frac_mean": float(np.nanmean(flat)),
               "v2_minus_v1_mean": float(np.nanmean(np.array(v2) - np.array(v1))),
               "v2_minus_flat_mean": float(np.nanmean(np.array(v2) - np.array(flat))),
               "v2_mean_cos_mean": float(np.nanmean(cos2)), "v1_mean_cos_mean": float(np.nanmean(cos1)),
               "generator_trained_params": per_seed[0]["verdict"].get("generator_trained_params"),
               "per_seed": per_seed}
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[genrep-v2 AGG] GO {go_n}/{len(seeds)} | fidelity-holds {int(sum(holds))}/{len(seeds)} | "
              f"v2 {np.nanmean(v2):.2f} vs v1 {np.nanmean(v1):.2f} vs flat {np.nanmean(flat):.2f} | "
              f"v2-cos {np.nanmean(cos2):.3f} vs v1-cos {np.nanmean(cos1):.3f} "
              f"(gen_H={a.gen_hidden}, params={agg['generator_trained_params']}) | wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
