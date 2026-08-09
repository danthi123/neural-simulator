"""TEACHER-LOOP PRIORITIZED-REPLAY DE-RISK (2026-08-09): bound the per-sleep CONSOLIDATION COMPUTE.

WHERE THE ARC STANDS. Retention at N=20 is CLOSED: a fixed-size non-forgetting generative-replay generator matches
the flat O(N) store (0.958 vs 0.950, 6-seed GO, 0933fdb7a). But BOTH the flat store AND the generator replay ALL N
facts EVERY sleep -- per-sleep replay-event count is O(N). Bounding STORAGE (the generator) does NOT bound per-step
COMPUTE: a year of learning still pays an O(N) consolidation bill every night. That is the owner's original SPEED
concern, and it is the residual this de-risk attacks.

THE HYPOTHESIS. The brain does NOT replay every memory every night. Replay is PRIORITIZED by expected value of a
backup -- the memories most in NEED of consolidation are reactivated preferentially (Mattar & Daw 2018,
doi:10.1038/s41593-018-0232-z, "prioritized memory access"; schema-gated systems consolidation, Tse et al. 2007,
doi:10.1126/science.1135935). Replay only a BOUNDED subset k<<N per sleep -- the facts most AT RISK of being
forgotten, chosen by a cheap NEURAL signal read from the brain's OWN readout -- and retention MATCHES full O(N)
replay, so the per-sleep replay-event count is O(k), INDEPENDENT of lifetime N.

THE NEURAL PRIORITY SIGNAL (the load-bearing anti-cheat). The at-risk score is the fact's CURRENT RECALL MARGIN read
from the cortical readout's OWN output on the hippocampal engram (the brain reactivates its stored cue and reads how
strongly its readout still points to the fact it encoded): margin_j = p[stored_label_j] - max_{c != label} p[c],
computed on the CURRENT net BEFORE this sleep's replay. LOW / NEGATIVE margin = the readout is losing the fact = HIGH
forgetting risk = HIGH replay priority. This is substrate-derived self-monitoring (a hippocampal-cortical match read),
NOT a host oracle peeking at which facts WILL be forgotten later: `_select_indices` receives ONLY (net, hippo) -- no
env, no future milestone, no post-hoc forgetting labels; the only label used is the one the hippocampus ITSELF stored
at wake (the brain's own memory of the fact). Tripwire `_priority_used_future` stays False.

FOUR ARMS (same net build / seed / env / wake budget / engram store; the ONLY difference is WHICH k of the N stored
engrams get self-replayed each sleep):
  * prioritized_k = TREATMENT. Replay the k lowest-recall-margin (most at-risk) engrams. O(k) events/sleep.
  * full          = TARGET, MEASURED in-run. Replay ALL N engrams (the current O(N) consolidation). Grows with N.
  * random_k      = CONTROL. Replay a RANDOM k engrams (same budget, no prioritization). Prioritized must BEAT this,
                    else the win is just "replaying fewer costs less", not "replaying the RIGHT ones".
  * recency_k     = optional 2nd control. Replay the k least-recently-touched engrams (a decaying recency trace, reset
                    on replay/encode -- oldest-since-refresh = at risk). A cheap non-readout heuristic; prioritized
                    (which reads the actual readout state) should at least match it.

STORAGE vs COMPUTE (kept distinct, on purpose). The engram store here is UNBOUNDED (holds all N) -- bounding STORAGE
is the generative-generator's job (already closed). This de-risk isolates the PER-SLEEP COMPUTE: the number of
consolidation replay EVENTS per sleep. `replay_set_size` per sleep is the witness: full = N (grows), k-arms <= k
(constant in N). Honest scope: the PRIORITY SCAN is an O(N) cheap INFERENCE forward-pass per stored engram (no
training); the expensive TRAINING consolidation is what becomes O(k). We report both -- the headline is the O(k)
replay-event (training) count; the O(N) selection scan is a cheap inference read named as a residual (a bounded
candidate pool closes it, --cand-pool, optional).

THE KEY MEASUREMENTS: (A) retention -- does prioritized_k match full (within margin) at N=20 AND N=50? (B) per-sleep
replay-event count -- prioritized O(k) vs full O(N)? (C) prioritized vs random_k -- does the neural priority BEAT a
random budget of the same size?

GO (largest N): prioritized_k within 0.15 of full AND >= 0.5; prioritized_k > random_k + 0.10; replay-set <= k at
every milestone (k << N) while full == N; priority signal neural (no future peek); acquisition >= 0.85; substrate
byte-identical; sim/ clean. If prioritized_k CANNOT match full, that is an HONEST NEGATIVE naming WHY (e.g. the margin
read is not predictive of which facts a future wake step will overwrite; the k budget is below the working-set churn).

DISCIPLINE: reuse-by-import (the fixed de-clamped slow cortex + anti-cheat asserts from the CLS two-store de-risk;
the hippocampal engram store + neural self-replay from the sleep-replay de-risk; the scaling teacher machinery;
ReferentEnv). NO sim/ edit. de-clamped bdsp_wmax=1e9. cfg.seed byte-identical (NOT actual_seed_used). SIM_BACKEND=numpy
(tiny launch-bound net). Test N=20 AND N=50; 6 seeds if feasible else >=3.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_prioritized_replay_derisk --seed 42 \
      --n-max 20 --milestones 10 20 --k 6 --slow-hidden 100 \
      --out research/findings/raw/teacher_loop_prioritized_replay_s42.json
  N=50 (single seed):  ... --n-max 50 --milestones 25 50 --k 6 --slow-hidden 220
  6-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_prioritized_replay_derisk --seeds 42 43 44 45 46 47 \
      --n-max 20 --milestones 10 20 --k 6 --slow-hidden 100 \
      --out research/findings/raw/teacher_loop_prioritized_replay_N20.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # tiny launch-bound net -> CPU faster (the teacher-loop runners are numpy)
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
# reuse-by-import: the fixed de-clamped slow cortex + the byte-identity / sim-clean asserts (CLS two-store de-risk);
# the hippocampal engram store + neural self-replay (sleep-replay de-risk); the scaling teacher machinery; the world.
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _build_slow_cortex, _assert_byte_identical_substrate, _git_sim_diff_empty,
)
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import Hippocampus  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._onbridge_eprop_port_derisk import _softmax  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_prioritized_replay.json"


# ===================== the engram store: unbounded, with SUBSET self-replay (a bounded k per sleep) =====================
class ReplayStore(Hippocampus):
    """The brain's own engram store (inherits wake `encode` + brain-owned generative `generate_replay`). UNBOUNDED
    (holds all N -- storage is the generator's problem, already closed). ADDS `generate_replay_subset`: self-generate
    replay draws from ONLY the selected engram indices, using the SAME brain-owned RNG + variability as the full store
    -- so replaying k engrams costs O(k) consolidation events while the store still holds N."""

    def generate_replay_subset(self, indices, per_fact):
        """Sleep self-generation restricted to `indices` (the prioritized subset). Identical formula to
        generate_replay (engram + brain_noise*N(0,1)), just over a chosen subset -> O(len(indices)) events."""
        idx = list(indices)
        if not idx:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
        Xs, ys = [], []
        for j in idx:
            engram, lab = self.engrams[j]
            for _ in range(per_fact):
                Xs.append(engram + self.replay_noise * self.rng.standard_normal(engram.shape[0]))
                ys.append(int(lab))
        return np.asarray(Xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


# ============================= the NEURAL priority signal: current recall margin from the readout =============================
def _recall_margin(net, engram, cls):
    """The at-risk score, read from the brain's OWN readout on a reactivated hippocampal engram. Forward the stored
    cue through the cortical readout, softmax the logits, return margin = p[stored_label] - max_{c != label} p[c].
    LOW / NEGATIVE = the readout is losing this fact = HIGH forgetting risk. Uses ONLY the current net + the engram +
    the label the hippocampus stored at wake -- NO env, NO future, NO post-hoc forgetting oracle."""
    sp, vv, acts = net._forward_record(np.asarray(engram, dtype=np.float64))
    p = _softmax(net._logits_from(sp, vv, acts) / net.logit_temp)
    other = float(np.max(np.delete(p, int(cls)))) if len(p) > 1 else 0.0
    return float(p[int(cls)] - other)


def _select_indices(net, hippo, k, mode, recency_trace, sel_rng, cand_pool=0, stochastic=True, beta=4.0):
    """Choose which stored engrams to replay this sleep. Returns (indices, used_future_flag). ANTI-CHEAT: this
    function has NO env / milestone / future-forgetting argument -- the ONLY per-fact information it may read is the
    CURRENT readout state (margin mode) or a substrate recency trace, plus the hippocampus's OWN stored labels.
    mode in {full, prioritized, random, recency}.

    prioritized uses the NEURAL recall margin as a forgetting-RISK weight. STOCHASTIC sampling (default; Mattar & Daw
    2018's stochastic prioritized sweep) samples k without replacement with prob ~ exp(beta * risk), risk = -margin:
    at-risk (low-margin) facts are far more likely, but EVERY fact keeps nonzero probability -> the greedy-top-k
    starvation (a few corrupted facts hog all k while others silently decay) is avoided. beta=0 -> uniform (==random);
    large beta -> greedy. GREEDY top-k is kept as an ablation (stochastic=False)."""
    n = len(hippo.engrams)
    all_idx = list(range(n))
    if mode == "full" or k <= 0 or n <= k:
        return all_idx, False
    # optional bounded candidate pool -> makes even the SELECTION scan O(pool) not O(N) (default 0 = scan all).
    if cand_pool and cand_pool > 0 and cand_pool < n:
        cand = list(sel_rng.choice(n, size=int(cand_pool), replace=False))
    else:
        cand = all_idx
    if mode == "random":
        return list(sel_rng.choice(n, size=int(k), replace=False)), False
    if mode == "recency":
        # least-recently-touched first (most-decayed trace = oldest since refresh = at risk). Substrate trace, no readout.
        order = sorted(cand, key=lambda j: recency_trace[j])
        return order[:int(k)], False
    if mode == "prioritized":
        # NEURAL: the readout itself scores each fact's recall margin; low margin = losing the fact = high risk.
        risk = np.array([-_recall_margin(net, hippo.engrams[j][0], hippo.engrams[j][1]) for j in cand], dtype=np.float64)
        if not stochastic:
            order = [j for _r, j in sorted(zip(-risk, cand))]        # greedy: highest risk (lowest margin) first
            return order[:int(k)], False
        w = np.exp(float(beta) * (risk - risk.max()))                # softmax weights over risk (numerically safe)
        w = w / w.sum()
        pick = sel_rng.choice(len(cand), size=int(k), replace=False, p=w)
        return [cand[int(p)] for p in pick], False
    raise ValueError(f"unknown mode {mode}")


# ============================================ one arm of the sequential curriculum ============================================
def _run_arm(arm, seed, referents, env, K, n_in, slow_hidden, k, settle, epochs, batch, eprop_lr, w_clip, n_draws,
             milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance, bdsp_wmax, recency_decay,
             cand_pool, priority_greedy, priority_beta):
    """arm in {prioritized_k, full, random_k, recency_k}. All arms share the fixed slow reservoir + the same wake
    budget + the same unbounded engram store; the ONLY manipulation is WHICH k of the N stored engrams are replayed
    each sleep. full replays all N (the O(N) target); the k-arms replay <= k (O(k))."""
    mode = {"prioritized_k": "prioritized", "full": "full", "random_k": "random", "recency_k": "recency"}[arm]

    net, slow_active0 = _build_slow_cortex(n_in, K, seed, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax,
                                           env, referents)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)   # brain-owned RNG for the interleaved consolidation shuffle
    sel_rng = np.random.default_rng(seed + 271)     # selection RNG (random_k / candidate pool)
    hippo = ReplayStore(seed, replay_noise=replay_noise)

    recency_trace = {}          # per-fact decaying recency trace (recency_k mode); reset to 1.0 on encode/replay
    acquire_acc = []
    retention = {}
    replay_set_sizes = []       # number of engrams replayed each sleep (the O(k) vs O(N) witness)
    replay_events = []          # replay TRAINING draws each sleep = replay_set_size * per_fact
    priority_scan_sizes = []    # forward-pass reads for selection each sleep (honest O(N)-inference residual)
    slow_active_trace = []
    used_future = False
    for i, r in enumerate(referents):
        # --- WAKE: teacher teaches fact i from the world (env draws = the legitimate sensory environment) ---
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acquire_acc.append(_fact_acc(net, env, r, i, n=test_n))     # IMMEDIATE acquisition, before consolidation
        # --- the store captures the engram of this episode ---
        hippo.encode(X, i)
        recency_trace[i] = 1.0
        # --- SLEEP: select a BOUNDED subset by the neural priority, self-replay ONLY that subset ---
        for j in list(recency_trace):
            recency_trace[j] *= recency_decay                        # decay all traces (oldest-since-refresh -> low)
        sel, uf = _select_indices(net, hippo, k, mode, recency_trace, sel_rng, cand_pool=cand_pool,
                                  stochastic=(not priority_greedy), beta=priority_beta)
        used_future = used_future or uf
        priority_scan_sizes.append(len(hippo.engrams) if mode == "prioritized" and (not cand_pool or cand_pool <= 0
                                   or cand_pool >= len(hippo.engrams)) else (int(cand_pool) if mode == "prioritized"
                                   else 0))
        Xr, yr = hippo.generate_replay_subset(sel, replay_per_fact)
        if len(Xr) > 0:
            _teach_fact(net, Xr, yr, replay_epochs, batch, brain_rng)
        for j in sel:
            recency_trace[j] = 1.0                                   # replaying refreshes the recency trace
        replay_set_sizes.append(len(sel))
        replay_events.append(len(sel) * int(replay_per_fact))
        slow_active_trace.append(int(net.n_active))
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "replay_set_size_this_sleep": int(replay_set_sizes[-1]),
                "max_replay_set_size_so_far": int(max(replay_set_sizes)),
                "slow_reservoir_active": int(net.n_active),
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
            }
    return {
        "arm": arm, "mode": mode, "k_budget": int(k),
        "slow_reservoir_active_start": slow_active0,
        "slow_reservoir_active_constant": bool(len(set(slow_active_trace)) == 1),
        "max_replay_set_size": int(max(replay_set_sizes)) if replay_set_sizes else 0,
        "replay_set_sizes": [int(x) for x in replay_set_sizes],
        "max_replay_events_per_sleep": int(max(replay_events)) if replay_events else 0,
        "total_replay_events": int(sum(replay_events)),
        "max_priority_scan_size": int(max(priority_scan_sizes)) if priority_scan_sizes else 0,
        "priority_used_future": bool(used_future),           # MUST be False (anti-cheat)
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention,
    }


def _assert_priority_signal_is_neural():
    """anti-cheat (structural): the selection function must NOT accept env / future / forgetting-label arguments --
    only (net, hippo, ...). Confirms the priority is read from the substrate, not a host oracle peeking at the
    future. Returns (ok, param_names)."""
    import inspect
    params = list(inspect.signature(_select_indices).parameters)
    banned = {"env", "future", "forgot", "will_forget", "oracle", "milestone", "referents", "accs"}
    ok = ("net" in params and "hippo" in params and not (set(params) & banned))
    return bool(ok), params


def run(seed, n_max, milestones, k, slow_hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
        test_n, replay_epochs, replay_per_fact, replay_noise, recency_decay, cand_pool, priority_greedy,
        priority_beta, arms_to_run, bdsp_wmax):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    referents = [f"ref{i}" for i in range(n_max)]

    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, K, seed, slow_hidden, settle, eprop_lr,
                                                               w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()
    neural_ok, sel_params = _assert_priority_signal_is_neural()

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        env = ReferentEnv(seed, d_p=d_p, noise=noise)          # fresh env per arm: identical referents + draw stream
        for r in referents:
            env.proto(r)
        env.rng = np.random.default_rng(seed + 101)            # reset the draw-stream so each arm sees the SAME percepts
        arms[arm] = _run_arm(arm, seed, referents, env, K, n_in, slow_hidden, k, settle, epochs, batch, eprop_lr,
                             w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise,
                             chance, bdsp_wmax, recency_decay, cand_pool, priority_greedy, priority_beta)
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(kk) for kk in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {arm:14s}] {arms[arm]['wall_seconds']:.0f}s | max-replay-set {arms[arm]['max_replay_set_size']:3d} "
              f"| immediate-acq {arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}",
              flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "k_budget": int(k), "slow_hidden": int(slow_hidden),
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "priority_signal_neural_structural": neural_ok, "select_params": sel_params,
            "config": {"k_budget": k, "slow_hidden": slow_hidden, "settle_steps": settle, "epochs": epochs,
                       "batch": batch, "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p,
                       "noise": noise, "test_n": test_n, "replay_epochs": replay_epochs,
                       "replay_per_fact": replay_per_fact, "replay_noise": replay_noise,
                       "recency_decay": recency_decay, "cand_pool": cand_pool, "priority_greedy": priority_greedy,
                       "priority_beta": priority_beta, "bdsp_wmax": bdsp_wmax, "frozen_hidden": True},
            "arms": arms}


def _verdict(result):
    """Verdict + GO. TEETH:
      (KEY) prioritized_k retention MATCHES full within 0.15 AND >= 0.5 at the largest N -- bounded compute at EQUAL
            retention is the win.
      (BEAT-random) prioritized_k > random_k + 0.10 -- the NEURAL priority beats a same-size random budget (else the
            'win' is just replaying fewer, not replaying the RIGHT ones).
      (O(k)) prioritized_k / random_k replay-set <= k at every milestone (k << N); full replay-set == N (grows).
      Anti-cheats: neural priority (no future peek), byte-identical substrate, sim/ clean, slow reservoir constant,
      acquisition >= 0.85. If prioritized_k CANNOT match full -> HONEST NEGATIVE naming why."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    arms = result["arms"]
    chance = result["chance"]
    k = result["k_budget"]

    def frac_at(arm, N):
        return arms.get(arm, {}).get("retention_curve", {}).get(str(N), {}).get("frac_recalled", float("nan"))

    def maxset_at(arm, N):
        return arms.get(arm, {}).get("retention_curve", {}).get(str(N), {}).get("max_replay_set_size_so_far", None)

    def big_of(arm):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return max((int(kk) for kk in rc), default=None)

    big = big_of("prioritized_k") or big_of("full") or big_of("random_k")
    ms = result["milestones"]

    pri_f = frac_at("prioritized_k", big)
    full_f = frac_at("full", big)
    rnd_f = frac_at("random_k", big)
    rec_f = frac_at("recency_k", big)
    pri_acq = arms.get("prioritized_k", {}).get("mean_acquire_acc_immediate", float("nan"))

    # partial-arm probe: if the treatment or the random control is missing, emit measured curves, skip GO.
    if "prioritized_k" not in arms or "random_k" not in arms:
        return {"largest_N": big, "prioritized_k_frac": pri_f, "full_frac": full_f, "random_k_frac": rnd_f,
                "status": "PARTIAL"}

    # O(k) boundedness: prioritized/random replay-set <= k at every milestone; full == N.
    pri_bounded = all((maxset_at("prioritized_k", N) is not None and maxset_at("prioritized_k", N) <= k) for N in ms)
    rnd_bounded = all((maxset_at("random_k", N) is not None and maxset_at("random_k", N) <= k) for N in ms)
    full_grows = (all((maxset_at("full", N) == N) for N in ms) if "full" in arms else None)
    k_much_less = bool(k <= max(ms) // 2)                         # k << N (at least half the largest milestone)
    slow_constant = bool(arms["prioritized_k"].get("slow_reservoir_active_constant"))
    neural_priority = bool(result.get("priority_signal_neural_structural")
                           and not arms["prioritized_k"].get("priority_used_future"))

    # the retention must be carried by WHICH facts get replayed (prioritized vs random, same budget), not just the budget.
    attributable_to("neural prioritization (prioritized_k vs random_k, same k budget)", pri_f, rnd_f)
    if not np.isnan(full_f):
        attributable_to("bounded k-replay matches full O(N) replay (prioritized_k vs full)", pri_f, full_f)

    v = Verdict("teacher-loop prioritized replay (bounded per-sleep compute)", chance=chance)
    if not np.isnan(full_f):
        v.reaches("(KEY) bounded prioritized-k retains near the full O(N) replay", before=rnd_f, after=pri_f)
        v.require("(KEY') prioritized_k within 0.15 of full AND above 0.5", (pri_f >= full_f - 0.15 and pri_f >= 0.5),
                  expect=True, note=f"prioritized {pri_f:.2f} vs full {full_f:.2f} @ N={big}")
    v.control("(BEAT-random) neural priority beats a random budget (prioritized_k vs random_k)", treatment=pri_f,
              control=rnd_f, min_separation=0.10)
    v.require("(O(k)) prioritized replay-set <= k at every milestone (k << N)", (pri_bounded and k_much_less),
              expect=True, note=f"k={k}, N_max={max(ms)}, pri max-set {[maxset_at('prioritized_k', N) for N in ms]}")
    if "full" in arms:
        v.require("(O(N)) full replay-set == N at every milestone (the grows-with-lifetime baseline)",
                  bool(full_grows), expect=True, note=f"full max-set {[maxset_at('full', N) for N in ms]}")
    v.require("(NEURAL) priority signal is substrate-derived (no future/forgetting oracle)", neural_priority,
              expect=True, note=f"select params {result.get('select_params')}")
    v.require("(decoupled) slow reservoir CONSTANT across the curriculum", slow_constant, expect=True)
    v.floor("(acq) prioritized_k immediate acquisition stays high", pri_acq, floor=0.85)
    v.require("(seed) substrate byte-identical across two builds at one seed",
              bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    go = (pri_bounded and k_much_less and neural_priority and slow_constant and pri_acq >= 0.85
          and (pri_f > rnd_f + 0.10) and bool(result["substrate_byte_identical"]) and bool(result["sim_diff_empty"]))
    if "full" in arms:
        go = go and bool(full_grows)
    if not np.isnan(full_f):
        go = go and (pri_f >= full_f - 0.15) and (pri_f >= 0.5)
    decision = v.decide(go=go)

    return {
        "largest_N": big, "k_budget": k,
        "prioritized_k_frac_recalled": pri_f, "full_frac_recalled": full_f, "random_k_frac_recalled": rnd_f,
        "recency_k_frac_recalled": rec_f, "prioritized_k_immediate_acq": pri_acq,
        "prioritized_minus_random": float(pri_f - rnd_f) if not np.isnan(rnd_f) else None,
        "prioritized_minus_full": (float(pri_f - full_f) if not np.isnan(full_f) else None),
        "retention_vs_N": {str(N): {"prioritized_k": frac_at("prioritized_k", N), "full": frac_at("full", N),
                                    "random_k": frac_at("random_k", N), "recency_k": frac_at("recency_k", N),
                                    "prioritized_max_set": maxset_at("prioritized_k", N),
                                    "full_max_set": maxset_at("full", N)} for N in ms},
        "prioritized_max_replay_set": arms["prioritized_k"].get("max_replay_set_size"),
        "full_max_replay_set": arms.get("full", {}).get("max_replay_set_size"),
        "prioritized_max_replay_events_per_sleep": arms["prioritized_k"].get("max_replay_events_per_sleep"),
        "full_max_replay_events_per_sleep": arms.get("full", {}).get("max_replay_events_per_sleep"),
        "prioritized_max_priority_scan_size": arms["prioritized_k"].get("max_priority_scan_size"),
        "prioritized_replay_set_bounded": pri_bounded, "random_replay_set_bounded": rnd_bounded,
        "full_replay_set_grows": full_grows, "k_much_less_than_N": k_much_less,
        "priority_signal_neural": neural_priority, "priority_used_future": arms["prioritized_k"].get("priority_used_future"),
        "slow_reservoir_constant": slow_constant,
        "substrate_byte_identical": result["substrate_byte_identical"], "sim_diff_empty": result["sim_diff_empty"],
        **decision,
    }


def _one_seed(a, seed, arms_to_run):
    result = run(seed, a.n_max, a.milestones, a.k, a.slow_hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact, a.replay_noise,
                 a.recency_decay, a.cand_pool, a.priority_greedy, a.priority_beta, arms_to_run, a.bdsp_wmax)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop PRIORITIZED REPLAY: replay only a bounded k<<N subset of "
                                             "stored engrams per sleep, chosen by the neural recall-margin (forgetting "
                                             "risk), to bound per-sleep consolidation compute at equal retention.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--milestones", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--k", type=int, default=6, help="the BOUNDED per-sleep replay budget (k << N; constant in N)")
    ap.add_argument("--slow-hidden", type=int, default=100, help="the FIXED slow cortical reservoir size (born fully "
                                                                 "at start; sized to the corpus, NEVER grown per-fact)")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20, help="per-fact WAKE teaching epochs")
    ap.add_argument("--replay-epochs", type=int, default=12, help="SLEEP self-replay epochs over the selected subset")
    ap.add_argument("--replay-per-fact", type=int, default=8, help="self-generated replay draws per selected engram")
    ap.add_argument("--replay-noise", type=float, default=0.10, help="brain-owned variability on the replayed engram")
    ap.add_argument("--recency-decay", type=float, default=0.8, help="recency_k: per-sleep trace decay (reset on replay)")
    ap.add_argument("--cand-pool", type=int, default=0, help="prioritized: bounded candidate pool for the margin scan "
                                                            "(0 = scan all N; >0 makes even selection O(pool))")
    ap.add_argument("--priority-greedy", action="store_true", help="prioritized: greedy top-k by margin (ablation) "
                                                                 "instead of the default STOCHASTIC risk-weighted sweep")
    ap.add_argument("--priority-beta", type=float, default=4.0, help="stochastic prioritized: risk-weight temperature "
                                                                   "(0 -> uniform==random; large -> greedy)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9, help="1e9 = de-clamped (required; 6 = the CLAMP that "
                                                                "silences the reservoir, bound-trap 8ca014ff2)")
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["prioritized_k", "full", "random_k", "recency_k"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    arms_to_run = list(a.arms)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  (k={a.k}, slow_hidden={a.slow_hidden}, n_max={a.n_max})\n"
              + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, arms_to_run)
        summary = {"probe": "teacher_loop_prioritized_replay", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "arms_run": arms_to_run, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        print(f"[prio-replay] seed {s} @ N={rv.get('largest_N')}: prioritized {rv.get('prioritized_k_frac_recalled', float('nan')):.2f} "
              f"| full {rv.get('full_frac_recalled', float('nan')):.2f} | random "
              f"{rv.get('random_k_frac_recalled', float('nan')):.2f} | recency "
              f"{rv.get('recency_k_frac_recalled', float('nan')):.2f} (chance {result['chance']:.2f})", flush=True)
        rvn = rv.get("retention_vs_N", {})
        for N in result["milestones"]:
            d = rvn.get(str(N), {})
            print(f"    N={N:3d}: prioritized {d.get('prioritized_k', float('nan')):.2f} (max-set={d.get('prioritized_max_set')}) "
                  f"| full {d.get('full', float('nan')):.2f} (max-set={d.get('full_max_set')}) | "
                  f"random {d.get('random_k', float('nan')):.2f} | recency {d.get('recency_k', float('nan')):.2f}",
                  flush=True)
        print(f"[prio-replay] prioritized-random {rv.get('prioritized_minus_random')} | prioritized-full "
              f"{rv.get('prioritized_minus_full')} | replay-events/sleep prioritized "
              f"{rv.get('prioritized_max_replay_events_per_sleep')} vs full {rv.get('full_max_replay_events_per_sleep')} "
              f"| k<<N {rv.get('k_much_less_than_N')} | neural {rv.get('priority_signal_neural')} "
              f"(future {rv.get('priority_used_future')}) | acq {rv.get('prioritized_k_immediate_acq', float('nan')):.3f} "
              f"| VERDICT {rv.get('status')}", flush=True)
        print(f"[prio-replay] byte-identical {rv.get('substrate_byte_identical')} | sim-clean {rv.get('sim_diff_empty')} | "
              f"wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        pri = [p["verdict"].get("prioritized_k_frac_recalled", float("nan")) for p in per_seed]
        full = [p["verdict"].get("full_frac_recalled", float("nan")) for p in per_seed]
        rnd = [p["verdict"].get("random_k_frac_recalled", float("nan")) for p in per_seed]
        rec = [p["verdict"].get("recency_k_frac_recalled", float("nan")) for p in per_seed]
        agg = {"probe": "teacher_loop_prioritized_replay_AGG", "seeds": seeds, "backend": os.environ.get("SIM_BACKEND"),
               "k_budget": a.k, "slow_hidden": a.slow_hidden, "n_max": a.n_max,
               "go_count": go_n, "n_seeds": len(seeds),
               "prioritized_k_frac_mean": float(np.nanmean(pri)), "full_frac_mean": float(np.nanmean(full)),
               "random_k_frac_mean": float(np.nanmean(rnd)), "recency_k_frac_mean": float(np.nanmean(rec)),
               "prioritized_minus_random_mean": float(np.nanmean(np.array(pri) - np.array(rnd))),
               "prioritized_minus_full_mean": float(np.nanmean(np.array(pri) - np.array(full))),
               "prioritized_max_replay_events_per_sleep": per_seed[0]["verdict"].get("prioritized_max_replay_events_per_sleep"),
               "full_max_replay_events_per_sleep": per_seed[0]["verdict"].get("full_max_replay_events_per_sleep"),
               "per_seed": per_seed}
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[prio-replay AGG] GO {go_n}/{len(seeds)} | prioritized {np.nanmean(pri):.2f} vs full {np.nanmean(full):.2f} "
              f"vs random {np.nanmean(rnd):.2f} vs recency {np.nanmean(rec):.2f} | "
              f"replay-events/sleep pri {agg['prioritized_max_replay_events_per_sleep']} vs full "
              f"{agg['full_max_replay_events_per_sleep']} (k={a.k}, N={a.n_max}) | wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
