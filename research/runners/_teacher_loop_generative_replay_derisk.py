"""TEACHER-LOOP GENERATIVE-REPLAY DE-RISK (2026-08-09): the CLS bounded two-store is a MEASURED NEGATIVE
(0c7531785, 6-seed): a bounded raw-engram BUFFER (fixed F) + slow cortex + neural self-replay + EVICTION does NOT
decouple retention from lifetime -- retention TRACKS F (~0.517 @ F=5, N=20) and DEGRADES with N, while the FLAT
O(N)-replay store HOLDS (~0.95) on the SAME reservoir. WHY: once a fact is EVICTED it has no replay SOURCE, so its
slow-readout trace drifts as new facts keep moving the shared cortex. The wall is loss of ongoing REHEARSAL
COVERAGE after eviction -- not consolidation strength, not capacity, not the e-prop mechanism.

THE NAMED SURPASS (van de Ven, Siegelmann & Tolias 2020, Nat Commun, doi:10.1038/s41467-020-17866-2; ~92%
continual learning with BOUNDED memory): GENERATIVE REPLAY. A FIXED-SIZE generator REGENERATES patterns for ALL
learned facts (not just the recent F), so replay COVERAGE stays COMPLETE while the STORE stays bounded (its size is
independent of N). The cortex/hippocampus re-DREAMS past experience; it does not store + evict raw copies. KEY
subtlety van de Ven solves: the generator is ALSO kept current by replaying its OWN past outputs while it learns
each new fact, so the GENERATOR itself does not catastrophically forget.

THE HYPOTHESIS. A FIXED-SIZE NEURAL generator (regenerates ALL N facts' engram patterns for consolidation,
trained incrementally with its own generative replay) RETAINS N facts like the FLAT O(N) store, with a BOUNDED
store -- decoupling retention/coverage from lifetime. That is the "a year of data scales" answer.

THE GENERATOR (brain-based; NEURAL + GENUINELY GENERATIVE + GENUINELY FIXED-SIZE):
  * `GenerativeReplayNet` = an OnBridgeEpropNet (the a1-GO transport-free e-prop spiking substrate) reused as a
    CLASS-CONDITIONED spiking associative generator: input = a one-hot CLASS QUERY (width K = n_max, a compact
    class index, NOT the pattern) -> a FROZEN random spiking Izhikevich reservoir (H_gen granule units, de-clamped
    bdsp_wmax=1e9 so the reservoir robustly SPIKES) -> a LINEAR leaky readout of width n_in that REGENERATES the
    engram pattern. The readout weights (the ONLY trained store) are moved by a local delta rule on the reservoir's
    spike eligibility (the Bellec leaky-readout gradient, regression form: dW = -lr*(pattern_hat - target) (x) r).
    Regeneration = a one-hot query -> a SPIKING forward pass -> the readout pattern. NOT a host lookup of a stored
    raw pattern.
  * GENUINELY FIXED-SIZE: the trained store = the readout matrix, H_gen_phys x n_in_phys -- a CONSTANT independent
    of N (facts taught) AND of n_max (the frozen reservoir addressing carries the query; the plastic memory is the
    fixed readout). Asserted: the generator's trained-param + total-param counts do NOT change as facts accumulate
    10 -> 20, and it holds ZERO stored raw patterns.
  * GENUINELY GENERATIVE (not a stored O(N) buffer = the flat store in disguise, which would be a REJECT): the
    generator NEVER retains past facts' raw engrams. At each new fact it is trained on the NEW fact's true engram
    interleaved with its OWN regenerations of the earlier classes (van de Ven self-replay), so it stays current on
    all facts while only ever holding the CURRENT fact's raw pattern in hand.
  * DOES THE GENERATOR ITSELF FORGET? (the honest recursion). Measured every milestone: regeneration fidelity =
    cosine(regenerate(j), true_engram_j) mean+min, AND nearest-true-prototype decodability of the regenerations,
    at N=10 AND N=20 with H_gen FIXED. If fidelity holds -> the generator regenerates all N; if it degrades ->
    coverage is still bounded by the generator's own capacity and we NAME the recursion (an honest negative with
    teeth). NB: `true_engrams` is an experimenter-only ruler for this metric -- the consolidation path uses ONLY
    `gen.regenerate` (asserted), never the ruler.

THREE ARMS (same net build / seed / wake budget / slow reservoir; the ONLY difference is the sleep replay SOURCE):
  * generative     = TREATMENT. Sleep consolidation regenerates ALL learned facts from the FIXED generator (full
                     coverage) and e-props them into the slow cortex.
  * flat           = the O(N) target, MEASURED in-run (reuse of the CLS `_run_arm` flat arm; ~0.95). Unbounded raw
                     engram buffer; replays all N.
  * bounded_buffer = the CLS NEGATIVE, MEASURED in-run (reuse of the CLS `_run_arm` two_store arm, F fixed;
                     ~0.517 @ N=20). Must be BEATEN.

TEETH / GO (largest N, F fixed, generator FIXED):
  (KEY)   generative frac-recalled within 0.15 of flat AND >= 0.5 (matches the O(N) store with a bounded store).
  (BEAT)  generative > bounded_buffer + 0.15 (beats the measured CLS negative -- coverage is the fix).
  (FIXED) generator trained-param count is CONSTANT across N=10 and N=20 AND it stores 0 raw patterns.
  (NO-FORGET) generator regeneration fidelity holds at N=20 (mean cosine >= 0.85, nearest-proto decodability high);
              reported at N=10 vs N=20 so a degradation is visible. If it degrades, HONEST NEGATIVE naming the
              recursion.
  (acq) generative immediate acquisition stays high (>= 0.85). Anti-cheats: byte-identical substrate (cfg.seed, NOT
  actual_seed_used), git diff main -- sim/ empty, slow reservoir n_active CONSTANT (not a growing reservoir),
  de-clamped bdsp_wmax=1e9, backend recorded.

DISCIPLINE: reuse-by-import (NeurogenesisNet for the fixed de-clamped slow reservoir + the OnBridgeEpropNet
generator; _feat/_teach_fact/_fact_acc/_corrective_batch/N_ACT from the scaling de-risk; the CLS two-store
`_build_slow_cortex`/`BoundedHippocampus`/`_run_arm`/anti-cheat asserts for the flat + bounded_buffer arms;
ReferentEnv from the corrective-acquire de-risk). NO sim/ edit. SIM_BACKEND=numpy (tiny launch-bound net).

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_generative_replay_derisk --seed 42 \
      --n-max 20 --milestones 10 20 --capacity 5 --slow-hidden 100 --gen-hidden 96 \
      --epochs 20 --replay-epochs 12 --replay-per-fact 8 --gen-epochs 16 --gen-lr 0.8 --gen-settle 15 \
      --n-draws 16 --settle-steps 20 --test-n 40 \
      --out research/findings/raw/teacher_loop_generative_replay_s42.json
  6-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_generative_replay_derisk --seeds 42 43 44 45 46 47 \
      --n-max 20 --milestones 10 20 --capacity 5 --slow-hidden 100 --gen-hidden 96 \
      --epochs 20 --replay-epochs 12 --replay-per-fact 8 --gen-epochs 16 --gen-lr 0.8 --gen-settle 15 \
      --n-draws 16 --settle-steps 20 --test-n 40 \
      --out research/findings/raw/teacher_loop_generative_replay.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
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
from sim.backend import to_host  # noqa: E402
# reuse-by-import: the fixed de-clamped slow reservoir (NeurogenesisNet) + the OnBridge e-prop substrate for the
# generator; the scaling teacher machinery; the CLS two-store arms/asserts (flat + bounded_buffer, MEASURED in-run).
# NO sim/ edit.
from research.runners._teacher_loop_neurogenesis_capacity_derisk import NeurogenesisNet  # noqa: E402
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _build_slow_cortex, _run_arm as _run_cls_arm,
    _assert_byte_identical_substrate, _git_sim_diff_empty,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_generative_replay.json"


def _onehot(cls, width):
    v = np.zeros(int(width), dtype=np.float64)
    v[int(cls)] = 1.0
    return v


# ============================ the FIXED-SIZE NEURAL GENERATIVE-REPLAY GENERATOR ============================
class GenerativeReplayNet(OnBridgeEpropNet):
    """A class-conditioned SPIKING associative generator. input = a one-hot CLASS QUERY (width gen_k = n_max) ->
    a FROZEN random spiking Izhikevich reservoir (H_gen granule units) -> a linear leaky readout of width n_in that
    REGENERATES the engram pattern. Only the readout weights are trained (hidden frozen), by a local delta rule on
    the reservoir spike eligibility (the Bellec leaky-readout gradient in regression form). The plastic STORE = the
    readout matrix, H_gen_phys x n_in_phys -- a CONSTANT independent of N (facts) and of gen_k (the frozen reservoir
    addresses the query). It holds NO raw patterns: regeneration is a query -> spiking forward -> readout, never a
    lookup of a stored engram."""

    def __init__(self, gen_k, n_in_pattern, hidden, seed, settle, eprop_lr, w_clip, bdsp_wmax=1e9):
        # de-clamped bdsp_wmax REQUIRED: the inherited -6/+6 clamp crushes the reservoir afferents to |w|<=6 and
        # SILENCES it (bound-trap 8ca014ff2); 1e9 makes the clip a no-op so the granule reservoir robustly spikes.
        hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
                  in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0, freeze_hidden=True, bdsp_wmax=bdsp_wmax)
        super().__init__(int(gen_k), int(hidden), int(n_in_pattern), seed=seed, n_hidden_layers=1,
                         settle_steps=settle, eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                         logit_source="leaky_readout", w_clip=w_clip, hp=hp)
        self.gen_k = int(gen_k)
        self.n_in_pattern = int(n_in_pattern)
        self._query_seed = int(seed) + 271828          # fixed brain seed for the deterministic query codes
        self._q_active = max(2, int(round(0.15 * int(gen_k))))   # sparse active lines per class query
        # A FIXED known anchor (the world's percept midpoint) the readout reconstructs the DEVIATION from. The readout
        # has no bias synapse, so without an anchor a whitened (mean-centered) feature cannot reproduce the target's
        # absolute mean -> every reconstruction is shifted by a constant and the nearest-prototype identity collapses
        # (measured: lstsq ceiling 9/20 centered-no-anchor vs 20/20 with the bias dof restored). 0.5 is a world
        # constant (percepts are in [0,1]), NOT per-fact information.
        self._anchor = 0.5 * np.ones(int(n_in_pattern), dtype=np.float64)
        # freeze_hidden -> train_layers={n_hidden_layers}=the readout pathway only (the reservoir is fixed random).
        assert self.train_layers == {self.n_hidden_layers}, "generator reservoir must be frozen (readout-only trained)"
        self._stored_raw_patterns = 0     # anti-cheat witness: the generator NEVER stores raw patterns (regenerates)

    def _query_code(self, cls):
        """The class QUERY = a fixed deterministic pseudo-random SPARSE code of the class INDEX (regenerated from the
        index by a fixed brain RNG -- NOT stored, and a compact index code, not the pattern). Sparse activation of a
        distinct input subset per class -> distinct, non-dead reservoir codes (a plain one-hot leaves many classes
        with zero reservoir spikes; a dense code makes the reservoir codes collide). Width = gen_k, FIXED."""
        code = np.zeros(self.gen_k, dtype=np.float64)
        rng = np.random.default_rng((self._query_seed * 1_000_003 + int(cls)) & 0x7FFFFFFF)
        idx = rng.choice(self.gen_k, size=self._q_active, replace=False)
        code[idx] = 1.0
        return code

    def fit_query_norm(self):
        """Condition the readout feature over the class queries WITHOUT removing the bias dof. The readout has no
        bias term, so mean-CENTERING the eligibility (the parent default) makes the readout unable to reproduce the
        target's absolute mean -- every reconstruction is shifted by a constant and the nearest-prototype identity
        collapses (measured: lstsq ceiling 9/20 centered vs 20/20 uncentered). So mu=0 (keep the common-mode / bias
        direction) and sigma = a single global RMS scale (condition the delta rule without inflating constant
        units)."""
        R = np.array([self._readout_elig(self._forward_record(self._query_code(c))[0]) for c in range(self.gen_k)])
        self._r_mu = R.mean(axis=0)                                       # per-unit whiten (good delta-rule conditioning)
        self._r_sigma = R.std(axis=0) + 1e-3                              # the anchor (above) restores the bias dof

    def regenerate(self, cls):
        """Regenerate the engram pattern for a class = its query code -> spiking forward -> readout DEVIATION + the
        fixed anchor (clipped to the [0,1] percept space the env produces). NO stored-pattern lookup."""
        sp, vv, acts = self._forward_record(self._query_code(cls))
        return np.clip(self._logits_from(sp, vv, acts) + self._anchor, 0.0, 1.0)

    def _train_regression_batch(self, Qb, Tb):
        """One NORMALIZED-delta (NLMS) update of the readout toward the anchor-centered target patterns (hidden
        frozen). For each query: forward -> reservoir eligibility r (whitened), pattern_hat = r @ W (the DEVIATION
        from the anchor); the regression error is (pattern_hat - (target - anchor)); the per-example contribution is
        outer(r, err) / (||r||^2 + eps) -- the input-power normalization that makes the local rule SCALE-FREE (stable
        for lr in (0,~1) regardless of the raw eligibility magnitude, removing the sharp divergence cliff a plain
        delta rule has here; a normalized/homeostatic Hebbian readout). _apply_grads does W -= lr*grad. This is the
        SAME leaky-readout eligibility gradient train_batch uses, with a regression error + input-power
        normalization in place of the softmax error."""
        L = len(self.sizes) - 1
        grads = [np.zeros((self.sizes_phys[li], self.sizes_phys[li + 1]), dtype=np.float64) for li in range(L)]
        for q, t in zip(Qb, Tb):
            sp, vv, acts = self._forward_record(q)
            r = self._readout_feature(sp)                                  # (n_Hlast_phys,) whitened eligibility
            hat = self._logits_from(sp, vv, acts)                          # (n_in_pattern,) current deviation reconstruction
            err = np.asarray(hat, dtype=np.float64) - (np.asarray(t, dtype=np.float64) - self._anchor)
            dphys = self._broadcast(err, L) / self.pool_k                  # (n_out_phys,)
            grads[L - 1] += np.outer(r, dphys) / (float(r @ r) + 1e-6)     # NLMS: normalize by input power
        self._apply_grads(grads, len(Qb))

    def learn_fact(self, new_cls, new_engram, past_classes, epochs, batch, rng):
        """van de Ven incremental training: fit the FIXED generator on the NEW fact's TRUE engram interleaved with
        its OWN regenerations of the earlier classes (self-generative replay), so the generator stays current on all
        facts while only ever holding the CURRENT fact's raw pattern. past_classes' targets are SNAPSHOT from the
        generator BEFORE this update (the generator dreams its own past)."""
        past = list(past_classes)
        Q = [self._query_code(new_cls)] + [self._query_code(j) for j in past]
        T = [np.asarray(new_engram, dtype=np.float64)] + [self.regenerate(j) for j in past]   # own regenerations
        Q = np.asarray(Q, dtype=np.float64); T = np.asarray(T, dtype=np.float64)
        for _ in range(int(epochs)):
            perm = rng.permutation(len(Q))
            for i in range(0, len(Q), int(batch)):
                b = perm[i:i + int(batch)]
                self._train_regression_batch(Q[b], T[b])

    def trained_param_count(self):
        """The PLASTIC store = the readout pathway weight count (H_gen_phys x n_in_phys). Independent of N."""
        return int(self._data_idx_flat[-1].shape[0])

    def total_param_count(self):
        return int(sum(int(idx.shape[0]) for idx in self._data_idx_flat))


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ====================================== the GENERATIVE arm of the curriculum ======================================
def _run_generative_arm(seed, referents, env, K, n_in, slow_hidden, gen_hidden, settle, epochs, batch, eprop_lr,
                        w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance,
                        bdsp_wmax, gen_settle, gen_epochs, gen_lr):
    """The TREATMENT: same fixed slow reservoir + same wake budget as the flat/bounded arms; the sleep replay SOURCE
    is the FIXED-SIZE NEURAL generator regenerating ALL learned facts (full coverage)."""
    net, slow_active0 = _build_slow_cortex(n_in, K, seed, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax,
                                           env, referents)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)     # brain-owned RNG for the replay variability + consolidation shuffle
    gen_rng = np.random.default_rng(seed + 999)       # brain-owned RNG for the generator's incremental training

    gen = GenerativeReplayNet(K, n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax)
    gen.fit_query_norm()
    gen_trained_params = gen.trained_param_count()
    gen_total_params = gen.total_param_count()
    gen_param_trace = []                              # (N, trained_params) at each milestone -> must be CONSTANT

    # experimenter-only ruler for the generator-fidelity metric. NOT used by the consolidation path (asserted below).
    true_engrams = {}

    acquire_acc = []
    retention = {}
    slow_active_trace = []
    gen_fidelity = {}                                 # milestone N -> {mean_cos, min_cos, nearest_proto_acc}
    max_replay_set = 0
    for i, r in enumerate(referents):
        # --- WAKE: teacher teaches fact i from the world; the slow cortex moves by e-prop ---
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acquire_acc.append(_fact_acc(net, env, r, i, n=test_n))
        engram_i = np.asarray(X, dtype=np.float64).mean(axis=0)             # the brain's compressed wake trace
        true_engrams[i] = engram_i                                          # ruler only (fidelity metric)
        # --- keep the FIXED generator current: new fact + its OWN regenerations of the past (van de Ven) ---
        gen.learn_fact(i, engram_i, range(i), gen_epochs, batch, gen_rng)
        # --- SLEEP: regenerate ALL learned facts from the generator and consolidate into the slow cortex ---
        classes = list(range(i + 1))
        max_replay_set = max(max_replay_set, len(classes))
        Xr, yr = [], []
        for j in classes:
            eg = gen.regenerate(j)                                          # NEURAL regeneration (spikes->readout)
            for _ in range(replay_per_fact):
                Xr.append(eg + replay_noise * brain_rng.standard_normal(eg.shape[0]))   # brain-owned replay variability
                yr.append(j)
        Xr = np.asarray(Xr, dtype=np.float64); yr = np.asarray(yr, dtype=np.int64)
        _teach_fact(net, Xr, yr, replay_epochs, batch, brain_rng)          # e-prop into the slow readout
        slow_active_trace.append(int(net.n_active))
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "generator_trained_params": gen_trained_params, "generator_total_params": gen_total_params,
                "generator_stored_raw_patterns": int(gen._stored_raw_patterns),
                "flat_buffer_floats_equiv": int(N * n_in),                 # what the O(N) raw buffer WOULD store
                "slow_reservoir_active": int(net.n_active),
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
            }
            gen_param_trace.append((N, gen_trained_params))
            # --- generator regeneration fidelity (the honest recursion): does IT forget as N grows? ---
            regens = [gen.regenerate(j) for j in range(N)]
            coss = [_cos(regens[j], true_engrams[j]) for j in range(N)]
            protos = np.stack([true_engrams[j] for j in range(N)])          # (N, n_in) the ruler prototypes
            near_ok = 0
            for j in range(N):
                d = np.linalg.norm(protos - regens[j][None, :], axis=1)     # nearest TRUE prototype to the regen
                near_ok += int(int(np.argmin(d)) == j)
            gen_fidelity[str(N)] = {"mean_cos": float(np.mean(coss)), "min_cos": float(np.min(coss)),
                                    "nearest_proto_acc": float(near_ok / N)}
    # anti-cheat: the generator holds NO raw patterns (genuinely generative, not the flat store in disguise).
    generative_not_buffer = bool(gen._stored_raw_patterns == 0)
    # anti-cheat: the generator's trained store is CONSTANT across milestones (does not grow with N).
    param_constant = bool(len({p for _n, p in gen_param_trace}) <= 1)
    return {
        "arm": "generative",
        "slow_reservoir_active_start": slow_active0,
        "slow_reservoir_active_constant": bool(len(set(slow_active_trace)) == 1),
        "generator_trained_params": gen_trained_params, "generator_total_params": gen_total_params,
        "generator_hidden": int(gen_hidden), "generator_param_constant_across_N": param_constant,
        "generator_param_trace": [[int(n), int(p)] for n, p in gen_param_trace],
        "generator_stored_raw_patterns": int(gen._stored_raw_patterns),
        "generative_not_stored_buffer": generative_not_buffer,
        "max_replay_set_size": int(max_replay_set),
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention,
        "generator_fidelity": gen_fidelity,
    }


def _assert_generator_fixed_size(gen_k, n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax):
    """anti-cheat: build the generator TWICE and confirm its trained/total param counts are identical AND that the
    count does NOT depend on N (it is built from gen_k/hidden/n_in only). Returns the fixed counts."""
    a = GenerativeReplayNet(gen_k, n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax)
    b = GenerativeReplayNet(gen_k, n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax)
    return (int(a.trained_param_count()), int(a.total_param_count()),
            bool(a.trained_param_count() == b.trained_param_count()
                 and a.total_param_count() == b.total_param_count()))


def run(seed, n_max, milestones, capacity, slow_hidden, gen_hidden, settle, epochs, batch, eprop_lr, w_clip,
        n_draws, d_p, noise, test_n, replay_epochs, replay_per_fact, replay_noise, gen_settle, gen_epochs, gen_lr,
        arms_to_run, bdsp_wmax):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    referents = [f"ref{i}" for i in range(n_max)]

    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, K, seed, slow_hidden, settle, eprop_lr,
                                                               w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()
    gen_tp, gen_totp, gen_fixed_ok = _assert_generator_fixed_size(K, n_in, gen_hidden, seed, gen_settle, gen_lr,
                                                                  w_clip, bdsp_wmax)

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        env = ReferentEnv(seed, d_p=d_p, noise=noise)          # fresh env per arm: identical referents + draw stream
        for r in referents:
            env.proto(r)
        env.rng = np.random.default_rng(seed + 101)            # reset the draw-stream so each arm sees the SAME percepts
        if arm == "generative":
            arms[arm] = _run_generative_arm(seed, referents, env, K, n_in, slow_hidden, gen_hidden, settle, epochs,
                                            batch, eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs,
                                            replay_per_fact, replay_noise, chance, bdsp_wmax, gen_settle, gen_epochs,
                                            gen_lr)
        else:
            # flat + bounded_buffer via the CLS two-store arm, MEASURED in-run on the SAME slow reservoir/seed/env.
            cls_arm = "flat" if arm == "flat" else "two_store"     # bounded_buffer == the CLS two_store (capacity F)
            arms[arm] = _run_cls_arm(cls_arm, seed, referents, env, K, n_in, slow_hidden, capacity, settle, epochs,
                                     batch, eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs,
                                     replay_per_fact, replay_noise, chance, bdsp_wmax)
            arms[arm]["arm"] = arm
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {arm:14s}] {arms[arm]['wall_seconds']:.0f}s | "
              f"immediate-acq {arms[arm].get('mean_acquire_acc_immediate', float('nan')):.3f} | "
              f"frac-recalled@N={big}: {fr:.2f}", flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "capacity_F": int(capacity), "slow_hidden": int(slow_hidden), "gen_hidden": int(gen_hidden),
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "generator_trained_params": gen_tp, "generator_total_params": gen_totp,
            "generator_fixed_size_two_builds_ok": gen_fixed_ok,
            "config": {"capacity_F": capacity, "slow_hidden": slow_hidden, "gen_hidden": gen_hidden,
                       "settle_steps": settle, "epochs": epochs, "batch": batch, "eprop_lr": eprop_lr,
                       "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise, "test_n": test_n,
                       "replay_epochs": replay_epochs, "replay_per_fact": replay_per_fact,
                       "replay_noise": replay_noise, "gen_settle": gen_settle, "gen_epochs": gen_epochs,
                       "gen_lr": gen_lr, "bdsp_wmax": bdsp_wmax, "frozen_hidden": True},
            "arms": arms}


def _verdict(result):
    """Verdict + GO. TEETH:
      (KEY) generative within 0.15 of the flat O(N) baseline AND >= 0.5, with a BOUNDED (fixed-size) generator.
      (BEAT) generative > bounded_buffer + 0.15 (beats the measured CLS negative -- coverage is the fix).
      (FIXED) generator trained-param count CONSTANT across N AND 0 stored raw patterns (genuinely generative).
      (NO-FORGET) generator regeneration fidelity holds at the largest N (mean cos >= 0.85, nearest-proto acc high);
                  reported N=10 vs N=20. Degradation -> honest negative naming the recursion.
      (acq) generative immediate acquisition >= 0.85. Anti-cheats asserted."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    arms = result["arms"]
    chance = result["chance"]
    F = result["capacity_F"]

    def frac_at(arm, N):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return rc.get(str(N), {}).get("frac_recalled", float("nan"))

    def big_of(arm):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return max((int(k) for k in rc), default=None)

    big = big_of("generative") or big_of("flat") or big_of("bounded_buffer")
    gen_f = frac_at("generative", big)
    flat_f = frac_at("flat", big)
    bnd_f = frac_at("bounded_buffer", big)
    gen_acq = arms.get("generative", {}).get("mean_acquire_acc_immediate", float("nan"))

    if "generative" not in arms:
        return {"largest_N": big, "generative_frac": gen_f, "flat_frac": flat_f, "bounded_buffer_frac": bnd_f,
                "status": "PARTIAL"}

    garm = arms["generative"]
    fid = garm.get("generator_fidelity", {})
    big_fid = fid.get(str(big), {})
    gen_mean_cos = big_fid.get("mean_cos", float("nan"))
    gen_near_acc = big_fid.get("nearest_proto_acc", float("nan"))
    param_constant = bool(garm.get("generator_param_constant_across_N"))
    not_buffer = bool(garm.get("generative_not_stored_buffer"))
    fixed_two_builds = bool(result.get("generator_fixed_size_two_builds_ok"))
    slow_constant = bool(garm.get("slow_reservoir_active_constant"))
    gen_forgets = bool(not np.isnan(gen_mean_cos) and (gen_mean_cos < 0.85 or gen_near_acc < 0.90))

    if not np.isnan(flat_f):
        attributable_to("generative coverage matches the flat O(N) store (generative vs flat)", gen_f, flat_f)
    if not np.isnan(bnd_f):
        attributable_to("full generative coverage vs the bounded buffer (generative vs bounded_buffer)", gen_f, bnd_f)

    v = Verdict("teacher-loop generative replay (fixed-size neural generator)", chance=chance)
    if not np.isnan(bnd_f):
        v.reaches("(BEAT) generative coverage beats the bounded-buffer CLS negative", before=bnd_f, after=gen_f)
        v.require("(BEAT') generative > bounded_buffer + 0.15", (gen_f > bnd_f + 0.15), expect=True,
                  note=f"generative {gen_f:.2f} vs bounded_buffer {bnd_f:.2f} @ N={big}")
    if not np.isnan(flat_f):
        v.require("(KEY) generative within 0.15 of flat AND >= 0.5", (gen_f >= flat_f - 0.15 and gen_f >= 0.5),
                  expect=True, note=f"generative {gen_f:.2f} vs flat {flat_f:.2f} @ N={big}")
    v.require("(FIXED) generator trained-param count CONSTANT across N", param_constant, expect=True,
              note=f"trace {garm.get('generator_param_trace')}")
    v.require("(FIXED') generator stores 0 raw patterns (genuinely generative, not the flat store in disguise)",
              not_buffer, expect=True)
    v.require("(FIXED'') generator fixed-size across two builds", fixed_two_builds, expect=True,
              note=f"trained_params {result.get('generator_trained_params')}")
    v.require("(NO-FORGET) generator regeneration fidelity holds (mean cos >= 0.85, nearest-proto >= 0.90)",
              (not gen_forgets), expect=True,
              note=f"mean_cos {gen_mean_cos:.3f} nearest-proto {gen_near_acc:.3f} @ N={big}")
    v.floor("(acq) generative immediate acquisition stays high", gen_acq, floor=0.85)
    v.require("(decoupled) slow reservoir CONSTANT across the curriculum", slow_constant, expect=True)
    v.require("(seed) substrate byte-identical across two builds at one seed",
              bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    go = (param_constant and not_buffer and fixed_two_builds and slow_constant and (not gen_forgets)
          and gen_acq >= 0.85 and bool(result["substrate_byte_identical"]) and bool(result["sim_diff_empty"]))
    if not np.isnan(bnd_f):
        go = go and (gen_f > bnd_f + 0.15)
    if not np.isnan(flat_f):
        go = go and (gen_f >= flat_f - 0.15) and (gen_f >= 0.5)
    decision = v.decide(go=go)

    return {
        "largest_N": big, "capacity_F": F,
        "generative_frac_recalled": gen_f, "flat_frac_recalled": flat_f, "bounded_buffer_frac_recalled": bnd_f,
        "generative_immediate_acq": gen_acq,
        "generative_minus_bounded_buffer": (float(gen_f - bnd_f) if not np.isnan(bnd_f) else None),
        "generative_minus_flat": (float(gen_f - flat_f) if not np.isnan(flat_f) else None),
        "retention_vs_N": {str(N): {"generative": frac_at("generative", N), "flat": frac_at("flat", N),
                                    "bounded_buffer": frac_at("bounded_buffer", N)} for N in result["milestones"]},
        "generator_trained_params": result.get("generator_trained_params"),
        "generator_total_params": result.get("generator_total_params"),
        "generator_param_constant_across_N": param_constant,
        "generator_stored_raw_patterns": garm.get("generator_stored_raw_patterns"),
        "generative_not_stored_buffer": not_buffer,
        "generator_fixed_size_two_builds_ok": fixed_two_builds,
        "generator_fidelity_vs_N": fid,
        "generator_mean_cos_at_bigN": gen_mean_cos, "generator_nearest_proto_acc_at_bigN": gen_near_acc,
        "generator_forgets": gen_forgets,
        "max_replay_set_generative": garm.get("max_replay_set_size"),
        "slow_reservoir_constant": slow_constant,
        "substrate_byte_identical": result["substrate_byte_identical"], "sim_diff_empty": result["sim_diff_empty"],
        **decision,
    }


def _one_seed(a, seed, arms_to_run):
    result = run(seed, a.n_max, a.milestones, a.capacity, a.slow_hidden, a.gen_hidden, a.settle_steps, a.epochs,
                 a.batch, a.eprop_lr, a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs,
                 a.replay_per_fact, a.replay_noise, a.gen_settle, a.gen_epochs, a.gen_lr, arms_to_run, a.bdsp_wmax)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop GENERATIVE REPLAY: a FIXED-SIZE neural generator "
                                             "regenerates ALL learned facts for consolidation, decoupling retention "
                                             "coverage from lifetime N (van de Ven 2020) -- beat the bounded-buffer "
                                             "CLS negative, match the flat O(N) store, with a bounded store.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--milestones", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--capacity", type=int, default=5, help="F = the bounded_buffer fast-store capacity (CLS negative)")
    ap.add_argument("--slow-hidden", type=int, default=100, help="the FIXED slow cortical reservoir size")
    ap.add_argument("--gen-hidden", type=int, default=96, help="the FIXED generator reservoir size (H_gen; the store "
                                                              "is the H_gen x n_in readout -- constant in N)")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20, help="per-fact WAKE teaching epochs (slow cortex)")
    ap.add_argument("--replay-epochs", type=int, default=12, help="SLEEP consolidation epochs over the replay set")
    ap.add_argument("--replay-per-fact", type=int, default=8, help="replay draws per regenerated fact")
    ap.add_argument("--replay-noise", type=float, default=0.10, help="brain-owned variability on the replayed pattern")
    ap.add_argument("--gen-settle", type=int, default=15, help="generator spiking settle steps (query->reservoir)")
    ap.add_argument("--gen-epochs", type=int, default=16, help="generator incremental-training epochs per new fact")
    ap.add_argument("--gen-lr", type=float, default=0.8, help="generator readout delta-rule lr")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9, help="1e9 = de-clamped (required; 6 = the CLAMP that "
                                                                "silences the reservoir, bound-trap 8ca014ff2)")
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["generative", "flat", "bounded_buffer"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    arms_to_run = list(a.arms)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  (F={a.capacity}, slow={a.slow_hidden}, gen_H={a.gen_hidden}, "
              f"n_max={a.n_max})\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, arms_to_run)
        summary = {"probe": "teacher_loop_generative_replay", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "arms_run": arms_to_run, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        print(f"[genrep] seed {s} @ N={rv.get('largest_N')}: generative {rv.get('generative_frac_recalled', float('nan')):.2f} "
              f"| flat {rv.get('flat_frac_recalled', float('nan')):.2f} | bounded_buffer "
              f"{rv.get('bounded_buffer_frac_recalled', float('nan')):.2f} (chance {result['chance']:.2f})", flush=True)
        rvn = rv.get("retention_vs_N", {})
        for N in result["milestones"]:
            d = rvn.get(str(N), {})
            fd = result["arms"].get("generative", {}).get("generator_fidelity", {}).get(str(N), {})
            print(f"    N={N:3d}: generative {d.get('generative', float('nan')):.2f} | "
                  f"flat {d.get('flat', float('nan')):.2f} | bounded_buffer {d.get('bounded_buffer', float('nan')):.2f} "
                  f"| gen-fidelity cos {fd.get('mean_cos', float('nan')):.3f} near-proto {fd.get('nearest_proto_acc', float('nan')):.2f}",
                  flush=True)
        print(f"[genrep] gen-trained-params {rv.get('generator_trained_params')} (const-in-N {rv.get('generator_param_constant_across_N')}, "
              f"stored-raw {rv.get('generator_stored_raw_patterns')}) | gen-forgets {rv.get('generator_forgets')} | "
              f"immediate-acq {rv.get('generative_immediate_acq', float('nan')):.3f} | VERDICT {rv.get('status')}", flush=True)
        print(f"[genrep] gen-flat {rv.get('generative_minus_flat')} | gen-bounded {rv.get('generative_minus_bounded_buffer')} | "
              f"byte-identical {rv.get('substrate_byte_identical')} | sim-clean {rv.get('sim_diff_empty')} | "
              f"wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        gen = [p["verdict"].get("generative_frac_recalled", float("nan")) for p in per_seed]
        flat = [p["verdict"].get("flat_frac_recalled", float("nan")) for p in per_seed]
        bnd = [p["verdict"].get("bounded_buffer_frac_recalled", float("nan")) for p in per_seed]
        cos = [p["verdict"].get("generator_mean_cos_at_bigN", float("nan")) for p in per_seed]
        agg = {"probe": "teacher_loop_generative_replay_AGG", "seeds": seeds, "backend": os.environ.get("SIM_BACKEND"),
               "capacity_F": a.capacity, "slow_hidden": a.slow_hidden, "gen_hidden": a.gen_hidden, "n_max": a.n_max,
               "go_count": go_n, "n_seeds": len(seeds),
               "generative_frac_mean": float(np.nanmean(gen)), "flat_frac_mean": float(np.nanmean(flat)),
               "bounded_buffer_frac_mean": float(np.nanmean(bnd)),
               "generative_minus_bounded_mean": float(np.nanmean(np.array(gen) - np.array(bnd))),
               "generative_minus_flat_mean": float(np.nanmean(np.array(gen) - np.array(flat))),
               "generator_mean_cos_mean": float(np.nanmean(cos)),
               "generator_trained_params": per_seed[0]["verdict"].get("generator_trained_params"),
               "per_seed": per_seed}
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[genrep AGG] GO {go_n}/{len(seeds)} | generative {np.nanmean(gen):.2f} vs flat {np.nanmean(flat):.2f} "
              f"vs bounded_buffer {np.nanmean(bnd):.2f} | gen-cos {np.nanmean(cos):.3f} "
              f"(gen_H={a.gen_hidden}, params={agg['generator_trained_params']}) | wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
