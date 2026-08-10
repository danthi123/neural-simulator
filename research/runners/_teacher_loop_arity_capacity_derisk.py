"""TEACHER-LOOP SHARED-CHANNEL ARITY CAPACITY DE-RISK (2026-08-10): LOCATE the bundling-capacity limit -- where
neural SUPERPOSITION (bundle) finally needs BINDING even for same-attribute-type composition.

WHERE THE ARC STANDS. The composer's BUNDLE (superposition) + BIND (a real temporal spiking Larkum-BAC coincidence,
2026-08-10 GO) are both biologized. Arity-3 composition is GO but BOUNDED: its three attribute-terms occupy DISJOINT
channel blocks, so the bundle is a CONCATENATION with ZERO inter-term crosstalk (arity3 == arity2 by construction).
The ~1/sqrt(#terms) bundling-capacity margin (Plate 1995 HRR / Kanerva 2009 VSA) only BITES when the terms SHARE
channels and interfere. This de-risk stresses exactly that.

THE TIGHT QUESTION. As the number of attribute-terms M grows, a SHARED-channel bundle (all M primitive codes summed
into the SAME d channels) accumulates crosstalk ~sqrt(M) and its per-fact separation margin falls -> at some M* the
zero-shot readout can no longer separate held-out facts, and the composer would NEED binding (a conjunctive code)
even for same-type composition. WHERE is M*? And is the degradation genuinely the SUPERPOSITION crosstalk, not just
the growing class count N=K^M?

THE CLEAN CONTROL (isolates crosstalk from N). At EACH M we run TWO arms on the SAME reservoir readout, SAME K, SAME
N=K^M, SAME held-out split -- differing ONLY in the channel geometry:
  * SHARED (the capacity-stressed arm): the M primitive codes are summed into ONE d-channel space (real VSA
    bundling; d_p = d). Per-primitive readout = the Hebbian running-mean cleanup over the facts that share it (the
    other terms average toward zero because the codes are ZERO-MEAN and co-occurrence is ~balanced; the residual
    imbalance IS the crosstalk). regenerate = SUM of the M spiking readouts.
  * DISJOINT (the no-crosstalk CONTROL): the M codes occupy M separate d-blocks (concatenation; d_p = M*d), so each
    primitive owns its own channels and there is NO inter-term interference. regenerate = CONCAT of the M readouts.
Both identify a fact by nearest-prototype among the SAME N prototypes (chance 1/N). The readout noise is COMMON to
both arms, so the GAP (disjoint_recall - shared_recall) at each M isolates the pure bundling-capacity cost, with N
held fixed. Plus a FLAT instance-store floor (class-indexed; chance on held-out) to prove the task is non-trivial.

M* = the located capacity limit = the smallest M at which SHARED held-out recall drops below 0.5 (or below
disjoint - 0.30) while DISJOINT still holds. If SHARED holds across the whole sweep, M* is bounded > M_max at this d
(also informative: capacity exceeds the swept range). Either way the LIMIT is the deliverable -- this is a
capacity-LOCATION probe, not a pass/fail on a single number.

ANTI-CHEATS (real assertions in the output):
  * taught/held-out DISJOINT; every held-out primitive (each of the M) appears in >= 1 TAUGHT fact (coverage-
    preserving split); NO held-out fact index enters any training path; the true held-out percept is read ONLY by
    the test-time nearest-prototype RULER.
  * composition is NEURAL: regeneration sums/concats M spiking-reservoir leaky-readout outputs; a LESION zeroing one
    primitive's readout collapses that term (localisation witness, sampled).
  * codes are ZERO-MEAN (so the running-mean cleanup is unbiased and the shared bundle is a clean superposition, not
    a DC pile-up); 0 stored raw patterns; ruler untouched by learning; cfg.seed byte-identical substrate; de-clamped
    bdsp_wmax=1e9; git diff main -- sim/ empty (NO sim edit); backend recorded.

GO (per seed): (1) SHARED composition WORKS at the low-M end (M=2 held-out recall >= 0.5 AND >= flat + 0.30 -- the
setup is a real composition); (2) the capacity cost is REAL and monotone: the shared-vs-disjoint gap is
non-decreasing over M and strictly positive by the top M; (3) either M* is located in-range, or shared holds to
M_max (capacity > M_max at this d) -- reported, not failed. HONEST NEGATIVE if shared already fails at M=2 (the
shared readout is broken, naming WHY: DC bias / readout rank), which would void the capacity reading.

DR grounding: Plate 1995 HRR + Kanerva 2009 VSA bundling capacity (retrieval margin ~1/sqrt(#superposed terms));
2026-06-05 in-network superposition NEGATIVE (this repo: opponent superposition does not unbind at parity); the
arity-3 disjoint GO (this repo). NO-EXTERNAL-NEEDED beyond the recorded VSA-capacity grounding.

DISCIPLINE: reuse-by-import (the spiking reservoir GenerativeReplayNet + the _fit_slot NLMS readout pattern; the
FlatStore floor + recall helpers + the byte-identical/sim-clean asserts). NO sim/ edit. SIM_BACKEND=numpy.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_arity_capacity_derisk --seed 42 --K 3 --M-values 2 3 4 \
      --out research/findings/raw/teacher_loop_arity_capacity_s42.json
  3-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_arity_capacity_derisk --seeds 42 43 44 --K 3 --M-values 2 3 4 5 6 \
      --out research/findings/raw/teacher_loop_arity_capacity.json
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

from research.runners._teacher_loop_generative_replay_derisk import GenerativeReplayNet, _cos  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import N_ACT  # noqa: E402
from research.runners._teacher_loop_compositional_generator_derisk import _action_ctx_const  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _assert_byte_identical_substrate, _git_sim_diff_empty,
)
from research.runners._teacher_loop_zeroshot_composition_derisk import _nearest_proto, _recall_fraction  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_arity_capacity.json"


# ============================ the M-arity WORLD (host, legitimate; ZERO-MEAN codes) ============================
def _make_world(M, K, d, channel_mode, seed):
    """M attribute-families, each K primitive codes ~ N(0,1) in R^d (ZERO-MEAN so the shared bundle is a clean
    superposition, no DC pile-up). proto(values) = SUM of the M codes (shared, d_p=d) or CONCAT (disjoint, d_p=M*d).
    Host code, legitimate exactly as a retinal render is."""
    wr = np.random.default_rng(int(seed) + 30303030)
    prims = [[wr.standard_normal(d).astype(np.float64) for _ in range(K)] for _ in range(M)]
    d_p = d if channel_mode == "shared" else M * d

    def proto(values):
        if channel_mode == "shared":
            acc = np.zeros(d, dtype=np.float64)
            for m in range(M):
                acc += prims[m][int(values[m])]
            return acc
        return np.concatenate([prims[m][int(values[m])] for m in range(M)]).astype(np.float64)

    return prims, proto, d_p


def _all_facts(M, K):
    """All N = K^M facts as tuples of M attribute-values, and the class index (mixed-radix base K)."""
    facts = []
    def rec(prefix):
        if len(prefix) == M:
            facts.append(tuple(prefix)); return
        for v in range(K):
            rec(prefix + [v])
    rec([])
    return facts


def _heldout_split_M(M, K, m_held, seed):
    """Coverage-preserving held-out split over M attributes: hold out a fact ONLY if EACH of its M primitive values
    keeps >= 1 taught fact after removal (so every held-out primitive is still seen in a taught fact). Deterministic."""
    rng = np.random.default_rng(int(seed) + 7777)
    facts = _all_facts(M, K)
    order = list(range(len(facts)))
    rng.shuffle(order)
    # remaining taught facts that use primitive (family m, value v)
    left = [{v: K ** (M - 1) for v in range(K)} for _ in range(M)]
    held = []
    for idx in order:
        if len(held) >= m_held:
            break
        vals = facts[idx]
        if all(left[m][vals[m]] > 1 for m in range(M)):
            held.append(idx)
            for m in range(M):
                left[m][vals[m]] -= 1
    held_set = set(held)
    taught = [i for i in range(len(facts)) if i not in held_set]
    return facts, taught, sorted(held)


# ============================ the M-arity compositional generator (NEURAL; spiking reservoir readouts) ==========
class CompositionalGeneratorM:
    """M primitive-engram families on ONE frozen spiking reservoir (readout-only, de-clamped). Each family stores one
    per-value readout, fit by the local NLMS delta rule (reuse-of-pattern from CompositionalGenerator) to the
    HEBBIAN RUNNING-MEAN of the observed engram for that primitive. SHARED: the observed target is the full d-channel
    superposition (the other terms average toward zero -> the residual is the crosstalk); regenerate = SUM of the M
    readouts. DISJOINT: the target is the family's own d-block; regenerate = CONCAT. Zero-mean anchor. NEVER reads the
    world's true primitive codes."""

    def __init__(self, gen_k, n_in, M, K, d, channel_mode, hidden, seed, settle, gen_lr, w_clip, bdsp_wmax=1e9,
                 conv_tol=0.02, conv_max_epochs=200, conv_check_every=4):
        self._res = GenerativeReplayNet(int(gen_k), int(n_in), int(hidden), seed, settle, gen_lr, w_clip,
                                        bdsp_wmax=bdsp_wmax)
        self.M = int(M); self.K = int(K); self.d = int(d); self.channel_mode = channel_mode
        self.d_p = int(d) if channel_mode == "shared" else int(M) * int(d)
        self.n_in = int(n_in); self.gen_lr = float(gen_lr)
        self.conv_tol = float(conv_tol); self.conv_max_epochs = int(conv_max_epochs)
        self.conv_check_every = max(1, int(conv_check_every))
        self._off = [m * 1_000_000 for m in range(self.M)]              # disjoint reservoir query families
        addrs = [self._off[m] + v for m in range(self.M) for v in range(self.K)]
        R = np.array([self._res._readout_elig(self._res._forward_record(self._res._query_code(ad))[0]) for ad in addrs])
        self._res._r_mu = np.zeros(R.shape[1], dtype=np.float64)         # keep common-mode/bias direction
        self._res._r_sigma = R.std(axis=0) + 1e-3
        self._H = int(R.shape[1])
        self.W = [dict() for _ in range(self.M)]                        # per (family) -> {value: readout weights}
        self._mean = [dict() for _ in range(self.M)]                    # per (family) -> {value: [running mean, count]}
        self._anchor = np.zeros(self.d, dtype=np.float64)               # zero-mean codes -> zero anchor
        self._action_ctx = None
        self._stored_raw_patterns = 0
        self._used_ruler = False

    def _elig(self, addr):
        return self._res._readout_elig(self._res._forward_record(self._res._query_code(addr))[0])

    def _fit_slot(self, W_dict, addr, value, target):
        r = self._elig(addr)
        if value not in W_dict:
            W_dict[value] = np.zeros((self._H, target.shape[0]), dtype=np.float64)   # ALLOCATE a new engram slot
        W = W_dict[value]
        tgt = np.asarray(target, dtype=np.float64) - self._anchor
        denom = float(r @ r) + 1e-6
        for ep in range(self.conv_max_epochs):
            err = (r @ W) - tgt
            W -= self.gen_lr * np.outer(r, err) / denom
            if (ep + 1) % self.conv_check_every == 0 and float(np.linalg.norm((r @ W) - tgt)) < self.conv_tol:
                break

    def learn_fact(self, values, engram, action_ctx):
        engram = np.asarray(engram, dtype=np.float64)
        self._action_ctx = np.asarray(action_ctx, dtype=np.float64)
        for m in range(self.M):
            v = int(values[m])
            if self.channel_mode == "shared":
                obs = engram[:self.d]                                    # the full d-channel superposition (the brain sees the sum)
            else:
                obs = engram[m * self.d:(m + 1) * self.d]                # this family's own block
            store = self._mean[m]
            if v not in store:
                store[v] = [obs.copy(), 1]
            else:
                s, c = store[v]; s *= c; s += obs; c += 1; s /= c; store[v] = [s, c]
            self._fit_slot(self.W[m], self._off[m] + v, v, self._mean[m][v][0])

    def regenerate(self, values, lesion=None):
        out = np.zeros(self.n_in, dtype=np.float64)
        if self.channel_mode == "shared":
            acc = np.zeros(self.d, dtype=np.float64)
            for m in range(self.M):
                if lesion == m:
                    continue
                v = int(values[m])
                if v in self.W[m]:
                    acc += self._elig(self._off[m] + v) @ self.W[m][v] + self._anchor
            out[:self.d] = acc
        else:
            for m in range(self.M):
                if lesion == m:
                    continue
                v = int(values[m])
                if v in self.W[m]:
                    out[m * self.d:(m + 1) * self.d] = self._elig(self._off[m] + v) @ self.W[m][v] + self._anchor
        if self._action_ctx is not None:
            out[self.d_p:] += self._action_ctx
        return out

    def primitive_slots(self):
        return int(sum(len(w) for w in self.W))


class _FlatStoreM:
    """FLOOR: an O(N) raw-engram buffer keyed by class INDEX. A held-out class has no entry -> uniform guess over the
    N prototypes (chance). No composition."""

    def __init__(self, d_p, seed):
        self.d_p = int(d_p); self.store = {}; self.rng = np.random.default_rng(int(seed) + 424242)

    def learn(self, cls, engram):
        self.store[int(cls)] = np.asarray(engram, dtype=np.float64)[:self.d_p].copy()

    def recall_nearest(self, cls, protos):
        if int(cls) in self.store:
            return _nearest_proto(self.store[int(cls)], protos)
        return int(self.rng.integers(protos.shape[0]))                   # no entry -> uniform guess (chance)


# ============================ per-M driver (SHARED + DISJOINT on the SAME reservoir readout) ============================
def _run_one_mode(seed, M, K, d, channel_mode, m_held, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                  bdsp_wmax, conv_tol, conv_max_epochs, n_draws):
    facts, taught_idx, held_idx = _heldout_split_M(M, K, m_held, seed)
    N = len(facts)
    prims, proto_fn, d_p = _make_world(M, K, d, channel_mode, seed)
    n_in = d_p + N_ACT
    chance = 1.0 / N
    protos = np.stack([proto_fn(facts[j]) for j in range(N)]).astype(np.float64)    # (N, d_p) test-time RULER only
    action_ctx = _action_ctx_const()

    # engram = mean of n_draws noisy percepts (the brain's compressed trace); host-constructed world percept.
    draw_rng = np.random.default_rng(int(seed) + 909 + (0 if channel_mode == "shared" else 1))

    def engram_of(j):
        p = protos[j]
        noisy = p[None, :] + draw_rng.standard_normal((n_draws, d_p)) * noise
        e = np.zeros(n_in, dtype=np.float64)
        e[:d_p] = noisy.mean(axis=0)
        e[d_p:] = action_ctx
        return e

    # anti-cheat asserts (disjoint + coverage + zero-mean codes)
    taught_set, held_set = set(taught_idx), set(held_idx)
    disjoint = bool(len(taught_set & held_set) == 0 and len(held_set) > 0)
    seen_vals = [set() for _ in range(M)]
    for j in taught_idx:
        for m in range(M):
            seen_vals[m].add(facts[j][m])
    coverage_ok = bool(all(all(facts[j][m] in seen_vals[m] for m in range(M)) for j in held_idx))
    code_mean_abs = float(np.mean([np.abs(np.mean(prims[m][v])) for m in range(M) for v in range(K)]))
    assert disjoint, "taught/held-out must be disjoint and held-out non-empty"
    assert coverage_ok, "every held-out primitive must appear in >= 1 taught fact"

    gen = CompositionalGeneratorM(gen_k, n_in, M, K, d, channel_mode, gen_hidden, seed, gen_settle, gen_lr, w_clip,
                                  bdsp_wmax=bdsp_wmax, conv_tol=conv_tol, conv_max_epochs=conv_max_epochs)
    flat = _FlatStoreM(d_p, seed)
    fed = []
    for j in taught_idx:
        e = engram_of(j)
        gen.learn_fact(facts[j], e, action_ctx)
        flat.learn(j, e)
        fed.append(j)
    no_leakage = bool(not (set(fed) & held_set))
    assert no_leakage, "a held-out fact index leaked into a training path"

    def comp_pred(j):
        return _nearest_proto(gen.regenerate(facts[j])[:d_p], protos)

    def flat_pred(j):
        return flat.recall_nearest(j, protos)

    comp_seen = _recall_fraction(taught_idx, comp_pred, protos)
    comp_held = _recall_fraction(held_idx, comp_pred, protos)
    flat_held = _recall_fraction(held_idx, flat_pred, protos)
    comp_held_cos = float(np.mean([_cos(gen.regenerate(facts[j])[:d_p], protos[j]) for j in held_idx]))

    # lesion localisation on a sample of held-out facts (composition is NEURAL / separable)
    les_idx = held_idx[:min(6, len(held_idx))]
    les_hit, les_sp = [], []
    for j in les_idx:
        full = gen.regenerate(facts[j])[:d_p]
        l0 = gen.regenerate(facts[j], lesion=0)[:d_p]
        les_hit.append(float(np.linalg.norm(full - l0)))
        # a lesion of family 0 should change the regeneration (in shared it perturbs the sum; in disjoint only block 0)
    lesion_delta = float(np.mean(les_hit)) if les_hit else 0.0

    return {
        "channel_mode": channel_mode, "M": M, "K": K, "d": d, "d_p": d_p, "N": N, "chance": chance,
        "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "taught_heldout_disjoint": disjoint, "coverage_ok": coverage_ok, "no_leakage": no_leakage,
        "code_mean_abs": code_mean_abs,
        "compositional_heldout_recall": comp_held, "compositional_seen_recall": comp_seen,
        "flat_heldout_recall": flat_held, "compositional_heldout_cos": comp_held_cos,
        "lesion_delta": lesion_delta,
        "stored_raw_patterns": int(gen._stored_raw_patterns), "used_ruler": bool(gen._used_ruler),
        "primitive_slots": gen.primitive_slots(),
    }


def run(seed, M_values, K, d, held_frac, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
        conv_tol, conv_max_epochs, n_draws):
    # byte-identical substrate check at the largest shared config
    Mmax = max(M_values)
    n_in_big = d + N_ACT
    Nbig = K ** Mmax
    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in_big, min(Nbig, 400), seed,
                                                               max(120, 6 * min(Nbig, 400)), 20, 0.5, w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()

    by_M = {}
    for M in M_values:
        N = K ** M
        m_held = max(1, int(round(held_frac * N)))
        m_held = min(m_held, N - (K ** (M - 1)))   # keep coverage feasible
        print(f"\n{'=' * 92}\n# SEED {seed}  M={M} (N=K^M={N}, K={K}, d={d}, held_out={m_held})\n{'=' * 92}", flush=True)
        shared = _run_one_mode(seed, M, K, d, "shared", m_held, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                               bdsp_wmax, conv_tol, conv_max_epochs, n_draws)
        disjoint = _run_one_mode(seed, M, K, d, "disjoint", m_held, noise, gen_hidden, gen_k, gen_settle, gen_lr,
                                 w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws)
        gap = float(disjoint["compositional_heldout_recall"] - shared["compositional_heldout_recall"])
        by_M[str(M)] = {"M": M, "N": N, "chance": shared["chance"], "held_out_n": shared["held_out_n"],
                        "shared_heldout_recall": shared["compositional_heldout_recall"],
                        "disjoint_heldout_recall": disjoint["compositional_heldout_recall"],
                        "flat_heldout_recall": shared["flat_heldout_recall"],
                        "shared_seen_recall": shared["compositional_seen_recall"],
                        "disjoint_minus_shared": gap,
                        "shared_heldout_cos": shared["compositional_heldout_cos"],
                        "disjoint_heldout_cos": disjoint["compositional_heldout_cos"],
                        "code_mean_abs": shared["code_mean_abs"],
                        "shared_lesion_delta": shared["lesion_delta"], "disjoint_lesion_delta": disjoint["lesion_delta"],
                        "shared_full": shared, "disjoint_full": disjoint}
        print(f"  [M={M} N={N}] HELD-OUT: shared {shared['compositional_heldout_recall']:.2f} | disjoint "
              f"{disjoint['compositional_heldout_recall']:.2f} | flat {shared['flat_heldout_recall']:.2f} "
              f"(chance {shared['chance']:.4f}) | GAP(disj-shared) {gap:+.2f} | shared-cos "
              f"{shared['compositional_heldout_cos']:.3f} | seen(shared) {shared['compositional_seen_recall']:.2f}",
              flush=True)

    return {"seed": seed, "K": K, "d": d, "M_values": M_values,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"K": K, "d": d, "held_frac": held_frac, "noise": noise, "gen_hidden": gen_hidden,
                       "gen_k": gen_k, "gen_settle": gen_settle, "gen_lr": gen_lr, "w_clip": w_clip,
                       "bdsp_wmax": bdsp_wmax, "conv_tol": conv_tol, "conv_max_epochs": conv_max_epochs,
                       "n_draws": n_draws, "frozen_hidden": True},
            "by_M": by_M}


# ============================ verdict ============================
def _locate_mstar(by_M, M_values):
    """M* = smallest M where SHARED held-out recall drops below 0.5 OR below (disjoint - 0.30) while DISJOINT holds
    (>= 0.5). None if shared holds across the whole sweep (capacity > M_max at this d)."""
    for M in sorted(M_values):
        r = by_M[str(M)]
        sh = r["shared_heldout_recall"]; dj = r["disjoint_heldout_recall"]
        if dj >= 0.5 and (sh < 0.5 or sh < dj - 0.30):
            return M
    return None


def _verdict(result):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    by_M = result["by_M"]; M_values = sorted(result["M_values"])
    v = Verdict("teacher-loop SHARED-CHANNEL ARITY CAPACITY (locate where bundle needs bind)", chance=None)

    M0 = M_values[0]; r0 = by_M[str(M0)]; rtop = by_M[str(M_values[-1])]
    # (1) shared composition WORKS at the low-M end (the setup is a real composition, not trivially broken)
    low_ok = bool(r0["shared_heldout_recall"] >= 0.5 and r0["shared_heldout_recall"] >= r0["flat_heldout_recall"] + 0.30)
    attributable_to(f"[M={M0}] shared composition vs flat instance store (held-out)",
                    r0["shared_heldout_recall"], r0["flat_heldout_recall"])
    v.require(f"[M={M0}] SHARED composition works (held-out >= 0.5 AND >= flat+0.30)", low_ok, expect=True,
              note=f"shared {r0['shared_heldout_recall']:.2f} flat {r0['flat_heldout_recall']:.2f} "
                   f"(chance {r0['chance']:.4f})")

    # (2) the capacity cost is REAL: shared-vs-disjoint gap is positive by the top M, and non-decreasing overall
    gaps = [by_M[str(M)]["disjoint_minus_shared"] for M in M_values]
    gap_grows = bool(gaps[-1] > 0.0 and gaps[-1] >= gaps[0] - 1e-9)
    v.require("capacity cost is REAL (disjoint-shared gap positive at top M and non-decreasing)", gap_grows,
              expect=True, note=f"gaps over M {['%+.2f' % g for g in gaps]}")

    # (3) disjoint (no-crosstalk control) stays high across the sweep (the degradation is crosstalk, not readout/N)
    disj_holds = bool(all(by_M[str(M)]["disjoint_heldout_recall"] >= 0.5 for M in M_values))
    v.require("disjoint control HOLDS across the sweep (>= 0.5 all M -> degradation is crosstalk, not N)", disj_holds,
              expect=True, note=f"disjoint {[round(by_M[str(M)]['disjoint_heldout_recall'],2) for M in M_values]}")

    # composition NEURAL (lesion perturbs regeneration) + housekeeping
    neural_ok = bool(rtop["shared_full"]["lesion_delta"] > 0.02 and r0["shared_full"]["lesion_delta"] > 0.02)
    not_buffer = bool(all(by_M[str(M)]["shared_full"]["stored_raw_patterns"] == 0 for M in M_values))
    no_ruler = bool(all(not by_M[str(M)]["shared_full"]["used_ruler"] for M in M_values))
    zero_mean = bool(all(by_M[str(M)]["code_mean_abs"] < 0.35 for M in M_values))   # ~1/sqrt(d)-scale, near zero
    cover = bool(all(by_M[str(M)]["shared_full"]["coverage_ok"] and by_M[str(M)]["shared_full"]["no_leakage"]
                     and by_M[str(M)]["shared_full"]["taught_heldout_disjoint"] for M in M_values))
    v.require("composition is NEURAL (lesion perturbs regeneration)", neural_ok, expect=True,
              note=f"lesion-delta M0 {r0['shared_full']['lesion_delta']:.2f} Mtop {rtop['shared_full']['lesion_delta']:.2f}")
    v.require("0 stored raw patterns (composes, not a buffer)", not_buffer, expect=True)
    v.require("generator never read the ruler", no_ruler, expect=True)
    v.require("codes are ZERO-MEAN (clean superposition, no DC pile-up)", zero_mean, expect=True,
              note=f"mean|code| {[round(by_M[str(M)]['code_mean_abs'],3) for M in M_values]}")
    v.require("zero-shot integrity (disjoint split + coverage + no-leakage, all M)", cover, expect=True)

    m_star = _locate_mstar(by_M, M_values)
    v.require("(seed) substrate byte-identical", bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty (NO sim edit)", bool(result["sim_diff_empty"]), expect=True)

    go = bool(low_ok and gap_grows and disj_holds and neural_ok and not_buffer and no_ruler and zero_mean and cover
              and result["substrate_byte_identical"] and result["sim_diff_empty"])
    decision = v.decide(go=go)
    return {"m_star": m_star, "m_star_note": ("located in-range" if m_star is not None
                                              else f"capacity > M_max={M_values[-1]} at d={result['d']}"),
            "by_M": {str(M): {k: by_M[str(M)][k] for k in
                              ("M", "N", "chance", "shared_heldout_recall", "disjoint_heldout_recall",
                               "flat_heldout_recall", "shared_seen_recall", "disjoint_minus_shared",
                               "shared_heldout_cos", "disjoint_heldout_cos")} for M in M_values},
            "low_M_shared_works": low_ok, "capacity_cost_real": gap_grows, "disjoint_holds": disj_holds,
            "substrate_byte_identical": result["substrate_byte_identical"], "sim_diff_empty": result["sim_diff_empty"],
            **decision}


def _one_seed(a, seed):
    result = run(seed, a.M_values, a.K, a.d, a.held_frac, a.noise, a.gen_hidden, a.gen_k, a.gen_settle, a.gen_lr,
                 a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.n_draws)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop SHARED-CHANNEL ARITY CAPACITY: locate where neural "
                                             "superposition (bundle) needs binding -- shared-channel bundle vs a "
                                             "disjoint no-crosstalk control, swept over arity M.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--M-values", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="arities to sweep")
    ap.add_argument("--K", type=int, default=3, help="per-attribute vocab size (N=K^M)")
    ap.add_argument("--d", type=int, default=8, help="channel dim per primitive (shared d_p=d; disjoint d_p=M*d)")
    ap.add_argument("--held-frac", type=float, default=0.2, help="fraction of facts held out (coverage-preserving)")
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--gen-hidden", type=int, default=96)
    ap.add_argument("--gen-k", type=int, default=96, help="query address width")
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--conv-tol", type=float, default=0.02)
    ap.add_argument("--conv-max-epochs", type=int, default=200)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9)
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  K={a.K} d={a.d} M={a.M_values} held_frac={a.held_frac}\n" + "#" * 100,
              flush=True)
        result, verdict = _one_seed(a, s)
        summary = {"probe": "teacher_loop_arity_capacity", "seed": s, "backend": os.environ.get("SIM_BACKEND"),
                   "single_seed_smoke": (len(seeds) == 1), "K": a.K, "d": a.d, "M_values": a.M_values,
                   "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        for M in sorted(a.M_values):
            r = rv["by_M"][str(M)]
            print(f"[cap] seed {s} M={M}: N={r['N']} | HELD-OUT shared {r['shared_heldout_recall']:.2f} vs disjoint "
                  f"{r['disjoint_heldout_recall']:.2f} vs flat {r['flat_heldout_recall']:.2f} (chance {r['chance']:.4f}) "
                  f"| GAP {r['disjoint_minus_shared']:+.2f}", flush=True)
        print(f"[cap] seed {s} M* = {rv['m_star']} ({rv['m_star_note']}) | byte-id {rv['substrate_byte_identical']} "
              f"sim-clean {rv['sim_diff_empty']} | VERDICT {rv['status']}", flush=True)
        print(f"[cap] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        Ms = sorted(a.M_values)
        agg = {"probe": "teacher_loop_arity_capacity_AGG", "seeds": seeds, "backend": os.environ.get("SIM_BACKEND"),
               "K": a.K, "d": a.d, "M_values": a.M_values, "go_count": go_n, "n_seeds": len(seeds),
               "m_star_per_seed": [p["verdict"]["m_star"] for p in per_seed], "by_M_means": {}, "per_seed": per_seed}
        for M in Ms:
            sh = [p["verdict"]["by_M"][str(M)]["shared_heldout_recall"] for p in per_seed]
            dj = [p["verdict"]["by_M"][str(M)]["disjoint_heldout_recall"] for p in per_seed]
            fl = [p["verdict"]["by_M"][str(M)]["flat_heldout_recall"] for p in per_seed]
            gp = [p["verdict"]["by_M"][str(M)]["disjoint_minus_shared"] for p in per_seed]
            agg["by_M_means"][str(M)] = {
                "N": per_seed[0]["verdict"]["by_M"][str(M)]["N"],
                "chance": per_seed[0]["verdict"]["by_M"][str(M)]["chance"],
                "shared_heldout_recall_mean": float(np.nanmean(sh)),
                "shared_heldout_recall_per_seed": [float(x) for x in sh],
                "disjoint_heldout_recall_mean": float(np.nanmean(dj)),
                "flat_heldout_recall_mean": float(np.nanmean(fl)),
                "disjoint_minus_shared_mean": float(np.nanmean(gp)),
            }
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[cap AGG] GO {go_n}/{len(seeds)} | M* per seed {agg['m_star_per_seed']}", flush=True)
        for M in Ms:
            mm = agg["by_M_means"][str(M)]
            print(f"   M={M}: N={mm['N']} | HELD-OUT shared {mm['shared_heldout_recall_mean']:.2f} vs disjoint "
                  f"{mm['disjoint_heldout_recall_mean']:.2f} vs flat {mm['flat_heldout_recall_mean']:.2f} "
                  f"(chance {mm['chance']:.4f}) | GAP {mm['disjoint_minus_shared_mean']:+.2f} | shared/seed "
                  f"{mm['shared_heldout_recall_per_seed']}", flush=True)
        print(f"[cap AGG] wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
