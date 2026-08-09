"""TEACHER-LOOP COMPOSITIONAL-GENERATOR DE-RISK (2026-08-09): a GENUINELY SUB-LINEAR (compressing) generator.

WHERE THE ARC STANDS. Continual learning of N referent facts DECOMPOSED (all on main, adversarially verified):
RETENTION closed@N=20 (the non-forgetting generative_v2 matches the flat O(N) store, 0.958 vs 0.950, commit
0933fdb7a) but with a FIXED O(1) store that CAPS -- acquisition/fidelity SLIPS at N=100; STORAGE bounded but a
truly COMPRESSING generator OPEN; fixed-k prioritized replay NEGATIVE at scale (loses coverage like the bounded
buffer). THE UNIFYING INSIGHT: all three open axes fail for the SAME reason -- we treat N facts as N INDEPENDENT
things. Real experience SHARES STRUCTURE. If facts are COMPOSITIONAL (built from P<<N shared primitives), a
generator that stores the PRIMITIVES and COMPOSES needs O(P) store, O(P) replay, O(P) acquisition -- SUB-LINEAR in
N. That is the biological answer (learn regularities, not instances) and ties to the project compositionality core.

NO FREE LUNCH. Compression is IMPOSSIBLE without shared structure (random independent facts cannot be compressed).
So this de-risk uses a COMPOSITIONAL fact WORLD: N facts = a K1 x K2 grid of two attribute-vocabularies, so
P = K1 + K2 primitives << N = K1 * K2 facts (P grows as ~sqrt(N)). The host builds this compositional WORLD
(world/body is legitimate host code); the BRAIN learns the primitives + composes them via spikes/synapses.

STEP 1 -- the COMPOSITIONAL WORLD (host, legitimate). `CompositionalReferentEnv`: K1 attribute-A primitive
prototypes in [0,1]^d_a and K2 attribute-B primitives in [0,1]^d_b (fixed, seed-controlled). Fact (a,b)'s percept
prototype = concat(primA[a], primB[b]) on DISJOINT feature channels (distinct sensory attributes -- color-units vs
shape-units) -- a deterministic COMPOSITION of its two primitives. A presentation = clip(proto + noise*N(0,1)).
N = K1*K2 facts share exactly P = K1+K2 primitive codes.

STEP 2 -- the COMPOSITIONAL GENERATOR (brain-based; NEURAL composition; O(P) store). `CompositionalGenerator`:
ONE frozen spiking Izhikevich reservoir (reused from the a1-GO generator; de-clamped bdsp_wmax=1e9 so it spikes).
It stores ONE PRIMITIVE ENGRAM per DISTINCT attribute value -- a per-primitive linear leaky-readout weight-set W_p
(the primitive's synaptic trace) trained by the local NLMS delta rule on the reservoir's spike eligibility for that
primitive's query address. Teaching fact (a,b): if primitive a (or b) is UNSEEN, ALLOCATE + learn its engram
(the store GROWS by ONE slot); if already seen, REFINE it (running mean) -- NO new slot. So after N facts the store
holds exactly P = (distinct A) + (distinct B) primitive engrams = O(P), NOT O(N). REGENERATION of a fact =
NEURAL SUPERPOSITION: fire the reservoir with primitive-a's query -> a-engram outputs its block on the A channels;
fire it with primitive-b's query -> b-engram outputs its block on the B channels; the composed percept is the SUM
of the two spiking-readout population outputs (bundling). NOT a host algebra, NOT a lookup of N stored patterns.

STEP 3 -- THREE ARMS (same net build / seed / env / wake budget; the only difference is the SLEEP replay SOURCE):
  * compositional_gen = TREATMENT. The primitive-storing generator (O(P) slots) regenerates ALL N facts by
    NEURAL composition and e-props them into the slow cortex.
  * flat              = the O(N) target, MEASURED in-run (the flat CLS arm; unbounded raw engram buffer, all N).
  * noncompositional_gen = the CONTROL = the FIXED generative_v2 generator (O(1) store) on the SAME compositional
    facts, taught each fact as an INDEPENDENT class target. Answers "does compositionality help BEYOND a fixed
    generator?" -- and its store, unlike compositional's, does NOT expose the P primitives (it fits N targets).

TEST 2+ N VALUES so the store-vs-N (sub-linear) claim is MEASURABLE: N=16 (4x4, P=8) and N=36 (6x6, P=12).
For a K x K grid, P = 2K = 2*sqrt(N): store SLOTS ratio 12/8 = 1.5 = sqrt(36/16), while flat's O(N) store ratio is
36/16 = 2.25. THE KEY MEASUREMENTS: (A) retention of ALL N facts (compositional near flat, at BOTH N); (B) the
generator STORE SIZE at each N -- does it grow SUB-LINEARLY (~P~sqrt(N)) vs flat O(N)?

BONUS (composition is REAL, not memorization): ZERO-SHOT compositional generalization -- a FRESH generator taught
only a subset that OMITS some (a,b) facts (each held-out fact's a AND b still appear in OTHER trained facts) can
REGENERATE the held-out facts by composing already-learned primitives (high cos), while the fixed v2 generator
taught the SAME subset CANNOT (it never saw those class targets -> ~chance). Compose > memorize, made mechanical.

ANTI-CHEATS (each a REAL assertion in the output):
  * store GENUINELY SUB-LINEAR: compositional primitive-slot count == P == K1+K2 (NOT N) at each grid; store floats
    grow slower than flat by a clear margin; slot ratio across grids ~= sqrt(N ratio). Reported at each N.
  * composition NEURAL: regeneration is a SUM of two spiking-reservoir leaky-readout outputs (superposition/
    bundling); a LESION zeroing one primitive's engram breaks ONLY that primitive's block (its channels), proving
    each block is carried by a distinct neural engram, not a joint lookup.
  * NOT a stored-O(N)-pattern buffer: `_stored_raw_patterns == 0`; the generator holds P primitive engrams, never
    the N composed percepts.
  * facts GENUINELY compositional: P << N asserted (P/N <= 0.6; 0.50 at 4x4, 0.33 at 6x6).
  * consolidation NEVER reads the true primitive prototypes (the fidelity ruler): tripwire `_used_ruler == False`.
  * cfg.seed byte-identical substrate (NOT actual_seed_used); de-clamped bdsp_wmax=1e9 (the -6/+6 clamp silences
    the reservoir, bound-trap 8ca014ff2); git diff main -- sim/ empty; backend recorded.

GO (per seed, BOTH grids): compositional retention within 0.15 of flat AND >= 0.5 at each N; compositional store
grows SUB-LINEARLY (slot count == P << N, store-float ratio < flat ratio with margin); composition neural (lesion
localises); 0 stored raw patterns; ruler untouched; acquisition >= 0.85; byte-identical; sim/ clean. If the neural
generator cannot compose (retention fails / lesion does not localise) or the store is not sub-linear, HONEST
NEGATIVE naming WHY. DR grounding: van de Ven 2020 generative replay (doi:10.1038/s41467-020-17866-2);
compositional-generalization literature.

DISCIPLINE: reuse-by-import (the v2 non-forgetting generator + its arm for the noncompositional control; the CLS
flat arm + anti-cheat asserts; the scaling teacher machinery; ReferentEnv). NO sim/ edit. SIM_BACKEND=numpy.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_compositional_generator_derisk --seed 42 \
      --grids 4x4 6x6 --gen-hidden 96 --gen-k 64 \
      --out research/findings/raw/teacher_loop_compositional_generator_s42.json
  3-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_compositional_generator_derisk --seeds 42 43 44 \
      --grids 4x4 6x6 --gen-hidden 96 --gen-k 64 \
      --out research/findings/raw/teacher_loop_compositional_generator.json
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
# reuse-by-import: the v2 non-forgetting generator + its arm (the noncompositional CONTROL); the CLS flat arm +
# anti-cheat asserts; the scaling teacher machinery; ReferentEnv. NO sim/ edit.
from research.runners._teacher_loop_generative_replay_derisk import GenerativeReplayNet, _cos  # noqa: E402
from research.runners._teacher_loop_generative_replay_v2_derisk import (  # noqa: E402
    GenerativeReplayNetV2, _run_generative_arm_v2,
)
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _build_slow_cortex, _run_arm as _run_cls_arm,
    _assert_byte_identical_substrate, _git_sim_diff_empty,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_compositional_generator.json"


# ============================ STEP 1: the COMPOSITIONAL WORLD (host, legitimate) ============================
class CompositionalReferentEnv(ReferentEnv):
    """The world's compositional sensory render. K1 attribute-A primitive prototypes ([0,1]^d_a) + K2 attribute-B
    prototypes ([0,1]^d_b), fixed + seed-controlled. Fact (a,b)'s percept prototype = concat(primA[a], primB[b])
    on DISJOINT feature channels (distinct sensory attributes) -- a deterministic COMPOSITION of its two primitives.
    N = K1*K2 facts share exactly P = K1+K2 primitive codes. Host code is legitimate EXACTLY as a retinal render is:
    the brain reads this percept through its OWN learned weights."""

    def __init__(self, seed, K1, K2, d_a=8, d_b=8, noise=0.12):
        super().__init__(seed, d_p=int(d_a) + int(d_b), noise=noise)
        self.K1 = int(K1); self.K2 = int(K2); self.d_a = int(d_a); self.d_b = int(d_b)
        wr = np.random.default_rng(int(seed) + 20260809)                    # a dedicated WORLD RNG (deterministic)
        self.primA = [wr.random(self.d_a).astype(np.float64) for _ in range(self.K1)]
        self.primB = [wr.random(self.d_b).astype(np.float64) for _ in range(self.K2)]
        self.fact_attrs = {}                                                # referent -> (a, b)

    def register(self, referent, a, b):
        self.fact_attrs[referent] = (int(a), int(b))

    def proto(self, referent):
        if referent not in self.protos:
            a, b = self.fact_attrs[referent]
            self.protos[referent] = np.concatenate([self.primA[a], self.primB[b]]).astype(np.float64)  # COMPOSE
        return self.protos[referent]


def _grid_facts(K1, K2):
    """The N = K1*K2 referents of the grid, class index = a*K2 + b, and each referent's (a,b) attributes."""
    refs, attrs = [], []
    for a in range(K1):
        for b in range(K2):
            refs.append(f"r{a}_{b}"); attrs.append((a, b))
    return refs, attrs


def _make_comp_env(seed, K1, K2, d_a, d_b, noise, referents, attrs):
    env = CompositionalReferentEnv(seed, K1, K2, d_a=d_a, d_b=d_b, noise=noise)
    for r, (a, b) in zip(referents, attrs):
        env.register(r, a, b)
        env.proto(r)                                                        # instantiate the composed prototype
    env.rng = np.random.default_rng(seed + 101)                            # reset draw-stream => every arm sees SAME percepts
    return env


# ============================ STEP 2: the COMPOSITIONAL GENERATOR (neural; O(P) store) ============================
class CompositionalGenerator:
    """A PRIMITIVE-STORING, COMPOSING generator. ONE frozen spiking reservoir (reused from the a1-GO generator).
    Store = ONE PRIMITIVE ENGRAM per DISTINCT attribute value -- a per-primitive leaky-readout weight-set W_p
    (the primitive's synaptic trace) trained by the local NLMS delta rule on the reservoir's spike eligibility for
    that primitive's query address. The store GROWS by ONE slot per NEW primitive (O(P)=O(sqrt(N))), NOT O(N).
    Regeneration = NEURAL SUPERPOSITION of two spiking-readout population outputs (bundling), never a host algebra
    and never a lookup of the N composed patterns. `true_primA/primB` (the world prototypes) are NEVER read by the
    learning path -- tripwire `_used_ruler` stays False."""

    def __init__(self, gen_k, n_in, d_a, d_b, K1, K2, hidden, seed, settle, gen_lr, w_clip, bdsp_wmax=1e9,
                 conv_tol=0.02, conv_max_epochs=200, conv_check_every=4):
        # the SHARED frozen spiking reservoir + eligibility + query-code machinery (readout-only trained; de-clamped).
        self._res = GenerativeReplayNet(int(gen_k), int(n_in), int(hidden), seed, settle, gen_lr, w_clip,
                                        bdsp_wmax=bdsp_wmax)
        self.d_a = int(d_a); self.d_b = int(d_b); self.n_in = int(n_in); self.d_p = int(d_a) + int(d_b)
        self.K1 = int(K1); self.K2 = int(K2); self.gen_lr = float(gen_lr)
        self.conv_tol = float(conv_tol); self.conv_max_epochs = int(conv_max_epochs)
        self.conv_check_every = max(1, int(conv_check_every))
        self._A_off = 0; self._B_off = 1_000_000                            # disjoint reservoir query-address families
        # condition the readout eligibility (mu/sigma) over the ACTUAL primitive addresses (keep the bias dof: mu=0
        # on the common-mode as fit_query_norm does -- the anchor restores the absolute mean the bias-less readout
        # cannot reproduce). A single global RMS scale conditions the delta rule.
        addrs = [self._A_off + a for a in range(self.K1)] + [self._B_off + b for b in range(self.K2)]
        R = np.array([self._res._readout_elig(self._res._forward_record(self._res._query_code(ad))[0]) for ad in addrs])
        self._res._r_mu = np.zeros(R.shape[1], dtype=np.float64)            # keep the bias / common-mode direction
        self._res._r_sigma = R.std(axis=0) + 1e-3
        self._H = int(R.shape[1])
        self.Wa = {}; self.Wb = {}                                          # primitive engrams: a-> (H,d_a), b-> (H,d_b)
        self._A_mean = {}; self._B_mean = {}                               # running-mean observed block per primitive
        self._anchor_a = 0.5 * np.ones(self.d_a, dtype=np.float64)          # world constant (percepts in [0,1]); NOT per-fact
        self._anchor_b = 0.5 * np.ones(self.d_b, dtype=np.float64)
        self._action_ctx = None                                            # the world's constant action-context dims
        self._stored_raw_patterns = 0                                      # anti-cheat: NEVER stores raw composed patterns
        self._used_ruler = False                                           # anti-cheat: learning never reads true primitives
        self._slot_trace = []                                              # (facts_taught, n_slots) growth witness

    # --- reservoir eligibility for a query ADDRESS (spiking forward -> whitened readout feature) ---
    def _elig(self, addr):
        sp, _vv, _acts = self._res._forward_record(self._res._query_code(int(addr)))
        return self._res._readout_feature(sp)                              # (H,) whitened spike eligibility

    def _fit_slot(self, W_dict, addr, local, block_mean, anchor):
        """NLMS train-to-convergence of ONE primitive engram: map the reservoir eligibility for this primitive's
        query address -> its (anchor-centered) block mean. A rank-1 fixed-target regression on the frozen reservoir
        basis -> converges in a few steps. ALLOCATES the slot if new (the store grows by ONE)."""
        r = self._elig(addr)
        if local not in W_dict:
            W_dict[local] = np.zeros((self._H, block_mean.shape[0]), dtype=np.float64)   # ALLOCATE a new engram
        W = W_dict[local]
        tgt = np.asarray(block_mean, dtype=np.float64) - anchor
        denom = float(r @ r) + 1e-6
        for ep in range(self.conv_max_epochs):
            err = (r @ W) - tgt
            W -= self.gen_lr * np.outer(r, err) / denom
            if (ep + 1) % self.conv_check_every == 0 and float(np.linalg.norm((r @ W) - tgt)) < self.conv_tol:
                break

    def learn_fact(self, a, b, engram, action_ctx):
        """Teach fact (a,b) from its wake engram (the brain's compressed trace = mean of noisy draws). Update the
        running-mean block estimate for primitive a and b (a repeats across K2 facts -> the estimate DE-NOISES),
        then (re)fit each primitive engram. NEW primitive => a new slot (store grows); seen primitive => refine
        only. NEVER reads the world's true primitive prototypes."""
        a = int(a); b = int(b)
        engram = np.asarray(engram, dtype=np.float64)
        blockA = engram[:self.d_a]; blockB = engram[self.d_a:self.d_p]
        self._action_ctx = np.asarray(action_ctx, dtype=np.float64)         # the world's constant action context
        # running mean per primitive (de-noise across the facts that share it)
        for store, key, blk in ((self._A_mean, a, blockA), (self._B_mean, b, blockB)):
            if key not in store:
                store[key] = [blk.copy(), 1]
            else:
                s, c = store[key]; s *= c; s += blk; c += 1; s /= c; store[key] = [s, c]
        self._fit_slot(self.Wa, self._A_off + a, a, self._A_mean[a][0], self._anchor_a)
        self._fit_slot(self.Wb, self._B_off + b, b, self._B_mean[b][0], self._anchor_b)

    def regenerate(self, a, b, lesion_a=False, lesion_b=False):
        """Regenerate fact (a,b) = NEURAL SUPERPOSITION of two spiking-readout population outputs. Fire the reservoir
        with primitive-a's query -> a-engram writes its block on the A channels; with primitive-b's query -> b-engram
        writes its block on the B channels; the composed percept is the SUM (bundling) + the constant action context.
        `lesion_*` zeroes one primitive's contribution (the localisation anti-cheat)."""
        a = int(a); b = int(b)
        out = np.zeros(self.n_in, dtype=np.float64)
        if not lesion_a and a in self.Wa:
            rA = self._elig(self._A_off + a)
            out[:self.d_a] += np.clip(rA @ self.Wa[a] + self._anchor_a, 0.0, 1.0)
        if not lesion_b and b in self.Wb:
            rB = self._elig(self._B_off + b)
            out[self.d_a:self.d_p] += np.clip(rB @ self.Wb[b] + self._anchor_b, 0.0, 1.0)
        if self._action_ctx is not None:                                    # world-constant action dims (host-legit)
            out[self.d_p:] += self._action_ctx
        return np.clip(out, 0.0, 1.0)

    # --- store metrics (the sub-linear claim) ---
    def primitive_slots(self):
        return int(len(self.Wa) + len(self.Wb))                            # == P once all primitives seen

    def store_code_floats(self):
        """The learned primitive CODES the generator can regenerate = the compressed memory content, O(P). Directly
        comparable to the flat store's N*n_in raw floats."""
        return int(len(self.Wa) * self.d_a + len(self.Wb) * self.d_b)

    def store_readout_floats(self):
        """The plastic store = the per-primitive readout weights (H x block per slot). Also O(P)."""
        return int(sum(W.size for W in self.Wa.values()) + sum(W.size for W in self.Wb.values()))


# ============================ STEP 3a: the compositional (treatment) arm ============================
def _action_ctx_const():
    a = np.zeros(N_ACT, dtype=np.float64); a[0] = 1.0                       # the scaling atom's fixed action ("eats")
    return a


def _run_compositional_arm(seed, referents, attrs, env, K, n_in, d_a, d_b, K1, K2, slow_hidden, gen_hidden, gen_k,
                           settle, epochs, batch, eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs,
                           replay_per_fact, replay_noise, chance, bdsp_wmax, gen_settle, gen_lr, conv_tol,
                           conv_max_epochs):
    """TREATMENT: same fixed slow cortex + same wake budget as flat; the sleep replay SOURCE is the COMPOSITIONAL
    generator (O(P) primitive engrams) regenerating ALL N facts by NEURAL composition (superposition)."""
    net, slow_active0 = _build_slow_cortex(n_in, K, seed, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax,
                                           env, referents)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)

    gen = CompositionalGenerator(gen_k, n_in, d_a, d_b, K1, K2, gen_hidden, seed, gen_settle, gen_lr, w_clip,
                                 bdsp_wmax=bdsp_wmax, conv_tol=conv_tol, conv_max_epochs=conv_max_epochs)
    action_ctx = _action_ctx_const()

    acquire_acc, slow_active_trace = [], []
    retention, gen_fidelity, store_curve = {}, {}, {}
    max_replay_set = 0
    for i, (r, (a, b)) in enumerate(zip(referents, attrs)):
        # --- WAKE: teacher teaches fact i from the world; the slow cortex moves by e-prop ---
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acquire_acc.append(_fact_acc(net, env, r, i, n=test_n))
        engram_i = np.asarray(X, dtype=np.float64).mean(axis=0)             # the brain's compressed wake trace
        # --- keep the COMPOSITIONAL generator current: learn/refine primitive a and b (store grows only if new) ---
        gen.learn_fact(a, b, engram_i, action_ctx)
        gen._slot_trace.append((i + 1, gen.primitive_slots()))
        # --- SLEEP: regenerate ALL learned facts by COMPOSING their two primitives, e-prop into the slow cortex ---
        classes = list(range(i + 1))
        max_replay_set = max(max_replay_set, len(classes))
        Xr, yr = [], []
        for j in classes:
            aj, bj = attrs[j]
            eg = gen.regenerate(aj, bj)                                     # NEURAL composition (spikes -> superposition)
            for _ in range(replay_per_fact):
                Xr.append(eg + replay_noise * brain_rng.standard_normal(eg.shape[0]))
                yr.append(j)
        Xr = np.asarray(Xr, dtype=np.float64); yr = np.asarray(yr, dtype=np.int64)
        _teach_fact(net, Xr, yr, replay_epochs, batch, brain_rng)
        slow_active_trace.append(int(net.n_active))
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a2 >= max(0.5, chance + 0.15) for a2 in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "generator_primitive_slots": gen.primitive_slots(),
                "generator_store_code_floats": gen.store_code_floats(),
                "generator_store_readout_floats": gen.store_readout_floats(),
                "generator_stored_raw_patterns": int(gen._stored_raw_patterns),
                "flat_buffer_floats_equiv": int(N * n_in),
                "slow_reservoir_active": int(net.n_active),
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a2) for a2 in accs],
            }
            store_curve[str(N)] = {"slots": gen.primitive_slots(), "code_floats": gen.store_code_floats(),
                                   "readout_floats": gen.store_readout_floats(), "flat_floats": int(N * n_in)}
            # --- generator regeneration fidelity vs the TRUE composed prototypes (ruler only) ---
            regens = [gen.regenerate(*attrs[j]) for j in range(N)]
            protos = np.stack([env.proto(referents[j]) for j in range(N)])  # (N, d_p) noiseless composed prototypes
            coss = [_cos(regens[j][:gen.d_p], protos[j]) for j in range(N)]
            near_ok = 0
            for j in range(N):
                dd = np.linalg.norm(protos - regens[j][None, :gen.d_p], axis=1)
                near_ok += int(int(np.argmin(dd)) == j)
            gen_fidelity[str(N)] = {"mean_cos": float(np.mean(coss)), "min_cos": float(np.min(coss)),
                                    "nearest_proto_acc": float(near_ok / N)}
    # --- LESION localisation anti-cheat: zeroing primitive a must break ONLY the A block, primitive b ONLY the B block ---
    lesion = _lesion_localises(gen, attrs, env, referents)
    generative_not_buffer = bool(gen._stored_raw_patterns == 0)
    used_ruler = bool(gen._used_ruler)
    return {
        "arm": "compositional_gen", "gen_k_query_width": int(gen_k),
        "slow_reservoir_active_start": slow_active0,
        "slow_reservoir_active_constant": bool(len(set(slow_active_trace)) == 1),
        "generator_primitive_slots": gen.primitive_slots(),
        "generator_store_code_floats": gen.store_code_floats(),
        "generator_store_readout_floats": gen.store_readout_floats(),
        "generator_stored_raw_patterns": int(gen._stored_raw_patterns),
        "generative_not_stored_buffer": generative_not_buffer,
        "consolidation_used_ruler": used_ruler,
        "generator_slot_trace": [[int(n), int(p)] for n, p in gen._slot_trace],
        "max_replay_set_size": int(max_replay_set),
        "acquire_acc_immediate": [float(a2) for a2 in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention, "generator_fidelity": gen_fidelity, "store_curve": store_curve,
        "lesion_localises": lesion,
    }


def _lesion_localises(gen, attrs, env, referents, n_probe=8):
    """Composition-is-neural teeth: for a sample of facts, lesion primitive a -> the A block collapses toward the
    anchor (its channels lose their engram) while the B block is UNCHANGED, and vice-versa. Returns the mean block
    errors so localisation is visible, plus a boolean that the lesion hits the RIGHT block much harder."""
    idxs = list(range(0, len(attrs), max(1, len(attrs) // n_probe)))[:n_probe]
    a_hit, a_spare, b_hit, b_spare = [], [], [], []
    for j in idxs:
        a, b = attrs[j]
        full = gen.regenerate(a, b)
        les_a = gen.regenerate(a, b, lesion_a=True)                         # remove primitive a
        les_b = gen.regenerate(a, b, lesion_b=True)                         # remove primitive b
        a_hit.append(float(np.linalg.norm(full[:gen.d_a] - les_a[:gen.d_a])))          # A block changed by lesion-a
        a_spare.append(float(np.linalg.norm(full[gen.d_a:gen.d_p] - les_a[gen.d_a:gen.d_p])))  # B block spared
        b_hit.append(float(np.linalg.norm(full[gen.d_a:gen.d_p] - les_b[gen.d_a:gen.d_p])))    # B block changed by lesion-b
        b_spare.append(float(np.linalg.norm(full[:gen.d_a] - les_b[:gen.d_a])))         # A block spared
    ma_hit, ma_sp = float(np.mean(a_hit)), float(np.mean(a_spare))
    mb_hit, mb_sp = float(np.mean(b_hit)), float(np.mean(b_spare))
    localises = bool(ma_hit > 0.05 and mb_hit > 0.05 and ma_hit > 5.0 * (ma_sp + 1e-9)
                     and mb_hit > 5.0 * (mb_sp + 1e-9))
    return {"lesionA_Ablock_delta": ma_hit, "lesionA_Bblock_delta": ma_sp,
            "lesionB_Bblock_delta": mb_hit, "lesionB_Ablock_delta": mb_sp, "localises": localises}


# ============================ BONUS: zero-shot compositional generalization ============================
def _zeroshot_generalization(seed, referents, attrs, env, n_in, d_a, d_b, K1, K2, gen_hidden, gen_k, settle,
                             gen_lr, w_clip, bdsp_wmax, conv_tol, conv_max_epochs, batch, gen_epochs, n_draws):
    """Compose > memorize, made mechanical. Hold out a set of (a,b) facts such that EACH held-out fact's a AND b
    still appear in OTHER (trained) facts. Teach a FRESH compositional generator + a FRESH v2 (noncompositional)
    generator on ONLY the trained complement, then regenerate the HELD-OUT facts. The compositional generator
    composes already-learned primitives (high cos); v2 never saw those class targets (~chance)."""
    N = len(attrs)
    # held-out = the anti-diagonal minus the corners, guaranteeing every held-out a,b appears elsewhere.
    heldout = []
    for j, (a, b) in enumerate(attrs):
        if K1 >= 3 and K2 >= 3 and (a + b) == (K1 - 1) and 0 < a < K1 - 1:
            heldout.append(j)
    if not heldout:                                                        # tiny grid fallback: a single interior cell
        for j, (a, b) in enumerate(attrs):
            if 0 < a < K1 - 1 and 0 < b < K2 - 1:
                heldout.append(j); break
    heldout = set(heldout)
    trained = [j for j in range(N) if j not in heldout]
    if not heldout:
        return {"held_out_n": 0, "note": "grid too small for a valid held-out set"}
    # verify each held-out primitive appears in the trained complement (else it is not a compositional test)
    trained_a = {attrs[j][0] for j in trained}; trained_b = {attrs[j][1] for j in trained}
    valid = all(attrs[j][0] in trained_a and attrs[j][1] in trained_b for j in heldout)

    action_ctx = _action_ctx_const()
    # compositional generator: teach only trained facts
    cgen = CompositionalGenerator(gen_k, n_in, d_a, d_b, K1, K2, gen_hidden, seed, settle, gen_lr, w_clip,
                                  bdsp_wmax=bdsp_wmax, conv_tol=conv_tol, conv_max_epochs=conv_max_epochs)
    for j in trained:
        a, b = attrs[j]
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)
        cgen.learn_fact(a, b, np.asarray(Xj, dtype=np.float64).mean(axis=0), action_ctx)
    # v2 fixed generator: teach only trained facts (class = original index)
    vgen = GenerativeReplayNetV2(int(gen_k), n_in, gen_hidden, seed, settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax,
                                 conv_tol=0.05, conv_max_epochs=120, conv_check_every=4, new_mult=3)
    vgen.fit_query_norm()
    vgen_rng = np.random.default_rng(seed + 999)
    seen = []
    for j in trained:
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)
        vgen.learn_fact(j, np.asarray(Xj, dtype=np.float64).mean(axis=0), list(seen), gen_epochs, batch, vgen_rng)
        seen.append(j)
    # regenerate the held-out facts
    comp_cos, v2_cos = [], []
    for j in heldout:
        a, b = attrs[j]
        proto = env.proto(referents[j])
        comp_cos.append(_cos(cgen.regenerate(a, b)[:cgen.d_p], proto))
        v2_cos.append(_cos(vgen.regenerate(j)[:d_a + d_b], proto))
    return {"held_out_n": len(heldout), "held_out_valid_primitives_seen": bool(valid),
            "compositional_heldout_cos": float(np.mean(comp_cos)),
            "noncompositional_v2_heldout_cos": float(np.mean(v2_cos)),
            "compose_beats_memorize": bool(np.mean(comp_cos) > np.mean(v2_cos) + 0.20 and np.mean(comp_cos) >= 0.85)}


# ============================ per-grid driver ============================
def _run_grid(seed, K1, K2, d_a, d_b, noise, slow_hidden, gen_hidden, gen_k, settle, epochs, batch, eprop_lr,
              w_clip, n_draws, test_n, replay_epochs, replay_per_fact, replay_noise, gen_settle, gen_epochs,
              gen_lr, conv_tol, conv_max_epochs, bdsp_wmax, arms_to_run):
    N = K1 * K2; P = K1 + K2; K = N
    chance = 1.0 / K
    n_in = d_a + d_b + N_ACT
    referents, attrs = _grid_facts(K1, K2)
    milestones = [N]
    slow_h = int(slow_hidden) if slow_hidden and int(slow_hidden) > 0 else max(120, 6 * N)

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        env = _make_comp_env(seed, K1, K2, d_a, d_b, noise, referents, attrs)   # fresh env per arm: SAME percepts
        if arm == "compositional_gen":
            arms[arm] = _run_compositional_arm(seed, referents, attrs, env, K, n_in, d_a, d_b, K1, K2, slow_h,
                                               gen_hidden, gen_k, settle, epochs, batch, eprop_lr, w_clip, n_draws,
                                               milestones, test_n, replay_epochs, replay_per_fact, replay_noise,
                                               chance, bdsp_wmax, gen_settle, gen_lr, conv_tol, conv_max_epochs)
            arms[arm]["zeroshot"] = _zeroshot_generalization(
                seed, referents, attrs, env, n_in, d_a, d_b, K1, K2, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                bdsp_wmax, conv_tol, conv_max_epochs, batch, gen_epochs, n_draws)
        elif arm == "noncompositional_gen":
            arms[arm] = _run_generative_arm_v2(
                arm, GenerativeReplayNetV2,
                dict(conv_tol=0.05, conv_max_epochs=120, conv_check_every=4, new_mult=3),
                gen_k, seed, referents, env, K, n_in, slow_h, gen_hidden, settle, epochs, batch, eprop_lr, w_clip,
                n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance, bdsp_wmax,
                gen_settle, gen_epochs, gen_lr)
        else:  # flat = the O(N) target (CLS flat arm)
            arms[arm] = _run_cls_arm("flat", seed, referents, env, K, n_in, slow_h, 5, settle, epochs, batch,
                                     eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact,
                                     replay_noise, chance, bdsp_wmax)
            arms[arm]["arm"] = arm
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        fr = rc.get(str(N), {}).get("frac_recalled", float("nan"))
        print(f"  [grid {K1}x{K2} N={N} P={P}] arm {arm:22s} {arms[arm]['wall_seconds']:.0f}s | "
              f"acq {arms[arm].get('mean_acquire_acc_immediate', float('nan')):.3f} | frac-recalled {fr:.2f}",
              flush=True)
    return {"K1": K1, "K2": K2, "N": N, "P": P, "chance": chance, "n_in": n_in, "slow_hidden": slow_h,
            "arms": arms}


def run(seed, grids, d_a, d_b, noise, slow_hidden, gen_hidden, gen_k, settle, epochs, batch, eprop_lr, w_clip,
        n_draws, test_n, replay_epochs, replay_per_fact, replay_noise, gen_settle, gen_epochs, gen_lr, conv_tol,
        conv_max_epochs, bdsp_wmax, arms_to_run):
    n_in = d_a + d_b + N_ACT
    # anti-cheats independent of grid size: byte-identical substrate + sim/ clean. Use the biggest grid's K.
    Kbig = max(k1 * k2 for k1, k2 in grids)
    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, Kbig, seed, max(120, 6 * Kbig), settle,
                                                               eprop_lr, w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()

    per_grid = {}
    for (K1, K2) in grids:
        print(f"\n{'=' * 90}\n# SEED {seed}  GRID {K1}x{K2} (N={K1*K2}, P={K1+K2})\n{'=' * 90}", flush=True)
        per_grid[f"{K1}x{K2}"] = _run_grid(seed, K1, K2, d_a, d_b, noise, slow_hidden, gen_hidden, gen_k, settle,
                                           epochs, batch, eprop_lr, w_clip, n_draws, test_n, replay_epochs,
                                           replay_per_fact, replay_noise, gen_settle, gen_epochs, gen_lr, conv_tol,
                                           conv_max_epochs, bdsp_wmax, arms_to_run)
    return {"seed": seed, "grids": [f"{k1}x{k2}" for k1, k2 in grids], "d_a": d_a, "d_b": d_b, "n_in": n_in,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"d_a": d_a, "d_b": d_b, "noise": noise, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "settle_steps": settle, "epochs": epochs, "batch": batch, "eprop_lr": eprop_lr,
                       "w_clip": w_clip, "n_draws": n_draws, "test_n": test_n, "replay_epochs": replay_epochs,
                       "replay_per_fact": replay_per_fact, "replay_noise": replay_noise, "gen_settle": gen_settle,
                       "gen_epochs": gen_epochs, "gen_lr": gen_lr, "conv_tol": conv_tol,
                       "conv_max_epochs": conv_max_epochs, "bdsp_wmax": bdsp_wmax, "frozen_hidden": True},
            "per_grid": per_grid}


def _frac(grid, arm):
    rc = grid["arms"].get(arm, {}).get("retention_curve", {})
    N = grid["N"]
    return rc.get(str(N), {}).get("frac_recalled", float("nan"))


def _verdict(result):
    """Verdict + GO. THE KEY: the compositional generator's STORE grows SUB-LINEARLY (~P~sqrt(N)) while retention
    holds near flat, at BOTH grids. Anti-cheats: composition neural (lesion localises), P<<N slots (not N), 0 stored
    raw patterns, ruler untouched, acquisition high, byte-identical, sim/ clean."""
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    grids = result["per_grid"]
    gkeys = sorted(grids, key=lambda g: grids[g]["N"])
    v = Verdict("teacher-loop compositional generator (sub-linear compressing generator)", chance=None)

    # per-grid: retention within 0.15 of flat AND >= 0.5; slots == P << N; neural; anti-cheats
    ret_ok_all, store_ok_all = True, True
    per_grid_summary = {}
    comp_acqs = []
    for g in gkeys:
        gr = grids[g]; N = gr["N"]; P = gr["P"]
        carm = gr["arms"].get("compositional_gen", {})
        comp_f = _frac(gr, "compositional_gen"); flat_f = _frac(gr, "flat"); v2_f = _frac(gr, "noncompositional_gen")
        acq = carm.get("mean_acquire_acc_immediate", float("nan"))
        comp_acqs.append(acq)
        rc = carm.get("retention_curve", {}).get(str(N), {})
        slots = rc.get("generator_primitive_slots", None)
        code_floats = rc.get("generator_store_code_floats", None)
        flat_floats = rc.get("flat_buffer_floats_equiv", None)
        fid = carm.get("generator_fidelity", {}).get(str(N), {})
        lesion = carm.get("lesion_localises", {})
        not_buffer = bool(carm.get("generative_not_stored_buffer"))
        no_ruler = bool(not carm.get("consolidation_used_ruler"))
        slow_const = bool(carm.get("slow_reservoir_active_constant"))
        zs = carm.get("zeroshot", {})

        # ATTRIBUTION: whose is the retention, and does composition beat the fixed generator (store + zero-shot)?
        if not np.isnan(flat_f):
            attributable_to(f"[{g}] compositional retention vs the flat O(N) store", comp_f, flat_f)
        if not np.isnan(v2_f):
            attributable_to(f"[{g}] compositional vs the fixed noncompositional generator (retention)", comp_f, v2_f)
        if zs.get("compositional_heldout_cos") is not None and zs.get("noncompositional_v2_heldout_cos") is not None:
            attributable_to(f"[{g}] zero-shot: composing primitives vs memorizing (compositional vs v2)",
                            zs["compositional_heldout_cos"], zs["noncompositional_v2_heldout_cos"])
        ret_ok = bool((not np.isnan(flat_f)) and comp_f >= flat_f - 0.15 and comp_f >= 0.5)
        slots_ok = bool(slots == P and P < N)
        store_smaller = bool(code_floats is not None and flat_floats is not None and code_floats < flat_floats)
        neural_ok = bool(lesion.get("localises"))
        p_over_n = P / N

        v.require(f"[{g}] retention within 0.15 of flat AND >= 0.5", ret_ok, expect=True,
                  note=f"comp {comp_f:.2f} vs flat {flat_f:.2f} (v2 {v2_f:.2f}) @ N={N}")
        v.require(f"[{g}] store slots == P == {P} (<< N={N})", slots_ok, expect=True, note=f"slots {slots}")
        v.require(f"[{g}] store floats < flat O(N) floats", store_smaller, expect=True,
                  note=f"comp {code_floats} vs flat {flat_floats}")
        v.require(f"[{g}] composition NEURAL (lesion localises to the right block)", neural_ok, expect=True,
                  note=f"A:{lesion.get('lesionA_Ablock_delta',0):.2f}/{lesion.get('lesionA_Bblock_delta',0):.2f} "
                       f"B:{lesion.get('lesionB_Bblock_delta',0):.2f}/{lesion.get('lesionB_Ablock_delta',0):.2f}")
        v.require(f"[{g}] facts genuinely compositional (P/N={p_over_n:.2f} <= 0.6)", bool(p_over_n <= 0.6),
                  expect=True)
        v.require(f"[{g}] 0 stored raw patterns (composes, not a buffer)", not_buffer, expect=True)
        v.require(f"[{g}] consolidation never read the ruler", no_ruler, expect=True)

        ret_ok_all = ret_ok_all and ret_ok
        store_ok_all = store_ok_all and slots_ok and store_smaller and neural_ok and not_buffer and no_ruler and slow_const
        per_grid_summary[g] = {
            "N": N, "P": P, "compositional_frac": comp_f, "flat_frac": flat_f, "noncompositional_v2_frac": v2_f,
            "compositional_minus_flat": (float(comp_f - flat_f) if not np.isnan(flat_f) else None),
            "compositional_minus_v2": (float(comp_f - v2_f) if not np.isnan(v2_f) else None),
            "store_slots": slots, "store_code_floats": code_floats, "flat_store_floats": flat_floats,
            "store_readout_floats": rc.get("generator_store_readout_floats"),
            "p_over_n": p_over_n, "gen_mean_cos": fid.get("mean_cos"), "lesion_localises": neural_ok,
            "immediate_acq": acq, "zeroshot": zs,
        }

    # SUB-LINEAR store across grids: slot ratio ~= sqrt(N ratio) and < flat ratio with margin
    gsmall, gbig = gkeys[0], gkeys[-1]
    Ns, Nb = grids[gsmall]["N"], grids[gbig]["N"]
    Ps, Pb = grids[gsmall]["P"], grids[gbig]["P"]
    slot_ratio = Pb / Ps if Ps else float("nan")
    flat_ratio = Nb / Ns if Ns else float("nan")
    sqrt_ratio = float(np.sqrt(Nb / Ns))
    code_small = per_grid_summary[gsmall]["store_code_floats"]; code_big = per_grid_summary[gbig]["store_code_floats"]
    code_ratio = (code_big / code_small) if code_small else float("nan")
    sublinear = bool((slot_ratio < flat_ratio - 0.3) and (code_ratio < flat_ratio - 0.3)
                     and abs(slot_ratio - sqrt_ratio) < 0.35)
    v.require("(KEY sub-linear) store grows ~sqrt(N), slower than flat O(N)", sublinear, expect=True,
              note=f"slot x{slot_ratio:.2f} code x{code_ratio:.2f} vs flat x{flat_ratio:.2f} (sqrt x{sqrt_ratio:.2f}) "
                   f"[{gsmall}->{gbig}]")

    acq_ok = bool(np.nanmin(comp_acqs) >= 0.85)
    v.floor("(acq) compositional immediate acquisition stays high (min over grids)", float(np.nanmin(comp_acqs)),
            floor=0.85)
    v.require("(seed) substrate byte-identical", bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    go = bool(ret_ok_all and store_ok_all and sublinear and acq_ok and result["substrate_byte_identical"]
              and result["sim_diff_empty"])
    decision = v.decide(go=go)

    return {
        "grids": gkeys, "per_grid": per_grid_summary,
        "store_slot_ratio": slot_ratio, "store_code_ratio": code_ratio, "flat_store_ratio": flat_ratio,
        "sqrt_ratio": sqrt_ratio, "store_sublinear": sublinear,
        "retention_ok_all_grids": ret_ok_all, "store_anticheats_ok_all_grids": store_ok_all,
        "min_immediate_acq": float(np.nanmin(comp_acqs)),
        "substrate_byte_identical": result["substrate_byte_identical"], "sim_diff_empty": result["sim_diff_empty"],
        **decision,
    }


def _parse_grid(s):
    a, b = s.lower().split("x"); return (int(a), int(b))


def _one_seed(a, seed, grids, arms_to_run):
    result = run(seed, grids, a.d_a, a.d_b, a.noise, a.slow_hidden, a.gen_hidden, a.gen_k, a.settle_steps,
                 a.epochs, a.batch, a.eprop_lr, a.w_clip, a.n_draws, a.test_n, a.replay_epochs, a.replay_per_fact,
                 a.replay_noise, a.gen_settle, a.gen_epochs, a.gen_lr, a.conv_tol, a.conv_max_epochs, a.bdsp_wmax,
                 arms_to_run)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop COMPOSITIONAL GENERATOR: a sub-linear compressing "
                                             "generator that stores P<<N primitive engrams and COMPOSES facts by "
                                             "neural superposition -- O(P)=O(sqrt(N)) store, retention near flat.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--grids", nargs="+", default=["4x4", "6x6"], help="K1xK2 grids (each a full curriculum); >=2 "
                                                                       "so the store-vs-N sub-linear claim is measurable")
    ap.add_argument("--d-a", type=int, default=8, help="attribute-A block dim (A primitive code width)")
    ap.add_argument("--d-b", type=int, default=8, help="attribute-B block dim")
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--slow-hidden", type=int, default=0, help="0 => auto max(120, 6*N) per grid (the FIXED slow cortex)")
    ap.add_argument("--gen-hidden", type=int, default=96, help="the FIXED generator reservoir size (H_gen)")
    ap.add_argument("--gen-k", type=int, default=64, help="FIXED query address width (sparse collision-free codes)")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--replay-epochs", type=int, default=12)
    ap.add_argument("--replay-per-fact", type=int, default=8)
    ap.add_argument("--replay-noise", type=float, default=0.10)
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-epochs", type=int, default=16, help="v2 (noncompositional control) self-replay epochs/fact")
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--conv-tol", type=float, default=0.02, help="compositional primitive-engram NLMS convergence tol")
    ap.add_argument("--conv-max-epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9)
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["compositional_gen", "flat", "noncompositional_gen"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grids = [_parse_grid(g) for g in a.grids]
    arms_to_run = list(a.arms)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  grids={a.grids} gen_H={a.gen_hidden} gen_k={a.gen_k}\n" + "#" * 100,
              flush=True)
        result, verdict = _one_seed(a, s, grids, arms_to_run)
        summary = {"probe": "teacher_loop_compositional_generator", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "arms_run": arms_to_run, "grids": a.grids, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        for g in rv["grids"]:
            pg = rv["per_grid"][g]
            zs = pg.get("zeroshot", {})
            print(f"[compgen] seed {s} {g}: N={pg['N']} P={pg['P']} | comp {pg['compositional_frac']:.2f} "
                  f"| flat {pg['flat_frac']:.2f} | v2 {pg['noncompositional_v2_frac']:.2f} | slots {pg['store_slots']} "
                  f"code-floats {pg['store_code_floats']} (flat {pg['flat_store_floats']}) | cos {pg['gen_mean_cos']} "
                  f"| lesion-loc {pg['lesion_localises']} | zeroshot comp/v2 "
                  f"{zs.get('compositional_heldout_cos')}/{zs.get('noncompositional_v2_heldout_cos')}", flush=True)
        print(f"[compgen] SUB-LINEAR store {rv['store_sublinear']}: slot x{rv['store_slot_ratio']:.2f} "
              f"code x{rv['store_code_ratio']:.2f} vs flat x{rv['flat_store_ratio']:.2f} (sqrt x{rv['sqrt_ratio']:.2f}) "
              f"| acq {rv['min_immediate_acq']:.3f} | byte-id {rv['substrate_byte_identical']} "
              f"sim-clean {rv['sim_diff_empty']} | VERDICT {rv['status']}", flush=True)
        print(f"[compgen] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        agg = {"probe": "teacher_loop_compositional_generator_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "grids": a.grids, "go_count": go_n, "n_seeds": len(seeds),
               "sublinear_count": int(sum(1 for p in per_seed if p["verdict"].get("store_sublinear"))),
               "per_grid_means": {}, "per_seed": per_seed}
        for g in per_seed[0]["verdict"]["grids"]:
            comp = [p["verdict"]["per_grid"][g]["compositional_frac"] for p in per_seed]
            flat = [p["verdict"]["per_grid"][g]["flat_frac"] for p in per_seed]
            v2 = [p["verdict"]["per_grid"][g]["noncompositional_v2_frac"] for p in per_seed]
            slots = per_seed[0]["verdict"]["per_grid"][g]["store_slots"]
            code = per_seed[0]["verdict"]["per_grid"][g]["store_code_floats"]
            flatf = per_seed[0]["verdict"]["per_grid"][g]["flat_store_floats"]
            zs_c = [p["verdict"]["per_grid"][g].get("zeroshot", {}).get("compositional_heldout_cos") for p in per_seed]
            zs_v = [p["verdict"]["per_grid"][g].get("zeroshot", {}).get("noncompositional_v2_heldout_cos") for p in per_seed]
            agg["per_grid_means"][g] = {
                "N": per_seed[0]["verdict"]["per_grid"][g]["N"], "P": per_seed[0]["verdict"]["per_grid"][g]["P"],
                "compositional_frac_mean": float(np.nanmean(comp)), "flat_frac_mean": float(np.nanmean(flat)),
                "noncompositional_v2_frac_mean": float(np.nanmean(v2)),
                "store_slots": slots, "store_code_floats": code, "flat_store_floats": flatf,
                "compositional_frac_per_seed": [float(x) for x in comp],
                "zeroshot_compositional_cos_mean": float(np.nanmean([x for x in zs_c if x is not None])) if any(x is not None for x in zs_c) else None,
                "zeroshot_v2_cos_mean": float(np.nanmean([x for x in zs_v if x is not None])) if any(x is not None for x in zs_v) else None,
            }
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[compgen AGG] GO {go_n}/{len(seeds)} | sub-linear {agg['sublinear_count']}/{len(seeds)}", flush=True)
        for g, m in agg["per_grid_means"].items():
            print(f"   {g}: N={m['N']} P={m['P']} slots={m['store_slots']} code={m['store_code_floats']} "
                  f"(flat {m['flat_store_floats']}) | comp {m['compositional_frac_mean']:.2f} vs flat "
                  f"{m['flat_frac_mean']:.2f} vs v2 {m['noncompositional_v2_frac_mean']:.2f} | zeroshot comp/v2 "
                  f"{m['zeroshot_compositional_cos_mean']}/{m['zeroshot_v2_cos_mean']}", flush=True)
        print(f"[compgen AGG] wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
