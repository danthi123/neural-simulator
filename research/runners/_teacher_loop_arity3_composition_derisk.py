"""TEACHER-LOOP ARITY-3 ZERO-SHOT COMPOSITION DE-RISK (2026-08-09): does neural superposition stay zero-shot
separable when the bundle carries THREE terms, not two?

WHERE THE ARC STANDS. The arity-2 zero-shot lever is GO (banked on main, ea77003c5): facts = a K1xK2 grid of two
attribute-vocabularies; a percept = concat of TWO primitive codes on DISJOINT channels; the compositional generator
stores P=K1+K2 PRIMITIVE engrams and regenerates a fact by NEURAL SUPERPOSITION (SUM) of its TWO primitive
spiking-readout outputs. Held-out (a,b) combinations NEVER taught as facts are recalled ~1.00 zero-shot, while a
fixed class-indexed generator (v2) and a flat instance store sit at chance (they address facts by CLASS INDEX and
never saw those classes). Genuine NEURAL compositional generalization at arity 2.

THE TIGHT QUESTION (this de-risk). Superposition of MORE terms shrinks the pairwise separation margin (each added
bundled vector is crosstalk to the others). Does the SAME mechanism stay zero-shot-separable at ARITY 3 -- facts =
(a,b,c) from THREE attribute-vocabularies (K1xK2xK3), a percept = concat of THREE primitive codes on DISJOINT
channels, regenerate = SUM of THREE per-primitive spiking readouts? MINIMAL extension of the arity-2 harness: ONE
extra attribute channel, the generator stores P=K1+K2+K3 primitives, the composed percept sums THREE blocks.

THE MEASURE (arm-symmetric, identical to arity 2). RECALL of a fact = regenerate its percept, then
nearest-prototype identify it among ALL N true prototypes (chance = 1/N). Reported at ARITY 3 (K=4 -> N=64) AND at
the ARITY-2 baseline (K=4 -> N=16) IN THE SAME RUN, vs the two floors:
  * compositional (3-way bundle): regenerate(a,b,c) = NEURAL SUPERPOSITION of primitive-a's, primitive-b's and
    primitive-c's spiking readouts on their disjoint channel blocks. Held-out (a,b,c): all three primitives were
    seen in OTHER taught facts -> the three engrams exist -> it COMPOSES the never-taught percept.
  * noncompositional_v2 (FLOOR): the fixed class-indexed generator addresses a fact by its class INDEX. A held-out
    class was never trained -> untrained near-anchor readout -> ~chance identity.
  * flat (FLOOR): an O(N) raw-engram buffer keyed by class index. No entry for a held-out class -> uniform guess
    over the N classes -> chance (1/N).

ANTI-CHEATS (each a REAL assertion in the output, unchanged in spirit from arity 2):
  * taught / held-out DISJOINT; every held-out primitive (a AND b AND c) appears in >= 1 TAUGHT fact (coverage-
    preserving split: a cell is held out ONLY if its a-row, b-column AND c-depth each keep >= 1 taught fact).
  * NO LEAKAGE: no held-out fact index ever enters ANY training path (asserted DISJOINT from the fed set); the true
    held-out percept is read ONLY by the test-time nearest-prototype RULER, never by a learning call.
  * composition at test is NEURAL: regeneration is a SUM of THREE spiking-reservoir leaky-readout outputs; a LESION
    zeroing one primitive's engram breaks ONLY that primitive's block on the HELD-OUT facts (3-way localisation).
  * 0 stored raw patterns; ruler untouched by learning; cfg.seed byte-identical substrate (NOT actual_seed_used);
    de-clamped bdsp_wmax=1e9; git diff main -- sim/ empty; backend recorded.

GO (arity 3, per seed): compositional HELD-OUT recall >= 0.5 AND >= v2 held-out + 0.30 AND >= flat held-out + 0.30;
compositional TAUGHT (seen) recall >= 0.85; lesion localises on held-out; disjoint + coverage + no-leakage; 0 raw
patterns; ruler untouched; byte-identical; sim/ clean. If the 3-term bundle degrades (held-out recall near the
floor while arity 2 held ~1.00), that is a FIRST-CLASS HONEST NEGATIVE naming WHY (low regen cos = binding
interference from the third term; high cos but wrong nearest = primitive-code collision at higher N).

DR grounding: Plate 1995 HRR / Kanerva 2009 VSA bundling capacity (superposition margin falls ~1/sqrt(#terms));
van de Ven 2020 generative replay; the arity-2 zero-shot GO (this repo).

DISCIPLINE: reuse-by-import (the arity-2 zero-shot harness for the arity-2 baseline in-run + its held-out split /
FlatStore / recall helpers; the compositional generator's reservoir + NLMS primitive-engram fit + the world's
CompositionalReferentEnv extended to a 3rd channel; the v2 fixed generator = the floor; the byte-identical +
sim-clean asserts). NO sim/ edit. SIM_BACKEND=numpy.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_arity3_composition_derisk --seed 42 --K 4 --held-out 16 \
      --gen-hidden 96 --gen-k 96 \
      --out research/findings/raw/teacher_loop_arity3_composition_s42.json
  3-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_arity3_composition_derisk --seeds 42 43 44 --K 4 --held-out 16 \
      --gen-hidden 96 --gen-k 96 \
      --out research/findings/raw/teacher_loop_arity3_composition.json
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

# reuse-by-import: the compositional generator's reservoir + NLMS primitive fit + the world env; the v2 fixed
# generator (the FLOOR); the arity-2 zero-shot harness (its held-out split / FlatStore / recall helpers + the WHOLE
# arity-2 grid driver for the in-run baseline); the scaling teacher machinery; byte-identical + sim-clean asserts.
from research.runners._teacher_loop_compositional_generator_derisk import (  # noqa: E402
    CompositionalReferentEnv, CompositionalGenerator, _action_ctx_const,
)
from research.runners._teacher_loop_generative_replay_v2_derisk import GenerativeReplayNetV2  # noqa: E402
from research.runners._teacher_loop_generative_replay_derisk import GenerativeReplayNet, _cos  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import _corrective_batch, N_ACT  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _assert_byte_identical_substrate, _git_sim_diff_empty,
)
from research.runners._teacher_loop_zeroshot_composition_derisk import (  # noqa: E402
    _run_grid as _run_arity2_grid, _nearest_proto, _recall_fraction, FlatStore,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_arity3_composition.json"


# ============================ the ARITY-3 compositional WORLD (host, legitimate) ============================
class CompositionalReferentEnv3(CompositionalReferentEnv):
    """Extend the arity-2 world with a THIRD disjoint attribute channel. K1 A-primitives ([0,1]^d_a), K2 B-primitives
    ([0,1]^d_b), K3 C-primitives ([0,1]^d_c), fixed + seed-controlled. Fact (a,b,c)'s percept prototype =
    concat(primA[a], primB[b], primC[c]) on DISJOINT channels -- a deterministic COMPOSITION of THREE primitives.
    N = K1*K2*K3 facts share exactly P = K1+K2+K3 primitive codes. Host code, legitimate exactly as a retinal render."""

    def __init__(self, seed, K1, K2, K3, d_a=8, d_b=8, d_c=8, noise=0.12):
        # ReferentEnv.__init__ via CompositionalReferentEnv sets d_p; override to include the C block.
        super().__init__(seed, K1, K2, d_a=d_a, d_b=d_b, noise=noise)
        self.K3 = int(K3); self.d_c = int(d_c)
        self.d_p = int(d_a) + int(d_b) + int(d_c)
        wr = np.random.default_rng(int(seed) + 30303030)                    # a dedicated C-channel WORLD RNG
        self.primC = [wr.random(self.d_c).astype(np.float64) for _ in range(self.K3)]
        self.fact_attrs = {}                                                # referent -> (a, b, c)
        self.protos = {}

    def register(self, referent, a, b, c):
        self.fact_attrs[referent] = (int(a), int(b), int(c))

    def proto(self, referent):
        if referent not in self.protos:
            a, b, c = self.fact_attrs[referent]
            self.protos[referent] = np.concatenate(
                [self.primA[a], self.primB[b], self.primC[c]]).astype(np.float64)   # COMPOSE three
        return self.protos[referent]


def _grid_facts3(K1, K2, K3):
    """The N = K1*K2*K3 referents of the cube, class index = (a*K2 + b)*K3 + c, and each referent's (a,b,c)."""
    refs, attrs = [], []
    for a in range(K1):
        for b in range(K2):
            for c in range(K3):
                refs.append(f"r{a}_{b}_{c}"); attrs.append((a, b, c))
    return refs, attrs


def _make_comp_env3(seed, K1, K2, K3, d_a, d_b, d_c, noise, referents, attrs):
    env = CompositionalReferentEnv3(seed, K1, K2, K3, d_a=d_a, d_b=d_b, d_c=d_c, noise=noise)
    for r, (a, b, c) in zip(referents, attrs):
        env.register(r, a, b, c)
        env.proto(r)                                                        # instantiate the composed prototype
    env.rng = np.random.default_rng(seed + 101)                            # reset draw-stream => every arm same percepts
    return env


# ============================ the ARITY-3 compositional generator (neural; SUM of THREE readouts) ============
class CompositionalGenerator3(CompositionalGenerator):
    """The arity-2 primitive-storing generator extended to a THIRD engram family. ONE frozen spiking reservoir; store
    = ONE primitive engram per DISTINCT attribute value across THREE families (Wa, Wb, Wc), each a per-primitive
    leaky-readout weight-set trained by the local NLMS delta rule on the reservoir's spike eligibility for that
    primitive's query address. Regeneration = NEURAL SUPERPOSITION of THREE spiking-readout population outputs
    (bundling) on their disjoint channel blocks. The world prototypes are NEVER read by learning (_used_ruler=False)."""

    def __init__(self, gen_k, n_in, d_a, d_b, d_c, K1, K2, K3, hidden, seed, settle, gen_lr, w_clip, bdsp_wmax=1e9,
                 conv_tol=0.02, conv_max_epochs=200, conv_check_every=4):
        # the SHARED frozen spiking reservoir + eligibility + query-code machinery (readout-only; de-clamped).
        self._res = GenerativeReplayNet(int(gen_k), int(n_in), int(hidden), seed, settle, gen_lr, w_clip,
                                        bdsp_wmax=bdsp_wmax)
        self.d_a = int(d_a); self.d_b = int(d_b); self.d_c = int(d_c)
        self.n_in = int(n_in); self.d_p = int(d_a) + int(d_b) + int(d_c)
        self.K1 = int(K1); self.K2 = int(K2); self.K3 = int(K3); self.gen_lr = float(gen_lr)
        self.conv_tol = float(conv_tol); self.conv_max_epochs = int(conv_max_epochs)
        self.conv_check_every = max(1, int(conv_check_every))
        self._A_off = 0; self._B_off = 1_000_000; self._C_off = 2_000_000    # disjoint reservoir query families
        # condition the readout eligibility (mu=0 keeps the bias dof; a single global RMS scale) over ALL primitive
        # addresses across the three families -- same conditioning the arity-2 generator uses.
        addrs = ([self._A_off + a for a in range(self.K1)]
                 + [self._B_off + b for b in range(self.K2)]
                 + [self._C_off + c for c in range(self.K3)])
        R = np.array([self._res._readout_elig(self._res._forward_record(self._res._query_code(ad))[0]) for ad in addrs])
        self._res._r_mu = np.zeros(R.shape[1], dtype=np.float64)             # keep the common-mode / bias direction
        self._res._r_sigma = R.std(axis=0) + 1e-3
        self._H = int(R.shape[1])
        self.Wa = {}; self.Wb = {}; self.Wc = {}                            # THREE primitive-engram families
        self._A_mean = {}; self._B_mean = {}; self._C_mean = {}            # running-mean observed block per primitive
        self._anchor_a = 0.5 * np.ones(self.d_a, dtype=np.float64)          # world constant (percepts in [0,1])
        self._anchor_b = 0.5 * np.ones(self.d_b, dtype=np.float64)
        self._anchor_c = 0.5 * np.ones(self.d_c, dtype=np.float64)
        self._action_ctx = None
        self._stored_raw_patterns = 0                                      # anti-cheat: NEVER stores raw composed patterns
        self._used_ruler = False                                           # anti-cheat: learning never reads true primitives
        self._slot_trace = []

    def learn_fact(self, a, b, c, engram, action_ctx):
        """Teach fact (a,b,c) from its wake engram. Update the running-mean block estimate for primitives a, b, c
        (each repeats across the facts that share it -> de-noises), then (re)fit each primitive engram. NEW primitive
        => a new slot (store grows); seen primitive => refine only. NEVER reads the world's true prototypes."""
        a = int(a); b = int(b); c = int(c)
        engram = np.asarray(engram, dtype=np.float64)
        dab = self.d_a + self.d_b
        blockA = engram[:self.d_a]; blockB = engram[self.d_a:dab]; blockC = engram[dab:self.d_p]
        self._action_ctx = np.asarray(action_ctx, dtype=np.float64)
        for store, key, blk in ((self._A_mean, a, blockA), (self._B_mean, b, blockB), (self._C_mean, c, blockC)):
            if key not in store:
                store[key] = [blk.copy(), 1]
            else:
                s, cc = store[key]; s *= cc; s += blk; cc += 1; s /= cc; store[key] = [s, cc]
        self._fit_slot(self.Wa, self._A_off + a, a, self._A_mean[a][0], self._anchor_a)
        self._fit_slot(self.Wb, self._B_off + b, b, self._B_mean[b][0], self._anchor_b)
        self._fit_slot(self.Wc, self._C_off + c, c, self._C_mean[c][0], self._anchor_c)

    def regenerate(self, a, b, c, lesion_a=False, lesion_b=False, lesion_c=False):
        """Regenerate fact (a,b,c) = NEURAL SUPERPOSITION of THREE spiking-readout population outputs on their disjoint
        channel blocks + the constant action context. `lesion_*` zeroes one primitive's contribution."""
        a = int(a); b = int(b); c = int(c)
        dab = self.d_a + self.d_b
        out = np.zeros(self.n_in, dtype=np.float64)
        if not lesion_a and a in self.Wa:
            out[:self.d_a] += np.clip(self._elig(self._A_off + a) @ self.Wa[a] + self._anchor_a, 0.0, 1.0)
        if not lesion_b and b in self.Wb:
            out[self.d_a:dab] += np.clip(self._elig(self._B_off + b) @ self.Wb[b] + self._anchor_b, 0.0, 1.0)
        if not lesion_c and c in self.Wc:
            out[dab:self.d_p] += np.clip(self._elig(self._C_off + c) @ self.Wc[c] + self._anchor_c, 0.0, 1.0)
        if self._action_ctx is not None:
            out[self.d_p:] += self._action_ctx
        return np.clip(out, 0.0, 1.0)

    def primitive_slots(self):
        return int(len(self.Wa) + len(self.Wb) + len(self.Wc))             # == P once all primitives seen


# ============================ coverage-preserving ARITY-3 held-out split ============================
def _heldout_split3(K1, K2, K3, m, seed):
    """Choose M held-out (a,b,c) cells such that EVERY primitive still appears in >= 1 TAUGHT fact: greedily hold out
    a cell ONLY if its a-row, b-column AND c-depth each keep >= 1 taught fact after removal. Deterministic per seed.
    Returns (taught_idx, heldout_idx) as class-index lists (class = (a*K2 + b)*K3 + c)."""
    rng = np.random.default_rng(int(seed) + 5551)
    cells = [(a, b, c) for a in range(K1) for b in range(K2) for c in range(K3)]
    rng.shuffle(cells)
    a_left = {a: K2 * K3 for a in range(K1)}              # taught facts remaining that use A-primitive a
    b_left = {b: K1 * K3 for b in range(K2)}
    c_left = {c: K1 * K2 for c in range(K3)}
    held = []
    for (a, b, c) in cells:
        if len(held) >= m:
            break
        if a_left[a] > 1 and b_left[b] > 1 and c_left[c] > 1:              # keep >= 1 taught fact per primitive
            held.append((a, b, c)); a_left[a] -= 1; b_left[b] -= 1; c_left[c] -= 1
    held_idx = sorted((a * K2 + b) * K3 + c for (a, b, c) in held)
    taught_idx = [j for j in range(K1 * K2 * K3) if j not in set(held_idx)]
    return taught_idx, held_idx


def _lesion_localises3(gen, held_attrs, env, held_refs, n_probe=8):
    """Composition-is-neural teeth at arity 3: for a sample of HELD-OUT facts, lesioning primitive a collapses ONLY
    the A block toward its anchor while B and C are spared, and likewise for b and c. Returns the mean block deltas +
    a boolean that each lesion hits its OWN block much harder than the two spared blocks."""
    idxs = list(range(0, len(held_attrs), max(1, len(held_attrs) // n_probe)))[:n_probe]
    da, db, dc, dp = gen.d_a, gen.d_b, gen.d_c, gen.d_p
    dab = da + db
    a_hit, a_sp, b_hit, b_sp, c_hit, c_sp = [], [], [], [], [], []
    for j in idxs:
        a, b, c = held_attrs[j]
        full = gen.regenerate(a, b, c)
        la = gen.regenerate(a, b, c, lesion_a=True)
        lb = gen.regenerate(a, b, c, lesion_b=True)
        lc = gen.regenerate(a, b, c, lesion_c=True)
        a_hit.append(float(np.linalg.norm(full[:da] - la[:da])))                    # A changed by lesion-a
        a_sp.append(float(np.linalg.norm(full[da:dp] - la[da:dp])))                 # B+C spared by lesion-a
        b_hit.append(float(np.linalg.norm(full[da:dab] - lb[da:dab])))              # B changed by lesion-b
        b_sp.append(float(np.linalg.norm(np.concatenate([full[:da], full[dab:dp]])
                                         - np.concatenate([lb[:da], lb[dab:dp]]))))  # A+C spared
        c_hit.append(float(np.linalg.norm(full[dab:dp] - lc[dab:dp])))              # C changed by lesion-c
        c_sp.append(float(np.linalg.norm(full[:dab] - lc[:dab])))                   # A+B spared
    ma_h, ma_s = float(np.mean(a_hit)), float(np.mean(a_sp))
    mb_h, mb_s = float(np.mean(b_hit)), float(np.mean(b_sp))
    mc_h, mc_s = float(np.mean(c_hit)), float(np.mean(c_sp))
    localises = bool(ma_h > 0.05 and mb_h > 0.05 and mc_h > 0.05
                     and ma_h > 5.0 * (ma_s + 1e-9) and mb_h > 5.0 * (mb_s + 1e-9) and mc_h > 5.0 * (mc_s + 1e-9))
    return {"lesionA_Ablock_delta": ma_h, "lesionA_other_delta": ma_s,
            "lesionB_Bblock_delta": mb_h, "lesionB_other_delta": mb_s,
            "lesionC_Cblock_delta": mc_h, "lesionC_other_delta": mc_s, "localises": localises}


# ============================ per-cube ARITY-3 zero-shot driver ============================
def _run_cube(seed, K1, K2, K3, m, d_a, d_b, d_c, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
              conv_tol, conv_max_epochs, batch, gen_epochs, n_draws):
    N = K1 * K2 * K3
    d_p = int(d_a) + int(d_b) + int(d_c)
    n_in = d_p + N_ACT
    chance = 1.0 / N
    referents, attrs = _grid_facts3(K1, K2, K3)

    # --- the held-out split + anti-cheat asserts (disjoint + coverage) ---
    taught_idx, held_idx = _heldout_split3(K1, K2, K3, m, seed)
    taught_set, held_set = set(taught_idx), set(held_idx)
    disjoint = bool(len(taught_set & held_set) == 0 and len(held_set) > 0)
    trained_a = {attrs[j][0] for j in taught_idx}
    trained_b = {attrs[j][1] for j in taught_idx}
    trained_c = {attrs[j][2] for j in taught_idx}
    coverage_ok = bool(all(attrs[j][0] in trained_a and attrs[j][1] in trained_b and attrs[j][2] in trained_c
                           for j in held_idx))
    assert disjoint, "taught and held-out sets must be disjoint and held-out non-empty"
    assert coverage_ok, "every held-out primitive (a AND b AND c) must appear in >= 1 taught fact"

    # --- the world (fresh env; SAME percepts across arms) ---
    env = _make_comp_env3(seed, K1, K2, K3, d_a, d_b, d_c, noise, referents, attrs)
    protos = np.stack([env.proto(referents[j]) for j in range(N)]).astype(np.float64)  # (N, d_p) test-time RULER only
    action_ctx = _action_ctx_const()

    # --- ARM 1: compositional 3-way generator, taught on the TAUGHT set only ---
    cgen = CompositionalGenerator3(gen_k, n_in, d_a, d_b, d_c, K1, K2, K3, gen_hidden, seed, gen_settle, gen_lr,
                                   w_clip, bdsp_wmax=bdsp_wmax, conv_tol=conv_tol, conv_max_epochs=conv_max_epochs)
    fed_comp = []
    for j in taught_idx:
        a, b, c = attrs[j]
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)          # ONLY taught percepts ever drawn
        cgen.learn_fact(a, b, c, np.asarray(Xj, dtype=np.float64).mean(axis=0), action_ctx)
        fed_comp.append(j)

    # --- ARM 2: fixed v2 generator (FLOOR), taught on the TAUGHT set only (class = original fact index) ---
    vgen = GenerativeReplayNetV2(int(gen_k), n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax,
                                 conv_tol=0.05, conv_max_epochs=120, conv_check_every=4, new_mult=3)
    vgen.fit_query_norm()
    vgen_rng = np.random.default_rng(seed + 999)
    fed_v2, seen = [], []
    for j in taught_idx:
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)
        vgen.learn_fact(j, np.asarray(Xj, dtype=np.float64).mean(axis=0), list(seen), gen_epochs, batch, vgen_rng)
        seen.append(j); fed_v2.append(j)

    # --- ARM 3: flat O(N) buffer (FLOOR), taught on the TAUGHT set only ---
    flat = FlatStore(d_p, seed)
    fed_flat = []
    for j in taught_idx:
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)
        flat.learn(j, np.asarray(Xj, dtype=np.float64).mean(axis=0))
        fed_flat.append(j)

    # --- NO-LEAKAGE assert ---
    no_leakage = bool(not (set(fed_comp) & held_set) and not (set(fed_v2) & held_set)
                      and not (set(fed_flat) & held_set))
    assert no_leakage, "a held-out fact index leaked into a training path"

    # --- RECALL: taught (sanity) + held-out (zero-shot), nearest-prototype identity, arm-symmetric ---
    def comp_pred(j):
        a, b, c = attrs[j]
        return _nearest_proto(cgen.regenerate(a, b, c)[:d_p], protos)       # NEURAL 3-way superposition -> identify

    def v2_pred(j):
        return _nearest_proto(vgen.regenerate(j)[:d_p], protos)

    def flat_pred(j):
        return flat.recall_nearest(j, protos)

    comp_seen = _recall_fraction(taught_idx, comp_pred, protos)
    comp_held = _recall_fraction(held_idx, comp_pred, protos)
    v2_seen = _recall_fraction(taught_idx, v2_pred, protos)
    v2_held = _recall_fraction(held_idx, v2_pred, protos)
    flat_seen = _recall_fraction(taught_idx, flat_pred, protos)
    flat_held = _recall_fraction(held_idx, flat_pred, protos)

    # --- regeneration fidelity (cosine to the TRUE composed prototype) diagnostic (ruler only) ---
    comp_held_cos = float(np.mean([_cos(cgen.regenerate(*attrs[j])[:d_p], protos[j]) for j in held_idx]))
    comp_seen_cos = float(np.mean([_cos(cgen.regenerate(*attrs[j])[:d_p], protos[j]) for j in taught_idx]))
    v2_held_cos = float(np.mean([_cos(vgen.regenerate(j)[:d_p], protos[j]) for j in held_idx]))

    # --- composition-is-NEURAL teeth: 3-way lesion localisation on the HELD-OUT facts ---
    held_refs = [referents[j] for j in held_idx]
    held_attrs = [attrs[j] for j in held_idx]
    lesion = _lesion_localises3(cgen, held_attrs, env, held_refs, n_probe=min(8, len(held_idx)))

    # --- honest-negative diagnostic: is a comp miss low-cos (interference) or high-cos wrong-nearest (collision)? ---
    held_miss_lowcos = 0; held_miss_collision = 0
    for j in held_idx:
        if comp_pred(j) != j:
            if _cos(cgen.regenerate(*attrs[j])[:d_p], protos[j]) < 0.85:
                held_miss_lowcos += 1
            else:
                held_miss_collision += 1

    return {
        "arity": 3, "K1": K1, "K2": K2, "K3": K3, "N": N, "P": K1 + K2 + K3, "chance": chance,
        "n_in": n_in, "d_p": d_p, "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "held_out_idx": held_idx, "held_out_attrs": [list(attrs[j]) for j in held_idx],
        "taught_heldout_disjoint": disjoint, "every_heldout_primitive_seen_in_taught": coverage_ok,
        "no_leakage_heldout_never_trained": no_leakage,
        # THE HEADLINE: arity-3 zero-shot held-out recall, compositional vs the two floors
        "compositional_heldout_recall": comp_held,
        "noncompositional_v2_heldout_recall": v2_held,
        "flat_heldout_recall": flat_held,
        "compositional_seen_recall": comp_seen,
        "noncompositional_v2_seen_recall": v2_seen,
        "flat_seen_recall": flat_seen,
        "compositional_heldout_cos": comp_held_cos, "compositional_seen_cos": comp_seen_cos,
        "noncompositional_v2_heldout_cos": v2_held_cos,
        "lesion_localises_heldout": lesion,
        "compositional_stored_raw_patterns": int(cgen._stored_raw_patterns),
        "compositional_used_ruler": bool(cgen._used_ruler),
        "heldout_miss_lowcos_interference": held_miss_lowcos,
        "heldout_miss_highcos_collision": held_miss_collision,
    }


# ============================ orchestration (arity 3 + arity-2 baseline in-run) ============================
def run(seed, K, held3, held2, d_a, d_b, d_c, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
        conv_tol, conv_max_epochs, batch, gen_epochs, n_draws):
    n_in3 = int(d_a) + int(d_b) + int(d_c) + N_ACT
    N3 = K * K * K
    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in3, N3, seed, max(120, 6 * N3), 20,
                                                               0.5, w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()

    # --- ARITY 3 (K x K x K, N = K^3) ---
    print(f"\n{'=' * 90}\n# SEED {seed}  ARITY-3 CUBE {K}x{K}x{K} (N={N3}, P={3*K}, held_out={held3})\n{'=' * 90}",
          flush=True)
    a3 = _run_cube(seed, K, K, K, held3, d_a, d_b, d_c, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                   bdsp_wmax, conv_tol, conv_max_epochs, batch, gen_epochs, n_draws)
    print(f"  [arity3 {K}x{K}x{K} N={N3}] held-out recall: comp {a3['compositional_heldout_recall']:.2f} | "
          f"v2 {a3['noncompositional_v2_heldout_recall']:.2f} | flat {a3['flat_heldout_recall']:.2f} "
          f"(chance {a3['chance']:.4f}) | seen(comp) {a3['compositional_seen_recall']:.2f} | "
          f"held-cos {a3['compositional_heldout_cos']:.3f} | lesion-loc "
          f"{a3['lesion_localises_heldout'].get('localises')}", flush=True)

    # --- ARITY 2 BASELINE (K x K, N = K^2) IN THE SAME RUN (reuse the banked arity-2 grid driver) ---
    print(f"\n{'=' * 90}\n# SEED {seed}  ARITY-2 BASELINE {K}x{K} (N={K*K}, held_out={held2})\n{'=' * 90}", flush=True)
    a2 = _run_arity2_grid(seed, K, K, held2, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                          bdsp_wmax, conv_tol, conv_max_epochs, batch, gen_epochs, n_draws)
    print(f"  [arity2 {K}x{K} N={K*K}] held-out recall: comp {a2['compositional_heldout_recall']:.2f} | "
          f"v2 {a2['noncompositional_v2_heldout_recall']:.2f} | flat {a2['flat_heldout_recall']:.2f} "
          f"(chance {a2['chance']:.4f}) | seen(comp) {a2['compositional_seen_recall']:.2f}", flush=True)

    return {"seed": seed, "K": K, "d_a": d_a, "d_b": d_b, "d_c": d_c, "n_in3": n_in3,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"K": K, "held3": held3, "held2": held2, "d_a": d_a, "d_b": d_b, "d_c": d_c, "noise": noise,
                       "gen_hidden": gen_hidden, "gen_k": gen_k, "gen_settle": gen_settle, "gen_lr": gen_lr,
                       "w_clip": w_clip, "bdsp_wmax": bdsp_wmax, "conv_tol": conv_tol,
                       "conv_max_epochs": conv_max_epochs, "batch": batch, "gen_epochs": gen_epochs,
                       "n_draws": n_draws, "frozen_hidden": True},
            "arity3": a3, "arity2": a2}


# ============================ verdict ============================
def _verdict(result):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    a3 = result["arity3"]; a2 = result["arity2"]
    v = Verdict("teacher-loop ARITY-3 zero-shot composition (3-way neural superposition)", chance=None)

    N3 = a3["N"]; chance3 = a3["chance"]
    comp3 = a3["compositional_heldout_recall"]; v2_3 = a3["noncompositional_v2_heldout_recall"]
    flat3 = a3["flat_heldout_recall"]; seen3 = a3["compositional_seen_recall"]
    lesion_ok = bool(a3["lesion_localises_heldout"].get("localises"))
    disjoint = a3["taught_heldout_disjoint"]; cover = a3["every_heldout_primitive_seen_in_taught"]
    noleak = a3["no_leakage_heldout_never_trained"]
    not_buffer = bool(a3["compositional_stored_raw_patterns"] == 0)
    no_ruler = bool(not a3["compositional_used_ruler"])

    attributable_to("[arity3] zero-shot held-out: 3-way composing vs v2 memorizing", comp3, v2_3)
    attributable_to("[arity3] zero-shot held-out: 3-way composing vs flat instance store", comp3, flat3)

    beats_v2 = bool(comp3 >= v2_3 + 0.30)
    beats_flat = bool(comp3 >= flat3 + 0.30)
    comp_ok = bool(comp3 >= 0.5)
    seen_ok = bool(seen3 >= 0.85)

    v.require("[arity3] compositional HELD-OUT recall >= 0.5 (3-way zero-shot composition)", comp_ok, expect=True,
              note=f"comp {comp3:.2f} (chance {chance3:.4f}) N={N3}")
    v.require("[arity3] compositional held-out >= v2 held-out + 0.30 (composes, not memorizes)", beats_v2,
              expect=True, note=f"comp {comp3:.2f} vs v2 {v2_3:.2f}")
    v.require("[arity3] compositional held-out >= flat held-out + 0.30 (beats the instance floor)", beats_flat,
              expect=True, note=f"comp {comp3:.2f} vs flat {flat3:.2f}")
    v.require("[arity3] compositional SEEN (taught) recall >= 0.85 (sanity)", seen_ok, expect=True,
              note=f"seen {seen3:.2f}")
    v.require("[arity3] composition NEURAL on held-out (3-way lesion localises)", lesion_ok, expect=True,
              note=f"A:{a3['lesion_localises_heldout'].get('lesionA_Ablock_delta',0):.2f} "
                   f"B:{a3['lesion_localises_heldout'].get('lesionB_Bblock_delta',0):.2f} "
                   f"C:{a3['lesion_localises_heldout'].get('lesionC_Cblock_delta',0):.2f}")
    v.require("[arity3] taught/held-out DISJOINT", bool(disjoint), expect=True)
    v.require("[arity3] every held-out primitive seen in >= 1 taught fact", bool(cover), expect=True)
    v.require("[arity3] NO leakage (held-out never trained)", bool(noleak), expect=True)
    v.require("[arity3] 0 stored raw patterns (composes, not a buffer)", not_buffer, expect=True)
    v.require("[arity3] generator never read the ruler", no_ruler, expect=True)

    # baseline sanity: arity-2 held-out (the ~1.00 to compare) is high in this same run
    comp2 = a2["compositional_heldout_recall"]
    v.floor("[arity2 baseline] compositional held-out recall (the ~1.00 reference)", comp2, floor=0.5)

    v.require("(seed) substrate byte-identical", bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    arity3_go = bool(comp_ok and beats_v2 and beats_flat and seen_ok and lesion_ok and disjoint and cover
                     and noleak and not_buffer and no_ruler)
    go = bool(arity3_go and result["substrate_byte_identical"] and result["sim_diff_empty"])
    decision = v.decide(go=go)
    return {
        "arity3_go": arity3_go,
        "arity3": {
            "N": N3, "chance": chance3, "held_out_n": a3["held_out_n"], "taught_n": a3["taught_n"],
            "compositional_heldout_recall": comp3, "noncompositional_v2_heldout_recall": v2_3,
            "flat_heldout_recall": flat3, "compositional_seen_recall": seen3,
            "compositional_minus_v2": float(comp3 - v2_3), "compositional_minus_flat": float(comp3 - flat3),
            "compositional_heldout_cos": a3["compositional_heldout_cos"],
            "lesion_localises": lesion_ok,
            "heldout_miss_lowcos_interference": a3["heldout_miss_lowcos_interference"],
            "heldout_miss_highcos_collision": a3["heldout_miss_highcos_collision"],
        },
        "arity2_baseline": {
            "N": a2["N"], "chance": a2["chance"], "held_out_n": a2["held_out_n"],
            "compositional_heldout_recall": comp2,
            "noncompositional_v2_heldout_recall": a2["noncompositional_v2_heldout_recall"],
            "flat_heldout_recall": a2["flat_heldout_recall"],
            "compositional_seen_recall": a2["compositional_seen_recall"],
        },
        "substrate_byte_identical": result["substrate_byte_identical"],
        "sim_diff_empty": result["sim_diff_empty"], **decision,
    }


def _one_seed(a, seed):
    result = run(seed, a.K, a.held_out, a.held_out2, a.d_a, a.d_b, a.d_c, a.noise, a.gen_hidden, a.gen_k,
                 a.gen_settle, a.gen_lr, a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.batch,
                 a.gen_epochs, a.n_draws)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop ARITY-3 ZERO-SHOT COMPOSITION: regenerate + recall "
                                             "never-taught (a,b,c) facts from THREE primitives seen in OTHER facts "
                                             "(3-way neural superposition), vs a fixed generator + flat store "
                                             "(the floors); arity-2 baseline measured in the same run.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--K", type=int, default=4, help="per-attribute vocab size (arity3 cube = KxKxK; arity2 = KxK)")
    ap.add_argument("--held-out", type=int, default=16, help="arity-3 held-out fact count (coverage-preserving)")
    ap.add_argument("--held-out2", type=int, default=4, help="arity-2 baseline held-out fact count")
    ap.add_argument("--d-a", type=int, default=8, help="attribute-A block dim")
    ap.add_argument("--d-b", type=int, default=8, help="attribute-B block dim")
    ap.add_argument("--d-c", type=int, default=8, help="attribute-C block dim (the 3rd channel)")
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--gen-hidden", type=int, default=96, help="FIXED generator reservoir size (H_gen)")
    ap.add_argument("--gen-k", type=int, default=96, help="FIXED query address width (>= N so class codes are full-rank)")
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--conv-tol", type=float, default=0.02, help="compositional primitive-engram NLMS convergence tol")
    ap.add_argument("--conv-max-epochs", type=int, default=200)
    ap.add_argument("--gen-epochs", type=int, default=16, help="v2 (floor) self-replay epochs/fact")
    ap.add_argument("--batch", type=int, default=16)
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
        print("\n" + "#" * 100 + f"\n# SEED {s}  K={a.K} held3={a.held_out} held2={a.held_out2} gen_H={a.gen_hidden} "
              f"gen_k={a.gen_k}\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s)
        summary = {"probe": "teacher_loop_arity3_composition", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "K": a.K, "held_out": a.held_out, "held_out2": a.held_out2,
                   "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        a3 = rv["arity3"]; a2 = rv["arity2_baseline"]
        print("\n" + "=" * 100, flush=True)
        print(f"[arity3] seed {s}: N={a3['N']} held={a3['held_out_n']} | HELD-OUT recall comp "
              f"{a3['compositional_heldout_recall']:.2f} vs v2 {a3['noncompositional_v2_heldout_recall']:.2f} vs flat "
              f"{a3['flat_heldout_recall']:.2f} (chance {a3['chance']:.4f}) | seen(comp) "
              f"{a3['compositional_seen_recall']:.2f} | comp-v2 +{a3['compositional_minus_v2']:.2f} comp-flat "
              f"+{a3['compositional_minus_flat']:.2f} | lesion {a3['lesion_localises']} | arity3_GO {rv['arity3_go']}",
              flush=True)
        print(f"[arity2 baseline] seed {s}: N={a2['N']} held={a2['held_out_n']} | HELD-OUT recall comp "
              f"{a2['compositional_heldout_recall']:.2f} vs v2 {a2['noncompositional_v2_heldout_recall']:.2f} vs flat "
              f"{a2['flat_heldout_recall']:.2f} (chance {a2['chance']:.4f})", flush=True)
        print(f"[arity3] seed {s} byte-id {rv['substrate_byte_identical']} sim-clean {rv['sim_diff_empty']} | "
              f"VERDICT {rv['status']}", flush=True)
        print(f"[arity3] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        comp3 = [p["verdict"]["arity3"]["compositional_heldout_recall"] for p in per_seed]
        v2_3 = [p["verdict"]["arity3"]["noncompositional_v2_heldout_recall"] for p in per_seed]
        flat3 = [p["verdict"]["arity3"]["flat_heldout_recall"] for p in per_seed]
        seen3 = [p["verdict"]["arity3"]["compositional_seen_recall"] for p in per_seed]
        comp2 = [p["verdict"]["arity2_baseline"]["compositional_heldout_recall"] for p in per_seed]
        agg = {"probe": "teacher_loop_arity3_composition_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "K": a.K, "held_out": a.held_out, "held_out2": a.held_out2,
               "go_count": go_n, "n_seeds": len(seeds),
               "arity3": {
                   "N": per_seed[0]["verdict"]["arity3"]["N"],
                   "chance": per_seed[0]["verdict"]["arity3"]["chance"],
                   "compositional_heldout_recall_mean": float(np.nanmean(comp3)),
                   "compositional_heldout_recall_per_seed": [float(x) for x in comp3],
                   "noncompositional_v2_heldout_recall_mean": float(np.nanmean(v2_3)),
                   "flat_heldout_recall_mean": float(np.nanmean(flat3)),
                   "compositional_seen_recall_mean": float(np.nanmean(seen3)),
               },
               "arity2_baseline": {
                   "N": per_seed[0]["verdict"]["arity2_baseline"]["N"],
                   "chance": per_seed[0]["verdict"]["arity2_baseline"]["chance"],
                   "compositional_heldout_recall_mean": float(np.nanmean(comp2)),
                   "compositional_heldout_recall_per_seed": [float(x) for x in comp2],
               },
               "per_seed": per_seed}
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[arity3 AGG] GO {go_n}/{len(seeds)}", flush=True)
        print(f"   arity3: N={agg['arity3']['N']} | HELD-OUT comp "
              f"{agg['arity3']['compositional_heldout_recall_mean']:.2f} vs v2 "
              f"{agg['arity3']['noncompositional_v2_heldout_recall_mean']:.2f} vs flat "
              f"{agg['arity3']['flat_heldout_recall_mean']:.2f} (chance {agg['arity3']['chance']:.4f}) | seen "
              f"{agg['arity3']['compositional_seen_recall_mean']:.2f} | per-seed "
              f"{agg['arity3']['compositional_heldout_recall_per_seed']}", flush=True)
        print(f"   arity2 baseline: N={agg['arity2_baseline']['N']} | HELD-OUT comp "
              f"{agg['arity2_baseline']['compositional_heldout_recall_mean']:.2f} | per-seed "
              f"{agg['arity2_baseline']['compositional_heldout_recall_per_seed']}", flush=True)
        print(f"[arity3 AGG] wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
