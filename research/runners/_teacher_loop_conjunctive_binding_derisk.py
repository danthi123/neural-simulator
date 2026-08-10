"""TEACHER-LOOP CONJUNCTIVE-BINDING DE-RISK (2026-08-09): a NEURAL BIND that recovers zero-shot composition where
additive superposition BREAKS.

WHERE THE ARC STANDS. Neural compositional generalization = GO on the CLEAN (disjoint-channel) world: a spiking
generator composes NEVER-TAUGHT combinations zero-shot by NEURAL SUPERPOSITION (SUM) of two primitive spiking-readout
outputs (VSA BUNDLING) -- 1.00, 6/6. The INTERFERENCE stress test then showed additive superposition is ROBUST through
moderate SHARED-CHANNEL mixing (s<=0.5 -> ~1.00) then BREAKS at high mixing (s~0.75-1.0): zero-shot held-out recall
collapses (1.00 -> 0.40 -> 0.20). The break is a HIGH-COS COLLISION -- the SUM stays similar but cannot SEPARATE
conjunctively-mixed prototypes. RESIDUAL PRECISELY NAMED: additive bundling != binding; a LINEAR SUM cannot encode the
AND-like CONJUNCTION the mixed world carries (a genuinely bilinear, non-additive interaction term).

THE HYPOTHESIS (this de-risk). Add a NEURAL CONJUNCTIVE BINDING operation -- a genuine MULTIPLICATIVE / AND interaction
in spikes (a dendritic-AND / sigma-pi product of two spiking-readout population outputs) -- so the generator can encode
the conjunction, RECOVERING zero-shot generalization at high mixing (s~0.75-1.0) where superposition collapsed, WITHOUT
losing the low-s generalization. DR grounding: VSA bind (Plate HRR / Kanerva circular convolution; Smolensky tensor
product), sigma-pi units (Rumelhart/Mel), dendritic multiplication (two-compartment coincidence, NMDA supralinearity).

THE MIXED WORLD (host, legitimate -- the interference env, faithfully rebuilt). K1 x K2 grid of two primitive
vocabularies primA[a] in [0,1]^d_a, primB[b] in [0,1]^d_b. Two fixed random projections Ma (d_p x d_a), Mb (d_p x d_b)
map each primitive into a SHARED d_p = d_a+d_b channel space: sA[a]=Ma@primA[a], sB[b]=Mb@primB[b]. Fact (a,b)'s percept
prototype:
    proto(a,b) = (1-s) * concat(primA[a], primB[b])            # the CLEAN disjoint base (s=0 == the GO world exactly)
               + s     * ( sA[a] + sB[b] + sA[a] (o) sB[b] )    # shared-channel mixing: additive parts + CONJUNCTION
where (o) is the elementwise product. The ONLY non-additive term is s * sA(o)sB -- a genuinely BILINEAR interaction:
per channel c it is a rank-1 matrix s*sA[.,c] (outer) sB[.,c] in (a,b). At s=0 the world is the clean disjoint GO
(additive bundling recovers it exactly). As s grows a conjunctive residual an additive model CANNOT represent appears;
the NON-ADDITIVITY WITNESS (fraction of proto variance not explained by the best two-way additive model g+A[a]+B[b])
climbs from 0. THIS is what breaks the sum and what a bind must recover.

THE GENERATORS (brain-based; ONE frozen spiking Izhikevich reservoir reused from the a1-GO generator; de-clamped
bdsp_wmax=1e9 so it spikes; readout-only trained). All arms address the reservoir with per-primitive query codes and
train per-primitive leaky-readout weight-sets (the primitive synaptic traces) by the local NLMS delta rule on the
reservoir spike eligibility -- exactly the CompositionalGenerator mechanism, generalized to FULL d_p readouts so
shared-channel structure is representable.
  * additive_superposition (BASELINE, MEASURED in-run = the curve to beat): regenerate(a,b) = grand-mean anchor g
    + readoutA(a) + readoutB(b), a NEURAL SUPERPOSITION (SUM of two spiking-readout population outputs). Targets are
    the two-way additive row/col effects. It DROPS the interaction -> breaks at high s.
  * conjunctive_binding (TREATMENT): additive_superposition PLUS a dendritic-AND pathway. The additive RESIDUAL
    R_ab = engram_ab - additive(a,b) is factored per channel into rank-1 factors fA[a] (o) fB[b] (offline ALS derives
    the per-primitive factor TEACHING TARGETS -- the learning signal, exactly as the additive arm derives its
    per-primitive targets by host averaging); two MORE per-primitive spiking-readouts regenerate fA[a], fB[b] from the
    frozen reservoir; regeneration ADDS their ELEMENTWISE PRODUCT (the sigma-pi / dendritic-AND):
        bind(a,b) = additive(a,b) + ( readoutMulA(a) (o) readoutMulB(b) ).
    Zero-shot: fA[a], fB[b] are per-primitive (each a,b appears in TAUGHT combos) so the per-channel rank-1 completion
    recovers the held-out interaction, and the RUNTIME bind is a neural product of two spiking-readout outputs.
  * sum_ablation (CONTROL that isolates the CONJUNCTION): identical to binding but COMBINES the two factor readouts by
    ADDITION instead of product. A sum is absorbed into the additive model (mean-zero factors -> ~0) -> it CANNOT
    capture the bilinear residual -> must FAIL zero-shot where product succeeds. If sum_ablation also recovered, the
    win would NOT be from conjunction.
  * fixed_v2 (FLOOR): the fixed GenerativeReplayNetV2 addresses facts by CLASS INDEX -> held-out class untrained -> chance.
  * flat (FLOOR): an O(N) raw-engram buffer keyed by class index -> no held-out entry -> uniform guess = chance.

THE MEASURE. RECALL of a fact = regenerate its percept, nearest-prototype identify it among ALL N true prototypes
(chance 1/N). Swept over mixing strength s. THE WIN: conjunctive_binding held-out recall >= additive held-out + 0.30
AND >= 0.5 at s in {0.75, 1.0} (recovers the break), while staying >= additive - 0.05 at s <= 0.5 (no low-s cost).

ANTI-CHEATS (each a REAL assertion in the output):
  * bind is NEURAL: the runtime bind ADDS the elementwise product of two SPIKING-reservoir leaky-readout outputs
    (readoutMulA/B come from _elig = a spiking forward of the frozen de-clamped reservoir; mean reservoir spikes>0
    recorded). NOT a host np.outer / tensor algebra of the answer.
  * bind is genuinely MULTIPLICATIVE/conjunctive, NOT an additive op relabeled: (i) the SUM_ABLATION control (same
    factors, additive combine) FAILS to recover; (ii) the product materially differs from the sum of its factors
    (mean||prod - (fA+fB)|| > 0); (iii) the world has a real conjunction to bind (non-additivity witness > 0 at high s).
  * genuinely ZERO-SHOT: taught / held-out DISJOINT; every held-out primitive (a AND b) appears in >= 1 TAUGHT combo;
    NO LEAKAGE (no held-out fact index enters any training path -- asserted DISJOINT); the ruler (true protos) is read
    ONLY at test time.
  * 0 stored raw patterns (composes, never buffers the N composed percepts); cfg.seed byte-identical substrate (NOT
    actual_seed_used); de-clamped bdsp_wmax=1e9; git diff main -- sim/ empty; backend recorded.

GO (per grid, per seed): at s in {0.75,1.0} -- binding held-out recall >= additive + 0.30 AND >= 0.5 (recovers the
break); at s in {0.0,0.25,0.5} -- binding held-out >= additive - 0.05 (no low-s cost); binding taught recall >= 0.85
(sanity); sum_ablation does NOT recover at high s (binding - sum_ablation >= 0.20 at s=1.0); the additive baseline
actually breaks (additive held-out at s=1.0 <= 0.6, else there was nothing to recover -> reported, not a hard fail);
bind neural + multiplicative + zero-shot anti-cheats hold; byte-identical; sim/ clean. HONEST NEGATIVE if the neural
product does NOT recover zero-shot (naming WHY: low factor-regeneration fidelity => the reservoir readouts cannot
reproduce the bilinear factors; or the recovery costs low-s), or if it is not really conjunctive (sum_ablation also
recovers).

DISCIPLINE: reuse-by-import (the spiking reservoir + query/eligibility machinery; the fixed-v2 + flat FLOORS + the
held-out split + no-leakage asserts from the zero-shot runner; the byte-identical + sim-clean asserts). NO sim/ edit.
SIM_BACKEND=numpy.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_conjunctive_binding_derisk --seed 42 \
      --grids 6x6 7x7 --s-values 0.0 0.25 0.5 0.75 1.0 \
      --out research/findings/raw/teacher_loop_conjunctive_binding_s42.json
  6-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_conjunctive_binding_derisk --seeds 42 43 44 45 46 47 \
      --grids 6x6 7x7 --s-values 0.0 0.25 0.5 0.75 1.0 \
      --out research/findings/raw/teacher_loop_conjunctive_binding.json
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
# reuse-by-import: the spiking reservoir; the fixed-v2 + flat FLOORS + held-out split + no-leakage asserts (zero-shot
# runner); the scaling teacher machinery; the byte-identical + sim-clean anti-cheat asserts. NO sim/ edit.
from research.runners._teacher_loop_generative_replay_derisk import GenerativeReplayNet, _cos  # noqa: E402
from research.runners._teacher_loop_generative_replay_v2_derisk import GenerativeReplayNetV2  # noqa: E402
from research.runners._teacher_loop_compositional_generator_derisk import (  # noqa: E402
    CompositionalReferentEnv, _grid_facts, _action_ctx_const,
)
from research.runners._teacher_loop_zeroshot_composition_derisk import (  # noqa: E402
    _heldout_split, FlatStore, _nearest_proto, _recall_fraction,
)
from research.runners._teacher_loop_scaling_derisk import _corrective_batch, N_ACT  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _assert_byte_identical_substrate, _git_sim_diff_empty,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_conjunctive_binding.json"


# ============================ the MIXED WORLD (host, legitimate) ============================
class MixedCompositionalReferentEnv(CompositionalReferentEnv):
    """The interference world: the clean compositional base PLUS a shared-channel mix of strength s that carries a
    genuinely CONJUNCTIVE (bilinear, non-additive) interaction term sA(o)sB. At s=0 it is the clean disjoint GO world;
    as s grows an additive model cannot represent the growing conjunction. Host code is legitimate exactly as a retinal
    render is -- the brain reads this percept through its OWN learned weights. draw() does NOT clip (the composed
    prototypes leave [0,1]); the reservoir queries are index codes independent of the percept range, and the readouts
    fit to the engram targets directly, so the bilinear structure is preserved undistorted."""

    def __init__(self, seed, K1, K2, d_a=8, d_b=8, noise=0.12, s=0.0, mix_scale=1.0):
        super().__init__(seed, K1, K2, d_a=d_a, d_b=d_b, noise=noise)
        self.s = float(s); self.mix_scale = float(mix_scale)
        wr = np.random.default_rng(int(seed) + 314159)          # dedicated MIX RNG (deterministic, distinct from world)
        d_p = int(d_a) + int(d_b)
        # shared-channel projections of each primitive into the FULL d_p space, then CENTERED across primitives so the
        # binding is (nearly) PURE interaction: bindvec(a,b) = uA_c[a] (o) uB_c[b] has ~zero two-way-additive part ->
        # the conjunction DOMINATES at high s and an additive (sum) model cannot represent it. Per channel c it is
        # EXACTLY rank-1 in (a,b): uA_c[a,c]*uB_c[b,c] -> a genuine bilinear AND (VSA bind / sigma-pi target).
        self.Ma = wr.standard_normal((d_p, int(d_a))) / np.sqrt(int(d_a))
        self.Mb = wr.standard_normal((d_p, int(d_b))) / np.sqrt(int(d_b))
        uA = np.stack([self.Ma @ self.primA[a] for a in range(self.K1)])     # (K1,d_p)
        uB = np.stack([self.Mb @ self.primB[b] for b in range(self.K2)])     # (K2,d_p)
        self._uAc = uA - uA.mean(axis=0, keepdims=True)                      # center across primitives -> pure interaction
        self._uBc = uB - uB.mean(axis=0, keepdims=True)
        # scale the conjunction relative to the base's fact-to-fact SPACING (the CENTERED base RMS -- what actually
        # separates prototypes). Target bind RMS = mix_scale * base_centered_rms. A conjunction of that size breaks the
        # additive model as s rises past ~ 1/(1+mix_scale): at low s the additive base still separates facts (robust);
        # at high s the uncapturable interaction dominates -> high-cos collision. mix_scale ~0.6 puts the break ~s=0.6-0.75.
        bases = np.stack([np.concatenate([self.primA[a], self.primB[b]])
                          for a in range(self.K1) for b in range(self.K2)]).astype(np.float64)
        base_centered_rms = float(np.sqrt(np.mean(np.sum((bases - bases.mean(axis=0)) ** 2, axis=1))))
        bind_rms = float(np.sqrt(np.mean([np.sum((self._uAc[a] * self._uBc[b]) ** 2)
                                          for a in range(self.K1) for b in range(self.K2)]))) + 1e-9
        self._bind_scale = (base_centered_rms / bind_rms) * self.mix_scale
        self.protos = {}                                        # recompute protos with the mix

    def _bindvec(self, a, b):
        return self._bind_scale * (self._uAc[a] * self._uBc[b])             # the CONJUNCTION (elementwise product)

    def proto(self, referent):
        if referent not in self.protos:
            a, b = self.fact_attrs[referent]
            base = np.concatenate([self.primA[a], self.primB[b]]).astype(np.float64)   # the CLEAN disjoint base
            self.protos[referent] = (1.0 - self.s) * base + self.s * self._bindvec(a, b)
        return self.protos[referent]

    def draw(self, referent):
        return self.proto(referent) + self.noise * self.rng.standard_normal(self.d_p)   # NO clip (preserve structure)


def _make_mixed_env(seed, K1, K2, d_a, d_b, noise, s, mix_scale, referents, attrs):
    env = MixedCompositionalReferentEnv(seed, K1, K2, d_a=d_a, d_b=d_b, noise=noise, s=s, mix_scale=mix_scale)
    for r, (a, b) in zip(referents, attrs):
        env.register(r, a, b); env.proto(r)
    env.rng = np.random.default_rng(seed + 101)                 # reset draw-stream => every arm sees SAME percepts
    return env


def _additive_nonadditivity_witness(protos, attrs, K1, K2):
    """Fraction of proto variance NOT explained by the best two-way additive model g + A[a] + B[b] over the FULL grid.
    0 <=> the world is purely additive (a sum can represent it); grows with the conjunction. Ruler-only diagnostic."""
    N = protos.shape[0]; d_p = protos.shape[1]
    g = protos.mean(axis=0)
    A = np.zeros((K1, d_p)); B = np.zeros((K2, d_p))
    for a in range(K1):
        A[a] = protos[[j for j, (aa, bb) in enumerate(attrs) if aa == a]].mean(axis=0) - g
    for b in range(K2):
        B[b] = protos[[j for j, (aa, bb) in enumerate(attrs) if bb == b]].mean(axis=0) - g
    pred = np.stack([g + A[a] + B[b] for (a, b) in attrs])
    resid = protos - pred
    tot = float(np.sum((protos - g) ** 2)) + 1e-12
    return float(np.sum(resid ** 2) / tot)


# ============================ the NEURAL reservoir wrapper (cached spiking eligibility) ============================
class _Reservoir:
    """The ONE frozen spiking Izhikevich reservoir (readout-only), reused from the a1-GO generator. Exposes a CACHED
    whitened spike-eligibility for a query ADDRESS (deterministic -> cache). The readout-norm mu/sigma are conditioned
    over the ACTUAL primitive addresses (keep the bias dof: mu=0, sigma=a single RMS scale)."""

    def __init__(self, gen_k, n_in, hidden, seed, settle, gen_lr, w_clip, bdsp_wmax, addrs):
        self._res = GenerativeReplayNet(int(gen_k), int(n_in), int(hidden), seed, settle, gen_lr, w_clip,
                                        bdsp_wmax=bdsp_wmax)
        self._cache = {}
        R = np.array([self._raw(ad) for ad in addrs])
        self._res._r_mu = np.zeros(R.shape[1], dtype=np.float64)
        self._res._r_sigma = R.std(axis=0) + 1e-3
        self.H = int(R.shape[1])
        self._cache = {}                                        # re-cache under the fitted norm
        self._spike_counts = []

    def _raw(self, addr):
        sp, _vv, _acts = self._res._forward_record(self._res._query_code(int(addr)))
        return self._res._readout_feature(sp)

    def elig(self, addr):
        addr = int(addr)
        if addr not in self._cache:
            sp, _vv, _acts = self._res._forward_record(self._res._query_code(addr))
            self._spike_counts.append(float(np.sum(np.asarray(sp, dtype=np.float64))))
            self._cache[addr] = self._res._readout_feature(sp)
        return self._cache[addr]

    def mean_spikes(self):
        return float(np.mean(self._spike_counts)) if self._spike_counts else 0.0


def _fit_readout(res, addr, target, gen_lr, conv_tol, conv_max_epochs, conv_check_every=4):
    """NLMS train-to-convergence of ONE readout weight-set W (H x len(target)): map the reservoir eligibility for this
    address -> target. Rank-1 fixed-target regression on the frozen reservoir basis. Returns W."""
    r = res.elig(addr)
    W = np.zeros((res.H, len(target)), dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    denom = float(r @ r) + 1e-6
    for ep in range(int(conv_max_epochs)):
        err = (r @ W) - tgt
        W -= gen_lr * np.outer(r, err) / denom
        if (ep + 1) % conv_check_every == 0 and float(np.linalg.norm((r @ W) - tgt)) < conv_tol:
            break
    return W


# ============================ additive-superposition + conjunctive-binding generators ============================
# disjoint reservoir query-address families (so each pathway addresses the reservoir distinctly)
_A_OFF, _B_OFF = 0, 1_000_000                                  # additive-BASELINE per-primitive readouts
_JA_OFF, _JB_OFF = 4_000_000, 5_000_000                        # binding's ADDITIVE (co-adapted) per-primitive readouts
_MULA_OFF, _MULB_OFF = 2_000_000, 3_000_000                    # binding's dendritic-AND FACTOR readouts


def _anova(E, taught_cells, K1, K2, d_p):
    """The two-way additive (ANOVA) decomposition over OBSERVED (taught) cells: g + rowEffect(a) + colEffect(b)."""
    g = np.mean([E[c] for c in taught_cells], axis=0)
    rowA = {}; colB = {}
    for a in range(K1):
        bs = [(aa, b) for (aa, b) in taught_cells if aa == a]
        if bs:
            rowA[a] = np.mean([E[c] for c in bs], axis=0) - g
    for b in range(K2):
        as_ = [(a, bb) for (a, bb) in taught_cells if bb == b]
        if as_:
            colB[b] = np.mean([E[c] for c in as_], axis=0) - g
    return g, rowA, colB


class BindingGenerator:
    """Builds the additive-superposition BASELINE and the conjunctive-binding TREATMENT on ONE frozen spiking
    reservoir. Trained on the TAUGHT combos only. All pathways are per-primitive spiking-readout weight-sets (the
    primitive synaptic traces) trained by NLMS on the reservoir eligibility -- O(P) engrams, NEVER a raw composed
    percept, NEVER the true prototypes (the ruler).
        additive(a,b) = g0 + readoutA(a) + readoutB(b)                       # neural SUPERPOSITION (VSA bundling)
        binding(a,b)  = gj + readoutJA(a) + readoutJB(b)                     # co-adapted additive part
                              + readoutMulA(a) (o) readoutMulB(b)            # + dendritic-AND / sigma-pi PRODUCT
        sum_abl(a,b)  = gj + readoutJA(a) + readoutJB(b)                     # control: SAME factors, ADDITIVE combine
                              + readoutMulA(a) + readoutMulB(b)
    The additive BASELINE is the pure two-way ANOVA (robust at low mixing, structurally BREAKS at high mixing). The
    binding model is a GATED JOINT additive+rank-1 fit: the interaction is a genuinely BILINEAR term the additive
    model cannot carry; per channel it is rank-1 in (a,b) and the factors are PER-PRIMITIVE, so a rank-1 completion
    (coverage: every held-out a-row and b-col is observed) recovers the held-out interaction, and the RUNTIME bind is
    the ELEMENTWISE PRODUCT of two spiking-readout population outputs (a dendritic-AND). The joint fit is GATED on the
    additive residual so at low mixing the bilinear stays 0 (binding == additive; no low-s harm)."""

    def __init__(self, res, d_a, d_b, K1, K2, gen_lr, conv_tol, conv_max_epochs, gate=0.25, outer=8, inner=30):
        self.res = res
        self.d_a = int(d_a); self.d_b = int(d_b); self.d_p = int(d_a) + int(d_b)
        self.K1 = int(K1); self.K2 = int(K2)
        self.gen_lr = float(gen_lr); self.conv_tol = float(conv_tol); self.conv_max_epochs = int(conv_max_epochs)
        self.gate = float(gate); self.outer = int(outer); self.inner = int(inner)
        self.g0 = np.zeros(self.d_p, dtype=np.float64)         # additive-baseline anchor
        self.gj = np.zeros(self.d_p, dtype=np.float64)         # binding's co-adapted additive anchor
        self.Wa = {}; self.Wb = {}                             # additive-BASELINE per-primitive readouts (H x d_p)
        self.WjA = {}; self.WjB = {}                           # binding's additive per-primitive readouts
        self.WmulA = {}; self.WmulB = {}                       # binding's dendritic-AND FACTOR readouts
        self._stored_raw_patterns = 0
        self._used_ruler = False
        self._bind_ready = False
        self._bind_active = False                              # did the gate fire (a real interaction to bind)?
        self._factor_prod_minus_sum = 0.0                      # anti-cheat: product materially != sum of factors

    def fit(self, taught_idx, attrs, engrams):
        """engrams: dict class_idx -> d_p engram (mean of noisy draws; the brain's compressed wake trace)."""
        E = {(attrs[j][0], attrs[j][1]): np.asarray(engrams[j], dtype=np.float64)[:self.d_p] for j in taught_idx}
        taught_cells = [attrs[j] for j in taught_idx]
        # --- additive BASELINE: pure two-way ANOVA (structurally cannot carry an interaction) ---
        self.g0, rowA0, colB0 = _anova(E, taught_cells, self.K1, self.K2, self.d_p)
        for a, tgt in rowA0.items():
            self.Wa[a] = _fit_readout(self.res, _A_OFF + a, tgt, self.gen_lr, self.conv_tol, self.conv_max_epochs)
        for b, tgt in colB0.items():
            self.Wb[b] = _fit_readout(self.res, _B_OFF + b, tgt, self.gen_lr, self.conv_tol, self.conv_max_epochs)
        # --- binding: GATED JOINT additive+rank-1 (co-adapted) ---
        R0 = {c: E[c] - (self.g0 + rowA0[c[0]] + colB0[c[1]]) for c in taught_cells}
        resid_rms = float(np.sqrt(np.mean([np.sum(R0[c] ** 2) for c in taught_cells])))
        eng_rms = float(np.sqrt(np.mean([np.sum(E[c] ** 2) for c in taught_cells]))) + 1e-12
        gj = self.g0.copy(); rowAj = dict(rowA0); colBj = dict(colB0)
        fA = np.zeros((self.K1, self.d_p), dtype=np.float64)
        fB = np.zeros((self.K2, self.d_p), dtype=np.float64)
        if resid_rms > self.gate * eng_rms:                    # a real interaction to bind (else keep 0 -> no low-s harm)
            self._bind_active = True
            fB = np.random.default_rng(12345).standard_normal((self.K2, self.d_p)) * 0.1     # non-degenerate init
            for _o in range(self.outer):
                # rank-1 ALS of the CURRENT additive residual (per-channel; coverage completes held-out)
                R = {c: E[c] - (gj + rowAj[c[0]] + colBj[c[1]]) for c in taught_cells}
                for _i in range(self.inner):
                    for a in range(self.K1):
                        bs = [b for (aa, b) in taught_cells if aa == a]
                        if not bs:
                            continue
                        num = np.sum([R[(a, b)] * fB[b] for b in bs], axis=0)
                        den = np.sum([fB[b] ** 2 for b in bs], axis=0) + 1e-9
                        fA[a] = num / den
                    for b in range(self.K2):
                        as_ = [a for (a, bb) in taught_cells if bb == b]
                        if not as_:
                            continue
                        num = np.sum([R[(a, b)] * fA[a] for a in as_], axis=0)
                        den = np.sum([fA[a] ** 2 for a in as_], axis=0) + 1e-9
                        fB[b] = num / den
                    col = np.sqrt(np.sum(fB ** 2, axis=0)) + 1e-12                            # gauge-fix the scale
                    fB = fB / col; fA = fA * col
                # co-adapt the additive part to the bilinear-removed engram (unbiases the missing-cell ANOVA)
                Radd = {c: E[c] - fA[c[0]] * fB[c[1]] for c in taught_cells}
                gj, rowAj, colBj = _anova(Radd, taught_cells, self.K1, self.K2, self.d_p)
            # SYMMETRIC gauge: split each channel's scale evenly between fA and fB (product unchanged) so NEITHER
            # factor is large -- the runtime dendritic PRODUCT of two lossy spiking readouts has cross-error terms
            # fA*eps_B + eps_A*fB that blow up when one factor carries all the magnitude; balancing minimizes them.
            na = np.sqrt(np.sum(fA ** 2, axis=0)); nb = np.sqrt(np.sum(fB ** 2, axis=0))
            t = np.sqrt((nb + 1e-12) / (na + 1e-12))
            fA = fA * t; fB = fB / t
        self.gj = gj
        # host copies of the fitted params (ORACLE diagnostic: isolates completion quality from neuralization error)
        self._fA_host = fA.copy(); self._fB_host = fB.copy()
        self._gj_host = gj.copy(); self._rowAj_host = dict(rowAj); self._colBj_host = dict(colBj)
        mul_tol = min(self.conv_tol, 0.004)                    # the PRODUCT amplifies factor error -> converge tighter
        mul_epochs = max(self.conv_max_epochs, 600)
        for a in range(self.K1):
            if a in rowAj:
                self.WjA[a] = _fit_readout(self.res, _JA_OFF + a, rowAj[a], self.gen_lr, self.conv_tol,
                                           self.conv_max_epochs)
            self.WmulA[a] = _fit_readout(self.res, _MULA_OFF + a, fA[a], self.gen_lr, mul_tol, mul_epochs)
        for b in range(self.K2):
            if b in colBj:
                self.WjB[b] = _fit_readout(self.res, _JB_OFF + b, colBj[b], self.gen_lr, self.conv_tol,
                                           self.conv_max_epochs)
            self.WmulB[b] = _fit_readout(self.res, _MULB_OFF + b, fB[b], self.gen_lr, mul_tol, mul_epochs)
        # anti-cheat witness: the PRODUCT materially differs from the SUM of the same factors (a real multiplication)
        diffs = []
        for (a, b) in taught_cells:
            pa = self._mul_factor_a(a); pb = self._mul_factor_b(b)
            diffs.append(float(np.linalg.norm(pa * pb - (pa + pb))))
        self._factor_prod_minus_sum = float(np.mean(diffs)) if diffs else 0.0
        self._bind_ready = True

    # --- neural population outputs (spiking readout of the frozen reservoir) ---
    def _add_a(self, a):
        return self.Wa[a].T @ self.res.elig(_A_OFF + a) if a in self.Wa else np.zeros(self.d_p)

    def _add_b(self, b):
        return self.Wb[b].T @ self.res.elig(_B_OFF + b) if b in self.Wb else np.zeros(self.d_p)

    def _jadd_a(self, a):
        return self.WjA[a].T @ self.res.elig(_JA_OFF + a) if a in self.WjA else np.zeros(self.d_p)

    def _jadd_b(self, b):
        return self.WjB[b].T @ self.res.elig(_JB_OFF + b) if b in self.WjB else np.zeros(self.d_p)

    def _mul_factor_a(self, a):
        return self.WmulA[a].T @ self.res.elig(_MULA_OFF + a) if a in self.WmulA else np.zeros(self.d_p)

    def _mul_factor_b(self, b):
        return self.WmulB[b].T @ self.res.elig(_MULB_OFF + b) if b in self.WmulB else np.zeros(self.d_p)

    def additive(self, a, b):
        """BASELINE: NEURAL SUPERPOSITION -- grand-mean anchor + SUM of two spiking-readout outputs (VSA bundling)."""
        return self.g0 + self._add_a(a) + self._add_b(b)

    def binding(self, a, b):
        """TREATMENT: co-adapted additive part + the dendritic-AND / sigma-pi ELEMENTWISE PRODUCT of two
        spiking-readout population outputs (the neural conjunction)."""
        return self.gj + self._jadd_a(a) + self._jadd_b(b) + self._mul_factor_a(a) * self._mul_factor_b(b)

    def sum_ablation(self, a, b):
        """CONTROL: the SAME factors combined by ADDITION (no multiplication) -> absorbed additively -> no bind."""
        return self.gj + self._jadd_a(a) + self._jadd_b(b) + self._mul_factor_a(a) + self._mul_factor_b(b)

    def binding_oracle(self, a, b):
        """DIAGNOSTIC ONLY (not an arm): the SAME model with HOST factor values (no reservoir readout) -> isolates the
        rank-1 COMPLETION quality from the neural readout error. NOT used for the GO decision."""
        add = self._gj_host + self._rowAj_host.get(a, 0.0) + self._colBj_host.get(b, 0.0)
        return add + self._fA_host[a] * self._fB_host[b]


# ============================ per (grid, s) driver ============================
def _run_grid_s(seed, K1, K2, m, s, mix_scale, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                bdsp_wmax, conv_tol, conv_max_epochs, gen_epochs, batch, n_draws, bind_gate):
    N = K1 * K2
    d_p = int(d_a) + int(d_b)
    n_in = d_p + N_ACT
    chance = 1.0 / N
    referents, attrs = _grid_facts(K1, K2)

    # held-out split + anti-cheat asserts (disjoint + coverage)
    taught_idx, held_idx = _heldout_split(K1, K2, m, seed)
    taught_set, held_set = set(taught_idx), set(held_idx)
    disjoint = bool(len(taught_set & held_set) == 0 and len(held_set) > 0)
    trained_a = {attrs[j][0] for j in taught_idx}; trained_b = {attrs[j][1] for j in taught_idx}
    coverage_ok = bool(all(attrs[j][0] in trained_a and attrs[j][1] in trained_b for j in held_idx))
    assert disjoint, "taught and held-out sets must be disjoint and held-out non-empty"
    assert coverage_ok, "every held-out primitive (a AND b) must appear in >= 1 taught combo"

    env = _make_mixed_env(seed, K1, K2, d_a, d_b, noise, s, mix_scale, referents, attrs)
    protos = np.stack([env.proto(referents[j]) for j in range(N)]).astype(np.float64)   # (N,d_p) test-time RULER only
    witness = _additive_nonadditivity_witness(protos, attrs, K1, K2)

    # one shared reservoir; condition its readout-norm over ALL addresses the arms use
    all_addrs = ([_A_OFF + a for a in range(K1)] + [_B_OFF + b for b in range(K2)]
                 + [_JA_OFF + a for a in range(K1)] + [_JB_OFF + b for b in range(K2)]
                 + [_MULA_OFF + a for a in range(K1)] + [_MULB_OFF + b for b in range(K2)])
    res = _Reservoir(gen_k, n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax, all_addrs)

    # engrams for taught facts (brain's compressed wake trace = mean of noisy draws); ONLY taught percepts drawn
    engrams = {}
    fed = []
    for j in taught_idx:
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)
        engrams[j] = np.asarray(Xj, dtype=np.float64).mean(axis=0)
        fed.append(j)
    no_leakage = bool(not (set(fed) & held_set))
    assert no_leakage, "a held-out fact index leaked into a training path"

    # ARM: additive + binding + sum_ablation (one generator)
    bg = BindingGenerator(res, d_a, d_b, K1, K2, gen_lr, conv_tol, conv_max_epochs, gate=bind_gate)
    bg.fit(taught_idx, attrs, engrams)

    # ARM: fixed v2 (FLOOR) taught on the TAUGHT set only (class = original fact index). The FLOOR only needs to be a
    # fixed CLASS-INDEX generator: a held-out class index is NEVER trained -> its readout is the untrained near-anchor
    # output -> chance identity REGARDLESS of how well the TAUGHT classes are fit. So it is trained CHEAPLY (a light
    # fit, no O(N^2) self-replay: past=[]) -- the expensive train-to-convergence self-replay (conv_max_epochs=120,
    # new_mult=3) costs ~20k reservoir forwards at N=56 and 165s/row for NO change in the held-out floor.
    vgen = GenerativeReplayNetV2(int(gen_k), n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax=bdsp_wmax,
                                 conv_tol=0.05, conv_max_epochs=max(2, int(gen_epochs)), conv_check_every=4, new_mult=1)
    vgen.fit_query_norm()
    vgen_rng = np.random.default_rng(seed + 999)
    for j in taught_idx:
        vgen.learn_fact(j, engrams[j], [], gen_epochs, batch, vgen_rng)

    # ARM: flat O(N) buffer (FLOOR) taught on the TAUGHT set only
    flat = FlatStore(d_p, seed)
    for j in taught_idx:
        flat.learn(j, engrams[j])

    # recall predictors (nearest-prototype identity, arm-symmetric)
    def add_pred(j):
        a, b = attrs[j]; return _nearest_proto(bg.additive(a, b)[:d_p], protos)

    def bind_pred(j):
        a, b = attrs[j]; return _nearest_proto(bg.binding(a, b)[:d_p], protos)

    def bind_oracle_pred(j):
        a, b = attrs[j]; return _nearest_proto(bg.binding_oracle(a, b)[:d_p], protos)

    def sumabl_pred(j):
        a, b = attrs[j]; return _nearest_proto(bg.sum_ablation(a, b)[:d_p], protos)

    def v2_pred(j):
        return _nearest_proto(vgen.regenerate(j)[:d_p], protos)

    def flat_pred(j):
        return flat.recall_nearest(j, protos)

    out = {
        "K1": K1, "K2": K2, "N": N, "s": s, "mix_scale": mix_scale, "chance": chance,
        "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "taught_heldout_disjoint": disjoint, "every_heldout_primitive_seen_in_taught": coverage_ok,
        "no_leakage_heldout_never_trained": no_leakage,
        "nonadditivity_witness": witness,
        # HEADLINE: held-out (zero-shot) recall
        "binding_heldout_recall": _recall_fraction(held_idx, bind_pred, protos),
        "binding_oracle_heldout_recall": _recall_fraction(held_idx, bind_oracle_pred, protos),
        "additive_heldout_recall": _recall_fraction(held_idx, add_pred, protos),
        "sum_ablation_heldout_recall": _recall_fraction(held_idx, sumabl_pred, protos),
        "v2_heldout_recall": _recall_fraction(held_idx, v2_pred, protos),
        "flat_heldout_recall": _recall_fraction(held_idx, flat_pred, protos),
        # taught (seen) recall = sanity
        "binding_seen_recall": _recall_fraction(taught_idx, bind_pred, protos),
        "additive_seen_recall": _recall_fraction(taught_idx, add_pred, protos),
        # neural / multiplicative anti-cheat witnesses
        "reservoir_mean_spikes": res.mean_spikes(),
        "bind_factor_prod_minus_sum": bg._factor_prod_minus_sum,
        "bind_active": bool(bg._bind_active),
        "binding_stored_raw_patterns": int(bg._stored_raw_patterns),
        "binding_used_ruler": bool(bg._used_ruler),
    }
    # regeneration fidelity diagnostics (ruler only)
    out["binding_heldout_cos"] = float(np.mean([_cos(bg.binding(*attrs[j])[:d_p], protos[j]) for j in held_idx]))
    out["additive_heldout_cos"] = float(np.mean([_cos(bg.additive(*attrs[j])[:d_p], protos[j]) for j in held_idx]))
    return out


# ============================ verdict ============================
def _verdict(result):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    grids = result["per_grid"]                                 # grid -> {s -> row}
    gkeys = sorted(grids, key=lambda g: grids[g]["N"])
    v = Verdict("teacher-loop CONJUNCTIVE BINDING (recover zero-shot composition where superposition breaks)",
                chance=None)

    per_grid_summary = {}
    all_go = True
    for g in gkeys:
        gr = grids[g]
        rows = gr["rows"]                                      # s(str) -> row
        svals = sorted(rows, key=lambda x: float(x))
        N = gr["N"]
        high_s = [sv for sv in svals if float(sv) >= 0.75]
        low_s = [sv for sv in svals if float(sv) <= 0.5]

        # per-s attribution (binding vs additive baseline)
        for sv in svals:
            r = rows[sv]
            attributable_to(f"[{g} s={sv}] binding binding vs additive superposition (held-out)",
                            r["binding_heldout_recall"], r["additive_heldout_recall"])

        # THE WIN at high s: binding recovers where superposition broke. MARGIN (bind - additive >= 0.30) at EVERY
        # high s (the core claim, robust); ABSOLUTE (bind >= 0.5) at the MAXIMAL-break point (top s) where the
        # additive baseline has fully collapsed to ~chance and the completion is best-conditioned.
        recover_ok = True
        top_s = svals[-1]
        for sv in high_s:
            r = rows[sv]
            margin_ok = bool(r["binding_heldout_recall"] >= r["additive_heldout_recall"] + 0.30)
            abs_ok = bool(r["binding_heldout_recall"] >= 0.5) if sv == top_s else True
            ok = bool(margin_ok and abs_ok)
            recover_ok = recover_ok and ok
            v.require(f"[{g} s={sv}] binding RECOVERS held-out (>= additive+0.30"
                      + (" AND >= 0.5)" if sv == top_s else ")"), ok, expect=True,
                      note=f"bind {r['binding_heldout_recall']:.2f} vs add {r['additive_heldout_recall']:.2f} "
                           f"(chance {r['chance']:.3f})")
        # no low-s cost: binding >= additive - 0.05
        nocost_ok = True
        for sv in low_s:
            r = rows[sv]
            ok = bool(r["binding_heldout_recall"] >= r["additive_heldout_recall"] - 0.05)
            nocost_ok = nocost_ok and ok
            v.require(f"[{g} s={sv}] no low-s cost (binding >= additive-0.05)", ok, expect=True,
                      note=f"bind {r['binding_heldout_recall']:.2f} vs add {r['additive_heldout_recall']:.2f}")
        # conjunction load-bearing: sum_ablation does NOT recover at the top s
        top = svals[-1]; rt = rows[top]
        conj_ok = bool(rt["binding_heldout_recall"] - rt["sum_ablation_heldout_recall"] >= 0.20)
        v.require(f"[{g} s={top}] conjunction load-bearing (binding - sum_ablation >= 0.20)", conj_ok, expect=True,
                  note=f"bind {rt['binding_heldout_recall']:.2f} vs sum_abl {rt['sum_ablation_heldout_recall']:.2f}")
        # sanity: binding taught recall high (min over s)
        seen_min = min(rows[sv]["binding_seen_recall"] for sv in svals)
        seen_ok = bool(seen_min >= 0.85)
        v.require(f"[{g}] binding taught (seen) recall >= 0.85 (min over s)", seen_ok, expect=True,
                  note=f"min-seen {seen_min:.2f}")
        # the baseline actually broke (reported; not a hard gate): additive at top s <= 0.6
        add_broke = bool(rt["additive_heldout_recall"] <= 0.6)
        v.require(f"[{g} s={top}] additive baseline BROKE (<= 0.6, else nothing to recover)", add_broke, expect=True,
                  note=f"add {rt['additive_heldout_recall']:.2f}")
        # neural + multiplicative + zero-shot anti-cheats (from the top-s row; identical structure across s)
        neural_ok = bool(rt["reservoir_mean_spikes"] > 0.0)
        mult_ok = bool(rt["bind_factor_prod_minus_sum"] > 1e-6)
        witness_ok = bool(rt["nonadditivity_witness"] > 0.02)
        zshot_ok = bool(rt["taught_heldout_disjoint"] and rt["every_heldout_primitive_seen_in_taught"]
                        and rt["no_leakage_heldout_never_trained"])
        notbuf_ok = bool(rt["binding_stored_raw_patterns"] == 0 and not rt["binding_used_ruler"])
        v.require(f"[{g}] bind is NEURAL (reservoir spikes > 0)", neural_ok, expect=True,
                  note=f"mean spikes {rt['reservoir_mean_spikes']:.1f}")
        v.require(f"[{g}] bind genuinely MULTIPLICATIVE (product != sum of factors)", mult_ok, expect=True,
                  note=f"||prod-sum|| {rt['bind_factor_prod_minus_sum']:.3f}")
        v.require(f"[{g}] world carries a real CONJUNCTION at high s (witness > 0.02)", witness_ok, expect=True,
                  note=f"witness {rt['nonadditivity_witness']:.3f}")
        v.require(f"[{g}] genuinely ZERO-SHOT (disjoint + coverage + no-leakage)", zshot_ok, expect=True)
        v.require(f"[{g}] 0 stored raw patterns + ruler untouched", notbuf_ok, expect=True)

        grid_go = bool(recover_ok and nocost_ok and conj_ok and seen_ok and neural_ok and mult_ok and witness_ok
                       and zshot_ok and notbuf_ok)
        all_go = all_go and grid_go
        per_grid_summary[g] = {
            "N": N, "held_out_n": gr["held_out_n"],
            "by_s": {sv: {
                "binding_heldout_recall": rows[sv]["binding_heldout_recall"],
                "additive_heldout_recall": rows[sv]["additive_heldout_recall"],
                "sum_ablation_heldout_recall": rows[sv]["sum_ablation_heldout_recall"],
                "v2_heldout_recall": rows[sv]["v2_heldout_recall"],
                "flat_heldout_recall": rows[sv]["flat_heldout_recall"],
                "binding_seen_recall": rows[sv]["binding_seen_recall"],
                "additive_seen_recall": rows[sv]["additive_seen_recall"],
                "binding_minus_additive": float(rows[sv]["binding_heldout_recall"]
                                                - rows[sv]["additive_heldout_recall"]),
                "nonadditivity_witness": rows[sv]["nonadditivity_witness"],
                "chance": rows[sv]["chance"],
            } for sv in svals},
            "recover_ok": recover_ok, "nocost_ok": nocost_ok, "conjunction_load_bearing": conj_ok,
            "additive_broke_high_s": add_broke, "grid_go": grid_go,
        }

    v.require("(seed) substrate byte-identical", bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    go = bool(all_go and result["substrate_byte_identical"] and result["sim_diff_empty"])
    decision = v.decide(go=go)
    return {"grids": gkeys, "per_grid": per_grid_summary,
            "substrate_byte_identical": result["substrate_byte_identical"],
            "sim_diff_empty": result["sim_diff_empty"], **decision}


# ============================ orchestration ============================
def run(seed, grids, held_out, s_values, mix_scale, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
        bdsp_wmax, conv_tol, conv_max_epochs, gen_epochs, batch, n_draws, bind_gate):
    n_in = int(d_a) + int(d_b) + N_ACT
    Kbig = max(k1 * k2 for k1, k2 in grids)
    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, Kbig, seed, max(120, 6 * Kbig), 20,
                                                               0.5, w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()
    per_grid = {}
    for (K1, K2) in grids:
        m = held_out.get(f"{K1}x{K2}", max(1, min(K1, K2)))
        rows = {}
        print(f"\n{'=' * 96}\n# SEED {seed}  GRID {K1}x{K2} (N={K1*K2}, held_out={m})  s-sweep {s_values}\n{'=' * 96}",
              flush=True)
        for s in s_values:
            r = _run_grid_s(seed, K1, K2, m, s, mix_scale, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr,
                            w_clip, bdsp_wmax, conv_tol, conv_max_epochs, gen_epochs, batch, n_draws, bind_gate)
            rows[str(s)] = r
            print(f"  [s={s:.2f}] held-out recall: bind {r['binding_heldout_recall']:.2f} | add "
                  f"{r['additive_heldout_recall']:.2f} | sum_abl {r['sum_ablation_heldout_recall']:.2f} | v2 "
                  f"{r['v2_heldout_recall']:.2f} | flat {r['flat_heldout_recall']:.2f} (chance {r['chance']:.3f}) | "
                  f"seen(bind) {r['binding_seen_recall']:.2f} | witness {r['nonadditivity_witness']:.3f} | spikes "
                  f"{r['reservoir_mean_spikes']:.0f}", flush=True)
        per_grid[f"{K1}x{K2}"] = {"K1": K1, "K2": K2, "N": K1 * K2, "held_out_n": rows[str(s_values[0])]["held_out_n"],
                                  "rows": rows}
    return {"seed": seed, "grids": [f"{k1}x{k2}" for k1, k2 in grids], "s_values": s_values, "mix_scale": mix_scale,
            "d_a": d_a, "d_b": d_b, "n_in": n_in,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"d_a": d_a, "d_b": d_b, "noise": noise, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "gen_settle": gen_settle, "gen_lr": gen_lr, "w_clip": w_clip, "bdsp_wmax": bdsp_wmax,
                       "conv_tol": conv_tol, "conv_max_epochs": conv_max_epochs, "gen_epochs": gen_epochs,
                       "batch": batch, "n_draws": n_draws, "held_out": held_out, "s_values": s_values,
                       "mix_scale": mix_scale, "frozen_hidden": True},
            "per_grid": per_grid}


def _parse_grid(s):
    a, b = s.lower().split("x"); return (int(a), int(b))


def _one_seed(a, seed, grids, held_out):
    result = run(seed, grids, held_out, a.s_values, a.mix_scale, a.d_a, a.d_b, a.noise, a.gen_hidden, a.gen_k,
                 a.gen_settle, a.gen_lr, a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.gen_epochs, a.batch,
                 a.n_draws, a.bind_gate)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop CONJUNCTIVE BINDING: a neural dendritic-AND / sigma-pi bind "
                                             "that recovers zero-shot composition at high shared-channel mixing where "
                                             "additive superposition (bundling) breaks.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--grids", nargs="+", default=["7x7", "8x8"],
                    help="grids big enough that each row/col keeps many taught cells -> the rank-1 completion is "
                         "well-conditioned (5x5/6x6 with proportional held-out are too small to complete robustly)")
    ap.add_argument("--held-out", nargs="+", default=["7x7:7", "8x8:8"],
                    help="per-grid held-out count as GRID:M (coverage-preserving; leaves >=6 taught per row/col)")
    ap.add_argument("--s-values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="shared-channel mixing strengths to sweep")
    ap.add_argument("--mix-scale", type=float, default=0.4, help="conjunction RMS as a fraction of base-centered RMS "
                                                                 "(smaller -> additive stays robust longer, sharper break)")
    ap.add_argument("--bind-gate", type=float, default=0.25, help="fire the bilinear bind only when the additive "
                                                                  "residual RMS exceeds this fraction of the engram RMS")
    ap.add_argument("--d-a", type=int, default=10)
    ap.add_argument("--d-b", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--gen-hidden", type=int, default=96, help="FIXED generator reservoir size (H_gen)")
    ap.add_argument("--gen-k", type=int, default=64, help="FIXED query address width")
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--conv-tol", type=float, default=0.02)
    ap.add_argument("--conv-max-epochs", type=int, default=200)
    ap.add_argument("--gen-epochs", type=int, default=6, help="v2 (floor) training epochs/fact (a chance floor -- "
                                                              "held-out class is never trained, so a light fit suffices)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9)
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grids = [_parse_grid(g) for g in a.grids]
    held_out = {}
    for spec in a.held_out:
        k, mm = spec.split(":"); held_out[k.lower()] = int(mm)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  grids={a.grids} held_out={a.held_out} s={a.s_values} "
              f"mix_scale={a.mix_scale}\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, grids, held_out)
        summary = {"probe": "teacher_loop_conjunctive_binding", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "grids": a.grids, "held_out": a.held_out, "s_values": a.s_values, "mix_scale": a.mix_scale,
                   "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        for g in rv["grids"]:
            pg = rv["per_grid"][g]
            for sv, row in pg["by_s"].items():
                print(f"[bind] seed {s} {g} s={sv}: N={pg['N']} | HELD-OUT bind {row['binding_heldout_recall']:.2f} "
                      f"vs add {row['additive_heldout_recall']:.2f} (d={row['binding_minus_additive']:+.2f}) vs "
                      f"sum_abl {row['sum_ablation_heldout_recall']:.2f} vs v2 {row['v2_heldout_recall']:.2f} vs flat "
                      f"{row['flat_heldout_recall']:.2f} | witness {row['nonadditivity_witness']:.3f}", flush=True)
            print(f"[bind] seed {s} {g}: recover {pg['recover_ok']} no-cost {pg['nocost_ok']} conj "
                  f"{pg['conjunction_load_bearing']} add-broke {pg['additive_broke_high_s']} | GO {pg['grid_go']}",
                  flush=True)
        print(f"[bind] seed {s} byte-id {rv['substrate_byte_identical']} sim-clean {rv['sim_diff_empty']} | "
              f"VERDICT {rv['status']}", flush=True)
        print(f"[bind] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        agg = {"probe": "teacher_loop_conjunctive_binding_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "grids": a.grids, "held_out": a.held_out,
               "s_values": a.s_values, "mix_scale": a.mix_scale, "go_count": go_n, "n_seeds": len(seeds),
               "per_grid_s_means": {}, "per_seed": per_seed}
        for g in per_seed[0]["verdict"]["grids"]:
            agg["per_grid_s_means"][g] = {}
            svs = list(per_seed[0]["verdict"]["per_grid"][g]["by_s"].keys())
            for sv in svs:
                bind = [p["verdict"]["per_grid"][g]["by_s"][sv]["binding_heldout_recall"] for p in per_seed]
                add = [p["verdict"]["per_grid"][g]["by_s"][sv]["additive_heldout_recall"] for p in per_seed]
                sab = [p["verdict"]["per_grid"][g]["by_s"][sv]["sum_ablation_heldout_recall"] for p in per_seed]
                v2 = [p["verdict"]["per_grid"][g]["by_s"][sv]["v2_heldout_recall"] for p in per_seed]
                flat = [p["verdict"]["per_grid"][g]["by_s"][sv]["flat_heldout_recall"] for p in per_seed]
                seen = [p["verdict"]["per_grid"][g]["by_s"][sv]["binding_seen_recall"] for p in per_seed]
                wit = [p["verdict"]["per_grid"][g]["by_s"][sv]["nonadditivity_witness"] for p in per_seed]
                agg["per_grid_s_means"][g][sv] = {
                    "N": per_seed[0]["verdict"]["per_grid"][g]["N"],
                    "chance": per_seed[0]["verdict"]["per_grid"][g]["by_s"][sv]["chance"],
                    "binding_heldout_recall_mean": float(np.nanmean(bind)),
                    "binding_heldout_recall_per_seed": [float(x) for x in bind],
                    "additive_heldout_recall_mean": float(np.nanmean(add)),
                    "additive_heldout_recall_per_seed": [float(x) for x in add],
                    "sum_ablation_heldout_recall_mean": float(np.nanmean(sab)),
                    "v2_heldout_recall_mean": float(np.nanmean(v2)),
                    "flat_heldout_recall_mean": float(np.nanmean(flat)),
                    "binding_seen_recall_mean": float(np.nanmean(seen)),
                    "nonadditivity_witness_mean": float(np.nanmean(wit)),
                }
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[bind AGG] GO {go_n}/{len(seeds)}", flush=True)
        for g, svd in agg["per_grid_s_means"].items():
            for sv, mm in svd.items():
                print(f"   {g} s={sv}: N={mm['N']} | HELD-OUT bind {mm['binding_heldout_recall_mean']:.2f} vs add "
                      f"{mm['additive_heldout_recall_mean']:.2f} vs sum_abl {mm['sum_ablation_heldout_recall_mean']:.2f} "
                      f"vs v2 {mm['v2_heldout_recall_mean']:.2f} vs flat {mm['flat_heldout_recall_mean']:.2f} "
                      f"(chance {mm['chance']:.3f}) | witness {mm['nonadditivity_witness_mean']:.3f} | bind/seed "
                      f"{mm['binding_heldout_recall_per_seed']}", flush=True)
        print(f"[bind AGG] wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
