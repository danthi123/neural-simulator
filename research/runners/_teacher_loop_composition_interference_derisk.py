"""TEACHER-LOOP COMPOSITION-INTERFERENCE DE-RISK (2026-08-09): stress-test neural superposition-binding.

WHERE THE ARC STANDS. ZERO-SHOT neural compositional generalization = GO (ea77003c5, 6/6 both grids): a spiking
generator recalls NEVER-TAUGHT (a,b) combinations at 1.00 by NEURAL SUPERPOSITION (sum) of two primitive spiking-
readout outputs; fixed/flat floors at chance. THE HONEST BOUNDARY that GO named: the world was a CLEAN concat-of-
primitives on DISJOINT channels (percept = [primA | primB] on separate feature blocks), so LINEAR superposition is
near-exact. Real compositional structure INTERFERES: primitives mix on SHARED channels / non-linearly. This de-risk
stress-tests robustness and LOCATES the breaking point.

THE HYPOTHESIS + STAKES. Build a HARDER compositional world where the two primitives are NOT on disjoint channels:
percept = a NON-LINEAR, SHARED-CHANNEL mixing f(primA, primB) with a MIXING-STRENGTH knob s in [0,1] (0 = clean
disjoint concat -> 1 = strong shared-channel non-linear mixing with an explicit product/AND conjunction term). Re-run
the SAME zero-shot held-out test (coverage-preserving split) SWEEPING s. TWO first-class outcomes:
  (a) if LINEAR neural superposition STILL generalizes zero-shot across s -> the composer is robust to interference;
  (b) if it FAILS at some s -> HONEST NEGATIVE that NAMES the real binding problem: linear superposition (bundling)
      is insufficient for a non-linearly-mixed world; the residual is a NON-LINEAR binding operator (conjunctive /
      sigma-pi / dendritic-AND / tensor / circular-convolution) -- exactly what the project VSA composer approximates.

STEP 1 -- the HARDER world. `MixedCompositionalReferentEnv` subclasses the clean `CompositionalReferentEnv` and
overrides ONLY `proto`. For fact (a,b) with clean concat c = [primA[a] | primB[b]] (d_p = d_a+d_b):
  * conj = primA[a] (.) primB[b]  (elementwise product on the overlap; a pure AND interaction needing BOTH),
  * m01  = 0.5*(tanh(M @ [primA | primB | conj]) + 1)  -- a fixed seeded random SHARED-channel mix (every output
    channel is a non-linear function of ALL A feats, ALL B feats, AND the conjunction), rescaled to [0,1],
  * proto(a,b; s) = clip( (1-s)*c + s*m01 , 0, 1 ).
At s=0 this is EXACTLY the clean disjoint concat (reproduces the 1.00 baseline in-run); at s=1 it is a genuinely
non-additive, channel-shared percept. Host code -- the WORLD's render -- is legitimate (a retinal image; the brain
reads it through its OWN weights). The BRAIN is unchanged: the SAME disjoint-block spiking-superposition generator.

STEP 2 -- re-run the zero-shot held-out test, arms compositional_gen (neural superposition) vs fixed-v2 vs flat, and
SWEEP s = 0.0, 0.25, 0.5, 0.75, 1.0 to locate the breaking point. HEADLINE: zero-shot held-out recall vs s.

ANTI-CHEATS (each a REAL assertion / measurement in the output):
  * held-out GENUINELY never taught + coverage + NO leakage (the clean-GO asserts, reused verbatim: taught/held-out
    disjoint; every held-out primitive seen in >=1 taught combo; no held-out fact index enters any training path).
  * composition at test is NEURAL: regeneration is a SUM of two spiking-reservoir leaky-readout outputs; a LESION
    zeroing one primitive's engram breaks ONLY that primitive's block on held-out facts (localisation).
  * the hard world is GENUINELY non-linearly mixed (NOT secretly linear/disjoint), asserted by TWO instruments per s:
      - SHARED-channel leak: || proto(a,b)[:d_a] - proto(a,b')[:d_a] || averaged over b!=b'. ==0 in the clean world
        (A-block depends only on a); > tol at s=1 => channels are genuinely shared.
      - NON-ADDITIVITY residual: || proto(a,b) - [proto(a,b0)+proto(a0,b)-proto(a0,b0)] ||. ==0 for ANY additive
        OR linear-shared world (a linear mix of an additive code stays additive); > tol at s=1 => genuinely non-
        linear binding energy that NO sum of independent A and B codes can represent. This is the binding residual.
  * cfg.seed byte-identical substrate (NOT actual_seed_used); de-clamped bdsp_wmax=1e9; git diff main -- sim/ empty;
    backend recorded.

VERDICT (honest negative is FIRST-CLASS -- do NOT force a GO):
  * GO (robust): compositional held-out recall >= 0.5 AND >= each floor + 0.30 at EVERY mixing strength.
  * PARTIAL: holds at clean/mild but breaks by strong -> the breaking point is LOCATED (report s*).
  * NEGATIVE: breaks by mild mixing -> linear superposition is insufficient; the residual is non-linear binding.
The s=0 clean baseline (~1.00) is measured IN-RUN for contrast; the fixed-v2 + flat floors stay ~chance at every s.

DR grounding: the binding problem (Treisman 1980, feature-integration); VSA binding vs bundling (Plate HRR;
Kanerva); the project composer/VSA notes (superposition = bundling is lossy for CONJUNCTIVE structure -> needs a
binding operator). Reuse-by-import of the clean zero-shot harness + the compositional generator. NO sim/ edit.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_composition_interference_derisk --seed 42 \
      --grids 5x5 6x6 --mix-sweep 0.0 0.25 0.5 0.75 1.0 \
      --out research/findings/raw/teacher_loop_composition_interference_s42.json
  6-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_composition_interference_derisk --seeds 42 43 44 45 46 47 \
      --grids 5x5 6x6 --mix-sweep 0.0 0.25 0.5 0.75 1.0 \
      --out research/findings/raw/teacher_loop_composition_interference.json
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
# reuse-by-import: the clean compositional WORLD + the neural-superposition generator + neural regenerate + the
# lesion-localisation check; the clean zero-shot harness (held-out split, floors, recall helpers); the v2 fixed
# generator floor; the scaling teacher machinery; the byte-identical + sim-clean anti-cheat asserts. NO sim/ edit.
from research.runners._teacher_loop_compositional_generator_derisk import (  # noqa: E402
    CompositionalGenerator, CompositionalReferentEnv, _grid_facts, _action_ctx_const, _lesion_localises,
)
from research.runners._teacher_loop_zeroshot_composition_derisk import (  # noqa: E402
    _heldout_split, FlatStore, _nearest_proto, _recall_fraction,
)
from research.runners._teacher_loop_generative_replay_v2_derisk import GenerativeReplayNetV2  # noqa: E402
from research.runners._teacher_loop_generative_replay_derisk import _cos  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import _corrective_batch, N_ACT  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _assert_byte_identical_substrate, _git_sim_diff_empty,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_composition_interference.json"


# ============================ STEP 1: the HARDER (non-linearly-mixed, shared-channel) WORLD ============================
class MixedCompositionalReferentEnv(CompositionalReferentEnv):
    """The clean world's SHARED-CHANNEL, NON-LINEAR sibling. Overrides ONLY `proto`: fact (a,b)'s percept is a
    strength-`mix` interpolation from the clean disjoint concat toward a fixed seeded random SHARED-channel non-
    linear mix that carries an explicit product/AND conjunction of the two primitives. At mix=0 it IS the clean
    concat (byte-identical baseline); at mix=1 it is genuinely non-additive and channel-shared. Everything else
    (draw/noise/register/rng) is inherited unchanged -- the SAME percepts are shown to every arm."""

    def __init__(self, seed, K1, K2, d_a=8, d_b=8, noise=0.12, mix=0.0):
        super().__init__(seed, K1, K2, d_a=d_a, d_b=d_b, noise=noise)
        self.mix = float(mix)
        self._dm = min(self.d_a, self.d_b)                                   # conjunction overlap width
        # a fixed seeded random SHARED-channel mixing matrix: every output channel (d_p) is a non-linear function of
        # ALL A feats, ALL B feats, AND the conjunction feats. Dedicated mix-RNG so it is independent of primA/primB.
        mr = np.random.default_rng(int(seed) + 770077)
        n_feat = self.d_a + self.d_b + self._dm
        self._M = mr.standard_normal((self.d_p, n_feat)).astype(np.float64) / np.sqrt(n_feat)

    def proto(self, referent):
        if referent not in self.protos:
            a, b = self.fact_attrs[referent]
            cA = self.primA[a]; cB = self.primB[b]
            c = np.concatenate([cA, cB]).astype(np.float64)                  # clean disjoint concat (d_p)
            s = self.mix
            if s <= 0.0:
                self.protos[referent] = c.copy()                             # EXACTLY the clean baseline
            else:
                conj = cA[:self._dm] * cB[:self._dm]                         # AND / product interaction (needs BOTH)
                feats = np.concatenate([cA, cB, conj]).astype(np.float64)
                m01 = 0.5 * (np.tanh(self._M @ feats) + 1.0)                 # shared-channel NON-LINEAR mix -> [0,1]
                self.protos[referent] = np.clip((1.0 - s) * c + s * m01, 0.0, 1.0)
        return self.protos[referent]


def _make_mixed_env(seed, K1, K2, d_a, d_b, noise, mix, referents, attrs):
    env = MixedCompositionalReferentEnv(seed, K1, K2, d_a=d_a, d_b=d_b, noise=noise, mix=mix)
    for r, (a, b) in zip(referents, attrs):
        env.register(r, a, b)
        env.proto(r)                                                        # instantiate the (mixed) prototype
    env.rng = np.random.default_rng(seed + 101)                            # reset draw-stream => every arm sees SAME percepts
    return env


# ============================ world-is-genuinely-mixed instruments (anti-cheat) ============================
def _mixing_witness(env, referents, attrs, K1, K2, d_a):
    """TWO instruments that PROVE the world at this mix strength is genuinely non-linearly, shared-channel mixed
    (NOT secretly linear/disjoint) -- both read the true (noiseless) prototypes only.
      shared_channel_leak: mean || proto(a,b)[:d_a] - proto(a,b')[:d_a] || over b != b'. ==0 in the clean world
        (A-block depends only on a) => >0 proves channels are SHARED (b bleeds into the A-block channels).
      nonadditivity_residual: mean || proto(a,b) - [proto(a,b0)+proto(a0,b)-proto(a0,b0)] ||. ==0 for ANY additive
        OR linear-shared-mix world => >0 proves genuinely NON-LINEAR binding energy (the binding residual)."""
    idx = {(a, b): j for j, (a, b) in enumerate(attrs)}
    P = np.stack([env.proto(referents[j]) for j in range(len(referents))]).astype(np.float64)
    # shared-channel leak
    leaks = []
    for a in range(K1):
        for b in range(K2):
            for b2 in range(b + 1, K2):
                leaks.append(float(np.linalg.norm(P[idx[(a, b)]][:d_a] - P[idx[(a, b2)]][:d_a])))
    shared_leak = float(np.mean(leaks)) if leaks else 0.0
    # non-additivity residual (relative to the a0=0,b0=0 reference)
    a0, b0 = 0, 0
    res, denom = [], []
    p00 = P[idx[(a0, b0)]]
    for a in range(K1):
        for b in range(K2):
            add = P[idx[(a, b0)]] + P[idx[(a0, b)]] - p00                    # the additive/linear-shared prediction
            res.append(float(np.linalg.norm(P[idx[(a, b)]] - add)))
            denom.append(float(np.linalg.norm(P[idx[(a, b)]]) + 1e-9))
    nonadd = float(np.mean(res))
    nonadd_rel = float(np.mean(np.asarray(res) / np.asarray(denom)))
    return {"shared_channel_leak": shared_leak, "nonadditivity_residual": nonadd,
            "nonadditivity_residual_rel": nonadd_rel}


# ============================ per-(grid, mix) zero-shot driver ============================
def _run_grid_mix(seed, K1, K2, m, mix, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
                  conv_tol, conv_max_epochs, batch, gen_epochs, n_draws):
    """Re-run the SAME coverage-preserving zero-shot held-out test on the mixed world at strength `mix`. Arms:
    compositional_gen (neural superposition), fixed-v2 (floor), flat (floor). Mirrors the clean zero-shot _run_grid,
    swapping the CLEAN env for the MIXED env and adding the world-is-mixed witnesses."""
    N = K1 * K2
    d_p = int(d_a) + int(d_b)
    n_in = d_p + N_ACT
    chance = 1.0 / N
    referents, attrs = _grid_facts(K1, K2)

    # --- coverage-preserving held-out split (SAME split every mix strength: depends only on seed) + asserts ---
    taught_idx, held_idx = _heldout_split(K1, K2, m, seed)
    taught_set, held_set = set(taught_idx), set(held_idx)
    disjoint = bool(len(taught_set & held_set) == 0 and len(held_set) > 0)
    trained_a = {attrs[j][0] for j in taught_idx}
    trained_b = {attrs[j][1] for j in taught_idx}
    coverage_ok = bool(all(attrs[j][0] in trained_a and attrs[j][1] in trained_b for j in held_idx))
    assert disjoint, "taught and held-out sets must be disjoint and held-out non-empty"
    assert coverage_ok, "every held-out primitive (a AND b) must appear in >= 1 taught combo"

    # --- the MIXED world (fresh env; SAME percepts across arms) ---
    env = _make_mixed_env(seed, K1, K2, d_a, d_b, noise, mix, referents, attrs)
    protos = np.stack([env.proto(referents[j]) for j in range(N)]).astype(np.float64)   # (N, d_p) test-time RULER only
    action_ctx = _action_ctx_const()

    # world-is-genuinely-mixed witnesses (true prototypes only)
    witness = _mixing_witness(env, referents, attrs, K1, K2, d_a)

    # --- ARM 1: compositional generator (neural superposition), taught on the TAUGHT set only ---
    cgen = CompositionalGenerator(gen_k, n_in, d_a, d_b, K1, K2, gen_hidden, seed, gen_settle, gen_lr, w_clip,
                                  bdsp_wmax=bdsp_wmax, conv_tol=conv_tol, conv_max_epochs=conv_max_epochs)
    fed_comp = []
    for j in taught_idx:
        a, b = attrs[j]
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)          # ONLY taught (mixed) percepts ever drawn
        cgen.learn_fact(a, b, np.asarray(Xj, dtype=np.float64).mean(axis=0), action_ctx)
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

    # --- NO-LEAKAGE assert: no held-out fact index ever entered ANY training path ---
    no_leakage = bool(not (set(fed_comp) & held_set) and not (set(fed_v2) & held_set)
                      and not (set(fed_flat) & held_set))
    assert no_leakage, "a held-out fact index leaked into a training path"

    # --- RECALL: taught (sanity) + held-out (zero-shot), nearest-prototype identity, arm-symmetric ---
    def comp_pred(j):
        a, b = attrs[j]
        return _nearest_proto(cgen.regenerate(a, b)[:d_p], protos)          # NEURAL superposition -> identify
    def v2_pred(j):
        return _nearest_proto(vgen.regenerate(j)[:d_p], protos)
    def flat_pred(j):
        return flat.recall_nearest(j, protos)

    comp_seen = _recall_fraction(taught_idx, comp_pred, protos)
    comp_held = _recall_fraction(held_idx, comp_pred, protos)
    v2_held = _recall_fraction(held_idx, v2_pred, protos)
    flat_held = _recall_fraction(held_idx, flat_pred, protos)

    # --- regeneration fidelity (cos to the TRUE mixed prototype) for the diagnostic (ruler only) ---
    comp_held_cos = float(np.mean([_cos(cgen.regenerate(*attrs[j])[:d_p], protos[j]) for j in held_idx]))

    # --- composition-is-NEURAL teeth: lesion localisation on the HELD-OUT facts ---
    held_refs = [referents[j] for j in held_idx]
    held_attrs = [attrs[j] for j in held_idx]
    lesion = _lesion_localises(cgen, held_attrs, env, held_refs, n_probe=min(8, len(held_idx)))

    # --- honest-negative diagnostic: is a comp miss low-cos (interference) or high-cos wrong-nearest (collision)? ---
    held_miss_lowcos = 0; held_miss_collision = 0
    for j in held_idx:
        if comp_pred(j) != j:
            if _cos(cgen.regenerate(*attrs[j])[:d_p], protos[j]) < 0.85:
                held_miss_lowcos += 1
            else:
                held_miss_collision += 1

    return {
        "mix": float(mix), "K1": K1, "K2": K2, "N": N, "P": K1 + K2, "chance": chance,
        "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "taught_heldout_disjoint": disjoint, "every_heldout_primitive_seen_in_taught": coverage_ok,
        "no_leakage_heldout_never_trained": no_leakage,
        # THE HEADLINE: zero-shot held-out recall vs mixing strength
        "compositional_heldout_recall": comp_held,
        "noncompositional_v2_heldout_recall": v2_held,
        "flat_heldout_recall": flat_held,
        "compositional_seen_recall": comp_seen,
        "compositional_heldout_cos": comp_held_cos,
        # composition-neural teeth
        "lesion_localises_heldout": lesion,
        # world-is-genuinely-mixed witnesses
        "world_shared_channel_leak": witness["shared_channel_leak"],
        "world_nonadditivity_residual": witness["nonadditivity_residual"],
        "world_nonadditivity_residual_rel": witness["nonadditivity_residual_rel"],
        # anti-cheat witnesses
        "compositional_stored_raw_patterns": int(cgen._stored_raw_patterns),
        "compositional_used_ruler": bool(cgen._used_ruler),
        # honest-negative diagnostic
        "heldout_miss_lowcos_interference": held_miss_lowcos,
        "heldout_miss_highcos_collision": held_miss_collision,
    }


# ============================ orchestration (one seed) ============================
def run(seed, grids, held_out, mix_sweep, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
        conv_tol, conv_max_epochs, batch, gen_epochs, n_draws):
    n_in = int(d_a) + int(d_b) + N_ACT
    Kbig = max(k1 * k2 for k1, k2 in grids)
    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, Kbig, seed, max(120, 6 * Kbig), 20,
                                                               0.5, w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()
    per_grid = {}
    for (K1, K2) in grids:
        m = held_out.get(f"{K1}x{K2}", max(1, min(K1, K2)))
        sweep = {}
        print(f"\n{'=' * 92}\n# SEED {seed}  GRID {K1}x{K2} (N={K1*K2}, P={K1+K2}, held_out={m})  MIX-SWEEP {mix_sweep}\n"
              f"{'=' * 92}", flush=True)
        for mix in mix_sweep:
            gr = _run_grid_mix(seed, K1, K2, m, mix, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                               bdsp_wmax, conv_tol, conv_max_epochs, batch, gen_epochs, n_draws)
            sweep[f"{mix:.2f}"] = gr
            print(f"  [mix {mix:.2f}] held-out recall: comp {gr['compositional_heldout_recall']:.2f} | "
                  f"v2 {gr['noncompositional_v2_heldout_recall']:.2f} | flat {gr['flat_heldout_recall']:.2f} "
                  f"(chance {gr['chance']:.3f}) | seen(comp) {gr['compositional_seen_recall']:.2f} | "
                  f"held-cos {gr['compositional_heldout_cos']:.3f} | world shared-leak "
                  f"{gr['world_shared_channel_leak']:.3f} nonadd {gr['world_nonadditivity_residual']:.3f} | "
                  f"lesion-loc {gr['lesion_localises_heldout'].get('localises')}", flush=True)
        per_grid[f"{K1}x{K2}"] = sweep
    return {"seed": seed, "grids": [f"{k1}x{k2}" for k1, k2 in grids], "mix_sweep": list(mix_sweep),
            "d_a": d_a, "d_b": d_b, "n_in": n_in,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"d_a": d_a, "d_b": d_b, "noise": noise, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "gen_settle": gen_settle, "gen_lr": gen_lr, "w_clip": w_clip, "bdsp_wmax": bdsp_wmax,
                       "conv_tol": conv_tol, "conv_max_epochs": conv_max_epochs, "batch": batch,
                       "gen_epochs": gen_epochs, "n_draws": n_draws, "held_out": held_out, "mix_sweep": list(mix_sweep)},
            "per_grid": per_grid}


# ============================ verdict (honest negative is first-class) ============================
def _verdict(result):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    grids = result["per_grid"]
    gkeys = sorted(grids, key=lambda g: grids[g][sorted(grids[g])[0]]["N"])
    mix_keys = [f"{m:.2f}" for m in result["mix_sweep"]]
    v = Verdict("teacher-loop composition INTERFERENCE (zero-shot held-out vs mixing strength)", chance=None)

    per_grid_summary = {}
    # clean baseline (mix=0) + robustness + breaking-point localisation, per grid
    robust_all = True                                # GO iff robust at EVERY mix strength on EVERY grid
    world_mixed_ok_all = True                        # the hard world is genuinely non-linear at the strong end
    for g in gkeys:
        sweep = grids[g]
        N = sweep[mix_keys[0]]["N"]; chance = sweep[mix_keys[0]]["chance"]
        clean_key = mix_keys[0]
        clean = sweep[clean_key]
        clean_comp = clean["compositional_heldout_recall"]
        # the strongest-mix world must be genuinely non-linear + shared-channel (anti-cheat)
        strong = sweep[mix_keys[-1]]
        world_mixed = bool(strong["mix"] > 0.0 and strong["world_shared_channel_leak"] > 0.05
                           and strong["world_nonadditivity_residual"] > 0.05)
        world_mixed_ok_all = world_mixed_ok_all and world_mixed
        v.require(f"[{g}] the strong-mix world is genuinely non-linear + shared-channel (not linear/disjoint)",
                  world_mixed, expect=True,
                  note=f"shared-leak {strong['world_shared_channel_leak']:.3f} "
                       f"nonadd {strong['world_nonadditivity_residual']:.3f} @ mix {strong['mix']:.2f}")
        # clean-world sanity: at mix=0 superposition still generalizes zero-shot (the 1.00 baseline, in-run)
        if abs(clean["mix"]) < 1e-9:
            v.floor(f"[{g}] CLEAN (mix=0) zero-shot held-out recall (the 1.00 baseline, in-run)", clean_comp,
                    floor=0.5)

        breaks_at = None
        per_mix = {}
        grid_robust = True
        for mk in mix_keys:
            r = sweep[mk]
            comp_h = r["compositional_heldout_recall"]; v2_h = r["noncompositional_v2_heldout_recall"]
            flat_h = r["flat_heldout_recall"]
            attributable_to(f"[{g} mix {mk}] zero-shot held-out: composing vs v2 memorizing", comp_h, v2_h)
            beats = bool(comp_h >= 0.5 and comp_h >= v2_h + 0.30 and comp_h >= flat_h + 0.30)
            if not beats and breaks_at is None and float(mk) > 0.0:
                breaks_at = float(mk)
            grid_robust = grid_robust and beats
            per_mix[mk] = {
                "mix": r["mix"], "compositional_heldout_recall": comp_h,
                "noncompositional_v2_heldout_recall": v2_h, "flat_heldout_recall": flat_h,
                "compositional_seen_recall": r["compositional_seen_recall"],
                "compositional_heldout_cos": r["compositional_heldout_cos"],
                "compositional_minus_v2": float(comp_h - v2_h), "compositional_minus_flat": float(comp_h - flat_h),
                "world_shared_channel_leak": r["world_shared_channel_leak"],
                "world_nonadditivity_residual": r["world_nonadditivity_residual"],
                "world_nonadditivity_residual_rel": r["world_nonadditivity_residual_rel"],
                "lesion_localises": bool(r["lesion_localises_heldout"].get("localises")),
                "heldout_miss_lowcos_interference": r["heldout_miss_lowcos_interference"],
                "heldout_miss_highcos_collision": r["heldout_miss_highcos_collision"],
                "beats_floors": beats,
            }
        robust_all = robust_all and grid_robust
        # NOTE: robustness is the MEASURED OUTCOME, not a validity gate -- it is fed to decide(go=...) so a break
        # earns a clean NO-GO (honest negative), NOT the UNDEFINED a failed require() would poison the verdict with.
        # A break here is EXPECTED under the honest-negative hypothesis; only the anti-cheats below gate validity.
        per_grid_summary[g] = {"N": N, "chance": chance, "held_out_n": sweep[mix_keys[0]]["held_out_n"],
                               "clean_heldout_recall": clean_comp, "breaks_at": breaks_at,
                               "world_mixed_at_strong": world_mixed, "per_mix": per_mix}

    # composition stays NEURAL regardless of world (lesion localises at strong mix) + anti-cheats
    strong_lesion_ok = all(bool(grids[g][mix_keys[-1]]["lesion_localises_heldout"].get("localises")) for g in gkeys)
    not_buffer = all(int(grids[g][mk]["compositional_stored_raw_patterns"]) == 0 for g in gkeys for mk in mix_keys)
    no_ruler = all(not bool(grids[g][mk]["compositional_used_ruler"]) for g in gkeys for mk in mix_keys)
    disj = all(bool(grids[g][mk]["taught_heldout_disjoint"]) for g in gkeys for mk in mix_keys)
    cover = all(bool(grids[g][mk]["every_heldout_primitive_seen_in_taught"]) for g in gkeys for mk in mix_keys)
    noleak = all(bool(grids[g][mk]["no_leakage_heldout_never_trained"]) for g in gkeys for mk in mix_keys)
    v.require("composition NEURAL at strong mix (lesion localises on held-out)", strong_lesion_ok, expect=True)
    v.require("0 stored raw patterns at every mix (composes, not a buffer)", not_buffer, expect=True)
    v.require("consolidation never read the ruler at any mix", no_ruler, expect=True)
    v.require("taught/held-out DISJOINT at every mix", disj, expect=True)
    v.require("every held-out primitive seen in >=1 taught combo at every mix", cover, expect=True)
    v.require("NO leakage (held-out never trained) at every mix", noleak, expect=True)
    v.require("(seed) substrate byte-identical", bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    anticheats = bool(world_mixed_ok_all and strong_lesion_ok and not_buffer and no_ruler and disj and cover
                      and noleak and result["substrate_byte_identical"] and result["sim_diff_empty"])
    go = bool(robust_all and anticheats)
    # first breaking point across grids (the located s*)
    breaks = [per_grid_summary[g]["breaks_at"] for g in gkeys if per_grid_summary[g]["breaks_at"] is not None]
    first_break = min(breaks) if breaks else None
    decision = v.decide(go=go)
    # GO/PARTIAL/NEGATIVE (honest negative is first-class). GO = robust at every s (+ anti-cheats). PARTIAL = holds
    # through the first non-zero (mild) mix but breaks by a stronger one -> breaking point LOCATED. NEGATIVE =
    # breaks at the very first non-zero mix -> superposition insufficient for even mild non-linear interference.
    nonzero_mix = [float(mk) for mk in mix_keys if float(mk) > 0.0]
    first_nonzero = min(nonzero_mix) if nonzero_mix else None
    if go:
        go_status = "GO"
    elif not anticheats:
        go_status = "UNDEFINED"
    elif first_break is not None and first_nonzero is not None and first_break > first_nonzero + 1e-9:
        go_status = "PARTIAL"
    else:
        go_status = "NEGATIVE"
    return {"grids": gkeys, "mix_keys": mix_keys, "per_grid": per_grid_summary,
            "robust_all_mix": robust_all, "world_mixed_ok_all": world_mixed_ok_all,
            "anticheats_ok": anticheats, "first_break_s": first_break, "go_status": go_status,
            "substrate_byte_identical": result["substrate_byte_identical"],
            "sim_diff_empty": result["sim_diff_empty"], **decision}


def _parse_grid(s):
    a, b = s.lower().split("x"); return (int(a), int(b))


def _one_seed(a, seed, grids, held_out, mix_sweep):
    result = run(seed, grids, held_out, mix_sweep, a.d_a, a.d_b, a.noise, a.gen_hidden, a.gen_k, a.gen_settle,
                 a.gen_lr, a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.batch, a.gen_epochs, a.n_draws)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop COMPOSITION INTERFERENCE: re-run the zero-shot held-out "
                                             "test on a HARDER non-linearly-mixed shared-channel world, SWEEPING "
                                             "mixing strength to locate where neural superposition breaks.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--grids", nargs="+", default=["5x5", "6x6"])
    ap.add_argument("--held-out", nargs="+", default=["5x5:5", "6x6:8"],
                    help="per-grid held-out count GRID:M (coverage-preserving; each held-out primitive stays taught)")
    ap.add_argument("--mix-sweep", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="mixing strengths (0=clean disjoint baseline -> 1=strong shared-channel non-linear mix)")
    ap.add_argument("--d-a", type=int, default=8)
    ap.add_argument("--d-b", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--gen-hidden", type=int, default=96)
    ap.add_argument("--gen-k", type=int, default=64)
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--conv-tol", type=float, default=0.02)
    ap.add_argument("--conv-max-epochs", type=int, default=200)
    ap.add_argument("--gen-epochs", type=int, default=16)
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
    mix_sweep = list(a.mix_sweep)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  grids={a.grids} held_out={a.held_out} mix_sweep={mix_sweep}\n"
              + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, grids, held_out, mix_sweep)
        summary = {"probe": "teacher_loop_composition_interference", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "grids": a.grids, "held_out": a.held_out, "mix_sweep": mix_sweep,
                   "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        for g in rv["grids"]:
            pg = rv["per_grid"][g]
            row = " ".join(f"s{mk}:{pg['per_mix'][mk]['compositional_heldout_recall']:.2f}" for mk in rv["mix_keys"])
            print(f"[interf] seed {s} {g}: N={pg['N']} | clean {pg['clean_heldout_recall']:.2f} | held-out-vs-mix "
                  f"[{row}] | breaks@s={pg['breaks_at']}", flush=True)
        print(f"[interf] seed {s} robust {rv['robust_all_mix']} world-mixed {rv['world_mixed_ok_all']} "
              f"first-break s={rv['first_break_s']} byte-id {rv['substrate_byte_identical']} "
              f"sim-clean {rv['sim_diff_empty']} | VERDICT {rv['status']}", flush=True)
        print(f"[interf] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        agg = {"probe": "teacher_loop_composition_interference_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "grids": a.grids, "held_out": a.held_out,
               "mix_sweep": mix_sweep, "go_count": go_n, "n_seeds": len(seeds), "per_grid_means": {},
               "per_seed": per_seed}
        gkeys = per_seed[0]["verdict"]["grids"]; mix_keys = per_seed[0]["verdict"]["mix_keys"]
        for g in gkeys:
            agg["per_grid_means"][g] = {"N": per_seed[0]["verdict"]["per_grid"][g]["N"],
                                        "chance": per_seed[0]["verdict"]["per_grid"][g]["chance"], "per_mix": {}}
            for mk in mix_keys:
                comp = [p["verdict"]["per_grid"][g]["per_mix"][mk]["compositional_heldout_recall"] for p in per_seed]
                v2 = [p["verdict"]["per_grid"][g]["per_mix"][mk]["noncompositional_v2_heldout_recall"] for p in per_seed]
                flat = [p["verdict"]["per_grid"][g]["per_mix"][mk]["flat_heldout_recall"] for p in per_seed]
                seen = [p["verdict"]["per_grid"][g]["per_mix"][mk]["compositional_seen_recall"] for p in per_seed]
                nonadd = [p["verdict"]["per_grid"][g]["per_mix"][mk]["world_nonadditivity_residual"] for p in per_seed]
                shared = [p["verdict"]["per_grid"][g]["per_mix"][mk]["world_shared_channel_leak"] for p in per_seed]
                agg["per_grid_means"][g]["per_mix"][mk] = {
                    "compositional_heldout_recall_mean": float(np.nanmean(comp)),
                    "compositional_heldout_recall_per_seed": [float(x) for x in comp],
                    "noncompositional_v2_heldout_recall_mean": float(np.nanmean(v2)),
                    "flat_heldout_recall_mean": float(np.nanmean(flat)),
                    "compositional_seen_recall_mean": float(np.nanmean(seen)),
                    "world_nonadditivity_residual_mean": float(np.nanmean(nonadd)),
                    "world_shared_channel_leak_mean": float(np.nanmean(shared)),
                }
        breaks = [p["verdict"]["first_break_s"] for p in per_seed if p["verdict"]["first_break_s"] is not None]
        agg["median_first_break_s"] = float(np.median(breaks)) if breaks else None
        agg["n_seeds_with_break"] = len(breaks)
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[interf AGG] GO {go_n}/{len(seeds)} | median-first-break s={agg['median_first_break_s']} "
              f"({agg['n_seeds_with_break']}/{len(seeds)} seeds break)", flush=True)
        for g in gkeys:
            print(f"   {g}: N={agg['per_grid_means'][g]['N']} (chance {agg['per_grid_means'][g]['chance']:.3f})",
                  flush=True)
            for mk in mix_keys:
                mm = agg["per_grid_means"][g]["per_mix"][mk]
                print(f"      mix {mk}: comp {mm['compositional_heldout_recall_mean']:.2f} "
                      f"(per-seed {mm['compositional_heldout_recall_per_seed']}) vs v2 "
                      f"{mm['noncompositional_v2_heldout_recall_mean']:.2f} vs flat "
                      f"{mm['flat_heldout_recall_mean']:.2f} | seen {mm['compositional_seen_recall_mean']:.2f} | "
                      f"world nonadd {mm['world_nonadditivity_residual_mean']:.3f} shared "
                      f"{mm['world_shared_channel_leak_mean']:.3f}", flush=True)
        print(f"[interf AGG] wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
