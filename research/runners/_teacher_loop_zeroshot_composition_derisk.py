"""TEACHER-LOOP ZERO-SHOT COMPOSITION DE-RISK (2026-08-09): the DEFINITIVE test of the compositional lever.

WHERE THE ARC STANDS. The compositional generator (e4417698d, PARTIAL-GO) stores P=K1+K2 PRIMITIVE engrams
(sub-linear O(sqrt(N))) and regenerates a fact by NEURAL SUPERPOSITION of its two primitive spiking-readout outputs.
Verified there: store genuinely O(sqrt(N)); composition NEURAL (lesion-localised to each primitive block); 0 stored
raw patterns; retention parity with the flat store. HONEST negative there: at small low-rank N it is NOT a retention
WIN over a fixed generator -- because the facts are low-rank, a fixed readout fits them too.

THE UNIQUE ADVANTAGE a fixed generator CANNOT have: ZERO-SHOT COMPOSITION -- regenerate a combination NEVER taught as
a fact, from primitives seen in OTHER combinations. This is the definitive test of the compositional lever AND the
mission core (compositional generalization = the composer, being biologized; the spiking-superposition generator IS a
biological VSA composer).

THE HYPOTHESIS. Train on a SUBSET of the KxK grid such that EVERY primitive appears in at least one taught combination,
but M target combinations are HELD OUT (never taught as facts). At test, regenerate + recall the HELD-OUT combinations
by COMPOSING their two (individually-seen) primitives. A fixed non-compositional generator (v2) and the flat store
CANNOT (they address facts by CLASS INDEX, never saw those classes) -> ~chance on held-out. Compositional >> fixed/flat
on HELD-OUT = genuine NEURAL compositional generalization.

THE MEASURE. RECALL of a fact = regenerate its percept, then NEAREST-PROTOTYPE identify it among ALL N true prototypes
(chance = 1/N). This is ARM-SYMMETRIC (every arm produces a reconstruction, every arm is scored the same way):
  * compositional_gen: regenerate(a,b) = NEURAL SUPERPOSITION of primitive-a's and primitive-b's spiking readouts
    (CompositionalGenerator.regenerate, lines ~239-254 of _teacher_loop_compositional_generator_derisk: two
    self._elig(...) spiking-reservoir forwards + a SUM of the two leaky-readout outputs). Held-out (a,b): both
    primitives were seen in OTHER taught combos -> the engrams exist -> it COMPOSES the never-taught percept.
  * noncompositional_v2 (FLOOR): the fixed GenerativeReplayNetV2 addresses facts by CLASS INDEX j. Held-out class j
    was never trained -> its readout is the untrained near-anchor output -> ~chance identity.
  * flat (FLOOR): an O(N) raw-engram buffer keyed by class index. No entry for a held-out class -> it can only ever
    output a TAUGHT fact, never the held-out one -> its best strategy is a uniform guess -> chance (1/N).

ANTI-CHEATS (each a REAL assertion in the output):
  * taught / held-out DISJOINT (asserted).
  * every held-out primitive (its a AND its b) appears in >= 1 TAUGHT combo (asserted; else it is not a compositional
    test but an unseen primitive). The held-out split is built to GUARANTEE this: it never removes the last taught
    cell of any row or column.
  * NO LEAKAGE: the full held-out PERCEPT never enters ANY training path -- only taught fact indices are fed to
    _corrective_batch / learn_fact (the fed-index set is asserted DISJOINT from the held-out set). env.proto of a
    held-out fact is read ONLY by the test-time nearest-prototype RULER, never by a learning call.
  * composition at test is NEURAL: regeneration is a SUM of two spiking-reservoir leaky-readout outputs; a LESION
    zeroing one primitive's engram breaks ONLY that primitive's block on the HELD-OUT facts (localisation) -> each
    block is carried by a distinct neural engram, not a joint lookup.
  * 0 stored raw patterns (compositional composes, never buffers the composed percepts); ruler untouched by learning.
  * cfg.seed byte-identical substrate (NOT actual_seed_used); de-clamped bdsp_wmax=1e9 (the -6/+6 clamp silences the
    reservoir, bound-trap 8ca014ff2); git diff main -- sim/ empty; backend recorded.

GO (per grid, per seed): compositional HELD-OUT recall >= 0.5 AND >= v2 held-out + 0.30 AND >= flat held-out + 0.30
(the floors near chance); compositional TAUGHT (seen) recall >= 0.85 (sanity: it still recalls what it was taught);
lesion localises on held-out (composition neural); disjoint + coverage + no-leakage all hold; 0 raw patterns; ruler
untouched; byte-identical; sim/ clean. If the spiking superposition does NOT generalize zero-shot (held-out recall
near the floor), that is an HONEST NEGATIVE naming WHY (low regeneration cos = binding interference; high cos but wrong
nearest = primitive-code collision).

DR grounding: van de Ven 2020 generative replay (doi:10.1038/s41467-020-17866-2); compositional-generalization
literature; the project composer/VSA notes (the spiking-superposition generator as a biological VSA bundling composer).

DISCIPLINE: reuse-by-import (the compositional generator + its env/world + neural regenerate + lesion check; the v2
fixed generator = the floor; the scaling teacher machinery; the byte-identical + sim-clean anti-cheat asserts). NO
sim/ edit. SIM_BACKEND=numpy.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_zeroshot_composition_derisk --seed 42 \
      --grids 5x5 6x6 --gen-hidden 96 --gen-k 64 \
      --out research/findings/raw/teacher_loop_zeroshot_composition_s42.json
  6-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_zeroshot_composition_derisk --seeds 42 43 44 45 46 47 \
      --grids 5x5 6x6 --gen-hidden 96 --gen-k 64 \
      --out research/findings/raw/teacher_loop_zeroshot_composition.json
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
# reuse-by-import: the compositional generator + its compositional WORLD + neural regenerate + the lesion-localisation
# check; the v2 fixed generator (the FLOOR); the scaling teacher machinery; the byte-identical + sim-clean asserts.
from research.runners._teacher_loop_compositional_generator_derisk import (  # noqa: E402
    CompositionalGenerator, CompositionalReferentEnv, _make_comp_env, _grid_facts, _action_ctx_const,
    _lesion_localises,
)
from research.runners._teacher_loop_generative_replay_v2_derisk import GenerativeReplayNetV2  # noqa: E402
from research.runners._teacher_loop_generative_replay_derisk import _cos  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import _corrective_batch, N_ACT  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import (  # noqa: E402
    _assert_byte_identical_substrate, _git_sim_diff_empty,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_zeroshot_composition.json"


# ============================ the HELD-OUT split (coverage-preserving) ============================
def _heldout_split(K1, K2, m, seed):
    """Choose M held-out (a,b) cells such that EVERY primitive still appears in >= 1 TAUGHT combo: greedily hold out a
    cell ONLY if its row a and its column b each keep >= 1 taught cell after removal. Deterministic per seed. Returns
    (taught_idx, heldout_idx) as class-index lists (class = a*K2 + b, matching _grid_facts ordering)."""
    rng = np.random.default_rng(int(seed) + 5551)
    cells = [(a, b) for a in range(K1) for b in range(K2)]
    rng.shuffle(cells)
    row_left = {a: K2 for a in range(K1)}                 # taught cells remaining in each row
    col_left = {b: K1 for b in range(K2)}                 # taught cells remaining in each column
    held = []
    for (a, b) in cells:
        if len(held) >= m:
            break
        if row_left[a] > 1 and col_left[b] > 1:           # keep >= 1 taught cell in this row AND this column
            held.append((a, b)); row_left[a] -= 1; col_left[b] -= 1
    held_idx = sorted(a * K2 + b for (a, b) in held)
    taught_idx = [j for j in range(K1 * K2) if j not in set(held_idx)]
    return taught_idx, held_idx


# ============================ the flat O(N) FLOOR (raw buffer keyed by class index) ============================
class FlatStore:
    """The O(N) target as a raw-engram buffer keyed by CLASS INDEX (the flat CLS store, distilled to its recall
    behavior). It can regenerate a TAUGHT class exactly (its stored engram), but a HELD-OUT class has NO entry -- it
    can only ever output a TAUGHT fact, never the held-out one. So its best zero-shot strategy is a uniform guess over
    the N classes -> chance (1/N). This is the honest floor a non-compositional instance store bottoms out at."""

    def __init__(self, d_p, seed):
        self.d_p = int(d_p)
        self.store = {}                                    # class index -> stored percept engram (taught only)
        self._rng = np.random.default_rng(int(seed) + 7777)

    def learn(self, cls, engram):
        self.store[int(cls)] = np.asarray(engram, dtype=np.float64)[:self.d_p].copy()

    def recall_nearest(self, cls, protos):
        """RECALL of a fact by class index. Taught -> nearest-prototype of the stored engram. Held-out (no entry) ->
        uniform guess over all N classes (the store has no held-out representation; a taught retrieval could never
        identify AS the held-out class, so chance is the fairest floor)."""
        cls = int(cls)
        if cls in self.store:
            d = np.linalg.norm(protos - self.store[cls][None, :], axis=1)
            return int(np.argmin(d))
        return int(self._rng.integers(protos.shape[0]))    # no held-out representation -> uniform guess = chance


# ============================ recall helpers ============================
def _nearest_proto(vec_dp, protos):
    d = np.linalg.norm(protos - np.asarray(vec_dp, dtype=np.float64)[None, :], axis=1)
    return int(np.argmin(d))


def _recall_fraction(idxs, predict_fn, protos):
    """Fraction of `idxs` whose predicted nearest-prototype identity == the fact itself (recall)."""
    if not idxs:
        return float("nan")
    ok = sum(int(predict_fn(j) == j) for j in idxs)
    return float(ok / len(idxs))


# ============================ per-grid zero-shot driver ============================
def _run_grid(seed, K1, K2, m, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
              conv_tol, conv_max_epochs, batch, gen_epochs, n_draws):
    N = K1 * K2
    d_p = int(d_a) + int(d_b)
    n_in = d_p + N_ACT
    chance = 1.0 / N
    referents, attrs = _grid_facts(K1, K2)

    # --- the held-out split + anti-cheat asserts (disjoint + coverage) ---
    taught_idx, held_idx = _heldout_split(K1, K2, m, seed)
    taught_set, held_set = set(taught_idx), set(held_idx)
    disjoint = bool(len(taught_set & held_set) == 0 and len(held_set) > 0)
    trained_a = {attrs[j][0] for j in taught_idx}
    trained_b = {attrs[j][1] for j in taught_idx}
    coverage_ok = bool(all(attrs[j][0] in trained_a and attrs[j][1] in trained_b for j in held_idx))
    assert disjoint, "taught and held-out sets must be disjoint and held-out non-empty"
    assert coverage_ok, "every held-out primitive (a AND b) must appear in >= 1 taught combo"

    # --- the world (fresh env; SAME percepts across arms) ---
    env = _make_comp_env(seed, K1, K2, d_a, d_b, noise, referents, attrs)
    protos = np.stack([env.proto(referents[j]) for j in range(N)]).astype(np.float64)   # (N, d_p) test-time RULER only
    action_ctx = _action_ctx_const()

    # --- ARM 1: compositional generator, taught on the TAUGHT set only ---
    cgen = CompositionalGenerator(gen_k, n_in, d_a, d_b, K1, K2, gen_hidden, seed, gen_settle, gen_lr, w_clip,
                                  bdsp_wmax=bdsp_wmax, conv_tol=conv_tol, conv_max_epochs=conv_max_epochs)
    fed_comp = []
    for j in taught_idx:
        a, b = attrs[j]
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)          # ONLY taught percepts ever drawn
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
        return _nearest_proto(vgen.regenerate(j)[:d_p], protos)             # class-index readout (untrained if held-out)

    def flat_pred(j):
        return flat.recall_nearest(j, protos)

    comp_seen = _recall_fraction(taught_idx, comp_pred, protos)
    comp_held = _recall_fraction(held_idx, comp_pred, protos)
    v2_seen = _recall_fraction(taught_idx, v2_pred, protos)
    v2_held = _recall_fraction(held_idx, v2_pred, protos)
    flat_seen = _recall_fraction(taught_idx, flat_pred, protos)
    flat_held = _recall_fraction(held_idx, flat_pred, protos)

    # --- regeneration fidelity (cosine to the TRUE composed prototype) for the diagnostic (ruler only) ---
    comp_held_cos = float(np.mean([_cos(cgen.regenerate(*attrs[j])[:d_p], protos[j]) for j in held_idx]))
    comp_seen_cos = float(np.mean([_cos(cgen.regenerate(*attrs[j])[:d_p], protos[j]) for j in taught_idx]))
    v2_held_cos = float(np.mean([_cos(vgen.regenerate(j)[:d_p], protos[j]) for j in held_idx]))

    # --- composition-is-NEURAL teeth: lesion localisation on the HELD-OUT facts specifically ---
    held_refs = [referents[j] for j in held_idx]
    held_attrs = [attrs[j] for j in held_idx]
    lesion = _lesion_localises(cgen, held_attrs, env, held_refs, n_probe=min(8, len(held_idx)))

    # --- diagnostic for an HONEST NEGATIVE: is a comp miss low-cos (interference) or high-cos wrong-nearest (collision)? ---
    held_miss_lowcos = 0; held_miss_collision = 0
    for j in held_idx:
        pj = comp_pred(j)
        if pj != j:
            if _cos(cgen.regenerate(*attrs[j])[:d_p], protos[j]) < 0.85:
                held_miss_lowcos += 1
            else:
                held_miss_collision += 1

    return {
        "K1": K1, "K2": K2, "N": N, "P": K1 + K2, "chance": chance, "n_in": n_in, "d_p": d_p,
        "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "held_out_idx": held_idx, "held_out_attrs": [list(attrs[j]) for j in held_idx],
        "taught_heldout_disjoint": disjoint, "every_heldout_primitive_seen_in_taught": coverage_ok,
        "no_leakage_heldout_never_trained": no_leakage,
        # THE HEADLINE: zero-shot held-out recall, compositional vs the two floors
        "compositional_heldout_recall": comp_held,
        "noncompositional_v2_heldout_recall": v2_held,
        "flat_heldout_recall": flat_held,
        # sanity: taught (seen) recall
        "compositional_seen_recall": comp_seen,
        "noncompositional_v2_seen_recall": v2_seen,
        "flat_seen_recall": flat_seen,
        # regeneration fidelity diagnostics
        "compositional_heldout_cos": comp_held_cos, "compositional_seen_cos": comp_seen_cos,
        "noncompositional_v2_heldout_cos": v2_held_cos,
        # composition-neural teeth
        "lesion_localises_heldout": lesion,
        # anti-cheat witnesses
        "compositional_stored_raw_patterns": int(cgen._stored_raw_patterns),
        "compositional_used_ruler": bool(cgen._used_ruler),
        # honest-negative diagnostic
        "heldout_miss_lowcos_interference": held_miss_lowcos,
        "heldout_miss_highcos_collision": held_miss_collision,
    }


# ============================ verdict ============================
def _verdict(result):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    grids = result["per_grid"]
    gkeys = sorted(grids, key=lambda g: grids[g]["N"])
    v = Verdict("teacher-loop ZERO-SHOT composition (never-taught combinations from seen primitives)", chance=None)

    per_grid_summary = {}
    all_go = True
    for g in gkeys:
        gr = grids[g]
        N = gr["N"]; chance = gr["chance"]
        comp_h = gr["compositional_heldout_recall"]; v2_h = gr["noncompositional_v2_heldout_recall"]
        flat_h = gr["flat_heldout_recall"]; comp_s = gr["compositional_seen_recall"]
        lesion_ok = bool(gr["lesion_localises_heldout"].get("localises"))
        disjoint = gr["taught_heldout_disjoint"]; cover = gr["every_heldout_primitive_seen_in_taught"]
        noleak = gr["no_leakage_heldout_never_trained"]
        not_buffer = bool(gr["compositional_stored_raw_patterns"] == 0)
        no_ruler = bool(not gr["compositional_used_ruler"])

        # attribution: compositional zero-shot beats each floor
        attributable_to(f"[{g}] zero-shot held-out: compositional composing vs v2 memorizing",
                        comp_h, v2_h)
        attributable_to(f"[{g}] zero-shot held-out: compositional composing vs flat instance store",
                        comp_h, flat_h)

        beats_v2 = bool(comp_h >= v2_h + 0.30)
        beats_flat = bool(comp_h >= flat_h + 0.30)
        comp_ok = bool(comp_h >= 0.5)
        seen_ok = bool(comp_s >= 0.85)

        v.require(f"[{g}] compositional HELD-OUT recall >= 0.5 (zero-shot composition)", comp_ok, expect=True,
                  note=f"comp {comp_h:.2f} (chance {chance:.3f}) N={N}")
        v.require(f"[{g}] compositional held-out >= v2 held-out + 0.30 (composes, not memorizes)", beats_v2,
                  expect=True, note=f"comp {comp_h:.2f} vs v2 {v2_h:.2f}")
        v.require(f"[{g}] compositional held-out >= flat held-out + 0.30 (beats the instance floor)", beats_flat,
                  expect=True, note=f"comp {comp_h:.2f} vs flat {flat_h:.2f}")
        v.require(f"[{g}] compositional SEEN (taught) recall >= 0.85 (sanity)", seen_ok, expect=True,
                  note=f"seen {comp_s:.2f}")
        v.require(f"[{g}] composition NEURAL on held-out (lesion localises)", lesion_ok, expect=True,
                  note=f"A:{gr['lesion_localises_heldout'].get('lesionA_Ablock_delta',0):.2f}/"
                       f"{gr['lesion_localises_heldout'].get('lesionA_Bblock_delta',0):.2f} "
                       f"B:{gr['lesion_localises_heldout'].get('lesionB_Bblock_delta',0):.2f}/"
                       f"{gr['lesion_localises_heldout'].get('lesionB_Ablock_delta',0):.2f}")
        v.require(f"[{g}] taught/held-out DISJOINT", bool(disjoint), expect=True)
        v.require(f"[{g}] every held-out primitive seen in >= 1 taught combo", bool(cover), expect=True)
        v.require(f"[{g}] NO leakage (held-out never trained)", bool(noleak), expect=True)
        v.require(f"[{g}] 0 stored raw patterns (composes, not a buffer)", not_buffer, expect=True)
        v.require(f"[{g}] consolidation never read the ruler", no_ruler, expect=True)

        grid_go = bool(comp_ok and beats_v2 and beats_flat and seen_ok and lesion_ok and disjoint and cover
                       and noleak and not_buffer and no_ruler)
        all_go = all_go and grid_go
        per_grid_summary[g] = {
            "N": N, "P": gr["P"], "chance": chance, "held_out_n": gr["held_out_n"], "taught_n": gr["taught_n"],
            "compositional_heldout_recall": comp_h, "noncompositional_v2_heldout_recall": v2_h,
            "flat_heldout_recall": flat_h, "compositional_seen_recall": comp_s,
            "compositional_minus_v2": float(comp_h - v2_h), "compositional_minus_flat": float(comp_h - flat_h),
            "compositional_heldout_cos": gr["compositional_heldout_cos"],
            "noncompositional_v2_heldout_cos": gr["noncompositional_v2_heldout_cos"],
            "lesion_localises": lesion_ok,
            "heldout_miss_lowcos_interference": gr["heldout_miss_lowcos_interference"],
            "heldout_miss_highcos_collision": gr["heldout_miss_highcos_collision"],
            "grid_go": grid_go,
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
def run(seed, grids, held_out, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
        conv_tol, conv_max_epochs, batch, gen_epochs, n_draws):
    n_in = int(d_a) + int(d_b) + N_ACT
    Kbig = max(k1 * k2 for k1, k2 in grids)
    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, Kbig, seed, max(120, 6 * Kbig), 20,
                                                               0.5, w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()
    per_grid = {}
    for (K1, K2) in grids:
        m = held_out.get(f"{K1}x{K2}", max(1, min(K1, K2)))
        print(f"\n{'=' * 90}\n# SEED {seed}  GRID {K1}x{K2} (N={K1*K2}, P={K1+K2}, held_out={m})\n{'=' * 90}", flush=True)
        gr = _run_grid(seed, K1, K2, m, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip, bdsp_wmax,
                       conv_tol, conv_max_epochs, batch, gen_epochs, n_draws)
        per_grid[f"{K1}x{K2}"] = gr
        print(f"  [grid {K1}x{K2} N={gr['N']}] held-out recall: comp {gr['compositional_heldout_recall']:.2f} | "
              f"v2 {gr['noncompositional_v2_heldout_recall']:.2f} | flat {gr['flat_heldout_recall']:.2f} "
              f"(chance {gr['chance']:.3f}) | seen(comp) {gr['compositional_seen_recall']:.2f} | "
              f"held-cos {gr['compositional_heldout_cos']:.3f} | lesion-loc "
              f"{gr['lesion_localises_heldout'].get('localises')}", flush=True)
    return {"seed": seed, "grids": [f"{k1}x{k2}" for k1, k2 in grids], "d_a": d_a, "d_b": d_b, "n_in": n_in,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"d_a": d_a, "d_b": d_b, "noise": noise, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "gen_settle": gen_settle, "gen_lr": gen_lr, "w_clip": w_clip, "bdsp_wmax": bdsp_wmax,
                       "conv_tol": conv_tol, "conv_max_epochs": conv_max_epochs, "batch": batch,
                       "gen_epochs": gen_epochs, "n_draws": n_draws, "held_out": held_out, "frozen_hidden": True},
            "per_grid": per_grid}


def _parse_grid(s):
    a, b = s.lower().split("x"); return (int(a), int(b))


def _one_seed(a, seed, grids, held_out):
    result = run(seed, grids, held_out, a.d_a, a.d_b, a.noise, a.gen_hidden, a.gen_k, a.gen_settle, a.gen_lr,
                 a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.batch, a.gen_epochs, a.n_draws)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop ZERO-SHOT COMPOSITION: regenerate + recall NEVER-TAUGHT "
                                             "combinations from primitives seen in OTHER combinations (neural "
                                             "superposition), vs a fixed generator + flat store (the floors).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--grids", nargs="+", default=["5x5", "6x6"], help="K1xK2 grids where a held-out split is meaningful")
    ap.add_argument("--held-out", nargs="+", default=["5x5:5", "6x6:8"],
                    help="per-grid held-out count as GRID:M (coverage-preserving; each held-out primitive stays taught)")
    ap.add_argument("--d-a", type=int, default=8, help="attribute-A block dim")
    ap.add_argument("--d-b", type=int, default=8, help="attribute-B block dim")
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--gen-hidden", type=int, default=96, help="FIXED generator reservoir size (H_gen)")
    ap.add_argument("--gen-k", type=int, default=64, help="FIXED query address width (sparse collision-free codes)")
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
    grids = [_parse_grid(g) for g in a.grids]
    held_out = {}
    for spec in a.held_out:
        k, mm = spec.split(":"); held_out[k.lower()] = int(mm)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  grids={a.grids} held_out={a.held_out} gen_H={a.gen_hidden} "
              f"gen_k={a.gen_k}\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, grids, held_out)
        summary = {"probe": "teacher_loop_zeroshot_composition", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "grids": a.grids, "held_out": a.held_out, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        for g in rv["grids"]:
            pg = rv["per_grid"][g]
            print(f"[zeroshot] seed {s} {g}: N={pg['N']} held={pg['held_out_n']} | HELD-OUT recall comp "
                  f"{pg['compositional_heldout_recall']:.2f} vs v2 {pg['noncompositional_v2_heldout_recall']:.2f} vs "
                  f"flat {pg['flat_heldout_recall']:.2f} (chance {pg['chance']:.3f}) | seen(comp) "
                  f"{pg['compositional_seen_recall']:.2f} | comp-v2 +{pg['compositional_minus_v2']:.2f} comp-flat "
                  f"+{pg['compositional_minus_flat']:.2f} | lesion {pg['lesion_localises']} | GO {pg['grid_go']}",
                  flush=True)
        print(f"[zeroshot] seed {s} byte-id {rv['substrate_byte_identical']} sim-clean {rv['sim_diff_empty']} | "
              f"VERDICT {rv['status']}", flush=True)
        print(f"[zeroshot] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        agg = {"probe": "teacher_loop_zeroshot_composition_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "grids": a.grids, "held_out": a.held_out,
               "go_count": go_n, "n_seeds": len(seeds), "per_grid_means": {}, "per_seed": per_seed}
        for g in per_seed[0]["verdict"]["grids"]:
            comp = [p["verdict"]["per_grid"][g]["compositional_heldout_recall"] for p in per_seed]
            v2 = [p["verdict"]["per_grid"][g]["noncompositional_v2_heldout_recall"] for p in per_seed]
            flat = [p["verdict"]["per_grid"][g]["flat_heldout_recall"] for p in per_seed]
            seen = [p["verdict"]["per_grid"][g]["compositional_seen_recall"] for p in per_seed]
            agg["per_grid_means"][g] = {
                "N": per_seed[0]["verdict"]["per_grid"][g]["N"],
                "chance": per_seed[0]["verdict"]["per_grid"][g]["chance"],
                "held_out_n": per_seed[0]["verdict"]["per_grid"][g]["held_out_n"],
                "compositional_heldout_recall_mean": float(np.nanmean(comp)),
                "compositional_heldout_recall_per_seed": [float(x) for x in comp],
                "noncompositional_v2_heldout_recall_mean": float(np.nanmean(v2)),
                "flat_heldout_recall_mean": float(np.nanmean(flat)),
                "compositional_seen_recall_mean": float(np.nanmean(seen)),
            }
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[zeroshot AGG] GO {go_n}/{len(seeds)}", flush=True)
        for g, mm in agg["per_grid_means"].items():
            print(f"   {g}: N={mm['N']} held={mm['held_out_n']} | HELD-OUT comp "
                  f"{mm['compositional_heldout_recall_mean']:.2f} vs v2 "
                  f"{mm['noncompositional_v2_heldout_recall_mean']:.2f} vs flat {mm['flat_heldout_recall_mean']:.2f} "
                  f"(chance {mm['chance']:.3f}) | seen {mm['compositional_seen_recall_mean']:.2f} | "
                  f"per-seed {mm['compositional_heldout_recall_per_seed']}", flush=True)
        print(f"[zeroshot AGG] wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
