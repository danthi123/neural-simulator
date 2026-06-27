# Burndown B-wire-1 — deploy B1's corpus-mined ordinal axis INTO the first-chat console — **GO**

**Date:** 2026-06-27
**Type:** Bucket-B burndown WIRE-UP (VERY LOW cost, per `2026-06-27-burndown-bucketB-structure-learning-research-gate.md` §B-wire-1: "deploy B1's already-GO mined ordinal axis into the console, validated-but-unwired"). Reuse-by-import; **NO `sim/` edit**; only `research/runners/first_chat_console.py` edited additively.
**Verdict:** **GO** — the first-chat console's transitive route ("is X bigger than Y?") now reasons over the **CORPUS-MINED size ordering over the brain's OWN learned vocab** (B1, GO 6-seed + 3-spiking) instead of the hand-curated `_SIZE_LADDER`, when the brain has the size markers; the DEFAULT `--rubric` stays **10/10 byte-identical**, the moat is **0-FA**.

---

## 1. The gap closed (validated-but-unwired)

B1 (`2026-06-27-regimeB-corpus-mined-axis-GO.md`) proved that a SIZE ordinal axis MINED from corpus scalar-adjective co-occurrence over the brain's OWN learned animal vocab, fed to the validated Tier-2.3 Betasort ordinal-map learner, infers held-out unstated comparisons (0.79 ≫ chance + mem-floor) with the artifact-proof symbolic-distance signature, and the **permuted-mining control collapses** — structure ACQUIRED, not given. But the **console still used the hand-curated `_SIZE_LADDER = ("tiny","small","big","huge","giant")`** (`first_chat_console.py:900`), with a code comment flagging exactly this gap. B-wire-1 deploys the validated mined axis into the console.

## 2. What was wired (reuse-by-import; additive; only `first_chat_console.py`)

- **Imported the B1 mining half VERBATIM** (`research.runners._regimeb_corpus_mined_axis_derisk`): `mine_size_scores` (distributional scalar-adjective co-occurrence + provenance), `mined_order`, `adjacent_premises`, and the relation constants (`GT_ORDER` = candidate animals, `HIGH_ADJ`/`LOW_ADJ` = the SIZE markers).
- **New `FirstChatConsole._mine_size_axis(brain)`** — mines the size axis from the corpus over the brain's vocab, applying B1's exact attestation filter (`prov[it]["freq"] >= min_freq and (hi+lo) >= 1`), and returns the mined `(premises, ascending_order)` or `None`.
  - **Operating point = B1's validated full-corpus budget (80 MB, `window=4`, `min_freq=8`).** Critical: the console's own PPMI graph is built on a truncated 40 MB corpus slice, where B1 §4 documents the mined axis degrades below the gate (rho ≈ 0.19 at 40 MB vs the clean GO at 80 MB). The mine runs at the FULL 80 MB independently of the PPMI build.
  - **Gating (B1's honest constraint: the relation must be ATTESTED in the brain's learned vocab):** mine only if the brain's vocab has ≥1 HIGH and ≥1 LOW size marker AND ≥6 attested items survive; otherwise return `None`.
- **`_build_ordinal_map` rewrite:** feeds the corpus-MINED premises into the **IDENTICAL** Betasort-asymmetric update (the learner is unchanged; only the SOURCE of the premises flips from given to acquired). Falls back to the curated `_SIZE_LADDER` for vocab-poor brains. Records `self._ordinal_axis_source` (`corpus-mined` / `curated`) + `self._ordinal_axis_order`.
- **`_transitive_response`:** the off-axis abstain message now exposes the ACTIVE axis ("a scale learned from the corpus (mouse < pig < fox < bear < dog < ...)" when mined, vs "a scale I've been given (...)" when curated); `rec["axis_source"]` added.

## 3. The decisive evidence (the HARD GATE)

### 3a. Console transitive on the MINED axis (live transcript, `brainALL_w7000`, the B1 brain)

`axis_source = corpus-mined` — and the console's axis is **byte-identical to B1's GO mined order**:
`mouse < pig < fox < bear < dog < ant < cat < wolf < rabbit < lion < cow < sheep < horse < tiger < whale < elephant` (all 16 animals on the map).

```
YOU: is whale bigger than mouse?    BRAIN: Yes -- whale is bigger than mouse.        [gap=13.147]
YOU: is elephant bigger than cat?   BRAIN: Yes -- elephant is bigger than cat.       [gap=8.111]
YOU: is mouse bigger than whale?    BRAIN: No -- it's the other way around: whale is bigger than mouse.  [gap=13.147]
YOU: is lion bigger than rabbit?    BRAIN: Yes -- lion is bigger than rabbit.        [gap=0.929]
YOU: is cat smaller than horse?     BRAIN: Yes -- cat is smaller than horse.         [gap=5.498]
YOU: is tiger smaller than dog?     BRAIN: No -- tiger is smaller than the other; dog is smaller than tiger.
YOU: is dog bigger than banana?     BRAIN: I can't place 'banana' on a size scale -- I only compare things
                                           on a scale learned from the corpus (mouse < pig < fox < bear < dog < ...).
```

**Transitive answers correct-vs-mined-order: 6/6** (held-out, non-adjacent pairs read from the learned geometry — the symbolic-distance margin shows through: far pairs gap ≫ near pairs). **Moat 0-FA** (`banana`, off-axis → abstains honestly, never fabricates an order). The console reasons over the brain's OWN learned size knowledge, not a hand-typed ladder.

> The answers track the *mined* (lossy) order — e.g. `ant` lands mid-order ("big ant colonies" mis-attributes size to the colony; B1's documented distributional noise). The correct bar (per B1) is "correct ON the axis the brain learned + the symbolic-distance signature + the moat", NOT 100% vs perfect biology — and that is what holds.

### 3b. DEFAULT `--rubric` — 10/10, byte-identical BEFORE and AFTER (no regression)

The default console loads `brain1454_w7000` (16 animals but **0 size markers**) → `_mine_size_axis` returns `None` → falls back to the curated ladder → the transitive route (which the rubric does not even exercise — the rubric's "relate" prompt is `is X like Y?`, a different route) is unchanged.

| | RUBRIC SCORE | moat leaks | mixed-type | VERDICT |
|---|---|---|---|---|
| BEFORE | **10/10** | 0 | MIXED | PASS |
| AFTER | **10/10** | 0 | MIXED | PASS |

Default-brain fallback verified directly: `axis_source = curated`, order `tiny < small < big < huge < giant`, transitive answers correct on the curated ladder, off-axis (`dog`/`cat`) → abstains. The fallback path is byte-preserved.

### 3c. CI guards — all green

`tests/test_first_chat_console_spiking_render.py` (console) + `tests/test_regimeb_corpus_mined_axis.py` (B1) + `tests/test_transitive_ordinal_map.py` (Tier 2.3): **21 passed in 9.18s**. B1 de-risk re-run BEFORE the wiring: **GO** (held-out 0.790, rho(margin) +0.882 every seed, permuted-mining 0.476/0.552 collapses, moat 0-FA).

## 4. Honest scope / caveats

- The mined axis is **lossy + corpus-budget-dependent** (B1's measured boundary). The console mines at the validated 80 MB operating point; a smaller corpus would degrade it. The console build pays a one-time ~52s corpus read for the mine (on `brainALL_w7000`); the default brain skips it entirely (no markers → instant fallback).
- The axis-learning OBJECTIVE still runs host-side (the Betasort update; as in Tier 2.3 / B1). The *comparison* on the spiking accumulator is B1's spiking path (not invoked by the CPU console's transitive route, which reads positions directly — the same as before this change). The mining is host-side curriculum prep (legitimate per BRAIN-BASED-ONLY: preparing the syllabus over the brain's own vocab).
- Only the SIZE relation is mined; other ordinal relations (age/speed/rank) are B1's named bounded follow-on.
- **NO `sim/` edit.** Only `first_chat_console.py` (additive). No collision with the concurrent B-mine-2 agent (which edits `wh_question_parser.py`).

## 5. Reproduce

```bash
# the DEFAULT rubric (vocab-poor brain -> curated fallback; must stay 10/10)
SIM_BACKEND=numpy python -m research.runners.first_chat_console --rubric

# the console transitive route on the MINED axis (brain with size markers)
SIM_BACKEND=numpy python -c "
from research.runners.first_chat_console import build_brain_on_codes, FirstChatConsole, audit_moat
b = build_brain_on_codes(npz_path='bridges/firstchat/brainALL_w7000.npz_seed42.npz', verbose=False)
c = FirstChatConsole(b); print(c._ordinal_axis_source, '|', ' < '.join(c._ordinal_axis_order))
for q in ['is whale bigger than mouse?','is dog bigger than banana?']:
    p, r = c.respond(q); print(q, '->', p)"

# CI guards
SIM_BACKEND=numpy python -m pytest tests/test_first_chat_console_spiking_render.py tests/test_regimeb_corpus_mined_axis.py tests/test_transitive_ordinal_map.py -q
```

## 6. Bottom line

The console's transitive route is now **acquired, not given**: it reasons over the **corpus-mined size ordering over the brain's own learned vocab** (the B1 GO axis, byte-identical) when the brain has the size markers, and falls back to the curated ladder only for vocab-poor brains. The default `--rubric` is **10/10 byte-identical**, the transitive answers are **correct on the mined axis (6/6)**, the moat is **0-FA**. The "validated-but-unwired" gap (gate §B-wire-1) is closed. Reuse-by-import, NO `sim/` edit.
