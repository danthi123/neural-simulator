---
type: finding
status: go
date: 2026-08-10
mechanism: corpus-mined SVO facts stored in the RF-phasor VSA composer, wired into the live one-brain chat via an additive vocab kwarg
lane: H-memory (conversation integration / grounded-knowledge breadth)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/stageA/corpus_facts_live_chat_s42.json
  - research/findings/raw/lanes/stageA/corpus_facts_live_chat_s43.json
  - research/findings/raw/lanes/stageA/corpus_facts_live_chat_s44.json
  - research/findings/raw/lanes/stageA/corpus_facts_live_chat_s100.json
  - research/findings/raw/lanes/stageA/corpus_facts_live_chat_s101.json
  - research/findings/raw/lanes/stageA/corpus_facts_live_chat_s102.json
---

# INTEGRATION #6 — corpus-LEARNED grounded facts wired into the live chat (6-seed GO): the brain says MORE, the moat holds

**2026-08-10.** Target: the live multi-turn chat could only talk grounded-ly about **dog** and **cat** (2
subjects, 6 hand-taught facts), so most turns were silent. INTEGRATION #6 lets the brain say MORE the RIGHT way
(the EMERGENCE BAR): it stores relational facts **mined from the corpus it "heard"** (TinyStories), so the same
retrieval/moat/prose pipeline now covers more subjects — with the no-confab moat holding BY CONSTRUCTION. This is
**not** hand-added phatic handlers; it is learned CONTENT, and the out-of-domain turns still abstain.

## The wire-in (Route #3, corpus-SVO)

The live chat's content chokepoint is `comp.kb` (the RF-phasor VSA fact store) + the composer VOCAB, both set in
`_stageA_full_integration_derisk.build_one_brain`. Facts stored via `comp.store(a,v,p)`; retrieval is
`_gm_retrieve_neighbourhood(comp,topic,actions)` → `comp.query_patient` (the RF-VSA moat, abstain→None). So
expanding the VOCAB + storing more facts propagates automatically through retrieval, the moat, and prose. The ONE
additive edit to the live loop is a `vocab=DEFAULT_VOCAB` kwarg threaded through `build_one_brain` to both
`n_total_for(...)` and `CoResidentOneBrainComposer(vocab=...)`; the default keeps byte-identity (proven below).
Runner: [`research/runners/_corpus_facts_into_live_chat_derisk.py`](../runners/_corpus_facts_into_live_chat_derisk.py).

## Result (6/6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, cfg.seed-controlled)

- **Tier 0 (core claim, no bridge).** Mine top-K clean noun-verb-noun triples from TinyStories (K=40, verb
  normalised past→present), V = DEFAULT_VOCAB ∪ mined ∪ curated (|V|=42), store 6 curated + 40 mined facts (46
  total) via `comp.store`. Every seed: **recall 1.0** on the stored (a,v) cues; **BREADTH rises from 2 (dog,cat)
  to 9** distinct subject-topics with a non-empty grounded neighbourhood; **0 moat false-accepts**.
- **Tier 1 (the live loop).** The expanded conversation = the shipped 14 human turns +
  teacher probes about the newly-learned corpus subjects, driven through the LIVE spiking faculties (SEAM-C affect
  ladder differential + curiosity want + the shared 3-way arbiter). **Grounded-reply count rises 4 → 9** (delta
  +5) vs the matched 6-fact baseline (same turns, same faculties, only the facts+vocab differ), every seed;
  **confabulated = 0** on all turns in BOTH conditions; the OOD turns still abstain/deflect; `_detect_ungrounded`
  reads **0** surface confabulations.
- **Post-hoc moat teeth.** `_gm_posthoc_verify` drops **100%** of unsupported (invented foreign-patient)
  propositions on every seed (unsupported_drop_rate 1.0, 12/12).

<!--derived-->
Aggregate: recall 1.0 (6/6), breadth 9 (6/6), moat false-accepts 0 (6/6, all K), grounded +5 (6/6), confab 0
(6/6), teeth drop-rate 1.0 (6/6). GO = 6/6.

## Anti-cheats (all required, all pass)

1. **Permuted-corpus provenance** — shuffle the token order, re-mine: mined-set overlap **0.00–0.07** across seeds
   (all < 0.5). The knowledge is corpus-ORDER-derived, not a hand list.
2. **Expanded moat battery** — untaught in-vocab cues + the OOD turns → **0 false-accepts** (query_patient → None),
   every seed, every K.
3. **Empty-kb control (the KEY anti-cheat)** — with the EXPANDED vocab but 0 stored corpus facts, breadth stays
   **2** and every new-subject probe abstains (new-subject-answers = **0**). Competence comes from the FACTS, not
   the vocab expansion. Attribution (`tools.lab.attributable_to`): breadth is ~78% attributable to the facts
   (treatment 9 vs empty-kb same-vocab control 2); the grounded-reply rise is ~56% attributable to the facts
   (treatment 9 vs 6-fact baseline 4).
4. **Capacity sweep K=10/20/40** — breadth 5 → 7 → 9; recall 1.0 and 0 moat false-accepts hold at every K (no
   VSA mis-bind onset up to 46 facts / |V|=42, well under the ~320-concept single-bridge bound).
5. **Surface-confab scan** — `_detect_ungrounded` = 0 on every emitted reply, both conditions.
6. **Byte-identity (the additive-param guard)** — `build_one_brain(seed)` vs `build_one_brain(seed,
   vocab=DEFAULT_VOCAB)` is bit-identical on the substrate threshold hash, num_neurons (25531), the composer
   concept codes, AND the full mouth-free transcript (each brain runs build→store→chat before the next, so the
   build's het-seeding resets the RNG the OU read draws from). A larger V grows the rf slice → a genuinely larger,
   different brain; only the default path is guarded identical.

## What is brain-based vs a declared scaffold (per THE LAW + docs/TERMS.md)

**Genuinely brain-based / emergent.** The knowledge is corpus-DERIVED — token order carries it (permuted overlap
~0), the emergence-bar win: the brain talks about what it "heard". Recall and the no-confab moat are RF-VSA brain
reads (`query_patient` = a spiking VSA unbind + cleanup on the co-resident merged bridge). The live faculties
(affect ladder, curiosity, arbiter) drive each turn off `cp_firing_states`.

**Declared scaffolds (named, not hidden).** (a) SVO mining — a host POS/noun-filter (the "linguistic environment"
boundary). (b) `comp.store` — a host VSA write (the composer-as-idealization shortcut). (c) The frame-render
`_gm_fact_to_english` — a host text interface, the SAME status the generator mouth's conditioning has; the
generator MOUTH itself is a GPU/torch scaffold, deliberately OFF here (this is CPU work) — the grounded CONTENT is
the RF-VSA read, which is what the mouth would render.

**Honest instrument note.** The post-hoc SVO re-parse drops 100% of INVENTED propositions (the moat is sound) but
its keep-rate on TRUE frame-rendered props is ~0.83 — the host frame-render has grammatical quirks (e.g. "gos"
for "goes") that the re-parse occasionally cannot recover, so it over-drops a true prop. That is the CONSERVATIVE
(safe) failure direction — it never emits a confabulation — and it is an instrument limit of the host render, not
a moat failure. In the live replies the full grounded neighbourhood is emitted and confab stays 0.

**Honest scope of the silence reduction.** The 8 out-of-domain silences on the shipped 14-turn chat (capital of
France, arithmetic, death, humor, …) are CORRECT abstentions and stay silent — that is the moat as deliverable.
The win is that the brain now has grounded content about MORE subjects, shown by the +5 grounded replies on the
learned-subject probes. We reduce silence the RIGHT way — by learning more, not by fake-answering the OOD turns.

## Burn-down successors (named, per THE LAW)

The residual scaffolds have named neural successors: the stream cortex learning the co-occurrence matrix M in
synapses (`research/findings/raw/_foundational_curriculum_scaling_scoping.md`) replaces the host mine+store; the
teacher-loop plasticity (`2026-08-08-teacher-loop-corrective-acquisition-*`) replaces host-taught storage once its
own learned-moat leak is closed. Both are the path from "the brain talks about what it heard (host-mined)" to
"the brain learned to talk about what it heard (synaptic)".

## Reproduce

```bash
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._corpus_facts_into_live_chat_derisk \
    --seeds 42,43,44,100,101,102 --K 40 \
    --out research/findings/raw/lanes/stageA/corpus_facts_live_chat_6seed.json
```
(The 6-seed sweep here was run one process per seed for parallelism; each per-seed JSON is cited above, byte-identity
computed on seed 42.)
