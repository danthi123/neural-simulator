# Strategic plan — the faithfulness tradeoff, the data lever, and the path to a brain that learns, grows, and communicates

*2026-09-18. Written on the owner's three decisions (faithfulness relaxation approved honesty-conditioned;
scaling gated on deep research; continuous-learning agreed) and the adversarially-verified scaling go/no-go
(finding 2026-09-18-scaling-go-no-go-...). This is the forward plan; the live board is
[GAP_CLOSURE_MISSION.md](../../GAP_CLOSURE_MISSION.md), the skim view is [ROADMAP.md](../../ROADMAP.md).*

## 0. The one-paragraph thesis

The science is largely de-risked; the remaining distance to the goal (a brain that learns, grows, and
communicates fluently + broadly) is **scale + integration + one deliberate tradeoff**, not mechanism discovery.
The single load-bearing wall is the brain's own voice (neural-render), which gates ~half the remaining
scaffold retirements. The go/no-go research settled the biggest question: **raw compute scale is NOT the lever
for that wall — data (throughput + curation) atop a modest capacity floor is** — so we do not buy hardware on
faith; we run a decisive zero-purchase test first, and we spend the newly-approved faithfulness slack where it
actually buys throughput, honestly.

## 1. The faithfulness tradeoff — the per-mechanism rule (owner-approved, honesty-conditioned)

Faithfulness does two different jobs; only one is the bet.
- **KEEP as invariants (they ARE the consciousness/AGI bet):** brain-based cognition (neurons/synapses between
  sensation and action), ONE shared spiking substrate, emergent-structure-from-experience, and the honesty
  boundary (functional read-outs only; never assert phenomenal experience).
- **RELAX per-mechanism (bit-exact biophysical realism) ONLY when all three hold:** (a) it buys a *significant*
  performance gain, (b) it loses *nothing* important to the end-goal capabilities, (c) the trade is *documented*
  (what was approximated, why, and the measured before/after). Biology-strong stays the default and the
  differentiator; a relaxation is a logged exception, never a silent drift.
- **The test to apply at each candidate:** "does the emergent FUNCTION change?" If a tractable approximation
  yields the same emergent behavior at materially less compute, relax it and log it. If it changes the function
  the bet depends on, keep the faithful version.

## 2. The scaling verdict — what it changes (do NOT buy hardware yet)

The adversarially-verified review of our own record + the external laws found:
- **Raw parameter/GPU scale is falsified as the fluency lever** on the broad domain (our token-supply sweep
  capacity-saturated; the one realistic large-token run regressed cross-domain). The owner's skepticism is
  vindicated by our own saturated data.
- **The lever is DATA** — fluency is a data-quality problem, breadth is a token-throughput + curation problem
  atop a modest one-time capacity floor (TinyStories / Qwen-0.5B / SmolLM / Chinchilla / the ~2-bits-per-param
  knowledge floor all agree).
- **The free analytical gate was run and is inconclusive** (the irreducible-loss estimate straddles the fluency
  band because our largest model was never trained on enough data to pin the curve). It also proved the planned
  cheap follow-up would not settle it.
- **The decisive, still-zero-purchase test:** extend the largest model (d384) to a token budget comparable to the
  smaller models (~370M tokens, up from ~43M) on the EXISTING 3090, then confirm the winning point survives in
  the DEPLOYABLE biological form (local-rule readout, ≥16k vocab) within a single consumer GPU's envelope.
- **Honest fork (recorded):** if the deployable/biological voice cannot reach fluency within the 3090 envelope,
  keeping Qwen as the conditioned articulation scaffold is the honest call — a bigger GPU would buy a fluent
  BPTT model that cannot deploy as the one-brain spiking mouth. Two risks a GPU never touches: biologization
  transfer (all pro-scale results are BPTT at small vocab; the deployed readout was never re-run at scale) and
  the lexical cap (small vocab is unk-soup; real fluency needs ~30-40k vocab, which multiplies params against
  the consumer-hardware envelope).

**Implication for the plan:** the mouth sub-project's lever is a **data pipeline (curation + throughput)**, not a
hardware buy. Compute is likely necessary but demonstrably not sufficient. Hardware is gated on the decisive test
above passing AND the biological-deployability check.

## 3. The critical path (in priority order)

1. **Land the one-brain culmination.** The 11-organ default-on flip is root-caused + fixed (heterogeneity
   seed-trap + homeostasis alignment); re-run the 38-faculty battery → land on all_pass. Biggest single
   scaffold-retirement + the one-brain default.
2. **Resolve the scaling question with the decisive local test** (extended-d384 + biological-readout confirm) —
   BEFORE any hardware decision. This is the gate for everything mouth-shaped.
3. **If the test says data-scalable: build the data pipeline** (curated, staged curriculum — quality first, then
   throughput), on the existing card + modest cloud bursts for the throughput runs, spending the faithfulness
   slack (§1) where it buys throughput without changing the emergent function. If the test says otherwise: keep
   Qwen as the declared scaffold and redirect to the deployable-biological-fluency mechanism question.
4. **Make continuous-learning the default operating mode** (owner-agreed): extend the per-turn plasticity +
   consolidation/replay so the brain learns *through* conversation by default, not as a separate arc. "Learns +
   grows" and "fluent + broad" become one substrate learning from experience.
5. **Keep per-faculty de-risking in the BACKGROUND** (the mini-PC pool, 0 tokens) — it is cheap there and should
   not be the main thread; the main thread is integration + the mouth data lever + go-continuous.

## 4. What we are explicitly NOT doing

- Not buying compute on faith (gated on §2's decisive test).
- Not relaxing the invariants (§1) — brain-based / one-brain / emergent / honest stay fixed.
- Not retiring Qwen prematurely — only when the deployable spiking mouth is genuinely fluent.
- Not treating the non-scale walls (deep-credit, affect noise-robustness→multimodal, the flip calibration) as
  scale problems — a GPU does not touch them.

## 5. Immediate next steps (post-game, when local compute resumes)

- Resume the GPU queue; re-run the onebrain flip battery → land the flip on all_pass.
- Run the decisive extended-d384 token test (existing 3090) + the biological-readout-at-deployable-vocab confirm.
- Stand up the data-curation pipeline design (quality-first curriculum) in parallel on the pool. **NOTE
  (2026-09-18): this is a GAP-ANALYSIS on EXISTING infra, not greenfield — the record already has
  `corpus_stream.py`, `_corpus_develop_curriculum.py`, `build_distill_corpus.py`, `_knowledge_core_curate.py`,
  a `_curriculum_*` family, and a `data/corpus/.tokcache/` disk-backed pre-tokenized `.npz` mechanism (up to
  681M). Design = what quality-first staged curation is MISSING atop these, not a rebuild. Best sequenced AFTER
  the memory-efficient loader lands (know the loading mechanism) + the decisive test result is in hand (it sizes
  the data need) — designing it before both risks re-derivation + designing on incomplete information.**
- Harvest the queued semantic-recall prodscale 6-seed + the #203 freeze-tests.
- On the scaling test's result, finalize the hardware decision and update this plan.
