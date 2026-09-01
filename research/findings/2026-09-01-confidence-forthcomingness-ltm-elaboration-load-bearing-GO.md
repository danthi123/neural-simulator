---
type: finding
status: positive
date: 2026-09-01
lane: introspection-self-model
integration_faculty: confidence-forthcomingness
board: 94
verdict: The content-exhaustion residual named in 2026-08-27-confidence-forthcomingness-retest-PARTIAL.md is CLOSED. With BRAIN_ELABORATE_FROM_LTM_SHARD=1 AND BRAIN_CONFIDENCE_FORTHCOMING=1 together, through the REAL webapp.server.brain_chat handler, on the TRUE un-overridden production floor (max_sentences=4, NO BRAIN_CONFIDENCE_FORTHCOMING_FLOOR override), a confident turn now keeps 5 sentences (the reach's bonus fact GRANTED) and a genuinely-uncertain turn (real, noise-degraded metacog confidence read, not hardcoded) truncates to 4 -- a real, measured 5-vs-4 difference where the pre-fix shape was an identical 2-vs-2 regardless of confidence. 6/6 seeds GO on the TRUE production composer (OneBrainComposer, composer_kind="onebrain", the webapp.server._COMPOSER_KIND_DEFAULT). The lesion (BRAIN_METACOG_LESION=1) collapses the 5-vs-4 difference to 4-vs-4 on all 6 seeds -- the coupling rides the spiking confidence margin, not a host heuristic. Byte-identical-off confirmed on all 6 seeds (both flags off == a chat built with no LTM tier at all, n=2 both; each flag ALONE reproduces either the floor with no coupling, or the OLD hollow 2-vs-2 shape exactly -- proving both flags are required together). Honest scope: demonstrated on a controlled fixture (a small buffer + a routed LTM shard, the SAME construction pattern the already-merged BRAIN_ELABORATE_FROM_LTM_SHARD's own 6-seed-cupy-GO used), not literally the shipped 15k wikidata_core_15k bundle -- that bundle's machine-generated vocabulary (bill_clinten, country_of_citizenship, ...) does not route through the live NL question parser's supported surface shapes, a SEPARATE, newly-surfaced residual named below. The MECHANISM is proven genuinely load-bearing + lesion-attributable + byte-identical-off; the production-default flip is therefore mechanism-ready but still gated on the owner's UX call (not flipped here) and on either (a) closing the NL-parser vocabulary gap against the real shipped KB, or (b) a richer default-floor vocabulary whose natural chains exceed the floor on real out-of-the-box traffic.
mechanism: confidence read (real, unforced) x LTM-shard elaboration reach, both flags together, on the TRUE production floor via the real handler -- 6-seed GO on the production composer
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.json
runner: research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.py
---

# Confidence-forthcomingness IS load-bearing on the true production floor once elaboration reaches the LTM shard -- 6-seed GO, lesion-attributable, byte-identical-off; the flip is mechanism-ready (owner UX call pending)

## The precise question this closes

[`2026-08-27-confidence-forthcomingness-retest-PARTIAL.md`](2026-08-27-confidence-forthcomingness-retest-PARTIAL.md)
left the confidence-forthcomingness flip (board #94) default-OFF with ONE precisely-characterized residual:
`RichAnswerComposer._chain_facts`/`_facts_about`/`_facts_mentioning` read only `TieredFactStore`'s conversational
BUFFER tier, never the routed cortical LTM shard -- so on the TRUE un-overridden production floor
(`NEUTRAL_SENTENCES=4`), a confident and an uncertain turn both exhausted the same buffer content identically
(3 sentences either way, `reason: nothing_to_cap`) -- a hollow flip.

That elaboration-reaches-LTM fix (`BRAIN_ELABORATE_FROM_LTM_SHARD`, additive, default-OFF) was **already built and
merged to `main`** before this session started (`1b64d563d`/`719072135`, numpy-3-seed-GO), then cross-backend
confirmed cupy-6-seed-GO the next day
([`2026-08-28-ltm-shard-elaboration-cupy-6seed-GO-unblocks-confidence-forthcomingness.md`](2026-08-28-ltm-shard-elaboration-cupy-6seed-GO-unblocks-confidence-forthcomingness.md)),
which named the precise next rung: *"re-test confidence->forthcomingness with `BRAIN_ELABORATE_FROM_LTM_SHARD=1`
on the true floor, through the real `/api/brain-chat` handler; if high-vs-low confidence now yields a genuinely
different number of grounded sentences on the production floor (not just under the test-override), it is
flip-viable."* A same-day retest
([`2026-08-28-confidence-forthcomingness-94-retest-COMPLETE...`](2026-08-28-confidence-forthcomingness-94-retest-COMPLETE-read-discriminates-flip-viable-on-rich-content.md))
confirmed the confidence READ genuinely discriminates through the real handler, but did **not** combine that with
the LTM-shard flag or measure the reach-cap's sentence-count effect -- that combined measurement is what this
finding does.

## What was built

Nothing new in `sim/` or the production coupling code -- the elaboration-reaches-LTM mechanism
(`research/runners/rich_answer_composer.py` `_ltm_facts_about`/`_with_ltm`/`_facts_about`/`_facts_mentioning`,
`research/runners/tiered_fact_store.py TieredFactStore`) and the confidence-caps-forthcomingness coupling
(`webapp/confidence_forthcoming_chat.py`, `webapp/server.py`'s reach-then-cap block) were both already shipped,
additive, default-OFF. What was missing was the **combined, real-handler, TRUE-floor, real-confidence, 6-seed
measurement** that neither prior finding ran. That is
`research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.py`.

## Fixture (why not the shipped 15k KB)

The shipped `knowledge_bundles/wikidata_core_15k` core's machine-generated entity/relation vocabulary
(`bill_clinten`, `country_of_citizenship`, `member_of_political_party`, ...) was tried against every supported
live NL question-routing shape (`_extract_route`'s generic "what does X V" positional parse,
`_relation_fronted_route`'s "what \<relation\> is X" single-word-relation shape,
`_definitional_copula_route`'s "what/who is X") through the real handler -- **every natural phrasing abstained**
(checked empirically on the real `tiny-demo +LTM` production brain before building this fixture). This is a
SEPARATE, newly-surfaced residual (the NL comprehension layer's surface-form coverage against this specific
bundle's vocabulary), not a defect in the elaboration or confidence mechanisms themselves, and is not attempted
to be closed here.

So this finding reuses the SAME construction pattern the already-merged, already-6-seed-cupy-GO
`research/findings/raw/_ltm_shard_elab/verify_ltm_shard_elab.py` used -- a small conversational BUFFER + a
`ShardedPhasorStore` LTM behind the SAME production `TieredFactStore` -- sized so a real 2-hop chain plus
LTM-fed elaboration genuinely EXCEEDS the floor:

- BUFFER: `(brain, use, spikes)`, `(spikes, carry, information)` -- 2 facts, a real chain.
- LTM (routed shard): `(spikes, travel, axon)`, `(spikes, generate, current)`, `(spikes, trigger, synapse)`,
  `(spikes, require, threshold)` -- 4 facts, agent-role under `spikes` (the chain's end concept and the
  elaboration topic).
- `Q = "what does the brain use"` -> direct `(brain, use, spikes)`; chain extends one hop via the buffer's own
  `(spikes, carry, information)`; topic becomes `spikes`.

**Composer kind: `onebrain`**, not the `rf` toy composer the isolated LTM-shard verify used for speed -- checked
empirically first (this session's own scratchpad probes, not committed): the `rf` composer's cleanup "margin"
field is a RAW cosine DIFFERENCE (`top_raw - runner_raw`), structurally capped around ~0.35-0.48 for this fixture
even at zero added noise and D up to 512 -- it never crosses `ROLE_CONF_HI=0.50`, because that band was
calibrated (issue #181) against `OneBrainComposer`'s RATIO margin `(peak-runner_up)/peak`, a different scale that
centers near 0.6 on a genuine clean recall. A `confident=True` arm is therefore structurally unreachable on `rf`
regardless of fixture design -- not a fixture bug, a composer-margin-scale mismatch. `onebrain` is also simply
the correct choice: it is `webapp/server.py`'s actual `_COMPOSER_KIND_DEFAULT`, the TRUE production substrate.

## The measurement, through `webapp.server.brain_chat` in-process, 6 seeds

Per seed [42, 43, 44, 100, 101, 102], with `max_sentences=4`/`max_elaborations=2` NEVER overridden (the true
floor) and `BRAIN_CONFIDENCE_FORTHCOMING_FLOOR` unset:

| condition | flags | n_sentences | reason |
|---|---|---|---|
| no LTM tier at all | -- | 2 | (reference) |
| both flags OFF | -- | 2 | identical to the no-LTM-tier reference (byte-identical) |
| `BRAIN_ELABORATE_FROM_LTM_SHARD=1` alone | elaboration only | 4 | floor, no `confidence_forthcoming` key |
| `BRAIN_CONFIDENCE_FORTHCOMING=1` alone | confidence only | 2 | reproduces the OLD hollow shape exactly |
| **both ON, CLEAN (real confidence)** | both | **5** | `high_confidence`, reach GRANTED |
| **both ON, UNCERTAIN (real, noise-degraded confidence)** | both | **4** | `low_confidence_capped`, reach TRUNCATED |
| both ON, LESIONED (`BRAIN_METACOG_LESION=1`), clean | both + lesion | 4 | `confident=False` unconditionally |
| both ON, LESIONED, uncertain | both + lesion | 4 | `confident=False` unconditionally |

The UNCERTAIN arm's confidence is a REAL read off the co-resident metacog organ (`_metacog_qualify` in
`webapp/server.py`, unmodified), not a hardcoded boolean: the SAME established synaptic-noise degradation model
(`research.runners._emergent_graceful_degradation_derisk._noise`, the identical model
2026-08-27/2026-08-28's own confidence-discrimination findings used) perturbs `comp.buffer.store_conns`, and a
per-seed bounded sigma-scan (0.3/0.6/0.9/1.2/1.5/2.0, matching `_confidence_read_discrimination_derisk.py`'s own
bounded-scan discipline -- no runaway sweep) finds the FIRST sigma that still answers (not an abstain), still
recovers the SAME direct fact `["brain","use","spikes"]` (not a misrecall -- the moat), and reads
`confident=False`. All 6 seeds found a crossing sigma in [0.9, 1.5] (`per_seed[i].sigma_used` in the artifact:
42->1.5, 43->0.9, 44->0.9, 100->1.2, 101->0.9, 102->1.2), dropping the read below `ROLE_CONF_HI=0.50`. (A
companion scratchpad probe, not committed, measured the CLEAN read at seed 42 in the same ballpark as the
shipped tiny-demo's own clean read reported by the 2026-08-27 retest; the artifact itself records only the
boolean `confident` per turn, not the raw scalar, so that comparison is not re-asserted as a cited number here.)

Per-check, all 6 seeds PASS:
- **byte_identical_off** -- both flags off reproduces the no-LTM-tier reference exactly (n=2, no
  `confidence_forthcoming` key), and each flag alone reproduces either "no coupling, always floor" (elaboration
  alone) or "the exact pre-fix hollow shape" (confidence alone) -- proving BOTH flags are required together, not
  either in isolation.
- **load_bearing** -- clean (5, `high_confidence`) vs uncertain (4, `low_confidence_capped`) genuinely differ,
  both moat-clean.
- **lesion_reverts** -- `BRAIN_METACOG_LESION=1` collapses BOTH arms to `confident=False` / 4 sentences -- the
  5-vs-4 difference VANISHES under the metacog organ's own lesion, on every seed.
- **moat** -- every gathered fact in every condition is one of the known buffer/LTM facts (no confabulation).

Artifact: `research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.json` (`GO`, all 4
checks PASS on all 6 seeds; `tools.verdict.Verdict`-earned, `preconditions` block carried in the artifact).
Runner: `research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.py` (numpy backend,
CPU; 2130.5s / ~35.5 min for the full 6-seed sweep -- dominated by the `onebrain` composer's per-build bridge
construction (~90s x 2 builds/seed) and the dlPFC spreading-activation planner's per-turn bridge rebuild
(~15-20s/turn, ~13 turns/seed); speed is secondary per the project's non-negotiables, no attempt made to
optimize this). The full run log (SIM_BRIDGE build noise, gitignored) is not committed; the JSON artifact above
carries every number this finding cites.

## Decision: mechanism-ready, flip NOT changed here

`_CONFIDENCE_FORTHCOMING_DEFAULT_ON` in `webapp/confidence_forthcoming_chat.py` stays `False`, and
`_elaborate_from_ltm_enabled`'s underlying flag stays default-OFF -- **the owner's UX call, not made here.** What
changes: the specific "hollow flip" reason both 2026-08-27 findings gave for staying off (content-exhaustion; the
reach-cap never has anything to trim on the true floor) is now **directly refuted** by a 6-seed, real-handler,
real-confidence, lesion-attributable measurement. The coupling is proven genuinely load-bearing, not merely
mechanically-sound-when-forced.

**What is NOT yet shown**: this exact demonstration on the literal shipped out-of-the-box `tiny-demo +LTM` brain
real traffic (the NL-parser vocabulary gap against `wikidata_core_15k`'s machine-generated relation names blocks
that specific measurement -- a separate residual from the one this finding closes). Two honest paths to the final
non-hollow real-traffic GO, neither attempted here: (a) close the NL-parser surface-form gap against the shipped
bundle's vocabulary (e.g. extend `_relation_fronted_route`/`_extract_route` to handle underscored multi-word
relation names), or (b) demonstrate the same 5-vs-4 pattern on a richer/more natural-language-friendly default-
floor vocabulary whose chains genuinely exceed `NEUTRAL_SENTENCES=4`. Until one of those lands, the flip is
**mechanism-ready but not yet real-out-of-the-box-traffic-ready**.

## Ledger

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `confidence-forthcomingness` row is updated: `scaffold_retired`'s
residual-1 (buffer-tier-only elaboration) is marked CLOSED with this finding as evidence; `on_by_default` stays
`NO`, its note updated to cite this finding and name the NL-parser vocabulary gap as the new, separate condition
for a real-out-of-the-box-traffic demonstration. `de_risked` stays `YES` (unchanged -- already true before this
finding). No `default_on_spiking_faculties` count change (reuses the already-counted E1 metacog organ + the
already-merged LTM-shard elaboration; no new spiking pool).
