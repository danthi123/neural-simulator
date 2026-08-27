---
type: finding
status: superseded
date: 2026-08-27
mechanism: confidence-caps-forthcomingness
lane: introspection-self-model
integration_faculty: confidence-forthcomingness
artifacts:
  - research/findings/raw/_confidence_forthcoming_prodflip/verify.json
  - research/findings/raw/_confidence_forthcoming_prodflip/verify_part_f_fixed.json
  - research/findings/raw/_confidence_forthcoming_prodflip/soak_summary_6seed.json
runner: research/findings/raw/_confidence_forthcoming_prodflip/verify_confidence_forthcoming_prodflip.py
---

> ⛔ **SUPERSEDED same-day (2026-08-27)** by
> [`2026-08-27-confidence-forthcomingness-chain-trace-fix-still-default-OFF-NOGO.md`](2026-08-27-confidence-forthcomingness-chain-trace-fix-still-default-OFF-NOGO.md).
> This doc's PARTIAL claim survives: checks A-F genuinely pass WITH forced evidence (the mechanism itself
> works). The claim that died: "ON BY DEFAULT" / "genuinely load-bearing" — the owner's hard rule against
> hollow default-on flips applies, because (undisclosed here) the coupling's cap/grant NEVER fires on real,
> unforced production traffic (`mean_role_confidence` was structurally None on every real turn at the time this
> doc was written). The superseding doc fixes that structural bug too, and finds the coupling STILL never
> fires on real traffic for a different reason (confidence saturates at 1.0 on this demo's clean vocabulary).
> `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `confidence-forthcomingness` row now reads `on_by_default: NO`.

# Confidence-caps-forthcomingness (board #94) — ⛔ SUPERSEDED, the default-ON flip attempted here was REVERTED same-day (see banner above)

## Result
`BRAIN_CONFIDENCE_FORTHCOMING` flips from default-OFF to default-ON. `webapp/confidence_forthcoming_chat.py`
gains `_CONFIDENCE_FORTHCOMING_DEFAULT_ON = True`; `confidence_forthcoming_enabled()` now mirrors the
`_bg_select_flag_on` convention (an UNSET env var reads as ON; `BRAIN_CONFIDENCE_FORTHCOMING=0` is the guarded,
explicit escape). The coupling was re-verified through the REAL `/api/brain-chat` handler against the NOW-LIVE
production brain (`brain="tiny-demo"`, which auto-attaches the 15k-fact wikidata cortical LTM since the
2026-08-26 `tiered-knowledge-ltm` default-on flip — `source` reads `"tiny-demo +LTM"`), and a 6-seed
no-regression soak confirms the live chat pipeline does not regress. `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`
gains a new `confidence-forthcomingness` row: `wired: YES`, `on_by_default: YES`, `scaffold_retired: NO`.

## A genuine bug found + fixed en route
`TieredFactStore` (`research/runners/tiered_fact_store.py`) had NO `__setattr__`. Its own docstring says it
"delegates every other attribute ... to the buffer", but that promise only held for READS (`__getattr__`);
`webapp/server.py`'s per-turn activity-trace flip (`_composer.trace = True; _composer.last_trace = None`)
silently created SHADOW instance attributes on the wrapper instead of reaching `self.buffer` — so
`activity`/`metacog.confident` read `None` on EVERY `tiny-demo +LTM` turn since the 2026-08-26 knowledge-core
flip, regardless of the turn's real confidence. Fixed by adding `TieredFactStore.__setattr__`, forwarding every
attribute except `buffer`/`ltm` to the buffer. Verified: `activity` is no longer `None` on a real `tiny-demo
+LTM` turn (check E). Logged to `research/FAILURE_LOG.md` (2026-08-27).

## A correction to the original finding's stated residual
`research/findings/2026-08-27-confidence-drives-forthcomingness-GO.md` hoped that "moving verification to a
richer knowledge base" (the 2026-08-26 LTM flip) would dissolve the `mean_role_confidence`-returns-None
residual. Measured directly against the now-live LTM brain, it does NOT: the root cause is structural and
independent of KB size. `RichAnswerComposer._chain_facts` (`max_chain_hops=3`) always issues one MORE
`composer.query_patient` call past a successful direct match, and `OneBrainComposer.query_patient` resets
`self.last_trace = None` unconditionally at entry (`research/runners/one_brain_composer.py:1473-1495`) — so the
chain's inevitable dead-end hop clobbers the good trace the direct match left behind, on BOTH a buffer-sourced
answer ("what does the brain use", 3 sentences) and a genuine 7-fact LTM subject (frank_lincoln_wright). Both
still read `metacog: null` in this session's own measurement (check E). A SEPARATE next rung; not attempted here
(large blast radius across every other `last_trace` consumer).

Separately, `RichAnswerComposer`'s elaboration (`_chain_facts`/`_facts_about`/`_facts_mentioning`, which read
`self.composer.kb`) can ONLY ever see the small conversational BUFFER tier — `TieredFactStore.kb` delegates
solely to `self.buffer`, never the routed cortical LTM shard — so a genuine LTM-sourced direct answer cannot
itself be elaborated by this coupling today. Both residuals are recorded in the ledger row (`scaffold_retired:
NO`) rather than silently dropped.

## Load-bearing verify — through the REAL `/api/brain-chat` handler, GO 6/6 (checks A-F)
<!--derived--> (rounded reads of the cited artifacts; full precision is in the JSON)

- **(A) elaboration count differs, high vs low, base fact identical.** "what does the brain use" (forced-HIGH
  evidence) -> 2 sentences, `confident=True`, `granted=True`. "what does the dog chase" (forced-LOW evidence) ->
  1 sentence, `confident=False`, capped. Direct facts unchanged (`[brain,use,spikes]` / `[dog,chase,cat]`). Both
  turns ran on the genuinely-attached `"tiny-demo +LTM"` brain. PASS.
- **(B) LESION collapses the difference.** `BRAIN_METACOG_LESION=1` on both turns -> both settle at 1 sentence,
  `confident=False` unconditionally (the HIGH turn's 2 sentences drop to 1). PASS.
- **(C) explicit-OFF byte-identical.** `BRAIN_CONFIDENCE_FORTHCOMING=0` on two independent runs -> no
  `confidence_forthcoming` key, identical answer + sentence count. PASS.
- **(D) moat-safe.** The HIGH-confidence reply's `verified=True`; its 2 `supporting_facts` exactly match its 2
  rendered sentences. PASS.
- **(E) LTM reachability + the fix.** An UNPATCHED query against `frank_lincoln_wright` (a real 7-fact wikidata
  subject) is answered correctly + `verified=True` on the same real brain (`source="tiny-demo +LTM"`), and
  `activity` is no longer `None` (the `TieredFactStore.__setattr__` fix, confirmed). PASS.
- **(F) DEFAULT-ON guard, both arms in one process.** Env UNSET behaves IDENTICALLY to explicit
  `BRAIN_CONFIDENCE_FORTHCOMING=1` (same sentence count, same `confidence_forthcoming` key presence) and
  DIFFERENTLY from explicit `=0` (no key). Guards the `os.environ.pop()`-as-OFF staleness pattern named in the
  task brief. First run false-FAILed on a TEST-HARNESS bug (the floor override wasn't applied to the unset arm,
  producing a floor mismatch, not a real ON/OFF difference); fixed and re-verified in a lean 3-session re-run.
  PASS (`verify_part_f_fixed.json`).

Artifact: research/findings/raw/_confidence_forthcoming_prodflip/verify.json (A-E) +
research/findings/raw/_confidence_forthcoming_prodflip/verify_part_f_fixed.json (F, corrected) · runner
research/findings/raw/_confidence_forthcoming_prodflip/verify_confidence_forthcoming_prodflip.py.

## 6-seed no-regression flip-soak — GO
Mirrors the `_bg_action_selection_flip_soak.py` PART A / PART B split. **PART A** (metacog organ physiology,
seeds 42/43/44/100/101/102): forced-HIGH evidence reads `confident=True`, forced-LOW reads `confident=False`,
BOTH lesion to `confident=False`, on every seed — 6/6. **PART B** (handler, ONE pass, the REAL, non-isolated
production default — every other faculty at its own shipped default): on real/unpatched control turns, explicit
ON and explicit OFF are answer-IDENTICAL (the coupling is a safe no-op on today's real traffic, per the
residual above — `mean_role_confidence` essentially never populates, so `confident` never reads True without
forced evidence); no crash; the no-confab moat holds. Overall: `organ_go=True`, `handler.no_regression=True` ->
`overall_go=True`. Artifact: research/findings/raw/_confidence_forthcoming_prodflip/soak_summary_6seed.json.

The first PART B run also false-FAILed on a TEST-HARNESS bug: it compared the FULL response dict (`off == on`)
rather than the OBSERVABLE fields, so it flagged a "regression" that was only the ADDITIVE
`confidence_forthcoming` diagnostic key being present (by design, whenever the coupling is in-scope, even with
`granted=False`) — the answer text, `recalled_svo`, `n_sentences`, and `verified` were already byte-identical in
that first run's own `pairs_diff`. Fixed to compare only observable fields (plus a separate output-path bug,
`parents[3]`->`parents[4]`) and re-run clean.

## Escape / lesion knobs
```
BRAIN_CONFIDENCE_FORTHCOMING=0            # disable (byte-identical to pre-flip); default is now ON (unset=ON)
BRAIN_CONFIDENCE_FORTHCOMING_FLOOR="1,0"  # optional: force the floor (mirrors the #84 mood-INDUCE pattern)
BRAIN_METACOG_LESION=1                    # reused lesion (collapses confident->False unconditionally)
```
