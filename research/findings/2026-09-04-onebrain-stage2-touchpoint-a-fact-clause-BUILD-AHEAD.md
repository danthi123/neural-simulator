---
type: finding
status: wired
date: 2026-09-04
mechanism: BUILD-AHEAD -- a new, default-OFF, scope-limited flag `_touchpoint_a_fact_clause_enabled()` /
  `BRAIN_TOUCHPOINT_A_FACT_CLAUSE` in `RichAnswerComposer._render_one_verified`
  (research/runners/rich_answer_composer.py:859-900) that, on a Touchpoint-A miss (the bounded transitive-SVO
  spiking Broca mouth, `chat.spiking_recall_surface`, did not cover this gathered SVO), tries ONE more
  brain-based render -- `webapp.wkv_mouth_generator.render_fact_sentence` (the already-6-seed-GO
  `SpikingClauseProducer` driven by the closed-class `RELATION_LEXICON`, the SAME mechanism Surface B's own
  fact-clause fallback already reuses) -- BEFORE falling through to `chat.renderer.render_svo` (Qwen on a CUDA
  host / the template-stub under `SIM_BACKEND=numpy`). A miss (relation not lexicon-covered, or an exception)
  degrades straight to the pre-existing renderer, unchanged. Paired with a measure+retire de-risk runner
  (`research/runners/_touchpoint_a_fact_clause_derisk.py`) that instruments both the pre-existing Surface-A
  counters and this new path, and computes a GO-gate over three STRUCTURAL invariants (scope untouched on
  non-recall turns, gathered-fact content preserved, the flag genuinely inert when OFF).
lane: language (own-voice mouth / retire the Qwen scaffold) + one-brain (substrate consolidation)
seeds: [42]
seed-waiver: this task's own instruction was explicitly PREP-ONLY -- build the flag + runner + GO-gate +
  anti-cheats and run a TINY smoke, and defer the full de-risk to when compute frees. The smoke itself reuses
  the SAME real-traffic pattern (`webapp.server.brain_chat`, in-process, through the real `/api/brain-chat`
  entry point) the project's own precedent instruments use, at `seed=42` -- the fixed internal generation seed
  every real turn already uses and the SAME seed-waiver precedent
  `research/findings/2026-09-04-recall-gate-reaches-real-ltm-GO.md` and
  `research/findings/2026-09-04-onebrain-stage1-qwen-fallback-retire-GO.md` already established for this
  instrument family. The fast regression suite (`tests/test_touchpoint_a_fact_clause_flag.py`, 20 tests) is
  deterministic mocked-fixture unit coverage, independent of any seed. NEITHER claims the mechanism's efficacy
  at scale -- that is the deferred full battery's job, at the precedent's own `--n-known 4`.
instrument: research/runners/_touchpoint_a_fact_clause_derisk.py --smoke (NEW, this session) plus
  tests/test_touchpoint_a_fact_clause_flag.py (NEW, this session, 20 tests, mocked fixtures, no
  GPU/heavy-model/real-brain dependency). Both CPU-forced (`CUDA_VISIBLE_DEVICES=""`, `SIM_BACKEND=numpy`).
runner: research.runners._touchpoint_a_fact_clause_derisk ; tests/test_touchpoint_a_fact_clause_flag.py
external: REUSES (not a fresh search) the FrugalGPT cascade precedent already recorded THIS SAME DAY for this
  EXACT lane (Chen, Zaharia & Zou 2023, arXiv:2305.05176, https://arxiv.org/abs/2305.05176 <!--derived--> -- see
  `research/queue/.external_searches.jsonl`, ts 2026-09-05T00:03:52Z, the Stage-1 sibling finding's own
  recording) -- directly applicable here too: this mechanism ALSO routes away from an expensive stage (the
  off-bridge Qwen / host template) once a cheaper, already-computed-equivalent path (the already-6-seed-GO
  fact-clause render) is available, the same cost-cascade logic in the same direction Stage-1 applied. Clears
  `gates/deep_research_at_wall` for this finding's lane via that still-fresh (same-day, same-lane) record.
verdict: WIRED (reachable from the real `/api/brain-chat` endpoint on a request with the flag set -- satisfies
  docs/TERMS.md's `wired` condition), default-OFF, NOT flipped, NOT the full de-risk. This session's own
  --smoke run (2 probes: one known-topic recall, one unknown-topic scope guard) is COMPLETE and its own
  structural GO-gate PASSED (`research/findings/raw/_touchpoint_a_fact_clause_smoke.json`,
  `go_gate.structural_checks_passed=true`): content_preserved (the gathered `supporting_facts` triples are
  identical off vs on), scope_untouched (the unknown-topic row's answer text and fact-clause call count are
  identical off vs on), flag_off_inert (zero fact-clause calls anywhere in the flag-off pass). On this smoke's
  one known-topic probe the mechanism also engaged fully (`fact_clause_engaged_on_known_rows=1`,
  `touchpoint_a_render_calls_delta=-3` -- all 3 Qwen/template render calls the flag-off pass paid were replaced
  by 3 fact-clause hits) -- a promising signal, but ONE topic is not the project's own N=4 battery and is
  reported here as smoke-scale, not a landed capability claim. See Honest residuals for a real, disclosed
  fluency gap this smoke's own answer text surfaces. The 20/20 mocked regression suite passes independent of
  any live brain build.
artifacts:
  - research/findings/raw/_touchpoint_a_fact_clause_smoke.json (this session's own --smoke run; complete: true,
    2 probes x 2 passes, peak RSS well under budget, wall ~6.5 min dominated by two cold tiny-demo brain
    builds)
---

# One-brain Stage-2 build-ahead: wire + smoke-test a fact-clause retirement of "Touchpoint A"

**One-brain Stage 2** (`research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md` SS4: "retire
'Touchpoint A': the Surface-A open-prose recall fallback"), the follow-on to Stage 1
(`research/findings/2026-09-04-onebrain-stage1-qwen-fallback-retire-GO.md`). This task was scoped explicitly as
**PREP, not a landed result**: build the flag, the measure+retire runner, the GO-gate, and the anti-cheats, run
only a tiny smoke to confirm the wiring genuinely works end-to-end, and defer the full de-risk battery to when
compute frees. Everything below reports exactly that scope -- no more.

## 1. The residual this build-ahead targets (re-read, not re-measured, from an already-committed artifact)

`research/findings/raw/_per_touchpoint_qwen_share_shipped_default_AFTER_recall_gate_fix.json` (landed by
`research/findings/2026-09-04-recall-gate-reaches-real-ltm-GO.md`, commit `bbff50765`) already measured, on
Surface A (`BRAIN_OPEN_ENDED` unset, the production default): 5 of 6 known-topic rows (`known_factual` x4,
`known_followup` x1; only `known_multi_sentence`'s disclosed lead-in coverage gap still abstains) now return a
genuine grounded multi-fact answer with `spiking_hit_count=0` and `render_calls>0` on EVERY one. The bounded
transitive-SVO spiking Broca mouth (`chat.spiking_recall_surface`) never covers this newly-reachable LTM
content, so `RichAnswerComposer._render_one_verified` falls through to `chat.renderer.render_svo` -- Touchpoint
A, exactly as the roadmap named it -- for 100% of it. Touchpoint A now carries the FULL weight of the
recall-gate fix's own headline win.

A second prior finding, surfaced by this session's own `tools/before_you_build.sh` corpus check (see
Provenance), sharpens WHY this residual is worth closing, not just cheap to close:
`research/findings/2026-09-02-open-ended-qwen-routed-fact-clause-fallback.md` measured Qwen supplying
"confident, specific, WRONG parametric detail" on real traffic -- calling `castleford_f_c` "a professional
football club" when the store's only sport fact is `rugby_league`. Retiring Touchpoint A onto a render that can
ONLY ever assert the fact's own subject/object NP plus fixed closed-class predicate words (moat-safe by
construction, per `render_fact_sentence`'s own docstring) removes a class of error the current renderer does
not just risk paying compute for -- it can actively supplement with wrong specifics the brain never asserted.

## 2. The candidate fix: reuse Surface B's already-6-seed-GO fact-clause render, do not build a new one

That same 2026-09-02 finding already proved the mechanism in a DIFFERENT call site: on Surface B, a known-topic
hit tries `render_fact_sentence(facts, seed=seed)` directly, and "if it returns a clause, THAT becomes `raw`
and Qwen is bypassed entirely for the turn" (`generator` traces `"spiking_clause"`). The recall-gate finding's
own SS5 independently measured `RELATION_LEXICON` (the closed-class lexicon driving that render) covering
34/34 live relation types in the SAME sampled `wikidata_core_15k` store this build-ahead targets. So the
coverage this fix would need was already measured to exist -- what had never been built or measured is wiring
the SAME mechanism into Touchpoint A's OWN miss path on Surface A, and checking it there specifically. That is
what this session built.

## 3. The wiring (`BRAIN_TOUCHPOINT_A_FACT_CLAUSE`, default OFF)

Inside `_render_one_verified`, between the existing `chat.spiking_recall_surface` miss and the existing
`chat.renderer.render_svo` fallback:

```python
if _touchpoint_a_fact_clause_enabled():
    try:
        from webapp.wkv_mouth_generator import render_fact_sentence
        fc_seed = int(getattr(self.chat.inner, "seed", 42))
        fc_surface = render_fact_sentence([svo], seed=fc_seed)
    except Exception:
        fc_surface = None
    if fc_surface:
        return fc_surface, True
```

Three properties, by construction, not merely intended:

- **Cannot substitute a different fact.** `render_fact_sentence` is called with a ONE-item list, `[svo]` --
  never the composer's broader gathered set -- so its own `pick_covered_fact` can only ever return THIS svo or
  `None`. It cannot pick a different fact than the one already gate-approved for this sentence.
- **Trusted the same way the pre-existing spiking-mouth hit already is.** A non-empty return is accepted
  directly (`verified=True`), without an extra `_verify_rendered` call -- the SAME trust model
  `chat.spiking_recall_surface`'s own hit branch already uses, because `render_fact_sentence`'s own docstring
  states it is "moat-safe by construction" (every token is either the fact's own subject/object NP or a fixed
  closed-class predicate/determiner word).
- **A miss or an exception degrades to the pre-existing path, unchanged.** `None` (uncovered relation) or any
  raised exception falls straight through to `chat.renderer.render_svo`, byte-for-byte the pre-flag behaviour.

**Scope, disclosed, not hidden.** Only the sequential render path (`_render_one_verified`) is wired; the
batched sibling (`_render_paragraph_batched`) is untouched (production runs `BRAIN_RICH_BATCH_RENDER=0` by
default, the sequential path, per the precedent measurement's own disclosed scope note) -- a named residual for
whoever extends this, not a gap this doc hides.

## 4. The measure+retire de-risk runner + its GO-gate

`research/runners/_touchpoint_a_fact_clause_derisk.py` (new) reuses `_sample_known_topics`/`build_probes` from
the precedent instrument (`_per_touchpoint_qwen_share_measure.py`) by import, runs the SAME probe set twice
through two independent fresh sessions (flag OFF, then flag ON, the second pass's session built only after the
first is evicted from the session cache to bound peak RSS), and instruments both the pre-existing Surface-A
counters (spiking hit/miss, renderer calls) and a NEW counter on `webapp.wkv_mouth_generator.render_fact_sentence`
itself.

`compute_go_gate(rows_off, rows_on)` distinguishes STRUCTURAL invariants (must hold on every run, smoke or
full) from the READINESS signal (only meaningful at full-battery scale):

- **content_preserved** -- every known-topic row's `supporting_facts` (the gate-sourced (agent, action,
  patient) triples `webapp/server.py`'s `brain_chat` already surfaces in its JSON response) must be IDENTICAL
  flag-off vs flag-on. The wording engine may differ; the grounded content never may.
- **scope_untouched** -- unknown/dangerous/open-ended/greeting rows must be answer-text identical flag-off vs
  flag-on, and the fact-clause path must fire zero times on them.
- **flag_off_inert** -- the fact-clause render must be called zero times across every flag-OFF row (the flag
  genuinely gates the call, not merely its visible effect).
- **touchpoint_a_render_calls_delta** / **fact_clause_engaged_on_known_rows** -- the actual retirement signal
  (informational on a smoke; the real answer once the full N=4 battery runs).

`main()` exits 1 if the structural checks fail, so a queue dispatcher can detect a genuine regression, not just
completion.

## 5. This session's own smoke: complete, structural gate passed, a real (small) signal, one honest gap

`--smoke` runs exactly two probes -- one known-topic recall (`angora_turkey`, deliberately NOT a slice of the
full battery, which would give 3 known-topic probes and never exercise the scope guard in 2 items) and one
unknown-topic scope guard (`zorplaxian`) -- through the real tiny-demo brain twice. Full numbers in
`research/findings/raw/_touchpoint_a_fact_clause_smoke.json`:

| | flag OFF | flag ON |
|---|---|---|
| known_factual (`angora_turkey`): render_calls / fact_clause_calls / fact_clause_hits | 3 / 0 / 0 | 0 / 3 / 3 |
| known_factual: `supporting_facts` | `[[angora_turkey, located_in_time_zone, kaliningrad_time], [angora_turkey, instance_of, city_work], [angora_turkey, country, the_republic_of_turkey]]` | IDENTICAL |
| unknown (`zorplaxian`): spiking_hit / render_calls / fact_clause_calls | 1 / 0 / 0 | 1 / 0 / 0 |
| unknown: answer text | "Sure -- the tell abouts the zorplaxian -- worth going further here." | IDENTICAL |

`go_gate`: `structural_checks_passed=true`, `content_preserved=true`, `scope_untouched=true`,
`flag_off_inert=true`, `touchpoint_a_render_calls_delta=-3`, `fact_clause_engaged_on_known_rows=1`. Every
structural invariant this session set out to prove held, on the one topic this smoke exercised.

**An honest, disclosed fluency gap, visible directly in the smoke's own answer text.** The flag-OFF answer
reads "...kaliningrad_time. The angora_turkey instance_ofs city_work. The angora_turkey countrys
the_republic_of_turkey. -- worth..." (the template-stub renderer's own per-sentence trailing period). The
flag-ON answer reads "...Kaliningrad Time the Angora Turkey is a City Work the Angora Turkey is located in the
The Republic of Turkey -- worth..." -- the SAME three facts, content-preserved and moat-safe, but
`render_fact_sentence`'s own clause does not carry a trailing period the way the template-stub's `render_svo`
does, so consecutive fact-clause sentences run together without punctuation when `render_paragraph`'s
`" ".join(sentences)` concatenates them; a `slug_to_np` capitalization quirk also doubles a determiner ("the
The Republic of Turkey"). Both are real, nameable, fixable residuals for whoever runs the full de-risk -- NOT a
content or moat problem (the gate above already proves the facts stay correct and unchanged), but a genuine
readability regression this build-ahead surfaces rather than hides.

## 6. Honest residuals

- **One topic, one seed, a 2-probe smoke -- not the full battery.** `fact_clause_engaged_on_known_rows=1` of 1
  known-topic row in this smoke is not evidence the mechanism covers the other 3 known topics the precedent's
  own `--n-known 4` battery samples, still less the store at large. The deferred full run
  (`--n-known 4`, matching every precedent measurement in this family) is the actual readiness test.
- **The punctuation/capitalization gap above (SS5) is real and un-fixed.** The full de-risk's GO-gate should
  treat a fluency regression (sentences running together, a doubled determiner) as a finding to act on, not
  wave past because content_preserved passed -- content correctness and prose fluency are different properties
  and this mechanism has only been shown to hold the first.
- **Only the sequential render path is wired** (SS3) -- the batched `_render_paragraph_batched` sibling is an
  explicit, disclosed gap, not silently unhandled.
- **`known_multi_sentence`'s lead-in coverage gap is unrelated and untouched** -- that probe still abstains
  before this flag's branch is ever reached (unchanged from the recall-gate finding's own disclosed residual).
- **`BRAIN_RICH` (Surface A itself) is the production default already; this flag does not change that.** What
  it changes, when eventually flipped, is WHO words the recall content on that already-default surface.
- **The ledger (`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`) is untouched by this task, on purpose** -- no
  faculty's `retire_status` moves off `BLOCKED:neural-render` until the full battery lands a genuine GO; moving
  it now would be exactly the "closed capability whose default still runs the host shortcut" misuse
  `docs/TERMS.md` itself warns against.

## Provenance

Shipped code written this session (2026-09-04): `research/runners/rich_answer_composer.py`
(`_touchpoint_a_fact_clause_enabled` + its flag-block comment, inserted after
`_direct_ltm_topic_fallback_enabled`; the new branch inside `_render_one_verified`, both new, no other line
changed), `research/runners/_touchpoint_a_fact_clause_derisk.py` (new, ~230 lines),
`tests/test_touchpoint_a_fact_clause_flag.py` (new, 20 tests). Read, not modified:
`webapp/wkv_mouth_generator.py` (`render_fact_sentence`/`pick_covered_fact` :650-698),
`research/runners/spiking_mouth_recall_prod.py` (the `getattr(self.inner, "seed", 42)` seed convention this
task's own new call mirrors), `webapp/server.py` (`brain_chat`'s `"supporting_facts": facts` response field,
confirmed as the exact hook this de-risk's content-preservation check needed with no new instrumentation).

Corpus check run before the first lever (`bash tools/before_you_build.sh`, logged to
`research/queue/.corpus_checks.jsonl`) -- surfaced `2026-09-02-open-ended-qwen-routed-fact-clause-fallback.md`
(SS1's Qwen-hallucination citation) and `2026-08-26-spiking-broca-mouth-recall-surface-production-wirein-GO.md`
(confirming Touchpoint A's "open arbitrary prose... keeps the Qwen fallback" residual was already disclosed
back on 2026-08-26).

Tests: `CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest
tests/test_touchpoint_a_fact_clause_flag.py -q` -> `20 passed` in ~1s.

Smoke: `CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m research.runners._touchpoint_a_fact_clause_derisk
--smoke --out research/findings/raw/_touchpoint_a_fact_clause_smoke.json` -> exit 0,
`go_gate.structural_checks_passed=true`.

A filesystem-only, git-untracked dependency (`data/corpus/tinystories.txt`, gitignored, required somewhere in
the tiny-demo warm-up path) was symlinked from the primary checkout into this worktree to run the smoke -- the
SAME disclosed convenience `research/findings/2026-09-04-recall-gate-reaches-real-ltm-GO.md` and this task's
Stage-1 sibling both already used; not a code or git-tracked change.

Builds on: `research/findings/2026-09-04-onebrain-stage1-qwen-fallback-retire-GO.md` (Stage 1, the sibling this
task follows on from), `research/findings/2026-09-04-recall-gate-reaches-real-ltm-GO.md` (the fix that made
Touchpoint A's residual carry its full current weight), `research/findings/2026-09-03-one-brain-mouth-
integration-ROADMAP.md` (Stage 2, the mission this build-ahead preps), `research/findings/2026-09-02-open-
ended-qwen-routed-fact-clause-fallback.md` (the fact-clause mechanism this task reuses, and the Qwen-
hallucination evidence for why reusing it here matters), `research/findings/2026-08-26-spiking-broca-mouth-
recall-surface-production-wirein-GO.md` (names Touchpoint A's residual for the first time).
