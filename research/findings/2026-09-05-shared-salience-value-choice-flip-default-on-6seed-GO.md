---
type: finding
status: live
date: 2026-09-05
mechanism: shared-salience-value-choice-production-flip
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
runner: research/runners/_shared_salience_flip_soak.py
artifacts:
  - research/findings/raw/_shared_salience_flip_soak/verify_seed42.json
  - research/findings/raw/_shared_salience_flip_soak/verify_seeds_43_44_100_101_102.json
  - research/findings/raw/_shared_salience_flip_soak/verify_6seed.json
  - research/findings/raw/_shared_salience_afferent/verify_6seed_postflip.json
external: NO-EXTERNAL-NEEDED -- a production-flip verification of already-validated in-repo mechanisms (rank-4's
  shared spiking salience afferent, 6-seed GO; rank-20's REAL value-critic through that afferent, 6-seed GO); no
  new biology claim is made, only that the already-proven wiring stays no-regression + load-bearing when its own
  flag is flipped from an opt-in default to the production default.
---

# `BRAIN_SHARED_SALIENCE` flipped default-ON -- the value-choice consumer's no-regression + anti-hollow soak through the REAL ChatBrain (6-seed GO, rank-20 flip verification)

**Verdict: GO.** `research/coordination/scaffold_retirement_backlog.md` Track-1 flip campaign, rank-20's slice: the
shared spiking salience/novelty afferent (`research/runners/shared_salience_afferent.py`, backlog rank-4) is
flipped from default-OFF to **default-ON** (`_SHARED_SALIENCE_DEFAULT_ON = True`, mirroring the `BRAIN_VALUE_CHOICE`
2026-08-26 flip's own idiom), and this finding is the dedicated no-regression + anti-hollow soak that flip needed
-- run through the **REAL production `ChatBrain`** (`research.runners.brain_chat_tui.ChatBrain`, the object
`/api/brain-chat` builds and drives via `.gate()`/`.render()`), not the `_FakeChat` stand-in every prior de-risk of
this mechanism used, scoped to the **value-choice consumer** specifically (rank-20's mandate; rank-4/rank-5's own
consumers -- `da_mode_drives_chat` and `bg_action_selection_production_organ` -- are separate, parallel
verification efforts this finding does not claim to close).

## What this closes

Both mechanisms this flip activates were already 6-seed-GO at the wiring/context-function level, and both explicitly
named the SAME residual as future work:

- Rank-4 (`research/findings/2026-09-05-shared-spiking-salience-afferent-wired-GO.md`): *"A default-ON flip needs
  its own no-regression soak on the live production default (mirroring how da-gated-curiosity/da-gated-encoding
  were flipped only after a dedicated soak), which this de-risk does not attempt."*
- Rank-20 (`research/findings/2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md`): *"This does
  NOT flip any default... 'GO' here names this de-risk's OWN verdict... not a production-integration claim."* Its
  own 6-seed sweep against the REAL trained critic used a `_FakeChat` fixture (`stored_facts`/`agent.held_referent()`
  stubs), not `ChatBrain.gate()`/`.render()` -- i.e. not the path `/api/brain-chat` actually calls.

This finding closes that gap for the value-choice consumer: the REAL `ValueChoiceProductionOrgan` (the heavy trained
`striosome_value` critic, `value_train_trials=40`), wrapped exactly as production wraps it
(`value_choice_production_organ.install_value_choice`, `BRAIN_VALUE_CHOICE` left at its OWN existing production
default throughout -- this soak varies ONLY `BRAIN_SHARED_SALIENCE`), driven through `ChatBrain.gate()`/`.render()`
on the `onebrain` composer (the interactive/production default, not the `rf` fast-path), at all 6 project-standard
seeds.

## What changed

- **`research/runners/shared_salience_afferent.py`**: `shared_salience_enabled()` now resolves `BRAIN_SHARED_SALIENCE`
  unset as ACTIVE (`_SHARED_SALIENCE_DEFAULT_ON = True`); explicit `{0,false,no,off,''}` stays the byte-identical
  escape back to every consumer's pre-wiring host arithmetic. No other line in the gate function changed.
- **A pre-existing test-infra bug this flip exposed, fixed**: `research/runners/_shared_salience_afferent_derisk.py`'s
  `_clear_flags()` unset the env var to represent the OFF arm -- correct when the default was OFF, silently WRONG
  (both "off" and "on" arms read ON) the moment the default flips. This is exactly the class
  `tools/gates/flip_offarm_staleness.py` exists to catch (born from the 2026-08-27
  `BRAIN_SPIKING_MOUTH_RECALL` incident); `_clear_flags()` now sets the explicit `"0"` escape. Because
  `research/runners/_value_choice_neural_context_6seed_derisk.py` (rank-20's own runner) imports this exact helper,
  the fix repairs both files' OFF-arm semantics at the source, not per-caller. Verified: `flip_offarm_staleness`'s
  own selftest and a full-tree check both pass clean post-fix (0 violations).
- **Three consumer-file docstrings/comments updated** (`bg_action_selection_production_organ.py`,
  `value_choice_production_organ.py`, `webapp/da_mode_drives_chat.py`) that stated "default-OFF" or "OFF (unset) ->
  byte-identical" -- now accurate under the new default, with the explicit escape spelled out. No behavior changed
  by these edits, only stale prose.
- **New runner**: `research/runners/_shared_salience_flip_soak.py` (this finding's own instrument).
- **Re-confirmed, not just trusted**: rank-4's own 6-seed gate (`_shared_salience_afferent_derisk.py --seeds 42 43
  44 100 101 102`, organ core + all 3 consumer sites at the context-function level) re-run under the flipped default
  with the fix applied -- **6/6 seeds, `all_gates_pass=true`**
  (`research/findings/raw/_shared_salience_afferent/verify_6seed_postflip.json`).

## Method

`_build_chat(seed, "onebrain")` builds the tiny-demo `BrainConversationalAgent` (genuinely-spiking recall,
`use_multiturn=True`), wraps it as a `ChatBrain`, installs the GNW deliberation keystone (best-effort, matching
production) and `value_choice_production_organ.install_value_choice` with its DEFAULT `context_fn`
(`default_context_fn(chat)` -- the REAL shared-salience-reading engagement context, not an override), then drives
it through `chat.gate(question)` / `chat.render(...)` exactly as the webapp's single-fact path does. Each seed runs
in its OWN subprocess (the trained critic + the process-shared curiosity-organ singleton are not safely
re-buildable across seeds in one process, the same reason rank-4/rank-20's own runners subprocess-fan).

**Facts on top of the tiny-demo base corpus** (which already stores `dog chase cat` / `cat eat fish`): `dog chase
ball`, `dog chase shoe`, `dog chase stick` (recency ladder `[0, 1/3, 2/3, 1]`, mirrors rank-20's own S4 near-tie
construction) and `bird eat worm`, `bird eat seed` (recency `[0, 1]`, mirrors rank-20's own S1 wide separation), a
fresh (agent, action) pair so it cannot interact with the 4-candidate scenario.

**Turns, each seed:**
- ORDINARY (no ambiguity, value-choice never engaged -- `len(candidates) < 2`): *"what does cat eat"* (confident
  recall), *"what does fox hunt"* (untaught, abstain), *"what do you know about it"* (self/identity). Compared
  `BRAIN_SHARED_SALIENCE` OFF vs ON, `BRAIN_VALUE_CHOICE` untouched throughout.
- TRIGGER_2CAND *"what does bird eat"* (2 candidates, wide separation) and TRIGGER_4CAND *"what does dog chase"* (4
  candidates, near-tie) -- each run OFF / ON / ON+LESION (`BRAIN_SHARED_SALIENCE_LESION=1`). A read-only diagnostic
  call into the SAME `default_context_fn` the live turn just used reports the fed engagement floats alongside the
  categorical commit, so load-bearing/lesion-collapse are checked on the ACTUAL numbers reaching the critic, not
  inferred from the commit alone (rank-20's own S1-S3 evidence: the commit can legitimately stay unchanged even
  when the mediation is genuinely live).

## Result -- 6/6 seeds, all gates pass

<!--derived-->

The table + narrative below are computed FROM the 3 cited artifacts (rounded, aggregated across seeds/scenarios,
or hand-typed from the per-seed JSON) -- block-marked rather than inline-marked per number, per `docs/WRITING.md`'s
own `<!--derived-->` convention; the raw values are in the cited JSON at full float precision.

```
research/runners/_shared_salience_flip_soak.py  (seed 42 standalone; seeds 43/44/100/101/102 via --seeds)
verdict: GO   seed_pass: 6/6   ordinary_preserved: 6/6   on_loadbearing: 6/6   lesion_collapses: 6/6   moat_holds: 6/6
```

| seed | build (s) | 2cand off/on/lesion | 4cand off/on/lesion | 4cand reorder | on spread (2c / 4c) | lesion spread (2c / 4c) |
|---|---|---|---|---|---|---|
| 42 | 201.9 | seed / seed / *(abstain)* | shoe / shoe / ball | no | 1.060 / 1.080 | 0.0014 / 0.0000 |
| 43 | 282.0 | seed / seed / *(abstain)* | shoe / **stick** / stick | **yes** | 1.060 / 1.080 | 0.0014 / 0.0000 |
| 44 | 228.0 | seed / seed / *(abstain)* | ball / **stick** / stick | **yes** | 1.060 / 1.080 | 0.0014 / 0.0000 |
| 100 | 238.8 | seed / seed / *(abstain)* | shoe / **stick** / ball | **yes** | 1.060 / 0.764 | 0.0014 / 0.0000 |
| 101 | 198.0 | seed / seed / *(abstain)* | shoe / shoe / ball | no | 1.060 / 0.764 | 0.0014 / 0.0000 |
| 102 | 189.7 | seed / seed / *(abstain)* | shoe / shoe / ball | no | 1.060 / 0.764 | 0.0014 / 0.0000 |

`spread_attributable_to_drive_pathway` (`tools.lab.attributable_to`): **1.0** on every one of 12 (seed, scenario)
pairs -- the ON-arm spread's entire magnitude is attributable to the shared-salience pathway relative to its own
lesioned twin (the lesion spread is not merely smaller, it is the exact floor `_diag_context()` reports: `[0.0014,
0.0]` for 2cand and `[0.0,0.0,0.0,0.0]` EXACT for 4cand, on every seed).

**2-candidate (wide separation): 6/6 match, 6/6 lesion-abstain.** OFF and ON always commit `seed` (the higher-recency
candidate); LESION always declines (`fed_spread_hz=0.417 < v_margin_hz=2.0`), reverting to the inner gate's own
result (`"the bird eats the worm"`, the first-stored candidate) -- the wrapper's documented decline-to-verbatim
contract, not a crash or a wrong answer.

**4-candidate (near-tie): 3/6 match, 3/6 genuine reorderings (seeds 43, 44, 100).** The OFF arm itself splits across
seeds (`shoe` x5, `ball` x1 at seed 44) reflecting each seed's own trained critic having a near-tied value gradient
between the two highest-recency candidates, exactly the "a near-tie in the HOST arm's own value gradient, not a
mechanism failure" signature rank-20's own S4 analysis reports. The ON arm converges on
`stick` (the TRUE highest-recency candidate, `e=1.0`) on 4/6 seeds vs OFF's 1/6 -- a similar qualitative
"numerically more seed-consistent" pattern to rank-20's own S4 observation, reported as an observation (n=6 is not
statistical power for a reliability claim), not a headline. LESION reverts to `ball` on 4/6 seeds and `stick` on 2/6
(43, 44) -- NOT a single clean floor label the way rank-20's own S4 saw ("cat" x6), because once the fed context
is exactly zeroed the remaining ~6-19 Hz of residual variation across candidates is OU-noise jitter (see Honest
scope), and which candidate that jitter favors is not the gated property -- **the gated property, and the one that
holds clean on 6/6 seeds with no exception, is that the FED CONTEXT SPREAD itself collapses to EXACTLY `0.0` under
lesion**, a complete floor-collapse of the mediation's own output.

**Ordinary turns: byte-identical on 6/6 seeds**, e.g. seed 42: `["the cat eats the fish", "I don't know about
that.", "I don't know about that."]`, OFF and ON. This holds by construction (value-choice's wrapper returns the
inner gate result verbatim whenever `len(candidates) < 2`, so the shared-salience flag's blast radius never reaches
these turns in this harness) and is verified rather than assumed, per every seed's own `ordinary_off == ordinary_on`
check.

## Anti-cheats

- **`c_ordinary_preserved` (6/6).** The 3 ordinary turns are BYTE-IDENTICAL text, `BRAIN_SHARED_SALIENCE` off vs on,
  every seed -- the flip's blast radius does not leak into non-ambiguous turns through this consumer.
- **`c_on_loadbearing` (6/6, both scenarios).** The fed engagement context measurably differs ON vs OFF on both
  trigger scenarios, every seed -- confirmed on the ACTUAL floats reaching the REAL critic through the live
  `chat.gate()` call, not a value the harness derived separately.
- **`c_lesion_collapses` (6/6, both scenarios).** `BRAIN_SHARED_SALIENCE_LESION` collapses the ON-arm spread to
  <=0.5x on every seed (in practice: to the EXACT lesioned floor, 0.0014 / 0.0000, on every single seed) --
  `tools.lab.attributable_to` reports 1.0 on all 12 (seed, scenario) pairs, never banked unattributed.
  `BRAIN_SHARED_SALIENCE_LESION` is a DIFFERENT, upstream lesion from the pre-existing, already-6-seed-GO'd
  `BRAIN_VALUE_CHOICE_LESION` mean-pin (untouched by this soak) -- it severs only the shared ASK-pool afferent
  feeding the context, not the critic's own value gradient.
- **`c_moat_holds` (6/6).** Every commit across every arm, every scenario, every seed is either `None` (abstain) or
  one of the STORED candidates -- grep-verified against the exact candidate sets, never an invented patient.
- **Reused, not re-derived, flag helpers.** `_set_flags`/`_clear_flags` are imported from
  `_shared_salience_afferent_derisk.py` (now fixed), the SAME helpers rank-4's own suite uses -- no parallel,
  possibly-divergent flag-toggling logic.
- **A verdict carries what earned it.** `tools.verdict.Verdict` accumulates the 4 gates + a 5th ("every subprocess
  worker returned a result, none crashed/timed out") as `require()` preconditions; `decide()` computes GO only when
  all hold (`tools/gates/verdict_preconditions.py`-compliant).

## Honest scope, terms, and open questions (per `docs/TERMS.md`)

- **Scoped to the value-choice consumer, not all 3.** This finding does not independently re-verify
  `da_mode_drives_chat`/`bg_action_selection_production_organ`'s OWN behavior through an integrated `ChatBrain` under
  this SAME flip (that is rank-4/rank-5's own parallel Track-1 scope). It DOES reconfirm, at the context-function/
  direct-organ level, that rank-4's own 6-seed gate covering all 3 consumers still passes cleanly post-flip
  (`verify_6seed_postflip.json`, 6/6, after the `_clear_flags` fix) -- so the flip is not KNOWN to break the other
  two consumers, but their OWN integrated-level soak is a separate deliverable this finding does not claim.
- **A genuine, pre-existing, unrelated non-determinism was found and is NOT fixed here.**
  `value_choice_production_organ._stable_seed(a, v)` uses Python's built-in `hash()` on a tuple of strings, which is
  salted per-process (`PYTHONHASHSEED` unset in this repo's env setup) -- verified directly (two fresh interpreter
  invocations of `hash(('dog','chase'))` returned different values). This feeds the WTA's value-independent
  salience-bias baseline and can flip the categorical winner on a NEAR-TIED scenario across independent runs of
  "the same seed" (reproduced: an earlier seed-42 run committed `stick`/`stick` on the 4-candidate trigger; the
  canonical run in the table above committed `shoe`/`shoe` -- otherwise byte-identical on every gated quantity).
  This is LIVE IN PRODUCTION TODAY under `BRAIN_VALUE_CHOICE` (default-ON since 2026-08-26) wherever an ambiguity's
  value gradient is itself near-tied, independent of this flip. It does not touch any gate in this finding (every
  gate is built on the fed CONTEXT floats/spread, confirmed byte-identical across both runs; only the un-gated
  categorical near-tie winner moved). Logged: `research/FAILURE_LOG.md` (2026-09-05 entry), flagged as a follow-up
  task (fix `_stable_seed` to a process-stable hash, e.g. `zlib.crc32`), not fixed in this verification-scoped
  session.
- **`n_seeds_with_4cand_reorder=3/6` is a snapshot of this process's hash seed, not a fixed per-seed property** --
  see the non-determinism note above. The property that IS a fixed per-seed gate is the fed-context spread
  collapsing to exactly `0.0` under lesion, which held on all 6 seeds across BOTH the canonical run and the earlier
  hash-seed-varied rerun.
- **Does not touch `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`.** No row exists yet for this mechanism; adding one is
  left to the controller's consolidated update once the parallel Track-1 ranks (4, 5, 12, 13, 20) land, to avoid 5
  isolated worktrees racing edits to one shared cross-cutting file.
- **This finding does NOT merge the flip to `main`.** It verifies the flip on a feature branch
  (`worktree-wf_5ba45863-4fb-4`); the controller merges GOs.
- **FUNCTIONAL correlate, not phenomenal** -- inherited from rank-4/rank-20/RANK-1; this finding changes neither
  claim.
- **CO-RESIDENT, not yet merged onto the one-brain substrate** -- inherited from rank-4; unchanged here.

## Files
`research/runners/shared_salience_afferent.py` (the flip), `research/runners/_shared_salience_afferent_derisk.py`
(the offarm-staleness fix + a docstring update), `research/runners/_shared_salience_flip_soak.py` (new runner, this
finding's own instrument), `research/runners/bg_action_selection_production_organ.py` +
`research/runners/value_choice_production_organ.py` + `webapp/da_mode_drives_chat.py` (stale-prose fixes only, no
behavior change), `research/FAILURE_LOG.md` (the `_stable_seed` non-determinism entry). Artifacts:
`research/findings/raw/_shared_salience_flip_soak/verify_seed42.json`,
`research/findings/raw/_shared_salience_flip_soak/verify_seeds_43_44_100_101_102.json`,
`research/findings/raw/_shared_salience_flip_soak/verify_6seed.json` (the merged 6-seed gate),
`research/findings/raw/_shared_salience_afferent/verify_6seed_postflip.json` (rank-4's own gate, reconfirmed).

## Citations
- Scaffold-retirement map (this de-risk's mandate): `research/coordination/scaffold_retirement_backlog.md` rank-20,
  Track-1 flip campaign status updates.
- The wiring this flips (reused verbatim, not re-derived): `research/findings/2026-09-05-shared-spiking-salience-afferent-wired-GO.md`
  (rank-4).
- The REAL-critic 6-seed gate this extends to the live entry point: `research/findings/2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md`
  (rank-20).
- The critic itself (reused verbatim, not modified): `research/findings/2026-07-23-value-critic-closure-RANK1-GO.md`
  (RANK-1 GO).
- The precedent flip-soak pattern this mirrors: `research/findings/2026-08-26-value-driven-choice-production-wirein-GO.md`
  and its runner `research/runners/_value_choice_flip_soak.py`.
- The offarm-staleness class this fix belongs to: `tools/gates/flip_offarm_staleness.py` (born 2026-08-27).
- Verdict discipline: `tools/verdict.py`. Attribution discipline: `tools/lab.py::attributable_to`.
