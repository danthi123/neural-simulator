---
type: finding
status: contributing
verdict: NO-GO
mechanism: rank-13 production-flip verification — BRAIN_NEURAL_SELFID / BRAIN_NEURAL_ANAPHORA_ABSTAIN default-ON in research/runners/brain_chat_tui.py, verified against the actual production ChatBrain + the installed GNW ignition bus (webapp/server.py::brain_reply's own combiner), both the numpy fast-path composer (rf) and the TRUE production composer (onebrain), 6 genuinely-varied seeds
lane: integration-first (WIRING BACKLOG rank-13) — Track 1, ship-the-validated-wins
integration_faculty: content-selection
date: 2026-09-05
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/runners/_rank13_selfid_anaphora_prodflip_verify.py
  - research/findings/raw/_rank13_selfid_anaphora_prodflip/result_fast_only.json
  - research/findings/raw/_rank13_selfid_anaphora_prodflip/result.json
verification: |
  SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m \
    research.runners._rank13_selfid_anaphora_prodflip_verify --skip-onebrain
    (fast/rf battery, both combiners, 6 genuinely-varied seeds; thread-capped only to be a considerate neighbor
    on a heavily-shared host -- the decisive seed=44 finding was separately confirmed under DEFAULT/unrestricted
    threading too, see "Threading sensitivity")
  SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m \
    research.runners._rank13_selfid_anaphora_prodflip_verify --onebrain-only
    (slow/onebrain battery, the TRUE production composer + installed GNW bus, 6 genuinely-varied seeds)
  LESION (the crux): 6/6 seeds, onebrain composer -- intact=['brain','use','spikes'] correct, lesioned=None
  (collapses), reflex='spikes' (survives) on EVERY seed; attribution 100.0% every seed. SOLID, unambiguous.
  REGRESSION (the reason for NO-GO): anaphora-HIT ("what does dog chase?" -> established referent 'cat', then
  "what does it eat?" 7+ unrelated turns later) is CORRECT on 5/6 seeds and WRONG (incorrectly abstains) on
  seed=44, on BOTH composer kinds (rf AND onebrain), reproduced across 2 independent threading configurations
  for rf. A second, less robust (threading-dependent) manifestation at seed=43/rf additionally returns a
  confabulated ['brain','use','spikes'] for a genuine anaphora-MISS instead of abstaining. See "Root cause".
---

# Rank-13 production-flip verification: NO-GO — a stale discourse referent lets the self-alias resolution misroute a legitimate anaphoric "it"

**Verdict: NO-GO.** The mechanism's LOAD-BEARING crux is genuinely solid (6/6 seeds, unambiguous). But requirement
1 (no regression) fails on seed=44, reproducibly, on BOTH composer kinds the flip is supposed to cover — a
previously-correct anaphora-HIT recall breaks under the flip once the conversation is long enough for the
discourse-WM referent to go stale. This is the honest deliverable this verification exists to produce. Raw
per-seed data: `research/findings/raw/_rank13_selfid_anaphora_prodflip/result_fast_only.json` (rf composer,
both combiners) and `research/findings/raw/_rank13_selfid_anaphora_prodflip/result.json` (onebrain composer).

## What this verifies, and why it differs from the de-risk that earned the flag

[`2026-09-05-rank13-selfid-anaphora-scaffold-derisk-GO-6of6.md`](2026-09-05-rank13-selfid-anaphora-scaffold-derisk-GO-6of6.md)
earned a 6-seed GO with `BRAIN_NEURAL_SELFID`/`BRAIN_NEURAL_ANAPHORA_ABSTAIN` default OFF. This session flipped both
to default ON (`research/runners/brain_chat_tui.py`: `_NEURAL_SELFID_DEFAULT_ON`/`_NEURAL_ANAPHORA_ABSTAIN_DEFAULT_ON`
= True) and built `research/runners/_rank13_selfid_anaphora_prodflip_verify.py` to verify the flip against the
REAL production `ChatBrain` + the installed GNW bus, on genuinely-varied substrate.

**A methodological repair, found while building this verification (logged
[`research/FAILURE_LOG.md`](../FAILURE_LOG.md) 2026-09-05):** `webapp.server._build_chat_brain('tiny-demo', ...)`
calls `_build_tiny_demo(42, ...)` with a HARDCODED literal `42`. The de-risk's own `_build(seed, ...)` accepted a
`seed` parameter but never threaded it anywhere -- its "6 seeds" built the IDENTICAL substrate six times (the
documented `cfg.seed` trap, CLAUDE.md, one call-stack level up). This verification's `_real_seed()` monkeypatches
`research.runners.brain_chat_tui._build_tiny_demo` for the duration of each build (`_build_chat_brain`'s import
of it is a LOCAL, re-resolved-per-call import) to thread a REAL seed through -- confirmed genuinely different
codebooks across seeds, empirically, not assumed. **This repair is the reason the regression below was ever
seen**: the de-risk's accidental seed-42-only testing could never have caught a seed=44-specific failure.

## Requirement 2 (load-bearing, not hollow): SOLID, 6/6 seeds, the true production composer

The de-risk's own honest scoping named class (a) -- the self-referential factual SVO ("what do you use?") -- as
the ONLY class here with a genuinely-neural mechanism (the on-brain `BridgeParser.role_of`) to lesion. This
verification re-ran that lesion on the TRUE production default composer (`onebrain`, `OneBrainComposer` bound to
production pool #1, `Pool1BoundOneBrainComposer`), at 6 genuinely-different seeds:

| seed | intact | lesioned | reflex (`query_patient`) | attribution |
|---|---|---|---|---|
| 42 | `['brain','use','spikes']` | `None` | `'spikes'` | 100.0% |
| 43 | `['brain','use','spikes']` | `None` | `'spikes'` | 100.0% |
| 44 | `['brain','use','spikes']` | `None` | `'spikes'` | 100.0% |
| 100 | `['brain','use','spikes']` | `None` | `'spikes'` | 100.0% |
| 101 | `['brain','use','spikes']` | `None` | `'spikes'` | 100.0% |
| 102 | `['brain','use','spikes']` | `None` | `'spikes'` | 100.0% |

Every seed: the intact answer is correct, the lesion COLLAPSES it to abstain, the parser-independent recall
reflex SURVIVES (the substrate still has the fact; only the comprehension that would route to it is gone). This
is the textbook dissociation `docs/TERMS.md`'s "integrated / production-default" entry requires for level-3
"spiking on-by-default" credit, and it holds without exception. **The hollow-mouth failure mode does not apply
to this class**: toggling the lesion visibly, causally, and completely changes the outcome.

self-factual and self-identity correctness + host-router retirement (0 calls to `_gate_router_combine`/
`QuestionRouter.match_fact`, call-count-measured) also held 6/6 seeds, both composer kinds, both combiners
(`plain` host `gate()` and the installed GNW-bus `gate_via_bus` -- the actual `brain_reply` combiner).

## Requirement 1 (no regression): FAILS on seed=44 — this is the NO-GO

The verification battery asks a 12-question panel in one ordered conversation per build (self-factual x3,
self-identity x2, STORED x3, UNSTORED x2, then an anaphora tail: "what does it eat?" [a legitimate follow-up to
the STORED "what does dog chase?" turn, which established the referent 'cat'] then "what does it fly?" [a
genuine anaphora-miss -- cats have no fly fact]). On **5 of 6 seeds**, both composer kinds, "what does it eat?"
correctly recalls `['cat','eat','fish']`, flag-on and flag-off alike. **On seed=44, flag ON, it INCORRECTLY
abstains (`None`)** -- on the `rf` composer AND independently on the TRUE production `onebrain` composer:

| seed | rf composer, "it eat" | onebrain composer, "it eat" |
|---|---|---|
| 42 | `['cat','eat','fish']` correct | `['cat','eat','fish']` correct |
| 43 | `['cat','eat','fish']` correct | `['cat','eat','fish']` correct |
| **44** | **`None` WRONG** | **`None` WRONG** |
| 100 | `['cat','eat','fish']` correct | `['cat','eat','fish']` correct |
| 101 | `['cat','eat','fish']` correct | `['cat','eat','fish']` correct |
| 102 | `['cat','eat','fish']` correct | `['cat','eat','fish']` correct |

This is not a fluke of one run: seed=44's failure reproduced independently across 3 separate process invocations
(the original battery, an isolated re-trace, and a from-scratch rerun) and across 2 different threading
configurations (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` capped at 2, and fully default/
unrestricted -- the condition that actually matches `webapp/server.py`, which sets none of these). A SECOND,
distinct manifestation appeared at seed=43 on the `rf` composer under the thread-capped configuration only (NOT
reproduced under default threading, see "Threading sensitivity" below): "what does it fly?" (the anaphora-MISS)
returned a confabulated `['brain','use','spikes']` instead of abstaining -- a CONFIDENT WRONG ANSWER, strictly
worse than either the correct abstain or the pre-flip host-router's own (also-abstaining, on this fixture)
fallback.

### Root cause (read from source, then confirmed by direct instrumentation)

`ChatBrain.gate()` resolves anaphora BEFORE extraction: `q = self._resolve_anaphora(question)` substitutes a
literal "it"/"that"/"they"/"this" with `self.agent.held_referent()[0]` -- but ONLY if that referent is not
`None`. When the discourse-WM referent has decayed (a real property of the referent-holding circuit, unrelated
to this flip -- confirmed by direct trace: `held_referent()`'s confidence value drifts continuously turn-to-turn
and both its value and the point at which it goes `None` vary with the substrate's own state), `_resolve_anaphora`
returns the question UNCHANGED -- the literal token "it" survives into `_extract_route`.

`_extract_route`'s SELFID extension then runs `content = [self.router._resolve_self(t) for t in content]`
UNCONDITIONALLY whenever `BRAIN_NEURAL_SELFID` is on -- and "it" is itself a member of `DEFAULT_SELF_ALIASES`
(`{"you","your","yours","i","me","my","it","its","yourself","itself"}`, `brain_chat_tui.py:530`). The extension
cannot distinguish "it" that anaphora resolution already failed to substitute (meaning it) from a genuine
self-referential "it" ("what is it made of", asking about the brain itself) -- it resolves BOTH to `'brain'`.
For "it eat" this makes the query `(brain, eat)`; `what_does('brain','eat')` misses, and -- unlike the pre-flip
behavior, which would have fallen through to the host router (`_gate_router_combine`, whose own bag-of-words
matcher apparently CAN find `cat eat fish` for this specific fixture) -- the query is now trapped in the
self-referential branch and abstains. For "it fly" at seed=43, the SAME misroute additionally hit the SELFID
miss-only candidate-relation retry (`has`/`have`/`is`/`uses`/`use` against agent `'brain'`), which found `'use'`
and returned `['brain','use','spikes']` -- a fabricated answer to a question that was never about the brain.

**This is a genuine, flip-attributable regression, not a test-harness artifact of the verification's own 12-turn
panel.** The panel's length is what makes the referent-decay condition easy to REACH in one build, but the
condition itself (a discourse referent going stale before a real follow-up "it" question) is an ordinary property
of any moderately-extended real conversation, and the flip's response to it -- silently reinterpreting an
un-resolved anaphoric "it" as self-referential -- is a genuine defect in the flip's own design, independent of
how the defect was found. A companion isolated 2-turn test (matching the de-risk's own probe shape exactly:
"what does dog chase?" then immediately "what does it fly?", NO intervening turns) was clean at all 3 seeds
tested (42/43/44) -- so the defect requires SOME conversational distance from the referent-establishing turn to
manifest, but nothing in the mechanism bounds that distance to "never happens in production."

### Threading sensitivity (a second, independent methodological finding)

The seed=43/"it fly" manifestation reproduced under thread-capped execution (`OMP_NUM_THREADS=2` etc., adopted
mid-session purely to be a considerate neighbor on a heavily-shared machine) but did NOT reproduce for the
IDENTICAL seed/panel/flag-state under default/unrestricted threading. The seed=44/"it eat" manifestation, by
contrast, reproduced under BOTH configurations -- establishing it as the robust, environment-independent finding
this NO-GO rests on. The mechanism responsible is almost certainly the discourse-referent circuit's own
floating-point-order sensitivity (a chaotic spiking system's discrete outcome depending on BLAS parallel-reduction
order, which varies with thread count) rather than anything about `cfg.seed` -- worth a follow-up in its own
right (a "seed" alone does not fully pin outcome-determinism for this circuit), but outside this verification's
scope. `webapp/server.py` sets no thread-count env vars, so DEFAULT threading is what production actually runs;
that is why seed=44's threading-independent failure, not seed=43's threading-dependent one, is the basis for
this NO-GO.

## Requirement 3 (real default): holds

`BRAIN_NEURAL_SELFID`/`BRAIN_NEURAL_ANAPHORA_ABSTAIN` resolve `True` with the env vars UNSET (checked directly,
zero-cost) -- the flip source edit itself is correct and does what it says. This requirement is not in question;
it is requirements 1 and 2's INTERACTION (a real default change exercising a real, if narrow, gap) that fails.

## Bottom line

**NO-GO.** Bank the finding, do not ship the flip as currently implemented. The failing METHOD is narrow and
nameable: `_extract_route`'s self-alias resolution must not treat a literal, un-substituted anaphoric "it" as
self-referential -- it should distinguish "anaphora resolution ran and left 'it' unresolved" (do NOT self-alias
it; fall through to the pre-flip abstain-or-router path, preserving today's behavior for this class) from "the
user typed a genuinely bare 'it'/'you' with no antecedent" (the case the flip is actually meant to cover). A
concrete, scoped fix: thread `anaphora_used` (already computed in `gate()`) into `_extract_route`, and skip the
self-alias resolution of literal `it`/`its`/`itself` specifically (the anaphora-capable aliases) whenever
`anaphora_used` is true OR a referent was ever held this conversation but is now stale -- `you`/`i`/`me`/`my` etc.
are never anaphora targets and are unaffected. This is a METHOD-level fix, not a capability retreat: THE LAW
holds -- bank this NO-GO, take the next lever, keep the capability open.

## Honest scope

- The onebrain battery's regression check (STORED/UNSTORED, flag-on vs the true default) was not separately
  rebuilt under an explicit-OFF arm on the onebrain composer (matching the original de-risk's own scoping
  reasoning: neither class contains a self-alias token, so the flag structurally cannot perturb them) -- the load-
  bearing NO-GO above does not depend on that scope choice.
- The seed=43 confabulation manifestation was not re-swept across all 6 seeds under default threading (only
  seed=44's robustness was fully cross-checked this way, since it alone is sufficient grounds for NO-GO); it is
  reported as an additional, independently-real-but-less-robustly-reproduced data point, not double-counted in
  the verdict.
- A companion single-build sanity pass against the OWNER's actual deployed bundle (`bridges/developed/scale787/
  day_33`, `composer_kind='rf'`, 404 real facts -- the brain the owner actually talks to, distinct from the
  `tiny-demo` fixture both this and the de-risk use) was attempted and NOT completed: that bundle's `onebrain`-
  scale `dlpfc_wm`/`cortex_ctx` pool allocates 112,640 neurons and did not finish connection-generation within
  ~5 minutes on this shared host. Banked, not deleted: `research/runners/_rank13_selfid_anaphora_realbundle_sanity.py`.
