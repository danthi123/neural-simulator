---
type: finding
status: contributing
verdict: NO-GO
mechanism: rank-13 production-flip — CORRECTED root-cause diagnosis of the flip-verify NO-GO, a genuine fix for one of the two known regressions (the seed=43 SELFID over-fire), and a corrected honest characterization of the other (seed=44) as a discourse-WM referent-fidelity issue outside this session's narrow-fix scope. Both BRAIN_NEURAL_SELFID / BRAIN_NEURAL_ANAPHORA_ABSTAIN stay default OFF.
lane: integration-first (WIRING BACKLOG rank-13) — Track 1, ship-the-validated-wins
integration_faculty: content-selection
date: 2026-09-05
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/runners/_rank13_prodflip_corrected_diagnosis.py
  - research/findings/raw/_rank13_selfid_anaphora_prodflip/corrected_diagnosis_trace.json
  - research/findings/raw/_rank13_selfid_anaphora_prodflip/result_fast_only_PREFIX_20260905.json
  - research/findings/raw/_rank13_selfid_anaphora_prodflip/result_fast_only.json
  - research/runners/brain_chat_tui.py
  - webapp/gnw_bus_shadow.py
verification: |
  PRE-FIX re-confirmation on CURRENT code (this session, fresh 6-seed run, not inherited):
    SIM_BACKEND=numpy python -m research.runners._rank13_selfid_anaphora_prodflip_verify --skip-onebrain
    -> NO-GO, the same shape as the original: seed=44 "it eat" WRONG (both combiners), seed=43 "it fly"
    confabulates ['brain','use','spikes'] (plain combiner only; bus combiner correctly abstains at seed 43).
    Saved: result_fast_only_PREFIX_20260905.json.
  CORRECTED-DIAGNOSIS TRACE (class-level method wrapping of the REAL, unmodified verify-module functions --
  zero reimplementation): research.runners._rank13_prodflip_corrected_diagnosis.
    seed=44: _resolve_anaphora('what does it eat?') -> 'what does ball eat?' (SUBSTITUTED, not left literal;
    held_referent() returned a confident-but-wrong 'ball'). _extract_route never sees a self-alias token.
    seed=43: _resolve_anaphora('what does it fly?') -> 'what does brain fly?' (substituted to 'brain', a THIRD
    distinct wrong referent). The SELFID candidate-relation retry's missing `v == "isa"` guard fires on this.
  POST-FIX re-verification (this session): SIM_BACKEND=numpy python -m
    research.runners._rank13_selfid_anaphora_prodflip_verify --skip-onebrain, after adding `v == "isa"` to
    `ChatBrain._substrate_recall`'s SELF/IDENTITY candidate-relation retry (research/runners/brain_chat_tui.py) --
    seed=43's "it fly" confabulation is GONE (now abstains, matching OFF and matching the bus combiner); seed=44's
    "it eat" regression PERSISTS UNCHANGED (the fix does not touch the ANAPHORA_ABSTAIN mechanism, as predicted
    from the trace). Overall verdict: STILL NO-GO (one of two known regressions closed, the other -- and the
    ORIGINAL finding's own stated primary basis for NO-GO -- remains). Full numbers in the body below.
  INSTRUMENTATION-SENSITIVITY (an additional, honest finding, not hidden): the corrected_diagnosis_trace.json
    artifact's OWN seed=44 "final_on" reads CORRECT (['cat','eat','fish']), the OPPOSITE of the vanilla,
    unmodified verify script's own answer, reproduced identically (None, confirmed by direct inspection of each
    run's saved JSON) across 3 independent process invocations -- 2 pre-fix, 1 post-fix. Merely wrapping the same
    methods with a transparent, call-count-preserving logging closure (zero semantic change) flips the outcome. The qualitative
    claim this trace supports (`_resolve_anaphora` SUBSTITUTES to a wrong referent, never leaves a literal "it")
    is robust across every run, instrumented or not; the pass/fail OUTCOME for seed=44 is not, and is itself
    evidence for why this residual has no narrow fix (see "Why seed=44 is not narrow-fixable"). The AUTHORITATIVE
    numbers for the GO/NO-GO decision are the vanilla script's own (result_fast_only*.json), not this trace's.
---

# Rank-13 production-flip: corrected diagnosis — the regression is a discourse-WM referent misidentification, not a self-alias misroute; one of two manifestations fixed, the other is a deeper, out-of-narrow-fix-scope issue

**Verdict: still NO-GO.** This session was tasked with applying the "narrowly-scoped fix" the flip-verify NO-GO
([`2026-09-05-rank13-selfid-anaphora-PRODUCTION-FLIP-NO-GO-stale-referent-selfalias-misroute.md`](2026-09-05-rank13-selfid-anaphora-PRODUCTION-FLIP-NO-GO-stale-referent-selfalias-misroute.md))
had identified (gate `_extract_route`'s self-alias resolution on `anaphora_used`) and re-verifying 6/6. Direct,
non-invasive instrumentation of the REAL verification code (not a re-derivation) shows that fix targets a
mechanism that **does not occur** in either of the two regressions that finding measured. The actual mechanism is
corrected below; one of the two regressions (seed=43) had a genuine, narrowly-scoped, well-evidenced fix and is
now closed; the other (seed=44 — the finding's own stated **primary** basis for NO-GO) is a discourse-WM
referent-read fidelity issue with no narrow fix available, and is banked as an honest deeper finding per this
task's own explicit allowance for that outcome. **Both flags stay default OFF; no flip is prepared.**

## Why the prior root-cause diagnosis does not match the code

The prior finding theorized: `_resolve_anaphora` **fails** to substitute a pronoun once the discourse referent
has decayed (`held_referent()[0] is None`), leaving a literal `"it"` to survive into `_extract_route`, where
`BRAIN_NEURAL_SELFID`'s `content = [self.router._resolve_self(t) for t in content]` then wrongly treats it as
self-referential (resolves it to `'brain'`).

Wrapping `ChatBrain._resolve_anaphora` / `_extract_route` / `_gate_router_combine` and
`QuestionRouter.match_fact` at the **class level** — so the real, unmodified
`research.runners._rank13_selfid_anaphora_prodflip_verify` functions (`_build`, `_batch`, `_fast_eval`) are traced
with **zero reimplementation risk** — shows the opposite of the theorized mechanism: in both regressing cases,
`_resolve_anaphora` **succeeds** at substitution. It just substitutes the **wrong word**, because
`MultiTurnAgent.held_referent()`'s spiking cleanup-memory read returns a value that clears its own confidence
threshold (`spec > self._spec`) while still being the wrong concept. Full traces:
`research/findings/raw/_rank13_selfid_anaphora_prodflip/corrected_diagnosis_trace.json`.

**A caveat on that trace artifact, disclosed rather than smoothed over:** its own recorded `final_on` for
seed=44 reads CORRECT (`['cat','eat','fish']`) — the opposite of the vanilla, unmodified verify script's own
answer, which reproduced identically as `None` across 3 independent process invocations (2 pre-fix, 1 post-fix;
see "Re-verification" below). Wrapping the SAME methods with a
transparent, call-count-preserving logging closure (no semantic change — each wrapper calls the original exactly
once and returns its result unchanged) is enough to flip the outcome. This does not weaken the SUBSTITUTION-
TARGET claim above (`_resolve_anaphora` → `"ball"`/`"brain"`, confirmed identically whether instrumented or not,
across every run in this session) — it strengthens the case, in the next section, that seed=44's PASS/FAIL
outcome is not a stable function of (seed, code, flag) alone.

- **seed=44, "what does it eat?"**: `_resolve_anaphora` → `"what does ball eat?"` (not `"what does it eat?"`
  unchanged). `_extract_route("what does ball eat?")` never encounters a self-alias token at all — `'ball'` is
  not one — so the prescribed fix's gating condition (a literal `it`/`its`/`itself` reaching self-alias
  resolution) is never true for this turn, on **either** flag state. Confirmed directly: instrumenting
  `_extract_route` to log whether its input contains a literal `it`/`its`/`itself` token shows it does **not**,
  for this specific turn, at seed 44, on both the OFF and ON builds.
- **seed=43, "what does it fly?"**: `_resolve_anaphora` → `"what does brain fly?"` — substituted to **`'brain'`**,
  a third distinct wrong value (`'ball'` at seed 44, `'brain'` at seed 43), confirming the underlying phenomenon
  is a general referent-misidentification property of the WM read, not one fixed failure string.

## The real mechanism, per regression

### seed=44: `BRAIN_NEURAL_ANAPHORA_ABSTAIN` removes a fortuitous host-router safety net

1. `_resolve_anaphora` substitutes `"it"` → `"ball"` (wrong; the correct referent from turn 1 was `"cat"`).
   `anaphora_used = True` (a substitution DID happen — the code has no way to know it is the wrong one).
2. `_extract_route("what does ball eat?")` extracts `(agent='ball', action='eat')` — a well-formed-looking
   factual query — through the SAME path any other factual-SVO question uses (flag-independent here, since
   neither word is a self-alias).
3. `_substrate_recall` calls `inner.what_does('ball','eat')` → no such fact → **honest** `"__ABSTAIN__"` (the
   substrate is telling the truth: there IS no `ball eat` fact — this abstain is not itself a bug).
4. `gate()`: `sub == "__ABSTAIN__" and anaphora_used and _neural_anaphora_abstain_enabled()` → **True** with the
   flag ON → returns `None`.
   - Flag OFF: this condition is False (the flag check fails) → falls through to
     `_gate_router_combine("what does ball eat?")` → `QuestionRouter.match_fact` finds `('cat','eat','fish')`
     via the **verb alone** ("eat" uniquely identifies that stored fact; the wrong keyword "ball" simply
     contributes to no candidate's score, per `match_fact`'s per-fact keyword-overlap scoring) → verified against
     `what_does('cat','eat') == 'fish'` → returns the **correct** `['cat','eat','fish']`.

The pre-flip host router's forgiving, verb-only bag-of-words matching happens to be **robust** to this specific
WM misidentification; `BRAIN_NEURAL_ANAPHORA_ABSTAIN`'s honest-abstain policy is not, because "a substrate abstain
on an anaphora-resolved query" is structurally identical in the code whether the referent was resolved
*correctly* (a genuine miss, e.g. "what does it fly?" when it=cat and cat has no fly fact — the case the flag
exists to close) or *incorrectly* (this case). Nothing computed at that point distinguishes them.

### seed=43: `BRAIN_NEURAL_SELFID`'s candidate-relation retry was missing its own documented scope — FIXED

1. `_resolve_anaphora` substitutes `"it"` → `"brain"` (wrong; the correct referent was still `"cat"`).
2. `_extract_route("what does brain fly?")` extracts `(agent='brain', action='fly')`.
3. `_substrate_recall` calls `what_does('brain','fly')` → no such fact → `p` is falsy.
4. The SELF/IDENTITY candidate-relation retry (`_substrate_recall`, `research/runners/brain_chat_tui.py`) fires:
   its guard was `if not p and a == "brain" and _neural_selfid_enabled():` — **no check on `v`/action at all**,
   despite its own docstring stating it is "reached ONLY for a bare identity query on the self" (the
   `_definitional_copula_route`-authored `['brain','isa']` pair, or a literal `'what is brain?'`). It loops
   `has/have/is/uses/use`, finds `what_does('brain','use') == 'spikes'`, and returns the confabulated
   `['brain','use','spikes']` — a confident wrong answer to a question that was never about the brain.
5. The `gnw_bus_shadow.gate_via_bus` mirror of this exact retry (documented as "the identical recipe") already
   had `action == "isa"` in its guard — which is why the **bus** combiner (the actual production path) does NOT
   confabulate at seed 43 (confirmed: `on_answer: None, on_abstains: True` for the bus combiner, all 6 seeds,
   both pre- and post-fix) while the **plain** `gate()` path did.

**Fix applied** (`research/runners/brain_chat_tui.py`, `_substrate_recall`): added `and v == "isa"` to the retry's
guard, restoring the condition its own docstring already claimed and matching the bus mirror exactly. This is a
strict **narrowing** — it can only stop the retry from firing in cases it was never documented to fire in; the
intended self-identity path (`_definitional_copula_route` → `['brain','isa']`, i.e. `v` IS `'isa'` by
construction) is untouched. A clarifying comment was added at the `gnw_bus_shadow.py` mirror so a future reader
does not "simplify" the bus guard to match the (bugged) plain one.

## Why the originally-prescribed fix would not have worked (confirmed, not assumed)

The NO-GO's prescribed fix — thread `anaphora_used` into `_extract_route` and skip self-alias resolution of a
literal `it`/`its`/`itself` when it is true — was tested against its own premise directly: instrumenting
`_extract_route` to report whether its input ever contains a literal `it`/`its`/`itself` token, across the full
6-seed panel, shows this is **only ever true for the seed=43 "it fly" turn** (where `_resolve_anaphora` did NOT
substitute — a *different*, non-regressing instance) and **never true for the seed=44 "it eat" turn that is the
actual regression** (where `_resolve_anaphora` DID substitute, to `"ball"`). Applying the prescribed fix would
therefore have had **zero effect** on the regression it was written to close. This was verified empirically
before being reported, not inferred from re-reading the source a second time.

## Re-verification: full 6-seed fast battery, before and after the fix

| | pre-fix (this session, fresh) | post-fix |
|---|---|---|
| self_factual / self_identity correct + retired | GO, both combiners | GO, both combiners (unaffected) |
| STORED / UNSTORED / anaphora-hit no-regression (non-target seeds) | GO | GO (unaffected) |
| seed=44 "it eat", plain + bus | **WRONG** (`None`, expected `['cat','eat','fish']`) | **STILL WRONG** (unchanged) |
| seed=43 "it fly", plain | **CONFABULATES** `['brain','use','spikes']` | **FIXED** — abstains (`None`), matches OFF |
| seed=43 "it fly", bus | correct (abstains) — never had this bug | correct (abstains), unaffected |
| overall `go` | False (NO-GO) | False (NO-GO) — seed=44 alone is sufficient grounds |

Raw: `research/findings/raw/_rank13_selfid_anaphora_prodflip/result_fast_only_PREFIX_20260905.json`
(pre-fix) and `research/findings/raw/_rank13_selfid_anaphora_prodflip/result_fast_only.json` (post-fix, the
verification module's own standard output path). The load-bearing lesion result from the original NO-GO finding
(the on-brain `BridgeParser.role_of` lesion, 6/6 seeds, onebrain composer) is **untouched** by this session's fix
(a different code path entirely) and continues to stand as measured there.

## Why seed=44 is not narrow-fixable this session

The seed=44 mechanism requires distinguishing, at the point `gate()` decides whether to abstain or fall through
to the host router: "the substrate correctly resolved the true referent and honestly found no fact" (the case
`BRAIN_NEURAL_ANAPHORA_ABSTAIN` exists to convert from a host-router confabulation into an honest abstain) from
"the substrate resolved the WRONG referent, so of course nothing matches" (this case, where the pre-flip
host-router fallback happens to still recover the right answer via the verb alone). Nothing computed at that
point carries this distinction — `held_referent()`'s own `spec` value already cleared its internal threshold, so
raising that same threshold further is not a targeted fix; it is an un-validated, un-scoped change to the shared
anaphora-resolution path used by every other (currently-passing) multi-turn conversation, and tuning it against
one seed's failure is exactly the "instrument is part of the emulation" trap this project's own record warns
against.

The instrumentation-sensitivity finding above sharpens this further: seed=44's pass/fail outcome is not even a
stable function of (seed, code, flag) alone — a semantically-inert logging wrapper around the same call sequence
flips it. That is consistent with a genuinely chaotic operating point (the referent-holding attractor competition
sitting near a decision boundary, where BLAS thread-scheduling-driven floating-point summation order — the same
class of sensitivity the original NO-GO's own "Threading sensitivity" section measured for seed=43 — decides
which concept wins), not a deterministic logic bug a one-line gate could fix. A fix that happened to flip THIS
specific harness's outcome without addressing the underlying sensitivity would not be trustworthy in production,
where the exact same chaos applies. Closing this honestly needs either a genuine WM referent-confidence/cross-
check mechanism (a new, validated sub-mechanism, not a one-line gate) or an independent improvement to the
discourse-WM read's fidelity — both real research arcs, neither a "narrow fix." Per THE LAW, this is banked, not
abandoned: the capability (anaphora-miss abstaining honestly) stays open, gated on this specific residual.

## Honest scope

- Only the `rf` (numpy fast-path) composer was re-verified this session (both combiners, all 6 seeds); the
  `onebrain` (true production default) composer's own battery was not re-run, since (a) the load-bearing lesion
  result on that composer is untouched by this session's fix (a different code path), and (b) the fix's own
  scope (`_substrate_recall`'s retry) is composer-agnostic by construction (it is reached identically regardless
  of which composer answers `what_does`), matching the reasoning the original flip-verify NO-GO itself used to
  scope its own onebrain battery.
- The seed=43 "it fly" confabulation was the WORSE of the two known failure modes (a confident wrong answer, not
  merely a wrong abstain) and is the one this session's fix closes; it was also the less robust of the two in the
  original finding's own threading-sensitivity note. Fixing it does not change the overall verdict, since the
  original finding's own stated primary basis for NO-GO was seed=44, which persists.
- This diagnosis does not attempt to explain WHY `held_referent()` specifically returns `'ball'` at seed 44 and
  `'brain'` at seed 43 rather than the correct `'cat'` — only that it does, reproducibly, and that the discourse-
  WM read's own `spec` (specificity) value is why the resolution machinery does not itself reject either value.

## Bottom line

**Still NO-GO — both flags stay default OFF, no flip is prepared.** The seed=43 confabulation (the worse of the
two known failure modes) is genuinely fixed via a one-token correction that restores the retry's own documented
scope. The seed=44 regression — the finding's own stated primary reason for NO-GO — persists, is now correctly
attributed to a discourse-WM referent-read fidelity limitation rather than a self-alias control-flow bug, and is
banked as the next lever: an honest, deeper finding, not a license to abandon the anaphora-abstain capability.
