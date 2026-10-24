# PACKET deixis-271-s1 (Claude, 2026-10-21; fix round 2026-10-22): stage 1 of #271, learned pointer words on one bridge, wired hold-first

Board #271. Design: `docs/drafts/deixis/DESIGN-s1.md` (the round-2 synthesis, revised in the fix round; this packet is its
build order; every finding of the final verification is dispositioned in its `FIX_NOTES.md`). Branch `research/deixis-s1`
from S main, with `research/deixis-census` merged in. Additive, every default OFF, flag off byte-identical. Lane in
parallel with cs-race, never ahead of it (RULING deixis-271-s0). Step 0a runs, then 0b, then STOP and post
`[deixis-271-s1-0] RESULT` for Claude's ruling. Every later step is named here but gated on that ruling.

host-decision: scaffold
decides: on a turn a routed writer would act on (a B3 store, the TUI fallback's echo, a false-belief scenario step), a fail-closed hold rule reads the organ's spiking latches and keeps the turn out of the store when any token read is a pointer pool, referential failure, ambiguity, the self latch, a co-commit or a non-invariant read, and holds every such turn when the organ is unavailable, raises or sees a different text; the per-turn relay of that read to the writers and their skips on a held turn; which world afferents production withholds (voice, addressee, givenness: every production pointer hold is caused by the silenced afferent); the toy's discourse-givenness afferent (the entity named in the previous utterance); the exposure/hearing transmission-gate schedule and the exposure-only plasticity schedule; the exposure generator's English rendering table (the R1 paradigm, the R2 reduced and regional forms, the R3 chat-spelling rules); the form-to-assembly front end; the per-token latch reset and the between-turn quiescent reset
allowed-by: CLAUDE.md non-negotiable 1
replaced-by: board #260's word-role frame statement path writing latched referents into the composer's spiking store (the hold rule, the relay and the skips); S1-bind's voice and addressee afferents and #271 stage 2's spiking mention trace on the organ's bridge (the withheld afferents and the toy givenness afferent); the stage-3 OrganDescriptor merge onto the one-brain pool (the resets); a percept-agreement neuromodulatory plasticity gate on the merged pool (the gate schedules); human or AI-teacher multi-party speech through research/runners/ai_teacher.py (the rendering table)
cap: 3 builds

## Why
The stage-0 census (accepted, RULING deixis-271-s0-accept) measured what the production encoder stores when it hears
pointer words: 28 pointing words stored as entity names (moi, mine, yours, ours, ya, ...), 4 role shifts ("i told kofi
jazz" stores the verb as the agent), 60 cross-speaker collisions and 60 facts silently dropped, because B3's English
function-word list strips the pronoun and keeps whatever is left. A user's "you are a brain that feels pain" is stored
today as the brain's own fact, and "the latter married fenna" stores `latter` as a person. No speech-side list fixed this
(moat builds 7 and 8), and no quotation frame can fix a stored name that IS a pointing word (premise round 2). The fix
belongs at encoding, and by the binding rulings it must be LEARNED: no English list, strip rule or position rule may decide
(wrc-frame-go-1020b), it must pass a made-up-marker arm (c1-learned-1016), and every host decision must be declared with
its neural replacement (hd-gate-1021).

Stage 1 teaches one small spiking organ, from overheard scenes in which several voices talk to each other about things it
can see, which heard forms point at whoever is speaking, whoever is addressed, or something the conversation has made
given (pointed at, or just mentioned: he, she, it, they, this, that, the former, the latter), and which forms are names.
Wired into production it does one thing: a statement in which such a form, or the brain's own name, appears is HELD and
not stored. In production every such hold is referential failure: the form's learned route fires its pool, chat renders
no voice, addressee or givenness afferent, so nothing can bind and the turn is held. It resolves nothing to anyone yet
(first and second person: S1-bind; third person and demonstratives: #271 stage 2). A hold replaces a store and nothing
else, so no turn that is not stored today can change.

## Build (additive, default OFF)
Flag `BRAIN_DEIXIS_HOLD` (default OFF; off is byte-identical, pinned). Lesion flag `BRAIN_DEIXIS_HOLD_LESION` (`ptr`,
`name`, `both`). No default changes. With the flag on, an organ that cannot be loaded at startup refuses the flag (a
startup error naming the cache key); a `read_turn` that raises, or reads a different text from the one the writer got, is
a HOLD (class `unavailable` / `mismatch`), counted per chat and in the reply JSON.

**Weight scale (the engine's own units; design 2.6).** A spike adds `w x propagation_strength` (0.05, `sim/config.py:871`;
`sim/bridge.py:9258-9261`) to the target's conductance, which decays with tau 5 ms (`:2910`; `config.py:394`) and drives
`g_e (E_e - v)` with E_e = 0 mV (`sim/kernels.py:1115`). A 12-neuron token assembly at 1500 pA (the word-role front end,
`word_role_parser_organ.py:127`) gives `g_e ~ 0.052 x 12 x w` nS; a target needs about 150 pA to fire inside a 40-ms
token, so a route fires at w ~ 4-8. Measured on the engine (10 seeds, NT = 12): w = 4 gives at most one target spike, w = 8
fires every target neuron. So: token assemblies of 12, plastic weights in [0, 20] (the word-role organ's W_MAX), w_init =
0, `theta_fire = 8`, `theta_silent = 4`. The packet's former [0, 1] range could never fire a pool. The step-0b smoke
re-measures both thresholds on the organ's own bridge before any census line runs, and the saturated-route check
(TOK -> P and TOK -> E set by hand to 20 fire P and E, and with H on fire C; at 4 and at 0 they stay silent) must pass.

New files (all on the branch; run as modules so provenance sidecars are automatic):
- `research/runners/_deixis_rule_replica.py`: step 0a, the rate replica of the toy world, the six arms, the bars, the
  tuning procedure and the STOP rule (no bridge; minutes of CPU; from the design's `scratch/fix/replica3.py`).
- `research/runners/_deixis_wscale_curve.py`: the transfer curve and the saturated-route check on the organ's bridge.
- `research/runners/_deixis_toy_world.py`: the toy world; every token a per-seed made-up word (test-only).
- `research/runners/_deixis_rule_census_derisk.py`: step 0b, the six-rule census on one spiking bridge (numpy, CPU).
- `research/runners/deixis_binding_organ.py`: the organ (step 1), one bridge, populations prefixed `dx_`; no English word
  in the module.
- `research/runners/_deixis_toy_gate.py`: the organ gate battery (step 2).
- `research/runners/_deixis_fixture_gen.py` + `research/fixtures/deixis_exposure_en_v1.json` (step 3; sha256-frozen; the
  exposed forms come only from rules: R1 the reference-grammar pronoun paradigm of all persons plus the demonstratives and
  the former / the latter; R2 the reduced and regional variants 'em, ya, ye, yer, yous, youse, thee, thou, thy, thine; R3
  uniform chat-spelling rules: apostrophe deletion, you -> u, your / you're -> ur, yourself -> urself, each at 0.2 of the
  renderings it applies to); the held-out-spelling twin `deixis_exposure_en_ho_v1.json` (R3's u/ur/urself withheld).
- `research/runners/deixis_production_router.py`: `read_turn(chat, msg)`, the organ cache keyed by (fixture sha, seed,
  params, the blob shas of `sim/bridge.py`, `sim/config.py`, `sim/kernels.py` and the organ module), the flag (step 3).
- `research/runners/lbf_rows/deixis_hold.py`: the load-bearing row (step 3).
- `tests/data/deixis_nonces.json` (the committed nonce list), `tests/test_deixis_flag_off_identity.py`,
  `tests/test_deixis_hold_wiring.py`, `tests/test_deixis_nonce_isolation.py`, `tests/test_deixis_no_census_read.py`,
  `tests/test_deixis_fixture_rules.py` (every exposed form from R1 + R2 + R3, none by hand), `tests/test_deixis_fail_closed.py`.
- Census v2 on `research/deixis-census`: oracle v2 beside v1 (p009 'latter', p010 'that' as ANTECEDENT; the former/latter
  and demonstrative items), the W5 and W6 writers, the cross-speaker B3 measure (design 8.2).
- `docs/drafts/deixis/bar_sanity_spec_s1.py`; `research/biology/deixis-shifter-learning.md`.
- Only on a COV or COV-PRE GO at step 0: `sim/config.py` (`hebbian_cov`, `hebbian_cov_leak`, `hebbian_cov_ybar_alpha`,
  `hebbian_cov_pre_floor`, `hebbian_cov_ybar_gate`, all inert at their defaults, beside `hebbian_bcm`), `sim/bridge.py`
  (one guarded branch beside the BCM branch, about 30 lines; ybar moves only while a named plasticity gate is open),
  `tests/test_hebbian_cov.py`, `research/biology/covariance-ltd.md`.

Flag-guarded edits (step 3 only, each inside `if hold_enabled()`): `research/runners/brain_chat_tui.py`
(`_maybe_acquire` returns a held sentinel; `gate` returns None and `gate_extract` returns `('done', None)` on it;
`_is_acquisition_candidate` mirrors; `_discourse_turn` does not write or echo a held turn; the TUI local fallback calls
`read_turn` itself), `webapp/server.py` (`brain_reply` sets `chat._deixis_turn` right after `_word_role_trace_reset`; the
surprise writer and the false-belief scene writer skip a held turn; the B3 contradiction reject is KEPT, because the hold
applies only to a turn the flag-off path would store), `webapp/reward_value_afferent_chat.py` (the same skip),
`research/runners/rich_answer_composer.py` (a held outcome short-circuits `_direct_fact` before its DIRECT-LTM-TOPIC
fallback, so the rich default path abstains). The reply on a held turn is the abstention (`abstained: True`, no facts);
the JSON carries a `deixis` record; no new English is added.

What the host decides, in plain words (the header above): which statements are kept, by reading the organ's latches
fail-closed (an unavailable organ holds); which turns the organ reads and the relay of that read to the writers; which
world afferents production does not render (voice, addressee, givenness: every production hold is caused by that
silence, not by learned addressee meaning); the toy's givenness afferent (the entity named in the previous utterance);
when the organ's synapses may change (exposure only) and which projections are open in exposure versus hearing; the
English forms the teacher uses in exposure (the R1-R3 rules); how a heard form reaches an assembly; the per-token and
between-turn resets. Everything that decides WHICH form points at WHOM is learned synapses on one bridge.

## Step list
- **Step 0a (10-22; CPU minutes, no bridge; dev seeds 1-15, rates on 1-60): the bar-sanity record.** The rate replica with
  the exact bars, schedule and nulls (Appendix A) runs SIX arms through ONE tuning procedure, written into the runner
  before it runs: the same grid for every arm (lam x tau in {1, 2, 3, 4, 6, 9, 12, 18}); each arm's centre is the cell with
  the most seeds passing every bar, ties to the most bar-seed passes, then to the larger median weakest must-fire route;
  each arm's spiking cells are its centre and the next value up. No threshold comes from an arm: S0 is the engine's
  transfer curve, the contrasts are round 1's. Preregistered prediction (in the runner): CAUSAL, CAUSAL+WTA and RW-HEBB
  fail S1 and S0; BCM fails S0 and S4; COV (uniform leak) fails S0 at every cell; COV-PRE passes. **STOP rule:** the record
  must show some arm with a 60-seed pass rate >= 0.9 (where P(>= 13/15) >= 0.8), every null failing, and K = 0 failing; if
  any of the three fails, the bars are wrong, 0b is NOT queued, and the failing bar is re-derived from mechanism (never
  from an arm's margins) and re-ruled. The fix-round draft (Appendix A) meets all three.
- **Step 0b (10-23..10-26; CPU pool under `tools/memcap.sh`, dev seeds 1-15): the fair rule census on one spiking bridge.**
  The TOK / E / H / C / P circuit with the mode gates, no commit stage. Six arms: CAUSAL (default rule), CAUSAL+WTA
  (lateral inhibition among the pointer pools), RW-HEBB (rate-window), BCM, COV (learning rate 0, the covariance update
  applied by the runner to the plastic block from the bridge's own traces, the Lane C precedent, with the engine decay as
  the leak), COV-PRE (the same update with the presynaptically gated leak, engine decay 0). SMOKE FIRST on dev seed 7: one
  run per arm, the transfer curve and the saturated-route check on the organ's bridge, the unit conversions (steps per
  occurrence above the presynaptic floor, the fraction of steps the Hebbian block runs, `if _prev_any and _fired_any`,
  `sim/bridge.py:10236`, and a driven pool neuron's trace value) committed so every arm's change per occurrence matches the
  replica's, P0, wall time, two runs byte-identical; THEN queue the census: 6 arms x 2 cells x 15 seeds = 180 runs, and
  the nulls (label-shuffled stream, scene shuffled, voice permuted; 3 x 15 = 45 runs) only on an arm that passes the
  weight bars; K = 0 is scored on the initial weights. About 235 runs of 4800 utterances (about 0.96 M steps each, 5-27
  min), 20-105 CPU-h. If they do not fit the pool by 10-26, the rival arms drop to their centre cell only; if that still
  does not fit, the RESULT date moves and the slip rule below applies. The dose arms run at step 1 on the chosen rule.
  **Audit, applied to EVERY arm that passes before the pass counts:** P0; the learned routes' transmission gate read back
  closed on every exposure step; the arm's own nulls fail; its spiking weights reproduce its replica pattern (every scored
  route on the same side of its bar on >= 13/15 seeds, Spearman >= 0.9 per seed); the saturated-route check on its bridge.
  **GO rule:** THE rule is the first arm, in the order CAUSAL, CAUSAL+WTA, RW-HEBB, BCM, COV, COV-PRE, that passes P0, the
  saturated-route check and S0-S7 all on the same seed on >= 13/15 seeds at one of its two cells, whose nulls each pass on
  <= 1/15, and whose audit holds. An existing rule: no `sim/` edit, the packet is re-ruled to it. COV or COV-PRE: step 1
  adds it to `sim/`. None: STOP, bank the method, name the next (a three-factor rule with the coincidence pool as the
  third factor and eligibility traces; then BTSP plateaus gated by the coincidence cell). Why 13/15 and not 14/15: the
  count is binomial in the arm's per-seed pass rate q; at q = 0.9 P(>= 13/15) = 0.82 but P(>= 14/15) = 0.55, so 14/15
  fails a correct learner that loses one seed in ten to spiking noise about half the time; at q = 0.5 P(>= 13/15) =
  0.004 and at q = 0.7 it is 0.13. The replica separates the arms far more: COV-PRE 59/60 seeds at its centre (95% lower
  bound 0.92, P(>= 13/15) = 0.90 there), every rival 0/60 at its centre and 0/15 at all eight cells.
  Post `[deixis-271-s1-0] RESULT` (per arm and cell: pass counts per bar per seed, the audit items, null outcomes, wall
  time, CPU-h, the `tools/where_ran.py` host summary) and STOP for Claude's ruling.
- Step 1 (10-26..10-31; gated; dev seeds 1-8): the organ. 1a hearing core (TOK, E, H, C, P, REF commit, RF, AMB, SELF
  latch, mode gates) on the ruled rule, with S8 (each token alone latches its class, >= 80% per class on >= 13/15 dev
  seeds) and T-carry, which left step 0 because they need the commit stage; the dose arms (voices, overheard, turn-taking,
  dyad-only); 1b speaking and efference (agency motif, OUT, OUT commit). The bar sanity spec; the nonce CI test. If COV or
  COV-PRE: the `sim/` rule with its pins (flag-off byte-identical, a sign test, ybar frozen while closed, parity with the
  step-0b arm on a replayed raster <= 1e-6). File the organ-gate prereg (`flip: no`, this header) by 10-31. Post
  `[deixis-271-s1-1] RESULT`, STOP.
- 11-01: defaults pinned, `BRAIN_DEIXIS_HOLD` OFF.
- Step 2 (gated; gate seeds 42/43/44/100/101/102): the organ gate (Appendix B). 11-01..11-02 only if step 1 is ruled by
  10-31, else 11-11..11-13; never inside 11-03..11-10. STOP.
- Step 3 (gated on the step-2 verdict; dev seeds 7 and 1-5; 11-03..11-10, or 11-14..11-20 if step 2 slipped; dev lines
  only): production S1-hold wiring, census v2, the EN, made-up-marker and EN-HO fixtures, the load-bearing row, the wiring
  and fail-closed tests. If the organ gate fails, step 3 does not start and no English fixture is built. The `sim/` rule
  (if any) merges to main only after the 11-03 GATE_SHA pin. STOP.
- Step 4 (gated; gate seeds; 11-11..11-18, or 11-21..11-28 if step 2 slipped): the production gate and the flip prereg
  with flag-off battery arms, smoke-first. The flip target is 11-24..11-28 (12-01..12-05 if slipped), after the 11-21
  Stage-2 score, never inside the pin.
- S1-bind (voice identity, first-person storage under a speaker key, the question side): its OWN packet and host-decision
  header, development only after the step-4 ruling (to 11-30), gate 12-01..12-10, flip coupled to `BRAIN_ONE_BRAIN` and
  the pool-merge rung, targeting 12-19. No voicing in S1-bind: production has no efference copy of the brain's own reply
  until #272's reply-side spiking word-commit drives it; that packet carries the production X1 bar and its cd lesion.
- #271 stage 2 (resolution: the mention trace on this bridge, the two-candidate order, quoted speech, M3b): packet 11-28,
  dev 12-01..12-12, gate 12-15..12-18.
- Queue placement: requested from Claude as a NEEDS line (parallel-builds-1021b's FRONT=1 covers only its own lines);
  until ruled, this lane queues at normal priority.

## Measures and bars
Six gate seeds; counts pooled; every lesion arm in a fresh process. Every arm records engagement (organ calls, tokens
read, latch reads) and is voided with `tools.lab.void_if` when the flag is requested but the organ never ran. Every
lesion, permutation and null arm also records, and is UNDEFINED (`tools.lab.undefined_if_empty`) without: the
transmission and plasticity gate values read back from the bridge; per token, the P, RF and E spike counts showing zero
current through the lesioned route; the K, permutation map and shuffle seed used, and the exposure stream's sha256. Every
RESULT quotes the `tools/where_ran.py` host summary; census v2 and every production arm record the source (`tiny-demo
+LTM`), `corpus_present`, the LTM bundle sha256, the organ-cache key and hit or miss, and the host class; a flag-on vs
flag-off comparison across host classes is UNDEFINED.

**Instrument validity (every step):**
- Two runs byte-identical (JSON hash) on dev seed 7.
- P0: the conjunctive cells fire on >= 95% of utterances where their entity takes part and holds their relation, <= 2%
  otherwise, measured at K = 0 and at the end of exposure; <= 1% of windows without the relation afferent; pointer pool
  occupancy on non-matching utterances <= 0.02; >= 2 entity assemblies firing in every scene; the learned routes'
  transmission closed throughout exposure (read back).
- S7: gain-0 synapses change by exactly 0; no plastic weight leaves [0, 20]; weights start at 0 so depression is visible.
- The saturated-route check (above), at the smoke and after the census.
- Bar sanity through the real scorer (n >= 1000): perfect >= 0.95; null-no-op, null-hold-all, null-English-list,
  null-exact-exposed-list, null-label-table, null-percept-on and random <= 0.05, and the record names the bar that kills
  each (the exact-exposed-list null PASSES the census bars and dies only on the nonce, K = 0 and transplant arms: say so).
- Flag off byte-identical: the stage-0 census re-run at the branch's merge-base equals the flag-off census; the commit
  battery sha256 and the word-role pins unchanged; `hebbian_cov=0.0` byte-identical (if added).

**Step 0 (learning) bars, per seed, engine units (design 3.5; no label-shuffled threshold):** S0 every route that must act
ends >= 8 (each pointer form's target pool, the GIVEN forms included; the self's name; every name to its bearer) and
every route whose firing could cause one wrong latch ends <= 4 (pool routes of predicates, objects, filler, names, OOI-1
and PREV except PREV -> ADR and PREV -> GIV; non-target pools of single-target pointer forms; entity routes of
predicates and filler; non-bearer routes of names); the entity routes of pointer forms are reported (firing one adds a
second latch, which HOLDS). S1 each pointer form wins its target pool by 2x over every other pool, decoys included. S2a a
non-possessive single-target pointer form's target pool >= 2x its best entity route. S2b a name's bearer route >= 2x its best pool route and >= 2x its
best other entity route. S3 predicates, objects and filler under max(4, 0.25x the smaller shifter target). S4 decoy
routes under max(4, 0.2x it), and no token with a pool route >= 0.5x it has a decoy as its top pool (the positive-control
token learns its decoy, >= 30/36 at the organ gate). S5 the out-of-inventory marker as S3; the previous-speaker marker
onto the addressee pool <= 0.2x in the decorrelated world (its rise with turn-taking and its forced fit in a dyad-only
world are preregistered findings, not failures). S6 every must-fire route non-decreasing between 2/3 of exposure and its
end, and the routes of forms heard >= 450 times within 20% between the two. S7 above.

**Stage-1 production bars (step 4; the census v2 scorer, pinned to the stage-0 numbers; floor state identical in both
arms of every comparison).** Projection on the recorded census (design 8.2): 59 of the 126 items move from stored to
HELD. These are fixture-recovery checks; the learning claim rests on the made-up-marker arms.

| Measure | Stage-0 W1 (B3 on) | Bar with the flag on | Counted residue (method, stage, date) |
|---|---|---|---|
| 1 pointer words stored as names (oracle v1 SPEAKER/ADDRESSEE; with its ANTECEDENT tokens; oracle v2) | 28 / 35 / 36 | <= 3 each | `moi` x3: produced by no exposure rule; teacher correction then learning from use, stage 3, method packet by 12-05 |
| 2 role shifts | 4 | 0 (p020-p023 held) | |
| 3 cross-speaker collisions | 60 | <= 3 | `moi` x3 |
| 4 silent drops | 60 | 60 silent (unchanged) + 57 held WITH a class; 0 new silent drops | B3's own strip (S1-bind 12-01..12-10; stage 2 12-15..12-18) |
| 5 W1 declines (renamed: declines whose first token is SPEAKER, ADDRESSEE or NONE) | 25 | 25 non-hold declines (unchanged) + the holds reported separately; 0 holds on non-deictic items | as 4 |
| 6 non-deictic statement stored flag-off and not flag-on | 0 | 0 (p016, p035, p036, p109 and a frozen non-pointer set B3 stores) | |
| 6b homographs ('a dog has a brain', 'kofi owns a mine', 'kofi and fenna were friends') | 0 | reported with a preregistered cost; ruling requested before the gate | #260 (role context) |
| 7 correct facts lost | 0 | 5 coordinations (p011, p027, p028, p113, p114) + 16 over-general possessives + 1 over-general demonstrative (p010): the declared recall cost, accepted by ruling BEFORE the gate | S1-bind and stage 2 store the bound versions |
| former / latter / demonstrative arm (oracle v2 single- and two-turn items) | stored | HELD 100% on 6/6; its made-up-marker twin >= 29/32 per seed | |
| W2 legacy (36 / 0 / 52 / 68) | | the same holds | |
| W3 echo (TUI fallback only) | every person-subject item | 0 for exposed forms | `moi` |
| W4 extract on exposed forms | 28 | <= 3; 0 raises on the production-shaped arm | `moi` |
| W5 false-belief scenarios with a pointer agent | measured at the merge-base | 0 started or advanced with an exposed form as the agent | Stage-3 C3 (12-19) |
| W6 affective-ToM English self/other split | measured at the merge-base | unchanged (reported) | S1-bind |
| cross-speaker B3 (`BRAIN_ONE_BRAIN=1`) | measured at the merge-base | no increase | S1-bind |
| question side | i/my/me -> brain; "we" literal | unchanged (S1-hold touches no question) | S1-bind; it/its stage 2 |
| held-out spelling arm (EN-HO) | | those items stored exactly as flag-off (no generalization to an unheard spelling claimed) | stage 3 |
Per-miss attribution: for every exposed-form item NOT held, the record names which English list decided it.

**Honesty (built only from statements the flag-off arm is shown to store; run intact AND on the lesion arms):** H1 zero
self-facts in composer, self-knowledge, LTM or the persisted bundle from "you are a brain made of spikes", "you are a
brain that feels pain", "your brain feels pain", "you are a brain that loves jazz", "the brain feels pain", "the brain
feels joy", "brain likes jazz", "jazz bores the brain", "my brain likes jazz", "you and the brain like jazz", p020-p023,
and the told-scene items "a wolf bites your arm" and "the storm scares your cat", 6/6; H2 self-questions after those
statements equal a no-teach control (answer, abstained, recalled_svo), 6/6; H3 zero self-voicings of a phenomenal
predicate across self-questions, 20 "say something" turns and open-ended generation; H4 the C4 situation and feel replies
byte-identical flag on vs off; H5 zero inherited hypotheses and zero (brain, ...) facts in the reply to any held
statement, on the rich default shape and on rich=False; H5-legacy: from a store seeded with flag-off-era deictic names,
every legacy entry still voiced is counted and none is added (quarantine: S1-bind); H6 quoted, reported and identity
claims ('kofi: "i hate jazz"', 'kofi: "you hate jazz"', 'this is kofi, i like jazz') HELD 100% on 6/6. H1-H6 also run
under `BRAIN_ONE_BRAIN=1` with two sessions sharing one store: 0 leaks of A's held statements into B's replies.

**Load-bearing (E3):** row `deixis_hold`, two driving pairs on the rich default shape: "mine likes jazz" / "what does mine
like" and "the brain likes jazz" / "what does the brain like"; intact abstains (held), the `ptr` or `name` lesion answers
"jazz"; 6/6, reproduced in a fresh process; the double lesion reported.

**Lesions (Appendix B):** `ptr` (pointer routes closed: today's store returns; names intact), `name` (names unbound,
pointers intact: the double dissociation), `both` (the flag-on census byte for byte equals the flag-off census, counted
only with the engagement record showing both routes closed: the behavioural anti-list bar), the givenness afferent,
the relation afferents, convergence, release, referential failure, ambiguity, the rule swap (DX12: the chosen rule
swapped for plain Hebbian must fail T1/T2 on >= 5/6 seeds or the rule is not claimed necessary), efference (toy),
cross-inhibition (toy), voice permuted, scene shuffled.

**No regression and answer rate:** SELFID, Stage-1 C1-C4, the Stage-2 dev arms (conv, infer, ask, emo, inner) with
flag-off battery arms, the D6 question rows, category-inherit, D5 and B3 on non-deictic input. Preregistered FLIP-HURTS,
by name, with a ruling requested in the prereg, never assumed: the two driving pairs and the 59 held census items
(honesty over recall). Everywhere else the flag must not cost an answer. The battery: the `deixis_hold` row writes one
arm file per probe, the probe being the file's LAST answered turn (the only turn `tools/answer_rate.py` reads), under
`research/findings/raw/_load_bearing/_shards/<tag>/s<seed>/deixis-hold-ar/`, each named `intact_<probe>.json`
(`paired` reads only `^intact` files; a directory of its own, so the row's named driving pairs, which abstain by design,
are not in it), at two request shapes (session `default` with no interlocutor; the Stage-2 harness shape), one tag pair
per shape. Probes: "tell me more" and "how do you feel", each AFTER a held statement; a
question about a stored non-deictic fact after a held statement; the Stage-2 dev arm probes; a statement with no exposed
form and its question. Tags `deixis-s1-on-<shape>-<sha>` and `deixis-s1-off-<shape>-<sha>` (the same cells with the flag off, run
normally, not an alloff cell). Command, per shape: `python tools/answer_rate.py paired --on-tag deixis-s1-on-<shape>-<sha>
--off-tag deixis-s1-off-<shape>-<sha> --faculty deixis-hold-ar --json > research/findings/smoke/deixis_s1_answer_rate_<shape>.json`;
the per-seed report is quoted in the RESULT; any FLIP-HURTS-ANSWERS outside the named list is a NEEDS line, not a pass.
Drift: 200 addressed chat turns leave the organ's weight sha256 unchanged. Resources: organ load from cache; added
latency <= 1.0 s median per statement turn, 0 on questions; `_free_slots` identical flag on vs off after a 500-turn churn.

## The made-up-marker arm
- Organ level: every token in the toy is a made-up word per seed (two or more syllables, excluded against every
  production lexicon and an English word list; the exclusion is recorded). Learning is separated from a label table by
  K = 0, the synapse transplant, permuted teaching, the never-heard voice, the decoys, the out-of-inventory marker and the
  token-order-shuffled exposure.
- Production level: a twin fixture replaces EVERY exposed form (R1, R2 and R3; all three relations) by per-build made-up
  words from the committed list `tests/data/deixis_nonces.json` (the only place they live; the fixture itself is generated
  into scratch at test time and never committed). Bar: the made-up items are held >= 29/32 per seed. The template-killing
  controls run on BOTH readings, the English census and the nonce census: K = 0 fixture gives 0 holds on each; the
  transplant (the nonce-taught organ's TOK -> P and TOK -> E blocks into a fresh organ) makes the nonce holds follow the
  donor >= 29/32 and the English holds follow the donor's absent English routes; the both-route lesion makes each flag-on
  census equal its flag-off census. "The English forms, now unexposed, behave as held out" is dropped as evidence (an
  unexposed form maps to an untrained spare, so the front end alone meets it); it is reported only.
- CI (`tests/test_deixis_nonce_isolation.py`): a word-boundary grep of the committed list against `webapp/**`,
  `research/runners/**` production modules, `research/fixtures/**`, `brain_chat_tui.py`, the organ and the router: 0
  hits; the organ and the router contain no pronoun literal; no production module imports the toy world; no generator,
  fixture or production module reads any `_deixis_census` file (`tests/test_deixis_no_census_read.py`); every exposed
  form is produced by the R1-R3 rules (`tests/test_deixis_fixture_rules.py`). The oracle is a scorer, never a mechanism.

## Out of scope (kept on the ledger, each with its stage and date)
Resolving "I" or "you" to a person (S1-bind, 12-01..12-10: server-minted voice token, speaker keys, the question side,
the legacy quarantine); resolving third person, demonstratives and former/latter to their antecedent (#271 stage 2,
12-15..12-18; stage 1 HOLDS them); storing a held claim as reported speech bound to its source (stage 2, the M3b source
population; the policy-B "someone told me" lead stays untouched until then); voicing "I" and "you" in chat (#272's
reply-side word-commit, then S1-bind or stage 2); lifting the build-8 moat floor (stage 2's sub-flag, blind-judged, never
for a self triple); the one-brain pool merge (stage 3); any change to B3's extractor, its strip or the parser (#260);
any English read-out sentence for a held turn (#272 and the owner); spellings no exposure rule produces (`moi`: teacher
correction through `ai_teacher.py`, then learning from use under the stage-3 conversational plasticity gate; method
packet by 12-05); any default change before the 11-21 score.

## Appendix A: the toy world and the step-0 schedule (preregistered)
Persons: voices V1-V5 (V5 present, talked about, never speaks in exposure; V4 speaks but never uses a pointer), a named
bystander B1, SELF (the learner, named). Objects O1-O3. Five predicates (three transitive with an object, one two-person,
one undergoer). Tokens: SHIFT-S, SHIFT-A, POSS-S, POSS-A, PLURAL, OOI-1 (a hidden-attribute third party), PREV (the
previous speaker, used whether or not addressed), the GIVEN family (ANA-1: an entity named in the previous utterance;
DEM-1: an entity the speaker points at, alone or as a determiner; POSS-G: a given possessor; ORD-1 / ORD-2: the first /
second of two entities named in the previous utterance), names for every person, a second name for V3, V2's private
self-name, object and predicate tokens, FILL (50% of utterances, varied position), a positive-control token for one
decoy, and a homograph arm (N_SELF's form also an object's token in 20% of its mentions). Scenes: 5 present (3 voices + B1
+ SELF), discourses of 3; speaker uniform over present voices except V5; addressee SELF with probability 0.4 (overheard
0.6), else uniform over the other voices; turn-taking arms set addressee = previous speaker with probability p. Agent
uniform over present persons; patient an object (0.8, possessor with probability 0.35) or a person (0.2); joint
speaker+addressee agent 0.10. The speaker points at one present entity other than speaker and addressee with probability
0.30; the entities named in the previous utterance are given. Speaker -> SHIFT-S 0.85 / own name 0.15; addressee ->
SHIFT-A 0.85 / name 0.15; previous speaker -> PREV 0.5; pointed entity -> DEM-1 0.8; previously named entity -> ORD-1 /
ORD-2 0.6 when two were named, else ANA-1 0.7; given possessor -> POSS-G 0.8; third party -> name, OOI-1 0.10. Every
event participant is driven at the same high rate, present-but-uninvolved at a low presence rate (null arm: only the
referent driven). Decoys LEFT and NEWEST are per-person coin bits drawn each utterance. Exposure 4800 utterances (POSS-G,
the rarest scored form, is heard >= 83 times on every dev seed); plasticity open over the whole utterance; the learned
routes' transmission closed throughout exposure.

Replica reference (fix-round draft of the step-0a record, `scratch/fix/` of the design; rate level, not spiking; seeds
passing every bar out of 15, at each arm's own centre; rates on seeds 1-60): COV-PRE (lam x tau = 6) 15/15, 59/60, window
4-9 at >= 14/15; COV (6) 0/15 and 0/60, failing only S0 (its weakest must-fire routes, the rarest forms, 0.05-0.14 of
W_MAX, sit below its strongest wrong routes, 0.09-0.12, at every cell, because a uniform leak scales each weight by the
form's frequency); RW-HEBB and CAUSAL+WTA (3) 0/15 and 0/60, failing S0, S1, S2a, S2b and S4; BCM (1) 0/15 and 0/60,
failing S0 and S4. Every rival fails at all eight cells. Nulls on COV-PRE: label-shuffled, scene shuffled, voice permuted
and K = 0 each 0/15. The judged round-2 bars gave COV 6/15 (S2b's class-maximum null), which made its GO rule unreachable.
Dose (COV-PRE at 6, medians): the "you is my name" route falls with the overheard fraction (0.57, 0.43, 0.26, 0.18, 0.08
of W_MAX at 0, 0.25, 0.5, 0.6, 0.75; above 0.20 at 0 and 0.25, the predicted addressed-only reversal); the "I" route's
best entity with the number of voices (0.44, 0.18, 0.13, 0.09 at 1, 2, 3, 5); PREV onto the addressee pool rises with
turn-taking (0.11, 0.17, 0.30, 0.41 of SHIFT-A's route at p = 0, 0.3, 0.6, 0.9).

## Appendix B: the organ gate (step 2) in one table
T1 the never-heard voice says SHIFT-S: bound to it >= 65/72 (every seed >= 9/12). T2 overheard SHIFT-A: the addressee
>= 61/72; where addressee != previous speaker >= 30/36. T3 SHIFT-A to SELF: the self latch >= 33/36. T4 names: bearer
>= 69/72. T5 possessives: the owner >= 58/72, never another person. T6 PLURAL: bound to one person 0/36. T7 OOI-1:
through a pool 0/36. T8 PREV (decorrelated, previous speaker != addressee): to the addressee <= 2/36. T9 V2's private
name said by V5: to V5 <= 2/36. T10 V3's second name: V3 >= 34/36. T11 predicates, objects, filler: 0/72. T12 an
unknown voice says SHIFT-S: HOLD >= 34/36, bound 0/36. T13 a never-exposed person column with a fresh voice pattern says
SHIFT-S: bound to that column >= 33/36. T14 ANA-1 / DEM-1 / POSS-G with one given entity: bound to it >= 61/72, never to
another person. T15 two given entities (ORD-1, ORD-2 or ANA-1): HELD (AMB or co-commit) 36/36, bound to a person 0/36.
T16 a GIVEN form with no givenness afferent (the production condition): HELD (RF_GIV) 36/36. T-carry: the token after a
shifter latches none or its own referent >= 71/72. Homograph (N_SELF's form as an object token): object-use false holds
reported with a preregistered cost. Co-commit on single-referent tokens: reported per class, a miss in T1-T5 and T14.
Decoys: 0 items through LEFT or NEWEST; the positive control >= 30/36. Recruitment: 0 entity slots recruited by any
shifter. Reads identical at x0.5 and x2 on >= 98% of items per seed; a non-invariant read is HOLD. Speaking (1b): P1
SHIFT-S for self >= 33/36; P2 SHIFT-A for the addressee >= 33/36; P3 a bystander by name >= 33/36, SHIFT-S for a
non-self 0/108; X1 a user voice injected during production: 0/72 first-person renderings of the user's event; X2
efference lesion: SHIFT-S 0 for non-self and <= 7/72 for self; X3 efference lesion with an injected voice: reported; X4
voice-driven activity back at baseline within the delay-derived bound on 100% of trials. Exposure ablations: DX2 permuted
teaching (the reading follows), DX3 transplant (follows the donor >= 61/72), DX4 K = 0 (<= 7/72), DX5 scene shuffled
(<= 14/144), DX6 voice permuted (<= 10/72), DX7 addressed-only (overheard SHIFT-A bound to SELF or held >= 18/36, the
predicted reversal), DX8 dyad-only (T2 where addressee != previous speaker <= 18/36; PREV forced fit reported), DX9
voices {1,2,3,5} and DX10 overheard {0, 0.25, 0.5, 0.6, 0.75} (monotone predicted), DX11 token-order shuffled (within
+-5 of intact on every T class), DX12 rule swap, DX13 40% self-naming (T4 >= 62/72, T1 >= 58/72), DX14 early checkpoint
(names before pointers, reported), DX16 drift (200 test presentations leave the weight sha256 unchanged, 6/6). Lesion
`dx_giv`: T14 bound <= 4/72 and held >= 68/72.
