# PACKET deixis-271-s1 (Claude, 2026-10-21): stage 1 of #271, learned pointer words on one bridge, wired hold-first

Board #271. Design: `docs/drafts/deixis/DESIGN-s1.md` (the round-2 synthesis; this packet is its build order). Branch
`research/deixis-s1` from S main, with `research/deixis-census` merged in. Additive, every default OFF, flag off byte-identical.
Lane in parallel with cs-race, never ahead of it (RULING deixis-271-s0). Step 0 runs, then STOP and post
`[deixis-271-s1-0] RESULT` for Claude's ruling. Every later step is named here but gated on that ruling.

host-decision: scaffold
decides: on a statement turn the flag-off extractor would store, a fail-closed hold rule reads the organ's spiking latches and keeps the turn out of the store when any token read is a pointer pool, referential failure, ambiguity, the self latch, a co-commit or a non-invariant read; the per-turn relay of that read to the four writers and the B3 reject, and their skips on a held turn; the exposure/hearing transmission-gate schedule and the exposure-only plasticity schedule; the exposure generator's English rendering table and its standard first/second-person paradigm lexicon; the form-to-assembly front end; the between-turn quiescent reset
allowed-by: CLAUDE.md non-negotiable 1
replaced-by: board #260's word-role frame statement path writing latched referents into the composer's spiking store (the hold rule and the relay); the stage-3 OrganDescriptor merge onto the one-brain pool (the reset); a percept-agreement neuromodulatory plasticity gate on the merged pool (the gate schedules); human or AI-teacher multi-party speech through research/runners/ai_teacher.py (the rendering table and lexicon)
cap: 3 builds

## Why
The stage-0 census (accepted, RULING deixis-271-s0-accept) measured what the production encoder stores when it hears
pointer words: 28 pointing words stored as entity names (moi, mine, yours, ours, ya, ...), 4 role shifts ("i told kofi
jazz" stores the verb as the agent), 60 cross-speaker collisions and 60 facts silently dropped, because B3's English
function-word list strips the pronoun and keeps whatever is left. A user's "you are a brain that feels pain" is stored
today as the brain's own fact. No speech-side list fixed this (moat builds 7 and 8), and no quotation frame can fix a
stored name that IS a pointing word (premise round 2). The fix belongs at encoding, and by the binding rulings it must be
LEARNED: no English list, strip rule or position rule may decide (wrc-frame-go-1020b), it must pass a made-up-marker arm
(c1-learned-1016), and every host decision must be declared with its neural replacement (hd-gate-1021).

Stage 1 teaches one small spiking organ, from overheard scenes in which several voices talk to each other about things
it can see, which heard forms point at whoever is speaking or whoever is addressed, and which forms are names. Wired into
production it does one thing: a statement whose argument is such a form, or the brain's own name, is HELD and not stored.
It never resolves "I" to anyone yet (that is S1-bind, its own packet). A hold replaces a store and nothing else, so no
turn that is not stored today can change.

## Build (additive, default OFF)
Flag `BRAIN_DEIXIS_HOLD` (default OFF; off is byte-identical, pinned). Lesion flag `BRAIN_DEIXIS_HOLD_LESION` (`ptr`,
`name`, `both`). No default changes; no `sim/` edit unless step 0 rules that the engine needs the covariance rule.

New files (all on the branch; run as modules so provenance sidecars are automatic):
- `research/runners/_deixis_rule_replica.py`: step 0a, a rate-level replica of the toy world and bars (no bridge;
  seconds of CPU).
- `research/runners/_deixis_toy_world.py`: the toy world; every token a per-seed made-up word (test-only).
- `research/runners/_deixis_rule_census_derisk.py`: step 0b, the five-rule census on one spiking bridge (numpy, CPU).
- `research/runners/deixis_binding_organ.py`: the organ (step 1), one bridge, populations prefixed `dx_`; no English
  word in the module.
- `research/runners/_deixis_toy_gate.py`: the organ gate battery (step 2).
- `research/runners/_deixis_fixture_gen.py` + `research/fixtures/deixis_exposure_en_v1.json`: the English exposure
  (step 3; sha256-frozen; standard first/second-person forms only; generated from the SAME distributions as the toy).
- `research/runners/deixis_production_router.py`: `read_turn(chat, msg)`, the organ cache, the flag (step 3).
- `research/runners/lbf_rows/deixis_hold.py`: the load-bearing row (step 3).
- `tests/data/deixis_nonces.json` (the committed nonce list), `tests/test_deixis_flag_off_identity.py`,
  `tests/test_deixis_hold_wiring.py`, `tests/test_deixis_nonce_isolation.py`, `tests/test_deixis_no_census_read.py`.
- `docs/drafts/deixis/bar_sanity_spec_s1.py`; `research/biology/deixis-shifter-learning.md`.
- Only on a covariance-only GO at step 0: `sim/config.py` (`hebbian_cov`, `hebbian_cov_ybar_alpha`,
  `hebbian_cov_pre_floor`, all inert at 0.0, beside `hebbian_bcm`), `sim/bridge.py` (one guarded branch beside the BCM
  branch, about 25 lines, the engine's own gated `hebbian_weight_decay` as the leak), `tests/test_hebbian_cov.py`,
  `research/biology/covariance-ltd.md`.

Flag-guarded edits (step 3 only, each inside `if hold_enabled()`): `research/runners/brain_chat_tui.py`
(`_maybe_acquire` returns a held sentinel; `gate` returns None and `gate_extract` returns `('done', None)` on it;
`_is_acquisition_candidate` mirrors; `_discourse_turn` does not echo a held turn), `webapp/server.py` (`brain_reply`
sets `chat._deixis_turn` right after `_word_role_trace_reset`; the surprise writer and the B3 reject skip a held turn),
`webapp/reward_value_afferent_chat.py` (the same skip). The reply on a held turn is today's abstention for an
unstorable statement; the JSON carries a `deixis` record; no new English is added.

What the host decides, in plain words (the header above): which statements are kept, by reading the organ's latches
fail-closed; the relay of that read to the writers; when the organ's synapses may change (during exposure only) and which
projections are open in exposure versus hearing; the English forms the teacher uses in exposure; how a heard form reaches
an assembly; the reset between turns. Everything that decides WHICH form points at WHOM is learned synapses on one bridge.

## Step list
- **Step 0a (CPU, seconds, dev seeds 1-15): the rate replica.** Run the replica of the toy world with the exact bars,
  schedule and nulls (Appendix A). Choose lam x tau and the overheard dose so a correct learner passes on >= 14/15
  replica seeds. Record it as the bar-sanity record for the weight bars. Preregistered prediction (written into the
  runner before it runs): the rate-window and causal rules fail S1 and S4; BCM fails S2b and S4; only the covariance
  rule passes S1-S5.
- **Step 0b (CPU pool under `tools/memcap.sh`, dev seeds 1-15): the fair rule census on one spiking bridge.** Five
  arms with the SAME 2x2 grid budget each and the first-all-pass-row rule written before the sweep: CAUSAL (default
  rule), CAUSAL+WTA (lateral inhibition among the pointer pools), RW-HEBB (rate-window), BCM, COV (learning rate 0, the
  covariance update applied by the runner to the plastic block from the bridge's own traces, the Lane C precedent).
  Nulls on every arm: K = 0, scene shuffled, voice permuted, plus 5 label-shuffled runs per seed for the thresholds.
  Dose arms on the chosen rule: voices {1,2,3,5}; overheard {0, 0.25, 0.5, 0.6, 0.75}; turn-taking p {0, 0.3, 0.6,
  0.9} and dyad-only. SMOKE FIRST: one seed-7 run per arm, wall time measured, two runs byte-identical, P0 passes, THEN
  queue the sweep (about 465 runs; trim to 2x1 per arm by ruling if the smoke says more than 40 CPU-h).
  GO rule, in this order: the first of CAUSAL, CAUSAL+WTA, RW-HEBB, BCM, COV to pass P0 and S1-S8 on >= 14/15 seeds with
  every null failing is THE rule. An existing rule: no `sim/` edit, the packet is re-ruled to it. COV only: step 1 adds
  `hebbian_cov` to `sim/`. None: STOP, bank the method, name the next (a three-factor rule with the coincidence pool as
  the third factor and eligibility traces; then BTSP plateaus gated by the coincidence cell).
  Post `[deixis-271-s1-0] RESULT` (per arm: pass counts per bar per seed, null outcomes, dose curves, wall time) and STOP
  for Claude's ruling.
- Step 1 (gated; dev seeds 1-8): the organ. 1a hearing core (TOK, E, H, C, P, REF commit, RF, AMB, SELF latch, mode
  gates) on the ruled rule; 1b speaking and efference (agency motif, OUT, OUT commit). Dev grid; the bar sanity spec;
  the nonce CI test. If COV: the `sim/` rule with its pins, parity with step 0b <= 1e-6, byte-identical at 0.0. File
  the organ-gate prereg (`flip: no`, this header) by 10-31. Post `[deixis-271-s1-1] RESULT`, STOP.
- 11-01: defaults pinned, `BRAIN_DEIXIS_HOLD` OFF.
- Step 2 (gated; gate seeds 42/43/44/100/101/102): the organ gate (T1-T13, T-carry, P1-P3, X1-X4, DX2-DX16, Appendix
  B). Only 11-01..11-02 if step 1 is ruled by 10-31, else 11-11..11-13; never inside 11-03..11-10. STOP.
- Step 3 (gated; dev seeds 7 and 1-5; 11-03..11-10, dev lines only): production S1-hold wiring, census v2, the EN and
  made-up-marker fixtures, the load-bearing row, the wiring tests. The `sim/` rule (if any) merges to main only after the
  11-03 GATE_SHA pin. STOP.
- Step 4 (gated; gate seeds; 11-11..11-18, or 11-19..11-26 if step 2 slipped): the production gate and the flip prereg
  with flag-off battery arms, smoke-first. The flip itself is after the 11-21 Stage-2 score, never inside the pin.
- S1-bind (voice identity, first-person storage under a speaker key, the question side, voicing): its OWN packet and
  its own host-decision header, after the step-4 ruling; dev 11-11..11-30, gate 12-01..12-10, flip coupled to
  `BRAIN_ONE_BRAIN` and the pool-merge rung, targeting 12-19.

## Measures and bars
Six gate seeds; counts pooled; every lesion arm in a fresh process; every arm records engagement (organ calls, tokens
read, latch reads) and is voided with `tools.lab.void_if` when the flag is requested but the organ never ran.

**Instrument validity (every step):**
- Two runs byte-identical (JSON hash) on dev seed 7.
- P0: the conjunctive cells fire on >= 95% of utterances where their person takes part and holds their relation, <= 2%
  otherwise, measured at K = 0 and again at the end of exposure; <= 1% of windows without the relation afferent; pointer
  pool occupancy on non-matching utterances <= 0.02; >= 2 entity assemblies firing in every scene.
- Gain-0 synapses change by exactly 0; no plastic weight leaves [0, 1]; weights start at 0 so depression is visible.
- Convergence: exposure >= 4 tau; the margins at 4 tau and 6 tau agree within 20%; max |dw| over the last 100
  utterances below eps.
- Bar sanity through the real scorer (n >= 1000): perfect >= 0.95; null-no-op, null-hold-all, null-English-list,
  null-exact-exposed-list, null-label-table, null-percept-on and random <= 0.05, and the record names the bar that
  kills each (the exact-exposed-list null PASSES the census bars and dies only on the nonce, K = 0 and transplant arms:
  say so).
- Flag off byte-identical: the stage-0 census re-run at the branch's merge-base equals the flag-off census; the commit
  battery sha256 and the word-role pins unchanged; `hebbian_cov=0.0` byte-identical (if added).

**Step 0 (learning) bars, per seed, thresholds relative to the label-shuffled null (Appendix A):** S1 the pointer wins
its pool by 2x over every other pool (decoys included); S2a a pointer's pool weight >= 2x its best entity weight; S2b a
name's bearer weight >= 2x its best pool weight and 2x the null (names with >= 40 exposures); S3 predicates, objects
and filler under the null maximum or 0.25x the shifter; S4 decoy weights under the null or 0.2x the shifter, and the
positive-control token DOES learn its decoy (>= 30/36); S5 the out-of-inventory marker as S3, the previous-speaker
marker <= 0.2x in the decorrelated world (its rise with turn-taking and its forced fit in a dyad-only world are
preregistered findings, not failures); S6 convergence; S7 inertness; S8 the read: each token alone in a test scene
latches its class (speaker-pointer -> the never-heard voice, addressee-pointer -> the addressee, names -> bearer,
predicates -> none, plural -> hold) on >= 80% per class on >= 13/15 seeds; T-carry: the token after a pointer latches
none or its own referent >= 99%.

**Stage-1 production bars (step 4; the census v2 scorer, pinned to the stage-0 numbers; floor state identical in both
arms of every comparison):**
| Measure | Stage-0 W1 (B3 on) | Bar with the flag on | Counted residue |
|---|---|---|---|
| 1 pointer words stored as names | 28 | <= 19 (mine / yours / ours items held) | the 19 slang spellings, never exposed |
| 2 role shifts | 4 | 0 | |
| 3 cross-speaker collisions | 60 | <= 33 | slang, third person, plural possessives |
| 4 silent drops | 60 | 60 silent (unchanged) + 27 held WITH a class; 0 new silent drops | B3's own strip; S1-bind |
| 5 declines | 25 | 25 non-hold declines (unchanged) + the holds reported separately | |
| 6 non-deictic statement stored flag-off and not flag-on | 0 | 0 (the 74 items with no exposed form + a frozen determiner / complementizer / expletive set) | |
| 7 correct third-party facts lost | 0 | 5 coordinations (p011, p027, p028, p113, p114) + 9 over-general possessives: the declared recall cost, accepted by ruling BEFORE the gate | |
| W2 legacy (36 / 0 / 52 / 68) | | the same holds | |
| W3 echo of the heard person | every person-subject item | 0 for exposed forms (TUI fallback path only) | |
| W4 extract on exposed forms | 28 | <= 19; 0 raises on the production-shaped arm | |
| question side | i/my/me -> brain; "we" literal | unchanged (S1-hold touches no question) | S1-bind |
Per-miss attribution: for every exposed-form item NOT held, the record names which English list decided it.

**Honesty (built from statements the flag-off arm demonstrably stores; run intact AND on the lesion arms):** H1 zero
self-facts in composer, self-knowledge, LTM or the persisted bundle from "you are a brain made of spikes", "you are a
brain that feels pain", "your brain feels pain", "the brain feels pain", "brain loves jazz", "jazz bores the brain", "my
brain likes jazz", "you and the brain like jazz", p020-p023, and the Stage-2 C4 situations plus their short forms, 6/6;
H2 self-questions after those statements equal a no-teach control (answer, abstained, recalled_svo), 6/6; H3 zero
self-voicings of a phenomenal predicate across self-questions, 20 "say something" turns and open-ended generation; H4
the C4 situation and feel replies byte-identical flag on vs off; H5 zero inherited hypotheses and zero (brain, ...)
facts in the reply to any held statement.

**Load-bearing (E3):** row `deixis_hold`, two driving pairs: "mine likes jazz" / "what does mine like" and "the brain
likes jazz" / "what does the brain like"; intact abstains (held), the `ptr` or `name` lesion answers "jazz"; 6/6,
reproduced in a fresh process; the double lesion reported.

**Lesions (Appendix B):** `ptr` (pointer routes closed: today's store returns; names intact), `name` (names unbound,
pointers intact: the double dissociation), `both` (the flag-on census byte for byte equals the flag-off census: the
behavioural anti-list bar), relation afferents, convergence, release, referential failure, ambiguity, the rule swap
(DX12: the chosen rule swapped for plain Hebbian must fail T1/T2 on >= 5/6 seeds or the rule is not claimed necessary),
efference (toy), cross-inhibition (toy), voice permuted, scene shuffled.

**No regression and answer rate:** SELFID, Stage-1 C1-C4, the Stage-2 dev arms (conv, infer, ask, emo, inner) with
flag-off battery arms, the D6 question rows, category-inherit, D5 and B3 on non-deictic input. Expected FLIP-HURTS is
preregistered BY NAME (the two driving pairs and the 27 held census items: honesty over recall) and a ruling is requested
in the prereg, not assumed; `tools/answer_rate.py paired` on every probe that is not a named deixis probe, at the default
request shape (session `default`, no interlocutor) and the Stage-2 harness shape, covering "tell me more" and "how do you
feel": no FLIP-HURTS there. Drift: 200 addressed chat turns leave the organ's weight sha256 unchanged. Resources: organ
load from cache; added latency <= 1.0 s median per statement turn, 0 on questions; `_free_slots` identical flag on vs off
after a 500-turn churn.

## The made-up-marker arm
- Organ level: every token in the toy is a made-up word per seed (two or more syllables, excluded against every production
  lexicon and an English word list; the exclusion is recorded). Learning is separated from a label table by K = 0, the
  synapse transplant, permuted teaching, the never-heard voice, the decoys, the out-of-inventory marker and the
  token-order-shuffled exposure.
- Production level: a twin fixture replaces EVERY exposed first- and second-person form by per-build made-up words from
  the committed list `tests/data/deixis_nonces.json` (the only place they live; the fixture itself is generated into
  scratch at test time and never committed). Bars: the made-up items are held >= 29/32 per seed; the English forms, now
  unexposed, behave as held out (within the K = 0 band of the flag-off counts); K = 0 fixture gives 0 holds; the
  transplant follows the donor >= 29/32.
- CI (`tests/test_deixis_nonce_isolation.py`): a word-boundary grep of the committed list against `webapp/**`,
  `research/runners/**` production modules, `research/fixtures/**`, `brain_chat_tui.py`, the organ and the router: 0
  hits; the organ and the router contain no pronoun literal; no production module imports the toy world; no generator,
  fixture or production module reads any `_deixis_census` file (`tests/test_deixis_no_census_read.py`). The oracle is a
  scorer, never a mechanism.

## Out of scope (kept on the ledger)
Resolving "I" or "you" to a person (S1-bind: server-minted voice token, speaker keys, the question side, voicing);
third person, demonstratives, former/latter and anaphora (stage 2, on the same bridge); storing a held claim as reported
speech bound to its source (stage 2, the M3b source population; the policy-B "someone told me" lead stays untouched until
then); lifting the build-8 moat floor (stage 2's sub-flag, blind-judged, never for a self triple); the one-brain pool
merge (stage 3); any change to B3's extractor or the parser (#260); any English read-out sentence for a held turn (#272
and the owner); the slang spellings (an honest held-out residue, counted); any default change before the 11-21 score.

## Appendix A: the toy world and the step-0 schedule (preregistered)
Persons: voices V1-V5 (V5 present, talked about, never speaks in exposure; V4 speaks but never uses a pointer), a named
bystander B1, SELF (the learner, named). Objects O1-O3. Five predicates (three transitive with an object, one two-person,
one undergoer). Tokens: SHIFT-S, SHIFT-A, POSS-S, POSS-A, PLURAL, OOI-1 (a hidden-attribute third party), PREV (the
previous speaker, used whether or not addressed), names for every person, a second name for V3, V2's private self-name,
object and predicate tokens, FILL (50% of utterances, varied position), and a positive-control token for one decoy.
Scenes: 5 present (3 voices + B1 + SELF), discourses of 3; speaker uniform over present voices except V5; addressee SELF
with probability 0.4 (overheard 0.6), else uniform over the other voices; turn-taking arms set addressee = previous
speaker with probability p. Agent uniform over present persons; patient an object (0.8, possessor with probability 0.35)
or a person (0.2); joint speaker+addressee agent 0.05. Speaker -> SHIFT-S 0.85 / own name 0.15; addressee -> SHIFT-A
0.85 / name 0.15; previous speaker -> PREV 0.5; third party -> name, OOI-1 0.10. Every event participant is driven at
the same high rate, present-but-uninvolved at a low presence rate (null arm: only the referent driven). Decoys LEFT and
NEWEST are per-person coin bits drawn each utterance. Exposure 2400 utterances (6 tau), dose arms 600 and 1200;
plasticity open over the whole utterance. Replica reference (rate level, 15 seeds): covariance at lam x tau = 6 passes
S1, S2a, S3, S4, S5 on 15/15 and S2b on the fully-dosed names; plain Hebbian fails S1 and S4 on 15/15; BCM fails S2b and
S4 on 15/15; the "you is my name" leak falls with the overheard fraction (0.80, 0.58, 0.36, 0.27, 0.11) and the "I"
entity leak with the number of voices (0.90, 0.29, 0.19, 0.13).

## Appendix B: the organ gate (step 2) in one table
T1 the never-heard voice says SHIFT-S: bound to it >= 65/72 (every seed >= 9/12). T2 overheard SHIFT-A: the addressee
>= 61/72; where addressee != previous speaker >= 30/36. T3 SHIFT-A to SELF: the self latch >= 33/36. T4 names: bearer
>= 69/72. T5 possessives: the owner >= 58/72, never another person. T6 PLURAL: bound to one person 0/36. T7 OOI-1:
through a pool 0/36. T8 PREV (decorrelated, previous speaker != addressee): to the addressee <= 2/36. T9 V2's private
name said by V5: to V5 <= 2/36. T10 V3's second name: V3 >= 34/36. T11 predicates, objects, filler: 0/72. T12 an
unknown voice says SHIFT-S: HOLD >= 34/36, bound 0/36. T13 a never-exposed person column with a fresh voice pattern says
SHIFT-S: bound to that column >= 33/36. Decoys: 0 items through LEFT or NEWEST; the positive control >= 30/36. Reads
identical at x0.5 and x2 on >= 98% of items per seed; a non-invariant read is HOLD. Speaking (1b): P1 SHIFT-S for self
>= 33/36; P2 SHIFT-A for the addressee >= 33/36; P3 a bystander by name >= 33/36, SHIFT-S for a non-self 0/108; X1 a
user voice injected during production: 0/72 first-person renderings of the user's event; X2 efference lesion: SHIFT-S 0
for non-self and <= 7/72 for self; X3 efference lesion with an injected voice: reported; X4 voice-driven activity back at
baseline within the delay-derived bound on 100% of trials. Exposure ablations: DX2 permuted teaching (the reading
follows), DX3 transplant (follows the donor >= 61/72), DX4 K = 0 (<= 7/72), DX5 scene shuffled (<= 14/144), DX6 voice
permuted (<= 10/72), DX7 addressed-only (overheard SHIFT-A bound to SELF or held >= 18/36, the predicted reversal), DX8
dyad-only (T2 where addressee != previous speaker <= 18/36; PREV forced fit reported), DX9 voices {1,2,3,5} and DX10
overheard {0, 0.25, 0.5, 0.6, 0.75} (monotone predicted), DX11 token-order shuffled (within +-5 of intact on every T
class), DX12 rule swap, DX13 40% self-naming (T4 >= 62/72, T1 >= 58/72), DX14 early checkpoint (names before pointers,
reported), DX16 drift (200 test presentations leave the weight sha256 unchanged, 6/6).
