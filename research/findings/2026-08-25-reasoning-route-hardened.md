---
type: finding
status: live
date: 2026-08-25
mechanism: reasoning-route-chain-routing-moat-hardening
lane: integration
seeds: [42]
artifacts:
  - research/findings/raw/_reasoning_frontier_verify/battery_cupy1.json
  - research/findings/raw/_reasoning_frontier_verify/battery_cupy1.jsonl
runner: research/findings/raw/_reasoning_frontier_verify/hardened_battery_driver.py
---

# Reasoning-route (chain-routing) moat hardening: reqs #1/#4/#5 closed, #6 confirmed-not-redone, verified live against the real `/api/brain-chat`

**Branch:** `research/reasoning-frontier-hardened` (from `origin/research/reasoning-frontier`, NOT `main`).
**Spec:** [`2026-08-25-reasoning-route-moat-audit-hardening-spec.md`](2026-08-25-reasoning-route-moat-audit-hardening-spec.md) (10 requirements; on `main`, not this branch — read via `git show origin/main:...`).
**De-risk it supersedes-as-load-bearing:** [`2026-08-25-fhrr-decode-rate-at-scale.md`](2026-08-25-fhrr-decode-rate-at-scale.md) measured the deployed D=128/15k-fact FHRR cue-role false-hop rate at ~0 and wrong-patient at <!--derived: quoted from the cited finding's own headline, not re-measured here-->0.0067%<!--derived--> — the per-hop confidence floor (audit req #2) is therefore optional defense-in-depth, NOT the load-bearing safety mechanism, and was correctly NOT built in this pass per the task's explicit direction.

## What changed (three commits on this branch)

1. `research/runners/compositional_chain_route.py` — `_distinct_patients`/`_hop` now scan the composer's `.kb`
   (generically across `OneBrainComposer`, bare `RFPhasorComposer`, and a `TieredFactStore` buffer+`ShardedPhasorStore`
   LTM) for every DISTINCT patient under a hop's (agent, action); a hop with >=2 distinct patients ABSTAINS THE
   WHOLE CHAIN instead of taking `query_patient`'s first-match. `ChainedSVO` (a `list` subclass, same pattern as
   `HypothesisSVO`) carries `.derived_from` (the two verified hop-facts). A new `frame_derived_answer()` renders
   "I derived this from: `<fact1>`; `<fact2>`." — applied UNCONDITIONALLY to every chain answer, not gated behind
   the optional default-off `#129 BRAIN_SOURCE_PROVENANCE_HONESTY` monitor.
2. `webapp/server.py` (both the rich and single-fact `/api/brain-chat` paths) — a chain-route answer now reports
   `recalled_svo=null` / `derived=true` / `derived_from=[[a,v,p],...]` instead of a bare `[a,v,p]`
   indistinguishable from a direct recall; is excluded from the episodic `note_topic` write; and, when the
   optional `#129` monitor IS enabled, is encoded as `PROVENANCE_GENERATED` (never `PROVENANCE_PERCEIVED`).
3. `research/runners/rich_answer_composer.py` — a side channel (`_last_direct_derived`/`_last_direct_derived_from`)
   survives `gather()`'s fact-list normalization (which strips the `ChainedSVO` type) so `answer()` can frame +
   flag the rich-path paragraph the same way, and exclude it from the rich path's own episodic write.

A prior WIP hardening pass in worktree `agent-ae8d5a9e5cabe67da` (uncommitted, on the same `research/reasoning-
frontier` base) built the sound `_distinct_patients`/`ChainedSVO` design this reuses; two things changed from
that pass per this task's explicit spec: (a) `recalled_svo` is now `null` on a derived answer (that pass kept
the raw triple), and (b) the derived-answer framing is UNCONDITIONAL rather than gated behind the optional
provenance monitor (a derived answer must read as an inference even with `BRAIN_SOURCE_PROVENANCE_HONESTY`
unset, which is the production default).

## Requirement closure

| Audit req | Status | Evidence |
|---|---|---|
| #1 multi-valued-hop abstain | **DONE** | `_distinct_patients`/`_hop`; live CONFLICT4/5/5b below |
| #4 GENERATED provenance, hop-facts surfaced | **DONE** | `frame_derived_answer`; live A3/A3b/C2/PROV3 below |
| #5 distinct API shape + kept out of episodic memory | **DONE** | `recalled_svo=null`/`derived`/`derived_from`; episodic guard in both paths |
| #6 parser truncation fix | **CONFIRMED, not redone** | `_neural_question_parse` still pads to 2 content tokens (unfixed) — moot for this route because `parse_possessive_chain_question` runs its OWN regex on the raw question BEFORE that parser ever sees a compositional question (verified by code read + the live OVERRUN1/FP1 results below, which show the *pre-existing* truncation bug firing on OTHER, non-compositional questions, never on a routed chain question) |
| #2 confidence floor | **Not attempted** (explicit direction: optional defense-in-depth, not load-bearing at deployed D=128/15k) |
| #3, #7, #8, #9, #10 | **Out of scope** for this task (rate-sweep, lemmatizer table/shard-routing, `query_chain`-forced-actions, structural-marker gating, confidence threading) — named, not silently dropped; see "Known gaps" below |

## VERIFY: 37-turn battery through the REAL `/api/brain-chat` (in-process `TestClient`, `SIM_BACKEND=cupy`, `renderer=stub`)

**Anti-wedge note.** The prior agent's unverified re-verify reportedly hung on a full cupy warm. This run did NOT
hang — the FIRST call (`E1`) took 234.4s (the one-time `TieredFactStore`/onebrain-composer + several lazily-built
subsystem warm-ups, GPU actively at 99% the whole time, confirmed via `nvidia-smi`, not idle); every subsequent
call on the SAME warm brain was 2-21s (a chain hop pair, or a fresh vocabulary-growth teach, ~10-20s; a repeat
recall, 2-8s). Two lazily-built subsystems (the affect/rich-composer organs, and the CA3/dlPFC episodic organ)
added two further one-time costs of 70.4s (`E3`) and 49.2s (`E4`) later in the same run. Total wall-clock for the
Bash tool call exceeded its nominal 240000ms budget but was NOT killed (the tool kept it backgrounded rather than
terminating a live, GPU-busy process) and completed cleanly (exit 0, 37/37 records written, no exceptions). A
SEPARATE follow-up script (`hardened_ambiguity_provenance_driver.py`, meant only as a belt-and-suspenders re-check
of reqs #1/#4 on plainer vocabulary) WAS killed after ~4 minutes of genuine stall (GPU dropped to 25%, log
stopped growing) while contending for the GPU alongside the main run — it is deleted, unused, and not needed:
the main battery already gives clean, decisive, live evidence for every requirement below.

**All 37 calls returned HTTP 200 with no server exception.** Full transcript:
`research/findings/raw/_reasoning_frontier_verify/battery_cupy1.json` (also `.jsonl`, one record per line).

| # | Label | Message | Answer (truncated) | abstained | derived | recalled_svo |
|---|---|---|---|---|---|---|
| 1 | E1 regression recall | what does the dog chase? | The dog chases cat. | False | False | [dog,chase,cat] |
| 2 | E2 regression abstain | what does the dragon breathe? | I don't know about that. | True | False | null |
| 3 | D1 teach inflected | the wolf hunts the deer | The wolf hunts deer. | False | False | [wolf,hunt,deer] |
| 4 | D2 recall base form | what does the wolf hunt? | On wolf, then — The wolf hunts deer. | False | False | [wolf,hunt,deer] |
| 5 | IRR1 teach irregular past | the fox caught the mouse | The fox caughts mouse. | False | False | [fox,caught,mouse] |
| 6 | IRR2 recall base form | what did the fox catch? | (discourse-WM path) no assembly completes | True | n/a | null |
| 7 | HOMO1 teach tool noun | the carpenter used the saw | "I don't know the words 'carpenter' or 'saw' yet" | True | n/a | null |
| 8 | HOMO2 teach irregular verb | the girl saw the bird | The girl sees bird. | False | False | [girl,see,bird] |
| 9 | HOMO3 recall tool noun | what did the carpenter use? | I don't know about that. | True | False | null |
| 10 | HOMO4 recall irregular verb | what did the girl see? | (discourse-WM path) no assembly completes | True | n/a | null |
| 11 | A1 teach hop1 | the wolf eats the deer | On wolf, then — The wolf eats deer. | False | False | [wolf,eat,deer] |
| 12 | A2 teach hop2 | the deer eats the grass | On deer, then — The deer eats grass. | False | False | [deer,eat,grass] |
| 13 | **A3 derive chain (single-fact)** | what does the wolf's prey eat? | "I derived this from: wolf hunt deer; deer eat grass. deer eat grass [unverified render]" | False | **True** | **null** |
| 14 | **A3b derive chain (rich default)** | what does the wolf's prey eat? | "I derived this from: wolf hunt deer; deer eat grass. The deer eats grass...." | False | **True** | **null** |
| 15 | CONFAB1 crux fresh subject | what does the shark eat? | I don't know about that. | True | False | null |
| 16 | CONFAB2 near-miss teach | the fox chases the rabbit | On fox, then — The fox chases rabbit. | False | False | [fox,chase,rabbit] |
| 17 | CONFAB2b near-miss query | what does the fox eat? | I don't know about that. | True | False | null |
| 18 | B1 moat teach hop1-only | the wolverine hunts the badger | "I don't know the words 'wolverine' or 'badger' yet" | True | n/a | null |
| 19 | B2 moat unsupported hop2 | what does the wolverine's prey eat? | I don't know about that. | True | False | null |
| 20 | CONFLICT1 teach a | the lion eats the antelope | The lion eats antelope. | False | False | [lion,eat,antelope] |
| 21 | CONFLICT2 teach b | the lion eats the zebra | On lion, then — The lion eats zebra. | False | False | [lion,eat,zebra] |
| 22 | CONFLICT3 teach c | the zebra eats the grass | On zebra, then — The zebra eats grass. | False | False | [zebra,eat,grass] |
| 23 | **CONFLICT4 single-hop ambiguous** | what does the lion eat? | On lion, then — I don't know about that. | **True** | False | null |
| 24 | **CONFLICT5 chain ambiguous (single-fact)** | what does the lion's prey eat? | I don't know about that. | **True** | False | null |
| 25 | **CONFLICT5b chain ambiguous (rich default)** | what does the lion's prey eat? | I don't know about that. | **True** | False | null |
| 26 | PROV1 teach hop1 | the eagle hunts the rabbit | On rabbit, then — The eagle hunts rabbit. | False | False | [eagle,hunt,rabbit] |
| 27 | PROV2 teach hop2 | the rabbit eats the clover | The rabbit eats clover. | False | False | [rabbit,eat,clover] |
| 28 | **PROV3 derive, provenance ON** | what does the eagle's prey eat? | "I derived this from: eagle hunt rabbit; rabbit eat clover. The rabbit eats clover." | False | **True** | **null** — `provenance.label="generated"`, `encoded_as="generated"`, `agrees_with_encoded=True` |
| 29 | **PROV4 direct recall, provenance ON** | what does the wolf hunt? | On wolf, then — The wolf hunts deer. | False | False | [wolf,hunt,deer] — `provenance.label="perceived"`, `encoded_as="perceived"`, `agrees_with_encoded=True` |
| 30 | PROV5 provenance off again | what does the wolf hunt? | The wolf hunts deer. | False | False | [wolf,hunt,deer] |
| 31 | **C1 lesion ON** (`BRAIN_CHAIN_ROUTE=0`) | what does the wolf's prey eat? | I don't know about that. | **True** | False | null |
| 32 | **C2 lesion OFF (restored)** | what does the wolf's prey eat? | "I derived this from: wolf hunt deer; deer eat grass. The deer eats grass." | False | **True** | **null** |
| 33 | FP1 modifier-laden single-hop | what does the big hungry cat eat? | On cat, then — I don't know about that. | True | False | null |
| 34 | SHARD1 plural agent | what do cats eat? | I don't know about that. | True | False | null |
| 35 | OVERRUN1 three-hop shape | what does the wolf's prey's food eat? | I don't know about that. | True | False | null |
| 36 | E3 affect-lead regression | Wow, I absolutely love wolves...! What does the wolf hunt? | On wolf, then — Gladly — The wolf hunts deer.... | False | False | [wolf,hunt,deer] |
| 37 | E4 swap-lead regression | what does the dog chase? | On dog, then — Gladly — The dog chases cat.... | False | False | [dog,chase,cat] |

## Reading the results against the 12-item spec battery + the naive build's own (a)-(e)

- **(a) chain derive / #5 chain correctness — PASS** (#13, #14): "grass" derived correctly via both the
  single-fact and the DEFAULT rich path, never abstaining, never stopping at the first hop.
- **(b) unsupported-hop moat / #1 CONFAB CRUX / #2 CONFAB near-miss — PASS** (#2, #15, #17, #19): every genuinely
  unsupported query (fresh subject, near-miss relation, unsupported 2nd hop) honestly abstains, never fabricates.
- **#3 MOAT-BYPASS conflict — PASS, and this is the decisive req #1 result** (#23-25): after teaching the lion two
  competing `eat` facts (antelope, zebra), BOTH the single-hop question (`CONFLICT4`, via the pre-existing GNW
  deliberation gate) AND the chain question (`CONFLICT5`/`CONFLICT5b`, via this arc's new `_distinct_patients`/
  `_hop` ambiguity check) abstain instead of silently walking to `grass` via the zebra branch. Traced: for "what
  does the lion's prey eat?", hop1 candidate `hunt` finds 0 patients (no `lion hunts X` fact) and continues; hop1
  candidate `eat` finds 2 distinct patients (`antelope`, `zebra`) and returns `ambiguous=True`, aborting the
  whole chain — exactly the designed behavior, never reached by the prior naive `query_patient` first-match.
- **#4 MOAT-BYPASS provenance — PASS** (#28 vs #29): with the optional monitor on, the SAME two-hop derivation
  used for #13/#14 is (a) framed as "I derived this from: eagle hunt rabbit; rabbit eat clover." — the exact
  hop-facts, not a generic disclaimer; (b) judged `generated` by the live spiking opponent-comparator, agreeing
  with the `PROVENANCE_GENERATED` encoding; (c) `recalled_svo=null`. The CONTRAST turn (#29, a genuine direct
  recall with the SAME monitor on) is judged `perceived` — confirming the GENERATED/PERCEIVED split is driven by
  `_is_chain_route`, not a blanket change to provenance behavior.
- **#5 chain terminal API shape / episodic exclusion — PASS** (#13, #14, #28, #32, all four derived turns):
  every derived answer reports `recalled_svo=null` + `derived=true` + `derived_from=[[a,v,p],[a,v,p]]`. The
  episodic-store guard (`not _is_chain_route` / `not _rich_derived`) is a straight code read, not independently
  observable through the API (no endpoint exposes the episodic organ's contents) — see "Honest scope" below.
- **#6 lemmatization store/query mismatch — PASS** (#3, #4): "hunts" taught, "hunt" recalls the same fact
  (pre-existing `lexical_lemma.lemma_verb`, confirmed still working, not touched by this arc).
- **#7 IRREGULAR inflection — FAIL, confirmed OUT OF SCOPE** (#5, #6): "caught" is NOT in `_IRREGULAR_VERBS`
  (only `ate/ran/went/saw/gave/made/took/came/did/had/was/were/is/are/am/said/got/knew/thought`), so it is
  stored raw and "what did the fox catch?" cannot find it. This is the audit's OWN named req #7 gap
  (lemmatizer-table completeness), explicitly not part of this task's 4-item scope. Honest, not silently dropped.
- **#8 HOMOGRAPH separation — INCONCLUSIVE, for a reason orthogonal to hardening** (#7-#10): teaching "the
  carpenter used the saw" was REJECTED by the substrate's pre-existing in-loop vocabulary-growth mechanism
  ("I don't know the words 'carpenter' or 'saw' yet") — an intermittent, pre-existing constraint unrelated to
  this arc (see "Incidental finding" below); the fact never entered the store, so `HOMO3`'s later abstain proves
  nothing about homograph safety either way. `HOMO2`/`HOMO4` (the "saw"-as-verb side) taught fine but then
  abstained via the SAME discourse-WM completion-failure path `IRR2` hit (also orthogonal — not a moat issue).
  The homograph-separation CLAIM itself remains true BY STATIC CONSTRUCTION (verbs are always lemmatized via
  `lemma_verb`, patients/nouns never are — different token slots, so a verb "saw"→"see" and a noun "saw" cannot
  collide at the store layer), but this run did not produce a clean LIVE confirmation of it.
- **#9 GOAL-BLIND relation substitution — PASS BY CONSTRUCTION, not live-tested this run**: `resolve_compositional_
  chain` never calls `chain_of_thought`/`_select_next_relation` (confirmed by reading `compositional_chain_route.py`
  end to end) — there is no association-strength mechanism in this route for a stronger unrelated relation to hijack.
- **#10 FALSE-POSITIVE routing — the chain route does NOT mis-fire (confirmed), but the turn still abstains for an
  UNRELATED, pre-existing reason** (#33): "what does the big hungry cat eat?" never matches
  `_POSSESSIVE_CHAIN_RE` (no apostrophe) — `derived=False` proves the chain route was never engaged. It still
  abstains because `_neural_question_parse` (`brain_chat_tui.py`) pads to `[content[0], content[1], "__q__"]`
  regardless of question length, so "big"/"hungry" (not stopwords) occupy the two extracted slots instead of
  "cat"/"eat" — the SAME primitive named in audit req #6, but manifesting on an ORDINARY question, not a
  compositional one. This is a real, pre-existing, out-of-scope bug (spawned as a separate task; see below), not
  a hardening regression: the compositional detector itself has NO word-count trigger to guard (req #9's concern
  is moot in this codebase by construction — the only gate is the apostrophe-`'s` regex).
- **#11 SHARD-ROUTING lemmatization (noun/plural) — FAIL, confirmed OUT OF SCOPE** (#34): "what do cats eat?"
  abstains (no noun-side lemmatization at `ShardedPhasorStore.route`/`shard_for`) — audit req #7's noun half,
  explicitly out of scope.
- **#12 OVER-RUN / deep chain — PASS** (#35): the 3-hop "wolf's prey's food eat" shape does not match the 2-hop
  regex at all, falls through, and abstains — no fabricated terminal.
- **LOAD-BEARING LESION — PASS** (#31 vs #32): `BRAIN_CHAIN_ROUTE=0` reverts the identical question to abstain;
  restoring the flag immediately re-derives "grass" — the route drives the derived answer, not decoration.
- **Regression (naive build's own (e), affect/swap) — PASS** (#1, #36, #37): pre-existing recall/abstain, the
  affect lead ("Gladly —") and swap lead ("On X, then —") all compose correctly around a derived OR direct
  answer with no crash and no change to non-compositional turns.

## Incidental finding (not part of this hardening, flagged separately)

Two vocabulary-growth artifacts were observed, both PRE-EXISTING and untouched by this arc:
1. An intermittent in-loop vocabulary-growth rejection ("I don't know the words 'X' or 'Y' yet") on SOME
   two-brand-new-word sentences (`HOMO1`, `B1`) but not others (`CONFLICT1` "the lion eats the antelope",
   `PROV1` "the eagle hunts the rabbit" — both also 2 brand-new words, both succeeded). The exact trigger
   condition was not isolated in this session (not this arc's scope; no code touched here is implicated).
2. `_neural_question_parse`'s position-only 2-content-token extraction (`brain_chat_tui.py`, the SAME primitive
   named in audit req #6) causes ANY question with >2 content words — compositional or not — to mis-extract
   (`FP1` above). Confirmed via code read: `padded = [content[0], content[1], "__q__"]` never reads a 3rd token.

## Honest scope

- The episodic-store exclusion (req #5) is verified by CODE READ (`not _is_chain_route` / `not _rich_derived`
  gates the `note_topic` call in both paths) plus the CONSISTENT absence of any derived fact being fed back as a
  `recalled_svo` in a later turn — no endpoint exposes the episodic organ's contents directly to confirm by
  black-box observation alone.
- `verified` is left `True` on a derived answer (a hop-verified derivation, not a lie) rather than forced
  `False`; the distinguishing signal the task asked for is `recalled_svo=null` + `derived=true`, both confirmed
  present on every derived turn above.
- No claim of "fully spiking" or "brain-based-only" is made for the ROUTING/detection layer (the apostrophe
  regex, the role-noun hint table, the ambiguity scan) — these remain documented HOST SCAFFOLDS on the ladder to
  a learned replacement, exactly as the base build's own docstring states; the DATA READS (`query_patient` per
  hop) are the genuinely spiking substrate op, unchanged and re-checked at each hop.
- The `_distinct_patients` ambiguity check is a DETERMINISTIC `.kb` scan, not the audit's top-ranked fix (routing
  through the SPIKING GNW conflict read via `webapp/gnw_deliberation.py`'s `all_candidate_patients`, which
  requires `_iter_facts`/`unbind`, RF-composer-only). It gets the SAME outcome (abstain on ambiguity) generically
  across composer kinds including the production `OneBrainComposer`; the honest gap against the spiking version
  is named, not hidden.
