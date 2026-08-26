---
type: finding
status: contributing
date: 2026-08-25
mechanism: reasoning-route-chain-routing-moat-hardening-spec
lane: integration
---

<!--derived: this is an ANALYSIS/SPEC doc, not a fresh measurement — every quantitative value in it is quoted or reasoned from the cited audit journal research/findings/raw/reasoning_route_moat_audit/audit_result_wf_89e66a22-2cb.json (code line-cites are the "measurement"). The FHRR decode rates it calls UNMEASURED are being measured separately in 2026-08-25-fhrr-decode-rate-at-scale.-->

# Reasoning-route (chain-routing) moat-hardening spec + adversarial test battery

**Board #141.** Before wiring the live `/api/brain-chat` handler to the brain's existing multi-hop inference engine (`ShardedPhasorStore.query_chain` / `chain_of_thought`), a read-only adversarial audit (4 lenses -> synthesis -> completeness critic; workflow `wf_89e66a22-2cb`) mapped the confabulation / moat-bypass surface. This is the authoritative hardening spec AND the review battery the implementation must pass. It supersedes the naive 'just call query_chain, verify per hop' framing in the original build brief.

**Artifact:** the raw 6-agent audit output (4 lenses + synthesis + critic, structured JSON per agent) is
preserved at `research/findings/raw/reasoning_route_moat_audit/audit_result_wf_89e66a22-2cb.json` — every
claim below traces to a `{"type":"result"}` row there. The audit "measured" the confabulation surface of the
current store code (`rf_phasor_composer.py`, `sharded_phasor_store.py`) by reading it; the code line-cites are
the measurement.

## Moat verdict

SAFE-WITH-FIXES. The per-hop no-confab moat is preserved BY CONSTRUCTION for genuinely-unknown cues: query_chain (rf_phasor_composer.py:1034-1047) returns None the instant any hop's query_patient returns None, and that None comes from _scan_first_match (712-721), which decodes the (agent,action) roles of every TRULY-STORED composite and matches them by EXACT STRING EQUALITY (`w == val`, line 719) against the queried pair. Because the nearest-neighbour cleanup (_cleanup, 658-663; always returns the argmax word, never None) is applied to recover a STORED value and then compared for equality against the cue — not thresholded on the cue itself — a 'cleanup always returns something' argmax cannot by itself fabricate a matching hop, and an out-of-vocab subject can never be matched at all. So routing natural compositional questions to query_chain is genuinely moat-safe for real unknowns — this is a strength, not a hole. BUT the gate has NO minimum-similarity floor, so its entire safety rests on ONE load-bearing, currently-UNMEASURED condition: the rate at which, at the deployed D=128 / 15k-fact / real-vocab scale, FHRR bundling crosstalk pushes a wrong-agent fact's decoded agent+action onto an in-vocab query word (a false hop) — which the chain then LAUNDERS into a confident multi-hop answer with no low-confidence signal, since every intermediate is re-discretised to a clean codebook word. The existing '0 confabulations' evidence never tested this (it probed only UNSTORED cues that abstain structurally). Secondarily, the plan's literal wording — 'call ShardedPhasorStore.query_chain directly' — would bypass the GNW deliberation conflict-abstain (a multi-valued hop that the single-hop path withholds) and reuse the hardcoded PROVENANCE_PERCEIVED wire (server.py:5164-5175), asserting an inference as a taught, 'verified' fact. The single load-bearing condition is therefore: the per-hop cue+answer decode must carry a confidence floor (reuse the already-computed _cleanup_all_score_stats margin) with the near-miss false-positive rate measured and driven to zero at deployed scale, AND each hop must pass through chat.gate's conflict-abstain and be framed as GENERATED — with those in place the route is moat-clean; without them it is a moat bypass.

**Critic correction to the verdict:** The verdict's core is CORRECT and well-grounded, not alarmist: for genuinely-unknown out-of-vocab cues the moat holds by construction — verified that _scan_first_match matches by exact string equality of a DECODED STORED cue against the query (rf_phasor_composer.py:719 `w == val`), _cleanup always returns an argmax word (:658-663) but is COMPARED to the cue rather than thresholded on it, so an out-of-vocab subject can never match. The verdict is slightly TOO OPTIMISTIC in ONE structural way: it names a SINGLE unmeasured load-bearing condition (the cue-role false-hop crosstalk rate) and explicitly treats the answer decode as laundered-safe ('every intermediate is re-discretised to a clean codebook word'). But re-discretisation is exactly what HIDES a supported-hop WRONG-PATIENT misrecall — _render returns a bare argmax with no equality check against ground truth (:557) — so there are TWO unmeasured load-bearing rates at deployed scale, not one: the cue-role false-hop rate AND the answer/attribute-role misrecall rate, and the second is arguably LIKELIER to bite because a 15k LTM guarantees a large answer codebook and many supported hops. It also under-weights that multi-valued (agent,action) hops are the COMMON case on a bulk LTM (so the deliberation-bypass is frequent, not a corner case) and leaves the single-store-reachability and bounded-chain-latency assumptions unstated. Corrected: SAFE-WITH-FIXES, but the load-bearing unmeasured surface is TWO decode rates (cue AND answer/attribute) at D=128/15k/real-V, the deliberation conflict-abstain must be inherited on what will be a HIGH-FREQUENCY multi-valued-hop path, and safety additionally rests on an unstated single-store reachability invariant and a bounded per-turn chain latency.

## The core mechanism (confabulation lens, grounded in code)

- `_cleanup` / `_cleanup_all` (`rf_phasor_composer.py:658-663 / 700-710`) are **bare argmax** — they NEVER return None; they always return the nearest codebook word.
- The ONLY abstain is `_scan_first_match` (`:712-721`): it decodes the (agent,action) roles of every TRULY-STORED composite and matches by **exact string equality** (`w == val`, `:719`) — **no similarity floor, no top1/top2 margin**.
- **Safe by construction for out-of-vocab cues** (argmax can only emit an in-vocab word, so an unknown subject can never match) — this is why the existing '0 confabulations' soak evidence holds: it only probes UNSTORED cues, which abstain structurally.
- **The hole:** inside a compositional chain every concept is in-vocab, so the gate rests entirely on the decode landing on the exact query string. FHRR bundling crosstalk (store() bundles up to 5-6 terms) can push a wrong-agent fact's decoded agent+action onto the query word at ANY similarity -> a fabricated hop, then LAUNDERED into a confident multi-hop answer with no low-confidence signal.
- **Two** unmeasured load-bearing decode rates at deployed `D=128` / 15k-fact / real-vocab scale (critic): the cue-role false-hop rate AND the answer/attribute-role wrong-patient misrecall rate (`_render` `:557` is also a floorless bare argmax with no equality check against ground truth — the channel the synthesis wrongly called 'laundered-safe').

## Hardening requirements (the implementation MUST meet these; most-critical first)

1. ROUTE EACH HOP THROUGH chat.gate, NOT a raw composer.query_chain / ShardedPhasorStore.query_chain call. Verified: the deliberation conflict-abstain lives in the gate wrapper via all_candidate_patients (gnw_deliberation.py:122-134); query_chain's per-hop query_patient instead silently takes _scan_first_match's FIRST match (rf_phasor_composer.py:721 `return int(idx[0])`). A direct composer call — which the plan's literal wording says — bypasses the entire GNW consensus/comprehension/conflict stack. A hop whose (agent,action) has >=2 distinct stored patients MUST abstain the whole chain (the single-hop path would say 'I don't know'), never march through an arbitrary first-match.

2. ADD A PER-HOP CONFIDENCE FLOOR to the abstain gate — the moat is exact-string-equality-gated, NOT confidence-gated (rf_phasor_composer.py:719 `mask &= (w == val)`; _cleanup at 658-663 always returns the argmax word, no None). Replace the bare `w == val` with `w == val AND winner_score_raw >= tau AND margin >= delta` using the already-computed _cleanup_all_score_stats (724-755, currently trace-only). Gate BOTH cue roles (agent, action) AND the answer decode (patient/attribute at 1010-1012) so a thin-margin crosstalk near-miss abstains (returns None) instead of emitting the nearest word. Calibrate tau/delta from the same-vs-different distribution the code already builds (_calibrate_pe_labile, 870-889).

3. MEASURE, before trusting the route, the per-hop cue-decode FALSE-POSITIVE rate (a wrong-agent fact whose agent+action both crosstalk-decode onto an IN-VOCAB query word) and the wrong-patient decode rate at the DEPLOYED D=128 (tiered_fact_store.py:65 / developed_brain_io.py:369) and 15k-fact / real-vocab scale. The existing '0 confabulations' evidence (_knowledge_scale_flip_soak.py:174-188) only probes UNSTORED cues that abstain STRUCTURALLY; it never stresses in-vocab near-misses. If nonzero, raise D (faithfulness > speed). This is the single number the floor-less gate's safety rests on.

4. FRAME A DERIVED ANSWER AS DERIVED, NOT PERCEIVED. The #129 wire hardcodes PROVENANCE_PERCEIVED (webapp/server.py:5164-5175) under an invariant comment (5154-5156) that a chain route breaks. Encode PROVENANCE_GENERATED so provenance_framed_text (source_provenance_honesty.py:120-134) emits the 'my own inference' framing, surface the supporting hop-facts ('I derived this from <fact1> and <fact2>'), and fix the now-false invariant comment. Reusing the PERCEIVED wire verbatim inverts the plan's own honest-derivation goal into a provenance lie.

5. GIVE A DERIVED ANSWER A DISTINCT API SHAPE AND KEEP IT OUT OF EPISODIC MEMORY. verified='[unverified render' not in answer (server.py:5148-5150) only checks the SURFACE re-parses — NOT that the triple is stored — so a synthesized terminal (precedent: gnw_multistep_deliberation.py out[2]=terminal) reports verified=true / recalled_svo=[wolf,eat,grass] indistinguishably from a real recall. Set recalled_svo=None (or derived_from=[[a,v,p]...]) + derived:true, and do NOT feed the synthesized terminal SVO into note_topic (server.py:5181-5184) or the discourse-WM referent write (brain_chat_tui.py:380-382). Verify each HOP-fact (each is stored), not the composed terminal.

6. FIX THE EXTRACTION TRUNCATION FIRST — run compositional detection + relation-sequence extraction on the FULL stopword-stripped content list BEFORE _extract_route's two-word collapse. _neural_question_parse pads to `[content[0], content[1], '__q__']` (brain_chat_tui.py:652) and never reads content[2]+, and _STOP (696-698) deletes 'of' and does not split possessive 's, so 'what does the wolf's prey eat?' loses the verb 'eat' before any chain logic runs. Treat 's and 'of' as structural relation markers, not stopwords. Any detector bolted on AFTER this collapse cannot work.

7. LAND EXACTLY ONE CANONICAL, ROLE-GATED LEMMATIZER applied to BOTH store and query sides. No canonicalization exists today (rf_phasor_composer.store, 835-852, stores raw strings) — this is the documented live hunts/hunt abstain bug. Apply it INSIDE ShardedPhasorStore's store/query_patient/query_chain immediately before route()/shard_for() (route hashes the raw agent, sharded_phasor_store.py:93-96) so a caller cannot bypass it. Role-gate it (verb-lemma only on the action slot, noun-lemma only on agent/patient) to avoid homograph collapse (saw/see, left/leave, rose/rise), include an irregular table (caught->catch, ate->eat, mice->mouse) since a suffix-only stemmer leaves irregulars silently broken, and add a store<->query parity test. Do NOT reuse one of the ~6 divergent ad hoc helpers without deduping them.

8. ROUTE EXPLICIT COMPOSITIONAL QUESTIONS THROUGH query_chain(cue, [explicit asked actions]), NOT chain_of_thought. _select_next_relation (rf_phasor_composer.py:1063-1076) picks max co-occurrence assoc GOAL-BLIND — the `goal` arg only stops the loop, it does not steer relation choice — so chain_of_thought can chase a different, stronger relation than the one asked and return a real-but-off-question answer no moat flags. query_chain forces the asked relations, so a missing asked hop abstains. Reserve chain_of_thought for genuinely open-ended free-association turns.

9. GATE THE COMPOSITIONAL DETECTOR ON A STRUCTURAL MARKER (genitive 's on the wh-subject, or a member of a closed relation-noun->verb table), NOT WORD COUNT, and fail closed to the single-hop path when absent. 'what does the big hungry wolf eat?' strips to 4 content words (adjectives are not in _STOP) — the same length signature a naive >=3-word trigger fires on — but is a plain single-hop question. Mirror the safe closed-whitelist precedent (gnw_multistep_deliberation.py detect_chase: no marker -> pure pass-through).

10. THREAD AN END-TO-END CONFIDENCE THROUGH query_chain and let the handler hedge/decline below threshold. Per-hop error compounds ~ (1-p)^H and today query_chain returns only the bare terminal or None (rf_phasor_composer.py:1034-1047) with no confidence — a 4-hop chain silently degrades. Have query_patient also return the min cue-margin/answer-margin per hop, aggregate (min or product), return (terminal, confidence).

## Adversarial test battery (RUN these against the real `/api/brain-chat` implementation)

### CONFAB CRUX — unsupported hop on a distinct in-vocab subject must abstain (baseline structural safety)
- **Setup:** Teach exactly one fact: 'the deer eats grass.' Then ask: 'what does the wolf eat?' (wolf is a fresh, out-of-store subject).
- **Expect:** Abstain — 'I don't know about that.' No patient returned; abstained=true; recalled_svo=null.
- **Catches:** The core no-confab moat: _scan_first_match must find zero facts whose decoded (agent,action) string-equal (wolf,eat). If the argmax cleanup of the stored (deer,eat,grass)'s agent role crosstalk-decodes onto 'wolf', a fabricated 'wolf eats grass' leaks. This is the finding-1/finding-2 confabulation channel.

### CONFAB near-miss AT SCALE — in-vocab subject with no fact under the asked relation, on the 15k LTM
- **Setup:** On the production/15k-attached brain (many in-vocab agents+actions), teach 'the fox chases the rabbit.' Then ask 'what does the fox eat?' — 'fox' and 'eat' are both in-vocab but no (fox,eat,*) is stored.
- **Expect:** Abstain. Must NOT return the rabbit (from the chase fact) or any patient from a wrong-agent eat fact.
- **Catches:** The floor-less argmax gate handing a crosstalk filler at deployed D=128/15k scale — the single unmeasured false-positive rate the moat rests on. Exposes whether the per-hop confidence floor (hardening #2) actually fires.

### MOAT-BYPASS conflict — a multi-valued hop must inherit the deliberation conflict-abstain
- **Setup:** Teach: 'the wolf eats deer.' AND 'the wolf eats sheep.' AND 'the deer eats grass.' AND 'the sheep eats clover.' Then ask: 'what does the wolf eat?' (single hop) and 'what does the wolf's prey eat?' (chain).
- **Expect:** Both abstain. The (wolf,eat) hop is multi-valued, so the single-hop deliberation gate abstains, and the chain must abstain too — NOT silently pick the first-match patient and march to 'grass' or 'clover'.
- **Catches:** Direct composer.query_chain bypassing all_candidate_patients (gnw_deliberation.py:122-134) and taking _scan_first_match idx[0]. Confirms each hop is driven through chat.gate (hardening #1).

### MOAT-BYPASS provenance — a derived answer must NOT be framed as perceived / reported verified
- **Setup:** With BRAIN_SOURCE_PROVENANCE_HONESTY enabled, teach 'the wolf hunts the deer.' AND 'the deer eats grass.' Then ask 'what does the wolf's prey eat?'
- **Expect:** Answers 'grass' framed as an INFERENCE (e.g. 'I derived this from: the wolf hunts the deer; the deer eats grass'), with the provenance label GENERATED. recalled_svo must NOT be [wolf,eat,grass]; verified must NOT report it as a directly-recalled fact; the episodic store (note_topic) must NOT ingest [wolf,eat,grass].
- **Catches:** The hardcoded PROVENANCE_PERCEIVED wire (server.py:5164-5175) inverting honesty, verified=true on a synthesized triple (server.py:5148-5150), and episodic pollution (server.py:5181-5184). Covers hardening #4 and #5.

### CHAIN correctness — the two-hop question returns the chained answer, not the first hop only
- **Setup:** Teach 'the wolf hunts the deer.' AND 'the deer eats grass.' Ask 'what does the wolf's prey eat?'
- **Expect:** 'grass' (the correct 2-hop terminal). NOT an abstain, and NOT the first-hop-only 'the wolf hunts deer.'
- **Catches:** The _extract_route truncation (brain_chat_tui.py:652) discarding the second relation so the question is answered as an atomic single hop, and whether the relation-noun 'prey'->hunt-inverse decomposition actually reaches query_chain. Covers hardening #6.

### LEMMATIZATION store/query mismatch — the documented hunts/hunt regression
- **Setup:** Teach 'the wolf hunts the deer.' Then ask 'what does the wolf hunt?' (query verb 'hunt' vs stored 'hunts').
- **Expect:** 'deer' — must NOT abstain.
- **Catches:** Absent canonicalization making the exact-string gate (w=='hunts' vs val=='hunt') miss a just-taught fact — a silent recall regression that HIDES inside the honest-abstain UI, indistinguishable from a true unknown. Confirms one canonical lemmatizer on both sides (hardening #7).

### IRREGULAR inflection — under-merge (a suffix-only stemmer leaves it broken)
- **Setup:** Teach 'the fox caught the mouse.' Then ask 'what did the fox catch?' (caught->catch verb; mouse/mice noun agreement).
- **Expect:** 'mouse' — must NOT abstain.
- **Catches:** A regular-suffix stemmer silently fixing hunts/hunt while leaving high-frequency irregulars (ate, saw, caught, went, mice) broken. Forces the irregular table to be a first-class deliverable (hardening #7).

### HOMOGRAPH separation — over-merge (POS-blind irregular rewrite fuses two senses)
- **Setup:** Teach 'the carpenter used the saw.' (patient=saw, tool noun) AND 'the girl saw the bird.' (saw = past-of-see verb). Ask 'what did the carpenter use?' AND 'what did the girl see?'
- **Expect:** 'saw' (the tool) and 'bird' respectively — the two senses stay separately recallable. The noun 'saw' must NOT collapse onto the verb 'see' key.
- **Catches:** A conjunctive_parser._IRREGULAR-style token-blind saw->see rewrite (or an ungated verb-lemma) applied to a patient-slot token, fusing two distinct facts onto one codebook key — a store-side corruption the no-confab moat cannot catch. Confirms role-gated lemmatization (hardening #7).

### GOAL-BLIND relation substitution — chain_of_thought answers a different chain than asked
- **Setup:** Teach 'the wolf hunts the deer.' AND a higher-co-occurrence wolf relation, e.g. 'the wolf runs.' twice (so 'run' outweighs 'hunt' in assoc). Ask 'what does the wolf's prey eat?' — if routed via chain_of_thought.
- **Expect:** Follows the ASKED relations (prey/hunt-inverse then eat) -> 'grass' (given a deer-eats fact) or abstains if the asked hop is missing; must NOT chase 'run' just because it has higher association.
- **Catches:** _select_next_relation (rf_phasor_composer.py:1063-1076) picking max-assoc goal-blind. Confirms explicit compositional questions route through query_chain with the asked action list (hardening #8).

### FALSE-POSITIVE routing — a modifier-laden single-hop question must not be detoured into the chain engine
- **Setup:** Teach 'the wolf eats grass.' Ask 'what does the big hungry wolf eat?' (4 content words after strip, but single-hop).
- **Expect:** 'grass' via the single-hop path — must NOT abstain and must NOT be routed into query_chain.
- **Catches:** A word-count-based compositional detector (adjectives are not in _STOP) mis-triggering the chain path and regressing a previously-working simple question. Confirms structural-marker gating (hardening #9).

### SHARD-ROUTING lemmatization (noun side) — plural subject must reach the stored shard
- **Setup:** Teach 'the wolf eats grass.' (agent stored as 'wolf'). Ask 'what do wolves eat?' (agent 'wolves').
- **Expect:** 'grass' — must NOT abstain.
- **Catches:** A lemmatizer placed only in the composer wrapper but bypassed at ShardedPhasorStore.route (sharded_phasor_store.py:93-96), so 'wolf' and 'wolves' hash to different shards and never meet — a failure identical to a genuine unknown-agent abstain, at the routing layer. Confirms normalization INSIDE the shard router (hardening #7).

### OVER-RUN / deep chain — an unknown terminal hop must abstain, not fabricate
- **Setup:** Teach 'the wolf hunts the deer.' AND 'the deer eats grass.' (nothing about what grass does). Ask 'what does the wolf's prey's food eat?' (3 hops: hunt->deer, eat->grass, eat->??? grass has no stored eat fact).
- **Expect:** Abstain — the third hop dead-ends, so the whole chain returns 'I don't know', NOT a confident fabricated terminal.
- **Catches:** Per-hop error compounding into a silent wrong terminal, and whether query_chain still abstains on an over-run chain. Also exercises the end-to-end confidence thread (hardening #10).

## Ranked correctness levers

- **[M] Drive each hop THROUGH chat.gate (inheriting the deliberation conflict-abstain + comprehension veto + the response tail's provenance/metacog/verify), rather than calling composer.query_chain / ShardedPhasorStore.query_chain directly.** — The single highest-leverage routing decision. It collapses the three top moat-bypass findings at once: the multi-valued-hop conflict abstain (all_candidate_patients, gnw_deliberation.py:122-134, which query_chain's idx[0] first-match ignores), the perceived-framing / verified=true on a synthesized triple, and the raw-early-return bypass of the honesty tail. The plan's literal 'call query_chain' wording is the bypass; gated hops are safe.

- **[M] Put a confidence floor + margin on the abstain decision — replace `w == val` in _scan_first_match with `w == val AND winner_score_raw >= tau AND margin >= delta`, on cue AND answer roles, reusing the already-built _cleanup_all_score_stats (rf_phasor_composer.py:724-755).** — The moat is exact-string-equality-gated with NO minimum-similarity floor (line 719); _cleanup always returns the nearest word. That is genuinely safe for out-of-vocab cues but relies entirely on decode-perfection for in-vocab near-misses, where FHRR bundling crosstalk can push a wrong fact's argmax onto the query word. A floor turns a silent false hop into an honest abstain.

- **[M] Measure the per-hop near-miss false-positive rate and wrong-patient rate at the DEPLOYED D=128 / 15k-fact / real-vocab scale on IN-VOCAB near-miss cues, and raise D until it is zero.** — This is the one number that decides whether the floor-less gate is actually safe or merely safe on the cues the soak happened to pick. The existing '0 confabulations' evidence (_knowledge_scale_flip_soak.py:174-188) only tested unstored cues that abstain structurally. Faithfulness over speed is the project standard, so raising D is in-scope.

- **[M] Frame derived answers as GENERATED provenance with surfaced hop-facts, give them a distinct API shape (recalled_svo=null / derived_from + derived:true), and keep the synthesized terminal out of the episodic store and discourse-WM.** — Prevents the honesty inversion — the #129 wire hardcodes PROVENANCE_PERCEIVED (server.py:5164-5175) and verified only checks a surface re-parse (5148-5150), so a chain terminal would be asserted as a taught, verified fact and then ingested as perceived, polluting session memory. This directly serves the plan's own goal-#3 ('I derived this from X and Y').

- **[M] Fix _extract_route/_neural_question_parse to run compositional detection + relation extraction on the FULL stopword-stripped content list BEFORE the two-word collapse, and treat possessive 's / 'of' as relation markers.** — Load-bearing precondition: _neural_question_parse pads to [content[0], content[1], '__q__'] (brain_chat_tui.py:652) and never reads content[2]+, and _STOP deletes 'of' and does not split 's, so the relation/possessive is discarded before any chain logic can run. A detector bolted on after this point cannot work at all.

- **[L] Land ONE canonical, role-gated lemmatizer applied to both store and query sides, inside ShardedPhasorStore before route(), with irregular verb+noun tables, a homograph guard, and a store<->query parity test.** — Fixes the documented live hunts/hunt abstain without introducing over-merge (homograph collapse saw/see) or under-merge (irregulars silently unfixed). It is L because it must dedupe ~6 divergent ad hoc helpers, cover every store+query call site including the shard-routing hash, and add a homograph regression battery.

- **[S] Route explicit compositional questions through query_chain(cue, [asked actions]); reserve chain_of_thought for open-ended free-association turns.** — Cheap, strictly-safer win. _select_next_relation (1063-1076) is goal-blind max-assoc, so chain_of_thought can answer a different chain than asked with no moat flag. query_chain forces the asked relations, so a missing asked hop abstains instead of being replaced by an available-but-unasked relation.

- **[M] Gate the compositional detector on an unambiguous structural marker (genitive 's or a closed relation-noun->verb table), fail-closed to the single-hop path, rather than on content-word count.** — Prevents regressing modifier-laden single-hop questions ('what does the big hungry wolf eat?', 4 content words) into the chain engine. Mirrors the safe closed-whitelist precedent already in the codebase (gnw_multistep_deliberation.py detect_chase: no marker -> pure pass-through).

- **[M] Thread an end-to-end confidence (min or product of per-hop margins) through query_chain and have the handler hedge/decline below a threshold.** — Per-hop error compounds ~ (1-p)^H, and today query_chain returns only a bare terminal-or-None (1034-1047) with no confidence — longer chains degrade silently. Aggregating the per-hop margins converts a silent wrong chain into an explicit low-confidence hedge or abstain.

## Completeness critic — failure modes the lenses missed (all confirmed against code)

- SUPPORTED-HOP WRONG-PATIENT/ATTRIBUTE DECODE (the channel the verdict wrongly calls laundered-safe). _render returns a bare _cleanup(rec) argmax (rf_phasor_composer.py:557) and the attribute decode is a bare unbind (query_patient:1011) — NEITHER has an equality check against ground truth NOR a floor. The moat's equality gate (_scan_first_match w==val, :719) touches ONLY the cue roles, never the answer. So a hop whose (agent,action) genuinely match a real fact can still return the WRONG stored patient via codebook crosstalk against the large 15k-vocab codebook: a confident wrong terminal that is NOT an out-of-vocab confab and is invisible to every equality-based moat. Re-discretisation to a clean codebook word is exactly what HIDES this, not what prevents it.

- MULTI-VALUED HOPS ARE THE COMMON CASE ON A 15k BULK LTM, NOT AN EDGE CASE. A corpus stores the same (agent,action) with many patients ('dog eat X' from many sentences), so _scan_first_match's idx[0] (rf_phasor_composer.py:721) — kb-INSERTION-ORDER first-match — silently picks an arbitrary patient at potentially every hop. A direct ShardedPhasorStore.query_chain bypasses the single-hop deliberation abstain (all_candidate_patients, gnw_deliberation.py:122-134). The spec treats deliberation-bypass as a corner fix; at LTM scale it is a HIGH-FREQUENCY correctness hazard, and the chain answer becomes INSERTION-ORDER dependent (load() preserves order but a re-ingest in a different order flips the terminal) = a determinism hazard.

- STORE-IDENTITY / MULTI-TURN REACHABILITY. 'Attach the 15k LTM to the default brain' leaves unspecified whether in-session hear() facts and the LTM live in ONE queryable store. If the LTM is a separate ShardedPhasorStore, a chain needing one just-taught hop + one LTM hop cannot resolve (the taught fact isn't in the LTM store; the LTM fact isn't in inner.composer), and a fact taught 5 turns ago — past the discourse-WM window (brain_chat_tui.py:758-764) — may be unreachable. No requirement pins the single-store invariant the chain silently assumes.

- INVERSE-DIRECTION RELATION-NOUNS. query_chain is AGENT-pivot FORWARD only (patient becomes next agent). 'the wolf's prey' happens to be forward-aligned (patient of hunt), but 'the deer's predator / X's employer / X's parent' need the AGENT of (?,rel,X) = a reverse query_agent that fans out ALL shards (sharded_phasor_store.py:152-169) — query_chain cannot express it and will mis-hop or abstain. Neither a hardening requirement nor a test covers inverse possessives; the battery's one relation-noun ('prey') is the lucky forward case.

- DOWNSTREAM COUPLINGS RE-INFLATE A DERIVED ANSWER. server.py appends a da_drives suffix (:5130-5131) and prepends world-model/surprise/reconsolidation/pragmatic (:5191-5193) plus a metacog hedge (:5195+). The metacog reads _read_activity() = the LAST substrate op's confidence, which for a 4-hop chain reflects only the final hop — mislabeling a compounded chain's confidence. An affect/DA suffix can also make a GENERATED-framed inference read MORE confident, fighting provenance_framed_text's 'I reasoned that myself' frame (source_provenance_honesty.py:134-137). No requirement orders provenance/metacog to dominate the affect wrappers for derived answers.

- LATENCY OF A 4-HOP CHAIN IN THE LIVE LOOP. Each hop = a full RF resonate + a (K_shard x V) codebook cleanup (_cleanup_all) + a V-word answer cleanup, run SYNCHRONOUSLY in /api/brain-chat. A 4-hop chain is ~4x that at deployed 15k/real-V. No requirement sets a per-turn latency budget or a hop-cap/timeout; the orchestration-latency wall is the project's known real-time bottleneck, so a multi-second chain blocks the handler.

- PARSER-DECLINE PRE-EMPTS THE CHAIN. _extract_route returns '__DECLINE__' -> '__ABSTAIN__' when the on-brain BridgeParser cannot role-assign a >=2-content-word question (brain_chat_tui.py:716-719), and the has_self_alias gate (:714) diverts too. A genuinely compositional 3+word question can hit this abstain BEFORE any chain detector runs. Hardening #6 fixes the two-word pad truncation (:652) but not this SECOND abstain gate.

## Battery gaps (hardening requirements with no test that catches them)

- Hardening #2's ANSWER-ROLE floor is UNTESTED. No scenario teaches a SUPPORTED hop and asserts the patient/attribute decode returns the true stored value (or abstains on a thin margin) at 15k scale — every confab scenario probes UNSUPPORTED hops that abstain structurally. The supported-hop-wrong-patient channel (_render:557 bare argmax) has no battery test. (The soak's single-hop recall_rate exists but is not in the battery and is not chain-level.)

- Hardening #3 demands a false-positive RATE across MANY in-vocab near-miss cues; the battery offers ONE hand-built near-miss (scenario 2: 'fox eat' with a chase fact). A single passing example is not a rate. No swept-cue rate-measurement item exists, so the one number the floor-less gate's safety rests on is never actually produced by the battery.

- Hardening #10's GRADED confidence has no test. No scenario checks that an all-supported-but-THIN-MARGIN long chain returns a LOW-CONFIDENCE HEDGE (as opposed to a bare confident answer). The over-run scenario tests abstain-on-deadend, not hedge-on-degraded-but-complete.

- No LATENCY or DETERMINISM test: no 4-hop wall-clock budget at deployed 15k/real-V, and no chain insertion-order-independence test on a multi-valued (agent,action) hop (the idx[0] first-match, rf_phasor_composer.py:721).

- No MULTI-TURN / SINGLE-STORE test: a chain mixing an in-session-taught hop with an LTM hop, or a chain over a fact taught 5 turns ago (past the discourse-WM window), is never exercised — the chain scenarios teach both facts in the same turn.

- No INVERSE-RELATION-POSSESSIVE test ('what does the deer's predator eat?'). The battery's relation-noun coverage is the single forward-aligned 'prey', so an agent-direction possessive that query_chain cannot express is uncaught.

- No DOWNSTREAM-COUPLING test: that the DA suffix / affect+surprise prefixes / metacog hedge do not re-inflate a GENERATED-framed derived answer, and that metacog confidence reflects the whole chain rather than only the last hop's _read_activity().

## Honest scope

Read-only audit (no code changed by the workflow; no GPU; no live-handler warm). The compositional detector / parser / lemmatizer discussed are HOST scaffolds on the emergence-bar ladder to a learned replacement — declared as such, not relabeled as biology. The inference itself runs over the phasor store (the brain's own bound facts) = legitimate substrate. Source: workflow `wf_89e66a22-2cb`, 6 agents, 827k subagent tokens; per-agent journal preserved.
