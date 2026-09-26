# Open honesty-check issues (opus re-verify, 2026-09-25)

Fix every item; re-check each number against the ledger and findings.

## 1. README.md (same framing in CHANGELOG.md and docs/CURRENT-STATE.md)

- Quote: "with adequate measurement-only probe fixes not yet shipped"
- Problem: Wrong framing of the headline pair. The B2a prereg (research/findings/2026-09-24-production-default-battery-B2a-PREREGISTRATION.md) says the 0.949 battery 'measures the default production brain as it actually ships': adequate probe set, no fix flag in the environment. Nothing has to ship to get 0.949. The two numbers measure the same shipped brain with two instruments, and the ledger's reporting standard calls the thin reading 'thin-probe UNDER-measurement (instrument artifact, not a brain limit)'. Two related problems: (a) the same wording appears in CHANGELOG ('probe fixes enabled (not shipped)') and CURRENT-STATE ('probe fixes not yet shipped'); (b) the 0.603 comes from the 2026-09-24 flipdefaults-thin battery. Only the 0.949 (b2a0924) was re-scored on 2026-09-25, yet all three docs attribute both numbers to 'a 2026-09-25 re-scored 6-seed battery'.
- Suggested fix: Suggested wording: 'Measured on the shipped default brain two ways: with the adequate measurement-only probe set, mean 0.949 (robust core 24 of 26 exercised, union 25, SD 0.018; 2026-09-25 re-scored B2a battery); with the older thin probe set, mean 0.603 (robust core 15, union 16, SD 0.018; 2026-09-24 flip battery), a known under-measurement.' Drop 'not yet shipped' / '(not shipped)' in all three docs.

## 2. README.md

- Quote: "A six-seed result makes that mood caused by a simulated body-state read by dedicated interoceptive neurons, and cutting those synapses makes the feeling stop tracking the body. Affect is load-bearing on the surface (the difference vanishes when the embodiment pathway is lesioned)"
- Problem: This conflates two separate affect reads and is stale since 2026-09-25. Forthcomingness and manner come from the Gate-B ladder (affect-coloring row; server.py ~L4960-5015). The body-state interoceptive #81 ladder is the affect-drives-response row, and the ledger says it is 'orthogonal to the Gate-B BRAIN_AFFECT prose-manner path'. So 'that mood' points at the wrong read. The #81 read's only effect on the reply was the prepended marker word, now OFF (BRAIN_AFFECT_MARKER_SURFACE default 0). Its embodiment lesion now shows only on the recorded affect_drives.lead field, not on the reply, so 'load-bearing on the surface' is no longer true. 'The feeling stop tracking the body' is also felt-state wording.
- Suggested fix: Split the two reads. Suggested wording: 'A separate body-state mood read (dedicated interoceptive neurons, six-seed GO; cutting those synapses stops the mood signal tracking the body) is still computed and recorded each turn and is lesion-verified on that recorded field. Since the marker word was retired on 2026-09-25 it no longer changes the reply text by default. The reply-level affect effect is the Gate-B mood read: lesioning its read-back collapses the forthcomingness and manner coloring.' Replace 'the feeling' with 'the mood signal'.

## 3. README.md

- Quote: "As of 2026-08-19 three of the brain's own signals were made load-bearing on the conversation itself: its mood colors phrasing, its thought-swap decision steers which topic the turn engages, and its dopamine mode sets how engaged the reply is."
- Problem: Stale, and left in an edited row. The 2026-08-19 mood coupling (board #84, affect-drives-response) was the prepended marker word, retired from replies on 2026-09-25. swap-drives-response is load-bearing on 0/6 seeds in both current batteries (per-faculty table of the B2a re-scored finding), and its adequate probe is verified on 1/6 seeds (2026-09-24 status finding). Of the three signals, only the dopamine mode (6/6) still holds as stated.
- Suggested fix: Date it and update it. Suggested wording: '...of these, the dopamine mode is load-bearing on 6/6 seeds today; the mood marker was retired from replies on 2026-09-25 (mood still shapes forthcomingness and manner via Gate-B); the thought-swap lead is 0/6 in the current load-bearing battery.'

## 4. README.md

- Quote: "Default-on and spiking: a claim-level no-confabulation moat that drops ungrounded content"
- Problem: Overclaim, and it contradicts USER_GUIDE's new paragraph, which says the moat combines spiking recall with host machinery. Per the ledger, only the per-clause role parse is on the substrate. moat-verify: 'host keyword abstain gate + host clause decomposition/coverage bracket the on-substrate per-clause role-parse'. open-ended-generation and tiered-knowledge-ltm: the RF-composer exact-inverse moat is host.
- Suggested fix: Move the moat out of the 'Default-on and spiking' list. Suggested wording: 'a default-on claim-level no-confabulation moat (its per-clause role parse is on the spiking substrate; the abstain gate, clause decomposition and the RF-composer exact-inverse check are host code)'.

## 5. README.md

- Quote: "As of 2026-09-05 four core thinking organs — surprise, the forward world-model, self-monitoring, and phrasing — share one literal neural pool by default"
- Problem: Stale, and it contradicts docs/CURRENT-STATE.md. The Wave-3 pool has been default-on since 2026-09-19 (_WAVE3_POOL_DEFAULT_ON=True, commit 8ee5e6817): 11 organs validated in one pool, 8 routed through it on the default chat path (ledger onebrain-merge-organs). CURRENT-STATE states 11/8; README still gives the older 4-organ picture as the current state.
- Suggested fix: Replace with the 2026-09-19 state. Suggested wording: 'Since 2026-09-19, eleven cortical organs are validated together on one shared spiking pool and eight are routed through it on the default chat path. Most organ-to-organ boundaries are still host relays (co-residency on one pool, not one self-organized substrate); two learned cross-edges are default-on.'

## 6. docs/CURRENT-STATE.md

- Quote: "spans two separate spiking bridges rather than one merged brain"
- Problem: Stale text left below the refreshed header contradicts it. The new disclaimer covers only the 'Demonstrated' and 'Partially Achieved' tables. These sections are outside it and are written as current state: Main Blockers #6 (this quote), 'Highest-Value Work' ('merge the two co-resident spiking bridges into one brain (the "one brain" step)', and the priority framed as the fourteen-turn chat rather than the load-bearing fraction). Inside the tables, the Conversation row ('The language "mouth" is a conventionally trained spiking model kept as an articulation scaffold') and the 'A shared brain' row ('two separate spiking bridges ... Merging ... is the named next arc') directly contradict the new permanent-Qwen and 11/8 shared-pool paragraphs.
- Suggested fix: Either re-audit Main Blockers #3/#6 and Highest-Value Work to the 2026-09-25 state (Wave-3 pool, permanent Qwen mouth, load-bearing fraction as the #1 metric), or add dated 'as of 2026-08-11' markers to those sections too. At minimum, strike the 'merge the two co-resident bridges' next step, since that merge landed.

## 7. CHANGELOG.md

- Quote: "a separate later-learning-interference variant is NO-GO on 3 of 6 seeds — on the 2 failing seeds the fact was told and stored with a normal synaptic weight"
- Problem: Internal contradiction: three seeds failed, but the text says 'the 2 failing seeds'. Per research/findings/2026-09-25-sleep-forgetting-interference-fi-6seed-NO-GO.md (n_go=3), seeds 43 and 101 failed on a night-1 capture-margin miss, and seed 100 failed FI6 (a twice-re-mentioned fact lost by night 4). The 'stored, not never stored' part is correct per the finding (L129). The orchestrator brief's 'never stored' is wrong, so keep that part.
- Suggested fix: Suggested wording: 'NO-GO (3 of 6 seeds pass): on 2 failing seeds (43, 101) the weak fact was stored with a substantial weight but its night-1 reactivation missed the capture margin; the third (100) lost a twice-re-mentioned fact by night 4.'

## 8. CHANGELOG.md

- Quote: "A load-based (rather than fixed-percent) nightly forgetting rule for unimportant facts is designed and built but not yet default-on or 6-seed-validated end to end."
- Problem: This is the same mechanism as the 'later-learning-interference variant' earlier in the same bullet: BRAIN_SLEEP_LOAD_RENORM, the fi family. It has been 6-seed tested and is NO-GO 3/6. Saying it is 'not yet ... 6-seed-validated' implies it is untested, and it double-counts one mechanism as two. The prioritized-memory design ('remember what matters') is design-only with open review issues and is not mentioned.
- Suggested fix: Delete the sentence, or fold it into the fi clause, e.g. 'the load-based nightly renormalization (BRAIN_SLEEP_LOAD_RENORM, default OFF) is that variant'. Optionally add: 'a prioritized-memory design (remember what matters) is written, with open review issues; the capture pair waits on it.'

## 9. CHANGELOG.md

- Quote: "a night-load-dependent synaptic-tagging mechanism distinct from the basic offline sleep-replay consolidation"
- Problem: Two problems. (a) Wrong mechanism label: BRAIN_SLEEP_REPLAY_CAPTURE is sleep-replay-triggered tag-and-capture. At sleep onset one SWR epoch re-tags DA-tag-capture blocks and drives the D1/PRP pools (rc finding). The 'night-load-dependent' part is the separate fi renormalization flag. (b) 'independently re-checked' omits the verify-go review's verdict. That review (2026-09-25-da-capture-sleep-replay-pair-verify-go-review.md) found SOUND-WITH-ISSUES: the pair's flip rationale does not hold as stated (a fact told 4 h before sleep is not rescued on 6/6 seeds, while today's default keeps it), and the production path was never exercised.
- Suggested fix: Suggested wording: 'a sleep-onset replay that re-tags and captures recently written DA-tagged facts (distinct from the default-on offline sleep replay)'. Then add: 'the independent review re-graded the GO exactly but found the pair not yet flip-ready (4-hour-delay facts not rescued 6/6; production path unexercised)'.

## 10. CHANGELOG.md

- Quote: "has merged (2026-09-23/24). Whether it carries over to full network size is still open: a full-size-network run landed exactly at chance on every repeat as of 2026-09-24."
- Problem: Three problems. (a) Wrong date: the merge is 2af73bfdd, 2026-09-25 02:42. (b) It omits that the fix merged as a default-OFF engine knob (cfg.bdsp_pbar_ratio_tau_ms, byte-identical off). (c) The cited full-size run is the wrong run: the 2026-09-24 run predates the fix (C21 flags). The current result is the 2026-09-25 C26 full-size run with the sliding baseline (research/findings/2026-09-25-gap4-sliding-baseline-c25-c27-and-c26-fullsize-UNDEFINED.md). It removes the clamp saturation at full size (0/3 replicates past threshold), but held-out accuracy never clears chance with headroom (0/3 interpretable), and the training-fit gain transfers on only 1/3. Verdict: UNDEFINED, dev seed 7 only.
- Suggested fix: Suggested wording: 'merged 2026-09-25 as a default-OFF engine knob. At full size (2026-09-25, dev seed 7) it removes the weight pile-up (0/3), but the read-out is still not interpretable (0/3, UNDEFINED); no evaluation-seed run yet.'

## 11. CHANGELOG.md

- Quote: "A harder awake-rest extension of the same route (a fact told hours before sleep, not immediately before it) passes on 5 of 6 seeds"
- Problem: This leaves out the registered verdict. research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md is 'NO-GO 5/6': seed 101's rescue did not fire, and the bar is 6/6. It is also a separate default-OFF flag (BRAIN_AWAKE_REPLAY_CAPTURE), not the capture route itself. Read on its own, 'passes on 5 of 6 seeds' sounds like a pass.
- Suggested fix: Suggested wording: 'A separate awake-rest replay (BRAIN_AWAKE_REPLAY_CAPTURE, default OFF) for facts told ~4 h before sleep is NO-GO: 5 of 6 seeds pass, and seed 101's rescue did not fire.'

## 12. README.md

- Quote: "each verified byte-identical against main"
- Problem: This overclaims for one-brain-substrate. Only the four 2026-09-16 retirements were checked by a byte-identical branch-vs-main /api/brain-chat differential. one-brain-substrate was a 2026-09-02 'proof retirement' (the rf _scan demoted to an opt-out oracle). Its lesion_note says the claim is a mechanism claim proven by an activity trace, not an answer diff. CHANGELOG words this correctly ('Four faculties ... joining the earlier one-brain-substrate retirement').
- Suggested fix: Suggested wording: 'the four 2026-09-16 retirements each verified byte-identical against main; one-brain-substrate was retired 2026-09-02 by demoting the host rf scan to an opt-out oracle'.

## 13. USER_GUIDE.md

- Quote: "language model it graduated from (Qwen2.5-0.5B) is treated as a **permanent**"
- Problem: This contradicts itself. 'Graduated from' says the brain has moved past Qwen, while the same sentence says Qwen is the permanent mouth. The wording is left over from the retired 'scaffold being phased out' framing.
- Suggested fix: Replace 'the external language model it graduated from (Qwen2.5-0.5B)' with 'the external Qwen2.5-0.5B-Instruct language model'.

## 14. USER_GUIDE.md

- Quote: "(Qwen2.5-0.5B renders the reply by default; the brain supplies and verifies the content; needs a GPU)"
- Problem: Overclaim, and it contradicts this file's own later paragraph. That paragraph says the no-confab check combines spiking recall with host machinery that is not yet neural. The removed prose had the same 'brain supplies and verifies' phrasing; it was reintroduced here in the command comment. The comment also omits that a bounded single-role SVO recall is spoken by the default-on spiking recall mouth.
- Suggested fix: Suggested wording: '(Qwen2.5-0.5B phrases most replies by default; the brain supplies the content, which a partly-host no-confab check verifies; needs a GPU)'.

## 15. USER_GUIDE.md

- Quote: "which fact to surface is decided by the spiking substrate"
- Problem: True for the question parse and for the conversation-buffer recall (ledger content-selection). The production default also answers from the ~79k-fact LTM tier, though. For any recall that falls through to that tier, the ledger's semantic-recall row (corrected 2026-09-16) says the VSA unbind (host np.conj) and the cleanup selection (host np.argmax over the matched-filter membrane) still run on the host by default. So for bulk knowledge, the final pick of which fact to surface is host code.
- Suggested fix: Add a qualifier: '...decided by the spiking substrate for the question parse and the recent-conversation store; for the large background knowledge base, the final cleanup selection is still a host argmax (ledger row semantic-recall).'

## 16. USER_GUIDE.md

- Quote: "The fluent prose you read is phrased by the external Qwen2.5-0.5B language"
- Problem: This is inconsistent with the other three docs, which state that a bounded spiking recall mouth is default-on. The ledger row spiking-mouth-recall is on_by_default YES (_RECALL_MOUTH_DEFAULT_ON=True): it speaks in-frame single-role transitive-SVO recalls such as 'the brain uses the spikes'. The paragraph also omits that GPU-free hosts get the template stub, not Qwen (server.py _default_brain_renderer).
- Suggested fix: Add one sentence: 'A narrow spiking renderer speaks simple single-role subject-verb-object recalls directly (default-on). Without a GPU, a multi-sentence template stub replaces Qwen.'

## 17. CHANGELOG.md (also README.md 08-26 list)

- Quote: "basic offline sleep-replay consolidation has been default-on since 2026-08-26"
- Problem: This fails docs/TERMS.md. 'Consolidation' requires that the trace survives a lesion of the source structure. The ledger's own sleep-replay-consolidation row says it is 'Not "consolidation" in the docs/TERMS.md systems sense ... the claim is DIRECT store retention'. README's 08-26 list has the same wording ('offline sleep-replay consolidation').
- Suggested fix: Suggested wording: 'offline sleep replay that strengthens recently stored episodes (direct store retention) has been default-on since 2026-08-26'. Use the same wording in README.

## 18. CHANGELOG.md

- Quote: "isolated-lesion-load-bearing since 2026-08-12 but found integrated-hollow until a probe fix on 2026-09-20"
- Problem: Misleading as a worked example. The ledger's episodic_correction says the hollow reading was a PROBE ARTIFACT (the probe queried an empty session); the brain never changed. The fix was a default-off measurement flag (LB_EPISODIC_DRIVE_PROBE), and episodic-memory is still outside the thin-probe robust core. As written, it reads as the brain being hollow and then fixed.
- Suggested fix: Suggested wording: '...its integrated-hollow reading turned out to be an instrument artifact (the probe asked about an empty session); a store-then-recall probe (2026-09-20) shows it load-bearing, 6/6 in the 2026-09-25 adequate battery.'

## 19. CHANGELOG.md

- Quote: "the brain preferentially keeps a counterfactual or lie it was actually told over an unstated "true" fact"
- Problem: Overstated. T5/T6 in research/findings/2026-09-24-ai-teacher-chat-only-synaptic-learning-GO-6seed.md show the brain encodes what the chat channel said: a teacher-asserted counterfactual, and teacher errors on corrupted facts while clean ones are kept. They do not show it read off a hidden ground truth. There was no preference test between competing told facts, and 'lie' implies intent the teacher code does not have.
- Suggested fix: Suggested wording: 'what gets encoded is what the teacher actually said (a counterfactual, or an error on corrupted facts), not a hidden ground truth'.

## 20. CHANGELOG.md

- Quote: "nowhere near a VRAM limit"
- Problem: Category error. The capacity finding measured peak host RSS on pool CPU nodes (124-233 MB) against a 24 GB memory bar; there was no VRAM on those runs. The CPU-only framing and the 'RTX 3090 unmeasured' statement are correct per the finding. The orchestrator brief's 'on one GPU' is wrong, so keep those parts.
- Suggested fix: Replace with 'nowhere near the 24 GB memory bar'.

## 21. docs/CURRENT-STATE.md

- Quote: "validated together on one shared spiking pool, with 8 of them routed through"
- Problem: The numbers are accurate, but the co-residency caveat is missing. Most organ-to-organ boundaries are still host relays; the pool is shared neuron state, with only two learned cross-edges default-on. README carries this caveat ('still co-residency'); CURRENT-STATE does not, so it can read as one integrated substrate.
- Suggested fix: Add: 'this is co-residency on one pool: most organ-to-organ exchanges are still host relays, with two learned cross-edges default-on (BRAIN_ONEBRAIN_XEDGE, BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6).'

## 22. README.md (same in docs/CURRENT-STATE.md)

- Quote: "beat a simple word-pair baseline on simple text at a deployable size"
- Problem: Imprecise. The finding (2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-...) is against 'a fair interpolated trigram', which most readers will not take as a 'word-pair' (bigram) baseline. The milestone is dated 2026-09-03; README says 'As of 2026-09-04' (the broad-domain below-trigram result is 09-04).
- Suggested fix: Suggested wording: 'beat a simple three-word (trigram) statistical baseline on simple text (2026-09-03) but fell below it on broad text (2026-09-04)'.

## 23. README.md (same in docs/CURRENT-STATE.md)

- Quote: "As of the 2026-09-16 ledger head"
- Problem: Minor date inaccuracy. The ledger head is 2026-09-25 (commit 969894855). The 69/30/5 counts have been unchanged since 2026-09-16: the 2026-09-23 sub-rung flips explicitly left them unchanged.
- Suggested fix: Suggested wording: 'As of the current ledger (2026-09-25; counts unchanged since 2026-09-16)'.

## 24. USER_GUIDE.md

- Quote: "5 of 69 tracked faculties (of 30 default-on spiking"
- Problem: Confusing. It reads as 5 out of both 69 and 30, which implies the retired five are a subset of the 30. 'Verified against the spiking substrate' also misstates the check: it was a byte-identical branch-vs-main answer differential (and a demotion to an oracle for one-brain-substrate).
- Suggested fix: Suggested wording: 'the ledger tracks 69 faculties, 30 of them default-on and spiking; 5 have had their host scaffold fully retired'. Drop 'verified against the spiking substrate' or replace it with 'verified by a branch-vs-main answer comparison'.

## 25. CHANGELOG.md

- Quote: "with nine more faculties merged off-by-default pending their own 6-seed tests"
- Problem: Minor overcount of 'faculties'. ROADMAP's 2026-09-24 evening entry says 'nine more pieces merged', and several are instruments rather than faculties: a faster warm start, a learned-content provenance test, the multi-turn settle test. Also, 3 of the 49 battery rows are parked (self-schema retracted; false-belief and transitive-reasoning are default-off capability rows).
- Suggested fix: Suggested wording: 'nine more pieces (faculties and measurement instruments) merged off by default', and note 'three of the 49 rows are parked'.
