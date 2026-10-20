# Word-role frame: learned question-frame role evidence (the word-role next method): build packet

Claude, 2026-10-20. The word-role flip was HELD on 10-18 (`research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md`,
AMENDMENT G READ; RULINGS wrc-hold-1018 and wrc-next-method-1018). At the flipped default the spiking commit declines
on "what does the cat eat" on gate seeds 42/43/44/101 and the reply abstains, while the flag-off positional parser
answers. The capability stays open (no-defer). This packet is the next method. Its prereg comes after step 2.

Evidence base: three read-only readers (`/home/dant123/Projects/sim-rev/wrc-next/readers.json`) and a panel of three
designers and two judges (`/home/dant123/Projects/sim-rev/wrc-next/panel.json`, workflow `wf_4a9719be-bf0`). The
mechanism follows the panel's biology-first design; the test battery follows its falsifiability-first design; the
exposure follows its integration-first design. Both judges asked for that combination.

## 1. What the decline actually is (verified 2026-10-20 on committed data)

- The organ never sees the question's frame: the host `_STOP` strip deletes what/does/the before any parser runs, so
  it reads `['cat', 'eat']` at positions 0 and 1 (`research/runners/brain_chat_tui.py:1490-1499, 1661-1664`).
- "cat" was taught once as patient ("dog chase cat") and once as agent ("cat eat fish"). Its word synapses drive both
  role ensembles equally (the Hebbian rule has no depression), and the position input is deliberately subthreshold.
- **Most declines are a gate-layer quench.** On the committed gate-seed organ data
  (`research/findings/raw/_s2p/word-role-commit/c64f2c7/O/s*.json`, `diagnostics.dual_role` against
  `dual_role_excl_lesion`), dual-role reads are 17/24 right intact and 21/24 with the commit gates' cross-inhibition
  lesioned, 0 wrong in either <!--derived-->. The lesion rescues cat@0 on seeds 43, 44 and 101 and cat@2 on 42. So the
  two partly driven gates silence each other. The 10-14 finding said the cross-inhibition was "not shown necessary";
  that holds for co-commits only, and a correction line is added there.
- What the real system runs here and the organ lacks: a held frame. A fronted wh-word and the auxiliary are kept in
  working memory while the clause unfolds, and that frame biases which role a reversible noun takes (top-down context,
  biased competition), learned from input statistics. Nothing in the organ persists between words except the latches.

## 2. The mechanism (flag `BRAIN_WORD_ROLE_FRAME`, default OFF; with it off the organ is byte-identical)

- **Every token is heard.** On the frame path the whole question, including what/does/the or any nonce marker, is
  presented in order with NO position code (pos=None). The host strip is not applied on this path.
- **Persistent traces.** Each word assembly drives a small trace population T_w, self-recurrent through slow NMDA
  synapses so it outlives the word (delay activity, the held filler). Traces are kept across the per-word reset, next
  to the slot latches.
- **Learned frame bias.** A new plastic projection T -> role ensembles, NMDA-routed so it is modulatory: weak on a
  resting role ensemble, additive with word drive.
- **Discharge on commit.** Every commit-gate burst drives a shared inhibitory pool that silences all traces:
  integrating an argument discharges the pending expectation. A token that commits a role silences its own trace, so
  only tokens that do not integrate (function words, markers) carry context forward. That split comes from the
  dynamics; no list draws it. Discharge also stops the frame from potentiating onto the ACTION role, or bleeding into
  the verb item and quenching it, which both judges found in the designs without it.
- **Entry point is decided by step 0, not assumed.** Role layer first (the frame need only quiet the patient side
  enough to stop its gate quenching agent). The gate layer (frame -> G_r through `SC.commit_plan` sources) stays a
  live option, not a fallback.

**Exposure, teach-free (the preferred arm).** Question episodes are presented with NO teacher current. Single-role
anchor words fire the roles they already learned from statements (semantic bootstrapping), and the pending traces
potentiate onto those. No host label from the answer text enters training. Episodes are a fixed, versioned teacher
fixture declared as developmental exposure, like the five built-in facts (`brain_chat_tui.py:2312-2320`), in the AI
teacher's lesson format (`research/runners/ai_teacher.py:23-33`). Statement training runs first; then word->role and
pos->role are frozen (a byte-identity hash before and after the question phase is a test) and only T->role learns.
Questions are still never stored as facts. A labelled-teacher arm (the answer's triple drives the role) is a declared
fallback only if teach-free fails, and is reported as a host residual if used.

**Plasticity notes.** Do not use `hebbian_mean_subtract`: the mean is subtracted before the plastic mask, so gain-0
synapses would drift (`sim/bridge.py` around 10101-10121). Report the saturating-Hebbian rule against a rate-window
plus BCM variant as a necessity ablation.

## 3. Build order with STOP lines

**Step 0, the deadlock read (read-only probe, no organ edit, numpy, dev seeds 1-15, about 15 min).** Exactly the
production organ (`WordRoleParserOrgan(sorted(facts), seed, *commit_organ_params(0))`) on the iw1 turn-5 fact set (the
five built-ins plus wolf-hunt-deer). Per seed record:
- (a) the production read of cat, eat: role, gate and slot counts per item, plus the role- and gate-inhibitory pools;
- (b) the same read with the exclusion lesion (this confirms the gate-quench finding on the production fact set);
- (c) cat with pos=None, and a pos=None commit of EVERY single-role word (dog, wolf, brain; chase, eat, hunt, use;
  deer, spikes, memory, words, fish);
- (d) a bias titration in cat's drive window, 0/50/100/200/400/800 pA, at pos 0 and pos=None, into (i) the agent role
  ensemble, (ii) the agent gate, (iii) negative current into the patient ensemble (competitor suppression): B_min (the
  smallest bias that latches agent alone) and B_false (the bias at which a patient-only word latches agent, or "eat"
  or an unheard spare latches anything);
- (e) "eat" still commits action with the agent bias held on;
- (f) the learned weights cat->agent/patient and fish->agent/patient, to explain cat@0 2/6 against fish@0 6/6.

GO rule, written before the probe runs: on >= 14 of 15 dev seeds, a role-layer B_min exists at pos=None with
B_false >= 2 x B_min, and every single-role word commits its own role at pos=None. If only the gate layer has a
window, the entry moves to the gate layer. If single-role words fail at pos=None, the pos-free path is not viable as
written: STOP and post the counts. Post `[wrc-frame-0] RESULT`; Claude rules before step 1.

**Step 1, the frame build (default off) and its early measurements.** In `WordRoleParserOrgan`, an additive
`frame=` build: the traces, the T->role projection at the entry step 0 chose, the discharge pool with its own
transmission-gate lesion, the trace-loop lesion, `_reset_keep_slots` extended for the traces. Measure first, on dev
seeds: the number of T->role co-firing events per presentation before discharge, and the NMDA frame gain under the
per-item role-conductance reset. If either is too small, use an AMPA frame WITH discharge (never drop discharge).
Identity tests at frame=None: BOTH pins, the commit battery (`tests/test_bg_commit.py:28-39`,
`A1_COMMIT_BATTERY_SHA256`) and count mode (`tests/test_word_role_commit.py:134-138`). Bring `wr_route.py` from
`research/wrc-flip-packet` (11a1513b9) so the route is persisted per turn.

**Step 2, the dev operating point, the cheat runs and bar sanity.** The grid (epochs of question exposure from
{4, 8, 12}, trace and frame gains) with the first-all-pass-row rule written before the sweep. Run the real cheats on
the nonce harness at dev seed 7: C1 the flag-off strip plus BridgeParser, C2 the commit organ without the frame, C3
`wh_question_parser.py`. Write `docs/drafts/word-role-frame/bar_sanity_spec_mm.py` (MM bars and O-REG) and
`bar_sanity_spec_reply.py` (R bars), each calling the real scorer, n >= 1000, with generators: perfect (5% item
error), perfect-slow (10% declines), null-positional, null-strip-BridgeParser, null-strip-unknown,
null-English-template, null-commit-no-frame, null-noun-prior, null-host-label-table, null-marker-claims-slot, random.
Every perfect >= 0.95, every null <= 0.05 on the composite; the record names which bar kills each null, and the
host-label-table null must die on MM3 and MM6. Plus `tools/answer_rate.py paired` smoke at dev seeds 7 and 8 on the
packet probes and iw1/iw2, with a probe cell ending on "what does the cat eat" (the Amendment G blind spot). A CI grep
confirms the nonce strings appear in no production module.

## 4. The made-up-marker battery (organ level, the capability test)

Both frames share the layout `[zim, M, the, N, V]`: "zim" replaces "what" (shared, uninformative), "dax" replaces
"does" (frame AG: N is the agent, the query the patient), and "wug" in the same slot marks frame PA (N is the patient,
the query the agent). "the" is kept. Statement facts (replayed as now): dog chase cat; cat eat fish; fish eat worm;
owl hunt mouse; mouse eat seed; hawk hunt owl; snake eat frog; frog eat bug; bug eat leaf; fox chase hen; hen eat
corn; boy kick ball; cow eat grass; girl read book; man ride horse.

Exposure, teach-free, K = 4 episodes per marker: dax episodes use agent-only anchor nouns (fox, boy, girl, man), wug
episodes patient-only anchors (corn, grass, ball, horse). Held-out test items (never seen in question form):
(a) six dual-role nouns (cat, fish, mouse, owl, frog, bug), each asked in both frames: 12 items, 6 PAIRS per seed;
(b) six novel-noun items ("zim dax the zorp eat"), which must commit nothing; (c) four single-role controls (dog,
hawk under dax; worm, seed under wug). A pair is right iff the noun takes the marker's role in BOTH items and the
verb commits action in both. Any positional rule, noun prior or the current organ reads both members of a pair the
same way and scores 0 pairs. A dose curve over K in {0, 1, 2, 4, 8} is reported, not gated.

Bars (gate seeds 42/43/44/100/101/102 through a `flip: no` prereg; final numbers fixed by the bar-sanity record):
- **MM1 intact:** pooled pairs >= 28/36; every seed >= 3/6; opposite-role noun commits <= 6/72; 0 slots claimed by a
  marker or "the" item; novel-noun commits <= 1/36; single-role controls >= 22/24.
- **MM2 permuted teaching:** the same facts and episodes with the markers swapped; the reading must follow the
  permuted pairing at MM1 thresholds.
- **MM3 frame lesion** (T->role gate closed at read, persistence verified): pooled pairs <= 6/36, and the lever moves
  >= 3/6 pairs on every seed. Engagement precondition: the marker's trace is ON during the noun item in >= 95% of
  items and w(T_marker -> implied role) >= 5 x w(T_marker -> other content role) on 6/6 seeds.
- **MM4 trace-loop lesion:** pooled pairs <= 6/36 (the frame state must survive the reset causally).
- **MM5 K = 0:** pooled pairs <= 6/36. **MM6 weight transplant:** only the T->role block of the permuted-taught
  organ copied into the intact-taught one on the same seed: the reading follows the donor, >= 28/36. A host label
  table cannot follow a synapse transplant.
- **MM7 discharge lesion (reported):** its effect on the verb read and on marker-claimed slots.
- **English double dissociation:** "what does the cat eat" on the marker-taught organ is no better than the K = 0
  baseline + 1; a shuffled-order exposure arm separates an order-sensitive trace from a bag of words.
- **O-REG:** the commit battery all-pass 6/6; patient-position dual-role reads (cat@2 + fish@2) >= 9/12 with 0 wrong;
  frame=None byte-identical (both pins).

## 5. Reply level: the flip v2 (a separate flip prereg, after the organ verdict)

Arms at the flip revision on iw1 and iw2: frame-on default, off:word-role, frame lesion, an intact repeat. Bars, from
the falsifiability-first design: R1 the AMENDMENT G rule verbatim (zero HURTS turns on all six gate seeds; HELPS never
offsets); R2 load-bearing on iw1 turn 5 (the frame lesion changes the reply, both repeats reproduce, 6/6, with the
route recorded); R3 single-role no-regression (iw1 t6, iw2 t0, iw2 t13 answers equal to off:word-role, 6/6); R4
identity (F2, F3 as in the flip prereg); R5 off-target statement turns unchanged under the lesion; R6 LBF rows
word_role_parse, word_role_commit and a new word_role_frame row load-bearing at the default; R7 answer-rate with no
FLIP-HURTS-ANSWERS. If MM1, MM2 or MM6 fail, the English flip is NOT filed: the method is banked and the next one named.

## 6. Calendar (honest)

- 10-21..10-22: step 0. 10-23..10-27: step 1 and step 2 on dev seeds.
- 10-27, hard go/no-go: the organ-level `flip: no` prereg is filed by then, or the Stage-2 pin carries word-role OFF.
  In both cases word-role defaults stay OFF at the 11-01 pin; Stage 2 is measured on the shipped parse.
- The organ-level gate seeds may run before 11-03 (numpy, CPU, cheap). The reply-level flip v2 runs after the Stage-2
  window (11-11..11-21) or the flip moves to the Stage-3 attempt (12-19). No-defer: a miss names the next method.

## 7. The same mechanism closes Stage-2 C1 later

RULING c1-learned-1016 asks for constructions learned from grounding or teacher exposure and tested with a nonsense
word in place of "by" or "in"; Stage-2 C1 is expected to fail honestly at the November gate because no such method
exists. The trace->role projection is that method: a passive "by" or a locative "in" is a marker whose held trace
biases the role of the noun that follows. After the wh-frame organ verdict, a C1 learned-construction prereg reuses
this battery's shape (same-layout pairs, permuted teaching, transplant, novel nouns) for "by" and "in". Target: the
Stage-3 attempt (12-19). `docs/drafts/learned-constructions/PLAN.md` (never written) is superseded by this section.

## 8. Who does what

Hermes builds on `research/wrc-frame`, posting `[wrc-frame-0]`, `[wrc-frame-1]` and `[wrc-frame-op]` RESULT lines.
Claude rules at each STOP, writes the preregs, reviews and merges. Priority: after cs-race steps 0-1, above C3.
