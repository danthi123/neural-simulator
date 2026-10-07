# Spiking commit: one reusable all-or-none decision stage, and its first application (word-role-parse)

*Claude, 2026-10-06, branch `claude-review/spiking-commit`. Phase: prove who owns the computation
(`docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md`). Runbook RULINGS this answers:
`word-role-verify-go`, `word-role-credit` (the spiking all-or-none commit that replaces ROLE_MIN / margin and the
per-role argmax), `iw1-wmf-hold` (the question cue biases the referent competition). Preregistration for A1:
`research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md`. Hermes steps: `HANDOFF.md` in this folder.*

## 1. The defect: one pattern at four seams

This week's reviews narrowed four results the same way: the reply depends on a spiking path, but host code makes the
final choice between the spiking alternatives.

| seam | spiking part | host part that decides | record |
|---|---|---|---|
| word-role-parse | role-ensemble spike counts per word | `count >= ROLE_MIN and >= 2 x runner-up`, then the highest-count word per role | finding 2026-10-06-word-role-parse-gate, caveat 1 |
| wm-focus-bind | 5-way lateral-inhibition WTA over held referents | argmax over WTA rates with `FOCUS_WTA_MIN_RATE` and `FOCUS_WTA_DEAD_MARGIN`; the race ignores the question | `d6_multiref_wm_production_organ.py` `_retrieve`; RULING iw1-wmf-hold |
| Stage-1 C2 / C3 | recruit pool, lived-memory recall | host picks the word / the fact (k-select); C3 host share 0.967 | 2026-10-04-stage1-exit-gate-RESULT.md, Claude review |
| d6-hebbian-write | Hebbian block write | a host threshold and a block-to-words map | RULING iw1-R |

The same read appears in the GNW bus (`norgan_hop`: argmax over slot rates + threshold), the affect-marker and
question-route WTAs (argmax + floor + dead margin) and the navigation commit (argmax over `commit_X` counts with a
`sel_X` argmax fallback). A competition is spiking; its outcome is chosen by a comparison in Python.

## 2. What the record already has (read before building)

`bash tools/before_you_build.sh "host argmax/threshold decides between spiking alternatives"` and the RAG, then the
sources:

- **Navigation commit burst** (`g11_bg_runner.py`, finding `2026-06-19-spiking-decision-default-on-GO.md`): Wang-2002
  accumulators `sel_X` drive all-or-none burst pools `commit_X` (Lo & Wang 2006 SC threshold); default-on for
  navigation, 100% of decisions by the burst. But the read is still `max(range(N), key=commit_counts)` with a
  fallback to the `sel_X` argmax and a random last resort: the burst makes the argmax nearly idle, the code still
  compares counts. A constant omnipause (OPN) gate rebound-bursts on this substrate, so it is off.
- **GNW rung-1 ignition** (`_gnw_rung1_ignition_curve_derisk.py`, finding `2026-07-07-GNW-rung1-...-GO.md`): a dense
  self-recurrent assembly either dies out or latches a self-sustaining state after the drive stops; settled rate
  0.333 ignited vs <= 0.053 not (bimodal), lesion of the recurrence removes it, AMPA-only recurrence also ignites.
  This is the all-or-none primitive with the cleanest read: persistence after input ends.
- **GNW N-organ bus** (`webapp/gnw_bus_shadow.py`, default on in production): consensus ignition over K slots with
  shared inhibition; read = argmax over occupied slots + `THR`. A default-off "unique ignition read"
  (`BRAIN_GNW_UNIQUE_IGNITION_READ`) commits only if exactly one slot ignited.
- **BG live-seam retirement** (`2026-10-02-bg-live-seam-retirement-RESULT.md`): **NO-GO on G4**. A per-channel latch
  read at a fixed rate changed its committed action on 25% of events when the read threshold moved x0.5 / x2: the
  host read still owned part of the commit. Lesson carried here: an all-or-none stage must be *measured* all-or-none,
  by re-reading the same spikes at other read levels, and the read should be of persistence, not of a graded rate.
- **Lateral-inhibition WTA primitive** (`_affect_marker_wta_derisk._build_bridge`, reused by question-route and the
  d6 focus WTA): N assemblies, each with its own fast-spiking pool inhibiting the others. A good competition; every
  user reads it with argmax + floor + margin.
- **gap#3 biased competition** (2026-06-17 NEGATIVE; a 2026-07-21 rate-rung GO, since retracted:
  ⛔ `2026-07-21-gap3-biased-competition-multireferent-6seed-GO.md`): content bias into a WTA. The cue-bias idea is
  sound; the spiking form never landed. A2 below is its spiking form, with the commit stage as its read.

**Choice.** No new kind of circuit. The commit stage composes two pieces this substrate has already shown to work:
the exclusive burst gate of the navigation commit and the bistable latch of GNW rung 1, wired into the SAME bridge as
the competition it reads (no cross-bridge relay), and read by persistence with a registered threshold-invariance
bar (the G4 lesson). It replaces the comparison, not the competition: the competing populations (role ensembles,
focus assemblies, slots) stay as they are.

## 3. The mechanism

For K alternatives, each with a source population `X_k` that already competes (lateral inhibition upstream):

- **Commit gate `G_k`** (NG excitatory Izhikevich neurons per alternative). `X_k -> G_k` dense excitation (`W_RG`),
  `G_k -> G_k` recurrence (`W_GG`: a regenerative burst once enough of `G_k` fires), `G_k -> GI_k` (an inhibitory
  pool of NGI) and `GI_k -| G_j` for every j != k (`W_GIG`). Within one presentation at most one gate bursts.
- **Latch `S_k`** (NS excitatory per alternative). `G_k -> S_k` (`W_GS`), dense `S_k -> S_k` recurrence (`W_SS`) strong
  enough to self-sustain after its input stops: a bistable assembly (persistent activity). Latches do not inhibit
  each other, so several alternatives of DIFFERENT slots can be held at once (an agent and an action).
- **What makes it all-or-none.** Two thresholds in series, both neural: the gate's knee (input weight x intrinsic
  excitability, sharpened by recurrence and cross-inhibition) and the latch's bistability (it either holds itself
  after the drive ends or decays to zero). The read happens only after a zero-input hold, when an un-latched
  assembly has died out; graded input cannot survive to the read.
- **Read (a transducer, not a comparison).** Over the last READ_STEPS of the hold, a latch is ON iff its spike count
  reaches `READ_FRAC x NS x READ_STEPS`. The outcome is WHICH latch turned on during the presentation: exactly one ->
  that alternative; none -> abstain; two or more -> co-commit (abstain, counted). No count is ever compared with
  another alternative's count. The same spikes are re-read at `READ_FRAC x0.5` and `x2`; the registered bar is that
  the outcome does not change (on dev seeds 1-8 at row 0 the latched minimum is 77-84 spikes, the unlatched maximum 0:
  the read is invariant from about 1 to 77 spikes).
- **Sequential decisions.** For a sequence of presentations (the words of a question), the inter-item reset spares
  the latches: a role filled by an earlier word stays filled and cannot turn on again, so the first presentation to
  commit an alternative keeps it. This replaces a per-role argmax over items with incremental assignment.
- **Abstain.** Nothing latched within the window. The moat (an unheard word gets no role) becomes a latch that never
  ignites, not a host floor.
- **Biasing inputs.** A cue population `Q` projects onto the competing sources `X_k` (or onto `G_k`) with
  cue-matched synapses (`Q_c -> X_k` strong where alternative k matches cue c). The bias shifts which source wins
  the competition and therefore which gate bursts; it never touches the read. A2 uses this.
- **Lesions (each a transmission gate on the same bridge).** `role_commit` (`X -> G`): nothing commits, the sources
  still fire exactly as intact (verified: identical role counts). `slot_loop` (`S -> S`): latches cannot hold; the
  read finds nothing. `commit_excl` (`GI -| G`): exclusion off; co-commits become possible.
- **Parameters.** All wired, none learned: NG 12, NGI 8, NS 20, `W_RG` 50, `W_GG` 20, `W_GGI` 60, `W_GIG` 60, `W_GS` 12,
  `W_SS` 60, hold 100 ms, read the last 30 ms, `READ_FRAC` 0.03. AMPA only (a slow-NMDA variant was tried on dev
  seed 7 and was unstable: Mg-block overflow, non-monotone in weight; banked).
- **Where the threshold went.** The host constant ROLE_MIN became the gate's input weight `W_RG`. On dev seeds the
  sensitivity is the same thin margin the role layer always had (heard words 6-34 role spikes, unheard 0-7): `W_RG`
  40 drops some heard words, 60 commits one unheard word on one seed. The commit does not create information the
  role layer lacks; it moves the decision into the substrate and makes its read threshold-invariant.

### Biology

- Decision commitment as a threshold crossing in a downstream population, separate from the accumulating one:
  superior colliculus / FEF movement neurons (Hanes & Schall 1996, Science 274:427; Lo & Wang 2006, Nat Neurosci
  9:956) and the LIP-accumulate / SC-commit split shown causally by Stine, Trautmann, Jeurissen & Shadlen 2023
  (Neuron 111:2601; SC inactivation removes commitment, not evidence).
- Winner-take-all by recurrent excitation and shared or cross inhibition: Wang 2002 (Neuron 36:955); leaky competing
  accumulators (Usher & McClelland 2001, Psychol Rev 108:550).
- All-or-none ignition of a recurrent assembly: Dehaene & Changeux 2011 (Neuron 70:200).
- Bistable persistent activity holding a chosen item after the stimulus ends: Goldman-Rakic 1995 (Neuron 14:477);
  Compte, Brunel, Goldman-Rakic & Wang 2000 (Cereb Cortex 10:910).
- Top-down bias of a competition by a task cue: Desimone & Duncan 1995 (Annu Rev Neurosci 18:193).
- Incremental assignment of thematic roles as words arrive, with early commitment and later revision costs:
  Bornkessel & Schlesewsky 2006 (Psychol Rev 113:787).
A `research/biology/` entry is written with the first finding (it needs a `current_finding`).

### Reuse

A1 implements the stage inside `WordRoleParserOrgan` (built only with `commit=` params: `COMMIT_GRID`,
`_commit_run`, `_commit_read`, `commit_parse_from_items`). A2 is the second user; it factors the wiring
(`G`, `GI`, `S` plan entries for K sources on any bridge) and the read (`_commit_read` on recorded latch counts) into
`research/runners/spiking_commit.py`, and A1 imports it there with a byte-identity test. One copy, two users, before a
third.

## 4. Application order

**A1 word-role-parse (built, this branch).** Replace ROLE_MIN / ROLE_MARGIN and the per-role argmax. Flag
`BRAIN_WORD_ROLE_COMMIT` (default OFF; only read when `BRAIN_WORD_ROLE_PARSE` is on), row `BRAIN_WORD_ROLE_COMMIT_OP`,
lesion `BRAIN_WORD_ROLE_COMMIT_LESION`. Section 5 is its preregistration summary.

**A2 wm-focus-bind (next; Claude writes the packet after A1's step O).** Today the held registers drive the focus WTA
through a host rate relay from another bridge (`FOCUS_WTA_GAIN_PA x rate`), the race ignores the question, and the
winner is an argmax with a rate floor and a dead margin; flag-on abstains on 2 of 4 sessions the default answers.
Method: (a) build the focus WTA on the D6 organ's own bridge so the held bumps drive it through synapses (removes the
relay); (b) add a cue population driven by the question's content word (the anaphor's feature: e.g. animacy or the
verb's selectional preference) with cue-match synapses onto the focus assemblies, so the question biases which
referent wins; (c) put the commit stage on the focus assemblies and read the latch. Registered tests: the held
referent changes with the question at a fixed history (cue swap), the cue lesion returns the per-seed preference, the
commit lesion abstains, and the answer rate on the iw1 probe is not below the default (RULING flip-hurts-answers).

**A3 (later).** C3 lived-memory fact selection (k-select, host share 0.967): candidate facts as competing slot
assemblies driven by recall strength, committed by the stage. d6-hebbian-write: the write threshold as a commit
latch on the write-enable population, and the block-to-words map as a learned readout. Each gets its own
preregistration; none is built before A2's step O.

## 5. A1 preregistration (summary; the registered copy is the PREREGISTRATION file)

- **Seeds.** Step 1 (operating point): COMMIT_GRID rows 0-5 on dev seeds 1-8; selection = the first row in grid order
  with `all_pass` on all 8 (`tools/s2p_score.py select --packet word-role-commit`). Gate seeds 42/43/44/100/101/102 for
  steps O and R, never before the prereg is committed.
- **Arms.** Intact (commit on); mechanism lesioned (`BRAIN_WORD_ROLE_COMMIT_LESION=1`: role -> commit gate cut, role
  layer intact); host-rule baseline, reported not gated: (i) organ level, the count rule re-applied to the SAME run's
  role counts (`host_rule` per case, `host_rule_roles` per reply), (ii) reply level, the word-role-parse R data at
  9ea7411 (same probe, same seeds, count mode). Organ-level ablations in every battery: word -> role cut, commit cut,
  latch cut (all must decline); exclusion cut on the dual-role reads (reported).
- **Step O (organ) bar.** PASS iff on 6/6 gate seeds every battery case is right (svo, self, adverb, swapped, novel,
  word-lesion, commit-lesion, latch-lesion) AND every case's read is invariant at READ_FRAC x0.5 and x2.
- **Step R (reply) bar.** PASS iff on 6/6 gate seeds the commit lesion changes the reply to "what do you usually
  use?" (intact states the self fact with 'spikes'; the two intact arms agree; the lesion reproduces; the lesion
  reply drops the content) AND the control "what do you usually drink?" abstains in both arms on 6/6, under every
  s2p step-R rule.
- **Engagement (required in every R arm).** `word_role_trace.route == "word_role_commit"`, `error` null,
  `commit_lesioned` true in lesion arms and false in intact arms (scorer: `engagement.lesion_key`); the trace also
  carries `commit.per_item`, `commit.read_invariant`, `commit.read_variants`, `commit.co_commit`, `commit.relatch`,
  and `commit.host_rule_roles`. A cell without it is UNDEFINED.
- **Outcome -> verdict.**

| step 1 | O | R | verdict | next |
|---|---|---|---|---|
| NONE | - | - | method banked: no row passes dev 1-8 | NEEDS-CLAUDE (no-defer: next method) |
| SELECTED | PASS | PASS | PASS: the commit stage decides the parse and the reply depends on it | Claude verify-go; then the flip packet |
| SELECTED | PASS | FAIL | NO-GO at reply level | NEEDS-CLAUDE |
| SELECTED | FAIL, a case | - | NO-GO at organ level (the failing case names it) | NEEDS-CLAUDE |
| SELECTED | FAIL, read only | - | NO-GO: the read owns part of the decision (the G4 failure) | NEEDS-CLAUDE |
| any | UNDEFINED | UNDEFINED | no verdict | re-queue the missing cells once, then NEEDS-CLAUDE |

- **FULL credit for word-role-parse** (RULING word-role-credit) requires all of: O PASS and R PASS here; the host
  count rule and argmax absent from the decision path with the flag on (pinned by
  `tests/test_word_role_commit.py::test_host_rule_cannot_change_a_commit_parse` and the commit-lesion test); then a
  flip packet making `BRAIN_WORD_ROLE_PARSE` and `BRAIN_WORD_ROLE_COMMIT` default on, whose F1 at the default passes
  and whose answer-rate check shows no FLIP-HURTS-ANSWERS. The declared residuals stay residuals: training stimuli
  transcribed from the fact index, self-alias resolution, stop-word strip, the word-form dictionary, the latch read
  (a threshold-invariant transducer), the inter-word reset that spares the latches, and the organ being its own
  bridge (one-brain debt).
- **bar-sanity** (n 1000 per brain, `tools/bar_sanity.py`). Organ bar:
  `docs/drafts/spiking-commit/bar_sanity_record_a1_organ.json`: perfect 1.000, perfect_noisy 1.000; null_positional,
  null_lesion_inert, null_graded (no latch) and random 0.000. Reported: a weak latch (N(50, 6) slot spikes) passes
  0.643, because the bar asks every latched read to stay above the x2 read level (36 spikes): the bar demands a
  factor-of-two gap on both sides, which the dev seeds show (77-84 vs 0). Reply bar:
  `docs/drafts/spiking-commit/bar_sanity_record_a1_reply.json`: perfect 1.000; lesion-inert, wrong-content,
  confabulating-control and unengaged-lesion brains 0.000; random 0.001. Reported: if the two intact arms disagree on
  a seed with probability 0.1, the step reads PASS 0.508 (the rest UNDEFINED, by the existing null-noise rule).
  The specs call the real decision code; the reply spec replaces only the shard-validity reader (`judge_dir`).

**Dev smoke of the R probe (organ level, dev seeds 1-8, no handler).** An organ trained on the tiny-demo brain's five
build facts (`_build_tiny_demo`: brain use spikes / learn words / store memory, dog chase cat, cat eat fish) at row 0:
[brain, usually, use] -> (brain, use) on 8/8 with an invariant read; [brain, usually, drink] -> (brain, None) on 8/8
(declines); the commit lesion -> (None, None) on 8/8; the count organ at GRID 0 gives (brain, use) on 8/8. This
assumes the handler's fact index at the probe is those five facts; step R measures the real one.

## 6. What A1 will not show

That the stage is learned (it is wired); that the role layer is better (the commit reads it, it does not improve
it: dual-role words read right 18/32 at row 0 on dev seeds, against 17/32 for the count organ at GRID 0); that
exclusion is necessary on this battery (the exclusion lesion added 1 co-commit across 32 dev dual-role reads: the
role layer's own inhibition already separates them; reported, not claimed); anything about the one-brain substrate
(the organ is still its own bridge).
