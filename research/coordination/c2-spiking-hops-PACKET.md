# PACKET c2-spiking-hops (Claude, 2026-10-27): Stage-2 C2, two separate methods: hop reads on the spiking store (M1) and acquisition of all-novel transitives through a learned order cue (M3)

host-decision: scaffold
decides: M1: the possessive-question dispatch (_POSSESSIVE_CHAIN_RE), the hop-1 relation candidates (_ROLE_NOUN_HINTS plus the same-verb fallback), the two-hop relation order taken from that parse, the block selection inside query_patient at the production default (a host first-match over spiking-decoded rows), the slot-to-word map of the workspace, and the relay of the conflict unit's spike count into a workspace slot drive; M3: which utterances the comprehension read judges (extract_transitive's 3-content-token test and its function-word list), the role assignment of the stored triple (the positional parse after the host strip), and the utterance-onset event
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement)
replaced-by: the learned relation-noun binding (M2, its own packet), the cs-race spiking store competition for block selection, the word-role frame commit for scope and stored roles, and one-bridge co-residence of the K-way sequencer and the GNW workspace
cap: 2 builds

Status: DESIGN ONLY, for Claude's review. No production code, no default flipped, no `sim/` edit. Gate seeds
42/43/44/100/101/102 are not run by anything here. Branch `research/c2-spiking-hops` (cut from main cadf0f40f).
Ruling this answers: stage2-fixes-1026, C2 item ("then two separate methods"). M2 (a learned relation-noun binding in
place of `_ROLE_NOUN_HINTS`) is NOT in this packet; nothing below claims it, and no bar here needs it.

Inputs read: `research/findings/2026-10-26-stage2-fail-diagnoses.md` (C2 section);
`research/findings/2026-10-26-c2-kb-scramble-host-ownership.md`; the C2 diagnosis and its skeptic's verdict
(workstation `sim-rev/diag_1026/workflow_result.json`, `result[0]`, keys `diagnosis.next_method` and
`verdict.next_method_concerns`); `research/runners/compositional_chain_route.py`;
`research/runners/_gnw_reentrant_metacog_gated_deliberation_derisk.py` (`confidence_gated_chase` :340,
`acc_conflict_gate` :326) and its production users `webapp/gnw_multistep_deliberation.py` (C5) and
`webapp/gnw_deliberation.py`; `research/runners/stage2_exit_gate.py` (`infer` group :234-270, `score_infer` :1107,
`teach_acquired` :1024, `kb_scramble_read` :1330); `research/runners/one_brain_composer.py`;
`research/runners/comprehension_production_organ.py`; `docs/drafts/word-role-frame/PACKET.md` and RULING
wrc-frame-1-1024; `docs/drafts/cs-race/PACKET.md` and `research/coordination/cs-race-f1h-PACKET.md`.

## 0. The skeptic's concerns, and where each is answered

| # | concern (verdict.next_method_concerns, condensed) | answer |
|---|---|---|
| 1 | M1 does not touch the cause of 0/6 (the D4 refusal of the permuted teach) | Agreed. M1 is the ownership method; M3 is the acquisition method. Section 4 says what the 11-21 C2 verdict depends on. |
| 2 | `confidence_gated_chase` takes one `rel`, PROPOSE is first-match, distractors are host-random | M1 does not reuse the chase loop. It reuses the per-hop EVALUATE (`_deliberate_hop_conf`) and `acc_conflict_gate`, with NO distractors (2.3). |
| 3 | `query_patient` is first-match; onebrain has no `_iter_facts`, so `all_candidate_patients` is empty; the multi-valued moat would silently regress | The conflict read is a named new circuit on the K-way sequencer's match pools (2.3), plus a counted multi-valued-hop arm scored on BOTH routes (2.8, arms A7-A10). |
| 4 | Not free of English tables until M2; the madeup_role bar bundles M1 and M2 | M1 claims nothing about tables; `_ROLE_NOUN_HINTS` is declared in the header; no madeup_role bar here. |
| 5 | Turn-scoped engagement bars are already met or unreachable | Counters are scoped to the hop reads inside the route, with per-hop records (2.7). |
| 6 | kb_scramble bar only on derived_svo/derived_from; derangement with distinct patients; engagement read-back | Adopted as written; the existing arm machinery already does the derangement and read-back (2.8, A3). |
| 7 | No flag-off/answer-rate arms; M3 needs a gate-off arm and a gibberish arm; the word-role flip is HELD | Flag-off, gate-off and garbled arms are in 2.8 and 3.6; M3's dependency on the word-role lane is stated (3.8). |
| 8 | M3 as scoped does not touch the role-branch refusals (goat/hay, crab/leaf) | Agreed and kept out of M3's bars; reported arm with a registered prediction (3.6, R-goat). |

## 1. The defect, as measured

- **The 0/6 is an acquisition refusal.** The default-ON D4 comprehension gate refuses 'the heron hunts the gnat' as
  fully out-of-vocabulary (margin 0.0 against 0.2), so the permuted premise is never stored (finding 2026-10-26, C2).
- **With the gate off the permuted exposure is acquired and the conclusion tracks it.** Diagnostic arms with
  `BRAIN_COMPREHENSION_GATE=0` on dev seeds 1-3 (committed:
  `research/findings/raw/_stage2_diag/c2_comprehension_gateoff/s1/infer__permuted_gateoff.json`, same for s2, s3):
  t_heron_hunt recalls heron-hunt-gnat, p_prey_eat is derived `[gnat, eat, frog]` from `heron hunt gnat ; gnat eat
  frog`, c_frog_eat abstains. The intact arms with the gate off (`.../s1/infer__intact_a_gateoff.json`, s2, s3) are
  unchanged: derived `[frog, eat, gnat]`. In ask, 'the goat eats the hay' is stored with the gate off
  (`.../s1/ask__intact_a_gateoff.json`, s2, s3). Dev seed 7 showed the same permuted result on the workstation with
  the corpus absent (diagnostic, uncommitted). The gate off is a departure from production, used here only as a
  reference.
- **The conclusion is host-owned.** Deranging `OneBrainComposer.kb` while the bound vectors stay intact moves
  derived_svo and derived_from to the list's conclusion (`frog eat wren`); forcing the route onto its spiking fallback
  gives the bound vectors' conclusion (`frog eat gnat`), dev seeds 7 and 1
  (`research/findings/raw/_stage2_diag/c2_kb_scramble/kb_scramble_summary_dev7_dev1.json`).

Two defects, two methods. M1 moves the decision onto the spiking store; it does not change acquisition, so on the
frozen permuted arm it still reads 0/6. M3 changes acquisition; it does not change who owns the hops. Each has its
own flag, lesions and bars, and either can land without the other.

## 2. M1: hop reads on the spiking store (flag `BRAIN_CHAIN_SPIKING_HOPS`, default OFF)

### 2.1 What first-match means today

`OneBrainComposer.query_patient` (`one_brain_composer.py:1787`) returns ONE patient per (agent, action). Its block
selection goes through `_seq_block` (:1680). At the production default (`_INTEGRATED_LOOP_DEFAULT_ON = False`,
`webapp/server.py:3910`) that is a host first-match over the spiking-decoded rows (:1685-1690): every stored block is
reconstructed and unbound on the RF substrate (`_read_blocks`), then a Python loop returns the first row whose decoded
agent and action equal the cue. With `BRAIN_INTEGRATED_LOOP=1` the selection is the K-way sequencer's BG first-match
priority WTA (`_phaseB_onebrain_sequencerK_derisk.py`, block i inhibits every j>i). Either way a second patient under
the same cue is invisible to the caller. The patient CONTENT is the decoded bound vector in both cases; `.kb` is read
inside `query_patient` only to route a clause patient (:1822) and an attributed patient (`_attributed_patient`).

The single-hop GNW conflict gate (`webapp/gnw_deliberation.py`, default ON) does not help on this composer: it
requires `_iter_facts`, which `OneBrainComposer` and its `TieredFactStore` wrapper do not expose, so `deliberate_svo`
returns `no_scannable_composer` and passes the first match through (code reading; confirmed or refuted in step 0).

### 2.2 Mechanism

Under the flag, `resolve_compositional_chain` keeps its parse and its candidate loop (hint relations, then the same
verb) and replaces `_hop` with `_hop_spiking`. Per hop (x, rel):

1. **PROPOSE.** `target = composer.query_patient(x, rel)`: the spiking decode of the bound vectors, selected as in 2.1.
2. **CONFLICT READ.** The conflict unit of 2.3 reads how many stored blocks match (x, rel) in spikes.
3. **EVALUATE.** One workspace cycle on the warm P1.2 workspace the C5 chase uses (`_get_bridge`, `_get_theta` from
   `webapp/gnw_deliberation.py`): `target` drives slot 0 at `IGNITE_PA`; slot 1 is driven only by the conflict relay;
   no distractor is driven (the C5 chase's host-random `_pick_distractors` is not used). `_deliberate_hop_conf` then
   `acc_conflict_gate` decide: n_ignited 0 -> hop miss; n_ignited 1 with conf >= theta_hi -> hop resolved;
   n_ignited >= 2 -> RETRY to R_max, then ABSTAIN (the whole chain abstains, as today's ambiguity rule).
4. **BROADCAST.** The next hop is cued by the workspace's committed winner (the word in the winning slot), not by the
   PROPOSE return value, so a hop that does not ignite cannot pass its candidate on.

The control flow over hops (two hops; candidate relations in order; abstain on any ambiguity) is the route's own and
is declared in the header. The C5 chase loop is not used: its single `rel` and emergent depth do not fit a question
whose grammar fixes two relations.

### 2.3 The conflict read: which circuit

**The K-way sequencer's per-block match pools, plus one new conflict population.** The circuit already exists up to
its match stage (`build_sequencerK_bridge`, `_phaseB_onebrain_sequencerK_derisk.py`; wired in the composer through
`_ensure_sequencer`, `one_brain_composer.py:1641`): per block b, a decoded word-line drive (from the composer's own
cleanup scores, `make_block_drives`) passes a gate that the cue word-line opens, so `mw` fires only when the decoded
word equals the cue; role OR-pools feed a block AND pool `m{b}` that fires only when agent AND action match. Today the
`m{b}` pools feed the first-match priority chain, which hides a second match.

New (additive builder option `conflict_unit=True`, default False, sequencer byte-identical when off): a population
`acc_conf` (20 Izhikevich RS cells) on the sequencer bridge receiving every `m{b}` through one shared excitatory weight
`w_c`, with no other input. `w_c` is set once, on design seed 7 (2.10, step 1), between the summed drive of one active
match pool and of two, so one matching block leaves `acc_conf` below rheobase and two co-active blocks drive it. It
inhibits nothing in this build; its spike count over the sequencer settle window is the conflict read. This is
response-conflict monitoring as co-activation of competing channels (Botvinick, Braver, Barch, Carter and Cohen 2001,
Psychol Rev 108:624), placed on the substrate's own match result. The sequencer is built for the hop reads even when
`integrated_loop` is off (the selection stays the composer's own; only the match pools and `acc_conf` are read).

**Relay to the workspace (declared host residual).** The sequencer and the workspace are separate bridges, so the
count crosses by host relay: slot-1 drive = `IGNITE_PA * min(1, n_conf / n_conf_ref)`, linear, no threshold;
`n_conf_ref` is the `acc_conf` count for a two-match cue on design seed 7. The decision threshold is the workspace's
ignition, read by the unchanged `acc_conflict_gate`.

**LTM tier.** `TieredFactStore` falls through to the routed LTM shard when the buffer abstains. The sequencer covers
the buffer only. For an LTM-sourced hop the conflict read is `all_candidate_patients` on the shard's
`RFPhasorComposer` (role-unbind of the stored composites; the shard exposes `_iter_facts`) into the existing
`conflict_gate`; that enumeration is host closed-form and is recorded per hop as `conflict_read: rf_shard_unbind`.
The C2 fixture words are not LTM-resident (finding 2026-10-26), so this path is reported, never barred here.

**Risk named before data.** The integrated loop over-abstains at the tiny-demo's small vocabulary
(`webapp/server.py:3903-3909`); at V=320 it missed 6-14 of 35 recalls (finding
2026-09-05-rank2-integrated-loop-at-scale-answer-preservation-UNDEFINED-recall-incomplete). A match pool that misses
a true match would make `acc_conf` miss a conflict, which is the unsafe direction. Step 0 measures the match pools on
this fixture before anything is built, with a STOP line.

### 2.4 `.kb` becomes an un-consulted audit oracle

Under the flag the route never calls `_distinct_patients` before it returns. After the decision, the route calls it
once per hop as an audit (`phase: audit`), and records what the list would have said beside what the spiking read
said. The audit result is never read by any branch that follows. A unit test fixes this: a composer whose `.kb`
raises on access while the route's decision scope is open must still return the spiking result.

### 2.5 Flags and lesions

- `BRAIN_CHAIN_SPIKING_HOPS` (default OFF): with it unset, `resolve_compositional_chain` is byte-identical (golden
  hash over a fixture battery in a unit test, plus the existing infer arms' primary fields).
- `BRAIN_CHAIN_SPIKING_HOPS_CONFLICT_LESION`: `w_c` = 0 (acc_conf hears nothing). Predicts: a multi-valued hop
  commits the first match.
- `BRAIN_CHAIN_SPIKING_HOPS_WS_LESION`: the hops use the workspace built with assembly self-recurrence zeroed
  (`_get_bridge(seed, lesion=True)`, the C5 lesion's bridge). Predicts: no hop ignites, the conclusion vanishes.
- `qp_lesion` (the existing in-process patch, `query_patient` returns None): predicts the conclusion vanishes, the
  reversal of today's 7/7 invariance.

### 2.6 What stays host under M1 (declared in the header)

The regex dispatch, `_ROLE_NOUN_HINTS` and the same-verb fallback, the two-hop order, the default selection inside
`query_patient` (a host first-match over decoded rows; under the conflict read it only ever sees one matching row
when it answers, so it has no choice left to make, and the step-2 record checks that claim per hop), the slot-to-word
map, the cross-bridge relay, the derived framing (`frame_derived_answer`) and the verify re-parse
(`brain_chat_tui.py:1922-1942`). M1 claims one thing: the identity of each hop's patient and the abstain on ambiguity
are decided by spiking reads of the bound store.

### 2.7 Engagement record, scoped to the hop reads

Instrumentation lives in the stage-2 runner's arm worker (behaviour-preserving wrappers, as the kb_scramble arms do).
A hop scope opens when the route enters `_hop_spiking` and closes when it returns. Per hop:
`{turn, hop, agent, relation, proposed (query_patient result), selected_block, source_tier, m_counts (per block),
blocks_matched (m{b} over match_thresh), acc_conf_spikes, slot1_drive, n_ignited, conf, gate_decision, committed,
route_kb_reads, composer_kb_reads}`. `route_kb_reads` counts `.kb` accesses whose caller frame is
`compositional_chain_route` while a hop scope is open; `composer_kb_reads` counts those from `one_brain_composer`
(the clause test and the attribute check), listed by line. Turn-level counts (`_chain_facts`,
`rich_answer_composer._stored_facts`, elaboration) are reported separately and never barred.

### 2.8 Arms (stage-2 runner, `infer` group, all informational, never counted in the Stage-2 verdict)

Session tables: `infer` intact (the registered one) and two new ambiguity tables, run under the same harness
(numpy CPU, tiny-demo +LTM, corpus present, `PYTHONHASHSEED=0`, `rich` omitted):
- `infer_amb1` (hop-1 ambiguity): heron hunts frog; frog eats gnat; heron hunts fish; then p_prey_eat, c_heron_hunt,
  c_dog_chase.
- `infer_amb2` (hop-2 ambiguity): heron hunts frog; frog eats gnat; frog eats fly; then the same probes.
  Each teach in both tables contains a word already in the brain's vocabulary or the hand animacy table, so the D4
  gate passes it at the default (stated, and read back from the teach turns).

| arm | table | flags | purpose |
|---|---|---|---|
| A0 off | infer | none (production) | today's route, the pairing side for answer rate |
| A1 m1 | infer | M1 on | the conclusion through spiking hops |
| A2 m1_b | infer | M1 on, independent rebuild | the null for A1 |
| A3 m1_kb_scramble | infer | M1 on + the existing `kb` derangement | who owns the conclusion now |
| A4 m1_qp_lesion (+ .rep0) | infer | M1 on + qp patch | PROPOSE necessary |
| A5 m1_ws_lesion (+ .rep0) | infer | M1 on + WS lesion | workspace ignition necessary |
| A6 m1_seq | infer | M1 on + `BRAIN_INTEGRATED_LOOP=1` | sequencer selection, reported only |
| A7 amb1_off / A8 amb1_m1 | infer_amb1 | off / M1 on | hop-1 ambiguity, both routes |
| A9 amb2_off / A10 amb2_m1 | infer_amb2 | off / M1 on | hop-2 ambiguity, both routes |
| A11 amb1_conflict_lesion (+ .rep0) | infer_amb1 | M1 on + conflict lesion | the conflict unit necessary |
| A12 m1_permuted_gateoff | infer permuted | M1 on + `BRAIN_COMPREHENSION_GATE=0` | conclusion tracks exposure through spiking hops (departure, reference) |

New arms are added as informational entries, so every pre-existing arm keeps its turn table, env, patch and kind,
pinned by the golden hash in `tests/test_stage2_kb_scramble.py`; `pool_job_lines` grows by the new arms only.

### 2.9 Registered predictions and bars (dev read; numbers are per dev seed, 1-6)

Predictions, written before any M1 code exists:
- P1 (A1): p_prey_eat derived_svo `[frog, eat, gnat]`, derived_from `[[heron, hunt, frog], [frog, eat, gnat]]`,
  the "I derived this from" lead present. Same as A0 on these fields.
- P2 (A3): engagement as in the 10-26 arm (3 scoped facts changed, read-back equals plan, patients distinct, the
  conclusion fact changed) and derived_svo/derived_from equal the oracle's `predicted_from_bound`
  (`[frog, eat, gnat]`), not `predicted_from_list` (`[frog, eat, wren]`). The answer text keeps list-read sentences
  from other readers, and c_frog_eat and p_yesno stay flagged guesses (the direct-recall verify reads
  `chat.stored_facts`, outside M1; reported, not barred).
- P3 (A4) and P4 (A5): p_prey_eat has no derived answer, reproduced on .rep0.
- P5 (A7-A10): p_prey_eat abstains on both routes in both ambiguity tables; in A8 and A10 the abstaining hop shows
  blocks_matched = 2 and acc_conf_spikes > 0.
- P6 (A11): p_prey_eat is derived through the FIRST-taught patient (frog), reproduced on .rep0.

Bars (each on every dev seed 1-6; design seed 7 is excluded from the count):
- **B1 conclusion:** P1 holds.
- **B2 ownership:** P2's two fields hold; the bar reads derived_svo and derived_from only.
- **B3 PROPOSE necessary:** P3 holds and reproduces.
- **B4 workspace necessary:** P4 holds and reproduces.
- **B5 multi-valued abstain:** P5 holds on both routes; engagement precondition: two decoded rows under the
  ambiguous cue (the default appended a re-told patient in the 10-12 dev-7 retell smoke; a REWRITE makes the seed
  UNDEFINED, never PASS).
- **B6 conflict unit necessary:** P6 holds and reproduces.
- **B7 hop scope:** both hops recorded on p_prey_eat in A1; `route_kb_reads` = 0 inside every hop scope;
  `query_patient` calls inside hop scopes = hops attempted; the query_patient selected block lies in
  `blocks_matched` on every resolved hop.
- **B8 moat and controls:** c_prey_chase abstains; c_dog_chase recalls cat; c_heron_hunt and c_frog_eat primary
  fields equal A0.
- **B9 answer rate:** per label, flag-on (A1, A8, A10) against flag-off (A0, A7, A9) on p_prey_eat, c_heron_hunt,
  c_frog_eat, c_prey_chase, p_yesno, c_dog_chase: no label answered off and abstained on. `tools/answer_rate.py` reads
  only the last turn of an arm file, so a `--labels` mode is part of the build; FLIP-HURTS-ANSWERS is reported.
- **B10 flag-off identity:** with the flag unset, A0's primary fields equal an `intact_a` arm run at the base
  revision (before any M1 code) on the same seed, and the unit golden holds.

Every lesion reports engaged / not engaged first (docs/TERMS.md, lesion); a lesion that did not hold is reported as
not held, never as no effect. All infer arms were seed-invariant in non-text fields at the 10-26 gate, so the
workspace and sequencer counts per seed (2.7) are reported to show whether six seeds are six measurements here.

### 2.10 M1 build order, with STOP lines

**Step 0, the engagement smoke (read-only; no production edit; dev seed 7, then 1-6).** In the arm worker, after
the teach turns of `infer`, `infer_amb1` and `infer_amb2`:
- (a) call `composer.query_patient` for (heron, hunt), (heron, eat), (frog, eat), (frog, chase) and record result,
  selected block and the decoded rows that match;
- (b) build the K-way sequencer for the buffer's K with the current drives (`_ensure_sequencer`) and record every
  block's `m{b}` count for the same cues, plus the sequencer decision against the host first-match;
- (c) record `chat._last_gnw_delib` on c_heron_hunt in `infer_amb1` (the single-hop conflict gate on onebrain);
- (d) record that the D4 gate stored every ambiguity-table teach, and the row count under each ambiguous cue.
GO rule (written here, before the run): on 7/7 seeds, every true match drives its `m{b}` over `match_thresh`, no
non-matching block does, and the smallest two-match summed `m` count exceeds the largest one-match count by at least
1.5x (a weight window for `w_c` exists). If a true match is missed on any seed, STOP: post the counts; the circuit is
banked and the named next carrier is the cs-race engram race read for co-latching (two latches ON = conflict), which
waits on cs-race F1. Post `[c2-hops-0] RESULT`, STOP for Claude.

**Step 1, the build (default OFF).** `conflict_unit` in the sequencer builder; `_hop_spiking`, the audit call, the
three flags in the route; `w_c` and `n_conf_ref` set on seed 7 by the rule above and frozen in code with the seed-7
counts in a comment; unit tests: flag-off golden, the raising-`.kb` test, the lesion flags reach their targets.
Module docstring corrected (it says the hops are list scans today; under the flag they are spiking reads).

**Step 2, the dev read.** A0-A12 on dev seeds 1-6 through the pool (numpy, memcap 10), scored against 2.9 by a
scorer written before the runs. Post `[c2-hops-2] RESULT`, STOP. A failed bar is a RESULT, never a retune.

**Step 3 (Claude).** A `flag: no` prereg carrying A1-A11 as informational arms of the 11-21 gate attempt, or a flip
packet if every bar holds; the default stays OFF until a flip prereg passes (smoke-first, flag-off battery arms on
the gate seeds, answer-rate).

## 3. M3: acquisition of well-formed all-novel transitives through a learned order cue (flag `BRAIN_COMPREHENSION_LEARNED_ORDER`, default OFF)

### 3.1 Why D4 refuses, and what it can read

`ComprehensionProductionOrgan.judge` (`comprehension_production_organ.py:738`) scores an assertion only when
`competent()` (:719) holds: fully covered by the hand ANIMACY and VERB_SELECTS tables (or their learned lexicons), or
fully out-of-vocabulary. The read drives the SEMANTIC cues (animacy, verbfit) of `SpikingRoleCompetition`. For a
fully-OOV sentence both cues are absent, so the margin is 0.0 by construction and every such sentence is refused,
well-formed or not. The organ's learned position cue is not used by D4, and it would not help: its value is a host
vote by noun index (`_position_vote`, `_phaseB_multicue_competition_spiking_derisk.py:109`) that separates noun 0
from noun 1 in ANY 3-token string, garbled or not. It is registered below as a null that the garbled bar must kill.

What separates "the blick gorps the dax" from "the blick the dax gorps" when every content word is new is the frame:
where the closed-class words sit and the order in which roles arrived in everything heard before. Children use
exactly this: determiners cue nouns before they know them (Hoehle, Weissenborn, Kiefer, Schulz and Schmitz 2004), and
word order cues argument roles for novel verbs at 21 months (Gertner, Fisher and Eisengart 2006).

### 3.2 Mechanism (organ level, on the word-role parser)

On `WordRoleParserOrgan` (`research/runners/word_role_parser_organ.py`) with the word-role frame path
(`docs/drafts/word-role-frame/PACKET.md`: every token heard in order, no host position code, held traces T_w with a
plastic T->role projection and discharge on commit), add a `role_seq=` build option (default None, byte-identical):
- **Onset trace O.** A held population started by the utterance onset (a world-side boundary event, declared).
- **Role traces RT_agent, RT_action, RT_patient.** Each is started by its role's commit-gate burst and held until
  the next commit discharges it (the frame packet's discharge pool).
- **Plastic O->role and RT->role projections**, trained by the engine's Hebbian co-firing rule on the statements the
  brain already hears (the teacher channel), with word->role and pos->role frozen while they learn (the frame
  packet's rule). With SVO exposure, O->agent, RT_agent->action and RT_action->patient potentiate. No code names an
  order: the order is whatever the exposure contained.
- **The read.** A heard assertion is comprehended when agent, action and patient each commit exactly once, every
  content token commits exactly one role, and no closed-class token commits a role (the frame packet's MM1 clause).
  The M3 margin is the minimum over the three roles of the commit gate's winner-minus-runner-up spike count,
  normalised by the in-sample median.
- **Carrier.** The traces must outlive the per-word reset. The carrier is whichever of these passes its engagement
  items first: the cs-race f1h held apical plateau (`research/coordination/cs-race-f1h-PACKET.md`, items C0-C3), or
  the frame packet's NMDA trace once wrc-frame-1b's starvation is fixed (T->role moved on 1 of 5760 training steps,
  RULING wrc-frame-1-1024 and the 10-26 WORKLOG line). Neither holds yet.

### 3.3 D4 integration (default OFF)

Under the flag, only the fully-OOV branch of `judge()` changes: it reads the M3 margin over the raw token sequence
instead of the SEMANTIC margin. The fully-covered branch, the out-of-competence pass-through and the known-fact
honour rule are untouched (byte-identical for every input that is not fully OOV). The M3 threshold is calibrated at
build from the exposure statements re-heard (in-sample well-formed only; no garbled item enters calibration), as
D4's own build calibration does. The stored triple keeps the production store path; the M3 commit roles are
recorded beside it, and a teach is stored only when the two agree (a disagreement refuses; counted and reported).

### 3.4 Tables: what M3 adds and what stays

M3 adds no word to any table and reads no word list. The test items are nonce CVC forms drawn by a fixed-seed
generator, never in ANIMACY, VERB_SELECTS, the learned lexicons or the corpus vocabulary (a CI grep confirms they
appear in no production module). What stays host, declared in the header: `extract_transitive` decides which
utterances reach the read (3 content tokens after its function-word strip), the store path's positional role
assignment, and the onset event. M3 makes no claim about English tables beyond adding none.

### 3.5 Organ battery (dev seeds 1-15, numpy, cheap)

Exposure: the frame packet's statement set, replayed as now. Test items, all nonce:
- **W, well-formed:** 12 items per seed in the exposure frame ("the blick gorps the dax").
- **G, garbled, inside D4's scope (3 content tokens, no wh-word):** G1 "the blick the dax gorps"; G2 "gorps the blick
  the dax"; G3 "the blick gorps dax the"; G4 "blick dax gorps" (no closed-class word); G5 uniform shuffles of the
  five tokens excluding the well-formed order. 6 items per class per seed.
- **Out-of-scope strings** (4 content tokens and the like): reported only. D4 never judges them today and they pass
  to the store path; M3 does not change that, and the report says so.
- **Order-permuted exposure:** the same statements heard verb-initial ("chases the dog the cat").
- **Nonce determiner:** the same statements with "ko" for "the".

Arms: intact; role-sequence lesion (RT->role and O->role gates closed at read, persistence verified); carrier lesion
(traces not held across the reset); frame lesion (T->role closed); transplant (only the O->role and RT->role blocks of
the verb-initial organ copied into the SVO organ, same seed).

Bar-sanity generators, scored by the real scorer, n >= 1000: perfect; null-position-cue (the multicue organ's
position vote: accepts G); null-accept-all-OOV (gate off); null-refuse-all-OOV (today's D4); null-English-template (a
host "the N V the N" pattern: fails the order-permuted and nonce-determiner bars); null-bag-of-words; random. Every
perfect >= 0.95 and every null <= 0.05 on the composite; the record names the bar that kills each null.

### 3.6 Reply-level arms (stage-2 runner, informational; dev seeds 1-6)

- **m3_permuted:** the frozen permuted `infer` table with M3 on. t_heron_hunt acquired (`teach_acquired`), and
  p_prey_eat derived with terminal frog.
- **m3_intact:** intact `infer` with M3 on: primary fields equal the flag-off arm.
- **m3_garbled / off_garbled / gateoff_garbled:** a new table: the four intact teaches, then G1-G3 teaches built
  from real unknown words (heron, gnat, hunts, and the like), then probes on those words. Prediction: refused under
  M3 and under the production default; STORED with the gate off (the arm has teeth: the gate-off brain stores word
  salad).
- **answer rate:** m3 against flag-off on every infer and ask probe label, `--labels` mode; FLIP-HURTS-ANSWERS
  reported.
- **R-goat (reported only):** `ask` with M3 on. Prediction: 'the goat eats the hay' and 'the crab eats the leaf'
  are still refused, because they go through the covered branch M3 does not touch. The named next method for them is
  the learned order read added as a third cue on the covered branch, decided after M3's organ verdict.

### 3.7 Predictions and bars

Organ (each per seed on dev seeds 1-15; at least 14 of 15 seeds must meet every bar):
- **OM1:** W comprehended >= 11/12.
- **OM2:** every G class refused >= 5/6, pooled G refused >= 0.95; G4 included (prediction: refused, because no
  determiner trace supports an argument commit).
- **OM3 order learned:** the verb-initial organ comprehends verb-initial W >= 11/12 and refuses SVO W >= 11/12.
- **OM4 no English frame:** the "ko" organ comprehends "ko" W >= 11/12 and refuses "the" W >= 11/12.
- **OM5 transplant:** the SVO organ with the verb-initial blocks follows the donor (OM3 thresholds).
- **OM6 necessity:** the role-sequence lesion and the carrier lesion each drop W to <= 2/12.
- **OM7 identity:** with `role_seq=None` the frame-packet pins hold; the D4 decision is unchanged on every
  non-fully-OOV item of the committed D4 battery.

Reply (dev seeds 1-6, every seed): R1 m3_permuted acquires t_heron_hunt and p_prey_eat ends on frog; R2 every G teach
in m3_garbled is refused and no later probe recalls it; R3 m3_intact equals flag-off on primary fields; R4 no
FLIP-HURTS-ANSWERS; R5 flag unset is byte-identical.

### 3.8 M3 build order, dependencies, STOP lines

- **Step 0 (read-only, now, parallel to M1 step 0).** (a) The D4 decision (scope, competence branch, margin) on
  every W, G and out-of-scope item at production defaults; (b) the word-role parser's commit read on W and G with
  today's organ (prediction: nothing commits, since position alone is subthreshold by design); (c) the carrier status
  read from the cs-race f1h step-0 RESULT and the wrc-frame-1b RESULT. Post `[c2-acq-0] RESULT`, STOP.
- **Step 1 (organ build, default OFF)** starts only when one carrier holds a trace across the per-word reset.
- **Step 2:** the organ battery and bar sanity on dev seeds; post RESULT, STOP.
- **Step 3:** D4 integration and the reply-level dev read; post RESULT, STOP. Then Claude writes the preregs.

Dependency, stated plainly: both carriers are open. RULING focus-1025 parked the word-role frame and cs-race lanes
until the Stage-2 gate RESULT, which landed on 10-26; reopening them as M3's carrier is a ruling this packet asks
for. M3 is not expected to land before the 11-21 attempt. No-defer: M3 continues after 11-21 if it misses.

## 4. What neither method does

- **The 11-21 C2 verdict.** On the frozen permuted arm, M1 alone leaves the D4 refusal and 0/6 in place. Without
  M3 by then, the options are Claude's: an instrument amendment giving `score_infer` an acquisition precondition like
  ask's A4 (an unacquired permuted teach makes the seed UNDEFINED, so C2 cannot reach six defined seeds: UNDEFINED,
  not PASS), or a new prereg with an acquirable permuted exposure. Neither is ordered here.
- **The derived construct (V4).** The conclusion still asserts premise 2 verbatim (`conclusion_is_hop` True). Whether
  that is the inference Stage 2 intends is a separate question.
- **The void `transitive_chat` arm** (its flag is already ON and the route sits only on the single-fact path) and
  **the direct-recall framing change under kb_scramble** (a list-backed verify over `chat.stored_facts`) are named,
  not addressed.

## 5. Seeds, resources, where it runs

- Design seed 7 (M1 `w_c`, `n_conf_ref`); M1 dev read seeds 1-6; M3 organ dev seeds 1-15, reply dev seeds 1-6. The
  gate seeds are run only through a later prereg.
- Stage-2 arms: about 6.0-6.5 GB peak each, `tools/memcap.sh 10`, numpy CPU, pool lines. Step 0 sequencer builds:
  memory measured on seed 7 first under `tools/mem_ok.sh`. Organ probes: under 0.1 GB (wrc step 0 peaked 85.3 MiB).
- No GPU. No AWS.

## 6. Order and who does what

1. M1 step 0 and M3 step 0, in parallel (both read-only, CPU). STOP for Claude.
2. M1 steps 1-2 (the ownership lane; unblocked). STOP for Claude.
3. M3 steps 1-3 after a carrier holds and Claude reopens its lanes.
The driver runs the steps and posts the RESULT lines; Claude rules at every STOP, writes the preregs, reviews and
merges.
