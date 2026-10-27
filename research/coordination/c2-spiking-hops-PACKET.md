# PACKET c2-spiking-hops (Claude, 2026-10-27; revised the same day after its skeptic): Stage-2 C2, two separate methods: hop reads on the spiking store (M1) and acquisition of all-novel transitives through a learned order cue (M3)

host-decision: scaffold
decides: M1: the possessive-question dispatch (_POSSESSIVE_CHAIN_RE), the hop-1 relation candidates (_ROLE_NOUN_HINTS plus the same-verb fallback), the two-hop relation order taken from that parse, the block selection inside query_patient at the production default (a host first-match over spiking-decoded rows), the slot-to-word map of the workspace, the relay of the conflict unit's spike count into a workspace slot drive, and the mapping of acc_conflict_gate's four outputs onto hop outcomes (section 2.2); M3: the competent() branch selection of ComprehensionProductionOrgan, which is a host English table (the hand ANIMACY and VERB_SELECTS coverage with their corpus-learned extensions, plus the _brain_vocab membership test, decide whether an assertion is fully covered, fully out-of-vocabulary or passed through, and so decide which assertions M3 ever judges), which utterances the comprehension read judges (extract_transitive's 3-content-token test and its function-word list), the role assignment of the stored triple (the positional parse after the host strip), and the utterance-onset event
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement)
replaced-by: the learned relation-noun binding (M2, its own packet), the cs-race spiking store competition for block selection, the word-role frame commit for scope and stored roles, and one-bridge co-residence of the K-way sequencer and the GNW workspace
cap: 2 builds

Cap accounting (a ruling is asked for, not assumed). Three build steps ship code under this one header: M1 step 1
(2.10), M3 step 1 (the organ, 3.8) and M3 step 3 (the D4 integration, 3.8). The cap of 2 covers M1 step 1 and M3
step 1. CLASS HD lets a cap only go down, so the header is not raised. The count is review's job: the gate counts
prereg amendments as its proxy and does not count packet steps (`tools/gates/host_decision_declared.py`, limit (4)).
M3 step 3 does not start without a Claude cap ruling for this scaffold group (+1 build, a section-6 RULING of the
runbook, cited in this file by its key once it exists). The driver relays the ask as `[c2-hops-cap] NEEDS-CLAUDE`
when it posts this revision. If the ruling refuses, M3 step 3 does not run under this method.

Status: DESIGN ONLY, for Claude's review. No production code, no default flipped, no `sim/` edit. Gate seeds
42/43/44/100/101/102 are not run by anything here. Branch `research/c2-spiking-hops` (cut from main cadf0f40f; first
draft a7799b30c). The packet's skeptic REFUTED the first draft as written; every required fix is applied (section 0b)
and Claude's ruling of 2026-10-27 is in section 7. Ruling this answers: stage2-fixes-1026, C2 item ("then two separate
methods"). M2 (a learned relation-noun binding in place of `_ROLE_NOUN_HINTS`) is NOT in this packet; nothing below
claims it, and no bar here needs it.

Inputs read: `research/findings/2026-10-26-stage2-fail-diagnoses.md` (C2 section);
`research/findings/2026-10-26-c2-kb-scramble-host-ownership.md`; the C2 diagnosis and its skeptic's verdict
(workstation `sim-rev/diag_1026/workflow_result.json`, `result[0]`, keys `diagnosis.next_method` and
`verdict.next_method_concerns`); the packet skeptic's REFUTED verdict on a7799b30c (fixes (1)-(8), section 0b);
`research/runners/compositional_chain_route.py`; `research/runners/_gnw_reentrant_metacog_gated_deliberation_derisk.py`
(`_ignite_and_read_nmda` :158, `_conf_from_nmda` :204, `_deliberate_hop_conf` :222, `acc_conflict_gate` :326,
`confidence_gated_chase` :340) and its production users `webapp/gnw_multistep_deliberation.py` (C5) and
`webapp/gnw_deliberation.py` (`conflict_gate` :144); `research/runners/_phaseB_onebrain_sequencerK_derisk.py`
(`build_sequencerK_bridge` :66, `run_sequencerK` :215); `research/runners/_phaseB_onebrain_sequencer_derisk.py`
(`block_cleanup_scores` :64); `sim/regions.py` (`RegionPathway.stp_disabled`); `research/runners/stage2_exit_gate.py`
(`infer` group :234-270, `run_worker` :659, `teach_acquired` :1024, `score_infer` :1107, `kb_scramble_read` :1330);
`research/runners/one_brain_composer.py`; `research/runners/comprehension_production_organ.py` (`extract_transitive`
:339, `competent` :719, `judge` :738); `webapp/server.py` (`_brain_vocab` :3640, the D4 call :6844);
`research/findings/2026-10-12-validation-battery-prereg.md` (B-C5e, the dev-7 re-tell smoke);
`docs/drafts/word-role-frame/PACKET.md` and RULING wrc-frame-1-1024; `docs/drafts/cs-race/PACKET.md`,
`research/coordination/cs-race-f1h-PACKET.md`, and on main (not on this branch) `cs-race-f1h-0-RESULT.md` and
`cs-race-f1i-PACKET.md`.

## 0. The diagnosis skeptic's concerns, and where each is answered

| # | concern (verdict.next_method_concerns, condensed) | answer |
|---|---|---|
| 1 | M1 does not touch the cause of 0/6 (the D4 refusal of the permuted teach) | Agreed. M1 is the ownership method; M3 is the acquisition method. Sections 4 and 7 say what the 11-21 C2 verdict depends on. |
| 2 | `confidence_gated_chase` takes one `rel`, PROPOSE is first-match, distractors are host-random | M1 does not reuse the chase loop or `_deliberate_hop_conf`. It builds its own workspace drive vector and calls `_ignite_and_read_nmda`, `_conf_from_nmda` and `acc_conflict_gate` directly, as `conflict_gate` does, with NO distractors (2.2). |
| 3 | `query_patient` is first-match; onebrain has no `_iter_facts`, so `all_candidate_patients` is empty; the multi-valued moat would silently regress | The conflict read is a named new circuit on the K-way sequencer's match pools that counts distinct patient assemblies (2.3), plus counted multi-valued-hop arms scored on BOTH routes (A7-A10) and a duplicate-fact pair (A13-A14). |
| 4 | Not free of English tables until M2; the madeup_role bar bundles M1 and M2 | M1 claims nothing about tables; `_ROLE_NOUN_HINTS` is declared in the header; no madeup_role bar here. |
| 5 | Turn-scoped engagement bars are already met or unreachable | Counters are scoped to the hop reads inside the route, with per-hop records (2.7). |
| 6 | kb_scramble bar only on derived_svo/derived_from; derangement with distinct patients; engagement read-back | Adopted; the existing arm machinery does the derangement and read-back, and the reference arms are re-run pinned (2.8, A3/A3a/A3b). |
| 7 | No flag-off/answer-rate arms; M3 needs a gate-off arm and a gibberish arm; the word-role flip is HELD | Flag-off, gate-off, garbled and lesion arms are in 2.8 and 3.6; M3's carrier is ruled (section 7). |
| 8 | M3 as scoped does not touch the role-branch refusals (goat/hay, crab/leaf) | Agreed and kept out of M3's bars; reported arm with a registered prediction (3.6, R-goat). |

## 0b. The packet skeptic's required fixes (REFUTED as written), and where each is applied

| # | fix required | where |
|---|---|---|
| 1 | The garbled reply arm was vacuous: its G teaches reused heron/gnat/hunts, which the intact teaches store, so `competent()` flips and D4 never judges them | 3.6: the garbled table uses never-taught nonce CVC words, disjoint across G teaches; engagement precondition (`comprehension` non-null with `in_scope` true on every G teach of the off arm, else UNDEFINED) |
| 2 | The M1 conflict unit counted matching BLOCKS, not distinct PATIENTS (`OneBrainComposer.store` appends without dedupe; re-tells append a slot) | 2.3: the unit counts distinct patient assemblies on the substrate (shared patient word lines behind m{b} gates, depressing synapses into `acc_conf`), no host string compare; registered duplicate-fact pair `infer_dup` (A13/A14, P7, B11) |
| 3 | The per-hop EVALUATE reuse claim was imprecise: `_deliberate_hop_conf` cannot take a graded slot-1 drive | 2.2: the new drive vector, passed to `_ignite_and_read_nmda` as `conflict_gate` does; every output of `acc_conflict_gate` mapped to a hop outcome |
| 4 | The 2.4 unit test must raise only when the caller frame is `compositional_chain_route` (`query_patient` itself reads `.kb`) | 2.4 |
| 5 | CLASS HD header accuracy: M3's `decides:` must name the `competent()` branch selection; cap accounting for three build steps against cap 2 | header `decides:` (M3 part) and the cap accounting paragraph (a cap ruling is asked for) |
| 6 | An M3 reply-level lesion arm; 'ko blick gorps ko dax' is outside `extract_transitive` | 3.6 (m3_rs_lesion + .rep0, R6); 3.5 (out of D4's reach, organ-only) |
| 7 | Disclose the kb_scramble evidence's provenance; a pinned pool re-run before P2/B2 rest on it | section 1 (disclosure); 2.8 (A3a/A3b pinned re-run); 2.9 (P2, B2 UNDEFINED until then) |
| 8 | B6/A11 engagement precondition | 2.9, B6 |

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
  **Provenance, disclosed.** The committed artifact is the summary only (its sidecar records git 5eeef1ae2, dirty
  tree). The six raw shards it aggregates ran on the workstation only, at branch revision 8491e0df1 with provenance
  off, on two dev seeds (7 and 1). They stay outside the repository; their sha256 values are in the summary. This is
  diagnostic evidence. P2 and B2 below do not rest on it: the two reference arms are re-run through the pool at the
  pinned step-2 revision with provenance on, on dev seeds 1-6 (A3a, A3b in 2.8), and B2 is UNDEFINED until they exist.

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
the same cue is invisible to the caller. The patient CONTENT is the decoded bound vector in both cases. `query_patient`
reads `.kb` itself: `len(self.kb)` in `_read_blocks` (:1389) and `_fact_shard_active` (:1398), `self.kb[idx]` to route
a clause patient (:1822), and the attribute check in `_attributed_patient` (:1782).

The store does not deduplicate. `hear`, `hear_multiframe` and `store` (:702-732) write a block and append to `.kb` on
every call. In the 10-12 dev-7 re-tell smoke a verbatim re-tell appended a slot
(`research/findings/2026-10-12-validation-battery-prereg.md`, B-C5e, `c5_s7_retell.json`). Today's route is not
fooled by that: `_distinct_patients` deduplicates by patient STRING on the host list.

The single-hop GNW conflict gate (`webapp/gnw_deliberation.py`, default ON) does not help on this composer: it
requires `_iter_facts`, which `OneBrainComposer` and its `TieredFactStore` wrapper do not expose, so `deliberate_svo`
returns `no_scannable_composer` and passes the first match through (code reading; confirmed or refuted in step 0).

### 2.2 Mechanism

Under the flag, `resolve_compositional_chain` keeps its parse and its candidate loop (hint relations, then the same
verb) and replaces `_hop` with `_hop_spiking`. Per hop (x, rel):

1. **PROPOSE.** `target = composer.query_patient(x, rel)`: the spiking decode of the bound vectors, selected as in 2.1.
2. **CONFLICT READ.** The conflict unit of 2.3 reads, in spikes, how many DISTINCT patients the matching blocks carry,
   and the relay turns its spike count into `slot1_drive`.
3. **EVALUATE.** One workspace read on the warm P1.2 workspace the C5 chase uses: `_get_bridge(_DEFAULT_SEED,
   lesion)` and `_get_theta(_DEFAULT_SEED)` from `webapp/gnw_deliberation.py`. `_DEFAULT_SEED` reads
   `BRAIN_CHAT_SEED`, which the stage-2 worker sets to the arm's seed before it imports the server
   (`run_worker`, :666); it is recorded per hop as `ws_seed`. `_deliberate_hop_conf` is NOT used: it builds its own
   drive vector (slot 0 at `IGNITE_PA`, distractor slots at `IGNITE_PA * DISTRACTOR_FRAC`) and has no argument for a
   graded slot-1 drive. M1 builds the vector itself, as `conflict_gate` (`webapp/gnw_deliberation.py:144`) does:
   `drives = [0.0] * n`; `drives[0] = IGNITE_PA` when `target` is not None; `drives[1] = slot1_drive`; every other
   slot 0 (no distractors, so the C5 chase's host-random `_pick_distractors` is not used). Then
   `rates, g_nmda = _ignite_and_read_nmda(bridge, xp, slots, snap, drives)`,
   `conf, winner, n_ignited = _conf_from_nmda(rates, g_nmda)`, and
   `acc_conflict_gate(conf, n_ignited, cycles_on_hop, R_MAX_DEFAULT, theta_hi, theta_lo)` with `conflict_gate`'s
   retry walk. The read restores the same snapshot and is deterministic, so a RETRY repeats the same read and the
   walk only counts the budget (as in `conflict_gate`); the count is recorded as `gate_cycles`.
4. **BROADCAST.** The next hop is cued by the word in the winning slot, not by the PROPOSE return value, so a hop that
   does not ignite cannot pass its candidate on.

The gate's outputs, from its own code (`_gnw_reentrant_metacog_gated_deliberation_derisk.py:326-337`; `theta_lo` is
passed and not read), and what each means for a hop:

| `acc_conflict_gate` returns | when | M1 hop outcome |
|---|---|---|
| COMMIT | `n_ignited == 0` | Hop MISS: no patient. In the chase COMMIT means "terminal: keep the last resolved word"; here no word is committed. The route treats it as `_hop`'s `(None, False)`: hop 1 tries the next candidate relation; a hop-2 miss tries the next hop-1 candidate, as today. |
| ADVANCE | `n_ignited == 1` and `conf >= theta_hi` | Hop RESOLVED with the word of the winning slot. Slot 1 carries no word: a slot-1-only winner ABSTAINS (recorded `slot1_only`). |
| RETRY | `n_ignited >= 2` and `cycles_on_hop < R_max` | The walk continues (same deterministic read) until R_max (3). |
| ABSTAIN | `n_ignited >= 2` at R_max, or `n_ignited == 1` with `conf < theta_hi` | The WHOLE chain abstains (the route's ambiguity rule). The second case is the graded one: a slot-1 drive below ignition can raise slot 1's NMDA enough to pull `conf` under `theta_hi` while only slot 0 ignites. The chain then abstains on the first read, with no retry. |

Registered regimes: one distinct patient gives `slot1_drive = 0`, which is `calibrate_theta`'s SOLO regime, so the
gate returns ADVANCE. Two distinct patients give `slot1_drive = IGNITE_PA`, its CONFLICT regime, so the gate returns
RETRY to R_max, then ABSTAIN. An intermediate count can take either ABSTAIN path or ADVANCE; every hop's
`(slot1_drive, n_ignited, conf, decision)` is recorded, and the step-2 report lists every intermediate case.

The control flow over hops (two hops; candidate relations in order; abstain on any ambiguity) is the route's own and
is declared in the header. The C5 chase loop is not used: its single `rel` and emergent depth do not fit a question
whose grammar fixes two relations.

### 2.3 The conflict read: which circuit

**Match stage, existing.** The K-way sequencer (`build_sequencerK_bridge`, `_phaseB_onebrain_sequencerK_derisk.py`;
wired in the composer through `_ensure_sequencer`, `one_brain_composer.py:1641`) has, per block b, decoded agent and
action word lines driven from the composer's own cleanup scores (`block_cleanup_scores`, `make_block_drives`); each
passes a gate that the cue word line opens, so `mw` fires only when the decoded word equals the cue; role OR-pools
feed a block AND pool `m{b}` that fires only when agent AND action match. The `m{b}` pools count matching BLOCKS. Since
the store appends a block per re-tell (2.1), a block count would read a duplicate re-tell of one fact as a conflict
and abstain where today's route commits. That would be a regression and not a conflict.

**What conflicts is responses, not traces.** Response conflict is co-activation of incompatible response channels
(Botvinick, Braver, Barch, Carter and Cohen 2001, Psychol Rev 108:624). Here the responses are the candidate patients.
Two memories that support the same patient do not conflict, so the unit counts distinct patient assemblies:

New, additive builder option `conflict_unit=True` (default False; the sequencer is byte-identical when it is off):
- **Patient decoded lines.** Per block b, word lines `d{b}P_w` driven from the block's PATIENT cleanup membrane. That
  membrane is already computed by the same per-block read: `_seq_cleanup_conns` installs cleanup synapses for every
  main role, and `block_cleanup_scores` returns only the agent and action slices. The build adds a
  `with_patient=True` return of the patient slice (no extra substrate run), through the same `scores_to_drive` at
  `drive_frac` 0.9.
- **Shared patient lines.** One line `pP_w` per vocabulary word (`n_word` cells), shared by all blocks: one lexical
  assembly per word, however many memories point at it. `d{b}P_w -> pP_w` passes a per-block transmission gate
  `gp{b}` opened by `m{b}` firing (`couple_gate_to_pool`, the primitive the sequencer already uses for `gblk{b}`), so
  only a matching block's patient reaches the patient lines.
- **The conflict population.** `acc_conf` (20 Izhikevich RS cells) receives every `pP_w` through one shared excitatory
  weight `w_c` on DEPRESSING synapses (the engine's Tsodyks-Markram STP). The bridge sets
  `enable_short_term_plasticity=True` only when `conflict_unit` is on, and marks every pre-existing sequencer pathway
  `stp_disabled=True` (`sim/regions.py`, the gap#5 mossy-detonator precedent), so the sequencer's own match and
  decision are unchanged (unit-tested, 2.10 step 1). A depressing synapse above its limiting frequency transmits a
  nearly rate-independent drive (Tsodyks and Markram 1997, PNAS 94:719; Abbott, Varela, Sen and Nelson 1997, Science
  275:220). So two blocks that drive the SAME patient line deliver about what one block delivers, and two DIFFERENT
  lines deliver about twice as much. `w_c` is set once, on design seed 7, between one line and two lines (2.10
  step 1); one active line, from one block or from two, leaves `acc_conf` below rheobase.
- `acc_conf` inhibits nothing in this build; its spike count over the sequencer settle window is the conflict read.
  No host string compare deduplicates: a patient's identity is the word line its decoded patient drives on the
  substrate. A decode error that lights a different line for a duplicate reads as a conflict and abstains (the safe
  direction); it is recorded per hop.

The sequencer is built for the hop reads even when `integrated_loop` is off (the selection stays the composer's own;
only the match pools, the patient lines and `acc_conf` are read).

**Relay to the workspace (declared host residual).** The sequencer and the workspace are separate bridges, so the
count crosses by host relay: `slot1_drive = IGNITE_PA * min(1, n_conf / n_conf_ref)`, linear, no threshold;
`n_conf_ref` is the `acc_conf` count for a two-distinct-patient cue on design seed 7. The decision threshold is the
workspace's ignition, read by the unchanged `acc_conflict_gate` (2.2).

**LTM tier.** `TieredFactStore` falls through to the routed LTM shard when the buffer abstains. The sequencer covers
the buffer only. For an LTM-sourced hop the conflict read is `all_candidate_patients` on the shard's
`RFPhasorComposer` (role-unbind of the stored composites; the shard exposes `_iter_facts`) into the existing
`conflict_gate`; that enumeration is host closed-form and deduplicates by host string compare, and it is recorded per
hop as `conflict_read: rf_shard_unbind`. The C2 fixture words are not LTM-resident (finding 2026-10-26), so this path
is reported, never barred here.

**Risk named before data.** The integrated loop over-abstains at the tiny-demo's small vocabulary
(`webapp/server.py:3903-3909`); at V=320 it missed 6-14 of 35 recalls (finding
2026-09-05-rank2-integrated-loop-at-scale-answer-preservation-UNDEFINED-recall-incomplete). A match pool that misses
a true match would make `acc_conf` miss a conflict, which is the unsafe direction. Step 0 measures the match pools and
the patient lines on this fixture before anything is built, with STOP lines.

### 2.4 `.kb` becomes an un-consulted audit oracle

Under the flag the route never calls `_distinct_patients` before it returns. After the decision, the route calls it
once per hop as an audit (`phase: audit`), and records what the list would have said beside what the spiking read
said. The audit result is never read by any branch that follows.

The unit test that fixes this raises only on the ROUTE's reads. `query_patient` reads `.kb` itself (2.1), so a test
that raised on every access would fail on the composer's own reads and would test nothing about the route. The test
composer exposes `.kb` through a guard (a property plus a list proxy over `__len__`, `__iter__` and `__getitem__`)
that inspects the caller frame: it raises when the caller's module is `research.runners.compositional_chain_route`
AND the route's decision scope is open, and passes every other access through (the composer's own frames in
`one_brain_composer`). Assertions: (a) with the guard armed, the flagged route returns the spiking result; (b) with
the decision scope closed (the audit call), the same guard records the access instead of raising; (c) replacing the
audit's return value with garbage leaves the route's result identical.

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
`query_patient` (a host first-match over decoded rows), the slot-to-word map, the cross-bridge relay, the mapping of
the gate's outputs onto hop outcomes (2.2), the derived framing (`frame_derived_answer`) and the verify re-parse
(`brain_chat_tui.py:1922-1942`). Under the conflict read the first-match answers only when the matching rows drive a
single patient line, so it has no patient choice left to make; the step-2 record checks per hop that the PROPOSE word
is the word of the single active patient line. M1 claims one thing: the identity of each hop's patient and the
abstain on ambiguity are decided by spiking reads of the bound store.

### 2.7 Engagement record, scoped to the hop reads

Instrumentation lives in the stage-2 runner's arm worker (behaviour-preserving wrappers, as the kb_scramble arms do).
A hop scope opens when the route enters `_hop_spiking` and closes when it returns. Per hop:
`{turn, hop, agent, relation, proposed (query_patient result), selected_block, source_tier, m_counts (per block),
blocks_matched (m{b} over match_thresh), patient_line_counts (per pP_w line with any spike), patient_lines_active,
acc_conf_spikes, slot1_drive, drives, ws_seed, n_ignited, conf, winner, gate_decision, gate_cycles, slot1_only,
committed, route_kb_reads, composer_kb_reads}`. `route_kb_reads` counts `.kb` accesses whose caller frame is
`compositional_chain_route` while a hop scope is open; `composer_kb_reads` counts those from `one_brain_composer`
(the clause test and the attribute check), listed by line. Turn-level counts (`_chain_facts`,
`rich_answer_composer._stored_facts`, elaboration) are reported separately and never barred.

### 2.8 Arms (stage-2 runner, `infer` group, all informational, never counted in the Stage-2 verdict)

Session tables: `infer` intact (the registered one) and three new tables, run under the same harness (numpy CPU,
tiny-demo +LTM, corpus present, `PYTHONHASHSEED=0`, `rich` omitted):
- `infer_amb1` (hop-1 ambiguity): heron hunts frog; frog eats gnat; heron hunts fish; then p_prey_eat, c_heron_hunt,
  c_dog_chase.
- `infer_amb2` (hop-2 ambiguity): heron hunts frog; frog eats gnat; frog eats fly; then the same probes.
- `infer_dup` (duplicate facts): the four intact teaches, then verbatim re-tells of the hop-1 fact ('the heron hunts
  the frog', label t_heron_hunt_re) and of the hop-2 fact ('the frog eats the gnat', t_frog_eat_re), then the
  `infer` probes.
  Each teach in the three tables contains a word already in the brain's vocabulary or the hand animacy table, so the
  D4 gate passes it at the default (stated, and read back from the teach turns).

| arm | table | flags | purpose |
|---|---|---|---|
| A0 off | infer | none (production) | today's route, the pairing side for answer rate |
| A1 m1 | infer | M1 on | the conclusion through spiking hops |
| A2 m1_b | infer | M1 on, independent rebuild | the null for A1 |
| A3 m1_kb_scramble | infer | M1 on + the existing `kb` derangement | who owns the conclusion now |
| A3a kb_scramble (pinned re-run) | infer | the existing informational arm, flag off | the list-owned reference, at the step-2 revision |
| A3b kb_scramble_spiking_control (pinned re-run) | infer | the existing informational arm, flag off | the bound-vector reference, at the step-2 revision |
| A4 m1_qp_lesion (+ .rep0) | infer | M1 on + qp patch | PROPOSE necessary |
| A5 m1_ws_lesion (+ .rep0) | infer | M1 on + WS lesion | workspace ignition necessary |
| A6 m1_seq | infer | M1 on + `BRAIN_INTEGRATED_LOOP=1` | sequencer selection, reported only |
| A7 amb1_off / A8 amb1_m1 | infer_amb1 | off / M1 on | hop-1 ambiguity, both routes |
| A9 amb2_off / A10 amb2_m1 | infer_amb2 | off / M1 on | hop-2 ambiguity, both routes |
| A11 amb1_conflict_lesion (+ .rep0) | infer_amb1 | M1 on + conflict lesion | the conflict unit necessary |
| A12 m1_permuted_gateoff | infer permuted | M1 on + `BRAIN_COMPREHENSION_GATE=0` | conclusion tracks exposure through spiking hops (departure, reference) |
| A13 dup_off / A14 dup_m1 | infer_dup | off / M1 on | a duplicated fact is not a conflict, both routes |

New arms are added as informational entries, so every pre-existing arm keeps its turn table, env, patch and kind,
pinned by the golden hash in `tests/test_stage2_kb_scramble.py`; `pool_job_lines` grows by the new arms only. A3a and
A3b are the existing arms, re-run at the step-2 revision, not new code.

### 2.9 Registered predictions and bars (dev read; numbers are per dev seed, 1-6)

Predictions, written before any M1 code exists:
- P1 (A1): p_prey_eat derived_svo `[frog, eat, gnat]`, derived_from `[[heron, hunt, frog], [frog, eat, gnat]]`,
  the "I derived this from" lead present. Same as A0 on these fields.
- P2 (A3): engagement as in A3a's record (3 scoped facts changed, read-back equals plan, patients distinct, the
  conclusion fact changed), and A3's derived_svo/derived_from equal A3b's on the same seed and revision (the
  bound-vector conclusion, `[frog, eat, gnat]`) and differ from A3a's (the list conclusion, `[frog, eat, wren]`). The
  answer text keeps list-read sentences from other readers, and c_frog_eat and p_yesno stay flagged guesses (the
  direct-recall verify reads `chat.stored_facts`, outside M1; reported, not barred).
- P3 (A4) and P4 (A5): p_prey_eat has no derived answer, reproduced on .rep0.
- P5 (A7-A10): p_prey_eat abstains on both routes in both ambiguity tables; in A8 and A10 the abstaining hop shows
  blocks_matched = 2, patient_lines_active = 2 and acc_conf_spikes > 0.
- P6 (A11): p_prey_eat is derived through the FIRST-taught patient (frog), reproduced on .rep0.
- P7 (A13, A14): p_prey_eat derived_svo `[frog, eat, gnat]` on both routes; in A14 both hops show blocks_matched = 2,
  patient_lines_active = 1, acc_conf_spikes = 0 and gate_decision ADVANCE.

Bars (each on every dev seed 1-6; design seed 7 is excluded from the count):
- **B1 conclusion:** P1 holds.
- **B2 ownership:** P2's two fields hold; the bar reads derived_svo and derived_from only. It is scored only against
  A3a and A3b run at the step-2 revision; until they exist B2 is UNDEFINED. A seed where A3a does not follow the list
  (the 10-26 result does not reproduce) is UNDEFINED for B2, and the non-reproduction is reported first.
- **B3 PROPOSE necessary:** P3 holds and reproduces.
- **B4 workspace necessary:** P4 holds and reproduces.
- **B5 multi-valued abstain:** P5 holds on both routes; engagement precondition: two decoded rows with two different
  decoded patients under the ambiguous cue (the default appended a re-told patient in the 10-12 dev-7 retell smoke; a
  REWRITE makes the seed UNDEFINED, never PASS).
- **B6 conflict unit necessary:** P6 holds and reproduces. Engagement precondition, on the same seed and the same hop
  (hop 1 of p_prey_eat): A8 shows blocks_matched = 2, patient_lines_active = 2 and acc_conf_spikes > 0 (the unit
  sees the conflict when intact), and A11 and its .rep0 show blocks_matched = 2, patient_lines_active = 2,
  acc_conf_spikes == 0 and slot1_drive == 0 (the lesion held). Otherwise the seed is UNDEFINED for B6, never PASS.
- **B7 hop scope:** both hops recorded on p_prey_eat in A1; `route_kb_reads` = 0 inside every hop scope;
  `query_patient` calls inside hop scopes = hops attempted; the query_patient selected block lies in
  `blocks_matched`, and the PROPOSE word is the word of the single active patient line, on every resolved hop.
- **B8 moat and controls:** c_prey_chase abstains; c_dog_chase recalls cat; c_heron_hunt and c_frog_eat primary
  fields equal A0.
- **B9 answer rate:** per label, flag-on (A1, A8, A10, A14) against flag-off (A0, A7, A9, A13) on p_prey_eat,
  c_heron_hunt, c_frog_eat, c_prey_chase, p_yesno, c_dog_chase: no label answered off and abstained on. Read by
  `tools/answer_rate.py --labels`, which the M1 build adds (section 7); FLIP-HURTS-ANSWERS is reported.
- **B10 flag-off identity:** with the flag unset, A0's primary fields equal an `intact_a` arm run at the base
  revision (before any M1 code) on the same seed, and the unit golden holds.
- **B11 duplicate is not a conflict:** P7 holds. Engagement precondition: on that seed the re-tells appended, i.e.
  A14's hop records show blocks_matched = 2 on both hops with one decoded patient; otherwise the seed is UNDEFINED for
  B11. If every seed is UNDEFINED, B11 is UNDEFINED (never PASS), and the circuit-level duplicate test of step 1 is
  reported as the only evidence.

Every lesion reports engaged / not engaged first (docs/TERMS.md, lesion); a lesion that did not hold is reported as
not held, never as no effect. All infer arms were seed-invariant in non-text fields at the 10-26 gate, so the
workspace and sequencer counts per seed (2.7) are reported to show whether six seeds are six measurements here.

### 2.10 M1 build order, with STOP lines

**Step 0, the engagement smoke (read-only; no production edit; dev seed 7, then 1-6).** In the arm worker, after
the teach turns of `infer`, `infer_amb1`, `infer_amb2` and `infer_dup`:
- (a) call `composer.query_patient` for (heron, hunt), (heron, eat), (frog, eat), (frog, chase) and record result,
  selected block and the decoded rows that match;
- (b) build the K-way sequencer for the buffer's K with the current drives (`_ensure_sequencer`) and record every
  block's `m{b}` count for the same cues, plus the sequencer decision against the host first-match;
- (c) for every matching block, read every main role's cleanup slice with the existing probe
  `_phaseC_task1_cleanup_probe.block_cleanup_scores_full`, check that its agent and action slices equal
  `block_cleanup_scores` on the same block, and record which patient word lines `scores_to_drive` at 0.9 lights;
- (d) record `chat._last_gnw_delib` on c_heron_hunt in `infer_amb1` (the single-hop conflict gate on onebrain);
- (e) record that the D4 gate stored every new-table teach, the row count under each ambiguous and each re-told cue,
  and `ws_seed` against the arm seed.
GO rule (written here, before the run): on 7/7 seeds, every true match drives its `m{b}` over `match_thresh` and no
non-matching block does; every matching block lights exactly one patient line, the stored patient's; the two blocks
of a re-told cue light the same line and the two blocks of an ambiguous cue light different lines; `ws_seed` equals
the arm seed. If a true match is missed on any seed, STOP: post the counts; the circuit is banked and the named next
carrier is the cs-race engram race read for co-latching (two latches ON = conflict). If a patient line is missing,
wrong or not unique, STOP: the patient identity the unit would count is not on the substrate at this fidelity. If the
re-tells did not append on any seed, B11 is predicted UNDEFINED and the step-1 circuit test carries the duplicate
case. Post `[c2-hops-0] RESULT`, STOP for Claude.

**Step 1, the build (default OFF).** `conflict_unit` in the sequencer builder (patient lines, `gp{b}` gates,
depressing `pP_w -> acc_conf`, every pre-existing pathway `stp_disabled`); the `with_patient` return of the cleanup
read; `_hop_spiking`, the audit call and the three flags in the route; `tools/answer_rate.py --labels` (per-label
reads over every turn of an arm file, with its own unit test). `w_c` and `n_conf_ref` set on seed 7 by this rule,
then frozen in code with the seed-7 counts in a comment: `acc_conf` silent for one line from one block and for one
line from two blocks (a re-told cue), and firing for two lines; the two-block-same-line input to `acc_conf` within
1.1x of the one-block input, and the two-line input at least 1.5x of it (a weight window exists). If either ratio
fails, STOP: the deduplication is not on the substrate; post the counts. Unit tests: flag-off golden; the
caller-frame `.kb` test (2.4); the lesion flags reach their targets; `conflict_unit=False` builds the sequencer
byte-identically; with `conflict_unit=True` the m{b} rates and the decision equal the `False` build on a fixture; a
direct `composer.store` of the same triple twice leaves `acc_conf` silent, and two distinct patients under one cue
make it fire. Module docstring corrected (it says the hops are list scans today; under the flag they are spiking
reads).

**Step 2, the dev read.** A0-A14 (with A3a and A3b) on dev seeds 1-6 through the pool at one pinned revision with
provenance on (numpy, memcap 10), scored against 2.9 by a scorer written before the runs. Post `[c2-hops-2] RESULT`,
STOP. A failed bar is a RESULT, never a retune.

**Step 3 (Claude).** A `flag: no` prereg carrying A1-A14 as informational arms of the 11-21 gate attempt, or a flip
packet if every bar holds; the default stays OFF until a flip prereg passes (smoke-first, flag-off battery arms on
the gate seeds, answer-rate).

## 3. M3: acquisition of well-formed all-novel transitives through a learned order cue (flag `BRAIN_COMPREHENSION_LEARNED_ORDER`, default OFF)

### 3.1 Why D4 refuses, and what it can read

`ComprehensionProductionOrgan.judge` (`comprehension_production_organ.py:738`) scores an assertion only when
`competent()` (:719) holds. `competent()` is the branch selection, and it is a host English table: an assertion is
fully covered when its verb is in VERB_SELECTS and both nouns in ANIMACY (the hand tables, or their corpus-learned
extensions, both default ON), and fully out-of-vocabulary when none of the three is in those tables and neither noun
is in `_brain_vocab(chat)` (`webapp/server.py:3640`, the agents, actions and patients of the stored facts); anything
else is passed through unjudged. So the tables and the vocabulary test decide which assertions D4, and M3 after it,
ever judge. The read drives the SEMANTIC cues (animacy, verbfit) of `SpikingRoleCompetition`. For a fully-OOV sentence
both cues are absent, so the margin is 0.0 by construction and every such sentence is refused, well-formed or not.
The organ's learned position cue is not used by D4, and it would not help: its value is a host vote by noun index
(`_position_vote`, `_phaseB_multicue_competition_spiking_derisk.py:109`) that separates noun 0 from noun 1 in ANY
3-token string, garbled or not. It is registered below as a null that the garbled bar must kill.

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
- **Carrier (ruled, section 7).** The traces must outlive the per-word reset. M3's carrier is the cs-race persistent
  word trace now in flight, feeding the word-role parser: the f1h held apical plateau
  (`research/coordination/cs-race-f1h-PACKET.md`; its step-0 RESULT on main: the plateau held, C0 8/8 on every grid
  cell, and the older word's plateau never ended, C3 0/8) and the f1i dendrite-targeting (SOM-like) inhibition that
  ends the older word's plateau (`research/coordination/cs-race-f1i-PACKET.md`, on main). M3 step 1 waits for those
  results. The frame packet's NMDA trace stays parked under RULING focus-1025 and is not M3's carrier.

### 3.3 D4 integration (default OFF)

Under the flag, only the fully-OOV branch of `judge()` changes: it reads the M3 margin over the raw token sequence
instead of the SEMANTIC margin. The branch selection itself (`competent()`, 3.1) is unchanged and stays host. The
fully-covered branch, the out-of-competence pass-through and the known-fact honour rule are untouched (byte-identical
for every input that is not fully OOV). The M3 threshold is calibrated at build from the exposure statements re-heard
(in-sample well-formed only; no garbled item enters calibration), as D4's own build calibration does. The stored
triple keeps the production store path; the M3 commit roles are recorded beside it, and a teach is stored only when
the two agree (a disagreement refuses; counted and reported).

### 3.4 Tables: what M3 adds and what stays

M3 adds no word to any table and reads no word list. The test items are nonce CVC forms drawn by a fixed-seed
generator, never in ANIMACY, VERB_SELECTS, the learned lexicons or the corpus vocabulary (a CI grep confirms they
appear in no production module). What stays host, declared in the header: the `competent()` branch selection,
`extract_transitive`'s decision of which utterances reach the read (3 content tokens after its function-word strip),
the store path's positional role assignment, and the onset event. M3 makes no claim about English tables beyond
adding none.

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

**Outside D4's reach, stated.** "ko" is not in `extract_transitive`'s function-word list, so 'ko blick gorps ko dax'
has five content tokens and `extract_transitive` returns None: D4 never sees it, and under M3 such a sentence still
passes to the store path untouched at the reply level. The nonce-determiner items (OM4) and the order-permuted items
(OM3, OM5) are heard by the organ directly; they are organ properties, never a reply-level claim.

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
- **m3_rs_lesion (+ .rep0):** the frozen permuted `infer` table with M3 on and the role-sequence lesion
  (`BRAIN_COMPREHENSION_LEARNED_ORDER_LESION=1`: RT->role and O->role closed at read). Prediction: t_heron_hunt
  reverts to refused, as at the production default (not acquired), and p_prey_eat has no derived answer, reproduced on
  .rep0. Engagement precondition: t_heron_hunt's `comprehension` record is non-null with `in_scope` true and carries
  the M3 read marked lesioned, and the closed gates are verified closed at read; otherwise the seed is UNDEFINED.
- **m3_garbled / off_garbled / gateoff_garbled:** a new table `infer_garbled`: the four intact teaches, then three
  garbled teaches, each built from three never-taught nonce CVC words, disjoint from each other and from every other
  word of the session (so a G teach the gate-off brain stores cannot make a later G teach known):
  - G1 (det N det N V) 'the zib the vop tavs', then the probe 'what does the zib vop';
  - G2 (V det N det N) 'pels the kig the dov', then 'what does the pels kig';
  - G3 (det N V N det) 'the bim ruvs fep the', then 'what does the bim ruvs'.
  Each probe cues the (agent, action) that the store path's positional parse would record. Step 0 checks the nine
  forms against ANIMACY, VERB_SELECTS, the learned lexicons and the corpus vocabulary; a form found there is redrawn
  by the fixed-seed generator before the table is pinned. Engagement precondition: on off_garbled, every G teach
  turn's `comprehension` record is non-null with `in_scope` true (D4 judged it in its fully-OOV branch); on
  m3_garbled, the same, carrying the M3 margin. Otherwise the seed is UNDEFINED. Prediction: every G teach refused
  under M3 and under the production default, and STORED with the gate off (the teeth: on a seed where the gate-off
  brain stores fewer than 2 of the 3, the table has no teeth and the seed is UNDEFINED for R2).
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
in m3_garbled is refused and no later probe recalls it (with the 3.6 precondition and teeth); R3 m3_intact equals
flag-off on primary fields; R4 no FLIP-HURTS-ANSWERS; R5 flag unset is byte-identical; R6 m3_rs_lesion refuses
t_heron_hunt and derives nothing on p_prey_eat, on every seed and its .rep0 (with the 3.6 precondition).

### 3.8 M3 build order, dependencies, STOP lines

- **Step 0 (read-only, now, parallel to M1 step 0).** (a) The D4 decision (scope, competence branch, margin) on
  every W, G and out-of-scope item at production defaults; (b) the word-role parser's commit read on W and G with
  today's organ (prediction: nothing commits, since position alone is subthreshold by design); (c) the carrier status
  read from the cs-race f1h step-0 RESULT and, when posted, the f1i RESULT; (d) the nine `infer_garbled` nonce forms
  checked as in 3.6. Post `[c2-acq-0] RESULT`, STOP.
- **Step 1 (organ build, default OFF)** starts only when the cs-race carrier (f1h plus f1i) holds a trace across the
  per-word reset and ends the older one.
- **Step 2:** the organ battery and bar sanity on dev seeds; post RESULT, STOP.
- **Step 3:** D4 integration and the reply-level dev read; post RESULT, STOP. Then Claude writes the preregs. Step 3
  is the third build step under this header and waits for the cap ruling (cap accounting, top).

Dependency, stated plainly: the carrier is open. Claude's ruling (section 7) names the cs-race persistent word trace
as M3's carrier; M3 step 1 waits for its results, and M1 proceeds independently. M3 is not expected to land before
the 11-21 attempt. No-defer: M3 continues after 11-21 if it misses.

## 4. What neither method does

- **The 11-21 C2 verdict.** On the frozen permuted arm, M1 alone leaves the D4 refusal and 0/6 in place. Claude has
  ruled the instrument side (section 7): `score_infer` gets an acquisition precondition like ask's A4, declared as a
  re-pin amendment before any re-attempt data, so a refused permuted teach makes the permuted arm UNDEFINED. UNDEFINED
  is not a pass: C2 still needs the brain to acquire well-formed novel transitives, which is M3's job. The alternative
  of a new prereg with an acquirable permuted exposure is rejected (section 7).
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
3. M3 steps 1-2 after the cs-race carrier results (f1h, f1i). M3 step 3 after the cap ruling.
4. Independently of both: the `score_infer` acquisition-precondition amendment, before any 11-21 re-attempt data.
The driver runs the steps and posts the RESULT lines; Claude rules at every STOP, writes the preregs, reviews and
merges.

## 7. Claude ruling 2026-10-27

(Claude, 2026-10-27; written into this revision as given.)
- **C2 instrument.** The C2 instrument gets an acquisition precondition in `score_infer`: a refused teach makes the
  permuted arm UNDEFINED, not "not exposure-driven". It is declared as a re-pin amendment to
  `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` BEFORE any re-attempt data. UNDEFINED is not a
  pass: Stage-2 C2 still needs the brain to acquire well-formed novel transitives.
- **Rejected.** The option "a new prereg with an acquirable permuted exposure" (choosing test words the hand tables
  already cover) is REJECTED: it selects test items around a known host-table limit.
- **M3's carrier.** M3's carrier is the cs-race persistent word trace now in flight (f1h held apical plateau; f1i
  SOM-like dendritic overwrite; `research/coordination/cs-race-f1h-PACKET.md` and `cs-race-f1i-PACKET.md`) feeding
  word-role. M3 step 1 waits for those results; M1 proceeds independently.
- **Answer rate.** `tools/answer_rate.py --labels` mode is part of the M1 build.
- Not ruled here: the cap. The third build step's cap ruling is asked for at the top of this packet.
