# PACKET c2-spiking-hops (Claude, 2026-10-27; revised twice the same day after two skeptics): Stage-2 C2, two separate methods: hop reads on the spiking store (M1) and acquisition of all-novel transitives through a learned order cue (M3)

host-decision: scaffold
decides: M1: the possessive-question dispatch (_POSSESSIVE_CHAIN_RE), the hop-1 relation candidates (_ROLE_NOUN_HINTS plus the same-verb fallback), the two-hop relation order taken from that parse, the block selection inside query_patient at the production default (a host first-match over spiking-decoded rows), the sequencer's reduced vocabularies built from the host fact list (_ensure_sequencer's self.kb[:K] read, one_brain_composer.py:1654, which decides which stored agent, action and patient words get word lines) and the host abstain of the read when a cue word has no line (:1709), the relay of each block's cleanup scores into the sequencer (block_cleanup_scores read to host and clipped at 0, injected as input_gain x score into the divnorm score bridge by make_block_drives and onbridge_divnorm_drive, where the on-bridge divisive normalisation and the Izhikevich rheobase decide which word pools fire, then turned by the host into a fixed 1500 pA decoded-line drive and remapped to the reduced vocabularies with a lit word outside them dropped), the per-query STP reset of the conflict synapses, the slot-to-word map of the workspace, the relay of the conflict unit's spike count into a workspace slot drive, the mapping of acc_conflict_gate's four outputs onto hop outcomes (section 2.2), the cross-tier candidate slots (all_candidate_patients on the routed LTM shard, kept when the host string differs from the PROPOSE word, each driven into its own workspace slot), and, for an LTM-sourced hop (the TieredFactStore's tier routing picks the source), the conflict enumeration all_candidate_patients (host closed-form unbind with a host string dedup); M3: the competent() branch selection of ComprehensionProductionOrgan, which is a host English table (the hand ANIMACY and VERB_SELECTS coverage with their corpus-learned extensions, plus the _brain_vocab membership test, decide whether an assertion is fully covered, fully out-of-vocabulary or passed through, and so decide which assertions M3 ever judges), which utterances the comprehension read judges (extract_transitive's 3-content-token test and its function-word list), the role assignment of the stored triple (the positional parse after the host strip), the store-or-refuse agreement check between M3's commit roles and that positional parse, the M3 margin arithmetic (minimum over roles of winner-minus-runner-up, median-normalised) and its calibrated threshold (0.5 times the 10th percentile of the in-sample well-formed normalised margins), and the utterance-onset event
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement)
replaced-by: the learned relation-noun binding (M2, its own packet), the cs-race spiking store competition for block selection, the word-role frame commit for scope and stored roles, and one-bridge co-residence of the K-way sequencer and the GNW workspace
cap: 2 builds

Cap accounting (a ruling is asked for, not assumed). Three build steps ship code under this one header: M1 step 1
(2.10), M3 step 1 (the organ, 3.8) and M3 step 3 (the D4 integration, 3.8). The cap of 2 covers M1 step 1 and M3
step 1. CLASS HD lets a cap only go down, so the header is not raised. The gate counts prereg amendments as its proxy
and does not count packet steps (`tools/gates/host_decision_declared.py`, limit (4)); the count is review's job. M3
step 3 does not start until Claude posts a keyed section-6 RULING in the runbook on main that grants +1 build to this
scaffold group, in the gate's form (a `cap-raise` line naming the M3 prereg file once that file exists, which the
prereg then cites by the RULING's key). The ask is `[c2-hops-cap]` (section 7). If the ruling refuses, M3 step 3 does
not run under this method.

Status: DESIGN ONLY, for Claude's review. No production code, no default flipped, no `sim/` edit. Gate seeds
42/43/44/100/101/102 are not run by anything here. Branch `research/c2-spiking-hops` (cut from main cadf0f40f; first
draft a7799b30c). The first skeptic REFUTED a7799b30c (fixes in 0b; self-skeptic pass in 0c). A second skeptic
REFUTED the revision fca06bd9a; every one of its fixes is applied (0d) and this revision's own self-skeptic pass is in
0e. Section 7 copies Claude's 2026-10-27 ruling text; the keyed runbook RULING, which Claude posts, is the
authoritative record. Ruling this answers: stage2-fixes-1026, C2 item ("Then two separate methods", finding
2026-10-26-stage2-fail-diagnoses.md:137-141). M2 (a learned relation-noun binding in place of `_ROLE_NOUN_HINTS`) is
NOT in this packet; nothing below claims it, and no bar here needs it.

Inputs read: `research/findings/2026-10-26-stage2-fail-diagnoses.md` (C2 section);
`research/findings/2026-10-26-c2-kb-scramble-host-ownership.md`; the C2 diagnosis and its skeptic's verdict
(workstation `sim-rev/diag_1026/workflow_result.json`, `result[0]`, keys `diagnosis.next_method` and
`verdict.next_method_concerns`); both packet skeptics' REFUTED verdicts (0b, 0d);
`research/runners/compositional_chain_route.py` (`_distinct_patients` :196, `_hop` :244);
`research/runners/_gnw_reentrant_metacog_gated_deliberation_derisk.py` (`_ignite_and_read_nmda` :158,
`_conf_from_nmda` :204, `_deliberate_hop_conf` :222, `acc_conflict_gate` :326, `confidence_gated_chase` :340) and its
production users `webapp/gnw_multistep_deliberation.py` (C5) and `webapp/gnw_deliberation.py`
(`all_candidate_patients` :124, `conflict_gate` :144); `research/runners/one_brain_composer.py` (`_seq_imports` :51,
`_ensure_sequencer` :1641, `_seq_block` :1680); `research/runners/_seq_vocab_shrink_derisk.py`
(`build_sequencerK_reduced_bridge` :98, `reduced_cue_vocab` :183, `run_sequencerK_reduced_with_drive` :261);
`research/runners/_phaseB_onebrain_sequencerK_derisk.py` (`build_sequencerK_bridge` :66, `reset_sequencerK_state`
:179); `research/runners/_phaseB_onebrain_sequencer_derisk.py` (`block_cleanup_scores` :64, `scores_to_drive` :92);
`research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py` (`make_block_drives` :198);
`research/runners/_phaseC_S5_divnorm_derisk.py` (`build_divnorm_score_bridge` :115, `onbridge_divnorm_drive` :158);
`research/runners/_phaseC_task1_cleanup_probe.py` (`block_cleanup_scores_full` :33);
`research/runners/_seq_fused_fabric.py` (`fused_seq_block` :336); `research/runners/tiered_fact_store.py`
(`query_patient_source` :197); `research/runners/sharded_phasor_store.py` (`shard_for` :166); `sim/regions.py`
(`RegionPathway.stp_disabled` :572); `sim/config.py` (STP defaults :902-905 and per-type :1295-1298); `sim/bridge.py`
(STP state init :1288-1289); `research/runners/stage2_exit_gate.py` (`infer` group :234-270, `run_worker` :659,
`teach_acquired` :1024, `score_infer` :1107, `kb_scramble_read` :1330);
`research/runners/comprehension_production_organ.py` (`_verb_selects_of` :209, `_animacy_of` :233,
`extract_transitive` :339, `ensure_built` :631, `competent` :719, `judge` :738); `webapp/server.py` (`_brain_vocab`
:3640, `_integrated_loop_enabled` :3913, the D4 call :6844);
`research/findings/2026-10-12-validation-battery-prereg.md` (B-C5e, the dev-7 re-tell smoke);
`docs/drafts/word-role-frame/PACKET.md`, RULING wrc-frame-1-1024 and RULING batch-1026 (5) (the wrc-frame-1b
diagnosis); `docs/drafts/cs-race/PACKET.md`, `research/coordination/cs-race-f1h-PACKET.md`, and on main (not on this
branch) `research/coordination/cs-race-f1h-0-RESULT.md`, RULING f1h-0-1027 and `cs-race-f1i-PACKET.md`.

## 0. The diagnosis skeptic's concerns, and where each is answered

| # | concern (verdict.next_method_concerns, condensed) | answer |
|---|---|---|
| 1 | M1 does not touch the cause of 0/6 (the D4 refusal of the permuted teach) | Agreed. M1 is the ownership method; M3 is the acquisition method. Sections 4 and 7 say what the 11-21 C2 verdict depends on. |
| 2 | `confidence_gated_chase` takes one `rel`, PROPOSE is first-match, distractors are host-random | M1 does not reuse the chase loop or `_deliberate_hop_conf`. It builds its own workspace drive vector and calls `_ignite_and_read_nmda`, `_conf_from_nmda` and `acc_conflict_gate` directly, as `conflict_gate` does, with NO distractors (2.2). |
| 3 | `query_patient` is first-match; onebrain has no `_iter_facts`, so `all_candidate_patients` is empty; the multi-valued moat would silently regress | The conflict read is a named new circuit on the K-way sequencer's match pools that counts distinct patient assemblies (2.3), plus counted multi-valued-hop arms scored on BOTH routes (A7-A10), a duplicate-fact pair (A13-A14) and a cross-tier pair (A15-A16). |
| 4 | Not free of English tables until M2; the madeup_role bar bundles M1 and M2 | M1 claims nothing about tables; `_ROLE_NOUN_HINTS` is declared in the header; no madeup_role bar here. |
| 5 | Turn-scoped engagement bars are already met or unreachable | Counters are scoped to the hop reads inside the route, with per-hop records (2.7). |
| 6 | kb_scramble bar only on derived_svo/derived_from; derangement with distinct patients; engagement read-back | Adopted; the existing arm machinery does the derangement and read-back, and the reference arms are re-run pinned (2.8, A3/A3a/A3b). |
| 7 | No flag-off/answer-rate arms; M3 needs a gate-off arm and a gibberish arm; the word-role flip is HELD | Flag-off, gate-off, garbled and lesion arms are in 2.8 and 3.6; M3's carrier is in section 7's ruling text. |
| 8 | M3 as scoped does not touch the role-branch refusals (goat/hay, crab/leaf) | Agreed and kept out of M3's bars; reported arm with a registered prediction (3.6, R-goat). |

## 0b. The first packet skeptic's required fixes (REFUTED a7799b30c), and where each is applied

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

## 0c. Self-skeptic pass on the first revision (Claude as refuter, 2026-10-27)

| # | finding | fix |
|---|---|---|
| S1 | The M3 read required "no closed-class token commits a role", which needs a host closed-class word list inside the brain's decision, against "M3 reads no word list" | 3.2: the brain's read counts role commits only; the closed-class check moved to the scorer |
| S2 | Undeclared host decisions: the cleanup-score relay into the sequencer, the LTM-hop enumeration with its string dedup, M3's store-or-refuse agreement check, the M3 margin arithmetic and threshold | header `decides:`, 2.6, 3.2, 3.3 (the relay as then described was wrong; corrected in 0d P1) |
| S3 | Arms passable without engagement: B1, A3, B3, B4, B5, B9, R1; A12 had no prediction | 2.9 (preconditions, general `ws_seed` rule, P8), 3.6 (R1) |
| S4 | The M3 role-sequence lesion closes two projections; the carrier lesion is not specific to the order cue; the transplant had no engagement check | 3.5 |
| S5 | The route's reads were not stated | 2.3: the route reads only `acc_conf`'s count and the workspace read; the rest is record-only |

## 0d. The second packet skeptic's required fixes (REFUTED fca06bd9a), and where each is applied

Each cited file:line was re-read for this revision before it was written down.

| # | fix required | where |
|---|---|---|
| P1 | M1's default sequencer is `build_sequencerK_reduced_bridge` (`enable_seq_vocab_shrink=True`, `one_brain_composer.py:133`, branch :1651), not `build_sequencerK_bridge`, plus a fused branch; the real relay is `make_block_drives` with on-bridge divnorm, `input_gain` and the rheobase threshold of `onbridge_divnorm_drive`, not `scores_to_drive` at 0.9 | 2.1, 2.3 (match stage, relay, fused branch; `conflict_unit` in both builders), 2.6, 2.10 step 0 (b)-(c), header `decides:` |
| P2 | `_ensure_sequencer`'s `self.kb[:K]` read (:1654) builds the reduced vocabulary; a cue word without a line returns None (:1709) | header `decides:`, 2.3 (cue outside the reduced vocabulary), 2.7 (`composer_kb_reads` by line) |
| P3 | `reset_sequencerK_state` never resets STP and drains only 20 steps against tau_d 200 ms | 2.3 (STP reset at every read), 2.10 step 1 (repeat-cue, hop-order and post-query window tests) |
| P4 | The cross-tier case (buffer one patient, LTM another) is invisible to a buffer-only read | 2.2, 2.3 (cross-tier slots), 2.7 (`audit_disagree`, `xtier_disagree`), 2.8 (`infer_xtier`, A15/A16), 2.9 (P9, B12) |
| P5 | G5 must exclude every frame-conforming order under either noun assignment, with one exclusion shared by generator and scorer | 3.5 |
| P6 | M3 learning-engagement precondition (wrc-frame-1b's 1/5760 starvation); OM2/R2 only where OM1/R1 hold; B5 only where A1 hops ADVANCE | 3.5, 3.7, 2.9 (B5) |
| P7 | The M3 threshold rule, the `w_c` selection rule and the `n_conf_ref` fixture, written before data | 3.3, 2.10 step 1 |
| P8 | The f1h citation was wrong | 3.2 (carrier) |
| P9 | The 'contains a known word' rationale was false; the nonce check needs `_brain_vocab(chat)` after the intact teaches | 2.8 (per-teach `competent()` branch table; 'the frog eats the fly' replaced), 3.6, 2.10 step 0 (e) |
| P10 | Step 0 (c) must compare probe and production after `np.maximum(., 0)` over non-reserve slots only | 2.10 step 0 (c) |
| P11 | The in-packet ruling is not a tracking mechanism | section 7 (keyed runbook RULING authoritative; the asks listed), cap paragraph |
| P12 | Conflict-unit cell count and workspace-build memory and latency; a reduced patient vocabulary | 2.3 (V'_P), 2.10 step 1, section 5 |
| m1 | Infer arms are seed-invariant (one effective measurement) | 2.9 (checked, and how it is reported) |
| m2 | The gate-off shards' hosts | section 1 |

## 0e. Self-skeptic pass on this revision (Claude as refuter, 2026-10-27, on 2901b314c)

Same three questions as 0c, plus: does every new rule have a reference that can actually move?

| # | finding | fix |
|---|---|---|
| T1 | B12 (b) barred `audit_disagree` on every M1 arm, but A3's `.kb` is deranged by design (its audit is no oracle) and A11's lesion blinds the unit by design (an ADVANCE on a two-patient cue there is B6's evidence): B12 would fail by construction | 2.9 B12 (b): A3 and A11 excluded with reasons; their hops still listed |
| T2 | The audit counted the buffer and the ROUTED shard only; the fix asked for every shard | 2.3, 2.7: `audit_ltm_all` (record-only scan of `ltm.shards`); `audit_disagree` uses the larger LTM count |
| T3 | The learning-engagement floor was "10% of word->role's update steps", but word->role is frozen while O->role and RT->role learn (3.2), so the floor was 0 and the precondition vacuous | 3.5: floor = n_stmt update steps (one co-firing per statement per projection), sum dw > 0, and the 90% co-firing fraction; word->role's frame-packet count is context only |
| T4 | "conflict_unit in both builders" left the full-V builder's patient layer size and rebuild condition unstated | 2.3: V'_P in both builders, in both rebuild branches |
| T5 | The `w_c` grid (5-600, linear) could miss the window for range reasons alone, and STOP on the grid instead of the physics; the ratio checks named no `w_c` | 2.10 step 1: 64 log-spaced points 1-3000, one 10x extension at an edge, geometric midpoint; ratios at the chosen `w_c` |
| T6 | The `infer_xtier` LTM write was "before the first turn" without saying the brain exists by then | 2.8: after the chat brain is built, before its first turn |
| T7 | Section 4 said a spiking cross-tier read waits for "co-residence of the buffer and the LTM shard" and attributed that to the `replaced-by` line, which names no such thing | section 4: the cross-tier slots and the LTM-hop enumeration are retired by the cs-race store competition once it spans the LTM tier, which `replaced-by` does name; `replaced-by` itself is unchanged, so the scaffold keeps its group identity |

## 1. The defect, as measured

- **The 0/6 is an acquisition refusal.** The default-ON D4 comprehension gate refuses 'the heron hunts the gnat' as
  fully out-of-vocabulary (margin 0.0 against 0.2), so the permuted premise is never stored (finding 2026-10-26, C2).
- **With the gate off the permuted exposure is acquired and the conclusion tracks it.** Diagnostic arms with
  `BRAIN_COMPREHENSION_GATE=0` on dev seeds 1-3 (committed:
  `research/findings/raw/_stage2_diag/c2_comprehension_gateoff/s1/infer__permuted_gateoff.json`, same for s2, s3):
  t_heron_hunt recalls heron-hunt-gnat, p_prey_eat is derived `[gnat, eat, frog]` from `heron hunt gnat ; gnat eat
  frog`, c_frog_eat abstains. The intact arms with the gate off (`.../s1/infer__intact_a_gateoff.json`, s2, s3) are
  unchanged: derived `[frog, eat, gnat]`. In ask, 'the goat eats the hay' is stored with the gate off
  (`.../s1/ask__intact_a_gateoff.json`, s2, s3). **Hosts, stated:** the nine shards ran at the Stage-2 pin 9bdb92ee8
  on mixed hosts, per their sidecars: pool nodes node-dl5d243, node-dl4g243 and node-mxl1462drl, and the workstation
  pool lane (dant123-wk) for s1 `infer__permuted_gateoff` and s3 `infer__intact_a_gateoff`; every shard records
  `corpus_present` true. Dev seed 7 showed the same permuted result on the workstation with the corpus absent
  (diagnostic, uncommitted). The gate off is a departure from production, used here only as a reference.
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
priority WTA (block i inhibits every j>i), built by `_ensure_sequencer` (:1641). At the default
`enable_seq_vocab_shrink=True` (:133, branch :1651-1663) that is the REDUCED fabric,
`build_sequencerK_reduced_bridge` (`_seq_vocab_shrink_derisk.py:98`), with word lines only for the distinct stored
agents and actions; `build_sequencerK_bridge` over all V words is built only with that option off (:1665-1670). A
third branch, `integrated_loop == "fused"`, selects through `fused_seq_block` (`_seq_fused_fabric.py:336`); the chat
server never takes it, because `_integrated_loop_enabled` returns a bool (`webapp/server.py:3913`, passed at :4377 and
:4470). In every case a second patient under the same cue is invisible to the caller, and the patient CONTENT is the
decoded bound vector. `query_patient` reads `.kb` itself: `len(self.kb)` in `_read_blocks` (:1389) and
`_fact_shard_active` (:1398), `self.kb[idx]` to route a clause patient (:1822), and the attribute check in
`_attributed_patient` (:1782); `_ensure_sequencer` reads `self.kb[:K]` (:1654).

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
2. **CONFLICT READ.** The conflict unit of 2.3 reads, in spikes, how many DISTINCT patients the matching buffer blocks
   carry, and the relay turns its spike count into `slot1_drive`. For a buffer-sourced hop with an LTM present, the
   cross-tier candidates of 2.3 are listed.
3. **EVALUATE.** One workspace read on the warm P1.2 workspace the C5 chase uses: `_get_bridge(_DEFAULT_SEED,
   lesion)` and `_get_theta(_DEFAULT_SEED)` from `webapp/gnw_deliberation.py`. `_DEFAULT_SEED` reads
   `BRAIN_CHAT_SEED` at import, which the stage-2 worker sets to the arm's seed before it imports the server
   (`run_worker`, :666); it is recorded per hop as `ws_seed`. `_deliberate_hop_conf` is NOT used: it builds its own
   drive vector (slot 0 at `IGNITE_PA`, distractor slots at `IGNITE_PA * DISTRACTOR_FRAC`) and has no argument for a
   graded slot-1 drive. M1 builds the vector itself, as `conflict_gate` (`webapp/gnw_deliberation.py:144`) does:
   `drives = [0.0] * n`; `drives[0] = IGNITE_PA` when `target` is not None; `drives[1] = slot1_drive`;
   `drives[2 + i] = IGNITE_PA` for each cross-tier candidate i (at most n - 2, the dropped count recorded); every
   other slot 0 (no distractors, so the C5 chase's host-random `_pick_distractors` is not used). Then
   `rates, g_nmda = _ignite_and_read_nmda(bridge, xp, slots, snap, drives)`,
   `conf, winner, n_ignited = _conf_from_nmda(rates, g_nmda)`, and
   `acc_conflict_gate(conf, n_ignited, cycles_on_hop, R_MAX_DEFAULT, theta_hi, theta_lo)` with `conflict_gate`'s
   retry walk. The read restores the same snapshot and is deterministic, so a RETRY repeats the same read and the
   walk only counts the budget (as in `conflict_gate`); the count is recorded as `gate_cycles`.
4. **BROADCAST.** The next hop is cued by the word of the winning slot, which under the table below can only be slot
   0; a hop that does not ignite cannot pass its candidate on.

The gate's outputs, from its own code (`_gnw_reentrant_metacog_gated_deliberation_derisk.py:326-337`; `theta_lo` is
passed and not read), and what each means for a hop:

| `acc_conflict_gate` returns | when | M1 hop outcome |
|---|---|---|
| COMMIT | `n_ignited == 0` | Hop MISS: no patient. In the chase COMMIT means "terminal: keep the last resolved word"; here no word is committed. The route treats it as `_hop`'s `(None, False)`: hop 1 tries the next candidate relation; a hop-2 miss tries the next hop-1 candidate, as today. |
| ADVANCE | `n_ignited == 1` and `conf >= theta_hi` | Hop RESOLVED with the word of slot 0 (the PROPOSE word). A winner in slot 1 (no word) or in a cross-tier slot ABSTAINS (recorded `winner_kind`). |
| RETRY | `n_ignited >= 2` and `cycles_on_hop < R_max` | The walk continues (same deterministic read) until R_max (3). |
| ABSTAIN | `n_ignited >= 2` at R_max, or `n_ignited == 1` with `conf < theta_hi` | The WHOLE chain abstains (the route's ambiguity rule). The second case is the graded one: a slot-1 drive below ignition can raise slot 1's NMDA enough to pull `conf` under `theta_hi` while only slot 0 ignites. The chain then abstains on the first read, with no retry. |

Registered regimes: one distinct patient and no cross-tier candidate gives `slot1_drive = 0` and slot 0 alone, which
is `calibrate_theta`'s SOLO regime, so the gate returns ADVANCE. Two distinct buffer patients give `slot1_drive =
IGNITE_PA`, and one buffer patient with one different LTM patient gives a cross-tier slot at `IGNITE_PA`; both are its
CONFLICT regime, so the gate returns RETRY to R_max, then ABSTAIN. An intermediate count can take either ABSTAIN path
or ADVANCE; every hop's `(slot1_drive, xtier_slots, n_ignited, conf, decision)` is recorded, and the step-2 report
lists every intermediate case.

The control flow over hops (two hops; candidate relations in order; abstain on any ambiguity) is the route's own and
is declared in the header. The C5 chase loop is not used: its single `rel` and emergent depth do not fit a question
whose grammar fixes two relations.

### 2.3 The conflict read: which circuit

**Match stage, existing (the default path).** M1's conflict read uses the sequencer `_ensure_sequencer` builds
(`one_brain_composer.py:1641`), whether or not `integrated_loop` is on. At the default `enable_seq_vocab_shrink=True`
(:133) that is `build_sequencerK_reduced_bridge` (`_seq_vocab_shrink_derisk.py:98`): role A has word lines only for
the distinct stored agents and role X only for the distinct stored actions, listed by `reduced_cue_vocab` (:183) from
`self.kb[:K]` (:1654), a host read of the fact list (declared). With the option off it is `build_sequencerK_bridge`
over all V words. The new `conflict_unit` option is added to BOTH builders with the same semantics, its patient layer
over the reduced patient vocabulary V'_P in both, and V'_P joins the rebuild condition in both branches of
`_ensure_sequencer` (:1651-1670); the reduced builder is the path every arm runs. Per block b, decoded agent and action word lines each pass a gate `g{b}{role}_w` that the cue
word line opens, so `mw{b}{role}_w` fires only when the decoded word equals the cue; role OR-pools `mA{b}`, `mX{b}`
feed a block AND pool `m{b}` through gate `gblk{b}`, so `m{b}` fires only when agent AND action match. The `m{b}` pools
count matching BLOCKS. Since the store appends a block per re-tell (2.1), a block count would read a duplicate re-tell
of one fact as a conflict and abstain where today's route commits. That would be a regression and not a conflict.

**The relay into the sequencer, as it is (declared).** The decoded lines are not driven from the cleanup membrane on
the substrate. `_ensure_sequencer` (:1672-1679) reads each block's cleanup membrane to host (`block_cleanup_scores`,
`_phaseB_onebrain_sequencer_derisk.py:64`, clipped at 0 at :87-88). `make_block_drives`
(`_phaseB_onebrain_sequencerK_k32_margin_derisk.py:198`, retreat "divnorm", peak_mult 1.0) then calls
`onbridge_divnorm_drive` (`_phaseC_S5_divnorm_derisk.py:158`) once per role: the host injects `input_gain * max(score,
0)` (input_gain 1.0, the composer default at :132) as current into V word pools of the divnorm score bridge
(`build_divnorm_score_bridge` :115, sigma 1.0, gain 0.11); the bridge divides each pool's drive by the per-query mean
on every step; a word pool counts as driven iff it fired at least once in a 20-step settle, so the Izhikevich rheobase
is the threshold. The host turns "fired" into a fixed 1500 pA decoded-line drive (:181), and
`run_sequencerK_reduced_with_drive` (`_seq_vocab_shrink_derisk.py:261`) remaps it into the reduced index space and
drops a lit word outside the reduced vocabulary. `scores_to_drive` at drive_frac 0.9 belongs to the older Phase-B run
(`run_sequencerK_reduced`, :199) and is NOT on this path. The patient lines below use the same relay.

**What conflicts is responses, not traces.** Response conflict is co-activation of incompatible response channels
(Botvinick, Braver, Barch, Carter and Cohen 2001, Psychol Rev 108:624). Here the responses are the candidate patients.
Two memories that support the same patient do not conflict, so the unit counts distinct patient assemblies.

New, additive builder option `conflict_unit=True` (default False; the sequencer is byte-identical when it is off):
- **Reduced patient vocabulary.** V'_P is the distinct stored patients over the same `self.kb[:K]` read, in first-
  appearance order (the same rule as `reduced_cue_vocab`), and it joins the fabric's rebuild signature, so the fabric
  is rebuilt when the stored patient set changes. A patient line exists only for a stored patient (section 5 gives
  the cell count; the full-V layer would be about V/V'_P times larger).
- **Patient decoded lines.** Per block b, word lines `d{b}P_w` for w in V'_P, driven by the relay above from the
  block's PATIENT cleanup slice. That slice is already computed by the same per-block read: `_seq_cleanup_conns`
  installs cleanup synapses for every main role, and `block_cleanup_scores` returns only the agent and action slices.
  The build adds an optional `with_patient=True` return of the patient slice (no extra substrate run) and an optional
  third role in `make_block_drives` and `run_sequencerK_reduced_with_drive` (each default off, byte-identical). A lit
  patient word outside V'_P is dropped and counted (`spurious_P`, recorded per hop).
- **Shared patient lines.** One line `pP_w` per word of V'_P (`n_word` = 20 cells), shared by all blocks: one lexical
  assembly per word, however many memories point at it. `d{b}P_w -> pP_w` passes a per-block transmission gate
  `gp{b}` opened by `m{b}` firing (`couple_gate_to_pool`, the primitive the sequencer already uses for `gblk{b}`), so
  only a matching block's patient reaches the patient lines. `reset_sequencerK_state` closes every transmission gate
  at query start, `gp{b}` included.
- **The conflict population.** `acc_conf` (20 Izhikevich RS cells) receives every `pP_w` through one shared excitatory
  weight `w_c` on DEPRESSING synapses (the engine's Tsodyks-Markram STP). The builder sets
  `enable_short_term_plasticity=True` only when `conflict_unit` is on, and marks every pre-existing sequencer pathway
  `stp_disabled=True` (`sim/regions.py:572`, the gap#5 mossy-detonator precedent), so the sequencer's own match and
  decision are unchanged (unit-tested, 2.10 step 1). A depressing synapse above its limiting frequency transmits a
  nearly rate-independent drive (Tsodyks and Markram 1997, PNAS 94:719; Abbott, Varela, Sen and Nelson 1997, Science
  275:220). So two blocks that drive the SAME patient line deliver about what one block delivers, and two DIFFERENT
  lines deliver about twice as much. `w_c` is chosen once, on design seed 7, by the rule in 2.10 step 1; one active
  line, from one block or from two, leaves `acc_conf` below rheobase.
- **STP reset at every read.** At the start of every conflict read, after `reset_sequencerK_state`, the read sets
  `cp_stp_x = 1` and `cp_stp_u = cfg.stp_U` (the bridge's own initial values, `sim/bridge.py:1288-1289`) on the
  `pP_w -> acc_conf` synapse indices. Reason: `reset_sequencerK_state` (`_phaseB_onebrain_sequencerK_derisk.py:179`)
  clears membranes, recovery variables, gates, coupling EMAs and conductances but never touches STP state, and its
  drain is 20 steps (20 ms at dt 1 ms) against a depression recovery time constant of 200 ms (`sim/config.py:904`;
  per-type E->E 200 ms, :1298). Without the reset, a read right after another would start depressed and its count
  would depend on query history. The reset is a host write of synaptic state between queries, declared in the header
  beside the existing per-query housekeeping.
- `acc_conf` inhibits nothing in this build; its spike count over the sequencer settle window (60 steps) is the
  conflict read. No host string compare deduplicates WITHIN the buffer: a patient's identity is the word line its
  decoded patient drives. A decode error that lights a different line for a duplicate reads as a conflict and abstains
  (the safe direction); it is recorded per hop. A clause or attributed patient (none in these fixtures) has no single
  patient word line; such a hop is recorded with its `patient_kind` and reported, never barred.

**Cue outside the reduced vocabulary.** `run_sequencerK_reduced_with_drive` assumes the cue words have lines (the
caller abstains otherwise, `_seq_block` :1709). The conflict read does the same: if the cue agent or action has no
line, the read does not run, `acc_conf_spikes` is recorded as None with `cue_in_reduced_vocab: false`, and
`slot1_drive` is 0. Such a cue has no stored buffer block, so the buffer's PROPOSE is None too; the hop can still be
LTM-sourced (below). This host abstain is declared.

**Fused branch.** M1 adds no conflict unit to the fused fabric. Its conflict read always uses `_ensure_sequencer`'s
bridge, also on a composer built with `integrated_loop="fused"` (a unit test pins that the flagged route builds that
bridge there). The chat server never builds a fused composer (2.1), so no arm runs one.

**What the route reads.** Exactly two things from the substrate: `acc_conf`'s spike count (through the relay) and the
workspace's `(conf, n_ignited, winner)`. The sequencer's own first-match decision, `blocks_matched` and
`patient_lines_active` are record fields that no branch of the route reads (checked in code review and by B7's
record). The cross-tier candidate list is a host input to the workspace drive (below), declared.

**Relay to the workspace (declared host residual).** The sequencer and the workspace are separate bridges, so the
count crosses by host relay: `slot1_drive = IGNITE_PA * min(1, n_conf / n_conf_ref)`, linear, no threshold.
`n_conf_ref` is the `acc_conf` count for the two-distinct-patient cue of the step-1 circuit fixture on design seed 7
(2.10 step 1), never a count from a bar table. The decision threshold is the workspace's ignition, read by the
unchanged `acc_conflict_gate` (2.2).

**LTM tier, and the cross-tier case.** `TieredFactStore.query_patient` asks the buffer first and the LTM only when the
buffer abstains (`tiered_fact_store.py:194-195` through `_tiered`; tier provenance in `query_patient_source` :197-216). The sequencer covers the buffer only. Two cases:
- **LTM-sourced hop** (the buffer abstained): the conflict read is `all_candidate_patients` on the routed shard's
  `RFPhasorComposer` (role-unbind of the stored composites; the shard exposes `_iter_facts`) into the existing
  `conflict_gate`. That enumeration is host closed-form and deduplicates by host string compare; it is recorded per
  hop as `conflict_read: rf_shard_unbind`.
- **Cross-tier case** (the buffer answers and the routed LTM shard holds a different patient under the same cue).
  Today's route abstains there, because `_distinct_patients` scans the buffer and the LTM
  (`compositional_chain_route.py:196-241`: the shard `shard_for(agent)` returns, which holds every fact about the agent
  under agent-hash routing, `sharded_phasor_store.py:166`, else every shard). A buffer-only read cannot see the LTM
  patient, so without more M1 would commit the buffer's patient where today's route abstains: a moat regression. M1
  therefore adds cross-tier slots: for a buffer-sourced hop with an LTM present, `all_candidate_patients(
  ltm.shard_for(x), x, rel)` (`webapp/gnw_deliberation.py:124`) lists the shard's patients, and each one whose string
  differs from the PROPOSE word gets its own workspace slot at `IGNITE_PA` (2.2). One buffer patient and one different
  LTM patient then put the gate in its CONFLICT regime, and it ABSTAINs. The enumeration and the string compare are
  host decisions, declared; M1's spiking claim covers the within-buffer read only (2.6).
- **Registered either way.** Every hop records `ltm_candidates`, `xtier_slots`, and the audit's distinct-patient
  counts split by tier: `audit_buffer` and `audit_ltm` from the 2.4 audit call (the buffer and the routed shard), and
  `audit_ltm_all`, a record-only scan of every shard in `ltm.shards` (`sharded_phasor_store.py:105-124`) in case a
  fact sits outside its routed shard. Two flags follow: `audit_disagree` (the buffer plus the larger LTM count holds
  2 or more distinct patients while `gate_decision` is ADVANCE) and, within it, `xtier_disagree` (the second patient
  is only in the LTM). Both flags are reported on every M1 arm, and B12 bars them.

**Risk named before data.** The integrated loop over-abstains at the tiny-demo's small vocabulary
(`webapp/server.py:3903-3909`); at V=320 it missed 6-14 of 35 recalls (finding
2026-09-05-rank2-integrated-loop-at-scale-answer-preservation-UNDEFINED-recall-incomplete). A match pool that misses
a true match would make `acc_conf` miss a conflict, which is the unsafe direction. Step 0 measures the match pools and
the patient lines on this fixture before anything is built, with STOP lines, and B12 (b) bars any ADVANCE the audit
contradicts.

### 2.4 `.kb` becomes an un-consulted audit oracle

Under the flag the route never calls `_distinct_patients` before it returns. After the decision, the route calls it
once per hop as an audit (`phase: audit`), and records what the list would have said beside what the spiking read
said, split by tier. The audit result is never read by any branch that follows.

The unit test that fixes this raises only on the ROUTE's reads. `query_patient` and `_ensure_sequencer` read `.kb`
themselves (2.1), so a test that raised on every access would fail on the composer's own reads and would test nothing
about the route. The test composer exposes `.kb` through a guard (a property plus a list proxy over `__len__`,
`__iter__` and `__getitem__`, slices included) that inspects the caller frame: it raises when the caller's module is
`research.runners.compositional_chain_route` AND the route's decision scope is open, and passes every other access
through (the composer's own frames in `one_brain_composer`). Assertions: (a) with the guard armed, the flagged route
returns the spiking result; (b) with the decision scope closed (the audit call), the same guard records the access
instead of raising; (c) replacing the audit's return value with garbage leaves the route's result identical.

### 2.5 Flags and lesions

- `BRAIN_CHAIN_SPIKING_HOPS` (default OFF): with it unset, `resolve_compositional_chain` is byte-identical (golden
  hash over a fixture battery in a unit test, plus the existing infer arms' primary fields).
- `BRAIN_CHAIN_SPIKING_HOPS_CONFLICT_LESION`: `w_c` = 0 (acc_conf hears nothing). Predicts: a within-buffer
  multi-valued hop commits the first match.
- `BRAIN_CHAIN_SPIKING_HOPS_WS_LESION`: the hops use the workspace built with assembly self-recurrence zeroed
  (`_get_bridge(seed, lesion=True)`, the C5 lesion's bridge). Predicts: no hop ignites, the conclusion vanishes.
- `qp_lesion` (the existing in-process patch, `query_patient` returns None): predicts the conclusion vanishes, the
  reversal of today's 7/7 invariance.

### 2.6 What stays host under M1 (declared in the header)

The regex dispatch, `_ROLE_NOUN_HINTS` and the same-verb fallback, the two-hop order, the default selection inside
`query_patient` (a host first-match over decoded rows), the reduced vocabularies read from the fact list
(`_ensure_sequencer` :1654) and the host abstain when a cue word has no line, the relay of each block's cleanup scores
into the sequencer as it is (host read and clip, `input_gain` current into the divnorm score bridge, the rheobase
deciding which pools fire, the host conversion to a fixed 1500 pA drive, the remap with a lit word outside the reduced
vocabulary dropped), the per-query STP reset, the slot-to-word map, the cross-bridge relay of the conflict count, the
mapping of the gate's outputs onto hop outcomes (2.2), the cross-tier candidate slots and the LTM-hop enumeration
`all_candidate_patients` with their host string compares (2.3), the derived framing (`frame_derived_answer`) and the
verify re-parse (`brain_chat_tui.py:1922-1942`). Under the conflict read the first-match answers only when the
matching rows drive a single patient line, so it has no patient choice left to make; the step-2 record checks per hop
that the PROPOSE word is the word of the single active patient line. M1 claims one thing: within the buffer, the
identity of each hop's patient and the abstain on ambiguity are decided by spiking reads of the bound store. Across
tiers the abstain is host-listed (2.3) and claimed as no more than that.

### 2.7 Engagement record, scoped to the hop reads

Instrumentation lives in the stage-2 runner's arm worker (behaviour-preserving wrappers, as the kb_scramble arms do).
A hop scope opens when the route enters `_hop_spiking` and closes when it returns. Per hop:
`{turn, hop, agent, relation, proposed (query_patient result), selected_block, source_tier, cue_in_reduced_vocab,
m_counts (per block), blocks_matched (m{b} over the composer's match_thresh, 0.06 at :132), patient_line_counts (per
pP_w line with any spike), patient_lines_active, spurious_P, stp_reset (cp_stp_x and cp_stp_u read back on the
conflict synapses at the read's first step), acc_conf_spikes, slot1_drive, ltm_candidates, xtier_slots,
xtier_dropped, drives, ws_seed, n_ignited, conf, winner, winner_kind, gate_decision, gate_cycles, committed,
audit_buffer, audit_ltm, audit_ltm_all, audit_disagree, xtier_disagree, route_kb_reads, composer_kb_reads}`. `route_kb_reads` counts
`.kb` accesses whose caller frame is `compositional_chain_route` while a hop scope is open; `composer_kb_reads` counts
those from `one_brain_composer`, listed by line: `_read_blocks` :1389, `_fact_shard_active` :1398,
`_ensure_sequencer` :1654, `_attributed_patient` :1782, the clause test in `query_patient` :1822, and `_seq_block`
:1700 on the integrated-loop arm. Turn-level counts (`_chain_facts`, `rich_answer_composer._stored_facts`,
elaboration) are reported separately and never barred.

### 2.8 Arms (stage-2 runner, `infer` group, all informational, never counted in the Stage-2 verdict)

Session tables: `infer` intact (the registered one) and four new tables, run under the same harness (numpy CPU,
tiny-demo +LTM, corpus present, `PYTHONHASHSEED=0`, `rich` omitted):
- `infer_amb1` (hop-1 ambiguity): heron hunts frog; frog eats gnat; heron hunts fish; then p_prey_eat, c_heron_hunt,
  c_dog_chase.
- `infer_amb2` (hop-2 ambiguity): heron hunts frog; frog eats gnat; frog eats apple; then the same probes.
- `infer_dup` (duplicate facts): the four intact teaches, then verbatim re-tells of the hop-1 fact ('the heron hunts
  the frog', label t_heron_hunt_re) and of the hop-2 fact ('the frog eats the gnat', t_frog_eat_re), then the
  `infer` probes.
- `infer_xtier` (cross-tier ambiguity): after the chat brain is built and before its first turn, the arm worker
  writes (heron, hunt, fish) into the LTM with `ltm.store` (the call `build_ltm_from_facts` uses, `tiered_fact_store.py:108`; a fixture step, not a turn, so
  the buffer never holds it); then the four intact teaches and the `infer` probes. Hop 1 of p_prey_eat then has frog
  in the buffer and fish in the LTM.

**Which `competent()` branch each new teach takes** (`comprehension_production_organ.py:719`: fully covered = verb in
VERB_SELECTS and both nouns in ANIMACY, hand or learned; fully OOV = verb in no verb table and both nouns in no animacy
table and not in `_brain_vocab(chat)`; anything else passes through unjudged and is stored). Hand-table facts below
are from `_phaseB_multicue_competition_spiking_derisk.py:54-75`; a learned-lexicon fact needs the corpus, which is
absent on the workstation, so step 0 (e) reads every branch back from the teach turn's `comprehension` record.

| teach | branch (expected) | consequence |
|---|---|---|
| 'the heron hunts the frog', 'the frog eats the gnat' (every table, and the `infer_dup` re-tells) | passed through: frog is in hand ANIMACY and hunt in no verb table; eat is in hand VERB_SELECTS and gnat in no animacy table (the permuted 'the heron hunts the gnat' was judged fully OOV, so neither heron, hunt nor gnat is covered) | stored unjudged |
| 'the heron hunts the fish' (`infer_amb1`) | passed through: fish is in hand ANIMACY, hunt in no verb table | stored unjudged |
| 'the frog eats the apple' (`infer_amb2`) | fully covered: frog and apple in hand ANIMACY, eat in hand VERB_SELECTS | JUDGED by the SEMANTIC margin; stored only if comprehended. An animate agent eating an inanimate patient is the canonical well-formed case, so comprehension is expected; step 0 reads it back and STOPs if refused |

The previous draft's 'the frog eats the fly' was claimed to pass because it "contains a known word". That was false:
with fly covered by the learned animacy lexicon it is fully covered and judged, and an animate patient under eat
(whose hand entry selects an inanimate patient) is the kind of item the margin can refuse, which would remove the
hop-2 ambiguity. 'apple' is a hand-table noun, so the branch does not depend on the learned lexicon.

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
| A15 xtier_off / A16 xtier_m1 | infer_xtier | off / M1 on | a cross-tier conflict still abstains, both routes |

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
- P8 (A12, reported only, never a bar): with t_heron_hunt acquired on that seed (`teach_acquired`; else the seed is
  UNDEFINED for P8), p_prey_eat is derived `[gnat, eat, frog]` from `[[heron, hunt, gnat], [gnat, eat, frog]]`
  through two recorded spiking hops, as the gate-off diagnostic arm derived it through the list.
- P9 (A15, A16): p_prey_eat abstains on both routes; in A16 hop 1 shows source_tier buffer, proposed frog,
  patient_lines_active = 1, acc_conf_spikes = 0, xtier_slots = [fish], n_ignited >= 2 and gate_decision ABSTAIN.

General engagement rule for every M1 arm: the arm's hop records must show `ws_seed` equal to the arm seed and, where
the bar reads a hop, a hop scope that opened and closed without an exception. A seed that fails this is UNDEFINED for
that bar, never PASS and never FAIL.

Bars (each on every dev seed 1-6; design seed 7 is excluded from the count):
- **B1 conclusion:** P1 holds. Engagement precondition: A1's p_prey_eat carries both hop records (B7), each with
  gate_decision ADVANCE; with no hop record the old `_hop` may have answered, so the seed is UNDEFINED for B1.
- **B2 ownership:** P2's two fields hold; the bar reads derived_svo and derived_from only. It is scored only against
  A3a and A3b run at the step-2 revision; until they exist B2 is UNDEFINED. A seed where A3a does not follow the list
  (the 10-26 result does not reproduce) is UNDEFINED for B2, and the non-reproduction is reported first. A3's
  p_prey_eat must carry both hop records (the flag engaged under the scramble), else UNDEFINED.
- **B3 PROPOSE necessary:** P3 holds and reproduces. Engagement precondition: hop 1 of p_prey_eat was entered and
  recorded `proposed` None with `drives[0]` = 0 (the lesion held inside the hop); a missing hop record or a route
  exception makes the seed UNDEFINED (no derived answer for another reason is not the lesion's effect).
- **B4 workspace necessary:** P4 holds and reproduces. Engagement precondition: hop 1 of p_prey_eat recorded
  `proposed` = frog, `drives[0]` = IGNITE_PA, the lesioned workspace bridge, and `n_ignited` = 0; otherwise UNDEFINED
  (an absent conclusion with PROPOSE empty or the route crashed says nothing about the workspace).
- **B5 multi-valued abstain:** P5 holds on both routes. Engagement preconditions: (i) B1's engagement holds on that
  seed (A1's hops ADVANCE: an M1 that never advances abstains on every cue, so its abstain on an ambiguous one says
  nothing); (ii) two decoded rows with two different decoded patients under the ambiguous cue (a REWRITE makes the
  seed UNDEFINED, never PASS); (iii) in A8 and A10 the abstain comes from the abstaining hop's `gate_decision`
  ABSTAIN, not from a miss or an exception. Otherwise UNDEFINED.
- **B6 conflict unit necessary:** P6 holds and reproduces. Engagement precondition, on the same seed and the same hop
  (hop 1 of p_prey_eat): A8 shows blocks_matched = 2, patient_lines_active = 2 and acc_conf_spikes > 0 (the unit
  sees the conflict when intact), and A11 and its .rep0 show blocks_matched = 2, patient_lines_active = 2,
  acc_conf_spikes == 0 and slot1_drive == 0 (the lesion held). Otherwise the seed is UNDEFINED for B6, never PASS.
- **B7 hop scope:** both hops recorded on p_prey_eat in A1; `route_kb_reads` = 0 inside every hop scope;
  `query_patient` calls inside hop scopes = hops attempted; the query_patient selected block lies in
  `blocks_matched`, and the PROPOSE word is the word of the single active patient line, on every resolved hop;
  `stp_reset` reads x = 1 and u = cfg.stp_U on every read.
- **B8 moat and controls:** c_prey_chase abstains; c_dog_chase recalls cat; c_heron_hunt and c_frog_eat primary
  fields equal A0.
- **B9 answer rate:** per label, flag-on (A1, A8, A10, A14, A16) against flag-off (A0, A7, A9, A13, A15) on
  p_prey_eat, c_heron_hunt, c_frog_eat, c_prey_chase, p_yesno, c_dog_chase: no label answered off and abstained on.
  Read by `tools/answer_rate.py --labels`, which the M1 build adds (section 7); FLIP-HURTS-ANSWERS is reported. A
  seed whose flag-on arms carry no hop record on p_prey_eat is UNMEASURED (equal answer rates from a flag that never
  engaged are not evidence of no harm).
- **B10 flag-off identity:** with the flag unset, A0's primary fields equal an `intact_a` arm run at the base
  revision (before any M1 code) on the same seed, and the unit golden holds.
- **B11 duplicate is not a conflict:** P7 holds. Engagement precondition: on that seed the re-tells appended, i.e.
  A14's hop records show blocks_matched = 2 on both hops with one decoded patient; otherwise the seed is UNDEFINED for
  B11. If every seed is UNDEFINED, B11 is UNDEFINED (never PASS), and the circuit-level duplicate test of step 1 is
  reported as the only evidence.
- **B12 cross tier:** (a) P9 holds. Engagement precondition: B1's engagement holds on that seed, and the arm's own
  read-back after the teaches shows the buffer's decoded rows under (heron, hunt) carrying frog only,
  `ltm.query_patient(heron, hunt)` returning fish, and the audit counting one patient per tier; otherwise UNDEFINED.
  (b) On every M1 arm except A3 and A11 (A1, A2, A4-A6, A8, A10, A12, A14, A16) and every dev seed, no hop has
  `audit_disagree` true. Every hop that does is listed with its tier counts, `xtier_disagree` and its seed; one such
  hop fails B12 on that seed. A3 is excluded because its `.kb` is deranged by design, so its audit is no oracle; A11
  and its .rep0 are excluded because the lesion blinds the unit by design (an ADVANCE on a two-patient cue there is
  B6's evidence, not a failure). Their `audit_disagree` hops are still listed.

Every lesion reports engaged / not engaged first (docs/TERMS.md, lesion); a lesion that did not hold is reported as
not held, never as no effect.

**Seed invariance, stated.** At the 10-26 gate the six gate seeds' `infer__intact_a` shards
(`research/findings/raw/_stage2_exit/gate/s{42,43,44,100,101,102}/infer__intact_a.json`) carry identical non-text
primary fields (abstained, recalled_svo, derived, derived_from, derived_svo, verified, source) on all ten labels; only
the answer text differs (checked for this revision). So on today's route six seeds are one effective measurement of
the outcome. Every M1 bar is reported as n_pass / n_defined AND n_effective, the number of distinct non-text outcome
vectors across the dev seeds, with the per-seed substrate counts beside it (m{b}, patient-line counts, acc_conf,
conf, n_ignited), since the sequencer and the workspace are seeded per arm and may vary where the outcome does not. A
bar that passes with n_effective = 1 is reported as "PASS, one effective measurement", never as six.

### 2.10 M1 build order, with STOP lines

**Step 0, the engagement smoke (read-only; no production edit; dev seed 7, then 1-6).** In the arm worker, after
the teach turns of `infer`, `infer_amb1`, `infer_amb2`, `infer_dup` and `infer_xtier`:
- (a) call `composer.query_patient` for (heron, hunt), (heron, eat), (frog, eat), (frog, chase) and record result,
  selected block and the decoded rows that match;
- (b) build the sequencer the production path builds (`_ensure_sequencer` at the composer's defaults, so the REDUCED
  fabric), and record every block's `m{b}` count for the same cues through `run_sequencerK_reduced_with_drive` with
  the composer's own drives, plus the sequencer decision against the host first-match, and the build's memory
  (`tools/mem_ok.sh` first; seed 7 measured before seeds 1-6);
- (c) for every matching block, read every main role's cleanup slice with the existing probe
  `_phaseC_task1_cleanup_probe.block_cleanup_scores_full`, and check that its agent and action slices equal
  `block_cleanup_scores` on the same block AFTER `np.maximum(., 0)` and over non-reserve slots only (the production
  read clips at 0, `_phaseB_onebrain_sequencer_derisk.py:87-88`, and installs no cleanup synapses on an uncommitted
  reserve slot, `_seq_cleanup_conns` :1634; the probe does neither); then pass the patient slice through the
  production relay (`onbridge_divnorm_drive` on the composer's score bridge at `input_gain` 1.0) and record which
  patient word pools fire, which of them are in V'_P, and which would be dropped as spurious;
- (d) record `chat._last_gnw_delib` on c_heron_hunt in `infer_amb1` (the single-hop conflict gate on onebrain);
- (e) read back each teach's `competent()` branch and decision from its `comprehension` record (2.8's table), the
  row count under each ambiguous and each re-told cue, the `infer_xtier` fixture (buffer rows under (heron, hunt),
  `ltm.query_patient(heron, hunt)`, `all_candidate_patients` on the routed shard, the audit's counts per tier), and
  `ws_seed` against the arm seed.
GO rule (written here, before the run): on 7/7 seeds, every true match drives its `m{b}` over the composer's
`match_thresh` and no non-matching block does; every matching block's patient slice fires exactly one patient pool
under the production relay, the stored patient's, and it is in V'_P; the two blocks of a re-told cue fire the same
pool and the two blocks of an ambiguous cue fire different pools; every teach of 2.8's table took its expected
branch and was stored; the `infer_xtier` fixture landed (frog in the buffer only, fish in the LTM only); `ws_seed`
equals the arm seed. STOP lines: a missed true match on any seed: post the counts; the circuit is banked and the named
next carrier is the cs-race engram race read for co-latching (two latches ON = conflict). A patient pool missing, wrong
or not unique: the patient identity the unit would count is not on the substrate at this fidelity. The probe's
clipped agent or action slice differing from `block_cleanup_scores` on a non-reserve slot of any block: the probe is
not reading what the build would read; post the difference. A new-table teach refused or taking another branch: the
table lacks its ambiguity there; post it, and Claude rules on the word. The `infer_xtier` fixture not landing (for
example a word outside the LTM codebook): post it, and Claude rules on the fixture words. If the re-tells did not
append on any seed, B11 is predicted UNDEFINED and the step-1 circuit test carries the duplicate case. Post
`[c2-hops-0] RESULT`, STOP for Claude.

**Step 1, the build (default OFF).** `conflict_unit` in both sequencer builders (V'_P patient lines, `gp{b}` gates,
depressing `pP_w -> acc_conf`, every pre-existing pathway `stp_disabled`, V'_P in the rebuild signature); the optional
patient role in `block_cleanup_scores`, `make_block_drives` and `run_sequencerK_reduced_with_drive`; the per-read STP
reset; `_hop_spiking` with the cross-tier slots, the audit call and the three flags in the route;
`tools/answer_rate.py --labels` (per-label reads over every turn of an arm file, with its own unit test).

`w_c` and `n_conf_ref` are set on design seed 7 by this rule, written before data, then frozen in code with the
seed-7 counts in a comment. The fixture is the step-1 circuit fixture, never a bar table: OneBrainComposers built
with the tiny-demo onebrain buffer's kwargs on seed 7, written by direct `composer.store` calls. Cue (heron, hunt)
over (heron, hunt, frog) and (heron, hunt, fish) is the two-line case; cue (frog, eat) over (frog, eat, gnat) stored
twice is the one-line-two-blocks case; cue (frog, eat) over (frog, eat, gnat) stored once (a separate composer) is the
one-line-one-block case. Sweep `w_c` over 64 log-spaced grid points from 1 to 3000 (the sequencer's own weights are
300-320, `_seq_vocab_shrink_derisk.py:99`, and depression scales the effective weight down). The window is the set of
grid points where `acc_conf` is silent (0 spikes) in both one-line cases AND fires at least 20 spikes (one per cell)
over the settle window in the two-line case. If the window touches an end of the grid, the grid is extended once by
a factor of 10 on that side and the rule re-applied. If the window then holds fewer than 3 contiguous grid points,
STOP: no weight window; post the counts. Otherwise `w_c` is the grid point nearest the window's geometric midpoint,
sqrt(lowest * highest), and `n_conf_ref` is the two-line count at that `w_c`. Reported beside it, at the chosen `w_c`:
the summed input current into `acc_conf` for one line from two blocks against one line from one block (must be within
1.1x) and for two lines against one line from one block (must be at least 1.5x); if either fails, STOP: the
deduplication is not on the substrate. The sweep runs twice, on a fresh bridge and right after a prior
two-line read, and the window must be the same grid points (the STP reset test).

Unit tests: flag-off golden; the caller-frame `.kb` test (2.4); the lesion flags reach their targets;
`conflict_unit=False` builds both sequencers byte-identically; with `conflict_unit=True` the m{b} rates and the
decision equal the `False` build on a fixture; a direct `composer.store` of the same triple twice leaves `acc_conf`
silent, and two distinct patients under one cue make it fire; the STP reset: the same two-patient cue read twice in a
row gives identical `acc_conf` counts, hop 1 then hop 2 gives hop 2 the same count as hop 2 read alone on a fresh
bridge, and `cp_stp_x` and `cp_stp_u` on the conflict synapses equal 1 and `cfg.stp_U` at the first step of every
read; a cross-tier fixture (buffer frog, LTM fish) gives a cross-tier slot and ABSTAIN; a `"fused"` composer under
the flag builds `_ensure_sequencer`'s bridge for the read. Module docstring corrected (it says the hops are list
scans today; under the flag they are spiking reads within the buffer).

Resource read (seed 7, under `tools/mem_ok.sh`, before any pool line; posted with the step-1 RESULT): the conflict
unit's cell and synapse counts read back from the built bridge against section 5's formula; build time and RSS of the
sequencer with the unit, of the two workspace bridges (`_get_bridge` intact and lesioned) and of `_get_theta`'s
calibration; the per-hop latency (drive rebuild, settle, workspace read); and the peak RSS of one `infer_amb1` arm
with the flag on against off. If the flag-on arm's peak exceeds 8 GB (the line memcap of 10 GB less 2 GB headroom),
STOP for a memcap ruling.

**Step 2, the dev read.** A0-A16 (with A3a and A3b) on dev seeds 1-6 through the pool at one pinned revision with
provenance on (numpy, memcap 10), scored against 2.9 by a scorer written before the runs. Post `[c2-hops-2] RESULT`,
STOP. A failed bar is a RESULT, never a retune.

**Step 3 (Claude).** A `flag: no` prereg carrying A1-A16 as informational arms of the 11-21 gate attempt, or a flip
packet if every bar holds; the default stays OFF until a flip prereg passes (smoke-first, flag-off battery arms on
the gate seeds, answer-rate).

## 3. M3: acquisition of well-formed all-novel transitives through a learned order cue (flag `BRAIN_COMPREHENSION_LEARNED_ORDER`, default OFF)

### 3.1 Why D4 refuses, and what it can read

`ComprehensionProductionOrgan.judge` (`comprehension_production_organ.py:738`) scores an assertion only when
`competent()` (:719) holds. `competent()` is the branch selection, and it is a host English table: an assertion is
fully covered when its verb is in VERB_SELECTS and both nouns in ANIMACY (the hand tables, or their corpus-learned
extensions, both default ON), and fully out-of-vocabulary when its verb is in no verb table and neither noun is in an
animacy table or in `_brain_vocab(chat)` (`webapp/server.py:3640`, the agents, actions and patients of the stored
facts); anything else is passed through unjudged. So the tables and the vocabulary test decide which assertions D4,
and M3 after it, ever judge. The read drives the SEMANTIC cues (animacy, verbfit) of `SpikingRoleCompetition`. For a
fully-OOV sentence both cues are absent, so the margin is 0.0 by construction and every such sentence is refused,
well-formed or not. The organ's learned position cue is not used by D4, and it would not help: its value is a host
vote by noun index (`_position_vote`, `_phaseB_multicue_competition_spiking_derisk.py:109`) that separates noun 0 from
noun 1 in ANY 3-token string, garbled or not. It is registered below as a null that the garbled bar must kill.

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
- **The read.** A heard assertion is comprehended when agent, action and patient each commit exactly once over the
  utterance and no token commits two roles, and its M3 margin clears the threshold of 3.3. The read uses no word
  list: which tokens are closed-class is never told to the brain. The margin arithmetic and its threshold are host,
  declared in the header. The instrument, not the brain, records which token committed which role; a role committed
  on a closed-class item fails that item in the battery (the frame packet's MM1 clause, applied by the scorer, which
  knows each item's structure).
- **Carrier (section 7's ruling text).** The traces must outlive the per-word reset. M3's carrier is the cs-race
  persistent word trace now in flight, feeding the word-role parser. Its state, from the f1h step-0 RESULT on main
  (`research/coordination/cs-race-f1h-0-RESULT.md`, accepted as scored by RULING f1h-0-1027): the held apical plateau
  latches, C6 8/8 on every grid cell; the older word's plateau never ended, C3 0/8 on every cell (0 of 216 pairs);
  STOP-CAL and STOP-SMOKE both fired, so no cell was chosen. RULING f1h-0-1027 sends the predeclared branch to f1i,
  the dendrite-targeting (SOM-like) inhibition that should end the older plateau
  (`research/coordination/cs-race-f1i-PACKET.md`, on main), with its base cell ruled at soma-g 40, W_IM 40 by a stated
  rule. M3 step 1 waits for the f1i result. The frame packet's NMDA trace stays parked under RULING focus-1025 and is
  not M3's carrier. The starvation that stopped the frame packet (below, 3.5) is the same family, which is why the
  learning-engagement precondition of 3.5 exists.

### 3.3 D4 integration (default OFF), and the M3 threshold rule

Under the flag, only the fully-OOV branch of `judge()` changes: it reads the M3 margin over the raw token sequence
instead of the SEMANTIC margin. The branch selection itself (`competent()`, 3.1) is unchanged and stays host. The
fully-covered branch, the out-of-competence pass-through and the known-fact honour rule are untouched (byte-identical
for every input that is not fully OOV).

**The M3 margin and threshold, written before data.** For an utterance u and role r, let s_win(u, r) and s_run(u, r)
be the commit-gate spike counts of the token that won role r and of the runner-up token for r, over the utterance.
d(u) = min over r of (s_win(u, r) - s_run(u, r)), and d(u) = 0 when any role commits zero times or more than once, or
a token commits two roles. The calibration set W_cal is every exposure statement re-heard once after training
(in-sample, well-formed only; no garbled, permuted or nonce-determiner item enters it). The normalised margin is
m(u) = d(u) / median{d(w) : w in W_cal}. The threshold is theta_M3 = 0.5 * P10{m(w) : w in W_cal}, the 10th
percentile taken by linear interpolation; u is comprehended iff m(u) >= theta_M3 (with the commit-count conditions of
3.2). theta_M3 is computed once at build, frozen, and recorded with W_cal's size and quantiles. If the median of d
over W_cal is 0 or P10 is 0 (more than a tenth of the in-sample well-formed items fail to commit cleanly), the build
STOPs: the organ has not acquired the frame, and no bar is scored.

The stored triple keeps the production store path; the M3 commit roles are recorded beside it, and a teach is stored
only when the two agree (a disagreement refuses; counted and reported). That agreement check is a host decision about
what the brain keeps, declared in the header.

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
  five tokens. 6 items per class per seed.
- **Frame-conforming exclusion, shared.** One function, `frame_conforming(classes, frames)`, decides whether an
  item's class sequence (determiner, noun, verb, as the item generator assigned them) equals the class sequence of
  any statement frame in the exposure set. Because the two nouns share a class, this excludes every conforming order
  under either noun assignment ("the blick gorps the dax" and "the dax gorps the blick" alike). The G generator
  rejects and redraws any conforming item (G5's shuffles above all), and the scorer calls the same function and
  marks any conforming G item INVALID instead of scoring it. A unit test pins both noun assignments and asserts
  G1-G4 are non-conforming.
- **Out-of-scope strings** (4 content tokens and the like): reported only. D4 never judges them today and they pass
  to the store path; M3 does not change that, and the report says so.
- **Order-permuted exposure:** the same statements heard verb-initial ("chases the dog the cat").
- **Nonce determiner:** the same statements with "ko" for "the".

**Outside D4's reach, stated.** "ko" is not in `extract_transitive`'s function-word list, so 'ko blick gorps ko dax'
has five content tokens and `extract_transitive` returns None: D4 never sees it, and under M3 such a sentence still
passes to the store path untouched at the reply level. The nonce-determiner items (OM4) and the order-permuted items
(OM3, OM5) are heard by the organ directly; they are organ properties, never a reply-level claim.

**Learning engagement (a precondition for every organ bar on a seed, and for the D4 build of 3.6).** The wrc-frame-1b
diagnosis found the frame packet's T->role projection updated on 1 of 5760 training steps against word->role's 3595,
because its pre-side trace population was almost silent (RULING batch-1026 (5), runbook on main); every bar read on
that organ tested a frozen projection. So, per organ and seed, the training record must show: O->role and RT->role
each updated on at least n_stmt training steps, where n_stmt is the number of exposure statements heard in training
(each statement offers each projection at least one co-firing), with sum |dw| > 0 on each; and at the training steps
where a role's commit gate fires, the pre-side trace that should predict it (O for agent, RT_agent for action,
RT_action for patient) has spiked within the rule's co-firing window on at least 90% of them. word->role is not the
reference here: it is frozen while O->role and RT->role learn (3.2), so its count in this phase is 0 by design; its
count from the frame packet's own training is reported as context only. The counts, sums and fractions are reported
for every organ arm. A seed failing this is UNDEFINED for every organ bar of that
organ, never PASS and never FAIL.

Arms: intact; role-sequence lesion (RT->role and O->role gates closed at read, persistence verified); carrier lesion
(traces not held across the reset); frame lesion (T->role closed); transplant (only the O->role and RT->role blocks of
the verb-initial organ copied into the SVO organ, same seed). The role-sequence lesion closes two projections that
together carry the learned order; the two halves (O->role only, RT->role only) are reported beside it, never barred,
so the record shows which half carries the effect. The carrier lesion removes every held trace (T, O and RT), so it
is not specific to the order cue; the role-sequence lesion is the specific one. Transplant engagement: the copied
blocks of the recipient equal the donor's weights after the copy, and every other block equals the recipient's own;
otherwise OM5 is UNDEFINED.

Bar-sanity generators, scored by the real scorer, n >= 1000: perfect; null-position-cue (the multicue organ's
position vote: accepts G); null-accept-all-OOV (gate off); null-refuse-all-OOV (today's D4); null-English-template (a
host "the N V the N" pattern: fails the order-permuted and nonce-determiner bars); null-bag-of-words; random. Every
perfect >= 0.95 and every null <= 0.05 on the composite; the record names the bar that kills each null.

### 3.6 Reply-level arms (stage-2 runner, informational; dev seeds 1-6)

Every reply-level M3 arm also needs 3.5's learning-engagement precondition on the organ D4 builds at that seed (read
from the build record); otherwise the seed is UNDEFINED for every R bar.
- **m3_permuted:** the frozen permuted `infer` table with M3 on. t_heron_hunt acquired (`teach_acquired`), and
  p_prey_eat derived with terminal frog. Engagement precondition: t_heron_hunt's `comprehension` record is non-null
  with `in_scope` true and carries the M3 margin (the M3 branch decided the teach); otherwise the seed is UNDEFINED
  for R1, since an acquisition through any other path is not M3's.
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
  forms against ANIMACY, VERB_SELECTS, the learned lexicons, the corpus vocabulary AND `_brain_vocab(chat)` read in
  the worker right after the four intact teaches, on every dev seed; a form found in any of them is redrawn by the
  fixed-seed generator before the table is pinned (a form in `_brain_vocab` would make its G teach partially known,
  so `competent()` would pass it through unjudged). Engagement precondition: on off_garbled, every G teach turn's
  `comprehension` record is non-null with `in_scope` true (D4 judged it in its fully-OOV branch); on m3_garbled, the
  same, carrying the M3 margin. Otherwise the seed is UNDEFINED. Prediction: every G teach refused under M3 and under
  the production default, and STORED with the gate off (the teeth: on a seed where the gate-off brain stores fewer
  than 2 of the 3, the table has no teeth and the seed is UNDEFINED for R2).
- **answer rate:** m3 against flag-off on every infer and ask probe label, `--labels` mode; FLIP-HURTS-ANSWERS
  reported.
- **R-goat (reported only):** `ask` with M3 on. Prediction: 'the goat eats the hay' and 'the crab eats the leaf'
  are still refused, because they go through the covered branch M3 does not touch. The named next method for them is
  the learned order read added as a third cue on the covered branch, decided after M3's organ verdict.

### 3.7 Predictions and bars

Organ (each per seed on dev seeds 1-15; at least 14 of 15 seeds must meet every bar; a seed UNDEFINED under 3.5's
learning-engagement precondition counts toward neither side and is reported):
- **OM1:** W comprehended >= 11/12.
- **OM2:** every G class refused >= 5/6, pooled G refused >= 0.95; G4 included (prediction: refused, because no
  determiner trace supports an argument commit). Scored on a seed only where OM1 holds on that seed (refusing G means
  nothing in an organ that comprehends nothing); otherwise UNDEFINED.
- **OM3 order learned:** the verb-initial organ comprehends verb-initial W >= 11/12 and refuses SVO W >= 11/12.
- **OM4 no English frame:** the "ko" organ comprehends "ko" W >= 11/12 and refuses "the" W >= 11/12.
- **OM5 transplant:** the SVO organ with the verb-initial blocks follows the donor (OM3 thresholds).
- **OM6 necessity:** the role-sequence lesion and the carrier lesion each drop W to <= 2/12. Scored only where OM1
  holds on that seed; otherwise UNDEFINED.
- **OM7 identity:** with `role_seq=None` the frame-packet pins hold; the D4 decision is unchanged on every
  non-fully-OOV item of the committed D4 battery.

Reply (dev seeds 1-6, every seed): R1 m3_permuted acquires t_heron_hunt and p_prey_eat ends on frog (with the 3.6
precondition); R2 every G teach in m3_garbled is refused and no later probe recalls it (with the 3.6 precondition and
teeth), scored on a seed only where R1 holds on that seed, else UNDEFINED; R3 m3_intact equals flag-off on primary
fields; R4 no FLIP-HURTS-ANSWERS; R5 flag unset is byte-identical; R6 m3_rs_lesion refuses t_heron_hunt and derives
nothing on p_prey_eat, on every seed and its .rep0 (with the 3.6 precondition), scored only where R1 holds.

### 3.8 M3 build order, dependencies, STOP lines

- **Step 0 (read-only, now, parallel to M1 step 0).** (a) The D4 decision (scope, competence branch, margin) on
  every W, G and out-of-scope item at production defaults; (b) the word-role parser's commit read on W and G with
  today's organ (prediction: nothing commits, since position alone is subthreshold by design); (c) the carrier status
  read from the cs-race f1h step-0 RESULT and RULING f1h-0-1027 and, when posted, the f1i RESULT; (d) the nine
  `infer_garbled` nonce forms checked as in 3.6, `_brain_vocab(chat)` after the intact teaches included; (e) the
  `frame_conforming` exclusion run over the exposure set's frames and a G5 draw, with the excluded count. Post
  `[c2-acq-0] RESULT`, STOP.
- **Step 1 (organ build, default OFF)** starts only when the cs-race carrier (f1h plus f1i) holds a trace across the
  per-word reset and ends the older one. The build's first record is 3.5's learning-engagement read on seed 7; if it
  fails, STOP with the counts (no battery runs on a starved projection).
- **Step 2:** the organ battery and bar sanity on dev seeds; post RESULT, STOP.
- **Step 3:** D4 integration and the reply-level dev read; post RESULT, STOP. Then Claude writes the preregs. Step 3
  is the third build step under this header and waits for the cap ruling (cap accounting, top).

Dependency, stated plainly: the carrier is open. Section 7's ruling text names the cs-race persistent word trace as
M3's carrier; M3 step 1 waits for its results, and M1 proceeds independently. M3 is not expected to land before the
11-21 attempt. No-defer: M3 continues after 11-21 if it misses.

## 4. What neither method does

- **The 11-21 C2 verdict.** On the frozen permuted arm, M1 alone leaves the D4 refusal and 0/6 in place. The
  instrument side is the ask (b) of section 7: `score_infer` gets an acquisition precondition like ask's A4, declared
  as a re-pin amendment before any re-attempt data, so a refused permuted teach makes the permuted arm UNDEFINED.
  UNDEFINED is not a pass: C2 still needs the brain to acquire well-formed novel transitives, which is M3's job. The
  alternative of a new prereg with an acquirable permuted exposure is rejected (section 7).
- **The derived construct (V4).** The conclusion still asserts premise 2 verbatim (`conclusion_is_hop` True). Whether
  that is the inference Stage 2 intends is a separate question.
- **The void `transitive_chat` arm** (its flag is already ON and the route sits only on the single-fact path) and
  **the direct-recall framing change under kb_scramble** (a list-backed verify over `chat.stored_facts`) are named,
  not addressed.
- **Cross-tier conflict on the substrate.** M1 keeps today's cross-tier abstain through host-listed slots (2.3). Those
  slots and the LTM-hop enumeration are retired by the cs-race spiking store competition once it spans the LTM tier
  (the `replaced-by` line names that competition for block selection); no build here does that.

## 5. Seeds, resources, where it runs

- Design seed 7 (M1 `w_c`, `n_conf_ref`, the resource read; M3 build engagement); M1 dev read seeds 1-6; M3 organ
  dev seeds 1-15, reply dev seeds 1-6. The gate seeds are run only through a later prereg.
- **Conflict unit size, before measurement.** With K stored buffer blocks and V'_P distinct stored patients, the unit
  adds (K + 1) * V'_P * 20 + 20 cells (per-block patient lines, shared patient lines, `acc_conf`) and
  K * V'_P * 400 + V'_P * 400 synapses (`d{b}P_w -> pP_w` and `pP_w -> acc_conf` at density 1.0, 20 x 20 per pair).
  For example K = 12 and V'_P = 10 give 2,620 cells and 52,000 synapses. A patient layer over all V words would be
  V/V'_P times larger, which is why the layer uses the reduced patient vocabulary (stored patients only, 2.3). The
  reduced sequencer is built in every M1 arm even with `integrated_loop` off, so its own size is part of the cost.
- **Measured at step 1 under `tools/mem_ok.sh`** (2.10): the counts read back from the built bridge, the sequencer
  and workspace build memory and time (`_get_bridge` intact and lesioned, `_get_theta`), the per-hop latency, and the
  flag-on against flag-off peak RSS of one arm; a flag-on peak over 8 GB STOPs for a memcap ruling.
- Stage-2 arms: about 6.0-6.5 GB peak each today, `tools/memcap.sh 10`, numpy CPU, pool lines. Step 0 sequencer
  builds: memory measured on seed 7 first under `tools/mem_ok.sh`. Organ probes: under 0.1 GB (wrc step 0 peaked 85.3
  MiB).
- No GPU. No AWS.

## 6. Order and who does what

1. Claude posts the keyed runbook RULING for section 7's asks.
2. M1 step 0 and M3 step 0, in parallel (both read-only, CPU). STOP for Claude.
3. M1 steps 1-2 (the ownership lane; unblocked). STOP for Claude.
4. M3 steps 1-2 after the cs-race carrier results (f1i on top of f1h). M3 step 3 after the cap ruling.
5. Independently of both: the `score_infer` acquisition-precondition amendment, before any 11-21 re-attempt data.
The driver runs the steps and posts the RESULT lines; Claude rules at every STOP, writes the preregs, reviews and
merges.

## 7. Claude's 2026-10-27 ruling text, and the keyed runbook RULING that tracks it

**Authority.** The text below is a copy for the reader, not the tracking mechanism. The authoritative record is the
keyed section-6 RULING that Claude posts in `research/coordination/LOCAL_LLM_RUNBOOK.md` on main. Until it exists,
nothing in this section is in force for the driver; where the two differ, the runbook RULING wins. When the driver
posts this revision it relays these asks under one line, `[c2-hops] NEEDS-CLAUDE`:
- (a) `[c2-hops-cap]`: +1 build for this scaffold group, so that M3 step 3 (the D4 integration) can run, granted in
  the gate's form once the M3 prereg file exists (cap accounting, top).
- (b) The `score_infer` acquisition-precondition re-pin amendment to
  `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md`, declared before any 11-21 re-attempt data.
- (c) Confirmation, by key, of the ruling text below (C2 instrument, the rejected option, M3's carrier, answer rate),
  so this packet can cite the key in place of this section.

**Ruling text as given (Claude, 2026-10-27; not yet keyed):**
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
- Not ruled here: the cap (ask (a)).
