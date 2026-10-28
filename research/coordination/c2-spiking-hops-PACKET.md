# PACKET c2-spiking-hops (Claude, 2026-10-27; revised twice that day after two skeptics, again on 2026-10-28 after a third, round 4, the same day after the round-4 synthesis of three more skeptics, round 5, after the round-5 synthesis, round 6, after the round-6 fix-checker, round 7, after a fresh skeptic, round 8, and after a verifier, round 9): Stage-2 C2, two separate methods: hop reads on the spiking store (M1) and acquisition of all-novel transitives through a learned order cue (M3)

host-decision: scaffold
decides: M1: the possessive-question dispatch (_POSSESSIVE_CHAIN_RE), the hop-1 relation candidates (_ROLE_NOUN_HINTS plus the same-verb fallback), the two-hop relation order taken from that parse, the PROPOSE block selection inside query_patient at the production default, which is the fact-shard fast path (default ON: per-role DG sparse indices that _ensure_fact_shard builds from the .kb agent, action and patient strings, one_brain_composer.py:1400-1432 with the entry read at :1420, intersected by _fact_shard_candidates :1434-1466, an empty candidate set being an abstain before any block is decoded, then a host first-match over the spiking decodes of the candidate blocks, :1468-1481), and on escalation or with the fact shard off a host first-match over every spiking-decoded row (_seq_block :1685-1690), the label-only .kb reads on the PROPOSE path (the patient-type clause test at :1822 in _finish_query_patient and the attribute-key test at :1782), the sequencer's reduced agent and action vocabularies built from the host fact list (_ensure_sequencer's self.kb[:K] read, :1654, reading the agent and action fields only from step 1 on, which decides which stored agent and action words get word lines) and the host abstain of the read when a cue word has no line (:1709), the block count K that the per-block reads iterate over (len(self.kb)), read for the conflict read at one new site keyed as kind `len` (section 2.4), the drive rebuild after every store write while the conflict unit is built (section 2.3), the set union, in block order, of the patient word pools that the relay below lights across all K blocks, which forms the patient vocabulary V'_P without reading any .kb entry, and the fabric rebuild when that set changes (section 2.3), the relay of each block's cleanup scores into the sequencer (block_cleanup_scores read to host and clipped at 0, injected as input_gain x score into the divnorm score bridge by make_block_drives and onbridge_divnorm_drive, where the on-bridge divisive normalisation and the Izhikevich rheobase decide which word pools fire, then turned by the host into a fixed 1500 pA decoded-line drive and remapped to the reduced vocabularies with a lit word outside them dropped), the per-query STP reset of the conflict synapses, the slot-to-word map of the workspace, the relay of the conflict unit's spike count into a workspace slot drive, the drive vector the host builds for every hop whichever tier serves it, the host read-outs of the workspace (_conf_from_nmda's NMDA margin formula, its ignition count over the fixed threshold THR and its argmax winner, _gnw_reentrant_metacog_gated_deliberation_derisk.py:204-219 with THR at :105) and the gate's calibrated theta_hi and retry budget R_max (acc_conflict_gate :326-337, R_MAX_DEFAULT at :124, theta_hi from _get_theta), the mapping of acc_conflict_gate's four outputs onto hop outcomes (section 2.2), and, for the LTM tier (the TieredFactStore's tier routing picks the source), the candidate enumeration all_candidate_patients on the routed shard (host closed-form unbind with a host string dedup), whose patients other than the PROPOSE word each get their own workspace slot, on a buffer-served hop (the cross-tier slots) and on an LTM-served hop alike; M3: the competent() branch selection of ComprehensionProductionOrgan, which is a host English table (the hand ANIMACY and VERB_SELECTS coverage with their corpus-learned extensions, plus the _brain_vocab membership test, decide whether an assertion is fully covered, fully out-of-vocabulary or passed through, and so decide which assertions M3 ever judges), which utterances the comprehension read judges (extract_transitive's 3-content-token test and its function-word list), the role assignment of the stored triple (the positional parse after the host strip), the store-or-refuse agreement check between M3's commit roles and that positional parse, the M3 margin arithmetic (minimum over roles of winner-minus-runner-up, median-normalised) and its calibrated threshold (0.5 times the 10th percentile of the in-sample well-formed normalised margins), and the utterance-onset event
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement)
replaced-by: the learned relation-noun binding (M2, its own packet), the cs-race spiking store competition for block selection, the word-role frame commit for scope and stored roles, and one-bridge co-residence of the K-way sequencer and the GNW workspace
cap: 2 builds

**Round-9 changes (2026-10-28, after a verifier returned five fixes on 33a6ac58c, none blocking: two minor text
fixes, one bar that could never hold (R4, the class of round 8's item 3), one provisioning gap and one forward tool
gap; design only, no run and no code; all five applied). Every citation below was read at origin/main c394b764a
(`git show`). Every file round 8 cites is blob-identical there to 34a4e5f75 (an empty `git diff` over them), so round
8's line numbers hold; `tools/pool_provision.sh`, `tools/gpu_remote.sh` and `tools/pool_queue.sh` are cited at
c394b764a. No bar is loosened and no gate seed is read.** Item number, then the sections changed:
1. (minor; stale text that contradicted round 8's item 2) 2.7's "Teach log and store read-back" paragraph still said
   that the bound store is read only in the hop records and the overwrite fixtures, and that the substrate read-back
   of every premise (decoded buffer rows, `ltm.query_patient`) runs in step 0 only. Round 8's per-turn
   `store_snapshot` (2.7, 2.9) is a decoded-row substrate read on every non-teach turn of every M1 arm. The paragraph
   now names the snapshot as the M1 arms' per-probe-turn bound-store read, keeps the hop records' and the fixtures'
   own reads, says what a flag-off arm reads instead, and names step 0 (f) as a separate step-0-only read of the whole
   premise map with step 0 (h)'s inert check beside it. Sections: 2.7.
2. (minor; scope) 2.9's "A probe turn with no `store_snapshot` ... is UNDEFINED for every bar that reads that probe"
   did not say "on an M1 arm". The flag-off arms record no snapshot by design (2.7, 2.9), so a scorer reading the
   sentence literally would make every flag-off B9 cell, and B8 (controls) on A0, UNDEFINED. The sentence now names M1
   arms, and a flag-off arm's probes are scored on facts-present sources (1)-(3) only. The term is pinned once, in
   2.8: an **M1 arm** runs with `BRAIN_CHAIN_SPIKING_HOPS` on, a **flag-off arm** runs without it, and "every arm of
   2.8" names both. The packet had also used "M1 arm" for rules that bind flag-off shards too (2.9's general rule,
   whose (i) B9's flag-off side needs; round 5's item 9; section 5's bundle line). Those scopes are now written out:
   rules (i) and (iv) bind every arm of 2.8 that a bar reads, and (ii), (iii) and (v) bind M1 arms, the only arms
   that make hop records. Round 5's item 9 is history and is not rewritten; its "every M1 arm" meant every arm of
   2.8. Sections: 2.8 (the term), 2.9 (general rule, facts present), section 5.
3. (a bar that could never hold; round 8 item 3's class, on the M3 side) 3.6's answer-rate bullet and R4 read "m3
   against flag-off on every infer and ask probe label" and named no arm pairs and no per-pair label lists.
   `infer_garbled` carries none of the six `_INFER_PROBES` labels (3.6), and the ask labels exist only on R-goat's
   arm, so under rule (vii) R4 was UNDEFINED on every seed. Fix: R4's (pair, label) cells are written into the scorer
   as data, as B9's are. The pairs are m3_intact against `infer` `intact_a` and m3_permuted against `infer` `permuted`,
   each on the six `_INFER_PROBES` labels, and R-goat's arm, now named `m3_ask`, against `ask` `intact_a` on the
   fifteen non-teach `ask` labels (`stage2_exit_gate.py:163-170` and :173-197 at origin/main). m3_garbled with
   off_garbled, m3_rs_lesion and gateoff_garbled are not R4 pairs (reasons in 3.6). The same `turns_for` equality
   unit test as B9's, asserted on both arms of each pair, lands in M3 step 3 with the arms, and `m3_ask` joins the
   test allowlist as `ask`'s one extra. R3's "flag-off arm" is pinned to the same `infer` `intact_a`. R4's
   no-engagement UNMEASURED clause stays per seed (reason in 3.7). Sections: 2.8 (test amendment), 3.6, 3.7 (R3, R4),
   3.8 step 3.
4. (provisioning) The step-1 `--labels` unit test read its shard from `research/findings/raw`, which pool-provisioned
   revisions exclude (`tools/pool_provision.sh:91-92`, `':(exclude)research/findings/raw'`) and which
   `tools/gpu_remote.sh`'s sparse worktrees exclude too (:47). Fix: the shard is vendored under `tests/fixtures/`
   (path in 2.10 step 1); the test checks the copy's git blob id against the committed shard's, and a missing or
   different copy FAILS the test, never skips it. The step-1 unit tests are named to run in the build's own full
   worktree under `tools/memcap.sh 6 --`, and the vendored copy keeps the test valid on a provisioned or sparse
   checkout as well. Sections: 2.10 step 1.
5. (forward; non-blocking) `--labels` was pinned to dev seeds 1-6, and the prefixed and paired modes report every
   stage-2 shard UNMEASURED (round 8, item 4), so step 3's later flip prereg (answer rate on the gate seeds, 2.10
   step 3) had no working tool for stage-2 shards. Fix: `--labels` takes an explicit `--seeds`, default 1-6 and never
   `SEEDS`; a gate seed in the list is refused (exit 2) unless `--prereg` names a file that exists at the read
   revision and whose name contains "prereg", the pool guard's name check (`tools/pool_queue.sh:248-261`). Sections:
   2.9 (B9), 2.10 steps 1 and 3, section 7 (packet note).

**Round-8 changes (history; 2026-10-28, after a fresh skeptic returned four fixes on fc476672a, two of them blocking; design
only, no run and no code; all four applied as given, with one citation corrected in item 4; where round 9 changes a point, round 9 holds). Every citation below was
read at origin/main 34a4e5f75 (`git show`), where every file it cites (`webapp/server.py`,
`research/runners/shared_ltm.py`, the route, the composer, the stage-2 runner, the tiered and sharded stores,
`tools/answer_rate.py` and `tests/test_stage2_kb_scramble.py`) is blob-identical to the branch base cadf0f40f, so the
line numbers hold on both. No bar is loosened and no gate seed is read.** Item number, then the sections changed:
1. (blocking) The `infer_xtier` fixture could never land at the default. Round 5 wrote (heron, hunt, fish) into
   `chat.inner.composer.ltm` with `ltm.store`, on the claim that each conversation loads its own LTM copy
   (`webapp/server.py:4290-4300`, :4395). That claim is false at origin/main. `_SHARED_LTM_DEFAULT_ON = True` (:4203),
   so `_ltm_store_for_conversation` returns `_shared_ltm.view_for(...)` (:4307-4316), which the tiny-demo attach calls
   at :4396 (the comment at :4395, "default: this conversation's own copy", is stale). So `.ltm` is a `SharedLTMView`,
   and its `store()` raises `SharedLTMWriteError` (`research/runners/shared_ltm.py:215-216`). Fix: A15, A16 and the
   step-0 `infer_xtier` read run with `BRAIN_SHARED_LTM=0` and `BRAIN_LTM_RNG_ISOLATE=1` in both entries' env, the
   `offiso` configuration. With the shared store off and isolation on, the conversation gets a private,
   RNG-isolated copy (`_load_ltm_isolated`, :4280-4290, returned at :4306), so `.ltm` is a `ShardedPhasorStore`. The
   shared-LTM flip battery measured offiso equal to the shared default on the whole record, on 6/6 gate seeds (A1,
   comment at :4198-4199). The fixture's engagement record holds `type(chat.inner.composer.ltm).__name__`, which must
   be `ShardedPhasorStore`. A `SharedLTMView` on A15 or A16 (or any other type, or a write that raised) makes that arm
   UNDEFINED for every prediction and bar that reads it. No write goes through `ltm.shared_store`. `BRAIN_ONE_BRAIN`,
   which forces the shared store on (:4223-4224), stays unset (default OFF, `webapp/one_brain.py:49`). The false
   citation is corrected in 2.8 and in 0g item 7, and the step-0 STOP remedy for the fixture is replaced. Sections:
   0g item 7, 2.8 (the `infer_xtier` paragraph, the arm table), 2.9 (P9, facts-present source (3), B9's A16/A15
   pair through 2.8, B12 (a)), 2.10 step 0 ((5), (e), the GO rule, the STOP lines) and step 1 (the env and its unit
   test).
2. (blocking for the scorer) The facts-present rule (2.9) took each map cue's bound-store side from the hop records,
   but many cues are never hopped. A non-chain probe returns before any hop (`compositional_chain_route.py:274-276`),
   a hop-1 ABSTAIN ends the chain (:289-291), and a hop-1 miss moves on to the next relation without a hop 2
   (:292-293). So B3, B4, B5, B6, B13 (a), B8 (2)-(4) and B9's control cells would be UNDEFINED on every seed. Fix:
   on every probe turn of every M1 arm (every turn whose role is not `teach`), chain or not, the arm worker records the
   FULL decoded store once after the reply, `store_snapshot`. It is the same `_read_blocks` call that 2.7's
   post-decision read already makes, which decodes every block (the per-block loop, `one_brain_composer.py:1389`, at
   the production layout; `_read_all_blocks`, `len` at :1206, on A21). All K rows are kept, each with its block id.
   The snapshot also holds the record-only LTM scan of every shard (the `audit_ltm_all` scan) under each cue of the
   arm's premise-map rows. Each map cue's bound-store side is read from that turn's snapshot, and a probe turn without
   a snapshot is UNDEFINED for every bar that reads it. The snapshot is outside every decision scope and is covered by
   the step-1 whole-table behaviour-preservation test. Sections: 2.7, 2.9 (facts present), 2.10 step 1.
3. (a bar that could never hold) B9's cells crossed all five arm pairs with six labels, but `infer_amb1` and
   `infer_amb2` carry only p_prey_eat, c_heron_hunt and c_dog_chase of the six `_INFER_PROBES`
   (`stage2_exit_gate.py:163-170`; their tables in 2.8). Three cells of each of those two pairs therefore had no
   response on either arm, so B9 was UNDEFINED on every seed. Fix: B9's cells are the labels each pair's own table
   carries, written into the scorer as data, as B12 (b)'s arm list is. For A1/A0, A14/A13 and A16/A15 that is the six
   `_INFER_PROBES` labels, and for A8/A7 and A10/A9 it is p_prey_eat, c_heron_hunt and c_dog_chase. Sections: 2.9
   (B9), 2.10 step 1.
4. (minor) `tools/answer_rate.py`'s `probe_body` (:85-95) reads only top-level dicts that carry `answer`. A stage-2
   shard nests its replies under `responses`, so the prefixed and paired modes report UNMEASURED on every committed
   `research/findings/raw/_stage2_diag/c2_comprehension_gateoff/s{1,2,3}/infer__*.json` (checked: `probe_body`
   returns None on all six files). The step-1 `--labels` unit test reads `responses[label]` from a committed stage-2
   shard (`s1/infer__intact_a_gateoff.json`) and takes dev seeds 1-6, never the gate-seed default. Citation corrected
   from the fix as given: `SEEDS` is at `tools/answer_rate.py:52`, not :55 (a comment line at origin/main); it is the
   default of `paired_per_seed` (:162) and of `paired --seeds` (:224). Sections: 2.9 (B9), 2.10 step 1, section 7.

**Round-7 changes (history; 2026-10-28, after the round-6 fix-checker returned two minor fixes on 1c84c367c; design
only, no run and no code; both applied; where round 8 changes a point, round 8 holds). Each was read at origin/main
53d64c2ef, where every file cited below is blob-identical to the f0fd444f7 of 0h.**
1. The named test amendment of round 6 (item 12) left out A17-A20. They build as a new mode, `kb_overwrite`, of
   `apply_kb_scramble` (2.8), and the worker enters a scramble mode only through `arm_scramble` reading the informational
   entry's `scramble` key (`stage2_exit_gate.py:364-369`; the worker reads it at :686-693). So A17-A20 carry a scramble,
   and `tests/test_stage2_kb_scramble.py:65-68` (no other arm carries a scramble) breaks for them as it does for A3. The
   allowlist now names the arms carrying a scramble as the two kb arms (their modes `kb` and `kb_spiking` unchanged), A3
   (mode `kb`) and A17-A20 (mode `kb_overwrite`), each with its mode asserted. Sections: 2.8 (the fixtures paragraph and
   the test amendment), 2.10 step 0 (7), 0h item 12.
2. Citation. 2.3 and 0h item 2 cited `_phaseB_onebrain_sequencerK_derisk.py:127-147` for the `RegionPathway` list. At
   origin/main the pathway constructors run from :127 to :155 (the last, `inh{b}` -> `abstain`, spans :154-155) and
   `cfg.region_pathways = P` is set at :156. Corrected to :127-155 (assignment at :156). The same convention was applied to
   the neighbouring `_seq_vocab_shrink_derisk.py` cite, whose last constructor spans :164-165 and whose assignment is at
   :166: :146-164 became :146-165. Sections: 2.3, 0h item 2.

**Round-6 changes (history; 2026-10-28, after the round-5 synthesis ruled 3059090939 SURVIVES-WITH-FIXES; all fourteen
items applied, six blocking and eight not; the item-by-item table, with the code at origin/main f0fd444f7 that each
citation was checked against, is 0h; where round 7 changes a point, round 7 holds).** Item number, then the sections
changed:
1. (blocking) Patient-field counting had no insertion rule and no entry-level control: entries arrive as plain dicts
   (the composer's appends, the `kb_overwrite` fixture, `update_on_mismatch`). The proxy now wraps the fact dict on
   `append`, `insert`, `extend`, `+=` and `__setitem__` (slices included), supports `clear` and `pop`, and a class
   property wraps any reassignment of `.kb`. One unit test per insertion path. Entry-level positive control: every hop
   whose buffer PROPOSE selected a block must list the clause-test patient read of that block id, else B7 (c)'s patient
   clause, `vp_kb_entry_reads` and B13 (b) are UNDEFINED. Sections: 2.4, 2.7, 2.9 (B7, B13, hop-record table), 2.10
   step 1.
2. (blocking) B6 could never hold: `RegionPathway` wiring floors every weight at 0.01, so the conflict lesion's `w_c` =
   0 left 0.01 and the read-back `cp_w_max` = 0 was unreachable. The lesion now writes the built `pP_w -> acc_conf`
   weights to exactly 0.0 after every build (explicit wiring, as the workspace lesion uses, is the named alternative).
   A step-1 unit test reads `cp_w_max` == 0.0 with `cp_n_syn` equal to the intact build's and > 0. Sections: 2.3, 2.5,
   2.9 (B6), 2.10 step 1.
3. (blocking) B7 (c) could never pass. The conflict read must read K at a new site, which the line-keyed allowlist did
   not list, and step 1 itself moves the listed lines. Each allowed site is now keyed by (function, statement, kind).
   The new K site is allowed as kind `len` only, the map is re-pinned at the step-2 revision before any data, and a
   unit test maps each key to its current statement. Sections: 2.3, 2.4, 2.7, 2.9 (B7), 2.10 step 1.
4. (blocking) B3's patch-call precondition was unmeasurable: the patch keeps one cumulative counter per session. The
   worker snapshots the count per turn and per hop. Precondition restated: the per-turn delta is >= 1 and the hop-1
   record's `propose_path` is `patched`. Sections: 2.7, 2.9 (B3, hop-record table), 2.10 step 1.
5. (blocking) B8 (1b) passed with dead match pools: on A21 PROPOSE is the host first-match over the decoded rows,
   independent of the match pools, and LIVE-A8 was measured on the fact-shard layout. (1b) now needs, on the same A21
   seed, p_prey_eat's hop-1 record under (heron, hunt) with `blocks_matched` = 1 containing the selected block and
   `patient_lines_active` = 1, else UNDEFINED. B7's "selected block lies in `blocks_matched`" applies to every
   resolved hop of every M1 arm. Sections: 2.7, 2.9 (B7, B8, hop-record table).
6. (blocking) An UNDEFINED item let a bar pass on the rest (A4 (a) excludes undefined items and scores the rest). One
   global rule, (vii): any UNDEFINED item makes the bar UNDEFINED on that seed; A4 (a)'s exclude-and-score is not used;
   B9, B8 (2)-(4) and B12 (b) are named. Sections: 2.9 (general rule, facts present, hop-record preamble, B8, B9, B12),
   3.7.
7. B1's precondition contained its outcome. The precondition is now an engaged hop read (the B1 row and the positive
   control); ADVANCE on both hops is the outcome, and a hop-1 or hop-2 ABSTAIN or COMMIT is a FAIL. REF-A1 stays the
   reference for B3, B4, B5, B12 and B13. One part was not applied as stated: the read-count precondition is
   `spiking_hop_reads` >= 1, not >= 2, because a hop-1 ABSTAIN ends the chain after one read, so >= 2 would turn the
   FAIL this item requires into an UNDEFINED (evidence in 0h). Sections: 2.9 (REF-A1, B1, B5, B8, B12, hop-record table).
8. The abstain source was unrecorded: `acc_conflict_gate` also ABSTAINs when one slot ignites below `theta_hi`. Every
   hop records `abstain_regime` (ignition or graded, with `conf` and `theta_hi`). B5 and B13 (a) need `n_ignited` >= 2
   at R_max and `xtier_slots` == [] on the abstaining hop, else that item is UNDEFINED. Sections: 2.2, 2.7, 2.9 (P5,
   P10, B5, B13).
9. An in-place rewrite left the drives and V'_P stale: the drives are dirtied only under the integrated loop, and D2
   rewrites at unchanged K. Step 1 dirties the drives on every store write while the conflict unit is built
   (byte-identical when it is off), with a unit test. Each hop records the turn and K at which its V'_P was formed, and
   B13 (a) requires V'_P formed after the A17 overwrite. Sections: 2.3, 2.7, 2.9 (B13), 2.10 step 1.
10. B13 (a) tested `.kb` only: `chat.stored_facts` and its sets kept worm. Both remedies are applied. 2.6's claim is
    narrowed to the buffer `.kb`. A17 and its twin A18 also take the teach-log read first, then refresh
    `chat.stored_facts` and its sets from the overwritten list, reset `_fact_shard_built_K`, and read back. A step-1
    review check confirms that the route and the V'_P builder take no chat handle. Sections: 2.6, 2.7, 2.8, 2.9 (B13),
    2.10 step 1.
11. B7 (e) labelled A6's sequencer misses as host misses. `host_routed` is now defined only for `propose_path`
    `fact_shard` or `escalated`; A6's misses are `seq_miss`, reported and never barred. Sections: 2.6, 2.7, 2.9 (B7,
    B8).
12. An existing test breaks and was unnamed (`tests/test_stage2_kb_scramble.py:53-55` and :65-68). Its amendment is
    named in steps 0 and 1; the GOLDEN pins stay. Sections: 2.8, 2.10 steps 0-1.
13. The D4 organ's seed was pinned per process only (`_ORGAN` is a singleton). Unit tests that reach D4, the bus or the
    server assert seed 7 on the organ and on both `_DEFAULT_SEED`s, or reset `_ORGAN`. Sections: 2.10 step 1, 3.8.
14. Minor: "every reserve slot zeroed" is defined as `_silent_reserve(j)` True; the STP citations are re-pinned to
    origin/main; the step-1 circuit twin uses heron, hunt and worm, not fish; B12 (b) keys `audit_disagree` on
    `committed`, not on `gate_decision` ADVANCE. Sections: inputs, 2.3, 2.7, 2.9 (B12), 2.10 steps 0-1.

**Round-5 changes (history; 2026-10-28, after three skeptics REFUTED bdd949cc5, all three on fix 1; the fix-by-fix
table with the code each citation was checked against is 0g; where round 6 changes a point, round 6 holds).** Fix
number, then the sections changed:
1. (blocking) B13 (a) could not tell the two V'_P designs apart, because fish stays a `.kb` patient through the
   built-in (cat, eat, fish). `infer_amb1`'s second hop-1 patient is now worm, which is in the D4 hand ANIMACY table
   and in no other `.kb` row. Overwrite condition (iv) and `vp_not_in_kb` gate B13 (a), and the false round-4
   counterfactual is struck. Worm missing from `vp` stays a FAIL, so the bar is not loosened (0g). Sections: 2.6, 2.8 (tables, branch table, D2, fixtures), 2.9 (P10, premise map, B13), 2.10
   step 0.
2. (blocking) V'_P provenance gets a data-flow test, and the frame guard can no longer be bypassed. The patient field
   is dropped from the :1654 read and every patient-field read is counted by line. One bar covers every `.kb` access
   from route entry to the returned decision, except listed lines. B7's frame count is stated to be no provenance
   evidence. A positive control requires the proxy to be live on every hop. Sections: 2.4, 2.7, 2.9 (B7, B13), 2.10
   step 1.
3. (blocking) The default PROPOSE path is the fact shard, not `_seq_block`. It is declared in the header and in 2.1,
   with `propose_path` and the candidate blocks recorded per hop. Every hop runs the buffer conflict read and builds an
   explicit drive vector. Host-routed misses are listed. B8 (1) is now a no-regression item, and its spiking-miss
   claim moves to a fact-shard-off twin (A21). The moat absence premise comes from the teach log and `builtin_facts`
   only, so a confabulated row that leads to an answer is a FAIL. Sections: header, 2.1, 2.2, 2.3, 2.6, 2.7, 2.8
   (A21), 2.9 (premise map, hop records, B3, B7, B8).
4. (blocking) Cross-seed rule: every M1 bar and every R bar holds only if it PASSES on all six dev seeds. An UNDEFINED
   or UNMEASURED seed counts as not meeting it. Sections: 2.9 (general rule, B11, seed invariance), 2.10 step 3, 3.7.
5. Same-seed references, nulls and liveness for the lesion and necessity bars. B3 and B4 need B1's engagement and
   A2 = A1. B6 needs A8's gate ABSTAIN and a new A8 null twin (A8b). B11 and B13 (b) need the unit live on A8. B4 (iii)
   needs the intact loop weights. A11's lesion is read back on the synapses. Sections: 2.7, 2.8 (A8b), 2.9 (B3, B4, B6,
   B11, B13).
6. The hop-1 patient VALUE is now intervened on: A19 and A20 overwrite the FIRST (heron, hunt) entry, frog -> worm.
   B13 (b) is restated as a test of route-level list reads only. Sections: 2.6, 2.8, 2.9 (P11, B13).
7. Runner changes named in the build lists: per-arm turn tables, informational `.rep0`, the live `.kb` proxy in the
   arm worker, and the `infer_xtier` write after turn 0 with a re-read. Sections: 2.4, 2.8, 2.10 steps 0-1, 3.8.
8. The step-0 (c) V'_P read takes the patient slice off the cleanup membrane right after the production
   `block_cleanup_scores` call (reserve slots carry no cleanup synapses there), never from the full probe. Section:
   2.10 step 0 (c).
9. Every M1 arm needs `sources` == {`tiny-demo +LTM`} and `ltm_bundle` == `wikidata_100k`, and a missing
   `corpus_present` key is UNDEFINED. Sections: 2.9 (general rule, premise map), section 5.
10. No step reads a gate seed by default: M3 step 0 names dev seeds and runs through the stage-2 worker. Every step-1
    unit test pins `BRAIN_CHAT_SEED`. Design seed 7 leaves the M3 organ count (seeds 1-6 and 8-16). Sections: 2.10
    step 1, 3.5, 3.7, 3.8, section 5.
11. B9 is UNMEASURED per arm pair. B12 (b) on A10 needs the hop-2 record. The D4 recorder joins step 0 (h)'s inert
    comparison, records the inner lexicon returns, records the OR check instead of raising, and labels the third
    `competent()` caller. The header lists the workspace's host read-outs. Sections: header, 2.7, 2.9 (B9, B12), 2.10.
12. Citation corrections: the clause test is in `_finish_query_patient`, `builtin_facts` is latched after turn 0, and
    the workstation main checkout has the corpus. One item was NOT applied as stated: production `_read_blocks` takes the
    per-block loop at :1389, not `_read_all_blocks` (the fact shard forces `no_batched_region`; evidence in 0g). Sections:
    2.1, 2.7, 2.9, section 5.

**Round-4 changes (history; 2026-10-28, after the third skeptic REFUTED bad12d536; the fix-by-fix table is 0f; where
a later round changes a point, the later round holds).**
1. B4 is scored on ENGAGEMENT of the spiking hop read: the turn's route trace must record `spiking_hop_reads` > 0 and
   the lesion must be read back as held (bridge key and zeroed loop weights). `n_ignited` and the conclusion are now
   the OUTCOME, so a held lesion that leaves slot 0 ignited is a FAIL. A turn with no recorded spiking read is
   UNDEFINED. The same per-turn rule now holds for every M1 bar (2.9, general rule).
2. B8 and B12 (b) have registered hop-record preconditions: which trace fields must exist and be non-empty, per bar,
   arm and probe, scored like the ask group's A4 (a) (2.9, hop-record table). Zero hop records are UNDEFINED, never
   PASS.
3. Step 0 records each teach's branch from the `competent()` call itself (its inputs and its return, written to the
   shard by a recorder in the arm worker), never from the `comprehension` field afterwards (2.7, 2.8, 2.10, 3.8).
4. The 'the frog eats the apple' row is corrected: (frog, eat) -> gnat is already held when it arrives, so D4's
   known-binding rule stores it whatever its margin; no comprehension is predicted. D2 surprise and reconsolidation
   are named, and step 0 STOPs on any in-place rewrite of an ambiguity or re-tell teach (2.8, 2.10).
5. V'_P ownership: the patient lines are built from the substrate (the patient pools that the relay lights across
   the stored blocks), never from `.kb` strings. Two new `.kb`-overwrite arm pairs (A17-A20) and bar B13 test it:
   take a patient out of the list, or add one to it, with the bound store intact. M1's abstain must follow the store,
   and today's route must follow the list (2.3, 2.6, 2.8, 2.9).
6. `corpus_present` and facts present: every shard must record `corpus_present` true, and before a probe is scored
   its premises must be read back from the stores the brain reads (the teach log, the decoded buffer rows, the
   built-in facts, the LTM). A missing premise makes that probe UNDEFINED, because an acquisition failure is not an
   inference failure (2.9, 2.10, section 5).
7. Also from the third verdict: the organ pass rule now counts an UNDEFINED seed as not meeting (3.7). R4 gets an
   UNMEASURED clause (3.7). The focus-1025 citation is corrected (3.2), and line numbers are dated (inputs).

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
REFUTED the revision fca06bd9a; every one of its fixes is applied (0d), and that revision's own self-skeptic pass is
in 0e. A third skeptic REFUTED bad12d536; its fixes are applied in the round-4 revision (0f). Three
more skeptics REFUTED that revision, bdd949cc5, all three on the same V'_P test; the twelve ranked fixes of their
synthesis are applied in the round-5 revision (0g). The round-5 synthesis ruled that revision, 3059090939,
SURVIVES-WITH-FIXES; its fourteen items are applied in the round-6 revision (0h). The round-6 fix-checker's two
minor fixes are round 7, a fresh skeptic's four fixes on fc476672a are round 8, and a verifier's five fixes on
33a6ac58c are round 9 (all three lists at the top). Section 7 copies Claude's 2026-10-27 ruling text; the keyed runbook RULING, which Claude posts, is the
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
:179); `research/runners/_p1_2_workspace_deliberation_loop_derisk.py` (`build_workspace_bridge` :148, the lesion's
zero weight on the `workspace_loop_{i}` populations :204-208); `research/runners/_phaseB_onebrain_sequencer_derisk.py` (`block_cleanup_scores` :64, `scores_to_drive` :92);
`research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py` (`make_block_drives` :198);
`research/runners/_phaseC_S5_divnorm_derisk.py` (`build_divnorm_score_bridge` :115, `onbridge_divnorm_drive` :158);
`research/runners/_phaseC_task1_cleanup_probe.py` (`block_cleanup_scores_full` :33);
`research/runners/_seq_fused_fabric.py` (`fused_seq_block` :336); `research/runners/tiered_fact_store.py`
(`query_patient_source` :197); `research/runners/sharded_phasor_store.py` (`shard_for` :166); `sim/regions.py`
(`RegionPathway.stp_disabled` :572); `sim/config.py` and `sim/bridge.py` (the STP defaults and the STP state init,
cited at origin/main f0fd444f7 below); `research/runners/stage2_exit_gate.py` (`infer` group :234-270, `apply_kb_scramble`
:452, `KbProbe` :505, `run_worker` :659 with its `corpus_present` read :671, `arm_validity` :908, `teach_acquired`
:1024, `acquired_teaches` :1041, `score_infer` :1107, `score_ask` and its A4 (a) rule :1132-1190, `kb_scramble_read`
:1330); `research/runners/comprehension_production_organ.py` (`_verb_selects_of` :209, `_animacy_of` :233,
`extract_transitive` :339, `ensure_built` :631, `competent` :719, which returns a bool, `judge` :738);
`webapp/server.py` (`_brain_vocab` :3640, `_integrated_loop_enabled` :3913, the D4 block :6826-6931 with the call
:6844, `known_binding` :6852-6855 and the abstain condition :6875, the D2 surprise and reconsolidation block
:6933-7013 with `extract_assertion` :6961 and `what_does` on the surface verb :6967);
`research/runners/surprise_production_organ.py` (`extract_assertion` :193, surface tokens, no lemma);
`research/runners/brain_conversational_agent.py` (`what_does` :772, a plain `query_patient`);
`research/runners/reconsolidation_production_organ.py` (`_inplace_rewrite` :111, `reconsolidate` :143); the third
packet skeptic's REFUTED verdict on bad12d536 (0f). Line numbers are at the branch base cadf0f40f, except the two
`sim/` files. **Round 6 re-pinned every citation it writes to origin/main f0fd444f7.** Every cited file other than
`sim/bridge.py` and `sim/config.py` is blob-identical there to the branch base (checked by blob hash: the composer,
the stage-2 runner, `sim/regions.py`, both sequencer builders, the gate module, the route, the chat brain, the D4
organ, the bus, `gnw_deliberation`, the workspace builder, the tiered store, the reconsolidation organ, the D6 store,
`webapp/server.py` and `tests/test_stage2_kb_scramble.py`), so their line numbers hold on both. The two `sim/` files
moved, and are cited at origin/main only: the bridge STP state init is `sim/bridge.py:1413-1414` (`cp_stp_x` ones,
`cp_stp_u` = `cfg.stp_U`), the global `stp_U` and `stp_tau_d` are `sim/config.py:949-950`, and the per-type depression
time constants are `sim/config.py:1343-1344` (E->E 200 ms). Earlier rounds' `sim/bridge.py:1288-1289` and :1341-1342,
and `sim/config.py:902-905`, :904, :934, :1295-1298 and :1327-1328, are stale and are no longer cited. The step-1 build
re-reads every cited line at its own base.
Round 5 also read, at main ee349f8b1, where every file it cites is blob-identical to the branch base cadf0f40f
(checked by blob hash, so the line numbers hold on both): `one_brain_composer.py` (the fact-shard default :355-372,
`no_batched_region` forcing `enable_batched` off :470-482, `_silent_reserve` :988, `_read_all_blocks` :1195,
`_read_blocks` :1380, `_fact_shard_active` :1392, `_ensure_fact_shard` :1400, `_fact_shard_candidates` :1434,
`_fact_shard_first_match` :1468, `_trace_query` :1580, `_seq_cleanup_conns` :1620, `_attributed_patient` :1777,
`query_patient` :1787, `_finish_query_patient` :1813); `_seq_vocab_shrink_derisk.py` (`reduced_cue_vocab` :183-196,
which never reads a fact's third field); `research/runners/brain_chat_tui.py` (`_build_tiny_demo`'s five built-in facts
:2449-2455 and its vocabulary :2459); `research/runners/brain_conversational_agent.py` (the onebrain buffer's kwargs
:285-322, `vocab_headroom` 128, no `confidence_gate`); `_gap3_learned_feature_compat_derisk.py` (the base ANIMACY table
:27-32) and `_phaseB_multicue_competition_spiking_derisk.py:52-61` (the table `comprehension_production_organ` imports
at :74-78); `comprehension_production_organ.py` (`competent` :719-735, its callers `judge` :747 and `repair_target`
:780, the organ's `seed=42` defaults :362 and :836); `webapp/gnw_three_organ_bus.py:204` (the third `competent()`
caller) and its `_DEFAULT_SEED` :114; `webapp/gnw_multistep_deliberation.py` (`conclusion_is_stored` :150-179, called
at :300); `webapp/gnw_deliberation.py` (`_DEFAULT_SEED` :69, `all_candidate_patients` :124, `conflict_gate` :144);
`_gnw_reentrant_metacog_gated_deliberation_derisk.py` (THR :105, R_MAX_DEFAULT :124, `_conf_from_nmda` :204-219,
`acc_conflict_gate` :326-337); `research/runners/stage2_exit_gate.py` (EXPECTED_SOURCE :71, BASE_ENV :98-100,
`turns_for` :321-328, `arms_for` :331-342, `_kb_triples` :448, `process_record` :623, `run_worker` :659-735 with the
`builtin_facts` latch :710-715, `arm_validity` :908-940 with its `corpus_present` default :933); `webapp/server.py`
(`_brain_chat_seed` :4038, `_default_ltm_bundle_dir` :3817-3847, `_resolve_ltm_bundle` :4112).
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

## 0f. The third packet skeptic's required fixes (REFUTED bad12d536), and where each is applied

Each cited file:line was re-read at the branch base for this revision. Items 1-6 are the six fixes Claude's round-4
brief named; items 7-9 are the rest of the verdict's required fixes, applied because each one tightens a bar.

| # | fix required | where |
|---|---|---|
| 1 | B4's precondition demanded its own effect (`n_ignited` = 0), so a held workspace lesion that left slot 0 ignited read UNDEFINED and B4 could never read NO-GO. The manipulation must be the engagement of the spiking hop read, not the flag | 2.9 B4: precondition = `spiking_hop_reads` > 0 on the turn + PROPOSE delivered + lesion read back as held (`ws_bridge_key`, `ws_loop_w_max` = 0); outcome = `n_ignited` and the conclusion; a turn with no recorded spiking read is UNDEFINED. 2.7 adds the per-turn counter; 2.9's general rule applies it to every M1 bar |
| 2 | B8 and B12 (b) had no hop-record precondition; B12 (b) was vacuously true at zero hop records, and B8's controls equal A0 whenever the flag fails to engage | 2.9: the hop-record precondition table (fields that must exist and be non-empty, per bar, arm and probe), scored like A4 (a); B8 and B12 (b) rewritten against it |
| 3 | Step 0 inferred each teach's `competent()` branch from the `comprehension` field, which is null for a pass-through, an out-of-scope turn, the gate off and a swallowed exception alike | 2.7 (the D4 recorder: `competent()`'s inputs and return written per call), 2.8 (branch table read from it), 2.10 step 0 (e) and GO rule, 3.6, 3.8 step 0 (a) |
| 4 | The 'frog eats apple' row stated the wrong consequence: (frog, eat) is already known, so D4 stores it whatever its margin; "comprehension is expected" is contradicted by goat/hay's covered-branch refusal. D2 surprise and reconsolidation (default ON) were never named; a lemma fix in D2 would turn the ambiguity tables into in-place rewrites | 2.8 (row rewritten, D2 paragraph, labels), 2.10 step 0 (`known_binding`, surprise and reconsolidation read back, STOP on any rewrite, named fixture fallback) |
| 5 | V'_P ownership hole: V'_P was built from `.kb` patient strings and a lit patient outside it was dropped, so the conflict read saw only patients the host list names; A3 keeps the patient multiset and the audit reads the same list, so nothing could detect it | 2.3 (V'_P from the relay-lit patient pools of every stored block, no `.kb` entry read), 2.6 (claim qualified), 2.8 (A17-A20: `.kb` patient-set overwrites with the bound store intact, both routes), 2.9 (P10, P11, B13; B7 bars `spurious_P` and V'_P provenance), 2.10 |
| 6 | Step 0 checked nonce forms and branches against the learned lexicons and the corpus vocabulary, which degrade silently without the corpus; nothing required `corpus_present` | 2.9 (general rule; the facts-present rule and premise map: a probe whose premises are not in the stores is UNDEFINED), 2.10 step 0 (f), 3.6, 3.8 step 0, section 5 |
| 7 | The organ pass rule's denominator was undefined when some seeds are UNDEFINED | 3.7: an UNDEFINED seed counts as not meeting |
| 8 | R4 had no UNMEASURED-when-not-engaged clause | 3.7 R4 |
| 9 | Minor: RULING focus-1025 was binding only until the 10-26 gate RESULT; some STP line numbers moved on main | 3.2; inputs paragraph |

## 0g. The round-4 synthesis's twelve ranked fixes (three skeptics REFUTED bdd949cc5), and where each is applied

Every code statement in this table was checked for this revision against the file at main ee349f8b1 (blob-identical
to the branch base for every file cited). Where a check needed a run, it was a seed-7 composer fixture in a scratch
process (numpy, a 9-word vocabulary, no gate seed, nothing committed).

| # | fix required | where | checked against |
|---|---|---|---|
| 1 | B13 (a) could not separate the designs: the built-in (cat, eat, fish) keeps fish a `.kb` patient after A17's fish -> frog, so round 3's host-listed V'_P keeps a fish line and abstains too. Use a word in no other `.kb` row and in the D4 hand ANIMACY; add overwrite condition (iv) and require `vp_not_in_kb`; strike the false 2.6 counterfactual | 2.6, 2.8, 2.9 (P10, premise map, B13 (a)), 2.10 step 0 (g) and GO rule | worm is in the hand ANIMACY that `comprehension_production_organ` imports (`_gap3_learned_feature_compat_derisk.py:29`, merged at `_phaseB_multicue_competition_spiking_derisk.py:52-61`; confirmed by import, `ANIMACY['worm'] == 'animate'`), so `competent(heron, hunt, worm)` passes it through as it did fish. worm is in no built-in fact (`brain_chat_tui.py:2449-2455`; it is only a vocabulary word, :2459) and in no string of any stage-2 group table, teaches and probes included (all of `GROUPS` scanned); this packet's other tables use apple, fish (LTM only) and nonce forms. bird qualifies on the same checks; worm was chosen. Applied with one tightening: the synthesis made `vp_not_in_kb` a precondition (else UNDEFINED); worm in `vp` but not in `vp_not_in_kb` is UNDEFINED as asked, but worm missing from `vp` with (iv) read back stays a FAIL, as round 4's "fish in `vp`" outcome was, so the bar is not loosened |
| 2 | No data-flow test of V'_P provenance; the frame guard was bypassable (:1654 read the patient field inside an allowed frame; B7's union check passes whenever list = store; `route_kb_reads` saw only route frames; `conclusion_is_stored` read `.kb` unbarred) | 2.4 (patient field dropped at :1654, counting by field and line, live proxy), 2.7 (`kb_reads`, `kb_patient_reads`, positive control), 2.9 (B7 (c)-(e), the provenance statement), 2.10 step 1 (sentinel unit twin) | :1654 builds `(agent, action, patient)` tuples and `reduced_cue_vocab` (`_seq_vocab_shrink_derisk.py:183-196`) never reads the third field, so dropping it is byte-identical; `conclusion_is_stored` (`gnw_multistep_deliberation.py:150-179`) is called only at :300, on the multistep chase path, which acts only on a chase-marker question; it is declared label-only and counted |
| 3 | The default PROPOSE path was declared wrongly: the fact shard is default ON and serves `query_patient`; (frog, chase) can miss on the host index; qp_lesion and the moat hop produced no drive or `acc_conf` record; the moat premise read from the bound store turned a confabulation into UNDEFINED | header, 2.1, 2.2, 2.3, 2.6, 2.7 (`propose_path`, `fs_candidates`, `host_routed`), 2.8 (A21), 2.9 (premise map, hop-record table, B3, B7, B8) | `_FACT_SHARD_DEFAULT_ON = True` (:363), not set by stage-2 `BASE_ENV` (:98-100); `_fact_shard_active` (:1397-1398) needs integrated loop off and `confidence_gate` 0, the agent's default (`brain_conversational_agent.py:285-322` passes none). Fixture: `enable_fact_shard` True. One correction to the synthesis: on the fixture the (frog, chase) candidate set was NOT empty (one non-matching block from a DG bucket collision, decoded and rejected), so the miss is "the index's candidates plus a host compare over their decodes", which 2.1 states; `fs_candidates` records the set |
| 4 | No cross-seed combination rule | 2.9 (general rule (vi), B11, seed invariance), 2.10 step 3, 3.7 | packet text |
| 5 | Lesion and necessity bars lacked same-seed references, nulls and liveness (`docs/TERMS.md`, lesion and load-bearing) | 2.7 (`cp_w_max`, `cp_n_syn`), 2.8 (A8b), 2.9 (B3, B4, B6, B11, B13) | `ws_loop_w_max` is recorded as 0.0 when no synapse is found (2.7), hence the intact and synapse-count references |
| 6 | The hop-1 patient value was never intervened on (A3 keeps (heron, hunt); A19 overwrote the re-tell, so the first match was unchanged) | 2.6, 2.8 (A19, A20 plan), 2.9 (P11, B13 (b)) | under the fact shard, `query_patient` selects the FIRST matching candidate block and `_finish_query_patient` reads that block's `.kb` patient for the clause test (:1822), so the first entry is the one a list-first-match reader sees |
| 7 | Runner changes were missing from the build lists | 2.4, 2.8, 2.10 steps 0 and 1, 3.8 | `turns_for` gives `permuted_turns` only to the arm named `permuted` and order tables only to A4 order arms (`stage2_exit_gate.py:321-328`); `arms_for` adds `.rep0` only to lesion arms (:331-342); `run_worker` builds the brain inside the first `brain_chat` (`reset=(i == 0)`, :699). Round 8 corrects the rest of this cell: round 5 also wrote "each conversation loads its own LTM copy (`webapp/server.py:4290-4300`, :4395)", which is false at the default. The shared store is default ON (:4203), the tiny-demo's `.ltm` is a `SharedLTMView` (:4307-4316, called at :4396), and its `store()` raises (`shared_ltm.py:215-216`), so A15, A16 and the step-0 `infer_xtier` read run offiso (round-8 item 1, 2.8) |
| 8 | Step 0 (c)'s V'_P read used a slice production never sends | 2.10 step 0 (c) | `block_cleanup_scores_full` wires every slot (`_phaseC_task1_cleanup_probe.py:33-57`); production skips unrecruited reserve slots (`_seq_cleanup_conns` :1634, `_silent_reserve` :988-998, whose docstring records `__free10__` 5270 against `look` 4080); the chat buffer has `vocab_headroom` 128 (`brain_conversational_agent.py:285`) |
| 9 | Shard validity depended on the LTM bundle outside the repository, and a missing `corpus_present` read as present | 2.9 (general rule (i)), premise map, section 5 | `_default_ltm_bundle_dir` (`webapp/server.py:3817-3847`) looks outside the checkout and falls back to `wikidata_core_15k`, which gives the same `source` string; `process_record` records `ltm_bundle` (`stage2_exit_gate.py:623-638`); `arm_validity` reads `shard.get("corpus_present", True)` (:933) |
| 10 | M3 step 0 could run on gate seed 42 | 2.10 step 1 (unit tests), 3.5, 3.7, 3.8 step 0, section 5 | `_brain_chat_seed()` returns 42 when `BRAIN_CHAT_SEED` is unset (`webapp/server.py:4038-4062`), as do `gnw_deliberation._DEFAULT_SEED` (:69) and `gnw_three_organ_bus._DEFAULT_SEED` (:114); the organ's own `seed=42` defaults (`comprehension_production_organ.py:362`, :836) |
| 11 | B9 UNMEASURED per pair; B12 (b) on A10 needs the hop-2 record; the D4 recorder in step 0 (h); inner returns recorded; OR recorded, not raised; the third `competent()` caller; the header's host read-outs | header, 2.7, 2.9 (B9, B12), 2.10 step 0 (h) | `gnw_three_organ_bus.py:204` calls `competent()`; `_conf_from_nmda` and `acc_conflict_gate` as cited in the header |
| 12 | Citation corrections | 2.1, 2.7, 2.9, section 5 | Applied: the :1822 clause test is in `_finish_query_patient` (:1813); `builtin_facts` is latched after turn 0 (`stage2_exit_gate.py:710-715`): the five built-ins plus turn 0's fact; the workstation main checkout HAS `data/corpus/tinystories.txt`, and the detached `sim-rev` worktrees lack it. NOT applied as stated: "production `_read_blocks` goes to `_read_all_blocks`; :1389 is not taken". With the fact shard on, `__init__` sets `no_batched_region = True` (:371-372), which forces `enable_batched = False` (:480-482), so `_read_blocks` takes the per-block loop at :1389. The fixture read back `enable_batched` False, and `_read_blocks` called `_read_block` once per block. `_read_all_blocks` (:1195, `len` at :1206) runs only with `BRAIN_FACT_SHARD_RETRIEVAL=0` (A21), where `enable_batched` read back True |

## 0h. The round-5 synthesis's fourteen items (3059090939 SURVIVES-WITH-FIXES), and where each is applied

Every code statement in this table was checked for this revision at origin/main f0fd444f7, by reading the file there
(`git show`), not the branch. Every file cited below except `sim/bridge.py` and `sim/config.py` is blob-identical at
origin/main to the branch base cadf0f40f (inputs paragraph). No run was made, and no gate seed was read except the
committed s42 `infer__qp_lesion` shard of the 10-26 gate, read as a file for item 4.

| # | item | where | checked against (origin/main f0fd444f7) |
|---|---|---|---|
| 1 | (blocking) Patient-field counter: no insertion rule, no entry-level control | 2.4 (counting by site with block id, insertion rule, class property setter, entry-level control), 2.7 (`kb_unwrapped`, positive control), 2.9 (B7 (c), B13 (b), table), 2.10 step 1 (one test per insertion path) | plain-dict appends `one_brain_composer.py:709` (`hear`), :724 (`hear_multiframe`), :732 (`store`), :753 (`store_fact`); the fixture pattern `stage2_exit_gate.py:479-480`; `update_on_mismatch` :2159-2163. Also found and covered: the reconsolidation fallback `reconsolidation_production_organ.py:140`, `tiered_fact_store.py:263` (`clear`), `d6_hebbian_store.py:266` and :397 (`pop`), and `TieredFactStore.__setattr__` (:132-151) forwarding a `.kb` reassignment to the buffer, which round 5's "only :499 assigns" claim missed; hence a property setter, not an `__init__` wrapper |
| 2 | (blocking) B6 unreachable: RegionPathway floors weights at 0.01 | 2.3 (lesion bullet), 2.5, 2.9 (B6), 2.10 step 1 | `sim/regions.py:1009` (pathways) and :945 (within-region) `weights.append(max(0.01, float(w)))`; both builders wire every pathway as `RegionPathway` (`_seq_vocab_shrink_derisk.py:146-165`, `_phaseB_onebrain_sequencerK_derisk.py:127-155`; round 7 re-read both ranges at origin/main 53d64c2ef: the lists are assigned to `cfg.region_pathways` at :166 and :156); the workspace lesion's explicit wiring keeps 0.0 (`_p1_2_workspace_deliberation_loop_derisk.py:204-214`, `_gnw_rung1_ignition_curve_derisk.py:91-106` fills `np.full(..., attractor_weight)` unclamped) |
| 3 | (blocking) B7 (c) unpassable: new K site unlisted; step 1 moves listed lines | 2.3 (where K is read), 2.4 (keyed sites, re-pin, unit test), 2.7, 2.9 (B7), 2.10 step 1 | `_ensure_sequencer(self, K)` :1641; its only production caller `_seq_block` reads `K = len(self.kb)` at :1700 and calls it at :1705, after the integrated-loop-off host path :1685-1691 has returned (`git grep` at origin/main: no other production caller) |
| 4 | (blocking) B3 precondition unmeasurable | 2.7 (per-turn trace, `patch_calls_in_hop`), 2.9 (B3, table), 2.10 step 1 | `install_patch` :373-388, one counter `rec["calls"]` (:379, incremented at :383); written once in the shard at :720 inside the output dict :718-726; the s42 gate shard `infer__qp_lesion.json` reads `{'patch': 'qp_lesion', 'calls': 29}` over 10 turns |
| 5 | (blocking) B8 (1b) passes with dead pools | 2.7 (`blocks_matched` as a list), 2.9 (B7 containment scope, B8 (1b), table) | A21: `no_batched_region` forced only with the fact shard on (:371-372), so off it stays False and `enable_batched` True (:480-482); A21's PROPOSE is `_seq_block`'s host first-match over `_read_blocks()` (:1685-1691), which reads no match pool |
| 6 | (blocking) UNDEFINED item lets a bar pass on the rest | 2.9 (rule (vii), facts present, hop-record preamble, B8, B9, B12 (b)), 3.7 | `score_ask` :1132; A4 (a) drops unacquired moats (:1144-1148) and scores the rest (:1150-1154) with `ASK_MOAT_MIN_DEFINED = 1` (:85) |
| 7 | B1 precondition contains its outcome | 2.9 (REF-A1 with hop 2, B1, B5, B8, B12 (a), table) | Applied, except one part, NOT applied as stated: the read-count precondition is `spiking_hop_reads` >= 1, not >= 2. Evidence: `compositional_chain_route.py:289-291` returns None at once on an ambiguous hop 1 (`if ambiguous1: return None`), a control flow `_hop_spiking` keeps (2.2), so a hop-1 ABSTAIN leaves exactly one hop read on the turn. A >= 2 precondition would make that case UNDEFINED, while the same item requires it to be a FAIL. With >= 1 it is a FAIL, which is stricter. The hop-2 record and both ADVANCEs moved to the outcome. B5, B8 and B12 (a), which leaned on "B1's engagement" for both ADVANCEs, now read REF-A1, which carries hop 2's ADVANCE explicitly, so none of them is loosened |
| 8 | Abstain source unrecorded | 2.2 (which abstain), 2.7, 2.9 (P5, P10, B5, B13 (a), table) | `acc_conflict_gate` :326-337: ABSTAIN when `n_ignited >= 2` at R_max, or `n_ignited == 1` with `conf < theta_hi` (:332, :336); `conf` from `_conf_from_nmda` :204-219 (:216). Added a third label, `winner`, for M1's own ADVANCE-with-a-non-slot-0-winner abstain (2.2 table) |
| 9 | In-place rewrite leaves drives and V'_P stale | 2.3 (drive-dirty bullet), 2.7 (`vp_formed_*`), 2.8, 2.9 (B13 (a)), 2.10 step 1 | `_write_block` :857, `_seq_dirty` only under `integrated_loop` (:874-875); `update_on_mismatch` :2144, in place at :2158-2163; rebuild condition :1671. Same pattern found and covered at :956-957 (homeostatic scaling), `webapp/da_tag_capture.py:189-190` and :509-510, `webapp/one_brain.py:674-675`, `d6_hebbian_store.py:271-272` and :402-403 |
| 10 | B13 (a) tests `.kb` only | 2.6 (narrowed claim; review check), 2.7 (teach log first), 2.8 (A17/A18 chat-list refresh, index reset, read-backs), 2.9 (B13 (a)), 2.10 steps 0-1 | `brain_chat_tui.py:1010-1014` (`stored_facts` and the three sets from `.kb`, inside `_refresh_facts` :987-1016, called only at :985, :1418, :1431, :2336); the fixture leaves them (`stage2_exit_gate.py:455-457`); `resolve_compositional_chain(composer, question)` (`compositional_chain_route.py:262`); the index holds codes and block ids (`one_brain_composer.py:1416-1432`), rebuilt only on a K change (:1410-1412). Both remedies applied: narrowed claim AND the A17/A18 overwrite; D6's engram flags, which would make `_refresh_facts` read the engrams instead, are default OFF (`d6_hebbian_store.py:216`, :280) |
| 11 | B7 (e) mislabels A6 misses | 2.6, 2.7 (`host_routed` scope, `seq_miss`), 2.9 (B7 (e)) | `_seq_block`'s spiking path under the integrated loop (:1699-1718) versus the host path (:1685-1691); on A21 the host path scans the very rows recorded, so a miss with decoded rows cannot occur there (listed as an instrument fault if it does) |
| 12 | Existing test breaks unnamed | 2.8 (test amendment), 2.10 steps 0 (7) and 1 | `tests/test_stage2_kb_scramble.py:53-55` (extras: only the two kb arms, only in `infer`) and :65-68 (no other arm carries a scramble); `arms_for` at origin/main lists exactly those two extras (run on an archive of origin/main). Round 7: A17-A20 also carry a scramble (mode `kb_overwrite`), so the :65-68 allowlist names them too (`arm_scramble` `stage2_exit_gate.py:364-369`, worker :686-693) |
| 13 | D4 organ seed pinned per process only | 2.10 step 1 (seed test), 3.8 step 3 | `_ORGAN` :833, `get_organ(seed: int = 42)` :836, built once :848-869; `ComprehensionProductionOrgan.seed` set at :363; `gnw_deliberation._DEFAULT_SEED` :69 and `gnw_three_organ_bus._DEFAULT_SEED` :114 read `BRAIN_CHAT_SEED` at import with fallback 42 |
| 14 | Minor: reserve slot, STP citations, circuit twin words, B12 (b) key | 2.10 step 0 (c); inputs, 2.3; 2.10 step 1; 2.3, 2.9 (B12 (b)) | `_silent_reserve` :988-998 (`__free` prefix), skipped at :1634; `sim/bridge.py:1413-1414`, `sim/config.py:949-950`, :1343-1344 at origin/main. Round 6 also resolves one scope ambiguity of round 5's B7 by taking the wider reading: the single-active-line check covers every buffer-resolved hop of every M1 arm except A11, not A1 alone |

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

`OneBrainComposer.query_patient` (`one_brain_composer.py:1787`) returns ONE patient per (agent, action). **At the
production default its block selection is the fact-shard fast path, not `_seq_block` (round 5; round 4 had this
wrong).** `_FACT_SHARD_DEFAULT_ON = True` (:363-367), and the stage-2 `BASE_ENV` (`stage2_exit_gate.py:98-100`) does not
set `BRAIN_FACT_SHARD_RETRIEVAL`. `_fact_shard_active` (:1392-1398) is True whenever the integrated loop is off
(`_INTEGRATED_LOOP_DEFAULT_ON = False`, `webapp/server.py:3910`), `confidence_gate` is 0 (the agent passes none, so the
default 0.0, :129) and the store is non-empty. `query_patient` (:1805-1808) then calls `_fact_shard_first_match`
(:1468). `_fact_shard_candidates` (:1434) intersects per-role DG sparse indices that `_ensure_fact_shard` (:1400) builds
from the `.kb` strings (`self.kb[i][0].get(r)` at :1420, for r in agent, action and patient; rebuilt only when
`len(self.kb)` changes, :1410-1412). The candidate blocks are then decoded one at a time on the RF substrate
(`_read_one_block`), and the first one whose decoded agent and action equal the cue is selected (:1478-1481). The
candidate set is a host index over the list's agent and action strings. An empty set returns (None, None) before any
block is decoded. A non-empty set whose decodes all mismatch also returns (None, None), and on a seed-7 fixture the
(frog, chase) set held one such non-matching block (a DG bucket collision, 0g). Either way the buffer abstains and
`TieredFactStore` asks the LTM. Only when the set cannot be formed (`_FS_ESCALATE`, an unindexed cue role, :1476-1477)
does the full path run: `_seq_block` (:1680), which at integrated loop off is a host first-match over every decoded row
(:1685-1690). **The fact shard also changes the read layout.** A fact-shard composer sets `no_batched_region = True`
(:371-372), which forces `enable_batched = False` (:480-482). So production `_read_blocks` (:1380) takes the per-block
loop at :1389 (`_read_block(i)` for i in `range(len(self.kb))`). The batched `_read_all_blocks` (:1195, `len(self.kb)`
at :1206) runs only with `BRAIN_FACT_SHARD_RETRIEVAL=0`, the A21 twin of 2.8 (both read back on a seed-7 fixture, 0g).
With `BRAIN_INTEGRATED_LOOP=1`, `_fact_shard_active` is False and the selection is the K-way sequencer's BG first-match
priority WTA (block i inhibits every j>i), built by `_ensure_sequencer` (:1641). At the default
`enable_seq_vocab_shrink=True` (:133, branch :1651-1663) that is the REDUCED fabric,
`build_sequencerK_reduced_bridge` (`_seq_vocab_shrink_derisk.py:98`), with word lines only for the distinct stored
agents and actions; `build_sequencerK_bridge` over all V words is built only with that option off (:1665-1670). A
third branch, `integrated_loop == "fused"`, selects through `fused_seq_block` (`_seq_fused_fabric.py:336`); the chat
server never takes it, because `_integrated_loop_enabled` returns a bool (`webapp/server.py:3913`, passed at :4377 and
:4470). In every case a second patient under the same cue is invisible to the caller, and the patient CONTENT is the
decoded bound vector. `query_patient` reads `.kb` itself. It reads `len(self.kb)` in `_fact_shard_active` (:1398) and
in `_ensure_fact_shard` (:1410), and every entry's role strings when that index is rebuilt (:1420). The index feeds
`_fact_shard_candidates`. On the full path it reads `len(self.kb)` in the per-block `_read_blocks` loop (:1389, the
production layout) or in `_read_all_blocks` (:1206, the A21 layout). In `_finish_query_patient` (:1813), it reads
`self.kb[idx][0].get("patient")` for the clause test that routes a clause patient (:1822; label-only, the answer is the
decoded word). It reads the attribute-key check in `_attributed_patient` (:1782; the production agent builds the
composer with `enable_attributed=True`, `brain_conversational_agent.py:178`). With the composer's trace on, it reads
`len(self.kb)` in `_trace_query` (:1589). `_ensure_sequencer` reads `self.kb[:K]` (:1654).

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

1. **PROPOSE.** `target, tier = composer.query_patient_source(x, rel)` (`tiered_fact_store.py:197-216`: the same
   per-tier calls, in the same order, as `query_patient`). The answer is the spiking decode of a bound block, selected
   as in 2.1: at the default by the fact-shard candidate set and a host first-match over the candidates' decodes.
   Recorded per hop (2.7): `propose_tier`, the buffer's `propose_path` (`fact_shard`, `escalated` or `seq_block`), and the
   fact-shard candidate block ids `fs_candidates`.
2. **CONFLICT READ, on every hop, whichever tier serves it (round 5).** The buffer conflict unit of 2.3 reads, in
   spikes, how many DISTINCT patients the matching buffer blocks carry, and the relay turns its spike count into
   `slot1_drive`. It runs whatever PROPOSE returned (None included) and whichever tier answered. Only a cue word
   without a line skips it (2.3), which is recorded as `cue_in_reduced_vocab: false` with `slot1_drive` 0. With an LTM
   present, the routed shard's candidates are listed on every hop as well (2.3, `ltm_candidates`). Each one whose
   string differs from `target` (every one, when `target` is None) gets its own slot. On a buffer-served hop these are
   the cross-tier slots, and on an LTM-served hop they are the LTM's other patients. No hop calls `conflict_gate`.
3. **EVALUATE.** One workspace read on the warm P1.2 workspace the C5 chase uses: `_get_bridge(_DEFAULT_SEED,
   lesion)` and `_get_theta(_DEFAULT_SEED)` from `webapp/gnw_deliberation.py`. `_DEFAULT_SEED` reads
   `BRAIN_CHAT_SEED` at import, which the stage-2 worker sets to the arm's seed before it imports the server
   (`run_worker`, :666); it is recorded per hop as `ws_seed`. `_deliberate_hop_conf` is NOT used: it builds its own
   drive vector (slot 0 at `IGNITE_PA`, distractor slots at `IGNITE_PA * DISTRACTOR_FRAC`) and has no argument for a
   graded slot-1 drive. M1 builds the vector itself, as `conflict_gate` (`webapp/gnw_deliberation.py:144`) does:
   `drives = [0.0] * n`; `drives[0] = IGNITE_PA` when `target` is not None; `drives[1] = slot1_drive`;
   `drives[2 + i] = IGNITE_PA` for each listed other-tier candidate i (at most n - 2, the dropped count recorded);
   every other slot 0 (no distractors, so the C5 chase's host-random `_pick_distractors` is not used). The vector is
   built and recorded in full on every hop, all-zero included (a miss, the qp lesion), so every hop has a `drives`
   record and one workspace read. Then
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
lists every intermediate case. A hop with no PROPOSE word, a silent `acc_conf` and no listed candidate drives nothing,
so `n_ignited` is 0 and the gate returns COMMIT, a miss read off an all-zero drive vector. Two such hops are the
qp-lesion hop 1 and the moat hop (frog, chase). The vector and the read are recorded there too (round 5).

**Which abstain (round 6).** The gate returns ABSTAIN in two regimes (`acc_conflict_gate`, :334-336): IGNITION
(`n_ignited >= 2` after the retry walk reached R_max) and GRADED (`n_ignited == 1` with `conf < theta_hi`, where
`conf` is the host margin formula of `_conf_from_nmda`, :204-219, `|g1 - g2| / (g1 + g2 + NORM_EPS)` over the top two
slots' NMDA). Only the first is the conflict regime of `calibrate_theta` that the registered regimes above name; the
second can come from any sub-ignition NMDA on another slot. A third abstain is M1's own: an ADVANCE whose winner is not
slot 0 (table above). Every abstaining hop records `abstain_regime` (`ignition`, `graded` or `winner`) with `conf`,
`theta_hi`, `n_ignited`, `gate_cycles` and `winner_kind` beside it. B5 and B13 (a) read only the ignition regime with
no cross-tier slot (2.9).

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
over the patient vocabulary V'_P in both (read from the substrate, below, never from `self.kb[:K]`), and V'_P joins
the rebuild condition in both branches of `_ensure_sequencer` (:1651-1670); the reduced builder is the path every
arm runs. Per block b, decoded agent and action word lines each pass a gate `g{b}{role}_w` that the cue
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
- **Patient vocabulary from the substrate (round 4).** V'_P is NOT read from `.kb`. On every drive rebuild (a write
  dirtied the store, or K changed: `_ensure_sequencer` :1672-1679), the PATIENT cleanup slice of every stored block
  b = 0..K-1 goes through the production relay (the same `onbridge_divnorm_drive` call on the composer's score bridge
  at `input_gain` 1.0 that the agent and action slices take), and V'_P is the set of word pools that fire for at
  least one block, in block order. The relay runs before the fabric check: if the lit set differs from the built
  fabric's V'_P, the reduced fabric is rebuilt with the new V'_P before the drives are set (V'_P joins the rebuild
  signature beside `(agentsA, actionsX)`). The builder reads the block count K (`len(self.kb)`, the same count
  `_read_blocks` reads, declared) and no `.kb` patient field. **Where K is read (round 6).** `_ensure_sequencer(self,
  K)` takes K as an argument (:1641). Its only production caller is `_seq_block`, which reads `K = len(self.kb)` at
  :1700, on the integrated-loop path only; at the default the host path (:1685-1691) returns before it, so production
  never builds the sequencer today. The conflict read therefore reads `len(self.kb)` at a NEW site, its own entry point
  in the composer (a step-1 method; its name is fixed in the step-1 build), and passes it to `_ensure_sequencer`. That
  site is allowed in 2.4's map as kind `len` only, keyed by its function and statement; an entry or patient-field read
  there is unlisted. The builder runs inside `_ensure_sequencer`, whose own :1654 read
  took every entry's patient field until round 5, so a caller FRAME cannot tell the builder's reads from :1654's.
  From step 1, :1654 reads the agent and action fields only (2.4), and the 2.4 guard counts patient-field reads by
  field and by SITE (round 6: a site is keyed by function, statement and kind, since step 1 itself moves the line
  numbers). `vp_kb_entry_reads` is the count of patient-field reads from `_ensure_sequencer`, the builder or the
  conflict read's entry point, at any site, and B7 bars it at 0. Host parts, declared in the header: the relay, the set union in block order and
  the rebuild trigger. Consequences: a patient the bound store holds gets a line whatever the list says, and a string
  the list holds but no block decodes gets none. After the decision the audit records `vp_not_in_kb` and `kb_not_in_vp`
  (record-only; never read by the route). A patient line exists only for a patient some block lights (section 5 gives
  the cell count; the full-V layer would be about V/V'_P times larger). B7's consistency checks pass whenever the list
  equals the store, so they are not provenance evidence (2.9). The provenance test is the data-flow arm A17 (2.8). It
  removes a patient string from every `.kb` entry while the store keeps it, and B13 (a) requires the abstain to follow
  the store.
- **Patient decoded lines.** Per block b, word lines `d{b}P_w` for w in V'_P, driven by the relay above from the
  block's PATIENT cleanup slice. That slice is already computed by the same per-block read: `_seq_cleanup_conns`
  installs cleanup synapses for every main role, and `block_cleanup_scores` returns only the agent and action slices.
  The build adds an optional `with_patient=True` return of the patient slice (no extra substrate run) and an optional
  third role in `make_block_drives` and `run_sequencerK_reduced_with_drive` (each default off, byte-identical). Since
  V'_P is the union of the lit pools of the same rebuild, a matching block's lit patient is in V'_P by construction.
  A lit patient outside V'_P at a read (`spurious_P` > 0, recorded per hop) therefore means the fabric and the read
  disagree (a stale fabric); it is dropped, never silently, and B7 bars it.
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
- **The conflict lesion holds a weight of exactly zero (round 6).** Both sequencer builders wire every pathway as a
  `RegionPathway` (`_seq_vocab_shrink_derisk.py:146-165`; `_phaseB_onebrain_sequencerK_derisk.py:127-155`, the list assigned to
  `cfg.region_pathways` at :156), and
  `RegionPathway` wiring clamps every sampled weight to `max(0.01, w)` (`sim/regions.py:1009`; the within-region
  recurrent wiring does the same at :945). A `pP_w -> acc_conf` pathway built at `w_c` = 0 would therefore hold 0.01 on
  every synapse, and A11's read-back `cp_w_max` = 0 (B6) could never be met. So the lesion is not a build weight. Both
  the intact and the lesioned builds wire `pP_w -> acc_conf` the same way at `w_c`. Under the lesion, right after every
  fabric build (a V'_P or K rebuild included), the builder writes the weights at that pathway's synapse indices to
  exactly 0.0 on the bridge's weight array, and drops any cached connectivity derived from them. The synapse set is
  then identical to the intact build's, at zero weight. The named alternative, used if a later rebuild or cache
  restores the floored value (the read-back at the read decides, not a read right after the build), is explicit wiring
  of that pathway in both builds through `inject_explicit_wiring` at `w_c` or 0.0, as `build_workspace_bridge` wires
  its lesioned loop (`_p1_2_workspace_deliberation_loop_derisk.py:204-214`; `_build_assembly_loop_population` keeps the
  given weight unclamped). Either way `cp_w_max` and `cp_n_syn` are read back from the bridge at the read (2.7), and a
  step-1 unit test requires `cp_w_max` == 0.0 with `cp_n_syn` equal to the intact build's and > 0 (2.10).
- **STP reset at every read.** At the start of every conflict read, after `reset_sequencerK_state`, the read sets
  `cp_stp_x = 1` and `cp_stp_u = cfg.stp_U` (the bridge's own initial values, `sim/bridge.py:1413-1414` at origin/main)
  on the `pP_w -> acc_conf` synapse indices. Reason: `reset_sequencerK_state` (`_phaseB_onebrain_sequencerK_derisk.py:179`)
  clears membranes, recovery variables, gates, coupling EMAs and conductances but never touches STP state, and its
  drain is 20 steps (20 ms at dt 1 ms) against a depression recovery time constant of 200 ms (`stp_tau_d`,
  `sim/config.py:950`; per-type E->E 200 ms, :1343-1344). Without the reset, a read right after another would start
  depressed and its count would depend on query history. The reset is a host write of synaptic state between queries,
  declared in the header beside the existing per-query housekeeping.
- **The drives follow every store write (round 6).** `_write_block` sets `_seq_dirty` only when `self.integrated_loop`
  is on (`one_brain_composer.py:874-875`), and D2's `update_on_mismatch` (:2144) rewrites a block in place at unchanged
  K (:2158-2163: `_write_block` at :2162, the `.kb` entry at :2163). At the default (integrated loop off) the conflict read would then keep the
  pre-rewrite drives and the pre-rewrite V'_P, since `_ensure_sequencer` rebuilds the drives only when dirty or when K
  changed (:1671). Step 1 therefore sets `_seq_dirty` on every `_write_block` whenever the composer's conflict unit is
  built, whatever `integrated_loop` says. With the conflict unit not built nothing changes, so the default is
  byte-identical. The other writers of `store_conns` that dirty the drives only under the integrated loop get the same
  rule: homeostatic scaling (:956-957), `webapp/da_tag_capture.py:189-190` and :509-510, `webapp/one_brain.py:674-675`,
  and `research/runners/d6_hebbian_store.py:271-272` and :402-403. A step-1 unit test rewrites a patient between two
  conflict reads and requires V'_P and the patient lines of the second read to follow the rewrite. Each hop records the
  turn label and the K at which the V'_P it read was formed (`vp_formed_turn`, `vp_formed_K`, 2.7).
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
record). The LTM candidate list is a host input to the workspace drive (below), declared. So are the PROPOSE word's
presence in slot 0 and the fact-shard candidate set that selected it (2.1).

**Relay to the workspace (declared host residual).** The sequencer and the workspace are separate bridges, so the
count crosses by host relay: `slot1_drive = IGNITE_PA * min(1, n_conf / n_conf_ref)`, linear, no threshold.
`n_conf_ref` is the `acc_conf` count for the two-distinct-patient cue of the step-1 circuit fixture on design seed 7
(2.10 step 1), never a count from a bar table. The decision threshold is the workspace's ignition, read by the
unchanged `acc_conflict_gate` (2.2).

**LTM tier, and the cross-tier case.** `TieredFactStore.query_patient` asks the buffer first and the LTM only when the
buffer abstains (`tiered_fact_store.py:194-195` through `_tiered`; tier provenance in `query_patient_source`
:197-216). The sequencer covers the buffer only. Round 5: on EVERY hop the buffer conflict read runs and, with an LTM
present, `all_candidate_patients(ltm.shard_for(x), x, rel)` (`webapp/gnw_deliberation.py:124`) lists the routed shard's
patients; both go into the one drive vector of 2.2, whichever tier answered. Two cases:
- **LTM-sourced hop** (the buffer abstained): the buffer read normally finds no matching block (`slot1_drive` 0,
  recorded); the LTM's PROPOSE word takes slot 0 and every other listed LTM patient takes its own slot at `IGNITE_PA`,
  so two distinct LTM patients put the gate in its CONFLICT regime, as `conflict_gate` would with two candidates. The
  enumeration is host closed-form (role-unbind of the shard's stored composites through `_iter_facts`) and
  deduplicates by host string compare; it is recorded per hop as `conflict_read: rf_shard_unbind`. A buffer abstain
  whose decoded buffer rows under the cue are not empty is listed `host_routed` (2.7): the fact-shard index, not the
  bound store, made that miss.
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
  2 or more distinct patients while the hop `committed` a patient; round 6: keyed on `committed`, not on
  `gate_decision` ADVANCE, because the route's commitment, not the gate's label, is what a reader downstream gets) and,
  within it, `xtier_disagree` (the second patient
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
instead of raising; (c) replacing the audit's return value with garbage leaves the route's result identical. Round 4:
the same guard counts any `.kb` entry read (`__getitem__`, `__iter__`) whose caller frame is the V'_P builder of 2.3
as `vp_kb_entry_reads`, and raises on it in the unit test; `__len__` (the block count K) passes.

**Round 5: counting by field and line, one bar over the whole decision, and a live proxy.** The round-4 guard could
be bypassed in three ways. The V'_P builder runs inside `_ensure_sequencer`, whose :1654 read took every entry's
PATIENT field in an allowed frame. `route_kb_reads` counted only frames of `compositional_chain_route` inside an open
hop scope. And a `.kb` reader in another module, such as `conclusion_is_stored`
(`webapp/gnw_multistep_deliberation.py:150-179`), was never barred. So:
- **:1654 reads the agent and action fields only (a step-1 change on the default path).** The read becomes
  `(f.get("agent"), f.get("action"), None)`. `reduced_cue_vocab` unpacks `(a, x, p)` and never reads `p`
  (`_seq_vocab_shrink_derisk.py:183-196`), so the reduced vocabularies, the fabric and every decision are
  byte-identical (unit-tested: the vocabularies, the rebuild signature and the fabric's synapse arrays are equal with
  and without the change, on the step-1 fixtures and on the `infer` and `infer_amb1` stores).
- **Patient-field reads are counted by site, not by frame.** Each entry's fact dict is held as a counting `dict`
  subclass (so every `isinstance(fact, dict)` check is unchanged), which returns the plain values and records every
  read of its `patient` key with the caller's file, function, line and statement text, plus the entry index (block id)
  through which the read reached it: the proxy stamps that index on the counting dict at each `__getitem__` and at
  each step of an iteration (round 6, for the entry-level control below). The forms counted are `get`, `[]`,
  `items`, `values`, `keys` followed by `[]`, and a `dict(...)` copy (the subclass overrides `__iter__`, so CPython's
  `dict(...)` leaves its fast path for exact-iterator dicts and reads through `keys` and `__getitem__`). A unit test
  enumerates these forms and asserts each is counted and returns the plain value. These counts are
  `kb_patient_reads` (2.7).
- **One bar over the decision scope.** The decision scope opens at the route's entry (`resolve_compositional_chain`
  with the flag on) and closes when the route has fixed its decision, before the audit calls. Every `.kb` access in
  that scope is recorded with its caller (file, function, line, statement) and kind (`len`, entry, patient field),
  from any module and any frame. That covers the route, the composer, the V'_P builder, `conclusion_is_stored` and any
  other reader. The allowed sites are the listed composer sites of 2.7 and three declared label-only reads: the
  attribute-key test in `_attributed_patient` (:1782 at origin/main), the clause test in `_finish_query_patient`
  (:1822; it reads the patient field, and the answer is the decoded word) and `conclusion_is_stored` (the multistep
  chase's provenance label; these probes carry no chase marker, so it is expected unreached and any count is listed).
  Every other access counts as `kb_reads_unlisted`. A patient-field read at any site other than the clause test, the
  fact-shard index build in `_ensure_fact_shard` (:1420; a query_patient cue queries only the agent and action
  indices, and every fact-shard lookup inside a hop records the cue roles it queried) and `conclusion_is_stored` also
  counts as unlisted. B7 bars `kb_reads_unlisted` at 0 (2.9). **Sites are keyed by (function, statement, kind), not
  by line (round 6).** Step 1 edits `_ensure_sequencer` (the :1654 read) and adds code to the composer, which shifts
  :1700, :1782 and :1822, so a line-keyed allowlist would move under the build it guards. Each allowed site is the
  triple of its enclosing function's qualified name, its statement text with whitespace normalised (for example
  `_finish_query_patient`, `stored = self.kb[idx][0].get("patient") if idx < len(self.kb) else None`, kind patient
  field), and its kind. The line number at the revision is recorded beside the key and is never the key. The map
  gains one new site: the conflict read's own `len(self.kb)` read (2.3), allowed as kind `len` only. The map is
  re-pinned at the step-2 revision and committed before any step-2 data, and a step-1 unit test resolves every key to
  exactly one statement of the current source and fails on a key that matches none or more than one.
- **The proxy is live in the arm worker, not only in a unit-test composer.** The stage-2 arm worker installs it before
  the server import builds any composer. Round 6 replaces round 5's `__init__` wrapper with a class-level `kb`
  property on `OneBrainComposer`: its setter wraps any list assigned to `.kb` into the counting list, at :499 and at
  any later assignment, including one forwarded by `TieredFactStore.__setattr__` (`tiered_fact_store.py:132-151`, which
  sends `composer.kb = ...` on to the buffer). Round 5's "nothing reassigns `.kb`" claim did not cover that forwarding.
  It guards the buffer's `.kb`, the list that the route scanned and that the overwrite and scramble arms change.
  LTM-tier reads inside the decision (the LTM `query_patient` and `all_candidate_patients` on the routed shard) are
  declared host reads (2.3), recorded as calls with their returns, not per entry. The proxy's behaviour preservation
  is tested at step 1 on whole turn tables, flag off and on (2.10 step 1).
- **Insertion rule (round 6).** Entries reach `.kb` as plain dicts, so a proxy that wrapped only the entries present
  when it was installed would stop counting at the first new fact. The composer appends a plain `(fact, None)` in
  `hear` (`one_brain_composer.py:709`), `hear_multiframe` (:724), `store` (:732) and `store_fact` (:753); the
  `kb_overwrite` fixture, like `apply_kb_scramble`, writes `buf.kb[i] = (dict(fact, patient=p), handle)`
  (`stage2_exit_gate.py:479-480`); `update_on_mismatch` writes `f2 = dict(...)` and `self.kb[idx] = (f2, None)`
  (:2159-2163); the reconsolidation organ's fallback writes `composer.kb[idx] = (fact2, None)`
  (`reconsolidation_production_organ.py:140`); `promote_buffer_to_ltm` calls `self.buffer.kb.clear()`
  (`tiered_fact_store.py:263`); and the D6 store pops (`d6_hebbian_store.py:266`, :397). So the counting list wraps the
  fact dict of every `(fact, handle)` tuple that enters it, through `append`, `insert`, `extend`, `+=` and
  `__setitem__` (slices included), and through the property setter. It supports `clear`, `pop`, `remove` and
  `__delitem__` with list semantics. Any other mutating method raises in the arm worker, so no path can store an
  unwrapped dict silently. A non-dict fact (a foreign type) is stored as is and recorded as `kb_unwrapped`, which
  makes every hop of that turn UNDEFINED for the bars that read patient-field counts. One step-1 unit test per
  insertion path (`store`, `store_fact`, `hear`, `hear_multiframe`, the `kb_overwrite` fixture, `update_on_mismatch`,
  the reconsolidation fallback, a reassignment through `TieredFactStore`, then `clear` and `pop`) asserts that the
  entry it leaves is a counting dict and that a patient read on it is counted.
- **Entry-level positive control (round 6).** The site-level control of 2.7 shows that the proxy saw some read on the
  hop. It does not show that the entry the hop actually used was a counting dict. So every hop whose buffer PROPOSE
  selected a block (`selected_block` not None; every arm except A4, whose patch replaces PROPOSE) must list, in
  `kb_patient_reads`, the clause-test patient read of `_finish_query_patient` (:1822 at origin/main) with that block
  id. A hop without it is UNDEFINED for B7 (c)'s patient-field clause, for `vp_kb_entry_reads` and for B13 (b), and it
  is listed (2.7, 2.9).

### 2.5 Flags and lesions

- `BRAIN_CHAIN_SPIKING_HOPS` (default OFF): with it unset, `resolve_compositional_chain` is byte-identical (golden
  hash over a fixture battery in a unit test, plus the existing infer arms' primary fields).
- `BRAIN_CHAIN_SPIKING_HOPS_CONFLICT_LESION`: the `pP_w -> acc_conf` weights written to exactly 0.0 after every
  fabric build, the synapses kept (acc_conf hears nothing). Round 6: not a build weight of 0, which `RegionPathway`
  would floor to 0.01 (2.3). Predicts: a within-buffer multi-valued hop commits the first match.
- `BRAIN_CHAIN_SPIKING_HOPS_WS_LESION`: the hops use the workspace built with assembly self-recurrence zeroed
  (`_get_bridge(seed, lesion=True)`, the C5 lesion's bridge). Predicts: no hop ignites, the conclusion vanishes.
- `qp_lesion` (the existing in-process patch, `query_patient` returns None): predicts the conclusion vanishes, the
  reversal of today's 7/7 invariance.

### 2.6 What stays host under M1 (declared in the header)

The regex dispatch, `_ROLE_NOUN_HINTS` and the same-verb fallback, the two-hop order, the default PROPOSE selection
inside `query_patient` (the fact-shard candidate set, a host index over the list's agent and action strings, and a host
first-match over the candidates' spiking decodes; on escalation or with the fact shard off, a host first-match over
every decoded row, 2.1), the label-only `.kb` reads at :1782 and :1822, the reduced agent and action vocabularies read
from the fact list (`_ensure_sequencer` :1654, agent and action fields only from step 1) and the host abstain when a cue
word has no line, the block count K, the union of the relay-lit patient pools into V'_P with its rebuild trigger
(2.3), the relay of each block's cleanup scores into the sequencer as it is (host read and clip, `input_gain` current into the divnorm score bridge, the rheobase
deciding which pools fire, the host conversion to a fixed 1500 pA drive, the remap with a lit word outside the reduced
vocabulary dropped), the per-query STP reset, the slot-to-word map, the cross-bridge relay of the conflict count, the
mapping of the gate's outputs onto hop outcomes (2.2), the per-hop drive vector and the workspace read-outs (the
NMDA margin, the ignition count over THR, the argmax winner, theta_hi and R_max), the LTM candidate slots of every
hop (the cross-tier slots on a buffer-served hop, the LTM's other patients on an LTM-served hop) with the enumeration
`all_candidate_patients` and its host string compares (2.3), the derived framing (`frame_derived_answer`) and the
verify re-parse (`brain_chat_tui.py:1922-1942`). Under the conflict read the first-match answers only when the
matching rows drive a single patient line, so it has no patient choice left to make; the step-2 record checks per hop
that the PROPOSE word is the word of the single active patient line. M1 claims one thing: within the buffer, the
identity of each hop's patient and the abstain on ambiguity are decided by spiking reads of the bound store. Across
tiers the abstain is host-listed (2.3) and claimed as no more than that. A buffer MISS is not claimed at the default
(round 5). There the fact-shard candidate set, a host index over the list's strings, decides which blocks are ever
decoded, so a miss is the index's empty or non-matching candidate set (2.1). A hop whose `propose_path` is
`fact_shard` or `escalated`, whose decoded buffer rows under the cue are not empty, and whose PROPOSE returned None is
listed `host_routed` (2.7, B7). On A6 (`propose_path` `seq_block` with the integrated loop on) the same pattern is a
miss of the spiking K-way sequencer, not of a host path, so it is listed `seq_miss`, reported and never barred
(round 6). The miss read through the decoded store and the match pools is claimed only on the fact-shard-off twin
A21 (2.8, B8 (1b)).

**What the claim does and does not cover, and the interventions that test it.** The cue lines (agents, actions) are
host-listed from `.kb`; the claim does not cover them. The patient lines are not: V'_P is the set of patient pools
the relay lights from the bound blocks (2.3), so neither the line set nor the conflict count is read from the list.
Three kinds of intervention would expose a host supply, and each has a bar. (1) The patient VALUE at the list level:
A3 deranges the `.kb` patient strings with the bound vectors intact (multiset kept). The conclusion must follow the
bound vectors, not the list (B2). (2) The patient SET and the ABSTAIN, which a multiset-preserving derangement cannot
test. A17 takes a patient out of the list. On `infer_amb1` the second hop-1 patient is worm, which is in no other `.kb`
row (no built-in, no other teach; 0g). The `.kb` string of the worm row is overwritten with frog, so after the overwrite
worm is in NO `.kb` entry while the bound store still decodes it (condition (iv), 2.8). Round 3's host-listed V'_P would
then give worm no line: the worm block's lit pool would fall outside V'_P and be dropped, one line would stay, and M1
would ADVANCE. Under V'_P from the store, worm keeps its line and M1 abstains. That difference is B13 (a). It holds only
when (iv) is read back and A17's hop-1 record lists worm in `vp_not_in_kb`. **Struck (round 5):** round 4 stated the same
counterfactual for fish. It was false, because the built-in (cat, eat, fish) keeps fish a `.kb` patient after the
overwrite, so a host-listed V'_P would keep a fish line and abstain too, and A17 could not tell the two designs apart.
(3) The hop-1 patient VALUE as a first-match list reader sees it, and route-level list reads. A19 overwrites the FIRST
(heron, hunt) entry of `infer_dup`, t_heron_hunt's, from frog to worm. The fact shard selects that block first, and
`_finish_query_patient` reads its `.kb` patient for the clause test (:1822), so a reader that took the hop-1 value from
the list would now get worm. M1 must still derive through frog, and the list's second patient must not supply an
abstain. B13 (b) tests route-level list reads only. It cannot test V'_P provenance: under the `gp{b}` gating a line is
lit only through a matching block, so a line that the list adds stays silent, and a list-built V'_P would also
ADVANCE on A19. The flag-off twins (A18, A20) show that each overwrite bites: today's route follows the list. B13 bars
both (2.9). Until B13 passes, the claim that the patient lines belong to the substrate is a design statement, not a
result.

**Which list the claim is tested against (round 6).** The interventions above write the composer buffer's `.kb`, and
the claim is narrowed to that list: "not read from the list" means not read from the buffer's `.kb`. Two other host
lists hold the same strings. `ChatBrain._refresh_facts` builds `chat.stored_facts` and the `agents_set`,
`actions_set` and `patients_set` from `.kb` (`brain_chat_tui.py:987-1016`, the sets at :1012-1014), and refreshes them
only on a teach, at build and on `list_facts` (:985, :1418, :1431, :2336), so a `.kb` write between turns leaves them
stale. The kb_scramble fixture leaves them untouched by design (`stage2_exit_gate.py:455-457`). The fact-shard index
holds concept codes and block ids, not strings, and is rebuilt only when K changes
(`one_brain_composer.py:1410-1432`), so a same-K overwrite leaves it as built before the write. Neither list can reach
the M1 decision by the route's own signature: `resolve_compositional_chain(composer, question)`
(`compositional_chain_route.py:262`) takes no chat handle, and the V'_P builder runs inside the composer. A step-1
code-review check confirms that neither the route, `_hop_spiking`, the conflict read's entry point nor the V'_P
builder takes or reaches a chat handle, and the decision-scope recording (2.4) would list any `.kb` read they make.
A17 and its flag-off twin A18 also overwrite the other lists, so that worm leaves every host list the chat holds
(2.8): the teach log is read first, then `chat._refresh_facts()` rebuilds `chat.stored_facts` and its sets from the
overwritten `.kb`, the buffer's `_fact_shard_built_K` is set to -1 so the index is rebuilt from the overwritten list at
its next use, and all three are read back. The circuit-level sentinel twin of step 1 has no chat object. It
overwrites `.kb`, resets `_fact_shard_built_K`, and reads both back; its claim is the narrowed one.

### 2.7 Engagement record, scoped to the hop reads

Instrumentation lives in the stage-2 runner's arm worker (behaviour-preserving wrappers, as the kb_scramble arms do).
A hop scope opens when the route enters `_hop_spiking` and closes when it returns. Per hop:
`{turn, hop, agent, relation, proposed (query_patient result), propose_tier, propose_path, fs_candidates,
selected_block, source_tier, host_routed, cue_in_reduced_vocab, m_counts (per block), blocks_matched (m{b} over the
composer's match_thresh, 0.06 at :132), patient_line_counts (per pP_w line with any spike), patient_lines_active,
spurious_P, stp_reset (cp_stp_x and cp_stp_u read back on the conflict synapses at the read's first step),
cp_n_syn, cp_w_max, acc_conf_spikes, slot1_drive, ltm_candidates, xtier_slots, xtier_dropped, drives, ws_seed,
ws_bridge_key, ws_loop_w_max, ws_loop_n_syn, n_ignited, conf, winner, winner_kind, gate_decision, gate_cycles,
committed, decoded_rows, decoded_rows_source, vp, vp_lit_by_block, vp_kb_entry_reads, vp_not_in_kb, kb_not_in_vp,
audit_buffer, audit_ltm, audit_ltm_all, audit_disagree, xtier_disagree, route_kb_reads, composer_kb_reads,
kb_patient_reads, kb_reads_unlisted, theta_hi, abstain_regime, seq_miss, vp_formed_turn, vp_formed_K,
patch_calls_in_hop, kb_unwrapped}`.
Round 6 adds these fields and definitions:
- `blocks_matched` is the LIST of block ids whose `m{b}` cleared `match_thresh`. "blocks_matched = n" in a bar means the
  list holds n ids, and "the selected block lies in `blocks_matched`" means `selected_block` is one of them.
- `abstain_regime` (2.2) is `ignition`, `graded` or `winner` on an abstaining hop and null otherwise, with `theta_hi`
  recorded beside `conf`, `n_ignited`, `gate_cycles` and `winner_kind` on every hop.
- `host_routed` is defined only when `propose_path` is `fact_shard` or `escalated`, the two paths on which a host
  index decides which blocks are decoded (item 11). `seq_miss` is the same pattern (PROPOSE None, `decoded_rows` under
  the cue not empty) when `propose_path` is `seq_block` with the integrated loop on (A6): a miss of the spiking K-way
  sequencer, reported and never barred. On A21 (`seq_block` with the integrated loop off) PROPOSE is the host
  first-match over the very rows recorded (`decoded_rows_source: propose_scan`, `_seq_block` :1685-1691), so a miss
  with decoded rows under the cue cannot occur there. If one is recorded it is listed as an instrument fault, and the
  hop is UNDEFINED for every bar that reads it. On A4 (`patched`) neither flag is defined.
- `vp_formed_turn` and `vp_formed_K` are the turn label and the K of the drive rebuild that formed the V'_P the read
  used (2.3). Each overwrite fixture records the turn index after which it ran (2.8), so a bar can tell whether V'_P
  was formed after the overwrite.
- `patch_calls_in_hop` is, on A4, the number of calls the installed `qp_lesion` patch received inside the hop scope (the
  worker snapshots the patch record's cumulative `calls` at scope open and close). `propose_path` is `patched` only when
  this is >= 1, meaning the patch served the hop's buffer PROPOSE.
- `kb_unwrapped` counts `.kb` entries that the proxy could not wrap (2.4); it is 0 on every hop a bar reads.
Round 5 adds these fields:
- `propose_tier` is the tier `query_patient_source` reports (buffer, ltm or None).
- `propose_path` is the buffer's selection path, read by behaviour-preserving wrappers of `_fact_shard_first_match`,
  `_fact_shard_candidates` and `_seq_block`. It is `fact_shard` when the fact shard served the cue (the default), or
  `escalated` when the shard returned `_FS_ESCALATE` and `_seq_block` ran. It is `seq_block` when the fact shard was
  inactive (A6, A21), and `patched` on A4.
- `fs_candidates` is the candidate block id list `_fact_shard_candidates` returned (empty included), with the cue
  roles it queried.
- `host_routed` is true when the buffer's PROPOSE was None while `decoded_rows` under the cue is not empty. That is a
  miss the fact-shard index or another host path made against the bound store, and it is listed on every arm (2.9 B7).
  Round 6 restricts it to `propose_path` `fact_shard` or `escalated` (above).
- `cp_n_syn` and `cp_w_max` are the synapse count and the largest absolute weight on the `pP_w -> acc_conf` pathway,
  read back from the bridge at the read.
`decoded_rows` are the decoded `(agent, action, patient)` rows under the hop's cue, read from the bound store, never
from `.kb` (record-only). The fact shard serves almost every cue at the default and decodes only its candidates, so
the instrumentation makes one `_read_blocks` call after the decision. At the production layout that is the per-block
loop :1389 (2.1), and on A21 it is `_read_all_blocks`. It records `decoded_rows_source: post_decision_scan`, or
`propose_scan` when PROPOSE itself scanned every block (`escalated` or `seq_block`). `ws_bridge_key` is the
`_get_bridge` cache key the read used, `(seed, lesion)`. `ws_loop_w_max` is the largest absolute weight on that
bridge's `workspace_loop_{i}` populations, read back from the bridge at the read. It is 0 under the lesion, which
builds them at zero weight by explicit wiring (`_p1_2_workspace_deliberation_loop_derisk.py:204-214`), and it is also recorded as 0.0
when the populations hold no synapse, so their synapse count `ws_loop_n_syn` is recorded beside it, and B4 reads both
(2.9). `vp` is the V'_P of the fabric the read used (2.3), and `vp_lit_by_block` the per-block lit patient pools of the
rebuild that formed it. `route_kb_reads` counts `.kb` accesses whose caller frame is `compositional_chain_route` while
a hop scope is open.

`composer_kb_reads` counts the reads from `one_brain_composer`, by site. Round 6: every listed site below is keyed by
(function, statement, kind), as 2.4 states; the line numbers are origin/main's (blob-identical to the branch base), are
recorded beside the key, and are re-pinned at the step-2 revision before any data. Round 6 adds one site: the conflict
read's entry point, `len(self.kb)` (kind `len` only; the K it passes to `_ensure_sequencer`, 2.3). The listed composer
lines (round 5; corrected) are these. `_fact_shard_active` :1398 (`len`). `_ensure_fact_shard` :1410 (`len`) and :1420 (entry read of
every main role's string, the patient's included, when the index is rebuilt), the index that `_fact_shard_candidates`
(:1434-1466) queries. The per-block `_read_blocks` loop :1389 (`len`; the production layout, 2.1). `_read_all_blocks`
:1206 (`len`; A21 only). `_ensure_sequencer` :1654 (entry read, agent and action fields only from step 1). `_trace_query`
:1589 (`len`, when the composer's trace is on). `_attributed_patient` :1782 (`len` and the attribute-key test,
label-only). `_finish_query_patient` :1822 (`len` and the patient field, the clause test, label-only). `_seq_block`
:1700 (`len`; A6 only). `kb_patient_reads` lists every patient-field read by caller site, with the block id through
which it was reached (round 6). `kb_reads_unlisted` counts every `.kb` access in the decision scope (2.4) whose caller
is neither a listed composer site nor a declared label-only read (:1782, :1822, `conclusion_is_stored`). It also
counts every patient-field read at any site except the clause test (:1822), the index build (:1420) and
`conclusion_is_stored`. A per-turn total of the same counts is kept for the whole decision scope, since a
reader outside the hop scopes counts there too. Turn-level counts outside the route (`_chain_facts`,
`rich_answer_composer._stored_facts`, elaboration) are reported separately and never barred.

**Positive control on the proxy (round 5).** A hop whose `composer_kb_reads` is empty means the proxy was not live, and
a zero bar count there would prove nothing. So every hop must show (a) the conflict read's :1654 read (its fabric check
runs before the line test) and (b), on every arm except A4 (whose patch replaces PROPOSE), a read on its PROPOSE path.
That is :1398 when `propose_path` is `fact_shard` or `escalated`, :1389 or :1206 when PROPOSE scanned every block, and
:1700 on A6. A hop missing either is UNDEFINED for every bar that reads it, and it is listed. Round 6 adds (a') the
conflict read's own `len` site (2.3), whose K the :1654 read uses, and the ENTRY-level control of 2.4: on every hop
whose buffer PROPOSE selected a block (`selected_block` not None, every arm except A4), `kb_patient_reads` lists the
clause-test patient read of `_finish_query_patient` (:1822) with that block id. A hop missing the entry-level read is
UNDEFINED for B7 (c)'s patient-field clause, for `vp_kb_entry_reads` and for B13 (b), and it is listed; the sites are
matched by key, never by line.

**Per-turn route trace (round 4).** For every turn, flag on or off: `{turn, route_dispatched (the possessive parse
matched), spiking_hop_reads, n_hop_records, route_exception}`. `spiking_hop_reads` counts the `_hop_spiking` scopes of
that turn that made their workspace read (a non-null `n_ignited` was recorded) and closed without an exception. It is
the engagement evidence the bars read (2.9); the flag's env value never is. On a flag-off arm it is 0 by construction,
and on a non-chain turn (`route_dispatched` false) it is 0 and recorded. **Per-turn patch calls (round 6).** On an arm
with an installed patch (A4 and its .rep0), the trace also records `patch_calls_before`, `patch_calls_after` and
`patch_calls_delta` for the turn. Reason: `install_patch` keeps one cumulative counter, `rec["calls"]`
(`stage2_exit_gate.py:379-384`), which `run_worker` writes once into the shard's `patch` field after the last turn
(:718-726, the field at :720). The gate's s42 `infer__qp_lesion` shard reads `calls` 29 over its 10 turns, so no turn's
own calls can be read from it. The worker snapshots `rec["calls"]` right before and right after each `brain_chat` call.
**Per-turn store snapshot (round 8).** On every M1 arm, every turn whose role is not `teach` (chain probe or not)
records `store_snapshot = {turn, K, rows, read, ltm_under_cue}`, taken once right after `brain_chat` returns and
before the next turn: all K decoded rows of the buffer with their block ids, from one `_read_blocks` call (the
per-block loop :1389 at the production layout, `_read_all_blocks` :1206 on A21), and the record-only `audit_ltm_all`
scan under each cue of the arm's premise-map rows. The facts-present rule reads every map cue's bound-store side from
it (2.9), because the hop records exist only for cues the route hopped. It is outside every decision scope and hop
scope, and its reads are reported under the label `store_snapshot`, never barred.

**D4 recorder (every arm this packet adds, A0-A21 with A8b and the M3 arms, and step 0).** It is a set of
behaviour-preserving wrappers installed in the arm worker as `KbProbe` is, with no production edit. Each wrapper calls
the original and returns its value unchanged. They wrap `ComprehensionProductionOrgan.competent`, the organ module's
`extract_transitive`, and (round 5) the module functions `_verb_selects_of` and `_animacy_of`. `competent()` resolves
those two as module globals at call time (`comprehension_production_organ.py:719-735`), so wrapping the module
attributes sees the very calls it makes. For every `competent` call the recorder writes these items:
- the turn label;
- the caller, from the caller frame (round 5: three callers exist, `judge` :747, `repair_target` :780, and the
  three-organ bus's organ C, `webapp/gnw_three_organ_bus.py:204`);
- the inputs `(n0, v, n1)`, each word's membership in the `brain_vocab` set that the caller passed, and that set's size;
- the ordered inner calls of `_verb_selects_of` and `_animacy_of` made inside this `competent` call, with their
  RETURNS as recorded (round 5: never recomputed, since a second learned-lexicon call is a second read and need not
  match);
- the return.
`fully_covered` and `fully_oov` are derived from those recorded returns, by replaying `competent()`'s own boolean
expression; a term that short-circuit evaluation skipped is marked `not_evaluated`. Their OR is compared with the
return, and the result is written as `or_matches_return`. It is recorded, never raised (round 5). A False is listed and
makes that call's branch UNDEFINED. `extract_transitive` calls are recorded with their text and return. A teach's
branch is read from its `judge`-frame call and nothing else; a bus-frame or `repair_target`-frame call is recorded and
reported, never read as the branch. True with `fully_covered` is `covered`, True with `fully_oov` is `oov`, and False is
`pass_through`. If `extract_transitive` returned None the branch is `out_of_scope`. If no `judge`-frame call was
recorded on the turn the branch is `not_called` (the gate is off, or D4 was skipped), never `pass_through`. The teach
turn's response fields `comprehension` (margin, `comprehended`, `known_binding`), `surprise` and `reconsolidation` are
copied beside it.

**Teach log and store read-back (every arm this packet adds).** After the teach block and BEFORE any `.kb` overwrite
fixture (round 6: the teach-log read comes first, because A17 and A18 now also rewrite `chat.stored_facts`, 2.8), the
worker records `stored_facts_after_teach`, a list read of `chat.stored_facts` (`_facts_of`, no substrate run, so no
reply can change). On an overwrite arm a second read, `stored_facts_after_overwrite`, follows the fixture and is
reported; the premise map never reads it. **Where the bound store is read (round 9: rewritten, because round 8's
snapshot made the earlier text false).** On an M1 arm (the term as pinned in 2.8) the bound store is read on every
probe turn: the per-turn `store_snapshot` above is the M1 arms' per-probe-turn bound-store read. It is a decoded-row
substrate read (one `_read_blocks` call decoding all K blocks) plus the record-only `audit_ltm_all` scan under each
map cue, taken after the reply on every non-teach turn, and the facts-present rule reads each map cue's bound-store
side from it (2.9). Its inertness is shown by the step-1 whole-table behaviour-preservation test, which runs with it
installed (2.10). Two other reads keep their own scope. The hop records hold the decoded rows and the LTM candidates
under each hop cue, which B5, B8 and B12 (a) read on the hops they name. The fixtures' read-backs (`infer_xtier`'s LTM
write, the `.kb` overwrites, 2.8) belong to their arms. A flag-off arm makes no snapshot and no hop record, so its
premises come from the teach log, `builtin_facts` and the `infer_xtier` fixture's read-back only (2.9, sources
(1)-(3)). Step 0 (f) is a separate read, made in step 0 only: the whole premise map on the decoded buffer rows and on
the LTM through `ltm.query_patient` and `all_candidate_patients`, with step 0 (h)'s inert-read check beside it (2.10).

### 2.8 Arms (stage-2 runner, `infer` group, all informational, never counted in the Stage-2 verdict)

Session tables: `infer` intact (the registered one) and four new tables, run under the same harness (numpy CPU,
tiny-demo +LTM, corpus present and recorded, `PYTHONHASHSEED=0`, `rich` omitted). Teach labels are given where a
bar or the premise map (2.9) reads them. **How an arm gets its table (round 5; a runner change, built in step 0).**
`turns_for` returns `permuted_turns` only to the arm named `permuted` and an order table only to the A4 order arms
(`stage2_exit_gate.py:321-328`). Every informational arm would otherwise run `g["turns"]`. So an informational entry
may carry its own `turns` (the four tables below, and `infer_garbled` of 3.6) or `table: "permuted"` (A12, and
m3_permuted and m3_rs_lesion of 3.6). `turns_for` returns that table, and `arm_validity`'s turn-table check (:940-944),
which calls `turns_for`, follows. `arms_for` adds `.rep0` only to lesion arms (:331-342), so an informational entry
with `rep: true` also gets its `.rep0` arm (A4, A5 and A11 here, m3_rs_lesion in 3.8). `arm_spec` already strips
`.rep` (:348). The golden hash of `tests/test_stage2_kb_scramble.py` pins every pre-existing arm.
- `infer_amb1` (hop-1 ambiguity): 'the heron hunts the frog' (t_heron_hunt), 'the frog eats the gnat' (t_frog_eat),
  'the heron hunts the worm' (t_heron_hunt_worm); then p_prey_eat, c_heron_hunt, c_dog_chase. **Round 5: worm, not
  fish.** fish is a built-in patient ((cat, eat, fish), `brain_chat_tui.py:2454`), so it never leaves the `.kb` patient
  set and A17 could not remove it (2.6). worm is in the D4 hand ANIMACY table that `comprehension_production_organ`
  imports (`_gap3_learned_feature_compat_derisk.py:29`, merged at `_phaseB_multicue_competition_spiking_derisk.py:52-61`)
  and in no other `.kb` row. It is not in the five built-in facts (`brain_chat_tui.py:2449-2455`; worm is only in the
  vocabulary, :2459). It is in no string of any stage-2 group table, and in no other table of this packet (0g). It is
  a vocabulary word from the start, so it needs no reserve slot.
- `infer_amb2` (hop-2 ambiguity): t_heron_hunt, t_frog_eat, 'the frog eats the apple' (t_frog_eat_apple); then the
  same probes. The chain under test: hop 1 (heron, hunt) -> frog, one patient; hop 2 (frog, eat) -> gnat and apple,
  two patients, so p_prey_eat must abstain on both routes (P5). Roles: agent frog, action eat, patient apple, as D4's
  `extract_transitive` and the store's positional parse both read it.
- `infer_dup` (duplicate facts): the four intact teaches, then verbatim re-tells of the hop-1 fact ('the heron hunts
  the frog', label t_heron_hunt_re) and of the hop-2 fact ('the frog eats the gnat', t_frog_eat_re), then the
  `infer` probes.
- `infer_xtier` (cross-tier ambiguity): the four intact teaches and the `infer` probes, with one fixture write.
  **Round 5: the write comes after turn 0, not "before the first turn".** `run_worker` builds the brain, and with it
  the `TieredFactStore` that holds the LTM handle, inside the first `brain_chat` call (`reset=(i == 0)`,
  `stage2_exit_gate.py:699`; the tiny-demo attach, `webapp/server.py:4390-4400`), so before turn 0 there is no LTM
  handle to write to. Pre-building the session and sending turn 0 with `reset=False` would change turn 0's build path,
  a confound, so it is not used. **Round 8: whose LTM, and the env that makes it private.** Round 5 justified the
  write with "each conversation loads its own LTM copy" (`webapp/server.py:4290-4300`, :4395). At origin/main that is
  false at the default: `_SHARED_LTM_DEFAULT_ON = True` (:4203), so `_ltm_store_for_conversation` returns
  `_shared_ltm.view_for(...)` (:4307-4316), called for the tiny-demo at :4396 (the comment at :4395 is stale), and
  `chat.inner.composer.ltm` is a `SharedLTMView`, whose `store()` raises `SharedLTMWriteError`
  (`research/runners/shared_ltm.py:215-216`). So A15 and A16, and the step-0 `infer_xtier` read (2.10), carry
  `BRAIN_SHARED_LTM=0` and `BRAIN_LTM_RNG_ISOLATE=1` in their informational entries' `env`, the same two values on both
  arms, so the pair still differs only in the M1 flag. `run_worker` sets an entry's env before it imports the server
  (`stage2_exit_gate.py:664-665`, the import at :673). This is the `offiso` configuration: with the shared store off
  and isolation on, each conversation loads a private, RNG-isolated copy (`_load_ltm_isolated`, :4280-4290, returned at
  :4306), a `ShardedPhasorStore`. The shared-LTM flip battery measured offiso equal to the shared default on the whole
  record, on 6/6 gate seeds (A1, `webapp/server.py:4198-4199`), so the LTM these two arms read is the production one
  in content. `BRAIN_ONE_BRAIN`, which forces the shared store on (:4223-4224), is not set (default OFF,
  `webapp/one_brain.py:49`). Right after turn 0 (t_heron_hunt) returns, the arm worker records the fixture's engagement
  record: `ltm_type`, which is `type(chat.inner.composer.ltm).__name__`, and the two env values as the worker read
  them. Only when `ltm_type` is `ShardedPhasorStore` does it write (heron, hunt, fish) into that session's private LTM,
  `chat.inner.composer.ltm`, with `ltm.store` (the call `build_ltm_from_facts` uses, `tiered_fact_store.py:108`;
  `ShardedPhasorStore.store`, `sharded_phasor_store.py:214`). Any exception the write raises is recorded as
  `store_exception`. No write ever goes through `ltm.shared_store`, the one process-shared store every conversation
  reads. A `SharedLTMView` (or any type other than `ShardedPhasorStore`), a recorded `store_exception`, or a missing
  `ltm_type` makes the fixture not engaged, and the arm (A15 or A16) UNDEFINED on that seed for every prediction and
  bar that reads it (P9, B9's A16/A15 pair, B12 (a) and (b)). It is listed, never scored around. It is a
  fixture step, not a turn, so the buffer never holds it. The worker reads it back with `ltm.query_patient(heron, hunt)`
  and `all_candidate_patients(ltm.shard_for("heron"), "heron", "hunt")` right after the write, and again right before
  the first probe. The second read-back is the one the premise map uses. Hop 1 of p_prey_eat then has frog in the
  buffer and fish in the LTM.

**Which `competent()` branch each new teach takes** (`comprehension_production_organ.py:719`: fully covered = verb in
VERB_SELECTS and both nouns in ANIMACY, hand or learned; fully OOV = verb in no verb table and both nouns in no animacy
table and not in `_brain_vocab(chat)`; anything else passes through unjudged and is stored). `competent()` returns a
bool, and a pass-through, an out-of-scope turn, the gate off and a swallowed exception all leave `comprehension` null,
so the branch is never read from that field. Step 0 reads it from the D4 recorder (2.7): the `judge`-frame call's
inputs, the vocabulary membership of each word, `fully_covered`, `fully_oov` and the return, written to the shard at
the call. Hand-table facts below are from `_phaseB_multicue_competition_spiking_derisk.py:52-75` (its ANIMACY merges
the base table of `_gap3_learned_feature_compat_derisk.py:27-32`), the tables `comprehension_production_organ` imports
(:74-78). A learned-lexicon fact needs the corpus, so a step-0 shard without `corpus_present` true is UNDEFINED for
every branch check (2.10).

| teach | branch (expected) | consequence |
|---|---|---|
| 'the heron hunts the frog', 'the frog eats the gnat' (every table, and the `infer_dup` re-tells) | `pass_through`: frog is in hand ANIMACY and hunt in no verb table; eat is in hand VERB_SELECTS and gnat in no animacy table (the permuted 'the heron hunts the gnat' was judged fully OOV, so neither heron, hunt nor gnat is covered) | stored unjudged |
| 'the heron hunts the worm' (`infer_amb1`, round 5) | `pass_through`: worm is in hand ANIMACY (`_gap3_learned_feature_compat_derisk.py:29`), so it is not OOV, and hunt is in no verb table, so the teach is not fully covered | stored unjudged |
| 'the frog eats the apple' (`infer_amb2`) | `covered`: frog and apple in hand ANIMACY, eat in hand VERB_SELECTS | judged by the SEMANTIC margin (recorded), but D4 cannot refuse it: when it arrives, (frog, eat) -> gnat is already held, so `known_binding` is True, and the server abstains only on a teach that is neither comprehended nor known (`webapp/server.py:6852-6875`). It is stored whatever its margin. No comprehension outcome is predicted: the 10-26 diagnosis records 'the goat eats the hay' refused on the covered branch at margin 0.0167 against 0.2. Whether it is APPENDED (the ambiguity) or rewritten in place is D2's decision (below) |

The previous draft's 'the frog eats the fly' was claimed to pass because it "contains a known word". That was false:
with fly covered by the learned animacy lexicon it is fully covered and judged. 'apple' is a hand-table noun, so the
branch does not depend on the learned lexicon. Round 3's rationale ("stored only if comprehended; comprehension
expected; STOP if refused") was also wrong, for the known-binding reason in the row; it is withdrawn, and step 0
records `known_binding` instead.

**D2 surprise and reconsolidation (both default ON) on these teaches.** A declarative that asserts a patient under an
(agent, action) the brain already holds runs D2's spiking mismatch read (`webapp/server.py:6933-7013`). A firing
surprise on a declarative opens reconsolidation, which rewrites the stored patient IN PLACE
(`reconsolidation_production_organ._inplace_rewrite` :111) instead of appending a second row. That would turn 'the
heron hunts the worm' (after frog) and 'the frog eats the apple' (after gnat) into rewrites: one row, no ambiguity,
and B5, B6 and B13 (a) UNDEFINED everywhere. Today they append, for an incidental reason. D2 looks up the expectation
with the SURFACE tokens: `extract_assertion` (`surprise_production_organ.py:193`) keeps 'hunts' and 'eats', and
`what_does` (`brain_conversational_agent.py:772`) is a plain `query_patient`. The store holds the lemma ('hunt', 'eat'),
so no expectation is found and nothing fires (code reading; step 0 reads it back). The `infer_dup` re-tells restate the
held patient, so even a lemma-correct D2 would see no contradiction there. Step 0 reads back `surprise` and
`reconsolidation` on every `infer_amb1`, `infer_amb2` and `infer_dup` teach, and STOPs on any `rewrite` (2.10). The
named fallback, for that STOP or for a later lemma fix in D2: write the second fact of each ambiguity table with a
`composer.store` fixture after the teach block, as `infer_xtier` writes its LTM fact. Claude rules on it at the STOP.

**The `.kb` overwrite fixtures (A17-A20, round 4).** These reuse `apply_kb_scramble`'s write and read-back pattern
(`stage2_exit_gate.py:452`) in a new mode, `kb_overwrite`, with a fixed one-entry plan. **How the mode is entered (round
7).** Their informational entries declare `scramble: "kb_overwrite"`, because the worker enters a scramble mode only
through `arm_scramble` reading that key (`stage2_exit_gate.py:364-369`; `scr_mode` at :686, the `_scramble_now` call at
:692-693). So A17-A20 count as arms carrying a scramble for the test amendment below. The same key also installs `KbProbe`
on them (:688-689), which forces nothing outside mode `kb_spiking` (:524), and routes their shards through
`kb_scramble_read` (:1538). After the teach block and
before the first probe, the worker overwrites the patient string of ONE `.kb` entry with a fresh `(fact, handle)`
tuple, as `update_on_mismatch` does. On M1 arms that write goes through the counting proxy's `__setitem__`, which wraps
the new fact dict (2.4, round 6). The bound vectors, the drives and the sequencer are not written, and on A19 and A20
`chat.stored_facts` is not written either. **A17 and A18 also rewrite the chat's lists (round 6).** After
`stored_facts_after_teach` is read (2.7), the fixture overwrites the `.kb` entry, then calls `chat._refresh_facts()`
(`brain_chat_tui.py:987-1016`; at the default, with the D6 engram flags off, it rebuilds `chat.stored_facts`,
`agents_set`, `actions_set` and `patients_set` from `.kb`), and sets the buffer's `_fact_shard_built_K` to -1, so the
fact-shard index is rebuilt from the overwritten list at its next use (`one_brain_composer.py:1410-1432`). It reads
back that worm is in no triple of `chat.stored_facts` and not in `patients_set`, and that `_fact_shard_built_K` is -1
before the first probe. Both arms of the pair get the same fixture, so they differ only in the flag. Each fixture
records the index of the turn after which it ran, which B13 (a) compares with `vp_formed_turn` (2.7). The engagement
record reads the list back. It counts as engaged only if (i) exactly the planned entry
changed, (ii) the decoded rows under the cue (`_read_blocks`, the bound store) are identical before and after, and
(iii) the list oracle (`list_chain` over the overwritten list) predicts a different p_prey_eat outcome from the same
oracle over the decoded rows. (iv) (round 5) The intervention is the one the plan names. For a removal (A17, A18),
the removed string is in no field of any entry of the buffer's `.kb` after the overwrite (`_kb_triples(buf)`,
`stage2_exit_gate.py:448`, read back over all K rows, the built-ins included), while the decoded rows under the cue
still hold it, and (round 6) worm is also absent from `chat.stored_facts` and `patients_set` after the refresh, with
`_fact_shard_built_K` read back as -1. For an addition (A19, A20), the added string is the patient of the FIRST `.kb` entry under the cue,
so a first-match list read returns it. It is also in no decoded row under that cue (the bound store still holds the
original patient in every matching block). Plans:
- A17 and A18 (`infer_amb1`): the entry t_heron_hunt_worm wrote, patient worm -> frog. The list now holds one patient
  under (heron, hunt), and worm is in no `.kb` entry at all; the store holds two. Round 4's fish could not satisfy
  (iv), because the built-in (cat, eat, fish) keeps fish in the list.
- A19 and A20 (`infer_dup`, round 5): the FIRST (heron, hunt) entry, the one t_heron_hunt wrote, patient frog -> worm.
  Round 4 overwrote the re-tell's entry, which left the first match unchanged. The list now holds worm first and frog
  second under (heron, hunt), so the fact shard's first candidate block, whose `.kb` patient `_finish_query_patient`
  reads for the clause test (:1822), now carries worm in the list. The store still holds frog in both blocks.
The sequencer fabric is built lazily at the first conflict read, which comes after the overwrite. So the build reads
a list that disagrees with the store: that is the case under test (2.6). Round 6: this is recorded, not assumed. The
hop record's `vp_formed_turn` must be a turn after the fixture's turn index, and with the drive-dirty rule of 2.3 a
later store write would re-form V'_P at the next read.

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
| A8b amb1_m1_b (round 5) | infer_amb1 | M1 on, independent rebuild | the null for A8 (B6) |
| A9 amb2_off / A10 amb2_m1 | infer_amb2 | off / M1 on | hop-2 ambiguity, both routes |
| A11 amb1_conflict_lesion (+ .rep0) | infer_amb1 | M1 on + conflict lesion | the conflict unit necessary |
| A12 m1_permuted_gateoff | infer permuted | M1 on + `BRAIN_COMPREHENSION_GATE=0` | conclusion tracks exposure through spiking hops (departure, reference) |
| A13 dup_off / A14 dup_m1 | infer_dup | off / M1 on | a duplicated fact is not a conflict, both routes |
| A15 xtier_off / A16 xtier_m1 | infer_xtier | off / M1 on; both with `BRAIN_SHARED_LTM=0` and `BRAIN_LTM_RNG_ISOLATE=1` (offiso, round 8: a private `ShardedPhasorStore` LTM the fixture can write) | a cross-tier conflict still abstains, both routes |
| A17 amb1_kbset_m1 / A18 amb1_kbset_off | infer_amb1 + overwrite worm -> frog | M1 on / off | a patient taken out of the list (the data-flow arm for V'_P): M1's abstain follows the store, today's route the list (B13 a) |
| A19 dup_kbset_m1 / A20 dup_kbset_off | infer_dup + overwrite of the FIRST (heron, hunt) entry, frog -> worm | M1 on / off | the hop-1 value a first-match list reader sees, and a patient added to the list: M1 still advances through frog, today's route abstains (B13 b, route-level list reads only) |
| A21 m1_fs0 (round 5) | infer | M1 on + `BRAIN_FACT_SHARD_RETRIEVAL=0` | PROPOSE on the full decoded scan (`propose_path` `seq_block`, the batched layout): the moat read through the decoded store and the match pools (B8 (1b)) |

**One term, pinned here (round 9).** An **M1 arm** is an arm that runs with `BRAIN_CHAIN_SPIKING_HOPS` on: the "M1
on" rows of the table above, that is A1-A6, A8, A8b, A10, A11, A12, A14, A16, A17, A19 and A21, with the `.rep0` arms
of A4, A5 and A11. A **flag-off arm** runs without it: A0, A3a, A3b, A7, A9, A13, A15, A18 and A20. "Every arm of
2.8" names both sets. Only an M1 arm enters `_hop_spiking`, so only an M1 arm makes hop records, spiking hop reads and
the per-turn `store_snapshot` (2.7). Every use of "M1 arm" in this packet's current text means the first set. A rule
that also binds flag-off shards names its scope as every arm of 2.8 (2.9's general rule, section 5); round 5's item 9
is history, and its "every M1 arm" meant every arm of 2.8.

New arms are added as informational entries, so every pre-existing arm keeps its turn table, env, patch and kind,
pinned by the golden hash in `tests/test_stage2_kb_scramble.py`; `pool_job_lines` grows by the new arms only. A3a and
A3b are the existing arms, re-run at the step-2 revision, not new code. **The test amendment this needs (round 6).**
Two assertions of that test break as soon as any arm here is added, and they are amended by name, in the same
commit that adds the arms. `test_every_pre_existing_arm_has_its_frozen_turn_table_and_spec` asserts that `infer`'s
only arms outside GOLDEN are the two kb arms and that no other group has any (`tests/test_stage2_kb_scramble.py:53-55`).
`test_kb_arms_are_informational_never_counted_and_run_the_intact_turn_table` asserts that no arm other than the two kb
arms carries a scramble (:65-68). A0-A21, A8b, the `.rep0` arms of A4, A5 and A11, and A3 (scramble `kb`) break the
first, and A3 and A17-A20 (scramble `kb_overwrite`, round 7) break the second. The amendment replaces each expected value with an explicit allowlist: `infer`'s
extras equal the two kb arms plus this packet's named arms (and, in M3 step 3, the M3 arms of 3.6 that run an `infer`
table); `ask`'s extras are empty until M3 step 3 adds `m3_ask`, R-goat's arm (3.6; round 9), and then equal
{`m3_ask`}; every other group's extras stay empty, and if a table of this packet is ever implemented as its own group, that group's arms are named
in the allowlist too; the arms carrying a scramble are the two kb arms (modes `kb` and `kb_spiking`, as now), A3 (mode `kb`) and A17-A20 (mode
`kb_overwrite`, round 7), written as an explicit arm-to-mode map that the test asserts equal to `arm_scramble` over every
group and arm. The GOLDEN dict and
the per-arm fingerprint assertion (:48-52) are not changed, so every pre-existing arm stays pinned. Step 0 lands the
amendment for the arms it adds, and step 1 extends the allowlist (2.10).

### 2.9 Registered predictions and bars (dev read; numbers are per dev seed, 1-6)

Predictions, written before any M1 code exists:
- P1 (A1): p_prey_eat derived_svo `[frog, eat, gnat]`, derived_from `[[heron, hunt, frog], [frog, eat, gnat]]`,
  the "I derived this from" lead present. Same as A0 on these fields.
- P2 (A3): engagement as in A3a's record (3 scoped facts changed, read-back equals plan, patients distinct, the
  conclusion fact changed), and A3's derived_svo/derived_from equal A3b's on the same seed and revision (the
  bound-vector conclusion, `[frog, eat, gnat]`) and differ from A3a's (the list conclusion, `[frog, eat, wren]`). The
  answer text keeps list-read sentences from other readers, and c_frog_eat and p_yesno stay flagged guesses (the
  direct-recall verify reads `chat.stored_facts`, outside M1; reported, not barred).
- P3 (A4): p_prey_eat has no derived answer, reproduced on .rep0.
- P4 (A5): p_prey_eat has no derived answer and its hop-1 record shows `n_ignited` = 0 (no slot ignites on the
  lesioned workspace), reproduced on .rep0.
- P5 (A7-A10, A8b): p_prey_eat abstains on both routes in both ambiguity tables; in A8, A8b and A10 the abstaining hop
  shows blocks_matched = 2, patient_lines_active = 2, acc_conf_spikes > 0 and gate_decision ABSTAIN (hop 1 under
  (heron, hunt) in A8 and A8b, hop 2 under (frog, eat) in A10), with `abstain_regime` `ignition` (n_ignited >= 2 at
  `gate_cycles` = R_max) and `xtier_slots` = [] (round 6).
- P6 (A11): p_prey_eat is derived through the FIRST-taught patient (frog), reproduced on .rep0.
- P7 (A13, A14): p_prey_eat derived_svo `[frog, eat, gnat]` on both routes; in A14 both hops show blocks_matched = 2,
  patient_lines_active = 1, acc_conf_spikes = 0 and gate_decision ADVANCE.
- P8 (A12, reported only, never a bar): with t_heron_hunt acquired on that seed (`teach_acquired`; else the seed is
  UNDEFINED for P8), p_prey_eat is derived `[gnat, eat, frog]` from `[[heron, hunt, gnat], [gnat, eat, frog]]`
  through two recorded spiking hops, as the gate-off diagnostic arm derived it through the list.
- P9 (A15, A16): p_prey_eat abstains on both routes; in A16 hop 1 shows source_tier buffer, proposed frog,
  patient_lines_active = 1, acc_conf_spikes = 0, xtier_slots = [fish], n_ignited >= 2 and gate_decision ABSTAIN.
  Round 8: both arms run offiso, and each arm's fixture record shows `ltm_type` `ShardedPhasorStore` and no
  `store_exception` (2.8); otherwise that arm is UNDEFINED for P9.
- P10 (A17, A18; a patient taken out of the list, round 5 with worm): A18 (flag off) derives p_prey_eat
  `[frog, eat, gnat]` from `[[heron, hunt, frog], [frog, eat, gnat]]`, because the list now holds one patient under
  (heron, hunt). A17 (M1 on) abstains on p_prey_eat; its hop-1 record shows worm in `vp` AND in `vp_not_in_kb`,
  blocks_matched = 2, patient_lines_active = 2, acc_conf_spikes > 0, spurious_P = 0 and gate_decision ABSTAIN, with
  `abstain_regime` `ignition` (n_ignited >= 2 at `gate_cycles` = R_max), `xtier_slots` = [] and `vp_formed_turn`
  after the overwrite (round 6).
- P11 (A19, A20; the first entry's patient value changed, round 5): A20 (flag off) abstains on p_prey_eat, because the
  list now holds worm and frog under (heron, hunt). A19 (M1 on) derives `[frog, eat, gnat]` from
  `[[heron, hunt, frog], [frog, eat, gnat]]` (worm named nowhere in the answer or `derived_from`); its hop-1 record
  shows `proposed` frog, blocks_matched = 2, patient_lines_active = 1, acc_conf_spikes = 0 and gate_decision ADVANCE.
- P12 (A21, round 5): p_prey_eat is derived as on A1. On c_prey_chase, the hop-2 record under (frog, chase) shows
  `propose_path` `seq_block`, empty `decoded_rows`, blocks_matched = 0, acc_conf_spikes = 0, an all-zero `drives`,
  n_ignited = 0 and gate_decision COMMIT, and c_prey_chase abstains.
- P13 (A8b, round 5): A8b's primary fields equal A8's on every probe turn (the null of B6).

General engagement rule (round 4; (i) and (vi) tightened in round 5; (vii) added in round 6; round 9: the scope of each
part stated, with the term of 2.8). Rules (i) and (iv) bind every arm of 2.8 that a bar reads, flag on or off: B9's
flag-off cells, B8's A0 controls and the A18 and A20 teeth of B13 read flag-off shards, which need the corpus, the
source and the bundle as much as an M1 arm does. Rules (ii), (iii) and (v) read records that only an M1 arm makes (hop
records, spiking hop reads), so they bind M1 arms. (i) The shard records
`corpus_present` true (`run_worker` :671); without the corpus the learned production organs degrade silently, so a
corpus-absent shard is UNDEFINED for every bar it feeds. A shard with NO `corpus_present` key is UNDEFINED too: the M1
scorer never uses `arm_validity`'s default, `shard.get("corpus_present", True)` (`stage2_exit_gate.py:933`), which would
read a missing key as present. The shard also records `sources` equal to {`tiny-demo +LTM`} (EXPECTED_SOURCE,
`stage2_exit_gate.py:71`) and `process.ltm_bundle` equal to `wikidata_100k` (`process_record`, :623-638). The LTM
bundle lives outside the repository (`_default_ltm_bundle_dir`, `webapp/server.py:3817-3847`, which looks in
`$BRAIN_DATA_ROOT` and two `sim-data` paths), and its fallback `wikidata_core_15k` yields the same `source` string, so
the bundle name, not the source string, shows which LTM the shard read. A missing or different value makes the shard
UNDEFINED for every bar it feeds. (ii) The arm's hop records show `ws_seed` equal to the arm seed. A seed failing
(i) or (ii) is UNDEFINED for every M1 bar on that arm. (iii) A flag-on turn that a bar reads as the route's outcome
counts as engaged only through its route trace: `spiking_hop_reads` >= 1 on that turn (2.7), meaning a hop scope opened, made its
workspace read and closed without an exception. The flag's env value is never the evidence. A turn with no recorded
spiking hop read is UNDEFINED for that bar, never PASS and never FAIL. (iv) The probe's premises are present (the
facts-present rule below). (v) The hop-record preconditions of the table below hold, and every hop a bar reads passes
2.7's positive control on the proxy.

(vi) **Cross-seed rule (round 5; one rule for every M1 bar of this section and every R bar of 3.7).** A bar HOLDS only
if it PASSES on every one of the six dev seeds 1-6. A seed on which the bar is UNDEFINED or UNMEASURED counts as not
meeting it, exactly as a FAIL does. n_pass / n_defined and n_effective (below) are reported beside the verdict and
never replace it. So one defined seed of six can never carry a bar, a bar some of whose seeds are UNDEFINED does not
hold, and step 3's "every bar holds" (2.10) means that every bar passed on all six seeds. No bar has an "UNDEFINED only
if every seed is UNDEFINED" exception any more (B11 had one in round 4).

(vii) **Within-seed rule (round 6; one rule for every M1 bar of this section and every R bar of 3.7).** Any UNDEFINED
or UNMEASURED item of a bar (an arm, a seed's reference arm, a probe, a turn or a hop) makes that bar UNDEFINED on
that seed, unless a defined item of the same bar FAILs there, which makes it FAIL. A bar PASSES on a seed only when
every one of its items is defined there and passes. The ask group's A4 (a) rule, which drops undefined items and
scores the rest (`score_ask`, `stage2_exit_gate.py:1144-1154`, with `ASK_MOAT_MIN_DEFINED` = 1 at :85, so one defined
moat of three carries the rate), is NOT used by any bar of this packet. Wherever an earlier round wrote "as in A4 (a)",
"scored like A4 (a)" or "excluded", this rule holds instead. Named, because their items are several per seed: B9 (one
item per label and per arm pair; an UNDEFINED or UNMEASURED pair or label makes B9 UNDEFINED or UNMEASURED on that
seed); B8 (items (1), (1b), (2), (3) and (4); B8 passes on a seed only when all five pass, and an UNDEFINED control
among (2)-(4) makes B8 UNDEFINED there); and B12 (b) (one item per listed arm and per hop; one UNDEFINED arm or hop
makes B12 (b) UNDEFINED on that seed). Under rule (vi) an UNDEFINED seed then counts as not meeting the bar.

**Facts present (round 4; an acquisition failure is not an inference failure).** Before a probe is scored on an arm
and seed, the facts its chain needs are read from the stores the brain reads, and the probe is UNDEFINED for every bar
that reads it unless they are there. The idea comes from the ask group's A4 (a), where a taught moat whose teach was
not acquired is UNDEFINED; A4 (a)'s scoring, which then scores the rest, does not (rule (vii)): an UNDEFINED probe makes
every bar that reads it UNDEFINED on that seed. A premise is present when (1) its teach turn acquired it (`teach_acquired`,
A4 (a)'s read on the teach turn's own reply) and it is in `stored_facts_after_teach` (2.7); or (2) for a built-in
premise, it is in the shard's `builtin_facts`. That is the chat's stored facts, which `run_worker` latches once after
turn 0 (`stage2_exit_gate.py:710-715`): the five built-ins of `_build_tiny_demo` (`brain_chat_tui.py:2449-2455`) plus
turn 0's own fact, t_heron_hunt's (heron, hunt, frog) on these tables (round 5 correction: round 4 called it "the
tiny-demo store"). Or (3), for the `infer_xtier` fixture, the fixture's second read-back, right before the first
probe, returned it, on a fixture record whose `ltm_type` is `ShardedPhasorStore` (round 8, 2.8). "Exactly" below
means that the patients stored under the cue in these records are the listed set, with no extra one. A store holding
another patient under a hop cue makes the probe test a different chain, so it is UNDEFINED too. On M1 arms the
bound-store side is added for every cue the map lists with a non-empty patient set: the decoded rows under the cue,
and the LTM count under it (0 except for `infer_xtier`'s (heron, hunt), which is 1). **Round 8: both are read from
the probe turn's `store_snapshot`, never from the hop records.** The hop records cannot carry them, because many cues
are never hopped: a non-chain probe returns before any hop (`compositional_chain_route.py:274-276`), a hop-1 ABSTAIN
ends the chain (:289-291), and a hop-1 miss moves on to the next relation without a hop 2 (:292-293). On every probe
turn of every M1 arm (every turn whose role is not `teach`), chain or not, the arm worker therefore records the FULL
decoded store once, right after `brain_chat` returns the reply and before the next turn: `store_snapshot = {turn, K,
rows, read, ltm_under_cue}`. `rows` holds all K decoded rows, each as (block id, agent, action, patient), none
dropped or filtered. They come from one `_read_blocks` call, the same call 2.7's post-decision read makes, which
decodes every block: the per-block loop (`one_brain_composer.py:1389`) at the production layout, and
`_read_all_blocks` (`len` at :1206) on A21; `read` names which ran. `ltm_under_cue` holds, for each cue of the arm's
map rows, the patients of the record-only scan of every shard in `ltm.shards` that `audit_ltm_all` makes (2.3), a list
read with no substrate run. A map cue's decoded rows are the snapshot rows whose decoded agent and action equal the
cue, and its LTM count is the length of `ltm_under_cue` under it. On an M1 arm (flag on, the term as pinned in 2.8), a
probe turn with no `store_snapshot`, or with a snapshot whose `rows` count differs from its K, is UNDEFINED for every
bar that reads that probe, and it is listed. A flag-off arm (A0, A3a, A3b, A7, A9, A13, A15, A18 and A20) records no
snapshot by design (2.7), so a missing snapshot there never makes a probe UNDEFINED (round 9): its probes are scored
on sources (1)-(3) above only, as B9's flag-off cells and B8 (controls) on A0 are.
The snapshot runs outside every decision scope and every hop scope, so its `.kb` accesses are recorded under their own
label `store_snapshot` and never enter `composer_kb_reads`, `kb_reads_unlisted` or B7. Its inertness is covered by
the step-1 whole-table behaviour-preservation test (2.10), which runs with it installed. The hop records' own
`decoded_rows` stay what B5, B8 and B12 (a) read on the hops they name. These LTM reads, and the A15/A16 fixture
rows, count only on a shard whose `ltm_bundle` is `wikidata_100k` (rule (i)). A disagreement with this map makes the probe UNDEFINED, and
it is listed. An ABSENCE premise (the moat's (frog, chase)) is never read from the bound store or the LTM (round 5; the
c_prey_chase row). Step 0 checks the whole map on the bound store and the LTM (2.10). An UNDEFINED probe goes into the
bar's `probes_undefined` list with the missing or extra fact, and under rule (vii) the bar is UNDEFINED on that seed.
It never masks a FAIL that a defined probe of the same bar decides.

| table (arms) | probe | hop cue -> patients stored, exactly | read from |
|---|---|---|---|
| infer (A0-A6, A21) | p_prey_eat | (heron, hunt) -> {frog}; (frog, eat) -> {gnat} | t_heron_hunt, t_frog_eat |
| infer | c_heron_hunt | (heron, hunt) -> {frog} | t_heron_hunt |
| infer | c_frog_eat | (frog, eat) -> {gnat} | t_frog_eat |
| infer | c_prey_chase (moat) | (heron, hunt) -> {frog}; (frog, chase) -> {} (nothing) | t_heron_hunt; the absence premise is read from the teach log and `builtin_facts` ONLY (round 5): (frog, chase, *) in neither `stored_facts_after_teach` nor `builtin_facts`. The hop-2 record's `decoded_rows` and `ltm_candidates` under (frog, chase) are NOT premises: a non-empty one is a confabulated or extra row, scored by B8 (1): a moat FAIL if it leads to an answer, and item (1) UNDEFINED with the row listed if it does not; it never turns an answer into an UNDEFINED |
| infer | p_yesno (reported only) | (crow, chase) -> {wren}; (wren, chase) -> {moth} | t_crow_chase, t_wren_chase |
| every table | c_dog_chase | (dog, chase) -> {cat} | `builtin_facts` |
| infer_amb1 (A7, A8, A8b, A11, A17, A18) | p_prey_eat, c_heron_hunt | (heron, hunt) -> {frog, worm}; (frog, eat) -> {gnat} | t_heron_hunt, t_heron_hunt_worm, t_frog_eat (on A17 and A18 from the teach log, never from the overwritten `.kb`) |
| infer_amb2 (A9, A10) | p_prey_eat, c_heron_hunt | (heron, hunt) -> {frog}; (frog, eat) -> {gnat, apple} | t_heron_hunt, t_frog_eat, t_frog_eat_apple |
| infer_dup (A13, A14, A19, A20) | as infer | as infer | as infer, plus t_heron_hunt_re and t_frog_eat_re acquired (on A19 and A20 from the teach log) |
| infer_xtier (A15, A16) | p_prey_eat | buffer (heron, hunt) -> {frog}; LTM (heron, hunt) -> {fish}; (frog, eat) -> {gnat} | t_heron_hunt, t_frog_eat, and the fixture read-back |
| infer permuted (A12) | p_prey_eat | (heron, hunt) -> {gnat}; (gnat, eat) -> {frog} | P8's own `teach_acquired` rule (reported only) |

**Hop-record preconditions (round 4; registered, the scorer's map).** For each bar, the records and fields that
must exist before the bar is scored. A field is present when its key exists; it is non-empty when it is a non-null
scalar, or a list or dict with at least one element. The exceptions are named in the table: `proposed` may be None
(a miss), and `ltm_candidates` may be an empty list. An item (an arm, seed and turn) without its required records or
fields is UNDEFINED. It goes into the bar's `undefined_items` list with the missing field, and the bar is UNDEFINED on
that seed (rule (vii); round 6: not excluded and scored around, as A4 (a) does). Scoring follows the 2026-10-13
ruling: a bar is FAIL on a seed when any defined item fails; it is PASS only when every item is defined and passes;
otherwise it is UNDEFINED. An undefined item never masks a FAIL that a defined item decides. The other bars keep the preconditions written in their own text, and the
general rule above applies to all.

Round 5: every hop now carries a `drives` record and a buffer conflict read (2.2), so the B3 and moat rows can be met
by construction; `fs_candidates`, `decoded_rows` and `ltm_candidates` may be empty lists where the table says so.

| bar | arm(s) | turn | records required | fields present and non-empty |
|---|---|---|---|---|
| B1 | A1 | p_prey_eat | round 6: `spiking_hop_reads` >= 1 and the hop-1 record under (heron, hunt); no `route_exception`. The hop-2 record and both ADVANCEs are B1's OUTCOME, not its precondition (a hop-1 ABSTAIN ends the chain after one read) | on every hop record of the turn: `proposed`, `propose_path`, `drives`, `gate_decision`, `n_ignited`, `conf`, `committed`, `abstain_regime` (key present), `composer_kb_reads` (2.7 positive control, entry-level control included where a block was selected) |
| B3 | A4 and its .rep0 | p_prey_eat | `spiking_hop_reads` >= 1; the hop-1 record under (heron, hunt); round 6: the turn's `patch_calls_delta` >= 1 | `proposed` (key present; None expected), `propose_path` (`patched`, i.e. `patch_calls_in_hop` >= 1), `patch_calls_in_hop`, `drives`, `acc_conf_spikes`, `n_ignited`, `gate_decision`, `composer_kb_reads` (:1654 present) |
| B4 | A5 and its .rep0; A1 (the references) | p_prey_eat | `spiking_hop_reads` >= 1; the hop-1 record under (heron, hunt) | `proposed`, `drives`, `ws_bridge_key`, `ws_loop_w_max`, `ws_loop_n_syn`, `n_ignited`, `gate_decision` |
| B5 (round 6) | A8, A8b, A10; A1 (REF-A1) | p_prey_eat | `spiking_hop_reads` >= 1; on A8 and A8b the hop-1 record under (heron, hunt), on A10 the hop-2 record under (frog, eat) (the abstaining hop) | on the abstaining hop: `decoded_rows`, `blocks_matched`, `patient_lines_active`, `acc_conf_spikes`, `xtier_slots` (list, must be empty), `n_ignited`, `gate_cycles`, `conf`, `theta_hi`, `abstain_regime`, `gate_decision` |
| B6 | A8, A8b, A11 and its .rep0 | p_prey_eat | `spiking_hop_reads` >= 1; the hop-1 record under (heron, hunt) | `blocks_matched`, `patient_lines_active`, `acc_conf_spikes`, `slot1_drive`, `cp_n_syn`, `cp_w_max`, `gate_decision` |
| B8 (arm engagement) | A1 | p_prey_eat | as B1, plus round 6: both hop records with `gate_decision` ADVANCE (REF-A1 with hop 2 at ADVANCE, since B1's ADVANCE moved to B1's outcome) | as B1 |
| B8 (moat) | A1 | c_prey_chase | `spiking_hop_reads` >= 2; a hop-1 record under (heron, hunt) with `committed` frog, and at least one hop-2 record under (frog, chase) | hop 2: `proposed` (key present; None expected), `propose_path`, `fs_candidates` (list, may be empty), `decoded_rows` (list, may be empty), `cue_in_reduced_vocab`, `blocks_matched`, `acc_conf_spikes`, `ltm_candidates` (list, may be empty), `drives`, `n_ignited`, `gate_decision` |
| B8 (1b, moat on the decoded store) | A21 | c_prey_chase; round 6: and p_prey_eat (the match-pool liveness) | as B8 (moat); round 6: on the same A21 seed, p_prey_eat's hop-1 record under (heron, hunt) | as B8 (moat), with `propose_path` `seq_block`; on the p_prey_eat hop-1 record: `selected_block`, `blocks_matched`, `patient_lines_active`, `m_counts` |
| B8 (controls) | A1 and A0 | c_heron_hunt, c_frog_eat, c_dog_chase | the per-turn route trace | `route_dispatched` and `spiking_hop_reads` present (false and 0 expected on these non-chain turns; any other value is listed) |
| B11 | A14; A8 (liveness) | p_prey_eat | A14: both hop records; A8: the hop-1 record under (heron, hunt) | A14 on both hops: `blocks_matched`, `patient_lines_active`, `acc_conf_spikes`, `gate_decision`; A8: `patient_lines_active`, `acc_conf_spikes`, `gate_decision` |
| B12 (b) | A1, A2, A4-A6, A8, A8b, A10, A12, A14, A16, A21 | every chain probe of the arm's table (p_prey_eat; c_prey_chase where the table has it) | `spiking_hop_reads` >= 1 on each such probe; round 5: on A8 and A8b the hop-1 record under (heron, hunt), and on A10 the hop-2 record under (frog, eat) (the hop that carries the ambiguity; a hop-1 record alone does not satisfy it) | on every hop: `audit_buffer`, `audit_ltm`, `audit_ltm_all` (integers), `gate_decision`, `audit_disagree`, `xtier_disagree` (booleans) |
| B13 (a) | A17 | p_prey_eat | `spiking_hop_reads` >= 1; the hop-1 record under (heron, hunt); round 6: the fixture's turn index and its chat-list and `_fact_shard_built_K` read-backs | `vp`, `vp_not_in_kb`, `vp_formed_turn`, `vp_formed_K`, `blocks_matched`, `patient_lines_active`, `acc_conf_spikes`, `spurious_P`, `xtier_slots` (list, must be empty), `n_ignited`, `gate_cycles`, `conf`, `theta_hi`, `abstain_regime`, `gate_decision` |
| B13 (b) | A19; A8 (liveness) | p_prey_eat | `spiking_hop_reads` >= 2; the hop-1 record under (heron, hunt); A8 as in B11 | `proposed`, `selected_block`, `blocks_matched`, `patient_lines_active`, `acc_conf_spikes`, `gate_decision`, `kb_patient_reads` (round 6: holding the clause-test read of the selected block id, 2.4's entry-level control) |

Bars (each scored on every dev seed 1-6, and each holding only under the cross-seed rule (vi); design seed 7 is
excluded from the count). Two same-seed references recur below (round 5, `docs/TERMS.md` lesion and load-bearing). The
first is **REF-A1**. On that seed A1 derives p_prey_eat `[frog, eat, gnat]` (P1), with hop 1 at gate_decision ADVANCE
and n_ignited = 1 and hop 2 under (frog, eat) at gate_decision ADVANCE, and A2's primary fields on p_prey_eat equal
A1's (the null). Round 6: hop 2's ADVANCE is written out here because B1's ADVANCE is now B1's outcome, and REF-A1
alone carries it for B3, B4, B5, B8, B12 and B13. The second is **LIVE-A8**. On that
seed A8's hop 1 under (heron, hunt) shows patient_lines_active = 2, acc_conf_spikes > 0 and gate_decision ABSTAIN, so
the conflict unit can fire on that seed and a silent `acc_conf` elsewhere is evidence.
- **B1 conclusion (round 6: the precondition no longer contains the outcome).** Engagement precondition: the B1 row
  of the hop-record table holds (A1's p_prey_eat turn records `spiking_hop_reads` >= 1, the hop-1 record under (heron,
  hunt) exists and no route exception was recorded), and every hop record of the turn passes 2.7's positive control,
  the entry-level control included. With no hop record the old `_hop` may have answered, so the seed is UNDEFINED for
  B1. Outcome: P1 holds AND hop 1 under (heron, hunt) and hop 2 under (frog, eat) are both recorded with
  gate_decision ADVANCE (so the turn has at least two spiking hop reads). A hop-1 or hop-2 ABSTAIN or COMMIT is a
  FAIL, whatever the answer text says. A hop-1 ADVANCE with no hop-2 record and no recorded exception is an
  instrument fault: it is listed and the seed is UNDEFINED. Why the read-count precondition is >= 1 and not the
  synthesis's >= 2: an ambiguous hop 1 abstains the whole chain at once (`compositional_chain_route.py:289-291`, the
  control flow `_hop_spiking` keeps, 2.2), so a hop-1 ABSTAIN leaves one read on the turn, and a >= 2 precondition
  would turn that FAIL into an UNDEFINED. A hop-1 COMMIT moves on to the next candidate relation, so two or more reads
  without an ADVANCE pair are possible, and they FAIL as well.
- **B2 ownership:** P2's two fields hold; the bar reads derived_svo and derived_from only. It is scored only against
  A3a and A3b run at the step-2 revision; until they exist B2 is UNDEFINED. A seed where A3a does not follow the list
  (the 10-26 result does not reproduce) is UNDEFINED for B2, and the non-reproduction is reported first. A3's
  p_prey_eat must carry both hop records (the flag engaged under the scramble), else UNDEFINED.
- **B3 PROPOSE necessary:** P3 holds and reproduces. Engagement preconditions, on A4 and on its .rep0: the B3 row of
  the hop-record table holds. Hop 1 of p_prey_eat was entered and recorded `proposed` None with `drives[0]` = 0, and
  the lesion held inside the hop at the read. Round 6 restates that last clause in measurable terms: the turn's
  `patch_calls_delta` is >= 1, and the hop-1 record's `propose_path` is `patched` (`patch_calls_in_hop` >= 1). The
  shard's `patch.calls` is one session-wide total and cannot show a turn's calls (2.7). Round 5: REF-A1
  holds on the same seed, so the intact PROPOSE answers and A2 = A1. A missing hop record, a route exception or a
  failed REF-A1 makes the seed UNDEFINED (no derived answer for another reason is not the lesion's effect, and a
  "necessary" read needs an intact reference that works and reproduces).
- **B4 workspace necessary (round 4).** The manipulation is the ENGAGED spiking hop read on a lesioned workspace, not
  the flag. Engagement precondition, on A5 and on its .rep0, both read from the route trace: (i) p_prey_eat's turn
  records `spiking_hop_reads` >= 1 and the B4 row of the hop-record table holds; (ii) the input reached the lesioned
  stage: hop 1 under (heron, hunt) recorded `proposed` = frog and `drives[0]` = IGNITE_PA; (iii) the lesion held:
  `ws_bridge_key` = (arm seed, True) and `ws_loop_w_max` = 0, read back from the bridge at the read. Round 5:
  `ws_loop_w_max` is recorded as 0.0 when no synapse is found, so a zero alone does not show a held lesion. (iii) also
  needs A1's hop 1 on the same seed to show `ws_loop_w_max` > 0, and A5's `ws_loop_n_syn` to equal A1's and be > 0:
  the same loop synapses exist, at zero weight. (iv) REF-A1 holds on the same seed. A turn with no recorded spiking
  read, or a failure of (ii), (iii) or (iv), is UNDEFINED, never PASS and never FAIL. Then the outcome,
  which is no longer part of the precondition: B4 PASSES when p_prey_eat has no derived answer AND hop 1 has
  `n_ignited` = 0, on A5 and on .rep0. It FAILS when the preconditions hold and either run has a derived answer or
  hop 1 has `n_ignited` >= 1 (a held lesion that leaves slot 0 ignited is a FAIL of "workspace necessary").
- **B5 multi-valued abstain:** P5 holds on both routes. Engagement preconditions: (i) REF-A1 holds on that seed (round
  6: REF-A1, which now carries both of A1's ADVANCEs, replaces "B1's engagement"; an M1 that never advances abstains
  on every cue, so its abstain on an ambiguous one says nothing); (ii) two decoded rows with two different decoded
  patients under the ambiguous cue (a REWRITE makes the seed UNDEFINED, never PASS); (iii) in A8, A8b and A10 the
  abstain comes from the abstaining hop's `gate_decision` ABSTAIN, not from a miss or an exception; (iv) round 6: that
  ABSTAIN is the ignition regime with no cross-tier slot, meaning `abstain_regime` `ignition`, `n_ignited` >= 2 at
  `gate_cycles` = R_max, and `xtier_slots` == []. A graded or winner abstain, or a cross-tier slot on the hop, makes
  that item UNDEFINED (and B5 with it, rule (vii)): the abstain may then come from sub-ignition NMDA or from the LTM
  rather than from the conflict unit. The B5 row of the hop-record table holds. Otherwise UNDEFINED.
- **B6 conflict unit necessary:** P6 holds and reproduces. Engagement precondition, on the same seed and the same hop
  (hop 1 of p_prey_eat): A8 shows blocks_matched = 2, patient_lines_active = 2 and acc_conf_spikes > 0 (the unit
  sees the conflict when intact), and A11 and its .rep0 show blocks_matched = 2, patient_lines_active = 2,
  acc_conf_spikes == 0 and slot1_drive == 0. Round 5 adds four conditions. First, the intact reference works on the
  same seed: A8's p_prey_eat abstains, and the abstain comes from hop 1's gate_decision ABSTAIN, not from a miss or an
  exception. Second, the null: A8b's primary fields equal A8's (P13). Third, the lesion is read back on the synapses:
  A11's and its .rep0's `cp_w_max` = 0 with `cp_n_syn` equal to A8's and > 0, so the same conflict synapses exist
  at zero weight. Fourth, the B6 row of the hop-record table holds. Otherwise the seed is UNDEFINED for B6, never
  PASS. Round 6: the third condition is reachable only because the lesion now writes the built weights to exactly 0.0
  after every build (2.3, 2.5). A `RegionPathway` built at `w_c` = 0 holds 0.01 on every synapse (`sim/regions.py:1009`),
  so round 5's B6 could never be met; the step-1 unit test (2.10) shows `cp_w_max` == 0.0 before any data.
- **B7 hop scope:** both hops recorded on p_prey_eat in A1; `route_kb_reads` = 0 inside every hop scope;
  `query_patient` calls inside hop scopes = hops attempted; the PROPOSE word is the word of the single active patient
  line on every resolved hop whose buffer PROPOSE selected a block, on every M1 arm except A11 and its .rep0 (round 6:
  the wider of round 5's two readings, stated; A11's lesion lets a two-line hop resolve by design, which is B6's
  evidence, as B12 (b) also excludes it); `stp_reset` reads x = 1 and u = cfg.stp_U on every read. **Containment (round 6:
  scope stated).** On every resolved hop of every M1 arm whose buffer PROPOSE selected a block (`selected_block` not
  None; A4's patched hops and LTM-served hops select none), the selected block lies in `blocks_matched`, A11 and A21
  included: the conflict lesion leaves the match pools intact, and on A21, where PROPOSE is a host first-match
  independent of the match pools (2.1), this is the check that the match pools agree with the host's choice. Round 4,
  on every hop of every M1 arm: `spurious_P` = 0 (a lit patient of a matching block outside V'_P means the fabric and
  the read disagree); `vp_kb_entry_reads` = 0 (no patient-field read from `_ensure_sequencer`, the V'_P builder or the
  conflict read's entry point, at any site, 2.4); and `vp` equals the union of `vp_lit_by_block`, in block order.
  Round 5, on every turn and every hop of every M1 arm: (c) `kb_reads_unlisted` = 0 over the whole decision scope, from
  the route's entry to its returned decision, every caller in every module (2.4, 2.7), so the only `.kb` accesses are
  the listed composer sites (round 6: keyed by function, statement and kind, with the conflict read's own `len` site
  added) and the three declared label-only reads; (d) every hop passes 2.7's positive control (the proxy was live),
  else that hop is UNDEFINED, not passed. Round 6: (c)'s patient-field clause and `vp_kb_entry_reads` are scored on a
  hop only when that hop also passes the entry-level control (2.4: the clause-test read of the selected block id is
  listed in `kb_patient_reads`), and a hop that fails it is UNDEFINED for them. A turn with `kb_unwrapped` > 0 is
  UNDEFINED for both as well. (e) no hop is `host_routed`. Round 6: `host_routed` is defined only for `propose_path`
  `fact_shard` or `escalated` (2.7), so A4's `patched` hops carry no such flag (round 5's A4 exception is no longer
  needed), and A6's misses of the spiking sequencer are `seq_miss`, listed and reported, never barred. A hop that
  breaks one of these fails B7 on that seed and is listed. **What B7 is not (round 5).** Its
  V'_P checks (`vp` equal to the union of `vp_lit_by_block`, and no builder patient-field read) are consistency and
  frame checks. They pass whenever the list equals the store, which it does on every arm without an overwrite, so they
  are not evidence that V'_P comes from the store. That evidence is the data-flow intervention B13 (a), on A17, where a
  patient string is in no `.kb` entry while the store holds it, together with (c).
- **B8 moat and controls (round 4; item (1) restated and (1b) added in round 5).** Preconditions: the B8 rows of the
  hop-record table (the arm engaged on p_prey_eat: B1's precondition plus, round 6, REF-A1 with both hops at ADVANCE,
  since B1's ADVANCE is now B1's outcome; a hop-2 record on c_prey_chase), the general rule, and the facts-present map
  for each item. c_prey_chase's absence premise is read from the teach log and `builtin_facts` only. Items, each
  recorded on its own and combined by rule (vii) (round 6): B8 passes on a seed only when (1), (1b), (2), (3) and (4)
  all pass there, an UNDEFINED item makes B8 UNDEFINED there, and a defined FAIL makes it FAIL:
  (1) **No-regression moat on A1.** c_prey_chase abstains on A1 (A0's c_prey_chase is reported beside it). Every
  hop-2 record under (frog, chase) shows `proposed` None, an all-zero `drives`, `n_ignited` 0 and `gate_decision` COMMIT. A hop-2
  ADVANCE, an abstain from a gate ABSTAIN or a slot-1 winner, or a derived or recalled answer on c_prey_chase FAILS.
  A non-empty `decoded_rows` or `ltm_candidates` under (frog, chase) is a row nothing taught. If it leads to an answer
  on c_prey_chase, that is a moat FAIL, never UNDEFINED. If it does not, it is listed with its source and item (1) is
  UNDEFINED on that seed, as round 4 scored it (the cue did not hold the registered nothing, so the abstain was not
  tested on it); the same hop is also `host_routed` when the buffer missed it, and fails B7 (e). This item does
  NOT claim that the miss is spiking. At the default the hop-2 `propose_path` is `fact_shard`, and the fact-shard
  candidate set (a host index over the list's strings) decides which blocks are ever decoded, so the buffer miss is a
  host-index decision (2.1, 2.6).
  (1b) **The moat read through the decoded store, on A21** (`BRAIN_FACT_SHARD_RETRIEVAL=0`). c_prey_chase abstains, and
  every hop-2 record under (frog, chase) shows `propose_path` `seq_block`, an empty `decoded_rows`, blocks_matched = 0,
  acc_conf_spikes = 0, an all-zero `drives`, n_ignited = 0 and gate_decision COMMIT (P12). That is the claim round 4
  made under (1): the miss is read off the full decoded store and the conflict unit's match pools. It is still a host
  first-match over decoded rows (2.1, declared) and no more. **Match-pool liveness on A21's own layout (round 6).** On
  A21 PROPOSE is `_seq_block`'s host first-match over the decoded rows (`one_brain_composer.py:1685-1691`), which never
  reads the match pools; and A21 runs with `no_batched_region` False and `enable_batched` True (:371-372, :480-482), a
  different read layout from the fact-shard arms on which LIVE-A8 is measured. So `blocks_matched` = 0 under (frog,
  chase) would also be read off match pools that never fire. (1b) therefore also requires, on the same A21 seed,
  p_prey_eat's hop-1 record under (heron, hunt) to show `blocks_matched` = 1, containing its `selected_block`, and
  `patient_lines_active` = 1. Without that record, or with other values, (1b) is UNDEFINED on that seed, never PASS.
  (2) c_dog_chase recalls cat. (3) and (4): the primary fields of c_heron_hunt and c_frog_eat equal A0's. These
  controls are not chain turns, so their identity is read with the arm-level engagement, never with a per-turn read
  they cannot have. Scored by rule (vii): a defined item that fails is a FAIL, and an UNDEFINED item among (2), (3) and
  (4) makes B8 UNDEFINED on that seed; it is never dropped while the other controls are scored.
- **B9 answer rate:** per label, flag-on (A1, A8, A10, A14, A16) against flag-off (A0, A7, A9, A13, A15), on the
  labels each pair's own table carries: no label answered off and abstained on. **Round 8: the cells, written into
  the scorer as data, as B12 (b)'s arm list is.** Until round 8, B9 crossed every pair with all six `_INFER_PROBES` labels
  (`stage2_exit_gate.py:163-170`), but `infer_amb1` and `infer_amb2` carry only p_prey_eat, c_heron_hunt and
  c_dog_chase of them (2.8), so three cells of A8/A7 and of A10/A9 had no response on either arm and B9 could never
  hold. The cells are now:

  | pair | table | labels (the cells) |
  |---|---|---|
  | A1/A0 | infer | p_prey_eat, c_heron_hunt, c_frog_eat, c_prey_chase, p_yesno, c_dog_chase |
  | A8/A7 | infer_amb1 | p_prey_eat, c_heron_hunt, c_dog_chase |
  | A10/A9 | infer_amb2 | p_prey_eat, c_heron_hunt, c_dog_chase |
  | A14/A13 | infer_dup | p_prey_eat, c_heron_hunt, c_frog_eat, c_prey_chase, p_yesno, c_dog_chase |
  | A16/A15 | infer_xtier | p_prey_eat, c_heron_hunt, c_frog_eat, c_prey_chase, p_yesno, c_dog_chase |

  A step-1 unit test asserts that each pair's cell list equals the non-teach labels of that pair's turn table as
  `turns_for` returns it, so the data cannot drift from the tables. A label outside its pair's list is never a cell.
  Read by `tools/answer_rate.py --labels`, which the M1 build adds (section 7), reading `responses[label]` from each
  stage-2 shard (round 8: never the prefixed or paired modes, whose `probe_body`, `tools/answer_rate.py:85-95`, reads
  only top-level dicts carrying `answer` and so reports every stage-2 shard UNMEASURED), over dev seeds 1-6 (round 9:
  the default of its `--seeds`, 2.10 step 1); FLIP-HURTS-ANSWERS is reported. Round
  5: UNMEASURED is decided per arm PAIR and seed, not per seed. A pair (A1/A0, A8/A7, A10/A9, A14/A13, A16/A15) whose
  flag-on arm shows `spiking_hop_reads` = 0 on p_prey_eat is UNMEASURED on that seed (equal answer rates from a flag
  that never engaged are not evidence of no harm), and under rule (vi) an UNMEASURED pair counts as not meeting B9.
  Round 6, rule (vii): B9's items are the (pair, label) cells. One UNMEASURED pair, or one UNDEFINED cell (a missing
  or errored response on either arm, or a probe the facts-present map makes UNDEFINED), makes B9 UNMEASURED or
  UNDEFINED on that seed; the other pairs are reported and never carry the seed. A defined cell answered off and
  abstained on makes B9 FAIL on that seed.
- **B10 flag-off identity:** with the flag unset, A0's primary fields equal an `intact_a` arm run at the base
  revision (before any M1 code) on the same seed, and the unit golden holds.
- **B11 duplicate is not a conflict:** P7 holds. Engagement preconditions: on that seed the re-tells appended, i.e.
  A14's hop records show blocks_matched = 2 on both hops with one decoded patient; and (round 5) LIVE-A8 holds on the
  same seed, so A14's silent `acc_conf` comes from a unit that can fire. Otherwise the seed is UNDEFINED for B11, and
  under rule (vi) it counts as not meeting B11. Round 4's "B11 is UNDEFINED only if every seed is UNDEFINED" is
  withdrawn. The circuit-level duplicate test of step 1 is reported beside B11, never in its place.
- **B12 cross tier:** (a) P9 holds. Engagement precondition: REF-A1 holds on that seed (round 6: in place of "B1's
  engagement", whose ADVANCE is now B1's outcome); the `infer_xtier`
  fixture's read-backs (round 5: both, the one right after the turn-0 write and the one right before the first probe)
  returned fish from `ltm.query_patient(heron, hunt)`; the shard's `ltm_bundle` is `wikidata_100k`; A16's hop-1
  `decoded_rows` under (heron, hunt) carry frog only; and the audit counts one patient per tier. Otherwise UNDEFINED.
  Round 8: both A15 and A16 run offiso, and the fixture record of A16 (and of A15, for the flag-off half of P9) shows
  `ltm_type` `ShardedPhasorStore` and no `store_exception` (2.8); a `SharedLTMView` there is UNDEFINED, never PASS.
  (b) On every M1 arm except A3, A11, A17 and A19 (A1, A2, A4-A6, A8, A8b, A10, A12, A14, A16, A21) and every dev
  seed, no hop has `audit_disagree` true (round 6: `audit_disagree` is keyed on the hop having `committed` a patient,
  not on `gate_decision` ADVANCE, 2.3, so an ADVANCE whose winner the route did not commit is not counted, and a
  committed patient on a multi-patient cue is counted whatever the gate label). Round 5: on A10 the hop-2 record under (frog, eat) must exist (the hop that
  carries A10's ambiguity), and on A8 and A8b the hop-1 record under (heron, hunt). `spiking_hop_reads` >= 1 alone
  does not satisfy it. Precondition (round 4): the B12 (b) row of the hop-record table, per arm and seed.
  Every listed arm carries at least one hop record on each chain probe of its table, and every hop carries the audit
  fields. An arm and seed with zero hop records, or with a hop missing an audit field, is UNDEFINED for B12 (b),
  never PASS. A defined hop with `audit_disagree` true is listed with its tier counts, `xtier_disagree` and its
  seed, and one such hop fails B12 (b) on that seed. B12 (b) PASSES on a seed only when every listed arm is defined
  there and none has such a hop. Round 6, rule (vii) named: one UNDEFINED arm or hop among the listed ones makes B12 (b)
  UNDEFINED on that seed; the defined arms are reported and never carry the seed. A3 is excluded because its `.kb` is deranged by design, so its audit is no oracle;
  A17 and A19 for the same reason (their `.kb` is overwritten); A11 and its .rep0 because the lesion blinds the unit
  by design (an ADVANCE on a two-patient cue there is B6's evidence, not a failure). Their `audit_disagree` hops are
  still listed.
- **B13 the patient set follows the store, and no route-level list read decides a hop (round 4; rewritten in round
  5).** (a) **V'_P provenance, the data-flow test.** P10 holds on A17: p_prey_eat abstains, and its hop-1 record
  shows worm in `vp` AND in `vp_not_in_kb`, blocks_matched = 2, patient_lines_active = 2, acc_conf_spikes > 0,
  spurious_P = 0 and gate_decision ABSTAIN. Engagement preconditions: REF-A1 holds on that seed; the overwrite
  engaged under all four conditions (2.8 (i)-(iv)), so worm is in no `.kb` entry, built-ins included, while the
  decoded rows under (heron, hunt) still hold it; whenever worm is in A17's hop-1 `vp`, the record also lists it in
  `vp_not_in_kb` (if it is in `vp` but not in `vp_not_in_kb`, the list still held the word as the hop read it, the
  test cannot separate the designs, and B13 (a) is UNDEFINED on that seed); the facts-present map holds for
  `infer_amb1` from the teach log; the B13 (a) row of the hop-record table holds. Round 6 adds three conditions.
  First, the abstain is the ignition regime with no cross-tier slot: the hop-1 record shows `abstain_regime`
  `ignition`, `n_ignited` >= 2 at `gate_cycles` = R_max, and `xtier_slots` == []. A graded or winner abstain, or a
  cross-tier slot, makes the item UNDEFINED (2.2). Second, the V'_P the hop read was formed after the overwrite:
  `vp_formed_turn` is a turn after the fixture's recorded turn index, else UNDEFINED. Third, the chat-level read-backs
  of 2.8 hold: worm is absent from `chat.stored_facts` and `patients_set`, and `_fact_shard_built_K` read -1 before the
  first probe. B13 (a)'s claim is about the buffer `.kb` (2.6); these read-backs remove the other host lists the chat
  holds. And the teeth: A18 on the same seed derives `[frog, eat, gnat]`. If A18 does not derive it, the overwrite did not bite the
  list route, B13 (a) is UNDEFINED on that seed, and the non-bite is reported first. With the preconditions met, a
  hop-1 ADVANCE or any derived answer on A17 is a FAIL: it is the host-listed patient set the round-3 design allowed
  (worm would have had no line). Worm missing from `vp` altogether, with (iv) read back, is also a FAIL, never an
  UNDEFINED: it is P10's outcome, as in round 4, where "fish in `vp`" was part of the outcome. Making it a precondition
  would have loosened the bar.
  (b) **Route-level list reads only.** P11 holds on A19: p_prey_eat is derived `[frog, eat, gnat]` from
  `[[heron, hunt, frog], [frog, eat, gnat]]`, worm is named nowhere in the answer or in `derived_from`, and the hop-1
  record shows `proposed` frog, blocks_matched = 2, patient_lines_active = 1, acc_conf_spikes = 0 and gate_decision
  ADVANCE. The hop-1 `kb_patient_reads` shows the :1822 clause-test read of the overwritten first entry, which is
  label-only and does not reach the answer. Round 6: that read is also 2.4's entry-level control. It must carry the
  hop's `selected_block` as its block id, and that id must be the overwritten entry's (the first (heron, hunt) entry),
  so the clause test read worm. Without it, B13 (b) is UNDEFINED on that seed: a zero count of other reads proves
  nothing on an entry the proxy never saw, and a hop that selected the re-tell's block never met the overwritten value. Any other
  patient-field read in the hop is at a listed site (the fact-shard index build, :1420, if the index was rebuilt
  there), so `kb_reads_unlisted` is 0 (B7 (c)). Engagement preconditions: REF-A1 and
  LIVE-A8 hold on that seed; the overwrite engaged under (i)-(iv); facts present for `infer_dup` from the teach log,
  with the re-tell appended (blocks_matched = 2 on A19's hop 1); the B13 (b) row of the hop-record table; and the
  teeth: A20 abstains on p_prey_eat on the same seed (else UNDEFINED, reported first). With the preconditions met, a
  hop-1 ABSTAIN, an abstain on p_prey_eat, or worm in the answer is a FAIL. The first is an abstain the list supplied,
  and the other two are a hop-1 value the list supplied. B13 (b) does not test V'_P provenance. Under the `gp{b}`
  gating a line lights only through a matching block, so the list can only remove a line, never add one that fires.
  A list-built V'_P would ADVANCE here too. B13 (a) is the provenance test.

Every lesion reports engaged / not engaged first (docs/TERMS.md, lesion); a lesion that did not hold is reported as
not held, never as no effect.

**Seed invariance, stated.** At the 10-26 gate the six gate seeds' `infer__intact_a` shards
(`research/findings/raw/_stage2_exit/gate/s{42,43,44,100,101,102}/infer__intact_a.json`) carry identical non-text
primary fields (abstained, recalled_svo, derived, derived_from, derived_svo, verified, source) on all ten labels; only
the answer text differs (checked for this revision). So on today's route six seeds are one effective measurement of
the outcome. Every M1 bar is reported as n_pass / n_defined AND n_effective, the number of distinct non-text outcome
vectors across the dev seeds, with the per-seed substrate counts beside it (m{b}, patient-line counts, acc_conf,
conf, n_ignited), since the sequencer and the workspace are seeded per arm and may vary where the outcome does not. A
bar that passes with n_effective = 1 is reported as "PASS, one effective measurement", never as six. These counts sit
beside the verdict of the cross-seed rule (vi), which alone decides whether a bar holds.

### 2.10 M1 build order, with STOP lines

**Step 0, the engagement smoke (read-only; no production edit; dev seed 7, then 1-6).** Where it runs (round 4):
through the pool, on a lane whose checkout holds `data/corpus/tinystories.txt` (the workstation pool lane qualified
for the 10-26 gate-off shards, section 1); never from a checkout without the corpus. Every step-0 shard records
`corpus_present` (`run_worker` :671). A shard that records false is UNDEFINED for every check in it: the learned
lexicons, the corpus vocabulary and the built-in store degrade silently without the corpus, so the branch, nonce and
premise checks would pass vacuously. It is re-run on a corpus lane and is never a GO or STOP input. Every step-0 read
goes through the stage-2 arm worker, which sets `BRAIN_CHAT_SEED` to the arm seed before it imports the server
(`run_worker` :666), so no read falls back to the default seed 42 (round 5).

**Runner changes step 0 needs (round 5; research runner only, no production edit; the golden hash of
`tests/test_stage2_kb_scramble.py` pins every pre-existing arm).**
(1) Per-arm turn tables. The informational `turns` key and `table: "permuted"`, which `turns_for` and therefore
`arm_validity` read (2.8), give `infer_amb1`, `infer_amb2`, `infer_dup` and `infer_xtier` their own tables. Today every
informational arm runs `g["turns"]` (`stage2_exit_gate.py:321-328`).
(2) The D4 recorder of 2.7: the wrappers of `competent`, `extract_transitive`, `_verb_selects_of` and `_animacy_of`,
installed before the server import.
(3) The `stored_facts_after_teach` read, taken before any overwrite fixture (round 6, 2.7).
(4) The `kb_overwrite` fixture mode with its engagement record, conditions (i)-(iv) (2.8), including (round 6) A17's
and A18's chat-list refresh, the `_fact_shard_built_K` reset, their read-backs and the fixture's turn index.
(5) The `infer_xtier` hook: the LTM write right after turn 0 and its two read-backs (2.8). Round 8: the step-0
`infer_xtier` entry carries `BRAIN_SHARED_LTM=0` and `BRAIN_LTM_RNG_ISOLATE=1` in its `env` (offiso), as A15 and A16
do, and the hook records `ltm_type` (`type(chat.inner.composer.ltm).__name__`), the two env values as read, and any
`store_exception`. It writes only when `ltm_type` is `ShardedPhasorStore`, and never through `ltm.shared_store`.
(6) Read-only wrappers of `_fact_shard_first_match`, `_fact_shard_candidates` and `_seq_block` that record
`propose_path` and `fs_candidates` (2.7).
(7) Round 6: the amendment of `tests/test_stage2_kb_scramble.py` named in 2.8. Its assertions at :53-55 (the extras
of `infer` are exactly the two kb arms, and no other group has any) and :65-68 (no other arm carries a scramble) become
explicit allowlists, extended by each step that adds arms. The :65-68 allowlist is the arm-to-mode map of 2.8: the two kb
arms, A3 (`kb`) and A17-A20 (`kb_overwrite`, round 7). The GOLDEN dict and the fingerprint loop (:48-52) are
unchanged. Step 0 adds the informational entries it runs and lands the amendment for them in the same commit.
The `.rep0` change and the live `.kb` proxy are step-1 items.

In the arm worker, with the D4 recorder installed from the first turn, after the teach turns of `infer`, `infer_amb1`,
`infer_amb2`, `infer_dup` and `infer_xtier`:
- (a) call `composer.query_patient_source` for (heron, hunt), (heron, eat), (frog, eat), (frog, chase), and record the
  result, the tier, the selected block, `propose_path`, `fs_candidates` (the fact-shard candidate block ids and the cue
  roles queried), and the decoded rows that match from a full `_read_blocks` scan (at this layout the per-block loop,
  :1389). Any cue whose decoded rows are not empty while the buffer's PROPOSE is None is recorded `host_routed`
  (round 5);
- (b) build the sequencer the production path builds (`_ensure_sequencer` at the composer's defaults, so the REDUCED
  fabric), and record every block's `m{b}` count for the same cues through `run_sequencerK_reduced_with_drive` with
  the composer's own drives, plus the sequencer decision against the host first-match, and the build's memory
  (`tools/mem_ok.sh` first; seed 7 measured before seeds 1-6);
- (c) **the patient slice the build will relay, read where production computes it (round 5).** For EVERY stored block
  b (the substrate V'_P of 2.3), call the production `block_cleanup_scores(composer, b)`
  (`_phaseB_onebrain_sequencer_derisk.py:64-89`). Right after it returns, read the PATIENT row of the cleanup
  membrane (`cp_membrane_potential_v[c_base + i_P * V : c_base + (i_P + 1) * V]`, with i_P the index of 'patient' in
  `main_roles`), clipped at 0 as the agent and action rows are (:87-88). That is the slice step 1's
  `with_patient=True` return will hand on, computed by the same op and the same cleanup synapses
  (`_seq_cleanup_conns`, :1620-1639, which wires the patient row too and skips every unrecruited reserve slot, :1634).
  The V'_P read uses this slice and nothing else. The full probe `_phaseC_task1_cleanup_probe.block_cleanup_scores_full`
  (:33-57) wires EVERY slot, reserve slots included, and the chat buffer carries 128 of them
  (`vocab_headroom`, `brain_conversational_agent.py:285`). An unrecruited slot can outscore the true word there
  (`__free10__` 5270 against `look` 4080, `_silent_reserve`'s docstring, :988-998). So the probe is used only as a
  cross-check, under the P10 treatment: on every matching block, its agent, action and patient slices, after
  `np.maximum(., 0)` and with every reserve slot zeroed, must equal the production read's slices on the non-reserve
  slots. Round 6 defines the terms: a reserve slot is a slot j for which `composer._silent_reserve(j)` returns True at
  the read (`one_brain_composer.py:988-998`: `self.words[j]` still starts with `__free`), the exact rule
  `_seq_cleanup_conns` skips by (:1634). A recruited reserve slot holds a word (fixture words are recruited into
  reserve slots), so it is not a reserve slot under this rule; it is compared like any other word, never zeroed by its
  index range. Then pass each block's patient slice through the production relay (`onbridge_divnorm_drive` on the composer's
  score bridge at `input_gain` 1.0), and record which patient word pools fire per block, their union in block order
  (V'_P as the build will form it), and the set of decoded stored patients beside it;
- (d) record `chat._last_gnw_delib` on c_heron_hunt in `infer_amb1` (the single-hop conflict gate on onebrain);
- (e) from the D4 recorder (2.7), never from the `comprehension` field: each teach's `extract_transitive` return and
  its `judge`-frame `competent()` call, meaning the inputs `(n0, v, n1)`, each word's membership in the `brain_vocab`
  set passed and that set's size, the recorded inner `_verb_selects_of` and `_animacy_of` returns, `fully_covered`,
  `fully_oov`, `or_matches_return` and the return, with the branch read from them by 2.7's rule. Every
  `competent()` call of another caller (`repair_target`, the three-organ bus) is listed beside it. Beside it: t_frog_eat_apple's `comprehension.margin` and `comprehension.known_binding`; the `surprise`
  and `reconsolidation` records of every `infer_amb1`, `infer_amb2` and `infer_dup` teach; the row count and the
  decoded patients under each ambiguous and each re-told cue; the `infer_xtier` fixture (round 8: its `ltm_type`, its
  env values and any `store_exception` first; then buffer rows under (heron, hunt), `ltm.query_patient(heron, hunt)`,
  `all_candidate_patients` on the routed shard, the audit's counts per tier);
  and `ws_seed` against the arm seed;
- (f) facts present on the bound store (round 4): for every row of 2.9's premise map, the buffer's decoded rows
  (`_read_blocks`, never `.kb`) under each hop cue, plus `ltm.query_patient` and `all_candidate_patients` on the
  routed shard under the same cue, beside the arm's own `teach_acquired`, `stored_facts_after_teach` and
  `builtin_facts`;
- (g) the `.kb` overwrite fixtures of A17 and A19 (2.8, round-5 plans: worm -> frog on `infer_amb1`'s worm entry; the
  FIRST (heron, hunt) entry frog -> worm on `infer_dup`), applied after the teach block with the flag off (no M1 code
  exists yet): the engagement record, conditions (i)-(iv), with the full `_kb_triples(buf)` read back for (iv);
- (h) an inert-read check on seed 7: the `infer` turn table run in three fresh workers. One is bare (no recorder, no
  reads). One has the D4 recorder only (round 5: the recorder is instrumentation too). One has the recorder and the
  reads (a)-(c), (f) and (g). The three must give identical primary fields on every probe turn, and the recorder-only
  worker's teach turns must match the bare worker's `comprehension`, `surprise` and `reconsolidation` fields. A
  difference is posted, and (d) is then re-read in a worker that makes none of the other reads.

GO rule (written here, before the run): on 7/7 seeds, with `corpus_present` true on every shard, every true match
drives its `m{b}` over the composer's `match_thresh` and no non-matching block does; every stored block's patient
slice fires exactly one patient pool under the production relay, its decoded stored patient's, so the relay-formed
V'_P equals the set of decoded stored patients; the two blocks of a re-told cue fire the same pool and the two blocks
of an ambiguous cue fire different pools; every teach of 2.8's table took its expected branch according to the D4
recorder and its fact is in the decoded buffer rows after the turn; t_frog_eat_apple recorded `known_binding` true;
no `reconsolidation` record on an `infer_amb1`, `infer_amb2` or `infer_dup` teach shows `rewrite`, each ambiguous cue
holds two decoded rows with two patients, and each re-told cue holds two rows with one patient; every row of 2.9's
premise map holds exactly on the bound store and the LTM; both overwrite fixtures engage under (i)-(iv) (round 5: worm
in no `.kb` entry after A17's overwrite, built-ins included, and still decoded; worm the first (heron, hunt) entry's
patient after A19's and in no decoded row); the `infer_xtier` fixture landed (round 8: `ltm_type`
`ShardedPhasorStore` with no `store_exception`; frog in the buffer only, fish in the LTM only, on both read-backs); `ws_seed` equals the arm seed; the recorder's `or_matches_return` is true on every call; no
probe cue of (a) is `host_routed`; the probe's P10-treated slices equal the production read's on every non-reserve
slot (c); and the reads and the recorder are inert (h).

STOP lines: a missed true match on any seed: post the counts; the circuit is banked and the named next carrier is
the cs-race engram race read for co-latching (two latches ON = conflict). A patient pool missing, wrong or not unique
(a block lighting no pool, two pools, or a word no block decodes): the patient identity the unit would count is not on
the substrate at this fidelity, and so neither is V'_P. The probe's clipped agent or action slice differing from
`block_cleanup_scores` on a non-reserve slot of any block, or its P10-treated patient slice differing from the
membrane read of (c): the two reads disagree about what the build would relay; post the difference (the V'_P read
stays the membrane read). A probe cue recorded `host_routed` (decoded rows under it, PROPOSE None): the fact-shard
index missed a block the store holds; post the candidate ids and the decoded rows, and Claude rules on whether A21
carries the affected bars. An `or_matches_return` false: post the call; that teach's branch is UNDEFINED. A new-table teach refused or taking another branch than 2.8's: post the recorder's inputs, terms and
return; the table lacks its ambiguity there, and Claude rules on the word. t_frog_eat_apple with `known_binding` false:
the hop-2 table then rides on the margin; post it. A `rewrite` on an ambiguity or re-tell teach, or a cue holding one
row where two are expected: D2 took the fact; post it, and Claude rules on the named fallback (the second fact written
by a `composer.store` fixture after the teach block, 2.8). A premise-map row failing on the bound store or the LTM (a
missing premise or an extra patient under a hop cue): post it; the probes that row feeds are predicted UNDEFINED,
and Claude rules on the table words. A (frog, chase) fact in the teach log or `builtin_facts` is a premise failure of
the same kind. A decoded or LTM (frog, chase) row that nothing taught is not a premise failure (2.9): post it, because
B8 then predicts a moat FAIL wherever that row reaches an answer. An overwrite fixture not engaging: post the record; its half
of B13 is predicted UNDEFINED. The `infer_xtier` fixture not landing (round 8: the remedy depends on the record, read
in this order). If `ltm_type` is `SharedLTMView` (or any type other than `ShardedPhasorStore`) or a
`SharedLTMWriteError` was recorded, the offiso env did not reach the server: post `ltm_type` and the two env values as
read, fix the entry's `env` (`BRAIN_SHARED_LTM=0`, `BRAIN_LTM_RNG_ISOLATE=1`, `BRAIN_ONE_BRAIN` unset) and re-run that
step-0 read. The remedy is never a write through `ltm.shared_store` and never a write on the shared default, and A15
and A16 stay UNDEFINED until a re-run records `ShardedPhasorStore`. If the type is right and the write or a read-back
still fails (for example a word outside the LTM codebook), post it, and Claude rules on the fixture words. Reads that are not inert: post the differing turns. If the re-tells
did not append on any seed, B11 is predicted UNDEFINED and the step-1 circuit test carries the duplicate case. Post
`[c2-hops-0] RESULT`, STOP for Claude.

**Step 1, the build (default OFF).** `conflict_unit` in both sequencer builders (V'_P patient lines, `gp{b}` gates,
depressing `pP_w -> acc_conf`, every pre-existing pathway `stp_disabled`, V'_P in the rebuild signature); the optional
patient role in `block_cleanup_scores`, `make_block_drives` and `run_sequencerK_reduced_with_drive`; the per-read STP
reset; `_hop_spiking` with the cross-tier slots, the audit call and the three flags in the route;
`tools/answer_rate.py --labels` (per-label reads over every turn of an arm file, with its own unit test; round 9:
with `--seeds`, default 1-6, behind a gate-seed guard). Round 4
adds: V'_P formed from the relay-lit patient pools of every stored block, with the relay run before the fabric check
(2.3); in the stage-2 arm worker, the per-turn route trace, the D4 recorder, `stored_facts_after_teach` and the
`kb_overwrite` fixture mode with its engagement record (2.7, 2.8); and in the scorer, written before the step-2
runs, the premise map and the hop-record table of 2.9 as data, with `probes_undefined` and `undefined_items` lists.

Round 5 adds the following items (the step-0 runner changes already exist by then).
- **Production paths, all default-off or byte-identical.** The :1654 read becomes agent and action only (2.4). The
  route's `_hop_spiking` runs the buffer conflict read and builds and records the full drive vector on every hop,
  whichever tier serves it, with the LTM candidate slots for both tiers and no `conflict_gate` call (2.2, 2.3).
  `_hop_spiking` reads `gnw_deliberation._DEFAULT_SEED` as a module attribute at call time, never as a name bound at
  import, so a test can pin it.
- **In the stage-2 arm worker.** The live counting `.kb` proxy, installed by the `OneBrainComposer.__init__` wrapper
  before the server import (2.4). The decision-scope recording of every `.kb` access by caller and kind,
  `kb_patient_reads`, `kb_reads_unlisted` and 2.7's positive control. The `propose_path`, `fs_candidates` and
  `host_routed` records, and the `cp_n_syn` and `cp_w_max` read-backs. `arms_for` emits `.rep0` for an informational
  entry with `rep: true` (A4, A5 and A11; today only lesion arms get one, `stage2_exit_gate.py:331-342`). The new
  informational entries A8b and A21, with A21's env `BRAIN_FACT_SHARD_RETRIEVAL=0`. A12's `table: "permuted"` (2.8).
- **In the scorer.** The cross-seed rule (vi), REF-A1 and LIVE-A8, B8 (1b), and the validity reads of rule (i):
  `sources`, `process.ltm_bundle`, and the `corpus_present` key present and true.

Round 6 adds the following items.
- **Production paths, all default-off or byte-identical.** The conflict lesion writes the built `pP_w -> acc_conf`
  weights to exactly 0.0 after every fabric build, with explicit wiring as the named alternative (2.3, 2.5). The
  conflict read's own entry point reads `len(self.kb)` once and passes it to `_ensure_sequencer` (2.3). `_write_block`,
  and every other `store_conns` writer listed in 2.3, sets `_seq_dirty` whenever the conflict unit is built, whatever
  `integrated_loop` says. The rebuild that forms V'_P records its turn label and K.
- **In the stage-2 arm worker.** The counting proxy as a class-level `kb` property with a wrapping setter, in place of
  the `__init__` wrapper, wrapping on every insertion path and supporting `clear`, `pop`, `remove` and `__delitem__`
  (2.4). The site map keyed by (function, statement, kind), with the new `len` site, and the entry-level positive
  control (2.4, 2.7). The per-turn `patch_calls_before`, `patch_calls_after` and `patch_calls_delta`, and the per-hop
  `patch_calls_in_hop` that sets `propose_path` `patched` (2.7). The `abstain_regime`, `theta_hi`, `seq_miss`,
  `vp_formed_turn`, `vp_formed_K` and `kb_unwrapped` records, and `host_routed` restricted to `fact_shard` and
  `escalated` (2.7). `audit_disagree` keyed on `committed` (2.3). The extension of the `tests/test_stage2_kb_scramble.py`
  allowlists of step 0 (7) to A8b, A21 and the `.rep0` arms of A4, A5 and A11 (2.8).
- **In the scorer.** Rule (vii) (2.9), B1's precondition and outcome as now separated, REF-A1 with hop 2 at ADVANCE,
  B3's per-turn and per-hop patch counts, B5's and B13 (a)'s ignition-regime requirement, B8 (1b)'s A21 liveness, and
  B13 (a)'s `vp_formed_turn` and chat-list read-backs.
- **Before step-2 data.** The site map is re-pinned at the step-2 revision (each key's current line recorded) and
  committed, and the step-1 unit test that resolves every key to one statement passes at that revision.
- **Review check.** The step-1 review confirms, in code, that neither `resolve_compositional_chain`, `_hop_spiking`,
  the conflict read's entry point nor the V'_P builder takes or reaches a chat handle (2.6).

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
the flag builds `_ensure_sequencer`'s bridge for the read. Round 4: the V'_P builder reads no `.kb` entry (the
2.4 guard raises on `__getitem__` and `__iter__` from the builder's frame and allows `__len__`); a composer whose
`.kb` patient strings are overwritten (both fixture plans of 2.8, on step-1 circuit composers) forms the same V'_P
and the same `acc_conf` count as its untouched twin (the circuit-level twin of B13; round 6: the plans' words, not the
`w_c` fixture's fish. For A17's plan the twin stores (heron, hunt, frog), (heron, hunt, worm) and (frog, eat, gnat)
and overwrites the worm entry's patient to frog. For A19's plan it stores (heron, hunt, frog) twice and (frog, eat,
gnat) twice and overwrites the FIRST (heron, hunt) entry's patient to worm. Each twin also sets
`_fact_shard_built_K` to -1 and reads it back, as A17's fixture does); the D4 recorder is behaviour-
preserving (the committed D4 battery's decisions are identical with and without it) and its two terms OR to the
return on every call; `spiking_hop_reads` counts exactly the hop scopes that made a workspace read (a fixture with a
forced exception inside a hop scope does not count it). Module docstring corrected (it says the hops are list
scans today; under the flag they are spiking reads within the buffer). Round 5 adds these unit tests:
- **:1654 change.** With the patient field dropped, the reduced vocabularies, the rebuild signature and the fabric's
  synapse arrays are equal to the unchanged read's, on the circuit fixtures and on stores holding the `infer` and
  `infer_amb1` facts.
- **Counting dict and proxy.** Every access form of 2.4 (`get`, `[]`, `items`, `values`, `keys` then `[]`, a `dict(...)`
  copy) is counted with its caller line and returns the plain value. `isinstance(fact, dict)` still holds. A
  patient-field read placed at an unlisted line inside a decision scope raises `kb_reads_unlisted` to 1. A read by a
  helper in another module inside the scope is counted too, so the frame bypass is closed.
- **Behaviour preservation on whole tables.** The live proxy and the recorders are behaviour-preserving: the `infer`
  and `infer_amb1` tables run on seed 7 with and without the arm-worker instrumentation, flag off and flag on, give
  identical primary fields on every turn. Round 8: the instrumentation under test includes the per-turn
  `store_snapshot` (2.7, 2.9), so the post-reply `_read_blocks` call on every probe turn is shown inert by the same
  comparison.
- **Sentinel data-flow twin.** On step-1 circuit composers, overwriting EVERY `.kb` patient string with a sentinel
  absent from the vocabulary leaves V'_P, `acc_conf`'s count and the PROPOSE words of the flagged hop identical to the
  untouched twin. Only the :1822 clause test reads the sentinel, and its count is the only patient-field count. Round
  6: the twin also sets `_fact_shard_built_K` to -1 after the overwrite and reads it back, so the fact-shard index
  (which holds codes and block ids, built from the strings at the last K change, :1410-1432) is rebuilt from the
  sentinel list. A circuit composer has no chat object, so this twin's claim is the narrowed one of 2.6 (the buffer
  `.kb`); A17 carries the chat-list overwrite (2.8).
- **Every hop records its read.** A qp-lesioned composer and a (frog, chase)-style miss each record an all-zero drive
  vector, an `acc_conf` read and a COMMIT.
- **Seeds pinned.** Every unit test that reaches `_get_bridge`, `_get_theta` or `_hop_spiking` pins `BRAIN_CHAT_SEED`
  to design seed 7. It sets the variable before importing `webapp.gnw_deliberation` (or the server), or it sets the
  module attribute `_DEFAULT_SEED` to 7, and it asserts that the recorded `ws_seed` is 7. A test that records 42 fails:
  `_DEFAULT_SEED` falls back to 42, a gate seed, when the variable is unset (`webapp/gnw_deliberation.py:69`).
- **Recorder checks.** The D4 recorder records the inner `_verb_selects_of` and `_animacy_of` returns in call order.
  `or_matches_return` is true on every call of the committed D4 battery (asserted in the test; at run time it is only
  recorded). The caller label separates `judge`, `repair_target` and the three-organ bus.
Round 6 adds these unit tests:
- **One test per insertion path (item 1).** `store`, `store_fact`, `hear`, `hear_multiframe`, the `kb_overwrite`
  fixture (`buf.kb[i] = (dict(fact, patient=p), handle)`), `update_on_mismatch`, the reconsolidation organ's fallback
  `composer.kb[idx] = (fact2, None)`, a slice assignment, `insert`, `extend` and `+=`, and a reassignment of `.kb`
  through `TieredFactStore` each leave every entry a counting dict on which a patient read is counted with its block
  id. `clear` and `pop` keep list semantics, and a mutating method outside the supported set raises.
- **Entry-level control.** On a circuit composer with the flag on, a hop whose PROPOSE selected block b lists the
  clause-test read with block id b. A composer whose entry b is replaced by a plain dict behind the proxy's back (the
  test's own bypass) makes the hop UNDEFINED for B7 (c)'s patient clause, `vp_kb_entry_reads` and B13 (b).
- **Conflict lesion at exactly zero (item 2).** Under `BRAIN_CHAIN_SPIKING_HOPS_CONFLICT_LESION`, the read-back
  `cp_w_max` == 0.0 at the read, after a first build and again after a V'_P rebuild, with `cp_n_syn` equal to the
  intact build's on the same fixture and > 0. A pathway built at weight 0 through `RegionPathway` is shown to read
  0.01 in the same test, so the test fails if the lesion falls back to a build weight.
- **Site map (item 3).** Every key of the allowed-site map (function, statement, kind) resolves to exactly one
  statement of the current source, and the recorded line is that statement's line. A key that matches no statement or
  more than one fails the test. On a flagged circuit hop, the conflict read's `len` site is recorded with kind `len`,
  and an entry read added at that site for the test counts as unlisted.
- **Rewrite between two conflict reads (item 9).** With the conflict unit built and the integrated loop off, a
  two-patient cue is read, then `update_on_mismatch` (with `pe_labile=0.0`, as the reconsolidation organ calls it,
  `reconsolidation_production_organ.py:132-133`) rewrites one block's patient in place (same K), then the cue is
  read again. The second read's V'_P, `vp_lit_by_block` and patient lines follow the rewrite, and its `vp_formed_turn`
  and `vp_formed_K` are the second read's. With the conflict unit not built, `_seq_dirty` is untouched by the write
  (byte-identical).
- **The organ and bus seeds (item 13).** `_ORGAN` is a per-process singleton that `get_organ` builds once
  (`comprehension_production_organ.py:833-869`), so an earlier test in the same pytest process can leave an organ built
  at the default seed 42. Every unit test that reaches D4, the three-organ bus or the server either asserts
  `comprehension_production_organ.get_organ().seed == 7` together with `gnw_deliberation._DEFAULT_SEED == 7` and
  `gnw_three_organ_bus._DEFAULT_SEED == 7` (`webapp/gnw_three_organ_bus.py:114`), or sets
  `comprehension_production_organ._ORGAN = None` before its first D4 call and asserts the rebuilt organ's seed is 7.
Round 8 adds these items:
- **In the stage-2 arm worker (item 2).** The per-turn `store_snapshot` on every non-teach turn of every M1 arm: one
  `_read_blocks` call right after `brain_chat` returns, all K rows kept with their block ids, plus the record-only
  `audit_ltm_all` scan under each cue of the arm's premise-map rows (2.7, 2.9). Unit test: on a flagged step-1 circuit
  composer, a non-chain turn and a chain turn whose hop 1 abstains each record a snapshot with K rows, and a map cue
  never hopped on the turn reads its decoded rows from the snapshot.
- **In the scorer (items 2 and 3).** The facts-present rule reads every map cue's bound-store side from the probe
  turn's `store_snapshot` (a turn without one is UNDEFINED), and B9's per-pair cell lists are data (2.9). Unit test:
  each pair's cell list equals the non-teach labels of that pair's turn table as `turns_for` returns it.
- **The `--labels` unit test (item 4; round 9: a vendored copy, and the seeds as an argument).** It reads
  `responses[label]` from a stage-2 shard vendored under `tests/`, `tests/fixtures/stage2_infer_intact_a_gateoff_s1.json`,
  a byte copy of the committed
  `research/findings/raw/_stage2_diag/c2_comprehension_gateoff/s1/infer__intact_a_gateoff.json` (git blob
  `5bddc4178c2bcc66254ee8eca7feef2ebe081953`, 65,162 bytes, the same at origin/main 34a4e5f75 and c394b764a), and
  classifies each of its six probe labels from there. It never reads `research/findings/raw`: pool-provisioned
  revisions exclude it (`tools/pool_provision.sh:91-92`) and so do `tools/gpu_remote.sh`'s sparse worktrees (:47). The
  test computes the copy's git blob id and asserts that it equals that id. A missing copy or a different id FAILS the
  test; it is never skipped (no `pytest.skip`, `skipif` or `importorskip` guards it). It also asserts that the
  existing `probe_body` (`tools/answer_rate.py:85-95`) returns None on that file, which pins the reason `--labels`
  exists. The mode takes `--seeds`, default 1-6. It never falls back to `SEEDS` (`tools/answer_rate.py:52`, the gate
  seeds 42/43/44/100/101/102), the default of `paired_per_seed` (:162) and of `paired --seeds` (:224). A gate seed in
  `--seeds` is refused with exit 2 unless `--prereg <path>` names a file that exists at the read revision and whose
  name contains "prereg", the name check of the pool's gate-seed guard (`tools/pool_queue.sh:248-261`); the record
  carries the seeds read and the prereg path. The test asserts that the default seeds read are 1-6, that a dev seed
  with no shard is reported UNMEASURED, that `--seeds 42` with no `--prereg` and `--seeds 42 --prereg
  research/coordination/LOCAL_LLM_RUNBOOK.md` both exit 2, and that a `--prereg` naming an existing `*prereg*.md`
  file is accepted.
- **Where the step-1 unit tests run (round 9).** In the step-1 build's own worktree, a full checkout of the branch
  (not a pool-provisioned archive and not a `gpu_remote.sh` sparse worktree), under `tools/memcap.sh 6 --
  .venv/bin/python -m pytest`, numpy CPU; the step-1 RESULT names the checkout and its revision. Any other step-1
  test that reads a file under `research/findings/raw` follows the `--labels` test's rule: the file is vendored under
  `tests/fixtures/` with its blob id checked, or the test FAILS when the file is absent, never skips. So no step-1
  test can pass by skipping on a provisioned or sparse checkout.
- **In the A15, A16 and step-0 `infer_xtier` entries (item 1).** `env` carries `BRAIN_SHARED_LTM=0` and
  `BRAIN_LTM_RNG_ISOLATE=1`, and the fixture hook records `ltm_type`, the env values as read and any
  `store_exception` (2.8). Unit test: under that env, a server-built tiny-demo +LTM session's
  `chat.inner.composer.ltm` is a `ShardedPhasorStore` that accepts `store`; under the default env it is a
  `SharedLTMView` whose `store` raises `SharedLTMWriteError`, which the hook records without writing.

Resource read (seed 7, under `tools/mem_ok.sh`, before any pool line; posted with the step-1 RESULT): the conflict
unit's cell and synapse counts read back from the built bridge against section 5's formula; build time and RSS of the
sequencer with the unit, of the two workspace bridges (`_get_bridge` intact and lesioned) and of `_get_theta`'s
calibration; the per-hop latency (drive rebuild, settle, workspace read); and the peak RSS of one `infer_amb1` arm
with the flag on against off. If the flag-on arm's peak exceeds 8 GB (the line memcap of 10 GB less 2 GB headroom),
STOP for a memcap ruling.

**Step 2, the dev read.** A0-A21 with A8b (and A3a and A3b) on dev seeds 1-6 through the pool at one pinned revision with
provenance on (numpy, memcap 10), scored against 2.9 by a scorer written before the runs. Post `[c2-hops-2] RESULT`,
STOP. A failed bar is a RESULT, never a retune.

**Step 3 (Claude).** A `flag: no` prereg carrying A1-A21 and A8b as informational arms of the 11-21 gate attempt, or
a flip packet if every bar holds under the cross-seed rule (vi) of 2.9 (each bar PASSES on all six dev seeds; an
UNDEFINED or UNMEASURED seed is not a pass); the default stays OFF until a flip prereg passes (smoke-first, flag-off battery arms on
the gate seeds, answer-rate; round 9: the answer rate on the gate seeds is read by
`tools/answer_rate.py --labels --seeds 42,43,44,100,101,102 --prereg <that flip prereg>`, behind the gate-seed guard
of 2.10 step 1).

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
  rule. M3 step 1 waits for the f1i result. The frame packet's NMDA trace was parked by RULING focus-1025, which
  bound only until the 10-26 Stage-2 gate RESULT; no ruling has un-parked it since, and it is not M3's carrier
  (section 7's ruling text names the cs-race trace). The starvation that stopped the frame packet (below, 3.5) is the same family, which is why the
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

### 3.5 Organ battery (dev seeds 1-6 and 8-16, numpy, cheap)

Round 5: design seed 7 is excluded from the organ count. The M3 build engagement read and any build-time choice are
made on seed 7 (3.8), so seed 16 replaces it and the count stays 15 seeds. No seed here is a gate seed.

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
  so `competent()` would pass it through unjudged). That check needs the corpus: a step-0 shard without
  `corpus_present` true is UNDEFINED for it (round 4). Engagement precondition: on off_garbled, every G teach turn's
  `comprehension` record is non-null with `in_scope` true AND the D4 recorder (2.7) shows its `judge`-frame
  `competent()` call returning True with `fully_oov` (D4 judged it in its fully-OOV branch); on m3_garbled, the same,
  carrying the M3 margin. Otherwise the seed is UNDEFINED. Prediction: every G teach refused under M3 and under
  the production default, and STORED with the gate off (the teeth: on a seed where the gate-off brain stores fewer
  than 2 of the 3, the table has no teeth and the seed is UNDEFINED for R2).
- **answer rate (round 9: pairs and cells named).** `--labels` mode, FLIP-HURTS-ANSWERS reported, per label, each
  M3-on arm against the flag-off arm that runs the same table at the same pinned revision and dev seed, on the
  non-teach labels that pair's own table carries. Until round 9 this bullet read "m3 against flag-off on every infer
  and ask probe label" and named no pairs; `infer_garbled` carries none of the six `_INFER_PROBES` labels and the
  ask labels exist only on R-goat's arm, so under rule (vii) R4 was UNDEFINED on every seed. The cells are R4's
  (3.7), written into the scorer as data, as B9's are (2.9):

  | pair (M3 on / flag off) | group, table | labels (the cells) |
  |---|---|---|
  | m3_intact / `infer` `intact_a` | `infer`, the registered table | p_prey_eat, c_heron_hunt, c_frog_eat, c_prey_chase, p_yesno, c_dog_chase |
  | m3_permuted / `infer` `permuted` | `infer`, `permuted_turns` | p_prey_eat, c_heron_hunt, c_frog_eat, c_prey_chase, p_yesno, c_dog_chase |
  | `m3_ask` / `ask` `intact_a` | `ask`, the registered table | g_eel, m_yak, g_crab, m_moose, g_newt, m_goat, lp_ibex_1, lp_ibex_2, u_toad_1, c_dog_1, u_toad_2, c_dog_2, u_toad_3, c_dog_3, u_toad_4 |

  The labels are `_INFER_PROBES` (`stage2_exit_gate.py:163-170`) and the non-teach turns of `_ASK_TEACH +
  _ASK_PROBES` (:173-197; t_ibex, a teach, is not a cell), at origin/main c394b764a. The flag-off sides are the
  groups' registered arms, run with M3 unset (its default) at the step-3 revision on each dev seed 1-6, beside the
  M3 arms; the D4 recorder the M3 arms carry is behaviour-preserving (2.10 step 1), so the pair differs only in the
  flag. Not R4 pairs: m3_garbled with off_garbled, because `infer_garbled` carries no infer or ask label (its G
  probes are R2's evidence, and an off-arm answer there is a garbled store, which R2 bars, not an answer M3 must
  keep); m3_rs_lesion and its .rep0, because a lesion is not the flip; gateoff_garbled, because the gate is not the
  flag. A step-3 unit test asserts that each pair's cell list equals the non-teach labels of BOTH arms' turn tables
  as `turns_for` returns them (after step 3's `table` and `turns` keys), so the data cannot drift from the tables. A
  label outside its pair's list is never a cell.
- **R-goat (reported only):** `ask` with M3 on, the arm `m3_ask` (round 9: named, because R4 reads its replies;
  the prediction below stays reported only). Prediction: 'the goat eats the hay' and 'the crab eats the leaf'
  are still refused, because they go through the covered branch M3 does not touch. The named next method for them is
  the learned order read added as a third cue on the covered branch, decided after M3's organ verdict.

### 3.7 Predictions and bars

Organ (each per seed on dev seeds 1-6 and 8-16; round 5: design seed 7 excluded, 3.5). Pass rule, fixed before any
data (round 4): at least 14 of the 15 seeds must meet every bar, and a seed that is UNDEFINED for a bar (under 3.5's learning-engagement precondition, or under
the bar's own precondition) is reported and counts as NOT meeting it. So at most one seed per bar may be UNDEFINED or
failing.
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

Reply (dev seeds 1-6, every seed; every shard with `corpus_present` true, else UNDEFINED; round 5: each R bar holds
only under the cross-seed rule (vi) of 2.9, i.e. it PASSES on all six dev seeds, and a seed on which it is UNDEFINED or
UNMEASURED counts as not meeting it; the general rule (i) of 2.9 on `sources`, `ltm_bundle` and the `corpus_present`
key applies to every arm an R bar reads, the flag-off sides of R4's pairs included (round 9); round 6: the
within-seed rule (vii) of 2.9 applies too, so any UNDEFINED or UNMEASURED item of an R bar, such as one G teach of R2
or one (pair, label) cell of R4, makes that bar UNDEFINED or UNMEASURED on that seed,
and A4 (a)'s exclude-and-score is not used): R1 m3_permuted acquires
t_heron_hunt and p_prey_eat ends on frog (with the 3.6 precondition). The acquisition of t_heron_hunt is R1's outcome,
not a precondition (else R1 could never fail); the facts-present rule of 2.9 applies to the other premise: 'the gnat
eats the frog' (t_frog_eat, passed through and stored at production) must be acquired and in
`stored_facts_after_teach`, else the seed is UNDEFINED for R1; R2 every G teach in m3_garbled is refused and no later probe recalls it (with the 3.6 precondition and
teeth), scored on a seed only where R1 holds on that seed, else UNDEFINED; R3 m3_intact equals flag-off on primary
fields, the flag-off arm being `infer` `intact_a` at the same revision and seed, R4's pairing side (round 9: pinned);
R4 no FLIP-HURTS-ANSWERS, where a seed whose m3 arms show no M3 branch decision on any teach (no
`judge`-frame `competent()` call recorded True with `fully_oov` and carrying an M3 margin) is UNMEASURED for R4, never
PASS (equal answer rates from a flag that never decided anything are not evidence of no harm). Round 9: R4's items are
the (pair, label) cells of 3.6's answer-rate table, read from the scorer's data. A defined cell answered on the
flag-off arm and abstained on the M3 arm makes R4 FAIL on that seed; a missing or errored response on either arm of a
pair makes that cell UNDEFINED, and R4 with it (rule (vii)); a cell outside the table is never read. The
no-engagement clause stays per seed, over the M3 arms of all three pairs, and is not made per pair: `m3_ask`'s teaches
take the covered branch by R-goat's prediction, so a per-pair clause would leave that pair, and so R4, UNMEASURED on
every seed; R5 flag unset is
byte-identical; R6 m3_rs_lesion refuses t_heron_hunt and derives
nothing on p_prey_eat, on every seed and its .rep0 (with the 3.6 precondition), scored only where R1 holds.

### 3.8 M3 build order, dependencies, STOP lines

- **Step 0 (read-only, now, parallel to M1 step 0; on a corpus lane, every shard recording `corpus_present` true,
  else UNDEFINED for (a) and (d) and re-run).** **Seeds (round 5).** (a) and (d) run through the stage-2 arm worker,
  which sets `BRAIN_CHAT_SEED` to the arm seed before it imports the server (`run_worker` :666). They run on design
  seed 7 first, then on dev seeds 1-6, as a step-0 informational table of the four intact `infer` teaches followed by
  the W, G and out-of-scope items heard as teach turns (its own `turns`, 2.8). They never run in a bare process: there
  `_brain_chat_seed()` (`webapp/server.py:4038-4062`), `gnw_deliberation._DEFAULT_SEED` (:69) and
  `gnw_three_organ_bus._DEFAULT_SEED` (:114) all fall back to 42, a gate seed. Nor does step 0 build the D4 organ
  directly at its own default (`ComprehensionProductionOrgan(seed=42)`, `get_organ(seed=42)`,
  `comprehension_production_organ.py:362`, :836). (b) builds `WordRoleParserOrgan` with its seed passed explicitly (its
  default is 1, `word_role_parser_organ.py:256`) on design seed 7 and the organ dev seeds 1-6 and 8-16, with the
  organ's seed read back and recorded; no server import. Every step-0 record carries its seed, and a record with seed
  42, 43, 44, 100, 101 or 102 voids its step. (a) The D4 decision on every W, G and out-of-scope item at production
  defaults: the scope and the competence branch from the D4 recorder (2.7: `extract_transitive`'s return and the
  `judge`-frame `competent()` call's inputs, terms and return, never the `comprehension` field), and the margin; (b) the word-role parser's commit read on W and G with
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
  is the third build step under this header and waits for the cap ruling (cap accounting, top). Runner changes it
  needs (round 5, named here because the step-0 and step-1 ones do not cover them): m3_permuted and m3_rs_lesion carry
  `table: "permuted"`, since `turns_for` gives `permuted_turns` only to the arm named `permuted`; m3_garbled,
  off_garbled and gateoff_garbled carry `infer_garbled` as their own `turns`; and m3_rs_lesion carries `rep: true` for
  its `.rep0` (2.8; `arms_for` emits `.rep0` only for lesion arms today). The D4 recorder is installed in every M3 arm.
  Round 9 adds three items. `m3_ask`, R-goat's arm, is an informational entry of the `ask` group with M3 on and the
  group's own `turns`, named in the test allowlist as `ask`'s one extra (2.8). The flag-off sides of R4's pairs
  (`infer` `intact_a`, `infer` `permuted` and `ask` `intact_a`) run as pool lines at the same pinned revision on dev
  seeds 1-6. The scorer, written before the step-3 runs, carries R4's (pair, label) cells of 3.6 as data, with the
  unit test that each pair's cell list equals the non-teach labels of both arms' turn tables as `turns_for` returns
  them.
  Every unit test that reaches the server, the D4 organ or the workspace pins `BRAIN_CHAT_SEED` to design seed 7, as
  in 2.10 step 1. Round 6: pinning the variable is not enough for the D4 organ, which is a per-process singleton
  (`_ORGAN`, built once by `get_organ`, `comprehension_production_organ.py:833-869`), and for the two `_DEFAULT_SEED`s,
  read once at import. So each such test also asserts `get_organ().seed == 7`, `gnw_deliberation._DEFAULT_SEED == 7`
  and `gnw_three_organ_bus._DEFAULT_SEED == 7`, or resets `comprehension_production_organ._ORGAN = None` first and
  asserts the rebuilt organ's seed is 7 (2.10 step 1).

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

- Design seed 7 (M1 `w_c`, `n_conf_ref`, the resource read, the step-1 unit tests; M3 build engagement); M1 dev read
  seeds 1-6; M3 organ dev seeds 1-6 and 8-16 (round 5: seed 7 excluded from the count), reply dev seeds 1-6. The gate
  seeds are run only through a later prereg. Wherever a code path would default to seed 42 (`BRAIN_CHAT_SEED` unset:
  `_brain_chat_seed()`, `gnw_deliberation._DEFAULT_SEED`, `gnw_three_organ_bus._DEFAULT_SEED`; the D4 organ's own
  `seed=42`), the step runs through the stage-2 worker or pins the seed (2.10 step 1, 3.8 step 0).
- **Conflict unit size, before measurement.** With K stored buffer blocks and V'_P distinct stored patients, the unit
  adds (K + 1) * V'_P * 20 + 20 cells (per-block patient lines, shared patient lines, `acc_conf`) and
  K * V'_P * 400 + V'_P * 400 synapses (`d{b}P_w -> pP_w` and `pP_w -> acc_conf` at density 1.0, 20 x 20 per pair).
  For example K = 12 and V'_P = 10 give 2,620 cells and 52,000 synapses. A patient layer over all V words would be
  V/V'_P times larger, which is why the layer uses the reduced patient vocabulary (only the patients that the stored blocks light, 2.3). The
  reduced sequencer is built in every M1 arm even with `integrated_loop` off, so its own size is part of the cost.
- **Measured at step 1 under `tools/mem_ok.sh`** (2.10): the counts read back from the built bridge, the sequencer
  and workspace build memory and time (`_get_bridge` intact and lesioned, `_get_theta`), the per-hop latency, and the
  flag-on against flag-off peak RSS of one arm; a flag-on peak over 8 GB STOPs for a memcap ruling.
- **Corpus (round 4; corrected in round 5).** Every shard this packet scores, or reads for a GO or STOP, runs where
  `data/corpus/tinystories.txt` exists: a pool lane, the workstation pool lane included when its checkout has the
  corpus (it did for the 10-26 gate-off shards). Round 4 said the workstation's main checkout has no corpus; that was
  false. `/home/dant123/Projects/sim/data/corpus/tinystories.txt` exists there (an untracked file, excluded by
  `.git/info/exclude`). The detached `sim-rev` review worktrees lack it, so a read run from one of them records
  `corpus_present` false. Each shard records `corpus_present` (`run_worker` :671). A shard recording false, or
  recording no such key, is UNDEFINED for everything it feeds and is re-run on a corpus lane.
- **LTM bundle (round 5).** The LTM bundle lives outside the repository (`_default_ltm_bundle_dir`,
  `webapp/server.py:3817-3847`: `$BRAIN_DATA_ROOT`, the checkout's sibling `sim-data`, then `~/Projects/sim-data`).
  Every arm of 2.8, M1 arm or flag-off arm, and every arm an R bar reads (the flag-off sides of R4's pairs included)
  needs `process.ltm_bundle` = `wikidata_100k` and `sources` = {`tiny-demo +LTM`} (2.9 rule (i); round 9: scope
  written out).
  A lane that resolves the 15k fallback, or no bundle, gives UNDEFINED shards and is not used. The learned lexicons, the corpus
  vocabulary and the built-in store degrade silently without the corpus, so the branch, nonce and premise checks
  would otherwise pass vacuously.
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

Packet note (round 8, not part of the ruling text above): `--labels` reads `responses[label]` from stage-2 shards over
dev seeds 1-6, because the existing modes' `probe_body` (`tools/answer_rate.py:85-95`) reads only top-level dicts that
carry `answer` and so reports every stage-2 shard UNMEASURED (2.9 B9, 2.10 step 1). Round 9: dev seeds 1-6 are the
default of an explicit `--seeds`; a gate seed is read only with `--prereg` naming a preregistration (the pool guard's
name check), so a later flip prereg can read the answer rate on the gate seeds (2.10 steps 1 and 3).
