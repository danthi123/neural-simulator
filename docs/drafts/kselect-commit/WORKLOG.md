# k-select spiking commit: WORKLOG

*Claude build agent, 2026-10-14, branch `claude-build/kselect-commit` from S main 6bb27d920. Task: retire the k-select
live seam with the spiking commit stage (`docs/drafts/spiking-commit/PLAN.md` section 4, A3). Dev seeds only.*

## 2026-10-14: STEP 1 design check. Verdict: STOP. The candidates are host-evaluated; no build.

Read first: PLAN.md (all), the A1 commit code (`research/runners/word_role_parser_organ.py` 175-245, 494-640),
`2026-10-02-bg-live-seam-retirement-RESULT.md`, `2026-10-02-host-share-baselines-RESULT.md`,
`research/runners/host_share.py` 238-306 (the k-select wrappers), `docs/drafts/seam-map/CLAUDE-REVIEW-2026-09-29.md`
(row A5 and the K-SELECT row; the `host-decision-seam-map.md` the packet names is not on S main),
`2026-10-04-stage1-host-share-RESULT.md` (C3 block), `2026-10-04-stage1-exit-gate-RESULT.md` (C3 review),
`research/findings/raw/_spiking_sequencer_smallvocab_scoping.md`. `tools/before_you_build.sh` and the RAG surfaced no
prior k-select commit attempt.

### Where k-select is (file:line, S main 6bb27d920)

- Default path. `OneBrainComposer.query_patient` (`research/runners/one_brain_composer.py:1757`) calls
  `_fact_shard_first_match` at 1776, because the DG-CA3 fact shard is default on (`_FACT_SHARD_DEFAULT_ON = True`, 363)
  and `_fact_shard_active` (1371-1377) holds while `integrated_loop` is off. The decision is
  `one_brain_composer.py:1458-1461`: `for i in shard: got = self._read_one_block(i); if all(got.get(r) == w ...): return i, got`,
  then `(None, None)` (abstain). This is a host loop in ascending block order with a host string equality.
- Escalation path. `_seq_block` (1655), with `integrated_loop` off: the same first-match loop over `_read_blocks()`
  (1660-1665).
- The competing alternatives are the shard's candidate block indices (`_fact_shard_candidates`, 1413-1436). The shard
  is a host `DGSparseIndex` projection (a declared host-rate stand-in, comment at 352-358). Each candidate is decoded
  one at a time through the SAME shared Q and cleanup neurons (`_read_block`, 1088): kick the block's trigger, resonate,
  unbind, cleanup. The cleanup word is picked by `_spiking_select` (1042-1072): the host reads the cleanup membrane,
  normalises it, drives a separate Izhikevich bank and takes the argmax over its firing (with an argmax fallback).
- On the measured probes the choice is not even a competition. In the Stage-1 C3 shards
  (`research/findings/raw/_stage1_host_share/gate/s42/s1_lived__k-select_intervene.json`) every applied k-select event
  has exactly one decoded match (`decoded_matches [5]` wolf-bite-apple, `[3]` dog-chase-cat). The intervention is a match
  flip, so k-select at C3 is "this one block matches, commit or abstain", decided by a string compare.

### Does each candidate have its OWN spiking population whose activity reflects recall?

**No, on the production path.** A stored block owns an RF region (`store_base + i*block`: a trigger and D phasor
neurons), but that region is active only while the host loop kicks it, one block at a time, and its activity does not
depend on the cue. The cue meets the block content only after the decoded words reach the host, as `got.get(r) == w`.
Nothing per candidate fires concurrently with the others, nothing competes, and the per-candidate evidence is a host
boolean. Driving commit gates from those booleans (or from cleanup margins) would make new spiking sources from host
numbers. PLAN section 4 A2 names this defect, so per the packet I STOP here.

**The one off-default exception, and why it does not change the call.** The K-way sequencer (`integrated_loop` True or
`"fused"`, production flag `BRAIN_INTEGRATED_LOOP`, default off) has real per-candidate spiking pools: `m{b}`
(gated-disinhibition cue match per block), `ans{b}`, `inh{b}`, and a tonic abstain. Its read is still host:
`run_sequencerK_with_drive` takes `next(b for b in range(K) if m_rate[b] > match_thresh)`
(`research/runners/_phaseB_onebrain_sequencerK_divnorm_derisk.py:135-141`). That is the A1 pattern, and the commit stage
could replace it. But:
1. The per-block content reaches `m{b}` as host arrays cached at write time (`_seq_drives`, built in `_ensure_sequencer`,
   `one_brain_composer.py:1646-1653`, rebuilt only when a write sets `_seq_dirty`; the fused path caches `_fused_drives`
   the same way). They are written as currents onto another bridge. At query time the sequencer never touches the store:
   its "recall" is a replay of an earlier decode. This is the A2 cross-bridge relay, upstream of the competition.
2. Flag-on would move production off the default fact shard (the shard requires `integrated_loop` off) onto a path that
   over-abstains on the small-vocab codes (06-25 scoping: the agent-role cleanup reads zero on at least half the blocks).
   Host-share SC for k-select (the sequencer substituted) is 0.233 at baseline and 0.167 at C3. So the sequencer changes
   the reply on about one k-select event in five. A reply change under `BRAIN_KSELECT_COMMIT` would confound "the commit
   decides" with "the sequencer replaced the shard".
3. Wiring the commit there would retire the sequencer's read, not the default path's first match, which is what the
   host-share measured (HS 1.000 at baseline, C3 panel 0.967).

### Options for Claude (a design call, not built)

- **O1, commit on the K-way sequencer (cheapest, partial).** Sources `X_b = m{b}` on the sequencer or fused fabric
  bridge, using the factored stage. This replaces the `m_rate > match_thresh` and lowest-index host read. Honest only with
  (a) the block drives recomputed from the store per query, not cached, and (b) the sequencer's agreement with the shard
  path established first (the small-vocab over-abstain fixed: the 06-25 scoping ranks code margin as the cheap lever).
  Even then the content crosses bridges through the host, so it retires the read, not the relay.
- **O2, per-block cue-match assemblies on the composer's own bridge (the real fix; a new mechanism).** Present the cue,
  bind(agent role, a) plus bind(action role, x), to every shard block at once. Each block projects through a matched
  filter (its stored composite) onto its own match assembly, so each assembly fires in proportion to its block's match
  with the cue: a parallel content-addressed match (CA3-style completion; the 06-25 scoping's biology pointer). Cross-
  block lateral inhibition, then the commit stage (G, GI, S) on the same bridge, read as which latch is on after a
  zero-input hold. Open feasibility question to check first: whether one bridge steps the RF store and Izhikevich
  `cp_connections` populations together (the fused fabric notes it runs as a separate plain Izhikevich bridge, not the RF
  step). Needs its own prereg. This is A3 done as the PLAN intends ("competing slot assemblies driven by recall"), without
  the host relay.
- **O3, ruled out:** commit gates driven from the shard's decoded-match booleans or the cleanup margins. That only moves
  the first match into a host rate relay.
- **Sequencing note.** Phase A's "one live seam retired" exit item may land sooner on a seam whose alternatives already
  compete on one bridge. A2 (wm-focus-bind, the PLAN's declared next) qualifies after its own step (a). So do the GNW unique-
  ignition read and the BG commitment race. k-select via O2 is a mechanism build, not a wiring task.

### Host steps in the k-select path today (declared, for whichever option is chosen)

DG shard projection (host `DGSparseIndex`); the sequential per-block decode loop; the cleanup winner (a host-relayed
Izhikevich bank, argmax over firing, argmax fallback); the cue match (string equality); first-match order (ascending
block index); the clause/attribute routing reads the host `kb` (`_finish_query_patient`, 1792). Only the RF
resonate/unbind/cleanup matvec is on the substrate.

No code changed. Steps 2-4 (factoring `spiking_commit.py`, the flag, the prereg, the HANDOFF) are not started, pending
the design call.
