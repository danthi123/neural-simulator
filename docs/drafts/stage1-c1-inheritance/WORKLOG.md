- 2026-10-06 17:40: step 1 done: 104 staged files under _stage1_c1_category/gate committed on main
  (RULING c1-hop-first: step 1 only; nothing else queued for C1).
- 2026-10-07 12:52: step 2 done (RULING c1-hop-handoff): READY (claude-review/c1-hop ancestor of main,
  HOP_FLAG present); DEV_SHA=PIN 2c2d55952 (main 676f27725 differs by doc-only commits: pin->main delta
  under research/runners, tests, tools is EMPTY); pytest 12 passed / 30 deselected; both selftests OK.
  Provision record for the pin: state=ok rc=0 on pool41/42/40 (+wspool marker).
- 2026-10-07 12:55: step 3: flip-dev canary (seed 1, side on, group cat_disc) is ALREADY RUNNING on
  pool42 since 12:16:24 - pool_queue.sh add refused it as a running duplicate (correct guard). Waiting
  for research/findings/raw/_stage1_c1_flipdev/on/s1/cat_disc__intact_a.json to land and pass the
  step-3 checks; the 95 flip-dev lines and the 54 gate_hop lines stay unqueued until then.
- 2026-10-07 13:34: step 3 canary LANDED and PASSED every HANDOFF expectation: 16 turns, flag_value 1,
  hop 1 in flip_dev, sources ['tiny-demo +LTM'], corpus True, trace on every turn True, canary
  False None ['canary', 'eat', 'worm'], read hop True 30.
- 2026-10-07 13:39: the other 95 flip-dev lines queued (94 in queue + 1 dispatched at 13:34; add loop
  exit 0, no ADD-FAILED). The 54 gate_hop lines were REFUSED by the pool-canary guard: no
  (stage1_c1_category, 2c2d55952) record in pool_canaries.tsv (expected - the guard exists for exactly
  this case, FAILURE_LOG 2026-09-30). Per the refusal's own remedy and RULING canary-node, launched
  `pool_canary.sh run pool40 '<gatehop line 1>' --wait-s 5400` (seed 42 gate line, output redirected
  to _pool_canary/, verdict recorded to pool_canaries.tsv). The 54 adds follow the PASS.
- 2026-10-07 14:05: gate-seed canary PASS recorded: (stage1_c1_category, 2c2d55952) PASS pool40 gate,
  rc=0 wall=1297s json=1 errors=0 (pool_canaries.tsv). 14:32: starting the 54 gate_hop adds against
  the satisfied guard.
- 2026-10-07 14:35: all 54 gate_hop lines queued (added 54, loop rc=0, no ADD-FAILED, queue depth 181).
  Both waves now out: 54 gate_hop + 95 flip-dev (92 waiting, 3 dispatched/landed by 14:32). Step 4
  scores when cells land (or 120 h after queueing); step 5 writes [c1-gatehop] and [c1-flipdev].

- 2026-10-07 21:05 (Hermes) — GAP_CLOSURE Owed item done: the 2 missing s42 wiring-on gate cells
  (`gate/s42/cat_disc__lesion_lateral_learn.json`, `gate/s42/cat_disc__lesion_lateral_read.json`) re-queued at pin
  `6c2cb1362b34d08d5be1807ce82ad333f221f84c` from `research/queue/requeue_stage1_c1_category_6c2cb1362.tsv`
  (54-line battery; the HANDOFF add-loop skipped the 52 existing outputs and added exactly 2). Canary:
  `stage1_c1_category @ 6c2cb1362` PASS recorded (pool42, 09-30). pool_queue reported both original claims
  (2026-10-04 23:42) dead → queued, depth 140→142. gate/ now 52 landed + 2 re-queued = 54/54 accounted.

- 2026-10-07 23:35 EDT (Hermes, driver) — CORRECTION: my 10-07 21:05 re-queue of the 2 missing
  s42 wiring-on cells (lesion_lateral_learn + lesion_lateral_read at 6c2cb1362, committed
  c0d675504 as a GAP_CLOSURE "Owed") was barred by RULING pool-hold-c1-closed (10-06: "Do not
  re-add them; if they land anyway, commit them with the shards, no re-score"). Withdrew both
  queued lines this wake (pool_queue.sh withdraw _stage1_c1_category/gate/s42, expect 2, logged
  in research/queue/withdrawn.log). Neither had dispatched. The GAP_CLOSURE board entry that
  asked for the re-queue is stale; NEEDS [c1-requeue] filed for Claude to fix the board.

- 2026-10-08 10:05 (Hermes) STEP 4 (gate_hop scored): all 54 gate_hop cells landed (workstation 26 +
  pool dl4g243/dl5d243/mxl1462drl 11/9/10; the aggregate row = this server). Score from $P (runners
  byte-identical 2c2d55952<->main): `stage1_c1_category --aggregate .../gate_hop --hop --seeds
  42,43,44,100,101,102` -> **C1 gate_hop PASS** (correct 6/6, lesion effect 6/6, reproduced 6/6,
  permuted 6/6; flip-readiness READY: safety 6/6, specificity 6/6; informational arms: off inherits
  0/6, changes a C1 probe 6/6, lesion_lateral_learn 0/6, lesion_lateral_read 5/6, lesion_draw 6/6).
  aggregate.json + provenance at raw/_stage1_c1_category/gate_hop/. Validity is host-class
  independent (same pinned revision's runner; provisioned LTM/corpus on every node; where_ran NOTE
  honoured: cited counts span workstation+pool shards, no provisioner-absent input).
  [c1-gatehop] RESULT filed with DEV_SHA 2c2d55952.
  STEP 4 (flipdev): `c1_flip_dev_check --score raw/_stage1_c1_flipdev` -> UNDEFINED (0/8 defined):
  every dev seed's on/cat_told, on/lived, on/val_novel report flag_on_not_engaged (flag-on turns
  missing the category_inherit trace), seed 4 additionally off/word + on/lived missing. Remedy
  (HANDOFF 4 + scorer docstring), run ONCE: the 2 missing cells re-queued from
  c1_flipdev_2c2d55952.tsv (both had died on wspool 00:17), and the invalid groups' cells (46 files
  incl. off-side pairs) moved to raw/_stage1_c1_flipdev__invalid_attempt1/ and re-queued -> 48
  flipdev lines queued (no /usr/bin/time in this tsv). Rescore + [c1-flipdev] RESULT next wake when
  the re-runs land; if flag-on still does not engage, that is a defect verdict (no further re-run).

- 2026-10-08 10:30 (Hermes) STEP 4/5 (flipdev rescore after the one permitted re-run): the 46 invalid
  cells re-ran and landed (attempt-1 originals kept on disk at raw/_stage1_c1_flipdev__invalid_attempt1/,
  deliberately NOT committed: their .prov sidecars were replaced by the re-runs, so the sidecar-less
  copies fail device-and-cost + corpus-check honestly — the committed re-run outputs carry the same
  violation set; only s4's off/word + on/lived remain queued — they cannot change the verdict). Rescore = engagement
  STILL broken, identical violation set: on-side reply fields missing the category_inherit stamp —
  cat_told `t_bird_eat` (8/8 seeds), lived `p_wolf_discuss` (7/8), val_novel `feel_*` (6 fields x
  8 seeds = 48); off-side 0 wrongly stamped (control clean); bridge = worm guess on 8/8 dev seeds
  (canary_eat_on). Engine intent is to stamp every flag-on reply (webapp/server.py:6934/7313,
  "per-turn trace … absent when off"), so cat_told/lived = real reply-path bypasses; val_novel's
  feel_* fields may be non-turn meter readouts — whether they count as turns is bar/check
  interpretation (Claude's). Per the scorer docstring ("re-run that cell once; otherwise FAIL naming
  each violation"): [c1-flipdev] RESULT FAIL filed with DEV_SHA 2c2d55952. The gate_hop PASS stands;
  the flip stays blocked on this dev-bar failure (next method = Claude's call, RULING c1-gatehop).
- 2026-10-08 14:06 (Hermes) DIAGNOSE (RULING c1-flipdev-diagnose): one dev seed (s1), the failing
  cell (cat_told), flag-ON, run with the live per-turn category_inherit trace read straight from
  the chat object (server `_BRAIN_CHATS` + `category_inherit_chat.turn_trace`) — script + log kept
  at scratch/fx1_run.py, fx1_run_seed1_cat_told_on.json (the landed cell's own wall was 990s;
  this run reproduced it). Three answers: (1) the inherit pathway did NOT fire for t_bird_eat —
  its observe row still carries the PREVIOUS turn's tuple (svo salmon/isa/fish, member_slot 3 =
  t_salmon_isa's row) while t_fish_eat afterwards got member_slot 4, i.e. nothing was ever
  observed/stored for bird/eat/worm, and read = null (teach turn). All other turns' traces are
  correct: robin→canary observe rows slots 0-5, c_bird_eat read reason=agent_never_observed
  (downstream of the same gap), probes read with rates. (2) The flag changed nothing for this
  turn: t_bird_eat's flag-ON reply is byte-identical to its flag-OFF reply (same 14 keys incl.
  not_understood/repair/abstained, same body) — the abstain answer is identical either side.
  (3) The stamp belongs at webapp/server.py:6265: the comprehension-repair payload (built
  6239-6252, returned 6265, its own comment says "RETURNS here, before the main assembly sites")
  exits before both stamp sites (6934-6935 rich, 7313-7314 single-fact), so an abstaining turn
  carries no category_inherit key. Trigger note: role-binding failed on "the bird eats the worm"
  (patient unresolved) while the structurally identical "the fish eats the bug" bound fine.
  Reading: a missing reply stamp + missing observe record on the ABSTAIN path, not a missing
  flip behaviour — the pathway behaves on every parsed turn. No flip, no scorer change (RULING).

## 2026-10-10 23:47 - flipdev rerun (c1-flipdev-rerun) scored: UNDEFINED, bridge bar clean
Both arms complete (48/48 cells each; 3 stale `.part.json` leftovers from retried lines, final
files all present; no re-queue). Scored with the UNCHANGED scorer: `flipdev_score.json`
(provenance stamped). **Verdict UNDEFINED, n_pass=None/8 (0 of 8 seeds defined).**
- Two numbers the ruling asked for: pathway FIRED in **16/48 ON cells** (bridge group 8/8 +
  cat_disc and word on seeds 5-8); NOT ENGAGED in **32/48 ON cells** (`flag_on_not_engaged`:
  all cat_told/lived/val_novel across all 8 seeds; cat_disc/word on seeds 1-4).
- **Pass count: UNDEFINED** - every seed has >=3 not-engaged groups, so no seed is defined
  and the bar cannot PASS or FAIL.
- **Miss pattern: systemic non-engagement, NOT the t_bird_eat slot-collision cluster.** The
  engagement precondition (a strict recall miss on that turn) simply does not trigger in
  cat_told/lived/val_novel (any seed) and rarely in cat_disc/word (fires seeds 5-8 only).
- Bridge group (the part that IS defined): prereg behavior is exact - `bat swim` -> flagged
  guess `lake` on **8/8 ON seeds**, `canary swim`/`robin swim` (two hops) unchanged on all
  seeds, OFF side silent on all three; canary-eat C1 probe fired (worm) 8/8 ON, null 8/8 OFF.
- Where: workstation 34 + pool (dl4g243 11, dl5d243 11, mxl1462drl 43) + server 1 (canary);
  validity does not depend on host class (scorer reads recorded replies only).
STOP honored: report only, no fixes, no scorer change, no re-run (this was the one re-run).
NEEDS [c1-flipdev] RESULT filed to the runbook. Next call (C1 method question: why the
engagement precondition rarely fires) = Claude.

## 2026-10-11 06:12 EDT - tree-c1-engage step 1: trigger-vs-threshold audit (zero-compute, rerun cells)

- Audited all 48 ON cells of `research/findings/raw/_stage1_c1_flipdev_rerun_2c2d55952/` per turn: whether
  `category_inherit` is stamped, and the read's `reason` / rates. Dominant cause = **branch (a): the signal never
  reaches the organ**, not a threshold miss:
  - val_novel: 72 reads `organ_not_built` (no organ existed: nothing was ever observed in-session), 62 responses have
    no trace at all.
  - lived: 16 `organ_not_built`, 16 `agent_never_observed`, 8 `below_floor`; 23 responses untraced.
  - cat_told: 8 `agent_never_observed`, 16 `below_floor` (rates present), 16 untraced.
  - word: 8 `organ_not_built`, 8 `agent_never_observed`, 8 `below_floor`.
  - Engaging categories are clean: cat_bridge 136/136 traced (16 reached / 16 below_floor with rates), cat_disc
    reached 8 / below_floor 8.
- Route traced: the observe path is `webapp/category_inherit_chat.py::_ci_gate_extract` (affirmative gate-extract
  triples only, line 109); `_ci_gate` returns `organ_not_built` when no organ was ever fed (line 127-129). Told /
  lived / val_novel signals arrive by channels that never pass through that observe path.
- Per RULING tree-c1-engage branch (a): next = wire the gap (default-OFF path), unit test, rerun flipdev on dev
  seeds, then RESULT. Branch (b) margins recorded above for the below_floor turns.

## 2026-10-11 07:10 EDT - tree-c1-engage branch (a) wire: built + tested, commit gate-blocked (ASK filed)
The branch-(a) wire is implemented as the default-OFF path `BRAIN_CATEGORY_INHERIT_DIRECT_CONSULT` (webapp/
category_inherit_chat.py::_ci_gate): when "1", the inherit organ is consulted on EVERY strict-recall route,
including turns the production bus answers directly (the short-circuit the audit showed kept told/lived probes
from ever reaching the organ). Production behaviour is untouched: the env defaults off and an organ abstain
keeps the direct answer. 5 new unit tests in tests/test_category_inherit_organ.py (default-off, engage-on-
committed + anaphora-direct, off = production short-circuit kept, abstain keeps the direct answer); 32/32 pass
in 38.89s (tests/test_category_inherit_organ.py + tests/test_c1_flip_dev_check.py).
NOT YET ON MAIN OR ON THE BRANCH COMMIT: the code commit is deterministically blocked by the pre-commit
registry - run_all reports corpus-check-required's own selftest ghost (raw/g.json, 'selftest fallback probe'),
consumer-hardware-reference's informational VRAM line and lane-starvation 4/5 as blocking, although every one
of those gates exits rc=0 when run standalone on the same worktree. The two files are STAGED, uncommitted, in
.claude/worktrees/c1-direct-consult (branch research/c1-direct-consult, from main 8b300d109). Gates are
Claude-reserved; NEEDS-CLAUDE `[c1-direct-consult-gates]` ASK filed. Next (after Claude lands the commit):
touch research/queue/provision.req/<branch-tip-sha> for the pool revision, then queue the flipdev rerun on dev
seeds with BRAIN_CATEGORY_INHERIT_DIRECT_CONSULT=1 on the ON-side lines and score it per the tree.
2026-10-11 07:07 EDT - tree-c1-engage step 2: branch (a) wire built (BRAIN_CATEGORY_INHERIT_DIRECT_CONSULT, default OFF) + 5 unit tests, 32/32 pass, STAGED on research/c1-direct-consult; run_all gate-blocked (corpus-check ghost, consumer-hardware info line, lane-starvation) -> [c1-direct-consult-gates] ASK. RULING c1-direct-consult-hold: do NOT land; files kept unstaged on the branch as the record; branch synced with main (1f3215580).

2026-10-11 08:00 EDT - tree-c1-observe (a): teach-observe wire at the store write (ChatBrain._ci_teach_observe on both _maybe_acquire hear paths), BRAIN_CATEGORY_INHERIT_TEACH_OBSERVE default OFF, AFFIRM-only, exactly-once vs gate_extract (wrapper guard added, byte-identical when off); 25 tests in test_category_inherit_organ.py + 54 in the 4 touching files all green; committed 220c1213c on research/c1-teach-observe; provision.req filed (smoke pair queue refused until provisioned -> queues next wake). Unit tests carry the ruling's spec: told + lived one teach turn -> organ exists AND observed that agent: PASS.

RESIDUAL val_novel (reported, NOT silently passed): the expose sentence is 5 content words, the B3 extractor declines it, so NO store write exists to hook; kb stays 5/5 across all 12 exposures; 0/8 ON cells ever observed; pinned by test_val_novel_exposure_writes_no_store. The exposure->observe site is a design call -> [c1-probe-bar] ASK.

(b) direct-hit counts (turns with NO category_inherit trace = organ never consulted), ON side, 8 seeds: val_novel feel 48/48 and probes 14/16; lived probes 15/18; word probes 12/24 and controls 4/24; cat_told probes 9/18 and controls 9/18; cat_disc probes 4/16 and controls 8/24; cat_bridge 0/32 (fully consulted). Teach-role turns traced 100% everywhere.

(c) scorer verdict, REPORTED not changed: flag_on_not_engaged fires only when SOME turn carries NO trace; organ_not_built/agent_never_observed reads DO carry a trace and are therefore counted as ENGAGED. The 32/48 failures are the NO-trace bypass turns (chain-route, open-ended, episodic, feel, and bare-word endpoints answering before chat.gate).

PROPOSED TREE for [c1-probe-bar]: if direct hits must consult the organ -> wire organ consult on the bypass endpoints behind the same default-OFF flag, then rerun flipdev; if a direct-knowledge answer may count as inheritance -> the scorer stands and the UNDEFINED verdict is honest. val_novel: if an exposure must build the organ -> approve an exposure->observe hook (new design); else organ_not_built reads there stay by construction.
2026-10-11 08:05 EDT - MiMo Pro consult (Q1 bar): direct hits must NOT count as inheritance-consulted — a direct-knowledge answer is byte-identical with the flag off, i.e. zero evidence about inheritance, and counting it would let failing cells pass while the organ does nothing. Proposed bar: split turns into IN-SCOPE (SVO-shaped routable strict misses the organ's input class can consume) vs OUT-OF-SCOPE direct-knowledge (chain/open-ended/episodic/feel/bare-word); engagement = an in-scope turn whose observe fired for that agent or whose read reached a patient; organ_not_built/agent_never_observed are NOT engagement; out-of-scope turns become a separate per-group direct-hit coverage stat.

2026-10-11 08:05 EDT - MiMo Pro consult (Q2 val_novel): keep organ_not_built BY CONSTRUCTION — fabricating a triple at an exposure hook is the forbidden host shortcut (the brain's own extractor rejected the utterance; no SVO exists). If the flip must test val_novel, fix the STIMULUS: the teacher (legitimate host role) delivers an explicit AFFIRM-SVO teach turn during exposure ('the glorp eats the wug' form) so the existing legal path fires. (Q3): no new bypass wiring before Claude lands the bar + probe set; then re-score the HELD DIRECT_CONSULT restricted to in-scope SVO-shaped bypass turns only. Transcript: /opt/data/cache/delegation/live/deleg_6f6e6fda/task-0.log
2026-10-11 08:20 EDT - provision 220c1213c OK on pool41/pool42/pool40/wspool; flipdev smoke pair QUEUED first (priority lane): cat_disc group, dev seed 1, both sides, BRAIN_CATEGORY_INHERIT_TEACH_OBSERVE=1, out research/findings/raw/_stage1_c1_teachobs/<side>/s1/, mem_gb=7 / memcap 9, max_age_h=72. Smoke-first: record peak RSS + wall from the result sidecar when it lands, then set the batch mem_gb=ceil(1.3 x peak); the 96-cell dev batch queues only after the smoke returns clean.
2026-10-11 08:35 EDT - Claude rulings c1-bar-A1 + c1-valnovel received (2de2d6dbe): direct-hit probes are NOT inheritance evidence; engagement = in-scope turn where observe fired or the read reached a patient; AMENDMENT C1-A1 must be registered in the C1 prereg BEFORE the 96-cell batch (prereg = section-2 reserved -> driver HOLD, filed [c1-batch-a1] STOP); val_novel = teacher delivers explicit AFFIRM-SVO teach turn during exposure, registered inside A1, natural-sentence exposure stays a reported secondary; DIRECT_CONSULT stays held. Smoke pair DISPATCHED 08:12:44 pool40 (on side) / 08:12:51 pool41 (off side), cat_disc seed 1, revision 220c1213c sandboxed + memcap 9G; bridges up 08:27 (on: 2760 neurons/190317 synapses; off: 1044 neurons/43707 synapses); job_status row lands only on completion. Next per tree: smoke clean + A1 committed -> queue the 96-cell dev batch with TEACH_OBSERVE=1, score under A1, outcomes (a/b/c).
2026-10-11 08:55 EDT - RULING tree-c1-rescore executed: c1_flip_dev_check.py updated to Amendment B-1 (registered in PLAN.md 45bee4aa3). Engagement = flag_value recorded + >=1 trace per ON cell (the old every-turn rule is gone); untraced bypass turns REPORTED per group (seed 1: cat_disc 3, cat_told 2, word 4, lived 4, val_novel 8 labels); an untraced turn whose reply changes FAILS the seed (why untraced_turn_changed, no ALLOWED exemption); query-shaped organ reads (carrying lesion) must record hop:true, while the organ_not_built DECLINE record is not a read (an intermediate strict check briefly made all 8 seeds UNDEFINED on word/lived/val_novel before this fix); flag-off unchanged; the bar text (a)(b)(c)+hop+PASS/UNDEFINED/FAIL unchanged.
2026-10-11 08:56 EDT - Same commit: selftest 27/27 including the B-1 trio; bar_sanity_record_flipdev.json regenerated with the new scorer (OK: perfect 1.000, null 0.000, n=1000, seed 0). Rescore of the COMMITTED rerun cells, no new run -> OUTCOME (a): PASS 8/8 seeds defined+pass, violations 0, correct 8/8, bridge_correct 8/8, bridge moat 8 of 8 defined. Artifacts: research/findings/raw/_stage1_c1_flipdev_rerun_2c2d55952/flipdev_score_B1.json - Claude reads it for the flip (PLAN step C). The cancelled smoke pair had already completed rc=0 on both sides (pool40 on / pool41 off, cat_disc dev seed 1, branch 220c1213c) before the cancel arrived; teach-observe stays parked as a section-5 strengthening item and its batch is cancelled.
2026-10-11 09:28 EDT - _earn check 6 initially re-implemented the bridge moat clause as n_seeds_with_moat_guess_on == n_defined; the registered decide_dev never enforces the moat (report-only), so that invented rule downgraded the bar-sanity perfect brain to UNDEFINED (perfect passes 0.000). Check 6 now mirrors decide_dev (every defined seed passes the bar). NOTE for Claude: the real rerun cells carry n_moat_guess_on > 0 on 8/8 defined seeds while decide_dev PASSes - a docstring-(b)-vs-code gap in the registered step, NOT changed here (bar semantics are yours).
