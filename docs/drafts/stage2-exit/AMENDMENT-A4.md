# Stage-2 exit gate, Amendment A4: C3's order confound and unacquired moat; C1's named gaps (Claude, 2026-10-13)

Amends `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` (kept byte-identical, so `PREREG_SHA` is
unchanged) together with Amendment A3. It implements Claude's 2026-10-13 ruling on HANDOFF NEEDS-CLAUDE #15, #16 and #17,
and the fixes and rulings of the two-reviewer review of the first A4 commit (`1c934ae9c`, same day; section "Review
record"). Filed before any gate-seed job was queued. No gate-seed data exist (the window is 2026-11-03..11-10). The only
data read are the dev-seed-7 shards, which govern nothing (prereg section 11). Criteria C1, C2, C4 and C5 keep every
bar, arm, turn and seed. C3 gains one counted arm, one informational arm and one undefined rule. Every counted arm of
every group must have run its own registered turn table. A FAIL of C1 also prints the gaps named below.

## Why (dev seed 7, `ask__intact_a.json`)

1. **Order confound.** The DA/engagement crave threshold (`BRAIN_CURIOSITY_DA`, default ON since 2026-08-21) raises the
   ASK pool's effective threshold after every low-engagement abstain. At dev 7 the DA level fell from 0.554 (`g_eel`) to
   0.046 (`m_goat`), and the effective threshold rose from 17.7 Hz to 59.8 Hz. The want rate stayed at 38.9 Hz. After
   the second ask (`g_crab`), nothing asks: `g_newt`, `lp_ibex_1` and every `u_toad_*` were silent at graded novelty
   0.96-0.98. So in the registered order `lp_learnable` and `lp_unlearnable` fail because of the probes before them,
   not because of their own topic, and the position of a block can decide a clause.
2. **Unacquired moat.** The handler did not acquire `the goat eats the hay`. It abstained at role binding ("my
   role-binding didn't resolve the PATIENT"), stored nothing and read `goat` as novel (0.976). The scorer still counted
   `m_goat` as a taught-topic moat.

## A4.1 (a) A taught moat whose teach turn was not acquired is UNDEFINED

- **Acquired** (`teach_acquired`, `acquired_teaches` in the runner) means the teach turn's reply did not abstain AND
  the taught triple `[S, lemma, O]` is among the facts the reply carries: `supporting_facts` together with
  `recalled_svo`. The handler puts only the first fact in `recalled_svo` and every voiced fact in `supporting_facts`
  (dev 7 `t_yak`: yak eat grass, dog chase cat, cat eat fish).
- **Direction (review fix 1).** The first A4 commit read `recalled_svo` alone. A brain that stored the taught triple but
  echoed another fact first then had that moat dropped, so it could ask about topics it knew and escape the moat rate.
  `null_stored_echoed_differently_asks` passed that scorer 1.000 and passes this one 0.000. Reading more of what the
  reply carries can only mark MORE moats acquired, so the fix can only keep moats in the rate. A brain that echoes a
  triple it did not store is still counted as acquiring it; its moat probe then reads as known.
- **Which turn decides (ruling 2026-10-13).** The correctness sessions (`intact_a`, `intact_b`, `reversed`) each read
  their own teach turns. The control arms (`permuted`, `lesion_graded` with its `.rep0`) take the moat set from
  `intact_a`'s acquisition only, by teach label. The permuted arm's `t_goat` teaches `the newt eats the hay`, so `g_newt`
  is undefined there exactly when `intact_a` did not acquire `t_goat`. A control arm's own teach echoes never drop a
  scored turn. An exposure-blind permuted arm therefore stays FAIL "not exposure-driven", never UNDEFINED.
- An unacquired moat is left out of `ask_rate_moat` and **never counted as novel**: the gap set stays the three
  never-taught topics. It is reported per seed as `moats_undefined`, with `moat_items` (topic, acquired, asked,
  counted) and the sub-clause count `all_taught_moats_acquired`.
- **No acquired moat (ruling 2026-10-13, narrowing the first ruling).** The moat rate needs at least
  `ASK_MOAT_MIN_DEFINED = 1` acquired moat. With none, the seed is UNDEFINED only if the missing moat rate is what
  decides it: every other correctness clause holds in every correctness session. That includes the gap-floor half of
  `discriminates` (gap rate >= 2/3), which needs no moat rate. If any other clause fails, the seed is scored incorrect
  (FAIL) and the moat rate is reported undefined (`discriminates` None, `moat_rate_decides` False). The aggregate's
  precondition "the ask group's taught-moat ask-rate defined wherever it decides a correctness session (amendment A4
  (a))" counts only the deciding case.
- The `lesion_graded` arm is scored on the taught moats `intact_a` acquired (`effective_scored_labels`). The lesion
  arm's own echoes are not read. When `intact_a` acquired none, the lesion has no scored turn and its effect is None
  (not an effect). The seed is then decided by `intact_a`'s rule above: UNDEFINED, or incorrect.
- `all_gaps_abstain` still covers all six discrimination probes, including an unacquired moat (unchanged, stricter).
- **Stated honestly:** (a) is a validity correction, not a tightening. It can turn a pre-A4 FAIL into a PASS, but only
  where a "taught" moat was never learned. A curiosity-perfect brain that failed to acquire two of the three taught facts
  (and so asks about them) passed the pre-A4 bar 0.000 of the time and passes this one 1.000 (bar-sanity below). A brain
  cannot reach a PASS by acquiring nothing (`null_no_moat_acquired` 0.000). With one acquired moat the ratio bar
  tolerates no moat ask. With two it tolerates one, and only when the gap rate is 1.

## A4.2 (b) A counted REVERSED-ORDER arm: every C3 correctness clause must hold in BOTH orders

- **Arm `reversed`** (group `ask` only): production defaults, nothing installed, its own session
  (`s2_ask_reversed_<seed>`). It uses the same turns: the teach block first (the exposure the moats need, not a probe
  block), then the three probe blocks in reverse order: unlearnable (`u_toad_1..4` with the dog controls), learnable
  (`lp_ibex_1`, `t_ibex`, `lp_ibex_2`), then discrimination (`g_eel` .. `m_goat`). Each block keeps its own order.
  `_ASK_REVERSED` in the runner; the registered order is pinned by `tests/test_stage2_exit_gate_a4.py`.
- It is a **counted** arm. It is emitted as a pool line, required (missing means UNDEFINED, never FAIL) and validated
  like every counted arm (source, hash seed, corpus, errored turn, organ error, turn table).
- **Turn table (review fix 4), every group.** `arm_validity` requires each counted arm's recorded turns (label and
  message, in order) to equal `turns_for(group, seed, arm)`. The scorers read by label, so otherwise a `reversed` shard
  recorded in the registered order would pass "both orders" vacuously. The same holds for a permuted shard recorded with
  the intact teaching. Such an arm is UNDEFINED under the precondition "every counted arm ran its own registered turn
  table". All 16 dev-seed-7 shards pass it.
- **Correct** for a C3 seed is now `score_ask(intact_a)["all"]` AND `score_ask(reversed)["all"]`. That covers every
  correctness clause of prereg 4.3 (`discriminates`, `lp_learnable`, `lp_unlearnable`, `all_gaps_abstain`,
  `builtin_dog_cat`), plus rule (a), in both orders. A FAIL reads "capability absent on k of 6 seeds (correct per order:
  registered a of 6, reversed b of 6)". The sub-clauses `reversed_all`, `reversed_discriminates`, `reversed_lp_learnable`
  and `reversed_lp_unlearnable` are reported beside it, and the report prints each seed's reversed-order line.
- **Accepted scope (ruling 2026-10-13).** The correctness clauses must hold in both orders. The lesion clauses
  (`lesion_curiosity`, `lesion_graded`, `.rep0`) and the permuted clause are read in the registered order only, since
  there is no reversed lesion or permuted arm. This is accepted: a brain that is correct in the reversed order must ask in
  the final discrimination block, which is where the graded lesion's effect also shows. The reversed arm has no null pair
  of its own; the build's determinism is the intact pair's null at the same seed and revision.

## A4.3 (c) Informational DA-lesion arm (reported, never counted)

The production lesion exists: `BRAIN_CURIOSITY_DA_LESION=1` (`webapp/da_curiosity_drives_chat.py`) pins the DA
modulation to 0, gain 1.0. Arm `da_lesion` (group `ask`, the registered order) reports the ask rates, `discriminates`,
`lp_learnable`, `lp_unlearnable`, `moats_undefined` and `asked_by_label` per seed. It is never a pass condition, never
a precondition and never in the stage verdict. Its question is whether the order confound vanishes without the DA
coupling and whether the unlearnable gap then stops at all, which would show a learning-progress signal.

**Lesion held (review fix 5; docs/TERMS.md "lesion").** The arm reports `lesion_held` (`da_lesion_held`). It is True
only if there is at least one `curiosity.curiosity_da` trace and every trace says `lesioned` True and `da_crave_gain`
== 1.0. If it did not hold, the arm is reported as "attempted lesion (persistence unverified)", NOT HELD, and its C3
reads are withheld: it is never presented as a lesion read.

## A4.4 Reporting rules (no bar change, no wording change)

- **C1 `conv`** (NEEDS-CLAUDE #17 ruling). A C1 FAIL prints a `Named gaps` line (`A4_NAMED_GAPS`) beside prereg
  section 8's sub-bar reading, whichever sub-clause failed: "wh-locative question parse; locative frame storage (the
  teach echo drops the preposition); discourse elaboration for 'tell me more'". Dev 7 shows all three. `where does the
  deer sleep` abstains at the question parse, so the prepositional sub-bar measures the wh-parse before the frame
  inventory. The teach echo `the deer sleeps the meadow` drops the preposition. `tell me more` returns "I don't have
  anything more to add". The probe wording is kept: `deer_sleep` and `more` stay as registered.
- **C3 `ask`** (review fix 6). A C3 FAIL prints, in the per-seed C3 block right after the `da_lesion` lines, "C3 known
  architecture gaps (not the measured cause): global DA engagement threshold; no per-topic learning-progress signal"
  (`A4_KNOWN_ARCH_GAPS`). The line has no ruling as a measured cause. The `da_lesion` read beside it is the only
  measurement in this gate that bears on the first gap.

## A4.5 NEEDS-CLAUDE #15 resolved (C5 private): no C5 change here

Amendment A3's clause stands: the idle fact must be a stored fact (`private_voices_stored_fact`) and must follow the
exposure (`private_selected_by_state`). The "fixed demo lexicon" blocker is gone at any revision with the idle-replay
organ ON. `research/runners/idle_replay_organ.py` voices a fact from the brain's own store in the same remark frame
("Something's been on my mind -- <a> <v> <p>") that the scorer parses. It is default ON on main since `8c10af86a`
(`_IDLE_REPLAY_DEFAULT_ON = True`, PROVISIONAL per its battery prereg of 2026-10-12). **This branch does not contain
`8c10af86a`.** The gate revision must contain it (see Consequences). If that flip is reverted before the gate, C5's
private half reads the demo lexicon again and FAILs "capability absent", as A3 states.

## Lane

Jobs: 186 -> 198 (6 seeds x 33 arms): +6 counted `ask/reversed`, +6 informational `ask/da_lesion`, both `mem_gb=9`,
memcap 10, like every line. The `ask` arm measured 6.3-6.5 GB peak and 751-913 s at dev 7 (workstation). The 12 extra
jobs come to about 3.0 h at workstation speed and about 4.3 h on the pool (~1.4x slower). The gate total becomes about
73 job-hours (prereg section 10 + A2: about 69). `--no-informational` drops `da_lesion` but never `reversed`.

## Bar-sanity (n = 1000 per brain, seed 0; `tools/bar_sanity.py`)

New C3 generators (`bar_sanity_s2_common.py`). The teach turns now echo the stored triple in `recalled_svo` and
`supporting_facts`, as dev 7's handler does. The reversed arm reads the same brain in the reversed order, and the
DA-throttle null applies "silent after two asks" along each arm's own probe order. Pass rates (n = 200) under the
pre-A4 scorer (HEAD `4b19aafb8`), the first A4 commit (`1c934ae9c`) and this one:

| generator | kind | pre-A4 | A4 `1c934ae9c` | A4 (this) |
|---|---|---|---|---|
| `perfect_goat_unacquired` (dev 7's shape: goat novel, asked) | perfect | 1.000 | 1.000 | 1.000 |
| `perfect_two_moats_unacquired` | perfect | 0.000 | 1.000 | 1.000 |
| `null_no_moat_acquired` | null | 0.000 | 0.000 | 0.000 |
| `null_stored_echoed_differently_asks` (moose/goat stored, echoed second, asked anyway) | null | 0.000 | 1.000 | 0.000 |
| `null_order_carried` (registered order perfect, the reversed order DA-throttled) | null | 1.000 | 0.000 | 0.000 |
| `null_order_carried_no_stop` (the reversed order never stops on the unlearnable gap) | null | 1.000 | 0.000 | 0.000 |
| `info_topic_habituation` (asks about an unknown topic at most twice; reported only) | info | 1.000 | 1.000 | 1.000 |

Every other C3 null stays 0.000 and every perfect brain 1.000. The stage spec gains `null_ask_reversed_order_carried`.
C1, C2, C4 and C5 were re-measured at the amended scorer (their records' scorer sha moved). All six records read OK and
are hashed against the scorer at this commit:

bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c3_ask_A4.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_stage_A4.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c1_conv_A4.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c2_infer_A4.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c4_emo_A4.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c5_inner_A4.json

The records they replace (the A3 scorer; stale by sha, never to be cited as the Stage-2 bar-sanity):

bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c3_ask_A3.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_stage_A3.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c1_conv_A3.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c2_infer_A3.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c4_emo_A3.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c5_inner_A3.json

## Dev seed 7 under A4 (reported, not tuned to)

`ask/intact_a` and `intact_b` read: `m_goat` undefined (goat not acquired), moat rate 0 of 2, gap rate 2/3, so
`discriminates` holds; `lp_learnable` and `lp_unlearnable` are False. The registered order alone already fails, and the
prereg's prediction (FAIL on `lp_unlearnable`) stands. The `lesion_graded` arm is scored on `m_yak` and `m_moose`.
The `ask/reversed` dev smoke (`dev/s7/ask__reversed.json`: valid, peak RSS 6.5 GB, 751 s; provenance in the WORKLOG)
shows the confound in mirror image. `u_toad_1` and `u_toad_2` ask (DA 0.554, then 0.246); from `u_toad_3` on, the DA
level is 0.046 and the effective threshold 59.8 Hz, so nothing asks: `lp_ibex_1`, every gap and every moat are silent
(gap rate 0, `discriminates` False). `lp_unlearnable` reads True there, but for the same history reason (the threshold
rose), not a learning-progress signal. Each order passes the clauses the other fails, so neither order alone carries C3,
which is what A4.2 requires. These reads are identical under the review fixes.

## What this does NOT establish

- **The residual loophole is per-topic habituation (review fix 7).** A brain that asks about an unknown topic at most
  twice passes 2 of 2 in both orders. It asks each gap once, asks the ibex once and then learns it, and asks the toad on
  probes 1 and 2 but not on 4. That brain has no learning-progress signal (`info_topic_habituation`, reported only,
  passes 1.000). A per-session ask budget does NOT pass. In the reversed order the toad probes come first, so a budget
  that stops the toad by probe 4 is spent before the ibex and the discrimination block. A PASS therefore says "the brain
  asks by topic history, at least by count", not "a learning-progress organ".
- The DA-lesion arm is one informational read and attributes nothing.
- Rule (a) trusts the teach reply as the read of what was stored. A brain that voices a triple it did not store is
  counted as acquiring it; its moat probe then reads as known.

## Review record (2026-10-13; two reviewers of `1c934ae9c`, one MERGE, one FIX_FIRST, no blocker)

Applied: (1) acquisition reads `supporting_facts` together with `recalled_svo`, plus the null above; (2) control arms
take `intact_a`'s moat set; (3) an undefined moat rate never masks a FAIL another clause decides; (4) the turn-table
validity rule; (5) `lesion_held` on the DA-lesion arm; (6) the C3 line reworded as known architecture gaps beside the
DA-lesion read, not named gaps; (7) the habituation residual and the accepted scope; (8) the lane figures; (9) the dev
shard's provenance note (WORKLOG); (10) a test that pushes a C1 FAIL through the report's `Named gaps` line.

## Consequences for the packet

- `PREREG_SHA` is unchanged (`24dac499e`). The prereg file is not touched.
- **`GATE_SHA` (`43763e637`) is void for the gate.** That revision has no `reversed` arm, so every `ask/reversed` pool
  line would die there. The gate must be re-pinned at a revision that contains this file, the amended runner and tests,
  and main at or after `8c10af86a` (idle replay ON, A4.5). The check: `git cat-file -e $SHA:docs/drafts/stage2-exit/AMENDMENT-A4.md`
  and `git merge-base --is-ancestor 8c10af86a $SHA`. Claude did not write `GATE_SHA`; Hermes re-pins it after merging
  main (HANDOFF section 3).
- No dev line is re-run. The worker, turns and arms of the registered arms are unchanged, so every existing dev shard
  re-scores under the new code.
