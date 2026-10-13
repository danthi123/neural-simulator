# Stage-2 exit gate, Amendment A4: C3's order confound and unacquired moat; the C1 / C3 named gaps (Claude, 2026-10-13)

Amends `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` (kept byte-identical, so `PREREG_SHA` is
unchanged) together with Amendment A3. It implements Claude's 2026-10-13 ruling on HANDOFF NEEDS-CLAUDE #15, #16 and #17.
Filed before any gate-seed job was queued. No gate-seed data exist (the window is 2026-11-03..11-10). The only data read
are the dev-seed-7 shards, which govern nothing (prereg section 11). Criteria C1, C2, C4 and C5 keep every bar, arm, turn
and seed. C3 gains one counted arm, one informational arm and one undefined rule. A FAIL of C1 or C3 also prints the gaps
named below.

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

- **Acquired** (`teach_acquired`, `acquired_topics` in the runner) means the teach turn's reply did not abstain and its
  `recalled_svo` (the handler's echo of what it stored) is exactly the taught triple `[S, lemma, O]`. It is read per arm
  from that arm's own teach turns: `intact_a`, the reversed arm and the permuted arm (whose moats are eel/crab/newt).
- An unacquired moat is left out of `ask_rate_moat` and **never counted as novel**: the gap set stays the three
  never-taught topics. It is reported per seed as `moats_undefined`, with `moat_items` (topic, acquired, asked,
  counted) and the sub-clause count `all_taught_moats_acquired`.
- The moat rate needs at least `ASK_MOAT_MIN_DEFINED = 1` acquired moat. With none, the rate does not exist and the seed
  is UNDEFINED for C3, never FAIL. The aggregate gains the precondition "the ask group's taught-moat ask-rate defined in
  every arm that reads it (amendment A4 (a))". An acquisition failure is a comprehension gap (C1's), not a curiosity
  result.
- The `lesion_graded` arm's scored turns lose any taught moat that was not acquired in both `intact_a` and the lesion
  arm (`effective_scored_labels`). This only removes turns, so the lesion clause can only get harder to meet.
- `all_gaps_abstain` still covers all six discrimination probes, including an unacquired moat (unchanged, stricter).
- **Stated honestly:** (a) is a validity correction, not a tightening. It can turn a pre-A4 FAIL into a PASS, but only
  where a "taught" moat was never learned. A curiosity-perfect brain that failed to acquire two of the three taught facts
  (and so asks about them) passed the pre-A4 bar 0.000 of the time and passes this one 1.000 (bar-sanity below). A brain
  cannot reach a PASS by acquiring nothing (UNDEFINED; `null_no_moat_acquired` 0.000). With one acquired moat the ratio
  bar tolerates no moat ask. With two it tolerates one, and only when the gap rate is 1.

## A4.2 (b) A counted REVERSED-ORDER arm: every C3 clause must hold in BOTH orders

- **Arm `reversed`** (group `ask` only): production defaults, nothing installed, its own session
  (`s2_ask_reversed_<seed>`). It uses the same turns: the teach block first (the exposure the moats need, not a probe
  block), then the three probe blocks in reverse order: unlearnable (`u_toad_1..4` with the dog controls), learnable
  (`lp_ibex_1`, `t_ibex`, `lp_ibex_2`), then discrimination (`g_eel` .. `m_goat`). Each block keeps its own order.
  `_ASK_REVERSED` in the runner; the registered order is pinned by `tests/test_stage2_exit_gate_a4.py`.
- It is a **counted** arm. It is emitted as a pool line, required (missing means UNDEFINED, never FAIL) and validated
  like every counted arm (source, hash seed, corpus, errored turn, organ error).
- **Correct** for a C3 seed is now `score_ask(intact_a)["all"]` AND `score_ask(reversed)["all"]`. That covers every
  correctness clause of prereg 4.3 (`discriminates`, `lp_learnable`, `lp_unlearnable`, `all_gaps_abstain`,
  `builtin_dog_cat`), plus rule (a), in both orders. A FAIL reads "capability absent on k of 6 seeds (correct per order:
  registered a of 6, reversed b of 6)". The sub-clauses `reversed_all`, `reversed_discriminates`, `reversed_lp_learnable`
  and `reversed_lp_unlearnable` are reported beside it, and the report prints each seed's reversed-order line.
- **Scope, stated as a limit:** the lesion clauses (`lesion_curiosity`, `lesion_graded`, `.rep0`) and the permuted
  clause are read in the registered order only, since there is no reversed lesion or permuted arm. The reversed arm has
  no null pair of its own: the build's determinism is the intact pair's null at the same seed and revision.
- **Reading a C3 FAIL** (ruling): it names the gap "global DA engagement threshold; no per-topic learning-progress
  signal". The report prints it whichever clause failed (A4.4).

## A4.3 (c) Informational DA-lesion arm (reported, never counted)

The production lesion exists: `BRAIN_CURIOSITY_DA_LESION=1` (`webapp/da_curiosity_drives_chat.py`) pins the DA
modulation to 0, gain 1.0. Arm `da_lesion` (group `ask`, the registered order) reports the ask rates, `discriminates`,
`lp_learnable`, `lp_unlearnable`, `moats_undefined` and `asked_by_label` per seed. It is never a pass condition, never
a precondition and never in the stage verdict. Its question is whether the order confound vanishes without the DA
coupling and whether the unlearnable gap then stops at all, which would show a learning-progress signal.

## A4.4 Reporting rule for C1 and C3 (no bar change, no wording change)

A FAIL of these groups prints a `Named gaps` line (`A4_NAMED_GAPS`) beside prereg section 8's sub-bar reading,
whichever sub-clause failed:

- **C1 `conv`** (NEEDS-CLAUDE #17): "wh-locative question parse; locative frame storage (the teach echo drops the
  preposition); discourse elaboration for 'tell me more'". Dev 7 shows all three. `where does the deer sleep` abstains at
  the question parse, so the prepositional sub-bar measures the wh-parse before the frame inventory. The teach echo
  `the deer sleeps the meadow` drops the preposition. `tell me more` returns "I don't have anything more to add". The
  probe wording is kept: `deer_sleep` and `more` stay as registered.
- **C3 `ask`** (NEEDS-CLAUDE #16): "global DA engagement threshold; no per-topic learning-progress signal".

## A4.5 NEEDS-CLAUDE #15 resolved (C5 private): no C5 change here

Amendment A3's clause stands: the idle fact must be a stored fact (`private_voices_stored_fact`) and must follow the
exposure (`private_selected_by_state`). The "fixed demo lexicon" blocker is gone at any revision with the idle-replay
organ ON. `research/runners/idle_replay_organ.py` voices a fact from the brain's own store in the same remark frame
("Something's been on my mind -- <a> <v> <p>") that the scorer parses. It is default ON on main since `8c10af86a`
(`_IDLE_REPLAY_DEFAULT_ON = True`, PROVISIONAL per its battery prereg of 2026-10-12). **This branch does not contain
`8c10af86a`.** The gate revision must contain it (see Consequences). If that flip is reverted before the gate, C5's
private half reads the demo lexicon again and FAILs "capability absent", as A3 states.

## Lane

Jobs: 186 -> 198 (6 seeds x 33 arms): +6 counted `ask/reversed`, +6 informational `ask/da_lesion`, both
`mem_gb=9`, memcap 10, like every line. The `ask` arm measured 6.3-6.5 GB peak and 913 s at dev 7 (workstation), so
the addition is about 3 job-hours on the pool (~1.4x slower). `--no-informational` drops `da_lesion` but never `reversed`.

## Bar-sanity (n = 1000 per brain, seed 0; `tools/bar_sanity.py`)

New C3 generators (`bar_sanity_s2_common.py`). The teach turns now echo the stored triple, as dev 7's handler does. The
reversed arm reads the same brain in the reversed order, and the DA-throttle null applies "silent after two asks" along
each arm's own probe order. Pass rates under the pre-A4 scorer (HEAD `4b19aafb8`, n = 200) and this one:

| generator | kind | pre-A4 | A4 |
|---|---|---|---|
| `perfect_goat_unacquired` (dev 7's shape: goat novel, asked) | perfect | 1.000 | 1.000 |
| `perfect_two_moats_unacquired` | perfect | 0.000 | 1.000 |
| `null_no_moat_acquired` | null | 0.000 | 0.000 |
| `null_order_carried` (registered order perfect, the reversed order DA-throttled) | null | 1.000 | 0.000 |
| `null_order_carried_no_stop` (the reversed order never stops on the unlearnable gap) | null | 1.000 | 0.000 |

Every other C3 null stays 0.000 and every perfect brain 1.000. The stage spec gains `null_ask_reversed_order_carried`.
C1, C2, C4 and C5 were re-measured at the amended scorer (their records' scorer sha moved). All six records read OK:

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
The `ask/reversed` dev smoke (`dev/s7/ask__reversed.json`: valid, peak RSS 6.5 GB, 751 s) shows the confound in mirror
image. `u_toad_1` and `u_toad_2` ask (DA 0.554, then 0.246); from `u_toad_3` on, the DA level is 0.046 and the effective
threshold 59.8 Hz, so nothing asks: `lp_ibex_1`, every gap and every moat are silent (gap rate 0, `discriminates`
False). `lp_unlearnable` reads True there, but for the same history reason (the threshold rose), not a learning-progress
signal. Each order passes the clauses the other fails, so neither order alone carries C3, which is what A4.2 requires.

## What this does NOT establish

Both orders passing is necessary for "the brain asks by topic history", not proof of it. A brain with a per-session ask
budget large enough for both orders would pass, and no learning-progress organ is implied by a PASS. The DA-lesion arm is
one informational read and attributes nothing. Rule (a) trusts the teach echo as the read of what was stored. A brain
that echoes the triple but stores something else would be counted as acquiring it; the moat probes would then expose it.

## Consequences for the packet

- `PREREG_SHA` is unchanged (`24dac499e`). The prereg file is not touched.
- **`GATE_SHA` (`43763e637`) is void for the gate.** That revision has no `reversed` arm, so every `ask/reversed` pool
  line would die there. The gate must be re-pinned at a revision that contains this file, the amended runner and tests,
  and main at or after `8c10af86a` (idle replay ON, A4.5). The check: `git cat-file -e $SHA:docs/drafts/stage2-exit/AMENDMENT-A4.md`
  and `git merge-base --is-ancestor 8c10af86a $SHA`. Claude did not write `GATE_SHA`; Hermes re-pins it after merging
  main (HANDOFF section 3).
- No dev line is re-run. The worker, turns and arms of the registered arms are unchanged, so every existing dev shard
  re-scores under the new code.
