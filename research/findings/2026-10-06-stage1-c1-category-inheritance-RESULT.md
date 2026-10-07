---
type: finding
status: live
date: 2026-10-06
lane: integration-first (Stage-1 exit criterion C1; stage exits on the calendar)
mechanism: Stage-1 C1 category held-out inheritance on the integrated production brain with the default-off category-inheritance wiring (BRAIN_CATEGORY_INHERIT) switched on, six gate seeds, scored by its registered scorer; plus the diagnosis of why C1 failed at production defaults in Stage-1 exit attempt 1
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/stage1_c1_category.py
artifacts:
  - research/findings/raw/_stage1_c1_category/gate/aggregate.json
---

# Stage-1 C1 with the category-inheritance wiring on: PASS 6/6, narrowed (Claude review, 2026-10-06)

Prereg: `research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md` (committed alone at
e3ef82263; gate revision 6c2cb1362, prereg byte-unchanged there, section 13 empty). Scored only by
`python -m research.runners.stage1_c1_category --aggregate <gate dir> --seeds 42,43,44,100,101,102`. Every number
below is read from research/findings/raw/_stage1_c1_category/gate/aggregate.json or from the 52 gate shards beside it
(the shards were written 2026-10-04/05 by the pool and the workstation and were never scored until this review).

**Verdict: PASS.** Correct 6/6, lesion effect 6/6, reproduced 6/6, permuted tracks 6/6, six defined seeds, every
precondition met. Flip-readiness as registered: READY (safety 6/6, specificity 6/6). The production default is
unchanged: the wiring stays off.

| seed | `what does the canary eat` (flag on) | organ read: worm / bug | lesion_inherit | permuted (canary swims in the lake) |
|---|---|---|---|---|
| 42 | flagged guess: worm | 0.1208 / 0.0 | abstains | bug (0.1067 / worm 0.0) | <!--derived-->
| 43 | flagged guess: worm | 0.1258 / 0.0 | abstains | bug (0.1142 / 0.0) | <!--derived-->
| 44 | flagged guess: worm | 0.11 / 0.0 | abstains | bug (0.1075 / 0.0) | <!--derived-->
| 100 | flagged guess: worm | 0.1083 / 0.0 | abstains | bug (0.1133 / 0.0) | <!--derived-->
| 101 | flagged guess: worm | 0.12 / 0.0 | abstains | bug (0.0858 / 0.0) | <!--derived-->
| 102 | flagged guess: worm | 0.1117 / 0.0 | abstains | bug (0.1183 / 0.0) | <!--derived-->

Organ rates are spikes per neuron per step (values marked derived are read from the per-arm shards in the gate directory and, for the bridge, from `research/findings/raw/_stage1_c1_category/dev/organ_dev_seeds1-8.json`); the read's floor is 0.02 and its margin 1.25x. Informational arms:
`off` (the production default) abstains 6/6; `lesion_lateral_learn` (the frame->frame association never forms)
abstains 5/5; `lesion_lateral_read` (that pathway silenced at read only) keeps worm 5/5 (0.074-0.098); `lesion_draw` <!--derived-->
(the Stage-1 prereg's own C1 lesion) changes nothing 6/6. Seed 42's two lateral arms never landed (RULING
pool-hold-c1); neither is counted.

## Why C1 failed in Stage-1 exit attempt 1 (from the data)

Attempt 1 (`research/findings/raw/_stage1_exit/gate/`, pool only, revision c238ff325) read C1 FAIL with the
capability absent on 6 of 6 seeds. The shards say where the property was lost:

- **Not routing, not retrieval.** The question parse and the recall path worked on every seed: the strict probe's
  `activity.roles` reads canary/eat with 16 facts scanned, the identity controls recall robin -> worm and trout -> bug
  6/6, and in the permuted arm `what does the canary swim` recalls lake 6/6 (a stored fact is found when it exists).
- **Encoding: the category link was never written.** `matched_fact_index` is null on every seed because no structure
  in the production brain records that the canary shares a context with robin, sparrow and finch. Each heard fact is
  stored as an isolated (agent, action, patient) block, and the strict route is an exact match on (agent, action).
- **The generative route never proposed.** `what might the canary eat` returns before the recall path (`activity`
  null, no hypothesis) on 6/6: its plausibility gate needs a canary-eat co-occurrence that was never given. So the
  registered lesion (`lesion_draw`) had nothing to remove and no engagement record; its "0/6" is not evidence about
  the DRAW.
- **`cat_told` (diagnostic) lost the premise at encoding too.** `the bird eats the worm` was never acquired on any
  seed: the comprehension monitor asked which of bird and worm is the patient. So `what does the bird eat` abstains
  6/6 although `what is a canary` recalls bird 6/6. Told inheritance needs that sentence acquired before any
  composition can be tested.

**The wall-reframe question** ("what does the real system run alongside this that we replaced with a constant?"):
when a new episode shares elements with stored ones, the hippocampus reactivates the overlapping memories while
encoding it, so inferential links form at encoding time (integrative encoding; Shohamy & Wagner 2008, Neuron;
Zeithamova, Dominick & Preston 2012, Neuron). Production replaced that process with a constant zero: storage never
consulted what the canary's context had co-occurred with. The 09-30 organ supplies it, and its own ablations agree
with the account: the inherited link is written while the canary's fact is rehearsed (the read-only lateral lesion
keeps worm 5/5), and it needs the learned frame->frame association (the learning lesion removes it 5/5).

## Verify-go (Claude, 2026-10-06): two independent skeptics; both NARROWED, neither refuted

Lenses: gate-cheat and instrument trust; control integrity, reproducibility and flip safety; the three required
lenses. Re-score: this review's aggregate equals a first independent scoring field for field.

- **Engagement (a).** Every counted arm records it: the lesion arms read `lesion: all`, `lesion_held: true`, every
  admitted frame 0.0; the permuted arm's canary observe is `swim lake`; every flag-on turn carries the trace with no
  swallowed organ error; the `off` arm carries none.
- **Answer rate (b).** Flag-on abstains on 2 probes per seed, flag-off on 3; the only content change is
  `p_canary_eat` (abstain -> flagged guess). No FLIP-HURTS-ANSWERS.
- **Where ran (c).** `tools/where_ran.py`: 52 files, pool 19 (node-mxl1462drl 15, node-dl5d243 2, node-dl4g243 2),
  workstation 33 (dant123-wk). All 52 sidecars read git_sha 6c2cb1362, git_dirty false, corpus present, numpy. The
  organ and wiring read no file under research/findings/raw/. The null is clean across hosts (seed 101 intact_a on
  the workstation, intact_b on the pool), so validity does not depend on the host.
- **Not a frequency artifact.** Three bird members carry `eat worm` and two fish carry `eat bug`, but the permuted
  arm reads bug 0.086-0.118 with worm at exactly 0.0 on every seed: the read follows the canary's exposure. <!--derived-->
- **No host lookup produces the answer.** The read drives the member slot for 60 steps and counts spikes per frame
  slot; no fact table is consulted. The lesion zeroes both learned pathways for the read and restores them after.

Caveats that travel with the word PASS:
1. **This is the wiring's prereg, not Stage-1's C1 as frozen.** The Stage-1 prereg counts `lesion_draw`, which does
   not engage on this route. Under Amendment A1's rule (a no-effect lesion with no engagement record is UNDEFINED),
   Stage-1 attempt 2 would read C1 UNDEFINED unless Amendment A2, registered before any attempt-2 data, counts the
   pathway cut (`lesion_pathway`, patch `lesion_inherit_pathway`; RULING stage1-c1-lesion). Text:
   `docs/drafts/stage1-c1-inheritance/ATTEMPT2-C1-AMENDMENT-TEXT.md`.
2. **One deterministic outcome replicated on six substrates.** OU noise is off and seeds vary only per-neuron
   heterogeneity; the null and the repeat can only read zero differences. The organ was tuned on dev seeds 1-8 with
   this same script.
3. **Measured on the 09-30 stack, numpy only.** The gate revision is about 440 commits behind main; main changed
   `webapp/server.py`, `brain_chat_tui.py` and several defaults since. On the merged stack (this branch) the wrapper
   is still the outermost gate wrapper (only the default-off open-ended trace sits outside it), but nothing has been
   measured there yet. Production on cupy is unmeasured.
4. **Flip safety beyond the C1 script is unmeasured through the handler.** The C2/C3 moats abstained at organ level
   (dev seeds 1-8, and one skeptic's organ-only rerun on seed 42 including the permuted C2/C3 facts and a mixed
   session), not through the real handler.
5. **A bridging member over-generalizes.** At organ level (dev seeds 1-8) a bat that eats the bug and flies in the
   sky makes `what does the canary swim` reach lake on 8/8 (0.064-0.084) and ties the canary's eat read on 4/8. The <!--derived-->
   reply would be a flagged guess, but it breaks a moat-style probe in a richer session.
6. **Capacity.** 24 member and 24 frame slots; beyond that a fact is silently not encoded (`reason: capacity`), so
   inheritance degrades to abstaining in long sessions.
7. **Ownership.** The answer is spiking-selected inside the organ but host-admitted: frame tokens, slot recruitment
   and admission by the asked verb's string are host, and the floor plus 1.25x margin is a host comparison of spike
   counts (the k-select class). The organ is a standalone bridge, not the one substrate, so it is not self-organized
   in the docs/TERMS.md sense. The generative route still never proposes (`p_canary_might` abstains 6/6).

**What this shows:** on the integrated handler, with the wiring on, a never-told property is assigned from a category
discovered from shared contexts, the reply follows the exposure, and cutting the organ's learned pathways removes
the answer. **What it does not show:** that the flipped default is safe in other conversations, that Stage-1 C1
passes as frozen, or that neurons made the final choice.

Next (docs/drafts/stage1-c1-inheritance/PLAN.md, HANDOFF.md): the flip dev check on the merged stack (dev seeds 1-8,
all Stage-1 groups, flag on vs off, plus the bridging probe, reported), then the flip, then the attempt-2 C1
amendment and attempt 2.

Ledger wording: `Stage-1 C1 with the category-inheritance wiring ON (default off), 09-30 stack, numpy: PASS 6/6 (reply
level; organ lesion load-bearing and reproduced; permuted exposure tracked; answer host-admitted, spiking-selected;
standalone organ). Not yet a Stage-1 pass: attempt 2 needs the flip and Amendment A2's C1 lesion arm (lesion_pathway).`
