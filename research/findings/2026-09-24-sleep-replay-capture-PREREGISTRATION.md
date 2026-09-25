---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: sleep-replay-triggered synaptic capture (webapp/sleep_replay_capture.py, BRAIN_SLEEP_REPLAY_CAPTURE, default OFF) -- one SWR epoch at sleep onset reads every DA-tag-capture-managed store block back through the store's own cleanup, re-tags it in proportion, and drives the SAME spiking D1 pool / PRP pool with SWR-coupled DA, so an ordinary fact told once can be captured overnight
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_sleep_replay_capture/design_fake_substrate.json
  - research/findings/raw/_awake_replay_capture/design_fake_substrate.json
  - research/findings/raw/_sleep_forgetting_interference/design_fake_substrate.json
---

# PRE-REGISTRATION — a sleep route for ordinary facts under DA tag-and-capture (2026-09-24)

Committed on its own, BEFORE any run of the family it governs (no `--family rc` output exists at this commit). The
code it governs is commit `fd664ef3f` on branch `research/sleep-replay-capture` (off `main` at `3f1d0bd01`, with
`research/da-tag-capture-ltm-on` at `cce3c1dbd` merged in). Every constant below is fixed there. Terms follow
`docs/TERMS.md`; the word "consolidation" is not used for this mechanism (it is capture in the same store: no
transfer, no source lesion).

prereg-same-commit: the one artifact committed with this file is the FAKE-substrate design sweep (no brain, no gate seed, no gate reads it) that fixed the one-epoch design; no rc-family gate row exists yet

## Why (board #227, the flip blocker)

With `BRAIN_DA_TAG_CAPTURE` on, a plainly told fact is gone the next day: G3 of the 6/6 runner-level GO
(`2026-09-24-da-tag-capture-chat-wire-6seed-GO-runner-level-ltm-off.md`) requires it, and branch
`research/da-tag-capture-ltm-on` added the `neu_night_off_intact` arm and the `ordinary_fact_flip_forgetting` field to
make that cost visible. Flipping the ledger on would therefore make the brain forget every ordinary thing it is told
overnight, which contradicts the owner's 2026-09-24 directive that knowledge be learned synaptically and grow. The
verdict of the ltm-on branch was (c) MISSING: nothing in the code base lets an unsalient block reach late phase.

The wall question, "what else does the real system run alongside tagging-and-capture that we replaced with a
constant?", has a direct answer in the local corpus: sleep. The hippocampus replays the day's modified pathways in
sharp-wave ripples, and Buzsáki (note 28, p.347) writes that the replay mechanism "could replace the tagging mechanism
or the two processes could work in parallel". Clopath et al. 2008 give the trigger (PRP synthesis starts when enough
tags are set; the co-released phasic DA is modeled as proportional to the tag count). The v3 ledger had no sleep
event, so the PRP supply at night was the constant zero. Binding: `research/biology/sleep-replay-tag-capture.md`
(9 sources, all resolving; Clopath read in full text, PMC2596310).

Corpus check first: `bash tools/before_you_build.sh "ordinary facts forgotten overnight under DA tag-and-capture; no
sleep replay or repetition route to capture"` (logged). The existing BRAIN_SLEEP_REPLAY (#64) reactivates the
EPISODIC organ's CA3 assemblies, a separate store that is not built on the numpy probe path, and #138 only retimes
Turrigiano scaling; neither touches the ledger's blocks.

## What was built (default OFF, byte-identical off)

- `webapp/sleep_replay_capture.py`. At sleep onset (`continuous_engine.SLEEP_IDLE_SEC` after the last observed turn,
  the engine's own sleep-depth criterion) ONE SWR epoch: (1) the ledger rewrites the store to its value at that
  moment; (2) every managed block's trigger is driven on the composer's resonate-and-fire substrate and read back
  by the store's own cleanup (`OneBrainComposer._block_role_scores`); R_i = the smallest cleanup decisiveness margin
  over agent/action/patient; (3) replay tag = R_i x |increment| per synapse, the synapse's tag being the larger of
  write tag and replay tag; (4) SWR-coupled DA = 0.5 + (1.24 - 0.5) x min(1, sum_i R_i), read by the SAME spiking D1
  pool in ten 30-s reads (5 min), fed to the SAME PRP pool through both existing lesion edges; (5) the v3 per-synapse
  late-phase dynamics decide capture. No threshold compare anywhere in the route.
- `BRAIN_SLEEP_REPLAY_CAPTURE_LESION=1` cuts the reactivation edge: the reads still run, R_eff = 0, so no replay tag
  and the SWR DA stays tonic.
- `webapp/da_tag_capture.py`: `_integrate` uses max(write tag, replay tag) only for a block that carries a replay
  tag; none ever does with the flag off. The v3 synthetic-scenario hash is unchanged (`82f8856d...`, pinned in
  `tests/test_sleep_replay_capture.py`). `webapp/da_tag_capture_chat.py`: `_catch_up` runs the due epoch first when
  the flag is on.
- `research/runners/_da_tag_capture_chat_probe.py --family rc`: `RC_ARMS`, `grade_seed_rc`, `aggregate_rc`. The base
  family (`ARMS`, `grade_seed`, `aggregate`) is not modified, so no committed `seed*.json` is re-graded.

### Constants (a priori; none fitted to any gate seed)

| constant | value | where it comes from |
|---|---|---|
| sleep onset | `SLEEP_IDLE_SEC` = 300 s after the last turn | the engine's existing sleep-depth criterion (imported) |
| SWR epochs per night | 1 (at sleep onset, the first N3) | Kandel ch.44 "begins with a rapid descent into stage N3"; see the one-epoch reason below |
| SWR bout | 5 min in 10 reads of 30 s | the v3 `CAPTURE_PROTOCOL_MIN` and `TURN_DRIVE_H`, reused |
| DA at full replay | 1.24 | the D1 pool's own calibration ceiling (`_DA_CAL_HI`), where its activation is 1 by construction |
| replay-to-DA map | tonic + span x min(1, sum R) | declared operating point (Clopath: DA proportional to tags) |

**Why one epoch per night.** Measured before this commit on a FAKE substrate (no brain;
`research/runners/_sleep_replay_capture_design.py`, artifact
`research/findings/raw/_sleep_replay_capture/design_fake_substrate.json`, key `summary`): with five epochs 90 min
apart, an older fact A was captured at every age tried, up to 8 h before sleep, although its read-back at the first
epoch was at baseline (a sub-threshold late-phase z is expressed in the weight and raises the next read-back: a
runaway). The real brake, synaptic downscaling in sleep (Tononi & Cirelli, Kandel ch.44), is not modeled. With one
epoch, A is captured when told up to 2 h before sleep and not when told 3 h or more before, consistent with Kandel's
2-3 h capture window; with zero epochs neither A nor the just-told fact B survives. This is a design decision made on
synthetic data, before any gate seed; it is recorded here, not tuned later.

## Arms (each a fresh tiny-demo brain in its own subprocess; numpy; LTM off)

Every arm: `BRAIN_DA_TAG_CAPTURE=1`, `BRAIN_DA_TAG_CAPTURE_CLOCK=turn`, `BRAIN_LTM_SHIP_DEFAULT=0`, the seed through
`BRAIN_CHAT_SEED` (threaded to `cfg.seed` in every build, `webapp/server.py:_brain_chat_seed`; never
`actual_seed_used`). RC = `BRAIN_SLEEP_REPLAY_CAPTURE=1`. Conversations are the existing battery groups: `datn`
(neutral: "the cat is here" / "the ball is here" / "the cat chases the ball" / "the cat is here" / "the ball is
here"), `datc` (the same fact inside surprising news), `datni` (neutral, asked at once). Night = the existing 24 h
world step through `continuous_engine.tick_idle_sessions`. Recall: "what does the cat chase".

| arm | group | extra env |
|---|---|---|
| `neu_night_norc` | datn | (none): the flag-off baseline |
| `neu_night_rc_a` | datn | RC |
| `neu_night_rc_b` | datn | RC (null-control rebuild) |
| `neu_night_rc_replaylesion` | datn | RC + `BRAIN_SLEEP_REPLAY_CAPTURE_LESION=1` |
| `neu_night_rc_dalesion` | datn | RC + `BRAIN_DA_ENCODING_LESION=1` (REPORTED) |
| `neu_imm_rc` | datni | RC |
| `sal_night_rc` | datc | RC |
| `sal_night_rc_dalesion` | datc | RC + `BRAIN_DA_ENCODING_LESION=1` |
| `sal_night_rc_caplesion` | datc | RC + `BRAIN_DA_CAPTURE_LESION=1` |
| `sal_night_rc_replaylesion` | datc | RC + `BRAIN_SLEEP_REPLAY_CAPTURE_LESION=1` |

## Gates (per seed; `grade_seed_rc` implements them verbatim)

Outcome per recall reply: correct (recalled_svo = cat/chase/ball), abstain, confab, undefined.

**UNDEFINED** (never a pass or a fail) if any of:
- G0: `neu_night_rc_a` and `neu_night_rc_b` differ in outcome, recalled_svo, abstained, the ledger state at recall,
  the per-block ledger summary, or the sleep record;
- P1: `neu_imm_rc` is not correct (the fact was not stored and recalled at once, so the night contrast would be
  about encoding);
- I1: on any RC night arm the replay branch did not execute (no SWR epoch, or a block had no substrate read), or the
  immediate arm ran an epoch;
- I2: a lesion did not hold on the sleep record itself (replay lesion: R_eff all 0 and SWR DA = tonic; capture
  lesion: D1 drive 0; DA-encoding lesion: DA seen by D1 = tonic);
- gamma differs across companion-ON arms; any arm error; any undefined outcome.

**GO for the seed** iff all of:
- RC1 (the rescue): `neu_night_rc_a` correct AND `neu_night_norc` abstain;
- RC2 (the replay edge carries it): `neu_night_rc_replaylesion` abstain;
- RC3 (salient-vs-neutral separation kept): `sal_night_rc_replaylesion` correct (with the replay edge cut, the salient
  fact survives on its waking DA capture while the ordinary one, RC2, does not: the G3 contrast with the route armed);
- RC4 (the capture lesion still blocks salient capture): `sal_night_rc_caplesion` abstain;
- RC5 (DA stays load-bearing with the route armed): `sal_night_rc` correct AND `sal_night_rc_dalesion` abstain;
- RC6: no confab in any arm.
Otherwise NO-GO.

**REPORTED, never gating:** the `neu_night_rc_dalesion` outcome (prediction: abstain, since the sleep route runs
through the same D1 edge); per-arm R per block, SWR DA and mean D1 drive at the epoch; the pre-sleep fraction of
synapses with z > 1/2 for the neutral and salient fact blocks (prediction: salient higher, the waking DA capture
already under way); the number of managed blocks.

**6-seed verdict** (`--family rc --aggregate`): GO iff all six seeds read GO; INCOMPLETE if a seed is missing;
otherwise NO-GO. Reported with it: the ordinary-fact next-day correct rate with the route on vs off, and the one-sided
exact sign-flip p over seeds for RC1 and for RC5 (1/64 at 6/6).

### How this maps onto the requested GO

- "Ordinary facts survive the night with the flag on at a rate clearly above flag-off on every seed" = RC1 on every
  seed (one fact per seed, so the per-seed rate is 1 vs 0; across seeds 6/6 vs 0/6).
- "Salient-vs-neutral separation of the existing G2-G4 gates is kept": G2 and G4 cannot change (no night in the
  immediate arms, so no epoch, asserted by I1; the route is inert with the companion off, asserted by
  `test_inert_without_the_tag_capture_ledger`), and the base family's G0-G6 verdict is untouched because the flag-off
  path is byte-identical (`test_flag_off_is_identical_to_plain_ledger_path`, the pinned v3 hash). G3 ("neutral NOT
  kept") reads the opposite with the route on BY DESIGN; its contrast is carried by RC2 + RC3 instead.
- "A lesion of the replay edge removes the rescue" = RC2.
- Added beyond the request: RC4 (the capture lesion at behaviour level) and RC5 (so that flipping both flags does not
  make da-gated-encoding lose its next-day load-bearing read, the point of #227).

## What each gate can and cannot show

- RC1 shows that the flag changes the next-day reply for an ordinary fact. On its own it cannot show WHY: RC2 is what
  ties the change to the reactivation edge, and I1/I2 show the branch ran and the lesion held on the record.
- RC5's lesion half, like G1's, is partly an integrity check: pinning the DA the D1 pool sees to tonic removes the
  drive by construction. What it adds is that nothing else in the route (the replay tag alone, the D1 pool's tonic
  noise over the 5-min bout) captures without DA.
- The selection is brain-read (the store's own cleanup margin), but the replay-to-DA map and the one-epoch schedule
  are declared operating points (module docstring, HOST SHORTCUTS). A GO here does not retire them.
- One fact per conversation: this does not measure interference between many facts competing for one night's replay
  and PRP. That is a follow-on family, not claimed here.

## Declared deviations and compute

- LTM off (`BRAIN_LTM_SHIP_DEFAULT=0`, buffer only), as in the base family; the LTM-on configuration is not read here.
- A seed-42 smoke of this family is run AFTER this commit, as a de-risk only, to a separate directory
  (`research/findings/raw/_sleep_replay_capture_smoke`). It is not a gate row. The six gate rows, seed 42 included,
  are the pool runs at a pinned revision containing this pre-registration. If the smoke forces any code change, that
  change is an amendment committed before the pool runs, and the pin moves with it.
- Numpy CPU, one arm at a time per seed (`--workers 1`), each seed run under the pool's memory cap.

## Amendment 1 (2026-09-24, r2) — long delay, sleep downscaling, a byte-identical-OFF check that cannot go stale

Committed on its own, BEFORE any run it governs (no `--family r2` output and no counterfactual offcheck output exist
at this commit). Governs code commit `1d011480a` on branch `research/sleep-replay-capture-r2` (off `main` at
`acca762fa`). The review of the merged route (SOUND-WITH-ISSUES) asked for a long-delay arm on a real build; the
owner's follow-up asked for sleep downscaling and a counterfactual offcheck.

**What r2 changes for the rc family above: nothing it measures.** An idle stretch now runs one SWR epoch per night
(t_ref + onset + k x 24 h). Every rc arm recalls within 24 h + onset of its last turn, so it still runs exactly one
epoch (`test_one_night_protocol_still_one_epoch_three_nights_three`). The 6-seed rc run in flight at revision
`269ae8f76` is not touched and its gates above stand.

### Item 1 — a fact told ~4 h before sleep onset (REPORTED, not gated)

Group `datl`: the neutral telling, then the `awake_4h` world step (environment clock +4 h, every live ledger marked
awake through that time, no idle tick), then the usual 24 h night, then "what does the cat chase". Arms (`--family
r2`): `ld_ledger_off` (ledger off: today's production default), `ld_norc` (ledger on, route off), `ld_rc` (route on),
`ld_rc_replaylesion` (route on, replay edge cut), and `neu_imm_rc` (precondition: the neutral fact is recalled at once).

Verdict per seed (`grade_seed_r2`, `LD_verdict`): UNDEFINED if `neu_imm_rc` is not correct, an arm errs or reads
undefined, gamma differs across ledger-on arms, or the instrument fails (each route arm ran exactly one SWR epoch and
it started AFTER the awake mark, which is >= 4 h; the replay lesion held on the record). Otherwise RESCUED iff `ld_rc`
is correct and both `ld_norc` and `ld_rc_replaylesion` abstain; NOT-RESCUED iff `ld_rc` and `ld_norc` both abstain;
OTHER for any other pattern.

**Prediction: NOT-RESCUED**, with the reactivation read at sleep onset near baseline (the fake-substrate design sweep
above captured nothing told 3 h or more before sleep).

**Should it rescue? Stated before the run.** On the biology the route should NOT rescue this fact by itself. Late-phase
capture of a weak input needs its tag to still be live, and Kandel ch.54 puts the window at 2-3 h. Human declarative
memory gains from sleep "when sleep follows within a few hours of learning" (Gais, Lucas & Born 2006, Learn Mem 13:259,
abstract via PubMed). What the biology does NOT support is the model's TOTAL loss: in Gais 2006 delayed sleep gives a
smaller benefit, not zero retention. The model loses everything because it has none of the other routes a person has
for a fact told in the morning: waking reactivation (thinking about it, retrieval re-tagging it, re-mention), and
hippocampal-cortical transfer. Those are the named gap. A NOT-RESCUED reading is the route working as designed, and it
does NOT show that forgetting the fact is correct.

### Item 2 — three nights with sleep downscaling (GATED; `BRAIN_SLEEP_DOWNSCALING`, default OFF)

Each night, after the reactivation, each managed block's learned increment is multiplied by 1 - 0.18 x (1 - R_i).
The 0.18 is from de Vivo et al. 2017 (Science 355:507): the axon-spine interface was ~18% smaller after sleep.
Protection by the block's own read R_i follows González-Rueda et al. 2018 (Neuron 97:1244): inputs that drive
postsynaptic spiking are protected. Host steps, all declared in the module docstring: the multiply, the constant,
and R_i (the replay's cleanup margin) as the protection read. The baseline is not downscaled.

Groups: `d3w` WEAK telling (the four habituating turns come first, the fact is said last), three nights, recall;
`d3c` salient telling, three nights, recall; `d3r` the weak telling, re-mentioned ("the cat chases the ball") after
night 1 and after night 2, recall after night 3. Arms: `d3w_rc` (route on, downscaling OFF), `d3w_shy_a`, `d3w_shy_b`
(null rebuild), `d3c_shy`, `d3r_shy` (route + downscaling ON).

Per seed (`grade_seed_r2`, `seed_verdict`):
- UNDEFINED if: `d3w_shy_a` and `d3w_shy_b` differ (outcome, recalled_svo, abstained, ledger state, block summaries,
  sleep record); P2 `d3w_rc` is not correct (the weak fact must survive three nights WITHOUT downscaling, or a fade
  under downscaling is not attributable to it); any three-night arm did not run exactly three epochs, or carries a
  downscaling record without the flag, or lacks one with it; an arm errs or reads undefined; gamma differs.
- GO iff SHY1 `d3w_shy_a` abstains (the weak, never re-mentioned fact fades), SHY2 `d3c_shy` correct (salient
  survives), SHY3 `d3r_shy` correct (re-mentioned survives), SHY4 no confab in any arm. Otherwise NO-GO.
- REPORTED: R, downscaling factor, increment and baseline magnitude per night and at recall, for each three-night arm.

**Predictions.** SHY2 and SHY3 hold. **SHY1 is uncertain, and that is declared here.** With the reads the merged
smoke measured (R about 0.3 for a unit-gain write, 0.43 for the neutral telling), each night multiplies the
increment by 0.87-0.90. Three nights give 0.66-0.73 of it, and nothing measures where on the real substrate the
read stops recalling. A NO-GO on SHY1 would mean that downscaling at the literature magnitude does not erase a
sleep-captured weak fact within three nights. It would not mean that downscaling does nothing.

6-seed verdict (`--family r2 --aggregate`): item 2 GO iff all six seeds GO, with a one-sided sign-flip p over seeds for
"`d3w_rc` kept minus `d3w_shy_a` kept"; item 1 reported as counts of `LD_verdict`.

### Item 3 — byte-identical OFF against a counterfactual built from the current tree

`--offcheck` now compares the committed HEAD with the flags unset against HEAD with the feature's OWN commits
reverse-applied, in two temporary worktrees under `/home/dant123/Projects/sim/.claude/worktrees/` that are removed
afterwards. The feature commits are derived on every run: every non-merge commit that touches
`webapp/da_tag_capture*.py` / `webapp/sleep_replay_capture.py`, or that adds or removes a `da_tag_capture` /
`sleep_replay_capture` line in `webapp/server.py` / `webapp/continuous_engine.py`. Instrument, battery, findings,
tests and docs are held equal in both trees. The same salient next-day conversation is hashed on each tree (replies
+ store synapses), and HEAD is also run twice as a null control.
Verdict: IDENTICAL iff the null control is identical and both hashes match; DIFFERENT iff the null control is
identical and a hash differs (the first differing turn is saved); UNDEFINED if the reverse-apply fails or the null
control differs. Run once locally under `bash tools/memcap.sh 8 -- ... --offcheck`, output to a file named
`offcheck_counterfactual` (JSON) in the directory `research/findings/raw/_sleep_replay_capture_r2/`.

### Compute for this amendment

A seed-42 smoke of `--family r2` (de-risk, NOT a gate row) goes to `research/findings/raw/_sleep_replay_capture_r2_smoke`.
The six gate rows are pool runs at a pinned revision containing this amendment; if the smoke forces a code change, that
is Amendment 2 and the pin moves.

## Amendment 2 (2026-09-24, r2) — the counterfactual offcheck's construction, fixed before its first run

Committed before the counterfactual offcheck has run. It changes only the item-3 instrument, not item 1 or 2. The
seed-42 r2 smoke was already running at `2f3bd2087`. It is unaffected, because this edit touches only the offcheck
functions of the probe, which its arm workers never import.

A git-only dry run of Amendment 1's construction (no brain build) failed in two ways, and the check would have read
UNDEFINED. **First, the reverse-apply conflicted in `webapp/server.py`.** Later, unrelated commits (the D6 chat
observability block) inserted code right after the feature's hook hunks. So the 3-way reverse of `a201293f5` saw
adjacent edits. **Second, a standalone runner would not reverse.** `492231df3` created
`_da_encoding_natural_drive_synaptic.py`, and later non-feature commits edited it, so reversing its creation cannot
apply. The construction is therefore now:
- Revert only `production_scope()`: `webapp/`, `sim/`, and every `research/runners` module that a `webapp/*.py`
  imports, derived from the tree on each run (123 paths at `2f3bd2087`). Everything else stays at HEAD in both
  trees, including standalone runners, the instrument, the battery, findings, tests and docs. None of it is on the
  `/api/brain-chat` reply path.
- For each commit, try a 3-way reverse first. If it conflicts, roll that commit's partial application back and
  retry with a zero-context reverse, which matches only the feature's own lines. If that also fails, the check
  reads UNDEFINED.
- No import of a feature module may remain in the counterfactual `webapp/`; otherwise UNDEFINED. Prose mentions in
  other modules' docstrings do not count.

Dry run with this construction, at HEAD `2f3bd2087`. Six feature commits were reversed: five by 3-way,
`a201293f5` by zero-context. No residual import remained. Relative to HEAD, the counterfactual deletes
`webapp/da_tag_capture*.py` and `webapp/sleep_replay_capture.py`. It restores the pre-isolation
`_da_write_gain_spiking_derisk.py`, which is a pure refactor. It removes exactly the three `server.py` hook blocks
and the one `continuous_engine.py` hook block, and nothing else. The verdict rule of Amendment 1 is unchanged.

## Amendment 3 (2026-09-25, r2) — a REPORTED ten-night horizon for the downscaling NO-GO

Committed before any run of the arms it adds. The seed-42 r2 smoke (a de-risk, not a gate row; artifacts committed
at `27fe9202`) read item 2 NO-GO on SHY1 alone. The weak, never re-mentioned fact was still recalled after three
nights of downscaling, as Amendment 1 had declared possible. Three nights leave open whether downscaling at the
literature magnitude erases the fact at all, and when. Rather than retune the constant on a gate seed, this
amendment MEASURES that horizon.

- Group `d10w`: the weak telling, then ten nights, each followed by "what does the cat chase". In this model the
  recall is a read-only probe: the store read writes nothing, and the question stores no fact. The daily question
  is an observed turn, so it starts a new idle stretch, and the next night's epoch begins sleep-onset after it.
- Arms `d10w_rc` (downscaling off) and `d10w_shy` (downscaling on), both route on. They are REPORTED ONLY: excluded
  from every gate, error count, gamma check and UNDEFINED rule of item 2, so they cannot change a seed verdict. For
  each arm and seed, `grade_seed_r2` reports the daily outcomes, the first night the fact is not recalled, and the
  daily increment/baseline magnitude and read.
- Predictions (from the three-night trajectory: increment x0.88 per night, and the read falls as it shrinks):
  `d10w_rc` is recalled on all ten nights. For `d10w_shy`, the first night the fact is not recalled falls in 4-10,
  or it is still recalled at night 10. Either is reported as measured.
- Items 1 and 2 keep their Amendment-1 rules. Item 2's gate stays at three nights: this amendment does not move it.

A seed-42 de-risk of just these two arms (`--only d10w_rc,d10w_shy`, ungraded) goes to
`research/findings/raw/_sleep_replay_capture_r2_horizon_smoke`. The 6-seed pool lines run the full r2 family,
horizon included, at a revision containing this amendment.

## Amendment 4 (2026-09-25, branch research/awake-replay-capture) — awake-rest replay for the long-delay fact

Committed on its own, BEFORE any run of the family it governs (no `--family arc` output exists at this commit). It
governs code commit `5651ce549` on branch `research/awake-replay-capture`. That branch is off
`research/sleep-replay-capture-r2` at `50c791bf9`, which is not merged to `main` yet. Every constant below is fixed at
`5651ce549`. Terms follow `docs/TERMS.md`: the new route re-potentiates the same store, so it is not "consolidation"
(no transfer, no source lesion).

The one artifact committed with this amendment is a fake-substrate design sweep, run at `5651ce549`:
`research/findings/raw/_awake_replay_capture/design_fake_substrate.json`. It has no brain and no gate seed, and no
gate reads it.

### Why

Amendment 1's item 1 was smoked on seed 42 (a de-risk, not a gate row;
`research/findings/raw/_sleep_replay_capture_r2_smoke/seed42.json`). It read NOT-RESCUED. The neutral fact was about
4 h old at sleep onset, and the night's read-back was 0.005609. A fresh fact read between 0.356535 and 0.466427 in the
committed smokes (`research/findings/raw/_sleep_replay_capture_smoke/seed42.json`). Today's default (`ld_ledger_off`,
ledger off) recalled the fact. Amendment 1 named the missing waking routes.

The wall question is: what does the real hippocampus run alongside tag-and-capture during WAKE that the model held at
a constant? The answer in the corpus is awake sharp-wave-ripple replay during quiet rest. In quiet rest after learning
the hippocampus replays recent experience (Kandel ch.5; Buzsáki p.344-349; Carr, Jadhav & Frank 2011). The
reactivation induces LTP, graded by how strongly the assembly is reactivated (Sadowski, Jones & Mellor 2016, read in
full text). Interrupting awake SWRs impairs learning (Jadhav et al. 2012). Between turns the model's early phase only
decayed. Binding: `research/biology/awake-replay-tag-capture.md` (13 sources resolve). Corpus check first:
`bash tools/before_you_build.sh "fact told 4 h before sleep onset lost under DA tag-and-capture: ..."` (logged).

The two other candidates were checked and not built. Behavioural tagging by a later salient event already exists: the
ledger's PRP pool is cell-wide, so a salient turn within the tag window captures a neutral fact (Moncada & Viola 2007).
Re-mention already rewrites the block with a fresh early phase and tag (`d3r`, Amendment 1). Retrieval-driven
re-tagging (a recall that re-tags) is not built: the recall read writes nothing in this model (Amendment 3). It stays
a named candidate.

### What was built (default OFF: `BRAIN_AWAKE_REPLAY_CAPTURE`)

- `webapp/awake_replay_capture.py`. A bout can run on the continuous engine's idle tick (`tick_idle_sessions`, which
  only ticks a session idle for at least `IDLE_SEC`). It runs when the ledger's own clock says the brain is still
  awake: world-now is earlier than (end of waking) + sleep onset, where the end of waking is the later of the last
  turn and any awake mark. At most one bout runs per 5 min of world time. A turn never runs a bout.
- In a bout, the ledger first integrates to world-now and rewrites the store. Then every managed block's trigger is
  driven on the composer's resonate-and-fire substrate and read back by the store's own cleanup. This is the sleep
  route's read (`reactivation_strength` -> `OneBrainComposer._block_role_scores`): R_i is the smallest cleanup margin
  over agent, action and patient. No host list, ranking or threshold picks a fact.
- Each block's early-phase expression e becomes e + R_i (1 - e). From then it decays with the written trace's own
  `TAU_EARLY_H`. Its tag is re-set to the same level (e x the local increment, v3's invariant), and the larger of that
  and any live replay tag is kept. The late-phase z is not touched.
- A bout reads no D1 and adds nothing to the PRP pool (next section).
- `BRAIN_AWAKE_REPLAY_CAPTURE_LESION=1` cuts the edge: the reads still run, R_eff = 0, and nothing is changed.
- `webapp/da_tag_capture.py`: `early_expression()` is the write's decay, or the bout's re-induced early phase for a
  block that carries one. `weight_factor` uses it, and a rewrite clears it. With the flag off no block carries one.
- `webapp/da_tag_capture_chat.py`: `tick()` runs the bout after the sleep catch-up.
- Tests (`tests/test_awake_replay_capture.py`): with the flag off, the store hash equals the pre-branch `tick()`
  (exact compare), and the awake-lesion store equals the flag-off store.
- Battery (label-only groups): the `awake_rest_*` world steps. Per rest period the clock moves, every live ledger is
  marked awake, and the engine runs ONE light idle tick at `IDLE_SEC` after the last request (never sleep depth).
  Every arm of a group gets the same ticks; only the flags differ.

### Does awake replay supply PRP? No, by default and in this code (decided before any run)

The PRP trigger in this model is dopamine through D1/D5 (Kandel ch.54). Moncada & Viola 2007 found that a NOVEL, not a
familiar, environment supplies the PRPs, D1/D5-dependently. Gomperts, Kloosterman & Wilson 2015 recorded VTA cells
during quiet-wake SWR replay; the cells that coordinated with replay were reward-responsive, on appetitive tasks. A
plainly told, familiar fact has neither reward nor novelty, so its awake reactivation has no basis for a DA
co-release. A PRP from rest alone would also make every ordinary fact permanent during the day with no role for DA,
against Bethus, Tse & Morris 2010. So a rested neutral fact is held in early phase. It becomes late-phase only if PRP
arrives while its tag is live: from a later salient waking turn, or from the night's SWR-coupled DA.

The same Gomperts abstract reports that reward-responsive coordination with SWRs was diminished in slow-wave sleep.
That is a tension for the sleep route's SWR-coupled DA, which rests on Clopath et al. 2008's modelling assumption. It
is recorded in the biology file and not resolved here.

### Constants (a priori; none fitted to a gate seed)

| constant | value | where it comes from |
|---|---|---|
| bout window | 5 min (`AWAKE_BOUT_H`) | the sleep route's `SWR_BOUT_H` = the v3 `CAPTURE_PROTOCOL_MIN`, reused; Sadowski replayed the first 5 min of rest |
| induction | e <- e + R_i (1 - e) | declared form, shaped by Sadowski's graded (pairing-count) dependence; no free constant |
| tag after a bout | e_new x the local increment | v3's invariant (the write tag is the increment times the early phase) |
| rest tick depth | `IDLE_SEC` after the last request | the engine's first idle tick; never sleep depth |
| primary rest protocol (`datr`) | a rest tick every 5 min through the 4 waking hours (48) | the most rest the bout window allows |

### Design sweep and the choice of protocol (fake substrate; not brain evidence)

The sweep runs the real ledger, sleep and awake code on a fake read-back and a linear D1. The read-back is a Hill curve
fitted to the ten (increment-to-baseline ratio, read) pairs of the committed seed-42 smokes; its largest error on them
is 0.004003. Nothing was measured between ratio 0.14 and 1.1, so that part of the curve is interpolated. The fact is
the neutral telling, told 4 h before the awake mark; sleep follows.

| schedule | bouts | expression at 4 h | read at sleep onset | SWR DA | captured by 28 h |
|---|---|---|---|---|---|
| no rest | 0 | 0.069483 | 0.009973 | 0.507380 | no |
| a bout every 5 min | 48 | 0.930665 | 0.420416 | 0.811108 | yes |
| every 30 min | 8 | 0.548409 | 0.309722 | 0.729194 | yes |
| every 60 min | 4 | 0.226352 | 0.098737 | 0.573065 | no |
| every 5 min, awake edge cut | 48 | 0.069483 | 0.009973 | 0.507380 | no |
| every 5 min, night edge cut | 48 | 0.930665 | 0.420416 | 0.5 | no |
| rest in the first hour only | 12 | 0.125962 | 0.033296 | 0.524639 | no |
| rest in the last hour only | 13 | 0.921379 | 0.418855 | 0.809952 | yes |

Two properties show before any brain run. First, the dose matters: on this curve a bout every 30 min is enough and one
an hour is not. Second, rest regrows a faint trace: in the last-hour row the first bout read 0.042683 and the fact was
still captured. The brakes the animal presumably has are absent here: competition among many recent assemblies for
the SWR's content, and synapse-by-synapse reversal of decaying E-LTP (the ledger shrinks the whole pattern
uniformly). Both properties are measured on the brain as REPORTED arms (`lq_arc`, `lz_arc`), not gated.

### Arms (`--family arc`; each a fresh tiny-demo brain in its own subprocess; numpy; LTM off)

Env as in the base family: `BRAIN_DA_TAG_CAPTURE_CLOCK=turn`, `BRAIN_LTM_SHIP_DEFAULT=0`, and the seed through
`BRAIN_CHAT_SEED` (threaded to `cfg.seed`; never `actual_seed_used`). ON = `BRAIN_DA_TAG_CAPTURE=1`; RC =
`BRAIN_SLEEP_REPLAY_CAPTURE=1`; ARC = `BRAIN_AWAKE_REPLAY_CAPTURE=1`. Groups:
- `datr`: the neutral telling (as `datn` and `datl`), then `awake_rest_4h` (48 rest periods of 5 min), the 24 h night,
  recall.
- `datcr`: the salient telling (as `datc`), the same rest, night, recall.
- `datq`: the neutral telling, `awake_rest_hourly_4h` (4 rest periods of 1 h), night, recall.
- `datz`: the neutral telling, `awake_3h` (no tick), then `awake_rest_1h` (12 periods of 5 min), night, recall.
- `datl`: Amendment 1's group (4 h awake, no idle tick); `datni`: immediate recall.

| arm | group | env | role |
|---|---|---|---|
| `lr_arc_a` | datr | ON + RC + ARC | ARC1 |
| `lr_arc_b` | datr | ON + RC + ARC | G0 null rebuild |
| `lr_noarc` | datr | ON + RC | ARC1 (the same rest, awake route off) |
| `lr_arc_lesion` | datr | ON + RC + ARC + `BRAIN_AWAKE_REPLAY_CAPTURE_LESION=1` | ARC2 |
| `ln_arc` | datl | ON + RC + ARC | ARC3 (no idle tick) |
| `lr_arc_sleeplesion` | datr | ON + RC + ARC + `BRAIN_SLEEP_REPLAY_CAPTURE_LESION=1` | ARC4 |
| `lr_arc_dalesion` | datr | ON + RC + ARC + `BRAIN_DA_ENCODING_LESION=1` | ARC5 |
| `lsr_arc_sleeplesion` | datcr | ON + RC + ARC + `BRAIN_SLEEP_REPLAY_CAPTURE_LESION=1` | ARC6 |
| `neu_imm_arc` | datni | ON + RC + ARC | P1 |
| `lq_arc` | datq | ON + RC + ARC | REPORTED (rest dose) |
| `lz_arc` | datz | ON + RC + ARC | REPORTED (late rest: regrowth) |
| `lr_ledger_off` | datr | ledger OFF | REPORTED (today's default, same rest) |

### Gates (per seed; `grade_seed_arc` implements them verbatim)

The REPORTED arms (`lq_arc`, `lz_arc`, `lr_ledger_off`) enter no gate, error count, gamma check or UNDEFINED rule,
except ARC7, which reads every arm.

**UNDEFINED** (never a pass or a fail) if any of:
- G0: `lr_arc_a` and `lr_arc_b` differ in outcome, recalled_svo, abstained, the ledger state at recall, the per-block
  ledger summary, the sleep record or the awake record;
- P1: `neu_imm_arc` is not correct;
- I1: on a gated arm the awake branch did not run as scheduled. An ARC arm's bout count differs from its group's rest
  ticks (`datr` and `datcr` 48, `datl` 0, `datni` 0); a bout lacks a substrate read for a block; a night arm does not
  have exactly one sleep epoch, after an awake mark of at least 4 h, with every bout before it; or a flag-OFF arm has
  an awake record;
- I2: a lesion did not hold on the record. Awake lesion: every bout's R_eff is 0, no bout changed the expression, and
  no block carries an awake early phase. Night-edge lesion: the sleep epoch's R_eff are all 0 and its SWR DA is tonic.
  DA-encoding lesion: the DA the D1 pool saw at the epoch is tonic;
- I3: an awake bout supplied PRP. On a gated arm with at least two bouts, the number of D1 drive entries changed
  across the bouts, or the PRP pool rose from one bout to the next;
- gamma differs across the gated ledger-on arms; a gated arm errs; a gated arm reads undefined.

**GO for the seed** iff all of:
- ARC1 (the rescue): `lr_arc_a` correct AND `lr_noarc` abstain;
- ARC2 (the awake edge carries it): `lr_arc_lesion` abstain;
- ARC3 (the rescue comes from rest, not from the flag): `ln_arc` abstain;
- ARC4 (awake replay alone does not make an ordinary fact permanent; the night's DA-coupled capture does):
  `lr_arc_sleeplesion` abstain;
- ARC5 (DA stays the gate with both routes armed): `lr_arc_dalesion` abstain;
- ARC6 (salient-vs-neutral separation kept): `lsr_arc_sleeplesion` correct. With the night's edge cut, the salient
  fact is kept by its waking DA capture, while the neutral one (ARC4) is not;
- ARC7: no confab in any arm, the REPORTED arms included.
Otherwise NO-GO.

**REPORTED, never gating:** the outcomes and I1 records of `lq_arc`, `lz_arc` and `lr_ledger_off`. For every arm: the
bout count, the read at the first and the last bout, the expression after the last bout, the read and SWR DA at sleep
onset, and the fraction of synapses with z > 1/2 at sleep onset and at recall.

**Predictions.** ARC1-ARC7 hold. On `lr_arc_a` the sleep-onset read is close to a fresh fact's, and z at sleep onset
is near 0 (the fact is held in early phase, not captured while awake). `lr_ledger_off` is correct. `lq_arc` abstains
and `lz_arc` is correct, as on the fake curve; both are uncertain, because the curve's middle range is interpolated.
If `lz_arc` is correct on the brain, the regrowth is a real property of this model, and its brake is the named next
rung.

**6-seed verdict** (`--family arc --aggregate research/findings/raw/_awake_replay_capture`): GO iff all six seeds read
GO; INCOMPLETE if a seed is missing; otherwise NO-GO. Reported with it: the one-sided exact sign-flip p over seeds for
`lr_arc_a` minus `lr_noarc`, and for `lr_arc_a` minus `lr_arc_lesion` (1/64 at 6/6), and the correct counts of the
REPORTED arms.

### What each gate can and cannot show

- ARC1 and ARC2 tie the rescue to the awake reactivation edge. `lr_noarc` controls for everything else the idle ticks
  run (mood relaxation, the Turrigiano pass, the wander). ARC3 shows the flag does nothing without idle ticks.
- ARC4 and I3 show, by behaviour and on the record, that the awake route supplies no PRP.
- ARC5, like G1 and RC5, is partly an integrity check: pinning the DA the D1 pool sees to tonic removes the night's
  drive by construction. What it adds is that nothing else in the two routes captures without DA.
- The selection is brain-read (the store's own cleanup margin). The induction law, the bout window and the rest
  protocol are declared operating points (module docstring, HOST SHORTCUTS); a GO does not retire them.
- One fact per conversation: no measurement of many facts competing for rest replay. Not claimed.
- The primary protocol is the most rest the bout window allows. A GO says rest can carry a fact to sleep in this model,
  not how much rest a person needs.

### Compute for this amendment

- LTM off, as in the other families.
- A seed-42 smoke of `--family arc` runs AFTER this commit, under `bash tools/memcap.sh 8`, with `--workers 1`, to
  `research/findings/raw/_awake_replay_capture_smoke`. It is a de-risk, not a gate row. If it forces a code change,
  that change is Amendment 5 and the pin moves.
- The six gate rows are pool runs at a full-SHA-pinned revision containing this amendment:
  `--family arc --seed N --ltm off --workers 1 --out research/findings/raw/_awake_replay_capture`. They are not queued
  with this commit.

## Amendment 5 (2026-09-25, branch research/awake-replay-capture) — a corpus guard in the probe; the seed-42 smoke

Committed before any gate row of the arc family runs (no pool line is queued). The change is to the instrument only:
`run_seed` now refuses (exit code 3) when `data/corpus/` lacks the four core corpus files, as `load_bearing_fraction`
already does. No arm, group, constant or gate changes. The pin for the six gate rows moves to a revision containing
this amendment.

**Why.** The first local launch of the seed-42 smoke ran from this branch's worktree, which has no `data/corpus/`
(those files are excluded from version control). The brain's cross-edge build failed, and the brain degraded to
standalone organs. No arm record carries an error; only a log line showed it. That launch was stopped after one arm,
and its output was deleted. The smoke was re-run with the pool's five corpus files linked into the worktree
(`tools/pool_provision.sh` ships the same files to the pool). Logged in `research/FAILURE_LOG.md`, closed by
`tests/test_awake_replay_capture.py::test_probe_refuses_to_run_without_the_corpus`.

**The seed-42 smoke** (a de-risk, not a gate row). Artifact: `research/findings/raw/_awake_replay_capture_smoke/seed42.json`.
It ran at `84190dbae` with the corpus present, under `tools/memcap.sh 8` with one worker; the probe tree's peak RSS
was 0.85 GB. The grader reads GO: G0, P1, I1, I2 and I3 hold, ARC1-ARC7 hold, and no arm confabulated.
- `lr_arc_a`: correct, with 48 bouts. The read was 0.425497 at the first bout and 0.417001 at the last, and the
  expression after the last bout was 0.929758. At sleep onset the read was 0.416992 and the SWR DA 0.808574; z was 0
  at sleep onset and 1 at recall. `lr_arc_b` is identical.
- `lr_noarc`, `lr_arc_lesion` and `ln_arc` abstain. Their sleep-onset read is 0.005609, the Amendment-1 value.
- `lr_arc_sleeplesion` abstains, with the expression held at 0.929758 and the SWR DA at tonic. `lr_arc_dalesion`
  abstains. `lsr_arc_sleeplesion` is correct, with z already at 1 at sleep onset (captured while awake).
  `neu_imm_arc` is correct.
- REPORTED: `lq_arc` (one rest tick per hour) abstains, as predicted. `lr_ledger_off` is correct, as predicted.
  `lz_arc` (rest in the last hour only) abstains, AGAINST the prediction. The regrowth seen on the fake curve did not
  happen on the brain: the first late bout read 0.008269, not the fake curve's 0.042683, and the expression was held
  near 0.132748, not regrown. The fake curve overstated the read of a faint trace. On the brain, one hour of rest held
  what was left of a 3-h-old trace but did not bring it back.

## Amendment 6 (2026-09-25, branch research/sleep-forgetting-interference) — the companion is later learning; the three-night criterion is withdrawn

Committed on its own, BEFORE any run of the family it governs (no `--family fi` output exists at this commit). It
governs code commit `cbeccb54c` on branch `research/sleep-forgetting-interference`, off `main` at `f793b6945` (the
mechanism is `6a778561e`; `cbeccb54c` fixes the told-fact list). Every constant below is fixed there. Terms follow `docs/TERMS.md`: the mechanism depresses synapses of the same store, so it
is not "consolidation".

The one artifact committed with this amendment is a fake-substrate design sweep, run at `cbeccb54c`:
`research/findings/raw/_sleep_forgetting_interference/design_fake_substrate.json`. It has no brain and no gate seed,
and no gate reads it.

### Why

Amendment 1's item 2 read NO-GO at seed 42 on SHY1 alone (a de-risk, not a gate row;
`research/findings/raw/_sleep_replay_capture_r2_smoke/seed42.json`). After three nights of `BRAIN_SLEEP_DOWNSCALING`
the weak, never re-mentioned fact still recalled, with increment magnitude 0.756291 against a baseline of 0.757967.
Amendment 3's horizon arm (`research/findings/raw/_sleep_replay_capture_r2_horizon_smoke/seed42/d10w_shy.json`)
recalled it through night 6 and lost it on night 7. The constant was not retuned, and this amendment does not retune it.

The wall question: what does the real system run alongside passive downscaling that this model replaced with a
constant? Corpus check first: `bash tools/before_you_build.sh "weak never-re-mentioned fact still recalled after 3
nights of sleep downscaling (SHY1 NO-GO); ..."` (logged). Three candidates, weighed against sources read for this
amendment (binding: `research/biology/sleep-load-dependent-renormalization.md`, 12 sources, the local ones resolving):

1. **Interference from later learning.** Everyday forgetting of recent memories is retroactive interference from
   later memory formation, even when the later material is dissimilar (Wixted 2004, Annu Rev Psychol 55:235). At the
   synapse, LTP decay is driven by later NMDA-receptor-dependent plasticity: blocking NMDA receptors for a week after
   induction blocked the decay (Villarreal et al. 2002). Repeated enriched-environment exposure reversed LTP that was
   otherwise stable for months (Abraham et al. 2002). The model's three-night protocol contains no later learning at all.
2. **Competitive selection in sleep.** It is already in the model twice: untagged early LTP decays within hours
   (`TAU_EARLY_H` 1.5 h), and the night's protection is the block's own reactivation read R_i. With one fact per
   conversation there is nothing for the fact to compete with, so in the three-night protocol it cannot act. It acts
   once later facts exist.
3. **Homeostasis on the store's total, not per trace.** The night's renormalization is the price of the day's
   plasticity. Sleep slow-wave activity after an enriched environment is "positively correlated with the amount of the
   time spent exploring", and the decrease "is exponential and self-limiting" (Tononi & Cirelli 2014, Neuron 81:12). A
   local learning task raises local slow-wave activity (Huber et al. 2004). Kandel ch.44: learning enlarges synapses,
   "requiring that some excitatory inputs be reduced". The 0.18 of de Vivo et al. 2017 is a sleep-versus-wake
   difference measured after normal waking, not after a day with no learning. As a constant, it charges a night after
   a day with no learning the same 18 %.

All three point to one variable the constant stood in for: how much the brain learns after the trace. The three-night
protocol sets it to zero. **Built:** candidate 3 in a form that carries 1 and 2. The night's amplitude is the measured
fraction of the store's strength that the preceding wake added, and the R_i protection decides which traces pay.
**Not built, named next:** wake-time depotentiation by novel experience (Xu, Anwyl & Rowan 1998), and similarity-
dependent overlap of later facts on the same synapses. The store gives every fact its own block, so neither has a
substrate here yet.

### Is the three-night criterion biologically justified? No

- The three-night group is the minimum-interference condition: a fact, then three days with nothing else learned.
  Wixted's account predicts little forgetting there. It explains why sleep, alcohol and benzodiazepines improve memory
  for a recently learned list (they reduce later encoding). Under candidate 3, the nights after an empty day cost nothing.
- People keep a sentence they were told a few times for longer than three days, while living an ordinary week full
  of interference. In Rivera-Lares et al. 2022 (Mem Cognit 50:1706, full text), cued recall of sentences presented two to
  six times was at floor by one week, so the authors moved to three days, where it was above floor. Fisher & Radvansky
  2018 (J Mem Lang 102:130) found propositional (textbase) memory retained for about seven days and then dropped.
  Forgetting in humans follows a power law (Buzsáki 2006, p.123), with no fixed number of days at which a memory is gone.

**So SHY1 is withdrawn as a requirement.** SHY1 is "the weak, never re-mentioned fact is not recalled after three idle
nights". The r2 rules are not edited: `grade_seed_r2` still computes SHY1, and any r2 row reads what it read. A
SHY1 failure is to be read as "the fact is kept after three idle nights", which the biology above predicts. It does
not show a defect, and it is not re-scored as a pass. `BRAIN_SLEEP_DOWNSCALING` keeps its constant. The seed-42 horizon (lost on
night 7) is in the range the human data give, but for the wrong reason: with no later learning it should not fade at all.

What the biology does require replaces SHY1. The fact is kept when nothing else is learned. It is lost as later
learning accumulates, and faster the more is learned. Salience and re-mention protect it. The loss comes through the
renormalization edge. That is the `fi` family below.

### What was built (default OFF: `BRAIN_SLEEP_LOAD_RENORM`)

- `webapp/sleep_replay_capture.py` (r3). The flag is read only inside an SWR epoch, so it is inert without
  `BRAIN_SLEEP_REPLAY_CAPTURE`. After the night's reactivation, re-tag and SWR-coupled D1 drive:
  - dW = sum over managed blocks written or rewritten after the previous night's epoch of mean_k |weight factor_k x
    increment_k| at sleep onset: the learned strength the preceding wake added that is still expressed.
  - W = sum over every store block, managed and build-time, of mean_k |w_k|: the store's total synaptic strength.
  - delta = dW / W, clipped to [0, 1]. Each managed block's increment is multiplied by 1 - delta x (1 - R_i), with
    r2's protection. The baseline and the build-time blocks are not depressed (r2's choice, and Tononi & Cirelli 2014:
    renormalization must not make one "forget old friends").
  - With both `BRAIN_SLEEP_DOWNSCALING` and this flag on, the measured delta replaces the constant.
  - Each epoch records `load = {dW, W, delta_read, delta, lesioned, n_new_blocks, t_since}`.
- `BRAIN_SLEEP_LOAD_RENORM_LESION=1` cuts the load edge: dW and W are still read and recorded, the applied delta is 0,
  and every scale is 1.
- Host steps, declared in the module docstring: the two sums, the ratio and the multiply.
- `tests/test_sleep_load_renorm.py` (13 tests). With the flag off, the store hash equals the pre-branch module's, for the
  route alone and for r2's constant downscaling. Both hashes were computed with `main:webapp/sleep_replay_capture.py`
  swapped in and are pinned in the test. The tests also check that a night after an empty day depresses nothing, that
  the old trace falls in dose order (0 > 1 > 3 facts a day), that the constant ignores the dose, and that the lesion
  reads the load and applies nothing.
- `research/runners/_da_tag_capture_chat_probe.py --family fi`: `FI_ARMS`, `grade_seed_fi`, `aggregate_fi`, and
  selftest rows that read GO, NO-GO and UNDEFINED. No other family's arms, graders or aggregates change.
- `research/runners/onebrain_regression_battery.py`: label-only groups `fiv`, `fil`, `fih`, `fis`, `fir` (below). None
  is in `PROBE_TURNS`, so the regression battery is unchanged.

### Constants (a priori; none fitted to a gate seed)

| constant | value | where it comes from |
|---|---|---|
| night amplitude | dW / W, measured each night | Tononi & Cirelli 2014; Huber et al. 2004; Kandel ch.44 (no free constant) |
| protection | 1 - delta (1 - R_i) | r2, unchanged (González-Rueda et al. 2018) |
| nights | 7, recall asked each morning | the human horizon: at floor by one week (Rivera-Lares 2022; Fisher & Radvansky 2018) |
| heavy dose | 3 facts a day on days 2-7 | the smallest daily count for which the fake predicts loss by night 7 over the whole recall band (below) |
| low dose | 1 fact a day on days 2-7 | the smallest nonzero dose |

The store holds 32 blocks (tiny-demo `k_max`); 5 are build-time. The heavy re-mention group writes 21, within capacity.

### Design sweep (fake substrate; not brain evidence)

`research/runners/_sleep_load_renorm_design.py` runs the real ledger, gamma calibration and sleep epochs, in constant
or load mode, on a fake store. The store has five build-time blocks of unit magnitude (an assumption, declared) and a
block per told fact. The read-back is the Hill curve fitted to the committed seed-42 (ratio, R) pairs, imported from the
awake design runner. A fact counts as recalled while its increment-to-baseline ratio is above the midpoint of the band
the seed-42 horizon arm measured: recalled at 0.638 on night 6, lost at 0.537 on night 7. <!--derived--> The Turrigiano
pass is not modelled. It leaves every ratio unchanged but changes W, so the fake's delta is approximate.

| row | first night not recalled (band) | ratio night 3 | ratio night 7 |
|---|---|---|---|
| constant (r2), no later learning, 10 nights | 7 (7-8) | 0.99741 | 0.543895 |
| load, 0 facts a day | never | 1.28358 | 1.28358 |
| load, 1 fact a day | never in 7 | 1.044618 | 0.774625 |
| load, 2 facts a day | 7 (6 to later than 7) | 0.904428 | 0.569353 |
| load, 3 facts a day | 6 (5-6) | 0.821384 | 0.460043 |
| load, 4 facts a day | 5 (4-5) | 0.753466 | 0.389835 |
| load, 3 a day, load edge cut | never | 1.461 | 1.461 |
| load, 3 a day, salient telling | never in 7 | 1.624374 | 1.052703 |
| load, 3 a day, re-mentioned after nights 1 and 2 | never in 7 (the re-mention blocks) | 0.874636, 1.200591 | 0.524126, 0.76307 |

The fake reproduces the brain's r2 horizon: the constant loses the fact on night 7, as seed 42 did. On the telling
night, the load read on the fake is delta 0.188877693 (dW 1.187710092 over W 6.288249694). That sits next to de Vivo's
0.18 without being set to it. On the brain it is a prediction, not a calibration. Under load, nothing fades after an
empty day. After that, the loss night moves earlier as the dose grows.

### The told facts (the environment), checked before this commit

Every told sentence uses an animate agent (dog, bird, fish, worm), `use` or `store`, and an inanimate patient (river,
memory, spikes). None shares a content word with the fact. The tiny-demo's spiking comprehension gate does not store
`eat` or `learn` sentences, animate patients, or most `words` patients. An environment check (tag-capture ledger on, no
night, no recall of the fact, no gate read) stored each of the 18 sentences below as one new block, with no external rewrite,
at seed 42 (all 18, in this order) and at seed 7 (13 of the 18; the other five were not tried there). The pool seeds were not checked. At a seed where a told sentence is not stored, the dose was
not delivered, and I1 reads that seed UNDEFINED.

Days 2-7, three a day, in order (`_FI_FACTS` in `research/runners/onebrain_regression_battery.py`):
- day 2: the bird uses the river / the dog stores the memory / the fish stores the spikes;
- day 3: the worm uses the river / the fish stores the memory / the dog uses the spikes;
- day 4: the bird stores the spikes / the worm stores the memory / the dog uses the river;
- day 5: the fish uses the river / the bird stores the memory / the dog stores the spikes;
- day 6: the bird uses the spikes / the fish uses the spikes / the worm stores the spikes;
- day 7: the dog uses the memory / the bird uses the memory / the fish uses the memory.

The low dose tells the first six, one a day.

### Arms (`--family fi`; each a fresh tiny-demo brain in its own subprocess; numpy; LTM off)

Env as in the other families: `BRAIN_DA_TAG_CAPTURE_CLOCK=turn`, `BRAIN_LTM_SHIP_DEFAULT=0`, and the seed through
`BRAIN_CHAT_SEED` (threaded to `cfg.seed`; never `actual_seed_used`). ON = `BRAIN_DA_TAG_CAPTURE=1`; RC =
`BRAIN_SLEEP_REPLAY_CAPTURE=1`; LR = `BRAIN_SLEEP_LOAD_RENORM=1`; SHY = `BRAIN_SLEEP_DOWNSCALING=1`. Every group is the
weak telling (as `d3w`), then seven nights. After each night the recall question is asked, which is a read-only probe
in this model. On days 2-7 the group's facts are told after that morning's question.

| group | telling | later facts | re-mention |
|---|---|---|---|
| `fiv` | weak | none | - |
| `fil` | weak | 1 a day | - |
| `fih` | weak | 3 a day | - |
| `fis` | salient (as `d3c`) | 3 a day | - |
| `fir` | weak | 3 a day | "the cat chases the ball" after nights 1 and 2, before that day's facts |

| arm | group | env | role |
|---|---|---|---|
| `fiv_lr` | fiv | ON + RC + LR | FI1 |
| `fil_lr` | fil | ON + RC + LR | FI3 (its ratio); outcome REPORTED |
| `fih_lr_a` | fih | ON + RC + LR | FI2 |
| `fih_lr_b` | fih | ON + RC + LR | G0 null rebuild |
| `fih_lr_lesion` | fih | ON + RC + LR + `BRAIN_SLEEP_LOAD_RENORM_LESION=1` | FI4 |
| `fih_shy` | fih | ON + RC + SHY | REPORTED (r2's constant under the same dose) |
| `fis_lr` | fis | ON + RC + LR | FI5 |
| `fir_lr` | fir | ON + RC + LR | FI6 |
| `neu_imm_fi` | datni | ON + RC + LR | P1 |

### Gates (per seed; `grade_seed_fi` implements them verbatim)

The daily outcome is read at each morning's question: correct (recalled_svo = cat/chase/ball), abstain, confab or
undefined. `fih_shy` enters no gate, error count, gamma check or UNDEFINED rule, except FI7, which reads every arm.

**UNDEFINED** (never a pass or a fail) if any of:
- G0: `fih_lr_a` and `fih_lr_b` differ in the daily outcomes, the daily recalled_svo, the ledger state at recall, the
  per-block ledger summary, or the sleep record;
- P1: `neu_imm_fi` is not correct;
- I1: on a seven-night arm, fewer or more than seven epochs ran; a block had no substrate read; a load record is
  present without LR or missing with it; a scale record is present without LR or SHY; a told sentence (fact or
  re-mention) was not stored as exactly one new block; or on an LR arm the load read did not count the registered
  dose: 1 new block on night 1, then the group's daily count, plus 1 on a re-mention day;
- I2: the load lesion did not hold on the record. Every lesion epoch must read `lesioned`, an applied delta of 0,
  every scale 1, and a read delta above 0 on nights 2-7. On an intact LR arm the applied delta must equal the read.
- I3: `fiv_lr`, `fil_lr`, `fih_lr_a`, `fih_lr_b` and `fir_lr` differ in the outcome, the ledger blocks or the sleep
  record at the first morning. They are the same brain until the first dose.
- gamma differs across the gated arms; a gated arm errs; a gated daily outcome reads undefined.

**GO for the seed** iff all of:
- FI1 (kept when nothing else is learned): `fiv_lr` correct on all seven mornings;
- FI2 (later learning erases it within a week): `fih_lr_a` abstains on the seventh morning;
- FI3 (ordered by dose): at the seventh morning, the fact block's increment-to-baseline ratio is `fiv_lr` >
  `fil_lr` > `fih_lr_a`;
- FI4 (the load edge carries it): `fih_lr_lesion` correct on all seven mornings;
- FI5 (salience protects): `fis_lr` correct on the seventh morning;
- FI6 (re-mention protects): `fir_lr` correct on the seventh morning;
- FI7: no confab on any morning of any arm, nor on `neu_imm_fi`.
Otherwise NO-GO.

**REPORTED, never gating.** Per arm:
- the first morning the fact is not recalled;
- whether it is recalled on the third morning (the human three-day anchor);
- the fact block's ratio, R, delta, dW and W per night;
- the managed block count per morning.
Also the `fil_lr` and `fih_shy` outcomes.

**Predictions** (from the fake; uncertain where noted):
- FI1-FI7 hold.
- `fih_lr_a` is still recalled on the third morning and first lost on night 5 or 6. This is uncertain: the brain's W
  (Turrigiano, the true build-block magnitudes), its R for new facts and its DA-gated write gains differ from the fake.
- `fil_lr` is recalled on all seven mornings.
- `fih_shy` is first lost on night 7 or later, as the constant ignores the dose.
- On the brain, delta on night 1 is close to 0.19.
A NO-GO on FI2 would mean this renormalization, at the store's own measured load, does not erase a weak fact within a
week at three facts a day. It would not mean that later learning does nothing: FI3 reports the ordering either way.

**6-seed verdict** (`--family fi --aggregate research/findings/raw/_sleep_forgetting_interference`): GO iff all six
seeds read GO; INCOMPLETE if a seed is missing; otherwise NO-GO. Reported with it: the one-sided exact sign-flip p over
seeds for `fiv_lr` minus `fih_lr_a` correct on the seventh morning, and for `fih_lr_lesion` minus `fih_lr_a` (1/64 at
6/6), and the first-lost night per seed for each arm.

### What each gate can and cannot show

- FI1 and FI2 together tie the loss to later learning: the same brain, the same telling, the same first night (I3).
  FI4 ties it to the renormalization edge. Without the edge the same told facts leave the fact alone, so it is not lost
  by crowding the store or by any other route.
- FI3 is the dose-response on a continuous read. It holds even if the outcome boundary falls between doses.
- FI5 and FI6 show that the loss is selective to the unrehearsed, unsalient trace. The protection is the r2 R_i read.
  The salient trace starts larger and the re-mention writes fresh blocks.
- Declared, not measured: whether this model's daily dose corresponds to any human day. The store grows by a block per
  fact, so W grows with knowledge. A real brain renormalizes its total back each night, so this model's delta falls
  faster with accumulated knowledge than a real brain's would. The bias is towards retention.
- The told facts are dissimilar to the target. Similarity-dependent interference (A-B, A-C) is not tested here.

### Compute for this amendment

- A seed-42 smoke of `--family fi` runs AFTER this commit, under `bash tools/memcap.sh`, to
  `research/findings/raw/_sleep_forgetting_interference_smoke`. It is a de-risk, not a gate row. If it forces a code
  change, that change is Amendment 7, and the pin moves.
- The six gate rows are pool runs at a full-SHA-pinned revision containing this amendment:
  `--family fi --seed N --ltm off --workers 1 --out research/findings/raw/_sleep_forgetting_interference`. They are not
  queued with this commit.
