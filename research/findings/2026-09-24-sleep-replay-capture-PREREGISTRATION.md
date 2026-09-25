---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: sleep-replay-triggered synaptic capture (webapp/sleep_replay_capture.py, BRAIN_SLEEP_REPLAY_CAPTURE, default OFF) -- one SWR epoch at sleep onset reads every DA-tag-capture-managed store block back through the store's own cleanup, re-tags it in proportion, and drives the SAME spiking D1 pool / PRP pool with SWR-coupled DA, so an ordinary fact told once can be captured overnight
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_sleep_replay_capture/design_fake_substrate.json
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
