---
type: finding
status: no-go
lane: load-bearing
date: 2026-09-24
mechanism: A10 (midnight plan S15c) seed-7 de-risk of the surprise-organ prediction-error input to da-mode-drives-response's SNc afferent (flags BRAIN_REWARD_VALUE_AFFERENT / BRAIN_REWARD_VALUE_LESION, default-OFF), per research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md and its AMENDMENT-1 and AMENDMENT-2
seeds: [7]
artifacts:
  - research/findings/raw/_reward_value_afferent_derisk/s7.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_verdict.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/rf/s7_verdict.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/offidentity_module.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_off_a.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_on_a.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/rf/s7_arms_on_a.json
---

# A10 seed-7 de-risk: NO-GO at the pre-registered criteria in both runs; the rerun shows (A) holds and traces the lesion residual to surprise-block identity

Governed by `research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md` (57f0ebfd0),
`...-PREREG-AMENDMENT-1.md` and `...-PREREG-AMENDMENT-2.md`. Seed 7 only: a dev/calibration seed, not a gate verdict.
The flags stay default-OFF.

## Fix round 2 (after the review of 7d5c2743d): a flag-ON side effect the v2 record did not mention

**The v2 arms show that turning the flag on changed the default-ON surprise faculty's reading.** The server's own
surprise block, which runs later in the same turn, read CONFIRM `surprise.surprise_hz` 0.4050925925925926 Hz in off_a,
off_b, les and the pre-patch reference, and 0.3472222222222222 Hz in on_a, on_b and rf/on_a
(`research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_off_a.json`, `research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_on_a.json`, `research/findings/raw/_reward_value_afferent_derisk/v2/rf/s7_arms_on_a.json`). The CONTRADICT read is equal in every
arm, and no `surprised` decision flipped at seed 7. The earlier text of this finding did not report the shift, and
the pre-registration's statement that the lesion does not touch the production surprise block is withdrawn for the
intact read (AMENDMENT-2).

- **Cause.** The A10 read runs first in the turn, on the same process-shared organ. Reads on the shared merged pool
  depend on read history (the pool bridge has no `_rest_extra`, so `_hard_reset` does not restore the surprise
  slice's adaptive thresholds, activity EMA or refractory state). With the flag on, the production read was the
  organ's second CONFIRM read of the turn. Reconsolidation (default-ON, off in the arms) gates on that read.
- **A second lesion asymmetry.** The lesion arm's A10 read uses the standalone twin, so its production read stayed at
  0.4050925925925926 Hz while the intact arm's moved. The v2 (B) attribution compares arms that differ in this too.
- **Fix.** The A10 read now leaves no footprint: it snapshots every piece of state it can mutate and restores it
  right after the read (`webapp/reward_value_afferent_chat.py`; unit pins in `tests/test_reward_value_afferent.py`
  fail on the pre-fix module). AMENDMENT-2 adds criterion (D): the production `surprise` and `reconsolidation`
  blocks of on_a and les equal off_a's on both turns. Scored on the v2 arms, (D) fails, as it should.
- **Measurement of the fix.** A module-level check on the production organ
  (`research/runners/_reward_value_afferent_footprint.py`) and the v3 arms on the pool, both governed by
  AMENDMENT-2. Results are reported below as they land; until then the fix is unmeasured at the handler level.

## Corrections to the earlier text of this finding (fix round, after the adversarial review of 58c400ff6)

The first version of this file said "(A) GO, (B) GO, (C) UNDEFINED". Three of those words were wrong:

- **(B) was not GO.** The pre-registration's (B) includes `attributable_to(...) against the lesioned control >= 0.9`.
  The runner left that term out. The measured attribution is 0.8795180722891565, below 0.9. So (B) FAILS.
- **(A) was not measured as pre-registered, and "byte-identical" was not earned.** The pre-registration asks for
  `da_drives` identical to a pre-patch build with the same env. The runner only checked that the OFF turns had no
  `reward_value` key and compared two OFF turns with each other. docs/TERMS.md requires byte-identity to be shown in
  data. The reviewer then ran a pre-patch vs post-patch module-level check (sha256 equal, flag unset), but that is
  the reviewer's evidence, not this run's.
- **(C) is NO-GO, not UNDEFINED.** The criterion was measured (differential 0.10767 > 1e-6) and it is false. The
  runner reported UNDEFINED only because every criterion was wired as a `Verdict.require` precondition, so the
  runner could never print NO-GO.

The first version also stated a cause for the lesion residual as fact: homeostat-equalized trained blocks vs spare
blocks. That is withdrawn as unverified. The lesioned twin is a standalone bridge that never runs the homeostat,
and with patient_expected->surprise zeroed the cue has no route to the surprise pool. The cause is now a hypothesis
(surprise block s vs block t), and the rerun tests it (AMENDMENT-1, instrument change 5).

## The first run's numbers (artifact `s7.json`, commit 58c400ff6)

Provenance caveat: the run started at 12:10:38 with `git_sha` 57f0ebfd0 and `git_dirty=true`, before the code was
committed at f3fa99c4a (12:13:06). It used the `rf` composer (host closed-form recall), and its six tiny-demo brain
builds ran on the local box. Its numbers are kept as a record; the v2 rerun below replaces them.

| arm | turn | surprise_hz | normalized | surprised |
|---|---|---|---|---|
| ON | CONFIRM ("the dog chase the cat") | 0.3472222222222222 | 0.0646029609690444 | false |
| ON | CONTRADICT ("the dog chase the fish") | 5.150462962962964 | 0.9582772543741588 | true |
| LESION | CONFIRM | 4.62962962962963 | 0.8613728129205921 | true |
| LESION | CONTRADICT | 5.208333333333334 | 0.9690444145356663 | true |

- Live differential 0.8936742934051144; lesion differential 0.10767160161507416; attribution
  0.8795180722891565 (pre-registered bar 0.9).
- (B) FAIL (attribution below 0.9). (C) FAIL (differential not below 1e-6). **At the pre-registered criteria the
  first run is NO-GO on (B) and (C).** (A) is unmeasured.
- The run stored only the `reward_value` sub-dict for the ON and lesion arms: no `da_drives.mode`, `lead`,
  `afferent_pA` or reply text. So it cannot say whether the reply changed.
- The OFF arm shows the pre-existing afferent cannot tell the two turns apart. Both read `afferent_pA`
  427.6363636363635 and `mode` "neutral". But the novelty term in that path was already spiking (`spiking_novelty`)
  and went through the spiking `shared_salience` afferent. The first text called it "the host `engagement_of()`
  scalar", which understated it.

## What the review changed in the mechanism (see AMENDMENT-1)

- The surprise read now replaces only the per-turn engagement mix, inside `DaModeDrivesWorkspace.observe`. The
  first version replaced the whole SNc current, skipping the novelty organ, the salience afferent, the EMA and the
  content-free HOLD.
- The affect-valence fallback is removed. It was untested and it would have fired on most turns.
- An error no longer drives the SNc at 0 pA.
- The signal is named for what it is: an unsigned prediction-error salience, not a reward value.

## Rerun under AMENDMENT-1 (v2, seed 7): NO-GO at the pre-registered criteria

Verdicts: `research/findings/raw/_reward_value_afferent_derisk/v2/s7_verdict.json` (production-default composer,
onebrain) and `research/findings/raw/_reward_value_afferent_derisk/v2/rf/s7_verdict.json` (forced `rf`). Arms ran on pool1 from git-archive revisions: the head at
d82ce8c8e and the pre-patch reference at the main merge parent 306ef27d7 (`git_dirty=false`, source manifest
verified at exit). Score mode ran locally at 0e0984522 and e17a4bddf on a clean tree; it builds no brain. Every
scored number below is the same under both composers: the surprise path recalls the same patient ("cat") either
way. The handler's `activity.composer` label reads `onebrain` in every default arm and `rf` in every forced arm.

**Preconditions (all hold, both composers).** Every arm built and ran both turns. The pre-patch reference built.
The surprise read drove both ON turns and reached the workspace (`turn_signal_source == spiking_surprise`). The
lesioned read drove both turns. The lesion cut holds at read time on both turns: the twin's
patient_expected<->surprise weight sum is 0.0, against 96768.0 on the intact organ. The ON null control
(on_a vs on_b) and the OFF null control (off_a vs off_b) are clean.

| criterion (pre-registered) | measured | result |
|---|---|---|
| (A) OFF: no `reward_value` key; `da_drives` and `answer` equal to the pre-patch main build, same env | equal on both turns; the whole OFF-arm output file equals the pre-patch file byte for byte (sha256 prefix 210e7bdc70a76853 onebrain, 6ef6ac4551473a3c rf) | holds |
| (B) source `surprise` on both turns | yes | holds |
| (B) contra > confirm, `normalized` | 0.9582772543741588 vs 0.07537012113055182 | holds |
| (B) contra > confirm, `da_drives.afferent_pA` | 840.0 vs 0.0 | holds |
| (B) `attributable_to(live vs lesion) >= 0.9` | 0.8780487804878047 | FAILS |
| (C) lesion differential < 1e-6 | 0.10767160161507416 (0.8613728129205921 confirm, 0.9690444145356663 contra) | FAILS |

**Verdict: NO-GO** (runner status `NO-GO`; A holds, B and C fail). Seed 7 is a dev seed, so this is a de-risk
result, not a gate verdict. Live differential 0.8829071332436069, lesion differential 0.10767160161507416.

**Module-level OFF identity** (`research/findings/raw/_reward_value_afferent_derisk/v2/offidentity_module.json`, commit d0e6f2036): `observe_turn` pre-patch (main
306ef27d7) vs post-patch, with the flag unset and set to "0", over fake sessions: every sha256 is 059a4afd... The
pre-patch run repeated gives the same hash (determinism control). The flag ON with a stubbed drive gives a different
hash, b41201b2... (sensitivity control). With the handler-level byte-for-byte equality above, the OFF path is
byte-identical in the data on these two turns at seed 7 (docs/TERMS.md: asserted by hash and exact compare).

### Where the lesion residual comes from (block-asymmetry control, measured)

The CONTRADICT read drives surprise block 8 (the asserted patient "fish"); the CONFIRM read drives block 0 (the
stored patient "cat"). On a standalone copy of the lesioned twin (same seed, same build path), the runner
replicated both lesioned reads exactly (4.62962962962963 Hz and 5.208333333333334 Hz) and then read each block
with NO cue: block 0 at 4.62962962962963 Hz, block 8 at 5.208333333333334 Hz. So under the lesion each read equals
its own block's cue-free rate, and the whole lesion residual (0.5787037037037042 Hz, `cuefree_fraction_of_residual`
1.0) is the rate difference between block 8 and block 0 with no cue at all. The hypothesis in the correction above
is confirmed in the data: the residual is surprise-block identity, not a prediction effect that survives the cut.

What this means for the criteria: (C) asks two DIFFERENT surprise blocks to read the same rate under the lesion.
The design routes the asserted patient to a different block on every CONTRADICT turn, so a block-rate difference
fails (C) whatever the prediction pathway does. (B)'s attribution uses the same lesion differential as its control,
so the same block difference is counted as effect not owed to the live read. The pre-registered criteria stay as
written and the verdict stays NO-GO. The block-matched criterion this paragraph once pointed to was withdrawn in
fix round 2 (it cannot fail once the cut holds); the next rung is the substrate-matched within-block (B'), see Next
action. Declared, still true: the twin is a separate bridge with no homeostat, so the intact and lesioned
arms still differ in substrate as well as in the zeroed edges. The block control explains the residual WITHIN the
twin; it does not make the twin a substrate-matched cut of the intact read.

### Does the reply change under the lesion? (midnight plan S15 success check, reported, not part of the GO)

| turn | intact (on_a; on_b identical) | lesion | changes |
|---|---|---|---|
| CONFIRM "the dog chase the cat" | mode `rest`, afferent 0.0 pA, DA level 0.04616293556102311, answer "the dog chases the cat" | mode `focus`, afferent 671.9999999999999 pA, DA level 0.7793389498539759, answer "the dog chases the cat — worth going further here." | yes: mode, lead and answer text |
| CONTRADICT "the dog chase the fish" | mode `focus`, afferent 840.0 pA, DA level 0.8965045558416256, answer with the " — worth going further here." lead | the same | no |

The lesion changes the reply on the CONFIRM turn only, which is what finding
2026-09-20-hollow-surprise-monitor-confirm-probe predicts (the prediction cancels the surprise pool on a confirming
assertion; the lesion removes that cancellation). The ON null control is clean and the OFF path is byte-identical,
so the change is owed to the lesion. One seed only: this is not a load-bearing measurement in the LBF sense.
Added in fix round 2: in the intact arms the production surprise read on CONFIRM was also shifted by the A10 read
(0.3472222222222222 vs 0.4050925925925926 Hz, `surprised` False either way), so the surprise NOTICE and the answer
text were not affected by it at seed 7; the lesion arm's production read was not shifted. (D) re-measures this.

### Other observations

- The intact CONFIRM rate moved from the first run's 0.3472222222222222 Hz to 0.4050925925925926 Hz. The CONTRADICT
  and lesioned rates are unchanged. **Corrected (fix round 2): the cause is read history, not the uncommitted code or
  the main merge.** The v1 runner (f3fa99c4a `main`) ran every arm in ONE process through `brain_chat`: OFF CONFIRM,
  OFF CONTRA, then the ON CONFIRM turn. So v1's A10 CONFIRM read was the process-shared organ's third read and its
  second CONFIRM read. v2 ran each arm in a fresh process, so its A10 CONFIRM read was the organ's first. v2's own
  arms show a second CONFIRM read gives exactly 0.3472222222222222 Hz (the production read in on_a, see Fix round 2).
  This is the same mechanism as the side effect above. The module-level check reproduces the v1 order as a reported
  sequence. (The earlier text of this bullet pointed at uncommitted code or the main merge; that is withdrawn.)
- `reward_value.composer` read null on every turn of this run: the module looked for `chat.inner.agent.composer`,
  but the agent holds `.composer` itself. Fixed after the run in 3e860cdc6 (pinned by a test); the handler's own
  `activity.composer` label is the composer record for this run.

## Next action

1. Harvest the v3 runs AMENDMENT-2 governs (module-level footprint check; v3 arms and fresh pre-patch references
   for both composers) and score them. (D) decides whether the fix holds at the handler level.
2. **Withdrawn (fix round 2): the block-matched criterion (C') proposed here earlier.** It read
   |lesion_hz - cuefree_hz(block)| < 1e-6 on the twin. With patient_expected->surprise zeroed there is no other route
   from the cue to the surprise pool, so once the read-time cut holds the lesioned read equals the cue-free rate by
   construction (v2 shows exact equality). It tests the cut, not whether the live read carries the effect, and it
   would be scored on seed-7 data already seen. It must not go to B2b as a GO criterion.
3. The meaningful next rung is a substrate-matched lesion with a within-block attribution (B'): zero the
   patient_expected->surprise edges on the intact organ's own slice of the shared pool for the read (restored after
   it) instead of reading the standalone twin, and compare the CONFIRM-block suppression (cue-free rate minus CONFIRM
   rate) intact vs lesioned on one substrate. This is a mechanism change and needs its own amendment before a run;
   B2b's gate seeds are the first data it may be scored on.
4. The S15(c) trace still fails on host classification on the path (both registered in `docs/SCAFFOLD-LEDGER.md`):
   the `extract_assertion` regex/keyword gate, AND the organ's host addressing in `read_surprise`. That addressing
   maps the recalled and asserted patient strings to blocks through the host dict `_block_for`, and decides confirm
   vs contradict by a host string compare (`str(p_asserted).lower() == str(p_stored).lower()` sets t = s, plus a
   collision step forcing t != s). It also drives the cue at block s from the recalled word, not from agent/action.
   The spiking circuit then turns s == t into a low rate. Both need an owner waiver or a spiking replacement.
5. Out of this lane: the production surprise read is read-history dependent on the merged pool (a second CONFIRM
   assertion in one process reads 0.3472222222222222 Hz, not 0.4050925925925926 Hz). Logged in
   `research/FAILURE_LOG.md` for its own lane; fixing it changes a default-ON faculty and needs 6 seeds.

## Scope

Default-OFF. The 6-seed gate (seeds 42/43/44/100/101/102) belongs to B2b (S28), and default-ON would also need a
SOUND independent review. `extract_assertion` (a regex/keyword gate) and the organ's host string-identity block
addressing are on the path, both registered in `docs/SCAFFOLD-LEDGER.md`; the S15(c) trace cannot pass without an
owner waiver.

For B2b: merging this branch adds one row to the default LBF registry (38 -> 39 faculties, thin 2 -> 3). The row is
`thin` unless `BRAIN_REWARD_VALUE_AFFERENT` is set in the harness, so the headline fraction is unchanged. What the
row's lesion arm measures is whether the surprise PREDICTION reaches the DA mode through A10: the lesion swaps the
source organ's read for its disinhibited twin's read. It does not cut the afferent the row is named for.

Provenance note: the module-level OFF-identity runner ran three times at the same output path, the first time on a
dirty tree. Only the clean run at 3992d343b (`research/findings/raw/_reward_value_afferent_derisk/v2/offidentity_module.json`, committed in d0e6f2036) is cited; the
reviewer's independent rerun against main 355db9c79 reproduced its hashes.
