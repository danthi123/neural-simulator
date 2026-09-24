---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: A10 (midnight plan S15c) seed-7 de-risk of the surprise-organ prediction-error input to da-mode-drives-response's SNc afferent (flags BRAIN_REWARD_VALUE_AFFERENT / BRAIN_REWARD_VALUE_LESION, default-OFF), per research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md and its AMENDMENT-1
seeds: [7]
artifacts:
  - research/findings/raw/_reward_value_afferent_derisk/s7.json
---

# A10 seed-7 de-risk: the first run is NO-GO on (B) and (C) at the pre-registered criteria; (A) was not measured

Governed by `research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md` (57f0ebfd0) and
`...-PREREG-AMENDMENT-1.md`. Seed 7 only: a dev/calibration seed, not a gate verdict. The flags stay default-OFF.

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
builds ran on the local box. Its numbers are kept as a record; the rerun replaces them.

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

## Rerun under AMENDMENT-1

Pending at the time of this correction: arms on the pool at the committed SHA, the pre-patch reference at main, and
the module-level OFF-identity check. The result is appended here when it lands.

## Scope (unchanged)

Default-OFF. The 6-seed gate (seeds 42/43/44/100/101/102) belongs to B2b (S28), and default-ON would also need a
SOUND independent review. `extract_assertion` is a regex/keyword gate on the path, registered in
`docs/SCAFFOLD-LEDGER.md`; the S15(c) trace cannot pass without an owner waiver.
