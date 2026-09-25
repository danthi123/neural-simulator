---
type: finding
status: qualified
lane: load-bearing
date: 2026-09-23
mechanism: DA-gated synaptic tagging-and-capture (webapp/da_tag_capture.py, BRAIN_DA_TAG_CAPTURE default OFF) under a natural surprising-vs-expected conversational drive, read as 24 h recall; v2 at the production composer D=128
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO at runner level, 6 of 6 seeds (pre-registered v2 gates) — but the 24 h outcome is set by a DECLARED HOST RULE, so it is not credited to the brain (review 2026-09-23); NOT wired, companion default OFF; superseded by the v3 synaptic test
artifacts:
  - research/findings/raw/_da_encoding_natural_drive_D128/aggregate.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed42.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed43.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed44.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed100.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed101.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed102.json
---

# DA-gated encoding under a natural drive, v2 (D=128): the brain's spiking DA separates surprising from expected statements; a host tag-and-capture rule turns that into 24 h recall — 6/6 GO at runner level (2026-09-23)

> **Correction after adversarial review (2026-09-23).** The original title said "the brain's own dopamine decides what
> is still recalled the next day". That overclaims. The 24 h outcome is decided by host arithmetic in
> `webapp/da_tag_capture.py` (`TagCaptureLedger`): `observe_da` compares the DA scalar to 0.62, and `factor()` scales an
> uncaptured block by exp(-24/1.5) ≈ 1e-7, which reads as pure baseline. Once G1 and G2 hold, G3, G4 and G5 follow
> arithmetically. **What the brain did, and what was measured:** its spiking DA separated surprising from expected
> statements at the Go boundary on 6/6 seeds (G1); a D=128 store with a baseline stayed readable (G2); pure-baseline
> blocks abstained instead of confabulating. **What the host did:** a tag-and-capture RULE converted that separation
> into 24 h recall. The 24 h intact/lesion difference is not "load-bearing" brain behaviour and the gate formerly
> named `G3_load_bearing` is only the host rule's output. The test that puts the DA effect through synapses is v3:
> `2026-09-23-da-encoding-natural-drive-v3-synaptic-capture-PREREGISTRATION.md`.

Pre-registrations: `2026-09-23-da-encoding-natural-drive-24h-persistence-PREREGISTRATION.md` (gates) and
`2026-09-23-da-encoding-natural-drive-v2-production-D128-PREREGISTRATION.md` (D=128 only). v1 at D=64 is
UNDEFINED and stays so: `2026-09-23-da-encoding-natural-drive-v1-D64-UNDEFINED-immediate-recall-precondition.md`.
Artifacts: `research/findings/raw/_da_encoding_natural_drive_D128/seed<s>.json` and
`research/findings/raw/_da_encoding_natural_drive_D128/aggregate.json`. Terms follow `docs/TERMS.md`.

## Result

**All six seeds GO. The aggregate is GO** under the pre-registered v2 gates. Every precondition held (natural DA
contrast, deterministic DA trace, 4/4 immediate recall in every arm at the primary point, the lesion reaching both
the write gain and the PRP read). The GO describes a host rule's output given the brain's DA (see the correction).

<!--derived-->
| seed | salient DA min | neutral DA max | salient intact 24 h | lesion DA->encoding | neutral intact | lesion capture only | companion off (sal / neu) |
|---|---|---|---|---|---|---|---|
| 42 | 0.897 | 0.549 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 43 | 0.798 | 0.411 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 44 | 0.897 | 0.413 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 100 | 0.897 | 0.335 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 101 | 0.897 | 0.573 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 102 | 0.792 | 0.196 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |

- **The reply changes under the lesion, through the host rule.** A fact told as surprising news is answered the next day ("what did the goat eat?" returns "passport"). With the DA->encoding edge lesioned (`BRAIN_DA_ENCODING_LESION=1`) the same question returns an abstention. The per-seed p = 1/70 and the pooled fact-label null treat the 4 facts of one conversation as exchangeable; they are not (one session, one DA trace, one capture rule), so those p-values are **not valid inference** and are withdrawn. With the seed as the unit, a one-sided sign-flip test on 6/6 positive seeds gives p = 1/64.
- **It needs the surprise.** The same facts told plainly, after their words were already introduced, are forgotten by the next day in the intact brain. `attributable_to` gives 1.0 of the salient effect as absent from the neutral comparison, on every seed.
- **Capture is the sub-edge the host rule reads.** Severing only DA->capture (`BRAIN_DA_CAPTURE_LESION=1`) forgets every salient fact even though the write gain stays high (2.6-2.8). The magnitude boost alone does nothing at 24 h, which matches the 2026-09-20 honest negative.
- **Production default reproduces the old null.** With the companion off, every fact is recalled at 24 h in both conditions, so today's chat cannot show the effect.
- **Encoding is intact.** At the primary point every arm recalled 4/4 one minute after the conversation, as Bethus, Tse & Morris 2010 found for D1/D5 blockade.
- **Robustness band: one informative point held; the rest could not fail or were UNDEFINED.** The E-LTP decay points (1.0 h, 3.0 h) cannot fail at a 24 h read: every uncaptured trace is gone at any decay time in 1-3 h (at most exp(-8) ≈ 3e-4 of the increment is left). The baseline ratio 2.0 point failed the immediate-recall precondition on 6/6 seeds, with confabulations on seeds 43, 101 and 102; the pre-registered "where immediate recall holds" clause dropped it. Only the baseline ratio 0.67 point was informative, and it held. The GO stands under the pre-registered rule, but "robust across the band" was wrong.

## Honest residuals (verify these before building on the result)

<!--derived-->
- **Runner level, not wired.** The capture ledger is driven by `research/runners/_da_encoding_natural_drive_persistence.py`. Nothing in `webapp/server.py` constructs it, and the battery has no world-clock turn. Under `docs/TERMS.md` this is "GO at runner level", not wired and not closed. `da-gated-encoding` stays hollow on the battery until the next rung lands.
- **The effect runs through a host model.** The decay, the capture decision and the threshold compare are host arithmetic on the store synapses (declared). The brain supplies the DA (spiking habituation novelty, spiking salience afferent, spiking SNc) and the recall (the on-substrate store read). The load-bearing link from DA to "kept by tomorrow" is the host rule, so the 0/4 lesion result follows from how the rule is written once G1 holds. The empirical content is G1 (the brain's own DA separates surprising from expected statements at its existing Go boundary on every seed), G2 (the store stays readable with a realistic baseline at D=128), and the absence of confabulation at the primary point.
- **Thin neutral margin on two seeds.** The neutral conversation's highest DA was 0.047 below the PRP threshold on seed 101 and 0.071 below on seed 42. A slightly more engaging neutral turn would count as a PRP event and capture the neutral facts too (the biology predicts exactly that, but it would flip G4). <!--derived-->
- **The baseline ratio 2.0 still breaks low-gain writes.** At that band point the unit-gain lesion arm recalled 0-3 of 4 immediately, with 0-2 confabulations per seed. The pre-registered band rule excludes those points from G8, so the GO holds, but the moat weakness under a weaker-LTP assumption is real.
- **The DA write gain is itself the less faithful half.** In v1 at D=64 it made immediate recall depend on DA, which contradicts Bethus 2010. At D=128 the SNR hides this. A capture-only coupling would be more faithful.
- **Stimuli differ in wording.** Surprising turns carry more fresh content words than the plain ones, which is how the brain's novelty/salience read tells them apart. The intact-vs-lesion comparison is within the same stimuli, so it is unaffected; G4 (specificity) is not.
- **Forgetting neutral facts is not production-safe.** Without the other routes to late-phase LTP (strong or repeated stimulation, sleep replay), turning the companion on would make the chat forget ordinary facts by the next day.
- **Arm results depended on run order (write gain only).** On every seed the first arm run in the process got different spiking write gains from later arms fed the identical DA trace (seed 42: neutral/intact [1.54, 1.34, 1.49, 1.51] vs [1.71, 1.39, 1.46, 1.50] in later arms; seed 100: [1.0, 1.22, 1.0, 1.0] vs [1.0, 1.0, 1.02, 1.0]). Cause: the spiking gain reader is built lazily inside the first arm and later reads draw OU noise from a global RNG stream the snapshot does not restore; the runner also logged a separate peek read, not the gain the store used. The 24 h verdict does not change (the host rule ignores the gain once decay has run), but "each arm a fresh composer build at the same seed" was false for the gain. Pinned by `tests/test_da_encoding_arm_isolation.py::test_v2_runner_is_order_dependent`; fixed in the v3 runner (fresh process per arm, state reset, used gain logged).
- **No adversarial verify-go pass yet.** This lane agent had no subagent tool; the controller should run `verify-go` before the result moves any board status.
- The exploratory novelty-lesion arm was mixed (neutral facts kept on 3 seeds, lost on 3); it says nothing yet about where the drive originates.

## Next rungs (ordered)

1. Wire the ledger into the live chat store behind `BRAIN_DA_TAG_CAPTURE` (a store hook plus a world-clock advance on the idle tick), and add a battery probe pair with a "next day" gap so the battery reads it.
2. Add the other late-phase routes (repetition-triggered PRP, sleep replay) so neutral facts that matter are kept.
3. Move the capture decision onto synapses (a bistable late-phase synaptic variable, Clopath et al. 2008) — done as v3, see its prereg and finding.
4. Test a capture-only coupling (DA off the write magnitude) against Bethus 2010.
