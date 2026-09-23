---
type: finding
status: qualified
lane: load-bearing
date: 2026-09-23
mechanism: DA-gated synaptic tagging-and-capture (webapp/da_tag_capture.py, BRAIN_DA_TAG_CAPTURE default OFF) under a natural surprising-vs-expected conversational drive, read as 24 h recall; v2 at the production composer D=128
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO at runner level, 6 of 6 seeds (pre-registered v2 gates); NOT wired into /api/brain-chat, companion default OFF
artifacts:
  - research/findings/raw/_da_encoding_natural_drive_D128/aggregate.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed42.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed43.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed44.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed100.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed101.json
  - research/findings/raw/_da_encoding_natural_drive_D128/seed102.json
---

# DA-gated encoding under a natural drive, v2 (D=128): the brain's own dopamine decides what is still recalled the next day — 6/6 GO at runner level (2026-09-23)

Pre-registrations: `2026-09-23-da-encoding-natural-drive-24h-persistence-PREREGISTRATION.md` (gates) and
`2026-09-23-da-encoding-natural-drive-v2-production-D128-PREREGISTRATION.md` (D=128 only). v1 at D=64 is
UNDEFINED and stays so: `2026-09-23-da-encoding-natural-drive-v1-D64-UNDEFINED-immediate-recall-precondition.md`.
Artifacts: `research/findings/raw/_da_encoding_natural_drive_D128/seed<s>.json` and
`research/findings/raw/_da_encoding_natural_drive_D128/aggregate.json`. Terms follow `docs/TERMS.md`.

## Result

**All six seeds GO. The aggregate is GO.** Every precondition held (natural DA contrast, deterministic DA trace,
4/4 immediate recall in every arm at the primary point, the lesion reaching both the write gain and the PRP read).

<!--derived-->
| seed | salient DA min | neutral DA max | salient intact 24 h | lesion DA->encoding | neutral intact | lesion capture only | companion off (sal / neu) |
|---|---|---|---|---|---|---|---|
| 42 | 0.897 | 0.549 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 43 | 0.798 | 0.411 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 44 | 0.897 | 0.413 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 100 | 0.897 | 0.335 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 101 | 0.897 | 0.573 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 102 | 0.792 | 0.196 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |

- **The reply changes under the lesion.** A fact told as surprising news is answered the next day ("what did the goat eat?" returns "passport"). With the DA->encoding edge lesioned (`BRAIN_DA_ENCODING_LESION=1`, the lesion the battery already uses) the same question returns an abstention. Exact permutation p per seed = 1/70; the pooled effect (24 facts) sits above a 10 000-draw label-permutation null whose 95th percentile is 6.
- **It needs the surprise.** The same facts told plainly, after their words were already introduced, are forgotten by the next day in the intact brain. `attributable_to` gives 1.0 of the salient effect as absent from the neutral comparison, on every seed.
- **Capture is the load-bearing sub-edge.** Severing only DA->capture (`BRAIN_DA_CAPTURE_LESION=1`) forgets every salient fact even though the write gain stays high (2.6-2.8). The magnitude boost alone does nothing at 24 h, which matches the 2026-09-20 honest negative.
- **Production default reproduces the old null.** With the companion off, every fact is recalled at 24 h in both conditions, so today's chat cannot show the effect.
- **Encoding is intact.** At the primary point every arm recalled 4/4 one minute after the conversation, as Bethus, Tse & Morris 2010 found for D1/D5 blockade.
- **Robust across the pre-registered band** (baseline ratio 0.67 / 2.0, E-LTP decay 1.0 h / 3.0 h) wherever immediate recall held.

## Honest residuals (verify these before building on the result)

<!--derived-->
- **Runner level, not wired.** The capture ledger is driven by `research/runners/_da_encoding_natural_drive_persistence.py`. Nothing in `webapp/server.py` constructs it, and the battery has no world-clock turn. Under `docs/TERMS.md` this is "GO at runner level", not wired and not closed. `da-gated-encoding` stays hollow on the battery until the next rung lands.
- **The effect runs through a host model.** The decay, the capture decision and the threshold compare are host arithmetic on the store synapses (declared). The brain supplies the DA (spiking habituation novelty, spiking salience afferent, spiking SNc) and the recall (the on-substrate store read). The load-bearing link from DA to "kept by tomorrow" is the host rule, so the 0/4 lesion result follows from how the rule is written once G1 holds. The empirical content is G1 (the brain's own DA separates surprising from expected statements at its existing Go boundary on every seed), G2 (the store stays readable with a realistic baseline at D=128), and the absence of confabulation at the primary point.
- **Thin neutral margin on two seeds.** The neutral conversation's highest DA was 0.047 below the PRP threshold on seed 101 and 0.071 below on seed 42. A slightly more engaging neutral turn would count as a PRP event and capture the neutral facts too (the biology predicts exactly that, but it would flip G4).
- **The baseline ratio 2.0 still breaks low-gain writes.** At that band point the unit-gain lesion arm recalled 0-3 of 4 immediately, with 0-2 confabulations per seed. The pre-registered band rule excludes those points from G8, so the GO holds, but the moat weakness under a weaker-LTP assumption is real.
- **The DA write gain is itself the less faithful half.** In v1 at D=64 it made immediate recall depend on DA, which contradicts Bethus 2010. At D=128 the SNR hides this. A capture-only coupling would be more faithful.
- **Stimuli differ in wording.** Surprising turns carry more fresh content words than the plain ones, which is how the brain's novelty/salience read tells them apart. The intact-vs-lesion comparison is within the same stimuli, so it is unaffected; G4 (specificity) is not.
- **Forgetting neutral facts is not production-safe.** Without the other routes to late-phase LTP (strong or repeated stimulation, sleep replay), turning the companion on would make the chat forget ordinary facts by the next day.
- **No adversarial verify-go pass yet.** This lane agent had no subagent tool; the controller should run `verify-go` before the result moves any board status.
- The exploratory novelty-lesion arm was mixed (neutral facts kept on 3 seeds, lost on 3); it says nothing yet about where the drive originates.

## Next rungs (ordered)

1. Wire the ledger into the live chat store behind `BRAIN_DA_TAG_CAPTURE` (a store hook plus a world-clock advance on the idle tick), and add a battery probe pair with a "next day" gap so the battery reads it.
2. Add the other late-phase routes (repetition-triggered PRP, sleep replay) so neutral facts that matter are kept.
3. Move the capture decision onto the substrate (a bistable late-phase synaptic variable, Clopath et al. 2008).
4. Test a capture-only coupling (DA off the write magnitude) against Bethus 2010.
