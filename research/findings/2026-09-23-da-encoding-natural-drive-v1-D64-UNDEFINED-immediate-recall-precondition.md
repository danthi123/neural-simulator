---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
mechanism: DA-gated synaptic tagging-and-capture (webapp/da_tag_capture.py, BRAIN_DA_TAG_CAPTURE default OFF) under a natural surprising-vs-expected conversational drive, read as 24 h recall; v1 at composer D=64
seeds: [42, 43, 44, 100, 101, 102]
verdict: UNDEFINED (5 of 6 seeds fail the pre-registered G2 immediate-recall precondition; 1 of 6 GO)
artifacts:
  - research/findings/raw/_da_encoding_natural_drive/aggregate.json
  - research/findings/raw/_da_encoding_natural_drive/seed42.json
  - research/findings/raw/_da_encoding_natural_drive/seed43.json
  - research/findings/raw/_da_encoding_natural_drive/seed44.json
  - research/findings/raw/_da_encoding_natural_drive/seed100.json
  - research/findings/raw/_da_encoding_natural_drive/seed101.json
  - research/findings/raw/_da_encoding_natural_drive/seed102.json
---

# DA-gated encoding, natural drive, v1 (D=64): UNDEFINED on the immediate-recall precondition (2026-09-23)

Pre-registration: `2026-09-23-da-encoding-natural-drive-24h-persistence-PREREGISTRATION.md` (commit 44f32c5b1).
Graded exactly as registered. Terms follow `docs/TERMS.md`. Artifacts: per-seed
`research/findings/raw/_da_encoding_natural_drive/seed<s>.json` and
`research/findings/raw/_da_encoding_natural_drive/aggregate.json`.

## Verdict: UNDEFINED, not GO and not NO-GO

The aggregate status is UNDEFINED: 1 of 6 seeds is GO (seed 101) and 5 are UNDEFINED. Every UNDEFINED seed failed
the same precondition, G2: immediate recall must be 4/4 in every arm. The 24 h comparisons that G3-G8 read did not
fail anywhere. Because G2 is a precondition, those comparisons earn no verdict on the five seeds. They are reported
below as measurements, not as a result.

## What the 24 h reads measured (all six seeds, primary point)

<!--derived-->
| seed | salient DA min | neutral DA max | salient intact | salient lesion (DA->encoding) | neutral intact | salient lesion (capture only) | companion off (sal / neu) |
|---|---|---|---|---|---|---|---|
| 42 | 0.897 | 0.549 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 43 | 0.798 | 0.411 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 44 | 0.897 | 0.413 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 100 | 0.897 | 0.335 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 101 | 0.897 | 0.573 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |
| 102 | 0.792 | 0.196 | 4/4 | 0/4 | 0/4 | 0/4 | 4/4 / 4/4 |

- **G1, the natural drive, held on all six seeds.** The brain's own spiking DA put every surprising fact turn above 0.79 and kept every neutral-conversation turn below 0.58, on both sides of the existing Go boundary (0.62). No arousal was induced and no damage knob was used.
- The pooled intact-minus-lesion effect was 24 facts against a label-permutation null whose 95th percentile is 6 (`aggregate.json`).
- The production default (companion off) recalled everything in both conditions, so it reproduces the 2026-09-20 honest negative.
- The robustness band held at every point where immediate recall held.

## Why G2 failed: the baseline makes low-gain writes unreliable at D=64

The companion gives each store synapse a baseline strength as large as a unit increment (beta = 1). A 64-synapse
block then carries a unit-gain write only unreliably.

<!--derived-->
- The failures sit in the low-gain arms. The DA->encoding lesion pins the write gain to 1; it recalled 2 or 3 of 4 immediately on the five failing seeds. Neutral-conversation arms (gain about 1.3-1.5 from the spiking gain population) failed on some seeds. Salient intact arms (gain 2.4-3.0) never failed at the primary point.
- At beta = 2 immediate recall collapsed (0-1 of 4) on low-gain arms on every failing seed.
- The misreads include CONFABULATIONS: a wrong fact answered instead of an abstention (for example "cat" to "what did the goat eat" on seed 42). Across the five failing seeds there were 3 to 22 confabulations per seed across all arms and band points, almost all at the immediate read. That is a moat problem as well as a G2 failure.

## What this means for the biology

In Bethus, Tse & Morris 2010, D1/D5 blockade left immediate recall intact. In this model the DA->encoding lesion
lowers the write MAGNITUDE (gain 1 instead of 2.4-3.0). Once a realistic baseline exists, that lowers immediate
recall too. So the magnitude half of the existing coupling is the less faithful half. Biologically, dopamine acts
mainly on which early-phase changes are kept (capture), not on how large they are at first.

## What is NOT claimed

- Not a GO: 5 of 6 seeds are UNDEFINED under the pre-registered rule, and that stands.
- Not load-bearing in production: the companion is default-OFF and the battery does not exercise it.
- The capture decision, the decay and the threshold compare are host arithmetic on the store synapses (declared shortcuts).
- The "neutral facts forgotten by the next day" behaviour would be wrong in production without the other routes to late-phase LTP (strong or repeated stimulation, sleep replay). It must not be flipped on as is.

## Next (already staged, pre-registered before running)

- **v2 at the production composer size D=128** (`2026-09-23-da-encoding-natural-drive-v2-production-D128-PREREGISTRATION.md`). The v1 probe copied D=64 from the earlier probe; production builds D=128. Same gates, same beta = 1.
- **If v2 G2 also fails:** move DA's effect off write magnitude and onto capture only (write gain pinned at the recall-safe level in every arm, DA reaches only the PRP read), matching Bethus 2010. Pre-register it before running.
- The exploratory novelty-lesion arm was mixed (neutral facts kept on 3 seeds, lost on 3), so it says nothing yet about where the drive originates.
