---
type: plan
status: live
date: 2026-05-24
---

# Direction K reviewer fix #3 — route through validated FHRR biologization stack

**Date:** 2026-05-24
**Status:** DESIGN (queued; cheapest scientifically-meaningful next test after Direction K BLOCK)
**Predecessor:** Direction K honest characterization (substrate NOT load-bearing at N_DIM=3200; pillar n=105 NOT VALIDATED per reviewer)
**Frozen bar:** 0.80 multi-seed STRICT TOP-1

## Goal

Re-run Direction K sequence storage routing through the VALIDATED FHRR
biologization pipeline (pillar n=87: resonate-and-fire neurons +
attractor clean-up with familiarity gate + mean-centered grounded
symbols) instead of plain `cosine_real()`. Determine whether the
biologized pipeline ALSO passes with random phasors (mechanism is
still dim-overkill), OR whether it specifically requires
substrate-grounded codes (the biologization adds load-bearing
constraints the simple algebra didn't).

## Why this test matters

Reviewer noted: "discipline concern Q10: bypass of validated
biologization. cosine_real() is plain numpy cosine. The runners DO
NOT invoke the validated FHRR biologization pipeline. Calling this
'substrate-grounded FHRR' overclaims: it is 'FHRR algebra over
substrate spike-count vectors with no FHRR biologization'."

If the biologized pipeline (with its 0.2 familiarity threshold and
attractor settle dynamics) STILL passes with random phasors at
N_DIM=3200, then even the biologized version is just dim-overkill
algebra. If biologized + substrate beats biologized + random, the
biologized pipeline's clean-up + familiarity gate creates a
substrate-specific load.

## Mechanism

Reuse byte-unchanged:
- `research/runners/resonate_fire_fhrr.py` (pillar n=87 biologization)
  - `ResonateFireFHRR` class (resonate-and-fire neurons)
  - `ResonateFireTPAM` class (attractor clean-up)
  - Separated familiarity gate (shortcut-3 RESOLVED)
- Direction K substrate activity capture (`capture_no_teacher_activity`)
- Direction K position phasor generation

Replace:
- `cosine_real()` → ResonateFireTPAM.identify() (settled attractor
  output) + familiarity gate

## Pre-registered test

Same as Direction K NO-TEACHER + smell test:
- Multi-seed 3 seeds [42, 43, 44]
- 8 sequences x SLOT_COUNT=3 per seed
- Pre-registered FROZEN bar 0.80 multi-seed STRICT TOP-1
- Same smell test (3 controls): permutation, random phasors, same-position
- Add: random-phasor-with-biologized-pipeline (the discriminating test)

## Expected outcomes

**(a) Biologized + substrate PASS AND random fails:**
The biologization (attractor + familiarity) adds load-bearing
substrate-specific constraints. Pillar n=105 VALIDATED with
"substrate-grounded biologized FHRR sequence storage."

**(b) Both biologized + substrate AND biologized + random PASS:**
Even biologization is dim-overkill at N=3200. Honest finding:
the load-bearing piece is FHRR algebra; biologization doesn't change
this. No pillar n=105.

**(c) Biologized + substrate FAILS, biologized + random FAILS:**
Familiarity gate threshold is too strict for either; biologization
isn't transferable to sequence storage. Honest BOUNDARY.

**(d) Biologized + substrate FAILS, biologized + random PASS:**
Substrate's overlap (0.20 mean) crosses the familiarity threshold;
the substrate is actually a HANDICAP for biologized pipeline. Honest
NEGATIVE (substrate too overlapping for biologized clean-up).

## Cost

- Coding: ~2 hr (adapt Direction K runner to invoke ResonateFireTPAM
  + familiarity gate)
- GPU/CPU run: ~30 min (same activity capture as Direction K
  no-teacher; biologized pipeline is post-capture)
- Smell test: ~10 min
- Adversarial review: ~10 min
- Total: ~3 hr

## Implementation order (subagent-driven-development)

1. Read `research/runners/resonate_fire_fhrr.py` — understand the
   validated biologized pipeline API
2. Write `direction_K_biologized_pipeline.py` — adapter that:
   - Loads vocab activities from Direction K NO-TEACHER cache
   - Wraps each vocab activity as a complex phasor (cos+i*sin
     phase encoding; or repurpose ResonateFireFHRR phasor formation)
   - Bind with position phasor; bundle; unbind
   - Pass unbound vector through ResonateFireTPAM attractor settle
   - Apply familiarity gate (separated; pillar n=87 RESOLVED
     pattern)
   - Score: top-1 = identified concept
3. Multi-seed run; same frozen bar
4. Smell test mirror (random vocab + biologized pipeline; same
   verdict logic)
5. Adversarial review

## Honest scope (pre-stated)

This is the BIOLOGIZATION SCRUTINY test. The result is honest
regardless of outcome:
- PASS substrate >> random: substrate-grounded biologized FHRR
  sequence storage VALIDATED (pillar n=105)
- PASS both: biologization doesn't add substrate-specific load at
  this dim (honest NEGATIVE on biology-uniqueness)
- FAIL both: biologization too strict for sequence task (honest
  BOUNDARY on biologization scope)
- FAIL substrate, PASS random: substrate too overlapping for
  biologized clean-up (honest NEGATIVE on substrate)

## Status

QUEUED. The autonomous chain reaches this design + a clear next
implementation order; actual implementation in next session OR via
watchdog continuation.

Reviewer's other recommendation (#3) is the load-bearing scientific
test; recommendation #1 (scope-tighten claim) DONE; recommendation
#2 (dim-scaling probe) DONE; recommendation #4 (UNTRAINED-bridge
exploit) DONE.
