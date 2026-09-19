---
type: finding
status: live
lane: load-bearing
date: 2026-09-19
---

# Load-bearing fraction — BASELINE first reading: 16/26 = 0.615 <!--derived--> (2026-09-19)

**The new #1 metric of the 2026-09-19 arc has its first number.** Since the owner ratified Qwen as the
permanent articulation mouth, the load-bearing question replaced "% scaffold retired": across the brain's
conversational faculties, in how many does *lesioning the brain's contribution provably change the reply*?
A faculty is **load-bearing** iff, on a fixed probe conversation, the decision fields differ intact-vs-lesion
AND that change is absent from the null control (intact-vs-intact-rebuild) AND it reproduces on a rebuild.

## Result

**load_bearing_fraction = 16 / 26 exercised = 0.6153846153846154** (≈0.615 <!--derived-->).

| bucket | n |
|---|---|
| faculties in the map | 38 |
| env-lesion coverable | 28 |
| **exercised** (coverable − not-exercised) | **26** |
| **load-bearing** (brain drives the reply) | **16** |
| not load-bearing (LLM would say the same) | 10 |
| not-exercised | 2 |
| noisy | 0 |
| lesion-knob-missing | 0 |
| not env-lesionable (in-process 4 · thin 2 · mechanism-only 1 · proposed 3) | 10 |

**Load-bearing (16):** comprehension-monitor · comprehension-learned-animacy-cue ·
comprehension-learned-verb-selects · affect-drives-response · affect-marker-spiking-wta ·
da-mode-drives-response · source-provenance-honesty · confidence-forthcomingness · pragmatic-implicature ·
metacog-monitor · worldmodel-forward · curiosity-followup · reconsolidation · gnw-multistep-deliberation ·
self-initiated-utterance · vision-identity-spiking-hmax.

**NOT load-bearing (10) — the actionable "hollow" set** (present/wired but the reply is unchanged when the
brain's contribution is lesioned, so the Qwen mouth carries them): noncontradiction-gate · affect-coloring ·
da-gated-encoding · common-ground-drives · prospective-memory · surprise-monitor · episodic-memory ·
discourse-register · open-ended-generation · bg-action-selection.

## Trust

`deterministic = True`; 28/28 null controls checked, **0 dirty**. Every treatment change is therefore cleanly
attributable to the lesion, not run-to-run noise — the standard failure mode of this kind of measurement is
absent here. Each lesion verdict also reproduced under `--repeats 2`. This makes the point estimate solid,
with the caveat below.

## Caveats (honest)

- **Single seed (42).** repeats=2 is the lesion-reproduce anti-noise check, NOT a multi-seed spread. A 6-seed
  pass (42/43/44/100/101/102) is the next step to put an error bar on 0.615 <!--derived-->. The clean null controls make a
  large seed-swing unlikely, but it is not yet measured.
- **Coverage gap, not a score.** 10 faculties have no clean env-lesion knob yet (in-process/thin/
  mechanism-only/proposed). They are excluded from the denominator and named individually — NOT counted as 0
  and NOT faked. The minimal lesion knob to add for each is in `FACULTY_LESIONS`' per-row `note`.
- **A lesion diff is a functional read-out.** "Load-bearing" means the brain's signal changes the produced
  decision on this probe; it asserts nothing about phenomenal experience.

## What it tells the arc

~⅔ of the measurable faculties provably drive the reply — the brain is genuinely doing work under the LLM
mouth, not decorative. The 10-faculty hollow set is the map of where it is NOT yet driving, and the most
striking are **episodic-memory** and **prospective-memory** (memory should matter to a reply and currently
does not move it) plus **surprise-monitor**, **discourse-register**, and **open-ended-generation**. These
are the targets for "make the faculty load-bearing" work, in priority order to be set with the ledger
re-instrumentation.

## Provenance

- Artifact: `research/findings/raw/_load_bearing/load_bearing_baseline.json` (+ 38 per-faculty raw arms in
  that dir).
- Runner: `research.runners.load_bearing_fraction --repeats 2` (CPU/numpy, stub renderer, no LLM; each arm a
  full-brain build at seed 42). Instrument selftest PASS; lesion-map coverage verified.
- The run was paused mid-flight for the owner's gaming (RAM), then resumed via the new opt-in
  `LB_RESUME_SKIP_EXISTING` (commit 391a055d) — 26 completed faculties loaded off disk (0 rebuilds), only the
  tail rebuilt, byte-identical to an uninterrupted run.

## Next

1. Re-instrument `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` to carry each faculty's load-bearing verdict (the
   new #1 metric surfaced per-faculty), replacing "% scaffold_retired" as the headline.
2. 6-seed the fraction for an error bar.
3. Attack the hollow set, memory first (episodic + prospective).
