---
type: finding
status: contributing
date: 2026-08-11
mechanism: heterosynaptic-LTD competitive ENCODING (one-shot target-source commitment — foreign cross-talk synapses depressed at the encoding step) for source monitoring
lane: laneC / source-monitoring
seeds: [244, 259, 700, 701, 702, 703]
verdict: mechanism SOLVES the weak-encoding crux (244/259 FAIL->PASS, structural no-harm); full-6 aggregate 5/6 gated only by seed 702's NO-LEARNING control failure (instrument-invalid on that seed), not a mechanism failure
runner: research/runners/_laneC_source_monitor_competitive_encoding_gate.py
artifacts:
  - research/findings/raw/parallel_gates/source_monitor_competitive_encoding_decisive_244-259-700-703.json
instrument: the source-monitor gate (independent population-coded pools + up-only homeostatic scaling) PLUS a one-shot heterosynaptic-LTD commitment at encoding — for each source pool, foreign synapses whose presynaptic episode neuron was active during ANOTHER source's encoding window are depressed (anti-Hebbian pre-active-elsewhere / post-silent -> LTD), orthogonalizing the source codes so a rival cannot spuriously co-fire at another source's recall. Default OFF (byte-identical to the popcode+homeo baseline). SIM_BACKEND=numpy.
---

# Source monitoring — heterosynaptic-LTD competitive ENCODING fixes the weak-encoding crux that recall-time gain could not (244/259 FAIL->PASS); the full-6 aggregate is 5/6, gated by a per-seed control failure not the mechanism

The population-coding source-monitor de-risk
(`2026-08-11-source-monitor-no-harm-boundary-is-an-operating-point-population-coding-derisk.md`) resolved the no-harm
boundary STRUCTURALLY but was a 5/6 NO-GO because recall-time gain is not a universal floor-clearer: a weak-encoding
class (seeds like 244, 259) stayed below the 0.15 source-margin floor. Both that arc and the overlap-side fair-inhibition
NO-GO named the same next mechanism: **competitive ENCODING** (heterosynaptic LTD at the learning step). This de-risk
builds + tests it.

## The diagnostic (why recall-time gain failed) — the weak source is NOT weakly encoded

<!--derived-->
The weak source (`heard`) fires at a HEALTHY rate (~0.19, the same as the strong sources). The margin is small because
a RIVAL pool (`seen`) spuriously CO-FIRES at heard's recall — a cross-talk / orthogonalization deficit, not a weak own
code. Recall-time gain over-drove that rival (why both prior sides NO-GO'd). Heterosynaptic LTD at encoding removes the
cross-talk directly, at its source.

## Result — the weak-encoding crux is SOLVED; full-6 is 5/6 (`research/findings/raw/parallel_gates/source_monitor_competitive_encoding_decisive_244-259-700-703.json`)

<!--derived-->
- **WEAK-ENCODING CRUX seeds (the class recall-time gain could not fix): 2/2 FAIL -> PASS.** Seed 244: weakest margin
  +0.1396 (baseline FAIL) -> +0.1988 (CE PASS). Seed 259: +0.0717 (FAIL) -> +0.1892 (PASS). Both clear the 0.15 floor.
- **No-harm is STRUCTURAL:** only the weak source's margin moved (no_harm_min_gain +0.0000 every seed — no strong source
  is reduced; CE only ever depresses FOREIGN cross-talk synapses, so every margin can only rise).
- **Fresh no-harm guards 700, 701, 703: PASS** (CE correctly INERT where there is no cross-talk — foreign_l1 0->0 — and
  no-harm holds trivially).
- **Seed 702: DECISIVE_FRESH_FAIL — but on the `learning_off_has_no_source_recall` control, NOT the mechanism.** On 702
  the no-learning baseline itself recalls a source (the seed's own instrument-validity precondition fails), which
  confounds the measurement there; CE is inert on it (margin +0.1975 unchanged). Per the verdict-preconditions
  discipline a failed precondition is UNDEFINED, not a mechanism negative — so honestly the mechanism is validated on
  EVERY seed where the instrument is valid, plus it solves the 2 weak crux seeds. The runner's strict all-6 aggregate
  nonetheless reads **5/6 -> NO-GO** (it counts 702 as a fail).

Anti-cheats (executed via `tools.lab`, 19/19 True on the weak seeds): load-bearing (lesion restores baseline EXACTLY;
floor FAILS with CE off, PASSES on), attribution, no-harm structural, lever-moved (foreign L1 depressed on the weak
seeds; correctly UNCHANGED on clean seeds — see the runner fix below).

## Scope / honesty + next step

<!--derived-->
NO-EXTERNAL-NEEDED: the 5/6 is a single per-seed CONTROL failure (an instrument-validity precondition), not a
fundamental-limit claim; the mechanism (anti-Hebbian heterosynaptic LTD) is textbook biology and the capability is
ADVANCED here, not walled.

- **Headline:** competitive encoding at the learning step FIXES the weak-encoding class that recall-time gain (from both
  the overlap and disjoint sides) could not — the residual the source-monitoring NO-GO named is resolved on its target.
- **Runner fix banked with this finding:** the lever-moved anti-cheat now requires movement ONLY where there is foreign
  cross-talk to depress (`foreign_l1_before > 0`); on a clean seed CE is CORRECTLY a no-op, which is the expected
  no-harm case, not a VOID test (the first 6-seed run crashed on this edge case — the instrument is part of the
  emulation).
- **Cross-lane convergence:** this same heterosynaptic-LTD-at-encoding biology is the NAMED next mechanism for the
  emergence-engine allocation wall (`2026-08-11-emergence-engine-selective-write-store-...`) — keeping allocation/source
  codes disjoint under pressure is one mechanism serving two lanes.
- **Named residual scaffold:** the consolidation is host-computed/host-timed and reads the recorded per-source encoding
  activity (the source-afferent-identity + learning-window scaffold already declared for this arc); the label-free,
  spiking, ONLINE implementation (derive the foreign-synapse depression from the substrate's own firing during
  encoding) is the burn-down. NO `sim/` edit.
