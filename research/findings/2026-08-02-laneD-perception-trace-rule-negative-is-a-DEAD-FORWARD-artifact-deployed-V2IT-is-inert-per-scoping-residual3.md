---
type: finding
status: contributing
date: 2026-08-02
mechanism: perception-v2it-invariance
artifacts:
  - research/findings/raw/lanes/perception/v2it_fairdrive_s42.json
  - research/findings/raw/lanes/perception/v2it_moderate_s42.json
---

# lane D (perception V2/IT invariance): the first-attempt trace-rule 6-seed NEGATIVE is a DEAD-FORWARD-PASS artifact (VOID as a trace-rule verdict) — the deployed retina→V1→V2→IT STDP hierarchy has NO operating point that both propagates AND stays selective (inert below saturation, negative-RSA at saturation), confirming scoping residual #3's "deployed V2/IT is orphaned/inert"; the Földiák trace rule is NOT refuted (already 6-seed GO on the competitive pooler)

<!--derived-->
**One-line verdict.** lane D's first-attempt result (`research/findings/raw/lanes/perception_v2it_trace_rule.json`,
TRACE-NOGO 6/6, IT held-position decode 0.319 ≈ chance 0.333, `it_fires_all_seeds=False`, RSA-to-pixels 0.0) was scored
as a trace-rule NEGATIVE, but the read is NOISE: the forward pass is DEAD — retina fires 78-161 spikes yet
`cortex_v1_simple = 0` at EVERY seed (in propagation probes AND during training), so V1_complex/V2/IT are silent, IT never
fires, and the decode falls back to a degenerate depolarization code. The trace rule was never exercised (STDP had zero
postsynaptic activity to bind). So the negative is VOID as a mechanism verdict. This was caught research-first: the
mandated `before_you_build.sh` surfaced `2026-07-02-emerge50-trace-rule-GO` (the SAME Földiák trace rule is a confirmed
6-seed GO on the `OnSubstratePooler`) and `2026-07-23-perception-closure-scoping` residual #3 (the deployed
`cortex_v2→cortex_it` STDP is "orphaned + unproven, possibly inert"), which redirected the work from re-testing the trace
rule to checking the substrate. No `sim/` edit (additive drive/operating-point flags only).

## The operating-point sweep — dead below saturation, non-selective at saturation

<!--derived-->
| operating point | propagation retina→V1s→V1c→V2→IT | IT heldpos | frozen-IT | RSA-to-pixels (IT / V1c) | verdict |
|---|---|---|---|---|---|
| deployed (orig 6-seed) | 78-161 → **0** → 0 → 0 → 0 | 0.319 (≈chance) | 0.333 | 0.0 / — | dead (VOID) |
| fair-drive s42 (~20-25x) | 2202 → 493 → 2010 → 870 → **108** | 0.167 | **0.417** | **-0.07 / -0.14** | saturated / RETIRE |
| moderate s42 (~3-6x) | 552 → **6** → 0 → 0 → 0 | 0.333 | 0.333 | 0.0 / — | still dead |

<!--derived-->
Across a ~20x drive sweep the deployed hierarchy jumps DEAD → SATURATED with no selective operating point between.
Fair-drive revives firing but OVER-saturates: even the FIXED-Gabor V1_complex goes to NEGATIVE RSA-to-pixels (-0.14) — it
stops tracking the stimulus, so there is no invariant signal for IT to inherit, and trained IT (0.167) falls BELOW frozen
IT (0.417) with shuffled==trained. Artifacts `research/findings/raw/lanes/perception/v2it_fairdrive_s42.json` and
`research/findings/raw/lanes/perception/v2it_moderate_s42.json`. Seed 42 alone is decisive (the signals are STRUCTURAL — dead forward / negative RSA /
trained ≤ frozen — not the noisy 12-image decode), so 3-seed was not run.

## The missing companion process + the record-grounded next mechanism

<!--derived-->
This is the project's missing-companion-process signature: the ventral stream runs **divisive normalization** and
**homeostatic gain control** ALONGSIDE feedforward drive, and the deployed STDP hierarchy replaced them with a static
weight/drive constant that is either too weak (dead) or too strong (saturated, non-selective). The genuinely-next
mechanism, record + biology cited: (1) add the companion processes — feedforward **divisive normalization** (Carandini &
Heeger 2012, Nat Rev Neurosci 13:51) + **homeostatic synaptic scaling / intrinsic plasticity** (Turrigiano 2008, Cell
135:422) — to pin V1→V2→IT at a stable sparse-selective regime; OR (2) the cleaner path the record already validates —
route the Földiák trace rule through the **V1→competitive OnSubstratePooler** (EMERGE-50, whose k-WTA competition IS the
missing normalization) instead of the orphaned deployed STDP V2/IT, exactly as scoping residual #3 prescribes ("retire
the deployed V2/IT as unvalidated/inert; standardize grounding on the validated V1→pooler codon").

## Honest scope

<!--derived-->
This RETIRES the deployed retina→V1→V2→IT STDP hierarchy as a validated invariance substrate (no selective operating
point), and VOIDS the first-attempt trace-rule negative (dead forward pass) — it does NOT refute the Földiák trace rule
(already 6-seed GO on the competitive pooler) and does NOT itself claim invariant IT. Per THE LAW the verdict is on the
deployed SUBSTRATE/method, not the capability; the next mechanism is named and record-grounded. A worked demonstration of
the research-first discipline: the corpus check redirected the work from a doomed re-test to the correct diagnosis.
