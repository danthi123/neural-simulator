---
type: finding
status: undefined
date: 2026-08-27
mechanism: onebrain-integration-surprise-episodic-crossedge
lane: onebrain-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_6seed.json
runner: research/runners/_onebrain_integration_surprise_episodic_crossedge.py
builds_on:
  - research/findings/2026-08-27-onebrain-integration-R1-wm-to-comprehension.md
  - research/findings/2026-08-27-onebrain-r4-selfschema-provenance-production-GO.md
  - research/findings/2026-08-27-declarative-cross-edges-framework-GO.md
  - research/findings/2026-08-27-onebrain-completeness-audit.md
---

# One-brain INTEGRATION — surprise (D2 prediction-error) -> source_provenance encoding-gate: 6/6 F1/F3/F4/emergence/migration, F2 crux UNDEFINED (its own lesion control does not cleanly hold)

Artifact: `research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_6seed.json`.

**One-line:** finishing a WIP de-risk (committed at `research/onebrain-surprise-episodic-crossedge@d4fdeda03`) that
mirrors R1/R4's learned-cross-edge template on a THIRD pairing — the D2 `surprise` expectation-violation circuit
driving `source_provenance`'s `prov_generated` encoding-commitment pool (the audit-sanctioned substitute for the
still-Group-C-deferred `d5_episodic`, see scope note below). A single plastic cross-edge grows cleanly from 0.05 to
3.98-7.74 (6 seeds) by the substrate's own Hebbian rule, and the substrate's own F1/F3/F4/emergence/lesion-recovers-
migration arms are all clean 6/6. **The crux (F2, vary-then-lesion) does not clear its pre-registered floor on ANY
of the 6 seeds**, even after raising `hebbian_learning_rate` 0.05->0.15 as the prior WIP session's dose-response
check recommended — but per `tools/verdict.py`'s own precondition framework, that raw miss is **NOT a validated
negative**: the F2 lesion control itself (`delta_lesion` must fall under 34% of `delta_intact`) FAILS on 5 of the 6
seeds, meaning the manipulation does not cleanly isolate the cross-edge's contribution to begin with. **UNDEFINED,
not PARTIAL/NO-GO** — a run whose own precondition does not hold yields UNDEFINED, never a negative
(`tools/gates/verdict_preconditions.py`; the runner's `main()` initially computed its raw tag/verdict BEFORE
checking this precondition, the exact "affect eviction" bug class that gate exists to catch — found and fixed this
session, see below).

## What this session did (continuing WIP, not re-deriving it)

The runner (793 lines, `_onebrain_integration_surprise_episodic_crossedge.py`) was pulled into this branch verbatim
from `research/onebrain-surprise-episodic-crossedge@d4fdeda03` and committed unmodified first
(`958fa60b9`). Per the prior session's own state: seed-42 already passed F1/F3/F4/lesion-recovers-migration/
emergence; only F2 (the vary-then-lesion crux) was short of its pre-registered `F2_INTACT_FLOOR=0.010`, and a
dose-response check on seed 42 was reported to show the shortfall scaling with `hebbian_learning_rate` (i.e. an
undertrained-edge problem, not a ceiling) with `0.05 -> 0.15` identified as the fix.

This session applied exactly that one change — `hebbian_learning_rate` for the cross-edge's own training window
(`SurpriseEpisodicPool.__init__`, the standard non-rate-window Hebbian hyperparameters) raised from `0.05` to a new
named constant `CROSS_EDGE_LR = 0.15` — and nothing else (`N_EPISODES=150`, `HMAX=20.0`, `CUE_PA=2000.0`, all F-gate
floors unchanged from the committed WIP). Then ran the full pre-registered 6-seed set (42/43/44/100/101/102),
detached, and read the result: F2 missed its floor on all 6 seeds (see table below).

**A second, genuine bug was then found and fixed in the runner's own `main()`, not re-derived**: the WIP's
`preconditions` block (`tools.verdict.Verdict`, registering `f2_lesion_removes_shift` among others) was computed
and attached to the JSON payload, but the human-readable `tag`/`verdict`/`GO` fields were built from the raw
F1-F4/emergence/migration pass counts BEFORE that block was ever computed, and never revised afterward — so a
failing precondition never reached the asserted verdict text. This is the exact "affect eviction" bug pattern
`tools/verdict.py`'s own docstring names ("the runner COMPUTED `arm_valid=False` on 3/3 seeds and printed NO-GO
anyway"), and `tools/gates/verdict_preconditions.py` (which blocks a commit asserting a verdict beside a FAILED
precondition) caught it immediately on the first commit attempt. Fixed by reordering `main()` to compute
`Vd.decide()` first and read its `status`/`go` fields into the tag; the 6-seed run was then re-executed with the
fixed runner (a verdict-string/tag-only change — the underlying physiology/measurements are identical) to produce
the artifact this finding cites.

## Result — F2 misses its floor on all 6 seeds, AND its own lesion control fails on 5 of 6 — UNDEFINED, not a negative

<!--derived-->

Per-arm (raw pass tally, informational only — see the precondition failure below before reading this as a
verdict): **F1 6/6 - F2 0/6 - F3 6/6 - F4 6/6 - emergence 6/6 - lesion-recovers-migration 6/6.**

| seed | block(c,c') | final w (from 0.05) | F2 delta_intact | F2 delta_lesion | frac_attrib | floor gap (vs 0.010) | F2 PASS |
|---|---|---|---|---|---|---|---|
| 42  | (1,5) | 3.978 | +0.00434 | +0.00305 | 0.297 | -0.00566 | False |
| 43  | (6,7) | 6.112 | +0.00672 | +0.00297 | 0.558 | -0.00328 | False |
| 44  | (0,4) | 7.178 | +0.00871 | +0.00305 | 0.650 | -0.00129 | False |
| 100 | (3,0) | 5.690 | +0.00367 | +0.00238 | 0.351 | -0.00633 | False |
| 101 | (4,0) | 6.756 | +0.00645 | +0.00176 | 0.727 | -0.00355 | False |
| 102 | (0,4) | 7.736 | +0.00848 | +0.00352 | 0.585 | -0.00152 | False |

`F2_INTACT_FLOOR=0.010`, `F2_LESION_RATIO=0.34` (unchanged from the WIP's pre-registered calibration). The closest
seed (44) reaches 87% of the floor; the furthest (100) reaches 37% — but the floor miss is the SECONDARY problem.
The PRIMARY one is the lesion-ratio column: `F2_LESION_RATIO` requires `delta_lesion < 0.34 * delta_intact` (the
lesion must remove MOST of the shift), and this holds on only 1 of 6 seeds (101, at 0.273). On the other 5, the
lesioned (cross-edge-zeroed) condition still shows 35%-73% of the intact shift — `tools.lab.attributable_to` flags
every seed with `⛔ MOST OF THIS EFFECT IS IN THE CONTROL`. This is the `f2_lesion_removes_shift` precondition
(`tools/verdict.py`) that FAILS as a whole (it requires ALL 6 seeds to hold), which is what makes this crux
UNDEFINED rather than a validated negative: when the control itself does not cleanly isolate the manipulation, the
intact-condition number cannot be read as evidence that the cross-edge's effect is merely "too small" — a
comparable-or-larger fraction of it is present with NO cross-edge at all, so we do not actually know how much of
`delta_intact` is the edge versus the same fixed residual visible in `delta_lesion`.

F1/F3/F4/emergence/migration are unaffected by the LR change and remain exactly as strong as the WIP's seed-42
indicator, now confirmed on all 6 seeds: battery accuracy 1.000 with `min_d` 0.613-0.746 (floor 0.50); CONFIRM-vs-
CONTRADICT separation 3333x on every seed; the trained block grows specifically (non-participating blocks stay at
0.050, drift < `OTHER_BLOCK_DRIFT_MAX=0.03`); `frozen_weight_maxdrift < 1e-6` on every seed; base connectivity is
structurally identical to the plain no-cross-edge pool with `sp_battery_maxerr`/`surprise_reads_maxerr` both under
the 0.05 tolerance on every seed.

## Is this still a scaling issue, or a real floor? (the question the task asked to answer honestly)

<!--derived-->

Two distinct questions are tangled here, and it is worth separating them explicitly. (1) Does `delta_intact` clear
`F2_INTACT_FLOOR`? No, on any seed. (2) Does the lesion control validate that `delta_intact` is actually
attributable to the cross-edge in the first place? Also no, on 5 of 6 seeds — which means question (1) is not even
cleanly answerable yet: a floor comparison on a number whose own control says "much of this isn't the mechanism
under test" is not evidence about the mechanism. The analysis below addresses the scaling-vs-ceiling question
raised by the task, but it should be read as characterizing WHY the raw numbers behave the way they do, not as
rescuing question (1) into a validated negative — the honest bottom line remains UNDEFINED on both counts.

The evidence points away from "simply undertrained, keep pushing the same lever" and toward the READOUT
saturating faster than the weight does — not conclusively a hard ceiling, but the naive one-lever fix used here is
refuted:

- **Weight growth was real and substantial** (3.98-7.74, i.e. 80x-155x the 0.05 seed value) and NOT yet bound-limited
  (`HMAX=20.0`; the furthest seed reached 7.736, well under the bound; F3's `decelerating` check — window-over-
  window growth slowing — still passed on every seed, meaning growth had not fully saturated by end of training
  either).
- **But delta_intact did not scale anywhere close to proportionally with weight.** Comparing the lowest-weight seed
  (42, w=3.978, delta=0.00434) to the highest (102, w=7.736, i.e. 1.94x the weight), delta only rose to 0.00848
  (1.95x) — that one pair looks roughly linear, but seed 100 (w=5.690, mid-range) has the LOWEST delta of all six
  (0.00367), and seed 43 (w=6.112) sits well below seed 44 (w=7.178) despite a similar weight. There is no clean
  monotonic weight-to-delta relationship across seeds — the per-seed random block assignment (which two of 8
  trained concepts play cue/false-assertion) is confounding the read as much as or more than the trained weight is.
- **Direct comparison to the prior (uncommitted) session's own seed-42 reference**: this session's `CROSS_EDGE_LR`
  change plus the already-committed `N_EPISODES=150` bump (up from the WIP's own earlier 60-episode calibration,
  which the runner's own docstring records as producing `delta ~0.004-0.005` on seed 42 at the OLD, lower dose) —
  a combined ~2.5x-3x increase in total training exposure — still lands seed 42 at `delta_intact=0.00434`,
  statistically indistinguishable from that OLD, much-less-trained reference point. Tripling the learning rate
  while already having 2.5x the episode budget produced essentially NO improvement on the seed the fix was
  calibrated against.
- **The fixed non-cross-edge residual is comparable in magnitude to the genuine effect.** `delta_lesion` clusters
  tightly at 0.0018-0.0035 across all 6 seeds regardless of the (quite different) trained weights — consistent
  with the runner's own docstring describing a "layout-mediated coupling" residual from the multi-step spiking read
  (`N_READS` was already raised to 8 to fight exactly this). The genuine edge-attributable component
  (`delta_intact - delta_lesion`) is only 0.0007-0.0057 across seeds — never approaching a magnitude that would
  clear 0.010 on its own even added to the full residual.

**Read honestly, this is closer to a floor of THIS SPECIFIC READOUT CONSTRUCTION than to a straightforward
undertraining gap**: the rate-based margin (`rate_generated - rate_perceived`, averaged over `N_READS=8` runs of
`RECALL_STEPS=100` steps) appears to saturate in its sensitivity to this cross-edge's magnitude well before the
weight itself saturates. This is a NAMED, proven phenomenon in spiking-network theory, not a novel mystery this
arc discovered: **Sanzeni, Histed & Brunel (2020, PLOS Comput Biol, "Response nonlinearities in networks of
spiking neurons")** show that a network's rate-response transfer function becomes SUBLINEAR and saturates at
higher input/drive levels because of the refractory period (their eq. 13), and that the width of this nonlinear
near-saturation region scales with coupling strength/connectivity — i.e. a rate-based readout compresses as the
driving synaptic weight grows, exactly the pattern seen here (weight up to 155x the seed value, `delta_intact`
essentially flat). This does not resolve whether pushing further (toward `HMAX=20`) would eventually clear the
floor before the readout fully saturates, but it reframes the open question: not "why doesn't this scale
linearly" (rate-based spiking readouts generically do not), but "is there a training-budget regime where the
edge-attributable signal clears the floor before saturation dominates it" — untested here (Sanzeni, Histed & Brunel 2020, DOI
10.1371/journal.pcbi.1008165). Whether a genuinely
different lever (many more episodes at this same rate to push the weight toward the `HMAX=20` bound; a larger
`CUE_PA`/`CTX_DRIVE_PA` to strengthen the coincidence signal itself rather than only its learning rate; or a
change to the read protocol — e.g. reading `prov_generated`'s membrane current/conductance rather than its
thresholded spike rate, to stay in the pre-saturation linear regime — to reduce the fixed residual) would clear
the floor is UNTESTED here and is a genuine open question, not something this bounded task's single-lever fix
resolved. Per the task's own instruction, the floor is NOT being lowered to force a GO.

## Adversarial verification (verify-go skeptic, inline/blocking, not a silent Monitor)

<!--derived-->

Two independent single-seed reruns, in fresh processes, reproduced the 6-seed run's numbers exactly:

- **Seed 42, twice more** (once via `--smoke`, which hardcodes seed 42 regardless of `--seeds`, and once as part of
  the original pre-6-seed indicator run): `delta_intact=+0.00434`, `delta_lesion=+0.00305`, `frac_attributable=
  0.297` — byte-identical across all three independent invocations (the original smoke check, the 6-seed run, and
  this skeptic rerun).
- **Seed 44 alone** (`--seeds 44`, the closest-to-floor seed, chosen deliberately as the hardest case to dismiss):
  `delta_intact=+0.00871`, `delta_lesion=+0.00305`, `frac_attributable=0.650` — byte-identical to its line in the
  6-seed run.

This confirms the run is deterministic (no race condition or seed-leakage artifact inflating or deflating any
single seed's numbers) and that both the floor miss AND the lesion-control shortfall are real and reproducible,
not a fluke of one process's execution. The `lesion` claim itself also verifies per `docs/TERMS.md`'s condition
(the manipulation — zeroing the cross-edge's weight — is applied and read back within the same process before any
further plasticity step can regrow it; `apply_cross_edge_freeze()`'s whitelist keeps every other synapse's
`cp_plasticity_rate_gain=0` throughout, so nothing else could silently repair the lesion between the zero-and-read)
— the lesion mechanism itself is sound; it is the MAGNITUDE of what it removes, relative to the intact delta, that
falls short of the pre-registered ratio on most seeds.

## F1 - the faculty still works (edge present, no F2 hold)

<!--derived-->

`source_provenance`'s OWN 8-item battery keeps perfect sign accuracy (1.000) with `min_d` 0.613-0.746, clear of its
own `D_FLOOR`. `surprise`'s own CONFIRM-vs-CONTRADICT discrimination on this seed's randomly-assigned block pair
stays enormous (3333x separation) on every seed. Neither organ's own faculty is perturbed by the cross-edge.

## F3 - no runaway

<!--derived-->

`prov_generated`/`prov_perceived` rates stay in the physiological band on every seed; the trained-block weight is
bounded (`<=HMAX=20.0`, never reached) and growth decelerates window-over-window; the pool stays alive throughout.

## F4 - the moat / honesty holds

<!--derived-->

(a) The surprise-inducing hold with NO content drive at all stays sub-decision (well under `F4A_FRAC=0.5` of a
genuine decision's magnitude) on every seed — no confabulated provenance from bias alone. (b) A clear,
already-correctly-encoded battery item is NOT flipped by a co-occurring wrong-context surprise hold, retaining
`>=F4B_RETAIN=0.5` of its no-hold margin on every seed.

## emergence - the edge LEARNED, not hand-set

<!--derived-->

The cross-edge grows from 0.05 to 3.98-7.74 by the substrate's own standard same-step Hebbian rule on every seed,
specifically on the randomly per-seed assigned trained block (anti-cheat): the other 11 concept blocks' edges into
`prov_generated` stay within `OTHER_BLOCK_DRIFT_MAX=0.03` of the 0.05 seed value on every seed.
`frozen_weight_maxdrift < 1e-6` on every seed (the whitelist held; no migrated or already-trained weight moved).

## lesion-recovers-migration - integration added ONLY the declared edge

<!--derived-->

With the cross-edge lesioned, base connectivity is structurally identical to the plain no-cross-edge merged pool on
every seed, and both organs' own reads match within the pre-registered 0.05 tolerance (`sp_battery_maxerr`,
`surprise_reads_maxerr`) on every seed.

## Scope substitution (unchanged from the WIP, restated for completeness)

The literal ask was surprise -> the D5 episodic organ; `d5_episodic` remains `GROUP_A_DEFERRED` in
`onebrain_merge_framework.py` (a ~2000-neuron CA3 with two-compartment apical dendritic-dAP + slow-NMDA
reverberation + BTSP formation — a single BTSP store measured ~510s on numpy@2000 neurons, 2026-08-12 finding),
gated by the completeness audit's own roadmap on migrating `d5_episodic` first (step 6), a separate heavy lane out
of this de-risk's CPU/numpy budget. `source_provenance`'s `prov_generated` pool is the audit's own named
alternative target for this exact rung ("surprise -> episodic/PROVENANCE ENCODING gate"), already `GROUP_A`-migrated
and already validated as a cross-edge target by R4 — this substitution is unchanged from the committed WIP and is
NOT what caused the F2 shortfall (F1/F3/F4/emergence/migration are all fine on this same pairing).

## Honest scope / residuals (declared)

- **The crux (F2) is UNDEFINED, not a validated negative.** Raising `hebbian_learning_rate` alone, at the WIP's
  already-bumped `N_EPISODES=150`, does not close the floor gap on any of 6 seeds, AND the F2 lesion control does
  not cleanly isolate the cross-edge's contribution on 5 of 6 seeds — so this session cannot honestly claim either
  "the mechanism doesn't reach the floor" (that presumes the intact delta IS the mechanism, which the control does
  not support) or "the mechanism works but is unmeasurable" (the floor miss is real regardless). Do not re-attempt
  the `hebbian_learning_rate` lever alone without also addressing why the lesion control's residual is so large
  relative to the intact effect (see analysis above) — a fix that only grows the trained weight further will not
  by itself fix a control that already fails on the CURRENT (fairly large) weights.
- **A real runner bug was found and fixed this session** (see "What this session did"): `main()`'s tag/verdict
  computation ignored its own `preconditions` block. Any FUTURE re-run of this runner will correctly surface
  `UNDEFINED` when `f2_lesion_removes_shift` (or any other registered precondition) fails, rather than silently
  asserting a tallied GO/NO-GO beside it.
- Two-factor Hebbian (no reward/dopamine gating); a host-chosen cross-edge topology (surprise -> prov_generated
  only); a host-curated training schedule (co-driving a CONTRADICT trial + `ctx_generated` directly, not via an
  organic dialogue turn) — the same class of scaffold-residual R1/R4 declared for their own schedules.
- `prov_generated` firing is an ENCODING-COMMITMENT PROXY, not a literal CA3 autobiographical memory trace — the
  full `d5_episodic` pairing rides the Group-C migration (step 6), a named follow-on.
- **Not a production flip**: this remains a standalone research runner, additive, no `sim/` edit, no production
  wiring, no default flip — and given F2's UNDEFINED status, none is warranted yet regardless.

## Files

- `research/runners/_onebrain_integration_surprise_episodic_crossedge.py` — the runner (F1-F4 gate + emergence +
  lesion-recovers-migration; 6-seed; numpy CPU; NO `sim/` edit). Pulled in from
  `research/onebrain-surprise-episodic-crossedge@d4fdeda03` and committed unmodified (`958fa60b9`), then this
  session's changes: (1) `hebbian_learning_rate` for the cross-edge's own training window, `0.05 -> 0.15`
  (`CROSS_EDGE_LR`); (2) `main()` reordered so its `tag`/`verdict`/`GO` fields are derived FROM
  `Vd.decide()`'s `status`/`go`, not computed independently of it (the affect-eviction-class bug this session
  found and fixed).
- `research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_6seed.json` — the 6-seed artifact
  (`GO: false`, `verdict` beginning `"UNDEFINED -- ..."`, `preconditions[0].name="f2_lesion_removes_shift"
  ok=false`), auto-provenance-stamped.

Functional read-outs only; no phenomenal-experience claim.
