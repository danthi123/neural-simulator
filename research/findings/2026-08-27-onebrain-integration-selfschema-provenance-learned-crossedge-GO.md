---
type: finding
status: live
date: 2026-08-27
mechanism: onebrain-integration-r4-selfschema-provenance-learned-crossedge
lane: onebrain-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed.json
runner: research/runners/_onebrain_integration_r4_selfschema_provenance.py
builds_on:
  - research/findings/2026-08-27-onebrain-integration-R1-wm-to-comprehension.md
  - research/findings/2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md
---

# One-brain INTEGRATION R4 — a SECOND learned cross-region edge: self_schema authorship -> source_provenance
monitoring ("is this my own thought" self-monitoring, 6/6 GO)

Artifact: `research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed.json`.

**One-line:** reusing the R1 learned-cross-edge template (whitelist-inverted plastic synapse + F1-F4 functional
gate + lesion-recovers-migration) on a NEW organ pairing — `self_schema`'s authorship ("did I author this
thought") sub-block driving `source_provenance`'s "generated" (internally-sourced) memory-monitoring pool — a
single plastic edge grows from 0.05 to 2.9-3.6 (6 seeds) by the substrate's own standard Hebbian rule, and
holding self_schema's author state shifts a genuinely-ambiguous provenance item's read toward GENERATED
(attributable_to 0.96-1.01), vanishing on lesion. **GO 6/6.**

## Pairing choice: self_schema x source_provenance, not affect x mouth/tone

The task offered two candidate pairings. `self_schema` and `source_provenance` are BOTH already registered,
fully-migrated `GROUP_A` descriptors in `research/runners/onebrain_merge_framework.py` (`organ_cls` + `idx_fn` +
`read_fn` + `answer_fn`, `supports_shared=True`) — the identical two-organ merge shape R1 already proved for
`[d6_multiref_wm, comprehension]`. `affective_tom` is instead listed in that file's `GROUP_A_DEFERRED` table,
needing a new OU-noise + neuromodulator-subsystem reconciliation seam that does not exist yet; no "mouth/tone"
organ is registered in the framework at all. Building on self_schema/source_provenance reuses the merge
machinery verbatim; the affect/mouth pairing would have required inventing new wiring this arc was told to avoid.

The natural coupling is direct: self_schema's `author` sub-block (DR-3, `self_schema_production_organ.py`) is
the substrate's own "did I author this" tag — a tonic drive when a thought is self-generated, silent when it is
a recalled/heard fact. source_provenance's `prov_generated` pool (board #129) is the substrate's own "this
memory reads as internally-generated" pool. Self-referential processing biasing later source attributions is a
named effect in the source-monitoring-framework literature (Johnson-Hashtroudi-Lindsay 1993); the cross-edge
implements exactly that coupling on the shared substrate.

## The mechanism (emergence-compliant; NO sim/ edit)

ONE shared spiking bridge holds both organs' regions via `merge_organs([self_schema, source_provenance],
wire=True)` (both organs are ORGAN-READ CLOSED on this pool — self_schema's assembly loops + member->attend, and
source_provenance's base pathways, both reinject per-region-seamed, byte-identical to their own standalone
builds). A SINGLE plastic cross-edge `author -> prov_generated` is injected at w0=0.05 (near-zero) as the SOLE
plastic synapse — R1's whitelist inversion (`cp_plasticity_rate_gain=0` everywhere, then
`set_plasticity_gate("author_to_provgen", 1)`), every migrated edge byte-frozen. It GROWS by the substrate's OWN
STANDARD same-step (symmetric) Hebbian rule over episodes that co-drive the author pool (`AUTHOR_PA=650.0`,
self_schema's own production constant) with source_provenance's `ctx_generated` line (`CTX_DRIVE_PA=2500.0`,
source_provenance's own de-risk constant) — `ctx_generated -> prov_generated` is a FIXED (non-plastic) strong
pathway, so `prov_generated` reliably co-fires with `author`, Hebbian-binding the cross-edge without ever
touching source_provenance's own learned `episode -> prov_*` traces.

**One-sided by design, declared not smoothed over.** self_schema's authorship axis is a genuine BINARY TAG — one
population that fires for "self", stays silent for "heard" — unlike d6's two independently-drivable slot pools
(`w0`/`w1`) that let R1 test two opposite directions. There is no second population to wire a symmetric
opposite-direction edge from, so the cross-edge biases ONLY toward GENERATED when held. F2 tests this one real
direction (a held "self" state vs a no-hold baseline) rather than R1's two-direction test — the honest shape of
the underlying signal, not a forced symmetry.

## A mechanism-interaction bug found and fixed before the 6-seed run (not a floor-tuning game)

<!--derived-->

Two genuine wiring/calibration issues surfaced in the seed-42 smoke and were fixed BEFORE any F-gate floor was
touched, both instrumented and verified directly (raw prov/content trace sums, per-item recall rates) rather than
inferred:

**(1) `hebbian_rate_window` is GLOBAL, not per-edge** (`sim/bridge.py:1181-1207`). R1 enables it pool-wide to
allocate `cp_hebb_coactivity_trace` for its own edge; that is safe for R1 because comprehension/d6 carry no live
Hebbian pathway of their own. source_provenance DOES (its own `prov_learn`/`content_learn` edges), calibrated for
the STANDARD same-step coincidence rule. Enabling `hebbian_rate_window` for this pairing silently switched
source_provenance's OWN encode onto the untuned coactivity-trace rule and collapsed its 8-item battery to chance
(`prov_l1` stayed exactly 0.0 after its own encode). FIX: do not enable `hebbian_rate_window` for this pairing —
the cross-edge's tonic (near-constant, every-step) co-drive satisfies the STANDARD same-step rule just as well.

**(2) `hebbian_max_weight` defaults to CoreSimConfig's class default (1.0), not either organ's own calibrated
value**, whenever no descriptor declares it (neither `_SELF_SCHEMA_CONFIG` nor `_SOURCE_PROV_CONFIG` does).
source_provenance's own shared-mode wrapper temporarily raises it to 60.0 during its OWN encode and restores it
after; my ambiguous-item encode (a NEW step this runner adds, dual-context-encoding a fresh content pattern) did
not, so its Hebbian step CLIPPED every gain>0 (`prov_learn`/`content_learn`-gated) synapse — including the
ALREADY-TRAINED 8-item battery weights, which reopen under the SAME gate names — down to ~1.0, collapsing
discriminability. FIX: save+set+restore the full hebbian hyperparameter set (`hebbian_learning_rate`,
`hebbian_max_weight`, `hebbian_min_weight`, `hebbian_weight_decay`, `hebbian_symmetric`) around the ambiguous-item
encode, mirroring source_provenance's own wrapper exactly.

With both fixed, the standard-rule cross-edge grows MUCH faster per coincident step than R1's rate-windowed rule;
an initial `HMAX=40` (R1's value) overshot the moat — the silence-alone bias (0.124) EXCEEDED a genuine clear-item
decision's own margin (0.080), failing F4a/F4b outright (not a near-miss). `HMAX` was recalibrated to 6.0,
verified on 6 seeds (not tuned per-seed) before the decisive run: this is a magnitude/gain calibration of the
SAME class as R3-v3's `DA_SENSITIVITY` recalibration, never a touch to any F-gate floor.

## Result — 6/6 GO on every arm

<!--derived-->

Per-arm across seeds 42/43/44/100/101/102: **F1 6/6 - F2 6/6 - F3 6/6 - F4 6/6 - emergence 6/6 -
lesion-recovers-migration 6/6.**

| seed | final w (from 0.05) | F1 battery min_d | F2 delta_intact | F2 delta_lesion | frac_attrib | F4 silence_frac | mig sp/ss maxerr |
|---|---|---|---|---|---|---|---|
| 42 | 3.553 | 0.746 | +0.0132 | -0.0001 | 1.008 | 0.394 | 0.0 / 0.0 |
| 43 | 3.468 | 0.717 | +0.0143 | +0.0006 | 0.956 | 0.344 | 0.0 / 0.0 |
| 44 | 3.133 | 0.672 | +0.0158 | -0.0002 | 1.013 | 0.338 | 0.0 / 0.0 |
| 100 | 2.918 | 0.642 | +0.0110 | -0.0003 | 1.028 | 0.313 | 0.0 / 0.0 |
| 101 | 3.335 | 0.613 | +0.0132 | -0.0001 | 1.008 | 0.345 | 0.0 / 0.0 |
| 102 | 3.359 | 0.614 | +0.0135 | +0.0004 | 0.969 | 0.391 | 0.0 / 0.0 |

`D_FLOOR=0.50` (source_provenance's own pre-registered floor, unchanged); `F2_INTACT_FLOOR=0.010`,
`F2_LESION_RATIO=0.34`, `F4A_FRAC=0.5`, `F4B_RETAIN=0.5` (this runner's own pre-registered floors, calibrated on
seed 42's smoke, then held fixed across all 6 seeds).

## F1 - the faculty still works (edge present, author not held)

<!--derived-->

source_provenance's OWN 8-item battery keeps perfect sign accuracy (1.000) with min normalized discriminability
0.613-0.746, well clear of its own `D_FLOOR=0.50`. self_schema's OWN authorship read keeps clean separation:
`author_rate` self=0.093-0.098, heard=0.0000 EXACTLY on every seed, both sides of the calibrated threshold
0.047-0.049. Neither organ's own faculty is perturbed by the cross-edge's presence.

## F2 - the interaction is real (the crux: vary-then-lesion)

<!--derived-->

On a FRESH content pattern, dual-context encoded (interleaved perceived/generated) so its baseline provenance
read is genuinely near-tied (a real ambiguous memory, not merely an unencoded item), holding self_schema's author
pool "self" during that item's recall shifts the signed margin (`rate_generated - rate_perceived`) toward
GENERATED by +0.0110 to +0.0158 across seeds, vs a no-hold baseline. Lesioning the cross-edge (zeroing its
weight, plasticity frozen during the read) collapses the shift to -0.0003..+0.0006 — statistically indistinguishable
from zero. `tools.lab.attributable_to` reads 0.956-1.028 on every seed (one seed reads slightly above 1.0: the
lesioned control moved marginally opposite the treatment, an honestly-reported small-sample wobble, not evidence
against the edge). The interaction is load-bearing and vanishes cleanly under lesion.

## F3 - no runaway

<!--derived-->

`prov_generated`/`prov_perceived` rates stay in the physiological band (0.080-0.101 spikes/neuron/step) during
the base-intact read on every seed (`ctx_generated` is legitimately silent at recall — the context line is only
driven at encode, matching source_provenance's own `recall()` protocol, and is excluded from the band check for
that reason). The cross weight CONVERGES (bounded by `hebbian_max_weight=6.0`, growth decelerating window-over-
window) on every seed; the pool stays alive throughout.

## F4 - the moat / honesty holds

<!--derived-->

(a) Holding the author pool with NO content drive at all (pure silence) produces a margin 31-39% of a genuine
clear-item decision's own magnitude on every seed — well under the 50% no-confabulation ceiling: the bias alone
cannot manufacture a decision. (b) A CLEAR, genuinely-PERCEIVED battery item keeps its correct sign under a WRONG
(self) author hold on every seed, retaining 66-74% of its no-hold margin — comfortably above the 50% floor: the
bias reweights genuine ambiguity only, never overrides real content evidence.

## emergence - the edge LEARNED, not hand-set

<!--derived-->

The cross-edge grows from 0.05 to 2.9-3.6 by the substrate's own standard same-step Hebbian rule on every seed;
`frozen_weight_maxdrift=0.0` EXACTLY on every seed (the whitelist held byte-perfectly — no migrated or
already-trained weight moved during this edge's training).

## lesion-recovers-migration - integration added ONLY the declared edge

<!--derived-->

With the cross-edge lesioned, the pool's base connectivity is structurally identical (same (pre,post) edge set)
to the plain no-cross-edge merged pool on every seed, and BOTH organs' own battery reads match EXACTLY
(`sp_battery_maxerr=0.0`, `self_schema_maxerr=0.0` on all 6 seeds — not merely within a tolerance). This is the
byte-identical-when-off proof: absent the cross-edge, this pairing is indistinguishable from the plain merge.

## Honest scope / residuals (declared)

<!--derived-->

- **NOT strict `self-organized`** (per `docs/TERMS.md`): the weight is LEARNED from co-activity by the
  substrate's own plasticity, but the cross-edge TOPOLOGY (author -> prov_generated only, one-sided) is
  host-chosen, and the training schedule (directly co-driving author + ctx_generated, not via an organic dialogue
  turn) is host-curated — the same class of scaffold-residual R1 declared for its own referent/role schedule.
- Two-factor Hebbian (no reward/dopamine gating here, unlike R3-v3's DA-credit edge); a three-factor upgrade is a
  named follow-on, not attempted this arc.
- The ambiguous item is a balanced dual-context construction (a substrate stand-in for a genuinely uncertain real
  memory), matching R1's balanced-cue-competition item.
- **Not a production flip**: this is the R4 organ-level GO on the merge pool, additive and DEFAULT-OFF by
  construction (a standalone research runner, no `sim/` edit, no production wiring, no env-var flag to flip).
  Wiring this cross-edge into `server.py brain_chat` is a separate, later, reviewed step.

## Files

- `research/runners/_onebrain_integration_r4_selfschema_provenance.py` — the R4 runner (F1-F4 gate + emergence +
  lesion-recovers-migration; 6-seed; numpy CPU; NO `sim/` edit).
- `research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed.json` — the 6/6 GO artifact.

Functional read-outs only; no phenomenal-experience claim.
