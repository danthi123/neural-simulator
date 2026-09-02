---
type: finding
status: live
date: 2026-09-01
mechanism: onebrain-merge-pool1-composer-cleanup-region-fold
lane: onebrain-merge
artifacts:
  - research/findings/raw/_onebrain_merge_framework_smoke_composer_6seed.json
runner: research/runners/onebrain_merge_framework.py
builds_on:
  - research/findings/2026-08-27-onebrain-merge-framework-pool2-fold.md
  - research/findings/2026-08-27-merged-substrate2-retirement-framework-backed.md
---

# Board #179 — pool #1's "extra chat-wiring" regions (composer + cleanup) are now REGISTRY rows too, GO 6/6, closing the region-reservation half of the fold `2026-08-27-onebrain-merge-framework-pool2-fold.md` left open

**One-line:** Board #179 ("4 backstage brain-merging shortcuts share one general mechanism, but two still can't
be retired") named pool #1 (surprise + world-model) as the 2 remaining unfoldable organs — blocked because the
shipped `MergedSubstrate` class (`onebrain_merge_production.py`, 53KB) ALSO carries the RF-phasor recall
COMPOSER region and its phase->spike transducer CLEANUP region (pool #1's "extra chat-wiring": the faculties
that let the merged bridge answer a live chat turn), which the framework's `SURPRISE`/`WORLDMODEL`
`OrganDescriptor`s never modeled. This session registers `COMPOSER` and `CLEANUP` as two more `OrganDescriptor`
rows in `onebrain_merge_framework.py`, reproducing the shipped class's region layout **byte-identical, 6/6
seeds** (`max_init_delta=0.0` on every per-neuron init array, both regions, every seed).

## 1. The 2 remaining shortcuts and the blocker, verified against the code (not just prior findings' summaries)

`onebrain_merge_production.py`'s `MergedSubstrate` (pool #1: surprise + world-model, `_MERGE_DEFAULT_ON=True`,
53KB) is NOT a clean 2-organ merge in production: `get_merged_substrate()` also reserves a `"composer"` region
(the RF-phasor recall's resonate-and-fire ops) and a `"cleanup"` region (the phase->spike transducer), and — when
`composer_merge_enabled()`/`parser_merge_enabled()` are on (both **default-ON**, `_COMPOSER_IN_POOL1_DEFAULT_ON`
/ `_PARSER_IN_POOL1_DEFAULT_ON`) — an `"onebrain_composer"` region sized to the FULL production `OneBrainComposer`
layout span, with its Hebbian PARSER transplanted onto the pool (`_bind_parser_onto_pool`: weight transplant +
a permanent per-synapse gain-0 freeze + isolation-wrapped stepping). None of that is representable by
`SURPRISE`/`WORLDMODEL` alone, which is exactly why `2026-08-27-onebrain-merge-framework-pool2-fold.md` left
pool #1's retirement as "NOT met" while pool #2 (`MergedSubstrate2`, metacog+pragmatic) WAS thinned to a
framework-backed shim the same day (`2026-08-27-merged-substrate2-retirement-framework-backed.md`,
`2c771f605`). **4/4 of the underlying pool organs (surprise, world-model, metacog, pragmatic) have been
declaratively expressible since 2026-08-27** — the blocker was never the two ORGANS, it was pool #1's
CLASS carrying additional non-organ wiring the registry didn't model.

## 2. What this session folded — the region-RESERVATION half, not the composer's BIND

`COMPOSER` and `CLEANUP` are two new `OrganDescriptor` rows (`onebrain_merge_framework.py:816-847`, right after
`WORLDMODEL`): each a trivial descriptor per the schema's own comment ("a trivial organ needs only
key/regions/spec_fn/config; the rest default off") — one IDLE `BrainRegion` (`exc_fraction=1.0,
internal_density=0.0`, zero weights, `plastic_internal=False`), no pathways, no `organ_cls`/`read_fn`. Sized
from the SAME constants (`_COMPOSER_D`, `_COMPOSER_KMAX`, `_CLEANUP_BLK`, `_CLEANUP_VOCAB`) `MergedSubstrate`'s
own defaults use, reused by import — one definition of the geometry, no copy to drift.

**What stays exactly where it is, on purpose:** the composer's BIND (`Pool1BoundComposer.bind_to_pool1`,
`Pool1BoundOneBrainComposer`/`_pool1_onebrain_init`'s index-rebase math, `_bind_parser_onto_pool`'s weight
transplant + permanent gain-0 freeze + isolation-wrapped stepping) is genuinely composer-specific business
logic — subclassing a third-party `OneBrainComposer`/`RFPhasorComposer`, not a generic cross-organ merge
concern — and is NOT migrated here. It already only needs a `.bridge` / `.ensure_built()` / a region-index
accessor from whatever pool it binds to, all of which `MergedPool` already provides, so nothing about this fold
requires touching it. This is the SAME division of labor the R1/R4 `CrossEdge` migrations drew between "the
engine wires the edge" and "the organ's own read/answer logic stays the organ's."

The `_ONEBRAIN_SPAN`-sized `"onebrain_composer"` region (the production-default b-closer path) and the parser
transplant are **not** folded this pass — `_onebrain_layout_span`'s dynamic sizing (computed from `D`/`vocab`/
`k_max`/`enable_attributed`/`vocab_headroom` at call time, mutated into a module-global before first build) and
`_bind_parser_onto_pool`'s CONFLICT-C homeostasis flip (conditionally overriding SURPRISE/WORLDMODEL's own
`region_flags`) don't fit a fixed, import-time-registered descriptor the way `COMPOSER`/`CLEANUP`'s STATIC sizing
does. Declared honestly as the genuine remaining gap (§4), not hidden.

## 3. Verification — byte-identical, 6/6 seeds, the SAME bar `_smoke` set for pool #1's two organs

New CLI mode `--smoke-composer` (`onebrain_merge_framework.py`, mirrors `_smoke` exactly): builds
`merge_organs([SURPRISE, WORLDMODEL, COMPOSER, CLEANUP], seed=seed)` and the shipped
`MergedSubstrate(seed=seed, organs=("surprise","worldmodel","composer","cleanup"))`, then compares every
per-neuron init array (`cp_neuron_firing_thresholds`, `cp_membrane_potential_v`, `cp_recovery_variable_u`, the
5 Izhikevich params) over all 4 regions.

**Result, 6 seeds (42, 43, 44, 100, 101, 102; numpy CPU, bit-exact)**
(`research/findings/raw/_onebrain_merge_framework_smoke_composer_6seed.json`): 6/6 GO, `max_init_delta=0.0`
every seed, `n_engine==n_shipped==6064` every seed. The composer/cleanup regions carry no pathways (idle
placeholders in both constructions), so this init-only bar already covers everything either path does to them
— there is no wiring for `wire=True` to add.

**Regression (unchanged code paths, re-run after the edit):** `--smoke` (pool #1, 2-organ, seed 42):
`max_init_delta=0.0`, unchanged from before this session. `--smoke2` (pool #2, seed 42): `init_delta=0.0`,
`read_delta=0.0`, `all_go=True`. `--determinism2` (seed 42): `identical=True`. `REGISTRY` now has 13 keys
(11 before + `composer` + `cleanup`); `--keys all`/`GROUP_A_KEYS` is a SEPARATE list untouched by this addition,
so no existing batch sweep newly includes these two non-organ rows.

**Term check (`docs/TERMS.md`):** "byte-identical" is asserted from the data above (exact 0.0-delta compare on
every cited array, every seed), never inferred from reading the code. "GO" is the smoke's own `byte_identical`/
`all_go` verdict, not a metric lifted from elsewhere.

## 4. What this does NOT close — the honest residual

**`MergedSubstrate` is still NOT retired.** This fold makes the region LAYOUT declaratively expressible; it does
not touch `MergedSubstrate.ensure_built()` itself (still the same ~150-line hand build), and it does not attempt
the `"onebrain_composer"` region or the parser transplant (the PRODUCTION-DEFAULT path — `composer_merge_enabled()`
and `parser_merge_enabled()` are both True by default). Two further rungs remain before the family-wide
"`MergedSubstrate*` can be retired" claim holds:

1. **Thin-wrapper the RF-phasor path** (`organs=("surprise","worldmodel","composer","cleanup")`): rewrite
   `MergedSubstrate.ensure_built()` to delegate to `merge_organs([SURPRISE, WORLDMODEL, COMPOSER, CLEANUP],
   seed=self.seed)` internally (the `MergedSubstrate2` pattern, `2026-08-27-merged-substrate2-retirement-
   framework-backed.md`), preserving `.bridge`/`.cfg`/`.xp`/`composer_idx()`/`cleanup_idx()`/`surprise_idx_map()`/
   `worldmodel_idx_map()`/`read_isolation()` unchanged, verified against EVERY existing caller (not just the
   production entry point) the way `_onebrain_merge2_retire_verify.py` did for pool #2. This rung is now
   well-de-risked by this session's byte-identity result but not yet attempted.
2. **The `"onebrain_composer"` + parser-transplant path** (the ACTUAL production default): needs either (a) a
   descriptor-factory mechanism that accepts a runtime-computed region size (closing the gap between
   `OrganDescriptor.spec_fn(seed)`'s fixed-at-registration-time contract and `_onebrain_layout_span`'s
   call-time-dependent span) plus a way to express CONFLICT-C's conditional `region_flags` override on
   SURPRISE/WORLDMODEL (only when parser_on_pool), or (b) accepting the composer/parser BIND as a permanent,
   documented exception the registry constructs regions for but never binds — this session's fold is consistent
   with either follow-up, since it does not touch the bind logic at all. Materially bigger than rung 1: it
   crosses from "declare a static idle region" into "the registry must accept a build-time size and a
   cross-descriptor conditional config," which is new mechanism, not a data row under the existing schema.

No `sim/` edit. No production code path touched (`onebrain_merge_production.py` unmodified). Additive only:
new descriptors + a new smoke mode in `onebrain_merge_framework.py`.

## Files

`research/runners/onebrain_merge_framework.py` — `_composer_spec`/`_cleanup_spec`/`COMPOSER`/`CLEANUP`
(after `WORLDMODEL`), `REGISTRY` (now 13 keys), `_smoke_composer`, `--smoke-composer` CLI mode. Read against:
`research/runners/onebrain_merge_production.py` (`MergedSubstrate.ensure_built`, `_COMPOSER_D`/`_COMPOSER_KMAX`/
`_CLEANUP_BLK`/`_CLEANUP_VOCAB`, `Pool1BoundComposer`, `_bind_parser_onto_pool`, `_pool1_onebrain_init` — all
unmodified) · `research/findings/2026-08-27-onebrain-merge-framework-pool2-fold.md` (named this exact gap) ·
`research/findings/2026-08-27-merged-substrate2-retirement-framework-backed.md` (the pattern rung 1 above
follows) · `research/findings/2026-08-27-onebrain-merge-framework-DESIGN.md` (the `OrganDescriptor` schema).
