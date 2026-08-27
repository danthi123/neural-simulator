---
status: live
type: finding
lane: onebrain-merge
date: 2026-08-27
---

# `MergedSubstrate2` (pool #2) RETIRED to a framework-backed shim — production repointed to `merge_organs()`

**Verdict: GO (6/6 seeds, all combos).** This closes the rung `2026-08-27-onebrain-merge-framework-pool2-fold.md`
named "well-scoped, not a discovered blocker": the bespoke `onebrain_merge_production2.MergedSubstrate2` class
(pool #2 — E1 metacog + D-pragmatics) no longer builds its own `SimulationBridge` by hand. Its `ensure_built()`
now DELEGATES to the declarative `onebrain_merge_framework.merge_organs()` engine — the same engine every other
registered organ (surprise/world-model + the 7 Group-A organs) builds through. `get_merged_substrate2()` (the
production factory `/api/brain-chat`'s metacog/pragmatic organs read through) is unchanged in signature and is
now framework-backed transitively, because it just constructs `MergedSubstrate2(...)`.

Artifact: `research/findings/raw/2026-08-27-merged-substrate2-retirement-framework-backed.json` (the pre-repoint
regression gate — the bespoke class vs the framework engine, two INDEPENDENT implementations, before the repoint)
Artifact: `research/findings/raw/2026-08-27-merged-substrate2-retirement-post-repoint-regression.json` (the same
runner re-run AFTER the repoint, confirming the delegation wiring itself introduces no exception/behavior change)
Runner: `research/runners/_onebrain_merge2_retire_verify.py`

## Why this rung, and why it was safe to take now

`2026-08-27-onebrain-merge-framework-pool2-fold.md` had already proven the PRODUCTION 2-organ combo
(`organs=("metacog","pragmatic")`, what `get_merged_substrate2()` builds) round-trips byte-identically —
per-region init arrays AND the real `judge()`/`interpret()` reads — against the framework registry pool, 6/6
seeds (`onebrain_merge_framework.py --smoke2`). It explicitly declined to flip the class because `MergedSubstrate2`
ALSO has single-organ constructor callers outside the production entry point — `_metacog_robust_confidence_derisk.py`'s
`MergedSubstrate2(organs=("metacog",))` / `(("pragmatic",))`, and `_onebrain_production_flip2_verify.py`'s identical
CORESIDENT-baseline pattern — whose exact behavior a thin-wrapper refactor must ALSO preserve, a code path the
existing `--smoke2` never exercises.

This session's runner (`_onebrain_merge2_retire_verify.py`) closes that gap: for each of 6 seeds it builds THREE
combos two ways each (the bespoke `MergedSubstrate2` vs the framework `merge_organs([...], wire=True)`) — the
production 2-organ combo, the metacog-solo combo, and the pragmatic-solo combo — and requires byte-identity on
(a) a whole-bridge SHA1 fingerprint (every per-neuron init array + the full wired connectivity, sorted to a
canonical edge order — stronger than the existing `--smoke2`'s per-region-slice compare, since it also proves no
divergence anywhere else on the bridge) and (b) the REAL production reads (`MetacogProductionOrgan.judge()`'s
balance margin, confident/uncertain decision, and self-calibrated threshold; `PragmaticProductionOrgan.interpret()`'s
belief distribution and rendered enriched-interpretation phrase). All 18 (3 combos x 6 seeds) checks passed before
any repoint was made — genuine independent-implementation agreement, not a tautology.

**Term check (`docs/TERMS.md`):** "byte-identical" is asserted from the data above (SHA1 hash equality + exact
0.0-delta compare on every numeric read, captured in the pre-repoint artifact), never inferred from reading the
code. "GO" is the gate's own verdict (`all_go` in the per-seed JSON), not a metric lifted from elsewhere.

## What changed in `research/runners/onebrain_merge_production2.py`

`MergedSubstrate2.ensure_built()` was rewritten from a ~85-line hand build (constructing `CoreSimConfig`,
`SimulationBridge`, the region/pathway union, the wiring inject, the settle-to-rest snapshot) to a ~15-line
delegation: it selects the `METACOG`/`PRAGMATIC` `OrganDescriptor`s matching `self.organs` and calls
`onebrain_merge_framework.merge_organs(descs, seed=self.seed, wire=True)`, then copies `.bridge`/`.cfg`/`.xp`/
`.snap` off the returned `MergedPool`. `metacog_idx()`/`pragmatic_item_dev()` were thinned to dispatch to the
pool's own `metacog_idx()`/`pragmatic_item_dev()` methods (`onebrain_merge_framework._metacog_idx_fn`/
`_pragmatic_idx_fn` — the SAME computation these methods used to run inline). `_metacog_specs()`/
`_pragmatic_specs()` are UNCHANGED (they remain the single source of geometry the framework's
`_pool2_metacog_specs`/`_pool2_pragmatic_specs` reuse-by-import via a throwaway instance — changing them would
have created a circular self-reference). The public API (`.bridge`/`.cfg`/`.xp`/`.snap`/`.metacog_idx()`/
`.pragmatic_item_dev()`/`.ensure_built()`, the `organs=` constructor) is unchanged; `get_merged_substrate2()`,
`merge2_enabled()`, and the `BRAIN_ONEBRAIN_MERGE2` gate (including the `=0` escape to two separate bridges) are
untouched. Dead imports the old build used (`WS_LOOP_GATE`, `DEFAULT_ATTRACTOR_WEIGHT`, `DEFAULT_NMDA_TAU`, the
`_gnw_rung1_ignition_curve_derisk` wiring/settle helpers, `numpy`) were removed; nothing outside this file
imported them (checked: `grep -rn "from research.runners.onebrain_merge_production2 import"`).

`MergedSubstrate2` was NOT deleted — `_metacog_robust_confidence_derisk.py`, `_onebrain_production_flip2_verify.py`,
and `onebrain_merge_framework.py` itself (reuse-by-import of `_metacog_specs`/`_pragmatic_specs`) all import it
directly, several with the single-organ constructor pattern. It is retired IN SUBSTANCE (a thin shim over the
declarative engine, per the module's new "RETIRED-TO-A-SHIM" docstring section) rather than in name.

## Regression sweep after the repoint (all re-run against the worktree post-edit)

| check | result |
|---|---|
| `_onebrain_merge2_retire_verify.py --seeds 42..102` (post-repoint; now the shim vs itself through the framework — a wiring/exception sanity check, not independent agreement) | 6/6 GO, identical numbers to the pre-repoint run |
| `onebrain_merge_framework.py --smoke2 --seeds 42,100` (pre-existing framework self-check, also builds `MergedSubstrate2` directly) | 2/2 PASS, `init_delta=0.0` / `read_delta=0.0` |
| `_onebrain_production_flip2_verify.py --seeds 42` (the production flip's own end-to-end regression gate; explicitly named in this rung's scope as "must keep working") | seed 42: `one_pool=True(N=450)`, MERGED==CORESIDENT byte-id `mcΔ=0.00e+00 prΔ=0.00e+00`, `FULL FLIP-GO=1/1` — unchanged from its historical result |
| `_metacog_robust_confidence_derisk.py --seeds 42` (the other single-organ-constructor caller) | `nmda_norm` read `FULL FLIP-GO=1/1` (matches `2026-08-13-metacog-robust-confidence-GO.md`'s documented production default); `balance` read `FULL FLIP-GO=0/1` (the KNOWN pre-existing negative for that non-default read — unaffected by this change) |

No `sim/` file was touched. No pytest module references `onebrain_merge_production2`/`MergedSubstrate2`/
`onebrain_merge_framework` (checked via `grep -rl` over `tests/`), so this is a pure runner/webapp-side migration.

## What this does NOT close

Pool #1 (`onebrain_merge_production.MergedSubstrate`, surprise + world-model) is UNTOUCHED — it also carries the
RF-phasor recall composer + parser production wiring the framework's `SURPRISE`/`WORLDMODEL` descriptors do not
yet model (`2026-08-27-onebrain-merge-framework-pool2-fold.md`'s "What is NOT done" #2), so the family-wide
"`MergedSubstrate*` can be retired" claim still does not hold — only pool #2 is retired here. This is a
MIGRATION-safety result (the framework reproduces the bespoke build byte-for-byte); it is not a claim about
cross-organ INTEGRATION (zero cross-synapses before and after, unchanged).
