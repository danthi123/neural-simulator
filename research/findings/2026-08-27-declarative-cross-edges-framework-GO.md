---
type: finding
status: live
date: 2026-08-27
mechanism: declarative-cross-edges-framework
lane: onebrain-integration/framework
artifacts:
  - research/findings/raw/_onebrain_declarative_crossedge_r1_repro_6seed.json
  - research/findings/raw/_onebrain_merge_framework_smoke_s42.json
runner: research/runners/onebrain_merge_framework.py
supersedes_diagnosis_of: research/findings/2026-08-27-onebrain-completeness-audit.md
---

# Cross-organ synapses are now a REGISTRY ROW, not a bespoke ~40KB runner — `cross_edges=[CrossEdge(...)]` on `merge_organs`, verified to reproduce R1's hand-wired d6->comprehension edge bit-for-bit across 6 seeds — GO

**One-line:** The completeness audit's #1-ranked framework investment (`research/findings/2026-08-27-onebrain-
completeness-audit.md` §4 step 2) is built: `onebrain_merge_framework.py` gained a `CrossEdge` dataclass + a
`cross_edges=` param on `merge_organs`/`MergedPool` + `MergedPool.apply_cross_edge_freeze()`, which apply a
declared cross-organ synapse via the SAME `inject_explicit_wiring` + gain-0-freeze-whitelist machinery the
bespoke R1/R3-v3/R4 runners each hand-wrote. Re-expressing R1's bespoke d6->comprehension edge as 4 `CrossEdge`
rows reproduces it **bit-for-bit** (grown-weight max|delta| = 0.000) across 6 seeds, and the framework's
byte-identical-off path is unchanged (re-verified against the independently hand-written `MergedSubstrate`/
`MergedSubstrate2`).

## 1. The schema

`CrossEdge(key, source_key, source_region, target_key, target_region, init_weight=0.05, plastic=True, gate=None,
learn_rule="rate_hebbian", freeze_rest=True)` in `research/runners/onebrain_merge_framework.py:75-121`.
`source_region`/`target_region` are region NAMES (e.g. `"w0"` -> `"sel_agent"`); `gate` defaults to `key`;
`learn_rule` is a documentation field (`"rate_hebbian" | "da_credit" | "none"`) — the caller still supplies the
hebbian/reward hyperparameters via a config-only descriptor unioned into `config_descriptors`, exactly as R1's
`types.SimpleNamespace(config={"hebbian_rate_window": True})`; no new hyperparameter-union mechanism was needed.

`merge_organs(descriptors, ..., wire=True, cross_edges=[CrossEdge(...), ...])` — a new optional param, threaded
to `MergedPool.__init__`. `MergedPool.ensure_built()` raises `MergeConflict` if `cross_edges` is given without
`wire=True` (a cross-synapse needs the actual wired substrate, fail loud not silent). When `wire=True`, the
existing `_install_organ_read_wiring` step (which already unions the per-region-seamed base wiring plan +
each descriptor's `explicit_wiring_fn`) now ALSO unions `_cross_edge_dense(bridge, ce)` for each declared edge —
a small new helper that builds the same dense pre/post/weight population shape R1's hand-typed `_dense(...)`
built, reading its two endpoints off `region_manager.indices(name)`. One `inject_explicit_wiring` call installs
base + organ + cross wiring together, in the SAME insertion order R1's hand-typed union used.

`MergedPool.apply_cross_edge_freeze()` — a new method generalizing R1Pool's 3 hand-typed lines
(`set_plasticity_gate(GATE,1.0)` / `cp_plasticity_rate_gain[:]=0.0` / `set_plasticity_gate(GATE,1.0)`) to every
registered edge with `freeze_rest=True`: ensure each edge's gate exists, zero the WHOLE pool's plasticity gain,
then re-open only the declared edges' gates. It is a no-op when no edge wants it. The caller invokes it AFTER
constructing any `organ_cls(shared=pool)` that installs its own weights (comprehension's cue->role validities),
exactly the point R1Pool's hand-typed version runs — so an organ's own build-time setup is never clobbered by a
freeze that ran before the organ existed.

## 2. Byte-identical-off (cross_edges=None/empty)

Every new code path is gated on `self.cross_edges` being truthy: the wiring-union loop, the `MergeConflict`
guard, and `apply_cross_edge_freeze`'s freeze body all no-op when the list is empty (the default). Re-ran the
framework's existing independent-comparison gates after the patch — each compares the descriptor-built pool
against a SEPARATE hand-written class (`MergedSubstrate`/`MergedSubstrate2`), not a self-referential hash:
`--smoke` (pool #1, surprise+worldmodel): `max_init_delta=0.0`, byte_identical=True. `--smoke2` (pool #2,
metacog+pragmatic, wire=True): `max_init_delta=0.0`, `read_max_delta=0.0`, both answer batteries preserved,
`all_go=True`. `--determinism2` (build-twice-at-one-seed hash): `identical=True`. `tests/test_determinism.py`:
9 passed, 2 skipped (unchanged). None of these paths pass `cross_edges`, so this is the same evidence the
framework already carried before this patch, re-confirmed clean after it.

## 3. The reproduction proof — R1's d6->comprehension edge, re-expressed declaratively

New file `research/runners/_onebrain_declarative_crossedge_r1_repro.py`. `DeclarativeR1Pool` subclasses the
bespoke `R1Pool` (`_onebrain_integration_r1_wm_comprehension.py`) and overrides ONLY `__init__`: instead of the
bespoke `_build_pool`'s hand-typed `_dense(...)` union (injected twice — once by the framework's own automatic
wire=True path, thrown away, once manually with the cross edges added) + the 3 hand-typed whitelist lines, it
calls `merge_organs([D6, COMP], wire=True, cross_edges=CROSS_EDGES)` (ONE inject; cross edges already included)
then `pool.apply_cross_edge_freeze()`. `CROSS_EDGES` is 4 `CrossEdge` rows matching R1's `x_w0_sela`/`x_w0_selp`/
`x_w1_sela`/`x_w1_selp` edges exactly (same source/target regions, same `W0=0.05` seed weight, same `GATE` name).
Every downstream method (`_hard_reset`, `_drive`, `train`, `amb_read`, `cross_weights`, `_wmean`) is INHERITED
unchanged, so the train/read PROTOCOL is provably identical between the bespoke and declarative arms — only pool
CONSTRUCTION differs. Both arms then run through R1's own unmodified F1/F2/F3/F4/emergence/lesion-recovers-
migration functions (reuse-by-import, no reimplementation of the functional gate).

**Result, 6 seeds (42, 43, 44, 100, 101, 102), both arms run fresh in the SAME process per seed (not compared
against stale historical numbers):** 6/6 seeds — bespoke GO, declarative GO, reproduction check GO. Grown-weight
max|delta| between the two arms = **0.000** at every seed (bit-for-bit: e.g. seed 42 `w0->A` 13.61==13.61,
`w1->P` 12.44==12.44; seed 102 `w0->A` 10.52==10.52, `w1->P` 11.54==11.54). F2's lesion-attributable fraction
(`tools.lab.attributable_to`) reads **1.0/1.0 in BOTH arms at every seed** — the WM-held-referent shift is
100% attributable to the cross-edge and vanishes on lesion, identically whether the edge was hand-wired or
declared. This is a stronger bar than the pre-registered check (weight max|delta|<1.0, attributable-fraction
within 0.15): the two construction paths produce numerically IDENTICAL substrates, not merely
functionally-equivalent ones — expected, since `_cross_edge_dense`'s `np.repeat`/`np.tile` construction and
insertion position reproduce R1's hand-typed `_dense(...)` exactly.

Self-check against the CLAUDE.md research-workflow discipline: this is an engineering-abstraction-fidelity claim
(does path B reproduce path A), not a biological "surpass" claim, so the full `verify-go` adversarial-skeptic
protocol was judged not to apply — but its spirit was followed directly: both arms were re-run fresh in-process
(no stale numbers), the full 6-seed sweep is reported unfiltered (no cherry-picked seed), and the comparison used
exact equality, not a loose threshold.

Artifact: `research/findings/raw/_onebrain_declarative_crossedge_r1_repro_6seed.json`.

## 4. What this does NOT close

The `learn_rule` field is documentation-only — a `"da_credit"` edge still needs the caller to union in the right
`enable_reward_modulation`/STDP config (R3-v3/R4's pattern), the framework does not yet auto-derive it. R3-v3 and
R4 were not re-expressed here (R1 was chosen as the simplest of the three per the task scope); the audit's step
3 (R4 into production) still needs its own work. No production default changed; no `sim/`/`webapp/` file edited.

## Files

`research/runners/onebrain_merge_framework.py` — `CrossEdge:75`, `_cross_edge_dense:113`,
`MergedPool.__init__` (`cross_edges` param), `MergedPool.ensure_built` (the `MergeConflict` guard),
`_install_organ_read_wiring` (the cross-edge union loop), `MergedPool.apply_cross_edge_freeze`, `merge_organs`
(the `cross_edges` param). `research/runners/_onebrain_declarative_crossedge_r1_repro.py` — the reproduction
proof (new file). Read against: `research/runners/_onebrain_integration_r1_wm_comprehension.py` (the bespoke R1
runner, unmodified), `research/findings/2026-08-27-onebrain-completeness-audit.md` (the ranking that named this
investment #1), `research/findings/2026-08-27-onebrain-integration-phase-DESIGN.md` (the original `cross_edges`
aspiration this closes).
