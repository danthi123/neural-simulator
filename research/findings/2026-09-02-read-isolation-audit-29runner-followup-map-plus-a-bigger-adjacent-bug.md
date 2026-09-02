---
type: finding
status: live
date: 2026-09-02
mechanism: read-isolation-audit-c2-bug-class-29runner-followup
board: one-brain integration / measurement-integrity
artifact: research/findings/raw/_read_isolation_audit_29/audit_29runners.json
---

# Read-isolation C2 audit, ~29-runner follow-up: 5 files carry their own hand-rolled reset (1 real leak fixed, 1 hygiene-ported, 3 confirmed clean), 26 inherit an already-fixed base — and diagnosing the leaks surfaced a BIGGER, separate bug live in production

**2026-09-02.** `research/FAILURE_LOG.md`'s 2026-09-02 row (the `ProvToAuthorPool._hard_reset` fix) flagged
"~29 further `research/runners/*.py` files" sharing the hand-rolled `_hard_reset` shape beyond the original
14-runner audit (`2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`) as an un-audited gap.
This finding closes that gap: a precise re-enumeration, the same mechanism-zeroed two-read diagnostic the
original audit used, and — because a "clean" verdict earns the same scrutiny as a "leak" verdict — a deep dive
into two files whose repeat-read divergence did NOT close after the C2 port, which surfaced a second,
independent, and more consequential bug.

## Step 1 — precise enumeration

`grep -rl 'def _hard_reset\|def _reset_dynamics\|def _snapshot_dynamics' research/runners/*.py` gives 21 files
with an exact `def`; broadening to any MENTION of those names (calls, not just defs) gives 51. Subtracting the
17 files the original 14-runner audit + its landed follow-ups already cover (the 14 + `_crossedge_surprise_metacog_derisk.py`
the origin file + `_d3_spiking_attractor_derisk.py` H-2 + `_onebrain_integration_r3v3_functional_drive.py`) and
the 5 pmem/prospective_memory (Wave-3) files this task excludes leaves **31 files**, matching the FAILURE_LOG
row's "~29" figure. Full lists are in `audit_29runners.json`'s `enumeration` + `excluded_out_of_scope` fields.

## Step 2 — an AST scan cuts the real audit surface from 31 to 5

`research/findings/raw/_read_isolation_audit_29/scan_hard_reset.py` parses every target file's AST and reports
whether it defines its OWN `_hard_reset`/`_reset_dynamics`/`_snapshot_dynamics` (a genuine, independent reset
site that needs its own audit) or only IMPORTS a Pool/Organ class from another file (inheriting whatever that
base's reset already does). **26 of the 31 have no own definition** — they are thin production-organ wrappers
(`surprise_production_organ.py`, `worldmodel_production_organ.py`), declarative-crossedge subclasses
(`DeclarativeR4Pool(R4Pool)`, `DeclR1Pool(R1Pool)`), or xedge production co-locators
(`onebrain_xedge_curiosity_d6_production.py`) whose base file is either already fixed by the original audit's
follow-ups, already in the original audit's CLEAN bucket, or self-contained-correct (3 files —
`_onebrain_composer_merge_derisk.py`, `_one_brain_merge_configsuperset_production_derisk.py`,
`onebrain_merge_production.py` — each declares its own complete `_PER_NEURON_STATE` tuple with all 4 C2 arrays,
verified by direct read). These 26 are classified by inheritance chain, **not individually re-run** (time
budget) — an honest limitation, not a claim of dynamic verification; a static scan cannot catch a monkeypatched
override, though none was observed.

**5 files carry their own hand-rolled reset** and got the full dynamic two-read diagnostic:
`comprehension_production_organ.py`, `sc_orienting_production_organ.py`, `_navsc_merged_opcheck.py`,
`_laneC_source_provenance_opponent_derisk.py`, `_replay_dg_pattern_separation_gate.py`.

## Step 3 — the 5 dynamic results

**`comprehension_production_organ.py` (`ComprehensionProductionOrgan`) — LIVE production (Gate-B, every
conversational turn) — real leak, fixed, 82-96% reduction.** The original `_hard_reset` restored (v,u) and
zeroed conductances/firing/current but zeroed a NONEXISTENT `cp_refractory` (the exact dead typo the original
audit found elsewhere) and never restored `cp_prev_firing_states` / the homeostatic pair / 3 NMDA-recurrent
conductance arrays. A repeat-read on the SAME (n0,v,n1) diverged 0.3375 vs 0.2958333333333334 (delta 0.0417, ~12%
relative); order-dependence delta 0.0514. Porting the full C2 set + the NMDA-recurrent conductances dropped
these to 0.007638888888888917 (repeat) and 0.002083333333333326 (order) — an 82% and 96% reduction respectively. `selftest_read_isolation()`
(`--selftest`) regression-guards this, disabling OU explicitly (see next paragraph for why).

**The residual 0.007638888888888917 is NOT OU noise and NOT the C2 class — it is a separate, bigger bug.** A before/after
per-array diff (every `cp_*` attribute on the bridge, snapshotted immediately after each of two `_hard_reset()`
calls bracketing one full read) found **zero differences of any kind** — the per-neuron dynamical state is
provably bitwise-identical after the fix. `enable_ou_process` is confirmed `False` for this bridge (not the
source), and `OMP_NUM_THREADS=1` reproduces the identical residual (rules out BLAS nondeterminism). The actual
cause: `cp_connections.data` (excluded from the first diff pass by an oversight, then checked directly) DOES
change between reads — max abs diff 13.8 after ONE read, growing to **56.1 after 30 successive reads** on one
process-shared organ (exactly how `get_organ()` is used in production — a single instance read repeatedly
across a server process's lifetime), approaching the `hebbian_max_weight=60.0` clamp, while the well/ill margin
drifts from ~0.33 to ~0.35 (~6% relative). Root cause, confirmed directly: `cp_plasticity_rate_gain` (the array
the runtime Hebbian update actually consults) has exactly 23040 zeros — the 4 NAMED cue gates × 5760 synapses —
and 1710 values at 1.0 (fully plastic); **1710 is the exact count of synapses independently observed to
change weight.** Those 1710 are the INTERNAL recurrent `sel_agent`/`sel_patient` + `sel↔sel_FS` pathways,
every one declared `plastic=False` / `plastic_internal=False` in the `BrainRegion`/`RegionPathway` spec.
`cp_plastic_mask` exists as a per-synapse array but is grep-confirmed **never read** by the Hebbian update
kernel — only referenced during CSR-rebuild remap bookkeeping (`sim/bridge.py:5087-5155`). **A pathway declared
non-plastic is NOT actually protected unless it is ALSO assigned a named `plasticity_gate` held at 0** — the
declaration alone is dead configuration whenever the bridge's `enable_hebbian_learning` is globally `True`.
The SAME pattern reproduces in `_laneC_source_provenance_opponent_derisk.py` (max abs diff 13.3 then 9.6 across
2 frozen recalls, `prov_learn`/`content_learn` gates held at 0 throughout — the opponent-inhibition pathways are
the ungated ones there). This is a bigger, framework-level, LIVE-production concern than the C2 array class this
task was scoped to audit — see "Adjacent bug" below; it is NOT fixed in this finding.

**`sc_orienting_production_organ.py` — clean; the visible jitter is genuine, by-design OU noise, not a leak.**
Missing `cp_prev_firing_states` + the homeostatic pair (this organ's `CoreSimConfig` default
`enable_homeostasis=True` is never overridden, unlike several sibling runners, so these are NOT config-inert
here). A repeat-read showed N-cardinal counts of 15 vs 16 (sc_total 185 vs 184) — but this NUMBER IS IDENTICAL
BEFORE AND AFTER THE FIX, and disabling `enable_ou_process` (confirmed nonzero `ou_std_current_pA=6.0`,
deliberately set — "Low background OU so the retinotopic input forms a clean bump") makes repeat-reads bitwise
identical on BOTH the pre-fix and post-fix code. `enable_hebbian_learning=False` for this bridge (confirmed),
so it is immune to the adjacent bug above. The fix (added anyway, cheap defense-in-depth) changed nothing
observable — cortex_N wins 15-18 vs 0 every trial regardless, vastly clear of the banked N1 6-seed GO margin
(SC/host 0.883 <!--derived--> — from the N1 CLOSED finding this organ packages, not this audit's own artifact).

**`_navsc_merged_opcheck.py` — confirmed clean, no fix applied.** Missing the homeostatic pair +
`cp_prev_firing_states` (`cp_refractory_timers` was already correctly zeroed). This is a STEP-0 co-residence
scoping probe (gates a downstream 6-seed A/B, not itself a banked verdict) that builds the FULL merged
nav+conv bridge (~3.5 min CPU — the only target expensive enough to warrant background execution). A
repeat-read and an order-dependence test (present the same (agent,goal) case twice, then with an intervening
different case) were **bitwise identical** in every field (peak Hz, per-cardinal counts, reward_us rate) — the
160-step read window with a 30-step forced-drive warmup fully washes out the missing arrays' residue, matching
the original audit's R1/R2 clean pattern. No fix applied; none needed.

**`_laneC_source_provenance_opponent_derisk.py` (`ProvenanceBrain`) — banked GO (#129), leak present but
margin ≫ leak, hygiene-ported.** `_DYN_ATTRS` already covered `cp_refractory_timers` and
`cp_prev_firing_states`; missing the homeostatic pair (config-inert — `enable_homeostasis=False`) and the
synaptic delay ring buffer (`cp_synapse_pulse_timers`/`progress` — undrained, the exact mechanism
`_replay_dg_pattern_separation_gate.py`'s own docstring names: "an in-flight spike from one
replay/probe event cannot leak into the next"). A repeat-recall diverged `rate_perceived` 0.0984375 → 0.099375,
computing normalized discriminability d=0.898 → 0.904 (delta 0.006) against this file's own
`D_FLOOR=0.50` with the seed's own margin at d~0.9 (~0.4 headroom — the delta is 60×+ below the floor
clearance). Porting the pulse-timer drain + homeostatic restoration changed NOTHING (same numbers before and
after) — confirms the dominant driver here is the SAME adjacent weight-drift bug found in
`comprehension_production_organ.py` (opponent-inhibition pathways are the ungated ones), not the C2/pulse-timer
class. Documented honestly as hygiene, not a fix, since the file's own residual leak is dominated by a
different, unfixed mechanism.

**`_replay_dg_pattern_separation_gate.py` — confirmed clean before AND after a hygiene port.** Already drained
the synapse-pulse ring buffer and restored `cp_prev_firing_states`; only missing `cp_refractory_timers`
(the homeostatic pair is config-inert — `enable_homeostasis=False`, confirmed in `build_bridge`). A repeat-probe
and order-dependence test (using `smoke_config()`, after `_replay_consolidate` populates real learned weights so
the test isn't measuring an all-zero degenerate case) were **bitwise identical both before and after** adding
the `cp_refractory_timers` zeroing — the `replay_settle_steps` window already washes out the missing residue.
This runner's own banked verdict (`2026-08-03-...-NO-GO.md`) already survived a prior, DIFFERENT read-boundary
investigation (board #91, `_pop_state`); this result adds the C2 angle to that record as also-clean.

## Verdict-flip candidates: zero

No file in this follow-up round changed a banked GO/NO-GO/UNDEFINED verdict. The one file with a real,
measurable C2 leak (`comprehension_production_organ.py`) is not itself a discrete GO/NO-GO gate — it is a
threshold-calibrated production monitor — and its leak was fixed. Every other file's leak (where present) sits
1-2 orders of magnitude below its own margin.

## Adjacent bug discovered (NOT fixed here — flagged for follow-up)

**`RegionPathway(plastic=False)` / `BrainRegion(plastic_internal=False)` is not enforced by the runtime Hebbian
rule.** Only a synapse assigned to a NAMED `plasticity_gate` held at 0 is actually protected;
`cp_plasticity_rate_gain` defaults to 1.0 (fully plastic) for every other synapse whenever
`enable_hebbian_learning` is globally `True`. Confirmed in two files (evidence above); root cause lives in
`sim/bridge.py`'s Hebbian update kernel, not in any `research/runners/` file, so it is out of this audit's
scope and NOT attempted here. Severity is HIGH: `comprehension_production_organ.py` is Gate-B, wired into
`/api/brain-chat` by default, and a live server process holds ONE process-shared organ instance
(`get_organ()`) that silently drifts toward its plasticity clamp purely from being read, no retraining signal
involved — the build-time-calibrated well/ill abstain threshold measurably goes stale over a session's
lifetime. See `research/FAILURE_LOG.md`'s new row and the spawned follow-up task for the reproduction recipe.

## Fix recipe applied (mirrors the original audit's Port A/B)

Same two ports as the original 14-runner audit: restore a true-rest snapshot of the 4 C2 arrays
(`cp_refractory_timers`, `cp_prev_firing_states`, `cp_neuron_activity_ema`, `cp_neuron_firing_thresholds`) plus,
where relevant, the NMDA-recurrent conductances and the synapse-pulse ring buffer, on every hand-rolled
`_hard_reset`/`_reset_dynamics`. Every port here is either a demonstrated real reduction
(`comprehension_production_organ.py`) or a verified-inert hygiene addition (the other 4) — none changes a
verdict. `comprehension_production_organ.py` additionally gained a `selftest_read_isolation()` /
`--selftest` entry point, verified in both directions against a `git stash` snapshot of the pre-fix file.

**Files changed:** `research/runners/comprehension_production_organ.py`,
`research/runners/sc_orienting_production_organ.py`,
`research/runners/_laneC_source_provenance_opponent_derisk.py`,
`research/runners/_replay_dg_pattern_separation_gate.py`.
**Artifact:** `research/findings/raw/_read_isolation_audit_29/audit_29runners.json` (per-runner detail) +
the `diag_*.py` scripts in the same directory (each number above is reproducible by re-running its script).
**Tests:** `tests/test_comprehension_learned_cues_joint.py`, `tests/test_comprehension_learned_animacy_cue.py`,
`tests/test_comprehension_learned_verbselects_cue.py`, `tests/test_source_provenance_honesty_wirein.py` — 35/35
pass unmodified. **Branch:** `research/readfix-29runner-audit`.
