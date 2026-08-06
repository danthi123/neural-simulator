---
type: finding
status: negative
date: 2026-08-06
lane: laneC
mechanism: source-monitor-coresidency-instrument
runner: research/runners/_laneC_source_monitor_coresidency_gate_v9.py
artifacts:
  - research/findings/raw/source_monitor_v9_generalization/stepping_history_confound_fixed.json
  - research/findings/raw/source_monitor_v9_generalization/fixed_instrument_revalidation.json
  - research/findings/raw/source_monitor_v9_generalization/stepping_history_confound.json
---

# Instrument fix: the source-monitor margin was stepping-history-dependent; under a full per-recall state reset, BOTH v6 and v9 CALIBRATION GO flip to NO-GO (they were artifacts)

## The confound cause: which state carried over

`SourceMonitorCoresidencyGateV6._settle_to_quiescence` guarantees no residual
SPIKES before a read, but does NOT reset the fast Izhikevich sub-threshold state:
membrane `v`, recovery/adaptation `u`, the synaptic conductances (`g_e`, `g_i`,
`g_nmda`, `g_nmda_rise`), the refractory + synaptic-pulse timers, the firing
flags, and the neuron activity EMA. Each `recall` therefore begins from whatever
sub-threshold state the previous recalls left behind. The evaluator reads the
intact (competition-ON) margins `M` at recall calls #1-3 and the
competition-lesion (OFF) margins `L` at calls #8-10, after four intervening
recalls (mixed, unseen, source-path-lesion, acc-lesion). With OU noise off
(`ou_std_current_pA = 0`) the dynamics are fully deterministic, so the two arms —
sampled at different stepping-history depths — differ by a deterministic history
offset that pushed `min(M)` above `min(L)`.

## The fix (instrument only; no mechanism change, no criterion loosened)

Added `SourceMonitorCoresidencyGate.reset_dynamical_state()`: at construction it
snapshots the clean fast-dynamical state, and it restores that snapshot at the
START of every `SourceMonitorCoresidencyGateV6.recall` (before settle). Both arms
are now sampled from an IDENTICAL clean state and differ ONLY in the competition
gate and the driven episode pattern. The snapshot deliberately EXCLUDES the
learned memory (`cp_connections`) and adapted thresholds
(`cp_neuron_firing_thresholds`), so the v7 threshold / v8 scaling / v9 iSTDP
mechanism state survives the reset. v6/v7/v8/v9 all inherit this recall, so one
edit fixes the whole arc; the frozen criterion and thresholds are untouched.

## Proof the artifact is gone (`stepping_history_confound_fixed.json`)

The finding's exact control — a window that changes ZERO weights (Hebbian OFF,
iSTDP OFF, competition ON) inserted before the runner's exact recall order — run
under both instruments. The OLD instrument (reset disabled) reproduces the
recorded confound; the FIXED instrument yields `strict=False` with
`min(M)==min(L)` on every seed. A shipped assertion (`--control`) now FAILS if the
fixed instrument ever manufactures `strict=True`.

| seed | old min M | old min L | old strict | fixed min M | fixed min L | fixed strict |
|---:|---:|---:|:---:|---:|---:|:---:|
| 652 | .1842 | .1775 | yes | .1858 | .1858 | no |
| 653 | .1642 | .1442 | yes | .1608 | .1608 | no |
| 654 | .1867 | .1825 | yes | .1825 | .1825 | no |

The old-instrument rows match the recorded `stepping_history_confound.json`
byte-for-byte, so the fix removes the confound and nothing else.

## Re-validation table (`fixed_instrument_revalidation.json`)

Per version, old recorded verdict -> new verdict under the fixed instrument
(calibration seeds 650/651; development 652/653/654; NumPy, deterministic):

| version | phase | old recorded | fixed instrument | change |
|---|---|---|---|---|
| v6 | calibration | GO | **NO-GO** | **FLIP (was the artifact)** |
| v6 | development | NO-GO | NO-GO | none (already honest) |
| v7 | calibration | characterization | NO-GO | none (floor fails; homeostasis broke competition) |
| v7 | development | NO-GO | NO-GO | none |
| v8 | calibration | characterization | NO-GO/UNDEFINED | none (competition not lesionable) |
| v8 | development | NO-GO | NO-GO | none |
| v9 | calibration | CALIBRATION_PASS (650,651) | **NO-GO** | **FLIP (was the artifact)** |
| v9 | development | NO-GO (seal-corrected) | NO-GO | none (now honest, no manual correction) |

Only two verdicts change, both calibration GO -> NO-GO. NO development NO-GO
flips to GO: every development NO-GO was real. Under the fixed instrument v6 and
v9 calibration fail on the SINGLE component `weakest_source_margin_strictly_improved`
(`min(M)==min(L)` exactly), while all other 19 components still pass.

## What this means: the criterion is unsatisfiable under this protocol

With disjoint episode patterns + silent-by-construction recall the recall-time
rival burden is 0, so the competition (fixed v6 OR plastic v9) provably cannot
change the weakest source's OWN margin: `min(M)==min(L)` EXACTLY. So
`weakest_source_margin_strictly_improved` cannot be satisfied by ANY competition
mechanism here; the recorded v6/v9 calibration GOs only appeared satisfiable
because the intact arm was measured at a shallower stepping-history depth than the
lesion arm. This confirms and GENERALISES the v9 mechanism-level conclusion (the
inhibitory rule is inert; rival burden 0) to the calibration partition.

## Prior verdicts now wrong — flagged for retraction/correction (files NOT edited)

- `research/findings/2026-08-06-source-monitor-coresidency-v6-calibration-GO-learning-off-silent-by-construction.md` ⛔ its `status: positive` GO is VOID — v6 calibration is NO-GO under the fixed instrument. Register in `docs/RETRACTED.md`.
- The v9 finding's calibration characterization (650/651 = CALIBRATION_PASS) is wrong (both are NO-GO), but the v9 finding's overall verdict (development NO-GO) STANDS.

## Corrected next mechanism for the source lane (do NOT start this turn)

The criterion is unsatisfiable by any competition mechanism under the current
protocol, so the next method must attack the PROTOCOL or the weak source's OWN
excitatory gain, not the (inert) inhibition: (1) introduce genuine episode pattern
OVERLAP so a real recall-time rival burden exists that competition + iSTDP can
causally reduce, then re-run the arc under the fixed instrument; or (2) a BCM
sliding-threshold / metaplastic selectivity rule on the episode->source recall
synapses that RAISES the weakest source's own firing gain/selectivity (increase
the weak source, do not equalise rates like v8).

## Provenance

The instrument fix is runner-side only (`reset_dynamical_state` on the base gate;
the reset call in v6's recall; the `--control` diagnostic + assertion on v9). No
`sim/` edit. All runs are NumPy-backend, deterministic across re-runs. Artifacts:
the proof control `research/findings/raw/source_monitor_v9_generalization/stepping_history_confound_fixed.json`
and the re-validation table `research/findings/raw/source_monitor_v9_generalization/fixed_instrument_revalidation.json`,
each with a `.prov.json` sidecar; the recorded confound is
`research/findings/raw/source_monitor_v9_generalization/stepping_history_confound.json`.
