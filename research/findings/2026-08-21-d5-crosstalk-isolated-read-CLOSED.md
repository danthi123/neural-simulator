---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-memory-separator-crosstalk-decidable-isolated-read
lane: memory
integration_faculty: d5-live-consolidation
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_crosstalk_isolated_read_derisk.py — re-runs Layer B of
  _d5_pattern_separation_setpoint_derisk (form A(slot0)+B(slot1) on the shared CA3 dendritic-dAP readout, read neighbour
  B, CONSOLIDATE A = re-form at the production encode consol_te, read B again -> |dB|) with the DETERMINISTIC
  snapshot-isolated read (complete reset to the clean-rest snapshot + inject the current weights) INSTEAD of the period-2
  live `_apical_depth_read`. Classifies each (seed x separation OFF/ON) build by whether B's within-weights are
  byte-identical after consolidating A, and reports the isolated AND live repeated-read noise floors. Reuses Layer B's
  formation/assemblies/graded-read verbatim; only the READ is swapped. NO sim/ edit.
runner: research/runners/_d5_crosstalk_isolated_read_derisk.py
external: NO-EXTERNAL-NEEDED — an in-repo characterization of the D5 crosstalk with a deterministic readout; the read is
  the SAME cp_v_apical the production recall reads. No literature question.
artifacts:
  - research/findings/raw/_d5_crosstalk_iso/summary_store20_consol40_6seed.json
  - research/findings/raw/_d5_crosstalk_iso/confirm_store40_consol40_3seed.json
  - research/findings/raw/_d5_crosstalk_iso/smoke_cupy.json
---
# D5 memory-separator crosstalk (#73) is decidably CLOSED: the deterministic snapshot-isolated read gives |ΔB|=0 EXACTLY on every byte-identical (disjoint) neighbour (5/5) and ≤0.02 mV on overlapping neighbours (7/7), while the LIVE read carried up to 9.69 mV of period-2 noise on the SAME weight-untouched neighbours — so the prior multi-mV "crosstalk" was ENTIRELY read noise; consolidation is genuine (A rises everywhere)

Artifact: `research/findings/raw/_d5_crosstalk_iso/summary_store20_consol40_6seed.json` (6-seed, cupy, verdict GO) +
`research/findings/raw/_d5_crosstalk_iso/confirm_store40_consol40_3seed.json` (exact prior-regime confirm). Every number
below is read from those artifacts.
<!--derived-->

**One line.** Board #73's long-open question — "does consolidating memory A move a NEIGHBOR memory B's surfaced recall
strength?" — was UNDECIDABLE: the production surfaced read is a period-2 limit cycle (the stabilized-read finding,
`2026-08-21-d5-stabilized-read-NEGATIVE.md`) whose few-to-many-mV read noise BURIED the effect, so the prior te=40
confirm saw a byte-identical (weight-untouched) neighbour "move" 1.79 mV and could not tell leak from noise. This
de-risk re-runs the crosstalk measurement with the DETERMINISTIC snapshot-isolated read (a complete reset to the
clean-rest baseline + inject the current weights — the biology's own return-to-rest between recalls), which is a pure
function of B's stored weights. The answer is now decidable and CLOSED: consolidating A does NOT move a disjoint
neighbour (|ΔB|_iso = 0 EXACTLY on 5/5 byte-identical-B builds), and the TRUE weight-mediated crosstalk on overlapping
builds is ≤0.017 mV — three orders of magnitude below the live read's period-2 noise (up to 9.69 mV). The prior "crosstalk"
was read noise, not an A→B leak.

## The isolated read is DETERMINISTIC (repeated-read std = 0 on 12/12 builds) — the enabling instrument
<!--derived-->
Across all 12 builds (6 seeds × {OFF = unmodified emergent assemblies, ON = winner-fatigue set-point}), the isolated
read's repeated-read noise floor (K=8 reads of the SAME neighbour B, no consolidation between them) is `iso_noise_std =
0.0` on every build (`summary_store20_consol40_6seed.json`, `metrics.max_iso_noise_std = 0.0`). The SAME neighbour read
through the LIVE bridge (only `hard_silence`+`_reset_apical_latch` between reads) swings a period-2 cycle with
`live_noise_std` ranging 0.535–9.69 mV across builds (`metrics.max_live_noise_std = 9.68956`). The control confirms the
upgrade: `treatment` (max live noise) 9.68956 vs `control` (max iso noise) 0.0. A read that is a stationary function of
the weights makes "did B move?" decidable; a read that alternates does not.

## On byte-identical (disjoint) neighbours, |ΔB|_iso = 0 EXACTLY — consolidating A does NOT touch a non-overlapping neighbour
<!--derived-->
5 of the 12 builds have B's within-weights byte-identical after consolidating A (the ON winner-fatigue set-point drives
DISJOINT assemblies: `shared_within_conn = 0` on s43/s44/s100/s101/s102 ON). On every one, the deterministic read gives
`|dB|_iso = 0.0` exactly (`metrics.byteid_shifts_iso = [0.0, 0.0, 0.0, 0.0, 0.0]`, `max_byteid_shift_iso = 0.0`). This is
the clean confirmation that the read is genuinely WEIGHT-LOCAL: a neighbour whose read-path weights are provably
untouched reads identically. The contrast is decisive on the SAME builds — e.g. s102 ON: `|dB|_iso = 0.0` while
`live_noise_std = 9.20159` mV; s43 ON: `|dB|_iso = 0.0` while `live_noise_std = 5.112329` mV (this is exactly the build
the prior te=40 confirm reported as a 1.79 mV neighbour "shift" on a byte-identical B — that shift was read noise).

## On overlapping neighbours, |ΔB|_iso reveals the TRUE weight-mediated crosstalk — and it is ≤0.02 mV (negligible)
<!--derived-->
7 of the 12 builds are genuinely overlapping (B's within-weights CHANGE when consolidating A writes a shared connection:
`B_within_weights_byte_identical = False`). With the deterministic read, their |ΔB|_iso is the true weight-mediated
crosstalk, no longer buried in noise: `metrics.overlap_shifts_iso = [0.017048, 0.01161, 0.005768, 0.0, 0.0, 0.0, 0.0]`
— max 0.017048 mV (s102 OFF, `shared_within_conn = 2`), next 0.01160975 (s101 OFF), 0.00576755 (s43 OFF,
`shared_within_conn = 6`), the other four exactly 0. So even where A's consolidation DOES overwrite a shared read-path
connection, the surfaced neighbour read moves at most 0.017 mV. Against the live read's 0.535–9.69 mV period-2 noise on
the same builds, the structural crosstalk is ~3 orders of magnitude smaller — it was never observable through the live
read, and it is negligible on its own terms.

## The consolidation is GENUINE — A's own strength rises on every build (this is not a no-op)
<!--derived-->
Unlike the prior te=40 confirm (store=consol=40, where A saturated and its rise read −0.089/−0.236/−0.110), this run
stores A at te=20 (rise headroom) and consolidates to the production encode consol_te=40, so A's within-assembly weight
genuinely grows and its surfaced read rises on all 12 builds: `A_rise_iso` ranges +0.346669 (s100 OFF) to +3.169863
(s101 OFF); e.g. s100 ON A `27.053622 -> 30.184495` (+3.130872), with `w_A` `68.283 -> 83.0398`. So the neighbour is
being read while A is genuinely being consolidated — the |ΔB|_iso = 0 result is "a real consolidation of A leaves a
disjoint B untouched", not "nothing happened".

## CONFIRM at the EXACT prior-regime operating point (store=consol=40, the te=40 production encode) — same verdict, and it directly retires the prior UNDECIDABLE reading
<!--derived-->
Re-run at the EXACT operating point of the prior UNDECIDABLE verdict — store=consol=40, the saturated te=40 production
encode — on that confirm's own seeds (`confirm_store40_consol40_3seed.json`, cupy, verdict GO, 6 builds). The picture is
identical: the isolated read is deterministic (`max_iso_noise_std = 0.0`), the 3 byte-identical-B builds give
`byteid_shifts_iso = [0.0, 0.0, 0.0]` (exactly 0), and the 3 overlapping builds give `overlap_shifts_iso = [0.001594,
0.0, 0.0]` (max 0.001594 mV). This directly refutes the prior read-noise verdict on the SAME builds: the prior te=40
confirm (`research/findings/raw/_d5_separation/cupy_confirm_te40.json`) reported the byte-identical neighbours moving
1.79 mV (s43 ON), 0.57 mV (s100 ON) and 1.84 mV (s100 OFF) with the LIVE read — here each is exactly 0, and the live
noise floor I measure on those same builds is `live_noise_std` 2.586934 (s43 ON), 4.97661 (s100 ON), 4.910091 (s100 OFF)
mV. The prior 5.57 mV neighbour "shift" on s42 OFF collapses to |dB|_iso = 0.00159395 (a genuine but negligible
weight-mediated change on 1 shared connection). And `A_rise_iso = 0.0` on all 6 builds confirms te=40 SATURATES A (the
exact read-ceiling limit the prior finding named) — which is why the primary run stores A at te=20 for a GENUINE
consolidation; the crosstalk verdict is CLOSED at both operating points regardless.

## Verdict: CLOSED — the crosstalk question is decidable and the read-side crosstalk is resolved
<!--derived-->
The task GO is "crosstalk decidably CLOSED (|ΔB|_iso ~0 on disjoint, and any overlapping residual quantified)". All
preconditions hold (`summary_store20_consol40_6seed.json`, verdict GO): the isolated read is deterministic (decidable);
byte-identical-B neighbours give exactly 0 (the read is weight-local); overlapping neighbours are quantified at ≤0.017
mV; and the live read carries the period-2 noise the isolation removes. What is now KNOWN, and was the point:
- **Consolidating A does NOT move a disjoint neighbour** (|ΔB|_iso = 0 exactly, 5/5). The pattern-separation set-point,
  where it reaches full disjointness, gives ZERO crosstalk by construction.
- **The prior multi-mV "crosstalk" (te=18 ~7.5 mV, te=40 ~1.8 mV) was ENTIRELY period-2 read noise** — the true
  weight-mediated effect on overlapping neighbours is ≤0.017 mV.
- **The instrument is part of the emulation**: the same refutation that read as "crosstalk remains" collapses once the
  read is made a stationary function of the weights. Isolation (complete reset to clean rest) is the missing companion
  process the incomplete reset had replaced with a constant.

## What this does and does NOT unblock (scope honesty)
<!--derived-->
This CLOSES the READ-SIDE crosstalk blocker for the D5 default-on flip: the memory-separator crosstalk that the
`_d5_graded_flip_soak` no-regression violation flagged is a read-noise artifact, not an A→B weight leak; on disjoint
membership it is exactly 0, and even on overlapping membership it is ≤0.017 mV. It does NOT flip on_by_default. The
REMAINING blocker before default-on is the SEPARATE, deterministic saturating-tail read-window residual — the
conversation-visibility rise-to-6/6 that the stabilized-read finding disentangled from this noise (a plateau-depth
read saturating non-monotonically near the top, NOT read noise). Two secondary knobs remain, neither a substrate wall:
the winner-fatigue set-point reaches full disjointness (`shared_within_conn = 0`) on ~3/6 seeds at sep_bias=500 (a
bias/operating-point sweep for 6/6), and the rise-to-6/6 read-window. NO `sim/` edit; ADDITIVE, default-off; the binary
moat gate is unchanged (the surfaced strength is a faithful spiking read, not a phenomenal claim).
