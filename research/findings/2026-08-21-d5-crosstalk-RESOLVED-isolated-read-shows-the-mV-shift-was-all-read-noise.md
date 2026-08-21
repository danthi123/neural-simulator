---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-crosstalk-isolated-read
lane: memory
seeds: [42, 43, 100]
seed-waiver: a MEASUREMENT/decision experiment through the cupy D5 organ — the load-bearing evidence is the
  deterministic-read noise floor (0.0) making the |dB| shift decidable, and the |dB|_iso magnitudes, not a
  stochastic effect size. 3 crosstalk seeds × the set-point's disjoint + overlapping builds (6 builds).
instrument: research/runners/_d5_crosstalk_isolated_read_derisk.py — re-runs the neighbour-crosstalk measurement
  (form two memories, read neighbour B, consolidate A, read B again → |dB|) using the DETERMINISTIC snapshot-isolated
  read instead of the period-2 live read, on the te=40 production encode, with tools.verdict.Verdict.
runner: research/runners/_d5_crosstalk_isolated_read_derisk.py
artifacts:
  - research/findings/raw/_d5_crosstalk_iso/confirm_store40_consol40_3seed.json
---
# D5 memory crosstalk (#73) is RESOLVED — the "one memory shifts another" was ENTIRELY read noise; the deterministic isolated read shows ~0 crosstalk (GO)

Artifact: research/findings/raw/_d5_crosstalk_iso/confirm_store40_consol40_3seed.json (verdict GO, cupy, seeds
42/43/100). Every number below is read from that artifact.
<!--derived-->

**One line.** The long-open D5 memory-separator crosstalk question (#73) — "does consolidating memory A shift a
NEIGHBOUR memory B's surfaced recall strength?" — was UNDECIDABLE because the surfaced read was a period-2 limit cycle
(noise std 4.98 mV) that buried the effect. Tonight's snapshot-isolated read (deterministic, noise std 0.0) makes it
DECIDABLE, and the answer is: **there is essentially NO crosstalk.** Consolidating A moves a DISJOINT neighbour's
isolated read by **EXACTLY 0** (the read is weight-local), and even on OVERLAPPING builds the true residual is
**0.0016 mV** (negligible). The old apparent "1.79–7.478 mV crosstalk" was ALL the read noise. So the D5 learn-through-
use flip's crosstalk blocker is CLOSED on the read side; only the separate rise-to-6/6 read-window residual remains.

## The measurement (cupy, te=40 production encode, the deterministic isolated read)
<!--derived-->
- **Decidability upgrade:** the isolated read's repeated-read noise floor is `max_iso_noise_std = 0.0` over 6 builds
  (NOISE_TOL 1e-4) vs the live read's `max_live_noise_std = 4.977 mV` (the period-2 cycle). |separation| 4.977 > 0 —
  the isolated read removes the noise that made the question undecidable.
- **Disjoint / byte-identical-B builds (3):** consolidating A moves B's isolated read by `|dB|_iso == 0 EXACTLY`
  (`max_byteid_shift_iso = 0.0`, shifts [0.0, 0.0, 0.0]). ⇒ the surfaced read is WEIGHT-LOCAL: a neighbour whose
  read-path weights are untouched does NOT move when A is consolidated. This is the crosstalk claim, resolved by
  construction with a deterministic weight-local read.
- **Overlapping builds (3):** where B's read-path weights DO change (shared cells), the residual is quantifiable and
  TINY — `max_overlap_shift_iso = 0.0016 mV` (shifts [0.0016, 0.0, 0.0]). The true weight-mediated crosstalk, when two
  memories genuinely share cells, is negligible — not the 1.79–7.478 mV previously attributed to it (that was noise).

## What this closes + the remaining residual
This RESOLVES #73's crosstalk on the READ side: the D5 learn-through-use default-on flip is no longer blocked by a
neighbour-crosstalk no-regression concern (the effect is ~0 with the deterministic read). The REMAINING blocker before
the flip is the SEPARATE rise-to-6/6 residual named in [[2026-08-21-d5-stabilized-read-NEGATIVE]] — a saturating-tail
non-monotonicity in the weight→read curve (a read-window / operating-point choice, not read noise and not a substrate
wall): read the rise over the pre-saturation window, or use the bounded `soft` read. NO `sim/` edit (a formation-time
read probe composing the isolated-read helper + the set-point's crosstalk measurement). Parent-finalized from the
artifact after the build agent completed the run but stalled before writing the finding.
