# PACKET cs-race-f1h (Claude, 2026-10-24): an intrinsic word trace, a held dendritic plateau in the mention slot

host-decision: scaffold
decides: which engram each heard cue or mention word connects to (the probe wires cue->E_k and mention->E_k at write time, the cs-race declared residual), which words an encode turn presents to the mention layer (the fact's three words in heard order), and that a heard word's sensory volley reaches both the basal (somatic) and the apical arbor of its word-form slot
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement); docs/drafts/cs-race/PACKET.md (the declared write-time residual)
replaced-by: write-time Hebbian binding of heard word forms to the engram being written, on the composer's spiking store write path, with heard word forms arriving through the hearing path and its own laminar targeting
cap: 2 builds

Inputs: `cs-race-f1h-INPUTS.md` (research/cs-race), ruling `cs-race-f1g-0c-1022` (the recurrent trace is banked: silent at
W_MM <= 15, runaway at W_MM >= 30), and the engine's validated held plateau. Probe-local, CPU, numpy, pool, memcap. No `sim/`
edit. Gate seeds are not run. Hermes runs STEP 0 only, then posts the RESULT and STOPs.

## Why this mechanism
The real system holds a heard item with an intrinsic plateau that needs no recurrent loop. The f1f sources give persistent
firing of 4.5-6.5 Hz for tens of seconds after a brief trigger. The engine's one validated held state that also reaches the
soma is the gap4 configuration:
- BDSP apical with `bdsp_apical_bistable` (self-regenerating sustain plus a KIR down-state);
- with `bdsp_apical_couples_soma`.

In the gap4 6/6-seed GO (2026-07-18), a brief apical drive latches `v_apical` near -24 mV for seconds, while the transient
control decays. A bistable plateau has no recurrent runaway, which was what killed the f1g trace: it holds or it does not.
So the next carrier is this held plateau in the mention slot's cells.

## Method (new flags; every default reproduces today's circuit exactly)
- **Mention slot cells.** The BDSP two-compartment apical with `bdsp_apical_bistable` on and `bdsp_apical_couples_soma` on.
  The self-regen strength and KIR take the gap4 GO values, read from that finding's record and frozen, never tuned here.
  `W_MM` is 0: no recurrence at all.
- **Heard word.** The somatic current stays at 150 pA (ruling `-0b-1022`, the engagement-chosen drive). A brief apical drive
  pulse of the same duration as the word (50 ms for an encode word, 150 ms for a mention) goes to the same slot. Its
  amplitude is the gap4 GO pulse value, frozen; this is the declared world-side laminar targeting.
- **Overwrite.** The f1g shared inhibitory pool (slot -> pool `W_MI` 20, pool -> every slot `W_IM`). Whether somatic
  inhibition ends an older word's apical plateau is MEASURED (item C3), not assumed.
- `W_MENT` stays 12. The controls (positive and decisive) present no words to the mention layer.
- **Flags:** `--trace intrinsic`, `--soma-g`, `--w-im`, `--apical-pulse` (default off).

## Step 0 (Hermes, now): seeds 41-48 only, no race, W_GIS 30, row 3
- **P0, engine support (read-only, first).** Show with file:line that `cs_race_f1`'s bridge can host the BDSP apical on the
  mention slot cells only: which flags allocate it, whether it can be restricted to those indices, and which array carries
  an apical drive. If the BDSP apical cannot run on this probe's cells, that is STOP-ENGINE; post what blocks it.
- **P1, the clause-a oddity (f1h-INPUTS section 3), read-only or one seed-41 run.** `clause_a_per_k[0]` reads exactly 0.0
  at 150 pA while clause-b's identical encode schedule delivers current. Settle it from the per-ms rasters for [100, 200) ms
  in both clauses: instrument or circuit, with file:line. An instrument defect is fixed before the grid, and the identity
  check is re-run.
- **Grid (frozen):** `--soma-g` in {0.002, 0.004, 0.008} (the validated default 0.004 in the middle) x `W_IM` in {20, 40, 80}.
- **Items** (as in f1g, on 8/8 seeds):
  - C0: the pulse fires each cell.
  - C1: 2-20 Hz in the 650 ms after the pulse.
  - C2: unpulsed slots below 1 Hz, and no slot above 1 Hz in 2 s without input.
  - C3: overwrite, both orders and the three pairs.
  - C4: no engram burst or latch from the trace alone.
  - C5: delivered pA, held engram minus other, at [100, 200) ms after the would-be cue onset, both clause protocols.
  - C6 (new, engagement): the pulsed slot's `v_apical` stays above `v_hold` across the 650 ms window, and every unpulsed
    slot's stays below it. A cell whose plateau never latched is UNDEFINED, never a FAIL.
- **Selection:** the smallest `--soma-g`, then the smallest `W_IM`, among cells passing C0-C4 and C6 on 8/8.
- **STOP rules:** STOP-ENGINE (P0), STOP-CAL (no cell passes), STOP-SMOKE (median C5 below 2 pA in either clause),
  STOP-ID (the flag-off identity check fails).
- **Branch already decided:** if C6 passes but C3 fails (somatic inhibition does not end an older plateau), post it. The
  next ruling then designs dendrite-targeting inhibition (the SOM-like path) rather than retuning this grid.
- **Post** `[cs-race-f1h-0] RESULT` (every item per cell, the P0 and P1 answers, the chosen cell, C5 against the +4/+6 pA
  flip scale, the STOP verdicts, branch @ sha). Then STOP.

## Step 1 (preregistered; runs only after a Claude RULING on step 0)
- **Arms:** TRACE (the chosen cell), LESION (apical pulse off, so no plateau, with the somatic 150 pA kept), and NO-ENC.
- **Bars:** the f1g step-1 ownership bars, unchanged. Clauses (a) and (b) each pass at 8 or more of 10 on at least 7/8 DEV
  1-8 AND at least 14/16 CONFIRM 25-40, with the positive control, clause (c) and clause (g) intact on those seeds. LESION
  passes (a) and (b) on at most 2/8 DEV.
- **Engagement:** per-draw mention spike counts and the plateau-latched flag; a draw with neither is UNDEFINED.
