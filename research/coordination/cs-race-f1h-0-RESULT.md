# cs-race-f1h step 0 RESULT (Hermes, 2026-10-27)

Packet: `research/coordination/cs-race-f1h-PACKET.md` (amendments A1 grid {20,40,80}, A2 seeds
41,45-51). Ruling: f1h-queued-1027 Driver (1). Batch: 72/72 artifacts + 72 prov sidecars landed on
main (`5b8228a89` + the final batch this session), revision `3b1ccaf3443cfef619ea19189e6ee15980fcca57`,
backend numpy, device cpu, mode f1g_step0_calibration with f1h flags (trace intrinsic, apical-pulse
120, soma-g, w-im, encode-words, ment-pa 150), grid row 3, W_GIS 30.

## Per-cell items (8/8 seeds 41,45,46,47,48,49,50,51)

| cell | C0 | C1 | C2 | C3 | C4 | C6 | med C5a | med C5b | C1 rate Hz range |
|---|---|---|---|---|---|---|---|---|---|
| sg20_wim20 | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 8/8 | 0.000 | 0.000 | 0.38-1.15 |
| sg20_wim40 | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 8/8 | 0.000 | 0.000 | 0.19-1.15 |
| sg20_wim80 | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 8/8 | 0.000 | 0.000 | 0.38-1.54 |
| sg40_wim20 | 8/8 | 8/8 | 8/8 | 0/8 | 6/8 | 8/8 | 0.588 | 3.741 | 8.08-11.73 |
| sg40_wim40 | 8/8 | 8/8 | 8/8 | 0/8 | 7/8 | 8/8 | -0.790 | 5.819 | 7.69-10.77 |
| sg40_wim80 | 8/8 | 8/8 | 8/8 | 0/8 | 7/8 | 8/8 | 2.721 | 1.848 | 7.88-10.38 |
| sg80_wim20 | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 8/8 | -1.040 | 0.739 | 22.12-27.88 |
| sg80_wim40 | 8/8 | 0/8 | 8/8 | 0/8 | 6/8 | 8/8 | 0.219 | 0.557 | 23.46-27.31 |
| sg80_wim80 | 8/8 | 0/8 | 8/8 | 0/8 | 5/8 | 8/8 | -0.123 | -1.567 | 22.12-26.35 |

## Verdicts

- **P0 (engine support): HOSTABLE** - `research/coordination/cs-race-f1h-0-PROGRESS.md`
  (file:line evidence; flags allocate `cp_v_apical`, drive array `cp_bdsp_apical_drive`,
  restriction exact on `self.ment` indices). Not STOP-ENGINE.
- **P1 (clause-a oddity): CIRCUIT, not instrument, at 150 pA with the f1h flags - and the
  'clause-a never delivers' observation from f1g does NOT reproduce here.** The field
  (`total_exc_current_pa_per_k`, compute `research/runners/cs_race_f1.py:706-711`, sampled on
  absolute [100,200) ms) reads >0 in clause A on 9 of 9 cells (first-slot range
  0.00..119.93 pA), so the instrument is live
  in clause A; where it reads 0.0 the circuit delivered no excitatory spikes into slot-0 engram
  cells in that window (encode schedule :380-405, formula :639-642). The raw per-ms rasters the
  packet asks for are not in the artifacts; if a stricter settle is wanted it needs the one
  seed-41 instrumented run the packet allows.
- **STOP-ENGINE: NO** (P0 hostable).
- **STOP-CAL: YES** - selection pool is EMPTY: no cell passes C0-C4 and C6 on 8/8. C3 is 0/8 on
  all 9 cells; C1 passes only the soma-g 40 row (sg20 under-band 0.19-1.54 Hz, sg80 over-band
  22.12-27.88 Hz); C4 drops to 7/8 (sg40 wim40/80) and 5-6/8 (sg80 wim40/80).
- **STOP-SMOKE: YES** - global median C5 clause_a = 0.000 pA, clause_b = 0.000 pA, both
  < the 2 pA threshold; even the C1-passing sg40 rows sit at or below the +4/+6 pA flip scale on at
  least one clause (sg40_wim20: 0.588/3.741).
- **STOP-ID: NOT JUDGED** - no f1h flags-off identity artifact exists in this batch (identity
  records exist for f1d/f1g only). Judgment is deferred to the identity run; nothing here asserts
  it.
- **C6: 8/8 PASS on all 9 cells** (0 UNDEFINED): the pulsed slot's v_apical holds above v_hold
  (-35 mV) across the 650 ms window and every unpulsed slot stays below it, at every grid point.
- **C3 branch (packet line 70-71): FIRES AS PREDECLARED.** C6 passes but C3 fails everywhere:
  `older_plateau_ended` is False on 0/216 pairs while `newer_plateau_held` is True
  on all of them - somatic inhibition never ends an older word's held apical plateau, exactly the
  predicted result. Per the packet, the next ruling designs dendrite-targeting (SOM-like)
  inhibition rather than retuning this grid; a packet for that path is already merged on main
  (`research/coordination/cs-race-f1i-PACKET.md` via ab28cc4b7/8e13b44cc).
- **Chosen cell: NONE** (STOP-CAL precedes selection).

<!--derived-->
All medians/counts computed over the 72 committed artifacts
(`research/findings/raw/_cs_race/f1h/step0/`), script `a_score_0752.py` this session.

**Then STOP** (packet line 73; ruling f1h-queued-1027 Driver (1)).
