# cs-race f1i STEP 0 SCORE (Claude scorer, 2026-10-28)

Packet: `research/coordination/cs-race-f1i-PACKET.md` (Step 0 items C0-C8, Selection, STOP rules, Branches; ruling R1).
Base cell: RULING f1h-0-1027 (soma-g 40, W_IM 40). Driver post: `[f1i-0] RESULT` and `[f1i-score] ASK` (runbook, 2026-10-28 09:57).
Data: the 155 step-0 artifacts in `research/findings/raw/_cs_race/f1i/step0/` (landed c8c9a2595), every one run at the pin
ee349f8b11e19a69456b2addc06218c4c0f0bbbf, numpy, CPU, sidecar clean, argv equal to its pool line.
Scorer: `research/runners/cs_race_f1i_score.py` (tests `tests/test_cs_race_f1i_score.py`, 24 pass), run as a module at b5ee567a8:

    bash tools/memcap.sh 2 -- env SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.cs_race_f1i_score --out research/findings/raw/_cs_race/f1i/score/f1i0_score.json

Output: `research/findings/raw/_cs_race/f1i/score/f1i0_score.json` and its `.prov.json` (git_sha b5ee567a8, git_dirty false).
The scorer re-derives C7 and C8 from the raw the runner keeps (per-ms `v_apical` min / max and `g_ap` for the pulsed words, per-ms
SOM counts, per-spike SOM cell events, the lesion arm) and re-scores C0-C6 from the same artifacts. Its readings of each item are in
its docstring. It adds no bar and loosens none.

## Verdict

- **No cell is selected.** The driver's provisional `sa0p125_ms215` passes C0-C4, C6 and C8 on 8/8, but its C7 reads 2 PASS / 5 FAIL /
  1 UNDEFINED. No other cell passes C1, C3 or C6 on 8/8.
- **STOP-CAL: YES.** No cell passes C0-C4 and C6-C8 on 8/8.
- **STOP-ATTRIB: YES.** The only C3-passing cell (`sa0p125_ms215`, C3 8/8) is UNATTRIBUTED. Its C7(c) is a defined pass on 11 of 23
  defined pairs (12 defined fails, 1 UNDEFINED pair); C7(d) passes on 23 of 23 defined pairs.
- **STOP-ENGINE: NO.** The E build's files are at the pin. `tests/test_apical_gaba_routing.py` passes (28) on this branch, where `sim/`,
  that test and the runner are unchanged since the pin. Every `--som` record reads back `g_ap` allocated on the mention cells only.
- **STOP-ID: NO.** Identity (i): seeds 41 and 45 are hash-equal minus `wall_s`, and byte-equal, to the landed f1h `cal_sg40_wim40` records.
  Identity (ii): 72 of 72 lesion records read the `cs_som` gate 0.0 at measurement and `g_ap` 0 at every recorded ms, on every trial,
  while the SOM cells still fire. Per-type STP read-back (S2) matches on all 153 `--som` records: slot -> SOM U 0.049, F 1797 ms, D 399 ms
  on type 1, engram recurrence on type 0. No reset read-back problem.
- **STOP-SMOKE: not judged** (no selected cell). At the C3 cell the C5 medians are 13.76 pA (clause a) and 15.52 pA (clause b), above
  2 pA and above the +4/+6 pA flip scale. The median over all 72 TRACE records is 0.0 / 0.0, which comes from the cells that fail C1/C6.
- **Branches B1-B4: none fires by its text.** B1 needs C7(a) passing and C7(b) failing on every cell. B2 needs C8 failing on every cell.
  B3 needs C7(c) passing and C7(a) failing on every cell. B4 needs a cell with C3 and C7 passing.
- **Decision-tree branch:** the packet says STOP-CAL and STOP-ATTRIB post and STOP. No pre-decided branch fires, so the next ruling has to
  choose the next method. Step 1 does not run: it needs a chosen cell.

## Per-cell items (P/F/U over seeds 41, 45-51; 8/8 = 8/0/0)

| cell | C0 | C1 | C2 | C3 | C4 | C6 | C7 | C8 | med C5a pA | med C5b pA | selectable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sa0p125_ms215 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 | 2/5/1 | 8/0/0 | 13.7575 | 15.5205 | no |
| sa0p125_ms304 | 8/0/0 | 4/4/0 | 8/0/0 | 6/2/0 | 8/0/0 | 4/4/0 | 1/5/2 | 8/0/0 | 15.1547 | 15.5144 | no |
| sa0p125_ms429 | 8/0/0 | 3/5/0 | 8/0/0 | 0/8/0 | 8/0/0 | 1/7/0 | 0/5/3 | 4/4/0 | 0.0 | 15.5822 | no |
| sa0p25_ms215 | 8/0/0 | 4/4/0 | 8/0/0 | 2/6/0 | 8/0/0 | 4/4/0 | 0/4/4 | 8/0/0 | 0.0 | 7.1358 | no |
| sa0p25_ms304 | 8/0/0 | 3/5/0 | 8/0/0 | 2/6/0 | 8/0/0 | 3/5/0 | 0/4/4 | 8/0/0 | 0.0 | 6.9995 | no |
| sa0p25_ms429 | 8/0/0 | 1/7/0 | 8/0/0 | 0/8/0 | 8/0/0 | 0/8/0 | 0/4/4 | 7/1/0 | 0.0 | 0.0 | no |
| sa0p5_ms215 | 8/0/0 | 0/8/0 | 8/0/0 | 0/8/0 | 8/0/0 | 0/8/0 | 0/1/7 | 8/0/0 | 0.0 | 0.0 | no |
| sa0p5_ms304 | 8/0/0 | 1/7/0 | 8/0/0 | 0/8/0 | 8/0/0 | 1/7/0 | 0/1/7 | 8/0/0 | 0.0 | 0.0 | no |
| sa0p5_ms429 | 8/0/0 | 0/8/0 | 8/0/0 | 0/8/0 | 8/0/0 | 0/8/0 | 0/0/8 | 8/0/0 | 0.0 | 0.0 | no |

Totals over the 72 TRACE records: C1 24, C3 18, C6 21 PASS / 51 FAIL. These match the driver's `[f1i-0]` counts, which were scored
independently. C4 is 8/8 on every cell. f1h's base cell read C4 7/8 (RULING f1h-0-1027 asked for the two side by side).

## C7 sub-items and the lesion arm

The (0)-(d) columns count defined passes over 24 pairs. A pair is defined when its older plateau latched before Y, i.e. C7(0) holds.
Latencies are in ms after Y_on, over the defined pairs. The lesion columns come from the `--som-lesion` record of the same cell and seed.

| cell | (0) | (a) | (b) | (c) | (d) | latency ms | lesion rate-C3 pass | lesion older held | lesion C6 | C1 Hz range | C8 Hz max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sa0p125_ms215 | 23 | 23 | 23 | 11 | 23 | 57-137 (med 72) | 0/24 | 24/24 | 8/8 | 7.50-11.73 | 0.00 |
| sa0p125_ms304 | 19 | 19 | 18 | 9 | 19 | 29-63 (med 56) | 0/24 | 24/24 | 8/8 | 0.38-11.73 | 0.83 |
| sa0p125_ms429 | 13 | 13 | 6 | 2 | 13 | 8-51 (med 41) | 0/24 | 24/24 | 8/8 | 0.19-11.15 | 2.29 |
| sa0p25_ms215 | 9 | 9 | 8 | 0 | 9 | 54-88 (med 64) | 0/24 | 24/24 | 8/8 | 0.19-11.73 | 0.00 |
| sa0p25_ms304 | 9 | 9 | 6 | 0 | 9 | 44-56 (med 52) | 0/24 | 24/24 | 8/8 | 0.00-11.54 | 0.00 |
| sa0p25_ms429 | 6 | 6 | 0 | 0 | 6 | 19-47 (med 26.5) | 0/24 | 24/24 | 8/8 | 0.00-5.77 | 1.04 |
| sa0p5_ms215 | 1 | 1 | 1 | 0 | 1 | 58 | 0/24 | 24/24 | 8/8 | 0.00-0.77 | 0.00 |
| sa0p5_ms304 | 2 | 2 | 0 | 0 | 2 | 52-54 | 0/24 | 24/24 | 8/8 | 0.00-8.27 | 0.00 |
| sa0p5_ms429 | 0 | 0 | 0 | 0 | 0 | none defined | 0/24 | 24/24 | 8/8 | 0.00-0.58 | 0.00 |

The lesion's share of the overwrite on the defined pairs (`tools.lab.attributable_to`, reported, not a bar) is 1.0 at every cell that has
a defined pair. At `sa0p125_ms215` the older plateau ended on 23 of 23 pairs with the SOM path and on 0 of 23 with it lesioned. Rate-C3
passed on 23 of 23 pairs with the path and on 0 of 23 lesioned. Under the lesion the somatic pool (W_IM) never silenced the older soma
while its plateau held: rate-C3 passed on 0 of 216 lesion pairs.

## C7 at sa0p125_ms215, per pair

| seed | pair | C7 | (0) older min [150,300) mV | termination ms (latency) | (b) newer min [450,950) mV | SOM spikes per cell [300,600) | older g_ap: first rise / peak ms (value) | (c) | lesion: older min [600,950) mV / rate-C3 | (d) |
|---|---|---|---|---|---|---|---|---|---|---|
| 41 | vole>frog | PASS | -27.0 | 361 (61) | -27.1 | 2-3 | 346 / 356 (0.614) | yes | -24.2 / fail | yes |
| 41 | frog>vole | PASS | -29.6 | 361 (61) | -25.2 | 2-3 | 349 / 356 (0.641) | yes | -24.2 / fail | yes |
| 41 | lynx>grass | PASS | -28.6 | 357 (57) | -24.2 | 2-3 | 346 / 352 (0.659) | yes | -24.2 / fail | yes |
| 45 | vole>frog | FAIL | -25.1 | 359 (59) | -24.2 | 2-2 | 348 / 360 (0.573) | no | -24.2 / fail | yes |
| 45 | frog>vole | PASS | -24.4 | 368 (68) | -24.3 | 2-2 | 356 / 367 (0.542) | yes | -24.2 / fail | yes |
| 45 | lynx>grass | FAIL | -29.0 | 367 (67) | -25.3 | 2-2 | 356 / 367 (0.612) | no | -24.2 / fail | yes |
| 46 | vole>frog | FAIL | -25.9 | 374 (74) | -24.3 | 1-2 | 358 / 428 (0.515) | no | -24.2 / fail | yes |
| 46 | frog>vole | FAIL | -24.3 | 376 (76) | -24.2 | 1-2 | 358 / 427 (0.628) | no | -24.2 / fail | yes |
| 46 | lynx>grass | FAIL | -24.3 | 388 (88) | -24.3 | 1-2 | 358 / 433 (0.612) | no | -24.2 / fail | yes |
| 47 | vole>frog | PASS | -32.5 | 372 (72) | -25.9 | 2-2 | 355 / 370 (0.706) | yes | -24.2 / fail | yes |
| 47 | frog>vole | PASS | -30.3 | 370 (70) | -26.3 | 2-2 | 352 / 368 (0.674) | yes | -24.2 / fail | yes |
| 47 | lynx>grass | PASS | -29.8 | 371 (71) | -25.7 | 2-2 | 356 / 370 (0.676) | yes | -24.2 / fail | yes |
| 48 | vole>frog | PASS | -28.1 | 358 (58) | -24.9 | 2-5 | 345 / 357 (0.718) | yes | -24.2 / fail | yes |
| 48 | frog>vole | PASS | -27.1 | 371 (71) | -28.7 | 1-5 | 354 / 369 (0.574) | yes | -24.2 / fail | yes |
| 48 | lynx>grass | FAIL | -30.4 | 374 (74) | -30.0 | 1-5 | 348 / 376 (0.456) | no | -24.2 / fail | yes |
| 49 | vole>frog | FAIL | -25.4 | 437 (137) | -24.6 | 1-2 | 379 / 441 (0.544) | no | -24.2 / fail | yes |
| 49 | frog>vole | FAIL | -25.8 | 403 (103) | -24.6 | 1-2 | 364 / 432 (0.566) | no | -24.2 / fail | yes |
| 49 | lynx>grass | FAIL | -24.5 | 372 (72) | -24.2 | 1-2 | 358 / 429 (0.597) | no | -24.2 / fail | yes |
| 50 | vole>frog | PASS | -29.0 | 373 (73) | -29.2 | 2-2 | 356 / 371 (0.570) | yes | -24.2 / fail | yes |
| 50 | frog>vole | UNDEFINED | -77.3 | (not latched) | -29.2 | 2-2 | - | - | -24.2 / fail | - |
| 50 | lynx>grass | PASS | -26.5 | 365 (65) | -24.5 | 2-2 | 350 / 363 (0.576) | yes | -24.2 / fail | yes |
| 51 | vole>frog | FAIL | -25.8 | 395 (95) | -24.4 | 1-2 | 357 / 421 (0.413) | no | -24.2 / fail | yes |
| 51 | frog>vole | FAIL | -25.9 | 382 (82) | -24.6 | 1-2 | 360 / 433 (0.376) | no | -24.2 / fail | yes |
| 51 | lynx>grass | FAIL | -24.9 | 373 (73) | -26.7 | 1-2 | 357 / 379 (0.398) | no | -24.2 / fail | yes |

"First rise" is a diagnostic, not a bar: the first ms after Y_on at which `g_ap` increases, i.e. a routed SOM spike arriving. The
`f1i0_score.json` field is `g_ap_onset_ms`. On the 23 defined pairs it is always before the termination ms. Every SOM cell fired at least
one spike in [300, 600) on all 24 pairs.

## What fails, read from the raw (diagnostics, never bars)

1. **C7(c), the timing clause.** All 12 defined (c) fails at the corner, and all 60 over the grid, have one cause: the older slot's
   `g_ap` PEAK over [300, 950) falls at or after its termination ms. The facilitating slot -> SOM input makes the volley's later SOM
   spikes the stronger ones. `g_ap` therefore keeps rising after the plateau has already crossed `v_hold`, by 0-57 ms at the corner
   (seed 46: termination 374, peak 428) and by up to 120 ms over the grid. The first rise precedes the termination on 23 of 23 pairs. The packet's self-skeptic
   record states the intent as "SOM spikes and a `g_ap` rise precede the termination". The bar text says peak, and the text is scored.
2. **Self-termination.** The word's own volley recruits the SOM pool, and the resulting `g_ap` (0.32-2.64 at its max in [150, 250) on the 51 failing records) outlasts the
   word's sensory apical drive, which ends at 150 ms. C6 fails on 51 of 72 TRACE records. In 45 of them the c1 word's own plateau fell
   within 50 ms of its pulse end (153-199 ms). The other 6 fell at 308-620 ms, each on a seed with 4-8 SOM spikes after 200 ms: the held
   trace recruiting its own eraser, C8's concern. The defect grows with W_SA and W_MS and drives the C1 failures (a dead plateau leaves the
   soma near 0 Hz). The corner's one UNDEFINED pair is the same event: at seed 50, frog as X recruited SOM spikes at 136-169 ms, `g_ap`
   reached 0.34 at 160 ms, and the plateau crossed `v_hold` at 178 ms, before Y arrived. Under the lesion the same frog plateau held.
3. **What holds.** Wherever the older plateau latched (82 defined pairs over the grid), the SOM path ended it, within 8-137 ms of Y_on
   (bar 300 ms; 0 defined C7(a) fails). The newer plateau
   survived at the corner (23 of 23 defined). The lesion owns the overwrite. This is the first spiking route on this lane that ends an
   older word's held plateau (f1h's somatic pool: 0 of 216).

## Readings where the text admits two

- C7(c) spikes, every SOM cell >= 1 (primary, the stricter) against the pooled count >= 8: no cell-level verdict differs, because every SOM
  cell fired on every pair.
- C7(c) `g_ap` peak over [300, 950) (primary, the C7 span) against the whole trial: no cell-level verdict differs.
- C7(c) with no termination ms (the plateau held): the timing clause has nothing to precede, so (c) reads the spikes and `g_ap` > 0. That
  is exactly the case B3's text names. It matters only to B3, which does not fire.
- C8 per-cell rate (primary, the unit of every rate in the runner and the packets) against the pool's summed rate: this differs only at
  `sa0p125_ms304`, which fails C1, C3, C6 and C7 anyway.
- **Robustness of the STOP verdicts.** STOP-CAL does not depend on the C7(c) timing text. Even under a rise reading the corner would be
  C7 7 PASS / 0 FAIL / 1 UNDEFINED, because seed 50's pair is UNDEFINED by C7(0). STOP-ATTRIB does depend on it. Under a rise reading the
  corner's (c) would have 23 defined passes and 0 defined fails. It would then be UNATTRIBUTED only through the UNDEFINED pair.

## R1 Q5 (engram STP carry-over, seed 41)

The `--stp-reset` arm changes no item verdict on any of the 9 cells (C0-C4, C6-C8, and every C7 sub-item per pair). The reset reads back:
engram u_max is 0.0 at the c1 trial start on all 9 cells, against 0.201 carried from the build in TRACE (and 0.0-0.144 at later trials). By R1 Q5 the carry-over is therefore not an instrument defect
for step 0. The driver's "stpreset c6 P1" equals TRACE seed 41, where C6 passes only at the corner.

## Draft ruling (for Claude; nothing below is a ruling until Claude signs it)

RULING f1i-0-1028 (draft; reads research/coordination/cs-race-f1i-0-SCORE.md, scorer `research/runners/cs_race_f1i_score.py` @ b5ee567a8).
Accepted as scored: STOP-CAL and STOP-ATTRIB fire. STOP-ENGINE and STOP-ID do not. STOP-SMOKE is not judged because no cell was
selected; the C3 cell reads 13.76 / 15.52 pA. No pre-decided branch B1-B4 fires by its text. The driver's provisional `sa0p125_ms215`
is not selected (C7 2/5/1). P0' is HOSTABLE. Identity (i) and (ii) pass.

  Step 1 does not run. It needs a chosen cell. Running TRACE / LESION / NO-ENC / NOFAC at an unselected cell would waive C7.

  Banked as a METHOD verdict, never a capability verdict: the shared facilitating SOM pool -> apical GABA_A ends the older word's held
  plateau through real synapses, and the lesion owns that (corner 23 vs 0 of 23 defined pairs; the first spiking route on this lane,
  f1h 0/216). It misses the step-0 bars for two reasons. (1) C7(c) compares the `g_ap` PEAK with the termination ms, and facilitation
  keeps `g_ap` rising after the plateau falls (12/23; the first rise precedes the fall 23/23). (2) The pulsed word's own SOM IPSP outlasts
  its 150 ms sensory apical drive and ends its own plateau: C6 51/72 FAIL, 45 of them within 50 ms of the pulse end, and the corner's
  UNDEFINED pair (seed 50). Reason (2) alone keeps every cell off 8/8. No reading of (1) changes STOP-CAL.

  The C7(c) text is NOT amended on these seeds (that would be a post-data change). The next packet states its engagement clause before
  any run, as the first `g_ap` increase after Y_on preceding the termination ms with the peak reported, carrying f1i's S1-S8 guards.

  Q3 class-label gap (E build): the MWT98 triple stays. The method uses its facilitation, and no alternative D is verified.

  Next method (Claude chooses; recommendation): the measured defect is the one B1 names, "the newer word dies under its own inhibition".
  Banked alternative 2 is that branch's method: lateral per-slot SOM sparing its own slot, i.e. slot i -> SOM_i -> the apicals of
  slots j != i. It is the Martinotti-mediated disynaptic inhibition between pyramidal cells of Silberberg & Markram 2007 (PMID 17329212,
  in the packet; abstract-level, read before binding). It removes reason (2) by structure. Its cost is selectivity by wiring, which must
  be in `decides:`.
  Alternative: before any new wiring, ask the companion-process question (CLAUDE.md, at a wall): what keeps a real newer word's plateau
  up while its own SOM IPSP decays? Candidates are a sensory or top-down apical input that does not stop at a hard 150 ms edge, and the
  plateau's own regenerative conductance against a shunt. Run `bash tools/before_you_build.sh "the pulsed word's own SOM IPSP ends its
  apical plateau after its drive stops"` first.
  Not recommended: extending this grid below the corner on fresh seeds. That is the retune the packet's branch rule forbids, and the
  corner already shows reason (2) on seed 50.

  Gate note: f1j would join f1h and f1i's CLASS HD scaffold group (same `replaced-by:`, `cap: 2 builds`). Check the group's build count
  with `tools/gates/host_decision_declared.py` before writing it. A third build needs a Claude-signed `cap-raise:` RULING line.
