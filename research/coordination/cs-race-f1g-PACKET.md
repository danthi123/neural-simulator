# PACKET cs-race-f1g (Claude, 2026-10-22): the carrier must still be there when the race is decided

host-decision: scaffold
decides: which engram each heard cue or mention word connects to (the probe wires the cue->E_k and mention->E_k edges at write time from each fact's words, the cs-race packet's declared host residual), and which words an encode turn presents to the mention layer (the fact's own three words, in heard order)
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement); docs/drafts/cs-race/PACKET.md (the declared write-time residual)
replaced-by: write-time Hebbian binding of heard word forms to the engram being written, on the composer's spiking store write path, with the heard sentence's word forms arriving through the hearing path
cap: 2 builds

Board: the cs-race lane (content selection by a spiking race). Inputs: `cs-race-f1e-RESULT.md` (953cf304f),
`cs-race-f1f-RESEARCH.md` (41daa7fd8), and `research/runners/cs_race_f1.py` on `research/cs-race`, which Claude read for
this packet. Probe-local, CPU, numpy, CPU pool under `tools/memcap.sh`. No `sim/` edit and no production change. Gate seeds
42/43/44/100/101/102 are not run. Hermes runs STEP 0 only, then posts its RESULT and STOPs. Step 1 is preregistered
below and runs only after a Claude RULING on step 0.

## Diagnosis (code read, then checked against the f1f numbers)

- **Mention carrier.** `race()` pulses the mentioned word's slot for `MENTION_MS` (150 ms), then runs `GAP_B_MS` (150 ms) of
  zero input before the cue. Mention slots have no recurrence and no persistence, so nothing holds the mention after its
  pulse. Ignition lands 160-210 ms into the cue (f1d point 2), so by the time the race is decided the mention input has been
  off for more than 300 ms.
- **Recency carrier.** Tsodyks-Markram facilitation sits on each engram's OWN recurrence only (`ee_`); every other pathway,
  the cue included, is `stp_disabled`. Facilitation scales recurrent transmission only once the engram fires, and the shared
  cue ignites both engrams within 3-4 ms of each other (f1d point 2). The recency difference therefore arrives after the
  decision.
- **The f1f numbers agree.** Cue-raster asymmetry is 0.0000 pA on 240/240 draws per cell. That is expected: both engrams
  share the cue words. The total excitatory asymmetry is 0.1-0.4 pA, against roughly ±10 pA of noise. The steer ladder
  needs +4 pA to reach 0.71 and +6 pA to reach 0.80, from 0.50 at zero.
- **Reading.** Steps f1 to f1e measured two carriers that were absent from the window that decides the race. That is a
  verdict on this carrier METHOD (a transient pulse, plus a gain that acts only after firing), and it is banked as such. It
  is not evidence that content carriers cannot own a spiking race.
- **Wall question.** What does the real system run alongside a heard word that this probe replaced with a constant?
  Persistent activity: f1f item 1 cites persistent firing at 4.5-6.5 Hz lasting tens of seconds after a brief trigger under
  cholinergic tone (PMC3773044). The probe replaced it with zero after the pulse. The cs-race packet's own K1 already names
  this as the next carrier: a recurrent persistent mention trace.

## Method: one carrier for both clauses, a persistent word trace in the mention layer

- **Trace.** Each mention slot (8 cells) gets recurrent self-excitation through the engine's slow NMDA receptor
  (`receptor="nmda_slow"`, the receptor the engram's `eek_nmda_` path already uses; `cfg.enable_nmda_recurrent` on) at
  weight `W_MM`.
- **Competition.** All mention slots share one inhibitory pool of `NIM` = 8 cells (slot -> pool `W_MI`, E_TO_I; pool -> every
  slot `W_IM`, I_TO_E, GABA_A). A newly heard word's trace therefore suppresses the older one. This is working-memory
  overwrite, with a capacity of one held word here.
- **Mention -> engram.** `W_MENT` stays 12 (primes, never ignites).
- **Encode turns are heard.** Each encode turn now also presents its fact's three words (agent, action, patient) to their
  mention slots in heard order, 50 ms each at `MENT_PA` (40 pA), starting at the encode pulse's onset. This is world-side: a
  heard sentence activates its word forms, and the host passes no per-candidate number. The context mention turn is
  unchanged.
- **Controls stay bare.** The positive control and the decisiveness control (`ABsim`) present no words to the mention layer,
  exactly as today, so they still measure the race alone.
- **What each clause then sees.** Recency becomes a property of the trace. After "lynx chase vole ... lynx chase frog" the
  held word is frog, because it was heard last. After the context mention "vole", the held word is vole. The shared cue words
  (lynx, chase) prime both engrams equally and carry no asymmetry; only a held patient word does. No host code picks the
  held word.
- **New flags.** Defaults reproduce today's circuit exactly: `--w-mm` (0), `--w-mi` (0), `--w-im` (0), `--nim` (8),
  `--encode-words` (off). With no new flag the step-1d identity check must still pass.

## Step 0 (Hermes, now): calibrate the trace with no race

Use calibration seeds 41-48 only. No arm has run on them, and nothing may be tuned on DEV 1-8 or CONFIRM 25-40. Every run
uses grid row 3 (W_EE 40, W_IE 40), W_GIS 30, plasticity as in f1e, and no cue.

**Grid (frozen):** `W_MM` in {10, 15, 20, 30, 40, 60}; `W_IM` in {20, 40, 80}; `W_MI` 20; `NIM` 8.

**Items** (each per seed, on 8/8 calibration seeds; read from spike counts, never from a host flag):
- **C1, persistence.** After a 150 ms pulse at `MENT_PA`, the pulsed slot fires at 2-20 Hz (mean rate per cell) across the
  next 650 ms (`GAP_B_MS` + `CUE_MS`).
- **C2, quiescence.** Every unpulsed slot fires below 1 Hz in that window. With no pulse at all, no slot fires above 1 Hz in
  2,000 ms.
- **C3, overwrite.** Pulse slot X, wait 150 ms, then pulse slot Y. From 300 ms after Y's onset to 650 ms after it, Y fires
  at 2-20 Hz and X below 1 Hz. Test both orders and three word pairs: (vole, frog), (frog, vole) and (lynx, grass).
- **C4, no ignition by the trace.** With the trace held and no cue, the engram it primes shows no burst (3 or more spikes
  in 5 ms) and no latch for 650 ms. This is clause (g)'s condition.
- **C5, delivered current.** Run the clause-(a) and clause-(b) protocols with the encode words on but the cue withheld (no
  race). Read the mean total excitatory synaptic current per cell (the f1e field definition) into the held word's engram
  minus the other engram, in the window where the race would sit, [100, 200] ms after the cue would have started. Report
  the per-seed value, in pA, beside the +4/+6 pA flip scale.

**Selection (frozen):** among cells passing C1-C4 on 8/8 seeds, take the smallest `W_MM`, then the smallest `W_IM`.

**Record:** each cell's per-seed rates, the C3 matrix, the C5 values, `enable_nmda_recurrent` and the heterogeneity flags
read back, and every run's `.prov.json` (module runs through `research/runners/__init__.py`).

**STOP rules (each is a STOP with its own RESULT; nothing is re-run with other settings without a Claude RULING):**
- **STOP-CAL:** no cell passes C1-C4 on 8/8 seeds. The RESULT names each cell's failing items.
- **STOP-SMOKE:** at the chosen cell, the median-seed C5 asymmetry is below 2 pA in either clause protocol. The trace would
  then be forecast too weak to flip a race that needs +4 to +6 pA.
- **STOP-ID:** with no new flag, the step-1d identity check fails.

**Post** `[cs-race-f1g-0] RESULT` with every item per cell, the chosen cell, C5 against the flip scale, every STOP verdict
and the branch @ sha. Then STOP.

## Step 1 (preregistered now; runs only after a Claude RULING on step 0)

- **Scope.** The full clause battery (a)-(i), the positive control and the decisiveness control, on rows 3 and 4 at the
  chosen cell, with W_GIS 30 held as the declared read-out repair. Seeds: DEV 1-8 and CONFIRM 25-40, 10 draws each, with
  noise seeds paired across arms.
- **Arms:**
  - TRACE: the trace plus the encode words.
  - LESION: W_MM 0 and W_IM 0, encode words kept. This isolates persistence.
  - NO-ENC: the trace without the encode words. Only the context mention can create a trace here, so clause (a)'s recency
    is attributable.
- **Ownership (same bars as f1e).** The carrier OWNS the race iff, in TRACE, clauses (a) and (b) each pass (8 or more of 10
  per seed) on at least 7/8 DEV AND at least 14/16 CONFIRM. On those same seeds, the positive control must be intact
  (9 or more of 10), clause (c) must pass (one-role cue with maximal priming gives 0 latches), and clause (g) must pass
  (clean before the cue). LESION must pass (a) and (b) on at most 2/8 DEV seeds.
- **Attribution.** If NO-ENC passes (a), the recency came from something other than the heard encode words, and the RESULT
  says so.
- **Full K1.** The "every clause on 8/8" verdict is reported separately and is not the ownership bar.
- **Required lenses.** Engagement: each arm's trace rate and C5 value are recorded per draw, or the arm is UNDEFINED.
  Where-ran: run `tools/where_ran.py` on every result. Raw magnitudes for both arms of every A/B.
