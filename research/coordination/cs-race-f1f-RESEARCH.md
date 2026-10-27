# cs-race-f1f RESEARCH — read-only round: how much current do the carriers
# actually deliver?  (ruling `cs-race-f1e-1022`)

**Date:** 2026-10-22 · **Author:** driver (local-LLM session) · **Branch:**
`research/cs-race` · **Status:** READ-ONLY round complete — no simulations
were run; no runner, scorer or bar was touched. All numbers below are read
from EXISTING records (paths given) or from cited published sources.

```yaml
preconditions:
  task: "RULING cs-race-f1e-1022 (main ba35fa088): read-only research round
         after the f1e verdict; estimate carrier-delivered pre-ignition
         current vs the +4/+8 pA steer; list companion processes the probe
         replaced with constants."
  inputs frozen: "Existing records only — research/findings/raw/_cs_race/f1e/
         row{3,4}_s{1..8,25..40}.json and worktree _cs_race/f1d/m/*.json;
         no new runs (compute-placement), no runner/scorer/bar edits (§2)."
  estimate rule: "Per clause-(a)/(b) draw, ignition-relative pre_ign window:
         |Δcue_only_current_pa_per_k| and Δtotal_exc_current_pa_per_k between
         the two cue-active engrams; steer reference = clauses.e.ladder +
         steer_star_pa (same records) and the f1d +4/+8 aggregates (a6f70651d)."
  failure branch: "NEXT = Claude designs the next carrier method (§2
         reserved); driver posts [cs-race-f1f] RESEARCH and stops."
```

Ruled task (RULING `cs-race-f1e-1022`, main `ba35fa088`): carriers do not
own the race (f1e verdict, `cs-race-f1e-RESULT.md` @ `953cf304f`); before
any next method: (1) research round, (2) estimate the extra current each
carrier delivers to its favoured engram pre-ignition vs the +4/+8 pA
steer that moved the winner in f1d (5/8, 7/8 seeds), (3) list what the
real system runs alongside a recency/mention carrier that the probe
replaced with a constant.

## 1. Research round as executed

- `bash tools/deep_research.sh "content carrier strength in a spiking
  selection race"` — local-record half (before_you_build, rc=0, logged to
  `research/coordination/corpus_checks_shared.jsonl`) plus the external
  half run by hand below; external queries recorded via
  `tools/record_external_search.sh` (4 entries, this date).
- `rag_search` (corpus `finding`, `plan`, `coord`) for
  "carrier strength recency mention bias spiking winner-take-all race
  engram". Top local pointers (read): `research/coordination/
  cs-race-f1d-RULING.md`, `cs-race-f1b-RESULT.md`, `cs-race-f1c-RESULT.md`
  (the arc's own carrier history), and finding
  `fm-neural-wta-readout-lateral-inhibition-negative-vs-host-argmax`
  (local WTA readout is inhibition-based; the negative there concerns the
  host-argmax comparison, not this probe).
- External searches (top sources read in full text as served):
  1. WTA/biased-competition bias size — Frontiers Comput Neurosci 2019
     "Analysis of Biased Competition and Cooperation for Attention";
     PNAS 2019 corticostriatal biased competition; NeurIPS 1991
     Dynamically-Adaptive WTA.
  2. Engram allocation competition — Rashid et al. 2016 Science
     ("Competition between engrams…"); PMC11112642 (excitability vs
     inhibition allocation model); PMC9623596 (Memory Allocation review);
     PMC9730430 (engram sparsity microcircuit).
  3. Short-term facilitation — Tsodyks & Markram 1998 (Neural Networks
     with Dynamic Synapses); Markram et al. 1998 (Neuropharmacology);
     Scholarpedia Short-term synaptic plasticity.
  4. Persistent activity — PMC3773044 (cholinergic persistent firing,
     MEC layer III); Molecular Brain BDNF/proBDNF persistent activity;
     Egorov 2002 graded persistent activity (via Neuron 2006 model).

## 2. The estimate — what the carriers deliver (existing records only)

Records: `research/findings/raw/_cs_race/f1e/row{3,4}_s{1..8,25..40}.json`
(48 files), field path `results[0].per_draw[draw].pre_ign` (the
ignition-relative pre-ignition window; `t_ign` is later than
`window_ms`, and `latch_onset` is later than `t_ign`, so nothing here is
post-latch). Two cue-active engrams (k0/k1) per clause-(a)/(b) draw; k2
is cue-silent there (`cue_only_current_pa_per_k[k]=0`).

Per-draw asymmetry between the two cue-active engrams, pooled 240 draws
per cell (48 records x 10 draws x clause):

| cell | mean \|Δcue_only\| (pA) | exact-zero draws | mean Δtotal (pA) | median Δtotal (pA) | range Δtotal (pA) | mean ΔVm (mV) | winner-sign match |
|---|---|---|---|---|---|---|---|
| row3 clause a (recency) | 0.0000 | 240/240 | +0.157 | +0.026 | [-7.34, +10.88] | -0.048 | 76/240 |
| row3 clause b (mention) | 0.0000 | 240/240 | +0.124 | -0.029 | [-8.37, +10.73] | -0.081 | 76/240 |
| row4 clause a (recency) | 0.0000 | 240/240 | +0.399 | +0.133 | [-7.83, +13.09] | -0.024 | 82/240 |
| row4 clause b (mention) | 0.0000 | 240/240 | +0.139 | -0.012 | [-10.22, +13.03] | -0.078 | 79/240 |

Definitions taken from the records themselves:
`cue_only_current_pa_per_k` = "W_CUE * cue raster convolved with the
engine conductance kernel" (the field's own `cue_only_formula`);
`total_exc_current_pa_per_k` = "g_e * (E_e - V) per cell, E_e = 0 mV,
mean over engram cells in the window" (the field's own
`total_exc_current_note`); `vm_mean_mv_per_k` = same window mean V.
Δ = k(first active) - k(second active). `<!--derived-->` means computed
by this script from those two fields, not stored.

**Reading:** in rows 3-4 the carrier delivers **exactly 0.0000 pA of
cue-raster asymmetry** to its favoured engram before ignition (240/240
draws per cell), and the total excitatory asymmetry that does exist
averages +0.12 to +0.40 pA (median ≈ 0) with a ±8-13 pA noise spread
whose SIGN predicts the eventual winner only 76-82/240 (32-34%, below
the 50% chance line) — the pre-ignition current does not carry the
clause signal at all; the outcome is decided later, at the gate/latch
race, consistent with the f1e OWNERSHIP: False verdict.

Against the steer (same records, `results[0].clauses.e.ladder` +
`steer_star_pa`, pooled over the 48 records):

| injected steer (pA) | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| a_wins / 480 | 242 | 292 | 340 | 383 | 406 | 428 | 461 | 473 |
| win fraction | 0.50 | 0.61 | 0.71 | 0.80 | 0.85 | 0.89 | 0.96 | 0.99 |

`steer_star_pa` (the star = the ladder's saturating step recorded per
record): {0: 9, 2: 3, 4: 6, 6: 15, 8: 4, 10: 4, 12: 5, 16: 2} — mode
and median 6 pA. The f1d aggregate reference in the ruling (engram steer
+4 pA moved the winner in 5/8 dev seeds, +8 pA in 7/8; f1d RESULT
`cs-race-f1d-RESULT.md` @ `a6f70651d`) agrees with this ladder.

**Estimate (answers ruling item 2):** a content carrier in rows 3-4
delivers **≈ 0.2 - 0.4 pA** of extra pre-ignition current to its
favoured engram (0 pA from the cue raster by construction; 0.1-0.4 pA
total, riding on ±10 pA noise), while the steer that moves the race from
chance (0.50) to clause-bar performance needs **+4 to +6 pA**
(f1d: +4 pA = 5/8 seeds). **The carriers are roughly 10-20x under the
current scale that flips this race** — and their residual asymmetry
fails even the sign test. UNDEFINED in these records: (i) a per-draw
"carrier-predicted winner" label (the sign test therefore uses the
recorded `winner`, which is the measured race outcome); (ii) any pA
equivalent of the mV-scale companions in item 3 (would need the cells'
input resistance, not recorded here).

## 3. What the real system runs alongside, that the probe replaced with a constant

The wall question ("what else does the real system run alongside a
recency/mention carrier that we replaced with a constant?") applied to
this probe — the probe's recency/mention input is a fixed W_CUE raster
into a fixed gate window. Real circuits add at least:

1. **Persistent activity (intrinsic plateau).** Muscarinic CAN/TRPC
   plateau potentials keep entorhinal pyramidal cells firing for tens of
   seconds after a brief trigger: a 50 pA / 2 s step under carbachol
   (2-10 uM) yields persistent firing in 91% of MEC layer III cells at
   4.5-6.5 Hz with plateau depolarization **7.6-9.8 mV**, stable >
   30-50 s (PMC3773044); BDNF raises the plateau from 1.71 +/- 0.48 mV
   to 12.71 +/- 1.88 mV and rate 0.007 -> 2.80 Hz (Molecular Brain,
   proBDNF/BDNF). The probe replaced this self-sustained bias with the
   constant gate window (pA equivalent UNDEFINED without input
   resistance).
2. **Short-term facilitation (dynamic synapses).** Pyramidal ->
   interneuron facilitation with U_SE ~ 0.01-0.05 and slow tau_facil
   builds release utilization over a burst: for U=0.15, tau_fac=750 ms at
   20 Hz the steady utilization is u ~ 0.73, i.e. **~5x the first-spike
   efficacy** (Tsodyks & Markram 1998); L5 pyramidal pairs span
   U = 0.1-0.95, mostly depressing (Markram et al. 1998). A recency
   carrier whose last cue arrives as a burst would ride this multiplier;
   fixed W_CUE cannot.
3. **Neuromodulatory gain (ACh/NE state).** The same cholinergic line:
   carbachol moves persistent-firing incidence 6/14 -> 10/11 (43% ->
   91%) and rate 0.43 -> 5.00 Hz (~12x) in MEC III (PMC3773044); in the
   allocation models the winner of the excitability-vs-inhibition
   balance flips with neuromodulator-linked excitability (Rashid et al.
   2016 Science: engram competition decided by relative excitability
   within ~6 h; PMC11112642: excitability boost vs inhibition level
   trade-off sets co-/dis-allocation). The probe holds excitability and
   inhibition constant across cues — the real system gates the same
   synaptic input through a modulator state.
4. **Attention / top-down biased competition.** The top-down bias needed
   to flip a cortical biased-competition network scales LINEARLY with
   the input gap (slope ~140/6 per unit gap, feedback strength 0.1-0.5
   x feedforward — Frontiers Comput Neurosci 2019); biases may ride
   firing rate OR coherence and are read out at 100 ms (accumulator) vs
   5 ms (coincidence) timescales (PNAS 2019), and "even weak activity
   biases strongly influence downstream attractor dynamics". Attention
   in the real system is an EXTERNAL bias channel the probe has no
   representation of.

Headline for the next method design (sources, not a proposal): the
scales that real circuits use to bias a competition are (i) plateau
plateaus of ~8-10 mV sustained for seconds, (ii) burst multipliers of
~5x from facilitation, (iii) modulator-linked ~12x firing-rate gains,
(iv) an attention bias linear in the input gap — all one to two orders
of magnitude above the ~0.2-0.4 pA the current carriers deliver, and
roughly at or above the +4-6 pA that this race is known to flip by.

## 4. Deliverable status

Read-only round CLOSED per ruling. NEXT = Claude's design of the next
carrier method informed by sections 2-3 (§2 reserved: runner/method
design, preregs, bars are Claude's). Posted as NEEDS
`[cs-race-f1f] RESEARCH`; driver stops here.
