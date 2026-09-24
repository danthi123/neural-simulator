# Pre-registration: curiosity x metacog v4 — a PHASIC lc_ne that raises ASK's response gain by withdrawing ASK's own slow negative feedback

Runner: `research/runners/_curiosity_lcne_phasic_gain_derisk.py` (committed before this document; the constants and
thresholds below are the runner's module constants, frozen). Branch: `research/curiosity-lcne-phasic-gain`.
Builds on: `docs/plans/2026-09-23-curiosity-metacog-neuromod-gain-PREREG.md` (v3) and its 6-seed NO-GO,
`research/findings/2026-09-24-curiosity-metacog-lcne-modulator-6seed-NOGO-calibration-seed-only.md`.

**Evaluation set: seeds 42/43/44/100/101/102, ALL SIX held out.** Every constant below was chosen on DEV seeds
7/8/9/10 only. No evaluation seed has been run with this mechanism. (Seeds 42-102 were run with v3's DIFFERENT
mechanism and that verdict is public; nothing here was fitted to it — see §5 for how that knowledge is used.)

## 0. Why this rung, and the wall question

v3 wired a spiking `lc_ne` population (graded by metacog evidence; its G10 passed on every seed) onto curiosity's
`ask` pool as ADDITIVE sub-threshold current. It owned 8-25% of the ASK dynamic range (floor 0.2) and passed on
the calibration seed only. The verdict is on the METHOD — a tonic rate delivered as added current.

External record (read before building; `bash tools/before_you_build.sh "lc-ne multiplicative gain on a target pool"`
logged, and its two hits read in full: the 2026-08-10 NE real-substrate finding and the 2026-09-04 GNW LC finding):

- Aston-Jones & Cohen 2005, Annu Rev Neurosci 28:403-450, doi:10.1146/annurev.neuro.28.061604.135709 — LC PHASIC
  bursts are locked to the decision and facilitate the ensuing response; tonic firing is a different mode.
- Fan, To & Sciolino 2026, Cell Rep 45(9):117990, doi:10.1016/j.celrep.2026.117990 (abstract; no full text in PMC) —
  phasic LC activation expands the dynamic range of a cortical population's representation, attributed to
  multiplicative gain plus tuning changes; tonic activation engages fewer neurons and does not expand it.
- Madison & Nicoll 1986, J Physiol 372:221-244, doi:10.1113/jphysiol.1986.sp016006 (and 1982, Nature 299:636) —
  noradrenaline (beta1) reduces the slow Ca2+-activated K+ afterhyperpolarization (sAHP), reducing accommodation,
  so the responses to depolarizing stimuli are enhanced.
- Ermentrout 1998, Neural Comput 10:1721-1729, doi:10.1162/089976698300017106 — slow negative feedback on a steep
  F-I curve linearizes it; the details of the adaptation mechanism do not change this as long as it is slow.
- Holt & Koch 1997, Neural Comput 9:1001-1013, doi:10.1162/neco.1997.9.5.1001 — shunting inhibition (a
  conductance change) is SUBTRACTIVE on firing rate, not divisive.
- Chance, Abbott & Reyes 2002, Neuron 35:773-782, doi:10.1016/s0896-6273(02)00820-6 — balanced background E+I
  input sets gain divisively; the divisive effect needs the NOISE that the background brings.
- Murphy & Miller 2003, J Neurosci 23:10040-10051, doi:10.1523/JNEUROSCI.23-31-10040.2003 — additive input can
  look multiplicative when the rate curve is an expansive power law (why §3's G11 does not test slope).
- Williams, Henderson & North 1985, Neuroscience 14:95-101, doi:10.1016/0306-4522(85)90166-6 — alpha2
  adrenoceptors on LC neurons open a K+ conductance (the autoinhibition that ends a burst).

**What the real system runs alongside the edge that v3 replaced with a constant:** ASK's own output-proportional
slow negative feedback (the sAHP / accommodation). v3's ASK had none, so `lc_ne` could only add current. NE's
cellular gain action is to REMOVE that feedback. Removing a feedback that is proportional to the output changes
the slope and not the threshold: where ASK is silent there is no feedback to remove.

**Why not the balanced-background route the v3 finding named as a candidate:** in this pool there is no OU or
conductance noise (the production metacog config), so by Holt & Koch a background withdrawal is a conductance
change that shifts ASK's threshold — the additive signature. Chance et al.'s divisive effect needs the noise.
This route was not built; it is recorded here as a reason, not as a measurement.

## 1. Mechanism under test (every step neurons + synapses)

Pool: `[metacog (production, param-het), curiosity, metacog_margin (comparator), lcne_gain_organ]`, plus v3's
frozen point-edge `meta_schema -> ask` (weight 4.0). Host code drives only metacog's input evidence, as production
does, and reads spike rasters after each step. No host scalar reaches any neuron; no neuromodulator subsystem.

| synapse | type | weight | role |
|---|---|---|---|
| `meta_schema -> lc_ne` | dense E (AMPA) | W_CMP_LC 5.0 | the comparator drives LC (v3's value) |
| `lc_ne -> lc_ne` | GIRK (receptor `gaba_b`, E=-90 mV, tau 150 ms) | W_LC_AUTO 3.0 | alpha2 autoinhibition: one burst, then a pause |
| `ask -> ask_fb` | dense E | W_ASK_FB 50.0 | ASK's output drives a fast-spiking relay |
| `ask_fb -> ask` | GIRK | W_FB_ASK 16.0 | ASK's output-proportional slow negative feedback |
| `lc_ne -> ask_fb` | GIRK | W_LC_FB 35.0 | the phasic burst silences the relay: NE withdraws the feedback |
| `meta_schema -> lc_add`, `lc_add -> ask` | dense E | 5.0, W_ADD_ASK 2.0 | ADDITIVE CONTROL (v3's lc_ne clone); output CLOSED in every mechanism arm |

Populations: `lc_ne` 20 RS neurons typed inhibitory (its output is Gi/GIRK-coupled on every target); `ask_fb` 20 FS
neurons; `lc_add` 20 RS excitatory. All three carry per-region parameter heterogeneity. All weights fixed.

**Engine workaround (declared; no `sim/` edit).** `inject_explicit_wiring` keeps a stale GABA_B routing mask on a
second call (the `is None` guard from commit 9ff3b6353), and the merge framework injects twice. Measured on dev
seed 7 with `enable_gabab` on: 871 metacog `workspace` synapses were routed into GIRK and metacog's balance moved
from 0.022-0.093 to 0.705-0.754. The organ's wiring function sets the mask to None before the second inject.
Two integrity preconditions (§3) check that it held. The engine fix was flagged as a separate task.

**Residuals (declared, not closed by this rung):** the sAHP is carried by a relay population, not an intrinsic
spike-triggered K+ current; NE's beta1/cAMP block of the sAHP is represented by a Gi-type GIRK inhibition of that
relay; all weights are hand-set on dev seeds; `lc_ne` is driven by the comparator's summed output, not by a
dedicated ACC/OFC utility monitor; ASK's operating point relative to the edge's drive is whatever the seed gives
(no homeostatic set-point process) — see §5.

## 2. Operating point and how it was chosen (dev seeds 7/8/9/10 only)

**Criterion (fixed before choosing):** the worst case over dev seeds 7, 8 and 10 must clear every floor the
MECHANISM controls (G3, G10, G11, G12) with margin. Dev seed 9 is excluded from the criterion and reported: the
point-edge alone drives its ASK to 0.43 Hz at the most uncertain level, below G1's 1.0 Hz range floor, so there is
no response for a gain to act on. Dev seed 8 fails G1 in every configuration, including with the modulator
closed; G1 is a property of the base circuit there (§5), so it is not part of the criterion either.

Measured on the scratch calibration grids (numpy, 11-level evidence grid; edge-drive sweep at evidence 0):

| choice | alternatives measured | dev-seed result that decided it |
|---|---|---|
| W_CMP_LC 5.0 | 8.0 | at 8.0, lc_ne fires one volley at 20-35 ms at EVERY evidence level on seed 7 (onset-locked, not uncertainty-locked; probe run before per-region heterogeneity was added to lc_ne) |
| W_LC_AUTO 3.0 | 8.0, 16.0 | rho(evidence, lc_ne Hz) on seed 8: -0.61 / -0.10 / -0.30; on seed 10: -0.72 / -0.50 / flat |
| W_ASK_FB 50.0 (W_FB_ASK 16, W_LC_FB 35) | 20, 35 | dev-worst G3 attributable fraction: 0.06 (20, W_LC_FB 20) / 0.20 (35) / 0.29 (50) |
| FS relay, no GIRK cap | RS relay; `gabab_conductance_max` 1.5/3/6 | RS relay: dev-worst G3 0.03-0.14; every cap: dev-worst G3 <= 0.17, and caps 1.5 and 3 fail G11 on some dev seed |
| W_ADD_ASK 2.0 | gate scale 1.0/1.5/2.0 | the additive control's own max ASK ratio is 1.36-1.70 on the rising limb (a real effect), and its G11c trend is -0.5 to -1.0 on seeds 7/8/10 |

At the frozen point, dev seeds 7/8/10: G3 attributable 0.52 / 0.48 / 0.29; G11 gain at the top of the rising limb
2.06 / 1.75 / 2.72 with ratio trend +0.87 / +0.5 / +1.0, while the additive control's trend is -1.0 / -0.5 / -1.0.
lc_ne: per-neuron burst concentration 0.984 / 1.0 / 0.992 (seed 9: 1.0); alpha2-lesion spike ratio 3.78 / 2.69 /
3.24 (seed 9: 1.75). These are calibration measurements, not evidence; a dev-seed smoke from this committed
revision follows in a separate commit.

**A finding from calibration that shaped G11:** above the operating drive, the lc-off (reference) ASK response
FALLS with more drive. Seed 7, edge_drive 0.8 / 0.9 / 1.0 / 1.1 / 1.2: 0.78 / 1.77 / 2.48 / 1.69 / 1.41 Hz. The
strong drive makes ASK fire on the comparator's onset transient, which recruits the slow feedback before any phasic
lc_ne burst exists. A ratio over a falling denominator would inflate "gain", so G11 scores only the rising limb.

**Rejected instrument change, recorded:** a floored G1 (level means below 0.25 Hz tied at 0 before ranking) lifts
dev seed 8 from -0.607 to -0.863 but drops dev seed 10 from -0.977 to -0.786, because the ties cap the attainable
rho. G1 stays v3's raw statistic; the floored rho is reported only.

## 3. Gates (per seed; GO requires every REQUIRED gate on 6/6 of 42/43/44/100/101/102)

| id | required | pass condition |
|---|---|---|
| G1 | yes | v3's, unchanged: Spearman rho(evidence, level-mean ASK Hz) on the combined arm <= -0.8 AND ASK range >= 1.0 Hz; None fails |
| G3 | yes | closing `lc_ne -> ask_fb` (edge intact) removes >= 0.2 of the combined arm's ASK range (`tools.lab.attributable_to`) |
| G6 | yes | the combined-arm digest (ASK and lc_ne per-rep rates, balances, ask_fb rate, metacog/comparator/lc_ne raster hashes) is identical in a fresh subprocess |
| G7 | yes | class swap (evidence into the other assembly): Spearman rho <= -0.8 (v3's statistic) |
| G8 | yes | comparator-relay lesion (relay->meta_schema weights zeroed): rho > -0.5; None fails |
| G10 | yes | v3's: rho(evidence, lc_ne level-mean Hz) <= -0.3 AND lc_ne range >= 0.3 Hz |
| G11 | yes | MULTIPLICATIVE, NOT ADDITIVE — below |
| G12 | yes | PHASIC — below |
| G4 | integrity, reported | edge AND lc lesioned: rho > -0.8 or None |
| G5 | precondition | metacog balance/confident/raster and comparator raster EXACT across every lesion arm |
| S1, G9 | secondary | production-threshold reach; per-rep permutation null (both as v3) |

**G11 (multiplicative, not additive).** At evidence 0.0, the edge's `transmission_gate` scales the edge's drive
over edge_drive = 0 and 0.6 / 0.7 / 0.8 / 0.9 / 1.0 / 1.1 / 1.2, in three arms: `on` (mechanism intact), `off`
(`lc_ne -> ask_fb` closed), `add` (off + the additive control opened). `lc_ne`'s raster must hash identical
across all 24 reads (the modulator's input held fixed; else UNDEFINED). A point is DEFINED when off >= 0.25 Hz; the
RISING LIMB is the defined points up to the drive where `off` peaks; >= 3 limb points or UNDEFINED.

- G11a: |on - off| at edge_drive 0 <= 0.05 Hz (the modulator does nothing without the drive).
- G11b: on/off at the top of the rising limb >= 1.15.
- G11c: Spearman(drive, on/off) over the limb >= 0.0, ratios quantized to 0.05 before ranking.
- Instrument validity, same seed: the additive control must FAIL G11c while its own max ratio is >= 1.15. If it
  passes G11c, this instrument cannot separate additive from multiplicative on that seed: UNDEFINED, never a pass.

Why the ratio's TREND and not a slope: an additive input shift gives a ratio ((x+c-t)/(x-t))^n that falls with drive
for any threshold-power-law rate curve, including the expansive case where additive input looks multiplicative in
slope (Murphy & Miller 2003); so does an input gain f(kx) with a threshold. Only a response (output) gain with no
threshold shift keeps the ratio flat or rising. The selftest (`--selftest`) verifies both failing directions on
synthetic linear and power-law curves, plus a collapsing reference that must not pass.

**What kind of gain this is, stated before the data:** withdrawing a feedback that engages above the relay's own
threshold raises the slope with no threshold shift, and the size of the change grows with the response. It is not
a constant-factor scaling. G11 is built to accept that and to reject an added current.

**G12 (phasic).** Pooled over every level x rep x neuron of the combined arm: (a) >= 0.95 of lc_ne's spikes fall
within 25 steps of that neuron's first spike in that rep; (b) lesioning the alpha2 autoinhibition raises lc_ne's
total spike count by >= 1.5x (the comparator's drive continues after the burst, and the autoinhibition is what
silences lc_ne — part (a) alone reads 0.89-0.97 on v3's lc clone on the dev seeds, so it cannot separate a burst
from a sparsely driven cell); (c) lc_ne fires on >= 1 rep at evidence 0.

**Integrity preconditions (any failure makes the seed's verdict UNDEFINED, never a pass):** metacog balance varies
over the grid; byte-off (the pool minus exactly the declared synapses — everything touching `lc_ne`/`ask_fb`/
`lc_add`, plus the point-edge — equals the conflict_xedge rung's `coupled=False` pool key-for-key, and the removed
count equals the declared count); the GIRK-routed set is exactly the declared GIRK rows plus curiosity's own
`striosome_value -> snc`; metacog's read is EXACTLY the conflict_xedge coupled pool's (balances, confident flags,
metacog and comparator raster hashes); G5; restore exact; every lesion read back off the bridge at measurement
(gate values; relay weight sum 0); lc_ne affects ASK only through the loop (with `ask_fb -> ask` closed, ASK is
identical with lc_ne on or off); curiosity's non-ASK regions emit 0 spikes in every read; no host novelty scalar
and no neuromodulator subsystem.

## 4. What a GO would and would not mean

A 6/6 GO means: on this 4-organ merged pool, a phasic, evidence-graded spiking lc_ne raises the response gain of
curiosity's ASK pool on metacog's edge drive — the effect grows with the drive and has no offset; an additive
control of comparable size fails the same test on every seed — and it owns >= 20% of ASK's dynamic range. All
other circuits are left exact.

It does NOT mean: that ASK reaches production's threshold (S1, secondary); that any weight self-organized; that
the sAHP or the NE receptor pharmacology is modelled biophysically (§1 residuals); that this runs on the 11-organ
production pool or the chat path; anything about felt states. Functional read-outs only.

## 5. Known risks, stated before any evaluation run

1. **Base-circuit G1 failures.** Dev seed 8 fails G1 (raw rho -0.607) with the modulator OPEN or CLOSED: its ASK has
   a 0.02 -> 0.10 Hz tail at the confident end. v3's public 6-seed data shows eval seed 44 failing G1 the same way
   (a 1.1 Hz reading at evidence 1.0). A gain cannot repair a rank reversal at levels where ASK barely fires. No
   threshold here was set from v3's evaluation data; G1 is v3's unchanged.
2. **Weak base drive.** Where the point-edge alone barely drives ASK (dev seed 9: 0.43 Hz), there is nothing to
   scale; G1's range floor and G3 fail. v3's public data shows eval seed 101 with the smallest base range (1.66 Hz).
   The companion process that would place each seed's ASK on the responsive part of its rate curve — a homeostatic
   set-point — is named as the next rung, not built here (the 2026-08-10 NE real-substrate finding reached the same
   wall from a different circuit).
3. **Early reference peak.** A seed with a stronger base drive may peak below edge_drive 1.0; the grid extends to 0.6
   so that the rising limb can still hold >= 3 points.

## 6. Compute and the combined verdict

- Dev confirmation smoke (after this commit, separate artifact commit, labeled DEV-SMOKE, not evidence):
  `SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -u -m research.runners._curiosity_lcne_phasic_gain_derisk --seeds <7|8|9|10> --dev --out research/findings/raw/_curiosity_lcne_phasic_gain_dev_s<seed>.json`
- Evaluation: one pool job per seed, pinned to the pushed revision that contains this document and the runner,
  `--seeds <s> --out research/findings/raw/_curiosity_lcne_phasic_gain_s<s>.json` for s in 42/43/44/100/101/102.
- **The ONLY pre-registered verdict** is `--combine` over the six per-seed files, which refuses unless the union
  is exactly {42,43,44,100,101,102} with no duplicate, one mechanism string, one operating point, and one runner
  blob across the inputs' provenance SHAs:
  `python -m research.runners._curiosity_lcne_phasic_gain_derisk --combine <six files> --out research/findings/raw/_curiosity_lcne_phasic_gain_6seed_combined.json`

## 7. Amendment log

v1, 2026-09-24: initial registration, written after the dev-seed calibration grids and before any run of this
mechanism on seeds 42/43/44/100/101/102. The runner revision it governs is the one committed immediately before
this document on `research/curiosity-lcne-phasic-gain`.

v1.1, 2026-09-24, still before any evaluation-seed run of this mechanism. The dev confirmation smoke (seeds 7-10,
revision cb4ad0c2e) failed G7 on dev seeds 8 and 10. A scratch check showed the conflict_xedge base pool (no v4
organ) gives the identical class-swap curve on both (rho -0.776 and -0.271; seed 10's ASK rises to 0.76 Hz at
swapped evidence 1.0), so these are base-circuit failures of the kind §5 risk 1 names. To let the evaluation
attribute any G7 failure without a post-hoc run, ONE reporting-only arm is added: the class swap with
`lc_ne -> ask_fb` closed (`rho_swap_gain_lesion_arm`, arm `class_swap_gain_lesion`). No gate, threshold,
constant or statistic changes; G7 is still scored on the intact class-swap arm exactly as in v1. The governed
runner revision is now the commit that carries this amendment. The v1 dev artifacts are superseded by a re-run
from that revision; the evaluation pool lines pin that revision.
