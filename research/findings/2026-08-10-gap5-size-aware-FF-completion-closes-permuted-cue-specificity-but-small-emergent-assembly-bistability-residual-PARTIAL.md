---
type: finding
status: contributing
date: 2026-08-10
mechanism: ca3-completion
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: The end-to-end gap#5 episodic loop is composed on ONE spiking substrate (n_ca3=2000) — emergent-DG SELECTION (the 2026-07-21 recovered-at-scale sparse mossy-detonator: d0.02/w3000, acw12, drv2000, theta0.15, mossy_stp_disabled) picks the NATURAL >=theta CA3 assembly per DG input, and that DG-selected membership is fed via the ADDITIVE `assemblies_ext` seam into the committed slow-NMDA reverberatory BTSP-formation+completion instrument (`_gap5_btsp_forms_nmda_slow_reverberatory_derisk.run_seed`). THE FIX UNDER TEST is the SEAM finding's named mechanism — an assembly-SIZE-AWARE completion inhibition realized as an E%-max FEEDFORWARD divisive-normalization CA3 basket (`ca3_ff_inhib` -> the `ca3_ff_basket` region already wired in `_build` for SELECTION; de Almeida-Idiart-Lisman 2009 / Pouille-Scanziani 2001). During the COMPLETION read DG is silent, so the `ca3->ca3_ff_basket` arm makes it a DISYNAPTIC FEEDFORWARD inhibition driven by the CUE VOLLEY (active cells -> basket -> held-out cells) whose gain SCALES with the active-population size — the same divisive-normalization companion process that made emergent-DG SELECTION robust across a >10x input range. Threaded ADDITIVELY (`ca3_ff_inhib=None` => byte-identical, no basket built) through `_build_bridge`/`run_seed`; NO other `sim/`-facing change. ARMS per (density, ff_inhib) condition: CONTROL (ca3_fb_inhib=60, ca3_ff_inhib=None, density 0.12 = the fixed readout the SEAM used) which MUST STILL FAIL (load-bearing test), vs SIZE-AWARE (sparser density {0.05-0.10} x ca3_ff_inhib {200-800}); each condition runs the full instrument — HAND-INSTALL cross-check (perfect within-assembly W, isolates the READOUT operating point from BTSP formation), BTSP one-shot plateau-gated formation (isolated per-assembly episodes), NO-PLATEAU lesion, NO-ENCODING, RECURRENCE-ZERO, permuted cue, silent-rest nocue, cross_dw/nonmem_dw, genuine-formation. Anti-cheats: EMERGENT membership (mossy-LESION collapses every assembly to size 0 -> membership is DG-derived; Jaccard vs the random-permutation set <=0.34; sizes are the emergent ~14-35, not the readout's 0.18*N pre-assigned mask); plasticity FROZEN at recall; OU OFF (the seam is deterministic); build-twice threshold-hash SEEDED; cfg.seed set explicitly; SIM_BACKEND=cupy; `attributable_to` on membership-vs-mossy-lesion, BTSP-vs-no-encoding, correct-cue-vs-permuted-cue. GO gate (6-seed): a SIZE-AWARE working point gives held_cue>=0.20 AND held_cue>=3*held_perm AND held_cue>=3*held_nocue AND held_nocue<=0.10 (BTSP-formed + genuine) on >=5/6 seeds, AND the fixed-inhib CONTROL still fails.
---

# Gap #5 completion seam — the size-aware FEEDFORWARD divisive-normalization basket is LOAD-BEARING and ELIMINATES the permuted-cue cross-talk (perm 0.13 -> 0.00, 6/6) and silences the rest state (nocue 0.25 -> 0.05), the afferent-specificity half of the seam, but a residual BISTABILITY gap remains and this is NOT gap#5 closure: a ~23-cell emergent assembly has NO recurrent-attractor operating point that is simultaneously cue-ignitable (cue>=0.20) and rest-silent (nocue<=0.10) — because cue-completion and self-ignition share the recurrent gain, and the FF-basket normalizes the AFFERENT volley but not the SELF-drive

The 2026-08-10 SEAM finding
(`2026-08-10-gap5-e2e-episodic-loop-emergent-selection-composes-but-completion-operating-point-is-assembly-size-SEAM-NEGATIVE.md`)
mapped the gap#5 composition seam: emergent-DG SELECTION composes (6/6) and BTSP FORMATION is genuine, but the slow-NMDA
+ FS-basket COMPLETION readout — a FIXED feedBACK inhibition (`ca3_fb_inhib=60`) tuned for the LARGER, UNIFORM ~72-cell
pre-assigned assemblies of the 2026-08-10 formation GO — is NON-SPECIFIC on the SMALL (~23-cell), VARIABLE emergently-selected
assemblies (perm ≈ nocue ≈ cue), fails even HAND-INSTALLED, and a recurrent-density sweep UP {0.12,0.35,0.5} made it
WORSE. It named the fix: make the completion inhibition assembly-SIZE-AWARE via a FEEDFORWARD divisive-normalization
basket that scales with the active-population size, plus SPARSER recurrence (Guzman–Jonas CA3 ~1–2%). **This finding
BUILDS that fix, and the outcome is a genuine, precisely-quantified PARTIAL: the size-aware inhibition is LOAD-BEARING
and closes the permuted-cue (afferent-specificity) half of the seam, but it does NOT reach the cue-specific bistable GO
on the small emergent assembly — because the residual is a bistability problem the FF-basket cannot touch.**

## What the size-aware FF-basket FIXES — permuted-cue specificity (load-bearing)
<!--derived-->
Enabling the E%-max `ca3_ff_basket` on the completion READ collapses the permuted-cue response to ~0 at EVERY density
and every ff gain, on the SAME ~23-cell emergent assemblies where the fixed-fb control gave perm ≈ cue. On the
seed-42 operating-point search (`research/findings/raw/_gap5_e2e/size_aware_window_s42.json`, BTSP arm; the fixed-fb
CONTROL row is the 6-seed table below and its seed-42 smoke `size_aware_smoke_s42.json`, cue 0.243 / perm 0.168 /
nocue 0.290 — non-specific):

| condition | density | ff_inhib | held_cue | held_perm | held_nocue | w_within |
|---|---|---|---|---|---|---|
| d05_ff200 | 0.05 | 200 | 0.106 | 0.000 | 0.117 | 4017 |
| d06_ff300 | 0.06 | 300 | 0.116 | 0.000 | 0.157 | 4092 |
| d06_ff400 | 0.06 | 400 | 0.118 | 0.000 | 0.040 | 4068 |
| d07_ff300 | 0.07 | 300 | 0.117 | 0.000 | 0.149 | 4167 |
| d07_ff400 | 0.07 | 400 | 0.110 | 0.000 | 0.139 | 4157 |
| d08_ff600 | 0.08 | 600 | 0.137 | 0.000 | 0.189 | 4169 |
| d08_ff800 | 0.08 | 800 | 0.124 | 0.000 | 0.179 | 4272 |
| d10_ff600 | 0.10 | 600 | 0.174 | 0.000 | 0.236 | 4423 |

- **perm collapses to ≈0.000 with the FF-basket, at every density and gain.** A permuted (non-assembly) cue no longer
  ignites the held-out cells — the divisive-normalization inhibition driven by the cue volley suppresses the
  off-assembly drive. Across 6 seeds the control-perm mean is 0.130 (per-seed 0.000–0.279) and the size-aware-perm mean
  is 0.000; `attributable_to` reports control-perm vs size-aware-perm as a genuine, non-byte-identical change — the
  FF-basket is verifiably ENGAGED, not a dead arm.
- **This is exactly the afferent-specificity function the SEAM finding predicted the FF-basket would restore** (the same
  companion process that made emergent-DG SELECTION robust across a >10× input range). It is LOAD-BEARING: the fixed-fb
  CONTROL, on the identical assemblies, stays non-specific and self-igniting (6-seed mean cue 0.246 ≈ nocue 0.245).

## What it does NOT close — the cue-vs-rest BISTABILITY window (the residual)
<!--derived-->
Across the ENTIRE grid (density 0.05–0.10 × ff_inhib 200–800, both HAND-INSTALL and BTSP arms), **no operating point
is simultaneously cue-ignitable (held_cue ≥ 0.20) and rest-silent (held_nocue ≤ 0.10)** for the ~23-cell emergent
assembly. GO = 0/(9 conditions), seed 42:
- **The only silent-rest points have a dead cue.** At d0.06/ff400 the rest is genuinely silent (nocue = 0.040) but the
  cue barely completes (cue = 0.118 < 0.20) — the sparse within-assembly recurrence is too weak for the small assembly
  to reverberate.
- **The only cue-strong points self-ignite.** At d0.10/ff600 the cue reaches 0.174 but the rest state self-ignites
  (nocue = 0.236) — a silent reset spontaneously latches the attractor high.
- **cue and nocue rise and fall TOGETHER.** They are coupled through the SAME within-assembly recurrent gain: raising
  density or W to lift the cue-completion also lifts the self-ignition, and the FF-basket (which normalizes the AFFERENT
  volley) cannot separate them because both are driven by the SELF (within-assembly) reverberation, not the afferent.
- **The HAND-INSTALL arm proves this is the readout, not BTSP, and that a STRONGER weight is the WRONG direction.** At
  d0.06/ff400 a perfect uniform W=5000 (hand-install) SELF-IGNITES MORE (nocue = 0.175) than the BTSP-formed w≈4068
  (nocue = 0.040) — a stronger within-attractor makes the rest state LESS stable, so boosting the attractor to lift the
  cue is self-defeating. genuine_formation = True on every BTSP row (w_within grows from `fused_btsp_update`, cross_dw ≈
  0), so the completion failure is a READOUT/bistability seam, not dead formation.

## 6-seed confirmation (the decisive deliverable)
<!--derived-->
6/6 seeds (42/43/44/100/101/102), OU-off, BTSP-formed + genuine, emergent-membership anti-cheat 6/6, index-space 6/6.
Artifact `research/findings/raw/_gap5_e2e/size_aware_completion_6seed.json`: **size-aware-GO 0/6, control-GO 0/6**
(`n_size_aware_go=0`, `n_control_go=0`), status NO-GO. The fixed-inhib CONTROL fails on all 6 (it self-ignites:
cue ≈ nocue, ± perm), AND the size-aware arm reaches no cue-specific bistable working point on any seed. BTSP arm,
mean over 6 seeds (per-seed cue spread shown to prove it is not a single-seed artifact):

| condition | mean cue | mean perm | mean nocue | per-seed cue (42/43/44/100/101/102) | GO |
|---|---|---|---|---|---|
| control_d012 (fixed fb=60, no ff) | 0.246 | 0.130 | 0.245 | 0.236/0.217/0.229/0.259/0.260/0.275 | 0/6 |
| sa_d06_ff400 (size-aware, silent-rest point) | 0.070 | 0.000 | 0.047 | 0.088/0.083/0.029/0.061/0.067/0.091 | 0/6 |
| sa_d08_ff400 (size-aware, cue-up point) | 0.102 | 0.003 | 0.123 | 0.122/0.113/0.085/0.084/0.082/0.128 | 0/6 |

- **The size-aware inhibition is LOAD-BEARING (it transforms the dynamics, 6/6):** vs the fixed-inhib control it
  ELIMINATES the permuted-cue cross-talk (perm 0.130 → 0.000/0.003) AND silences the rest state (nocue 0.245 → 0.047
  at the sparse d0.06 point) — the afferent-specificity + rest-silence the SEAM finding predicted. The control, on the
  identical emergent assemblies, stays non-specific and self-igniting (cue 0.246 ≈ nocue 0.245) on every seed.
- **But it does NOT open a bistable window (0/6):** turning down the recurrent gain (sparse density + FF inhibition)
  turns EVERYTHING down together — the cue-completion collapses WITH the self-ignition (cue 0.246 → 0.070). At the only
  silent-rest point (sa_d06) there is a WEAK cue-preference (mean cue 0.070 > mean nocue 0.047) — a real but
  sub-threshold completion — an order of magnitude below the held_cue ≥ 0.20 / 3× GO bar. The small assembly cannot
  build a reverberation strong enough to hold a high cue state without also self-igniting the rest state.
- emergent sizes per seed: 42[36,12,23] 43[21,25,16] 44[35,13,15] 100[20,26,19] 101[25,27,22] 102[31,25,22] — mean
  ~22 cells, the ~23-cell emergent regime; anti-cheat #1 (mossy-lesion collapse + low Jaccard) holds 6/6.

## Why this is the residual the SEAM finding named — assembly SIZE, and the concrete next lever
<!--derived-->
The contrast with the 2026-08-10 formation GO is the size-dependence, made explicit:
- **~72-cell UNIFORM pre-assigned assemblies → 6/6 cue-specific bistable GO**
  (`2026-08-10-gap5-BTSP-emergently-forms-the-slow-nmda-reverberatory-attractor-6seed-GO-preassigned-assemblies.md`):
  a larger assembly has enough within-assembly fan-in that a moderate density gives cue-completion while the rest stays
  below the ignition threshold — a wide bistable window.
- **~23-cell VARIABLE emergent assemblies → NO window at any inhibition** (this finding): the within-assembly fan-in is
  ~1/3, so the density needed for cue-completion is the density at which the rest self-ignites; the FF-basket removes
  the AFFERENT cross-talk (perm) but the SELF-drive coupling is intrinsic to the recurrent-attractor mechanism at this
  size.

**The residual is the DG producing assemblies an order of magnitude too small for a recurrent bistable attractor, and
the FF-basket is the wrong lever for a SELF-drive (not afferent) coupling.** Per THE LAW (a wall is a verdict on a
METHOD), the two named next mechanisms, both one-brain:
1. **DG detonator-GAIN to produce LARGER (~50–72-cell) assemblies** — a mossy detonator-strength / theta-threshold lever
   on the SELECTION front-end to match the completion's viable-window regime. Tension: it fights the DG's
   pattern-separation (sparse is the point), so it is a bounded lever, not a free one.
2. **INTRINSIC per-cell dendritic BISTABILITY for the COMPLETION READ** — a plateau/dAP-latched high state per held
   cell (size-INDEPENDENT) instead of a recurrent-population attractor, so the completion no longer depends on
   within-assembly recurrent gain and cue-ignition/rest-silence decouple. This is a READOUT bistability, explicitly
   NOT the two-compartment/dendritic/BDSP/burstprop deep-CREDIT-assignment rule, which is tested-NEGATIVE for hidden
   credit on spikes (`research/findings/2026-05-17-dendritic-credit-assignment-NEGATIVE.md`,
   `research/findings/2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS-deep-research.md` — the topology
   is faithful; the frozen fixed-random feedback SIGNAL is the cause) — the dendrite here holds a completion state, it
   does not carry a learning gradient. (The 2026-07-18 dendritic "learned CLOSED" completion characterization is
   ⛔ RETRACTED [self-sustaining + Wang confound]; this names the readout-bistability MECHANISM as the next build, NOT
   that prior result. Run `bash tools/before_you_build.sh` before building it.)

## Verdict
<!--derived-->
**PARTIAL (honest, load-bearing) — the size-aware FEEDFORWARD divisive-normalization basket ELIMINATES the permuted-cue
cross-talk (afferent-specificity) and silences the rest state, the half of the gap#5 completion seam it can reach
(6-seed mean perm 0.130 → 0.000, nocue 0.245 → 0.047; the fixed-inhib CONTROL stays non-specific 6/6, so the fix is
load-bearing) but does NOT reach the cue-specific bistable GO on the ~23-cell emergent assembly (size-aware-GO 0/6,
control-GO 0/6).** No inhibition operating point over density 0.05–0.10 × ff 200–800 is simultaneously cue-ignitable
(cue ≥ 0.20) and rest-silent (nocue ≤ 0.10), because cue-completion and self-ignition share the within-assembly recurrent
gain and the FF-basket normalizes the AFFERENT volley, not the SELF-drive. This is NOT gap#5 closure. It ADVANCES the
seam from "completely non-specific (perm ≈ nocue ≈ cue)" to "afferent-specific but bistability-limited," and pins the
remaining residual to assembly SIZE with two named, one-brain next levers (DG detonator-gain for larger assemblies; or
intrinsic dendritic bistability that decouples cue-ignition from self-drive).

Artifacts (SIM_BACKEND=cupy; provenance sidecar records backend + argv + git SHA):
`research/findings/raw/_gap5_e2e/size_aware_completion_6seed.json` (6-seed CONTROL vs SIZE-AWARE, OU-off, carries the
Verdict `preconditions` block), `research/findings/raw/_gap5_e2e/size_aware_window_s42.json` (seed-42 density×ff
operating-point search). Runner: `research/runners/_gap5_size_aware_completion_derisk.py` (reuses the emergent selection
via `_gap5_emergent_end_to_end_episodic_loop_derisk.emergent_assemblies` + the completion instrument via
`_gap5_btsp_forms_nmda_slow_reverberatory_derisk.run_seed`). The `ca3_ff_inhib` seam is ADDITIVE (byte-identical when
None): `research/runners/_gap5_btsp_forms_nmda_slow_reverberatory_derisk.py`. NO `sim/` edit.

### Sources
- de Almeida L., Idiart M., Lisman J.E. *A second function of gamma frequency oscillations: an E%-max winner-take-all mechanism selects which cells fire.* J. Neurosci. 29:7497–7503 (2009).
- Pouille F., Scanziani M. *Enforcement of temporal fidelity in pyramidal cells by somatic feed-forward inhibition.* Science 293:1159–1163 (2001).
- Guzman S.J., Schlögl A., Frotscher M., Jonas P. *Synaptic mechanisms of pattern completion in the hippocampal CA3 network.* Science 353:1117–1123 (2016).
- Bittner K.C., Milstein A.D., Grienberger C., Romani S., Magee J.C. *Behavioral time scale synaptic plasticity underlies CA1 place fields.* Science 357:1033–1036 (2017).
- Wang X-J. *Probabilistic decision making by slow reverberation in cortical circuits.* Neuron 36:955–968 (2002).
