---
type: finding
status: contributing
date: 2026-08-10
mechanism: ca3-completion
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: The end-to-end episodic loop is COMPOSED on ONE spiking substrate (n_ca3=2000) by feeding EMERGENTLY-SELECTED CA3 assemblies into the committed slow-NMDA reverberatory BTSP-formation instrument via an ADDITIVE `assemblies_ext` seam (byte-identical when None; git diff shows the else-branch is the original permutation path). SELECTION arm - the R1 recovered-at-scale sparse mossy-detonator (d0.02/w3000, acw12, drv2000, theta0.15, mossy_stp_disabled) drives n_patterns=3 distinct DG inputs and reads the NATURAL >=theta CA3 assembly per input (global indices). FORM+READ arm - tonight's `_gap5_btsp_forms_nmda_slow_reverberatory_derisk.run_seed` on the emergent membership across a density sweep {0.12,0.35,0.5} x wmax {5000,9000}: hand-install cross-check, BTSP isolated-episode formation, no-plateau lesion, no-encoding, recurrence-zero, permuted cue, silent-rest nocue, cross_dw/nonmem_dw, OU off/on. The DECISIVE decomposition is HAND-INSTALL vs BTSP - because the hand-install arm (a perfect within-assembly W, no plasticity) ALSO fails cue-specificity on these assemblies, the seam is on the COMPLETION READOUT operating point, NOT on BTSP formation. Anti-cheat #1 (emergent membership) - sizes are the emergent ~13-35 (NOT the readout's 0.18*N=360 pre-assigned size); Jaccard vs the random-permutation set ~0.003-0.023; mossy-LESION (dg->ca3 weight 0) collapses every assembly to size 0 (DG->CA3 detonation load-bearing = the membership is DG-derived). Index-space verify - the selection bridge and the readout/formation bridge place CA3 at identical global indices (same region sizes), 6/6. Decomposed quantities per (density, wmax) - held_cue, held_perm, held_nocue, w_within, cross_dw, no_encoding_held_cue, recurrence_zero, genuine_formation, btsp_noplateau. attributable_to per working point (BTSP-formed vs no-encoding; correct-cue vs permuted-cue). Frozen plasticity at recall; cfg.seed set explicitly; build-twice threshold-hash SEEDED. SIM_BACKEND=cupy. GO gate (6-seed) - held_cue>=0.20 AND held_cue>=3*held_perm AND held_cue>=3*held_nocue AND held_nocue<=0.10.
---

# Gap #5 end-to-end episodic loop — the three individually-GO pieces do NOT naively compose: emergent-DG SELECTION composes (6/6, membership genuinely DG-selected), but the slow-NMDA + FS-basket COMPLETION readout is NON-SPECIFIC on the SMALL, VARIABLE (~23-cell) emergently-selected assemblies (perm ≈ nocue ≈ cue), FAILS even HAND-INSTALLED (so the seam is the completion readout, NOT BTSP formation), and a recurrent-density sweep makes it WORSE — the completion operating point is assembly-SIZE-dependent and tuned for a LARGER, UNIFORM regime the emergent DG does not produce

Three individually-GO gap#5 pieces were de-risked in ISOLATION: emergent-DG assembly SELECTION
(`2026-07-19-gap5-emergent-DG-SELECTION-de-risked-GO-6seed-mossy-detonator-stable-separated.md` +
recovered-at-scale `2026-07-21-gap5-emergent-DG-SELECTION-recovered-at-scale-sparse-detonator-GO.md`, core criteria
6/6 at n_ca3=2000), BTSP one-shot FORMATION of the slow-NMDA reverberatory attractor
(`2026-08-10-gap5-BTSP-emergently-forms-the-slow-nmda-reverberatory-attractor-6seed-GO-preassigned-assemblies.md`, 6/6
on PRE-ASSIGNED assemblies), and the slow-NMDA reverberatory + FS-basket COMPLETION readout (`483587c0b` /
`2026-08-10-gap5-somatic-slow-nmda-reverberatory-attractor-...`; peer readout
`2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`). This is the INTEGRATION build: replace the
PRE-ASSIGNED assemblies tonight's formation used with EMERGENTLY-SELECTED ones and run the same committed instrument.
**The integration DOES NOT compose cleanly, and the seam is precisely mapped and 6-seed robust.**

## Why this was genuinely un-composed (STEP 1 checked)
<!--derived-->
Tonight's BTSP-forms-slow-NMDA GO formed the attractor on PRE-ASSIGNED (random-permutation) ~72-cell assemblies; its
own "Honest scope" names emergent SELECTION as the open front-end. The one prior end-to-end chain
(`2026-07-21-gap5-emergent-DG-store-complete-GO-chain-demonstrated-SWR-2assembly-boundary.md`) read completion through
the DENDRITIC/bistable path whose "learned CLOSED" characterization was later RETRACTED (self-sustaining + Wang
confound), NOT through the current standing slow-NMDA reverberatory readout. So "emergent selection -> tonight's
slow-NMDA reverberatory formation + completion" was the untested composition. It is built here on ONE spiking substrate
via an additive `assemblies_ext` seam (byte-identical when off), NO `sim/` edit.

## What COMPOSES — emergent-DG SELECTION (anti-cheat #1 passes 6/6)
<!--derived-->
The DG genuinely selects the assembly EMERGENTLY; it is NOT a hand-set/pre-assigned mask (6/6 seeds):
- **sizes are the emergent regime** — mean 22.7 cells (per-seed means 18.3–25.7; individual assemblies 13–35) vs the
  readout's pre-assigned 0.18*N = 360-cell mask.
- **nearly disjoint from the pre-assigned set** — Jaccard vs the random-permutation the readout would use ≈ 0.003–0.023
  on every assembly/seed.
- **mossy-LESION collapses it** — set the dg->ca3 detonator weight to 0 and every assembly drops to size 0 (all 6
  seeds: [0,0,0]); the DG->CA3 detonation is load-bearing, so the membership is DG-derived, not hand-assigned.
- **index-space verify 6/6** — the selection bridge and the readout/formation bridge place CA3 at identical global
  indices (same region sizes), so the emergent indices refer to the same physical CA3 cells.

## What does NOT compose — the COMPLETION READOUT is non-specific on the small, variable emergent assemblies
<!--derived-->
On the emergently-selected assemblies (~23 cells at n_ca3=2000), the completion is NON-SPECIFIC at EVERY density and
every BTSP ceiling — a permuted cue (perm) and even NO cue (nocue) ignite the held-out cells as strongly as the correct
cue. **Crucially this holds for the HAND-INSTALL arm too** (a perfect within-assembly W, zero plasticity), so it is
NOT a BTSP-formation failure. **BTSP-completion GO = 0/6 (OU-off, 6 seeds × density × wmax).** OU-on was NOT run: the
non-specificity is DETERMINISTIC (present with zero membrane noise, and even HAND-INSTALLED), so OU membrane noise can
only ADD self-ignition, never recover cue-specificity — declared as a scoped `disabled_process` in the run's verdict.
6-seed means (OU-off, mean over 6 seeds × 2 wmax):

| density | arm | held_cue | held_perm | held_nocue | cross_dw |
|---|---|---|---|---|---|
| 0.12 | handinstall | 0.299 | 0.307 | 0.294 | — |
| 0.12 | btsp | 0.292 | 0.292 | 0.284 | −0.016 |
| 0.35 | handinstall | 0.436 | 0.456 | 0.460 | — |
| 0.35 | btsp | 0.435 | 0.453 | 0.462 | +0.004 |
| 0.50 | handinstall | 0.469 | 0.476 | 0.488 | — |
| 0.50 | btsp | 0.469 | 0.476 | 0.467 | +0.000 |

- **perm ≈ nocue ≈ cue everywhere** — the attractor fires but is not cue-selective and SELF-IGNITES. Even at the
  lowest density (0.12) perm (0.292) EQUALS cue (0.292) — no residual specificity — while nocue (0.284) shows the
  held-out cells fire almost as much with NO cue at all (near-full self-ignition); every arm sits far below the 3× GO
  bar. `attributable_to` reports correct-cue vs permuted-cue as NON-attributable (the control equals or exceeds the
  treatment).
- **the density sweep makes it WORSE, not better** — raising CA3 recurrent density to give the small assembly
  within-assembly fan-in monotonically drives cue, perm AND nocue UP together toward ~0.47 (the whole assembly firing);
  by density 0.5 perm (0.476) EXCEEDS cue (0.469). Higher density adds cross-assembly cross-talk + self-ignition — the
  WRONG direction. There is no (density, wmax) operating point where the emergent assembly completes cue-specifically.
- **BTSP FORMATION is genuine and weight-specific** (that piece is intact) — `no_encoding` held_cue = 0.000 max (the
  attractor is load-bearing for firing), `btsp_noplateau` = 0.000 max / w_within=1.5 (plateau-gated one-shot, not mere
  co-fire; formation collapses without the plateau), `recurrence_zero` = 0.000 max (completion is the reverberation),
  `cross_dw ≈ 0` (−0.042..+0.011), `w_within` tracks `btsp_w_max`, `genuine_formation`=True on every row. BTSP writes
  the correct weight-specific attractor; the READOUT cannot read it cue-specifically at this assembly size.

## The seam (quantified) and the named fix — per THE LAW, a characterized boundary, not a wall
<!--derived-->
**The seam = the slow-NMDA + FS-basket completion OPERATING POINT (`ca3_fb_inhib=60` assembly-selective inhibition +
recurrent density + recall drive) was implicitly matched to the LARGER, UNIFORM ~72-cell pre-assigned assemblies
tonight's GO used; it does NOT give cue-specificity on the SMALL (~23), VARIABLE assembly-size regime the emergent DG
produces.** Which piece's output does not match the next piece's required input: the emergent-DG SELECTION output is a
sparse (~1%) ~23-cell assembly (an order of magnitude smaller than the completion's tuned ~72-cell regime and highly
size-variable across the 3 co-stored assemblies, 13–35 cells), while the slow-NMDA completion's fixed global FS-basket
inhibition gain needs a larger, ~uniform active population to give a stable silent-low / cue-ignitable-high bistable
window. The FS-basket inhibition is a fixed constant proxying the homeostatic E/I balance that in biology SCALES with
the active-population size — the "operating point IS the mechanism / companion process proxied by a constant" pattern
(a fixed `fb_inhib=60` is right for one assembly size and wrong for another; density, the other knob, moves specificity
the wrong way).

**Named fix (the NEXT build — not built here):**
1. **Make the completion operating point assembly-SIZE-AWARE (principled, biological).** SPARSER recurrence (higher
   density HURT — go sparser, Guzman–Jonas CA3 recurrent ~1–2%) + assembly-selective FS-basket inhibition SCALED to the
   (small) emergent assembly size + a size-matched `recall_k_thresh`. Better still, drive the CA3 read-time basket
   FEEDFORWARD from the afferent cue volley (as `ca3_ff_inhib` already does for SELECTION: de Almeida–Idiart–Lisman
   2009 / Pouille–Scanziani 2001 divisive normalization) so the inhibition scales with the active-population size
   automatically → a size-invariant cue-ignitable bistable window. This is the same divisive-normalization companion
   process that made SELECTION robust across a >10× input range.
2. **OR make the DG mossy-detonator produce LARGER, more-UNIFORM assemblies** (a gain / detonator-strength lever) to
   match the completion's tuned regime — but this fights the DG's pattern-separation (sparse is the point), so (1) is
   the more biological fix.
Either closes the seam; both are one-brain operating-point work. Then re-run the loop with the size-aware completion.

## Verdict
<!--derived-->
**INTEGRATION-SEAM (honest negative that MAPS the seam) — the three pieces do NOT naively compose; this is NOT gap#5
closure and NOT a composed episodic loop.** Emergent-DG SELECTION composes (6/6 emergent-membership anti-cheat,
Jaccard ≈ 0.003–0.023 vs pre-assigned, mossy-lesion collapses to 0, index-space 6/6). BTSP FORMATION is genuine and
weight-specific (no-plateau 0, no-encoding 0, recurrence-zero 0, cross_dw ≈ 0, genuine 6/6). The slow-NMDA + FS-basket
COMPLETION readout does NOT give cue-specificity on the ~23-cell emergently-selected assemblies — perm ≈ nocue ≈ cue,
BTSP-completion GO 0/6 (OU-off; OU-on not run — the seam is deterministic) — and it fails HAND-INSTALLED too, so the
seam is the completion operating
point's dependence on assembly SIZE, not BTSP. The density sweep worsening (specificity monotonically degrades as
density rises) rules out density as the fix. The precise, 6-seed-robust map of the one remaining seam
(assembly-size-aware completion) with the biological fix named is the deliverable.

**Reproducibility note (honest):** the emergent assembly MEMBERSHIP has minor run-to-run variation (a few cells per
assembly — GPU matvec reduction non-determinism the per-neuron-threshold seeding does not cover; the build-twice
threshold-hash is SEEDED). The SEAM conclusion is robust to it: every seed, every density, the ~13–35-cell regime
completes NON-specifically. The size REGIME, not the exact membership, is the load-bearing quantity.

Artifacts (SIM_BACKEND=cupy; provenance sidecar records backend + argv + git SHA):
`research/findings/raw/_gap5_e2e/e2e_6seed.json` (6-seed, OU-off, density×wmax sweep; carries the Verdict
`preconditions` block). Composition runner:
`research/runners/_gap5_emergent_end_to_end_episodic_loop_derisk.py`. Reused instrument (additive `assemblies_ext`,
byte-identical when None): `research/runners/_gap5_btsp_forms_nmda_slow_reverberatory_derisk.py`. NO `sim/` edit.

### Sources
- Bittner K.C., Milstein A.D., Grienberger C., Romani S., Magee J.C. *Behavioral time scale synaptic plasticity underlies CA1 place fields.* Science 357:1033–1036 (2017).
- Milstein A.D., Li Y., Bittner K.C., et al. *Bidirectional synaptic plasticity rapidly modifies hippocampal representations.* eLife 10:e73046 (2021).
- Wang X-J. *Probabilistic decision making by slow reverberation in cortical circuits.* Neuron 36:955–968 (2002).
- de Almeida L., Idiart M., Lisman J.E. *A second function of gamma frequency oscillations: an E%-max winner-take-all mechanism selects which cells fire.* J. Neurosci. 29:7497–7503 (2009).
- Pouille F., Scanziani M. *Enforcement of temporal fidelity in pyramidal cells by somatic feed-forward inhibition.* Science 293:1159–1163 (2001).
- Guzman S.J., Schlögl A., Frotscher M., Jonas P. *Synaptic mechanisms of pattern completion in the hippocampal CA3 network.* Science 353:1117–1123 (2016).
