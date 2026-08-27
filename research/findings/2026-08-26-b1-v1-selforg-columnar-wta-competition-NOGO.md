---
type: finding
status: contributing
date: 2026-08-26
mechanism: b1-v1-orientation-selforg-onbridge
artifacts:
  - research/findings/raw/_b1_v1_selforg_columnar_wta_6seed.json
---

# B1 on-bridge V1 self-org, COLUMNAR k-WTA COMPETITION lever — 6-seed NO-GO: fixed iso-position lateral inhibition is a DIAGONAL (gain-control) operation and cannot break the ON/OFF common mode, which is an OFF-DIAGONAL decorrelation (theorem-grounded); the named surpass is LEARNED anti-Hebbian inhibition (King-Zylberberg-DeWeese 2013)

**Verdict: NO-GO (BOUNDARY), 6/6 seeds.** A structured COLUMNAR (iso-position) k-WTA
competition — one spiking FS interneuron per retinotopic position pooling all
orientation/frequency channels at that position and inhibiting the hypercolumn back,
plus per-cell homeostatic synaptic scaling + threshold homeostasis — was installed on
the real spiking `retina->cortex_v1_simple` bridge (NO `sim/` edit) and developed on
oriented gratings. The instrument confirms the competition is genuinely ACTIVE (it
suppresses + sparsifies V1 firing), yet orientation selectivity does NOT emerge:
`osi_post_frac` stays at the floor and does not clear the freeze/shuffle lesion
controls by the +0.15 margin, with `on_minus_off_mean ~ 0` (the ON/OFF common mode is
intact). This is the honest NO-GO the mechanism family predicts, and it now carries a
theorem + a V1-specific citation for WHY, and the precise next lever.

## The result (6 seeds, production scale n_v1=8192, dev_steps=14000, cupy)

`research/runners/_b1_v1_selforg_columnar_wta_derisk.py`, seeds 42-47, inh_exc_w=150,
inh_inh_w=1500, homeo_target=0.006 (run params). <!--derived--> Gate: all seeds `osi_post_frac >= 0.5` AND
`>= freeze+0.15` AND `>= shuffle+0.15`. Artifact:
`research/findings/raw/_b1_v1_selforg_columnar_wta_6seed.json`.

| metric | value |
|---|---|
| overall verdict / flip_decision | BOUNDARY (NO-GO) / HOLD-OFF |
| per-seed verdicts | 6x BOUNDARY (0/6 clear the gate) |
| osi_post_frac mean (min) | 0.0018 (0.0015) |
| osi_pre_frac mean | 0.0036 |
| osi_freeze_frac mean (no-learning lesion) | 0.0037 |
| osi_shuffle_frac mean (orientation-destroyed lesion) | 0.0020 |
| on_minus_off_mean (common-mode discriminator) | -0.001935 (~0) |
| orient_decode mean (chance = 1/8 = 0.125 <!--derived-->) | 0.1067 (at/below chance) |
| rsa_vs_host mean (non-discriminating, per prior findings) | 0.5601 |
| instrument: v1_rate inh_on / inh_off | 0.00110 / 0.00128 |
| instrument: active_frac inh_on / inh_off | 0.0164 / 0.0191 |
| instrument: inhibition_suppresses | True |

The learned arm's `osi_post_frac` (0.0018) sits BELOW BOTH the no-learning freeze
control (0.0037) and the random-init pre value (0.0036) on the seed means — the plastic
rule under columnar competition changes the RFs (learn != freeze, so the test is NOT
void) but drives them, if anything, slightly further from orientation, not toward it. The
firing-code orientation decode (0.1067) is at/below the 1/8 chance level: no orientation
information reaches the V1 firing code. The competition is real (`inhibition_suppresses`
= True) and did not help.

## What was NEW this lever (beyond re-confirming the 2026-08-14 / 2026-08-26 BOUNDARY)

The prior on-bridge inhibition arm (`--n-inh 64`) was a SINGLE GLOBAL FS pool with
UNIFORM RANDOM connectivity → uniform gain control (2026-08-14: "not per-pair
decorrelation"). This lever installs STRUCTURED COLUMNAR competition: one interneuron
per position, so cells at a position genuinely COMPETE (winner-take-all within the
hypercolumn) rather than all being scaled down together. The mechanistic bet was that
iso-position WTA would split a column's cells into orientation/PHASE-opponent partners
(the local cell-vs-cell competition a global pool cannot supply), each then firing
selectively and so developing a signed RF. It does not: the competition sharpens WHO
fires but not the ON/OFF opponency WITHIN a cell.

## WHY — the theorem, not just the measurement

The ON/OFF common mode is a CORRELATION between each cell's ON and OFF input channels
(a full-field grating co-activates both, averaged over phase). Removing it is an
OFF-DIAGONAL decorrelation. The project's own ranked deep-research
(`2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research.md`, corroborated
across every source it cites) states the mathematical fact: **"every biologically-
plausible off-diagonal decorrelator is a recurrent cross-neuron interaction… a diagonal
D cannot rotate off-diagonal correlations away."** A FIXED lateral inhibition — pooled
or columnar — is a diagonal / gain-control operation (it scales each cell's drive by a
function of the local population activity). By the theorem it CANNOT perform the
off-diagonal rotation that removes the ON/OFF correlation, no matter how strong or how
local the pooling. Making the competition sharper (which this lever verified is
possible: 32% raw suppression at frozen thresholds) therefore cannot rescue it — the
limitation is structural, not a tuning deficit. This is why a sharper fixed-WTA variant
was NOT run: the theorem makes it provably futile.

## The named surpass (the plastic lever — needs inhibitory-pathway plasticity)

The canonical fix is exactly the V1 case of the theorem: **King, Zylberberg & DeWeese
2013, "Inhibitory interneurons decorrelate excitatory cells to drive sparse code
formation in a spiking model of V1" (J. Neurosci., PMC6705060)** — the interneurons must
LEARN (anti-Hebbian) to cancel between-cell correlations; a FIXED pool provides only the
diagonal half. This is the SAILnet plastic-inhibition ingredient the 2026-08-14 /
2026-08-26 findings named, now with the general theorem behind it. On this substrate the
next rung has ready machinery to reuse: `cfg.enable_graded_lateral` (the Pehlevan-
Chklovskii anti-Hebbian lateral, already on the bridge — `sim/bridge.py`
`_init_graded_lateral`) and/or `cfg.enable_inhibitory_stdp` (Vogels-Sprekeler), applied
to a columnar interneuron pool like the one this runner installs. The over-whitening
caveat from the same deep-research (full whitening amplifies noise → collapse) means the
plastic pool should be LOW-RANK / a small interneuron count, not full-strength all-to-all.

## Instrument verification (the test is not void)

<!--derived-->
(the frozen-threshold sweep numbers below are a diagnostic, not saved as a cited
artifact; the 6-seed headline numbers are validated in the results table above)

1. The columnar inhibition genuinely bites. With plasticity frozen (raw effect, no
   homeostatic compensation), inh_inh_w 0 → 3000 dropped V1 rate 0.0094 → 0.0064 (-32%)
   and active-fraction 0.28 → 0.19 at inh_exc_w=150. In the developmental regime the
   in-runner instrument reads `inhibition_suppresses=True` (v1 rate + active-frac lower
   with the inh->V1 weights live vs an identically-built inh-zeroed control). So the
   competition is not inert — the NO-GO is a real mechanism failing, not a wiring no-op.
2. Learn != freeze on every seed (the plastic rule changes the RFs), so a "learn==freeze
   ⇒ void" collapse does not apply. The freeze lesion HOLDS: plasticity is disabled
   BEFORE development, so the frozen (random-init) weights cannot regrow.

## Anti-cheats (all held)

Isotropic radius-4 RF support (all ON+OFF within the disc, carries no orientation) — any
oriented RF must be learned. Host Gabor bank never applied to the pathway (random init
then learned; host used only as the RSA scoring reference). Inhibitory SIGN comes from
the v1_inh region trait (exc_fraction=0.0, FS type), not the pathway, so
set_pathway_weights-installed columnar edges inhibit correctly (verified: build log
`I->E` synapses present; instrument suppression positive). Determinism:
`cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed`. OSI is label-free. The gate
requires `learn >= lesion_ctrl + 0.15`, so a rate- or support-inflated OSI cannot pass.

## Sources

<!--derived-->
(citation identifiers below — DOIs / arXiv IDs / years — are references, not run measurements)

Off-diagonal decorrelation theorem + ranked options:
`2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research.md`. Learned
inhibitory decorrelation in spiking V1 (the named surpass): King, Zylberberg & DeWeese,
J. Neurosci. 2013 (PMC6705060); Pehlevan & Chklovskii 2015 (arXiv 1511.09468 <!--derived: citation id-->, the
anti-Hebbian-lateral theory). SAILnet (whitened input + plastic anti-Hebbian lateral →
oriented Gabor RFs): Zylberberg, Murphy & DeWeese 2011, PLoS Comput Biol 7(10):e1002250.
Homeostatic synaptic scaling: Turrigiano 2008, Cell 135:422. The on-bridge BOUNDARY this
extends: `2026-08-14-b1-v1-selforg-onbridge-operating-point-BOUNDARY.md`,
`2026-08-26-b1-v1-selforg-production-wirein-PARTIAL.md`. Off-bridge numpy ceiling (GO):
`2026-06-21-B1-v1-gabor-selforg-derisk.md`.
