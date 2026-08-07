# Source-monitoring RECALL-SIDE: CA3-attractor-competition scoping (2026-08-07)

Research round after TWO encoding-side de-risks (hetero-depression `8aca3c62`, conjunctive-tag `1a5d2db6`) both NO-GO,
converged: a SOURCE-BLIND recall drive over a fully-shared core can't separate co-resident sources. The wall is
RECALL-SIDE. No code yet; this is the pre-build spec + the joint-knob risk + demoted candidates.

## Architecture fact (load-bearing, verified in code)
All pathways are strictly FEEDFORWARD: `episode→source_memory_{s}` (the ONLY learned weights), fixed
`source_afferent→memory` (encoding-only, ZERO at recall = the honesty guard), `memory→apfc/acc`. `internal_density=0`,
no recurrence, NO lateral connection between the 3 source-memory pops (each n=12). Each pure-source pattern = shared
`core` (k=overlap·12) ∪ source-unique cells; pattern[3] is a mixed seen+heard episode that extra-potentiates
`core→seen/heard`, making **`self_generated` the structurally weakest source** (lower core weight, never mixed-
reinforced → negative margin when cued). The ONLY recall-time signal distinguishing the correct source: its UNIQUE
cells are active in the cue while rivals' unique cells are silent (recall is episode-only). Quantity to move is
unchanged (`min_s M_s ≥ 0.15` AND `> min_s L_s`); locus moves from fan-out weights to READ-OUT DYNAMICS.

## Why all 6 prior levers failed (convergent): all were LINEAR, per-cell, FEEDFORWARD
4 recall-activity levers (fair/blanket GABA = anti-divisive per-cell WTA; own-gain = saturates; synaptic scaling =
equalizes rate; symmetric lateral GABA = rich-get-richer) + 2 encoding-weight levers (hetero-depression redistributes
the shared core's burden; conjunctive-tag re-mixed by source-blind recall). Whatever differentiation is placed at
encoding is re-mixed the instant a source-blind drive reactivates the shared core. Residual = a NONLINEAR,
ATTRACTOR-LEVEL, recall-time competition — a locus none of the 6 touched.

## Proposed mechanism: CA3-style attractor competition among the source-memory assemblies
Add (fixed-weight, symmetric across the 3 sources, scaled by ONE gain `g_comp`, NO source-specific term):
- within-population recurrent excitation (each source_memory an autoassociative attractor; NMDA-carried);
- between-population lateral inhibition (each source_memory drives a shared inhibitory pool suppressing the other 2).
Assembly IDENTITIES are pre-defined region memberships (no formation/learning); discrimination STILL comes only from
the learned `episode→source` fan-out. Biology: Rolls CA3 recurrent-collateral autoassociator (completion + feedback-
inhibition WTA at the ATTRACTOR level, Rolls Hippocampus 2013; Rolls & Treves); Buzsáki 2006 (CA3 collateral matrix =
ideal autoassociative attractor); Norman & O'Reilly 2003 CLS completion-vs-competition. Matches the project's own
banked prescription: `2026-05-31-DG-composition-NULL` — "DG separates, CA3 COMPLETES/stabilizes; separation necessary
but not sufficient — within-concept reliability needs CA3 completion."

### Why it escapes the source-blind-reactivation trap (mechanistic, not optimism)
It does NOT try to unmix the fan-out. It exploits the one recall-time asymmetry: source_memory_s gets COINCIDENT drive
from core+uniq_s; rivals get core-ONLY (less, less coincident). Three nonlinear effects separate at the ASSEMBLY
level: (1) supralinear recurrent/NMDA amplification → correct assembly crosses its attractor basin boundary and
latches high; core-only rivals stay low (a basin crossing, not a per-cell threshold race — exactly what symmetric-
GABA lacked); (2) temporal priority: uniq coincidence gives the correct assembly a recurrent head-start; (3) lateral
inhibition from the latched winner quenches rivals to ~0 within the read window → `M_s = winner_rate − ~0` is large.
Margin = the nonlinear attractor-state GAP (high vs low), not the small linear input difference the 6 levers fought.

## Cheapest single-variable de-risk
Extend `_laneC_source_monitor_overlap_sweep.evaluate_overlap` (honest v6 recall + `reset_dynamical_state` per recall +
`_source_margin`; reproduces the 1/5 NO-GO). NO `sim/` edit — add recurrent-E within-region + lateral-I cross-region
pathways via the bridge build (RegionPathway/`internal_density`), reused by config as the CA3 GO reused kernels. ONE
knob `g_comp` (fixed-RATIO recurrent-E + lateral-I, tied → single variable). **`g_comp=0` ≡ overlap NO-GO byte-
identically** (null control; the competition-lesion arm `L` is `g_comp=0`). Anti-cheats (ALL hold): (a) g_comp=0
byte-identical; (b) HONESTY: source-afferent current=0 AND firing=0 at recall, AND the competition module is
PARAMETER-SYMMETRIC across sources (no source term → structurally can't encode which is cued); non-vacuity: a forced
source afferent at recall moves the winner; (c) ⭐ **`all_dominant_correct` stays True on EVERY source incl. the
weakest** — a hard WTA that silences 2 pops REGARDLESS of correctness trivially maximizes margin = a cheat, caught
here; (d) no source's own-recall rate collapses. GO (frozen v6): `min M > min L` AND `min M ≥ 0.15`, calib 650/651 →
dev 652/653/654 → held-out 655/656/657. numpy, minutes/seed.

## Honest closability + the JOINT-knob risk
Correct next locus + first nonlinear recall-side lever. BUT honestly a JOINT storage+recall change (substrate has NO
recurrence/lateral-I — both must be ADDED). Real risk: attractor competition amplifies the LARGEST-total-input
assembly; when the mixed-episode asymmetry makes `core→rival` exceed the weakest source's `core→correct+uniq`, WTA
amplifies the WRONG source → `all_dominant_correct` FAILS (rich-get-richer, now at attractor level). The uniq
coincidence head-start overcomes it only ABOVE a separation floor = the 2026-05-31 boundary transferred exactly. ⇒
**plan for a JOINT knob (competition gain × separation/overlap), not recall-only** — if single `g_comp` can't thread
(winner-bias vs weak-margin), co-tune the storage-side separation (larger uniq fraction / sparser core). Scale: the
2026-07-14 CA3 arc found LEARNED-completion scale-bounded/seed-fragile at 150 neurons (here n=12) — BUT that was
held-out-member COMPLETION with a formed attractor; this is COMPETITION among PRE-DEFINED pops (a rank, no formation)
→ strictly easier, formation-fragility removed by construction; the 150-neuron completion wall may not transfer (the
key empirical uncertainty the de-risk resolves cheaply). Verdict: closable-LOOKING + clearly the right method; the
single-`g_comp` de-risk discriminates in minutes/seed — a GO closes it; `all_dominant_correct=False` ⇒ go joint.

## Demoted candidates
- Theta-gamma phase separation: the honest form needs the reader to know which gamma slot = cued source = a phase-
  encoded source-label LEAK; + the project hit a decisive 5-architecture theta-gamma ceiling (2026-05-20). Fallback only.
- Recall-time divisive normalization ALONE = the failed fair-inhibition lever; meaningful only as the lateral-I half
  of the attractor mechanism above (folded in).
