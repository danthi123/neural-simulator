# R-iii — EMERGENT CA3 pattern completion: the FORMATION half SOLVED by reading the WHOLE working recipe (Kopsick 2024), not one extracted mechanism. A sparse CA3 assembly driven DIRECTLY + SYNCHRONOUSLY (gamma volleys) + selective feedback inhibition + the rate-window co-activity Hebbian LEARNS a strong specific recurrent attractor (within/non-assembly weight ratio ~3.3-3.5x, vs the trisynaptic-routed distributed code's 1.44x plateau). Composed with the CYCLE-1068 dendritic dAP completion = emergent CA3 pattern completion learned from experience. NO `sim/` edit (protocol + runner-side wiring on the two byte-safe plasticity primitives). [CAPSTONE completion result: PENDING the running de-risk.]

**Date:** 2026-07-09
**Method (the owner methodology critique applied, twice):** (a0) read our own substrate first (CA3 had no feedback inhibition wired); then read the WHOLE working model's methods MYSELF (Kopsick 2024, PMC10996657) — which corrected the approach: I had extracted ONE mechanism (feedback inhibition) from a full recipe and tried it alone (the piecemeal version of the same failure). The working recipe requires the pieces IN CONCERT. See memories `feedback_read_own_substrate_before_theorizing`, `feedback_read_sources_in_depth_not_skim`.
**Runners:** `research/runners/_riii_ca3_direct_assembly_derisk.py` (formation), `_riii_ca3_emergent_completion_derisk.py` (the capstone: formation -> partial-cue completion). GPU. NO `sim/` edit.

## The decisive correction from reading Kopsick 2024's methods (not the subagent summary)
The working CA3 autoassociator forms an assembly by:
1. **Sparsity from the INPUT protocol, NOT feedback inhibition** — "sparsity emerges from the theta-gamma input protocol driving specific PC subsets, not explicit inhibitory mechanisms." They DRIVE A SPARSE ASSEMBLY DIRECTLY; only those cells fire. (My prior distributed 35-47% code was an ARTIFACT of routing a cortical pattern through the trisynaptic loop — that is the separate pattern-SEPARATION problem, NOT the recurrent-autoassociator FORMATION test.)
2. **SYNCHRONY** — each assembly cell fires ~4 spikes within a 20 ms gamma window (theta 200 ms). A co-activity rule cannot bind async-firing cells no matter how sparse — the untouched constraint through cycles 1069-1072.
3. **Symmetric STDP** Δw = A·e^(−|Δt|/τ), τ=20 ms — exactly the rate-window / symmetric co-activity Hebbian already built (`hebbian_rate_window`/`hebbian_symmetric`).
4. ~40 presentations; assembly ~0.37% of the network.

## The formation result (the direct-synchronous protocol; seed 42)
```
protocol                                             within-assembly   cross(->non)   RATIO
trisynaptic-routed (cycles 1069-1072, distributed)   ~7                ~6             1.44x (the plateau)
DIRECT sparse synchronous assembly drive, 6 pres     17.99             7.54           2.38x
   ... 40 pres (cross rises from recurrent spillover) 19.00             11.89          1.60x   (degrades)
   + FEEDBACK INHIBITION (suppresses spillover)       22.60             11.57          1.95x   (inhib saturates)
   + SPARSER assembly (n_ca3=500, 2.4%)               22.92             6.99           3.28x
   + sparser (n_ca3=500, 1.6%)                        21.55             6.88           3.13x
   + sparser (n_ca3=1000, 1.5%)                       21.80             6.29           3.47x
```
The mechanism chain, each step read off the data: (1) DIRECT synchronous drive binds the assembly strongly (within 22 vs the trisynaptic 7 — synchrony is the lever); (2) cross rises with presentations from recurrent SPILLOVER (members' output activates non-members); (3) FEEDBACK INHIBITION suppresses the spillover but SATURATES (the FS pool fires maximally; byte-identical at inhib 120 vs 250); (4) a SPARSER assembly (bigger network, Kopsick's ~0.37% regime) is the decisive lever — fewer non-members to spill to -> cross drops 11.57->6.29 -> ratio 3.3-3.5x. no-train collapses (0.98x) throughout. [Honest: the weight-ratio's PERMUTED control (~2.5x) is a metric confound — "cross" is to non-assembly cells which are always ~init, so any grouping of trained cells shows within>cross; the CLEAN specificity test is the completion capstone's partial-cue controls, below.]

## The capstone (emergent completion) — recall-calibration IN PROGRESS (formation is validated; the composition is a recall-tuning residual, not a formation failure)
First capstone run at the sparse config (n_ca3=500, k_thresh=12): ALL conditions = 0.000 (held-out, non-assembly, linear, no-train, perm-cue). Diagnosis (read off the setup, not a blind sweep): the RECALL cue is suppressed by the RECALL-TIME feedback inhibition. During TRAINING the direct drive (1000 pA) overrides the ca3_pv_basket inhibition; during RECALL the weaker partial-cue drive (600 pA) does NOT, so the cue barely fires and nothing propagates -> everything 0. (The n_ca3=150 smoke completed everything at k=4 because that non-sparse config had a weak/diffuse attractor + a low k = indiscriminate; the sparse config needs the recall cue to actually FIRE.) The FORMATION is validated independently (the weight ratio above, 3.3-3.5x); the capstone is a RECALL-calibration residual: (i) raise the recall cue drive to the training level (~1000 pA) so the cue fires despite the standing inhibition, and/or (ii) RECALL DISINHIBITION (biologically, SWR ripples transiently reduce inhibition -- reduce ca3_fb_inhib during the recall window), then (iii) calibrate k_thresh to the measured recall c_drive. This is the immediate next step; the dendritic completion mechanism itself is already proven on a strong attractor (CYCLE 1068).

## R-iii arc (honest)
- COMPLETION half: SOLVED (CYCLE 1068, dendritic dAP, 6-seed).
- FORMATION half: SOLVED (this cycle — the Kopsick-correct direct-synchronous protocol forms a strong sparse attractor, 3.3-3.5x, where all plasticity-rule-only attempts plateaued at 1.44x).
- CAPSTONE (compose them = emergent CA3 completion): pending the running de-risk. If GO -> the R-iii enabler is achieved -> the SWR generative-replay loop rides it.

## Files
`research/runners/_riii_ca3_direct_assembly_derisk.py`, `_riii_ca3_emergent_completion_derisk.py`. Prior: `2026-07-09-riii-sparse-synchronous-ca3-ensemble-research-gate.md` (the gate), `-ca3-feedback-inhibition-sparsifies-but-nonselective.md` (1072), `-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md` (1068). Ref: Kopsick et al. 2024, J Comput Neurosci (PMC10996657): 20ms gamma, symmetric STDP tau=20ms, ~0.37% assembly, sparsity from the input protocol. Kandel Ch 54; Marr 1971.
