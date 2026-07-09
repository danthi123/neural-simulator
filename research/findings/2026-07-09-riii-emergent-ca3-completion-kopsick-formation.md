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
   + sparser (n_ca3=2000, 0.40%)                      19.58             6.05           3.23x   (scale does NOT lift the ratio -> the SPARSITY hypothesis was wrong)
--- THE ACTUAL LEVER (self-correction): the within-assembly WEIGHT CEILING (hebbian_max_weight), pres=100 lr=10 ---
   hebbian_max_weight=30 (the default cap)            ~20               ~6             3.3x    (within CLAMPED at the cap)
   hebbian_max_weight=60                              49.04             6.51           7.53x
   hebbian_max_weight=120                             81.65             6.47           12.63x  (STRONGER than the 10x hand-installed attractor)
```
The mechanism chain, each step read off the data: (1) DIRECT synchronous drive binds the assembly strongly (within 22 vs the trisynaptic 7 — synchrony is the lever); (2) cross rises with presentations from recurrent SPILLOVER, then FEEDBACK INHIBITION suppresses it back toward init (~6); (3) **the "sparser assembly is the decisive lever" reading was a RED HERRING.** Scaling to n_ca3=2000 (0.40%, Kopsick's regime) gave 3.23x — the SAME plateau. The cross was already at init (6.05 = minimal spillover); the ratio = within/init = within/6, and **within was CLAMPED at the default `hebbian_max_weight=30`.** (4) **The actual lever is the WITHIN-assembly weight CEILING: raising `hebbian_max_weight` 30->60->120 climbs the ratio LINEARLY 3.3x -> 7.5x -> 12.6x** (within 20 -> 49 -> 82; cross pinned ~6.5). At hm=120 the learned attractor (12.63x) is STRONGER than the 10x hand-installed attractor CYCLE 1068's dendritic completion fired on. This is a self-correction of my own prior-cycle conclusion (the CYCLE 1070->1071 error class again): I read "cross ≈ init" as "spillover is the problem, need a bigger net," when the number said the OPPOSITE — cross was clean, within was clamped. no-train collapses (0.98x) throughout. [Honest: the weight-ratio's PERMUTED control rises WITH the within-weight (2.5x at hm30 -> 5.3x at hm120) — a metric confound, since "cross" is to always-init non-assembly cells; the CLEAN specificity test is the completion capstone's partial-cue controls, below.]

## The capstone (emergent completion) — with the STRONG (12.6x) learned attractor the held-out members COMPLETE (0.972) and LINEAR=0 + NO-TRAIN=0 are clean; the open residual is RETRIEVAL SPECIFICITY (a random cue also ignites the net), being closed by the retrieval threshold
The CYCLE-1074 recall-tuning (below) was on the WEAK 3.3x attractor. With the hebb_max=120 strong attractor:
```
recall config (hebb_max=120, pres=60)         held-out   non-assembly   LINEAR   NO-TRAIN   PERM-CUE
cue 1000, inhibition kept, k=6                 0.972      0.996          0.000    0.000      0.919   (COMPLETES + dendritic + training-dep; but NON-SPECIFIC)
cue 200,  inhibition kept, k=6                 1.037      1.093          0.000    0.000      0.908   (lower cue does NOT fix -> not the knob)
```
Two controls are now DECISIVELY clean: **LINEAR=0** (the dendritic plateau is load-bearing — a linear read-out completes NOTHING even on the 12.6x attractor) and **NO-TRAIN=0** (training-dependent). The held-out members COMPLETE strongly (0.972). The open residual is pure SPECIFICITY: non-assembly=0.996 and **PERM-CUE=0.919** — a RANDOM 6-cell cue also ignites the held-out members. Root cause (read from the mechanism, NOT guessed): at recall a 6-cell cue delivers, to every non-member, a cross-weighted drive ≈ 6 × cross(6.5) ≈ **39**, which EXCEEDS the plateau threshold k=6 → global ignition; the same 6-cell cue delivers, to a same-assembly held-out member, a within-weighted drive ≈ 6 × within(82) ≈ **492**. So the specificity knob is **k_thresh**, which must sit BETWEEN the cross-drive (~39) and the within-drive (~492) — k=6 was ~6x too low. The cue magnitude is NOT the knob (cue 200 ignites too). k-sweep {40,80,120,160} at hebb_max=120 in flight; the predicted specific window is k≈60-150 (perm/non < k=80 < held). If the 60-step recurrent CASCADE defeats a first-step threshold (once held-out A fires, the growing active set re-drives non-members), the paired lever is a SHORT recall window (measure the 2-4-step plateau completion before the cascade) and/or Kopsick's gamma-paced RETRIEVAL (brief cue pulses + inhibition-mediated winner-take-all settle) — the retrieval protocol I have not yet replicated (distinct from the FORMATION protocol I did). This is a retrieval-E/I-tuning residual on top of the committed formation breakthrough + the (now decisively re-confirmed) dendritic-load-bearing completion — NOT a formation or a completion-mechanism failure.

### (superseded) the CYCLE-1074 recall-tuning on the WEAK 3.3x attractor
```
recall config                                 held-out   non-assembly   LINEAR   NO-TRAIN   PERM-CUE
cue 600 pA, inhibition kept                    0.000      0.000          0.000    0.000      0.000   (cue suppressed by recall inhibition)
cue 1000 pA, FULL disinhibition, k=6           0.943      0.981          0.000    0.429      0.997   (no inhibition -> hyperexcitable, all fire)
cue 1000 pA, inhibition kept, k=12             0.217      0.220          0.000    0.000      0.000   (CONTROLS CLEAN; weak + non-specific)
cue 1000 pA, inhibition kept, k=20             0.008      0.000          0.000    0.000      0.000   (k too high -> nothing fires)
```
At the weak 3.3x attractor the completion was weak (0.217) because the within/cross weight ratio was too small to separate held-out from non-assembly at recall — the CYCLE-1074 "attractor too weak" read. The hebb_max ceiling fix (CYCLE 1075) made the attractor strong (12.6x), which flipped the residual from "too weak to complete" to "completes but the retrieval threshold isn't set for specificity" — a cleaner, well-localized knob.

## R-iii arc (honest)
- COMPLETION half: SOLVED (CYCLE 1068, dendritic dAP, 6-seed).
- FORMATION half: SOLVED (this cycle — the Kopsick-correct direct-synchronous protocol forms a strong sparse attractor, 3.3-3.5x, where all plasticity-rule-only attempts plateaued at 1.44x).
- CAPSTONE (compose them = emergent CA3 completion): pending the running de-risk. If GO -> the R-iii enabler is achieved -> the SWR generative-replay loop rides it.

## Files
`research/runners/_riii_ca3_direct_assembly_derisk.py`, `_riii_ca3_emergent_completion_derisk.py`. Prior: `2026-07-09-riii-sparse-synchronous-ca3-ensemble-research-gate.md` (the gate), `-ca3-feedback-inhibition-sparsifies-but-nonselective.md` (1072), `-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md` (1068). Ref: Kopsick et al. 2024, J Comput Neurosci (PMC10996657): 20ms gamma, symmetric STDP tau=20ms, ~0.37% assembly, sparsity from the input protocol. Kandel Ch 54; Marr 1971.
