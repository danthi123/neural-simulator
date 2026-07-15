# R-iii CA3 emergent attractor FORMATION — COMPETITIVE-HEBBIAN surpasses the 2026-07-09 saturation boundary (6-seed GO)

**Date:** 2026-07-14
**Status:** FORMATION = 6-seed GO (boundary surpassed). Downstream completion payoff = characterized next rung (below).
**NO `sim/` edit** anywhere — the committed EMERGE-40 kernel is imported + applied runner-side.

## The boundary (2026-07-09) that this surpasses

`2026-07-09-riii-formation-rules-saturate-ensemble-dynamics-is-the-blocker.md`: forming a SELECTIVE within-ensemble
CA3 attractor FROM EXPERIENCE saturated — ALL FOUR pure-LTP rules (causal-offset / symmetric / rate-window BCM)
formed only a WEAK ~1.44× within-ensemble separation (member→silent grew in LOCKSTEP with member→member). The
dendritic-dAP completion READ-OUT was already GO (CYCLE-1068, `2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed`)
— but only on a HAND-INSTALLED attractor. Forming the attractor was the wall.

**Reference reproduced FIRST-HAND (2026-07-14, a0):** even the full sparse+synchronous combo (feedback-inhibition +
mossy detonator + gamma-paced synchronous encoding + rate-window rule) still gives within 4.84 ≈ member→silent 4.85
→ separation **−0.01** (the H2 "NON-SPECIFIC RULE" verdict). ⇒ sparsity + synchrony ALONE do not fix it.

## The mechanism (boundary-surpassing deep-research gate, adversarially verified)

**COMPETITIVE-HEBBIAN formation** = the committed `sim/kernels.fused_htm_winner_inactive_depression` (EMERGE-40,
verified sim/kernels.py:432) applied to the ca3→ca3 RECURRENT weights for the **FIRST time** (it had only ever run
FEEDFORWARD, EMERGE-38/39). Alongside the bridge's rate-window LTP, each encoding window we DEPRESS the recurrent
synapses of an assembly member to/from cells that are NOT in the assembly (both directions — the kernel called
twice with swapped args). Net = LTP lifts member→member UP while the heterosynaptic term forces member→silent DOWN
→ a BIMODAL (winner-take-all-in-weight-space) SELECTIVE attractor (Zenke-Agnes-Gerstner 2015 Nat Commun 6:6922;
Litwin-Kumar & Doiron 2014).

**Why it is the missing ingredient (not a rule-form/lr failure):** every 2026-07-09 rule is pure homosynaptic LTP
with only a soft weight ceiling and NO term coupling within-assembly potentiation to depression of the same cell's
OTHER synapses. So in the distributed 35–47%-active code, "silent" non-members co-fire enough to potentiate in
lockstep. The competition DOWN-term supplies exactly the per-postsynaptic competition the pure-LTP rules lacked.

**Genuinely new (a-1 confirmed):** the entire tried set (`2026-07-08-riii-ca3-attractor-formation-symmetric-hebbian`,
`-coincidence-wired-but-blocked`, `2026-07-09-formation-rules-saturate`) is pure LTP; none has a
competitive/heterosynaptic DOWN-term keyed to a winner mask.

**The key robustness detail — ensemble-mask keying (not per-event):** the "winner" mask keyed to the per-EVENT fire
craters within-ensemble too (the distributed code fires async, so a within-member that is momentarily silent on an
event gets its incoming/outgoing synapses wrongly depressed). Keying the competition to the CUMULATIVE ensemble
mask (a cell that fired ≥ `ens_thresh` times across the pattern's events is a stable assembly member → protected)
fixes this: within-ensemble survives, member→silent craters.

## Formation result — 6-seed GO

Single-variable anti-cheat (identical config; the ONLY difference is the competition rate `lam_dep_wi`):

| config | within (member→member) | member→silent | ratio | verdict |
|---|---|---|---|---|
| **lam=0 (control)** | 4.89 | 4.87 | **1.01** | reproduces the pure-LTP saturation |
| lam=0.5 seed 42 | — | — | **8.90** | GO |
| lam=0.5 seed 43 | — | — | **8.04** | GO |
| lam=0.5 seed 44 | — | — | **6.13** | GO |
| lam=0.5 seed 100 | — | — | **6.92** | GO |
| lam=0.5 seed 101 | — | — | **5.19** | GO |
| lam=0.5 seed 102 | — | — | **7.80** | GO |

**6/6 GO — ratio 5.2–8.9× (mean ~7.2×) vs the pure-LTP 1.01×.** The 2026-07-09 ~1.44× saturation is decisively
surpassed. Mechanism-discriminated: member→silent DROPS below init 6.0 (the gain is heterosynaptic competition, not
the hebbian_max weight-ceiling lever). Anti-cheat: lam=0 → ratio 1.01 (competition is load-bearing).

Runner: `research/runners/_riii_ca3_competitive_formation_derisk.py` (reuse-by-import of the CYCLE-1066 harness).

## Downstream completion payoff — the characterized NEXT rung (honest)

GO BAR 2 = feed the LEARNED attractor into the CYCLE-1068 dendritic-dAP completion read-out and require held-out
c_drive > non-stored (reversing the documented held 75.9 < non-stored 84.0). Result: **direction correct** (held-out
c_drive > non-stored, all lam) but a real tension surfaces:

- The competition's selectivity is DEPRESSION-based → the absolute within-ensemble drive drops (c_drive ~0.4), far
  below the plateau `k_thresh=18` calibrated to the pure-LTP scale → the plateau does not fire (completion activity ~0).
- The held-vs-**all-non-stored** margin is thinned to ~1.45× (vs 3.7× against TRULY-silent) by "fired-somewhat" cells
  the competition only weakly suppresses.
- `k_thresh` is ENTANGLED with training (it is passed to `_build`, so lowering it floods the plateau DURING encoding
  → everything potentiates → indiscriminate firing) — so it is NOT a clean post-hoc read-out calibration.

⇒ The workflow's COUPLED recommendation is confirmed: competition (selection-in-weight-space, DONE) needs the FS-WTA
**sparsification** companion (eliminate the fired-somewhat cells → non-stored becomes truly-silent → the completion
margin widens to the formation's 3.7–8.9×, and the synchronous sparse assembly keeps the absolute within-ensemble
drive high). Feedback-inhibition alone gave frac_active 0.35 (not sparse enough); the sparsification sweep
(competition + stronger `ca3_fb_inhib` + lower drive) is the active next rung. Payoff runner:
`research/runners/_riii_ca3_competitive_completion_payoff_derisk.py`.

## Bottom line

The competitive-Hebbian mechanism (heterosynaptic winner-inactive depression on the ca3→ca3 recurrents, keyed to the
stable assembly) is the first mechanism to form a **SELECTIVE** (not uniform) CA3 attractor from experience — 6-seed
GO, surpassing the 2026-07-09 saturation boundary. The clean end-to-end pattern-completion on the learned attractor
needs the FS-WTA sparsification companion to widen the completion margin — a characterized next mechanism, not a wall.
