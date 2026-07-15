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

## ⚠️ HONEST CORRECTION / EXTENSION (2026-07-14, same session — anti-cheat investigation of the completion payoff)

The formation section above STANDS (competition surpasses the 2026-07-09 **weight-ratio** saturation, 6-seed). But a
rigorous lesion/operating-point investigation of the downstream FUNCTIONAL completion reversed the optimistic framing
of "the completion just needs the sparsification companion." The findings:

- **The competition's weight-ratio surpass does NOT translate to functional pattern completion.** With FS-WTA
  sparsification (fb_inhib=20 + low training drive), the LESION control (lam=0, NO competition) COMPLETES IDENTICALLY
  to lam=0.5 on seed 42 (held-out 0.875 / non-stored 0.087) — and competition HURTS on other seeds (held-out → 0.000
  on seed 43). So whatever completion occurs is driven by the SPARSIFICATION, not the competition. The weight-ratio
  (8.9×) is real but functionally inert-to-harmful for completion — i.e. the 2026-07-09 within/silent **ratio** was
  the wrong proxy for the functional goal.
- **The functional completion itself is SEED-FRAGILE at this scale** (n_ca3=150, 2 memories, ~15-cell ensembles). The
  best operating point (sparsification, fb20 + low drive) completes cleanly on only 2/6 seeds (42, 44 → held-out
  0.875) and gives held-out **0.000** on 3/6 (43, 100, 101). Neither the mossy-detonator (2026-07-09 Rung 2; held-out
  0.125/0.000/0.109) nor a decoupled strong recall cue (held-out 0.062/0.000/0.058) helped; a strong recall drive was
  WORSE (the sparse low-drive recall is better).

**⇒ Honest state:** the FORMATION weight-ratio saturation is surpassed (competition, 6-seed GO — a real advance on the
specific documented metric), but ROBUST functional pattern-completion on the emergent learned attractor is an OPEN
residual — seed-fragile, and most likely SCALE-limited (tiny ensembles + a tiny cue/held split make the bottom-firing
held members hard to complete). This is a boundary = an undiscovered mechanism, not a wall: the ranked next levers are
(1) SCALE (larger CA3 → larger, more redundant ensembles → robust completion) and (2) theta-gamma SYNCHRONIZATION
(2026-07-09 Rung 3 — synchronous member co-firing for a dense, strongly-coupled recurrent loop). The weight-ratio was
a misleading proxy; the functional-completion metric (does the held-out member reactivate from a partial cue?) is the
one to gate on.

## Deep a0 mechanism reading (the root cause + the ranked next mechanisms)

Reading our own decisive substrate/findings IN DEPTH (a0) pinned WHY the functional completion is fragile and what
the robust recipe is:

- **The CYCLE-1068 completion GO read-out is the two-compartment DENDRITIC dAP** (`two_comp=True`, `apical_R=50`,
  `k_thresh` CALIBRATED to the per-step coincident drive ~6-7 of a CLEAN 10× hand-installed attractor). My payoff
  runner had been using the POINT-NEURON read-out (`two_comp=False`, `k_thresh=18`) that CYCLE-1067 PROVED fails even
  on a good attractor — an a0 catch. But installing the validated dendritic read-out (two_comp + k_thresh=6) on the
  LEARNED attractor ALSO failed (held-out 0.000/0.000/0.175): the k_thresh=6 was calibrated to the hand-installed
  attractor's WEIGHT SCALE, which the learned attractor does not match, and k_thresh is entangled with training.
  ⇒ the read-out is validated but does NOT transfer to the learned weight scale without co-calibration.
- **The ROOT cause:** the CYCLE-1068 read-out was validated on a CLEAN attractor (within-ensemble W_HIGH ~10-30,
  member→silent W_LOW — a ~10× ABSOLUTE separation). Neither competition (high ratio, LOW absolute ~0.46) nor
  sparsification (higher absolute but seed-inconsistent) reliably produces that clean structure. The robust recipe
  (Kopsick-Ascoli 2024, the direct spiking-CA3 model; PMC10996657) is a SPARSE (<1%) + STRONGLY-firing +
  SYNCHRONOUS assembly: the selected assembly PCs are DRIVEN together in a 20 ms gamma window (4 spikes) so symmetric
  STDP binds a strong within-assembly attractor, with assembly-SELECTIVE inhibition (PMC12244581) keeping
  non-members silent WITHOUT suppressing members. My tests drove the UPSTREAM input (not the assembly cells directly)
  and tested the pieces separately — never the full Kopsick recipe.

**⇒ Ranked next mechanisms (research-gated) — then the DECISIVE DIAGNOSTIC that bounded the arc.** A focused
deep-research gate (adversarially, primary sources) returned the exact Vogels-Sprekeler iSTDP rule + a crucial
correction (the CA3 selective-inhibition paper Kim-Kim 2025 PMC12244581 actually uses plastic **E→I** symmetric STDP,
NOT Vogels I→E), and — most valuably — the single highest info-per-minute move BEFORE building any plastic inhibition:
a **recall-time inhibition-knob DIAGNOSTIC**. Scale `ca3_pv_basket→ca3` down by g at RECALL only (encoding unchanged);
if completion robustifies in some g<1 → "members crushed by inhibition" is the bottleneck → build iSTDP; if no g
helps → the recurrent-weight structure is the bottleneck → iSTDP won't help.

**RESULT (fragile seed 43, baseline held-out 0.000): g ∈ {0.7, 0.5, 0.3, 0.0} ALL give held-out 0.000** (non-stored
rises 0.029→0.057 as inhibition relaxes). ⇒ **even ZERO recall inhibition does not rescue the held-out members — the
bottleneck is NOT inhibition level; it is the RECURRENT-WEIGHT STRUCTURE (the learned within-ensemble cue→held weights
are too weak to fire the held members).** iSTDP / assembly-selective inhibition is RULED OUT as the fix (the ~30-min
diagnostic saved that build). The genuine residual = FORM A HIGH-ABSOLUTE within-ensemble attractor: the members must
fire STRONGLY + SYNCHRONOUSLY at encoding (Kopsick-Ascoli 2024: mossy-detonator-driven assembly + a 20 ms gamma
window → dense co-firing → strong within-ensemble LTP → W_HIGH like the hand-installed attractor) — with the sparse
code keeping non-members silent. The remaining ranked next mechanisms: (1) **Kopsick strong-synchronous encoding**
(the direct fix — high-absolute within-ensemble via dense synchronous co-firing); (2) engram intrinsic-excitability
boost of stored members (Josselyn-Silva; cheap, but a partial model); (3) theta-gamma sequential recall
(Lisman-Idiart E%-max — the runner already has `ca1_pv_basket` E%-max wiring to port to CA3).

## Bottom line

The competitive-Hebbian mechanism forms a **SELECTIVE weight structure** (within/truly-silent 8.9× vs the pure-LTP
1.01×, 6-seed GO) — surpassing the 2026-07-09 within/silent-**ratio** saturation. But that weight-ratio surpass does
NOT deliver robust functional pattern completion: rigorous investigation across competition / sparsification /
mossy-detonator / scale / point-neuron-vs-dendritic read-out shows the functional completion is deeply seed-fragile,
because the LEARNED attractor is not as CLEAN (high-absolute + selective) as the hand-installed attractor the
CYCLE-1068 read-out was validated on. The robust functional completion is the honest open residual — a genuine
boundary = an undiscovered mechanism, with the specific ranked next mechanisms above (assembly-selective iSTDP; the
full Kopsick sparse-synchronous recipe; a co-calibrated read-out), research-gated.
