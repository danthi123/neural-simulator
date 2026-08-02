---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes/multihop_6seed_learned.json
  - research/findings/raw/gap4/realspikes/multihop_6seed_fixed.json
---

# gap#4 crux — MULTI-HOP CHAINED spiking port (6-seed NEGATIVE): the rate depth-rescue does NOT transfer to the spiking read regime — the depth "lift" is LABEL-AGNOSTIC and KP fails to align on spikes; confirmed across a 2nd architecture

<!--derived-->
**One-line verdict.** The rate overturn (`...transport-free-ceiling-FALSIFIED...`) proved transport-free deep
credit works AT RATE (chained FA + σ′ clears depth-2; KP-learned feedback rescues MNIST depth-4). It does NOT
transfer to the real-spikes read regime. A genuine 2-plastic-layer chained real-spikes port (chained
transport-free credit `e_l = (e_{l+1} @ Y_l)·σ′(v−θ)_l`, each Y_l KP-learned, transport-free per layer asserted
in-run) is a 6-seed NEGATIVE: the multihop lift over the frozen reservoir is **label-agnostic** — DIRECTED
credit (credit − permuted) clears the margin on only 1/6 seeds (learned) / 0/6 (fixed), dcs>0 2/6, `anti_ok
False`. This CONFIRMS the single-layer spiking negative (`...single-layer-does-NOT-beat-frozen...`) across a
second architecture, and re-confirms the 2026-07-14 graded-credit-decisive conclusion (the spiking substrate
wall is the credit-STRUCTURE / read regime, not the rule) with the NEW factors (KP-learned feedback + σ′(v−θ) +
multi-hop) — none of which rescue it. This is a METHOD+REGIME negative, NOT a capability wall: the mechanism
works at rate; the spiking READ regime defeats it. No `sim/` edit (additive subclass).

## Result — 6 seeds (42/43/44/100/101/102), 2 plastic layers

Artifacts: `research/findings/raw/gap4/realspikes/multihop_6seed_learned.json` and `..._fixed.json` (numpy/CPU,
40 epochs, 200/200 columns, top-k=8 inter-layer coupling).

<!--derived-->
| gate (multihop, both arms) | learned (KP) | fixed (FA) |
|---|---|---|
| beats frozen reservoir | 1/6 | 1/6 |
| **DIRECTED credit > permuted + margin** | **1/6** (need 6) | **0/6** |
| dcs > 0 | 2/6 | 2/6 |
| anti-cheats clean | False | False |

The multihop lift over frozen is real but LABEL-AGNOSTIC: permuted-label and wrong-sign training lift the
held-out codon as much as (or more than) correct-label credit — any σ′-gated, renorm-bounded perturbation of the
right scale makes the top codon more separable, regardless of label correctness. KP's feedback does not align to
the forward weights on spikes (weak per-layer alignment), and the credit reaching the deep layer is attenuated
(~3×) — so the depth-rescue that carried the rate MNIST result (FA 0.531 → KP 0.876 at depth-4) does not engage
here. Consistent with the σ′-decomposition of WF-Act-PC ([arxiv 2607.13380](https://arxiv.org/html/2607.13380v1))
holding at RATE while the finite-spike / coupling read regime breaks the directed signal on spikes.

## What this maps, and the next mechanism (named, not deferred)

<!--derived-->
The rate→spikes transfer has now failed on BOTH the single-layer and the multi-hop chained port at 6 seeds, each
via the same failure signature: a generic-plasticity representational lift that is NOT directed (label-agnostic)
and NOT feedback-specific (KP ≈ fixed FA; KP does not align). So the boundary is cleanly located: the
transport-free deep-credit CLASS assigns directed credit at RATE (the overturn), but the SPIKING READ REGIME
(σ′(v−θ) from a membrane whose columns never somatically spike; finite-spike coupling attenuation) does not carry
the directed signal at this budget. **Next mechanisms (the frontier continues):** (1) CONTROL the label-agnostic
plasticity confound — a harder task where generic plasticity provably cannot lift held-out, so any gain IS
credit; (2) FORCE KP to align on spikes — higher kp_lr / longer budget to grow feedback alignment, or a coupling
/ σ′ regime where deep-layer credit does not attenuate; (3) the 2026-07-14 graded-credit-decisive direction (a
lower-CV somatic read). The rate overturn stands as the session's result; the spiking read regime is the
distinct, harder, well-mapped frontier.
