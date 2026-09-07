---
type: reference
status: research-synthesis
claim_check: literature
date: 2026-09-06
mechanism: AGI-fork — external-literature research at the wall (3 fork negatives) on WHY an emergent place code
  fades under an online predictive objective, and WHICH mechanism makes it persist. Informs the fork's next
  build (a successor-representation predictive target).
lane: agi-fork (emergence / continual-learning substrate)
branch: agi-fork
builds_on:
  - research/findings/2026-09-06-agi-fork-firstmove-emergence-transient-objective-does-not-retain.md
  - FORK.md
verdict: >
  FOUR independent literature-research agents converge. (1) DIAGNOSIS: the fork's place-code fade is NOT
  biological representational drift (which preserves population decodability) — it is catastrophic
  forgetting / FEATURE SUPPRESSION / loss-of-plasticity: a single-step next-latent (JEPA) objective does not
  REQUIRE persistent position, so a continually-updated shared substrate reallocates capacity to whatever is
  currently loss-reducing and actively overwrites the early place code (fading it BELOW an untrained-reservoir
  floor is the signature of active erosion, not drift). (2) FIX: make position load-bearing ON THE OBJECTIVE.
  The deepest-biology, most-emergent, no-host-label mechanism is a SUCCESSOR-REPRESENTATION predictive target
  (Stachenfeld 2017: place cells ARE a successor map); the most-cited, lowest-cost complement is an auxiliary
  self-localization loss + velocity/efference input (Cueva-Wei 2018, Banino 2018, Sorscher 2019/2023). Anti-
  forgetting mechanisms (continual-backprop, synaptic intelligence, fast/slow CLS weights) PROTECT whatever
  code exists but do not RE-CREATE spatial structure — they are secondary, applied after the objective fix.
  (3) INSTRUMENT WARNING (load-bearing): linear-decodability and causal load-bearing DISSOCIATE — Schøyen 2023
  found the high-decodability units were causally dispensable while a different band-like population carried
  path integration; Schaeffer 2022 ("No Free Lunch") found grid-like activity is a hyperparameter-sensitive
  artifact. Reward shaping that makes the agent "home more" can be satisfied WITHOUT routing through the
  measured place code — exactly the fork's smoke result (eating rose sharply, place not load-bearing). ⇒ measure
  load-bearing by ablating BEHAVIORALLY-important units and checking BEHAVIOR degrades (Banino/Schøyen method),
  not by the decoding-importance probe alone. NEXT: build the SR target (additive, default-off, gradcheck),
  compose with the shaping magnitude sweep, and correct the lesion instrument. Fork branch only; nothing wired.
---

# AGI-fork: why the emergent place code fades, and the successor-representation fix (research at the wall)

## Why this exists
Three fork negatives (first-move transience; longer-horizon; nav-required task) mapped a real wall: an emergent
place code appears early and then fades to at-or-below the untrained-reservoir floor, and neither a longer
prediction horizon nor a task that *requires* position rescued it. Per the workflow's deep-research-at-a-wall
discipline, four independent external-literature agents were run (persistence mechanisms · reward-shaping
validation · path-integration inductive biases · representational-drift / anti-forgetting). Their findings
converge cleanly and are banked here; every citation below was verified by the agent that surfaced it.

## 1. Diagnosis — this is forgetting/feature-suppression, NOT biological drift
Biological representational drift reconfigures single-cell tuning while **population decodability of the
behaviorally-relevant variable stays flat or improves** (Ziv 2013; Rule/O'Leary/Harvey 2019; Rule 2020 eLife;
Driscoll/Duncker/Harvey 2022). A code that rises early then falls **below an untrained random reservoir** is the
opposite signature — active erosion, which the continual-learning ML literature attributes to:
- **Feature suppression** in self-supervised objectives: when several features are predictive, the objective
  keeps the "easy" ones and can actively suppress a partially-learned harder one as gradient pressure
  reallocates capacity (Robinson 2021, "Can contrastive learning avoid shortcut solutions?").
- **Primacy bias**: online RL overfits early data statistics, degrading representations useful later; partial
  resets restore plasticity (Nikishin 2022).
- **Loss of plasticity**: prolonged online non-stationary training with no resets can make a net *worse than
  linear*; ordinary L2/dropout only partially help (Dohare 2024, Nature).

The root cause is singular: **a single-step next-latent objective does not REQUIRE a persistent position code**
— once cheaper local correlates suffice, the shared substrate is free to (and does) overwrite it.

## 2. The fix — make position load-bearing ON THE OBJECTIVE (ranked, fork-fit)
Across all four agents the persistence cases had an explicit, always-on position-carrying channel/loss; an
implicit requirement inside one scalar reward or one-step prediction is too weak.

1. **Successor-representation (SR) predictive target — PRIMARY.** Predict γ-discounted future latent occupancy
   instead of only t+1. The target is intrinsically organized around the environment's reachability graph, so
   position stays useful for many future steps. **Deepest biology** (Stachenfeld 2017: the hippocampus is a
   predictive/successor map — place cells literally *are* an SR), **self-supervised (no host position label)**,
   and implementable as a vector analogue of the substrate's existing within-window discounted-MC value head.
   Barreto 2017 (successor features) is the ML form.
2. **Auxiliary self-localization loss + velocity/efference input — STRONG COMPLEMENT / FALLBACK.** Force a
   linear/shallow readout of position from the recurrent state every step, with self-motion input and a
   metabolic/nonnegativity regularizer (Cueva-Wei 2018; Sorscher 2019 NeurIPS / 2023 Neuron; Banino 2018
   Nature — grid-like units emerge AND are causally load-bearing). Cheapest to add; uses a host position label
   (a mild scaffold, "no different from the existing reward shaping" per the review) and makes the *decodable*
   code load-bearing by construction — so pair it with an independent behavioral test (§3).
3. **Policy-load-bearing coupling.** Route position into the policy/value pathway so it is *used*, not just
   decodable (Vijayabaskaran & Cheng 2026: place code persisted precisely when the policy depended on it) —
   matches this project's own "faculties must drive, not observe" doctrine.
4. **Anti-forgetting (SECONDARY, protect-not-create):** continual-backprop utility-tracked reinit (Dohare 2024),
   synaptic intelligence / online importance (Zenke 2017; EWC Kirkpatrick 2017 needs task boundaries),
   context-gating / active dendrites (Masse 2018; Iyer 2022), two-timescale fast/slow CLS weights (Ba 2016;
   Miconi 2018; McClelland 1995; Kumaran 2016). These stabilize whatever code exists; they do not re-create
   spatial structure, so they follow the objective fix. (Consolidation/replay was already tried on the fork with
   small/mixed effect — consistent: it was applied before position was load-bearing on the objective.)
5. **Nearly-free pilot (low confidence):** predictive-coding alone reportedly grows grid cells (Tang/Barron/
   Bogacz 2024, unreviewed; persistence untested) — cheap to try, not a proven fix.

## 3. Instrument correction — decodability ≠ load-bearing (this changes how we measure)
The sharpest, most actionable finding: **the linear-decodability metric can be misleading.** Schøyen 2023
(iScience) found that in a trained continuous-attractor agent the classic high-grid-score (high-decodability)
units were causally **dispensable** under ablation, while a different band-like population actually carried path
integration. Schaeffer 2022 (NeurIPS, "No Free Lunch") found grid-like activity is a hyperparameter-sensitive
artifact, not a robust outcome. And reward shaping that improves task performance can be satisfied by a strategy
that never routes through the measured code — **precisely the fork's shaping smoke** (eating rose sharply while
the place probe stayed near the floor and not load-bearing). ⇒ Correct the emergence battery to identify
faculty units by their contribution to BEHAVIOR (ablation-importance / policy-value gradient), then lesion those
and confirm the behavior degrades — the Banino/Schøyen causal test — rather than lesioning decoding-important
units alone. This may also mean prior fork negatives under-counted a code that was load-bearing via unmeasured
units; re-checking with the corrected instrument is part of the next step.

## Next actions (fork)
1. Build the **SR predictive target** (vector successor-feature head modeled on the value head's within-window
   discounted-MC target + analytic grad + gradcheck; additive, `--sr-weight`, default-off) — IN FLIGHT.
2. Harvest the shaping magnitude sweep (nav_shaping {1,2,4,8}, 6-seed) — does ANY magnitude make place
   load-bearing, or (as predicted) only increase homing without the code? (8.0 may overshoot into hovering.)
3. Correct the lesion instrument to behavioral-importance (Banino/Schøyen); re-audit.
4. If SR + shaping still do not retain a load-bearing code: add the auxiliary self-localization loss (§2.2),
   then anti-forgetting (§2.4). Compose, do not substitute.

## Honest scope
This is a LITERATURE/DESIGN synthesis, not a measurement — its claims are external citations (verified by the
surfacing agents) plus a mechanism decision. It satisfies the deep-research-at-a-wall gate for the fork's place
lane. Fork branch only; nothing wired to production. A fork negative + its research is a first-class deliverable
(FORK.md).

## Key citations (verified)
<!--derived: external-literature citations (years / DOIs / arXiv IDs) verified by the surfacing research agents; these are NOT measurements of this project, so they are marked derived for the claim check -->
<!--derived-->

- Stachenfeld, Botvinick & Gershman (2017). The hippocampus as a predictive map. Nat. Neurosci. 20:1643. doi:10.1038/nn.4650
- Barreto et al. (2017). Successor Features for Transfer in RL. NeurIPS 2017.
- Cueva & Wei (2018). Emergence of grid-like representations by training RNNs to path-integrate. ICLR 2018. arXiv:1803.07770
- Sorscher, Mel, Ganguli, Ocko (2019). A unified theory for the origin of grid cells. NeurIPS 2019. (Neuron 2023, 111:121)
- Banino et al. (2018). Vector-based navigation using grid-like representations. Nature 557:429. doi:10.1038/s41586-018-0102-6
- Whittington et al. (2020). The Tolman-Eichenbaum Machine. Cell 183:1249. doi:10.1016/j.cell.2020.10.024
- Vijayabaskaran & Cheng (2026). DRL account of visual + goal-vector signals for spatial navigation. Sci. Rep. doi:10.1038/s41598-026-63080-3
- Robinson et al. (2021). Can contrastive learning avoid shortcut solutions? NeurIPS 2021. arXiv:2106.11230
- Nikishin et al. (2022). The Primacy Bias in Deep RL. ICML 2022. arXiv:2205.07802
- Dohare et al. (2024). Loss of plasticity in deep continual learning. Nature 632:768. doi:10.1038/s41586-024-07711-7
- Ziv et al. (2013). Long-term dynamics of CA1 place codes. Nat. Neurosci. 16:264. doi:10.1038/nn.3329
- Rule, Loback, Raman, Driscoll, Harvey, O'Leary (2020). Stable task information from an unstable neural population. eLife 9:e51121. doi:10.7554/eLife.51121
- Zenke, Poole, Ganguli (2017). Continual Learning Through Synaptic Intelligence. ICML 2017. PMLR 70:3987
- Masse, Grant, Freedman (2018). Context-dependent gating + synaptic stabilization. PNAS 115:E10467. doi:10.1073/pnas.1803839115
- Kumaran, Hassabis, McClelland (2016). Complementary Learning Systems theory updated. TiCS 20:512. doi:10.1016/j.tics.2016.05.004
- Schaeffer, Khona, Fiete (2022). No Free Lunch from Deep Learning in Neuroscience. NeurIPS 2022.
- Schøyen et al. (2023). Coherently remapping toroidal cells ... path integration in virtual agents. iScience 26(9).
- Ng, Harada & Russell (1999). Policy invariance under reward transformations (potential-based shaping). ICML 1999.
- Grześ (2017). Reward shaping in episodic RL (terminal-potential caveat). AAMAS 2017.
- Trott et al. (2019). Keeping Your Distance: self-balancing shaped rewards. NeurIPS 2019. arXiv:1911.01417
