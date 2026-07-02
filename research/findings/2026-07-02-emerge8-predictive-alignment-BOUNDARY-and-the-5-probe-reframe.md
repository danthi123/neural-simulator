# EMERGE-8 (rung-3) — Predictive Alignment, faithfully implemented + verified against the paper, does NOT beat a fixed reservoir on this substrate (it collapses FASTER under capacity load). This is the 5th confirmation of a robust strategic conclusion: SUPERVISED local recurrent-weight training does not beat a fixed reservoir + trained readout at toy rate scale → the biology-grounded emergent path is UNSUPERVISED self-organizing sequence learning.

**2026-07-02 (autonomous; full cores).** Runner `research/runners/_emerge8_predictive_alignment_derisk.py`; results `research/findings/raw/_emerge8_predictive_alignment.json` + `_emerge8_capacity_sweep.log`. Reuse-by-import; NO `sim/` edit; CPU/numpy; multi-seed 42/43/44.

## Mechanism (verified verbatim from the paper before building)
Predictive Alignment (Asabuki & Clopath 2025, Nat Commun 16:6784, DOI 10.1038/s41467-025-61309-9). Recurrent = FIXED sparse chaotic `G` (p=0.1, g=1.2) + PLASTIC `M`. Rule (Eqs. 6/14): `ΔM = η_M(Qz − Ĵr)rᵀ`, `Ĵ = M − αG`, `Q` fixed-random N×K readout-feedback; readout delta `ΔW = η_W(f−z)rᵀ`; state `τẋ = −x + Gr + Mr + W_in·I + σξ`, r=tanh. PA aligns the recurrent prediction with the readout-feedback (tames chaos) rather than minimizing output error — fully local, NO `Wᵀ`, no inverse-correlation matrix, no BPTT (`used_transpose` False, all arms/seeds). I fetched + confirmed the exact equations from the PMC full text before coding, and independently verified the FORCE/Laje-Buonomano/Nicola-Clopath citations.

## The test + results
Autonomous generation of a periodic sinusoid-superposition target after a brief cue, then the **mechanism-native robustness metric**: inject a mid-trajectory PULSE and measure post-pulse readout-vs-target correlation (a stable attractor recovers; a memorized chaotic run diverges). PA vs a FIXED-RESERVOIR baseline (same `G`, `M`=0, readout trained — the strong bar my pre-design scratch flagged).

**`eta_M` sensitivity (diagnosed):** PA's clean generation recovers monotonically as `eta_M` drops (0.01→0.47, 0.001→0.79, **0.0003→0.85**); my initial 0.01 destabilized `M`. Correctly tuned (eta_M=0.0003), PA is implemented faithfully and converges.

**Single trajectory (n_traj=1):** PA recovery **0.845** ≈ fixed reservoir **0.833** (tie). The fixed reservoir already forms a robust attractor for a periodic target across ALL chaos gains (recovery 0.86–0.93 for g=1.5–3.0) — no room for PA to improve.

**Capacity sweep (the decisive test — reservoirs have finite attractor capacity, so PA should win as trajectory count grows):**

| n_traj | PA recover | reservoir recover | PA − reservoir |
|---|---|---|---|
| 1 | 0.845 | 0.833 | +0.01 (tie) |
| 4 | 0.349 | 0.642 | **−0.29** |
| 8 | 0.198 | 0.498 | **−0.30** |
| 16 | 0.045 | 0.414 | **−0.37** |
| 24 | 0.025 | 0.238 | −0.21 |

**PA COLLAPSES FASTER than the fixed reservoir** as trajectory count grows — the opposite of the hypothesis. The single shared plastic `M`, pulled toward many conflicting attractors, suffers catastrophic interference, while the fixed reservoir's random dynamics + a flexible linear readout degrade gracefully. Due-diligence retune at n_traj=8 confirms the gap is NOT a tuning artifact: PA = 0.185 (eta_M 0.001, ep 120) / 0.223 (0.0003, 120) / 0.273 (0.0003, 240) / 0.263 (0.0001, 240) — all far below the fixed reservoir's 0.603. More epochs help PA only marginally (0.22→0.27); the reservoir dominates across every PA setting.

## Verdict: BOUNDARY (build-informative) + a robust 5-probe strategic conclusion
Predictive Alignment — the scoped, citation-verified, fully-local, spiking-compatible chaos-taming rule — does not beat a fixed reservoir on this substrate; it underperforms and interferes under capacity load. This is the **5th independent confirmation** of one pattern:

1. **rung-3a** (target-based recurrent credit): one-step map fine, autonomous recall dead (exposure bias / free-run destabilization).
2. **rung-3a iter-3** (proper e-prop first-order eligibility): re-localized the wall to generation-stability, not credit quality.
3. **pre-design scratch RFLO**: naive local recurrent credit underperforms a fixed reservoir under noise.
4. **EMERGE-7** (next-symbol): a fixed reservoir memorizes high-order context (train 1.0), local credit DEGRADES it.
5. **EMERGE-8** (Predictive Alignment): PA underperforms + collapses faster than the fixed reservoir under capacity load.

**The robust conclusion: on toy rate-recurrent tasks, a fixed random reservoir + a locally-trained readout (reservoir computing / echo-state; biologically the cortex-as-reservoir / Maass liquid-state-machine hypothesis) is a very strong, robust baseline, and SUPERVISED local recurrent-WEIGHT training — target-based, e-prop, RFLO, and now Predictive Alignment — does not beat it at this scale.** The recurrent-credit advantage is not demonstrable on cheap toy rate probes; chasing it further on this vehicle is a false economy (the anti-config-thrash + the standing "don't over-invest in a comfortable path" discipline both say stop here).

## The reframe (the honest next frontier)
The whole rung-3 arc tested **SUPERVISED** recurrent credit (train weights to reproduce a target trajectory) — a motor/production framing. But biology's sequence cortex largely **SELF-ORGANIZES** sequence structure from experience via local Hebbian plasticity + inhibition (+ dendritic mechanisms), with NO explicit target — and that is the master-directive path (emergent, self-organizing, from streaming data). The mechanism is in hand and keeps recurring in this arc: **Bouhadjar, Wouters, Diesmann & Tetzlaff 2022 (PLoS Comput Biol 18:e1010233) — unsupervised high-order sequence learning, prediction, AND replay in a spiking network, fully Hebbian + WTA, NO teacher** (it was EMERGE-7's positive control + Task-B template). This sidesteps the entire "local credit vs reservoir" dead-end: it does not train recurrent weights to a target; it lets sequence structure + prediction EMERGE from experience.

**⇒ NEXT FRONTIER: pivot rung-3 from supervised recurrent credit to UNSUPERVISED self-organizing spiking sequence learning (Bouhadjar-Diesmann 2022).** Research-gate it first (a new mechanism class), then cheap-first de-risk: does a Hebbian+WTA (dendritic-AP) spiking network self-organize high-order sequence prediction from a streaming corpus, with anti-cheats (context beats a Markov floor; lesions collapse it; multi-seed)? This is more biology-faithful, self-organizing (not supervised motor mimicry), and aligned with the emergent-artificial-life master directive.

## Honest scope / caveats
- This does NOT prove PA is useless — a larger-scale / different-regime / spiking-native PA might win; but 5 probes show the recurrent-credit advantage is not cheaply demonstrable at toy rate scale, and the reservoir baseline is the pragmatic sequence substrate for now.
- Reservoir computing (fixed recurrent + plastic readout) is itself biologically defensible (cortex-as-reservoir / liquid-state-machine), so "the reservoir wins" is not a cheat — it is a legitimate substrate finding.
- All arms fully local (`used_transpose` False); PA implemented + tuned faithfully to the published equations; the fixed-reservoir bar is the honest control the pre-design scratch de-risk mandated.
- Do NOT start the `sim/` rung-4 port (no rung-3 GO). The next build is gated on the Bouhadjar unsupervised-sequence research gate.

## Artifacts
`research/runners/_emerge8_predictive_alignment_derisk.py` (PA net + capacity/n_traj + perturbation-recovery metric + eta_M-sensitivity default), `research/findings/raw/_emerge8_predictive_alignment.json`, `_emerge8_capacity_sweep.log`. Prior: `2026-07-02-emerge7-fork2-nextsymbol-task-misdesigned-but-3rd-confirmation-local-credit-degrades-reservoir.md`, `2026-07-02-emerge6b-rung3a-eprop-eligibility-relocalizes-wall-to-generation-stability.md`, `2026-07-02-fork2-predesign-scratch-derisk-reservoir-is-the-bar.md`, `2026-07-02-rung3-generation-stability-mechanisms-scoping.md`.
