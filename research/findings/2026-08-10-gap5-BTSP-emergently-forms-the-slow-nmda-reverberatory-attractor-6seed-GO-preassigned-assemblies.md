---
type: finding
status: contributing
date: 2026-08-10
mechanism: ca3-completion
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: an EMERGENTLY-FORMED within-assembly ca3->ca3 attractor (BTSP plateau-gated one-shot storing via the on-bridge `fused_btsp_update` block; NO hand-set weight) is read by the IDENTICAL somatic slow-NMDA reverberatory gate as the 2026-08-10 hand-install de-risk (n_ca3=400, density 0.12, 3 disjoint 72-cell assemblies, exc_receptor="nmda_slow", shared FS basket ca3_fb_inhib=60, drive 300 pA, warm 200 + hold 200, frozen plasticity, hard-silence clears g_nmda_recurrent between conditions). The three-way ARM DECOMPOSITION is the instrument: (btsp) emergent formation vs (handinstall) the idealized-outcome cross-check vs (btsp_noplateau) the gate-lesion. Each assembly is formed in its OWN isolated encoding episode (fresh bridge, only that assembly driven) so cross-assembly potentiation is zero by construction (the DG pattern-separation function). Decomposed quantities: held_cue, held_perm (specificity teeth), held_nocue (silent-rest teeth), held_sustain (bistable persistence), plus the FORMED within-assembly weight w_within, cross_dw and nonmem_dw (genuine-formation specificity teeth). Load-bearing controls: HAND-INSTALL cross-check (reproduces the committed 6/6 GO on this bridge = readout fidelity), NO-ENCODING (no BTSP -> collapse), NO-PLATEAU (co-fire without the apical plateau -> no IS_post -> no potentiation -> collapse), RECURRENCE-ZERO (zero ca3->ca3 -> collapse), OU-off/OU-on. GO gate (6-seed): held_cue>=0.20 AND held_cue>=3*held_perm AND held_cue>=3*held_nocue AND held_nocue<=0.10.
---

# Gap #5 emergent-formation residual REACHES GO — BTSP one-shot plateau-gated plasticity EMERGENTLY FORMS (not hand-installs) a within-assembly attractor that the somatic slow-NMDA reverberatory gate COMPLETES: cue-specific + bistable, 6/6 seeds OU-off AND 6/6 OU-on — on PRE-ASSIGNED assemblies (emergent assembly SELECTION remains a SEPARATE open residual)

The 2026-08-10 slow-NMDA reverberatory de-risk
(`research/findings/2026-08-10-gap5-somatic-slow-nmda-reverberatory-attractor-bistable-specific-completion-6seed-GO.md`)
reached a 6/6 bistable + cue-specific CA3 completion on the point soma, but on a HAND-INSTALLED perfect within-assembly
potentiation W; its own "Honest scope" named the OPEN residual: **EMERGENT FORMATION into this reverberatory operating
point** — "BTSP one-shot plateau-gated storing that writes a completion-scale within-assembly slow-NMDA recurrent
WITHOUT the rate-Hebbian collapse, then read by THIS gate." This reaches GO on exactly that residual. NO `sim/` edit —
a new runner reuses the committed `enable_btsp` / `fused_btsp_update` block and the committed `exc_receptor="nmda_slow"`
readout (both byte-inert when off).

## Why this was genuinely untested (not a re-derivation)
The substrate's rate-Hebbian rule COLLAPSES ca3->ca3 to a uniform fixed point, so the attractor never forms via
rate-LTP (`research/findings/2026-07-17-gap5-completion-ROOT-CAUSED-hebbian-collapse-not-a-floor-workflow-6agent-verified.md`).
BTSP formation of a recurrent assembly was previously read only via the DENDRITIC two-compartment coincidence readout
(a standing readout surpass, `research/findings/2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`),
NEVER by the SOMATIC slow-NMDA reverberatory gate. That combination — BTSP-formed weights fed to the slow-NMDA
reverberatory completion — is the untested variable here.

## The mechanism (brain-based; ONE spiking substrate; runner-side only)
<!--derived-->

- ENCODE — BTSP plateau-gated ONE-SHOT storing (`fused_btsp_update`; Bittner-Magee 2017; Milstein-Magee 2021).
  Driving an assembly to co-fire (presynaptic eligibility) while a dendritic plateau is present (IS_post,
  `_is_post = max(cp_v_apical - v_hold, 0)`) potentiates the WITHIN-assembly recurrent one-shot. The weight is the
  RULE's output, saturating toward `btsp_w_max` — never a hand-set constant.
- The bistable apical compartment supplies ONLY the plateau IS_post gate; BDSP *learning* is OFF
  (`bdsp_learning_rate=0.0`) — this is NOT the dendritic deep-credit / BDSP hidden-credit rule, which is
  tested-NEGATIVE (`research/findings/2026-05-17-dendritic-credit-assignment-NEGATIVE`, cited here only to disclaim it).
- ISOLATED (pattern-separated) episodes — each assembly is formed on its OWN fresh bridge where ONLY that assembly is
  ever driven, giving cross-assembly dW == 0 by construction. This models the DG's pattern-separation across
  temporally-distinct memory-encoding events. (Diagnosed: a single-bridge SEQUENTIAL encode LEAKED — the seconds-long
  BTSP eligibility trace bled across assemblies, potentiating cross-assembly synapses, so a permuted cue drawn from
  other assemblies ignited the target; isolated episodes remove it. This is the one non-trivial design finding.)
- READ — the IDENTICAL slow-NMDA reverberatory gate (hard-silence clears the tau=100 ms g_nmda_recurrent, drive, warm,
  accumulate held-out firing over a long window; frozen plasticity). Readout machinery copied verbatim from the
  committed runner; the HAND-INSTALL cross-check arm empirically confirms it reproduces the committed 6/6 GO on this
  bridge.

## Result (6-seed 42/43/44/100/101/102; frozen at recall; both OU modes)
<!--derived-->

| arm | OU | per-config GO | w_within (grows with btsp_w_max) | held_cue | held_perm | held_nocue |
|---|---|---|---|---|---|---|
| **btsp (emergent)** | OFF | wmax2500 5/6 · **wmax5000 6/6** · wmax9000 6/6 | 2324 / 4936 / 8991 | 0.196–0.451 | **0.000** | **0.000** |
| **btsp (emergent)** | ON  | wmax2500 2/6 · **wmax5000 6/6** · wmax9000 5/6 | 2324 / 4936 / 8991 | 0.178–0.435 | 0.000 (101: 0.109) | 0.000 |
| handinstall (cross-check) | OFF | W2500 6/6 · W5000 6/6 · W9000 6/6 | 2500 / 5000 / 9000 | 0.215–0.450 | 0.000 | 0.000 |
| handinstall (cross-check) | ON  | W2500 5/6 · W5000 4/6 · W9000 2/6 | 2500 / 5000 / 9000 | 0.197–0.442 | 0.000–0.150 | self-ignites at high W |
| btsp_noplateau (gate lesion) | both | **0/6** | **≈1.5 (baseline, no attractor forms)** | 0.000 | — | — |

- **BTSP emergent formation = 6/6 seeds GO** (each seed has a GO working point) at OU-off AND OU-on; **wmax5000 is the
  robust point: 6/6 at BOTH OU**. The BTSP cue-vs-weight curve TRACKS the hand-installed idealization
  (btsp wmax5000 cue 0.324–0.351 vs hand-install W5000 0.328–0.356) — the emergently-formed attractor reaches the SAME
  completion the hand-install did.
- **≥ hand-install under noise:** at OU-on, btsp wmax5000 is 6/6 while hand-install W5000 is 4/6 — the emergent
  formation is at least as noise-robust as the idealization at the matched weight.
- **The formation is GENUINE, not a re-install** (`genuine_formation`=True every row): w_within is the RULE output
  (saturates toward btsp_w_max, ≤ ceiling), and the **btsp_noplateau lesion = 0/6, w_within≈1.5** — remove the plateau
  and NO attractor forms. The plateau IS load-bearing.
- **The formation is SPECIFIC by weight-change decomposition:** within-assembly potentiates to w_within 2324–8991 while
  **cross_dw ≈ 0 and nonmem_dw ≈ 0** (means −0.0002 / −0.0004 across all rows — even slightly negative). BTSP
  potentiates ONLY the within-assembly synapses; combined with the read-time FS basket this gives held_perm = 0.
- **All anti-cheats have teeth:** NO-ENCODING held_cue = 0.000 (attractor load-bearing), RECURRENCE-ZERO = 0.000
  (completion is the reverberation), permuted-cue = 0.000 (specific), silent-rest nocue = 0.000 (no self-ignition);
  determinism SEEDED (build-twice `cp_neuron_firing_thresholds` hash identical).

## Attribution
<!--derived-->
The runner emits `attributable_to` per working point (`tools.lab.attributable_to`): **BTSP-formed completion vs the
NO-ENCODING baseline = 100% attributable** (control = 0.000), and **correct-cue vs PERMUTED-cue = 100% attributable**
(control = 0.000). The `btsp_noplateau` arm is the causal foil for the STORING step (0/6, w_within≈1.5).

## Honest scope / what remains open (per THE LAW — a characterized boundary, not a wall)
<!--derived-->

- **This closes emergent FORMATION of the completion-attractor WEIGHTS, on PRE-ASSIGNED assemblies.** Which cells
  belong to which assembly is GIVEN. Emergent SELECTION of the assemblies (the DG's job) is a SEPARATE, still-open
  residual (`research/findings/2026-07-19-gap5-emergent-DG-ROOT-CAUSE-trisynaptic-feedforward-does-not-conduct-unifies-with-gap4-BTSP.md`).
  Frame this as *emergent formation of the completion attractor on pre-assigned assemblies*, NOT full gap#5 closure.
- **Read-time specificity is partly FS-basket-carried.** As the slow-NMDA finding established, the FS basket + sparse
  density + long integration carry much of the read-time perm=0; BTSP is load-bearing for the STORED-weight
  specificity (cross_dw ≈ 0) and for the FORMATION itself, not for all of the read-time specificity.
- **Ceiling-set magnitude (idealization, unchanged from the hand-install).** BTSP saturates the plateau-selected
  synapses to `btsp_w_max`, chosen at the operating-point scale (2500–9000; w_max ≫ W0=1.5, so this is genuine upward
  potentiation, NOT the bound-trap of `research/biology/btsp-place-field-formation.md`). BTSP supplies the SELECTION
  and SPECIFICITY emergently; it does not "discover" the magnitude — it fills the selected synapses to the ceiling.
- **Connectivity-sparsity residual (quantified).** The per-synapse operating-point magnitude (~5000) is large because
  each cell has only ~8.8 within-assembly inputs at density 0.12, so the reverberatory drive is the SUM (~43 500/cell).
  Kopsick homeostatic divisive downscaling (available in the runner, `--kopsick-t`) normalizes that SUM for
  seed-robustness but cannot make the per-synapse weight physiological at this sparsity — that needs denser recurrence
  or larger assemblies (the next mechanism if a physiological per-synapse weight is required; the completion itself is
  already emergent + GO).
- **OU-on bound:** at the HIGHEST ceiling (wmax9000) OU-on, one seed self-ignites (nocue 0.397) — the same
  inhibition/noise trade-off the hand-install hit at high W. wmax5000 is the noise-robust operating point (6/6 OU-on).

## Verdict
**EMERGENT FORMATION into the somatic slow-NMDA reverberatory operating point REACHES GO — 6/6 seeds, OU-off and OU-on
(robust point wmax5000).** BTSP one-shot plateau-gated storing forms a cue-specific, bistable within-assembly attractor
that matches the hand-installed idealization the 2026-08-10 de-risk used, with all load-bearing controls (no-plateau
0/6, no-encoding 0, recurrence-zero 0, cross_dw≈0) confirming the formation is genuine and specific. The rate-Hebbian
collapse is escaped by plateau-gated one-shot storing + isolated (pattern-separated) encoding. OPEN: emergent assembly
SELECTION (pre-assigned here) and a physiological per-synapse magnitude (connectivity-sparsity residual).

Artifacts (SIM_BACKEND=cupy; provenance sidecars record backend + argv + git SHA):
`research/findings/raw/_gap5_btsp_nmda/btsp_forms_6seed.json` (84 rows, both OU),
`research/findings/raw/_gap5_btsp_nmda/btsp_forms_6seed.log`.
Reproducer: `research/runners/_gap5_btsp_forms_nmda_slow_reverberatory_derisk.py`. NO `sim/` edit.

### Sources
- Bittner K.C., Milstein A.D., Grienberger C., Romani S., Magee J.C. *Behavioral time scale synaptic plasticity underlies CA1 place fields.* Science 357:1033–1036 (2017).
- Milstein A.D., Li Y., Bittner K.C., et al. *Bidirectional synaptic plasticity rapidly modifies hippocampal representations.* eLife 10:e73046 (2021).
- Wang X-J. *Probabilistic decision making by slow reverberation in cortical circuits.* Neuron 36:955–968 (2002).
- Kopsick J.D., Kilgore J.A., Adam G.C., Ascoli G.A. *Formation and Retrieval of Cell Assemblies in a Biologically Realistic Spiking Neural Network Model of Area CA3.* (2024) PMC10996657.
