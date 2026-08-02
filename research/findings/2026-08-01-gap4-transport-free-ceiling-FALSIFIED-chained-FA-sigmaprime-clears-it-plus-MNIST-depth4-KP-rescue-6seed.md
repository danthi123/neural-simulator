---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/graded_feedback_ladder/ladder_6seed.json
  - research/findings/raw/gap4/layerwise_kp_6seed.json
  - research/findings/raw/gap4/graded_feedback_ladder/cube_424344.json
  - research/findings/raw/gap4/graded_feedback_ladder/cube_454647.json
---

# gap#4 crux — the "fundamental transport-free ceiling" (b7549514) is FALSIFIED: a transport-free local rule (chained multi-hop FA + σ′ + graded credit) clears the depth-2 ceiling 6-seed and KP-learned feedback rescues MNIST depth-4 — the wall was direct-DFA-without-σ′, NOT the transport-free credit class

<!--derived-->
**One-line verdict.** b7549514's MEASUREMENTS stand (its two methods — direct one-hop DFA and binary
coincidence-gated BDSP — do cap at held-out ~0.63 on the depth-2 XOR→threshold toy), but its INFERENCE —
"a FUNDAMENTAL limit of the local transport-free credit class ... the frontier is a different-paradigm
(equilibrium-propagation) question" — is **FALSIFIED**. A transport-free local rule that adds the two factors
those methods lacked — **chained multi-hop feedback + the activation-derivative σ′** — clears the ceiling
(6-seed held-out 0.935 vs banked 0.63, oracle 0.974) and, on MNIST at genuine depth, transport-free
**KP-learned feedback rescues depth-4** where fixed-random FA collapses (FA 0.531 → KP 0.876, 6/6). This
matches the external SOTA that named the exact missing factor: WF-Act-PC ([arxiv 2607.13380](https://arxiv.org/html/2607.13380v1))
shows FA collapses at depth precisely because it drops σ′, and Frozen-Backprop ([arxiv 2505.13741](https://arxiv.org/pdf/2505.13741))
handles the transport-free feedback half by periodic sync. This finding is a RATE reference (numpy/CPU); the
spiking port is the open frontier. No `sim/` edit (subclass + additive `--feedback-ladder`; runner untouched
when off, byte-identical verified).

## The overturn — the toy 6-seed ladder (emerge1 depth-2 XOR→threshold, single variable = descending feedback)

<!--derived-->
Graded credit (no binary gate) + σ′ at every hidden layer; the feedback matrix is the only variable. Artifact:
`research/findings/raw/gap4/graded_feedback_ladder/ladder_6seed.json` (numpy/CPU, 6 seeds 42/43/44/100/101/102).

| arm | held-out (6-seed mean) | note |
|---|---|---|
| oracle / truegrad (W⊤) | 0.974 | reproduces backprop exactly, all seeds |
| **dfa — fixed-random Y, chained + σ′, TRANSPORT-FREE** | **0.9355** (per-seed 0.994/0.958/0.947/0.975/0.858/0.880) | clears the banked ~0.63 ceiling on EVERY seed |
| kp — learned Y, transport-free | 0.8273 | not needed at depth-2 (a tuning miss; collapses to 0.591 on 1 seed) |
| chance | 0.5241 | |

Anti-cheats clean 6/6 (dfa permuted → 0.48–0.53 ≈ chance; wrong-sign → chance). Held-out is genuinely
disjoint (all 1024 patterns enumerated, `input_overlap=0` every seed; a linear probe on RAW inputs is below
chance 0.42–0.47 while on the hidden reps it reaches 0.76–0.98 — real emergent nonlinearity, not memorization).

## The MNIST real-task confirmation — transport-free learned feedback rescues DEPTH (6-seed)

<!--derived-->
Artifact: `research/findings/raw/gap4/layerwise_kp_6seed.json` (layerwise recursion `e_l = (e_{l+1} @ Fb)·σ′`,
hidden 128, n_train 8000, 6 seeds; transport-free asserted in-run — the KP feedback update contains no `self.W`).

| arm | depth-2 (784-128-128-10) | depth-4 (784-128⁴-10) |
|---|---|---|
| reservoir (frozen hidden) | 0.773 | 0.114 = chance (fails at depth) |
| FA (fixed-random) | 0.932 | 0.531 (degrades; seed-UNSTABLE 0.10–0.70) |
| **KP (learned, transport-free)** | 0.934 | **0.876 (STABLE 0.85–0.89)** |
| backprop | 0.944 | 0.929 |
| KP closes FA→BP gap | 0/6 (FA already suffices) | **6/6** |

At depth-2 fixed-random FA already ≈ backprop (nothing to close). At depth-4, FA genuinely breaks and
transport-free KP-learned feedback closes the gap on every seed (kp_permuted 0.09–0.13 ≈ chance/10). This is
the WF-Act-PC / July-D2 learned-apical-feedback mechanism — the one I had WRONGLY banked as "KP hurts."

## Corrected attribution — the binary gate was a RED HERRING (adversarial verification refined my own claim)

<!--derived-->
A clean single-variable 2×2×2 cube (gate: binary/graded × feedback: direct/chained × σ′: off/on, 6 seeds 42–47;
`cube_424344.json` + `cube_454647.json`) gives the marginal main effects (off→on, mean over the other two
factors). **My earlier report — that "the 0.63 wall was entirely the binary coincidence gate" — is REFUTED by
this cube.**

| factor | main effect | reading |
|---|---|---|
| **σ′ off→on** | **+0.230** (sd 0.014, largest + tightest) | strictly NECESSARY — off collapses the headline 0.951→0.465 |
| feedback direct→chained | +0.123 (sd 0.025) | largest single-flip lift (+0.330) but ONLY with σ′ on |
| gate binary→graded | **−0.070** (sd 0.030, NEGATIVE) | the gate is a RED HERRING — removing it does not clear the plateau |

The feedback × σ′ interaction (+0.301) exceeds any single main effect: no single flip from the canonical
plateau arm (binary·direct·σ′-off = 0.653) clears the ceiling; only the CONJUNCTION chained-feedback + σ′-on
does (graded·chained·σ′-on = 0.951, while graded·chained·σ′-off = 0.465 and graded·direct·σ′-on = 0.621 are
both still at/below plateau). So b7549514 plateaued because its methods were **direct-DFA AND lacked σ′**, not
because BDSP was binary. The honest cause is σ′ (necessary, largest) + chained multi-hop feedback (jointly).

## Adversarial verification — 4/4 skeptics hold (workflow wf_417c0e53-569)

<!--derived-->
- **Transport-free:** no leak. dfa credit reads only fixed-random `Y` + local activity, never a forward `W`
  (W⊤ appears only in the truegrad oracle arm); `Y` is byte-identical (SHA256) before/after an epoch for dfa
  (frozen), while the kp control's `Y` changes; σ′ is label-independent (acts(X,y)==acts(X,shuffled_y) to
  0.00e+00); cos(Y, W⊤) ≈ 0 at init.
- **Leakage / anti-cheats:** held-out genuinely disjoint (input_overlap=0 all 6 seeds); anti-cheats collapse to
  chance on every seed.
- **Net-depth:** the overturn SURVIVES net-depth 3 AND 4 (the graded rule is already depth-general). The genuine
  depth ceiling beyond ~4 is the SUBSTRATE's optimizability (deep-sigmoid + Xavier + plain-momentum SGD, which
  even true backprop cannot train past ~4 hidden layers), NOT the transport-free credit path.
- **Independent reimpl:** a fully independent ~65-line numpy reimplementation reaches dfa 0.953/0.950/0.955
  (seeds 42–44), clearing the ceiling — not a runner-specific bug.

## Honest scope + residuals

<!--derived-->
- **Scope:** a numpy/CPU DEPTH-2 toy result (emerge1 10-bit XOR→threshold, [10,64,64,2], 300 epochs), plus the
  MNIST rate confirmation at depth-4. Credit-path validated to survive net-depth 4. NOT yet on spikes.
- **Noisier than the oracle:** dfa drops to 0.858/0.880 on 2 of 6 headline seeds; the KP sibling collapsed to
  0.591 on one seed; worst-case train−heldout gap ~0.14 (oracle ~0.04). It generalizes on every seed but is not
  oracle-tight.
- **It is a STRONGER RULE, not a mismeasurement.** b7549514's ~0.63 for ITS methods stands; the ceiling is
  cleared by ADDING chained-feedback + σ′. This is why the disposition is AMEND (the inference), not a data
  retraction.

## Disposition + next

b7549514 is AMENDED: its "fundamental limit of the transport-free class / different-paradigm question" headline
is superseded by this finding; its measurements are retained. Mechanism entry `deep-credit-on-spikes.md` gets a
rate-reference reconciliation note (the spiking `deep_credit_share ≈ 0` is a spiking-substrate-optimizability
issue, NOT evidence that the transport-free credit CLASS is walled — the rate reference shows it is not). **Next
(the frontier, un-deferred by speed-secondary): the SPIKING port** — chained/KP transport-free feedback + a
graded low-CV credit read (σ′(v−θ) = distance-to-threshold, per the 2026-07-14 "graded-credit-decisive" note)
at a real budget, designed to NOT repeat that finding's failure (it tested graded credit on spikes with plain-FA
at cheap scale, never KP-learned feedback at real depth).
