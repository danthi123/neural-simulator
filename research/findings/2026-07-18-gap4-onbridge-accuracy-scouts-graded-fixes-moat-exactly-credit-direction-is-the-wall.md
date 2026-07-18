# gap#4 keystone — on-bridge learning-to-accuracy SCOUTS (2026-07-18): the graded `E·P` credit fixes the moat EXACTLY (lesion hidden-dw 0.000) + flows strong credit + no weight transport, but at cheap scale held-out ≈ chance-or-below = the CREDIT-DIRECTION wall (2026-07-14), NOT read-variance. The D1-scale accuracy run is the genuine test. Two `sim/`-free runner levers added (`--graded-credit`, threaded).

**2026-07-18.** The board's named gap#4 NEXT was the never-completed on-bridge learning-to-accuracy run
(`_d1_onbridge_learn_to_accuracy`, the depth-2 emerge1 XOR-of-pairs→majority-threshold task, generalizing). Scouted it;
records the mechanism-confirmation + the two candidate credit forms + the honest scope (a-1 RAG).

## Scouts (single-seed, emerge1, seed 42)
| config | B_apical | held BDSP | held LESION | held wrong-sign | chance | note |
|---|---|---|---|---|---|---|
| microcircuit + bistable, h24 ep12 | 0.005 | 0.615 | 0.615 (dw 90) | 0.615 | 0.549 | moat LEAKS (measured-B) |
| microcircuit no-bistable, h24 ep12 | 0.044 | 0.615 | 0.615 (dw 143) | 0.615 | 0.549 | bistable REDUCES B here |
| microcircuit + graded, h24 ep12 | — | 0.615 | 0.615 (**dw 0.000**) | 0.615 | 0.549 | **moat holds EXACTLY** |
| microcircuit + graded, **h128 ep100 train64** | — | 0.406 | 0.406 (**dw 0.000**) | 0.406 | 0.549 | moat exact; credit flows (dw 100) but **held-out below chance** |

## What the scouts ESTABLISH
1. **The graded `E·P` form FIXES the moat EXACTLY.** Measured-B credit `dev = B − Pbar·E` leaks the moat: with
   `B_rest=0` but `bdsp_p0=0.30`, `dev_rest = 0 − 0.30·E < 0` → spurious LTD → the apical-lesion arm still moves the
   hidden weights (dw 90–143). The graded form `dev = E·(P − Pbar)` has `P=Pbar` at rest → `dev=0` exactly → **lesion
   hidden-dw = 0.000** (perfect moat) — the no-spurious-learning moat holds by construction. It is ALSO bidirectional
   (P∈[0,1] around Pbar signals LTP AND LTD; the measured B≥0 can't fall below rest to signal LTD). Biologically the
   Larkum BAC-firing analog dendritic-Ca²⁺ coincidence (somatic event × apical plateau strength) — a three-factor rule
   over neural quantities (spiking E × apical voltage), the analog dendritic form the point-neuron-limit position
   endorses. Committed as the runner lever `--graded-credit` (sets the existing `enable_bdsp_graded_credit`; NO `sim/`
   edit; the `sim/` mechanism was added 2026-07-12).
2. **BUT the credit DIRECTION is the wall, not read-variance or the moat.** At h128 the graded credit flows strongly
   (BDSP hidden-dw 100) yet held-out 0.406 = lesion = wrong-sign, BELOW chance — the feedback-alignment credit moves
   weights in a NON-solving direction. This reproduces the 2026-07-14 decisive result (graded 0/6 ≈ binary on the
   semantic-inheritance task; the wall is CREDIT-STRUCTURE = the FA direction at depth on point-neurons) — now seen on
   emerge1 too, at cheap scale.
3. **The scouts are badly under-scaled on DATA + epochs.** train_subset 64 (of ~665 train patterns), ep 100, online
   (batch=1). D1's on-bridge accuracy config was train_subset 200, ep 300–600, batch-tuned → 0.664 (measured-B, below
   the 0.75 bar) and the numpy microcircuit reference reached 0.964. A generalizing depth-2 task cannot be learned from
   64 examples. ⇒ the cheap scouts CONFIRM the mechanism (moat/credit/no-transport) but cannot test ACCURACY.

## Honest scope (a-1 RAG, prevents drift#12 re-derivation)
- The graded path was already REFUTED 6-seed on the **semantic-inheritance HARD task** (`2026-07-14`); the wall there is
  credit-STRUCTURE, and surrogate-BPTT reaches 0.972/0.673 (the spiking substrate is VIABLE; the LOCAL rule's
  weight-finding is the wall). `emerge1` is a DISTINCT depth-2 target (D1: probe 0.92, held-out 0.664 batch-fragile) —
  the board-named accuracy run — so testing it at D1 scale is legitimate, NOT a re-derivation.
- **The genuine open gap#4 accuracy question:** does the on-bridge microcircuit/graded rule clear the 0.75 held-out bar
  on emerge1 at the D1 scale (train 200–300, ep 300–600)? D1's on-bridge measured-B never cleared it (0.664); only the
  numpy microcircuit did (0.964). **RUNNING: the D1-scale graded run (h128, ep300, train300) — if it clears, that's the
  completed accuracy milestone; if it stalls at ~chance, the credit-DIRECTION wall bites emerge1 too and the next
  method is a fresh research gate for an on-spike credit-DIRECTION rule (not graded — spent).**

## Infra
`--graded-credit` runner flag (+ `graded_credit` threaded through `OnBridgeBDSPNet`/`_run_bridge_arm`); NO `sim/` edit.
Determinism CI 9/9 green (the earlier structured-BTSP `sim/` edits are byte-safe broadly).
