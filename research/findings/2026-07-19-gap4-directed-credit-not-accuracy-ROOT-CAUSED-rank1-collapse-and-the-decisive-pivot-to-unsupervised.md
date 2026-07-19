# gap#4 keystone — directed-credit-≠-accuracy ROOT-CAUSED (rank-1 linear-discriminant collapse on a zero-discriminant task + a saturated hidden gate), a RE-DERIVATION of a concluded boundary + the decisive 4-lens pivot to the UNSUPERVISED stream cortex.

**2026-07-19.** This session ran the board's named gap#4 action to accuracy. The mechanism PIPELINE-VALIDATES (soma-coupling
delivers 927× clean directed hidden-layer credit, moat holds, no weight transport — `2026-07-19-gap4-soma-coupling-flips-BOUNDARY-to-PIPELINE-VALIDATED`),
but held-out accuracy stays at the FLOOR on a clean discriminator task (`cleanxor`: oracle 0.994, linear floor 0.514=chance,
margin 0.48): **BDSP 0.486 == LESION == WRONG-SIGN == chance.** A 4-lens diagnosis workflow (our-record RAG · external
literature · mechanism trace · alternative methods · synthesis; `wf_8de0688b-dcf`, 1.09M subagent tokens) root-caused it
and delivered a decisive mission call.

## THE ROOT CAUSE (Lens 3 mechanism trace — the cleanest yet, supersedes prior "the block is the rule")
On `cleanxor` the on-bridge BDSP hidden update **collapses to a rank-1 linear-discriminant learner, and cleanxor's linear
discriminant is identically zero by construction** — so the 927× directed movement lies WHOLLY in a zero-information
subspace = orthogonal to the XOR solution. Two compounding causes:
1. **Rank-1 collapse (2-class scalar error).** `e = onehot − softmax`, `sum(e)=0` ⇒ `e=[e₀,−e₀]` = ONE scalar. Projected
   through the fixed/learned feedback `Y`, every hidden unit gets the SAME scalar `e₀ × d_j` (`d_j=Y_{0j}−Y_{1j}`, frozen).
   The in→hid update accumulates to `dw[k,j] ∝ d_j·(⟨E_in|class0⟩ − ⟨E_in|class1⟩)` = ±the class-mean-input-difference =
   the linear discriminant. **cleanxor kills that vector:** `P(bit=1|class0)=P(bit=1|class1)=0.5` for every bit (that IS
   the 0.514 floor); the XOR info lives entirely in the 2nd-order `latA==latB` correlation a single-scalar-through-Y credit
   can never teach.
2. **Saturated hidden gate.** ML feedback-alignment solves XOR because the hidden delta `Y@e` is gated by `f'(a_j)` — an
   input-dependent per-unit nonlinearity that lets random units specialize. The only substrate analog is the hidden event
   rate `E_hidden_j`, but the runner's standing `hidden_bias=520 pA` SATURATES it into a near-constant, input-insensitive
   rate ⇒ no input×hidden product nonlinearity ⇒ the rule degenerates to the rank-1 push AND overwrites the random
   nonlinear init basis rather than refining it.
- **wrong-sign-at-floor is the exact fingerprint:** negating `e` gives the negated (still zero-information) discriminant ⇒
  still chance, NOT below. Below-chance would need a task-aligned component to anti-compute; there is none. (The 2026
  FA-evaluation paper "What Accuracy and Gradient Cosine Miss," arXiv 2606.21126, names this: fixed-random FA "fails to
  demonstrate depth utility — performance remains near the shallow/linear floor.")

## LITERATURE (Lens 2) — the field reaches accuracy ONLY with LEARNED feedback, but that is necessary-NOT-sufficient here
The field DOES reach accuracy with local burst credit — but on depth-requiring tasks ONLY with LEARNED (→symmetric,
Kolen-Pollack) feedback, not fixed-random FA: BurstCCN (Greedy 2022) CIFAR-10 **39% random → 23% symmetric** (≈ANN);
Payeur 2021 Burstprop matches BP / learns ImageNet ONLY with learned feedback; Akrout 2019 weight-mirror/KP near-match BP,
plain FA insufficient at scale; Bartunov 2018 FA fails beyond MNIST. **BUT** KP fixes the DIRECTION of a *linearly-informative*
credit; cleanxor's credit is rank-1 with a zero discriminant → nothing to align to. KP is necessary-not-sufficient HERE;
per Lens 3 it must be paired with (a) hidden-gate desaturation (restore the `f'`-analog) + (b) rank-breaking (≥3 classes /
auxiliary parity heads). **The running FA-direction sweep (KP + epochs/width/lr/gain) omits BOTH root-cause levers, so an
all-at-floor sweep must NOT be misread as "boundary confirmed"** — it never tested the binding constraint. The correctly-
levered decision-gate (hidden-bias desaturation, this session's `desat_*` probes) is the honest test.

## THIS WAS A RE-DERIVATION (Lens 1 — drift #12, logged honestly)
The project record has MULTIPLY concluded on-bridge supervised local-credit-to-accuracy is a boundary and the mission path
is UNSUPERVISED: `2026-07-14` (graded credit 0/6, "the block is the RULE"), `2026-07-16` (the only on-bridge accuracy
positive was ~80% a fixed reservoir; "6-seed GO" = 3 dev seeds, SIGNAL=False), `2026-07-17-THE-SEED-NEVER-CONTROLLED...`
(FULL vs FROZEN measured on DIFFERENT neurons — unseeded confound), `2026-07-17-...commit-to-the-unsupervised-path`,
`2026-07-19-gap4-keystone-accuracy-NEGATIVE` (degenerate readout). I re-fired the gate. **Genuine value-add this session:**
the soma-coupling PIPELINE-VALIDATION (directed credit demonstrably works — a real gate the record listed as "never
completed") + the CLEAN rank-1 mechanistic root-cause (more precise than "the block is the rule") + the correctly-levered
decision-gate. The lesson (drift #12): RAG the record BEFORE re-investigating a boundary — the a-1 check would have flagged
this as concluded.

## THE DECISIVE MISSION CALL (all 4 lenses converge) — PIVOT the primary to the UNSUPERVISED stream cortex
- **The mission does not need supervised deep credit:** gaps #2/#3/#5 are CLOSED without it; gap#1 (open fluent generation)
  is served by the minimized-transformer scaffold behind the gate + unsupervised grounding codes; EMERGE-30..55 already
  deliver deep, generalizing, category/grammar/grounded codes. Real language depth is **linear/compositional** ("depth
  lives in composition"); the true remaining supervised frontier is RECURRENT off-diagonal credit — a different problem
  than cleanxor.
- **Biology is on the unsupervised side** (cortex learns overwhelmingly locally/predictively/unsupervised) → more
  biology-faithful, emergent by construction, one-brain, already validated.
- **Keep supervised alive only as** (a) this cycle's correctly-levered decision-gate (`desat_*` + the FA sweep), then close
  either way, and (b) a thin PREDICTIVE-CODING track (dual-use with the unsupervised predictive cortex; escapes rank-1 FA
  by construction via relaxation-to-equilibrium → a task-aligned hidden error; `sim/predictive_coding.py` already commits
  the Rao-Ballard error). Do NOT put supervised-deep-credit-to-accuracy back on the critical path.

## Status + the gate
- **DONE:** mechanism pipeline-validated; root-caused (rank-1 collapse + saturated gate); 4-lens diagnosis; re-derivation
  acknowledged. **IN FLIGHT:** the FA-direction sweep (KP/epochs/width/lr/gain — predicted all-at-floor per Lens 3) + the
  correctly-levered desaturation probes (hidden-bias 260/160/90 + KP + differential-readout — the root-cause test). GATE:
  held-out > floor AND wrong-sign < chance. **Then, EITHER WAY:** formally state gap#4-supervised-to-accuracy's honest
  status + PIVOT the mission-primary to the unsupervised stream cortex.
- NO `sim/` edit (the BDSP kernel + apical path are correct; the limitation is runner operating-point). Diagnosis workflow:
  `wf_8de0688b-dcf`; runner `_d1_onbridge_learn_to_accuracy_derisk.py` (`--task cleanxor`, `--soma-g`, `--feedback learned`,
  `--hidden-bias`).
