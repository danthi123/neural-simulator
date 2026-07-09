# D3 recurrent multi-hop composition — first arc: a BPTT-RNN LEARNS same-length group composition but does NOT length-generalize (representability ≠ learnability); the harness + learnability oracle are validated; the next lever is intermediate-state supervision / e-prop

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_group_composition_derisk.py` (numpy CPU, NO `sim/` edit). Research gate: `2026-07-09-D3-recurrent-multihop-composition-research-gate.md` (subagent-scoped, controller-verified).
**Verdict:** honest NEGATIVE for end-to-end length generalization (3 levers), harness-validated, next mechanism named.

## The frontier + the task (controller-verified theory)
The feedforward deep-credit arc is COMPLETE (`2026-07-08-deep-credit-feedforward-arc-COMPLETE...md`): FF depth-benefit is real but narrow (nonlinear conjunction over a pooled rep, e.g. XOR-over-pool). The genuine depth-required LANGUAGE capability is RECURRENT/multi-hop composition. THE TASK (research-gated): **streaming non-abelian GROUP-WORD composition = STATE TRACKING** — a sequence of group elements g_1..g_L arrives one per step; the running product `s_t = s_{t-1}·g_t` is the literal bind→re-bind; target = a 2-way property (random 2-coloring) of the FINAL state; each element = a noisy ±1 pool code (no lookup shortcut); non-abelian (S3) → order matters (permuted-order collapses). **Theory (I verified myself, not the summary):** an RNN simulates a DFA and group composition IS a DFA → linear-size RNN-representable; transformers/diagonal-SSMs are TC⁰ (Merrill-Petty-Sabharwal, arXiv:2404.08819 "Illusion of State" — I fetched it: it proves SSMs *cannot* express permutation composition; it does NOT itself state "1-layer RNN solves S5", so I use the elementary RNN-simulates-DFA argument + Barrington 1986 NC¹-hardness of A5). The EMPIRICAL separation for solvable S3 is **length generalization**: FF over the flattened sequence memorizes trained lengths but can't ITERATE; a true iterative RNN generalizes.

## The de-risk (3 arms + a LEARNABILITY ORACLE, the load-bearing design fix)
Arms: (a) **FF-oracle** (MLP over flattened seq, depth 0/1/2/3 — the feedforward ceiling); (b) **fixed reservoir + ridge**; (c) **BPTT-RNN** (`h_t=tanh(W_rec h_{t-1}+W_in x_t)`, read `h_L`, full BPTT — the ~40-line NEW piece, since `sim/bptt_snn.py` has only within-neuron leak recurrence, NO recurrent weight `W_rec`, grep-confirmed). Two eval splits: **SAME** (held-out seqs at TRAIN lengths = the learnability oracle) and **DEEPER** (held-out LONGER = the generalization test). Anti-cheats: recurrence-OFF lesion, permuted-ORDER (order-matters frac), 1-hop Markov floor, multi-seed.

## RESULT (S3, seed 42) — the RNN LEARNS but does NOT length-generalize

| lever | SAME (learnability) | DEEPER (generalization) |
|---|---|---|
| FF-oracle | 0.828 | 0.504 |
| fixed reservoir + ridge | 0.607 | 0.503 |
| **BPTT-RNN (vanilla, train 1/2/3)** | **0.883** (lesion 0.653) | **0.521** (lesion 0.481) |
| BPTT-RNN + WIDER curriculum (train 1-5) | 0.698 | 0.504 |
| BPTT-RNN + ORTHOGONAL W_rec (train 1/2/3) | 0.862 | 0.522 |

**Reading (honest):** the learnability oracle PASSES — the BPTT-RNN genuinely LEARNS same-length group composition (0.88 > FF 0.83 > reservoir 0.61; the lesion collapses same-length 0.65<0.88, so the recurrence is load-bearing for even representing it; order-matters 0.41 confirms the non-abelian task is order-dependent). **But NO arm length-generalizes** — every DEEPER acc sits at chance (~0.50), the RNN included. Neither a wider-length curriculum (which HURT same-length, 0.88→0.70 — more lengths, harder to memorize) nor orthogonal (norm-preserving) W_rec init lifted DEEPER. ⇒ **the BPTT-RNN finds a LENGTH-SPECIFIC solution, not the ITERATIVE update** — the theoretical recurrent advantage (an RNN CAN represent the DFA) does NOT materialize under vanilla end-to-end BPTT (it does not LEARN the generalizing iteration). This is the classic **representability ≠ learnability** gap = the field-wide systematic/length-generalization wall, exactly the D3 target.

## What this establishes + the next lever (the arc continues — NOT a wall)
- **The harness is validated + reusable** (learnability oracle, 3 arms, anti-cheats, configurable group/lengths, S3→A5 ready). The fixed reservoir is confirmed CAPPED (0.51 deeper) — fixed dynamics is not enough (as predicted).
- **The honest boundary:** vanilla BPTT does not learn length-generalizing composition from end-of-sequence labels alone. This LAUNCHES the next mechanism (per the workflow), it does not close the question:
  1. **INTERMEDIATE-STATE SUPERVISION** (the highest-leverage cheap next lever): supervise the running state `s_t` at EVERY step (not just the final) → forces the RNN to learn the group-multiplication UPDATE RULE, which then generalizes to any length. Biologically apt (per-step feedback) and the known algorithmic-length-generalization fix. This tests whether a recurrent credit path CAN learn the iterative composition WITH the right training signal.
  2. **e-prop** (bio-plausible local recurrent credit; the teacher-forced classification regime where EMERGE-6b's e-prop DID learn the one-step map, distinct from the free-run GENERATION regime that failed 5×).
  3. **A5** (non-solvable → the FF route is provably NC¹-hard, sharpening the separation) — but only meaningful once a lever length-generalizes on S3.
- **Cross-cutting:** the recurrent LANGUAGE-depth frontier is genuinely open + hard (the field hasn't solved end-to-end length generalization either); the mission is to solve it under the constraints. The next rung (intermediate-state supervision) is building now.

## Files
`research/runners/_d3_group_composition_derisk.py` (`--group S3/S4/A5 --train-lens --test-lens --orthogonal-rec`); research gate `2026-07-09-D3-recurrent-multihop-composition-research-gate.md`.
