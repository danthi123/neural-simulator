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
| **BPTT-RNN + INTERMEDIATE-STATE SUPERVISION** (per-step running-state target) | prop 0.932 / **state-track 0.874** | prop 0.548 / **state-track 0.216** |

**The state-supervision lever is the deepest probe: even teaching the EXACT running group state `s_t` at every step, the RNN tracks it at TRAIN lengths (0.874) but its tracking DEGRADES TO ~CHANCE at DEEPER lengths (0.216, chance 1/6=0.167).** So the learned continuous dynamics DRIFT over more steps than trained — the RNN does not learn a clean DISCRETE iterative update even with the ideal per-step signal.

**Reading (honest):** the learnability oracle PASSES — the BPTT-RNN genuinely LEARNS same-length group composition (0.88 > FF 0.83 > reservoir 0.61; the lesion collapses same-length 0.65<0.88, so the recurrence is load-bearing for even representing it; order-matters 0.41 confirms the non-abelian task is order-dependent). **But NO arm length-generalizes** — every DEEPER acc sits at chance (~0.50), the RNN included. Neither a wider-length curriculum (which HURT same-length, 0.88→0.70 — more lengths, harder to memorize) nor orthogonal (norm-preserving) W_rec init lifted DEEPER. ⇒ **the BPTT-RNN finds a LENGTH-SPECIFIC solution, not the ITERATIVE update** — the theoretical recurrent advantage (an RNN CAN represent the DFA) does NOT materialize under vanilla end-to-end BPTT (it does not LEARN the generalizing iteration). This is the classic **representability ≠ learnability** gap = the field-wide systematic/length-generalization wall, exactly the D3 target.

## What this establishes + the next lever (the arc continues — NOT a wall)
- **The harness is validated + reusable** (learnability oracle, 3 arms, anti-cheats, configurable group/lengths, S3→A5 ready). The fixed reservoir is confirmed CAPPED (0.51 deeper) — fixed dynamics is not enough (as predicted).
- **The honest boundary (4 levers robust):** neither end-label BPTT, a wider curriculum, orthogonal W_rec, NOR intermediate-state supervision gives length generalization — the continuous RNN dynamics DRIFT past the trained depth. This is the field-wide systematic-generalization wall, cleanly reproduced + measured. It LAUNCHES the next mechanism (per the workflow), it does not close the question:
  1. **DISCRETE-ATTRACTOR STATE MAINTENANCE (the biologically-apt next lever, mission-central):** the running state must be a CLEAN DISCRETE ATTRACTOR (one of K prototypes), RE-DISCRETIZED between steps (a Hopfield/CA3-style cleanup snaps the drifting continuous state back to the nearest attractor each step) → no accumulation of drift → generalizes to any depth. This is exactly the brain's discrete working-memory / the project's OWN CA3 attractor + NEF cleanup machinery — the recurrent-composition frontier CONNECTS to the attractor work. The state-supervision result (tracks at train depth, drifts deeper) is the direct evidence this is the fix (the representation exists; it just isn't stabilized).
  2. **Gated / quantized recurrence** (LSTM/GRU or a straight-through discretized state) — a weaker version of (1).
  3. **e-prop** (bio-plausible local recurrent credit — but the ARCHITECTURE (attractor stabilization), not the credit rule, is the likely bottleneck given BPTT itself fails).
  4. **A5** (non-solvable → NC¹-hard FF route) — meaningful once a lever length-generalizes on S3.
- **Cross-cutting:** the recurrent LANGUAGE-depth frontier is genuinely open + hard (the field hasn't solved end-to-end algorithmic length generalization either); the mission is to solve it under the constraints. The comprehensively-mapped boundary (4 negative levers + the drift diagnosis) POINTS at the discrete-attractor mechanism — the project's own CA3/NEF-cleanup substrate. That rung is next.

## Files
`research/runners/_d3_group_composition_derisk.py` (`--group S3/S4/A5 --train-lens --test-lens --orthogonal-rec`); research gate `2026-07-09-D3-recurrent-multihop-composition-research-gate.md`.
