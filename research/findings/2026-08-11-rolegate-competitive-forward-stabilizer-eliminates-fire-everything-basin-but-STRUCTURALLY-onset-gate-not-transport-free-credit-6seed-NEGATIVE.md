---
type: finding
status: contributing
claim_check: synthesis
date: 2026-08-11
mechanism: ROLE-GATE x COMPETITIVE FORWARD STABILIZER (LEVER 3) — add a competitive/normalising forward stabilizer to the 2-layer role-gate's OWN populations, trained WITH the transport-free chained-FA+sigma' / KP rule. (1) OUTPUT feedback inhibition: a pooled inhibitory integrator of the load-unit's own within-sentence output subtracts from the load logit (z2_eff = z2 - out_lambda*s_inh; s_inh resets per sentence) -> repeated loading self-limiting, the all-load fixed point dynamically UNREACHABLE (the WM one-update-at-a-time gating / spike-frequency-adaptation motif). (2) HIDDEN divisive normalisation (Carandini-Heeger): h_i = r1_i/(1+div_k*mean_j r1_j), keeping the hidden ensemble out of the saturated regime where sigma' vanishes. Replaces the scalar homeostatic-bias proxy LEVER 2 used for the competition the real WM circuit runs.
lane: emergence engine / working memory x gap#4 / role-gate transport-free reliability
verdict: 6-SEED (42 43 44 100 101 102) real spiking D3 slot at L=2/3/4 (GO distance = L4, chance 0.250, marker ceiling 1.000/all seeds/all L, held-out NOVEL fillers). HONEST NEGATIVE (role_go=False) — RELIABLE ROLE ACHIEVED but STRUCTURAL, not from the transport-free CREDIT. The competitive stabilizer ELIMINATES the LEVER-2 seed-dependent FIRE-EVERYTHING basin: transport-free chained-KP goes from NOSTAB 0.472 [min 0.133], fire pos>0 max 0.24 (the collapse) to STAB 1.000 [min 1.000], fire 1.00/0.00, gap +1.00 [min +1.00] on ALL 6 seeds, and the stabilizer is LOAD-BEARING (the lesion = stabilizer-off re-collapses to min 0.133; the aligned transport ceiling stays 1.000 WITH the stabilizer; the identity crux fails gap +0.00). BUT the permuted-reward anti-cheat REACHES THE SAME role (KP-stab permuted acc 1.000 [min 1.000], gap +1.00) AND an UNTRAINED stabilized gate (zero learning, random weights) already scores 1.000 [min 1.000], fire 1.00/0.00 on all 6 seeds — so the role is produced ENTIRELY by the STRUCTURE, not the learning signal. The feedback-inhibition fire-ONCE budget gives ONLY position 0 zero accumulated inhibition, so the competition is inherently an ONSET gate that loads the first token = the subject in THIS subject-first stream, with NO credit. The fire-everything basin is eliminated by structural competition, but "does the TRANSPORT-FREE CREDIT induce role" is now UNTESTABLE here (onset == the answer). chained-FA + stabilizer is mixed (over-suppresses the collapse seeds toward silence: 0.422 [min 0.222] -> 0.761 [min 0.178]) — the noisy fixed feedback fights the onset prior on some seeds. NEXT LEVER: a VARIABLE-subject-position stream where onset != the answer, so the structural onset prior can no longer solve it and the transport-free credit becomes testable. NO sim/ edit; SIM_BACKEND=numpy.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_var_bind_rolegate_competitive_stabilizer_derisk.py
artifacts:
  - research/findings/raw/_rolegate_competitive_stabilizer/competitive_stabilizer_6seed.json
  - research/findings/raw/_rolegate_competitive_stabilizer/untrained_stabilized_gate_smokinggun.json
---

# LEVER 3 (competitive forward stabilizer): the fire-everything basin is eliminated — but STRUCTURALLY, by an onset gate, not by the transport-free credit

## The question this lever attacked

LEVER 2 (`2026-08-11-rolegate-hidden-layer-chained-FA-sigmaprime-...-NEGATIVE-ceiling-clears-6seed.md`) isolated the
role-gate's transport-free residual to a **seed-dependent collapse into a fire-everything basin**: the transport-exact
(aligned) credit reaches role 1.000/6, but the transport-free chained-FA/KP arms collapse on some seeds (fire pos0 ≈
fire pos>0). It is NOT depth, NOT sigma', NOT feedback alignment (co-adapting KP recovers cos(B,Wᵀ) 0.92–1.00 on every
seed yet role still collapses). The standing lesson — *"at a wall, ask what else the real system runs alongside this that
we replaced with a CONSTANT"* — points at **COMPETITION**: cortical/WM circuits run lateral inhibition + divisive
normalisation that structurally forbid the all-active state, and LEVER 2 proxied that with only a scalar homeostatic-bias
nudge. Hypothesis: a competitive/normalising forward stabilizer that makes fire-everything dynamically unreachable,
trained WITH the transport-free rule, eliminates the collapse → reliable transport-free role.

## What was built (runner-side; NO sim/ edit)

`research/runners/_var_bind_rolegate_competitive_stabilizer_derisk.py` subclasses the LEVER-2 `HiddenChainedGate` and
adds a competitive forward stabilizer to the gate's own populations. The fire-everything basin is **temporal** (the
single scalar load-unit fires at *every* position rather than only pos0), so the biologically-faithful competition is:

- **Output feedback inhibition (the load-bearing one).** A pooled inhibitory interneuron integrates the load-unit's own
  recent output within the sentence — `s_inh_t = leak*s_inh_{t-1} + p_{t-1}` — and subtracts from the load logit:
  `z2_eff = z2 - out_lambda*s_inh`. `s_inh` resets to 0 each sentence, so the FIRST load faces zero inhibition (never
  suppressed), but after ~one load `s_inh≈1` clamps every subsequent load → the all-load fixed point is dynamically
  UNREACHABLE. This is the WM "one-update-at-a-time" gating motif (BG-thalamocortical loops re-inhibit after a stripe
  updates; O'Reilly & Frank 2006 PBWM) / spike-frequency adaptation (M-current) as a biophysical "fire-once" budget.
- **Hidden divisive normalisation (Carandini & Heeger 2012).** `h_i = r1_i/(1 + div_k*mean_j r1_j)` keeps the hidden
  ensemble out of the saturated regime where the hidden sigma' (`r1(1-r1)`) vanishes and the chained-FA credit has no
  gradient. The diagonal local slope `sp1_eff = hidden_gain*r1(1-r1)/denom` is used in the credit (transport-free, local
  — no cross-unit Jacobian, no weight transport).

When `stabilize=False` the gate DELEGATES to the LEVER-2 gate verbatim → the no-stab arms are byte-identical to the
banked collapse baseline AND serve as a clean lesion. Everything else (the SAME-POOL positional stream, the REAL spiking
D3 `SpikingSlot` at eval, the token-identity crux, held-out NOVEL fillers, marker/HTM/n-gram/perm-pos teeth) is
reuse-by-import. `out_lambda=4.0, div_k=2.0, inh_leak=1.0, b2_init=0.3, homeo=0.10, hidden=32, episodes=80`.

## Result — 6-seed at the GO distance L=4 (dist 5, chance 0.250, marker ceiling 1.000)

| arm | acc mean [min] | fire pos0/pos>0 (pgt0 max) | id-gap [min] |
|---|---|---|---|
| identity (code-only crux control) | 0.261 [0.178] | 0.44 / 0.44 (0.58) | +0.00 [+0.00] |
| aligned NOSTAB (transport ceiling) | 1.000 [1.000] | 1.00 / 0.00 | +1.00 [+1.00] |
| **aligned STAB** (ceiling + stabilizer) | **1.000 [1.000]** | 1.00 / 0.00 | +1.00 [+1.00] |
| chained-FA NOSTAB (LEVER-2 lesion) | 0.422 [0.222] | 0.45 / 0.21 (0.37) | +0.23 [−0.14] |
| chained-FA STAB | 0.761 [0.178] | 0.74 / 0.04 (0.18) | +0.69 [+0.01] |
| chained-KP NOSTAB (LEVER-2 lesion) | 0.472 [0.133] | 0.47 / 0.13 (0.24) | +0.31 [+0.00] |
| **chained-KP STAB** (transport-free candidate) | **1.000 [1.000]** | 1.00 / 0.00 | +1.00 [+1.00] |
| FA-STAB no-sigma' | 0.785 [0.189] | 0.78 / 0.02 | +0.76 [+0.16] |
| FA-STAB permuted-reward (anti-cheat) | 0.606 [0.233] | 0.66 / 0.09 | +0.56 |
| **KP-STAB permuted-reward (anti-cheat)** | **1.000 [1.000]** | 1.00 / 0.00 | +1.00 |
| **UNTRAINED stabilized gate (zero learning)** | **1.000 [1.000]** | 1.00 / 0.00 | +1.00 |

teeth: marker 1.000 · HTM 0.000 · n-gram floor 0.272 · lesion-the-hold 0.046 · permuted-position (marker) 0.259 ·
identity crux gap +0.00 (fails). The `attributable_to` reports 86.7% of the KP-stab-vs-KP-nostab min improvement is owned
by the stabilizer.

The stabilizer eliminates the fire-everything basin for the transport-free chained-KP arm: **KP-stab = 1.000 [min 1.000]
on all 6 seeds** (fire 1.00/0.00), up from the KP-nostab lesion **0.472 [min 0.133]** with fire pos>0 up to 0.24 on the
collapse seeds. The lesion (stabilizer off) re-collapses → the stabilizer is **load-bearing**. The aligned transport
ceiling stays 1.000 WITH the stabilizer (the working case is not broken). Per-seed KP-nostab lesion at L=4 (the basin):
seed42 0.256, seed43 0.256, seed44 0.133, seed100 0.411, seed101 0.778, seed102 1.000 → KP-stab is 1.000 on every one.

## Why this is a NEGATIVE, not a GO — the role is STRUCTURAL, not credit-driven

Two anti-cheats fire together and settle it:

1. **Permuted-reward (KP-stab) = 1.000 [min 1.000].** With the verb target shuffled per sentence — the learning signal
   carries NO signal — the stabilized gate STILL reaches perfect role. The credit is not what induces role.
2. **An UNTRAINED stabilized gate (random weights, `train_chained` never called) = 1.000 [min 1.000], fire 1.00/0.00 on
   all 6 seeds** (`untrained_stabilized_gate_smokinggun.json`). Zero learning already solves the task.

Mechanistically this is inevitable and it is the finding: the feedback-inhibition **fire-once budget gives ONLY position
0 zero accumulated inhibition**, and with a small positive output bias (`b2_init=0.3`, `gain=4`) the gate fires at the
first token (p≈0.77 at pos0, then `z2_eff` drops by `out_lambda`≈4 → p≈0 thereafter). So the competition is inherently
an **ONSET gate that loads the first token**, which in this subject-first stream *is* the subject. The competition
supplies the positional answer; the transport-free credit rides along and is untestable here. Chained-FA is worse than
chained-KP precisely because its noisy fixed-random feedback sometimes *fights* the onset prior and drives the gate to
near-silence on the collapse seeds (seed100 FA-stab 0.178, fire 0.09/0.08) — a different failure (over-suppression), not
the basin.

So the LEVER-2 fire-everything basin **is eliminated by structural competition** — a real, load-bearing result — but the
question "does the transport-free CREDIT induce role" is now **unanswerable on this task**, because onset == the answer.
The runner's own verdict is `role_go=False` (the permuted-reward anti-cheat does not collapse).

## The precisely-isolated residual and the next lever

The residual is not depth, not sigma', not alignment, and not the fire-everything basin (that is now removed). It is that
**this task is solvable by a positional onset prior**, so any "fire once early" competition trivially clears it without
credit. To make the transport-free credit *testable*, the task must be changed so **onset ≠ the answer**: a
**variable-subject-position stream** where the subject appears at a position drawn at random each sentence and the gate
must fire the subject *regardless of ordinal*. There, the structural onset gate loads the wrong token (a distractor) and
only a learned, content/context-conditioned credit signal can solve it — and the permuted-reward + untrained controls
would then bite. That is the named next rung for the role-gate transport-free lane.

This also refines the LEVER-2 reading (Refinetti et al. 2021 "Align, then memorise", ICML, arXiv:2011.12428 — FA aligns
but the memorise phase need not fit): the memorise-phase basin the transport-free credit fell into is removable with a
structural competitive prior, but on a task whose answer coincides with that prior, removing the basin removes the test.

## Non-negotiables

Brain-based competition mechanism (feedback inhibition + divisive normalisation on the gate's own populations); the
credit rule stays transport-free/local; ONE spiking substrate (the REAL D3 `SpikingSlot`) at deployment; 6-seed
(42 43 44 100 101 102); honest-negative-is-a-deliverable; `SIM_BACKEND=numpy`; **NO sim/ edit** (empty `git diff main --
sim/`). The 2-layer net + chained credit + the competition are HOST math; their on-substrate spiking DA-gated /
lateral-inhibitory realisation is the named next rung.

## Sources

- Carandini & Heeger 2012, "Normalization as a canonical neural computation", Nat Rev Neurosci 13:51–62 — divisive/
  shunting gain control as a canonical cortical computation (the hidden divisive normalisation).
- O'Reilly & Frank 2006, "Making working memory work", Neural Comput 18:283–328 (PBWM) — BG-thalamocortical WM gating
  updates one stripe at a time with self-terminating disinhibition (the output "fire-once" gating budget).
- Refinetti, d'Ascoli, Ohana, Goldt 2021, "Align, then memorise: the dynamics of learning with feedback alignment",
  ICML, arXiv:2011.12428 — alignment is necessary but does not guarantee the memorise phase fits (the LEVER-2 residual).

## Reproduce

```
# 6-seed fan (each seed a process), then merge:
for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m \
  research.runners._var_bind_rolegate_competitive_stabilizer_derisk --seeds $s --distances 2 3 4 --n-test 90 \
  --out research/findings/raw/_rolegate_competitive_stabilizer/seed_$s.json & done ; wait
SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_competitive_stabilizer_derisk \
  --merge-from research/findings/raw/_rolegate_competitive_stabilizer/seed_*.json \
  --out research/findings/raw/_rolegate_competitive_stabilizer/competitive_stabilizer_6seed.json
```
