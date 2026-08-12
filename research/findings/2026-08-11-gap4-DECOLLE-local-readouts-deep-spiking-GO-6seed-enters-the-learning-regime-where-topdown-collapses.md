---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes
artifacts:
  - research/runners/_gap4_decolle_local_readouts_derisk.py
  - research/findings/raw/_gap4_decolle/decolle_xor_6seed_summary.json
  - research/findings/raw/_gap4_decolle/decolle_N3_s42.json
  - research/findings/raw/_gap4_decolle/decolle_N4_s42.json
  - research/findings/raw/_gap4_decolle/hier3smoke_N3_s42.json
---

# gap#4 crux — the FIRST CRACK in the spiking-credit wall: DECOLLE per-layer LOCAL readouts get a DEEP (N=3 AND N=4) spiking net INTO the learning regime on REAL spikes (6-seed GO), where the top-down CHAINED FA/KP rule collapses to majority-class — the named surpass, on the TRAINABLE surrogate-gradient substrate

<!--derived-->
**One-line verdict.** The wall located this month was UPSTREAM of the feedback question: on the trainable LIF SNN the
transport-free CHAINED multi-hop FA (and its KP repair) does NOT get a deep (N>=3) spiking net into the learning regime
at all — both collapse to BYTE-IDENTICAL majority-class output at N=3 and N=4
(`2026-08-02-gap4-depth-rescue-untestable-on-spikes-...`). The crux's own exhaustively-earned NAMED SURPASS was "a
genuinely TRAINABLE spiking substrate ... as the field's working deep-spiking trainers do (e-prop, DECOLLE, SuperSpike)"
(`2026-08-02-gap4-crux-wall-LOCATED-...` Update 4). This builds it: **DECOLLE** (Kaiser, Mostafa, Neftci 2020, *Synaptic
Plasticity Dynamics for Deep Continuous Local Learning*, Front. Neurosci. 14:424) — each layer has its OWN fixed-random
local readout and is trained by its OWN local classification error, "errors do NOT propagate through neurons and across
layers" (paper, verbatim). On the SAME LIF SNN + SAME XOR task + SAME 6-seed harness as the wall findings, **DECOLLE
ENTERS the learning regime at N=3 AND N=4 (6/6 GO), leaving majority-class on real spikes, where the top-down chained
FA/KP collapse to majority-class (6/6)**. No `sim/` edit (the credit rule is a runner-side function; forward + surrogate
reused-by-import from `sim/bptt_snn_gpu`).

## Result — 6 seeds (42/43/44/100/101/102), XOR->threshold, matched forward init across every arm

<!--derived-->
Every arm shares the IDENTICAL LIF-SNN forward init (rng `seed+1`, `w_scales [2.5,1.0,...]`), so the comparison is on
the CREDIT RULE alone. `hidden 32, T 24, epochs 200, lr 0.05, train-subsample 2000`; BPTT ceiling tuned
(`hidden 128, epochs 400`). Artifacts: `research/findings/raw/_gap4_decolle/decolle_xor_6seed_summary.json` +
per-seed `decolle_N{3,4}_s{seed}.json` (numpy/CPU).

<!--derived-->
| arm (held-out acc, mean over 6 seeds) | N=3 | N=4 | role |
|---|---|---|---|
| **DECOLLE** (per-layer local readouts) | **0.926** (min 0.877) | **0.941** (min 0.908) | the candidate |
| chained fixed-FA (transport-free) | 0.500 (modal 1.00) | 0.500 (modal 1.00) | WALL baseline — collapsed 6/6 |
| chained KP-learned FA (transport-free) | 0.483 (modal 1.00) | 0.500 (modal 1.00) | WALL baseline — collapsed 6/6 |
| DFA e-prop (output error, direct feedback) | 0.867 | 0.840 | reference (not DECOLLE — see below) |
| surrogate-BPTT (labelled ceiling, scaffold) | 0.792 | 0.779 | confirms a target exists 6/6 |
| frozen reservoir (trained readout only) | 0.556 | 0.522 | R3 floor |
| frozen reservoir OPTIMAL ridge (matched) | 0.623 | 0.615 | stronger reservoir floor |
| DECOLLE on SHUFFLED labels (anti-cheat) | 0.481 | 0.503 | collapses 6/6 |
| chance (modal-class frequency) | 0.524 | 0.524 | majority-class |

<!--derived-->
**GO 6/6 at BOTH depths.** The runner's per-seed gate (`decolle above chance+0.20 AND leaves majority-class AND beats
frozen reservoir by >=0.10 AND chained-FA AND KP collapse to majority-class AND shuffled collapses AND BPTT confirms a
target`) passes 6/6 at N=3 and 6/6 at N=4. DECOLLE beats the frozen reservoir by **+0.370** (N=3) / **+0.418** (N=4),
min per-seed +0.337/+0.379. Anti-cheats clean 6/6: no readout is byte-equal to any forward weight or its transpose
(`no_readout_transport_all True`); the substrate is genuinely seeded by `seed` (build-twice-identical,
`seeded_substrate_all True` — the `actual_seed_used` trap does not apply, seeding is pure-numpy `rng(seed+1)`).
Attribution (`tools.lab.attributable_to`, N=3): **48%** of DECOLLE's held-out accuracy is NOT present in the
shuffled-label control and **40%** is not present in the frozen reservoir — the correct-label local error and the
hidden-layer training own the lift (the rest is the ~chance floor both controls also reach); this is not a
clamp/baseline-dominated effect.

## The ENTER-THE-REGIME signature (the wall's own metric) — DECOLLE reshapes the deep layers, the reservoir cannot

<!--derived-->
The wall's fingerprint is majority-class collapse (FA/KP give byte-identical single-class output). DECOLLE's
`pred_modal_frac` is ~0.52-0.60 (balanced predictions), FA/KP's is 1.00 (a single class). Per-hidden-layer LINEAR
(ridge, 5-fold-CV) class-decodability of the TRAINED layers' summed spikes — how class-SELECTIVE each layer became,
reported with the frozen-reservoir baseline (no hidden training) and the collapsed-FA comparison:

<!--derived-->
| per-hidden-layer ridge class-decodability (mean 6 seeds) | N=3 | N=4 |
|---|---|---|
| DECOLLE | [0.948, 0.942, 0.941] | [0.953, 0.946, 0.946, 0.949] |
| frozen reservoir (hidden never trained) | [0.546, 0.588, 0.569] | [0.546, 0.588, 0.569, 0.537] |
| chained-FA (collapsed) | [0.404, 0.553, 0.543] | [0.412, 0.514, 0.479, 0.476] |

<!--derived-->
DECOLLE drives EVERY hidden layer — including the DEEPEST (furthest from the output) — to ~0.95 class-decodability,
where the fixed reservoir sits at ~0.55 and the collapsed top-down FA at ~0.45-0.55. This is the decisive read: DECOLLE
RESHAPES the deep hidden representations (its win is not a readout free-riding on a fixed reservoir), and it does so
WITHOUT any descending credit — each layer trained only by its own fixed-random local readout error. The DECOLLE-native
per-hidden local-readout accuracies (the signal it actually trains toward) are ~0.91-0.95, confirming each layer
independently learns to classify. Real spikes confirmed: per-layer mean spike rates are healthy (non-silent,
non-saturated) at every layer; the read is argmax over summed OUTPUT spikes — the SAME read every arm uses.

## The mechanism (DECOLLE Eq. 8, ported to the LIF SNN; brain-based LOCAL rule)

<!--derived-->
Per hidden layer `li`, per timestep `t`, with the layer's OWN fixed-random readout `B_l (k, n_l)` (a SEPARATE seed
stream, transport-free): local logits `y_l = spikes_li . B_l^T`; local error `d_l = (softmax(y_l) - onehot(target))/T`;
error at the layer `e_l = (d_l . B_l) * sigma'(v_li - theta)` (backproject through the FIXED readout x atan-surrogate);
weight grad `dW_li += eps_li^T . e_l`, `eps_li = alpha_leak*eps_li + pre` (the e-prop forward eligibility). NO chain —
`e_l` depends ONLY on layer `li`'s own spikes, its own `B_l`, and the target, never on any layer-above error. The
OUTPUT LIF layer is trained by the true output error IDENTICALLY to the frozen/chained/bptt arms, so (decolle - frozen)
isolates EXACTLY the DECOLLE hidden contribution. This is a brain-based LOCAL rule (fixed-random local projections +
local loss, no global backprop, no weight transport — the cortical local-error / apical-target family); the BPTT arm is
a labelled ceiling only (a scaffold, not brain-based).

## Why this is the first real crack, and the reconciliation with the crux TERMINUS (honest, exact)

<!--derived-->
The crux's spiking-side TERMINUS concluded "top-down credit has no purchase on spikes; even DECOLLE local losses give
`directed = 0`; the deep layer is reservoir-redundant" (`2026-08-02-gap4-crux-wall-LOCATED-...` Updates 2-4). That
conclusion was measured on the FROZEN movable-plateau coincidence-plateau RESERVOIR substrate — a NON-trainable forward
where a local readout was bolted onto fixed hidden layers. Update 4 ITSELF named the resolution: the surpass is "a
genuinely TRAINABLE spiking substrate". This finding runs DECOLLE on exactly that substrate — the surrogate-gradient LIF
SNN whose forward weights ARE plastic — and DECOLLE gets full purchase: it reshapes the deep layers (selectivity 0.95 vs
the reservoir's 0.55) and enters the learning regime 6/6. So the two results do not conflict; they bound the claim
precisely: **on a FROZEN reservoir the deep layer is reservoir-redundant and NO credit signal (top-down or local) helps;
on a TRAINABLE substrate a LAYER-LOCAL credit signal (DECOLLE) DOES get purchase, while the TOP-DOWN CHAINED signal
(FA/KP) still does not enter the regime at N>=3.** The wall was never "spikes can't carry credit" — it was "the
top-down/chained credit path does not enter the deep-spiking learning regime"; DECOLLE bypasses that path entirely.

## Arm placement — DECOLLE vs DFA vs BPTT (what enters, what does not)

<!--derived-->
Three transport-free rules were run on the identical substrate: (1) **chained FA/KP** (error descended hop-by-hop) —
does NOT enter the regime at N>=3 (the located wall, reproduced 6/6); (2) **DFA e-prop** (the OUTPUT error projected
DIRECTLY to each hidden by a fixed-random `B_direct`) — DOES enter (0.84-0.87), answering the depth-rescue Update's open
question "does DFA scale to N>=3?" affirmatively on the LIF net; (3) **DECOLLE** (each layer's OWN local readout, no
output error at the hidden layers at all) — enters most strongly (0.93-0.94). The ordering is mechanistically coherent:
the CHAIN compounds FA misalignment per hop (Nokland 2016) and collapses; both DFA and DECOLLE avoid the chain (each
hidden gets a direct target), and DECOLLE's per-layer local objective is the easiest to optimize — it even exceeds the
surrogate-BPTT ceiling (0.93 vs 0.79), consistent with deep through-time surrogate credit itself degrading with depth
while a local per-layer objective does not. BPTT's role here is ONLY to confirm a target exists (>chance+0.15, 6/6); it
is not the max.

## Honest scope + the residual (what this is NOT)

<!--derived-->
**XOR is a depth-2 task**, so at N=3/N=4 the extra layers are REDUNDANT. This result therefore proves the wall as it was
LOCATED — "the transport-free top-down/chained rule does not get a DEEP (N>=3) spiking net into the learning regime" —
and that DECOLLE DOES, at N=3 AND N=4, on real spikes. It does NOT by itself prove credit flowing through 3 GENUINELY
OBLIGATORY nonlinear stages. **The obligatory-depth-3 corroboration (2-seed smoke, `hier3smoke_N3_s{42,43}.json`):** on
the obligatory-depth-3 `hier3` task (k=9, chance 0.167) DECOLLE STILL enters the regime — held-out 0.407/0.278 (above
chance +0.24/+0.11), pred-modal 0.33/0.52, per-layer selectivity 0.4-0.48 — where FA/KP are pinned at chance/majority
(0.167, modal 1.00). But `hier3` does not reach a full GO there because BPTT ITSELF cannot fit it (0.11-0.13, below
chance) — the INDEPENDENTLY-DOCUMENTED `hier3` task-construction wall (0/17 configs separate depth-2 from depth-3
generalization; `2026-08-02-gap4-depth-rescue-untestable-...`). So on the obligatory-depth task DECOLLE's absolute
ceiling is capped by the task's own fit wall, NOT by DECOLLE — the enter-the-regime crack holds on BOTH tasks. The clean
next mechanism is a genuinely-depth-3 task that surrogate-BPTT CAN fit (a fan-in-2 compositional hierarchy without the
parity/fan-in traps), then re-run the DECOLLE-vs-FA/KP comparison to test obligatory-depth credit; and the on-bridge
Izhikevich port of DECOLLE (the FA-convergence root cause differs on point-neuron Izhikevich —
`2026-08-02-gap4-FA-convergence-is-the-onbridge-credit-root-cause-...`).

## Reproduce

<!--derived-->
```
# 6-seed, per depth (fan across processes; numpy/CPU):
for N in 3 4; do for S in 42 43 44 100 101 102; do SIM_BACKEND=numpy .venv/bin/python -m \
  research.runners._gap4_decolle_local_readouts_derisk --task-xor --seeds $S --n-hidden-layers $N \
  --hidden 32 --epochs 200 --lr 0.05 --train-subsample 2000 --bptt-hidden 128 --bptt-epochs 400 \
  --out research/findings/raw/_gap4_decolle/decolle_N${N}_s${S}.json & done; wait; done
```
