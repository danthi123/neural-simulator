---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/_gap4_ff/aggregate_gap4_ff.json
  - research/findings/raw/_gap4_ff/ff_xor_seed42.json
  - research/findings/raw/_gap4_ff/ff_xor_seed43.json
  - research/findings/raw/_gap4_ff/ff_xor_seed44.json
  - research/findings/raw/_gap4_ff/ff_xor_seed100.json
  - research/findings/raw/_gap4_ff/ff_xor_seed101.json
  - research/findings/raw/_gap4_ff/ff_xor_seed102.json
---

# gap#4 — a per-layer LOCAL contrastive (spiking Forward-Forward) rule ENTERS the deep-spiking learning regime at N=3 AND N=4, where every top-down credit path collapsed to chance — the wall's first crack (6-seed)

<!--derived-->
**One-line verdict.** On the SAME trainable LIF SNN + SAME depth-2 XOR→threshold task where the located wall
sits, a spiking **Forward-Forward** rule — each hidden layer trained by its OWN local contrastive-goodness
objective, **no top-down credit path at all** — LEAVES majority-class at N=2, N=3 AND N=4, at 6 seeds, exactly
where the chained multi-hop transport-free rule (fixed-FA AND KP-learned) did not. The 6-seed aggregate gate
returns GO at all three depths; per-seed it is 5/6 (one borderline seed). This is the wall's **first crack**: a
LOCAL contrastive objective gets a deep (N≥3) spiking net into the learning regime that a top-down credit path
— even a perfect Wᵀ oracle — could not open.

## The located wall this addresses (do NOT re-derive — read the two 2026-08-02 findings)
<!--derived-->

- **DOC1** (`2026-08-02-gap4-crux-wall-LOCATED-at-the-spiking-read-regime-…`): on the movable-plateau RESERVOIR
  substrate, even a perfect Wᵀ oracle (feedback-alignment ≈0.999) gives NO directed credit through the
  finite-spike σ′(v−θ) read (oracle == permuted; 5 controls agree).
- **DOC2** (`2026-08-02-gap4-depth-rescue-untestable-on-spikes-…`): on the TRAINABLE LIF SNN, the CHAINED
  multi-hop transport-free rule (fixed-FA AND KP-learned) does NOT leave majority-class at N≥3 — FA and KP give
  byte-identical held accuracy 0.45–0.54 == chance at N=3,4 on this depth-2 XOR task (the degenerate-dynamics
  fingerprint). Only N=2 entered the regime.

Every prior arm (FA, KP, the Wᵀ oracle, BDSP, dendritic) routes a top-down error hop-by-hop and re-gates it by
σ′(v−θ) at each hop. The wall kills exactly that. Forward-Forward removes the target of the wall: there is **no
top-down credit path to fail through**.

## The mechanism (the single genuinely-new class — LOCAL, no transport, no cross-layer credit)
<!--derived-->

Spiking **Forward-Forward** (Hinton 2022, arXiv:2212.13345; spike-native traces per Traces-Propagation
arXiv:2509.13053; Kohan Signal-Propagation contrastive idea). Each hidden layer trains from its OWN local
contrastive **goodness** on its OWN forward spike rate:

- goodness `g_l(x) = mean_j r_{l,j}²`, where `r_{l,j} = (1/T) Σ_t s_{l,j}[t]` is the per-neuron spike RATE.
- **paired** θ-free objective (SymBa/contrastive-FF): per example, `L_l = −log σ(g_l^pos − g_l^neg)`. Positive =
  input with the CORRECT overlaid label; negative = the SAME input with a WRONG overlaid label. Push the
  correct pairing's goodness above the wrong pairing's, per layer, independently.
- local update `dW_l = pre_l^T @ ((dL/dr_l) ⊙ ψ_l)`, `ψ_l = mean_t σ′(v_l[t]−θ_v)` (the surrogate-gradient
  eligibility). This uses ONLY layer l's own input current, voltage and rate — **no feedback matrix, no delivered
  top-down error**. Between layers the rate vector is RMS layer-normalized (orientation only passes), so a deeper
  layer cannot free-ride on the goodness magnitude below it.
- inference: for each candidate label, overlay it, accumulate goodness across hidden layers, argmax.

The σ′ still appears — but here it gates a LOCAL, strongly label-dependent goodness error (`g^pos` vs `g^neg`),
not a weak top-down error routed through misaligned feedback and re-gated hop-by-hop. That is the mechanistic
reason it opens the regime the top-down path could not.

## The decisive result — 6 seeds (42 43 44 100 101 102), XOR held-out (unseen bit patterns)
<!--derived-->

Majority-class fraction is 0.524. Arms on the SAME LIF forward + SAME task. `beatRes` = FF − optimal-ridge
frozen reservoir; `>perm` = FF − permuted-label FF; `ff_min` = worst seed. From `aggregate_gap4_ff.json`:

| N | FF held | FF min | +majority | topLayer+maj | resv-ridge | resv-FF | BPTT ceiling | permuted | beatRes | >perm | per-seed GO | 6-seed gate |
|---|---------|--------|-----------|--------------|-----------|---------|--------------|----------|---------|-------|-------------|-------------|
| 2 | 0.791 | 0.758 | +0.266 | +0.266 | 0.609 | 0.501 | 0.810 | 0.495 | +0.182 | +0.296 | 5/6 | GO |
| 3 | 0.780 | 0.730 | +0.256 | +0.152 | 0.623 | 0.508 | 0.804 | 0.496 | +0.157 | +0.284 | 5/6 | GO |
| 4 | 0.771 | 0.713 | +0.247 | +0.130 | 0.615 | 0.511 | 0.789 | 0.489 | +0.156 | +0.282 | 5/6 | GO |

The ENTER-THE-REGIME check (DOC2's own metric): **6/6 seeds leave majority-class at every depth** (`n_enters_regime`
= 6 at N=2,3,4). FF held-out is +0.25 to +0.27 above majority (and above chance 0.5 by ≥0.20), the worst seed
(`ff_min` 0.758/0.730/0.713) is clearly above majority, FF captures ~96–98% of the BPTT ceiling (0.810/0.804/0.789),
beats the OPTIMAL-ridge frozen reservoir by +0.156 to +0.182, the frozen-FF and permuted arms both sit at chance
(0.49–0.51), and the BPTT ceiling confirms the target is learnable. The 6-seed aggregate gate is GO at N=2,3,4.

## NOT weak-coupling — the deep layers are obligatory (per-layer held-out accuracy)
<!--derived-->

`per_layer_acc_mean` (index 0 = first hidden … last = top hidden; majority 0.524):

- N=3: [0.763, 0.789, 0.676] — all three above majority.
- N=4: [0.760, 0.782, 0.694, 0.654] — **all four above majority**; the deepest-from-input layers still read
  0.694 and 0.654, far from chance (`n_weak_coupling` = 0 at every depth).

The full-net accumulation is not riding a single shallow layer. There is a mild, honest dilution: the TOP layer's
own accuracy above majority falls with depth (`top_layer_above_majority` +0.266 → +0.152 → +0.130 for N=2→3→4) —
the very top layer of a taller stack is individually less discriminative — but it never approaches chance, and
the accumulated read stays +0.25 above majority. So depth is contributing, not decorative; the residual is a
gentle top-layer dilution, not the weak-coupling (trained-but-not-obligatory) pathology.

## Why it worked — the operating point was the wall (the "constant we replaced")

Two operating-point fixes were decisive, both instances of a homeostatic/competitive process we had proxied with
a constant:

1. An L2-unit inter-layer normalization silenced the deep layers (they received entries ~1/√H → near-zero drive
   → no goodness → no local gradient). **RMS layer-norm** (unit root-mean-square, entries O(1), matching the ±1
   feature scale) keeps every layer in its firing regime. This is the difference between "FF cannot learn at
   depth" and the table above.
2. A mis-set absolute goodness threshold θ gave no gradient at the operating point. The **paired θ-free** objective
   (contrast g^pos vs g^neg per example) removes θ entirely and is robust.

## Brain-based status (honest)

The spiking FORWARD is the substrate; the contrastive-goodness + surrogate-eligibility weight update is
host-computed bookkeeping — the SAME shortcut status as every credit rule in this arc (BPTT, FA, KP). It is,
however, **LOCAL** (no cross-layer credit, no weight transport) and **three-factor** (pre × post × surrogate, a
STDP-modulated eligibility trace), so it is a genuine candidate for a spiking/synaptic realization — unlike a
top-down transport rule, it has no non-local delivered error to biologize away. The BPTT arm is a labelled
CEILING only and is never shipped.

## Verdict and next mechanism

**GO (6-seed aggregate gate, N=2,3,4; per-seed 5/6).** A per-layer LOCAL contrastive objective gets a deep
(N=3,4) spiking net into the learning regime where the top-down credit path — a perfect Wᵀ oracle (DOC1) and the
chained fixed-FA/KP rule (DOC2) — collapsed to majority-class. This is the first mechanism to crack the located
finite-spike-read wall on the trainable substrate. It does NOT yet close gap#4 as a shipped capability: the rule
is still host-computed (to be biologized toward the one spiking substrate), the XOR task is depth-2-obligatory
(so N=3,4 test depth-robustness, not obligatory-depth-3 — the hier3/obligatory-depth-3 test on this rule is the
next step, reusing `make_task_hier3`), and the mild top-layer dilution at depth is worth understanding. But the
capability the wall denied — directed credit into a deep spiking hidden stack with a local, transport-free rule —
is demonstrated.

## Reproduce

```
# smoke (one seed, N=3):
SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_forwardforward_local_derisk \
    --task-xor --seed 42 --n-list 3 --epochs 300 --lr 1.0 --label-gain 3.0 \
    --bptt-hidden 128 --bptt-epochs 250 --bptt-lr 0.2 \
    --out research/findings/raw/_gap4_ff/ff_confirm_seed42_N3.json
# 6-seed (one process per seed, sweeping N=2,3,4; fan across cores):
for S in 42 43 44 100 101 102; do SIM_BACKEND=numpy .venv/bin/python -m \
    research.runners._gap4_forwardforward_local_derisk --task-xor --seed $S --n-list 2 3 4 \
    --epochs 350 --lr 1.0 --label-gain 3.0 --bptt-hidden 128 --bptt-epochs 200 --bptt-lr 0.2 \
    --out research/findings/raw/_gap4_ff/ff_xor_seed${S}.json & done; wait
SIM_BACKEND=numpy .venv/bin/python -m research.runners.aggregate_gap4_ff_seeds \
    "research/findings/raw/_gap4_ff/ff_xor_seed*.json"
```
