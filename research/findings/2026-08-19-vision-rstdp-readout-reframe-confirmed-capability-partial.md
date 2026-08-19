---
type: finding
status: mixed
date: 2026-08-19
lane: perception
mechanism: reward-modulated-STDP-discriminative-readout
runner: research/runners/_vision_rstdp_readout_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/vision_rstdp_readout_ns64_6seed.json
  - research/findings/raw/lanes/perception/vision_rstdp_readout_ns32_6seed.json
  - research/findings/raw/lanes/perception/vision_rstdp_readout_ns16_6seed.json
---

# REWARD-MODULATED STDP readout on the fully-spiking HMAX vision path (board #75): the #72 REFRAME's CORE is CONFIRMED -- on spikes, learning becomes LOAD-BEARING (learned 0.38 vs random 0.24 held, +0.14, 5/6) where the rate case had random==learned -- but capability recovery is PARTIAL (0.38 clears the 0.34 NO-GO floor on 5/6 raw, does not reach config-B 0.56) and the reframe's SPARSITY sub-clause is REFUTED: the load-bearing learned code is DENSE/distributed, not sparse, and forcing sparsity WEAKENS it

**One-line verdict.** The spiking HMAX de-risk (board #72,
[`2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited`](2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited.md))
showed the LIF S1->C1 front end PRESERVES position-invariant configural recognition on spikes (config B,
count held **0.5625** <!--derived-->, 6/6 arch load-bearing) but FULLY spike-coding the S2->C2 readout is a NO-GO
(config C, held **0.34**, position leaks 0.97, 0/6), and it REFRAMED the cause: on RATE a random S2 ==
a learned one ("learning not load-bearing"), but that is a rate artifact -- on spikes the distributed
random code is quantization-fragile, so a DISCRIMINATIVE readout learned with reward-modulated STDP
(R-STDP) was PREDICTED to become load-bearing. This runner builds that named, untried mechanism and
tests the prediction, 6 seeds. **Result is MIXED, and the split is the finding:**

- **The reframe's CORE prediction is CONFIRMED, and localised to the DISCRIMINATIVE readout.** Read
  through the class-assigned spiking-WTA (the discriminative readout R-STDP builds), the learned bank
  BEATS an identical RANDOM bank on spikes: held **0.3802 vs 0.2413** (Δ**+0.1389** pooled), **learning
  load-bearing 5/6** (per-seed Δ +0.26/+0.03/+0.18/+0.14/+0.11/+0.11). Held positions {1,3,5,7} are NEVER
  trained; random sits at chance (0.24≈0.25), learning moves it to 0.38 -- genuine position-generalizing
  discriminative learning. **Control decode (anti-shopping):** an UNSUPERVISED cosine-centroid on the
  SAME held spike code shows learned≈random (**0.3906 vs 0.3646, Δ+0.0261**), i.e. NO learning effect --
  exactly like the rate case. So the effect is not a property of the spike code in general; it is that
  R-STDP builds a class-DISCRIMINATIVE readout, and it is through THAT readout that learning becomes
  load-bearing on spikes. That is precisely the reframe's claim.
- **Capability recovery is PARTIAL, and modest in absolute terms.** The robust positive is the
  within-architecture learned-vs-random dissociation above, NOT a large absolute jump: fully-spiking
  held 0.38 (spiking-WTA) / 0.39 (centroid) only modestly exceeds the config-C NO-GO 0.34 (spiking-WTA
  raw beat 5/6; seed 43 = 0.31 is the miss; not by the strict +0.10 margin -- only seed 42 reaches 0.45)
  and does NOT approach config-B (0.5625) or the rate ceiling (0.5972). Note the RANDOM centroid (0.3646) <!--derived-->
  ≈ config-C's random (0.36), so on the unsupervised decode nothing much moved. The binding residual is
  position: the learned discriminative spiking code stays position-entangled (position still decodes
  0.58-0.88 off the C2 code; object 0.23-0.54), so train (0.50-0.62) >> held (0.31-0.45).
- **The reframe's SPARSITY sub-clause is REFUTED (a first-class negative).** #72 predicted the fix would
  be a SPARSE SELECTIVE code (few strongly-firing units). It is not. The load-bearing learned code is
  DENSE/distributed -- at n_s2=64 all 16 units/class fire and the top-5 carry only **0.35** of the
  class-spike mass vs random's **0.40** (learned is if anything LESS concentrated). Forcing sparsity
  monotonically WEAKENS the reframe: dropping the bank to 32 (8/class) then 16 (4/class) takes
  learning-load-bearing from 5/6 -> 4/6 -> 2/6 and Δ(learned-random) from +0.1389 -> +0.1337 -> +0.0781; a
  C2-readout kWTA (keep top-k) HURTS (random benefits from it, killing Δ). **The quantity that makes
  learning load-bearing on spikes is per-unit class-DISCRIMINATIVENESS of a distributed population, not
  spatial sparsity.**

`EXTERNAL-SEARCH-RAN:` R-STDP grounding -- Frémaux & Gerstner 2016 (three-factor plasticity framework);
Izhikevich 2007 (DA-modulated STDP); Mozafari, Ganjtabesh, Nowzari-Dalini, Thorpe & Masquelier 2018,
IEEE TNNLS 29(12):6178-6190 ("First-spike-based visual categorization using reward-modulated STDP"),
the DIRECT precedent: R-STDP on a Thorpe/Masquelier conv-SNN, decision neuron correct -> STDP / wrong ->
anti-STDP. Our own [`2026-06-02-step2a`](2026-06-02-step2a-spiking-visual-word-recognition-characterization.md)
scoped this exact remaining piece ("V1 latency + per-band kWTA + R-STDP / learned readout"; vanilla
UNSUPERVISED STDP insufficient, a supervised/reward readout needed) -- which is why config C's
unsupervised trace/random S2 failed and a reward-supervised readout was the right lever.

## Mechanism (built here; no `sim/` edit, reuses the #72 spiking front end BY IMPORT)

Reused config-B LIF S1->C1 (spiking, PRESERVES the capability) -> convolutional S2 cosine drive -> S2
lateral inhibition across the template bank per location (z, winner-relative contrast so the
near-threshold LIF is not saturated by the ~0.8 cosine common-mode) -> **LIF S2 coincidence spikes** at
every location -> **C2 spiking WTA global MAX-pool** over locations -> per-class spike-sum -> **spiking
WTA over the class-assigned populations = the prediction**. The S2 template bank (round-robin
class-assigned, n_s2/n_classes per class) is LEARNED by online three-factor R-STDP:

- **pre** = the C1 patch at each template's winning location; **post** = the C2 spike; **third factor** =
  a global correct/incorrect dopamine sign. On each presentation the true class's fired templates are
  POTENTIATED toward their winning-location patch (reward / corrective teacher); on error the
  wrongly-predicted class's fired templates are DEPRESSED (anti-STDP). Weights are non-negative
  (excitatory) and L2-renormalised each update (homeostatic bound -- no unbounded growth).
- **RANDOM arm** = identical architecture + identical spiking forward + identical decodes, W UNTRAINED
  (the same init the learned arm starts from). This IS config-C's random arm and the like-for-like
  control for the reframe: LEARNED must beat RANDOM on spikes.

## Result -- 6 seeds (42/43/44/100/101/102), chance 0.25, primary code = count

| bank n_s2 (per class) | LEARNED spkwta held | RANDOM spkwta held | Δ learned-random | learn load-bearing | beats NO-GO 0.34 (raw) | win-class active units | learned/random top-5 mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| **64 (16) -- PRIMARY** | **0.3802** | 0.2413 | **+0.1389** | **5/6** | **5/6** | 16.0 (dense) | 0.353 / 0.4044 |
| 32 (8) | 0.3629 | 0.2292 | +0.1337 | 4/6 | 4/6 | 8.0 | 0.6661 / 0.7243 |
| 16 (4) | 0.3229 | 0.2448 | +0.0781 | 2/6 | 2/6 | 3.9 (sparse) | 1.00 / 1.00 |

<!--derived-->
Reference floors/ceilings (same task, same front end): config-C fully-spiking NO-GO **0.34**; config-B
(LIF S1->C1 + rate S2/C2 MAX) **0.5625**; rate ceiling **0.5972**; V1-direct held 0.42; flat-pool held
0.30. Nulls (primary, per seed): held pixel-scramble decode ~chance; label-shuffle ~chance. Determinism:
config-64 re-run byte-identical per-seed (seeds 42,43 verified). No `sim/` file modified.

## What survives on spikes, and what does not -- the decomposition IS the finding

- **The reframe is real and directional.** #72's rate finding said learning was inert because a random
  projection already separates the (linearly separable) configural classes and the cosine-centroid
  decode divides out the common mode. On spikes that random code is quantization-fragile (config C,
  0.34), and a reward-supervised discriminative readout recovers a real, seed-robust margin over random
  (Δ+0.1389, 5/6). This converts #72's PREDICTION into a DEMONSTRATION.
- **But the mechanism is not the one #72 named.** #72 said "sparse selective". The load-bearing learned
  code is DENSE. The three bank sizes are a clean gradient: the more we sparsify (64->32->16 units, or a
  C2 kWTA read), the WEAKER the learned-vs-random effect. So the discriminative signal that survives
  spike quantization lives in the JOINT firing of a distributed class-population, not in a few sharply
  tuned units. This is the honest correction to the reframe: on THIS substrate, DISCRIMINATIVENESS (not
  sparsity) is what learning buys that random cannot.
- **Position is the binding residual.** The fully-spiking learned readout fits train (0.50-0.62) far
  better than held (0.31-0.45); position still decodes 0.58-0.88 off the C2 code. The innate global MAX
  pool is exactly position-invariant on RATE (config B pooled position out) but the spike-coded pool +
  the learned templates (patches from train-position C1 codes) leave the held-position match magnitude-
  and selection-entangled with position. Longer LIF integration (T2 up to 200) does NOT help -- this is
  a generalization/invariance limit, not a quantization one.

## The NAMED next mechanism (this is a call to attack, not a wall)

The residual is a POSITION-INVARIANT discriminative spiking readout. Two grounded, untried levers:
1. **Translation-augmented R-STDP** -- present each configural object as a continuity stream that
   translates across (and between) the C1 grid during training, so the reward-modulated eligibility
   averages the winning-location patch over shifts (Földiák-style trace + R-STDP), tuning templates to
   be shift-tolerant rather than train-position-specific.
2. **Discriminative spiking DECISION layer on the position-pooled representation** -- keep config-B's
   innate global MAX pool as a graded complex-cell op (position-invariant, the same concession config B
   made) and spike-code + R-STDP-learn only the FINAL class layer on that invariant C2. This trades a
   little of the "fully spiking S2->C2" purity for the invariance config B already proved, and tests
   whether a discriminative SPIKING decision on an invariant code closes to 0.56.

## Honest residuals / scope

- **MIXED, not a GO.** capability_go 0/6 (the position-pooled-out gate + strict +0.10 margin both fail).
  The positive that IS robust is the reframe (learning load-bearing on spikes, 5/6); the capability
  (fully-spiking position-invariant configural recognition) is PARTIALLY recovered (0.34 -> 0.38), not
  closed to 0.56.
- **Readout concession (flagged).** The final class selection is an argmax over class-population spike
  COUNTS -- a spiking WTA / lateral-inhibition read, the SAME host-readout concession as config C's
  cosine-centroid decode. It is convertible to a first-spike FS-WTA per
  [`2026-07-13-PAST-RESERVOIR-ONBRIDGE-spiking-readout`](2026-07-13-PAST-RESERVOIR-ONBRIDGE-spiking-readout-fully-spiking-end-to-end-6seed-GO.md);
  the plasticity, S1/S2 somata (LIF: leak, threshold, reset, absolute refractory, membrane noise) and
  the C-layer MAX (feedforward-inhibition WTA) are all spiking/synaptic.
- **Innate scaffolds (flagged).** Retinotopic weight-sharing + pooling windows remain innate
  developmental scaffolds -- the same defended concession as config B/C.
- **Production wiring: BLOCKED, unchanged from #72.** The live conversational path (`POST /api/brain-chat`
  -> `ChatBrain`) ingests only text; there is no live object-anywhere vision consumer, so the honest
  scope is the spiking CAPABILITY above, not a production flip.

## Reproduce

```bash
# PRIMARY (n_s2=64, headline op-point = runner defaults):
SIM_BACKEND=numpy OMP_NUM_THREADS=2 .venv/bin/python -u -m research.runners._vision_rstdp_readout_derisk \
  --seeds 42 43 44 100 101 102 --code count --n-s2 64 \
  --out research/findings/raw/lanes/perception/vision_rstdp_readout_ns64_6seed.json
# SPARSITY SWEEP (the reframe weakens as the bank shrinks):
#   ... --n-s2 32 --out research/findings/raw/lanes/perception/vision_rstdp_readout_ns32_6seed.json
#   ... --n-s2 16 --out research/findings/raw/lanes/perception/vision_rstdp_readout_ns16_6seed.json
```

## Sources

- Frémaux, N. & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and theory of
  three-factor learning rules. *Front. Neural Circuits* 9:85.
- Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine
  signaling. *Cereb. Cortex* 17(10):2443-2452.
- Mozafari, M., Ganjtabesh, M., Nowzari-Dalini, A., Thorpe, S. J. & Masquelier, T. (2018). First-spike-
  based visual categorization using reward-modulated STDP. *IEEE TNNLS* 29(12):6178-6190.
- Riesenhuber, M. & Poggio, T. (1999). Hierarchical models of object recognition in cortex. *Nat.
  Neurosci.* 2:1019-1025. (HMAX; the complex-cell MAX pooling op.)
- Prior on this substrate: board #72
  [`2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited`](2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited.md);
  [`2026-06-02-step2a-spiking-visual-word-recognition`](2026-06-02-step2a-spiking-visual-word-recognition-characterization.md).
