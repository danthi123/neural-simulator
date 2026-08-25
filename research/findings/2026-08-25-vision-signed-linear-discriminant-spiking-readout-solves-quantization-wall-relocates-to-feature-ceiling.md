---
type: finding
status: mixed
date: 2026-08-25
lane: perception
board: 75
mechanism: signed-linear-discriminant-spiking-readout
runner: research/runners/_vision_lindiscrim_readout_derisk.py
supersedes_method: R-STDP sparse readout (mapped dead-end, board #75)
artifacts:
  - research/findings/raw/lanes/perception/vlin_readout_6seed.json
---

# A SIGNED linear-discriminant spiking readout (excitatory + feedforward-inhibition, + temporal evidence integration) SOLVES the fully-spiking S2->C2 quantization wall (board #75) — learning is now load-bearing 6/6 and the spike-port quantization gap collapses to ~0.02 — but the capability wall RELOCATES from the spike code to the z-normalized C2 feature ceiling (mixed: clears the config-C NO-GO 0.34 6/6 and beats the random readout 6/6, but does not beat the V1-direct floor)

**One-line verdict.** After REWARD-MODULATED STDP NO-GO'd across the entire 2D operating-point sweep (board
#75; the sparse-readout method is a MAPPED dead-end), a DIFFERENT fully-spiking readout — a supervised
SIGNED linear discriminant, spike-ported as excitatory + feedforward-inhibitory LIF class populations, with
temporal evidence integration — is a MIXED result that moves the wall. It **clears the config-C NO-GO
floor (0.34) on 6/6 seeds** (held 0.4375 mean), **makes readout LEARNING load-bearing 6/6** (learned 0.4375
vs a random signed readout 0.2552 ≈ chance; dLEARN +0.1823), **beats the config-C centroid on identical
features** (learned 0.4375 vs centroid 0.4010), and — the decisive diagnostic — **collapses the spike-port
quantization gap to +0.0243** (rate-linear 0.4653 → spike-linear 0.4410; the decision-port cost is +0.0035),
versus the config-C
centroid's ~0.19 rate→spike drop. The quantization problem that sank config C is SOLVED. BUT the fully-
spiking readout does **not** reach a usable capability: it does not beat the V1-direct floor (0.4375 vs
0.4184; capability_go 0/6), because the bottleneck has RELOCATED — the RATE linear-separability ceiling of
the z-normalized C2 feature code is only ~0.47, ~0.09 below config-B's raw-rate MAX+centroid ceiling
(0.56). The residual is a FEATURE-CODE limit, not a spike-quantization or readout-class limit.

## Why a new mechanism (R-STDP is banked, not re-tuned)

The spiking-HMAX de-risk (#72,
[`2026-08-19-vision-spiking-hierarchy-frontend-holds-...`](2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited.md))
established: the LIF-spiking S1->C1 FRONT END preserves position-invariant configural recognition on spikes
(config B: held 0.5625 <!--derived--> with a RATE S2/C2 MAX readout, arch load-bearing 6/6), but FULLY spike-coding the
S2->C2 configural readout is a NO-GO (config C: 0.34), because the rate C2 discrimination is a fine
DISTRIBUTED cosine modulation (across-template std ~0.042 <!--derived--> on a common-mode ~0.80) below the per-unit spike
quantization floor. The #72 reframe named REWARD-MODULATED STDP (a sparse selective readout); board #75
built it
([`2026-08-19-vision-rstdp-readout-reframe-confirmed-capability-partial.md`](2026-08-19-vision-rstdp-readout-reframe-confirmed-capability-partial.md))
and it NO-GO'd across the ENTIRE 2D op-point sweep (24-256 S2 x 30-150 epochs; artifacts `vrstdp_*`),
capping ~0.40 held.

**The measured flaw both prior readouts share** (the "companion process replaced with a constant"): neither
can subtract the common mode at the read. Config C = a nearest-cosine-centroid (unsupervised prototype);
R-STDP = a FIXED non-negative round-robin class block-sum (`cscore = r @ class_mat.T`, `class_mat` 0/1),
with only EXCITATORY S2 templates learned. A fine modulation riding on a large common mode is a SIGNED
linear-discriminant problem — reading it requires NEGATIVE weights, which the brain supplies as EXCITATORY
afferents + FEEDFORWARD INHIBITION (an interneuron carrying the negatively-weighted / common-mode pool;
Dale's law). That companion process was absent from both prior readouts. This runner supplies it.

## The mechanism (built here; no `sim/` edit; front end REUSED BY IMPORT)

Keep config B's FIXED spiking front end (LIF S1->C1, a FIXED random S2 template bank — template LEARNING is
not the lever, #72). Read the SAME C2 spike code config C reads (per-template MAX over locations =
position-invariant), AVERAGED over G=2 independent LIF glimpses (temporal evidence integration; the
quantization floor is per-presentation, and the animal accrues evidence over multiple fixations — speed is
explicitly secondary here). Then, instead of a centroid / block-sum, LEARN a SIGNED readout by RIDGE-
regularized least squares (the standard trained LINEAR readout of a spiking reservoir; Maass et al. 2002).
Ridge lambda is the homeostatic regularizer (biologically = synaptic scaling): large lambda shrinks toward
the class-mean-difference (centroid) direction that generalizes across held positions, small lambda = full-
covariance-whitened LDA that overfits — lambda=0.5 was chosen on a 42/43/100 exploration, leaving
44/101/102 out-of-sample. PORT TO SPIKES honestly: standardize the C2 code by its train mean/std (the
common-mode-rejecting FF interneuron + divisive normalization), decompose the effective weights
w = w+ - w-, and drive a POPULATION of LIF class somata per class with EXCITATORY current (w+ . r) MINUS
FEEDFORWARD-INHIBITORY current (w- . r) plus a tonic bias; a spiking WTA (argmax over class-population
spike counts) is the FULLY-SPIKING prediction. The RANDOM arm is the identical spike port with V untrained
(random signed) — the like-for-like control.

## Result — 6 seeds (42/43/44/100/101/102), chance 0.25, count code

| quantity (held-out positions) | mean | per-seed / note |
|---|---:|---|
| LEARNED signed-linear SPIKING-WTA held | **0.4375** | 0.49/0.38/0.46/0.47/0.43/0.41 |
| RANDOM signed spiking readout (control) | 0.2552 | ≈ chance 0.25 |
| dLEARN = learned − random (load-bearing) | **+0.1823** | 6/6 >= +0.10 |
| config-C centroid on the SAME spike code (NO-GO repro) | 0.4010 | learned 0.4375 beats it <!--derived--> |
| RATE signed-linear ceiling (z-normed C2 features) | 0.4653 | the new bottleneck |
| spike-port quantization gap (rate-lin − spk-lin) | **+0.0243** | vs config-C centroid ~0.19 |
| spike-port cost of the DECISION (lin-score − spk-WTA) | +0.0035 | the E/I population port is ~free |
| V1-direct floor / flat-pool floor | 0.4184 / 0.3003 | learned − V1 = +0.0191 (NOT beaten) |
| object decode / position decode (off class-pop spikes) | 0.4271 / 0.3437 | position POOLED (config C leaked 0.97 <!--derived-->) |
| pixel-scramble null / label-shuffle null | 0.2465 / 0.2917 | ≈ chance (retrained on shuffled labels) |
| LEARNED train accuracy | 0.9722 | overfits train; ridge sets the held generalization |

Verdict tallies: **learning_load_bearing 6/6**; **beats config-C NO-GO floor (raw > 0.34) 6/6**; beats the
strict +margin floor (>= 0.44) 3/6 (42/44/100); beats V1-direct by margin (capability_go) 0/6.
DETERMINISTIC: two independent processes at seed 42 produce a byte-identical decode block (every RNG
derived from the `seed` arg; this runner uses a standalone numpy LIF, not the CoreSimConfig bridge, so the
`cfg.seed`/`actual_seed_used` trap does not apply — determinism is by explicit per-op seeds + this byte-
compare). No `sim/` file modified.

## What the mechanism SOLVES, and where the wall MOVED — the decomposition IS the finding

- **Spike quantization of the readout is SOLVED.** The rate→spike drop for the signed-linear readout is
  +0.0243 (and the E/I population decision-port costs only +0.0035), versus the config-C centroid's ~0.19 <!--derived-->
  rate→spike collapse. Temporal integration (G glimpses) + a population readout with the RIGHT signed
  weights recover essentially all the rate-linear-decodable signal from spikes. The per-unit quantization
  floor is no longer the binding constraint.
- **Readout LEARNING is now load-bearing, 6/6.** The random signed readout sits at chance (0.2552); the
  learned one reaches 0.4375. This REFUTES, for a signed common-mode-subtracting readout, the #72 verdict
  that "random == learned" (which held for the unsupervised centroid). The `tools.lab.attributable_to`
  helper flags "58% of the effect is in the control", but that is a chance-baseline artifact of the helper:
  the control is a CLASSIFIER whose null is chance 0.25, not zero — baseline-subtracted, (learned−chance)
  = 0.188 vs (random−chance) = 0.005, i.e. ~97% of the ABOVE-CHANCE accuracy is in the treatment arm. <!--derived-->
- **Position no longer leaks.** Config C leaked position at 0.97; here the class-population spike code
  decodes position at 0.3437 (near chance-position 0.25 + margin) while object decodes at 0.4271 — the MAX-
  over-locations read + the discriminative readout pool position out.
- **The capability is NOT reached, and the reason is measured.** The RATE linear-separability ceiling of
  the z-normalized C2 code is only ~0.4653, ~0.09 below config-B's raw-rate MAX+centroid ceiling (0.56).
  The z-norm lateral inhibition — REQUIRED to keep the near-threshold LIF graded (a raw cosine common-mode
  ~0.80 saturates it) — removes the common-mode MAGNITUDE the raw-rate centroid exploited. So the fully-
  spiking readout, having solved quantization, is now limited by the FEATURE CODE it reads, not by spikes.

## Honest residual + the next mechanism (no-defer)

The wall is no longer "spike quantization of the readout" (SOLVED: gap ~0.02) — it is the **~0.47 linear-
separability ceiling of the z-normalized C2 feature code**, ~0.09 below the raw-rate ceiling. Two named,
focused next mechanisms (not an abandonment; the capability is already carried on spikes through C1):

1. **Move common-mode rejection ENTIRELY to the readout.** Feed the S2 LIF a LIGHTER-normalized (or raw)
   cosine drive at a LOWER `s2_gain` so it stays in the graded (non-saturating) regime and preserves
   magnitude, and let the signed readout + its FF inhibition do the common-mode subtraction (its job). The
   pre-readout z-norm is a CONSTANT that competes with the readout's own inhibition and destroys magnitude
   — the classic "companion process replaced with a constant". Tune the LIF gain/normalization operating
   point JOINTLY with the readout (a 2D `s2_gain` x `s2_norm` sweep on the mini-PC pool — CPU, no agent
   tokens).
2. **Lift the linear-separability ceiling with a NONLINEAR spiking readout.** The z-normed configural
   distinction may not be fully linearly separable; a 2-layer spiking readout (a hidden layer of LIF
   conjunction / dendritic-coincidence units before the class populations) can exceed 0.47. Cortical
   readouts are not single-layer perceptrons.

The task's literal GO bar ("clears the NO-GO floor AND beats the random-readout control at >=5/6") is met
6/6 on the raw floor (>0.34) and 6/6 on beating random; the STRICT capability bar (beat V1-direct + reach
the config-B rate ceiling) is not. Reported as MIXED to avoid overclaim.

## Brain-based status

Somata genuinely SPIKE (LIF: leak, hard threshold, reset, absolute refractory, per-step membrane noise) at
S1, S2 AND the readout class populations. Common-mode rejection = feedforward inhibition (an interneuron
carrying the negatively-weighted pool; a Dale-compliant E/I decomposition). The decision = a spiking WTA
over class populations. The readout weights are set by a supervised ridge least-squares solve — a host-
computed teacher scaffold, the SAME status as config C's host centroid or R-STDP's host eligibility; its
online biological equivalent is an L2-decayed (synaptic-scaling) three-factor delta rule with the closed
form as its exact fixed point + instrument. FLAGGED innate developmental scaffolds (as config B/C):
retinotopic weight-sharing + pooling windows; the fixed random S2 bank. No live conversational vision
consumer exists (grepped #72), so production wiring is N/A; the scope is the spiking CAPABILITY.

## Reproduce

```bash
# 6-seed decisive (CPU/numpy, like the whole vision de-risk family; ~2.5 min/seed at idle):
SIM_BACKEND=numpy OMP_NUM_THREADS=4 .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
  --seeds 42 43 44 100 101 102 --ridge 0.5 --n-glimpses 2 --n-s2 96 \
  --out research/findings/raw/lanes/perception/vlin_readout_6seed.json
```

## Sources

- Maass, W., Natschläger, T. & Markram, H. (2002). Real-time computing without stable states. *Neural
  Comput.* 14:2531-2560. (A trained LINEAR readout of a spiking reservoir.)
- Frémaux, N. & Gerstner, W. (2016). Neuromodulated STDP and theory of three-factor learning rules. *Front.
  Neural Circuits* 9:85. (The supervised three-factor readout rule.)
- Brunel, N., Hakim, V., Isope, P., Nadal, J.-P. & Barbour, B. (2004). Optimal information storage and the
  distribution of synaptic weights: perceptron versus Purkinje cell. *Neuron* 43:745-757.
- Pouget, A., Dayan, P. & Zemel, R. (2000). Information processing with population codes. *Nat. Rev.
  Neurosci.* 1:125-132.
- Carandini, M. & Heeger, D. J. (2012). Normalization as a canonical neural computation. *Nat. Rev.
  Neurosci.* 13:51-62. (Divisive normalization = the FF-inhibition common-mode rejection.)
- Prior on this substrate: `2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited.md`
  (#72); `2026-08-19-vision-rstdp-readout-reframe-confirmed-capability-partial.md` (#75, the banked method).
