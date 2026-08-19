---
type: finding
status: mixed
date: 2026-08-19
lane: perception
mechanism: invariance-from-temporal-continuity
runner: research/runners/_vision_hmax_spiking_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/vhmax_spk_A_rate_baseline_6seed.json
  - research/findings/raw/lanes/perception/vhmax_spk_B_frontend_spiking_6seed.json
  - research/findings/raw/lanes/perception/vhmax_spk_C_full_spiking_6seed.json
---

# SPIKING port of the position-invariant CONFIGURAL HMAX hierarchy (board #72): the LIF-spiking S1->C1 FRONT END preserves the capability at near-rate accuracy, but FULL spike-coding of the S2->C2 configural readout is QUANTIZATION-LIMITED + position-leaky — the rate capability lives in a fine DISTRIBUTED cosine code (random==learned) that falls below the per-unit spike floor

**One-line verdict.** The rate de-risk
([`2026-08-19-vision-hmax-hierarchy-composed-pooling-...`](2026-08-19-vision-hmax-hierarchy-composed-pooling-solves-position-invariance-learning-not-load-bearing.md))
cleared position-invariant configural recognition at RATE level (HMAX held **0.5972** vs V1-direct **0.3698** vs
flat-pool **0.2674**, 5/6) and flagged the honest next step: build the S->C stack on SPIKING neurons. Built (no `sim/`
edit; reuses the deployed Gabor/V1 front end + rate rendering/decode BY IMPORT). Result, 6 seeds, decomposed by which
stages spike: **(A) rate control reproduces the published GO EXACTLY** (0.5972 / 0.3698 / 0.2674, 5/6 — pipeline
verified). **(B) LIF-spiking S1->C1 (real threshold + absolute refractory + membrane noise) + rate S2/C2 complex-cell
MAX largely PRESERVES the capability** — spike-count held **0.5625** (only −0.035 vs rate <!--derived-->), architecture load-bearing
**6/6**, beats V1-direct (0.4184) and flat (0.3003), position pooled out, scramble at chance, template-learning still
NOT load-bearing (random 0.5451 ≈ trace 0.5625). **(C) FULLY spike-coding the S2->C2 configural readout is a NOGO**:
held collapses to **0.3438** (latency) / 0.3594 (count), position LEAKS (decode 0.97), GO 0/6. This is NOT the substrate
failing to spike — the S1->C1 spiking (the perceptually HARD, position-variant stage) works; it is that the rate C2
DISCRIMINATION is a fine DISTRIBUTED cosine modulation (std ~0.04 on a ~0.8 common-mode; random projection == learned,
so distributed not sparse) that falls below the per-unit spike-count/latency quantization floor, and spike-coded global
pooling does not achieve the rate MAX's clean position-invariance.

`EXTERNAL-SEARCH-RAN:` the neural CODE prior is our own
[`2026-06-02-step2a-spiking-visual-word-recognition`](2026-06-02-step2a-spiking-visual-word-recognition-characterization.md)
(reading sparse spiking vision on THIS substrate needs first-spike LATENCY / rank-order + per-band kWTA lateral
inhibition, not spike-count — Thorpe/Masquelier 2007; Kheradpisheh et al. 2018, arXiv 1611.01421 <!--derived-->). The HMAX S/C
architecture + MAX-pool op are Riesenhuber & Poggio 1999 (verified against source in the rate finding). This runner tests
BOTH codes and reports the operating point; the strict GO is read off the latency code per that prior.

## Design — the ONLY change from the rate runner is rate arithmetic -> genuine spikes

Everything except the S/C layer arithmetic is REUSED BY IMPORT from the rate runner
(`_vision_hmax_hierarchy_derisk`): the histogram-MATCHED CONFIGURAL objects (3 oriented strokes at 3 fixed slots;
identity = the ARRANGEMENT; identical global orientation histogram so a flat pool is FORCED to chance), the deployed
Gabor/V1 front end (`sim.visual_cortex.build_v1_simple_weights` via `build_gabor_response_matrix`/`encode_v1`), the
hypercolumn competition+gate, the C1 innate local MAX-pool, the imprint/trace/random S2 template learners, and the
nearest-cosine-centroid decode. Held-out positions: train {0,2,4,6}, test {1,3,5,7} NEVER seen.

The new machinery (one function, `lif_spike_read`) is a genuinely SPIKING leaky integrate-and-fire layer: `dv =
(1/tau)(-v + gain*drive) + noise`; hard threshold -> reset -> absolute refractory; per-step membrane noise -> discrete
spike EVENTS. Read as spike COUNT (rate code) or FIRST-SPIKE RECENCY = T - t_first (latency / rank-order code). Stage
toggles (`--s1-mode`, `--s2-mode` = spiking|rate) isolate WHERE any drop lives:

- **S1 SPIKING**: the hypercolumn-shaped Gabor drive -> LIF somata -> spike code -> per-band kWTA (Thorpe/Masquelier
  lateral inhibition) -> C1 innate local MAX-pool per orientation (a spiking WTA = first-spike / strongest-count wins).
- **S2 SPIKING**: convolutional cosine template match -> lateral inhibition ACROSS the template bank per location
  (expose the winner-relative contrast) -> LIF coincidence units -> C2 global pool (MAX, top-k SUM, or a decoupled
  "MAX-the-drive-then a POPULATION of C2 somata spike" variant). The MAX op is the Riesenhuber-Poggio complex-cell
  pooling nonlinearity (feedforward inhibition), the SAME concession the rate finding made; the retinotopic
  weight-sharing + pooling windows are FLAGGED innate developmental scaffolds (complex-cell RFs are developmental).

## Result — 6 seeds (42/43/44/100/101/102), chance 0.25

| config (which stages SPIKE) | code | PRIMARY held | V1-direct held | flat-pool held | random-S2 held | strict GO | arch load-bearing | position decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A** rate / rate (control = the published rate path) | latency | **0.5972** | 0.3698 | 0.2674 | 0.5851 | 5/6 | 6/6 | pooled |
| **B** LIF S1->C1 + rate S2/C2 MAX | count | **0.5625** | 0.4184 | 0.3003 | 0.5451 | 3/6 | 6/6 | 0.3507 (pooled) |
| **B** LIF S1->C1 + rate S2/C2 MAX | latency | 0.4115 | 0.3767 | 0.2812 | 0.4167 | 1/6 | 4/6 | 0.3125 (pooled) |
| **C** FULLY spiking (LIF S1->C1 AND LIF S2->C2) | latency | 0.3438 | 0.3767 | 0.2812 | 0.3819 | 0/6 | 1/6 | 0.9688 (LEAKS) |
| **C** FULLY spiking (LIF S1->C1 AND LIF S2->C2) | count | 0.3594 | 0.4184 | 0.3003 | 0.3629 | 0/6 | 2/6 | 0.9688 (LEAKS) |

<!--derived-->
Nulls (all configs): pixel-scramble held ≈ chance (0.244-0.257); label-shuffle ≈ chance (0.236-0.288). Determinism:
config B re-run byte-identical per-seed. No `sim/` file modified (verified `git status`).

## What survives on spikes, and what does not — the decomposition IS the finding

- **A verifies the pipeline**: routing the SAME objects through this runner with both stages in rate mode reproduces the
  published rate GO to 4 decimals (0.5972 / 0.3698 / 0.2674 / 0.5851, 5/6, arch 6/6). Any spiking drop below is a real
  spike effect, not a reimplementation artifact.
- **B — the perceptually HARD front end survives genuine LIF spikes.** Converting S1 (V1 simple) to LIF somata (hard
  threshold, absolute refractory, per-step noise) and C1 to a spiking WTA MAX-pool costs only −0.035 <!--derived--> held with the
  spike-COUNT code (0.5625 vs 0.5972), keeps architecture load-bearing 6/6, beats both the V1-direct and flat-pool
  floors, pools position out, and reproduces the rate finding's central verdict — template-LEARNING is NOT load-bearing
  (random 0.5451 ≈ trace 0.5625). The position-invariant CONFIGURAL capability is carried on spikes by the innate
  composed-pooling topology, exactly as at rate.
- **B's strict per-seed GO is only 3/6 (count) / 1/6 (latency) — a characterized PARTIAL, not a GO.** The reason is
  specific and measured: the spike-COUNT V1-direct floor RISES to 0.4184 (vs the rate 0.3698), because the C1 local
  MAX-pool over a spike-count code confers extra position tolerance on the position-specific read too — so the HMAX
  advantage over V1-direct shrinks below the strict per-seed +0.10 bar on 3 seeds. The pooled HMAX-over-V1 margin is
  +0.144 <!--derived--> (count); the effect is real and MODERATE, and the op-point was not tuned to force 6/6.
- **B latency < B count here (0.41 vs 0.56).** This is the opposite of the 2026-06-02 prior (latency > count), and the
  reason is instructive: that prior read recognition DIRECTLY off ~0.03-rate V1 (each cell ~3 spikes/window -> count is
  hopeless, latency wins); HERE the C1 local MAX-pool over a moderate-drive count already denoises, and the fine
  configural discrimination is better preserved by graded counts than by coarse first-spike recency. The right code is
  stage- and drive-regime-dependent, not universal.

## The FULL-spiking NOGO — quantified, and it names the next mechanism (not a wall)

Spike-coding the S2->C2 configural READOUT (config C) drops held to ~0.34-0.36 and LEAKS position (decode 0.97). Root
cause, measured: the rate C2 code discriminates by a fine DISTRIBUTED cosine modulation — across-template response std
**~0.042 on a common-mode of ~0.80** <!--derived--> — and, per the rate finding, a RANDOM S2 projection matches the learned one, so
the discriminative signal is DISTRIBUTED (Johnson-Lindenstrauss), NOT a sparse set of strongly-selective units. Two
independent spike costs then bite:

1. **Sub-quantization discrimination.** A per-unit spike-count/latency code over a finite window cannot resolve a ~5%
   (0.04/0.8) modulation; the LIF threshold nonlinearity further compresses the narrow cosine band (median 0.61, p10-p90
   0.30-0.79) by saturating (~79% of locations fire at usable gain), washing out the pattern the rate cosine-centroid
   decode divides out.
2. **Spike-coded pooling is not cleanly position-invariant.** The rate global MAX over all locations is exactly
   position-invariant; a spike-coded top-k SUM / population read lets the noisy top-k SELECTION and absolute magnitude
   carry position into the C2 pattern (survives cosine-normalisation), so position decodes at 0.97.

**The biological surpasses were TRIED (>=5 lever-classes), and recover it only PARTWAY**: lateral inhibition across the
S2 template bank (submean / z-competition), per-band kWTA, longer integration (T2 up to 500), population coding (n_S2 up
to 1024; a decoupled MAX-the-drive-then-population-of-C2-somata read), and both codes. On a CLEAN (rate) C1 the best
spiking S2/C2 reaches ~0.42 (vs rate 0.59); the decoupled-population read climbs to ~0.44-0.46 but re-introduces the
position leak. This is a characterized OPERATING POINT, not a hard limit.

**Named next mechanism (the method the wall points to).** The rate finding's "template-LEARNING not load-bearing"
(random == learned) is, on spikes, precisely the fragility: a DISTRIBUTED random code is quantization-fragile, whereas a
SPARSE SELECTIVE code (few strongly-firing, sharply-tuned S2 units) survives spike coding and is what the proven
Thorpe/Masquelier conv-SNN uses. So the spiking port REFRAMES the rate verdict: **on spikes, DISCRIMINATIVE sparse S2
learning (reward-modulated STDP / a supervised sparse readout) is predicted to become LOAD-BEARING** where unsupervised
trace/random are not. That is a focused sub-arc (a named method), not an abandonment of the capability — the capability
is already carried on spikes through C1; what remains is a spike-ROBUST configural readout.

## Production wiring (secondary) — BLOCKED on no live vision consumer

Grepped the live conversational path: `POST /api/brain-chat` (`webapp/server.py`) ingests only `message: str` (text);
`ChatBrain` (`research/runners/brain_chat_tui.py`) has no image/retina/visual input. The `--enable-visual-cortex` flag
is a separate NAV-training pathway (K-v2 gridworld agent/goal blobs), not object-anywhere recognition in the live brain.
Per the task ("if production has no live vision consumer, say so and STOP; don't invent one"): **production wiring is
blocked on the absence of a live conversational vision endpoint** — the honest scope is the spiking CAPABILITY above, not
a production flip.

## Honest residuals / scope

- The headline positive (B) is a PARTIAL (strict GO 3/6), not a GO, and it is NOT "fully spiking" — its S2/C2 readout is
  the rate Riesenhuber-Poggio MAX (a biologically-shaped host op). The fully-spiking target (C) is a characterized NOGO.
- The result is backend-independent in the sense that matters: a RATE model is generous, and the fully-spiking
  configural readout underperforms it — exactly the "if rate fails, spiking will not save it" logic run forward. The
  next lever is a spike-robust (sparse selective) code, not more op-point tuning.
- The retinotopic weight-sharing + pooling windows remain FLAGGED innate scaffolds (the rate finding's defended
  concession; complex-cell RFs are developmental).

## Reproduce

```bash
# A rate control (reproduces the published rate GO):
SIM_BACKEND=numpy OMP_NUM_THREADS=2 .venv/bin/python -u -m research.runners._vision_hmax_spiking_derisk \
  --seeds 42 43 44 100 101 102 --code latency --s1-mode rate --s2-mode rate --n-s2 128 \
  --out research/findings/raw/lanes/perception/vhmax_spk_A_rate_baseline_6seed.json
# B front-end spiking (LIF S1->C1 + rate S2/C2 MAX):
SIM_BACKEND=numpy OMP_NUM_THREADS=2 .venv/bin/python -u -m research.runners._vision_hmax_spiking_derisk \
  --seeds 42 43 44 100 101 102 --code both --s1-mode spiking --s2-mode rate --n-s2 128 \
  --out research/findings/raw/lanes/perception/vhmax_spk_B_frontend_spiking_6seed.json
# C fully spiking (LIF S1->C1 AND LIF S2->C2):
SIM_BACKEND=numpy OMP_NUM_THREADS=2 .venv/bin/python -u -m research.runners._vision_hmax_spiking_derisk \
  --seeds 42 43 44 100 101 102 --code both --s1-mode spiking --s2-mode spiking --s2-norm z --s2-gain 2.0 \
  --c2-pool ksum --c2-k 5 --n-s2 256 \
  --out research/findings/raw/lanes/perception/vhmax_spk_C_full_spiking_6seed.json
```

## Sources

- Riesenhuber, M. & Poggio, T. (1999). Hierarchical models of object recognition in cortex. *Nature Neuroscience*
  2:1019-1025. (HMAX; the complex-cell MAX pooling op.)
- Thorpe, S. & Masquelier, T. and Kheradpisheh, S. R. et al. (2018), arXiv 1611.01421 <!--derived-->. (Latency / rank-order coding +
  per-layer lateral inhibition for spiking object recognition.)
- Prior on this substrate: `research/findings/2026-06-02-step2a-spiking-visual-word-recognition-characterization.md`.
