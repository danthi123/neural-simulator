# SCALE levers (real text) — naively growing the reservoir HURTS at fixed data: the emergent generator's CE margin over the bigram PEAKS at a moderate reservoir and DECLINES with size; regularization only partly mitigates → the scale lever is DATA co-scaled with reservoir size, not raw capacity

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_realcorpus_derisk.py` (`--n-pool`, `--weight-decay`); TinyStories V=200, seed 42, 1400 train / 300 held-out sentences. Reuse-by-import; NO `sim/` edit.
**Verdict:** **How to scale the emergent generator, characterized honestly (prompted by owner questions on GPU + scaling).** On real text the reservoir generator's advantage over the bigram (`+0.152` nats CE at n_pool=300, the committed real-text GO) does NOT grow with reservoir size at fixed data — it DECLINES monotonically and goes NEGATIVE. Regularization partially mitigates the largest reservoir but does not restore it. ⇒ the scale lever is **DATA (co-scaled with reservoir size)**, with regularization a secondary knob — NOT raw reservoir capacity.

## Result — TinyStories V=200, seed 42 (bigram CE ≈ 3.42; margin = bigram_CE − reservoir_CE, higher = better)
**Reservoir-size sweep (read-out weight-decay fixed at 0.001):**
| n_pool | reservoir CE | margin over bigram |
|---|---|---|
| **300** | 3.264 | **+0.152** (beats bigram — the committed GO) |
| 600 | 3.436 | −0.020 (ties) |
| 1000 | 3.836 | **−0.420** (much worse) |

**Regularization sweep at n_pool=1000 (baseline wd=0.001 → −0.420):**
| weight-decay | reservoir CE | margin |
|---|---|---|
| 0.001 | 3.836 | −0.420 |
| **0.01** | 3.666 | **−0.250** (best — partial recovery) |
| 0.05 | 3.922 | −0.506 (over-regularized) |
| 0.1 | 4.098 | −0.682 (underfits) |

## The honest diagnosis
- **Bigger reservoir ⇒ bigger read-out ⇒ over-parameterized for 1400 sentences.** The read-out is `V × (n_pool+1)`; tripling n_pool triples its parameters against a fixed data budget, and held-out CE degrades — the classic reservoir-computing over-capacity effect.
- **But it is NOT pure read-out overfitting** fixable by regularization: the best weight-decay at n_pool=1000 (0.01) recovers only from −0.420 to −0.250 — still worse than the bigram AND worse than the small n_pool=300 (+0.152). Over-regularizing (0.05/0.1) underfits and worsens it. So the large reservoir is **data-starved**: it has the capacity to capture more structure but not enough data to fit it without overfitting. The moderate n_pool=300 is simply matched to the 1400-sentence budget.
- **⇒ The scale lever is DATA, co-scaled with reservoir size.** To benefit from a larger reservoir you must give it more data; regularization is a secondary tuning knob (a mild wd helps, too much hurts). Consistent with the project's other "data-bound, not compute-bound" findings (e.g. the 88.6M generator data-bound at 41M tokens).

## Ties to the GPU/infra question (owner-asked)
This is why the reservoir-LM has run on CPU: at the current data scale it is fast (~70–170s), and *growing the reservoir* — the thing GPU would accelerate — HURTS. The real lever (more DATA) makes the **sentence-serial reservoir forward the CPU bottleneck** (linear in sentences), and THAT is when the GPU work pays off — but not by flipping `enable_rf_cudagraph` (that megakernel is RF-resonate-specific; the reservoir uses the general Izhikevich `_run_one_simulation_step`, plus a per-step `to_host` device→host sync). The enabling infra for the data-scale path is: (1) a CUDA-graph capture of the Izhikevich step, (2) on-device firing accumulation (drop the per-step `to_host`), (3) sentence batching — all precedented in the project, none built for this path. See the AUTONOMOUS_STATE infra note.

## OPEN (the decisive confirmation, deferred — heavy CPU run, held while the owner games)
Run n_pool ∈ {300, 1000} with MUCH more data (e.g. 4000–8000 train sentences) and check: does the n_pool=300 margin GROW with data (the emergent generator improving with scale — the promising direction), and does the n_pool=1000 margin recover/exceed it once data is co-scaled (confirming the data-lever diagnosis)? This is the single most important scale experiment; it is a long CPU run (and its scaled form is what the batched-GPU infra above would accelerate).

## Files
`_emerge_reservoir_lm_realcorpus_derisk.py`; raw `research/findings/raw/_rc/ts_s42.json`, `_rc_scale/np{600,1000}_s42.json`, `_rc_reg/np1000_wd{0.01,0.05,0.1}_s42.json`. Follows `2026-07-11-SCALE-emergent-generator-beats-bigram-on-real-text.md`.
