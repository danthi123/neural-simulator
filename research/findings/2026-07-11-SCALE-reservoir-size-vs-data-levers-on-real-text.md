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

## DATA sweep at n_pool=300 (the crucial one) — the bigram CATCHES UP with more data; the reservoir's edge is REGIME-DEPENDENT
| train sentences | reservoir CE (acc) | bigram CE (acc) | margin |
|---|---|---|---|
| 1400 | 3.264 (0.320) | 3.416 (0.365) | **+0.152** (reservoir beats bigram) |
| 5000 | 3.072 (0.338) | **3.040** (0.369) | **−0.032** (bigram overtakes) |

**With more data the BIGRAM improves MORE than the reservoir** (bigram CE 3.416→3.040 = −0.376; reservoir 3.264→3.072 = −0.192): the bigram was *data-starved* at 1400 sentences (many unseen bigrams under add-1 smoothing), and with 5000 it fits its ~V² table far better and edges past the 300-neuron reservoir, which is near its capacity ceiling. (Permuted-corpus control still collapses at 5000 → the reservoir does capture real structure; it just no longer *beats* a well-estimated bigram.)

## ⇒ the honest, refined picture (this TEMPERS the real-text GO)
Neither lever helps in isolation: **more reservoir overfits at fixed data** (+0.152→−0.42 as n_pool 300→1000 at 1400 sents), and **more data helps the bigram more at a fixed small reservoir** (+0.152→−0.032 as data 1400→5000 at n_pool=300). So the emergent generator's edge over the bigram on real text is REAL but **regime-dependent** — it lives in a narrow small-data + moderate-reservoir window, not a robust advantage that grows with either axis alone. A robust edge would require **co-scaling reservoir size WITH data** (a bigger reservoir, which overfits at 1400 sents, may become matched — and beat the bigram again — at 5000+). The committed 3-seed real-text GO (`-emergent-generator-beats-bigram-on-real-text.md`) stands but must be read at its scale (V=200, 1400 sentences, n_pool=300 — the bigram-data-starved regime); it is a foothold, not a robust win.

## OPEN (the single decisive experiment, deferred — heavy CPU run, held while the owner games)
**Does co-scaling recover a robust edge?** Run n_pool=1000 (which overfit at 1400 sents) at 5000–8000 sentences: if the larger reservoir — now matched to more data — beats the well-estimated bigram, the path scales by co-scaling size+data (promising, and exactly what the batched-GPU infra would accelerate). If it still ties/loses, the reservoir+linear-read-out architecture plateaus near the bigram on real text at this scale, and the honest wall is the field's (much bigger models + much more data). This co-scale run is the most important remaining scale experiment.

## Files
`_emerge_reservoir_lm_realcorpus_derisk.py`; raw `research/findings/raw/_rc/ts_s42.json`, `_rc_scale/np{600,1000}_s42.json`, `_rc_reg/np1000_wd{0.01,0.05,0.1}_s42.json`. Follows `2026-07-11-SCALE-emergent-generator-beats-bigram-on-real-text.md`.
