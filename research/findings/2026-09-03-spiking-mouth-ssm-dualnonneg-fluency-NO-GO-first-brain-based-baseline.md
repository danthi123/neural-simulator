---
type: finding
status: contributing
date: 2026-09-03
mechanism: spiking own-voice mouth (--recurrence ssm --dual-nonneg --uniform-decay, n_layers=1) fluency on Simple-Wiki BPE
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO
artifacts:
  - research/findings/raw/_emerge_wkv_lm_ssm_dualnonneg_simplewiki_6seed.json
---

# Spiking own-voice mouth (ssm/dual-nonneg) fluency — NO-GO: the FIRST brain-based-only baseline at this scale

**Status:** NO-GO — a first-class deliverable. This is the FIRST time the *deployable spiking* language-cortex family has been measured for fluency at this scale; the board's prior "near fluency" narrative was all on the non-deployable exact-math `wkv` family (see finding `00a6b5a6`). A wall defers a METHOD (this recurrence at this token budget), never the capability.

## Why this run matters

The deployed spiking mouth uses `--recurrence ssm --dual-nonneg --uniform-decay --n-layers 1` (few-spike leaky integrators, read via `Wo_sp`) — architecturally distinct from the exact-math `wkv` crux, which cannot be spike-realized (the trainer asserts n_layers==1 in the ssm branch, and reading a wkv checkpoint through the ssm math yields garbage). So the wkv crux was only ever an upper-bound proxy. This run swaps ONLY the recurrence family, apples-to-apples with the crux (same BPE V=8001, d192, batch128, corpus, 6 seeds), at the crash-stable 300W cap. Elapsed 0.98 h.

## Result — NO-GO (deepest bucket 10-99, means over 6 seeds; from the cited artifact's per_seed by_depth)

<!--derived-->

| model | deep-bucket NLL | margin_vs_trigram | margin_vs_bigram |
|---|---|---|---|
| spiking ssm/dual-nonneg (this run) | 4.616 | **−0.461** (all 6 seeds neg) | +0.90 |
| exact-math wkv (crux `00a6b5a6`) | 4.280 | −0.125 | +1.24 |
| trigram baseline (same corpus) | 4.155 | — | — |
| bigram baseline | 5.52 | — | — |

<!--derived-->
Per-seed spiking `margin_vs_trigram`: −0.492, −0.468, −0.470, −0.416, −0.459, −0.458 (seeds 42/43/44/100/101/102).

**The spiking recurrence DOES use context** — it beats a bigram by +0.90 nats (so it exploits more than 1-token history) and beats its own memoryless variant. **But it loses to a plain trigram by 0.46 nats on every seed** — a weaker content-selector than even 2-token count statistics, and markedly weaker than the exact-math wkv (which loses to the trigram by only 0.125). A fluent LM must beat a trigram; the deployable spiking mouth does not, at ~9.5M BPE tokens.

(Note: an intra-review correction — an earlier eyeball compared this run's 4.62 to the *wkv's* memoryless 4.63 and mislabeled it "memoryless-level". That was a cross-model mis-comparison; the spiking's own memoryless is 5.84, so its recurrence contributes ~1.2 nats. The load-bearing, model-independent read is the trigram margin above.)

## The mechanism diagnosis + next lever (NOT a wall)

The gap is architectural, not merely token-starvation: `dual-nonneg` (two independent positive leaky integrators) discards RWKV's numerator/denominator NORMALIZATION — the division by an accumulated decay-weighted denominator that gives `wkv` its content-addressed, softmax-like weighting over past tokens. Without it, the spiking state is per-channel leaky accumulation with no cross-time competition, so it selects "which past token matters now" worse than count-based n-grams.

**Convergent next mechanism (a wall → a new biology method):** restore RWKV-style normalization in a biologically-realizable form — a **divisive-normalization gate** on the dual-nonneg state (Carandini & Heeger contrast/sensory normalization: `R_i = drive_i^n / (sigma^n + pool)`). This is a FORWARD-pass content-selection computation — NOT a credit-assignment rule, and explicitly DISTINCT from the already-refuted dendritic/two-compartment/BDSP/burstprop deep-credit line (that addressed hidden-credit-on-spikes and failed on the frozen fixed-random feedback SIGNAL, not the topology; see `2026-07-22-gap4-real-issue-NOT-dendrites`). It is the SAME normalization the vision lane is independently working (`satdiv`, 2026-09-03, BORDERLINE — the strongest readout in that arc) — a genuine cross-lane transfer, and untested on the language recurrence. A secondary lever is a stacked-`ssm` depth path (blocked today by the n_layers==1 assert). The token lever (`--contiguous`, ~2.1× tokens) is more relevant to the wkv than to fixing a recurrence that is architecturally normalization-free.

## Arc convergence

Three walls this session point at ONE mechanism: (1) own-voice fluency (this NO-GO — the spiking recurrence can't content-select), (2) vision board #135 readout (the affine family exhausted; `satdiv` divisive-norm BORDERLINE), (3) scaffold-retirement (mostly BLOCKED on `neural-render`, the shared language frontier). Divisive normalization is the shared next build.

## Reproduce

```bash
# The exact command that produced the cited artifact (300W cap standing):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence ssm --dual-nonneg --uniform-decay --n-layers 1 --d-model 192 --batch 128 --tokenizer bpe \
    --corpus data/corpus/simplewiki.txt \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 \
    --save-ssm bridges/wkv_ckpt/wkv_ssm_bpe8k_d192_simplewiki \
    --json research/findings/raw/_emerge_wkv_lm_ssm_dualnonneg_simplewiki_6seed.json
```
