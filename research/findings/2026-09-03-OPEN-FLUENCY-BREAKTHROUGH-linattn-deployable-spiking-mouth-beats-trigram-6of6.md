---
type: finding
status: milestone
date: 2026-09-03
mechanism: --recurrence linattn (normalized Hebbian fast-weight linear attention) — the deployable spiking own-voice mouth
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: BREAKTHROUGH — the deployable spiking content-addressable read BEATS a fair trigram at deep context on ALL 6 seeds, the first open-fluency crossing at deployable scale
artifacts:
  - research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json
---

# Open-fluency breakthrough: the deployable spiking own-voice mouth beats the trigram, 6/6

**Status:** MILESTONE. The brain's own from-scratch SPIKING language mouth (`--recurrence linattn`) beats a fair interpolated trigram at deepest context on all 6 seeds — the first time in the arc the deployable (spiking-realizable) mouth has crossed the fluency bar it kept losing to. Honestly bounded below (this is a small from-scratch LM; the flip is owner-gated + needs a live brain-grounded/honest verification).

## Result — 6/6, tight

<!--derived-->
From `research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json`, deepest bucket (10-99), per-seed margin_vs_trigram: +0.049, +0.053, +0.051, +0.051, +0.060, +0.039 → **mean +0.0505** (min +0.039, max +0.060; 6/6 > 0). Anti-cheats clean every seed: perm-collapse ~+4.0 (uses order), memoryless-collapse ~+1.4 (uses content).

<!--derived-->
| deployable-mouth family (depth-2, contiguous, 6-seed) | mean margin_vs_trigram |
|---|---|
| bag content-addressable attention (no order) | −0.347 |
| spiking SSM dual-nonneg (recurrence) | −0.125 |
| ordered attention (assoc_t) | −0.147 |
| **linattn (normalized Hebbian fast-weight)** | **+0.0505 (6/6 cross)** |

## The mechanism + the chain that got here

`linattn` is a **normalized Hebbian fast-weight linear attention**: a real-valued D×D outer-product KV trace `M_t = λ⊙M_{t-1} + φ(k_t)⊗v_t` with a running denominator `zden_t`, read `= φ(q_t)ᵀM_t / (φ(q_t)ᵀzden_t + ε)`, read OUT by spikes (SpikeGPT's num/den generalized to a full outer product; bio: short-term synaptic plasticity / fast weights, CA3 autoassociation, divisive normalization). It GENERALIZES the exact-math wkv and — the load-bearing bit — RESTORES the content-weighted normalization that the previous spiking mouth (dual-nonneg) dropped, which the diagnosis identified as the missing piece.

<!--derived-->
The chain (each a checked step): the ssm/dual-nonneg spiking mouth was NO-GO (−0.46 → −0.125 with depth+tokens, bound); the diagnosis (external spiking-LM literature: every working spiking LM keeps a content-weighted normalizer, those that drop it lag) named the fix; the design (`066dde61`) specified linattn; the CPU ablation CONFIRMED norm is load-bearing (norm-ON +0.456 vs norm-OFF +0.190); the 6-seed GPU de-risk here = **+0.0505, 6/6**; the generation-coherence check confirmed it writes genuinely structured English (viable mouth, better than the ssm control which mode-collapsed); the `LinAttnReadout` deployment read-back is EXACT-parity-verified (~4.4e-16 vs the torch layer, a test that caught a real silent-wrong-recurrence bug). <!--derived-->

## Honest bounds (the flip is NOT automatic)

- This is a SMALL from-scratch LM on ~13.5M BPE tokens of Simple-English-Wikipedia; coherence is childlike, not GPT-level, and the crossing margin is modest (+0.05). The claim is precisely "beats a fair trigram at deep context, 6/6," not "fluent like an LLM."
- The `num/den` division is one graded host op (its shunting-inhibition spike-native realization is a named later rung / honest residual). State is real-valued/graded (same concession as the shipped ssm mouth), read-out genuinely spiking.
- Known quality limit: the BPE tokenizer drops capital letters (task-chipped) — the mouth can't yet capitalize.
- **Production flip is OWNER-GATED and gated on a LIVE verification** (running next): does linattn-as-mouth produce (i) fluent, (ii) BRAIN-GROUNDED (vary brain state → reply changes; lesion → vanishes — the anti-hollow test), (iii) HONEST (the no-confab moat holds; a broad-vocab mouth is a NEW fabrication surface) turns in `answer_turn`. The deployment wiring (`BRAIN_WKV_MOUTH_RECURRENCE=linattn`) is built + parity-verified + default-off.

## Reproduce

```bash
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence linattn --n-layers 2 --uniform-decay --d-model 192 --batch 128 --tokenizer bpe \
    --corpus data/corpus/simplewiki.txt --contiguous --max-len 40 \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 --tok-cache \
    --save-ssm bridges/wkv_ckpt/wkv_linattn_depth2_contiguous \
    --json research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json
```
