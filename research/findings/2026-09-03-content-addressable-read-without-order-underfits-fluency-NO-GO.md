---
type: finding
status: contributing
date: 2026-09-03
mechanism: content-addressable associative read (learned-key attention, `--recurrence assoc`) WITHOUT a positional/temporal signal — the first attention-family attempt at the own-voice fluency surpass
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO (worse than the linear-recurrence bound — a diagnosed underfit, not the attention ceiling)
artifacts:
  - research/findings/raw/_emerge_wkv_lm_assoc_depth2_contiguous_6seed.json
---

# Content-addressable read WITHOUT order underfits — a diagnosed NO-GO, not the attention ceiling

**Status:** NO-GO, and — important — the negative is a **diagnosed, fixable underfit**, not the attention family's ceiling. Owner steer 2026-09-03 is *pursue open fluency FULLY*; content-addressable / attention-like reads are ACCEPTED if biologically grounded (CA3 pattern-completion / modern-Hopfield). This is the first attention-family datum on the open-perplexity-vs-trigram target.

## Result — bag-of-tokens attention loses to a trigram, and to the linear recurrences

<!--derived-->
From `research/findings/raw/_emerge_wkv_lm_assoc_depth2_contiguous_6seed.json`, deepest bucket (10-99), per-seed `margin_vs_trigram`: −0.355, −0.332, −0.350, −0.342, −0.340, −0.361 → **mean −0.3467** (min −0.361, max −0.332; tight). Anti-cheat reads are healthy: mean memoryless-collapse +1.072 (the read DOES use content — a memoryless permute degrades it) and mean sequence-permute-collapse +3.375.

<!--derived-->
| deployable-mouth family (depth-2, contiguous, 6-seed) | mean margin_vs_trigram |
|---|---|
| spiking SSM dual-nonneg (`ssm`) | −0.1252 |
| **content-addressable read, no order (`assoc`)** | **−0.3467** |

So a learned-key content-addressable read, built WITHOUT any positional/temporal signal, is **~0.22 WORSE** than the simple spiking linear recurrence — even though "attention" is the more expressive family. That is the signature of an **underfit**, not a capacity ceiling.

## The diagnosis (read the training loss, not just the eval)

<!--derived-->
The `assoc` TRAINING loss converged to ~4.79 (epoch-5), WORSE than the recurrences' ~4.36 on the identical data — it fits the *training* set worse, so it is under-fitting, not over-fitting. Root cause: the read was built deliberately without a positional/temporal code, so within a causal prefix it sees an unordered **bag** of past tokens — it cannot use word ORDER, which a sequential recurrence gets inherently. (Sequence-permute still degrades it +3.375 because permuting the whole sequence changes which tokens fall in each prefix — the *set* changes; that is not evidence of within-prefix ordering.)

## The surpass (built + queued the same cycle — NOT a wall)

Per NO-DEFER, the negative launches the fix, not a stop. The diagnosed fix is a biologically-grounded temporal "when" signal on the read — hippocampal **time cells** (MacDonald et al. 2011) + the **Temporal Context Model** (Howard & Kahana 2002) — added to the Q/K projections only (content value untouched), so the content-addressable recall becomes *ordered*. Landed as `--recurrence assoc_t` (commit `ab2aa24a`, additive / other recurrences byte-identical).

<!--derived-->
Smoke confirms the direction (numbers quoted from the build-agent CPU smoke, not a committed 6-seed artifact): the temporal code LOWERED training loss (6.243 → 6.131) and improved deep-bucket margin (−0.415 → −0.381) at tiny scale. The 6-seed GPU de-risk of `assoc_t` is queued (ahead of the confirmatory wkv upper-bound). Open honesty flag from the smoke: at 1 epoch the permute-collapse anti-cheat did NOT strengthen as predicted (an absolute position code can pick up sentence-position statistics that survive permutation) — re-check at full scale.

## Reproduce

```bash
# Produced research/findings/raw/_emerge_wkv_lm_assoc_depth2_contiguous_6seed.json (300W cap standing):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence assoc --n-layers 2 --d-model 192 --batch 128 --tokenizer bpe \
    --corpus data/corpus/simplewiki.txt --contiguous --max-len 40 \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 \
    --json research/findings/raw/_emerge_wkv_lm_assoc_depth2_contiguous_6seed.json
```
