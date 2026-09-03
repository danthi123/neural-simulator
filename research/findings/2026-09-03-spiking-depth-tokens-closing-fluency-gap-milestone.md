---
type: finding
status: contributing
date: 2026-09-03
mechanism: spiking own-voice mouth with DEPTH (ssm/dual-nonneg n_layers=2) + tokens (contiguous) on Simple-Wiki BPE
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO (closing — biggest lever of the arc; still short of trigram-parity by ~0.12)
artifacts:
  - research/findings/raw/_emerge_wkv_lm_ssm_depth2_contiguous_6seed.json
---

# Spiking mouth + depth + tokens — closing the fluency gap (6-seed milestone)

**Status:** NO-GO but the biggest, most robust lever of the arc. The DEPLOYABLE spiking mouth (`--recurrence ssm --dual-nonneg --uniform-decay --n-layers 2 --contiguous`) closes most of the deepest-bucket trigram gap (numbers in the table below), up from its bare baseline. It still loses to a trigram (a wall defers a METHOD), but the path is now legible + measured.

## Result — depth is the strongest lever; the levers stack

<!--derived-->
From `research/findings/raw/_emerge_wkv_lm_ssm_depth2_contiguous_6seed.json`, deepest bucket (10-99), per-seed `margin_vs_trigram` (rounded): −0.125, −0.125, −0.125, −0.124, −0.115, plus seed 102 (tight cluster; mean in the table).

<!--derived-->
| deployable spiking-mouth config | mean margin_vs_trigram (6-seed) |
|---|---|
| bare (`n_layers=1`, sentence-mode) | −0.46 |
| + tokens (`--contiguous`) | −0.30 |
| **+ tokens + depth (`n_layers=2`)** | **−0.1252** |

<!--derived-->
Depth alone added +0.175 (−0.30 → −0.125) — the largest single lever measured. The three prior single-lever attempts (divnorm, tokens-alone, depth-implied) each fell short; **tokens + depth STACK** to close most of the gap. The deployable mouth has now climbed to the exact level the exact-math `wkv` sat at at sentence-mode (−0.125, finding `00a6b5a6`).

## The structural clue + next lever (NOT a wall)

<!--derived-->
The exact-math `wkv` went from −0.125 (sentence-mode, ~9.5M tokens) to +0.02 (contiguous, ~20.7M tokens) — the token boost carried it across the trigram. The spiking-depth mouth is at −0.125 WITH the contiguous token boost already applied — i.e. it is ~2× less token-efficient than the exact-math `wkv` (it needed depth + tokens just to reach the wkv's pre-token level). So the deployable mouth is "one more token-doubling behind" trigram-parity, NOT architecturally stuck.

**⛔ CORRECTION (deep-research-at-wall, same day): "more tokens" is the ALREADY-EXHAUSTED data lever, NOT the next step — and this whole arc partly re-derived a July 2026 conclusion.** The record already establishes (finding `2026-07-15-selective-ssm-generator-trigram-bound-both-levers-exhausted-not-a-scale-wall`) that the SSM/reservoir language family is TRIGRAM-BOUND with BOTH data AND size levers EXHAUSTED — an architectural capacity boundary, not a scale wall (the trigram saturates while the model stays ~0.26-0.33 above). More tokens does not cross it. Tonight's genuinely-new datum is that DEPTH is a stronger lever than the July size/data levers (closing to −0.125 vs their ~−0.30), but it remains on the same bound.

**The named surpass (already scoped July 2026, largely untested here):** a STRUCTURED HiPPO multi-timescale SSM basis (a principled basis of leaky integrators spanning fast→slow time constants) rather than dual-nonneg's hand-picked leaks — the field's spiking long-range mechanism (SpikingSSMs, arXiv:2408.14909; P-SpikeSSM, arXiv:2406.02923; scoped in `2026-07-13-fresh-gate-spiking-SSM-fixed-structured-multitimescale-recurrence-for-longrange`, runner `_ssm_fixed_structured_reservoir_derisk.py` exists). Secondary: a LOCAL-pool divisive-normalization (the over-everything pool was the divnorm failure `b500d421`).

**⭐ STRATEGIC FORK (for the owner — the July record reframes the metric):** finding `2026-07-15` states that beating a trigram in OPEN perplexity is a PROXY LENS that does NOT gate the deployed conversational capability — the mouth's deployed role is bounded-frame fluent wording behind the no-confab gate (rendering EMERGE frames on spikes), not open-perplexity trigram-beating. So the question is whether the own-voice mouth's TARGET is (a) open-ended fluency (→ the SSM family is architecturally bound; the surpass is a new class — HiPPO-structured SSM, or content-addressable learned-key attention which edges toward the transformer the project is retiring: a real brain-based-only tension), or (b) bounded-frame rendering (→ trigram-bound is adequate for the role; the metric is a proxy). This is a strategic decision, not a mechanism choice.

## Reproduce

```bash
# Produced research/findings/raw/_emerge_wkv_lm_ssm_depth2_contiguous_6seed.json (300W cap standing):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence ssm --dual-nonneg --uniform-decay --n-layers 2 --d-model 192 --batch 128 --tokenizer bpe \
    --corpus data/corpus/simplewiki.txt --contiguous --max-len 40 \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 \
    --json research/findings/raw/_emerge_wkv_lm_ssm_depth2_contiguous_6seed.json
```
