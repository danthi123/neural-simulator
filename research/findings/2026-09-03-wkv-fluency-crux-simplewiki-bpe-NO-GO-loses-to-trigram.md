---
type: finding
status: contributing
date: 2026-09-03
mechanism: WKV own-voice language cortex (exact-math, --recurrence wkv) fluency on Simple-English-Wikipedia BPE
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO
artifacts:
  - research/findings/raw/_emerge_wkv_lm_simplewiki_6seed.json
---

# WKV own-voice fluency crux (Simple-Wiki BPE) — NO-GO: the recurrence loses to a trigram at deep context

**Status:** NO-GO — a first-class deliverable (maps the wkv-family fluency limit at this scale; defers the METHOD/token-budget, not the capability). A wall defers a method, never the capability.

## What ran

The own-voice fluency retrain: `_emerge_wkv_lm_derisk --recurrence wkv --n-layers 2 --d-model 192 --batch 128 --tokenizer bpe --corpus data/corpus/simplewiki.txt` (BPE V=8001, n_train=908493 filtered 3-16-word sentences ≈ the full sentence-filtered pool), 6 seeds, at the crash-stable 300W cap (the 320W-overclock Xid-109 crashes were root-caused; see GAP_CLOSURE 2026-09-03 cont.). Elapsed 11402 s.

## Result — NO-GO, established tokenization-independently

The deepest-context bucket (positions 10-99) of `research/findings/raw/_emerge_wkv_lm_simplewiki_6seed.json`, `per_seed[<seed>].by_depth["10-99"]`, stores per-seed `wkv`, `bigram`, `trigram`, `wkv_memoryless`, `margin_vs_trigram`, `margin_vs_bigram`.

<!--derived-->
The per-seed `margin_vs_trigram` values (rounded from the artifact) are **−0.147, −0.128, −0.138, −0.092, −0.124, −0.123** (seeds 42/43/44/100/101/102) — every seed negative.

Means across the 6 seeds (computed from the artifact's per-seed buckets):

<!--derived-->

| metric (deep bucket 10-99), mean over 6 seeds | value |
|---|---|
| WKV NLL | 4.280 |
| trigram NLL | 4.157 |
| bigram NLL | 5.52 |
| WKV memoryless NLL | 4.63 |
| margin_vs_trigram | −0.125 (all 6 seeds negative) |
| margin_vs_bigram | +1.24 |

**The WKV recurrence LOSES to a plain trigram at deep context on every seed.** A fluent language model must beat a trigram (a trivially weak, 2-token-context baseline); this one does not. It beats a bigram (+1.24) and its own memoryless variant (4.28 < 4.63), so the recurrence captures *some* context, more than 1-gram — but at ~9.5M BPE tokens it captures it less efficiently than an explicit trigram's local counts. That is the robust, tokenization-independent NO-GO.

<!--derived-->
The absolute NLL 4.28 sits ~0.59 above the [3.0, 3.69] fluency band, but that band was calibrated on WORD-level runs (board #193's "~0.084 above" is word-level V=2000 wikitext103), so the cross-tokenization ABSOLUTE comparison is NOT load-bearing here — the internal trigram margin is.

## Honest read + next lever (NOT a wall)

<!--derived-->
The residual is LARGER than the word-level "~0.084 nats" narrative implied — consistent with board #193's token-starvation framing (fluency keeps improving with more training text), but the gap to even trigram-parity means more tokens alone may not suffice, and the loses-to-trigram signal is the thing to watch: does more data push the wkv past the trigram, or is its local modeling structurally weaker than count-based n-grams at this capacity?

**Prepared next lever (NO-GO branch, ready to fire):** `--contiguous --max-len 40` unlocks the sentence-filter-discarded text (~2.1× tokens ≈ 20.7M), the validated #193 methodology. VRAM caveat: max-len-40 sequences are ~2.4× longer → peak est ~17-23 GB on the 24 GB 3090; single-seed VRAM pre-check + `--batch 64` fallback first.

## The MORE important test, now running

The wkv model here is EXACT-MATH and, at n_layers=2, cannot be spike-realized (the deployed spiking mouth is `--recurrence ssm --dual-nonneg --uniform-decay --n-layers 1`, and the trainer asserts n_layers==1 in the ssm branch). The board's whole fluency narrative has been measured on this non-deployable exact-math family. The **deployable spiking (`ssm/dual-nonneg`) fluency de-risk is running now** (auto-queued after this crux; its artifact is pending — the run is still training) — that number is the actual brain-based-only baseline, never before measured at this scale. Honest prediction: likely NO-GO; next bio mechanism = a divisive-normalization / shunting-inhibition gate (a cross-lane transfer from the vision-lane `satdiv` work, itself BORDERLINE 2026-09-03).

## Reproduce

```bash
# The exact command that produced the cited artifact (deterministic given the seeds; 300W cap standing):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence wkv --n-layers 2 --d-model 192 --batch 128 --tokenizer bpe \
    --corpus data/corpus/simplewiki.txt \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 \
    --json research/findings/raw/_emerge_wkv_lm_simplewiki_6seed.json
```
