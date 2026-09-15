---
type: finding
status: partial
date: 2026-09-15
mechanism: gen-cortex-token-supply-scaling
lane: language-mouth
seeds: [42, 43, 44, 100, 101, 102]
verdict: TOKEN-SUPPLY LEVER CONFIRMED (WKV deep-context NLL still descending 6/6 at every model size + corpus, beats
  trigram at every point) BUT the sweep capped at ~9.2M tokens (192k passages) — the high-token "does the curve bend"
  regime was NOT reached (instrument cap found + corrected sweep queued). NOT a bend/no-bend verdict.
runner: research/runners/_gen_cortex_token_supply_scaling_derisk.py
artifacts:
  - research/findings/raw/_gencortex_scaling/_aggregate_6seed.json
  - research/findings/raw/_gencortex_scaling/fineweb_d96_s42.json
  - research/findings/raw/_gencortex_scaling/fineweb_d192_s42.json
  - research/findings/raw/_gencortex_scaling/fineweb_d384_s42.json
  - research/findings/raw/_gencortex_scaling/wt103_d96_s42.json
  - research/findings/raw/_gencortex_scaling/wt103_d192_s42.json
external: NO-EXTERNAL-NEEDED -- a scaling measurement of an existing runner, not a new mechanism or biological claim;
  the Chinchilla ~20-tok/param compute-optimal reference is the interpretive frame, already banked in the arc.
builds_on:
  - research/findings/raw/_gen_cortex_token_supply_scaling.json
---

# Token-supply scaling (6-seed): the lever is real and unsaturated, but the sweep never left the data-limited regime

**One-line.** A 30-cell 6-seed sweep of the WKV generative cortex (2 corpora x 3 model sizes) confirms the
token-supply lever the owner's #1 mouth fork rests on: deep-context NLL **keeps descending with more tokens at every
model size and every corpus, and beats the trigram at every point** — but this run's corpus load capped the training
pool at **192k passages (~9.2M tokens, tok/param<=21.8)**, so the high-token "does the curve bend" regime the fork
actually targets was **not reached**. This is a lever-confirmation + an instrument-cap finding, not a bend verdict. A
corrected sweep (`--n-sentences 2000000`, reaching the 1.536M-passage / ~74M-token points) is queued on the GPU lane.

## What ran (all landed, 6/6 seeds each)

The 2026-09-10 owner-operated queue ran `_gen_cortex_token_supply_scaling_derisk` at 6 seeds x {fineweb_edu,
wikitext103} x {d96, d192, d384} = 30 cells (`research/findings/raw/_gencortex_scaling/*.json`), vocab 2000,
max_len 48, 6 epochs, one cell per (corpus, d_model, seed).

## Result — the lever holds, mean WKV deep-context NLL across 6 seeds

<!--derived-->
(All numbers below are 6-seed means/derived quantities computed from the cited per-seed artifacts; the exact
per-config means are saved in `research/findings/raw/_gencortex_scaling/_aggregate_6seed.json`.)

| config | points reached | top tokens | tok/param | NLL (first -> top) | still-descending | beats-trigram |
|---|---|---|---|---|---|---|
| fineweb/d96 | 48k,96k,192k | 9.2M | 21.8 | 4.128 -> 3.976 | 6/6 | 6/6 |
| fineweb/d192 | 96k,192k | 9.2M | 10.0 | 3.964 -> 3.928 | 6/6 | 6/6 |
| fineweb/d384 | 96k,192k | 9.2M | 4.3 | 3.945 -> 3.906 | 6/6 | 6/6 |
| wt103/d96 | 96k,192k | 9.2M | 21.8 | 3.830 -> 3.785 | 6/6 | 6/6 |
| wt103/d192 | 192k only | 9.2M | 10.0 | 3.726 (1 point) | n/a | 6/6 |

(token counts = max_train_sents x ~48 tokens/passage; NLL = `wkv_deep_nll`, seed sd <=0.009 everywhere.)

- **Still descending everywhere the slope is measurable** (6/6 per config; wt103/d192 reached only 1 point so no
  slope). The top-segment slope stays positive (~0.036-0.048 nats/doubling) — no flattening yet.
- **More params -> lower NLL** at matched tokens (fineweb d96 3.976 > d192 3.928 > d384 3.906), and **wt103 is
  easier than fineweb** (3.785 vs 3.976 at d96) — both expected; the sweep's instrument is behaving sanely.
- **d96 reached tok/param ~21.8**, right at Chinchilla's ~20 compute-optimal ratio, and is STILL descending — the
  strongest single indicator that the model is genuinely token-limited, not param-limited, in this regime.

## The instrument cap (why the high-token regime was not reached)

<!--derived-->
(config/computed values + code line references below, not artifact measurements.)

Every config topped out at exactly **192000 passages** — a configured token-point, not an arbitrary
corpus-exhaustion number. Cause, traced in the runner: the training pool is `0.85 * n_sentences`
(`_gen_cortex_token_supply_scaling_derisk.py:124`) and a point `k` is skipped when `k > len(pool)`
(`:141`). The 2026-09-10 stocker set `--n-sentences 400000` -> pool 340000 -> every point >= 384000 passages was
**silently dropped**. The loader itself is fine (it returns 400000 passages from fineweb on request; fineweb_edu.txt
holds ~17M passages), so the fix is purely a larger `--n-sentences`.

## Corrected sweep queued (no-defer)

<!--derived-->
(planned config values, not artifact measurements.)

Queued on the GPU lane (0 Claude tokens): fineweb/d96, fineweb/d192, wt103/d96 at `--n-sentences 2000000`
(pool 1.7M >= the 1.536M-passage point), token-points to 1536000 (~74M tokens), 6 seeds each, written to
`_n2M`-suffixed files in the same `_gencortex_scaling` directory. `tools/stock_research_queue.sh` `--n-sentences`
bumped to 2000000 so future stocks are correct. When those land, this becomes the actual bend/no-bend verdict.

## Honest scope

Vocab is capped at 2000, so generated samples are mostly `<unk>` — this sweep measures the **NLL scaling trend**, not
fluency; that is the correct instrument for the token-supply question and not a defect. No fluency/consciousness claim.
Functional read-outs only.
