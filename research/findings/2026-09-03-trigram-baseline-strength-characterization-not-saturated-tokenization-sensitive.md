---
type: finding
status: contributing
date: 2026-09-03
mechanism: empirical characterization of the FAIR interpolated-trigram baseline (`fit_interp_trigram`) that `margin_vs_trigram` measures own-voice fluency models against, on Simple-English-Wikipedia BPE
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44]
verdict: characterization (not a GO/NO-GO) — the trigram is NOT saturated at the crux's own token budget (still improving ~0.02-0.05 nats/M tokens through 13.5M), so "more tokens is an exhausted lever" (quoted from a DIFFERENT, V=200/tens-of-thousands-of-tokens arc) does not hold at this scale; per-token tokenization sensitivity is largely a granularity artifact (char-level's apparent strength inverts once normalized to nats/word) but BPE vocab SIZE genuinely moves the raw per-token bar
artifacts:
  - research/findings/raw/_trigram_baseline_characterization.json
  - research/findings/raw/_emerge_wkv_lm_assoc_depth2_contiguous_6seed.json
---

# The fair-trigram baseline: NOT saturated at crux scale, and its per-token strength is a tokenization-granularity artifact

**Status:** characterization, empirical complement to the parallel literature investigation. Answers three questions about the `fit_interp_trigram` baseline that `margin_vs_trigram` (the own-voice fluency gate's deep-context, d10-99, metric) measures every recurrence family against: (1) does the trigram saturate with more tokens at the crux's actual scale, (2) is BPE tokenization making it an unusually strong baseline, and (3) is the standing −0.125 bound better explained by a strong baseline, a data-limited model, or both. <!--derived-->

## Method — reuse, not reimplementation

`fit_interp_trigram`, `_BPEVocabAdapter`, `BUCKETS`/`_bucket`, `DEFAULT_BPE_PATH`, `load_sentences`, `load_stories` are imported **verbatim** from `research.runners._emerge_wkv_lm_derisk` (new runner `research/runners/_trigram_baseline_characterization_derisk.py`, reuse-by-import, no `sim/` edit, CPU/numpy only — confirmed importing the module pulls in neither `torch` nor `cupy`). `--verify` reproduces one cell of the actual crux exactly (corpus=`data/corpus/simplewiki.txt`, production BPE V=8001, seed=42, `--n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000`) and checks it against `research/findings/raw/_emerge_wkv_lm_simplewiki_6seed.json`'s seed-42 record:

<!--derived-->

| | replica | target (crux artifact, seed 42) |
|---|---|---|
| deep-bucket (10-99) trigram NLL | 4.1567 | 4.157 |
| lambdas (l0,l1,l2,l3) | (0.05, 0.05, 0.2, 0.7) | (0.05, 0.05, 0.2, 0.7) |
| n_eval (deep bucket) | 21426 | 21426 |

**PASS — exact match.** A useful byproduct: the replica also measures the EXACT training-token count the crux config uses, which the record's "~9.5M tokens" shorthand undercounts — the actual full sentence-mode pool is **13,479,719 train tokens** (908,493 sentences; the whole pool, before the 15% eval split, tokenizes to 15,856,890 tokens under the production BPE, matching `GAP_CLOSURE_MISSION.md`'s "~15.85M BPE tokens" corpus-level figure exactly).

## Sweep 1 — token budget (sentence-mode, production BPE V=8001, holding tokenizer/regime fixed): NOT saturated

<!--derived-->

| train tokens (target) | mean deep (10-99) trigram NLL, 3 seeds (42/43/44) | per-seed |
|---|---|---|
| ~2.0M | 4.8041 | 4.8171 / 4.7409 / 4.8543 |
| ~5.0M | 4.4563 | 4.4731 / 4.4053 / 4.4905 |
| ~9.5M | 4.2457 | 4.2619 / 4.2050 / 4.2702 |
| ~13.48M (full sentence-mode pool, seed 42 only — the crux's own operating point) | 4.1567 | (n/a, single-seed cross-check; the crux's own 6-seed mean at this exact point is **4.155**) |

<!--derived-->
Marginal improvement rate (mean over seeds, budget-sweep points; the 9.5M→13.5M step is single-seed):

| interval | Δ trigram NLL | tokens added | rate (nats / 1M tokens) |
|---|---|---|---|
| 2.0M → 5.0M | −0.348 | +3.0M | −0.116 |
| 5.0M → 9.5M | −0.211 | +4.5M | −0.047 |
| 9.5M → 13.5M (1 seed) | −0.089 | +4.0M | −0.022 |

The curve is the classic diminishing-but-nonzero count-based n-gram data-scaling shape (roughly halving slope each time the interval widens) — **not a plateau**. At the crux's own 13.5M-token operating point the trigram is still improving at a clearly measurable, non-negligible rate.

**This directly contradicts a claim currently load-bearing in the record.** `research/findings/2026-09-03-spiking-depth-tokens-closing-fluency-gap-milestone.md` states: *"the SSM/reservoir language family is TRIGRAM-BOUND with BOTH data AND size levers EXHAUSTED — an architectural capacity boundary, not a scale wall (the trigram saturates while the model stays ~0.26–0.33 above). More tokens does not cross it,"* citing `research/findings/2026-07-15-selective-ssm-generator-trigram-bound-both-levers-exhausted-not-a-scale-wall.md` as having "already establish[ed]" this for the current arc. Reading that July source directly: its saturation claim is measured at **V=200 word-level vocabulary** on a **tens-of-thousands-of-tokens** corpus (`nt=24000→48000`, its own text: *"the trigram saturates ~2.42"*) — three-to-four orders of magnitude smaller in both vocabulary and token count than the 2026-09-03 arc's BPE-V=8001/simplewiki/millions-of-tokens regime, and a different absolute NLL scale entirely (~2.4 vs ~4.2 nats — different vocab size changes the entropy floor). **The July finding's saturation verdict does not transfer to the current arc's regime; it was cited across a regime boundary without being re-verified there, and the direct re-measurement above falsifies it at the regime that actually matters.** "More tokens is an exhausted lever" is not established for the BPE-V8001/simplewiki arc — the trigram side of that lever is demonstrably still live.

## Sweep 2 — contiguous regime (~20-28M tokens): weaker, but a REGIME change, not a token-count effect

<!--derived-->
`--contiguous --max-len 40` (`load_stories`, chops raw text into fixed 40-word windows regardless of sentence boundaries — the crux's own "unlock more tokens" lever) gives, at the same production BPE:

| | trigram deep (10-99) NLL |
|---|---|
| this runner, seed 42 (n_train_tok = **28,223,786**, precisely measured — not the "~20.7M" figure quoted in the board, which appears to be a raw-word approximation, not an actual BPE token count) | 4.4146 |
| `research/findings/raw/_emerge_wkv_lm_assoc_depth2_contiguous_6seed.json`, 6 seeds (tracked in git) | 4.415 / 4.438 / 4.404 / 4.42 / 4.414 / 4.401 → mean **4.4153** |
| the same 6 seeds' trigram NLL in the (currently uncommitted, live-tree-only) `_emerge_wkv_lm_ssm_depth2_contiguous_6seed.json` | **byte-identical per seed** to the row above (a clean cross-run determinism check: the trigram fit depends only on seed+data, not on which recurrence trains alongside it) |

My seed-42 replica (4.4146) matches the existing 12-independent-fit evidence (4.415) closely. **The trigram gets WEAKER (higher NLL, 4.155 → 4.415, +0.26 nats) going from sentence-mode (~13.5M tokens, clean 3–16-word filtered sentences) to contiguous mode (~28.2M tokens, arbitrary 40-word raw-text chops) — despite having ~2× more training tokens.** This means the sentence-mode → contiguous "token boost" is entangled with a genuine regime change (headers/lists/mid-sentence cuts are harder for a local count-based trigram than clean sentences), not a controlled more-tokens-same-distribution comparison. The single-seed `_emerge_wkv_lm_contiguous40_1seed_vramcheck.json` datum quoted elsewhere in the record as "the exact-math wkv crosses to +0.02 (contiguous, ~20.7M tokens)" is **genuinely single-seed** (the matching 6-seed run, `_emerge_wkv_lm_contiguous40_6seed.json`, exists only as a 1-line log — it was queued and apparently deprioritized behind the ssm-depth2/assoc/assoc_t runs on the single GPU) — worth flagging since that framing is used prose-side as settled. Whatever the wkv's real 6-seed contiguous margin turns out to be, part of any apparent "crossing" is attributable to the trigram baseline itself growing measurably weaker in this noisier regime, not solely to the model improving.

## Sweep 3 — tokenization (same underlying text — the full 908,493-sentence pool — held fixed, only granularity varies)

<!--derived-->

| tokenizer | V | mean train tokens (3 seeds, or 1 where noted) | mean deep trigram NLL (per-token, the metric's own unit) | tokens/word | **nats/word** (NLL × tokens/word — the tokenization-invariant comparison; see Reproduce for the citation) |
|---|---|---|---|---|---|
| char-level (built from the pool) | 29 | 52,835,280 | **2.1076** | 5.633 | **11.87** |
| BPE, simplewiki-trained | 2001 | 18,693,078 | 3.2261 | 1.993 | 6.43 |
| BPE, simplewiki-trained | 8001 | 14,478,408 | 3.9874 | 1.544 | 6.16 |
| BPE, **production** (wikitext103-trained, applied to simplewiki — what the crux actually uses) | 8001 | 13,479,719 (seed 42 only) | 4.1567 | 1.437 | **5.97** |
| BPE, simplewiki-trained | 16001 | 12,283,200 | 4.6443 | 1.310 | 6.08 |

**The per-token ordering (the metric's own units) and the nats/word ordering DISAGREE, and the disagreement is the finding.** On raw per-token NLL — the unit `margin_vs_trigram` actually uses — char-level (2.11) looks dramatically stronger than any BPE variant (3.2–4.6), a >2× spread. That is a **granularity artifact**: predicting one of 29 characters at a time is trivially easier per prediction step than predicting one of 8001+ subword pieces, regardless of how good the underlying model of the text is (external grounding: Mielke, *"Can you compare perplexity across different segmentations?"*, https://sjmielke.com/comparing-perplexities.htm — per-token perplexity/NLL is not comparable across tokenizations with different tokens-per-word ratios; only the underlying per-document log-likelihood, equivalently nats/word or nats/char, is segmentation-invariant). Once normalized to **nats per word**, the ordering **inverts**: char-level is the *weakest* baseline of the five (11.87, roughly double every BPE variant), and among the BPE variants there is a small, monotonic, real effect — **larger vocabulary → lower nats/word** (2000→6.43, 8000-simplewiki→6.16, 16000→6.08) — because a bigger vocabulary packs more of a word's identity into fewer, more informative tokens. The production tokenizer (8001, trained on wikitext103 rather than simplewiki) scores best of all (5.97) — a training-corpus effect distinct from vocab size (it also used different bounded-training hyperparameters, `top_k_words=5000` vs this sweep's `8000`, so the 8000-vs-8000 comparison is not a clean single-variable ablation; noted as a caveat, not over-interpreted).

**Answering "is BPE-trigram unusually strong":** not in the sense of being an inflated artifact relative to a "true" weaker baseline hiding underneath — the opposite. Char-level's low per-token numbers are the illusion; BPE's higher per-token numbers are closer to the real per-word information content. Switching tokenization to weaken the trigram is not a free lunch. The one place tokenization genuinely, non-artifactually moves the **raw per-token bar** (the metric the gate actually reads) is BPE **vocabulary size**: V=16000 raises the trigram's own per-token NLL from 4.157 (V=8001) to 4.644 — a **+0.49 nats easier target in the metric's own units**, *if* a model's own per-token NLL does not degrade by more than 0.49 nats when moving to the larger, sparser vocabulary. That is untested here (this sweep only measured the trigram side, as scoped) and is flagged below as a cheap, concrete next lever — not resolved.

## Decomposition verdict

Is the standing **−0.125** deep-context bound (wkv, ssm/dual-nonneg, hippo, and assoc all converge near it — see `research/findings/2026-09-03-spiking-depth-tokens-closing-fluency-gap-milestone.md`) better explained by a strong-baseline artifact, a data-limited model, or both? <!--derived-->

**Both, and more precisely: BOTH SIDES are demonstrably data-limited at this exact token budget, which the record had NOT verified for the trigram side.** The trigram itself is not saturated at 13.5M tokens (Sweep 1) — its own value is still declining at a clear, measurable rate exactly where the neural models are evaluated against it. Four *architecturally distinct* neural families (a normalized linear recurrence, a leaky-integrator recurrence, a fixed multi-timescale SSM, and a content-addressable read) converging on the *same* ~−0.125 margin *at the same token budget* is consistent with a shared architectural ceiling (the record's standing interpretation) but **equally consistent with all of them, trigram included, being commonly data-limited at this budget** — a hypothesis the record had ruled out on the strength of a saturation claim that does not hold at this scale (see Sweep 1's regime-boundary citation issue). This does not prove more tokens would close the margin (that requires re-running a neural model at a larger, controlled sentence-mode budget, which this CPU-only trigram sweep does not do) — it reopens the tokens lever as a live, untested question rather than an exhausted one, and it means the −0.125 headline number should not be read as "the model's absolute ceiling relative to a settled baseline." <!--derived-->

BPE tokenization is not making the baseline artificially strong (Sweep 3) — if anything it is a legitimately efficient, closer-to-fair representation than the alternative (char-level) that would make the metric's raw numbers look easier to beat. The one real, cheap tokenization lever is BPE vocab size, untested against the model side.

## Cheap next levers (not run here, in scope for a follow-up)

1. **Controlled sentence-mode data-scaling on the trigram, extended further** (this runner already supports it): confirm whether the ~0.02 nats/M-token marginal rate at 13.5M keeps declining smoothly or starts flattening past ~20M sentence-mode tokens (the sentence-filtered pool caps near there; beyond that requires `--contiguous`, which changes regime — see Sweep 2's caveat).
2. **Re-run a neural arm (e.g. `--recurrence hippo`, cheapest of the converged family) at V=16000** vs the production V=8001, same token budget, to test whether the model's own NLL degrades by less than the trigram's +0.49-nats easier bar — a direct, cheap (one extra BPE tokenizer + one extra GPU run) test of whether vocab size is a real lever on `margin_vs_trigram`, not just on the trigram side alone.
3. **Complete the queued 6-seed `_emerge_wkv_lm_contiguous40_6seed.json`** (currently a 1-line log) before treating the single-seed "+0.02" contiguous-mode crossing as established — Sweep 2 shows the contiguous regime's own trigram baseline is weaker, so part of any crossing needs to be attributed to that, not solely to the model.

## External grounding (recorded to `research/queue/.external_searches.jsonl`, lane `language (own-voice mouth / retire the Qwen scaffold)`)

<!--derived-->
- Krishnan, Alabi, Klakow (2023), *"On the N-gram Approximation of Pre-trained Language Models,"* Interspeech 2023, arXiv:2306.06892 — interpolating with a large sampled corpus improves test perplexity over a baseline trigram by 15%; corroborates that a FAIR interpolated trigram (`fit_interp_trigram`'s deleted-interpolation method, exactly reused here) is a genuinely strong, non-strawman baseline even against PLM-derived text, consistent with this sweep's finding that the trigram is a live, still-improving-with-data baseline rather than a fixed weak target.
- Mielke, *"Can you compare perplexity across different segmentations?"*, https://sjmielke.com/comparing-perplexities.htm (see also Mielke & Eisner 2019, arXiv:1904.02879) — the direct grounding for Sweep 3's nats/word normalization and the char-level-inversion finding.

## Reproduce

```bash
# verify the replica against the crux artifact (seed 42, ~7 min: mostly one-time BPE tokenization, no disk cache):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._trigram_baseline_characterization_derisk --verify

# the full sweep that produced this finding's numbers (~42 min wall-clock, CPU-only, 8 worker processes,
# fork-shared tokenized corpora — see the runner's own docstring for the design):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._trigram_baseline_characterization_derisk \
    --run-all --seeds 42 43 44 --contiguous-seeds 42 \
    --budgets 2000000 5000000 9500000 --n-workers 8 \
    --json research/findings/raw/_trigram_baseline_characterization.json
```
