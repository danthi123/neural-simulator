---
type: finding
status: contributing
date: 2026-09-03
mechanism: held-out token-coverage + teacher-forced perplexity measurement of the deployed `--recurrence linattn` BPE checkpoint (bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{42,43,44,100,101,102}.npz) + its shipped BPE tokenizer (bridges/wkv_ckpt/wkv_bpe8k.json), against the 124-utterance realistic-chat probe corpus, to set the un-measured `BRAIN_WKV_MOUTH_SCOPE=broad` coverage threshold
lane: language (own-voice mouth / one-brain-mouth-integration roadmap, de-risk #1)
seeds: [42, 43, 44, 100, 101, 102]
verdict: MEASURED, not yet a gate flip. Production-faithful ("asfed") token coverage is 71.67%; content-word vocabulary coverage (the genuine topic-coverage signal, case/punctuation-normalized) is 100% hard-OOV-free with 77.9% of content words represented as a single whole BPE piece. The BIGGER finding: a case-folding bug (the BPE tokenizer's merge table was trained exclusively on lowercase text, so every capital letter maps to `<UNK>`) — not genuine vocabulary/topic coverage — is the dominant driver of asfed's poor score, worth ~5.6x in teacher-forced perplexity on its own (12827 to 2284, <!--derived-->), independent of whatever SCOPE threshold is chosen. Recommend `whole_word_frac>=0.6` as the coverage cutoff (93.55% of the probe still served) pending that fix, with the full tradeoff curve reported.
artifacts:
  - research/findings/raw/_wkv_mouth_linattn_broad_scope_coverage.json
  - research/runners/_wkv_mouth_linattn_broad_scope_coverage_derisk.py
---

# How much does the linattn own-voice mouth actually cover a real conversational prompt?

**One-brain-wiring de-risk #1** (per the roadmap `research/findings/2026-09-03-one-brain-mouth-integration-
ROADMAP.md`, branch `research/one-brain-mouth-roadmap` / `f943cc28`, not yet merged into `main` as of this
writing — §3 item 1: *"Measure the linattn BPE checkpoint's held-out coverage → set the
`BRAIN_WKV_MOUTH_SCOPE=broad` threshold... Unblocks Stage 1"*). Pure measurement, CPU/numpy only, no `sim/`
edit, no `webapp/` edit — this doc does not flip any default.

## The gate this measures

`webapp/wkv_mouth_generator.py::in_vocab_scope` gates whether a conversational prompt is served by the
from-scratch spiking WKV mouth or falls back to the Qwen2.5-0.5B scaffold (`webapp/open_ended_chat.py`
`answer_turn`, lines ~578-603). Its default ("vocab") mode is a hard word-overlap check against the
CLOSED V=1000 TinyStories checkpoint — not meaningful for the linattn BPE checkpoint (a general subword
vocabulary has essentially no OOV at the character level). `BRAIN_WKV_MOUTH_SCOPE=broad` — the mode the
live 6/6-trigram-crossing verification actually ran under (`2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-...md`) —
bypasses the check entirely and **admits every prompt unconditionally**. That module's own comment names this
a placeholder: *"set from the 6-seed's own held-out coverage, NOT guessed here."* This is that measurement.

## Method

**Probe corpus — reused verbatim, nothing invented.** `research.runners._wkv_mouth_chat_topic_vocab_coverage_
derisk._build_probe_corpus` (an earlier rung's own 124-utterance realistic-chat probe, already used against the
closed-vocab checkpoint): 14 Turing-test conversational-register turns ("Hi there! How are you doing today?"),
10 "Tell me about &lt;famous everyday topic&gt;" queries, and 100 seeded "Tell me about &lt;live
`wikidata_core_15k` agent&gt;" queries — i.e. genuinely production-shaped chat, not a fresh ad hoc sample.
Reusing it makes this measurement directly comparable to the earlier finding's numbers on the OLD checkpoint.

**Checkpoint + tokenizer.** All 6 non-negotiable seeds of `bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_
seed{seed}.npz` (loaded via the production `LinAttnReadout`, `research.runners._wkv_fewspike_read_derisk`) and
the shipped `bridges/wkv_ckpt/wkv_bpe8k.json` (the SAME file `webapp.wkv_mouth_generator._get_bpe_tokenizer`
loads by default). Every seed's checkpoint reports the identical `V=8001` vocabulary (`<!--derived-->`
verified in the artifact's `vocab_size_seed_invariant: true`), confirming coverage is a tokenizer property,
not a per-seed accident. `d_model=192`, depth-2.

**Two tokenizations, scored separately** (why this isn't a rerun of the existing subword-coverage runner,
whose own `_bpe_row` lowercases before encoding — a cleaner input than production code actually uses):

1. **`asfed`** — EXACTLY what `_free_gen`/`_free_gen_linattn` do to a raw prompt: `bpe.encode(prompt)`, no
   lowercasing, no punctuation stripping. The production-faithful number.
2. **`content_word`** — the earlier findings' methodology: lowercased, `_WORD_RE`-extracted, minus
   `_FUNCTION_WORDS`, each content word BPE-encoded independently. Isolates genuine topic/vocabulary coverage
   from the case/punctuation artifact `asfed` also carries.

Plus **teacher-forced perplexity** under the checkpoint's own next-token distribution (`LinAttnReadout.advance`
/`.logits`, pure matrix ops — NOT the few-spike WTA sampler under test elsewhere), scored per seed, as a genuine
model-confidence signal independent of vocabulary coverage.

## Result 1 — token coverage and OOV rate (the headline numbers)

<!--derived-->
From `research/findings/raw/_wkv_mouth_linattn_broad_scope_coverage.json`, `coverage.overall` (124 utterances,
tokenizer-only — seed-invariant by construction):

| view | token coverage | OOV/hard-OOV rate | note |
|---|---|---|---|
| **asfed** (production-real: raw case + punctuation, exactly what `_free_gen*` feeds the tokenizer) | 71.67% | 28.33% | dominated by capital letters and terminal punctuation, see Result 2 |
| **lowercased** (case-fold-fix counterfactual; punctuation still flows through) | 89.99% | 10.01% | isolates the case bug specifically |
| **content-word** (lowercased, function-words excluded — the genuine vocabulary-coverage view, comparable to the 2026-09-02 closed-vocab finding's methodology) | 100.0% | 0.0% hard-OOV | matches the prior near-0%-hard-OOV baseline for BPE; confirms no regression |

<!--derived-->
Within the content-word view, 77.9% of content words BPE-encode to exactly ONE known piece (`whole_word_frac`,
mean 1.505 pieces/content-word overall) — i.e. representable but with a real, non-zero fragmentation cost, not
a binary in/out gate. By probe group: `wikidata_known_agents` (n=100, the actual production-shaped "ask about a
real store topic" traffic) scores lowest at 76.78% whole-word / 69.93% asfed coverage; `everyday_real_world_
topics` (n=10, hand-picked famous single-word nouns) scores highest at 87.5% whole-word.

**The "prompt-serve fraction" curve** (fraction of the 124 probe prompts whose `whole_word_frac` clears a given
cutoff — the load-bearing number for a coverage-gated `SCOPE=broad`, from `coverage.overall.serve_fraction_at_
cutoff`): 99.19% at >=0.5, 93.55% at >=0.6, 68.55% at >=0.7, 61.29% at >=0.8, 17.74% at >=0.9 (== at ==1.0 — no
prompt scores in (0.9, 1.0), an artifact of short prompts having few, coarsely-quantized content words).

## Result 2 — the bigger finding: a case-folding bug dominates production degradation, not topic mismatch

`sim.bpe_tokenizer.BPETokenizer`'s merge table was trained exclusively on lowercase `[a-z']+` corpus text
(`_train_bpe_bounded`'s `raw.lower()`), and the checkpoint's own LM training corpus (`data/corpus/
simplewiki.txt`) is **itself pre-lowercased on disk** — verified by reading the raw file. So the model never
saw a capital letter during training at all. `webapp.wkv_mouth_generator._free_gen`/`_free_gen_linattn` encode
the raw prompt with no `.lower()` call (`pid = bpe.encode(prompt or "")`), so every capital letter in a real
sentence — every sentence-initial word, every proper noun — becomes a character outside the trained alphabet
and maps straight to the tokenizer's `<UNK>` id 0. Directly observed on 3 real probe prompts:

```
'Hi there! How are you doing today?'  -> 4/16 tokens UNK; decodes to "i there ow are you doing today"
'Tell me about paris.'                -> 2/8 tokens UNK;  decodes to "ell me about paris"
'Tell me about Ac Le Havre.'          -> 5/13 tokens UNK; decodes to "ell me about c e avre"
```

Every capitalized word loses its first letter to `<UNK>` (silently dropped by `BPETokenizer.decode`, so it
never surfaces as visible garbage in generated OUTPUT — but it DOES corrupt what the model reads as PROMPT
CONTEXT, since the corrupted token still gets consumed by `advance()`). Proper-noun-heavy prompts — i.e.
exactly the `wikidata_known_agents` group, the closest analog to real "ask the brain about a topic" traffic —
are hit hardest, since every capitalized word in a multi-word entity name loses a letter.

<!--derived-->
**Teacher-forced perplexity confirms this is the dominant term, not a minor artifact**, cross-referenced
against a fixed in-domain reference sample (20 genuine `data/corpus/simplewiki.txt` sentences, the checkpoint's
own training register) — all three numbers from `research/findings/raw/_wkv_mouth_linattn_broad_scope_
coverage.json`'s `case_fold_impact`, averaged across all 6 seeds and the full 124-utterance probe:

| condition | mean perplexity (V=8001 vocab; uniform-guess ceiling = 8001) |
|---|---|
| **asfed** (production-real) | 12827.32 |
| **lowercased** (case-fold-fix counterfactual) | 2283.64 |
| **in-domain reference** (genuine simplewiki sentences, the model's own training register) | 938.34 |

<!--derived--> Fixing just the case fold (a one-line `.lower()` at the two `_free_gen`/`_free_gen_linattn`
prompt-encode call sites — NOT applied in this measurement-only rung) would cut mean perplexity by roughly
5.6x on its own. A further ~2.4x gap remains between the lowercased chat probe and the in-domain reference,
reflecting a genuine (much smaller) register mismatch between conversational "Tell me about X" phrasing and
the model's Simple-English-Wikipedia encyclopedic training distribution — real, but secondary to the case bug.

**Side observation, not this rung's focus:** every checkpoint's `unk_idx` auto-detection (`WKVReadout`/
`LinAttnReadout.__init__`: `unk_idx = len(words)-1 if words[-1]=="<unk>" else -1`) reads `-1` for all 6 seeds
(`teacher_forced_by_seed.*.checkpoint_unk_idx_detected`) because this checkpoint's real sentinel is `<UNK>`
(uppercase) at position **0**, not lowercase `<unk>` at the last position — so the generation-time UNK
suppression (`if ro.unk_idx >= 0: lg[ro.unk_idx] = -1e30`) never fires for this checkpoint family. `bpe.decode`
already drops `<UNK>` from rendered text, so this does not currently produce visible garbage, but it is a
related, easily-fixed gap in the same code path.

## Result 3 — coverage vs. model confidence: an honest, non-monotonic result

<!--derived-->
Bucketing the probe by `whole_word_frac` and looking at cross-seed mean **lowercased** perplexity (the
case-artifact-free view, so this correlation isn't just re-measuring Result 2) does **not** show a clean
monotonic "higher coverage -> lower perplexity" relationship (`coverage_vs_confidence_buckets` in the cited
artifact):

| whole_word_frac bucket | n | mean lowercased ppl (6-seed) | median |
|---|---|---|---|
| [1.0, 1.0] (fully covered) | 22 | 3874.16 | 2909.65 |
| [0.7, 0.9) | 63 | 2239.86 | 1338.98 |
| [0.5, 0.7) | 38 | 1465.55 | 1098.04 |
| [0.0, 0.5) | 1 | 1138.63 | 1138.63 |

The fully-covered bucket has the *highest* perplexity, not the lowest. The most likely explanation (per-group
breakdown in the artifact's `per_group_lowercased_ppl`): the fully-covered bucket is dominated by SHORT, 1-2-
content-word queries (`everyday_real_world_topics`: "Tell me about jupiter.") whose per-prompt perplexity is a
high-variance geometric mean over very few autoregressive steps — one hard transition (predicting whatever
follows the out-of-training-register "tell me about" template) dominates the whole estimate with no dilution
from other tokens. The larger, more naturally-varied `wikidata_known_agents` group (n=100, lower per-word
coverage on average) shows a LOWER mean perplexity (2148.39) than the smaller `everyday_real_world_topics`
group (2890.62) despite using the identical sentence template, consistent with a length/variance confound
rather than coverage itself being anti-predictive. **Reported honestly rather than forced into a clean
narrative** — this measurement does not, on its own, prove `whole_word_frac` is a strong per-prompt confidence
predictor; it establishes that vocabulary/token coverage is a real, reproducible, well-behaved (monotonic by
construction) quantity, while perplexity at this small-prompt scale is a noisier, secondary signal.

## Recommendation

**Do the case-fold fix regardless of whatever SCOPE threshold is chosen.** It is a ~1-line change (lowercase
the prompt before `bpe.encode()` at the two `_free_gen`/`_free_gen_linattn` call sites in
`webapp/wkv_mouth_generator.py`) with a far larger, more certain effect (~5.6x perplexity, <!--derived-->) than
any defensible coverage-cutoff choice — this measurement did not apply it (pure-measurement scope), so it is
named as a follow-up, not banked as done. (Spawned as a separate task — see below.)

**Recommended `BRAIN_WKV_MOUTH_SCOPE=broad` coverage threshold: `whole_word_frac >= 0.6`** (fraction of a
prompt's content words that BPE-encode to a single known piece), replacing today's unconditional admit.
Rationale and tradeoff:

- At **0.6**: 93.55% of the realistic probe is still served by the own-voice mouth (116/124) — preserves the
  project's own-voice-over-Qwen push for the large majority of realistic traffic — while excluding the tail
  whose MAJORITY of content words are not even singly-represented in the checkpoint's vocabulary (a genuine,
  if imperfect, higher-fabrication-risk signal, consistent with the design doc's original "coverage/confidence
  decision" framing).
- **Lower** (e.g. 0.5, 99.19% served): closer to today's admit-everything behavior — more own-voice coverage,
  more exposure to the genuinely under-covered tail.
- **Higher** (e.g. 0.7, 68.55% served; 0.8, 61.29% served): meaningfully more conservative (more Qwen
  fallback, less fabrication surface from poorly-known topics) at real cost to how much traffic the own-voice
  mouth actually gets to serve — undercutting the point of shipping it. 0.9+ serves under 18% of the probe and
  is not a serious candidate.
- **Epistemic honesty on this recommendation:** Result 3 means this cutoff is NOT validated against a clean
  confidence/quality gradient at the individual-prompt level in this measurement — it is a principled
  vocabulary-coverage cutoff, not a proven fabrication-risk gate. The roadmap's own Stage-1 GO gate (fluent +
  moat-honest + Qwen-call-rate -> ~0 on in-coverage prompts) is the correct place to validate whether THIS
  cutoff, once wired, actually holds up on live generation — that live check is out of scope for this
  measurement-only rung and is the natural next step this de-risk unblocks.

## Honest residuals

- This is a **tokenizer/vocabulary coverage + teacher-forced-confidence** measurement, not a live generation-
  quality (fluency/grounding/honesty) evaluation — the roadmap's own Step-0 GO gate and Stage-1 GO gate are
  the place for that, separately.
- The recommended cutoff is not wired into `in_vocab_scope`/`scope_mode` here (no `webapp/` edit in this rung,
  by design — pure measurement).
- The in-domain reference sample (20 sentences) is illustrative context for this small model's own baseline
  confidence, not a reproduction of the training run's own held-out split (`_emerge_wkv_lm_derisk`'s internal
  `--max-train-sents`/`--max-eval-sents` boundary is not replicated here).
- The `unk_idx` detection gap (Result 2's side observation) is named, not fixed, in this rung.

## Reproduce

```bash
SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m research.runners._wkv_mouth_linattn_broad_scope_coverage_derisk \
    --out research/findings/raw/_wkv_mouth_linattn_broad_scope_coverage.json
```

Runs in well under a minute on CPU (6 checkpoints x ~16 MB + a 124-utterance probe; no GPU, no
`SimulationBridge`, no torch — `LinAttnReadout` is pure `np.load` + numpy matrix ops).
