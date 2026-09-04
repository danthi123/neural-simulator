---
type: finding
status: positive
date: 2026-09-04
mechanism: BPE-caps fix for the linattn own-voice WKV mouth (webapp/wkv_mouth_generator.py) -- lowercase the
  prompt before BPE-encoding it (INPUT half, `_bpe_encode_prompt`/`bpe_lowercase_enabled`) plus a lightweight
  sentence-initial/pronoun-"I"/known-name truecasing heuristic on the mouth's generated text (OUTPUT half,
  `_truecase`/`truecase_enabled`), both independently guarded, default-ON, verified through the REAL production
  code path (not a reimplementation) across all 6 non-negotiable seeds
lane: language (own-voice mouth / one-brain-mouth-integration roadmap) -- closes the case-fold-fix follow-up
  named by research/findings/2026-09-03-linattn-mouth-broad-scope-coverage-threshold.md's Recommendation
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO, 6 seeds (42/43/44/100/101/102). The INPUT fix recovers the FULL ~5.6x teacher-forced-perplexity gap
  the coverage de-risk measured -- mean asfed ppl 12827.32 -> 2283.64, an EXACT match (<!--derived-->) to that
  finding's own cited `lowercased` counterfactual, confirmed byte-identical to the fixed production ids on all
  124 probe utterances x 6 seeds in BOTH the fix-ON and fix-OFF direction (id-for-id `==`, not inferred). The
  OUTPUT fix restores readable capitalization on real `generate()` output (genuine few-spike spiking read, not
  a stub) without perturbing `in_vocab_scope`/`fact_grounding_ids` (the moat/fact-routing gates the caller also
  uses) under any combination of the two new flags. Both fixes are additive, independently guarded, and
  byte-identical to the pre-fix code the instant their flag is off -- verified in the data, not assumed.
artifacts:
  - research/findings/raw/_wkv_mouth_bpe_caps_fix_verify.json
  - research/findings/raw/_wkv_mouth_bpe_caps_fix_verify.json.prov.json
  - research/findings/raw/_wkv_mouth_linattn_broad_scope_coverage.json
---

# Fixing the BPE-caps bug the broad-scope coverage de-risk found

**One-brain-wiring de-risk #1's own named follow-up.** `research/findings/2026-09-03-linattn-mouth-broad-scope-
coverage-threshold.md` measured the deployed `--recurrence linattn` BPE checkpoint's held-out coverage and found
a case-folding bug -- not topic mismatch -- was the DOMINANT driver of poor production ("asfed") coverage: mean
teacher-forced perplexity 12827.32 as-fed vs 2283.64 once capitals are lowercased, a ~5.6x hit. That finding was
pure measurement (no `webapp/` edit); this rung is the fix it named as the natural next step, plus the paired
output-side readability gap the same root cause implies.

## The problem (recap, see the cited finding for the full derivation)

`sim.bpe_tokenizer.BPETokenizer`'s merge table (`bridges/wkv_ckpt/wkv_bpe8k.json`) was trained exclusively on
lowercase text -- the trainer's own regex plus the fact `data/corpus/simplewiki.txt` is pre-lowercased on disk
-- so the checkpoint never saw a capital letter during training. `webapp/wkv_mouth_generator.py`'s `_free_gen`/
`_free_gen_linattn` encoded a raw, un-lowercased prompt (`bpe.encode(prompt or "")`), so every capital letter in
real chat -- every sentence-initial word, every proper noun -- fell outside the trained alphabet and BPE-encoded
to `<UNK>`. Separately (and independently confirmed in this rung, see below), the SAME lowercase-only training
regime applies to the OTHER checkpoint family too: `bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz`'s
word-level vocabulary carries zero uppercase characters across all 1000 entries. Both checkpoint families
therefore structurally CANNOT emit a capital letter on the way out, regardless of which is loaded.

## The fix

### 1. INPUT -- lowercase the prompt before BPE-encoding it

`webapp/wkv_mouth_generator.py` adds `_bpe_encode_prompt(bpe, prompt)`, the single call site both `_free_gen`
and `_free_gen_linattn` now use in place of their old direct `bpe.encode(prompt or "")`:

```python
def _bpe_encode_prompt(bpe, prompt: str) -> list:
    text = prompt or ""
    if bpe_lowercase_enabled():
        text = text.lower()
    return bpe.encode(text)
```

`bpe_lowercase_enabled()` (`BRAIN_WKV_MOUTH_BPE_LOWERCASE`, default ON) mirrors the file's own established
"closes a measured gap, not an opt-in trial" pattern (`_LEARNED_HEAD_DEFAULT_ON`,
`wkv_mouth_affect_enabled`) -- `=0` reverts to the exact pre-fix raw-case encode. Both call sites (`_free_gen`
line ~791, `_free_gen_linattn` line ~857) are reached ONLY inside their existing `if bpe is not None:` branch,
which itself only executes when `tokenizer_mode() == "bpe"` (`BRAIN_WKV_MOUTH_TOKENIZER=bpe`, still default
`"word"`) -- so this fix is scoped EXCLUSIVELY to the BPE mouth path; the word-tokenizer `else` branch (its
own `.lower()`-normalized `_WORD_RE` lookup) is untouched, pinned by a dedicated test
(`test_word_level_path_never_reads_this_flag`, see Test coverage below).

### 2. OUTPUT -- truecase the generated text

A new block adds `_truecase(text)` (sentence-initial capitalization on `.`/`!`/`?`-delimited segments, the
standalone pronoun "i"/"i'm"/"i've"/"i'll"/"i'd", and a small, explicit, hand-curated known-name allowlist
`_KNOWN_CAPITALIZED_WORDS` -- TinyStories character names + "mr"/"mrs"/"dr", deliberately EXCLUDING candidates
that collide with an ordinary English word in the same vocabulary, e.g. "will"/"rose"/"mark"/"hope"/"joy"),
applied to `generate()`'s final return text regardless of which internal path produced it (free-gen OR
`render_fact_sentence`):

```python
text = _RNG.run(seed, _run)
if truecase_enabled():
    text = _truecase(text)
return text, round(time.time() - t0, 3)
```

`truecase_enabled()` (`BRAIN_WKV_MOUTH_TRUECASE`, default ON) is pure string post-processing on the
ALREADY-CHOSEN word sequence -- it never touches token selection or the genuine few-spike spiking read
(`reader.read(p)`); `=0` reverts to the checkpoint's raw all-lowercase text, byte-identical to before this
function existed. This is explicitly a lightweight, bounded, host articulation scaffold, not a general NER --
`_KNOWN_CAPITALIZED_WORDS`'s own comment documents the excluded ambiguous names, and `_truecase` never fights a
proper noun `render_fact_sentence`'s own `slug_to_np` already capitalized correctly (verified idempotent, see
Test coverage).

## Verification

Both fixes were verified against the REAL production code path -- `webapp.wkv_mouth_generator`'s own functions,
not a hand-copied reimplementation -- by a new runner,
`research/runners/_wkv_mouth_bpe_caps_fix_verify_derisk.py`, run CPU-only
(`CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy`), peak RSS **367.2 MB** (well under the 4 GB budget).

### A/B -- input recovery + byte-identical-off, all 124 probe utterances x 6 seeds

<!--derived-->
From `research/findings/raw/_wkv_mouth_bpe_caps_fix_verify.json`
(`part_A_B_input_recovery_and_byte_identical_off`), reusing the SAME 124-utterance probe corpus and
`LinAttnReadout` teacher-forced-perplexity methodology the cited coverage finding used:

| check | result |
|---|---|
| fix ON (default) ids == the cited finding's own `lowercased` counterfactual, every one of 124 utterances | 124/124 match (`all_match: true`) |
| fix OFF ids == the cited finding's own raw pre-fix `asfed` ids, every one of 124 utterances | 124/124 match (`all_match: true`) |
| mean teacher-forced ppl, fix ON (production `asfed` NOW) | **2283.64** |
| mean teacher-forced ppl, fix OFF (pre-fix raw `asfed`) | **12827.32** |
| recovery ratio | **5.617x** |
| cross-check vs the cited artifact's `mean_asfed_ppl_across_probe_and_seeds` (12827.32) / `mean_lowercased_ppl_across_probe_and_seeds` (2283.64) | EXACT match to 2 decimal places (same checkpoints, same probe, same methodology) |

Per-seed fix-ON / fix-OFF mean ppl (all 6 non-negotiable seeds):

| seed | fix ON | fix OFF |
|---|---|---|
| 42 | 2080.04 | 10487.62 |
| 43 | 2293.34 | 16375.71 |
| 44 | 2368.87 | 10814.67 |
| 100 | 2508.04 | 13986.04 |
| 101 | 2340.82 | 11790.60 |
| 102 | 2110.76 | 13509.25 |

The exact match between this run's from-scratch recomputation and the cited finding's own numbers is expected
and load-bearing: once the fix is active, `_bpe_encode_prompt`'s output is BY CONSTRUCTION identical to what
that finding's own `_lowercased_ids` counterfactual computed, over the identical checkpoints and probe -- not an
approximation of the earlier result, the SAME computation reached through the real code path.

### C -- real `generate()` output + moat/fact-routing regression

<!--derived-->
Four demo prompts, real `LinAttnReadout` + `FewSpikeWordRead` (genuine few-spike Izhikevich spiking read, 512
neurons, `connections_per_neuron=0`), `max_new_tokens=20`, seed 42, both fixes ON (default) vs both OFF:

```
"Hi there! How are you doing today?"
  ON : Hi there how are you doing today and the first time in the world war I was the first time in the world
  OFF: i there ow are you doing today and the first person to be a member of the world war i was the first person

"Tell me about Paris."
  ON : Tell me about paris and was a member of the new york city from to he was the th cong
  OFF: ell me about aris is a city in the u s state of south carolina it is the county seat of the city of

"What do you know about the United Kingdom?"
  ON : What do you know about the united kingdom is a kind of people who are called the people who are called...
  OFF: hat do you know about the nited ingdom and the other women in the world the first women to be the first...
```

The fix-OFF text reproduces, verbatim in shape, the exact corruption pattern the cited finding hand-verified
("Hi" -> "i", "How" -> "ow", "Paris" -> "aris", "United" -> "nited", "Kingdom" -> "ingdom" -- one leading letter
dropped per capitalized word) -- direct, independent confirmation that the byte-identical-off path is genuinely
the SAME bug the coverage finding measured, and that fix-ON genuinely repairs it, not merely relabels it.

`n_fix_on_produced_a_capital: 4/4` prompts; `n_fix_off_produced_a_capital: 0/4` -- a STRUCTURAL check, not a
style preference: neither checkpoint vocabulary contains an uppercase symbol at all (0/1000 word-level entries,
0/8001 BPE symbols), so fix-OFF producing zero capitals is not a coincidence of this sample, it is the only
possible outcome.

**Moat / fact-routing regression** (`in_vocab_scope`, `fact_grounding_ids` -- the same gates
`webapp/open_ended_chat.py::answer_turn` calls): run under all 4 combinations of the two new flags
(on/on, off/off, on/off, off/on) against the same probe message + sample facts. Result: `identical_regardless_
of_new_flags: true` -- neither function's output moved by so much as one character across any flag combination
(expected: neither function references `BRAIN_WKV_MOUTH_BPE_LOWERCASE`/`BRAIN_WKV_MOUTH_TRUECASE` at all; this
is an empirical pin of that fact, not an inference from reading the code).

## Guards (byte-identical when off)

- `BRAIN_WKV_MOUTH_BPE_LOWERCASE=0` -> `_bpe_encode_prompt` reverts to `bpe.encode(prompt or "")`, byte-identical
  id-for-id to the pre-fix code on every one of the 124 probe utterances (verified above, and pinned by
  `tests/test_wkv_mouth_bpe_caps_fix.py::TestBpeLowercaseInputFix::test_fix_off_is_byte_identical_to_raw_encode`).
- `BRAIN_WKV_MOUTH_TRUECASE=0` -> `generate()`'s return text is the checkpoint's raw output, unchanged --
  structurally verified (zero uppercase possible either way) and pinned by
  `tests/test_wkv_mouth_bpe_caps_fix.py::TestTruecaseOutputFix::test_generate_return_carries_zero_uppercase_when_flag_off`.
- Both fixes are reached only when `tokenizer_mode() == "bpe"` / `BRAIN_OPEN_ENDED_WKV_MOUTH` are themselves
  already opted into (both still default-OFF) -- so today's actual shipped default conversational path has
  ZERO exposure to either flag's default value, on OR off.
- `tests/test_wkv_mouth_bpe_decode_wiring.py::test_end_to_end_generate_in_bpe_mode` (an existing regression pin
  for BPE round-trip decode wiring, predating this rung) now explicitly sets `BRAIN_WKV_MOUTH_TRUECASE=0` --
  its own purpose is encode/decode round-tripping, not casing, and it now passes with the SAME exact-case
  assertion it always had, proving the OFF guard restores its original text byte-for-byte.

## Test coverage

`tests/test_wkv_mouth_bpe_caps_fix.py` (25 tests, fast/pure-Python -- no `SimulationBridge`, no real checkpoint):
default-on/explicit-off/explicit-on for both flags; `_bpe_encode_prompt` recovers full coverage and is
byte-identical off; `_truecase`'s sentence-initial/pronoun/known-name/idempotence/empty-string behavior; the
word-level path's static exclusion from `_bpe_encode_prompt`; the deliberate exclusion of ambiguous names
(`will`/`rose`/`mark`/`hope`/`joy`) pinned so a future edit cannot silently re-add one unnoticed. All 25 pass;
`tests/test_wkv_mouth_bpe_decode_wiring.py` (37 tests) and `tests/test_wkv_invocab_scope_leadin_fix.py` (14
tests) pass unmodified in behavior (one line added to the former's `clean_env` fixture + one explicit
`TRUECASE=0` setenv, both scoping-only, no assertion weakened).

**Pre-existing, unrelated test-suite issues found while verifying (NOT caused by this fix, NOT fixed here):**
running `tests/test_wkv_mouth_bpe_decode_wiring.py` before `tests/test_wkv_invocab_scope_leadin_fix.py` in the
same pytest process makes 2 of the latter's tests fail (`test_genuine_multiword_content_still_passes_with[out]_
a_leadin`) -- reproduced byte-for-byte identically on the UNMODIFIED baseline (stashed this rung's changes and
re-ran the exact same command), so this is a pre-existing, collection-order-dependent module-level-cache
(`_CKPT_CACHE`) pollution issue, not a regression. Two further pre-existing, unrelated failures were also
confirmed against baseline: `tests/test_grounded_wkv_renderer.py::test_fluidchat_wkv_grounded_and_gatefirst_
moat` and `tests/test_wkv_readout_multilayer.py::TestMultiLayerNumericalCorrectness::test_state_dict_key_
layout_matches_documented_contract` (a state-dict key-naming mismatch, `extra.*` vs `extra_ssm.*`, unrelated to
casing). Flagged as a follow-up task, not addressed in this rung (out of scope for a BPE-caps fix).

## Honest residuals

- **Punctuation still flows through unfixed.** The cited finding's own `lowercased` counterfactual (89.99% token
  coverage, not 100%) already named this: a trailing `.`/`!`/`?` is itself a character outside the trained
  lowercase-letters-and-apostrophe alphabet, so it still maps to `<UNK>` regardless of this fix. Structurally,
  this also means neither checkpoint can currently GENERATE punctuation at all (confirmed: `<UNK>` symbols are
  dropped, not rendered, by `BPETokenizer.decode`) -- `_truecase`'s multi-sentence splitting is therefore
  currently a no-op in practice (there is nothing to split on), written generally rather than assuming that
  stays true forever.
- **`_truecase`'s proper-noun coverage is a small, bounded, TinyStories-specific allowlist, not general NER.**
  It will rarely fire on the BPE/simplewiki checkpoint family's very different vocabulary (place names, historical
  figures, etc.) -- a cross-domain gazetteer (e.g. sourced from the live `wikidata_core_15k` store's own agent
  names) was considered and NOT built here; named as the natural next lever, not attempted in this lightweight
  pass.
- **The deeper fix is a retrain with a CASED BPE vocabulary** (train `sim.bpe_tokenizer.BPETokenizer` on
  case-preserving text instead of `raw.lower()`-folded text) -- that would let the model itself learn which
  words are conventionally capitalized (proper nouns, sentence starts) rather than restoring casing with a
  host-side heuristic after the fact. Out of scope here (this rung is explicitly a decode-time/host
  articulation-scaffold fix, additive and non-retraining, per the task's own framing) -- named as the honest
  next rung, not a silent ceiling.
- **The 4 pre-existing test-suite issues named above** (test-isolation cache pollution + 2 unrelated failures)
  are real, reproduced bugs, left open for a separate task.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python \
    -m research.runners._wkv_mouth_bpe_caps_fix_verify_derisk \
    --out research/findings/raw/_wkv_mouth_bpe_caps_fix_verify.json

CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m pytest \
    tests/test_wkv_mouth_bpe_caps_fix.py tests/test_wkv_mouth_bpe_decode_wiring.py \
    tests/test_wkv_invocab_scope_leadin_fix.py -q
```

Runs in well under a minute on CPU (6 x ~small linattn checkpoints, a 124-utterance probe, and 8 small real
`generate()` calls on a 512-neuron spiking bank -- no GPU, no torch).
