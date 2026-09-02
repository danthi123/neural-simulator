---
status: design + measurement-complete
lane: e-mouth-fluency / board #99, #112, #199 (open-ended-mouth wire-in — the OOV-coverage surpass)
type: finding
date: 2026-09-02
verdict: >
  SUBWORD-CLOSES-OOV-REPRESENTABILITY (measured, CPU) + GPU-RETRAIN-QUEUED (design). A byte/char-level BPE
  subword tokenizer (the project's own sim.bpe_tokenizer, Sennrich-2016) drives the self-taught WKV mouth's
  chat-topic HARD-OOV from 39.88% (production V=1000 word-level, closed) to 0.0%, and full-utterance
  representability from 5.65% to 100.0%, on the SAME 124-utterance probe the vocab-coverage finding used --
  at a cost of ~1.6-2.0 subword pieces per content word (each piece = one spiking-WTA read; ~2.6-3.0 pieces
  for the proper-noun words the word vocab could not represent at all). The WKV/SSM cortex is
  tokenizer-AGNOSTIC by construction (the embedding/head are id-indexed; the recurrence never sees token
  identity), so a subword mouth needs a TRAINER tokenizer-swap + a DECODE detokenizer (both specified,
  neither an architecture change), then a GPU retrain (queued, not run -- GPU saturated). REPRESENTABILITY
  is proven here; FLUENCY (coherent subword-piece sequences) is the retrain's burden, gated below.
seed-waiver: measurement-only (a fixed-tokenizer vocabulary/coverage profile, not a stochastic capability
  GO/NO-GO -- the BPE merge table is deterministic given (corpus, vocab_size); there is nothing to seed on
  the coverage side. The FUTURE retrain IS 6-seed-gated, see the go/no-go in §6.)
instrument: research/runners/_subword_mouth_tokenizer_coverage_derisk.py -- the SAME 124-utterance probe
  (imported verbatim via _wkv_mouth_chat_topic_vocab_coverage_derisk._build_probe_corpus), the SAME
  content-word definition (WKV._WORD_RE minus WKV._FUNCTION_WORDS), the ACTUAL production word vocab
  (WKV._get_readout) as the baseline (verified to reproduce the finding: 39.88% OOV / 5.65% fully, matching
  its 39.9% / 5.65%), and the project's own sim.bpe_tokenizer.BPETokenizer as the subword candidate
  (reuse-by-import; NO sim/ edit; pure CPU -- merge-table training + coverage counting, no model training).
runner: research/runners/_subword_mouth_tokenizer_coverage_derisk.py
artifacts:
  - research/findings/raw/_subword_mouth_tokenizer_coverage.json (the coverage proof: word-vocab baseline vs
    BPE at V=4000/8000/16000 on TinyStories + wikitext, per group + overall)
  - research/findings/raw/_subword_mouth_tokenizer_coverage.json.prov.json (auto-stamped provenance)
  - bridges/wkv_ckpt/wkv_bpe8k.json (the recommended shipped tokenizer: BPE V=8000 on wikitext, coverage-
    verified 0.0% hard-OOV on the probe -- the --bpe-path the queued retrain in section 6 loads)
  - research/findings/2026-09-02-self-taught-mouth-vocab-coverage-for-chat-INSUFFICIENT.md (the finding this
    surpasses: named subword as its surpass #2)
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
    (the coupled FLUENCY lever: matched-quality token supply on the biologizable WKV cortex)
  - webapp/wkv_mouth_generator.py + research/runners/_wkv_fewspike_read_derisk.py (the mouth + the WKVReadout
    decode format -- both read here to establish tokenizer-agnosticism)
  - research/runners/_emerge_wkv_lm_derisk.py (the WKV/SSM training instrument -- the retrain touchpoint)
external: >
  Sennrich, Haddow & Birch 2016, "Neural Machine Translation of Rare Words with Subword Units", arXiv:1508.07909 <!--derived-->
  (ACL 2016, P16-1162) -- introduces open-vocabulary generation via subword (BPE) units; the exact algorithm
  sim.bpe_tokenizer implements. Radford et al. 2019 (GPT-2) byte-level BPE -- zero-OOV by construction (any
  string is a byte sequence). Recorded to research/queue/.external_searches.jsonl (lane e-mouth-fluency).
---

# A SUBWORD self-taught mouth closes the chat-topic OOV gap (0% hard-OOV, 100% representable, measured) — design + queued GPU retrain

**Artifact:** `research/findings/raw/_subword_mouth_tokenizer_coverage.json` (word-vocab baseline vs BPE, CPU).

## 0. The question and the verdict

`2026-09-02-self-taught-mouth-vocab-coverage-for-chat-INSUFFICIENT.md` measured the production V=1000
WORD-level, CLOSED-vocabulary WKV mouth against a 124-utterance chat-topic probe: **39.9% content-word OOV,
5.65% fully-in-vocab, 10.48% `in_vocab_scope` gate-pass** — a closed word vocab CANNOT partially express an
OOV word, so topic-naming turns collapsed to 0-2%. Its named surpass #2 was "a genuinely wider-vocabulary or
SUBWORD-capable checkpoint — an OOV word becomes spellable from subword pieces." This finding is the
verify-first design of that surpass plus the concrete pure-CPU de-risk of its central claim.

**Verdict:** subword tokenization closes the OOV **representability** gap completely and cheaply on the
measurement side (0.0% hard-OOV, 100% representable, ~1.6-2.0 pieces/word). The WKV cortex is
architecturally subword-ready. What remains is a GPU retrain (queued, not run) whose burden is subword
**fluency**, not representability. This is a design + coverage finding, NOT a capability claim: the mouth is
not wired, on-by-default, or scaffold-retired, and nothing here asserts the mouth "works" — see §7.

## 1. Verify-first: is the WKV cortex subword-compatible as-is? (YES architecturally; two small changes)

Read from the code, not assumed:

- **The WKV/SSM cortex is tokenizer-AGNOSTIC at the weight level.** `_emerge_wkv_lm_derisk.WKV` is
  `emb = nn.Embedding(V, D)` + a WKV/SSM recurrence over a `D`-dim hidden state + `head = nn.Linear(D, V)`.
  The recurrence (both the `wkv` linear-attention and the `ssm` dual-nonneg leaky-integrator branches)
  operates on `D`-dim vectors and **never sees token identity** — token ids only index the embedding on the
  way in and the head on the way out. `V` (vocab size) sets the embedding/head dimensions and nothing else.
  A subword vocabulary is therefore a drop-in for a word vocabulary; only `V` changes.
- **The decode format is tokenizer-agnostic too.** `WKVReadout` (`_wkv_fewspike_read_derisk.py`) loads
  `emb.weight / ln / Wv / Wr / Wo_sp / head.weight / head.bias / w / words` from the npz and reads `words`
  purely as the id→string table. The production few-spike Izhikevich soft-WTA (`FewSpikeWordRead.read`)
  reads a categorical winner over the top-k logits — it does not care whether the classes are words or
  subword pieces.
- **The production checkpoint's producing command is known** (verified from the npz: `w` is a scalar
  ⇒ uniform-decay; `Wo_sp` is 128→256 ⇒ dual-nonneg): `--recurrence ssm --dual-nonneg --uniform-decay
  --d-model 128 --vocab 1000 --save-ssm`. A subword retrain reuses this exact recurrence config.

**The two changes a subword mouth needs (neither is an architecture change):**

1. **TRAINER tokenizer-swap** in `_emerge_wkv_lm_derisk.py` (~15-25 lines, additive, default `word` ⇒
   byte-identical): a `--tokenizer {word,bpe}` + `--bpe-path` option; when `bpe`, build/load
   `sim.bpe_tokenizer.BPETokenizer` on the corpus, produce `tr_ids/ev_ids` via `tok.encode(passage_text)`,
   set `V = tok.vocab_size`, and in `--save-ssm` write `words = tok.vocab` (the symbol table). The
   trigram/bigram/PPMI controls operate on integer ids, so they remain valid *fair* baselines over subword
   ids unchanged.
2. **DECODE detokenizer** in `webapp/wkv_mouth_generator.py` (`_free_gen` currently ` " ".join`s
   `ro.words[i]`) + `WKVReadout`: concatenate the generated subword symbols and split on the `</w>`
   end-of-word marker — which is *exactly* what `BPETokenizer.decode` already does, so this is reuse, not
   new logic. With subword decode, `in_vocab_scope`'s hard-OOV REJECT purpose dissolves (every prompt is now
   representable); it becomes a fluency/confidence gate, not a hard gate.

**Is the existing V=4000 word checkpoint a useful interim, or a dead-end?** Confirmed **word-level**: its
4002 entries are whole TinyStories-domain words (`grandma`, `daisy`, `owl`), no `</w>` markers anywhere. It
is a *wider word vocab*, so it still cannot spell an OOV word — the finding measured it lifting gate-pass
only 10.48%→17.74% while `everyday_real_world_topics` stayed at exactly 0%. So V4000 is a **zero-build
interim mitigation, and a dead-end for the OOV CAPABILITY** — useful only as a stopgap, not the path.

## 2. The CPU coverage proof (the concrete de-risk)

Apples-to-apples with the finding: the SAME 124 probe utterances (three groups: `conversational_register`
14, `everyday_real_world_topics` 10, `wikidata_known_agents` 100 from the live store bundle), the SAME
content-word definition, the ACTUAL production word vocab as the baseline. The subword candidate is the
project's own `sim.bpe_tokenizer.BPETokenizer` trained at V∈{4000,8000,16000} on two corpora — TinyStories
(the current, child-story corpus the finding named as the limiter) and wikitext (a broad corpus with proper
nouns / adult vocab) — purely on CPU (bounded merge-table training + coverage counting; RSS ~0.5 GB, no GPU,
no torch). Metrics: **hard_oov_word_rate** (a content word whose BPE encoding hits the `<UNK>` sentinel —
i.e. a character absent from the training alphabet), **fully_representable** (every content word of an
utterance representable), and **mean pieces per content word** (each piece = one spiking-WTA emission = the
read-out cost).

## 3. Results — subword closes hard-OOV completely; the residual is pieces-per-word

Overall, across the 124-utterance probe (`hard_oov` = content-word hard-OOV; `repr` = fully-representable
utterances; `pieces/w` = mean subword pieces per content word; `pieces/OOVw` = mean pieces for the content
words the V=1000 WORD vocab could NOT represent — the proper-noun hard cases):

| tokenizer | hard_oov | fully-representable | mean pieces/word | mean pieces/OOVword |
|---|---|---|---|---|
| **WORD V=1000 (production)** | **39.88%** | **5.65% (7/124)** | 1.0 (by defn) | — (unrepresentable) |
| BPE TinyStories V=4000 | 0.0% | 100.0% (124/124) | 1.98 | 2.97 |
| BPE TinyStories V=8000 | 0.0% | 100.0% (124/124) | 1.72 | 2.80 |
| BPE TinyStories V=16000 | 0.0% | 100.0% (124/124) | 1.72 | 2.80 |
| BPE wikitext V=4000 | 0.0% | 100.0% (124/124) | 1.89 | 2.64 |
| BPE wikitext V=8000 | 0.0% | 100.0% (124/124) | 1.51 | 2.23 |
| BPE wikitext V=16000 | 0.0% | 100.0% (124/124) | 1.50 | 2.22 |

(Note the CPU budget: these tokenizers were trained on the top 5000 frequent words, so the achieved vocab
tops out at ~7.5k (TinyStories) / ~8.2k (wikitext) symbols — the V=16000 rows therefore ≈ V=8000, a
training-budget artifact, not a property of subword. The direction is what matters and is clean: **more
vocab ⇒ fewer pieces**, and a broad corpus encodes proper nouns in FEWER pieces than a child corpus —
wikitext's proper-noun `pieces/OOVword` falls to 2.22 vs TinyStories' 2.80, ~20% fewer, and its overall
pieces/word to 1.51 vs 1.72. A full-budget 16k on a broad corpus would push both lower still.)

**The headline is unambiguous and holds across every corpus and vocab size: hard-OOV 39.88% → 0.0%,
representability 5.65% → 100.0%.** A subword vocabulary can spell any ASCII string (the probe is ASCII;
`_WORD_RE` excludes digits/punctuation), so the closed-vocabulary wall the finding hit is a property of
word-level tokenization specifically, not of the mouth. Per group, the collapse the finding found on
topic-naming turns (`everyday_real_world_topics` 0% representable, `wikidata_known_agents` 1%) goes to 100%
for both under subword — the proper nouns (`beethoven`, `tokyo`, `prefecture`, `shakespeare`) that were
simply ABSENT become spellable from 2-3 familiar pieces.

## 4. The honest residual — pieces-per-word is the read-out cost, and it is tractable; fluency is the real burden

- **Read-out cost.** Each subword piece is one `FewSpikeWordRead.read()` (production: `read_window=40`,
  `pop=8`, `topk=64` ⇒ a 512-neuron Izhikevich soft-WTA over 40 steps). At ~1.7-2.0 pieces/word, a 20-word
  reply is ~34-40 WTA reads vs ~20 at word-level — a **~1.7-2x linear** increase in spiking reads. The head
  grows V=1000→8000/16000 (a larger `argpartition` per step, negligible; the WTA cost is `O(topk=64)`, not
  `O(V)`). So the read-out stays tractable — the pieces-per-word cost is real but small and linear.
- **The real residual is FLUENCY, not representability.** Subword decode must emit a COHERENT SEQUENCE of
  pieces that detokenize to real words — a strictly harder learning target than whole-word selection (the
  model can now produce non-words if the piece sequence is wrong). This finding proves representability
  (coverage); it does NOT prove the retrained subword-WKV will be fluent. That is the retrain's burden, and
  the go/no-go in §6 gates exactly it.
- **Coupled corpus lever.** Representability is corpus-agnostic (any corpus's BPE spells any string). But
  small-model FLUENCY is token-supply-bound: `2026-09-01-generative-cortex-token-supply-lever...` shows the
  broad-domain plateau is TOKEN-STARVATION, and its S7(a) names **matched-quality token supply**
  (broad-TOPIC simple-STYLE, the TinyStories/phi distillation-as-data recipe) as the fluency lever. wikitext
  is broad-topic but complex-style (high NLL at small capacity). So the subword tokenizer and the retrain
  CORPUS are two coupled levers: subword for OOV coverage of proper nouns; a matched-quality broad corpus
  for fluency at the substrate's capacity.

## 5. The choice

- **Tokenizer:** the project's own `sim.bpe_tokenizer.BPETokenizer` (Sennrich-2016 word-frequency BPE,
  self-contained, no external runtime dependency — the same tokenizer the constrained-decode / generative
  cortex work already ships). It reaches 0% hard-OOV on the ASCII probe because its base alphabet covers the
  `[a-z']` set the content words are drawn from. For a hard zero-OOV *guarantee* on arbitrary bytes
  (non-ASCII, emoji), a **byte-level** BPE (256-byte base, GPT-2-style) is the strict form and a small
  variant of the same class; the char-level BPE here is sufficient for English chat and is what ships today.
- **Vocab size:** **V=8000** is the recommended production point — it lowers mean pieces/word versus V=4000
  (1.89→1.51 on the recommended broad wikitext corpus; 1.98→1.72 on TinyStories) and proper-noun pieces to
  2.23, while keeping the head small enough for the WTA read-out; V=16000 buys little more at these corpus
  sizes. Tunable at retrain time.
- **Corpus:** train the BPE on a **broad** corpus (wikitext, for proper-noun piece efficiency); train the
  WKV **retrain** on the largest matched-quality broad-topic corpus available, per the token-supply lever
  (the ideal broad-topic-simple-style corpus is that finding's own S7(a) build).

## 6. GPU training plan — QUEUED, GUARDED (do NOT run; GPU saturated)

**Prerequisite:** the TRAINER tokenizer-swap (§1 change 1) must land first — it does not exist in
`_emerge_wkv_lm_derisk.py` yet, so the command below is *ready to arm*, not runnable today. Once the swap is
in AND the GPU frees, queue it (headless, 0 agent tokens) — mirrors the production checkpoint's ssm /
dual-nonneg / uniform-decay recurrence, at wider `d_model` and the subword `V`:

```
tools/gpu_queue.sh add 'SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
  --seeds 42 43 44 100 101 102 \
  --corpus data/corpus/wikitext103.txt --contiguous --max-len 64 \
  --tokenizer bpe --bpe-vocab 8000 --bpe-path bridges/wkv_ckpt/wkv_bpe8k.json \
  --recurrence ssm --dual-nonneg --uniform-decay --d-model 256 \
  --n-sentences 400000 --max-train-sents 200000 --epochs 12 --batch 256 \
  --save-ssm bridges/wkv_ckpt/wkv_ssm_bpe8k_d256'
```

**GO / NO-GO gate (6-seed):**
1. **Self-NLL parity** — held-out per-token self-NLL ≤ the word-level V=1000 checkpoint's, a fair comparison
   now that both are open-vocabulary. (This is the token-supply finding's own metric.)
2. **Representability re-measure** — re-run `_subword_mouth_tokenizer_coverage_derisk.py` against the SHIPPED
   subword checkpoint's actual vocab: expect the 0% hard-OOV / 100% representable of §3 to hold end-to-end.
3. **Generation coherence (the fluency burden)** — sampled prose from the retrained mouth detokenizes to
   ≥ a set fraction of real dictionary words AND reads as coherent English on the chat probe. **A NO-GO here
   is a verdict on the CORPUS/token-scale METHOD (⇒ the token-supply S7(a) matched-quality-corpus lever),
   NOT on subword tokenization** — representability (this finding) is independent of and prior to fluency.

## 7. Ranked plan (honest effort / value)

1. **Interim: wire the V=4000 word checkpoint.** ~0 build (it exists on disk). Value LOW: doubles gate-pass
   10.48%→17.74% but stays word-level (0% on famous-entity naming) — a dead-end for the OOV capability. Do
   only if a stopgap is wanted before the subword path lands.
2. **Subword tokenizer (THIS finding, proven CPU).** Effort DONE for the coverage side. Value HIGH: closes
   hard-OOV 39.88%→0%, representability 5.65%→100%. This is the load-bearing surpass.
3. **Subword-WKV retrain (queued GPU, §6).** Effort MED: the trainer tokenizer-swap (small, §1) + the decode
   detokenizer (reuse `BPETokenizer.decode`, §1) + the 6-seed retrain. Value MED-HIGH: turns representability
   into an actual open-vocabulary mouth; fluency is the gated burden (§6.3), coupled to the corpus lever.
4. **Tail-learning on the store's entity vocabulary.** Value RE-SCOPED by subword: with subword, entities
   are already REPRESENTABLE, so tail-learning is no longer about coverage — it shifts to FLUENCY/frequency
   on entity-specific pieces (the retrain seeing enough of them). Lower urgency once subword lands; a
   fine-tune knob on top of step 3, not a separate coverage fix.

## 8. What this is NOT

Not a capability claim: the subword mouth is not built, not wired, not on-by-default, not scaffold-retired —
per `docs/TERMS.md`, none of "wired / on-by-default / integrated / works" is asserted. This is a coverage
measurement + a design + a queued plan. Not a fluency result — §3 measures representability and pieces-per-
word, never generation quality (that is §6's gated retrain). Not a re-measurement of the word-level ceiling
(the finding's 39.9%/5.65%/10.48% stand — reproduced here as the baseline). Not the recurrence
training-provenance question (board's second open question) — untouched. The honesty boundary holds: the
one thing PROVEN is that subword tokenization makes any chat-topic string spellable at ~2 pieces/word; the
one thing NOT proven is that a retrained subword mouth is fluent.
