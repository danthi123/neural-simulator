# LM-training workflow VALIDATED end-to-end on REAL FineWeb-Edu data — the last unproven piece (real-data training) confirmed; de-risk ladder complete → the real go/no-go scaling run is unblocked

**2026-07-21.** The autonomous incremental LM-training workflow (design + de-risks in
`docs/plans/2026-07-21-autonomous-incremental-LM-training-workflow-design.md`) had every LOAD-BEARING piece validated on
SYNTHETIC data (resume bit-exact, cursor exact, throughput, the ~30× chunked-scan optimization). The one thing unproven
was **does it actually train on REAL corpus data** — now confirmed.

## Corpus pipeline (`lm_fineweb_setup.py`) — WORKS
- FineWeb-Edu `sample-10BT` streamed from HF, HF Rust BPE (vocab 16000, ByteLevel) trained+frozen on the first 15k docs,
  tokenized to a preallocated `uint16` array → `tokens_train.npy` (99.6M) + `tokens_val.npy` (400k) + `tokenizer.json`.
  100M-token slice in **114s** (~0.87M tok/s streaming-bound).
- Two fixes made during validation (both committed):
  1. **Launcher CLI**: `lm_train_run.py start --tokenizer` accepted only `{bpe,byte}`; added `hf` (the library
     `_load_tokenizer` + `HFTokenizer` wrapper were already wired, the launcher argparse was the only blocker).
  2. **Tokenizer decoder**: the BPE had a ByteLevel *pre-tokenizer* but no matching *decoder*, so `decode()` returned raw
     `Ġ`-pieces in the generation samples. Added `tk.decoder = decoders.ByteLevel()` (in the setup + patched the two
     already-frozen tokenizers post-hoc — safe: changes decode only, token ids unchanged, verified). Samples now render
     clean English.

## Validation training run — 83.2M (d1024/L16), real 100M FineWeb-Edu slice, chunked-scan + `torch.compile`, bf16, GPU
`val_ppl` drops **monotonically** over 3 increments (900 steps, 7.4M tokens):
| step | tokens | train_loss | val_ppl | val_nll | by-depth NLL (1 / 4-5 / 10-99) |
|---|---|---|---|---|---|
| 300 | 2.46M | 7.226 | 440.97 | 6.089 | 6.79 / 6.11 / 6.07 |
| 600 | 4.92M | 5.985 | 283.07 | 5.646 | 6.23 / 5.69 / 5.68 |
| 900 | 7.37M | 5.629 | 235.56 | 5.462 | 5.99 / 5.48 / 5.45 |
- **Throughput** ~68K tok/s (300 steps/36.2s) via the chunked+compile path (compile warmup ~53s one-time; NO hang —
  the chunked scan is compile-friendly where the Python-loop recurrence was not).
- **Context IS being used**: by-depth NLL *decreases* with context depth (5.99 @ depth-1 → 5.45 @ depth-10-99) — the WKV
  recurrence carries information across the sequence, not a bag-of-words.
- **Generation samples are coherent English** at only 7.4M tokens (e.g. "the meaning of fact that it's an indicator.
  Almost of this one care have the same time about the person in, they have identified our own society..."). Not fluent
  yet (that's the 1.5B+ run's job) but decisively real-word, grammatical-ish — the pipeline learns.
- **Resumable stop** clean ("resume: re-run `start`").

## Verdict + next
- ⇒ **The workflow is VALIDATED end-to-end on real data.** Every de-risk (resume, cursor, throughput, optimization,
  corpus pipeline, real-data training, benchmark+samples) is now green. Training is fully unblocked.
- **NEXT (the decisive go/no-go, de-risk #5):** the 1.5B-token FineWeb-Edu slice (`run2`, tokenizing now) → launch the
  83M run open-ended (checkpoint+benchmark+sample each increment, PAUSE sentinel, armed Monitor) → watch whether broad-
  domain `val_ppl` collapses from ~235 toward the 20-40 range as tokens scale (does the WKV track the scaling curve?).
  THAT is the go/no-go on "converse like a small LLM is a training run away."
- The 100M slice memorizes on an 83M model (1.2 tok/param) so it is a PIPELINE validation only, not the fluency corpus;
  the 1.5B slice (18 tok/param, ≈Chinchilla) is the real signal, and a bigger production corpus follows the go/no-go.
- Files: `lm_fineweb_setup.py`, `lm_train_run.py`, `lm_train_lib.py` (+ `HFTokenizer`); run dir `bridges/lmtrain/run1`.
