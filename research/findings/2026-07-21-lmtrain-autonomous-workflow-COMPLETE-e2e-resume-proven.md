# Autonomous incremental LM-training workflow — COMPLETE + proven end-to-end (bit-exact resume; ready to launch on FineWeb-Edu + compute-commit)

**2026-07-21.** The full workflow the owner asked for is BUILT + validated. `research/runners/lm_train_lib.py` (core) +
`lm_train_run.py` (launcher). NO `sim/` edit.

## What it is
- **Corpus pipeline:** `tokenize` trains + FREEZES a BPE vocab (`sim.bpe_tokenizer`) into the lineage + writes memmap-able
  `tokens_{train,val}.npy`; the resumable `TokenStream` reads it. Corpus path is a config flag (FineWeb-Edu drops in).
- **Model:** the ~30× chunked-scan multi-layer WKV (imported from `_lmtrain_chunked_scan`, gate 4.77e-07), AdamW(fused) +
  cosine-warmup, bf16 autocast, optional `--compile`.
- **Checkpointing:** atomic save of {model, optimizer, LR-sched, TokenStream cursor, torch+cuda+numpy+python RNG, step,
  tokens_seen, config} + rolling history + best pointer + a frozen-config guard (refuses silent arch change on resume).
- **Benchmark:** per-depth held-out NLL (reused `eval_perdepth`) + overall ppl + fixed-prompt generation samples at each
  checkpoint → progress JSONL + human log + samples.txt (check-in any time).
- **Autonomous loop + launcher** (`start/pause/resume/status/tokenize/selftest/e2e`): while not-paused & budget-remaining:
  train chunk → checkpoint → benchmark → log → repeat. PAUSE sentinel (stop at checkpoint boundary, zero loss).
  `--max-tokens/--max-hours/--max-increments` caps. Mirrors the proven `develop_run.py` pattern.

## END-TO-END PROOF: PASS
- **Bit-exact resume** (`selftest`, controller-verified through the REAL save/load): uninterrupted == checkpoint→resume,
  **max|loss diff| 0.00e+00**, restored step correct.
- **CLI full loop** (`e2e`): start → steps 20,40 (ckpt each) → SIMULATED RESTART → resumed at step 40 (no redo/skip) →
  60,80; val_ppl monotone DOWN THROUGH the restart [448→421→395(resume)→370], train_loss keeps dropping, benchmark logs
  accumulate, PAUSE stops cleanly. **GPU/bf16 production path** validated (1.6M WKV, chunked-scan, cuda/bf16: resumed at
  step 30, val_ppl 1147→663→504 dropping across the restart). A real bug found+fixed (RNG ByteTensor→CUDA on load).

## Ready to launch — remaining is SETUP + the owner's compute-commit
- FineWeb-Edu download (one-time; `--corpus-path` flag). The pure-Python BPE is fine for slices but slow for multi-GB —
  use a bounded `--bpe-train-chars` sample or a faster/sharded tokenizer for the real corpus.
- `--compile 1` for the full ~30× on GPU. Arm a coverage-complete Monitor alongside the long `start` run (controller job).
- Pick model size + `--lr-decay-steps` for the token budget (start ~34-67M → fluency in ~1 day at ~90K tok/s).
⇒ every risky piece proven, ~30× optimized, resumable + pausable + autonomous. The workflow is DONE.
