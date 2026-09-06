---
type: finding
status: locked-not-executed
claim_check: measured
date: 2026-09-06
mechanism: mouth token-scaling STEP-2 build — a plain-text FineWeb-Edu corpus pipeline that plugs into
  `_emerge_wkv_lm_derisk.py --contiguous --eval-corpus` UNCHANGED, so the eval instrument (wt103 held-out
  deep-context margin_vs_trigram) stays byte-comparable to the 2026-09-05 STEP-1 baseline while the TRAINING
  corpus scales 3-300x via FineWeb-Edu
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [43]
seed-waiver: this is a PLUMBING/pipeline-verification smoke (tiny d_model, tiny data, 1 seed), not a scaling
  claim — no generalization is asserted from it; the decisive GPU cells (below) are QUEUED, not run here.
runner: research/runners/_emerge_wkv_lm_derisk.py
artifacts:
  - research/runners/lm_fineweb_to_txt.py
  - research/findings/raw/_mouth_token_scaling_fineweb_pipeline_smoke.json
  - research/findings/raw/_mouth_token_scaling_fineweb_pipeline_smoke.json.prov.json
external: >
  Hoffmann et al. 2022 "Training Compute-Optimal LLMs" (Chinchilla) arXiv:2203.15556 (~20 tok/param optimal). <!--derived-->
  Allal et al. 2025 SmolLM2 arXiv:2502.02737 and "Beyond Chinchilla-Optimal" arXiv:2401.00448 (quality keeps <!--derived-->
  improving into the 100-10000 tok/param regime) — already the external grounding for this lane's own
  2026-09-01 GO. NEWLY cited here: Muennighoff et al. 2023 "Scaling Data-Constrained Language Models"
  arXiv:2305.16264 (repeating a fixed pool of training tokens up to ~4 epochs is close in value to that many <!--derived-->
  fresh tokens; value degrades gracefully, not catastrophically, beyond that) — grounds the epoch-for-RAM
  tradeoff this spec uses at the top of the grid (below).
builds_on:
  - research/findings/2026-09-05-mouth-token-scaling-step1-simplewiki-domain-mix-NO-GO.md
  - research/findings/2026-09-05-eval-corpus-instrument-and-token-supply-decisive-test-queued.md
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
verdict: >
  PIPELINE BUILT + VERIFIED end-to-end on a bounded CPU smoke; a 700M-word FineWeb-Edu .txt corpus is PRODUCED
  at data/corpus/fineweb_edu.txt; the decisive GPU cells (one local smoke + a 3x3 AWS capacity x token-supply
  grid) are SPECIFIED below and QUEUED (locked-not-executed — this agent does not run GPU jobs). NO runner
  code change was needed: the additive `--eval-corpus` flag already merged 2026-09-05 is the exact missing
  piece, and `--contiguous`'s `load_stories` loader is memory-SAFE for an arbitrarily large corpus FILE
  (bounded read) — the real, and only, scaling constraint identified is the in-memory PASSAGE-POOL size at
  train time, which is bounded per-cell below via the epoch/uniqueness split, not by the corpus file.
---

# Mouth token-scaling STEP-2: a comparable FineWeb-Edu pipeline (built + smoke-verified) and the AWS capacity x token-supply grid spec

## 0. What this is (and is not)

This is a BUILD + SPEC deliverable, not a scaling result. STEP-1
(`2026-09-05-mouth-token-scaling-step1-simplewiki-domain-mix-NO-GO.md`) tested a domain-MIX proxy (wt103 +
simplewiki) and came back flat-to-slightly-worse (deep margin -0.312 vs the -0.286 wt103-only baseline) — <!--derived-->
informative, but it never touched the SAME-DOMAIN token-supply lever the 2026-09-01 6-seed GO actually names,
because wt103 is ~exhausted (~2.1M sentences) and the same-domain scale test needs a corpus DOWNLOAD (owner
already delegated this fork 2026-09-05; FineWeb-Edu is the approved corpus). This document (1) decides and
builds the pipeline that makes that download comparable to the existing baseline, (2) proves the pipeline runs
end-to-end on a bounded CPU smoke, and (3) hands the controller exact GPU commands for a local smoke and a
full AWS grid — none of which were run here (GPU is busy with the brain-experiment battery; this agent's job is
build + spec, not GPU time).

## 1. The pipeline decision — plain-text FineWeb-Edu, not the npy/`lm_train_run` path

**Read first** (verify-first, drift-#12): `research/runners/_emerge_wkv_lm_derisk.py` (`load_stories`,
`fit_interp_trigram`, `eval_perdepth`, the `--eval-corpus` flag and its decontamination logic, `main()`'s
corpus-loading block) and `research/runners/lm_fineweb_setup.py` (the sibling FineWeb->npy prep, already
running as pid ~708720 at `data/corpus/fineweb_edu_run/` when this session started, targeting a 500M-token
uint16 array via ITS OWN from-scratch 16k-vocab BPE tokenizer).

**Decision: plain-text `.txt` corpus, fed through the EXISTING `--contiguous --eval-corpus` path, zero runner
changes.** Reasons, in order of weight:

1. **Comparability is the hard requirement, and the npy path cannot meet it without rewriting the eval
   instrument.** The 2026-09-05 STEP-1/baseline numbers (-0.286, -0.312) come from <!--derived-->
   `_emerge_wkv_lm_derisk.py`'s OWN `load_stories`/`fit_interp_trigram`/`eval_perdepth` running over the
   `wkv_bpe8k.json` 8001-token BPE vocabulary. `lm_fineweb_setup.py` tokenizes to a uint16 array with a
   DIFFERENT, freshly-trained 16000-token BPE vocab — neither `load_stories` (raw text) nor `load_sentences`
   (raw text) can consume that array, and the array's own vocab could not produce numbers on the SAME trigram/
   bigram/eval-bucket instrument without re-deriving it against this lane's tokenizer. Adapting the npy path
   would mean touching the eval instrument itself, which the fork's own spec rules out.
2. **The additive `--eval-corpus` flag (merged 2026-09-05, `201742d8`) is already the exact missing piece.**
   Read in full (`_emerge_wkv_lm_derisk.py:1718-1728, 1991-1997, 2053-2074`): when set, it draws the held-out
   EVAL set from a SEPARATE corpus using the SAME rng/permutation(0.85 cut)/truncate arithmetic as a standalone
   run on that corpus (byte-identical eval set at a shared seed), decontaminates any exact-string train/eval
   overlap, and leaves every path byte-identical when unset. This is precisely "hold wt103 eval fixed, scale
   the training corpus" — nothing to add.
3. **The memory objection ("does `_emerge_wkv_lm_derisk.py` scale to billions of tokens?") resolves in the
   `.txt` path's favor once the ACTUAL bottleneck is identified — it is not the file read.**
   `load_stories` (the `--contiguous` loader STEP-1 already uses) does `open(path).read(max_stories * max_len *
   8)` — a BOUNDED byte read regardless of the file's true size, so an arbitrarily large FineWeb `.txt` is
   memory-safe to read from (contrast `load_sentences`, the NON-contiguous default, which does
   `Path(path).read_text()` — an UNBOUNDED whole-file slurp; NOT used here, `--contiguous` is already in the
   STEP-1 recipe). The genuine constraint, MEASURED (not assumed) on this box (Sec 3 below), is the in-memory
   Python list-of-passages the runner materializes for the loaded pool (`n_sentences` items) — a function of
   the TRAINING-TOKEN TARGET, not of the source file's size or format. That constraint would recur under
   ANY loader (including a rewritten npy path, which would need its own comparably-sized in-memory training
   set) so it is not an argument for switching mechanisms — it is a per-cell RAM budget to plan around, done in
   Sec 4-5.

## 2. What was built

**`research/runners/lm_fineweb_to_txt.py`** (new, additive, ~90 lines). Streams `HuggingFaceFW/fineweb-edu`
(`datasets.load_dataset(..., streaming=True)`, same call `lm_fineweb_setup.py` uses) document-by-document,
whitespace-flattens each document to ONE line (matching `data/corpus/wikitext103.txt`/`simplewiki.txt`'s own
flat-line shape), writes incrementally (never buffers the corpus — the same bounded-memory shape as the sibling
npy prep), and stops once a `--target-words` budget of whitespace-word matches (the SAME unit `load_stories`'s
`[a-z']+` regex consumes) is written. Skips documents under `--min-doc-words` (default 20). Includes a
documented `os._exit(0)` workaround for an observed HF-`datasets`/pyarrow interpreter-teardown segfault
(`PyGILState_Release` on a stale thread state) that fires AFTER the output file is fully written and closed —
cosmetic (a clean exit code for callers like `gpu_queue.sh`), not a data-integrity issue; confirmed by checking
the file is complete and correct before the crash in both test runs below.

**Produced `data/corpus/fineweb_edu.txt`** (satisfies task item 1's "modest slice now, ~500M-1B tokens"):
`python -m research.runners.lm_fineweb_to_txt --out data/corpus/fineweb_edu.txt --target-words 700000000
--subset sample-10BT` — **700,000,529 words, 907,134 documents, 4,326,798,376 bytes, in 196 seconds**
(streaming throughput ramped from ~0.1M to ~3.6M words/s as the HF dataset connection warmed up; the sibling
npy job's own log shows a comparable ~0.6-0.7M TOKENS/s sustained rate for its heavier full-BPE-encode
path, so this order of magnitude is corroborated by an independent concurrent job on the same box/network).
0 skipped-short documents.

## 3. The bounded CPU smoke (task 2) — plumbing verified, numbers NOT a result

Three SIM_BACKEND=numpy, `CUDA_VISIBLE_DEVICES=""`-forced-CPU runs (no GPU touched), all `--recurrence linattn
--uniform-decay --tokenizer bpe --contiguous --max-len 40 --eval-corpus data/corpus/wikitext103.txt`:

1. d_model=16, n-sentences=2000, max-train-sents=1500, max-eval-sents=300, epochs=2, seed 1, corpus = a 3M-word
   scratch FineWeb slice — completed in 49.1s, depth-10-99 `margin_vs_trigram` = **+0.660**. <!--derived-->
2. Same tiny config against the PRODUCTION `data/corpus/fineweb_edu.txt` (epochs=1) — completed in 37.3s,
   depth-10-99 `margin_vs_trigram` = **-0.080**. <!--derived-->
3. A slightly larger pass, still bounded — d_model=32, n-sentences=8000, max-train-sents=6000, max-eval-sents=
   500, epochs=3, seed 43, against `data/corpus/fineweb_edu.txt` — completed in 180.5s, artifact committed at
   `research/findings/raw/_mouth_token_scaling_fineweb_pipeline_smoke.json` (auto-provenance-stamped:
   `git_sha b23281913`, full argv, confirming the automatic-provenance system CLAUDE.md describes actually
   fires from a fresh runner with no changes needed). depth-10-99 `margin_vs_trigram` = **+0.471**, with a
   large permutation-collapse (+1.147) and a smaller but positive memoryless-collapse (+0.114); the runner's <!--derived-->
   own internal per-arm heuristic (margin > 0.02 AND both collapses > 0.05) prints its pass line at this scale.

**What this proves:** the pipeline runs end-to-end — corpus load, `--eval-corpus` held-out draw from wt103,
content-based decontamination (0 overlap found in all three runs, as expected: FineWeb-Edu and wikitext103 are
disjoint sources, and the decontamination logic correctly found no exact-string matches to drop) — and emits
output in the EXACT SAME JSON schema as the STEP-1/baseline artifacts
(`per_seed.{seed}.by_depth.{bucket}.{wkv,bigram,trigram,margin_vs_trigram,wkv_perm,wkv_memoryless}`), i.e. it is
byte-comparable to `-0.286`/`-0.312` in format. <!--derived-->

**What it does NOT prove, and the three numbers above must not be read as a direction signal:** each run
trains 1500-6000 passages (60K-240K words) for 1-3 epochs at d_model 16-32 — two-to-three orders of magnitude
below even the cheapest real cell in Sec 5. At this scale the FAIR interpolated trigram (Sec "fit_interp_trigram")
is itself severely data-starved (counts fit on a few thousand passages), which inflates or deflates the WKV's
apparent margin against it in either direction depending on the random split — exactly why the three toy runs
above disagree in SIGN (+0.660, -0.080, +0.471) despite testing the "same" question: this is sampling noise at <!--derived-->
toy scale, not a real effect. **Nothing here is being reported as a lift, a lever confirmation, or a GO** —
only as evidence the wiring is correct. The decisive test is Sec 5's local-smoke cell (d192, ~STEP-1 scale).

## 4. The real scaling constraint, measured — and how the grid stays inside it

`load_stories`'s bounded file-read removes the naive "does it crash on a huge file" concern (Sec 1.3). The
constraint that remains is RAM for the in-memory `sents`/`tr_ids` passage pool the runner holds for the whole
run. Measured directly on this box (a synthetic benchmark: 500,000 lists of 40 distinct random 3-9-character
words, `resource.getrusage(...).ru_maxrss` before/after): **~2.9 KB/passage** for the raw word-list alone;
adding the (tok-cache or not) BPE-id list roughly doubles this to **~4.7 KB/passage** in practice. RAM for a
cell ≈ `n_sentences x 4.7 KB`, where `n_sentences ~= max_train_sents / 0.85` (the loader's 85/15 train/eval
split). This is INDEPENDENT of `d_model` — it is a property of the passage pool, not the network.

The STEP-1 baseline itself is the calibration point: `n_sentences=3,000,000` ran to completion (on the
project's GPU box) at an implied ~14.1 GB for the pool — proof this order of magnitude is safe. Scaling the
SAME recipe's `max_train_sents`/`n_sentences` linearly with the training-token target (holding `epochs=4` fixed,
matching every existing run in this lane) gives the RAM column in Sec 5's grid table; at the top of the grid it
gets large enough (Sec 5, the 10B-token cells) to need an explicit epoch-for-RAM tradeoff, using MORE
repetition over fewer unique passages to hit the same total training-token throughput on far less RAM —
Muennighoff et al. 2023 (arXiv:2305.16264) is the citation that repetition up to several x is not free but is <!--derived-->
not a fresh-token equivalent either; both options are given explicitly in Sec 5.

**Active-parameter counts** (measured, not estimated — instrumented directly via the runner's own
`build_and_train_wkv`/`WKV` class, `--recurrence linattn --uniform-decay --n-layers 2`, counting only
parameters that actually receive a gradient in the forward path used, i.e. excluding the model's legacy
always-allocated-but-unused-by-`linattn` base WKV block):

| d_model | active params | total allocated params (incl. unused legacy base block) |
|---|---|---|
| 96  | 1,636,931 | 1,766,534 |
| 192 | 3,450,179 | 3,967,430 |
| 384 | 7,629,635 | 9,696,326 |

d192's 3.45M active params matches the "~3M-param deployable mouth" framing in this lane's own record.

**Throughput anchor** (empirical, from the two already-run 2026-09-05 GPU jobs at this exact d192/n_layers=2/
V=8001/batch=128/`--contiguous --max-len 40` configuration on the project's 3090): `_emerge_wkv_lm_linattn_wt103_scale_s43.json`
(wt103-only, 2.5M train passages x 40 x 4 epochs = 400M-token throughput) ran in 13,698s; the
wt103+simplewiki STEP-1 run (`_emerge_wkv_lm_linattn_wt103plus_simplewiki_evalwt103_s43.json`, ~2.27M
decontaminated passages, same shape) ran in 15,418.8s. Both give **~26,000-29,000 training tokens/sec**; this
spec uses **27,500 tok/s** as the d192 anchor. Relative-throughput scaling across `d_model` is modeled from
the runner's own per-token compute shape — a `D x V` output-projection cost (linear in D, V=8001 fixed) plus an
`n_layers x D^2`-ish recurrent-state-update cost (the LinAttnLayer outer-product trace, quadratic in D) —
giving d96 ~2.21x faster and d384 ~2.39x slower than the d192 anchor: **d96 ~60,900 tok/s, d384 ~11,500
tok/s**. This is an ESTIMATE (flagged as such); Sec 5's local-smoke cell re-anchors d192 exactly and the grid's
own d96/d384 cells should log their observed rate at the start so the controller can re-scale the remaining
schedule if the model is off.

## 5. Exact commands (QUEUED — not run by this agent)

### 5a. LOCAL SMOKE (one `gpu_queue.sh` job, direction-test, hours) — task 3a

Reuses the STEP-1 recipe BYTE-FOR-BYTE except `--corpus` (proven-safe RAM ~14.1 GB, proven wall-clock ~4-4.5h
on the 3090), so its deep-bucket `margin_vs_trigram` is directly comparable to BOTH the wt103-only baseline
(-0.286) and the wt103+simplewiki STEP-1 result (-0.312, NO-GO): <!--derived-->

```
cd /home/dant123/Projects/sim && EXT=json
SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
  --recurrence linattn --uniform-decay --batch 128 --tokenizer bpe \
  --bpe-path bridges/wkv_ckpt/wkv_bpe8k.json \
  --corpus data/corpus/fineweb_edu.txt \
  --eval-corpus data/corpus/wikitext103.txt \
  --contiguous --max-len 40 --max-eval-sents 4000 --epochs 4 --tok-cache \
  --n-layers 2 --d-model 192 --n-sentences 3000000 --max-train-sents 2500000 \
  --seeds 43 \
  --json research/findings/raw/_emerge_wkv_lm_linattn_fineweb_evalwt103_s43.$EXT \
  > research/findings/raw/_emerge_wkv_lm_linattn_fineweb_evalwt103_s43.log 2>&1
```

`data/corpus/fineweb_edu.txt` (700M words, Sec 2) already has ~7x the ~100M words this recipe reads per epoch —
no further corpus prep needed. Direction bar: deep margin lift >= +0.03 over -0.286 (the same bar STEP-1 used). <!--derived-->

### 5b. AWS capacity x token-supply grid — task 3b

3x3 grid: `d_model in {96, 192, 384}` x total-training-token-throughput in `{~0.4B, 2B, 10B}` (throughput =
`max_train_sents x max_len x epochs`, epochs=4 fixed to match every existing run in this lane so `d_model`/
token-supply stay the only two varying axes). All cells: `--recurrence linattn --uniform-decay --n-layers 2
--tokenizer bpe --bpe-path bridges/wkv_ckpt/wkv_bpe8k.json --contiguous --max-len 40 --max-eval-sents 4000
--tok-cache --eval-corpus data/corpus/wikitext103.txt` (the fixed comparable eval, unchanged across every
cell). RAM is the passage-pool cost from Sec 4 and does NOT depend on `d_model`; GPU-hours use the Sec 4 rate
estimates (rate anchored at d192, re-derive from cell (a) below if the local smoke's own throughput differs).

| token-pt | max_train_sents | n_sentences | passage-pool RAM | source words needed (1 epoch) |
|---|---|---|---|---|
| ~0.4B (= STEP-1 recipe exactly) | 2,500,000 | 3,000,000 | ~14.1 GB | 100M (have 700M, 7x headroom) |
| 2B | 12,500,000 | 14,750,000 | ~69.3 GB | 500M (have 700M, 1.4x headroom) |
| 10B | 62,500,000 | 73,600,000 | ~346 GB | 2.5B (NEED a bigger corpus, Sec 5c) |

| d_model | rate (tok/s, estimate) | GPU-h @ 0.4B | GPU-h @ 2B | GPU-h @ 10B |
|---|---|---|---|---|
| 96  | ~60,900 (est.) | 1.82 | 9.12 | 45.6 |
| 192 | 27,500 (measured anchor) | 4.04 | 20.2 | 101.0 |
| 384 | ~11,500 (est.) | 9.66 | 48.3 | 241.5 |

**Full-grid total (1 seed/cell): ~481 GPU-hours** (~20 GPU-days on one 3090-class card) — too large to queue
blind. Recommended STAGING (cheapest-first, matching this lane's own established pattern of a 1-seed direction
test before a 6-seed confirmation):

- **Stage A (~35.2 GPU-h): d96 x {0.4B, 2B} + d192 x {0.4B, 2B}.** d192/0.4B IS cell 5a above (the local
  smoke) — do not double-queue it. Mirrors the 2026-09-01 capacity-matched d96 validation plus the deployable
  d192 scale in one pass.
- **Stage B (~204.6 GPU-h, only if Stage A shows a lift over -0.286/-0.312): d96x10B + d192x10B + d384x{0.4B, <!--derived-->
  2B}.** Needs the bigger corpus (Sec 5c) and a >=70GB-RAM instance for the 2B/10B cells.
- **Stage C (~241.5 GPU-h, only if Stage B confirms): d384x10B**, the single most expensive cell — gate behind
  strong prior evidence, and reconsider whether d192 (the actual deployable-mouth capacity) is the better use
  of this budget before spending it.

Concrete command for one representative cell (d192, 2B token-point; every other cell substitutes its own
`--d-model`/`--n-sentences`/`--max-train-sents`/`--json` from the tables above, same fixed flags otherwise):

```
cd /home/dant123/Projects/sim && EXT=json
SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
  --recurrence linattn --uniform-decay --batch 128 --tokenizer bpe \
  --bpe-path bridges/wkv_ckpt/wkv_bpe8k.json \
  --corpus data/corpus/fineweb_edu.txt \
  --eval-corpus data/corpus/wikitext103.txt \
  --contiguous --max-len 40 --max-eval-sents 4000 --epochs 4 --tok-cache \
  --n-layers 2 --d-model 192 --n-sentences 14750000 --max-train-sents 12500000 \
  --seeds 43 \
  --json research/findings/raw/_emerge_wkv_lm_linattn_fineweb_evalwt103_d192_2B_s43.$EXT
```

For the d96/d384 x {0.4B,2B} cells, same command with `--d-model 96` or `384` and the matching
`--n-sentences`/`--max-train-sents` from the table (0.4B row: 3,000,000/2,500,000; 2B row:
14,750,000/12,500,000); name the output analogously, swapping in the capacity/token-point
(`_emerge_wkv_lm_linattn_fineweb_evalwt103_d{96,384}_{0.4B,2B}_s43` + `.$EXT`).

**10B-cell RAM tradeoff (Sec 4).** Option A (true low-repetition, `max_train_sents=62,500,000`,
`n_sentences=73,600,000`) needs ~346 GB RAM — a memory-optimized-class instance paired with a GPU (a real cost
premium; VRAM is NOT the constraint here, the model is a few million params). Option B (recommended default):
keep `epochs=16` instead of 4 for the 10B cells ONLY, `max_train_sents=15,625,000`, `n_sentences=18,400,000` ->
same 10B total throughput (same GPU-hours, Sec 4/5b's GPU-hour numbers are unaffected by the epoch/uniqueness
split) at ~86.5 GB RAM — the same RAM class as the 2B cells, at the cost of more repetition (Muennighoff et
al. 2023, Sec 4). Example (d192, 10B, Option B):

```
cd /home/dant123/Projects/sim && EXT=json
SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
  --recurrence linattn --uniform-decay --batch 128 --tokenizer bpe \
  --bpe-path bridges/wkv_ckpt/wkv_bpe8k.json \
  --corpus data/corpus/fineweb_edu_2b6.txt \
  --eval-corpus data/corpus/wikitext103.txt \
  --contiguous --max-len 40 --max-eval-sents 4000 --epochs 16 --tok-cache \
  --n-layers 2 --d-model 192 --n-sentences 21600000 --max-train-sents 15625000 \
  --seeds 43 \
  --json research/findings/raw/_emerge_wkv_lm_linattn_fineweb_evalwt103_d192_10B_s43.$EXT
```

### 5c. Bigger corpus needed for the 2B (tight) and 10B cells

`data/corpus/fineweb_edu.txt` (700M words) covers the 0.4B cells outright and the 2B cells with only ~1.4x
headroom (tight but sufficient at `--max-len 40`'s exact accounting). The 10B cells need ~2.5B (Option A) or
~625M (Option B, `epochs=16`) source words. `lm_fineweb_to_txt.py`'s observed sustained rate (~3.5M words/s,
Sec 2) makes either cheap to produce when the controller is ready to queue that stage:

```
# Option A source corpus (2.5B+ words, ~12-15 min at the observed rate):
/home/dant123/Projects/sim/.venv/bin/python -m research.runners.lm_fineweb_to_txt \
  --out /home/dant123/Projects/sim/data/corpus/fineweb_edu_2b6.txt --target-words 2600000000 --subset sample-10BT

# Option B source corpus (625M+ words, already covered by the existing 700M-word file — no extra prep needed).
```

## 6. Honest scope

No GPU was used by this agent (SIM_BACKEND=numpy, `CUDA_VISIBLE_DEVICES=""` on every smoke). No `sim/` edit.
No production change — the deployed mouth default remains linattn/wt103-trained, unaffected by this build. The
three smoke numbers in Sec 3 are plumbing evidence only (see the explicit non-claim there) — this document
asserts NOTHING about whether FineWeb-Edu token-scaling lifts the deep margin; that is exactly what cell 5a
(queued) will measure. The throughput-scaling model for d96/d384 (Sec 4) is a computed estimate, not measured
— flagged, with the re-anchoring path named. RAM estimates (Sec 4) come from one synthetic Python-object
benchmark on this dev box, not from the actual GPU box STEP-1 ran on; the ~14.1GB/~3M-passage figure is
corroborated by STEP-1 itself having completed, but the 2B/10B RAM figures are extrapolations and should be
watched (a `free -h` on the target instance before launch) rather than trusted blind.
