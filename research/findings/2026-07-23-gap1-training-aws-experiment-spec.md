# gap#1 fluency-training — AWS experiment spec (job "a"), read-only scoping (2026-07-23)

**Task:** produce a ready-to-launch spec for the highest-value gap#1 fluency-training experiment to run on **AWS GPU
in parallel with the local run3** (which owns the 3090 indefinitely). READ-ONLY — no code edited, no training launched.

**Bottom line (recommendation up front):** run a **width-scaling ladder of the run3 WKV architecture on the identical
6B FineWeb-Edu corpus — `d_model` 1024→1536→2048 (83M→162M→267M), `n_layers=16` fixed** — on a single A10G box.
Primary decision instrument = the **~270M** point (Chinchilla-optimal for the 6B corpus); ideal = add the **~160M**
point on a 2nd concurrent box for the full scaling curve. **~$25–90 total, ~2–4 GPU-days, zero 3090 contention.**
It decisively answers the one **live, untested** lever on the fluency critical path: *does making the generator bigger
materially improve its fluency, or is run3 at the fluency floor for this corpus?* **Verdict: worth doing now** — run3
has plateaued, the data lever is already spent, and this is the "~150M first real run" the plan itself called for.

---

## 1. State of the world (measured, not assumed)

### run3 = the local run, and what it actually is
- **Architecture:** WKV/RWKV-style multi-layer diagonal-SSM (`_lmtrain_chunked_scan.WKV`), `d_model=1024`,
  `n_layers=16`, `vocab=16000` (HF byte-level BPE), `seq_len=256`, `batch=32`, `chunk_c=16`.
  **Verified param count = 83.2M** (emb 16.4M + 16×[3·1024²+…] ≈ 50.3M + head 16.4M). Matches the "~83M" framing.
- **Corpus:** **FineWeb-Edu `sample-10BT`, capped at 6.0B tokens** (`corpus_meta.json`: 5.976B train / 24M val).
  Already tokenized to `bridges/lmtrain/run3/tokens_train.npy` (12 GB uint16) + `tokens_val.npy` (46 MB) +
  `tokenizer.json` (1.1 MB). **The dataset lever is already exercised** — this is not the old SimpleWiki setup.
- **Optimizer:** AdamW lr 3e-4, warmup 2000, cosine `lr_decay_steps=3,000,000` (min ratio 0.1), wd 0.1, bf16 autocast,
  `torch.compile max-autotune`. Launched/auto-resumed via `lmtrain-resume.service` on boot.
- **State (2026-07-23 09:32):** **step 920,000, tokens_seen 7.54B (~1.26 epochs), best_val_nll 3.987 (ppl 53.9)**,
  current ppl ~55–57. **1214 benchmarked increments.** Throughput measured **~74.5k tok/s** on the 3090
  (500 steps × 8192 tok / 55 s).
- **Samples at ppl ~57 are genuinely fluent** local English (grammatical, coherent sentences; confabulates facts, as
  expected of a fluency-only generator). This matters: **the base is already fluent enough to be a fluency-only
  generator behind the gate** — which reframes candidate #3 (RA fine-tune) below.

### run3 has PLATEAUED (the key fact driving the recommendation)
Val-ppl on the fixed held-out shard:

| tokens | 0.8B | 1B | 2B | 3B | 4B | 6B | 7.5B (now) | best-ever |
|---|---|---|---|---|---|---|---|---|
| val_ppl | 70.25 | 66.47 | 64.68 | 60.48 | 59.43 | 56.30 | 55.55 | **53.89** @7.39B |

From **3B→7.5B tokens (4.5B tokens, ~48 GPU-hours) ppl moved 60.5 → 55.6 (~8%)**, and 6B→7.5B barely at all. The
cosine schedule still has lr at ~80% of peak (set to decay over 3M steps ≈ 24.6B tokens), so run3 *will* keep inching
down, but at **~72 tokens/param it is well past Chinchilla-optimal (20×) for its size** — it is **capacity-bound, not
data-bound.** Continued 3090-days on the 83M tail yield diminishing fluency.

### Correcting the "data-bound" note
CLAUDE.md's *"88.6M params / 41M tokens ≈ 0.46 tok/param → more DATA is the next lever"* is about the **old Gen-F
generator on 41M SimpleWiki tokens** — a genuinely data-starved regime. **run3 already fixed that** (6B FineWeb-Edu =
72 tok/param). So "more data on the run3 arch" is **not** an open lever; the open lever is **scale**. The
`months-scale plan` design doc says as much and specifies *"Start ~150M for the first real run"* — **run3 at 83M
undershot the intended first size.** This experiment realizes it.

---

## 2. Candidate ranking — value(fluency critical path) / (cost + setup)

| Rank | Candidate | Value | Cost/setup | Verdict |
|---|---|---|---|---|
| **1** | **BIGGER MODEL (scale lever) on the same 6B corpus** | **High — the one untested live lever; run3 is at its capacity floor; the plan's intended "~150M first run"** | **Low — data already tokenized (upload 12GB or re-tokenize), single A10G, ~$25–90, apples-to-apples eval** | **DO THIS** |
| 3→2 | RA grounded-render fine-tune | Medium — on the grounding path, and run3 *is* plateaued enough to fine-tune a snapshot | **Wrong instrument for AWS: it's CHEAP (small data, few steps) — hours on the 3090 in a pause, not GPU-days on cloud** | Do locally, not as the AWS job |
| 1→3 | DATA-SCALING (more/broader corpus, same 83M) | **Low — 83M is already at 72 tok/param on 6B; more data won't move a capacity-bound model.** Broader-corpus *diversity* is a real but secondary question | Low cost but low value at this size | Not now (fold corpus-diversity into the bigger-model run if desired) |
| — | Spiking-forward conversion of run3 / arch search | Medium but **belongs on the sim substrate (cupy/on-bridge), not a pure-torch AWS box**; 88.6M already validated | n/a | Off-target for this AWS job |

**Why #1 beats #3 (RA fine-tune), explicitly:** the RA render/QA fine-tune (EMERGE-57 lineage) is a *format*
fine-tune on a small grounded-frame set interleaved with anti-forgetting data — it converges in a small number of
steps on a tiny corpus. It is **not GPU-days-bound**, so spending an AWS GPU-day on it wastes the parallel-cloud
opportunity. It's also premature to bake it into run3 while run3 is still (slowly) improving; fine-tune a frozen
*best.pt* snapshot when needed, on the 3090 during a PAUSE, or on a cheap short cloud box. **The AWS job should be the
thing that genuinely needs GPU-days: a bigger base.** (And a bigger base makes any later RA fine-tune strictly
better.)

**Why #1 beats data-scaling:** the data lever is already spent for an 83M model. A bigger corpus only pays off with a
bigger model — which is exactly candidate #1. If corpus *breadth/diversity* is wanted, it rides for free on the
bigger-model run (see §7 option).

---

## 3. TOP RECOMMENDATION — ready-to-launch spec

### 3.1 The experiment
Train the **same WKV architecture and identical hyperparameters as run3, varying only `d_model`**, on the **identical
6B FineWeb-Edu tokens run3 trained on**, and benchmark on the **identical fixed val shard** — a clean controlled
width-scaling ladder:

| model | d_model | n_layers | params | ratio vs 83M | Chinchilla data (20×) |
|---|---|---|---|---|---|
| run3 (local, exists) | 1024 | 16 | 83.2M | 1.0× | 1.7B |
| **A "~160M"** (conservative) | **1536** | 16 | 162.5M | 1.95× | 3.2B |
| **B "~270M"** (primary) | **2048** | 16 | 267.0M | 3.21× | **5.3B ≈ the whole corpus** |

Everything else **held identical to run3**: lr 3e-4, warmup 2000, `lr_decay_steps` 3e6, wd 0.1, batch 32, seq_len 256,
chunk_c 16, vocab 16000, seed 42, bf16, `torch.compile max-autotune`. Only `d_model` changes ⇒ any ppl delta is
**attributable to scale alone.** (6B tokens is Chinchilla-optimal for ~270M — the data budget already sitting on disk
is *exactly right* for model B; no new data needed.)

### 3.2 The decision it informs (why it's worth $ + GPU-days)
**"Is the generator's fluency ceiling materially higher at larger, still-plausibly-spiking-convertible scale — enough
to make a bigger model the production generator target?"** This is the gap#1 SCALE lever, the last untested one:
- **Win** (ppl drops clearly at matched tokens) ⇒ upgrade the generator target; it justifies extending the
  spiking-forward convertibility validation past 88.6M, and any RA fine-tune / grounded-render work lands on a
  better base.
- **Null** (ppl ≈ run3 at matched tokens) ⇒ **bank "83M is fluency-efficient at this corpus,"** stop spending 3090-days
  on the run3 tail, and redirect to corpus *diversity* or the emergence engine. Either outcome is decisive and
  changes what we do next — that is what makes it worth the spend.

### 3.3 Data — needed, and whether it must be prepped
**No new data prep required.** Two ways to feed the AWS box the *identical* tokens (identical ⇒ the comparison is
exact):
- **(preferred) Upload run3's frozen artifacts** — `tokens_train.npy` (12 GB), `tokens_val.npy` (46 MB),
  `tokenizer.json` (1.1 MB) — via S3. Drop them into the new run dir; `lm_train_run start` sees them cached and skips
  tokenization entirely. **Guarantees byte-identical tokens + val shard** ⇒ perfectly apples-to-apples vs run3.
- **(fallback if the 12 GB uplink is slow) Re-tokenize on-box** with the committed
  `research/runners/lm_fineweb_setup.py --subset sample-10BT --max-tokens 6e9 --vocab-size 16000` (~1.5–2 h streaming
  at ~0.9M tok/s + BPE). Deterministic, but only *byte-identical to run3 if the FineWeb-Edu revision + stream order
  are unchanged* — so **prefer the upload** for a clean control. (Set `HF_TOKEN` to avoid the unauthenticated
  rate-limit warning seen in run3's setup.log.)

### 3.4 AWS instance, runtime, cost, quota
- **Instance: `g5.2xlarge`** — 1× **A10G 24 GB**, 8 vCPU, 32 GB RAM. VRAM need for 267M training is **~10 GB**
  (fp32 weights 1.1 GB + grads 1.1 GB + AdamW moments 2.1 GB ≈ 4.3 GB fixed + ~5 GB bf16 activations at
  batch 32/T256) ⇒ **fits comfortably at batch 32, headroom to 48–64.** 32 GB RAM comfortably page-caches the 12 GB
  token memmap. `g5.xlarge` (16 GB RAM, $0.2/hr cheaper) also works (memmap doesn't fully load) — use it to cost-min.
- **⛔ Do NOT use a multi-GPU box** (g5.12xlarge/48xlarge): **`lm_train_run.py` is single-device (no DDP)** — extra
  GPUs sit idle. A10G is the right price/perf tier here; an A100/H100 (~2–3× faster, ~3–4×$) is overkill.
- **Runtime + cost** (from run3's measured 74.5k tok/s, scaled by param ratio, A10G taken as 0.8–1.0× a 3090;
  on-demand g5.2xlarge ≈ $1.21/hr, spot ≈ $0.35–0.45/hr — *verify current pricing*):

| model | throughput (est.) | to 1B tok (early check) | to 4B tok (floor) | cost to 4B (spot / on-dem) |
|---|---|---|---|---|
| A ~160M | 30–38k tok/s | 7–9 GPU-hr | **29–36 GPU-hr** | **~$12–15 / ~$35–44** |
| B ~270M | 19–23k tok/s | 12–15 GPU-hr | **48–60 GPU-hr** | **~$20–25 / ~$58–73** |

  **Single-run (B only) to 4B tok: ~2–2.5 GPU-days, ~$20–25 spot.** Ladder (A+B, two concurrent boxes to 4B):
  **~$32–40 spot / ~$95–115 on-demand.** Cap with `--max-tokens 4_000_000_000` (single epoch, no repeat).
- **64-vCPU G quota:** g5.2xlarge = 8 vCPU ⇒ the A+B ladder = 16 vCPU, **trivially within a 64-vCPU quota** (fits ~8
  concurrent). No quota-increase request needed. Use **spot** (the workflow is fully checkpoint-resumable — a spot
  reclaim just resumes bit-exact on the next box).

### 3.5 Dependency setup (Deep Learning AMI)
The training path needs **only `torch` + `numpy` + `tokenizers`** — verified: `lm_train_lib`, `_lmtrain_chunked_scan`,
`_lmtrain_stream_cursor_derisk`, `_emerge_wkv_lm_derisk.eval_perdepth`, and `sim.bpe_tokenizer` import **no cupy, no
`sim.bridge`, no GPU-sim stack.** Clean, minimal box.

```bash
# On an AWS Deep Learning AMI (Ubuntu, PyTorch) g5.2xlarge:
git clone https://github.com/danthi123/neural-simulator && cd neural-simulator
python -m venv .venv && . .venv/bin/activate
pip install "torch>=2.4" numpy tokenizers          # cu12x wheel; DLAMI usually has a recent torch already
#   (run3 is on torch 2.11.0+cu128; any torch 2.4+ with cu12x works — the chunked scan is standard torch.)

# --- get the identical data (preferred: S3 upload of run3's frozen artifacts) ---
RUN=bridges/lmtrain/run_awsB ; mkdir -p $RUN
aws s3 cp s3://<bucket>/run3/tokens_train.npy $RUN/tokens_train.npy   # 12 GB
aws s3 cp s3://<bucket>/run3/tokens_val.npy   $RUN/tokens_val.npy
aws s3 cp s3://<bucket>/run3/tokenizer.json   $RUN/tokenizer.json

# --- launch model B (~270M). Config freezes on first start; --max-tokens caps the decision budget ---
python -m research.runners.lm_train_run start --root $RUN \
    --tokenizer hf --vocab-size 16000 --seq-len 256 --batch 32 --chunk-c 16 \
    --d-model 2048 --n-layers 16 \
    --lr 3e-4 --warmup-steps 2000 --lr-decay-steps 3000000 --weight-decay 0.1 --seed 42 \
    --device cuda --amp 1 --compile 1 --compile-mode max-autotune \
    --chunk-steps 1000 --max-tokens 4000000000
# (early GO check: set --max-tokens 1000000000 first, inspect, then re-run start to CONTINUE to 4B — resumable.)
# Model A ~160M on a 2nd box: identical command, --d-model 1536, --root run_awsA.
```
Results push-back: `progress.jsonl` + `samples.txt` + `metadata.json` are tiny — push to git or S3 for live owner
inspection; the `ckpt/best.pt` (~3–4 GB incl. optimizer state) to S3 if you want to keep the model. Arm a
coverage-complete Monitor (done/crash/hang; a spot reclaim is a resume, not a failure).

### 3.6 GO / decision criteria (identical val shard ⇒ direct ppl comparison to run3)
Benchmark on the **same fixed held-out shard** run3 uses (`make_held_out`: 200 seqs × 128 tok). run3 reference:
**@1B 66.5 · @3B 60.5 · @4B 59.4 · floor 53.9.** Then:

- **Early positive (@1B tokens):** model B ppl **≤ 60** (vs run3 66.5) — scale is helping; continue to 4B.
- **Clear win (@3B tokens):** model B ppl **≤ 52** (vs run3 60.5; ≥14% relative) — the scale lever bites.
- **DECISIVE GO (floor, ~3–4B tok):** model B ppl **≤ 45** (≥17% below run3's 53.9 floor; expected ~40–46 if a
  standard ~5–10% NLL/decade-of-params reduction holds) **AND** fixed-prompt samples visibly more coherent
  ⇒ **upgrade the generator target; schedule the >88.6M spiking-convertibility check.**
- **Scaling curve (A vs B):** a monotone ppl drop 83M→162M→267M with A between run3 and B confirms it's genuine
  scale, not a fluke, and tells us whether the return is still steep (keep scaling) or flattening.
- **NO-GO / BANK:** model B within **~5% ppl of run3** at matched tokens (floor ppl **≳ 51**) ⇒ the corpus/regime is
  the limiter, not size — **bank "83M is fluency-efficient," stop the 83M tail, pivot to corpus-diversity or the
  emergence engine.**

---

## 4. Honest verdict — is job (a) worth doing now?

**Yes — it is the single highest-value GPU-days-bound experiment for the fluency path, and now is the right time.**

Reasons it clears the bar:
1. **run3 has plateaued** (3B→7.5B tok = ~8% ppl for ~48 GPU-hours; 6B→7.5B negligible). At 72 tok/param it is
   **capacity-bound**, so more 83M training is low-marginal-value — exactly the regime where the *scale* question
   should be tested instead.
2. **Scale is the one untested live lever.** The data lever is already spent (6B FineWeb-Edu). We genuinely do not
   know if a bigger generator is materially more fluent on this corpus. That's a real, decision-changing unknown on
   the critical path.
3. **The plan already intended it** — *"start ~150M for the first real run"*; 83M undershot. This is the intended
   first-scale run, just executed on cloud in parallel.
4. **Cheap, trivial, decisive, fully parallel:** data already tokenized (upload 12 GB or re-tokenize), minimal deps
   (torch/numpy/tokenizers), single A10G on **spot** (~$20–40), **apples-to-apples** ppl on the identical val shard,
   **zero 3090 contention** — the whole point of the parallel-cloud job.
5. **Gate C (cloud spend) is my call and is met:** the workflow is proven (bit-exact resume selftest; run3 at 920k
   steps), so the compute trains the *real generator*, not a scaffold racing a bigram. Decisive experiment, not a
   fishing expedition.

Honest caveats (do not oversell):
- **This advances a SCAFFOLD, not the core mission.** The generator is an ANN behind the gate-first moat; spiking
  conversion is deferred. Per the master directive, the PRIMARY effort is the emergence engine / dendritic cortex —
  this is a *sanctioned parallel use of otherwise-idle cloud capacity*, framed exactly as the owner posed it, not a
  redirection of the main effort.
- **Convertibility risk at 267M:** spiking-forward is validated only to 88.6M; 267M is ~3× beyond. A fluency win must
  pass a downstream >88.6M convertibility check before it becomes the *production* generator. **This is why the
  ladder includes the ~160M point** (1.95×, much closer to the validated regime) — a 160M win is more directly
  actionable even if 270M turns out non-convertible. The fluency *measurement* is valid at any size.
- **If run3 is genuinely data/corpus-bound rather than capacity-bound, the bigger model won't help** — but that is a
  clean, useful negative that redirects effort, not a wasted run.
- **Not the AWS job, but do soon (locally):** an RA grounded-render fine-tune on a frozen run3/best-model snapshot
  (cheap, hours on the 3090 during a PAUSE) — it's on the grounding path but does not need cloud GPU-days.

**Recommendation:** launch job (a) = the **~270M primary** (or the **~160M + ~270M ladder** on two concurrent spot
A10G boxes for the full curve), on the identical uploaded 6B tokens, `--max-tokens 4e9`, early GO check at 1B. Budget
**~$20–40 spot**, **~2–4 GPU-days**, wholly in parallel with run3.

---
*Read-only scoping; no code changed, no training launched. Doc: `research/findings/2026-07-23-gap1-training-aws-experiment-spec.md`.*
