# Mouth token-scale — next-step launch plan (ready for the d192×2B verdict, ~Sep 8)

**Status:** the decisive test is RUNNING on AWS — `i-046d3211935253398` (g6.8xlarge, 1×L4-24GB, 128GB RAM,
us-east-1, ~$2/hr), the **d192×2B FineWeb-Edu → eval-wt103** cell at `--batch 64` (batch 128 OOM'd the L4 twice —
genuine, not fragmentation). ETA ~Sep 8; 60h self-terminate backstop 2026-09-09 03:10 UTC. **Decisive question:
does `margin_vs_trigram` (depth 10-99) go > 0?** (Current deployable d192×0.4B = −0.082, a +0.204 lift over the
wt103-only −0.286 baseline — small margin, still below the trigram.) Harvest: scp the json, read the deep-margin,
then `bash tools/aws_gpu.sh stop`.

**Frame (banked):** architecture is a CLOSED axis on this test (predictive-coding / delta-rule / content-addressing
all measured flat, 2026-09-05 NO-GO). The lever is **token supply** (Chinchilla arXiv:2203.15556; over-train past
optimal per SmolLM2 arXiv:2502.02737 + Beyond-Chinchilla arXiv:2401.00448 — quality keeps rising into the
100–10,000 tok/param band for deploy-size models). d192 = 3.45M active params → Chinchilla-optimal ~69M tokens;
2B ≈ 29× optimal (deliberate over-training, the small-deployed-model recipe).

## Data-prep that can start now (non-GPU) — but note the local-memory limit
- **Bigger FineWeb-Edu pull (only if the 10B cell uses the low-repetition path):** `.venv/bin/python -m
  research.runners.lm_fineweb_to_txt --out data/corpus/fineweb_edu_2b6.txt --target-words 2600000000
  --subset sample-10BT` (~15min, ~16GB disk, deterministic superset — safe, doesn't touch the AWS job).
- **⚠️ The 10B cell can AVOID any download** via Option B: `--epochs 16` on the existing 700M-word
  `data/corpus/fineweb_edu.txt` (~86.5GB RAM, fits the 128GB AWS instance). Prefer this.
- **⚠️ The tokenizer-cache pre-build + any `--n-sentences ≥ ~3M` job is AWS-ONLY** — the 46GB local box thrashes
  into swap on a 2B-token pool (≈69GB RAM). Do NOT run it locally.

## Branch A — d192×2B CROSSES (margin > 0): token-scaling confirmed
1. **6-seed confirm at d192×2B** (`--batch 64`, seeds 42 43 44 100 101 102, same command as the running cell) →
   `_emerge_wkv_lm_linattn_fineweb_evalwt103_d192_2B_6seed.json`. (6 seeds reuse the same corpus pool — no extra
   RAM/corpus.)
2. **Trace the frontier** (grid spec `2026-09-06-mouth-token-scaling-fineweb-pipeline-and-grid-spec.md` §5b):
   d192×10B (Option B, epochs=16, ~222h/~$444 on the L4), then d384×{0.4B,2B} — but VRAM-smoke d384 first (its
   D×D state ≈4× d192's; likely needs batch ≤32 → longer/costlier). d256×2B (~$126) is a cheaper intermediate.
3. **Wire it (owner-gated):** flip the deployed checkpoint → re-run the coverage/honesty gate
   (`2026-09-03-linattn-mouth-broad-scope-coverage-threshold.md`, fix its ~1-line case-fold bug at the same time)
   → the live brain-grounded verification gate before any production default change.

## Branch B — does NOT cross: more of the proven lever, not a new mechanism
1. **Push to the 10B cell** (same d192×10B command) — a not-yet-crossing 2B point + a positive base direction
   means the curve is likely still descending.
2. **Base 6-seed at d192×0.4B on AWS** (~4h/seed) — is −0.082 seed-stable, or could variance cross zero?
3. **Only if 10B also fails:** name the next levers per THE LAW + the deep-research-at-wall gate — (a) the joint
   capacity×token surface (bigger d-model AND more tokens together), (b) distillation-as-data / TinyStories-phi
   (a second data-quality lever on top of FineWeb-Edu's filtering, named as the fallback in 2026-09-01) — NOT a
   characterized-scaling-limit stop.

## AWS cost/time (L4 = 2.2× slower than the local 3090; ~$2/hr)
d192×2B ~44h/~$88 (measured) · d192×10B ~222h/~$444 · d96×2B ~20h/~$40 · d256×2B ~63h/~$126 · d384×2B ~106h/~$213
(2-4× if batch drops) · d384×10B ~531h/~$1062 (gate behind strong prior evidence).

## Deployability (consumer-hardware-reference) — CLEAR at inference
The production mouth readout (`webapp/wkv_mouth_generator.py`) is pure numpy (no torch/cupy); a d384 checkpoint is
~30MB — trivial vs the 24GB 3090 budget. Scaling d_model up does NOT threaten single-3090 deployability at
inference. (The VRAM constraint is TRAINING-time batch size only; and IF the mouth later folds onto the shared
spiking substrate, re-check its state's VRAM against the one-brain budget then.)

_Source: mouth-next-step design pass 2026-09-07 (agent), grounded in the cited repo findings + scaling-law papers._
