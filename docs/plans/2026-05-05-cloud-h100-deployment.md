# Cloud H100 deployment plan — getting maximum value per dollar

**Date:** 2026-05-05
**Trigger:** User noted H100 80GB rentable at ~$2/hr on cloud providers.
That's affordable enough to use for big validation sweeps; the question
is how to structure runs to get the most science per dollar.

---

## Cost vs throughput math

| GPU | VRAM | Bandwidth | Compute (FP32 TFLOPS) | $/hr cloud |
|---|---|---|---|---|
| RTX 3090 (local) | 24 GB | 936 GB/s | ~36 | $0 (electricity ~$0.08/hr) |
| RTX 4090 | 24 GB | 1008 GB/s | ~83 | ~$0.30-0.50/hr cloud |
| A100 40 GB | 40 GB | 1555 GB/s | ~20 (FP32, but tensor cores boost) | ~$1-1.50/hr |
| A100 80 GB | 80 GB | 2 TB/s | ~20 | ~$1.50-2/hr |
| **H100 80 GB** | **80 GB** | **3.35 TB/s** | **~67** | **~$2/hr** |
| H200 141 GB | 141 GB | 4.8 TB/s | ~67 | ~$3-4/hr |

**For our workload (sparse SNN, mostly bandwidth-bound):**
- H100 vs 3090: ~3.6× bandwidth + ~1.9× compute → realistic **~3-4× wall-clock speedup** per single process
- Plus H100's 80GB lets us run **parallel-12+ at bio scale** (vs 6 on 3090) → another ~2× throughput
- Combined: **~6-8× faster sweeps** on H100

## Cost analysis on real workloads

| Sweep | Local 3090 wall time | H100 wall time (est) | Cost on H100 |
|---|---|---|---|
| bio_sanity_check (24 runs) | ~1 hour | ~10-15 min | ~$0.50 |
| bio_proof_of_concept (12 runs) | ~5 hours (parallel=2) | ~30-45 min (parallel=12) | ~$1-1.50 |
| **bio_b3_validation (18 runs)** | ~9 hours (parallel=2) | **~1 hour (parallel=12)** | **~$2** |
| bio_three_factor (18 runs) | ~9 hours | ~1 hour | ~$2 |
| **Full investigation arc (4 sweeps)** | **~24 hours** | **~3 hours** | **~$6** |

For comparison, local 3090 electricity at ~$0.08/hr × 24h = $1.92. So
**cloud H100 costs ~3× the electricity but completes in 1/8 the time.**
The trade-off is wall time, not money.

## When to use cloud vs local

**Use cloud H100 for:**
- Large multi-condition validation sweeps (≥12 runs)
- Final paper-level results where confidence matters
- One-off "is this idea worth pursuing" experiments at high scale
- Burst capacity when local is busy
- Architecture testing at sizes that exceed 24 GB VRAM

**Stick with local 3090 for:**
- Iterative development (5-10 min smoke tests)
- Single-seed exploration runs
- Anything during normal work hours when you want the GPU "warm"
- Plumbing/integration testing (cloud cold-starts ~1-2 min, kills iteration)

## What we need to deploy on cloud

The codebase is already mostly cloud-ready:

✅ **`requirements.txt`** has cupy + dependencies.
✅ **No hardcoded paths** — uses relative `research/findings/raw/...`.
✅ **YAML-driven sweeps** — `experiment_runner.py` is environment-agnostic.
✅ **Git-versioned** — clone + checkout main works anywhere.

⚠️ **Needed:**

1. **CUDA version match.** H100 cloud images typically run CUDA 12.x.
   Our `cupy-cuda12x` line in requirements.txt should match. Verify
   on first deploy.

2. **Run command.** Cloud doesn't have our PowerShell launchers. Need
   a portable bash equivalent OR use the YAML configs directly:
   ```bash
   python -m research.experiment_runner experiments/bio_b3_validation.yaml
   ```

3. **Result extraction.** Cloud instance terminates → results lost.
   Either:
   - `git push` results back to repo (commit per run)
   - Or rsync results to local at end
   - Recommended: at end of sweep, `git add research/findings/raw/g11_bg/text_eval_*.json && git commit && git push`

4. **GPU memory awareness.** H100 has 80 GB. Bio-scale uses ~2 GB/proc.
   Could do parallel=30+ in theory, but Python GIL/process-count
   becomes the limit. Reasonable max: parallel=12 (each gets ~6 GB
   headroom for OS/CUDA overhead).

## Concrete cloud-deploy script

```bash
#!/bin/bash
# scripts/deploy_to_cloud.sh — clones, installs, runs a YAML sweep,
# pushes results back. Designed for vast.ai / RunPod / Lambda images.
set -e

REPO_URL="https://github.com/danthi123/neural-simulator"
SWEEP_YAML="${1:-experiments/bio_b3_validation.yaml}"

# Clone + setup
cd /workspace
[ -d neural-simulator ] || git clone "$REPO_URL"
cd neural-simulator
git pull

# Install (idempotent)
pip install -q -r requirements.txt

# Run
python -m research.experiment_runner "$SWEEP_YAML" 2>&1 | tee sweep.log

# Result aggregation
python -m research.result_aggregator --config "$(basename $SWEEP_YAML .yaml)" \
    --out "research/findings/$(date +%Y-%m-%d)-cloud-$(basename $SWEEP_YAML .yaml)-results.md"

# Push results back
git config user.email "cloud@neural-sim"
git config user.name "Cloud Run"
git add research/findings/raw/g11_bg/text_eval_*.json research/findings/*-cloud-*.md
git commit -m "cloud(${SWEEP_YAML}): results from $(hostname)"
git push origin main
```

## H100-specific tuning

The bigger VRAM (80 GB vs 24 GB) lets us push parallelism + scale further:

**Recommended H100 YAML overrides:**
```yaml
parallelism: 12          # was 6 on 3090
n-lang-input: 4096       # was 2048 — bigger arch saturates compute
n-motor-per-action: 1000 # was 500 — better population coding signal
```

This gives a **larger** experiment, not just faster. Same wall time
as our 3090 setup but with 4× the synapse count (more biologically
faithful population coding).

## Recommended cloud workflow

1. **Develop locally**: smoke-test changes on 3090, get to passing point.
2. **Push to git**: commit + push. Both remotes (origin + gitea).
3. **Spin up cloud H100** (RunPod / Lambda / vast.ai, $2/hr).
4. **Run via deploy script**: 1 hour wall time for 18-run validation.
5. **Auto-push results** back to git from cloud.
6. **Spin down cloud** (auto on idle, set 10-min idle timeout).
7. **Pull results locally**: `git pull` to get cloud findings.

Per major experiment: ~$2-3 cloud cost + 1 hour wait = realistic
science iteration speed for paper-level claims. No need to own bigger
hardware.

## When NOT to bother with cloud

- Iteration loops (the cold-start time dominates)
- Single-seed exploration
- Anything < 4 hours of compute on local (cloud overhead > saved time)

## Risks + mitigations

- **Result loss if instance terminates mid-sweep**: deploy script
  should `git commit` per individual run completion, not just at end.
  Modify experiment_runner's `run_experiment` to add a commit hook.
- **Cuda mismatch**: verify cupy version on first cloud spin. If
  mismatch, `pip install cupy-cuda12x` (or whatever matches).
- **VRAM-related differences**: things that work at parallel=12 on H100
  may OOM at parallel=6 on 3090 with bigger arch. Always test locally
  first at the smaller scale, then scale up on cloud.

## Decision tree for next sweep

The currently-running 3-factor chain is on local 3090. After it
finishes:

- **If gradient_works (≥4/9 aligned)**: validation sweep is the next
  step. **Run on cloud** for faster results ($2 for 1 hour vs 9 hours
  local).
- **If gradient_partial or gradient_fails**: implementation work
  needed (sparse codes test, dendritic learning, etc.). **Stay local**
  while iterating on the new runner.

This makes the cost-benefit clean: cloud for committed sweeps,
local for exploratory work.
