# Claude-downtime dispatch MANIFEST — consolidation dendritic operating-point sweep on the mini-PC pool (2026-07-25)

**Purpose:** Claude usage hits ~100% ~Sat and resets **Tue 11 AM** (~2.5–3 days of Claude downtime). To make the wait
productive, the **consolidation dendritic operating-point sweep** (the design-gate's prescribed NEXT-(a)) runs on the
**free mini-PC pool** (pool40/41/42, CPU/numpy), DETACHED, WITHOUT this session. This doc lets ANY fresh post-reset
session collect + continue. The 3090 is OFF-limits during downtime (owner gaming/away); this lane is CPU-only + free.

## The science (why this sweep)
Confirmed boundary (`2026-07-25-consolidation-coactivation-...` + `-dendritic-surpass-DESIGN-...`): the co-activation
potentiation fix works (directional `ca1->slot` write) + the two-compartment bistable plateau ENGAGES, but at the
operating points tried the slots OVER-FIRE non-selectively — a CLIFF with NO per-slot **c_drive separation**
(`c_drive[slot_i|fact_i] ≈ c_drive[slot_j|fact_i]`, ratio ≈ 1.0). The design's NEXT-(a): a comprehensive
operating-point sweep to find a NARROW selective plateau (self_regen ↓ so it doesn't latch-all · lower slot_drive so
the write is only the strongly-co-active `ca1_i->slot_i` · stronger WTA), **measuring per-slot c_drive DIRECTLY**
(does slot_i get more drive than slot_j under fact_i's tag?) rather than inferring from the ignition cliff. NEXT-(b):
if NO operating point separates → the deeper dendritic **LINE/BUMP** attractor is the named next mechanism.

## What runs (grid + measurement)
Runner `research/runners/_consol_dendritic_opsweep.py` (numpy CPU backend; NO sim/ edit — pure config reuse of
`nmda_compositional_consolidation`). Grid = **480 configs** (self_regen{0,.05,.1,.15,.2} × k_thresh{2,3,4,5} ×
wta{5,10,20,40} × kir_g{1,3,5} × slot_drive{700,1400}) × **6 seeds** (42,43,44,100,101,102) = **2880 cells**. Each
cell = **dendritic arm + LINEAR control** (coincidence OFF, SAME potentiated wires → the load-bearing "plateau does the
work" check), 40 replay cycles. Per cell writes `research/findings/raw/consol_opsweep/op<NNN>_seed<S>.json` with:
- `cdrive`: per-slot g_coincidence + g_e under each fact's tag → **mean_ratio** (own/other) + **n_separated** (ratio>1.5)
- `ignition`: which slot each tag ignites → **selective** (fact i→slot i) / N, + **hold** after drive-off (bistable sig)
- `VERDICT.candidate` = dendritic separates (≥⌈N/2⌉) AND selective (≥⌈N/2⌉) AND beats the linear control.
Runtime: ~24–40 min/cell on 1 core; 36 workers (3×12); **~37 h normal**, worst-case-bounded **~60 h** (2700 s/cell
timeout) — inside the window. RESUME-SAFE (skips cells with existing JSON; incremental write preserves the dendritic
arm if the linear arm times out).

## Access + LAUNCH (run once, just before downtime)
- **Pool:** `ssh pool40|pool41|pool42` (user `node`, key `~/.ssh/id_ed25519`). Provisioned: `~/simvenv` (numpy 2.2.6,
  scipy 1.15.3, h5py, pyyaml) + code at `~/derisk-pool/sim`. Re-provision (idempotent) with `bash tools/pool_provision.sh`.
- **LAUNCH the sweep (detached, survives this session):**
  ```bash
  cd ~/Projects/sim
  bash tools/pool_provision.sh              # ensure code + venv current (re-syncs the runner)
  bash tools/pool_opsweep_dispatch.sh       # shards 2880 cells round-robin, launches 12 workers/node detached
  ```
- **Sentinel per node:** `~/derisk-pool/sim/research/findings/raw/consol_opsweep/QUEUE_DONE_<hostname>.txt`.

## CHECK progress (any session, mid-downtime)
```bash
cd ~/Projects/sim && bash tools/pool_opsweep_dispatch.sh --status   # per-node json count + candidate count + sentinel
```

## COLLECT + ANALYZE (on return, Tue)
```bash
cd ~/Projects/sim && bash tools/pool_opsweep_collect.sh   # rsyncs all JSON local + ranks candidates / top-by-separation
```
- **If candidates found** (some operating point separates + is selective + beats linear across ≥5/6 seeds): promote to a
  6-seed GO run + the FULL anti-cheat suite (no-replay / no-co-activation / apical-lesion / permuted-tag) + HOLD test →
  write the GO finding, the dendritic surpass CLOSES the selectivity boundary. Per THE LAW, verify-go first.
- **If NO candidate** (no point-plateau operating point robustly separates): this CONFIRMS the design's NEXT-(b) — the
  N-independent-point-plateau approach cannot give selective one-of-N; the deeper dendritic **LINE/BUMP** attractor (a
  graded moving bump over the slots, Ecker/continuous-attractor style) is the named next mechanism. A mapped boundary
  that LAUNCHES the next mechanism, not a wall.

## Files created this session (all committed)
- `research/runners/_consol_dendritic_opsweep.py` — the sweep driver (config-index isolated, dendritic+linear+cdrive).
- `tools/pool_provision.sh` — idempotent pool provisioning (apt venv/pip via passwordless sudo + numpy/scipy/h5py + code rsync).
- `tools/pool_opsweep_dispatch.sh` — shard + launch detached (+ `--status`).
- `tools/pool_opsweep_collect.sh` — rsync back + rank candidates.
- This manifest.

## THREE-LANE COMPUTE STATE (owner greenlit AWS + parallel 3090 use, 2026-07-25 ~07:30)
The owner clarified: the 3090 IS usable now (the downtime-only constraint was about Claude's absence, not the box); use
remaining Claude time + the 3090 in parallel; AWS is fine (free credits) — "best use for the credits." So THREE lanes run:

1. **mini-PC pool (CPU/numpy, FREE) — the downtime frontier sweep.** As above: 2880 cells consolidation dendritic
   operating-point sweep. LAUNCHED + verified (36 workers). Downtime backup / independent 6-seed confirmation.
   Status: `bash tools/pool_opsweep_dispatch.sh --status` · Collect: `bash tools/pool_opsweep_collect.sh`.

2. **3090 (GPU/CuPy) — the FAST preview of the SAME sweep, for a frontier read TODAY.** `SIM_BACKEND=cupy`, 480 configs @
   seed 42, 5 concurrent → ~3h → `research/findings/raw/consol_opsweep_gpu/op*_seed42.json`. ~2 min/cell (dendritic
   over-firing is launch-bound even on GPU). Gives the seed-42 answer (does ANY operating point separate?) while Claude is
   up, so the frontier decision (found-a-selective-plateau vs build-the-line/bump-attractor NEXT-(b)) can be made today.
   Analyze the same way as the pool collect. If seed-42 shows candidates, extend to 6 seeds on the GPU.

3. **AWS g5.xlarge (GPU) — the 267M LM width-training resume (the FLOP-bound run; gap#1 fluency scaffold).** The mission's
   spiking work is LAUNCH-bound (3090-better); the ANN LM is FLOP-bound (GPU-appropriate) → the best AWS use. Resumes
   `bridges/lmtrain/run4_d2048` (267M, d_model=2048/L16) from the local checkpoint (step 151k, best_val_nll 3.985 →
   val_ppl ~54, NOT converged — beats 83M run3's ~55 at matched tokens; capacity lever). Instance i-039987364d92e7792,
   `deploy/aws/aws_train.sh` manages it. On-demand (not spot → not reclaimed), tmux, untended-capable for the 2.5-day
   window. **⚠️ MUST-VERIFY: that training actually STARTS after the 12GB token + 3.2GB ckpt upload** (the AMI's pre-rsync
   `cd sim` torch-verify fails harmlessly; watch for the tmux `train session LIVE` + a falling val_ppl in progress.jsonl).
   Status: `bash deploy/aws/aws_train.sh status` · Collect: `bash deploy/aws/aws_train.sh collect` · **STOP (frees billing):
   `bash deploy/aws/aws_train.sh stop`** (collects ckpt + terminates + verifies no leftover — DO on return if still live).
