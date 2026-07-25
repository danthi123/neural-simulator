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

## NOT running on AWS (deliberate)
The free mini-PC lane covers the frontier consolidation run; no AWS GPU spend is needed for it. AWS (`claude-ec2-driver`,
us-east-1, keys `~/.ssh/aws-train/`) remains available if the owner wants an ADDITIONAL GPU-bound run (e.g. the sweep at
larger N, or LM training) — that needs a separate scoping + launch.
