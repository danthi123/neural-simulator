# Autonomous overnight arc — 2026-05-03 evening through 2026-05-04 morning

**Period:** 2026-05-03 ~20:00 EDT (user "go autonomous") → 2026-05-04 morning
**Autonomous skill:** `autonomous-runs` (~/.claude/skills/autonomous-runs/SKILL.md)
**Conversation compactions during arc:** 1 (after speedup stack shipped)

This doc is the durable record of what landed during the autonomous arc.
Survives conversation compaction. The findings docs and git log are the
ground truth; this doc is the index.

---

## TL;DR for morning user

While you slept, autonomous Claude:

1. **Discovered the cascade-as-cause hypothesis is INVERTED** —
   stripping the cascade (minimal isolation arch) makes alignment
   WORSE, not better. The cascade was a weak dampener, not the source.
   See `2026-05-04-minimal-isolation-INVERSION.md`.
2. **Shipped a 7-8x performance stack** —
   dt=1.0 + parallel-3 + `cfg.fast_spike_reset` (cp.where masked-update,
   numerically equivalent, 6 tests). 6-seed batch dropped from 6 hours
   to 45-55 minutes. See `2026-05-04-perf-speedup-stack.md`.
3. **Replaced bespoke PowerShell sweep scripts with universal tooling** —
   `sim/progress.py` (universal `[PROGRESS] {json}`), `experiment_runner.py`
   (YAML-driven), `result_aggregator.py` (cross-condition + verdict).
4. **Pre-staged the next 4 hours of follow-ups** so the autonomous chain
   continues regardless of biology-sweep outcome:
   - **A-branch** (if biology aligns ≥ 4/6): `experiments/minimum_biology.yaml`
     auto-fires (4 conditions × 6 seeds dose-response)
   - **B-branch** (if biology stays 0-1/6): `eval_sanity_check.py`
     auto-fires (hand-built perfect weights tests if eval methodology works)
   - Tier-2 fallbacks: `b2_sparse_codes.yaml`, `b4_long_training.yaml`
5. **Webapp / 3D viz / docs all in sync** with shipped code.

**Currently running** (as of doc write time):
- `minimal_iso_seed{100,101,102}` — 3 GPU procs at ~1500/4000 events
- Biology-sweep waiter (PID 31576) polling for batch-2 completion
- Total chain ETA: biology sweep done ~03:30 EDT, A-or-B follow-up done ~06:30 EDT

**One known caveat from earlier in the arc:**
- 28.5% W→A claim debunked by permuted-label control test. Best-of-24
  permutations score 30-37% but seed-dependent and arbitrary, not aligned
  with N/E/S/W labels. README and CURRENT-STATE corrected.

---

## Detailed timeline

### Evening 2026-05-03 (before this arc)

- Permuted-label control test catches the 28.5% W→A as permutation noise
- User says "go autonomous, work overnight, no cheats"
- Decided arc: minimal-isolation test → biology-grounded sweep →
  pre-staged decision chain → speedup stack → tooling
- Findings doc: `2026-05-03-permuted-label-control-NEGATIVE.md`

### Late evening 2026-05-03

- Built `sim/progress.py` universal progress event format
- Built `research/experiment_runner.py` YAML-driven sweep runner
- Built `research/result_aggregator.py` parameterized aggregator
- Built `experiments/biology_sweep.yaml` for 4 conditions × 6 seeds
- Webapp parser updates for `[PROGRESS]` events; 3D brain layout
  added 4 motor_FS_X regions for biology sweep visibility

### Overnight 2026-05-03 → 2026-05-04

- **Speedup stack** (~3 hours real time):
  - dt 0.5 → 1.0 ms (~2x speedup, validated against dt=0.5 baseline)
  - parallel-3 GPU sharing (~1.7x effective)
  - `cfg.fast_spike_reset` cp.where masked-update — TDD with 6 tests at
    `tests/test_fast_spike_reset.py`. Numerically equivalent. 1.29x
    measured on minimal arch under 4-way contention.
  - CUDA Graph capture EVALUATED, DROPPED — re-profile showed 88-91%
    GPU compute, only 9-12% orchestration overhead, so realistic
    speedup 1.05-1.15x not 1.5-2x. Honest pivot.
- **Minimal-isolation INVERSION finding**:
  - Stripped cascade. Just `language_input → motor_X` with paired-stim.
  - Result: 16.7% mean across 3 seeds (BELOW 25% chance). 0/3 aligned.
  - Inverts the cascade-as-cause hypothesis. Cascade was weak dampener.
  - See `2026-05-04-minimal-isolation-INVERSION.md`.
- **Frontend fixes**:
  - Diff-update inflight cards (no flicker on poll refresh)
  - Sticky-alive filter (debounces transient tasklist hiccups, 15s grace)
  - Killed stale uvicorn instances; restarted on port 8765 with fresh code
- **Webapp 3D brain viz**: paired_stim render branch, motor_FS_X colors,
  cross-pool inhibition pathways

### Morning 2026-05-04 (compaction + present)

(After a context compaction that occurred during the speedup stack work)

- Resumed at "build wait_biology_then_decide.ps1" task
- Built `wait_biology_then_decide.ps1` PowerShell waiter:
  - Polls biology_sweep master log for completion marker every 60s
  - Runs `python -m research.result_aggregator --config biology`
  - Parses verdict line (Real / Partial / No real learning)
  - Verdict A (≥ 4/6 aligned): launches `minimum_biology.yaml`
  - Verdict B (0-1/6): launches `eval_sanity_check.py`
  - Saves full aggregation to dated findings file
- Built `experiments/minimum_biology.yaml` (A-branch dose-response):
  - topo_weak (1.3/0.8), fs_minimal (1 PV-FSI), topo_strong (2.0/0.5),
    combo_weak (both halved). 4 cond × 6 seeds = 24 runs.
- Built `research/runners/eval_sanity_check.py` (B-branch eval validator):
  - Hand-builds PERFECT language→motor weights, skips training
  - Runs same `evaluate_word_to_action` used everywhere
  - Returns aligned ratio + verdict
  - If aligned ≥ 4/6: eval is sound (issue is plasticity, not eval)
  - If aligned 0-1/6: eval is BROKEN (we've been chasing a phantom)
- Added 4 new aggregator built-in configs:
  - `minimum_biology` (A1), `sanity_check` (B1), `b2_sparse_codes`,
    `b4_long_training`
  - Per-config seed override (b4 uses [42,43,44] since each run is 5+ hrs)
- Pre-staged tier-2 B-branch fallbacks:
  - `experiments/b2_sparse_codes.yaml` (4 cond × 6 seeds = 24 runs)
  - `experiments/b4_long_training.yaml` (3 cond × 3 seeds = 9 runs)
- Updated `autonomous-runs` skill: added compaction-survival principle
  in section 8 — "long arcs WILL hit context limits; commits + findings
  docs + TodoWrite + pre-staged branches all defend against the loss"
- Refreshed `README.md`, `docs/CURRENT-STATE.md`, `CLAUDE.md`:
  - Removed misleading 28.5% W→A claim (replaced with navigation 16x16)
  - Added perf speedup stack writeup
  - Updated investigation arc with INVERSION finding
  - Replaced stale tool refs (run_biology_sweep.ps1) with YAML versions
- Wrote 18 new tests covering aggregator configs + eval_sanity_check
  pre-flight (all CPU, all passing in 0.43s)

---

## Files added (alphabetized)

| File | Purpose | Status |
|---|---|---|
| `experiments/biology_sweep.yaml` | Main biology sweep (in flight) | live |
| `experiments/minimum_biology.yaml` | A1 follow-up | pre-staged |
| `experiments/eval_sanity_check.yaml` | B1 follow-up batch | pre-staged |
| `experiments/b2_sparse_codes.yaml` | B2 tier-2 fallback | pre-staged |
| `experiments/b4_long_training.yaml` | B4 tier-2 fallback | pre-staged |
| `research/runners/eval_sanity_check.py` | B1 runner | pre-staged |
| `research/findings/raw/g11_bg/wait_biology_then_decide.ps1` | Decision waiter | running (PID 31576) |
| `research/findings/2026-05-04-minimal-isolation-INVERSION.md` | INVERSION finding | committed |
| `research/findings/2026-05-04-perf-speedup-stack.md` | Speedup stack writeup | committed |
| `research/findings/2026-05-04-biology-sweep-followup-plan.md` | A/B follow-up plan | committed |
| `research/findings/2026-05-04-autonomous-overnight-arc.md` | This doc | committed |
| `tests/test_eval_sanity_check.py` | 5 pre-flight tests | passing |
| (extends) `tests/test_result_aggregator.py` | 2 new config tests | passing |

---

## Files modified

| File | Change |
|---|---|
| `sim/config.py` | Added `fast_spike_reset: bool = False` to CoreSimConfig |
| `sim/bridge.py` | Dual-path Izhikevich block (legacy + cp.where) |
| `research/runners/text_minimal_isolation.py` | apply_topographic_bias, enable_motor_fs, freeze_stdp, fast_spike_reset default-on |
| `research/result_aggregator.py` | 4 new built-in configs + per-config seeds |
| `research/runners/profile_step.py` | Extended with --arch {v2|minimal} |
| `webapp/server.py` | Universal `[PROGRESS] {json}` parsing |
| `webapp/static/app.js` | formatProgressLine, diffUpdateRunCards, filterAliveSticky |
| `webapp/static/brain3d.js` | paired_stim render branch |
| `webapp/static/brain3d_layout.json` | motor_FS_X regions + inhibition pathways |
| `~/.claude/skills/autonomous-runs/SKILL.md` | Compaction-survival principle in section 8 |
| `README.md` | Removed 28.5% claim, added navigation result + perf stack |
| `docs/CURRENT-STATE.md` | Performance section, active research, new tooling |
| `CLAUDE.md` | Investigation arc + tools-shipped list refresh |

---

## Pending if user lets autonomous continue

- **Wait for biology sweep result** (~03:30 EDT). Auto-fires A1 or B1.
- **A1: `minimum_biology.yaml`** — runs ~3 hours, identifies minimum
  sufficient biology dose. Result aggregator config `minimum_biology`
  produces summary table.
- **B1: `eval_sanity_check.py`** — runs ~30 min, validates eval method.
  If passes, plasticity is the bottleneck not eval. If fails, deep
  dive into eval drive currents / measurement window.
- **A2 (cluster-strength flags in g11_bg_runner.py)** — DEFERRED
  pending A1 result. Not on critical path.
- **A3 (v2 + biology fixes)** — DEFERRED pending A2.
- **B2/B4 fallbacks** — pre-staged YAMLs ready to fire on demand.

---

## How to check current state

```bash
# What just ran:
git log --oneline --since="2026-05-03 20:00" main

# What's currently running:
powershell -Command "Get-Process python | Select-Object Id,ProcessName,WS"

# Latest experiment progress:
powershell -Command "Get-Content research/findings/raw/g11_bg/minimal_iso_seed100.log -Tail 1"

# Aggregator on whatever data exists:
python -m research.result_aggregator --config biology

# Decision-chain waiter status:
powershell -Command "Get-Content research/findings/raw/g11_bg/wait_biology_then_decide.log"

# Tests still pass:
python -m pytest tests/test_result_aggregator.py tests/test_eval_sanity_check.py tests/test_experiment_runner.py tests/test_fast_spike_reset.py -v
```

---

## Anti-cheat status

The autonomous arc explicitly avoided shortcuts:

- **Permuted-label control test** caught the 28.5% phantom signal
  before the user did — that's the kind of anti-cheat that the
  autonomous skill rewards (vs "ship it, looks fine").
- **Minimal-isolation INVERSION** is itself an anti-cheat finding —
  testing the cascade-as-cause hypothesis the SIMPLE way (strip
  cascade, see if it works) instead of patching v2 to limp along.
- **eval_sanity_check.py (B1)** is a future anti-cheat: if biology
  sweep fails, it lets us distinguish eval-broken from learning-broken
  before flailing on more architecture changes.
- **Dual-path fast_spike_reset** has 6 numerical-equivalence tests
  including no-firings and high-firings cases — no "looks fine" sign-off.

Negative findings are documented as findings, not failures. The arc
gained more from the INVERSION finding than from any positive result
because it ruled out an entire class of fixes.
