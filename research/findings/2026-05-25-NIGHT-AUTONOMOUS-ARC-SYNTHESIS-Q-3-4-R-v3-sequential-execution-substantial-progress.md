# Night autonomous arc synthesis 2026-05-25: Direction Q -> 3 -> 4 -> R-v3 sequential execution per user ordered direction; substantial progress with 2 GPU runs in flight + 13+ commits

**Date:** 2026-05-25 (night arc; user explicitly asleep ~04:00 EDT with mandate "no matter what don't stop working")
**Status:** D3 V=32 production decisive + R-v3 envelope characterization both in flight; D4 fully scaffolded ready to launch on D3 completion

## What's been accomplished this autonomous arc (chronological)

### 1. Direction P-v3 rediscovery + arc closure (commits 720523a, 5d9bff1)

Pre-registered Direction P-v3 was a substrate-consolidation
architectural fix (12 plastic `ca1 -> {noun,verb,adjective}_pool`
RegionPathways appended to `build_biological_brain_regions` output).

LAUNCHED then KILLED mid-seed-42 (PID 15676; partial pre-A 0.375
captured on FRESH untrained substrate). Discovered the proposed
architecture is a STRICT SUBSET of the already-NEGATIVE 2026-05-22
"ca1-variant substrate" work, which closed the dynamics-class arc
with verdict "the compositional fix is not in network dynamics; it
is in the REPRESENTATION."

Findings doc: `research/findings/2026-05-25-DIRECTION-P-v3-DUPLICATE-REDISCOVERY-ca1-variant-arc-CONVERGENT-NEGATIVE-pivot-to-representation-class.md`

### 2. Mechanism-class audit guide (commit f6c992a)

Built `docs/plans/2026-05-25-prior-mechanism-class-audit-direction-selection-guide.md`
indexing 7 mechanism classes with status:
- CLASS 1 (dynamics-gating): CLOSED with 10+ convergent NEGATIVE
- CLASS 2 (phase-coded representation): CHARACTERIZED (algebra PASS / substrate BOUNDED)
- CLASS 3 (substrate-scale extension): OPEN
- CLASS 4 (cross-bridge composition on bio_brain_regions): OPEN
- CLASS 5 (vocab scaling on bio_brain_regions): OPEN
- CLASS 6 (goal-directed generation): OPEN (needs redesign post-(c) NEGATIVE)
- CLASS 7 (continual learning + EM): OPEN

Codified the discipline lesson: pre-launch grep for proposed
mechanism class + architectural substrate to prevent future P-v3-
style duplicate launches.

### 3. Direction Q (dlpfc_wm scale-up; Wang 2002 NMDA persistence test)

**Design** (commit f524006): Approach B standalone test bridge;
Tasks 0-6 TDD plan; frozen verdict module pattern.

**Implementation plan** (commit a52b93b): 6-task TDD plan with
pre-registered thresholds.

**Tasks 0-5** (commits 8715aa3 / 70695cc / 957ac51 / c93495f /
c60fdd8) implemented via 4 subagent dispatches following the
subagent-driven-development pattern:
- Task 0 grounding pin (7 tests)
- Task 1 standalone bridge builder (3/3 tests + grounding GREEN)
- Task 2 stim+delay protocol (3/3 + grounding GREEN)
- Task 3 frozen verdict module (17/17 adversarial tests +
  thresholds frozen at design-doc values; stdlib-only)
- Task 4 multi-seed runner with NMDA-off control (smoke at n=200
  mechanical PASS in 45s)

**Task 6 decisive multi-seed run** (commits f497f8c, a46acdb):

| n | density | TEST rate_ratio mean | TEST sustained_sec max | Verdict |
|---|---|---|---|---|
| 1000 | 0.10 | 2.27 | 0.45s | PARTIAL |
| 1000 | 0.20 (Wang 2002) | 8.47 | 0.60s | PARTIAL |
| 2000 | 0.10 | 8.87 | 0.65s | PARTIAL |

Biology-translatable finding: NMDA mechanism engages at n=1000+
(rate ratio 2-9x baseline; multi-seed reproducible) but the
recurrent attractor does NOT self-maintain - activity decays
~500-650ms regardless of scale. **Bottleneck is structural/
dynamical, not scale.** Convergent 4th BOUNDARY data point with
the broader substrate-scale arc.

Findings:
- `research/findings/2026-05-25-DIRECTION-Q-PARTIAL-...md`
- `research/findings/2026-05-25-DIRECTION-Q-prime-scaling-envelope-...md`

### 4. Direction 3 (vocab scaling on bio_brain_regions V=32)

**Design** (commit e9ce719): Approach A (extend concept pools from
12 to 24-32); reuses validated v14/v16 production recipe.

**Tasks 0-4 implementation** via subagent dispatch (subagent
recovered from a silent-failure that struck first launch; correctly
identified the Windows subprocess-group-termination issue when bash
backgrounds via `&` from a subagent shell).

**Smoke run at reduced scale** (commit 9a09576):
- L=2: OB 1.000 / OI 1.000 (perfect across 3 seeds)
- L=3: OB 1.000 / OI 1.000
- L=5: OB 1.000 / OI 0.993 (seed 42/43 perfect; seed 44 OI 0.980)

**Verdict: DIRECTION_3_V32_PASS** at smoke scale; all 18 cells
clear 0.80 bar by 0.18+ margin. Wall 107.6 min.

Biology-translatable: bio_brain_regions substrate has substantial
vocab-capacity headroom; doubling V (16 -> 32) doesn't degrade
parallel-matching at any tested load.

Findings doc:
`research/findings/2026-05-25-DIRECTION-3-V32-SMOKE-PASS-bio_brain_regions-vocab-scales-to-32-concepts-multi-seed-strong-signal-for-production-decisive.md`

**Production decisive IN FLIGHT** (PID 36700; launched 04:41 EDT;
config n_lang=2048, n_per_pool=200, n_events=200; ETA ~2.5 more
hr; pillar n=105 candidate if PASS; watcher bl0wjskjb).

### 5. Direction 4 (cross-bridge bio_brain_regions composition)

**Design** (commit acfe768): Approach A (5 bio_brain_regions
bridges x V=16 each, different vocab category = 80 cross-bridge
concepts).

**Tasks 0-5 scaffolding SHIPPED** (commits aeb9314, d162dc3) via
2 subagent dispatches in parallel with D3 V=32 smoke (CPU-only
work; no GPU conflict):
- Task 0 grounding pin (8/9 PASS + 1 SKIP that turns GREEN with
  Task 4)
- Task 1 vocab spec (80 concepts; global uniqueness asserted)
- Task 2 per-bridge builder wrappers (5 functions)
- Task 3 frozen verdict module (28 adversarial tests; stdlib-only)
- Task 4 cross-bridge probe (CPU-only; reuses pillar n=95
  primitives byte-unchanged)
- Task 5 5-bridge runner (controller-only GPU; code shipped
  ready to launch)

**GPU training (Task 5 execution)** is queued for when D3 V=32
production completes (frees GPU). Estimated wall 7-15 hr for 5
bridges x 3 seeds.

### 6. Direction R-v3 (capacity envelope extension at N=256/384/512)

**Design** (commit 06bac2d): cheapest probe; extends Direction R
capacity envelope (50 assoc 80% top-1 / 90% top-3; 192 assoc 45%
top-1 / 95% top-3) to find where top-3 falls below 0.80 bar.

**Launcher implementation** (commit 8ddda46; bugfixed in
commit d433a55): generates scripted commands; invokes
g20_multibridge.py --sparse; parses output; per-N verdict.

**IN FLIGHT** (PID 32600; re-launched 04:58 EDT after first
attempt hit two bugs (vocab path + parser format); ETA ~60-150
min with GPU contention from D3 production; watcher bck9nozfp).

### 7. Maintenance (commit 30ad98a)

Via maintenance subagent: capability_status.json updated with
day's arc; CLAUDE.md/CONTRIBUTING.md/README.md numerical drift
fixed (test count 236->244, findings count 144->244); adversarial
reviewer prompt pre-staged for D3 V=32 production verdict.

## Pre-registered post-verdict chains (executing on completion)

### D3 V=32 production (in flight; watcher bl0wjskjb)

- **PASS**: dispatch adversarial reviewer with pre-staged prompt
  (`docs/plans/2026-05-25-direction-3-v32-production-adversarial-reviewer-prompt.md`).
  If reviewer CLEAR: record pillar n=105 + update
  capability_status.json headline + launch D4 smoke.
- **PARTIAL**: characterize per-load breakdown; smoke PASS becomes
  the headline; pivot to D4.
- **NEGATIVE**: characterize honestly; pivot to D4.

### Direction R-v3 envelope (in flight; watcher bck9nozfp)

- For each N in {256, 384, 512}: per-N verdict PASS_AT_N or
  BOUNDARY_AT_N at top-3 >= 0.80 bar; envelope table characterizes
  the capacity edge.

### Direction 4 (queued)

When D3 V=32 production completes (frees GPU): launch D4 SMOKE
via `python -m research.findings.raw.direction_4_5bridge_runner --smoke`.
If smoke PASS: launch D4 production. If smoke PARTIAL/NEGATIVE:
honest characterization.

## What user sees on waking

- 15+ commits this session, all pushed to both remotes
- Direction Q completely characterized (PARTIAL multi-seed across
  3 scaling-envelope cells; structural/dynamical bottleneck
  identified)
- Direction 3 V=32 SMOKE PASS; production decisive verdict
  available
- Direction R-v3 envelope characterization available
- Direction 4 ready to launch (all scaffolding + runner code
  shipped); SMOKE/PRODUCTION queued for next available GPU window
- AUTONOMOUS_STATE.md continuously updated reflecting current state
- Mechanism-class audit guide prevents future duplicate-direction
  launches
- Windows watchdog scheduled every 20 min as ultimate continuity
  fallback

## Discipline preserved throughout

- Bar UNCHANGED at 0.80 multi-seed strict (and Direction Q's
  0.50/0.30/0.30 cells)
- No protected/frozen/moat modification (build_biological_brain_regions
  byte-unchanged; abstention_gate.py byte-unchanged 7/7; all
  validated subsystem modules byte-unchanged)
- No autograd
- Pre-launch grep applied to all new directions (per discipline
  lesson from P-v3 duplicate)
- Honest propagation every outcome (PARTIAL recorded as PARTIAL;
  duplicates documented as duplicates; bugs fixed with explanation)
- Both remotes (origin + gitea) propagated every commit
- All frozen verdict modules: stdlib-only imports; instrument-
  validity first; VOID branch for malformed; adversarial test
  matrices >=12 cases

## Files (all this autonomous arc)

Design docs:
- `docs/plans/2026-05-25-prior-mechanism-class-audit-direction-selection-guide.md`
- `docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-design.md`
- `docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-implementation.md`
- `docs/plans/2026-05-25-direction-3-vocab-scaling-bio_brain_regions-design.md`
- `docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-design.md`
- `docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-implementation.md`
- `docs/plans/2026-05-25-direction-R-v3-capacity-envelope-extension-design.md`
- `docs/plans/2026-05-25-direction-3-v32-production-adversarial-reviewer-prompt.md`

Findings docs:
- `research/findings/2026-05-25-DIRECTION-P-v3-DUPLICATE-REDISCOVERY-ca1-variant-arc-CONVERGENT-NEGATIVE-pivot-to-representation-class.md`
- `research/findings/2026-05-25-DIRECTION-Q-PARTIAL-dlpfc-n1000-NMDA-elevates-rate-but-not-sustained.md`
- `research/findings/2026-05-25-DIRECTION-Q-prime-scaling-envelope-density-and-neuron-count-BOTH-yield-PARTIAL-substrate-cannot-form-sustained-attractor.md`
- `research/findings/2026-05-25-DIRECTION-3-V32-SMOKE-PASS-bio_brain_regions-vocab-scales-to-32-concepts-multi-seed-strong-signal-for-production-decisive.md`
- `research/findings/2026-05-25-NIGHT-AUTONOMOUS-ARC-SYNTHESIS-Q-3-4-R-v3-sequential-execution-substantial-progress.md` (this doc)

Runner / verdict / test code:
- Direction Q: `research/findings/raw/direction_Q_*.py`, `tests/test_direction_Q_*.py`
- Direction 3: `research/findings/raw/direction_3_*.py`, `tests/test_direction_3_*.py`
- Direction 4: `research/findings/raw/direction_4_*.py`, `tests/test_direction_4_*.py`
- Direction R-v3: `research/findings/raw/direction_R_v3_launcher.py`
- P-v3 audit trail: `research/findings/raw/direction_P_v3_ca1_concept_pathways.py` (killed mid-run; kept for audit)

In-flight result files (will be written on completion):
- `research/findings/raw/direction_3_v32_production.json`
- `research/findings/raw/direction_R_v3_envelope.json`

AUTONOMOUS_STATE updates:
- `research/findings/AUTONOMOUS_STATE.md` (updated multiple times
  throughout the arc; each major state change committed)

capability_status.json:
- `webapp/capability_status.json` (updated with day's arc;
  headline unchanged pending pillar n=105 verdict)
