# g11_bg_runner Troubleshooting & Gotchas

A collection of subtle issues that have bitten this project across multiple
sessions. Read before diving into experiments.

## Statistical confidence

### 3-seed claims are unreliable
Multiple times this project has had 3-seed results flip on 6-seed validation.
**Always validate with 6+ seeds before claiming a config beats baseline.**

Examples:
- Asymmetric adaptive DA "33% improvement" (3-seed) → 11% non-significant on 6 seeds
- Multi-goal partial freeze "5.9% improvement" (3-seed) → 0% on 6 seeds (exactly tied)
- Trajectory replay "looked promising" (3-seed) → neutral on full task

Standard 6-seed protocol: seeds 42, 43, 44, 100, 101, 102.

## Plasticity gotchas

### `cfg.stdp_w_max` must be above design weights
STDP is **soft-bound**: `Δw_LTP = A_plus * (w_max - w) * exp(...)`. When
`weight_mean > stdp_w_max`, every "LTP" event is strongly negative and weights
collapse to w_max within ms.

The g11 runner sets `cfg.stdp_w_max = 30.0` to support cortex→D1 weight_mean=25.
If you add a new pathway with weight_mean > 30, weights collapse silently.
**Either lower the weight or raise stdp_w_max.**

### Plasticity gates affect ALL plasticity sources
When you set `bridge.set_plasticity_gate("name", 0.0)`, it freezes:
- STDP weight delta
- Eligibility-trace accumulation
- Reward modulation update
- Hebbian potentiation
- Hebbian decay
- Synaptic scaling

This is intentional (truly "frozen" pathway) but be aware that even slow
homeostatic processes are halted.

### Initial gain is 1.0 even before tagged
When a pathway is tagged with `plasticity_gate="name"`, its initial gain is
1.0 (full plasticity). The runner must explicitly call
`set_plasticity_gate("name", 0.0)` if it wants to start frozen.

## Curriculum gotchas

### `--curriculum` requires `--hippocampus` for the gates to activate
The gates `hippo_to_cortex` and `sensory_to_cortex` only exist when their
respective regions exist. Without those flags, curriculum has no gates to
control and effectively does nothing.

### `--goal-schedule curriculum` flips goal AT the end of warmup
Default behavior: with `--curriculum-warmup-steps 600 --goal-schedule curriculum`,
goal flips at step max(1200, warmup+600) = 1200. So:
- 0-599: warmup, goal=(6,6)
- 600-1199: phase 2 (hippo learning), goal still=(6,6) — hippo overfits
- 1200-1799: goal=(1,6)

Without `--goal-schedule curriculum`, default schedule flips at step 300, so
the agent sees both goals during warmup. The two schedules give very
different learning dynamics.

### Partial freeze (`--curriculum-phase2-cortex-gain 0.2`) generalizes better
Full freeze (gain=0.0) is slightly better on 2-goal but loses on multi-goal
(curriculum doesn't help fast-change). Partial freeze (gain=0.2) is roughly
tied with full freeze on 2-goal AND less catastrophic on multi-goal.

For research: partial freeze is more general. For peak 2-goal: full freeze
gives a marginal edge.

## Cortex WTA gotchas

### Don't combine WTA with curriculum
Cortex WTA (`--cortex-wta`) on the breakthrough config (sensory+hippo+curriculum)
hurts performance by ~50% (sum 4.72 → 8.87). The heuristic provides cortex
selectivity natively; WTA on top adds commitment penalty.

### Motor WTA tradeoffs
`--motor-lateral-inhibition` PARTIAL — exploitation+, readaptation−. Net
negative when stacked with adaptive DA. Don't use unless task is pure
single-goal exploitation.

## Heuristic gotchas

### The heuristic is NOT removable with current architecture
The heuristic-off test (heuristic_decay_after_step) consistently shows the
agent collapses to random walk when heuristic is removed, even after 4200
steps of plastic learning. The plastic input layers (sensory, hippo) cannot
generate enough cortex selectivity to replace the heuristic without
architectural changes (sparse encoding, LTD for inactive pathways, real
perception).

This is an open arc — see `docs/plans/2026-04-27-perception-arc-plan.md`
for the multi-week project to address it.

### Stronger plastic input weights don't fix it
Tried sensory_to_cortex_weight = 10, 25, 50. All produced identical collapse
when heuristic is removed. The issue is selectivity (1-of-4 cortex pool drive),
not magnitude.

## Sleep gotchas

### Sleep replay is currently neutral on this task
- Random replay: neutral (3.91 vs no-sleep 3.87)
- Trajectory replay (full log): slightly worse (4.32, biased by stale entries)
- Trajectory replay (recency-bounded 200): neutral (3.96)

The 2100-step task with 300-step sleep window doesn't reward consolidation —
the wake test isn't long enough for compound benefits. Sleep replay
infrastructure works correctly; needs different task structure to demonstrate
benefit.

### Agent freezes during sleep (intentional)
During the sleep_replay_after_step window, `new_x, new_y = x, y` (no movement).
This is intentional (NREM sleep features behavioral suspension) but means
distance metrics during sleep are flat.

## Scaling gotchas

### Recipe is tuned for 8×8
Default `--curriculum-warmup-steps 600`, `--n-steps 1800` are tuned for the
8×8 grid. On 16×16:
- Random walk baseline rises to ~10 (Manhattan diameter ~28 vs 14 on 8×8)
- Sparser learning per place cell (256 cells vs 64, same training time)
- Recipe gets sum 5.26, baseline gets 4.44 — recipe LOSES at 16×16

For 16×16+: needs longer training, possibly larger initial weights, possibly
denser place cell coverage. Re-tuning required.

## Reward gotchas

### Reward sparsity
Default reward is +1 if Manhattan distance decreased, -1 if increased, 0 if
same. This is sparse and discrete. Continuous distance-shaped reward might
help but hasn't been tested rigorously.

### Reward EMA decay matters
For adaptive DA, `--adaptive-da-ema-decay-negative 0.7` (faster decay on
negative reward) is the validated setting. Without this, asymmetric DA
performance varies seed-to-seed.

## PFC gotchas

### PFC region needs goal_cells
The PFC pathways are:
- `goal_cells → PFC` (only added if `enable_hippocampus=True`)
- `PFC → cortex_X`

Without hippocampus, PFC has no input from goal info. The runner adds the
goal_cells→PFC pathway only if both `enable_pfc` and `enable_hippocampus`
are true.

### PFC seed 42 is the worst seed for PFC
In 6-seed validation, seed 42 is the WORST PFC config (sum 4.83) but a GOOD
no-PFC sensory config (sum 3.23). Other seeds (43, 44, 100, 101, 102) tell
the opposite story. Don't draw conclusions from seed 42 alone for
PFC-related comparisons.

## File organization gotchas

### `g11_bg_runner.py` is now ~1500 lines
The single-file runner has accumulated significant complexity. Major sections:
1. `build_bg_brain_regions()` (~250 lines) — region + pathway declarations
2. `run_moving_goal_episode()` setup (~200 lines) — config, brain init
3. Trial loop (~600 lines) — per-step gate updates, drives, motor selection,
   reward, plasticity
4. CLI parsing + main() (~300 lines)

Future refactoring opportunity: split into `g11_build.py`, `g11_train.py`,
`g11_cli.py`. Currently the single-file structure is OK but getting unwieldy.

### Memory files in `~/.claude/projects/<workspace>/memory/`
Auto-memory for this project includes:
- `MEMORY.md` (index)
- `project_phase_c_resolved.md`
- `project_remaining_cheats.md`
- `project_next_priorities.md`
- `project_plasticity_gating_infra.md`
- `project_pfc_working_memory.md`
- `feedback_6seed_validation.md`
- (older entries from earlier sessions)

These persist across conversations. Update when key facts change.

## When in doubt

1. **Read `research/findings/INDEX.md`** for the chronological research arc.
2. **Read the most recent finding for the topic you're working on** — they have
   per-seed data and statistical analysis.
3. **Run a smoke test** (`--n-steps 100`) before any 1800-step acid test.
4. **Always 6-seed validate** before claiming a config beats baseline.
