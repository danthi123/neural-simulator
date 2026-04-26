# Research Findings Index

Session-by-session log of the research arc on this codebase. Negative
results are first-class findings — they're how the architecture got de-risked.
Each finding includes verdict (GO / NO-GO / PARTIAL / NEGATIVE) and the
runner / commit that produced it.

For chronological + thematic context, see [`docs/SCIENCE_ROADMAP.md`](../../docs/SCIENCE_ROADMAP.md)
(in particular [Pillar 4: Reward-Driven Learning Architecture](../../docs/SCIENCE_ROADMAP.md#pillar-4-reward-driven-learning-architecture)).

## At a glance

| Date | Finding | Verdict |
|------|---------|---------|
| 2026-04-26 | [Learned perception — cold-start fail](2026-04-26-learned-perception-cold-start-fail.md) | NEGATIVE — sensory→cortex doesn't bootstrap from random in 1800 trials |
| 2026-04-26 | [**Asymmetric adaptive DA**](2026-04-26-asymmetric-adaptive-da.md) | **GO (new best)** — sum=3.53 (-33% vs baseline), phase 1 gap nearly closed |
| 2026-04-26 | [Adaptive per-action DA targeting](2026-04-26-adaptive-da-targeting.md) | **GO** — best phase 0 (1.85) AND best total (3.99); adapts via reward EMA |
| 2026-04-26 | [Per-action dopamine targeting — same trade-off as WTA](2026-04-26-per-action-da-mixed.md) | PARTIAL — same exploitation+/readaptation− pattern; opt-in |
| 2026-04-26 | [Motor WTA lateral inhibition — mixed result](2026-04-26-wta-lateral-inhibition-mixed.md) | PARTIAL — kept opt-in (exploitation+, readaptation−) |
| 2026-04-25 | [RNG drift from IZH preset additions](2026-04-25-rng-drift-from-izh-presets.md) | benign drift; locked-baseline relocked 170→149 |
| 2026-04-25 | [Phase B acid test — REAL win](2026-04-25-phase-b-acid-test-real-win.md) | **GO** — phase 1 finalQ 1.76 vs G9 baseline 6.74 |
| 2026-04-25 | [Phase B cascade-stability fix (n_cortex)](2026-04-25-phase-b-cascade-stability-fix.md) | bug fixed |
| 2026-04-25 | [Phase B honest correction](2026-04-25-phase-b-honest-correction.md) | retracted intermediate claim |
| 2026-04-25 | [Phase B BG acid test — initial (overstated)](2026-04-25-phase-b-bg-acid-test.md) | trail kept |
| 2026-04-25 | [HH presets after per-gate Q10 fix](2026-04-25-hh-presets-after-q10-fix.md) | 15/17 fire APs |
| 2026-04-25 | [HH temperature bug](2026-04-25-hh-temperature-bug.md) | MAJOR bug fixed |
| 2026-04-25 | [HH preset audit](2026-04-25-hh-preset-audit.md) | 11/17 broken |
| 2026-04-25 | [Izhikevich preset audit](2026-04-25-izh-preset-audit.md) | 2/7 work, 5/7 silent fallback |
| 2026-04-25 | [PFC bistability — NEGATIVE](2026-04-25-pfc-bistability-negative.md) | NEGATIVE; pivoted |
| 2026-04-25 | [Session G motor exploration](2026-04-25-session-g-motor-exploration.md) | partial |
| 2026-04-24 | [Session E.1 neuromodulator subsystem](2026-04-24-session-e1-neuromodulator-subsystem.md) | framework GO; NE-on-trap NO-GO |
| 2026-04-24 | [Session D Part C: Pavlovian at scale](2026-04-24-session-d-part-c.md) | GO |
| 2026-04-24 | [Session D Part A: gate redesign](2026-04-24-session-d-part-a.md) | gate retired |
| 2026-04-24 | [Session C: synaptic-gain neuromod](2026-04-24-session-c.md) | shelved |
| 2026-04-24 | [Route B profile](2026-04-24-route-b-profile.md) | optimisation arc |
| 2026-04-24 | [RNG drift](2026-04-24-rng-drift.md) | infra lockdown |
| 2026-04-24 | [G9 sim-native R-STDP](2026-04-24-g9.md) | NO-GO at runner-side |
| 2026-04-24 | [G8 goal-context probe](2026-04-24-g8.md) | diagnostic |
| 2026-04-21 | [Signed-eligibility branch](2026-04-21-signed-eligibility-branch.md) | review package |
| 2026-04-21 | [G7 moving-goal readaptation](2026-04-21-g7.md) | NO-GO (3 variants) |
| 2026-04-21 | [G6 2D gridworld](2026-04-21-g6.md) | PARTIAL |
| 2026-04-21 | [G5.v3 signed perceptron](2026-04-21-g5v3.md) | **GO** |
| 2026-04-20 | [G5.v2 reward-modulated loop](2026-04-20-g5v2.md) | NO-GO |
| 2026-04-20 | [G5 sensorimotor closed loop](2026-04-20-g5.md) | GO (weak form) |
| 2026-04-20 | [G3 persistence](2026-04-20-g3.md) | **GO** |
| 2026-04-20 | [G2 sim-local plasticity](2026-04-20-g2.md) | NO-GO |
| 2026-04-20 | [G1 encoder-decoder roundtrip](2026-04-20-g1.md) | **GO** (v3) |

## By theme

### Reward-driven learning architecture (Pillar 4)
The main research arc. Started with G1 minimum-viable pipeline, hit the
silent-motor trap on G6/G7, tried 7 runner-side variants in Sessions D–I
(all NEGATIVE), then resolved structurally with Phase B BG cascade.

- **Frontier resolved (2026-04-25):** [Phase B acid test — REAL win](2026-04-25-phase-b-acid-test-real-win.md).
- **The silent-motor trap arc:** [G7](2026-04-21-g7.md) → [Session C](2026-04-24-session-c.md) → [G8](2026-04-24-g8.md) → [G9](2026-04-24-g9.md) → [Session D.A](2026-04-24-session-d-part-a.md) → [Session E.1](2026-04-24-session-e1-neuromodulator-subsystem.md) → [PFC bistability NEGATIVE](2026-04-25-pfc-bistability-negative.md) → [Session G motor exploration](2026-04-25-session-g-motor-exploration.md) → resolved by Phase B.
- **Phase B trail:** [BG acid test (initial, overstated)](2026-04-25-phase-b-bg-acid-test.md) → [honest correction](2026-04-25-phase-b-honest-correction.md) → [cascade-stability fix (n_cortex bug)](2026-04-25-phase-b-cascade-stability-fix.md) → [REAL win after two-bug fix](2026-04-25-phase-b-acid-test-real-win.md).

### Biology preset audit (Phase A, 2026-04-25)
A full audit of every neuron-model preset surfaced a major Q10 bug and
rebuilt the preset library. 30 working biological presets across HH + Izh + AdEx.

- [HH temperature bug (root cause)](2026-04-25-hh-temperature-bug.md)
- [HH preset audit (11/17 broken)](2026-04-25-hh-preset-audit.md)
- [HH presets after per-gate Q10 fix (15/17 work)](2026-04-25-hh-presets-after-q10-fix.md)
- [Izhikevich preset audit](2026-04-25-izh-preset-audit.md)

### Research-gate progression (G1–G7)
Initial 1D / encoder-decoder / sensorimotor gates. Established the
runner framework and the gate-criterion approach. Some gates are PARTIAL
because their metrics misfire even when the underlying learning works.

- [G1 — encoder-decoder roundtrip](2026-04-20-g1.md) (GO, v3)
- [G2 — STDP local learning](2026-04-20-g2.md) (NO-GO)
- [G3 — persistence](2026-04-20-g3.md) (GO)
- [G5 — sensorimotor closed loop](2026-04-20-g5.md) (GO weak form)
- [G5.v2 — reward-modulated loop](2026-04-20-g5v2.md) (NO-GO; unsigned eligibility ceiling)
- [G5.v3 — signed perceptron](2026-04-21-g5v3.md) (GO; LR decay decisive on seed 44)
- [G6 — 2D gridworld](2026-04-21-g6.md) (PARTIAL; metric misfires)
- [G7 — moving-goal readaptation](2026-04-21-g7.md) (NO-GO; 3 variants)
- [Signed-eligibility branch](2026-04-21-signed-eligibility-branch.md) (sim-side fix proposal)

### Subsystem frameworks
Composable opt-in subsystems extracted from earlier ad-hoc mechanisms.

- [Session E.1 — neuromodulator subsystem](2026-04-24-session-e1-neuromodulator-subsystem.md) (framework GO)
- (Session E.2 — brain-region framework: design in [`docs/plans/2026-04-24-brain-region-framework.md`](../../docs/plans/2026-04-24-brain-region-framework.md); merged 2026-04-24)

### Negative results (architectural ceilings)
Closing the search space matters as much as opening it.

- [G2 STDP NO-GO](2026-04-20-g2.md)
- [G5.v2 unsigned-eligibility NO-GO](2026-04-20-g5v2.md)
- [G7 moving-goal NO-GO (3 variants)](2026-04-21-g7.md)
- [G9 sim-native R-STDP NO-GO at runner-side](2026-04-24-g9.md)
- [PFC bistability NEGATIVE](2026-04-25-pfc-bistability-negative.md)
- [Session C synaptic-gain shelved](2026-04-24-session-c.md)

### Infrastructure
- [RNG drift lockdown](2026-04-24-rng-drift.md) (gamma benchmark variance fixed)
- [Route B profile](2026-04-24-route-b-profile.md) (where inner-loop time goes)
- [G8 goal-context probe](2026-04-24-g8.md) (diagnostic only)
- [Session D.A gate redesign](2026-04-24-session-d-part-a.md) (Q1→Q4 metric retired)
- [Session D.C Pavlovian at scale](2026-04-24-session-d-part-c.md) (canonical biology demonstration)

## Conventions

- Each finding starts with **Date**, **Gate** (or Session/Branch), **Verdict**.
- Raw data lives in `research/findings/raw/<gate>/`, indexed from the finding.
- Findings cite the runner (`research/runners/...`) and the commit hash.
- Negative findings are kept (don't delete; the trail of attempts is the
  research). When a finding is later contradicted or superseded, write a
  new finding and link both ways — don't edit history.
- Filenames: `YYYY-MM-DD-<short-tag>.md`.
