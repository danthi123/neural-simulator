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
| 2026-04-29 | [Catalog-driven remediation pass](../../docs/plans/2026-04-29-catalog-remediation-pass.md) | **REMEDIATION (11 of 12 done; 1 design-doc deferred)** — 13 sim-level corrections from Kandel-6e + supplemental texts. Bug-tier (R1): per-region E_inh override (MSN −60, SNc DA −55), FSI cross-action WTA (replaces anatomically-backwards MSN→MSN). Architectural (R3): GPe PV+/PV− split, striosome/matrix split, SNr→SNc disinhibition, sparse cortex→MSN, NaP+SK+Ih GPi tuning, D1/D2 neuropeptide arms (dynorphin/SP/enkephalin via from_region_firing rule). Naming (R2): striatal interneuron taxonomy clarification, asymmetric aversive reward (0.5× magnitude). Deferred: MSN KIR2/Kv2 kernel. 340 tests pass post-remediation. |
| 2026-04-28 | [GPU throughput investigation](2026-04-28-throughput-investigation.md) | **INFRA** — concurrency knee at 4-6, MPS Linux-only, motor-counting code fix REVERTED, webapp ships `--progress-print-interval=20` default for non-interactive launches. |
| 2026-04-28 | [Cheat #5 v3 + v3.1 results](2026-04-28-cheat5-v3-results.md) | **v3 GO / v3.1 NO-GO** — v3 lateral inhibition (`--bg-lateral-inhibition`) 4.26 ± 0.50 (no regression, permanent default). v3.1 cross-projections layered on v3 still breaks phase-2 readaptation (8.92 ± 2.44). |
| 2026-04-28 | [Cheat #5 v4 — developmental pretraining](2026-04-28-cheat5-v4-results.md) | **NO-GO** — v4 (5K pretraining + freeze) is *worse* than v3.1 single-goal: 3-seed 11.34 ± 1.85, P0 4.88, P1 6.46. Originally framed as "closed by design"; later reframed (see post-v4 doc below). |
| 2026-04-28 | [Cheat #5 post-v4 reframe](2026-04-28-cheat5-post-v4-reframe.md) | **REFRAME — ON HOLD pending biology buildout.** After multi-goal eval correction + option 1 NO-GO (22.46) + patch-matrix HIGH-VARIANCE PARTIAL (8.76 ± 2.54, seed 44 hit 5.88 beating baseline), cross-projections are not fundamentally broken — they're under-constrained. Cluster-by-cluster biology buildout (B → A → C → D → E) is the path forward; closure-by-design framing was too quick. |
| 2026-04-28 | [Cluster B.1 — D1/D2 plasticity asymmetry](2026-04-28-cluster-b1-d1d2-asymmetry-results.md) | **PARTIAL SIGNAL — first cluster-buildout evidence.** Patch-matrix + B.1 multi-goal: 7.62 ± 1.23 (n=3) vs patch-matrix alone 8.76 ± 2.54. Variance halved (std 2.54 → 1.23), Phase 2 catastrophe eliminated (P2 mean 3.36 → 1.92, std 2.09 → 0.77). Mean still 7% above v3 baseline (7.08); cheat-5 not fully closed by B.1 alone. Biology probe PASS (`research/probes/d1_d2_asymmetry_probe.py`: D1↑/D2↓ under +reward, inverted under −reward). Continuing to Cluster B.2 (FSIs) + B.3 (TANs). |
| 2026-04-28 | [Cluster B.2 — striatal FSIs](2026-04-28-cluster-b2-striatal-fsis-results.md) | **MIXED.** Mean 8.44 ± 0.62 with cross-projections (n=3); variance keeps dropping (1.23 → 0.62) but mean slightly worse than B.1 alone (7.62). Phase-decomposed: P1+P2+P3 (4.72) BEATS v3 baseline (4.89) — steady-state action selection improves. Phase 0 is broken (3.72 vs 1.83) because FSIs broadcast inhibition before agent commits to winner. Architectural issue: real FSIs have tonic baseline + burst dynamics + high-pass filtering on cortex drive; our model has none. Initial weight 8.0 gave catastrophic 19.78 sum; retuned to 2.0 gives 9.50 (no cross) / 8.44 (cross). Proceeding to B.3 per unit-cluster strategy; if full cluster still bad, retune cortex_to_str_fs_weight 30→10. |
| 2026-04-28 | [Cluster B.3 — cholinergic TANs](2026-04-28-cluster-b3-tans-results.md) | **NULL on cheat-5 + INFRASTRUCTURE.** Implementation correct (47 tests pass, biology probe PASS) but TAN-on vs TAN-off statistically neutral at n=3 multi-goal (B.1+B.2 alone 18.02 vs +TANs 18.59; patch-matrix variants 15.18 vs 14.83). Why no-op: gate fires inside reward-modulation block which is skipped when reward=0 (between rewards); at reward steps `pause_on_reward` drops ACh → gate ≈ 1 (no suppression). Real TAN function requires tonic DA-driven plasticity for ACh to gate; our model has only phasic DA. Real win retained: bridge step-order bug fix (`59dc1fc`) — `manager.step()` now runs BEFORE reward modulation. Methodology finding: multi-goal benchmark regressed at seed 42 (v3 7.08 → 12.05; bisect proves predates B.3 changes). Cluster B done (3/3) — pivoting to Cluster A (closed BG loop). |
| 2026-04-28 | [Flagship 4.08 — data labeling correction](2026-04-28-flagship-4.08-data-correction.md) | **CORRECTION** — the 4.08 result is real; the prior finding referenced wrong filenames. Headline + recipe stand. |
| 2026-04-28 | [Cheat #5 v2 — Zero-Init Cross-Projections](2026-04-28-cheat5-v2-NEGATIVE.md) | **NEGATIVE** — 3-seed mean 7.89. Phase 0 fixed (mean 2.49, structural integrity restored), but phase 1 destroyed (mean 5.40) by thaw-time STDP corruption of converged policy. Diagnosis: missing winner-take-all biology (MSN lateral inhibition). v3 adds it. |
| 2026-04-28 | [Cheat #5 v1 — Curriculum-Staged Cross-Projections](2026-04-28-cheat5-v1-NEGATIVE.md) | **NEGATIVE** — 3-seed mean 10.87. Plasticity gate freezes learning but not synaptic transmission; cross-projection synapses with weight=5.0 disrupted BG disinhibition from step 0. |
| 2026-04-27/28 | [**🎉🎉🎉🎉 NEW BEST: 4 of 5 Cheats Closed**](2026-04-27-NEW-BEST-4cheats-closed.md) | **GO (CURRENT FLAGSHIP)** — 6/6 seeds, sum 4.08 vs baseline 5.88, **p=0.00045**, **30.6% improvement**. Adds sensed reward (beacon-intensity gradient) on top of full perception arc. **Biology-grounded (4.08) BEATS cheats-allowed (4.41).** Cheat #5 (BG cross-projections) tested — NEGATIVE, kept opt-in. |
| 2026-04-27 | [**🎉🎉🎉 Item 1: FULL PERCEPTION ARC COMPLETE**](2026-04-27-FULL-PERCEPTION-ARC-COMPLETE.md) | **GO** — 6/6 seeds, sum 4.56 vs baseline 5.88, p=0.00819, 22.4% improvement. **Agent navigates with NO direct (gx,gy) AND NO direct (x,y) AND NO heuristic — all 3 major perception cheats closed.** Superseded by ★ above (which adds sensed reward as cheat #4). |
| 2026-04-27 | [Item 1 Stage 3: full perception (no goal coords + no heuristic)](2026-04-27-stage3-full-perception-BREAKTHROUGH.md) | GO — 6/6 seeds, sum 4.77, p=0.00188. Closes 2 of 3 perception cheats; superseded by Stage 2 (which adds landmark place cells). |
| 2026-04-27 | [Item 1 Stage 1: Goal-beacon perception (6-seed)](2026-04-27-stage1-beacon-perception.md) | **PARTIAL GO** — 6-seed: 5/6 beat baseline (5.36 vs 5.88, p=0.34). Closes goal-cell coordinate cheat. Direction-positive but not significant alone. |
| 2026-04-27 | [PFC Stage 2: delayed-response 3-seed](2026-04-27-pfc-stage2-delayed-response.md) | PARTIAL — 3-seed PFC drop 17% smaller than no-PFC during goal silence (3.48 vs 4.19, d=0.73, p=0.51). Direction supports working memory; significance needs more seeds. |
| 2026-04-27 | [**🎉 PFC working memory**](2026-04-27-pfc-working-memory.md) | **GO** — 5/6 seeds (4.41 sum, p=0.018, 25% over baseline). Adds recurrent prefrontal region with goal_cells → PFC → cortex pathways. Statistically significant improvement over prior best (4.63). Superseded as flagship by perception arc + sensed reward (4.08). |
| 2026-04-27 | [Perception cheats investigation](2026-04-27-perception-cheats-investigation.md) | NEGATIVE — simple weight tuning (sensory_to_cortex_weight 10→25→50) doesn't enable heuristic-free navigation. Removing the heuristic requires architectural changes (sparse encoding, LTD for inactive pathways, real perception). Multi-week scope. |
| 2026-04-27 | [16×16 spatial scaling test](2026-04-27-16x16-scaling.md) | PARTIAL — architecture scales cleanly (1251 neurons / 70K synapses) but recipe tuned for 8×8 underperforms baseline at 16×16 (5.26 vs 4.44). Re-tuning needed for larger grids. CLI flags --grid-size and --n-hippocampus-per-layer added. |
| 2026-04-27 | [Sleep-replay infrastructure](2026-04-27-sleep-replay-infrastructure.md) | PARTIAL — sleep-replay infrastructure works correctly (gates fire, agent freezes); random replay neutral (sum 3.91 vs no-sleep 3.87). Future: trajectory replay of learned sequences. |
| 2026-04-27 | [**📋 Overnight summary**](2026-04-27-overnight-summary.md) | **MILESTONE** — plastic-input-layer arc closed; major infrastructure (per-pathway gating, NM-driven gates, real curriculum) landed. ~14h autonomous work, 8 commits, all pushed. |
| 2026-04-27 | [Task-adaptive curriculum](2026-04-27-task-adaptive-curriculum.md) | **GO** — partial freeze (gain=0.2) generalizes across slow-change (2-goal: 4.79, 5/6) and fast-change (4-goal: 7.83, beats baseline 8.32 by 5.9%). Single hyperparameter controls task adaptation. NM-driven plasticity gates landed too. |
| 2026-04-27 | [Sensory layer additive (multi-input)](2026-04-27-perception-additive.md) | **GO** — sensory + hippo + curriculum: 5/6 seeds beat baseline (4.63 vs 5.88, p=0.05). Three plastic input layers now compose. Heuristic-off test confirms inputs augment, don't replace, the heuristic (biologically expected). |
| 2026-04-27 | [**🎉 Plastic-input-layer arc RESOLVED**](2026-04-27-plastic-input-layer-RESOLVED.md) | **GO** — per-pathway plasticity gating + real curriculum + no WTA: 6/6 seeds beat baseline (sum 4.72 vs 5.88, p=0.02). First time ever with a plastic input layer. Hippocampus genuinely learns place→action mapping. |
| 2026-04-26 | [Curriculum learning (drive-gated)](2026-04-26-curriculum-fail.md) | NEGATIVE — suppressing hippo drive during warmup doesn't crack the plastic-input-layer ceiling; true curriculum needs bridge-level staged plasticity |
| 2026-04-26 | [Cortex WTA + adaDA + hippo combo](2026-04-26-cortex-wta-adapda-combo.md) | PARTIAL — adaDA provides ~14% over WTA+hippo (9.26 → 8.01) but combo still 1.36× worse than baseline; closes off "more flags will fix it" approach |
| 2026-04-26 | [Cortex-level WTA — selectivity fix works but readaptation breaks](2026-04-26-cortex-wta.md) | PARTIAL — fixes plastic-input-layer cold-start (16% over hippo-alone) but introduces motor-WTA-style readaptation penalty; combo with adaptive DA is the next logical test |
| 2026-04-26 | [Hippocampal module additive — same cold-start pattern](2026-04-26-hippocampus-additive-fail.md) | NEGATIVE — fourth consecutive plastic-input-layer failure; closes off the search space until curriculum or cortex-WTA is added |
| 2026-04-26 | [Pavlovian conditioning demo](2026-04-26-pavlovian-demo.md) | **GO** — clean classical conditioning, CS rate 5.56→16.32 Hz |
| 2026-04-26 | [Informed-init perception — doesn't solve cold-start](2026-04-26-informed-init-perception-fail.md) | NEGATIVE — directional prior helps marginally but BG cascade needs cleaner cortex selectivity than graded sensory→cortex provides |
| 2026-04-26 | [**6-seed correction**](2026-04-26-six-seed-correction.md) | **CORRECTION** — asym DA win was overstated; LR boost is actually best on 6 seeds (4.92 vs baseline 5.88) |
| 2026-04-26 | [**Surprise-boosted learning rate**](2026-04-26-surprise-lr-boost.md) | **GO** — most robust across task types (4.02 on 2-goal, 9.11 on multi-goal) |
| 2026-04-26 | [Multi-goal stress test reverses asym adaDA win](2026-04-26-multi-goal-stress-test.md) | CONDITIONAL — asym adaDA HURTS on fast-changing tasks (4-goal: 9.97 vs baseline 8.32) |
| 2026-04-26 | [Night summary](2026-04-26-night-summary.md) | overview of overnight session |
| 2026-04-26 | [DA-gated WTA — still net negative](2026-04-26-da-gated-wta.md) | NEGATIVE — adaptive WTA scaling can't rescue WTA on this task; asym adaDA alone wins |
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
- **Plastic-input-layer arc (4 consecutive NEGATIVE — same cold-start mode, then RESOLVED 2026-04-27):**
  - [Learned perception cold-start fail](2026-04-26-learned-perception-cold-start-fail.md)
  - [Informed-init perception fail](2026-04-26-informed-init-perception-fail.md)
  - [Hippocampus additive fail](2026-04-26-hippocampus-additive-fail.md)
  - [Cortex WTA partial](2026-04-26-cortex-wta.md), [WTA + adaDA combo](2026-04-26-cortex-wta-adapda-combo.md), [Drive-gated curriculum fail](2026-04-26-curriculum-fail.md) — three more partial/negative attempts.
  - **[Plastic-input-layer RESOLVED](2026-04-27-plastic-input-layer-RESOLVED.md)** — per-pathway plasticity gating + real curriculum + no WTA: 6/6 seeds beat baseline (4.72 vs 5.88, p=0.02). The architectural ceiling is now broken.

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
