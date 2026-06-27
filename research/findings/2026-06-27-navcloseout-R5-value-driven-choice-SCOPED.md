# nav close-out R5 — a VALUE-DRIVEN-CHOICE task proves the spiking value-critic LOAD-BEARING BY ITS FUNCTION (2026-06-27)

**Status:** BUILT + CPU R1-a gate **GO 3/3 seeds** (all 4 gates pass). The nav-embodied R1-b form is SCAFFOLDED;
its GPU eval is **FLAGGED FOR THE CONTROLLER** below. Reuse-by-import, **NO `sim/` edit** (`git diff HEAD -- sim/`
empty), the no-confab moat preserved BY CONSTRUCTION (the decision organ has NO RF/conversational slices).

**Research gate (read first):** `research/findings/2026-06-27-nav-value-loadbearing-research-gate.md` (SHA
`1a0cac04`, on `main`) — RANK 1 = a value-DRIVEN-CHOICE task transplanting the project's EXISTING spiking
value-WTA, the genuine fix for the R4 value-IRRELEVANT confound.

---

## 0. The one-paragraph result

R4 (`2026-06-27-navcloseout-R4-delayed-reward-value-task-NEGATIVE.md`) delayed the reward on a SINGLE moving goal
whose optimal action ("reduce distance to the one goal") is UNCHANGED by reward timing → the value V was a
PASSENGER, not a limiter (the deep-research gate confirmed: all six 2×2 arms in a 2.48–2.69 band; value-OFF within
±0.11 of value-ON everywhere; the 415/388-synapse GABA_B lesion LANDED but moved nothing). That is the
`feedback_validate_signal_by_its_function` failure mode — the value-ON/OFF A/B run on a task where V is not
load-bearing BY CONSTRUCTION — NOT a substrate limit. **R5's RANK-1 fix is a value-DRIVEN-CHOICE task** (two
options of DIFFERENT learned value; picking the higher-value option REQUIRES V), so the value is the ONLY signal
that can drive the correct choice and the lesion is FORCED to be load-bearing-or-not. The CPU-first gate (R1-a)
transplants the project's existing spiking value-driven WTA (`_value_salience_appraisal_derisk.SpikingSpeakAccumulator`,
speak-vs-silence → option-A-vs-option-B) and is **GO 3/3 seeds**: value-ON picks the higher-value option at **0.90**
(≫ chance 0.50); the value lesion **COLLAPSES** it to **0.461 ≈ chance** (drop 0.439) — *here the lesion HAS
something to collapse, the R4 fix*; the **EQUAL-value control** (the validate-by-function discriminator R4 LACKED)
shows the lesion is **NEUTRAL** (intact-vs-lesion choice agreement **1.000** — when value carries no gradient the
lesion changes nothing → the effect is value-gradient-SPECIFIC, not a general lesion artifact); and the
**permuted-value** control collapses to **0.461 ≈ chance**. The catalog says this binary value-choice (O.22 +
O.19 + C.34 + L.41) is THE canonical way to make a value signal load-bearing, and the prior evidence (V-A trace
6/6 GO + this existing spiking value-WTA) predicted the GO.

---

## 1. THE BUILD (RANK 1 from the gate)

**Runner:** `research/runners/_navcloseout_R5_value_driven_choice.py` (~95% reuse-by-import; NO `sim/` edit).

### R1-a (CPU-first, THIS module's GATE) — the spiking value-WTA-CHOICE mechanism in isolation

A two-option (extensible to N) **spiking value-driven WTA**, transplanting `_value_salience_appraisal_derisk.py`'s
`SpikingSpeakAccumulator` (the speak-vs-silence Izhikevich WTA whose pool DRIFT = a candidate's WORTH = f(DA-value),
with the value lesion that reverts it to baseline) → an **option-A-vs-option-B** choice:
- Two option pools `opt_0`/`opt_1` = Wang-2002 NMDA integrators; a shared `wta_fs` FS pool gives biased
  competition (each option drives the FS; the FS inhibits all options) = soft-WTA (the GO sel/commit/OPN template).
- `drift(option_i) = base + value_gain · VALUE_i + salience_gain · SALIENCE_i` (catalog O.19/C.34: **value
  modulates the accumulator DRIFT**).
- The DECISION = whichever option pool wins the spiking race (a neural pool's FIRING, **NOT** a host argmax over
  drives — the brain-based-only requirement). The DV: **CORRECT = pick the HIGHER-value option.**
- The OU-noise realization is HELD FIXED per drive-vector (snapshot/restore the global RNG, seeded from the rounded
  drives — the appraisal probe's pattern), so the decision is a deterministic FUNCTION of the drives → the
  value-vs-lesion comparison is a clean ABLATION.
- The **VALUE** axis is a CPU stand-in for the merged-bridge spiking SNc/striosome_value critic, seeded from a
  reward-tagging RNG **structurally DISTINCT** from a per-option **SALIENCE** bias (corr(value, salience) ≈ 0,
  asserted) so "value drives the choice" is **not circular** (the GPU R1-b reads the real shared `dopamine`).

**The value LESION (drive-level matched, the faithful biological ablation):** the lesion replaces each option's
VALUE with the **MEAN value over the options** (`value_gain · mean(values)`, applied uniformly) instead of the
per-option value. This removes the option-SPECIFIC value CONTRAST that grades the SNc burst across options (the
GABA_B value-DIFFERENTIAL) while holding the TONIC DA level (the mean) and thus the operating point fixed. ⇒ the
lesion removes ONLY the value GRADIENT: when the gradient is the sole differentiator (distinct-value HEADLINE) the
choice collapses to chance; when there is no gradient (EQUAL-value control) the lesion is byte-identical to intact
→ a clean validate-by-function isolation. (The naïve "drop the value term entirely" lesion FAILED the EQUAL-value
discriminator — choice-agreement only 0.62–0.77 — because dropping the whole term lowered the operating point and
the soft-WTA's marginal decisions shifted with the overall drive level; that was a *measurement* artifact, not a
genuine value-specificity failure. The drive-level-matched lesion removes the gradient cleanly and the
discriminator now reads agreement **1.000**. Documented honestly here.)

### R1-b (the nav-embodied form) — SCAFFOLD ONLY, FLAGGED FOR THE CONTROLLER

Two simultaneous beacons/goals of DIFFERENT value on the grid; the BG action selector (the spiking sel/commit/OPN
decision, default-on) must approach the HIGHER-value goal (the missing O.22 Q(s,a) read-out the catalog flags).
`r1b_two_beacon_kwargs(...)` assembles the `run_moving_goal_episode` kwargs (the spiking-default merged nav config,
grid-32). **`run_moving_goal_episode` is CuPy-only**, so the full episode is the GPU eval (below). The honest scope
(per the gate): R1-b is the higher-variance arm — R1-a (CPU, the value-WTA in isolation) is the decisive proof; the
nav read-out is the controller's optional confirmation.

---

## 2. THE R1-a GATE — GO 3/3 seeds (the decisive CPU mechanism result, run inline)

`SIM_BACKEND=numpy python -m research.runners._navcloseout_R5_value_driven_choice --r1a --seeds 42,43,44`
(11.4 s total, 3 seeds, n_options=2, 60 trials/seed; JSON `research/findings/raw/navcloseout_R5/R5_r1a.json`).

| seed | acc value-INTACT | acc value-LESION | lesion drop | EQUAL-value agreement | acc PERMUTED | corr(val,sal) |
|---|---|---|---|---|---|---|
| 42 | 0.917 | 0.467 | 0.450 | **1.000** | 0.483 | −0.126 |
| 43 | 0.883 | 0.350 | 0.533 | **1.000** | 0.417 | +0.053 |
| 44 | 0.900 | 0.567 | 0.333 | **1.000** | 0.483 | +0.157 |
| **mean** | **0.900** (min 0.883) | **0.461** | **0.439** | **1.000** | **0.461** | (|max| 0.157) |

chance = 0.500.

**The 4 gates (validate-by-function), ALL 3/3:**
- **(G_HEADLINE)** value-ON picks the higher-value option ABOVE chance: **3/3** (0.90 ≫ 0.50; ≥0.20 above-chance).
- **(G_LESION, the headline anti-cheat)** the value lesion COLLAPSES the high-value choice to ~chance (the EXTRA
  correct choices VANISH): **3/3** (drop 0.439 ≥ 0.20; lesion 0.461 within ±0.15 of chance). **This is the R4 fix
  — here the lesion HAS a value-driven choice to collapse, unlike R4's value-irrelevant task.**
- **(G_DISCRIM, the validate-by-function control R4 LACKED)** the EQUAL-value task (V(A)=V(B)): the lesion is
  NEUTRAL (intact-vs-lesion trial-by-trial choice agreement **1.000** ≥ 0.80): **3/3**. ⇒ the lesion's G_LESION
  effect is value-GRADIENT-SPECIFIC, NOT a general lesion artifact (when value can't help, the lesion does nothing).
- **(G_PERMUTE)** permuting the option↔value contingency collapses the advantage to ~chance: **3/3** (0.461 ≈ 0.50).
- **NON-CIRCULAR (value distinct from salience):** **3/3** (|corr(value, salience)| ≤ 0.157 ≤ 0.35).
- **MOAT preserved BY CONSTRUCTION:** **3/3** (no RF/conversational slices on this decision organ).

**Verdict: GO** — the spiking value is load-bearing BY ITS FUNCTION on a value-driven choice; the value lesion
collapses the higher-value choice (G_LESION), the equal-value control discriminates it as value-SPECIFIC
(G_DISCRIM), and permuting the contingency vanishes the advantage (G_PERMUTE). This directly answers the R4
question (the value WAS load-bearing once the task exercises its choice function).

---

## 3. THE CPU SMOKE — PASS (well-formedness)

`SIM_BACKEND=numpy python -m research.runners._navcloseout_R5_value_driven_choice --smoke` → **PASS** (7 checks):
(a) value↔salience corr −0.074 (non-circular); (b) the equal-value vector is FLAT, the distinct-value vector
varies; (c) the lesion removes the GRADIENT cleanly (drop == `value_gain·(value − mean)`, the lesion value-term
FLAT across options; intact favors the high-value option, lesion favors salience); (d) the permuted control
preserves the value multiset but breaks the contingency (n=2 swap); (e) the SPIKING value-WTA picks the
higher-drive pool ([90,260]→opt_1, [260,90]→opt_0) and has NO RF slices (moat by construction); (f) the verdict
aggregator returns GO on a synthetic GO row-set, `HONEST_NEGATIVE_lesion_does_not_collapse` on an R4-like row-set,
and `HONEST_NEGATIVE_lesion_artifact_not_value_specific` on an artifact row-set; (g) the R1-b nav scaffold kwargs
are well-formed (spiking-default merged nav config, grid-32).

---

## 4. ⚑ FOR CONTROLLER TO RUN — the R1-b nav two-beacon GPU eval (optional confirmation; R1-a is the decisive gate)

R1-a (CPU, §2) already proves the spiking value-WTA-choice is load-bearing by its function. R1-b is the
nav-embodied read-out (the missing O.22 Q(s,a)), the higher-variance arm per the gate. It is **CuPy-only** and is
the controller's call. Two RUNNER-SIDE wirings (NO `sim/` edit) of a two-beacon value-choice on
`run_moving_goal_episode`:

**Wiring (the controller picks one; both reuse the existing per-trial `homeostatic_hook` / `enable_beacon_perception`
— no `sim/` edit):**
- **(i) differential-reward two-goal** (recommended): a per-trial `homeostatic_hook` that (a) sets the "goal"
  override to whichever of the two beacons the agent is nearer (reusing the existing `{"goal": ...}` per-trial
  override `run_moving_goal_episode` already supports), and (b) returns a HIGHER reward magnitude when the agent
  reaches the HIGH-value beacon vs the LOW-value beacon. The value (which beacon is worth more) enters through the
  reward magnitude → the critic learns V(high) > V(low) → value-ON should reach/prefer the HIGH-value beacon.
- **(ii) two-beacon-perception**: `enable_beacon_perception=True` with two beacons of different `beacon_max_intensity`
  proxying value (the existing two-sensor path).

**The de-risk (mirror the R1-a gates on the nav score — `mean_distance_to_high_value_goal` LOWER better / high-value
reward harvested HIGHER better; grid-32, 6 seeds for the variable effect):**
- **G_HEADLINE_nav:** value-ON reaches/prefers the HIGH-value goal ABOVE the value-OFF (lesioned) arm and above the
  EQUAL-value baseline.
- **G_LESION_nav (the headline):** the value lesion (`make_value_lesion_hook` from
  `_navcloseout_R4_delayed_reward_value.py` — the established `cp_gabab_synapse_mask` zeroing, `prebuilt_post_init_hook`;
  `n_gabab_cut`>0 confirms it landed) **COLLAPSES** the high-value preference (the agent reverts to the nearer /
  salience-default goal). Must DROP, not merely shift.
- **G_DISCRIM_nav (the validate-by-function control R4 LACKED):** an EQUAL-value two-beacon control (both goals same
  reward magnitude) → the lesion is NEUTRAL (no high-value preference to lose; value-ON ≈ value-OFF). Proves the
  G_LESION_nav effect is value-SPECIFIC.
- **G_PERMUTE_nav:** permute which beacon is high-value vs the trained contingency → the preference advantage vanishes.
- **AC_MOAT:** if run on the merged agent, re-assert `check_moat` (the nav `cp_*` arrays are array-disjoint from the
  composer's `cp_rf_w_re/im` → preserved by construction). This standalone scaffold builds the nav bridge only.
- **AC_REGIME:** deterministic regime faithfulness; **grid-32 (NEVER grid-8** — the documented false-GO scale); **6
  seeds** for the variable effect.

**EXACT GPU command(s) FOR THE CONTROLLER (after the controller wires (i) or (ii) into a small `run_arm` like R4's;
the scaffold-check confirms the kwargs are well-formed today):**

```bash
# 0) confirm the scaffold kwargs (CPU, no bridge) — already PASS:
SIM_BACKEND=numpy python -m research.runners._navcloseout_R5_value_driven_choice --r1b-scaffold-check

# 1) the R1-a gate at 6 seeds (CPU, fast — the decisive value-WTA proof; promote 3-seed GO to 6-seed):
SIM_BACKEND=numpy python -m research.runners._navcloseout_R5_value_driven_choice \
    --r1a --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/navcloseout_R5/R5_r1a_6seed.json

# 2) (optional nav confirmation) the R1-b two-beacon value-choice on the real bridge, per seed (GPU):
#    — the controller adds the two-beacon homeostatic_hook + the value-lesion arm to r1b_two_beacon_kwargs
#      (reusing make_value_lesion_hook from _navcloseout_R4_delayed_reward_value.py), then runs the 4 arms
#      { value-ON-distinct, value-OFF-distinct(lesion), value-ON-equal, value-permuted } × seeds 42..102 at
#      grid-32, n_steps 1800. The de-risk = the G_*_nav gates above.
```

**De-risk criteria (GO):** across the seeds (3-seed indicator → 6-seed for any generalization claim, per
`feedback_6seed_validation`): R1-a all 4 gates pass at 6 seeds; (optional) R1-b G_HEADLINE_nav + G_LESION_nav +
G_DISCRIM_nav + G_PERMUTE_nav pass at grid-32, 6 seeds. **A GO is the likely outcome** (R1-a is already GO 3/3; the
V-A trace is 6/6 GO; the spiking value-WTA is pre-validated). A R1-b NEGATIVE that survives its own SURPASS round
would localize the genuine *instrumental* value-choice-via-the-BG-cascade boundary (the legitimate juncture for the
deferred dendritic question) — but R1-a having isolated the value-WTA as GO makes that a BG-cascade-readout issue,
not a value-mechanism one.

**DISCIPLINE NOTE (the stall lesson):** this build + the CPU R1-a gate were run INLINE (CPU, ~11 s). The R1-b nav
GPU eval is NOT run here and is NOT backgrounded — it is the controller's to run (a subagent cannot resume on
background-completion).

---

## 5. Anti-cheats (this is the 2nd attempt at this exact question — R4 failed a validate-by-function confound)

- **value-lesion COLLAPSES, not shifts** (G_LESION): lesion → 0.461 ≈ chance (drop 0.439). The R4 fix: the lesion
  has a value-driven choice to collapse.
- **EQUAL-value discriminator** (G_DISCRIM, the control R4 LACKED): equal value → lesion NEUTRAL (choice agreement
  1.000). This is the validate-by-function control — it proves the lesion effect is value-SPECIFIC, not a general
  lesion artifact. The drive-level-matched lesion (removes the gradient, holds the operating point) makes this
  exact.
- **permuted-value** (G_PERMUTE): shuffle option↔value → advantage vanishes (0.461 ≈ chance) → the headline is the
  value STRUCTURE, not a fixed pool bias.
- **non-circular value** (asserted): corr(value, salience) ≤ 0.157 ≤ 0.35 → the value is NOT a relabeled salience.
- **MOAT by construction**: the decision organ has NO RF/conversational slices (`cp_rf_w_re/im` None) →
  array-disjoint from any composer; the no-confab moat is preserved by construction.
- **grid-32** is the R1-b nav verdict scale (NEVER grid-8); R1-a is the value-WTA in isolation (vocab-free).

---

## 6. NO `sim/` edit + the no-confab MOAT

- **NO `sim/` edit.** `git diff HEAD -- sim/` is empty. The runner is built entirely from `BrainRegion` /
  `RegionPathway` / `SimulationBridge` (the same primitives `g11_bg_runner` uses) + reuse of
  `_value_salience_appraisal_derisk`'s spiking-WTA pattern + `_navcloseout_R4_delayed_reward_value`'s value-lesion
  hook (for R1-b). All runner-side.
- **The no-confab moat is preserved BY CONSTRUCTION.** The value-WTA decision bridge has NO conversational/RF
  slices (`has_rf_slices` asserted False each seed). The nav `cp_*` arrays (R1-b) are array-disjoint from the
  composer's `cp_rf_w_re/im` → preserved by construction (re-assert `check_moat` if R1-b runs on the merged agent).

---

## 7. Deliverables

- **Runner:** `research/runners/_navcloseout_R5_value_driven_choice.py` (the value-WTA-choice R1-a gate + the R1-b
  nav scaffold + the CPU smoke).
- **JSON:** `research/findings/raw/navcloseout_R5/R5_r1a.json` (R1-a GO 3/3, seeds 42/43/44).
- **This findings doc** (the scope + the GO result + the EXACT GPU command + de-risk criteria FLAGGED FOR
  CONTROLLER).

---

## 8. Sources

- **The deep-research gate (the RANK-1 plan):** `research/findings/2026-06-27-nav-value-loadbearing-research-gate.md`
  (`1a0cac04`).
- **The R4 value-IRRELEVANT NEGATIVE (the confound this fixes):**
  `2026-06-27-navcloseout-R4-delayed-reward-value-task-NEGATIVE.md` + `-SCOPED.md`.
- **The V-A proof the value is load-bearing BY FUNCTION (mirror the lesion + control design):**
  `2026-06-21-shortcut9-trace-conditioning-value-derisk.md` (`1a861f87`) — lesion collapses the trace CR 6/6 on
  real spikes; the value-irrelevant DELAY control survives 6/6.
- **The transplanted spiking value-WTA + value lesion:** `research/runners/_value_salience_appraisal_derisk.py`
  (the `SpikingSpeakAccumulator` Izhikevich WTA; drift = f(DA-value); the value-pin lesion).
- **The value-lesion hook (R1-b):** `research/runners/_navcloseout_R4_delayed_reward_value.py`
  (`make_value_lesion_hook` — the established `cp_gabab_synapse_mask` zeroing).
- **The nav episode + spiking decision:** `run_moving_goal_episode` (`g11_bg_runner.py:3256`); the spiking
  sel/commit/OPN read-out (`readout_source="spiking_wta"`, default-on, `2026-06-19-spiking-decision-default-on-GO.md`).
- **Catalog (the value-driven-choice paradigm):** **O.22** striatal action-value coding → action-value MSNs feed an
  argmax-with-noise selector (the missing Q(s,a) read-out) · **O.19** vmPFC/OFC subjective value modulates the
  accumulator DRIFT (same drift-diffusion math as perceptual choice) · **C.34** DA codes utility/risk → "critical
  for two-arm bandit / binary-gamble / delay-discounting choice tasks" · **L.41** reward = goal-object for approach +
  economic choice → "binary-choice tasks revealing transitive subjective preferences" · **G.16/G.59** drift-diffusion
  bounded accumulation "holds for perceptual AND value-based judgments".
- **Literature:** Schultz 2016 (striatal action value + DA utility); Samejima et al. 2005 (action-value neurons);
  Lau & Glimcher 2008, Ito & Doya 2009 (matching-law action values); Padoa-Schioppa (OFC economic choice);
  Sugrue-Corrado-Newsome (matching); Wang 2002 (NMDA decision attractor); Lo & Wang 2006 (the commit threshold);
  Usher & McClelland 2001 (the leaky competing accumulator); Berridge (incentive salience "wanting"); Sutton & Barto
  2e Ch 11 (actor-critic).

_The value-WTA-choice gate is the brain-based decision (a neural pool's FIRING, NOT a host argmax). The value axis
is a CPU stand-in for the merged-bridge spiking SNc/striosome critic (the GPU R1-b reads the real shared
`dopamine`). NO `sim/` edit; reuse-by-import; the no-confab moat is array-disjoint from the nav cascade and
untouched. grid-32 is the R1-b nav verdict scale (never grid-8)._
