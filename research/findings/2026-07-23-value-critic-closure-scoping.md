# Value-critic closure scoping — 🟨→done on the project's terms (fully spiking, one brain, biology, no host shortcut) (2026-07-23)

**Type:** READ-ONLY scoping (deep-research gate style). NO code written, NO `sim/` edit, NO experiments / GPU run
(a training run + a CPU de-risk are live). Single deliverable = this doc; it PROPOSES, it does not implement. Every
load-bearing claim below was verified against the actual source + the R5 raw JSONs + the finding text.

---

## 0. TL;DR — the value critic is 🟨 for a *demonstration* reason, not a *computation* reason

**The precise state, which the "🟨 partial" label blurs:** value is ALREADY computed on the spiking substrate — the
`striosome_value` critic LEARNS `V(s)` by DA-gated STDP (neurons + synapses), its slow GABA_B/GIRK conductance
SUBTRACTS `V` at the SNc membrane, and the SNc's FIRING **is** `delta = r − V` (the RPE), routed to the actor's
plasticity as the signed third factor via the `dopamine` modulator. That pathway is spiking end-to-end and is
already the deployed nav default (`--spiking-snc --enable-neural-critic --spiking-reward-us`).

**What is NOT closed — the two residual shortcuts, both in the LOAD-BEARING DEMONSTRATION, not the mechanism:**
1. **`sim/td_value_critic.py` is a HOST numpy TD(λ) critic** — `w = w + ALPHA·step·e`, `delta = r + γ·V(s') − V(s)`
   all in numpy (`:81`,`:86`). Confirmed host-computed. **BUT it is a REFERENCE/CEILING, not a deployed nav
   shortcut** — it is imported only by `td_critic_gate.py`, `research/run_pavlovian_parallel.py`, and the
   `test_*_no_harm.py` anti-cheat comparisons; it is NOT in the nav value pathway. Its host-ness is by design (it is
   the Schultz-98 ceiling the spiking version is measured against). So the thing to "close" is **not** the TD
   arithmetic in this file.
2. **The R5 GO proof (`2026-06-27-navcloseout-R5-value-driven-choice-GO.md`) used a HOST value STAND-IN.** R5-R1a
   proved a *spiking value-driven WTA* is load-bearing by its function (6/6 seeds) — but the VALUE scalar fed into
   the WTA was `build_concept_value` (`_value_salience_appraisal_derisk.py:114`), explicitly *"a CPU stand-in for
   the merged-bridge spiking SNc/striosome_value critic, seeded from a reward-tagging RNG."* So the DECISION organ is
   spiking, but the VALUE it reads is host-injected. The on-substrate form (**R1-b**, which reads the REAL shared
   spiking `dopamine`/`striosome_value` off the merged bridge) is **SCAFFOLD ONLY — NEVER RUN** (confirmed:
   `research/findings/raw/navcloseout_R5/` holds only `R5_r1a*.json`; no R1-b artifact exists).

**⇒ VERDICT: CLOSEABLE-CHEAPLY.** The mechanism (spiking V-learning + spiking `r−V` + DA routing) exists and is
validated piecewise. The residual is a **demonstration gap**: (a) wire R5's spiking value-WTA to read the REAL
LEARNED spiking `V` (retire the host stand-in), (b) add an **untrained-critic** anti-cheat so the critic's *learning*
is load-bearing (not just a host tag), and (c) generalize past the single 2-option R5 task. All of (a)/(b) are
CPU-first, reuse-by-import, NO `sim/` edit. The GPU nav read-out (R1-b) is the optional higher-variance confirmation.

---

## 1. DIAGNOSIS — host vs already-spiking in the value pathway (which exact computation is the residual)

| Computation | Status | Where |
|---|---|---|
| `V(s)` LEARNED on the substrate | **SPIKING** — `striosome_value` MSN-D1/RS critic; the plastic `vs_place_context→striosome_value` weight grows by pair-then-reward **DA-gated STDP** (three-factor + eligibility) | `_merged_navcritic_valuetrain.run_value_train` (port of `g11_bg_runner._run_place_value_training`) |
| `V` read as a firing rate | **SPIKING** — the critic's firing rate = `V` (read from `cp_firing_states`) | `_merged_navcritic_valuetrain._critic_rate_via_afferent` |
| `delta = r − V` (the RPE) | **SPIKING** — `reward_us→snc` (exc = `r`) minus `striosome_value→snc` via slow **GABA_B/GIRK** K⁺ conductance (= `−V`); the SNc's FIRING is `delta` | `_limbic_core_rpe_battery_derisk.build_limbic_core`; `_shortcut9_trace_conditioning_bridge_probe` |
| Signed RPE → actor plasticity | **SPIKING** — `dopamine` modulator `from_region_firing_signed` over SNc → the signed `effective_signal` in `Δw = lr·effective_signal·eligibility` (`bridge.py:6904/6952`) | `_advantage_actor_critic_probe` (verified by code-read: actor is advantage-gated `r−V(place)`) |
| Schultz-98 dip / omission (signed RPE on spikes) | **SPIKING** — the signed rule produces the omission dip | `snc_pavlovian_probe` |
| `V` is load-bearing for PREDICTION (bridge a gap) | **PROVEN, 6/6 GO on real spikes** — lesion the value → the trace CR collapses; the DELAY control survives | `_shortcut9_trace_conditioning_bridge_probe`; `2026-06-21-shortcut9-trace-conditioning-value-derisk.md` |
| The value scalar that DRIVES a choice (R5) | **HOST STAND-IN** ← residual #2 | `_value_salience_appraisal_derisk.build_concept_value`; used by R5-R1a |
| `V` load-bearing for a value-driven CHOICE reading REAL spiking `V` | **NOT DEMONSTRATED** (R1-b scaffold never run) ← residual #2 | `_navcloseout_R5_value_driven_choice.r1b_two_beacon_kwargs` |
| `V` load-bearing for INSTRUMENTAL nav | **NEGATIVE / accepted-deep wall** — R4 (value-IRRELEVANT task, honest NULL); hidden-goal advantage-actor-critic (3× NEGATIVE, the point-neuron spatial-credit wall; dendrite = named unlocker) | `_navcloseout_R4_*`; `_advantage_actor_critic_probe`; `2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md` |
| `sim/td_value_critic.py` TD update | **HOST numpy** — but a REFERENCE/CEILING, NOT a deployed nav shortcut | `sim/td_value_critic.py:81/86` |

**The residual, stated exactly:** the *value computation* is spiking; the *value-drives-behavior close* is (i)
demonstrated only with a host value stand-in (R5-R1a) and (ii) only on ONE 2-option task, and (iii) the on-bridge
form that reads the real `V` was never run. Biology grounding (glossary `:1435`): **striatal matrix = actor,
striosome = critic** — the deployed `striosome_value` region is exactly the biological critic; the close is the
critic's *behavioral load-bearingness by its own spiking value*, not a new organ.

---

## 2. What "closed" means concretely (the acceptance definition)

A value critic is DONE (🟩) when a **single value-driven behavioral decision** satisfies ALL of:

- **(C1) V is LEARNED on the substrate** — `V(option_i)` comes from the trained spiking `striosome_value` critic
  (DA-gated STDP), read as its firing rate. NO host value tag anywhere in the decision path. *(retires residual #2's
  stand-in)*
- **(C2) V DRIVES the choice via a spiking decision** — the option pools' accumulator DRIFT is set by the spiking
  `V_i` (catalog O.19/C.34: value modulates the drift); the DECISION is a neural pool's FIRING (the Wang-2002 /
  Lo-Wang commit), NOT a host argmax.
- **(C3) LOAD-BEARING BY ITS FUNCTION** — lesioning the value (drive-level-matched: remove the value GRADIENT, hold
  the operating point) collapses the higher-value choice to chance; an EQUAL-value control shows the lesion is
  NEUTRAL (validate-by-function per `feedback_validate_signal_by_its_function`); the critic's LEARNING is
  load-bearing (an UNTRAINED critic → flat `V` → chance choice — the new anti-cheat R5-R1a lacked because its value
  was a host tag).
- **(C4) GENERALIZES beyond the one R5 task** — holds for **≥N options** (not a 2-option coin-flip artifact) AND on
  **≥1 second value paradigm** (e.g. delay-discounting or a two-arm bandit with variance, catalog C.34) OR the
  **nav-embodied read-out** (the missing O.22 `Q(s,a)`: approach the higher-value of two beacons).
- **(C5) one-brain + moat** — the critic/decision organ is array-disjoint from the RF/conversational slices
  (`cp_rf_w_re/im` separate from `cp_connections`) → the no-confab moat is preserved by construction; re-assert
  `check_moat` if run on the merged agent.

Honest scope boundary (per the research gate): the close deliberately stays with **value-CUED** choice, NOT
spatially-HIDDEN value — the hidden-goal place→action spatial-credit family is the accepted-deep dendritic wall (3×
NEGATIVE, months-scale `NeuronModel.TWO_COMPARTMENT`). "Generalize" here means paradigm/option-count breadth, not
conquering instrumental spatial credit.

---

## 3. RANKED cheap-first mechanisms (biology-grounded), + reusable machinery

All reuse existing project machinery; NO `sim/` edit expected (the spiking WTA, the DA-gated value-train, the value
lesion, and the two-goal `homeostatic_hook` all exist runner-side).

### RANK 1 (CHEAPEST, DECISIVE, CPU) — feed the spiking WTA the REAL LEARNED spiking `V` (retire the host stand-in)
Replace R5-R1a's `build_concept_value` host tag with the trained-and-read spiking `striosome_value` critic. Per
option `i` with a distinct learned reward value: (a) train `V(cue_i)` on the substrate via the ported DA-gated STDP
value-train; (b) read `V_i` = the critic's firing rate when `cue_i`'s code is presented (a real `cp_firing_states`
read); (c) drive option pool `i` of the spiking WTA with `drift_i = base + gain·V_i`; the DECISION is the winning
pool's firing. This converts the demonstrated close from *"spiking decision + host value"* to *"spiking decision +
spiking value"* — the exact residual-#2 fix — and adds the **untrained-critic** anti-cheat (C3).
- **Biology:** O.22 striatal action-value MSNs → a downstream WTA selector; O.19 value modulates the accumulator
  drift; C.34 DA utility → binary/economic choice.
- **Reuse:** `run_value_train` + `_critic_rate_via_afferent` (`_merged_navcritic_valuetrain.py`) for the learned
  spiking `V`; `SpikingSpeakAccumulator` (extended to `n_options`, already scaffolded in
  `_navcloseout_R5_value_driven_choice.py`) for the decision; `lesion_gabab` for the value lesion.
- **Cost:** CPU-first, minutes/seed (R5-R1a was 26 s for 6 seeds; the numpy value-train is small). ~1 day build.

### RANK 2 (nav-embodied + one generalization axis, GPU) — R1-b two-beacon value-choice on the REAL merged bridge
The `run_moving_goal_episode` value-choice that reads the REAL shared spiking `dopamine`: two beacons/goals of
DIFFERENT reward value; the BG action selector must APPROACH the higher-value goal. This is the missing catalog
O.22 `Q(s,a)` nav read-out and the "value drives the body's action" close (C4 via the nav read-out; C2 via the
default spiking sel/commit/OPN decision).
- **Reuse:** `r1b_two_beacon_kwargs` (`_navcloseout_R5_value_driven_choice.py:577`, well-formed today) +
  `run_moving_goal_episode`'s per-trial `homeostatic_hook` (differential-reward two-goal — the same hook R4 uses at
  `:7773`) + `make_value_lesion_hook` (`_navcloseout_R4_*.py:166`) + `check_moat`.
- **Cost:** GPU-only (CuPy), grid-32, 6 seeds × 4 arms {ON-distinct, OFF-lesion, ON-equal, permuted}. Higher variance
  (per the gate, the honest higher-risk arm). **Contends with the live GPU training run — schedule when GPU free.**

### RANK 3 (generalization within the choice organ, CPU) — N-option + a second value paradigm
Extend RANK 1 to N>2 options and to a delay-discounting or two-arm-bandit-with-variance schedule (catalog C.34's
named next value tasks). Cheap CPU extension; closes C4's paradigm-breadth without the GPU. Kills the "2-option
coin-flip" reading (which is exactly why R5's G_PERMUTE needed the deterministic permutation-average fix).

### NOT ranked for a build (accepted-deep) — instrumental spatial credit
The hidden-goal place→action actor-critic (`r − V(place)` carving a policy on the point-neuron cascade) is the 3×
NEGATIVE spatial-credit wall; the dendrite is the named unlocker. The close stays value-CUED to avoid it (the same
V-A sidestep). Only a value-CUED-choice NEGATIVE that survived its own SURPASS round would localize a genuine
value-mechanism boundary here — the piecewise evidence (V-A 6/6 GO + R5-R1a 6/6 GO) makes a GO the likely outcome.

| Reusable primitive | What it gives the close | Where |
|---|---|---|
| DA-gated value-train (learned spiking `V`) | `V(cue)` learned by three-factor STDP on `striosome_value` | `_merged_navcritic_valuetrain.run_value_train` |
| Critic-rate read (`V` as firing) | `V_i` read from `cp_firing_states` (no host arithmetic) | `_merged_navcritic_valuetrain._critic_rate_via_afferent` |
| Spiking value-driven WTA (n_options) | the choice DECISION = a neural pool's firing (drift = worth) | `_value_salience_appraisal_derisk.SpikingSpeakAccumulator`; R5's n_option scaffold |
| GABA_B value lesion | the drive-level-matched value lesion (collapses the gradient, holds the op-point) | `_merged_navcritic_valuetrain.lesion_gabab`; R5's drive-level-matched lesion |
| V-A validate-by-function gate logic | the G_LESION-collapse / G_DISCRIM-equal / permuted controls, proven 6/6 | `_shortcut9_trace_conditioning_*`; R5's 4-gate harness |
| Two-goal differential-reward hook | the nav-embodied value-choice (R1-b) with zero `sim/` edit | `run_moving_goal_episode` `homeostatic_hook` (`g11_bg_runner.py:7773`) |
| `check_moat` / array-disjoint | one-brain moat preserved + re-asserted | `_merged_navcritic_valuetrain.check_moat` |

---

## 4. RECOMMENDED cheap-first de-risk (the smallest experiment that shows a spiking critic computes value load-bearingly)

**Build ONE CPU de-risk = RANK 1: "the LEARNED spiking `V` drives the value choice."** Smallest decisive form:
1. K options (start K=2, then K=4 for C4), each with a distinct reward value. Train `V(cue_i)` on the substrate
   (`run_value_train`, DA-gated STDP on `striosome_value`).
2. Read `V_i` = critic firing rate for `cue_i` (`_critic_rate_via_afferent`).
3. Drive WTA option pool `i` with `drift_i = base + gain·V_i`; DECISION = winning pool's firing. CORRECT = pick the
   highest-`V` option.

**Anti-cheat controls (validate-by-function — the gates):**
- **G_HEADLINE:** choice picks the higher-`V` option ≫ chance (≥0.20 above), ≥5/6 seeds.
- **G_LESION (headline):** the **drive-level-matched** value lesion (`lesion_gabab` / pin DA-value to the option
  MEAN — remove the GRADIENT, hold the operating point, per R5's lesson that dropping the whole term shifts the
  op-point and fails G_DISCRIM) → the higher-value choice collapses to chance.
- **G_UNTRAINED (the NEW anti-cheat R5-R1a lacked):** run the SAME pipeline with the critic UNTRAINED (init weight,
  `V` flat) → the choice is at chance. **This is what makes the *learning* load-bearing** — it proves the value is
  the brain's LEARNED spiking `V`, not a host tag (residual #2's fix, verified). Mirrors
  `_merged_navcritic_valuetrain`'s untrained-flat anti-cheat.
- **G_DISCRIM (validate-by-function):** EQUAL-value options → lesion NEUTRAL (choice agreement ~1.0). The lesion's
  effect is value-GRADIENT-specific, not a general lesion artifact.
- **G_PERMUTE:** permute the option↔value contingency via the **deterministic permutation-average** (R5's fix) →
  advantage → chance exactly, zero coin-flip variance.
- **G_NONCIRCULAR:** the learned `V` is decorrelated from any orienting/salience cue (assert corr ≈ 0), so "value
  drives the choice" is not a relabeled salience.
- **AC_MOAT:** the critic/WTA organ has NO RF slices → array-disjoint → moat by construction (re-assert on merge).
- **Regime fidelity:** deterministic OU/conductance/homeostasis regime (the #6 lesson); the value-vs-lesion compare
  is a clean ablation (hold the OU realization fixed per drive-vector, R5's snapshot/restore pattern).

**GO = G_HEADLINE + G_LESION + G_UNTRAINED + G_DISCRIM + G_PERMUTE all ≥5/6.** This is the R5 result upgraded from a
host value stand-in to the LEARNED spiking `V`, with the untrained-critic control proving the substrate's learning
is load-bearing. Then RANK 3 (K=4 + a second paradigm) for C4; RANK 2 (nav read-out) as the GPU confirmation.

---

## 5. Effort / cost + concurrency with language / gap#5

- **RANK 1 (the recommended de-risk):** ~1 day build, CPU-first, minutes/seed. Reuse-by-import from 3 existing
  runners; NO `sim/` edit. **Cleanly concurrent** with the language / gap#5 work: the value-critic organ is
  array-disjoint from the conversational/RF slices (moat by construction), and RANK 1 needs **no GPU** — it contends
  only for CPU cores with the live CPU de-risk, not for the GPU training run.
- **RANK 3 (generalization, CPU):** +0.5–1 day; also GPU-free; concurrent.
- **RANK 2 (nav R1-b, GPU):** ~hours (grid-32, 6 seeds × 4 arms). **NOT concurrent with the live GPU training run** —
  schedule when the GPU is free. It is the optional higher-variance nav confirmation; RANK 1 is the decisive proof,
  so a GPU-busy period does not block the close.
- **Risk:** LOW for RANK 1/3 (V-A 6/6 GO + R5-R1a 6/6 GO + the spiking value-train GO all predict a GO once the WTA
  reads the learned `V`). RANK 2 carries the documented BG-cascade-readout variance (a NEGATIVE there localizes a
  cascade read-out issue, not a value-mechanism one, since RANK 1 isolates the value-WTA).

**Bottom line:** the value critic is one CPU de-risk (RANK 1) away from a genuine spiking-value close — retire R5's
host value stand-in by feeding the WTA the LEARNED spiking `V`, add the untrained-critic anti-cheat, then generalize
(RANK 3 CPU + RANK 2 GPU-when-free). The `sim/td_value_critic.py` numpy TD is a reference/ceiling and is NOT the
thing to close; the deployed nav value pathway is already spiking.

---

## Sources (verified this pass)

- `sim/td_value_critic.py` (`:81`/`:86` host TD update) — imported only by `td_critic_gate.py`,
  `research/run_pavlovian_parallel.py`, `test_*_no_harm.py` (reference/ceiling, NOT deployed).
- `research/findings/2026-06-27-navcloseout-R5-value-driven-choice-GO.md` (R1-a GO 6/6 with a host value stand-in;
  R1-b flagged for the controller) + `2026-06-27-nav-value-loadbearing-research-gate.md` (the RANK-1 value-driven-
  choice plan; the R4 value-irrelevant reconciliation).
- `research/findings/raw/navcloseout_R5/` — only `R5_r1a.json` + `R5_r1a_6seed_permfix.json` (R1-b NEVER run).
- `research/runners/_value_salience_appraisal_derisk.py` (`build_concept_value:114` = the CPU value stand-in;
  `SpikingSpeakAccumulator:153` = the spiking value-WTA decision).
- `research/runners/_merged_navcritic_valuetrain.py` (`run_value_train` = learned spiking `V` via DA-gated STDP;
  `_critic_rate_via_afferent` = `V` read; `lesion_gabab`; `check_moat`).
- `research/runners/_shortcut9_trace_conditioning_bridge_probe.py` + `2026-06-21-shortcut9-trace-conditioning-value-
  derisk.md` (V-A 6/6 GO on spikes — value load-bearing for PREDICTION; the G2-collapse/G3-survive gate logic).
- `research/runners/_advantage_actor_critic_probe.py` (verified: `dopamine` `from_region_firing_signed` = the signed
  `effective_signal` in `Δw = lr·effective_signal·eligibility`, `bridge.py:6904/6952` — the actor IS advantage-gated
  `r−V(place)`); `_limbic_core_rpe_battery_derisk.build_limbic_core` (the `r−V` SNc organ); `snc_pavlovian_probe`
  (the spiking omission dip).
- `research/runners/_navcloseout_R4_delayed_reward_value.py` (`make_value_lesion_hook:166`; the two-goal
  `homeostatic_hook` at `run_moving_goal_episode` `g11_bg_runner.py:7773`).
- `references/glossary.md:1435` (striatal matrix = actor, striosome = critic); catalog O.22 / O.19 / C.34 / L.41
  (value-driven / economic choice as the canonical load-bearing paradigm).

_READ-ONLY scoping. No code, no `sim/` edit, no experiments / GPU run. The value computation is already spiking; the
residual is the load-bearing demonstration on the LEARNED spiking `V` (retire R5's host stand-in) + generalization.
The no-confab moat is array-disjoint from the value critic organ and untouched. grid-32 is the RANK-2 nav verdict
scale (never grid-8)._
