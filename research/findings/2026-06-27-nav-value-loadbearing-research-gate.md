# Nav spiking VALUE-critic load-bearing — the R4 delayed-reward NEGATIVE reconciled (deep-research gate, 2026-06-27)

**Type:** READ-ONLY deep-research + reference-catalog gate (the project's standing "deep research FIRST at a
confirmed NEGATIVE" move, CLAUDE.md). NO code written, NO `sim/` edit, NO experiments / GPU run, NO
background-and-wait. Single deliverable = this doc. Every load-bearing claim was trust-but-verified against the
actual finding text + the R4 raw JSONs (read in full) + the R4 runner source + the current code + the catalog.

**The NEGATIVE under the gate:** `2026-06-27-navcloseout-R4-delayed-reward-value-task-NEGATIVE.md` — the spiking
nav value-critic (N-2, the CYCLE-1B default) is NOT behaviorally load-bearing on the R4 delayed-reward task
(2-seed: `improvement_delayed` ≤ 0 [−0.004, −0.112], value×delay interaction +0.001/−0.118 inconsistent,
`neutral_on_immediate` True, `permute_control_ok` True).

**The four-move SURPASS round (the mandate):** isolate+quantify → reframe via biology → rank cheap-first → verdict.

---

## 0. TL;DR — the headline reconciliation, then the verdict

**The cause is candidate #1 (TASK-DESIGN), confirmed by inspecting the R4 runner + the raw JSONs: the R4 task is
VALUE-IRRELEVANT — the value V is a PASSENGER, not a limiter.** R4 delays the per-step reward on the *same single
moving goal* whose optimal action ("reduce Manhattan distance to the one goal") is UNCHANGED by when the reward
arrives. There is no choice in the task, so a value baseline cannot change behavior, and the raw data shows exactly
that: **all six 2×2 arms cluster in a razor-thin band (mean_distance 2.48–2.69), and the value-OFF arm is within
±0.11 of value-ON on EVERY arm including delayed.** The limiter is the TASK, not the δ (candidate #2) and not a
spatial-credit dendritic wall (candidate #3).

**Decisively — the project ALREADY proved the spiking value IS load-bearing-when-required, BY ITS FUNCTION.**
`2026-06-21-shortcut9-trace-conditioning-value-derisk.md` (commit `1a861f87`, on `main`) is a **6/6 GO on REAL
SPIKES** on the canonical Pavlovian trace-conditioning task: lesioning the dendrite-graded value COLLAPSES the
trace-arm CR (G2: 100 Hz → 0–1.7 Hz, 6/6) while the immediate-reward DELAY control SURVIVES (G3: ≥10 Hz, ≥3× the
trace floor, 6/6). So "is the spiking value load-bearing for the function it computes (credit across a gap)?" is
**already answered YES** — R4 simply re-tested it on a task that does not exercise that function.

**R4 is the INSTRUMENTAL (V-B, act-over-gap) variant — and in its weakest possible form.** The scoping
(`2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` §4.3/§7.2) explicitly predicted this: V-B is
"where a real substrate wall MIGHT appear … a V-B NEGATIVE would be the honest characterized deliverable." But R4
is not even a genuine V-B *choice* task — it is delay-on-a-single-orient-solvable-goal, where the reward timing
changes nothing the agent must DO. So R4 is NOT the V-B substrate-wall finding either; it is the orient-solvable
confound, repeated. The genuine V-B test requires a task where the value DRIVES the choice.

**VERDICT: CLOSEABLE-CHEAPLY (value-driven-CHOICE task).** The single cheapest decisive test of whether the
spiking value is load-bearing-WHEN-REQUIRED in an *instrumental* setting is a **two-goal / two-option choice task
of DIFFERENT reward value** where picking the higher-value option REQUIRES V — and the project already has the
spiking value-driven-WTA-decision mechanism built and validated (`_value_salience_appraisal_derisk.py`: a real
Izhikevich WTA whose drift = a candidate's WORTH = f(DA-value), with a lesion that collapses it to baseline). It is
NOT the δ (the δ already grades correctly and is load-bearing on V-A), and NOT the accepted-deep spatial-credit
dendritic wall (that wall is the hidden-goal place→action arc, 3× NEGATIVE; the value-CHOICE task deliberately
sidesteps spatial credit, exactly as V-A did). A value-CHOICE NEGATIVE would then be the genuine
instrumental-act-over-gap boundary (the legitimate juncture for the deferred dendritic question) — but the prior
evidence (V-A GO + the existing spiking value-WTA) makes a GO the likely outcome.

---

## MOVE 1 — ISOLATE + QUANTIFY: which candidate is it? (the TASK, not the δ; V is a passenger)

### 1.1 Does the R4 task GENUINELY require value-driven action? — NO. Confirmed by inspecting the runner.

The mandate's crux: "does the R4 task genuinely require value-driven action, or is it value-irrelevant (the actor
reaches the goal regardless, so V is a passenger)?" **Value-irrelevant. Confirmed three ways:**

**(a) The task structure (the runner).** `_navcloseout_R4_delayed_reward_value.py`:
- The goal schedule is `multi_goal_schedule(grid_size, n_steps)` — a SINGLE moving goal that jumps to a new
  location at each quarter (the documented flagship moving-goal benchmark). **At every step there is exactly ONE
  goal and exactly one best action: the one that reduces Manhattan distance to it.**
- The "delayed" manipulation is `make_delay_hook(delay)` (`:114`): a pure FIFO that buffers each step's reward and
  releases it `delay` steps later. **It does NOT change which action is optimal at any step** — it only changes
  *when the SNc burst / corticostriatal STDP sees* the (unchanged) reward. The optimal policy ("go toward the one
  goal") is identical for `delay=0` and `delay=12`.
- The score is `mean_distance_overall` to that one goal (LOWER better). A value baseline V(s) cannot improve "go
  toward the single visible goal" — the SC/orienting/place machinery already reaches it without any *predictive*
  value (the documented orient-solvable property).

⇒ **The reward delay does NOT, and cannot, change behavior** — the task has no decision whose correct answer
depends on a learned value. This is the `feedback_validate_signal_by_its_function` failure mode in its purest form:
the A/B (value ON vs OFF) is run on a task for which the value is not load-bearing **by construction**.

**(b) The raw data (the JSONs, read in full).** All six arms, both seeds, cluster in `mean_distance` 2.48–2.69:

| arm | seed 42 md | seed 43 md | value lesion landed? |
|---|---|---|---|
| value_on_immediate  | 2.594 | 2.600 | — |
| value_off_immediate | 2.589 | 2.607 | `n_gabab_cut`=415 / 388 |
| value_on_delayed    | 2.630 | 2.610 | — |
| value_off_delayed   | 2.626 | 2.498 | `n_gabab_cut`=415 / 388 |
| value_on_delayed_permuted  | 2.694 | 2.712 | — |
| value_off_delayed_permuted | 2.653 | 2.483 | `n_gabab_cut`=415 / 388 |

The value-OFF − value-ON improvement is essentially zero/noise on EVERY arm: immediate −0.005/+0.007, delayed
−0.004/−0.112. The value critic is a **complete passenger** — silencing 415/388 real GABA_B synapses (the lesion
LANDED — `n_gabab_cut`>0, AC_LESION-LANDED holds) does not move navigation, on the immediate OR the delayed arm.
The seed-43 "delayed" Δ (−0.112) goes the WRONG way (value-OFF slightly BETTER) and the permuted Δ is more negative
(−0.228), i.e. pure seed noise in a band where V does nothing — not a value signal.

**(c) The prior deploy (independent confirmation).** The #9 nav deploy already measured the identical confound:
**dendcritic 8.47 ≈ value-lesion 9.08 (Δ7.2%)**, with the WHOLE gain over the point-neuron baseline coming from the
NMDA on the critic slice, not the value (`ctrl_nmda` 8.72 ≈ dendcritic; SNc flat 50 Hz) — re-run + verified in the
B4 scoping §2.2. R4 was built precisely to fix that confound but **inherited it**, because delaying the reward on a
single-goal orient-solvable task still does not make the value load-bearing.

### 1.2 Is the limiter the TASK or the δ?

**The TASK.** Three independent facts rule OUT the δ (candidate #2) as the R4 limiter:
- The δ is GRADED in the right direction and is **already load-bearing on the task that needs it** (V-A trace: G2
  lesion collapses the CR 6/6 on real spikes). A weak-but-correct δ that *works on V-A* cannot be the reason it is
  inert on R4 — R4 is inert because V has nothing to predict that changes behavior.
- Even if the δ were lifted to the host ceiling (R3, the graded dendritic plateau → δ=1.33), R4 would STILL be flat:
  a perfectly-graded value baseline cannot change "go toward the one visible goal." The δ-magnitude is orthogonal to
  the R4 NEGATIVE.
- The weak merged-δ (~1.3×, the position-blind up-state floor, `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md`)
  is a REAL characterized boundary — but it is a *δ-quality* boundary (how cleanly V grades the dopamine burst), not
  a *behavioral-load-bearing* boundary. R4 measures behavior, and behavior is task-limited, not δ-limited.

### 1.3 Is it the accepted-deep spatial-credit dendritic wall (candidate #3)? — NO, not here.

The accepted-deep wall is the **hidden-goal place→action actor-critic-credit family** (3× NEGATIVE: the 2026-05-05
global-scalar W→A, the limbic-load-bearing hidden-goal diagnostic, the advantage-routing de-risk — the
`r − V(place)` failing to carve a place→action policy on the point-neuron cascade; named unlocker = the dendrite).
R4 does NOT touch that wall: R4's goal is VISIBLE (orient-solvable) and the value is a passenger, so R4 cannot be
the spatial-credit boundary — it never exercises spatial credit at all. (A future value-CHOICE task that put the
*value* behind a hidden contingency could approach that family; the cheap close in MOVE 3 deliberately keeps the
choice value-cued, not spatial-hidden, to isolate "is V load-bearing for choice" from "can the cascade do spatial
credit" — the same V-A-style sidestep.)

**Isolated cause: candidate #1 (TASK-DESIGN). The R4 task is value-irrelevant; V is a passenger; the limiter is the
task, NOT the δ and NOT a dendritic wall.**

---

## MOVE 2 — REFRAME via biology: what task makes a value-critic PROVABLY load-bearing on a spiking point-neuron substrate?

The biology question: what task + read-out makes a value signal *behaviorally* load-bearing — value-DRIVEN choice
(pick the higher-value option), graded value — on a point-neuron spiking substrate? The catalog names the exact
substrate and the exact paradigm (verified against `sim-catalog/references/feature-catalog.md`):

### 2.1 The canonical value-DRIVEN-CHOICE substrate (catalog, verified)

- **O.22 Striatal action-value coding** (`:546`, Schultz16-JNT; Samejima 2005, Lau & Glimcher 2008, Ito & Doya
  2009, Kim 2009, Seo 2012). The strict Sutton-Barto criterion: an action-value neuron `Q(s,a)` "needs to be coded
  for each action by separate neurons **irrespective of the action being chosen**." Striatal MSNs satisfy this
  empirically — subgroups fire for the value of ONE specific action (e.g. left-arm) regardless of which action is
  chosen. Schultz's basic decision model: the DA RPE updates cortex→striatum weights (three-factor + eligibility);
  the striatal action-value neurons feed a **downstream competitive decision mechanism = argmax-with-noise
  selector.** ⇒ **value-driven CHOICE = action-value MSNs → a spiking WTA selector.** **Sim status (catalog,
  verified `:547`): partial — the structure IS there (the per-action `str_D1_X` pools) but the explicit Q(s,a)
  read-out / value-driven choice is NOT measured.** This is the precise gap a value-CHOICE task fills.
- **O.19 Value-based decisions — vmPFC/OFC encode subjective value** (`:4996`). "Value modulates the DRIFT RATE of
  the accumulator. … Same drift-diffusion math as perceptual decisions." Decisions about preferences reduce to
  evidence accumulation where each option's evidence is its subjective-value samples. **Sim status (`:5002`):
  partial — DA-modulated cortex→D1 implements value LEARNING; no separate region encodes scalar value ACROSS
  actions independently of the selector.** ⇒ a value-CHOICE task is exactly the missing measurement.
- **C.34 DA codes utility / risk** (`:640`): "**Becomes critical for any future risk-sensitive choice tasks
  (binary-gamble selection, two-arm bandit with variance differences, delay-discounting tasks).**" The catalog
  itself flags the two-arm bandit as the natural next value task.
- **L.41 Reward = goal object for approach + economic choice** (`:559`): function 2 of reward is "assigns value to
  options for selection. Operationalized by binary-choice tasks revealing transitive subjective preferences and
  certainty equivalents. Substrate: DA-encoded utility (C.34) + striatum action-value (O.22) + OFC/vmPFC scalar
  value (O.19)." ⇒ **the binary value-choice task is THE catalog-canonical way to make a value signal load-bearing.**
- **G.16 / G.59 drift-diffusion bounded accumulation** (`:2826`, `:2819`): the bound-crossing decision "holds for
  perceptual *and* value-based judgments" — the value-choice read-out is the SAME spiking accumulator the project's
  nav decision already uses (Wang-2002 NMDA + Lo-Wang commit burst).

### 2.2 The biology of the right task (the reframe)

The brain makes a value signal load-bearing not by delaying a reward on a fixed action, but by making the agent
**CHOOSE between options of different value** — economic choice / two-arm bandit / matching-law foraging
(Padoa-Schioppa OFC economic choice; Sugrue-Corrado-Newsome matching; Lau-Glimcher action-value). The value of each
option is the ONLY thing that can drive the correct choice: lesion the value and the agent reverts to chance /
salience / a fixed bias. This is the value analogue of the V-A trace dissociation (lesion collapses the
value-requiring behavior; a value-irrelevant control survives). **The wrong hypothesis R4 tested:** "does delaying
the reward make V load-bearing?" — NO, because delay alone does not create a value-dependent decision. **The right
hypothesis:** "does a CHOICE between different-value options make V load-bearing?" — which the catalog (O.22/O.19/
C.34/L.41) says is exactly the paradigm, and which the project can build by reuse.

### 2.3 The biology already validated in-project (the strongest reframe evidence)

Two project results already realize "value load-bearing on the point-neuron substrate," both via the value DRIVING
a behavior, neither needing a dendrite:
- **V-A Pavlovian trace (`2026-06-21-shortcut9-trace-conditioning-value-derisk.md`, 6/6 GO on spikes):** the value
  is load-bearing for PREDICTION across a gap; lesion collapses the CR. (Predict, not choose.)
- **The choose-to-SPEAK value appraisal (`_value_salience_appraisal_derisk.py`):** a real spiking Izhikevich WTA
  whose speak-pool DRIFT = the candidate's WORTH = f(DA-value), structurally distinct from plausibility (corr≈0);
  **lesion the value (pin DA-value to baseline) → the decision reverts to the plausibility-only baseline (the
  value-driven emissions vanish).** This is a value-DRIVEN-CHOICE (speak vs. silence) decided by a spiking
  accumulator, with the exact lesion logic a nav value-choice needs — already built, CPU, no `sim/` edit. **It is
  the proof-of-concept that a spiking, point-neuron value-driven WTA choice with a load-bearing lesion WORKS in
  this codebase** — the nav value-choice task transplants it onto the BG action selector.

---

## MOVE 3 — RANK cheap-first: the decisive tests, cheapest first

Anti-cheats reuse the cluster battery: validate-by-FUNCTION (the lesion must collapse the value-dependent
behavior), the value LESION (`lesion_gabab` / pin DA-value to baseline), a value-IRRELEVANT discriminating control
(the V-A `delay`/the equal-value control), the PERMUTED contingency control, 6-seed for the variable effect (grid-32
NEVER grid-8), regime fidelity (OU/conductance-noise/homeostasis OFF), and the no-confab MOAT (nav `cp_*` arrays
array-disjoint from the composer's `cp_rf_w_re/im` → preserved by construction + re-asserted via `check_moat`).

### RANK 1 (CHEAPEST, DECISIVE) — a VALUE-DRIVEN-CHOICE task: re-test N-2 by its TRUE function

Build ONE two-option choice task where picking the higher-value option REQUIRES the learned value V, then run the
SAME value-ON/value-OFF 2×2 logic. This is the direct fix for the R4 task-design failure — it makes the value the
ONLY signal that can drive the correct behavior, so the lesion is forced to be load-bearing-or-not.

- **R1-a (cheapest — reuse the existing spiking value-WTA pattern; CPU-first).** Mirror
  `_value_salience_appraisal_derisk.py` onto a minimal TWO-ACTION choice: two options A/B (e.g. two cued goals, or
  a 2-arm bandit) with DIFFERENT learned values V(A) ≠ V(B), seeded from a reward-tagging RNG **structurally
  distinct from any orienting/salience cue** (so "value drives the choice" is not circular). A spiking Izhikevich
  WTA (the GO sel→commit→OPN template, drift = the chosen option's value) decides A-vs-B. **De-risk (validate-by-
  function):** (G_HEADLINE) value-ON picks the higher-value option ABOVE chance; (G_LESION, the headline) pin the
  DA-value to baseline → the choice reverts to chance / the salience-bias baseline (the value is the load-bearing
  signal); (G_DISCRIM) an EQUAL-VALUE control (V(A)=V(B)) shows the lesion does NOT change the (already-chance)
  choice — the value-irrelevant discriminator, the direct analogue of the V-A `delay` arm; (G_PERMUTE) shuffling
  the option↔value contingency collapses the advantage. ≥5/6 seeds for the effect (CPU smoke first, then GPU
  multi-seed). **Anti-cheat:** the value axis decorrelated from the orienting cue (the appraisal probe's
  corr(value, plausibility)≈0 precedent); MOAT untouched (this is a critic/decision organ, array-disjoint).
  **NO `sim/` edit** (the spiking WTA, the value seed, the lesion all exist runner-side).
- **R1-b (the nav-embodied form, if R1-a GO and a nav read-out is wanted).** Two simultaneous beacons/goals of
  different value on the grid; the BG action selector (`str_D1_X` per-action pools, O.22) drives approach; the
  agent must approach the HIGHER-value goal. **De-risk:** value-ON approaches the higher-value goal (lower mean
  distance to IT, higher value harvested) > value-OFF (which approaches the nearer/salience-default); the
  EQUAL-value control shows no value advantage; lesion collapses the preference. ⇒ this is the genuine
  instrumental value-choice on nav — the missing O.22 Q(s,a) read-out the catalog flags. *(Honest scope: R1-b is
  the higher-variance arm — if the BG selector can't carve a value-graded choice it localizes to the O.22/O.19
  "explicit action-value read-out missing" gap, which is the legitimate juncture for the dendrite question. R1-a
  first, CPU, isolates the value-WTA from the nav cascade — the same predict-first/choose-second discipline as
  V-A→V-B.)*

### RANK 2 — R3, deploy the graded dendritic plateau into the merged critic (the δ-MAGNITUDE half, NOT the R4 cause)

`2026-06-27-nav-loop-closure-research-gate.md` RANK 3. Wire `--dendrite-critic` (the graded plateau, δ=1.33 = host
ceiling on a critic bridge, 6/6) into the merged value critic to lift the weak merged δ (~1.3×) toward the host
ceiling. **This is candidate #2 (the weak-δ), and it is a REAL improvement to δ-QUALITY — but it is NOT the R4
NEGATIVE's cause** (R4 is task-limited; a perfect δ stays inert on a value-irrelevant task). R3 is worth doing for
δ-quality + the V-A/value-choice read-out fidelity (G4: the value GRADES the burst, not flat-50-Hz), but it does
not, by itself, make the value load-bearing — RANK 1 (the task) does. The plateau is a byte-reviewed default-OFF
`sim/` flag; the two point-neuron controls fail (LINEAR 0 Hz; all-or-none over-clamp); plateau-lesion collapses δ;
MOAT default-OFF for conversational slices. **Order: RANK 1 first (it is the actual fix); R3 is the δ-quality
companion, valuable for the value-choice burst-grading but not the load-bearing close.**

### RANK 3 (NOT a build now / accepted) — the spatial-credit dendritic-months wall

The accepted-deep boundary is the **hidden-goal place→action actor-critic spatial-credit family** (3× NEGATIVE;
`r − V(place)` carving a place→action policy on the point-neuron cascade; the dendrite is the named unlocker —
`NeuronModel.TWO_COMPARTMENT`, catalog T3.A, ~10× compute, months-scale, and the broader dendrite is already 3×
NEGATIVE on binding + apical-basal credit per `2026-06-20-boundary-ledger-dendritic-audit.md`). **R4 is NOT this
wall** (MOVE 1.3), and the RANK-1 value-CHOICE task deliberately keeps the value CUED (not spatially hidden) to
test "is V load-bearing for choice" WITHOUT entangling spatial credit — the same sidestep V-A used. **If RANK-1
R1-b (the nav-embodied value-choice) NEGATIVE survives its own SURPASS round, THAT would be the genuine
instrumental-act-over-gap boundary and the legitimate juncture for the dendritic question — but the prior evidence
(V-A GO + the existing spiking value-WTA) predicts a GO.** Not ranked for a build; the accepted-deep characterization
if the cheap tests ever wall.

---

## MOVE 4 — VERDICT

**The R4 NEGATIVE is candidate #1 (TASK-DESIGN): the R4 delayed-reward task is value-IRRELEVANT — the value V is a
PASSENGER (the actor reaches the single visible goal regardless of reward timing or the value baseline), confirmed
by the runner structure (single moving goal, optimal action unchanged by delay) AND the raw JSONs (all six arms in
a 2.48–2.69 band; value-OFF within ±0.11 of value-ON everywhere; the 415/388-synapse lesion LANDED but moved
nothing). It is NOT the weak δ (candidate #2 — the δ is graded, correct, and already load-bearing on V-A; lifting
it cannot change a value-irrelevant task) and NOT the accepted-deep spatial-credit dendritic wall (candidate #3 —
R4 never exercises spatial credit; the visible goal is orient-solvable).**

**This is the `feedback_validate_signal_by_its_function` lesson, repeated:** R4 was built to fix the #9 deploy's
orient-solvable confound but inherited it, because delaying a reward on a single-goal orient-solvable task does not
create a value-dependent decision. The R4 NEGATIVE is an HONEST, correctly-anti-cheated NULL result (the permute
control is sound, the lesion landed, neutral-on-immediate is correct) — but it tests value LOAD-BEARING on a task
where, by construction, the value cannot be load-bearing. It does NOT regress anything (the critic still runs on
spikes; the conversational moat is untouched by construction).

### The single cheapest DECISIVE test of "is the spiking value load-bearing-WHEN-REQUIRED?"

**A VALUE-DRIVEN-CHOICE task (RANK 1):** two options of DIFFERENT learned value where picking the higher-value one
REQUIRES V, decided by the EXISTING spiking value-WTA (the `_value_salience_appraisal_derisk` Izhikevich
sel→commit accumulator, drift = the chosen option's WORTH), with the value LESION forced to be load-bearing-or-not
(G_LESION: pin DA-value to baseline → choice reverts to chance/salience baseline) and a value-IRRELEVANT
EQUAL-VALUE discriminator (G_DISCRIM, the V-A-`delay` analogue). CPU-first (R1-a, reuse the appraisal probe's WTA +
lesion), then the nav-embodied form (R1-b, the two-beacon BG-selector choice = the missing O.22 Q(s,a) read-out)
if a nav read-out is wanted. **~95% reuse-by-import, NO `sim/` edit** (the spiking WTA, the structurally-distinct
value seed, the GABA_B/DA-value lesion, the per-action `str_D1_X` pools all exist).

### Why a GO is the LIKELY outcome (the close, not a boundary)

The project has ALREADY shown the spiking value is load-bearing-when-required, twice, on the point-neuron substrate,
neither needing a dendrite: (1) **V-A Pavlovian trace = 6/6 GO on real spikes** (lesion collapses the CR; the
value-irrelevant `delay` control survives) — the load-bearingness IS established for the function the value
computes; (2) **the choose-to-speak value appraisal** — a spiking value-driven WTA choice with a load-bearing
lesion that reverts to baseline, already built. The value-CHOICE nav task transplants (2)'s validated mechanism
onto the catalog-canonical economic-choice paradigm (O.22 + O.19 + C.34 + L.41), tested by (1)'s validate-by-
function logic. So the honest expectation: **CLOSEABLE-CHEAPLY → GO** once the task exercises the value's choice
function.

### The precise accepted-deep characterization (the ONE, only if the cheap test ever walls)

The spatial-credit family — `r − V(place)` carving a place→action policy on the point-neuron cascade (the
hidden-goal 3× NEGATIVE; the dendrite the named unlocker; months-scale `NeuronModel.TWO_COMPARTMENT`). **R4 is NOT
this wall**, and the RANK-1 value-choice task is designed to avoid it (value cued, not spatially hidden). Only a
value-CHOICE NEGATIVE that survives its own SURPASS round would localize a genuine instrumental-act-over-gap
boundary here — and the V-A GO + the existing spiking value-WTA make that unlikely.

### The honest-negative that IS the deliverable (BRAIN-BASED-ONLY)

Per the owner standard, the R4 NULL is itself a small, honest, correctly-controlled deliverable: it maps that the
spiking nav value-critic is NOT behaviorally load-bearing on a value-IRRELEVANT (orient-solvable, delay-only) task
— which is the *expected and correct* result, and which sharpens the map by pointing at the task that WOULD
exercise it (value-driven choice). The weak merged δ (~1.3×, the position-blind up-state floor) remains a separate,
characterized δ-QUALITY boundary (a structural property of the A1+A2 critic), not the R4 behavioral cause.

---

## Reusable machinery (point any RANK-1 build at these proven primitives — NO `sim/` edit expected)

| Primitive | What it gives the value-choice task | Where / status |
|---|---|---|
| Spiking value-driven WTA choice + value lesion | a spiking accumulator whose DRIFT = a candidate's WORTH=f(DA-value); lesion → reverts to baseline (the exact value-choice + load-bearing-lesion pattern) | `research/runners/_value_salience_appraisal_derisk.py` (CPU, no `sim/` edit; speak-vs-silence) |
| Spiking commit-burst decision (#4) | the action EMERGES from spiking competition (sel→commit→OPN, Wang-2002 NMDA + Lo-Wang commit), default-on lib | `g11_bg_runner.py:2094-2203`; `2026-06-19-spiking-decision-default-on-GO.md` (1.16× host) |
| Per-action `str_D1_X` action pools (O.22 structure) | the action-value substrate the value-choice read-out reads from (the missing Q(s,a) measurement) | `build_bg_brain_regions`; catalog O.22 (`:546`, sim status partial) |
| V-A trace-conditioning validate-by-function harness | the load-bearing-lesion + value-irrelevant-control gate logic (G2 collapse / G3 survive), proven 6/6 on spikes | `_shortcut9_trace_conditioning_{numpy,bridge}_probe.py`; `2026-06-21-shortcut9-trace-conditioning-value-derisk.md` (`1a861f87`) |
| `lesion_gabab` / pin-DA-value-to-baseline | the established value-critic lesion (GABA_B→SNc zeroed, or DA-value pinned) | `_merged_navcritic_valuetrain.lesion_gabab`; `_value_salience_appraisal_derisk` (DA-value pin) |
| Graded dendritic plateau read-out (R3 companion) | grades the choice burst (δ=1.33 = host ceiling on-substrate) — the δ-QUALITY half, not the load-bearing cause | `enable_graded_dendritic_plateau` (default-OFF, byte-reviewed `d69cc0ab`/`52dafaeb`); `--dendrite-critic` |
| `check_moat` / array-disjoint nav-vs-conv | the no-confab moat preserved by construction + re-asserted | `_merged_navcritic_valuetrain.check_moat`; `cp_rf_w_re/im` disjoint from `cp_connections` |

---

## Citations

**Project findings (read in full, trust-but-verified against the text + the raw JSONs + the runner + code):**
- `2026-06-27-navcloseout-R4-delayed-reward-value-task-NEGATIVE.md` (the NEGATIVE under the gate)
- `2026-06-27-navcloseout-R4-delayed-reward-value-task-SCOPED.md` (the R4 build + the 2×2 design; the honest-scope
  note that a no-help-on-delayed result localizes ACT-over-gap)
- `research/findings/raw/navcloseout_R4/R4_factorial_seed{42,43}.json` (READ IN FULL — all six arms in a 2.48–2.69
  band; value-OFF ≈ value-ON everywhere; `n_gabab_cut`=415/388 = the lesion LANDED)
- `research/runners/_navcloseout_R4_delayed_reward_value.py` (the runner — `make_delay_hook:114` is a pure FIFO that
  does NOT change the optimal action; single `multi_goal_schedule`; `mean_distance_overall` score)
- `2026-06-21-shortcut9-trace-conditioning-value-derisk.md` (`1a861f87`, on `main`) — **the V-A value IS load-bearing
  BY FUNCTION: 6/6 GO numpy + 6/6 GO on real spikes** (G2 lesion collapses the trace CR; G3 delay control survives)
- `2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` (`f47e6e39`) — the value-load-bearing design; the
  #9 deploy qualified-NEGATIVE Δ7.2% (re-run); the V-A-safe / V-B-genuine-substrate-wall localization
- `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md` — the weak merged δ (~1.3×, position-blind up-state floor) =
  candidate #2, a δ-QUALITY boundary (NOT the R4 behavioral cause)
- `2026-06-20-dendrite-stage1-snc-calibration.md` — the graded dendritic plateau → δ=1.33 = host ceiling
  on-substrate, 6/6 (R3, the δ-quality companion)
- `2026-06-20-shortcut9-dendrite-critic-deploy.md` — the `--dendrite-critic` deploy wiring
- `2026-06-27-nav-loop-closure-research-gate.md` (`7e66e81a`) — the parent gate; R4 = RANK 4; it PRE-PREDICTED the
  DELAY-arm-survives outcome (RANK 4 G3: "the DELAY arm (no gap) does NOT collapse it")
- `_value_salience_appraisal_derisk.py` — **the existing spiking value-driven WTA choice + load-bearing lesion** (the
  RANK-1 reuse template)
- `2026-06-20-boundary-ledger-dendritic-audit.md` — the dendrite ruled out for binding + credit; the accepted-deep
  spatial-credit family (candidate #3)

**Code (verified this pass):** `_navcloseout_R4_delayed_reward_value.py` `make_delay_hook:114` (pure FIFO),
`multi_goal_schedule:91` (single moving goal), `build_episode_kwargs:209` (the spiking-default config + the delay
hook); `run_moving_goal_episode` (`g11_bg_runner.py:3256`) — single `goal_pos`/`goal_schedule`, no two-goal /
differential-reward-magnitude kwarg; `_value_salience_appraisal_derisk.py:1-56` (the spiking WTA value-choice +
lesion).

**Catalog (`sim-catalog/references/feature-catalog.md`, verified):** **O.22** Striatal action-value coding →
action-value MSNs feed an argmax-with-noise selector; sim status partial, Q(s,a) read-out not measured (`:546`/
`:547`) · **O.19** Value-based decisions: vmPFC/OFC subjective value modulates the accumulator drift; same
drift-diffusion math as perceptual choice; sim status partial (`:4996`/`:5002`) · **C.34** DA codes utility/risk →
"critical for two-arm bandit / binary-gamble / delay-discounting choice tasks" (`:640`) · **L.41** reward =
goal-object for approach + economic choice → "binary-choice tasks revealing transitive subjective preferences"
(C.34 + O.22 + O.19) (`:559`) · **G.16 / G.59** drift-diffusion bounded accumulation "holds for perceptual AND
value-based judgments" (`:2826`/`:2819`) · **C.30** actor-critic (`:594`/`:4994`) · **F.22** trace conditioning +
the delay-vs-trace × lesion 2×2 (the V-A paradigm).

**Literature (value-driven choice / economic choice / action-value, via the catalog):** Schultz 2016 (J. Neural
Transm.; Nat. Rev. Neurosci.) — striatal action value + DA codes subjective value/utility; Samejima et al. 2005
(action-value neurons); Lau & Glimcher 2008, Ito & Doya 2009 (matching-law action values); Padoa-Schioppa
(OFC economic choice); Sugrue-Corrado-Newsome (matching); Sutton & Barto 2e Ch 11 (actor-critic), Ch 6/7/12
(TD/eligibility/cue-shift). Trace-conditioning (V-A) lit: Hesslow-Yeo 2002; Moyer-Deyo-Disterhoft 1990; the
eNeuro-2025 NAc-DA-encodes-the-trace-period result; Yagishita et al. 2014 (the ~1 s eligibility window).

_READ-ONLY deep-research gate. No code, no `sim/` edit, no experiments / GPU run. The no-confab moat is
array-disjoint from the nav cascade and untouched. grid-32 is the verdict scale (never grid-8). Every load-bearing
"already-GO" / "load-bearing-by-function" claim verified against the actual finding text + the R4 raw JSONs + the
runner source + the current code._
