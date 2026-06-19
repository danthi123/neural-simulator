# Next spike-ification frontier — what to DEFAULT-ON after #4, and the honest GREEN_INERT-vs-load-bearing call (2026-06-19)

> **READ-ONLY deep-research + scoping. No code edited, no GPU run (this doc is the only write).** Produced
> per the standing "deep research + catalog review FIRST at a new direction" directive (CLAUDE.md;
> `feedback_deep_research_at_roadblocks`). The just-completed brain-based-purity chapter closed #4
> (action-decision = DEFAULT-ON, 1.16× host) and recorded #5/#3 as honest boundaries
> (`2026-06-19-spiking-decision-default-on-GO.md`, `-place-code-sparsify-default-BOUNDARY.md`,
> `-merged-TD-cueshift-opsearch-BOUNDARY.md`). The question this doc answers: **of the remaining spiking
> organs, which (if any) should be default-on'd next — and is that a genuine behavioral win or a cosmetic
> GREEN_INERT one?** Trust-but-verify the `[VERIFY]`-flagged load-bearing claims before building.

---

## 0. TL;DR for the controller (the honest crux first)

- **The remaining nav organs are ALREADY BUILT, validated, AND lifted onto the merged "one brain" as opt-ins.**
  Since the 2026-06-18 roadmap audit, the limbic core (`co_resident_limbic`), the FULL nav reward/critic
  (`co_resident_nav_critic`), the neural reward route, and the value-train all SHIPPED — the merge note's "no
  limbic core at all" is **out of date**. So "default-on the remaining nav organs" is no longer a *build*; it
  is a one-line **default flip** (mostly) on already-co-resident, moat-safe slices.
- **BUT every one of those nav-organ default-flips is GREEN_INERT, not a behavioral win.** The merged gridworld
  is **orient-solvable** — the spiking superior colliculus + BG cascade reach the goal *without* a reward/value
  gradient — and the project has now **measured this directly**: the neural-reward-on-the-one-brain finding
  reports nav **NOT regressed** by switching reward host→neural precisely because "the reward is not strongly
  behaviorally load-bearing" (`2026-06-18-merged-neural-reward-GO.md` §3). Default-on'ing reward/value/dopamine
  on this task would be **cosmetic** (co-resident + non-regressing, like the nav-gate-(a) byte-identical-inert
  result), exactly the low-value pattern the prompt warns about. The mechanism is real; the *deployment* is
  inert because nothing in the task needs it.
- **⇒ The strategic conclusion: do NOT spend the next chapter chasing more cosmetic GREEN_INERT nav default-ons.**
  The honest move is one of two things, and they are **complementary**, not exclusive:
  - **(B) Give the limbic core a task that makes it LOAD-BEARING** — a harder embodied task (a Morris-water-maze-
    style "no orienting cue, learn-where-reward-is" navigation) where value/dopamine *must* drive behavior. This
    **converts the GREEN_INERT organs into genuine wins** (the same default-on becomes load-bearing) AND is the
    canonical setting for the project's own validated spiking actor-critic. This is the highest-leverage
    *substrate-aligned* move.
  - **(#6 / DA→composer) the ONE non-cosmetic nav→cortex closure already scoped** — wire the shared spiking
    dopamine to the conversational composer's cleanup sharpness (`OneBrainComposer.confidence_gate`). This is the
    "one self" closure (the limbic core reaches the cortex on BOTH halves), is moat-safe-by-construction, and
    exercises the substrate *functionally* — but its own GO bar may find "no recall headroom" (a real result that
    re-points to encoding-gating).
- **#1 RECOMMENDATION (ranked below): start (B) — the harder embodied task — with a cheap-first standalone
  `g11_bg_runner` Morris-maze-style probe that establishes the host-reward baseline FIRST (does the spiking SC
  orienting alone solve it? if yes, the task is still orient-solvable and must be hardened further), THEN the
  load-bearing test (does the neural reward/value/dopamine limbic core beat a reward-lesion on the SAME task?).**
  This is the cheapest decisive probe of the one thing that makes ALL the nav-organ default-ons worth shipping,
  and its anti-cheat is the lesion the project already trusts (sever `sc_rostral→reward_us` → behavior collapses,
  not just the SNc burst). Default-on'ing the limbic core on the *current* task without this is shipping a
  cosmetic green.

---

## 1. The remaining-default-on inventory (post-#4) — verified against the repo

Legend: **DEFAULT** = on in the deployed default loop. **OPT-IN(built)** = validated + co-resident on the
merged bridge behind a flag, a default flip away. **OPT-IN(standalone)** = validated in `g11_bg_runner` but the
merge wiring exists. Each row: (a) validated? (b) buildable/on-merge? (c) default-on gate (d) cheap-first de-risk
(e) anti-cheat — and the honest GREEN_INERT-vs-load-bearing call (§2 expands).

### 1a. Navigation organs

| Organ | State | (a) Validated | (b) On the merged bridge? | (c) Default-on gate | (d) Cheap-first de-risk | (e) Anti-cheat |
|---|---|---|---|---|---|---|
| **Action-decision read-out** (the #4 deliverable) | ✅ **DEFAULT-ON** | YES (6-seed, 1.16× host, 100% commit-burst) | YES — `run_moving_goal_episode` default `readout_source="spiking_wta"` | **DONE** (CYCLE 235) | — | — (array-disjoint from parser/composer → moat by construction) |
| **Spiking SC orienting** (`--enable-spiking-sc`) | OPT-IN(standalone), wired to merge via the N5/SC-reward path | YES (6-seed beats host reflex; scrambled-retinotopy lesion regresses 2.4×) | YES — built when the SC-reward flags are threaded through `run_moving_goal_episode` (`g11_bg_runner.py`, the `install_spiking_sc_wiring` call site) | nav-not-regressed (6-seed) + conv answer-identity 15/15 | env-var op-point already found (`SC_RET_SC`/`SC_REC`/`SC_ROS_US`/`SC_RET_DRIVE`, het-off merged tuning) | scrambled-retinotopy lesion → orienting collapses |
| **Neural reward `r`** (`--enable-spiking-sc-approach`/`--spiking-reward-us`) | ✅ **OPT-IN(built), GO on the merged bridge** | YES — **6-seed RPE battery GO on the merge** (`2026-06-18-merged-neural-reward-GO.md`): graded proximity corr ≤ −0.5, SNc burst ≥ 1.3×, **lesion collapses it** | YES — the `:7140` reward-routing fix + env-var op-point already shipped (runner-only, byte-identical when off) | DONE (the 6-seed RPE battery is the de-risk) | **sever `sc_rostral→reward_us` → `reward_us`→0** (decisive: `r` is the synaptic SC proximity, not a host scalar) | same lesion |
| **Spiking SNc + the limbic core** (`co_resident_limbic` / `co_resident_nav_critic`) | ✅ **OPT-IN(built), lifted onto the merge** | YES — standalone Schultz RPE battery 6/6 (`2026-06-18-limbic-core-rpe-battery-GO.md`); the **full nav critic lift BUILDS-CLEAN (45 regions) + f-I-restored + MOAT-safe** (`2026-06-18-merged-limbic-core-lift.md`, AUTONOMOUS_STATE CYCLE 209) | YES — `build_merged_nav_conv_bridge(co_resident_nav_critic=True)` (the full nav reward/critic) or `co_resident_limbic=True` (the minimal organ); **mutually exclusive** (asserted) | nav-not-regressed + conv 15/15 + the RPE-battery gates hold co-resident | the RPE battery (DONE) + **a load-bearing behavioral test** (the GAP — see §2) | reward-lesion (§2) + critic-GABA_B lesion (value collapses) |
| **Place / position code** | OPT-IN(built); host-Gaussian `vs_place_context` is the better-δ scaffold | self-org place COMPOSES (value-(a) GO) but the **δ-lift is a BOUNDARY** (all-or-none critic read-out can't grade) | YES — `nav_critic_place_selforg` opt-in (NOT the default critic afferent; GO bar δ>1.3 unmet) | DONE (the boundary is the finding) | **DO NOT default-on** — needs a graded-rate read-out or the dendrite | n/a (boundary) |
| **TD-bootstrap cue-shift** (Schultz signature (a)) | OPT-IN(standalone GO); merged = BOUNDARY | standalone GO lesion-clean 3/3; **merged reached r=−0.719 but FAILS the cue-lesion anti-cheat** | merged slice lifts clean; the migration does not | DONE (the boundary is the finding) | **DO NOT default-on the merged migration** (a merged-SNc onset transient survives the lesion) | the cue-lesion (which is exactly why it's a boundary) |

### 1b. Conversational organs (the cortex side)

| Organ | State | (a) Validated | (b) On the one brain? | (c) Default-on gate | (d) Cheap-first de-risk | (e) Anti-cheat |
|---|---|---|---|---|---|---|
| **Spiking NEF cleanup** (`enable_spiking_cleanup`) | OPT-IN(built); numpy argmax is the production default | YES — validated == numpy on the capability matrix at D=2048 multi-seed (`CoreSimComposer`); a thresholded NEF matched filter + Izhikevich WTA | YES — `CoreSimComposer(enable_spiking_cleanup=True)` builds the cleanup bridge from the codebook | answer-identity (full who/what matrix incl. the `is None` abstentions) + the 11-test CI verbatim + no latency regression | a small A/B: spiking-cleanup vs numpy argmax answer-identical at D=2048, 6-seed | the no-confab moat (abstain decisions bit-identical) |
| **Neural sentence-render** (`enable_neural_render`) | ✅ **EFFECTIVELY DEFAULT in the demos** | YES — the SVO frame's word order is produced by the spiking competitive-queuing read-out (6/6); permuted/lesion collapse | YES — `BrainConversationalAgent(enable_neural_render=True)`; the two production demos (`consolidated_320_conversation_demo`, `multi_turn_conversation_demo`) pass it `True` | answer-identity + the no-confab moat (`test_neural_render_describe`) | DONE | permuted-order + no-learning controls collapse to chance |

**Key correction to the prompt's premise:** the prompt's inventory (drawn from the 06-18 audit) lists the limbic
core / neural reward / spiking SNc as "default = host, built only behind flags / only on the standalone runner."
**That is now stale** — the 06-18 audit was the *opening* of the TRUE-ONE-BRAIN arc; CYCLEs 206–219 then *built*
the lifts. The accurate post-#4 state is: **the nav limbic organs are co-resident on the merged bridge,
validated, moat-safe, and a default flip away — but their behavioral inertness on the orient-solvable task is the
reason none has been flipped, and is the load-bearing strategic fact.** `[VERIFY: read
`nav_conv_merged_bridge.py` for the `co_resident_limbic` / `co_resident_nav_critic` kwargs and the
`merged-neural-reward-GO` / `merged-limbic-core-lift` findings — both confirm built + validated + moat-safe + the
nav-not-regressed/orient-solvable caveat.]`

---

## 2. The honest GREEN_INERT-vs-load-bearing assessment (the crux)

A "default-on" is only a *win* if the organ is **load-bearing** — if removing it (lesion) changes the deployed
behavior. The project's own discipline names the failure mode precisely: the nav-gate-(a) consolidation was
"byte-identical-inert" (GREEN_INERT) — co-resident and non-regressing but not a genuine behavioral win
(AUTONOMOUS_STATE; `feedback_validate_signal_by_its_function`: "validate a signal by its function, not a task
that ignores it"). Assessing each remaining nav organ against that bar:

| Organ | Default-on outcome on the CURRENT (orient-solvable) task | Evidence | Verdict |
|---|---|---|---|
| **Spiking SC orienting** | **Partly load-bearing** — orienting IS what solves the orient-solvable task, so a spiking SC that orients correctly *is* behaviorally load-bearing (the scrambled-retinotopy lesion regresses nav 2.4×). This is the one nav organ whose default-on is a genuine win. | the lesion regression (the audit) | **GENUINE WIN** (the task rewards orienting) |
| **Neural reward `r`** | **GREEN_INERT** — the finding measured it: nav NOT regressed host→neural reward "because the gridworld is orient-solvable, so the reward is not strongly behaviorally load-bearing." | `2026-06-18-merged-neural-reward-GO.md` §3 (explicit) | **COSMETIC** on this task |
| **Spiking SNc / dopamine / value critic (limbic core)** | **GREEN_INERT** — the DA δ=r−V gates the actor's three-factor *learning*, but the orient-solvable task is solved by the orienting pathway without needing the value-gated learning to converge. The lift is "BUILDS-CLEAN + f-I-restored + MOAT-safe" — none of which is a behavioral win. | `merged-limbic-core-lift.md` (lift is structural); the δ-on-task is not load-bearing because the task is orient-solvable | **COSMETIC** on this task |
| **Place code** | n/a — BOUNDARY (δ can't grade); not a default-on candidate | `place-code-sparsify-default-BOUNDARY.md` | — |
| **TD cue-shift** | n/a — BOUNDARY (cue-lesion fails on merge) | `merged-TD-cueshift-opsearch-BOUNDARY.md` | — |

**The decisive synthesis.** Of the remaining nav organs, **only the spiking SC orienting** is a genuine default-on
win on the current task — and it is a *breadth* win (the position code becomes spiking), not a new behavior, the
same shape as #4. **The reward/value/dopamine limbic core — the project's highest-leverage SHARED system per the
owner's own #1 priority — is GREEN_INERT on this task.** Shipping it default-on now would be "co-resident +
non-regressing but not a genuine behavioral win." Per the prompt's instruction ("if GREEN_INERT, SAY SO — a
cosmetic default-on is low-value"): **the limbic-core default-on is low-value UNTIL the task makes it load-bearing.**
This is not a failure of the organs (they are validated at the mechanism level by the full Schultz RPE battery);
it is a property of the *task* — and it is the exact lesson `feedback_validate_signal_by_its_function` records.

This reframes the whole "default-on the rest" axis: the cheap default flips are cosmetic; the real work is **either**
(a) make the limbic core load-bearing with a harder task, **or** (b) exercise the substrate functionally via the
DA→composer closure (#6) / a conversational capability, where the substrate IS genuinely worked.

---

## 3. The strategic fork + RANKED recommendation

Four candidate directions, scored on **leverage × cheapness × alignment-with-"everything-spiking" × genuine-vs-
cosmetic**. The North star (`project_actual_goal_artificial_life_brain_analogue`): a biology-faithful
artificial-life brain; honest negatives are the deliverable; capabilities are instrumental.

### Option A — default-on the remaining nav organs (the #4 pattern)
- **What:** flip `spiking_sc` orienting + `co_resident_nav_critic` (reward/value/dopamine) to default on the merged
  nav loop. Mostly a one-line default flip (already built + validated + moat-safe).
- **Leverage: LOW–MEDIUM.** The SC-orienting flip is a genuine breadth win (one more host shortcut → spiking, like
  #4). The reward/value/dopamine flip is **GREEN_INERT / cosmetic** (§2) — it ships "everything spiking" on paper
  but changes no behavior, and shipping a cosmetic green dilutes the honest-deliverable standard.
- **Cheapness: HIGHEST** (default flip + the two existing gates).
- **Alignment: HIGH** (literally "everything onto the shared spiking substrate").
- **Genuine vs cosmetic: SPLIT** — SC-orienting genuine; limbic-core cosmetic on this task.
- **Verdict:** do the **SC-orienting** flip as a cheap breadth win (it's a real win); **do NOT** ship the
  limbic-core default-on as a headline (it's cosmetic until Option B). Ranked **#3** overall (the SC half is worth
  a small cycle; the limbic half should wait for B).

### Option B — a HARDER embodied task that makes the limbic core LOAD-BEARING  ⟵ **#1 RECOMMENDATION**
- **What:** replace/augment the orient-solvable gridworld with a task where **value must be learned** — the
  canonical **Morris-water-maze analogue** (no beacon/orienting cue to the goal; the agent must learn *where the
  reward is* from experience, so the dopamine RPE + value critic are the only path to the goal). The project's own
  validated spiking actor-critic is exactly this setting: the Frémaux–Sprekeler–Gerstner continuous-time spiking
  actor-critic (PLOS Comput Biol 2013) **solves a Morris-water-maze navigation task in animal-consistent trial
  counts** — a reward-load-bearing task, *not* orient-solvable.
- **Leverage: HIGHEST (substrate-aligned).** This is the one move that **converts the GREEN_INERT organs into
  genuine wins** — once the task needs value/dopamine, default-on'ing the limbic core is load-bearing, and the
  reward-lesion anti-cheat (`sever sc_rostral→reward_us` → *behavior* collapses, not just the SNc burst) becomes a
  real behavioral discriminator. It directly serves the owner's #1 shared-system priority (reward/value/dopamine
  limbic core) AND the North star (an artificial-life agent that *learns from reward*, the deepest biology). It also
  re-opens #3 (the TD cue-shift) and #5 (place code) as *needed* rather than cosmetic — a hidden-goal maze needs a
  graded value over a place code.
- **Cheapness: MEDIUM.** No new organ (all built); the new work is the **environment + reward render** (legitimate
  host per the brain-based bar: world state + sensory render) and a curriculum. The cheap-first probe (below) is a
  standalone `g11_bg_runner` run, days not weeks.
- **Alignment: HIGH** — it doesn't move anything OFF the substrate; it makes what's ON the substrate *matter*.
- **Genuine vs cosmetic: GENUINE.** A reward-load-bearing task is the definition of "not cosmetic."
- **Honest risk:** the first probe may find the task is **still** orient-solvable (the spiking SC orients even
  without a beacon, from incidental cues) — in which case the deliverable is "the task must be hardened further
  (remove cue X)", a clean iterative result, not a failure. This is *why* the cheap-first probe establishes the
  orienting-only baseline FIRST.
- **Verdict:** **#1.** Highest leverage, substrate-aligned, genuine, and it is the move that *redeems* the entire
  GREEN_INERT nav-organ inventory. It is also the natural continuation of the owner's TRUE-ONE-BRAIN #1 priority
  (the limbic core), which is currently built-but-inert.

### Option #6 — DA → the conversational composer (the "one self" closure)  ⟵ **#2 RECOMMENDATION**
- **What:** wire the shared spiking dopamine to the composer's cleanup sharpness (`OneBrainComposer.confidence_gate`
  rises with DA = the D1 inverted-U "sharpen tuning by suppressing nonpreferred responses"). The limbic core reaches
  the cortex on BOTH halves — the deepest "one self" step. **Fully scoped already**
  (`2026-06-18-DA-NM-composer-closure-scoping.md`, Option A): NO `sim/` edit (composer-runner-layer read of
  `get_concentration("dopamine")`), **moat-safe by construction** (DA can only *raise* the gate → stricter
  abstention, never a new false-accept).
- **Leverage: MEDIUM–HIGH (genuine integration, not cosmetic).** Unlike the nav default-ons, this *functionally
  couples* two substrate halves that currently don't interact — it exercises the substrate. It is the owner-named
  NEUROMOD shared-system closure (the audit's roadmap #6).
- **Cheapness: HIGH** (reuse-by-import, NO `sim/` edit, numpy-CPU cheap-first; the knob already exists from the
  graceful-degradation work).
- **Alignment: HIGH** (a shared spiking signal reaching a second consumer = "one brain").
- **Genuine vs cosmetic: GENUINE-but-bounded.** The honest GO bar is two-limbed and may find **"no recall headroom"**
  (the composer's cleanup is already decisive in the functional regime) → a real result that re-points to
  encoding-gating (Option B of that scoping doc, the Lisman–Grace novelty→LTP hook). So it's genuine integration but
  its *measurable* effect may live only in the degrading/encoding regime.
- **Verdict:** **#2.** Cheaper than the harder-task build, genuinely functional (not cosmetic), zero `sim/` edit,
  moat-safe. Excellent to run *in parallel with or right after* the Option-B cheap-first probe (they're file-disjoint
  — B is `g11_bg_runner`/env, #6 is the composer runner). If B's environment build is non-trivial, #6 is the
  faster genuine win to land first.

### Option C — conversational architecture to basic-LLM-competitive (the prior PRIMARY, task #55)
- **What:** the productive-syntax breadth pass (learned multi-frame render into the agent + learned query-frame
  comprehension + constituent-as-slot) — the prior PRIMARY direction, fully scoped
  (`2026-06-17-capability-frontier-to-basic-LLM-scoping.md`, Option 1).
- **Leverage: HIGH for the capability target** (closes real LLM-gap KINDS: novel-order comprehension, flexible Q&A,
  constituency) — **and the substrate IS genuinely exercised** (the prompt's note: this is where the substrate is
  worked, unlike the cosmetic nav flips).
- **Cheapness: HIGH** (reuse-by-import, NO `sim/` edit, numpy cheap-first in minutes; everything on validated
  point-neuron pieces).
- **Alignment: MEDIUM** with "everything-spiking" specifically (it adds *capability* on the already-spiking
  conversational substrate rather than moving host→spiking) — but HIGH with the deeper North star (a conversing
  artificial-life brain).
- **Genuine vs cosmetic: GENUINE.** New capability on the working substrate.
- **Verdict:** **#4** *for the "everything-spiking" framing specifically* (it's already spiking; it adds capability,
  not spike-ification), but a strong standalone direction if the owner re-prioritizes "compete with a basic LLM"
  over "everything onto the shared substrate." It does not advance the spike-ification axis the current directive
  names, so it ranks below B/#6/A-SC-half on *this* fork — while remaining the top capability direction.

### Option D — the dendrite unlocker (learnable multi-attribute COMPOSITION)
- **What:** the deepest open conversational blocker — multi-attribute BUNDLING is not learnable from scratch on
  point neurons; the candidate unlocker is dendritic multiplication (`fused_coincidence_plateau`, already on the
  bridge, guarded). The cheapest-first is a CPU/numpy A/B re-running the 2026-06-16 bundling NEGATIVE harness with
  a **fixed self-inverse bind + learned filler codes** (the untested middle), fully specified, confirmed un-run
  (`2026-06-18-step3-dendritic-learned-bind-frontier-scoping.md`).
- **Leverage: HIGHEST capability ceiling, HIGHEST risk.** The dendrite is fair game (`feedback_dendritic_substrate_fair_game`)
  and may be the one place point neurons provably can't reach. But the cheapest A/B might show the fixed-self-inverse
  bind already bundles (= no learned-dendritic justification), or the wall holds (the deliverable).
- **Cheapness: the *probe* is cheap** (an afternoon CPU A/B); the *build* (if justified) is weeks-scale additive
  wiring.
- **Alignment: HIGH with the North star, LOW with "everything-spiking onto the SHARED substrate"** (it's a
  conversational-cortex capability deepening, on a *new* substrate primitive, not consolidation/spike-ification).
- **Genuine vs cosmetic: GENUINE** but pre-registered with a build gate not yet met.
- **Verdict:** **#5.** The deepest lever but the least aligned with the *current* spike-ification directive, and
  correctly sequenced after the cheaper localizing work. The owner un-benched it; keep the cheap A/B as a queued
  decisive probe, do not commit the build before it.

### The ranking, one line each
1. **(B) Harder embodied task** — makes the limbic core LOAD-BEARING; redeems the whole GREEN_INERT nav inventory; substrate-aligned; genuine. **#1.**
2. **(#6) DA → composer** — the genuine "one self" cortex closure; cheap; NO `sim/` edit; moat-safe; run in parallel with B's env build. **#2.**
3. **(A-SC half) default-on the spiking SC orienting** — a real (not cosmetic) breadth win; one cheap cycle. (The limbic-core half of A waits for B.) **#3.**
4. **(C) conversational → basic-LLM** — top *capability* direction; genuinely exercises the substrate; but it's *adding* capability, not *spike-ifying*, so it ranks below on the current directive. **#4.**
5. **(D) dendrite** — highest ceiling/risk; keep the cheap A/B queued; do not build before it localizes. **#5.**

---

## 4. The #1 cheapest-first FIRST step + its anti-cheat (Option B)

**Goal of the first step (cheap, decisive, standalone — no merge wiring, no GPU contention with conversation):**
establish whether a reward-load-bearing task can be built on the existing nav machinery, and whether the validated
limbic core is *behaviorally* load-bearing on it — the one thing that turns the GREEN_INERT organs into genuine
wins.

**Step 1 — the orienting-only BASELINE (does the task even need reward?).** Build the cheapest hidden-goal variant
in `g11_bg_runner`: a moving/fixed goal with **the beacon removed from the retina** (the goal is *not* rendered —
`render_gridworld_to_image(goal_pos=None)` / no goal-cue channel), so the spiking SC has no orienting target. Run
the **spiking SC orienting + BG cascade with reward OFF** (host or neural, doesn't matter — the point is the agent
has no value signal). Measure nav score.
- **If the agent still solves it** (score ≈ the beacon task): the task is *still* orient-solvable from incidental
  cues → **harden further** (remove the next cue; this is the clean iterative deliverable, not a failure).
- **If the agent fails** (score ≈ random walk): the task is now reward-load-bearing → proceed to Step 2.

**Step 2 — the LOAD-BEARING test (does the limbic core earn its keep?).** On the hardened (reward-load-bearing)
task, A/B the **full neural limbic core** (`spiking_snc + enable_neural_critic + spiking_reward_us`, the validated
δ=r−V) against itself with the **reward pathway LESIONED**:
- **intact:** reward/value/dopamine drive the actor's three-factor learning → the agent learns where the reward is →
  nav score improves over trials toward the goal.
- **lesion (`sever sc_rostral→reward_us`):** the reward `r`→0 → no RPE → no value learning → nav score stays at the
  no-value floor.
- **GO:** intact ≫ lesion on nav score, ≥5/6 seeds (the standing 6-seed rule for a variable effect), AND the
  learning curve shows trial-over-trial improvement only in the intact arm.

**The anti-cheat (the load-bearing discriminator).** The lesion must collapse the *behavior*, not merely the SNc
burst — this is the exact upgrade `feedback_validate_signal_by_its_function` demands ("N5 reward looked validated by
a nav A/B but the task was orient-solvable; the proper test was the dopamine-RPE battery with the reward sourced from
neurons + a lesion anti-cheat — match the control to what the signal computes"). Concretely:
1. **Reward-pathway lesion → behavior collapses** (sever `sc_rostral→reward_us`; the agent can no longer find the
   hidden goal). If nav is *unchanged* by the lesion, the task is **still orient-solvable** → it is NOT yet a
   load-bearing task (go back to Step 1 / harden more). This is the decisive cosmetic-vs-genuine test.
2. **Critic (value) lesion** (zero the `striosome_value→snc` GABA_B conductance) → if the agent still learns the
   goal, value isn't load-bearing (a reward-only / Rescorla-Wagner regime suffices) — a finer-grained, still-honest
   result.
3. **Reward stays NEURAL** (`r` = `reward_us` firing, not a host scalar written to `current_reward_signal`) — the
   lesion proves it; a host-scalar reward would be lesion-insensitive at the firing level. Host residual is limited
   to the environment (the maze + the reward render) and the body (acting on the motor pool), per the brain-based bar.
4. **The no-confab moat intact throughout** (the conversational gates 15/15) when/if the limbic core is later
   default-on'd on the merged bridge — already moat-safe (the limbic slice is nav-inert, array-disjoint).
5. **6-seed for the behavioral effect; numpy only for the tiny Step-1 baseline smoke, CuPy for the decisive Step-2
   A/B** (`feedback_gpu_not_numpy`).

**Expected wall-clock:** Step 1 is a few standalone `g11_bg_runner` runs (~the existing nav episode cost, minutes
each, numpy-smoke-able for the baseline). Step 2 is a 6-seed GPU A/B (the existing nav episode + the validated
limbic critic, hours). No new organ; the new code is the environment/reward render (host, legitimate) + the
lesion harness (reuse `sc_n5_rpe_probe.py`'s sever pattern).

**Why this is the right cheap-first:** it tests the ONE thing that determines whether *any* limbic-core default-on
is a genuine win or cosmetic (is the task load-bearing?), it reuses 100% validated organs (no new biology), its
three outcomes each cleanly route the next move (still-orient-solvable → harden; load-bearing + lesion-collapse →
default-on the limbic core as a *genuine* win; load-bearing but lesion-survives → an honest negative about the
substrate's value-learning on this task), and it directly executes the project's own
`feedback_validate_signal_by_its_function` discipline.

---

## 5. Cross-check against the owner memories (MEMORY.md pointers)

- **`project_actual_goal_artificial_life_brain_analogue`** (artificial-life brain; honest negatives ARE the
  deliverable; capabilities are instrumental): **directly supports #1 (B).** A reward-load-bearing task is the
  deepest artificial-life behavior (an agent that learns from reward), and the honest "still orient-solvable →
  harden" / "lesion-survives" outcomes are themselves deliverables. It also supports *not* shipping cosmetic green
  default-ons (they aren't honest wins).
- **`feedback_validate_signal_by_its_function`** ("validate a signal by its function, not a task that ignores it"):
  **the load-bearing memory for this whole fork.** It is the exact pattern (N5 reward looked validated by an
  orient-solvable A/B; the proper test needed the reward sourced from neurons + a lesion anti-cheat). §2's
  GREEN_INERT call and §4's Step-1-baseline-then-lesion design are this memory applied literally.
- **`feedback_move_everything_to_shared_spiking_substrate`** (the top-level directive): B keeps everything ON the
  substrate and makes it *matter*; #6 adds a genuine cross-half spiking coupling; the SC-orienting flip (A-half) is
  a real host→spiking conversion. The honest nuance the memory itself implies: "everything spiking" is only
  meaningful if the spiking organs are *doing the work* — which is precisely the GREEN_INERT problem B solves.
- **`feedback_moat_not_hard_lossy_memory_ok`** (moat is a plus, not a hard gate; lossy memory OK if it buys
  scaling/dev-speed while biological): #6 (DA→composer) is *free* to keep the moat (DA can only tighten it), so it
  honors this with no trade. B doesn't touch the moat (nav-inert limbic slice). No proposal here trades the moat.
- **`feedback_dendritic_substrate_fair_game`** (the dendrite is fair game; don't bench for length; build when the
  obvious unlocker): supports keeping Option D's cheap A/B queued (it's the obvious unlocker for multi-attribute
  composition) — but the memory also says "do it when it's the *obvious* unlocker," and the current directive is
  spike-ification/consolidation, so D is correctly #5 (the cheap localizing A/B first, the build only if justified).
- **`feedback_dont_gate_on_approval`** (don't gate on approval; sim/ edits fine if justified): #1 (B) and #2 (#6)
  need **no `sim/` edit** and no approval gate — proceed cheapest-first. (B's environment build is host/runner; #6
  is composer-runner.)
- **`project_one_brain_substrate_vs_functional`** ("one brain" is substrate consolidation, not functional
  integration — co-location ≠ interaction): **directly motivates ranking #6 above the cosmetic nav flips.** The
  nav-organ default-ons are co-location (GREEN_INERT); #6 is the functional cross-region interaction the memory
  names as the real, highest-leverage one-brain work. And B makes the co-located limbic core functionally
  load-bearing within the nav half.

**No contradiction** between the recommendation and the memories: the artificial-life North star + the
validate-by-function discipline + the substrate-vs-functional distinction all converge on "don't ship cosmetic
green; make the limbic core load-bearing (B) and let it functionally reach the cortex (#6)."

---

## 6. Honest scope / non-claims

- I did **not** re-litigate the settled pieces: the FHRR bind (principled idealization), the generalizing PPMI
  stream cortex (closed-positive on point neurons), or the #4 action-decision (DEFAULT-ON). Those are out of scope.
- The GREEN_INERT call is **task-specific, not organ-specific** — the limbic organs are validated at the mechanism
  level (the full Schultz RPE battery 6/6, co-resident, moat-safe). Their *deployment* inertness is a property of
  the orient-solvable gridworld, which is exactly what B fixes. This is not a knock on the organs; it is the
  strategic reason a harder task is the next leverage point.
- Option B's first probe may find the task still orient-solvable (the spiking SC orients from incidental cues even
  without a beacon). That is a *clean iterative deliverable* (harden cue X), not a failure — and it is why the
  cheap-first establishes the orienting-only baseline before the lesion A/B.
- Option #6's GO bar may find "no recall headroom" (the cleanup is already decisive in the functional regime) → a
  real result re-pointing to encoding-gating (Lisman–Grace novelty→LTP), per its own scoping doc. Either outcome is
  a deliverable.
- The inventory correction (§1: the limbic organs are already lifted onto the merged bridge, not "standalone-only"
  as the 06-18 audit framed) is `[VERIFY]`-flagged and should be confirmed against `nav_conv_merged_bridge.py` +
  the `merged-neural-reward-GO` / `merged-limbic-core-lift` findings before building.

---

## 7. EXACT NEXT

Start **Option B** with the **Step-1 orienting-only baseline** (cheapest, decisive, standalone): a hidden-goal
`g11_bg_runner` variant (beacon removed from the retina render) run with the spiking SC orienting + BG cascade and
reward OFF — measure whether the agent still solves it. If it does, harden further (the clean iterative result); if
it fails, the task is reward-load-bearing → run the **Step-2 limbic-core A/B with the reward-pathway lesion**
(intact ≫ lesion on nav score, 6-seed, the lesion collapses *behavior* not just the SNc burst). On a load-bearing
GO, default-on the limbic core as a *genuine* win (moat-safe, the conversational gates 15/15). **In parallel**
(file-disjoint), land **Option #6** (the DA→composer cheap-first de-risk from `2026-06-18-DA-NM-composer-closure-scoping.md`
§4: map DA→`confidence_gate`, clamped to only sharpen; the moat held-or-stricter at every DA level; the lesion
abolishes the effect) — the genuine "one self" closure with NO `sim/` edit. Reuse-by-import throughout; honest
negatives (still-orient-solvable; no-recall-headroom) ARE the deliverable.

---

### Catalog + literature anchors (cited)
- **C.28 / C.30 / C.31** TD error / actor-critic / bootstrapping (the limbic-core structure: SNc=δ, striosome=critic
  V, matrix=actor) — the value the harder task makes load-bearing. **C.32** two-component DA (salience = the #6
  composer hook). **C.33** PPN→SNc reward driver. **C.04** mesocortical DA→PFC (the same DA, two consumers). **C.22 /
  O.02** Schultz RPE signatures (the RPE battery). Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.
- **Frémaux, Sprekeler & Gerstner 2013**, *PLOS Comput Biol* 9(4):e1003024 — **continuous-time spiking actor-critic
  solving a Morris-water-maze navigation task** in animal-consistent trial counts (reward-modulated STDP, TD critic).
  The canonical reward-LOAD-BEARING spiking navigation setting = the Option-B task model.
  [DOI](https://doi.org/10.1371/journal.pcbi.1003024) (PMID 23592970).
- **Vijayraghavan, Wang, Birnbaum, Williams & Arnsten 2007**, *Nat Neurosci* 10:376–384 — D1 inverted-U "sharpens
  tuning by suppressing nonpreferred responses" (the #6 cleanup-sharpening mechanism + the moat-safety reason DA must
  *raise* not maximize). [DOI](https://doi.org/10.1038/nn1846) (PMID 17277774).
- **Lisman & Grace 2005**, *Neuron* 46:703–713 — hippocampal-VTA novelty→DA→LTP loop (the #6 encoding-gating
  follow-on). [DOI](https://doi.org/10.1016/j.neuron.2005.05.002) (PMID 15924857).

### Project findings reviewed
`2026-06-18-full-spikeification-shared-substrate-roadmap.md`, `-merged-limbic-core-lift.md`,
`-merged-neural-reward-GO.md`, `-limbic-core-rpe-battery-GO.md`, `-DA-NM-composer-closure-scoping.md`,
`-step3-dendritic-learned-bind-frontier-scoping.md`; `2026-06-19-spiking-decision-default-on-GO.md`,
`-place-code-sparsify-default-BOUNDARY.md`, `-merged-TD-cueshift-opsearch-BOUNDARY.md`, `-latency-csr-cache-GO.md`;
`2026-06-17-capability-frontier-to-basic-LLM-scoping.md`; `research/findings/AUTONOMOUS_STATE.md` (CYCLEs 205–235).
Memories: `feedback_move_everything_to_shared_spiking_substrate`, `feedback_validate_signal_by_its_function`,
`project_actual_goal_artificial_life_brain_analogue`, `project_one_brain_substrate_vs_functional`,
`feedback_moat_not_hard_lossy_memory_ok`, `feedback_dendritic_substrate_fair_game`, `feedback_dont_gate_on_approval`.
