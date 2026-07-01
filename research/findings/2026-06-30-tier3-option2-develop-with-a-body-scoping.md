# Tier-3 Option 2 — "develop-with-a-body" — deep-research / design scoping (READ-ONLY)

**Date:** 2026-06-30 (autonomous loop; owner-directed Tier-3 follow-on)
**Type:** Design / scoping. **READ-ONLY — NO code / `sim/` / GPU edit.** This doc isolates and answers the ONE genuine
design question for the SECOND Tier-3 synthesis slice: *can the `MergedNavConvAgent` (a perceiving/composing body) BE
the `develop_gpu` day-loop's per-day agent, with each day's knowledge LIVED (foraging perception) instead of the
scripted corpus WAKE — feeding the validated SLEEP/GROW/PERSIST/METRICS/BUNDLE scaffold — and where does that JOIN
cleanly vs where does it NOT?*

**Predecessors (do NOT re-derive):**
`2026-06-30-tier3-artificial-life-capstone-scoping.md` §4 Option 2 (the ranked follow-on this doc details) ·
`2026-06-30-tier3-live-and-remember-first-slice.md` (Option 1, **6/6 GO** — the live-perceive-ground-store loop this
slice reuses as its WAKE) · `2026-06-24-week1-develop-loop-console-capstone.md` (a brain develops over a simulated
week, scripted curriculum) · `2026-06-23-longitudinal-develop-loop-GPU-GO.md`.

Term defined once: **WAKE** = the develop-loop day-stage where the brain acquires the day's knowledge (today: hears a
text corpus + is taught authored facts). **The JOIN** = replacing that scripted WAKE with a live-and-remember
foraging day whose facts come from what the body PERCEIVED. **Two-substrate seam** = the fact that the develop-loop's
per-day brain is `StreamCortex` (text-corpus co-occurrence cortex) + a `build_agent` conversational agent, whereas the
live-and-remember body is a `MergedNavConvAgent` (grounds codes from LIVE perception) — different objects, wired
differently.

---

## 1. TOP-LINE — largely-done-in-pieces, or a genuine build?

**Largely-done-in-pieces, with a SMALL genuine residual — but the residual is bigger than Option 1's was, and it has
ONE real structural seam (not just an assembly).** Applying the SURPASS practice (pin the exact new bytes):

- **What already exists and is validated:** BOTH halves are separately GO. The **live-perceive-ground-store day** is
  the Option-1 `live()` loop (`_tier3_live_and_remember_derisk.py`, **6/6 GO**, 2026-06-30) — it already produces
  exactly the artifact a "lived WAKE" needs: `state.lived_facts` (`[(prev,"near",cur)]`) + `state.grounded_codes`
  (obj → phasor) from the agent's OWN drive-biased trajectory. The **day-loop scaffold** (WAKE→CONVERSE→SLEEP→GROW→
  METRICS→PERSIST + per-day bundles + the 24/7 pausable/resumable `develop_run.py`/`develop_loop_supervisor.py`
  harness) is `develop_gpu` (`_longitudinal_develop_loop_gpu.py`), robust and GO.

- **The genuine residual (the exact new bytes) — three items, one of them a real seam:**
  1. **(R-a) A per-day agent-substitution seam in `develop_gpu`.** Today the day's brain is HARD-BUILT inside the
     loop as a text-`StreamCortex` + a `build_agent` conversational agent (`_longitudinal_develop_loop_gpu.py:351`,
     `:407`). To let the `MergedNavConvAgent` be the per-day brain, `develop_gpu` needs a **per-day-agent factory
     seam** (an additive default-`None` callable, byte-identical when unset), exactly analogous to the
     `per_day_save_hook`/`should_continue` seams it already grew. This is genuinely NEW plumbing (not present today),
     but it is small and additive.
  2. **(R-b) The WAKE-source swap.** Replace `cortex.hear_day(...)` + `curriculum.day_stream(day)["new_concepts"]/
     ["facts"]` (`:393`, `:416`) with a live foraging day (the Option-1 `live()` call) whose emitted `lived_facts`
     become the day's facts and whose `encountered` objects become the day's vocab-growth. This is the crux and it is
     **~a runner-level rewire of one stage**, reusing the Option-1 `live()` verbatim.
  3. **(R-c) ONE real interface gap:** the develop-loop's METRICS call `agent.reason_chain(...)`
     (`_longitudinal_develop_loop.py:333`), but **`MergedNavConvAgent` does NOT expose `reason_chain`** (it exposes
     `hear`/`what_does`/`who_does`/`is_it_true`/`describe`/`elaborate` — `nav_conv_merged_bridge.py:2320-2438` — the
     composer has `query_chain` but the agent never surfaces it). This is a **~5-line adapter** (a `reason_chain`
     shim delegating to `self.composer.query_chain`), or simply *don't run chain probes on the body-day* (the lived
     corridor produces `near`-facts, not multi-hop chains, so chain probes are `None` anyway). Small, but a genuine
     seam that must be closed or side-stepped.

- **Quantified residual:** **one new runner** (`_tier3_develop_with_a_body_derisk.py`, ~350-450 LOC) that (i) drives
  a NEW multi-day loop reusing the Option-1 `live()` as WAKE + the Option-1 gates + the develop-loop SLEEP/GROW/
  PERSIST/METRICS/BUNDLE stages, **plus** (ii) either a ~5-line `reason_chain` shim OR a ~15-line additive
  `per_day_agent_factory` seam in `develop_gpu` if the cleaner path (reuse `develop_gpu` directly) is chosen. **A
  `sim/` edit is NOT predicted** (Option 1 needed none; every seam here is additive/default-off or lives in the new
  runner). The "give the day-loop a body" framing slightly overstates it: the body-day is already 6/6 GO, and the
  day-loop scaffold is already GO — this is their JOIN plus the small `reason_chain`/factory seam.

**Verdict in one line:** Option 2 is **largely done (two validated GO halves)**; the residual is a **runner-only JOIN
+ one thin agent-interface adapter (`reason_chain`) or one additive `develop_gpu` factory seam**, with the honest
caveat that the two per-day BRAINS are different substrates (§2) so the JOIN is **substitution, not fusion** — the
`MergedNavConvAgent` REPLACES the `StreamCortex`+`build_agent` pair on the body-day, it does not run alongside it.

---

## 2. DIAGNOSIS — what EXISTS + the exact two-substrate seam (cite file:line)

### 2a. The two per-day brains are DIFFERENT substrates (the crux)

| | develop-loop per-day brain (today) | live-and-remember body (Option 1) |
|---|---|---|
| **object** | `StreamCortex` (`_longitudinal_develop_loop_gpu.py:145`) **+** a fresh `build_agent(...)` per day (`:407`) | one persistent `MergedNavConvAgent` (`nav_conv_merged_bridge.py:1617`) |
| **how codes arise** | LEARNED from a **text corpus** (online rate-Hebbian co-occurrence over TinyStories windows; `hear_day` `:218`, `read_codes` `:249`) → grounded phasors injected into the composer via `_inject_grounded` (`:486`) | GROUNDED from **LIVE perception**: `perceive_and_ground(obj)` reads the spiking `gen_concept` response and writes the phasor into the co-resident composer's codebook (`nav_conv_merged_bridge.py:2093`) |
| **how facts arise** | AUTHORED: `curriculum.day_stream(day)["facts"]` taught via `_teach_fact` → `agent.hear(f"{a} {v} {p}")` (`:416`, `_longitudinal_develop_loop.py:299`) | LIVED: `state.lived_facts` = `[(prev,"near",cur)]` stored via `composer.store(...)` on encounter (`_tier3_live_and_remember_derisk.py:313`) |
| **body?** | NONE (grep: 0 `navigate`/`cortex_it`/`MergedNav`) — listen-only | the nav BG-cascade + drive + perception + composer, all co-resident |
| **conversational surface** | `MultiTurnAgent`/`BrainConversationalAgent` — HAS `reason_chain` | `MergedNavConvAgent` — **lacks `reason_chain`** (R-c); has `what_does`/`who_does`/`is_it_true`/`describe`/`elaborate` |
| **persistence** | JSON `DevelopState` (facts+vocab+tier+day) via `BridgeLineage`; codes RE-LEARNED on resume by re-hearing the vocab (`:361`) | JSON `LiveState` (body + `lived_facts` + `grounded_codes`) via `BridgeLineage`; codes re-instated verbatim on resume (`_tier3_live_and_remember_derisk.py:506`) |

**The seam is therefore substitution, not composition:** the two brains cannot both be "the day's brain." A
body-day's WAKE runs the `MergedNavConvAgent`; its facts (`lived_facts`) and vocab (`encountered`) flow into the
SLEEP/GROW/PERSIST/METRICS scaffold. The text-`StreamCortex` corpus-hearing is simply *not run* on a body-day (it is
the SCRIPTED path being replaced).

### 2b. Stage-by-stage: does each `develop_gpu` stage accept a `MergedNavConvAgent`?

`develop_gpu` (`_longitudinal_develop_loop_gpu.py:287`) day body, stage by stage:

- **WAKE (`:389-402`)** — `cortex.hear_day(new_concepts)` + `read_codes()`. **DOES NOT accept the body**: it is the
  text-corpus stream. **This is the stage being REPLACED** by the Option-1 `live()` (R-b). ✗-by-design.
- **CONVERSE (`:404-423`)** — `build_agent(...)` (fresh per day) + `_inject_grounded(agent, grounded)` + re-teach
  `state.facts` + teach `day_curr["facts"]`. **DOES NOT accept the body as-is**: it *builds its own* agent and injects
  corpus codes. On a body-day, the agent IS the persistent `MergedNavConvAgent` (already built once, alive across
  days) and its codes/facts come from `live()`, not from `_inject_grounded`+authored facts. ✗ as-written; the JOIN
  substitutes here.
- **SLEEP (`:425-428`)** — `consolidate(agent, state, ...)` (`_longitudinal_develop_loop.py:345`). It calls
  `_teach_fact(agent, f)` = `agent.hear(f"{a} {v} {p}")`. **ACCEPTS the body**: `MergedNavConvAgent.hear` exists and
  matches (`nav_conv_merged_bridge.py:2320`) — but see the CAVEAT (§2c: `hear` re-parses the string through the
  merged parser, whereas lived facts were stored via `composer.store`; self-replay must re-store the SAME way). ✓
  with a store-path caveat.
- **METRICS (`_measure` `:565` → `_query_recall`/`_query_yesno`/`_query_chain`)** — `_query_recall` calls
  `agent.what_does`/`who_does` ✓ (`:2335`/`:2349`); `_query_yesno` calls `agent.is_it_true` ✓ (`:2374`);
  **`_query_chain` calls `agent.reason_chain` ✗** — **the one hard interface gap (R-c)** (`MergedNavConvAgent` has no
  `reason_chain`). Fix = a shim or skip chain probes on body-days.
- **GROWTH (`maybe_grow` `:438`, `_longitudinal_develop_loop.py:371`)** — pure-Python `TierPromoter.step(mastery)` +
  a lineage growth-event. **ACCEPTS the body** (it never touches the agent object; it reads `dp["recall_acc"]`). ✓
- **PERSIST (`_save_state` `:453`, `_longitudinal_develop_loop.py:439`)** — persists the `DevelopState` (facts+vocab+
  tier+day) as JSON via `BridgeLineage`. **ACCEPTS the body** — BUT the body-day's load-bearing lived state is the
  Option-1 `LiveState` (body energy/position/drive + `grounded_codes` + `lived_facts`), which `DevelopState` does not
  hold. The JOIN must persist BOTH (merge the two JSON payloads, or persist a superset state). ✓ with a payload-merge
  requirement.
- **BUNDLE (`per_day_save_hook` `:460`)** — `save_developed_brain(agent, ...)` extracts `agent...composer.concepts`
  (codes) + `composer.kb` (facts). **ACCEPTS the body**: `MergedNavConvAgent.composer` exists (it is the co-resident
  composer), and its `.concepts`/`.kb` are populated by grounding+storing. ✓ (the bundle is the developed brain
  the console loads).
- **`should_continue` (`:376`)** — day-boundary pause predicate, agent-agnostic. ✓ (the pausable harness works
  unchanged).

### 2c. The precise seams the JOIN must handle (honest)

1. **(R-c, HARD) `reason_chain` missing on `MergedNavConvAgent`.** `_query_chain` → `agent.reason_chain(...)` will
   `AttributeError`. Fix: a ~5-line `reason_chain(self, cue, relations)` shim delegating to
   `self.composer.query_chain(cue, relations)` (the composer HAS it — it is what `BrainConversationalAgent.reason_chain`
   calls), OR set `probe_chain=[]` on every body-day (the corridor produces only `near`-facts, so chain probes are
   vacuous anyway). RECOMMENDED: add the shim (cheap, keeps the METRICS battery identical + is a correct capability
   the agent should expose).
2. **(R-b/CONVERSE, STRUCTURAL) the per-day agent lifecycle inverts.** `develop_gpu` builds+frees a FRESH agent EVERY
   day (`build_agent` `:407`, `_free_agent` `:468`) and re-injects corpus codes. The body is ONE PERSISTENT agent
   alive across all days (it holds the drive/body/composer state). So on a body-day the loop must **NOT** call
   `build_agent`/`_free_agent`/`_inject_grounded` — it must reuse the persistent `MergedNavConvAgent`. **Critical
   `_free_agent` hazard:** `_free_agent` (`:502`) frees `composer.bridge._cp`'s memory pool; for the co-resident
   composer that bridge **IS the shared merged bridge** — freeing it mid-life would corrupt the persistent brain.
   The JOIN sidesteps this by keeping one persistent agent (no per-day free), so the hazard does not fire — but it is
   exactly why the body-day cannot reuse `develop_gpu`'s per-day CONVERSE block verbatim.
3. **(SLEEP store-path) self-replay must re-store the way the fact was stored.** Lived facts are stored via
   `composer.store(prev,"near",cur)`; `consolidate`'s `_teach_fact` re-teaches via `agent.hear("prev near cur")`
   which re-parses through the merged parser. For a body-day the replay should re-`store` (the composer path), not
   re-`hear` (the parser path), to keep the replay faithful to the lived grounding. (Alternatively, since the
   composer's store is idempotent, re-storing is a no-op — but the parser round-trip could mis-parse a bare 3-token
   `near`-fact. RECOMMENDED: replay via `composer.store`.)
4. **(PERSIST payload) merge `LiveState` ⊕ `DevelopState`.** The body-day needs BOTH the Option-1 `LiveState`
   (body/drive/`grounded_codes`/`lived_facts`) AND the develop-loop `DevelopState` (day/tier/metrics/vocab) persisted
   so resume restores the exact life AND the developmental trajectory. Easiest: the new runner owns ONE combined
   payload (the Option-1 `LiveState.memory_payload()`+`body_payload()` plus `day`/`tier`/`metrics`), and resumes by
   re-instating the grounded codes + re-storing the lived facts (exactly Option-1's `_persistence_check` path
   `_tier3_live_and_remember_derisk.py:484`).

**Everything else composes cleanly** (GROWTH, BUNDLE, `should_continue`, the pausable harness, the recall/yesno
probes).

### 2d. Can `develop_run.py`/`develop_loop_supervisor.py`/`scripts/develop.ps1` drive an Option-2 loop?

**Partly — as-is they hard-wire `develop_gpu` + `StreamCortex` + a curriculum** (`develop_run.py:121-135`,
`:164-189`). They provide the VALUABLE reusable machinery — the stable lineage root, the `PAUSE` sentinel +
`should_continue`, the per-day bundle hook, the 24/7 crash-proof/pausable supervision, `--status` — but they call
`develop_gpu(...)` directly with the text-corpus stream cortex. To drive an Option-2 loop with NO Claude in the loop,
the cleanest path is a sibling entry-point (e.g. `develop_run.py --with-body` or a new `develop_with_body_run.py`)
that builds the SAME lineage/PAUSE/bundle/`should_continue` scaffold but calls the new body-day loop instead of
`develop_gpu`. The de-risk runner itself (`_tier3_develop_with_a_body_derisk.py`) does NOT need the 24/7 harness (it
runs a bounded multi-day ladder); the harness wire-up is a FOLLOW-ON once the de-risk is GO (so the owner can leave a
develop-with-a-body run running for a simulated week, hands-off). **Do the de-risk first; wire the harness after.**

---

## 3. THE BIOLOGY (catalog-first) — what a develop-from-lived-experience loop needs

Read from `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`. A loop where **each day's knowledge is
LIVED** rests on three catalog pillars — the drive that makes experience self-chosen, the episodic encode/store of
what was lived, and the sleep-replay that makes it STICK day-over-day:

### The motivational core (why the day's experience is self-chosen, not scripted) — cluster O
- **O.05** Hypothalamic Homeostatic Architecture (`:4803`, ⭐) — sensor→integrator→effector settling-points; Kandel
  6e Ch 41. The drive that biases the foraging day. Already realized: `co_resident_drive` 2-pool spiking hunger.
- **O.06** Arcuate POMC/AgRP/MC4R feeding loop (`:4815`, ⭐) — the AgRP(hunger)/POMC(satiety) antagonism = the
  `TwoPoolDrive`/`drive_agrp`+`drive_pomc` pools. Kandel Ch 41.
- **O.10** Incentive Motivation (`:4863`) — deprivation amplifies goal-stimulus reward value (Berridge/Toates); the
  reason a hungry brain forages toward food, shaping WHICH objects it passes (and therefore what it can later talk
  about).
- **O.11** Drive Reduction Theory (`:4875`, ⭐) — a deficiency state is aversive; consuming relieves it → the intrinsic
  drive-reduction reward `r = drive_before − drive_after` (Keramati-Gutkin; Sternson CPP). Kandel Ch 41. **This is the
  reward that makes the day's foraging self-directed** — the discriminator between a LIVED day and a scripted one.
- **O.21** Average-reward, undiscounted **continuing** tasks (`:533`) — the exact RL regime for a persistent
  (non-episodic) life spanning many days: relative reward `R − R̄`, not episodic discounted return. **The single most
  relevant formal entry for a multi-day life.** Sutton & Barto §11.3; Schwartz 1993. (Honest: the current loop uses a
  simple per-day Q with `GAMMA=0.9`; the average-reward formulation is the principled long-horizon upgrade — a
  follow-on, not a blocker for the first slice.)

### Episodic encode/store of what was lived (the day's knowledge) — cluster D
- **D.01** Episodic memory — encode/store/retrieve/consolidation cycle (`:1085`). The develop-loop's core loop; on a
  body-day the "encode" is the lived `perceive_and_ground` + `composer.store(prev,"near",cur)`.
- **D.14** Engram cells — sparse activity-tagged ensembles store specific memories (Tonegawa, `:1248`). Already a
  bridge API; the analogue here is the composer's per-fact stored composite (the lived `near`-fact).
- **D.23** Misplace system — hippocampal novelty detection driving EXPLORATION (`:1059`; O'Keefe-Nadel CA1
  displace/misplace units → investigative exploration, reciprocally updating the map). **The biological engine for
  "the agent chooses what to experience"** (novelty-seeking = open-ended experience). Relevant if a later slice adds
  an explicit curiosity drive; the first slice's open-endedness is the drive-biased trajectory over an
  agent-uncontrolled layout (sufficient).

### Sleep-dependent consolidation (the day's learning STICKS across days) — cluster N
- **N.07** Hippocampal Sharp-Wave Ripples — NREM replay (`:4629`, ⭐) / **D.19** SWRs in quiet wakefulness + NREM
  (`:1309`). The mechanism the develop-loop's SLEEP stage stands in for (self-replay + retention re-test).
- **N.12** Sleep-Dependent Memory Consolidation (`:4690`, ⭐; Stickgold/Tononi) — the day-over-day no-forgetting the
  RETENTION metric measures. **The reason a multi-day lived loop needs a SLEEP stage at all.**
- **N.17 (⭐)** Awake replay during behavioral PAUSES (`:1010`; Foster & Wilson 2006) — replay fires when the agent
  RESTS / reaches a goal, giving a **LIVED (not scheduled) consolidation trigger**. This is the biology for making the
  develop-loop's scripted SLEEP phase event-triggered (a body-day naturally has rest/eat pauses); it is the seam that
  Option 4 ("lived consolidation") pursues — a follow-on, but N.17 makes it the *natural* consolidation trigger for a
  body-day (replay what you just encountered, at the eat-pause).

**Cross-cutting catalog steer:** the day-loop's WAKE→SLEEP→(GROW) alternation is the CLS (complementary learning
systems) day/night rhythm at the coarsest grain; a lived WAKE (drive-biased foraging) + a SLEEP replay of what was
encountered is precisely the biological developmental loop — the animal explores by day (O-cluster drive), consolidates
by night (N-cluster replay), and its knowledge is a consequence of what it lived (D.01/D.23), not an author's syllabus.

---

## 4. RANKED cheap-first JOIN options

Each: mechanism · reusable machinery · behavioral signature · anti-cheat. The cheapest genuinely-lived + persistent +
MULTI-DAY slice, NOT the whole capstone.

### ★ OPTION 2A (RECOMMENDED) — a NEW body-day loop reusing the Option-1 `live()` as WAKE + the develop-loop SLEEP/GROW/PERSIST/METRICS/BUNDLE stages, in ONE new runner (no `develop_gpu` edit)

- **Mechanism:** a new `_tier3_develop_with_a_body_derisk.py` runs `develop_body(N_days)`: build ONE persistent
  `MergedNavConvAgent` (co-resident drive + perception + generalization + composer) once. For each simulated day:
  **WAKE** = run the Option-1 `live()` for that day's foraging stretch (the drive-biased survival loop that
  perceives+grounds+stores `near`-facts) → the day's NEW facts = the `lived_facts` produced THIS day, the day's
  vocab-growth = the objects `encountered` THIS day; **SLEEP** = re-`store` a sample of prior lived facts (the
  self-replay CLS proxy, per N.07/N.12) + the OLD-fact retention re-test; **METRICS** = `what_does`/`who_does`/
  `is_it_true` on the day's lived facts + the no-confab moat on never-encountered objects (chain probes skipped —
  corridor produces only `near`-facts); **GROWTH** = `TierPromoter` on the day's recall mastery; **PERSIST** = the
  combined `LiveState`⊕`DevelopState` JSON via `BridgeLineage`; **BUNDLE** = `save_developed_brain(agent)` per day.
- **Reusable machinery:** the whole Option-1 runner (`_tier3_live_and_remember_derisk.py`: `LiveState`, `live()`,
  `LivingWorld`, `SpikingHunger`, `_lived_recall`, `_moat_check`, `_persistence_check`, the grounding-corruption
  anti-cheat) + `develop_gpu`'s `TierPromoter`/`maybe_grow`/`consolidate`/`_measure`/`_save_state` PATTERNS (reused by
  import or re-implemented thinly) + `BridgeLineage` + `save_developed_brain`. **NO `sim/` edit; NO `develop_gpu`
  edit** (the new runner owns its own day loop, so it needs no factory seam in `develop_gpu`).
- **Behavioral signature:** over a simulated WEEK, the SAME persistent brain forages each day, and its vocab/facts
  GROW from what it PERCEIVED (day-N `encountered`/`lived_facts` ⊃ day-0), it recalls what it lived each day,
  RETAINS prior days' lived facts (no catastrophic forgetting), abstains on never-encountered objects (moat 0-FA
  daily), and RESUMES the exact life+memory after a reset — with loadable per-day bundles.
- **Anti-cheat:** Option-1's full set per day (drive-lesion/yoke STARVE; grounding-corruption collapses lived recall;
  no-persistence cold-start empties memory; reward-provenance no-distance-term; **no-confab moat byte-frozen in
  vivo**) + the develop-loop's (frozen-brain plasticity-off accumulates NO new facts; retention/no-replay arm; the
  LIVED-not-scripted assertion — the day's facts are drawn from `live()`'s `lived_facts`, NEVER an authored list).
- **Why cheapest + genuinely lived+persistent+multi-day:** it JOINS two GO halves with ZERO new mechanism class and
  ZERO `sim/`/`develop_gpu` edit; the only new code is the day-loop orchestration + the small `reason_chain` decision
  (here: skip chain probes). It sidesteps the two-substrate seam entirely by **not running the text `StreamCortex` at
  all** on a body-day — the body IS the day's brain.
- **Cost/risk:** LOW-MEDIUM. The one risk is wall-clock (each day builds no bridge if the persistent brain is reused,
  but each day runs `live()` which steps the merged bridge for groundings; see §6). The `reason_chain`/payload-merge
  seams are ~small.

### OPTION 2B — add an additive `per_day_agent_factory` seam to `develop_gpu`, then reuse `develop_gpu` verbatim with the body as the per-day agent

- **Mechanism:** add a default-`None` `per_day_agent_factory` (and a `per_day_wake_fn`) param to `develop_gpu`
  (byte-identical when unset, exactly like `per_day_save_hook`/`should_continue`); when set, the loop calls
  `per_day_wake_fn(day, state)` for WAKE (returns the day's facts/vocab from `live()`) and `per_day_agent_factory()`
  to get/hold the persistent `MergedNavConvAgent` instead of `build_agent`/`_free_agent`. Then the de-risk reuses
  `develop_gpu`'s validated SLEEP/GROW/METRICS/PERSIST/BUNDLE verbatim.
- **Reusable machinery:** ALL of `develop_gpu` (maximal reuse of the validated day-loop + the 24/7 harness) + Option-1
  `live()` as the wake fn. **NO `sim/` edit; ONE additive default-off seam in `develop_gpu`** (a research-runner file,
  not `sim/`).
- **Behavioral signature / anti-cheat:** identical to 2A.
- **Why maybe better:** maximal reuse of the *validated* day-loop (less new orchestration to get wrong; the harness
  drives it directly). **Why NOT the first pick:** it edits `develop_gpu`'s hot path (a validated file) to add the
  factory + wake seam + must guard the `_free_agent` hazard (§2c-2) + the `_inject_grounded` bypass + the
  `reason_chain` gap — more surface area to regress than a self-contained new runner. **Do 2A first** (self-contained,
  can't regress the existing develop-loop); promote to 2B (the factory seam) only once the body-day loop is GO and the
  owner wants the 24/7 harness to drive it directly.

### OPTION 2C (thinnest possible SMOKE, not the slice) — a 2-day body-day mechanics check, no growth/no bundles

- **Mechanism:** the smallest thing that proves the JOIN closes: run `live()` for day 0 → persist → reload → run
  `live()` for day 1 on the resumed brain → assert day-1 knows day-0's lived facts + day-1's new ones, moat holds,
  retention holds. No `TierPromoter`, no bundles.
- **Use:** the 1-seed smoke rung of the ladder (§5), NOT the deliverable.

### NOT-first-slice (deferred):
- The **learned spatial policy** from intrinsic reward = the Tier-4 dendrite wall (survival uses the validated
  rate-proxy Q stand-in). Off the critical path (Option 1 established survival is load-bearing with a simple policy).
- **N.17 event-triggered lived consolidation** (SWR replay fired at the eat-pause, not a scripted SLEEP phase) =
  Option 4 — a follow-on after 2A is GO.
- **True `cp_connections` synaptic persistence** — the combined-JSON resume suffices; raw-tensor persistence is a
  follow-on.
- **O.21 average-reward continuing-task RL** — the principled multi-day reward formulation; a follow-on upgrade to
  the per-day Q (the first slice keeps the validated Option-1 Q).

---

## 5. THE SINGLE RECOMMENDED CHEAP-FIRST DE-RISK

**Build OPTION 2A as `research/runners/_tier3_develop_with_a_body_derisk.py`** — a new runner, self-contained, reusing
the Option-1 `live()` as WAKE + the develop-loop SLEEP/GROW/PERSIST/METRICS/BUNDLE patterns, on ONE persistent
`MergedNavConvAgent`. **NO `sim/` edit predicted; NO `develop_gpu` edit** (the first slice owns its own day loop; the
`develop_gpu` factory seam, Option 2B, is a follow-on only if the 24/7 harness must drive it).

### Runner design (concrete)
- **Build once:** `agent = MergedNavConvAgent(seed, vocab=OBJECT_WORDS+ACTIONS, co_resident_composer=True,
  co_resident_composer_kind="rf", co_resident_perception=True, co_resident_generalization=True,
  perception_grounding="gen_spikes", co_resident_drive=True)` — the Option-1 `_build_agent` verbatim. `bridge =
  agent._merged_bridge`; `hunger = SpikingHunger(bridge)`.
- **Add the `reason_chain` decision:** set every body-day's `probe_chain=[]` (corridor → only `near`-facts), OR add a
  ~5-line `reason_chain` shim to the agent in the runner via composition (a small wrapper), so the METRICS battery is
  identical. RECOMMENDED: skip chain probes (vacuous on the corridor) for the first slice; note it.
- **`develop_body(agent, hunger, n_days, days_of_life, world_schedule)`:** for each day `d`:
  1. **WAKE** = `live(agent, hunger, state, world, n_steps, drive_reward="rate_proxy", perceive=True, grounded_obj_cache=cache)`
     for that day's stretch. The day's NEW facts = `state.lived_facts` added THIS day; the day's vocab = `state.encountered`
     added THIS day. (To keep the life OPEN-ENDED across days, either enlarge/rotate the world's object placement per
     day so new objects become reachable — the multi-day open-endedness upgrade — or keep one corridor and let the
     day-0 encounters be the lived facts and later days re-consolidate + add via a richer world; the first slice can
     use a per-day-rotated small world so `encountered` grows day-over-day.)
  2. **SLEEP** = re-`store` a sample of prior lived facts (`composer.store`, NOT `agent.hear`) + retention re-test on
     old lived facts (`is_it_true` on a prior `near`-fact).
  3. **METRICS** = recall on the day's lived facts (`what_does`/`who_does`) + retention + moat (`_moat_check` on
     never-encountered/held-out objects). Record `dp` (vocab/facts/recall/retention/moat_fa/tier).
  4. **GROWTH** = `TierPromoter.step(recall_acc)` + lineage growth-event.
  5. **PERSIST** = combined JSON (`LiveState` body+memory ⊕ day/tier/metrics) via `BridgeLineage`.
  6. **BUNDLE** = `save_developed_brain(agent, day_dir, ...)`.
- **Pause seam:** poll `should_continue()` at the top of each day (a `PAUSE` sentinel), exactly `develop_gpu`'s
  pattern, so the harness wire-up (2B/follow-on) is trivial later.

### Ladder
1. **1-seed GPU smoke (Option 2C mechanics):** the multi-day JOIN closes — the persistent brain forages day 0 and day
   1, day-1 knows day-0's lived facts + day-1's new ones, retention holds, moat 0-FA, the combined state persists and
   a reload resumes the exact life+memory. (No growth/bundles asserted yet.)
2. **Multi-day 1-seed (3-5 days):** vocab/facts GROW day-over-day from what was LIVED; a tier fires; per-day bundles
   save + are loadable; frozen-brain arm accumulates no new facts.
3. **6-seed** (42/43/44/100/101/102) for the develop-over-a-week + all-anti-cheats-collapse claims (the standing
   6-seed rule for any generalization/robustness claim).

### Decisive GO / BOUNDARY / NEGATIVE checks
1. **LIVED-not-scripted (the R-b discriminator, GO):** each day's stored facts are drawn from `live()`'s
   `lived_facts` (assert: no authored fact list is ever consulted on a body-day; the day's `new_concepts` = the
   objects the body `encountered` THIS day). A **permuted-world** control (shuffle object placement) → a DIFFERENT
   lived-fact set → the memory tracks the world the body actually traversed, not a fixed script.
2. **DEVELOPS-over-days (GO):** `vocab[-1] > vocab[0]` AND `facts[-1] > facts[0]` from lived experience; day-N brain
   measurably differs from day-0. BOUNDARY if the corridor saturates (only N objects) — mitigated by per-day-rotated
   world placement so new objects become reachable.
3. **NO-FORGETTING / retention across days (GO):** OLD (prior-day) lived facts stay recallable as new days accumulate
   (`is_it_true` on a day-0 fact still "yes" on day N); the **no-replay (consolidation-off) arm** degrades retention
   (the CLS contrast — honest caveat: the composer store is idempotent, so on the rate/symbol path retention is
   naturally high; the load-bearing interference contrast is the spiking-store follow-on, exactly as `develop_gpu`
   documents).
4. **FROZEN-BRAIN competence-must-not-rise (anti-cheat, GO):** with plasticity gated off / facts not committed, the
   brain forages but accumulates NO new facts → competence flat. (For the body, "plasticity off" = do not
   `composer.store` the lived facts.)
5. **NO-CONFAB MOAT byte-frozen (HARD):** every never-encountered / held-out query returns `None`; assert the
   conversational synapses (`cp_connections.data` for the parser slice) BYTE-IDENTICAL across the whole multi-day run.
   A breach is a HARD STOP.
6. **PERSISTENCE across reset (GO):** reload resumes the EXACT combined state (body + lived facts + grounded codes +
   day/tier); a no-persistence cold-start visibly differs (empty memory, day 0).
7. **REWARD-PROVENANCE (GO):** `r` = drive-reduction (from the spiking/interoceptive drive), asserted NO
   `r=f(distance)` — inherited verbatim from Option 1.

### Predicted `sim/` edit
**NONE.** Option 1 (the harder half) needed none; every Option-2A seam is additive/default-off or lives in the new
runner. The single interface gap (`reason_chain`) is handled in the runner (skip chain probes) or by a runner-level
wrapper — no agent/`sim/` edit required. If Option 2B (the `develop_gpu` factory seam) is later pursued for the 24/7
harness, that edit is in a **research-runner file** (`_longitudinal_develop_loop_gpu.py`), NOT `sim/`, and is additive
default-off (byte-identical to every current caller).

---

## 6. HONEST SCOPE / expected boundaries

- **The learned spatial policy stays the deferred Tier-4 dendrite wall.** Survival uses the validated Option-1
  rate-proxy Q stand-in; survival (not spatial optimality) is the discriminator. If the fixed BG-cascade's inability
  to *learn* an efficient forage policy makes multi-day survival underperform at some world-size, that maps the
  substrate — a precisely-localized NEGATIVE (develop-over-days = GO, spatial-optimality = dendrite) IS a valid
  deliverable, and per Option 1 it should NOT block the slice.
- **Persistence is JSON re-instate** (combined `LiveState`⊕`DevelopState`: body + lived facts + grounded codes +
  day/tier), NOT the raw `cp_connections` synaptic tensor. On resume, the grounded codes are re-instated + the lived
  facts re-stored (Option-1's exact path). True synaptic-tensor persistence of the merged bridge is a follow-on
  (the develop-loop / Option-1 cheap-first stand-in).
- **Open-endedness is encounter-driven.** Which objects the brain knows is a consequence of its drive-biased
  trajectory over a world it does not fully control (the R-b discriminator vs a scripted route/layout). For
  MULTI-DAY vocab GROWTH the first slice rotates/enlarges the small world's object placement per day so new objects
  become reachable; the richer 2D path-dependent-order world (where the order of encounters itself matters) is a
  follow-on.
- **The CLS retention contrast is a proxy at the rate/symbol level.** As `develop_gpu` already documents, the
  composer's fact-store is idempotent (re-storing is a no-op), so retention is naturally high; the load-bearing
  interference contrast (where new learning genuinely overwrites old) is the fully-spiking-store follow-on.
- **Consolidation is a scripted SLEEP phase, not event-triggered.** The N.17 awake-replay-at-pauses (lived
  consolidation) is Option 4 — a follow-on.
- **Wall-clock per simulated week.** Option 1 was **~8 min/seed** for ONE life (2 bridge builds/seed + `live()`
  stepping the merged bridge only for groundings, `rate_proxy` survival = pure host between groundings). A body-day
  loop over a WEEK reuses ONE persistent brain (build once, ~1 build/seed), so the per-week cost ≈ (one build) + (7
  days × the per-day `live()` stretch stepping the bridge for groundings + a short SLEEP/METRICS). Expect **roughly
  Option-1's per-life cost scaled by the number of days' foraging stretches** — order tens of minutes per seed for a
  short week on one 3090; a full simulated year is a hands-off overnight run via the harness (2B/follow-on). Confirm
  the actual per-day wall-clock in the 1-seed smoke (the develop-loop already reports `mean_day_seconds` +
  `compressed_week_eta_minutes`). This is a LOCAL run (no VRAM wall — one merged bridge fits 24GB comfortably); GPU
  required (`SIM_BACKEND=cupy`).

---

## 7. VERDICT for the owner

**Yes — Option 2 (develop-with-a-body) is the right next slice, and it is cheap.** It is **largely done in two
validated GO halves** (Option-1's live-perceive-ground-store loop, 6/6 GO, + the develop-loop's SLEEP/GROW/PERSIST/
METRICS/BUNDLE scaffold + 24/7 harness, GO). The genuine residual is a **runner-only JOIN** — a new
`_tier3_develop_with_a_body_derisk.py` that uses the Option-1 `live()` as the day's WAKE (so each day's knowledge is
LIVED, not a scripted curriculum) feeding the develop-loop stages — **plus one small honest seam**: the two per-day
brains are DIFFERENT substrates, so the JOIN is **substitution** (the `MergedNavConvAgent` REPLACES the text
`StreamCortex`+`build_agent` on a body-day; the text corpus is simply not run), and the agent lacks `reason_chain`
(handled by skipping chain probes on the corridor, or a ~5-line shim). **No `sim/` edit is predicted** (Option 1
needed none; every seam is additive/default-off or lives in the new runner).

**Recommended:** build Option **2A** (self-contained new runner, can't regress the existing develop-loop) on the
1-seed-smoke → multi-day → 6-seed ladder, with the seven decisive checks (LIVED-not-scripted · develops-over-days ·
no-forgetting · frozen-brain competence-flat · **no-confab moat byte-frozen** · persistence-across-reset ·
reward-provenance). Promote to Option **2B** (an additive default-off `per_day_agent_factory`/`per_day_wake_fn` seam
in `develop_gpu` — a research-runner file, still NO `sim/` edit) only as a FOLLOW-ON, once 2A is GO and the owner
wants the 24/7 `develop_run`/supervisor harness to drive a develop-with-a-body run hands-off for a simulated week.

**Stays deferred (off the critical path):** the learned spatial policy (Tier-4 dendrite wall), true `cp_connections`
synaptic persistence, N.17 event-triggered lived consolidation (Option 4), O.21 average-reward continuing-task RL, and
the richer 2D path-dependent world. This slice converts "a brain that develops over a scripted week" into **a brain
that develops over a week it LIVED** — the next honest step toward the persistent living agent, on the merged one
brain, moat intact.
