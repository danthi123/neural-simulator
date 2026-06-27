# Comprehensive shortcut inventory + master burndown plan — project-WIDE (2026-06-27)

**Type:** READ-ONLY deep-research audit (no code written, no `sim/`/composer edit). The definitive project-wide
inventory of every residual SHORTCUT + the ranked master close-out plan, per the CYCLE-667 owner directive: close
out EVERY remaining shortcut/cheat/non-biological thing that does NOT run DEFAULT-ON on the one-brain shared spiking
substrate, EXCEPT (a) accepted-deep limitations with months-long resolutions and (b) environmental/host (the legit
host = environment + body).

**Extends** the two prior inventories project-wide and brings them current:
- `2026-06-20-shortcut-burndown-inventory.md` (the prior project-wide audit; 12 shortcuts) — **now partly stale**:
  three nav LIBRARY defaults flipped to spiking on 2026-06-24, and the conversational cleanup/scan were converted
  on 2026-06-27 (C1–C4). This doc reconciles to current code.
- `2026-06-27-conversation-depth-brain-based-audit-and-burndown.md` + the C1–C4 / B-mine-1/2 / B-wire-1 GO docs
  (the conversational operations + the cheap structure-mining, now CLOSED as conversions/acquisitions).

**The bar (precise).** A residual SHORTCUT = anything COGNITIVE (between sensation and action) that is NOT, by
DEFAULT, running as spikes/synapses on the one-brain shared substrate. Includes: host-computed cognition
(argmaxes, formulas, host loops/selectors), numpy-reference paths that are the DEFAULT (vs an existing spiking
form), host-DESIGNED structure not learned/self-organized, and capabilities validated-OPT-IN but not default-on.
Host code is LEGITIMATE only for the ENVIRONMENT (world state + sensory rendering) and the BODY (acting on the
spiking motor/answer selection). (Framing: CLAUDE.md "Standing standard: BRAIN-BASED ONLY";
`feedback_brain_based_only_standard`, `feedback_end_state_fully_spiking_one_brain_path_by_efficiency`.)

**Terms (defined once).**
- *LIBRARY default* — the default value in a Python function/constructor signature (what runs when a caller passes
  nothing). *CLI/deployed default* — the default of the command-line flag the shipped runner exposes (what a user
  reproduces). The two often DIFFER and the gap is load-bearing here (the crux is what actually SHIPS).
- *numpy-reference / oracle path* — a CPU computation that reproduces, bit-for-bit, what a validated spiking circuit
  computes; kept on purpose so CI can assert the spiking path matches a ground truth AND so the agent runs on a
  GPU-less machine (`SIM_BACKEND=numpy`). Per standing directive (`feedback_close_arcs_to_full_capacity`) this is
  KEPT as the oracle even after the spiking default ships — so "the numpy path exists" is NOT itself the shortcut;
  "a USER SURFACE runs the numpy path by default" IS.
- *self-organized structure* — connectivity/lexicon the brain learned (Hebbian/STDP, corpus-mined), vs
  *host-designed structure* — a dict/matrix a human typed or a host formula computes.

---

## TOP-LINE COUNT (the honest call, reconciled to current code)

**The project is far closer to default-on-fully-spiking than the 2026-06-20 count suggested.** The conversational
who/what core's OPERATIONS are now CLOSED (C1–C4 converted them to the spiking onebrain substrate; the two biggest
hand-authored relational tables are corpus-LEARNED via B-mine-1/2). The navigation LIBRARY defaults flipped to
spiking on 2026-06-24. **What remains is dominated by two things: (1) cheap DEPLOYMENT flips — the spiking form is
validated and default at the LIBRARY level but a USER SURFACE / CLI still defaults to the host/numpy path; and (2)
the navigation reward/value/perception closed loop, which is genuine open research (the loop does not sustain).**

| Class | count | meaning |
|---|---|---|
| **(i) CLOSEABLE-CHEAP** (a spiking/learned form exists + is validated; the work is a default-flip / wiring / a small parity de-risk) | **7** | deployment flips + one cleanup-WTA scale follow-on + a borderline projection ruling |
| **(ii) RESEARCH-GATE-NEEDED** (a wall — the deep-research loop fires BEFORE building) | **4** | the navigation reward / value / place / SC-orienting closed loop (the documented NO-GO) |
| **(iii) ACCEPTED-DEEP** (EXCLUDE — owner-accepted, months-long) | **3** | learned recursive generative grammar; developmental dendritic binding self-org; B2 fluid analogy |
| **(iv) LEGIT-HOST** (EXCLUDE — environment + body) | **5** | world state, sensory render, motor/answer emission, scoring, bookkeeping |

So: **7 cheap closeables (mostly deployment flips), 4 research-gate items (all in the nav reward/value/place loop),
3 accepted-deep, plus the legit-host exclusions.** The conversational core is essentially done at the LIBRARY level
and one deployment flip from being default-on at the user surface; the navigation cascade is where the genuine open
research lives.

**The single most-impactful finding (the headline shortcut to close FIRST):** the flagship **first-chat CONSOLE
still defaults to `--composer rf`** (`first_chat_console.py:2138`, `default="rf"`) — i.e. the user-facing chat the
owner talks to runs the numpy REFERENCE composer by default, even though the spiking onebrain path (C3 flat + C4
typed) is built, validated answer-identical, and merely opt-in (`--composer onebrain`). Flipping that default
(keeping `rf` as the documented oracle / CPU fallback) is the single cheapest highest-value close-out: it makes the
flagship surface fully-spiking-one-brain.

---

## SECTION A — CONVERSATION (the who/what core + the depth tiers)

### A1. What is ALREADY closed (verified — do NOT re-open; listed for auditability)

| Item | Where | Why it is NOT a residual shortcut now |
|---|---|---|
| Role-filler **bind/bundle/unbind** | RF resonate-and-fire + complex synapses; `OneBrainComposer` | Spiking on-substrate (the ops; the *exact-inverse algebra* is the deep frontier #DEEP-1, separate). |
| **Cleanup SELECTION** (nearest concept) | `one_brain_composer.py:670-672` `_select` | **`enable_spiking_cleanup=True` by DEFAULT** in `OneBrainComposer` (host argmax only as a zero-spike fallback). Spiking Izhikevich-WTA. Closed by burndown-1 (2026-06-20) + the C3 wiring. |
| **Cue-MATCH / answer routing** | `one_brain_composer.py:1052-1055` `_scan` (+ the K-way sequencer) | Converted on the onebrain console path (C3): the whole flat who/what recall/answer runs on the persistent spiking bridge. (The host `for/if` `_scan` survives only on the numpy ORACLE path, where it is the legitimate reference.) |
| **Serial-order** (word order) | `neural_serial_order_renderer.py`; wired by C1+C2 | Spiking competitive-queuing renderer; `ArgStructureComposer.render` uses it (C1), the console flips `enable_neural_render=True` on the GPU path (C2). |
| **Typed verb-frame** (GOAL/THEME/…) | `one_brain_composer.py` typed-role API | Runs on the spiking onebrain substrate at D≥128 (C4, GO 6/6, moat 0-FA). |
| **No-confab moat** | spiking Bogacz-Brown familiarity gate + the abstain | Validated spiking; preserved 0-FA across every conversion. |
| **Concept CODES** (word meanings) | `bridges/firstchat/brain*.npz` | LEARNED from conversation (rate-Hebbian co-occurrence on the real bridge, `2026-06-15-...-GO.md`). Self-organized, not host-designed. |
| **Verb-frame LEXICON** + **wh→role map** | B-mine-1, B-mine-2 | Corpus-MINED over the brain's own vocab (permuted-mining collapses → the corpus carries it). Structure ACQUIRED, available as a validated drop-in. |
| **Ordinal relation axis** (size) | B1 / B-wire-1 | Corpus-MINED + wired into the console; the comparator is a spiking Wang-2002 accumulator. |

### A2. The residual CONVERSATION shortcuts (CLOSEABLE-CHEAP)

| # | shortcut (file:line) | default-on-spiking-on-one-brain? | the cheap close + one-line de-risk + anti-cheat |
|---|---|---|---|
| **C-1** | **first-chat CONSOLE composer default** = `--composer rf` (`first_chat_console.py:2138`, `default="rf"`) → the flagship user surface runs the numpy `RFPhasorComposer` reference, not the spiking onebrain | **NO** (the spiking C3/C4 path is opt-in `--composer onebrain`) | **Flip the GPU default to `onebrain`** (keep `rf` as the explicit oracle / `SIM_BACKEND=numpy` CPU path), exactly the `consolidated_320` pattern. De-risk: answer-parity on the 1454 codes (already shown 24/24 in C3) + rubric 10/10 + moat 0-FA. Anti-cheat: the numpy oracle path must stay byte-identical; moat 0-leak. **THE RECOMMENDED FIRST CLOSE-OUT.** |
| **C-2** | **mined verb-frame lexicon / wh-map NOT the composer default** (`argstructure_composer.py` `frame_lexicon=None` → hand `FRAME_LEXICON`; `wh_question_parser.py` `frame_roles=None` → hand map) | **NO** (B-mine-1/2 validated the mined structure but it is opt-in; the hand dict is the default) | **Deploy the mined frames as the default** where the brain's vocab covers them (rubric-gated; keep the hand dict as the fallback for vocab-poor brains / `send`/`put`/`when` differ). De-risk: rubric 10/10 unchanged + the typed matrix == hand on the covered verbs. Anti-cheat: permuted-mining must still collapse (already 0.033); moat 0-FA. *(This is the CYCLE-667 cheap-completion task already in flight — coordinate, don't collide.)* |
| **C-3** | **`rf` composer cleanup SELECTION** = host `np.argmax(sims)` (`rf_phasor_composer.py:386`, `enable_spiking_cleanup=False` default) | **NO** on the rf path — but this IS the numpy ORACLE | **ACCEPT as the oracle** (it is the CPU/test-reference; the spiking selection ships via the onebrain default once C-1 lands). No separate burn needed beyond C-1; the rf-numpy argmax is the legitimate reference. *(Counted as effectively-closed-by-C-1; listed so the boundary is explicit.)* |
| **C-4** | **cleanup-WTA at crowded D** — the spiking Izhikevich-WTA cleanup costs 1 safe-direction abstain at V=1454/D=128 (the C3 localization), so the console onebrain uses `enable_spiking_cleanup=False` to match the oracle | **PARTIAL** (the STORE/scan are spiking; only the final winner-PICK falls back to host-argmax to match the oracle at crowded D) | **Wider-D / shard the brain** so the spiking-WTA cleanup == argmax at the console scale (it already does at D=2048). De-risk: a wider-D brain → spiking-WTA recall == host argmax, moat 0-FA. **Needs a wider-D brain (a small build), not research.** Low priority — the substrate STORE is already spiking; this is the last winner-pick at scale. |
| **C-5** | **PPMI read-out NORMALIZATION** (offline code generation): host `double_center(L)` is the default (`_phaseB_onbridge_stream_conversation_derisk.py:120`, `--readout-norm default="host"`) → produces the cached codes the agent loads | **NO** (the `neural` path exists + is de-risked at 96% of host, but only seed-42 codes cached) | **Produce the 320-scale `--readout-norm neural` codes for seeds 43/44 and default the demo to them.** Offline learning-pipeline conversion (the circuit = per-hub spike-frequency adaptation + per-concept feedforward inhibition, validated). De-risk: the neural-norm codes pass the who/what + moat pipeline at parity. Anti-cheat: moat 0-FA on the neural codes. NOT in the live conversational turn. |
| **C-6** | **the grounding PROJECTION** `np.angle(proj @ code_vec)` (`consolidated_320_conversation_demo.py:78`) — host matmul mapping a learned cortex rate-code into the composer phasor code, run ONCE per concept at setup | **BORDERLINE** (a FIXED projection, run once — arguably a legit fixed cortico-cortical fan-in, not per-fact cognition) | **OWNER-RULE / cheap conversion.** Either (a) ACCEPT as a legit fixed synaptic fan-in (the step-3 grounding already routes a live `cortex_it` rate vector through a fixed complex projection on the bridge), or (b) realize it as a fixed complex bridge synapse. Low stakes either way; flagged for the owner to rule. |

**Conversation residual count: 4 genuine cheap closeables (C-1, C-2, C-4, C-5) + 2 boundary/oracle items (C-3
accepted-as-oracle, C-6 borderline-ruling).** C-1 is the keystone (the flagship surface flip); C-2 is already in
flight; C-4/C-5 are small follow-ons; C-3/C-6 are accept-or-rule.

---

## SECTION B — NAVIGATION + PERCEPTION + SHARED CORE

**The critical reconciliation (verified in current code this session).** The 2026-06-20 audit's nav rows are now
**split**: the **LIBRARY** `run_moving_goal_episode` defaults flipped to spiking on 2026-06-24 (`spiking_snc=True`
`:3480`, `enable_neural_critic=True` `:3506`, `readout_source="spiking_wta"` `:4124`, `perceived_approach_reward=True`
`:3346`), so the **merged "one brain" agent that the consolidated path builds runs the value/decision/reward on
spikes by default**. BUT the **shipped standalone CLI** (`g11_bg_runner.py` argparse + the merged-gate runner
`_nav_gate_merged_run.py:37`) still defaults these flags to HOST (`--readout-source` default `"motor"`;
`--spiking-snc`/`--enable-neural-critic`/`--perceived-approach-reward` are `store_true` = off), so every documented
standalone benchmark reproduces with host cognition. **The shortcut is therefore the CLI/deployed default, not the
library mechanism** — and for the decision/value the spiking path is validated, so those are CHEAP CLI flips; only
the reward/place/SC-orienting CLOSED LOOP is genuine open research.

### B1. The residual NAV shortcuts

| # | shortcut (file:line) | cognitive op | default-on-spiking-on-one-brain? | class + the close / gate |
|---|---|---|---|---|
| **N-1** | **ACTION DECISION** — LIBRARY `readout_source="spiking_wta"` (`g11_bg_runner.py:4124`) BUT the SHIPPED gate/CLI default `"motor"` (`_nav_gate_merged_run.py:37`; `g11_bg_runner.py` argparse) → host argmax over motor spike counts in standalone runs | pick the move | **SPLIT** — LIBRARY=spiking (merged agent), CLI=host | **CLOSEABLE-CHEAP.** The spiking commit-burst readout is validated default-on at 1.16× host (`2026-06-19-spiking-decision-default-on-GO.md`). **Flip the gate/demo CLI default to `spiking_wta` + `--urgency-max-pa 180`** (keep `"motor"` as an explicit `--readout-source motor` oracle to reproduce historical benchmarks). De-risk: the gate score == the library run; anti-cheat: the conversational moat unaffected (array-disjoint). |
| **N-2** | **VALUE / RPE baseline** — LIBRARY `spiking_snc=True` + `enable_neural_critic=True` (`:3480/:3506`); CLI `store_true` (off) → host `reward_ema` + `_V_scaffold = max(0, reward_ema_pre)` in standalone runs | the value V the dopamine RPE subtracts | **SPLIT** — LIBRARY=spiking critic, CLI=host scaffold | **RESEARCH-GATE (boundary).** The spiking striosome-D1 critic (GABA_B/GIRK → SNc) learns V on-substrate, but the merged value-train δ is graded-but-WEAK (~1.3× vs the 4–19× ceiling, capped by the position-blind up-state floor — `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md`). The LIBRARY default-on is the optimistic merged config; the *honest* status is a documented boundary → it travels with the reward/place loop (N-3/N-4) through the deep-research gate, NOT a clean CLI flip yet. |
| **N-3** | **REWARD computation** — host distance/eccentricity formula → `current_reward_signal = delivered_reward` (`g11_bg_runner.py:~7616-7661`); the synaptic SC-proximity approach-reward (`enable_spiking_sc_approach`, `:882`) is default-off | compute the scalar reward r | **NO** (host-default; the synaptic approach-reward is opt-in) | **RESEARCH-GATE.** The `sc_rostral → reward_us` synaptic approach-reward is QUALIFIED-GO in isolation (`2026-06-18-merged-neural-reward-QUALIFIED-GO.md`) but is part of the same NO-GO closed loop as N-4. Reward FROM the environment is legit-host (the world delivers it); reward COMPUTED by a host distance formula (the brain should appraise proximity) is the shortcut. |
| **N-4** | **SC-ORIENTING / salience heuristic** — host Manhattan compare `if gx > x: cortex_E += HEURISTIC_DRIVE_PA`, gated by `heuristic_strength=1.0` default (`g11_bg_runner.py:~4003`); `enable_spiking_sc` default off (`:876`) | decide which cardinal points at the goal + bias the action cortex | **NO** (host heuristic ON by default; spiking SC opt-in) | **RESEARCH-GATE (the documented NO-GO).** The spiking superior colliculus is N1-validated standalone, but the DEPLOYED closed loop is NO-GO (~58× worse, the actor goes silent — `2026-06-19-nav-spiking-sc-deploy-NO-GO.md`). The single most load-bearing nav cognitive shortcut. Needs the deep-research gate as a UNIT with N-2/N-3/N-5. |
| **N-5** | **PERCEPTION → place/goal state code** — host Gaussian over true (x,y)/(gx,gy): `place_drive = ...np.exp(-place_dsq/...)` (`g11_bg_runner.py:~6563`); `neural_place_selforg` default off (`:521`) | turn position + goal coords into a place/goal code | **NO** (host Gaussian default; self-org place opt-in) | **RESEARCH-GATE (dendritic flavor).** The self-organized spiking place code is NOT location-selective in the self-org read regime (a few cells fire everywhere) and over-clamps the SNc (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`). Reading the agent's grid POSITION is borderline-host (the body knows where it is); the hand-coded Gaussian PLACE FIELD (the brain should develop its own receptive fields) is the shortcut. |

**Nav residual count: 1 CLOSEABLE-CHEAP (N-1 decision CLI flip) + 4 RESEARCH-GATE (N-2/N-3/N-4/N-5 = the reward /
value / SC-orienting / place closed loop).** Per BRAIN-BASED-ONLY, an honest NEGATIVE here (the neural loop
underperforming the host shortcut) IS the scientific deliverable — these map what the point-neuron substrate can/can't
do for sustained reward-driven control. The four are one mechanism (the closed loop); they go through the gate
together.

---

## SECTION C — ACCEPTED-DEEP (EXCLUDE — owner-accepted, months-long; noted, not ranked)

| # | item | where | why deferred (the precise residual) |
|---|---|---|---|
| **DEEP-1** | **The exact-inverse FHRR bind ALGEBRA + the developmental self-organization of the binding connectivity** | `rf_phasor_composer.py:156-185`; `rf_set_complex_weights` injects the clean-invertible bind weights | The bind OPERATIONS are spiking; the residual idealization is the EXACT-INVERSE algebra (a clean binding a real cortex would LEARN, lossy + redundant) + the host-INJECTED bind connectivity (not self-organized). Both dendrite jobs tested cheap-first + ruled out (memorizes-not-generalizes `2026-06-19-dendritic-binding-toy-derisk.md`; apical-basal credit NEGATIVE). The path is more codes/capacity or the dendritic substrate — MONTHS, high variance, the artificial-life goal. |
| **DEEP-2** | **Learned RECURSIVE generative grammar** | `argstructure_composer.py` `FRAME_LEXICON` etc. — the *labeling* is now mineable (B-mine-1), but a grammar that GENERATES novel language is the frontier | The step from induced item-frames to a learned recursive hierarchical grammar (phrase structure / Merge / long-distance dependencies / productive novel composition) — the categorical free-generation gap (MEASURED 0.0 novel-composition). The only closer is the BPTT-SNN generative LM as a development-stand-in, consolidated onto one bridge with no-forgetting (gates met at toy scale; SCALE is the open work). `project_generative_sequence_frontier`. |
| **DEEP-3** | **B2 fluid analogy from raw learned codes** (king−man≈queen−woman) | `2026-06-27-tier2.1-analogy-NEGATIVE.md`; the +0.0015 delta-alignment residual | A NEGATIVE: the learned codes carry similarity but no additive RELATIONAL geometry; fluid analogy emerges at LM scale, gated on the deep-knowledge corpus, not a missing circuit. (Regime A — analogy over EXPLICIT factored relations — is already GO, `2026-06-27-tier2.1A-...-GO.md`; only the raw-code regime B is deferred.) |

**Also flagged ACCEPTED-DEEP-adjacent (representation frontiers, not mineable lexicons):** the common-ground TAG
VALUE (needs a listener model — `common_ground_composer`) + the tense TAG VALUE inferred from event semantics (needs
a Reichenbach reference-time representation — `tense_aspect_composer`). The tag BIND is spiking; only the VALUE is
host-supplied. MEDIUM–HIGH, deferred (not in the cheap sequence).

---

## SECTION D — LEGIT-HOST (EXCLUDE — environment + body; noted for auditability)

These were inspected and ruled legitimate (so the inventory is auditable):

- **The environment** — the gridworld state (agent/goal positions, the grid) + rendering the retinal image the
  neural retina receives; the reward the world DELIVERS (vs the brain's appraisal of it, which is N-3). Legit per
  BRAIN-BASED-ONLY rule (1).
- **The body** — acting on the spiking motor selection (moving the agent based on which commit pool bursted); the
  `max()` tie-break body-read of which commit pool won; emitting the answer token from the spiking who/what
  selection. Legit per rule (2).
- **Parser/composer firing READ-OUTS as body/route reads** — `brain_conversational_agent.py:135-137` (`max(rates,...)`
  reads which role ensemble fired to route a word), `OneBrainComposer` `to_host(cp_membrane_potential_v)` reads.
  Reading a SPIKING result to route/observe = legitimate (where the winner-PICK itself is a host argmax over a
  NON-spiking membrane, it IS counted — but the onebrain `_select` is spiking-default, so this is clean).
- **Host-side CURRICULUM prep** — the corpus mining (spaCy parse, co-occurrence counts) that prepares the syllabus
  the brain RENDERS/RECALLS through spikes (B-mine-1/2). Preparing the input is the environment's job; the brain
  learns from it.
- **Metrics / scoring / gate aggregation / CSR-plasticity bookkeeping** — recall/abstain counting, GO-gate
  aggregation, host index arithmetic to build/freeze synapse masks. Outside the sensation→action path = legitimate.

---

## SECTION E — THE MASTER BURNDOWN SEQUENCE (cheapest-first, then the research-gate)

### Phase 1 — CHEAP DEPLOYMENT FLIPS (closeable-cheap; the spiking/learned form is validated; flip the default)

1. **★ C-1 — flip the first-chat CONSOLE default to `--composer onebrain` on GPU (keep `rf` as the oracle/CPU
   default).** THE RECOMMENDED FIRST CLOSE-OUT. The flagship user surface becomes fully-spiking-one-brain (the whole
   flat who/what + typed verb-frame recall/answer on the persistent spiking bridge). De-risk: C3's 24/24
   answer-parity + rubric 10/10 + moat 0-FA already done; the flip is the deploy. One-line anti-cheat: the
   `SIM_BACKEND=numpy` / `--composer rf` oracle path must stay byte-identical.
2. **C-2 — deploy the mined verb-frame lexicon + wh-map as the composer/parser default** (rubric-gated; keep the
   hand dict as the fallback for vocab-poor brains). *Already in flight (CYCLE-667 task 1) — coordinate.* Anti-cheat:
   permuted-mining still collapses; moat 0-FA; rubric 10/10.
3. **N-1 — flip the nav gate/demo CLI default to `readout_source="spiking_wta"` + `--urgency-max-pa 180`** (keep
   `"motor"` as the explicit historical-benchmark oracle). The spiking decision is the validated library default;
   this retires the host-argmax decision in the SHIPPED gate. Anti-cheat: gate score == library run; conversational
   moat unaffected (array-disjoint).

### Phase 2 — SMALL FOLLOW-ON BUILDS (closeable-cheap; a small build/de-risk, no research wall)

4. **C-5 — produce the 320-scale `--readout-norm neural` codes for seeds 43/44 and default the demo to them.**
   Offline learning-pipeline conversion; the neural circuit is de-risked at 96% of host. Anti-cheat: moat 0-FA on
   the neural codes; who/what parity.
5. **C-4 — a wider-D / sharded brain so the spiking-WTA cleanup == argmax at the console scale.** Retires the last
   crowded-D winner-pick host fallback on the onebrain console path (it already matches at D=2048). Needs a wider-D
   brain (a build), not research.
6. **C-6 — RULE on the grounding projection** (`np.angle(proj @ code_vec)`): accept as a legit fixed cortico-cortical
   fan-in (recommended — it is run once, not per-fact), or realize it as a fixed complex bridge synapse. Owner call;
   low stakes.

### Phase 3 — THE RESEARCH-GATE (the deep-research loop fires BEFORE building — the documented NO-GO)

7. **N-2 + N-3 + N-4 + N-5 as ONE UNIT — the navigation reward / value / SC-orienting / place CLOSED LOOP.**
   This is the foremost OPEN boundary: the SC-orient + neural-reward + critic + SNc closed loop is documented NO-GO
   (~58× worse, the actor goes silent). The organs each work in isolation; the loop does not sustain. **Do NOT flip
   these CLI defaults until the loop is closed** — and per BRAIN-BASED-ONLY, an honest NEGATIVE here (the substrate
   limit of sustained reward-driven point-neuron control) IS the deliverable. The deep-research gate (catalog +
   Kandel + literature on sustained orienting / actor-critic stability / place-field self-organization) fires as the
   standing opening move; the host-heuristic fix is just ONE option the research ranks, never the default.

### Phase 4 — ACCEPTED-DEEP (the months-frontier; the owner's deliberate call, NOT this burndown's build)

8. **DEEP-1/2/3** — the exact-inverse bind algebra + developmental binding self-org; the learned recursive
   generative grammar (BPTT-SNN at scale + no-forget consolidation); B2 fluid analogy from raw codes. Tracked,
   deferred; the artificial-life horizon.

---

## VERDICT

- **Enumerated inventory (counts):** **7 CLOSEABLE-CHEAP** (C-1 console default · C-2 mined-frames default · N-1 nav
  decision CLI · C-4 cleanup-WTA wider-D · C-5 PPMI-neural codes · C-3 rf-argmax-as-oracle · C-6 grounding-projection
  ruling) + **4 RESEARCH-GATE-NEEDED** (N-2 value-critic · N-3 reward · N-4 SC-orienting · N-5 place — the one nav
  closed loop) + **3 ACCEPTED-DEEP excluded** (DEEP-1 bind-algebra+dendritic-binding-self-org · DEEP-2 recursive
  generative grammar · DEEP-3 B2 fluid analogy) + **LEGIT-HOST excluded** (environment · body · firing read-outs ·
  curriculum prep · scoring/bookkeeping).

- **The ranked closeable burndown sequence:** **(1) C-1** flip the console default to onebrain [the keystone] → **(2)
  C-2** deploy mined frames/wh-map as default [in flight] → **(3) N-1** flip the nav decision CLI to spiking_wta →
  **(4) C-5** the 320-scale neural-norm codes → **(5) C-4** wider-D cleanup-WTA → **(6) C-6** rule the grounding
  projection.

- **Which items need research-gates:** the **navigation reward/value/SC-orienting/place closed loop (N-2/N-3/N-4/N-5)
  as a UNIT** — the documented NO-GO; the deep-research loop fires before building, and an honest NEGATIVE is the
  deliverable. (N-1, the nav DECISION, does NOT — its spiking form is validated; it is a CLI flip.)

- **The precise ACCEPTED-DEEP boundary:** (1) the exact-inverse FHRR bind algebra + the developmental
  self-organization of the binding connectivity (host-injected bind weights → on-substrate; both dendrite jobs ruled
  out cheap-first; more-codes/capacity or the dendritic substrate, months); (2) a learned recursive GENERATIVE
  grammar that produces novel language (the BPTT-SNN frontier; toy gates met, SCALE open); (3) B2 fluid analogy from
  raw learned codes (a NEGATIVE; LM-scale corpus, not a circuit). Plus the common-ground/tense tag-VALUE inference
  representation frontiers. These are correctly excluded — months-long, owner-accepted.

- **The recommended FIRST close-out:** **C-1 — flip the first-chat CONSOLE default from `--composer rf` to
  `--composer onebrain` on the GPU path** (keep `rf` as the documented test-oracle + the `SIM_BACKEND=numpy` CPU
  fallback). It is the single cheapest, highest-value, lowest-risk close-out: the spiking onebrain path (C3 flat +
  C4 typed) is already built and validated answer-identical (24/24, moat 0-FA, rubric 10/10), and this one flip makes
  the flagship surface the owner actually talks to fully-spiking-on-one-brain — exactly the proven `consolidated_320`
  default-flip pattern.

All conversions are reuse-by-import; the no-confab moat is preserved throughout; the numpy/rf path is retained as the
test-oracle + CPU-portable path. NO `sim/` edit is contemplated for any Phase-1/2 closeable (the nav research-gate
loop may require protected edits, gated by the deep-research loop, as ever).
