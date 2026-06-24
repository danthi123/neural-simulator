# NEXT-FRONTIER SCOPING — the standing research-gate (read-only, 2026-06-24)

**Role:** read-only deep-research + scoping. NO edits, NO runs, NO webapp touch (live at :8765).
**Trigger:** the conversational goal + the inventory burndown + the artificial-life capstone+console are
comprehensively complete and live-verified (CYCLE 505-518 / 2026-06-24). Determine + research-gate the genuine
NEXT frontier from the actual state.

> **Headline:** the project just CLOSED its three biggest open arcs in the same overnight run — (1) the
> conversational goal (fluent multi-turn tested + done, CYCLE 512), (2) the inventory burndown (all 14 items
> closed, 0 `sim/` edits, moat 0-FA throughout — `2026-06-24-closeout-audit-default-on.md`), (3) the
> artificial-life week-1 develop loop + interact console (`2026-06-24-week1-develop-loop-console-capstone.md`).
> Both feared "deep walls" came back as **SURPASS-validated boundaries** (3B dendritic multi-attr bind =
> point-neuron boundary even at the oracle ceiling; 3F SC sustained-orienting = already-surpassed ~2.4× residual
> = the B-4 finite-size floor). **⇒ there is essentially NO pending code work, and the two deepest frontiers are
> RESOLVED-as-boundaries, not pending builds.** The next frontier is therefore a *deliberate direction choice*,
> and it falls squarely on the owner's standing TOP directive — **Tier-2 TRUE ONE BRAIN: the persistent
> integrated spiking loop** (make the whole pipeline ONE continuously-running interacting spiking loop, the
> between-op hand-offs synaptic, not host-orchestrated op-at-a-time). That arc is **substantially de-risked but
> NOT yet BUILT**, it is the owner's named #1 north-star for "one brain," and — uniquely among the candidates —
> it UNLOCKS the emergent features (graceful degradation, neuromod mood, reconsolidation) that an op-at-a-time
> host loop structurally cannot have.

---

## 0. STATE-OF-PLAY (what is actually closed, with citations)

| Arc | Status (2026-06-24) | Evidence |
|---|---|---|
| Conversational goal (fluent multi-turn, anaphora, firewall) | **DONE + owner-tested** | CYCLE 512; `_self_knowledge_multiturn_test.json` (FLUENT/ANAPHORA/FIREWALL all pass, recall 0.94, 0 leaks) |
| Inventory burndown (14 items) | **CLOSED — 0 `sim/` edits, moat 0-FA** | `2026-06-24-closeout-audit-default-on.md`; CYCLE 513-518 |
| Conversational spiking default-flips (1A: C-3/C-5 cleanup+assoc) | **default-ON, production** | CYCLE 512; rf byte-identity preserved |
| Nav limbic core spiking (1B: δ=r−V spiking, N-1/N-2) | **default-ON, deployed episode loop** | CYCLE 512; Schultz RPE battery 2/2; commit `c86d3441` |
| DA→composer read-side salience gate (I-4-a) | **default-ON** (`enable_da_salience_gate=True`) | CYCLE 512 `ac7d3acefa70e96ea`; moat 15/15 |
| Artificial-life week-1 develop loop + per-day console bundles | **DEMONSTRATED + LOCAL (~15 min/week)** | `2026-06-24-week1-develop-loop-console-capstone.md` |
| **3B — learned generalizable MULTI-ATTRIBUTE bind (FHRR-idealization residual / C-1/H-3)** | **SURPASS-validated BOUNDARY** (NEGATIVE even at oracle capacity) | `2026-06-24-burndown-3B-deep-dendritic-binder.md` |
| **3F — SC sustained-orienting loop (B-2)** | **SURPASSED/CLOSED** (~2.4×, = B-4 floor) | `2026-06-24-burndown-3F-sc-sustained-orienting-surpass.md` |
| Bridge co-residence of the 494M spiking Qwen faculty | **DEMONSTRATED (14 GB local, bit-exact); prefill 187 tok/s** | CYCLE 497-500 |
| Generative loop (train→generate→grow→no-forget, fully-spiking C1) | **DEMONSTRATED + multi-seed robust (toy)** | CYCLE 478-482 |

**The genuinely-open surface, classified:**
- **In-flight capability work (NOT a deferred default):** the fully-spiking generative-replay sampler (3E G2→G3),
  last item in flight at CYCLE 518 — a capability ADDITION (the owner's generative-sequence primary), not a gap.
- **Functional one-brain CONSOLIDATION (de-risked, partly BUILT, BUILD residual remains):** the cheap CPU de-risks
  landed in the overnight run (I-1-a op-handoff byte-identical GO; I-5-a synaptic parser→composer GO; I-4-a DA
  gate flipped on), but the **BUILD/consolidation is NOT done** — verified: `nav_conv_merged_bridge.py:540-541`
  still defaults `co_resident_perception=False`, `co_resident_command_route=False`, and the persistent op-handoff
  loop (I-1 BUILD) is not wired into the production composer. **This is the Tier-2 frontier.**
- **Deep-research boundaries (NOT pending code):** C-1/H-3 (3B, resolved boundary), B-1 graded place-value δ
  (dendritic, behaviorally inert), B-3 TD temporal-credit (dendritic), B-4 (~16% finite-size floor, closed).
- **Artificial-life HORIZON scaling (A2):** plateaus on the current ~25-word corpus (the develop-loop's vocab
  caps at the curriculum size — day 3-6 flat at vocab 24 / facts 11 in the week-1 run); a longer/richer
  taxonomy is needed to show more growth.

---

## 1. RANKED NEXT-FRONTIER CANDIDATES (leverage × readiness)

> Scoring axes: **leverage** = alignment with the owner's standing TOP directive (TRUE ONE BRAIN /
> artificial-life north-star) × how much it unlocks; **readiness** = how cheaply it can be de-risked/built from
> existing GO machinery vs. a months-scale / high-variance / already-bounded effort.

### ⭐ RANK 1 — Tier-2 TRUE ONE BRAIN: the **persistent integrated spiking loop** + consolidate the GO cross-region interactions into the DEFAULT merged loop (I-1 BUILD + I-4-a/b + I-5-a + I-7-b)
**Leverage: HIGHEST. Readiness: HIGH (consolidation + residual-closure of already-GO mechanisms, not new science).**

- **Why #1:** it IS the owner's standing TOP directive (`feedback_move_everything_to_shared_spiking_substrate`,
  `project_one_brain_integrated_pipeline_and_cleanup`, Tier-2 of `project_post_conversational_roadmap_tiers`).
  The conversational + burndown + capstone arcs are done; this is the *named exact-next* the owner has reiterated
  twice ("the persistent integrated spiking loop — the headline one-brain build; also unlocks the emergent
  features an op-at-a-time host loop can't have").
- **Why now-readiness is HIGH:** the deep-research GATE is already done (`2026-06-23-functional-one-brain-integration-scoping.md`)
  and its verdict is decisive — functional integration is **substantially already built + validated**, the work
  is **consolidation + residual-closure, NOT new mechanism**. The three cross-region interactions are each 6-seed
  GO (language→action `spoken_instruction_nav`; perception→memory `navigate_to_see_then_answer`;
  perception→compose `navigate_to_compose_then_answer`), the I-1-a register-handoff is byte-identical GO (CYCLE
  512), the I-5-a synaptic parser→composer is GO (CYCLE 512), the I-4-a DA read-gate is flipped on. **The
  residual is the BUILD: (a) wire the persistent op-handoff-as-spikes into the production composer (retire the 3
  remaining `to_host`+re-kick round-trips), (b) flip the three GO interactions into the merged DEFAULT, (c) close
  the host grounding projection `M` via the GO Hebbian convergence.** This is the unique candidate that is both
  the top directive AND mostly-cheap.
- **Honest scope:** mostly **compose-already-proven-pieces** + reuse-by-import; the genuine new work is small (the
  persistent register→register loop in the composer; the deep I-1-c clause-renormalize and I-7-c RF-dynamics DA
  threading are bounded `sim/`-edit follow-ons, deferred). See §2 for the full diagnosis + cheap-first de-risk.

### RANK 2 — Tier-3 artificial-life: scale the HORIZON with a richer developmental taxonomy (A2 + the corpus expansion)
**Leverage: HIGH (the north-star). Readiness: HIGH-but-plateaus-without-a-corpus.**

- **Why high-leverage:** the artificial-life persistent living agent IS the project's stated end-goal
  (`project_actual_goal_artificial_life_brain_analogue`, Tier-3). The week-1 loop is DEMONSTRATED; the natural
  next is a longer horizon (month/year) where the brain visibly develops.
- **Why NOT #1 — the corpus-cap plateau (the prompt's flagged caveat, confirmed):** a pure "run the develop loop
  longer" PLATEAUS — the week-1 run shows vocab flat at 24 / facts 11 from day 3 onward because the curriculum
  caps at ~25 words. To show MORE growth needs a **bigger developmentally-graded taxonomy** (a multi-hundred-word
  graded syllabus tracking the TierLadder 4→8→…→320, `auto_growth.py:43`). That is a content-authoring +
  loop-extension job (small, NO `sim/` edit), but it is *prerequisite* work before the longer run pays off — so
  the leverage is real but the "just run longer" framing is a trap. The stream cortex is already 320-scale
  validated (`stream_taxonomy_320.py`), so the substrate is ready; the gating piece is the graded multi-day
  curriculum + a longer-horizon loop run (overnight, LOCAL — ~13.5 hr/year per the ETA, <24 GB).
- **Honest scope:** **compose-already-proven-pieces** (stream cortex + MultiTurnAgent + consolidation + lineage +
  the existing develop-loop scheduler) + author a richer graded curriculum. NO new mechanism. The one caveat: the
  *generative-faculty* distribution-growth hits the C2 moderate-shift capacity wall — but that is the OPTIONAL
  free-generation upgrade, NOT the develop metric (no-forget is carried by stream cortex + lineage + SWR).
- **Relationship to #1:** complementary — #1 makes the brain ONE interacting loop; #2 makes that one brain LIVE
  LONGER. A natural sequence is #1 (the integrated loop) → #2 (run the integrated brain over a long horizon =
  the true persistent living agent), which is exactly the Tier-2→Tier-3 path.

### RANK 3 — The generative-sequence frontier: integrate the BPTT-SNN generative loop into the ONE conversational brain (3E)
**Leverage: MEDIUM-HIGH (the owner's 2026-06-22 named primary). Readiness: MEDIUM (core demonstrated; on-brain integration is the work).**

- **Why relevant:** `project_generative_sequence_frontier` is an owner-approved primary — close the MEASURED
  categorical free-generation gap (the composer RETRIEVES, never GENERATES). The core loop is DEMONSTRATED +
  multi-seed robust + fully-spiking-C1 (CYCLE 478-482); the b2 generative-replay proposer is GO (the first
  brain-mechanism novel-composition >0, CYCLE 511).
- **Why NOT #1/#2:** the **core is already proven**; what remains is integrating the BPTT-SNN generator + the b2
  proposer into the ONE conversational brain so the brain generates novel grounded discourse verified by the
  moat (3E in the burndown roadmap). That is a real BUILD but it is downstream of the one-brain integration (#1
  provides the persistent loop the generator would live in), and the standalone generative pieces are GO. It also
  carries a known scale caveat (the C2 moderate-shift capacity wall for in-band generation — needs ~50-200M
  params, likely-local). Best sequenced AFTER #1 (it is literally roadmap 3E, which the roadmap places after the
  2B integration).

### RANK 4 — Bridge-co-resident LLM: functional integration + perf (2A + 3D)
**Leverage: MEDIUM. Readiness: HIGH-cheap-flips but the payoff is usability, not a new capability.**

- **Why relevant:** bridge co-residence is DEMONSTRATED (24-layer Qwen on the live RF substrate, bit-exact, 14 GB
  local) but co-RESIDENT not INTERACTING; the gate→constrain→verify grounding loop is a host pipeline with 2-of-3
  ops on-bridge. The perf levers (O-1 on-GPU forward → ~7200 prefill / ~330 gen tok/s) make it usable-local.
- **Why NOT higher:** the LLM-perf scoping (`2026-06-24-bridge-llm-perf-integration-scoping.md`) shows this is
  mostly **cheap flips + host-forward** (only ~1 optional default-off `sim/` edit for the dense-RF purity); the
  payoff is *usable real-time-local language* + on-chip representation, which is engineering polish on an
  already-demonstrated capability, not a new frontier. It is a strong candidate to PARALLELIZE (the perf chain is
  independent of the integration arc), but not the headline next direction.

### RANK 5 (deprioritize — NOT pending builds) — the deep dendritic / learned-cortex walls (C-1/H-3, B-1, B-3, B-4)
**Leverage: would-be-high IF open; Readiness: RESOLVED-as-boundaries → not a frontier.**

- **Why deprioritized:** these were the feared "Tier-4 deep walls," and the burndown's SURPASS rounds resolved
  them as **honest point-neuron boundaries, not pending builds**:
  - **C-1/H-3 (learned generalizable multi-attribute bind):** NEGATIVE even at the **oracle capacity ceiling**
    (held-out 0.007, train 0.984 = pure memorization signature; the fixed ±1/FHRR primitive wins at 0.228
    *because it doesn't learn per-role priors*). 3 independent NEGATIVEs (learned-linear / single-layer / deep).
    The fixed FHRR self-inverse STAYS as a STRUCTURAL neural primitive (not a host shortcut). The months-scale
    `TWO_COMPARTMENT` dendrite build is gated behind a cheap-GO that already came back NEGATIVE.
    `2026-06-24-burndown-3B-deep-dendritic-binder.md`.
  - **B-2 (SC sustained-orienting):** SURPASSED at ~2.4× (the residual = the B-4 finite-size margin-SNR floor).
  - **B-1 / B-3:** the dendritic-candidate δ items are behaviorally INERT on the orient-solvable task (the host
    Gaussian is the better scaffold; the merged anti-cheat couldn't certify B-3 value-driven). Substrate-mapping
    deliverables, not capability gaps.
  - **B-4:** the ~16% spiking-decision cost is the irreducible BRAIN-BASED-ONLY residual, closed-default.
- **The one thing here that IS still genuinely a frontier (but DEEP/high-variance, owner-scoped):** the full
  `NeuronModel.TWO_COMPARTMENT` apical/basal substrate (catalog T3.A, ~10× compute). It is the only piece the
  dendrite scoping left as "months-scale," and it is gated behind an UNLIKELY cheap-GO (the cheap deep-binder
  de-risk already failed). Per the standing directive it stays an owner-steer call, **not the recommended next
  move** — building it would re-deliver an already-banked capability (semantic generalization is DONE on point
  neurons via PPMI) while the genuine residual (learned multi-attr bind) is a proven boundary.

---

## 2. TOP-RANKED FRONTIER (Rank 1) — full research-gate

### 2.1 Diagnosis (the precise gap/boundary)

"One brain" today is **substrate-consolidation with ONE genuinely-default interaction** (the DA read-gate,
I-4-a). The owner's bar is **functional integration**: the whole pipeline as ONE continuously-running interacting
spiking loop, where (a) the brain's own ops hand off as SPIKES (no host round-trip between ops), (b) the
cross-region influences (language→action, perception→memory/compose, limbic→cortex) are the DEFAULT merged loop,
and (c) no host quantity is smuggled across regions. The two disqualifiers (both present today):

1. **Host round-trip BETWEEN ops inside the composer (I-1).** Even on the one-brain path,
   `one_brain_composer.py` sequences ops via `to_host(rf_read_phases) → np.kick → next op` at 3 sites
   (`_compose_phases` / `_decode_clause` / `_recovered_patient_phases`). The megakernel (`rf_megastep`) fuses
   WITHIN an op; the BETWEEN-op handoff is host. **This is the literal "persistent interacting spiking loop"
   gap.** (CYCLE 512 de-risked the fix — I-1-a register→register handoff is byte-identical GO — but did NOT wire
   it into the production composer.)
2. **The three GO cross-region interactions are NOT the merged DEFAULT.** Verified in current code:
   `nav_conv_merged_bridge.py:540-541` defaults `co_resident_perception=False`, `co_resident_command_route=False`;
   the default `MergedNavConvAgent` has zero nav↔composer cross-synapses beyond the DA read-gate. The
   interactions live in standalone task runners / opt-in kwargs. And the perception→compose grounding still
   crosses the code via a **host projection** `composer.concepts[o] = angle(M @ live_rate)`
   (`navigate_to_compose_then_answer.py:_perceive_and_ground`, the I-4-resid / N-4 / H-2 residual).

### 2.2 Ranked biologically-grounded options (catalog-cited)

- **I-1 — op-handoff-as-spikes (the persistent interacting loop).** Biology: **reentrant
  cortico-BG-thalamo-cortical loops (catalog A.05** — parallel reverberatory channels; selection is an emergent
  property of the reentrant network) hold intermediate results in *sustained spiking/synaptic state*, not a host
  buffer; **WM attractors (catalog G.06 sustained delay-period activity / G.08 persistent activity for active
  maintenance)** hold a value across a delay without read-out-and-reinject.
  - **Option I-1-a [RECOMMENDED, de-risked GO]:** keep the composite ON-substrate between ops — a fixed
    identity-phasor route source-register → next-op input-register, so op N+1's resonate is DRIVEN by op N's
    register state directly. The megakernel already advances the whole masked RF slice each step. CYCLE 512
    proved byte-identity (atol 1e-9) across 9 cases; **the BUILD = wire it into the 3 production round-trip
    sites.** Likely reuse-by-import.
  - **Option I-1-b [bigger payoff, `sim/` edit]:** a single fused multi-op megakernel pass chaining
    bind→bundle→unbind→cleanup as one launch over the masked slice (folds the O-1/O-6 perf levers). Additive,
    default-off, byte-reviewed.
  - **Option I-1-c [deferred residual]:** the clause re-normalize circuit (an attractor that re-discretizes the
    phasor between hops) — the same magnitude-floor nonlinearity the composer already relies on; a bounded
    follow-on after the flat path lands.
- **I-4 — make the interactions the DEFAULT (+ kill host `M`).** Biology: perception→episodic/relational memory
  (**catalog D.02** Eichenbaum-Cohen relational binding; **D.14** Tonegawa engram cells — "the neurons that fired
  ARE the memory"); grounded/embodied semantics = the ATL convergence-zone story (**G.20** / D.02-supplemental,
  Patterson-Lambon Ralph).
  - **Option I-4-a [DONE]:** the DA read-side salience gate (flipped on).
  - **Option I-4-b [RECOMMENDED, GO mechanism]:** replace the host grounding `M` with the LEARNED Hebbian
    crossmodal convergence (`_genfrontier_onsubstrate_convergence_derisk.py`, held-out cat-acc 0.92 on SPIKES,
    NMDA-fired) already wired in the merged bridge (`gen_perception→gen_concept`). A wiring/consolidation job.
  - **Option I-4-c:** perception→memory via engram tags as the default (sidesteps the rate↔phasor wall for
    RECALL; cheapest, recall-only).
- **I-5 — synaptic parser→composer (replace the Python `{role:word}` dict).** Biology: dual-stream language
  (**catalog G.11**), Pulvermüller distributed cortical word ensembles (**G.20**). The synaptic precedent EXISTS
  (`unified_brain_bridge.hear_synaptic` + `_op_synaptic`). CYCLE 512 de-risked I-5-a GO; the BUILD = port
  `hear_synaptic` onto the merge for the `rf` composer.
- **I-7 — limbic→composer deep route ("one self").** Biology: dopamine gates entry into LTM (**catalog D.16**
  Lisman-Grace hippocampal-VTA loop); D1 inverted-U sharpens PFC tuning (Vijayraghavan/Arnsten). I-7-a (read-side)
  DONE; **I-7-b [cheap, GO] the encoding hook** (`encoding_gain_fn`, deployment-smoke gap); I-7-c (DA threads RF
  dynamics) = deferred `sim/` edit.

### 2.3 Reusable existing machinery (almost everything)

- `COMMAND_GATE` parser-firing→transmission-gate template (`spoken_instruction_nav.py`) — the canonical "firing
  opens a synaptic route" pattern.
- `couple_gate_to_pool`/`couple_gate_to_indices` + `set_transmission_gate` + `_apply_gate_couplings`
  (`sim/bridge.py:3179-3229`) — open any gate from a pool's firing, in-substrate, zero `sim/` edit.
- `hear_synaptic` + `_op_synaptic` (`unified_brain_bridge.py:447,509`) — the I-5 synaptic parser→composer route.
- `rf_megastep` masked megakernel (default-on for `OneBrainComposer`) — WITHIN-op fusion; the I-1-a register
  handoff is the BETWEEN-op residual.
- The three GO interaction runners (`spoken_instruction_nav`, `navigate_to_see_then_answer`,
  `navigate_to_compose_then_answer`) + the `co_resident_*` builder kwargs + `finalize_conv_for_nav_gate`
  (gate-safe re-injection on the framework bridge).
- `gen_perception→gen_concept` learned convergence (de-risked GO) for I-4-b; `_da_confidence_gate` (read-side,
  done) + `encoding_gain_fn` (encoding-side, GO) for I-7.

### 2.4 Recommended cheap-first de-risk (the smallest falsification probe)

**The CPU probes already passed (I-1-a byte-identical GO, I-5-a route==ground-truth GO).** The remaining
decisive de-risk is the **BUILD-level integration smoke** — the smallest probe that proves the *persistent
integrated loop* end-to-end without committing the full consolidation:

> **Probe:** on the production `OneBrainComposer`, replace the 3 `to_host`+re-kick handoff sites with the
> de-risked I-1-a register→register identity routes (flat path only), and run the full who/what conversational
> matrix + the `is None` no-confab abstentions on the persistent-loop composer.
> **GO bar:** every answer (all stored facts + every abstention) is **byte-identical (atol 1e-9)** to the
> host-round-trip composer, AND the 11-test CI (`test_one_brain_composer_agent.py`) passes verbatim, AND the
> moat is 0-FA. (Tractable: V≤64 smoke, reuse existing tests, foreground — per the CYCLE-512 lesson that
> subagents stall on big GPU runs.)

If byte-identical → the persistent loop is real and we wire it in. If the clause path drifts (expected, |Z|
decay) → that localizes exactly the I-1-c clause-renormalize follow-on, and the flat path still ships. This is
the cheapest probe that converts "de-risked op-handoff" into "the production composer IS a persistent spiking
loop."

### 2.5 Anti-cheat controls (the integration must be SYNAPTIC)

1. **Provenance — no host quantity smuggled across regions.** Grep the integrated path for any
   `cp_external_input_current[<B>] = f(to_host(<A>))` or `composer.concepts[o] = host_fn(...)`. The ONLY
   legitimate host writes are the environment presenting instruction TEXT / rendering the object (sensory
   render) + the body moving on the motor/sel winner. Cross-region coupling = a **0/1 gate STATE from firing** or
   a **fixed synaptic route**, never a value copy. (I-4-b must REPLACE the current `navigate_to_compose` host-`M`
   provenance assertion with a spikes-only one.)
2. **Lesion = the interaction vanishes.** Cut the cross-region route → the behavior collapses to chance / recall
   fails. The route must be NECESSARY (all three GO milestones carry this).
3. **Both-brains-required.** Isolated-nav (route lesioned) and isolated-conv (no body) each fail; only the
   coupled brain solves it.
4. **No-confab moat preserved — the HARD invariant.** No abstention the host returned may become a false-accept
   on any integrated path or at any DA level. (The DA hooks are moat-safe by construction — a higher gate only
   TIGHTENS abstention.) Every regression re-runs the three `is None` assertions. Per
   `feedback_moat_not_hard_lossy_memory_ok`, the moat is kept-where-free, never traded for a number.
5. **Byte-identity gates.** `test_nav_conv_merged_agent` 8/8 + `test_nav_conv_step2b_coresident` 7/7 must still
   pass under any default-flip; the persistent-loop composer must be answer-identical to the host-round-trip
   composer on the flat path.

---

## 3. HONEST SCOPE + owner-steer / `sim/`-edit flags

- **Rank 1 is mostly COMPOSE-ALREADY-PROVEN-PIECES** (the deep-research gate's load-bearing finding:
  "consolidation + residual-closure, NOT new mechanism"). The genuine new work is small:
  - **Reuse-by-import, NO `sim/` edit:** wire I-1-a into the 3 composer sites; flip the three GO interactions to
    the merged DEFAULT (regression-gated); port `hear_synaptic` (I-5-a); swap the host `M` for the GO Hebbian
    convergence (I-4-b); deploy-smoke the I-7-b encoding hook.
  - **Additive, default-off, BYTE-REVIEWED `sim/` edits (deferred deep residuals):** the I-1-b fused multi-op
    megakernel (only if the identity-route I-1-a is insufficient for perf/purity); the I-1-c clause-renormalize
    circuit; the I-7-c DA-threads-RF-dynamics. Per `feedback_dont_gate_on_approval`, reuse-by-import items
    proceed directly; the `sim/` edits get the standing byte-level diff review.
- **Owner-steer points (genuine forks, not resolvable from context):**
  1. **Direction choice itself** — Rank 1 (the integrated loop, the named #1 directive) vs. starting with Rank 2
     (the longer artificial-life horizon) vs. parallelizing Rank 4 (LLM perf). The recommendation is Rank 1 (it
     is the standing TOP directive AND mostly-cheap AND unlocks the emergent features), with Rank 2 as the
     natural Tier-3 follow-on once the brain is ONE integrated loop, and Rank 4 as an independent parallel track.
  2. **The TWO_COMPARTMENT dendrite (Rank 5 tail)** — the only remaining months-scale build, gated behind an
     already-NEGATIVE cheap de-risk; an explicit owner call, NOT the recommended next move.
- **Rank 2's prerequisite (flagged):** a "run the develop loop longer" PLATEAUS at the ~25-word corpus cap →
  needs a richer graded multi-hundred-word taxonomy FIRST (small, NO `sim/` edit, but prerequisite). The
  substrate (320-concept stream cortex) is ready; the gating piece is curriculum authoring + a longer-horizon
  loop run (overnight, LOCAL).
- **Cloud:** none of the recommended work needs cloud (all <24 GB, LOCAL). The ONLY potential cloud trigger is
  the OPTIONAL generative-faculty scale-up (Rank 3, the C2 in-band capacity wall, ~50-200M params — and even that
  is likely-local). Per `feedback_long_local_runs_ok_confirm_cloud_cause`, measure VRAM+throughput+ETA before any
  scale-up; do NOT propose cloud for the integration loop or the develop-loop horizon.

---

## 4. BOTTOM LINE

The project closed its three biggest arcs (conversation, burndown, artificial-life capstone) and resolved both
feared deep walls as SURPASS-validated boundaries — there is essentially no pending code work. The genuine next
frontier is a deliberate direction choice that lands on the owner's standing TOP directive:
**Tier-2 TRUE ONE BRAIN — the persistent integrated spiking loop**, consolidating the already-GO cross-region
interactions into ONE continuously-running interacting spiking loop with synaptic op-handoffs. It is the named
#1 north-star, it is mostly compose-already-proven-pieces (the deep-research gate is done; the cheap CPU de-risks
landed in the overnight run), and it uniquely unlocks the emergent features (graceful degradation, neuromod mood,
reconsolidation) an op-at-a-time host loop cannot have — then flows naturally into the Tier-3 longer-horizon
persistent living agent (Rank 2).

**Recommended cheap-first de-risk:** the BUILD-level persistent-loop smoke — wire the de-risked I-1-a
register→register handoff into the 3 production composer round-trip sites (flat path) and prove the full who/what
matrix + the `is None` abstentions are byte-identical (atol 1e-9) to the host-round-trip composer, CI 11/11,
moat 0-FA. **Anti-cheats:** provenance (no host quantity across regions), lesion-vanishes, both-brains-required,
moat-preserved (HARD), byte-identity gates. **`sim/`-edit flag:** the flat-path wins are reuse-by-import, NO
`sim/` edit; the deep residuals (I-1-b fused megakernel / I-1-c clause-renormalize / I-7-c DA-threads-RF) are
additive default-off byte-reviewed follow-ons. **Owner-steer:** the direction choice (Rank 1 vs Rank 2-first vs
parallel Rank 4) + the TWO_COMPARTMENT dendrite (the only months-scale build, gated behind an already-NEGATIVE
de-risk, deprioritized).
