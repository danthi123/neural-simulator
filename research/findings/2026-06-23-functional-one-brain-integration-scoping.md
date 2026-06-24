# Functional one-brain integration — scoping (DEEP-RESEARCH GATE, read-only)

**Date:** 2026-06-23. **Role:** read-only deep-research + scoping subagent (no edits, no GPU).
**Trigger:** burndown roadmap Phase 2B (`docs/plans/2026-06-23-inventory-burndown-roadmap.md`) +
inventory items **I-1 / I-4 / I-5 / I-7** (`research/findings/2026-06-23-cheats-shortcuts-integration-inventory.md`).
**North-star:** the owner's "real one brain" — navigation and conversation INTERACTING via synapses, the
brain's own ops handing off as SPIKES (not host round-trips), the limbic core reaching the cortex
(`project_one_brain_integrated_pipeline_and_cleanup`, `feedback_move_everything_to_shared_spiking_substrate`).

> **Headline (the load-bearing finding for the controller):** functional integration is **substantially
> already built and validated**, in a way the inventory under-credits. The 2026-06-10 design
> (`docs/plans/2026-06-10-functional-integration-one-brain-design.md`) and its three GO milestones already
> deliver: **(A) language→action = `spoken_instruction_nav.py`, 6-seed GO**; **(B) perception→memory recall =
> `navigate_to_see_then_answer.py`, 6-seed GO**; **(B′) perception→compose = `navigate_to_compose_then_answer.py`,
> 6-seed GO**; **I-7 DA→composer read-side = `nav_conv_merged_bridge.py:_da_confidence_gate`, GO, production-wired
> (opt-in)**. So the gaps are NOT "build cross-region interaction from scratch" — they are (1) **make the three
> GO interactions the DEFAULT merged loop** (today they are opt-in builder kwargs / separate runners), and
> (2) **close the residual host round-trips INSIDE the integrated path** (I-1 op-handoff glue; I-5 parser→composer
> Python dict; the host grounding-projection `M` inside (B′); I-7 encoding-hook + deep RF threading). This is a
> consolidation-and-residual-closure arc, not a new-mechanism arc — which sharply lowers its risk and cost.

---

## 1. DIAGNOSIS — what "functional integration" concretely means here, and the precise gaps

### 1.1 The operational definition (owner standing bar)
"Functional integration" = a cross-region influence carried by **neurons + synapses on the merged bridge**, such
that an isolated half cannot reproduce the behavior (`feedback_validate_signal_by_its_function`). The two
disqualifiers, both host shortcuts:
1. **Co-location without coupling** — disjoint frozen slices on one bridge with **zero cross-synapses** (gate (a)
   proved nav byte-identical with/without the conv half). "Capability-equivalent to separate brains" is the tell
   that the merge added no integrative behavior (I-4).
2. **A host round-trip between regions/ops** — `b_value = to_host(region_a.read()); region_b.kick(f(b_value))`.
   The firing of one region must open/drive the next via synapses, not via a Python value copy (I-1, I-5).

The **WORKING TEMPLATE** that satisfies both (the model every gap below should imitate): `spoken_instruction_nav.py`
— the parser's action-role ensemble FIRING opens a transmission gate (`COMMAND_GATE`) on the LEARNED
`language_input→cortex_X` route, and the cascade selects the move in spikes. 6-seed GO, lesion-confirmed,
provenance-asserted ("no parser-derived value written to any nav drive"; the cross-region coupling is a 0/1 gate
STATE from firing, not a value). This is `couple_gate_to_indices` (the index-based sibling of the public
`bridge.couple_gate_to_pool`), driving the public `bridge.set_transmission_gate` per-step inside
`_apply_gate_couplings` (`sim/bridge.py:3229`) — **all in-substrate, no `sim/` edit.**

### 1.2 The precise gaps (grounded against current code)

- **I-4 — nav+conv CO-LOCATED, not interacting *in the default merged loop*.** `nav_conv_merged_bridge.py`
  builds disjoint slices (nav cascade + parser + dlPFC + opt-in `rf`/`cortex_it`/`limbic`/`gen`). The
  **default** `MergedNavConvAgent` has zero cross-synapses between the nav cascade and the conversational
  composer. **BUT** the three interactions ARE built as separate runners / opt-in kwargs:
  - language→action: `spoken_instruction_nav.py` (`COMMAND_GATE`, the parser opens a synaptic route into
    `cortex_X`). GO 6-seed.
  - perception→memory (recall): `navigate_to_see_then_answer.py` — the agent navigates, perceives `cortex_it`
    live, engram-tags it, recalls via a trained `cortex_it→language_output` route. GO 6-seed.
  - perception→compose: `navigate_to_compose_then_answer.py` — the agent navigates, GROUNDS each perceived
    object into a composer concept code IN-EPISODE, composes a held-out fact on the co-resident `rf` slice,
    answers + abstains. GO 6-seed (`co_resident_perception`, `co_resident_rf` builder kwargs).
  ⇒ **The I-4 residual is not "no interaction exists" — it is "the interactions are not the merged DEFAULT, and
  one of them (B′) still crosses the code via a HOST projection."** See I-4-resid below.

- **I-4-resid — the perception→compose grounding is a HOST round-trip (the real cross-region cheat inside (B′)).**
  `navigate_to_compose_then_answer.py:_perceive_and_ground` (line 195) reads the LIVE `cortex_it` rate to the host,
  applies a fixed numpy complex projection, and WRITES the result into the composer codebook:
  `composer.concepts[o] = angle(M @ live_rate)`. That `to_host(rate) → M @ → host write` is exactly the
  "host quantity smuggled across regions" pattern. The projection `M` is also host-DESIGNED (= **N-4** grounding
  projection / **H-2** host-designed structure). **This is the genuine residual cross-region host op** even though
  the behavioral loop is GO.

- **I-1 — host-orchestrated op hand-offs INSIDE the composer.** Even on the one-brain path, `one_brain_composer.py`
  sequences ops via host round-trips: `_compose_phases` → `_write_block` → `_read_blocks`/`_read_block` →
  `_select`, each reading membrane/phases to host (`to_host(b.rf_read_phases())`, `to_host(b.cp_membrane_potential_v)`)
  and re-kicking the next op (`b.rf_kick(...)`). The megakernel (`rf_megastep`) fuses WITHIN an op; the BETWEEN-op
  handoff is host. The clause decode (`_decode_clause`, line 698) is explicit about it: it READS the intermediate
  clause composite to host and RE-KICKS it as a clean unit phasor before the 2nd hop. **This is the owner's
  "persistent interacting spiking loop" gap** (`project_one_brain_integrated_pipeline_and_cleanup`).

- **I-5 — parser→composer hand-off is a Python dict.** `one_brain_composer.hear` (line 288) does
  `roles = [self.parser.role_of(pos, voice) ...]; rmap = {roles[i]: words[i]}` then calls `_store_fact` with those
  string labels. The parse is synaptic (the parser FIRES the role), but WHICH word goes to WHICH bind is selected
  by a host dict, not by the parser's firing driving the composer's role bank. **The synaptic precedent EXISTS**:
  `unified_brain_bridge.hear_synaptic` (line 447) + `_op_synaptic` (line 509) route the parser firing → a
  per-role transmission gate (`role_route_<R>`) → the composer's role bank topographically, with the
  comprehend→latch→act pre-warm. The nav+conv MERGE reverted to the Python dict "for coexistence" (the merge
  is a framework/`inject_explicit_wiring` bridge; `hear_synaptic` was wired on the standalone `UnifiedBrainBridge`).

- **I-7 — DA/NM → conversational composer (limbic↔cortex, "#6 one self").** The **read-side** DA salience gate
  IS built AND production-wired: `nav_conv_merged_bridge.py:_da_confidence_gate` (~line 1316), opt-in
  `MergedNavConvAgent(enable_da_salience_gate=True)`, reads the shared spiking-SNc `dopamine` concentration →
  sharpens the recall confidence gate; **moat-safe by construction** (a higher gate only TIGHTENS abstention,
  Vijayraghavan/Arnsten D1 inverted-U); GO (`_da_composer_salience_cleanup_derisk.py`, 6-seed precision +
  lesion-clean). The **two open pieces**: (i) the **encoding hook** (`OneBrainComposer.encoding_gain_fn`, line 116
  / `_phaseB_dopamine_encoding_gain_derisk.py`) — DA-gated encoding STRENGTH (Lisman-Grace VTA-hippocampal loop) is
  de-risked GO but only has a *deployment smoke* gap, not a wire-up; (ii) the **deep RF-dynamics threading** (DA
  modulating the resonate dynamics itself) is a *sketched* `sim/` edit, unbuilt.

### 1.3 What is REUSABLE (the inventory of working machinery — almost everything is)

| Asset | Where | Reuse for |
|---|---|---|
| `COMMAND_GATE` parser-firing→transmission-gate template | `spoken_instruction_nav.py` (whole runner) | the canonical "parser firing opens a synaptic route" pattern, for I-5 and any new cross-region route |
| `couple_gate_to_pool` (name-based) / `couple_gate_to_indices` (index-based) | `sim/bridge.py:3205` / `unified_brain_bridge.py:123` | open ANY transmission gate from a control pool's FIRING, in-substrate, zero `sim/` edit |
| `set_transmission_gate` / `_apply_gate_couplings` | `sim/bridge.py:3179` / `:3229` | the gate primitive + per-step firing→gate hook (vectorized opt-in for K-way) |
| `set_plasticity_gate` / `cp_plasticity_rate_gain` index-mask | `sim/bridge.py:3153` | freeze a frozen slice's weights against the nav reward-STDP stressor (the 5a isolation) |
| `hear_synaptic` + `_op_synaptic` (parser→gate→role-bank, comprehend→latch→act) | `unified_brain_bridge.py:447,509` | the **I-5 synaptic parser→composer route** (port from the standalone bridge onto the merge) |
| transmission-gate machinery on the framework bridge | `nav_conv_merged_bridge.py` (`finalize_conv_for_nav_gate`, the gate-safe re-injection) | wiring new gated routes on the MERGED (framework) bridge without staling gate-index maps |
| engram-tag API (`start_engram_recording`/`commit_engram_tag`/`stimulate_tag`) | `sim/bridge.py` (P2, catalog D.14) | the **perception→memory write that sidesteps the rate↔phasor cross-code wall** (recall path, already used by navigate-to-see) |
| `navigate_to_see_then_answer` (perception→memory recall) | runner | the (B) recall interaction, GO 6-seed |
| `navigate_to_compose_then_answer` (perception→compose) | runner | the (B′) compose interaction, GO 6-seed (the host grounding `M` is the residual) |
| DA read-side salience hook `_da_confidence_gate` + `da_to_gate` | `nav_conv_merged_bridge.py:1316`, `_da_composer_salience_cleanup_derisk.py` | the **I-7 limbic→composer read route**, production-wired, moat-safe |
| DA encoding-gain hook `encoding_gain_fn` | `one_brain_composer.py:116`, `_phaseB_dopamine_encoding_gain_derisk.py` | the **I-7 encoding-side** DA route (Lisman-Grace), de-risked GO, deployment-smoke gap |
| `rf_megastep` masked megakernel | `sim/bridge.py` (O-4), default-on for `OneBrainComposer` | the WITHIN-op fusion (the I-1 BETWEEN-op handoff is the residual) |
| `co_resident_limbic` / `co_resident_nav_critic` shared limbic slice | `nav_conv_merged_bridge.py:667+` | the shared DA SOURCE (spiking SNc + `dopamine` modulator) both halves read |

---

## 2. CATALOG-GROUNDED RANKED OPTIONS per sub-gap

> Catalog: `E:/Documents/Projects/sim-catalog/references/feature-catalog.md`. Biology cited FIRST, then the
> cheap-first mechanism. All options are reuse-by-import unless a `sim/` flag is noted.

### Gap I-5 — the synaptic parser→composer route (replace the Python `{role:word}` dict)

- **Biology:** dual-stream language (catalog **G.11**, dorsal sensorimotor + ventral semantic, Kandel 6e Ch 62);
  Broca→premotor action priming (**G.12**); Pulvermüller distributed cortical word ensembles (**G.20**, the
  `language_input→cortex_X` somatotopy the project validated). The comprehension→binding hand-off is a
  cortico-cortical association projection, not a host symbol pass.
- **Option I-5-a [RECOMMENDED, cheapest, reuse-by-import] — port `hear_synaptic` onto the merged bridge.**
  Re-express the standalone-`UnifiedBrainBridge` route (`_op_synaptic`: per-role `role_src` pool → topographic
  gated route `role_route_<R>` → the composer role bank, gate coupled to the parser ensemble, comprehend→latch→act
  pre-warm) as the merged bridge's framework regions/pathways (exactly how `nav_conv_merged_bridge` already ports
  the parser + dlPFC). The composer here is the `rf` slice (`OneBrainComposer`), not the `CoreSimComposer` ±1
  role bank — so the role drive must enter the RF bind, not a Hadamard AND bank (see §6 honest risk).
- **Option I-5-b — the COMMAND_GATE pattern, generalized to all three roles.** Use `spoken_instruction_nav`'s
  exact primitive (`couple_gate_to_indices` over each role sub-block → a `role_route_<R>` transmission gate on the
  word→role-bank route). This is the *same* mechanism as I-5-a but framed from the nav side; pick whichever's
  index bookkeeping is cleaner on the merge.
- **Option I-5-c [defer] — a learned cortico-cortical binder.** Replace the topographic ±1 route with a learned
  spiking map (the step-3 cortex). Deep; this is C-1/H-3 territory, NOT this arc.

### Gap I-1 — op-handoff-as-spikes inside the composer (the persistent interacting loop)

- **Biology:** reentrant cortico-BG-thalamo-cortical loops (catalog **A.05**, parallel reverberatory channels) —
  the brain holds intermediate results in *sustained spiking / synaptic state*, not a host buffer. Working-memory
  attractors (**G.06/G.08**) hold a value across a delay without a read-out-and-reinject.
- **Option I-1-a [RECOMMENDED first, cheap-ish] — keep the composite ON-substrate between ops (no `to_host`
  re-kick).** Today each op does `to_host(rf_read_phases) → np.kick → next op`. Replace with a synaptic copy: a
  fixed identity-phasor route from the source register to the next op's input register, so the resonate of op N+1
  is DRIVEN by op N's register state directly (the megakernel already advances the whole masked RF slice each
  step; the registers persist in `cp_rf_w_re/im` + the v/u phasor). Probe whether a register→register identity
  route reproduces the `to_host`+re-kick byte-for-byte. This removes the *between-op* host round-trip while
  reusing the existing within-op megakernel.
- **Option I-1-b — a single fused multi-op megakernel pass.** Extend `rf_megastep` to chain bind→bundle→unbind→
  cleanup as one launch over the masked slice (the ops are already disjoint register blocks). Bigger `sim/` edit;
  higher payoff (also folds O-1/O-6 perf). Default-off, byte-reviewed.
- **Option I-1-c [the honest scope clamp] — accept the clause re-kick as a legitimate "fresh unit phasor".**
  `_decode_clause`'s re-kick is the oracle's fresh-per-hop kick (chaining the resonate through an unbind-DRIVEN
  register degrades |Z| and mis-reads). A truly on-substrate version needs a *cleanup/normalize-to-unit* circuit
  between hops (an attractor that re-discretizes the phasor) — that is the same magnitude-floor nonlinearity the
  composer already relies on. Scope: prove I-1-a for the flat (non-clause) path FIRST; the clause re-normalize is
  a bounded follow-on.

### Gap I-4 — make the cross-region interactions the DEFAULT merged loop (+ close the host grounding `M`)

- **Biology:** perception→episodic/relational memory (catalog **D.02** Eichenbaum-Cohen relational binding,
  perirhinal item stream → hippocampus → cortex; **D.01** episodic encode; **D.14** Tonegawa engram cells —
  "the neurons that fired ARE the memory"); the cognitive map binds what-is-where (**D.21**). Grounded/embodied
  semantics (concept codes ARE sensorimotor patterns) is the **D.02-supplemental / G.20** convergence-zone story
  (ATL hub-and-spoke; Patterson-Lambon Ralph — already cited in the project's generalization arc).
- **Option I-4-a [RECOMMENDED, cheap] — flip the three GO interactions on by default in `MergedNavConvAgent`.**
  Default-on `co_resident_rf` + `co_resident_perception` + the `COMMAND_GATE` route + the engram perception→memory
  route, so the *deployed* merged agent INTERACTS (not just co-locates). This is a default-flip + a regression pass
  (the conversational no-confab moat byte-test `test_nav_conv_merged_agent` 8/8 + `test_nav_conv_step2b_coresident`
  7/7 must still pass; the interactions are array-disjoint from the parser/composer by construction). **No new
  science.** Honest: this is "the merged DEFAULT now interacts", the cleanest win against I-4's literal
  "zero cross-synapses" charge.
- **Option I-4-b — replace the host grounding projection `M` with a LEARNED Hebbian crossmodal convergence
  (close I-4-resid / N-4).** The project ALREADY de-risked this: `_genfrontier_onsubstrate_convergence_derisk.py`
  (a structured-perception region + a concept region, population-Hebbian co-activation, held-out cat-acc 0.92 on
  SPIKES; NMDA lets the concept assembly fire) + the `co_resident_generalization` stack already wired in
  `nav_conv_merged_bridge` (`gen_perception→gen_concept` plastic convergence, trained-then-frozen). So the
  `composer.concepts[o] = M @ rate` host projection can be replaced by: percept fires `gen_perception` →
  the LEARNED convergence fires `gen_concept` → a fixed route grounds the composer code from the `gen_concept`
  spikes. **The mechanism is GO; this is a wiring/consolidation job, not new research.** Biology: ATL
  convergence zone (D.02-supplemental, Garagnani-Pulvermüller spiking precedent).
- **Option I-4-c — perception→memory via engram tags as the DEFAULT (sidesteps the rate↔phasor wall for RECALL).**
  Make `navigate_to_see_then_answer`'s engram-tag write the default perception→memory path: the perceived ensemble
  IS the memory (D.14), no phasor codebook, no host copy. Cheapest possible perception→memory interaction;
  recall-only (it cannot algebraically COMPOSE the percept — that's I-4-b / step-3).

### Gap I-7 — limbic→composer deep route (the "one self")

- **Biology:** dopamine gates entry into long-term memory (catalog **D.16** place-field stability requires D1/D5
  dopamine + late-LTP; Lisman-Grace hippocampal-VTA loop); D1 inverted-U sharpens PFC tuning (Vijayraghavan/
  Arnsten) — the read-side precision effect. Tonic-DA mood / neuromodulatory gain.
- **Option I-7-a [DONE — confirm + flip] — the read-side salience gate.** `_da_confidence_gate` is built +
  production-wired (opt-in). Action: confirm the GO + flip default-on where the shared limbic slice exists.
  Moat-safe by construction. Nearly free.
- **Option I-7-b [RECOMMENDED next, cheap] — wire + deploy-smoke the ENCODING hook.** `encoding_gain_fn` is
  de-risked GO (`_phaseB_dopamine_encoding_gain_derisk.py`): a salient/rewarded turn → higher DA → a higher-gain
  stored composite that reconstructs ABOVE the RF magnitude floor (`sim/bridge.py:5589`) under read damage where a
  neutral fact degrades below it. The ONE gap is the deployment smoke (read the shared `dopamine` at store time →
  `encoding_gain_fn`). Reuse-by-import; moat-safe (gain scales magnitude only, the cue-match abstention is
  unchanged).
- **Option I-7-c [deep, defer] — DA threads the RF resonate dynamics.** DA modulating the resonate λ/ω (gain on the
  complex update) — the *sketched* `sim/` edit. Higher-fidelity (neuromodulation of the substrate, not a read-side
  scalar) but unbuilt and needs a byte-reviewed additive default-off `sim/` change. Defer behind I-7-a/b.

---

## 3. CHEAPEST-FIRST DE-RISK per sub-gap (smallest probe that proves/refutes; numpy/CPU where possible)

- **I-5-a (synaptic parser→composer) — CPU probe, ~minutes.** On a small merged-ish bridge with the `rf`
  composer + the parser, wire ONE role's `role_route_agent` gate (`couple_gate_to_indices` over the agent
  sub-block), present "dog go north", and compare the STORED composite's agent-unbind to the Python-dict path:
  **GO bar = the agent role recovers "dog" via the gated route == the dict path** (and the gate opens only on
  parser firing, vanishes when the route is lesioned). This mirrors `_step2_gated_route_probe.py` /
  `_step2_synaptic_holdopen_validate.py` (the precedents). CPU is fine — static drive + short settle.
- **I-1-a (register→register handoff) — CPU probe, ~minutes.** Build a 2-op chain (bind→unbind) on the masked RF
  slice; replace the `to_host(rf_read_phases)+re-kick` between them with a fixed identity-phasor route source→dest;
  **GO bar = the unbind result is byte-identical (atol 1e-9) to the host-round-trip path** on the flat (non-clause)
  query. If it drifts, that localizes exactly the magnitude-decay the clause path warns about → I-1-c.
- **I-4-a (default-flip) — regression pass, no new probe.** Build `MergedNavConvAgent` with the interactions
  default-on; **GO bar = `test_nav_conv_merged_agent` 8/8 + `test_nav_conv_step2b_coresident` 7/7 still pass
  (moat 0-FA, nav score byte-identical Δ=0), and `spoken_instruction_nav` + `navigate_to_compose_then_answer`
  still GO under the default build.** (GPU for the agent path.)
- **I-4-b (learned grounding replaces `M`) — reuse the GO de-risk + one wiring probe, CPU/GPU.** The convergence
  is already GO; the new probe is "perceived object fires `gen_perception` → `gen_concept` spikes → a fixed route
  grounds `composer.concepts[o]`" reproduces the host-`M` compose; **GO bar = held-out compose >> memorization
  floor with the grounding read off `gen_concept` SPIKES (not the host matvec), lesion (sever the convergence)
  collapses it, moat 0-FA.**
- **I-7-b (encoding hook deploy) — CPU/GPU smoke.** Drive the shared SNc high vs tonic at store time → `encoding_gain_fn`
  reads `get_concentration("dopamine")`; **GO bar = a high-DA fact survives read damage that sinks a tonic-DA fact,
  moat 0-FA at both DA levels, lesion (DA pinned baseline) abolishes the differential** (the de-risk's exact bars).

---

## 4. ANTI-CHEAT CONTROLS (every sub-gap inherits these — the integration must be SYNAPTIC)

1. **Provenance assertions — no host quantity smuggled across regions.** Grep the integrated path for any
   `cp_external_input_current[<region B indices>] = f(to_host(<region A>))` or `composer.concepts[o] = host_fn(...)`.
   The ONLY legitimate host writes are (a) the environment presenting instruction TEXT to `language_input` and
   rendering the object into `cortex_it` (sensory render), and (b) the body moving on the motor/sel winner. The
   cross-region coupling must be a **0/1 gate STATE from firing** (`couple_gate_to_*`) or a **fixed synaptic route**,
   never a value copy. (`spoken_instruction_nav.provenance_facts` is the template assertion;
   `navigate_to_compose_then_answer` already asserts `composer.concepts[o] == grounded_phases(rate, proj)` — note
   that current assertion CONFIRMS the host `M` round-trip, so I-4-b must REPLACE that provenance with a
   spikes-only one.)
2. **Lesion = the interaction vanishes.** Cut the cross-region route (zero its synapses / never open the gate):
   the behavior must collapse to chance / the recall must fail. The route must be NECESSARY (the primary
   load-bearing test; the same control that resolves the nav-reward residual). All three GO milestones already
   carry this (LESION condition).
3. **Both-brains-required (isolated controls fail).** Isolated-nav (route lesioned) and isolated-conv (no body)
   must each fail; only the coupled brain solves it. (`spoken_instruction_nav` ISOLATED-NAV / ISOLATED-CONV.)
4. **No-confab moat preserved — the HARD invariant.** No abstention the host returned may become a false-accept
   on any integrated path or at any DA level. The DA hooks are moat-safe by construction (a higher gate only
   TIGHTENS abstention). Every integration regression re-runs the moat block (the three `is None`/"unknown"
   assertions). Per `feedback_moat_not_hard_lossy_memory_ok` the moat is a kept-where-free plus, never traded to
   make a number look better.
5. **Scramble (instruction-following genuineness).** For language→action, permuting word→direction must regress
   accuracy-vs-commanded (the agent follows what it COMPREHENDS). (`spoken_instruction_nav` SCRAMBLE.)

---

## 5. RECOMMENDED SEQUENCE (+ parallelism + `sim/`-edit flags)

**Framing for the controller:** order by *cheapest-and-bankable* first (this whole arc is consolidation +
residual-closure, NOT new science — most of the leverage is default-flips and porting existing GO mechanisms).

1. **I-4-a — flip the merged DEFAULT to INTERACT [cheapest, highest "one brain" symbolic value, NO `sim/` edit].**
   Default-on the three GO interactions in `MergedNavConvAgent`; regression-gate the moat + nav byte-identity.
   This alone retires I-4's literal "zero cross-synapses" charge for the deployed agent. *Reuse-by-import.*
2. **I-7-a/b — limbic→composer, both sides [cheap, NO `sim/` edit].** Confirm + default-on the read-side gate
   (I-7-a, DONE); wire + deploy-smoke the encoding hook (I-7-b). The limbic core then reaches the cortex on
   BOTH halves on the deployed agent. *Reuse-by-import.* **Parallelizes with #1** (disjoint code: the DA hooks
   are in the agent's read/store, the I-4-a flip is in the builder kwargs).
3. **I-5-a — synaptic parser→composer route [cheap-ish, NO `sim/` edit expected].** CPU probe first (§3), then
   port `hear_synaptic` onto the merge for the `rf` composer. *Reuse-by-import* (the gate primitives + the
   `hear_synaptic`/`_op_synaptic` bodies are public/borrowable). **Parallelizes with #1/#2** (read-only CPU probe;
   the build touches the agent's `hear`, disjoint from the DA hooks).
4. **I-1-a — op-handoff-as-spikes (flat path) [cheap-ish probe; build MAY need a small `sim/` edit].** CPU probe
   the register→register identity route (§3). If byte-identical → wire it (likely *reuse-by-import* via a fixed
   route + the existing megakernel). If a fused multi-op megakernel is needed (I-1-b) → **`sim/` edit, additive,
   default-off, byte-review** (it also folds the O-1/O-6 perf levers, so coordinate with the 2A perf chain).
   Sequential after #3 (the I-5 synaptic route changes which registers feed the first op).
5. **I-4-b — learned grounding replaces host `M` (close I-4-resid / N-4) [reuse the GO convergence; wiring job].**
   After #1, swap `navigate_to_compose_then_answer`'s host `composer.concepts[o] = M @ rate` for the LEARNED
   `gen_perception→gen_concept→ground` spiking convergence (already wired + de-risked). *Reuse-by-import.*
   **Parallelizes with #3/#4** (it touches the perception-grounding step, disjoint from the composer op-handoff).
6. **I-1-c / I-7-c — the deep residuals [defer; `sim/` edits].** The clause re-normalize circuit (I-1-c) and the
   DA-threads-RF-dynamics (I-7-c) are the bounded deep follow-ons; **each is an additive default-off `sim/` edit,
   byte-reviewed**, taken only after the cheap wins land.

**Parallelization map:** #1 ∥ #2 ∥ #3-probe ∥ #4-probe all run concurrently (read-only probes + disjoint
default-flips). Then #4-build and #5 parallelize; #3-build precedes #4-build. The deep `sim/` residuals (#6) are
last and serialized behind the byte-review.

**`sim/`-edit flags:** #1, #2, #3, #5 = **reuse-by-import, NO `sim/` edit** (all primitives public). #4 build =
**possible additive default-off `sim/` edit** (only if the fused multi-op megakernel I-1-b is chosen over the
identity-route I-1-a). #6 = **additive default-off `sim/` edits, byte-reviewed** (clause re-normalize; RF-dynamics
DA threading). Per `feedback_dont_gate_on_approval`, the reuse-by-import items proceed directly; the `sim/` edits
get the standing byte-level diff review.

**Net recommendation:** lead with **I-4-a (flip the merged default to interact)** + **I-7-b (encoding hook)** +
the **I-5-a / I-1-a CPU probes** as four concurrent cheap tracks; they bank the bulk of the "real one brain" claim
with no `sim/` edit. I-4-b (kill the host grounding `M`) and the deep I-1-c/I-7-c residuals follow. The arc is
mostly consolidation of already-GO mechanisms — the genuine *new* work is small (the I-5 synaptic route on the
`rf` composer, the I-1 register handoff probe), which is why this is the right next high-leverage move and not a
months-scale build.
