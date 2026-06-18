# Capability frontier → "compete with a basic small LLM" — scoping (2026-06-17)

> **Read-only deep-research + catalog scoping. No code edited (this doc is the only write).** Produced before any
> build per the standing "deep research + catalog review FIRST at new directions" directive (CLAUDE.md;
> `feedback_deep_research_at_roadblocks`). The controller should trust-but-verify the load-bearing claims flagged
> **[VERIFY]** inline, then push + present the recommendation before building.
>
> **Direction being scoped:** the biologization of the conversational pipeline is now largely complete (the
> binder's read-out learning was just made brain-based — real synaptic plasticity on the bridge + a neuralisable
> per-output error, CYCLE 158–159). The question is now a **capability** one, not a biologization one: **given the
> current validated stack, what is the highest-leverage next CAPABILITY to add to close the gap to a basic small
> LLM (Phi-3-mini / Llama-3.2-3B class), the biology-faithful way, BEFORE renting cloud compute for scaling?**
>
> **This doc supersedes the framing of `2026-06-17-conversational-architecture-to-basic-LLM-scoping.md` and
> `2026-06-17-productive-syntax-scoping.md`** by re-scoring against what has since been BUILT (those docs'
> top recommendations — the theta-gamma ordered-WM engine, and the productive-syntax easy half — are now GO; see
> §1.0). It does not re-litigate them; it asks what comes next.

---

## 0. Constraints (the bars every option is judged against)

- **BRAIN-BASED-ONLY** (`feedback_brain_based_only_standard`): only neurons / synapses / their communication count
  as the brain doing the work. Host code is legitimate ONLY for the environment (world state, sensory render) and
  the body (acting on motor output). A host formula (argmax, distance, an f-string word-orderer, a Python sort) is
  a documented SHORTCUT; an honest negative under strict biology is itself a deliverable. Every option below is
  annotated with any host shortcut it would introduce.
- **The no-confab "moat" must NEVER be weakened by a proposal.** The agent ABSTAINS when it has no matching fact
  rather than hallucinating. (It is a PLUS not a hard gate per `feedback_moat_not_hard_lossy_memory_ok`, but no
  proposal here trades it away — every de-risk asserts it intact.)
- **Point-neuron substrate is the default** (Izhikevich / HH / AdEx + resonate-and-fire). The two-compartment
  **dendritic neuron is on the table** (owner un-benched it 2026-06-17; D2 Phase 0–2 built a two-compartment
  neuron + learned graded cortex; Phase 3 pending) — admissible *only if* a capability genuinely needs it AND it
  is the obvious unlocker. Prefer the cheapest point-neuron-feasible de-risk first.
- **Reuse-by-import strongly preferred** over edits to protected `sim/` modules.
- **6-seed validation** for any variable effect (`feedback_6seed_validation`); CuPy for decisive/heavy runs, numpy
  only for the cheap-first probe (`feedback_gpu_not_numpy`).
- **North star** (`project_actual_goal_artificial_life_brain_analogue`): a biology-faithful artificial-life brain
  with biology-translatable insight. "Compete with a basic LLM" is the *capability target*, reached by mechanism,
  not by abandoning biology.

---

## 1. DIAGNOSIS — the capability gap (excluding raw scale)

### 1.0. What has been BUILT since the two prior scoping docs (so we don't re-recommend solved work)

The conversational stack moved fast over CYCLEs 135–159. The capabilities the two prior scoping docs ranked #1
are now **GO / production**:

| Capability | Prior-doc status | NOW (2026-06-17) |
|---|---|---|
| Ordered multi-item WM (theta-gamma gamma-slots) | Option 1, the "biggest lever," to build | ✅ **GO / production** — `ordered_position_wm.py` `OrderedPositionWM`, ordered recall **1.000 to the full 7-slot Lisman-Idiart span** at D=256 (CYCLE 135/140) |
| Multi-referent disambiguation | mapped NEGATIVE (plain recency loop) | ✅ **resolved BY SLOT** — read slot *k* = `unbind(C, pos_k)`; winner is "which slot you read", flips deterministically with discourse order (CYCLE 136, `test_multi_turn_ordered_wm` 31/31). The NEGATIVE was for the rate-attractor *set*; the order-encoded WM removes the competition the WTA was needed for. |
| Multi-sentence fluency / discourse | ⚠️ single utterances | ✅ **GO 6/6** — `MultiTurnAgentV2.narrate()` (ordered emission + cross-sentence pronoun coherence; CYCLE 137/138/142) |
| Productive syntax — **easy half** (novel word-order frames) | flagged, unbuilt | ✅ **GO 6/6 on spikes** — `_phaseB_learned_multiframe_word_order_derisk.py`: a LEARNED (not hardcoded) per-frame primacy gradient produces a non-native verb-initial frame the agent was never templated with; permuted-frame + lesion collapse to chance, moat 1.000 (CYCLE 143) |
| Reconsolidation (correct a fact through dialogue) | unbuilt | ✅ **GO 6/6 + production** — `RFPhasorComposer.update_on_mismatch` + `MultiTurnAgentV2.correct()` (CYCLE 147/148) |
| Multi-hop reasoning, anaphora, learn-from-stream, neural word order | various | ✅ all GO/production |

**So the four "real gaps" the architecture-scoping doc identified are now THREE-quarters closed.** The remaining
frontier is no longer "no ordered structure" — it is the **hard half of productive syntax** plus two adjacent
capability KINDS a basic LLM has that this stack still lacks. The diagnosis below is the *current* gap.

### 1.1. What a basic small LLM does that THIS stack does not yet do (excluding scale)

Excluding raw scale (vocabulary size / world knowledge — the deferred cloud axis), the genuinely-missing
capability KINDS, decomposed:

| Capability KIND | Basic-LLM behaviour | Project status today | Precise gap | Collides with a known wall? |
|---|---|---|---|---|
| **A. Productive syntax — HARD half (recursion depth + agreement)** | arbitrary center-embedding; long-range subject-verb agreement | ⚠️ depth-2 recursion with FLAT inner args only (one template); **attributed** deep inner args degrade; no number/gender agreement features at all | no mechanism for **arbitrary/learned** recursion depth, nor to bind a **non-adjacent dependency** (verb agrees with subject across a clause) | **DIRECT** — recursion → the **nested-composition / SNR wall** (`2026-06-02-full-320…hierarchical null`); agreement → the **multi-attribute BUNDLING NEGATIVE** (`2026-06-16-onsubstrate-learned-binder…bundling-NEGATIVE`). Both reduce to the *same* superposition-inverse / multiplicative-binding op the project has 4× found to be the point-neuron limit. |
| **B. Constituency as a slot-filler in an arbitrary position** | groups words into nested constituents, uses a phrase as a unit anywhere | ⚠️ flat SVO + 1 templated embedded clause (patient-slot only) | no mechanism to **form a constituent on the fly** ("[the big dog] chased [the small cat]") and place it as a unit in an *arbitrary* slot | **MILD–MODERATE** — single-attribute bind + resonator decode exist; *arbitrary-position* constituency + the on-the-fly *attributed* noun phrase ("big dog") is the new part (the attribute bind is the 1-attribute regime, GO on spikes; the bundling of >1 attribute is the wall). |
| **C. Instruction-following over a COGNITIVE operation** | "list X", "compare X and Y", "tell me about X then Y" — language selects an internal op | ⚠️ language→**body** is GO (`spoken_instruction_nav` steers nav); language→**cognitive-operation** absent | no mechanism to map an instruction utterance → which internal query/generation routine runs (a learned cue→operation router, the dlPFC frame-selector generalized from "which sentence frame" to "which cognitive act") | **MILD** — reuses the validated frame-SELECTION mechanism (CYCLE 143) + the dlPFC Control + the existing query/narrate routines; no binding wall. |
| **D. Flexible / open question-answering** | answers wh-questions of many shapes, "why/how", compositional queries | ⚠️ fixed who/what/yes-no + multi-hop chase | no mechanism for **novel query shapes** (e.g. "what did X do to Y and then Z?") beyond the templated set; queries are hand-specified frames | **MILD** — query frames are the comprehension-side mirror of the (now-learned) generation frames; learning a query frame inventory is the same CQ/role-slot machinery applied to parsing. |
| **E. In-context (few-shot) adaptation** | conditions output on a few in-context exemplars without weight training | ❌ absent | no mechanism to bias the current response from a handful of just-seen examples held in WM (not in weights) | **MODERATE** — the ordered-WM now holds exemplars; the missing piece is a *fast* WM→output bias (a one-shot prior over the response), which is a WM-readout mechanism, not a binding wall. |
| **F. Fluent open-ended surface generation** | paragraphs of novel free text | ❌ **known terminal WALL** (out of scope) | ~360M+ params + trillions of tokens; from-scratch spiking LM overfits at scale (SpikeGPT). **Documented terminal; do not re-attempt on the brain-based path.** | N/A — this is the deferred-cloud / engineering-hybrid axis, NOT a brain-based capability gap. |

**The structural theme of the genuinely-missing gaps (A, B, D):** they are all **productive syntax** — the
*structures themselves* learned and generative, handling sentence types never given as a template. The easy half
(novel word-order frames over the same 3 role slots) is GO; the **hard half is where the two named walls bite**,
and it splits cleanly into:
- **A2 / D-deep — arbitrary recursion depth** (nested composites): the nested-composition SNR wall.
- **A1 — non-adjacent agreement / long-range dependency binding**: the multi-attribute bundling NEGATIVE.

Gaps **C (instruction→operation)** and **E (in-context)** are *not* blocked by a binding wall — they are
point-neuron-feasible reuses of the now-validated frame-selection + ordered-WM machinery, and so are the cheaper,
lower-risk capability adds.

⇒ The next-capability decision is a genuine FORK between (i) the **deepest** lever — the hard half of syntax,
which collides with the dendritic-multiplication question, and (ii) the **cheapest high-value** levers —
instruction-following and richer query/constituent frames, which extend the easy-half win with no new substrate.
The ranking below leads with the option that best balances leverage against risk given the owner's stated
"compete with a basic LLM" target.

---

## 2. RANKED options for the next capability (biggest balanced lever first)

Bars: closes the most *LLM-gap-relevant* capability · direct catalog + literature grounding ·
point-neuron-feasible (prefer) · reuses validated machinery · minimal/no `sim/` edit · moat intact.

### Option 1 (RECOMMENDED) — Productive syntax, breadth pass: promote learned multi-frame word order into the agent + add a learned QUERY-frame inventory and CONSTITUENT-as-slot, all over the validated role-slot / CQ substrate

- **Mechanism (one paragraph):** Take the CYCLE-143 GO (a learned per-frame primacy gradient produces novel
  word-order frames on spikes) from a *de-risk* to a *production agent capability*, and broaden it from generation
  to comprehension. Three composable bricks, all on validated machinery: **(1a)** wire a learned multi-frame
  `render` into `BrainConversationalAgent` (the renderer selects a frame from an utterance-type cue via the dlPFC
  Control, applies that frame's learned gradient through the existing `NeuralSerialOrderRenderer` competitive-
  queuing read-out) behind a default-off flag — this is the "GO routes to" step the CYCLE-143 finding names and
  has NOT been done yet (`brain_conversational_agent.py` has no multi-frame render). **(1b)** mirror the learned
  frame inventory on the **comprehension** side: `BridgeParser` already learns a (word-position × voice) → role
  map; extend it to ≥2 learned frames so the agent *understands* non-canonical orders (closing gap **D**, flexible
  Q&A of novel shapes — a query is just a frame whose answer slot is unfilled). **(1c)** the smallest
  **constituency** step (gap **B**): allow a 1-attribute noun phrase ("big dog") as a slot-filler in an arbitrary
  role position — the attribute bind is the *single-attribute* regime already GO on real LIF (held-out 0.833 =
  100% of numpy, `2026-06-16-…single-attr-GO`), and the resonator decode already reads a clause-slot; the new part
  is placing it in an arbitrary slot, not the bind itself. **This option deliberately stays inside the
  single-attribute / role-filler regime and never touches multi-attribute bundling** — so it delivers genuinely
  more LLM-like syntactic flexibility WITHOUT meeting the wall.
- **Biological grounding:** catalog **G.07** (pre-SMA internally-generated *abstract* sequences) + **H.19**
  (premotor competitive queuing / parallel→serial order; Grossberg 1978, Bullock-Rhodes 2003) for the order
  read-out; **G.12** (Broca's grammatical processing — the catalog's own behavioral validation is the
  syntactic-complexity dissociation "the girl that the boy is chasing is tall") for frame-conditioned
  comprehension; **G.10/G.11** (language as a hierarchical symbolic system / dual-stream) for the
  comprehension↔production split; **N.15** (theta-gamma multiplexed slots; Lisman-Idiart 1995) = the ordered-WM
  the role slots live in. The Assembly-Calculus line (Mitropolsky-Papadimitriou **TACL 2021** parser; Dabagia
  **ALT 2024** sequences-of-assemblies) is the field's account that learned ordered syntactic structure is
  **point-neuron-realizable** — adopt the *principle* (`feedback_check_existing_sims_first`) without porting the
  whole NEMO substrate.
- **Which gaps it closes:** the production wiring of the easy half (the agent actually *uses* learned frames), gap
  **B** (constituent-as-slot, the 1-attribute case), and gap **D** (flexible Q&A via learned query frames). It
  does NOT attempt the hard half (recursion depth / agreement) — that is explicitly parked for Option 3.
- **Point-neuron feasibility / difficulty:** **HIGH feasibility, LOW–MEDIUM difficulty.** Everything reuses
  validated point-neuron pieces (CQ rate-ranking, ordered-WM gamma-slots on the RF substrate, dlPFC Control,
  single-attribute bind on real LIF, resonator decode). The only new learning is Hebbian (per-frame primacy
  gradient; cue→frame and cue→query-frame maps) — standard rate-Hebbian maps. **dt-bound caveat** (CLAUDE.md
  one-bridge step-3): rate-ranking ties when slots are equidistant; run the WM/order engine at its native dt=0.5
  and keep frame inventories small (2–4).
- **`sim/` edit needed?** **NO** — `NeuralSerialOrderRenderer`, `OrderedPositionWM`, `BridgeParser`, the dlPFC
  `SpikingSpreadingController`, and the single-attribute binder are all reuse-by-import; the new maps are
  runner-side Hebbian.
- **Host-shortcut check:** none introduced if the frame-selection, the order read-out, and the constituent bind
  all run on neurons (CYCLE-143 already proved the order is neural; the parser is already Hebbian-on-bridge). The
  one thing to keep honest: the final `" ".join` of the neurally-ordered, neurally-spelled words is the **body**
  emitting output (legitimate per the standard), exactly as the current `NeuralSerialOrderRenderer` documents.
- **Leverage:** **HIGH** — directly converts the project's biggest validated-but-unshipped result into agent
  capability and broadens it to the comprehension + constituency directions a basic LLM has, at low risk.

### Option 2 — Instruction-following over a cognitive operation (language → which internal routine runs)

- **Mechanism:** Generalize the validated **frame-SELECTION** mechanism (CYCLE 143: a context cue → which sentence
  frame) and `spoken_instruction_nav`'s **language→action** route (a parser firing opens a `command_route` gate →
  the learned word→action map) from "which sentence frame / which motor act" to "**which cognitive operation**":
  a learned Hebbian cue→operation router maps an instruction utterance ("list what X did", "tell me about X then
  Y", "compare X and Y") onto the existing query / `narrate` / `reason_chain` routines, gated by transmission
  gates and dispatched by the dlPFC Control. The instruction is *comprehended* by the parser; the *selection* of
  the internal routine is the same spiking WTA the frame-selector uses; the routine itself is already validated.
- **Biological grounding:** catalog **G.08** (PFC working memory / executive control — task-set selection) +
  **G.12** (Broca's: maps stored word-forms to operations) + the project's validated language→action route
  (`spoken_instruction_nav`, 6-seed GO, `2026-06-10-spoken-instruction-nav-GO.md`) as the precedent that a
  parser-firing can gate a downstream cognitive route. dlPFC rule/task-set selection (Miller-Cohen 2001 PFC
  guided-activity; catalog G.08) is the executive-control account.
- **Which gaps it closes:** gap **C** (instruction-following over a cognitive op) — a capability KIND a basic LLM
  has and this stack lacks entirely on the conversational side. It also makes the multi-sentence `narrate` and
  multi-hop `reason_chain` *invokable by language* rather than by host calls (a quiet BRAIN-BASED-ONLY upgrade:
  today the host decides which routine to run; this makes the *brain* decide from the instruction).
- **Point-neuron feasibility / difficulty:** **HIGH feasibility, LOW difficulty** — pure reuse of the
  frame-selector WTA + transmission-gate routing + dlPFC Control; the routines exist. No binding wall.
- **`sim/` edit needed?** **NO** — transmission gates (`set_transmission_gate`, `couple_gate_to_pool`) +
  the spiking WTA + dlPFC Control are reuse-by-import.
- **Host-shortcut check:** this option's *whole point* is to REMOVE a host shortcut (today the host chooses which
  query/narrate routine to call; after this the parser+dlPFC choose from the instruction). Keep honest: the
  routine's internal logic stays as-is (already neural for query/narrate); only the *dispatch* moves onto neurons.
- **Leverage:** **MEDIUM-HIGH** — cheap, no wall, and adds a visibly LLM-like behaviour ("do X for me") while
  improving biology-faithfulness. Strong candidate to run *in parallel* with or *right after* Option 1 (it depends
  on Option 1's frame-selector being agent-resident).

### Option 3 (the DEEP lever — a genuine fork) — Productive syntax HARD half: arbitrary recursion depth + non-adjacent agreement

- **Mechanism:** The two hard sub-capabilities reduce to the SAME operation (apply a role-specific *multiplicative
  inverse* to a superposition / deeper bundle). There are two routes past it, and choosing between them is the
  fork:
  - **(3a) Assembly-Calculus disinhibition recursion (point-neuron, control-flow):** the 2022 spiking
    center-embedding result (Mitropolsky-Ejaz-Shi-Yannakakis-Papadimitriou, arXiv:2206.13217) handles
    center-embedded recursion "exclusively through the spiking of neurons" via a **disinhibition / blocking
    control** that re-enters the role areas — a new characterization of context-free languages, **no software
    stack and no deeper binding**. This is point-neuron-feasible *in principle* (random areas + k-cap WTA +
    Hebbian = the project's sparse-pool + FS-PV/MSN lateral-inhibition motifs), and the project already has the
    transmission-gate disinhibition primitive. **The risk is scale**: the published parser uses n=10⁵–10⁶
    neurons/area; the project's pools are ~2000, and it would be a *parallel* assembly-substrate build (k-cap
    assemblies, not the deployed phasor VSA), not an extension of the deployed composer.
  - **(3b) Dendritic multiplicative binding (the un-benched D2 substrate):** use the two-compartment dendritic
    neuron to realize the multiplication / superposition-inverse that point neurons provably cannot do unaided —
    the candidate unlocker for nested-composite binding *and* the long-range agreement bind, per the recurring
    Mikulasch-Priesemann point-neuron limit. CYCLE 144 found the multiplication primitive *already exists* on the
    bridge as a guarded/inert kernel (`fused_coincidence_plateau`, `sim/kernels.py:253`), so the build is
    **additive wiring (weeks), not a new NeuronModel (months)** — but CYCLE 146's A/B found a fixed self-inverse
    role already bundles at 0.993 (= the production composer), so a *learned* dendritic bundler is only justified
    if it buys **generalization across similar concepts** (a different axis, already carried by the PPMI-cortex),
    not raw bundling accuracy.
- **Biological grounding:** Mitropolsky-Papadimitriou Assembly-Calculus line (TACL 2021 / arXiv:2206.13217 /
  ALT 2024); Pulvermüller-Knoblauch 2009 sequence-detectors/DCNAs; catalog **G.10/G.12/G.13** (hierarchical syntax
  / Broca / Wernicke), **G.07** (sequences); for (3b) the Mikulasch-Priesemann dendritic-prediction-error framing
  + the project's own D2 two-compartment arc.
- **Which gaps it closes:** gap **A** (recursion depth + agreement) — *all* of productive syntax in principle, the
  deepest capability gain toward a basic LLM.
- **Point-neuron feasibility / difficulty:** **(3a) MEDIUM feasibility / HIGH difficulty** (point-neuron in
  principle, unproven at this project's 100× smaller scale, substantial new runner machinery, parallel substrate).
  **(3b) the dendritic option, MEDIUM-HIGH difficulty** (weeks-scale additive wiring on the existing kernel; the
  open question is whether the LEARNED version buys anything over the fixed-algebra 0.993).
- **`sim/` edit needed?** (3a) probably none for the k-cap mechanics (sparse pools + WTA + Hebbian exist) but
  substantial new runner machinery (area graph + projection scheduler + disinhibition recursion control). (3b) the
  multiplication kernel exists but is guarded/inert — enabling its conversational use is an additive, byte-reviewed
  `sim/`-adjacent change.
- **Host-shortcut check:** both routes are explicitly brain-based; (3a) is the field's spiking-only recursion
  result, (3b) is a dendritic neural op. No host shortcut introduced.
- **Leverage:** **HIGHEST capability ceiling, HIGHEST risk.** This is the genuine FORK the CYCLE-143 finding and
  the productive-syntax scoping both name as "the *next* scoping question — AFTER the easy half localizes the hard
  half as the residual blocker." **Recommend it be sequenced AFTER Option 1**, because Option 1's breadth pass
  will *localize* whether deep recursion / agreement is actually the felt blocker for "compete with a basic LLM"
  or whether the easy half + instruction-following already close the perceived gap. Do NOT commit the deep build
  (especially the dendritic one) before that localization — its own pre-registered build gate (CYCLE 146) is not
  yet met.

### Option 4 (lower priority for THIS goal; a documented escape hatch) — In-context few-shot adaptation via fast WM-conditioned output bias

- **Mechanism:** Use the now-validated ordered-WM to hold a few just-seen exemplars and bias the current
  response from them WITHOUT weight training — a one-shot prior over the output read out of the WM (gain-bias on
  the response candidates from the held exemplars). This is a WM-readout mechanism, not a binding wall.
- **Biological grounding:** catalog **G.08** (PFC WM as the substrate for task-relevant transient bias) + **N.15**
  (theta-gamma slots holding the exemplars) + biased-competition (Desimone-Duncan 1995) for the exemplar→response
  bias.
- **Which gaps it closes:** gap **E** (few-shot in-context adaptation) — a real LLM-like capability, but the one
  the owner is least likely to be asking for first (it is "prompt-style" behaviour, less central than syntactic
  flexibility / instruction-following to "converse with the artificial-life agent").
- **Point-neuron feasibility / difficulty:** **MEDIUM** — the ordered-WM holds the exemplars; the open part is the
  fast bias read-out, which is new but additive.
- **`sim/` edit needed?** Likely NO.
- **Why ranked 4th:** it is a genuine LLM capability but lower on the owner's stated "converse with the agent"
  priority than syntactic flexibility (Option 1) and instruction-following (Option 2); sequence it after those.

**Ranking rationale.** **Option 1** is the highest *balanced* lever — it ships the project's biggest
validated-but-unshipped result (learned multi-frame syntax) into the agent and broadens it to the comprehension +
constituency directions a basic LLM has, on the validated point-neuron substrate, with no `sim/` edit and no
binding wall. **Option 2** (instruction-following) is the cheapest standalone LLM-flavoured win, has no wall, and
*improves* biology-faithfulness (moves routine-dispatch onto neurons) — run it in parallel with or immediately
after Option 1. **Option 3** (the hard half of syntax) is the deepest capability and the genuine fork (AC
disinhibition vs. dendritic multiplication), but it is correctly sequenced *after* Option 1 localizes whether deep
recursion / agreement is the felt blocker — and the dendritic commitment has a pre-registered build gate not yet
met. **Option 4** (in-context) is a real but lower-priority LLM capability for the stated goal.

---

## 3. REUSABLE MACHINERY for the top recommendation (Option 1)

| Piece | File (anchor) | Contributes to Option 1 |
|---|---|---|
| Competitive-queuing serial-order generator | `research/runners/neural_serial_order_renderer.py` (`NeuralSerialOrderRenderer.order/render`) + `research/runners/_phaseB_serial_order_spiking_derisk.py` (`build_pool_bridge`, `pool_rates`, `PRIMACY_pA`) | the per-frame word-ORDER read-out (graded current → spiking-rate ranking = order) |
| Learned multi-frame de-risk (the GO to promote) | `research/runners/_phaseB_learned_multiframe_word_order_derisk.py` (`FRAMES`, the per-frame Hebbian gradient + cue→frame WTA + permuted/lesion/moat controls) | the validated learned-frame mechanism + its anti-cheat harness to reuse verbatim |
| Order-encoded WM (gamma-slots = grammatical roles) | `research/runners/ordered_position_wm.py` (`OrderedPositionWM.encode_sequence/read_slot/calibrate_threshold`) | role slots on the RF substrate; the familiarity moat |
| The FHRR composer (bind/unbind/bundle/cleanup + moat) | `research/runners/rf_phasor_composer.py` (`RFPhasorComposer`, `_bind/_unbind_phases/_bundle/_cleanup`, `query_patient/query_agent/ask_yes_no`) | the single-attribute constituent bind + the fact store the query frames read |
| Comprehension-side parser (extend to ≥2 learned frames) | `research/runners/brain_conversational_agent.py` (`BridgeParser`, the Hebbian (position×voice)→role map) | learned QUERY-frame comprehension (gap D) |
| dlPFC content-selection / spreading Control | `research/runners/content_selection_spiking.py` (`SpikingSpreadingController`) | frame SELECTION (which frame / which query frame from a context cue) |
| The agent to wire `render` into | `research/runners/brain_conversational_agent.py` (`BrainConversationalAgent`, `describe`, `enable_neural_render`) + `research/runners/multi_turn_agent_v2.py` (`MultiTurnAgentV2.narrate/describe/correct`) | the production integration target (currently uses fixed-SVO `describe`; no multi-frame render yet) |
| Single-attribute learned binder on real LIF (constituent attribute) | `research/runners/_phaseB_onbridge_bind_nonlinearity_derisk.py` (held-out 0.833 = 100% of numpy) | the "big dog" attribute bind for constituency (gap B), stays in the single-attribute regime |
| Resonator decoder (constituent / nested decode) | `research/runners/nested_composition_agent.py` (resonator probes) | reading a constituent-slot back as a unit |
| Frozen verdict / anti-cheat instruments | `research/runners/song_g1_core.py` (`score_order`, `permuted_order_controls`) | the order-scoring + permuted-order controls to reuse verbatim |

---

## 4. RECOMMENDED #1 + its cheap-first de-risk

**Recommend Option 1, starting with brick 1b (learned QUERY-frame comprehension)** as the cheapest decisive probe
— because brick 1a (promote the learned *generation* render) is a near-mechanical integration of an already-GO
result (lower scientific risk, do it as the production follow-on), whereas the *open question* that gates Option
1's claim to "more LLM-like syntactic flexibility" is whether the agent can **comprehend** a non-canonical frame
(answer a question posed in a learned, non-native order) — the generation side is proven; the comprehension side
is the untested mirror.

**Cheap-first de-risk (CPU / numpy, reuse-by-import, no `sim/` edit, minutes):**

- **Setup.** Reuse the CYCLE-143 harness (`_phaseB_learned_multiframe_word_order_derisk.py`) + `BridgeParser`'s
  learned (position×voice)→role map. Train the parser on a handful of frame-tagged sentences in **two** frames
  over the SAME 3 role slots — native SVO and a non-native order (e.g. VSO "ran dog north", or an OSV question
  order). Store the resulting SVO facts in the `RFPhasorComposer`. **Test on HELD-OUT fillers:** pose a wh-question
  ("who/what …") in EACH frame about a stored fact the parser was never trained on, and check the agent assigns
  the correct roles → returns the correct answer, AND the dlPFC frame-selector routes the comprehension to the
  right frame.
- **Metric.** For a held-out fact queried in frame Fᵢ: (i) role-assignment accuracy (the parser maps the
  non-canonical order to the right roles), (ii) answer correctness (the composer returns the stored patient/agent),
  (iii) frame-selection accuracy (cue → correct comprehension frame).
- **Pre-registered GATE (FROZEN before data; ≥6 seeds; FRACTIONAL ≥5/6 bar per `feedback_6seed_validation`):**
  - **GO:** held-out answer correctness in the **non-native learned frame** ≥ 0.90 on ≥5/6 seeds, AND
    frame-selection ≥ 0.90 on ≥5/6 seeds, AND the **permuted-frame control collapses to chance** (shuffle
    frame→role-map → role assignment is random), AND the **lesion collapses it** (remove the learned frame map →
    falls back to the single native frame / chance on the non-native), AND **native SVO comprehension
    un-regressed**, AND the **no-confab moat intact** (a question about an unstored subject still ABSTAINS — `is
    None` — in both frames). ⇒ promote: wire learned multi-frame render + parse into the agent (default-off),
    escalate to a GPU 6-seed gate.
  - **BOUNDARY:** the non-native frame is comprehensible in isolation but frame-SELECTION is unreliable (the dlPFC
    can't route cue→comprehension-frame), OR it works for 2 frames but a 3rd interferes — a real partial result
    localizing selection vs. capacity as the next sub-problem.
  - **NEGATIVE:** the learned non-native frame is no better than chance / no better than forcing the native frame
    (the parser can't learn a *second* role map, only the hardcoded one), OR the permuted-frame control does NOT
    collapse (the "frame" is reading a fixed structural bias, not a learned map). ⇒ record it (a biology-
    translatable negative about learnable *comprehension* frames) and reconsider — possibly the AC projection
    parser (Option 3a) for order-as-projection-sequence.
- **Anti-cheat controls (mandatory — a "success" without all of these is an artifact):**
  1. **Held-out fillers never trained** — the frame maps are learned on example sentences; comprehension tested on
     novel (agent, action, patient) tuples (proves the frame generalizes over ROLES, not memorizes sentences).
  2. **Permuted-frame control (the load-bearing discriminator)** — shuffle the frame→role-map mapping; held-out
     role assignment MUST collapse to chance. If it survives the shuffle, the parser is reading a fixed native
     bias, not a learned frame (the standing `permuted_label_check.py` / `song_g1_core.permuted_order_controls`
     pattern).
  3. **Lesion control** — remove the learned frame map (or sever the dlPFC frame-selection route) → comprehension
     falls back to the single native frame / chance, proving the learned map is load-bearing.
  4. **The no-confab moat asserted intact THROUGHOUT** — an unstored subject queried in EITHER frame must ABSTAIN
     (`is None`); a frame change must NOT let the agent confabulate an answer. Run the abstention probe before and
     after; a regression here voids the result even if comprehension improves. (`feedback_brain_based_only_standard`.)
  5. **Native-frame no-regression** — SVO comprehension stays at its current accuracy (the new frame must not
     degrade the old).
  6. **≥6 seeds (fractional ≥5/6 bar); numpy for this cheap-first, CuPy for the decisive promotion**
     (`feedback_6seed_validation`, `feedback_gpu_not_numpy`).
  7. **Frozen bars, no config-cranking; pre-register GO/BOUNDARY/NEGATIVE before held-out data; reuse the frozen
     `song_g1_core` instruments verbatim.** Flag explicitly that this is the **comprehension** mirror of the
     CYCLE-143 generation GO, staying in the single-attribute / role-filler regime (never touches bundling).
- **Expected wall-clock:** **minutes** on CPU — the CYCLE-143 de-risk and the ordered-WM de-risks each ran
  multi-seed CPU in minutes; this composes them + the parser's Hebbian frame map. No GPU, no 5-bridge load for the
  probe.

**Why this is the right cheap-first:** it tests the one *open* sub-question of Option 1 (learned-frame
*comprehension*, the untested mirror of the proven generation side), reuses the exact validated CQ + parser +
composer + dlPFC pieces so a GO is a near-drop-in to the agent, runs in minutes, its three outcomes each cleanly
route the next move, and it *deliberately avoids* the bundling / nested walls (single-attribute role-filler only)
so a NEGATIVE is a clean statement about *learnable comprehension frames*, not a re-discovery of the SNR wall.

---

## 5. HONEST note on scale (the local-vs-cloud boundary for this capability)

- **Option 1 (productive-syntax breadth) is a MECHANISM capability, not a scale one — it is squarely in the local
  budget.** It validates at the project's probe vocabulary (V≈16 for the de-risk; V=320 for the deployed
  conversational matrix) and adds *kinds* of structure (more frames, query shapes, constituents), not *more*
  vocabulary. The cheap-first runs in minutes on CPU; the decisive promotion is a GPU 6-seed gate, no cloud.
- **Where it hits the vocabulary-scale ceiling:** the *flexibility* generalizes over ROLES (frame-agnostic of
  which concept fills a slot), so it does not itself demand more vocabulary. But the **perceived "LLM-likeness"**
  of the result is bounded by the deployed 320-concept ceiling — a syntactically-flexible agent with a 320-word
  vocabulary still reads as a small, closed-domain talker next to a basic LLM's open vocabulary + world knowledge.
  That breadth is the **deferred cloud axis** (the 320→thousands-of-concepts stream-cortex scaling needs a
  corpus-grounded taxonomy + a wall-clock training budget, per the CLAUDE.md "stream-scaling" note), NOT a
  substrate limit. ⇒ Option 1 closes a *capability-kind* gap locally; closing the *perceived-fluency* gap fully
  also needs the cloud vocabulary-scale axis — but the mechanism must be in place first, and it is local.
- **Option 3 (the hard half) is also mechanism-not-scale** for the recursion/agreement *operation*, but route 3a
  (Assembly Calculus) carries a genuine *scale risk* — the published parser runs at n=10⁵–10⁶ neurons/area vs the
  project's ~2000, and whether AC recursion works at the 100× smaller assemblies is the unproven part (a *local*
  question, but one that may push pool sizes up against local VRAM if it needs the published scale). Route 3b
  (dendritic) is local but weeks-scale build, not scale-bound.
- **Per the owner's profile (`feedback_prioritize_orchestration_overhead`):** the real-time-conversation wall is
  per-op LATENCY (now ~36 ms/turn after the megakernel, CYCLE 155 — real-time LOCALLY on the RTX 3090), NOT VRAM;
  none of Options 1/2/4 move VRAM materially. So the local-vs-cloud boundary for the recommended capability is
  clean: **build the mechanism locally now; the cloud is needed only for the vocabulary/world-knowledge SCALE that
  makes the flexible agent read as broadly as a basic LLM.**

---

### Catalog entries cited
**G.07** (pre-SMA/SMA internally-generated *abstract* sequences), **G.08** (PFC working memory / executive
control), **G.10** (language as a hierarchical symbolic system — phonemes/morphemes/words/syntax), **G.11**
(dual-stream language — dorsal production / ventral comprehension), **G.12** (Broca's grammatical processing —
behavioral validation = the "the girl that the boy is chasing is tall" center-embedding dissociation), **G.13**
(Wernicke comprehension), **H.19** (premotor sequential action / competitive queuing), **N.15** (theta-gamma
multiplexed cell-assembly buffer; Lisman-Idiart 1995), **N.19** (gamma binding-by-synchrony as attention),
**D.05/D.18** (CA3 sequential autoassociator / theta sequences). Catalog:
`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.

### Papers cited (links)
- Grossberg 1978; Bullock & Rhodes 2003 — competitive queuing for serial order (the validated CQ generator's basis).
- Lisman & Idiart 1995, *Science* — theta-gamma multiplexed STM buffer.
- Mitropolsky, Collins, Papadimitriou — **A Biologically Plausible Parser**, TACL 2021 (arXiv:2108.02189).
- Mitropolsky, Ejaz, Shi, Yannakakis, Papadimitriou — **Center-Embedding and Constituency in the Brain**, 2022
  (arXiv:2206.13217 — center-embedding **exclusively through spiking**; new context-free-language characterization,
  no software stack = the point-neuron route past arbitrary recursion, Option 3a).
- Dabagia, Papadimitriou, Vempala — **Computation with Sequences in a Model of the Brain**, ALT 2024
  (arXiv:2306.03812 — assemblies store/recall ordered sequences; FSM simulation).
- Pulvermüller & Knoblauch 2009, *Neural Networks* — sequence detectors / discrete combinatorial neuronal
  assemblies for word order (the comprehension-side primitive).
- Miller & Cohen 2001, *Annu. Rev. Neurosci.* — PFC guided-activity / task-set selection (the executive-control
  account for instruction→operation, Option 2).
- Desimone & Duncan 1995, *Annu. Rev. Neurosci.* — biased-competition attention (Option 4 exemplar→response bias).
- (point-neuron-limit framing for Option 3b) Mikulasch & Priesemann — why decorrelation & multiplicative binding
  are analog/dendritic, not point-neuron.

### Project files / findings reviewed
`CLAUDE.md` (conversational sections); `research/findings/AUTONOMOUS_STATE.md` (CYCLEs 135–160). Findings:
`2026-06-17-{conversational-architecture-to-basic-LLM-scoping, productive-syntax-scoping,
learned-multiframe-word-order-derisk, multireferent-disambiguation-NEGATIVE, multisentence-ordered-emission-derisk,
multihop-query-chain-GO}.md`,
`2026-06-16-{sentence-generation-serial-order-cheap-first-GO, sentence-generation-biologization-deep-research,
onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE}.md`. Runners:
`research/runners/{neural_serial_order_renderer, ordered_position_wm, _phaseB_learned_multiframe_word_order_derisk,
rf_phasor_composer, brain_conversational_agent, multi_turn_agent_v2, content_selection_spiking,
nested_composition_agent, _phaseB_serial_order_spiking_derisk, song_g1_core}.py`. Catalog:
`E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (clusters G, N, D, H).
