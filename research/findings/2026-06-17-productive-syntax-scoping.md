# Productive syntax — deep-research + catalog scoping (2026-06-17)

> **Read-only deep research. No code edited (this doc is the only write).** Produced before any build per the
> standing "deep research + catalog review FIRST at new directions" directive (CLAUDE.md;
> `feedback_deep_research_at_roadblocks`). The controller should trust-but-verify the load-bearing claims flagged
> **[VERIFY]** inline, then push + present the recommendation before building.
>
> **Direction being scoped:** the next *hard* core-conversational-roadmap lever after reconsolidation —
> **PRODUCTIVE SYNTAX** (option #2 in `2026-06-17-conversational-architecture-to-basic-LLM-scoping.md`: learned
> sequence-detectors + assembly-calculus recursion; Pulvermüller DCNAs; Mitropolsky center-embedding). The
> roadmap sequence (AUTONOMOUS_STATE, CYCLE 140) is: ✓scale → narrate-integration(in flight) →
> reconsolidation(medium) → **productive syntax(hard)** → dendritic credit-assignment(hardest, owner un-benched).

---

## 0. Constraints (the bars every option is judged against)

- **STRICT biology — "Definitely A"** (owner 2026-06-17): no LLM/generator hybrid. The deliverable is a
  biology-faithful brain analogue with biology-translatable insight; honest negatives under strict biology ARE the
  output (`project_actual_goal_artificial_life_brain_analogue`).
- **Point-neuron substrate is the default.** The **dendritic substrate is now on the table** (owner un-benched it
  2026-06-17; D2 Phase 0–2 already built a two-compartment neuron + learned graded cortex on the bridge, Phase 3
  pending) — so dendritic mechanisms are admissible *if syntax genuinely needs them*, but **prefer the cheapest
  point-neuron-feasible de-risk first** ("build dendrites when they are the obvious unlocker").
- **The no-confab moat is a PLUS, not a hard gate** (owner relaxed it 2026-06-17). Lossy/reconstructive is
  acceptable if it buys progress; still, keep it where it is free.
- **Reuse-by-import strongly preferred** over `sim/` edits.

---

## 1. DIAGNOSIS — what "productive syntax" requires BEYOND the current capability

The conversational stack is far past "fixed SVO Q&A." What is **already built and substrate-validated** and bears
directly on syntax (read in full from CLAUDE.md + the cited findings):

- **Fixed SVO frames + role assignment** — `BridgeParser` (`brain_conversational_agent.py`): a Hebbian-learned
  **(word-position × voice) → role** map (active: 1st→agent, 2nd→action, 3rd→patient; passive flips 1↔3),
  voice-invariant, on the bridge. The composer (`rf_phasor_composer.py`) binds the parsed roles into an SVO fact.
- **Recursive clauses / center-embedding to depth-2** — the resonator decoder
  (`nested_composition_agent.py`) decodes a fact whose patient is itself a clause ("dog see (cat chase (bird eat
  leaf))"); the unified-agent benchmark is **195/195 = 100%, no category below 100%**, with clause-depth2 resolved
  for **flat innermost args** (`2026-06-04-clause-depth2-ceiling-resolved-flat-inner.md`).
- **Neural word ORDER (parallel→serial)** — `NeuralSerialOrderRenderer` (`enable_neural_render`): the frame's
  **primacy gradient** (graded current into concept pools) → spiking-RATE ranking = emission order (competitive
  queuing, Grossberg/Bullock-Rhodes, catalog **G.07/H.19**). De-risked GO 6/6 vs permuted-order + equal-drive
  controls (`2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`).
- **Order-encoded WM (theta-gamma slots)** — `OrderedPositionWM` (CYCLE 135/140): items bound to gamma-slot
  POSITION phasors on the resonate-and-fire substrate; ordered recall **1.000 to the full 7-slot Lisman-Idiart
  span** at D=256 (Option-1 of the prior scoping, now built).
- **Multi-sentence ordered emission** — topic-sequencing over the ordered-WM (GO 6/6,
  `2026-06-17-multisentence-ordered-emission-derisk.md`).

**So the gap is NOT "no structure."** It is that **every structure is a FIXED, hand-specified frame**: one SVO
template; role = a hardcoded (position×voice) rule; word order = a hardcoded primacy tuple
(`_phaseB_serial_order_spiking_derisk.PRIMACY_pA` = "agent-before-action-before-patient"); recursion = one
template (patient-slot-is-a-clause) decoded by a fixed resonator. **Productive syntax = the structures themselves
are LEARNED and GENERATIVE** — the system handles sentence types it was never given as a template, generalizing
constituency/word-order rules to novel structures. Decomposition (the gap table):

| Sub-capability | Basic-LLM behaviour | Project status today | Precise gap | Collides with a known wall? |
|---|---|---|---|---|
| **Word-order grammar (novel sentence types)** | arbitrary; learned | ⚠️ ONE fixed SVO frame; order = a hardcoded primacy tuple | no mechanism to **learn ≥2 distinct word-order frames** and **select** the right one (e.g. SVO vs a ditransitive S-V-IO-DO, or a question order) | **mild** — different orders over the SAME slots ride the validated CQ + ordered-WM; this is the cheapest productive step |
| **Constituency / phrase structure** | groups words into nested constituents | ⚠️ flat SVO + 1 templated embedded clause | no mechanism to **form a constituent on the fly** ("[the big dog] [chased] [the small cat]") and treat it as a unit slot-filler in an arbitrary position | **mild–moderate** — single-attribute bind + resonator decode exist; *arbitrary-position* constituency is new |
| **Center-embedding / recursion DEPTH** | deep (humans struggle past ~2) | ⚠️ depth-2 with FLAT inner args (templated) | depth-2 **attributed** inner args degrade to the flat noun; **arbitrary**/learned recursion (not one template) absent | **DIRECT** — the **nested-composition / SNR wall** (`2026-06-02-full-320…hierarchical null`): binding composites-of-composites degrades |
| **Agreement / dependency binding** | subject-verb agreement, long-range deps | ❌ absent (no number/gender features) | no mechanism to **bind a non-adjacent dependency** (the verb agrees with the subject across an embedded clause) | **DIRECT** — non-adjacent dependency = bind a feature across a bundle = the **multi-attribute BUNDLING NEGATIVE** (`2026-06-16-onsubstrate-learned-binder…bundling-NEGATIVE`): bundling is NOT learnable from scratch on point neurons; rides on a fixed coincidence primitive |

**The two structural themes that decide the build:**
1. **The "easy half" of productive syntax is reachable on the validated substrate**: *novel word-order frames*
   and *constituent-as-slot-filler* re-use the CQ serial-order generator (order = a learned primacy gradient,
   not a hardcoded tuple), the ordered-WM (slots = grammatical roles), the single-attribute learned binder (GO on
   spikes), and the resonator (constituent decode). The bind these need is **single-attribute / role-filler**,
   which the project has VALIDATED on real LIF (held-out 0.833 = 100% of numpy).
2. **The "hard half" — arbitrary recursion depth + non-adjacent agreement — collides head-on with the two known
   walls** (nested-composition SNR; multi-attribute bundling-not-learnable). Those are the *same* walls
   `2026-06-16` and `2026-06-02` mapped, and they are where the **dendritic substrate becomes the candidate
   unlocker** (multiplication/superposition-inverse is a dendritic op, per Mikulasch-Priesemann — see §4).

⇒ **Productive syntax is two problems, not one.** A cheap-first de-risk should test the **smallest genuinely
productive step from the easy half** (a novel word-order frame the agent was NOT given as a template), and the
write-up should *explicitly* park the hard half (deep recursion + agreement) against the named walls so the build
is not mis-scoped as "all of syntax."

### 1a. The literature pivot the controller must hold (so we adopt, not reinvent)

There is a mature biologically-plausible-syntax research line — the **Assembly Calculus / NEMO** programme
(Papadimitriou, Vempala, Mitropolsky, Dabagia, Collins) — built on **EXACTLY this project's substrate primitives**
and it is load-bearing for the ranking:

- **The substrate is point-neuron** [VERIFY — confirmed from the language-organ paper, arXiv:2306.15364]: brain
  *areas* of excitatory neurons in **random Erdős-Rényi graphs** (p ≈ 0.001–0.05), a **k-cap winner-take-all**
  per area (the top-k by synaptic input fire — the excitatory/inhibitory balance, **no explicit dendrites**), and
  **multiplicative Hebbian** plasticity (w ← w(1+β), β ≈ 5–10%). Scale n = 10⁵–10⁶ per area, k = 50–1000. The
  authors state it is "implementable on any spiking neural network framework." **The project already has the
  matching assembly representation**: `concept_pool_sparse_distributed.build_sparse_pool_bridge` /
  `generate_sparse_patterns` — "the Pulvermüller / Kanerva form: each concept = sparse random K-of-N pattern"
  (K=100 of N=2000), and k-cap = the existing FS-PV / MSN lateral-inhibition WTA motifs. **So Assembly Calculus is
  point-neuron-realizable on this bridge BY CONSTRUCTION.** (`feedback_check_existing_sims_first`: adopt the proven
  mechanism.)
- **The capability ladder splits sharply** — and this is the nuance the prior scoping under-stated:
  - the **2023 "Language Organ"** result (arXiv:2306.15364) does only **part-of-speech categorization** — it
    explicitly **does NOT generate word order or handle recursion** ("perhaps the most important direction left
    open"). It is the *weak* AC result.
  - the **2021 parser** (Mitropolsky, Collins, Papadimitriou, **TACL 2021**, arXiv:2108.02189) parses "reasonably
    nontrivial" English sentences via projection between SUBJ/VERB/OBJ-type areas as words are read, and
    *discusses* recursion + embedding.
  - the **2022 center-embedding** paper (Mitropolsky, Ejaz, Shi, Yannakakis, Papadimitriou, arXiv:2206.13217)
    handles **center-embedded recursion "exclusively through the spiking of neurons,"** yielding "a new
    characterization of context-free languages" **WITHOUT an explicit software stack**. [VERIFY — confirmed from
    the abstract: spiking-only, center-embedding, CF-language characterization.]
  - the **2024 sequences-of-assemblies** paper (Dabagia, Papadimitriou, Vempala, **ALT 2024**, arXiv:2306.03812)
    shows assemblies **store/recall ordered sequences and simulate finite-state machines** — the sequential-syntax
    substrate.
- ⇒ **the productive-syntax mechanism EXISTS and is point-neuron** in the AC line; the open question for *this*
  project is **realizing it on the deployed FHRR/stream-cortex conversational substrate** at the project's scale
  (the AC parser runs at n=10⁵–10⁶/area; the project's pools are ~2000), and whether it **buys a genuinely-novel
  structure** over the fixed-frame baseline. That is precisely a cheap-first de-risk question.

---

## 2. RANKED biologically-grounded mechanisms

Bars: closes the most gap · direct catalog + literature grounding · point-neuron-feasible · reuses existing
machinery · minimal/no `sim/` edit · keeps the moat where free.

### Option 1 (RECOMMENDED) — Learned multi-FRAME word order: grammatical roles = ordered-WM slots, order = a LEARNED primacy gradient (CQ), frame SELECTED by the dlPFC

- **Mechanism (one paragraph):** Generalize the *fixed* SVO order into a small inventory of **learned word-order
  frames**. Each frame is a **learned primacy gradient** over the ordered-WM's grammatical-role slots — the SAME
  competitive-queuing read-out the project validated (`NeuralSerialOrderRenderer`: graded current → spiking-rate
  ranking → emission order), except the gradient is *learned per frame* (Hebbian) instead of the hardcoded
  `PRIMACY_pA` tuple. The role slots are the **theta-gamma gamma-slots** of the order-encoded WM (a grammatical
  role = a position phasor), so "which role is in which slot" is the validated ordered-WM mechanism. The dlPFC
  spreading-Control **selects** which frame (which primacy gradient) to apply for the current utterance type
  (declarative SVO vs. e.g. a ditransitive S-V-recipient-theme, or a yes/no-question order). This produces
  **sentence types the agent was not given as a template**, while every binding it needs stays **single-attribute
  / role-filler** — the validated-on-spikes regime — and never touches multi-attribute bundling.
- **Biology source:** catalog **G.07** (pre-SMA internally-generated sequences — abstract sequence vs. movement
  sequence) + **H.19** (premotor competitive queuing) for the parallel→serial order read-out (Grossberg 1978,
  Bullock-Rhodes 2003); catalog **G.12** (Broca's grammatical processing — the catalog's behavioral validation is
  the syntactic-complexity dissociation); **N.15** (theta-gamma multiplexed slots = the ordinal buffer, Lisman-
  Idiart 1995); AC **sequences-of-assemblies** (Dabagia 2024) as the assembly-level account of learned ordered
  sequences. The "frame = a learned primacy gradient selected by PFC" is the project's own validated CQ + dlPFC
  pieces recomposed.
- **Which gaps it closes:** *novel word-order grammar* (≥2 learned frames + selection) and the "real syntax vs
  fixed SVO" boundary CLAUDE.md flags as the open follow-on to the serial-order generator. It does NOT attempt
  deep recursion or agreement (the hard half — parked).
- **Point-neuron feasibility:** **HIGH (point-neuron-native).** Everything reuses validated point-neuron pieces:
  CQ rate-ranking, ordered-WM gamma-slots (RF substrate), dlPFC spreading-Control. The only new learning is a
  Hebbian primacy gradient per frame (a graded-current pattern), which is a standard rate-Hebbian map. **dt-bound
  caveat:** rate-ranking ties when slots are equidistant (CLAUDE.md one-bridge step-3); run the WM/order engine at
  its native dt=0.5 (the dlPFC already does) and keep frame inventories small (2–4 frames).
- **`sim/` edit needed?** **NO** — `NeuralSerialOrderRenderer` + `OrderedPositionWM` + `SpikingSpreadingController`
  are reuse-by-import; the learned primacy gradient is a runner-side Hebbian map.
- **Reusable machinery:** `neural_serial_order_renderer.py` + `_phaseB_serial_order_spiking_derisk.py` (the CQ
  read-out), `ordered_position_wm.py` (role-slots), `content_selection_spiking.SpikingSpreadingController` (frame
  selection), `brain_conversational_agent.BridgeParser` (the comprehension side, which already learns a
  position→role map and can be extended to ≥2 frames).

### Option 2 — Assembly-Calculus parser/generator + DCNA sequence-detectors (the deepest, highest-ceiling, point-neuron-feasible-but-unproven-at-this-scale option)

- **Mechanism:** Replace the fixed (position×voice)→role parser with a NEMO-style **projection parser** — brain
  *areas* (LEX, SUBJ, VERB, OBJ) of sparse assemblies; as each word is read, a projection + reciprocal-projection
  sequence with **k-cap WTA** and Hebbian plasticity routes it to its role area and binds dependencies; word order
  is consumed by the *sequence* of projections (Dabagia-2024 sequences-of-assemblies). Productive recursion /
  center-embedding is handled by the **2022 spiking center-embedding** mechanism (a disinhibition/blocking control
  that re-enters the role areas — "a new characterization of context-free languages," no software stack).
  **Sequence-detectors / DCNAs** (Pulvermüller-Knoblauch) are the comprehension-side primitive: assemblies that
  fire to AB but not BA, giving learned word-order sensitivity.
- **Biology source:** Mitropolsky-Collins-Papadimitriou **TACL 2021** (parser); Mitropolsky-Ejaz-Shi-Yannakakis-
  Papadimitriou **2022** (center-embedding/constituency, spiking-only); Dabagia-Papadimitriou-Vempala **ALT 2024**
  (sequences/FSM); Papadimitriou et al. **PNAS 2020** (Brain computation by assemblies); Pulvermüller-Knoblauch
  2009 (DCNAs); catalog **G.10/G.12/G.13** (hierarchical syntax / Broca / Wernicke), **G.07** (sequences).
- **Which gaps it closes:** *all four* — novel word order, constituency, recursion depth, dependency binding — in
  principle, because it is a genuine generative grammar over assemblies.
- **Point-neuron feasibility:** **MEDIUM (feasible in principle; unproven at this project's scale).** The
  substrate is point-neuron by construction (random areas + k-cap + Hebbian = the project's sparse-pool + WTA
  motifs). **The risk is scale**: the published parser uses n=10⁵–10⁶ neurons per area and ~10–20 training
  sentences per word; the project's conversational pools are ~2000 neurons. Whether the AC parsing/recursion works
  at the 100× smaller assemblies the conversational bridge uses is the unproven part — AND it would **not reuse the
  FHRR composer** (it is a *different* representational substrate: k-cap assemblies, not phasor VSA), so it is a
  parallel build, not an extension of the deployed agent.
- **`sim/` edit needed?** Probably **NO** for the assembly/k-cap mechanics (sparse pools + lateral-inhibition WTA +
  Hebbian exist), but it is **substantial new runner machinery** (the area graph + projection scheduler + the
  disinhibition recursion control).
- **Reusable machinery:** `concept_pool_sparse_distributed.py` (K-of-N assemblies = NEMO assemblies), the FS-PV /
  MSN WTA motifs (`g11_bg_runner --enable-msn-lateral-inhibition` etc.) as k-cap, the learned single-attribute
  binder (sequence-detector primitive), the transmission gate (`sim/bridge.py` — the disinhibition/blocking
  control for recursion re-entry), the dlPFC Control.

### Option 3 — Slot-grammar over the order-encoded WM (a thin, near-term subset of Option 1)

- **Mechanism:** Treat the validated **gamma-slots as grammatical roles** directly: a grammar is a learned
  *mapping* from an utterance type to a slot-assignment + a slot-emission-order. No new representational substrate;
  just (i) more than one slot-assignment rule and (ii) the CQ order read-out per rule. This is **Option 1 minus the
  Hebbian-learned gradient** (the orders are *configured* rather than *learned*) — useful only as the immediate
  scaffold / positive-control, not as the productive end-state (configured orders are a host shortcut for the
  learned gradient).
- **Biology source:** N.15 (slots) + G.07/H.19 (order) — same as Option 1.
- **Point-neuron feasibility:** **HIGH** (pure reuse).
- **Why ranked 3rd:** it does not *learn* the frames, so under the BRAIN-BASED-ONLY standard the frame inventory is
  a host shortcut. It is the right **de-risk scaffold** for Option 1 (prove the slots-as-roles + multi-order
  emission composes), then swap the configured gradient for a learned one.

### Option 4 (HARD HALF — flagged, not a near-term build) — Dendritic structured representations for recursion + agreement

- **Mechanism:** Use the D2 two-compartment dendritic neuron to realize the **multiplicative / superposition-
  inverse** operation that arbitrary recursion (nested-composite binding) and non-adjacent agreement (unbind a
  feature from a deep bundle) require — the operation point neurons provably cannot do unaided.
- **Biology source:** Mikulasch-Priesemann point-neuron limit (the project's recurring wall: whitening,
  decorrelation, and **binding-superposition** are analog/dendritic, not point-neuron); the project's D2 dendritic
  arc (Phase 0–2 built; Phase 3 pending).
- **Point-neuron feasibility:** **N/A — this IS the dendritic option.** It is the candidate unlocker for the
  *hard half only*, and only if the cheap-first easy-half de-risk confirms the hard half is the residual blocker.
- **Why flagged not built:** months-scale; the owner's "build dendrites when they are the obvious unlocker" gate is
  not yet met for syntax — the easy half (Options 1/3) must be exhausted first to *localize* whether deep
  recursion / agreement is truly the next blocker (see §4).

**Ranking rationale:** **Option 1** is the highest near-term lever — it delivers a genuinely-productive capability
(novel word-order frames the agent was never templated with) on the **validated point-neuron substrate**, reusing
the CQ generator + ordered-WM + dlPFC, with **no `sim/` edit** and the single-attribute bind regime (never
touching the bundling wall). **Option 3** is its de-risk scaffold (do first, then learn the gradient). **Option 2**
is the deepest capability but a *parallel* assembly-substrate build at an unproven-at-scale point — the strategic
end-state for full grammar, sequenced after Option 1 proves the easy half and *if* the project wants genuine
generative grammar rather than a learned frame inventory. **Option 4 (dendritic)** is the hard-half unlocker,
flagged for when the easy half localizes deep-recursion/agreement as the residual wall.

---

## 3. RECOMMENDED #1 + its cheapest decisive de-risk

**Recommend Option 1 (learned multi-frame word order).** The de-risk tests the **smallest genuinely-productive
step**: can the agent **generate a sentence in a word-order frame it was NOT given as a fixed template** — a
*second, learned* frame — and *select* the right frame, beating the fixed-SVO baseline and the anti-cheats. This is
deliberately **not** a re-test of the fixed SVO (which already passes 6/6); it is the minimal "produces novel
structure" probe.

**Cheap-first de-risk (CPU/numpy, reuse-by-import, no `sim/` edit, minutes):**

- **Setup.** Reuse `NeuralSerialOrderRenderer` (CQ rate-ranking) + `OrderedPositionWM` (role-slots) + a tiny
  Hebbian primacy-gradient map. Define **two** word-order frames over the SAME role slots, e.g. **F1 = SVO**
  (agent→action→patient) and **F2 = a distinct order** the agent must produce, e.g. a *verb-initial* "action agent
  patient" frame OR a *ditransitive* S-V-recipient-theme order (4 slots). **Train** the per-frame primacy gradient
  by Hebbian co-firing on a handful of F1 and F2 example utterances (frame-tagged). **Test on HELD-OUT
  fillers**: novel (agent, action, patient) tuples the gradient was never trained on, emitted in each frame; the
  dlPFC Control picks the frame from the utterance-type cue.
- **Metric.** For a held-out tuple under frame Fᵢ, does the emitted word ORDER match Fᵢ's order (the CQ rate
  ranking == the learned gradient ranking), AND does **frame-selection** route to the correct frame?
- **Pre-registered GATE (FROZEN before seeing data; ≥6 seeds, FRACTIONAL ≥5/6 bar per `feedback_6seed_validation`):**
  - **GO:** held-out emission in the **second, learned frame** is correct (order-accuracy ≥ 0.90) on ≥5/6 seeds,
    AND a frame-SELECTION test (utterance-type cue → correct frame) ≥ 0.90 on ≥5/6 seeds, AND the **permuted-frame
    control collapses to chance** (train the gradient on shuffled frame→order labels → emission order is random),
    AND the **lesion collapses it** (remove the learned gradient → emission falls back to a single fixed order /
    equal-drive chance). ⇒ the agent produces novel grammatical structure; promote to a GPU 6-seed gate + wire a
    learned multi-frame `render` into the agent (default-off).
  - **BOUNDARY:** the second frame's order is learnable in isolation but **frame-SELECTION is unreliable** (the
    dlPFC can't route utterance-type → frame), OR it works for 2 frames but a 3rd/4th frame interferes — a real,
    partial, publishable result that localizes selection vs. capacity as the next sub-problem.
  - **NEGATIVE:** the learned second frame is **no better than chance / no better than the fixed-SVO baseline** at
    held-out emission (the CQ gradient can't be *learned* per-frame, only hardcoded), OR the permuted-frame control
    does NOT collapse (the "frame" is reading a fixed structural bias, not a learned order). ⇒ learned word-order
    grammar is itself a wall on point neurons; record it (a biology-translatable negative about learnable serial
    order) and reconsider — possibly escalate to the AC projection parser (Option 2) for order-as-projection-
    sequence.
- **Expected wall-clock:** **minutes** on CPU (the serial-order de-risk and the ordered-WM de-risks each ran
  multi-seed CPU in minutes; this composes them + a tiny Hebbian map). No GPU, no 5-bridge load for the probe.

**Why this is the right cheap-first:** it is the *minimal* mechanism change that produces a genuinely-new structure
(a learned, not hardcoded, word-order frame), it reuses the exact validated CQ + ordered-WM + dlPFC pieces so a GO
is a near-drop-in to the agent, it runs in minutes, its three outcomes each cleanly route the next move, and it
*deliberately avoids* the nested/bundling walls (single-attribute role-filler only) so a NEGATIVE is a clean
statement about *learnable serial order*, not a re-discovery of the SNR wall.

### Anti-cheat controls (mandatory — a "success" without all of these is an artifact)

1. **Held-out fillers never trained** — the gradient is learned on example tuples; emission tested on novel tuples
   (proves the frame generalizes, not memorizes the training sentences).
2. **Permuted-frame control (the load-bearing discriminator)** — train the primacy gradient on **shuffled
   frame→order labels**; held-out emission order MUST collapse to chance. If order survives the shuffle, the model
   reads a fixed structural bias, not a learned frame (mirror of the standing `permuted_label_check.py`).
3. **Lesion** — remove the learned gradient (or sever the dlPFC frame-selection route) → emission falls back to a
   single fixed order / equal-drive chance, proving the learned gradient is load-bearing.
4. **Equal-drive control (reuse the CQ de-risk's own)** — no primacy gradient → no reliable order (already
   validated; re-assert it holds with the *learned* gradient).
5. **The no-confab moat asserted intact** — the frame machinery must not let the agent emit a confabulated
   filler; an unstored topic still abstains (the moat is free here — keep it).
6. **≥6 seeds, fractional ≥5/6 bar; CuPy for any decisive/heavy promotion, numpy only for the cheap-first**
   (`feedback_6seed_validation`, `feedback_gpu_not_numpy`).
7. **Frozen bars, no config-cranking; pre-register GO/BOUNDARY/NEGATIVE before held-out data.** Flag explicitly
   that this is a NEW *learned-frame* mechanism, not a re-run of the fixed-SVO serial-order GO.

---

## 4. HONEST RISK + the dendritic-unlock question

**Where this hits the walls.** Option 1's de-risk is the **easy half** and I expect it to GO (or BOUNDARY on
frame-selection) — it stays inside the single-attribute / role-filler regime that is **already validated on real
LIF** (held-out 0.833 = 100% of numpy), and the CQ + ordered-WM pieces are 6/6-GO. The genuine risk is the
**second frame's gradient not being *learnable* per-frame** (only hardcodable) — but that is a standard rate-
Hebbian map over graded currents, low-risk on point neurons.

**The hard half is where the walls bite, and they bite in two named places:**
- **Arbitrary recursion depth** (not one template) → the **nested-composition / SNR wall**
  (`2026-06-02-full-320…hierarchical null`): binding composites-of-composites degrades because superposition
  cross-talk grows with nesting. The project's depth-2-flat fix was a *decode-policy* patch, not a capacity lift;
  *attributed* deep inner args already degrade. Deeper/learned recursion will re-meet this.
- **Non-adjacent agreement / long-range dependency** → the **multi-attribute BUNDLING NEGATIVE**
  (`2026-06-16-onsubstrate-learned-binder…bundling-NEGATIVE`): unbinding a feature from a deep bundle is a
  role-specific *multiplicative inverse*, which a learned *linear* point-neuron unbind **structurally cannot**
  implement (0.056, breaks even single-attribute) — the same Mikulasch-Priesemann point-neuron limit
  (multiplication is dendritic).

**Is syntax the moment the dendritic substrate becomes the obvious unlocker?** **Partly — and the de-risk is
designed to answer it.** My honest call:
- **For the EASY half (novel word-order frames, constituent-as-slot-filler): NO.** It is point-neuron-feasible now
  (Option 1), and it is the right first build — it delivers genuinely-productive syntax (novel structures)
  *without* dendrites.
- **For the HARD half (arbitrary recursion depth + non-adjacent agreement): YES, this is the strongest candidate
  yet for the dendritic unlock** — both sub-capabilities reduce to the *exact* operation (superposition-
  inverse / multiplicative binding) that the project has now FOUR-times found to be the point-neuron wall
  (whitening, decorrelation, deep nesting, bundling), and that the AC center-embedding result handles with a
  *blocking/disinhibition control* (Option 2) rather than deeper binding — i.e. the two routes past the hard half
  are **(a) the dendritic multiplicative substrate** or **(b) the AC projection-parser's disinhibition recursion
  (no deeper binding, a control-flow trick).** **Recommendation:** run the Option-1 easy-half de-risk first
  (cheap, point-neuron, near-drop-in); if it GOes, the *next* scoping question is precisely "deep recursion +
  agreement: AC disinhibition-control (Option 2) vs. dendritic multiplication (Option 4)" — and *that* is the
  decision point where dendrites become the obvious unlocker, with the AC control-flow route as the point-neuron
  alternative to weigh against it. Do NOT commit to dendrites for syntax before the easy half localizes the hard
  half as the residual blocker.

---

## 5. Load-bearing claims the controller should trust-but-verify

1. **[VERIFY — most load-bearing]** The Assembly-Calculus/NEMO substrate is **point-neuron** (random Erdős-Rényi
   areas + **k-cap WTA** + multiplicative Hebbian, NO dendrites; n=10⁵–10⁶/area, k=50–1000, β≈5–10%) and "the
   center-embedding/constituency result is realized **exclusively through spiking**" with "a new characterization
   of context-free languages, **no software stack**." *(Confirmed from arXiv:2306.15364 §substrate + arXiv:2206.13217
   abstract; the parser is arXiv:2108.02189/TACL-2021. Read those to confirm scale + that recursion is
   disinhibition-control, not deeper binding.)* If right, Option 2 is point-neuron-feasible **in principle** and
   the only open risk is the project's 100× smaller assemblies.
2. **[VERIFY]** The project's `concept_pool_sparse_distributed.build_sparse_pool_bridge` K-of-N sparse pattern
   ("Pulvermüller / Kanerva form") **is** the AC assembly representation, and the FS-PV / MSN lateral-inhibition
   motifs **are** a k-cap WTA — i.e. Option 2's substrate already exists. *(Read the builder + one WTA motif.)*
3. **[VERIFY]** The CQ serial-order generator's frame is currently a **hardcoded primacy tuple**
   (`_phaseB_serial_order_spiking_derisk.PRIMACY_pA` = SVO), so "learn a *second* primacy gradient" is the minimal
   productive step and is a Hebbian map, not a new substrate. *(Read `neural_serial_order_renderer.py` +
   `_phaseB_serial_order_spiking_derisk.py`.)*
4. **[VERIFY — wall placement]** Arbitrary recursion depth reduces to the **nested-composition SNR wall**
   (`2026-06-02-full-320…`) and non-adjacent agreement reduces to the **multi-attribute bundling NEGATIVE**
   (`2026-06-16-onsubstrate-learned-binder…`) — i.e. the hard half is the *same* point-neuron limit the project
   has mapped, and dendrites/disinhibition-control are the two routes past it. *(Re-read those two findings'
   verdict sections.)*
5. **[VERIFY — scope]** The easy half (novel word-order frames, constituent-as-slot) stays in the **single-
   attribute / role-filler** regime that is validated on real LIF (held-out 0.833,
   `2026-06-16-onsubstrate-learned-binder…GO`), so Option 1's de-risk does NOT touch bundling. *(Confirm the
   de-risk binds only one filler per slot.)*

---

### Catalog entries cited
**G.07** (pre-SMA/SMA internally-generated sequences — abstract vs. movement sequence), **G.08** (PFC working
memory), **G.10** (language as hierarchical symbolic system — phonemes/morphemes/words/syntax), **G.11** (dual-
stream language), **G.12** (Broca's grammatical processing — behavioral validation = the "the girl that the boy is
chasing is tall" center-embedding dissociation), **G.13** (Wernicke comprehension), **H.19** (premotor sequential
action / competitive queuing), **N.15** (theta-gamma multiplexed cell-assembly buffer, Lisman-Idiart 1995),
**D.05** (CA3 sequential autoassociator), **D.18** (theta sequences / compression). Catalog:
`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.

### Papers cited (links)
- Mitropolsky, Collins, Papadimitriou — **A Biologically Plausible Parser**, TACL 2021:
  https://aclanthology.org/2021.tacl-1.81/ (arXiv:2108.02189)
- Mitropolsky, Ejaz, Shi, Yannakakis, Papadimitriou — **Center-Embedding and Constituency in the Brain**, 2022:
  https://arxiv.org/abs/2206.13217 (center-embedding **exclusively through spiking**; new CF-language characterization)
- Mitropolsky et al. — **The Architecture of a Biologically Plausible Language Organ**, 2023:
  https://arxiv.org/abs/2306.15364 (the NEMO substrate spec — point neurons, k-cap, Hebbian; POS-only, **no word
  order / recursion** = the open direction)
- Dabagia, Papadimitriou, Vempala — **Computation with Sequences in a Model of the Brain**, ALT 2024:
  https://arxiv.org/abs/2306.03812 (assemblies store/recall ordered sequences; FSM simulation)
- Papadimitriou, Vempala, Mitropolsky, Collins, Maass — **Brain computation by assemblies of neurons**, PNAS 2020:
  https://www.pnas.org/doi/10.1073/pnas.2001893117 (the Assembly Calculus foundation)
- Pulvermüller & Knoblauch 2009, *Neural Networks* — discrete combinatorial neuronal assemblies / sequence
  detectors for word order.
- Grossberg 1978; Bullock & Rhodes 2003 — competitive queuing for serial order (the validated CQ generator's
  basis).
- Lisman & Idiart 1995, *Science* — theta-gamma multiplexed STM buffer.
- (point-neuron-limit framing) Mikulasch & Priesemann — dendritic prediction-error / why decorrelation &
  multiplicative binding are analog/dendritic, not point-neuron.

### Project files / findings reviewed
`CLAUDE.md` (conversational sections); `research/findings/AUTONOMOUS_STATE.md` (CYCLE 135–140 roadmap);
`research/runners/{rf_phasor_composer, brain_conversational_agent, ordered_position_wm, neural_serial_order_renderer,
concept_pool_sparse_distributed, nested_composition_agent, content_selection_spiking}.py`; findings
`2026-06-17-{conversational-architecture-to-basic-LLM-scoping, multihop-reasoning-multiturn-dialogue-scoping,
multisentence-ordered-emission-derisk}.md`,
`2026-06-16-{onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE, sentence-generation-serial-order-cheap-first-GO}.md`,
`2026-06-04-clause-depth2-ceiling-resolved-flat-inner.md`, `2026-06-02-full-320-flat-distinct-composition-RESOLVES-multiseed.md`.
Catalog: `sim-catalog/references/feature-catalog.md` (clusters G, N, D, H).
