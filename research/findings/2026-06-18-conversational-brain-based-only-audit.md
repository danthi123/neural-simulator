# Conversational pipeline — BRAIN-BASED-ONLY host-shortcut audit (2026-06-18)

**Audit type:** read-only. No code edited, no experiments run.
**Standard audited against:** BRAIN-BASED-ONLY (CLAUDE.md + memory `feedback_brain_based_only_standard`).
A cognitive operation counts as "the brain doing the work" ONLY if realized by simulated neurons firing,
synapses, and their communication. Host (numpy/Python) code is legitimate ONLY for the ENVIRONMENT (world
state + sensory render) and the BODY (acting on neural motor output). A host computation that does cognitive
work is a shortcut **even if biologically correct**. Two non-cheats: (a) a **structural neural primitive** the
substrate genuinely has (binding-by-coincidence / dendritic multiplication, a fixed self-inverse role code),
and (b) a **teaching SCAFFOLD** explicitly flagged for conversion (an environment-supplied `target`).

**Production path audited:** `BrainConversationalAgent` (default `composer_kind="rf"`) →
`RFPhasorComposer` (the production default composer) + `BridgeParser` (comprehension) + the dlPFC dialogue
planner (`SpikingSpreadingController`) + opt-in `MultiTurnAgentV2` / `OrderedPositionWM` (multi-turn) +
opt-in `NeuralSerialOrderRenderer` (word order). The merged/unified bridges
(`nav_conv_merged_bridge.py`, `unified_brain_bridge.py`) delegate every conversational method to these same
pieces — they add NO new cognitive operation, so the table below covers them too.

**Scope note on the recently-biologized work (do NOT re-flag):** CYCLE 158–160 (2026-06-17) made a *learned
spiking binder's* read-out decoder `W_O` learned by real synaptic plasticity on the bridge
(`cp_per_synapse_reward_override`, `sim/bridge.py:6866-6878`) and showed the per-output teaching error is
neuralisable (a predictive-coding error population). That work lives in the SEPARATE learned-binder arc
(`OnBridgeLearnedComposer` / the planned `LearnedSpikingComposer`,
`_phaseB_onbridge_learned_composer_derisk.py`), which is **not yet wired into the production
`RFPhasorComposer`**. The production composer below still uses the fixed exact-inverse FHRR algebra. The only
residual scaffold of that read-out learning is the environment `target` (legitimate). I note it as done where
relevant and do not re-flag it.

---

## Per-operation table

Legend for column 3 — **ALREADY-NEURAL ✓** | **HOST-SHORTCUT** (genuine gap) | **DEFENSIBLE** (structural
primitive / env / body / flagged scaffold) | **IDEALIZATION** (principled stand-in).

| # | Operation | Current realization (file:line) | Class | Notes / fix if HOST-SHORTCUT |
|---|-----------|--------------------------------|-------|------------------------------|
| 1 | **Frame/role parse** (word position×voice → agent/action/patient) | NEURAL: 6 conjunction units → 3 role ensembles on a `SimulationBridge`, Hebbian-trained; role read by spiking firing. `brain_conversational_agent.py:110` (`_train`), `:123` (`role_of`) | ALREADY-NEURAL ✓ | The mapping is learned by real Hebbian co-firing; comprehension drives the (position,voice) conjunction and reads role-ensemble spikes. |
| 1b | **Parser role decision** (which role ensemble won) | HOST argmax over spike rates: `max(rates, key=rates.get)` `brain_conversational_agent.py:137` (merged variant `nav_conv_merged_bridge.py:161,172`) | DEFENSIBLE (read-out of spiking output) | Same class as #5/#7 below — a host argmax reading already-spiking populations. The competition is neural; the host only reports the winner. Per project precedent (NEF cleanup "argmax-over-firing") this is the accepted read-out boundary, but it is the same primitive flagged elsewhere; a spiking winner-take-all would make it fully neural. Low priority. |
| 2 | **The bind / unbind / bundle algebra** (role⊗filler, conj-unbind, superposition) | NEURAL ops on the substrate: resonate-and-fire phasor neurons + complex synapses. `rf_phasor_composer.py:105` (`_resonate`), `:122` (`_bind`), `:133` (`_bundle`), `:170` (`_unbind_phases`) | IDEALIZATION (ops are spiking; the *exactness* is the idealization) | The OPERATIONS run on real RESONATE_AND_FIRE neurons + complex synapses (Frady-Sommer 2019). What is idealized is that bind/unbind is an **exactly invertible** vector-symbolic algebra (Eliasmith Spaun / Semantic Pointer Architecture) demanding clean codes — a real cortex learns lossy read-outs. The learned-binder arc (CYCLE 149–160, `OnBridgeLearnedComposer`) is incrementally replacing the exact algebra with a learned spiking binder; not yet in production. Documented limitation. |
| 3 | **Concept-code cleanup / nearest-neighbour** (recovered phasor → nearest concept word) | **HOST (DEFAULT)**: `sims = mean(cos(rec − concepts[w]))` then `argmax`. `rf_phasor_composer.py:262-263` (`_cleanup`), batched form `:293` (`_cleanup_all`). **Opt-in neural exists**: `_spiking_cleanup` (matched filter on complex synapse + Izhikevich WTA) `rf_phasor_composer.py:208`, gated by `enable_spiking_cleanup` (default **False** `:77`). | HOST-SHORTCUT on the default path (opt-in neural version validated) | The default production path computes the nearest-concept match in numpy. The fully-on-bridge spiking version is built and validated (== numpy multi-seed) but **off by default**. Fix = flip `enable_spiking_cleanup=True` as the production default (reuse-by-import, no new mechanism); difficulty LOW. The rate composer has the same pattern (NEF cleanup, `core_sim_composition.py:362-373`, also default off). |
| 4 | **Fact store** (`self.kb`: the bound composite per fact) | **HOST (DEFAULT)**: a Python list of numpy phasor arrays. `rf_phasor_composer.py:99` (`self.kb=[]`), `:333` (`store` appends the numpy composite). **Opt-in neural exists**: `_store_substrate` holds the composite in per-fact complex synaptic weights (Crawford-Eliasmith weight-store) `:406`, `_retrieve_substrate` reads it back in spikes `:418`, gated by `enable_substrate_store` (default **False** `:73`). | HOST-SHORTCUT on the default path (opt-in neural version validated) | The default stores facts as numpy arrays in a list; retrieval indexes that list. The substrate weight-store (memory-in-synapses, read by firing the trigger) is built + validated (== numpy at parity) but off by default. Fix = make `enable_substrate_store=True` the production default; difficulty LOW–MEDIUM (the spiking store adds per-op bridge cost). Rate composer mirror: `enable_spiking_memory`, `core_sim_composition.py:419-426`. |
| 5 | **Abstention / no-confab moat decision** (return None / "unknown" when no stored fact's cue roles match) | HOST: the match is `self.unbind(comp, role) == cue_value` (string equality) inside the store-scan loop, then `return None`. `rf_phasor_composer.py:441-444` (`query_agent`), `:461-468` (`query_patient`), `:493-497` (`ask_yes_no`); batched form `_scan_first_match` `:305`. | HOST-SHORTCUT (with a validated neural replacement existing elsewhere) | The abstention *gate* in the production composer is a host equality + None. A neural familiarity gate (Bogacz-Brown novelty: a spiking match-strength threshold) is validated elsewhere and is the production gate in `OrderedPositionWM.read_slot` (`ordered_position_wm.py:120-131`, `match < threshold → abstain`) and in the V=320 familiarity-gate finding (`2026-06-11-familiarity-gate-v320-GO.md`). Fix = route the composer's match/abstain through the validated familiarity-gate population instead of `==`/None; difficulty MEDIUM. **Per owner memory `feedback_moat_not_hard_lossy_memory_ok`, the moat is a plus, not a hard gate — biologizing it is desirable but the lossy/learned path is explicitly OK to trade for.** |
| 6 | **Word-order emission** (impose serial SVO/clause order on recalled fillers) | **HOST (DEFAULT)**: `f"{agent} {ac} {pt}"` / `f"{a} {ac} {pt}"`. `rf_phasor_composer.py:520` (`render_fact`), `:167` (`_render` inner clause). **Opt-in neural exists**: `NeuralSerialOrderRenderer` (rate-coded competitive-queuing serial-order, `neural_serial_order_renderer.py:50` `order()`), wired behind `enable_neural_render` (default **False** `brain_conversational_agent.py:206`), passed as `order_fn` to `render_fact`/`query_patient` (`:228,:249`). | HOST-SHORTCUT on the default path (opt-in neural version de-risked GO 6/6) | The default ordering is a host f-string. The neural competitive-queuing generator (premotor/SMA serial order, catalog G.07/H.19) is built + de-risked but off by default. The final `" ".join` of already-ordered, already-spelled words is legitimately the BODY. Fix = make `enable_neural_render=True` the production default; difficulty LOW. Honest scope: only the native SVO/clause frame order is covered; multi-frame (real syntax) is a separate capability. |
| 7 | **Per-slot spelling** (concept index → its word, A→W read-out) | NEURAL: the validated concept-pool→language_output A→W read-out (`concept_speak_demo`, 100% multi-seed), passed as the `spell` callback to the renderer. `neural_serial_order_renderer.py:60-63` | ALREADY-NEURAL ✓ (when neural render is on) | The spelling primitive is a separately-validated spiking read-out. On the default (f-string) path the word is just the stored label string; on the neural-render path it is spelled by the spiking A→W read-out. |
| 8 | **Dialogue-plan relevance** (which associate to bring up about a topic) | NEURAL: spreading activation on a spiking cortico-PFC loop; relevance = first-spike LATENCY. `content_selection_spiking.py:371` (`turn_latency`), `:389` (`relevance_by_latency`), graph embodied as inter-assembly synapses `:315` (`_install_graph_edges`). | ALREADY-NEURAL ✓ (relevance) | The relevance computation is genuinely spiking (spike-timing/latency = graph distance). |
| 8b | **Dialogue-plan winner-select + inhibition-of-return** | HOST: `min(cands, key=latency)` winner pick + `SaidTrace` (a numpy fading dict). `content_selection_spiking.py:385` (`min(...)`), `content_selection.py:58-73` (`SaidTrace`) | HOST-SHORTCUT (minor) | The winner-take-all over latencies and the repetition-suppression are host. Biological replacements exist: a spiking WTA for the pick, and spike-frequency adaptation on the selected assembly for inhibition-of-return (the module itself flags the latter as "the documented Milestone-3b step", `content_selection_spiking.py:286-287`). Difficulty MEDIUM; low capability impact. |
| 9 | **Dialogue-plan association graph** (concept→{concept:weight} the planner spreads over) | **HOST (DEFAULT)**: recomputed from `self.kb` as a Python co-occurrence dict. `brain_conversational_agent.py:263-271` (`_assoc_graph`), `rf_phasor_composer.py:524-534`. **Opt-in neural exists**: `LearnedAssocGraph` learns concept co-occurrence in a plastic Hebbian recurrent (CA3 autoassociator) and reads it back from `cp_connections`, gated by `enable_learned_assoc` (default **False** `brain_conversational_agent.py:197`). | HOST-SHORTCUT on the default path (opt-in neural version validated) | Default builds the graph from a Python dict; the substrate-learned version (`learned_assoc_graph.py:31` `store_fact`, `:53` `graph`) is built + validated (24/24 edges, 9/9 top associate) but off by default. Fix = enable `enable_learned_assoc`; difficulty LOW–MEDIUM. |
| 10 | **Anaphora resolution** (pronoun → most-recent / addressed referent) | NEURAL: order-encoded WM, each referent bound to a gamma-slot position phasor; pronoun resolves by spiking `unbind` of the slot, familiarity-gated. `ordered_position_wm.py:102` (`encode_sequence`), `:120` (`read_slot`), `multi_turn_agent_v2.py:92` (`most_recent_referent`), `:103` (`referent_at`) | ALREADY-NEURAL ✓ | The resolution is a genuine spiking unbind on the RF substrate; the moat is the neural familiarity gate (#5's good version). The surface bookkeeping (which token is an anaphor, the antecedent-slot dict in narration) is host control flow, not a cognitive computation. |
| 11 | **Reconsolidation gate** (prediction-error-gated in-place fact rewrite) | HOST: PE = `1 − cos(rec − concepts[patient])` then threshold; rewrite if `pe >= gate`. `rf_phasor_composer.py:345` (`_patient_prediction_error`), `:372` (`update_on_mismatch`), `:351` (`_calibrate_pe_labile`) | HOST-SHORTCUT (small; biologically-shaped) | The PE is a host cosine + a host comparison; the structure (mismatch-gated labilization, Nader 2000 / Osan-Tort-Amaral 2011) is biology-faithful but the subtraction/threshold is host. Same neural-replacement as #5 (a predictive-coding error population + a gate). Opt-in/additive (default append-only). Difficulty MEDIUM; low priority — explicitly an opt-in tier, and the owner permits lossy/learned memory. |
| 12 | **Multi-hop reasoning** (pointer-chase over stored facts) | NEURAL (inherits the substrate ops): iterates `query_patient`, each hop's match/abstain via spiking unbind. `rf_phasor_composer.py:470` (`query_chain`) | ALREADY-NEURAL ✓ (as much as #2–#5 are) | Pure composition of the substrate unbind + cleanup + abstain; adds no new host cognition beyond the per-hop pieces already classed above. The loop control (Python `for action in actions`) is body/orchestration. |
| 13 | **Reward / value** | N/A in the conversational path | DEFENSIBLE (absent) | No reward/value computation in conversation. The only reward-channel use is the learned-binder's read-out *training* (CYCLE 158, separate arc), where the `target` is an environment teaching scaffold — legitimate. |
| 14 | **Concept codes** (the `{word: phases}` vocabulary) | HOST-generated random phasors by default `rf_phasor_composer.py:84`; OR substrate-learned codes via `grounded_codes` / the PPMI stream cortex (`:90`). | DEFENSIBLE (the interface) / context-dependent | Random codes are a stand-in; the production direction (CLAUDE.md "stream cortex") *learns* codes from the conversation stream on the spiking substrate and supplies them via `grounded_codes`. The composer's code-injection interface is validated == random. Not a cognitive *operation* per se (it is the representation); flagged for completeness. Producing meaningful grounded codes is the documented open embodied-cognition problem, not a naked cheat. |

---

## Ranked summary — the genuine remaining host shortcuts in the COGNITIVE path

There are **5 genuine host shortcuts on the DEFAULT production path** where the cognitive work is host-computed
AND a neural replacement is not currently the default. Four of the five already have a validated/de-risked
neural version sitting behind a default-off flag — i.e. they are "flip-the-flag" biologizations, not open
problems. Ranked hardest-hitting first:

1. **Fact store = numpy list (`self.kb`)** — `rf_phasor_composer.py:99,333`.
   *The memory itself is host.* This is the most load-bearing host shortcut: every query indexes a Python list
   of numpy arrays. **Recommendation: BUILD NOW** by making `enable_substrate_store=True` the production
   default (the substrate weight-store `_store_substrate`/`_retrieve_substrate` is already validated == numpy).
   Verify the full conversational suite + latency cost before flipping. (Note: owner memory
   `feedback_moat_not_hard_lossy_memory_ok` permits lossy memory — so a learned/lossy store is also acceptable
   if it scales better.)

2. **Concept-code cleanup = numpy argmax** — `rf_phasor_composer.py:262-263`.
   The nearest-concept decision is the single most-called host cognitive op (every unbind ends in it).
   **Recommendation: BUILD NOW** — flip `enable_spiking_cleanup=True` (the `_spiking_cleanup` matched-filter +
   Izhikevich WTA is validated == numpy multi-seed). Lowest-risk highest-frequency win.

3. **Abstention / no-confab moat = host `==` + None** — `rf_phasor_composer.py:441,461,493`.
   The moat decision is host string-equality, not a neural familiarity signal — even though the validated
   neural familiarity gate already gates `OrderedPositionWM` and passed at V=320.
   **Recommendation: DEFER but build** — route the composer's match/abstain through the validated
   Bogacz-Brown familiarity-gate population. Medium effort; per owner this is a "plus, not a hard gate," so it
   ranks below the store/cleanup flips, but it is the most conceptually important to make neural for the
   "brain analogue" claim.

4. **Word-order emission = f-string** — `rf_phasor_composer.py:520`.
   **Recommendation: BUILD NOW** — flip `enable_neural_render=True` (the competitive-queuing
   `NeuralSerialOrderRenderer` is de-risked GO 6/6; the moat is unaffected). Trivial flip; the `" ".join` stays
   legitimately the body. (Multi-frame/real-syntax ordering is a separate *capability*, not this flip.)

5. **Dialogue-plan association graph = Python co-occurrence dict** — `brain_conversational_agent.py:263-271`.
   **Recommendation: DEFER** — flip `enable_learned_assoc=True` (the `LearnedAssocGraph` plastic CA3 recurrent
   is validated). Lower call-frequency (only `elaborate`), so lower priority than 1–4.

**Lower-tier host shortcuts (minor, biologically-shaped, low capability impact):** the parser role-decision
argmax (#1b), the dialogue-plan winner-select + `SaidTrace` inhibition-of-return (#8b), and the reconsolidation
PE-gate (#11) are all host read-outs/comparisons over already-neural computations, each with a known neural
analogue (spiking WTA; spike-frequency-adaptation; predictive-coding error population). Document-as-defensible
for now (read-out boundary), convert opportunistically.

## Already-neural ✓ (no action) and principled idealizations (document, don't "fix")

- **Already neural:** the parser comprehension mapping (#1, learned Hebbian), the bind/unbind/bundle *ops* on
  RF neurons + complex synapses (#2 ops), per-slot spelling A→W read-out (#7), dialogue-plan relevance by
  spike latency (#8), anaphora resolution by spiking slot-unbind (#10), multi-hop reasoning (#12).
- **Principled idealization (NOT a naked cheat):** the **exact-inverse FHRR vector algebra** (#2) is the
  Eliasmith Spaun / Semantic Pointer hypothesis — its operations are spiking; only the exact invertibility +
  clean-code demand is idealized. The **learned-binder arc** (`OnBridgeLearnedComposer`, CYCLE 149–160) is the
  incremental de-idealization; it has already removed the binder's read-out-training host optimizer on the
  substrate (real synaptic plasticity, 6-seed GO) and neuralised the per-output error — but it is **not yet
  wired into the production `RFPhasorComposer`**. Production-wiring that learned binder is the deep
  (non-flip) path; the five flips above are the cheap closeout of the "biologization" track on the *current*
  production composer.
- **Already-done (per the prompt; not re-flagged):** the learned-binder read-out LEARNING (host optimizer →
  bridge three-factor plasticity; per-output error → predictive-coding error population). Only the env
  `target` remains, which is a legitimate teaching scaffold.

---

### Anti-cheat / verification pointers for the controller
Every file:line in the table was read in full this session. The load-bearing default-OFF flags to verify:
`RFPhasorComposer.__init__` defaults `enable_spiking_cleanup=False` (`:62,77`),
`enable_substrate_store=False` (`:63,73`); `BrainConversationalAgent.__init__` defaults
`enable_neural_render` / `enable_learned_assoc` both False (`:153,197,206`). The opt-in neural implementations
to confirm exist: `_spiking_cleanup` (`:208`), `_store_substrate`/`_retrieve_substrate` (`:406,418`),
`NeuralSerialOrderRenderer.order` (`neural_serial_order_renderer.py:50`),
`LearnedAssocGraph.graph` (`learned_assoc_graph.py:53`), the validated familiarity gate in
`OrderedPositionWM.read_slot` (`ordered_position_wm.py:120-131`).
