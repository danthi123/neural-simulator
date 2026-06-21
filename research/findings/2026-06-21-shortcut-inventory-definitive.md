# Definitive shortcut inventory — the whole "one brain", two criteria (2026-06-21)

**Purpose.** The owner's 2026-06-21 reaffirmed top priority: close ALL host shortcuts AND ensure the one brain runs
**fully spiking end-to-end, compatible with future optimized (neuromorphic) hardware**. This supersedes/refreshes
`2026-06-20-shortcut-burndown-status.md`. READ-ONLY audit (this doc is the only write).

## The two criteria (every cognitive op gets BOTH tags)

Each operation between sensation and action is scored on two independent axes. Host code is legitimate ONLY for the
**environment** (world state + rendering the sensory image) and the **body** (acting on the motor output); everything
cognitive must be neural.

- **Criterion 1 — RUNTIME-SPIKING?** Does the OPERATION run as genuine spikes/synapses at runtime, or as a host
  Python/numpy computation? Tags: **YES** (spiking) / **HOST** (a host computation between sensation and action).
- **Criterion 2 — HARDWARE-PORTABLE STRUCTURE?** Is the op's STRUCTURE (its weights/connectivity) on-substrate, or
  host-computed-and-injected? A spiking op whose weights are computed in Python and injected is "spiking at runtime,
  host-designed at the structural level" — still a residual for a neuromorphic port (the chip would need a host to
  compute+inject them). Tags:
  - **ON-SUBSTRATE** — learned-on-substrate (STDP/Hebbian) OR developmentally self-organized from a local wiring rule.
  - **DEV-RANDOM** — a one-time genome-style random draw (`rng.uniform(seed)`), accepted as self-organized (the
    feedback-alignment precedent, `sim/dendritic_neuron.py:25`). Hardware-portable (a one-time on-device config).
  - **HOST-DESIGNED** — weights computed by a host formula and injected (a Gabor bank, a Gaussian place blob, a
    pop-vector geometry). NOT hardware-free.

**Status vocabulary.**
- **CLOSED-fully** — runtime-spiking YES + structure ON-SUBSTRATE/DEV-RANDOM, and it is the **default** path.
- **CLOSED-op / structure-residual** — runtime-spiking YES (default), but the structure is HOST-DESIGNED (criterion 2
  open) OR the spiking version exists but is **opt-in** (not the default), so the default path is still HOST.
- **OPEN** — a genuine host cognitive shortcut with no validated spiking replacement yet.
- **IN-FLIGHT** — a spiking replacement is built + validated and is being folded to default-on / multi-seed-confirmed.
- **BOUNDARY-characterized** — an honest negative that survived a surpass round (the spiking version was built +
  faithful-tested + failed). NOT "open" (no cheap close exists), but **still a host shortcut on the substrate** for the
  hardware-port goal (the host scaffold stays).

---

# (A) The production conversational pipeline

Files: `research/runners/one_brain_composer.py` (`OneBrainComposer`, the production default via
`BrainConversationalAgent(composer_kind="onebrain")`), `rf_phasor_composer.py` (`RFPhasorComposer`, the rf reference /
test oracle / numpy-CPU path + the per-op substrate), `brain_conversational_agent.py` (parser + agent wiring),
`nav_conv_merged_bridge.py` (the merged bridge).

| op | file:function | runtime-spiking? | structure | status | the cheap close (if open) | hardware-port note |
|---|---|---|---|---|---|---|
| **A1 Parse / comprehend** | `brain_conversational_agent.py:BridgeParser._train/role_of`; on the onebrain path `one_brain_composer.py:OneBrainComposer.hear` (its own `BridgeParser` slice) | **YES** — Hebbian co-firing on Izhikevich neurons; role read by firing | **ON-SUBSTRATE** — `enable_hebbian_learning` learns the (position×voice)→role map | **CLOSED-fully** | — | The teacher current during training is a host scaffold (training-time only, retired at inference). Robust multi-cue / case parsers (`MultiCueRoleParser`, `CaseAwareRoleParser`) are also spiking, install-path validities (continual on-substrate validity LEARNING is a deferred Tier-1 item). |
| **A2 Bind (role⊗filler)** | `rf_phasor_composer.py:_bind`; `one_brain_composer.py:_compose_phases` | **YES** — diagonal complex synapse (the role phasor) on resonate-and-fire neurons; the bind happens THROUGH `cp_rf_w_re/im` | **DEV-RANDOM** (role phasor = `rng.uniform(seed)`) | **CLOSED-fully** | — | The bind operation + structure are host-free (the role code is a one-time developmental draw installed as the synapse). |
| **A3 Bundle (superpose)** | `rf_phasor_composer.py:_bundle`; `one_brain_composer.py:_compose_phases` | **YES** — unit complex synapses sum on the substrate | **ON-SUBSTRATE** (unit weights) | **CLOSED-fully** | — | — |
| **A4 Unbind (conj·role)** | `rf_phasor_composer.py:_unbind_phases`; `one_brain_composer.py:_unbind_conj` (the 6 unbind-structure sites) | **YES** — conj diagonal complex synapse | **DEV-RANDOM when `local_reciprocal_unbind=True` (FHRR-B mechanism 1); else HOST-DESIGNED (host `np.conj` over the role code)** | **CLOSED-op / structure-residual (default)** | flip `local_reciprocal_unbind=True` (byte-identical, == conj for a unit phasor; derives the unbind synapse from the bind synapse by a local quadrature flip) | The flag exists + is byte-identical but **defaults OFF on every production path** (`OneBrainComposer.__init__` `local_reciprocal_unbind=False`); the default unbind STRUCTURE is the host `np.conj` — spiking at runtime, host-computed at the structural level. Validated `2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md`. |
| **A5 Cleanup codebook (matched filter)** | `rf_phasor_composer.py:_cleanup_conj/_spiking_cleanup`; `one_brain_composer.py:_cleanup_conj` | **YES** — `conj(concept)` complex synapse → rectified membrane scores | same as A4: **DEV-RANDOM/LEARNED codes when `local_reciprocal_unbind=True`; else HOST-DESIGNED (host `np.conj` over the concept code)** | **CLOSED-op / structure-residual (default)** | same flag (`local_reciprocal_unbind`) routes the cleanup codebook through the local rule | Same structure residual as A4 (host `np.conj` by default). The concept CODES themselves are LEARNED (PPMI stream cortex) or DEV-RANDOM — fine; the residual is the codebook *conj wiring*. `2026-06-20-FHRR-B-cleanup-codebook-local-conj.md`. |
| **A6 Cleanup SELECTION (winner-pick)** | `rf_phasor_composer.py:_cleanup`; `one_brain_composer.py:_select/_spiking_select` | **HOST argmax by default; YES spiking Izhikevich WTA when `enable_spiking_cleanup=True`** | bank weights ON-SUBSTRATE (NEF) | **CLOSED-fully on the flagship 320 demo; CLOSED-op (opt-in) on the library default** | the spiking WTA exists; turn `enable_spiking_cleanup=True` (== argmax multi-seed @ D=2048) | The flagship `consolidated_320_conversation_demo --composer onebrain` (CYCLE 190) sets it ON → spiking. The **library constructor default** (`BrainConversationalAgent`/`OneBrainComposer` `enable_spiking_cleanup=False`) keeps the host argmax (the numpy-CPU + test-oracle path). `2026-06-05-composer-cleanup-NEF-GO.md`; burndown #1 `69fd355d`. |
| **A7 Persistent fact STORE** | `one_brain_composer.py:_write_block/store_conns`; `rf_phasor_composer.py:_store_substrate` (opt-in) | **YES** — each fact = a trigger→readout (1+D) complex-weight block in `cp_rf_w_re/im` (memory-in-synapses, Crawford-Eliasmith) | **ON-SUBSTRATE** (the composite phasor is the synapse) | **CLOSED-fully** | — | The onebrain default holds the store in synapses (the `kb` list is bookkeeping only — fact dicts for routing, with a `None` vector placeholder). Persistent to K=32. |
| **A8 Cue-match SCAN + first-match routing** | `one_brain_composer.py:_seq_block/_scan/query_agent/ask_yes_no/render_fact/count_facts/_find_cued_block`; `rf_phasor_composer.py:_scan_first_match/_iter_facts` | **per-block reconstruction (`_read_blocks`) is YES spiking; the WHICH-block-answers routing is HOST Python `==` loops by default. The (agent,action) hot-path routes through a spiking K-way sequencer when `integrated_loop=True`** | sequencer weights ON-SUBSTRATE (gated-disinhibition cascade + BG WTA) | **IN-FLIGHT** | fold the validated K-way sequencer (S0–S2 GO; K=32 GO at the production `match_thresh=0.06`, `2026-06-21-shortcut3-K32-capability-surpass.md`) into the composer default (S3) + scale to 320 (S4) | `integrated_loop` **defaults OFF** on every production path, and even ON it covers only the (agent,action) hot-path sites; `query_agent` (action,patient), `ask_yes_no` (full SVO), `render_fact`/`describe` (agent-only), the general `_scan` stay host (a swapped-cue + 1-role cascade, named bounded follow-ons). **This is the single largest live conversational host residual.** |
| **A9 Abstain / no-confab moat** | the `return None`/`"unknown"` in A8 (`_scan`, `query_*`, `ask_yes_no`); `confidence_gate` margin | inherits A8 (HOST `==`/`is None` by default; spiking via the sequencer's abstain channel when `integrated_loop`) | — | **IN-FLIGHT (with A8)** | the sequencer's abstain channel maps to the same `None`/"unknown" (0 false-accept on absent/cross cues; the absent cue WORD is caught before the sequencer) | The moat is a host check by default. The Bogacz-Brown familiarity gate is the validated neural abstention (`2026-06-11-familiarity-gate-v320-GO.md`) but is not wired into the production composer's moat. **Do not weaken the moat to close this.** |
| **A10 Negation / yes-no** | `one_brain_composer.py:ask_yes_no` (the bound AFFIRM/NEGATE polarity role) | **YES** for the bind+unbind+cleanup of the polarity tag; the yes/no/unknown decision is the HOST scan (A8) | DEV-RANDOM polarity codes | **CLOSED-op (rides A8)** | closes with A8 | The polarity role is a genuine 4th bind (spiking); only the match-and-branch is the A8 host residual. |
| **A11 Describe / render (word ORDER)** | `rf_phasor_composer.py:_render/render_fact`; `one_brain_composer.py:render_fact/_decode_clause`; `neural_serial_order_renderer.py` | **content decode YES spiking; the word-ORDER join is HOST f-string by default, spiking competitive-queuing when `order_fn` is passed** | CQ generator weights ON-SUBSTRATE | **CLOSED-op (opt-in; default-on in the agent)** | the agent passes `order_fn` when `enable_neural_render=True` (the agent default) → the SVO order is the spiking serial-order read-out | `BrainConversationalAgent(enable_neural_render=True)` is the agent default, so `describe()`'s SVO order is neural; the composer's own `render_fact`/`_render` default `order_fn=None` (host f-string) for the embedded-clause/Q&A path. Remaining host orders (adjective-noun, embedded-clause, dialogue replies) + multi-frame order-learning = bounded follow-ons. `2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`. The final string JOIN (the body's emission) is legitimately host. |
| **A12 Reason / multi-hop chain** | `*_composer.py:query_chain`; `brain_conversational_agent.py:reason_chain` | inherits A8 (iterates `query_patient`) | — | **IN-FLIGHT (rides A8)** | closes with A8 | The pointer-chase logic is the host loop over A8; the per-hop unbind+cleanup is spiking. `2026-06-17-multihop-query-chain-GO.md`. |
| **A13 Dialogue-plan / elaborate** | `*_composer.py:_assoc_graph/elaborate`; `brain_conversational_agent.py:_assoc_graph/elaborate`; `content_selection_spiking.py:SpikingSpreadingController` | **the spread/selection is YES spiking (the dlPFC loop-attractor Control); the association GRAPH is a HOST Python dict built from `kb` by default (substrate-learned when `enable_learned_assoc=True`)** | Control weights ON-SUBSTRATE; the graph edges HOST-DESIGNED by default / ON-SUBSTRATE (Hebbian CA3) when opt-in | **CLOSED-op / structure-residual (default)** | turn `enable_learned_assoc=True` → the graph is a substrate-learned sparse Hebbian recurrent (24/24 edges, 9/9 top associate, multi-seed) | The selection mechanism is spiking; the graph CONTENT is recomputed from the Python `kb` dict on the default path — a structural host residual. Default OFF. |
| **A14 Multi-turn anaphora / context** | `multi_turn_agent.py:MultiTurnAgent` + `SpikingLoopContextBuffer` | **YES** — a persistent spiking loop-attractor holds discourse referents across turns | ON-SUBSTRATE | **CLOSED-fully** | — | The pronoun→referent bind is the spiking buffer; reset/lesion break it (de-risked GO). Multi-REFERENT disambiguation (which of several held referents a bare pronoun binds) needs WTA biased-competition — a specified next mechanism, not on the critical path. `2026-06-17-multiturn-anaphora-derisk-GO.md` / `-multireferent-disambiguation-NEGATIVE.md`. |
| **A15 Reconsolidation** | `*_composer.py:update_on_mismatch/_calibrate_pe_labile` | the unbind/PE/rewrite are YES spiking; the find-cued-block rides A8 (host by default); the labilization gate is a host midpoint statistic | ON-SUBSTRATE | **CLOSED-op (rides A8)** | closes with A8 | The prediction-error computation + in-place rewrite are on-substrate; the gate calibration is a frozen data statistic (a teaching scaffold, not an inference shortcut). `2026-06-17-reconsolidation-update-derisk-GO.md`. |

**Concept codes (the binding representations).** Not a separate op, but load-bearing for criterion 2: the production codes
are **LEARNED from conversation** (the PPMI stream cortex, on the real spiking substrate; `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`) or **DEV-RANDOM** (`rng.uniform(seed)` defaults). Both are
hardware-portable. The bind FORM (exact-inverse FHRR algebra) is the **principled idealization / step-3 frontier** (FHRR-B
below) — not a host runtime shortcut (the ops are spiking), but a residual idealization of a learned cortex.

---

# (B) The merged navigation loop

Files: `research/runners/g11_bg_runner.py` (`build_bg_brain_regions`, `run_moving_goal_episode`, `install_spiking_sc_wiring`),
`nav_conv_merged_bridge.py` (`build_merged_nav_conv_bridge`, `MergedNavConvAgent`, `co_resident_*` flags).

> **Critical default-vs-opt-in distinction (load-bearing for this whole table).** The spiking conversions for reward (B2),
> value/critic (B3), dopamine-RPE (B4), and self-org place (B6) are **built + validated at the mechanism level** but are
> **opt-in flags that default OFF** in `run_moving_goal_episode` (`spiking_reward_us=False`, `enable_neural_critic=False`,
> `perceived_approach_reward=False`) AND in the production merge (`co_resident_nav_critic=False`, `nav_critic_spiking_sc=False`,
> `nav_critic_place_selforg=False`, `co_resident_limbic=False` in `build_merged_nav_conv_bridge`/`MergedNavConvAgent`). So the
> **default deployed nav loop runs the host-formula reward/RPE/place path.** This is the drift the prior burndown-status doc
> obscured by marking #7/#8 "CLOSED (converted)" without noting they are not default-on.

| op | file:function | runtime-spiking? | structure | status | the cheap close (if open) | hardware-port note |
|---|---|---|---|---|---|---|
| **B1 Perception (V1→V2→IT)** | `g11_bg_runner.py:~2509-2662` (visual hierarchy regions/pathways); `sim/visual_cortex.py:build_v1_simple_weights` | **YES** — 5-level spiking Izhikevich hierarchy (retina→V1_simple→V1_complex→V2→cortex_it) | **HOST-DESIGNED at V1 (the retina→V1_simple Gabor bank, computed by `gabor_kernel` + injected); LEARNED above (V1c→V2→IT STDP)** | **CLOSED-op / structure-residual** | none cheap — the Gabor RFs are a one-time host formula. (Biologically standard, and downstream layers learn.) | The Gabor weights are a deterministic host formula injected at init — a neuromorphic port would need a host to compute the Gabor bank (weaker than DEV-RANDOM). The retina IMAGE itself is the legitimate environment/sensory render. |
| **B2 Reward** | `g11_bg_runner.py:1208-1227` (`reward_us` pop), `~2606-2622` (`sc_rostral`→reward), `~7148/7540` (delivery); host write `snc += snc_reward_gain·max(0,reward)` at `~1214` | **HOST formula by default (a Python scalar from a Manhattan/beacon/`sc_rostral`-rate read, written as DA current); YES spiking when `spiking_reward_us=True` (a PPN-like `reward_us` US→SNc glutamate burst)** | the `sc_rostral` proximity pool geometry is HOST-DESIGNED (a Gaussian pool); `reward_us` synapse ON-SUBSTRATE | **CLOSED-op (opt-in) — default is HOST** | turn `spiking_reward_us=True` + `perceived_approach_reward=True` (the coord-free N5 approach reward fires `reward_us` into the SNc) | The spiking US→SNc reward burst is VALIDATED (`2026-06-10-N5-reward-CLOSED…`, `2026-06-10-N9-fully-spiking-reward-loop-MILESTONE.md`: Pavlovian probe bursts the SNc) but **defaults OFF**; the deployed reward is a host scalar. The reward MAGNITUDE source (proximity) is still a host-designed pool even when spiked. |
| **B3 Value / critic V(s)** | `g11_bg_runner.py:1229-…` (`striosome_value` MSN critic), `~1911` (the plastic `vs_place_context→striosome_value`), `~1958` (`striosome_value→snc` GABA_B); `--dendrite-critic` (`~4385`) | **HOST (no critic) by default; YES spiking critic pool when `enable_neural_critic=True` — the value is the MSN-D1 firing read by the SNc via GABA_B subtraction; the GRADED value via the dendritic plateau when `--dendrite-critic`** | the value WEIGHT is LEARNED (dopamine-gated STDP); the critic's AFFERENT is the **HOST-DESIGNED Gaussian place code** (`vs_place_context`, B6) by default | **CLOSED-op (opt-in) — default is HOST/absent; the GRADED read-out deploy is IN-FLIGHT** | `enable_neural_critic=True` (the r−V subtraction is then synaptic GABA_B) + `--dendrite-critic` for the graded V (δ=1.33, 6/6 seeds — supersedes the earlier "characterized dendritic boundary") | **#9 reclassification:** the graded point-neuron value read-out IS realizable via the on-substrate dendritic plateau (`enable_graded_dendritic_plateau`, a `sim/`-shipped dendritic compartment = a legitimate substrate mechanism, NOT a host shortcut), δ=1.33 6/6 (`2026-06-20-dendrite-stage1-snc-calibration.md`). The production multi-seed nav table is PENDING (`2026-06-20-shortcut9-dendrite-critic-deploy.md`). The earlier "characterized DENDRITIC boundary" verdict is OVERTURNED by the dendrite. The afferent (B6) remains a host-designed input on the default path. |
| **B4 Dopamine / RPE (SNc)** | `g11_bg_runner.py:1183-1206` (`snc` pool, `IZH2007_DOPAMINE`), `~1214` host write, `~1958` critic GABA_B | **HOST formula by default (the scalar `reward` writes DA current); YES spiking when `spiking_reward_us` + `enable_neural_critic` (δ=r−V = `reward_us` excitation − `striosome_value` GABA_B, both synaptic)** | SNc pool ON-SUBSTRATE; the r and V terms inherit B2/B3 | **CLOSED-op (opt-in) — default is HOST** | the same flags as B2+B3 (then the whole δ=r−V is two spiking populations) | The fully-spiking δ=r−V is VALIDATED end-to-end in the real config (`2026-06-10-N9-fully-spiking-reward-loop-MILESTONE.md`, Stage-B GO: graded δ, lesion-confirmed synaptic) but **defaults OFF**. The TD cue-shift temporal-credit piece is a BOUNDARY (`2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md`). |
| **B5 Orienting (superior colliculus read-out)** | `g11_bg_runner.py:201-338` (`install_spiking_sc_wiring`), `sc_map→cortex_{N,E,S,W}` read-out | **YES bump (sc_retina→sc_map→sc_fs Mexican-hat spiking); the orienting decision READOUT is the host-designed read-out weights → spiking cortex; the deployed DEFAULT orienting is the HOST heuristic (Manhattan centroid+argmax)** | the `sc_map→cortex` read-out weights are HOST-DESIGNED (a half-plane ramp OR a pop-vector geometry, injected) | **BOUNDARY-characterized** | none — the spiking read-out was built (pop-vector cosine + bump-mass divnorm + the #4 WTA ring) + faithful-tested across 7 mechanism variants → does NOT re-orient at grid-32; the residual is the actor's cascade N-bias (out of #6's read-out scope) | The spiking SC read-out is a comprehensively-characterized HONEST NEGATIVE at faithful grid-32 (`2026-06-20-shortcut6-nav-orienting-CLOSE.md`, supersedes the prior "closed" ledger row + the grid-8 false-GO). **The host orienting scaffold STAYS** — so a host shortcut remains on the substrate. Not "open" (no cheap close), but still un-converted. |
| **B6 Place / position code** | `g11_bg_runner.py:82-121` (`_n9_place_sensor_act`), `~1298` (inject each step); self-org: `~1175/~1783` (`place` pool, `place_sensors→place` threshold-WTA + plastic `place→striosome_value`) | **HOST formula by default (a Gaussian bump from (x,y), written as `cp_external_input_current` — "(x,y) enters the brain ONLY here, the position-leak boundary"); YES spiking self-org place cells when `neural_place_selforg=True` (with `enable_neural_critic`)** | HOST-DESIGNED (the Gaussian blob) by default; ON-SUBSTRATE (learned place fields) when self-org | **CLOSED-op (opt-in) — default is HOST; the self-org default-on is a characterized BOUNDARY** | `nav_critic_place_selforg=True` routes the critic afferent through the self-org `place` pool | **#5 status:** the self-org spiking place code is built (a-GO at the mechanism level) but the **sparsify-to-default** is a BOUNDARY (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`); the default position code is the host Gaussian (own-position is legitimate egocentric self-knowledge, but it is host-RENDERED, not computed by neurons). |
| **B7 Action selection (BG cascade)** | `g11_bg_runner.py:343` (`build_bg_brain_regions`), `~2163` (sel_X NMDA accumulators), `~2200` (commit_X burst), `~7280-7330` (the decision read) | **YES — the per-action cortex→D1/D2→GPi→thal→sel→commit cascade is fully spiking; the DECISION is the commit_X burst threshold-crossing (Lo-Wang); the host `max()` over commit_counts is an OBSERVER (loser counts ~0 under decisive commit; 100% commit-burst, 0 fallback)** | **ON-SUBSTRATE** — dopamine-gated STDP throughout the striatal cascade | **CLOSED-op / characterized residual (default-on)** | none needed — this IS default-on (`readout_source="spiking_wta"` is the library default, `2026-06-19-spiking-decision-default-on-GO.md`) | The spiking commit-burst is the LIBRARY default (1.16× host, 100% commit-burst, the ~16% residual = the irreducible finite-size/commit-timing floor = the honest brain-based deliverable). The host `max()` is a tie-break of last resort (never used under a decisive commit). The CLI `--readout-source` default stays `"motor"` (the host-argmax ORACLE) for benchmark-repro — a LEGIT override, not a cognitive shortcut. |
| **B8 Language→action route** | `spoken_instruction_nav.py` (parser firing opens a `command_route` gate → learned word→action route) | **YES** | ON-SUBSTRATE (learned word→action) | **CLOSED-fully** (opt-in cross-region feature) | — | `2026-06-10-spoken-instruction-nav-GO.md`. |
| **B9 Perception→memory / grounded compose** | `navigate_to_compose_then_answer.py` (live `cortex_it` rate → grounded phasor → the rf composer) | **YES** — a fixed complex projection grounds the live spiking percept into the phasor algebra | the grounding projection is HOST-DESIGNED (a fixed complex matrix `M`) | **CLOSED-op / structure-residual** (opt-in feature) | — | `2026-06-16-navigate-to-compose-then-answer.md`. The projection matrix is host-fixed (a one-time config). |

---

# Top-down ranked OPEN list (highest-leverage / most-load-bearing first)

Separating genuine OPEN/IN-FLIGHT shortcuts (a cheap close exists) from characterized BOUNDARIES (no cheap close; the host
scaffold stays). The hardware-port goal means a "spiking-but-host-structured" op (criterion 2 open) is also listed.

### Tier 1 — live host cognitive shortcuts with a validated cheap close (IN-FLIGHT or flip-a-flag)

1. **A8/A9/A12/A15 — the conversational cue-match SCAN + first-match routing + the moat (host Python `==` loops).**
   The single largest live conversational host residual: `_scan`, `query_agent`, `ask_yes_no`, `render_fact`, `count_facts`
   all branch on host `got.get("agent") == agent`. **Cheap close (IN-FLIGHT):** fold the validated K-way sequencer
   (`integrated_loop=True`, K=32 GO at `match_thresh=0.06`, `2026-06-21-shortcut3-K32-capability-surpass.md`) into the
   composer default (S3) and scale to 320 (S4); then extend from the (agent,action) hot-path to the swapped-cue
   (`query_agent`) + 1-role (`render_fact`) cascades. The moat closes WITH it (the sequencer's abstain channel; 0 FA).
   Do NOT weaken the moat.

2. **B2+B3+B4 — flip the nav limbic core (reward / value / dopamine-RPE) to default-on.** All three spiking conversions
   are VALIDATED at the mechanism level (the fully-spiking δ=r−V, `2026-06-10-N9-fully-spiking-reward-loop-MILESTONE.md`)
   but default OFF; the deployed nav loop runs host scalars. **Cheap close:** make `co_resident_nav_critic`/
   `spiking_reward_us`/`enable_neural_critic`/`perceived_approach_reward` the default on the production merge (+ multi-seed
   confirm). **Caveat (the reason it has not been flipped):** on the standard orient-solvable gridworld the limbic core is
   GREEN_INERT (validated spiking but behaviorally inert — the agent navigates by perception, reward never changes
   behavior; `2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md` NEGATIVE), so flipping it on is brain-based
   purity, not a behavior win. Still required for the "fully spiking end-to-end" goal.

3. **A4/A5 — the unbind + cleanup-codebook conj STRUCTURE (host `np.conj` by default).** Runtime-spiking YES, but the
   structure is host-computed `np.conj` on every production path. **Cheap close (flip a flag):** set
   `local_reciprocal_unbind=True` on the production `OneBrainComposer` (byte-identical; derives the unbind/cleanup synapse
   from the bind synapse by a local quadrature-flip wiring rule). This is the criterion-2 (hardware-port) close the FHRR-B
   mechanism-1 work already built; it just defaults OFF.

4. **B3 #9 — finish the dendrite graded-value deploy (multi-seed nav table PENDING).** The graded value read-out is
   mechanism-validated (δ=1.33, 6/6); the production-nav 3→6-seed table is in flight (`2026-06-20-shortcut9-dendrite-critic-deploy.md`).
   This OVERTURNS the prior "characterized dendritic boundary" — close it by landing the table + (if GO) flipping default-on.

5. **A6 — cleanup SELECTION on the library default.** Spiking WTA is ON in the flagship 320 demo but OFF in the library
   constructor default. **Cheap close:** make `enable_spiking_cleanup=True` the library default (== argmax multi-seed),
   keeping a `False` escape for the numpy-CPU + test-oracle path.

6. **A13 — the dialogue-planning association GRAPH (host dict by default).** The spread is spiking; the graph CONTENT is a
   host Python dict. **Cheap close:** make `enable_learned_assoc=True` the default (the substrate-learned sparse Hebbian
   recurrent, multi-seed-validated 24/24 edges).

7. **A11 — remaining host word-orders.** The SVO order is neural (agent default `enable_neural_render=True`); the
   adjective-noun, embedded-clause, and dialogue-reply orders + multi-frame order-learning are bounded follow-ons.

### Tier 2 — characterized BOUNDARIES (honest negatives that survived a surpass round; the host scaffold stays — still a residual for the hardware-port goal, NOT "open")

8. **B5 #6 — the spiking SC orienting read-out.** Comprehensively-characterized HONEST NEGATIVE at faithful grid-32 across
   7 mechanism variants (`2026-06-20-shortcut6-nav-orienting-CLOSE.md`). The residual is the actor's cascade N-bias
   (out of #6's read-out scope). The host orienting scaffold stays. **Not open** (no cheap close), but un-converted.
   The next direction (correcting the cascade N-bias) is a *different* shortcut.

9. **B6 #5 — the self-org place code as default.** The self-org spiking place code is built (a-GO at the mechanism level)
   but sparsify-to-default is a BOUNDARY (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`). The host Gaussian place
   blob stays the default afferent.

10. **B4 — the TD temporal-credit (cue-shift).** The merged TD cue-shift op-search is a BOUNDARY
    (`2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md`). The immediate δ=r−V is spiking; the temporal-difference
    backup across the delay is the un-closed piece.

11. **FHRR two-attribute (F=3 resonator) bind.** The single-attribute bind generalizes + is substrate-validated; the
    two-attribute bundling is NEGATIVE on the correlated learned codes (~29%; the K=5-load boundary) — deliberately not
    wired. A characterized capability boundary.

### Tier 3 — the deep frontier (the principled idealization; owner-sequenced LAST)

12. **FHRR-B — the exact-inverse VSA bind algebra → a learned cortical binder.** The bind OPERATIONS are spiking (A2–A5)
    and the structure can be made host-free (Tier-1 #3 above), but the bind FORM is the exact-inverse FHRR algebra (the
    principled Spaun/SPA idealization), not a learned lossy cortex. The learned-binder arc is NEGATIVE so far (memorizes,
    doesn't generalize on correlated codes; single-attribute generalizes, multi-attribute bundling needs the dendritic
    substrate). This is the step-3 frontier, owner-sequenced after all other shortcuts. The fork
    (`docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`): a semantically-FLAT learned cortex is
    achievable now; a semantically-STRUCTURED one that generalizes is the deepest open problem.

13. **Tier-2 limbic→composer integration.** The nav limbic core and the conversational composer are co-resident but the
    limbic value/reward does not yet GATE the composer (encoding-gain is a built opt-in, `2026-06-19-dopamine-encoding-gain-derisk.md`;
    the full integration is scoped, `2026-06-19-tier2-limbic-to-composer-scoping.md`, not built). This is a NEW
    capability/integration, not a host-shortcut close per se — listed for completeness as the Tier-2 "TRUE ONE BRAIN" item.

---

# Completeness check — shortcuts NOT previously tracked in the burndown-status doc

The prior `2026-06-20-shortcut-burndown-status.md` tracked a numbered list (#1–#12 + FHRR-B) but did NOT separately
surface these (the criterion-2 / hardware-port lens makes them visible):

- **(NEW) A13 — the dialogue-planning association GRAPH is a host Python dict by default.** Not in the burndown list. The
  spread is spiking (the dlPFC Control), but the graph content is recomputed from `kb` unless `enable_learned_assoc=True`.
  A structural host residual.
- **(NEW) B1 — the V1 Gabor read-out weights are host-designed.** Perception is spiking and was treated as "defensible,"
  but under criterion 2 the retina→V1_simple Gabor bank is a host formula injected at init — a hardware-port residual
  (weaker than DEV-RANDOM; a deterministic host formula, not a self-organized draw). Biologically standard; downstream
  layers learn. Worth listing as a known structure-residual.
- **(NEW, sharpened) A4/A5 — the conj STRUCTURE residual is on the PRODUCTION default, not just the rf reference.** The
  burndown noted FHRR-B mechanism-1 closed the unbind conj on "the production-default one-brain path," but the
  inventory's criterion-2 audit shows `local_reciprocal_unbind` **defaults OFF** on `OneBrainComposer`, so the shipped
  default STILL uses host `np.conj` for BOTH unbind (A4) and the cleanup codebook (A5). The flag exists; it is not the
  default. (This is the "spiking-at-runtime, host-designed-structure" case the owner's hardware-port lesson targets.)
- **(re-surfaced) A6 / A11 / A13 / B2 / B3 / B4 / B6 — the default-vs-opt-in gap.** The burndown marked several of these
  "CLOSED (converted)," but the inventory shows the spiking version is **opt-in and not the default** (the flagship demo
  flips some; the library constructor + the production merge default to host). "Closed" should require default-on.

---

# Counts (two-criteria)

Counting the 15 conversational ops (A1–A15, with the codes note) + 9 navigation ops (B1–B9) = 24 cognitive ops.

| Bucket | Count | Ops |
|---|---|---|
| **CLOSED-fully** (spiking + on-substrate/dev-random, default) | **7** | A1, A2, A3, A7, A14, B7, B8 |
| **CLOSED-op but host-structure / opt-in-not-default** (criterion 2 open OR spiking version not default) | **11** | A4, A5, A6, A10, A11, A13, A15, B1, B2, B3, B9 (B2/B3 also gate B4) |
| **OPEN** (genuine host shortcut, no validated default close — none purely, all live ones have a flip/fold) | **0 purely** | (the live host residuals are all IN-FLIGHT or flip-a-flag; see Tier-1) |
| **IN-FLIGHT** (validated spiking replacement being folded to default) | **3** | A8, A9, A12 (the cue-match scan + moat + multi-hop; the #3 sequencer fold) |
| **BOUNDARY-characterized** (honest negative, host scaffold stays) | **3** | B4 (TD cue-shift), B5 (#6 SC orienting), B6 (#5 place sparsify-default) |

(Plus the Tier-3 frontier: FHRR-B / the learned cortex (idealization, owner-sequenced last) + the two-attribute bind
boundary + the limbic→composer integration. B7 carries a characterized irreducible commit-timing residual but is
default-on and counted CLOSED-op.)

**One-line summary:** the heavy cognition on BOTH halves is genuinely spiking (parse, bind, bundle, store, accumulate-
then-commit selection, multi-turn context). The live work is (1) **fold the #3 sequencer** so the conversational
cue-match routing + moat are spiking (the largest live residual, IN-FLIGHT), (2) **flip the nav limbic core + the
`local_reciprocal_unbind`/`enable_spiking_cleanup`/`enable_learned_assoc` flags to default-on** (validated, but opt-in),
and (3) **land the #9 dendrite-value deploy table** (overturns the old boundary). Three genuine BOUNDARIES remain (SC
orienting, place-sparsify, TD cue-shift) where the host scaffold stays — still residuals for the neuromorphic-port goal,
not cheap-closable.

---

# Drift vs the prior burndown-status doc (`2026-06-20-shortcut-burndown-status.md`)

| Item | Burndown-status said | This inventory finds | Drift |
|---|---|---|---|
| #1 cleanup argmax (A6) | ✅ CLOSED (converted), OneBrain default | Spiking WTA is the **flagship-demo** default but the **library constructor** default is host argmax (`enable_spiking_cleanup=False`) | Partial — closed on the flagship, opt-in on the library. |
| #5/#7 reward (B2) | ✅ CLOSED (converted) — "neural reward population" | The spiking US→SNc reward is VALIDATED but `spiking_reward_us`/`perceived_approach_reward` **default OFF**; the deployed reward is a host scalar | **Drift — listed closed; is opt-in, not default-on.** |
| #8 value/critic (B3) | ✅ CLOSED (converted) — "competitive spiking value, δ 1.23×" | `enable_neural_critic` **defaults OFF**; default nav has no critic; the graded dendrite deploy table is PENDING | **Drift — listed closed; is opt-in + the deploy is in-flight.** |
| #9 value critic | ✅ CLOSED (characterized DENDRITIC boundary) — "graded point-neuron critic NOT realizable" | **OVERTURNED:** the dendrite graded plateau DOES realize the graded V (δ=1.33, 6/6); deploy multi-seed PENDING | **Drift — the boundary is overturned by the dendrite; reclassify to IN-FLIGHT deploy.** |
| #4 motor read-out (B7) | ✅ CLOSED (characterized), default-on | Confirmed — `readout_source="spiking_wta"` is the library default, 100% commit-burst | No drift (accurate). |
| #6 SC orienting (B5) | ✅ CLOSED (characterized honest-negative) | Confirmed as a comprehensively-characterized HONEST NEGATIVE (7 variants, grid-32) — the host scaffold stays | No drift, but note it is STILL a host shortcut on the substrate (a residual for the hardware-port goal), not "done." |
| #3 sequencer (A8) | 🔄 IN PROGRESS (S2 in flight) | Confirmed IN-FLIGHT; K=32 now GO at the production threshold (`2026-06-21`), fold (S3)+scale (S4) pending | Updated — K=32 capability is GO; the fold is the remaining work. |
| FHRR-B unbind conj (A4) | (mechanism 1 noted as closing "the production-default one-brain path") | The flag (`local_reciprocal_unbind`) **defaults OFF** on `OneBrainComposer`; the shipped default uses host `np.conj` for unbind AND cleanup-codebook | **Drift — the structure close is built but NOT the default; criterion-2 residual remains on the shipped path.** |
| (not listed) A13 assoc graph, B1 Gabor weights | — | Newly surfaced structure-residuals (host dict / host Gabor bank) | New (completeness win). |

**Net:** the prior doc over-counted "CLOSED" by conflating "a validated spiking version exists" with "it is the default
path." Under the two-criteria lens (runtime-spiking AND hardware-portable-structure AND default-on), the genuine state is:
7 fully closed, 11 spiking-but-residual/opt-in, 3 in-flight, 3 characterized boundaries. The fastest brain-based-purity
gains are the **default-on flips** (limbic core, `local_reciprocal_unbind`, `enable_spiking_cleanup`, `enable_learned_assoc`)
and the **#3 sequencer fold** — none requiring new mechanisms.
