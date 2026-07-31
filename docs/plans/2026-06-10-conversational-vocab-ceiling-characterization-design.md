---
type: plan
status: live
date: 2026-06-10
---

# Conversational vocab + capability ceiling characterization — design (2026-06-10)

**Status:** design only (read-only research pass produced it). Nothing built yet.

**One line:** Measure how the project's now-much-more-biological *consolidated conversational agent* (parser +
RF phasor composer + dlPFC dialogue planner, the whole loop) performs as the vocabulary grows from the V=16
probe set to 64 → 128 → 320 concepts, multi-seed — mapping the **biological conversational ceiling** (a
first-class project deliverable: an honest negative under strict biology *is* the science) and **sharpening the
deferred "true cortex" arc** (it tells the learned cortex exactly what it must add). This is mostly
*characterization* — re-running existing, validated machinery at larger scale — not new building.

**Where this sits in the roadmap.** The navigation brain and the conversational brain were just consolidated
onto one `SimulationBridge` (`MergedNavConvAgent`, STEP 2). The next deep arc is **Step 3 — replacing the
composer's exact-inverse VSA algebra idealization with a learned spiking-cortical binding** (the deepest,
highest-variance open problem, explicitly DEFERRED). Before paying for Step 3, this characterization arc:
(1) establishes the ceiling the idealized algebra reaches on the *consolidated substrate + full agent loop*,
and (2) localizes exactly which capabilities degrade and at what scale — the spec the learned cortex inherits.

---

## 0. Terms (defined once)

- **VSA (Vector Symbolic Architecture):** a scheme that represents structured knowledge ("dog go north") as
  high-dimensional vectors, with a *bind* operation (combine a role with a filler, e.g. agent⊗dog) and a
  *bundle* operation (superpose several bindings into one vector). *Unbind* (the approximate inverse of bind)
  recovers a filler given the role; a *clean-up* step snaps the noisy recovered vector to the nearest known
  concept code.
- **FHRR (Fourier Holographic Reduced Representation):** the VSA variant the production composer uses. A
  concept is a vector of **phases** (angles); bind = adding phases, unbind = subtracting them. Because every
  component has unit magnitude (information is in the *phase*, not the amplitude), there is no "common mode" to
  remove — which is why FHRR sidesteps the rate-coded opponency SNR wall that blocked the older composer.
- **RF (resonate-and-fire) neuron:** the spiking neuron model (`NeuronModel.RESONATE_AND_FIRE`) that holds a
  complex phasor in its state and is the substrate the production composer's bind/unbind run on, via complex
  synapses (`rf_set_complex_weights`, `rf_kick`, `rf_read_phases`).
- **D (dimension):** the number of phasor components per concept. Larger D → more signal-to-noise (SNR) per
  unbind, at linear time/memory cost. The production agent uses **D=128**.
- **Capability matrix:** the seven conversational capabilities the agent is validated on — who-Q&A, what-Q&A,
  abstention (the **no-confab moat**: return `None`/`unknown` when no stored fact matches), negation/yes-no,
  embedded clauses, one-attribute entities ("big apple"), and dialogue planning (`elaborate`). Generation
  (`describe`) is an eighth.
- **Two-attribute / K=5:** a fact whose patient has two adjectives ("big hot apple"). This binds five roles
  (agent, action, patient, attribute, attribute2). The older ±1 rate composer could not do it at all; the
  documented "K=5 boundary" is where that scheme topped out. Whether the FHRR composer lifts it on the agent is
  one of the two headline questions.
- **The consolidated agent:** `MergedNavConvAgent`
  (`research/runners/nav_conv_merged_bridge.py`) — parser + dlPFC on ONE merged nav+conv bridge; fact
  storage/retrieval delegated to an `RFPhasorComposer`. Its lighter sibling is `BrainConversationalAgent`
  (`research/runners/brain_conversational_agent.py`) — a standalone parser bridge + the same composer, no
  navigation co-residence.

---

## 1. What exactly to test, and why each matters

### The motivating gap (verified in code)

- The consolidated agent runs at the **V=16 probe vocabulary**. `RFPhasorComposer.DEFAULT_VOCAB` is exactly 17
  words (`rf_phasor_composer.py:25`); `build_merged_nav_conv_bridge` defaults `vocab=DEFAULT_VOCAB`
  (`nav_conv_merged_bridge.py:238`); both capability-matrix test files
  (`tests/test_brain_conversational_agent.py`, `tests/test_nav_conv_merged_agent.py`) assert the matrix **only
  at this V=16 set**.
- The composer's raw bind/compose is separately validated at **320 concepts** — but in **ISOLATION** and on a
  **different code path**: the 320 result (`2026-06-02-full-320-flat-distinct-composition-RESOLVES-multiseed.md`)
  is the `_insubstrate_*` ±1 *coincidence* harness
  (`research/findings/raw/_insubstrate_flatdist320_anybank_test.py` + `_insubstrate_bind_unbind_probe`), NOT the
  `RFPhasorComposer` the agent actually uses. CLAUDE.md lists "production 320-concept on the brain agent" as a
  DEFERRED follow-on — correctly: the 320 isolation result does not certify the *agent loop* (parser →
  composer → query, plus negation/clause/attribute/elaborate) at 320.

So there are three distinct things conflated in the current state, and this arc separates them:

| | code path | scale validated | what's missing |
|---|---|---|---|
| RF composer in isolation | `RFPhasorComposer` | V≈8–17, ≤10 facts (`tests/test_rf_phasor_composer.py`) | larger vocab + larger fact set |
| ±1 coincidence bind in isolation | `_insubstrate_*` probes | **V=320** (flat/structured/any-bank) | not the agent's composer; no parser/dlPFC |
| the full agent loop | `BrainConversationalAgent` / `MergedNavConvAgent` | **V=16** | the whole capability matrix at V>16, multi-seed |

### Test A — the full capability matrix at V = 16 → 64 → 128 → 320, multi-seed

Run the **whole agent loop** (comprehend via the Hebbian parser → store in the composer → query) and assert
**every** capability at each vocabulary scale, multi-seed: who-Q&A, what-Q&A, **abstention**, negation/yes-no,
embedded clause, one-attribute, generation, dialogue planning. Not just the composer in isolation — the parser
hand-off and the dlPFC must hold too.

*Why it matters:* this is the actual deliverable — "how big a vocabulary does the biological conversational
agent support, end to end?" The composer-in-isolation 320 result is necessary but not sufficient; the agent
adds the parser (vocabulary-agnostic, so expected to hold — but unverified at scale), the per-capability
extra bindings (polarity, attribute, clause-nesting each add a role → more crosstalk), and the dlPFC.

*Substrate choice (which agent):* run **both**, cheap-first. Primary heavy characterization on
`BrainConversationalAgent` (lighter: a ~126-neuron parser bridge + the composer's per-op RF bridges — no
~2900-neuron navigation cascade rebuilt per scale). Then a **co-residence confirmation** on `MergedNavConvAgent`
at the chosen scales to prove the matrix still holds on the *consolidated* one-bridge substrate (the navigation
populations present and, where a nav episode is interleaved, the conv weights frozen). The merged agent is the
load-bearing "consolidated substrate" claim; the standalone agent is the cheap workhorse that produces the
ceiling curve. They share the exact same composer + parser, so a holds-on-standalone / holds-on-merged pair is
the honest evidence.

### Test B — two-attribute on the agent: does it lift the documented K=5 boundary?

Exercise two-attribute entities ("big hot apple") **through the agent**, multi-seed, at each vocab scale.

*Why it matters, and an important correction to the CLAUDE.md framing.* CLAUDE.md says the FHRR pivot "unlocked
an F=3 two-attribute resonator that lifts the K=5 boundary" and that it "is NOT exercised on the agent." Reading
the code refines this in a load-bearing way:

- The production `RFPhasorComposer` **does not use an F=3 resonator**. It handles two-attribute by binding a
  second `attribute2` role and cleaning up each adjective independently
  (`rf_phasor_composer.py:266–268, 319, 343`). The existing composer-level test
  `test_rf_phasor_composer_two_attribute` (`tests/test_rf_phasor_composer.py:162`) asserts it **RESOLVES
  multi-seed at D=256**, explicitly: *"the boundary is LIFTED by the substrate, no F=3 resonator needed"* (the
  SNR-per-unbind dial 2N/M, raised by increasing D). That is the mechanism actually on the agent's path.
- The **F=3 resonator** (a genuine factorization decoder that needs D∝M², GPU-gated) lives in a **separate
  numpy-reference** agent, `SpikingUnifiedAgent` (`2026-06-04-capacity-curve-scaling-cost-model.md`), which is
  REFERENCE-only per CLAUDE.md, not the production substrate.

So Test B is really two crisp sub-questions:

- **B1 (the production claim):** does the RF composer's **D-dial** two-attribute (the `attribute2`-role + raise-D
  mechanism) hold **through the full agent** and **at larger vocab** (not just the V≈3-word composer unit test)?
  The composer test pins D=256 at tiny vocab; the open question is whether the agent at V=64/128/320 needs a
  larger D for two-attribute (the learned-code negative below says the *resonator* path needs D≈8192 at V=640 —
  but the production composer uses the simpler `attribute2`-role path, whose vocab-scaling is **unmeasured on the
  agent** and is exactly what B1 establishes).
- **B2 (optional, only if B1 plateaus):** if the D-dial path tops out at some vocab, that is the precise point
  where the F=3 resonator (the `SpikingUnifiedAgent` numpy reference, or a future on-substrate port) becomes the
  needed mechanism — and B2 measures the resonator's reach at that scale as the *next-mechanism* signal for
  Step 3. (This is reference-substrate, so flag it as the idealized ceiling, not "more brain-like.")

### Test C (optional, only after A+B) — a harder conversational task to find the top

Push past the V-scaling matrix to find where the *loop* (not just the codebook) breaks:

- **Larger fact sets** (the production-scale risk is spurious matches as the KB grows;
  `test_rf_phasor_composer_production_scale` does 10 facts at V≈10 — scale to 30–60 facts at V=320). This is the
  no-confab-moat stress: more facts → more chances for a false match.
- **Deeper clause nesting** (depth-2: a clause whose patient is itself a clause). The composer's recursive
  `_render` supports it structurally (`rf_phasor_composer.py:148–157`); depth-3 was a numpy-reference RESOLVES
  (`2026-06-03-recursive-clause-nesting-RESOLVES-depth3-capacity.md`), unverified on the production composer at
  scale.
- **Multi-fact `elaborate`** over a denser association graph (more topics, more neighbors) — the dialogue
  planner's difficulty scales with #facts, not vocab, so this is the orthogonal axis.

Test C is the "find the genuine top" probe; it is explicitly *after* A and B (which are the load-bearing
deliverables) and may be deferred if A/B already locate a clear ceiling.

---

## 2. The existing machinery to reuse (this is mostly characterization, not new building)

The point of this section: almost everything needed already exists. The work is a parameter sweep + a small
harness, not new mechanism.

### How vocabulary size is set

- **Composer:** `RFPhasorComposer(seed, D, vocab=<list of words>, period)`. `vocab=None` →
  `DEFAULT_VOCAB` (V=17). Pass any word list to set V. The composer generates its own deterministic random
  phasor code per word from `seed` (`rf_phasor_composer.py:78–80`), so **no external code cache is needed** for
  the RF path — this is a major simplification vs the rate composer.
- **Standalone agent:** `BrainConversationalAgent(seed, concepts=<{word: code} dict>)`. With the default
  `composer_kind='rf'`, the agent passes `vocab=sorted(concepts.keys())` to the RF composer
  (`brain_conversational_agent.py:174`). **Subtlety to record (anti-cheat §5):** the RF composer uses only the
  *keys* (the word set) and generates its OWN phasor codes — it **ignores the code values** in the dict. So to
  set V on the RF agent, the harness only needs V distinct words; the code dict's values are irrelevant on the
  RF path (they matter only for `composer_kind='rate'`).
- **Merged agent:** `MergedNavConvAgent(seed, vocab=<list>)` →
  `build_merged_nav_conv_bridge(vocab=...)`. The vocab also sizes the dlPFC assemblies (`n_dlpfc =
  max(600, 60*V)`, `nav_conv_merged_bridge.py:243`) — so larger V grows the dlPFC slice; budget for that.

### Do 320-concept codes already exist?

- **For the RF agent path: not needed** — the RF composer self-generates phasor codes from the seed. A 320-word
  *word list* is all that's required. Two ready sources of distinct words: `g20_vocab_spec_320.py` (the curated
  320 "age-5" word list) or synthetic `f"c{i:03d}"` labels (what the existing 320 probes use). Word identity is
  irrelevant to the RF composer (random codes), so synthetic labels are fine and avoid any vocab-table edits.
- **Caches that DO exist** (for reference / the rate path only): `denoise64_seed{42,43,44}.npz` (V=16 grounded
  concept-pool codes), `_flatdist320_codes.npz` (the 320 ±1 isolation codes), `g20_vocab_spec_320.py` (word
  list + sparse-pattern generator `generate_sparse_patterns`). The `production_codes()` helper
  (`_core_composer_grounded320_probe.py`) turns G.20 sparse patterns into dense projected codes — used by the
  existing 320 probes (it matters for the *rate* composer; on the RF agent only its word **keys** are used).

### Existing probes/tests that already do most of this — extend, don't reinvent

- **`research/findings/raw/_brain_agent_grounded320_probe.py`** — **already runs the FULL
  `BrainConversationalAgent` at V=320**: `hear()` → comprehend (parser) → store → `what_does`/`who_does` +
  abstain, multi-seed-able (`--seed`, `--vocab`, `--n-facts`). This is the **direct seed of Test A** — it
  covers who/what/abstain at 320 but **not** negation, clause, one-attribute, two-attribute, or generation. The
  Test-A harness is this probe **extended** to assert the whole matrix.
- **`research/findings/raw/_brain_agent_elaborate320_probe.py`** — already runs `elaborate` (dialogue planning)
  in the agent at V=320. Folds into Test A as the dialogue-planning row.
- **`tests/test_rf_phasor_composer.py`** — the capability matrix at the *composer* level: `..._full_matrix_at_scale`
  (5 facts), `..._production_scale` (10 facts), `..._two_attribute` (D=256, the K=5 lift), `..._clause`,
  `..._negation_yesno`, `..._one_attribute`, `..._dialogue`. These are the **assertion templates** — the same
  asserts, parameterized over V and D.
- **`tests/test_brain_conversational_agent.py` / `tests/test_nav_conv_merged_agent.py`** — the V=16 capability
  matrix. The new tests are these **parameterized over vocab scale**.
- **`SpikingUnifiedAgent(resonator_backend="cupy")`** + `_gpu_resonator_capacity.py` /
  `_capacity_curve_probe.py` — the numpy-reference F=3 resonator + its GPU port and capacity sweep, for **Test
  B2 only** (the reference ceiling; flag as idealized).

### What must be newly written (small)

Only a thin **sweep harness**: a runner that, for `seed × V × D`, builds the agent, stores a fixed mixed fact
set (flat / one-attribute / two-attribute / clause / negated), runs every capability assertion, and writes a
per-(seed,V,D) JSON with per-capability pass counts. It is `_brain_agent_grounded320_probe.py` generalized to
the full matrix + a vocab/D loop. No `sim/` edit; no new mechanism.

---

## 3. The cheap-first probe (run BEFORE any big sweep)

The de-risk question: **does the full agent hold past V=16 at all, and is two-attribute alive on the agent?**
The smallest run that answers it:

1. **Probe 0 — agent matrix at V=64, single seed (42), default D=128.** Build `BrainConversationalAgent(seed=42)`
   with a 64-word list; store the standard mixed fact set (the V=16 capability-matrix facts, re-used verbatim,
   among 64 distractor words); assert the **whole matrix** (who/what/abstain/negation/clause/one-attribute/
   describe/elaborate). This is `_brain_agent_grounded320_probe.py` + the negation/clause/attribute/describe
   asserts, at V=64. **PASS → the loop survives 4× vocab; proceed to the sweep. FAIL → localize which capability
   broke first (the ceiling is already at V=64 — itself a clean finding).**
2. **Probe 0b — two-attribute on the agent at V=64, default D, single seed.** One `store("dog","look",(("big",
   "hot"),"apple"))` + the `query_patient` assert from `test_rf_phasor_composer_two_attribute`, but through the
   **agent** and at V=64 (the composer unit test is V≈3, D=256). Tells us immediately whether the production
   D-dial two-attribute is alive on the agent at non-trivial vocab, and roughly what D it needs there.
3. **Probe 0c (only if 0 PASSes) — agent matrix at V=320, single seed.** Confirms the headline scale before
   spending on multi-seed. The existing `_brain_agent_grounded320_probe.py --vocab 320` already does the
   who/what/abstain slice in minutes; extend to the full matrix.

If Probe 0/0b/0c all pass at single seed, the multi-seed sweep is justified. If any fails, the ceiling is
mapped cheaply and the sweep narrows to "characterize the degradation," not "validate a pass."

### CPU vs GPU cost estimate per scale

The agent loop's spiking ops require **GPU (CuPy)**: the Hebbian `BridgeParser` and the dlPFC
`SpikingSpreadingController` are GPU-validated (both capability-matrix test files `skipif(not is_gpu_backend())`,
and the RF dialogue test skips off-GPU). NumPy is a tiny-smoke / import-check path only.

Cost drivers (measured/derived from the cost-model finding + the code):

- **The RF composer is the dominant cost and scales with D and #ops, NOT with V's cleanup** (cleanup is a numpy
  argmax over V codes — cheap; vectorized it is a single matmul). Each RF op runs `rf_resonate_steps(period+8)`
  ≈ 208 steps at `period=200` (`rf_phasor_composer.py:110`). A query is several ops (one unbind per role
  checked, plus the final cleanup). The FHRR-switch finding notes per-op latency is "seconds-scale at 320,
  D=512" and that `period=80` holds (a ~2.5× speedup lever if latency bites).
- **Per-capability-matrix run** (≈5 facts stored + ~10 queries): on GPU, **order of a few minutes** at D=128,
  V=64; the V=320 cleanup adds negligibly (argmax over 320 vs 64). The standalone agent has **no per-scale
  bridge rebuild cost** beyond the parser (~126 neurons, trained once per agent build).
- **Merged-agent confirmation runs** add the ~2900-neuron bridge build + the parser train pass
  (`build_merged_nav_conv_bridge`), order of a few minutes each — done at only the chosen confirmation scales,
  not the whole grid.
- **Test B2 (F=3 resonator reference)** is the only place GPU is *the enabler* (D=8192 at V=640 is ~11s on GPU,
  untenable on CPU). Run it only if B1 plateaus.

Rough total: the cheap-first probes are **tens of minutes** on GPU; the full `seed × V × D` matrix sweep is
**a few GPU-hours** (dominated by RF op latency × the grid size), tunable down via `period=80`.

---

## 4. Metrics + pass bars

Per **(seed, V, D)**, per capability, report the count correct / count attempted, then aggregate multi-seed:

| capability | metric | pass bar |
|---|---|---|
| who-Q&A / what-Q&A | fraction of stored facts whose cue returns the right filler | **≥ 0.80 multi-seed** (matches the FROZEN composer bars; ideally 1.00 as at V=16) |
| **abstention (no-confab moat)** | fraction of *unmatched* cues returning `None`/`unknown` | **must be ≥ the V=16 level — never weakens with scale.** A drop here is a *hard fail* regardless of the other rows (the moat is the project's defining property) |
| negation / yes-no | yes/no correct on AFFIRM/NEGATE facts + `unknown` on unknown | **≥ 0.80 multi-seed** |
| embedded clause | nested SVO rendered correctly | **≥ 0.80 multi-seed** |
| one-attribute | "adj noun" both decoded | **≥ 0.80 multi-seed** |
| **two-attribute (Test B1)** | both adjectives + noun decoded | **the bar IS the question:** PASS if it RESOLVES multi-seed at a *practical* D (≤256–512, matching the composer test). Record the **minimum D** needed per V — that curve is the deliverable. (Contrast: the K=5 boundary = "never resolves" on the old ±1 scheme; the learned-code *resonator* needed D≈8192 at V=640.) |
| generation (describe) | full sentence reconstructed; `None` on unknown subject | **≥ 0.80 multi-seed**, abstention exact |
| dialogue planning (elaborate) | on-topic associate ∈ graph neighbors; `None` on unconnected | **≥ 0.80 multi-seed**, abstention exact |

**Multi-seed = ≥ 6 seeds** (project standard; 3-seed is an indicator only). Report the per-V curve for each
capability: the headline product is a **ceiling map** — "capability X holds to V=___ at D=___; degrades at V=___".

**Headline summary metric:** the **largest V at which the full matrix holds multi-seed** (the biological
conversational vocabulary ceiling on the consolidated substrate), plus the **two-attribute minimum-D curve**.

---

## 5. Anti-cheats / honest controls

- **It must test the CONSOLIDATED substrate + the full loop, not the composer in isolation.** The standalone
  `BrainConversationalAgent` already routes through the parser (`hear` → `parser.parse` → `composer.store`); the
  merged-agent confirmation additionally proves the navigation populations co-reside (the
  `region_indices_dict()` asserts in `test_nav_conv_merged_agent.py` are the template). **Reuse the merged
  agent's existing anti-cheat asserts** (`MergedNavConvAgent.__init__`, `nav_conv_merged_bridge.py:618–630`):
  `parse_conj`/`dlpfc_wm` are merged-bridge regions and `elaborate`'s dlPFC context **is** the merged bridge — a
  silent fallback to a standalone parser/dlPFC fails loudly. A "high score" from accidentally running the
  composer alone is thereby excluded by construction.
- **The cleanup must be the substrate's, not numpy, for the substrate claim.** The default RF cleanup is a numpy
  argmax (the validated fast path); the **fully-on-bridge spiking cleanup** is opt-in
  (`enable_spiking_cleanup=True`, validated == numpy multi-seed). For at least one confirmation run at each
  headline scale, enable the spiking cleanup so the *selection* is in spikes — otherwise the honest claim is
  "the binding is on-substrate, the final argmax is numpy" (which is also true and must be stated, not hidden).
- **Abstention / permuted control so a high score is not trivially inflated.** Two controls:
  (a) **abstention floor** — for every passing query set, an equal number of *unmatched* cues must abstain
  (already in the matrix as a graded metric, not a footnote); a system that answers everything scores high on
  who/what but fails abstention. (b) **shuffled-fact control** — store the facts, then query with a
  *random permutation* of (agent,action)→patient pairings; correct answers should collapse to chance. This
  guards against a degenerate "echo the most-recent / most-frequent filler" mode masquerading as retrieval at
  scale (the analogue of the 2026-05-03 permuted-label control that caught the text-IO artifact).
- **The honest caveat — the genuinely-NEW signal.** The composer is a **principled idealization**: an
  exact-inverse VSA *algebra* that demands decorrelated full-precision codes (CLAUDE.md "composer-as-idealization").
  A clean 320 pass is therefore **the algebra working at 320**, NOT evidence the substrate became "more
  brain-like." The genuinely-new information this arc produces is narrower and real: **(i)** whether the algebra
  still holds once it runs on the *consolidated nav+conv one-bridge substrate* (frozen-weight isolation, dlPFC
  co-residence, the RF ops on a shared slice if `co_resident_composer=True`) and through the *full agent loop*
  (the parser hand-off + every capability), **not just the bind/unbind kernel in isolation**; and **(ii)** the
  per-capability **degradation map** (which capability breaks first, at what V, at what D) — which is exactly the
  spec for Step 3. State this caveat in the findings doc verbatim so the result is not over-read.

---

## 6. How this sharpens Step 3 (the learned cortex)

The whole point of characterizing the idealized algebra's ceiling first is to hand Step 3 a concrete spec. Each
possible outcome tells the learned cortex something specific it must ADD (the idealized algebra gets these
"for free"; a learned, lossy, redundant cortex must earn them):

- **If the matrix holds to V=320 multi-seed (the algebra scales cleanly):** the learned cortex's bar is set — it
  must reach 320-concept conversational competence **without** the exact-inverse algebra's two free lunches: the
  **clean-code demand** (the algebra needs decorrelated full-precision codes; a cortex must read whatever messy,
  correlated code arrives) and the **exact invertibility** (a cortex unbinds with *learned, lossy* read-outs).
  The deliverable is the explicit list of what the cortex must reproduce at 320: who/what, the no-confab moat,
  negation, clause, attribute, generation, dialogue — with the abstention property being the hardest to learn
  (a learned associator tends to confabulate; the algebra abstains by construction).
- **If a capability degrades at some V (say clause or two-attribute breaks at V=128):** that is the **precise
  capacity the cortex must exceed to be worth building** — the algebra itself runs out there, so a learned
  cortex that matched it would inherit the same wall. The degradation point + its mechanism (crosstalk / D-floor
  / multi-hop SNR) names the specific representational capacity Step 3 must improve on (e.g. learned sparse
  block codes, the deep-research Track-1 ~5000× capacity lever, rather than brute D).
- **If two-attribute needs an impractically large D on the agent (B1 plateaus):** that localizes the **one place
  a learned factorizing read-out (or the F=3 resonator) is genuinely required** — the simple `attribute2`-role +
  raise-D path can't carry multi-attribute composition at scale, so the cortex must learn a factorization, not
  just a bigger codebook. (And it confirms the F=3 resonator's reference role: the numpy-reference ceiling is
  what the on-substrate learned version must hit.)
- **The abstention result is the single most decisive Step-3 input.** The no-confab moat is free in the algebra
  (no fact matches → `None`). Whatever the learned cortex is, it must *preserve abstention at the ceiling
  vocabulary* — so this arc's abstention-at-scale curve is the hardest acceptance bar Step 3 inherits.

---

## 7. Sequencing + rough cost (cheap-first)

| order | step | scale | est. cost (GPU) | gate to proceed |
|---|---|---|---|---|
| 1 | Probe 0: standalone agent **full matrix**, V=64, seed 42, D=128 | 1 run | ~tens of min | matrix holds (else: ceiling found at 64 — stop, document) |
| 2 | Probe 0b: two-attribute on the agent, V=64, seed 42, sweep D∈{128,256,512} | 3 runs | ~tens of min | two-attribute resolves at some practical D (records min-D@64) |
| 3 | Probe 0c: standalone agent full matrix, **V=320**, seed 42 | 1 run | ~tens of min | headline scale holds at single seed |
| 4 | **Test A sweep:** standalone agent full matrix, V∈{16,64,128,320} × **6 seeds**, D=128 (+ min-D for two-attr) | ~24+ runs | a few GPU-hours | the multi-seed ceiling curve (the deliverable) |
| 5 | **Test B1 curve:** two-attribute min-D per V, 6 seeds | folded into 4 | (shared) | the two-attribute min-D-vs-V curve |
| 6 | **Merged-agent confirmation:** `MergedNavConvAgent` full matrix at the chosen headline scales (e.g. V=64, V=320), incl. one `enable_spiking_cleanup` run + the shuffled-fact + abstention controls | ~4–6 runs | a few GPU-hours | matrix + moat hold on the **consolidated** substrate |
| 7 | (optional) **Test B2:** F=3 resonator reference ceiling, only if B1 plateaued | GPU resonator sweep | ~minutes | the next-mechanism signal for Step 3 |
| 8 | (optional) **Test C:** larger fact sets / depth-2 clause / dense elaborate at V=320 | a few runs | a few GPU-hours | find the genuine top |

**Cheap-first discipline:** steps 1–3 are single-seed smokes that can kill or green-light the whole sweep in
under an hour. The sweep (4–6) is only paid for if the smokes pass; if a smoke fails, the arc pivots from
"validate a pass" to "characterize the degradation" (a smaller, targeted set of runs at the breaking scale).

**Where it pairs with the next arc (functional integration).** This characterization is the natural companion to
**STEP 2b** (the RF composer running *co-resident* on the merged bridge's `rf` slice via the owner-approved
sliced `rf_kick` — `MergedNavConvAgent(co_resident_composer=True)`): the step-6 merged-agent confirmation should
be run **in the co-resident configuration** once 2b lands, so the ceiling is measured on the *strict
single-instance* substrate (one bridge, RF ops on a slice), which is the most honest "consolidated substrate"
claim. The two arcs share the merged agent and the same capability assertions; run them together to avoid
double-paying the merged-bridge build. The ceiling this arc measures then directly scopes **Step 3** (the
learned cortex), as in §6.

---

## Appendix — file index (read-only ground truth for this design)

- Consolidated agent: `research/runners/nav_conv_merged_bridge.py` (`MergedNavConvAgent`,
  `build_merged_nav_conv_bridge`, `MergedRFComposer`)
- Standalone agent: `research/runners/brain_conversational_agent.py` (`BrainConversationalAgent`, `BridgeParser`)
- Production composer: `research/runners/rf_phasor_composer.py` (`RFPhasorComposer`, two-attribute at
  lines 266–268/319/343, recursive clause at 148–157)
- Rate composer (the ±1 K=5-bounded path, opt-in): `research/runners/core_sim_composition.py` (`CoreSimComposer`)
- Capability-matrix tests (V=16): `tests/test_brain_conversational_agent.py`,
  `tests/test_nav_conv_merged_agent.py`; composer matrix + two-attribute D=256 lift:
  `tests/test_rf_phasor_composer.py`
- Existing V=320 agent probes (Test-A seed): `research/findings/raw/_brain_agent_grounded320_probe.py`,
  `_brain_agent_elaborate320_probe.py`, `_core_composer_grounded320_probe.py` (`production_codes`)
- 320 composer-in-isolation validation (the separate ±1 path): `_insubstrate_flatdist320_anybank_test.py`,
  `2026-06-02-full-320-flat-distinct-composition-RESOLVES-multiseed.md`
- F=3 resonator reference + cost model (Test B2): `2026-06-04-capacity-curve-scaling-cost-model.md`,
  `2026-06-03-learned-code-agent-320-scale-boundary-HONEST-NEGATIVE.md`,
  `SpikingUnifiedAgent(resonator_backend="cupy")`
- 320 word list (if real words wanted): `research/runners/g20_vocab_spec_320.py`
- FHRR production switch (the agent default): `2026-06-05-fhrr-production-switch-DONE.md`
