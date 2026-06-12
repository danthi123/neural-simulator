# Learned-graded cortex → conversational pipeline: INTEGRATION design + the cheap-first CAPABILITY de-risk

> **Status:** present-before-build. READ-ONLY design (no `sim/` edit, no GPU run, no bridge built). The single
> deliverable is this doc + one commit. It is the project's standing "deep research + design BEFORE building"
> opening move for the next part of the sim — wiring the de-risked learned-graded cortex into the conversational
> agent and proving the **capability** (generalization in conversation) at small scale before the expensive
> 32-bridge build. **Date:** 2026-06-12. **Author role:** read-only design subagent. Every load-bearing claim
> is cited to a file read in full.

---

## 0. Where this sits (the gap this design closes)

The project's "step 3 true cortex" replaces the conversational composer's **idealized exact-inverse
vector-symbolic binding** — Fourier Holographic Reduced Representation ("FHRR": roles/fillers are unit-magnitude
phasor vectors, bind = element-wise complex product, unbind = multiply by the conjugate, cleanup = nearest
codebook entry; `research/runners/rf_phasor_composer.py` lines 117–167, 247–256) — with a **learned, brain-based,
semantically-STRUCTURED cortex** where similar concepts get similar codes, so the agent can GENERALIZE ("a cat is
like a dog"). FHRR *demands* decorrelated full-precision codes (the cleanup's nearest-neighbour argmax is only
reliable when codes are near-orthogonal), which is exactly why it cannot generalize across similar concepts.

**What is already de-risked (the MECHANISM, in isolation):**

| Piece | Validation (file) | Status |
|---|---|---|
| The CLS resolution of the central tension (graded cortex + decorrelated binder + **cortical reinstatement**) | `research/findings/2026-06-11-dual-CLS-cortex-channel-derisk-GO.md`; `docs/plans/2026-06-11-dual-CLS-architecture-design.md` | GO (3/3, on-substrate, synthetic graded codes) |
| The learned-graded LEARN recipe (homeostatic Oja recurrent + divisive-normalization read-out) | `research/findings/2026-06-11-learned-graded-embedding-homeostasis-GO.md`; `research/runners/learned_graded_embedding_homeostasis_probe.py` | GO (cycle-independent, V=320 multi-seed) |
| Multi-bridge cross-bridge composition + the no-confab moat **survive on correlated graded codes** | `research/findings/2026-06-12-multibridge-graded-derisk-GO.md`; `research/runners/multibridge_graded_derisk.py` | GO (3 bridges × 64, 3 seeds) |
| The no-confab abstention moat as a learned neural gate matching the host check | `research/findings/2026-06-11-familiarity-gate-v320-GO.md`; `research/runners/familiarity_gate_v320_validation.py` | GO (V=320, zero breaches) |

**THE GAP this design closes — the de-risk validated the cortex MECHANISM, NOT the CAPABILITY.** Concretely, the
multi-bridge de-risk measured three *isolated* mechanism quantities (read directly from
`multibridge_graded_derisk.py`):

- **M1/M2** — per-bridge graded codes form a *similarity metric* (Pearson of recovered-vs-true similarity, the
  within>between cosine margin). This is "the codes are graded," not "the agent uses them."
- **M3** — cross-bridge *fact recall* (a V-tag engram key→value retrieval: cue → target over the other 63
  concepts). This is key-value retrieval, **not** generative role-filler binding.
- **M4** — the moat (familiarity gate agrees with host abstention).

It did **NOT** validate: (a) the full conversational matrix (who/what Q&A, abstention, negation, one-attribute, a
clause) running ON the learned-graded cortex; (b) generative role-filler **binding** using the graded codes; and
(c) the **NEW capability — generalization IN conversation** (answer a query about concept X using a fact learned
about a similar concept Y). Those three are the goal, and they are what §2's cheap-first capability de-risk
proves before the 32-bridge build.

---

## 1. The integration architecture (concrete)

### 1.1 The pipeline as it exists today (what we plug into)

`research/runners/brain_conversational_agent.py` is the production loop. `BrainConversationalAgent.__init__`
builds two real bridges and delegates:

```
                 hear("dog go north")
                         │
         ┌───────────────▼──────────────────┐
         │  BridgeParser  (comprehension)    │   Hebbian (word-position × voice) → role
         │  brain_conversational_agent.py:28 │   ("dog go north" → {agent:dog, action:go, patient:north})
         └───────────────┬──────────────────┘
                         │ roles dict
         ┌───────────────▼──────────────────────────────────────────────┐
         │  RFPhasorComposer  (bind / store / unbind / cleanup)          │
         │  rf_phasor_composer.py:61                                     │
         │  • concept code  =  self.concepts[word]  (a phasor, line 80)  │
         │  • store():  _encode = bind each role→filler, bundle (l.144)  │
         │  • query_patient/query_agent/ask_yes_no:  unbind + _cleanup   │
         │  • _cleanup = argmax cos over the codebook  (l.247)  ◄── identity recovery
         │  • abstention (the moat): returns None when no fact matches   │
         └───────────────┬──────────────────────────────────────────────┘
                         │ (dialogue) elaborate(topic)
         ┌───────────────▼──────────────────┐
         │  dlPFC SpikingSpreadingController │   spreads over the agent's own association graph
         │  (content_selection_spiking)      │
         └───────────────────────────────────┘
```

The composer is **vocabulary-driven by `self.concepts` (line 80): a `{word: phases[D]}` dict**. Today those phases
are random (`rng.uniform`, decorrelated by construction) — that is the idealization. The composer ALSO already
accepts an injected `grounded_codes={word: phases[D]}` dict (lines 86–89) that *overrides* the random codes for
those words; this is the seam the cortex plugs into, and it is already-built (validated `== random` at parity).

### 1.2 The central tension and its resolution (cortical reinstatement)

**The tension (stated precisely).** The graded cortex codes are *correlated* (cat≈dog cosine HIGH) — that is the
whole point, it is what gives generalization. But FHRR binding/unbinding (and its cleanup) *require* decorrelated
codes — correlated codes make the unbind estimate ambiguous and the cleanup argmax unreliable (this is the
documented reason the composer cannot generalize, and why the rate composer's `decorrelate=True` ZCA path exists:
`core_sim_composition.py` lines 248–256, which deliberately drives between-cos to ~0 to make composition work).
**Graded-vs-decorrelated pull opposite ways at the binding step.**

**The resolution — the dual / complementary-learning-systems architecture (CLS), validated GO.** Do NOT try to
make ONE code serve both. Use **two representations linked by an identity round-trip**
(`docs/plans/2026-06-11-dual-CLS-architecture-design.md` §1; validated on-substrate in
`2026-06-11-dual-CLS-cortex-channel-derisk-GO.md`):

1. A **graded CORTEX channel** carries the correlated similarity codes (cat≈dog). Generalization runs HERE,
   *directly on the cortex codes, never through the binder.*
2. A **decorrelated HIPPOCAMPAL binder** (the existing DG pattern-separation → CA3/Hopfield recall, between-cos
   ≈ 0.05) produces clean codes that FHRR binds/unbinds/recalls reliably. Binding + the moat run HERE.
3. **Cortical reinstatement is the link.** The decorrelated recall recovers the *identity* (WHICH concept) —
   `recall_identity_and_settle` in `dual_cls_cortex_channel_derisk_probe.py:143` (Hopfield argmax → `recovered_idx`,
   identity 1.000 at the strong operating point). That identity is then used to **reinstate the recalled
   concept's stable graded cortex code** — `cortex_channel_roundtrip` line 188: `reinstated =
   cortex_codes[recovered_idx]`. So a query that goes round the loop comes back as a *graded* code (generalization
   intact), even though the binding itself ran on the decorrelated side.

**Why this is decisive (the GO finding, re-stated):** the round-trip Pearson(S, S′) = **+1.000** on the cortex
channel (vs +0.189 if you decode the degraded hippocampal *settled state* — twice-confirmed cap), AND
generalization is **1.000 / 4.0× chance** on the cortex channel with orthogonal + permuted controls collapsing to
chance (`2026-06-11-dual-CLS-cortex-channel-derisk-GO.md` headline table). The identity-gated round-trip is the
*easy* part (with perfect identity, reinstatement trivially returns the original graded codebook); the
*interesting, load-bearing* result is that generalization is RESTORED on the cortex channel after being at chance
on the decorrelated settle — and is genuinely the cortex channel's graded structure (the orthogonal codebook does
NOT generalize through the same pipeline).

### 1.3 The integration data flow (concept → cortex → binder → composer → dialogue)

Here is exactly how the existing components compose, naming each and flagging reuse vs new glue:

```
  word "dog"                                           graded-similarity lives in the cortex codes
      │                                                generalization reads them DIRECTLY (never via the binder)
      ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │ (A) CORTEX  — the LEARNED GRADED codebook  (NEW glue around validated parts)               │
 │     learner   = HomeostaticAssocGraph (Oja recurrent)   learned_graded_embedding_homeostasis_probe.py:141 │
 │     read-out  = divnorm_spreading_readout                learned_graded_embedding_divnorm_readout_probe.py │
 │     → cortex_codes[word] = a graded code (cat≈dog HIGH).  Per-bridge (one bridge per semantic shard).      │
 │     EXPOSED AS: a {word: code} dict per shard  (a similarity metric over the shard's concepts).            │
 └───────────────┬──────────────────────────────────────────────────────┬───────────────────┘
                 │ encode (DG pattern separation)                        │ generalization read (DIRECT)
                 ▼                                                        │  similarity_vote_infer over cortex_codes
 ┌───────────────────────────────────────────────┐                       │  (dual_cls_architecture_proof_probe.py:191)
 │ (B) DECORRELATED BINDER (the hippocampal side) │                       │
 │     DG-style separation → decorrelated code k  │                       │
 │     per concept (between-cos≈0.05).            │                       │
 │     The composer BINDS these clean codes.      │                       │
 │     RFPhasorComposer.concepts[word] := code k  │◄── this is the existing `grounded_codes` seam (l.86–89)   │
 │     store/unbind/cleanup unchanged.            │                       │
 └───────────────┬───────────────────────────────┘                       │
                 │ query → unbind → _cleanup → recovered identity (line 247: argmax = WHICH concept)            │
                 ▼                                                        │
 ┌───────────────────────────────────────────────┐                       │
 │ (C) CORTICAL REINSTATEMENT (the link)          │                       │
 │     recovered_idx → cortex_codes[recovered_idx]│───────────────────────┘  reinstate the graded code,
 │     dual_cls_cortex_channel_derisk_probe.py:188│                          so a recalled fact is graded again
 └───────────────┬───────────────────────────────┘
                 ▼
 ┌───────────────────────────────────────────────┐     ┌──────────────────────────────────────────┐
 │ (D) THE MOAT (no-confab abstention)            │     │ (E) DIALOGUE PLANNING (unchanged)          │
 │     RelationalFamiliarityGate + host abstention│     │     SpikingSpreadingController over the     │
 │     familiarity_gate_v320_validation.py:74     │     │     association graph. Substrate-agnostic.  │
 │     (gate ACCEPTS only where host accepts)     │     │     brain_conversational_agent.py:242       │
 └────────────────────────────────────────────────┘     └──────────────────────────────────────────┘
```

**The crucial design decision — which channel carries graded codes, and where binding's decorrelated codes come
from.** Three distinct code populations, NOT one whitened code:

- **Generalization** reads the **graded cortex codes directly** (the `similarity_vote_infer` k-nearest-graded-
  neighbour read, `dual_cls_architecture_proof_probe.py:191`). It NEVER goes through the binder. This is why
  graded↔decorrelated never collide for generalization: generalization simply does not touch the decorrelated side.
- **Binding** consumes **decorrelated codes** produced by the DG encode of the cortex codes (the
  `make_dg_encoder` random-projection + top-k separator, `dual_cls_architecture_proof_probe.py:287`, or its
  on-substrate spiking-DG equivalent `StrongDGEncoder`). The composer's `self.concepts[word]` is set to these
  decorrelated codes — i.e. the cortex *induces* the codes the composer binds, but the composer still binds clean
  decorrelated codes, so FHRR is unchanged and the no-confab cleanup stays reliable.
- **A recalled fact** comes back through **cortical reinstatement** (C): the composer's cleanup recovers the
  identity (which concept), and the cortex code for that identity is reinstated — so the recalled filler is a
  *graded* code, usable for a downstream generalization step.

**Generalization-in-conversation, end to end (the new capability).** "What does a dog eat?" with only "cat eats
meat" stored:
1. Parser → `{agent: dog, action: eat}`.
2. Composer relational query `query_patient("dog", "eat")` → the binder finds NO exact fact whose (agent, action)
   is (dog, eat). **Host moat would abstain here.** This is the crux (see §2.2 + §4): the integration adds, BEFORE
   abstaining, a **graded fallback** — find the nearest *cortex-similar* known agent for the same action. "dog" is
   graded-close to "cat" (cortex codes), and "cat eat ___" IS stored → answer "meat", flagged as a
   *generalized* (lower-confidence) answer.
3. The moat must STILL abstain when there is no graded-similar known fact (genuine absence), so the fallback is
   gated by the familiarity score on the *similar* cue, not a free pass.

### 1.4 Reuse vs new glue (summary)

| Component | Source (validated) | Reuse / NEW |
|---|---|---|
| Parser (comprehension) | `BridgeParser` (`brain_conversational_agent.py:28`) | **REUSE verbatim** (vocabulary-agnostic) |
| Graded cortex learner + read-out | `HomeostaticAssocGraph` + `divnorm_spreading_readout` | **REUSE** (wrap into a `CortexCodebook` that exposes `{word: graded_code}`) |
| DG decorrelating encoder | `make_dg_encoder` (numpy) / `StrongDGEncoder` (spiking) | **REUSE** |
| FHRR binder / store / unbind / cleanup / moat-abstention | `RFPhasorComposer` (`rf_phasor_composer.py`) | **REUSE verbatim** (inject decorrelated codes via the existing `grounded_codes` seam) |
| Cortical reinstatement link | `cortex_channel_roundtrip` (`dual_cls_cortex_channel_derisk_probe.py:188`) | **REUSE** the `cortex_codes[recovered_idx]` op |
| Generalization read | `similarity_vote_infer` (`dual_cls_architecture_proof_probe.py:191`) | **REUSE** |
| No-confab moat | `RelationalFamiliarityGate` + host `query_agent/query_patient` | **REUSE verbatim** |
| Dialogue planning | `SpikingSpreadingController` | **REUSE verbatim** |
| Cross-bridge composition (V-tag) | `GradedBridge` + `cross_bridge_eval` (`multibridge_graded_derisk.py:446`) | **REUSE** |
| **The graded FALLBACK in `query_patient`/`query_agent`** | — | **NEW glue** (the generalization-in-conversation step §1.3.2) |
| **A `CortexAugmentedAgent` wrapper** wiring cortex→binder→composer→fallback | — | **NEW glue** (a thin subclass of `BrainConversationalAgent`; no `sim/` edit) |

**Net:** the integration is an *assembly + one new step* (the graded relational fallback), not a from-scratch
build. The deep pieces (graded learn, decorrelating encode, FHRR bind, reinstatement, moat) are all validated; the
new code is the wrapper and the fallback.

---

## 2. The cheap-first CAPABILITY de-risk (the heart of it)

**Goal:** prove the CAPABILITY (the conversational matrix on the learned cortex + generalization-in-conversation
with the moat intact) on the **SMALL multi-bridge ensemble already built (3–8 bridges, NOT 32)** before the
32-bridge build. The mechanism is already GO at this scale; this de-risk converts "the codes are graded and the
cross-bridge layer recalls" into "the agent talks, generalizes, and still abstains."

**Where it runs.** Reuse the exact 3-bridge × 64-concept curated-shard ensemble the multi-bridge de-risk already
builds and trains (`multibridge_graded_derisk.py`: shards animals / foods / vehicles, `GradedBridge` per shard,
seeds 42/43/44). 3 bridges exercise every cross-bridge path; 8 is more fan-out, not a new code path. **No 32-bridge
build, no new mechanism, CPU/numpy where possible (the moat + generalization reads are numpy; the graded learn +
DG encode are the small spiking ops the de-risk already runs on GPU).**

The de-risk is a NEW runner, `research/runners/cortex_conversation_capability_derisk.py`, with **two capability
gates (A, B) and a battery of anti-cheats (C)**. All must pass for GO.

### 2.1 Capability A — the conversational matrix runs on the learned cortex (no regression)

**Build:** a `CortexAugmentedAgent` over ONE shard's concepts (e.g. the 64-concept `animals` bridge), wired exactly
as §1.3: cortex codebook (`HomeostaticAssocGraph` + divnorm) → DG-decorrelated codes injected into an
`RFPhasorComposer` via `grounded_codes` → parser + composer + moat. Store a handful of SVO facts whose words are
the shard's concepts. **Run the full conversational matrix** (the SAME assertions as
`tests/test_brain_conversational_agent.py`, adapted to the shard vocabulary):

| Capability | Query | Pass criterion |
|---|---|---|
| who/what Q&A | `what_does(a, act)` → patient; `who_does(act, p)` → agent | exact match on a stored fact |
| **abstention (moat)** | `what_does(a, act)` for a NEVER-stored (a, act) | returns `None` |
| negation / yes-no | `is_it_true(a, act, p)` for AFFIRM / NEGATE / unknown | yes / no / unknown |
| one-attribute | `what_does(a, act)` for a `(adj, noun)` patient | "big apple" decoded |
| a clause | `hear_clause_fact(a, act, Clause(...))` then `what_does(a, act)` | the nested clause renders |

**GATE A (GO):** the matrix passes for ≥ 5 of 5 capability cells, multi-seed (42/43/44), **AND the abstention cell
returns `None` on every never-stored cue** (the moat must hold on the cortex-induced decorrelated codes — this is
the on-cortex analogue of the de-risk's M4 + the test suite's `is None` assertions).
**BOUNDARY:** who/what + abstention pass but one of {negation, one-attribute, clause} fails (the cortex-induced
decorrelated codes are slightly worse for the higher-load bindings — a characterized capacity boundary, not a moat
failure). **NEGATIVE:** abstention fails (any false answer on a never-stored cue) — a moat breach on the cortex
codes, which is fatal.

*Why this is the honest "no regression" test:* the conversational matrix is the validated capability
(`test_brain_conversational_agent.py`, 8/8 on the random-code composer). Gate A asserts it survives when the
composer binds **cortex-induced decorrelated codes** instead of random ones. If the DG encode of the graded cortex
codes does not produce binding-clean decorrelated codes (between-cos ≈ 0.05), the higher-load cells degrade — and
gate A catches it.

### 2.2 Capability B — generalization IN conversation (THE NEW CAPABILITY)

**This is the reason the whole arc exists.** The agent answers a query about concept X using a fact learned about a
SIMILAR concept Y, where X's fact was never stored, Y's was, and X≈Y in the graded cortex — **with the no-confab
moat intact** (it must still abstain on genuinely-absent facts, not confabulate).

**Build (within one shard, where graded similarity lives — §1 the corpus must be sharded by semantic cluster):**
- Use a graded cortex codebook with clean cluster structure (the `animals` shard's hub-mediated sub-clusters give
  cat≈dog HIGH, cat≈truck LOW — `multibridge_graded_derisk.build_bridge_corpus`). Confirm the graded margin
  (within>between cosine) by the unit check (`codebook_similarity_stats`, `dual_cls_architecture_proof_probe.py:131`)
  — this is the precondition; if the codes are not graded, B is moot.
- Store facts for the *trained* sub-cluster members only (e.g. "cat eats meat", "wolf eats meat") and
  **explicitly HOLD OUT a graded-neighbour** (e.g. "dog", never stored in any `eat` fact). This is the
  Fodor-Pylyshyn held-out test the project already uses (`run_generalization`'s held-out split,
  `dual_cls_architecture_proof_probe.py:217`).
- Query the held-out neighbour: `what_does("dog", "eat")`. The integration's graded fallback (§1.3.2) answers
  "meat" by the cortex-similarity vote when the exact relational match abstains.

**The exact measurements + GO/BOUNDARY/NEGATIVE:**

- **B1 — generalization accuracy:** held-out-neighbour relational inference accuracy (the `run_generalization`
  accuracy over the graded cortex codes, restricted to the *stored relations*). **GO:** ≥ 0.7 (chance = 1/n_props,
  i.e. ≈ 4× chance, matching the de-risk's GO bar). **BOUNDARY:** 0.5–0.7. **NEGATIVE:** ≤ chance.
- **B2 — the moat STILL abstains on genuine absence:** a held-out cue whose graded-similar neighbours are ALSO
  never stored in that relation (no similar fact exists) must yield **abstention** (`None`), AND the abstention
  floor (≥ 20 genuinely-absent cues, the `make_unknown_ap_cues` construction from the familiarity-gate runner)
  must have **ZERO false-accepts**. **NEGATIVE if any false-accept** (a moat breach — "generalize" and "don't
  confabulate" must not collide; see §4).

**GATE B (GO) = B1 GO AND B2 zero false-accepts, multi-seed (42/43/44).** This is the genuinely-new capability:
graded → generalizes; the moat → still abstains. BOUNDARY if B1 is in 0.5–0.7 with B2 clean (real but weak
generalization). NEGATIVE if B1 ≤ chance OR B2 has any false-accept.

### 2.3 The anti-cheats (C) — generalization must be REAL and the moat must NOT weaken

All three are MANDATORY (the GO is void without them — exactly the structure that made the prior de-risks
decisive). They reuse the project's already-built controls:

- **C1 — permuted-similarity control (the headline anti-cheat).** Shuffle which concepts are "similar" (decouple
  the cluster/property label from the code structure) — `run_generalization_permuted`
  (`dual_cls_architecture_proof_probe.py:266`). Generalization-in-conversation (B1) **MUST collapse to chance.**
  If it does not, the "generalization" is code overlap unrelated to meaning, not real. *Mandatory; a B1 number
  without C1 is rejected.*
- **C2 — orthogonal-codes control (generalization should NOT occur).** Re-run B with the composer/cortex using the
  project's ORTHOGONAL sparse codes (`load_orthogonal_codes` = `generate_sparse_patterns`, between-cos ≈ 0.05,
  `dual_cls_architecture_proof_probe.py:151`). B1 **MUST collapse to chance** (equidistant codes have no graded
  neighbour to generalize from — this is the decisive graded-vs-orthogonal contrast that proves B is
  similarity-DRIVEN). The conversational matrix (A) should still pass on the orthogonal codes (binding is fine on
  decorrelated codes); only the *generalization* must vanish.
- **C3 — the moat is validated ALONGSIDE the host, NOT weakened.** Exactly the familiarity-gate protocol
  (`familiarity_gate_v320_validation.py`): the learned `RelationalFamiliarityGate` is computed alongside the host
  abstention; the **host-abstain/gate-accept cell must be 0** and the **abstention-floor false-accept rate must be
  0**, multi-seed; the **lesion** (zero the gate's learned weights) must collapse the novelty separation (proving
  the decision rides the learned gate). The graded fallback (§1.3.2) is gated by this familiarity score on the
  *similar* cue, so an unknown-with-no-similar-fact stays abstained.
- **C4 — random-shard control (the generalization needs semantic co-location).** Re-shard the concepts randomly
  (the de-risk's M6, `random_shard_anticheat`, `multibridge_graded_derisk.py:364`): with the cluster structure
  destroyed, B1 **MUST collapse to chance** — confirming B measured real within-shard graded co-location, not the
  architecture.

### 2.4 The combined verdict

**GO** = Gate A (matrix passes, moat holds) **AND** Gate B (B1 ≥ 0.7 + B2 zero false-accepts) **AND** all of C1
(permuted collapses), C2 (orthogonal collapses), C3 (moat zero-breach + lesion collapses), C4 (random shard
collapses), **multi-seed 42/43/44.** ⇒ the capability is proven at small scale; the 32-bridge build is justified.

**BOUNDARY** = A passes, B1 in 0.5–0.7 with B2 + all controls clean (real but weak generalization-in-conversation —
a characterized capacity result; the build proceeds with the weakness documented).

**NEGATIVE** = any moat breach (A abstention fails, or B2 / C3 has a false-accept) — fatal, the moat is
non-negotiable; OR B1 ≤ chance (no generalization); OR C1/C2/C4 fails to collapse (the "generalization" is an
artifact, not similarity-driven). A NEGATIVE here is itself the scientific deliverable (it maps a real boundary of
the substrate) and BLOCKS the 32-bridge spend until resolved.

---

## 3. Reusable machinery vs new glue

**Reusable (specific files/classes — the de-risk mostly imports validated parts):**
- `research/runners/brain_conversational_agent.py` — `BrainConversationalAgent` (subclass for the wrapper),
  `BridgeParser` (verbatim).
- `research/runners/rf_phasor_composer.py` — `RFPhasorComposer` (the FHRR binder; inject codes via the existing
  `grounded_codes` seam, lines 86–89; `store`/`query_patient`/`query_agent`/`ask_yes_no`/`render_fact`/`unbind`/
  `_cleanup` all verbatim).
- `research/runners/learned_graded_embedding_homeostasis_probe.py` — `HomeostaticAssocGraph`,
  `learn_W_homeostatic` (the graded LEARN).
- `research/runners/learned_graded_embedding_divnorm_readout_probe.py` — `divnorm_spreading_readout` (the
  brain-based graded read-out).
- `research/runners/dual_cls_architecture_proof_probe.py` — `build_graded_codebook`, `codebook_similarity_stats`
  (graded unit-check), `assign_properties`, `similarity_vote_infer` (the generalization read), `run_generalization`,
  `run_generalization_permuted` (C1), `load_orthogonal_codes` (C2), `make_dg_encoder` (DG decorrelate).
- `research/runners/dual_cls_cortex_channel_derisk_probe.py` — `recall_identity_and_settle`,
  `cortex_channel_roundtrip` (the cortical-reinstatement link).
- `research/runners/familiarity_gate_v320_validation.py` — `RelationalFamiliarityGate`, `build_composer_and_kb`,
  `make_unknown_ap_cues`/`make_unknown_aa_cues`, `evaluate_seed` (the moat protocol, C3).
- `research/runners/multibridge_graded_derisk.py` — `GradedBridge`, `build_bridge_corpus`, `cross_bridge_eval`,
  `random_shard_anticheat` (C4), the curated shards + the already-trained 3-bridge ensemble.
- `tests/test_brain_conversational_agent.py` — the conversational-matrix assertions to mirror for Gate A.

**New glue (the only code to write — all runner-side, NO `sim/` edits):**
1. `CortexCodebook` — a thin class wrapping `HomeostaticAssocGraph` + `divnorm_spreading_readout` to expose
   `{word: graded_code}` per shard, plus the DG-encode `{word: decorrelated_code}` for the composer.
2. `CortexAugmentedAgent(BrainConversationalAgent)` — overrides composer construction to inject the
   DG-decorrelated codes via `grounded_codes`, holds the `CortexCodebook` for the generalization read, and adds
   the **graded relational fallback** in `what_does`/`who_does` (the §1.3.2 step: on host-abstain, try the nearest
   graded-similar known agent/patient, gated by the familiarity score on the similar cue).
3. `cortex_conversation_capability_derisk.py` — the runner orchestrating Gates A/B + anti-cheats C1–C4 over the
   3-bridge ensemble, multi-seed, with the GO/BOUNDARY/NEGATIVE logic of §2.4.

---

## 4. Honest risk list

**RISK 1 (the deepest — graded-vs-decorrelated at the BINDING step, not just the similarity-metric step).** The
de-risk proved the cortex MECHANISM (the codes are a graded similarity metric; cross-bridge V-tag recall works).
It did NOT prove that **generative VSA role-filler binding runs on codes routed through the decorrelated binder
while cortical reinstatement preserves generalization.** Two ways this can fail:
- *(a) Binding precision degrades.* The cortex codes are graded (correlated). Binding REQUIRES decorrelated codes,
  so the integration runs the cortex codes through a DG encode (`make_dg_encoder` / `StrongDGEncoder`) to produce
  the decorrelated codes the composer binds. If that DG encode of the *graded* cortex codes does not reach
  between-cos ≈ 0.05 (binding-clean), the FHRR unbind estimate is ambiguous and the cleanup/abstention degrades —
  Gate A (esp. the higher-load negation/one-attribute/clause cells) catches this. *Mitigation:* the architecture
  proof already validated the round-trip at binding-viable operating points (`run_roundtrip`'s
  `binding_viable = expansion_cos_mean < 0.15` sweep, `dual_cls_architecture_proof_probe.py:437`), and the
  strong-DG encode reaches between-cos ≈ 0 + repro 1.000 (`2026-06-11-dual-CLS-cortex-channel-derisk-GO.md`). But
  that was on a SYNTHETIC graded codebook; this de-risk runs it on the LEARNED cortex codes for the first time —
  which is precisely why Gate A is needed before the build.
- *(b) Generalization degrades when binding goes through the loop.* The CLS resolution puts generalization on the
  cortex side *directly* (never through the binder), so generalization should NOT degrade from binding — but a
  *recalled* fact's filler comes back via reinstatement (identity → `cortex_codes[recovered_idx]`), so a recall
  IDENTITY error reinstates the WRONG cortex code and corrupts a downstream generalization. The de-risk validated
  identity 1.000 at the strong operating point (so this is benign there), but identity errors at scale would make
  generalization-after-recall degrade gracefully — the *correct* behaviour, but it must be measured, not assumed.
  **This is the single load-bearing technical unknown the capability de-risk exists to resolve at small scale
  before the 32-bridge spend.**

**RISK 2 — does the moat survive generalization? ("generalize" and "don't confabulate" pull against each other).**
The new generalization-in-conversation step (§1.3.2) *deliberately answers* a query that the strict moat would
abstain on (X's fact was never stored). That is in direct tension with the no-confab moat, whose whole job is to
abstain on absent facts. The risk: the graded fallback fires on a cue that has NO genuinely-similar known fact,
producing a confabulation — a moat breach. *Mitigation, designed into Gate B2 + C3:* the fallback is **gated by the
familiarity score on the SIMILAR cue** (the `RelationalFamiliarityGate.novelty` on the nearest graded neighbour's
relational composite), so it answers ONLY when a graded-similar fact is genuinely familiar, and abstains when the
similar cue is also novel. B2 (zero false-accepts on the genuine-absence floor) and C3 (host-abstain/gate-accept
cell = 0, lesion collapses) are the explicit, non-negotiable checks; a NEGATIVE on either blocks the build. The
honest framing: this *redefines* abstention from "exact fact absent → abstain" to "no similar known fact → abstain"
— a strictly larger answer set, so the moat's threshold and the graded-neighbour gate must be co-validated, which
is exactly what B2 + C3 do.

**RISK 3 — the graded learn is weaker than backprop (the classic wall, Phase 2.3a).** At production scale the
brain-based learn may yield coarser graded structure than at toy scale, so generalization could be weaker. The
homeostasis finding already reached the host ceiling at toy AND V=320 single-pool scale
(`2026-06-11-learned-graded-embedding-homeostasis-GO.md`), and this de-risk measures the gap honestly via B1 — a
coarser-but-real graded structure is still a capability gain (and an honest BOUNDARY if it underperforms). *Not a
blocker, but the reason B1 has a BOUNDARY band.*

**RISK 4 — semantic-cluster sharding is required and is a production design choice.** The graded generalization is
*within-bridge* (cat≈dog only if both live in the same shard); cross-bridge relationships go through the V-tag
composition, not a shared embedding (`docs/plans/2026-06-11-semantically-structured-cortex-BUILD-PLAN.md` §"Genuine
open questions"). C4 (random-shard collapse) confirms the co-location is load-bearing. The production sharding
(co-occurrence-graph clustering of the 320 concepts into 5–32 bridges) is a build-time design choice the cheap
de-risk uses a curated stand-in for (animals/foods/vehicles) — flag for the build, not a blocker for the de-risk.

---

## 5. Recommended controller sequence (design → capability de-risk → 32-bridge build)

1. **This design** (done) — present before building. Owner reviews; trust-but-verify the load-bearing claims
   (esp. the cortical-reinstatement resolution and the moat-vs-generalization gating).
2. **The cheap-first CAPABILITY de-risk (§2), at SMALL scale (3 bridges × 64, NOT 32), multi-seed 42/43/44.** This
   is the next build step and the gate for the spend. Write the new glue (§3): `CortexCodebook`,
   `CortexAugmentedAgent`, `cortex_conversation_capability_derisk.py`. Run Gates A + B + anti-cheats C1–C4 on the
   already-trained ensemble. Estimated cost: hours, not days (the graded learn + DG encode are the small spiking
   ops the de-risk already runs; the matrix, generalization, and moat reads are numpy). **CPU/numpy where possible;
   GPU only for the graded-bridge spiking ops.**
   - On **GO** → proceed to step 3.
   - On **BOUNDARY** → proceed with the weakness documented (the build characterizes it at scale).
   - On **NEGATIVE** → STOP; the NEGATIVE (a moat breach or a similarity-not-driven generalization) is the
     scientific deliverable and must be resolved before any 32-bridge spend.
3. **ONLY THEN the 32-bridge build** (the owner's explicit-go gate, piece iii of
   `docs/plans/2026-06-11-semantically-structured-cortex-BUILD-PLAN.md`): 32 bridges × 64 = 2,048 concepts,
   production semantic-cluster sharding, the full conversational matrix + generalization-in-conversation at
   multi-bridge fan-out, the moat at 6.4× the validated 5-bridge fan-out. This is the ~2–4 week sustained push,
   justified only after the capability is proven cheap-first.

**Why this ordering is the standing opening move:** every prior decisive pivot in this project (the whitening
reframe, the missing-accumulator fix, the ventral-vs-dorsal nav root-cause, and this very dual-CLS resolution) came
from a cheap read-only proof BEFORE committing build/GPU resources. The capability de-risk is the cheapest thing
that can falsify "the cortex generalizes IN CONVERSATION with the moat intact" — and the 32-bridge build is exactly
the kind of sustained, expensive effort that warrants that gate.

---

## Summary (the three things the controller needs)

**Integration architecture (3–5 sentences).** The learned-graded cortex plugs into the existing
`BrainConversationalAgent` as a third code population alongside the parser and the `RFPhasorComposer`: a `CortexCodebook`
(the validated `HomeostaticAssocGraph` Oja learner + `divnorm_spreading_readout`) produces *graded* per-shard codes
that carry similarity, generalization reads them DIRECTLY via a k-nearest-graded-neighbour vote, while a DG
pattern-separation encode of those same cortex codes produces *decorrelated* codes injected into the composer (via
its already-built `grounded_codes` seam) so FHRR binding/unbinding/cleanup and the no-confab moat run unchanged on
clean codes. The graded-vs-decorrelated tension is resolved by **cortical reinstatement** (validated GO): the
decorrelated binder recovers the *identity* (the composer's cleanup argmax = which concept), and the cortex code for
that identity is reinstated (`cortex_codes[recovered_idx]`), so a recalled fact comes back graded — binding lives on
the decorrelated side, generalization on the graded side, linked by identity. The only new glue is a thin
`CortexAugmentedAgent` wrapper plus a graded relational *fallback* in the who/what queries (answer via the nearest
graded-similar known fact when the exact match abstains, gated by the familiarity score).

**The cheap-first capability de-risk (what to build, what to measure, GO/NEGATIVE + anti-cheats).** Build
`CortexAugmentedAgent` over ONE 64-concept shard of the already-trained 3-bridge ensemble (NOT 32 bridges) and a
runner `cortex_conversation_capability_derisk.py`, multi-seed 42/43/44. **Gate A:** the full conversational matrix
(who/what, abstention, negation, one-attribute, a clause) passes on the cortex-induced codes AND the moat returns
`None` on every never-stored cue (GO); abstention breach = NEGATIVE. **Gate B (the new capability):** held-out
graded-neighbour relational inference — answer "what does a dog eat?" → "meat" because "cat eats meat" is stored and
cat≈dog — with B1 (generalization accuracy ≥ 0.7 ≈ 4× chance) AND B2 (the moat still abstains on genuine absence,
zero false-accepts on a ≥20-cue floor). **Anti-cheats (all mandatory):** C1 permuted-similarity collapses B1 to
chance (the headline — proves it is meaning-driven); C2 orthogonal codes collapse B1 (graded-vs-orthogonal contrast
proves it is similarity-driven, while the matrix still passes); C3 the moat is validated alongside the host with
zero host-abstain/gate-accept breaches + lesion collapses; C4 random-shard collapses B1 (the co-location is
load-bearing). GO = A ∧ B ∧ C1–C4 multi-seed; NEGATIVE = any moat breach or any control failing to collapse.

**The single deepest risk.** Can generative VSA role-filler binding actually run on codes routed through the
decorrelated DG encode of the *learned* graded cortex codes (first time on learned, not synthetic, codes) while
cortical reinstatement preserves generalization — without the DG encode failing to decorrelate the graded codes to
binding-clean between-cos ≈ 0.05 (degrading binding/abstention, Gate A), or recall-identity errors corrupting the
reinstated graded code (degrading generalization-after-recall, Gate B) — AND does the moat survive the
generalization step, given that "generalize" (answer a similar-fact query) and "don't confabulate" (abstain on
absent facts) pull against each other (mitigated by gating the graded fallback on the familiarity score of the
similar cue; enforced non-negotiably by B2 + C3, where any false-accept is a fatal NEGATIVE).
