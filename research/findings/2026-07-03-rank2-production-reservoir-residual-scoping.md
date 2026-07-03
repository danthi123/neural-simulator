# RANK-2 scoping — is a "production reservoir" (thematic ROLES → ordered emission) a genuine advance, or whack-a-mole re-learning what EMERGE-63/64 already self-organized?

**Date:** 2026-07-03
**Type:** read-only scoping analyst (the four SURPASS moves applied to RANK-2 of the next-frontier gate `research/findings/2026-07-03-next-frontier-beyond-templated-constructions-research-gate.md`; anti-whack-a-mole guard per the master directive + `feedback_spiking_structure_must_self_organize`).
**Scope:** RANK-1 (EMERGE-78/79) established the fronto-striatal reservoir on the COMPREHENSION side (form → thematic role, learned not enumerated). The gate ranked RANK-2 as "the reservoir replaces the hand ROUTER too (production: thematic ROLES → ordered word emission, Dominey 2015)." The controller flagged that most of the production path is ALREADY self-organized from corpus. This scoping reads the actual code to decide whether RANK-2 buys anything, or is whack-a-mole.
**Verdict up front:** **SKIP RANK-2 as whack-a-mole. The production side is already ~fully self-organized (EMERGE-62/62b/62c function words, EMERGE-63 slot order, EMERGE-64 slot inventory, EMERGE-72 registry selector). The genuine RANK-2 residual — `_construction_by_signature` + `decision_from_emerge` — is a TRIVIAL deterministic lookup that resolves ZERO ambiguity in the current console (the reasoner's semantic decision fully determines the construction), so a "production reservoir" that learns "role-set → which construction + what order" would RE-LEARN the order (EMERGE-63) + the inventory (EMERGE-64), i.e. exactly the whack-a-mole the anti-whack-a-mole gate warns against. The genuine Dominey-2015 value (construction SELECTION under information-structure ambiguity — active vs passive vs cleft from topic/focus) does NOT EXIST in the current corpus/reasoner (there is no topicalization, no voice alternation, no ambiguity where one message maps to several constructions). Highest-leverage next move: the RANK-1.5 SPIKING-LSM PORT of the already-GO EMERGE-78/79 rate reservoir onto the recurrent RF/Izhikevich substrate — the "fully-spiking one-brain" directive, on already-pre-registered ground.**

---

## MOVE 1 — ISOLATE + QUANTIFY the genuine RANK-2 residual (what on the PRODUCTION side is STILL hand-designed?)

I read the full production chain: `_emerge59_spiking_broca_frame_slots_derisk.py` (FRAMES + FrameSlotCQ + `decision_from_emerge`), `_emerge63_corpus_taught_slot_order_derisk.py` (slot ORDER), `_emerge64_mine_slot_inventory_derisk.py` (slot INVENTORY), `_emerge72_construction_registry_derisk.py` (the signature-keyed registry + selector), the `_emerge78/79` reservoirs, and the console selection path `_emerge58_unified_fluent_console.py`.

### What is ALREADY self-organized on the PRODUCTION side (NOT the residual)

| Production sub-problem | Where | Self-organized? | Anti-cheat that collapses it |
|---|---|---|---|
| **S2 — the function-word SET** (which tokens are closed-class: the/can/does/not/to/on) | EMERGE-62/62b/62c | ✅ DISCOVERED from distributional statistics (frequency + context-coverage + phrase-boundary + morphological-invariance "Goldilocks") | frequency/position/morphology-shuffle |
| **S1b — the per-construction slot ORDER** (det<subj<func<verb; does<not) | EMERGE-63 (`corpus_precedence` `_emerge63:191-209`, `order_from_precedence` `:212-235`) | ✅ LEARNED from corpus pairwise role-precedence; renders on real spikes via competitive-queuing rate ranking | shuffled-corpus / no-corpus (`_emerge63:406-423`) |
| **S1a — the per-construction slot INVENTORY** (which typed slots a construction licenses) | EMERGE-64 (`label_sentence` `:147-194`, `mine_inventory` `:262-298`) | ✅ MINED by role-type signature from the corpus | permuted-mining / no-corpus (`_emerge64:500-513`) |
| **Broadened inventory + a general SELECTOR** (5→7 constructions; signature-keyed registry) | EMERGE-72 (`mine_registry` `:238-264`, `ConstructionRegistry` `:427-464`) | ✅ registry mined from corpus; the "3-frame router" de-hard-coded | permuted-corpus / cross-construction / no-corpus (`_emerge72:640-668`) |
| **Every word rendered ON SPIKES** (content + function) | EMERGE-67/68/69 | ✅ A→W population read-out, gate-first moat | lesion → collapse |

So on the production side, **the function words, the slot order, the slot inventory, and even the broadened construction registry are already emergent/corpus-mined**, with the project's decisive input-destruction anti-cheats collapsing each. This is a genuinely large, already-done base — exactly what the controller flagged.

### The residual pinned to EXACT bytes — and it is TRIVIAL

Two artifacts remain hand-designed on the production side:

**(1) `_construction_by_signature()` (`_emerge72:290-294`) — a deterministic ground-truth-signature → construction-id map.**
```python
def _construction_by_signature():
    return {_gt_signature(name): name for name in CONSTRUCTION_NAMES}
```
This is used in two places (`_emerge72:451-455 ConstructionRegistry.build`): to (a) SCORE the mined registry against ground truth, and (b) ROUTE a mined signature to a construction id. Crucially, the runner comment itself concedes it: *"a mined signature that MATCHES a ground-truth construction's signature gets that id (validation routing); ANY other dominance-clearing signature would get a fresh anonymous id"* (`_emerge72:448-451`). **It is a lookup, not a decision.** The construction id it assigns is definitionally the mined signature's own identity; there is no learning, no ambiguity, no competition. The registry already mines the structure — this just names it.

**(2) `decision_from_emerge()` (`_emerge59:330-343`) — the reasoner-polarity → frame map.**
```python
def decision_from_emerge(gate, subject=None, verb=None, polarity=None, negated_modal=False):
    if gate == "ABSTAIN":       return {"gate": "ABSTAIN"}
    if negated_modal:           return {"gate": "ANSWER", "frame": "F_NEGMOD", ...}
    if polarity == "negate":    return {"gate": "ANSWER", "frame": "F_INTR", ...}
    return {"gate": "ANSWER", "frame": "F_MODAL", ...}          # affirm
```
Three deterministic `if` branches keyed on the reasoner's SEMANTIC decision. In production (`_emerge58_unified_fluent_console.py:114-140`), the reasoner (EMERGE-54) already emits `(gate, polarity, property)` — affirm / negate / deny / abstain — read from the taxonomy. `EMERGE-72:303-309 decision(...)` is the generalized form (`construction=` passed explicitly). **The mapping from a semantic decision to a construction is one-to-one and total: there is no case where the same `(predicate, roles, polarity)` decision could be expressed by two different constructions and the system must LEARN which to pick.** The reasoner's decision IS the construction choice.

### Is `_construction_by_signature` the only hand residual, and is it trivial or a genuine learning problem?

**It is the only production-side selection residual, and it is TRIVIAL — a deterministic lookup, NOT a learning problem, at this corpus/reasoner.** A reservoir buys nothing over it here because there is nothing to disambiguate: the mined signature already uniquely identifies the construction, and the reasoner's polarity already uniquely selects it. A "production reservoir" that maps `role-set → (construction, order)` would have to REPRODUCE:
- the slot ORDER — which EMERGE-63 already learns from corpus precedence (and proves load-bearing via shuffled-corpus collapse); and
- the slot INVENTORY — which EMERGE-64 already mines (permuted-mining collapse).

That is the definition of **whack-a-mole**: re-implementing already-self-organized, already-anti-cheated machinery inside a reservoir, for no new capability. (The note in `_emerge72:38-40` already reads Dominey-Hinaut production as "SELECTING the construction to express thematic roles" — but the current selector is a total function of the reasoner decision, so there is no selection problem to learn.)

---

## MOVE 2 — REFRAME: what does the Dominey-2015 PRODUCTION reservoir actually ADD, and does that value EXIST here?

**Dominey 2015 (Brain & Language, "Corticostriatal response selection in sentence production"):** production = SELECTING the construction to express a predicate + thematic roles from a **"focus hierarchy"** — i.e. WHICH argument is topicalized/focused drives active vs passive vs cleft vs dislocation ("the dog chased the cat" vs "the cat was chased by the dog" vs "it was the cat that the dog chased"). The reservoir read-out then activates word-coding units in the correct order. **The genuine value is (a) construction SELECTION under information-structure ambiguity — a real learning problem because ONE message (same predicate + roles) maps to SEVERAL surface constructions and the choice is driven by topic/focus, NOT (b) re-learning slot order (EMERGE-63 already did that) or inventory (EMERGE-64 already did that).**

**Does that value exist in the current system? NO — decisively.** Three checks against the actual code:

1. **No voice/information-structure alternation in the corpus.** `build_stream` (via `_emerge62`, reused everywhere) attests active constructions only; the mined constructions (`CONSTRUCTIONS`, `_emerge72:272-282`) are F_MODAL / F_INTR / F_NEGMOD / C_PPGOAL / C_PPLOC (+ EMERGE-73/74/77's attributive/transitive/ditransitive) — each a DISTINCT predicate-argument shape, none a topic/focus VARIANT of another. There is no "same roles, different construction" pair anywhere. So there is no focus hierarchy to read and no selection ambiguity to resolve.

2. **The reasoner decision is a total function to the construction.** `decision_from_emerge` (`_emerge59:330-343`) and the console reasoner (`_emerge58:114-140`) map `polarity`/`negated_modal` deterministically to exactly one frame. There is no branch where the reasoner emits a decision that under-determines the construction.

3. **Even EMERGE-78/79 (comprehension) found the necessity CONTINGENT.** EMERGE-78's non-local relative-clause result was honestly downgraded to a *constructed proof-of-mechanism* (the relativizer "that" was OOV/verb-colliding; `_emerge78:452-459` scope note). EMERGE-79 tested the UNCONTINGENT variable-distance version. The lesson: reservoir NECESSITY requires a genuine non-local/ambiguous dependency that a local rule can't handle. On the PRODUCTION side, no such ambiguity is even constructible from the current corpus + reasoner — construction choice is fully local to the reasoner's semantic decision.

**Reframe verdict:** the Dominey-2015 production reservoir's genuine value is **information-structure-driven construction selection under ambiguity**, and that phenomenon **does not exist in this project's corpus or reasoner yet**. Building a production reservoir now would learn "role-set → construction+order" over a set where the mapping is already one-to-one and already self-organized (EMERGE-63/64) — a solution in search of a problem. This is testing the WRONG hypothesis: RANK-2's premise ("the router is the residual") is false here, because the router is a trivial total function, and the interesting selection problem (topic/focus alternation) is a MISSING corpus/reasoner phenomenon, not a missing mechanism.

---

## MOVE 3 — RANK cheap-first options

Because the genuine RANK-2 residual is TRIVIAL (a deterministic lookup) and the interesting selection problem is absent from the corpus, the anti-whack-a-mole gate's own instruction applies: **say so honestly and recommend the higher-value alternative.** Three options, ranked:

### ★ OPTION A (RECOMMENDED) — RANK-1.5: the SPIKING-LSM PORT of the EMERGE-78/79 rate reservoir onto the recurrent RF/Izhikevich substrate
- **Why highest-leverage:** RANK-1 (EMERGE-78/79) validated the reservoir form→role mechanism at RATE level (CPU/numpy). The project's END-STATE directive is **fully-spiking on the one-brain substrate** (`feedback_end_state_fully_spiking_one_brain_path_by_efficiency`, `feedback_move_everything_to_shared_spiking_substrate`). A rate reservoir with a numpy `tanh` recurrent pool (`_emerge78:155-170 Reservoir`) is a tracked shortcut — the honest next move is to realize it as a genuine **liquid-state machine**: drive a recurrent RF/Izhikevich pool on a real `SimulationBridge` with the discovered closed-class stream, read the population final-state via the validated A→W/population read-out, keep the gate-first moat. This is EXACTLY the "reservoir + trained read-out" the EMERGE-6b generation-stability gate pre-registered, and it lands the comprehension win on the shared spiking substrate.
- **Cheapest single-variable de-risk:** port ONLY the reservoir (fixed-random recurrent RF/Izhikevich pool + ridge/population read-out) — reuse EMERGE-78's Encoder (discovered closed-class input), the corpus stream, and the EXACT EMERGE-78 anti-cheat harness (scramble→chance, closed-class-lesion→collapse, held-out-construction generalization, memorization-floor). GO bar: the spiking LSM matches the rate reservoir's form→role accuracy on the trained shapes AND the (uncontingent, EMERGE-79) graded-memory advantage over a fixed window, on real spikes, 6-seed. This is a substrate port of an already-GO mechanism (the low-risk kind), on already-pre-registered ground, advancing the fully-spiking directive.
- **Anti-cheats:** the EMERGE-78/79 set, verbatim, on the spiking read-out (scramble, non-degenerate closed-class-identity lesion, held-out-construction, mark-lesion for the variable-distance flip; moat abstains preserved).

### OPTION B — RANK-3: BOUNDED RECURSION via a WM-multiplexed buffer (theta-gamma / assembly-calculus stack)
- **Why second:** this is the honest "true productivity" rung — the plain reservoir's fading memory caps embedding depth. It is where genuine generative capability past the flat/bounded inventory comes from (catalog N.15 theta-gamma; Mitropolsky assembly-calculus center-embedding). But it is a real mechanism BUILD (higher variance) and correctly sits AFTER the reservoir is on the spiking substrate. Deferring it below Option A is right: port the validated reservoir first, then extend it with the stack, rather than building the stack around a numpy rate reservoir that itself still needs porting.

### OPTION C (NOT RECOMMENDED) — build RANK-2 as originally ranked (production reservoir over the current inventory)
- **Why not:** whack-a-mole. It would learn "role-set → construction+order" over a mapping that is already one-to-one and already self-organized (EMERGE-63 order + EMERGE-64 inventory + EMERGE-72 registry), resolving zero ambiguity, adding zero capability. If ever pursued, it should ONLY be after the corpus + reasoner are extended to attest a genuine information-structure alternation (voice: active/passive, or cleft/topicalization) — i.e. a corpus where ONE message maps to SEVERAL constructions and topic/focus drives the choice. THAT is the real Dominey-2015 selection problem, and it is a **corpus/reasoner extension**, not a mechanism the reservoir alone supplies. Building the reservoir before the ambiguity exists is premature.

---

## MOVE 4 — VERDICT: highest-leverage next move (framed to start immediately)

**SKIP RANK-2 as whack-a-mole. Do OPTION A — the RANK-1.5 spiking-LSM port of the EMERGE-78/79 rate reservoir onto the recurrent RF/Izhikevich substrate.**

Honest reasoning, so the controller can act immediately:

1. **RANK-2 is mostly-already-done, and its residual is trivial.** The production side's function words (EMERGE-62), slot order (EMERGE-63), slot inventory (EMERGE-64), and construction registry (EMERGE-72) are ALL self-organized from corpus with decisive anti-cheats. The only hand residuals are `_construction_by_signature` (`_emerge72:290-294`) and `decision_from_emerge` (`_emerge59:330-343`) — both deterministic total-function lookups that resolve NO ambiguity in the current console (the reasoner's semantic decision fully determines the construction). A production reservoir over this would re-learn the order + inventory that EMERGE-63/64 already self-organized — the exact whack-a-mole the anti-whack-a-mole gate warns against.

2. **The genuine Dominey-2015 value (information-structure-driven construction selection under ambiguity — active vs passive vs cleft from topic/focus) DOES NOT EXIST here.** The corpus attests no voice/focus alternation (no "same roles, different construction" pair), so there is no focus hierarchy to read and no selection to learn. That value would require a CORPUS + REASONER extension to attest the alternation FIRST — not a reservoir. Building the reservoir before the ambiguity exists is premature.

3. **The highest-leverage move is porting the ALREADY-GO reservoir (EMERGE-78/79) to spikes** — a low-risk substrate port of a validated mechanism that directly serves the non-negotiable fully-spiking-one-brain end state, on the pre-registered "fixed recurrent pool + trained read-out" ground (EMERGE-6b). It reuses EMERGE-78's Encoder + corpus stream + anti-cheat harness; the single new variable is the recurrent substrate (RF/Izhikevich LSM pool + population read-out) in place of the numpy `tanh` reservoir. GO bar: matches the rate reservoir's form→role accuracy + the uncontingent graded-memory advantage on real spikes, 6-seed, moat intact.

4. **RANK-3 (bounded recursion, theta-gamma / assembly-calculus stack) is the honest true-productivity rung and stays queued AFTER Option A** — build the stack around a spiking reservoir, not a numpy one.

**The goal is the emergent brain, not capability-count.** RANK-2 as originally framed would add a construction-selection MECHANISM where no selection PROBLEM exists — a mechanism in search of a phenomenon. The spiking-LSM port advances the actual frontier (the fully-spiking substrate) using work already de-risked. If the controller wants to pursue the genuine Dominey-2015 selection win later, the prerequisite is a corpus/reasoner that attests an information-structure alternation (voice/focus) — flag that as the RANK-2* re-entry condition, not a reservoir build.

---

## Files read (cited)
- `research/runners/_emerge72_construction_registry_derisk.py` — `_construction_by_signature` (`:290-294`, the residual), `mine_registry` (`:238-264`), `ConstructionRegistry.build` routing (`:440-456`), `RegistryProducer`/`decision` (`:303-309`, `:348-401`), the Dominey-Hinaut selector note (`:38-40`).
- `research/runners/_emerge59_spiking_broca_frame_slots_derisk.py` — `FRAMES` (`:98-105`), `FrameSlotCQ` (`:209-302`), `decision_from_emerge` (`:330-343`, the polarity→frame total function), moat producer (`:309-327`).
- `research/runners/_emerge63_corpus_taught_slot_order_derisk.py` — slot ORDER self-organized: `corpus_precedence` (`:191-209`), `order_from_precedence` (`:212-235`), shuffled/no-corpus controls (`:406-423`).
- `research/runners/_emerge64_mine_slot_inventory_derisk.py` — slot INVENTORY self-organized: `label_sentence` (`:147-194`), `mine_inventory` (`:262-298`), permuted-mining/no-corpus controls (`:500-513`).
- `research/runners/_emerge78_reservoir_form_to_role_derisk.py` — the RANK-1 rate reservoir (`Reservoir` `:155-170`, `Encoder` `:137-152`, `_slot_data`/`_fit_slots` `:179-204`); necessity-is-contingent scope note (`:452-459`).
- `research/runners/_emerge79_reservoir_variable_distance_derisk.py` — the uncontingent variable-distance follow-on (`:1-45` scope).
- `research/runners/_emerge58_unified_fluent_console.py` — the reasoner decision `(gate, polarity, property)` (`:114-140`), `_render_emerge` (`:247-253`) — confirms construction choice is a total function of the semantic decision.
- `research/findings/2026-07-03-next-frontier-beyond-templated-constructions-research-gate.md` — the parent gate (RANK-1/2/3 definitions).
