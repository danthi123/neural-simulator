# Kind-vs-instance (type/token) representation — scoping

**2026-07-01 (read-only deep-research + catalog review; the standing opening move for a new direction). NO code
edited, NO GPU run.** The owner wants the brain to eventually distinguish a KIND ("dogs" — the generic type) from
an INSTANCE ("the dog" — a specific referred-to individual). Today the conversational composer represents concept
KINDS only: the code `self.concepts["dog"]` is the single phasor for the type, and "the dog", "a dog", and "dogs"
all collapse onto it. This document diagnoses exactly what is missing, reframes it against the brain's own
semantic-vs-episodic / object-file machinery, ranks cheap-first mechanisms that reuse existing project pieces, and
gives an honest verdict.

Scope note: the composer is the FHRR idealization (a principled stand-in for a VSA-like cortex, not a functional
cortex). Everything below works AT the composer/agent layer (representation + routing), reuse-by-import, and does
not require the deferred dendritic rewrite. The no-confab moat (a plus, not a hard gate — `feedback_moat_not_hard`)
is preserved throughout by construction.

---

## 1. DIAGNOSIS — what specifically is missing

### What the composer does today (read from `rf_phasor_composer.py`, `brain_conversational_agent.py`)
- **Concepts are KIND codes.** `self.concepts = {word: rng.uniform(0,1,D)}` — one fixed per-seed phasor per word.
  There is exactly one "dog" vector; it is the type.
- **Facts are role-filler bundles keyed by exact string.** `store(agent, action, patient)` binds `role⊗filler`
  and bundles into a KB entry; every query (`query_patient`, `query_agent`, `ask_yes_no`, `render_fact`) matches by
  **exact string equality** on the decoded cue role (e.g. `unbind(comp,"agent") == agent`). This is the important
  structural fact for what follows: the KB does not care whether a subject string names a kind or an instance — it
  is just a symbol with a code. A distinct instance token is therefore a first-class citizen the moment it has (a)
  a distinct string and (b) a code.
- **`grounded_codes` already lets an entity's code come from elsewhere** (perception, Phase 8; the stream cortex,
  Phase 11). So "supply a code for a new token" is a validated, exercised path.
- **The agent never MINTS a token.** `hear("dog chase cat")` parses to flat SVO strings and stores `agent="dog"`.
  There is no step that says "this 'the dog' refers to a specific individual — create/lookup a token for it."
- **Multi-turn WM holds referents, but as KINDS.** `MultiTurnAgent` writes the patient string ("cat") into the
  `SpikingLoopContextBuffer` and resolves "it" back to that string. The held referent is the *kind symbol* "cat",
  not a distinct individual — so "the dog I saw" and "dogs in general" resolve to the same "dog".

### The three things an INSTANCE needs that a KIND code does not provide
An instance `the-dog-I-saw` must:
- **(a) be tagged as "a dog"** — an `isa` link to the kind, so it INHERITS the kind's facts ("dogs eat meat" ⇒ this
  dog eats meat) without re-teaching them per individual;
- **(b) carry its OWN episodic facts** — "the dog I saw was brown", "it chased the cat" — bound to the token, not
  the kind (so it does NOT make ALL dogs brown);
- **(c) be the thing a definite "the dog" / a pronoun resolves to** — the discourse-current individual, distinct
  from the generic kind that "dogs" / "a dog eats meat" queries.

### How much is ALREADY provided
Substantially more than expected — this is largely wiring, not a new mechanism:
- **(a) isa/inheritance:** the composer's exact-string KB gives inheritance almost for free. If the token has its
  own string `"dog#1"` and its own code, then `store("dog#1","isa","dog")` is a normal fact, and a query about the
  instance can *fall back to the kind* when the instance has no own fact for that relation (a 2-line routing rule,
  §3.1). The KB scan already abstains cleanly when there is no match — inheritance is "on instance-miss, retry with
  the isa-parent." No binding-algebra change.
- **(b) own episodic facts:** already works. `store("dog#1","be","brown")` and `store("dog#1","chase","cat")` are
  ordinary facts keyed on the distinct token; they do NOT touch the kind's facts. The composer needs a distinct code
  for `"dog#1"` (the default `rng.uniform` mints one automatically for any new string; or a grounded/perceived code
  via `grounded_codes`, Phase 8).
- **(c) discourse resolution:** the `SpikingLoopContextBuffer` (`MultiTurnAgent`, anaphora GO 2026-06-17) is exactly
  a **working-memory instance-token store** — it holds "which individual is currently in mind" across turns. Today
  it holds the kind string; pointing it at an instance token instead is a labeling change, not a new mechanism. This
  is the closest existing "instance/referent" machinery, and it maps 1:1 onto an object-file (§2).
- **The engram-tag API** (`start_engram_recording`/`commit_engram_tag`/`stimulate_tag`, catalog D.14) is the
  substrate-level "a specific ensemble = a specific memory/instance" primitive — a heavier, more biologically
  committed instance mechanism (§3, ranked lower for the conversational layer).

**⇒ Net diagnosis:** the missing piece is small and mostly at the AGENT layer: (1) a lightweight **instance token**
(a distinct symbol + code) minted/looked-up when discourse introduces a specific individual; (2) an **isa link** to
its kind with **on-miss inheritance** routing in the composer; (3) a **definite/generic router** in the parser/agent
that sends "the dog"/pronoun → the discourse-current token and "dogs"/"a dog" (generic) → the kind. The
binding/store/recall algebra and the moat are untouched. Phase 11 already flagged this exact next nuance
("Generic/definite in the console … 'the dog' → a held referent, else the kind (clarify 'which dog?')").

---

## 2. BIOLOGY REFRAME — the type/token split is one of the brain's oldest divisions

The kind-vs-instance distinction is not a linguistic add-on to bolt on; it is the semantic-vs-episodic memory split,
which the catalog and Kandel treat as two distinct systems.

- **Semantic memory = KINDS, in cortex.** Generic, decontextualized knowledge ("dogs eat meat", "a dog is an
  animal") is neocortical, incremental, and context-free. This is exactly what the composer's kind codes + kind
  facts already are (and what the PPMI stream cortex learns — CLAUDE.md 2026-06-15). Catalog D.22 (O'Keefe-Nadel
  locale-vs-taxon dual systems) makes the property list explicit: the semantic/taxon store is **incremental,
  context-bound, kind-general**.
- **Episodic memory = INSTANCES/TOKENS, in hippocampus.** A specific experienced individual bound into its
  spatiotemporal context ("the dog I saw at the park yesterday, which was brown") is medial-temporal-lobe /
  hippocampal: **one-trial, item-in-context, individuated** (catalog **D.01** episodic encode/retrieve, Kandel 6e
  Ch 52 pp 1296–1302, Tulving 1972; **D.02** Eichenbaum–Cohen relational binding — "items-in-context, distinguishes
  overlapping episodes that share elements (same restaurant, different visits) without interference", Ch 52 pp
  1301–1302). "Same restaurant, different visits" IS the type/token problem: one KIND (restaurant), many episodic
  TOKENS that must not interfere. **D.21** (cognitive-map theory) frames the human hippocampus as binding
  items-at-places-at-times — the individuating context that turns a kind into a specific instance.
- **The engram** (catalog **D.14 / D.63**, Tonegawa; Kandel 6e Ch 54 pp 1357–1359) is the substrate realization of
  a *specific* memory token: a sparse activity-tagged ensemble that IS one particular experience; reactivating it
  recalls that instance. This is the biologically-committed "instance = a specific ensemble" primitive.
- **Object-file / discourse-referent theory (the psychology of tokens).** Kahneman & Treisman (1984; Kahneman,
  Treisman & Gibbs 1992, *Cognitive Psychology* 24:175–219) propose **object files**: *temporary episodic
  representations of specific objects, kept separate from the long-term recognition (type/kind) network* — and their
  binding is explicitly a **type-token individuation** framework (repetition-blindness experiments show subjects
  struggle to individuate two TOKENS of the same TYPE). A file is opened for a specific individual, accumulates
  properties bound to THAT individual, and is addressed by spatiotemporal continuity — precisely a working-memory
  instance token. In linguistics, Kamp's **Discourse Representation Theory** (DRT) is the same idea for language: a
  definite/indefinite NP introduces a **discourse referent** (a token in the discourse model) that later pronouns
  and definites bind to; a generic ("dogs") predicates over the kind, not a referent.
- **Inheritance = an instance "isa" its kind.** The token inherits the kind's semantic facts (a specific dog is
  still a dog and eats meat) while overlaying its own episodic facts. This is the standard cognitive
  taxonomy/inheritance hierarchy (Collins & Quillian 1969 semantic networks; Rogers & McClelland 2004 semantic
  cognition) and maps onto hippocampal (instance) + neocortical (kind) complementary learning systems (McClelland,
  McNaughton & O'Reilly 1995 — CLS, already invoked project-wide).

**⇒ The reframe:** the composer today is a *semantic* store (kinds, cortex). An instance is an *episodic* token
(hippocampus / object file) that **isa** a kind and inherits from it. The project already has: a semantic store (the
composer), a working-memory token holder (the `SpikingLoopContextBuffer` = an object file), and an episodic-ensemble
primitive (the engram-tag API). Giving the substrate a kind/instance distinction is *connecting these three*, not
inventing a mechanism.

---

## 3. RANKED cheap-first mechanisms (reuse project machinery)

### #1 — Instance token = a fresh symbol bound to its kind by `isa`, with on-miss inheritance + a definite/generic router (RECOMMENDED)
**Mechanism.** Represent a specific individual as a *distinct token symbol* (e.g. `"dog#1"`) with its own concept
code, linked to its kind by a stored `isa` fact `store("dog#1","isa","dog")`. The instance carries its own episodic
facts (`store("dog#1","be","brown")`). Queries resolve **instance-first, kind-fallback**: `what_does("dog#1", act)`
tries the instance's own facts; on the composer's existing clean abstention (no match), retry with the `isa`-parent
("dog") — inheritance. A **definite/generic router** in the agent decides which symbol a surface NP maps to:
- **"the dog" / pronoun** → the discourse-current instance token (from the WM object-file, #2); if none is in
  discourse, fall back to the kind (Phase-11's stated behavior) or ask "which dog?";
- **"dogs" / "a dog" (generic) / bare generic statements** → the KIND symbol "dog" (unchanged behavior — plural
  normalization is already wired, Phase 11);
- **"a dog" (specific-introducing, "I saw a dog")** → MINT a new instance token (`dog#N`) + store `isa dog`, and
  make it the discourse-current referent.

**Catalog cite.** D.01/D.02 (episodic instance vs relational binding), D.21/D.22 (semantic-vs-episodic dual system),
object-file / DRT discourse-referent theory (§2); inheritance = Collins-Quillian / Rogers-McClelland; CLS
(McClelland 1995) for instance(HC)+kind(cortex).

**Reusable pieces (almost all of it already exists).**
- The composer's exact-string KB stores/queries any token with **zero algebra change** (a new string auto-gets a
  code via `rng.uniform`; or a grounded/perceived code via `grounded_codes`, Phase 8/11).
- `isa` is an ordinary relation/action in the existing role set — no new role needed. (Optionally reuse the unused
  `attribute` role or add an `isa` verb to the lexicon; the cheapest is treating `isa` as an action.)
- Inheritance = the composer's ALREADY-CLEAN abstention: on `query_patient(instance,...) is None`, retry
  `query_patient(kind,...)`. A ~5-line agent-layer wrapper; the moat is preserved (if BOTH miss → abstain).
- The definite/generic router lives in the agent's `hear`/query entry points (a lexical check: definite article /
  pronoun vs plural/generic — Phase 11's plural normalization is the existing hook).
- `MultiTurnAgent`/`SpikingLoopContextBuffer` supplies "the discourse-current individual" (#2).

**Cheap-first de-risk (numpy/CPU, ≥3 seeds, reuse-by-import, NO `sim/` edit).**
`_kindinst_isa_inheritance_derisk.py`:
- Teach the KIND: `store("dog","eat","meat")` (a generic dog-fact).
- Introduce an INSTANCE: mint `dog#1`, `store("dog#1","isa","dog")`, `store("dog#1","be","brown")`.
- Assert the target behaviors:
  1. **"what is the dog?" / "what color is the dog?"** (definite → the instance) → **brown** (the instance's OWN
     episodic fact), NOT the generic.
  2. **"what do dogs eat?"** (generic → the kind) → **meat** (the kind fact).
  3. **INHERITANCE:** "what does the dog eat?" → **meat** via the `isa` fallback (the instance has no own eat-fact,
     so it inherits the kind's). The transcript `"I saw a dog. the dog was brown. what is the dog? → brown; what do
     dogs eat? → meat"` is the headline.
  4. **NO LEAKAGE:** after `store("dog#1","be","brown")`, "what color are dogs?" (the kind) does NOT return brown
     (the instance fact is bound to the token, not the kind).

**Anti-cheats.**
- **Isa-lesion / inheritance load-bearing:** remove the `isa` fact → the inheritance query ("what does the dog
  eat?") must now MISS (abstain), proving inheritance rides the `isa` link, not a coincidental kind match.
- **Instance-fact load-bearing:** corrupt/remove the instance's own fact → "what is the dog?" abstains (not a
  fabricated brown).
- **Generic/definite routing control:** a bare generic query never resolves to an instance token, and a definite
  with NO discourse referent falls back to the kind (or asks) — never silently picks a random instance.
- **Moat 0-FA:** an un-introduced individual ("the cat", never seen) → abstain.
- **Distinctness control:** two instances of the same kind (`dog#1` brown, `dog#2` black) do NOT interfere — each
  answers its own color (the "same restaurant, different visits" D.02 test).

**Why #1:** it is nearly free (agent-layer routing over the existing exact-string KB + the existing WM), it directly
answers the owner's "which dog?" example, and it stays inside the validated moat. Honest limit: instance codes are
symbolic (auto-minted) unless grounded via perception — semantic *similarity* between an instance and its kind is
NOT automatic in the FHRR codes (a `dog#1` random code is not near "dog"); inheritance here is via the explicit
`isa` link (symbolic), which is exactly how a discourse model / DRT does it, and is sufficient for the
conversational goal.

### #2 — The working-memory OBJECT FILE: hold the discourse-current INSTANCE (not the kind) in the SpikingLoopContextBuffer
**Mechanism.** Point the existing multi-turn WM at instance tokens: when discourse introduces "a dog"/"the dog",
write the *token* (`dog#1`) into the `SpikingLoopContextBuffer` (an object file — a temporary episodic token in WM);
"the dog"/"it" in a later turn reads back the token, and the query routes instance-first (via #1). This upgrades
anaphora from "resolves to the kind symbol" to "resolves to the specific individual", so "the dog I saw was brown"
vs "dogs in general" become genuinely different resolutions.

**Catalog cite.** Object-file theory (Kahneman-Treisman 1984/1992 — object files ARE temporary episodic tokens in
WM); DRT discourse referents; catalog G (working memory) + D.01 (episodic). The 2026-06-17 anaphora GO already
validated the WM-carries-a-referent mechanism; this re-labels the carried thing as an instance token.

**Reusable pieces.** `MultiTurnAgent`, `SpikingLoopContextBuffer`, `_write_referent`/`held_referent` — all validated
(Phase 4 GO). The change is at the *symbol* written, plus the instance-first routing of #1.

**Cheap-first de-risk.** Extend the Phase-4 multi-turn de-risk: "I saw a dog. it was brown. what is it?" → the WM
holds `dog#1`, "it" resolves to `dog#1` → brown; "what do dogs eat?" (no pronoun, generic) → the kind → meat. Same
anti-cheats as Phase 4 (WM-lesion collapses; empty-WM abstains) PLUS the kind/instance distinctness of #1.

**Note on multi-referent.** Multiple *co-present* instances ("the dog and the cat … it") hit the known
multi-referent-disambiguation boundary — resolved only by the validated biased-competition WTA
(`enable_biased_competition`, 2026-06-19; 2026-06-17-multireferent-disambiguation-NEGATIVE mapped the requirement).
That path is a drop-in when multi-instance dialogue is prioritized; single dominant instance works today. #2 does
NOT re-open that boundary — it inherits the existing capability.

### #3 — Episodic ENGRAM tag as the substrate-level instance (a specific ensemble = a specific individual)
**Mechanism.** Realize an instance as a Tonegawa engram: `start_engram_recording("dog#1")` while the individual is
perceived/attended, `commit_engram_tag` to bind a sparse ensemble, `stimulate_tag` to reactivate that specific
individual. The instance is then a substrate object (an ensemble), the most biologically committed form. Its own
episodic facts consolidate onto the tagged ensemble; the kind stays in the semantic (composer/cortex) store.

**Catalog cite.** D.14/D.63 (engram cells, Tonegawa; Kandel 6e Ch 54 pp 1357–1359); D.01 (episodic system); CLS
(instance in HC-ensemble, kind in cortex).

**Reusable pieces.** The full engram-tag API in `bridge.py` (SHIPPED, 12/12 tests, catalog D.14). The Tier-3
live-and-remember loop (2026-06-30) already tags perceived objects — a perceived *specific* object is naturally an
instance token, so #3 composes with the embodied loop.

**Cheap-first de-risk.** On a small bridge, tag two individuals of the same kind (`dog#1`, `dog#2`), store distinct
episodic facts, reactivate each by tag, confirm no cross-talk (D.02 "same restaurant, different visits"), and route
kind queries to the semantic store.

**Why ranked #3.** Heavier (substrate ensembles + consolidation bookkeeping) than the symbolic-token approach, and
not needed to answer the conversational "which dog?" question. It is the RIGHT mechanism when the instance must be a
*perceived, consolidated episodic* memory (the embodied capstone — "the specific dog I saw during my walk"), which
is why it should be the follow-on once #1/#2 close the conversational layer.

### #4 — Composite/typed instance code (bind the kind code + an episodic-context tag) — OPTIONAL polish
**Mechanism.** Instead of an unrelated random `dog#1` code, MINT the instance code as `kind ⊗ token_id` in the FHRR
algebra (a bound composite of the kind's code and a unique episodic tag), so the instance code is algebraically
*derived from* its kind. This gives partial code-level similarity to the kind and lets `isa` be read by unbinding
the kind factor, rather than a separate stored fact.

**Catalog cite.** VSA type-token binding (Plate HRR; Eliasmith SPA "semantic pointers" as bound structures); the
composer's existing bind/unbind is exactly this operation.

**Reusable pieces.** `_bind`/`_unbind_phases` (the composer's core ops), used unchanged.

**Cheap-first de-risk.** Compare a `kind⊗id` instance code vs a random instance code on the #1 tests + a "read the
kind by unbinding the id" check. Anti-cheat: the derived code must still be *distinct enough* that two instances
don't collide (the FHRR capacity check).

**Why ranked #4.** A representational nicety, not required — #1's explicit `isa` link already delivers inheritance
and is simpler/clearer. Worth it only if code-level kind-similarity of instances becomes important (e.g. instance
generalization). Honest caveat: adds binding load (an extra factor) and must be capacity-checked.

---

## 4. VERDICT — honest

**Cheaply achievable NOW (at the conversational/composer layer, reuse-by-import, NO `sim/` edit):** a lightweight
kind-vs-instance distinction via **#1 (instance token + `isa` link + on-miss inheritance + definite/generic
router)** plus **#2 (the WM object-file holds the discourse-current instance, not the kind)**. This directly
delivers the owner's example — *"I saw a dog. the dog was brown. what is the dog? → brown (the instance); what do
dogs eat? → meat (the kind)"* — with inheritance, no cross-leakage, and the moat intact. The composer's exact-string
KB makes an instance a first-class citizen for free (a distinct symbol + code), inheritance is the composer's
already-clean abstention retried against the `isa`-parent, and the WM anaphora machinery (Phase-4 GO) is already an
object file — so this is **routing + labeling over validated pieces, not a new mechanism.** Phase 11 already named
this as the next nuance, so it is on the critical path. A cheap-first numpy de-risk (`_kindinst_isa_inheritance_
derisk.py`) with the isa-lesion / instance-fact-lesion / distinctness / moat anti-cheats is the right first step,
and I expect it to GO.

**What needs the deeper machinery (deliberately deferred):**
- **A perceived, consolidated episodic instance** — "the specific dog I saw on my walk", bound into its
  spatiotemporal context and consolidated — is the **engram-tag / hippocampal** path (#3, D.14/D.01). It is the
  right mechanism for the embodied capstone (composes with the Tier-3 live-and-remember loop), heavier than the
  conversational layer needs, and best done as the follow-on once #1/#2 land.
- **Multiple co-present instances of the same kind** in one turn re-uses the already-mapped biased-competition WTA
  boundary (2026-06-17 NEGATIVE → the specified WTA fix, `enable_biased_competition` GO 2026-06-19) — a drop-in when
  prioritized, not a new wall.
- **Code-level instance↔kind similarity / instance generalization** (an instance code near its kind's) is the
  optional #4 typed-code polish, and full similarity-based generalization is the separate PPMI/dendritic frontier
  (CLAUDE.md; NOT reopened here — explicit `isa` inheritance is sufficient for the conversational goal).

**Honest boundary in #1 itself:** inheritance is via an EXPLICIT symbolic `isa` link (a stored fact), not
automatic code similarity — which is exactly how a discourse model / DRT works, and is the correct, sufficient
mechanism for kind-vs-instance in conversation. It is NOT semantic generalization across similar instances (that is
the deferred frontier). The distinction is worth stating so the win is not overclaimed: this gives the substrate a
*type/token* distinction and inheritance, not similarity-based reasoning over instances.

---

### Reproduce / next
1. Write + run the cheap-first `_kindinst_isa_inheritance_derisk.py` (numpy, ≥3 seeds) — #1 + #2, all anti-cheats.
2. If GO, wire the definite/generic router + instance-first-kind-fallback into `MultiTurnAgent` / `FluidChat`
   (opt-in, default-off, byte-identical when off), and point the WM object-file at instance tokens.
3. Follow-on (embodied): the engram-tag episodic instance (#3), composing with the Tier-3 live-and-remember loop —
   "the specific dog I saw".

Reuse-by-import throughout; NO `sim/` edit anticipated for #1/#2 (agent-layer only). Sources: catalog D.01, D.02,
D.14/D.63, D.21, D.22 (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`); Kandel 6e Ch 52
pp 1296–1302, Ch 54 pp 1357–1359; Kahneman-Treisman 1984 / Kahneman-Treisman-Gibbs 1992 (object files, type-token);
Kamp DRT (discourse referents); Collins-Quillian 1969 / Rogers-McClelland 2004 (inheritance); McClelland-McNaughton-
O'Reilly 1995 (CLS). Project findings: Phase 11 grow-grounded-knowledge, Phase 8 experience-connection, Phase 4
multi-turn, 2026-06-17 multiturn-anaphora GO, 2026-06-17 multireferent-disambiguation NEGATIVE.
