# Embedded-clause PARSING from a flat token stream — the #3 conversational lever (deep-research + catalog scoping, 2026-06-19)

> **READ-ONLY deep-research + catalog scoping. No code edited, no jobs run; this doc is the only write.** Produced
> per the standing "deep research + catalog review FIRST at a new direction" directive (CLAUDE.md;
> `feedback_deep_research_at_roadblocks`). Every load-bearing project fact below was re-verified against the repo
> by file-read (cited inline). The controller should trust-but-verify the **[VERIFY]**-flagged claims, then push +
> present.
>
> **The decision this scopes:** conversational #1 (attributed + multi-frame consolidation) and #2 (multi-referent
> biased competition) both landed GO. The ranked **#3 lever** (from `2026-06-19-conversational-scaling-next-lever-scoping.md`)
> is **embedded-clause PARSING from a flat token stream**: comprehending "the dog that chased the cat ran" —
> SEGMENTING the embedded relative clause ("that chased the cat") from the matrix clause ("the dog … ran") and
> assigning roles in BOTH — turning the flat word stream into the nested structure the composer already binds.

---

## 0. The one-paragraph answer

**The crux of #3 is a parser that detects + segments an embedded clause from a flat token stream; it is NOT a
binder.** The COMPOSER already DECODES nested structure (a fact whose patient is a `Clause` → a 2-level
register→register unbind, the intermediate composite re-kicked as a clean unit phasor —
`OneBrainComposer._decode_clause`, `RFPhasorComposer._render`, "recursive embedded CLAUSES" in CLAUDE.md). What is
NOT built is the PARSER/comprehension side: **today every `Clause(...)` operand is constructed by HOST code in a
runner** (verified: `learned_nesting_demo.py`, `nested_composition_agent.py`, `unified_agent_benchmark.py` all
hand the composer pre-built `Clause` namedtuples; the only token-level clause "parser",
`phasor_chat.py:_parse_patient`, requires the embedded clause to ALREADY be parenthesized/segmented in the input
and uses a host POS lookup). **The concrete mechanism:** a **relative-pronoun cue word ("that"/"which"/"who")**
is the segmentation trigger — it OPENS a constituent boundary; the words after it up to the matrix-clause verb are
the embedded clause; both clauses are then role-assigned by the SAME conjunctive position-code parser the project
already validated (`AttributedBridgeParser`: `(from-START × from-END × voice) → role` in spikes, GO 6/6), each
operating over its OWN clause's local positions. **The WM-hold** (what holds the suspended matrix subject "the
dog" while the embedded clause is parsed, then re-binds it to the matrix verb "ran") is the project's existing
**dlPFC NMDA working-memory latch** (`cortex_ctx ↔ dlpfc_wm` self-attractor, weight 30 = genuinely NMDA-dependent;
`nav_conv_merged_bridge.py`) — the neural realization of Hagoort's MUC "Memory" + "Control" components, with the
**human ~2-level center-embedding depth limit reported as the honest biological bound, not a flaw**. This is a
**reuse-heavy build** (the position-code parser, the dlPFC latch, the gamma-slot `OrderedPositionWM`, the
composer's clause decode all exist) — the new science is the **segmentation-cue → open/close constituent control
discipline** and a **second parse pass over the segmented spans**. It is correctly **last of the three** (highest
ceiling, highest risk, most likely to surface an honest substrate boundary), and an honest NEGATIVE here — the
G.12 center-embedding boundary — is itself the deliverable.

---

## 1. THE CRITICAL DISTINCTION (verified — do not conflate the binder with the parser)

| Side | Status | Evidence (file-read this pass) |
|---|---|---|
| **BIND/STORE/DECODE nested structure** (composer) | ✅ **DONE** — recursive embedded clauses GO | `OneBrainComposer._decode_clause` (lines 448–492: a TWO-hop unbind; the intermediate clause composite is READ OUT and RE-KICKED as a clean unit phasor before hop 2). `RFPhasorComposer._render`/`_filler_phases` (lines 143–168: `_is_clause(filler)` → recursively `_encode` the clause). CLAUDE.md "recursive embedded CLAUSES (a fact whose patient is an SVO clause → a 2-level register→register unbind)". |
| **PARSE/SEGMENT nested structure from a flat token stream** (parser) | ❌ **NOT BUILT** — the #3 target | Every `Clause(...)` is **host-constructed in the runner** before the composer sees it: `learned_nesting_demo.py:38` `("dog","eat",Clause("cat","chase","river"))`; `nested_composition_agent.py:323`; `unified_agent_benchmark.py:81-83`. The lone token-level clause path `phasor_chat.py:_parse_patient` (lines 44–57) keys on `kinds == ["n","v","n"]` — i.e. it needs the embedded clause ALREADY isolated as the patient span AND a host POS lookup (`_kind`); it does NOT segment "the dog that chased the cat ran" from an unbracketed stream. |

**So #3 = the parser, not the binder.** The composer is the consumer that already works; the missing piece is the
front end that produces the `Clause` operand from raw words. This boundary is the single most important framing in
this doc.

---

## 2. THE MECHANISM MAP (the crux: the segmentation cue + the WM-hold)

### 2.1 What a single-level embedded relative clause looks like, and the two things the parser must do

Target sentence class (subject-extracted relative clause, the canonical depth-1 case):

```
   "the  dog  that  chased  the  cat  ran"
    └─── matrix subject ───┘                     (suspended)
              └──── embedded clause ────┘         ("the dog" chased the cat)
    └────────── matrix clause ──────────────┘     ("the dog" … ran)
```

The parser must (a) **SEGMENT**: find where the embedded clause begins and ends, splitting the flat stream into
matrix-clause tokens and embedded-clause tokens; and (b) **role-assign in BOTH**: produce `Clause(agent="dog",
action="chased", patient="cat")` for the embedded clause AND the matrix fact `(agent="dog", action="ran",
patient=…)`, sharing the head noun "dog" as the agent of both — handing the composer
`Clause("dog","ran", … )` with the embedded `Clause("dog","chased","cat")` as a bound constituent (or, for an
object-relative "the cat that the dog chased ran", the head as the embedded PATIENT). The composer then binds +
stores it, and answers who/what over both clauses.

### 2.2 The SEGMENTATION cue — a relative-pronoun trigger that OPENS a constituent (the brain-based realization)

**The cue is the relative pronoun ("that"/"which"/"who"), a closed-class function word.** This is both the
linguistically-correct trigger and the biologically-grounded one:

- **Function words mark constituent boundaries.** Closed-class items (relativizers, complementizers) are precisely
  the cues a left-corner / shift-reduce parser uses to open an embedded constituent. The relativizer "that"
  signals "a clause modifying the immediately preceding head noun begins here."
- **It maps onto a discrete neural control signal.** In the project's substrate the cue word fires a dedicated
  conjunction/marker unit (exactly as the parser's position conjunctions already fire) whose role is not "assign a
  thematic role" but "**toggle the parser's PUSH control gate**" — open a new clause frame. This is the neural
  analogue of the **left-corner parser's stack-push** (confirmed grounding: Localizing Syntactic Composition with
  Left-Corner RNNGs, *Neurobiology of Language* 2024 — "a storage component in human parsers, consumed when
  processing nested structures, similar to the stack of left-corner parsers", localized to LIFG/Broca).
- **The CLOSE cue is the matrix verb.** After the relativizer opens the clause, the embedded clause runs to its
  own verb-object ("chased the cat"); the NEXT verb encountered ("ran") with no remaining unconsumed subject is
  the matrix verb — it CLOSES the embedded clause (pop) and re-attaches to the suspended matrix subject. A
  **verb-count > 1 signal** (the project already has the `FrameSelector`'s verb-position detection,
  `frame_parser.py`) is the secondary structural cue that an embedding is present at all, and which span is matrix
  vs embedded.

**The honest scope of the cue front end:** detecting "is this token a relativizer / a verb" is a **host lexical
lookup** against the known function-word + verb sets — the same legitimate morphology/POS front end the project
already uses (`FrameParser._verb_position` does exactly this; `phasor_chat._kind` POS-classifies against known
noun/verb/adj sets). This is BRAIN-BASED-compliant: lexical access (which closed-class category a word is) is the
environment/lexicon front end; everything downstream — opening the constituent, holding the matrix subject,
assigning roles — is neural. (A fully-neural relativizer detector is a bounded follow-on, exactly as the
fully-neural verb detector is a follow-on for the frame parser.)

### 2.3 The WM-HOLD — what suspends the matrix clause during the embedded parse

When the parser opens the embedded clause at "that", the matrix subject "the dog" and the still-open matrix
predicate must be **held** until the embedded clause closes and the matrix verb arrives. This is the classic
**filler-gap / suspended-constituent** memory load — and it is exactly **Hagoort's MUC "Memory" component**
(verified literature: MUC = **M**emory [temporal-cortex lexical store] + **U**nification [Broca's binding] +
**C**ontrol [LIFG routing]; "syntactic unification is more extended in time for non-adjacent dependencies, which
requires on-line processing memory" — Hagoort, *On Broca, brain, and binding*, TiCS 2005; the MUC chapter).

**The project already has the mechanism — two complementary candidates, both validated:**

1. **The dlPFC NMDA working-memory latch (primary candidate).** `cortex_ctx ↔ dlpfc_wm` self-attractor at weight
   **30** — documented as "the genuinely NMDA-dependent attractor weight" (`nav_conv_merged_bridge.py:45`,
   `:612`, `:1037` `enable_nmda=True`), which **survives dt=1.0** (CLAUDE.md "one-bridge unification step3"; the
   NMDA slow conductance holds the assembly active across the delay). This is the direct analogue of a PFC
   delay-period attractor holding the suspended matrix constituent. It is already co-resident on the merged
   conversational bridge.
2. **The gamma-slot ordered WM (`OrderedPositionWM`, secondary / complementary).** N=7 Lisman-Idiart gamma slots on
   the RF phasor substrate (`ordered_position_wm.py`), where each held item lives in a disjoint position-bound
   subspace addressed by SLOT. A parser stack of suspended constituents is naturally a **small ordered store**:
   PUSH = bind the constituent to the next slot; POP = read the top slot. This is the cleaner substrate for the
   *stack* abstraction (the left-corner parser's storage component), while the dlPFC latch is the cleaner
   substrate for *sustaining* a single suspended subject across a delay. The cheapest-first probe (§4) should test
   the **ordered-WM-as-stack** realization first (it already has a position-binding API + a familiarity moat), and
   fall back to / combine with the dlPFC latch if a single sustained register suffices for depth-1.

**Why the WM-hold is the honest depth limiter (the biological bound, §5).** Verified literature: the human
center-embedding limit is **~2 levels**, and — decisively for this project — the bottleneck is **NOT storage
overload but "impoverished discrimination combined with poor support for serial order"** (similarity-based
interference between the held constituents; Chomsky-Miller; "Working Memory Constraints on Multiple
Center-Embedding"). That is *exactly* the failure mode of the project's spiking WM: bundle cross-talk +
finite-phase-resolution interference between bound constituents (the same reason `period=48` breaks the composer's
existing clause decode, `rf_phasor_composer.py` note — recursive nesting "needs more phase resolution than flat
queries"). **The substrate's syntactic depth limit and the human one have the same root cause** — making a depth-2
NEGATIVE a *biology-faithful* result, not a defect.

### 2.4 The two-pass control discipline (the genuinely new piece)

Concretely, the parser becomes a small **shift / push / pop** controller over the existing position-code role
reader:

1. **Scan** the token stream; the host lexicon tags each token's closed-class category (relativizer? verb? noun?
   adj?) — the legitimate front end.
2. **On a relativizer** → fire the PUSH marker unit → open an embedded clause frame; **hold** the matrix subject +
   open matrix predicate in the WM-hold (dlPFC latch / ordered-WM slot).
3. **Parse the embedded span** ("chased the cat") with the EXISTING `AttributedBridgeParser` over the embedded
   span's LOCAL positions (the head noun "dog" is injected as the embedded agent for a subject-relative, or
   embedded patient for an object-relative — determined by whether a subject is present inside the embedded span).
4. **On the matrix verb** → POP: close the embedded clause into a `Clause`, retrieve the suspended matrix subject,
   parse the matrix predicate, emit `Clause(matrix_subj, matrix_verb, …)` with the embedded `Clause` as a
   constituent.
5. **Hand the nested structure to the composer** → it binds/stores/decodes it (already validated).

Steps 1, 3, 5 are **100% reuse**. Steps 2, 4 (the push/pop control + the WM-hold of the suspended span) are the
new wiring. This is the **left-corner parsing strategy** (verified as "the psychologically plausible parsing
strategy", left-corner RNNGs best-localize syntactic composition to LIFG) realized as a neural shift-reduce
controller over a bounded store — and it has a **direct point-neuron precedent**: Mitropolsky & Papadimitriou, *A
Biologically Plausible Parser* (TACL 2021) + *Center-Embedding and Constituency in the Brain* (NALOMA 2022) — "a
parser of English effectuated by biologically plausible neurons and synapses… handles recursion, embedding".

---

## 3. REUSE-vs-NEW (prefer reuse-by-import / additive default-OFF; flag any `sim/` edit)

| Component | Reuse or new? | What / where |
|---|---|---|
| **Per-clause role assignment** | **REUSE** | `AttributedBridgeParser` (`attributed_parser.py`) — `(from-START × from-END × voice) → role` in spikes; GO 6/6. The from-END factor (head-noun adjacency cue) is exactly what disambiguates roles within a clause span. Run it once per segmented span. The flat `BridgeParser` (`brain_conversational_agent.py`) is the SVO fallback. |
| **"Is this an embedding / which span is the verb"** | **REUSE** | `FrameSelector` / `FrameParser._verb_position` (`frame_parser.py`) — verb-position detection (already host-lexical). A verb-count > 1 = embedding present. |
| **WM-hold of the suspended matrix subject** | **REUSE** | dlPFC NMDA latch (`cortex_ctx ↔ dlpfc_wm`, weight 30, `enable_nmda=True`; `nav_conv_merged_bridge.py`) AND/OR the gamma-slot `OrderedPositionWM` (`ordered_position_wm.py`) used as the parser stack (PUSH=bind-to-slot, POP=read-slot, with its familiarity moat). |
| **Nested bind / store / decode** | **REUSE** | `OneBrainComposer._decode_clause` + `RFPhasorComposer._render`/`_filler_phases`; the composer already consumes a `Clause` operand and 2-level-unbinds it. |
| **The no-confab moat** | **REUSE** | `OrderedPositionWM`'s calibrated familiarity gate (Bogacz-Brown) + the composer's cue-match abstention; re-assert unchanged. |
| **Relativizer/PUSH marker unit + the push/pop control gate** | **NEW (small, additive)** | A marker conjunction unit that fires on the relativizer and toggles a transmission/plasticity-style control gate that opens an embedded clause frame; a POP on the matrix verb. Prefer realizing it with the EXISTING transmission-gate + dlPFC-Control machinery (the project's `transmission_gate` + dlPFC routing), so it is reuse-by-composition, not new `sim/`. |
| **The two-pass `parse_nested(tokens, verbs, relativizers)` orchestrator** | **NEW (runner-level)** | A new method/runner (`embedded_clause_parser.py` or a method on `BridgeParser`/`FrameParser`) that runs the scan → push → per-span parse → pop loop and returns the nested `Clause` structure. Additive, default-OFF (`enable_embedded_clauses=True`), byte-identical when unused — mirroring the existing `enable_attributed` / `enable_multiframe` flag pattern. |

**`sim/` edit:** **none anticipated.** Everything is reuse-by-import + a runner-level controller + (if needed for
the PUSH gate) the EXISTING transmission-gate API. If a fully-neural relativizer detector or a new gate type turns
out to require a protected edit, flag it for byte-level diff review (per `feedback_brain_based_only_standard`) —
but the cheap-first path deliberately avoids it (host lexical cue + reused gates).

---

## 4. THE CHEAPEST-FIRST DE-RISK (the smallest test that decides it)

**The question it answers:** *does the parser SEGMENT a single-level embedded relative clause from a flat
(unbracketed) token stream and assign correct roles in BOTH the matrix and embedded clauses, which the composer
then binds + answers?* — **CPU/numpy first**, then the decisive GPU multi-seed.

**Config (cheap-first, CPU/numpy smoke → GPU multi-seed):**
- A small held-out sentence set of **subject-extracted depth-1 relatives** in the validated probe vocab, e.g.
  `"dog that chase cat run"`, `"cat that see bird eat"`, … (function words "the" dropped or treated as ignorable
  closed-class, as the project's probes already do; the relativizer "that" is the segmentation cue). Include a
  few **object-extracted** relatives (`"cat that dog chase run"`) to test the head=embedded-PATIENT path.
- Build the two-pass parser from the reused pieces (`AttributedBridgeParser` for per-span roles +
  `FrameParser._verb_position` for the verb cue + the WM-hold). Hand the resulting nested `Clause` to
  `OneBrainComposer` / `RFPhasorComposer`; then query.
- The **decisive read**: after `hear`-ing the nested sentence, the agent must answer BOTH (a) the embedded-clause
  who/what ("what did the dog chase?" → "cat") AND (b) the matrix who/what ("what did the dog do?" → "run", or the
  matrix patient if present) — i.e. **both clauses' roles resolve through the composer's existing decode.**

**Pre-registered GATE (FROZEN before data; the decisive matrix is multi-seed/GPU per `feedback_6seed_validation`):**
- **GO:** on a **held-out** sentence set — the **embedded-clause roles resolve correctly AND the matrix-clause
  roles resolve correctly** (both ≥ the composer's validated flat accuracy, e.g. ≥ 0.90 each) on **≥5/6 seeds**;
  the **no-confab moat is intact** (a query about a never-stored embedded fact → `None`, 0 false-accepts); AND
  the **flat-SVO + flat-attributed paths are un-regressed**. ⇒ promote `enable_embedded_clauses` toward
  default-on; extend to depth-2 as the boundary probe.
- **BOUNDARY:** segmentation works but roles in ONE clause degrade (e.g. the head-noun sharing mis-routes
  subject- vs object-relative; or the matrix verb is occasionally mis-segmented as embedded) — a real partial
  result localizing the **head-attachment / pop-timing** sub-problem, not a mechanism failure.
- **NEGATIVE:** the parser cannot reliably segment the embedded span from the flat stream (the verb-cue +
  relativizer control does not robustly find the boundary), OR the WM-hold loses the suspended matrix subject
  across the embedded parse (the held register decays / is overwritten) — an honest finding that the depth-1
  embedded parse needs a stronger control or hold mechanism than the reused pieces provide.

**Anti-cheat controls (mandatory — a "success" without ALL of these is an artifact):**
1. **No-segmentation baseline FAILS.** Feed the SAME flat sentences to the existing flat `BridgeParser` (no
   push/pop) — it MUST produce a wrong/degenerate parse (e.g. it reads "dog that chase cat run" as a single
   mangled 5-word fact). If the flat parser "passes" by luck, the task is not measuring segmentation. This is the
   load-bearing control (the analogue of the multi-frame "no-frame-selection baseline fails").
2. **Held-out generalization + leakage assertion.** Test sentences use noun/verb fillers in **novel combinations
   disjoint from any training/teacher frames** (assert disjointness in code). The parser's role assignment is by
   POSITION-conjunction (productive), so a pass on held-out combos is real; a pass only on memorized specific
   sentences is the failure to guard against. **Compare against a memorization floor** (a permuted control, #3).
3. **Permuted/scrambled control.** Scramble the token order within a clause (break the position structure) — roles
   must NOT resolve (the parser reads POSITION, so a scramble must degrade it). AND a **permuted head-attachment
   control** (attach the embedded clause to the WRONG noun) must score at floor — asserting the parse is
   structural, not a fixed template. If a scrambled sentence still "parses", it is memorizing.
4. **The path is NEURAL.** Per-clause role read-out is the spiking `AttributedBridgeParser` firing (not a host
   adjacency rule); the nested decode is the spiking resonate-and-fire 2-level unbind; the WM-hold is the spiking
   dlPFC latch / RF position-bind. Host is limited to the environment (the token string + closed-class lexical
   tagging) + body (emit). Assert the production path uses the neural components (`feedback_brain_based_only_standard`).
5. **The no-confab moat asserted intact THROUGHOUT** — abstention + 0 false-accepts on unstored embedded/matrix
   cues, before and after the wiring. A moat regression voids the result even if nested recall improves (trade the
   moat only deliberately, never as a wiring side effect; `feedback_moat_not_hard_lossy_memory_ok`).
6. **≥6 seeds (fractional ≥5/6 bar) on the decisive GPU run; CPU smoke is the cheap-first gate; frozen bars, no
   config-cranking** — reuse the attributed/multi-frame de-risks' own anti-cheat harnesses verbatim where possible.

**Expected wall-clock:** the reused parsers each validated in minutes (CPU) / a GPU matrix run; the new control +
two-pass orchestration is wiring — **hours, not weeks. No cloud** (`SIM_BACKEND=cupy` for the decisive
multi-seed; numpy for the CPU smoke).

---

## 5. HONEST RISK + the clear cheap-first GO vs NEGATIVE

**The biggest way #3 could mislead — a toy that only handles ONE fixed clause template (= memorization, not
parsing).** A parser that "works" on `"X that Y Z W"` by hard-wiring "split after token 2" is NOT a parser — it is
a memorized template and would silently fail on any structural variant (object-relatives, a 2-adjective head, a
different relativizer). **This is precisely what anti-cheats #1–#3 (no-segmentation baseline fails + held-out
combos + permuted/head-attachment scramble) exist to catch.** The cheap-first probe MUST include
object-extracted relatives and held-out fillers so a template-memorizer cannot pass. If it only passes on the
exact training template → NEGATIVE (template, not parser).

**The second risk — center-embedding depth the point-neuron WM cannot hold (the honest biological bound, NOT a
defect to brute-force).** Verified: the human center-embedding limit is **~2 levels**, root-caused to
**similarity-based interference + poor serial-order support, not storage** — the SAME failure mode as the
project's spiking WM (bundle cross-talk; the composer's existing depth-2 *decode* already costs phase resolution
/ a seed, `rf_phasor_composer.py` period note + CLAUDE.md "depth-2 nesting already costs a seed in the decode").
So **the expected outcome is: depth-1 GO, depth-2 BOUNDARY/NEGATIVE — and the depth-2 boundary is the G.12
deliverable** (a biology-faithful match to the human ~2-level limit, "the girl that the boy is chasing is tall"
comprehension failure, catalog G.12 behavioral validation). **Do NOT escalate a depth-2 NEGATIVE into a config
search** — report it as the substrate's syntactic depth limit (which coincides with the human one). Per the North
star (`project_actual_goal_artificial_life_brain_analogue`): an honest negative under strict biology IS the
scientific deliverable.

**The third risk — head-attachment ambiguity (the BOUNDARY case).** Sharing the head noun "dog" as the agent of
BOTH clauses (subject-relative) vs the patient of the embedded clause (object-relative) is the genuinely tricky
role-routing. It is *localizable* (the subject-vs-object-relative distinction is whether the embedded span
contains its own subject) and the anti-cheat (#3 permuted head-attachment) guards it — a BOUNDARY here names the
exact sub-problem (pop-timing / head re-attachment), not a mechanism failure.

**The clear three-state outcome (pre-registered, §4):** **GO** (both clauses' roles resolve on held-out
sentences, moat intact, flat un-regressed, ≥5/6) → promote + probe depth-2 as the boundary; **BOUNDARY**
(head-attachment or pop-timing degrades one clause) → localize that sub-problem; **NEGATIVE** (no robust
segmentation, or the WM-hold loses the suspended subject — most likely at depth-2) → the honest G.12
center-embedding boundary, the deliverable, re-scoping to "the substrate's syntactic depth limit." **Stop
criterion:** report the three-state outcome after the CPU probe → GPU matrix; a clean depth-boundary IS the
answer.

---

## 6. SUMMARY (the return)

- **The mechanism map (the crux):** the **segmentation cue is the relative pronoun ("that"/"which"/"who")** — a
  closed-class function word that fires a PUSH marker unit to OPEN an embedded constituent (the matrix verb later
  CLOSES it; a verb-count > 1 signal flags the embedding) — and the **WM-hold of the suspended matrix subject is
  the existing dlPFC NMDA working-memory latch** (`cortex_ctx ↔ dlpfc_wm`, weight 30, NMDA-dependent, survives
  dt=1.0) and/or the gamma-slot `OrderedPositionWM` used as the parser stack (PUSH = bind-to-slot, POP =
  read-slot). Both clauses are then role-assigned by the SAME validated conjunctive position-code parser
  (`AttributedBridgeParser`) over each clause's local positions, and the nested `Clause` is handed to the
  composer, which already binds/decodes it. This is the left-corner / shift-reduce strategy (Hagoort MUC;
  Mitropolsky-Papadimitriou Assembly-Calculus point-neuron precedent).
- **Reuse-vs-new:** **REUSE** the per-clause role parser (`AttributedBridgeParser`), the verb-position cue
  (`FrameParser`), the dlPFC NMDA latch + gamma-slot `OrderedPositionWM` (WM-hold), the composer's nested
  decode (`OneBrainComposer._decode_clause`), and the familiarity moat. **NEW** (small, additive, default-OFF):
  the relativizer/PUSH marker + the push/pop control gate (realized via the EXISTING transmission-gate + dlPFC
  Control), and a runner-level two-pass `parse_nested` orchestrator. **NO `sim/` edit anticipated** (reuse-by-
  import + reused gates); flag any protected edit for byte review.
- **The cheapest-first de-risk:** build the two-pass parser from the reused pieces; on a **held-out** set of
  depth-1 relatives (subject- AND object-extracted) in the probe vocab, `hear` the flat nested sentence and
  require BOTH the embedded-clause who/what AND the matrix-clause who/what to resolve through the composer.
  **GO** = embedded roles ≥0.90 ∧ matrix roles ≥0.90 ∧ flat un-regressed ∧ moat intact (0 false-accepts), ≥5/6
  seeds. **Anti-cheats:** no-segmentation baseline FAILS + held-out-combo leakage-asserted + permuted/scrambled +
  permuted-head-attachment controls + the path is NEURAL + moat-asserted-throughout + ≥6 seeds + frozen bars.
  CPU/numpy smoke → GPU multi-seed; hours, no cloud.
- **The honest biological bound:** the human **center-embedding depth limit (~2 levels)**, root-caused to
  similarity-interference + poor serial-order (the SAME failure mode as the project's spiking WM) — so the
  expected result is **depth-1 GO, depth-2 BOUNDARY/NEGATIVE**, and the depth-2 boundary is the **G.12
  deliverable** (a biology-faithful match to the human limit), NOT a defect to brute-force.

---

### Catalog entries cited
**G.10** (language as a hierarchical symbolic system), **G.11** (dual-stream language: dorsal sensorimotor +
ventral semantic — where the parser sits), **G.12** (Broca's grammatical processing — the center-embedding "the
girl that the boy is chasing is tall" comprehension-failure dissociation; the **behavioral target + the honest
depth bound**), **G.13** (Wernicke auditory→semantic), **G.08** (PFC working memory / executive control — the MUC
"Memory"+"Control" substrate, the WM-hold), **H.19** (premotor competitive queuing / serial order — the
serial-order support that center-embedding stresses), **N.15** (theta-gamma multiplexed cell-assembly buffer,
Lisman-Idiart 1995 — the gamma-slot ordered-WM / parser-stack substrate). Catalog:
`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.

### Papers cited (links)
- Hagoort — **On Broca, brain, and binding: a new framework**, *Trends in Cognitive Sciences* 2005
  (http://faculty.washington.edu/losterho/hagoort_trends.pdf) + **MUC (Memory, Unification, Control) and beyond**,
  2013 (https://pubmed.ncbi.nlm.nih.gov/23874313/ ; chapter PDF
  https://pure.mpg.de/rest/items/item_2193289/component/file_2193288/content) — the M+U+C decomposition; Broca =
  Unification; non-adjacent dependencies require on-line WM (the WM-hold).
- Mitropolsky, Collins, Papadimitriou — **A Biologically Plausible Parser**, *TACL* 2021
  (https://arxiv.org/abs/2108.02189) + Mitropolsky, Ejaz, Shi, Papadimitriou, Yannakakis —
  **Center-Embedding and Constituency in the Brain**, NALOMA 2022 (https://arxiv.org/pdf/2206.13217) — the
  point-neuron / Assembly-Calculus precedent that a parser handling recursion + embedding is realizable in
  neurons + synapses.
- Sugimoto, Yoshida, Oseki et al. — **Localizing Syntactic Composition with Left-Corner RNNGs**,
  *Neurobiology of Language* 2024 (https://direct.mit.edu/nol/article/5/1/201/117096) — left-corner = the
  psychologically plausible strategy; a stack/storage component consumed by nesting, localized to LIFG (the
  push/pop discipline).
- **Working Memory Constraints on Multiple Center-Embedding** (researchgate 345017757) + Karlsson —
  **Constraints on multiple center-embedding of clauses** (researchgate 231982450) — the **~2-level human depth
  limit**; the bottleneck is impoverished discrimination + poor serial order, NOT storage (the honest biological
  bound + its mechanistic match to the spiking-WM failure mode).
- Lisman & Idiart 1995, *Science* — theta-gamma multiplexed STM buffer (the gamma-slot ordered-WM / stack
  substrate, `OrderedPositionWM`).

### Project files / findings reviewed (this pass, file-cited)
- **The #3 framing source:** `2026-06-19-conversational-scaling-next-lever-scoping.md` (#3 = embedded-clause
  PARSING from a flat stream, "the hard half of syntax").
- **The binder side (already DONE — verified, do not conflate):** `one_brain_composer.py:448-492`
  (`_decode_clause`, the 2-level register→register unbind), `rf_phasor_composer.py:143-168`
  (`_filler_phases`/`_render`, recursive `_is_clause` decode), `core_sim_composition.py:35`
  (`Clause = namedtuple(...)`).
- **The parser machinery to REUSE:** `attributed_parser.py` (the `(from-START × from-END × voice) → role`
  conjunctive parser, GO 6/6 — the per-span role reader + the from-END head-adjacency cue), `frame_parser.py`
  (`FrameSelector`/`FrameParser._verb_position` — the verb-position cue), `brain_conversational_agent.py:28-143`
  (the flat `BridgeParser` — the SVO fallback + the `enable_attributed`/`enable_multiframe` flag pattern to
  mirror).
- **The WM-hold candidates (verified present + validated):** `nav_conv_merged_bridge.py:45,612,1037` (the dlPFC
  `cortex_ctx ↔ dlpfc_wm` NMDA self-attractor, weight 30, `enable_nmda=True`), `ordered_position_wm.py`
  (`OrderedPositionWM`, N=7 gamma slots, position-binding + calibrated familiarity moat).
- **The evidence clauses are host-constructed today (the load-bearing gap):** `learned_nesting_demo.py:38`,
  `nested_composition_agent.py:323`, `unified_agent_benchmark.py:81-83` (all pass pre-built `Clause` namedtuples);
  `phasor_chat.py:44-57` (`_parse_patient` — the only token-level clause path, requires a pre-segmented span +
  host POS lookup).
- `CLAUDE.md` (conversational sections, "recursive embedded CLAUSES", "one-bridge unification step3"). Catalog:
  `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (clusters G, H, N).
