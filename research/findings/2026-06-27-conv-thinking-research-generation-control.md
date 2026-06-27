# Deep research — LANGUAGE PRODUCTION + WORKING MEMORY + COGNITIVE CONTROL (2026-06-27)

**Read-only research analyst doc. No `sim/` edit, no code.** Produced per the standing
"deep research + catalog review FIRST at roadblocks / new directions" directive. The
controller should trust-but-verify the load-bearing claims (flagged inline) before any build.

**Scope (one cluster):** the *output + executive* half of conversation/thinking —
**(I) language production** (Levelt: conceptualization → lemma selection → grammatical
encoding/argument-structure → morphophonological encoding → articulation; serial order;
lexical retrieval & tip-of-tongue; fluency/disfluency & self-monitoring/self-repair);
**(II) working memory** (capacity ~4±1; persistent-activity vs activity-silent WM; holding
discourse referents); **(III) cognitive control** (the dlPFC central executive; attention,
inhibition, conflict monitoring; basal-ganglia gating of WM updates).

**Companion docs:** comprehension/parsing + binding are a *separate* cluster and are NOT
re-opened here (see CLAUDE.md's conversational arc). This doc treats the SELECTED content as
given and asks how the brain (and our sim) *formulates and utters* it, and how WM/control
schedule that.

**Sources used.** (1) Catalog `sim-catalog/references/feature-catalog.md` — Cluster G
(working memory / PFC / language: G.06–G.20), N.15 (theta-gamma WM buffer), H.19/G.07
(pre-SMA sequences); the catalog cites Kandel page anchors directly. (2) Kandel 6e
`textbooks/kandel-pns-6e/full-book.txt` (Ch 34 motor/PFC, Ch 52 cognition, Ch 55 language,
Ch 56 decision) — anchors verified present. (3) **Glossary `references/glossary.md` is
ABSENT** on disk (only `feature-catalog.md` + `biology-buildout-roadmap.md` + `textbooks/`);
noted, and substituted with literature + the catalog. (4) Literature via WebSearch (Levelt
1999; Bock & Levelt 1994; Cowan 2001; Stokes/Mongillo; O'Reilly & Frank 2006; Wang 2002;
Nozari/Botvinick; agrammatism) — cited at the end.

---

## 0. EXECUTIVE SUMMARY (read this first)

The owner-identified gaps cluster into **three distinct problems with three distinct biological
fixes**, in priority order:

1. **No argument structure / no message-formulation pipeline.** The sim emits bare SVO
   ("the boy goes park"). Biology's missing piece is the **verb lemma's subcategorization
   frame**: selecting the verb lemma `go` retrieves its argument structure (a GOAL argument
   realized as a PP "to the \_\_\_") and its closed-class scaffold (determiners, tense). This is
   the **Levelt functional → positional split** (Bock & Levelt 1994; Garrett). Our serial-order
   CQ generator already orders a *given* slot set; what's missing is the step that *builds the
   slot set* from a verb frame. **Fix: a small inventory of learned verb-frame slot templates,
   selected by the verb lemma, that expand the bare triple into ordered (role + closed-class)
   slots before CQ orders them.** Cheap, reuses `FrameCQ` + the A→W read-out. **Highest leverage.**

2. **The dlPFC-WM bridge BALLOONS with vocab/fact count.** Root cause located precisely:
   `content_selection_spiking.py:307,359` sizes the WM region `n = max(600, 60 *
   len(self._vocab))` — it installs one ~50-neuron Hopfield attractor **per vocabulary item**,
   so the bridge grows linearly with the *number of storable concepts*. This is the
   architectural *inverse* of biology: working memory is a **fixed-capacity buffer of ~4±1
   *active* items** (Cowan 2001), while the *storable* codes live in cortex/LTM. **Fix: decouple
   capacity from inventory — a FIXED-size WM (a bump-attractor / slot buffer holding ~4±1 active
   pointers) whose slots POINT at cortical/LTM concept codes, instead of a per-concept attractor
   bank.** Eliminates the balloon by construction. **Second-highest leverage (it's a freeze-at-
   scale bug, not just a nicety).**

3. **No conceptualization / no monitoring-and-repair, and persistent-activity WM is the only
   WM type.** The sim has Wang-2002 persistent activity but no **activity-silent** (synaptic)
   WM for the background discourse referents, and no **conflict-monitoring self-monitor** to
   catch/repair a garbled utterance. **Fix: (a) an activity-silent short-term-synaptic store for
   the *unattended* discourse set (Stokes/Mongillo) so only the *focused* referent needs
   persistent firing — this also relieves the balloon; (b) a conflict-monitor read-out
   (Nozari/Botvinick ACC) over the production competition that triggers abstain/re-select.**
   Lower-leverage, build after 1–2.

**The single biggest realization:** the WM balloon and "no argument structure" are *both*
symptoms of the same modeling error — **conflating storage (unbounded, in cortex/LTM) with the
active buffer (fixed ~4±1, in PFC)**. Biology keeps them separate; the sim currently fuses
them (one attractor per concept = storage living *inside* the buffer). Fixing that split fixes
the balloon AND gives a clean home for "the frame + its few active arguments" (which is exactly
~4±1 slots).

---

## I. LANGUAGE PRODUCTION

### The Levelt blueprint (the organizing frame for everything below)

**Human capability + the canonical staged architecture.** Levelt, Roelofs & Meyer (1999)
("A theory of lexical access in speech production", *BBS*; WEAVER++ model) decompose speaking
into a **feed-forward cascade of stages**:

```
CONCEPTUALIZATION   →  FORMULATION                                      →  ARTICULATION
(message / preverbal    (a) grammatical encoding:                          (motor execution)
 plan: what to say)         lemma selection  →  function assignment
                            →  constituent assembly (argument structure)
                        (b) morphophonological encoding:
                            morphemes → phonological segments → syllabified
                            phonetic plan
```

Two architectural commitments matter for us (Bock & Levelt 1994; Garrett 1975/1988):
- **The functional level vs the positional level.** At the **functional level**, lemmas are
  selected and bound to *relational/thematic roles* (agent, patient, goal) — **order-free**. At
  the **positional level**, that unordered functional structure is converted into a
  *serially-ordered, phonologically-specified* string, inserting closed-class morphemes
  (determiners, inflections). [Verified: Bock & Levelt 1994 PDF; Thompson et al. agrammatism
  PMC.]
- **Grammatical encoding is LEXICALLY DRIVEN.** Selecting a lemma retrieves its **syntactic
  properties**, and the **verb lemma** in particular specifies *the number of arguments, their
  thematic roles, and their categorical + positional realization* — its **subcategorization /
  argument-structure frame**. "go" projects a GOAL → realized as a PP "to the park"; "give"
  projects agent + recipient + theme. [Verified: ScienceDirect "Beyond linear order: argument
  structure in speaking" 2021; Bock & Levelt 1994.]

**This is the precise biology of the owner's gap.** "The boy goes park" is missing exactly
what the *verb's argument frame* supplies: the PP-goal slot ("to the \_\_\_"), the determiner
slot ("the"), and tense ("goes" vs "go"). Our composer stores bare (agent, verb, patient) and
the CQ generator orders *those three*; nothing currently retrieves the *frame* that says "this
verb needs a goal-PP and these closed-class words."

---

### I.1 Conceptualization / message planning (the Levelt "what to say")

- **(a) Capability + example.** Before any words, the speaker builds a **preverbal message**:
  selects the event, its participants, perspective (active vs passive), information structure
  (what's given vs new), and which propositions to express and in what discourse order. E.g.
  deciding to say "*The boy* went to the park" (boy = topic) vs "It was *the park* the boy went
  to" (park = focus). Multi-sentence: ordering "First the boy left. Then he reached the park."
- **(b) Biological mechanism.** Distributed: the **default-mode / medial-PFC + precuneus**
  constructive network for simulating the to-be-described scene (catalog **G.09**, Kandel Ch 52
  pp 1300–1302; Schacter/Addis/Buckner), plus **dlPFC executive** sequencing of which
  proposition to express (Ch 52 pp 1292–1294). This is *content selection + discourse ordering*,
  not wording.
- **(c) Our sim.** **PARTIAL — this is the one production sub-capability the sim has the most of.**
  The **dlPFC content-selection Control** (`content_selection.py` /
  `content_selection_spiking.py`) selects "what to say next" over an association graph and is
  validated (Milestone-1 coherent dialogue 5/5 seeds; spiking Milestone-2 holds the discourse
  set, 6/6 seeds after the noise-tipped-Hopfield fix). Multi-sentence **discourse ordering** is
  also de-risked (the **ordered-WM topic-sequencing** GO 6/6,
  `2026-06-17-multisentence-ordered-emission-derisk.md`: hold a topic sequence in an order-encoded
  WM, emit one sentence per slot in slot order). What's MISSING: the *conceptualizer* doesn't
  build a structured message with perspective / given-new / argument roles — it picks a topic and
  hands a bare fact to the formulator.
- **(d) Cheap-first options (ranked).**
  1. **Reuse the validated content-selection Control as the conceptualizer (no new mechanism).**
     It already selects content + (with ordered-WM) discourse order. Add only a *message frame
     tag* (declarative vs answer vs yes/no) it emits alongside the selected content — that tag
     becomes the input to the formulator (I.3). Reuses: `ContentSelectionController`,
     `OrderedPositionWM`, `FrameCQ` (the multi-frame serial-order engine already
     frame-conditioned).
  2. **Perspective / information-structure as a primacy bias** (defer): topic = highest primacy
     in the CQ planning layer (so it surfaces first / as subject). Cheap because CQ *already*
     orders by a primacy gradient; "make the topic the subject" = "give the topic-role slot the
     top primacy." Anti-cheat: a permuted information-structure control (random topic assignment
     must NOT reproduce the topicalized order).
  - **Anti-cheat (both):** held-out propositions only; a no-confab check (an un-storable topic →
    abstain, never confabulate a message); report against the host baseline.

### I.2 Lexical selection / retrieval & tip-of-the-tongue (lemma → lexeme, two stages)

- **(a) Capability + example.** Mapping a concept to a word is **two-stage**: first the **lemma**
  (the abstract lexical-syntactic entry: `cat` = noun, count, its gender in gendered languages),
  then the **lexeme / word-form** (phonological code /kæt/). The dissociation is visible in
  **tip-of-the-tongue (TOT)**: the lemma is selected (you know it's a word, its grammatical
  gender, its first letter, # syllables) but the **word-form retrieval fails** — lemma-without-
  lexeme. [Verified: WEAVER++ "phonological-code retrieval is strictly conditional on selecting
  the lemma"; Levelt 1999.]
- **(b) Biological mechanism.** Lemma selection: **left middle temporal gyrus** (ventral stream
  semantic→lexical interface, Wernicke-adjacent; catalog **G.13**, Kandel Ch 55 pp 1384–1385).
  Word-form encoding: **posterior temporal + Broca's** (the dorsal sensorimotor mapping; G.11/G.12,
  Ch 55 pp 1380–1387). Selection is competitive (semantic neighbors compete → semantic
  paraphasias "headman" for "president" when selection fails, G.13). [Verified: Hickok & Poeppel
  dual-stream; Indefrey & Levelt meta-analysis (general knowledge).]
- **(c) Our sim.** **PARTIAL but RE-FRAMED.** The sim does NOT have a lemma/lexeme split as such;
  its "word" is a **G.20 sparse concept code**, and the read-out to a spelled word is the **A→W
  primitive** (`concept_speak_demo`, drive a concept pool → decode the spoken word via
  `lang_output` cosine, **100% A→W multi-seed**). Functionally the concept code ≈ "lemma+lexeme
  fused"; the A→W read-out ≈ the word-form encoder. TOT-style two-stage retrieval is not modeled
  (no separable "I know the gender but not the form" state).
- **(d) Cheap-first options (ranked).**
  1. **Treat the existing concept code as the lemma and the A→W read-out as the lexeme stage —
     and make selection competitive (already partly there).** The composer's **spiking NEF /
     Izhikevich-WTA cleanup** (CLAUDE.md "composer cleanup NEF GO"; the shipped one-brain WTA
     cleanup) IS the competitive lemma-selection step (winner = selected concept; off-target
     emits zero spikes). Reuse it; label it as lemma-selection. Near-zero new work.
  2. **Lemma/lexeme split (only if a TOT or gender-agreement capability is wanted later):** add a
     thin **word-form layer** between the concept pool and `lang_output`, gated on lemma
     selection (form retrieval fires only after the lemma WTA settles). Anti-cheat: a TOT probe
     (perturb the form layer → the model should still report lemma-level info — category — while
     failing the spelling), and a no-confab check (don't emit a *wrong* in-category word).
  - **Honest note:** TOT / lemma-lexeme dissociation is a *nice-to-have* for psycholinguistic
    fidelity, NOT on the owner's critical path (which is argument structure + the balloon). Rank
    it low.

### I.3 Grammatical encoding / ARGUMENT STRUCTURE (THE owner gap — highest leverage)

- **(a) Capability + example.** Given the functional message (verb `go`, agent `boy`, goal
  `park`), **build the syntactic constituent structure**: the verb's frame projects a
  subject-NP + a goal-PP; insert the closed-class scaffold ("**the** boy **goes** **to the**
  park"); assign tense/agreement ("goes" not "go"). The owner's "the boy goes park" is precisely
  a *functional structure that never got positional realization* — exactly **agrammatic Broca's
  output** (telegraphic, content words retained, closed-class + argument-marking omitted).
  [Verified: agrammatism = "omission of function words + inflectional morphemes"; "telegraphic
  speech lacks verbs and determiners"; "a determiner cannot be realized without a finite verb"
  → verb-frame drives determiner insertion — Wikipedia/Goodglass; Thompson PMC; Bastiaanse.]
- **(b) Biological mechanism.** **Broca's area** (left posterior IFG, pars opercularis +
  triangularis; catalog **G.12**, Kandel Ch 55 pp 1382–1384, Fig 55-6) — "supports comprehension
  of grammatically complex sentences; damage → labored, agrammatic speech, retained noun
  selection, lost function-word/verb use." Functionally this is the **positional-level processor**
  (Bock & Levelt 1994; Garrett): convert the unordered functional structure to a serially ordered,
  phonologically specified, closed-class-filled string. The **verb lemma carries the
  subcategorization frame** that licenses the arguments + their order; closed-class items
  (determiners, inflection) are inserted as the frame is realized. [Verified: Bock & Levelt 1994;
  "grammatical encoding is lexically driven, the verb lemma specifies number of arguments,
  thematic roles, categorical + positional info" — ScienceDirect 2021.]
- **(c) Our sim.** **MISSING — and this is the precise diagnosis of the owner's "skeletal
  sentence" gap.** The composer stores **bare (agent, verb, patient)** triples; there is **no
  verb-frame retrieval, no argument-structure expansion, no closed-class insertion, no
  tense/agreement.** The serial-order CQ generator
  (`2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`) orders a *given* fixed set of
  ≤3 role slots (and `FrameCQ` learns *distinct orders for distinct frames*, 6/6 — "the seed of
  syntax"), and the "morphology is a light host polish" the owner notes is exactly the missing
  positional step done in host code. So the sim has the **serial-order engine** but not the
  **frame that tells it WHICH slots exist** (subject-NP, goal-PP, determiners, tense).
- **(d) Cheap-first, biology-grounded options (RANKED).**
  1. **(RECOMMENDED) A small inventory of LEARNED verb-frame templates, selected by the verb
     lemma, that EXPAND the bare triple into ordered (role + closed-class) slots; then `FrameCQ`
     orders them.** This is the *minimal* realization of lexically-driven grammatical encoding at
     the project's scale.
     - **Biology:** the verb lemma's subcategorization frame (Bock & Levelt 1994); Pulvermüller
       **discrete combinatorial neuronal assemblies / sequence detectors** (DCNAs that fire to a
       LEARNED ordered category-pattern, generalizing across a syntactic class —
       Pulvermüller & Knoblauch 2009) = the substrate for "this verb-class projects this frame."
       Catalog **G.07/H.19** (internally-generated sequences) + the project's own
       `FrameCQ` (frame-conditioned serial order, already 6/6).
     - **Mechanism:** each *verb class* (or each verb) maps to a frame = an ordered list of
       **typed slots**: `[DET, SUBJ-N, V(+tense), DET, (PREP), OBJ/GOAL-N]`. Content slots are
       filled by the recalled fact's fillers (the existing A→W read-out spells each). **Closed-
       class slots (determiner, preposition, tense morpheme) are a tiny fixed inventory** —
       themselves emitted as words by the same A→W mechanism from a small "function-word pool."
       The frame's slot order feeds the *already-validated* CQ primacy gradient; CQ emits them in
       order; A→W spells each. The **frame is SELECTED by the verb lemma** (a learned
       verb→frame association, Hebbian), not by a host `if`.
     - **Why it fits the scale:** a handful of frames × ≤6 slots, a function-word pool of ~5–10
       closed-class items — squarely in the project's validated range. **Reuses:** `FrameCQ`
       (frame-conditioned order), the A→W read-out (per-slot spelling incl. closed-class), the
       composer's stored facts (content), the spiking WTA cleanup (frame/lemma selection). It is
       *mostly assembly of validated parts* — same posture that made the serial-order GO cheap.
     - **Anti-cheat (load-bearing):** (i) **held-out facts** with *held-out verb→frame
       combinations* (a verb whose frame was learned must generalize to a new agent/goal — the
       Pulvermüller generalization claim; test it, don't assume it); (ii) **frame-scramble
       control** (the wrong frame for a verb must NOT produce the right utterance — reuse
       `FrameCQ`'s cross-frame control which already scores 0.000); (iii) **closed-class-ablation
       probe** (drop the function-word pool → output degrades to the *current* "boy goes park"
       telegraphic form — proves the frame is what adds argument structure, and is a clean
       agrammatism-reproduction validation, catalog G.12 Behavioral validation); (iv) **no
       host-`if` frame selector** — the verb→frame map must be neural (a learned association /
       WTA), or it just relocates the template (the explicit Option-3 risk in the
       2026-06-16 scoping). (v) Host baseline reported.
     - **Honest scope / wall:** this gives **fixed per-verb-class frames** (a few argument
       structures), NOT open productive syntax. That is the right scale and matches biology's
       *learned* frame inventory; "fully productive recursive syntax from scratch" is the known
       generation wall (the BPTT-SNN ~134K-param negative, ~4 orders too small —
       `2026-05-07-Phase-2.3a-NEGATIVE`). Don't re-attempt productive syntax; build the frame
       inventory.
  2. **Argument structure as recursive composer clauses (partial reuse).** The composer already
     handles **recursive embedded clauses** (a fact whose patient is itself an SVO clause —
     register→register unbind, CLAUDE.md one-brain A5 cleanup). A PP-goal could be modeled as a
     sub-clause filler. **Lower priority:** this captures *embedding* but not the *closed-class
     scaffold* (determiners/tense), which is the bulk of what "boy goes park" is missing — so it
     complements Option 1, doesn't replace it.
  3. **Theta-gamma slot multiplex for the frame's argument positions (defer).** Encode the
     frame's ordered slots as theta-phase positions (each slot in its gamma sub-cycle; catalog
     **N.15**, Lisman & Jensen 2013; Heusser 2016). Most biologically-faithful "word-order = slots"
     account, but **highest build cost** (the project has *no theta/gamma generator in this path*,
     N.15 "Sim status: missing") and the project's own **dt-bound finding** (CLAUDE.md one-bridge
     step 3: equidistant rank-order neighbors tie at dt=1.0) makes fine phase-ordinal coding
     fragile. Defer behind Option 1, same call the 2026-06-16 scoping made.

### I.4 Serial order in production (LARGELY SOLVED — for the SVO frame)

- **(a) Capability + example.** Produce items in the right *order*: "boy" before "goes" before
  "park"; the classic serial-order constraints (errors are mostly *exchanges* of same-category
  items — "the boy goes" → "the goes boy" is rare; "park goes" exchanges are position-respecting).
- **(b) Biological mechanism.** **Competitive Queuing (CQ)** — a parallel *planning* layer holds
  all to-be-produced items with a **primacy activation gradient**, a *competitive choice* layer
  (WTA) emits the strongest, then **suppresses it** (inhibition-of-return) so the next wins
  (Grossberg 1978; Houghton 1990; **Bullock & Rhodes 2003**; Bullock 2004 TICS). Substrate:
  **pre-SMA/SMA** internally-generated sequences (catalog **G.07**, Kandel Ch 34 pp 822–828;
  **H.19**). Directly observed as a planning-layer activity gradient in primate motor cortex
  (Averbeck et al. 2002, general knowledge). [CQ is well-grounded; verified in the project's own
  2026-06-16 scoping + catalog G.07.]
- **(c) Our sim.** **GO (validated).** The **rate-coded CQ serial-order generator** is de-risked
  6/6 on the spiking substrate (`2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`):
  primacy gradient = graded current → per-pool spiking *rate* ranking = emission order; emit-then-
  suppress via the project's spiking `SaidTrace`. `FrameCQ` extends it to **frame-conditioned
  order** (distinct orders per frame, cross-frame control 0.000, 6/6) = the seed of syntax /
  active-vs-passive voice. This is wired opt-in into `BrainConversationalAgent(enable_neural_
  render=True)` (`describe()` word order is the spiking CQ read-out, no-confab moat preserved).
- **(d) Cheap-first options.** **Mostly done.** Remaining: (1) **wire `FrameCQ` into the agent's
  multiple reply frames** end-to-end (statement / who-what answer / yes-no / "X and Y associated")
  — flagged as the integration follow-on in the GO doc; (2) feed Option-I.3's *expanded* slot set
  (frame + closed-class) into CQ instead of the bare triple. No new mechanism; it's the
  consumer of the frame expansion. Anti-cheat: the existing permuted-ORDER + no-learning controls
  (reuse `song_g1_core.score_order` + `permuted_order_controls`, the harness that made the GO
  trustworthy).

### I.5 Fluency / disfluency & self-monitoring / self-repair

- **(a) Capability + example.** Speakers **monitor their own output** and **repair**: "the boy
  went to the— to the park" (covert/overt repair of a wrong word before/after it surfaces);
  disfluencies (pauses, "uh", restarts) mark monitoring + planning load. The system catches a
  retrieval error and re-selects.
- **(b) Biological mechanism.** Two accounts, both relevant: **(i) comprehension-based monitor**
  (Levelt/Roelofs — the speaker re-perceives inner+overt speech through the *comprehension*
  system and compares to intent), and **(ii) conflict-based monitor** (**Nozari, Dell & Schwartz
  2011**; Botvinick conflict-monitoring) — *internal* monitoring reads the **amount of competition/
  conflict in the production network** (between co-active semantic features or phonological units),
  assessed by the **anterior cingulate cortex (ACC)**, which signals lateral PFC to exert control.
  A lesion double-dissociation supports a production-internal conflict signal independent of
  comprehension. [Verified: Roelofs 2020 J.Cognition; Nozari et al. 2011; Botvinick et al.]
- **(c) Our sim.** **MISSING as an active monitor — but the substrate is unusually well-suited.**
  The project's **no-confab moat** (the learned **Bogacz-Brown familiarity gate**, validated to
  match the host abstention decision at V=320 with zero moat breaches) IS a production-side
  *gate* — it abstains when the recalled content's familiarity is below threshold. And the
  composer's **WTA cleanup margin** (winner rate vs runner-up) is *exactly a conflict signal*. So
  the sim already has (a) a gate and (b) a conflict-readable competition; what it lacks is wiring
  them as a **monitor→repair loop**.
- **(d) Cheap-first options (ranked).**
  1. **(RECOMMENDED, low cost) A conflict-monitor read-out over the existing WTA/cleanup
     competition that triggers abstain-or-reselect.** Biology = Nozari/Botvinick ACC conflict
     (Hopfield-energy / co-activation conflict). Mechanism: read the **margin** between the
     top-1 and top-2 in the spiking cleanup (high conflict = low margin); if conflict exceeds a
     calibrated threshold, **abstain** (emit a clean stop / "uh") or **re-run selection** with
     stronger inhibition (the project's biased-competition #2 mechanism). **Reuses:** the spiking
     WTA cleanup, the familiarity gate, `SaidTrace`. Anti-cheat: a calibrated conflict threshold
     (frozen, control-set), a no-confab assertion (high-conflict cases must abstain, not emit the
     wrong word), and a *false-alarm* check (low-conflict correct cases must NOT abstain). Maps
     to the existing moat discipline.
  2. **Comprehension-based monitor (defer):** feed the *emitted* word sequence back through the
     parser and compare to intent. **Higher cost + a known wall:** the project's recorded
     **G1/G1.5/P NEGATIVE** is precisely that the recognition-only substrate *cannot read order
     back out of itself* (the self-comprehension JUDGE failed, AUC 0.775) — so a comprehension-
     based *order* monitor is the documented dead path. The conflict-based monitor (Option 1)
     deliberately avoids this (it reads the production competition, not a re-comprehension). Use
     Option 1; cite the G1 negative as the reason not to do Option 2.
  - **Honest note:** disfluency *generation* (inserting "uh"/pauses to mark load) is cosmetic and
    out of scope; the *functional* monitor (catch + abstain/repair an error) is the real
    capability and Option 1 delivers it cheaply.

---

## II. WORKING MEMORY

### II.0 The capacity principle that fixes the balloon bug

**Human capability.** Active working memory is **fixed and small: ~4±1 chunks** (Cowan 2001,
"The magical number 4" — the focus of attention holds 3–5 meaningful items in young adults;
this is a *constant underlying* store distinct from task-specific strategies). Note the catalog's
N.15 frames the related theta-gamma buffer at **7±2** (Miller/Lisman-Idiart); the modern *pure-
capacity* estimate of the attentional focus is **~4** (Cowan). Either way the number is **a small
constant set by neural dynamics, NOT by how many things you could in principle remember.**
[Verified: Cowan 2001 BBS; Journal of Cognition 2024 "Is the magical number four, seven…".]

**This is the diagnosis of the dlPFC-WM-balloon bug.** Located precisely:
`content_selection_spiking.py:307` and `:359`: `n = max(600, 60 * len(self._vocab))`. The WM
region installs **one ~50-neuron Hopfield concept-attractor per vocabulary item**
(`SpikingLoopContextBuffer` installs an attractor per vocab entry; capacity = "how many of those
coexist"). So the bridge **scales linearly with the *storable inventory*** (60 × #concepts):
320 concepts → ~19,200 neurons in the WM region alone; thousands → it freezes. **That is the
inverse of biology.** Biology stores the *codes* in cortex/LTM (unbounded, cheap, distributed)
and the *active buffer* holds only ~4±1 *pointers/items* at a time (fixed size). The sim fused
storage INTO the buffer (an attractor per concept = the whole lexicon living inside the WM
region).

**The biologically-correct fix (the headline WM recommendation).**
> **Decouple capacity from inventory.** Make the WM a **FIXED-size active buffer** (size set by
> the ~4±1 / theta-gamma slot count, INDEPENDENT of vocabulary) whose slots **POINT AT** cortical/
> LTM concept codes — not a per-concept attractor bank. The number of *storable* concepts is
> unbounded (they live in the G.20 cortex codes the project already has); the number of
> *concurrently held* items is the small constant.

Two concrete realizations, both reusing project machinery (ranked under II.1/II.2):

### II.1 Persistent-activity WM (the sim's current type) — fixed-capacity bump/slot attractor

- **(a) Capability + example.** Hold the **currently-attended** referent(s) by sustained firing
  across a delay: "the boy" stays active while you formulate the rest of the sentence; the focused
  discourse referent for pronoun resolution ("he" → boy).
- **(b) Biological mechanism.** **Wang 2002 / Compte-Brunel-Wang 2000** recurrent attractor:
  strong recurrent excitation between like-tuned cells + broad inhibition → a self-sustaining
  **"bump" of persistent activity**; **NMDA receptors are essential** (slow kinetics stabilize
  the bump against AMPA ping-pong). dlPFC delay-period activity (catalog **G.06/G.08**, Kandel Ch
  34 pp 827–842, Ch 52 pp 1292–1294). Capacity is **finite and emergent** — a continuous bump
  attractor holds ~1 location precisely; discrete-slot variants hold a few items before bumps
  merge/drift (the capacity limit is a *dynamical* property, exactly Cowan's point).
  [Verified: Wang lab bump-attractor refs; "NMDA importance to working memory" Wang 1999.]
- **(c) Our sim.** **HAVE the mechanism, MIS-SIZED.** The project's dlPFC WM is the genuine
  Wang-2002 NMDA persistent-activity attractor (validated: NMDA bistability survives dt=1.0 at the
  genuinely-NMDA-dependent weight 30, CLAUDE.md one-bridge step 3; the cortico-PFC **loop-
  attractor** holds a *specific* concept at 220× specificity, `2026-06-03` finding; the loop holds
  a **SET of ≥3** concepts simultaneously = a WM span). **The bug is purely the sizing rule**
  (per-concept attractor bank scaled by `len(vocab)`), not the dynamics.
- **(d) Cheap-first fix (RECOMMENDED).**
  1. **Fixed K-slot buffer holding POINTERS, not a per-concept bank.** Allocate a **fixed** WM of
     ~K=4 slots (each a small attractor / bump), sized by K (constant), NOT by vocabulary. A slot
     holds a *pointer* (a sparse index / a low-D phasor / an engram-tag handle) to the cortical
     concept code, which lives in the existing G.20 cortex (unbounded, already built). Update =
     write a pointer into the least-recently-used slot (the BG-gated update, III.3); read =
     dereference the pointer to drive the cortical code. **Reuses:** the validated loop-attractor
     WM (now fixed-size), the **engram-tag API** (`commit_engram_tag`/`stimulate_tag`) as the
     pointer→code dereference, the G.20 codes as the store. The composer's existing
     **`OrderedPositionWM`** (items bound to gamma-slot POSITION phasors, ordered recall 1.000 @
     loads {2,3,5}, native D=256) is *already a fixed-slot pointer buffer* — it binds K items to
     K position phasors, independent of vocabulary. **This is the off-the-shelf realization of the
     fix.** Adopt `OrderedPositionWM` (or its unordered sibling) as the discourse-WM, replacing the
     `n=60*len(vocab)` attractor bank.
     - **Anti-cheat:** (i) **capacity-curve test** — load K=2,4,6,8 items, measure recall; recall
       must degrade past ~4–5 (the *biological signature*; the ordered-WM stress sweep already
       shows K≈4 clean at D=128, eroding at 5–6 — that IS the capacity limit, and it's the
       desired behavior, not a bug); (ii) **inventory-independence** — WM neuron count must be
       constant as vocab goes 16→320→3200 (the direct fix verification: the balloon is gone);
       (iii) **no-confab** — an un-storable / never-seen pointer must abstain (the moat).
     - **Why this is high-leverage:** it's a *freeze-at-scale* bug fix (the owner says the bridge
       "froze at scale"), AND it gives the clean ~4±1 home for "the verb frame + its few active
       arguments" from I.3 (a frame is ~4–6 slots = exactly WM capacity). One fix, two payoffs.

### II.2 Activity-silent (synaptic) WM — for the *unattended* discourse set

- **(a) Capability + example.** Hold items **without sustained firing**: the *unattended* items in
  a multi-item discourse (the referents you're not currently focusing) are kept in the background
  and can be **reactivated** when needed — e.g. you mention "boy" and "dog" and "park", focus on
  "boy" (persistent), but "dog"/"park" are held *silently* and snap back into focus when referenced.
  Lewis-Peacock et al. (2012): an unattended WM item drops to baseline firing yet is decodable on
  reactivation.
- **(b) Biological mechanism.** **Stokes activity-silent WM / Mongillo et al. 2008 synaptic theory**:
  information is held in **short-term synaptic plasticity** (residual presynaptic Ca²⁺ →
  facilitated synapses) rather than spikes; a brief reactivation pulse "pings" the silent trace
  back to activity. This is *cheaper and higher-capacity* than persistent firing (no metabolic cost
  of sustained spiking; many silent traces can coexist). The PFC interplay of persistent +
  activity-silent dynamics underlies serial biases (Barbosa et al. 2020). [Verified:
  Stokes; Mongillo et al. 2008 *Science*; Pals et al. biorxiv spiking Ca²⁺-STSP model; Barbosa
  2020 PMC.]
- **(c) Our sim.** **MISSING as a WM mechanism — but the substrate primitive EXISTS.** The engine
  has **short-term plasticity** (Tsodyks-Markram `stp_U`, `stp_tau_d`, `stp_tau_f`; per-connection-
  type STP; `fused_stp_decay_recovery`). That is the *exact* substrate Mongillo's synaptic-WM
  theory runs on (facilitating synapses, `tau_f`). The project has never deployed STP as a *memory
  store* (it's used for synaptic dynamics realism). So activity-silent WM is **buildable from an
  existing primitive**, not a new mechanism.
- **(d) Cheap-first options (ranked).**
  1. **(RECOMMENDED) Activity-silent background set via facilitating STP, focus via persistent
     attractor.** Hold the *focused* referent in the fixed K-slot persistent attractor (II.1);
     hold the *rest of the discourse set* as **STP-facilitated traces** (drive each briefly →
     `tau_f` facilitation persists silently); reactivate a backgrounded item with a ping pulse when
     it's referenced. **This directly relieves the balloon** (only the ~1 focused item needs a
     persistent attractor; the rest are silent synaptic traces, which are cheap and don't need a
     per-concept firing attractor). **Reuses:** the engine's STP fields (`enable_per_type_stp`,
     `stp_tau_f`), the engram-tag drive as the ping.
     - **Anti-cheat:** (i) **silent-then-reactivate decode** — a backgrounded item produces ~0
       delay-period firing yet decodes correctly after the ping (the Stokes signature; lesion the
       STP `tau_f`→0 and it must fail); (ii) **higher capacity than persistent** — show more items
       held silently than the persistent buffer holds (the theory's prediction); (iii) no-confab on
       a never-stored ping.
     - **Honest wall:** STP-based WM is **time-limited** (`tau_f` ~ hundreds of ms to ~1 s) — it's
       genuinely a *short-term* store, decaying without refresh. That matches biology (and the
       owner's "memory is reconstructive/lossy/OK" stance, `feedback_moat_not_hard_lossy_memory`),
       but means it's a *discourse-span* buffer, not long-term storage (LTM = the G.20 codes). Flag
       this; it's a feature (it forces the storage/buffer split), not a defect.
  2. **Leave activity-silent WM unbuilt and just use the fixed persistent buffer (II.1) + LTM
     pointers (defer).** If II.1's fixed K-slot pointer buffer + the unbounded G.20 store already
     kills the balloon and holds the discourse, activity-silent WM is an *optional fidelity
     upgrade*, not a blocker. Rank II.2 *after* II.1 — it's the more-faithful refinement, II.1 is
     the load-bearing fix.

### II.3 Holding discourse referents (the conversational use of WM) — LARGELY DE-RISKED

- **(a) Capability + example.** Across turns, keep the referents available: turn 1 "the boy saw a
  dog", turn 2 "*it* ran away" → "it" = dog (the more-recent/animate referent held in WM).
- **(b) Biological mechanism.** PFC persistent activity for the focused referent + activity-silent
  traces for the rest (II.1+II.2); the **focus of attention** (~1 item, Cowan/McElree) vs the
  broader ~4±1 set. Multi-referent disambiguation needs **biased-competition WTA inhibition**
  between referent attractors.
- **(c) Our sim.** **DE-RISKED GO + one mapped boundary.** The **`SpikingLoopContextBuffer`** holds
  the discourse set; **`MultiTurnAgent` + `SpikingLoopContextBuffer`** resolve a turn-2 pronoun to
  the held concept (de-risked GO 3-seed; reset/lesion break it; empty-WM abstains —
  `2026-06-17-multiturn-anaphora-derisk-GO.md`). **Mapped boundary:** *multi-REFERENT*
  disambiguation (which of several held referents a bare pronoun binds) needs **WTA biased-
  competition inhibition** between referent attractors — NOT recency, NOT salience (two converging
  NEGATIVEs, `2026-06-17-multireferent-disambiguation-NEGATIVE.md`). That is the *specified next
  mechanism* whenever multi-referent dialogue is prioritized.
- **(d) Cheap-first options.** (1) Re-base `SpikingLoopContextBuffer` on the **fixed K-slot**
  buffer (II.1) so it stops ballooning (the same fix); (2) for multi-referent: add the
  **biased-competition inhibition** the NEGATIVE doc specifies (reuse the project's #2 biased-
  competition mechanism). Anti-cheat: reuse the multi-turn-anaphora harness (reset/lesion/empty-WM
  abstain) + a recency-vs-competition control (the disambiguation must come from competition, not
  a recency heuristic, per the NEGATIVE).

---

## III. COGNITIVE CONTROL

### III.1 The dlPFC central executive (task-set, rule maintenance, top-down bias)

- **(a) Capability + example.** Hold the *current goal/task-set* and bias processing toward it:
  "I'm answering a who-question" biases the formulator toward the agent slot; maintaining "be
  truthful" biases toward abstaining over confabulating.
- **(b) Biological mechanism.** **dlPFC** as the source of top-down bias (Miller & Cohen 2001
  guided-activation; catalog **G.06/G.08**, Kandel Ch 34/52). Persistent rule representations
  (II.1) project to posterior/output regions and *bias the competition* there. The frontoparietal
  control network coordinates this. [Miller & Cohen 2001 general knowledge; catalog G.08.]
- **(c) Our sim.** **PARTIAL.** The dlPFC content-selection Control IS a central-executive instance
  (it maintains a discourse context and biases selection). The *frame tag* (I.1/I.3) would be a
  task-set the executive maintains. No general rule-maintenance / task-switching beyond
  content-selection.
- **(d) Cheap-first options.** (1) **Frame/intention as the maintained task-set** — the
  conceptualizer's frame tag (I.1) is held in the fixed WM (II.1) and biases the formulator's
  frame selection (I.3); this is the cheapest "executive" addition because it's a *use* of the WM
  + content-selection the project has. (2) Task-switching (defer): switch frames mid-dialogue
  (statement→question) and show the maintained frame biases output. Anti-cheat: a frame-switch cost
  (switching frames should transiently increase conflict / latency — the biological signature) and
  a no-leak check (the old frame must not contaminate the new utterance).

### III.2 Attention / inhibition / conflict monitoring

- Covered functionally above: **inhibition** = the project's `SaidTrace` inhibition-of-return
  (validated) + WTA lateral inhibition; **conflict monitoring** = the ACC conflict-readout over
  the production competition (I.5 Option 1; Nozari/Botvinick). **Attention** (selecting which
  WM item is in focus) = the biased-competition WTA between WM slots (II.3). No new cluster — these
  are read-outs/gates over machinery the project has. Cheap-first: the conflict-monitor (I.5.1) is
  the one genuinely-new, genuinely-cheap addition; it doubles as the "attention/control" signal
  (high conflict → recruit stronger inhibition = the Botvinick ACC→lPFC loop).

### III.3 Basal-ganglia GATING of WM updates (PBWM) — the principled "when to write to WM"

- **(a) Capability + example.** **Selectively update** WM: write a new referent into a slot only
  when it's task-relevant, otherwise *protect* the current contents (don't overwrite "boy" with
  every passing word). E.g. ignore filler words, but write the new subject when a new clause
  starts.
- **(b) Biological mechanism.** **PBWM — Prefrontal-cortex Basal-ganglia Working Memory**
  (**O'Reilly & Frank 2006**; Frank et al. 2001; Hazy et al. 2007): the **striatum** triggers a
  **dynamic gate** — striatal activation → **disinhibition of thalamus** → modulates the stability
  of PFC representations; **dopamine** (striatal D1/D2 + mesocortical VTA) trains *when* to gate
  (the gating policy is RL-learned). Separate **input gates** (write) and **output gates** (read/
  act); PFC organized into independently-updatable **stripes**. [Verified: O'Reilly & Frank 2006
  Neural Comp; LibreTexts CCN3e PBWM chapter; Wikipedia PBWM.]
- **(c) Our sim.** **PARTIAL — the substrate is unusually mature; the *gate-as-WM-update* is not
  wired.** The project's **BG cascade** (`g11_bg_runner` — per-action `cortex→D1/D2→GPi→thal→motor`
  with **disinhibition gating**, validated as the action-selection mechanism) IS the PBWM gating
  circuit (D1 disinhibits thalamus → releases the gate). The **transmission_gate** primitive
  (`RegionPathway(transmission_gate=...)` + `bridge.set_transmission_gate`) scales a pathway's
  effective current in [0,1] at runtime = **a thalamocortical gate** (Logiaco-Abbott-Escola 2021),
  already validated (`tests/test_transmission_gate.py`: closed→silent, open→fires, re-bind with
  zero weight change). And **dopamine RPE** trains the cascade. So all PBWM ingredients exist
  (striatal disinhibition gate + transmission gate + DA training); they just haven't been pointed
  at **"gate WM updates"** instead of "gate motor actions."
- **(d) Cheap-first options (ranked).**
  1. **(RECOMMENDED) Re-target the existing disinhibition / transmission gate as the WM input
     gate.** A `transmission_gate` on the `input → WM-slot` pathway, opened by a BG-cascade
     decision (write-now vs protect), trained by DA reward (write the *relevant* referent → reward).
     **Reuses:** the BG cascade, `transmission_gate`, the DA RPE machinery, the fixed K-slot WM
     (II.1). This is the *cleanest* biology-grounded answer to "when to write to WM," and it's
     almost entirely assembly of validated parts. It also directly serves I.3 (gate the
     verb-frame's arguments into their slots) and II.3 (gate a new discourse referent in).
     - **Anti-cheat:** (i) **selective-update test** — irrelevant inputs must NOT overwrite WM
       (the "ignore-the-distractor" PBWM signature); a gate-frozen-open control must FAIL (it
       overwrites indiscriminately) — proves the gate is load-bearing; (ii) **DA-lesion** — zero
       the gating DA → updating becomes indiscriminate or frozen (the Parkinsonian WM-updating
       deficit, the project's existing DA-lesion probe pattern); (iii) **the gate must be neural**
       (BG decision), not a host `if relevant: write` (else it relocates the shortcut — the
       brain-based-only standard).
  2. **Output gating (defer):** a separate `transmission_gate` on `WM → formulator` controls *when*
     a held item drives output (so a held-but-not-yet-uttered referent waits). Lower priority;
     build after input gating works.
  - **Why this is high-leverage but ranked behind I.3/II.1:** it's the *principled* fix for WM
    updating and reuses a lot, but it only matters once there's a fixed WM to gate (II.1) and a
    frame whose arguments need gating in (I.3). It's the natural *third* build.

---

## IV. WHAT THE SIM HAS vs LACKS (compact table)

| Capability | Biological mechanism | Sim status | Cheap-first fix (rank) |
|---|---|---|---|
| Conceptualization / message plan | DMN + dlPFC (G.09, G.08) | PARTIAL (content-selection Control + ordered-WM discourse order, GO) | reuse Control as conceptualizer + emit a frame tag (low) |
| Lemma → lexeme / TOT | MTG lemma, Broca form (G.11–G.13) | PARTIAL (concept code ≈ lemma; A→W ≈ lexeme; WTA cleanup = selection) | label existing pieces; lemma/lexeme split only if TOT wanted (low) |
| **Argument structure / grammatical encoding** | **verb subcat frame, Broca, functional→positional (G.12)** | **MISSING (bare SVO; "boy goes park"=agrammatic)** | **learned verb-frame templates + closed-class pool → FrameCQ (TOP)** |
| Serial order | Competitive Queuing, pre-SMA (G.07/H.19) | **GO** (rate-CQ 6/6; FrameCQ frame-conditioned) | wire FrameCQ into all reply frames; consume frame slots (low) |
| Self-monitoring / repair | comprehension-monitor / ACC conflict (Nozari/Botvinick) | MISSING as a loop (but moat + WTA-margin exist) | conflict-margin read-out → abstain/reselect (MED) |
| Persistent-activity WM | Wang-2002 NMDA bump attractor (G.06/G.08) | HAVE mechanism, **MIS-SIZED (balloon: 60×#concepts)** | **fixed K-slot pointer buffer (OrderedPositionWM) (TOP-2)** |
| Activity-silent WM | Stokes/Mongillo STSP | MISSING (but STP primitive exists) | facilitating-STP background set + ping reactivate (MED) |
| Discourse referents / anaphora | PFC focus + silent set; biased competition | **GO** (multi-turn anaphora); multi-referent = mapped NEGATIVE | re-base on fixed WM; add biased-competition for multi-referent (MED) |
| dlPFC executive / task-set | Miller-Cohen guided activation (G.08) | PARTIAL (Control) | frame tag as maintained task-set (low) |
| Conflict monitoring | ACC (Botvinick/Nozari) | MISSING | = the self-monitor read-out (MED) |
| **BG gating of WM updates** | **PBWM (O'Reilly & Frank 2006)** | PARTIAL (BG cascade + transmission_gate + DA all exist, not pointed at WM) | **re-target disinhibition/transmission gate as WM input gate (TOP-3)** |

---

## V. THE HONEST WALLS (flag, don't paper over)

1. **Fully productive recursive syntax is OUT OF REACH at scale** (the BPTT-SNN ~134K-param
   negative is ~4 orders too small vs ~1B reference; `2026-05-07-Phase-2.3a-NEGATIVE`). The
   *frame-inventory* approach (I.3) deliberately targets a *learned finite set of argument
   frames* — biology's own answer at human scale is also a learned inventory plus generalization,
   so this is the right scope, not a cop-out. Do NOT re-open open-ended generation.
2. **Comprehension-based order monitoring is a documented dead path on this substrate** (G1/G1.5/P
   NEGATIVE: the recognition-only substrate can't read order back out of itself, AUC 0.775). The
   self-monitor must be **conflict-based** (read the production competition), not re-comprehension.
3. **Fine phase-ordinal coding (theta-gamma slots) is dt-fragile here** (CLAUDE.md one-bridge step
   3: equidistant rank-order neighbors tie at dt=1.0). Theta-gamma WM/serial-order is the most
   *faithful* account but the highest build cost + dynamics risk — defer behind rate-coded CQ +
   the fixed-slot attractor (which the project has already shown work).
4. **Persistent-activity WM has a real, small capacity** (~4±1; the ordered-WM erodes past K≈4 at
   D=128). This is a **feature to preserve** (the biological signature), not a bug to engineer
   away — the fix is the storage/buffer split (II.0), not "make the buffer hold 320 things."
5. **STP-based activity-silent WM is genuinely time-limited** (`tau_f` ~hundreds of ms–~1 s,
   decays without refresh). It's a discourse-span store, not LTM — which is correct biology and
   aligns with the owner's lossy-memory stance, but must not be claimed as durable storage (LTM =
   the G.20 cortex codes).

---

## VI. TOP 3 HIGHEST-LEVERAGE BUILD TARGETS (with why)

### 🥇 TARGET 1 — Verb-frame argument structure (the conceptualization→formulation pipeline)
**Build:** a small inventory of **learned verb-frame slot templates** selected by the verb lemma,
expanding the bare (agent, verb, patient) into ordered **(content + closed-class) slots**
[`DET, SUBJ-N, V(+tense), (PREP), DET, GOAL/OBJ-N`], fed to the validated `FrameCQ` serial-order
generator, with closed-class words emitted from a tiny function-word pool via the A→W read-out.
**Why #1:** it is the **direct fix for the owner's headline gap** ("the boy goes park" → "the boy
goes to the park"), it is **mostly assembly of already-validated parts** (FrameCQ, A→W, WTA
cleanup, stored facts), it is **squarely in the project's scale** (a handful of frames × ≤6
slots), and it has a **clean agrammatism-reproduction anti-cheat** (ablate the function-word pool
→ telegraphic output = the current behavior, proving the frame is what adds argument structure).
Biology: Bock & Levelt 1994 functional→positional; verb subcategorization (G.12 Broca);
Pulvermüller DCNAs for the learned, generalizing frame.

### 🥈 TARGET 2 — Fixed-capacity WM (kill the balloon) via the storage/buffer split
**Build:** replace the `n = 60 * len(vocab)` per-concept attractor bank with a **FIXED K≈4 slot
buffer holding POINTERS** to the unbounded G.20 cortical codes — adopt the project's own
**`OrderedPositionWM`** (already a fixed-slot, vocabulary-independent pointer buffer) as the
discourse-WM; dereference a slot via the engram-tag / pool-pattern drive.
**Why #2:** it fixes a **freeze-at-scale BUG** the owner explicitly flagged (the WM bridge balloons
with fact/vocab count), the fix is the **biologically-correct principle** (Cowan ~4±1 active buffer
vs unbounded LTM; storage ≠ buffer), the realization **already exists in the codebase**
(`OrderedPositionWM`), and it **also provides the clean ~4±1 home for Target 1's frame slots** (a
verb frame is ~4–6 slots = WM capacity). One fix, two payoffs. Anti-cheat: WM neuron count constant
as vocab 16→320→3200; capacity curve degrades past ~4–5 (the desired biological signature).

### 🥉 TARGET 3 — BG-gated WM updates (PBWM) + the conflict-monitor (paired control layer)
**Build:** (a) re-target the existing **disinhibition / `transmission_gate` + DA-RPE** machinery as
a **WM input gate** (write a referent/argument into a slot only when a BG decision says relevant,
trained by dopamine) — PBWM (O'Reilly & Frank 2006); (b) a **conflict-margin read-out** over the
spiking WTA cleanup (ACC-style, Nozari/Botvinick) that triggers **abstain-or-reselect** on high
production conflict.
**Why #3:** these are the **principled executive controls** that make Targets 1–2 *robust* in real
dialogue — (a) stops indiscriminate WM overwriting (so the frame's arguments and discourse
referents are protected/updated correctly), and (b) catches a garbled utterance and abstains/repairs
(the production-side complement of the no-confab moat). Both are **almost entirely assembly of
validated parts** (BG cascade, transmission gate, DA RPE, WTA cleanup, familiarity gate) and reuse
the project's existing anti-cheat discipline (DA-lesion, gate-frozen control, no-confab assertion).
Ranked behind 1–2 because they only bite once there's a fixed WM to gate (T2) and a frame whose
arguments need gating + monitoring (T1).

**Sequencing rationale.** T1 delivers the visible capability the owner wants; T2 unblocks scale AND
houses T1's slots; T3 hardens both. Each is cheap-first, reuse-by-import, and carries a
pre-registerable anti-cheat that reuses the project's existing harnesses (`song_g1_core`
permuted/held-out controls, the no-confab moat, the DA-lesion probe).

---

## Files / entries reviewed (provenance)

**Catalog** (`sim-catalog/references/feature-catalog.md`): Cluster G entries **G.06** (PFC delay
activity, Ch 34 827–842), **G.07** (pre-SMA internally-generated sequences, Ch 34 822–828),
**G.08** (PFC WM persistent activity, Ch 52 1292–1294), **G.09** (constructive future simulation,
Ch 52 1300–1302), **G.10** (language as hierarchical symbolic system, Ch 55 1370–1372), **G.11**
(dual-stream language, Ch 55 1380–1387), **G.12** (Broca's area / grammatical processing, Ch 55
1382–1384, Fig 55-6), **G.13** (Wernicke's area / lexical selection, Ch 55 1384–1385), **G.15–G.18**
(signal-detection / drift-diffusion / LIP accumulator, Ch 56), **N.15** (theta-gamma multiplexed WM
buffer, Buzsáki Cycle 12 350–353 / Lisman-Idiart 1995), **H.19** (premotor sequential action).
**Kandel 6e** `textbooks/kandel-pns-6e/full-book.txt` — Ch 34 (PFC/pre-SMA), Ch 52 (cognition/WM),
Ch 55 (language/Broca/Wernicke), Ch 56 (decision) — anchors verified present.
**Glossary** `references/glossary.md` — **ABSENT on disk** (noted; substituted with literature).
**Project findings:** `2026-06-16-sentence-generation-biologization-deep-research.md`,
`2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`,
`2026-06-17-multisentence-ordered-emission-derisk.md`,
`2026-06-03-content-selection-milestone2-spiking-dlpfc-persistence-CHARACTERIZED.md`,
`2026-04-27-pfc-working-memory.md`, `2026-06-17-multiturn-anaphora-derisk-GO.md` (+ the
`2026-06-17-multireferent-disambiguation-NEGATIVE.md` boundary), `2026-05-07-Phase-2.3a-NEGATIVE`
(generation-scale wall). **Source code located:** the balloon bug at
`research/runners/content_selection_spiking.py:307,359` (`n = max(600, 60 * len(self._vocab))`);
`OrderedPositionWM` (the fixed-slot pointer buffer) and `FrameCQ` (frame-conditioned serial order)
as the reusable parts; `transmission_gate` / BG cascade / DA-RPE as the PBWM ingredients.

## Literature cited (load-bearing)

- **Levelt, Roelofs & Meyer 1999**, *Behavioral & Brain Sciences* 22:1 — "A theory of lexical
  access in speech production" (WEAVER++): staged conceptualization→lemma→morphophonological
  encoding; phonological retrieval conditional on lemma selection.
  https://pubmed.ncbi.nlm.nih.gov/11301520/ ; blueprint:
  https://www.mpi.nl/world/materials/publications/levelt/Levelt_Producing_spoken_language_1999.pdf
- **Bock & Levelt 1994**, "Language production: grammatical encoding" — functional vs positional
  level; lexically-driven grammatical encoding.
  http://www.colinphillips.net/wp-content/uploads/2023/04/bock1994.pdf
- **"Beyond linear order: the role of argument structure in speaking" 2021**, *Cognitive
  Psychology* — verb lemma specifies # arguments, thematic roles, categorical+positional info.
  https://www.sciencedirect.com/science/article/abs/pii/S0010028521000219
- Agrammatism (Broca) = omission of function words + inflections; verb omission ↔ determiner
  omission. https://en.wikipedia.org/wiki/Agrammatism ;
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3026288/
- **Cowan 2001**, *Behavioral & Brain Sciences* 24:1 — "The magical number 4 in short-term memory":
  focus-of-attention capacity ~3–5 chunks. https://philpapers.org/rec/COWTMN ;
  modern review https://journalofcognition.org/articles/10.5334/joc.387
- **Wang 2002 / Compte-Brunel-Wang 2000 / Wang 1999** — recurrent NMDA bump-attractor persistent
  activity; NMDA essential to WM. https://pubmed.ncbi.nlm.nih.gov/10531461/ ; bump attractor:
  https://www.cns.nyu.edu/wanglab/publications/pdf/ens1397.revision-14september2007.pdf
- **Stokes (activity-silent WM); Mongillo, Barak & Tsodyks 2008**, *Science* — synaptic theory of
  WM (short-term synaptic plasticity holds info without firing); spiking Ca²⁺-STSP model
  https://www.biorxiv.org/content/10.1101/823559.full.pdf ; persistent×silent interplay (Barbosa
  2020) https://pmc.ncbi.nlm.nih.gov/articles/PMC7392810/
- **O'Reilly & Frank 2006**, *Neural Computation* — PBWM: striatal disinhibition gate + DA-trained
  selective updating; input/output gates; PFC stripes.
  https://cseweb.ucsd.edu//~gary/PAPER-SUGGESTIONS/OReillyFrank06_pbwm-neural-comp-2006.pdf ;
  https://en.wikipedia.org/wiki/Prefrontal_cortex_basal_ganglia_working_memory
- **Nozari, Dell & Schwartz 2011**, *Cognitive Psychology* — conflict-based self-monitoring (ACC
  reads production conflict). https://pubmed.ncbi.nlm.nih.gov/21652015/ ; comprehension-based
  defense (Roelofs 2020) https://journalofcognition.org/articles/10.5334/joc.61 ;
  Botvinick conflict-monitoring (ACC).
- **Bullock & Rhodes 2003**; Grossberg 1978; Houghton 1990; Bullock 2004 *TICS* — competitive
  queuing for serial order (the sim's validated CQ generator).
- **Pulvermüller & Knoblauch 2009**, *Neural Networks*; Pulvermüller 2010 — discrete combinatorial
  neuronal assemblies / sequence detectors (the substrate for learned, generalizing syntactic
  frames).
- **Lisman & Jensen 2013**, *Neuron* (θ-γ neural code); Heusser et al. 2016, *Nat. Neuro.* —
  theta-gamma ordinal-position code (the deferred faithful slot-multiplex option).
