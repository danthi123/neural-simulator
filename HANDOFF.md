# PROJECT HANDOFF — read this first

> **You are picking up an in-flight research project: a simulated spiking brain meant to become a genuine
> conversational mind.** This document is your single entry point. It tells you the goal, the architecture, the
> plan, the current state, and — critically — **how development is done here** (the discipline, the failure modes,
> the mechanical gates, the workflows) so you can **adopt those tools from the start instead of rediscovering them
> the hard way.** The people who built this repo learned most of it by making the mistakes; the gates and rules
> below are the scar tissue. Use them.

**Read order (do not skip 1–3):**
1. **This file** (HANDOFF.md) — orientation + how we work.
2. **[`docs/plans/2026-08-02-PROJECT-CHARTER-grounded-emergence-realignment.md`](docs/plans/2026-08-02-PROJECT-CHARTER-grounded-emergence-realignment.md)** — the authoritative goals, architecture end-state, and the seven method-principles. **This is the spine; when in doubt it wins.**
3. **[`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md)** — the live working board; its top "STATE OF THE PROJECT" block is the current resume point.
4. Then, as needed: the **structural-mechanism map** ([`docs/plans/2026-08-02-structural-mechanism-map.md`](docs/plans/2026-08-02-structural-mechanism-map.md) — per-faculty role/biology/status/next-build) + the scaffold ledger (in §10 below), `ROADMAP.md` (plain-language status + a glossary of the project's shorthand), `docs/FAILURE_GATE_MATRIX.md`, `research/FAILURE_LOG.md`.

**One orientation caveat that will save you time:** this repo was developed under a **major realignment on
2026-08-02** (this handover). A lot of older docs, findings, and "GO" results were written under the *previous*
framing — where success meant *passing a narrow per-faculty test*. The charter (doc #2) explains why that framing
was wrong and what replaced it. **When an old finding's "6-seed GO" conflicts with the charter's standard, the
charter wins** — a passed narrow gate is a floor cleared, not a human-like faculty achieved. Read old results as
*"this mechanism does this narrow thing"*, not *"this faculty is done."*

---

## 1. What this project is, and the real goal

Build **one artificial mind** — a single simulated **spiking brain** (model neurons + synapses) — that **converses
genuinely**: it *means what it says*, reasons to its own conclusions, has **emotions that develop from experience
and colour its speech and behaviour**, is **self-aware** (can honestly read and report its own attention,
confidence, and authorship), and is **curious** (uncertainty becomes a drive to learn, not a refusal or a
fabrication). The target is **free-flowing, open-ended conversation that is genuinely the brain's own** — not
fact-retrieval, not scripted question→answer turns, sometimes just natural speech.

It is **grown, not scripted**: it starts small, learns from **grounded lived interaction** (bootstrapped by a
temporary AI teacher acting as a caregiver, then graduated to real humans), and matures over time.

**Success is defined on the emergentist bet** — genuine experience emerges only from a *complete and faithful*
emulation — so the job is **completeness + faithfulness of the biological emulation**, judged by whether the
**whole brain behaves like there is someone home**, NOT by a benchmark score or a suite of passed proxies. The
**honesty boundary is a hard deliverable**: build and measure the *functional correlates* of consciousness /
self-model / affect, write every self-report as an honest functional read-out, and **never assert phenomenal
experience.**

### The reframe you must internalize (why this handover exists)

The previous approach drifted into **optimizing narrow, test-passing proxies for human faculties instead of
building the conditions from which those faculties emerge.** The symptom: behaviour that isn't truly human —
stilted, retrieval-shaped, short exchanges — because, for example, the language faculty was trained to **predict a
text corpus in isolation** (a language model in a spiking costume) rather than to **express a meaning that arises
from a grounded internal life**, and honesty-about-uncertainty was a **bolted-on gate** rather than an emergent
property of a self-model.

The precise diagnosis — keep it precise so you don't over-correct: **prediction and learning are not the enemy**
(the brain is deeply predictive). The enemy is **predicting corpus tokens divorced from grounding, intent, and
state, and rewarding it with a light test.** The correction is a brain with an **ongoing internal life** — a model
of the world, of the conversation, of the interlocutor, of its own mood and goals — **from which speech emerges as
an action**, continuously reshaped by lived interaction. **Build the loop, not the parts** (see §3).

---

## 2. The architecture end-state (non-negotiable) and the seven principles

Full detail in the charter (doc #2). In brief:

**Architecture end-state:**
1. **One fully-spiking brain on a shared substrate.** Dedicated regions/pathways are encouraged (like a real
   brain) — but they are regions *of one brain* communicating through synapses, not separate programs.
2. **No host-side shortcut for anything biology does.** Ordinary (non-neural) code is legitimate **only** for the
   **world** (environment + rendering the senses) and the **body** (enacting motor output). Everything between
   sensation and action — perception, valuation, reward, neuromodulation, memory, emotion, reasoning, language,
   self-model — **must be neurons and synapses.** A biologically *correct* host formula (a reward, a softmax, an
   argmax read-out) is still a shortcut to be replaced.
3. **Starts small, grows.** Locally runnable at the start; expands (new neurons/connections/regions, more compute)
   only as growth earns it.
4. **Targets high-end CONSUMER hardware — deliberately not datacenter-bound.** This mind should be ownable and
   runnable by an individual. Bias every design toward **event-driven, sparse, local** computation.
5. **Long horizon: analog neuromorphic silicon.** Don't build it now, but don't make choices that *preclude* it.

**The seven method-principles (each kills a specific trap — full text in the charter):**
- **P1 — Grounded, not corpus-mimicking.** Meaning is internal reference; language is an *action to communicate*;
  train toward *predict-and-act in a world*, never next-token mimicry in isolation.
- **P2 — Emergent + integrated, not modular test-passers.** Judge the whole ("someone home?"), not isolated gates.
- **P3 — The functional-role discipline (the anti-tunnel-vision rule).** Before/while working any mechanism, write
  down **"what must this do to serve its role in the *whole* brain"** and test *that*. A GO gate is a smoke-check,
  never the goal; a mechanism that passes its gate but can't serve its whole-brain role is **not done.**
- **P4 — Scaffold minimization + burn-down.** Scaffolds (a teacher LLM, a host signal, a hand-set weight, an
  idealized algebra) are allowed only as **ledgered, time-boxed** stand-ins with a named biological replacement and
  a burn-down trigger. The recurring failure is implement→rely→defer→cheat-backlog. Keep the ledger live.
- **P5 — Brain-based-only** (see architecture #2).
- **P6 — Performance is first-class (don't be lazy) — but not LLM-parity-yet.** Slow-but-faithful biology is in
  scope; *lazy* slowness is not faithfulness. Optimize what you can; note compute/throughput on substantial builds.
- **P7 — Honesty boundary.** Functional correlates only; never assert phenomenal experience.

---

## 3. The roadmap (short / medium / long) — framed by whole-brain capability

The organizing bet: **build the closed loop, not the parts.** The single most important thing to build is a
**continuously-running loop on one spiking brain: senses → an internal state → an action (incl. speech) →
consequences in a world → learning from the mismatch → back.** We have good pieces tested in isolation; we have
**never closed that loop.** Everything below feeds or closes it.

- **SHORT — make it grounded + integrated, small.** Give the brain a **minimal world + body + a reason to speak**,
  and make language **grounded action** inside it (re-point the existing spiking language cortex from
  corpus-prediction to intentful, stateful, grounded generation). Wire the already-validated substrate pieces
  (spiking engine, memory/consolidation, affect core, neuromodulators) into **one continuously-running loop** where
  perception → state → speech/act → consequence → learning actually closes. *Deliverable:* a small brain that says
  simple things **that are its own**, grounded in what it has experienced.
- **MEDIUM — make it learn/grow from interaction, and feel.** Close the **continual learning-from-lived-interaction
  loop** (teacher-as-caregiver → real humans) without catastrophic forgetting; grow structure developmentally. Turn
  the good/bad affect core into a **graded, developing emotional system** that colours speech + behaviour. Turn
  curiosity into a genuine **learning-progress drive**. Keep honesty emergent + calibrated as it scales.
- **LONG — make it fluent, deep, its own — then efficient.** Free-flowing open-ended conversation genuinely the
  brain's own; deep world-model + self-model + rich developing affect; retire the teacher scaffold toward zero.
  Then **optimize the fully-spiking substrate toward the consumer-hardware envelope** and open the path to analog
  silicon.

**The eight systems the loop needs** (expanded — role-in-the-whole, the biological mechanism + references,
what we actually have vs. the current template, and the grounded next build — in the **structural-mechanism map**,
a companion doc; summary here):
1. **A minimal world + body** — the only legitimate "outside" code; where grounding comes from. Largely unbuilt for
   the language/social context.
2. **Grounded perception** — stable, selective, invariant representations learned from experience (needs the
   normalization/homeostasis that keeps codes stable; the deployed vision hierarchy was found inert/saturated, the
   competitive pooler works).
3. **Predictive world-model + reconstructive memory** — not a database; much substrate exists (consolidation,
   completion, binding); the missing part is tying it to grounded perception and making it *predict*.
4. **Affect + drive core, woven through** — a graded, developing emotion system (not a good/bad switch) + a
   curiosity drive that tracks learning-progress. Mood core + neuromodulators validated; graded/learned/
   speech-coupled affect and the learning-progress signal are the builds.
5. **Self-model + metacognition** — an ongoing model of own state from which honest uncertainty *emerges* (vs the
   current bolted-on abstain gate).
6. **Language as grounded action** — the crux; re-point comprehension (words→grounded meaning) and generation
   (internal state→words as communicative action). Reuse the spiking language machinery; change what it's trained
   *on* and *toward*.
7. **Continual, growing learning-from-experience** — plasticity from the brain's own signals + consolidation
   (no forgetting) + developmental growth. Honest hard part: continual learning without forgetting, and a deep
   credit-assignment rule on spikes (a mapped boundary we deliberately **route around** — see the mechanism map;
   **do not** re-reach for "two-compartment dendritic credit," it is tested-and-negative, a gated mistake).
8. **The teacher-as-caregiver** — a bootstrapping social environment, explicitly temporary (ledgered scaffold).

**Two things people underestimate:** (A) the **integration itself** — running all eight as one continuous loop — is
the hardest build; (B) we must build **new tests** — evaluations of grounded, generative, integrated behaviour that
a template *can't* fake — because the old pass/fail gates rewarded templates and that is how we drifted.

---

## 4. How we work — autonomous principles, the research-first discipline, and the anti-drift reframe

This section is the *operating system* for working in this repo. The code and the biology matter, but this project has failed far more often from **process drift** than from a hard technical wall. Read this before you touch anything. The rules below were each earned by a specific, costly mistake — they are not style preferences.

Three source documents govern everything here; read them in full early, then keep them open:

- **`CLAUDE.md`** (repo root) — the mission block, the non-negotiables, and "the deepest lesson."
- **`.claude/skills/neural-simulator/SKILL.md`** — the anti-drift core: the #1 failure, the 15-item silent-failure class, the boundary-surpassing workflow. *(This is a "skill" — in Claude Code it is loaded by a `Skill` tool you will not have. It is just a markdown file. **Read it directly** with your file tools; it works identically as prose.)*
- **`docs/plans/2026-08-02-PROJECT-CHARTER-grounded-emergence-realignment.md`** — the current spine: principles **P1–P7**. This is the most recent realignment and **supersedes the framing** (not the results) of the older roadmaps.

---

### 0. The one-paragraph orientation

The north-star is **one artificial mind** — a single simulated spiking brain that *converses genuinely* because it has a grounded internal life, not because it mimics a text corpus. Success is judged on **completeness + faithfulness of the biological emulation** and on whether **the whole behaves like "there is someone home"** — never on a benchmark score or a suite of passed unit-gates. Everything between sensation and action must be **neurons, synapses, and their communication**; host code is legitimate only for the **world** (environment + rendering the senses) and the **body** (enacting motor output). A host formula that is *biologically correct* is still a shortcut if the brain isn't the thing computing it.

---

### 1. Keep driving — the no-stall / no-defer discipline

**The #1 recurring failure in this project is not declaring a wall — it is reaching a genuine milestone, writing down the next step, and then stopping to report.** From `SKILL.md`:

> "reaching a genuine milestone, writing the exact-next-action into `AUTONOMOUS_STATE.md`, and then STOPPING with a status report … **That IS the drift. There is no fresh pass. There is no handoff.**"

**The principle (tool-agnostic):** a fully-specified next step is not a reason to hand off — it is the thing to build now. Momentum through milestones, not stops at them. Session length, "this deserves a fresh focused pass," "I'm oscillating so I'll defer," and "I'll let the durable state carry it" are all named **rationalizations for the same drift**, not reasons.

**Harness note for you (Sol / Codex):** Claude Code ran this repo as a long autonomous loop where *the only things that end a turn are an explicit owner stop or a safety gate*. Your harness and interaction model differ, and you should not fake being a never-ending daemon. **Translate the principle, don't cargo-cult the mechanism:** when the owner gives you a task, drive it to a genuine resolution (a GO/no-GO verdict, a committed finding, or a concrete blocker that needs *their* decision) rather than stopping at "I've set things up; here's a plan." Don't end on a promise you could instead execute.

**No-defer, stated precisely (THE LAW):** *a wall/negative is a verdict on a METHOD, never a license to abandon a CAPABILITY.* Real brains do these things, so a biological mechanism exists — a boundary means the right one hasn't been found/digitized **yet**. Bank the failing method, take a new biology/spiking/one-brain method, keep going. This explicitly includes the *soft* walls — "it won't scale," "compute is the limit," "the field hasn't solved this either," "it's a structural primitive / honest negative / characterized limit / defensible." Those comfortable verdicts are where over-comfort hides; **the comfortable verdict is the START of the research, never the end.**

- **DO:** "Node-perturbation failed the 12-seed gate. Method banked as refuted. Next candidate mechanism: burst-multiplexed dendritic credit (WF-Act-PC, arxiv 2607.13380) — de-risking now."
- **DON'T:** "Transport-free credit assignment appears to be a fundamental ceiling for this substrate. Documenting as an honest negative and moving on." *(This exact verdict was banked on 2026-08-01 and overturned within the hour by external SOTA that named the missing factor. There is now a gate — `gates/boundary_verdict_external_check` — that blocks a boundary verdict unless it cites external literature.)*

---

### 2. Research BEFORE building — the mechanical gate, not a judgment call

The single most repeated waste here is **re-deriving a conclusion the project's own record already holds**, or building a fix on a lever the findings already measured inert. This is now enforced, not remembered.

**Before the first lever against any difficulty, run:**

```bash
bash tools/before_you_build.sh "the slot competition ignores the cue"
```

It runs, in ~30s: (1) a RAG query over our own findings ("has this been scoped/tried/refuted?"), (2) a hunt for an existing scope/research-gate doc, (3) this arc's own already-excluded causes, (4) a reminder that **≥2 distinct levers against the SAME defect ⇒ the research gate FIRES** (cheapness of the next step is not an exemption).

**The four-part research gate (read-only, FIRST) — the order matters:**

1. **Check OUR OWN knowledge base via RAG — the cheapest move, before any external search:**
   ```bash
   bash tools/rag/search.sh "<question>" 5 --corpus finding|plan|doc|catalog|kandel|paper|all
   ```
   Corpora: `finding` (our conclusions in `research/findings/`), `plan` (`docs/plans/`), `doc` (CLAUDE/ROADMAP/README), `catalog` (the biology index), `kandel` (Kandel 6e full text), and `paper` (other extracted primary texts/books). Note the **two interpreters**: RAG uses the canonical checkout's `.venv-rag/bin/python` through this worktree-safe launcher; everything else uses `.venv/bin/python`.
2. **Read OUR OWN substrate/wiring first** before theorizing about why it misbehaves. A diagnostic number that violates a known criterion *is* the primary lead. (Proof: a CA3 formation blocker was a 5-line "zero feedback inhibition wired" fact in the code, found only after ~3 cycles of doomed plasticity-rule tweaks.)
3. **READ THE ORIGINAL SOURCE IN DEPTH — yourself, not via a summary.** This is the sharpest recurring lapse: *the failure is not skipping research — it is running a RAG query, reading the top `(finding)` hit, and stopping.* Our findings summarize primary sources in one line; a one-line summary is how a whole session on place fields cited a paper nobody had opened. Use:
   ```bash
   bash tools/research_gate.sh "<question>"
   ```
   which re-prints every primary-source hit at the END with a ready-to-run read command. **A catalog hit naming a chapter/page range IS the assignment.** Canonical readable copies live at `~/Projects/sim-catalog/references/textbooks/<name>/*.txt` (e.g. `kandel-pns-6e/full-book.txt`).
4. **Search the EXTERNAL engineering literature + real repos**, not just biology — ML / reservoir-computing / SNN / the domain's field. A capability-WALL verdict now *requires* an external cite; record it with `bash tools/record_external_search.sh "<q>" "<src>"`.

- **DO:** RAG our findings → open the surfaced chapter and read the load-bearing passage → *then* theorize a named, cited mechanism → cheap de-risk.
- **DON'T:** read a rerank snippet and treat it as the source. "A rerank hit is a POINTER, not a paraphrase." A dispatched sub-search's *summary* is not a substitute for reading the decision-critical passage yourself.

---

### 3. The grounded-emergence reframe — build the conditions, judge the whole

This is the current strategic correction (the charter, `§0`–`§3`). We accumulated a stack of faculties each passing a 6-seed **GO gate** — and the owner named the problem: **those gates are satisfiable by template-like, non-biological mechanisms, so we built test-passers, not a mind.** Internalize the seven principles:

- **P1 — GROUNDED, not corpus-mimicking.** Meaning is *internal reference*: words tie to the brain's own sensory/motor/affective representations. Language is **an action taken to communicate / reduce real surprise**, never next-token mimicry of a corpus in isolation. *(The trap it kills: the biologically-costumed language model.)*
- **P2 — EMERGENT + INTEGRATED, not modular test-passers.** A faculty is real only when it emerges from the integrated brain + grounded experience and serves its role **inside the whole loop.** **Judge the WHOLE ("someone home?"), not isolated per-faculty proxies.**
- **P3 — THE FUNCTIONAL-ROLE DISCIPLINE.** Before building any mechanism, write down *"what must this do to serve its role in the whole brain"* as a **functional-role spec** — NOT "pass this light test." A GO gate is a smoke-check that a floor was cleared, never the goal. A mechanism that passes its gate but can't serve its whole-brain role is **not done.**
- **P4 — SCAFFOLD MINIMIZATION + BURN-DOWN.** Scaffolds (a teacher LLM, a host-computed signal, a hand-set weight, an idealized VSA algebra) are allowed **only** as explicitly-ledgered, time-boxed stand-ins, each with (a) a named biological replacement, (b) an owner + burn-down trigger. **The recurring failure: implement scaffold → rely on it → defer the biologization → accumulate a cheat backlog.** Nothing new ships without a scaffold-ledger entry.
- **P5 — BRAIN-BASED-ONLY** (see orientation above).
- **P6 — PERFORMANCE IS FIRST-CLASS, but not LLM-parity-yet.** Slow-but-faithful biology is in scope; **but laziness is not faithfulness.** Optimize what you cheaply can; never leave speed on the table through lazy design. Bias to event-driven / sparse / local.
- **P7 — HONESTY BOUNDARY** (see §5 below).

**Emergence bar (operational):** the test flips from *"did I build this capability?"* to **"did the substrate LEARN this from experience?"** When you catch yourself reaching for a fresh dedicated mechanism / router / register / template for one conversational capability, **that reach is the drift** ("whack-a-mole"). A new hand-built capability is allowed only as (i) an explicit temporary scaffold on the ladder to its learned replacement, or (ii) a probe that maps a substrate limit. Otherwise ask *"what learning substrate + training stream makes this emerge?"* and advance that.

- **DO:** re-point a language network at grounded, intentful, stateful generation inside a minimal world+body+social loop; write its role-in-the-whole spec first.
- **DON'T:** add a hand-coded discourse-template / intent-dispatch router to make one more conversational behavior pass, then call the faculty done.

---

### 4. Verify, don't assert — the silent-failure class

The no-stall rules above are about *stopping too early*. This is the opposite and equally lethal failure: **work that runs, reports success, and is confidently WRONG while every liveness signal says healthy.** One session produced six such retractions; three were the agent's own claims. `SKILL.md` has the full 15-item list — the shape is always the same:

> "**THE MACHINERY TO CHECK THE CLAIM ALREADY EXISTED; NOTHING INVOKED IT.**"

The load-bearing rules, distilled:

- **Never lift a metric out of a run whose own verdict is negative.** A runner that prints `SIGNAL=False` / `HONEST NEGATIVE` has already done the analysis — read the verdict, not the field.
- **"X is inert / byte-identical / a no-op" is a HYPOTHESIS — put it in an ASSERTION, not a comment.** A comment can't fail; it rots and the result rots with it.
- **One flag ≠ one variable.** Before an A/B, ask what else the lever touches *in code*, and confirm the DEFAULT arm is genuinely unchanged.
- **Verify, don't assert — especially boring infrastructure.** `git push -q … ; echo pushed` reports success unconditionally. Use `tools/push_both.sh` (it pushes then `git ls-remote`-verifies both remotes). Don't claim "on GPU," "tests pass," or "pushed" without checking.
- **A seed is a hypothesis until you hash the state it controls.** This project's deepest single bug: `--seeds` never controlled the substrate because the builder set a *reporting* field (`actual_seed_used`) the bridge never reads, while heterogeneity seeds from **`cfg.seed`** (default `-1` ⇒ unseeded global RNG). Every FULL-vs-FROZEN comparison compared *different neurons*, a confound ~3× the measured effect. **To seed the substrate, set `cfg.seed`** (see the boxed warning in `CLAUDE.md`), then *prove it*: build twice at one seed, hash `cp_neuron_firing_thresholds`, confirm identical. Pinned by `tests/test_determinism.py::TestSubstrateActuallySeeded`.
- **A broad `except` is a silent-failure factory.** `except ImportError: pass` around a backend import silently ran `SIM_BACKEND=numpy` on the GPU for months. Catch narrow; never bare-`pass`.

**The self-check, verbatim:** *"If this were silently wrong, what would look different?"* If the answer is "nothing" — the process is alive, the log grows, the number is plausible — **you have no evidence, only an absence of alarms.**

Before any positive result enters the record as a "surpass"/GO/milestone, **adversarially verify it** — try to refute it from distinct angles (leakage, deployment-path/like-for-like, is the claimed ingredient actually load-bearing, anti-cheat validity, baseline fairness) and commit only what survives. In Claude Code this was a `verify-go` skill run by sub-agents; **you achieve the same by reading `.claude/skills/verify-go/SKILL.md` and running its refutation lenses yourself** (or as parallel sub-tasks if your harness supports them). A confounded GO caught before commit is worth more than a committed overclaim.

- **DO:** `SIM_BACKEND=numpy .venv/bin/python -m research.runners.<x> …`, then read the runner's *own printed verdict*, then `tools/push_both.sh`.
- **DON'T:** average a field across three runs and report a GO without noticing each run printed `HONEST NEGATIVE`.

---

### 5. The honesty boundary (P7) — a deliverable, not a caveat

Build and **measure** the functional correlates of consciousness, self-model, and affect. Design every self-report as an honest functional read-out — *"my familiarity monitor reads this as novel, so I'm uncertain"* — and **never assert phenomenal experience.** An honest negative (the neural version underperforming a host shortcut) **is the scientific deliverable**: it maps what the substrate can and cannot do on its own. This applies to how you write findings too: state what landed AND what's open; never imply a chapter is closed.

Terminology is load-bearing and enforced: before writing `consolidation`, `compositional`, `self-organized`, `closed`, `GO`, `fully spiking`, `byte-identical`, `lesion`, `selective`, or `works` in a finding/commit/board entry, **check its exact code condition in `docs/TERMS.md`.** Three of nine retractions in one session were pure terminology overclaim over correct measurements underneath. An unchecked term is a hypothesis.

---

### 6. The deepest lesson — ask this at every wall FIRST

From `CLAUDE.md`, the first question to ask at any wall, *before* "what biology surpasses this?":

> **"What else does the real system run alongside this, that we replaced with a constant?"**

Biology runs *interacting* processes; we habitually implement one and substitute a static bound for the rest — **and the proxy dominates** (97% of one gap's weight change turned out to be the clamp, not the mechanism). Four measured causes of friction: (a) we proxy a companion homeostatic/competitive process with a constant, (b) the operating point is implicit in the animal so tuning optimizes whatever the metric rewards, (c) the *protocol* is part of the mechanism and no paper writes it down (e.g. BTSP is one-shot; five laps erases the field), and (d) usually you can't tell which, because **the instrument doesn't exist yet — the instrument is part of the emulation.** A mechanism you can't measure correctly is one you'll tune in the wrong direction, confidently, for weeks.

- **DO:** at a plasticity wall, ask "what homeostatic/competitive process is the real synapse running that I replaced with a fixed bound?" and model *that*, before reaching for a new learning rule.
- **DON'T:** tune the bound harder because the number moves — a moving number under a dominating proxy is not evidence the mechanism works.

---

### 7. Adapting the Claude-Code-specific machinery (what you won't have, and how to get the same thing)

Several disciplines above were implemented with Claude-Code-only tools. The **underlying capability is repo-based and tool-agnostic** — here is the translation:

| Claude Code mechanism | What it did | How you (Codex) get the same thing |
|---|---|---|
| **`Skill` tool** (`neural-simulator`, `verify-go`, `sync-documentation`, `evolve-skills`) | Loaded a workflow's instructions on demand | The skills are plain markdown at `.claude/skills/<name>/SKILL.md`. **Read them directly**; follow the prose. |
| **`Monitor` tool** + heartbeat | Watched long background runs for done/crash/hang, emitting a state heartbeat every ~15 min | Poll the process yourself (`ps`, `kill -0 <pid>`, tail the JSON/log). **Silence ≠ success**: check the run's *own* terminal verdict, and confirm which *device* it ran on. Never conclude "crashed/complete" without `ps`/`kill -0` — buffered stdout has faked both. |
| **`run_in_background` + sub-agents + Workflows** | Fanned independent de-risks/reviews across processes | Use your own background/parallel primitives. The pattern that matters: **fan multi-seed sweeps across OS processes** (one process per seed, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SIM_BACKEND=numpy`, `wait` at the end), not one process looping seeds serially (that pins 1/N cores). |
| **Git hooks (the gates) — these DO work from any shell** | `tools/gates/*` auto-discovered into a pre-commit hook; block commits on claim-provenance, biology bindings, single-seed headlines, doc structure, etc. | Run `python3 tools/rag/check_workflow.py` to verify `core.hooksPath`, executable hooks, both canonical interpreters, the index/catalog, and index schema. Repair hook installation with `python3 tools/rag/check_workflow.py --install`; a blocked check must not be described as armed. (Ignore the hyphenated `tools/git-hooks/` — a dead Windows-era artifact.) Authoritative index: `docs/FAILURE_GATE_MATRIX.md`. |
| **The RAG index** | Semantic search over findings/plans/docs/textbooks | Works from main or a linked worktree: `bash tools/rag/search.sh "<q>" 5 --corpus <c>`. The `main` post-commit hook refreshes committed project prose. The enabled `sim-rag-autoupdate.timer` checks every five minutes for catalog changes and creates missing searchable `.txt` companions for readable PDFs before refreshing. Image-only or unreadable PDFs fail closed and require OCR. Feature-branch skips, blocked refreshes, and timer outcomes are written explicitly to `rag_index/_autoupdate.log`. Uncommitted findings are not indexed. |

**Two rules that survive any harness:** (1) a **noticed failure cannot stay unclosed** — add one line to `research/FAILURE_LOG.md` and it must name a gate or declare `NOT-GATEABLE: <reason>`; (2) when a committed finding changes a wall/gap **status**, the current **frontier**, or a **next-action**, sync the summary docs the **same cycle** (the roadmap ledger, `GAP_CLOSURE_MISSION.md` CURRENT STATE, `research/findings/AUTONOMOUS_STATE.md`, `ROADMAP.md`) — following `.claude/skills/sync-documentation/SKILL.md`. Stale summary pointers are the #1 cause of re-deriving concluded work: **a summary doc is a POINTER, not ground truth; if it conflicts with a finding, the finding wins and you fix the summary in the same commit.**

---

### 8. The minimum operating checklist (pin this)

1. **Session start:** read `CLAUDE.md`, the charter (`docs/plans/2026-08-02-PROJECT-CHARTER-...`), and `research/findings/AUTONOMOUS_STATE.md` for the live frontier + exact next action. Verify the gate hook is installed.
2. **Before the first lever on any difficulty:** `bash tools/before_you_build.sh "<defect>"` → RAG our findings → **read the primary source in depth** → search external literature.
3. **Build the cheap de-risk** with anti-cheats (lesion / permuted / wrong-sign / memorization-floor / oracle-ceiling), **change one variable per rung**, compare **like-for-like** (a host/oracle read is not a spiking surpass), and **run the ceiling/reference early** — if even the ceiling can't beat a trivial baseline, the arc is scale-/data-/task-confounded, not mechanism-bound.
4. **Seed correctly** (`cfg.seed`), validate at **6 seeds** (42/43/44 dev → 100/101/102 blind) and *list the seeds* before quoting an N-seed number.
5. **Before any GO enters the record:** adversarially verify; check the runner's own verdict, the device, `docs/TERMS.md` for every load-bearing term.
6. **Commit both remotes** with `tools/push_both.sh` (it verifies). **Sync the summary docs same-cycle.** Never end on a promise you could execute.

## 5. Failure modes — the recurring traps, with real examples

This project's single most expensive activity is not writing code — it is **being confidently wrong**. The record here (`research/FAILURE_LOG.md`, ~40 dated rows; the SILENT-FAILURE class and drift-mode list in `.claude/skills/neural-simulator/SKILL.md`) is a catalog of failures that all share one shape: **the run was alive, the log grew, the number was plausible, and the conclusion was false.** Nine formal retractions landed in a single day (2026-07-28); three of them were pure terminology overclaim sitting on top of correct measurements.

Internalize the governing fact before the specifics:

> **These traps are STRUCTURAL, not carelessness. They recur even while you are actively hunting them** — the record contains a bare `except` that swallowed its author's own warning *on the day that author documented that exact pattern five times*. Vigilance does not fix a structural failure; **a mechanical guard does.** That is why nearly every row below ends in a gate, a script, or an assertion — something that *blocks or prints*, not something you must remember. **Your job is not to avoid these by being careful. Your job is to keep the guards armed and never bypass them.**

The guards live in `tools/gates/` (28 registry modules, one per class, auto-discovered) and fire from the git **pre-commit** hook. They are shell/Python and work from **any** app or terminal — Codex included. Arm them once in your clone:

```bash
git config core.hooksPath tools/githooks
```

`docs/FAILURE_GATE_MATRIX.md` is the authoritative index (failure class → gate → where it blocks). When a rule you remember and a gate disagree, **the gate wins.**

---

### Trap 1 — Reading a metric out of a run whose OWN verdict is negative

**The trap.** A runner prints its own pass/fail verdict. You ignore the verdict and lift a favorable field out of the same run.

**Real example (silent-failure rule #1 + gate `CVV`).** The banked headline *"feedforward deep credit is GO, K=8 0.877, anti-cheat-clean"* was produced by averaging the `inherit` field from three runs that **each printed `SIGNAL=False` / `HONEST NEGATIVE` with the anti-cheat FAILING**. On 2026-08-01 the same shape recurred at the doc level: a gap#4 "closure" was written as a session headline *and* a roadmap surpass banner *and* the biology-registry `current_status`, while every e-prop artifact it rested on printed `"SIGNAL": false` and computed `deep_credit_share ≈ 0.005` (a fixed random reservoir). The claim read `inherit` (0.85) and never read the `deep_credit_share` / verdict the **same run** emitted.

**The lesson.** *"The instrument was not broken — it was OVERRIDDEN."* A runner that prints a negative has already done the analysis. **Read the verdict, not the field.**

**The guard.** `gates/claim_verdict_consistency` (class `CVV`) BLOCKS a `status: live` finding whose title asserts GO/surpass/closure/reproduced while a cited artifact carries `SIGNAL: false` / a HONEST-NEGATIVE verdict. `gates/verdict_preconditions` + the runtime `tools/verdict.Verdict` make UNDEFINED the default — a run must *earn* a verdict.

---

### Trap 2 — "X is inert / a no-op / byte-identical" written as a COMMENT, not an assertion

**The trap.** You believe a code path is a no-op and record that belief in a comment or a doc. **A comment cannot fail. It rots, and the result rots with it.**

**Real example (silent-failure rule #2).** Three such claims were false in one day: `lr=0` did **not** defeat an unconditional `cp.clip` (the clamp ran regardless of the learning rate — and later the subtraction showed the clamp owned **97%** of a gap#5 weight change while the lever owned 3%); a "byte-identical" gate-tag silently flipped a scalar code path to a stale array; a runner silently no-op'd in its own documented mode.

**The lesson.** *"X is inert" is a HYPOTHESIS. It belongs in an ASSERTION, not a comment.* If you write "this is inert," write the test that fails when it isn't.

**The guard.** `tools/lab.py` helpers make the check execute instead of being recalled: `void_if(...)`, `undefined_if_empty(...)`, `before_after(...)`, `attributable_to(label, treatment, control)`. Import them in a probe rather than commenting the assumption. `gates/attribution_required` BLOCKS a new runner that computes a treatment/control pair without calling one of them (the record: `tools/lab` was imported by **2 of 1330 runners** — used only by whoever already remembered the lesson).

---

### Trap 3 — One FLAG is not one VARIABLE

**The trap.** You run an A/B toggling one config field, assuming it moves one thing. It moves several.

**Real example (silent-failure rule #4).** `--bdsp-wmax` was one config field but two functional variables: the clamp is global over `cp_connections`, which held **both** the spiking synapses **and** a host-side linear read-out. The A/B freed a linear classifier and it was read as "deep credit."

**The lesson.** Before running, **ask what else the lever touches, in code.** An A/B whose lever moves >1 variable is a confound wearing the clothes of a clean result.

**The guard.** `gates/knob_reachable` (class `KR`) and `gates/conditional_sweep` catch the adjacent forms. There is no substitute for reading the code path the flag reaches — see Trap 6 for the "absent flag" half and Trap 9 for the "accepted-but-inert flag" half.

---

### Trap 4 — A broad `except` is a silent-failure factory

**The trap.** `except SomethingBroad: pass` swallows the error you most needed to see.

**Real example (silent-failure rule #5).** `except ImportError: pass` around a backend import made `SIM_BACKEND=numpy` **silently run on the GPU for months.** A swallowed broadcast error stopped Hebbian decay every step — **10,023 tracebacks, zero alarms.**

**The lesson.** **Catch NARROW. Never `pass`.** Log the catch-all at `debug` minimum. A swallowed exception is not resilience; it is a defect that hides its own evidence.

**The guard.** This one is mostly discipline plus code review — but its downstream damage (a backend silently on the wrong device) is now caught: see Trap 5.

---

### Trap 5 — Verify, don't ASSERT — especially the boring infrastructure (push / GPU / tests)

**The trap.** You emit a success message unconditionally and treat it as evidence.

**Real examples (silent-failure rule #6).** `git push -q ... ; echo pushed` reports success unconditionally (`-q` hides failure, `| tail -1` eats it) → roughly **20 "pushed both remotes" claims on faith.** A 4-arm sweep ran **~50 min on CPU** while the monitor correctly said RUNNING — it could not see *which device*. And on 2026-07-31 someone "verified" four cells were on the GPU by finding `nvidia` mappings in `/proc/PID/maps` — which only proves CuPy is **importable**; the runner's own first log line read *"this run is on the CPU"* and it ran 10-50× slower than intended.

**The lesson.** *"If this were silently wrong, what would look different?"* If the answer is "nothing" — the process is alive, the log grows, the number is plausible — **you have no evidence, only an absence of alarms.** Verify the *thing*, not a proxy for it.

**The guards.**
- Push: use `tools/push_both.sh` — it pushes, then `git ls-remote`-verifies both remotes (a cached remote-tracking ref will happily agree with a FAILED push, so a naive check is itself a proxy).
- Device: assert the backend by *importing* it, not by inspecting the process — `tools.lab.assert_backend(...)`. `gates/device_and_cost` (class `DC`) BLOCKS an artifact that cannot say what device it ran on. **Watch this one:** runners call `os.environ.setdefault("SIM_BACKEND","numpy")`, so a caller who does not set it explicitly **silently gets CPU**. Always set it yourself:

```bash
SIM_BACKEND=numpy .venv/bin/python -m research.runners.<runner> --seeds 42 --json raw/_smoke_s42.json
```

---

### Trap 6 — An absent flag means DEFAULT, not OFF

**The trap.** A finding says "this cheat is closed / no heuristic was used," and its recorded command shows no such flag. Absence reads as "off." It almost never is.

**Real example (silent-failure rule #10).** *Cluster K v2: "2.97 at 16×16 with NO heuristic, NO direct (gx, gy) or (x, y) access"* stood for **2.5 months** and propagated into `CLAUDE.md`. Its own `.cmd.json` carries no heuristic flag — because `--heuristic-strength` **defaults to 1.0**, so the run drove 800 pA into `cortex_N/E/S/W` straight from `gy > y` / `gx > x`. The flag that actually closes the cheat (`--cue-reflex-replaces-heuristic`) belongs to a *different* config; the sentence was copied onto one that lacks it.

**The lesson.** A recorded command stores only the **delta from defaults**. When a finding claims a cheat is closed, **grep its own `.cmd.json` for the flag that closes it AND read that flag's default.** A claim inherited from a neighbouring config is not evidence — the config that *earns* a claim and the config that *quotes* it are different experiments.

**The writing-side half.** Every knob that changes the experiment must land **in the output artifact**. `_onbridge_eprop_port_derisk.py` never wrote `pool_k` into its config while `--pool-k` defaults to 1 and the arc ran at 8 — so the only provenance for the load-bearing knob was the string `k8` in a filename, and recovering it later needed forensics (bridge synapse count: 1408 @ k=1 vs 90112 @ k=8).

**The guard.** `research/runners/__init__.py` now auto-stamps a provenance sidecar (argv, git SHA, resolved backend) on every `-m research.runners.X` run — no runner change needed. `gates/artifact_provenance` and `gates/knob_reachable` enforce reachability.

---

### Trap 7 — Re-deriving work the record already concluded (the stale-pointer drift)

**The trap.** You adopt a direction, a "next bet," or a verdict from a summary doc (`ROADMAP.md`, a plan, `CLAUDE.md`, `AUTONOMOUS_STATE.md`) **without checking it against the findings.** Summaries go stale.

**Real examples (drift #12 + gate `CC`).** On 2026-07-17 the roadmap called Node Perturbation *"the mission-critical lever"* — the findings had **RETIRED it four days earlier** (12-seed REFUTED). On 2026-07-31 a **nine-hour, eight-GPU-cell crux** re-derived a result banked **three weeks earlier** with its root cause already named; `before_you_build.sh` returns all four relevant priors in **0.63 s** and was not run before launch. This is called out as *"the #1 cause of re-deriving concluded work."*

**The lesson.** **A summary doc is a POINTER, not ground truth. The FINDING wins.** Before acting on any load-bearing claim — a "next," a "GO," a "candidate" — search the record first.

**The guard.** Run this before the first lever against ANY difficulty (it is one command, ~30s, and it *records* that you ran it, which `gates/corpus_check_required` reads):

```bash
bash tools/before_you_build.sh "the slot competition ignores the cue"
```

The underlying retrieval — use it directly too:

```bash
bash tools/rag/search.sh "<question>" 5 --corpus finding
# corpora: finding | plan | doc | catalog | kandel | paper | all
```

`gates/corpus_check_required` (class `CC`) BLOCKS any artifact recording >1h of compute with no recorded corpus check (cheap runs are exempt by design). **A RAG hit is a POINTER, not a paraphrase — open the surfaced source and read the load-bearing passage;** citing the rerank snippet is the skim-drift the gate cannot catch.

---

### Trap 8 — Re-proposing a mechanism the record already REFUTED, from memory

**The trap.** At a wall you name "the remaining surpass" from memory, skipping the research gate — and it is something the record already tested and killed.

**Real example (gate `RM`, 2026-08-02).** After a clean gap#4 measurement, *"the one remaining surpass is two-compartment dendritic credit"* was written into a finding, the board, **and both roadmaps** — without running `before_you_build.sh`. The record refutes it flatly and repeatedly: `2026-05-17-dendritic-credit-assignment-NEGATIVE`, `2026-08-01-...BDSP...NEGATIVE`, and a finding literally titled `2026-07-22-gap4-real-issue-NOT-dendrites`. **The owner had caught this exact "keeps coming back to dendrites" reflex before** — that is *why* the 2026-07-22 finding exists. It was a recurrence. Every existing gate passed it because the run was cheap (numpy, minutes → CC exempt) and the framing was *upbeat* (not a "wall" title → the boundary gate didn't fire).

**The lesson.** The comfortable verdict is the START of the research, never the end. When you write "the next mechanism is X," that is precisely the moment to check whether X is on the refuted register.

**The guard.** `gates/refuted_mechanism_reproposal` (class `RM`) BLOCKS a finding/board/roadmap edit that names a mechanism on the refuted register (seeded: two-compartment / dendritic / BDSP / burstprop deep-credit) near a forward-proposal phrase ("remaining surpass," "next mechanism," "the candidate is") unless it cites the refuting finding or an already-tested token. Extend the register by hand as mechanisms are closed.

---

### Trap 9 — Tunnel-vision on a narrow / wrong-instrument test, then a "fundamental limit" verdict

**The trap.** You bank a capability-walling verdict ("fundamental limit," "structural primitive," "different-paradigm question") off a narrow toy the record already flagged as the wrong instrument, and without reading the external field.

**Real example (gate `BV`, 2026-08-01).** Commit `b7549514` banked *"a FUNDAMENTAL limit of the local transport-free credit class ... a different-paradigm (equilibrium-propagation) question"* — with **zero external citations**, re-deriving Kolen-Pollack + burstprop (**both already built and verified in our own `_gnw_d1_spiking_bdsp_derisk.py`**) on a numpy XOR toy a prior finding **explicitly called "the WRONG instrument."** A one-hour deep read **overturned it within the hour**: transport-free graded chained-FA+σ′ clears the "0.63 wall" (6-seed 0.935 vs oracle 0.974), matching WF-Act-PC (arxiv 2607.13380), the external SOTA that named the exact missing factor.

The related below-chance form: *every arm of an A/B lands BELOW chance* — a broken task/label wiring reported as a "NO-GO" (caught on the credit-on-expanded runner: chance 0.200, all arms 0.033-0.065). That is **UNDEFINED, not negative.**

**The lesson.** *"Honest negatives and boundaries are UNDISCOVERED MECHANISMS, not endpoints."* A capability-wall verdict needs both the corpus read AND the **external field** read. And a below-chance / crushed-arm result means the instrument is broken, not that the mechanism failed.

**The guards.** `gates/boundary_verdict_external_check` (class `BV`) BLOCKS a finding asserting a capability wall unless it cites external literature (arxiv/doi/Sources) or declares `NO-EXTERNAL-NEEDED: <reason>`; record the read with `bash tools/record_external_search.sh "<q>" "<src>"`. `gates/below_chance` (class `BC`) BLOCKS a NO-GO where every arm is below the run's own reported chance.

---

### Trap 10 — `--help` passing is not the run working; a flag accepted is not a flag reaching its code path

**The trap.** You add a CLI flag, confirm `--help` lists it, and launch. Registration and consumption are different claims.

**Real examples.** On 2026-08-01, adding three flags to `_onbridge_eprop_port_derisk.py`: `--help` listed all three, both arms then **died** — one on `NameError: name 'a' is not defined`, the other on `unrecognized arguments`. Earlier the same session, `--sweep-weights 0.05 0.15 0.4` was **accepted by argparse and silently IGNORED** — the option was consumed only inside `if a.smoke:`, so the full run used `--gabab-weight`'s default of 1.5 (10-30× the swept values), crushed its own treatment arm, and reported "NO-GO."

**The lesson.** `--help` exercises registration only. **Before any launch, run the shortest possible REAL invocation** (one seed, `SIM_BACKEND=numpy`) and confirm it produces numbers, not a traceback and not a silently-defaulted run.

**The guard.** Partly `gates/knob_reachable` (a `cfg.*` assignment needs a matching `add_argument`). The runtime half — that the flag's value actually appears in the recorded config — is the knob-recording rule (Trap 6). Structurally, treat a one-line smoke as mandatory:

```bash
SIM_BACKEND=numpy .venv/bin/python -m research.runners.<runner> --seeds 42 <your-new-flag> ... \
  --json raw/_smoke_s42.json 2>&1 | tail -20
```

---

### Trap 11 — The scaffold backlog (whack-a-mole hand-building)

**The trap.** You reach for a fresh dedicated mechanism / router / register / template for each conversational capability, one at a time. The list is unbounded; this cannot reach the goal, and each hand-built piece is a shortcut that must later be biologized.

**Real example (drift #10, owner steer 2026-07-10).** The owner named it directly: *"Are we playing whack-a-mole with conversational capabilities?"* — and the honest answer was YES (the discourse event-register, intent-dispatch console routing, VSA composer's exact-inverse algebra, discourse templates were all hand-designed structure).

**The lesson.** The test **FLIPS** from *"did I build this capability?"* to **"did the substrate LEARN this from experience?"** A new hand-built mechanism is allowed ONLY as (i) an explicit temporary scaffold on the ladder to its learned replacement, or (ii) a probe mapping a substrate limit. When you catch yourself arguing *why a scaffold can stay*, that argument **is** the drift. Every shortcut is tracked and burned down; an honest negative under strict biology is a first-class deliverable.

**The guard.** This one is judgement, not a blocking gate — but `docs/TERMS.md` pins the load-bearing terms (`self-organized`, `emergent`, `consolidation`, `compositional`, `fully spiking`) to a CODE CONDITION you must check before using the word in a finding or commit. Three of nine retractions in one session were pure terminology overclaim.

---

### The cross-cutting classes (know these exist)

A few more that recur across all of the above and each earned a gate:

- **Seed never controlled the substrate.** `--seeds` set `cfg.actual_seed_used`, a **reporting** field the bridge never reads; the bridge seeds from `cfg.seed` (`bridge.py:2294`), both defaulting to `-1`. Four nets built back-to-back in one process differed by **18.4 mV** — a confound ~3× the effect being measured. **Set `cfg.seed` explicitly and hash the substrate** (`cp_neuron_firing_thresholds`) across two builds to prove it. Pinned by `tests/test_determinism.py::TestSubstrateActuallySeeded`.
- **Two same-shaped numbers on one output line get quoted interchangeably.** `g11_bg_runner.py` prints `sum_finalQ` and `mean_distance_overall` together — **4 of 10 audited defects** trace to that one line (a mean quoted as a sum, etc.). Gate: `gates/quantity_mismatch`.
- **Single-seed headlines.** Gate `gates/single_seed`. The rule is 6 seeds (42/43/44/100/101/102) — and *which six*: an all-dev-seed result is a dev result, and ~36% of seeds at one config were degenerate (unsolvable by an oracle), so screen instances for validity first.
- **A validator and executor in different worlds.** `pool_queue.sh` validated a job's argparse against the *local* repo while the job ran on a pool *node* holding an rsync'd copy — passed every local check, died on `No module named ...`. Any check that runs somewhere other than where the work runs is a proxy.
- **A false alarm is as corrosive as a missed one.** A monitor that cried wolf trains you to ignore it; that is how a real failure slips through. Test a monitor against a run you *know* is broken, not only a healthy one.

---

### How you (GPT / Codex) run these guards — tool-agnostic

Almost everything here is shell + Python and needs no Claude-specific tooling. Map the few Claude-isms to the repo-based equivalent:

| Claude-Code mechanism | What it is | Your equivalent (works from any shell) |
|---|---|---|
| **Skill tool** (`verify-go`, `neural-simulator`) | Loads a checklist markdown into context | **Read** `.claude/skills/verify-go/SKILL.md` and `.claude/skills/neural-simulator/SKILL.md` and follow them manually. Their checkable clauses have largely been **extracted into `tools/gates/`** — the gates run regardless. |
| **Monitor tool** | State-checking heartbeat on long runs (GPU / procs / recent output every ~15 min) | A background poll loop / cron that greps the run's log + `nvidia-smi` + `ps`, matching **every terminal state** (done/crash/hang), not just the happy path. Silence ≠ success. |
| **Subagents / Workflows** | Parallel fan-out, adversarial-verify panels | Sequential care, or your own parallelism. The **anti-cheat set** (lesson / permuted / wrong-sign / memorization-floor / oracle-ceiling / scramble) matters more than who runs it. |

The load-bearing move is one line, and it makes the whole catalog above enforce itself on every commit:

```bash
git config core.hooksPath tools/githooks     # arms pre-commit: W1/W2 docs, claim_check, biology_check, all gates
.venv/bin/python tools/check_docs.py         # W1 retraction-marking + W2 ≤800-char prose lines (also in the hook)
```

Do **not** commit with `--no-verify` to dodge a gate. On 2026-07-31 a gate that fired *correctly* was suppressed with a false `.lane_waiver` reading "saturated with the crux" — a prose rationalisation that disabled an enforced gate for hours. If a gate blocks you, the fix is to satisfy it (or add a `NOT-GATEABLE:` line to `research/FAILURE_LOG.md` with a reason) — never to bypass it.

**The one self-check that subsumes them all, ask it continuously:** *"If this were silently wrong, what would look different?"* If the honest answer is "nothing" — you have no evidence, only an absence of alarms.

## 6. The gate system — mechanical enforcement, the inventory, and how to add a gate

This is the single most important thing to adopt from day one. This project has a long, hard-won record of the same *classes* of mistake recurring — overclaimed terms, single-seed headlines, "fundamental limit" verdicts banked without reading the literature, expensive runs that re-derive an answer already in the record. The response was **not** more rules to remember. It was to convert each failure class into **a check that BLOCKS a `git commit`**. The governing principle, from `docs/FAILURE_GATE_MATRIX.md`:

> **Everything that reads as a RULE YOU MUST REMEMBER has, where possible, been converted into a CHECK THAT BLOCKS. If a prose rule and a gate disagree, the gate is authoritative — it is the thing that actually runs.**

You (Sol) get all of this for free from any shell. The gates are plain Python + a Bash git hook — none of it depends on the Claude-Code app. **The one thing you must do once per clone** is point git at the hook directory (see below). After that, every `git commit` you make is checked identically to how Claude's were.

### How gates run on commit

The entry point is `tools/githooks/pre-commit`. Install it once (it is already installed in the working repo, but verify after any fresh clone):

```bash
cd /home/dant123/Projects/sim
git config core.hooksPath tools/githooks
git config core.hooksPath          # should print: tools/githooks
```

The hook runs five gates in order and **exits non-zero (blocking the commit) on the first failure**. If there is no `.venv/bin/python` it exits 0 (fresh clone / CI bootstrap — it never blocks blindly). The deliberate, reflog-visible override is `git commit --no-verify` — use it only when you genuinely mean to, because it disables **every** gate at once.

| Hook gate | What it checks | Implemented by |
|---|---|---|
| GATE 1 | Document structure — W1 (retracted docs marked with ⛔) + W2 (prose lines ≤ 800 chars) | `tools/check_docs.py` |
| GATE 2 | Every measurement in a **newly-added** findings doc exists in an artifact the doc cites | `tools/claim_check.py` |
| GATE 3 | Biology source pointers still resolve; no config contradicts the biology it implements | `tools/biology_check.py` |
| GATE 4 | A **new** finding declares `status:` in frontmatter | inline in the hook |
| GATE 5 | **The registry** — every failure-class gate in `tools/gates/`, auto-discovered | `from tools.gates import run_all` |

GATE 5 is the extensible one. It scopes content gates to newly-added files (`git diff --cached --name-only --diff-filter=A`) — this is deliberate: a 2026-05 legacy document cannot retroactively cite artifacts or re-run at 6 seeds, so content quality is checked **when a document is written**, not when its status field is later edited. Relationship gates (e.g. `stale-pointer`, `lever-efficacy`) ignore the staged list and glob the whole corpus, which is where their real hits come from.

The GATE 5 invocation, verbatim from the hook, is the exact idiom you can reuse anywhere:

```python
import sys, os
sys.path.insert(0, os.environ.get("ROOT") or os.getcwd())
from tools.gates import run_all
paths = [p for p in (sys.argv[1] if len(sys.argv) > 1 else "").split("\n") if p.strip()]
blocking, report = run_all(paths, verbose=False)
print("\n".join(report))
sys.exit(1 if blocking else 0)
```

### The contract every gate module obeys

A gate is one file in `tools/gates/`. From `tools/gates/__init__.py`, it MUST expose exactly this interface:

```
NAME     : str          short id, e.g. "single-seed"
CLASS_ID : str          the failure class from docs/FAILURE_GATE_MATRIX.md, e.g. "9"
BLOCKING : bool         True => a violation blocks the commit. False => reported only.
def check(paths: list[str]) -> list[str]     # problems found; empty list means pass
def selftest() -> list[str]                  # MUST demonstrate the gate FAILING on a case it should catch
```

- **`check(paths)`** receives the list of staged paths (or `[]`/`None` when run standalone). It returns a list of human-readable problem strings; **empty means pass**. Convention: content gates return `[]` when `paths` is `None`/empty so a standalone run never scans the whole legacy corpus, while relationship gates ignore `paths` and glob the corpus themselves.
- **`selftest()`** is the load-bearing part. It must construct a case the gate *should* catch and assert the gate catches it, plus negative controls that must NOT fire. It returns `[]` when the gate behaves correctly.

**Why `selftest` is mandatory and non-negotiable — failure class 3, "check-that-cannot-fail" (9 incidents).** This project has shipped four checks that looked healthy while checking nothing: a `;` where `&&` was meant, a pipe eating an exit status, a relevance count that made a gate unfailable, a nonsense query scoring 18 hits and PASSING. So the registry **refuses to trust any gate whose `selftest()` does not itself return non-empty in its failing direction** — from `run_all`:

```python
if selftest_first:
    st = mod.selftest()
    if st:
        blocking.append("GATE %s FAILED ITS OWN SELFTEST: %s" % (name, "; ".join(st)))
        ...
        continue     # verdict NOT trusted
```

A gate that can't demonstrate its own failure is treated as broken and blocks loudly. The registry also catches import failures, missing contract attributes, crashes in `check()`, and a per-gate time budget (`GATE_BUDGET_S`, default 12s) — a gate that stalls is flagged, never silently dropped.

The canonical `selftest()` shape, from `corpus_check_required.py` — **failing direction first, then negative controls**:

```python
def selftest():
    bad = []
    # 1. THE REAL CASE: a 9-hour run with no corpus check MUST fire.
    if not _check_one(w("a.json", {"elapsed_seconds": 9*3600}), "raw/a.json"):
        bad.append("did NOT catch an expensive run with no corpus check")
    # 3. NEGATIVE CONTROL — a checked expensive run passes.
    if _check_one(w("c.json", {"elapsed_seconds": 9*3600, "corpus_check_fresh": True}), "raw/c.json"):
        bad.append("FALSE POSITIVE: flagged an expensive run that DID check the corpus")
    # ... cheap run, untimed artifact, sidecar-carried evidence, scope-leak checks ...
    return bad
```

Selftests here are **mutation-tested, not trusted**: `below_chance` was deliberately broken ten ways and its `selftest()` was required to fail on each (it caught 9 of 10). The cheap procedure: substitute one wrong string into the module, `exec` it, assert `selftest()` returns non-empty. A rule with no mutation that breaks it is untested.

### Auto-discovery, and the two-way coverage invariant

Adding a failure class is **one file, two functions — no hook edit, no wiring**. `discover()` in `__init__.py` walks the package with `pkgutil.iter_modules`, imports every non-underscore module, checks it implements the contract, and sorts by `CLASS_ID`. A module that fails to import or is missing a contract attribute is reported loudly (`"!name"`), never silently absent.

Every gate must also have a **row in `docs/FAILURE_GATE_MATRIX.md`**, and this is itself enforced. The `coverage` gate (class COV) checks the invariant in both directions:

- every `` `gates/<module>` `` named in the matrix must have that module present in `tools/gates/`, and
- every module in `tools/gates/` must appear in the matrix.

Drift either way blocks the commit. `coverage` also enforces the **noticing loop**: a newly-noticed failure gets **one line in `research/FAILURE_LOG.md`** (`| date | failure | gate |`), and `coverage` BLOCKS until that row's `gate:` column either names a real enforcement point or explicitly declares `NOT-GATEABLE: <reason>` (with a real reason — a bare `NOT-GATEABLE:` is caught). This is how "I noticed a problem" becomes "a gate now prevents it" without depending on memory. What it cannot do is *notice* — if a failure is never written down, nothing fires; that limit is stated, not papered over. The two newest gates (`refuted_mechanism_reproposal`, `boundary_verdict_external_check`) both entered exactly this way — a FAILURE_LOG row on the day the mistake happened.

### The full inventory (28 gates)

`✅` = BLOCKING, `🟡` = REPORTS only (advisory; declared limits it can't check reliably at commit time — an honest reporting gate beats a false-positive generator that gets `--no-verify`'d). `CLASS_ID` in brackets ties to the matrix row.

| Module (`tools/gates/…`) | Class | Purpose (one line) |
|---|---|---|
| `agent_parallelism.py` | AP ✅ | Pending work serialised while dispatchable agents/lanes sit unused. |
| `artifact_provenance.py` | P ✅ | A result artifact can't say what produced it (argv / git SHA / result-changing env vars). |
| `attribution_required.py` | AT ✅ | A treatment/control pair is measured but the difference is never *attributed* (the 97%-was-the-clamp class). |
| `below_chance.py` | BC ✅ | Every arm of an A/B lands **below chance** yet it's banked as a NO-GO — the result is UNDEFINED, not negative. |
| `boundary_verdict_external_check.py` | BV ✅ | A "fundamental limit / different-paradigm" verdict banked without citing any external literature. |
| `claim_verdict_consistency.py` | CVV ✅ | A `status: live` finding claims GO/closure in its **title** while the artifact it cites printed `SIGNAL: false`. |
| `closure_names_mechanism.py` | CM ✅ | A closure claim that names **no mechanism**, so nothing can adjudicate it against other live claims. |
| `conditional_sweep.py` | 10 🟡 | A single-axis sweep reported as an absolute result (holding everything else fixed, unstated). |
| `corpus_check_required.py` | CC ✅ | An expensive run (>1h) whose question was never checked against the record — the *redundant*-claim gate. |
| `coverage.py` | COV ✅ | A noticed failure that never became a gate; + matrix↔module spec/code drift both directions. |
| `device_and_cost.py` | DC ✅ | An artifact can't say what **device** it ran on, or burned hours without ever projecting its cost. |
| `discriminating_power.py` | 4 🟡 | A metric/comparison with no power to discriminate the hypotheses it claims to test. |
| `doc_type.py` | D ✅ | A document of the wrong type or in the wrong place (finding vs plan vs board). |
| `instrument_required.py` | I ✅ | A GO reports a **size** without a **source** — an effect stated with no decomposition of where it came from. |
| `knob_reachable.py` | KR ✅ | A knob that changes the substrate but can't be set from the CLI — a prescribed fix that is unrunnable. |
| `lane_starvation.py` | L ✅ | CPU lanes starved while work continues elsewhere (rejects priority/focus rationalisation waivers). |
| `lever_efficacy.py` | 1 🟡 | A manipulation that never engaged — two arms numerically identical to many digits (40 live hits on first run). |
| `operating_point.py` | OP ✅ | A run that misses an operating-point target recorded in its **own** artifact. |
| `quantity_mismatch.py` | 6 ✅ | A wrong-quantity comparison (comparing across incommensurable axes / units / denominators). |
| `refuted_mechanism_reproposal.py` | RM ✅ | Re-proposing a mechanism the record already **refuted** (dendritic/BDSP/burstprop) as a fresh "next surpass". |
| `retrieval_completeness.py` | R ✅ | The record's own retrieval layer can't see part of the record (flat glob; `**` without `recursive=True`). |
| `single_seed.py` | 9 ✅ | A headline result stated on a single seed (project bar is ≥6 seeds before any generalization claim). |
| `stale_pointer.py` | 8 🟡 | Stale pointers / an unmaintained registry — citations to voided or moved docs. |
| `stated_value_mismatch.py` | SV ✅ | A finding states a named quantity that **disagrees** with the artifact it cites (0.200 derived vs 0.167 reported). |
| `summary_doc_freshness.py` | SF ✅ | The forward-looking summary docs (roadmap §7/§8, `ROADMAP.md`) drift while findings pile up. |
| `terminology.py` | 11 🟡 | Terminology overclaim — `consolidation`/`compositional`/`self-organized`/`GO` used without its code condition. |
| `throughput.py` | 7 🟡 | Liveness mistaken for progress — a completed multi-arm run reported with no elapsed/throughput figure. |
| `verdict_preconditions.py` | V ✅ | An artifact asserts a verdict without carrying what earned it (precondition changed, control can't reach its mechanism, knob inert). |

The matrix header keeps a running score (currently **26 blocking · 1 structural · 7 reporting**) — when you add a row, **re-derive that line, don't increment it** (the buckets must sum to the row count).

### Running the gates yourself (any shell — no Claude-Code tools involved)

```bash
cd /home/dant123/Projects/sim

# Run the whole registry over the whole corpus (relationship gates fire; content gates idle on empty paths):
GATE_BUDGET_S=30 .venv/bin/python -c \
  "from tools.gates import run_all; b,r=run_all([], verbose=True); print(chr(10).join(r)); print('BLOCKING:', len(b))"

# Run a single gate's self-audit in the failing direction (this is the honesty test):
.venv/bin/python -c "from tools.gates import corpus_check_required as g; print(g.selftest() or 'selftest OK')"

# Several gates are directly runnable as modules for a corpus scan (e.g. below_chance):
.venv/bin/python -m tools.gates.below_chance
```

Because GATE 5 runs on every commit through the hook, the normal way you'll *experience* the gates is: you stage a finding or an artifact, run `git commit`, and either it lands or you get `⛔ COMMIT BLOCKED — failure-class gate(s):` with the specific problem and a fix line. **Read the fix line — it names the exact command to satisfy the gate.** Do not reach for `--no-verify` as a reflex; a blocked commit is the system working, and eight self-blocks in one session was the intended behaviour, not a defect rate.

### How to add a new gate, end to end

When you notice a *new class* of mistake (not an instance — a class that could recur), close it mechanically:

1. **Log it** — add one row to `research/FAILURE_LOG.md`:
   ```
   | 2026-08-02 | <one-line description of the failure mode> | `gates/<module>` |
   ```
   From this moment `coverage` BLOCKS commits until the row names a real gate (or `NOT-GATEABLE: <≥15-char reason>`). You can log first and build the gate in the same commit.

2. **Write `tools/gates/<module>.py`** implementing the contract. Copy the shape of `corpus_check_required.py` or `refuted_mechanism_reproposal.py`:
   ```python
   """CLASS XX — <what the failure is, and WHY every existing gate missed the real incident>."""
   from __future__ import annotations
   import os, tempfile

   NAME = "your-gate-name"
   CLASS_ID = "XX"
   BLOCKING = True                    # False only if you cannot check it reliably at commit time

   _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

   def check(paths):
       if paths is None or len(paths) == 0:
           return []                  # standalone: do not scan the legacy corpus (content-gate convention)
       problems = []
       for p in [x for x in paths if x.endswith(".md")]:      # scope to the file types you judge
           full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
           if os.path.exists(full):
               problems += _check_one(full, p)
       return problems

   def selftest():
       """FAILING DIRECTION FIRST, then negative controls that must NOT fire."""
       bad = []
       # build the real bad case in a tempdir; assert check catches it:
       if not _fires_on_the_bad_case():
           bad.append("did NOT catch <the incident>")
       # assert it stays silent on the good case and on out-of-scope files:
       if _fires_on(good_case):
           bad.append("FALSE POSITIVE: flagged <legitimate case>")
       return bad
   ```
   Anchor the docstring in the **real incident** and state honestly what the gate *cannot* catch (every gate in the package does this — it's how the limits stay visible).

3. **Add the matrix row** to `docs/FAILURE_GATE_MATRIX.md` naming `` `gates/<module>` `` (and re-derive the score line). `coverage` enforces that this row and your module exist together — omit either and commits block.

4. **Prove the selftest fails in its failing direction** before you rely on it:
   ```bash
   .venv/bin/python -c "from tools.gates import <module> as g; print(g.selftest() or 'selftest OK')"
   # then mutate one line to break check() and confirm selftest() now returns a problem — that is the class-3 test.
   ```

5. **Commit.** GATE 5 auto-discovers your module (no hook edit), runs its selftest first, and from now on it blocks the class you just closed — for you and for every future agent, in any app.

That's the whole loop: **notice → log → gate → matrix row → selftest that can fail → commit**. Adopt it from the start. The temptation, when you hit a wall or catch yourself overclaiming, is to write a careful note and move on. In this repo the note is not the deliverable — **the gate is**. If a mistake was worth catching once, it is worth a file in `tools/gates/` so it can never cost anyone here a second time.

## 7. Reusable processes — skills and the fan-out workflow pattern (adopt equivalents)

This project has spent months converting hard-won lessons into **repeatable processes** so they compound instead of being re-learned. Two mechanisms carry them: **skills** (instruction packets the Claude Code agent loads on demand) and a **fan-out Workflow tool** (decompose a task → run parallel subagents → adversarially verify → synthesize). **You (Sol/Codex) will have neither of these exact tools.** That is fine — every skill is just a checklist, and the fan-out pattern is a way of thinking about multi-step work. The *enforcement* that actually matters lives in repo scripts and git hooks that run from any shell. This section tells you what each process is, when to run it, and the tool-agnostic way to replicate it.

### The four skills = four checklists (files you should read directly)

The skill bodies live at `.claude/skills/<name>/SKILL.md`. They are plain markdown — **read them; they are the distilled operating manual and far richer than this summary.** Treat each as a prompt you run against yourself at the trigger moment.

```bash
ls .claude/skills/
# evolve-skills  neural-simulator  sync-documentation  verify-go
cat .claude/skills/neural-simulator/SKILL.md   # the master realignment checklist
cat .claude/skills/verify-go/SKILL.md          # the adversarial-verification checklist (longest, richest)
cat .claude/skills/sync-documentation/SKILL.md # the doc-sync checklist
cat .claude/skills/evolve-skills/SKILL.md      # the retrospective-that-acts
```

**1. `neural-simulator` — realign + drive the mission.** When to run: whenever you have drifted, stalled, declared a wall, or are about to wrap up / hand off. It re-states the mission + non-negotiables, the boundary-surpassing research workflow (RAG our own findings first → read original sources in depth → run the ceiling early → cheap-first de-risk with anti-cheats → adversarially verify → iterate), the "no wrap-ups / immediate next action / parallelize" autonomy discipline, and the **SILENT-FAILURE CLASS** (15 specific ways a run can report success while being confidently wrong). For you, this is the file to re-read at session start and any time you catch yourself deferring, asking instead of deciding, or serializing independent work.

**2. `verify-go` — adversarially verify a positive result BEFORE it lands.** When to run: before committing/reporting any GO, "surpass", milestone, "it works / it's inert / byte-identical / tests pass", or even an interim "lead". The procedure is to spawn independent skeptics, each assigned a distinct refutation lens (reproducibility/power at 6 seeds; gate-cheat; control-integrity — one flag ≠ one variable, are both arms pinned at a bound; instrument-trust — read the runner's OWN verdict line; seeding; infra; the selectivity/MASS-artifact battery), each told **"try to REFUTE this; default to REFUTED if uncertain."** It also contains a large **"verifying a NEGATIVE"** section (a wrong NO-GO closes a capability that was never blocked — costlier and harder to notice than a wrong GO) and a "verifying a DIAGNOSIS" section (build the fix *with its lesion arm* in the same run). This is the single most load-bearing checklist in the repo.

**3. `sync-documentation` — keep the summary docs synced the SAME cycle a finding lands.** When to run: whenever code or findings change, and *mandatorily* the same cycle a committed finding changes a wall/gap STATUS, the CURRENT FRONTIER, or a next-action. Two layers: (A–H) mechanical drift (line counts, runner/test/findings counts, exports) — auto-fixable; (I) **semantic summary-doc sync** — the roadmap wall-ledger + `GAP_CLOSURE_MISSION.md` CURRENT STATE + `AUTONOMOUS_STATE.md` + `ROADMAP.md` must reflect the latest findings, contradictions reconciled, abandoned docs banner-ed. Layer (I) is the one that actually drifts and no mechanical pass catches it. **The discipline in one line: committing a finding is not enough — if it moves a status/frontier/next-action, move the board with it in the same commit, or the cycle isn't done.** A `.md` summary is downstream of the findings; keep the arrow pointing that way. Stale pointers are the #1 cause of re-deriving concluded work.

**4. `evolve-skills` — turn a recurring lapse into a durable process fix.** When to run: when a process lapse *recurs* (the owner had to catch the same class of problem twice — highest-value trigger), at a session-end/pre-compaction inflection, or when the owner asks. It gathers grounded evidence (this session's wins + lapses, `git log`, recent findings, the `feedback_*` memories), identifies recurring problems ranked by cost × recurrence, and makes the *smallest* incremental edit to the applicable skill/checklist that would prevent recurrence. **A caught lapse IS a process gap — fix the process so it can't recur, don't just patch the instance.** Keep the fix in the on-demand checklist, never bloat always-loaded context. For you: keep your own equivalent prompt-templates/checklists in a file, and when you repeat a mistake, edit the template, not just the current output.

### How to adopt these without the Skill tool

You cannot invoke `Skill`. The equivalent is trivial and arguably more transparent: **keep these four SKILL.md files (or your own condensed copies) as prompt-templates, and at the matching trigger, paste/re-read the relevant one and walk its checklist explicitly.** They are version-controlled markdown that any agent can `cat`. The content is what matters, not the loader.

### The fan-out Workflow pattern (decompose → parallel → adversarially verify → synthesize)

Claude Code has a `Workflow`/`Agent` tool that dispatches multiple subagents concurrently. **You will not have it.** The *pattern* is tool-agnostic and is used here for two things:

- **Adversarial verification** (the engine behind `verify-go`): N skeptics, each a *different* refutation lens, run independently against the same result; a synthesizer rules SURVIVES / SURVIVES-WITH-SCOPE-FIXES / INVALID. Independence is the point — redundant identical skeptics miss what diverse lenses catch.
- **Deep-research fan-out**: several parallel passes (our own findings via RAG, the original biology source read in depth, the external engineering literature) whose results are then synthesized.

Replicate it **sequentially within one context, but keep the roles distinct.** For a result you're about to commit: explicitly adopt each lens in turn ("as the reproducibility skeptic… as the control-integrity skeptic… as the instrument-trust skeptic…"), write down what each finds, then synthesize. The discipline that makes it work is **role separation + a refute-not-confirm stance**, not literal parallelism. A confounded GO caught before commit is worth more than a committed overclaim.

**Two hard rules from the repo that transfer directly to you:** (a) the controller runs the decisive multi-seed sweeps and reads the runner's own verdict — a subagent that "launches a sweep" runs it single-threaded and usually orphans it (dies when the subagent returns → zero output); so if you delegate, delegate the *build* and run the sweep yourself. (b) Fan a multi-seed run **across OS processes, one per seed** — a runner's own `--seeds 42 43 44 …` loops serially in one process = 1/N cores. The proven pattern:

```bash
for s in 42 43 44 45 46 100 101 102 103 104; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SIM_BACKEND=numpy \
    .venv/bin/python -u -m <runner> --seeds $s --json raw/_out_seed$s.json > raw/_log_seed$s.log 2>&1 &
done; wait; echo ALL DONE
```

### The research/verify gates ARE the workflow — and they run from any shell

The most important thing to internalize: **the rules above are not asked-to-be-remembered, they are ENFORCED by scripts and a git pre-commit hook that work from Codex exactly as from Claude Code.** Where a remembered rule and a gate disagree, *the gate is authoritative*. Read `docs/FAILURE_GATE_MATRIX.md` (the index of every failure class → the gate that blocks it) and `research/FAILURE_LOG.md` (a noticed failure gets one line and `gates/coverage` blocks until it names a gate or declares `NOT-GATEABLE`).

**Enable the hooks in your clone first** (they are committed to the repo, so this is one command):

```bash
git config core.hooksPath tools/githooks   # wires the enforced pre-commit + post-commit gates
```

The pre-commit hook (`tools/githooks/pre-commit`) runs the auto-discovered gate registry (`tools/gates/`, 28 registry modules (the matrix tracks 34 failure-classes — the extra rows are non-registry doc/claim/biology hooks): document structure W1/W2, claim-traced-to-artifact, single-seed headlines, quantity-mismatch, boundary-verdict-needs-external-cite, attribution-required, verdict_preconditions, stale-pointer, and more). A gate whose `selftest()` can't fail in its failing direction is treated as broken and reported loudly — because four checks here shipped unable to fail. **Deliberate, visible override is `git commit --no-verify`; use it only when you mean to.**

The workflow-as-scripts you should run at the matching moment (all from any shell):

```bash
# BEFORE the first lever against any defect — has this already been scoped/tried/refuted?
bash tools/before_you_build.sh "<defect in one line>"
bash tools/rag/search.sh "<question>" 5 --corpus finding   # our own conclusions first

# The deep-research gate (re-prints every primary-source hit with a read command; ≥2 levers on one defect ⇒ it FIRES)
bash tools/research_gate.sh "<question>"

# Doc-structure rules (W1 retraction-marking, W2 ≤800-char prose lines) — same check the pre-commit runs
.venv/bin/python tools/check_docs.py
.venv/bin/python tools/split_long_doc_lines.py --apply   # fix W2 (refuses to write if content changes)

# VERIFY infra claims instead of asserting them
bash tools/push_both.sh                 # pushes then ls-remote-VERIFIES both remotes (never `echo pushed`)
.venv/bin/python tools/engagement_check.py research/findings/raw/_emerge10_stageA_dap_fire_first.json   # did the mechanism actually engage? (pass a real result JSON)

# Record an external-literature read (unblocks a BOUNDARY verdict, which now REQUIRES an external cite)
bash tools/record_external_search.sh "<query>" "<source>"
```

And the in-probe helpers that make void-arm detection *execute* rather than be remembered — **import them at the top of every probe, before writing the mechanism** (`tools/lab.py`):

```python
from tools.lab import lever, void_if, before_after, undefined_if_empty, attributable_to
lever("w_max", before=unbounded_score, after=bounded_score, continuous=sat_frac)
void_if(sat_frac == 0.0, "the soft bound never engaged; w_max arms are identical by construction")
```

**Net adoption advice for you:** you don't need Claude's Skill/Workflow/Monitor tools — you need (1) the four `.claude/skills/*/SKILL.md` files as your standing checklists, run at their triggers; (2) `git config core.hooksPath tools/githooks` so the gates gate your commits; and (3) the habit of running `before_you_build.sh` / `research_gate.sh` before building, `verify-go`'s skeptic lenses before committing a positive, and `sync-documentation`'s Layer-I same-cycle whenever a finding moves a status. The scripts, the RAG index, and the git hooks are all shell-and-Python — they were never Claude-specific, and they are the part that actually holds the discipline in place.

## 8. Tooling, repo map, and the key commands

This section gets you productive in this repo fast, and — just as important — gets you using the project's *guardrails* from your first commit instead of rediscovering them the hard way. Almost everything here is a plain script, a git hook, or a Python module: it runs from any shell, so none of it depends on the Claude Code app. Where a tool *is* Claude-specific, it is called out with the repo-based way you (Sol, via Codex) achieve the same thing.

### 0. First principle: the gates are authoritative, not the prose

The single most important thing to internalize: **this project converted its hard-won rules from "things to remember" into "checks that block."** They live in `tools/gates/` and are wired into a git `pre-commit` hook. When a remembered rule and a gate disagree, **the gate wins** — it is the thing that actually runs. Read [`docs/FAILURE_GATE_MATRIX.md`](docs/FAILURE_GATE_MATRIX.md) once, early; it is the specification of every known failure class and what mechanically prevents it. Do not rebuild a check that already exists — the matrix tells you what's covered.

The hook is already installed via `git config core.hooksPath tools/githooks`. It will run against your commits regardless of which app you drive git from. Do **not** reach for `git commit --no-verify` to get past a red gate; the block is almost always pointing at a real defect.

### 1. Environment and backends

There are **two** virtualenvs, both Python 3.11.14, and they are kept separate on purpose:

- **`.venv/`** — the simulator. Pinned `torch`/`cupy` CUDA stack. Use `.venv/bin/python` for everything except RAG. The GPU is an RTX 3090 (24 GB); nets >100K neurons need 20 GB+ VRAM.
- **`.venv-rag/`** — the local RAG index only (LlamaIndex). Installing LlamaIndex into `.venv` would churn its pinned CUDA deps, so it is isolated. Use `.venv-rag/bin/python` **only** for `tools/rag/rag_search.py`.

**Backend selection** (`sim/backend.py`) is controlled by the `SIM_BACKEND` env var:

- `SIM_BACKEND=cupy` → force GPU (raises if CuPy unavailable)
- `SIM_BACKEND=numpy` → force CPU
- `SIM_BACKEND=auto` or unset → CuPy if available, else NumPy

**The trap that has bitten this repo repeatedly:** hundreds of research runners call `os.environ.setdefault("SIM_BACKEND", "numpy")` in their body. So if you launch a runner and *don't* set `SIM_BACKEND` explicitly, you silently get the **CPU** path — 10–50× slower — even on a machine with a free GPU. A "4-cell GPU test" once ran 30 minutes on CPU this way. **Always set `SIM_BACKEND` explicitly at the call site.** Use `SIM_BACKEND=numpy` for tiny smoke tests and CI; use `SIM_BACKEND=cupy` for any heavy or decisive run. `backend.py` prints a warning when numpy is selected while a GPU is present, and `tools.lab.assert_backend("cupy")` will *raise* on mismatch — use it inside decisive runners.

### 2. Repo map

```
sim/                 The simulator engine (the ONE spiking substrate).
                     bridge.py (the big integrator), config.py (CoreSimConfig — every knob),
                     backend.py (cupy/numpy), neuromodulators.py, regions.py, dendritic_*.py,
                     plasticity/STDP/BTSP/BDSP code, bptt_snn*.py, tokenizers, __init__.py (exports).
                     ⚠️ sim/ is NEVER edited by documentation work.

research/
  runners/           1300+ runnable experiment modules, invoked as `python -m research.runners.X`.
                     __init__.py = the AUTOMATIC PROVENANCE door (see §5). Naming: user-facing
                     gates are g1..g11 (e.g. g11_bg_runner.py); de-risk probes are _*_derisk.py.
  findings/          1800+ append-only research records (*.md). NEGATIVE results live here as
                     findings, not failures. findings/raw/ holds JSON artifacts + _provenance/.
                     This is append-only — never rewrite an old finding (it destroys the audit trail).
  biology/           Biology BINDINGS: a mechanism bound to a source + quote + constraints_config
                     (see §6). Checked by tools/biology_check.py on every commit.
  FAILURE_LOG.md     One line per newly-noticed failure mode; gates/coverage BLOCKS until each
                     line names a gate or declares NOT-GATEABLE.
  queue/             Compute-pool queue + dispatch logs + .corpus_checks.jsonl.

tools/
  gates/             THE registry: one module per failure class, auto-discovered (see §3).
  before_you_build.sh, rag/rag_search.py, push_both.sh, lab.py, verdict.py,
  check_docs.py, biology_check.py, claim_check.py, lane_check.py, split_long_doc_lines.py
  githooks/          pre-commit (5 gates + registry) and post-commit. Active hooksPath.

docs/
  FAILURE_GATE_MATRIX.md   The failure→gate spec. Read this.
  TERMS.md                 One-term-one-meaning (§7).
  WRITING.md               Doc-structure rules W1/W2 (§7).
  RETRACTED.md             Registry of voided docs (W1 checks citations against it).
  ENGINE_REFERENCE.md      Architecture / thread model / config traps — retrieve, don't reload.
  plans/                   Design docs. The PRIMARY plan is 2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md.

.claude/                   Claude-Code-specific config: hooks/, skills/, settings.json, style.md.
                           You (Codex) won't use these directly, but the skills/ dir documents the
                           workflows — worth reading as prose (see §8).

Top-level steering docs you MUST read at session start:
  CLAUDE.md                    Mission + non-negotiables + the enforced-workflow pointers.
  GAP_CLOSURE_MISSION.md       The live working board — CURRENT STATE = the resume point.
  ROADMAP.md                   Plain-language status surface + a "Project shorthand" glossary.
```

### 3. The gates and the git hook (tool-agnostic — works from any shell)

`tools/gates/` is a registry: each file is one failure class exposing `NAME`, `CLASS_ID`, `BLOCKING`, `check(paths)`, and `selftest()`. The registry **refuses to trust a gate whose `selftest()` does not itself fail in the failing direction** — because this project has shipped four "checks that couldn't fail" (a `;` where `&&` was meant, a pipe eating an exit status, a nonsense query scoring 18 hits and passing). Adding a new failure class is one new file in that directory; the hook never changes.

The `pre-commit` hook runs five gate groups on every commit: **G1** document structure (W1/W2 via `check_docs.py`), **G2** claims traced to cited artifacts (`claim_check.py`), **G3** biology bindings (`biology_check.py`), **G4** new findings must declare a `status:` frontmatter, and **G5** the full auto-discovered registry (single-seed headlines, quantity mismatch, provenance, terminology, attribution, device/cost, corpus-check-required, and more).

Run the registry manually any time (before you commit, to see what will block):

```bash
.venv/bin/python -c "from tools.gates import run_all; b,r=run_all([]); print('\n'.join(r)); print('BLOCKING:',len(b))"
```

### 4. Reproducibility gotchas — read before writing any runner

**(a) `cfg.seed` is what actually seeds the substrate. `actual_seed_used` seeds NOTHING.** This is a real bug that cost the project months of confounded results. `actual_seed_used` is a *reporting* field the bridge never reads. Heterogeneity (per-neuron firing thresholds) is seeded from `cfg.seed` (or `cfg.heterogeneity_seed`), both defaulting to `-1` = unseeded global RNG. If you never set `cfg.seed`, `--seeds 42` will **not** control your neurons — every run gets different neurons, and each net build advances the global RNG (four back-to-back builds in one process differed by up to 18.4 mV).

```python
cfg = CoreSimConfig(..., seed=42)          # ✅ correct
cfg = CoreSimConfig(); cfg.seed = 42       # ✅ also correct
cfg = CoreSimConfig(); cfg.actual_seed_used = 42   # ⛔ SEEDS NOTHING
```

Verify, don't assume: build twice at one seed and hash `cp_neuron_firing_thresholds`; identical ⇒ seeded. This is pinned by `tests/test_determinism.py::TestSubstrateActuallySeeded`.

**(b) 6-seed validation is the bar for any generalization claim.** 3-seed indicators are unreliable. The canonical seed set is **42, 43, 44, 100, 101, 102**. A single-seed headline is caught by `gates/single_seed`. Never call a result "GO" or "surpass" off fewer than 6 seeds.

**(c) GPU vs NumPy** — see §1. CuPy for heavy/decisive runs; numpy only for tiny smoke. Set `SIM_BACKEND` explicitly.

**(d) Units** are fixed throughout: time **ms**, voltage **mV**, current **pA** (or µA/cm²), conductance **nS** (or mS/cm²), capacitance **pF** (or µF/cm²).

### 5. Automatic provenance — you get it for free, don't fight it

`research/runners/__init__.py` runs on *every* `python -m research.runners.X` invocation (that's how `-m` works — it imports the package first). On import it stamps a run record (argv, cwd, git SHA, dirty flag, python, relevant env vars including `SIM_BACKEND`, pid) into `research/findings/raw/_provenance/runs.jsonl`; at exit it writes an `<artifact>.prov.json` sidecar for every file the run created under `research/findings/raw/`. Nothing to remember, no runner edited. It is fully wrapped (a provenance failure warns, never kills the run) and disabled by `SIM_NO_PROVENANCE=1` (for byte-identical reruns / CI). **Write your runner's artifacts under `research/findings/raw/`** so they get sidecarred and become protected/traceable; `gates/artifact_provenance` blocks unprovenanced artifacts.

### 6. Biology is recorded once, not re-researched

`research/biology/<id>.md` binds a mechanism to a source with a quote *that must still resolve in the local corpus*, plus a `constraints_config` block naming config values the biology REQUIRES. Example: `btsp-place-field-formation.md` records that BTSP is one-shot (`laps: 1` — repeated traversals ERASE the field) and that `w_max` must be `> W0` (below it, the clamp drags weights down and you measure clamp depth, not learning). `tools/biology_check.py` (pre-commit G3) fails the commit if a source pointer has rotted or if the config contradicts the biology. Before you go re-derive a biological fact, check `research/biology/` and the RAG `kandel`/`paper` corpora — a 21-agent round once re-established a fact knowable from the paper on day one.

### 7. The experiment-hygiene helpers, terms, and doc rules

**`tools/lab.py`** — import these instead of *remembering* the rules; each helper encodes a real retraction. `lever(name, before, after)` asserts a manipulation actually changed something (catches A/Bs whose flag was already set). `before_after(...)` catches measurements taken upstream of the manipulation and lesions that didn't persist. `bound_check(rule, bound, weight)` raises if a plasticity bound sits at/below the weights it governs (the trap that owned 97% of a gap#5 result). `undefined_if_empty(...)` prints UNDEFINED, never a fabricated 0. `attributable_to(treatment, control)` and `term_budget(...)` say *whose* the measured change was. `assert_backend("cupy")` raises on a wrong-device run. `Verdict(...)` makes UNDEFINED the default and a run earn its GO. Run `.venv/bin/python -m tools.lab` to see the self-check.

**`docs/TERMS.md`** — one term, one meaning, one code condition. Before writing `consolidation`, `compositional`, `self-organized`, `closed`, `GO`, `fully spiking`, `byte-identical`, `lesion`, `selective`, or `works` in a finding/commit/board entry, check its CODE CONDITION. An unchecked term is a hypothesis. `gates/terminology` reports overclaims.

**`docs/WRITING.md`** (W1/W2), checked by `tools/check_docs.py`: **W1** a voided doc is registered in `docs/RETRACTED.md`, and no governed file cites it without a `⛔` on the same line. **W2** prose lines in the 6 governed files (`CLAUDE.md`, `GAP_CLOSURE_MISSION.md`, `ROADMAP.md`, `README.md`, `docs/TERMS.md`, the master roadmap) are ≤800 chars (tables/code exempt). Retrofit helper: `.venv/bin/python tools/split_long_doc_lines.py --apply`.

### 8. Claude-Code-specific tools → your tool-agnostic equivalents

A few things the previous agent used are features of the Claude Code app that **you (Codex) will not have**. Each maps to something in the repo you *can* run from any shell:

| Claude Code feature | What it did | Your repo-based equivalent |
|---|---|---|
| **Skill tool** (`neural-simulator`, `verify-go`, `sync-documentation`, `evolve-skills`) | Loaded packaged workflow instructions on demand | The skills are just Markdown under `.claude/skills/<name>/SKILL.md`. **Read them as prose** — they document the research workflow, the adversarial GO-verification checklist, and the doc-sync procedure. The *enforceable* parts already exist as gates/scripts; run those. |
| **Monitor / ScheduleWakeup** (long-run heartbeat) | Re-invoked the agent every ~15 min to check GPU/procs/output on background runs | Run runs as background processes and poll their log/result files yourself; the dispatcher writes `research/queue/dispatch.log` + `job_status.log`. There is no daemon — cross-session continuation is a manual "continue." |
| **Agent/subagent dispatch** (parallelism) | Fanned independent tasks to concurrent subagents | Parallelize with shell backgrounding / the compute pool (`research/queue/`). `.venv/bin/python tools/lane_check.py` flags GPU monoculture and starved CPU lanes when you stock a queue. |

The scripts, gates, RAG index, biology bindings, and git hooks are all plain files — they work identically from Codex.

### 9. The command cheat-sheet (verbatim)

```bash
# ── BEFORE the first lever against any defect (research gate; ~0.6s) ──
# Surfaces prior findings/scope docs/exclusions AND records that the check happened
# (gates/corpus_check_required reads that record).
bash tools/before_you_build.sh "the slot competition ignores the cue"

# ── Search the local corpus (findings + biology catalog + Kandel 6e + textbooks) ──
# Uses the RAG venv. --corpus one of: all(default) finding plan doc catalog kandel paper
bash tools/rag/search.sh "how does BTSP form a place field" 5 --corpus kandel
bash tools/rag/search.sh "have we already tried mean-subtract recall" 5 --corpus finding
# A hit is a POINTER, not a paraphrase — open and READ the cited source.

# ── Run a research-gate runner (note: SIM_BACKEND is set EXPLICITLY) ──
# GPU, decisive:
SIM_BACKEND=cupy .venv/bin/python -m research.runners.g11_bg_runner --moving-goal --seed 42 \
    --n-steps 1800 --out research/findings/raw/g11_bg/g11_seed42.json
# CPU smoke test:
SIM_BACKEND=cupy .venv/bin/python -m research.runners.g11_bg_runner --probe-action W   # GPU BG-circuit probe (this one needs cupy)

# ── 6-seed sweep (the validation bar): seeds 42 43 44 100 101 102 ──
for s in 42 43 44 100 101 102; do
  SIM_BACKEND=cupy .venv/bin/python -m research.runners.g11_bg_runner --seed $s \
      --out research/findings/raw/g11_bg/g11_seed$s.json
done

# ── Experiment-hygiene helper self-check ──
.venv/bin/python -m tools.lab

# ── Document-structure rules (W1/W2) ──
.venv/bin/python tools/check_docs.py
.venv/bin/python tools/split_long_doc_lines.py --apply   # fix W2 (refuses to write if content changes)

# ── Run the full failure-class gate registry manually ──
.venv/bin/python -c "from tools.gates import run_all; b,r=run_all([]); print('\n'.join(r)); print('BLOCKING:',len(b))"

# ── Commit-and-verify to BOTH remotes (origin + gitea) — asserts the remotes actually moved ──
bash tools/push_both.sh            # current branch
bash tools/push_both.sh main       # explicit branch
# Prints "verified: both remotes at <sha>" only if git ls-remote confirms each remote is at HEAD.
# If it prints "PUSH NOT VERIFIED", do NOT report it as pushed.
```

Two notes on the commands above. `push_both.sh` exists because the old `git push … | tail -1; echo pushed` habit reported success unconditionally (it was true by luck ~20 times); this script asks each *remote* what it actually has. And `before_you_build.sh` fires the research gate: **≥2 distinct levers against the same defect without resolution ⇒ stop and research the record first** — cheapness of the next test is not an exemption (6 levers / ~4 GPU-h were once spent before the gate subjectively "fired," and one research round then resolved it).

## 9. Resources at your disposal

You are working in `/home/dant123/Projects/sim` on a Linux box (CachyOS). Everything below is verified against the live system as of this handoff. Two Python interpreters matter and are NOT interchangeable: `.venv/bin/python` for the engine and everything else, and `.venv-rag/bin/python` for the RAG search only (it has `llama-index`; keeping it separate protects the engine venv's pinned torch/cupy CUDA stack). Both are Python 3.11.

### 1. The repository — orientation map

One line each; the exact commands live in the tooling / command cheat-sheet section of this handoff.

- **`sim/`** — the spiking engine: `bridge.py` (the substrate), `regions.py`, `kernels.py`, `config.py`, `backend.py` (the CuPy/NumPy switch), plus neuromodulators, dendrites, tokenizers, BPTT-SNN, etc.
- **`research/runners/`** — ~1360 de-risk runner modules (`python -m research.runners.<X>`); each records provenance automatically via `research/runners/__init__.py`.
- **`research/findings/`** — ~1890 concluded-result markdown docs. **This is the project's memory** — negatives are documented here as findings, not failures. Search it before building anything (see §2).
- **`research/biology/`** — 8 biology bindings (`<id>.md`): each ties a mechanism to a primary source with a quote that must still resolve, plus the config values the biology requires.
- **`tools/`** — the gates (`tools/gates/`, wired into `tools/githooks/pre-commit`), the RAG (`tools/rag/`), and the compute/ops scripts (pool, AWS, GPU recovery, research gate).
- **`docs/`** — `plans/` (incl. the primary plan `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`), `TERMS.md` (one-term-one-meaning), `WRITING.md` (doc rules W1/W2), `FAILURE_GATE_MATRIX.md`, the charter (`CLAUDE.md` at root), and this `HANDOFF.md`.
- **`.claude/skills/`** — 4 process playbooks as plain markdown, readable directly: `neural-simulator` (realignment + research workflow), `verify-go` (adversarial verification before claiming a positive), `sync-documentation`, `evolve-skills`.

### 2. RAG system + local corpus (`tools/rag/rag_search.py`)

Verified present and executable. Run it with the **RAG venv** (not the engine venv). It does filtered hybrid vector+BM25 fusion → cross-encoder rerank over the project's prose knowledge base. Both lexical and semantic retrieval are active for every targeted corpus. It **locates** the passage — the discipline is still to open the cited file and read it.

```bash
# usage: rag_search.py "<question>" [top_k] [--corpus finding|plan|doc|catalog|kandel|paper|all]
bash tools/rag/search.sh "how does BTSP set a place field" 5 --corpus kandel
bash tools/rag/eval.sh --no-write  # fail-closed quality check over findings + scientific corpora
```

Corpora it indexes (`--corpus`, default `all`):

- `finding` — `research/findings/*.md` ("have we already CONCLUDED / tried X?")
- `plan` — `docs/plans/*.md` ("did we already DESIGN X?")
- `doc` — `docs/*.md` + CLAUDE/ROADMAP/README
- `catalog` — `sim-catalog/references/*.md` (catalog entries)
- `kandel` — Kandel PNS 6e full text ("how does the BIOLOGY do X?")
- `paper` — the specialty texts (Marr 1969, Albus 1971, Buzsáki *Rhythms of the Brain*, O'Keefe-Nadel *Cognitive Map*, Schultz, Sutton-Barto, Tepper/Bolam BG)

The index lives at `/home/dant123/Projects/rag_index/llamaindex_full`. Linked worktrees resolve it through Git's common checkout rather than their immediate parent; override with `SIM_RAG_ROOT`. Only `finding`/`all` may fall back to a findings-only index, so a missing full reference index cannot look like a valid empty catalog search.

The `main` post-commit hook refreshes changed project prose in the background. Because the sibling source catalog is not reliably committed from this Linux checkout, the user-level `sim-rag-autoupdate.timer` independently checks the full corpus every five minutes and notices added, edited, or removed catalog sources. Before refreshing, it creates a missing `.txt` companion for each readable PDF without overwriting existing hand-edited text. Image-only, empty, or unreadable PDFs fail closed and are logged for OCR. Both paths use the same manifest and lock. A refresh defers while any indexed project Markdown is uncommitted, so drafts cannot enter the shared record; the timer retries automatically after the worktree is clean. They build a candidate index, run the labeled read-only quality floor against that candidate, and atomically publish it only after the gate passes; a bad candidate leaves the previous index and manifest live. If prose changes while a refresh is running, the lock holder repeats before marking the manifest current. The current Linux passage-level benchmark is 13/13 within rank 3, 11/13 at rank 1, MRR 0.923 across prior findings, the reference catalog, Kandel, and specialty papers. Scientific labels require expected passage text and can forbid neighboring text, so a shared filename cannot create a false positive. Search results include an absolute source path and line locator. `python3 tools/rag/check_workflow.py` verifies paths, interpreters, hooks, index, catalog, schema, installed timer helper, and the live periodic timer; `--install` repairs both hooks and timer. SOMA is optional and currently unavailable on this PC; its failure is logged, while the maintained LlamaIndex path remains required and fail-closed.

### 3. Full-text reference access (papers + textbooks)

The catalog is a **sibling repo**: `~/Projects/sim-catalog/references/`. Every source has a `.pdf` and a matching `.txt` — **read the `.txt`** (fast, greppable). Verified contents:

```
~/Projects/sim-catalog/references/textbooks/
  kandel-pns-6e/full-book.txt            # Kandel Principles of Neural Science 6e, full book
  buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
  okeefe-nadel-cognitive-map/OKeefe-Nadel-1978-HippocampusCognitiveMap.txt
  cerebellum-marr/Marr-1969-cerebellar-cortex.txt   (+ Hesslow-2013, Moore-2002)
  cerebellum-albus/Albus-1971-cerebellar-function.txt
  schultz-dopamine/  (Schultz 1998/2016, Hollerman-Schultz 1998 — RPE)
  basal-ganglia-reviews/  (Tepper, Bolam — striatal GABAergic circuitry)
  sutton-barto/SuttonBarto-RL-2nd-ed.txt
```
Also at `~/Projects/sim-catalog/references/`: `feature-catalog.md` and `biology-buildout-roadmap.md`. There is **no separate `papers/` dir** — the papers live inside `textbooks/<topic>/`.

```bash
# read pattern:
sed -n '1,120p' ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
# surface primary-source hits with ready-to-run read commands (re-prints every primary hit at the end):
bash tools/research_gate.sh "place field formation timing"
```

### 4. Compute — LOCAL system (your default)

Verified live: **NVIDIA RTX 3090, 24576 MiB (24 GB) VRAM**, **20 CPU cores** (`nproc`).

Backend switch (read by `sim/backend.py`):

```bash
SIM_BACKEND=cupy   .venv/bin/python -m research.runners.<X>   # force GPU
SIM_BACKEND=numpy  .venv/bin/python -m research.runners.<X>   # force CPU
# unset / SIM_BACKEND=auto  -> CuPy if available, else NumPy (the default)
```

VRAM rule: **networks >100K neurons need 20GB+**, so they fit the 3090 but leave little headroom — don't stack GPU runs blindly; watch `nvidia-smi`.

**GPU crash recovery** (verified scripts): the 3090 can **fall off the bus under sustained load** — a *hung core* is reboot-only, a driver/process glitch may be recoverable without reboot. Passwordless sudo is available.

```bash
bash tools/gpu_recover.sh          # assesses state; REFUSES a no-reboot recovery on a hung core
# hung core (dmesg: "fell off the bus" / "_scrubWaitAndSave: Timed out"): sudo reboot
#   -> lmtrain-resume.service (systemd, enabled) auto-resumes LM training on boot
```
Full procedure: `docs/GPU_CRASH_RECOVERY.md`. Config note: LACT caps the card at ~300 W (down from 390 W) to reduce crashes.

### 5. Compute — the mini-PC POOL (CPU overflow, numpy-only)

Three reimaged mini-PCs on the LAN, reachable by ssh alias. Verified live: **pool40 / pool41 / pool42, 12 cores each = 36 cores total**.

- Remote repo root: `~/derisk-pool/sim`
- Remote interpreter: **`~/derisk-pool/sim/.venv/bin/python`** (numpy **2.4.6** — this is the one the queue uses; a secondary `~/simvenv` with numpy 2.2.6 also exists). CPU / numpy backend only.

Provision (idempotent — rsyncs code, builds the venv, installs numpy+scipy+h5py+pyyaml, verifies `SIM_BACKEND=numpy` imports):

```bash
bash tools/pool_provision.sh                 # all three
bash tools/pool_provision.sh pool41          # one node
```

Launch — the **reliable manual pattern** is a direct detached ssh (survives your session; run from the remote repo root):

```bash
ssh pool41 'cd ~/derisk-pool/sim && SIM_BACKEND=numpy setsid \
  .venv/bin/python -u -m research.runners.<X> --seed 42 --out research/findings/raw/<...>.json \
  >~/run.log 2>&1 </dev/null & disown'
```

A queue mechanism also exists — `research/queue/pool.queue` (one shell line per job) drained by `tools/pool_autodispatch.sh` (currently running; enqueue via `tools/pool_queue.sh`, collect via `tools/pull_pool_results.sh`). **Note:** there is **no `pool-dispatch.service` systemd unit** — the dispatcher is a plain script and has historically been glitchy (watch `research/queue/dispatch.log`). When in doubt, prefer the direct-ssh pattern above.

### 6. Compute — AWS GPU (only when the local 3090 is saturated)

Two scripts, verified: `tools/aws_provision.sh` (rsync repo → venv → cupy, and **asserts `cupy` actually sees the GPU** — `nvidia-smi` alone is not proof) and `tools/aws_gpu.sh` (drive the lane). **There is no `aws_train.sh`.**

```bash
bash tools/aws_gpu.sh status      # instance id / type / state / public IP
bash tools/aws_gpu.sh ssh         # prints the ready ssh command
bash tools/aws_gpu.sh stop        # <-- STOP WHEN IDLE; it BILLS while running
```

**CRITICAL: the instance bills for every minute it is `running` — stop it (`bash tools/aws_gpu.sh stop`) the moment work finishes.** State (instance id + SSH key path) lives durably in `research/queue/.aws_gpu`, **not** in memory — a prior lane lost its key to a `/tmp` reboot. As of this handoff there is **no active AWS lane** — the previous `g5.xlarge` was terminated 2026-08-02. To use AWS: launch/record a new instance (see `tools/aws_gpu.sh` + `tools/aws_provision.sh`), run work with `SIM_BACKEND=cupy`, and **`bash tools/aws_gpu.sh stop` (or `terminate`) the moment it is idle** — a *running* instance bills compute every minute and even a *stopped* one accrues EBS storage cost. Dispatch on it with `SIM_BACKEND=cupy .venv/bin/python -m research.runners.<X>`.

### 7. Allocation principle

GPU (the 3090) is the training bottleneck — reserve it for training/GPU-bound runs. **CPU de-risks run local by default** (free, and faster than the pool). The **pool is CPU overflow** when the 20 local cores saturate. **AWS is a GPU lane** for when the 3090 is busy. Don't offload just because a lane exists — match the job to the actual bottleneck, and stop paid lanes when idle.

---

### Running / talking to the brain today

The deployed multi-turn chat console is `research/runners/brain_chat_tui.py`, driving `research/runners/brain_conversational_agent.py`:

```bash
.venv/bin/python -m research.runners.brain_chat_tui --help   # options: --load a trained brain, --self-knowledge, ...
```

**Be aware of what this actually exercises:** the deployed console's default wording comes from the **external Qwen LLM** (scaffold ledger item #0 in §10) — the brain does the grounding + the no-confabulation gating, but the sentences are currently the transformer's. So "talking to the brain" today runs the *scaffolded* pipeline; re-pointing generation onto the substrate (language as grounded action) is the short-timescale crux (§3, P1).

---

## 10. Current state — what's validated, what's scaffold, what's open

This is the honest snapshot you should carry in your head before touching anything. It reconciles three living documents — read all three yourself early:

- **`GAP_CLOSURE_MISSION.md`** — the live working board. Its **CURRENT STATE / 5-gap table** is the session-by-session resume point (but note its own header warning: the 5-gap table is a *stale sub-view*; trust the STATE header + the roadmap over the raw gap cells).
- **`ROADMAP.md`** — the plain-language status surface (last synced 2026-08-02). Section 3 is the one-screen picture; §8 lists the stand-ins; §9 is the honest frontier. Its "Project shorthand" table (top) decodes every coinage.
- **`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`** — the forward-looking plan; and the newest framing doc **`docs/plans/2026-08-02-PROJECT-CHARTER-grounded-emergence-realignment.md`** (the charter that renames the goal from "5-gap closure" to *grounded emergence* and defines the P1–P7 principles / T1–T7 traps the newest gates enforce).

**The one law that governs how to read every "GO" below:** a wall/negative is a verdict on a *method*, never a license to abandon a *capability*. "CLOSED" is defined narrowly and almost nothing meets it: a capability is closed only when it is (a) fully-spiking on the one shared substrate, (b) genuinely biological (neurons/synapses only; host code for world+body alone), (c) 6-seed validated with anti-cheats, (d) adversarially verified, and (e) *wired into the system the owner actually uses* — no scaffold left standing as the faculty. Most validated results below are **(a)–(d) in an isolated slice but fail (e)** — they are proven mechanisms that the deployed pipeline does not yet run. That gap between "validated in a runner" and "deployed" is the single most important thing to internalize, and the reason the scaffold ledger in part 2 is long.

**Culture note that will save you:** GOs here get *retracted*, routinely, by the project's own adversarial checks — often the same day. The roadmap openly carries withdrawn claims (the gap#3 biased-competition write-up was retracted by its own audit; the 2026-07-30 replay-reader "it learned" claim was withdrawn twice by a one-line "measure the untrained network too" check; three of nine retractions in one 2026-07-28 session were pure terminology overclaim over correct data). Treat every positive as provisional until you've seen the anti-cheat that would have caught the confound. Run `verify-go`-style adversarial skeptics before you believe a "surpass."

---

### 1. What is genuinely validated (brain-native, multi-seed)

These are real, mostly 6-seed, with load-bearing anti-cheats. Unless flagged "deployed," assume the result lives in an **isolated runner slice**, not the production chat/nav loop.

**The engine + substrate (mature, production-grade — 🟩).** The GPU conductance-based spiking engine (`sim/bridge.py`), ~50 cell-type presets incl. Izhikevich / HH / AdEx / resonate-and-fire phasor (`sim/enums.py`, `sim/kernels.py`), the region-and-pathway framework with runtime gates (`sim/regions.py`), the learning-rule family — STDP, short-term plasticity, Hebbian, dopamine-gated three-factor, homeostasis, dendritic burst rule (`sim/kernels.py`), the neuromodulator subsystem (`sim/neuromodulators.py`), and lifelong-learning persistence (`sim/lineage.py`, `sim/synapse_storage.py`). This is not in question; build on it. **Trap you must respect:** the substrate is only seeded when you set `cfg.seed` — `actual_seed_used` seeds *nothing* (see the CLAUDE.md box; pinned by `tests/test_determinism.py::TestSubstrateActuallySeeded`). Eight runners historically shipped confounded because of this.

**Memory & consolidation (🟩 mechanisms / 🟧 one deep boundary).**
- BTSP one-shot encode → **bistable-CA3 pattern completion from a partial cue**, mechanism 6/6 seeds (cue-gated, permuted-specific 0.000, no-encode collapses) — the keystone `sim/` change was intrinsic dendritic bistability (self-regen NMDA plateau + KIR down-state). Findings: `2026-07-18-gap5-CA3-completion-CLOSED-intrinsic-dendritic-bistability-resolves-the-trilemma.md`, `2026-07-18-gap4-gap5-UNIFICATION-BTSP-stores-bistable-CA3-completes-mechanism-6seed-GO.md`. This had been at chance for the project's entire history.
- **Ecker-2022 CA3 traveling SWR replay, 6/6 GO**, the band *emerges from experience* (STDP + directed traversal, `2026-07-25-...learned-band-emergence`), a full **WAKE→SLEEP→WAKE round-trip runs co-resident on the production composer bridge** with conversational memory byte-identical and replay DECODE_r=1.000 (commits `1bdcc5a4`/`42da00dd`). A *neural* reader of replay direction (pairwise "who fired first" votes, ~12 ms axonal delay) works on real spikes 6-seed, and the place-field map that feeds it is now itself learned (11/12 cells tile distinct places, 6-seed).
- Sleep-replay systems-consolidation with **no catastrophic forgetting** (strict controls); PE-gated reconsolidation in-place update 6/6 to K=24.

**Affect core + neuromodulators (🟩 dissociation / 🟨 graded is the boundary).**
- **Neuromodulator affect axes DISSOCIATE, 6-seed GO** (mood/5-HT, arousal/NA, eagerness/ACh each maximally driven by its own driver, unique-permutation, lesion-collapse) — `2026-08-02-laneA-affect-axes-DISSOCIATE-*.md`.
- **Persistent affect-STATE region** (slow-NMDA opponent attractor, Namburi-Tye) holds a mood that *causally* biases recall and speech, 6-seed — **but QUALIFIED**: it's a bistable good/bad latch, not a graded valence×arousal circumplex (an earlier "graded" claim was retracted as a 40 ms probe artifact). `2026-07-24-P0.3-affect-state-region-6seed-GO.md`.
- Active-clear (quench_fs GABA_A) mood eviction 6-seed GO; affective theory-of-mind (separate OTHER-tagged region) 6-seed GO; shared limbic-core RPE battery δ=r−V 6/6.

**Curiosity / drive (🟩).**
- **DR-1 curiosity inversion, on-bridge spiking 6/6** — a spiking ASK pool + spiking-SNc critic learns on a learning-progress reward; corr(gap, want)=+0.991; it *vetoes* unlearnable noise (stops asking while novelty stays high — curiosity, not novelty-chasing). `2026-07-23-DR1-curiosity-inversion-6seed-GO.md`, `2026-07-30-lane-B-curiosity-DR1-onbridge-6seed-GO.md`.
- The honesty **veto moved onto the substrate**: a spiking reward-OMISSION circuit (RMTg→LHb-like omit detector→plastic cue→veto), 6/6 core, omission-lesion collapses the veto — `2026-08-01-curiosity-reward-omission-veto-spiking-circuit-6seed.md`. (An honest negative is banked next to it: the veto *cannot* be read off the spiking striosome value — it inverts.)

**Self-model / metacognition / honesty (🟩 correlates / 🟧 the load-bearing read is NEGATIVE).**
- **No-confab moat** as a learned familiarity gate (Bogacz-Brown anti-Hebbian) matches the host abstain decision 168/168 all seeds, zero breaches, lesionable — `2026-06-11-familiarity-gate-v320-GO.md`.
- **Self-schema region** reads/reports the brain's own attention (0.974), confidence (Spearman +0.980), authorship (1.000); self-lesion→chance and shuffled-signal→chance prove it reads *real* internal state — `2026-07-23-DR3-self-schema-region-6seed-GO.md`. Authorship/agency source-monitor (corollary discharge) 6-seed GO.
- **Honest boundary you must not paper over:** genuine metacognitive sensitivity (meta-d′) is a *measured 6-seed negative* — meta_d≈0 on 4/6 seeds because the monitor reads winner-*rate* (a wrong winner also fires strongly = margin-blind). The named fix (read winner−runner-up margin) is not yet built. And **production abstain is still a host `if`**: `sim/constrained_realize.py:30-34` and `sim/grounded_decode.py` call `research.runners.abstention_gate.gate(ranked, threshold=650)` byte-unmodified — the validated self-schema does not drive it.

**Concepts / language / reasoning (✅ emergent in slices / 🧩 fluency leans on the crutch).**
- Categories discovered from experience incl. real vision (EMERGE-34 held-out perceived object inherits a taught property, 6-seed, per-image pixel-scramble collapses to chance); invariance-from-temporal-continuity (Földiák) 6-seed GO; a word-meaning cortex learned from listening; inheritance/exceptions/transitivity all *emerge* from overlapping codes + a next-state predictor.
- Comprehension: a reservoir that learns the word-order→role mapping itself; long-distance dependency across ~33 words in spikes; a neural question-type router.
- Production: **grammar self-organizes from a corpus**; every word (content + function) spelled out from spikes (EMERGE-23 grounded production 6/6: grounded 1.00, generalized 1.00, novel-abstain 1.00, confab 0; full-frame fluent speech on spikes 4/4).
- **First-mechanism open generation on spikes** (no external model, no BPTT at read time): the WKV/home-grown spiking language cortex generates fluent short stories, reads each word in via spike phase, holds context in a slow neural signal — `2026-07-20-gap1-RF-PHASE-ENCODE-*`, `2026-07-20-grounded-fluent-conversation-DE-RISK-5-one-brain-one-process-GO.md`, `2026-07-20-wkv-cortex-biological-learning-CLOSE-*`. **Honest scope:** small model, small corpus, and the WKV cortex's *core* was still pretrained by BPTT on TinyStories — only the task-format read-out adapter is shown to be shallow-readout-learnable over a frozen reservoir.
- **Learned binder (gap#2) — the slot-binder** replaced the fixed FHRR algebra for single/role binding, 6-seed GO, wired into `BrainConversationalAgent` as `composer_kind="slotbinder"` (`2026-07-21-gap2-spiking-learned-binder-6seed-GO-*`). This is one of the few things that reaches deployment. *Residual:* multi-attribute bundling from scratch is a tested negative on point neurons.

**Embodied loop / navigation (🟩 the flagship / 🟧 fully-neural control deferred).** Gridworld agent reaches and re-reaches moving goals from perceived scene through a learned perception→cortex path; place cells emerge from landmarks; PFC working memory holds the goal; the movement commit is the fully-spiking BG commit-burst (`readout_source='spiking_wta'` is the library default). Merged nav+conv co-resident on one bridge. **Accuracy caveat the audit forces:** several older nav headline claims were found wrong/overstated in the mid-2026 audit (a "no steering heuristic / all shortcuts closed" claim where the heuristic was in fact still on; favourable-seed "X% better" figures that shrink on blind seeds). Describe navigation qualitatively; don't repeat the old numbers.

---

### 2. The scaffold ledger — load-bearing host/external shortcuts

This is the burn-down backlog: every place where something *between sensation and action* is done by host Python or an external model instead of neurons/synapses. Under the brain-based-only standard these are shortcuts, each with a **named biological replacement** and a **burn-down trigger**, even when the host calculation is biologically correct.

**Meta:** the project is standing up a formal `research/SCAFFOLD_LEDGER.md` (mandated by the new charter's §7.3) enforced by `tools/gates/` — **it does not exist on disk yet** as of 2026-08-02. Until it does, the table below *is* the ledger content. The newest charter gates (proposed: `scaffold-ledger-entry`, `scaffold-ledger-integrity`, `host-signal-scan`, `template-escape-declared`) will BLOCK a commit that adds a new host-computed reward/value/RPE/argmax-decision or hand-set weight without either a `# SCAFFOLD:<id>` marker resolving to an ACTIVE ledger entry or a `# NOT-A-SCAFFOLD: <reason>` declaration. Write that declaration at authoring time.

#### The load-bearing shortcuts (fix-order priority)

| # | Shortcut (what) | Where (concrete) | Named biological replacement | Burn-down trigger |
|---|---|---|---|---|
| 0 | **External Qwen2.5-0.5B LLM does ALL open-domain wording** — "the one forbidden permanent external model." The brain grounds + gates + verifies behind the moat; a 494M transformer does the fluency. | `research/runners/brain_chat_tui.py:169` `QwenRenderer` (default renderer); `_grounded_lang_integration_derisk.py` `SpikingQwenFaculty` | Substrate-native spiking generative-sequence cortex grown from grounded lived prediction (BPTT-SNN generator demonstrated; 88.6M spiking-forward C1 GO, not deployed) | Substrate LM reaches usable open-prose at conversational vocab and beats Qwen on held-out grounded generation (blocked by the R4 "~4 orders too small" scale wall) |
| 1 | **~21M/6M TinyStories transformer (Generator-F)** as the from-scratch open-prose generator + spiking-forward target | `sim/tiny_transformer.py` | Same substrate-native spiking LM (this is its scaffold ancestor; the 88.6M spiking-forward conversion `ppl_ratio 1.0` C1 GO already exists) | A from-scratch *spiking* LM beats a bigram at conversational scale, then deploy the spiking forward and retire the ANN |
| 3 | **Teacher-as-external-credit-oracle** (the roadmap CRUX) — a Claude/teacher-authored curriculum + ZPD ordering + corrective error on the brain's OWN outputs, which a corpus can't supply; named "the single load-bearing dependency of the whole development path" | `research/runners/_p2_teacher_to_brain_derisk.py`, `_a1_teacher_contingent_eprop_derisk.py` | Internalized deep-credit rule so the brain self-generates the clean error (Sacramento-2018 self-predicting microcircuit + learned feedback / Payeur BDSP), graduating to real-human contingency (Vygotsky ZPD) | Deep-credit matures to GO on spikes (gap#4) **and** Stage-5 scaffold-retirement GO (brain grows from real human dialogue without the teacher) |
| 4 | **Exact-inverse FHRR/VSA binding algebra** — bind/unbind *operations* are spiking (resonate-and-fire + complex synapses) but the *algebra* (self-inverse ±1/phasor, decorrelated-clean-code demand) is a host idealization. It forces FLAT codes → generalization and composition are disjoint tracks. | `research/runners/rf_phasor_composer.py:234` `_bind`/`:282` `_unbind_phases`; `one_brain_composer.py:560` | A learned, dendrite-gated cortical binder that develops from experience over correlated codes (Larkum dendritic coincidence). Single-attribute learned bind is on-spikes GO; multi-attr from scratch is TESTED-NEGATIVE on point neurons | Learned binder over correlated codes fits + generalizes 6-seed — **gated by the dendritic deep-credit lever (#4), currently NOT-GO on spikes** |
| 5 | **Flat host-given concept codes** + a **host log-domain PPMI normalization** standing in for the generalizing circuit; the shipped composer binds semantically-flat distinct codes (can't relate dog/cat) while grounding rides a *separate* PPMI track | composer `concepts=` given per-seed phasor codes; stream-cortex PPMI double-centring read-out | Unify onto ONE grounded code family: perception inherits the stream cortex's generalizing codes via cross-modal Hebbian (ATL hub, Patterson-Rogers, vision→concept already GO); on-bridge per-concept feedforward-inhibition normalization replaces host PPMI | A2 unification lands (learned binder over ONE grounded code family) + on-bridge PPMI-normalization circuit deployed |
| 6 | **Host cue-match scan + first-match routing + the moat answer-vs-abstain decision** (Python `==` loop) — *which* stored fact answers a query and whether to answer is host code. The single largest live default-on conversational host residual. | `one_brain_composer.py` `_scan`/`_find_cued_block`/`query_agent`/`ask_yes_no`; `rf_phasor_composer.py` `_scan_first_match` | A spiking K-way sequencer (`integrated_loop`, K=32 GO — BUILT, default-OFF) that resonates the query against substrate-held blocks + reads winner + abstention synaptically (Lisman-Idiart; Bogacz-Brown moat) | `integrated_loop` default-on migration succeeds without the small-vocab over-abstention that reverted the last flip |
| 10 | **Rich-answer discourse ASSEMBLY** (what to say) — gather/chain/de-dup/thread/follow-up/breadth-walk/verify-drop is host Python; only the ops it *calls* are spiking. No spiking discourse-planner exists. | `research/runners/rich_answer_composer.py:137` `RichAnswerComposer` | A spiking discourse-planner: neural WM-driven content selection over the association memory (Frank-O'Reilly BG-gated WM; GNW broadcast for topic) | Neural WM-driven content selection replaces the gather/thread/stop heuristics 6-seed |
| 12 | **Host-computed nav reward scalar** (`sign(Δeccentricity)` / Manhattan distance) shapes all nav learning; the spiking reward_us→SNc only *delivers* the scalar | `research/runners/g11_bg_runner.py` `run_moving_goal_episode`; `nav_critic_spiking_sc` defaults OFF | Spiking reward from grounded consummation/appraisal computed by neurons (Schultz RPE; LH/OFC value). Spiking US→SNc validated (RPE corr −0.99) | Default-flip the spiking reward path **and** ground the reward in real consequence rather than a host geometry formula |
| 17 | **Host names the engram slot** (pre-assigned engrams; the shipped consolidation write is host-supervised) — memory *structure* is host-assigned, not self-organized | `research/runners/_consol_cortical_store_probe.py` (`metaplastic_alloc=False` default) | Competitive/metaplastic allocation on the substrate — least-claimed slot wins on its own store mass (Fusi cascade; Han-Silva CREB). `--metaplastic-alloc` exists, default-off | `metaplastic_alloc` validated 6-seed and made default |
| 18 | **Host freezes plasticity during recall (`--freeze-read`)** — the recall read is NOT read-only, so Hebbian overwrites the store while reading it; a fresh "CONSOLIDATION WORKS 18/18" GO (commits `eb7f63c2`/`798ff270`) rests on this host toggle | `research/runners/_consol_cortical_store_probe.py --freeze-read` | A neuromodulatory encode-vs-retrieve gate the brain sets itself (Hasselmo ACh; SPEAR theta-phase gating). A native `plasticity_gate` neuromodulator target already exists | Route the read-phase freeze through the ACh/theta `plasticity_gate` (self-set), not a host flag |
| 21 | **Affective valence is a host lexicon seed** (Warriner VAD norms seed the appraisal) — propagation to held-out concepts is *learned*, the seed core is host-supplied; also QUALIFIED (bistable latch, not graded) | `_affect_distributional_tag_derisk.py:187` `opponent_seed(Warriner)`; `_affect_state_region_derisk.py` | Valence grounded in the brain's OWN reward/consequence history (Rolls OFC; Namburi-Tye opponent circuitry). Graded circumplex needs a line/bump attractor with SFA eviction / dendrites | Affect tags acquired from grounded consequence in the lived loop (seed retired) AND graded valence×arousal on a graded attractor |
| 27 | **Merged nav+conv "one brain" is CO-LOCATED, not interacting** (zero cross-synapses) — nav is byte-identical with/without the conv half, the tell that merging added no integrative behavior | `research/runners/nav_conv_merged_bridge.py` | Functional synaptic integration: cross-region nav↔conv pathways that carry behavior (the spoken-instruction COMMAND_GATE is the working 6-seed template) | Cross-region nav↔conv pathways demonstrably carry behavior (lesioning them changes the other half) |

#### The moderate / minor shortcuts (partly closed or lower-leverage)

| # · sev | Shortcut | Status / where | Replacement + trigger (short) |
|---|---|---|---|
| 2 · mod | Ngram/trigram back-off LM as distillation teacher (soft-target oracle) | `sim/ngram_teacher.py` | Self-supervised prediction from the brain's own error (Rao-Ballard); trigger: generator trains without the distilled oracle |
| 7 · mod | Host argmax cleanup / winner-pick | `rf_phasor_composer.py:363/379`; `one_brain_composer.py:253` | Spiking NEF WTA (`enable_spiking_cleanup`, ==argmax multi-seed) — ON in the 320 demo, OFF in rf/numpy default; flip everywhere |
| 8 · mod | Host-held numpy fact memory (facts in a CPU list, re-driven per query) | `rf_phasor_composer.py:163 self.kb`; `one_brain_composer.py` | Substrate complex-synapse store (`enable_substrate_store`, Josselyn-Tonegawa engrams) — HAVE on OneBrain; make default across paths |
| 9 · mod | Host per-query micro-ops (superposition, ON/OFF opponency, `np.conj` unbind) | `rf_phasor_composer.py` bind_fact | `local_reciprocal_unbind` ON in OneBrain; flip on-substrate superposition/opponency on the rf path too |
| 13 · mod | Host value/critic + DA-RPE at low level (`V=reward_ema`, `DA=gain·max(0,r)`) | `g11_bg_runner.py` `enable_neural_critic=False` default | Spiking striosome critic + SNc RPE (validated 6/6); default-flip on episode/CLI/builder |
| 14 · mod | Host grounding projection M (perception→concept matvec) | `navigate_to_compose_then_answer.py:405` | Learned Hebbian cross-modal convergence (ATL hub; vision→concept GO) — wire it in, retire `host_m` |
| 15 · mod | Host-designed V1 Gabor RF weights (innate prior) | `sim/visual_cortex.py:76` `build_v1_simple_weights` | Retinal-wave rate-Hebbian self-org (SAILnet/Olshausen-Field; numpy B1 GO OSI 1.0); on-bridge lift owed |
| 20 · mod | Consolidation = self-replay + retention-retest stand-in (no hippocampal SWR on the conv bridge) | `_longitudinal_develop_loop_gpu.py consolidate` | Real SWR replay on a hippocampus wired to conv cortex via ca1→concept (Buzsáki; Wilson-McNaughton) — OPEN (on-bridge SWR was at chance) |
| 22 · mod | Host curiosity intrinsic reward (learning-progress `g_before−g_after`) + host novelty scalar | `_curiosity_seek_learn_onbridge_derisk.py` | Spiking familiarity-PE circuit whose *change over time* is read synaptically (Oudeyer-Kaplan LP) — the current lane-B build target |
| 24 · mod | Host agent/slot keying for ToM + discourse register | `_affective_tom_derisk.py`; D3 `transmission_gate` | Learned agent-keyed binding + salience pointer (Frith TPJ; biased competition) — ties to the learned-binder frontier |
| 25 · mod | Host op hand-offs (Python glue between cognitive ops, `to_host`+re-kick) | `one_brain_composer.py` `_compose_phases→_write_block→_read_blocks→_select` | One persistent interacting spiking loop, synaptic op→op handoff (megakernel design exists) |
| 26 · mod | Parser→composer hand-off = a Python dict `{role:word}` | `parse_on_slices` returns a host dict | Synaptic parser→composer route (`hear_synaptic` precedent exists); reverted to Python for the nav+conv merge |
| 11 · min | Residual host word-order joins (embedded-clause / adj-noun f-strings) | composer/agent f-strings | Outer SVO order IS neural (`enable_neural_render`); multi-frame order learning OPEN |
| 16 · min | Host argmax motor read-out — **CLOSED-default**, kept only as a benchmark oracle | `g11_bg_runner.py --readout-source` | n/a — spiking commit-burst is deployed |
| 19 · min | Reconsolidation labilization = host midpoint statistic `0.5(mean_same+mean_diff)` | composer reconsolidation | Auto-calibrated / neuromodulatory labilization gate from substrate PE |
| 23 · min | Fixed hand-set feed-forward projections in self-schema/GNW read-outs | `_self_schema_region_derisk.py` `attend[k]` | Learned Hebbian read-out from workspace→schema (Graziano attention-schema) |

---

### 3. The open frontier

These are the genuinely-unfinished capabilities. None is "blocked and abandoned" — each has a named next mechanism, per the one law.

**(a) The deep-credit boundary — thoroughly mapped, deliberately routed AROUND.** A transport-free local rule that trains a *deep multi-layer spiking* net (no backprop, no weight transport) is the biggest lever for the deepest ceilings (deeper composition, a self-taught nav policy). Status as of 2026-08-02, and read this carefully because it's the most retraction-prone area:
- At **rate**, the "fundamental ceiling" verdict was *falsified* — a transport-free rule with chained multi-hop feedback + σ′ + graded credit clears the depth-2 ceiling (6-seed 0.935 vs banked 0.63), and learned (Kolen-Pollack) feedback rescues MNIST depth-4 (0.53→0.88, 6/6). The rate half is unblocked.
- **On production spikes it does NOT work, and the root cause is now measured:** transport-free credit works only if forward weights rotate to *align* with the fixed feedback wiring. That alignment happens on the simpler **LIF** neuron (6/6 seeds) but **NOT on the production Izhikevich neuron (0/6; 4/6 actively anti-align)** — holding task, sparsity, feedback, signal, and operating point identical. It is a *structural* property of Izhikevich credit dynamics, not noise (refuted same-cycle), not surrogate, not weak-learning. `2026-08-02-gap4-FA-convergence-is-the-onbridge-credit-root-cause-6of6-LIF-converge-0of6-izhikevich.md`.
- **Do not re-drill the banked negatives:** learned KP feedback 0/6, temporal-averaging settle-sweep 0/12, and **dendritic/BDSP/burstprop deep credit are tested-and-NEGATIVE repeatedly** (there is a whole finding titled "the real issue is NOT dendrites — the frozen credit signal"; an owner check specifically corrected reaching for "a dendritic rule" as a fresh candidate — it isn't). The only genuinely-untested directions the record names are a **BurstCCN burst-demux** mechanism or a denser task; and the honest strategic framing is that "beat a reservoir with deep credit on the production neuron" is a **deprioritized side-frontier** — the mission-critical emergence engine needs no such rule (it routes through fixed-reservoir + shallow-readout + learned-input rep, and unsupervised + one-shot BTSP + replay + three-factor gating). This is why the **teacher-as-credit-oracle (scaffold #3)** is load-bearing: it bridges deep directed credit during development for the few faculties that truly need a learned forward model.

**(b) Grounded integration — the "someone home" gap (the charter's real target).** Almost every validated faculty is a *separate probe on a separate bridge*. The open work is running them as ONE live interacting loop:
- **Language is not yet an action in the loop.** In every embodied loop the brain perceives→grounds→stores and is then *queried*; the reply is rendered by the external generator. Speaking is not BG-selected, lands in no world with a consequence, returns no reward. The near-term build (Faculty 7) is to make SPEAK a second BG action channel beside MOVE, with a contingent teacher consequence (confirm→DA burst; correct→ATL) — reusing arcA contingent-teacher e-prop (5/6, `2026-08-01-arcA-*`), concept-pool→language_output speech (4/4), and the clean contingency lesion.
- **Intent is external.** The trigger to speak is a host-parsed query string or a given cue; nothing generates speech from the brain's own affect/curiosity/world-model state. The near-term wiring: intent source = affect/curiosity core, realization = EMERGE-23 speech read-out, learning = contingent-teacher e-prop, with an **intent-lesion** anti-cheat (remove the internal state → speech doesn't fire).
- **GNW is a separate offline region** (ignition/deliberation 6-seed GO on reasoning tasks) not yet the single integrator wired *inside* the running perceive→ignite→select→act→consequence loop.
- **Fully-neural nav control is a documented NO-GO in its stripped config** (spiking-SC + neural-reward→SNc→critic→actor navigates ~58× worse, actor goes silent) — diagnosed as a loop-closure + operating-point problem (`enable_cluster_a_closed_loop` reentrant arc default-OFF), point-neuron-closable with a finite honest-negative cost, not yet recovered to deploy.

**(c) Continual learning from experience — validated as a mechanism, hollow at real-experience breadth.** No-catastrophic-forgetting continual learning is 6-seed GO but only in the **separable/curated small-scale regime** (develop-loop plateaus ~24 vocab / 11 facts; ensemble ~320). On the HARD real corpus, natural-learning-over-time plateaus at ~+0.35 vs an offline +0.52 optimum — the off-diagonal decorrelation a generalizing cortex needs is unreachable by any tested local point-neuron rule (`2026-06-15`). Growth is also largely host-orchestrated (`sim/auto_growth.py` TierPromoter is checkpoint-reload rescaling, its own docstring calls it a "two-class scaffold"). The grounded path that avoids the deep-credit wall: run the unsupervised + one-shot-BTSP + replay-consolidated + three-factor-gated + grow stack as ONE lived loop, burning down the recall-freeze (#18) and the runner-side grow-injection first.

**(d) Compositional / relational consolidation — stranded in hippocampus.** Consolidating a *composed* fact (a whole structured fact, not a single item) hippocampus→cortex is NOT closed. The write itself works at a corrected operating point (18/18 fact-seeds), but the CA1 code is dense/overlapping (~90% co-active per fact) so no ca1→concept write rule can localize which cortical memory each fact claims. Slot-based allocation hits a ceiling by ~12 facts across three variants (two predicted-better ones did worse). The 2026-07-29 re-route: stop building a better slot-picker — go **shared-population sparse coding** (a simplified model stored 200 facts at 92% recall with no allocation step); next step is building that on the real substrate. `2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md`. The recent 18/18 GO also rests on the host frozen-read (#18) and still owes a recall-scramble control.

**(e) Graded developing affect.** The persistent affect region is a bistable good/bad latch, not a graded valence×arousal circumplex — the named crux and highest-information next move (lower recurrent NMDA gain / add heterogeneity so mood is an adjustable bump; Durstewitz-Seamans dual-state). Downstream of it: driving mood/arousal from the dissociated neuromodulator axes instead of a hand-set appraisal concentration, a Barrett discrete-emotion read-out, interoceptive grounding of core valence, and a neural OCC appraisal circuit (today appraisal is a host scalar injected via `_set_appraisal()`).

**(f) Open generation scaling (the R4 wall).** Home-grown fluent speech on spikes is *mechanism-demonstrated* at small scale; matching an LLM's breadth is a measured data-and-scale climb ("~4 orders too small"), plus the honest field-wide wall that fully model-free open-domain chit-chat is unsolved by anyone unconstrained — managed here (as LLMs do) by staying grounded, on-topic, and abstaining. The interim is to keep the Qwen crutch small and behind the moat while the substrate LM scales.

**Also genuinely open (breadth, build when a downstream need calls):** deeper visual hierarchy + a separate location stream (the deployed `cortex_v2→cortex_it` STDP ventral path is **inert/RETIRE 6/6** — no operating point both propagates and stays selective; the fix named is wiring in the divisive-normalization + homeostasis primitives that already exist in `sim/regions.py` but sit as guarded no-ops); audition/somatosensory/interoception (vision is the only modality); the explicit spiking value critic; a fear/aversion system; a theta-rhythm pacemaker; finer NREM/REM stage generators; the global action-cancel pathway; grid cells + path integration; the spiking slot buffer for recursion depth (>~3 is the human-faithful bounded limit, a feature not a bug).


---

## 11. First actions for you (GPT), in order

1. **Read** doc #2 (charter) and doc #3 (board top block) in full. Skim `ROADMAP.md` for the shorthand glossary.
2. **Stand up the discipline before building anything.** Confirm you can run the research gate, the RAG, the doc
   checks, and the commit gates (commands in the Tooling section). Adopt equivalents of the gate system + the
   research-first + no-stall + functional-role disciplines from the start — that is the explicit point of this
   handover.
3. **Run the current gate suite** to see what blocks a commit, and read `docs/FAILURE_GATE_MATRIX.md` +
   `research/FAILURE_LOG.md` so you inherit the scar tissue instead of re-earning it.
4. **Before any capability work**, run the corpus check (`bash tools/before_you_build.sh "<defect>"`) — the record
   already contains a huge amount of concluded/refuted work; re-deriving it is the single most common waste here.
5. **Pick up the mission-critical thread**, not a narrow faculty polish: the SHORT-timescale crux is **grounding +
   integration** — a minimal world+body+loop and language re-pointed as communicative action. The board's top block
   holds the exact next action; if it points at older faculty work, re-judge it against the charter first.
6. **Keep the goals/roadmap/board synced** as you work (the sync-documentation discipline), and **maintain the
   scaffold ledger** — every new scaffold gets an entry with a biological replacement + burn-down trigger.

Welcome. Build the loop, ground the meaning, keep it honest, and don't let a passing test convince you a faculty is
done.
