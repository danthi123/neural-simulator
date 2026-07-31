# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Repository**: https://github.com/danthi123/neural-simulator

## ⭐ ACTIVE MISSION (2026-07-23 PIVOT) — read [`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md) + [`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md) (CURRENT STATE) FIRST, EVERY session

> **To SKIM (plain-language status surface):** read [`ROADMAP.md`](ROADMAP.md) — what's done, what's in progress, what's left, in prose, no shorthand. Its "Project shorthand" table decodes FHRR/BTSP/BDSP/GNW/gap#N/DR-N/RANK-N/EMERGE/the-moat/the-composer. The MASTER ROADMAP is the forward-looking PLAN; `GAP_CLOSURE_MISSION.md` CURRENT STATE is the live RESUME point.

**MAJOR DIRECTION PIVOT (owner, 2026-07-23).** The north-star is now a sim-brain that **CONVERSES GENUINELY** — reasons
to its OWN conclusions, with an **affective world-model + emotion + self-awareness + curiosity** — NOT fact-recall/RAG,
NOT LLM plausible-text. **Success is defined as TRUE CONSCIOUSNESS on the emergentist bet** (it emerges when a human
brain's full faculties + behavior are emulated COMPLETELY + FAITHFULLY enough), so the job is **completeness +
faithfulness of the biological emulation**, not a benchmark score. It is developed via a **TEMPORARY AI-teacher
scaffold** (accelerates early growth) that then **graduates to real-human interaction**; every scaffold is biologized
toward the one spiking brain (transformer minimized/retired). **HARD RULES: (1) DO NOT DEFER any functionality —
surpass EVERY wall with a mechanism rooted in real biology (no "characterized limit" / "honest negative" as a stopping
point); (2) speed is SECONDARY (slow-but-faithful biology — dendritic credit, seconds-long BTSP plateaus, sleep-replay
consolidation — is explicitly in scope; never trade faithfulness for speed); (3) ONE spiking substrate (host code only
for world + body + teacher-as-social-environment).** The **honesty boundary is a DELIVERABLE, not a caveat**: build +
measure every functional consciousness / self-model / affect CORRELATE, design every self-report as an honest
functional read-out ("my familiarity monitor reads this as novel, so I'm uncertain"), and NEVER assert phenomenal
experience.

**⭐ THE PRIMARY PLAN IS `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`** — a LIVING staged faculty roadmap
(complete faculty map · one-brain architecture · 6 developmental stages proto→human-ready · a 14-wall ledger where each
wall carries its biological surpass · a parallelization map). Foundation: `docs/plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md`.
**`GAP_CLOSURE_MISSION.md` remains the session-by-session working board** — its CURRENT STATE opens with this pivot
block + the live compute lanes; read the roadmap for the PLAN and the board's CURRENT STATE for the RESUME point. **The
prior "close the 5-gap cluster" framing is SUBSUMED** — the 5 gaps live on as the roadmap's faculty-map + walls-ledger
(still valid, now a sub-view), no longer the top-level mission.

**THE LAW (unchanged, now applied to every wall in the ledger): a wall/negative is a verdict on a METHOD, never a
license to abandon a CAPABILITY — bank the failing method, take a new biology/spiking/one-brain method, and keep going
until it WORKS. Closure cannot be deferred.** **SESSION START — OR ANY CONTINUATION (resumed from compaction): VERIFY a
within-session anti-stall + RUN-STATE heartbeat Monitor is live (a prior session's died with it; a continuation usually
has NONE), and arm one if not. It must be STATE-CHECKING (emits GPU / running-procs / recent-output every ~15 min) — a
text-only "are you idle?" nudge is insufficient (the 2026-07-24 failure was a live-but-stalled run, not idleness). Exact
recipe in `GAP_CLOSURE_MISSION.md` → "SESSION START". NEVER WAIT on a background run without a live state-heartbeat, and
never trust a subagent-armed Monitor / passive re-invocation to catch a completion. Then resume from CURRENT STATE.**
Cross-session continuation is MANUAL by owner choice (a plain "continue" + the roadmap + that board re-anchors) — no
watchdog/daemon.

## Keep the SUMMARY docs synced when a finding lands (2026-07-24 drift → the `sync-documentation` skill)

**Committing a finding is NOT enough — the summary docs must move WITH it, SAME cycle (deferring the sync IS the drift;
it happened all of 2026-07-24: findings committed, board left stale).** When a committed finding changes a wall/gap
STATUS, the CURRENT FRONTIER, or a "next action", **run the `sync-documentation` skill** — it now does BOTH the mechanical
drift (line counts / runner-test-findings counts / g11 flags / `sim/__init__` exports) AND the semantic summary-doc sync
(roadmap §7 wall-ledger + [`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md) CURRENT STATE + [`AUTONOMOUS_STATE.md`](research/findings/AUTONOMOUS_STATE.md)
+ `ROADMAP.md` status/frontier/next-action, contradictions, banners, the plain-language header + shorthand glossary). A
PostToolUse hook nudges it on `sim/` / `research/runners/` / findings changes — RUN it, don't just acknowledge the nudge.
Stale pointers are drift #12, the #1 cause of re-deriving concluded work.

## One term, one meaning ([`docs/TERMS.md`](docs/TERMS.md))

**Before writing `consolidation` · `compositional` · `self-organized` · `closed` · `GO` · `fully spiking` · `byte-identical` · `lesion` · `selective` · `works` in a finding, commit message, or board entry, check its CODE CONDITION in [`docs/TERMS.md`](docs/TERMS.md).** Earned 2026-07-28: three of nine retractions in one session were pure terminology overclaim with correct, reproducible measurements underneath — an experiment called *consolidation* whose replay branch never executed, *compositional* over a localist code, *self-organized* while the host supplied both factors of the learning rule.
An unchecked term is a HYPOTHESIS, exactly like a claim in a comment. (ASD-STE100's one-term-one-meaning discipline, scoped to the ~10 words here that carry load; the full spec was assessed and not adopted — see the file's notes.)

## Document structure ([`docs/WRITING.md`](docs/WRITING.md)) — two checked rules

**W1** a voided document is registered in [`docs/RETRACTED.md`](docs/RETRACTED.md), and no governed file cites it without `⛔` on the same line. **W2** prose lines in governed files are <=800 chars (tables and code exempt) — this is a PRECONDITION for W1, proven empirically: splitting one 14,222-char board line exposed two stale citations that had been 'marked' only by a `⛔` sitting 13,000 characters away.

Check both: `.venv/bin/python tools/check_docs.py` (CI: `tests/test_doc_rules.py`). Retrofit helper: `tools/split_long_doc_lines.py --apply` (splits at sentence/`·`/`;` boundaries and refuses to write if content changes). **These are STRUCTURE rules only — they cannot catch instrument failures; six of the nine 2026-07-28 retractions would have passed both.** Truth verification is `verify-go`; term conditions are [`docs/TERMS.md`](docs/TERMS.md).

## ⭐ THE WORKFLOW IS NOW ENFORCED, NOT REMEMBERED (2026-07-31) — read [`docs/FAILURE_GATE_MATRIX.md`](docs/FAILURE_GATE_MATRIX.md)

**Everything below this section that reads as a RULE YOU MUST REMEMBER has, where possible, been converted into a
CHECK THAT BLOCKS.** Prose is kept for the reasoning; the enforcement is in code. If a rule below and a gate
disagree, the gate is authoritative — it is the thing that actually runs.

**One entry point:** `tools/gates/` — one module per failure class, auto-discovered, wired into
`tools/githooks/pre-commit`. **Adding a class is one file; the hook never changes.** The registry REFUSES to
trust a gate whose `selftest()` does not fail in its failing direction, because four checks here have shipped
unable to fail (a `;` where `&&` was meant, a pipe eating an exit status, a nonsense query scoring 18 hits and
PASSING).

**What blocks a commit:** document structure (W1/W2) · claims not traced to a cited artifact · biology bindings
(source anchors must resolve; config must not contradict the biology) · one-mechanism-one-current-status ·
undeclared finding status · doc type/placement · single-seed headlines · wrong-quantity comparisons ·
artifact provenance · CPU-lane starvation · agent-level serialisation · a NOTICED failure left unclosed.

**A noticed failure cannot stay unclosed.** Add one line to [`research/FAILURE_LOG.md`](research/FAILURE_LOG.md)
and `gates/coverage` BLOCKS until it names a gate or declares `NOT-GATEABLE: <reason>`. Noticing is judgement;
closing is not.

**Provenance is automatic.** `research/runners/__init__.py` runs on every `-m research.runners.X` and records
argv, git SHA and the env vars that have silently changed results here, then sidecars every artifact the run
created. No runner was modified. (94% of 7127 artifacts previously could not say what produced them.)

**Biology is recorded once, not re-researched.** `research/biology/<id>.md` binds a mechanism to a source with a
quote that must still RESOLVE, plus `constraints_config` — config values the biology REQUIRES.

### ⛔ THE DEEPEST LESSON, and the first question to ask at any wall

Four causes of friction, each measured 2026-07-31
([finding](research/findings/2026-07-31-why-we-hit-walls-the-missing-companion-process.md)): biology runs
INTERACTING processes and we implement ONE, substituting a static bound for the rest — **and the proxy dominates**
(97% of a gap#5 weight change was the CLAMP); the OPERATING POINT is implicit in the animal, so tuning optimises
whatever the metric rewards; the PROTOCOL is part of the mechanism and no paper writes it down (BTSP is one-shot;
five laps erases the field); and we usually cannot tell WHICH, because the instrument does not exist yet.

> **At a wall, ask "what else does the real system run alongside this, that we replaced with a constant?" BEFORE
> "what biology surpasses this?"** — the answer is nearly always a homeostatic or competitive process we proxied
> with a bound. And: **the instrument is part of the emulation.** A mechanism you cannot measure correctly is one
> you will tune in the wrong direction, confidently, for weeks.

## Drift prevention is MECHANICAL, not remembered (2026-07-28: 9 retractions in one session)

**A rule you must remember is not a mechanism.** `verify-go` rule 3 was written and violated the same day; the corpus-first rule was in this file and skipped for six levers; the parallelize memory existed while the GPU sat at 0%. What actually held was executable: a physiological gate that printed VOID, `tools/check_docs.py` (found 3 stale citations), `push_both.sh` (verifies rather than claims). **Prefer converting a rule into a check that can FAIL LOUDLY over adding prose here.** Three such checks now exist:

- **Before the first lever against any defect:** `bash tools/before_you_build.sh "<defect>"` — runs the corpus query, lists existing research gates, and prints THIS ARC's own exclusions. (A 497-line gate for the identical defect was 2 days old and re-derived the hard way; a fix was later built on a variable the same findings doc had already measured inert.)
- **Inside any probe:** `from tools.lab import lever, before_after, undefined_if_empty, void_if` — makes lever-verification, before/after measurement placement, and "UNDEFINED, not a score of 0" execute instead of being recalled. Each helper names the retraction that earned it.
- **The session heartbeat** now flags **serialization** (GPU idle with room for ~5 more runs, 36 idle pool cores) as an explicit ACT line, not just liveness.

## Parallelize ACROSS ROADMAP LANES, not just fully — `tools/lane_check.py` (2026-07-29, owner-flagged twice)

**A full queue and a busy GPU look exactly like good prioritization from the inside, and are not.** On
2026-07-29 the GPU ran at 100% with a stocked queue for hours while **every job served ONE lane (H · Memory)**,
the roadmap's own crux (**F · gap#4**, *"the single load-bearing dependency"*) had ZERO allocation, and the
five **[CPU]** lanes — explicitly *disjoint*, free to run beside GPU work — sat unqueued next to 36 idle pool
cores. Run **`.venv/bin/python tools/lane_check.py`** when stocking a queue: it maps every running/queued job
to a roadmap lane and exits 1 on monoculture, an unserved crux, or no CPU lane. The heartbeat reports it each
cycle. Momentum substitutes for prioritization silently; this makes it fail loudly.

## Evolve the workflows themselves (the `evolve-skills` skill)

**When a process lapse RECURS** (the owner had to catch the same *class* of problem twice), at a **session-end /
pre-compaction inflection**, or **when the owner asks** — run the **`evolve-skills`** skill: it reviews (with evidence)
what's WORKING and what's RECURRINGLY FAILING in our workflows, then makes INCREMENTAL updates to the applicable skills
so the workflows compound instead of re-learning the same lessons. Grounded, honest, lean (edits the on-demand skills,
never CLAUDE.md/memory bloat). A caught lapse IS a skill gap — evolve the skill so it can't recur, don't just patch the
instance. (Born 2026-07-24 after the owner caught three process lapses in one session.)

## When Compacting (custom compaction instruction — must survive into every compaction)

When this session auto-compacts or `/compact` runs, the summary MUST preserve (and may drop everything else to fit):
- **The ACTIVE MISSION block + the non-negotiables** (brain-based-only, one-brain, no-defer, speed-secondary, the honesty boundary) and the pointers to [`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md) + [`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md).
- **The current frontier + the exact next action:** the wall being worked, its GO-gate command + the anti-cheat controls, and the literal next command to run.
- **Live background work:** every running run / workflow / subagent ID + its Monitor, and every uncommitted result awaiting a verdict (so nothing is lost or double-launched).
- **Files created/modified this session + their purpose**, plus any `NO sim/ edit` / additive-default-off scope flags.
- **Owner directives given this session** — verbatim intent, not a paraphrase.

Summarize aggressively (keep only what changes a decision): git log, verbose run logs (error lines only), search-result dumps, exploratory file reads. Preserve any test / benchmark / GO-gate command VERBATIM.

**Context hygiene (2026-07-23):** history lives in [`docs/project-history-archive.md`](docs/project-history-archive.md) (RAG-indexed, `--corpus doc`), NOT inline — retrieve it, don't reload it. Prefer `/clear` between unrelated arcs (nav → conversation → gap#5) over one mega-session; offload heavy reading/search to subagents (their context doesn't count against the main window).


## Standing practice: deep research + catalog review FIRST at roadblocks and new directions

**(2026-06-07, owner directive — make this the default first step, not an afterthought.)** Whenever the project hits a **significant roadblock** (a multiply-confirmed boundary / repeated NEGATIVE) **OR is about to begin work on a new part of the sim**, run a **deep research + reference-catalog review BEFORE committing build/GPU resources.** This has repeatedly been the decisive pivot:
- the conversational decorrelation/whitening blocker → reframed by the Mikulasch-Priesemann point-neuron limit (whitening is analog/pre-spike in biology);
- the navigation action-selection readout boundary → diagnosed as a *missing accumulator* (Wang 2002 NMDA attractor → Lo-Wang commit burst), which fixed it;
- the navigation perceptual cold-start → root-caused as a **wrong-pathway** problem (routed through the position-*invariant* ventral "what" stream / IT instead of the dorsal "where" stream + superior-colliculus orienting + place cells) via the catalog + Kandel + literature.

**The pattern (LOCAL-FIRST — 2026-07-23 repair: the local corpus + RAG had rotted to dead `E:` paths, so the workflow silently fell back to online search; paths fixed + this is the mandatory FIRST move):** the FIRST move is our OWN local corpus via the auto-updating RAG index — `.venv-rag/bin/python tools/rag/rag_search.py "<question>" 5 [--corpus finding|plan|doc|catalog|kandel|paper|all]` (hybrid vector+BM25 → cross-encoder rerank; auto-refreshes on commit; SOMA retired).
It spans our findings/plans/docs PLUS the canonical biology catalog (`~/Projects/sim-catalog/references/feature-catalog.md`, ~323 entries across clusters A–Q, the separate `sim-catalog` worktree), Kandel 6e full text (`~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt`), and 7 `.txt`-readable specialty textbooks/papers (Marr, Albus, Buzsáki, O'Keefe-Nadel, Schultz, Sutton-Barto, Bolam/Tepper BG — under `~/Projects/sim-catalog/references/textbooks/`), plus `references/glossary.md`. The RAG LOCATES; then READ the surfaced source in depth (a rerank hit is a pointer, not a paraphrase). ONLY after the local corpus is exhausted go external (WebSearch + the `bio-research` MCP).
A read-only research subagent may run this and produce a findings doc: **diagnosis → ranked biologically-grounded options → what existing project machinery is reusable → a recommended cheap-first de-risk → the anti-cheat controls it needs.** The controller reviews it (trust-but-verify the load-bearing claims), pushes the doc, and presents the recommendation before building. Treat this as the standing opening move for roadblocks and new-direction work.

**The research gate — the AUTOMATIC trigger (2026-06-20, owner directive — make it mechanical, not a judgment call I can rationalize past; it failed once because "is this a significant roadblock?" is rationalizable).** Before committing ANY build / GPU / `sim/`-edit effort to *overcome* a difficulty, the gate fires (dispatch the read-only deep-research subagent FIRST, present its ranked options before building) if **ANY** of these objective conditions hold:
- **(a) Confirmed boundary:** an experiment/de-risk returned NEGATIVE / BOUNDARY / NO-GO / "walls" / "can't on this substrate," and the next move is a mechanism to push past it.
- **(b) Known family:** the wall is the same family as a prior documented boundary (the graded-magnitude / divisive-normalization / rate-code / point-neuron-limit / whitening family) — even on the FIRST occurrence in a new place.
- **(c) Blocks a goal:** the difficulty blocks a stated roadmap/goal item (not a side-nicety).
- **(d) New mechanism:** about to design a mechanism *class* not previously built (vs. composing already-proven pieces).
- **(e) `sim/`-to-overcome:** the candidate fix edits protected `sim/` code specifically to push past a limit.
- **(f) Stuck:** ≥2 distinct approaches to the same goal have failed.

**The self-check (the exact failure to prevent):** the moment I write or read a verdict containing NEGATIVE / BOUNDARY / NO-GO / "walls" / "can't" AND my next instinct is "scope/build the fix" — *that instinct IS the trigger.* The next action is the research gate, and the fix I had in mind becomes just ONE option the research ranks (it is never the default).

**The SURPASS sharpening (2026-06-20, owner directive — after a single owner sentence + ONE deep-research round overturned a too-comfortable "closed as a structural primitive" verdict AND found a cheap fix the controller had missed).** The gate fires not only before BUILDING a fix but **before ACCEPTING a boundary** — and "boundary" includes the SOFTER comfortable verdicts that quietly END investigation without a fix: "structural primitive," "honest negative," "not a shortcut," "the cost IS the deliverable," "characterized limit," "defensible," "that's just how the substrate is." Those are **DISGUISED boundaries** and are exactly where over-comfort hides.
**Extended self-check:** the moment I write ANY conclusion that ends investigation of a difficulty *without a fix* — the hard NEGATIVE/BOUNDARY/NO-GO *or* the soft it's-a-primitive / honest-negative / not-a-shortcut / defensible — that IS the trigger. **The surpass deep-research round is MANDATORY and has FOUR moves (not just "diagnose + rank options"):** (1) **ISOLATE + QUANTIFY the genuine residual** — how big is the truly-irreducible part? Usually most of the "blocker" is already defensible or solved and the genuine residual is TINY (the FHRR-B "host-designed binding structure" was, on inspection, a single local `conj()` call; the rest was random-developmental codes + learned codes).
Never accept a vague "the structure/op is host/hard" — pin down EXACTLY which bytes are the residual and measure them. (2) **REFRAME via "how does REAL biology actually do this?"** — am I testing the WRONG hypothesis? (we'd tested "can the bind be LEARNED from task data," which fails — but biology DEVELOPS the structure from local wiring rules, a different category with a cheap answer). (3) **RANK cheap-first SURPASS mechanisms** — the cheapest path PAST it, not merely a diagnosis. (4) **Verdict: surpassable-and-how-cheaply, vs genuinely-irreducible-and-precisely-why-defensible.** A boundary is accepted ONLY after it SURVIVES this round; the comfortable verdict is the START of the research, never the end.

**⚠️ THE LOOPHOLE THAT DEFEATED THE GATE (2026-07-26): a SEQUENCE of individually-cheap config tests IS a build effort.** Six levers / ~4 GPU-hours were spent against ONE defect without the gate ever subjectively firing, because no single flag felt like "committing build effort"; the research round then resolved in one pass what the sequential guessing had not. **MECHANICAL: ≥2 distinct levers tested against the SAME defect without resolution ⇒ the gate FIRES.** Cheapness of the next step is not an exemption — the quantity that matters is cumulative effort against one difficulty. Details + the measurement-placement rules in `.claude/skills/verify-go/SKILL.md`.

**Does NOT fire (so the gate stays calibrated, not over-triggering) — proceed directly:** routine/mechanical bugs with a clear cause (a backend mismatch, an off-by-one, a crash with an obvious fix); engineering that *composes* already-de-risked mechanisms; the GPU / multi-seed *confirmation* of an already-de-risked result; documentation, refactors, frontend wire-up. When genuinely unsure whether the gate fires, it fires (the read-only research is cheap relative to building the wrong fix).

## Standing standard: BRAIN-BASED ONLY (neurons / synapses / their communication), or it is a shortcut

**(2026-06-08, owner directive — the load-bearing bar for "a proper brain analogue".)** Anything NOT done directly by the simulated brain — **neurons firing, synapses, and the communication between them** — is a **cheat/shortcut, EVEN IF the host-side calculation is biologically correct.** A prediction error computed by a Python formula, a "reflex" that reads pixels and returns a cardinal in code, a reward computed by a distance formula, an argmax over spike counts — all are shortcuts, because the *brain* is not doing them; the simulation's bookkeeping is.

**The boundary — host code is legitimate ONLY for:**
1. **The environment** — the world's state (agent/goal positions, the grid) and rendering the agent's sensory input (the retinal image the neural retina then receives).
2. **The body** — the agent acting on its motor output (moving based on which motor pool fires).

**Everything between sensation and action is the brain's job and MUST be neurons/synapses:** perception/salience, orienting decisions, reward, value, dopamine/neuromodulators, action selection. When a capability is realized by host computation (even biologically-shaped), it is a documented shortcut to be converted to a spiking/synaptic mechanism — and an **honest negative** (the neural version underperforming the host shortcut) **IS the scientific deliverable** (it maps what the substrate can/can't do on its own). Applies **PROJECT-WIDE** (navigation AND the conversational pipeline — e.g. the VSA composer's clean exact-inverse algebra is a host shortcut for what a learned cortex would do; see the "composer-as-idealization" note).
**Re-classification:** the recent nav wins (N1 SC reflex, N5 perceived reward, N6 thal/argmax readout, N9-step-1 scalar RPE) are biologically-*shaped* but partly **host-computed → they are now shortcuts**, with their spiking/synaptic versions (a spiking superior colliculus, a neural reward/value system, a spiking SNc, a neural position code, a minimal motor read-out) the real target. The host versions become the *teaching scaffolds* for their neural replacements (the innate-reflex-teaches-a-learned-circuit pattern).

> _Archived: **Recent-arc narratives** (was CLAUDE.md L76-236) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
## Common Commands

```bash
# Run headless auto-tuning (parameter sweep)
python neural-simulator.py --auto-tune
python neural-simulator.py --auto-tune --quick  # Faster reduced sweep
# Run a research-gate runner (G1..G11)
python -m research.runners.g11_bg_runner --moving-goal --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/g11_seed42.json
python -m research.runners.g11_bg_runner --probe-action W   # static cascade probe

```

## Architecture


### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation (fully isolated)
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`) for inter-thread messaging

### Config gotchas + the plasticity BOUND TRAPS (`sim/config.py`)

  - Inhibitory reversal: `E_inh = -75mV`, propagation scaled 0.7x for driving force compensation
  - HH numerical stability: dt auto-adjusts to 0.05ms when HH model selected
  - **Per-gate Q10**: `hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5` (fixed 2026-04-25 — uniform Q10=3 over-compressed dynamics at 37°C; see Phase A below)
  - **STDP bounds gotcha**: `stdp_w_max=2.0` default. The STDP rule is **soft-bound** (`Δw_LTP = A_plus * (w_max - w) * exp(...)`) so when `weight_mean > stdp_w_max`, every "LTP" event is strongly negative and weights collapse to w_max within ms. Set `cfg.stdp_w_max` above your design weights (e.g. cortex→D1 in Phase B uses `weight_mean=25` → set `stdp_w_max=30`).
    **⚠️ THIS TRAP IS PER-RULE AND HAS NOW HIT FOUR RULES — STDP (`stdp_w_max`), BDSP (`bdsp_w_max`, below), BTSP (`btsp_w_max` — saturation silently crushed a rank-1 write to a flat null, 2026-07-25) and HEBBIAN (`hebbian_max_weight` **defaults to 1.0**, far below typical design weights: at a 3.015 pathway every "potentiation" was strongly negative and collapsed the TRAINED and UNTRAINED pathways identically, reading as "the rule doesn't help here").
    **⛔ FIFTH INSTANCE, 2026-07-31 — and the pre-flight below existed the whole time, as PROSE, so it was skipped again.** gap#5's *tuned* operating point ran `w_max=150` against an initial weight `W0=250`: the clamp sat BELOW the weights, dragged every one DOWN, and **97% of the measured weight change was the clamp — identical in the `lr=0` control** (the lr lever moved 3%). The tuning then walked DEEPER in (`w_max` 110→150→220, 150 picked as the "interior optimum"), because what the metric rewarded was clamp depth. Compounding it, `circ_resultant` RECTIFIES, so the `lr=0` arm scored `circ_dW` **exactly 0.000000** at every seed and was quoted as a clean control — while its own mean `|dW|` was **21.94**. An exact zero meant every increment was NEGATIVE, not absent.
    **⇒ THE PRE-FLIGHT NOW EXECUTES — use it instead of remembering this:** `from tools.lab import bound_check, sign_budget`. `bound_check("btsp_w_max", cfg.btsp_w_max, W0)` RAISES when a bound sits at/below its weights; `sign_budget(label, dW)` reports what fraction of `|dW|` a rectifying metric is about to discard. Both are wired into the gap#5 runner and tested in both directions.
    **STANDING PRE-FLIGHT for ANY plasticity rule: compare its bound against the ACTUAL weight (`_mean_gate_weight(bridge, gate)` vs `cfg.<rule>_max_weight`), and verify the trained pathway moves DIFFERENTLY from an untrained control.** A bound below the weights does not merely fail to learn — it destroys weights uniformly, which reads as a substrate limitation.
  - **BDSP clamp-at-lr=0 gotcha** (2026-07-24, commit 6a9a44c3): `fused_bdsp_update` applies `cp.clip(w, bdsp_w_min=-5, bdsp_w_max=5)` **unconditionally — even at `lr=0`** (a frozen/control arm), so any weight outside ±5 is silently flattened to the bound (it collapsed a gap#5 encode store to `bdsp_w_max=5` and plausibly caps gap#4's ±5-bounded FF weights on a 9-way task). Set `bdsp_w_max` above your design weights, and don't assume `lr=0` means "no weight change" for BDSP. A `sim/` clamp-fix (gate the clip by lr / plasticity gain, mirroring the STDP masked-clip) is filed.

**Note on dt Auto-Adjustment**: When switching to Hodgkin–Huxley model, dt is automatically
reduced to 0.05ms for numerical stability of voltage-gated kinetics. When switching to Izhikevich
or AdEx, dt restores to 0.5ms. This occurs in `apply_simulation_configuration_core()`.


### UI-Config Roundtrip
Two critical functions must be kept in sync for profile save/load to work correctly:
- `_update_sim_config_from_ui()`: Extracts all parameter values from UI widgets and builds `CoreSimConfig`, `VisualizationConfig`, `RuntimeState`, and `GPUConfig` dataclasses
- `_populate_ui_from_config_dict()`: Takes a configuration dictionary and updates all UI widgets to reflect those values

These are inverse operations: any parameter exposed in the UI must have a corresponding getter and setter to ensure bidirectional sync between UI state and simulation configuration.

**Built-in target types:**
- `synaptic_gain` — multiplies effective synaptic strength (scope=all only)
- `plasticity_rate` — multiplies reward_learning_rate (scope=all)
- `excitability_drive` — adds pA to membrane drive (scope=all, trait:N, group:NAME)
**Group registration:**
Runners that want `scope="group:NAME"` targets must call
`bridge.neuromodulator_manager.set_group_indices({name: indices})`
after the engine groups are known. G9 runner does this automatically
for the standard input/hidden/motor groups.
- Bridge allocates `region_manager` BEFORE neuron arrays (so num_neurons
  is set from `region_manager.total_neurons()`).
- Wiring is generated by `build_wiring_plan()` and fed through
**Purpose:** Defeats the silent-motor trap (motor neurons that never fire in
phase 1 cannot acquire STDP eligibility, so reward-mediated weight updates
never reach them; agent stays glued to phase-1 winners even when reward
flips sign).
**Two non-obvious bugs that almost killed the architecture** (both fixed 2026-04-25):
1. `n_cortex=400` over-drove D1 to ~220 Hz (saturated, unphysiological), GPi couldn't silence past STN excitation. **Fix:** use `n_cortex=100` (25 cortex/action). The static probe used 100; the moving-goal runner shipped with 400, so the probe "passed" but the deployment failed. Lesson: probes must call the same builder with the same args as deployment.
2. `cortex→D1` weight_mean=25 against default `stdp_w_max=2` collapsed weights from 25→2 in milliseconds via soft-bound STDP. **Fix:** set `cfg.stdp_w_max = 30.0` in the runner.
> **GOTCHA — plasticity gate vs synaptic transmission (2026-04-28):**
> `cp_plasticity_rate_gain` and `set_plasticity_gate(...)` freeze weight UPDATES
> only — STDP, eligibility, Hebbian, synaptic scaling. They do NOT freeze
> synaptic CURRENT (`g_syn × (V - E)`). A frozen pathway with non-zero
> `weight_mean` still injects current and affects forward dynamics. To
> staged-introduce a new pathway without disrupting the system before
> the thaw step, initialize it with `weight_mean=0.0` (then let STDP grow
> it from zero after thaw) — OR add a runtime weight scale per gate
> (small bridge change, not yet implemented). The cheat-5 v1 NEGATIVE
> result (2026-04-28) was caused by missing this distinction; v2 fixes
> it via zero-init.
>
> **UPDATE (2026-06-03): the complement now EXISTS — `transmission_gate`.**
> `RegionPathway(transmission_gate="name")` + `bridge.set_transmission_gate(name, value)`
> scale a pathway's effective synaptic **CURRENT** in [0,1] at runtime
> (the `cp_transmission_gain` per-synapse multiplier in `_run_one_simulation_step`,
> mirroring `cp_plasticity_rate_gain` but on current, not weight updates).
> Pre-wire a route with a fixed weight, hold it CLOSED (gate=0, no current,
> no STDP cold-start), OPEN it on command → **thalamocortical dynamical
> gating**: binding = which gate is open, not which weight grew
> (Logiaco-Abbott-Escola 2021). Validated in spikes
> (`tests/test_transmission_gate.py`): closed → target silent; open → target
> fires; re-binding reroutes the same source with **zero weight change**,
> where grown weights could not. Default `None` = always-on (additive, zero
> overhead unused). See `2026-06-03-deep-research-surpassing-the-blockers-synthesis.md`.
**Usage:**
```bash
# Default (CuPy if available, else NumPy)
python -m research.runners.chat_repl --mode tier1 --seed 42

# Force NumPy backend (Mac M-series, GPU-less Linux, CI)
SIM_BACKEND=numpy python -m research.runners.chat_repl --mode tier1 --seed 42

# Force CuPy explicitly (or fail if unavailable)
SIM_BACKEND=cupy python -m research.runners.chat_repl --mode tier1 --seed 42
```
**Pattern for new code:** instead of `import cupy as cp`, use:

```python
from sim.backend import get_backend, fuse, synchronize, to_host
xp, backend_name = get_backend()

@fuse()
def my_kernel(a, b):
    return a + b  # works on both cupy + numpy backends
```
**Backend selection** (in priority order):
1. Explicit `get_backend("cupy")` or `get_backend("numpy")` (test code)
2. `SIM_BACKEND` env var (`cupy` / `numpy` / `auto`)
3. Cached backend from a prior call (sticky)
4. Auto-detect: CuPy if installed AND `cp.cuda.runtime.getDeviceCount() > 0`,
   else NumPy
and track per-pathway activity each simulation step. Inference still
uses the monolithic `cp_connections`; the store is observational +
foundation for Phase 4 auto-tiering. Per-pathway shards can be
**Known limitation — composer is a principled idealization, not a functional cortex (2026-06-06):** the
FHRR/VSA composer is a *principled idealization* (Eliasmith Spaun / Semantic Pointer Architecture — a
serious hypothesis that cortex binds VSA-like), NOT a functional reproduction of cortex. Its binding is
a clean, exactly-invertible ALGEBRA that DEMANDS decorrelated full-precision codes (the whole whitening
requirement is downstream of this); a real cortex has LEARNED, lossy, redundant read-outs that learn to
read whatever messy code arrives. The binding OPERATIONS are already on-substrate spiking (FHRR
resonate-and-fire + complex synapses); the residual idealization is the exact-inverse algebra + the
clean-code demand. The spike-native robustness ladder (a phase-encoded handoff, b temporal integration,
c population redundancy + attractor cleanup) makes the scaffold spike-FAITHFUL; the genuine-cortical
conversion (d: learned read-outs replacing the fixed algebra) is **BENCHED** below the planned work
(cheat/shortcut removal → single-brain consolidation → capability addition + scaling). NOT labelled a
"cheat," but stay cognizant it is not functionally identical to the cortex it stands in for. Trade-off:
the algebra buys the no-confab moat + compositional reliability ~free; a learned cortex does not.
See `research/findings/2026-06-06-composer-vsa-idealization-known-limitation.md`.
production D=2048 multi-seed; a `plastic=False` population still drifts under global Hebbian, so the
composer's fixed bind population is frozen by a per-synapse plasticity gate, `cp_plasticity_rate_gain=0`).
**The two standalone numpy phasor simulators are REFERENCE-only, NOT the production substrate:**
`research/runners/spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py` (and the unified agents that import
them — `nested_composition_agent` / `spiking_unified_agent` / `unified_agent_*`) carry a NUMPY-REFERENCE
header and are retained only as the FHRR validation ceiling. Do not treat them as "the brain analogue."
- **De-risk 5b (RF vs Izhikevich) — KILL confirmed → the minimal protected edit.** RF stores its complex
  phasor in the same `v`/`u` arrays Izhikevich uses; one Izhikevich step destroys a phasor (|z| 1.0 → 16.3).
  But the composer is stateless-per-op (re-kicks each op) and stores memory in complex synapses, so the
  minimal edit is to **slice the RF ops** (not a core-step-loop dual-dispatch): `rf_kick(..., neuron_mask=)`
  + `_rf_advance_one` mask all `v`/`u` writes to the RF slice. **Default `None` = byte-identical** (18/18
  (a) uses a HYBRID `run_moving_goal_episode` integration (4 additive no-op-default params + an index-based
  `finalize_conv_for_nav_gate` hook that runs AFTER the V1/SC post-init `set_pathway_weights(add_missing=True)`
  CSR rebuild — which re-sorts the data + stales gate-index maps + the Hebbian decay would erode the fixed
  perception weights; the hook handles all three by masking by index, not gate name). The **nav-on-merged
    TEST ORACLE + the numpy-CPU path** (`--composer rf`). NOT flipped (deliberate, safe): the library constructor
    defaults (`BrainConversationalAgent`/`MultiTurnAgent` `composer_kind="rf"`) + the CPU transcript demo — flipping
    those would force GPU on every default agent and break numpy-CPU portability. The bind stays the exact-inverse FHRR
- **UPDATE (2026-06-15) — the GENERALIZING learned cortex is achievable WITHOUT the (B) dendritic rewrite,
  and is REALIZED on the spiking substrate, learned from the conversation stream.** The fork's (B) framing
  ("decorrelate the correlated codes → needs the dendritic rewrite") was superseded by the CYCLE-88 reframe:
  the off-diagonal decorrelation was a **red herring**. A generalizing cortex needs **feedforward LOCAL
  normalization** (PPMI = log + per-hub + per-concept mean-subtraction + threshold, all local ops), NOT
  cross-neuron decorrelation (which would *destroy* generalization). PPMI codes reach host (+0.518) AND
  multi-attribute **bundling** (a fact = a superposition of bindings) is **not learnable from scratch** on the
  point-neuron substrate — additive has no inverse (0.193), a learned *linear* inverse cannot be a reciprocal
  (0.056, breaks even single-attribute), while a **fixed ±1 self-inverse bind bundles 0.989** on the same
  harness (positive control). ⇒ the conversational bind = **learned representations** (codes + single-attribute
  binding, both substrate-validated) flowing through a **fixed, biology-grounded coincidence/multiplicative
  binding primitive** (= the production composer binding the learned codes; binding-by-coincidence /
  dendritic-multiplication is a STRUCTURAL neural primitive — not a host shortcut, and not learnable from
  scratch on point neurons). Finding:
  `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`.
`SIM_BACKEND=cupy` (GPU) is required for the merged-bridge runs (numpy is a tiny-smoke / CI path only).
### 🧠🔗 Cross-region "one brain" FUNCTIONAL interaction + step-3 COMPOSE-PERCEIVED-CONTENT de-risked (2026-06-16)

**Roadmap step 2 merged nav + conversation onto one bridge but they were CO-LOCATED, not interacting** (owner
challenge [[project_one_brain_substrate_vs_functional]]). The cross-region SYNAPTIC interaction (the real "one
- **correlation boundary mapped** (`_step3_correlated_percept_boundary.py`): the compose algebra TOLERATES code
  correlation up to code-sim **≈0.98** (the role-binding decorrelates the cross-terms). CAVEAT recorded: this is
  compose-ROBUSTNESS to correlation, **NOT** generalization-across-similar-concepts (the separate dendritic/PPMI
  job; "decorrelation is a red herring", CYCLE 88). "Algebra tolerates correlation" ≠ "correlation buys
  generalization."
**🧠⚡ The merged "one brain" nav action-decision is now FULLY-SPIKING by DEFAULT (2026-06-19, roadmap #4 default-on).**
Per the owner's brain-based-purity directive, `run_moving_goal_episode`'s LIBRARY defaults are flipped to the
validated spiking config — `readout_source="spiking_wta"`, `sel_recurrent_weight=0.3`, `n_sel_per_action=n_commit_per_action=40`,
`urgency_max_pA=180.0` — so the action EMERGES from the spiking competition (Wang-2002 accumulator + Lo-Wang
commit-burst threshold-crossing), the host Python argmax RETIRED. Validated 6-seed grid-32/1800 at **1.16× host
(within the 25% deploy bar), 100% commit-burst** (zero argmax fallback) — down from the CYCLE-216 ~1.7× boundary via
two levers (Usher-McClelland accumulator LEAK + finite-size-noise N-scaling; the ~16% residual = the irreducible
commit-timing/finite-size floor, the honest BRAIN-BASED-ONLY deliverable). **The CLI `--readout-source` default stays
`"motor"`** so every documented standalone benchmark reproduces unchanged; `motor`/`thal` = the opt-in host-argmax
ORACLE (the tuned levers are inert under them). NO `sim/` edit (runner-only default flip); the spiking read-out is

Graceful error handling: missing tag names + empty tags silently
skipped. Caller manages awake/sleep gate transitions.
```bash
python -m research.runners.validate_trisynaptic_loop \
    --seed 42 --train-events 400 --ca3-recurrent-weight 5.0 \
    --direct-ca3-drive \
    --out research/findings/raw/g11_bg/trisynaptic_seed42.json
```
Methodology note: EC-driven test (drive lang_input, propagate
through trisynaptic chain) FAILED at all parameter combinations.
DIRECT-CA3 test (drive partial of stored CA3 ensemble directly) is
the cleaner Marr autoassociator test and PASSES at train=400 +
ca3_recurrent_weight=5.0.

> _Archived: **Realigned plan** (was CLAUDE.md L1499-1523) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **Concept-pool v1->v17 architecture + engram-composition saga** (was CLAUDE.md L1524-2523) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **160/320-concept G.20 sparse-distributed ensemble + 320 flat-distinct composition** (was CLAUDE.md L2524-2611) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._

> _Archived: **Path 3 Phase 3.2** (was CLAUDE.md L2613-2704) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._

**Default workflow (continuous mode):**
```bash
# Loads lineage 'main' if it exists, skips ~6-20 min training.
# Saves back on exit; previous state goes to history/.
python -m research.runners.chat_repl --mode synonym
```

**Science mode (multi-seed reproducibility):**
```bash
# Always trains from random init; does NOT touch lineage.
python -m research.runners.chat_repl --mode synonym --from-scratch --seed 42
```
**Compatibility:**
- Lineage stores `mode` + arch in metadata. Loading a `tier1` lineage
  with `--mode synonym` triggers a "fallback to fresh training"
  warning — no shape-mismatch crash.
- `save_checkpoint` doesn't preserve firing thresholds / STP /
  eligibility per the CLAUDE.md gotcha above. Self-recovers in ~10ms
  of free running. Fine for inference (REPL chat); documented.

> _Archived: **Recommended configuration** (was CLAUDE.md L2768-2943) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **Text I/O infrastructure** (was CLAUDE.md L2944-3252) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
**🎯 LATEST BREAKTHROUGH 2026-05-05: G v2.5 + K v2 SCALES to 32×32 at 2.57 ± 0.11 (n=6) — 13.3% BETTER than the 16×16 baseline.**
```bash
# G v2.5 + K v2 — biology-grounded, perception only, scales to 32×32:
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 32 --seed N --n-steps 1800
```

**Scaling result (2026-05-05 step 3).** ⚠️ **RE-CORRECTED 2026-07-16 (the FIRST correction, written the same day, was
itself WRONG — it declared "all figures are `sum_finalQ`" one line above a figure that is a MEAN, thereby CERTIFYING
the very conflation it was written to kill. It fixed the label without re-checking the number.)** The runner prints
BOTH metrics on one line (`g11_bg_runner.py:8158-8161`): `sum_finalQ` = the SUM over the 4 goal phases of each phase's
final-quarter mean Manhattan distance; `mean_distance_overall` = the mean over all steps. **They differ ~3× at 16×16
and the two headline rows below were quoted from DIFFERENT metrics.** Recomputed from the raw artifacts:
[`2026-07-16-anchor-claim-audit-...`](research/findings/2026-07-16-anchor-claim-audit-10-defects-in-the-record-incl-my-own-correction.md). ⚠️ **CORRECTED 2026-07-16 — the "closes 4 of 5 cheats
(heuristic, (gx,gy), (x,y), beacon)" claim was FALSE and is WITHDRAWN.** This config leaves
`--heuristic-strength` at its **default 1.0** → 800 pA into `cortex_N/E/S/W` derived from **direct
`gy > y` / `gx > x` goal reads**. The flag that actually closes the heuristic is
`--cue-reflex-replaces-heuristic` (`g11_bg_runner.py:7042-7045`), and it is **absent from this run's own
recorded command** (`raw/g11_bg/k_v2_stress_16x16_seed100.cmd.json`). The claim was copied from the
2026-04-27 flagship, which DOES carry that flag (so the "NO heuristic" line further down, for THAT
config, is correct). **The 2.97/2.57 numbers stand as measured — with the heuristic ON;** the visual
pathway's independent contribution is unquantified. Finding:
[`2026-07-16-clusterKv2-NO-heuristic-claim-is-FALSE-the-flag-that-closes-it-is-absent.md`](research/findings/2026-07-16-clusterKv2-NO-heuristic-claim-is-FALSE-the-flag-that-closes-it-is-absent.md). See
> _Archived: **Superseded/earlier nav flagships part 1** (was CLAUDE.md L3306-3405) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
> _Archived: **Superseded/earlier nav flagships part 2** (was CLAUDE.md L3410-3691) → [`docs/project-history-archive.md`](docs/project-history-archive.md); retrieve via `.venv-rag/bin/python tools/rag/rag_search.py "<q>" --corpus doc`._
Findings docs in `research/findings/` document each session's outcome; **negative results are real findings** and stored alongside positives. A new runner should be added whenever a new architectural variant is being tested.


## Units

- Time: milliseconds (ms)
- Voltage: millivolts (mV)
- Current: picoamperes (pA) or microamperes/cm² (µA/cm²)
- Conductance: nanosiemens (nS) or mS/cm²
- Capacitance: picofarads (pF) or µF/cm²

## Reproducibility

All RNG sources (CuPy, NumPy, random) are seeded together for determinism. The `RuntimeState.actual_seed_used` tracks the seed used. Separate seeds exist for heterogeneity and noise (`heterogeneity_seed`, `ou_seed`).

> ### ⛔ **`actual_seed_used` DOES NOT SEED ANYTHING. Set `cfg.seed`.** (a real bug, 2026-07-17 — read this before writing a runner)
>
> **`actual_seed_used` is a REPORTING field. The bridge never reads it.** Heterogeneity is seeded from **`cfg.seed`**
> (`bridge.py:2136`): `het_seed = cfg.heterogeneity_seed if cfg.heterogeneity_seed >= 0 else cfg.seed;
> if het_seed >= 0: cp.random.seed(het_seed)`. **Both default to `-1`**, so if you never set one, **the guard never
> fires** and the per-neuron firing thresholds (`bridge.py:1508`, `cp.random.uniform`) come from the **UNSEEDED GLOBAL
> RNG** — `--seeds 42` will NOT control your substrate.
>
> ```python
> cfg = CoreSimConfig(..., seed=42, ...)   # ✅ correct — what the determinism suite does
> cfg = CoreSimConfig(); cfg.seed = 42     # ✅ also correct
> cfg = CoreSimConfig(); cfg.actual_seed_used = 42   # ⛔ SEEDS NOTHING. Different neurons every run.
> ```
>
> **This cost the deep-credit arc months of confounded results**: two fresh processes at the same seed got different
> neurons; four nets built back-to-back in ONE process differed by up to **18.4 mV** (each build advances the global
> RNG), so every FULL-vs-FROZEN comparison compared **different neurons** — a confound **~3× the effect** being
> measured (`deep_credit_share` read **+0.333 / 0.000 / −0.333** on the *same* seed). **8 of 93 runners had this bug.**
>
> **The engine is fine** — it seeds correctly the moment you pass `seed=`. **Verify, don't assume:** build twice at one
> seed and hash `cp_neuron_firing_thresholds`; identical ⇒ seeded. Pinned by
> `tests/test_determinism.py::TestSubstrateActuallySeeded`. Finding:
> [`2026-07-17-THE-SEED-NEVER-CONTROLLED-THE-SUBSTRATE-...`](research/findings/2026-07-17-THE-SEED-NEVER-CONTROLLED-THE-SUBSTRATE-the-deep-credit-arc-was-confounded-by-unseeded-neurons.md).

## GPU Memory Considerations

- Networks >100K neurons require 20GB+ VRAM

## Agent Style

See `.claude/style.md` for the recommended agent identity and communication style when working on this codebase (computational neuroscience engineer with GPU computing expertise).
