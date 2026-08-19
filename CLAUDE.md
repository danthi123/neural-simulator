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
recipe in `GAP_CLOSURE_MISSION.md` → "SESSION START". **The heartbeat now ALSO runs `tools/parallel_audit.py` every
cycle: `⛔ UNDER-PARALLELIZED` (idle local/pool cores or GPU + ready Vikunja board tasks > in-flight lanes) is a STALL —
launch the listed independent work (agents for build/research · mini-PC pool for CPU · GPU for the big run) BEFORE
holding; holding is only earned at `✓ SATURATED`. Owner-flagged recurrence 2026-08-18: past fixes failed being
manual/advisory/passive.** NEVER WAIT on a background run without a live state-heartbeat, and
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

## When Compacting (custom instruction — MUST survive into every compaction)

**Target: compact at roughly 50% of the window, NOT at the ceiling.** Compaction near the limit is worse than
compaction early — the summariser gets less room to work in exactly when the session holds the most state.
**EXCEPTION, and it is mine to call: do not compact while a decisive run is mid-flight and its verdict is not yet
recorded**, because a summary written between "the run finished" and "the artifact was read" loses the one thing
the session existed to produce. Land the verdict, then compact.

**PRESERVE (drop anything else to fit):**
- **The ACTIVE MISSION + the non-negotiables** — brain-based-only, one-brain, no-defer, speed-secondary, the
  honesty boundary — and the pointers to the MASTER ROADMAP + `GAP_CLOSURE_MISSION.md`.
- **THE GATES ARE AUTHORITATIVE.** `tools/gates/` + [`docs/FAILURE_GATE_MATRIX.md`](docs/FAILURE_GATE_MATRIX.md)
  + [`research/FAILURE_LOG.md`](research/FAILURE_LOG.md). Where a remembered rule and a gate disagree, the gate
  wins. A newly-noticed failure gets ONE LINE in the failure log and `gates/coverage` blocks until it names a gate
  or declares NOT-GATEABLE. **Do not rebuild a check that exists — read the matrix first.**
- **THE PENDING LIST IS ON THE BOARD, NOT IN CHAT.** `GAP_CLOSURE_MISSION.md` CURRENT STATE carries the ordered
  next actions. This is load-bearing: a backlog living in conversation evaporates at exactly this moment, which
  is why it was moved.
- **The wall reframe:** at any wall ask *"what else does the real system run alongside this, that we replaced with
  a constant?"* BEFORE *"what biology surpasses this?"* — the proxy usually owns the measurement (97% of a gap#5
  weight change was the clamp). And **the instrument is part of the emulation.**
- **LIVE BACKGROUND WORK — every running run / workflow / agent / cloud instance and its state file**, so nothing
  is lost or double-launched: the crux (`gap4-crux.service`), the pool (`research/queue/pool.queue`,
  `dispatch.log`), the AWS lane (`research/queue/.aws_gpu`, **billing while running** —
  `bash tools/aws_gpu.sh stop`). Plus every uncommitted result awaiting a verdict.
- **Owner directives given this session — VERBATIM intent, never a paraphrase.**
- **Files created/modified + why**, plus any `NO sim/ edit` / additive-default-off scope flags.

**Summarise aggressively** (keep only what changes a decision): git log, verbose run logs (error lines only),
search dumps, exploratory reads. **Preserve any test / benchmark / GO-gate command VERBATIM.**

**Context hygiene:** history lives in [`docs/project-history-archive.md`](docs/project-history-archive.md) and
[`docs/ENGINE_REFERENCE.md`](docs/ENGINE_REFERENCE.md), RAG-indexed — RETRIEVE it, do not reload it. Prefer
`/clear` between unrelated arcs over one mega-session, and offload heavy reading to subagents (their context does
not count against this window).

## Research the record BEFORE building — `tools/before_you_build.sh` (the gate is MECHANICAL)

**The trigger is not a judgement call.** Run `bash tools/before_you_build.sh "<defect>"` before the first lever
against any difficulty; `workflow_check` fails when a finding is written without a source check. **≥2 distinct
levers against the SAME defect without resolution ⇒ the research gate FIRES** — cheapness of the next step is not
an exemption, because the quantity that matters is cumulative effort against one difficulty (6 levers / ~4 GPU-h
were spent before the gate ever subjectively fired; one research round then resolved it).

**LOCAL CORPUS FIRST:** `.venv-rag/bin/python tools/rag/rag_search.py "<q>" 5 --corpus finding|plan|doc|catalog|kandel|paper|all`
— our findings + the biology catalog + Kandel 6e + 7 textbooks. **The RAG LOCATES; a rerank hit is a POINTER, not
a paraphrase — READ the surfaced source.** Only then go external. (A 21-agent round produced a Kandel quote that
was NOT IN THE TEXT; `research/biology/`'s anchor check caught it. Verify what an agent cites.)

**THE SELF-CHECK, which is the part no tool can run:** the moment you write a verdict containing NEGATIVE /
BOUNDARY / NO-GO / "walls" / "can't" — *or the softer* "structural primitive" / "honest negative" /
"characterized limit" / "defensible" — **and your next instinct is to scope the fix, THAT INSTINCT IS THE
TRIGGER.** The comfortable verdict is the START of the research, never the end. Isolate and QUANTIFY the genuine
residual first: it is usually far smaller than the blocker felt.

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

## Engine reference → [`docs/ENGINE_REFERENCE.md`](docs/ENGINE_REFERENCE.md)

Architecture, thread model, config traps, backend selection, UI-config roundtrip, composer notes and the nav
recipes moved there 2026-07-31: 247 lines of a 494-line file, loaded in full every session, needed only when
touching the subsystem each describes. **The two that are genuinely load-bearing are now ENFORCED, not
remembered** — the plasticity BOUND TRAP raises via `tools.lab.bound_check` (it has bitten five rules), and the
`cfg.seed` trap is pinned by `tests/test_determinism.py::TestSubstrateActuallySeeded` (it confounded a whole arc
with unseeded neurons). Retrieve the rest: `rag_search.py "<question>" --corpus doc`.

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
