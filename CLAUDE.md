# CLAUDE.md

Guidance for Claude Code (and the local model via `tools/local_llm/llm.sh claude`) in this repository.
**Repository**: https://github.com/danthi123/neural-simulator. Rationale, history and the owner's full wording:
[`docs/CLAUDE_RATIONALE_ARCHIVE.md`](docs/CLAUDE_RATIONALE_ARCHIVE.md) (the pre-2026-09-25 CLAUDE.md, verbatim;
RAG-indexed, so retrieve from it instead of loading it).

## ⭐ ACTIVE MISSION: read the PLAN and the BOARD first, every session and every continuation

A sim-brain that **converses genuinely**: it reasons to its own conclusions, with an affective world-model, emotion,
self-awareness and curiosity (not fact-recall/RAG, not LLM plausible-text). Success is TRUE CONSCIOUSNESS on the
emergentist bet, so the job is **completeness + faithfulness of the biological emulation**, not a benchmark score. It
grows through a TEMPORARY AI-teacher scaffold that graduates to real humans; every scaffold is biologized toward the one
spiking brain, the transformer minimized. (Owner 2026-09-19: Qwen stays as the permanent conditioned-articulation
mouth; the #1 metric is the lesion-verified load-bearing fraction:
`docs/plans/2026-09-19-roadmap-with-a-permanent-llm-mouth.md`.)
**Current phase (owner 2026-09-25): prove who owns the computation.** Which part of the system caused the answer,
shown causally? A host-decision share, measured by intervention, sits beside the load-bearing fraction:
`docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md`.

- **PLAN:** [`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md)
  (faculty map, one-brain architecture, 6 stages, walls ledger; the old 5-gap cluster is a sub-view). Foundation:
  `docs/plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md`.
- **RESUME POINT:** [`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md) CURRENT STATE; the ordered pending list lives
  there, not in chat.
- **To SKIM:** [`ROADMAP.md`](ROADMAP.md), plain-language status; its "Project shorthand" table decodes
  FHRR/BTSP/BDSP/GNW/gap#N/DR-N/RANK-N/EMERGE/the-moat/the-composer.
- Cross-session continuation is MANUAL by owner choice ("continue" + the roadmap + the board): no watchdog/daemon.

## Non-negotiables

1. **Brain-based only.** Everything between sensation and action is neurons, synapses and their communication. Host
   code is legitimate ONLY for the world (its state + rendering the senses), the body (acting on motor output) and the
   teacher as social environment. A host computation is a shortcut EVEN IF biologically correct (a Python RPE, an
   argmax, a distance reward); at most it is a teaching scaffold for its neural replacement. Project-wide.
2. **One brain, ONE spiking substrate**, shared: faculties interact through synapses, not as co-located modules.
3. **No defer (THE LAW).** A wall or negative is a verdict on a METHOD, never a license to abandon a CAPABILITY: bank
   the method, take a new biology/spiking/one-brain method, keep going until it works. Closure cannot be deferred; a
   "characterized limit" or "honest negative" is documented as a finding, never a stopping point.
4. **Speed is secondary.** Slow-but-faithful biology (dendritic credit, seconds-long BTSP plateaus, sleep replay) is
   in scope; never trade faithfulness for speed. (Owner refinement 2026-09-18: a per-mechanism realism trade only for
   a significant performance gain that loses nothing important to the end goal, documented honestly.)
   **Faithfulness-first, but falsifiable (owner 2026-09-25):** biological fidelity never counts as evidence for a
   capability by itself. For every major capability, report which mechanisms were NECESSARY for it, shown by
   controlled six-seed ablation; mechanisms are not dropped because one benchmark does not need them.
5. **The honesty boundary is a deliverable.** Build and measure every functional consciousness / self-model / affect
   correlate; every self-report is an honest functional read-out ("my familiarity monitor reads this as novel, so I'm
   uncertain"); NEVER assert phenomenal experience.
6. **Six seeds** (42/43/44/100/101/102) before any generalization claim.
7. **The gates are authoritative:** `tools/gates/` (run by `tools/githooks/pre-commit`) +
   [`docs/FAILURE_GATE_MATRIX.md`](docs/FAILURE_GATE_MATRIX.md). A remembered rule that disagrees with a gate loses;
   read the matrix before building a new check. Never commit with `--no-verify`.
8. **Push to BOTH remotes with `bash tools/push_both.sh <branch>`** (it verifies instead of claiming).
9. **GPU:** one brain-loading GPU process at a time, all via `tools/gpu_queue.sh` (`pause --now` / `resume` for gaming).
10. **RAM:** heavy or full-brain jobs run under `tools/memcap.sh <gb> -- <cmd>` once `tools/mem_ok.sh <need_gb>`
    allows; never stack one onto a live training.
11. **Cost-routing + model tiering.** Mechanical work goes to non-Claude machinery (CPU sweeps → `tools/sweep_pool.sh`,
    GPU sweeps/long runs → `tools/gpu_queue.sh`, multi-seed → the controller's `--seeds`); agents only for genuine
    builds. Maximize parallelism AND minimize tokens. Tier every agent (haiku = mechanical · sonnet = moderate ·
    opus = hard judgment), never inherit Opus: `gates/workflow_cost_tiering`, `tools/cost_audit.py`, `cost-routing` skill.
12. **One driver at a time:** Claude, or one local-model driver (`tools/local_llm/llm.sh claude`; older: OpenHands or
    Hermes, `docs/OPENHANDS_TAKEOVER.md`). Hand the current one back before starting another.

## Workflow essentials

- **Session start, or ANY continuation (including after compaction):** verify a within-session STATE-CHECKING heartbeat
  Monitor is live (GPU / running procs / recent-output every ~15 min; a text-only "are you idle?" nudge is not enough)
  and arm one if not; recipe in `GAP_CLOSURE_MISSION.md` → "SESSION START". It runs `tools/parallel_audit.py` each
  cycle: `⛔ UNDER-PARALLELIZED` is a STALL, so launch the listed independent work before holding (holding is earned
  only at `✓ SATURATED`). Never wait on a background run without it, nor trust a subagent-armed Monitor or passive
  re-invocation to catch a completion. Stocking a queue: `.venv/bin/python tools/lane_check.py` (fails on a monoculture,
  an unserved crux or no CPU lane).
- **Research the record before building:** `bash tools/before_you_build.sh "<defect>"` before the first lever; two
  unresolved levers against one defect fire the research gate (a cheap next step is no exemption). Local corpus first:
  `.venv-rag/bin/python tools/rag/rag_search.py "<q>" 5 --corpus finding|plan|doc|catalog|kandel|paper|all`; a hit is a
  POINTER, so READ the source and verify what an agent cites. Then external: `bash tools/deep_research.sh "<wall>"`.
  In probes: `from tools.lab import lever, before_after, undefined_if_empty, void_if`.
- **At a wall, ask first** "what else does the real system run alongside this, that we replaced with a constant?", only
  then "what biology surpasses this?" (finding `2026-07-31-why-we-hit-walls-the-missing-companion-process`). The
  instrument is part of the emulation. Writing NEGATIVE / BOUNDARY / NO-GO / "walls" / "can't" / "structural
  primitive" / "honest negative" / "characterized limit" / "defensible" with the urge to scope a fix IS the research
  trigger: isolate and quantify the residual first.
- **Sync the summary docs in the same cycle** a finding changes a wall/gap status, the frontier or a next action: the
  `sync-documentation` skill (roadmap §7, `GAP_CLOSURE_MISSION.md` CURRENT STATE, `research/findings/AUTONOMOUS_STATE.md`,
  `ROADMAP.md`) + the Vikunja board (`vikunja` skill). The PostToolUse nudge means RUN it.
- **One term, one meaning:** before writing consolidation · compositional · self-organized · closed · GO · fully
  spiking · byte-identical · lesion · selective · works in a finding, commit message or board entry, check its code
  condition in [`docs/TERMS.md`](docs/TERMS.md).
- **Document structure ([`docs/WRITING.md`](docs/WRITING.md)):** W1, a voided document is registered in
  `docs/RETRACTED.md` and cited only with `⛔` on the same line; W2, prose lines in governed files are ≤ 800 chars.
  Check: `.venv/bin/python tools/check_docs.py` (CI `tests/test_doc_rules.py`; fixer
  `tools/split_long_doc_lines.py --apply`). Structure only; truth is the `verify-go` skill's job.
- **A noticed failure** gets one line in [`research/FAILURE_LOG.md`](research/FAILURE_LOG.md); `gates/coverage` blocks
  until it names a gate or declares `NOT-GATEABLE: <reason>`. Biology is bound once in `research/biology/<id>.md`.
- **Prefer a check that fails loudly over more prose here.** When a process lapse recurs, at session end or before
  compaction, or when asked: the `evolve-skills` skill.

## When Compacting (custom instruction — MUST survive into every compaction)

Compact at roughly 50% of the window, not at the ceiling. EXCEPTION (mine to call): not while a decisive run is
mid-flight with its verdict unrecorded; land the verdict, then compact. After a substantial arc, run `evolve-skills` first.

**PRESERVE (drop anything else to fit):**
- **The mission + the non-negotiables** (brain-based-only, one brain, no-defer, speed-secondary, the honesty boundary)
  and the pointers to the MASTER ROADMAP + `GAP_CLOSURE_MISSION.md`.
- **The gates are authoritative:** `tools/gates/` + `docs/FAILURE_GATE_MATRIX.md` + `research/FAILURE_LOG.md`; a new
  failure gets one line in the log. Do not rebuild a check that exists.
- **The pending list is ON THE BOARD** (`GAP_CLOSURE_MISSION.md` CURRENT STATE), not in chat.
- **The wall reframe question** (above): the proxy usually owns the measurement.
- **LIVE background work** — every running run / workflow / agent / cloud instance and its state file: the crux
  (`gap4-crux.service`), the pool (`research/queue/pool.queue`, `dispatch.log`), the AWS lane
  (`research/queue/.aws_gpu`, **billing while running** — `bash tools/aws_gpu.sh stop`), plus every uncommitted
  result awaiting a verdict.
- **Owner directives given this session — VERBATIM intent, never a paraphrase.**
- **Files created/modified + why**, plus any `NO sim/ edit` / additive-default-off scope flags.

Summarise aggressively (git log, run logs → error lines only, search dumps, exploratory reads). **Preserve any test /
benchmark / GO-gate command VERBATIM.** History lives in `docs/project-history-archive.md` and
`docs/ENGINE_REFERENCE.md` (RAG-indexed: retrieve, don't reload). Prefer `/clear` between unrelated arcs; offload heavy
reading to subagents.

## Reproducibility: set `cfg.seed`

`actual_seed_used` only REPORTS a seed and seeds nothing: pass `CoreSimConfig(..., seed=42)` (or `cfg.seed = 42`), else
per-neuron thresholds come from the unseeded global RNG (`heterogeneity_seed` / `ou_seed` override per source). Pinned
by `tests/test_determinism.py::TestSubstrateActuallySeeded`; the incident is in the archive.

## Engine reference → [`docs/ENGINE_REFERENCE.md`](docs/ENGINE_REFERENCE.md)

Architecture, threads, config traps, backends, UI-config roundtrip, composer and nav notes. The plasticity bound trap
raises via `tools.lab.bound_check`. Networks >100K neurons need 20GB+ VRAM.

## Common Commands

```bash
python neural-simulator.py --auto-tune [--quick]     # headless parameter sweep
# runners run as modules (-m research.runners.X), so provenance sidecars are automatic
python -m research.runners.g11_bg_runner --moving-goal --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/g11_seed42.json
python -m research.runners.g11_bg_runner --probe-action W   # static cascade probe
```

## Units and style

Time ms · voltage mV · current pA or µA/cm² · conductance nS or mS/cm² · capacitance pF or µF/cm². Agent identity and
communication style: `.claude/style.md`.
