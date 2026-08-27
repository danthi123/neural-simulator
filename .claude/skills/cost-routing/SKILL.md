---
name: cost-routing
description: Route each unit of work to the CHEAPEST resource that fits — the experiment engine (0 Claude tokens) for compute, weaker Claude models (haiku/sonnet) for mechanical/moderate agent work, Opus only for genuine judgment. Apply BY DEFAULT before dispatching any agent or Workflow, and whenever token burn is a concern. Owner-flagged 2026-08-19 (~50% of the weekly limit in 1.5 days).
---
# Cost-routing — spend tokens only where judgment actually needs them

**The burn problem (owner-flagged 2026-08-19).** Two leaks, both now DEFAULTS here, not reminders:
(1) every agent/workflow inherited **Opus** when most agent work is mechanical; (2) sweeps/evals/probes that
belong on the **free experiment engine** were run in agents or the main loop. A 6-agent Opus doc workflow burned
~500k tokens; the same on haiku/sonnet is a fraction. This skill is the decision procedure; the goal is FAST,
consistent progress at a fraction of the token cost — quality stays where judgment is needed, cheap everywhere else.

## Lever 1 — MODEL-TIER every agent (the single biggest lever, and the one that was missed)
Never inherit Opus for an agent by default. Choose the cheapest model that can plausibly do the stage
(Agent tool `model:` · Workflow `agent(prompt, {model, effort})`):

| Task class | model | effort | Examples |
|---|---|---|---|
| **Mechanical** | `haiku` | low | file edits, extraction, counts, run-a-command-and-report, grep/collect, apply a known patch, format/lint a doc, mechanical provenance |
| **Moderate** | `sonnet` | medium | ground-truth gathering, code-trace, scouting/design options, doc rewrite, review, synthesis, most "figure out X and report" |
| **Hard judgment** | `opus` (inherit) | high | adversarial verify of a subtle claim, core mechanism design, a delicate `sim/` build, the final honesty pass |

Default to the LOWEST tier that could do it; escalate ONLY the specific stage that genuinely needs judgment
(e.g. a 4-stage workflow is often haiku·sonnet·sonnet·opus, not opus×4). In a Workflow, set `model`/`effort`
per `agent()` call — mixed tiers in one script are normal and correct.

## Lever 2 — ENGINE-FIRST for compute (0 Claude tokens)
A parameter sweep, multi-seed run, or mechanical eval/probe MUST go on the experiment engine — never an agent,
never the main loop:
- CPU grids / tuning → `tools/sweep_pool.sh` (mini-PC pool)
- GPU sweeps / long runs → `tools/gpu_queue.sh add '<cmd>'`
- multi-seed of one config → the runner's `--seeds` (the controller fans out)
- a verification A/B, lesion probe, or backend check → a **backgrounded headless run** (Bash `run_in_background`), not an agent
Agents are for BUILDS/integration that need judgment. **If you are about to put a sweep or a scripted probe in an
agent, stop — it goes on the engine.** (This session's affect sensitivity map, lesion A/Bs and cupy verifies were
run this way: 0 Claude tokens. Keep doing that; do not regress them into agents.)

## Lever 3 — main-loop discipline (the priciest stream)
The main Opus loop is the most expensive token stream in the session. Reason tersely; do not re-derive
established facts; offload mechanical reads/edits to a haiku agent or the engine. Session **effort** (ultracode /
high / medium) governs the main loop and is the owner's lever — dropping to medium cuts main-loop burn directly;
model-tiering + engine-first are the levers Claude controls and should be applied regardless of effort.

## The self-check — run it before EVERY agent/workflow dispatch (this is the enforcement point)
1. **Engine?** Could the experiment engine do this for 0 tokens (sweep / multi-seed / scripted probe)? → route it there, don't spawn an agent.
2. **Tier?** For each agent stage: is it mechanical (`haiku`), moderate (`sonnet`), or genuine Opus judgment? → set `model` explicitly to the cheapest that fits. An agent with NO model override is a red flag — you defaulted to Opus.
3. **Main loop?** Am I about to reason at length for something an agent/engine should do? → delegate or shorten.
4. **Research-first?** Is this a NEW DIRECTION or a WALL (a design pass, a first attempt at a capability, or
   attacking a characterized limit)? → BEFORE dispatching, run `bash tools/before_you_build.sh "<capability>"`
   yourself, READ the prior work it surfaces (especially any existing `status:GO` / scoping doc / research-gate),
   and carry that prior work + an explicit "RAG the finding+kandel+paper corpora and read the surfaced sources"
   mandate INTO the agent's prompt. The commit-time `deep_research_at_wall` gate only catches a missing source
   check REACTIVELY — after the agent has burned the tokens re-deriving what the record already holds. The
   `neural-simulator` skill carries the full deep-research discipline; this item is the dispatch-time trigger for
   it. (Earned 2026-08-27, owner-flagged: an integration-phase DESIGN agent was dispatched without this;
   `before_you_build` then surfaced a 2026-08-11 `status:GO` cross-region synaptic pathway the design would
   otherwise have re-derived from scratch.)

`tools/parallel_audit.py`'s 💸 COST-ROUTING block is the per-heartbeat reminder + the engine pointers; THIS skill is
the decision procedure that turns "I know I should" into "which model / which lane, decided before I dispatch."
