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

**Verify the command BEFORE you queue it (2026-09-06).** A malformed engine command `rc=2`'s INSTANTLY and the
`gpu_queue`/pool daemon silently pops the NEXT job — so one wrong flag can burn through a whole queued battery
unnoticed (you see `runners=1` and assume progress). Before `gpu_queue.sh add` / `sweep_pool.sh` / a backgrounded
run, confirm the flags AND their FORMAT against the runner's argparse (`grep add_argument <runner>` or `--help`):
flag-EXISTS is not enough — check `nargs` / comma-vs-space / `type`. Earned this session: `_rank2 --seeds 42 43 44
...` rc=2'd (that runner splits ONE comma-string: `--seeds 42,43,44,...`) while a sibling runner's `--seeds` is
`nargs='+'` (space-separated) — same flag name, OPPOSITE format; and a mouth run rc=2'd on `--eval-corpus` launched
from a checkout that lacked the flag (verify the flag exists on the BRANCH/worktree the command runs in, not just
on main). Cheap check; a rc=2 wastes the slot + masks a stalled battery.

**RAM guard (2026-09-01) — the "careful with the computer's limits" half of engine-first.** The sim's 15k-LTM
brain build is ~2GB RSS and ACCUMULATES over repeated builds; several concurrent LOCAL builds OOM (the 2026-08-26
OOM; a heavy per-case verify this session). So: heavy 6-seed 15k-LTM sweeps go to the pool/gpu (above), and before
ANY *local* sim smoke run `free -m` — build locally only with clear headroom (~5GB+ available), else queue it.
For a DETERMINISTIC routing/parsing property, don't rebuild the brain per case at all — test via the unbound route
method + a mock `self` (pure parsing, seconds, ~0 RAM) and confirm recall separately with the already-6-seed-GO
primitive. Arm a `free -m` watchdog (fires below ~4GB) when fanning out many concurrent agents.

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
   otherwise have re-derived from scratch.) **FLIP tasks specifically: verify the actual CODE default (grep the
   `_DEFAULT_ON` constant / the `os.environ.get(..., "0"|"1")` default), NOT just the ledger's `on_by_default`
   row — the ledger is a SUMMARY that can lag the code (drift #12). Earned 2026-09-01: a stale `on_by_default: NO`
   on a faculty that had ALREADY been flipped default-ON hours earlier scoped a redundant ~196k-token flip-soak
   agent; the agent verify-first-caught it, but the code-grep would have caught it before dispatch for ~0 tokens.
   Corollary: when you flip a faculty, sync its ledger `on_by_default` row SAME-CYCLE, or the next scoping read is
   misled.**
5. **Verification: controller-harvests, not agent-waits?** Will this agent need a smoke / sweep / 6-seed / live-handler
   verify? Then do NOT spec it to "run the verification, THEN commit" — that pattern STRANDS (recurred ~7× through
   2026-08-28 despite explicit anti-strand prose in the prompt: the agent backgrounds the run + STOPS with an empty
   *"I'll wait for the verification to complete"*, having committed nothing, forcing a SendMessage round-trip to
   resume it). Two reliable shapes only: (a) the agent BUILDS + RETURNS the exact command(s) and the CONTROLLER runs
   them + banks; or (b) the agent runs ONLY bounded-INLINE smokes (hard time cap, drop to a smaller cell if over,
   commit the PARTIAL) and treats any deferral as a failed run. NEVER let an agent's deliverable depend on a
   backgrounded run it then waits on. The `neural-simulator` skill carries the full "controller runs every smoke"
   rule + the RECOVERY RECIPE (a stranded agent's work is NOT lost — harvest its `.claude/worktrees/agent-<id>/`,
   do NOT re-launch; and if an agent whose deliverable is ALREADY on main keeps re-notifying / re-running redundant
   heavy work — a runaway LOOP, not a one-time strand — `TaskStop <agent-id>` it, do not keep resuming it; earned
   2026-09-01: an agent looped ~146 min / 335k tokens re-running an OOM-ing 15k-LTM verify after its work had
   already landed); this item is the dispatch-time trigger for it. (Earned/re-earned 2026-08-28: the rule was
   already fully written in `neural-simulator` yet the lapse recurred all session — because the DISPATCH-TIME
   checklist, the thing actually run before spawning, did not carry it.)
6. **Isolation: worktree for any COMMITTING agent?** An agent (Agent tool) or workflow `agent()` that will
   `git checkout`/`commit`/push MUST run in its own git worktree — set `isolation: "worktree"` (Agent tool) /
   `agent(prompt, {isolation: 'worktree'})` (Workflow). WITHOUT it the agent runs in the SHARED main checkout and
   its `git checkout -b`/commit RACE the main session's and the other agents' git ops. (Earned 2026-09-01: two
   un-isolated compute agents in one session switched the MAIN checkout's HEAD onto their own branches and swept a
   main-session commit onto the wrong branch; one saw the main session's commits-of-its-work as a mystery
   "auto-commit process." Recovered cleanly but cost real diagnosis. The SAME session's isolated workflow-wave
   agents had ZERO races. Pure read/search agents that never commit don't need it.)

`tools/parallel_audit.py`'s 💸 COST-ROUTING block is the per-heartbeat reminder + the engine pointers; THIS skill is
the decision procedure that turns "I know I should" into "which model / which lane, decided before I dispatch."
