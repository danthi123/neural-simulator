---
name: before-build
description: Use BEFORE building, probing or testing anything (mechanism, runner, prereg, brain/ module), before writing any plan, map or list, before calling a result a wall, negative or limit, and before a 2nd fix on the same defect. Checks the record, the biology catalog and prerequisites first so nothing known is re-derived or rebuilt, then timeboxes the work.
---
# before-build: find what we already know, then build

Each `(...)` names the incident behind a step; read it only when you need the why.
## 1. Record (about 1 minute)
- `bash tools/before_you_build.sh "<topic in one line>"` (searches findings, catalog and docs). Exit 4 = stale corpus:
  run `bash tools/w_follow.sh` first, never search an old tree (FAILURE_LOG 2026-10-21, 15 days stale).
- A hit is a POINTER: open the source and read it, not the snippet. Building from a summary produced a wrong
  "refuted" (memory read_sources_in_depth 07-08). Read our own wiring before theorizing (memory 07-11).
- The index lags: also `git log --all --oneline --grep=<keyword>` and `ls research/findings/*<keyword>*`
  (a GO-6/6 faculty on main for 3 days was invisible to RAG, vikunja reference 2026-08-28).
- Never write a mechanism into a plan, finding or the board from memory: one already refuted was proposed again that
  way (FAILURE_LOG 2026-08-02). A result banked 3 weeks earlier was re-derived for 9 h and 8 GPU cells (FAILURE_LOG 2026-07-31).
- Look for an existing biology sim or a mechanism already built (Spaun, SPEAR, our own KP/burstprop) before
  writing a new one (memory check_existing_sims 05-19; FAILURE_LOG 2026-08-01).

## 2. Inventory before you create (any map, list, ladder or plan)
Read, in order, and EXTEND the file that already covers it. Never start a parallel one:
- `~/Projects/sim-catalog/references/feature-catalog.md` (implemented / partial / missing)
- `~/Projects/sim-catalog/references/biology-buildout-roadmap.md` (tier, cluster, dependencies)
- `research/biology/<id>.md` (existing biology binding) and `brain/PREREQUISITES.md` (our status, test-readiness)
- New core: `brain/ARCHITECTURE.md` and the newest `brain/research/*record-brief*.md`.

## 3. Retractions
- `grep -i "<mechanism>" docs/RETRACTED.md`: a doc listed there is never evidence; cite it only with ⛔.

## 4. Prerequisites and scale, stated up front
- Name the prerequisites the test assumes (brain/PREREQUISITES.md). If one is missing, write "expected to fail:
  missing <X>" BEFORE running. A negative from a 7k-neuron pool or a 0.1B cortex says nothing about the capability
  (memory judge_results_relative_to_scale 11-02).
- Declare host-decision: none, or a scaffold with a cap. Hand-written English rules are host templating (FAILURE_LOG 2026-10-21, 8 builds in 2 days).
- Run the ceiling or reference arm EARLY to bound the arc (memory run_ceiling_early 07-11).
- Validate a teaching signal by its own function, not by a task that does not need it (memory validate_signal_by_its_function).

## 5. Probe hygiene
- Smoke one seed first; `--help` passing is not the run working (FAILURE_LOG 2026-08-01). An argparse flag can be
  accepted and inert: show it changes something with `tools.lab.before_after` (FAILURE_LOG 2026-07-31).
- In probes: `from tools.lab import lever, before_after, undefined_if_empty, void_if`.

## 6. At a wall (before writing negative / boundary / can't / limit / NO-GO)
- Ask first: "what does the real system run alongside this that we replaced with a constant?"
  Then: "is this below the scale, or missing a prerequisite?" (finding 2026-07-31 missing-companion-process).
- Check the instrument: every arm below chance = broken task wiring, not a NO-GO (FAILURE_LOG 2026-07-31). Read
  state variables in physical units: a 333x pA-to-mV constant posed as a "boundary" (verify-go reference 2026-07-25).
- A verdict is on the METHOD, never the capability; name the next method.

## 7. Lever count
- 2 levers on one defect without success: research before a 3rd (record search, then
  `bash tools/deep_research.sh "<wall>"`). Cheapness is no exemption: 5 levers hit the forgetting wall before one
  query found our own CLS design (FAILURE_LOG 2026-08-09).

## 8. Timebox
- 2 days per problem. At the limit write KEEP (why, new limit) or KILL (what is banked) in HANDOFF.md.
  No silent third extension (FAILURE_LOG 2026-10-21: 8 polish builds; mouth arc 68 findings before retirement).

Done = a record brief (3-10 lines: what we knew, what is new, prerequisites met or missing, expected outcome)
in HANDOFF.md STATE or the prereg, then build.
