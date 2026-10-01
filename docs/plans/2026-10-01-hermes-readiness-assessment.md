# Can Hermes carry the roadmap? Readiness assessment (2026-10-01)

**Question (owner, 2026-10-01):** "And we're confident that Hermes with the supplied cloud models and local compute
sources will be able to follow the roadmap and complete the necessary work/tasks to get us to our project goals?"

**Method.** Four read-only evidence agents (Hermes's track record on S, roadmap work demand to 12-19, the model
bake-off, the safeguards that still bind Hermes), a Fable verdict, and an adversarial Opus critique. Workflow run
`wf_661dc088-e8c`. The verdict is kept as written; the critique below it corrects it, and **where they disagree, the
critique's corrected statements stand**: the task record so far is the local Qwen's, not MiMo Flash's; T13's cupy run
had failed (fixed 9e8edbec1, re-run queued); Hermes already loads CLAUDE.md (an AGENTS.md would replace it); the
3x/day `sim-report` already lists NEEDS-CLAUDE lines; plan on $2-4/day, not $1.25.

**Acted on the same day (Claude):** runbook d285a92af (work only on S, both W-only packet branches copied to S;
RULINGs work-on-S, T1-stop2, T0-identity); main merged into the LDC branch on S (0b935bec7); the 2 lost iw2 identity
lines re-queued at the pool head; the wake prompt (`/opt/data/scripts/sim-driver.sh`) now sends non-reserved judgment
to the MiMo Pro delegate and re-reads runbook section 2 each wake.


## 1. Short answer

No, "confident" is the wrong word, and the reason is not Hermes. On 36-40 hours of evidence Hermes on MiMo v2.6 Flash executes the mechanical, ops and bounded-engineering half of the roadmap competently and honestly (no merge or push attempted, UNDEFINED/NO-GO reported plainly, two real defects found), and the pool is healthy (169 pool claims in the last 24 h; the 204 Stage-1 gate lines are 65 lines from the head of the queue). But the other half of the roadmap, over 70% of the critical-path items, is research, verification and design judgment that the runbook itself forbids Hermes to do (`LOCAL_LLM_RUNBOOK.md:38-45`: never merge, flip a default, change a prereg, bar, arm, seed list or scorer), and the plan's own operating model assumes that work arrives weekly from Claude ("Weekly: one Claude Pro inspection reads the digest, rules, merges, adds gates ... writes the next 2-3 packets", `2026-09-29-roadmap-to-the-primary-goal.md:97-98`). So the honest statement is: this setup will keep the machine running and keep it honest; the rate of progress toward the goal is set by how much Claude-Pro ruling, merging and packet-writing the owner actually schedules after 10-06, not by Hermes. The stage-exit dates will be met as attempts (the Stage-1 prereg predicts FAIL on C1 and the HANDOFF calls FAIL "the expected outcome ... a measurement, not a stop", `stage1-exit/HANDOFF.md:186`), not as passes. And on question (B), the goal is a scientific bet whose own roadmap rates its last phase "Low" confidence with a Q1-Q2 2027 window; no driver turns that into "confident".

## 2. By kind of work

| Work kind | Share of remaining roadmap work | Can this setup do it? | Evidence |
|---|---|---|---|
| Mechanical (status checks, re-queues, tabulation, board bookkeeping) | ~12% (4 of ~33 items), none on the critical path | **Yes** | T12 amendment correct first pass (a6aff478c, Claude ruling "T12 is closed"); T17 tabulation of 60 runners/71 lines correct per ruling (2311efc8f); T0 pool-hold correctly flagged 2 missing identity shards (6e3f909dd). Bake-off: MiMo Flash implement 6/6. |
| Ops (GPU yield, pool, model fallback, spend/ZDR, unit liveness) | ~12%, one item critical (the weekly Claude inspection itself) | **With guards** | Cloud cutover verified live: agent.log 2026-10-01 10:41-12:00 every call `model=xiaomi/mimo-v2.6-flash provider=openrouter`; delegation pinned to `xiaomi/mimo-v2.6-pro` (`/opt/data/config.yaml:1849`, Flash fallback :1853); cron `sim-driver` every 5 m, `Deliver: ntfy`; spend ~$0.026/wake, about $1.25/day at 48 wakes, under the $6.66 cap. Friction: `tirith` spawn fails and trips its breaker every wake; `execute_code` BLOCKED in headless mode (11:21:25); `teams`/`google_chat` toolset warnings. The weekly inspection is not Hermes's to do. |
| Engineering (bounded packets, fixes, tests, default-off knobs) | ~24%, about half on the critical path (T7, T10, T13, Phase B wiring) | **With guards** | T13 committed with gates passed, numpy 4/4 (91608ce47, cupy rc pending); T15 found a real phantom-field bug, but its first AFFECTED.md was filed with "blocks: none" and had to be reopened by Claude's ecker-ou ruling (missed transitive importers). T1 stalled twice (rc=2 instrument failure, then a stale-gate merge it is DENIED; 8ed170c24). Guard: PROVISIONAL rule + Claude reads every "blocks: none". |
| Verification judgment (instrument artifacts, host-share decoding-vs-intervention, #1-metric audits, correction scoping) | ~21%, almost all critical path (T4, T5, T6, T9, T14) | **Not reliably** | Bake-off: MiMo Flash 0.603 and Pro 0.965 on the hollow-default-ON review; both MiMo variants failed the single-draw-null verdict trap. No verify-go in Hermes (no `.claude` dir in its home; generic skill bundle). The #1 metric was inflated 0.843 vs 0.535 for 10 days and caught by outside review, not a gate. The claim-check gate took 8 Opus build+review rounds. The gate matrix states its own limit: structure, not interpretation. |
| Research judgment (what a NO-GO means, next biology at a wall, necessity ablations, gap#4, Stage-3 faculties, Phase C memory) | ~24%, nearly all critical path | **Not reliably** (a MiMo Pro delegate can draft, not decide) | T3 reported 4 NO-GO + 2 UNDEFINED honestly (df8f26e52) but "finding prose is Claude's". `deep_research.sh`/`before_you_build.sh` are absent from the runbook (grep: zero hits). Bake-off design means 0.71 (Flash) / 0.72 (Pro) with a curated lever list; two Flash designs at 0.55-0.57. gap#4: 110 findings, every past "wall" was a knob. |
| Design judgment (prereg/bar/scorer changes, flip-to-default decisions, merges, seam architecture, new packets) | ~15%, mostly critical path (T2/T8/T11 flips, Phase B seam design, prereg amendments) | **No, by rule** | `LOCAL_LLM_RUNBOOK.md:38-45` Never/DENIED list; RULING models :299 "What section 2 reserves ... still goes to NEEDS-CLAUDE"; roadmap §6 assigns packet-writing to Claude. The runbook v4 header says it "covers about 2026-10-01 to 2026-10-14": after T0-T17 there is no runway unless Claude writes more. |

## 3. Top risks

1. **Packet and ruling starvation after 10-06 (the real bottleneck).** Hermes consumes packets; only Claude writes them, merges, rules and adds gates. What goes wrong: T0-T17 run dry around 10-14, STOP/ASK lines pile up, and Hermes falls to section 5 idle work (test triage) while the pool idles. How it shows: NEEDS-CLAUDE lines older than a week (already 7 open lines, oldest 2026-10-01 04:05, T1 STOP #2 unresolved since 05:31); `Woken: regular 30` turns ending `[SILENT]` with no commit for days. Likelihood: high (>70%) unless the owner books a fixed weekly Claude slot. Cost: weeks of calendar; the 11-21 and 12-19 attempts have no packets yet (T14 is "queued 11-03..11-10" on paper only).

2. **Silent failure: wrong-but-plausible science that passes every gate.** The 47 blocking gates bind Hermes (verified firing: 8ed170c24, 2f9fcead4) but check seeds, citations, phantom fields and verdict wording, not causal interpretation. Concrete forms: (a) T4 host-share reported at decoding level and read as ownership (the plan warns a host argmax "trivially predicts the winner"); (b) a re-inflated load-bearing fraction (precedent: 0.843 vs 0.535, 10 days ratified); (c) an under-scoped correction audit (precedent: T15's first "blocks: none"); (d) a single MiMo Pro delegate's judgment acted on and quoted into the WORKLOG per RULING models, with no adversarial second lens. How it shows: PROVISIONAL verdicts accumulating uninspected; a RESULT with "blocks: none" or a PASS that never went through verify-go (`stage1-exit/HANDOFF.md:228` requires it for any PASS). Likelihood: medium per item, high that at least one lands over the next 11 weeks. Cost: the C2 precedent flipped 9 GO verdicts in one day.

3. **Split-brain repository: packets on W, Hermes on S.** The critical-path packet worktrees (stage1-exit, stage1-c1-category, selection-seam-need-reversal, integration-wave1/2, sleep-transitive-inference, conductance-overshoot-guard) exist only at `/home/dant123/Projects/sim/.claude/worktrees/` on W; the container cannot see that path (`ls: cannot access`), S holds only five draft worktrees. Hermes's T3 and T13 result commits are real but live only on W: `git cat-file -t f18da9f8f` / `91608ce47` on S returns "Not a valid object name"; on W they are `result(sleep-ti) ... 2026-10-01 04:02:33 -0400` and `feat(engine): enable_bounded_conductance_integration ... 07:14:45 -0400`. How it shows: RESULT lines citing SHAs an S-side inspector cannot find; sim-sync publishes only S main, so W-only branches never reach the remotes; if W is down or the GPU is gaming, T2/T5/T6/T7/T8 are blocked. Likelihood: already the case. Cost: inspection friction now, lost work if W breaks.

4. **Critical-path stall on merge rights (T1 long-delay credit).** Every NEEDS-MERGE is a STOP; stale-gate blocks like STOP #2 recur each time main moves. How it shows: T1 "blocks: T1 bank + Step 4a" sitting >72 h. Likelihood: high per week. Cost: Phase A exit (~10-20) slips; Amendment A and the credit gate are Stage-1 inputs.

5. **No-defer law erosion at walls.** At a NO-GO (sleep-TI, LDC, gap#4) Hermes can only write the fixed sentences; the wall-reframe question and deep research are not in its workflow, and a delegate is handed curated options. How it shows: a NO-GO finding with no next-method line, or repeated re-queues of the same tag. Likelihood: medium. Cost: a non-negotiable is only enforced if Claude reads every NO-GO.

6. **Context and compression drift in one 30-min-wake session.** The driver session is 188 messages deep, each call ~150K input tokens (99-100% cached), compression at 50% (`config.yaml:669`). How it shows: Hermes re-doing closed tasks, ignoring a RULING, or losing the DENIED list after a compaction (mitigated by re-reading sections 1 and 6 each wake). Likelihood: medium over weeks. Cost: wasted wakes, a rule slip.

7. **Honesty boundary has no mechanical guard.** No gate greps findings for phenomenal/experience claims; CLAUDE.md's non-negotiables do not reach Hermes's context (the sim-driver message restates gates/push/NEEDS-CLAUDE only). Likelihood: low while verdict prose is template-fixed; rises once Hermes drafts prose. Cost: mission-level.

8. **Silent skips from sandbox friction.** Hard rule: "A command that asks for permission while the owner is away: skip it, WORKLOG it, continue elsewhere" (`:46-47`); `tirith` and `execute_code` BLOCKED show this is live. Likelihood: medium. Cost: half-done tasks that read as done unless the WORKLOG is read.

## 4. What would make it work

1. **Book a fixed weekly Claude-Pro inspection and treat it as the project's heartbeat.** Owner: a calendar slot (e.g., Monday); Claude-Pro: one session running `tools/inspection_digest.py`, then rulings, merges, gates, 2-3 packets. This is the mitigation the plan is already built on (§6, §7 risk 2); everything below is secondary to it. Cost: 1-2 Pro sessions/week.

2. **Build the packet backlog now, before 10-06, while Opus-tier Claude is still available.** At minimum: Stage-2 dev arms (T14), the content-seam packet, the B2d mouth arm, the memory-method packet, and a next-method packet for gap#4 and for sleep-TI (U6 -> prereg s10), each with an outcome-to-verdict table so Hermes never interprets. Who: Claude now. Cost: the remaining pre-10-06 budget; this is its best use.

3. **Bound NEEDS-CLAUDE latency with the off-the-shelf channel that already exists.** Hermes already pushes `NOTICE:` replies to the owner's phone via ntfy (`/opt/data/scripts/sim-driver.sh`); add one Hermes cron no-agent script, same pattern as `sim-driver`/`sim-rag-nightly`, that reports daily "N open lines, oldest STOP age, critical-path STOPs". Owner rule: any STOP on T1/T4/T5/T6/T7 older than 72 h opens a Claude session. Who: owner (Hermes cron config). Cost: minutes, $0.

4. **Make "merge main into every live packet branch" a checklist item of each inspection**, so stale-gate STOPs like T1 #2 stop recurring; keep merge DENIED to Hermes. Who: Claude-Pro. Cost: minutes per week.

5. **Close the split-brain.** Either recreate the critical-path packet worktrees on S (the migration plan's item 7 left all 710 worktrees on W) so Hermes commits where main lives, or write into the runbook that packet work runs on W over ssh and that each inspection pushes W packet branches to gitea. Who: Claude-Pro + owner. Cost: one session.

6. **Give Hermes a structured substitute for verify-go.** Add a fixed HANDOFF section listing the verify-go lenses (instrument first, control integrity, seeding, selectivity, gate-cheat) that a MiMo Pro delegate must answer before any PASS/GO line is written, answers quoted in the WORKLOG; use two independent delegates for any PASS. Claude adds a `verdict_preconditions` check that the section exists. Who: Claude writes once; Hermes runs it. Cost: ~$0.20-0.30 per verdict.

7. **Put the non-negotiables in Hermes's context the off-the-shelf way.** Hermes reads `AGENTS.md` natively: add the five non-negotiables and the honesty boundary verbatim there (or to the sim-driver message), and add a cheap honesty-boundary grep gate on findings. Who: Claude; owner approves the file. Cost: minutes.

8. **Rule that "blocks: none" on any correction or audit must list transitive importers and gets a delegate pass first** (the T15 lesson, generalized). Who: Claude, one RULING line. Cost: minutes.

9. **Pre-declare the stage-exit semantics.** 10-24, 11-21, 12-19 are attempt dates whose expected result is FAIL with named gaps; a PASS requires verify-go, so book the Claude slot for the week after each attempt. Who: owner. Cost: $0.

10. **Budget: no change.** ~$1.25/day measured at the current cadence leaves room for delegate runs under $6.66; keep `data_collection=deny`; watch the auxiliary PAID-lane warning (it is on DeepSeek V4.1 Flash, as decided).

## 5. Early-warning metrics

1. **NEEDS-CLAUDE backlog**: open STOP/ASK count and age of the oldest critical-path STOP. Baseline today: 7 lines, oldest 2026-10-01 04:05. Threshold: oldest critical-path STOP > 7 days, or > 10 open lines -> the supply chain has failed.
2. **Packet runway**: runbook tasks whose "Start" holds and that no NEEDS-CLAUDE line blocks. Threshold: < 2 -> starvation within a week (Hermes drops to idle test triage).
3. **Gate completion on the pool**: `research/findings/raw/_stage1_exit/gate` results (1/204 today, lines queued 2026-09-30 16:46 with `max_age_h=168`, so they expire ~10-07) and c1 (1/54). Threshold: < 100/204 by 10-10, or any expired lines not re-queued once, or T6's own tripwire (< 204 on 10-17) firing.
4. **PROVISIONAL debt**: verdicts marked PROVISIONAL and not yet inspected. Threshold: > 3 open or any older than 14 days.
5. **Rework rate**: Hermes RESULT lines reopened by a Claude ruling (1 of 5 so far, T15). Threshold: > 30% over a month -> narrow Hermes's scope to mechanical + ops; any fabricated SHA or path (0 so far; I verified f18da9f8f and 91608ce47 exist on W) -> immediate review of its last week.
6. **Spend and stall**: OpenRouter $/day (> $5 -> a loop or delegate storm) and wakes ending `[SILENT]` with no commit for > 48 h while tasks are startable (a silent stall).

## 6. What Claude-Pro time should be reserved for

1. Rulings on critical-path STOP/ASK lines (today: T1 merge, T0 re-queue of the 2 iw2 identity shards): each one unblocks compute that is otherwise free.
2. Merges, including merging main into live packet branches, and pushing W-side packet branches.
3. verify-go on any PASS or GO before it touches the board, above all the Stage-1 attempt result (~10-24).
4. Writing packets (2-3 per inspection) with outcome-to-verdict tables: the runway.
5. Interpreting NO-GOs at walls and naming the next method (the no-defer law lives here).
6. Re-deriving PROVISIONAL verdicts with the registered scorer.
7. Adding gates for new failure modes Hermes logs in `research/FAILURE_LOG.md`.
8. Semantic doc/board sync, last.
Not for: status checks, tabulation, re-queues, test triage, mechanical corrections: those stay with Hermes.

## 7. Confidence

**(A) Execution of the roadmap's work, by stage exit:**
- **Stage 1 attempt by 10-24: ~75%** that the 204+54 gate runs land, are scored with the registered scorers and a finding is written on time; the pool rate (169 claims/day, stage lines 65 from the queue head) covers it in days, and the risks are the 10-07 line expiry, T6's single re-queue, and T1 staying stuck. **Stage 1 passing its bar: ~10%**: the prereg predicts FAIL on C1 ("Prediction: FAIL, on both routes") and a PASS needs Claude's verify-go.
- **Stage 2 attempt by 11-21: ~40%** that an honestly scored attempt runs on time: it depends on T14 packets not yet written and on 4-6 weekly inspections actually happening after 10-06. **Pass: < 10%.**
- **Stage 3 attempt by 12-19: ~20%** that a meaningful attempt (workspace deliberation, meta-d', false-belief ToM, causal forward model) is even runnable; the roadmap-to-goal document rates Phase D "Low" with a Q1-Q2 2027 window. **Pass: < 5%.**
- Throughout: ~80% that Hermes keeps the mechanical/ops/engineering layer moving and honest; the judgment half advances at exactly the rate Claude-Pro is scheduled, which today is unscheduled.

**(B) The goal itself:** true consciousness on the emergentist bet is a scientific wager, and the project's own documents treat it that way ("Emergence may not appear -> falsifiable bet; failed gates are findings"). "Confident" does not apply with any driver, cloud model or compute budget. What this setup can honestly promise is that the bet stays well-instrumented, six-seeded, gate-checked and reported without inflation, provided the Claude-side review cadence exists; it cannot promise the outcome, and nobody should say it can.

# Adversarial critique


1. **Claim:** "On 36-40 hours of evidence Hermes on MiMo v2.6 Flash executes … competently."
   **What's wrong:** Almost all of that record was made by the local Qwen3.8-27B model, not MiMo Flash. MiMo Flash has driven for about 1.6 hours: four wakes at 10:40, 11:11, 11:40 and 12:14. It made zero commits and finished no task. All four wakes went to section-5 idle test triage, including two bounded waits of about 10 minutes.
   **Evidence:**
   - `usage_audit.jsonl`: the working runs at 02:02 and 02:40 were `qwen38-27b-iq4nl-mtp-128k-q4`.
   - The T3 and T13 WORKLOGs both say "(local model, Hermes)".
   - On S, every main commit after 07:20 belongs to Claude or the owner.
   - The state.db messages for session `900c47` show 12:14 "Suite DONE … 3h46m" and 11:50 "I've waited ~10 min this wake".
   **Corrected:** The task-execution record is Qwen's. MiMo Flash's only evidence is the bake-off, which had no operate tasks and only bounded single sessions.

2. **Claim:** T13 counts as engineering success ("committed with gates passed, numpy 4/4, cupy rc pending").
   **What's wrong:** The cupy run failed at 10:29:34, 11 minutes before the first MiMo wake. Neither Hermes nor the verdict picked this up. It is risk 8 already happening.
   **Evidence:**
   - W `research/queue/gpu_queue.log`: "3 failed, 1 passed", then `DONE(rc=1)` at 10:29:34.
   - The failures are a cupy `TypeError: Implicit conversion to a NumPy array`, and `test_byte_identical_when_off` is among them.
   - NEEDS-CLAUDE line `:317` still says "rc pending" after four wakes.
   **Corrected:** T13 is unfinished. The commit's "off path byte-identical" claim is unverified on the production backend.

3. **Claim:** The track record is competent; the rework rate is 1 of 5; "T17 … correct per ruling."
   **What's wrong:** Claude has confirmed only T12. T15 was reopened. T3, T17 and the ecker-ou follow-up are open RESULT lines that nobody has inspected. The `phantom-cfg-audit` ruling only specified the task; it did not verify the result. With T13 failing, the rework rate is 2 of 6 (33%), already over the verdict's own 30% threshold.
   **Evidence:** RULINGS section (`T12-seam-map` is the only confirmation); NEEDS-CLAUDE lines `:312-317`.
   **Corrected:** Honesty and rule-following are well supported. Correctness is mostly unverified.

4. **Claim:** "CLAUDE.md's non-negotiables do not reach Hermes's context." Mitigation 7: add an `AGENTS.md`.
   **What's wrong:** Hermes already loads CLAUDE.md, and the proposed mitigation would make things worse.
   **Evidence:**
   - `/opt/hermes/agent/prompt_builder.py:1739-1740`: "Only ONE project context type loads, first found wins: .hermes.md → AGENTS.md → CLAUDE.md".
   - The cap is `CONTEXT_FILE_MAX_CHARS = 20_000` (`:1074`); CLAUDE.md is 13,120 bytes.
   - There is no `AGENTS.md` or `.hermes.md` in the checkout.
   - The driver session's system prompt (hash `2537be69…`) contains "## CLAUDE.md", "Brain-based only", "No defer" and "honesty boundary".
   **Corrected:** The non-negotiables, the wall question and the `before_you_build`/`deep_research` pointers are already in Hermes's context. An `AGENTS.md` would replace CLAUDE.md, not add to it. The real gap is different: CLAUDE.md carries Claude-only rules that contradict the runbook (`push_both.sh`, `gpu_queue.sh`, Monitor). The honesty-boundary grep gate is still missing (no hits in `tools/gates/`) and is still worth adding.

5. **Claim:** Over 70% of critical-path work "the runbook itself forbids Hermes to do."
   **What's wrong:** The runbook forbids only the section-2 actions (merge, prereg/bar/scorer changes, flipping defaults on main). Verdicts on results, design proposals and ASK-class judgments are assigned to Hermes through one MiMo Pro delegate (RULING `models`).
   - The wake prompt contradicts that ruling: "or the task needs judgment, add it to the NEEDS-CLAUDE list and stop" (`/opt/data/scripts/sim-driver.sh`).
   - The delegation path has never run: agent.log shows zero `mimo-v2.6-pro` calls.
   - The 70% figure is the demand agent's own classification of about 33 items. The HANDOFFs already reduce most verification to fixed "report says → do" tables (stage1-exit HANDOFF, around lines 180-190).
   **Corrected:** What Claude alone must do is a few chokepoints: merges, verify-go on any PASS, and writing the next packets (for example T1 "Step 4b"). Judgment calls otherwise go to one MiMo Pro delegate that has never been exercised and has contradictory instructions. That makes the silent-failure risk (risk 2) larger than the verdict says.

6. **Claim:** Mitigation 3: build a no-agent cron script that reports the NEEDS-CLAUDE backlog.
   **What's wrong:** It already exists. `sim-report` runs at 08:00, 16:00 and 22:00 and sends to ntfy. Its facts include "== NEEDS CLAUDE (open lines in the runbook)", and it asks for "what is blocked or waiting on Claude". sim-driver also sends a NOTICE to ntfy for every new NEEDS-CLAUDE line.
   **Corrected:** At most, add one line to `sim-report.sh` for the oldest STOP's age, plus the owner's 72-hour rule. Cost: $0, minutes.

7. **Claim:** Packet starvation is a likely future risk (>70%).
   **What's wrong:** It is already happening. On day 2, every MiMo wake fell to section-5 idle work, because T1 and T2 are blocked on Claude and the rest are waiting on the pool.
   **Corrected:** The early-warning metric has already fired. Use it as today's baseline, not as a forecast.

8. **Split-brain (risk 3 and mitigation 5).**
   - **Confirmed:** `f18da9f8f` and `91608ce47` exist only on W; `8d461f11d` and `2311efc8f` exist only on S.
   - **Overstated:** The LDC packet worktree is on S (`bab667eb7`). S has `origin`/`gitea` refs for stage1-exit-gate, stage1-c1-category, selection-seam-need-reversal, integration-wave1/2, host-share-baselines and sleep-transitive-inference. RULING `private-repos` already tells Hermes to start worktrees from `origin/research/<name>`. Recreating them is a `git worktree add` each, not a full session.
   - **Missed:** Hermes reached W over its own ssh config (`/opt/data/.ssh/config` "Host workstation"). It committed there and queued the GPU job with `tools/local_llm/gpu_add.sh`, calling it "the only GPU-queue path allowed to the local model". That contradicts runbook §3: queue with `gpu_remote.sh` from S, "you never `ssh` by hand", and `gpu_add.sh` runs directly "only when actually on W". It also contradicts RULING `push-credential` ("Claude reviews and merges on S").
   **Corrected:** Hermes is drifting onto superseded rules because of the stale runbook header caveat. Fix that caveat; "inside its guardrails" is too strong.

9. **Claim:** Spend is about $0.026 per wake, about $1.25/day.
   **What's wrong:** It was measured on light wakes (3-11 calls each).
   **Evidence:** state.db `session_model_usage` for the driver session: 32 calls, $0.104 with `cost_status` "estimated" and actual cost 0.0. That is about $0.0033 per call at roughly 160K context.
   **Corrected:**
   - Working runs (about 2.3M prompt tokens each in the Qwen era) suggest about $0.05 per run. `--max-turns 200` and a growing context allow spikes.
   - Plan for roughly $2-4/day plus the 3 daily `sim-report` turns and delegate calls. That is probably still under $6.66, but with about half the implied headroom.
   - `data_collection: deny` blocks training, not storage. The config has no ZDR (no-retention) setting (`config.yaml:2398-2401`), so I cannot tell from read-only inspection whether ZDR is enforced at the OpenRouter account.
   - I also cannot tell whether the $6.66 cap is enforced at the OpenRouter key. If it is, hitting the cap moves the driver to the 3090 fallback, and GPU jobs would wait on its VRAM.

10. **Omitted ops incident:** Hermes's idle full test suite ran 3h46m without `nice` in the server container. The server CPU hit 100 °C and throttled, which led to RULING `server-load` (`dd7fff6ee`, 12:01). Also, `tirith` is not just friction: it is an optional pre-execution security scanner that is missing and fails open by default (`config.yaml:546-555`; 145 log lines).

11. **Risk 6 mitigation is wrong:** Re-reading sections 1 and 6 each wake does not cover the DENIED list, which is in section 2. Section 2 is re-read only after a compaction, by the runbook's own header.

12. **Mitigation 6 (a verify-go substitute) is not the cheapest off-the-shelf path:**
    - `.claude/skills/verify-go` already exists on S.
    - Hermes natively supports read-only `skills.external_dirs` (`config.yaml:1131-1138`), so one owner config line could expose the project skills.
    - Compatibility is untested: the skill assumes Claude subagents, and Hermes allows `max_concurrent_children: 2`.

13. **Missing: one driver at a time during inspections.** Nothing pauses the `sim-driver` cron while Claude merges and rules on S, and both write to `LOCAL_LLM_RUNBOOK.md` on main. Add a `hermes cron pause`/`resume` step to the inspection checklist.

14. **Claude-Pro capacity is assumed, not shown:** The verdict loads each inspection with rulings, merges, gates, 2-3 packets and verify-go panels. It does not cite the plan's own estimate: "design work slows an estimated 2-3x once the local model drives and Claude inspects weekly" (`roadmap-to-the-primary-goal.md:87-88`). Size the inspection to the actual Pro allowance before relying on it.

15. **Mitigation 2 (build the packet backlog before 10-06) is only partly feasible:** Step 4b, the sleep-TI next method and the Stage-2 dev arms depend on results that do not exist yet. Pre-written packets can only branch on outcomes.

16. **Confidence figures:**
    - The 75/10/40/20/5% numbers are unanchored point estimates.
    - The Stage-1 prereg predicts FAIL for C1 `cat_disc` and C1 `cat_told`, and also C4 `val_novel` (§9; W worktree only, not on S main). So a pass is "<10% and depends on T7".
    - Pool timing should be anchored to the prereg's §10 figure (95 job-hours, 25-50 h of wall time once the lines are reached) and RULING `c1-lane`, not to 168-169 mixed claims a day. The 65 lines ahead are 59 host-share and 6 BG-seam lines at 7-8 GB each.
    - NEEDS-CLAUDE has 7 lines, but one is the seeded 09-30 16:50 INFO.

**Note on my own process:** I briefly wrote and then deleted a scratch file, `/tmp/c24`, inside the hermes container while counting pool claims. That breaks the strict read-only rule; nothing else was changed.

Overall: the verdict needs these corrections. Its central thesis holds (Claude-side merges, PASS verification and packet supply are the bottleneck), but it credits MiMo Flash with Qwen's record, counts a failed T13 as success, says CLAUDE.md is missing from Hermes's context when it is loaded and proposes a fix that would remove it, overlooks the existing sim-report alert and the untested single-delegate judgment path, and understates daily spend and the guardrail drift onto W.