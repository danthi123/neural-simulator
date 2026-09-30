# Why six active months (not ten), and how to go significantly faster

*Acceleration report for the owner. Synthesized 2026-09-29 by Claude from seven evidence miners (git history, failures and retractions, owner directives, walls and dead ends, compute and infrastructure, process overhead, external options), merged with an independent verification pass (REPORT.md) that re-checked the miners' numbers against git and the cited files. Read-only: no repo file was changed.*

**How to read the evidence tags.** Every number carries a tag: `[G]` git-history miner, `[F]` failures miner, `[D]` directives miner, `[W]` walls miner, `[C]` compute miner, `[P]` process miner, `[X]` external miner, `[V]` re-verified by the synthesizer today with the command shown. Anything marked **(estimate)** is a judgment built on those numbers, not a measurement. Prices in section 5 are only the ones the external miner cited, with their sources; they are 2026 web prices and can change.

**Shorthand used in this report (defined once here).**

| Term | Meaning in this project |
|---|---|
| The brain / the substrate | The one spiking network (`sim/`) that must own all cognition between senses and action |
| Faculty | One capability of the brain (memory, affect, curiosity, ...); "on by default" means it runs in live chat |
| The mouth / Qwen | The small language model (Qwen2.5-0.5B) that only words what the brain decided; permanent since 2026-09-19 |
| Load-bearing fraction (LBF) | The #1 metric: the share of a reply that changes when a brain faculty is lesioned (switched off); "reply level" scores the words the user sees, "decision-field level" scores internal fields |
| Host-decision share | The new second metric (2026-09-25): how much of the answer is decided by host Python (loops, argmax) rather than by neurons, shown by intervention |
| Stage exit | The developmental milestones (Stages 1-6) in the master roadmap; none has been passed by the integrated brain |
| GO / NO-GO | A finding's verdict; GO must survive six seeds (42/43/44/100/101/102) and adversarial re-checking (`verify-go`) |
| gap#N | The old five-gap map; gap#4 = deep credit assignment on spikes, gap#5 = synaptic density |
| Wall | A method that stopped working; by the no-defer law it is a verdict on the method, never on the capability |
| The board | `GAP_CLOSURE_MISSION.md`, the state file CLAUDE.md says to read at every session start |
| Gate | A script in `tools/gates/` run by the pre-commit hook that blocks a known failure pattern |
| FAILURE_LOG / RETRACTED | `research/FAILURE_LOG.md` (one line per noticed failure) and `docs/RETRACTED.md` (voided claims) |
| Lane / pool / queue | Compute lanes: the local RTX 3090 (`tools/gpu_queue.sh`), the 3-node mini-PC CPU pool (`tools/sweep_pool.sh`), the AWS burst lane |
| Packet | A build finished to the point that a cheaper driver (the local model) can run it without judgment (transition plan section 2) |
| Prereg | A preregistration: bars frozen before the run, amendments dated |
| C2 bug | The 2026-09-02 read-isolation bug: `_hard_reset()` never restored refractory timers / firing state / homeostatic thresholds |

---

## 1. Executive summary

**The honest headline.** The "ten months" is really **six active months plus a four-month silence.** Zero commits landed between 2025-12-02 and 2026-04-05 inclusive, 125 days of no activity at all; counting the single-commit book-end days on either side of the silence (2025-12-01 `17dd8c47f`, then nothing until 2026-04-06 `cdf1f8b85`), the quiet stretch spans 127 calendar days, 42% of the ~305-day project span `[G][F][W][P][V]`. The repo records no cause for the gap — nothing in the memory directory or the project-history archive explains it, and this report does not guess.

Once active, the project ran hot: 11,521 commits in about 178 active days, roughly 65 a day, peaking at 444 on 2026-09-24 and 1,192 in the week of Sep 21-27 `[G][V]`. So the question is not "why so little activity" but **"why did six months of very high activity produce ~3,000 findings, 69 tracked faculties, 30 on by default, and no stage exit?"** (`docs/plans/2026-09-29-roadmap-to-the-primary-goal.md` section 2 `[V]`).

**The seven biggest findings.**

1. **The work optimized for findings, not for the integrated brain.** 62.6% of all commits touch `research/findings`, 35.2% touch `research/runners`, and only **2.7% (311 commits) touch `sim/`**, the one spiking substrate the non-negotiables are about `[G][V]`. Roughly 750-820 positive-GO findings were banked `[G]`, yet 39 of 69 faculties (57%) are still not on by default `[D]`, and no stage-exit gate has ever been attempted end to end `[D][W]`. De-risking in isolation was the product; integration was the residual.
2. **A large share of every month's commits are rework, and the size of "large" depends on how you count.** Fix/correct/retract/void/revert keywords anywhere in the full commit message (subject + body) appear in 41-64% of commits every month since April (Apr 55%, May 56%, Jun 41%, Jul 64%, Aug 50%, Sep 53%) `[G]`; the same keywords in the subject line only, the more conservative count, give 16-29% (Apr 18%, May 16%, Jun 17%, Jul 29%, Aug 19%, Sep 26%, re-verified today by `git log --pretty='%ad|%s'` `[V]`).
   Either way Jul and Sep are the two worst months. One commit in eight (12.1%) is a retraction, correction or NO-GO bookkeeping commit `[F]`. The largest single cluster was the C2 bug, which flipped nine banked "GO 6/6" verdicts to NO-GO/partial in one day (2026-09-02, confirmed by 9 dated rows in `docs/RETRACTED.md` `[V]`) across pool classes that each hand-rolled the same reset; 26 runner files still define their own `_hard_reset`/`_reset_dynamics` today, `sim/` defines none `[F][D][V]`.
3. **The #1 metric was overstated, and it took outsiders to notice.** LBF reads 0.843 at the decision-field level but 0.535 at the reply level, the level that matters; the 2026-09-29 roadmap lists "the metric overstated by counting decision fields" as measurement debt, and the 2026-09-25 phase was "adopted by the owner after two outside AI reviews" `[D][W][V]`. The metric was ratified 2026-09-19 without its own instrument being audited; the debt (three organs ignoring the seed, an unapplied replay-noise setting, a crashing lesion arm) surfaced ten days later `[W]`.
4. **Direction changes accelerated roughly 8x.** ~1 per 59 days in the first 236 active days (4 changes) versus ~1 per 7 days in the last 49 (7 changes) `[D][G]`. The own-voice mouth arc (68 findings, 110 commits in Sep 1-19) was pursued under a "PURSUE OPEN FLUENCY FULLY" note of 2026-09-03 and retired as falsified on 2026-09-19 `[D][W]`. 127 of 576 `research/*` branches (22%) were never merged, all from the Aug/Sep worktree era `[G][V]`.
5. **Walls were usually instruments, not biology.** The gap#5 "learning" was 97% a `w_max` clamp (identical with learning off); the gap#4 LIF wall stood ~10 days and "was a knob" (a shared learning rate); the 28-word wall was an unequal training count, overturned in one day; the satdiv vision verdict oscillated for 29 days around a 0.1-unit scale flip `[W]`. The project's own 2026-07-31 synthesis says so verbatim, and the pattern recurred after being named, because no gate enforces a matched control `[W]`.
6. **The GPU ran more than it idled, but a 90-hour dead window and a third of the CPU pool being unreachable still cost real days.** Correction: an earlier pass on this report misread the queue log and reported the GPU as 57% idle; re-verified directly against `research/queue/gpu_queue.log`, the 41.51-day window (2026-08-19 to 09-29) was busy 23.65 days (57%) and idle 17.9 days (43%) `[C][V]`.
   The single largest idle gap was 90.1 hours, 2026-09-25 18:22 to 2026-09-29 12:25, spanning the 3090 crash (confirmed by direct timestamp diff, not the 43.9h figure an earlier pass cited) `[C][V]`. Separately, 29.4% of GPU-queue jobs ended non-zero; the audit printed UNDER-PARALLELIZED for ~14.5 days straight (Sep 8-23) behind a waiver loophole; and `pool40` was UNREACHABLE in 2,390 of 2,389 sync cycles while `pool41`/`pool42` were unreachable 21-22 times `[C][V]`.
7. **Process grew faster than product, and the agent (Claude) caused a good share of it.** Gates plus hooks are 12,555 lines against 32,964 lines of `sim/` (38%); only 10 of 48 gates have a test; two sampled gates shipped broken (one at 0 true positives / 3 false positives) and needed three review rounds each; the board is 8,406 lines with 71 undeleted snapshots and is mandatory reading every session; commit messages average 117.5 words `[P][V]`. Claude's own recorded lapses include writing a verify-go rule and violating it the same day, a monitor false positive that declared running arms crashed, an Opus fan-out that burned ~50% of a weekly budget in 1.5 days, and 48 owner corrections in 133 days, about one every 2.8 days `[D][F][P]`.

**The five highest-value moves** (all free or near-free; details in section 4).

| # | Move | Why it is first |
|---|---|---|
| 1 | **Integration-first with a real stage-exit attempt this month**: freeze new-faculty de-risking; every research slot converts one de-risked faculty to on-by-default with a reply-level LBF number, and the Stage 1 exit gate is run end to end even if it fails | Directly attacks finding 1; 39 faculties are waiting; zero stage exits attempted in six active months |
| 2 | **Report only the reply-level LBF, always beside the host-decision share, and verify-go any new instrument the same session it is ratified** | Finding 3; the 10-day window of decisions taken on an inflated number; already scoped in the 09-25 and 09-29 plans |
| 3 | **Make the C2 class unrepresentable (one shared reset primitive + a CI invariant) and add a fairness pre-check before any wall/NO-GO is filed (equal data, per-arm learning rate, equal seeds, production-substrate smoke first)** | Findings 2 and 5; the two largest identified rework clusters, both recurring after being named |
| 4 | **Pivot dwell + retirement list**: no phase-level redirect for four weeks unless a decisive falsification forces it, and every pivot commit lists which branches it retires or banks | Finding 4; the 09-19 arc shows a run-to-falsification produced a durable decision, mid-course reframes did not |
| 5 | **Compute and process hygiene in one sweep**: fix or retire `pool40`, auto-requeue jobs killed by pause/OOM/memcap (rc 143/137/3), a reboot-only watchdog for the 3090, archive the board's 70 historical snapshots, add a cheap cloud LLM profile (`llm use`) for the mechanical tier before the Pro downgrade on 2026-10-06 | Findings 6 and 7; each item is a few hours and reuses existing tooling |

**One number to keep in view.** If the dormancy gap is excluded, the project's per-active-day output is already high; the lever is *conversion*, not effort. **(estimate)** Reworking the same findings twice (41-64% churn) plus the doc/board share (up to 39% of monthly commits in August) means that well under half of the active period's commit volume was first-pass work on new capability, and only a sliver of that reached the integrated brain.

---

## 2. How the ten months actually went

### 2.1 Dated narrative

| Period | What happened | Evidence |
|---|---|---|
| 2025-11-29 to 12-01 | First commits (10 in Nov, 42 in Dec); the simulator's origin | `git log --reverse`; `c58c97c56` `[G][F][V]` |
| 2025-12-01 to 2026-04-06 | **Zero commits for 125 days** (2025-12-02 to 2026-04-05 inclusive); counting the two book-end commit-days the silence spans 127 calendar days, 42% of the span. The synthesizer does not know why (life, other work, or a deliberate pause); nothing in the repo records the reason. CLAUDE.md and the whole gate/board apparatus first appear on 2026-04-06, so the project's process scaffolding post-dates the gap | `git log --since=2025-12-02 --until=2026-04-05` = 0, re-verified `[V]`; last pre-gap `17dd8c47f` (2025-12-01), first post-gap `cdf1f8b85` (2026-04-06) `[G][W][V]`; `[P]` on CLAUDE.md |
| 2026-04 | Restart and ramp: 451 commits | monthly histogram `[V]` |
| 2026-05 | Full speed: 1,978 commits. 05-11 standalone-agent plan; 05-19 "conversation via SPEAR + replay, not RAG"; first consolidation (A1) finding 05-07; the 28-word frontend wall declared 05-31 and overturned 06-01 (unequal training counts) | `[D][W]` |
| 2026-06 | 2,496 commits. 06-04 consolidate the conversational pipeline onto the core sim; 06-08 brain-based-only standard; 06-17 profile: 97.7% of a composer operation is 3,000-4,000 tiny CUDA kernel launches; **06-18 TRUE ONE BRAIN directive** (`d463f9046`); 06-22/23 artificial-life north star | `[G][D][X][V]`; `docs/plans/2026-06-18-hardware-acceleration-neuromorphic-roadmap.md` lines 30, 99-100 `[V]` |
| 2026-07 | 2,584 commits (the monthly peak). gap#4 and gap#5 arcs open (07-17/18). **07-23 MAJOR DIRECTION PIVOT to genuine cognition** (`89ef28fa7`) and the master roadmap. 07-19 a "3 arms crashed" monitor false positive (`212d3ae98`, which also rewrote 34.2M lines of line endings). 07-28 verify-go rule 3 written and violated the same day, 9 retractions in one session. 07-31 the failure taxonomy (21 agents, 11 classes, 32 FAILURE_LOG rows) and "the workflow is now enforced, not remembered" | `[G][D][F][P][V]` |
| 2026-08 | 1,802 commits (the dip). 08-02 gap#4 LIF "deep-credit wall"; 08-06 consolidation interference wall closed via SFA eviction (91 days after the arc opened); **08-11 INTEGRATION-FIRST PIVOT** (`e2e6a499f`) and the gap#4 ALLIN summary ("the wall stood ~10 days as biology; it was a knob"); **08-19 STRATEGIC REFRAME** Qwen-as-form (`2f8840ce0`) and the Opus fan-out cost blowout (~50% of the weekly budget in 1.5 days); 08-25 order-consolidation toy GO and real-scale CA1 port GO the same day; 08-26 second 34.4M-line line-ending rewrite (`0e22e97f1`); 08-27/28 stale-cache retractions. Merges jump from near zero to 152 | `[G][D][W][P][V]` |
| 2026-09 | 2,158 commits through 09-29. **09-01 auto-flip directive** (`c1e57875d`); 09-01/02 the C2 bug flips 9 GOs in a day, 43 runners audited; 09-03 "capability_go was 0/6 all along" (`41d4604b4`) and the first spiking-mouth NO-GOs (`e9a6d1e29`, `00a6b5a6e`); 09-04 scaffold-retirement re-order; 09-06 local agent-stack review; 09-08 to 09-23 UNDER-PARALLELIZED for ~14.5 days; **09-19 Qwen ratified as permanent mouth, LBF becomes #1** and the own-voice arc is closed as falsified; 09-22 "overclaimed fac_g robustness" caught (`73d1a2cbc`); 09-23 the waiver loophole is closed (`fd314254d`, `ad6ec8577`), the audit reads SATURATED, commits hit 331/444/353 on 09-23/24/25, and 112 of all 223 FAILURE_LOG rows are written in three days; **09-25 prove-who-owns-the-computation** after two outside AI reviews; 09-26 a 3090 "off the bus" crash; 09-28 the auto-reboot watchdog retired; **09-29** the roadmap-to-goal and the local-model transition plans, measurement debt on the #1 metric recorded, Max-to-Pro downgrade a week away; the local-serving bake-off shows the default llama-server profile passing 3/3 while ollama and unsloth-studio fail. Merges: 465 | `[G][D][F][C][W][V]` |

### 2.2 The velocity picture

**Commits per month** `[V]`: Nov 10, Dec 42, (gap), Apr 451, May 1,978, Jun 2,496, Jul 2,584, Aug 1,802, Sep 2,158 (through 09-29, re-verified today: `git log --date=format:%Y-%m --pretty=%ad | sort | uniq -c`, total 11,521 `[V]`). Commits landed on 153 of ~178 active days (86%), mean 75 per active day, median 63 `[G]`.

**What the commits touched (all-time, non-exclusive)** `[G][V]`:

| Path | Commits | Share |
|---|---|---|
| `research/findings` | 7,213 | 62.6% |
| `research/runners` | 4,057 | 35.2% |
| `docs/` | 1,170 | 10.2% |
| `GAP_CLOSURE_MISSION.md` alone | 1,098 | 9.5% |
| `webapp/` | 613 | 5.3% |
| `tools/` | 549 | 4.8% |
| `ROADMAP.md` | 359 | 3.1% |
| `sim/` (the substrate) | 311 | 2.7% |
| `tools/gates/` | 82 | 0.7% |

**Doc/board share of monthly commits rose while research share fell** `[G]`: 5.2% (Jun) to 17.8% (Jul), 39.1% (Aug), 30.4% (Sep); runners+findings fell from ~72% (May) to ~60% (Sep).

**GO throughput plateaued.** Filenames with a standalone GO token added per month: May 4, Jun 129, Jul 288, Aug 259, Sep 220 (about 82 of the ~900 are NO-GOs caught by the same pattern) `[G]`. Across the 2,993 findings, 815 filenames carry `-GO`, 147 carry NO-GO/NOGO, 60 "wall", 24 UNDEFINED; 194 (6.5%) carry a retraction/negative marker `[W][F][V]`.

**Merges are a last-two-months phenomenon.** 617 of the 631 merges in history (98%) landed in August (152) and September (465), when the parallel-worktree workflow ramped up; 722 worktree directories exist under `.claude/worktrees` `[G][P][V]`.

**Repo mass.** 58,933 file-changes, 114.2M insertions, 68.9M deletions all time; two commits (`212d3ae98` 2026-07-19 and `0e22e97f1` 2026-08-26) each rewrote ~34M lines of line endings (insertions equal deletions) and together account for ~60% of all insertions ever `[G][V]`. `research/findings/raw` holds 31,497 committed JSON result files (17 GB); 319 plan documents exist under `docs/plans/`; the board is 8,406 lines `[G][V]`. **The repo itself is heavy: `.git` is 25 GB** (`du -sh .git`, re-verified today), most of it the raw-JSON and line-ending history above `[V]`.

### 2.3 Pivots and reframes

| Date | Redirect | Source |
|---|---|---|
| 2026-05-11 | Standalone-agent plan | `[D]` |
| 2026-05-19 | Conversation via SPEAR + replay, not RAG | `[D]` |
| 2026-06-04 | Consolidate the conversational pipeline onto the core sim | `[G]` |
| 2026-06-08 | Brain-based-only standard | `[D]` |
| 2026-06-18 | TRUE ONE BRAIN (`d463f9046`) | `[G][V]` |
| 2026-06-22/23 | Artificial-life north star (`31473deec`, `62f0a2c43`) | `[G]` |
| 2026-07-23 | MAJOR DIRECTION PIVOT: genuine cognition, master roadmap (`89ef28fa7`) | `[G][V]` |
| 2026-08-11 | INTEGRATION-FIRST: all faculties on by default (`e2e6a499f`) | `[G][V]` |
| 2026-08-19 | STRATEGIC REFRAME: Qwen as form, continuous substrate (`2f8840ce0`) | `[G][V]` |
| 2026-09-01 | Auto-flip validated GOs to default-on (`c1e57875d`) | `[G][V]` |
| 2026-09-04 | Scaffold retirement before learning ("re-orders" the 08-19 reframe 16 days later) | `[D]` |
| 2026-09-06 | AGI-fork openness / per-mechanism realism trades | `[D]` |
| 2026-09-19 | Qwen permanent mouth; LBF is #1; own-voice arc falsified | `[D][W]` |
| 2026-09-25 | Prove who owns the computation (after two outside reviews) | `[D][V]` |
| 2026-09-29 | Roadmap to the primary goal; local-model transition | `[V]` |

Commit-message hits: "pivot" 245, "reframe" 287, "north-star" 54, "owner directive" 54 `[G]`. Many of these redirects were drafted by Claude and ratified by the owner, and several (08-11, 09-25) were corrections of Claude's own overclaiming or drift; the pivot cost is shared, not the owner's alone.

### 2.4 The long arcs

| Arc | Span | Size | Outcome |
|---|---|---|---|
| gap#4 deep credit on spikes | 2026-07-18 to 09-25, 69+ days, still UNDEFINED | 110 findings, 469 commits | LIF wall (08-02) was a per-arm learning-rate knob (08-11); toy-substrate results did not transfer to the Izhikevich bridge; the production few-spike wall isolated 08-02/08-19; still open `[W][V]` |
| gap#5 density | 2026-07-17 to 09-17, 62 days | 84 findings | 97% of the measured weight change was the `w_max=150` clamp against `W0=250`, identical at lr=0 `[W]` |
| Compositional consolidation (A1) | 2026-05-07 to 09-08+, ~4 months | v1 to v9 | Interference wall closed 08-06 (SFA eviction); toy and real-scale CA1 port GO the same day 08-25; v9 sparsity rung NO-GO 09-08; still not wired into production `[W]` |
| Own-voice / spiking mouth | 2026-08-08 to 09-19, ~6 weeks | 68 findings; 110 commits in Sep 1-19 | Three independent falsifications; retired 09-19; had been blocking ~56 of ~69 production rows tagged BLOCKED:neural-render `[W][D]` |
| Vision readout (satdiv) | 2026-08-19 to 09-17, 29 days | NO-GO to BORDERLINE to GO | Verdict flipped on a 0.1-unit scale change; "capability_go was 0/6 all along" (`41d4604b4`) `[W][G][V]` |
| 28-word frontend wall | 2026-05-31 to 06-01, 1 day | GATE1 to GATE2 | Unequal training counts (~50 vs 200 events/word) `[W]` |

---

## 3. What slowed us down: root causes ranked by estimated cost

### 3.1 The ranking

Costs are in calendar-days-equivalent where a measurement exists and are otherwise marked **(estimate)**. "Active" means the cause is still operating today.

| Rank | Cause | Category | Estimated cost | Active | Evidence |
|---|---|---|---|---|---|
| 1 | Four-month dormancy (2025-12-02 to 2026-04-05) | (e) owner bandwidth | 125 days measured, 41% of the span | No | `[V]` zero commits in range |
| 2 | Breadth de-risking in isolation instead of integrating; no stage exit ever attempted | (a) research strategy | **(estimate)** the dominant active-period cost: most of ~6 months' output is not in the integrated brain | Yes | 39/69 faculties off by default; `sim/` 2.7% of commits; 0 stage exits `[D][G][V]` |
| 3 | False GOs from instrument bugs, then mass rework | (b) measurement | 41-64% of monthly commits carry rework keywords; >=11 finding cycles redone in ~1-2 weeks (Aug 27 to Sep 5) `[G][D]`; **(estimate)** 25-40% of active commit volume is second-pass work | Yes | C2 bug: 9 GOs flipped in a day, 43 runners audited; stale COO cache; hollow "production default" flip `[F][D]` |
| 4 | Pivot frequency and stranded work | (a) research strategy | 3-4 weeks of research capacity on the own-voice arc retired 2.5 weeks after "pursue fully"; 127 unmerged branches `[D][G]` | Yes | 7 redirects in 49 days vs 4 in 236 `[D]` |
| 5 | Walls that were knobs; toy substrates that did not transfer; late ceiling and matched controls | (a) research strategy | 1 day (28-word) to ~10 days (gap#4 LIF) per instance; "weeks" of gap#4 toy findings `[W]`; 29-day satdiv oscillation | Yes | 2026-07-31 synthesis; 2026-08-11 ALLIN summary `[W]` |
| 6 | Idle and failing compute | (c) infrastructure | 14.5 days under-parallelized (4.8% of the span); GPU log busy 57% / idle 43% of the last 41.5 days; 29.4% of GPU jobs non-zero; CPU pool at 2/3 nodes for the whole retained log; up to 90.1h dead after a crash `[C][V]` | Yes | `research/queue/gpu_queue.log`, `pool_sync.log`, board lines 554-584 |
| 7 | Process and coordination overhead | (d) process | doc/board share up to 39% of monthly commits; 8,406-line mandatory read per session; ~3 review rounds per new gate; **(estimate)** 60-100 hours of commit-message writing; a waiver meta-process needing rounds 2-4 `[G][P]` | Yes | section 3.5 |
| 8 | The #1 metric ratified without auditing its instrument | (b) measurement | 10 days of decisions on an overstated number; caught by outsiders `[W][D]` | Yes (debt open) | roadmap 09-29 section 2 `[V]` |
| 9 | Claude token-budget blowouts | (d) process | ~50% of a weekly budget in 1.5 days (08-19); 10% in 3 hours after the 09-29 reset | Partly | cost-routing skill; transition plan header `[P][V]` |
| 10 | Solo researcher, no external review, no external funding or compute | (e) owner bandwidth | Unquantifiable; the overstatement in rank 8 was caught only from outside; $50/day AWS cap and one 3090 remain the ceiling | Yes | 11,516 of 11,521 commits (99.96%) one author `[V]`; no preprint/grant artifact in the repo `[X]` |

### 3.2 (a) Research-strategy causes

**A1. De-risking in isolation instead of integrating (rank 2).** The commit paths tell the story: findings and runners dominate, `sim/` is 2.7% `[G][V]`. The 2026-08-11 integration-first pivot named the problem but the mix did not change: August and September still produced 259 and 220 GO-tagged findings while the on-by-default count is 30 of 69 `[G][D]`. The 2026-09-19 roadmap recorded that ~56 of ~69 production rows were blocked on one dependency (neural render), i.e. integration was serialized behind a single arc that was then falsified `[W]`. *Fix:* section 4, MUST-1: an integration slot per cycle, a stage-exit attempt now, and "done" redefined as on-by-default with a reply-level LBF number.

**A2. Pivots (rank 4).** The same axis (how much host scaffold, how permanent the mouth) was re-litigated on 08-08, 08-19, 09-04 and 09-19 `[D]`. Each redirect used language that superseded in-flight plans ("NEW TOP-LEVEL DIRECTIVE", "supersedes gap#4-as-crux") without listing what it retired `[G]`; 22% of research branches were never merged `[G][V]`. The durable decision (09-19) came from running an arc to three falsifications, not from a reframe `[D][W]`. *Fix:* MUST-4: a four-week dwell and a retirement list per pivot.

**A3. Walls that were instruments (rank 5).** Named in the project's own words on 2026-07-31 ("Most walls here were not the biology being hard. They were measuring the wrong quantity and believing it") and again on 2026-08-11 ("Verify a wall on the PRODUCTION substrate before treating it as real") `[W]`. Both lessons were banked as prose; neither became a gate, and the pattern recurred (satdiv, 29 days) `[W]`. *Fix:* MUST-3b: a fairness pre-check gate and a production-smoke rule before any toy scaling; SHOULD-10: a replicated coarse-to-fine grid at the first BORDERLINE.

**A4. Long arcs without a ceiling bound.** gap#4 alone is 110 findings and 469 commits over 69+ days and still UNDEFINED `[W][V]`; the 2026-08-11 summary estimates the substrate-transfer lesson "would have saved this arc weeks" `[W]`. *Fix:* run the ceiling/reference on day one of an arc (the `neural-simulator` skill already says so); a time-boxed review at day 14 of any arc that asks "is this a knob?" before more findings are filed.

### 3.3 (b) Measurement and integrity causes

**B1. Shared-code defects audited per runner (rank 3).** The C2 bug (`_hard_reset()` incomplete) was rediscovered in eight pool classes and flipped nine GO 6/6 verdicts on 2026-09-02; 14 runners were audited that day and ~29 more flagged `[F][D]`. Two more retractions (08-27/28) came from a stale COO transmission cache; one "production default" claim was hollow because a field was structurally None on all real traffic `[D]`.

Detection, once looked at, is fast: 14 of 21 registered retractions happened the same day as the finding, none later than 6 days `[F]`. The bottleneck is the production rate of bad GOs, not slow detection; 26 runner files still hand-roll this same reset today, and `sim/` shares none `[V]`. *Fix:* MUST-3a: one shared, tested reset primitive plus a CI invariant ("no state array differs after reset() + no-op") so the class is unrepresentable rather than detectable.

**B2. The #1 metric overstated (rank 8).** 0.843 at decision-field level vs 0.535 at reply level; three organs ignore the seed; a replay-noise setting never applied; a lesion arm crashes `[V]`. Ratified 09-19, debt found 09-29, prompted by two outside AI reviews `[W][D][V]`. Earlier instances of the same class: "capability_go was 0/6 all along; reframe-control 93% was a 1-seed artifact (real 6-seed number ~11%)" (`41d4604b4`, 09-03); "overclaimed fac_g robustness (adversarial verify caught it)" (`73d1a2cbc`, 09-22) `[G][V]`; overstated claims still live in the board at lines 6226 and 6624 `[F]`. The terminology-overclaim gate is report-only, not blocking `[F]`. *Fix:* MUST-2: reply-level only, host share beside it, and verify-go pointed at the instrument in the ratification session.

**B3. Bursty triage.** 112 of 223 FAILURE_LOG rows (50%) were written in one three-day window (09-23 to 09-25); most of August has near-zero rows `[F]`; 124 rows (56%) are NOT-GATEABLE with no mechanical block `[F][V]`; gates marked BLOCKS still show new incidents months apart for the same class (lane_starvation 6 incidents 07-31 to 09-27) `[F]`. *Fix:* SHOULD-7: a weekly CPU-only triage sweep and mechanizing the top three NOT-GATEABLE classes by cost times recurrence.

### 3.4 (c) Infrastructure and compute causes

**C1. Under-parallelization behind a waiver loophole (rank 6).** The audit printed UNDER-PARALLELIZED for ~14.5 days (Sep 8-23), root-caused to rewritable waiver files with unkept promises, a serial one-run-at-a-time habit, one agent running against a floor of three, `pool41` unusable for days, and memcapped GPU jobs failing silently with rc=3 `[C]`. When `fd314254d` fixed the counting and the waivers were retired, daily commits went from 100s to 331/444/353 `[C][V]`.

**C2. The GPU runs more than it idles, but the idle windows that do happen are long.** Over the 41.51-day `gpu_queue.log` window (2026-08-19 to 09-29), the queue was busy 23.65 days (57%) and idle 17.9 days (43%) `[C][V]` — a prior pass on this report had the busy/idle figures swapped and stated 57% idle; re-checked directly against the log's START/DONE timestamps today. The largest single idle gap is 90.1 hours, 2026-09-25 18:22 to 2026-09-29 12:25, spanning the 3090 crash (verified by direct timestamp diff; a prior draft cited a different, smaller gap) `[C][V]`. Caveat kept from the verification pass: jobs launched outside the queue and gaming pauses are invisible to this log, so it undercounts true occupancy in either direction.

Of 637 completions, 450 (70.6%) exited 0; 96 rc=1, 59 rc=143 (SIGTERM: pause/preemption/timeout), 12 rc=3 (memcap silent fail), 8 rc=137 (OOM), 7 rc=127 (command not found) `[C][V]`. The auto-reboot safety net was retired 2026-09-28 ("nothing auto-reboots after a GPU crash any more", `docs/GPU_CRASH_RECOVERY.md` line 65 `[V]`).

**C3. A dead pool node.** `pool40` UNREACHABLE in 2,390 of 2,389 cycles; `pool41`/`pool42` 21-22 times; the retired `pool1`/`pool2` 371-378 times before removal `[V]`. The CPU pool has effectively been two nodes for the whole retained log.

**C4. GPU contention with the local model.** The default local-model profile peaks at 22.8 GB of 24 GB; the automatic swap stops the driver's own server mid-session (transition plan section 3 `[V]`); the explicit-yield design is written but not built as of 09-29 `[C][V]`.

**C4a. The local serving bake-off: one profile proven, two not.** `tools/local_llm/results/summary.md` and the 2026-09-29 17:57 rows of `research/queue/gpu_queue.log` (worktree `.claude/worktrees/serving-bakeoff`) show the default profile, `qwen38-27b-iq4nl-mtp-128k-q4` served via llama-server (the same 22.8 GB profile as C4), passing all three verification tasks (T1/T2/T3 all PASS). The other two engines tried the same run both failed: `ollama-qwen38-27b-iq4nl` failed all three tasks (T1/T2/T3), and `unsloth-studio-qwen38-27b-iq4nl` returned `HTTPError 400` on every task `[C][V]`.

So the 2026-10-06 Pro-plan handoff already has one bake-off-passing profile to run on, not zero as an earlier draft of this report concluded; what is unproven is only the easier-to-operate alternative servers, not the driver itself.

**C5. Faithfulness cost is structural, and a bad flag re-spends it.** Single dev seeds of sleep transitive inference take 17-70+ minutes; "gap#4 wall was a hyperparameter" means at least one full battery was burned on a bad setting `[C]`. This is the mandated cost of non-negotiable #4, not waste; the waste is running six seeds before one smoke seed passes.

### 3.5 (d) Process and agent-management causes, including Claude's own lapses

**D1. Enforcement code outgrew the product.** 12,555 lines of gates and hooks vs 32,964 lines of `sim/`; 48 gates, 10 with tests; `prereg_amendment_order.py` is 951 lines for ~36 amendment events in 2,500 commits; its first version scored 0 TP / 3 FP; the ssh-stdin gate's first version caught 1 of 3 real incidents; each needed three opus-tier rounds `[P]`. "round 2" x21, "round 3" x19, up to "round 8" x3; 15 SOUND-WITH-ISSUES re-review commits `[P]`. The waiver system that legitimizes `--no-verify` (40 commits, all citing lane starvation) needed "fix rounds 2-4" in the week of 09-23 `[P][V]`.

**D2. The board only grows.** 8,406 lines, 71 undeleted "STATE OF THE PROJECT" snapshots, lifetime +11,702/-3,321 lines, touched in 9.5% of all commits, and mandatory reading at every session start and continuation `[P][V]`. CLAUDE.md itself was archived to `docs/CLAUDE_RATIONALE_ARCHIVE.md` on 2026-09-25; the board never got the same treatment `[P]`.

**D3. Verbosity as a tax.** ~1,353,796 words across 11,521 commit messages, ~117.5 words each `[P]`; 970 `docs:`, 361 `board:`, 499 `state:`, 573 `cycle` prefixes (about 16-21% of subjects are bookkeeping) `[G]`.

**D4. Claude's own recorded lapses.** Being candid, because these are the ones a process change can remove:
- Wrote verify-go rule 3 and violated it the same day (2026-07-28), yielding 9 retractions in one session; the parallelize memory existed while the GPU sat at 0% `[D]`.
- Declared three gap#4 arms crashed while they were running at 100% CPU (`212d3ae98`, 07-19) `[G][V]`.
- Ran untiered Opus fan-outs (a 6-agent doc workflow ~500k tokens) until ~50% of a weekly budget was gone in 1.5 days (08-19), and spent 10% of the budget in the first three hours after the 09-29 reset `[P][V]`.
- Let the audit say UNDER-PARALLELIZED for 14.5 days while writing waiver files that carried unkept promises `[C]`.
- Overclaimed results that adversarial re-checks later corrected (`73d1a2cbc`, `41d4604b4`) and counted decision fields as the #1 metric until outside reviewers objected `[G][W][D]`.
- Shipped gates broken (0 TP / 3 FP) and fixed them on main across three review rounds `[P]`.
- Repeated the same zsh word-splitting bug three times in one day (2026-09-23); six separate memory files independently correct variants of unmonitored or stalled runs `[D]`.
- 48 owner-correction memory files in 133 days (one every 2.8 days); "#1 failure: milestone, report, STOP" is still the first working-discipline line in MEMORY.md as of today `[D]`.
- Appended to the board for months without trimming it; wrote 117-word commit messages on mechanical commits `[P]`.

**D5. Bookkeeping that recurs because the root cause is untouched.** Two whole-repo line-ending rewrites six weeks apart (no `.gitattributes` enforcing LF) `[G][V]`; 31,497 raw JSON results committed to git `[G]`.

### 3.6 (e) Owner-bandwidth causes

**E1. The dormancy (rank 1)** is the largest single number in this report and is entirely outside the repo's technical story. Whether any of it was avoidable is the owner's call; the levers below only make a recurrence visible.

**E2. Solo, unreviewed, unfunded.** 99.96% of commits from one author; no arXiv/bioRxiv submission, no grant application and no public thread anywhere in the repo `[X]`. The consequence showed on 2026-09-25: the overstatement was caught by two outside AI reviews, not by the internal process `[D][V]`. The 09-25 directive to seek outside checks is the right instinct; section 6 makes it structural.

**E3. Corrections as the owner's main lever.** 48 recorded corrective directives, 47 gates "because I happened to notice and happened to act" `[D]`. The owner has been the project's gate of last resort; every lever in section 4 that converts a remembered rule into a check reduces that load.

### 3.7 What worked (keep these)

- The parallel-worktree merge workflow: 98% of all merges in the last two months; the week of Sep 21-27 was the project's peak (1,192 commits) `[G][V]`.
- Gates that block: no new instance of the exact caught pattern after enforcement for plasticity-bound-trap, single-seed-headline, flip_offarm_staleness, workflow_cost_tiering `[F]`; the rebuilt ssh-stdin gate replays 0 FP over 486 commits and catches 24/24 mutations `[P]`.
- Partial retractions with named survivors (`docs/RETRACTED.md`), and 0-6 day detection lag once a result is looked at `[F]`.
- ALLIN arc summaries and the wall-reframe question ("what did we replace with a constant?"), which closed gap#5's clamp and the consolidation interference wall `[W]`.
- Running an arc to falsification (09-19) rather than reframing it produced the one durable decision `[D][W]`.
- Parallel dispatch when used: gap4-crux 15 cells in 13.5h vs ~101h serial (7.5x); a 10h pool battery in ~20 minutes on a 10-parallel AWS r7i.4xlarge (~30x); the $50/day cap and idle-stop actually stop instances `[C]`.
- Six-seed discipline is consistently applied; nearly every retraction cites an explicit seed ratio `[D]`.
- Brian2 as a free CPU oracle; the local Qwen3.8-27B deployment at zero hardware cost; shelving the second 3090 before the used price roughly doubled `[X]`.

---

## 4. Levers

Effort: S = hours, M = days, L = weeks. Risk is to the non-negotiables or to correctness, not to schedule.

### 4.1 MUST (free or near-free, largest measured cost behind them)

| # | Lever | Expected speedup | Cost | Effort | Risk | First concrete step |
|---|---|---|---|---|---|---|
| M1 | **Integration-first with a stage-exit attempt now.** Freeze new-faculty de-risking; each research cycle converts one de-risked faculty (GO 6/6, off by default) to on-by-default with a reply-level LBF number; run the Stage 1 exit gate end to end in October even if it fails | Converting 10-15 of the 39 waiting faculties is worth more toward Stage 4 than any new breadth `[D]`; a failed stage exit is the first real measurement of the integrated brain in six months | Free | M per faculty | Low; the gates already require six seeds and lesion evidence | Pick the three highest-LBF off-by-default faculties from the 69-row ledger; write one packet each; schedule the Stage 1 exit run on the pool |
| M2 | **Reply-level LBF only, host share beside it, verify-go the instrument at ratification** | Ends the decision-field inflation (0.843 vs 0.535); avoids another 10-day window of decisions on a wrong number `[W][V]` | Free | S | None | Land the reply-level re-score packet (transition plan section 2 item 6) first; add one line to `verify-go`: "a new headline metric is a positive result; refute its instrument before it is reported" |
| M3a | **One shared reset primitive + CI invariant** for every pool class (`_hard_reset()`/`_reset_dynamics()`) | Removes the class behind 9 of 21 retractions and a 43-runner audit; 26 runner files still hand-roll this reset today, `sim/` shares none `[F][D][V]` | Free | M | Low; CPU-only test | A CPU test that enumerates pool subclasses and asserts no state array differs after reset() then a no-op step |
| M3b | **Fairness pre-check gate + production-substrate smoke** before any wall/NO-GO is filed | Removes the 1-10 day "wall that was a knob" class and the toy-to-production non-transfer that cost gap#4 "weeks" `[W]` | Free | S-M | Low | Extend `gates/verdict_preconditions` (or a new `gates/matched_control`): a NO-GO finding must state equal training data, per-arm learning rate, equal seed budget, and one production-substrate run |
| M4 | **Pivot dwell and retirement list**: no phase-level redirect for four weeks unless a decisive six-seed falsification forces it; every pivot commit lists the branches and plans it retires or banks | **(estimate)** reclaims a large share of the ~30 of the last 49 days spent re-orienting; bounds the 22% never-merged branches `[D][G]` | Free | S | Low, if the falsification exception is explicit | One paragraph in CLAUDE.md's pivot convention (owner's edit) and a `docs/plans/` template field "Retires:" |
| M5 | **Compute hygiene sweep**: fix or formally retire `pool40`; auto-requeue rc 143/137/3 once; a reboot-only (never training-restart) systemd unit for the 3090; CPU-first default (already in the transition plan) | Up to +50% CPU-pool throughput; recovers ~19% of GPU attempts (80 of 635) whose compute was spent then discarded; ends unattended >20h dead windows after a crash `[C]` | Free | S each | The reboot unit must be hardware-only to respect the owner's no-session-watchdog preference (transition plan section 8) | Reboot/diagnose `pool40`; a 10-line change in `tools/gpu_queue.sh`; a unit that only power-cycles on the documented "off the bus" signature |
| M6 | **Archive the board's 70 historical snapshots** the way CLAUDE.md was archived (RAG-indexed), leaving the CURRENT STATE section | Cuts the mandatory per-session read by ~90% (8,406 lines to a few hundred) `[P]` | Free | S | None; archive, do not delete | One `sync-documentation` cycle moving snapshots to `docs/GAP_CLOSURE_MISSION_ARCHIVE.md` |
| M7 | **A cheap cloud LLM profile for the mechanical tier** before 2026-10-06, via the transition plan's `llm use <profile>` (already LLM-agnostic, section 9) | **(estimate)** 5-20x more mechanical-task throughput per dollar than Claude quota `[X]`; keeps packets moving when the local server is stopped for GPU windows or gaming | $0-18/mo at low volume (section 5) | S | Verdicts from any non-Claude driver stay PROVISIONAL per the transition plan's verdict rule | Add one profile file (base URL, token variable, model name) for DeepSeek V4.1 Flash or the GLM Lite plan; run the three bake-off tasks |
| M8 | **Dormancy trip-wire as an opt-in owner reminder** (calendar or a scheduled nudge), not a repo daemon | Recovers up to the size of the one 125-day gap per occurrence `[G][P]` | Free | S | Must stay opt-in to match the manual-continuation preference | A calendar entry "N days since last commit?" |

### 4.2 SHOULD (near-free, clear evidence, smaller or slower payoff)

| # | Lever | Expected speedup | Cost | Effort | Risk | First step |
|---|---|---|---|---|---|---|
| S1 | Pre-merge replay for any new gate (TP > 0, FP = 0 over recent history) using `tools/prereg_gates_replay.py` / `tools/ssh_stdin_gate_replay.py`; backfill tests for the 38 untested gates (local-model tier); simplify or retire low-yield gates (a 951-line gate for ~1.4% of commits) | Removes 2-3 review rounds per gate; prevents months-long silent gate errors `[P]` | Free | M | Low | Add "replay before merge" to the gate-authoring checklist; queue test backfill as packets |
| S2 | AWS burst as the default dispatch path for any six-seed CPU battery with serial ETA > 30-60 min, inside the $50/day cap | 10-30x wall time on qualifying batteries, already demonstrated `[C]` | A few dollars per battery | S | Spend-ledger undercount bug must be fixed first `[C]` | A rule in the pool dispatcher: ETA > 45 min and cap headroom -> AWS |
| S3 | Free GPU lanes: Kaggle (30 GPU-h/wk, no application) and a Google TPU Research Cloud application (free, requires public sharing) | +18-25% weekly GPU hours at $0; TRC could run an independent six-seed sweep `[X]` | $0 | M to port a headless runner | Kaggle 9h session cap; TRC turnaround days-weeks | Create the Kaggle account; file the TRC application |
| S4 | CUDA Graphs on the 208-step resonate loop (97.7% of a composer operation is 3,000-4,000 tiny kernel launches) | **(estimate)** 30-50% cut on the dominant component; the 2026-06-18 plan calls it "nearly free, large win... do this first" `[X][V]` | $0 | M | Must be byte-identical when off (non-negotiable #4 carve-out) | A bounded experiment on one operation, before/after with `tools.lab.before_after` |
| S5 | `.gitattributes` enforcing LF + a pre-commit line-ending check; stop committing raw JSON (31,497 files) and keep results on the pool/server storage | Prevents a third 34M-line rewrite; shrinks every git/gate/grep pass `[G]` | Free | S | Results still need provenance; keep the `.prov.json` sidecars in git | One config file and one `.gitignore` rule |
| S6 | Terse commit convention for mechanical commits; a weekly "doc/board commit share" metric with a budget near June's 5-10% | Reclaims capacity from the 5.2% to 39.1% drift `[G]`; speeds every future audit | Free | S | None | A `git log --stat`-based check next to `tools/parallel_audit.py` |
| S7 | Weekly CPU-only FAILURE_LOG triage; mechanize the top three NOT-GATEABLE classes by cost times recurrence | Shrinks exposure windows (the C2 bug lived >=6 days); targets the 56% ungated rows `[F]` | Free | S weekly | None | A standing packet for the local model each Friday |
| S8 | Branch triage: flag any `research/*` branch with no merge decision after N days (mirror of `lane_check.py`) | Bounds the 22% abandonment backlog `[G]` | Free | S | None | A script + cron on the workstation |
| S9 | Gate six-seed battery submission on a passing same-command single-seed smoke within a recent window | Avoids burning a battery on a bad flag (the gap#4 case) `[C]` | Free | S | None | An addition to `tools/gates/` reusing existing dev-seed runs |
| S10 | At the first BORDERLINE on a continuous hyperparameter, run a replicated coarse-to-fine grid on the pool | Collapses a 29-day oscillation into one sweep `[W]` | CPU pool time | S | None | A line in `verify-go`: BORDERLINE triggers a grid, never a re-check of one point |
| S11 | Extend `gates/workflow_cost_tiering` to ad-hoc subagent-spawning scripts | Caps budget spikes right when the plan drops to Pro `[F][P]` | Free | S | None | Extend `tools/cost_audit.py::find_untiered_agents` |
| S12 | A day-14 "is this a knob?" review in every arc, plus the ceiling/reference run on day one | Bounds arcs like gap#4 (69+ days) `[W]` | Free | S | None | A checklist item in the `neural-simulator` skill |

### 4.3 COULD (real value, more effort or owner-dependent)

| # | Lever | Expected speedup | Cost | Effort | Risk | First step |
|---|---|---|---|---|---|---|
| C1 | Preprint (arXiv cs.AI/q-bio or bioRxiv, "Independent Researcher") of two finished results (the dispatch-bound profile; the LBF methodology) and applications to Foresight's AI-for-Neuro RFP (deadline 2026-10-31), Emergent Ventures, Manifund | Indirect: outside reviewers (the thing that caught the metric), possible collaborators, funding that removes the $50/day and one-3090 ceilings `[X]` | $0 | M | Honesty boundary must be explicit in any public text | Adapt one existing finding into a 4-page note; draft the Foresight application by 10-20 |
| C2 | Benchmark GeNN or Brian2CUDA against the cupy substrate on a representative subnetwork, on a rented GPU | Unknown vs already-GPU code; published 200-1000x is vs CPU, so measure, do not assume `[X]` | ~$0.10-0.25/h vast.ai 3090 | L | Could be a regression if launch count is the real bottleneck | A like-for-like port of one region with anti-cheats |
| C3 | Cache the seed-independent parts of a brain build across a battery's six seeds | Up to 5/6 of the shared setup cost on every battery `[C]` | Free | M | Must be byte-identical when off | Profile one build to size the shared fraction |
| C4 | Rent (vast.ai/RunPod) instead of buying a second 3090 now that used prices roughly doubled | A second card's worth on demand for ~$72-180/mo continuous, far less for bursts `[X]` | Pay per use | S | Spot preemption | Only when a specific battery is GPU-bound |
| C5 | Public dashboard/site generated from the faculty ledger (the two ownership numbers per week, the stage-exit checklist) | Makes the #1 metric hard to overstate in private; attracts collaborators | $0 (GitHub Pages) | M | Same honesty caveat | A JSON ledger of the 69 faculties rendered to one page |
| C6 | Move the whole `raw/` store to the server or object storage with a manifest | Repo size and git-operation time | $0 | M | Provenance manifest required | After S5 |

---

## 5. Financial options

Only prices the external miner cited; sources as given by that miner; all are 2026 web prices and may change. Rows marked **(arith.)** are the synthesizer's arithmetic on those cited prices.

| Tier | Option | What it buys | Monthly cost | Notes / source |
|---|---|---|---|---|
| **FREE** | Local Qwen3.8-27B IQ4 on the 3090 (already deployed) | Daily driver | $0 | Peaks at 22.8 GB; conflicts with brain GPU runs (transition plan section 3) `[V]` |
| FREE | Kaggle | 30 GPU-h/wk (P100 16GB or 2xT4 32GB) + 20 TPU-h/wk, 9h sessions, no application | $0 | Kaggle docs via `[X]` |
| FREE | Google Colab free tier | T4/K80 12GB, ~12h sessions, 90-min idle timeout | $0 | `[X]` |
| FREE | Google TPU Research Cloud | Free TPU quota for approved projects; rolling application; must share results publicly | $0 | sites.research.google/trc `[X]` |
| FREE | OpenRouter `:free` models | 28+ models, 50 req/day, 20 req/min | $0 (1,000 req/day after a one-time $10 credit) | `[X]` |
| FREE | arXiv / bioRxiv preprint | External review, visibility | $0 | `[X]` |
| FREE | Brian2, Brian2CUDA, GeNN | Oracle (already) and candidate accelerated backends | $0 | `[X]` |
| FREE (applications) | Foresight AI-for-Science nodes RFP: $10K-$100K + compute + hub access, deadline 2026-10-31; Emergent Ventures (2-3 week response); Manifund regrantors $5K-$50K; LTFF (<$100K, funding-constrained); NVIDIA Academic Grant up to 30,000 H100-hours; NVIDIA Inception + AWS Activate up to $100K AWS + $100K DGX credits (startup-framed, solo eligibility unconfirmed); Intel INRC Loihi 2 (form reported inactive, email instead) | Funding and compute | $0 to apply | `[X]` |
| **Near-free (< $20/mo)** | GLM Coding Plan Lite | GLM-5.3 inside Claude Code via a prompt quota | $18 flat | `[X]` |
| Near-free | DeepSeek V4.1 Flash off-peak API | $0.15/$0.60 per M in/out, cache-hit input $0.006/M | **(arith.)** ~$9-18 at 1-2M tokens/day | Per-token prices `[X]`; the miner's own estimate is $50-180/mo at 5-20M tokens/day |
| Near-free | Google Colab Pro | ~57 T4-hours (100 compute units) | $10-12 | `[X]` |
| Near-free | OpenRouter paid unlock | 1,000 free req/day | $10 once | `[X]` |
| Near-free | MiniMax M2 API | $0.255/$1.02 per M, 196K context, tool calling | **(arith.)** ~$15-30 at 1-2M tokens/day | `[X]` |
| **Modest (< $100/mo)** | GLM Coding Plan Pro | Larger quota | $80 flat | `[X]` |
| Modest | Kimi K2.6/K2.7-Code API | $0.95/$4.00 per M, cache-hit $0.16-0.30/M, 262K context; 93.4% SWE-bench Verified | Usage-based | `[X]` |
| Modest | Qwen3.8 Flash API | $0.14/$0.42 per M | Usage-based | `[X]` |
| Modest | vast.ai RTX 3090 | ~$0.10-0.25/h (spot) | ~$72-180 continuous; bursts far less | `[X]` |
| Modest | RunPod RTX 3090 / 4090 | ~$0.20/h; $0.34 (community) to $0.69/h (secure) | Usage-based | `[X]` |
| Modest | Colab Pro+ | A100/V100 priority | $50 | `[X]` |
| Modest | Claude Max 5x | 5x Pro limits | $100 | Usage meter cut ~17% net on 2026-09-14 `[X]` |
| **Larger** | GLM Coding Plan Max | Largest quota | $168 | `[X]` |
| Larger | Claude Max 20x | 20x Pro limits | $200 | `[X]` |
| Larger | AWS lane at the current cap | r7i / g6 bursts | Up to $50/day = **(arith.)** up to ~$1,500/mo if used daily; one L4 burst actually cost ~$118 `[C]` | `[C]` |
| Larger | H100 / A100 rental | H100 spot ~$2.80/h, on-demand median $3.49/h, cheapest $1.49/h; A100 80GB $0.60-3.00/h | Usage-based | `[X]` |
| Larger | Used RTX 3090 purchase | One card | $1,287-$1,411 (up from $600-800 in March 2026) | `[X]`; shelved 09-25, correctly |
| Larger | DeepSeek V4 Pro / Qwen3.8 Max / Claude Sonnet 5 APIs | $1.32/$3.96 (off-peak $0.66/$1.98); $2/$6; $2/$10 per M | Usage-based | `[X]` |

**Recommended free/near-free stack (the synthesizer's recommendation, built only on the rows above).**

1. **Driver:** the local Qwen3.8-27B for daily packets (free), with **one cloud profile** for the mechanical tier so work continues during GPU windows and gaming: DeepSeek V4.1 Flash off-peak (usage-based, roughly $9-18/mo at 1-2M tokens/day, arithmetic) or GLM Lite ($18 flat) if a fixed bill is preferred. Claude Pro stays the weekly inspector, exactly as the transition plan divides the labour. Every non-Claude verdict remains PROVISIONAL.
2. **Compute:** the CPU pool restored to three nodes (free); AWS bursts inside the existing $50/day cap for batteries with serial ETA over ~45 minutes (a few dollars each); Kaggle for GPU-tolerant sweeps ($0); a TRC application filed now ($0, and its public-sharing condition dovetails with C1).
3. **Funding:** Foresight (deadline 2026-10-31) and Emergent Ventures applications, both $0 and both plausible for a solo researcher per the miner; NVIDIA Academic Grant if eligibility permits.
4. **Hardware:** buy nothing; rent by the hour only when a battery is GPU-bound.

**(estimate)** Total recurring cost of this stack: $0-40/month above what is already paid, versus the Max 5x/20x plans at $100/$200.

---

## 6. The reworked, best-in-class form

Each block lists what to START and what to STOP; every STOP is tied to a number in section 3.

### 6.1 Workflow

**Start.**
- **Integration-first milestones.** The unit of progress is a faculty that is on by default in live chat and whose lesion changes the reply, with the reply-level LBF and host share recorded in the ledger. A finding without an integration slot is a half-unit. Stage exits are scheduled on the calendar, not on readiness: Stage 1 in October, Stage 2 in November, each run even if it is expected to fail, because a failed exit is the first measurement of the integrated brain the project has ever had.
- **Ceiling test on day one.** Every arc opens with the reference/ceiling run and one production-substrate smoke, and has a day-14 "is this a knob?" review before further findings are filed (the gap#4 lesson, `[W]`).
- **One dashboard, two numbers.** Reply-level LBF and host-decision share, six seeds, per week, per faculty, generated from a JSON ledger by a script; the same page carries the stage-exit checklist, the compute-utilization figure and the FAILURE_LOG trend. Nothing else is a headline.
- **Fewer, better gates.** A gate exists only with a replay (TP > 0, FP = 0) and a test; gates are reviewed quarterly and retired or merged when their catch count is zero; a gate budget (lines of gate per line of `sim/`) is reported.
- **Packetized work for cheaper drivers.** As the transition plan already defines: a packet is committed prereg + runner + tests + one dev verification + HANDOFF with an outcome-to-verdict table. Claude writes packets and rules; the local or cloud model executes; verdicts are provisional until an independent re-derivation or inspection.
- **Weekly cadence.** Monday: Claude inspection reads the digest, rules on NEEDS-CLAUDE, merges, writes 2-3 packets. Daily: the driver executes. Friday: the FAILURE_LOG triage sweep and the branch-triage list. Monthly: the owner reads the two numbers and the stage-exit checklist, and decides the next seam. Pivots only at the monthly review, with a retirement list, unless a six-seed falsification forces one.
- **Matched controls as a precondition.** No wall or NO-GO is filed without equal data, per-arm learning rates, equal seed budgets and one production-substrate run.

**Stop.**
- Starting a new faculty while 39 de-risked ones are off by default (finding 1).
- Scaling a lever on a toy or surrogate substrate before a production smoke (gap#4 "weeks").
- Reporting any metric at a level other than the one the user sees (0.843 vs 0.535).
- Redirecting the phase more than once a month (7 redirects in 49 days).
- Rechecking a single BORDERLINE point (29 days on satdiv).
- Running six seeds before a single smoke seed passes (the burned battery).

### 6.2 Infrastructure

**Start.**
- **The harness on the server** (`sim-agent` container on 192.168.0.10, this week per the transition plan section 9), LLM-agnostic via `llm use`, with the restricted gateway to the workstation. Nothing here is new; it is the plan.
- **Explicit yield** for GPU windows (transition plan section 3) so the driver's session is never lost, plus auto-requeue for jobs killed by pause/OOM/memcap and a reboot-only hardware watchdog for the "off the bus" crash.
- **CPU-first, three nodes, then burst.** `pool40` fixed or retired; AWS by rule for long batteries; Kaggle for GPU-tolerant sweeps; a TRC application.
- **Simulator speed work under the non-negotiable #4 carve-out**: CUDA Graphs on the resonate loop (the project's own profile says 97.7% dispatch overhead), and per-battery build caching, each byte-identical when off and measured with `before_after` before any speedup is claimed.
- **Caching and storage.** Raw results on the server with a manifest; the git repo holds findings, provenance sidecars and code.

**Stop.**
- Leaving the 3090 idle 43% of the time (17.9 of 41.5 logged days), including a 90-hour dead window, with no rule that fills it overnight.
- Counting rc=143/137/3 kills as terminal failures.
- Committing 31,497 raw JSON files and rewriting 34M lines of line endings.

### 6.3 Digital: repo, docs, knowledge base, automation

**Start.**
- **Repo/doc structure.** The board becomes CURRENT STATE only (a few hundred lines) plus a RAG-indexed archive; the faculty ledger becomes a machine-readable JSON (name, on-by-default, reply-level LBF, host share, seeds, last lesion date) from which `ROADMAP.md`'s table, the dashboard and the stage-exit checklist are generated; `docs/plans/` gets a status header (active / superseded-by / retired) so 319 plans are navigable.
- **Knowledge base.** The RAG index is already the right tool; add the archive and the ledger to it, and make `before_you_build.sh` print the ledger row for any faculty named.
- **Dashboards.** One generated page; the compute figure comes from the queue logs (`gpu_queue.log` gaps, pool node reachability, job exit codes), so idle GPU time and a dead node are visible daily rather than found in an audit.
- **Automation that fails loudly.** Branch triage after N days; doc/board share per week; a dormancy nudge (opt-in); a weekly FAILURE_LOG sweep; the cost-tiering check extended to ad-hoc scripts.
- **Commit hygiene.** Terse messages for mechanical commits; narrative reserved for GO verdicts and rulings; the narrative goes in the finding, not the commit.

**Stop.**
- Appending state snapshots to a mandatory-read file.
- Maintaining a waiver meta-process to bypass gates; fix lane starvation (three nodes, AWS by rule) instead.

### 6.4 Presentation: public site, preprints, community, demo, collaborators and funding

**Start.**
- **A public site** generated from the same ledger and dashboard: what the brain does, the two ownership numbers, the stage-exit checklist, the honesty boundary stated plainly (functional read-outs only, no phenomenal claims), the retraction register in the open. Public numbers are harder to overstate than private ones; this is the structural version of the 09-25 lesson.
- **Two preprints** from finished work: the dispatch-bound profiling result (2026-06-17) and the load-bearing-fraction methodology with its reply-level correction. Each is a few days of adaptation of existing findings and costs nothing.
- **A demo** that shows the thesis: a live chat with a lesion toggle per faculty, where the reply visibly changes when a spiking faculty is switched off. That is the LBF made tangible, and the single most persuasive artifact for a collaborator or a reviewer.
- **Community.** A "good first probe" list from the packet backlog; the six-seed and verify-go rules as the contribution standard; outside reviews scheduled monthly rather than once.
- **Funding.** Foresight by 2026-10-31, Emergent Ventures, TRC; NVIDIA Academic Grant if eligible. Frame the ask around the falsifiable bet and the measurement discipline, which are the project's genuine strengths.

**Stop.**
- Working with no outside reader for months; the only two outside reviews to date changed the phase.

---

## 7. A 30/60/90-day plan

Compatible with `docs/plans/2026-09-29-local-model-transition-PLAN.md` (timeline section 7: packets 09-29/30, guardrails and yield 10-01/02, dry run 10-03, buffer 10-04/05; Pro from ~10-06) and with Phase A of the research plan (host-share baselines, selection seam, long-delay credit, sleep transitive inference; exit check ~Oct 20).

### Days 0-7 (Sep 29 to Oct 5): the transition week, plus the cheapest fixes

| Day | Transition plan (unchanged) | Added from this report | Lever |
|---|---|---|---|
| 09-29/30 | Packets for the running builds; deadlock fix; reply-level re-score; B2b scoring | Reply-level re-score lands first and becomes the only reported LBF; the decision-field number is retired from headlines | M2 |
| 10-01/02 | Guardrails; `llm yield`; GPU wrapper; inspection digest; runbook v3 with a two-week queue | Add the cloud profile to `llm use` (DeepSeek Flash or GLM Lite) and run the three bake-off tasks on it; fix or retire `pool40`; auto-requeue rc 143/137/3; `.gitattributes` LF | M7, M5, S5 |
| 10-03 | Dry run: the local model drives a full day | Include one integration packet in the queue (first faculty to on-by-default with a reply-level LBF) so the dry run exercises the new unit of progress | M1 |
| 10-04/05 | Buffer; packets for the next research steps | Archive the board's snapshots; write the C2 reset-contract test as a packet; add the fairness pre-check to `verify-go`; create the Kaggle account; file the TRC application; owner sets the dormancy reminder and rules on the reboot-only unit | M6, M3a, M3b, S3, M8, M5 |

### Days 8-30 (Oct 6 to Oct 29): Phase A on Pro, integration slot open

- Weekly Claude inspections (Mondays) per the operating model; each writes one integration packet alongside the research packets.
- Phase A steps 1-2 (host-decision map, host-share baselines) and steps 3-5 as planned; every number at both levels.
- Three de-risked faculties converted to on-by-default with reply-level LBF (M1); the **Stage 1 exit gate run end to end by Oct 24**, expected-to-fail allowed; the result is a finding either way.
- The C2 invariant test and the fairness gate merged after a pre-merge replay (M3, S1).
- Pivot dwell in force: the Phase A exit check on ~Oct 20 is the one scheduled decision point (M4).
- AWS-by-rule for long batteries (S2); the weekly Friday triage sweep starts (S7); doc/board share metric live (S6).
- Foresight application submitted by 10-31 (C1); a preprint draft of the profiling result started.

### Days 31-60 (Oct 30 to Nov 28): Phase B kickoff, second integration wave, speed experiment

- Phase B: content selection into the brain (roadmap section 3), with the host-decision share reported weekly.
- Faculties 4-8 to on-by-default; **Stage 2 exit attempt by Nov 21**.
- Gate test backfill as local-model packets (S1); branch triage script running (S8); cost-tiering extended (S11).
- The bounded CUDA Graphs experiment on the resonate loop, before/after measured, byte-identical when off (S4).
- Preprint 1 submitted; the demo (chat + lesion toggle) prototyped from the existing webapp.

### Days 61-90 (Nov 29 to Dec 28): consolidation into production, public face

- Faculties 9-15 to on-by-default; **Stage 3 exit attempt by Dec 19**; the plan is for the consolidation mechanism
  (closed 08-06, ported 08-25) to become one of the faculties wired into production by that attempt -- not yet done.
- The GeNN/Brian2CUDA head-to-head on a rented GPU if S4 left speed as the bottleneck (C2).
- Public site and dashboard live from the ledger (C5); second outside review scheduled; preprint 2 drafted.
- Owner monthly review with the two numbers and the stage-exit checklist; the first pivot allowed under the dwell rule.

**(estimate)** Success criterion for the 90 days: 15 of the 39 off-by-default faculties integrated with reply-level LBF numbers, three stage-exit attempts recorded (pass or fail), the monthly rework share below 30%, the GPU idle share below 30%, and no retraction from the C2 class.

---

## 8. Risks, and what not to change

**Do not change (the non-negotiables, verbatim intent).**
1. **Brain-based only.** No lever here adds host computation between sensation and action. Integration-first means wiring spiking faculties on, not replacing them with scaffolds to pass a stage.
2. **One brain, one spiking substrate.** Faculties interact through synapses; the integration slot wires into `sim/`, not into co-located modules.
3. **No defer.** A failed stage exit is a finding and a new method, never a reason to shelve the capability; the pivot dwell rule limits *redirects*, not the obligation to keep going.
4. **Faithfulness-first, but falsifiable.** Speed work (CUDA Graphs, build caching) is allowed only under the owner's carve-out: significant gain, nothing important lost, byte-identical when off, documented honestly; six-seed ablation still names the necessary mechanisms.
5. **The honesty boundary.** Every public artifact in section 6.4 states that self-reports are functional read-outs and that phenomenal experience is never claimed.
6. **Six seeds** before any generalization claim, including for the "faster" claims in this report.
7. **The gates are authoritative.** Fewer, better gates means replayed and tested gates, not bypassed ones; the waiver system goes away by removing lane starvation, not by removing gates.

**Risks of the plan itself.**

| Risk | Mitigation |
|---|---|
| "Faster" drifts into shortcuts (a host argmax to pass a stage) | The host-decision share is reported beside LBF; a stage exit with rising host share is a failure |
| Integration-first is used to skip the six-seed bar | The ledger row for a faculty requires six seeds and the lesion evidence; the gates already enforce this |
| A cheap cloud driver produces confident wrong verdicts | The transition plan's verdict rule: PROVISIONAL until re-derived or inspected; guard hooks apply to any backend |
| Pivot dwell blocks a genuine falsification | The exception is explicit: a decisive six-seed falsification may redirect at any time, with a retirement list |
| A reboot watchdog conflicts with the manual-continuation preference | Hardware-only, reboot-only, never restarts a session; the owner decides (transition plan section 8 already allows auto-resume only after a driver-initiated yield) |
| Archiving the board loses context | Archive and RAG-index, never delete; the CURRENT STATE section stays mandatory |
| Public exposure invites overclaiming | The retraction register is public too; the honesty boundary is in the site's header |
| Claude's lapses recur under a smaller budget | The lapses in 3.5 map to checks (cost-tiering extension, replay-before-merge, doc-share metric, matched-control gate); the weekly inspection reads the digest, not history |
| The dormancy recurs | Only the owner can prevent it; the reminder is opt-in and the harness keeps the queue stocked for two weeks at a time |

**What this report cannot know.** Why the 125-day gap happened; how many owner-hours any month actually took; whether the 2026-06-17 dispatch profile still describes today's hot loop (re-profile before S4); the exact token cost of a pivot. Every such gap is marked as an estimate above.

---

## Appendix: evidence index

- Git statistics: `git log --oneline | wc -l` (11,521 as of HEAD `3e1682192`); `--merges` (632); `git log --since=2025-12-02 --until=2026-04-05` (0); per-path counts; daily/weekly histograms; `git branch --list 'research/*' --no-merged main` (127/576) `[G][V]`. `.git` is 25 GB (`du -sh .git`) `[V]`.
- Named commits: `d463f9046`, `89ef28fa7`, `e2e6a499f`, `2f8840ce0`, `c1e57875d`, `73d1a2cbc`, `41d4604b4`, `e9a6d1e29`, `00a6b5a6e`, `0e22e97f1`, `212d3ae98`, `ad6ec8577`, `fd314254d`, `f3045a828` (all confirmed by `git log -1` today) `[V]`.
- Failure and retraction records: `research/FAILURE_LOG.md` (319 lines; 223 dated rows, re-verified; 124 NOT-GATEABLE), `docs/RETRACTED.md` (21 rows; 9 dated 2026-09-02, all C2-class, re-verified), `docs/FAILURE_GATE_MATRIX.md`, `research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md` `[F][V]`.
- Walls: `research/findings/2026-08-11-gap4-ALLIN-ARC-SUMMARY-*.md`, the 2026-07-31 synthesis finding, `research/findings/raw/gap5_density/AGG_clamp_budget.json`, `research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-*.md` `[W][C]`.
- Compute: `research/queue/gpu_queue.log` (1.4 GB, 14.7M lines; busy/idle and largest-gap figures re-derived today from its START/DONE timestamps), `research/queue/pool_sync.log` (2,389 cycles; `pool40` UNREACHABLE 2,390), `research/queue/aws_idle_stop.log`, `docs/GPU_CRASH_RECOVERY.md` line 65, `GAP_CLOSURE_MISSION.md` lines 536-651, 1815-1908, 2546, 6100, `tools/local_llm/results/summary.md` and worktree `.claude/worktrees/serving-bakeoff` (local serving bake-off rows) `[C][V]`.
- Process: `tools/gates/` (48), `tests/test_gate_*.py` (10), `research/coordination/prereg_gates_replay_2026-09-25.json`, `research/coordination/sr_gate_replay_2026-09-25.json`, `.claude/worktrees` (722) `[P][V]`.
- Directives: `~/.claude/projects/-home-dant123-Projects-sim/memory/` (48 feedback files), `docs/CLAUDE_RATIONALE_ARCHIVE.md`, `docs/plans/2026-09-19-roadmap-with-a-permanent-llm-mouth.md`, `docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md`, `docs/plans/2026-09-29-roadmap-to-the-primary-goal.md`, `docs/plans/2026-09-29-local-model-transition-PLAN.md` `[D][V]`.
- External: `docs/2026-09-06-local-agent-stack-review.md`, `docs/plans/2026-06-18-hardware-acceleration-neuromorphic-roadmap.md` lines 30 and 99-100, and the web sources listed per price in section 5 `[X][V]`.

---

## SUMMARY (10 lines)

1. "Ten months" is really 125 days of zero commits (2025-12-02 to 2026-04-05; 127 days / 42% counting the two book-end commit days either side) + ~178 active days at ~65 commits/day (11,521 commits total); the repo records no cause for the gap; the problem is conversion, not effort.
2. Output went to findings, not the brain: 62.6% of commits touch research/findings, 2.7% touch sim/; ~750-820 GO findings, yet 39/69 faculties are off by default and no stage exit was ever attempted.
3. Rework is 41-64% of every month's commits by full-message grep, 16-29% by subject-line only (both re-verified); the C2 `_hard_reset()` bug alone flipped 9 GO 6/6 verdicts in one day (2026-09-02) and forced a 43-runner audit; 26 runner files still hand-roll their own reset today, with no shared primitive in `sim/`.
4. The #1 metric was overstated (0.843 decision-field vs 0.535 reply-level), ratified 09-19 without auditing its instrument, and caught 10 days later by two outside AI reviews.
5. Pivots ran ~1 per 7 days over the last 49 days (vs ~1 per 59 before); the own-voice arc (68 findings) was retired 2.5 weeks after "pursue fully"; 127/576 research branches never merged.
6. Walls were mostly instruments (gap#5 clamp = 97% of "learning"; gap#4 LIF wall was a shared learning rate; satdiv flipped on a 0.1 scale change); no gate enforces a matched control.
7. Compute ran hotter than it looked but still leaked days: GPU-queue log was busy 57% / idle 43% of the last 41.5 days, with one 90-hour dead gap (09-25 18:22 to 09-29 12:25); 29.4% of GPU jobs died non-zero; pool40 unreachable in 100% of 2,389 sync cycles; 14.5 days UNDER-PARALLELIZED behind a waiver loophole. The local-serving bake-off shows one profile (llama-server default) passing 3/3 and two alternates (ollama, unsloth-studio) failing, so the 10-06 handoff has a working driver, just not yet a choice of servers.
8. Process outgrew product: 12,555 lines of gates/hooks vs 32,964 of sim/, 10/48 gates tested, an 8,406-line board read every session, ~117-word commit messages; Claude's own lapses (same-day rule violations, Opus budget burn, overclaims) are documented. The repo itself is heavy: `.git` is 25 GB, including 17 GB of raw JSON findings.
9. MUST levers (all free): integration-first with a Stage 1 exit attempt in October; reply-level LBF only with host share beside it; shared reset primitive + fairness gate; 4-week pivot dwell with retirement lists; pool40/auto-requeue/reboot-only watchdog; archive the board; a $0-18/mo cloud LLM profile before the 10-06 Pro downgrade; opt-in dormancy reminder.
10. Free/near-free stack: local Qwen + DeepSeek Flash or GLM Lite ($0-18/mo), Kaggle + TRC + AWS-by-rule inside $50/day, Foresight (deadline 2026-10-31) and Emergent Ventures applications, rent GPUs by the hour, buy nothing; 90-day target: 15 faculties integrated, three stage-exit attempts recorded, rework < 30%, GPU idle < 30%, zero C2-class retractions.

---

## Corrections applied (merge of SYNTH-FULL.md and REPORT.md, 2026-09-29)

This FINAL.md merges the original seven-miner synthesis (SYNTH-FULL.md) with a later verification pass (REPORT.md), re-checking every disagreement directly against git and the cited files before deciding which figure to keep. Changes from the original synthesis:

1. **The dormancy gap.** Zero commits between 2025-12-02 and 2026-04-05 is confirmed at exactly 125 days (`git log --since=2025-12-02 --until=2026-04-05` returns 0, re-run today). Reframed throughout as "127 days / 42% of the span" when counting the two single-commit book-end days (2025-12-01 `17dd8c47f`, 2026-04-06 `cdf1f8b85`) either side of the silence, matching the verification pass's framing; both the 125-day (strict) and 127-day (book-end-inclusive) figures are now stated together wherever the gap is discussed, so neither reading is asserted as the only one. Added: the repo records no cause for the gap.
2. **GPU queue busy vs idle — a real error, not a rounding difference.** The original synthesis stated the GPU was "idle" 57% of the 41.5-day queue-log window. Direct re-derivation from `research/queue/gpu_queue.log`'s START/DONE timestamps today confirms the reverse: **busy** 23.65 days (57%), **idle** 17.9 days (43%). Corrected everywhere this appeared (executive summary finding 6, the root-cause ranking table, section 3.4 C2, section 6.2, the SUMMARY).
3. **The largest GPU idle gap.** The original synthesis cited 43.9 hours (2026-09-26 13:57 to 09-28 09:50); that interval does not appear among the log's actual gaps. Direct timestamp-diff today confirms the true largest gap is 90.1 hours, 2026-09-25 18:22 to 2026-09-29 12:25 (spanning the 3090 crash). Corrected wherever cited.
4. **Rework share.** Kept the original's full-message-grep figure (41-64% by month) but added the verification pass's more conservative subject-line-only figure (16-29% by month), re-derived today with `git log --pretty='%ad|%s'` and a fix/correct/retract/void/revert match on the subject only. Both are now labeled by method rather than presented as a single number.
5. **The missing shared reset primitive.** Added the verification pass's finding that 26 runner files in `research/runners/` still define their own `_hard_reset`/`_reset_dynamics`/`hard_reset` today (re-confirmed by `grep -lE 'def (_hard_reset|_reset_dynamics|hard_reset)' research/runners/*.py`), and that `sim/` shares none, wherever the C2 bug is discussed. Corrected "10 of 21 retractions" to "9 of 21" (all nine are dated 2026-09-02 in `docs/RETRACTED.md`, re-counted today; no tenth C2-class row exists).
6. **Direction-change acceleration.** Restated explicitly as "~1 per 59 days" (first 236 active days) versus "~1 per 7 days" (last 49 days), alongside the existing "~8x" framing, per the verification pass.
7. **Repo weight.** Added `.git` size (25 GB, `du -sh .git`, re-verified today) alongside the existing 17 GB / 31,497-file raw-JSON figure.
8. **Branch count.** Corrected 127 of 575 unmerged `research/*` branches to 127 of 576 (re-counted today: `git branch --list 'research/*' | wc -l` = 576).
9. **The local-serving bake-off.** The verification pass characterized the local driver as having "not passed its own bakeoff" ahead of the 2026-10-06 Pro-plan handoff, citing the ollama and unsloth-studio failures. Checked directly against `tools/local_llm/results/summary.md` and the 2026-09-29 17:57 rows of `research/queue/gpu_queue.log`: ollama failed all three tasks and unsloth-studio returned HTTP 400, but the **default llama-server profile (`qwen38-27b-iq4nl-mtp-128k-q4`) passed all three (T1/T2/T3)**. Added as new finding C4a: the handoff rests on one proven profile, not zero; only the easier-to-operate alternative servers remain unproven.
10. **Stale counts refreshed against today's HEAD (`3e1682192`).** Total commits 11,467/11,519 → 11,521; merges 631 → 632; September commits 2,156 → 2,158; FAILURE_LOG dated rows 222 → 223 (319 total lines); one-author commit count 11,514/11,519 → 11,516/11,521. These are incremental drift from the repo continuing to receive commits while the miners and the verification pass ran, not disagreements of substance.
11. **Everything else** (the ~3,000-findings/30-faculty/no-stage-exit picture, the #1-metric overstatement, the wall-was-a-knob findings, the process-overhead findings, the levers, the 30/60/90-day plan, and the financial-options table) is carried over from the original synthesis unchanged; the verification pass did not dispute these and this pass found no repo evidence against them.