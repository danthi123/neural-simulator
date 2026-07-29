---
name: verify-go
description: Before COMMITTING or REPORTING a positive result — a GO, a "surpass", a milestone, an "it works / it's inert / byte-identical / tests pass" — run independent ADVERSARIAL skeptics that each try to REFUTE it from a distinct angle, then commit only what survives (commit-with-caveat, never silently drop, if a mischaracterization is found). Use whenever a result would change a wall/gap status, flip a default, or land in a findings doc / the board. This operationalizes drift-mode #8 (believing a surpass without verification) + the SILENT-FAILURE CLASS in the neural-simulator skill, which alone are a checklist — this skill makes running them a reflex before the commit.
---

# Verify-GO — adversarially verify a positive result before it lands

A GO that looks clean is the most dangerous thing in this project: it gets committed, flips a default, and mis-aims the next session. The 2026-07-24 session produced **four** over-claims caught only by adversarial verification (W3 "structural immunity", P1.2 affect-scalar, 2 gap#5 lucky-draws) and **six** silent-failure retractions — **three self-authored**. Vigilance doesn't catch these (I authored a bare-`except` swallowing my own warning the day I documented that pattern five times). A *procedure* does.

**Announce at start:** "Running verify-go: adversarially probing this result before it lands."

## When to run
- Before COMMITTING a finding that reports a GO / surpass / milestone / capability close.
- Before FLIPPING a default or claiming a cheat/shortcut is closed.
- Before writing "inert / byte-identical / no-op / tests pass / pushed / on GPU" as fact.
- **Before recording even an INTERIM "lead" / "partial positive" / "realizable" / "first non-flat result" into a finding or the board** (2026-07-25: a "lead" I wrote into TWO docs — "the two-sided read is REALIZABLE, fact-1 own/other 3.67" — was a **winner-slot artifact**; the disambiguating control, run only later by an adversarial subagent, refuted it. A non-flat number is not a lead until its control collapses; an interim positive mis-aims the next step exactly like a GO does).
- NOT for routine mechanical edits (a typo, a refactor with a passing test you watched pass).

## The procedure — dispatch independent skeptics, one per lens
Spawn N skeptics (a `Workflow` parallel stage, or concurrent `Agent`s) — each gets the result + the runner/finding and is told: **"Try to REFUTE this from the <lens> angle. Default to REFUTED if uncertain."** Independence matters: redundant identical skeptics miss failure modes diverse lenses catch. Assign distinct lenses:

1. **Reproducibility / power** — does it hold at 6 seeds (42/43/44/100/101/102)? Is the effect bigger than seed-to-seed noise? A 3-seed indicator is not a GO. Is one lucky seed carrying it?
2. **Gate-cheat** — can the gate PASS without its key control? Is the anti-cheat control DEFAULT-ON and actually INVOKED (grep the call site), not just defined? A gate that passes without its control IS the bug.
3. **Control-integrity** — is the A/B lever ONE variable? (One flag ≠ one variable — a global clamp touched both a spiking synapse and a host readout.) Does the DEFAULT arm genuinely differ from the treatment (print the lever's effect; if both arms froze, the verdict is void while looking plausible)? **AND: are BOTH arms pinned at the FLOOR or the CEILING? A comparison between two saturated arms has ZERO discriminating power and cannot detect a difference that exists — the resulting "no difference" is VOID, not evidence.** Always print the raw magnitudes of both arms and confirm they are in a responsive range before believing any null. (2026-07-25, TWICE in one session, opposite ends: a learning rate above the saturation knee drove every synapse into the soft bound and pinned a real selectivity signal at a flat ~1.0 — the signal appeared as soon as the rate came down; and a "did this regress?" A/B was run at a training budget where BOTH old and new code sat at 0/16, producing a confident "nothing regressed" that did not follow from the test at all, and had to be retracted once the un-floored budget showed 12.5% vs a recorded 87.5%.)
4. **Instrument-trust** — read the runner's OWN verdict line; NEVER lift a metric from a run that printed `SIGNAL=False` / `HONEST NEGATIVE`. Is the metric quantized/rounded so the effect is unfalsifiable by construction (differencing `round(x,4)` values)? A refutation needs the instrument verified exactly as much as a confirmation.
5. **Seeding** — is `cfg.seed` set (NOT `actual_seed_used`, which seeds nothing)? Build twice at one seed and hash `cp_neuron_firing_thresholds` — identical ⇒ actually seeded.
6. **Infra** — pushed? verify with `tools/push_both.sh` (ls-remote, not `echo pushed`). On GPU? `tools/monitor_runs.py` reports `[GPU]`/`ON CPU`. "byte-identical when off"? that's an ASSERTION (a test), not a comment.
7. **Selectivity-metric bias — the MASS artifact (three instances in one session, 2026-07-25; treat as the default suspect for ANY ratio).** For ANY "own-vs-other" / ratio / argmax-selectivity metric (own/other, recall picks the right slot, this cue selects that assembly, this window is more specific), the null hypothesis is **"one side simply has more MASS"** — more weight, more spikes, more cells. Three checks, all mandatory, all cheap:
   - **(a) Permuted-target / shuffled-label / random-source control** — recompute weighting by a RANDOM item's core instead of the true one. It MUST collapse to ~1.0. (Failed 2026-07-25: per-slot weight `[24, 80, 24]`, one slot 3.4× heavier ⇒ own/other 3.67 for whichever fact mapped to it, permuted control 3.37 = unearned.)
   - **(b) Magnitude-free form** — report the ratio's normalised twin (cosine / L2-normalised / rate-per-step) ALONGSIDE it, plus the RAW per-item masses. (Failed 2026-07-25: during-write ungated ceiling `[1.93, 0.88, 0.80]` looked like a strong positional effect; total spikes were `[1203, 701, 625]` — the first window fires 2× — and cosine specificity was FLAT `[1.16, 1.10, 1.11]`. A whole "cumulative degradation across the schedule" conclusion had to be retracted.) **Beware especially a threshold set as a fraction of a GLOBAL max: it silently favours the highest-mass item.**
   - **(c) Never let a MEAN stand in for a per-item requirement** — report per-item passes with a degenerate guard (a 1-4-element set makes any ratio a small-number artifact). (Failed 2026-07-25: a mean-over-facts verdict printed "GO 3/3 seeds" when only 1 of 3 facts passed, on 3-6-cell gates.)
   A ratio reported without (a), (b), and (c) is not evidence.

Not every lens applies to every result — pick the ones with teeth for this claim, but a GO that flips a default earns at least reproducibility + gate-cheat + control-integrity.

## Synthesize + act
- **Survives all lenses** → commit as a GO. Say which lenses were run (the verification is part of the deliverable).
- **A lens finds a mischaracterization** → do NOT silently drop or quietly downgrade. Commit-WITH-CAVEAT: state precisely what the result IS vs what it was claimed to be (W3 was a real GO but not "structurally immune"; the affect wiring was real but not the scalar claimed). An honest narrowed result is a first-class deliverable.
- **A lens refutes it** → retract the GO, bank the method as a NO-GO with the root cause (per THE LAW the capability stays open), take the next mechanism.

## Verifying a NEGATIVE — before you accept a BOUNDARY (2026-07-25: the costliest error of the session)

A wrong GO wastes a build. **A wrong NO-GO closes a capability that was never actually blocked** — and it is far harder
to notice, because a negative *feels* like rigour. Before recording any "wall / boundary / characterized limit / the
substrate can't", run these:

1. **IS THE SUBSTRATE PHYSICALLY VALID? Read its state variables in their PHYSICAL UNITS and check them against
   physiological range.** Membrane potentials belong in ≈ −90…+50 mV; conductances, currents and rates have known
   scales. **No metric can catch this — only looking at the numbers in units can.** (2026-07-25: an entire multi-hour
   "dense CA1 code ⇒ no write can localize" boundary — ~15 write variants, a sparsification battery, a two-sided read,
   two subagent builds, several confident findings — was a **333× miscalibration of a pA→mV constant** in ONE config
   line. `v_apical` was parked at ~2×10⁵ mV, driving every soma; the "dense code" was runaway current, not a hippocampal
   code. At a valid operating point the real code is sparse and near-disjoint and the write localizes, 6-seed.)
2. **Check every constant that was TUNED while the artifact was present.** They were fitted to broken dynamics and will
   silently re-break the fixed system. (Same session: a cell phenotype adopted to fight the artifact was undrivable once
   voltages were physiological; a learning rate fitted to ~100× inflated activity saturated the write and pinned the
   metric at exactly the null being "confirmed".)
3. **Is the measurement itself inert?** A measurement must never be plastic. If learning is enabled while you *read*,
   you are measuring the reader. (Same session: core sizes varied with the **write** learning rate although cores are
   defined pre-write — because the rule was still learning during the read.)
4. **Does the null survive at more than one operating point?** A null measured at a single point is a property of that
   point. Sweep the parameter the mechanism is most sensitive to (here: the learning rate through its saturation knee)
   and confirm the null is flat across it — a monotonic trend toward signal means you measured a bad operating point,
   not a bound.
5. **Do the positive controls fire?** If your harness cannot demonstrate the effect where it MUST exist, the harness is
   what you have measured.
6. **CITING A LINE IS NOT VERIFYING IT EXECUTES. Find the guarding flag and check its DEFAULT against the config in
   use.** A `grep` hit proves code EXISTS; it proves nothing about whether it RUNS — and a `file:line` citation makes an
   unverified claim look verified, which is worse than no citation. (2026-07-25: I reported a mechanism as *"the whole
   story"*, cited to `sim/bridge.py:838`. That line sits inside `_apply_branchless_hebbian`, *"opt-in via
   cfg.enable_branchless_plasticity"*, default **False** — **it never ran**. The rule actually executing was a different
   one with different dynamics, and an entire "characterized ceiling ⇒ the next step is structural" verdict was built on
   the dead branch.) **Corollary — enumerate EVERY rule acting on the pathway, not just the one you are tuning:** in the
   same arc `enable_stdp` defaults to **True** and the pathway was `plastic=True`, so STDP was writing it throughout and
   never entered a 7-hypothesis ledger. **Corollary 2 — a constant that appears in a measurement may be a FALLBACK:**
   `_hw_max = cfg.hebbian_max_weight if cfg.enable_hebbian_learning else 5.0` produced a "hard 5.0 ceiling" that was read
   as a result, when it was a literal reached because that arm disabled Hebbian — so the mechanism under test was never
   exercised at all.
7. **REPRODUCE BY CALLING THE ORIGINAL CODE PATH — never by re-implementing it. Then vary ONE documented parameter.**
   If a recorded result came from `runner.foo()`, call `runner.foo()`. A hand-rolled reconstruction silently omits
   steps you did not know were load-bearing, and the divergence surfaces hours later — after conclusions have been
   built on it. (2026-07-25: **FOUR withdrawn conclusions on one thread**, every one traceable to this. My
   reproduction of a recorded 87.5% omitted `apply_concept_topographic_bias` — a pre-training cortical-somatotopy step
   applied *before* any learning — and used a different `word_to_idx` ordering. On that basis I reported "the reference
   harness fails", escalated it to "a shared-code path is broken, past results may be invalid", then had to withdraw
   the lot. I also measured at 200 events when the record was 800, and with a 1-word probe when the effect lives at 16
   words.) **Corollary: when the original path takes a parameter, override THAT parameter surgically — do not fork the
   function.** A one-line monkeypatch of a documented knob keeps every other step honest.
8. **DID IT EVER WORK — and at WHAT SETTINGS? Search the record for a known-good configuration BEFORE debugging.**
   `rag_search` / `grep` the findings for the harness's own past PASSING numbers and the config that produced them, and
   compare against the defaults you are running. A default is often a FAST setting, not the validated one. (2026-07-25:
   I debugged an A1 harness through plasticity bounds, rule variants and homeostasis because its sanity check sat at
   chance — then found the project's own record: *"direct binding at **800ev** saturated training: 15/16, 13/16, 14/16
   = 87.5%"*, while the runner defaults to `--train-events 200`, **4× less**. The harness was never broken; it was
   under-trained. Four rounds of knob-debugging chased a non-bug.) **A failing sanity check is a question about the
   CONFIGURATION first and the mechanism second.**

**Trigger:** the moment you are about to write "boundary / wall / can't / characterized limit / honest negative", or a
mechanism has failed across many well-controlled variants. **Many variants failing identically is itself evidence of a
COMMON upstream cause — the shared substrate or a shared constant — not of independent confirmations.**

## Verifying a DIAGNOSIS — the prescribed fix must be able to REFUTE it (2026-07-26, earned twice in one session)

A "diagnosis" (*this mechanism causes that defect*) is a CLAIM and gets the same adversarial treatment as a GO. Twice
in one session I wrote **"DIAGNOSIS COMPLETE"**, prescribed the fix it implied, and the fix **refuted the diagnosis**:
(1) *"a single SHARED inhibitory pool causes the winner-take-all"* — built per-slot FS cross-inhibition; identical to
the shipped global pool in all three conditions. (2) *"the `hebbian_max_weight` inversion causes it"* — re-ran with the
bound above the init; the winner persisted. One of these was **independently corroborated by a read-only research
gate** — which was ALSO wrong, because it reasoned from the CODE, not the DATA.

**The rules:**
- **Always build the fix WITH its lesion arm** (the unfixed topology/config), in the SAME run. If fix ≈ lesion, the
  DIAGNOSIS is refuted — not the implementation. Without the lesion you will read "still broken" as "fix was too weak"
  and tune forever.
- **Test the fix in the condition where the defect APPEARS.** I first compared topologies with the gaps unfrozen,
  where uniform potentiation masks any topology effect — a null that meant nothing.
- **Corroboration from code-reading is not evidence about behavior.** A second reader agreeing with your mechanism
  story raises confidence in the READING, not in the CAUSE. Only a manipulation does that.
- **Prefer a MEASUREMENT that separates the candidate causes over another prescribed mechanism.** After 11 refuted
  hypotheses, what finally settled it was instrumenting the quantity itself (per-window `dw`), which showed the write
  was near-symmetric and the outcome a ~3% residual — a shape NONE of the 11 hypotheses predicted. **When two
  successive mechanisms are refuted, stop prescribing and go measure.**
- **A rate lever cannot move a FIXED POINT.** If the state settles at a soft bound (`dw ∝ (w_max − w)`), every
  learning-rate sweep is inert *by construction* — it changes how fast you reach the bound, not where it is. Before
  sweeping a rate across orders of magnitude, ask whether the observable is a fixed point; if it is, sweep the
  STRUCTURE instead. (This retroactively explains a long history of "invariant across every lever" on one pathway.)

## Parallel arms that vanish are a SILENT FAILURE, not a result (2026-07-26)

A `for … & done; wait` fan-out of GPU runs silently dropped 3-of-4 and then 3-of-3 arms (VRAM contention), exiting
**0** with empty output. The surviving arm looked like a clean single data point. **Rules:** echo each arm's
`${PIPESTATUS[0]}`; treat "an arm produced no output" as a FAILED RUN to reproduce, never as a null result; and when
arms contend for one GPU, run them SERIALLY — a slower correct sweep beats a fast one whose failures are invisible.

## ⛔ THE RESEARCH GATE HAS A LOOPHOLE: a SEQUENCE of cheap tests is a BUILD EFFORT (2026-07-26)

CLAUDE.md fires the research gate before "committing ANY build / GPU / `sim/`-edit effort to *overcome* a
difficulty". **I evaded it for hours without ever deciding to** — because each individual step was a *cheap config
flag*, never "a build". I tested weighted-vs-count coincidence, the self-regen latch, the Hebbian bound, inhibition
topology, an inter-window washout, and cue magnitude — **six levers, ~4 GPU-hours, all against ONE defect** — and
the gate never subjectively "fired" because no single test felt like a commitment. When I finally dispatched the
research round, it resolved in one pass what the sequential guessing had not.

**MECHANICAL RULE (no judgment call, matching the gate's design intent):** *if you have tested **≥2 distinct levers
against the same defect** without resolving it, the gate FIRES.* Cheapness of each individual test is **not** an
exemption — the relevant quantity is cumulative effort against one difficulty, not the cost of the next step.
Write the lever count in the findings doc so the counter is visible.

**Self-check:** "am I about to try a third thing against the same failure?" → that IS the trigger. Dispatch the
read-only research round; your candidate becomes one ranked option, never the default.

**⛔ AND THE ≥2-LEVER TRIGGER IS NOT ENOUGH — THE CHEAPEST GUARD RUNS BEFORE THE *FIRST* LEVER (2026-07-26).**
When the research round finally ran, its first finding was that a **497-line research gate for the identical
defect, on the identical substrate, with a ranked 6-mechanism ladder, was 2 DAYS OLD** — and its predecessor had
already written the failure down verbatim. A day of lever-chasing re-derived it. It also showed my "5th instance"
framing was wrong (**8th in the project, a documented FAMILY** ⇒ gate condition (b) fires on FIRST occurrence) and
that 3 of my "new" measurements were 4th independent confirmations of results already in the corpus.

**MANDATORY, ~30 SECONDS, BEFORE THE FIRST LEVER AGAINST ANY DEFECT:**
```
.venv-rag/bin/python tools/rag/rag_search.py "<the defect in one line>" 5 --corpus finding
```
Ask literally: *has this already been scoped, tried, or refuted?* Then READ any research-gate / scope doc it
surfaces before touching a flag. **Cost: one query. Benefit here: a day.** This is drift #12 (the stale/skipped
corpus) in its most expensive form — not acting on a stale summary, but never asking whether the answer already
existed.

## Measure the thing, at the time it happens (2026-07-26 — retraction #9)

Three separate ways a measurement can be *structurally incapable* of answering its own question, all seen in one
sub-arc:
1. **Placed upstream of the effect.** A per-fact weight table ran BEFORE the `coactivation_replay` it claimed to
   characterise; a "BOUNDARY LOCATED" finding was committed on numbers that could not have shown a replay effect.
   **Print/measure the quantity BEFORE and AFTER the manipulation, and report the delta** — then a misplacement is
   visible as a zero delta instead of a confident wrong table.
2. **A lesion that does not persist.** Zeroing `cp_connections.data` survived one step and regrew (plasticity was
   live). **Re-read the manipulated quantity at the moment of measurement, not when you issued it.**
3. **A metric too coarse to resolve a real lever.** An argmax decided by 400-vs-0 cannot move on a 1.7% change.
   **Always carry one CONTINUOUS quantity alongside any count/argmax metric.**

**⇒ A NULL A/B HAS THREE EXPLANATIONS — inert lever · misplaced measurement · coarse metric — and they are
indistinguishable from the summary alone.** I misread "byte-identical arms" as "inert lever" TWICE in one session,
wrong for a different reason each time. Distinguish them by measuring the lever's effect on a continuous quantity
*before* interpreting the outcome.

## What this skill MUST NOT do
- Rubber-stamp — a skeptic that "confirms" without trying to break it did nothing. The prompt must push to REFUTE.
- Verify only the happy path — test the claim against the case you'd EXPECT to break it (a run you KNOW is broken, the seed you fear).
- Treat a refutation as needing less scrutiny than a confirmation (rule 3 of the SILENT-FAILURE CLASS).
- Touch the science verdict itself dishonestly — this skill SHARPENS a claim to the truth, it never launders a weak result into a strong one.

## Why this skill exists
2026-07-24 (evolve-skills): adversarial verification was run ad-hoc (via one-off Workflow panels) and reliably caught real over-claims — but it lived only in-session, not in a skill, so it depended on remembering to do it. This encodes it as a reflex triggered by the commit itself. Pairs with the neural-simulator skill's SILENT-FAILURE CLASS (the specific checks) — this is the procedure that runs them before a GO lands.

## ASSERT THE GATE EXISTS BEFORE YOU FREEZE OR LESION IT (2026-07-29)

`_try_pgate` swallows the `KeyError` and returns `False` for a gate that does not exist. `_mean_gate_weight`
returns `0.0` for one. **Nothing checks either return value.** So freezing a NONEXISTENT gate is a silent
no-op that presents as a *perfect* freeze: drift exactly `+0.000000`.

That is not hypothetical — a gate's existence depends on config. `comp_no_pool_slot=True` (the value in
`BASE`) **drops the pool→slot pathway entirely**, so `concept_to_comp_attr` is simply absent, and every
read of it returns `nan`/`0.0` rather than raising.

**The rule:** before any freeze / lesion / weight-read on a NAMED gate, assert it exists:

```python
assert gate in bridge._plasticity_gate_indices_gpu, (
    "gate %r absent under this config -- freezing it is a SILENT NO-OP that reads as a perfect freeze" % gate)
```

**Why it matters beyond the bug:** an exact `+0.000000` freeze-drift is the signature of a real freeze AND
of a missing gate. They are indistinguishable in the log. A suspiciously perfect number is a prompt to
check the instrument, not evidence that the manipulation worked. (Found while chasing an unrelated
`nan`; the affected headline result was checked and STANDS, because its own probe sets
`comp_no_pool_slot=False` — but only checking revealed that.)

## THE ENGAGEMENT COUNTER IS THE CHEAPEST GUARD YOU HAVE — and `tools/lab.py` already implements it (2026-07-29)

**Every void arm this project has produced shares one shape: the manipulation never engaged, and the
resulting null looked like a scientific result.** Three in a single day:
- a metaplasticity toy where `theta0` was unreachable, so every trial fell to a fallback branch that did all
  the work — caught only because all six betas printed IDENTICAL maps and identical block counts;
- a substrate probe reading a gate that **did not exist** under its config — caught because it printed `nan`;
- a saturation test where the soft bound **never bound** (`sat_frac = 0.000` in every arm), so identical
  numbers across `w_max` proved nothing — caught by the engagement fraction it happened to print.

**The rule: every arm must report a number that goes to ZERO when the mechanism is inert**, and you must
look at it before reading any score. `blocks`, `sat_frac`, `n_overrides`, `dw`, `gate n_syn` — whatever the
mechanism *does*, count it.

**And USE THE HELPER — this is the actual lapse.** `tools/lab.py` exists precisely for this and was written
after the last round of void arms:

```python
from tools.lab import lever, void_if, before_after, undefined_if_empty
lever("w_max", before=unbounded_score, after=bounded_score, continuous=sat_frac)
void_if(sat_frac == 0.0, "the soft bound never engaged; w_max arms are identical by construction")
```

The saturation arm above was written WITHOUT importing it, by the same person who wrote it. A helper you do
not import is exactly as useful as a rule you do not remember — which is the whole reason `lab.py` exists.
**Import it at the top of every probe, not when you suspect trouble.**

## PARALLELIZATION IS A DISPATCHER, NOT A DECISION (2026-07-29 — the owner flagged this TWICE in one day)

**The failure was not missing information.** The heartbeat printed `UNDER-FILLED-GPU` every 15 minutes and
the 36-core pool sat at load 0.00 for hours, while substrate conclusions were being drawn from a SINGLE
seed against this project's documented 6-seed standard. The warning fired correctly and was acted on
minimally, twice, because responding to it required **inventing a job on the spot** — so the cheap response
was always "launch one more thing and move on".

**The mechanism:** `tools/lane_dispatch.sh <gpu|pool> <slots>` keeps N slots busy from a persistent queue
file (`research/queue/gpu.queue`), moving each line queue → `.running` → `.done` so state survives a
restart and nothing is double-run.

```bash
bash tools/lane_dispatch.sh gpu 7 &          # keeps 7 GPU jobs alive from the queue
cat >> research/queue/gpu.queue <<< "<one shell command per line>"
```

**THE ALARM MOVED, and that is the actual fix.** The heartbeat no longer warns on an idle lane — it warns
on **`GPU-QUEUE-LOW`** and **`DISPATCHER-DEAD`**. An idle lane with a stocked queue self-heals in seconds;
an empty queue *guarantees* future idleness. So the only standing obligation is **keep the queue stocked**,
which is the existing "build de-risks ahead of time so idle compute always has a ready job" directive made
executable instead of remembered.

**Stock it with the work that is already OWED, not with invented work.** On the day this was built the
queue filled instantly from two real debts: seeds 43/44/100/101/102 for a headline contrast that had only
seed 42, and the clean scale series whose confound had already been diagnosed and written down. If the
queue looks empty, the likelier truth is that owed replications are being skipped.

**POOL ADDENDUM (same day): the pool's under-use was NOT a dispatch problem, and a dispatcher there was
the wrong fix.** A `pool_dispatch.sh` was written, debugged through three failures (a `pgrep -fc` that
prints `0` AND exits nonzero so `|| echo 0` emitted two lines and broke the arithmetic; an `exec -a` marker
the wrapper shells also carried, over-counting 3x; a regex that silently missed a one-line function), and
then **deleted** — because the plain `xargs -P 12` per node that already worked ran a 72-cell × 6-seed
sweep (432 runs) across 36 cores in **under 60 seconds**.

**The real constraint on that lane is JOB SIZE, not dispatch.** Numpy sweeps drain the pool faster than any
queue can be stocked, so "keep the pool full" is the wrong goal — the right one is "send the pool work
worth 36 cores", i.e. batch the whole grid at 6 seeds instead of trickling 2-seed probes. The GPU
dispatcher stands because GPU arms run ~90 minutes; the same mechanism on a seconds-per-job lane is
ceremony. **Match the mechanism to the lane's job duration.**

## AN AD-HOC CHECK THAT DISAGREES WITH A VERIFIED TOOL IS WRONG UNTIL PROVEN OTHERWISE (2026-07-29)

`tools/check_docs.py` reported W2 clean. A one-line `awk 'length>800'` appeared to find five violations,
and the instinct was "the checker has a gap". It did not. The awk was wrong **three separate ways**: it
used `NR` (cumulative across files) instead of `FNR`, so its line numbers pointed into a different file;
it did not exempt table rows; and it did not track code fences. Every "violation" was an exempt line.

**The rule:** when a throwaway check contradicts a tool that has tests and a stated specification, debug
the throwaway check FIRST. The tool encodes exemptions and edge cases that a one-liner cannot. Reversing
that order costs time and — worse — can produce a "the checker is broken" finding that is itself the bug.

**Same session, same shape, three more times:** a `pgrep -fc` that prints `0` AND exits nonzero (so
`|| echo 0` emitted two lines and broke arithmetic); a margin metric read as improved when the weighting
had rescaled its units; and a `;` instead of `&&` that let a commit through while the checker printed FAIL.
**The gate existed in all three cases and was not honoured.** Prefer `&&` over `;` whenever a check gates
an action, and never let a hand-rolled measurement overrule an instrumented one without debugging it.

## LANE MONOCULTURE — a full lane is not a prioritized one (2026-07-29, owner-flagged)

After parallelization was fixed with a dispatcher, the GPU ran at 100% with a stocked queue for hours — and
**every job in it served ONE roadmap lane (H · Memory)** while lane **F · gap#4**, which the master roadmap
calls *"the single load-bearing dependency (the crux the whole roadmap pivots on)"* and *"the must-solve
core"*, had **zero allocation**. Not deprioritized after consideration — never considered. Simultaneously
lane **E · Language**, tagged `[CPU]` and *"disjoint from A/B/C"*, sat unqueued while the 36-core pool
idled; the first lane-E runner dispatched returned a **GO in 40 seconds**.

**The dispatcher made the lane efficient without ever checking it was the RIGHT lane.** That is the failure
this rule exists for: a full queue and a busy GPU look exactly like good prioritization from the inside.

**The mechanism that produced it:** resume into a live arc → it yields interesting results → interesting
results justify the next experiment → repeat. **Momentum substitutes for prioritization.** Nothing about
the arc grew more valuable; the question simply stopped being asked.

**The check, cheap and mechanical:** when stocking a queue, name the ROADMAP LANE each job serves
(the `§ parallelization map` table). If every job names the same lane, that is the alarm — go read the
roadmap's own crux statement before adding more. Prefer stocking ACROSS lanes: the CPU-tagged lanes are
explicitly disjoint and cost nothing to run alongside GPU work.

**AND THE NEW MONITOR IMMEDIATELY CRIED WOLF — fix your own instrument before trusting it.** `lane_check.py`
shipped alarming on *"no CPU lane running right now"*. Pool jobs finish in seconds-to-minutes, so that fired
every cycle on a pool that had just SUCCESSFULLY completed its work — treating success as neglect, and
training the reader to ignore the alert within two cycles. Then its replacement reported *"never
dispatched"* when CPU lanes had been dispatched 15 minutes earlier, because the staleness marker did not
exist yet. **Two false alarms from the anti-drift tool inside ten minutes of writing it.** Fixed to alarm on
STALENESS (no CPU-lane dispatch in 45 min) with the marker seeded on first use. A monitor is an instrument;
the rule that it must be verified before its output is trusted applies to the ones you write to enforce the
rules.

**QUEUED IS NOT PRIORITIZED — check POSITION, not membership (2026-07-29, same hour).** After rebalancing
toward the crux lane, `lane_check` reported `crux=2` and passed. But the dispatcher is FIFO and the two
gap#4 jobs sat at positions **4 and 5**, behind three incumbent-lane jobs, with seven more already running —
so the crux would have started dead last, hours later, while every indicator said "served". **A coverage
check counts membership; the schedule is what actually allocates.** When rebalancing a FIFO queue, MOVE the
under-served lane to the front and verify the order, or the fix is cosmetic.

**COVERAGE IS NOT PROGRESS — an "unserved" lane may be a FINISHED one (2026-07-29).** `lane_check` flagged
lanes A · Affect and C · Self/Workspace as unserved. Checking the roadmap BEFORE queueing them showed both
are **complete**: self-schema (DR-3) and false-belief (W3) are 6-seed GO, and the roadmap's own status line
reads *"Phase-0 keystones DR-1/DR-3/P0.3/P1.2/W3 all landed"*. Queueing them would have repeated, within the
hour, the exact duplication the crux-lane lapse had just cost. **The coverage check measures where compute
goes, not whether that lane still has open work — always read the lane's roadmap STATUS before stocking it.**
A lane with no open work is correctly empty.

## READ-THE-RECORD IS NOW ON THE EXECUTION PATH, NOT IN MEMORY (2026-07-29)

The most expensive lapse of the day was spending crux-lane GPU slots re-running a result banked five days
earlier. `tools/before_you_build.sh` already existed to prevent exactly that — and was skipped, because
running it was a thing to REMEMBER and an urgency (an owner critique about prioritization) made speed feel
appropriate. **Urgency defeats checks that live in memory.** So the check moved onto the path the work must
travel:

* **`tools/queue_add.sh <lane> "<cmd>" [reason]`** — the only sanctioned way to enqueue. It greps the
  findings for the runner, PRINTS every doc that already mentions it, and **refuses to enqueue** a runner
  with prior findings unless given an explicit reason, which is then recorded inline in the queue forever
  (`#checked:on-bridge-not-rate`).
* **`tools/lane_dispatch.sh`** refuses to dispatch any line lacking `#checked:` — it sidelines it to
  `<queue>.unchecked` (never drops it) and prints `[BLOCKED]`. The heartbeat alarms on a non-empty
  `.unchecked`.

**Verified by unit test, not by absence of failure.** The first live test looked like a pass — no leaked
job — but zero lines had been dispatched (all slots busy), so the gate was never exercised. *Absence of a
leak is not evidence when the mechanism never ran.* The gate's case-statement and sideline path were then
tested directly: marked lines pass, unmarked lines block and land in `.unchecked`, queue intact.

**`pkill -f` MATCHES YOUR OWN TOOLING — kill by PID (2026-07-29, third occurrence).** `pkill -f
"[l]ane_dispatch.sh"` killed the invoking shell (exit 144) AND the heartbeat monitor, because the monitor's
own command text contained the string `lane_dispatch.sh` in its restart hint. The bracket trick only
protects against the *grep* self-match, not against every other process whose command line mentions the
pattern. **Use `pgrep` then `kill <pid>`, and never embed a process name verbatim in a monitor that also
watches for it** (v10 splits it as `'lane''_dispatch'` so the monitor can never match itself).

**Also: raise SLOTS before choosing between lanes.** The crux sat at the queue FRONT and still could not
start, because all 7 slots were held by long-running incumbent-lane jobs — queue position does not free
capacity. VRAM was 11 GB of 24 GB, i.e. ~8 slots of headroom. Raising the dispatcher 7→10 started the crux
AND a second lane immediately, with no job killed and no priority call needed. **Check headroom before
treating a scheduling conflict as a prioritisation dilemma.**

**AN ALARM YOU SILENCE BY INVENTING WORK IS MIS-SPECIFIED (2026-07-29).** The heartbeat alarmed
`QUEUE-LOW(1)` while **ten long-running jobs occupied every slot** — a state that is entirely healthy. The
first instinct was to hunt for something to enqueue, and the first candidate (the gap#5 neural reader)
turned out to be a NEW BUILD rather than a runnable job: queueing it would have meant **inventing work to
satisfy an indicator**, the same shape as the lane-monoculture failure the indicators exist to prevent.

**Fixed by measuring the thing that actually matters:** starvation = a short queue AND **idle capacity right
now** (`slots - running > 0`). A drained queue with every slot busy is fine and no longer alarms. **When an
alarm's remedy is "manufacture something", the alarm is measuring the wrong quantity — fix the alarm, do
not feed it.**

## "IS THE RUNNING JOB DOING WHAT I THINK?" — `tools/device_check.sh` (2026-07-29)

Every other mechanism built today is a SCHEDULING check (right work, right lane, right order, checked
against the record). None asks whether a *running* job is doing what it appears to. The crux ran **47
minutes on the CPU** with every scheduling and liveness indicator green — CPU-time tracked elapsed-time at
99%, so it was genuinely computing, on the wrong device. The runner printed the cause in **line 1**
(`os.environ.setdefault('SIM_BACKEND','numpy')` silently winning) and it was scrolled past while reading
for a verdict.

`bash tools/device_check.sh [--quiet]` reads each running job's actual stdout via `/proc/<pid>/fd/1` and
reports its device; exit 1 on any CPU-bound job.

**IT SHIPPED WITH A FALSE PASS AND THAT IS THE REAL LESSON.** v1 pulled the log path from `ps` args — but
the shell consumes redirects, so the path was never there, every job read `unknown`, and it printed
**"OK — no job is silently on the CPU"** having determined *nothing*. A check that passes on no information
is worse than no check, because it manufactures confidence. Fixed twice over: the path now comes from the
process's real fd, and **`UNDETERMINED` is a FAILURE, never a pass**. Also handles STALE logs — a log older
than its process is a leftover from a previous run (the killed CPU crux logs still said "numpy" 62 minutes
later and were nearly misread as a repeat failure).
