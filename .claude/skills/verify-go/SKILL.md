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
