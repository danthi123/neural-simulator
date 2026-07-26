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
6. **REPRODUCE BY CALLING THE ORIGINAL CODE PATH — never by re-implementing it. Then vary ONE documented parameter.**
   If a recorded result came from `runner.foo()`, call `runner.foo()`. A hand-rolled reconstruction silently omits
   steps you did not know were load-bearing, and the divergence surfaces hours later — after conclusions have been
   built on it. (2026-07-25: **FOUR withdrawn conclusions on one thread**, every one traceable to this. My
   reproduction of a recorded 87.5% omitted `apply_concept_topographic_bias` — a pre-training cortical-somatotopy step
   applied *before* any learning — and used a different `word_to_idx` ordering. On that basis I reported "the reference
   harness fails", escalated it to "a shared-code path is broken, past results may be invalid", then had to withdraw
   the lot. I also measured at 200 events when the record was 800, and with a 1-word probe when the effect lives at 16
   words.) **Corollary: when the original path takes a parameter, override THAT parameter surgically — do not fork the
   function.** A one-line monkeypatch of a documented knob keeps every other step honest.
7. **DID IT EVER WORK — and at WHAT SETTINGS? Search the record for a known-good configuration BEFORE debugging.**
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

## What this skill MUST NOT do
- Rubber-stamp — a skeptic that "confirms" without trying to break it did nothing. The prompt must push to REFUTE.
- Verify only the happy path — test the claim against the case you'd EXPECT to break it (a run you KNOW is broken, the seed you fear).
- Treat a refutation as needing less scrutiny than a confirmation (rule 3 of the SILENT-FAILURE CLASS).
- Touch the science verdict itself dishonestly — this skill SHARPENS a claim to the truth, it never launders a weak result into a strong one.

## Why this skill exists
2026-07-24 (evolve-skills): adversarial verification was run ad-hoc (via one-off Workflow panels) and reliably caught real over-claims — but it lived only in-session, not in a skill, so it depended on remembering to do it. This encodes it as a reflex triggered by the commit itself. Pairs with the neural-simulator skill's SILENT-FAILURE CLASS (the specific checks) — this is the procedure that runs them before a GO lands.
