---
type: finding
status: live
date: 2026-08-21
mechanism: continuous-ideation-novel-between-turn-thought
lane: continuity
integration_faculty: continuous-ideation
seeds: [42, 43, 44, 45, 46, 47]
seed-waiver: The MECHANISM (novelty/blend of the 2-source attractor + the blend-not-noise / blend-not-single /
  untrained controls) IS run at 6 seeds x 2 scales (n=400/k=40/n_mem=6 and n=1200/k=60/n_mem=20). The INTEGRATION arm
  (does the real tick_session surface a FLAGGED novel concept ON, and is it byte-identical to today OFF) is a
  deterministic plumbing/honesty proof on one representative organ (seed 42) — not a stochastic effect size; the
  numpy light-path recall selection is deterministic. The blend attractor it drives is the 6-seed GO de-risk it reuses.
verdict: GO
runner: research/runners/_continuous_ideation_verify.py
artifacts:
  - research/findings/raw/_continuous_ideation/verify.json
  - research/runners/_continuous_ideation_verify.py
  - webapp/continuous_engine.py
---
# Continuous IDEATION: the brain occasionally GENERATES a novel blended thought between turns (default-OFF) — GO

Artifact: research/findings/raw/_continuous_ideation/verify.json (runner: research/runners/_continuous_ideation_verify.py)

**One line (continuous life — the creativity/novelty rung).** The between-turn idle wander today SELECTS one stored
basin and speaks its concept (recall). Behind a NEW default-OFF flag `BRAIN_CONTINUOUS_IDEATE`, the wander now
OCCASIONALLY (every Nth tick) GENERATES instead: it drives a BLENDED cue of the TWO most curiosity-active basins into
a sparse associative-attractor, which settles into a NOVEL recombination that was NEVER stored — novelty from the
DYNAMICS. The novel idea is FLAGGED (`kind=novel-association`, `is_fact=False`) and surfaced on a channel disjoint from
recall, so the next turn frames it as "a thought that occurred to me", never a stored fact. OFF is byte-identical to
today's live continuous wander (the just-flipped default is protected).

## The mechanism (reuse, not reinvent)
The novel blend rides the GO de-risk `research/runners/_generative_attractor_wander_derisk.py` verbatim
(Tsodyks-Feigelman sparse-Hopfield + the `ca3_ff_inhib` MEAN+std dynamic-threshold settle — a fixed
feedforward-inhibition threshold, not a forced top-k, is what lets a two-source blend stay BALANCED instead of
collapsing onto one source; finding `2026-08-20-generative-attractor-wander-derisk-blended-cue-settles-to-novel-state`).
This runner drives the **two-source** blend (the task's "two most curiosity-active basins") and adds a clean
DISCRIMINATING noise control the original de-risk lacked: the balance measured **on the two CUED sources** (a random
cue has no A/B structure, so it is not balanced on A,B — where the original "is noise balanced on SOME two" control
was capacity-limited at small scale).

## The verify (`research/runners/_continuous_ideation_verify.py`, a `tools.verdict.Verdict`; numpy-CPU, foreground)

<!--derived-->
_(values below are rounded means from the cited `research/findings/raw/_continuous_ideation/verify.json`; exact values there.)_

| proof | result |
| --- | --- |
| (A) OFF byte-identical | BRAIN_CONTINUOUS_IDEATE unset: no `ideation` key on the tick, `recent_ideation()` None, the recall wander == today's selection (`cat`); enabling the flag on a NON-ideation tick is a no-op |
| (B) MECHANISM (6 seeds x 2 scales) | 2-source blend: `blend_balance` = 0.724, `novelty` (max overlap any single stored) = 0.724 (a single recall reads 1.000), overlap with any OTHER non-cued basin = 0.153; fixed point at every seed |
| (B) INTEGRATION | the real `tick_session` records a FLAGGED novel-association (sources `cat`+`dog`); `recent_ideation()` surfaces it tagged (`is_fact=False`), consumed once |
| (C) blend-not-single | single-cue balance on the two sources = 0.093 « blend 0.724 (a single cue recovers ONE pattern, no novel blend) |
| (C) blend-not-noise | noise-cue balance on the two sources = 0.107 « blend 0.724 (a random cue is not balanced on the cued sources) |
| (C) untrained (W=0) | best overlap = 0.000 (the threshold rule alone does not fake completion) |
| (D) HONESTY | `is_fact=False`, `kind=novel-association`; NEVER enters the recall channel (`recent_wander()` None on an ideation tick); writes NO store / manufactures NO fact (organ store fingerprint unchanged) |

**GO** = OFF byte-identical AND ON produces a genuine flagged novel blend AND the blend-not-noise / blend-not-single /
untrained controls all separate by > 0.15 AND honesty preserved. All hold.

## What is spiking / mechanism vs what is host (declared — the honesty boundary is a deliverable)
- **The novelty rides the attractor DYNAMICS** (the de-risked sparse-Hopfield threshold settle) — a genuine
  never-stored recombination, not a lookup.
- **DECLARED SCAFFOLDS** (named, not hidden): (1) the fast standalone **numpy** attractor is the de-risked stand-in
  for the on-substrate CA3 blend (the SAME latency residual the self-init organ declares — cupy CA3 is ~seconds,
  numpy@scale is minutes; the on-substrate CA3 port is the mapped next step); (2) the SELECTION of the two source
  basins rides the organ's spiking curiosity gains (one-brain merge #1); (3) the every-Nth-tick cadence is a
  host-timed scheduler (WHEN to ideate, like the idle-tick clock). FUNCTIONAL creativity correlate, NOT a phenomenal claim.

## Integration state
Added to `webapp/continuous_engine.py` (`ideation_enabled` / `_ideation_wander` / `recent_ideation` / the tick branch)
+ `webapp/server.py` (a default-OFF `ideation_drives` lead in both response paths), strictly additive, DEFAULT-OFF
behind `BRAIN_CONTINUOUS_IDEATE`. A default-ON flip is NOT taken here — it needs (a) the on-substrate CA3 port of the
blend and (b) a no-regression / owner-UX soak that the occasional novel lead reads as a welcome creative aside, not
noise. Ledger row `continuous-ideation` (de_risked YES, wired YES, on_by_default NO).

## Residual / next step
Port the two-source blend cue onto the validated on-substrate CA3 harness (replace the numpy stand-in) at the organ's
operating point; then soak the cadence + framing for a default-ON flip. The novel idea currently surfaces as a bare
"A + B feel connected" association; giving it a spoken, grounded rationale (why the two connect) is a follow-on.
