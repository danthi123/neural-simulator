# N1 adaptive activity-gated weaning = NEGATIVE multi-seed (1/3) — the online readiness probe cannot robustly produce durable post-wean consolidation; post-wean durability is non-monotonic and chaotically sensitive to the exact commit step. N1 is BANKED as "biologizable-in-principle, robust auto-wean genuinely hard" per the reasonable-budget gate. N8 ✅ + N6 ✅ stand. — 2026-06-07

**Status:** NEGATIVE multi-seed → BANK (the owner-committed "one more targeted iteration, then stop"
reasonable-budget stopping point). NO `sim/` edits; the wean flags are additive + default-off. No orphan
processes. The README this session is unrelated (separate side task).

## The one-line result

Adaptive activity-gated weaning of the navigation action-heuristic — the biologically-correct robust fix the
fixed critical-period clock couldn't be (Hensch: real critical periods close when the circuit is *ready*, not on a
fixed clock) — DEMONSTRABLY weans each seed at a data-driven readiness point, but it does NOT robustly produce a
*durable* self-sufficient `IT→cortex` mapping. The longer sustained-readiness probe (500 steps, this run) scored
**1/3 HOLD** — WORSE than the prior 200-step-probe run's 2/3 — and seed 43 flipped HOLD→COLLAPSE on a mere
300-step-later commit. The readiness probe measures *transient post-teaching navigation*, not *durable
consolidation*, and the commit-timing→durability relationship is non-monotonic and seed-chaotic, so no online
readiness criterion reliably lands every seed's narrow consolidation window. **N1 is biologizable in principle but
not robustly auto-weanable in a reasonable budget.**

## The decisive table (grid-8, single fixed goal, 11000 steps; post-wean = last-quarter mean distance; HOLD ≤ 2.5 / COLLAPSE ≥ 4.0)

| seed | 200-step probe (`bpdythg0t`) commit → post-wean | 500-step probe (`b1f1y2oid`) commit → post-wean |
|---|---|---|
| 42 | @2200 → 5.89 ✗ (committed too early) | @4000 → **6.75 ✗** (over-shot the ~3000 sweet-spot) |
| 43 | @700 → 1.84 ✓ | @1000 → **8.33 ✗** (300 steps later → flipped to COLLAPSE) |
| 44 | @1200 → 1.62 ✓ | @1000 → **1.84 ✓** |
| **total** | **2/3 HOLD** | **1/3 HOLD** |

Seed 42's binned trajectory this run: `[4.32, 2.8, 2.99 | 6.75, 6.48, 6.71]` — navigates ~3 while the heuristic
teaches/ramps, then collapses to ~6.7 the moment it fully weans off (~step 5500) and stays there. Its probe history
shows the over-shoot directly: probes at 500/1500/2500 read 10.0/5.9/5.5 (not ready), the 3500–4000 probe read 1.91
(ready → commit at 4000) — but 4000 is past seed 42's narrow consolidation sweet-spot (~3000, where the fixed clock
held it at 2.14; the 5000-fixed-crit collapsed it). Seed 43's first probe (500–1000) read 2.45, *barely* under the
2.5 threshold → immediate commit at 1000 with only ~1000 teaching steps → under-consolidated → collapse 8.33.

## Why it can't be fixed by tuning the probe (the honest mechanism)

- **The probe measures the wrong thing.** During a readiness probe the heuristic is briefly OFF, but the agent was
  just taught up to the probe start, so the freshly-reinforced mapping holds it near goal for the probe window. That
  is *transiently propped-up* navigation, not *durable* consolidation. After a permanent wean, ongoing plasticity
  without the teacher lets the mapping decay and the agent drifts out and cannot re-acquire.
- **Durability is non-monotonic in commit timing, and seed-chaotic.** Seed 43 held at a 700-step commit but
  collapsed at a 1000-step commit (300 steps *more* teaching made it *worse*). Seed 42 held at a fixed 3000-crit but
  collapsed at 2200, 4000, and 5000. There is no monotone "more teaching → more durable" relationship for the probe
  to exploit, so a longer probe (which only shifts commit timing) cannot systematically improve durability — it just
  reshuffles which seeds land their window (2/3 → 1/3 here).
- **This is the reasonable-budget stopping point.** The owner set "one more targeted iteration, then stop"; the
  targeted iteration (sustained 500-step probe) not only failed to fix seed 42, it lowered the overall hold rate.
  Further wean-knob tuning is whack-a-mole and is not pursued.

## What stands, and the honest characterization

- **N8 ✅ (genuine GPi→thalamus disinhibition) and N6 ✅ (spiking accumulate-then-commit decision + thalamic-source
  readout) are biologized and beat/≈ the cheats** — those wins are unaffected by this result.
- **N1 = biologizable-in-principle, not robustly auto-weanable.** Every seed CAN navigate self-sufficiently from
  learned perception at *some* critical-period length (seed 42 holds at fixed-3000-crit 2.14; seed 43 holds at the
  200-probe 700-commit 1.84; seed 44 holds across multiple configs). The MECHANISM (an innate scaffold teaches a
  learned `IT→cortex` circuit, then fades — standard developmental biology) is real. But no fixed OR online-adaptive
  recipe robustly lands all three seeds' narrow, non-monotonic consolidation windows in a reasonable budget.
- **N1/N2/N7 are ONE characterized perceptual cold-start boundary.** This is the honest core of the nav cheats: the
  agent has a *weak attractive visual pull* but not *precise goal-localization*; reward+exploration does not
  bootstrap it (the perceptual-bootstrap gauge was BLOCKED), and neither a fixed nor an adaptive developmental
  scaffold robustly consolidates it. Closing N1 (the heuristic) genuinely requires solving this perceptual
  cold-start — a sharper-IT goal-localization front-end (a deeper investment), not more weaning tuning.

## Net for the nav arc + the decision fork (surfaced to owner)

**The basal-ganglia OUTPUT (N8) and the action DECISION + selection (N6) are biologized in spikes and beat/≈ the
original cheats. The perceptual INPUT layer (N1/N2/N7) is a well-characterized hard boundary** — a genuine,
multiply-confirmed honest negative, which (per the project goal) is itself a scientific deliverable. The remaining
nav cheats N5 (coord-free reward) and N9 (spiking SNc) are lesser and entangled with the same perception web.

Decision fork for the next step (reasonable-budget framing):
- **(i)** Invest in the deeper **sharper-IT goal-localization front-end** (the gauge's prescription) to genuinely
  close the N1/N2/N7 perceptual cold-start — the most expensive nav option.
- **(ii)** Tackle the lesser reward/dopamine cheats **N5/N9** (more tractable than the perception layer, but the
  audit found N5/N2/N9 form a reward+perception web, so N5 partly re-enters the perception boundary).
- **(iii) [recommended]** Declare the nav arc at a principled stopping point — **N8 ✅ + N6 ✅ biologized; the
  N1/N2/N7 perceptual layer characterized as an honest boundary; N5/N9 documented as lesser** — and advance to
  **roadmap step 3 (single-instance unification: fold navigation + conversational onto one always-on
  `SimulationBridge`)**, the higher-value owner-prioritized work, returning to the sharper-IT perceptual front-end
  later as a dedicated arc.

## Artifacts

- `research/findings/raw/_n1_adaptive_pw500_s{42,43,44}.json` (this 500-probe run); the prior 200-probe run's
  per-seed results are recorded in `AUTONOMOUS_STATE.md`.
- Finalizer: `research/findings/raw/_n1_adaptive_finalize.py` (one-command per-seed HOLD/COLLAPSE verdict).
- Prior: `2026-06-06-N1-critical-period-scaffold-TRACTABLE.md` (with the multi-seed correction header),
  `2026-06-06-perceptual-bootstrap-gauge-BLOCKED.md`, `2026-06-06-N1-heuristic-removal-BOUNDARY.md`,
  `2026-06-06-N6-decision-biologized-CONCLUSION.md`, `2026-06-06-N8N6-combined-readout-GO.md`,
  `2026-06-06-navigation-cheat-audit-and-conversion-plan.md`.
