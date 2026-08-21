---
type: finding
status: contributing
date: 2026-08-20
mechanism: dendritic-plateau-coincidence-burst
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_step5_conversation_visible_recall_derisk.py — an adaptive WEAK-encode D5 store,
  read through the REAL EpisodicRecallOrgan.recall (the apical_cue the reply quotes), consolidated across several
  mark_recall→tick rounds by the REAL continuous_engine.consolidate_used_memory, re-read through the handler; teeth =
  handler-moves / lesion-vanishes / specific / still-usable, with a tools.verdict block + attributable_to.
runner: research/runners/_d5_step5_conversation_visible_recall_derisk.py
external: NO-EXTERNAL-NEEDED — closes the step-4 residual (the handler-visible recall was flat) by finding the
  operating point; the redirect (a graded apical read) is an internal instrument change, not a literature question.
artifacts:
  - research/findings/raw/_d5_step5_visible/summary_6seed.json
  - research/findings/raw/_d5_step5_visible/seed42.json
  - research/findings/raw/_d5_step5_visible/seed43.json
  - research/findings/raw/_d5_step5_visible/seed44.json
  - research/findings/raw/_d5_step5_visible/seed100.json
  - research/findings/raw/_d5_step5_visible/seed101.json
  - research/findings/raw/_d5_step5_visible/seed102.json
---
# Learn-through-use CAN be made CONVERSATION-VISIBLE — the real recall rises after use (4/6 clean, weak encode); full reliability needs a GRADED apical read

Artifact: research/findings/raw/_d5_step5_visible/summary_6seed.json + the per-seed
research/findings/raw/_d5_step5_visible/seed42.json … seed102.json.

**One line.** Step-4 wired live learn-through-use but its gain was in the store's robustness RESERVE, invisible to the
production recall ([[2026-08-20-d5-live-consolidation-wired-default-off-strengthens-robustness-reserve-NOT-yet-conversation]]).
Step-5 asks: is there a regime where the PRODUCTION recall visibly moves? **Yes — the real `EpisodicRecallOrgan.recall`
`apical_cue` (the number the reply quotes) RISES after between-turn consolidation, on 5/6 seeds, clean GO (move +
lesion-vanishes + specific) on 4/6.** The difference between "the memory got more resilient" and "you can tell it
learned" is closed in principle; two honest limits keep it from 6/6.

## Why step-4 was flat: the OPERATING POINT, not saturation
<!--derived-->
A sensitivity trace shows `apical_cue` is MONOTONE in the within-dog weight with real headroom (~0.5 at w≈40 → 0.625 at
60 → 1.0 at 77). Step-4 encoded at `train_events=40` (landing w≈60, near the top) so consolidation pushed into the
ceiling → no visible move. **Step-5 regime: encode WEAK** — `train_events≈5-8` → w≈30-42, borderline `apical_cue`
~0.25-0.45, `in_memory=True` — so the memory starts a genuinely LABILE trace BELOW its read ceiling; the step-4 loop
then strengthens it and the handler read rises. Production-reachable: `in_memory=True` at turn T, so the server's
`mark_recall` (guarded on `in_memory`) arms consolidation exactly as in a live conversation, and "a few turns later" =
weight accumulated across several `mark_recall`→tick rounds.

## The 6-seed verdict (GO 4/6; moved 5/6) — measured through the REAL handler recall
<!--derived-->
| seed | te | apical_cue T → T+k | verdict |
|---|---|---|---|
| 43 | 5 | 0.2857 → 0.5000 | GO |
| 44 | 8 | 0.3750 → 0.5000 | GO |
| 100 | 5 | 0.3333 → 0.4167 | GO |
| 102 | 5 | 0.3571 → 0.5714 | GO |
| 101 | 5 | 0.4545 → 0.7273 | NO-GO (moved; crosstalk) |
| 42 | 6 | 0.4286 → 0.4286 | UNDEFINED (flat; ceiling) |

Every build: **lesion-off is byte-identical + flat** (`hash_before == hash_off`, the later read identical) → the move
vanishes exactly when the flag is off, driven by the consolidation loop (not decoration). NO `sim/` edit; measured on
the production `EpisodicRecallOrgan.recall`, not the arc's robustness-margin instrument.

## The two honest limits (why 4/6, not 6/6) — and the named redirect
<!--derived-->
1. **Structural read ceiling (seed42 UNDEFINED, the decisive one).** `apical_cue` is a per-held-cell BINARY UP-fraction,
   so each emergent membership has a STRUCTURAL ceiling set by how many held cells can ever latch. When the weakest
   *completing* encode already sits at that ceiling, there is no headroom → flat. A dedicated plateau-window (`b_adapt`)
   sweep PROVES this is not the tunable lever: the weight fixed point reached w≈70 at every window 0.8→0.0 while
   `apical_cue` stayed PINNED at 0.6667, and lowering the window only re-introduced the interference-runaway step-3's
   self-termination exists to prevent. ⇒ the obvious reliability knob is ruled out; the redirect is a **GRADED /
   continuous apical read** replacing the binary UP-fraction (escapes the quantization ceiling) — a next-mechanism, not
   a wall.
2. **Sub-threshold crosstalk (seed101 NO-GO).** It moved strongly (0.4545→0.7273) but dog consolidation nudged the
   control topic 'cat' apical 0.0→0.0667 (tripping the STRICT specificity tooth). This is **moat-preserving**:
   0.0667 ≪ the 0.20 completion threshold, so cat still reads `in_memory=False` — a bit of sub-threshold spillover on
   that membership, not a false recall.

## Bottom line — the corrected flip preconditions
The step-4 gap ("not load-bearing on the conversation") is closed IN PRINCIPLE: a used memory now recalls VISIBLY better
(the quoted completion rises), specifically, vanishing under lesion. But the production-default flip (step-5→flip,
owner-UX) needs MORE than soak + no-regression: (a) **`note_topic` must encode WEAK** (labile trace te≈5-8) — production's
`train_events=40` saturates past the read's headroom, so a strong-encoded memory shows no visible move; (b) the move is
**~4/6-reliable**, limited by emergent-membership variance (the structural ceiling + occasional moat-safe crosstalk),
NOT fixable by the plateau-window knob — a genuinely reliable flip wants a **graded apical read**, or acceptance of the
~4/6 rate with a disclosure that renders partial moves. (Agent-built; parent sanity-verified the 4/6 + the per-seed
apical moves + lesion-byte-identity + the moat-safe crosstalk from the artifacts; the structural-ceiling claim is the
agent's b_adapt sweep, and it names the graded-read redirect rather than declaring a wall.)
