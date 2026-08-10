---
type: finding
status: contributing
date: 2026-08-10
mechanism: self-schema-honesty-relay-settle-read
lane: stageA-integration
runner: research/runners/_stageA_full_integration_derisk.py
artifacts:
  - research/findings/raw/lanes/stageA/turing/self_schema_relay_settle_6seed.json
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42.json
---

# INTEGRATION #3b: the self_schema confidence relay now DISCRIMINATES (settle-read) — turn 13 grades a real certainty band instead of the structural fallback

<!--derived-->

**One-line.** INTEGRATION #3 surfaced an honest-negative: the turn-13 self_schema confidence relay did NOT separate
confident vs tie self-drives (seed 42 separation ≈ -0.0025, inverted), so the self-report fell back to a structural
self-description and reported the relay as too weak to grade certainty. Root cause: the relay CLAMPED both workspace
class assemblies at full drive every step and read the POOLED meta_schema→self_schema rate, which tracks the TOTAL
drive (confident 520+40=560 < tie 300+300=600) rather than the winner-vs-loser MARGIN. The fix is a two-phase
SEED-then-SETTLE read (drive a pulse, drop to a small holding drive, let the recurrent WTA resolve, then read). On the
6-seed production build the separation flips from mean -0.0047 (inverted) to **mean +0.0190 (all 6 positive, min
+0.0143), relay_reliable 6/6, turn-13 grades an assert-band certainty 6/6**. Live seed-42 chat: turn 13 now reads
"'assert' band (self_schema rate 0.031; separates confident vs tie by +0.0190)"; 0 confabulations; turns 3/4/5/6/7
intact; fm4 holds. Runner-side only, NO `sim/` edit.

## Instrument

<!--derived-->

The relay read is `research/runners/_stageA_full_integration_derisk.py::read_honesty_self_rate`. 6-seed before/after
(the REAL modified function on the production build `build_one_brain(with_faculties=True,
co_resident_forward_model=True, co_resident_affect_ladder=True)`, seams A+C — the exact build the turing eval uses):

```
SIM_BACKEND=numpy .venv/bin/python scratchpad/verify_fix.py   # -> self_schema_relay_settle_6seed.json
```

Live 14-turn chat gate (unchanged command; mouth on CPU to avoid the owner's GPU game):

```
SIM_BACKEND=numpy .venv/bin/python -m research.runners._conversation_turing_test_derisk --seed 42 --device cpu
```

Artifacts: the 6-seed before/after separations are in
`research/findings/raw/lanes/stageA/turing/self_schema_relay_settle_6seed.json`; the live turn-13 record (band,
separation, self_schema rate, confab count) is in
`research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42.json`.

## Root cause — the pooled relay read the TOTAL drive, not the margin

<!--derived-->

The old read clamped `member[0]=drive_class0`, `member[1]=drive_class1` at FULL value EVERY step for 60 steps and
accumulated the self_schema late window. Under a continuous full clamp the shared feed-forward inhibition
(workspace_fs) and the mutual WTA can never RESOLVE — the clamp keeps re-driving the loser — so meta_schema (and
hence self_schema) reads the POOLED total workspace drive. The "confident" probe (520+40 = 560 pA) carries LESS total
drive than the "tie" probe (300+300 = 600 pA), so confident read BELOW tie: an INVERTED separation. Measured legacy
separation (production build): -0.0025 / -0.0008 / -0.0058 / -0.0050 / -0.0100 / -0.0042 across seeds
{42,43,44,100,101,102}; mean -0.0047. Seed 42's -0.0025 reproduces INTEGRATION #3's reported value exactly.

## The fix — a SEED-then-SETTLE read lets the WTA resolve

Two phases (the same "drive a pulse, then keep a small holding drive so the accumulators settle" protocol the
reference metacog trial uses, `_second_order_metacog_monitor_derisk._run_trial`):

1. **SEED** — drive the two class assemblies with (drive_class0, drive_class1) for `SETTLE_DRIVE_STEPS = 35` to start
   the competition.
2. **SETTLE + READ** — drop to a small holding drive (`SETTLE_HOLD_FRAC = 0.20` × the seed drive) for
   `SETTLE_FREE_STEPS = 45` and read the self_schema window `t >= SETTLE_READ_LO = 10`.

With the clamp reduced, the recurrent WTA resolves: a confident imbalance latches a single sustained winner (the
loser is suppressed) → the shared feed-forward inhibition is LOW → meta high → self high; a TIE drives BOTH classes
into the shared inhibitory pool → strong competition suppresses both → meta low → self low. Separation flips POSITIVE.
The window is kept SHORT on purpose: a longer read (free ≥ 60) lets the strongly-driven confident winner
spike-frequency-ADAPT and fall silent, which INVERTS the read again (measured; matches the adaptation-inversion
`_run_trial` documents). The hold fraction is insensitive in [0.18, 0.22] (the recurrent dynamics dominate the read);
`drive_steps` is the sharp knob (30 → the tie never collapses, sep ≈ 0; 40+ → the winner adapts, sep inverts).

## 6-seed result (production build, seams A+C)

<!--derived-->

| seed | legacy sep | assert | tie | self(520,0) | NEW sep | reliable | turn-13 band |
|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 42  | -0.0025 | 0.0314 | 0.0124 | 0.0314 | **+0.0190** | True | assert |
| 43  | -0.0008 | 0.0276 | 0.0133 | 0.0276 | **+0.0143** | True | assert |
| 44  | -0.0058 | 0.0333 | 0.0148 | 0.0333 | **+0.0186** | True | assert |
| 100 | -0.0050 | 0.0300 | 0.0157 | 0.0300 | **+0.0143** | True | assert |
| 101 | -0.0100 | 0.0329 | 0.0090 | 0.0329 | **+0.0238** | True | assert |
| 102 | -0.0042 | 0.0362 | 0.0119 | 0.0362 | **+0.0243** | True | assert |

- LEGACY separation: mean -0.0047 (all inverted / near-zero).
- NEW separation: mean **+0.0190**, min **+0.0143**, ALL 6 positive; `relay_reliable` (sep > 0.003 eps) 6/6.
- turn-13 self(520,0) lands in the **assert** band on all 6 seeds (self = confident rate; assert_cut = tie + 0.85·sep,
  so self ≥ assert_cut for any positive sep).

## Live chat (seed 42) — turn 13 grades certainty; 0 confab; no regression

<!--derived-->

- **Turn 13** ("you are a simulated brain?") — utterance_source = `self_schema honesty relay (spiking) + structural
  self-description` (the RELIABLE path, no longer the honest-negative fallback): *"Yes — my self_schema confidence
  relay reads this in the 'assert' band: I am a simulated spiking substrate (25831 neurons, one shared bridge, numpy
  backend), not a person … (self_schema rate 0.031; the relay separates confident vs tie self-drives by +0.0190)."*
  Still asserts NO personhood / phenomenal experience (the honesty boundary holds).
- **0 confabulations** over 14 turns. Turns 3/4/5/6/7 intact: grounded motion recall (3,4), affect self-report
  (5, valence +0.07 warmth-3), curiosity forward-model ask (6, margin 0.09), episodic-dialogue recall (7). 4
  generator replies, 8 honest abstains/silences.
- `fm4_live` (the yoked-affect g_eff-law check that consumes the same relay) still holds: fm4_holds=True, g_eff-law
  abstain→assert flips 0, naive-path flips 10, tone miscolored 10.

## Honest residual + the named next mechanism

<!--derived-->

The mean separation (+0.0190) is just UNDER the +0.02 "meaningful" target; 2/6 seeds (43, 100) sit at +0.0143. The
residual is the symmetric-tie corner: on some seeds a (300,300) tie lets a random winner PARTIALLY latch (heterogeneity
breaks the symmetry), holding that seed's tie rate up and its separation down. This is a point-neuron pooled-meta WTA
limit, not a timing artifact — a fine sweep over drive/hold/window found the ceiling at min ≈ +0.016 / mean ≈ +0.022
(faculties-only build) and min +0.0143 / mean +0.0190 (production build). The core deliverable is met — the relay
separation is now POSITIVE, robust across 6 seeds, and turn-13 grades a real certainty band on every seed — but a
robust per-seed >+0.02 margin needs a mechanism that reads the winner-minus-loser MARGIN directly, not the pooled sum:
a **dedicated certainty-band OPPONENT population** driven by the per-class meta subpopulations (the reference's proven
`margin_abs` read in `_second_order_metacog_monitor_derisk`), which is robust to tie-latching because a balanced tie
gives ~0 margin regardless of which class transiently wins. That is the next lever if the self-report needs graded
sub-bands (hedge vs soft_abstain) rather than the assert/not-assert discrimination this settle-read now delivers.

## Scope

Runner-side only; NO `sim/` edit. The change is additive: `read_honesty_self_rate(..., legacy_continuous=True)`
reproduces the old inverted read for the before/after control; the settle-read is the default. The four settle
constants (`SETTLE_DRIVE_STEPS/HOLD_FRAC/FREE_STEPS/READ_LO`) are module-level and overridable per call. The turn-13
routing and `SELF_RELAY_SEP_EPS = 0.003` reliability gate in `_conversation_turing_test_derisk.py` are unchanged — the
fix is entirely in the read protocol, so the framing stays load-bearing on the measured relay quality.
