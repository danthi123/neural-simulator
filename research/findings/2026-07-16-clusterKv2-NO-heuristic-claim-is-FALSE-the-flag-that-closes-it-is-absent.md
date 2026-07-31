---
type: finding
status: live
date: 2026-07-16
---

# The Cluster-K-v2 "NO heuristic" claim is FALSE — the flag that closes the heuristic is absent from its own recorded command

**2026-07-16. Status: VERIFIED from the run's own artifacts + the runner source. LOGGED, NOT CHASED (owner directive).**
**Scope: a provenance/claim defect. NOT a re-measurement. No nav run was executed for this finding.**

## The claim

`research/findings/2026-05-01-cluster-k-v2-breakthrough.md:175`:

> Score: 2.97 ± 0.12 at 16×16 with NO hand-coded perception, **NO heuristic,
> NO direct (gx, gy) or (x, y) access**. Only simulated reward (Manhattan-
> based) remains as a non-biological signal.

Propagated to `CLAUDE.md:3196`: *"Closes 4 of 5 original cheats (heuristic, (gx,gy), (x,y), beacon)."*

## The fact

The heuristic was **ON at full strength** for that run. Four independent confirmations:

1. **The default is ON.** `g11_bg_runner.py:9475` — `--heuristic-strength`, `default=1.0`.
2. **The run's own recorded command has no heuristic flag.**
   `research/findings/raw/g11_bg/k_v2_stress_16x16_seed100.cmd.json` `extra_args` =
   `--moving-goal --goal-schedule multi --deterministic --enable-msn-lateral-inhibition
   --enable-d1-d2-asymmetry --enable-striatal-pv-fsi --enable-cluster-a-closed-loop
   --enable-cluster-e-topography --enable-dlpfc-wm --enable-pfc-nmda --enable-visual-cortex
   --visual-cortex-action-warmup-steps 600 --grid-size ...` — **no `--heuristic-*` anywhere.**
3. **Nothing in that command zeroes it.** `h_strength` reaches `0.0` on exactly three paths
   (`g11_bg_runner.py:7008-7044`): `in_sleep`, `in_goal_silence_step`, `heuristic_wean_adaptive`
   (default `False`, `:4123`, and NOT implied by `--enable-visual-cortex`), plus
   `cue_reflex_replaces_heuristic` — **none active here.** Control falls to `else: h_strength =
   heuristic_strength` (`:7046`) → `h_drive = 800.0 * 1.0` pA.
4. **`--visual-cortex-action-warmup-steps` does not touch it.** Its only effect (`:6722-6730`) is to open
   the `visual_cortex_action` **plasticity gate** at step 600. It never reads or writes `h_strength`.

And the heuristic reads the goal coordinates **directly** (`:7047-7078`): `gy > y` / `gx > x` → 800 pA into
`cortex_N/E/S/W`. So the same run also had **direct (gx, gy) and (x, y) access** — the second half of the
same sentence is false for the same reason.

## Root cause: a true claim copied onto a config that lacks the flag

The claim is **true for a different config**. The 2026-04-27 perception-arc flagship passes
`--cue-reflex --cue-reflex-replaces-heuristic`, and that pair genuinely sets `h_strength = 0.0`
(`:7042-7045`, *"Stage 3: reflex replaces heuristic. The reflex below computes cortex drive from beacon
sensor activations instead of (gx,gy)"*).

⇒ **`CLAUDE.md:3311`'s "NO direct (gx, gy), NO direct (x, y), NO heuristic" is CORRECT** — that config
carries the flag. Cluster K v2 does not, and inherited the sentence anyway.

This is the day's recurring failure mode, third instance: **the instrument recorded the truth and the
write-up asserted the opposite.** The `.cmd.json` sidecar has faithfully stored "no heuristic flag" since
2026-05-01; nobody read it back against the claim. Cf. the deep-credit GO (the runner printed
`SIGNAL=False` / "HONEST NEGATIVE" in all three files while the finding claimed "6-seed GO,
anti-cheat-clean").

## What this does and does not invalidate

**Does NOT invalidate the number.** 2.97 ± 0.12 was measured; nothing here says the run was broken.
What is invalid is its *description*. The honest statement is:

> 2.97 ± 0.12 at 16×16 was measured **with the hand-coded heuristic at full strength (800 pA driven from
> direct goal-coordinate reads)**. The independent contribution of the Cluster-K-v2 visual pathway is
> **unquantified**.

**Open, deliberately NOT chased** (owner: *"Log it. Don't let it become a reason to defer the longest pole"*):
- What the config scores at `--heuristic-strength 0`. That single run is the whole experiment — it splits
  "visual cortex navigates" from "the heuristic navigates and the visual cortex rides along."
- **Whether that run's visual cortex was inert at all.** On 2026-07-16 a reproduction
  (`raw/g11_bg/nav_linux_verify.json`) had its visual pathway silently disabled by
  `g11_bg_runner.py:6731-6732` — `except KeyError: pass  # Gate not present (no IT -> cortex synapses)`.
  The k_v2 artifacts carry **no stdout log**, so whether the gate opened in the original run **cannot be
  determined from the record**. Stated as unknown, not as an accusation.
- Note `sum_finalQ` is **not a stored field** — `k_v2_stress_16x16_seed100.json` holds
  `mean_distance_quarters`, and the headline is derived from it (`:8158`).

If both opens resolve badly the result is "the heuristic did the navigating," but **that is a hypothesis
this finding does not test.**

## Fixed here

- Correction block appended to `2026-05-01-cluster-k-v2-breakthrough.md`.
- `CLAUDE.md:3196` corrected (the "closes 4 of 5 cheats" claim).
- `CLAUDE.md:3311` **left alone — it is true.**

## The generalizable defect

`.cmd.json` sidecars record the truth. Nothing ever diffs a claim against them. A cheap, mechanical guard
exists: **when a finding asserts a cheat is closed, grep its own `.cmd.json` for the flag that closes it.**
Here that is a one-line check that would have caught this on 2026-05-01 — and the same check applied to
the deep-credit arc would have caught `SIGNAL=False` in 2026-07-15.
