# N1 (action heuristic) removal is PERCEPTION-BLOCKED — BOUNDARY, multi-seed (grid-8); the genuine visual cortex does NOT yet replace the hand-coded coordinate heuristic — 2026-06-06

**Status:** BOUNDARY. Gated on the cheat-5 multi-goal navigation score (sum of per-phase final-quarter mean
distance; LOWER is better), with the heuristic-ON N8/N6 production config as the control, **multi-seed (42/43) at
grid-8**. NO `sim/` edits, NO new flags required (the removal uses the existing `--heuristic-strength 0`). This is the
honest verdict on the audit's hardest target (N1, "removal requires perception quality first").

## The one-line result

With the action heuristic **genuinely removed** (`--heuristic-strength 0`), navigation collapses from the
heuristic-ON control's **~4.1** to **18.7 / 21.7** (seeds 42/43) — i.e. all the way down to the **bare-cascade floor
(22.4, no perception at all)**. The genuine visual cortex (goal rendered into pixels → retina→V1→V2→IT→cortex_X,
the only coordinate-FREE perception path) contributes only a marginal, seed-fragile improvement over that floor and
does NOT support realistic navigation. **N1 is not removable with the perception currently available; it is blocked
on perception quality**, exactly as documented (`2026-05-01-tier0-no-heuristic-perception-bottleneck.md`).

## The decisive table (grid-8, cheat-5 multi-goal sum-finalQ, LOWER is better; full 1800-step / 4-phase protocol, goal changes at 450/900/1350)

| condition | heuristic | perception | seed 42 | seed 43 | per-phase finalQ (s42) | n_at_goal (s42) | reading |
|---|---|---|---|---|---|---|---|
| N8/N6 production (control) | **single-pool ON** | — | **4.08** | **3.96** | [0.6, 0.5, 1.42, 1.55]* | ~575/1800* | the "before" (documented `N6-decision-CONCLUSION`) |
| heuristic OFF + visual cortex | **OFF (strength 0)** | genuine V1→IT | **18.70** | **21.67** | [4.64, 1.76, 5.63, 6.67] | 17/1800 (0.9%) | **BOUNDARY — collapses to ~floor** |
| heuristic OFF + NO perception | **OFF (strength 0)** | none (bare cascade) | **22.39** | — | [5.08, 3.49, 5.79, 8.04] | 15/1800 (0.8%) | the floor (≈ random-walk-level) |
| ~~multi-pool heuristic + visual cortex~~ | ~~multi-pool ON~~ | genuine V1→IT | ~~3.58~~ | — | ~~[0.93,0.76,0.81,1.08]~~ | ~~575/1800~~ | **INVALID — heuristic still on (see gotcha)** |

\*control per-phase from `2026-06-06-N6-decision-biologized-CONCLUSION.md` (same config); n_at_goal is the
documented ~32%.

The visual-cortex test (18.7/21.7) sits within ~17%/noise of the no-perception floor (22.4) and **~5× worse** than
the heuristic-ON control (4.0). Every phase fails to reach/hold the goal (per-phase 4.6/1.8/5.6/6.7 — only the first
goal change, phase 1, gets a partial dip to 1.76; phases 2–3 drift to 5.6–6.7 cells away). The agent does NOT
re-acquire goals realistically without the heuristic.

## Why — the heuristic is load-bearing, and the visual cortex's action pathway cold-starts

1. **The heuristic is the clean perceptual signal.** `g11_bg_runner.py:3949-3973` compares raw `(gx,gy,x,y)` on the
   host and injects 800 pA into the Manhattan-reducing cortex pool(s). That is a noise-free binary direction signal,
   grid-size-independent. The BG cascade (genuinely on-substrate) then selects perfectly — which is why the
   heuristic-ON control is 4.0 with near-zero variance.
2. **The only coordinate-free perception path is the visual cortex** (`--enable-visual-cortex`): the goal is rendered
   into the retina image (`render_gridworld_to_image(..., goal_pos=...)`, `g11_bg_runner.py:4230`), propagated
   retina→V1_simple→V1_complex→V2→IT, and IT→cortex_{N,E,S,W} drives action. But that IT→cortex pathway is
   **zero-init, STDP-grown only after a 600-step warmup** (`g11_bg_runner.py:1896-1904`, weight_mean=0.0, gate
   `visual_cortex_action` opened at step 600). With the heuristic gone there is no teacher during warmup, so STDP has
   nothing aligned to amplify — the classic cold-start failure seen across this project (learned-perception
   cold-start, the v16 zero-init compose pathway, etc.). The pathway never grows a strong, correctly-oriented signal
   within the 1800-step run → navigation ≈ bare cascade.
3. **This reproduces the documented bottleneck exactly.** `2026-05-01-tier0-no-heuristic-perception-bottleneck.md`
   found perception-only navigation at 16×16 = 15.47 (~4× worse than heuristic-on). Here at grid-8 it is 18.7/21.7
   (~5× worse). Same wall, multi-seed, now confirmed to persist even **with the N8/N6 biology fixes applied** (genuine
   GPi→thal disinhibition + spiking accumulate-then-commit readout). N8/N6 fixed the BG OUTPUT stage; they cannot
   manufacture a perceptual INPUT signal the visual cortex hasn't learned.

## ⚠️ Gotcha discovered (and corrected) — `--heuristic-single-pool` is NOT how you remove the heuristic

`heuristic_strength` **defaults to 1.0** (`g11_bg_runner.py:2650`). Dropping `--heuristic-single-pool` does NOT
disable the heuristic — it falls through to the `else: h_strength = heuristic_strength` branch
(`g11_bg_runner.py:3945-3946`) and the **multi-pool** heuristic (lines 3962-3973) drives every Manhattan-reducing
cortex pool. The first run here did exactly that and scored a misleadingly-good 3.58 (the heuristic was still
navigating). The heuristic is only actually OFF via `--heuristic-strength 0` (or Config A's
`--cue-reflex --cue-reflex-replaces-heuristic`, which sets h_strength=0). This is the genuine N1-removal switch and is
what the 18.7/21.7 BOUNDARY numbers use.

## On Config A's cue-reflex path — it is itself a coordinate cheat (N10–N12), not a true N1 removal

The audit's "Config A removes the heuristic" (`--cue-reflex-replaces-heuristic`, 4.08 6-seed) is real, but the
cue-reflex it substitutes (`g11_bg_runner.py:3984-4010`) **still reads raw `gx - x`, `gy - y`** to compute a bearing,
then spreads it over hand-coded directional beacon sensors. That is cheats N10 (hand-coded angular sensors) + N11
(beacon = 1/(1+Manhattan d)) + N12 (landmark geometry) — flagged cheats in the same audit. So Config A does not
make the agent coordinate-free; it trades the direct-coordinate heuristic for a coordinate-DERIVED bearing reflex.
A genuine N1 removal must come from the visual cortex (or another non-coordinate sensor), which is what fails here.
This test therefore did NOT spend a run re-confirming the cue-reflex path (it would "pass" only by using N10–N12).

## Honest scope / verdict

- **N1 (action heuristic) is NOT removable now: BOUNDARY.** With the heuristic genuinely off, no available perception
  path delivers realistic navigation. The genuine visual cortex — the only coordinate-free option — collapses to the
  bare-cascade floor, multi-seed.
- **The precise blocker is perception quality, specifically the cold-start of the IT→cortex_X action pathway.** It is
  zero-init and STDP-only; with the heuristic removed there is no teacher to bootstrap it during the critical-period
  warmup. (Closest-getting path: visual cortex at seed 42 = 18.7, a ~17% improvement over the 22.4 floor — a weak,
  seed-fragile partial signal, not navigation.)
- **N8/N6 do not change this.** The BG output/readout biology fixes are applied in every run above; they correctly
  select once a clean signal exists (the heuristic-ON control is 4.0), but they cannot substitute for the missing
  perceptual input.
- **This matches the audit's own prediction** ("N1 … removal requires perception quality first … Hardest; do after
  perception (N2/N7) improves") and the 2026-05-01 tier-0 finding. N1 is correctly ordered LAST among the perception
  cheats.

## What would lift the boundary (future work, in rough order)

1. **Teacher / informed-init for IT→cortex_X.** The heuristic's collapse is a cold-start, not a capacity limit.
   Options: (a) a critical-period phase where the heuristic teaches IT→cortex_X via STDP, then the heuristic is
   decayed to 0 (`--heuristic-decay-after-step` already exists — but that is still "trained by a cheat"); (b) an
   informed directional prior on IT→cortex_X analogous to the learned-perception `informed_init` block
   (`g11_bg_runner.py:3369-3412`) — a labeled developmental scaffold, not a runtime cheat.
2. **Richer / scaled visual cortex** (N2 + N7 first): the goal must be a *salient learned stimulus* the IT layer can
   reliably detect (currently it is one painted pixel-blob, N2), and V1 Gabor pre-init (N7) is a scaffold. Improving
   perception SNR is the documented prerequisite the audit names.
3. **Bigger retina / place-cell scaling** for grid-size-independent resolution (documented as necessary at ≥16×16).

So N1 stays open and is gated on N2/N7 (perception) being converted first — which is exactly where the conversion
plan already places it.

## Artifacts

- `research/runners/g11_bg_runner.py` — NO change. Removal uses existing `--heuristic-strength 0`; the heuristic code
  is `:3949-3973`, the cue-reflex (coordinate-derived) is `:3984-4010`, the visual-cortex IT→cortex_X (zero-init) is
  `:1896-1904` / gate-open `:3659-3671`, the retina render is `:4230-4239`.
- `research/findings/raw/_n1_vc_heur0_g8_seed42.json` + `.log` (18.70), `_n1_vc_heur0_g8_seed43.json` (21.67) — the
  BOUNDARY test (heuristic off + visual cortex).
- `research/findings/raw/_n1_noheur_noperc_g8_seed42.json` (22.39) — the bare-cascade floor.
- `research/findings/raw/_n1_vc_noheur_g8_seed42.json` (3.58) — INVALID (multi-pool heuristic still on; retained as
  the gotcha evidence).
- Control (documented, same N8/N6 config + heuristic ON): `2026-06-06-N6-decision-biologized-CONCLUSION.md`
  (4.08/3.96/6.10 grid-8).

## Cross-references

- `2026-06-06-navigation-cheat-audit-and-conversion-plan.md` (N1 = row 1; "removal requires perception quality first").
- `2026-05-01-tier0-no-heuristic-perception-bottleneck.md` (the original perception-bottleneck finding this reproduces).
- `2026-06-06-N8N6-combined-readout-GO.md` + `2026-06-06-N6-decision-biologized-CONCLUSION.md` (the N8/N6 fixes applied
  in every run here).

## Net

N1 (the action heuristic) is the most load-bearing navigation cheat and it is **NOT removable now**. With the
heuristic genuinely off, the only coordinate-free perception (the genuine visual cortex) collapses navigation to the
bare-cascade floor, multi-seed (18.7/21.7 vs the heuristic-on 4.0). The blocker is precisely the cold-start of the
zero-init IT→cortex_X action pathway — a perception-quality problem the N8/N6 BG-output fixes cannot solve. N1 stays
gated behind the perception cheats (N2 goal-in-image, N7 Gabor pre-init), exactly as the conversion plan ordered it.
