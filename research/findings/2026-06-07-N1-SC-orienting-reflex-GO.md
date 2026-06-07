# N1 perceptual cold-start broken BIOLOGICALLY — an innate superior-colliculus orienting reflex (image-sourced, NO coordinates) navigates as well as the coordinate heuristic, 3/3 seeds (6-seed confirming). The deep-research "wrong-pathway" diagnosis is validated; the agent orients to the goal from VISION. — 2026-06-07

**Status:** **6-SEED GO + grid-32 PASS** (grid-8: A/C 0.14–0.23 every seed, 6/6 navigate, mean A 4.49 ≈ cheat 4.55; grid-32 seed-42: A **4.10** vs floor **121.13**, A/C **0.03** — holds at 4× scale). **N1 perception cheat BIOLOGIZED (Rank 1).** NO `sim/` edit (protected set byte-empty); additive default-off flag; 7/7 helper unit tests. This is the **Rank-1 de-risk** from `2026-06-07-perceptual-bootstrap-deep-research.md`; the durable LEARNED circuit is **Rank 2 (owner-approved, building next)**.

## The one-line result

The navigation perceptual cold-start — the agent senses the goal's rough direction but cannot localize it, a boundary that resisted reward-bootstrap (`-gauge-BLOCKED`), a fixed critical-period scaffold (`-scaffold-TRACTABLE`, seed-fragile), and adaptive activity-gated weaning (`-adaptive-wean-...-bank`, 1/3) — is broken by giving the agent the **right pathway**. An innate superior-colliculus orienting reflex, reading the goal's retinal direction from the **rendered image alone (no coordinates)**, navigates **as well as the coordinate heuristic-teacher** and ~5–7× below the no-perception floor, on all **6 seeds**. The deep-research diagnosis (the project routed navigation through the position-*invariant* ventral "what"/IT stream, which structurally cannot localize) is validated: supply a "where" signal (collicular retinotopic salience) and the agent navigates from vision.

## The decisive table (grid-8, multi-goal cheat-5 sum-finalQ, LOWER better; inside the N8+N6 biologized back-end)

| seed | A — SC reflex (heuristic OFF, **no coords**) | C — floor (both OFF) | A/C |
|---|---|---|---|
| 42 | 4.21  [per-phase 1.13/0.94/1.02/1.12] | 25.82 | 0.16 |
| 43 | 4.39 | 19.69 | 0.22 |
| 44 | 4.88 | 26.27 | 0.19 |
| 100 | 4.58 | 26.56 | 0.17 |
| 101 | 3.81 | 27.33 | 0.14 |
| 102 | 5.07 | 22.49 | 0.23 |
| **mean A** | **4.49** | — | — |

Reference — **B (heuristic ON, the coordinate cheat), seed 42 = 4.55.** So **A (4.49 mean) ≈ B (4.55)** — the image-only reflex matches (slightly beats) the coordinate cheat — and every seed sits ~5–6× below the floor. The reflex's per-phase profile [~1.0 across all four phases including post-goal-change] is also *cleaner* than the multi-pool heuristic's (which spikes to 1.81 on a goal change), because the reflex drives a single orienting cardinal (no BG arbitration noise).

## The mechanism (biologized, grounded)

- **Innate SC orienting reflex** (`sc_orienting_cardinal_from_image` + `--sc-orienting-reflex` in `g11_bg_runner.py`): the rendered retinotopic image (`sim/visual_cortex.render_gridworld_to_image`) paints the agent as the bright ON blob and the goal as a dimmer ON blob. The reflex reads both blob CENTROIDS from the pixels and pushes the cardinal of the goal's offset from the agent (= the goal's eccentricity on a retina centred on the agent) into `cortex_X` — exactly what the superior colliculus does (a retinotopic salience map → reflexive orienting; Kandel 6e Ch 35; catalog A.07/H.25). Injected upstream of the unchanged N8 (GPi→thal disinhibition) + N6 (spiking accumulate-then-commit) cascade.
- **Why it's the legitimate replacement for the heuristic** (not a cheat): it is the textbook innate orienting scaffold (lamprey-conserved), it is released by the SNr/GPi disinhibition the project already biologized in N8, and it reads from VISION — no `(gx,gy)`. It is the biological version of "the heuristic-as-developmental-teacher" the N1 weaning arc was hand-faking with coordinates.

## Anti-cheat — the reflex reads ONLY the image (the single biggest correctness risk, controlled)

- `sc_orienting_cardinal_from_image(image)` takes the rendered image array as its **only** argument — coordinates are structurally incapable of entering. **7/7 unit tests** against the real render confirm the correct cardinal (E/N/W/S + co-located→None + dominant-axis).
- **Floor control C** (visual cortex on, reflex off, heuristic off) sits at the ~20–26 floor every seed — so A's navigation is the reflex, not seed-general drift.
- **Cheat baseline B** (coordinate heuristic on) = 4.55 — A matches it from vision.
- Gated on the **actual cheat-5 multi-goal nav score** (not a proxy); run inside the N8+N6 biologized back-end; NO `sim/` edit.

## Honest scope (carried forward)

- **Rank-1 is an INNATE reflex, not yet a LEARNED circuit.** It proves the agent CAN orient to the goal from vision (the "no innate teacher" half of the cold-start is solved — biologically). The DURABLE learned fix is **Rank 2**: re-source the existing `enable_learned_perception` plastic sensory→cortex pathway (currently coordinate-driven, `g11_bg_runner.py:~4138`) from the IMAGE salience map, so the LEARNED `where→cortex_X` circuit gets a position-PRESERVING input (the thing the position-invariant `IT→cortex_X` path could not); the reflex teaches it, then weans via `transmission_gate`.
- **Separate residual (N2):** the goal is still PAINTED into the render using its coordinates. The reflex reading WHERE the goal appears on the retina is genuine perception; the render placement is the separate, lesser N2 "goal-render" item, tracked on its own.
- **6-seed GO + grid-32 PASS (firm).** Decisive across all 6 canonical seeds (A/C 0.14–0.23, 6/6 navigate, mean A 4.49 ≈ cheat 4.55) — the standing 6-seed rule is satisfied; grid-32 seed-42 A 4.10 vs floor 121.13 (A/C 0.03) holds at 4× scale. Owner approved Rank 2 (the durable learned read-out) as the next build.

## Net for the nav arc

**N8 ✅ (BG output) + N6 ✅ (decision) + the perceptual front-end now navigating from vision via an innate collicular reflex.** The deepest, most-resistant navigation boundary — the perceptual cold-start that three prior approaches could not break — is broken by the deep-research "right-pathway" reframe (ventral-IT → dorsal/collicular where). This is the second consecutive arc where a deep-research + catalog review was the decisive pivot (now standing practice; CLAUDE.md).

## Production config + artifacts

```
... --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --genuine-thal-disinhibition --genuine-gpi-tonic-pa 1300 --genuine-thal-tonic-pa 750 \
    --readout-source spiking_wta --urgency-max-pa 180 \
    --heuristic-strength 0 --sc-orienting-reflex \
    --grid-size 8 --goal-schedule multi --deterministic --moving-goal
```
- Helper + flag: `g11_bg_runner.py` (`sc_orienting_cardinal_from_image`, `--sc-orienting-reflex` / `--sc-reflex-strength`; additive, default-off, NO `sim/` edit).
- Smoke/multi-seed: `research/findings/raw/_run_sc_reflex_smoke.ps1` + `_run_sc_reflex_multiseed.ps1` + `_run_sc_reflex_6seed.ps1`; analyzers `_sc_reflex_analyze.py` + `_sc_reflex_multiseed_analyze.py` (auto-detects seeds).
- Prior: `2026-06-07-perceptual-bootstrap-deep-research.md` (the diagnosis + Rank-1 recommendation), `2026-06-07-N1-adaptive-wean-multiseed-NEGATIVE-bank.md` (the boundary this resolves), `2026-06-06-N8N6-combined-readout-GO.md` + `2026-06-06-N6-decision-biologized-CONCLUSION.md` (the back-end this composes with).
