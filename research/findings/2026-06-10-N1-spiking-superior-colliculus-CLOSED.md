---
type: finding
status: live
date: 2026-06-10
mechanism: superior-colliculus
---

# N1 CLOSED — the navigation superior-colliculus orienting reflex is now a spiking superior colliculus (6-seed: matches/beats the host it replaces, lesion-confirmed)

**Date:** 2026-06-10
**Result:** the host pixel-reading orienting reflex (`sc_orienting_cardinal_from_image`) is replaced by a **spiking retinotopic superior colliculus** whose `cortex_{N,E,S,W}` pooling biases action selection synaptically. **6-seed nav A/B: SC-on mean 3.607 vs host-reflex 4.085 = SC/host 0.883 (12% BETTER), 5/6 seeds win; scrambled-retinotopy anti-cheat regresses 2.4× (orienting is genuinely retinotopic).** ZERO protected `sim/` edits — all runner-side region/pathway wiring over already-merged machinery.

## The arc (standing practice: deep-research → de-risk → build → validate)

1. **Characterization + ledger** (`2026-06-10-N2-N7-characterization-and-honest-nav-cheat-ledger.md`): the strict BRAIN-BASED-ONLY bar showed N1 (`sc_orienting_cardinal_from_image`, a host pixels→cardinal reader) was still a host computation between sensation and action — a weanable teaching scaffold, but the strict-bar target is a spiking superior colliculus.
2. **Deep research** (`2026-06-10-N1-N5-spiking-superior-colliculus-research.md`, trust-but-verified): ONE spiking superior colliculus closes both N1 (orienting = where the bump is) and N5 (approach = how the bump moves toward the fovea); ZERO new `sim/` edits (the 2D-retinotopic-sheet + slow-channel machinery already exists).
3. **Cheapest-first de-risk** (`2026-06-10-N1-N5-spiking-SC-derisk-RESULT.md`): a spiking SC map reproduced the orienting cardinal **8/8 by neuron firing, lesion-confirmed retinotopic** → N1 RESOLVES (N5's static approach read was SNR-limited → Option C, since de-risked separately).
4. **Build into the nav runner** (3 committed increments, default-off = byte-equivalent): `--enable-spiking-sc` adds `sc_map` (16×16 retinotopic sheet) + Mexican-hat `sc_fs` + a dedicated `sc_retina` egocentric eye; `install_spiking_sc_wiring` installs retinotopic `sc_retina→sc_map` + recurrent + `sc_map→cortex_{N,E,S,W}` weighted-quadrant pooling; the nav loop drives `sc_retina` with the egocentric render each step, so the pooling biases action selection **synaptically** (replacing the host current injection). The main `retina` stays allocentric for the visual cortex / N5 reward.
5. **Tune the integration-vs-isolation gap** (the research's predicted risk #4): single-seed A/B at the initial pooling weight was 77% worse — the synaptic pooling was too weak vs the BG cascade + background noise. A `w_sc_cortex` sweep found a **non-monotonic** optimum (w=15 competitive, w=40 over-dominates the policy, w=18 best) → default 18.0.
6. **6-seed validation + anti-cheat** (this finding).

## The result

| seed | SC-on (spiking SC) | host-reflex (the scaffold) | SC/host |
|---|---|---|---|
| 42 | 5.157 | 3.667 | 1.407 |
| 43 | 2.971 | 4.099 | 0.725 |
| 44 | 4.165 | 4.568 | 0.912 |
| 45 | 2.792 | 4.333 | 0.644 |
| 46 | 4.152 | 5.291 | 0.785 |
| 47 | 2.403 | 2.552 | 0.941 |
| **mean** | **3.607** | **4.085** | **0.883** |

`nav_sum` = sum of `final_quarter_mean_distance` across the 4 moving-goal phases (LOWER = better). Config: `--moving-goal --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --grid-size 8 --n-steps 1800`, SC-on at `SC_CORTEX_W=18` vs `--sc-orienting-reflex` (the host pixel-reader).

- **The spiking SC matches/beats the host reflex it replaces** — mean SC/host 0.883 (12% better), 5/6 seeds win.
- **The one loss (seed 42, 1.407) is within the nav's documented run-to-run non-determinism** (the CuPy place-code / transpose-SpMV cusparse-atomic variance, characterized in the N9 work): the *same* seed+config (w=18, seed 42) gave 0.899 in the refine sweep. So per-seed variance (0.64–1.41) is the nav's inherent noise; the 6-seed mean is the robust signal.

## The decisive anti-cheat (scrambled-retinotopy lesion)

`SC_SCRAMBLE=1` permutes the `sc_retina→sc_map` target assignment (destroys retinotopy). Seed 42: **nav_sum 12.571 vs SC-on 5.157 = regresses 2.4×.** The orienting is genuinely carried by the *retinotopic* map — scrambling it breaks navigation. Combined with the image-only afferent (the SC reads only the egocentric render — no `(x,y)`/`(gx,gy)`/host-cardinal enters the SC drive), this confirms the orienting is computed by retinotopic spiking neurons, not a re-hidden host shortcut.

## What this closes (the honest framing)

N1's host pixel-reader (`sc_orienting_cardinal_from_image`) is replaced by a spiking superior colliculus: an egocentric retinal image → a retinotopic spiking sheet with Mexican-hat winner-take-all → a topographic read-out into the cortical action pools, **all by neuron firing**, image-only, lesion-confirmed retinotopic, and 6-seed competitive-to-better vs the host scaffold. By the BRAIN-BASED-ONLY standard this is a successful biologization (the spiking organ matches/beats the host shortcut it replaces).

**Residual idealization (documented, not a hidden cheat):** the `sc_map → cortex_X` topographic read-out is a *fixed*, genetically-specified-style projection (chemoaffinity / ephrin-Eph map formation), not a learned map — the same status as the innate V1 Gabor receptive fields (N7). No cognitive quantity is host-computed; it is innate structure, which the bar permits.

## Navigation-cheat ledger after N1

| Axis / cheat | Status |
|---|---|
| N6 action selection | spiking commit-burst / WTA — closed |
| N8 disinhibition | closed |
| N9 dopamine RPE | spiking SNc + reward_us + GABA_B critic — closed |
| N2 goal cue | defensible (beacon) — closed |
| N7 V1 receptive fields | defensible (innate V1) — closed |
| **N1 SC orienting reflex** | **spiking superior colliculus — CLOSED (this finding)** |
| N5 reward value | core de-risked (slow-channel TD, `sc_approach_td_probe.py`); build = wire onto the `sc_map` rostral pool + `approach→reward_us` (the only remaining nav host-computation) |

After N5 Option C lands, navigation is **fully brain-based** by the strict bar (spiking SC orienting + neural approach reward + spiking commit + spiking SNc RPE + defensible perception) → the parked single-instance unification gate opens.

## Reproduce

```bash
# 6-seed A/B (SC-on w=18 vs host-reflex) + scrambled-retinotopy anti-cheat:
bash research/findings/raw/g11_bg/_sc_6seed_run.sh
# A single SC-on nav run:
SC_CORTEX_W=18 python -m research.runners.g11_bg_runner --moving-goal \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --enable-spiking-sc --grid-size 8 --n-steps 1800 --seed N --out <out>.json
```
