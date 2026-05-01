# 2026-05-01 — Tier 0 honest test: perception arc breaks at 16×16

**Context:** G v2.5 (NMDA on PFC + cortex_X + motor_X) achieved
2.00 ± 0.00 on cheat-5 multi-goal det at 8×8, 16×16, and 24×24 — but
all three used `--heuristic-single-pool`, which directly drives
cortex_X based on `gx > x` etc. The heuristic is a hand-coded
high-bandwidth perception cheat. Did the cascade really learn anything,
or was the heuristic doing all the work?

This Tier 0 test drops the heuristic and uses only the biology-grounded
perception arc (beacon + landmark + learned-perception + place/goal cells
+ cue reflex + sensed reward) at 16×16.

## Headline

**16×16 perception arc + G v2.5 NMDA: 15.47 ± 7.06 (n=3)** — about
**4× worse** than the 8×8 perception-arc baseline (4.08 ± 0.49) and
**8× worse** than the heuristic-on G v2.5 result (2.00).

| Seed | finalQ_0 | finalQ_1 | finalQ_2 | finalQ_3 | Sum | Mean dist | n at goal |
|---|---|---|---|---|---|---|---|
| 42 | 3.55 | 2.12 | 3.07 | 3.66 | 12.40 | 3.83 | 78/1800 (4.3%) |
| 43 | 2.41 | 2.12 | **11.89** | 7.13 | **23.55** | 8.35 | 51/1800 (2.8%) |
| 44 | 2.24 | 4.09 | 1.93 | 2.21 | 10.47 | 3.94 | 84/1800 (4.7%) |
| **mean** | | | | | **15.47** | **5.37** | **3.9%** |
| **stdev** | | | | | **7.06** | **2.58** | |

The high variance (std/mean = 0.46) reveals fragile dynamics: seeds 42
and 44 manage adequate goal-finding (~10-12) while seed 43 catastrophically
loses Phase 2 (gets stuck >11 cells from goal). Heuristic-on G v2.5 had
**zero variance** across seeds — the cascade was perfectly deterministic
once the heuristic delivered a clean signal.

## Comparison

| Config | Grid | Sum | Std | n_at_goal | Notes |
|---|---|---|---|---|---|
| Heuristic + G v2.5 | 8×8 | 2.00 | 0.00 | ~49% | flagship |
| Heuristic + G v2.5 | 16×16 | 2.00 | 0.00 | ~49% | this session |
| Heuristic + G v2.5 | 24×24 | 2.00 | 0.00 | ~47% | this session |
| **Perception arc (no NMDA, 2026-04-27)** | **8×8** | **4.08** | **0.49** | — | biology-grounded baseline |
| **Perception arc + G v2.5** | **16×16** | **15.47** | **7.06** | **3.9%** | **THIS — bottleneck** |

NMDA does NOT save the perception arc at scale.

## Diagnosis

The contrast is clarifying:
- At 8×8, perception arc reaches 4.08 — adequate even without NMDA.
- At 16×16, perception arc collapses to 15.47 — even with NMDA stabilization.

Why does heuristic + G v2.5 still work at 16×16 (2.00) while perception
arc + G v2.5 fails (15.47)? Two factors:

1. **Heuristic signal-to-noise** is independent of grid size — `gx > x`
   gives a clean binary direction signal. Perception arc signal-to-noise
   degrades as the goal moves further away (beacon falloff = 1/(1+d),
   so at distance 14 the beacon intensity is 7% of maximum).

2. **Spatial resolution** — beacon's 8 directional sensors give ~22.5°
   bearing resolution. On a 16×16 grid, adjacent cells differ by less
   than the bearing resolution at distance >5. Place cells were tuned
   for 8×8 (64 cells over 64 positions); at 16×16 (256 positions, same
   64 place cells) each place cell's tuning width is 4× too wide.

NMDA stabilizes goal representation IF the cascade has a clean signal
to lock onto. Heuristic provides clean. Perception arc at 16×16 doesn't.

## Implication: Cluster K v2 becomes critical

The visual cortex hierarchy (Hubel-Wiesel 1962, Felleman & Van Essen
1991) is the biology-correct way to provide *rich* perception that
scales with grid size:

- **Resolution** scales with retina pixels, not arbitrary sensor count.
  A 32×32 retina can distinguish all 16×16 cells in principle (and
  even 24×24). A 64×64 retina would scale further.
- **Hierarchical pooling** (V1 → V2 → IT) builds invariances. Real
  vision generalizes across scale, position, lighting. Beacon doesn't.
- **Learned features** — STDP refines V1 weights based on the actual
  visual statistics of the gridworld, rather than relying on hand-coded
  Gaussian-tuned beacon directions.

Cluster K v1 (committed this session) scaffolded the regions and the
env-loop image rendering, but does NOT yet wire IT → cortex_X. v2 is
now the highest-priority architectural work.

## Caveats

1. **n=3** — should run 6 seeds for definitive statistics, but the
   pattern is already clear (4× gap to 8×8 baseline, 8× gap to
   heuristic-on result).
2. **Same place-cell count (64) at 16×16** — could test whether
   scaling place cells to 256 helps. Likely necessary but probably
   insufficient — bearing-only beacon is fundamentally
   resolution-limited.
3. **Curriculum disabled** — the original 8×8 perception-arc result
   (4.08 ± 0.49) used `--curriculum --curriculum-warmup-steps 600`.
   Adding curriculum to this 16×16 test is a follow-up.
4. **No --learned-perception-informed-init** — the 8×8 baseline used
   informed init; without it, learned perception is cold-started.
   May explain part of the 16×16 degradation. Should re-test with
   `--enable-learned-perception-informed-init`.

## Files

- Stress runs: `research/findings/raw/g11_bg/no_heuristic_16x16_seed{42,43,44}.json`
- 8×8 perception arc baseline: `2026-04-27-perception-arc-COMPLETE.md`
- G v2.5 grid scaling (heuristic): `2026-05-01-cluster-g-grid-scaling.md`
- Cluster K v1 scaffold: `2026-05-01-cluster-g-nmda-breakthrough.md` (visual cortex section)

## Recommended next steps (priority order)

### 1. Quick wins (1-2 hours): retest with curriculum + informed init

Re-run Tier 0 with `--curriculum --curriculum-warmup-steps 600
--enable-learned-perception-informed-init`. If this brings the 16×16
result down to ~6-8, the architecture is salvageable with hyperparam
tuning. If still >12, the perception bottleneck is fundamental and
v2 work is required.

### 2. Tier 1 — Cluster K v2 (multi-day, high payoff)

Add the bridge `set_pathway_weights()` API + Gabor pre-init helper +
gated IT → cortex_X pathway. Run a v2 stress test on 16×16. If V1 →
V2 → IT delivers a richer signal than beacon, the perception arc
becomes 1) bearable at 16×16, 2) explicitly biology-grounded, 3)
generalization-ready (could switch from gridworld to natural images).

### 3. Place-cell scaling experiment (1 day)

Increase `--n-hippocampus-per-layer` from 64 → 256 to match grid_size².
Tests whether finer place-cell tuning alone helps. Cheap to try.

### 4. Bigger retina experiment (when K v2 is online)

Once IT → cortex_X works, test whether a 64×64 retina + larger V1/V2/IT
delivers further scaling. This is the multi-modal direction the user
mentioned (richer perception → emergent behavior).
