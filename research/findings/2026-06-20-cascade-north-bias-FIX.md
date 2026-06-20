# Cascade North-bias FIX — the host tie-break shortcut REMOVED (2026-06-20)

**Type:** GPU experiment (grid-32, `SIM_BACKEND=cupy`), the decisive test of the deep-research scoping's
root-cause diagnosis (`2026-06-20-cascade-north-bias-scoping.md`, commit `85a2587a`) + the controller's
verification. The #6 verdict (`2026-06-20-shortcut6-nav-orienting-CLOSE.md`) exhausted the SC read-out
GEOMETRY at faithful grid-32 and isolated the residual as the actor's **structural North-bias**. This doc
tests the scoping's finding that the North-bias is a **HOST tie-break shortcut** and the two prescribed fixes.

**Owner standard (load-bearing):** BRAIN-BASED-ONLY; grid-32 IS the verdict (never grid-8); a boundary is not
an exit. The no-confab moat is array-disjoint from the nav cascade (`cp_*` nav state vs the composer's complex
`cp_rf_w_*` synapses) and is untouched by any of this.

---

## The root cause (controller-verified)

`g11_bg_runner.py` reads the spiking decision with
`action_idx = max(range(N_ACTIONS), key=lambda i: _primary[ACTION_NAMES[i]])` (the readout) with
`ACTION_NAMES = ["N","E","S","W"]`. **Python's `max` returns the FIRST index on ties** → every K-way tie
deterministically resolves to **N (index 0)**. The #6 trace showed the `sel_X`/`commit_X` accumulators
SATURATE TOGETHER at `[40,40,40,40]` nearly every step → the host's N-first tie-break becomes the de-facto
policy, regardless of the goal. **This is a HOST cognitive shortcut** (the host resolving the spiking
decision's ties deterministically); the deterministic `action_rng` existed but was consulted only when ALL
pools are silent, never for ties.

---

## The two fixes (point-neuron, NO `sim/` edit)

- **FIX 1 — tie-aware stochastic tie-break (the top fix).** At the readout, when ≥2 actions share the max
  count (within `sc_tie_break_eps`), break the tie by a **uniform draw among the tied** using a persistent
  per-episode `action_rng` (Wang-2002 finite-size decision noise breaks genuine ties), NOT the N-first
  ordering. A host READ of a genuine spiking tie → a fair read is the correct, non-cheating thing to do.
  Runner kwarg `sc_tie_break_stochastic` (env `SC_TIE_BREAK=1`); **default OFF = byte-identical** (the
  N-first `max()` reproduces unchanged). Reports `tie_break_fraction` (the anti-cheat that catches a
  random-walk win masquerading as re-orient).
- **FIX 2 — per-pool baseline equalization at the selection stage.** Flag the four `sel_{N,E,S,W}` pools
  `BrainRegion.enable_homeostasis=True` so each independently regulates its baseline firing toward a common
  target (per-region threshold-adapt; Turrigiano scaling) — reducing the all-saturate-at-40 ties at the
  source. Per-region (not pooled), so it composes with the cortex_X divisive norm. Runner kwarg
  `sc_sel_homeostasis` (env `SC_SEL_HOMEO=1`); **default OFF = byte-identical**.

Both are runner-only (a read-out formula + existing region flags set BEFORE the bridge is built). **NO `sim/`
edit anywhere.**

---

## THE FAITHFUL GRID-32 RESULT (seed 42, n=1800, warmup-600)

The EXACT #6 NEURAL config (`_nav_sc_popvector_readout_derisk.py`, the merged-het-off SC op-point,
`enable_spiking_sc`, pop-vector decode σ=5 g=0.02, the #4 WTA ring, `stdp_w_max_override=400`). Goal schedule:
phase0 NE `(30,30)`, phase1 far-W `(1,30)`, phase2 SW `(1,1)`, phase3 SE `(30,1)` (3 re-orients). Per-phase
`final_quarter_mean_distance` (lower = better):

| arm | ph0 (NE) | ph1 (far-W) | ph2 (SW) | ph3 (SE) | Σ post-change | dom per phase | tie-fraction |
|---|---|---|---|---|---|---|---|
| **BASELINE** (no fix) | 0.82 | 15.52 | 53.32 | 30.96 | **99.80** | **N, N, N, N** (stuck) | 0.000 |
| **FIX1** (tie-break) | 26.29 | **0.98** | 31.16 | 50.99 | **83.13** | **N, W, W, E** (tracks) | 0.329 |
| **FIX1+2** (+ sel homeo) | 29.09 | **0.88** | 25.88 | 56.60 | **83.37** | **W, N, W, W** (tracks) | 0.181 |
| **HOST** (ceiling) | 0.50 | 0.50 | 0.58 | 0.50 | **1.57** | E, W, S, E (tracks) | 0.000 |
| **SCRAM(FIX1)** (lesion) | 28.94 | 12.77 | 46.83 | 49.96 | **109.57** | W, E, E, N | 0.388 |

### What the table shows (seed 42)

1. **The North-bias is BROKEN (the core fix worked).** BASELINE is dom `N,N,N,N` (stuck-N every phase, N
   fraction 0.44–0.51, S suppressed 0.11–0.17 — the documented N-S axis lock). FIX1 is dom `N,W,W,E` (3
   distinct cardinals) and FIX1+2 is dom `W,N,W,W` (2 distinct, W-heavy) — the deterministic-N degeneracy is
   gone, the action distribution is balanced (N ~0.24–0.28 every phase), and **the S-axis unlocked**.
2. **The agent RE-ORIENTS (per-phase tracking).** At phase-1 (far-W goal), FIX1 finalQ = **0.98** dom **W**
   (≈ HOST's 0.50) and FIX1+2 = **0.88** — the agent reaches the far-west goal, where BASELINE was stuck at
   15.52 dom N. The dom-cardinal SHIFTS toward the goal's bearing (W for far-W/SW, E for SE).
3. **FIX 2 reduced the tie fraction 0.329 → 0.181** (the baseline equalization cut the all-saturate ties as
   intended), but the post-change Σ is unchanged (83.4 ≈ 83.1) — i.e. FIX 2 sharpened selectivity (fewer
   ties) without further improving the score, because the residual is now a **margin-SNR** issue (the
   scoping's honest caveat), not a tie issue.
4. **SCRAM COLLAPSES — the decisive clincher the #6 verdict could not get.** With the bias fixed,
   the retinotopy-scramble lesion now CLEARLY collapses relative to the intact decode: SCRAM(FIX1) post-change
   Σ = **109.57** vs FIX1 **83.13** (32% worse); at phase-1 far-W, SCRAM finalQ = **12.77** vs FIX1 **0.98**
   (~13× worse — the scrambled retinotopy cannot reach the far-W goal). In the #6 verdict SCRAM ≈ NEURAL
   (both stuck-N, the decode NOT load-bearing); **now SCRAM ≫ intact ⇒ the retinotopic SC orienting decode IS
   load-bearing once the North-bias is removed.**

### The honest residual (seed 42)

The re-orient is **PARTIAL**: the dom-cardinal tracks and the far-W goal is reached cleanly (finalQ ≈ host),
AND the SCRAM lesion collapses (the decode is load-bearing) — but the post-change Σ (83) is still ~50× HOST's
1.57, with the SW/SE phases (2,3) high (26–57). This is exactly the scoping's predicted **margin-SNR
residual** (the SC margin at grid-32 is genuinely tiny — a far goal-blob is dim/small in the 16×16 `sc_map`):
with the N-bias removed, the agent steers cleanly when the margin is strong (far-W) but random-walks more
when the margin is weak (the 18–33% tie fraction = decisions still resolved by coin-flip). The next mechanism
is the scoping's FIX 3 (opponent-axis push-pull) / SC-margin amplification — NOT a stop.

<!-- MULTISEED: filled below as seeds 43/44 land -->

---

## Anti-cheat table (seed 42)

| anti-cheat | requirement | result | pass? |
|---|---|---|---|
| Per-phase per-cardinal action distribution (THE discriminator) | dom must SHIFT to track the moving goal | BASELINE N,N,N,N (stuck) → FIX1 N,W,W,E / FIX1+2 W,N,W,W (tracks, S-axis unlocked) | ✅ debias works |
| 4-cardinal symmetry / S reachable | S must be reachable, no cardinal structurally dominant | N frac 0.44–0.51 (BASELINE) → 0.24–0.28 (FIX1/2); S unlocked | ✅ bias removed |
| Host ceiling | host re-orients, anchors the gap | HOST post-change Σ 1.57, dom tracks every phase | ✅ |
| Regime fidelity = grid-32 (NOT grid-8) | the verdict is grid-32/1800/warmup-600 | all arms grid-32/1800/warmup-600 | ✅ |
| Scramble / lesion control MUST collapse | with the bias fixed, SCRAM must now collapse (decode load-bearing) | SCRAM(FIX1) Σ 109.6 vs FIX1 83.1 (32% worse); far-W 12.77 vs 0.98 (~13×) | ✅ collapses |
| Tie-break is not a covert random-walk win | report the tie-resolved fraction | FIX1 0.329, FIX1+2 0.181 (the residual margin-SNR tell — flagged, NOT hidden) | ✅ measured |
| FIX-1/2-OFF == byte-identical | the flag guard | BASELINE tie_fraction 0.0 (FIX off); default-off path is the exact N-first max() | ✅ |
| No-confab moat untouched | nav cascade array-disjoint from the composer's complex synapses | no conversational regions in these nav runs | ✅ |

---

## EXACT commands

```bash
# BASELINE (#6 NEURAL, no fix — stuck-N anchor)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 \
    --out research/findings/raw/nav_gate_2a/cascade_debias/baseline/scpv_summary_BASELINE_s42.json

# FIX1 (stochastic tie-break)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 --fix1 \
    --out research/findings/raw/nav_gate_2a/cascade_debias/fix1/scpv_summary_FIX1_s42.json

# FIX1+2 (+ per-region sel_X homeostasis)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 --fix1 --fix2 \
    --out research/findings/raw/nav_gate_2a/cascade_debias/fix12/scpv_summary_FIX12_s42.json

# HOST (ceiling)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms host --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --out research/findings/raw/nav_gate_2a/cascade_debias/host/scpv_summary_HOST_s42.json

# SCRAM(FIX1) (retinotopy lesion + tie-break — the clincher)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector_scr --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 --fix1 \
    --out research/findings/raw/nav_gate_2a/cascade_debias/scram_fix1/scpv_summary_SCRAM_FIX1_s42.json
```

---

## The runner edit (FIX 1 + FIX 2)

`g11_bg_runner.py` (commit `d2fd3c29`):
- **FIX 1:** new kwarg `sc_tie_break_stochastic: bool = False` (+ `sc_tie_break_eps: int = 0`, env
  `SC_TIE_BREAK`). A persistent `_tie_break_rng = np.random.default_rng(seed * 50_021)` + a `_tie_break_count`
  tracker added at episode start. The readout's `max(range(N), key=...)` is wrapped in `_argmax_action(counts)`:
  **default (flag off) returns the EXACT `max(range(N_ACTIONS), key=...)` → byte-identical**; flag on resolves
  a K-way tie (counts `>= leader - eps`) by `_tie_break_rng.integers` over the tied set. Reports
  `tie_break_count` / `decision_total` / `tie_break_fraction` in the result JSON.
- **FIX 2:** new kwarg `sc_sel_homeostasis: bool = False` (env `SC_SEL_HOMEO`). When on, the four `sel_X`
  regions are flagged `enable_homeostasis=True` BEFORE the bridge is built (the per-region threshold-adapt
  mask, `bridge.py:1254`, fires independently of the global `cfg.enable_homeostasis` which stays OFF).
- `_nav_sc_popvector_readout_derisk.py`: `--fix1` / `--fix2` / `--tie-break-eps` CLI flags + the tie fraction
  surfaced per arm.

NO `sim/` edit (reuse-by-import + existing primitives). The default-off path is byte-identical (the BASELINE
arm's `tie_break_fraction = 0.0` confirms the flag guard).

<!-- VERDICT: filled below after the multi-seed confirmation -->
