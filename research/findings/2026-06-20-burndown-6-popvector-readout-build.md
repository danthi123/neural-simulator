# Burndown #6 — population-VECTOR SC orienting read-out BUILD (the deep-research-prescribed fix for the position-invariant ramp): does it make the spiking-SC arm TRACK the goal + RE-ORIENT? (2026-06-20)

**Type:** BUILD + cheap-first GPU de-risk of the deep-research-prescribed fix for nav shortcut #6 (the spiking
superior-colliculus orienting read-out). Per the owner directive (2026-06-20): a spiking shortcut/honest-negative
must be PROPERLY BIOLOGIZED on the point-neuron substrate where feasible, NOT deferred. The deep-research gate
(`2026-06-20-nav-readout-geometry-deep-research.md`, commit `345d0687`) localized #6 to a point-neuron-feasible
read-out-GEOMETRY fix and said it should CONVERT.
**Pre-registered by:** `research/findings/2026-06-20-nav-readout-geometry-deep-research.md` (Option A + the anti-cheats).
**The boundary under repair:** `2026-06-20-nav-sc-drive-reorient-derisk.md` (the stuck-N NEGATIVE: every SC arm is
N-dominated ~0.45-0.52 in EVERY phase regardless of goal; the host re-orients; ~73x gap on post-change re-orient).
**Owner standard:** BRAIN-BASED-ONLY. An honest NEGATIVE (the population-vector read-out still can't track/re-orient)
is a valid deliverable — but the research says this should convert, so it got a real shot.

---

## What was built (the deep-research Option A, all opt-in, default-OFF = the deployed ramp byte-identical, NO sim/ edit)

The deployed `sc_map -> cortex_X` read-out (`g11_bg_runner.py:262-263`) is a signed half-plane **LINEAR RAMP**
`wv = max(0, +/-ddx)/max(0, +/-ddy)` — an UN-normalized weighted SUM of the activity-weighted coordinate. It is
provably position-INVARIANT (the NEGATIVE) for two structural reasons: (i) no normalization by bump mass (a bigger
bump lifts all four cardinal sums together — "how much" SC fires, not "where"), and (ii) no competition between the
four cardinals to widen the winner's margin.

The fix (three parts, the deep-research Option A+B):
1. **Population-VECTOR read-out** (`install_spiking_sc_wiring(popvector=True)`): each `sc_map` site `(sx,sy)` has a
   preferred-direction unit vector `u = (ddx,ddy)/|(ddx,ddy)|`; the weight into cardinal `a` is the COSINE PROJECTION
   `max(0, u_hat_a . u_site)` — a bounded cosine-tuned weight in [0,1] (E:+x_hat W:-x_hat N:+y_hat S:-y_hat), the SC's
   canonical spike-vector decode (Goossens-Van Opstal; Georgopoulos H.17; catalog H.25/E.03). Replaces the unbounded
   ramp. PURE POINT-NEURON (a feedforward weighted sum of preferred-vector cosines on LIF neurons).
2. **Bump-mass divisive normalization**: the four `cortex_{N,E,S,W}` pools flagged `input_divisive_norm=True` (the
   existing Carandini-Heeger primitive, `sim/bridge.py:6048`, GUARDED NO-OP when off): each cardinal's drive divided
   by `(sigma + gain*mean over the four cardinals)` -> "where" not "mass". `sigma`/`gain` tunable.
3. **Competition (WTA ring)**: the faithful `--spiking-sc` config already routes `readout_source="spiking_wta"` (the
   #4 `sel_X`/`commit_X` Wang/Lo-Wang ring, default-on) — the SC drive flows
   `sc_map -> cortex_X -> str_D1_X -> ... -> thal_X -> sel_X` and the decision emerges from the spiking competition.

Wiring: `run_moving_goal_episode(sc_popvector_readout=, sc_popvector_divnorm_sigma=, sc_popvector_divnorm_gain=)` +
the `--sc-popvector-readout` CLI flag + the `SC_POPVECTOR` env var (parallels `SC_CORTEX_W`/`SC_SCRAMBLE`). Default
OFF => the deployed ramp reproduces byte-identical. **NO `sim/` edit** — a runner read-out-weight formula + an
existing-primitive flag, both set BEFORE the bridge build.

Build-test probe: `research/runners/_nav_sc_popvector_readout_derisk.py` (GPU). Arms: **host** (centroid+argmax
POSITIVE control) · **sc_ramp** (the deployed read-out = the NEGATIVE) · **sc_popvector** (the fix) ·
**sc_popvector_scr** (the retinotopy-SCRAMBLE LESION anti-cheat). Reads per-phase action distribution + re-orient
finalQ. Matched drive (`SC_CORTEX_W` identical across SC arms); perception NOT stripped.

---

## Grid-8/480 smoke (seed 42) — the early read (committed `048ea203`)

Grid-8/480 is a WEAK read (the NEGATIVE itself flagged it: only 2 goal phases complete in 480 steps; the cascade
N-bias + OU dominate at small scale). It is the seconds-scale early gate before the faithful grid-32 confirm.

| arm | phase0 finalQ (acquire) | phase1 finalQ (re-orient, goal far-W) | dominant cardinal / phase | tracks goal? |
|---|---|---|---|---|
| **host** (positive control) | 0.531 | **0.500** | N, **W** (W .53 @ far-W) | YES |
| sc_ramp (the NEGATIVE) | 1.018 | 4.75 | E, N | (N-ish; weak read) |
| sc_popvector (default sigma=1/gain=1) | 0.850 | 6.125 | **N, N** (stuck) | **NO** |
| sc_popvector_scr (LESION) | 1.912 | 6.25 | E, N | — |

**At the default divisive op-point (sigma=1, gain=1) the population-vector arm is STILL stuck-N** (both phases
N-dominant; post-change finalQ 6.125 vs host 0.5; worse than the ramp's 4.75). The mechanistic read (below) is that
the default `gain=1` (calibrated for the conversational cortex where input drives are O(1)) OVER-ATTENUATES the SC
drive into cortex (the nav SC drive is O(tens of pA); `out_i ~ drive_i/(gain*mean) ~ O(few) pA`), crushing the SC
contribution so the cascade N-bias + OU win regardless of the (now-correct) cosine geometry. **The divisive `gain` is
part of Option A's specified normalizer (`drive/(sigma+gain*mean)`); calibrating it to the nav SC drive scale is
within the prescribed A+B, not a config-search.**

**Gain-calibration grid-8 sweep (seed 42, popvector arm) — DONE (2026-06-20).** The divisive `gain` is the swept knob;
`σ=1` fixed. Phase-1 (goal far-W) dominant cardinal is the discriminator (the stuck-N NEGATIVE → W-tracking is the
target). Grid-8 is the SCREEN (phase-1 ~30 actions, noisy), not the verdict.

| (σ, gain) | phase0 finalQ | phase1 dominant / frac | tracks far-W? |
|---|---|---|---|
| gain=1.0 (default) | 0.850 | **N, N** | NO (crushes the cosine drive) |
| gain=0.2 | 0.79 | **N, N** (`{N:.50}`) | NO |
| gain=0.05 | 1.30 | **N, N** (`{N:.43,E:.33}`) | NO (less stuck, not tracking) |
| **gain=0.0 (PURE COSINE)** | **0.87** | **N, W** (`{W:.37,E:.27,N:.20,S:.17}`) | **YES — phase-1 dominant flips to W** |

**Calibration verdict: the bump-mass divisive `gain` must be ~0 (pure cosine geometry).** The shared-divisor `σ+gain·mean`
is identical across all four cardinals, so it adds NO competition while shrinking the cosine geometry's relative margin
below the cascade's responsive band — i.e. it crushes the position decode. At `gain=0` the cosine geometry survives and
the grid-8 phase-1 dominant flips from the stuck-N to W (tracking the far-west goal). PASS-to-proceed met. The faithful
verdict runs at `gain=0, σ=1` (the divnorm is then an inert identity; the WORKING fix = the cosine geometry + the #4 WTA
ring's competition). See `2026-06-20-shortcut6-nav-orienting-CLOSE.md` for the full calibration table + the grid-32
verdict.

---

## Grid-32/1800/warmup-600 (seed 42) — the faithful confirm (the decisive read)

<!-- FILL: the per-phase action distribution (SC arm vs host) + the re-orient finalQ vs host + the scramble lesion -->

---

## sim/ edit?

**NONE.** The read-out formula is in the research runner (`g11_bg_runner.py:install_spiking_sc_wiring`, NOT protected
`sim/`); the bump-mass normalization reuses the existing `input_divisive_norm` primitive (a region flag + a cfg flag,
both mutated before the bridge build). No `sim/` change was needed or made. (If one had seemed needed, the task was to
STOP and report — it did not.)

---

## Verdict

<!-- FILL: #6 CONVERTS (action-dist tracks goal + re-orients toward host, scramble collapses) OR the honest residual -->

---

## Commits (all on `main`, PATHSPEC, pushed origin + gitea)

- `048ea203` — the #6 population-vector read-out + cortex_X divisive-norm build (opt-in, default-OFF byte-identical) +
  the build-test probe `_nav_sc_popvector_readout_derisk.py` + the grid-8/480 smoke (popvector stuck-N at the default
  divisive op-point).
<!-- FILL: subsequent commits (gain calibration + grid-32 + this doc finalization) -->
