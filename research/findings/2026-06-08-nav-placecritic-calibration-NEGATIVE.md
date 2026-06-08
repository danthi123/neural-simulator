# Nav place-critic drive calibration — HONEST NEGATIVE (the neural value subtraction doesn't port to deterministic nav)

**Date:** 2026-06-08
**Type:** decisively-mapped honest negative (no `sim/` edits; runner-side calibration + 8 GPU diagnostics).
**Predecessors:** `2026-06-08-nav-placecritic-smoke-PARTIAL.md` (silencing fixed + nav excellent, critic silent), `2026-06-08-gabab-girk-stageB-derisk-GO.md` + the place-code/value-leads de-risks (the mechanism that PASSED on the probe).

## Verdict

The mask-fix + place afferent + physiological GABA_B **resolved the nav silencing** and the agent navigates
excellently (overall distance 2.16, 828/1800 at goal). **But the `striosome_value` value critic cannot be made
to both fire AND learn a place-graded value while keeping nav sane.** Every lever that fires the critic
conflicts with another requirement. The mechanism that validated on the CPU de-risk does **not** port to the
integrated, deterministic nav regime. This is a real substrate boundary — a valid BRAIN-BASED-ONLY deliverable.

## The decisive diagnosis (the THIRD "probe ≠ deployment" gap in this arc)

The de-risk succeeded under **two conditions the deterministic nav does not provide**:

1. **OU background noise ON.** `CoreSimConfig()` defaults leave OU noise on; the de-risk ran that way, giving
   the MSN critic a fluctuating background depolarization (the biological **up-state**) so sparse place input
   could push it over threshold. The nav **disables OU** (`enable_ou_process=False`, runner line 3342) for
   determinism. With OU matched to nav (off), the MSN-D1 critic (measured rheobase ~700 pA) needs ~1200+ pA of
   place drive to fire at all — unreachable through the afferent at any weight (verified to w=25).
2. **A strong, dense place drive** (the probe drove the place region to 100+ Hz). In real nav,
   `sensor_place_readout` fires at only **~0.57 Hz** (instrumented over 6600 steps): at grid_size=32 the place
   tuning σ=0.5 (tuned for the 8×8 grid, cell spacing 1) is far too narrow for the ~4.43 cell spacing, and the
   region is `IZH2007_HIPPO_PYRAMIDAL` (intrinsically sparse). It is far too weak/sparse to drive a striatal
   critic.

Neither holds in deterministic nav, so the de-risk's success did not transfer — the same probe-vs-deployment
divergence that produced the GABA_B mask bug and the simultaneous-timing artifact earlier in this arc.

## The two-layer wall (re-smokes, seed 42, full flagship, GPU, deterministic — verified)

| Run | striov (critic) | weight Δ | nav overall | SNc |
|---|---|---|---|---|
| v1 (RS, w=12, σ=2.0, drive=600) | **0.00 silent** | 12.05→12.05 frozen | **2.16** (good) | ✅ 6.37 |
| v2 (RS, w=25, σ=2.5, **drive=1500**) | **0.53 FIRES** (674/1792 windows) | **25.10→24.44 (LTD, shrinks)** | **3.24** (degraded) | ✅ 7.38 |

Two independent failure layers:
- **Firing:** the critic only fires under a doubled place+goal drive (1500 pA), which **perturbs the actor**
  (tuned at 600 pA) → nav regresses (2.16 → 3.24).
- **Learning:** even when it fires, the weight undergoes net **LTD** (shrinks, doesn't learn the value) —
  the continuously-active place afferent into a fast-firing critic gives post-before-pre STDP timing.

Other levers, all rejected: a reward-window teacher current → post≫pre → LTD weight-collapse; a subthreshold
tonic → place-blind value (the tonic dominates, near≈far). So: strong drive → nav degrades + LTD; teacher →
LTD; tonic → place-blind. No runner-side config satisfies fire + learn + place-graded + nav-sane simultaneously.

## What is banked (real wins, unaffected)

- **GABA_B→GIRK conductance** (protected edit `6f73b5f0`/`a7370d49`): shipped, byte-reviewed, Pavlovian de-risk
  PASS 3/3. Audit shortcut #1 addressed. Unaffected.
- **The mask-fix** (`6f73b5f0`): resolved the per-synapse-growth silencing bug. Unaffected; a general fidelity fix.
- **Nav itself**: the silencing fix + place afferent give excellent nav (2.16) — but driven by the raw-reward
  RPE (Stage A), not the (silent) neural value critic.
- **The neural value-subtraction mechanism**: validated on the de-risk (it learns place-graded value + subtracts
  via GABA_B) — in the OU-on, strong-drive regime. The boundary is the *port* to deterministic continuous nav.

## Three fix directions (all larger than a runner-side calibration — controller/owner steer)

1. **A dedicated dense dorsal-place input for the critic** — a separate region (not the actor's sparse
   `sensor_place_readout`), properly tuned for the 32×32 grid + dense firing, strong enough to fire the critic
   on its own without perturbing the actor. Biologically: the ventral-striatal critic integrates a dense
   place/context representation, not a sparse motor-readout. Runner-side build, medium.
2. **Restore the MSN up-state** — a per-region background drive on the critic (the biological up-state). The
   tension: the up-state is *noisy* depolarization (OU-like), which conflicts with the deterministic-nav setup;
   a deterministic tonic is place-blind (rejected above). Needs either OU-on-for-the-critic (sacrificing strict
   determinism → more seeds for significance) or a per-region noise knob (a protected `sim/` edit).
3. **Fix the STDP timing** — make the place afferent LEAD the critic firing (the value-leads-reward finding),
   avoiding the post-before-pre LTD. Reuses the `transmission_gate` reward-window infra, but the continuous
   place activity in nav makes a clean lead hard to arrange.

## Honest verdict

The neural value critic **validates as a mechanism** (de-risk) but does **not port** to the deterministic,
sparse-place-code nav regime as built — a mapped substrate boundary. Per the BRAIN-BASED-ONLY standard this
honest negative IS the deliverable (it maps what the substrate can/can't do on its own). The GABA_B substrate
win + the excellent nav are banked. Whether to invest in one of the three fixes (each non-trivial, with a
determinism tension on the most biologically-principled one) vs bank the negative and move to the next step-1
nav item (the N5/N1/N6 host→neural conversions) is the open steer.
