# "Step 3" de-risked: a LEARNED binder generalizes systematically on the stream-cortex codes (2026-06-16, CYCLE 98)

## One-line

The deferred "step 3" — replace the fixed exact-inverse VSA bind *algebra* with a cortex that **learns to
bind** — is **reachable on the stream cortex's own learned codes**: a learned binder generalizes
**systematically** to never-seen role-filler combinations (held-out **0.889**, memorization floor **0.000**,
7/9 splits systematic, all anti-cheats decisive). The learned bind is no longer a hard wall; it is a build.

## Why this matters

After the biologization sweep (CYCLE 97) made the bind *operation* spiking (±1 coincidence, recall 0.92), the
honest residual was the bind *algebra*: a fixed, hand-designed, exactly-invertible scheme — not a *learned*
cortical bind. A learned bind is the genuine "step 3," long framed as the deep/deferred frontier (tangled with
the months-scale dendritic-decorrelation question). This de-risk asks the precise question: does a *learned*
binder generalize on the codes the bridge actually learned from conversation?

## Result (3 seeds, F=16 fillers from the 320 stream codes)

Reusing the validated systematicity protocol VERBATIM (`cortex_learned_binder_systematicity_probe.run_condition`
— leakage-free train/held-out splits + the `BilinearBinder` + all four anti-cheats), with the only change being
the filler codes = the cached 320 stream-learned codes:

| metric | value |
|---|---|
| learned binder train acc | 0.943 |
| **learned binder HELD-OUT acc** | **0.889** (seeds 0.750 / 1.000 / 0.917) |
| chance (1/F) | 0.062 |
| **memorization floor** (lookup-table held-out) | **0.000** |
| shuffled-label control | 0.417 (drops well below held-out) |
| FHRR exact-inverse reference | 1.000 |
| **systematic splits** | **7/9** |
| between-code cosine (the regime) | +0.047 (sweet spot) |

The held-out accuracy (0.889) is far above the memorization floor (0.000) and chance (0.062), the
shuffled-label control collapses, and 7/9 splits pass the strict systematic test. So the binder is learning the
**binding relation** (it generalizes to combinations it never saw), not memorizing pairs. This confirms on the
ACTUAL stream-learned codes what CYCLE 89 showed on synthetic sweet-spot codes — and these codes are exactly the
moderate-correlation regime where a learned bind both generalizes (semantically) and binds.

## Honest scope + the build

- The learned binder here is a numpy gradient-trained network (the `BilinearBinder`: a learned bind, NOT the
  fixed VSA algebra). It generalizes systematically. So the learned-bind **principle** is de-risked on these
  codes.
- The remaining build is the **on-substrate (spiking) learned binder** — realizing the learned bind in
  neurons/synapses (vs the numpy BilinearBinder). That is a further de-risk + build (does a spiking learned
  binder generalize as the rate network does?), and it is the genuine "step 3" arc.
- It also held-out at 0.889 vs the FHRR algebra's 1.000 — slightly below the exact-inverse ideal, but unlike the
  fixed algebra it is *learned from data* and *generalizes*, which is the whole point.

## Verdict

Step 3 (a cortex that learns to bind) is **reachable** on the stream cortex's own codes — a strong, decisive
de-risk of the deepest deferred frontier. Recommended next: scope + build the on-substrate spiking learned
binder (per the standing deep-research/cheap-first practice).

## Follow-on (CYCLE 99): the rate-code read-noise risk is RETIRED

The deep-research scoping (`2026-06-16-onsubstrate-learned-binder-deep-research-scoping.md`) flagged the
single biggest risk of the *spiking* realization: finite-population read noise (~1/√n_per) on the bound
vector + unbind estimate could collapse the systematic generalization toward the memorization floor. It also
verified the de-risked binder is **additive** (`tanh(role·W_R + filler·W_F)`), so the spiking realization
needs only existing point-neuron primitives (two synaptic projections + a saturating nonlinearity +
population reads) — dendritic multiplication is the route to a *stronger* binder, not a prerequisite.

The cheap-first (`_phaseB_learned_bind_ratenoise_derisk.py`, train clean / score held-out under swept read
noise, 3 seeds) **retires the risk**: held-out is **flat** across the sweep — 0.778 (clean) → 0.773
(noise 0.20 ≈ n_per 25) → 0.764 (noise 0.30 ≈ n_per 11), all ≫ memorization floor 0.000. So the rate-code
wall does **not** break the learned binding; the spiking binder (surrogate-BPTT / local three-factor, unified
by e-prop) is worth building.

## Follow-on (CYCLE 100-101): the SPIKING binder — single-rate fails, ON/OFF carries it

Realizing the spiking binder (surrogate-gradient, the additive bind), two cheap-firsts on the systematicity
protocol (stream codes, 3 seeds):
- **#2 single non-negative rate (sigmoid) — NEGATIVE.** Held-out collapses to **0.083** (vs tanh baseline
  0.750, floor 0.000): a single rate channel loses the **sign** the additive bind needs (the sigmoid clusters
  near 0.5).
- **#2b ON/OFF opponency rate coding — GO.** A signed value → two non-negative rate channels (ON=relu(h),
  OFF=relu(−h); the substrate's standard signed coding, used in the NEF cleanup / FHRR / biologization). With
  read noise + surrogate backward (`d_h = d_ON·1[h>0] − d_OFF·1[h<0]`), held-out **0.806** — **107% of the tanh
  baseline** (0.750), ≫ floor 0.000.

⇒ the spiking learned binder is **viable**: the additive bind realized with ON/OFF rate coding + surrogate-
gradient + finite-population read noise carries systematic generalization. The single-rate collapse was a
representational mistake (lost sign), not a substrate wall. Next: the brain-faithful *local* learning rule
(feedback alignment / three-factor / e-prop, no weight transport), then the full on-bridge LIF realization.

Runners: `_phaseB_learned_bind_streamcodes_derisk.py`, `_phaseB_learned_bind_ratenoise_derisk.py`,
`_phaseB_spiking_bind_derisk.py`, `_phaseB_spiking_bind_onoff_derisk.py`.
Raw: `research/findings/raw/_phaseB_learned_bind_streamcodes.json`, `_phaseB_learned_bind_ratenoise.json`,
`_phaseB_spiking_bind.json`, `_phaseB_spiking_bind_onoff.json`.
