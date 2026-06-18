# TRUE-ONE-BRAIN #2 — NEURAL reward on the merged "one brain": QUALIFIED GO (2026-06-18, CYCLE 213)

**Roadmap #2** of the spike-ification audit (`2026-06-18-full-spikeification-shared-substrate-roadmap.md` §3 #2):
make the merged nav episode source its reward `r` NEURALLY — from `sc_rostral→reward_us` firing (the N5
superior-colliculus proximity / goal-salience approach reward) — retiring the host Manhattan/sign reward
formula (`g11_bg_runner.py:6901-6946`). Combined with the CYCLE-211/212 value-train (learned V), this makes
**δ = r − V FULLY synaptic on the one brain** (r = SC-chain excitation, V = the critic's GABA_B subtraction,
δ = the SNc firing). Validation runner `research/runners/_merged_neural_reward_validate.py`.

## VERDICT — QUALIFIED GO (the mechanism is achieved; two documented operating-point caveats)

The runner's auto-verdict printed **BOUNDARY** because it demands all gates 6/6; by the project's standing
**≥5/6** rule every hard gate PASSES, so the #2 mechanism — a genuinely neural reward on the one brain — is
achieved. The honest residual is an operating-point caveat (gradient spread), not a failure of the mechanism.

| Gate | Result | Standing bar | |
|---|---|---|---|
| GRADED (corr(ecc, reward_us) ≤ −0.5) | **6/6**, mean corr **−0.814** | ≥5/6 | ✅ |
| BURST (SNc reward-burst ≥ 1.3× tonic, synaptic) | **5/6**, mean **1.41×** | ≥5/6 | ✅ |
| LESION (sever `sc_rostral→reward_us` → reward collapses) | **3/3** (reward_us 63.75→0 Hz) | 3 clean | ✅ |
| MOAT (no-confab intact) | **6/6** (`what_does('dog','go')='north'` + `('river','look')=None`) | ≥5/6 | ✅ |
| NAV not regressed (neural vs host reward) | mean ΔmeanD **−0.508** (neural navigates CLOSER) | not worse | ✅ |

Per-seed burst: 42=1.45, 43=1.42, 44=1.55, 100=1.52, 101=1.39, **102=1.13** (the lone miss). Per-seed nav
ΔmeanD (neural−host, lower=closer=better): 42=+0.095, 43=+0.007, 44=**−1.627** (host 3.37 → neural 1.75 =
neural FIXED a bad host-reward seed). corr is identically −0.8137 across all 6 seeds (deterministic SC bump).

## The two honest caveats (operating-point, NOT substrate walls)

1. **The het-off proximity gradient COMPRESSES to the nearest bin.** `us_rs` (reward_us firing per
   eccentricity) is non-zero ONLY at ecc=2 and 0.0 at ecc 4-7, every seed — so the −0.814 corr is carried by
   the binary close-vs-far contrast, NOT a finely-spread gradient. The SNc rate (`snc_rs`) shows a mild spread
   (~95-110 Hz at ecc 4-7 → 171 Hz at ecc 2), but the reward afferent itself is close-vs-far. Cause: het-off,
   the standalone SC weights starve (sc_map ~2 Hz, reward_us never crosses threshold), so the run uses an
   env-var-gated merged-tuned SC op (`SC_RET_SC=160`/`SC_REC=12`/`ros_us=40`/retina 3500) to make the SC bump
   form at all — and that bump only forms at ecc ≤ ~4 het-off. Lever to SPREAD it (a documented follow-on):
   per-region homeostasis on `sc_map`/`sc_rostral` (they are NOT in the homeostasis mask — only `reward_us`/`snc`
   are), or a stronger retinotopic recurrence.
2. **Seed 102's burst is 1.13× (the 5/6 miss)** — its SNc tonic ran hot (base_snc 165 vs ~120 others), so the
   reward-burst ratio compresses even though `close_snc` (186.7) is the highest. An SNc-tonic operating-point
   variance, not a mechanism failure.

## The build (runner-only, NO sim/ edit, default byte-identical)

`research/runners/g11_bg_runner.py`, two additive changes, both verified default-preserving:
- **The reward-routing fix (latent-bug fix):** the branch that zeros the host `reward_us` write gated on
  `"approach_n5" in region_indices_cp` — a region the N5 redesign DROPPED (always False) — so
  `enable_spiking_sc_approach` SILENTLY fell through to the host `reward_us` write, overriding the synaptic
  `sc_rostral→reward_us` pathway. Changed to `"sc_rostral" in region_indices_cp` → the host write is zeroed
  when the SC approach is on → the pathway carries `r` synaptically. `enable_spiking_sc_approach=False` → the
  host path is unchanged (the default + the standalone nav default are byte-identical).
- **The het-off SC op boost** (`SC_RET_SC`/`SC_REC` env-var override of the default 14.0): only active when the
  env vars are set AND `enable_spiking_sc_approach` is on; default = byte-identical.

The SC reward chain (`sc_retina→sc_map→sc_fs→sc_rostral→reward_us→snc`, built under `enable_visual_cortex`)
COMPOSES co-resident on the merged bridge (54 regions / 9468 neurons) with the conversational parser/dlPFC/
composer + the nav critic — moat intact.

## Anti-cheats (BRAIN-BASED-ONLY honest)

- **Decisive LESION (3/3):** cutting `sc_rostral→reward_us` drops reward_us 63.75→0.0 Hz and abolishes the SNc
  burst → the reward IS the synaptic SC proximity, not a re-hidden host scalar (the host-scalar version would
  be lesion-insensitive). This is the load-bearing proof that `r` is neural.
- **MOAT 6/6:** the dopamine `scope=all` broadcast (now driven by a neural reward) does NOT perturb the frozen
  conversational slice.
- **NAV-not-regressed is the honest control, not a win claim:** the gridworld is orient-solvable, so the reward
  is NOT strongly behaviorally load-bearing — a nav-neutral result is the EXPECTED, honest outcome (a swing
  would itself be the finding). Seed 44's improvement (host's bad-reward seed fixed) + 2 noise-level seeds.

## Scope / what this is and isn't

ACHIEVED: the reward `r` on the merged "one brain" is now a SYNAPTIC quantity (SC proximity → reward_us → SNc),
lesion-proven, moat-safe, nav-neutral; with the learned V, **δ=r−V is fully synaptic on the one brain**. NOT
claimed: a finely-spread proximity gradient (it compresses close-vs-far het-off — the op-point follow-on), nor
a behavioral nav improvement (the task doesn't need the reward). Both are documented, not buried.

## Reproduce
```bash
SIM_BACKEND=cupy SC_RET_SC=160 SC_REC=12 python -m research.runners._merged_neural_reward_validate --seeds 42 43 44 100 101 102
```
