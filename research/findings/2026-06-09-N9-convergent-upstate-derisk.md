# N9 convergent-excitation up-state value-critic de-risk — CuPy 3-seed: FIRE + LEARNS + ACTOR-OK, but PLACE-GRADED **NEGATIVE** (the dense up-state is position-blind)

**Date:** 2026-06-09
**Type:** runner-side implementation + decisive CuPy de-risk (the design's Option A, with the wiring-collision fix). NO `sim/` edits (`git status --short sim/` byte-empty throughout).
**Blueprint:** `research/findings/2026-06-09-N9-faithful-value-cell-design.md` (Option A two-component convergent place afferent) + `2026-06-09-N9-cupy-membrane-divergence-ROOT.md` (numpy disqualified; MSN-D1 rheobase ≈339 pA).
**Owner directive:** biologize N9, no banking, brain-based-only. **An honest negative IS the deliverable.**

---

## TL;DR — VERDICT: **PARTIAL → NEGATIVE on the load-bearing gate (PLACE-GRADED).** N9 convergent-up-state biologization de-risk does **NOT** GO for the nav A/B.

The convergent-excitation up-state mechanism, built faithfully and tested on **CuPy** (the production backend; numpy disqualified) across **3 seeds** in the deterministic-nav regime:

| Gate | Result | Notes |
|---|---|---|
| **1 FIRE** (critic ≥5 Hz at goal, init) | ✅ **3/3 PASS** | A1 up-state fires the critic at the goal **on CuPy** — directly refutes the silent-weak-afferent (~5 pA) result. |
| **3 LEARNS-V (LTP from init)** | ✅ **3/3 PASS** | A2 near-ensemble weight 0.20 → **~5.0** (w_far stays 0.20, ratio ~25×). **The load-bearing bootstrap gate the whole arc never satisfied on CuPy is now satisfied** — A1's up-state gives A2 a post-spike to pair with. |
| **4 ACTOR-NOT-PERTURBED** | ✅ **3/3 PASS** | actor cortex rate identical with/without the critic (ratio 1.000) — the dense afferents feed ONLY the critic. |
| **2 PLACE-GRADED** (critic NEAR ≥3× FAR) | ❌ **0/3 FAIL** | trained NEAR/FAR ratio caps **~1.2–1.4** at every A1 weight, with or without NMDA. |
| **5 GABA_B subtraction (SNc gap)** | ❌ not robust | 0/3 at the default config; **2/3 at one cherry-picked operating point** (a1=24, snc_w=4, lead=150). Lesion + GABA_A A/B controls behave correctly (gap vanishes / never opens). |

**Root cause of the negative (design Option D, confirmed):** the **dense NON-plastic A1 up-state arm is POSITION-BLIND.** Its per-location critic firing is set by *which afferent cells happen to wire onto the critic and their random jittered weights*, **NOT** by the goal location — even with `n_active` matched (85 vs 85 active cells), two positions fire the critic at **1.79 Hz vs 16.79 Hz** (~10× apart). So the critic enters the up-state *wherever a place bump exists*. A2's DA-gated LTP **adds** NEAR-selectivity (NEAR gets +20 Hz of learned drive, gate-3 strong) but **cannot suppress** the position-blind FAR up-state floor → the NEAR≫FAR grade a *value-of-location* requires never opens. **Option B (per-region NMDA on the critic) makes it WORSE** (NMDA deepens the up-state at *both* locations: NEAR 88 Hz / FAR 62 Hz, ratio 1.41).

**The genuine tension (irreducible within this mechanism):** gate-1 (fire at init) *requires* A1 to fire the critic from a place bump, but A1 firing the critic from any bump *means it fires at FAR too* → gate-2 fails. Lowering A1 to silence FAR also silences NEAR (A1 is position-blind), breaking gate-1. No A1 weight (swept 16→32) clears both.

**This maps a real substrate boundary:** the up-state **fires** and the value **learns**, but the spatial selectivity that makes it a value *critic* needs a **richer / self-organized place code** (a layer where NEAR and FAR are genuinely different up-state drives — e.g. learned place cells from landmark sensors), not a hand-rendered dense Gaussian read through a position-blind convergent blob. Per the design's graceful-FAIL contract: A failed → B (NMDA) failed → **Option D (bank the deeply-mapped negative).**

---

## What was built (runner-side ONLY; `sim/` byte-empty)

### The wiring-collision fix (verified)
`RegionManager.build_wiring_plan` keys pathways `f"pathway_{from}_to_{to}"` (`sim/regions.py:537`) → two pathways from ONE region COLLIDE. So the two arms are **two DISTINCT regions**, both drive-injected with the SAME grid-32 Gaussian place code each step:
- **A1 `vs_place_drive`** (NEW): dense (0.8), **NON-plastic**, weight ~28 (many weak synapses summing past the ~339 pA rheobase — NOT one giant synapse: ~12,821 synapses onto the 80-cell critic) → the B.02 convergent up-state, fires the cell from init.
- **A2 `vs_place_context`** (existing): sparse (0.4 in the probe / 0.5 in the runner), **PLASTIC** init 0.2, DA-δ-gated (gate `value_input`) → learns V(s).

### Runner edits — `research/runners/g11_bg_runner.py` (all opt-in, default-OFF, byte-equivalent when off)
- `build_bg_brain_regions(...)`: new params `enable_convergent_upstate=False`, `vs_place_drive_to_value_weight=28.0`, `vs_place_drive_to_value_density=0.8`. When on: adds the `vs_place_drive` `BrainRegion` + the dense NON-plastic `vs_place_drive→striosome_value` `RegionPathway`; the existing `vs_place_context→striosome_value` plastic pathway becomes the A2 arm. **Build smoke: OFF = 40 regions/54 pathways (no `vs_place_drive`); ON = 41 regions/55 pathways (+A1).**
- `run_g11_bg(...)`: same three params threaded through to the builder.
- **Nav-loop drive injection** (the per-step Gaussian render): when on, the SAME `vs_drive` is injected into `vs_place_drive` as well as `vs_place_context` (zeroed during sleep).
- **`_run_critic_warmup`**: when on, the LEARN window also drives `vs_place_drive` so the critic FIRES (A2 gets a post-spike). **Verified end-to-end in the full nav bridge (Gabor visual cortex on): the warm-up grew the critic weight 0.20 → 1.84** — the deployed-nav weight that was frozen at 0.20 in every prior attempt now learns.
- **CLI**: `--enable-convergent-upstate`, `--vs-place-drive-to-value-weight`, `--vs-place-drive-to-value-density`.
- A header comment at the call site records the de-risk outcome (FIRE/LEARNS/ACTOR pass, PLACE-GRADED fail) so the negative is documented in-code. **Shipped default-OFF; NOT in any flagship config.**

### Probe — `research/runners/n9_convergent_upstate_derisk.py` (CuPy-only, self-contained)
Builds the two-region critic + actor stub in the deterministic regime, trains value-leads-reward, runs gates 1–5 + anti-cheats. Hard-asserts `backend=="cupy"` and OU/conductance-noise/global-homeostasis OFF (anti-cheat d, regime fidelity). `--shuffle` (anti-cheat b), `--lesion` (gate 5), `--no-gabab` (A/B), `--nmda-critic` (Option B), `--near-x/y --far-x/y` (matched-position control).

---

## The decisive CuPy numbers (3 seeds 42/43/44, A1=28, NEAR=(8,24) FAR=(24,8))

```
                          seed42   seed43   seed44
FIRE  critic@NEAR init     8.57Hz   ~5-9Hz   ~5-9Hz   -> 3/3 PASS (>=5Hz)
LEARNS w_near 0.20-> ...   5.20     4.99     4.94     (w_far stays 0.20; ratio ~25x) -> 3/3 PASS
       V(near) early->late 4.3->10  5.4->9.8 3.6->9.3
ACTOR  with/without ratio  1.000    1.000    1.000    -> 3/3 PASS
PLACE-GRADED trained N/F   1.24     1.25     1.17     -> 0/3 FAIL (need >=3x)
       (NEAR/FAR Hz)       30.9/24.8 30.0/23.9 28.2/24.1
SNc gap (state-specific)   0        0        0        -> 0/3 at default config
```

### Why PLACE-GRADED fails — the position-blind up-state (CuPy diagnostics)
- **`_n9_placegrade_probe.py`** (A1 alone, bump swept across the diagonal): critic rate tracks **`n_active`** (firing afferent-cell count), peaking in the **grid center** (~120 cells → ~29 Hz) and dropping at the **corners** (~37 cells → 0 Hz). It is a "how many afferent cells are active" blob, not a value-of-location.
- **`_n9_matched_pos.py`** (NEAR/FAR with `n_active` MATCHED 85/85): critic baseline **1.79 Hz vs 16.79 Hz** at A1=24 — same input, ~10× different rate, set by random synaptic structure (which ensemble's synapses onto the critic are stronger).
- **A1-weight sweep 16→32** (`n_train=80`): no weight clears both gate-1 (init fire) AND gate-2 (≥3× grade); the trained ratio peaks ~1.2–1.4 at every weight.
- **Option B NMDA**: NEAR 88 Hz / FAR 62 Hz, ratio 1.41 — NMDA's regenerative current deepens the up-state at BOTH locations, making grading worse.

### Anti-cheat controls (all behave correctly)
- **(d) regime fidelity**: backend==cupy asserted; OU/conductance-noise/global-homeostasis OFF asserted (hard-fail otherwise). No per-region homeostasis on the critic (Option A fires by convergent current, NOT threshold collapse).
- **(a) population code**: NEAR/FAR are distinct dense ensembles (Jaccard 0.12–0.17).
- **(5) GABA_B lesion** (a1=24, 3 seeds): cutting the GABA_B mask → SNc bursts FULL at both NEAR (~84 Hz) and FAR (~85 Hz), gap ~1.0 → **0/3** (the subtraction is GABA_B-carried; cutting it removes it).
- **GABA_A A/B** (a1=24, 3 seeds): gap ~1.1 (both ~80–90 Hz) → **0/3** (the depolarized-SNc wall — only GABA_B can hyperpolarize the KCC2-lacking SNc).
- **(b) place-shuffle**: moot — gate-2 already fails in the TRUE case, so there is no place-of-location value for the shuffle to ablate (noted honestly; not claimed as a pass).
- **SNc gap robustness** (`_n9_snc_gap_probe.py`): the gap appears at only ONE hand-found operating point (a1=24, snc_w=4, lead=150) and even there is **2/3** seeds (seed 43 at 1.29, just under 1.30). Not robust.

---

## Honest bottom line + recommendation

The faithful convergent-excitation up-state **succeeds at firing the MSN-D1 critic on CuPy and at learning V via DA-gated LTP from a realistic init — both load-bearing wins the arc had never achieved on the production backend (the LTP-bootstrap deadlock is broken structurally, and the deployed-nav warm-up now grows the critic weight 0.20→1.84).** It **fails** at producing a place-graded *value of LOCATION*: the dense NON-plastic up-state arm is position-blind (it fires the critic wherever a bump exists), so the trained critic is never NEAR≫FAR, and the GABA_B value-subtraction gap is consequently not robust. NMDA (Option B) deepens the up-state at both locations and makes it worse.

**This is a clean, multiply-confirmed BRAIN-BASED-ONLY honest negative** (design Option D): the substrate *can* fire+learn a striatal value cell from convergent excitation, but a *value-of-location* needs the place input itself to be place-specific in the up-state — i.e. a **self-organized spiking place-cell layer** (learned place cells from landmark sensors, `--enable-landmarks`), not a host-rendered dense Gaussian read through a position-blind convergent blob. That place-code-as-shortcut is a **separate** item to biologize (flagged in the design §1.4), and is the real unlock for a graded neural critic.

**Recommendation:** do **NOT** run the 6-seed nav A/B for the convergent-up-state critic (gate-2 fails). The mechanism is shipped default-OFF as documented infrastructure. The next faithful step for N9's graded critic is a self-organized place code (the dorsal "where" stream), after which the convergent-up-state + A2-LTP machinery (which fires + learns) can be re-de-risked on a place input that is genuinely position-specific.

---

### Artifacts
- Implementation: `research/runners/g11_bg_runner.py` (`build_bg_brain_regions` + `run_g11_bg` + nav-loop + warm-up + CLI; opt-in `--enable-convergent-upstate`, default-OFF, byte-equivalent off).
- Probe: `research/runners/n9_convergent_upstate_derisk.py` (CuPy-only, gates 1–5 + anti-cheats).
- CuPy de-risk JSONs: `research/findings/raw/_n9_upstate_derisk_3seed.json` (primary 3-seed), `_n9_upstate_lesion_3seed.json` (gate-5 lesion), `_n9_upstate_gabaa_3seed.json` (GABA_A A/B), `_n9_upstate_navsmoke.json` (full-nav warm-up firing).
- CuPy diagnostics: `research/findings/raw/_n9_upstate_calib.py`, `_n9_upstate_instrument.py` (afferent f-I + critic g_e/V), `_n9_afferent_probe.py`, `_n9_upstate_calib2.py`, `_n9_placegrade_probe.py` (position-blindness), `_n9_matched_pos.py` (matched-`n_active` 10× rate split), `_n9_snc_gap_probe.py` (gap fragility 2/3).
- `git status --short sim/` byte-empty (verified before/after every stage).
