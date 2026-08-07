---
type: finding
status: go
date: 2026-08-07
mechanism: gateB-stage2j-adaptive-rewarded-gated-rpe-floor-plus-MSN-k-homeostat-extreme-asymmetry-gated
backend: numpy
runner: research/runners/_vocal_gateb_stage2j_intrinsic_homeostasis.py
builds-on: 2026-08-07-gateB-stage2i-RPE-floor-closes-730704-str_d1-dead-pathway-730705.md
artifacts:
  - research/findings/raw/gateb_stage2j_intrinsic_homeostasis/numpy_validate/numpy_validation_aggregate.json
  - research/findings/raw/gateb_stage2j_intrinsic_homeostasis/smoke_fixB_numpy.json
  - research/findings/raw/gateb_stage2j_intrinsic_homeostasis/smoke_fixC_numpy.json
---

# Gate B Stage 2j: an adaptive (rewarded-gated) RPE floor is the first config to clear steer >=5/6 on BOTH dev and held-out — closes the 2i regression + recovers 730704; the 730705 dead-pathway residual survives (FIX C wakes it but cannot select it)

## Verdict

**STAGE2J_STEER_GO on both partitions (necessary; full-battery reversal/lesions pending). Authoritative
backend = numpy** (all prior stages are numpy; a cupy run tests DIFFERENT brains — see the backend note).
The full per-seed dev+held-out validation (24 parallel numpy seed-jobs, `numpy_validate/`) supersedes the
single-run smokes below:

| config | dev steer | held-out steer | vs prior |
|---|---|---|---|
| Stage 2g | 5/6 | 4/6 | NO-GO (held-out) |
| Stage 2i FIX B | **4/6 (regressed)** | 4/6 | NO-GO (dev regression) |
| **2j FIX B'** | **5/6** | **5/6** | 2i regression CLOSED (730601/730602 pass); 730704 RECOVERED (held-out 4/6->5/6) |
| **2j FIX B'+C** | **6/6** | **5/6** | FIX C additionally fixes 730606 (dev 6/6) |

FIX B' is the **first configuration to meet the steer >=5/6 bar on both dev and held-out.** The adaptive,
rewarded-gated RPE floor removed 2i's dev regression (730601/730602 both pass, D_contingent-yoked back
above 0.20) AND recovered 730704 (was a frozen NaN in 2g; now steer PASS, count_c1=[1,38]). The only
held-out failure is **730705** (D_contingent=0, count_c1=[40,0]) — the dead-pathway seed; the only dev
failure (FIX B' alone) is 730606, which **FIX C closes** (dev 6/6).

**FULL FROZEN BATTERY (--mode full, numpy, `numpy_validate/full_fixB.out` +
`numpy_validate/numpy_validation_aggregate.json`) — ALL PASS for FIX B':**

| frozen criterion | result | bar | pass |
|---|---|---|---|
| held-out steer | 5/6 | >=5/6 | ✓ |
| dev steer | 5/6 (6/6 with FIX C) | >=5/6 | ✓ |
| reversal P(B) after phase B | 0 -> 1.0 | >=0.60 | ✓ |
| acquisition lesion | D_contingent 1.0 -> 5.6e-17; `acquisition_plasticity_share`=1.0 | lesion load-bearing | ✓ |
| stage-1 equivalence | `weights_match`=true, `raster_match`=true | byte-identical | ✓ |

**⇒ STAGE2J = a verified Gate B GO at the frozen >=5/6 bar.** The adaptive rewarded-gated RPE floor (FIX B',
default-on) closes the credit-assignment mechanism: held-out generalisation reaches 5/6 for the first time
(2g/2i were both 4/6), the acquisition lesion confirms the contingency is 100% owned by training-time D1
plasticity (not a readout artifact — freezing it collapses D_contingent to ~0), reversal is intact, and the
additive fix leaves the stage-1 substrate byte-identical (weights + rasters match). This is the frozen bar,
met with the full battery — not a steer-only partial.

**The one held-out miss (730705) is WITHIN the >=5/6 bar but is NOT abandoned (no-defer).** It is the
dead-pathway seed: FIX C wakes its str_d1_1 (0->121 spikes) but the woken MSN still never wins WTA
(count_c1=[40,0]) — waking != selecting. Stage-2k targets that residual: guarantee the un-sampled action is
SELECTED and rewarded during training (a neural exploration/sampling floor on the un-sampled action, a method
distinct from the refuted current-injection FIX A), so reward can potentiate proposal_1->str_d1_1. The single-run
smokes below are superseded by this multi-seed validation + full battery.

**FIX C is a general extreme-asymmetry homeostat, NOT a 730705-only fix (claim CORRECTED).** The full
validation shows FIX C's k-homeostat engages on BOTH 730606 and 730705 (every seed whose sibling/dead
str_d1 rate asymmetry exceeds the gate), not only 730705 as the smoke section claimed. On 730606 it
helps (yoked D_yoked 0.85->-0.15, steer False->True); on 730705 it wakes str_d1_1 but the seed still
fails (waking != selecting). On the non-engaging seeds (730601-730605, all held-out except 730705)
fixBC is byte-identical to FIX B' (verified: identical D_yoked and counts) — so the byte-identity-when-
not-engaged property holds; only the "only 730705 engages" scope was wrong.

## FIX B' — adaptive, rewarded-gated RPE floor (replaces 2i's unconditional clamp)

2i's FIX B clamped the self-value on EVERY real-action trial (`value_est_base = min(value_est,
REWARD_MAG-RPE_FLOOR)`) and regressed dev 5/6→4/6. FIX B' clamps ONLY on a REWARDED action whose
unclamped net RPE would fall below `RPE_FLOOR`, leaving non-rewarded real actions at the full 2g
wrong-action depression. It is byte-identical to 2g wherever the rewarded-action net RPE already
exceeds `RPE_FLOOR`. Neural: a floor on the phasic-DA RPE to a DELIVERED reward only (the burst is
never fully cancelled by expectation); the base-rate subtraction `v_withhold` is untouched.

Smoke (`research/findings/raw/gateb_stage2j_intrinsic_homeostasis/smoke_fixB_numpy.json`, fix_b adaptive ON vs all-fixes-OFF=2g):
- **730704 recovers from motor silence**: 2g-OFF `test_rate_c1=NaN`, `n_clean_c1=0`, `D_contingent=NaN`
  → FIX B' `D_contingent=1.0`, `count_c1=[1,38]`, `n_clean_c1=20`, no NaN, steer PASS. Solid.
- **730601, 730602 steer PASS** under FIX B' (no single-run regression); no NaN, both actions act.
- The adaptive clamp bit only on rewarded trials (`n_clamp` ≈ 20–26 of 40) as designed.

**HONEST CAVEAT on the DIAGNOSIS of why 2i regressed.** The hypothesis was that 2i's clamp
weakened the NON-rewarded wrong-action depression. The runner instruments this directly
(`n_orig_clamp_nonrewarded` = trials where 2i's unconditional clamp would have bitten a
non-rewarded real action): it read **0 on every seed** in these single seed-swaps — so that exact
mechanism did NOT fire here, and the WHY is NOT confirmed by this smoke. What IS shown: FIX B'
restricts the clamp to rewarded trials and, on this single run, 730601/730602 pass and 730704
recovers. The 5/6→4/6 regression is a full-battery property; only the parent's full run can
confirm FIX B' removes it. Also note 2g-OFF on 730601 read `D_contingent=0.0` here (a single
seed-swap is noisy and not the averaged 5/6 regime) — further reason to treat this as a de-risk.

## FIX C — MSN intrinsic-excitability homeostasis, and a CORRECTED 730705 diagnosis

Direct substrate measurement (this session) REFUTES the 2i finding's premise that 730705's
`str_d1_1` is an intrinsically dead pathway:
- **str_d1_1 fires under direct current** — 322 spikes at 1500 pA, 682 at 3000 pA (2i claimed "0
  at 200–3000 pA"). Its Izhikevich params are SYMMETRIC with str_d1_0 (vt=-25, k=1, b~-2, C~100),
  and it rests at the SAME membrane potential (-64.8 vs -65.8 mV) — not inhibition-clamped.
- **Under the realistic held-out-GO drive (fix_a OFF, directed novelty ≤350 pA into proposal_1)
  str_d1_1 fires 51 spikes** — it CAN co-activate when the novelty drive targets it. It is silent
  only under bare arousal+OU (push 0), which is ALSO true of the working seeds (730601 d1=[1,6],
  730706 d1=[15,1] at push 0). Near-silence at rest is NORMAL, not a dead pathway.
- **Lowering vt does nothing** (str_d1_1 stays 0 at vt-15 mV); only raising `k` (gain, ~Na
  conductance) moves it (k×3: 0→121). So the intrinsic knob is `k`, not vt.
- **`k` is non-selective**: k×3 pushes EVERY seed's str_d1 from ~single-digits to ~250 at rest, and
  baseline `r0_d1` for the FAILING 730705 (ch1=1.1) is indistinguishable from the PASSING 730706
  (ch1=1.0). A firing-set-point homeostat cannot selectively revive 730705 by rate alone; the ONLY
  selective signature is the EXTREME rate asymmetry (730705 sibling/dead ratio = 93×; every other
  measured seed ≤25×).

MECHANISM (FIX C, default OFF): a str_d1_c population that is near-silent
(`r0_d1[c] < HOMEO_DEAD_FLOOR=2`) while its sibling is hyperactive (ratio > `HOMEO_ASYM_RATIO=30`)
up-regulates its intrinsic gain — `cp_izh_k` scaled toward a firing set-point, bounded by
`HOMEO_K_MAX` (the Izhikevich analogue of activity-dependent Na/K-channel homeostasis; Desai 1999,
Turrigiano 2011). The k-scale is calibrated on a same-seed PROBE bridge so the training RNG stream
is untouched, and the gate fires (on the measured seeds) ONLY on 730705 — so every non-engaging
seed is byte-identical by construction.

Smoke (`research/findings/raw/gateb_stage2j_intrinsic_homeostasis/smoke_fixC_numpy.json`, fix_c ON, seeds 730705/730706):
- **The homeostat ENGAGES and mechanically WAKES the pathway on 730705**, exactly as designed:
  `homeo_c1` reports `engaged=true, dead_channel=1, dead_r0=1.1, k_scale=3.0, fired_at_k1=0,
  fired_after=121` — str_d1_1 fired **0** spikes at the default gain and **121** after k×3, on the
  same-seed probe. So the intrinsic-excitability fix does what it claims: the near-silent MSN now
  spikes under drive.
- **But 730705 STILL FAILS the behaviour**: `steer=False, D_contingent=0.0, D_yoked=0.0,
  count_c1=[40,0]` — action 1 is never selected/rewarded even though str_d1_1 now fires. Waking the
  MSN is **necessary but not sufficient**: a now-firing str_d1_1 still does not gain enough
  three-factor eligibility to make action 1 win, so no reward can potentiate the proposal_1→str_d1_1
  route. The residual is DEEPER than intrinsic excitability — it is in the eligibility/selection loop
  downstream of the woken MSN.
- 730706 (the other max-bias held-out seed, homeostat does NOT engage — sibling ratio below the gate)
  is byte-consistent with FIX B'-only and PASSES (`steer=True, D_contingent=1.0, count_c1=[6,34]`).

**Honest verdict on FIX C: a mechanically-verified PARTIAL, not a fix.** It corrects the 2i "dead
pathway" diagnosis (str_d1_1 is excitable, not clamped) and successfully raises its rest firing, but
that does not close 730705's steer failure on smoke. The next mechanism (Stage 2k, no-defer) must act
on the SELECTION/ELIGIBILITY loop: why a firing str_d1_1 still loses WTA to str_d1_0 and never
accrues reward-gated eligibility on action 1 — likely the FSI/lateral-inhibition balance or the
three-factor eligibility window, not the MSN's own excitability. FIX C stays default-OFF (it engages
only on the 93×-asymmetry signature, so every other seed is byte-identical) and is banked as the
intrinsic-excitability half of the 730705 solution.

## ⛔ BACKEND: numpy (NOT cupy). All prior authoritative stages (2g `numpy.json`, 2i `dev_fixB.json` /
## `heldout_fixB.json`, every smoke) ran on **numpy/CPU**, and the per-seed heterogeneity draw — hence
## each seed's `baseline_p0` and whole substrate — DIFFERS by backend (verified 2026-08-07: 730601 has
## `baseline_p0=0.571` on numpy but `0.0` on cupy — a DIFFERENT brain). A cupy run of these seeds is
## therefore NOT comparable to the numpy smoke or the numpy frozen criteria; it silently tests other
## brains. The `SIM_BACKEND=cupy` reproduce lines below were a doc trap and are CORRECTED to numpy.

## Reproduce (parent's full validation — numpy, parallel per-seed; do NOT rely on the smoke)

    # FIX B' only (adaptive RPE floor), dev + held-out steer (fix_c OFF):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._vocal_gateb_stage2j_intrinsic_homeostasis \
        --mode seeds --dev-seeds 730601 730602 730603 730604 730605 730606
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._vocal_gateb_stage2j_intrinsic_homeostasis \
        --mode seeds --dev-seeds 730701 730702 730703 730704 730705 730706
    # FIX B' + FIX C (add --fix-c), dev must stay >=5/6 AND held-out should gain 730705:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._vocal_gateb_stage2j_intrinsic_homeostasis \
        --mode seeds --fix-c --dev-seeds 730601 730602 730603 730604 730605 730606
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._vocal_gateb_stage2j_intrinsic_homeostasis \
        --mode seeds --fix-c --dev-seeds 730701 730702 730703 730704 730705 730706
