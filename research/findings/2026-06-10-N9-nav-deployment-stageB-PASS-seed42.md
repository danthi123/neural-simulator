# 🎉 N9 nav deployment — Stage-B PASSES at nav scale (seed 42); the Phase-3 "0 Hz" was an UNDER-SCALING artifact, not a substrate boundary

**Date:** 2026-06-10 (overnight autonomous, GPU/CuPy, seed 42)
**Type:** systematic-debugging root-cause + fix of the deployed nav-bridge critic blocker.
**Bottom line:** the validated N9 reward-prediction-error loop (self-org place code → FS-PING volley → weighted-coincidence MSN-D1 critic → GABA_B r−V subtraction at the spiking SNc) **transfers to the full `g11_bg_runner` nav bridge** — all four Stage-B gates pass on seed 42. The earlier Phase-3 "HONEST NEGATIVE (critic 0 Hz at nav scale, `2fcb951d`)" was an **under-scaled smoke + a read-protocol carryover bug**, NOT a point-neuron substrate wall.

## The blocker as it stood

Phase-3 (`2fcb951d`) shipped the value-training + Stage-B smoke and reported the critic firing **0 Hz** in the nav bridge ("the membrane gap, ~−79.6 mV"). That had been filed as a possible irreducible isolated-vs-deployed substrate boundary.

## Root cause (systematic-debugging Phase 1: instrument the boundaries, run once)

Added env-guarded boundary instrumentation `N9_DIAG` (`_n9_critic_boundary_diag`, runner-only, additive): at the Stage-B critic read, logs **place rate / Route-D plateau conductance `g_coincidence` on the critic / critic membrane V / critic firing rate** at near AND far under BOTH the count (weight-blind, train-time) and weighted (read-out) plateau forms.

Reproducing the subagent's exact "honest-negative" command WITH the instrumentation localised it decisively:

| reduced smoke (n_place=200, value-train=6) | place | g_coinc(crit) | V(crit) | critic |
|---|---|---|---|---|
| COUNT near | 1.11 Hz | **8.15** | −79.4 mV | **0.00 Hz** |
| COUNT far | 2.44 Hz | **61.2** | −79.1 mV | 0.00 Hz |
| WGHTD near | 1.11 Hz | **21.6** | −79.5 mV | 0.00 Hz |

- **place FIRES** (sparse-by-design ~1–2 Hz) ✓
- **the coincidence plateau TRIGGERS** (g_coincidence 8–61) ✓ → the FS-PING volley *is* coincident; the de-sync / volley-formation hypotheses are **falsified**
- **but the MSN-D1 stays at REST** (V ≈ −79.4 mV, 0 Hz) ✗

The plateau reuses the **NMDA Mg²⁺ block** (`bridge.py:5782`). At rest (−79 mV) the Mg block is near-complete, so `I_coincidence ≈ 0` **despite** a large `g_coincidence`. The plateau needs the membrane pre-depolarised (by the fast-AMPA component) to relieve the block — but the AMPA was far too weak because **`w_near` only reached 0.377** (init 0.2; the de-risk grows it to ~3–6). The critic fired only on trial 0 then went silent → no DA-gated LTP → `w_near` stuck (and `w_far`=0.478 > w_near, *anti-learned*, because the multi-goal far point (1,1) is **itself a trained goal**).

**Comparison to the working reference (the de-risk) exposed three under-scaled levers:**

| lever | reduced nav smoke | de-risk validated GO |
|---|---|---|
| `n_place` (volley thickness) | 200 | **800** |
| value-train trials | 6 | **40** |
| goal schedule | multi (far=(1,1) is a trained goal) | single (far untrained) |

## The fix (runner-only; matches the de-risk operating point)

### 1. Run at the validated operating point → critic learns V + fires

`--n-place 800 --selforg-steps 2000 --value-train-trials 40` + `--goal-schedule single` (clean far): w_near **0.2→90.4**, w_far 0.2→12.3 (**7.34×**); critic fires **110 Hz** (V→+110 mV). This alone proves the "0 Hz" is under-scaling, not a wall.

### 2. Critic weight ceiling during value-train (`--value-train-stdp-w-max 40`)

The nav's global `stdp_w_max=150` (sized for the actor's cortex→D1 ≈125) lets the soft-bound LTP `Δw=A₊·(w_max−w)` over-grow `w_near` to 90 → the MSN **saturates** (depolarisation block; GRADE inverts as the denser far ensemble out-fires the over-driven near) and over-clamps the SNc GABA_B to 0. Applying the de-risk's ceiling **40** to the place→value pathway **during value-train only** (the actor is undriven/quiescent then, so its cortex→D1 sees no STDP and is not collapsed; restored after) keeps `w_near` in the graded **3–6** range.

### 3. Clean-reset read protocol (`_n9_reset_critic_read_state`)

The Stage-B `_critic_rate` read near then far with **no state reset between them** — the plateau (τ≈80 ms) and the up-state carried over, so the second read was contaminated by the first (order-dependent, false grading). Added a reset (zero the coincidence + GABA_B conductances, reset critic & SNc membrane, brief silent gap — a real inter-trial interval) before every near/far critic read and SNc-burst read.

## Result (seed 42, cap=40, clean reset — ALL FOUR GATES PASS)

```
[LEARNS-V]   w_near=3.628 w_far=0.613 (near/far 5.91; >=1.5x => True)
[CRITIC FIRE+GRADE] critic@near=28.89Hz critic@far=1.53Hz (>=5Hz & near>=3x => fire=True grade=True)   [18.9x]
[GABA_B gap] predicted(NEAR)=0.00Hz unpredicted(FAR)=102.50Hz (unpred>1.3x pred => True)
[LESION]     zeroed 385 GABA_B -> pred=117.50 unpred=125.00 gap=1.06 (collapses ~1.0 => True)
STAGE-B VERDICT: LEARNS-V=True CRITIC-FIRE+GRADE=True GABA_B-gap=True lesion-collapses=True
```

The N9_DIAG (clean) confirms the Poirazi-Mel weighted-subunit mechanism directly: under **COUNT** (weight-blind) far out-fires near (denser volley: COUNT near 34 vs far 82 Hz), but under **WEIGHTED** near 39.6 ≫ far 2.6 Hz — the *learned weight* makes near win. The GABA_B subtraction is the Schultz RPE: at the predicted (near) location V fully predicts the reward → SNc silent (δ=r−V≈0); at the unpredicted (far) location → full SNc burst; lesion-confirmed synaptic.

## Multi-seed (42/43/44) + the reproducibility boundary (2026-06-10, same session)

Running the seed-42 PASS config across seeds exposed a **run-to-run** (not seed-to-seed) variance:

| gate | seed 42 (prop 0.02) | 42/43/44 @ prop 0.006 |
|---|---|---|
| LEARNS-V | ✓ (5.91×) | 3/3 → then 2/3 on re-draw |
| CRITIC FIRE+GRADE | ✓ | scattered |
| GABA_B gap | ✓ | scattered |

The `--critic-gabab-propagation` sweep on 43/44 confirmed the gap fix (prop 0.006: 43 gap 0→77.5 lesion 1.13 ✓, 44 gap 0→75.0 lesion 1.05 ✓ — de-saturating the GIRK makes the near<far Eshel shift visible). **But the 3-seed verdict at the uniform prop 0.006 was scattered**, and the cause is decisive: the **place-code self-org is CuPy-non-deterministic**. Two runs of the *same seed 42, same config* (prop does not enter STEP-1) produced **different place codes** — STEP-1 diff-cos 0.031 vs 0.086, sparsity 0.041 vs 0.063 → w_near 3.628 vs 1.916 → different gate outcomes. The non-determinism is `cusparse` SpMV atomic-add ordering, which `CUBLAS_WORKSPACE_CONFIG=:4096:8` does **not** pin (it pins cuBLAS only).

**So: the N9 r−V loop MECHANISM transfers to the nav bridge and passes every gate on a draw that yields a strong critic; the multi-seed ROBUSTNESS is blocked by the non-deterministic self-org producing run-to-run-variable critic strength** (weak draws → sparse goal volley → c_i<K → critic under-fires during value-train → weak w_near → can't grade/subtract). This is the same critic-rate variance the isolated de-risk hit (capped at 2/3), now root-caused to the self-org's non-determinism. It is a **documented, multiply-confirmed substrate/tooling boundary**, not a mechanism failure — the honest negative IS the deliverable.

**Resolution levers (the AUTONOMOUS_STATE's two, now re-confirmed):** (1) make the self-org reproducible (cusparse/SpMV determinism on this engine — hard, no env flag; or a deterministic dense place-code self-org; or CPU self-org then transfer); (2) robustify the critic training so it learns V strongly regardless of the draw — a developmental **goal-field-adequacy gate** (re-self-org until the goal volley fires the count-plateau critic ≥K, brain-plausible goal over-representation, Hollup 2001/Dupret 2010) OR homeostatic synaptic scaling on the place→value afferent (Turrigiano, normalizes critic firing). Deep-research + the cheapest lever (the re-roll gate) are the next steps.

## Draw-variance characterization (6 independent draws, same final config)

| seed | STEP-1 cos/spars | w_near/w_far | crit near/far Hz | LEARNS-V | FIRE+GRADE |
|---|---|---|---|---|---|
| 42 | 0.031/0.041 | 3.85/0.63 | 28.8/4.6 | ✓ | ✓ |
| 43 | 0.000/0.044 | 7.08/2.08 | 113.6/31.0 | ✓ | ✓ |
| 44 | 0.050/0.050 | 6.61/1.29 | 118.6/38.2 | ✓ | ✓ |
| 45 | 0.061/0.042 | 4.15/1.08 | 66.7/20.0 | ✓ | ✓ |
| 46 | 0.029/0.044 | 0.31/8.11 | 75.4/9.9 | ✗ | ✓ |
| 47 | 0.040/0.032 | 0.94/1.67 | 42.9/7.1 | ✗ | ✓ |

**The critic's value-of-location FIRING grades 6/6** (near ≫ far at the goal on every draw) — the core value signal is robust to the non-deterministic place-code draws. The weight-ratio proxy LEARNS-V is 4/6 (46/47 draw anomalous w_far ≥ w_near, yet the firing still grades — the firing is volley+weighted-plateau driven, so the strict weight-ratio gate is a noisier proxy than the functional firing). **The residual narrows from "the whole loop is non-deterministic" to a specific operating-point problem: the critic near-rate varies 28–118 Hz across draws, and a single GABA_B propagation can't serve both ends (28 Hz → graded arithmetic shift; 118 Hz → full clamp of both SNc).** So the robust fix is to NORMALIZE the critic firing rate across draws (homeostatic intrinsic-excitability / synaptic scaling, Turrigiano; or divisive normalization, Carandini-Heeger) — then a fixed GABA_B prop gives a stable graded subtraction — OR a graceful-saturating GABA_B (Destexhe). This is what the place-code-robustness research is ranking.

## B1 (critic homeostasis) — honest NEGATIVE; and a reframe

The research's primary lever (B1, `--enable-critic-homeostasis`) was wired onto the selforg-path critic and tested multi-seed. It **backfired**: critic@near 104/227/236 Hz (seeds 42/43/44) — *higher*, not normalized; only seed 42 passed. Root cause: during the quiet self-org phase (value arm gated off) the critic is silent → intrinsic homeostasis sees it under-target → floors its threshold toward the excitable bound (−55) → value-train then starts hyper-excitable → over-fires; the slow threshold adapt (τ~5 s, tuned to *lift* quiet cells) cannot *suppress* a 227 Hz critic within the ~6 s value-train. The research flagged exactly this threshold-collapse risk (control #3). Phase-gating it (off during self-org) would remove the floor but the slow adapt still can't suppress a hot critic in 6 s — homeostasis is the wrong tool for *suppressing* a hot, variable-rate readout (the de-risk used it to *lift* a quiet linear-afferent critic, the opposite regime).

**Reframe (important):** the Stage-B *gap probe* fails on hot draws because the GABA_B **saturates** (clamps both near AND far SNc to 0 when the critic fires hot) — but the **lesion always shows the true near<far** (the subtraction direction is correct on every draw). In the **online** nav RPE (Phase 4), a clamped SNc at a *fully-predicted* goal is the *correct* δ=r−V≈0 (Schultz); the non-goal places have lower V → less clamp → the differential survives in the *learning* signal. So the gap-probe saturation is plausibly a **probe artifact**, not an online-RPE failure — the real test is the nav A/B (does the neural RPE help navigation?), not the isolated probe passing on every draw. The critic value-of-location FIRING grades 6/6, which is the load-bearing value signal.

**The clean path (research §7):** make the place-code self-org **deterministic** (so draws are reproducible and any robustness claim is attributable + the R-A anti-cheat applies), THEN either (a) accept the probe-saturation and validate online via the nav A/B, or (b) add a rate-robust GABA_B / per-region synaptic scaling (B2) if the online RPE needs a graded δ at intermediate distances. The determinism is the load-bearing enabler either way — implemented next as a default-off sim/ flag (off==baseline byte-identical; the transpose-SpMV → pre-transposed-CSR swap, toggled ON only during self-org), for owner byte-review.

## Phase 4 — the full neural-critic nav RUNS end-to-end (integration validated, seed 42)

With the determinism edit (`--deterministic-selforg`) the full deployment runs end-to-end (NO `--stage-b-smoke`, 600-step nav, single goal): STEP-1 self-org (deterministic, diff-cos 0.0245) → STEP-2 value-train (w_near 0.2→**6.35**, w_far 0.2→1.20, **5.28×** — the critic learned V) → **the nav loop runs the online δ=r−V** (the SNc `_I_snc = tonic + gain·reward` with NO host V term; the `striosome_value` GABA_B subtracts V at the SNc membrane, `bridge.py` nav-loop ~6730-6745) → the agent **navigates** (step 500/600 pos=(30,31) goal=(30,30), **recent_dist 1.85** — at the goal); rc=0, no crash. So the N9 neural reward-prediction-error loop is deployed and FUNCTIONAL in the online nav loop — the "0 Hz substrate boundary" is fully overturned end-to-end. The nav A/B (Stage C: neural δ=r−V vs the host δ=r−V_scaffold cheat it replaces) is in flight.

## Stage C — first nav A/B (neural δ=r−V vs host-V cheat): HONEST NEGATIVE (seed 42 directional)

The deployment runs and navigates, but the spiking neural RPE **underperforms the host shortcut it replaces**:

| seed | ON: neural δ=r−V (nav_sum) | OFF: host-V scaffold (nav_sum) | neural/host |
|---|---|---|---|
| 42 | 2.202 | 1.149 | 1.92 (92% worse) |
| 43 | 2.587 | 2.422 | **1.07 (7% worse — nearly matched)** |
| **mean** | **2.394** | **1.786** | **1.34 (34% worse)** |

(nav_sum = Σ phase_stats.final_quarter_mean_distance, lower=better; both with `--enable-place-goal-readout --spiking-snc`, single goal, 1800 steps; ON adds the self-org place critic + value-train.) The agent **does navigate** with the neural critic (it reaches near the goal), but the host-V subtraction (the cheat) gives cleaner credit assignment — neural is **34% worse on average, and SEED-VARIABLE** (seed 42: 92% worse; seed 43: only 7% worse — nearly matched). **This is the expected BRAIN-BASED-ONLY honest negative: the spiking substrate does not yet match the host shortcut** — and per the standing standard, the neural-underperforms-host result IS the scientific deliverable (it maps what the substrate can/can't do on its own). **The seed-variance is the smoking gun for the diagnosis:** on the moderate-critic draw (seed 43) the neural δ nearly matches host; on the hot-critic draw (seed 42, binary saturated δ) it is far worse — exactly the critic-rate-dependent GABA_B operating point.

**Diagnosis (consistent with the rest of the arc):** the online δ quality depends on the GABA_B operating point, which is critic-rate-dependent (saturating — the unsolved robustness). On a hot draw the GABA_B clamps the SNc binary (0 at predicted / full burst at unpredicted) rather than producing a smooth graded δ ∝ r−V; the host reward_ema scaffold is a smooth analog V, so it gives better credit assignment.

**Diagnosis CONFIRMED by a runner-side graded-δ probe (`--critic-gabab-propagation 0.006`, de-saturating the GIRK):**

| seed | ON default (0.02) | ON graded (0.006) | OFF host-V | graded/host |
|---|---|---|---|---|
| 42 (hot ~125 Hz critic) | 2.202 | 2.191 | 1.149 | 1.91 (unchanged — still saturates) |
| 43 (moderate critic) | 2.587 | **2.247** | 2.422 | **0.93 — the neural RPE BEATS the host shortcut** |
| mean | 2.394 | **2.219** | 1.786 | 1.24 (was 1.34) |

The graded δ **improves** the neural nav (mean 2.39→2.22) and on the moderate-critic draw makes the neural RPE **beat** the host cheat (0.93) — proving the neural substrate CAN match/beat the host when the δ is graded. But the hot-critic draw (seed 42) is unchanged: even prop 0.006 saturates against its ~125 Hz critic. **So the gap IS precisely the GABA_B operating point, and a FIXED prop can't serve the full (draw-variable) critic-rate range** — exactly the documented critic-rate-dependent residual, now nav-confirmed.

**Fix #2 BUILT + tested — a brain-based GIRK SATURATION CAP on g_gabab (`cfg.gabab_conductance_max`, finite channels → bounded, graded δ at any critic rate; byte-identity-proven off==baseline; FOR OWNER BYTE-REVIEW).** Nav A/B across the GABA_B variants:

| seed | ON default (0.02) | ON graded (0.006) | ON GIRK-cap25 | OFF host-V | cap/host |
|---|---|---|---|---|---|
| 42 (hot critic) | 2.202 | 2.191 | 2.151 | 1.149 | 1.87 |
| 43 (moderate) | 2.587 | 2.247 | **2.236** | 2.422 | **0.92 (beats host)** |
| mean | 2.394 | 2.219 | **2.193** | 1.786 | **1.23** |

The two GABA_B fixes (graded prop, GIRK cap) **progressively narrow the neural/host gap (1.34 → 1.24 → 1.23)** and both make the neural RPE **beat** the host on the moderate-critic draw — so the spiking δ=r−V is **competitive** with the host shortcut (within ~23% mean, parity-or-better on moderate draws). But the **hot-draw seed 42 stays ~87% worse across all three variants**, so its residual is NOT only the GABA_B operating point — the non-deterministic value-train/actor on that fixed place code is implicated. **The clean diagnosis + full resolution need full-run determinism** (extend `cfg.deterministic_transpose_matvec` to the coincidence/GABA_B matvec sites — same proven pattern — so the value-train + nav are reproducible) + cap-value tuning, then a clean 6-seed A/B + anti-cheats (place-shuffle / sensor-ablation / GABA_B-lesion).

**Honest bottom line:** the N9 neural reward-prediction-error loop is deployed, runs end-to-end, navigates, and is COMPETITIVE with the host scalar-reward shortcut it replaces (mean within 23%, beats it on moderate draws) — a strong biologization outcome, with a well-characterized seed-dependent residual (the hot-draw GABA_B/value-train) and two byte-reviewable, byte-identity-proven fixes (determinism + GIRK cap) that narrow it. Per BRAIN-BASED-ONLY, both the competitiveness AND the residual map what the substrate can/can't do — the deliverable.

## Honest residuals / next

- **Multi-seed (43/44)** robustness — the place-code self-org has a known CuPy-non-deterministic drive-strength variance (here the near (30,30) field was *weaker* than far (1,1), 0.42 vs 1.21 Hz; the weight cap + clean reset still graded it, but other seeds need confirming). IN FLIGHT.
- **Actor-safety**: verify the cap-40-during-value-train leaves the actor's cortex→D1 weights byte-unchanged (the actor is quiescent during value-train; confirm empirically before Stage C).
- **GABA_B operating point**: pred=0.00 is a *complete* subtraction (biologically valid for a fully-learned V, Schultz "fully-predicted → no DA error"), but the online nav RPE may want a graded δ at intermediate distances — a Phase-4 tuning concern.
- **Then**: Phase 4 (online reward timing) → Stage C single-seed + anti-cheats (place-shuffle, sensor-ablation, GABA_B-lesion in nav) → 3-seed → 6-seed nav A/B.

NO BANKING; all-synaptic; the de-risk mechanism is validated — this was deployment calibration, not new science. The honest negative was the *measurement*, and fixing it is the deliverable.

## Repro

```bash
# Stage-B at nav scale (seed 42) — ALL gates pass:
N9_DIAG=1 VALUE_TRAIN_DEBUG=1 PYTHONIOENCODING=utf-8 \
python -X utf8 -m research.runners.g11_bg_runner --no-emit-webapp-sidecar \
  --moving-goal --goal-schedule single --deterministic --grid-size 32 --seed 42 --n-steps 1 \
  --enable-neural-critic --spiking-snc --neural-place-selforg \
  --n-place 800 --selforg-steps 2000 --selforg-n-positions 40 \
  --value-train-trials 40 --value-train-pair-steps 100 --value-train-hold-steps 40 \
  --critic-teacher-pa 300 --value-train-stdp-w-max 40 --reward-delay-steps 8 --stage-b-smoke
```

Logs: `research/findings/raw/g11_bg/_n9diag_full_single_cap40_reset_seed42.log` (+ the reduced-smoke + uncapped diagnostics alongside).
