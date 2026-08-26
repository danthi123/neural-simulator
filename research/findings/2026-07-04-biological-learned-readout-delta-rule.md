---
type: finding
status: live
date: 2026-07-04
mechanism: learned-readout-delta-rule
---

# Biologizing the reservoir→role read-out — a LEARNED (delta-rule) read-out resolves the degraded seed 44 + removes the host ridge shortcut (2026-07-04)

**One-line:** the seed-fragile read-out (CYCLE 918/919) had a residual NON-biological shortcut hiding in plain sight — the
read-out matrix `Ws` was a HOST RIDGE FIT (`np.linalg.solve`). A research gate reframed the fix: replace it with a
per-role **DELTA RULE learned ON the spiking substrate**. De-risk: seed 44 (host-fit 11/18, the degraded draw NO
fixed-circuit mechanism could crack) reaches **18/18**, and the scrambled-label anti-cheat is **0/18** (the learning is
genuinely role-specific). It scaled with training (E4→12/18, E12→18/18 — under-trained, not ceiling-limited). The 6-seed
blind generalization test (the headline claim) is in flight. Strictly CPU/numpy; **NO `sim/` edit**.

## Why the host ridge fit was the problem (measurement-grounded, CYCLE 919)

The read-out reproduces `argmax_r((f·Ws)[r])` on the spiking substrate. `Ws` is fit by a HOST ridge solve on the reservoir
RATE feature. The B-1c arc removed the read *step* shortcut (host `f@Ws`→synapses, host argmax→neural argmax), but the
*weights* stayed host-learned. Measured failure: the host-fit `Ws` delivered as synapses is SEED-FRAGILE — 18/18 on the
dev draws 42/43, but 11/18 on seed 44 and **7/9/5 out of 18 on the unseen 100/101/102** (near chance). Root cause (a
research gate verified by measurement): a **train/deploy objective MISMATCH**. The ridge minimizes `‖f·Ws − Y‖²` (a linear
reconstruction of a rate matrix), but deployment runs a spiking WTA whose winner is set by IGNITION ORDER — a
threshold-nonlinear, dynamics-dependent quantity the ridge never sees. So `Ws` is correct for the linear surrogate and only
COINCIDENTALLY correct for the spiking argmax on the dev draws; on an unseen draw the margin lands on the wrong side of the
WTA ignition inversion. (NOT a sub-1% margin, NOT a degraded feature, NOT the dendritic frontier — all refuted: DRIVE-WRONG
=0/18, isolated ens f-I monotone to 450 pA.)

## The mechanism — a per-role delta rule learned on the spiking substrate

Per training sentence + content slot k (the reservoir is FROZEN; only the `res2ens` synapses are plastic):
```
drive the frozen reservoir → ρ = reservoir firing,  a = ACTUAL ensemble firing (via run_with_ens, the REAL spiking read)
error_r = T_r − a_norm_r                 (T = one-hot on the KNOWN slot-k role label — environmental supervision)
W_k[r, :] += η · error_r · ρ             (Widrow-Hoff / cerebellar PF→Purkinje form; clip ≥ 0, Dale-legal excitatory)
```
The learned `W_k` ARE the read-out (delivered as `res2ens` synapses) — NO host `np.linalg.solve`, NO host `f@Ws`, NO host
argmax (winner = the neural argmax over the ensembles' firing). Three load-bearing properties, each grounded in the
project's own track record:
- **PER-ROLE-LOCAL error, not global scalar.** `(T_r − a_r)` is computed independently at each of the 3 role ensembles —
  the "per-region/per-role error" credit-assignment that passed **3/3** (supervised gradient) where a global DA scalar
  FAILED (sign-only 1/6, magnitude 0/6; `2026-05-05-W-to-A-VERDICT`). Same architecture, only the credit rule differs.
- **`a` is the REAL spiking ensemble firing** — so the f-I nonlinearity + the WTA ignition-order are INSIDE the error
  term. The rule doesn't reproduce a host matmul; it drives the correct ensemble to WIN THE SPIKING COMPETITION on THIS
  draw. Swap the draw, re-run the same local rule, it re-finds the winning weights → **generalizes by construction** (the
  project's own learned-cortex thesis: "learn to read whatever messy code arrives"; Gilra-Gerstner FOLLOW 2017).
- **Rate-Hebbian / delta, NOT spike-timing STDP** — the reservoir-feature × per-role-error co-activation is symmetric
  (Δt≈0), exactly where STDP is measured-NEGATIVE (`2026-06-15-on-bridge-hebbian-co-occurrence`, 656k events / 0 Δw).
- **FREEZE the reservoir** (training the recurrence HURTS: 0.25 vs 0.90, `_fork2_predesign`); the `res2ens` synapses are
  the only plastic pathway. Precedent: the project's own scratch prototype (fixed reservoir + local delta-rule read-out)
  scored **1.000** (`_fork2_predesign_local_credit_prototype.py`).

## Results (de-risk, seed 44 = the degraded draw; CPU/numpy)

| read-out | seed 44 |
|---|---|
| host ridge fit (committed) | 11/18 |
| learned delta rule, E4 (under-trained) | 12/18 |
| **learned delta rule, E12** | **18/18** |
| learned delta rule, E12, SCRAMBLED labels (anti-cheat) | **0/18** (must fail — it does) |

⇒ the mechanism WORKS (learning is real + role-specific: scrambled-label collapses to 0), it SCALES with training (the
E4→E12 climb proves under-training, not a ceiling), and it RESOLVES the degraded seed 44 that host-fit + ~20 fixed-circuit
read mechanisms could not.

## 6-SEED BLIND generalization (the headline) — a decisive surpass (5/6 at 18/18, 1 at 14/18)

Same FIXED protocol (E12/η0.05/N35, no per-subset tune) on all six:

| seed | host ridge fit (committed) | **learned delta rule** |
|---|---|---|
| 42 (dev) | 18/18 | **18/18** |
| 43 (dev) | 18/18 | **18/18** |
| 44 (degraded) | 11/18 | **18/18** |
| 100 (unseen) | 7/18 | **18/18** |
| 101 (unseen) | 9/18 | 14/18 |
| 102 (unseen) | 5/18 | **18/18** |

⇒ **the learned read-out GENERALIZES**: host-fit was GO on only 2/6 (42/43); the learned read-out is **5/6 at 18/18** and
lifts EVERY previously-failing unseen seed (100: 7→18, 102: 5→18, 44: 11→18, 101: 9→14). This is a real generalizing
biological surpass AND it retires the host ridge-fit shortcut. Seed 101 (14/18) is the lone laggard — almost certainly
under-trained at E12 (seed 44 climbed 12→18 with more training), the exact-next to close toward clean 6/6.

## Session update (2026-07-05) — c3 PROMOTED into the runner + CI green; seed-101 fix in flight; GPU stack completed

**Promotion DONE — `--mode c3` shipped in `_rungB1c_spiking_reservoir_synaptic_readout_derisk.py` (additive; c1/c2 byte-identical).**
`_learn_Ws_spiking` (the per-role delta rule) replaces the host ridge `_fit_Ws_spiking` on the c3 path; `run_seed` dispatches
c3 in FIT + scale (`chosen_scale=1.0`, the learned weights self-scale); `new_route_bridge` + all four anti-cheat sites
(source-check, reservoir-lesion, WTA-lesion, syn-readout-lesion) handle c3; the output/agg blocks + `--mode` choices include
c3. A source-check `_source_learned_readout_clean()` asserts the LEARN path has NO host ridge (`linalg`/`lstsq`/`pinv`/
`_fit_Ws`) and DOES drive the spiking reservoir (`run_with_ens`) + a per-role LOCAL delta (`outer(error, ρ)`).

**c3 read params are the VALIDATED set, consistent for learn AND deploy.** The delta rule's whole point is train==deploy
read consistency (the fix for the ridge's train/deploy objective MISMATCH), so c3 pins `READ_T_STEP=18` + `N_TRAIN=35` (the
step6-validated config), overriding c2's unvalidated + ~2× slower `READ_T=30`/`N=60` defaults; the override is idempotent per
mode so pytest test-order cannot leak the c3 read window into a c1/c2 run.

**CI green (fast, CPU/numpy):** `tests/test_rungB1c_spiking_reservoir_synaptic_readout.py` — 7 fast tests PASS incl. the new
`test_learned_readout_source_clean` (LEARN path host-solve-free), `test_learned_readout_returns_c2_format` (the learned Ws is
the `{slot: (n_res+1)×n_roles}` c2 format, Dale-legal ≥0, bias/GOAL/LOCATION rows zero). Added a `@pytest.mark.slow`
`test_seed42_c3_learned_readout_GO` (verified in the final 6-seed runner pass).

**GPU stack completed (the owner's cupy install ask).** cupy 14.1.1 + the full CUDA 12.9 math stack (nvrtc/runtime/nvcc +
curand/cublas/cusparse/cusolver/cufft/nvjitlink) installed; GTX 1660 Ti visible; kernels compile; `SimulationBridge` builds +
steps + runs GPU RNG on both a plain bridge AND the concept-pool A→W builder (`build_concept_bridge`, 2368 neurons). ⇒ the
mechanism's validated backend stays numpy, but the A→W next-arc is GPU-viable. Known gap: `build_unified_bridge` has a
cupy-path `cp_traits=None` bug (the rungB1c conversational bridge only runs on numpy; tracked, same class as the documented
`test_regions.py` cupy failures).

## 2026-07-05 — ADVERSARIAL AUDIT reframes the arc: c3 reads POSITION not ROLE (honest negative, precisely isolated); seed 101 CLOSED by temporal resolution

A 5-dimension adversarial audit (per ultracode) confirmed the host-shortcut removal is source-clean but caught two committed
defects: **D2** (`_source_learned_readout_clean()` was defined but NEVER called → the c3 verdict wouldn't catch a silent ridge
fallback in the LEARN path — **FIXED**, now AND-ed into the c3 `synaptic_source_clean` verdict), and the load-bearing **D1**:
the GO test set is canonical SVO only, where a content word's role is largely predictable from its **position**, so "5/6 at
18/18" does not prove the read-out reads grammatical **role**. (It also self-corrected: two auditors cited the scratchpad
`step6_full.py` with a "circular validation vs host ridge" defect that is a step6 artifact, NOT the committed runner, which
scores vs true test labels — the synthesis re-verified against shipped code.)

**The decisive objrel test (the real question) — HONEST NEGATIVE, and precisely isolated.** The object-relative construction
(`the PAT that the AGT V`: slot0 = THEME, not AGENT) makes role ≠ position; it is in `_TRAIN_KINDS`, so the delta rule *can*
learn to discriminate it. Scored per-slot vs TRUE roles (not the host argmax), held-out (`step7_objrel.py`,
`step7_signed_isolation.py`, `step7_ridge_spiking.py`):

| read-out | deploy | canonical | **objrel** | objrel slot0 (THEME) |
|---|---|---|---|---|
| ridge Ws (signed) | linear argmax | 1.00 | **1.00** | 1.00 |
| ridge Ws (positive-**shifted**) | linear argmax | 1.00 | **1.00** | 1.00 |
| ridge Ws (positive-shifted) | **SPIKING WTA** | 0.97 | **0.03** | 0.00 |
| learned delta-rule W | SPIKING WTA | 1.00 | **0.00** | 0.00 |

⇒ The reservoir **feature encodes objrel** (a linear read-out solves it 100%, *even positive-shifted* — so the Dale-positive
constraint is NOT the wall), but the **positive SPIKING WTA deploy destroys the structural read** — even with the CORRECT
ridge structural weights (0.03). The wall is the **ignition-order winner failing to resolve the subtle structural margin under
the positive-shift + `WS_ENS_FLOOR` common-mode baseline** (which the shift-invariant *linear* argmax cancels). Canonical
(strong position margin) survives; objrel (subtle structural margin) does not. This is the project's documented **common-mode
/ opponency / rate-code / point-neuron-limit** family (the composer's signed-difference SNR wall → FHRR-phasor pivot;
Mikulasch-Priesemann). **Honest scope:** the CANONICAL conversational task (the actual production use case — SVO facts) is
position-solvable, so the c3 read-out *works* for it; genuine structural (non-local role-from-form) reading is the harder open
capability → **research gate dispatched** (ranked cheap-first surpass: mean-subtracted / divisively-normalized spiking WTA, or
opponent ON/OFF signed deploy).

**SURPASS attempt (research-gate-driven) — NO generalizing surpass via a FIXED read-out; the multi-seed test caught a seed-42
overclaim.** The research gate verdict was "surpassable cheaply, runner-side"; its #1 cheap fixes were empirically REFUTED
(low-floor sweep 150→15: objrel stays 0.00 — the shift pedestal, not the floor, dominates; WTA-competition-lesion: objrel 0.00,
canonical drops to 0.67). The **signed conductance** read-out (`SignedReadout`: Wp exc + Wn inh relay → net drive = `Ws@f`
signed, no positive-shift pedestal; low floor) looked like a surpass on **seed 42** (objrel slot0 0.75 vs positive 0.42) — but
the **multi-seed test (44/100) REFUTED it**: objrel slot0 = **0.75 / 0.00 / 0.50** across 42/44/100, and the low-floor signed
harness is *degraded for canonical on every seed* (seed 44 canonical 0.28–0.33 for BOTH positive and signed — the harness
itself can't do canonical there). ⇒ **the fixed signed conductance is OPERATING-POINT/SEED-FRAGILE and does NOT generalize —
the exact overfit pattern the earlier signed arc documented and retracted** (`2026-07-04-conductance-domain-signed-readout-SURPASS.md`).
The seed-42 0.92 was operating-point-lucky; the honest read is a MULTIPLY-CONFIRMED boundary: the point-neuron spiking read of
the subtle objrel structural margin is unreliable across reservoir draws for EVERY fixed read-out tried (positive-shift WTA,
low-floor, no-competition, delta, signed conductance — fail or overfit). The FEATURE robustly encodes objrel (shift-invariant
linear argmax 100% every seed); the SPIKING deploy does not. NOT the irreducible Mikulasch-Priesemann decorrelation wall (the
info is present + linearly separable) — it is the **seed-adaptive-read** frontier the earlier arc named. **The precise open
frontier (uncertain, a genuine new mechanism — NOT claimed):** a **LEARNED SIGNED read** — the delta rule (which per-draw
adapts, and already generalized the CANONICAL read 6/6) extended to signed conductance-domain delivery, so it learns a signed
structural read that adapts to each draw's operating point. Whether learned-signed generalizes objrel where fixed-signed
overfit is the open question — the specified next arc, to be de-risked before any claim. Probes:
`research/findings/raw/signed_conductance/step7_{objrel,signed_isolation,ridge_spiking,objrel_lowfloor,objrel_nocompete,objrel_signed_conductance}.py`.

**Seed 101 CLOSED — genuinely, by TEMPORAL RESOLUTION (not the confound).** `READ_T=30` (the c2 CRUX window; c3 had used
step6's speed-compromise T=18) closes seed 101 → **18/18 at the DEFAULT reservoir position** (no shift): more temporal
spike-samples resolve the marginal WTA slots. `C3_READ_T_STEP` is now **30**. (The P160/P240 "fixes" were the reservoir-shift
CONFOUND, discarded.) This closes the canonical seed-101 residual with a principled position-independent lever.

## Honest scope / next (updated 2026-07-05)
- **Seed 101 → clean 6/6 (open; two hypotheses REFUTED).** (1) The original "under-training" hypothesis is refuted: E20 made
  101 WORSE (12/18 = overfitting on the small N). (2) **The "population lever (P160) closes 101 → 18/18" result is a CONFOUND,
  not a genuine fix — caught by a control check before it was baked in.** The `WTA_P_C2=160` monkeypatch did NOT widen the
  ensembles (they stayed 80-wide: `wire_wta_c2`'s `P=WTA_P_C2` is a def-time-frozen default arg); it only enlarged the WTA
  SLICE (`ROLE_WTA_N_C2` 280→520), which shifted the reservoir slice (base 2478→2758) onto different neuron indices → a
  different (luckier) per-neuron Izhikevich heterogeneity draw for seed 101. So the "fix" is a reservoir-POSITION re-draw
  artifact (picking a non-degraded draw by luck), NOT population-code resolution — and it does NOT generalize (a new unseen
  seed could be degraded at any position). Diagnosis: **seed 101's 14/18 is a DEGRADED-RESERVOIR-DRAW problem** — its specific
  heterogeneity at the default position under-separates the roles.
  - **Legit levers (no reservoir-position confound), status:** more DATA (N60/N80/N120) + smaller η(0.02) are running at the
    standard P80 position (they change training, not neuron count → reservoir unshifted) — a clean test of "can the delta rule
    extract enough from the degraded feature with more samples." A GENUINE population lever (actually 160-wide ensembles) OR a
    bigger reservoir (`RES_N`↑, richer LSM basis) are principled but BOTH also shift the reservoir position, so they must be
    ISOLATED from the position effect (pad to hold the reservoir base fixed) before either can be claimed.
  - If no legit lever closes 101 genuinely, this is a confirmed degraded-draw BOUNDARY → the research gate fires (a biological
    mechanism for read-out robust to a degraded/heterogeneous reservoir draw: feature normalization / reservoir homeostasis /
    genuine population averaging), per the no-defer directive. The 5/6 GENUINE GO (42/43/44/100/102 at 18/18, standard P80
    config) stands; only seed 101's fix is open.
  - **Also fixed this session:** a c3 integration bug — `_build_wired_bridge` routed c3 to c1's small P=20 WTA (`else` branch)
    instead of the P=80 c2 WTA the c3 delta rule LEARNS; c3 now shares the c2 WTA (matching step6, which built mode="c2").
- **Anti-cheats — CONFIRMED CLEAN, 2-seed (42/44).** `LEARNED 18/18 | scramble 0 | global 6 | syn-lesion 6` on BOTH seeds
  (chance = 6/18): the deranged-label scramble collapses to 0 (learning is role-specific, not a position artifact); the
  FAITHFUL global-scalar reward (R=±1 uniform to all roles, not the degenerate zero-sum mean) sits at chance (the per-role-
  LOCAL credit is load-bearing — reproduces the project's documented global-scalar failure); the syn-readout lesion (zero
  res2ens after learning) collapses to chance (the learned read-out synapses ARE the read-out). Plus source-clean (both the
  SELECT path and the LEARN path host-solve-free, D2 wired). ⇒ the delta-rule mechanism is genuine, not an artifact.
- Aligned with the project master goals: everything on the ONE spiking substrate, LEARNED (no host shortcut), the
  learns-and-grows artificial-life direction.
