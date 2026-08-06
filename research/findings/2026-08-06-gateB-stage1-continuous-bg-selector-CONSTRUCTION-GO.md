---
type: finding
status: qualified
date: 2026-08-06
mechanism: gateB-v13-stage1-continuous-center-surround-bg-selector
backend: numpy+cupy
runner: research/runners/_vocal_gateb_stage1_selector.py
seed-waiver: CONSTRUCTION gate (mechanism existence), single-seed by design per the v13 research gate and the Stage-0 re-anchor contract — it asserts the center-surround circuit CAN produce >=2 clean bounded actions cross-backend, not a multiseed capability rate. Multiseed capability validation is Stage 2 (reward learning). Seed-robustness IS reported here honestly: the baseline substrate is robust on 730501-730504 both backends; the un-learned inter-channel WTA is seed-fragile (numpy 3/4, cupy 2/4), which Stage 2 addresses.
artifacts:
  - research/findings/raw/gateb_stage1_selector/numpy.json
  - research/findings/raw/gateb_stage1_selector/cupy.json
---

# Gate B Stage 1: continuous center-surround BG selector is a cross-backend CONSTRUCTION GO

## Verdict

**STAGE1_GO on the construction seed (730501), both backends.** All fourteen
construction-gate checks pass on NumPy and CuPy with earned verdicts (no
UNDEFINED). Artifacts + provenance sidecars:
`research/findings/raw/gateb_stage1_selector/numpy.json` and
`research/findings/raw/gateb_stage1_selector/cupy.json`.

A continuously-operating basal-ganglia selector, built on the Stage-0 autonomous
tonic-output substrate (`2026-08-06-gateB-stage0-tonic-output-seed-robustness-and-stage1-reanchor.md`),
turns competing noisy cortical proposals into repeated, temporally bounded motor
actions with **no host stop-on-winner, no reset current, and zero external drive
to GPi/SNr**. GPi/SNr runs on immutable region-scoped `intrinsic_current_pA`
from step 0. Weights are immutable at runtime (no learning; that is Stage 2).

## What was built

From the Gate A v2 populations (`research/runners/_vocal_action_selector_gate.py`),
`selector_reset` and the host GPi tonic current were removed, and pathways were
added in the v13 research-gate's mechanism order (`2026-08-04-neural-vocal-credit-gateB-v13-continuous-bg-selector-RESEARCH-GATE.md`):
**(1)** proposal/cortex to shared STN (hyperdirect hold; Nambu 2002), **(2)** GPe
to same-channel GPi/SNr (direct pallidal control). A third mechanism was required
to make the selection focused: **striatal FSI feed-forward lateral inhibition**
(each channel's proposal drives its FSI, which inhibits the OTHER channel's
near-threshold MSNs) — the center-surround surround at the striatum
(Tepper/Koos). This is a re-enable of the Gate A v1 FSI population, not a new
downstream boundary topology (v11/v12's retired decomposition); it competes in
space at the striatum, not in time at the read-out.

## The measured mechanism (genuine, not an artifact)

Single-step tracing confirmed the causal chain: proposal onset -> winner D1
fires -> winner GPi/SNr pauses -> winner motor thalamus is disinhibited and
releases -> commit -> motor; the loser channel is held clamped at every stage.
Per-window numbers (winner vs loser), construction seed:

| Backend | baseline GPi | winner GPi (paused) | loser GPi | winner thal | loser thal | winner motor | loser motor | clean/4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numpy | 47.8 Hz | 20-24 Hz | 46-48 Hz | 121-134 | 0 | 412-439 | 0 | 3 |
| cupy  | 43.2 Hz | 25-31 Hz | 38-40 Hz | 95-113 | 26-50 | 228-392 | 0-38 | 3 |

The winner GPi/SNr is roughly halved from its tonic baseline (focused pause);
STN rises early on proposal onset (hyperdirect hold, `stn_early` 25-38 Hz from 0);
after each action the circuit returns autonomously to tonic GPi output (~44-49 Hz)
and motor silence with no reset. At least two clean actions come from one
uninterrupted brain (three of four windows on both backends).

## Construction-gate checks (all pass, both backends)

tonic GPi baseline; thalamus inhibited at baseline (0 spikes); motor silent at
baseline (0 spikes); early STN rise; focused GPi pause (winner GPi < loser GPi);
winner thalamus released (>=12 spikes); competitor thalamus suppressed (winner
dominates loser >2x); one clean motor action; competitor motor suppressed
(loser <=25% of winner); autonomous return to tonic + motor silence; >=2 clean
actions; weights immutable; intrinsic immutable; zero GPi external current.

## Operating point (found by transfer-function characterisation)

GPi `intrinsic_current_pA` = 140 pA (~45 Hz tonic, in the Stage-0 40-80 Hz band)
clamps a 300 pA motor-thalamus afferent; proposal->D1 (weight 40) pauses winner
GPi ~45->~22 Hz, releasing the thalamus; hyperdirect proposal->STN (weight 2)
gives an early global hold without saturating GPi; FSI lateral inhibition
(weight 32) is at the winner-take-all operating point — the response is strongly
non-monotonic (weaker gives ties, mid-range frustrates both channels, this value
commits cleanly to one winner). All weights symmetric across channels, immutable
at runtime.

## Two disclosed, non-hidden initialisation facts

**(1) One-time settle.** A 150-step intrinsic-only settle (thalamic afferent on,
no arousal, no reset) initialises the continuously-running brain into its
baseline attractor before scoring. This is NOT a per-action reset: there is one
settle, then the two+ actions run uninterrupted with no reset between them. The
cold-start-from-rest transient it absorbs is separately measured and reported
(`coldstart_motor_spikes` ~160-170; `settle_tail_motor_spikes` = 0, i.e. silence
is reached before scoring). **(2) GPi phase init.** GPi neurons start at
desynchronised sub-threshold phases (uniform in [vr, vt]) rather than all at
rest — a continuously-running output pacemaker is never "off"; this also avoids
a synchronised first volley. Both are seeded and deterministic.

## Honest residual (why status = qualified, not unqualified GO)

The **baseline substrate** (autonomous tonic GPi output + thalamic clamp + motor
silence) is seed-robust: 0 baseline thalamus and 0 baseline motor spikes on every
seed tested (730501-730504), both backends. The **inter-channel winner-take-all**
is NOT yet seed-robust: with two symmetric proposals broken only by OU noise,
some seeds produce ties (no clear winner) in all windows — NumPy passed 3/4 of
seeds 730501-730504, CuPy 2/4. This is expected: symmetric channels selected by
noise alone are inherently sometimes-tied. The v13 research gate reopens the v10
local reward-credit question only after this construction GO — and **Stage-2
reward learning is exactly the asymmetry that breaks the tie** (it strengthens
the rewarded channel's corticostriatal policy, so selection stops depending on a
lucky noise realisation). The tie-fragility is therefore a property of the
un-learned symmetric construction, not a substrate defect, and its resolution is
the point of Stage 2.

## Exact next mechanism (Stage 2)

Reopen the v10 local reward-credit question on this continuous selector: full
fixed action windows, clean selected-route eligibility before reward, contingent
vs reward-count-matched yoked controls, acquisition/expression lesions, fresh
multiseed development + held-out phases, and same-brain convention reversal.
Reward learning is also the mechanism that is expected to make winner selection
seed-robust. Optional hardening before Stage 2: structural lesions with filed
directional predictions (intrinsic drive, D1->GPi, D2->GPe, hyperdirect
proposal->STN, GPe->GPi, striatal FSI) to confirm each pathway is load-bearing.
