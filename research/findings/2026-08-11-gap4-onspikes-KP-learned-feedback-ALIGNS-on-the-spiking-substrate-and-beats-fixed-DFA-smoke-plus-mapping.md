---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes — porting the rate KP learned-feedback surpass onto the on-bridge e-prop LIF SNN, and MEASURING cos(Y,Wᵀ) feedback-alignment on spikes (the rate finding's central read-out, unmeasured on spikes until now)
lane: gap#4 / deep-credit
verdict: SMOKE GO (seed 42, N=2) + MAPPING. Transport-free Kolen-Pollack LEARNED feedback ALIGNS on the SPIKING substrate (deep-layer cos(Y_l,W_{l+1}ᵀ) rises from ~0 to positive through training — the rate finding's transport-free alignment signature, measured on spikes for the first time) AND beats fixed-DFA on a depth-2 spiking XOR (all 7 Verdict preconditions + the control PASS at seed 42). This directly answers the open "does KP align on spikes" question the 2026-08-01 multihop-negative could only INFER (it inferred NO; direct measurement shows YES at depth-2). RESIDUAL: on a REDUNDANT-depth-3 XOR the deep credit is dead for ALL arms and KP does not revive it; and a VALID depth-3-OBLIGATORY spiking instrument does not yet exist (the tent^k FIT + per-seed ceiling-gating that made the rate question decisive was never ported). SMOKE not headline (1 seed) — 6-seed command provided; the numbers are in the body against the smoke artifacts.
seeds: [42 smoke; 6-seed command below]
seed-waiver: labelled 1-seed SMOKE de-risk (not a generalization headline) — the EXACT 6-seed command is provided in the Scope section; the wall-ledger flip is gated on that 6-seed run, not on this finding.
artifacts:
  - research/findings/raw/gap4/realspikes/onspikes_kp_align_N2_ep200_seed42.json
  - research/findings/raw/gap4/realspikes/onspikes_kp_align_N2_ep800_seed42.json
  - research/findings/raw/gap4/realspikes/onspikes_kp_align_N3_ep400_seed42.json
  - research/findings/raw/gap4/realspikes/onspikes_kp_align_N2_ep200_6seed.json
runner: research/runners/_gap4_onspikes_kp_align_derisk.py
---

<!--derived-->
**⭐ 6-SEED CONFIRMATION (coordinator-run, `research/findings/raw/gap4/realspikes/onspikes_kp_align_N2_ep200_6seed.json`):
status GO across seeds 42/43/44/100/101/102.** Transport-free KP-learned feedback beats fixed-DFA by **+0.028** (mean
`kp_plus_over_fixed`) and its deep-layer feedback-alignment RISES by **+0.259** through training (`deep_align_delta`,
`aligning_sign=kp_plus`) — the rate finding's transport-free alignment signature, now confirmed on the spiking substrate
at 6 seeds (the smoke's +0.070 seed-42 margin narrows to +0.028 at 6 seeds but stays positive with the alignment intact).
This CONFIRMS the smoke: on the trainable LIF SNN, learned feedback aligns and out-credits fixed feedback WITHOUT weight
transport — the gap#4 deep-credit surpass reaches the one spiking substrate at de-risk level. Residual unchanged: no
depth-3-OBLIGATORY spiking instrument yet (the depth-3 rung stays UNDEFINED, not fabricated).

# gap#4 ON-SPIKES — Kolen-Pollack learned feedback ALIGNS on the spiking substrate and beats fixed-DFA at depth-2 (smoke GO) — plus the rate→spikes mapping and the depth-3 residual

The rate de-risk (`research/runners/_gap4_learned_feedback_derisk.py`, finding
`2026-08-11-gap4-learned-feedback-KP-reaches-the-3rd-hidden-layer-...`) showed transport-free Kolen-Pollack LEARNED
feedback REACHES the 3rd hidden layer where fixed-random DFA cannot, its central signature being cos(G_l, W_lᵀ) RISING
from ~0 to ~0.83 through training (co-adapted, never copied). THE END TARGET is that surpass on the ONE SPIKING
substrate. This de-risk maps the rule onto the on-bridge e-prop LIF SNN and MEASURES the alignment on spikes.

## The mapping (rate G ↔ spiking Y) — how KP ports to on-bridge e-prop

The on-bridge e-prop KP rule was ALREADY built: `research/runners/_gap4_bptt_snn_chained_fa_transport_free_derisk.py`
(`_chained_fa_grads`, arm `chained_fa_kp`) puts chained transport-free KP-learned feedback on the BPTT-viable LIF SNN
(reuse-by-import of `sim/bptt_snn_gpu`). Its `chained_fa` arm IS fixed-DFA (Y frozen at init = the freeze-G lever
endpoint); `kp_over_fixed_fa` is the learned-vs-fixed margin. So the MINIMAL change to make the feedback LEARNED (KP)
instead of FIXED (DFA) is flipping the arm from `chained_fa` (Y frozen) to `chained_fa_kp` (Y co-adapted by the matched
transport-free outer product). No substrate hook is needed — **NO `sim/` edit.**

| element | RATE (`_gap4_learned_feedback_derisk.py`) | SPIKING (on-bridge e-prop LIF SNN) |
|---|---|---|
| forward unit | rate MLP `a = φ(W·a + b)` | LIF `sim/bptt_snn_gpu.LIFLayerXP`: `v←leak·v(1−s)+xW`, spike `s=(v≥θ)` |
| activation deriv | `φ'(a)` | σ′(v−θ) = atan surrogate, RELATIVE-normalized (mean 1) |
| credit path | `e_l = (e_{l+1} @ G_l)·φ'(a_l)` | `e_l = (e_{l+1} @ Y_l)·σ′(v−θ)_l`, per timestep |
| presyn factor | activation `a_l` | eligibility trace `eps_l = α_leak·eps_l + z_l` (Bellec 2020 e-prop; the LIF leak-recurrence) |
| feedback matrix | `G_l` replaces `W_lᵀ`, separate random init | `Y_l` replaces `W_{l+1}ᵀ`, separate random stream (`_make_feedback`, seed+8888) |
| KP feedback update | `G_l −= step_lᵀ` (SAME Adam step as W, transposed) | `Y_l += kp_sign·lr·(kp_lr·outer − kp_decay·Y_l)`, `outer=(Σ_t e_{l+1}ᵀ @ z_l)/(B·T)` |
| transport-free | credit reads G not W; KP uses the activity-derived step | credit reads Y not W; KP reads post/pre spikes only, never a forward W |
| alignment signature | cos(G_l, W_lᵀ): ~0 → ~0.83 | cos(Y_l, W_{l+1}ᵀ): **MEASURED HERE for the first time on spikes** |

## The smoke result (seed 42; `research/findings/raw/gap4/realspikes/onspikes_kp_align_N2_ep200_seed42.json`)

<!--derived-->
Depth-2 XOR→threshold task (the rate-overturn task; NOT linearly reservoir-decodable), on the trainable LIF SNN
(hidden 32, T 24, epochs 200, subsample 800, numpy/CPU). Six arms on the SAME forward + task: BPTT ceiling, frozen
reservoir, fixed-DFA (Y frozen), KP+ (committed-runner sign), KP− (rate-matched decrement sign), permuted (anti-cheat).

| arm | inherit acc | deep cos(Y,Wᵀ) init→final |
|---|---|---|
| BPTT ceiling (hidden 96) | 0.744 | — |
| **KP+ (learned, deployed sign)** | **0.877** | **−0.000 → +0.276** |
| KP− (learned, rate-matched sign) | 0.836 | −0.000 → +0.385 |
| fixed-DFA (Y frozen = freeze-G) | 0.808 | frozen |
| frozen reservoir | 0.513 | — |
| permuted (anti-cheat floor) | 0.451 | — |
| chance (majority class) | 0.549 | — |

<!--derived-->
**Transport-free KP learned feedback ALIGNS on spikes and beats fixed-DFA.** The deployed-sign KP (KP+) reaches 0.877
vs fixed-DFA 0.808 (**+0.070**), and its deep-layer feedback cos(Y_l, W_{l+1}ᵀ) RISES from ~0 to +0.276 through
training — the rate finding's transport-free co-adaptation signature, on the spiking substrate. Both KP arms and
fixed-DFA crush the frozen reservoir (0.513) and the permuted floor (0.451). All 7 Verdict preconditions PASS →
**status GO** at seed 42: BPTT ceiling exists (0.744 > chance+0.15); permuted at/below chance; transport-free (init
max|cos(Y,Wᵀ)| = 0.044 < 0.8, no Y byte-equal any W or Wᵀ); the lever moved Y (fixed-DFA left it frozen); KP aligns
(deep cos delta +0.276 > 0.05); KP beats fixed-DFA (+0.070 > 0.02); the control confirms KP's fit differs from
fixed-DFA's (|sep| = 0.070).

## The sign question — resolved by measurement (my "sign bug" hypothesis was REFUTED)

<!--derived-->
The mapping surfaced an apparent sign discrepancy: the rate rule DECREMENTS G by the same step W is decremented by
(`G −= step^T`, same sign → cos rises), while the committed spiking runner ADDS the KP outer (`Y += lr·kp_lr·outer`)
while the forward SUBTRACTS (`W −= lr·wg`) — the OPPOSITE sign, which naively predicts anti-alignment. I tested BOTH
signs rather than asserting a bug. Measurement: the committed `+` sign (KP+) both ALIGNS (deep cos → +0.276) AND
performs BEST (0.877); the rate-matched `−` sign (KP−) aligns MORE (deep cos → +0.385 at ep200, +0.404 at ep800) but
OVERSHOOTS and HURTS accuracy at the longer budget (0.752 at ep800, BELOW fixed-DFA). So **stronger feedback alignment
does not monotonically map to better accuracy on spikes**, and the deployed `+` sign is the right one — the "sign bug"
is refuted. (Recorded because it is exactly the silent-failure class: an assertion would have "fixed" a non-bug.)

## The budget test — depth-2 alignment is saturated by epoch 200 (`..._N2_ep800_seed42.json`)

<!--derived-->
The rate finding showed KP feedback-alignment converges SLOWER than backprop (under-trains at low budget: rate seed-42
gap-close −6% at 3000 epochs → +62% at 8000). On the depth-2 spiking instrument the alignment is already achieved by
epoch 200: raising to 800 epochs leaves KP+ flat (0.877 → 0.875; +0.070 → +0.064 over fixed-DFA; deep cos +0.276 →
+0.277) and the depth-2 GO holds at BOTH budgets (ep800 also status GO). So the "needs more budget" effect lives at
DEPTH / harder depth-separation, not at depth-2 — consistent with the rate result, where the budget mattered for
reaching the DEEPEST layer.

## The residual, honestly — depth-3 (`research/findings/raw/gap4/realspikes/onspikes_kp_align_N3_ep400_seed42.json`)

<!--derived-->
On a REDUNDANT-depth-3 XOR (XOR obligates only depth-2, so the 3rd hidden layer is redundant), the deep credit is DEAD
for ALL arms: fixed-DFA = KP+ = KP− = 0.451 = the permuted floor (kp_over_fixed = 0.000), all BELOW the frozen
reservoir (0.565). Strikingly, the deepest feedback of KP+ ANTI-aligns (deep cos → −0.38) while KP− aligns strongly
(deep cos → +0.48) — yet NEITHER translates to accuracy, because the deepest layer is reservoir-redundant on this task
and the credit through it carries no label-usable selectivity (the 2026-08-01 "credit attenuates ~3× through depth /
label-agnostic lift" signature). The runner records this HONESTLY: the KP-vs-fixed accuracy lever does not move (KP ==
fixed-DFA), so the verdict is UNDEFINED (no verdict earned — not a fabricated GO/NO-GO). This is the honest boundary
point: KP's feedback CAN be driven to align on spikes at depth, but on a task where the depth is REDUNDANT the aligned
credit does nothing.

The load-bearing residual is therefore an INSTRUMENT gap, not a demonstrated capability wall: **there is no VALID
depth-3-OBLIGATORY spiking instrument yet.** The runner's own `--task-hier3` (the intended depth-3 compositional task)
FAILS its own stage0 depth-3 gate (l3_train ≈ 0.64, l3_inherit ≈ chance across 17 tuned configs × 2 seeds), and the
tent^k FIT target + PER-SEED ceiling-gating that made the RATE depth question decisive was never ported to spikes. The
depth-2 port GOes; the depth-3-obligatory port needs the instrument first.

## Anti-cheats + lever (all executed via tools.lab / Verdict, not asserted in prose)

<!--derived-->
- **Transport-free:** init max|cos(Y_l, W_{l+1}ᵀ)| = 0.044 < 0.8 (separate random stream, not a Wᵀ copy); no Y_l is
  byte-equal any forward W or its transpose (`_no_weight_transport`); the KP update reads only post/pre spikes + Y.
- **Freeze-G lever:** fixed-DFA (Y frozen) is KP with feedback-learning OFF; (KP − fixed-DFA) = +0.070 is the
  learned-feedback win; `lever()` confirms KP moved Y every step while fixed-DFA left it frozen.
- **permuted:** shuffled-label FA → 0.451 (at/below the majority-class chance 0.549); the signal is label-attributable.
- **ceiling:** BPTT (own width 96) solves the task (0.744 > chance+0.15) — a trainable-substrate ceiling exists.

## Scope + next (per THE LAW — the capability is OPEN, headline pending)

<!--derived-->
- **Smoke, not the headline.** 1 seed (42). The GO must survive the 6-seed set before the wall-ledger flips. EXACT
  6-seed command (self-aggregating; fan across cores for the sweep):
  `SIM_BACKEND=numpy python -m research.runners._gap4_onspikes_kp_align_derisk --seeds 42 43 44 100 101 102 --n-hidden-layers 2 --epochs 200 --bptt-epochs 300 --bptt-hidden 96 --train-subsample 800 --out research/findings/raw/gap4/realspikes/onspikes_kp_align_N2_ep200_6seed.json`
- **What this banks:** the rate KP surpass PORTS to spikes at depth-2 — transport-free learned feedback ALIGNS on the
  spiking substrate (measured, not inferred) and beats fixed-DFA — which OVERTURNS the inference in the 2026-08-01
  multihop-negative that "KP does not align on spikes." The negative there was on the RealSpikesPlateauExpander
  (columns that never somatically spike); on the BPTT-viable LIF SNN, KP aligns.
- **The next mechanism (named, not deferred):** build the VALID depth-3-OBLIGATORY spiking instrument — port the rate
  finding's per-seed ceiling-gating (score a seed only if BPTT-depth-3 solves AND BPTT-depth-2 underfits) to the LIF
  SNN, on a compositional depth-3 task that a depth-3 BPTT net CAN fit (unlike hier3/nested-XOR). Only then does the
  literal "KP reaches the 3rd hidden layer ON SPIKES" question become decidable. This smoke de-risks that build: the
  KP rule aligns and beats fixed-DFA on spikes; the remaining work is the depth-obligatory instrument, not the rule.
- **Data-only, NO `sim/` edit.** Additive runner; reuse-by-import of the LIF forward + BPTT + task. The BPTT arm uses
  Wᵀ only for the ceiling; the fixed-DFA and KP arms are fully transport-free.
