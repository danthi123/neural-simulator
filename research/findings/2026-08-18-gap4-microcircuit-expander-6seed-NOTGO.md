---
type: finding
status: contributing
date: 2026-08-18
mechanism: deep-credit-on-spikes
runner: research/runners/_gap4_representable_forward_plus_credit_derisk.py
artifacts:
  - research/findings/raw/gap4/rep_fwd_credit_xor_micro_expander_6seed.json
  - research/findings/raw/gap4/rep_fwd_credit_np3_micro_expander_6seed.json
  - research/findings/raw/gap4/rep_fwd_credit_np3_micro_raw_6seed.json
---

# gap#4: the Sacramento-Senn dendritic MICROCIRCUIT under the coincidence-plateau reliable expander does NOT rescue deep credit (6-seed NOT-GO) — a representable forward (oracle 0.94) is not enough; the on-bridge credit path stays at chance

<!--derived-->
**One-line verdict.** The 2026-07-25 coincidence-plateau expander made the on-bridge forward held-out linearly
decodable, and the 2026-08-02 verdict then localised the residual to "the wall is the FORWARD/credit rule at
scale, not task decodability". This session ran the last untried lever on that wall: a Sacramento-Senn apical/basal
**microcircuit** (learned self-predicting feedback) as the credit rule, on top of the reliable expander, 6-seed.
It is a **NOT-GO**. On XOR the expander does make the code representable (backprop oracle 0.94), but the on-bridge
e-prop credit rule under the microcircuit lands at chance — eprop-inherit ~ 0.52, frozen-reservoir ~ 0.50, chance
~ 0.52 — so directed credit buys nothing over a frozen random reservoir and never reaches the oracle. On the native
6-way `n_prop=3` task every arm (raw and expander) sits at its 0.167 chance floor. gap#4 stays a **mapped wall,
deprioritized** (the tractable path is LEARNED rate feedback, banked separately; gap#4 is not the conversation
blocker per the 2026-08-11 arc-summary). No `sim/` edit — additive runner flags, `SIM_BACKEND=numpy`.

## Result 1 — XOR, expander mode, microcircuit credit, 6 seeds

<!--derived-->
Artifact: `research/findings/raw/gap4/rep_fwd_credit_xor_micro_expander_6seed.json` (6 seeds 42/43/44/100/101/102,
chance ~0.52, `SIGNAL: false`). Per-seed inheritance accuracy (higher = better; oracle = ideal backprop readout,
frozen = untrained random-reservoir readout, eprop = the on-bridge microcircuit credit rule):

| seed | oracle | eprop | frozen | trains_the_task |
|---|---:|---:|---:|---|
| 42  | 0.9499 | 0.5460 | 0.5320 | False |
| 43  | 0.9276 | 0.4930 | 0.4735 | False |
| 44  | 0.9220 | 0.4763 | 0.4763 | False |
| 100 | 0.9582 | 0.5237 | 0.5125 | False |
| 101 | 0.9415 | 0.5822 | 0.5014 | **True** |
| 102 | 0.9415 | 0.4986 | 0.4763 | False |

<!--derived-->
Aggregate (from the artifact's `aggregate` block): `expander_oracle_inherit` 0.9401, `expander_eprop_inherit`
0.5200, `expander_frozen_inherit` 0.4954, against `chance` 0.5241; controls clean (`expander_permuted` 0.4930,
`expander_shuffle_dfa` 0.4768); `expander_codon_reproducibility` 1.000; `expander_trains_the_task_all_seeds` false
(only seed 101 trains, 1/6). Microcircuit health: `microcircuit_selfpred_cos` 0.5700, `microcircuit_apical_silent_ratio`
0.8894 — the apical/feedback compartment is silent ~89% of the time and the self-prediction cosine is weak, which is
the mechanistic reason the credit path carries almost no signal.

<!--derived-->
**Reading it honestly.** The decisive comparison is eprop ~ frozen ~ chance while oracle >> chance. Two facts follow,
and both say NOT-GO: (a) eprop does not beat a frozen reservoir, so the microcircuit's directed feedback assigns no
usable deep credit; (b) eprop sits far below the oracle, so a linearly-separable code exists but the spiking on-bridge
rule cannot extract it. Note this CORRECTS a loose framing worth flagging: because frozen inherit is ~0.50 ≈ chance
(not high), the expander did NOT make XOR "reservoir-solvable" here — a solvable reservoir would read high, not chance.
The `deep_credit_share` aggregate (raw +nan, expander +1.09) is computed on near-chance arms whose spread is noise, so
it is not evidence of credit and is not used as the verdict. `deep_credit_share` also lands `NaN`/wild per seed for the
same reason (0/0). Verdict text is preserved in the artifact's `verdict_note` field.

## Result 2 — native 6-way task (n_prop=3), raw and expander arms both at chance

<!--derived-->
Artifacts: `rep_fwd_credit_np3_micro_raw_6seed.json` and `rep_fwd_credit_np3_micro_expander_6seed.json` (6 seeds,
chance 0.1667, `SIGNAL: false` on both). The two runs each populated one arm:

| run | eprop-inherit | frozen-inherit | chance | controls |
|---|---:|---:|---:|---|
| raw arm      | 0.1883 | 0.1481 | 0.1667 | clean |
| expander arm | 0.1512 | 0.1914 | 0.1667 | clean |

<!--derived-->
Every arm sits within noise of the 0.167 floor; no arm separates. So on the actual gap#4 task (not the XOR proxy)
the microcircuit-plus-expander stack produces no learning at all, raw or expanded. `raw_deep_share` reads +1.583 but
again over near-chance arms — a ratio on ~0 denominator, not a signal.

## Why this is a NOT-GO and not a next lever <!--derived-->

<!--derived-->
This extends, and is consistent with, the prior record rather than re-deriving it:

- `2026-07-25-gap4-forward-representability-SURPASSED-ON-BRIDGE-coincidence-plateau-reliable-expander-6seed-GO.md`
  — the expander surpassed the FORWARD boundary but explicitly "does not by itself deliver gap#4 ACCURACY". This
  finding tests whether adding the microcircuit credit rule closes that gap. It does not.
- `2026-08-02-gap4-production-bridge-deep-credit-NOT-closed-by-XOR-the-wall-is-deeper-than-task-decodability-on-bridge-forward.md`
  — on the production Izhikevich bridge, e-prop cannot train XOR even where the oracle solves it, so "the wall is the
  forward, not the rule". The microcircuit was the remaining candidate for a stronger credit path; here it too fails,
  with the microcircuit's own apical-silent/self-predict health metrics showing the credit signal never propagates.

<!--derived-->
The `microcircuit` credit rule is therefore banked TESTED-NEGATIVE for gap#4 on the spiking on-bridge (this document
is the refuting evidence; it is not being proposed as a future surpass). The next step is NOT a further dendritic /
two-compartment variant — that class is already negative. The tractable direction, banked separately, is LEARNED
rate feedback; and per the artifact verdict's own column-read, if the microcircuit path were ever revisited it would
first need XOR made representable at `n_prop=3` (a stronger `--xor-encoding` / larger `--n-col`) so the credit rule
can be tested against a forward the oracle can solve. Both are lower priority than the mouth/Gate-B work: gap#4 is a
mapped wall, not the current blocker.

## Reproduce <!--derived-->

```bash
SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_representable_forward_plus_credit_derisk \
  --task-xor --act-th 3 --microcircuit --mode expander --seeds 42 43 44 100 101 102 \
  --epochs 60 --train-subsample 160 \
  --out research/findings/raw/gap4/rep_fwd_credit_xor_micro_expander_6seed.json
```

GO gate (not met): `deep_credit_share > 0.3` AND `frozen < eprop` AND `eprop` beats chance on the majority of seeds
AND controls clean. Here eprop ~ frozen ~ chance, so the gate fails on every clause but the controls.
