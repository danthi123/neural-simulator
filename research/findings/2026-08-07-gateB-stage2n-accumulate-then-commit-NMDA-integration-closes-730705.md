---
type: finding
status: no-go
date: 2026-08-07
mechanism: gateB-stage2n-FIXF-accumulate-then-commit-NMDA-integration
backend: numpy
runner: research/runners/_vocal_gateb_stage2n_accumulate_commit.py
builds-on: 2026-08-07-gateB-stage2m-bg-output-homeostat-inverts-thalamus-but-necessary-not-sufficient.md
grounded-in:
  - research/findings/2026-06-06-N6-accumulator-commit-readout-BOUNDARY.md
  - research/findings/2026-06-06-action-selection-readout-deep-research.md
artifacts:
  - research/findings/raw/gateb_stage2n_accumulate_commit/smoke_730705_numpy.json
  - research/findings/raw/gateb_stage2n_accumulate_commit/diag_730705.txt
---

# Gate B Stage 2n: FIX F ports the record-grounded accumulate-then-commit readout (NMDA-slow recurrent integration on the commit pools) onto the vocal Gate B commit stage — 730705 EXPRESSES action 1 at test (test_rate_c1 0→1.0), but the de-latch operating point required to let commit_1 enter FAILS the acquisition-lesion contingency control (a SHORTCUT). 730705 stands as a characterized heterogeneity boundary; Gate B holds at ≥5/6.

## Verdict (NO-GO — outcome (ii): flips, but an anti-cheat fails; a precisely-located shortcut)

FIX F ports the EXISTING accumulate-then-commit readout onto the vocal commit stage: NMDA-SLOW
recurrent self-excitation on commit_0/commit_1 (Wang 2002, τ_decay=100ms; the SAME
`cfg.enable_nmda` + per-neuron `cp_nmda_neuron_mask` mechanism the nav N6 readout used) so the
commit INTEGRATES a sustained drive, plus a GENTLE (lowered) commit_fs cross-inhibition so the
losing pool can ENTER and be integrated. Additive, DEFAULT-OFF, NO `sim/` edit, numpy-authoritative.

The full train→test smoke on 730705 (`smoke_730705_numpy.json`, fix_e off) shows:

| quantity | Stage 2k base | FIX F on |
|---|---|---|
| `count_c1` (train selections of action 1) | [37, 3] | [0, 40] |
| **`test_rate_c1`** (trained target=1 expresses action 1) | **0.0** | **1.0** ← flips |
| `test_rate_c0` (trained target=0 expresses action 0) | (≈1.0) | **0.1** ← breaks |
| `D_contingent` | 0.0 | 0.1 |
| `steer` (needs D_contingent≥0.30 & gap≥0.20) | false | **false** |

**730705's action 1 DOES express at test** (test_rate_c1 0→1.0) — the flip is real in the full
pipeline, not just the isolated commit cascade. But two anti-cheats fail:

1. **Acquisition-lesion FAILS (the decisive one).** On an UNTRAINED bridge (`plastic_d1=False`)
   with FIX F on, action 1 wins at test anyway: `acq_lesion_action1_does_not_win=false`,
   `p_action0_target1=0.0`, `test_rate_target1=1.0`, `D_contingent_acq_lesion=0.0`. The contingency
   is NOT owned by D1 plasticity.
2. **Target-0 contingency breaks.** `test_rate_c0` collapses 1.0→0.1 and `steer=false` — FIX F
   imposes a channel-1 bias rather than a contingent steering policy.

## Root cause — the FIX C × FIX F interaction (localized, `confirm2n`, numpy)

The de-latch converts FIX C's *target-blind* gain into an *unlearned* decision. On the UNTRAINED
(acq-lesion) bridge, target=1, 8-trial test cascade:

| config | str_d1 | commit | motor | action-1 |
|---|---|---|---|---|
| FIX C only (no FIX F) | [104,124] | [452,0] | [860,0] | **0/8** (veto holds) |
| **FIX C + FIX F (xinh×0.1)** | [104,124] | [233,416] | [463,624] | **7/8** ← shortcut |
| FIX F only (no FIX C) | [103,1] | [479,0] | [849,0] | 0/8 (no drive) |

FIX C (Stage 2j) applies a ×3 intrinsic-gain homeostat to WAKE the "dead" str_d1_1 MSN so it can
be selected, rewarded and LEARNED. Before FIX F, the commit veto correctly required a *learned*
(strong, ~286-spike) str_d1_1 to overcome thal_0's transient temporal head-start — so the mere
FIX-C gain (124 spikes, unlearned) stayed vetoed → action 0. **FIX F's de-latch removes exactly
that veto**, so the FIX-C-gained-but-unlearned channel now expresses. On the trained bridge the
learned str_d1_1 (286) also expresses, but the de-latch cannot distinguish 286-from-learning from
124-from-gain-alone → not contingency-preserving.

## Why there is no legitimate operating window (grounded, measured)

- **The gentle de-latch is REQUIRED to flip and is what breaks legitimacy.** commit_1 stays at 0
  spikes at xinhib ×0.25 / ×0.5 (never enters); only ×0.1 lets it fire. So the flip needs the
  permissive veto that also lets the unlearned FIX-C channel through.
- **NMDA integration is real but not separable here.** On the TRAINED bridge, NMDA+de-latch flips
  even without FIX E while the thalamic AGGREGATE still favors action 0 (thal [273,215],
  commit [184,504]) — genuine integration of the sustained-correct over the transient head-start
  (`diag_730705.txt`). But the same de-latch it needs is the illegitimacy source, so integration
  cannot close 730705 *legitimately* at any operating point tested.
- **Not a clean no-op on dev.** The FIX-F recruitment gate (the target-blind FIX-E
  extreme-asymmetry detector) fires on 5/12 seeds (730601, 730603, 730606, 730705, 730706), so
  FIX F is NOT byte-identical on 3 dev seeds — it would additionally require dev-regression clearance.

## Anti-cheat that PASSES

**Byte-identical when off.** `byte_identity_fixf_off` = true on 730703 and 730705 (mismatch `{}`)
— FIX F OFF reproduces the Stage-2k base exactly, so the Stage-2j/2k GO is unaffected.

## Banked method + honest boundary

BANKED (refuted at this operating point): porting the N6 NMDA-slow accumulate-then-commit readout
onto the vocal commit stage makes 730705 EXPRESS action 1 at test, and the NMDA integration
genuinely overcomes thal_0's temporal head-start — but the gentle cross-inhibition it requires to
let commit_1 enter also unmasks FIX C's target-blind intrinsic gain, so an untrained bridge picks
action 1 (acquisition-lesion fails). The Gate B commit veto is doing legitimate work: it enforces
that only a LEARNED str_d1 overcomes the head-start. **730705 remains a characterized
heterogeneity boundary — its str_d1 policy is correctly learned (str_d1=[104,286]) but its extreme
thal_0 initial-condition head-start cannot be overcome at the commit stage without a de-latch that
sacrifices the learning-legitimacy the veto provides.** Gate B stands at ≥5/6 (a first-class result).

The true upstream residual (per Stage 2l/2m and confirmed here) is the str_d1 baseline firing
asymmetry + thal_0 membrane head-start that FIX C wakes but FIX E only partially inverts; a
legitimate close would have to equalize the thal ENTRY STATE (the Stage-2m TRN-like onset reset
lead) or re-shape the head-start upstream of the commit, WITHOUT a commit veto that also passes
unlearned drive — not the commit-integration operating point tried here.

## Reproduce (numpy, orphan-proof)

```bash
export PYTHONPATH=$PWD SIM_BACKEND=numpy
# The full train→test smoke (byte-identity off + 730705 flip + acq-lesion legitimacy):
.venv/bin/python -m research.runners._vocal_gateb_stage2n_accumulate_commit --mode smoke \
  --smoke-seeds 730705 \
  --out research/findings/raw/gateb_stage2n_accumulate_commit/smoke_730705_numpy.json
# Isolated commit cascade (INTEGRATION evidence vs the head-start latch):
.venv/bin/python -m research.runners._vocal_gateb_stage2n_accumulate_commit --mode diag \
  --diag-seed 730705
# Byte-identity when off (all_byte_identical must be true -> GO protected):
.venv/bin/python -m research.runners._vocal_gateb_stage2n_accumulate_commit --mode byte
```
