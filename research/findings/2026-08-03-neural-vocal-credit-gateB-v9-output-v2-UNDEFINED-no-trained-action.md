---
type: finding
status: undefined
date: 2026-08-03
mechanism: neural-vocal-action-credit-v9-gabab-output-v2
runner: research/runners/_vocal_action_credit_gate_v9_graded_dendritic.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v9/output_v2_seed0_cupy.json
  - research/findings/raw/vocal_action_credit_gate_v9/output_v2_seed0_cupy.json.prov.json
---

# V9 output remains undefined because the trained action was never expressed

<!--derived-->
**Verdict: UNDEFINED.** The corrected CuPy protocol completed all six
conditions and passed every locked protocol check. In every reward and omission
block, however, the neural selector produced the identical sequence `1`, no
clean action, `1`, no clean action. Rewarded action `0` was never sampled, so
reward suppression and expected omission cannot be judged.

Artifact: `research/findings/raw/vocal_action_credit_gate_v9/output_v2_seed0_cupy.json`
with its `.prov.json` provenance sidecar.

## What the run establishes

- All six conditions had `12/12` clean training actions and six contingent
  rewards.
- The rewarded expectation route grew from `0.100` to `6.747-7.043` in the
  intact and output-lesion arms. It remained `0.100` when expectation learning
  was disabled.
- Baseline expectation was zero, expectation output stayed closed during
  training, probe gates matched their lesions, training changes stayed within
  declared routes, and every probe weight remained byte-identical.
- The four-probe action sequences matched exactly across lesions. This rules
  out a lesion-induced sampling mismatch, but it does not supply the missing
  trained-action observations.

The output-intact action-`1` rows did recruit pre-outcome expectation and SNc
GABA-B/GIRK. Those rows concern an action that never received training reward,
and they contain activity in both expectation channels. They are retained as
diagnostic telemetry only. The outcome checks shown as false in the artifact
operate on an empty action-`0` subset and are not mechanism failures.

## Decision

Do not run NumPy agreement, extend the block, repeat seed `0`, reopen the center
ladder, or assign formal seeds. V9's qualified engagement result remains valid,
but its reward/omission output is untested. A successor requires a new research
gate and preregistration that first makes the learned action reappear through
the neural action policy after training. It may not force an action from the
host or tune GABA-B against this undefined artifact.

The next analysis should use the preserved training and probe telemetry to
separate loss of clean commitment from learned-policy expression, then consult
the project record and primary action-selection literature before proposing a
biological intervention.
