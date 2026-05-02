# 2026-05-02 — ROOT CAUSE: text I/O chance-level results were Hebbian decay collapse

**TL;DR:** Two months of "text I/O stuck at ~30% baseline" were caused by `cfg.enable_hebbian_learning` being LEFT AT ITS DEFAULT (True) in `text_train_embodied.py`. Every g* research runner explicitly sets it to False. Default Hebbian applies a global `hebbian_weight_decay = 1e-5` per simulation sub-step. Over 100 ep × 30 steps × ~330 sub-steps = ~990,000 sub-steps, this multiplies weights by `(1-1e-5)^990000 ≈ 5e-5` — driving every plastic weight to the `hebbian_min_weight = 0.05` floor. STDP and reward modulation cannot differentially shape weights when the global decay is dragging everything to zero.

The "32.5% / 30% baseline" on combined.json was an artifact of east-prediction bias on east-heavy eval data (revealed when balanced sampling fixed the bug today). After balanced sampling, the network's true accuracy is at chance because the language pathway never learned anything — all weights had collapsed.

## How it was found

1. Partial-T1 run gave I→W 20% / W→A 25% — chance level despite balanced predictions.
2. Full Tier 1 revert (stim=200) gave I→W 20% / W→A 24% — still chance.
3. Same training-time correct moves (~30%) but eval at chance → suspect training problem, not eval.
4. Auto-checkpoint added in 7f50cf0 captured the trained bridge.
5. Weight diagnostic on the checkpoint:

```
pathway                           n_syn     mean      std      min      max
lang_in -> cortex_N                1230    0.050    0.000    0.050    0.054
lang_in -> motor_N                  781    0.050    0.000    0.050    0.053
cortex_N -> lang_out                613    0.050    0.000    0.050    0.053
... (all 13 text pathways at uniform 0.05)

Token-targeted analysis: 0/4 tokens have target-bias > 0
Verdict: CHANCE -- weights essentially unchanged from random init
```

But weights weren't "unchanged from random init" — they had collapsed FROM their initial values (2.0-3.0) DOWN to the 0.05 floor. Including pathways that should have learned via STDP (lang_in → motor_X with PFC-bypass weight=3.0 and density=0.30).

## The mechanism

In `sim/bridge.py:4690`, every simulation sub-step applies global weight decay:

```python
cp_connections.data *= (1.0 - cfg.hebbian_weight_decay)
```

Then clips to `[hebbian_min_weight, hebbian_max_weight]` = `[0.05, 1.0]`.

With `hebbian_weight_decay = 1e-5`:

| Sub-steps | Decay factor | Initial 3.0 → |
|---|---|---|
| 50,000 | (1-1e-5)^50000 ≈ 0.61 | 1.83 |
| 100,000 | (1-1e-5)^100000 ≈ 0.37 | 1.10 → clipped to 1.0 (hebbian_max) |
| 500,000 | (1-1e-5)^500000 ≈ 0.0067 | 0.020 → clipped to 0.05 (hebbian_min) |
| 1,000,000 | (1-1e-5)^1000000 ≈ 4.5e-5 | 1.4e-4 → clipped to 0.05 |

The bridge code itself comments on this (lines 4677-4679):

```python
# Skip global weight decay during experiments: over 50K training steps,
# decay (1-1e-5)^50000 ≈ 0.61 destroys 40% of non-STDP-reinforced weights,
# collapsing network baseline excitability by post-test.
```

But the experiment-engine guard (`_experiment_running = (self.experiment_engine is not None and self.experiment_engine.is_experiment_running)`) only suppresses decay when ExperimentEngine is actively running. Text training doesn't use ExperimentEngine — it directly drives the bridge — so decay applies every sub-step.

## Why every g* runner disables Hebbian

Searched the runners directory:

```
g1_network.py:        cfg.enable_hebbian_learning = False
g1_v2_runner.py:      cfg.enable_hebbian_learning = False
g1_v3_runner.py:      cfg.enable_hebbian_learning = False
g2_runner.py:         cfg.enable_hebbian_learning = False
g5_runner.py:         cfg.enable_hebbian_learning = False
g5_v2_runner.py:      cfg.enable_hebbian_learning = False
g5_v3_runner.py:      cfg.enable_hebbian_learning = False
g6_runner.py:         cfg.enable_hebbian_learning = False
g8_runner.py:         cfg.enable_hebbian_learning = False
g9_runner.py:         cfg.enable_hebbian_learning = False
g11_bg_runner.py:     cfg.enable_hebbian_learning = False  (line 2449 + 4563)
g11_bg_replicated_runner.py: cfg.enable_hebbian_learning = False
g11_bg_trajectory_train.py:  cfg.enable_hebbian_learning = False
text_train_embodied.py: ← MISSING. Default True.
```

Every research runner workaround applied; only text training was missed. The code worked despite the decay for short test runs (smoke = 5-10 ep, weights only mildly decayed). The decay only catastrophically collapsed in 100+ ep training. So during text I/O development (where most testing was at 5-10 ep smokes), the bug was invisible.

## What the 32.5% baseline really was

The May-1 19:22 baseline at I→W 32.5% / W→A 30% was using the SAME buggy code. So weights collapsed to 0.05 floor in that run too. Why did it score above chance?

Answer: pre-balanced training distribution. Without balanced sampling (the d961940 fix at May-1 19:33, AFTER the baseline at 19:22), targets were east/west-heavy due to `|dx|>=|dy|` tie-breaking. The agent learned a bias toward "predict east often" — a strategy that scored 50% on east-heavy eval samples (8/16 east correct), ~20% on others, summing to 32.5%. Predicted distribution: N:9 E:19 S:6 W:6 — heavy east bias. With balanced sampling fixing the bug, accuracy collapsed to chance, revealing there was no real learning.

## The fix

One line in `text_train_embodied.py` (commit 144eefd):

```python
cfg.enable_hebbian_learning = False
```

Matches every other g* runner. STDP + reward modulation handle the actual learning; Hebbian was contributing only catastrophic global decay.

## Followup test

Launched 2026-05-02 02:46 (PID 39408): 100-ep at seed=42 with `stim_steps=200, reset_steps=100, enable_hebbian_learning=False`. Same config as the previous full-revert run (PID 22124) except Hebbian off. Direct A/B test isolates the Hebbian effect.

ETA ~03:40. Saves checkpoint for weight diagnostic confirmation.

Predictions:
- Weights should retain initial magnitudes (lang_in → cortex_X near 2.0, lang_in → motor_X near 3.0)
- STDP should differentiate per token: W(north_active → motor_N) > W(north_active → motor_{E,S,W})
- Eval accuracy should jump ABOVE chance, possibly significantly

If predictions confirmed: the chance-level baseline was fully explained by Hebbian decay. Text I/O can finally be developed with proper plasticity.

If accuracy still chance even with weights preserved: there are additional issues beyond Hebbian (eval methodology, training duration, regime design). Will need further investigation.

## Implications for prior research

This bug affected ALL `text_train.py` and `text_train_embodied.py` results since the text I/O system was added 2026-05-01. Any prior conclusion drawn from those runs ("regime R3 vs R6 best", "balanced sampling is the bug", etc.) needs revisiting under the corrected configuration.

Most likely affected: the entire 2026-05-01 evening's experimentation that produced the "30% baseline". The architectural conclusions (need PFC-bypass, need balanced sampling) are still likely valid — but the 30% accuracy floor was the bug, not the model.

## Repro / files

- Buggy run checkpoint: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_NoT1_seed42.simstate.h5`
- Weight diagnostic JSON: `research/findings/raw/g11_bg/text_weight_diag_R3R6_NoT1_seed42.json`
- Fix commit: 144eefd
- New test run output: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_seed42.json` (ETA ~03:40)
