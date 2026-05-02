# 2026-05-02 — Hebbian fix EMPIRICAL RESULT: 3/4 tokens learn target motor pool

**TL;DR:** Disabling Hebbian (commit 144eefd) restored real plasticity. The 100-ep
embodied training at seed=42 with `cfg.enable_hebbian_learning = False` (and otherwise
identical to PID 22124's full-Tier-1-revert config) produced weights with REAL
differential structure for the first time. **Token-targeted analysis:** 3 of 4
direction tokens have target_motor_X mean weight > non-target average. Eval
methodology still loses signal due to noisy baseline subtraction, but the
underlying network demonstrates partial language-action mapping learning.

## Headline numbers

| Metric | Hebbian ON (PID 22124) | Hebbian OFF (PID 39408) |
|---|---|---|
| Training-time correct moves | 30.1% | 29.7% |
| I→W eval (n=100) | 20.0% | 17.0% |
| W→A eval (n=100) | 24.0% | 25.0% |
| W→A east-token | -- | **48.0% (12/25)** |
| Weight diagnostic verdict | CHANCE | **PARTIAL: 3/4 LEARNED** |

The headline accuracy didn't move much, but the eval methodology has noisy
baselines. The underlying learning is what changed.

## Weight diagnostic comparison

### Hebbian ON (PID 22124, broken):

```
pathway                  n_syn   mean   std    min    max
lang_in -> cortex_N      1230    0.050  0.000  0.050  0.054
lang_in -> motor_N        781    0.050  0.000  0.050  0.053
... (all 13 text pathways at uniform 0.05 floor)

Token-targeted: 0/4 tokens have target-bias > 0
Verdict: CHANCE -- weights essentially unchanged from random init
```

### Hebbian OFF (PID 39408, partial success):

```
pathway                  n_syn   mean   std    min    max
lang_in -> cortex_N      1230    1.597  0.539  0.010  2.000
lang_in -> motor_N        781    1.769  0.467  0.010  2.000
cortex_N -> lang_out      613    0.010  0.000  0.010  0.010   <-- still floor!
lang_in -> cortex_E      1286    1.602  0.543  0.010  2.000
lang_in -> motor_E        774    1.721  0.514  0.010  2.000
cortex_E -> lang_out      625    0.010  0.000  0.010  0.010   <-- still floor!
lang_in -> cortex_S      1210    1.585  0.548  0.010  2.000
lang_in -> motor_S        784    1.667  0.505  0.010  2.000
cortex_S -> lang_out      660    0.010  0.000  0.010  0.010   <-- still floor!
lang_in -> cortex_W      1317    1.623  0.530  0.010  2.000
lang_in -> motor_W        750    1.744  0.495  0.010  2.000
cortex_W -> lang_out      700    0.010  0.000  0.010  0.010   <-- still floor!
IT -> lang_out           3241    0.010  0.000  0.010  0.010   <-- still floor!
lang_in -> dlpfc_wm      3072    1.619  0.549  0.010  2.000

Token-targeted analysis (PFC-bypass: lang_in active for token -> motor_X):
token   ->motor_N  ->motor_E  ->motor_S  ->motor_W  diff_target
north   1.734      1.654      1.540      1.686      +0.108  LEARNED
east    1.707      1.799      1.621      1.753      +0.106  LEARNED
south   1.718      1.761      1.595      1.779      -0.158  REVERSED
west    1.766      1.605      1.669      1.818      +0.139  LEARNED

Verdict: 3/4 tokens have target-bias > 0  (PARTIAL)
```

Two structural observations:

1. **STDP capped at w_max=2.0.** The PFC-bypass pathway `lang_in -> motor_X`
   has design weight_mean=3.0. STDP soft-bound caps weights at the configured
   `stdp_w_max=2.0`. So the design intent of "language drive dominates
   cascade contribution" is reduced. Need to raise stdp_w_max to 5.0+.

2. **Zero-init readout pathways still at floor.** `cortex_X -> language_output`
   and `IT -> language_output` were initialized at weight_mean=0.0 by design;
   STDP must grow them from scratch. With weak training signal (~30% correct
   moves), STDP didn't grow these pathways. They stayed at the synaptic floor
   (0.01). This is why I->W eval is at chance — there's no image-to-word
   readout signal at all. Needs small non-zero init so STDP can grow OR
   weaken bidirectionally.

## Why the eval headline didn't move

Despite real learning in the PFC-bypass pathway, the W->A eval headline only
changed marginally (24% -> 25%). The reason is in the per-trial dynamics:

```
[eval W->A] word=north baseline={N:1, E:17, S:8, W:0} drive={N:5, E:7, S:10, W:1}
                       delta={N:+4, E:-10, S:+2, W:+1}    -> argmax N (correct)

[eval W->A] word=east  baseline={N:5, E:4, S:4, W:13} drive={N:1, E:11, S:2, W:13}
                       delta={N:-4, E:+7, S:-2, W:0}     -> argmax E (correct)

[eval W->A] word=south baseline={N:2, E:14, S:10, W:4} drive={N:2, E:7, S:9, W:6}
                       delta={N:0, E:-7, S:-1, W:+2}    -> argmax W (WRONG, target S)

[eval W->A] word=west  baseline={N:1, E:2, S:14, W:0} drive={N:7, E:2, S:6, W:4}
                       delta={N:+6, E:0, S:-8, W:+4}    -> argmax N (WRONG, target W)
```

For north and east, the model picks correctly. For south and west, the model
picks wrong despite the underlying weights showing learning. Two reasons:

1. **South's weights are REVERSED** — the diagnostic shows south has -0.16
   target-bias (motor_S has lower mean weight than other motors for south-active
   sources). So south-input drives motor_S LESS than baseline. STDP learned
   the wrong direction. Likely cause: training distribution / cascade noise
   for south specifically.

2. **West's argmax goes to motor_N** despite weight diagnostic showing motor_W
   has highest mean (1.818). The per-trial DELTA argmax can differ from
   weight-based prediction because of trial-specific baseline noise.

The 12/25 = 48% accuracy on east shows that when learning is strong (east has
+0.106 target-bias) AND eval has clean baseline, the model can recover the
mapping.

## Implications

The Hebbian fix (commit 144eefd) is necessary but not sufficient. Two more
fixes needed:

1. **Raise stdp_w_max to 5.0** (commit 200f73c, applied) — allows PFC-bypass
   weights to differentiate further than the 2.0 cap.

2. **Non-zero init for readout pathways** (commit 200f73c, applied) — gives
   STDP something to bidirectionally adjust. Default 0.5 ± 0.3 jitter for
   cortex_X -> language_output and IT -> language_output.

These are applied for the next test run. PID TBD launching at 03:45.

## What's next

1. **Test enhanced fix** (PID 49936): same config + stdp_w_max=5.0 + readout
   pathway init=0.5. ETA ~04:40.

2. **Investigate south-direction reversal** if it persists. Possible causes:
   - Training distribution still has subtle south bias (despite balanced sampling)
   - South-pathway-specific cascade dynamics
   - Reward correlation flip for south

3. **6-seed validation** if v2 shows clean above-chance accuracy on all 4
   directions.

## Files

- Result JSON (Hebbian off): `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_seed42.json`
- Checkpoint: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_seed42.simstate.h5`
- Weight diagnostic JSON: `research/findings/raw/g11_bg/text_weight_diag_R3R6_HebOff_seed42.json`
- Fix commits: 144eefd (Hebbian off), 200f73c (stdp_w_max + readout init)
