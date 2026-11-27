# KP vs B (test2 prereg sections 4-5, `life_compare.py`)

Mode: registered. Seeds scored over: [42, 43, 44, 100, 101, 102]; >= rule need 5/6. UNDEFINED seeds (neither pass nor fail): {42: ['B life missing'], 43: ['B life missing'], 44: ['B life missing'], 100: ['B life missing'], 101: ['B life missing'], 102: ['B life missing']}. Arm missing or ended early (scored on completed nights; not passing where a later night is needed): none. Unpaired lives: none. Ignored lives (seed not expected): [7].

Primary: {'LG-P1': 'UNDEFINED', 'LG-P2': 'UNDEFINED'}. Guards: {'G1': 'UNDEFINED', 'G2a': 'UNDEFINED', 'G2b': 'UNDEFINED', 'G3': 'UNDEFINED', 'G4': 'UNDEFINED'}.


| seed | side | split | device | provenance | completed nights | model | days | brain_kw |
|---|---|---|---|---|---|---|---|---|
| 7 | arm | test2 | cuda | cuda_precision=cuda fp16, torch=2.7.0+cu128, torch_cuda=12.8, gpu_name=NVIDIA A10G, tf32_matmul=True, tf32_cudnn=True, code_commit=4ad30c097449a4be9fca08ee9f6faa324c7526e3, aws_ami_id=ami-012ba162b9cd2729c | 2 | rwkv7-g1l-1.5b-20261031-ctx34304 | 2 | {"keep_bank": "prior", "sleep_seed": 7} |
| 7 | b | test2 | cuda | cuda_precision=cuda fp16, torch=2.7.0+cu128, torch_cuda=12.8, gpu_name=NVIDIA A10G, tf32_matmul=True, tf32_cudnn=True, code_commit=4ad30c097449a4be9fca08ee9f6faa324c7526e3, aws_ami_id=ami-012ba162b9cd2729c | 2 | rwkv7-g1l-1.5b-20261031-ctx34304 | 2 | {"sleep_seed": 7} |

Notes s42: arm life missing (not passing)

Notes s43: arm life missing (not passing)

Notes s44: arm life missing (not passing)

Notes s100: arm life missing (not passing)

Notes s101: arm life missing (not passing)

Notes s102: arm life missing (not passing)

## LG-P1: lang_guard.kl night 15 (arm < B per seed; pooled arm <= 0.5 x B)

| seed | KP KL | B KL | pass |
|---|---|---|---|
| 42 | - | - | UNDEFINED (B life missing) |
| 43 | - | - | UNDEFINED (B life missing) |
| 44 | - | - | UNDEFINED (B life missing) |
| 100 | - | - | UNDEFINED (B life missing) |
| 101 | - | - | UNDEFINED (B life missing) |
| 102 | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; **LG-P1: UNDEFINED**

Pooled (mean over seeds) KP - vs B -, ratio -, half of B -, pooled ok None.

## LG-P1 (night 10, reported): lang_guard.kl night 10 (arm < B per seed; pooled arm <= 0.5 x B)

| seed | KP KL | B KL | pass |
|---|---|---|---|
| 42 | - | - | UNDEFINED (B life missing) |
| 43 | - | - | UNDEFINED (B life missing) |
| 44 | - | - | UNDEFINED (B life missing) |
| 100 | - | - | UNDEFINED (B life missing) |
| 101 | - | - | UNDEFINED (B life missing) |
| 102 | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; LG-P1 (night 10, reported): UNDEFINED (reported, not a bar)

Pooled (mean over seeds) KP - vs B -, ratio -, half of B -, pooled ok None.

## LG-P2: NLL_arm(n15) - NLL_B(n1) <= 0.5 x (NLL_B(n15) - NLL_B(n1))

| seed | KP NLL n15 | KP own n1 | B n1 | B n15 | KP rise | 0.5 x B rise | pass |
|---|---|---|---|---|---|---|---|
| 42 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; **LG-P2: UNDEFINED**

## LG-P2 (night 10, reported): NLL_arm(n10) - NLL_B(n1) <= 0.5 x (NLL_B(n10) - NLL_B(n1))

| seed | KP NLL n10 | KP own n1 | B n1 | B n10 | KP rise | 0.5 x B rise | pass |
|---|---|---|---|---|---|---|---|
| 42 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; LG-P2 (night 10, reported): UNDEFINED (reported, not a bar)

## LG-FIX: the arm's own n15 KL <= 2x n1 ('language-drift defect fixed')

| seed | KL n1 | KL n15 | ratio | pass |
|---|---|---|---|---|
| 42 | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); LG-FIX: UNDEFINED (reported, not a bar)

## G1: H hippocampus-off nights 5-15, mean of 6 (arm >= B - 1.0)

| seed | KP | B | diff | pass |
|---|---|---|---|---|
| 42 | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; **G1: UNDEFINED**

Pooled KP - vs B -.

## G2a: hippocampus-off day-1-2 recall, nights 1-4, rate x 6 (arm >= B - 1.0)

| seed | KP k/n | KP x6 | B k/n | B x6 | diff | pass |
|---|---|---|---|---|---|---|
| 42 | - | - | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; **G2a: UNDEFINED**

## G2b: hippocampus-off day-1-2 recall, nights 5-15, rate x 6 (arm >= B - 1.0)

| seed | KP k/n | KP x6 | B k/n | B x6 | diff | pass |
|---|---|---|---|---|---|---|
| 42 | - | - | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; **G2b: UNDEFINED**

## G2 n1-15 (reported): hippocampus-off day-1-2 recall, nights 1-15, rate x 6 (arm >= B - 1.0)

| seed | KP k/n | KP x6 | B k/n | B x6 | diff | pass |
|---|---|---|---|---|---|---|
| 42 | - | - | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; G2 n1-15 (reported): UNDEFINED (reported, not a bar)

## G3: collapse seeds-ever per probe arm (arm <= B and <= 3)

| probe arm | KP seeds-ever | B seeds-ever | KP max share per seed | B max share per seed | ok |
|---|---|---|---|---|---|

**G3: UNDEFINED**

## G4: stock leak n10+15, arm not higher than B (per seed and pooled)

### all rows

| seed | KP k/n | KP share | B k/n | B share | pass |
|---|---|---|---|---|---|
| 42 | - | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; G4 (all): UNDEFINED (clause of the bar below)

Pooled KP 0/0 = - vs B 0/0 = -; reduction -; pooled ok True.

### non_refusal rows

| seed | KP k/n | KP share | B k/n | B share | pass |
|---|---|---|---|---|---|
| 42 | - | - | - | - | UNDEFINED (B life missing) |
| 43 | - | - | - | - | UNDEFINED (B life missing) |
| 44 | - | - | - | - | UNDEFINED (B life missing) |
| 100 | - | - | - | - | UNDEFINED (B life missing) |
| 101 | - | - | - | - | UNDEFINED (B life missing) |
| 102 | - | - | - | - | UNDEFINED (B life missing) |

Count 0/0 (6 UNDEFINED seeds dropped) (need 5); sign-flip p (one-sided, arm better, reported) - over 0 seeds; G4 (non_refusal): UNDEFINED (clause of the bar below)

Pooled KP 0/0 = - vs B 0/0 = -; reduction -; pooled ok True.

**G4: UNDEFINED** (all rows AND non-refusal rows)

