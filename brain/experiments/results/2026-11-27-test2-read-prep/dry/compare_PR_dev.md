# PR vs B (test2 prereg sections 4-5, `life_compare.py`)

Mode: DEV. Seeds scored over: [7]; >= rule need 1/1. UNDEFINED seeds (neither pass nor fail): {7: ['B life ended after night 2']}. Arm missing or ended early (scored on completed nights; not passing where a later night is needed): none. Unpaired lives: none. Ignored lives (seed not expected): none.

Primary: {'LEAK': 'DEV (1 seeds) UNDEFINED'}. Guards: {'G1': 'DEV (1 seeds) UNDEFINED', 'G2a': 'DEV (1 seeds) UNDEFINED', 'G2b': 'DEV (1 seeds) UNDEFINED', 'G3': 'DEV (1 seeds) UNDEFINED', 'G4': 'DEV (1 seeds) UNDEFINED', 'G6': 'DEV (1 seeds) UNDEFINED'}.


| seed | side | split | device | provenance | completed nights | model | days | brain_kw |
|---|---|---|---|---|---|---|---|---|
| 7 | arm | test2 | cuda | cuda_precision=cuda fp16, torch=2.7.0+cu128, torch_cuda=12.8, gpu_name=NVIDIA A10G, tf32_matmul=True, tf32_cudnn=True, code_commit=1e91dce6c9e6d8d8386e7a492cd1a92055ce039b, aws_ami_id=ami-012ba162b9cd2729c | 2 | rwkv7-g1l-1.5b-20261031-ctx34304 | 2 | {"replay_priority": "loss_a", "sleep_seed": 7} |
| 7 | b | test2 | cuda | cuda_precision=cuda fp16, torch=2.7.0+cu128, torch_cuda=12.8, gpu_name=NVIDIA A10G, tf32_matmul=True, tf32_cudnn=True, code_commit=4ad30c097449a4be9fca08ee9f6faa324c7526e3, aws_ami_id=ami-012ba162b9cd2729c | 2 | rwkv7-g1l-1.5b-20261031-ctx34304 | 2 | {"sleep_seed": 7} |

Notes s7: arm life ended after night 2: scored on nights 1-2, B restricted to the same nights

## LG-P1: lang_guard.kl night 15 (arm < B per seed; pooled arm <= 0.5 x B)

| seed | PR KL | B KL | pass |
|---|---|---|---|
| 7 | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; LG-P1: DEV (1 seeds) UNDEFINED (reported, not a bar)

Pooled (mean over seeds) PR - vs B -, ratio -, half of B -, pooled ok None.

## LG-P1 (night 10, reported): lang_guard.kl night 10 (arm < B per seed; pooled arm <= 0.5 x B)

| seed | PR KL | B KL | pass |
|---|---|---|---|
| 7 | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; LG-P1 (night 10, reported): DEV (1 seeds) UNDEFINED (reported, not a bar)

Pooled (mean over seeds) PR - vs B -, ratio -, half of B -, pooled ok None.

## LG-P2: NLL_arm(n15) - NLL_B(n1) <= 0.5 x (NLL_B(n15) - NLL_B(n1))

| seed | PR NLL n15 | PR own n1 | B n1 | B n15 | PR rise | 0.5 x B rise | pass |
|---|---|---|---|---|---|---|---|
| 7 | - | - | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; LG-P2: DEV (1 seeds) UNDEFINED (reported, not a bar)

## LG-P2 (night 10, reported): NLL_arm(n10) - NLL_B(n1) <= 0.5 x (NLL_B(n10) - NLL_B(n1))

| seed | PR NLL n10 | PR own n1 | B n1 | B n10 | PR rise | 0.5 x B rise | pass |
|---|---|---|---|---|---|---|---|
| 7 | - | - | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; LG-P2 (night 10, reported): DEV (1 seeds) UNDEFINED (reported, not a bar)

## LG-FIX: the arm's own n15 KL <= 2x n1 ('language-drift defect fixed')

| seed | KL n1 | KL n15 | ratio | pass |
|---|---|---|---|---|
| 7 | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); LG-FIX: DEV (1 seeds) UNDEFINED (reported, not a bar)

## G1: H hippocampus-off nights 5-15, mean of 6 (arm >= B - 1.0)

| seed | PR | B | diff | pass |
|---|---|---|---|---|
| 7 | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; **G1: DEV (1 seeds) UNDEFINED**

Pooled PR - vs B -.

## G2a: hippocampus-off day-1-2 recall, nights 1-4, rate x 6 (arm >= B - 1.0)

| seed | PR k/n | PR x6 | B k/n | B x6 | diff | pass |
|---|---|---|---|---|---|---|
| 7 | - | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; **G2a: DEV (1 seeds) UNDEFINED**

## G2b: hippocampus-off day-1-2 recall, nights 5-15, rate x 6 (arm >= B - 1.0)

| seed | PR k/n | PR x6 | B k/n | B x6 | diff | pass |
|---|---|---|---|---|---|---|
| 7 | - | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; **G2b: DEV (1 seeds) UNDEFINED**

## G2 n1-15 (reported): hippocampus-off day-1-2 recall, nights 1-15, rate x 6 (arm >= B - 1.0)

| seed | PR k/n | PR x6 | B k/n | B x6 | diff | pass |
|---|---|---|---|---|---|---|
| 7 | - | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; G2 n1-15 (reported): DEV (1 seeds) UNDEFINED (reported, not a bar)

## G3: collapse seeds-ever per probe arm (arm <= B and <= 3)

| probe arm | PR seeds-ever | B seeds-ever | PR max share per seed | B max share per seed | ok |
|---|---|---|---|---|---|

**G3: DEV (1 seeds) UNDEFINED**

## D2-PR2 primary: stock leak n10+15 below B per seed and >= 30% lower pooled

### all rows

| seed | PR k/n | PR share | B k/n | B share | pass |
|---|---|---|---|---|---|
| 7 | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; D2-PR2 leak primary (all): DEV (1 seeds) UNDEFINED (clause of the bar below)

Pooled PR 0/0 = - vs B 0/0 = -; reduction -; pooled ok False.

### non_refusal rows

| seed | PR k/n | PR share | B k/n | B share | pass |
|---|---|---|---|---|---|
| 7 | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; D2-PR2 leak primary (non_refusal): DEV (1 seeds) UNDEFINED (clause of the bar below)

Pooled PR 0/0 = - vs B 0/0 = -; reduction -; pooled ok False.

**LEAK: DEV (1 seeds) UNDEFINED** (all rows AND non-refusal rows)

## G4: stock leak n10+15, arm not higher than B (per seed and pooled)

### all rows

| seed | PR k/n | PR share | B k/n | B share | pass |
|---|---|---|---|---|---|
| 7 | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; G4 (all): DEV (1 seeds) UNDEFINED (clause of the bar below)

Pooled PR 0/0 = - vs B 0/0 = -; reduction -; pooled ok True.

### non_refusal rows

| seed | PR k/n | PR share | B k/n | B share | pass |
|---|---|---|---|---|---|
| 7 | - | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); sign-flip p (one-sided, arm better, reported) - over 0 seeds; G4 (non_refusal): DEV (1 seeds) UNDEFINED (clause of the bar below)

Pooled PR 0/0 = - vs B 0/0 = -; reduction -; pooled ok True.

**G4: DEV (1 seeds) UNDEFINED** (all rows AND non-refusal rows)

## G6: KL n15 <= 1.25 x B

| seed | PR | B | ratio | pass |
|---|---|---|---|---|
| 7 | - | - | - | UNDEFINED (B life ended after night 2) |

Count 0/0 (1 UNDEFINED seeds dropped) (need 1); **G6: DEV (1 seeds) UNDEFINED**

Pooled ratio -.

