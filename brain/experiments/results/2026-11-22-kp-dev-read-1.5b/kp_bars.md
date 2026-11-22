# K-P DEV read (prereg 2026-11-21-sleep-language-keep-prereg.md, section 3; A1 section 5, A2 section 6; launched gate-overridden)
Seeds 319/320/321 (DEV). B = lifeS1F, K-P = lifeKP (`keep_bank='prior'`). All values read from the JSONs and bank files by `kp_bars.py` (brain.eval.life_leak, life_echo, life_table, life_bars). Bars per seed and pooled as the prereg states. Pooled = mean over the three seeds (ratio of means).

## 0. Runs and devices (precision confound)

| arm | seed | device | host | brain_kw | last trunk night |
|---|---|---|---|---|---|
| B | 319 | cpu | node-dl4g243 | {'sleep_seed': 319} | 15 |
| B | 320 | cpu | node-dl5d243 | {'sleep_seed': 320} | 15 |
| B | 321 | cpu | node-mxl1462drl | {'sleep_seed': 321} | 15 |
| K-P | 319 | cuda | ip-172-31-41-224 | {'sleep_seed': 319, 'keep_bank': 'prior'} | 15 |
| K-P | 320 | cuda | ip-172-31-91-140 | {'sleep_seed': 320, 'keep_bank': 'prior'} | 15 |
| K-P | 321 | cuda | ip-172-31-66-19 | {'sleep_seed': 321, 'keep_bank': 'prior'} | 15 |
| PR-l | 319 | cpu | node-dl4g243 | {'sleep_seed': 319, 'replay_priority': 'loss'} | 15 |
| PR-l | 320 | cpu | node-dl5d243 | {'sleep_seed': 320, 'replay_priority': 'loss'} | 15 |
| PR-l | 321 | cpu | node-mxl1462drl | {'sleep_seed': 321, 'replay_priority': 'loss'} | 15 |
| PR-x | 319 | cpu | dant123-wk | {'sleep_seed': 319, 'replay_priority': 'strength_x_loss'} | 15 |
| PR-x | 320 | cuda | ip-172-31-35-61 | {'sleep_seed': 320, 'replay_priority': 'strength_x_loss'} | 15 |
| PR-x | 321 | cuda | ip-172-31-82-210 | {'sleep_seed': 321, 'replay_priority': 'strength_x_loss'} | 15 |

**Confound.** K-P ran CUDA fp16 (AWS A10G); B ran CPU fp32 (pool). Every K-P vs B difference below is method + device/precision together. Section 5 gives the available evidence on how much fp16 alone moves lang_guard KL.

## 1. Engagement (must hold, or the arm is UNDEFINED)

### 1a. keep_bank record on every trunk night (from the life JSON)

Bar: a `keep_bank` record on every trunk night 1-15, `bank_size` 512, `subset_n` set, a different `subset_hash` each night, `mean_keep_kl` > 0 from night 2.

| seed | nights with record | bank_size all 512 | subset_n values | distinct hashes / nights | min mean_keep_kl n2-15 (night) | mean_keep_kl n1 | B has keep_bank record | 1a |
|---|---|---|---|---|---|---|---|---|
| 319 | 15/15 | True | [64] | 15/15 | 0.0711 (n2) | 0.1033 | 0/15 nights | PASS |
| 320 | 15/15 | True | [64] | 15/15 | 0.0878 (n2) | 0.1069 | 0/15 nights | PASS |
| 321 | 15/15 | True | [64] | 15/15 | 0.0882 (n5) | 0.0949 | 0/15 nights | PASS |

mean_keep_kl by night (K-P):

| seed | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | 0.103 | 0.071 | 0.077 | 0.090 | 0.081 | 0.089 | 0.099 | 0.103 | 0.092 | 0.099 | 0.088 | 0.110 | 0.112 | 0.112 | 0.092 |
| 320 | 0.107 | 0.088 | 0.093 | 0.093 | 0.102 | 0.092 | 0.099 | 0.100 | 0.101 | 0.103 | 0.094 | 0.096 | 0.098 | 0.110 | 0.107 |
| 321 | 0.095 | 0.093 | 0.100 | 0.098 | 0.088 | 0.115 | 0.098 | 0.109 | 0.104 | 0.098 | 0.107 | 0.106 | 0.106 | 0.111 | 0.107 |

**1a: PASS** per seed {319: True, 320: True, 321: True}.

### 1b. Bank quality (from the bank files)

Bar: >=90% of the 512 texts distinct; mean sampled length >= 40 tokens (the bank's own `mean_tokens` field); <10% of texts contain a told key of that seed's ledger as a whole word, case-insensitive. Told keys = keys of ledger items with a telling (telling is not None: the 54 told facts; the two brain-own items dan/nova have no telling and are not 'told'); the sensitivity column also counts dan/nova keys. Whole word = regex `\bkey\b`.

| seed | bank | n texts | distinct | % distinct | mean_tokens | texts with a told key | % | (all keys incl dan/nova) | 1b |
|---|---|---|---|---|---|---|---|---|---|
| 319 | PROXY: CPU one-night-gate bank (different device, not the life's bank) | 512 (n field 512) | 511 | 99.8% | 63.38 | 2 | 0.4% | 2 | PASS (proxy) |
| 320 | not pulled, no proxy | UNDEFINED | | | | | | | UNDEFINED |
| 321 | AWS bank (the life's own) | 512 (n field 512) | 512 | 100.0% | 63.49 | 6 | 1.2% | 6 | PASS |

- s319: bank seed field 105048, sampler `full softmax, float64, np.random.choice (not rwkv top_p)`, T 1.0, top_p 1.0, pen 0.0/0.0, 57 told keys checked: ['14', 'admin', 'administrator', 'anselm', 'baker', 'barnaby', 'bramble', 'cacti', 'cactus', 'calvino', 'canal', 'cashew', 'cashews', 'cornelius', 'cress', 'crossword', 'fabrizio', "fabrizio's", 'fitzgerald', 'flowerpot', 'fourteen', 'gaggia', 'geology', 'gertrude', 'hampstead', 'lapsang', 'leeds', 'linden', 'ljubljana', 'manchego', 'marguerite', 'meridian', 'okonkwo', 'orange', 'oswin', 'ottoline', 'patagonia', 'pell', 'petal', 'philippa', 'pixel', 'porridge', 'portuguese', 'rashid', 'slowdive', 'sprocket', 'stalker', 'sysadmin', 'teal', 'thursday', 'tomato', 'tomatoes', 'tuesday', 'venkataraman', 'vespa', 'volvo', 'walrus'].
- s321: bank seed field 105050, sampler `full softmax, float64, np.random.choice (not rwkv top_p)`, T 1.0, top_p 1.0, pen 0.0/0.0, 56 told keys checked: ['admin', 'administrator', 'baker', 'bramble', 'cacti', 'cactus', 'calvino', 'canal', 'cashew', 'cashews', 'cornelius', 'cress', 'crossword', 'fabrizio', "fabrizio's", 'fitzgerald', 'flowerpot', 'gaggia', 'geology', 'gertrude', 'hampstead', 'ignatz', 'lapsang', 'leeds', 'linden', 'ljubljana', 'manchego', 'marguerite', 'meridian', 'okonkwo', 'orange', 'oswin', 'ottoline', 'patagonia', 'pell', 'petal', 'philippa', 'pixel', 'porridge', 'portuguese', 'rashid', 'slowdive', 'sprocket', 'stalker', 'sysadmin', 'tavistock', 'teal', 'thursday', 'tomato', 'tomatoes', 'tuesday', 'venkataraman', 'vespa', 'volvo', 'walrus', 'zbigniew'].

s320 bank quality is UNDEFINED (AWS bank not pulled, instance terminated, no proxy). s319 is a CPU one-night-gate bank (proxy only, labelled; same sampler code but a different device and run). Only s321 is the life's own bank.

**1b: s321 (life's own bank) PASS; s319 PASS as PROXY only; s320 UNDEFINED.** Bar 1 as a whole therefore cannot be asserted for 2 of 3 seeds: 1a holds from the logs, 1b is UNDEFINED for s320 and proxy-only for s319.

## 2. Primary: held-out language guard (nights[].lang_guard: kl, nll_on; nll_off is the pretrained cortex, constant)

| seed | night | B KL | K-P KL | K-P/B | B NLL_on | K-P NLL_on | nll_off (pretrained) |
|---|---|---|---|---|---|---|---|
| 319 | 1 | 0.1292 | 0.0667 | 0.516 | 1.991 | 1.971 | 1.931 / 1.931 |
| 319 | 5 | 0.9057 | 0.1606 | 0.177 | 2.751 | 2.020 | 1.931 / 1.931 |
| 319 | 10 | 1.5831 | 0.2622 | 0.166 | 3.359 | 2.139 | 1.931 / 1.931 |
| 319 | 15 | 2.9333 | 0.3510 | 0.120 | 4.747 | 2.258 | 1.931 / 1.931 |
| 320 | 1 | 0.1751 | 0.0704 | 0.402 | 2.011 | 1.988 | 1.931 / 1.931 |
| 320 | 5 | 1.0055 | 0.2369 | 0.236 | 2.822 | 2.075 | 1.931 / 1.931 |
| 320 | 10 | 1.8112 | 0.2360 | 0.130 | 3.580 | 2.121 | 1.931 / 1.931 |
| 320 | 15 | 2.3791 | 0.2936 | 0.123 | 4.107 | 2.179 | 1.931 / 1.931 |
| 321 | 1 | 0.1618 | 0.0914 | 0.565 | 2.014 | 1.998 | 1.931 / 1.931 |
| 321 | 5 | 0.9351 | 0.1767 | 0.189 | 2.675 | 2.031 | 1.931 / 1.931 |
| 321 | 10 | 1.9825 | 0.2578 | 0.130 | 3.845 | 2.157 | 1.931 / 1.931 |
| 321 | 15 | 2.8580 | 0.3234 | 0.113 | 4.674 | 2.200 | 1.931 / 1.931 |
| pooled mean | 1 | 0.1554 | 0.0762 | 0.490 | 2.005 | 1.986 | |
| pooled mean | 5 | 0.9488 | 0.1914 | 0.202 | 2.749 | 2.042 | |
| pooled mean | 10 | 1.7923 | 0.2520 | 0.141 | 3.595 | 2.139 | |
| pooled mean | 15 | 2.7235 | 0.3227 | 0.118 | 4.509 | 2.212 | |

KL by night, every trunk night (B / K-P):

| seed | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | 0.13/0.07 | 0.39/0.10 | 0.53/0.23 | 0.61/0.14 | 0.91/0.16 | 0.86/0.18 | 1.07/0.21 | 1.35/0.22 | 1.50/0.24 | 1.58/0.26 | 1.89/0.27 | 1.78/0.27 | 1.96/0.28 | 2.28/0.29 | 2.93/0.35 |
| 320 | 0.18/0.07 | 0.37/0.16 | 0.54/0.13 | 0.56/0.15 | 1.01/0.24 | 0.84/0.18 | 1.34/0.26 | 1.32/0.22 | 1.36/0.27 | 1.81/0.24 | 1.78/0.27 | 2.18/0.25 | 2.15/0.27 | 2.09/0.30 | 2.38/0.29 |
| 321 | 0.16/0.09 | 0.59/0.11 | 0.69/0.13 | 0.88/0.19 | 0.94/0.18 | 1.33/0.19 | 1.41/0.21 | 1.56/0.26 | 1.79/0.26 | 1.98/0.26 | 2.29/0.28 | 2.32/0.31 | 2.35/0.36 | 2.55/0.35 | 2.86/0.32 |

### 2a. KL bar (night 15: K-P pooled <= 0.5 x B pooled AND K-P < B on 3/3 seeds)

Night 15: pooled K-P 0.3227 vs B 2.7235; ratio 0.118; half-of-B bar 1.3617; K-P below B per seed {319: True, 320: True, 321: True}.
**KL n15 bar: PASS**

Night 10 (reported, same direction 3/3): pooled K-P 0.2520 vs B 1.7923; ratio 0.141; <= half: True; below B per seed {319: True, 320: True, 321: True}.

### 2b. NLL bar (night 15: K-P nll_on <= midpoint(night 1, B night 15), per seed, 3/3)

The prereg says "the midpoint between night 1 and B's night 15"; night 1 is ambiguous, so both readings are shown. (i) K-P's own night 1; (ii) B's night 1.

| seed | reading | night-1 ref | B n15 | midpoint | K-P n15 | <= midpoint |
|---|---|---|---|---|---|---|
| 319 | K-P own n1 | 1.971 | 4.747 | 3.359 | 2.258 | True |
| 320 | K-P own n1 | 1.988 | 4.107 | 3.048 | 2.179 | True |
| 321 | K-P own n1 | 1.998 | 4.674 | 3.336 | 2.200 | True |
| 319 | B n1 | 1.991 | 4.747 | 3.369 | 2.258 | True |
| 320 | B n1 | 2.011 | 4.107 | 3.059 | 2.179 | True |
| 321 | B n1 | 2.014 | 4.674 | 3.344 | 2.200 | True |

**NLL n15 bar: own-n1 reading PASS; B-n1 reading PASS.**

Night 10, same two readings (reported; the midpoint uses B night 10):

| seed | reading | night-1 ref | B n10 | midpoint | K-P n10 | <= midpoint |
|---|---|---|---|---|---|---|
| 319 | K-P own n1 | 1.971 | 3.359 | 2.665 | 2.139 | True |
| 320 | K-P own n1 | 1.988 | 3.580 | 2.784 | 2.121 | True |
| 321 | K-P own n1 | 1.998 | 3.845 | 2.922 | 2.157 | True |
| 319 | B n1 | 1.991 | 3.359 | 2.675 | 2.139 | True |
| 320 | B n1 | 2.011 | 3.580 | 2.796 | 2.121 | True |
| 321 | B n1 | 2.014 | 3.845 | 2.929 | 2.157 | True |

Night 10 direction (reported): KL below B {319: True, 320: True, 321: True}; NLL below B {319: True, 320: True, 321: True}; NLL <= midpoint 3/3: own-n1 PASS, B-n1 PASS.

**Bar 2 (primary): own-n1 reading PASS; B-n1 reading PASS.** Reference (Stage 1 TEST B): NLL 1.93 -> 3.8-4.25, KL 13-17x night 1.

- B KL n15 / n1 per seed: s319 22.70x, s320 13.58x, s321 17.66x (life_bars LG-KL bar is n15 <= 2x n1).
- K-P KL n15 / n1 per seed: s319 5.26x, s320 4.17x, s321 3.54x (life_bars LG-KL bar is n15 <= 2x n1).

## 3. Guards (any fail = KILL)

### 3.1 H with the hippocampus off, nights 5-15 (bar: K-P mean >= B mean - 0.5 of 6, per seed)

| seed | B mean /6 | K-P mean /6 | diff | ok |
|---|---|---|---|---|
| 319 | 2.545 | 4.000 | +1.455 | True |
| 320 | 1.818 | 3.273 | +1.455 | True |
| 321 | 1.818 | 3.818 | +2.000 | True |
| pooled | 2.061 | 3.697 | | |

Per night (B / K-P, of 6):

| seed | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | 5/5 | 6/5 | 2/5 | 2/5 | 2/1 | 2/4 | 2/4 | 2/4 | 2/5 | 3/4 | 0/2 |
| 320 | 2/5 | 2/1 | 1/5 | 2/5 | 1/5 | 2/5 | 2/0 | 2/1 | 2/5 | 2/1 | 2/3 |
| 321 | 3/5 | 2/5 | 2/6 | 2/5 | 1/1 | 1/4 | 2/4 | 1/4 | 2/3 | 2/1 | 2/4 |

**3.1 H-off: PASS** {319: True, 320: True, 321: True}

### 3.2 Direct hippocampus-off recall of day-1-2 facts (bar: K-P >= B - 1 of 6, per seed)

Definition as the D2-PR read: hippocampus-off trunk battery rows of kind H or rotating whose ledger item has min(told_days) <= 2, scored `ok`; rate x 6. The prereg names no window: the nights 5-15 window (as D2-PR) is the bar row; nights 1-4 and 1-15 shown beside.

| seed | window | B k/n (rate x6) | K-P k/n (rate x6) | diff /6 | ok |
|---|---|---|---|---|---|
| 319 | n5-15 | 30/74 (2.43) | 49/74 (3.97) | +1.54 | True |
| 320 | n5-15 | 20/73 (1.64) | 40/73 (3.29) | +1.64 | True |
| 321 | n5-15 | 21/76 (1.66) | 46/76 (3.63) | +1.97 | True |
| 319 | n1-4 | 16/29 (3.31) | 24/29 (4.97) | +1.66 | True |
| 320 | n1-4 | 27/29 (5.59) | 21/29 (4.34) | -1.24 | False |
| 321 | n1-4 | 15/29 (3.10) | 21/29 (4.34) | +1.24 | True |
| 319 | n1-15 | 46/103 (2.68) | 73/103 (4.25) | +1.57 | True |
| 320 | n1-15 | 47/102 (2.76) | 61/102 (3.59) | +0.82 | True |
| 321 | n1-15 | 36/105 (2.06) | 67/105 (3.83) | +1.77 | True |

**3.2 direct d1-2 (nights 5-15): PASS** {319: True, 320: True, 321: True}; nights 1-4 would be FAIL {319: True, 320: False, 321: True}; nights 1-15 PASS.

### 3.3 Collapse (life_bars.language: seed collapses in an arm if most-repeated reply share >= 0.4 on any trunk probe night of that arm; bar: seeds-ever <= B per probe arm)

| probe arm | B seeds-ever (max share per seed 319, 320, 321) | K-P seeds-ever (max share) | K-P <= B |
|---|---|---|---|
| both-off | 0 (0.06, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) | True |
| consolidated-off | 1 (0.50, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) | True |
| hippocampus-off | 2 (0.45, 0.55, 0.30) | 2 (0.35, 0.55, 0.55) | True |
| intact | 2 (0.40, 0.40, 0.30) | 1 (0.15, 0.10, 0.55) | True |
| intact-frozen | 2 (0.40, 0.55, 0.30) | 1 (0.10, 0.06, 0.55) | True |

life_bars bars: B: LG-KL DEV (3 seeds), LG-COL DEV (3 seeds), KILL-10 DEV (3 seeds); K-P: LG-KL DEV (3 seeds), LG-COL DEV (3 seeds), KILL-10 DEV (3 seeds)

**3.3 collapse: PASS**

### 3.4 Stock self-reply leak (life_leak, nights 10 and 15, intact + hippocampus-off pooled per seed; bar: K-P share <= B share per seed)

| arm | seed | leaks / n (n10+n15) | share | night 10 | night 15 |
|---|---|---|---|---|---|
| B | 319 | 39/156 | 0.250 | 27/82 | 12/74 |
| B | 320 | 54/152 | 0.355 | 28/82 | 26/70 |
| B | 321 | 42/152 | 0.276 | 24/82 | 18/70 |
| K-P | 319 | 0/156 | 0.000 | 0/82 | 0/74 |
| K-P | 320 | 3/152 | 0.020 | 1/82 | 2/70 |
| K-P | 321 | 14/152 | 0.092 | 14/82 | 0/70 |

Pooled over seeds: K-P 17/460 = 0.0370 vs B 135/460 = 0.2935. **3.4 leak: PASS** (K-P not higher than B per seed: {319: True, 320: True, 321: True}).

### 3.5 Transcript guard (whole-transcript read, new defect class, sampled-bank phrases in replies)

Computed elsewhere: **see blind read** (blind_KP). UNDEFINED in this script.

## 4. Secondary (report only)

### 4.1 Adapter norm per night (B / K-P)

| seed | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | 55/80 | 124/102 | 173/126 | 231/151 | 263/168 | 311/190 | 372/210 | 414/234 | 454/261 | 488/277 | 515/295 | 561/311 | 602/329 | 626/341 | 660/358 |
| 320 | 51/88 | 93/127 | 159/143 | 195/162 | 287/189 | 318/199 | 374/229 | 404/246 | 457/270 | 495/281 | 523/288 | 561/299 | 596/311 | 642/344 | 675/361 |
| 321 | 52/77 | 140/119 | 210/140 | 263/175 | 302/189 | 361/214 | 405/230 | 454/251 | 500/271 | 530/283 | 569/304 | 608/328 | 631/352 | 661/364 | 693/370 |

n15 / n1 ratio: s319 B 11.94x, K-P 4.49x; s320 B 13.20x, K-P 4.09x; s321 B 13.30x, K-P 4.81x

### 4.2 Recall of once-told E and F facts (battery rows kind H/rotating/neardup, `ok`), per arm, nights 1-15 pooled per seed and per night pooled over seeds

| seed | cls | arm | B k/n | K-P k/n |
|---|---|---|---|---|
| 319 | E | intact | 1/5 | 4/5 |
| 319 | E | hippocampus-off | 1/5 | 4/5 |
| 319 | F | intact | 2/12 | 10/12 |
| 319 | F | hippocampus-off | 0/12 | 6/12 |
| 320 | E | intact | 1/7 | 2/7 |
| 320 | E | hippocampus-off | 1/7 | 3/7 |
| 320 | F | intact | 4/11 | 7/11 |
| 320 | F | hippocampus-off | 4/11 | 7/11 |
| 321 | E | intact | 1/4 | 3/4 |
| 321 | E | hippocampus-off | 2/4 | 2/4 |
| 321 | F | intact | 1/11 | 2/11 |
| 321 | F | hippocampus-off | 1/11 | 5/11 |
| pooled | E | intact | 3/16 (0.188) | 9/16 (0.562) |
| pooled | E | hippocampus-off | 4/16 (0.250) | 9/16 (0.562) |
| pooled | F | intact | 7/34 (0.206) | 19/34 (0.559) |
| pooled | F | hippocampus-off | 5/34 (0.147) | 18/34 (0.529) |

Per night, pooled over seeds, E+F rows (B k/n -> K-P k/n), intact then hippocampus-off:

| arm | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intact | 0/0->0/0 | 0/1->1/1 | 1/3->3/3 | 1/3->1/3 | 0/3->1/3 | 1/4->2/4 | 0/3->2/3 | 0/3->1/3 | 1/6->2/6 | 2/6->5/6 | 0/2->0/2 | 2/6->5/6 | 0/1->1/1 | 1/5->4/5 | 1/4->0/4 |
| hippocampus-off | 0/0->0/0 | 0/1->1/1 | 1/3->3/3 | 0/3->1/3 | 0/3->1/3 | 1/4->2/4 | 0/3->2/3 | 1/3->2/3 | 0/6->2/6 | 2/6->3/6 | 1/2->0/2 | 1/6->4/6 | 0/1->1/1 | 1/5->4/5 | 1/4->1/4 |

Cue-ladder direct cue C0 rows (hippocampus-off, tracks E and F, all nights), `ok`:

| seed | track | B | K-P |
|---|---|---|---|
| 319 | E | 0/6 | 2/6 |
| 319 | F | 3/18 | 6/18 |
| 320 | E | 2/6 | 2/6 |
| 320 | F | 5/18 | 9/18 |
| 321 | E | 3/6 | 2/6 |
| 321 | F | 4/18 | 8/18 |
| pooled | E | 5/18 | 6/18 |
| pooled | F | 12/54 | 23/54 |

### 4.3 Items told in days 11-15 (week 3): intact and hippocampus-off battery H/rotating rows, nights 11-15

Concern: the blind emitter could not plant wrong-episode/perspective canaries for K-P, which would mean no intact week-3 battery row answered a week-3-told item correctly. Week-3-told = ledger items with any told day >= 11 (and told on or before the probe night). Split: `new` = first told on day >= 11; `re-told` = first told earlier, told again on day 11 (the AB items, told_days (2, 11)). Rows are battery rows of kind H or rotating (and neardup shown separately), scored `ok`.

| seed | arm | kinds | window | B: new ok/n | B: re-told ok/n | K-P: new ok/n | K-P: re-told ok/n |
|---|---|---|---|---|---|---|---|
| 319 | intact | H+rot | n15-15 | 0/0 | 0/1 | 0/0 | 0/1 |
| 319 | intact | H+rot | n11-15 | 0/2 | 0/3 | 2/2 | 1/3 |
| 319 | intact | neardup | n15-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| 319 | intact | neardup | n11-15 | 1/3 | 0/0 | 1/3 | 0/0 |
| 319 | hippocampus-off | H+rot | n15-15 | 0/0 | 0/1 | 0/0 | 0/1 |
| 319 | hippocampus-off | H+rot | n11-15 | 0/2 | 0/3 | 2/2 | 2/3 |
| 319 | hippocampus-off | neardup | n15-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| 319 | hippocampus-off | neardup | n11-15 | 0/3 | 0/0 | 1/3 | 0/0 |
| 320 | intact | H+rot | n15-15 | 0/0 | 0/0 | 0/0 | 0/0 |
| 320 | intact | H+rot | n11-15 | 1/1 | 0/1 | 0/1 | 0/1 |
| 320 | intact | neardup | n15-15 | 0/2 | 0/0 | 2/2 | 0/0 |
| 320 | intact | neardup | n11-15 | 0/4 | 0/0 | 2/4 | 0/0 |
| 320 | hippocampus-off | H+rot | n15-15 | 0/0 | 0/0 | 0/0 | 0/0 |
| 320 | hippocampus-off | H+rot | n11-15 | 1/1 | 0/1 | 0/1 | 0/1 |
| 320 | hippocampus-off | neardup | n15-15 | 0/2 | 0/0 | 2/2 | 0/0 |
| 320 | hippocampus-off | neardup | n11-15 | 0/4 | 0/0 | 2/4 | 0/0 |
| 321 | intact | H+rot | n15-15 | 0/0 | 1/1 | 0/0 | 0/1 |
| 321 | intact | H+rot | n11-15 | 0/1 | 1/4 | 1/1 | 1/4 |
| 321 | intact | neardup | n15-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| 321 | intact | neardup | n11-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| 321 | hippocampus-off | H+rot | n15-15 | 0/0 | 1/1 | 0/0 | 1/1 |
| 321 | hippocampus-off | H+rot | n11-15 | 0/1 | 1/4 | 1/1 | 1/4 |
| 321 | hippocampus-off | neardup | n15-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| 321 | hippocampus-off | neardup | n11-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| pooled | intact | H+rot | n15-15 | 0/0 | 1/2 | 0/0 | 0/2 |
| pooled | intact | H+rot | n11-15 | 1/4 | 1/8 | 3/4 | 2/8 |
| pooled | intact | neardup | n15-15 | 0/4 | 0/0 | 2/4 | 0/0 |
| pooled | intact | neardup | n11-15 | 1/8 | 0/0 | 3/8 | 0/0 |
| pooled | hippocampus-off | H+rot | n15-15 | 0/0 | 1/2 | 0/0 | 1/2 |
| pooled | hippocampus-off | H+rot | n11-15 | 1/4 | 1/8 | 3/4 | 3/8 |
| pooled | hippocampus-off | neardup | n15-15 | 0/4 | 0/0 | 2/4 | 0/0 |
| pooled | hippocampus-off | neardup | n11-15 | 0/8 | 0/0 | 3/8 | 0/0 |

Row-level detail, intact arm, night 15, kind H/rotating (item, first-told>=11, ok):

- B s319: ab15(re-told)=False
- B s320: no week-3 H/rotating rows at night 15
- B s321: ab09(re-told)=True
- K-P s319: ab15(re-told)=False
- K-P s320: no week-3 H/rotating rows at night 15
- K-P s321: ab09(re-told)=False

Read: at night 15 there is no intact week-3 NEW-item H/rotating row in either arm (0 rows), and the only night-15 week-3 rows are re-told AB items (B 1/2 correct, K-P 0/2 correct, pooled intact). Over nights 11-15 intact K-P DID answer week-3 items correctly (new 3/4, re-told 2/8 pooled) against B (new 1/4, re-told 1/8). So the premise 'no intact week-3 battery row answered a week-3-told item correctly' does NOT hold for nights 11-15 and is vacuous at night 15 (no new-item rows, 0/2 re-told right for K-P).

Same-day recall of week-3 items is a lived-turn (`sameday`) read, not part of the battery; not scored here.

### 4.4 Interview AUROC and invented-memory rate

UNDEFINED from the results JSON: `life_bars` does not report them. The interview AUROC and invented-memory rate come from `life_interview_read` over blind-reader claims (blind_KP / blind_S1F directories); those claims are not in the life JSONs and are not computed here. The battery's own never-item counters (never_false = a never-told item reported as known; never_invented) are shown instead, hippocampus-off and intact, summed over trunk nights 1-15:

| seed | arm | probe arm | never rows | never_false | never_invented |
|---|---|---|---|---|---|
| 319 | B | intact | 45 | 36 | 45 |
| 319 | B | hippocampus-off | 45 | 30 | 45 |
| 319 | K-P | intact | 45 | 33 | 44 |
| 319 | K-P | hippocampus-off | 45 | 29 | 38 |
| 320 | B | intact | 45 | 32 | 45 |
| 320 | B | hippocampus-off | 45 | 30 | 45 |
| 320 | K-P | intact | 45 | 22 | 45 |
| 320 | K-P | hippocampus-off | 45 | 18 | 39 |
| 321 | B | intact | 45 | 36 | 45 |
| 321 | B | hippocampus-off | 45 | 36 | 45 |
| 321 | K-P | intact | 45 | 27 | 41 |
| 321 | K-P | hippocampus-off | 45 | 29 | 45 |

### 4.5 First-person echo (life_echo, nights 10+15, echo counted on told-item battery rows not scored right)

| seed | arm | probe arm | n | ok | echo | echo with key | echo share |
|---|---|---|---|---|---|---|---|
| 319 | B | intact | 16 | 2 | 5 | 4 | 0.312 |
| 319 | B | hippocampus-off | 16 | 1 | 5 | 4 | 0.312 |
| 319 | K-P | intact | 16 | 6 | 7 | 7 | 0.438 |
| 319 | K-P | hippocampus-off | 16 | 7 | 0 | 0 | 0.000 |
| 320 | B | intact | 16 | 1 | 5 | 4 | 0.312 |
| 320 | B | hippocampus-off | 16 | 1 | 5 | 4 | 0.312 |
| 320 | K-P | intact | 16 | 9 | 1 | 1 | 0.062 |
| 320 | K-P | hippocampus-off | 16 | 10 | 0 | 0 | 0.000 |
| 321 | B | intact | 16 | 2 | 6 | 6 | 0.375 |
| 321 | B | hippocampus-off | 16 | 3 | 7 | 6 | 0.438 |
| 321 | K-P | intact | 16 | 1 | 4 | 3 | 0.250 |
| 321 | K-P | hippocampus-off | 16 | 10 | 0 | 0 | 0.000 |
| pooled | B | intact | 48 | 5 | 16 | 14 | 0.333 |
| pooled | B | hippocampus-off | 48 | 5 | 17 | 14 | 0.354 |
| pooled | K-P | intact | 48 | 16 | 12 | 11 | 0.250 |
| pooled | K-P | hippocampus-off | 48 | 27 | 0 | 0 | 0.000 |

### 4.6 Night seconds (K-P CUDA fp16 vs B CPU fp32; the devices differ, so this is NOT a like-for-like cost)

Night record `seconds` (sleep wall time as logged; K-P night 1 includes the bank sampling).

| seed | arm | device | night 1 | mean n2-15 | min n2-15 | max n2-15 |
|---|---|---|---|---|---|---|
| 319 | B | cpu | 1643 | 1565 | 1516 | 1597 |
| 319 | K-P | cuda | 1802 | 1141 | 1112 | 1168 |
| 320 | B | cpu | 1610 | 1537 | 1489 | 1587 |
| 320 | K-P | cuda | 1877 | 1175 | 1153 | 1193 |
| 321 | B | cpu | 1758 | 1678 | 1639 | 1723 |
| 321 | K-P | cuda | 1846 | 1146 | 1130 | 1168 |

Sleep-log lines of the K-P night (one example, s321 night 2): sleep step 0: replay 3.209 keep 0.0654; sleep step 460: replay 0.517 keep 0.0483

## 5. Precision confound: how much does fp16 alone move lang_guard KL?

No run isolates fp16 on the same method and seed. The available evidence is indirect; all of it is labelled.

### 5.1 PR-x reference (a different method, B-like replay with a priority weight; s319 CPU fp32 on W, s320/321 CUDA fp16 on AWS) against B (CPU fp32)

PR-l (all CPU fp32, pool) shown as the same-device control for the replay-priority change itself.

| seed | arm | device | KL n1 | ratio to B n1 | KL n5 | ratio | KL n10 | ratio | KL n15 | ratio to B n15 | NLL_on n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | B | cpu | 0.1292 | 1.000 | 0.9057 | 1.000 | 1.5831 | 1.000 | 2.9333 | 1.000 | 4.747 |
| 319 | PR-l | cpu | 0.1627 | 1.259 | 0.8833 | 0.975 | 1.5343 | 0.969 | 1.9118 | 0.652 | 3.731 |
| 319 | PR-x | cpu | 0.1309 | 1.013 | 0.8278 | 0.914 | 1.5114 | 0.955 | 2.2596 | 0.770 | 3.968 |
| 319 | K-P | cuda | 0.0667 | 0.516 | 0.1606 | 0.177 | 0.2622 | 0.166 | 0.3510 | 0.120 | 2.258 |
| 320 | B | cpu | 0.1751 | 1.000 | 1.0055 | 1.000 | 1.8112 | 1.000 | 2.3791 | 1.000 | 4.107 |
| 320 | PR-l | cpu | 0.1742 | 0.994 | 0.8219 | 0.817 | 1.2621 | 0.697 | 2.3616 | 0.993 | 4.186 |
| 320 | PR-x | cuda | 0.1522 | 0.869 | 0.7632 | 0.759 | 1.8131 | 1.001 | 2.0892 | 0.878 | 3.841 |
| 320 | K-P | cuda | 0.0704 | 0.402 | 0.2369 | 0.236 | 0.2360 | 0.130 | 0.2936 | 0.123 | 2.179 |
| 321 | B | cpu | 0.1618 | 1.000 | 0.9351 | 1.000 | 1.9825 | 1.000 | 2.8580 | 1.000 | 4.674 |
| 321 | PR-l | cpu | 0.1871 | 1.156 | 0.5430 | 0.581 | 1.1733 | 0.592 | 1.8590 | 0.650 | 3.633 |
| 321 | PR-x | cuda | 0.1616 | 0.999 | 0.5839 | 0.624 | 1.3207 | 0.666 | 1.8687 | 0.654 | 3.561 |
| 321 | K-P | cuda | 0.0914 | 0.565 | 0.1767 | 0.189 | 0.2578 | 0.130 | 0.3234 | 0.113 | 2.200 |

PR-x KL n15 / B: s319 (CPU) 0.770, s320 (fp16) 0.878, s321 (fp16) 0.654. PR-l KL n15 / B (CPU, same device): s319 0.652, s320 0.993, s321 0.650.
Mean of the two PR-x fp16 seeds 0.766 against the CPU seed 0.770. Night-1 KL ratio to B for PR-x (n1 weights are near strength-only) is the cleanest early read of a device effect: s319 1.013, s320 0.869, s321 0.999. These ratios mix a method change (loss-weighted replay) and the device/precision change, with 1 life per cell; they bound but do not isolate the precision effect. K-P/B at n15 is 0.118 pooled, against PR-x fp16/B of 0.766; so the K-P effect is far larger than the PR-x fp16 movement (by this crude comparison).

### 5.2 CPU fp32 one-night gate replications (brain/experiments/results/2026-11-22-kp-gate-replications-1.5b.md)

One extra night (night 16) from each seed's B night-15 home, B vs K-P, both CPU fp32 (s319 on W, s321 on pool40), `drift_read.py` held-out text. Values copied from that file:

| seed | arm | held-out KL | held-out NLL | replayed-episode NLL |
|---|---|---|---|---|
| 319 | before (n15) | 2.933 | 4.747 | 1.4 |
| 319 | B n16 | 2.792 | 4.625 | 0.977 |
| 319 | K-P n16 | 0.762 | 2.707 | 0.444 |
| 321 | before (n15) | 2.858 | 4.674 | 0.609 |
| 321 | B n16 | 2.76 | 4.507 | 0.546 |
| 321 | K-P n16 | 0.813 | 2.737 | 0.5 |
| 316 (gate) | B (one night) | 2.8 | 4.59 | 0.77 |
| 316 (gate) | K-P (one night) | 0.86 | 2.81 | 0.451 |

What this shows: on CPU fp32 (no precision difference), one K-P night takes held-out KL from about 2.9 to about 0.8 against B's 2.8, on 3/3 seeds. So the K-P effect on KL is reproduced without a device change. What it does NOT show: whether the 15-night KL in the AWS fp16 K-P lives matches what a CPU fp32 K-P life would give. These are one-night repairs from a drifted home, not a 15-night accumulation. The fp16-alone effect is not measured; the only same-method cross-device pair is PR-x (5.1).

## 6. Bar summary and prereg section 4 decision

| bar | verdict | per-seed / values |
|---|---|---|
| 1a engagement (logs) | **PASS** | {319: True, 320: True, 321: True} |
| 1b bank quality | **s321 PASS; s319 PASS (PROXY); s320 UNDEFINED** | s321 own bank; s319 CPU proxy; s320 not pulled |
| 1 engagement overall | **UNDEFINED (1a PASS 3/3; 1b s321 PASS, s319 proxy PASS, s320 undefined)** | |
| 2 KL n15 (<= half of B pooled, below B 3/3) | **PASS** | pooled 0.3227 vs B 2.7235 (ratio 0.118); below B {319: True, 320: True, 321: True} |
| 2 NLL n15 <= midpoint, K-P own n1 | **PASS** | s319 2.258 vs 3.359; s320 2.179 vs 3.048; s321 2.200 vs 3.336 |
| 2 NLL n15 <= midpoint, B n1 | **PASS** | s319 2.258 vs 3.369; s320 2.179 vs 3.059; s321 2.200 vs 3.344 |
| 2 night 10 direction (reported) | KL below B {319: True, 320: True, 321: True}; NLL midpoint own-n1 PASS, B-n1 PASS | |
| 3.1 H-off n5-15 | **PASS** | {319: True, 320: True, 321: True} |
| 3.2 direct d1-2 hippocampus-off (n5-15) | **PASS** | {319: True, 320: True, 321: True}; n1-4 FAIL |
| 3.3 collapse | **PASS** | seeds-ever per probe arm <= B |
| 3.4 leak n10+15 | **PASS** | {319: True, 320: True, 321: True} |
| 3.5 transcript guard | **UNDEFINED here (see blind read)** | |

### Prereg section 4 decision, conditional on the transcript guard

- Numeric guards (3.1-3.4): all PASS. KL bar PASS; NLL bar own-n1 PASS, B-n1 PASS.
- Bar 1 (engagement): UNDEFINED (1a PASS 3/3; 1b s321 PASS, s319 proxy PASS, s320 undefined).
- Window sensitivity of guard 3.2 (the prereg names no window): nights 5-15 PASS; nights 1-4 would FAIL on s320 (B 5.59 vs K-P 4.34 of 6, diff -1.24, bar -1.00); nights 1-15 PASS. The KILL guard is therefore window-dependent for s320 and the nights-1-4 reading would KILL; the as-D2-PR reading does not.
- Decision by these numbers: **KEEP if the transcript guard also finds no new defect class and bar 1 is accepted (1b is UNDEFINED for s320 and proxy-only for s319, so section 3.1 formally leaves the arm UNDEFINED); KILLED if the transcript read finds a new defect class**.
- The default flips only on a registered six-seed TEST on a FRESH TEST split (prereg section 4; land-result). The device/precision confound (K-P fp16 vs B fp32) and the missing s319/s320 life banks remain caveats on any KEEP.
