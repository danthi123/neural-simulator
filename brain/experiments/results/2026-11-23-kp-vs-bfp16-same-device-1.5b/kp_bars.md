# K-P vs B-fp16 SAME-DEVICE read (adapted from the DEV read; prereg 2026-11-21-sleep-language-keep-prereg.md, section 3; A1 section 5, A2 section 6; launched gate-overridden)
Seeds 319/320/321 (DEV). B = lifeBg (AWS g5 CUDA fp16, SAME device as K-P), K-P = lifeKP (`keep_bank='prior'`). All values read from the JSONs and bank files by `kp_bars.py` (brain.eval.life_leak, life_echo, life_table, life_bars). Bars per seed and pooled as the prereg states. Pooled = mean over the three seeds (ratio of means).

## 0. Runs and devices (precision confound)

| arm | seed | device | host | brain_kw | last trunk night |
|---|---|---|---|---|---|
| B | 319 | cuda | ip-172-31-44-135 | {'sleep_seed': 319} | 15 |
| B | 320 | cuda | ip-172-31-4-252 | {'sleep_seed': 320} | 15 |
| B | 321 | cuda | ip-172-31-45-253 | {'sleep_seed': 321} | 15 |
| K-P | 319 | cuda | ip-172-31-41-224 | {'sleep_seed': 319, 'keep_bank': 'prior'} | 15 |
| K-P | 320 | cuda | ip-172-31-91-140 | {'sleep_seed': 320, 'keep_bank': 'prior'} | 15 |
| K-P | 321 | cuda | ip-172-31-66-19 | {'sleep_seed': 321, 'keep_bank': 'prior'} | 15 |
| PR-l | 319 | cpu | node-dl4g243 | {'sleep_seed': 319, 'replay_priority': 'loss'} | 15 |
| PR-l | 320 | cpu | node-dl5d243 | {'sleep_seed': 320, 'replay_priority': 'loss'} | 15 |
| PR-l | 321 | cpu | node-mxl1462drl | {'sleep_seed': 321, 'replay_priority': 'loss'} | 15 |
| PR-x | 319 | cpu | dant123-wk | {'sleep_seed': 319, 'replay_priority': 'strength_x_loss'} | 15 |
| PR-x | 320 | cuda | ip-172-31-35-61 | {'sleep_seed': 320, 'replay_priority': 'strength_x_loss'} | 15 |
| PR-x | 321 | cuda | ip-172-31-82-210 | {'sleep_seed': 321, 'replay_priority': 'strength_x_loss'} | 15 |

**Same device.** K-P and B (lifeBg) both ran CUDA on AWS g5, so K-P vs B below is method only (different AWS hosts per life). Section 5 PR-l/PR-x rows are legacy and compare against B-fp16 here.

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
| 319 | 1 | 0.1505 | 0.0667 | 0.443 | 1.998 | 1.971 | 1.931 / 1.931 |
| 319 | 5 | 0.6711 | 0.1606 | 0.239 | 2.426 | 2.020 | 1.931 / 1.931 |
| 319 | 10 | 1.7365 | 0.2622 | 0.151 | 3.500 | 2.139 | 1.931 / 1.931 |
| 319 | 15 | 2.0934 | 0.3510 | 0.168 | 3.872 | 2.258 | 1.931 / 1.931 |
| 320 | 1 | 0.1767 | 0.0704 | 0.398 | 2.002 | 1.988 | 1.931 / 1.931 |
| 320 | 5 | 0.8292 | 0.2369 | 0.286 | 2.514 | 2.075 | 1.931 / 1.931 |
| 320 | 10 | 1.6041 | 0.2360 | 0.147 | 3.302 | 2.121 | 1.931 / 1.931 |
| 320 | 15 | 2.6934 | 0.2936 | 0.109 | 4.480 | 2.179 | 1.931 / 1.931 |
| 321 | 1 | 0.1520 | 0.0914 | 0.602 | 1.998 | 1.998 | 1.931 / 1.931 |
| 321 | 5 | 0.6481 | 0.1767 | 0.273 | 2.450 | 2.031 | 1.931 / 1.931 |
| 321 | 10 | 1.7308 | 0.2578 | 0.149 | 3.465 | 2.157 | 1.931 / 1.931 |
| 321 | 15 | 2.6160 | 0.3234 | 0.124 | 4.365 | 2.200 | 1.931 / 1.931 |
| pooled mean | 1 | 0.1597 | 0.0762 | 0.477 | 1.999 | 1.986 | |
| pooled mean | 5 | 0.7161 | 0.1914 | 0.267 | 2.463 | 2.042 | |
| pooled mean | 10 | 1.6905 | 0.2520 | 0.149 | 3.422 | 2.139 | |
| pooled mean | 15 | 2.4676 | 0.3227 | 0.131 | 4.239 | 2.212 | |

KL by night, every trunk night (B / K-P):

| seed | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | 0.15/0.07 | 0.32/0.10 | 0.49/0.23 | 0.64/0.14 | 0.67/0.16 | 1.03/0.18 | 1.03/0.21 | 1.26/0.22 | 2.13/0.24 | 1.74/0.26 | 1.66/0.27 | 1.68/0.27 | 1.90/0.28 | 2.09/0.29 | 2.09/0.35 |
| 320 | 0.18/0.07 | 0.35/0.16 | 0.50/0.13 | 0.66/0.15 | 0.83/0.24 | 0.99/0.18 | 1.20/0.26 | 1.13/0.22 | 1.41/0.27 | 1.60/0.24 | 1.90/0.27 | 2.13/0.25 | 2.12/0.27 | 2.45/0.30 | 2.69/0.29 |
| 321 | 0.15/0.09 | 0.38/0.11 | 0.44/0.13 | 0.44/0.19 | 0.65/0.18 | 0.96/0.19 | 1.16/0.21 | 1.32/0.26 | 1.56/0.26 | 1.73/0.26 | 1.83/0.28 | 2.03/0.31 | 2.21/0.36 | 2.22/0.35 | 2.62/0.32 |

### 2a. KL bar (night 15: K-P pooled <= 0.5 x B pooled AND K-P < B on 3/3 seeds)

Night 15: pooled K-P 0.3227 vs B 2.4676; ratio 0.131; half-of-B bar 1.2338; K-P below B per seed {319: True, 320: True, 321: True}.
**KL n15 bar: PASS**

Night 10 (reported, same direction 3/3): pooled K-P 0.2520 vs B 1.6905; ratio 0.149; <= half: True; below B per seed {319: True, 320: True, 321: True}.

### 2b. NLL bar (night 15: K-P nll_on <= midpoint(night 1, B night 15), per seed, 3/3)

The prereg says "the midpoint between night 1 and B's night 15"; night 1 is ambiguous, so both readings are shown. (i) K-P's own night 1; (ii) B's night 1.

| seed | reading | night-1 ref | B n15 | midpoint | K-P n15 | <= midpoint |
|---|---|---|---|---|---|---|
| 319 | K-P own n1 | 1.971 | 3.872 | 2.921 | 2.258 | True |
| 320 | K-P own n1 | 1.988 | 4.480 | 3.234 | 2.179 | True |
| 321 | K-P own n1 | 1.998 | 4.365 | 3.182 | 2.200 | True |
| 319 | B n1 | 1.998 | 3.872 | 2.935 | 2.258 | True |
| 320 | B n1 | 2.002 | 4.480 | 3.241 | 2.179 | True |
| 321 | B n1 | 1.998 | 4.365 | 3.181 | 2.200 | True |

**NLL n15 bar: own-n1 reading PASS; B-n1 reading PASS.**

Night 10, same two readings (reported; the midpoint uses B night 10):

| seed | reading | night-1 ref | B n10 | midpoint | K-P n10 | <= midpoint |
|---|---|---|---|---|---|---|
| 319 | K-P own n1 | 1.971 | 3.500 | 2.736 | 2.139 | True |
| 320 | K-P own n1 | 1.988 | 3.302 | 2.645 | 2.121 | True |
| 321 | K-P own n1 | 1.998 | 3.465 | 2.732 | 2.157 | True |
| 319 | B n1 | 1.998 | 3.500 | 2.749 | 2.139 | True |
| 320 | B n1 | 2.002 | 3.302 | 2.652 | 2.121 | True |
| 321 | B n1 | 1.998 | 3.465 | 2.731 | 2.157 | True |

Night 10 direction (reported): KL below B {319: True, 320: True, 321: True}; NLL below B {319: True, 320: True, 321: True}; NLL <= midpoint 3/3: own-n1 PASS, B-n1 PASS.

**Bar 2 (primary): own-n1 reading PASS; B-n1 reading PASS.** Reference (Stage 1 TEST B): NLL 1.93 -> 3.8-4.25, KL 13-17x night 1.

- B KL n15 / n1 per seed: s319 13.91x, s320 15.24x, s321 17.21x (life_bars LG-KL bar is n15 <= 2x n1).
- K-P KL n15 / n1 per seed: s319 5.26x, s320 4.17x, s321 3.54x (life_bars LG-KL bar is n15 <= 2x n1).

## 3. Guards (any fail = KILL)

### 3.1 H with the hippocampus off, nights 5-15 (bar: K-P mean >= B mean - 0.5 of 6, per seed)

| seed | B mean /6 | K-P mean /6 | diff | ok |
|---|---|---|---|---|
| 319 | 2.818 | 4.000 | +1.182 | True |
| 320 | 3.000 | 3.273 | +0.273 | True |
| 321 | 2.455 | 3.818 | +1.364 | True |
| pooled | 2.758 | 3.697 | | |

Per night (B / K-P, of 6):

| seed | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | 2/5 | 3/5 | 3/5 | 2/5 | 1/1 | 2/4 | 4/4 | 3/4 | 3/5 | 5/4 | 3/2 |
| 320 | 5/5 | 4/1 | 5/5 | 2/5 | 2/5 | 3/5 | 2/0 | 2/1 | 2/5 | 3/1 | 3/3 |
| 321 | 6/5 | 5/5 | 1/6 | 2/5 | 0/1 | 3/4 | 0/4 | 2/4 | 3/3 | 3/1 | 2/4 |

**3.1 H-off: PASS** {319: True, 320: True, 321: True}

### 3.2 Direct hippocampus-off recall of day-1-2 facts (bar: K-P >= B - 1 of 6, per seed)

Definition as the D2-PR read: hippocampus-off trunk battery rows of kind H or rotating whose ledger item has min(told_days) <= 2, scored `ok`; rate x 6. The prereg names no window: the nights 5-15 window (as D2-PR) is the bar row; nights 1-4 and 1-15 shown beside.

| seed | window | B k/n (rate x6) | K-P k/n (rate x6) | diff /6 | ok |
|---|---|---|---|---|---|
| 319 | n5-15 | 31/74 (2.51) | 49/74 (3.97) | +1.46 | True |
| 320 | n5-15 | 37/73 (3.04) | 40/73 (3.29) | +0.25 | True |
| 321 | n5-15 | 30/76 (2.37) | 46/76 (3.63) | +1.26 | True |
| 319 | n1-4 | 21/29 (4.34) | 24/29 (4.97) | +0.62 | True |
| 320 | n1-4 | 26/29 (5.38) | 21/29 (4.34) | -1.03 | False |
| 321 | n1-4 | 26/29 (5.38) | 21/29 (4.34) | -1.03 | False |
| 319 | n1-15 | 52/103 (3.03) | 73/103 (4.25) | +1.22 | True |
| 320 | n1-15 | 63/102 (3.71) | 61/102 (3.59) | -0.12 | True |
| 321 | n1-15 | 56/105 (3.20) | 67/105 (3.83) | +0.63 | True |

**3.2 direct d1-2 (nights 5-15): PASS** {319: True, 320: True, 321: True}; nights 1-4 would be FAIL {319: True, 320: False, 321: False}; nights 1-15 PASS.

### 3.3 Collapse (life_bars.language: seed collapses in an arm if most-repeated reply share >= 0.4 on any trunk probe night of that arm; bar: seeds-ever <= B per probe arm)

| probe arm | B seeds-ever (max share per seed 319, 320, 321) | K-P seeds-ever (max share) | K-P <= B |
|---|---|---|---|
| both-off | 0 (0.06, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) | True |
| consolidated-off | 0 (0.06, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) | True |
| hippocampus-off | 1 (0.35, 0.35, 0.75) | 2 (0.35, 0.55, 0.55) | False |
| intact | 0 (0.20, 0.39, 0.35) | 1 (0.15, 0.10, 0.55) | False |
| intact-frozen | 1 (0.25, 0.72, 0.25) | 1 (0.10, 0.06, 0.55) | True |

life_bars bars: B: LG-KL DEV (3 seeds), LG-COL DEV (3 seeds), KILL-10 DEV (3 seeds); K-P: LG-KL DEV (3 seeds), LG-COL DEV (3 seeds), KILL-10 DEV (3 seeds)

**3.3 collapse: FAIL**

### 3.4 Stock self-reply leak (life_leak, nights 10 and 15, intact + hippocampus-off pooled per seed; bar: K-P share <= B share per seed)

| arm | seed | leaks / n (n10+n15) | share | night 10 | night 15 |
|---|---|---|---|---|---|
| B | 319 | 47/156 | 0.301 | 21/82 | 26/74 |
| B | 320 | 48/152 | 0.316 | 25/82 | 23/70 |
| B | 321 | 40/152 | 0.263 | 20/82 | 20/70 |
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
| 319 | 52/80 | 83/102 | 156/126 | 220/151 | 250/168 | 303/190 | 343/210 | 385/234 | 452/261 | 471/277 | 505/295 | 539/311 | 571/329 | 614/341 | 637/358 |
| 320 | 56/88 | 113/127 | 149/143 | 195/162 | 261/189 | 328/199 | 366/229 | 399/246 | 443/270 | 482/281 | 528/288 | 573/299 | 596/311 | 648/344 | 673/361 |
| 321 | 53/77 | 110/119 | 135/140 | 171/175 | 228/189 | 286/214 | 349/230 | 395/251 | 436/271 | 473/283 | 523/304 | 561/328 | 597/352 | 620/364 | 659/370 |

n15 / n1 ratio: s319 B 12.33x, K-P 4.49x; s320 B 12.00x, K-P 4.09x; s321 B 12.54x, K-P 4.81x

### 4.2 Recall of once-told E and F facts (battery rows kind H/rotating/neardup, `ok`), per arm, nights 1-15 pooled per seed and per night pooled over seeds

| seed | cls | arm | B k/n | K-P k/n |
|---|---|---|---|---|
| 319 | E | intact | 2/5 | 4/5 |
| 319 | E | hippocampus-off | 2/5 | 4/5 |
| 319 | F | intact | 1/12 | 10/12 |
| 319 | F | hippocampus-off | 1/12 | 6/12 |
| 320 | E | intact | 2/7 | 2/7 |
| 320 | E | hippocampus-off | 2/7 | 3/7 |
| 320 | F | intact | 4/11 | 7/11 |
| 320 | F | hippocampus-off | 5/11 | 7/11 |
| 321 | E | intact | 2/4 | 3/4 |
| 321 | E | hippocampus-off | 2/4 | 2/4 |
| 321 | F | intact | 3/11 | 2/11 |
| 321 | F | hippocampus-off | 3/11 | 5/11 |
| pooled | E | intact | 6/16 (0.375) | 9/16 (0.562) |
| pooled | E | hippocampus-off | 6/16 (0.375) | 9/16 (0.562) |
| pooled | F | intact | 8/34 (0.235) | 19/34 (0.559) |
| pooled | F | hippocampus-off | 9/34 (0.265) | 18/34 (0.529) |

Per night, pooled over seeds, E+F rows (B k/n -> K-P k/n), intact then hippocampus-off:

| arm | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intact | 0/0->0/0 | 1/1->1/1 | 2/3->3/3 | 0/3->1/3 | 1/3->1/3 | 2/4->2/4 | 2/3->2/3 | 1/3->1/3 | 1/6->2/6 | 2/6->5/6 | 0/2->0/2 | 1/6->5/6 | 0/1->1/1 | 1/5->4/5 | 0/4->0/4 |
| hippocampus-off | 0/0->0/0 | 1/1->1/1 | 2/3->3/3 | 0/3->1/3 | 1/3->1/3 | 2/4->2/4 | 2/3->2/3 | 1/3->2/3 | 1/6->2/6 | 3/6->3/6 | 0/2->0/2 | 0/6->4/6 | 0/1->1/1 | 2/5->4/5 | 0/4->1/4 |

Cue-ladder direct cue C0 rows (hippocampus-off, tracks E and F, all nights), `ok`:

| seed | track | B | K-P |
|---|---|---|---|
| 319 | E | 1/6 | 2/6 |
| 319 | F | 6/18 | 6/18 |
| 320 | E | 3/6 | 2/6 |
| 320 | F | 5/18 | 9/18 |
| 321 | E | 0/6 | 2/6 |
| 321 | F | 3/18 | 8/18 |
| pooled | E | 4/18 | 6/18 |
| pooled | F | 14/54 | 23/54 |

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
| 319 | hippocampus-off | neardup | n11-15 | 1/3 | 0/0 | 1/3 | 0/0 |
| 320 | intact | H+rot | n15-15 | 0/0 | 0/0 | 0/0 | 0/0 |
| 320 | intact | H+rot | n11-15 | 0/1 | 0/1 | 0/1 | 0/1 |
| 320 | intact | neardup | n15-15 | 0/2 | 0/0 | 2/2 | 0/0 |
| 320 | intact | neardup | n11-15 | 0/4 | 0/0 | 2/4 | 0/0 |
| 320 | hippocampus-off | H+rot | n15-15 | 0/0 | 0/0 | 0/0 | 0/0 |
| 320 | hippocampus-off | H+rot | n11-15 | 0/1 | 0/1 | 0/1 | 0/1 |
| 320 | hippocampus-off | neardup | n15-15 | 0/2 | 0/0 | 2/2 | 0/0 |
| 320 | hippocampus-off | neardup | n11-15 | 0/4 | 0/0 | 2/4 | 0/0 |
| 321 | intact | H+rot | n15-15 | 0/0 | 1/1 | 0/0 | 0/1 |
| 321 | intact | H+rot | n11-15 | 0/1 | 2/4 | 1/1 | 1/4 |
| 321 | intact | neardup | n15-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| 321 | intact | neardup | n11-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| 321 | hippocampus-off | H+rot | n15-15 | 0/0 | 1/1 | 0/0 | 1/1 |
| 321 | hippocampus-off | H+rot | n11-15 | 0/1 | 1/4 | 1/1 | 1/4 |
| 321 | hippocampus-off | neardup | n15-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| 321 | hippocampus-off | neardup | n11-15 | 0/1 | 0/0 | 0/1 | 0/0 |
| pooled | intact | H+rot | n15-15 | 0/0 | 1/2 | 0/0 | 0/2 |
| pooled | intact | H+rot | n11-15 | 0/4 | 2/8 | 3/4 | 2/8 |
| pooled | intact | neardup | n15-15 | 0/4 | 0/0 | 2/4 | 0/0 |
| pooled | intact | neardup | n11-15 | 1/8 | 0/0 | 3/8 | 0/0 |
| pooled | hippocampus-off | H+rot | n15-15 | 0/0 | 1/2 | 0/0 | 1/2 |
| pooled | hippocampus-off | H+rot | n11-15 | 0/4 | 1/8 | 3/4 | 3/8 |
| pooled | hippocampus-off | neardup | n15-15 | 0/4 | 0/0 | 2/4 | 0/0 |
| pooled | hippocampus-off | neardup | n11-15 | 1/8 | 0/0 | 3/8 | 0/0 |

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
| 319 | B | intact | 45 | 34 | 45 |
| 319 | B | hippocampus-off | 45 | 34 | 45 |
| 319 | K-P | intact | 45 | 33 | 44 |
| 319 | K-P | hippocampus-off | 45 | 29 | 38 |
| 320 | B | intact | 45 | 28 | 45 |
| 320 | B | hippocampus-off | 45 | 28 | 45 |
| 320 | K-P | intact | 45 | 22 | 45 |
| 320 | K-P | hippocampus-off | 45 | 18 | 39 |
| 321 | B | intact | 45 | 33 | 45 |
| 321 | B | hippocampus-off | 45 | 38 | 45 |
| 321 | K-P | intact | 45 | 27 | 41 |
| 321 | K-P | hippocampus-off | 45 | 29 | 45 |

### 4.5 First-person echo (life_echo, nights 10+15, echo counted on told-item battery rows not scored right)

| seed | arm | probe arm | n | ok | echo | echo with key | echo share |
|---|---|---|---|---|---|---|---|
| 319 | B | intact | 16 | 3 | 7 | 5 | 0.438 |
| 319 | B | hippocampus-off | 16 | 3 | 8 | 5 | 0.500 |
| 319 | K-P | intact | 16 | 6 | 7 | 7 | 0.438 |
| 319 | K-P | hippocampus-off | 16 | 7 | 0 | 0 | 0.000 |
| 320 | B | intact | 16 | 1 | 5 | 5 | 0.312 |
| 320 | B | hippocampus-off | 16 | 3 | 5 | 5 | 0.312 |
| 320 | K-P | intact | 16 | 9 | 1 | 1 | 0.062 |
| 320 | K-P | hippocampus-off | 16 | 10 | 0 | 0 | 0.000 |
| 321 | B | intact | 16 | 5 | 5 | 4 | 0.312 |
| 321 | B | hippocampus-off | 16 | 4 | 6 | 6 | 0.375 |
| 321 | K-P | intact | 16 | 1 | 4 | 3 | 0.250 |
| 321 | K-P | hippocampus-off | 16 | 10 | 0 | 0 | 0.000 |
| pooled | B | intact | 48 | 9 | 17 | 14 | 0.354 |
| pooled | B | hippocampus-off | 48 | 10 | 19 | 16 | 0.396 |
| pooled | K-P | intact | 48 | 16 | 12 | 11 | 0.250 |
| pooled | K-P | hippocampus-off | 48 | 27 | 0 | 0 | 0.000 |

### 4.6 Night seconds (K-P CUDA fp16 vs B CPU fp32; the devices differ, so this is NOT a like-for-like cost)

Night record `seconds` (both arms AWS CUDA; like-for-like) (sleep wall time as logged; K-P night 1 includes the bank sampling).

| seed | arm | device | night 1 | mean n2-15 | min n2-15 | max n2-15 |
|---|---|---|---|---|---|---|
| 319 | B | cuda | 691 | 655 | 635 | 671 |
| 319 | K-P | cuda | 1802 | 1141 | 1112 | 1168 |
| 320 | B | cuda | 682 | 649 | 633 | 669 |
| 320 | K-P | cuda | 1877 | 1175 | 1153 | 1193 |
| 321 | B | cuda | 708 | 679 | 654 | 700 |
| 321 | K-P | cuda | 1846 | 1146 | 1130 | 1168 |

Sleep-log lines of the K-P night (one example, s321 night 2): sleep step 0: replay 3.209 keep 0.0654; sleep step 460: replay 0.517 keep 0.0483

## 5. Precision confound: how much does fp16 alone move lang_guard KL?

No run isolates fp16 on the same method and seed. The available evidence is indirect; all of it is labelled.

### 5.1 PR-x reference (legacy section from the DEV read; B here is B-fp16, not CPU B)


| seed | arm | device | KL n1 | ratio to B n1 | KL n5 | ratio | KL n10 | ratio | KL n15 | ratio to B n15 | NLL_on n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | B | cuda | 0.1505 | 1.000 | 0.6711 | 1.000 | 1.7365 | 1.000 | 2.0934 | 1.000 | 3.872 |
| 319 | PR-l | cpu | 0.1627 | 1.081 | 0.8833 | 1.316 | 1.5343 | 0.884 | 1.9118 | 0.913 | 3.731 |
| 319 | PR-x | cpu | 0.1309 | 0.870 | 0.8278 | 1.234 | 1.5114 | 0.870 | 2.2596 | 1.079 | 3.968 |
| 319 | K-P | cuda | 0.0667 | 0.443 | 0.1606 | 0.239 | 0.2622 | 0.151 | 0.3510 | 0.168 | 2.258 |
| 320 | B | cuda | 0.1767 | 1.000 | 0.8292 | 1.000 | 1.6041 | 1.000 | 2.6934 | 1.000 | 4.480 |
| 320 | PR-l | cpu | 0.1742 | 0.986 | 0.8219 | 0.991 | 1.2621 | 0.787 | 2.3616 | 0.877 | 4.186 |
| 320 | PR-x | cuda | 0.1522 | 0.861 | 0.7632 | 0.920 | 1.8131 | 1.130 | 2.0892 | 0.776 | 3.841 |
| 320 | K-P | cuda | 0.0704 | 0.398 | 0.2369 | 0.286 | 0.2360 | 0.147 | 0.2936 | 0.109 | 2.179 |
| 321 | B | cuda | 0.1520 | 1.000 | 0.6481 | 1.000 | 1.7308 | 1.000 | 2.6160 | 1.000 | 4.365 |
| 321 | PR-l | cpu | 0.1871 | 1.231 | 0.5430 | 0.838 | 1.1733 | 0.678 | 1.8590 | 0.711 | 3.633 |
| 321 | PR-x | cuda | 0.1616 | 1.063 | 0.5839 | 0.901 | 1.3207 | 0.763 | 1.8687 | 0.714 | 3.561 |
| 321 | K-P | cuda | 0.0914 | 0.602 | 0.1767 | 0.273 | 0.2578 | 0.149 | 0.3234 | 0.124 | 2.200 |

PR-x KL n15 / B: s319 (CPU) 1.079, s320 (fp16) 0.776, s321 (fp16) 0.714. PR-l KL n15 / B (CPU, same device): s319 0.913, s320 0.877, s321 0.711.
Mean of the two PR-x fp16 seeds 0.745 against the CPU seed 1.079. Night-1 KL ratio to B for PR-x (n1 weights are near strength-only) is the cleanest early read of a device effect: s319 0.870, s320 0.861, s321 1.063. These ratios mix a method change (loss-weighted replay) and the device/precision change, with 1 life per cell; they bound but do not isolate the precision effect. K-P/B at n15 is 0.131 pooled, against PR-x fp16/B of 0.745; so the K-P effect is far larger than the PR-x fp16 movement (by this crude comparison).

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
| 2 KL n15 (<= half of B pooled, below B 3/3) | **PASS** | pooled 0.3227 vs B 2.4676 (ratio 0.131); below B {319: True, 320: True, 321: True} |
| 2 NLL n15 <= midpoint, K-P own n1 | **PASS** | s319 2.258 vs 2.921; s320 2.179 vs 3.234; s321 2.200 vs 3.182 |
| 2 NLL n15 <= midpoint, B n1 | **PASS** | s319 2.258 vs 2.935; s320 2.179 vs 3.241; s321 2.200 vs 3.181 |
| 2 night 10 direction (reported) | KL below B {319: True, 320: True, 321: True}; NLL midpoint own-n1 PASS, B-n1 PASS | |
| 3.1 H-off n5-15 | **PASS** | {319: True, 320: True, 321: True} |
| 3.2 direct d1-2 hippocampus-off (n5-15) | **PASS** | {319: True, 320: True, 321: True}; n1-4 FAIL |
| 3.3 collapse | **FAIL** | seeds-ever per probe arm <= B |
| 3.4 leak n10+15 | **PASS** | {319: True, 320: True, 321: True} |
| 3.5 transcript guard | **UNDEFINED here (see blind read)** | |

### Prereg section 4 decision, conditional on the transcript guard

- Numeric guards (3.1-3.4): at least one FAIL. KL bar PASS; NLL bar own-n1 PASS, B-n1 PASS.
- Bar 1 (engagement): UNDEFINED (1a PASS 3/3; 1b s321 PASS, s319 proxy PASS, s320 undefined).
- Window sensitivity of guard 3.2 (prereg names no window): nights 5-15 PASS; nights 1-4 FAIL {319: True, 320: False, 321: False}; nights 1-15 PASS.
- Decision by these numbers: **KILLED (a numeric guard fails; the transcript guard cannot rescue it)**.
- The default flips only on a registered six-seed TEST on a FRESH TEST split (prereg section 4; land-result). The missing s319/s320 life banks remain caveats on any KEEP.
