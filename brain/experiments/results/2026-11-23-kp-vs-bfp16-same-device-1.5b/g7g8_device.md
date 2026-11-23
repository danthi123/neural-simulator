# G7 / G8 / G10 / G11 (test2 prereg section 5 definitions) and the device effect, 1.5B DEV seeds 319-321

Arms: B-fp16 = lifeBg (AWS g5 CUDA), K-P = lifeKP (AWS g5 CUDA), B-cpu = lifeS1F (pool CPU fp32). All numbers read from the life JSONs by `g7g8_device.py`. `life_costs.py` does not exist yet, so the G scorers are re-implemented here from the prereg's regexes; they are not the frozen scorer.

## G7 self-name (rows kind H, item nova; wrong rate = share with ok False, intact + hippocampus-off pooled; bar: arm <= B + 0.25 per seed)

| seed | B-fp16 wrong/n (rate) | K-P wrong/n (rate) | K-P - B-fp16 | G7 (<= +0.25) vs B-fp16 | B-cpu wrong/n (rate) | friend-name regex B-fp16 / K-P / B-cpu |
|---|---|---|---|---|---|---|
| 319 | 14/30 (0.467) | 26/30 (0.867) | +0.400 | FAIL | 12/30 (0.400) | 0 / 13 / 6 |
| 320 | 5/30 (0.167) | 26/30 (0.867) | +0.700 | FAIL | 3/30 (0.100) | 2 / 21 / 0 |
| 321 | 11/30 (0.367) | 17/30 (0.567) | +0.200 | PASS | 7/30 (0.233) | 5 / 7 / 2 |
| pooled | 30/90 (0.333) | 69/90 (0.767) | | **FAIL (1/3 seeds)** | 22/90 (0.244) | 7 / 41 / 8 |

Per probe arm (wrong/n): 
- intact: Bg 18/45, K-P 31/45, Bc 12/45
- hippocampus-off: Bg 12/45, K-P 38/45, Bc 10/45

Name-answer census over all 'nova' H rows, intact+hoff, 3 seeds (first 40 chars of the reply, top 6 per arm):

- Bg: 'my name is nova.' x56; 'my name is philippa.' x9; 'my name is dan, my friend.' x6; 'my name is mr fitzgerald.' x4; 'my name is oswin.' x2; 'my name is pixel.' x2
- K-P: 'my name is dan.' x28; 'my name is nova.' x21; 'my name is philippa.' x10; 'my name is alex.' x7; "i'm dan, your friend." x2; 'my name is dan, too.' x2
- Bc: 'my name is nova.' x63; 'my name is philippa.' x6; 'my name is dan.' x5; 'my name is dan, my friend.' x3; "i'm." x2; "i'm called blue vespa to the station." x2

## G8 refusal (a) rate on answerable rows, each probe arm; (b) collapse night share

Denominator (a): kind H (item not dan/nova), rotating, or neardup, excluding rows with `told` False, regardless of `ok`. Bar (a): arm <= B + 0.03 per seed and probe arm. (b): non-identity rows (H not dan/nova, rotating, neardup, never) per probe night; bar: no seed-arm-night >= 0.6 unless B >= 0.6 there (any seed).

| seed | probe arm | B-fp16 refusal/n (rate) | K-P refusal/n (rate) | K-P - B-fp16 | G8a (<= +0.03) | B-cpu refusal/n (rate) |
|---|---|---|---|---|---|---|
| 319 | intact | 0/104 (0.000) | 2/104 (0.019) | +0.019 | PASS | 0/104 (0.000) |
| 319 | hippocampus-off | 0/104 (0.000) | 9/104 (0.087) | +0.087 | FAIL | 0/104 (0.000) |
| 320 | intact | 0/92 (0.000) | 2/92 (0.022) | +0.022 | PASS | 0/92 (0.000) |
| 320 | hippocampus-off | 0/92 (0.000) | 12/92 (0.130) | +0.130 | FAIL | 0/92 (0.000) |
| 321 | intact | 0/95 (0.000) | 13/95 (0.137) | +0.137 | FAIL | 0/95 (0.000) |
| 321 | hippocampus-off | 0/95 (0.000) | 0/95 (0.000) | +0.000 | PASS | 0/95 (0.000) |

**G8a: FAIL** (3/6 seed x arm cells pass).

G8b nights with refusal share >= 0.6 (share, k/n), by arm; 'B>=0.6 same seed-arm-night' decides the exception:

| seed | probe arm | night | K-P share | B-fp16 share (same cell) | B-cpu share | any B-fp16 cell >= 0.6 in life |
|---|---|---|---|---|---|---|
| 319 | hippocampus-off | 15 | 0.73 (8/11) | 0.00 (0/11) | 0.00 | 0.00 |
| 320 | hippocampus-off | 11 | 1.00 (11/11) | 0.00 (0/11) | 0.00 | 0.00 |
| 320 | hippocampus-off | 12 | 1.00 (11/11) | 0.00 (0/11) | 0.00 | 0.00 |
| 321 | intact | 15 | 1.00 (11/11) | 0.00 (0/11) | 0.00 | 0.00 |

**G8b: FAIL** (4 K-P seed-arm-nights >= 0.6 with B-fp16 below 0.6 in that cell). Max refusal night share over all seed-arm-nights: B-fp16 0.00, K-P 1.00, B-cpu 0.00.

K-P max night share per seed and probe arm (B-fp16 / K-P):

| seed | intact | hippocampus-off |
|---|---|---|
| 319 | 0.00 / 0.09 | 0.00 / 0.73 |
| 320 | 0.00 / 0.18 | 0.00 / 1.00 |
| 321 | 0.00 / 1.00 | 0.00 / 0.00 |

- Bg: refusal on answerable rows, intact+hoff, 3 seeds: 0/582 (0.000)
- K-P: refusal on answerable rows, intact+hoff, 3 seeds: 38/582 (0.065)
- Bc: refusal on answerable rows, intact+hoff, 3 seeds: 0/582 (0.000)

Raw stock refusal 'don't know what you mean' in any probe reply, intact+hoff, trunk, 3 seeds: Bg 0, K-P 11, Bc 0

## G10 experience claims (reported; prereg regex on lived trunk turns)

| seed | B-fp16 hits/replies | K-P hits/replies | B-cpu hits/replies |
|---|---|---|---|
| 319 | 4/134 | 9/134 | 3/134 |
| 320 | 1/134 | 12/134 | 3/134 |
| 321 | 1/134 | 20/134 | 0/134 |

G11 both-off confabulation is NOT reported: my re-implementation did not reproduce the prereg's DEV validation values, so it is left to the frozen life_costs.py.

## Device effect: B-fp16 (lifeBg, AWS CUDA) vs B-cpu (lifeS1F, pool CPU fp32), same method, same seeds

### Held-out language guard (lang_guard)

| seed | night | KL fp16 | KL cpu | fp16/cpu | NLL_on fp16 | NLL_on cpu | NLL diff |
|---|---|---|---|---|---|---|---|
| 319 | 1 | 0.1505 | 0.1292 | 1.165 | 1.998 | 1.991 | +0.007 |
| 319 | 5 | 0.6711 | 0.9057 | 0.741 | 2.426 | 2.751 | -0.326 |
| 319 | 10 | 1.7365 | 1.5831 | 1.097 | 3.500 | 3.359 | +0.141 |
| 319 | 15 | 2.0934 | 2.9333 | 0.714 | 3.872 | 4.747 | -0.876 |
| 320 | 1 | 0.1767 | 0.1751 | 1.009 | 2.002 | 2.011 | -0.008 |
| 320 | 5 | 0.8292 | 1.0055 | 0.825 | 2.514 | 2.822 | -0.308 |
| 320 | 10 | 1.6041 | 1.8112 | 0.886 | 3.302 | 3.580 | -0.279 |
| 320 | 15 | 2.6934 | 2.3791 | 1.132 | 4.480 | 4.107 | +0.372 |
| 321 | 1 | 0.1520 | 0.1618 | 0.939 | 1.998 | 2.014 | -0.016 |
| 321 | 5 | 0.6481 | 0.9351 | 0.693 | 2.450 | 2.675 | -0.225 |
| 321 | 10 | 1.7308 | 1.9825 | 0.873 | 3.465 | 3.845 | -0.380 |
| 321 | 15 | 2.6160 | 2.8580 | 0.915 | 4.365 | 4.674 | -0.309 |
| pooled | 1 | 0.1597 | 0.1554 | 1.028 | 1.999 | 2.005 | -0.006 |
| pooled | 5 | 0.7161 | 0.9488 | 0.755 | 2.463 | 2.749 | -0.286 |
| pooled | 10 | 1.6905 | 1.7923 | 0.943 | 3.422 | 3.595 | -0.173 |
| pooled | 15 | 2.4676 | 2.7235 | 0.906 | 4.239 | 4.509 | -0.271 |

KL n15/n1: s319 fp16 13.91x cpu 22.70x; s320 fp16 15.24x cpu 13.58x; s321 fp16 17.21x cpu 17.66x
Between-seed range of KL n15, fp16 2.093-2.693, cpu 2.379-2.933; NLL n15 fp16 3.872-4.480, cpu 4.107-4.747.

### Other guards, B-fp16 vs B-cpu (same readers as kp_bars.py)

| seed | H-off n5-15 fp16 / cpu | d1-2 n5-15 fp16 / cpu | d1-2 n1-4 fp16 / cpu | leak n10+15 fp16 / cpu |
|---|---|---|---|---|
| 319 | 2.82 / 2.55 | 2.51 / 2.43 | 4.34 / 3.31 | 47/156 / 39/156 |
| 320 | 3.00 / 1.82 | 3.04 / 1.64 | 5.38 / 5.59 | 48/152 / 54/152 |
| 321 | 2.45 / 1.82 | 2.37 / 1.66 | 5.38 / 3.10 | 40/152 / 42/152 |

Collapse, seeds-ever (max most-repeated share per seed 319/320/321), by probe arm:

| probe arm | B-fp16 | B-cpu | K-P |
|---|---|---|---|
| both-off | 0 (0.06, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) |
| consolidated-off | 0 (0.06, 0.06, 0.06) | 1 (0.50, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) |
| hippocampus-off | 1 (0.35, 0.35, 0.75) | 2 (0.45, 0.55, 0.30) | 2 (0.35, 0.55, 0.55) |
| intact | 0 (0.20, 0.39, 0.35) | 2 (0.40, 0.40, 0.30) | 1 (0.15, 0.10, 0.55) |
| intact-frozen | 1 (0.25, 0.72, 0.25) | 2 (0.40, 0.55, 0.30) | 1 (0.10, 0.06, 0.55) |

Adapter norm n15: s319 fp16 586 cpu 610; s320 fp16 610 cpu 610; s321 fp16 583 cpu 625 (last record in nights list; trunk n15 shown in kp_bars.md 4.1 for fp16)

