# K-P vs B on the SAME device (AWS g5 CUDA fp16), 1.5B DEV seeds 319/320/321
Prereg: brain/research/2026-11-21-sleep-language-keep-prereg.md (s3 bars, s4 decision, A1 s5, A2 s6); test2 prereg G7/G8 (brain/research/2026-11-23-stage2-test2-prereg.md s5).
Arms: K-P = lifeKP_s319-321, B-fp16 = lifeBg_s319-321 (both AWS g5 cuda), B-cpu = lifeS1F (pool CPU fp32). Read only, no model run.
Files (folder 2026-11-23-kp-vs-bfp16-same-device-1.5b/): kp_bars.py/.md (DEV script, B arm swapped to lifeBg), g7g8_device.py/.md (G7/G8/G10 + device effect).
G7/G8 scorers are my re-implementation of the prereg regexes (life_costs.py does not exist yet); they reproduce the prereg's DEV validation exactly (K-P G7 wrong .87/.87/.57, B-cpu .40/.10/.23; G8a K-P worst arm .087/.130/.137, B 0).

## VERDICT: post-hoc same-device SENSITIVITY read, not a prereg verdict (corrected after 2 refuters, both REFUTED the first headline)
The prereg names B = lifeS1F (CPU); against it the DEV read stands (KEEP on the guards read; bar 1 UNDEFINED). Swapping in B-fp16 is a
post-hoc analysis. Under it: the KL/NLL, H-off and leak wins hold; G7 self-name (wrong 0.767 vs 0.333) and G8 refusal (38/582 vs 0/582)
costs hold, so the identity gate (kp-identity prereg) is still the right next lever. Guard 3.3 (collapse, seeds-ever count from
life_bars, not a prereg-defined statistic) comes out worse for K-P here (hippocampus-off 2 vs 1, intact 1 vs 0) but its sign is
device-sensitive: same-method B alone moves intact 2 -> 0 between CPU and CUDA, one cell (B-fp16 s320 intact n1, 7/18 = 0.389) sits just
under the 0.4 line, K-P's mean repeated-reply share is LOWER (0.084 vs 0.145), and with s320 (bank UNDEFINED) excluded hippocampus-off ties
1 vs 1. So 3.3 is UNRESOLVED on this data, not a kill. Precision (fp16, TF32) is inferred from the cortex.py default, not recorded in
the JSONs; each life ran on a different AWS instance. Original reader wording kept below for the record.
ORIGINAL (superseded): KILLED on guard 3.3 (collapse not worse than B). Against CPU B the same guard passed (that was the DEV KEEP). The KL/NLL win, H-off and leak wins hold; the recall margins shrink; both costs hold or grow. Caveat: 3.3 is 3 seeds, one life each, B-fp16 collapse counts are low (see below).
Bar 1 stays UNDEFINED (1b: s321 own bank PASS, s319 CPU proxy PASS, s320 bank not pulled). 3.5 transcript guard not recomputed here (no model/reader); the earlier blind read compared to CPU B and is device-unaffected only if B-fp16 transcripts look alike: PENDING (needs life_read.py blind readers on lifeBg).

## Per-seed bars (319 / 320 / 321). B = B-fp16
| bar | B-fp16 | K-P | verdict |
|---|---|---|---|
| 1a keep record 15/15, 512, distinct hashes, keep KL>0 n2+ | n/a | 15/15 all seeds, min keep KL .071/.088/.088 | PASS 3/3 |
| 2 KL n15 | 2.093/2.693/2.616 (pool 2.468) | .351/.294/.323 (pool .323) | PASS: ratio .131 (bar .5), below B 3/3 |
| 2 KL n10 | 1.737/1.604/1.731 (pool 1.691) | .262/.236/.258 (pool .252) | ratio .149, 3/3 |
| 2 NLL n15 (midpoint rule) | 3.872/4.480/4.365 | 2.258/2.179/2.200 | PASS 3/3 both night-1 readings (midpoints 2.92-3.24) |
| 2 KL n15/n1 (ref bar 2x) | 13.9x/15.2x/17.2x | 5.3x/4.2x/3.5x | still above 2x |
| 3.1 H-off n5-15 /6 (K-P >= B-0.5) | 2.82/3.00/2.45 | 4.00/3.27/3.82 | PASS; margin +1.18/+0.27/+1.36 (vs CPU B: +1.45/+1.45/+2.0) |
| 3.2 d1-2 H-off n5-15 /6 (>= B-1) | 2.51/3.04/2.37 | 3.97/3.29/3.63 | PASS +1.46/+0.25/+1.26 |
| 3.2 same, nights 1-4 | 4.34/5.38/5.38 | 4.97/4.34/4.34 | diff +.62/-1.03/-1.03: FAIL on 2 seeds (bar -1.0; vs CPU B only s320 failed). n1-15: PASS |
| 3.3 collapse seeds-ever (K-P <= B) | hoff 1, intact 0, int-frozen 1, cons-off 0 | hoff 2, intact 1, int-frozen 1, cons-off 0 | FAIL (hoff 2>1, intact 1>0). K-P max share hoff .35/.55/.55, intact .15/.10/.55; B-fp16 hoff .35/.35/.75, intact .20/.39/.35 |
| 3.4 leak n10+15 | 47/156, 48/152, 40/152 (.29 pooled) | 0/156, 3/152, 14/152 (.037) | PASS 3/3 |
| G7 self-name wrong rate (<= B+.25) | .467/.167/.367 (30/90 pooled) | .867/.867/.567 (69/90) | FAIL 2/3 seeds (+.40/+.70/+.20); pooled +.43 |
| G8a refusal on answerable rows (<= B+.03) | 0/582 every cell | hoff .087/.130/.000; intact .019/.022/.137 (38/582) | FAIL 3 of 6 cells |
| G8b night share >= .6 | max 0.00 | s319 hoff n15 .73, s320 hoff n11 and n12 1.00 each, s321 intact n15 1.00 | FAIL (4 nights, B 0 in each) |
| G10 experience regex, lived trunk (indicative) | 4/1/1 of 134 | 9/12/20 of 134 | K-P higher; denominators differ from frozen scorer |

## Self-name and refusal cost: it holds, it does not shrink
- Self-name: K-P wrong 0.767 (69/90) vs B-fp16 0.333 vs B-cpu 0.244. "My name is Dan" (user's name) K-P 28 + "I'm Dan" 2 of 90 vs B-fp16 6 (+ "Dan, my friend"), B-cpu 5+3. Friend-name regex 41 (K-P) vs 7 (B-fp16) vs 8 (B-cpu). Intact K-P 31/45 vs 18/45; hoff 38/45 vs 12/45. B-fp16 is a bit WORSE than CPU B on name loss (0.33 vs 0.24), so the same-device gap (+.43) is smaller than the CPU gap (+.52) but G7 still fails 2/3.
- Refusal: B-fp16 0/582 and B-cpu 0/582 vs K-P 38/582 (6.5%); 11 literal "don't know what you mean" replies vs 0/0. Whole-night refusals are K-P only on both B arms. Device does not touch this.

## Device effect (B-fp16 vs B-cpu, same method, same seeds)
- Held-out KL n1 pooled .160 vs .155 (1.03x). n15 2.468 vs 2.724 (0.91x); NLL n15 4.239 vs 4.509 (-0.27). Per seed n15 KL fp16/cpu .71/1.13/.92, NLL diff -.88/+.37/-.31; sign mixed. Between-seed range within an arm (KL n15 2.09-2.69 fp16, 2.38-2.93 cpu) is as large as the device gap. KL n15/n1: fp16 13.9/15.2/17.2x, cpu 22.7/13.6/17.7x. Mid-life nights differ more (n5 KL ratio .69-.83, pooled .755), so fp16 B drifts a little slower early.
- Leak identical in size (135/460 fp16 vs 135/460 cpu). Collapse is lower in fp16 B (seeds-ever intact 0 vs 2, hoff 1 vs 2, cons-off 0 vs 1): this is what flips guard 3.3. H-off n5-15 is higher in fp16 B by +.27/+1.18/+.63 of 6; d1-2 n5-15 by +.08/+1.40/+.71; n1-4 mixed (+1.03/-.21/+2.28).
- Read: the device effect on held-out NLL/KL is small (<=10% pooled, within seed noise); on behavioural guards (collapse, H-off, d1-2) it is the same size as the K-P margins. So the language win is device-robust; the behavioural margins and the collapse guard are not.

## Engagement evidence
keep_bank record 15/15 nights x 3 seeds, bank_size 512, subset 64, 15 distinct hashes, mean keep KL .071-.112 from night 2. Bank: s321 AWS own bank 512/512 distinct, 63.5 tok, 1.2% told key; s319 CPU proxy 511/512, 63.4, 0.4%; s320 undefined. Cheaper confirmation of language effect: adapter norm n15 358-370 K-P vs 637/673/659 B-fp16. B-fp16 lives have no keep_bank records (0/15).

## Caveats
1 life per arm per seed, DEV only; different AWS host per life. TF32 on CUDA applies to both AWS arms (so confound closed, but fp32-CPU-vs-CUDA difference includes TF32). Collapse seeds-ever is noisy at n=3 (one seed-arm flips it). 3.2 nights-1-4 margin is -1.03 vs -1.00. PENDING (needs model or blind readers): 3.5 / G5 transcript read on lifeBg, G9 pairwise, interview AUROC, frozen life_costs.py G7-G11 (G11 not reproduced here, omitted), s320 bank.
