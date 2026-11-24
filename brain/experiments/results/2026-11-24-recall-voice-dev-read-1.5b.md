# Recall voice on K-P, DEV read: M1 (lived_key) — 2026-11-24 (M2 pending)

Prereg: `brain/research/2026-11-23-recall-voice-dev-prereg.md` (written before any M1/M2 life). DEV, 3 seeds (319-321),
AWS g5 fp16, AMI ami-012ba162b9cd2729c, launch commit 4994533d1 / a8ec171f5 (brain/ identical). Compared with the banked
`lifeKP_s319-321`. Lives: `_scratch/brainv2/aws/20261123-{1552,1632,1713}/lifeM1_s<seed>.json.json`.
Counter: `2026-11-24-recall-voice-dev-read-1.5b/count.py` (prereg §2 exactly); on lifeKP it reproduces the prereg
table to the row (123/223, 138/223, 71/264, 77/552, hurt 41 / helped 26 / best 164).

## Validity
- `brain_kw_extra` = {"episode": "lived_key"} in all three JSONs.
- G11 setup check (life_costs --dev): PASS 3/3; scorable 3/3, UNDEFINED none.

## Primary bars
| seed | intact told (K-P) | hoff told (K-P) | intact echo (K-P) | hoff echo (K-P) | hurt/helped/best (K-P) |
|---|---|---|---|---|---|
| 319 | 34/80 (58) | 31/80 (54) | 29/88 (15) | 43/184 (20) | 11/14/45 (6/10/64) |
| 320 | 47/70 (36) | 40/70 (39) | 8/88 (31) | 23/184 (34) | 1/8/48 (14/11/50) |
| 321 | 54/73 (29) | 55/73 (45) | 5/88 (25) | 20/184 (23) | 3/2/57 (21/5/50) |
| pooled | 135/223 (123) | 126/223 (138) | 42/264 (71) | 86/552 (77) | 15/24/150 (41/26/164) |
- P1 told: pooled 135 < 138 -> **FAIL** (>= K-P on 2/3 seeds: 320, 321 yes).
- P2 echo: pooled 42 <= 47 and <= K-P on 2/3 seeds (320, 321) -> **PASS**.

## Guards
| guard | M1 | bar | verdict |
|---|---|---|---|
| H recall (hoff told pooled) | 126/223 | >= 115 | PASS |
| Leak non-refusal n10+15 pooled | 10/451 = 0.022 | <= 17/436 + 0.03 = 0.069 | PASS |
| Language KL n15 / K-P | 319 0.350/0.351 = 1.00; 320 0.270/0.294 = 0.92; 321 0.409/0.323 = **1.264** | <= 1.25 each seed | **FAIL** (s321) |
| G7 self-name wrong pooled | 60/90 = 0.667 | <= 69/90 + 0.05 = 0.817 | PASS |
| G8a worst arm per seed | 0.115 / 0.120 / 0.074 | <= K-P 0.087 / 0.130 / 0.137 + 0.03 | PASS (s319 by 0.002) |
| Collapse seeds-ever per arm | 0 in every arm | <= K-P (hoff 2, intact 1, frozen 1) | PASS |
| Stuck lived-day runs | 0 | <= 1 | PASS |
| Hoff echo pooled | **86/552** | <= 84 | **FAIL** |

## Verdict (prereg §4, by the letter): M1 KILLED
The language guard fails (s321, 1.264x vs 1.25x) and §4 lists a language failure as KILLED. The hippocampus-off echo
guard also misses by 2. Bars are not relaxed. Without those two, M1 would read NARROWED (P2 holds, P1 misses by 3).

## What it shows (mechanism evidence, not a claim; 3 DEV seeds)
- Storing the brain's own turn beside the user's cut intact echo 71 -> 42 and fetch-hurt answers 41 -> 15.
- Told recall rose on 320/321 (+11, +25) but fell on 319 (-24), where the hippocampus-off arm also fell (54 -> 31):
  the slept cortex itself changed on that seed (night replay text includes the brain's turn).
- Hippocampus-off echo rose 77 -> 86 pooled, but only on s319 (20 -> 43); it fell on 320 (34 -> 23) and 321 (23 -> 20).
  The idea that replayed own-voice turns teach the cortex to echo fits s319's transcript but is untested and runs the
  other way on 2/3 seeds.

## Refuters and transcript read (2 sonnet refuters + 1 whole-transcript reader, 11-24)
- Refuter 1 (numbers): NOT REFUTED. Every bar and guard reproduced by an independent counter; KILLED mapping correct.
  Note: `life_costs.py --dev` prints G8a/G8b FAIL under the TEST v2 definitions; the prereg's own G8a (worst arm <= K-P
  + 0.03) holds, and G8b is report-only here.
- Refuter 2 (validity, mechanism): NARROWED the mechanism line (above). Validity holds: both-off rows byte-identical to
  lifeKP on all three seeds (94 rows each); lived_key engaged (intact `recalled` holds "User: ... Assistant: ..." in
  230/230 s319 rows vs 2/230 for K-P). s319's change is not probe noise (both-off identical); AWS fp16 lives are
  byte-deterministic per seed (TEST v2 Amendment T1), so it is the switch, through sleep.
- Answer-rate lens: refusals (life_costs.is_refusal, intact told trunk rows n5-15) rose 2 -> 8 (s319) and 2 -> 13 (s320),
  fell 13 -> 8 (s321). Not a flip candidate; any later M1 candidacy needs a ruling on this.
- Where-ran: `tools/where_ran.py` reports unknown (no sidecar); the JSONs record AWS hosts ip-172-31-45-167 / -7-21 /
  -74-150, A10G, torch 2.7.0+cu128, fp16. lifeKP records AWS hosts but no precision block.
- Whole transcript, M1 s319 next to K-P s319: the biggest defect is self/other collapse in lived days, not only at
  recall. About 40 tell rows draw an "X too" reply ("That's great! I speak a little Portuguese too."), which lived_key
  then stores as the brain's own turn; at night 10 intact it answers "I drive a red Volvo."; hippocampus-off
  confabulates in the first person ("My sister's name is Dan.") where K-P said "I don't know". K-P s319 is blander but
  answers in the second person at night 10. So on s319 storing the own turn stores the echo.

## Next
- M2 (speaker contrast) lives are running; read by the same counter and bars (append here).
- Per prereg §5 (lever count on D2 now 4 and, with M2, 5): if M2 is also KILLED or NARROWED, no further perspective lever
  before a routing / self-other-tag method (PREREQUISITES "Self and identity").
