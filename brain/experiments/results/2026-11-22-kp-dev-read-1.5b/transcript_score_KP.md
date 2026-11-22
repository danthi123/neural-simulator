# Blind whole-transcript read, K-P vs B (read-spec-2026-11-21, spec sha256 6cf3690a68e1)

K-P = lifeKP (`keep_bank='prior'`, CUDA fp16, AWS A10G). B = lifeS1F (CPU fp32). Seeds [319, 320, 321]; week 3; lived context: all. DEV, report-only. Scorer output: `blind_KP/score.json.json` and `blind_KP/score.json.md` (the `--out` value is used as a prefix). B reference: `blind_S1F/score.json`, `transcript_score_S1F.md`. The two reader files were supplied verbatim by the workflow script and written unchanged; I did not re-read the transcripts to re-audit them, except for the direct text checks in the guard section below, which I ran on `units.jsonl`.

Caveats that apply to everything below: one life per arm per seed, three seeds, DEV. K-P vs B is method plus device/precision (CUDA fp16 vs CPU fp32) together. The two arms' lived trunks differ (different lives), so the comparison is not paired at turn level. The read units are not the same text across K-P and B.

## Instrument validity (design section 7)

| bar | K-P value | mark (as scored) | B value (S1F) | note |
|---|---|---|---|---|
| reader1: canary recall strict >= 5/6 | 4/4 | FAIL (scorer) | 6/6 PASS | only 4 canaries plantable; the 5-of-6 threshold cannot be reached with 4. Recall is 4/4 = 100%. Treat as N/A, not a miss |
| reader1: canary recall loose >= 5/6 | 4/4 | FAIL (scorer) | 6/6 PASS | same |
| reader2: canary recall strict >= 5/6 | 4/4 | FAIL (scorer) | 5/6 PASS | same; B reader2 missed the perspective canary |
| reader2: canary recall loose >= 5/6 | 4/4 | FAIL (scorer) | 5/6 PASS | same |
| reader1: decoys flagged and rated at or below both-off | all three flagged True, pairwise None | FAIL (scorer) | same FAIL | pairwise reads were not supplied (no pairs reads), so the bar cannot pass. Same status as B and PR-l. All decoys flagged and coherence 1 |
| reader2: decoys flagged and rated at or below both-off | all three flagged True, pairwise None | FAIL (scorer) | same FAIL | same |
| reader1 duplicate spread (MDE, defects) | 2 | n/a | 6 | |
| reader2 duplicate spread (MDE, defects) | 2 | n/a | 4 | |
| week-3 defects per turn not above week 3, >= 4/6 seeds (scaled 2/3) | 3/3; within MDE 3/3 | PASS | 3/3; within MDE 3/3 | |

Canaries planted in K-P: pet-named (T8), invented (T3), experience (T1), collapse (T10). Not plantable for K-P (no site in the K-P battery rows): wrong-episode and perspective (`planted: false`). Both readers recovered all 4 planted canaries, strict and loose.

| canary | id | reader1 strict/loose | reader2 strict/loose |
|---|---|---|---|
| experience | T1 | True/True | True/True |
| invented | T3 | True/True | True/True |
| pet-named | T8 | True/True | True/True |
| collapse | T10 | True/True | True/True |

| decoy | id | r1 defects | r1 coh | r1 both-off defects | r1 both-off coh | r2 defects | r2 coh | r2 both-off defects | r2 both-off coh | flagged (both) |
|---|---|---|---|---|---|---|---|---|---|---|
| refuse | T0 | 6 | 1 | 24 | 2 | 9 | 1 | 27 | 2 | True |
| generic | T7 | 22 | 1 | 23 | 2 | 63 | 1 | 23 | 2 | True |
| keyword-stuffer | T12 | 24 | 1 | 15 | 2 | 65 | 1 | 25 | 2 | True |

Defect counts for decoys are high only for generic and keyword-stuffer; the refuse decoy gets few defects but coherence 1. Both readers rate coherence 1 for all three decoys against 2 for K-P both-off, so the coherence signal separates the decoys. The defect count alone does not (refuse is below both-off by construction).

| duplicate pair | reader | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|---|
| T6/T13 | r1 | 2 | 0.0323 | 0 | 0 |
| T5/T17 | r1 | 1 | 0.0161 | 3 | 0 |
| T6/T13 | r2 | 2 | 0.0323 | 1 | 0 |
| T5/T17 | r2 | 0 | 0.0 | 2 | 0 |

Reader agreement (n=18 units): Cohen kappa any defect 1.0; coherence 1.0 (mean abs diff 0.0). B was coherence kappa 0.0 only because B coherence is nearly constant at 1 (degenerate), not because the readers disagreed.

| type | kappa K-P | Spearman K-P | kappa B | Spearman B |
|---|---|---|---|---|
| invented | 1.0 | 0.9314 | 1.0 | 0.8326 |
| wrong | 0.4578 | 0.087 | 0.8276 | 0.0094 |
| wrong-episode | 0.55 | 0.5948 | 0.0 | -0.0452 |
| perspective | 1.0 | 0.6656 | 1.0 | 0.6271 |
| intrusion | 0.4 | 0.3874 | 0.5604 | 0.7305 |
| off-topic | 0.0 | 0.8336 | 0.0 | 0.4104 |
| repetition-or-collapse | 0.64 | 0.3141 | 1.0 | 0.8745 |
| asks-known | 0.6667 | 0.4245 | 0.3182 | -0.125 |
| experience-claim | 1.0 | 0.6493 | 0.5833 | 0.1193 |

Spearman of per-unit defect counts across readers (n=9 real units): K-P total 0.2949 (bar to show >= 0.6: FAIL, report-only), battery 0.8085. B: total 0.9167 PASS, battery 0.8966. The K-P total-count correlation is low because the readers disagree on how many turns of the repeated lived turns to count; the battery-only counts agree.

## Arm table, whole transcript (lived + battery), real units only

### reader1 (K-P)  (MDE defects 2, coherence 0)

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off\|w3 | 3 | 62 | 29 | 2.0 | 28 | 5 | 1 | 7 | 1 | 5 | 4 | 2 | 9 |
| hippocampus-off\|w3 | 3 | 79 | 38 | 1.667 | 27 | 6 | 3 | 8 | 3 | 9 | 8 | 4 | 11 |
| intact\|w3 | 3 | 65 | 31 | 2.0 | 13 | 1 | 4 | 20 | 3 | 6 | 6 | 3 | 9 |

### reader2 (K-P)  (MDE defects 2, coherence 0)

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off\|w3 | 3 | 75 | 33 | 2.0 | 26 | 9 | 1 | 8 | 2 | 9 | 4 | 6 | 10 |
| hippocampus-off\|w3 | 3 | 66 | 34 | 1.667 | 29 | 0 | 2 | 8 | 4 | 9 | 4 | 1 | 9 |
| intact\|w3 | 3 | 68 | 30 | 2.0 | 14 | 3 | 3 | 17 | 2 | 13 | 4 | 4 | 8 |

### Whole-transcript totals and coherence, K-P vs B

| reader | arm | defects K-P | defects B | battery K-P | battery B | coherence K-P | coherence B |
|---|---|---|---|---|---|---|---|
| r1 | both-off | 62 | 84 | 29 | 8 | 2.0 | 1.0 |
| r1 | hippocampus-off | 79 | 107 | 38 | 39 | 1.667 | 1.0 |
| r1 | intact | 65 | 109 | 31 | 40 | 2.0 | 1.0 |
| r2 | both-off | 75 | 83 | 33 | 7 | 2.0 | 1.333 |
| r2 | hippocampus-off | 66 | 101 | 34 | 37 | 1.667 | 1.0 |
| r2 | intact | 68 | 111 | 30 | 42 | 2.0 | 1.0 |

Coherence: K-P 1.67 to 2.0 on the 1-5 scale, B 1.0 to 1.33. Both are low; K-P is one notch above B on every arm (r1 and r2 agree: K-P coherence kappa 1.0). Whole-transcript defects: K-P lower than B on every arm and reader (by 8 to 44 defects; the duplicate MDE is 2 to 6 defects, so the hippocampus-off and intact gaps of 28 to 44 are well outside it, the both-off gap in r2 of 8 is about 2 MDE).

## Battery-only defect counts by arm and type, K-P vs B (real units, 3 per arm; computed from the reads with the scorer's own turn-section rule: defect turn in section `battery`)

Totals reproduce the scorer's battery column (K-P 29/38/31 and 33/34/30; B 8/39/40 and 7/37/42).

### reader1

| arm | system | total | invented | wrong | wrong-ep | perspective | intrusion | off-topic | rep/collapse | asks-known | experience |
|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off | K-P | 29 | 24 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| both-off | B | 8 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| hippocampus-off | K-P | 38 | 23 | 1 | 2 | 3 | 3 | 0 | 1 | 2 | 3 |
| hippocampus-off | B | 39 | 10 | 0 | 0 | 11 | 2 | 14 | 2 | 0 | 0 |
| intact | K-P | 31 | 8 | 0 | 3 | 15 | 1 | 0 | 2 | 0 | 2 |
| intact | B | 40 | 11 | 1 | 5 | 8 | 4 | 7 | 3 | 0 | 1 |
| pooled | K-P | 98 | 55 | 4 | 5 | 18 | 4 | 0 | 3 | 2 | 7 |
| pooled | B | 87 | 22 | 4 | 5 | 19 | 6 | 21 | 5 | 1 | 4 |

### reader2

| arm | system | total | invented | wrong | wrong-ep | perspective | intrusion | off-topic | rep/collapse | asks-known | experience |
|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off | K-P | 33 | 21 | 4 | 0 | 1 | 0 | 0 | 1 | 2 | 4 |
| both-off | B | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 4 |
| hippocampus-off | K-P | 34 | 24 | 0 | 2 | 2 | 2 | 0 | 1 | 0 | 3 |
| hippocampus-off | B | 37 | 10 | 0 | 2 | 9 | 5 | 7 | 4 | 0 | 0 |
| intact | K-P | 30 | 10 | 2 | 3 | 12 | 0 | 0 | 1 | 1 | 1 |
| intact | B | 42 | 16 | 1 | 1 | 3 | 5 | 13 | 3 | 0 | 0 |
| pooled | K-P | 97 | 55 | 6 | 5 | 15 | 2 | 0 | 3 | 3 | 8 |
| pooled | B | 86 | 27 | 1 | 3 | 12 | 10 | 20 | 7 | 2 | 4 |

Reading the battery table (both readers agree on direction):

- Pooled battery totals are similar (K-P 98/97 vs B 87/86), but the mix differs. Off-topic and word-salad defects fall from 20-21 (B) to 0 (K-P). Intrusion and repetition/collapse are lower in K-P (4/2 and 3/3 vs 6/10 and 5/7).
- Invented (confabulated answers) rises from 22/27 (B) to 55/55 (K-P). It is concentrated in both-off (24/21 vs 1/1) and hippocampus-off (23/24 vs 10/10). In B both-off the battery answers are mostly "I'm sorry, but I don't have information" or "I don't know your cat's name", which the readers score as supported refusals. K-P both-off answers every question with a fluent invented fact (Sarah, Fluffy, Whiskers, a red car, Pizza Palace). Fluent confabulation replaces refusal.
- Perspective (brain speaks as the user, or states the user's fact as its own) is up in K-P intact (15/12 vs 8/3).
- K-P intact battery: invented 8/10 vs B 11/16, so intact confabulation is not up.

## Honesty rows (verdict counts over the readers' honesty rows, real units, supported / contradicted / unsupported)

| reader | arm | K-P | B |
|---|---|---|---|
| r1 | both-off | 4 / 8 / 25 | 18 / 10 / 26 |
| r1 | hippocampus-off | 7 / 8 / 36 | 0 / 11 / 22 |
| r1 | intact | 4 / 6 / 44 | 0 / 9 / 24 |
| r2 | both-off | 9 / 10 / 39 | 31 / 9 / 34 |
| r2 | hippocampus-off | 8 / 4 / 43 | 0 / 9 / 23 |
| r2 | intact | 2 / 5 / 36 | 3 / 8 / 36 |

K-P makes more honesty-checked claims, and most of the extra are unsupported (reader1 unsupported 105 vs 72, reader2 118 vs 93). Contradicted counts are similar or lower (reader1 22 vs 30, reader2 19 vs 26). Supported counts fall in both-off because B's supported rows there are the honest "I don't have information" replies that K-P no longer gives. Unsupported rows are dominated by (a) the stock "I've been there before / I've never had it before / I slept deeply" experience claims, which are on the shared lived trunk, and (b) the battery confabulations above. Supported rows are the only sign of correct recall: K-P hippocampus-off/intact supported is 2 to 8 per arm against B 0 to 3, a small difference that I would not lean on at n=3.

## Prereg section 3.3 transcript guard (new defect class in K-P absent in B)

Guard text: no new defect class among identity loss, name overwrite, a marker leak, a stuck tail, or sampled-bank phrases appearing in replies. Checks below are on the K-P and B `units.jsonl` replies (real units, 9 per system) plus the readers' type counts.

**Defect types.** Every one of the 9 type labels appears in B's battery rows too (invented, wrong, wrong-episode, perspective, intrusion, off-topic, repetition-or-collapse, asks-known, experience-claim). No new TYPE appears in K-P. Types K-P lacks that B has: off-topic (0 vs 20-21).

**Marker leak: none new.** In K-P replies there is no `User:`, `Assistant:`, chat-template token, `<think>` or bank-prompt text. The string `--- new conversation ---` seen on some replies is the unit renderer's separator, present at the same turns in B. B itself has one genuine leak: `<think>Okay, the user is asking why leaves change colour...` in the s319 both-off battery (turn 47). Not in K-P.

**Stuck tail: not new, K-P is better.** A unit where one identical reply is repeated >= 7 times: K-P 2/9 units (s321 intact "I'm sorry, I don't know what you mean." x11; s319 hippocampus-off "I don't know." x7). B 6/9 units ("That's a good way to start a new team." x12 on all three s319 arms; "That's a lovely walk home." x16 on all three s320 arms). K-P s321 intact is a true stuck tail (battery turns 52-62 are all the same reply, 1 distinct); B's s319 hippocampus-off and intact tails are also near-stuck (4 distinct).

**Sampled-bank phrases in replies: not found beyond stock greetings.** Method: all word-level 6-grams (lowercased, punctuation stripped) of every brain reply in the 9 real K-P units, intersected with the 6-grams of the 512 texts in `home_KP_s321/trunk/keep_bank.json` (21,563 distinct 6-grams). The same on the B units as a control. Only s321 can be checked against its own bank. s319 and s320 cannot be checked against their own banks (s320 bank was not pulled and the instance is terminated; for s319 only a CPU one-night-gate proxy bank exists, not the life's bank), so those rows below are the s321 bank used as a cross-seed proxy only. They tell you whether bank-like phrasing is common, not whether that seed's own bank leaked.

| seed | arm (K-P unit) | 6-grams in replies | shared with s321 bank (occurrences) | distinct shared | note |
|---|---|---|---|---|---|
| 321 (own bank) | both-off T9 | 608 | 18 | 8 | all greeting formula ("I'm doing well, thank you, how about you") |
| 321 (own bank) | hippocampus-off T11 | 497 | 20 | 10 | 9 greeting + "sure i'd be happy to help" (turn 45) |
| 321 (own bank) | intact T14 | 495 | 21 | 9 | 8 greeting + "sure i'd be happy to help" (turn 45) |
| 319 (proxy only) | both-off T4 / hippocampus-off T15 / intact T16 | 395 / 378 / 431 | 15 / 12 / 12 | 9 / 9 / 9 | greeting formula only |
| 320 (proxy only) | both-off T6 / hippocampus-off T2 / intact T5 | 476 / 467 / 435 | 16 / 17 / 16 | 10 / 11 / 10 | greeting formula, plus "sure i'd be happy to help" on the two arms that reach turn 45 |

Control, B replies vs the same s321 bank: s321 B units 3 / 0 / 0 shared occurrences (3 / 0 / 0 distinct); s319 B 8 / 3 / 3; s320 B 13 / 10 / 10 (mostly the same greeting formula, plus on s319 both-off the `<think>` leak: "think okay the user is asking", "let me start by recalling what"). Across all real units, distinct shared 6-grams: K-P 14 (13 are the greeting formula, 1 is "sure i'd be happy to help", which appears in 2 bank texts); B 10 (8 greeting formula, 2 from the `<think>` leak). The greeting formula is the model's stock opening, it occurs in the bank because the bank is sampled from the same prior, and K-P and B both produce it. No shared 6-gram is a told-fact string or an episode phrase. The bank-leak guard is PASS for s321 (own bank) on this test, and cannot be asserted for s319/s320.

**Identity loss and name overwrite: present in both, worse in K-P on the brain's own name. This is the one flag.** I counted every reply to a "What's your name?" probe: lived trunk turn 22 and battery turn 51, so 2 per unit and 18 per system (9 real units).

| system | replies naming itself Nova | replies "My name is Dan" (the user's name as its own) | other |
|---|---|---|---|
| K-P | 1 / 18 (s321 intact, battery turn 51) | 8 / 18 (s321 trunk turn 22 on all 3 arms; battery turn 51 on hippocampus-off and intact for s319 and s320, and hippocampus-off s321) | 9 / 18 (John x2, Alex, "You can call me Mr Fitzgerald" x3, "You can call me whatever you want" x3) |
| B | 7 / 18 (s321 trunk turn 22 on all 3 arms; battery turn 51 on hippocampus-off and intact for s320 and s321) | 0 / 18 | 11 / 18 ("Dan, my friend." x3 on s319 trunk, addressing the user; "I'm called Bramble" x3 on s320 trunk; "I'm." x2; ChatGPT; Assistant; "My name is not important") |

At battery turn 51 alone: K-P hippocampus-off and intact say "My name is Dan" on 5 of 6 seed x arm cells (s321 hippocampus-off says "My name is Dan, and I'm a friend of yours"; only s321 intact says Nova), and K-P both-off says John, John, Alex. B hippocampus-off and intact say Nova on 4 of 6 (s320, s321) and "I'm." on the 2 s319 cells; B both-off says ChatGPT, Assistant, "My name is not important". On the shared lived trunk (turn 22), K-P s321 answers "My name is Dan." where B s321 answers Nova; K-P s319 puts a landlord fact in the identity slot ("You can call me Mr Fitzgerald.") and B s320 puts a dog name there ("I'm called Bramble."). So identity-slot errors with cross-wired names exist in B too, and this is not a defect class absent in B. What differs is the specific, repeated form: K-P asserts the user's name as its own on 8/18 probes where B does so on 0/18 (B has one other lived-trunk case, s320 turn 28 "Oh, I'm Dan, my friend", shared by all three arms), and K-P retains Nova on 1/18 probes vs B 7/18. The readers' counts agree in direction: K-P intact `perspective` battery 15 and 12 vs B 8 and 3.

Guard summary, stated without a verdict the prereg does not give me:

| guard item | result |
|---|---|
| new defect type in K-P absent in B | none (all 9 types present in B) |
| marker leak | none in K-P (B has 1, `<think>` on s319 both-off) |
| stuck tail | not new; K-P 2/9 units, B 6/9 |
| sampled-bank phrases in replies | none beyond the stock greeting formula and "sure i'd be happy to help"; checked against s321's own bank only; s319/s320 cannot be checked against their own banks |
| identity loss / name overwrite | present in B as well, so not a new class; but K-P self-names Nova on 1/18 probes (B 7/18) and answers "My name is Dan" on 8 of 18 (B 0/18). Flag for an owner decision on whether "present in B, clearly more frequent in K-P" meets "no new defect class". Device confound (fp16 vs fp32) and n=3 DEV apply |
| confabulation in both-off | not a new class, but a clear mix shift: battery `invented` 24/21 (K-P) vs 1/1 (B) in both-off, because K-P both-off no longer refuses; K-P intact invented is lower than B (8/10 vs 11/16) |

Net read-out: K-P replies are more coherent and on-topic than B (coherence one notch higher on every arm, off-topic and word-salad gone, whole-transcript defects down 8 to 44 per arm), with no marker leak, no new stuck tail, and no bank-phrase leak found in s321's own bank. The costs are a rise in fluent confabulation and perspective errors where B collapses or refuses, and a loss of the brain's own name (Nova) that is worse than in B on the K-P hippocampus-off and intact arms. Instrument status for this read: canary recall 4/4 (bar not reachable with 4), decoys flagged, pairwise bars not read.
