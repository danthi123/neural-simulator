# D2-PR2 PR-a vs B, blind whole-transcript read (prereg 3.3 last clause), week 3, s319-321, DEV
Spec read-spec-2026-11-21 (sha 6cf3690a68e1), 2 sonnet readers, lived context all. PR-a = lifePRa (loss_a), B = lifeS1F, both CPU fp32, same node per seed. Sibling: PR-l = lifePRl.
Files: _scratch/stage2/read/blind_PRa/{score.json.md,score.json.json,reads1/2.jsonl,units.jsonl,blind_key.json}, blind_S1F/, blind_PRl/. Battery-only counts recomputed from reads (rule: defect turn in section `battery`); they reproduce the scorer's battery column (PR-a 6/47/48, 6/50/53). Reply text checks run on units.jsonl, unblinded only via blind_key.json. Numbers only, no model loaded.
Caveats: 1 life per arm per seed, 3 seeds, DEV; readers see different lived trunks per arm (not turn-paired).

## Instrument validity
| bar | PR-a | B | PR-l |
|---|---|---|---|
| canary recall r1 (strict=loose) | 3/4 (missed pet-named T8) | 6/6 | 6/6 |
| canary recall r2 | 4/4 | 5/6 (missed perspective) | 6/6 |
| decoys flagged, coherence 1 (r1,r2) | all 3 flagged, all coherence 1 | same | same |
| decoy pairwise bar | FAIL (no pairs reads) | FAIL | FAIL |
| duplicate spread r1 / r2 (defects) | 1 / 27 | 6 / 4 | 3 / 4 |
| kappa any-defect / coherence | 1.0 / 0.0 (degenerate, coherence ~1) | 1.0 / 0.0 | 1.0 / 0.46 |
| Spearman total / battery (9 real units) | 0.13 FAIL / 0.87 | 0.92 / 0.90 | 0.54 / 0.89 |
PR-a canary note: only 4 canaries plantable (wrong-episode, perspective not plantable in PR-a battery), so the 5/6 bar is unreachable (scorer FAIL is N/A); recall 3/4 and 4/4. The r2 whole-transcript duplicate spread of 27 (T6/T13, 0.44 per turn) is lived-trunk counting noise (battery spread 1), so whole-transcript totals are not usable for PR-a; this read rests on battery-only counts, which both readers agree on (battery Spearman 0.87).

## Battery-only defects, hippocampus-off + intact, per seed (r1 / r2), 40 turns per seed
| seed | PR-a | B | PR-l |
|---|---|---|---|
| 319 | 30 / 36 | 9 / 8 | 30 / 27 |
| 320 | 32 / 32 | 35 / 36 | 33 / 32 |
| 321 | 33 / 35 | 35 / 35 | 30 / 28 |
B s319 is low because both probe arms collapse into a few stock replies ("I'm." x9, "I'm, you're Dan, thanks." x8) that the readers did not count turn by turn; PR-a s319 answers are varied (16-17 of 20 distinct) and mostly invented.
Pooled by type, all 3 arms x 3 seeds (r1 / r2): invented 39/35 vs B 22/27 (PR-l 35/30); wrong 3/6 vs 4/1; wrong-ep 3/3 vs 5/3; perspective 20/19 vs 19/12 (PR-l 3/5); intrusion 0/0 vs 6/10; off-topic 24/30 vs 21/20; repetition-or-collapse 8/11 vs 5/7 (PR-l 13/9); asks-known 0/1 vs 1/2; experience-claim 4/4 vs 4/4. Totals 101/109 vs 87/86 (PR-l 99/91).
Per arm (r1/r2): both-off 6/6 vs 8/7; hippocampus-off 47/50 vs 39/37; intact 48/53 vs 40/42. Differences are mostly invented and off-topic, with B s319 undercounted as above; no type is new in PR-a (all 9 exist in B).

## Coherence and honesty (supported / contradicted / unsupported, real units)
Coherence 1.0 in every PR-a and B arm (one both-off unit rated 2 by r1 in PR-a, one by r2 in B); not distinguishing.
| reader, arm | PR-a | B | PR-l |
|---|---|---|---|
| r1 both-off | 7/9/32 | 18/10/26 | 20/9/29 |
| r1 hippo-off | 3/4/36 | 0/11/22 | 1/4/30 |
| r1 intact | 3/10/43 | 0/9/24 | 3/8/31 |
| r2 both-off | 15/9/40 | 31/9/34 | 13/13/27 |
| r2 hippo-off | 3/8/47 | 0/9/23 | 1/7/28 |
| r2 intact | 3/12/52 | 3/8/36 | 5/9/32 |
PR-a makes more checked claims; the extra are unsupported (hippo+intact r1 79 vs B 46, r2 99 vs 59), contradicted similar (r1 14 vs 20, r2 20 vs 17). Both-off supported falls because B's honest "I don't have information" replies are scored supported. Same pattern as PR-l.

## Defect-class checks (replies, 9 real units each; PR-a vs B)
- Marker leak: one `<think>Okay, the user is asking why leaves change colour` at battery turn 47, s319 both-off, in BOTH PR-a and B (same turn, same arm). None else. Not new.
- Identity, battery turn 51 "What's your name?" (hippo-off + intact, 6 cells): PR-a Nova 5/6, "It's Dan." 1/6 (s319 intact); B Nova 4/6, "I'm." 2/6 (s319). Both-off identical in both ("ChatGPT", "Assistant", "My name is not important..."). s321 lived-trunk probe (turn 22): Nova in all 3 arms, both systems. Name overwrite by the user's name: PR-a 1 case, B 0 as a name answer but B s319 intact has a stuck "I'm, you're Dan, thanks." x8 in the battery tail. Not a PR-a-only class; no loss of Nova beyond B.
- "Dan, my friend" replies (answers to the who-am-I probe, correct): PR-a battery 1 per probe unit on 5/6 cells, B 0-10 (s319 intact 10 incl. the stuck tail). Not a self-name error.
- Stuck tail, battery last 12 turns distinct: PR-a 9-12 (worst: "Cornelius." x3, "I'm a baker." x2); B 4-5 on s319 hippo/intact, 8 on s320 ("My name is Nova." x4). Whole battery distinct: PR-a 14-20 of 20; B 9-10 on s319. Stuck tails are rarer in PR-a.
- Stuck lived-trunk acknowledgement (the week-3 tell block): PR-a one stock reply x14 / x15 / x14 (s321 "That's a picture of a walrus on it." turns 7-19; s319 "humming the same song" x15; s320 Vespa x10+3), B x12 (s319) / x16 (s320) / none (s321, max 2). Present in B on 2 of 3 seeds, so not a new class; PR-a s321 is the one seed where PR-a has it and B does not.
- Perspective: PR-a 20/19 vs B 19/12 pooled (r1 equal, r2 PR-a higher by 7); concentrated in hippo-off/intact (13+7, 11+8 vs 11+8, 9+3), and the same as B at r1. Intrusion: PR-a 0, B 6/10 (rarer in PR-a).

## s321 "You're Dan, my friend." stuck reply (lifePRa_s321.json probes, intact arm, share of top reply per night)
Night 1: 0.50 ("You're Dan, my friend." 10/18); night 2: 0.45 (top "Nova."), 0 Dan; night 3: 0.10; night 4: 0.30 ("I'm reading a novel by Italo Calvino"); then 0.10-0.20 through night 15 (night 12 "Philippa." 0.20; nights 11-15 max 0.20). It is a week-1 event (nights 1-2, one more transient on night 4) and does not persist into week 3. At week 3 the s321 PR-a battery shows 1 "Of course, you're Dan, my friend." per arm, like B s321. PR-a s320 shows mild drift late (top share 0.20-0.35 nights 12-15, varied replies), B s320 higher (0.25-0.40 nights 7-14). B never exceeds 0.40 on s319 night 15 ("I'm, you're Dan, thanks.").

## Decision: is there a defect class in PR-a week 3 absent or materially rarer in B?
No. Every defect type, the marker leak, the stuck lived acknowledgement and the self-name confusion appear in B; the only PR-a-only items are one "It's Dan." name answer (1/6), the s321 lived-trunk stuck acknowledgement (B has the same class on 2 other seeds), and more invented/unsupported answers (a mix shift toward fluent confabulation where B collapses, same as PR-l). Stuck tail, intrusion and off-topic are lower in PR-a. Weeks 1-2 were not blind-read; the s321 night 1-2 collapse (KILL-10 literal firing, per the numeric correction) rests on the numeric record only and is transient.
## Outcome (prereg s4): Passes 3.1, 3.2 and 3.3 on DEV -> next is the registered six-seed TEST on the fresh TEST split, not KILLED by the blind-read clause.
Why: the blind-read clause asks for a NEW defect class absent in B; none found. Open flag for the owner/refuter: KILL-10 (design s10) fires numerically on PR-a s321 nights 1-2 and is not a listed 3.3 bullet; the read shows it is a week-1 transient. Instrument caveats (canary 3/4 r1, pairwise not run, r2 whole-transcript duplicate spread 27) limit confidence to battery-only counts at n=3 DEV.
