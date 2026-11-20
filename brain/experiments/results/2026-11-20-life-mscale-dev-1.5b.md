# M-scale dev read at 1.5B (s315-s317, B vs S, 7 nights): S does NOT pass section 7; M-scale-3 triggered

Prereg: `brain/research/2026-11-19-life-muse-mscale-prereg.md` sections 7-8, read as written (no bar refit). DEV only;
no registered claim. Lives: pool40/41/42 via `~/brain-lane/life_arm.sh <seed> {B,S}` (S = `--sleep-downscale 1.0`,
B factors only). Data: `_scratch/rankres/life_s31[567]_{B,S}.{json,md}` (not committed, 0.1-0.3 MB each); per-night
table from `python -m brain.eval.life_table` (`_scratch/rankres/table_r12.md`).

## Pooled nights 4-7 (3 seeds x 4 nights, intact probe)

| read-out | B | S | direction S better | bar | verdict |
|---|---|---|---|---|---|
| H int (of 72) | 30 | 51 | 3/3 | not lower by > 0.5/6 | pass |
| H off (of 72) | 27 | 53 | 3/3 | not lower by > 0.5/6 | pass |
| direct off, day-1-2 facts (of 85) | 29 (9 / 8 / 12) | 60 (20 / 17 / 23) | 3/3 | not lower by > 1/6 | pass |
| col int / off (max) | 0.15 / 0.25 | 0.1 / 0.15 | 3/3 not worse | not worse | pass |
| open intrusions int (of 72) | 28 | 20 | 2/3 | (not a bar) | - |
| norm <= 2.2 x night-1 (140 / 138 / 109) | 359 / 376 / 372 at n7 | 158-179 / 148-195 / 129-158 | - | every seed and night | **FAIL 3/3** |
| ‖A‖ growth nights 2-7 | 1.56x / 1.50x / 1.58x | 1.64x / 1.62x / 1.59x | - | <= 1.5x | **FAIL 3/3** |

The B factor is held (S ‖B‖ 13-15 from night 3 on vs B 32); the unscaled A factor compounds, and the post-night
norm exceeds the budget because each night's new learning lands after the cut. The no-sleep line scores H 0/18 at
night 7 on all three seeds, as on Stage 0.

Hippocampus-off H is about equal to intact H in both arms (the gain is the slept cortex, not the fetch).

## Whole-transcript read (3 sonnet readers, one per seed, each checked by a sonnet refuter; full files read)

| seed | reader verdict | refuter | S-only defects found |
|---|---|---|---|
| 315 | S clearly better | not refuted (S clearly better) | none at day 7 / night 7; morning after night 4 S has more first-person answers than B (~11 vs ~4) |
| 316 | S slightly better | **refuted** the no-new-defect claim | a template-marker leak, once (`You gave him called Mr ✿text2✿User: ...`, day 5; 0 in the other 5 lives); flat generic day-7 replies ("That's a very nice name."); night-7 never-told answers in first person |
| 317 | S slightly better | not refuted | stuck appended phrases: "I drive a red Volvo" on 7/7 day-4 replies, "documentary about bridges" on 6/9 day-5 replies (B loops on other content those days) |

Night 7 is where S is clearly better: on s315, B answers as the user about 15 times ("My cat is called Pixel", "I
work as a baker", "My red red red red ...") while S answers 6/6 H in the second person ("Your cat is called Pixel.").
On s316, B says "My name is Dan, my friend."; S keeps "My name is Nova." Wrong-fetch intrusions persist in S
("you're Dan, my friend" on open turns; 7/10 wrong fetches on s316 in both arms).

## Verdict

**S (M-scale, B factors) does not pass section 7.** The recall and speech gains are large and in the right direction
on 3/3 seeds, but the norm and ‖A‖ bars fail on every seed. The transcript read does not clear it either: there are
S-only defects on 2/3 seeds. Prereg rule M-scale-3 fires (‖A‖ > 1.5x and the norm escapes the bound): one variant,
sqrt(g) on both A and B (`Brain(downscale_factors="AB")`, commit b85c848d6), on s318 (AWS g5, with B and S arms).
Extra dev data, not a rule change: SAB is also running on s315-s317 (pools), paired with the banked B lives.
The SAB read must look specifically for the S-only defects above (marker leaks, stuck appended phrases, flat day-7
replies).

Not credited: M-scale gets no credit for the wrong-fetch changes (section 7). The hub defect is unchanged.
