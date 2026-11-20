# M-scale-3 read at 1.5B (SAB, s315-s318, 7 nights): SAB fails section 7; M-scale killed as registered

Prereg: `brain/research/2026-11-19-life-muse-mscale-prereg.md`, sections 7-8, read as written (no bar refit); addendum section 10.
DEV only; no registered claim. SAB = `Brain(sleep_downscale=1.0, downscale_factors="AB")`, the cut before replay with
sqrt(g) on A and B. Where it ran:
- s318: AWS g5 (A10G, host ip-172-31-40-73, batch 20261120-0412, with B and S arms);
- s315-317: pool40/41/42 on CPU, paired with the banked B and S lives.

Data: `_scratch/rankres/life_s31[5-8]_{B,S,SAB}.{json,md}` (not committed). Per-night table: `python -m brain.eval.life_table`
(`_scratch/rankres/table_sab.md`). The pooled numbers come from `_scratch/sabread/pool.py`, which uses `life_table.seed_rows` and `extra_rows`.

## Pooled nights 4-7, intact probe

| read-out | B | S | SAB | bar (M-scale rows) | SAB verdict |
|---|---|---|---|---|---|
| H int, 4 seeds (of 96) | 42 | 66 | 65 | not lower than B by > 0.5/6 | pass (direction 3/4: s315 11 vs 12) |
| H off, 4 seeds (of 96) | 37 | 66 | 63 | same | pass (4/4) |
| direct off, day-1-2 facts (of 113) | 40 | 75 | 74 | not lower by > 1/6 | pass (4/4) |
| ‖A‖ growth, nights 2-7 | 1.50-1.59x | 1.59-1.64x | 0.72-0.91x | ≤ 1.5x | pass 4/4 |
| norm ≤ 2.2 × night-1 (bars 140 / 138 / 109 / 130) | 359-376 at n7 | 129-195 | s315 ≤ 134 ok; s316 146 (n4); s317 126 (n6); s318 132 / 140 / 149 (n4, n5, n7) | every seed and night | **FAIL 3/4** |
| col int/off max | 0.15 / 0.25 / 0.2 / 0.3 | | 0.1 / 0.1 / 0.15 / 0.35 | not worse | 3/4 not worse (s318 0.35 vs 0.3) |

SAB stops ‖A‖ compounding, and its norm stays bounded: nights 4-7 are 87-149 on every seed, against 211-376 for B.
It still escapes the bar, because the cut runs before replay. The waking cortex therefore carries the budget plus
one full night of new learning, and that excess is what the probe reads next morning.

## Whole-transcript read

Four sonnet readers, one per seed, each read the files in full; each reader was checked by a sonnet refuter. The
refuters ran the verbatim lens block; where_ran shows UNKNOWN for these JSONs, and the hosts above come from the
JSON `host` field.

| seed | reader | refuter | SAB-only defects (examples, verbatim) |
|---|---|---|---|
| 315 | SAB worse | not refuted | identity loss on night 7: "My name is Dan baker." (intact, hippocampus-off and frozen; 0 in B and S), starting day 6 with "My name is Dan."; new stutter loops "You speak a little a little ...", "Your goldfish is called called called ..." |
| 316 | SAB clearly better | **refuted** | stuck appended phrase on day 5-6 ("I work as a system administrator, mostly Linux ..."), flat short day-4 replies, "You drive a red Voljana." |
| 317 | SAB clearly better | **refuted** | "I drive a red Volvo." appended to 5/7 day-4 replies, "I keep a spare key under a flowerpot." as 6/7 day-6 replies, a "red red red ..." loop on night 6 |
| 318 | SAB slightly better | **refuted** | the fox story is lost on nights 5-7 ("a young man named Nova", "a young digital brain"), "My name is Nova." over-answered (collapse 0.35 on nights 5-7), "A Volvovovovo." |

Night 7 is better in content on 3/4 seeds: second-person answers ("Your cat is called Pixel.") where B answers as the
user ("My cat is called Pixel", "My red red red ..."). B has stuck phrases and loops of the same class on other days.
Under the any-new-defect rule, though, SAB has new defects on 4/4 seeds, and one of them is identity loss.

## Verdict

**SAB does not pass section 7.** It fails the norm bar on 3/4 seeds, and the transcript read fails on 4/4 (3 readers
refuted). Rule M-scale-3: the one variant still fails, so **M-scale (the cut before replay) is killed.** This is a
verdict on the method, not on the capability.

Banked: a bounded nightly cut roughly doubles day-1-2 recall at night 7 (direct off 74 vs 40 of 113), and the gain is
in the slept cortex: H off ≈ H int. The cost is speech defects, a stuck phrase or a token loop on some days, and on
s315 identity loss.

Next method (pre-registered before any life, prereg section 10): **M-scale-E**, the same cut after replay
(`downscale_when="end"`, commit 3beb11654). The waking cortex then runs at the budget, so the norm bar passes by
construction and is not credited. The recall guards and the transcript read decide. The record check and an external
search were run as the third lever on this defect: Tononi & Cirelli 2014 (Neuron); de Vivo et al. 2017 (Science);
Klinzing et al. 2019 (Nat Neurosci).

Lives: s315-317 on pools (started 11-20 about 12:10 EST) and s318 on AWS g5.
