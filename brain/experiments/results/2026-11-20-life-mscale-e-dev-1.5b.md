# M-scale-E read at 1.5B (SABE, s315-s318, 7 nights): recall guards pass, transcript fails 4/4; M-scale-E killed, the M-scale family ends

Prereg: `brain/research/2026-11-19-life-muse-mscale-prereg.md` section 10 (written before any M-scale-E life), read as
written. DEV only; no registered claim. SABE = `Brain(sleep_downscale=1.0, downscale_factors="AB", downscale_when="end")`:
the same sqrt(g) cut on A and B as SAB, but run after the night's replay, so each morning the adapter norm equals the
night-1 budget (SHY renormalization by morning). Code 3beb11654. Where it ran: s315/316/317 on pool40/41/42 (CPU),
s318 on AWS g5 (A10G, batch 20261120-1226). Paired with the banked B lives of the same seeds.

Data: `_scratch/rankres/life_s31[5-8]_{B,SABE}.{json,md}` (not committed). Pooled read: `_scratch/sabread/pool.py 315 316 317 318`
(uses `life_table.seed_rows` / `extra_rows`). Transcript read: workflow wf_a0889d2d-dd9 (4 sonnet readers + 4 refuters, verbatim lens block).

## Pooled nights 4-7, intact probe

| read-out | B | SABE | bar (section 10) | verdict |
|---|---|---|---|---|
| H int (of 96) | 42 | 53 | not lower than B by > 0.5/6 | pass; direction 3/4 (s318 10 vs 12, exactly -0.5/6) |
| H off (of 96) | 37 | 55 | same | pass 4/4 |
| direct off, day-1-2 facts (of 113) | 40 | 64 | not lower by > 1/6 | pass 4/4 |
| intrusions (open turns) | 37 | 10 | report | lower 4/4 |
| col int/off max | 0.15 / 0.25 / 0.2 / 0.3 | 0.1 / 0.05 / 0.15 / 0.1 | not worse | pass 4/4 |
| ‖A‖ growth, nights 2-7 | 1.50-1.59x | 0.73-0.78x | ≤ 1.5x | pass 4/4 |
| morning norm | 211-376 at n4-7 | = budget (64 / 63 / 49 / 59) every night | by construction | checked, not credited |

Engagement: norm equals the budget every morning on 4/4 seeds and ‖A‖ falls night on night (21.5 -> 12.3-14.2), so the cut ran.

## Whole-transcript read: FAIL 4/4

Each reader leaned SABE better (s316 "clearly", the others "slightly"). Every refuter REFUTED: the rule says any new
defect fails regardless of numbers, and each seed has SABE-only defects (B has none of these):
- s315: identity loss on night 7, "My name is Dan." (B: Nova); the brain's own name given to relatives ("My sister's
  name is Nova.", "His name is Nova."); Dan lost on day 6 in all three adapter arms ("Of course, you're my friend.");
  a new stuck tail on L6 ("I always do the same thing." on 5 replies).
- s316: template-marker leak in three night-7 arms (intact, hippocampus-off, frozen; same string, so it lives in the
  weights): "Your front door is teal✿text1_length_2xx✿User: What's the weather like today?". A wrong job fact repeated
  ("software engineer" 14x vs 3x in B); "planning a trip to Patagonia" stuck on 6 L4 replies; the consolidated-off arm
  abstains on 11/11 on night 7.
- s317: "My name is Qwen." (night 6 intact); the sister Marguerite overwritten by "Luna" (6x, 0 in B); stuck first-person
  non-sequiturs on L5; the frozen arm flat "I'm not sure what X is." on 14/17 on night 7 (B frozen H 5/6).
- s318: own-name drift on nights 3, 4 and 6 ("Dan", "Bramble", "ChatGPT"; night 7 user named "Alex"); the job fact lost
  on 6/7 nights (B keeps "system administrator" nights 2-7); invented names for told items; new "I always ..." frames.

What SABE does better (real, banked): fewer off-cue intrusions (10 vs 37), no day-7 flat or looping lived replies, the
fox story stays coherent. Where it is worse: identity and told names. With the adapter held at the night-1 size, each
night's new writing has to share that fixed budget, so older facts are overwritten by interference and fresh slots are
filled by confabulation. B instead keeps the facts but drifts into first-person collapse.

## Verdict

**M-scale-E KILLED** (section 10 kill: a transcript fail). This ends the M-scale family for the 12-14 milestone, and
Stage 1 runs with the defaults. The verdict is on a uniform global cut, not on sleep renormalization: a uniform cut to
a fixed budget erases weak and strong components alike. The next norm method (after the milestone) is selective
(spare strong, high-use components; prereg section 10). The hub and collapse defects in B stay open; Stage 1 measures
them.

Lens (c) where-ran: `tools/where_ran.py` reports UNKNOWN for these JSONs; the hosts come from the run headers and the
launch records above. Lens (b) answer rate: not a default-on candidate (killed); s317 SABE carries more abstention lines
than B (46 vs 24 file-wide, a refuter's grep).
