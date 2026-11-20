# Offline screens P0-use / P0-scale on banked Stage-0 homes (1.5B, morning after night 7): s310 in, s311/s312 pending

Prereg: `brain/research/2026-11-19-life-muse-mscale-prereg.md` section 5. Runner `brain/experiments/life_screen.py`.
s310 ran on pool40 (the prereg named s311/s312; their homes sit on busy nodes and are screened when their nodes free;
s310 is the same banked Stage-0 baseline). Data in `2026-11-19-life-screens-1.5b/`.

## P0-use, s310: KILL rule fires (over-gating), n small
16 fetches in the frozen battery, 4 wrong, 3 right. The weighted winner is not the raw winner in 100% of wrong
fetches (62.5% of all): the hub IS the strength weight. But the raw margin is below 0.02 on 69% of fetches, so the
gate would deny 2 of 3 right fetches (67%, rule: > 40% = kill). Three right fetches is too few to rule on; the verdict
waits for s311/s312 (pooled). Frozen top-1 0.31 vs self-boosting 0.56 ("The bus was late again this morning."): the
battery itself feeds the hub.

## P0-scale, s310: PASS on the whole-transcript read
B scaled by g 0.259 (norm 405 -> 105), no re-replay. Intact: H 2/6 -> 4/6, open intrusions 3/6 -> 0/6, never-told
false 2/3 -> 0/3, collapse 0.10 -> 0.05; hippocampus-off H 2/6 -> 3/6, intrusions 3 -> 0. Read side by side
(`screen_scale_s310.md`): the unscaled brain answers open prompts with told facts or loops ("I keep a sourdough starter
named Gertrude." to a fox story, "Sure, you're Dan, my friend." to an evening suggestion, "Leaves."); the scaled one
tells the fox story, gives focus advice, explains chlorophyll, and answers H in the second person ("You drive a red
Volvo.", "Her name is Marguerite."). Costs, recorded: own name lost ("My name is ChatGPT."), the user's name dropped
("Of course, you're my friend."), and a low-use fact lost (sourdough starter Gertrude -> "Lauren"). Never-told questions
still get invented answers, now generic ones (Carolyn, Anthony) instead of told names.

One seed, one home, no re-replay: a screen, not a result. M-scale goes to dev lives (prereg section 6).
