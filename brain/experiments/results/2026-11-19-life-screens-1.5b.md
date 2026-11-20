# Offline screens P0-use / P0-scale on banked Stage-0 homes (1.5B, morning after night 7): M-use KILLED (over-gating), M-scale to dev lives

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

## Pooled over s310/s311/s312 (s311 on pool41, s312 on pool42)

**P0-use: KILL M-use as over-gating (prereg rule: over 40% of right fetches denied; do not retune MARGIN_USE).**
Right fetches denied: s310 2/3, s311 0/0 (no right fetch), s312 1/1, pooled 3/4 = 75%. Wrong fetches denied:
s310 100% weighted-only winners, s311 6/7 (86%), s312 2/4 (50%). The gate is not selective: it denies almost every
fetch whose raw margin is under 0.02, right or wrong, because at night 7 nearly all raw margins are that small. n=4
right fetches is thin, but the rule is pre-registered and the shape (raw margins collapse with the store) says a
margin gate on raw sims cannot separate them here. Verdict on the METHOD (a raw-margin confirmation gate); the hub
defect stays open. Next method for the hub is not a gate on the same weak raw sims (see HANDOFF).

**P0-scale: open speech recovers on 3/3 seeds; told-fact recall mixed.** Norm 405/376/359 -> 105 (g 0.26/0.28/0.29).
| seed | H int un->sc | H off un->sc | open intrusions int | never-told false int |
|---|---|---|---|---|
| 310 | 2 -> 4 /6 | 2 -> 3 | 3 -> 0 /6 | 2 -> 0 /3 |
| 311 | 2 -> 2 | 2 -> 2 | 1 -> 0 | 2 -> 0 |
| 312 | 4 -> 2 | 2 -> 3 | 5 -> 0 | 2 -> 1 |
Read side by side: on every seed the scaled brain tells the fox story, gives numbered focus advice, and answers the
leaves question in kind (s312: "a process called abscission"; s311: "changing light and temperature"), where the
unscaled one answers with told facts ("Sure, the fox is called Pixel.", "I found a candle in the drawer...") or
nonsense ("The leaves get a bit of wind chimes."). Costs, on s311/s312 too: own name changes (Nova -> Alex), the user's
name is dropped ("Of course, you're my friend."), and the slept "Pixel" answer still fills pet-name questions
("His name is Pixel." to a never-told question, s312). Intact H falls 4 -> 2 on s312 while hippocampus-off rises 2 -> 3:
the cut removes some slept facts along with the runaway. A post-hoc cut after 7 unbounded nights is not the method;
the dev lives test the nightly budget (prereg section 6). M-scale goes on; M-use does not.

Dev-life load after the kill: arms B and S only on seeds 315-317 (US and U dropped). Running 11-19 23:59:
pool40 s315 B, pool41 s315 S, pool42 s316 B; next s316 S, s317 B, s317 S.

## Stage-0 lives s313/s314 (added): night-7 intact H 1/6 and 0/6, wrong fetches 5/8 and 5/9, hub top-1 live 0.50
and 0.67; same degradation as s310-s312 (`2026-11-19-life-stage0-1.5b/life_s31[34].*`).
