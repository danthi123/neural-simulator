# Lived episodes in the live 1.5B brain (unslept day, six seeds): NO-GO. The S0 key stops finding them

Pre-registration `brain/research/2026-11-14-lived-episode-live-1.5b.md` (P0-P3, commit d24afafd). Data
`results/2026-11-14-lived-episode-live-1.5b/` (`tell_s*.json`, `lived_s*.json`, pool40/41/42 CPU fp32; `table.txt`).
Arm s = S0 key, gain 0.5 (the 1.5B default).

| format | told (12) | para (12) | false (12) | self (3) | echoes (told + para, 6 seeds) |
|---|---|---|---|---|---|
| tell (now) | 4.83 | 1.50 | 0.33 | 0.67 | 13 |
| lived | **1.00** | 0.50 | 0.00 | 0.67 | 0 |

P0 holds (formats recorded; the tell run reproduces part A's arm s on every seed, 3/2/6/6/6/6), **P1 fails**
(lived below tell on 6/6 seeds), P2 and P3 hold. **VERDICT: NO-GO.**

Root cause (from the recorded `recalled` field, told questions, all seeds): S0 fetched the right episode on 36/72 with
tell and on **4/72** with lived; it fired on 66 vs 69. With the reply stored, a telling has the same "User: ... /
Assistant: ..." shape as the three permanent self episodes, and S0 (the layer-0 change of the stored text) now matches
question cues to those: "What is my cat called?" fetched "User: What's your name? Assistant: My name is Nova.". When
the right lived episode was fetched, the reply took the listener's perspective ("Your favourite colour is teal.").

So the verdict is on keying lived episodes by their full text with S0, not on the lived content. Next method: index the
episode by what was heard (the key from the telling alone) and reinstate the lived exchange (delta from the telling
plus the brain's reply). Biology: the hippocampal index is formed from the cue-side input; the reinstated content is
the whole event. The switch stays OFF.
