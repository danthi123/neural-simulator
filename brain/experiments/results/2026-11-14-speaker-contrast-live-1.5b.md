# Speaker contrast in the live 1.5B brain (unslept, six seeds): NO-GO on told (0.00 < 1); echoes 13 -> 4, para +0.83

Pre-registration `brain/research/2026-11-14-speaker-contrast-live-1.5b.md` (commit 46da9f9e). Data
`results/2026-11-14-speaker-contrast-live-1.5b/` (`contrast_s*.json` pool40/41/42 with `Brain(speaker_contrast=3.0)`;
`tell_s*.json` copied from the lived-episode run, same code path; `table.txt`). Arm s = S0 key, gain 0.5.

| | told (12) | para (12) | false (12) | self (3) | echoes | answered given the right episode (told + para) |
|---|---|---|---|---|---|---|
| tell (now) | 4.83 | 1.50 | 0.33 | 0.67 | 13 | 37 / 54 |
| contrast mu 3 | 4.83 | 2.33 | 0.33 | 0.67 | 4 | 42 / 54 |

P0, P2, P3 hold; **P1 fails** (told +0.00; contrast >= tell on 4/6). **VERDICT: NO-GO.** Switch stays OFF.

Same shape as lived_key (`2026-11-14-lived-key-1.5b.md`): the routing is untouched (same 54 right-episode rows), the
contrast turns echoes into answers when the right episode is fetched (37 -> 42 of 54, mostly paraphrases), and the
told total is bound by S0 fetching the right episode on about half the told questions. The oracle's +20 of 108 does
not reach the live brain because the oracle always reinstates the right episode. No cost on false recall or self.

Both perspective fixes now work when routing works; routing is the limit. Next: the same test with the S_all key
(right episode on 11/12 told in the 11-03 routing data), which needs the key stored separately from the contrasted
content (S_all reads its key from the stored delta).
