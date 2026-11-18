# S_all bar refit on slept live reads (1.5B, stage 1): the right telling ranks first on rewordings; fit (0.165, 0.02)

Prereg: `brain/research/2026-11-15-sall-slept-bar-1.5b.md` stage 1. Board #282.
- Run on pool CPU fp32: seeds 42/43/44 on pool40/41/42. Command:
  `sall_live_bar_fit score --size-sleep --model 1.5b --device cpu`.
- Raw: `results/2026-11-18-sall-slept-fit-1.5b/` (score JSONs, `fit.json`).
- DEV bank: 12 told facts per seed, one night at the 1.5B size defaults.

Fire and right selection rates (36 cues per told kind, 72 untold, 36 never-told, 9 self), from `fit.json`:

| bar | orig | para_a | para_b | untold fire | never fire | self | J |
|---|---|---|---|---|---|---|---|
| forced top-1 (ranking only) | 0.86 | 0.83 | 0.69 | - | - | 0.78 | - |
| size bar 0.22 / 0.01 (unslept fit) | 0.36 | 0.33 | 0.19 | 0.11 | 0.06 | 0.67 | 0.255 |
| **slept fit 0.165 / 0.02** | 0.67 | 0.50 | 0.36 | 0.19 | 0.14 | 0.67 | **0.407** |
| S0 default bar 0.12 / 0.02 | 0.78 | 0.72 | 0.47 | 0.53 | 0.44 | 0.67 | 0.25 |

Mean slept scores:
- Top score: orig 0.217, para_a 0.184, para_b 0.162, untold 0.142, never-told 0.135.
- Score of the right telling: 0.208, 0.178, 0.153.

**Reading.**
- After a night, S_all still ranks the right telling first for most rewordings. The unslept bar sits above most
  slept paraphrase scores, so it rarely fires.
- This settles the open branch of `results/2026-11-18-sall-open-conv-1.5b.md`: the bar is the cause.
- Untold and never-told cues select nothing at any bar (sel 0); every fire on them is false. Their scores sit only
  0.02-0.05 below paraphrase scores, so the bar trades rewordings against false fires.

**Caveats.**
- This is the DEV bank: clean single-statement days. An open-conversation day stores more episodes (greetings, chat),
  so there are more lures.
- The bar is frozen as `key_bar="slept"` (switch, default OFF) and judged on open conversation next (prereg
  amendment 1). Nothing flips here.
