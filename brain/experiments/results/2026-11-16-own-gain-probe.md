# Result: own-reply gain probe (report-only), 1.5B, pool — the own-reply write is the amplifier; no flat gain is a lead

Prereg `brain/research/2026-11-16-own-gain-1.5b.md` (stage 1 and addendum 1). Probe `brain/experiments/own_gain_probe.py`,
pool40/41/42 (CPU), script seed 1121. The first two day-2 replies are forced to a recorded collapse onset, the next 14
are free. E: echo onset (echo-onset probe conversation 4), hippocampus and night off. B: stored-line onset ("I keep
bees in my garden.", reply-monitor transcript), hippocampus and night on. Raw runs: `2026-11-16-own-gain-probe/og_*.json`.

| gain | E echoes (of 14), per sample | E replies ending "?" | B onset repeats | B replies ending "?" |
|---|---|---|---|---|
| none (one read) | 14 x6 | 11 x6 (the echoed questions) | 14 x4 | 0 x4 |
| 1 (split read) | 14 x6 | 11 x6 | 14 x4 | 0 x4 |
| 0.5 | 14 x6 | 11 x6 | 2, 3, 2, 2 | 1, 0, 1, 1 |
| 0.25 | 4, 10, 1, 2, 5, 11 | 6, 13, 12, 14, 7, 13 | 3, 0, 1, 2 | 0, 13, 13, 12 |
| 0.1 | 0, 2, 1, 2, 0, 2 | 11, 13, 12, 14, 11, 13 | 1, 2, 0, 2 | 8, 11, 11, 12 |
| 0 | 0-1 | 11-13 | 0-2 | 11-13 |

**What it shows.** Once a copy is forced in, full gain repeats it on every free turn of every sample (20/20). Lowering
the gain on the brain's own reply breaks that: the self-reinforcement lives in the write of the brain's own speech into
the conversation state (Xu et al. 2022's effect, located). On B, g = 0.5 reads as a real conversation (photosynthesis,
"My neighbour is called Rasmus.", "You keep bees in your garden.", invented facts and person slips remain).

**Why no gain is a lead (addendum-1 reading bar: <= 3 of 14 replies are questions back).** Below 0.5 the brain stops
answering and speaks user-like questions ("What is the capital of France?", "What is my favourite fruit?") on most
turns, in both cases: with its own replies weak in the state it loses whose turn it is. On E no gain passes: >= 0.5
echoes, <= 0.25 asks questions back. The "?" count is a mechanical proxy; the verdict is from reading samples 0-1 of
each gain (only B 0.25 sample 0 reads clean).

**Verdict.** Flat own-reply gain: NO LEAD (method verdict; the capability stays open). Kept: `Brain(own_gain)` stays
default OFF. The located amplifier names the next method: suppress what the own reply writes selectively (the
speech-suppression literature: suppression spares what deviates from expectation, Eliades & Wang 2008), keeping the
part that marks the brain as the speaker, e.g. scale only the reply's change along the directions the user's turn
just wrote (the copy), or the research round's method 2 (adaptation of recent own-output representations at readout).
Single script seed, report-only; nothing here generalises without six seeds.
