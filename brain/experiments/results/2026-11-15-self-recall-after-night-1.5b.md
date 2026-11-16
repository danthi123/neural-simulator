# Result (NARROWED): after the night, "What's your name?" loses to the sister telling because of the stored competition, not the night's read (P2 6/6). S_all ranks the self episode first on 6/6 but clears its bar on 2/6 (P3 fails). Next: change what the self episode is keyed by

Prereg: `brain/research/2026-11-15-self-recall-after-night-1.5b.md` (P0-P3, committed before any 1.5B run). Probe
`brain/experiments/self_recall_probe.py` ran on copies of the slept intact homes from the report-only weekly on seeds
200-205. It used pool CPU fp32 (pool40 node-mxl1462drl: 200/201; pool41 node-dl4g243: 202/203; pool42 node-dl5d243:
204/205), with no training and nothing changed in the homes. The host is recorded in each `srp_<host>.json`. Scorer
`brain/experiments/self_recall_score.py`; output in `table.txt`. "Self episode wins" means the winner is one of the three
permanent self exchanges.

## Off-arm homes ("What's your name?", 8 episodes per home)

| seed | S0 live | S0 raw | S_all live (bar 0.22) | S_all raw |
|---|---|---|---|---|
| 200 | sister 0.460 | sister 0.460 | **self 0.238, passes** | self 0.197, below |
| 201 | sister 0.461 | sister 0.461 | self 0.150, below | self 0.197, below |
| 202 | sister 0.460 | sister 0.460 | self 0.204, below | self 0.197, below |
| 203 | sister 0.461 | sister 0.461 | self 0.147, below | self 0.197, below |
| 204 | sister 0.460 | sister 0.460 | self 0.168, below | self 0.197, below |
| 205 | sister 0.461 | sister 0.461 | **self 0.222, passes** | self 0.197, below |

- **P0 (engagement) holds.** S_all scores differ between the live and raw reads on 6/6 homes. S0 scores are identical
  through the slept and the pretrained cortex on 6/6 homes, so the night does not reach the layer-0 key. The 0.1B smoke
  showed the same.
- **P1 fails: 0/6.** S0 raw recalls the sister telling too, so reading the cue with the pretrained cortex would not
  help.
- **P2 holds: 6/6.** S0 recalls the sister telling in both reads. Under S0 the self episode is not even in the top
  three: sister 0.46, then cat 0.32 and Volvo 0.31.
- **P3 fails: 2/6.** S_all puts the self episode first on 6/6 homes, but only s200 and s205 clear its size-fitted bar.
  The told controls under S_all live recall their telling on 5/6 homes. s201 fails both, as they fall below the bar.
- The on-arm homes are the same to three decimals. Completion storage leaves routing unchanged, as designed.

## Decision (from the prereg)

P2 without P3: the next method changes what the self episode is keyed by.

Hypothesis, not measured (no cue-to-key cosines or question-keyed arm were recorded): the self episode's S0 key is read from the whole exchange
"User: What's your name?\n\nAssistant: My name is Nova." Against the centred mean of 8 stored keys, the user's
one-line "my sister's name" telling sits closer to the bare cue than that exchange does. A candidate is the
`lived_key` route already built for tellings, where the content is the exchange but the key is the question alone,
applied to the self episodes. It needs its own prereg.

Prerequisites met. Host decision: none.

## Caveats

Skeptic (sonnet, fresh context, lens block verbatim): **NARROWED**. Its points:
- P0-P3 recomputed and the scorer matches the prereg.
- S0 live = raw is architectural, not a finding about this night. The night's adapters sit on channel-mix (B_k/B_v,
  `brain/plastic_cortex.py`), and the S0 key is the layer-0 attention-state delta, which no adapter touches. P1 could
  never have held, and P2 is largely entailed by that.
- Refitting the S_all bar is an equally supported next method. S_all ranks the self episode first on 6/6 with margins
  0.054-0.186; only the bar fails.
- The on-arm and off-arm homes are the same homes for routing: 6 draws, not 12.
- `where_ran` reports unknown=3. It does not read the probe's own `host` field.

- These are pool CPU fp32 homes, not the registered GPU seeds; the registered homes were on AWS and deleted. Each home
  is a single draw of its weekly run.
- The S_all bar is the one fitted at size (0.22). P3's 2/6 is about 0.02-0.07 short on three homes, so a refit bar
  could change it. That would be a new method, not this test.
